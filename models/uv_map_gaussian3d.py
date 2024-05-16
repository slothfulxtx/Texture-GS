import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .base import BaseModel
from utils.graphics import BasicPointCloud
from plyfile import PlyData, PlyElement
from utils.general import inverse_sigmoid
from losses import *
# from pytorch3d.loss import chamfer_distance
import nvdiffrast.torch as dr

from .modules.uv_net import UVNet, InvUVNet

class UVMapGaussian3D(BaseModel):

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize


    def __init__(self, cfg, log, work_dir, debug=False):
        self.cfg = cfg
        self.log = log
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()

        self.pcd = torch.empty(0)

        self.inv_uv_net = InvUVNet(cfg.inv_uv_net_cfg).cuda()
        self.uv_net = UVNet(cfg.uv_net_cfg).cuda()
        self.geo_emb = nn.Embedding(1, cfg.geo_emb_dim).cuda()
        self.optimizer_uv = None
        self.scheduler_uv = None
        
    def initialize(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        (state_dict, _) = torch.load(self.cfg.init_from)
        (
            self._xyz,
            _,
            _,
            self._scaling,
            self._rotation,
            self._opacity,
            _,
            _,
            _,
        ) = state_dict['params']
        
        
        self.log.info("Number of points at initialisation : {}".format(len(self._xyz)))

        self._xyz = self._xyz.cuda().requires_grad_(False)
        self._scaling = self._scaling.cuda().requires_grad_(False)
        self._rotation = self._rotation.cuda().requires_grad_(False)
        self._opacity = self._opacity.cuda().requires_grad_(False)

        if self.cfg.pcd_load_from:
            self.pcd = torch.tensor(np.load(self.cfg.pcd_load_from)).float().requires_grad_(False).cuda()        
        
    def setup_optim(self, optim_cfg):

        l = [
            {'params': self.uv_net.parameters(), 'lr': optim_cfg.uv_net_lr},
            {'params': self.inv_uv_net.parameters(), 'lr': optim_cfg.inv_uv_net_lr},
            {'params': self.geo_emb.parameters(), 'lr': optim_cfg.uv_net_lr},
        ]

        self.optimizer_uv = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_uv = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_uv, start_factor=0.01, total_iters=100
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_uv,
                    milestones=optim_cfg.uv_net_milestones,
                    gamma=optim_cfg.uv_net_gamma,
                ),
            ])
    def state_dict(self):
        return dict(
            optim_state=(
                self.optimizer_uv.state_dict(),
                self.scheduler_uv.state_dict(),
            ),
            net_state=(
                self.uv_net.state_dict(), 
                self.inv_uv_net.state_dict(),
                self.geo_emb.state_dict()
            ),
        )

    def load_state_dict(self, state_dict, optim_cfg):
        self.initialize(None, None)
        (
            uv_net_state_dict,
            inverse_uv_net_state_dict,
            geo_emb_state_dict
        ) = state_dict['net_state']
        self.uv_net.load_state_dict(uv_net_state_dict)
        self.inv_uv_net.load_state_dict(inverse_uv_net_state_dict)
        self.geo_emb.load_state_dict(geo_emb_state_dict)
        self.setup_optim(optim_cfg)
        self.optimizer_uv.load_state_dict(state_dict['optim_state'][0])
        self.scheduler_uv.load_state_dict(state_dict['optim_state'][1])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def train(self):
        self.uv_net.train()
        self.inv_uv_net.train()
        self.geo_emb.train()

    def eval(self):
        self.uv_net.eval()
        self.inv_uv_net.eval()
        self.geo_emb.eval()

    @torch.no_grad()
    def save_point_cloud(self, path):
        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=self._xyz.device)).squeeze()
        sample_uvs = self.inv_uv_net.sample(8192, self._xyz.device)
        sample_inv_xyzs = self.inv_uv_net(sample_uvs, geo_emb)
        xyz = sample_inv_xyzs.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz,), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def depth2world(self, depth, full_proj_transform, zfar, znear):
        H, W = depth.shape
        pix_x = torch.arange(W, device=depth.device)
        pix_y = torch.arange(H, device=depth.device)
        ndc_x = (pix_x * 2 + 1 ) / W - 1.0
        ndc_y = (pix_y * 2 + 1 ) / H - 1.0
        ndc_y, ndc_x = torch.meshgrid(ndc_y, ndc_x)
        xyz = torch.stack([ndc_x*depth, ndc_y*depth, zfar*depth/(zfar-znear)-zfar*znear/(zfar-znear)], dim=-1).reshape(-1, 3)
        xyz = torch.cat([xyz, depth.reshape(-1, 1)], dim=-1) @ torch.linalg.inv(full_proj_transform)
        xyz = xyz[:, :3].reshape(H, W, 3)
        return xyz    

    def compute_loss(self, cur_iter, total_iter, viewpoint, render, loss_cfg):
        
        self.train()

        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self,
            override_color=torch.zeros_like(self._xyz)
        )
        depth, alpha = render_pkg["depth"], render_pkg["alpha"]

        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=depth.device)).squeeze()
        
        world_xyz = self.depth2world(depth.squeeze(0), viewpoint.full_proj_transform, viewpoint.zfar, viewpoint.znear)
        H,W = depth.shape[-2:]
        valid_mask = (alpha.reshape(-1) > 0.5) # H*W
        world_xyz = world_xyz.reshape(-1, 3)[valid_mask].contiguous().detach()
        uv = self.uv_net(world_xyz, geo_emb)

        loss = 0.0
        loss_stats = dict()

        if loss_cfg.lambda_inverse and self.in_range(cur_iter, loss_cfg.inverse_range):
            
            world_xyz_inv = self.inv_uv_net(uv, geo_emb)
            Linv = ((world_xyz - world_xyz_inv) ** 2).sum(-1)
            Linv = Linv.mean()
            loss += loss_cfg.lambda_inverse * Linv
            loss_stats.update(Linv=Linv)

        sample_uvs, sample_inv_xyzs = None, None

        if loss_cfg.lambda_chamfer and self.in_range(cur_iter, loss_cfg.chamfer_range):
            if sample_uvs is None:
                sample_uvs = self.inv_uv_net.sample(device=depth.device)
            if sample_inv_xyzs is None:
                sample_inv_xyzs = self.inv_uv_net(sample_uvs, geo_emb)
        
            Lchamfer, _ = chamfer_distance(sample_inv_xyzs.unsqueeze(0), self.pcd.unsqueeze(0))
            # npts,
            loss += loss_cfg.lambda_chamfer * Lchamfer
            loss_stats.update(Lchamfer=Lchamfer)
        
        if loss_cfg.lambda_patch_chamfer and self.in_range(cur_iter, loss_cfg.patch_chamfer_range):
            if sample_uvs is None:
                sample_uvs = self.inv_uv_net.patch_sample(device=depth.device)
            if sample_inv_xyzs is None:
                sample_inv_xyzs = self.inv_uv_net(sample_uvs, geo_emb)
        
            Lpatch_chamfer, _ = chamfer_distance(sample_inv_xyzs.unsqueeze(0), self.pcd.unsqueeze(0), single_directional=True)
            # npts,
            loss += loss_cfg.lambda_patch_chamfer * Lpatch_chamfer
            loss_stats.update(Lpatch_chamfer=Lpatch_chamfer)
        
        if loss_cfg.lambda_inverse2 and self.in_range(cur_iter, loss_cfg.inverse_range2):
            if sample_uvs is None:
                sample_uvs = self.inv_uv_net.sample(depth.device)
            if sample_inv_xyzs is None:
                sample_inv_xyzs = self.inv_uv_net(sample_uvs, geo_emb)
            sample_inv_uvs = self.uv_net(sample_inv_xyzs, geo_emb)
            Linv = ((sample_inv_uvs - sample_uvs) ** 2).sum(-1)
            Linv = Linv.mean()
            loss += loss_cfg.lambda_inverse2 * Linv
            loss_stats.update(Linv2=Linv)
 
        loss_stats.update(total_loss=loss)

        loss.backward()
        
        extra_info = dict()

        return loss, loss_stats, extra_info

    @torch.no_grad()
    def optimize_step(self, cur_iter, total_iter, train_cfg, extra_info):

        # Optimizer step
        self.optimizer_uv.step()
        self.optimizer_uv.zero_grad(set_to_none = True)
        self.scheduler_uv.step()
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def chessboard_texture(self, uv, resolution=6):
        # uv: npts, 3
        chessboard = torch.zeros(6, resolution*16, resolution*16, 3).float().to(uv.device)
        for i in range(0, resolution):
            for j in range(0, resolution):
                if (i+j) % 2 == 0:
                    chessboard[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = torch.tensor([0.0, 1.0, 1.0], device=uv.device) 
                else:
                    chessboard[:, i*16:(i+1)*16, j*16:(j+1)*16, :] = torch.tensor([1.0, 0.0, 0.0], device=uv.device)
        rgb = dr.texture(chessboard[None, ...], uv[None, None, ...].contiguous(), boundary_mode='cube')
        return rgb.squeeze()


    def visual_step(self, cur_iter, total_iter, viewpoint, render):
        self.eval()
        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self,
            override_color=torch.zeros_like(self._xyz)
        )
        image, depth, norm, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["norm"], render_pkg["alpha"]
        world_xyz = self.depth2world(depth.squeeze(0), viewpoint.full_proj_transform, viewpoint.zfar, viewpoint.znear)
        H,W = depth.shape[-2:]
        # H, W, 3
        valid_mask = (alpha.reshape(-1) > 0.5) # H*W
        index = torch.arange(H * W, device=depth.device)
        index = index[valid_mask]
        world_xyz = world_xyz.reshape(-1, 3)[valid_mask]

        background = torch.tensor([0, 0, 0], device=depth.device).float()
        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=depth.device)).squeeze()
        uv = self.uv_net(world_xyz, geo_emb)
        sample_rgb = self.chessboard_texture(uv)
        sample_rgb = alpha.reshape(-1)[valid_mask][:, None] * sample_rgb
        chess_image:torch.Tensor = background.reshape(1,3).repeat(H*W, 1) * (1-alpha).reshape(-1, 1) # H*W, 3
        chess_image = chess_image.scatter_add(dim=0, index=index[:, None].repeat(1, 3), src=sample_rgb) 
        chess_image = chess_image.reshape(H, W, 3).permute(2, 0, 1)
        torch.cuda.empty_cache()

        ret = dict(
            image=image,
            chess_image=chess_image,
            depth=depth,
            norm=norm,
            alpha=alpha,
        )
        return ret
