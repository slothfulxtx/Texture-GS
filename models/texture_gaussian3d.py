import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .base import BaseModel
from utils.graphics import BasicPointCloud
from plyfile import PlyData, PlyElement
from utils.general import inverse_sigmoid, get_expon_lr_func
from losses import *
import cv2
from torch.autograd.functional import jacobian

from .modules.uv_net import UVNet, InvUVNet
from .modules.NVDIFFREC import util

def rgb2sh0(rgb):
    return (rgb - 0.5) / 0.28209479177387814

def sh02rgb(sh0):
    rgb = 0.28209479177387814 * sh0 + 0.5
    return torch.clamp(rgb, 0.0, 1.0)

class TextureGaussian3D(BaseModel):

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
        
        self.inv_uv_net = InvUVNet(cfg.inv_uv_net_cfg).cuda()
        self.uv_net = UVNet(cfg.uv_net_cfg).cuda()
        self.geo_emb = nn.Embedding(1, cfg.geo_emb_dim).cuda()
        
        self._uv = None
        self._grad_uv = None

        self.active_sh_degree = 0
        self.max_sh_degree = cfg.tex_cfg.max_sh_degree
        self._texture = nn.Parameter(torch.zeros(6, cfg.tex_cfg.resolution, cfg.tex_cfg.resolution, 3).float().cuda())
        self._shs = None

        self.optimizer = None
        self.scheduler = None
        self.optimizer_uv = None
        self.scheduler_uv = None
        self.optimizer_tex = None
        self.scheduler_tex = None
        self.spatial_lr_scale = 0
        
    def initialize(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

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

        self._xyz = self._xyz.cuda().requires_grad_(True)
        self._scaling = self._scaling.cuda().requires_grad_(True)
        self._rotation = self._rotation.cuda().requires_grad_(True)
        self._opacity = self._opacity.cuda().requires_grad_(True)
        
        
        (state_dict, _) = torch.load(self.cfg.init_uv_map_from)

        (
            uv_net_state_dict,
            inverse_uv_net_state_dict,
            geo_emb_state_dict
        ) = state_dict['net_state']
        self.uv_net.load_state_dict(uv_net_state_dict)
        self.inv_uv_net.load_state_dict(inverse_uv_net_state_dict)
        self.geo_emb.load_state_dict(geo_emb_state_dict)
        
        if self.max_sh_degree > 0:
            self._shs = nn.Parameter(torch.zeros(self._xyz.shape[0], (self.max_sh_degree + 1) ** 2 - 1, 3).float().cuda())
        
    def setup_optim(self, optim_cfg):
        self.optim_cfg = optim_cfg
        l = [
            {'params': [self._xyz], 'lr': optim_cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': optim_cfg.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': optim_cfg.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': optim_cfg.rotation_lr, "name": "rotation"}
        ]

        
        if self.max_sh_degree > 0: 
            l.append({'params': [self._shs], 'lr': optim_cfg.tex_lr / 20.0, "name": "sh"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=optim_cfg.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=optim_cfg.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=optim_cfg.position_lr_delay_mult,
                                                    max_steps=optim_cfg.position_lr_max_steps)


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
    
        l = [
            {'params': self._texture, 'lr': optim_cfg.tex_lr},
        ]

        self.optimizer_tex = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def state_dict(self):
        return dict(
            hyperparams=(
                self.active_sh_degree,
                self.spatial_lr_scale,
            ),
            optim_state=(
                self.optimizer.state_dict(),
                self.optimizer_uv.state_dict(),
                self.optimizer_tex.state_dict(),
                self.scheduler_uv.state_dict(),
            ),
            net_state=(
                self.uv_net.state_dict(), 
                self.inv_uv_net.state_dict(),
                self.geo_emb.state_dict()
            ),
            params=(
                self._xyz,
                self._scaling,
                self._rotation,
                self._opacity,
                self._shs,
                self._texture,
            )
        )

    def load_state_dict(self, state_dict, optim_cfg):
        (self.active_sh_degree, self.spatial_lr_scale) = state_dict['hyperparams']  
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self._shs,
            self._texture,
        ) = state_dict['params']
        (
            uv_net_state_dict,
            inverse_uv_net_state_dict,
            geo_emb_state_dict
        ) = state_dict['net_state']
        self.uv_net.load_state_dict(uv_net_state_dict)
        self.inv_uv_net.load_state_dict(inverse_uv_net_state_dict)
        self.geo_emb.load_state_dict(geo_emb_state_dict)
        self.setup_optim(optim_cfg)
        self.optimizer.load_state_dict(state_dict['optim_state'][0])
        self.optimizer_uv.load_state_dict(state_dict['optim_state'][1])
        self.optimizer_tex.load_state_dict(state_dict['optim_state'][2])
        self.scheduler_uv.load_state_dict(state_dict['optim_state'][3])
    
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
    def get_texture(self):
        return self._texture

    @property
    def get_shs(self):
        return self._shs

    @property
    def get_grad_uvs(self):
        if self._grad_uv is not None:
            return self._grad_uv
        xyz = self._xyz.detach()
        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=xyz.device)).squeeze()
        geo_emb = geo_emb.detach()
        def func(inputs):
            return self.uv_net(inputs, geo_emb).float().contiguous().sum(dim=0)
        grad_uvs = jacobian(func=func, inputs=xyz)
        # 3, npts, 3
        return grad_uvs.permute(1, 0, 2).reshape(-1, 9).contiguous().detach().requires_grad_(False)

    @property
    def get_uvs(self):
        if self._uv is not None:
            return self._uv
        xyz = self._xyz
        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=xyz.device)).squeeze()
        uv = self.uv_net(xyz, geo_emb).float().contiguous()
        return uv
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def train(self):
        self.uv_net.train()
        self.inv_uv_net.train()
        self.geo_emb.train()
        self._uv = None
        self._grad_uv = None

    def eval(self):
        self.uv_net.eval()
        self.inv_uv_net.eval()
        self.geo_emb.eval()
        self._uv = self.get_uvs
        self._grad_uv = self.get_grad_uvs

    @torch.no_grad()
    def save_point_cloud(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz,), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)  

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def reset_min_scale(self):
        scaling_new = self._scaling.detach().clone()
        idx = torch.argmin(scaling_new, dim=1, keepdim=False)
        value = torch.ones_like(idx).float() * -20.0
        idx = [torch.arange(scaling_new.shape[0]).cuda().long(), idx]
        scaling_new.index_put_(idx, value)
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

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
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def compute_loss(self, cur_iter, total_iter, viewpoint, render, loss_cfg):
        self.train()

        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self,
        )
        image, depth, norm, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["norm"], render_pkg["alpha"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        loss = 0.0
        loss_stats = dict()

        gt_image = viewpoint.original_image.cuda()
        H, W = gt_image.shape[1], gt_image.shape[2]
        gt_alpha = viewpoint.alpha_mask.cuda() if viewpoint.alpha_mask is not None else torch.ones((1, H, W)).float().cuda()
        

        if loss_cfg.lambda_dssim and self.in_range(cur_iter, loss_cfg.rgb_range):
            Ll1 = l1_loss(image, gt_image)
            Lssim = 1.0 - ssim_loss(image, gt_image)
            loss += (1.0 - loss_cfg.lambda_dssim) * Ll1 + loss_cfg.lambda_dssim * Lssim
            loss_stats.update(
                Ll1=Ll1,
                Lssim=Lssim,
            )

        if loss_cfg.lambda_alpha and self.in_range(cur_iter, loss_cfg.alpha_range):
            Lalpha = l1_loss(alpha, gt_alpha)
            loss += loss_cfg.lambda_alpha * Lalpha
            loss_stats.update(Lalpha=Lalpha)
                
        if loss_cfg.lambda_depth and self.in_range(cur_iter, loss_cfg.depth_range):
            gt_depth = viewpoint.depth.cuda()
            Ld = l1_loss(depth, gt_depth)
            loss += loss_cfg.lambda_depth * Ld
            loss_stats.update(Ldepth=Ld) 
        
        
        if loss_cfg.lambda_norm and self.in_range(cur_iter, loss_cfg.norm_range):
            gt_norm = viewpoint.normal.cuda()
            Lnorm = norm_loss(norm, gt_norm, gt_alpha)
            loss += loss_cfg.lambda_norm * Lnorm
            loss_stats.update(Lnorm=Lnorm)

        if loss_cfg.lambda_norm_reg and self.in_range(cur_iter, loss_cfg.norm_reg_range):
            Lnorm_reg = norm_reg_loss(norm, depth, viewpoint, gt_alpha)
            loss += loss_cfg.lambda_norm_reg * Lnorm_reg
            loss_stats.update(Lnorm_reg=Lnorm_reg)

        if loss_cfg.lambda_norm_smooth and self.in_range(cur_iter, loss_cfg.norm_smooth_range):
            Lnsm = smooth_loss(gt_image, norm, gt_alpha)
            loss += loss_cfg.lambda_norm_smooth * Lnsm
            loss_stats.update(Lnorm_smooth=Lnsm)

        if loss_cfg.lambda_opacity_reg and self.in_range(cur_iter, loss_cfg.opacity_reg_range):
            Lor = zero_one_loss(self.get_opacity)
            loss += loss_cfg.lambda_opacity_reg * Lor
            loss_stats.update(Lopacity_reg=Lor)  
        
        if loss_cfg.lambda_no_sh and self.in_range(cur_iter, loss_cfg.rgb_no_sh_range):
            activate_sh_degree = self.active_sh_degree
            self.active_sh_degree = 0
            image_no_sh= render(
                viewpoint_camera=viewpoint, 
                gaussians=self
            )["render"]
            self.active_sh_degree = activate_sh_degree
            Ll1 = l1_loss(image_no_sh, gt_image)
            Lssim = 1.0 - ssim_loss(image_no_sh, gt_image)
            loss += ((1.0 - loss_cfg.lambda_dssim) * Ll1 + loss_cfg.lambda_dssim * Lssim) * loss_cfg.lambda_no_sh
            loss_stats.update(
                Ll1_nosh=Ll1,
                Lssim_nosh=Lssim,
            )

        geo_emb = self.geo_emb(torch.zeros(1, dtype=torch.long, device=depth.device)).squeeze()

        if loss_cfg.lambda_inverse and self.in_range(cur_iter, loss_cfg.inverse_range):
            world_xyz = self.depth2world(depth.detach().squeeze(0), viewpoint.full_proj_transform, viewpoint.zfar, viewpoint.znear)
            H,W = depth.shape[-2:]
            # H, W, 3
            valid_mask = (alpha.detach().reshape(-1) > 0.5) # H*W
            index = torch.arange(H * W, device=depth.device)
            index = index[valid_mask]
            world_xyz = world_xyz.reshape(-1, 3)[valid_mask]
            uv = self.uv_net(world_xyz.detach(), geo_emb).float().contiguous()
            xyz_inv = self.inv_uv_net(uv, geo_emb).float().contiguous()
            Linv = ((world_xyz - xyz_inv) ** 2).sum(-1)
            Linv = Linv.mean()
            loss += loss_cfg.lambda_inverse * Linv
            loss_stats.update(Linv=Linv)

        loss_stats.update(total_loss=loss)

        loss.backward()
        
        extra_info = dict(
            viewspace_point_tensor=viewspace_point_tensor, 
            visibility_filter=visibility_filter, 
            radii=radii
        )

        return loss, loss_stats, extra_info

    @torch.no_grad()
    def optimize_step(self, cur_iter, total_iter, train_cfg, extra_info):
        viewspace_point_tensor = extra_info['viewspace_point_tensor']
        visibility_filter = extra_info['visibility_filter'] 
        radii = extra_info['radii']
        
        if self.optim_cfg.gaussian_optim_range and self.in_range(cur_iter, self.optim_cfg.gaussian_optim_range):
            if train_cfg.min_scale_reset_interval and \
                (cur_iter - self.optim_cfg.gaussian_optim_range[0]) % train_cfg.min_scale_reset_interval == 0:
                self.reset_min_scale()
            self.optimizer.step()
            self.update_learning_rate(cur_iter - self.optim_cfg.gaussian_optim_range[0])
            if (cur_iter - self.optim_cfg.gaussian_optim_range[0]) % 2000 == 0:
                self.oneupSHdegree()

        if self.optim_cfg.uv_optim_range and self.in_range(cur_iter, self.optim_cfg.uv_optim_range):
            self.optimizer_uv.step()
            self.scheduler_uv.step()

        if self.optim_cfg.tex_optim_range and self.in_range(cur_iter, self.optim_cfg.tex_optim_range):
            self.optimizer_tex.step()

        self.optimizer.zero_grad(set_to_none = True)
        self.optimizer_uv.zero_grad(set_to_none = True)
        self.optimizer_tex.zero_grad(set_to_none = True)

    def sphere_map(self, resolution=[512, 1024]):
        rgb = sh02rgb(self._texture)
        color = util.cubemap_to_latlong(rgb, resolution)
        return color

    def cube_map(self):
        rgb = sh02rgb(self._texture)
        res = self._texture.shape[1]
        cubemap = torch.zeros((res*3, res*4, 3), dtype=rgb.dtype, device=rgb.device)
        cubemap[0:res, res:2*res, :] = rgb[2]
        cubemap[res:2*res, 0:res, :] = rgb[1]
        cubemap[res:2*res, res:2*res, :] = rgb[4]
        cubemap[res:2*res, 2*res:3*res, :] = rgb[0]
        cubemap[res:2*res, 3*res:4*res, :] = rgb[5]
        cubemap[2*res:3*res, res:2*res, :] = rgb[3]
        return cubemap
        
    def change_texture(self, cubemap_image, mode=0):
        res = cubemap_image.shape[0] // 3
        assert cubemap_image.shape == (3*res, 4*res, 3)
        cubemaps = [
            cubemap_image[res:2*res, 2*res:3*res, :],
            cubemap_image[res:2*res, 0:res, :],
            cubemap_image[0:res, res:2*res, :],
            cubemap_image[2*res:3*res, res:2*res, :],
            cubemap_image[res:2*res, res:2*res, :],
            cubemap_image[res:2*res, 3*res:4*res, :]
        ]
        new_tex = torch.stack(cubemaps, dim=0) # 6, res, res, 3
        ori_tex = sh02rgb(self._texture) # 6, res, res, 3
        assert ori_tex.shape == new_tex.shape
        if mode == -1:
            pass
        elif mode == 0:
            ori_tex = (ori_tex * 3).clamp(min=0, max=1)
            new_tex = new_tex * ori_tex.mean(dim=-1, keepdim=True)
        elif mode == 1:
            new_tex = new_tex * ori_tex
        elif mode == 2:
            new_tex = ori_tex / new_tex
        elif mode == 3:
            mask = (new_tex.sum(-1) > 0.01)
            ori_tex[mask] = (
                2
                * ori_tex[mask].mean(-1)[..., None]
                * new_tex[mask]
            )
            new_tex += ori_tex

        self._texture = rgb2sh0(new_tex)

    def visual_step(self, cur_iter, total_iter, viewpoint, render):
        self.eval()
        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self,
        )
        image, depth, norm, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["norm"], render_pkg["alpha"]
        
        activate_sh_degree = self.active_sh_degree
        self.active_sh_degree = 0
        image_no_sh= render(
            viewpoint_camera=viewpoint, 
            gaussians=self
        )["render"]
        self.active_sh_degree = activate_sh_degree

        ret = dict(
            image=image,
            image_no_sh=image_no_sh,
            depth=depth,
            norm=norm,
            alpha=alpha,
            envmap=self.sphere_map((512, 1024)).permute(2,0,1),
            cubemap=self.cube_map().permute(2,0,1)
        )
        return ret
