import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh import RGB2SH
from simple_knn._C import distCUDA2
from .base import BaseModel
from utils.graphics import BasicPointCloud
from utils.general import strip_symmetric, build_scaling_rotation, inverse_sigmoid, get_expon_lr_func, build_rotation
from losses import *

class Gaussian3D(BaseModel):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = F.normalize


    def __init__(self, cfg, log, work_dir, debug=False):
        self.cfg = cfg
        self.log = log
        self.active_sh_degree = 0
        self.max_sh_degree = cfg.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def initialize(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        self.log.info("Number of points at initialisation : {}".format(fused_point_cloud.shape[0]))

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def setup_optim(self, optim_cfg):
        self.percent_dense = optim_cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': optim_cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': optim_cfg.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': optim_cfg.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': optim_cfg.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': optim_cfg.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': optim_cfg.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=optim_cfg.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=optim_cfg.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=optim_cfg.position_lr_delay_mult,
                                                    max_steps=optim_cfg.position_lr_max_steps)

    def state_dict(self):
        return dict(
            hyperparams=(self.active_sh_degree, self.spatial_lr_scale,),
            optim_state=(self.optimizer.state_dict(),),
            params=(
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,    
            )
        )

    def load_state_dict(self, state_dict, optim_cfg):
        (self.active_sh_degree, self.spatial_lr_scale,) = state_dict['hyperparams']
        (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
        ) = state_dict['params']
        self.setup_optim(optim_cfg)
        self.optimizer.load_state_dict(state_dict['optim_state'][0])
        (self.xyz_gradient_accum, self.denom) = (xyz_gradient_accum, denom)


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
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_point_cloud(self, path):
        xyz = self._xyz.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz,), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def opacity_prune(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def reset_min_scale(self):
        scaling_new = self._scaling.detach().clone()
        idx = torch.argmin(scaling_new, dim=1, keepdim=False)
        value = torch.ones_like(idx).float() * -20.0
        idx = [torch.arange(scaling_new.shape[0]).cuda().long(), idx]
        scaling_new.index_put_(idx, value)
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def compute_loss(self, cur_iter, total_iter, viewpoint, render, loss_cfg):
        self.update_learning_rate(cur_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if cur_iter % 1000 == 0:
            self.oneupSHdegree()

        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self
        )
        image, depth, norm, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["norm"], render_pkg["alpha"]
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint.original_image.cuda()
        H, W = gt_image.shape[1], gt_image.shape[2]
        gt_alpha = viewpoint.alpha_mask.cuda() if viewpoint.alpha_mask is not None else torch.ones((1, H, W)).float().cuda()
        
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim_loss(image, gt_image)
        loss = (1.0 - loss_cfg.lambda_dssim) * Ll1 + loss_cfg.lambda_dssim * Lssim
        
        loss_stats = dict(
            Ll1=Ll1,
            Lssim=Lssim,
        )

        if loss_cfg.lambda_alpha and self.in_range(cur_iter, loss_cfg.alpha_range):
            Lalpha = l1_loss(alpha, gt_alpha)
            loss += loss_cfg.lambda_alpha * Lalpha
            loss_stats.update(Lalpha=Lalpha)        

        if loss_cfg.lambda_opacity_reg and self.in_range(cur_iter, loss_cfg.opacity_reg_range):
            Lor = zero_one_loss(self.get_opacity)
            loss += loss_cfg.lambda_opacity_reg * Lor
            loss_stats.update(Lopacity_reg=Lor)  

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

        if loss_cfg.lambda_norm_smooth and self.in_range(cur_iter, loss_cfg.norm_smooth_range):
            Lnsm = smooth_loss(gt_image, norm, gt_alpha)
            loss += loss_cfg.lambda_norm_smooth * Lnsm
            loss_stats.update(Lnorm_smooth=Lnsm)

        if loss_cfg.lambda_norm_reg and self.in_range(cur_iter, loss_cfg.norm_reg_range):
            Lnorm_reg = norm_reg_loss(norm, depth, viewpoint, gt_alpha)
            loss += loss_cfg.lambda_norm_reg * Lnorm_reg
            loss_stats.update(Lnorm_reg=Lnorm_reg) 

        loss_stats.update(total_loss=loss)

        loss.backward()
        
        extra_info = dict(
            viewspace_point_tensor=viewspace_point_tensor, 
            visibility_filter=visibility_filter, 
            radii=radii
        )

        return loss, loss_stats, extra_info

    def optimize_step(self, cur_iter, total_iter, train_cfg, extra_info):
        viewspace_point_tensor = extra_info['viewspace_point_tensor']
        visibility_filter = extra_info['visibility_filter'] 
        radii = extra_info['radii']
        # Densification
        if cur_iter <= train_cfg.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if train_cfg.opacity_prune_interval and cur_iter % train_cfg.opacity_prune_interval == 0:
                self.opacity_prune(train_cfg.opacity_prune_theshold)
            elif train_cfg.opacity_prune_iters and cur_iter in train_cfg.opacity_prune_iters:
                self.opacity_prune(train_cfg.opacity_prune_theshold)

            if cur_iter > train_cfg.densify_from_iter and cur_iter % train_cfg.densification_interval == 0:
                size_threshold = 20 if cur_iter > train_cfg.opacity_reset_interval else None
                self.densify_and_prune(train_cfg.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)
            
            if cur_iter % train_cfg.opacity_reset_interval == 0:
                self.reset_opacity()

            if train_cfg.min_scale_reset_interval and \
                    cur_iter > train_cfg.min_scale_reset_from_iter and \
                    cur_iter % train_cfg.min_scale_reset_interval == 0:
                self.reset_min_scale()
        else:
            if train_cfg.opacity_prune_interval and cur_iter % train_cfg.opacity_prune_interval == 0:
                self.opacity_prune(train_cfg.opacity_prune_theshold)
            elif train_cfg.opacity_prune_iters and cur_iter in train_cfg.opacity_prune_iters:
                self.opacity_prune(train_cfg.opacity_prune_theshold)

            if train_cfg.min_scale_reset_interval and \
                    cur_iter % train_cfg.min_scale_reset_interval == 0:
                self.reset_min_scale()

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
    
    def visual_step(self, cur_iter, total_iter, viewpoint, render):
        render_pkg = render(
            viewpoint_camera=viewpoint, 
            gaussians=self
        )
        image, depth, norm, alpha = render_pkg["render"], render_pkg["depth"], render_pkg["norm"], render_pkg["alpha"]
        return dict(
            image=image,
            depth=depth,
            norm=norm,
            alpha=alpha
        )