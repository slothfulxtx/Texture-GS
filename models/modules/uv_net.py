import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from addict import Dict
from .utils import build_mlp

class UVNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre_mlp = build_mlp(cfg.pre_mlp_cfg, 3, cfg.emb_dim)
        self.mlp = build_mlp(cfg.mlp_cfg, cfg.emb_dim, 3)
        self.aabb_min = torch.tensor(cfg.aabb_min) if cfg.aabb_min else None
        self.aabb_max = torch.tensor(cfg.aabb_max) if cfg.aabb_max else None
        self.xyz_offset = torch.tensor(cfg.xyz_offset) if cfg.xyz_offset else None
        self.xyz_scale = torch.tensor(cfg.xyz_scale) if cfg.xyz_scale else None
        self.cfg = cfg
    
    def forward(self, xyz, emb):
        # xyz: npts, 3
        # emb: emb_dim,
        if self.xyz_offset is not None and self.xyz_scale is not None:
            xyz_offset = self.xyz_offset.to(xyz.device)
            xyz_scale = self.xyz_scale.to(xyz.device)
            xyz = (xyz - xyz_offset) / xyz_scale

        if self.cfg.pre_mlp_cfg.hash_grid_cfg:
            aabb_min = self.aabb_min.to(xyz.device)
            aabb_max = self.aabb_max.to(xyz.device)
            xyz = (xyz - aabb_min) / (aabb_max - aabb_min)
        x = self.pre_mlp(xyz)
        x = x.float()
        x = F.relu(x+emb[None, :])
        x = self.mlp(x)
        x = x.float()
        return F.normalize(x, dim=-1)
    
class InvUVNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pre_mlp = build_mlp(cfg.pre_mlp_cfg, 3, cfg.emb_dim)
        self.mlp = build_mlp(cfg.mlp_cfg, cfg.emb_dim, 3)
        self.n_sample_points = cfg.n_sample_points
        self.patch_scale = cfg.patch_scale
        self.cfg = cfg
        self.xyz_offset = torch.tensor(cfg.xyz_offset) if cfg.xyz_offset else None
        self.xyz_scale = torch.tensor(cfg.xyz_scale) if cfg.xyz_scale else None

    def sample(self, n_sample_points=None, device='cuda'):
        if n_sample_points is None:
            n_sample_points = self.n_sample_points
        with torch.no_grad():
            points = torch.randn(n_sample_points, 3).to(device).float()
            points = F.normalize(points, dim=-1)
        return points

    def patch_sample(self, n_sample_points=None, device='cuda'):
        if n_sample_points is None:
            n_sample_points = self.n_sample_points
        with torch.no_grad():
            direction = torch.randn(3).to(device).float()
            direction = F.normalize(direction, dim=0)
            points = torch.randn(n_sample_points * self.patch_scale, 3).to(device).float()
            points = F.normalize(points, dim=-1)
            similarity = torch.sum(points * direction, dim=-1)
            _, idx = torch.topk(similarity, k=n_sample_points)
            points = points[idx].contiguous()
        return points

    def forward(self, xyz, emb):
        # xyz: npts, 3
        # emb: emb_dim,
        if self.cfg.pre_mlp_cfg.hash_grid_cfg:
            xyz = xyz / 2 + 0.5
        x = self.pre_mlp(xyz)
        x = x.float()
        x = F.relu(x+emb[None, :])
        x = self.mlp(x)
        x = x.float()
        
        if self.xyz_offset is not None and self.xyz_scale is not None:
            xyz_offset = self.xyz_offset.to(xyz.device)
            xyz_scale = self.xyz_scale.to(xyz.device)
            x =  x * xyz_scale + xyz_offset
        return x
    