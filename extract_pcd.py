import torch
from argparse import ArgumentParser
import yaml
from addict import Dict
from pytorch3d.ops import sample_farthest_points
from plyfile import PlyData, PlyElement
import numpy as np

from losses import *
from models import create_model, BaseModel
from utils.logger import get_logger

def extract(cfg, log):
    gaussians: BaseModel = create_model(cfg.model_cfg, log, cfg.work_dir, cfg.debug)
    (state_dict, start_iteration) = torch.load(cfg.resume_from)
    gaussians.load_state_dict(state_dict, cfg.optim_cfg)    
    xyzs = gaussians.get_xyz
    if xyzs.shape[0] > cfg.num_points:
        xyzs, _ = sample_farthest_points(xyzs.unsqueeze(0), K=cfg.num_points)
        xyzs = xyzs.squeeze(0)
    xyz = xyzs.detach().cpu().numpy()
    # print((np.max(xyz, axis=0) + np.min(xyz, axis=0)) / 2, np.max(xyz, axis=0)-np.min(xyz, axis=0))
    
    np.save(cfg.save_path, xyz)
    ply_save_path = cfg.save_path.replace('npy', 'ply')

    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz,), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_save_path)
        

def parse_args():
    parser = ArgumentParser(description="Textured Gaussian Splatting")
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--save_path', type=str, required=True, help='path to save file')
    parser.add_argument('--num_points', type=int, default=16384, help='the number of sampled points')
    parser.add_argument('--resume_from', type=str, required=True, help='path to checkpoint file')

    args = parser.parse_args()
    return args

def add_args_to_cfg(args, cfg):
    cfg.work_dir = './tmp'
    cfg.resume_from = args.resume_from    
    cfg.debug = True
    cfg.save_path = args.save_path
    cfg.num_points = args.num_points

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)
        
    log_file_dir = None
    log = get_logger(name='TextureGS', log_file=log_file_dir)

    extract(cfg, log)
    