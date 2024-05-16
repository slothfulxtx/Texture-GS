import torch
from argparse import ArgumentParser
import yaml
from addict import Dict
import numpy as np
import cv2

from losses import *
from models import create_model, BaseModel
from utils.logger import get_logger

def extract(cfg, log):
    gaussians: BaseModel = create_model(cfg.model_cfg, log, cfg.work_dir, cfg.debug)
    (state_dict, start_iteration) = torch.load(cfg.resume_from)
    gaussians.load_state_dict(state_dict, cfg.optim_cfg)    
    texture = gaussians.cube_map()
    texture = (torch.clamp(texture, 0, 1) * 255).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite(cfg.save_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
        

def parse_args():
    parser = ArgumentParser(description="Textured Gaussian Splatting")
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--save_path', type=str, required=True, help='path to save file')
    parser.add_argument('--resume_from', type=str, required=True, help='path to checkpoint file')

    args = parser.parse_args()
    return args

def add_args_to_cfg(args, cfg):
    cfg.work_dir = './tmp'
    cfg.resume_from = args.resume_from    
    cfg.debug = True
    cfg.save_path = args.save_path
    
if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)
        
    log_file_dir = None
    log = get_logger(name='TextureGS', log_file=log_file_dir)

    extract(cfg, log)
    