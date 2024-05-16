import os
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
import yaml
from addict import Dict
from datetime import datetime
from functools import partial
import numpy as np

from models import create_model, BaseModel
from dataset import create_dataset
from render import create_render_func
from utils.logger import get_logger
from losses import *

def render_images(viewpoints, gaussians: BaseModel, render, background, log, subset):
    
    if len(viewpoints) == 0:
        return []

    image_list = []

    for idx, viewpoint in tqdm(enumerate(viewpoints)):
        
        visual_pkg : Dict = gaussians.visual_step(0, 1, viewpoint, render)
        image = torch.clamp(visual_pkg.pop("image"), 0.0, 1.0)
        H, W = image.shape[1], image.shape[2]
        gt_alpha = viewpoint.alpha_mask.cuda() if viewpoint.alpha_mask is not None else torch.ones((1, H, W)).float().cuda()
        image = image * gt_alpha.repeat(3, 1, 1) + background[:, None, None].expand_as(image) * (1-gt_alpha.repeat(3, 1, 1))
        image = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8).copy()
        
        image_list.append((viewpoint.image_name, image))
        torch.cuda.empty_cache()

    return image_list

def visualize(cfg, log):
    gaussians = create_model(cfg.model_cfg, log, cfg.work_dir, cfg.debug)
    assert hasattr(gaussians, 'change_texture')
    scene = create_dataset(cfg.dataset_cfg, log, cfg.work_dir, cfg.debug)
    render_func = create_render_func(cfg.render_cfg)
    (state_dict, _) = torch.load(cfg.resume_from)
    gaussians.load_state_dict(state_dict, cfg.optim_cfg)
    background = torch.tensor(cfg.dataset_cfg.background, dtype=torch.float32, device="cuda")

    if cfg.load_texture_from:
        os.system('cp %s %s' % (cfg.load_texture_from, cfg.work_dir))

        ori_res = gaussians._texture.shape[1]
        cubemap_image = cv2.imread(cfg.load_texture_from)
        res = cubemap_image.shape[0] // 3
        assert cubemap_image.shape == (res*3, res*4, 3)
        cubemap_image = cv2.resize(cubemap_image, (ori_res*4, ori_res*3), interpolation=cv2.INTER_LINEAR)
        cubemap_image = cv2.cvtColor(cubemap_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cubemap_image = torch.tensor(cubemap_image, dtype=torch.float32, device="cuda")
        gaussians.change_texture(cubemap_image, mode=cfg.tex_mode)

    train_viewpoints = scene.getTrainCameras().copy()
    test_viewpoints = scene.getTestCameras().copy()

    render = partial(
        render_func, 
        cfg=cfg.render_cfg, 
        bg_color=background, 
        debug=cfg.debug
    )

    images = render_images(train_viewpoints, gaussians, render, background, log, 'train')
    if not cfg.debug:
        os.makedirs(os.path.join(cfg.work_dir, "train")) 
        for image_name, image in images:
            cv2.imwrite(
                os.path.join(cfg.work_dir, "train", "{}.png".format(image_name)),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

    images = render_images(test_viewpoints, gaussians, render,background, log, 'test')
    if not cfg.debug:
        os.makedirs(os.path.join(cfg.work_dir, "test")) 
        for image_name, image in images:
            cv2.imwrite(
                os.path.join(cfg.work_dir, "test", "{}.png".format(image_name)),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

def parse_args():
    parser = ArgumentParser(description="Textured Gaussian Splatting")
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--workspace', type=str,
                        default='./output', help='path to workspace')
    parser.add_argument('--run_name', type=str,
                        default=None, help='name of this run')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', help='choose which state to run')
    parser.add_argument('--resume_from', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--load_texture_from', type=str, default=None, help='path to cubemap texture file')
    parser.add_argument('--tex_mode', type=int, default=0)
    args = parser.parse_args()
    return args

def add_args_to_cfg(args, cfg):
    run_name = os.path.splitext(os.path.basename(args.config))[
        0] if args.run_name is None else args.run_name
    cfg.work_dir = os.path.abspath(os.path.join(
        args.workspace, run_name, 'tex_' + datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')))
    cfg.resume_from = args.resume_from    
    cfg.load_texture_from = args.load_texture_from
    cfg.tex_mode = args.tex_mode
    cfg.debug = args.debug
    cfg.detect_anomaly = args.detect_anomaly

    cfg.dataset_cfg.shuffle = False

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)
    
    if not cfg.debug:
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(os.path.join(cfg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg.to_dict(), f)

    log_file_dir = os.path.join(
        cfg.work_dir, 'TextureGS.log') if not cfg.debug else None
    log = get_logger(name='TextureGS', log_file=log_file_dir)

    if not cfg.debug:
        log.info("Work folder: {}".format(cfg.work_dir))


    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)    
    visualize(cfg, log)