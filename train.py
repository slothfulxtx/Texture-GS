import os
import torch
from random import randint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
import yaml
from addict import Dict
from datetime import datetime
from functools import partial

from losses import *
from models import create_model, BaseModel
from dataset import create_dataset
from render import create_render_func
from utils.logger import get_logger
from utils.metrics import psnr, ssim

def visualize(tb_writer, iteration, end_iteration, gaussians: BaseModel, scene, render, show_gt=False):
    
    def normalize_depth_map(depth, mask=None):
        # 1, H, W
        depth = depth.squeeze(0)
        if mask is not None:
            mask = mask.bool().squeeze(0)
        device = depth.device
        min_d = torch.min(depth[mask]) if mask is not None else torch.min(depth)
        max_d = torch.max(depth[mask]) if mask is not None else torch.max(depth)
        depth = (depth-min_d)/(max_d-min_d+1e-8)
        depth = torch.clamp(depth, 0.0, 1.0)
        d_color = cv2.applyColorMap(cv2.convertScaleAbs(depth.detach().cpu().numpy()*255, alpha=1), cv2.COLORMAP_JET)
        d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
        d_color = torch.tensor(d_color, device=device)
        if mask is not None:
            d_color[~mask] = 0
        return (d_color.float() / 255).permute(2, 0, 1)
        # # 3, H, W 
        # black for near, white for far

    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0

            for idx, viewpoint in enumerate(config['cameras']):
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                # gt_normal = torch.clamp(0.5 * (viewpoint.normal.to("cuda")+1.), 0.0, 1.0) if viewpoint.normal is not None else None
                gt_normal = viewpoint.normal.to('cuda') if viewpoint.normal is not None else None
                gt_alpha = torch.clamp(viewpoint.alpha_mask.to("cuda"), 0.0, 1.0) if viewpoint.alpha_mask is not None else None
                
                visual_pkg : Dict = gaussians.visual_step(iteration, end_iteration, viewpoint, render)
                image = torch.clamp(visual_pkg.pop("image"), 0.0, 1.0)
                raw_depth = visual_pkg.pop("depth")
                depth = normalize_depth_map(raw_depth, gt_alpha)
                alpha = torch.clamp(visual_pkg.pop("alpha"), 0.0, 1.0)
                # norm = torch.clamp(0.5 * (visual_pkg.pop("norm") + 1.), 0.0, 1.0)
                norm = visual_pkg.pop('norm')
                # psedo_norm_gt, psedo_norm_mask = norm_from_depth(raw_depth, viewpoint)
                # psedo_norm_gt = torch.clamp(0.5 * (psedo_norm_gt + 1.), 0.0, 1.0)

                l1_test += l1_loss(image, gt_image).mean().item()
                ssim_test += ssim(image, gt_image)
                psnr_test += psnr(image, gt_image).mean().item()
                if tb_writer and (idx < 5):
                    norm = torch.clamp(0.5 * (norm + 1.), 0.0, 1.0)
                    gt_normal = torch.clamp(0.5 * (gt_normal + 1.), 0.0, 1.0) if viewpoint.normal is not None else None
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                    tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)
                    tb_writer.add_images(config['name'] + "_view_{}/norm".format(viewpoint.image_name), norm[None], global_step=iteration)
                    # tb_writer.add_images(config['name'] + "_view_{}/psedo_norm_gt".format(viewpoint.image_name), psedo_norm_gt[None], global_step=iteration)
                    # tb_writer.add_images(config['name'] + "_view_{}/psedo_norm_mask".format(viewpoint.image_name), psedo_norm_mask[None], global_step=iteration)

                    for key, value in visual_pkg.items():
                        tb_writer.add_images(config['name'] + "_view_{}/{}".format(viewpoint.image_name, key), value[None], global_step=iteration)
    
                    if show_gt:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if gt_normal is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/gt_norm".format(viewpoint.image_name), gt_normal[None], global_step=iteration)
                        if gt_alpha is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/gt_alpha".format(viewpoint.image_name), gt_alpha[None], global_step=iteration)
                                
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])     
            ssim_test /= len(config['cameras'])
            log.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test))
            
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                
    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
    
    torch.cuda.empty_cache()

def train(cfg, log, tb_writer):
    gaussians: BaseModel = create_model(cfg.model_cfg, log, cfg.work_dir, cfg.debug)
    scene = create_dataset(cfg.dataset_cfg, log, cfg.work_dir, cfg.debug)
    render_func = create_render_func(cfg.render_cfg)
   
    if cfg.resume_from is None:
        gaussians.initialize(scene.scene_info.point_cloud, scene.cameras_extent)
        gaussians.setup_optim(cfg.optim_cfg)
        start_iteration = 0
    else:
        (state_dict, start_iteration) = torch.load(cfg.resume_from)
        gaussians.load_state_dict(state_dict, cfg.optim_cfg)    


    end_iteration = cfg.train_cfg.num_iterations

    background = torch.tensor(cfg.dataset_cfg.background, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoints = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(start_iteration, end_iteration))

    render = partial(
        render_func, 
        cfg=cfg.render_cfg, 
        bg_color=background, 
        debug=cfg.debug
    )

    for iteration in range(start_iteration+1, end_iteration+1): 

        iter_start.record()

        if len(viewpoints) == 0:
            viewpoints = scene.getTrainCameras().copy()
        if cfg.debug:
            viewpoint = viewpoints.pop(0)
        else:    
            viewpoint = viewpoints.pop(randint(0, len(viewpoints)-1))        

        loss, loss_stats, extra_info = gaussians.compute_loss(iteration, end_iteration, viewpoint, render, cfg.loss_cfg)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "L": f"{ema_loss_for_log:.{6}f}", 
                    "N": f"{len(gaussians.get_xyz)}", 
                })
                progress_bar.update(10)
            if iteration == end_iteration:
                progress_bar.close()

            if tb_writer:
                for loss_name, loss in loss_stats.items():
                    tb_writer.add_scalar('train_loss_patches/%s' % loss_name, loss.item() if type(loss) != float else loss, iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)


            # # Log and save
            # train_report(tb_writer, iteration, loss_stats, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in cfg.train_cfg.visual_iters) and not cfg.debug:
                log.info("\n[ITER {}] Saving Point Clouds {} points".format(iteration, len(gaussians.get_xyz)))
                os.makedirs(os.path.join(cfg.work_dir, 'pcds'), exist_ok=True)
                gaussians.save_point_cloud(os.path.join(cfg.work_dir, 'pcds', str(iteration) + ".ply"))

            if (iteration in cfg.train_cfg.visual_iters) or cfg.debug:
                visualize(tb_writer, iteration, end_iteration, gaussians, scene, render, show_gt=(iteration==min(cfg.train_cfg.visual_iters)))

            if (iteration in cfg.train_cfg.ckpt_iters) and not cfg.debug:
                log.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(os.path.join(cfg.work_dir, 'checkpoints'), exist_ok=True)
                torch.save((gaussians.state_dict(), iteration), os.path.join(cfg.work_dir, 'checkpoints', str(iteration) + ".pth") )

        gaussians.optimize_step(iteration, end_iteration, cfg.train_cfg, extra_info)


def parse_args():
    parser = ArgumentParser(description="Textured Gaussian Splatting")
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--workspace', type=str,
                        default='./output', help='path to workspace')
    parser.add_argument('--run_name', type=str,
                        default=None, help='name of this run')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', help='choose which state to run')
    parser.add_argument('--resume_from', type=str,
                        default=None, help='path to checkpoint file')
    args = parser.parse_args()
    return args

def add_args_to_cfg(args, cfg):
    run_name = os.path.splitext(os.path.basename(args.config))[
        0] if args.run_name is None else args.run_name
    cfg.work_dir = os.path.abspath(os.path.join(
        args.workspace, run_name, datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')))
    cfg.resume_from = args.resume_from    
    cfg.debug = args.debug
    cfg.detect_anomaly = args.detect_anomaly

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

    if cfg.debug:
        tb_writer = None
    else:   
        tb_writer = SummaryWriter(cfg.work_dir)
        log.info("Work folder: {}".format(cfg.work_dir))
    
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)
    
    train(cfg, log, tb_writer)
    