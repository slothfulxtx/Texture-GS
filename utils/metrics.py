#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from skimage.metrics import structural_similarity
from lpips import LPIPS
import numpy as np


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def mae(norm1, norm2, alpha=None):
    assert norm1.shape[0] == 3
    assert norm2.shape[0] == 3
    
    cos_sim = torch.cosine_similarity(norm1.view(3, -1), norm2.view(3, -1), dim=0, eps=1e-6)
    cos_sim = torch.clamp(cos_sim, (-1.0 + 1e-10), (1.0 - 1e-10))
    loss_rad = torch.acos(cos_sim)
    loss_deg = loss_rad * (180.0 / np.pi)
    if alpha is not None:
        loss_deg = loss_deg.reshape_as(alpha)
        return (loss_deg*alpha.float()).sum() / (alpha.float().sum())
    else:
        return loss_deg.mean()
    

def ssim(img1, img2):
    # 3, h, w
    assert img1.shape == img2.shape
    assert img1.shape[0] == 3 and len(img1.shape) == 3
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    return structural_similarity(img1, img2, channel_axis=0, data_range=1.0)


lpips_vgg = None

def lpips(img1, img2):
    global lpips_vgg
    if lpips_vgg is None:
        lpips_vgg = LPIPS(net="vgg").to(img1.device)
    else:
        lpips_vgg = lpips_vgg.to(img1.device)
    score = lpips_vgg(img1.unsqueeze(0), img2.unsqueeze(0), normalize=True)
    return score.item()

def avg_error(psnr, ssim, lpips):
    """The 'average' error used in the paper."""
    def psnr_to_mse(psnr):
        """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
        return np.exp(-0.1 * np.log(10.) * psnr)
    mse = psnr_to_mse(psnr)
    dssim = np.sqrt(1 - ssim)
    return np.exp(np.mean(np.log(np.array([mse, dssim, lpips])))).item()