import math
import torch
import torch.nn.functional as F

def filter2d(input, filter):
    # input: C, H, W
    # filter: 3, 3
    C, H, W = input.shape
    fH, fW = filter.shape 
    assert fW == 3 and fH == 3
    filter = filter.view(1, 1, fH, fW)
    input = F.pad(input, pad=(1, 1, 1, 1), mode="replicate")
    return F.conv2d(input.unsqueeze(0),filter.repeat(C, 1, 1, 1), stride=1, groups=C).squeeze(0)


def norm_from_depth(depth, viewpoint, threshold=1e-2):
    _, H, W = depth.shape
    device = depth.device
    tanfovx = math.tan(viewpoint.FoVx * 0.5)
    tanfovy = math.tan(viewpoint.FoVy * 0.5)
    pix_x = torch.arange(0, W, device=device).reshape(1, 1, W).repeat(1, H, 1)
    pix_y = torch.arange(0, H, device=device).reshape(1, H, 1).repeat(1, 1, W)
    # 1, H, W
    
    def pix2ndc(pix, S):
        return (2.0 * pix + 1.0) / S - 1.0

    ndc_x = pix2ndc(pix_x, W)
    ndc_y = pix2ndc(pix_y, H)
    coord_c = torch.cat([ndc_x * tanfovx * depth, ndc_y * tanfovy * depth, depth, torch.ones_like(depth)], dim=0)
    # 4, H, W
    view_matrix = viewpoint.world_view_transform
    # 4, 4
    coord_w = (torch.linalg.inv(view_matrix.transpose(0, 1)) @ coord_c.reshape(4, H*W)).reshape(4, H, W)
    xyz = coord_w[:3, :, :] # 3, H, W
    
    grad_l = filter2d(xyz, torch.tensor([
        [0, 0, 0],
        [-1, 1, 0],
        [0, 0, 0]], dtype=torch.float, device=device, requires_grad=False))
    grad_r = filter2d(xyz, torch.tensor([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]], dtype=torch.float, device=device, requires_grad=False))
    grad_u = filter2d(xyz, torch.tensor([
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 0]], dtype=torch.float, device=device, requires_grad=False))
    grad_d = filter2d(xyz, torch.tensor([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]], dtype=torch.float, device=device, requires_grad=False))

    grad_x = (grad_r + grad_l) / 2
    grad_y = (grad_d + grad_u) / 2

    mask = (torch.norm(grad_l, p=2, dim=0, keepdim=True) < threshold) \
        & (torch.norm(grad_r, p=2, dim=0, keepdim=True) < threshold) \
        & (torch.norm(grad_u, p=2, dim=0, keepdim=True) < threshold) \
        & (torch.norm(grad_d, p=2, dim=0, keepdim=True) < threshold) 

    norm = torch.cross(grad_y, grad_x, dim=0)

    return F.normalize(norm, p=2, dim=0, eps=1e-6), mask.float()

def norm_loss(pred, gt, mask=None):
    # 3, H, W
    if mask is None:
        return torch.mean(1.0 - torch.sum(pred * gt, dim=0))
    else:
        return torch.sum((1.0 - torch.sum(pred * gt, dim=0, keepdim=True))*mask) / (mask.sum()+1e-6)
    

def norm_reg_loss(norm, depth, viewpoint, gt_alpha):

    norm2, mask = norm_from_depth(depth.detach(), viewpoint)
    mask = gt_alpha * mask
    return norm_loss(norm, norm2, mask)