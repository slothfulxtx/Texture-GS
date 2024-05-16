import torch
import torch.nn.functional as F

def smooth_loss(rgb, value, mask=None, gamma=0.1):
    # 3, H, W 
    # C, H, W
    _, H, W = rgb.shape
    mask = mask.float()
    bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(0, keepdim=True) / gamma)
    loss = lambda x: torch.sum(torch.abs(x))
    w1 = bilateral_filter(rgb[:,:,:-1] - rgb[:,:,1:])
    w2 = bilateral_filter(rgb[:,:-1,:] - rgb[:,1:,:])
    w3 = bilateral_filter(rgb[:,:-1,:-1] - rgb[:,1:,1:])
    w4 = bilateral_filter(rgb[:,1:,:-1] - rgb[:,:-1,1:])

    if mask is not None:
        w1 *= mask[:,:,:-1] * mask[:,:,1:]
        w2 *= mask[:,:-1,:] * mask[:,1:,:]
        w3 *= mask[:,:-1,:-1] * mask[:,1:,1:]
        w4 *= mask[:,1:,:-1] * mask[:,:-1,1:]

    L1 = loss(w1 * (value[:,:,:-1] - value[:,:,1:])) / (torch.sum(w1) + 1e-6)
    L2 = loss(w2 * (value[:,:-1,:] - value[:,1:,:])) / (torch.sum(w2) + 1e-6)
    L3 = loss(w3 * (value[:,:-1,:-1] - value[:,1:,1:])) / (torch.sum(w3) + 1e-6)
    L4 = loss(w4 * (value[:,1:,:-1] - value[:,:-1,1:])) / (torch.sum(w4) + 1e-6)

    return (L1+L2+L3+L4) / 4


def filter2d(input, filter):
    # input: C, H, W
    # filter: 3, 3
    C, H, W = input.shape
    fH, fW = filter.shape 
    assert fW == 3 and fH == 3
    filter = filter.view(1, 1, fH, fW)
    input = F.pad(input, pad=(1, 1, 1, 1), mode="replicate")
    return F.conv2d(input.unsqueeze(0),filter.repeat(C, 1, 1, 1), stride=1, groups=C).squeeze(0)


def second_order_smooth_loss(value, rgb=None, depth=None, depth_threshold=1e-2, alpha=None):
    # C, H, W
    # 3, H, W
    device = value.device
    if rgb is not None:
        rgb_grad_x = rgb[:,:,1:] - rgb[:,:,:-1]
        # 3, H, W-1
        rgb_grad_y = rgb[:,1:,:] - rgb[:,:-1,:]
        # 3, H-1, W
        w_x = torch.exp(-torch.abs(rgb_grad_x).sum(0, keepdim=True))
        w_y = torch.exp(-torch.abs(rgb_grad_y).sum(0, keepdim=True))
        
    if depth is not None:
        depth_grad_x = depth[:,:,1:] - depth[:,:,:-1]
        depth_grad_y = depth[:,1:,:] - depth[:,:-1,:]
        w_x = (depth_grad_x < depth_threshold).float()
        w_y = (depth_grad_y < depth_threshold).float()
    
    if alpha is not None:
        w_x = alpha[:,:,1:] * alpha[:,:,:-1]
        w_y = alpha[:,1:,:] * alpha[:,:-1,:]        

    value_grad_x = value[:,:,1:] - value[:,:,:-1]
    # C, H, W-1
    value_grad_y = value[:,1:,:] - value[:,:-1,:]
    # C, H-1, W
    value_grad2_x = \
        F.pad((value_grad_x[:,:,1:] - value_grad_x[:, :, :-1]).abs().sum(dim=0, keepdim=True), (0,1), 'constant',0) + \
        F.pad((value_grad_x[:,1:,:] - value_grad_x[:, :-1, :]).abs().sum(dim=0, keepdim=True), (0,0,0,1), 'constant',0)
    # 1, H, W-1
    value_grad2_y = \
        F.pad((value_grad_y[:,:,1:] - value_grad_y[:, :, :-1]).abs().sum(dim=0, keepdim=True), (0,1), 'constant',0) + \
        F.pad((value_grad_y[:,1:,:] - value_grad_y[:, :-1, :]).abs().sum(dim=0, keepdim=True), (0,0,0,1), 'constant',0)
    # 1, H-1, W  
    if (rgb is not None) or (depth is not None) or (alpha is not None):
        value_grad2_x = value_grad2_x * w_x
        value_grad2_y = value_grad2_y * w_y
    
    return value_grad2_x.mean() + value_grad2_y.mean()