import torch

def zero_one_loss(value):
    epsilon = 1e-3
    val = torch.clamp(value, epsilon, 1 - epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss