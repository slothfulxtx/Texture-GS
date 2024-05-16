
from utils.graphics import BasicPointCloud
from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, cfg, log, work_dir, debug=False):
        raise NotImplementedError

    def in_range(self, iter, iter_range):
        if iter_range is None: return True
        start = 0 if (len(iter_range)!=2) or (iter_range[0] is None) else iter_range[0]
        end = 1e7 if (len(iter_range)!=2) or (iter_range[1] is None) else iter_range[1]
        return start < iter and iter <= end

    @abstractmethod
    def state_dict(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state_dict, optim_cfg):
        raise NotImplementedError
    
    @abstractmethod
    def initialize(self, pcd: BasicPointCloud, spatial_lr_scale : float):
        raise NotImplementedError
    
    @abstractmethod
    def setup_optim(self, optim_cfg):
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(self, cur_iter, total_iter, viewpoint, render, loss_cfg):
        raise NotImplementedError
    
    @abstractmethod
    def optimize_step(self, cur_iter, total_iter, train_cfg, extra_info):
        raise NotImplementedError

    @abstractmethod
    def save_point_cloud(self, path):
        raise NotImplementedError
    
    @abstractmethod
    def visual_step(self, cur_iter, total_iter, viewpoint, render):
        raise NotImplementedError