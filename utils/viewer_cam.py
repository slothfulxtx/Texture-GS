from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
import glm
from .cameras import Camera
from .graphics import getProjectionMatrix
from addict import Dict
import torch

class ViewerCam:

    def __init__(self):
        self.znear = 0.01
        self.zfar = 100.0
        self.h = self.w = 800
        self.fovy = self.fovx = np.pi/2
        
        self.position = np.array([0.0, 0.0, 3.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, -1.0, 0.0])
        self.yaw = -np.pi / 2
        self.pitch = 0
        
        self.last_mouse_position = None
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        
        self.rot_speed = 0.005
        self.trans_speed = 0.01
        self.zoom_speed = 0.08
        self.roll_speed = 0.03
    
    def load_from_camera(self, cam: Camera):
        self.znear, self.zfar = cam.znear, cam.zfar
        self.h, self.w = cam.image_height, cam.image_width
        self.fovx, self.fovy = cam.FoVx, cam.FoVy
        self.position = cam.camera_center.cpu().numpy().astype(np.float64)
        # TODO: add self.up load support!
    
    def get_viewpoint(self):

        view_matrix = np.array(glm.lookAt(self.position, self.target, self.up))
        view_matrix[[1, 2], :] = -view_matrix[[1, 2], :]
        view_matrix = torch.tensor(view_matrix.T).float().cuda()
        proj_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1).cuda()
        full_proj_matrix = (view_matrix.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)
        
        return Dict(
            FoVx = self.fovx,
            FoVy = self.fovy,
            image_height = self.h,
            image_width = self.w,
            world_view_transform = view_matrix,
            full_proj_transform = full_proj_matrix,
            camera_center = torch.tensor(self.position).float().cuda()
        )
    
    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def process_mouse(self, xpos, ypos):
        if self.last_mouse_position is None:
            self.last_mouse_position = (xpos, ypos)

        xoffset = xpos - self.last_mouse_position[0]
        yoffset = self.last_mouse_position[1] - ypos
        self.last_mouse_position = (xpos, ypos)

        if self.is_leftmouse_pressed:
            self.yaw += xoffset * self.rot_speed
            self.pitch += yoffset * self.rot_speed

            self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)

            front = np.array([np.cos(self.yaw) * np.cos(self.pitch), 
                            np.sin(self.pitch), np.sin(self.yaw) * 
                            np.cos(self.pitch)])
            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            self.position[:] = - front * np.linalg.norm(self.position - self.target) + self.target
            
        
        if self.is_rightmouse_pressed:
            
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_speed
            self.target += right * xoffset * self.trans_speed
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_speed
            self.target += cam_up * yoffset * self.trans_speed
            
        
    def process_wheel(self, dx, dy):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_speed
        self.target += front * dy * self.zoom_speed
        
    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_speed / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        
    def flip_ground(self):
        self.up = -self.up
    
    def update_resolution(self, height, width):
        self.h = height
        self.w = width

    def update_fov(self, fovy):
        self.fovy = fovy
        htany = np.tan(fovy / 2)
        htanx = htany / self.h * self.w
        self.fovx = np.arctan(htanx) * 2
        