import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import imageio
from addict import Dict
from argparse import ArgumentParser
import yaml
import torch
import cv2

from utils.viewer_cam import ViewerCam
from utils.viewer_renderer import CUDARenderer
from models import create_model, BaseModel
from utils.logger import get_logger
from dataset import create_dataset
from render import create_render_func
from functools import partial

g_camera = None
g_renderer = None
g_scale_modifier = 1.
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False

def impl_glfw_init():
    window_name = "Texture-GS Viewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_resolution(width, height)


def main(cfg, log):
    gaussians: BaseModel = create_model(cfg.model_cfg, log, '', False)
    if cfg.load_texture_from is not None:
        assert hasattr(gaussians, 'change_texture')
    scene = create_dataset(cfg.dataset_cfg, log, '', True)
    render_func = create_render_func(cfg.render_cfg)
    (state_dict, _) = torch.load(cfg.resume_from)
    gaussians.load_state_dict(state_dict, cfg.optim_cfg)
    background = torch.tensor(cfg.dataset_cfg.background, dtype=torch.float32, device="cuda")

    if cfg.load_texture_from is not None:
        ori_res = gaussians._texture.shape[1]
        cubemap_image = cv2.imread(cfg.load_texture_from)
        res = cubemap_image.shape[0] // 3
        assert cubemap_image.shape == (res*3, res*4, 3)
        cubemap_image = cv2.resize(cubemap_image, (ori_res*4, ori_res*3), interpolation=cv2.INTER_LINEAR)
        cubemap_image = cv2.cvtColor(cubemap_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        cubemap_image = torch.tensor(cubemap_image, dtype=torch.float32, device="cuda")
        gaussians.change_texture(cubemap_image)

    train_viewpoints = scene.getTrainCameras().copy()
    render = partial(
        render_func, 
        cfg=cfg.render_cfg, 
        bg_color=background, 
        debug=False
    )

    global g_camera, g_renderer, g_show_camera_win, g_show_control_win, g_show_help_win, g_scale_modifier
    
    g_camera = ViewerCam()
    
    g_camera.load_from_camera(train_viewpoints[0])

    imgui.create_context()
    imgui.get_io().font_global_scale = 2.0    
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    
    g_renderer = CUDARenderer(gaussians, g_camera, render, cfg)
    g_renderer.set_scale_modifier(g_scale_modifier)

    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        g_renderer.draw()

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")
                imgui.text(f"# of Gaus = {len(gaussians.get_xyz)}")
                # camera fov
                changed, new_fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fovy = %.3f"
                )
                g_camera.update_fov(new_fovy)
                
                # scale modifier
                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 1, "scale = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.
                    changed = True
                    
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)
                
                if imgui.button(label="rgb"):
                    g_renderer.set_render_mode('rgb')
                imgui.same_line()
                if imgui.button(label="norm"):
                    g_renderer.set_render_mode('norm')
                imgui.same_line()
                if imgui.button(label="depth"):
                    g_renderer.set_render_mode('depth')
                imgui.same_line()
                if imgui.button(label="alpha"):
                    g_renderer.set_render_mode('alpha')

                if imgui.button(label='save image'):
                    width, height = glfw.get_framebuffer_size(window)
                    nrChannels = 3
                    stride = nrChannels * width
                    stride += (4 - stride % 4) if stride % 4 else 0
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.rot_speed = imgui.slider_float(
                    "", g_camera.rot_speed, 0.002, 0.1, "rot_speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset"):
                g_camera.rot_speed = 0.005

            changed, g_camera.trans_speed = imgui.slider_float(
                    "", g_camera.trans_speed, 0.001, 0.03, "mov_speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset"):
                g_camera.trans_speed = 0.01

            changed, g_camera.zoom_speed = imgui.slider_float(
                    "", g_camera.zoom_speed, 0.001, 0.2, "zoom_speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset"):
                g_camera.zoom_speed = 0.08

            changed, g_camera.roll_speed = imgui.slider_float(
                    "", g_camera.roll_speed, 0.003, 0.1, "roll_speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset"):
                g_camera.roll_speed = 0.03

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.end()
        
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def parse_args():
    parser = ArgumentParser(description="Texture-GS Viewer")
    parser.add_argument('config', help='path to config file')
    parser.add_argument('--resume_from', type=str,
                        default=None, help='path to checkpoint file')
    parser.add_argument('--load_texture_from', type=str,
                        default=None, help='path to texture file')
    args = parser.parse_args()
    return args


def add_args_to_cfg(args, cfg):
    cfg.resume_from = args.resume_from
    cfg.load_texture_from = args.load_texture_from

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = Dict(cfg)
    add_args_to_cfg(args, cfg)
    log = get_logger(name='TextureGS', log_file=None)
    main(cfg, log)
