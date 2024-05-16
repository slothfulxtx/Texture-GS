'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''
from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER
import torch
from cuda import cudart as cu
from .viewer_cam import ViewerCam
from functools import partial
import cv2

class CUDARenderer:
    def __init__(self, gaussians, camera: ViewerCam, render_func, cfg):
        super().__init__()

        self.scale_modifier = 1.
        self.render_mode = 'rgb'
        self.render_func = render_func
        self.gaussians = gaussians
        self.camera = camera
        w, h = camera.w, camera.h
        gl.glViewport(0, 0, w, h)


        VERTEX_SHADER_SOURCE = """
        #version 450

        smooth out vec4 fragColor;
        smooth out vec2 texcoords;

        vec4 positions[3] = vec4[3](
            vec4(-1.0, 1.0, 0.0, 1.0),
            vec4(3.0, 1.0, 0.0, 1.0),
            vec4(-1.0, -3.0, 0.0, 1.0)
        );

        vec2 texpos[3] = vec2[3](
            vec2(0, 0),
            vec2(2, 0),
            vec2(0, 2)
        );

        void main() {
            gl_Position = positions[gl_VertexID];
            texcoords = texpos[gl_VertexID];
        }
        """

        FRAGMENT_SHADER_SOURCE = """
        #version 330

        smooth in vec2 texcoords;

        out vec4 outputColour;

        uniform sampler2D texSampler;

        void main()
        {
            outputColour = texture(texSampler, texcoords);
        }
        """

        def compile_shaders(vertex_shader, fragment_shader):
            active_shader = shaders.compileProgram(
                shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
            )
            return active_shader


        self.program = compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )
        
        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    
    def set_scale_modifier(self, modifier):
        self.scale_modifier = float(modifier)
    
    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def set_gl_texture(self, h, w):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")
    
    def set_render_resolution(self, w, h):
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)

    def draw(self):
        viewpoint = self.camera.get_viewpoint()
        render_func = partial(self.render_func, scaling_modifier = self.scale_modifier)
        with torch.no_grad():
            visual_pkg = self.gaussians.visual_step(0, 1, viewpoint, render_func)
            # dict(
            #     image=image,
            #     depth=depth,
            #     norm=norm,
            #     alpha=alpha
            # )
            if self.render_mode == 'rgb':
                img = visual_pkg['image']
                img = torch.clamp(img, 0.0, 1.0)
            elif self.render_mode == 'norm':
                norm = visual_pkg['norm']
                img = torch.clamp(0.5 * (norm + 1.), 0.0, 1.0)
            elif self.render_mode == 'alpha':
                alpha = visual_pkg['alpha'].repeat(3, 1, 1)
                img = torch.clamp(alpha, 0.0, 1.0)
            elif self.render_mode == 'depth':
                depth = visual_pkg['depth']
                alpha = visual_pkg['alpha']
                def normalize_depth_map(depth, mask):
                    # 1, H, W
                    depth = depth.squeeze(0)
                    if mask is not None:
                        mask = mask.bool().squeeze(0)
                    device = depth.device
                    min_d = torch.min(depth[mask])
                    max_d = torch.max(depth[mask])
                    depth = (depth-min_d)/(max_d-min_d+1e-8)
                    depth = torch.clamp(depth, 0.0, 1.0)
                    d_color = cv2.applyColorMap(cv2.convertScaleAbs(depth.detach().cpu().numpy()*255, alpha=1), cv2.COLORMAP_JET)
                    d_color = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
                    d_color = torch.tensor(d_color, device=device)
                    if mask is not None:
                        d_color[~mask] = 0
                    return (d_color.float() / 255).permute(2, 0, 1)
                depth = normalize_depth_map(depth, alpha)
                img = torch.clamp(depth, 0.0, 1.0)
            else:
                raise NotImplementedError
            
        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
