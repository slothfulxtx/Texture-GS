from .render import render
from .uv_tex_render import uv_tex_render

type2render_func = dict(
    render=render,
    uv_tex_render=uv_tex_render,
)

def create_render_func(cfg, *args, **kargs):
    return type2render_func[cfg.type]