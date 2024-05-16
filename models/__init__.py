from .gaussian3d import Gaussian3D
from .uv_map_gaussian3d import UVMapGaussian3D
from .texture_gaussian3d import TextureGaussian3D
from .base import BaseModel
type2model = dict(
    Gaussian3D=Gaussian3D,
    UVMapGaussian3D=UVMapGaussian3D,
    TextureGaussian3D=TextureGaussian3D,
)

def create_model(cfg, *args, **kargs):
    return type2model[cfg.type](cfg, *args, **kargs)