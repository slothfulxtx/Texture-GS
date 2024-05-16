from .scene import Scene


type2dataset = dict(
    scene=Scene
)

def create_dataset(cfg, *args, **kwargs):
    return type2dataset[cfg.type](cfg, *args, **kwargs)