
import torch
import math
from diff_gauss_uv_tex import GaussianRasterizationSettings, GaussianRasterizer


def uv_tex_render(viewpoint_camera, gaussians, cfg, bg_color, scaling_modifier = 1.0, extra_attrs=None, debug=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree if hasattr(gaussians, 'active_sh_degree') else 0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity

    
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    shs = gaussians.get_shs
    texture = gaussians.get_texture
    uvs = gaussians.get_uvs
    grad_uvs = gaussians.get_grad_uvs

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs=shs,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        uvs=uvs,
        gradient_uvs = grad_uvs,
        texture=texture,
        extra_attrs=extra_attrs)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth,
            "norm": rendered_norm,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "extra": extra,
            "radii": radii}