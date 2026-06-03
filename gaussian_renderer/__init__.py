#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError as exc:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _RASTERIZER_IMPORT_ERROR = exc
else:
    _RASTERIZER_IMPORT_ERROR = None
from scene.gaussian_model import GaussianModel
from utils.device_utils import get_device
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine"):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    if GaussianRasterizer is None or GaussianRasterizationSettings is None:
        raise ImportError("diff_gaussian_rasterization is required for rendering") from _RASTERIZER_IMPORT_ERROR

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    device = get_device()
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
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
        viewmatrix=viewpoint_camera.world_view_transform.to(device),
        projmatrix=viewpoint_camera.full_proj_transform.to(device),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.to(device),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        scales = pc._scaling
        rotations = pc._rotation
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    if scales is None:
        scales = pc._scaling
    if rotations is None:
        rotations = pc._rotation

    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        if deformation_point.any():
            try:
                means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(
                    means3D[deformation_point],
                    scales[deformation_point],
                    rotations[deformation_point],
                    opacity[deformation_point],
                    time[deformation_point],
                    camera=viewpoint_camera,
                )
            except TypeError:
                means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(
                    means3D[deformation_point],
                    scales[deformation_point],
                    rotations[deformation_point],
                    opacity[deformation_point],
                    time[deformation_point],
                )
            deformation_aux = pc._deformation.get_aux_outputs()
        else:
            means3D_deform = means3D.new_zeros((0, means3D.shape[-1]))
            scales_deform = scales.new_zeros((0, scales.shape[-1]))
            rotations_deform = rotations.new_zeros((0, rotations.shape[-1]))
            opacity_deform = opacity.new_zeros((0, opacity.shape[-1]))
            deformation_aux = {}
    # print(time.max())
    with torch.no_grad():
        if deformation_point.any():
            pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.covariance_activation(scales_final, scaling_modifier, rotations_final)
        scales_final = None
        rotations_final = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.to(device).repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    appearance_rgb_delta = deformation_aux.get("appearance_rgb_delta") if stage != "coarse" else None
    if appearance_rgb_delta is not None and deformation_point.any():
        if colors_precomp is None:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.to(device).repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None
        colors_precomp = colors_precomp.clone()
        colors_precomp[deformation_point] = torch.clamp(colors_precomp[deformation_point] + appearance_rgb_delta, 0.0, 1.0)

    geo_expert_means3d = deformation_aux.get("geo_expert_means3d") if stage != "coarse" else None
    use_pixel_routing = geo_expert_means3d is not None and deformation_point.any()

    if use_pixel_routing:
        num_experts = geo_expert_means3d.shape[1]
        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        geo_expert_scales = deformation_aux["geo_expert_scales"]
        geo_expert_rotations = deformation_aux["geo_expert_rotations"]
        geo_expert_opacity_logits = deformation_aux["geo_expert_opacity_logits"]
        gaussian_pi_geo_prior = deformation_aux["gaussian_pi_geo_prior"]
        vis_expert_rgb_delta = deformation_aux.get("vis_expert_rgb_delta")
        vis_expert_visibility_alpha = deformation_aux.get("vis_expert_visibility_alpha")
        lifecycle_expert_alpha = deformation_aux.get("lifecycle_expert_alpha")
        active_expert_prior = gaussian_pi_geo_prior[deformation_point].amax(dim=0) > 1e-8
        weight_raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros_like(bg_color),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(device),
            projmatrix=viewpoint_camera.full_proj_transform.to(device),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.to(device),
            prefiltered=False,
            debug=pipe.debug
        )
        weight_rasterizer = GaussianRasterizer(raster_settings=weight_raster_settings)

        if colors_precomp is None and shs is None:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.to(device).repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
            shs = None

        expert_renders = []
        expert_depths = []
        expert_radii_list = []
        pi_geo_weight_logits = []

        for expert_idx in range(num_experts):
            expert_means3D = torch.zeros_like(means3D)
            expert_scales = torch.zeros_like(scales)
            expert_rotations = torch.zeros_like(rotations)
            expert_opacity_logits = torch.zeros_like(opacity)

            expert_means3D[deformation_point] = geo_expert_means3d[deformation_point, expert_idx]
            expert_scales[deformation_point] = geo_expert_scales[deformation_point, expert_idx]
            expert_rotations[deformation_point] = geo_expert_rotations[deformation_point, expert_idx]
            expert_opacity_logits[deformation_point] = geo_expert_opacity_logits[deformation_point, expert_idx]

            expert_means3D[~deformation_point] = means3D[~deformation_point]
            expert_scales[~deformation_point] = scales[~deformation_point]
            expert_rotations[~deformation_point] = rotations[~deformation_point]
            expert_opacity_logits[~deformation_point] = opacity[~deformation_point]

            expert_opacity_final = pc.opacity_activation(expert_opacity_logits)
            if vis_expert_visibility_alpha is not None:
                vis_index = min(expert_idx, vis_expert_visibility_alpha.shape[1] - 1)
                expert_opacity_final = expert_opacity_final * vis_expert_visibility_alpha[:, vis_index]
            if lifecycle_expert_alpha is not None:
                life_index = min(expert_idx, lifecycle_expert_alpha.shape[1] - 1)
                expert_opacity_final = expert_opacity_final * lifecycle_expert_alpha[:, life_index]

            expert_scales_final = pc.scaling_activation(expert_scales)
            expert_rotations_final = pc.rotation_activation(expert_rotations)

            expert_cov3D = None
            expert_scales_raster = expert_scales_final
            expert_rotations_raster = expert_rotations_final
            if pipe.compute_cov3D_python:
                expert_cov3D = pc.covariance_activation(expert_scales_final, scaling_modifier, expert_rotations_final)
                expert_scales_raster = None
                expert_rotations_raster = None

            expert_colors = colors_precomp
            if expert_colors is not None:
                expert_colors = expert_colors.clone()
                if vis_expert_rgb_delta is not None:
                    vis_rgb_index = min(expert_idx, vis_expert_rgb_delta.shape[1] - 1)
                    expert_colors[deformation_point] = torch.clamp(
                        expert_colors[deformation_point] + vis_expert_rgb_delta[deformation_point, vis_rgb_index],
                        0.0,
                        1.0,
                    )

            expert_render, expert_radii, expert_depth = rasterizer(
                means3D=expert_means3D,
                means2D=means2D,
                shs=shs if expert_colors is None else None,
                colors_precomp=expert_colors,
                opacities=expert_opacity_final,
                scales=expert_scales_raster,
                rotations=expert_rotations_raster,
                cov3D_precomp=expert_cov3D,
            )

            weight_map = torch.zeros((means3D.shape[0],), device=means3D.device, dtype=expert_render.dtype)
            weight_map[deformation_point] = gaussian_pi_geo_prior[deformation_point, expert_idx]
            weight_render, _, _ = weight_rasterizer(
                means3D=expert_means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=weight_map.unsqueeze(-1).expand(-1, 3),
                opacities=expert_opacity_final,
                scales=expert_scales_raster,
                rotations=expert_rotations_raster,
                cov3D_precomp=expert_cov3D,
            )

            expert_renders.append(expert_render)
            expert_depths.append(expert_depth)
            expert_radii_list.append(expert_radii)
            weight_signal = weight_render.mean(dim=0)
            if not bool(active_expert_prior[expert_idx].item()):
                weight_signal = torch.zeros_like(weight_signal)
            pi_geo_weight_logits.append(weight_signal)

        expert_renders_stacked = torch.stack(expert_renders, dim=0)
        expert_depths_stacked = torch.stack(expert_depths, dim=0)
        expert_radii_stacked = torch.stack(expert_radii_list, dim=0)
        pi_geo_weight_logits = torch.stack(pi_geo_weight_logits, dim=0)
        expert_pixel_covered = pi_geo_weight_logits > 0
        coverage_present = expert_pixel_covered.any(dim=0, keepdim=True)
        masked_weight_logits = torch.where(
            expert_pixel_covered,
            pi_geo_weight_logits,
            torch.full_like(pi_geo_weight_logits, -1e9),
        )
        pi_geo_weights = torch.softmax(masked_weight_logits, dim=0)
        pi_geo_weights = torch.where(
            coverage_present.expand_as(pi_geo_weights),
            pi_geo_weights,
            torch.zeros_like(pi_geo_weights),
        )

        rendered_image = (expert_renders_stacked * pi_geo_weights.unsqueeze(1)).sum(dim=0)
        depth = (expert_depths_stacked * pi_geo_weights).sum(dim=0)
        radii = expert_radii_stacked.max(dim=0).values

        deformation_aux["expert_renders"] = expert_renders_stacked
        deformation_aux["pixel_routing_weights"] = pi_geo_weights
    else:
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "deformation_aux": deformation_aux if stage != "coarse" and deformation_point.any() else {},}
