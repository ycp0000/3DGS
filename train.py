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
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from lpipsPyTorch import LPIPS as LocalLPIPS
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.optimizer_utils import collect_optimizer_group_metrics
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
import cv2
from torchmetrics.functional.regression import pearson_corrcoef

try:
    import lpips as external_lpips
except ImportError:
    external_lpips = None
from utils.device_utils import get_device, safe_cuda_event
from utils.eval_utils import select_fixed_views
from utils.scene_utils import render_training_image
from utils.temporal_utils import nearest_adjacent_time, sorted_unique_times
from time import time
from models.tracking.cams_gs_moe_tracking import required_endomoeg_components
from scene.tracking_losses import compute_tracking_losses
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def validate_training_source_args(args):
    source_path = os.path.abspath(args.source_path)
    args.source_path = source_path

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Training source_path does not exist: {source_path}")

    if getattr(args, "extra_mark", None) == "endonerf":
        poses_bounds_path = os.path.join(source_path, "poses_bounds.npy")
        if not os.path.exists(poses_bounds_path):
            raise FileNotFoundError(
                "EndoNeRF training requires poses_bounds.npy in the scene root. "
                f"Expected: {poses_bounds_path}. "
                "Pass the absolute scene directory with -s, for example /root/3DGS/data/endonerf/<scene>."
            )

    return args


def should_reset_opacity(stage, iteration, opt, dataset):
    if stage != "coarse":
        return False
    return iteration % opt.opacity_reset_interval == 0 or (
        dataset.white_background and iteration == opt.densify_from_iter
    )


def save_completed_endomoeg_phase(
    gaussians,
    model_path,
    phase_name,
    component_output_dir="",
):
    component_map = {
        "moe_expert_global": ("shared_base", "global"),
        "moe_expert_local": ("local",),
        "moe_expert_full": ("full",),
        "moe_router_only": ("router",),
    }
    components = component_map.get(phase_name, ())
    if not components:
        return
    component_dir = (
        os.path.abspath(component_output_dir)
        if component_output_dir
        else os.path.join(model_path, "endomoeg_components")
    )
    for component in components:
        path = gaussians._deformation.save_endomoeg_component(component_dir, component)
        print(f"[EndoMoe] Saved {component} component to {path}")


def load_requested_endomoeg_components(gaussians, hyper):
    if getattr(hyper, "tracking_type", "") != "cams_gs_moe":
        return ()
    stage = str(
        getattr(hyper, "endomoeg_stage", "")
        or getattr(hyper, "cams_moe_stage", "")
    ).lower()
    components = required_endomoeg_components(stage)
    if not components:
        return ()
    component_dir = str(getattr(hyper, "endomoeg_component_dir", "") or "")
    if not component_dir:
        raise ValueError(
            f"endomoeg_component_dir is required for explicit EndoMoe stage '{stage}'"
        )
    loaded = gaussians._deformation.load_endomoeg_components(
        component_dir,
        components,
        strict=bool(getattr(hyper, "endomoeg_strict_component_loading", True)),
    )
    print(f"[EndoMoe] Loaded components for stage {stage}: {', '.join(loaded)}")
    return loaded


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    device = get_device()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = safe_cuda_event(enable_timing=True)
    iter_end = safe_cuda_event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    if external_lpips is not None:
        lpips_model = external_lpips.LPIPS(net="vgg").to(device)
    else:
        lpips_model = LocalLPIPS(net_type="vgg").to(device)
    video_cams = scene.getVideoCameras()
    temporal_times = sorted_unique_times(
        camera.time for camera in scene.getTrainCameras()
    )
    previous_tracking_phase_name = None
    final_tracking_phase_name = None
    
    for iteration in range(first_iter, final_iter+1):
        hyper.current_iteration = iteration
        hyper.iterations = opt.iterations
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        tracking_phase = gaussians._deformation.get_tracking_phase() if stage == "fine" else None
        if tracking_phase is not None:
            final_tracking_phase_name = tracking_phase.name
            if (
                previous_tracking_phase_name is not None
                and tracking_phase.name != previous_tracking_phase_name
            ):
                save_completed_endomoeg_phase(
                    gaussians,
                    scene.model_path,
                    previous_tracking_phase_name,
                    getattr(hyper, "endomoeg_component_output_dir", ""),
                )
            previous_tracking_phase_name = tracking_phase.name
        gaussians.update_learning_rate(iteration, phase=tracking_phase)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            batch_size = 16
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader")
                loader = iter(viewpoint_stack_loader)
                viewpoint_cams = next(loader)
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []

        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        deformation_aux_list = []
        temporal_aux_pair = None
        merged_d_mu_for_commit = None
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.to(device).float()
            gt_depth = viewpoint_cam.original_depth.to(device).float()
            mask = viewpoint_cam.mask.to(device)
            
            # depth_refine_iteration = 5
            # depth_refine_bounds = 4 * depth_refine_iteration
            # if iteration % depth_refine_iteration==0 and iteration <= depth_refine_bounds:
            #     depth_diff = torch.pow(gt_depth - depth, 2) * mask
            #     depth_diff = depth_diff.reshape(depth_diff.shape[0], -1)
            #     quantile = torch.quantile(depth_diff, 1.0 - 0.1, dim=1, keepdim=True)
            #     depth_to_refine = (depth_diff > quantile).reshape(*gt_depth.shape)
            #     gt_depth[depth_to_refine] = depth[depth_to_refine]
            #     viewpoint_cam.original_depth = gt_depth.detach().cpu()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            if stage != "coarse" and render_pkg.get("deformation_aux"):
                deformation_aux_list.append(render_pkg["deformation_aux"])

        if (
            stage == "fine"
            and deformation_aux_list
            and float(getattr(hyper, "lambda_geo_temp", 0.0) or 0.0) > 0.0
            and gaussians._deformation_table.any()
        ):
            reference_camera = viewpoint_cams[0]
            adjacent_time = nearest_adjacent_time(
                float(reference_camera.time),
                temporal_times,
            )
            if adjacent_time is not None:
                deformation_mask = gaussians._deformation_table
                adjacent_times = torch.full(
                    (int(deformation_mask.sum().item()), 1),
                    float(adjacent_time),
                    device=gaussians.get_xyz.device,
                    dtype=gaussians.get_xyz.dtype,
                )
                gaussians._deformation(
                    gaussians.get_xyz[deformation_mask],
                    gaussians._scaling[deformation_mask],
                    gaussians._rotation[deformation_mask],
                    gaussians._opacity[deformation_mask],
                    adjacent_times,
                    camera=reference_camera,
                )
                adjacent_aux = gaussians._deformation.get_aux_outputs()
                current_aux = deformation_aux_list[0]
                if (
                    "d_mu" in current_aux
                    and "d_mu" in adjacent_aux
                    and current_aux["d_mu"].shape == adjacent_aux["d_mu"].shape
                ):
                    temporal_aux_pair = (
                        torch.stack((current_aux["d_mu"], adjacent_aux["d_mu"]), dim=0),
                        current_aux["d_mu"].new_tensor(
                            (float(reference_camera.time), float(adjacent_time))
                        ),
                    )
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)
                
        # mask_tensor = None
        if iteration < 1000:
            color_diff = torch.pow(image_tensor-gt_image_tensor, 2).sum(dim=1, keepdim=True)
            color_diff = color_diff.reshape(color_diff.shape[0], -1)
            quantile = torch.quantile(color_diff, 0.98, dim=1)
            color_to_refine = (color_diff > quantile).reshape(*mask_tensor.shape)
            mask_tensor[color_to_refine] = torch.ones(color_to_refine.sum(), device=device, dtype=torch.bool)
                
        if iteration % 500 == 0 and iteration < 1000:
            tmp = (color_to_refine.squeeze().detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('color_to_refine.png', tmp)
            tmp = (image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('image.png', tmp)
            tmp = (gt_image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('gtimage.png', tmp)
            tmp = (mask_tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite('mask.png', tmp)
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)

        scene_mode = getattr(scene, "mode", "binocular")
        if (gt_depth_tensor!=0).sum() < 10:
            depth_loss = torch.zeros((), device=device)
        elif scene_mode == 'binocular':
            depth_pred = depth_tensor.clone()
            depth_gt = gt_depth_tensor.clone()
            depth_pred[depth_pred != 0] = 1 / depth_pred[depth_pred != 0]
            depth_gt[depth_gt != 0] = 1 / depth_gt[depth_gt != 0]
            depth_loss = l1_loss(depth_pred, depth_gt, mask_tensor)
        elif scene_mode == 'monocular':
            depth_pred = depth_tensor.reshape(-1, 1)
            depth_gt = gt_depth_tensor.reshape(-1, 1)
            mask_tmp = mask_tensor.reshape(-1)
            depth_pred = depth_pred[mask_tmp != 0, :]
            depth_gt = depth_gt[mask_tmp != 0, :]
            depth_loss = 0.001 * (1 - pearson_corrcoef(depth_gt, depth_pred))
        else:
            raise ValueError(f"{scene_mode} is not implemented.")

        depth_tvloss = TV_loss(depth_tensor, mask_tensor)
        img_tvloss = TV_loss(image_tensor, mask_tensor)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)

        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()

        loss = Ll1 + depth_loss + tv_loss

        geo_expert_names, vis_expert_names = gaussians._deformation.get_expert_names()
        tracking_metrics = {}
        if stage == "fine" and deformation_aux_list:
            merged_aux = {}
            for key in deformation_aux_list[0].keys():
                values = [a[key] for a in deformation_aux_list if key in a]
                if not values:
                    continue
                if not torch.is_tensor(values[0]):
                    merged_aux[key] = values[0]
                    continue
                if values[0].dim() == 0:
                    merged_aux[key] = torch.stack(values).mean()
                else:
                    merged_aux[key] = torch.cat(values, dim=0)

            if temporal_aux_pair is not None:
                merged_aux["d_mu_sequence"], merged_aux["time_sequence"] = temporal_aux_pair

            if stage == "fine" and iteration % 500 == 0 and deformation_aux_list and tb_writer is not None:
                for k, v in merged_aux.items():
                    if k.startswith("dbg_") and torch.is_tensor(v) and v.numel() == 1:
                        tb_writer.add_scalar(
                            f"{stage}/moe_debug/{k}",
                            float(v.detach().cpu().item()),
                            iteration,
                        )

            if "d_mu" in merged_aux:
                merged_d_mu_for_commit = merged_aux["d_mu"].detach()

            active_geo = tracking_phase.active_geo if tracking_phase is not None else len(geo_expert_names)
            active_vis = tracking_phase.active_vis if tracking_phase is not None else len(vis_expert_names)
            enable_visibility = tracking_phase.enable_visibility if tracking_phase is not None else False
            tracking_loss_dict = compute_tracking_losses(
                merged_aux,
                iteration,
                hyper,
                gaussians._deformation.get_previous_d_mu(),
                active_geo,
                active_vis,
                enable_visibility,
                geo_expert_names,
                vis_expert_names,
                force_geo_expert=tracking_phase.force_geo_expert if tracking_phase is not None else None,
                force_vis_expert=tracking_phase.force_vis_expert if tracking_phase is not None else None,
            )
            tracking_loss = torch.zeros((), device=loss.device)

            tracking_loss_names = sorted(
                name for name, value in tracking_loss_dict.items()
                if name.startswith("L_") and torch.is_tensor(value)
            )

            for loss_name in tracking_loss_names:
                tracking_loss = tracking_loss + tracking_loss_dict[loss_name]

            tracking_metrics = dict(tracking_loss_dict)
            if tracking_phase is not None:
                tracking_metrics["phase_active_geo"] = torch.tensor(float(tracking_phase.active_geo), device=loss.device)
                tracking_metrics["phase_active_vis"] = torch.tensor(float(tracking_phase.active_vis), device=loss.device)
                tracking_metrics["phase_visibility_enabled"] = torch.tensor(float(tracking_phase.enable_visibility), device=loss.device)
            tracking_metrics["L_tracking_total"] = tracking_loss.detach()

            loss = loss + tracking_loss

            if iteration % 500 == 0:
                print(f"\n[TRACKING LOSS DEBUG] iter={iteration}")
                print_keys = [
                    "L_tracking_total",
                    "L_sat_geo",
                    "L_mag_geo",
                    "L_raw_geo",
                    "L_route_conf_geo",
                    "L_route_conf_vis",
                    "L_expert_diversity_geo",
                    "route_max_prob_geo",
                    "route_max_prob_vis",
                    "route_margin_geo",
                    "route_margin_vis",
                    "route_zero_fraction_geo",
                    "route_effective_experts_geo",
                    "route_dense_effective_experts_geo",
                    "route_sparse_dense_l1_geo",
                    "route_balance_scale",
                    "route_confidence_scale",
                    "pixel_route_entropy_geo",
                    "pixel_route_max_prob_geo",
                    "pixel_route_covered_fraction",
                    "pixel_router_residual_abs_mean",
                    "pixel_router_residual_abs_max",
                    "expert_diversity_geo",
                    "L_balance_geo",
                    "L_balance_vis",
                    "L_entropy",
                    "L_geo_temp",
                    "geo_temp_velocity",
                    "temporal_pair_count",
                    "L_geo_spatial",
                    "geo_spatial_roughness",
                    "geo_spatial_sample_count",
                    "geo_spatial_neighbor_count",
                    "L_vis_sparse",
                    "L_decouple",
                ]

                for expert_name in geo_expert_names:
                    print_keys.extend(
                        [
                            f"usage_geo_{expert_name}",
                            f"dense_usage_geo_{expert_name}",
                            f"route_coverage_geo_{expert_name}",
                            f"pixel_coverage_geo_{expert_name}",
                            f"target_usage_geo_{expert_name}",
                            f"geo_disp_ratio_{expert_name}",
                            f"geo_saturation_{expert_name}",
                        ]
                    )
                for vis_name in vis_expert_names:
                    print_keys.extend(
                        [
                            f"usage_vis_{vis_name}",
                            f"target_usage_vis_{vis_name}",
                        ]
                    )

                for k in print_keys:
                    v = tracking_metrics.get(k, None)
                    if torch.is_tensor(v) and v.numel() == 1:
                        print(f"{k}: {float(v.detach().cpu().item()):.8e}")

        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
            loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        if stage == "fine" and iteration % 10 == 0:
            tracking_metrics.update(
                collect_optimizer_group_metrics(gaussians.optimizer)
            )
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        # if stage == "fine" and iteration % 500 == 0:
        #     net = gaussians._deformation.deformation_net
        #     for name, p in net.named_parameters():
        #         if any(s in name.lower() for s in ["geo", "router", "expert"]):
        #             if p.grad is None:
        #                 print("[NO GRAD]", name)
        #             else:
        #                 print("[GRAD]", name, p.grad.norm().item())



        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "psnr": f"{psnr_:.{2}f}",
                    "point":f"{total_point}",
                }
                if tracking_metrics:
                    if tracking_phase is not None:
                        postfix["phase"] = tracking_phase.name
                    for idx, expert_name in enumerate(geo_expert_names):
                        usage_key = f"usage_geo_{expert_name}"
                        if usage_key in tracking_metrics:
                            postfix[f"uG{idx}"] = f"{tracking_metrics[usage_key].detach().item():.2f}"
                    for idx, vis_name in enumerate(vis_expert_names):
                        usage_key = f"usage_vis_{vis_name}"
                        if usage_key in tracking_metrics:
                            postfix[f"uV{idx}"] = f"{tracking_metrics[usage_key].detach().item():.2f}"
                    if "route_max_prob_geo" in tracking_metrics:
                        postfix["pG"] = f"{tracking_metrics['route_max_prob_geo'].detach().item():.2f}"
                    if "route_max_prob_vis" in tracking_metrics:
                        postfix["pV"] = f"{tracking_metrics['route_max_prob_vis'].detach().item():.2f}"
                    if "L_sat_geo" in tracking_metrics:
                        postfix["Lsat"] = f"{tracking_metrics['L_sat_geo'].detach().item():.1e}"
                    if "L_mag_geo" in tracking_metrics:
                        postfix["Lmag"] = f"{tracking_metrics['L_mag_geo'].detach().item():.1e}"
                    if "L_raw_geo" in tracking_metrics:
                        postfix["Lraw"] = f"{tracking_metrics['L_raw_geo'].detach().item():.1e}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                [pipe, background],
                stage,
                tracking_metrics,
                lpips_model,
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                if visibility_filter.numel() == gaussians.max_radii2D.numel() and visibility_filter.any():
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if viewspace_point_tensor_grad.shape[0] == visibility_filter.shape[0] and visibility_filter.any():
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration * (opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (opt.densify_until_iter)
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (opt.densify_until_iter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if should_reset_opacity(stage, iteration, opt, dataset):
                    print("reset opacity")
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if stage == "fine" and merged_d_mu_for_commit is not None:
                gaussians._deformation.commit_previous_d_mu(merged_d_mu_for_commit)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if stage == "fine" and final_tracking_phase_name is not None:
        save_completed_endomoeg_phase(
            gaussians,
            scene.model_path,
            final_tracking_phase_name,
            getattr(hyper, "endomoeg_component_output_dir", ""),
        )

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    # Coarse stage: static reconstruction
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)

    # Save checkpoint at end of coarse stage
    coarse_checkpoint_path = scene.model_path + "/chkpnt_coarse.pth"
    print(f"\n[ITER {opt.coarse_iterations}] Saving coarse stage checkpoint to {coarse_checkpoint_path}")
    torch.save((gaussians.capture(), opt.coarse_iterations), coarse_checkpoint_path)

    # Load coarse checkpoint before fine stage to preserve static reconstruction
    print(f"[ITER {opt.coarse_iterations}] Loading coarse checkpoint for fine stage")
    (model_params, _) = torch.load(coarse_checkpoint_path)
    gaussians.restore(model_params, opt)
    load_requested_endomoeg_components(gaussians, hyper)

    # Fine stage: dynamic optimization
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    stage,
    tracking_metrics=None,
    lpips_model=None,
):
    if tb_writer:
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/l1_loss",
            Ll1.item(),
            iteration,
        )
        tb_writer.add_scalar(
            f"{stage}/train_loss_patches/total_loss",
            loss.item(),
            iteration,
        )
        tb_writer.add_scalar(
            f"{stage}/iter_time",
            elapsed,
            iteration,
        )

        # Automatically log all scalar tracking metrics.
        # This includes:
        #   L_sat_geo, L_mag_geo,
        #   sat_geo_e1_disp, sat_geo_e1_rot,
        #   mag_geo_e1_mu, mag_geo_e1_rot, etc.
        if tracking_metrics:
            for key, value in tracking_metrics.items():
                if not torch.is_tensor(value):
                    continue
                if value.numel() != 1:
                    continue

                if key.startswith("usage_"):
                    tag = f"{stage}/tracking/routes/{key}"
                elif key.startswith("route_"):
                    tag = f"{stage}/tracking/confidence/{key}"
                elif key.startswith("entropy_"):
                    tag = f"{stage}/tracking/entropy/{key}"
                elif key.startswith("target_usage_"):
                    tag = f"{stage}/tracking/targets/{key}"
                elif key.startswith("L_"):
                    tag = f"{stage}/tracking/losses/{key}"
                else:
                    tag = f"{stage}/tracking/stats/{key}"

                tb_writer.add_scalar(
                    tag,
                    float(value.detach().cpu().item()),
                    iteration,
                )

    if iteration not in testing_iterations:
        return

    validation_configs = (
        ("test", select_fixed_views(scene.getTestCameras(), count=4)),
        ("train", select_fixed_views(scene.getTrainCameras(), count=4)),
    )
    device = scene.gaussians.get_xyz.device
    with torch.no_grad():
        for split_name, cameras in validation_configs:
            if not cameras:
                continue
            metric_totals = {
                "l1": 0.0,
                "psnr": 0.0,
                "ssim": 0.0,
                "lpips": 0.0,
            }
            for view_index, viewpoint in enumerate(cameras):
                render_output = renderFunc(
                    viewpoint,
                    scene.gaussians,
                    *renderArgs,
                    stage=stage,
                    update_deformation_stats=False,
                )
                image = torch.clamp(render_output["render"], 0.0, 1.0).unsqueeze(0)
                gt_image = torch.clamp(
                    viewpoint.original_image.to(device).float(),
                    0.0,
                    1.0,
                ).unsqueeze(0)
                mask = viewpoint.mask.to(device).unsqueeze(0)
                masked_image = image * mask
                masked_gt = gt_image * mask

                metric_totals["l1"] += float(l1_loss(image, gt_image, mask).item())
                metric_totals["psnr"] += float(psnr(image, gt_image, mask).mean().item())
                metric_totals["ssim"] += float(ssim(masked_image, masked_gt).item())
                if lpips_model is not None:
                    metric_totals["lpips"] += float(
                        lpips_loss(masked_image, masked_gt, lpips_model).item()
                    )

                if tb_writer and view_index < 2:
                    image_name = getattr(viewpoint, "image_name", str(view_index))
                    tb_writer.add_image(
                        f"{stage}/validation/{split_name}/{image_name}/render",
                        image[0],
                        global_step=iteration,
                    )
                    if iteration == testing_iterations[0]:
                        tb_writer.add_image(
                            f"{stage}/validation/{split_name}/{image_name}/ground_truth",
                            gt_image[0],
                            global_step=iteration,
                        )

            view_count = float(len(cameras))
            averaged_metrics = {
                name: value / view_count
                for name, value in metric_totals.items()
            }
            print(
                f"\n[ITER {iteration}] Evaluating {split_name}: "
                f"L1 {averaged_metrics['l1']:.6f} "
                f"PSNR {averaged_metrics['psnr']:.3f} "
                f"SSIM {averaged_metrics['ssim']:.4f} "
                f"LPIPS {averaged_metrics['lpips']:.4f}"
            )
            if tb_writer:
                for metric_name, metric_value in averaged_metrics.items():
                    tb_writer.add_scalar(
                        f"{stage}/validation/{split_name}/{metric_name}",
                        metric_value,
                        iteration,
                    )

        if tb_writer:
            tb_writer.add_scalar(
                f"{stage}/scene/total_points",
                scene.gaussians.get_xyz.shape[0],
                iteration,
            )
            deformation_count = scene.gaussians._deformation_table.sum()
            tb_writer.add_scalar(
                f"{stage}/scene/deformation_rate",
                float(
                    deformation_count
                    / max(scene.gaussians.get_xyz.shape[0], 1)
                ),
                iteration,
            )

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
         torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 5000, 7_000, 9000, 10000, 14000, 20000, 30_000,45000,60000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    if int(getattr(args, "endomoeg_stage_iterations", -1)) > 0:
        args.iterations = int(args.endomoeg_stage_iterations)
        args.position_lr_max_steps = int(args.endomoeg_stage_iterations)
    args = validate_training_source_args(args)
    print(f"Training source_path: {args.source_path}")
    print(f"Training extra_mark: {getattr(args, 'extra_mark', None)}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.extra_mark)

    # All done
    print("\nTraining complete.")
