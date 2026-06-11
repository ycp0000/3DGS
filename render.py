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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.device_utils import get_device
from time import time
try:
    import open3d as o3d
except ImportError:
    o3d = None
from utils.graphics_utils import fov2focal


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def load_frozen_router_assembly(*args, **kwargs):
    from models.endomoeg.inference import load_frozen_router_assembly as loader

    return loader(*args, **kwargs)


def render_frozen_expert_ensemble(*args, **kwargs):
    from models.endomoeg.router_training import (
        render_frozen_expert_ensemble as renderer,
    )

    return renderer(*args, **kwargs)


def _tensor_to_hw_numpy(tensor, name):
    array = tensor.detach().cpu().numpy()
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim == 3:
        if array.shape[0] == 1:
            return np.ascontiguousarray(array[0])
        if array.shape[-1] == 1:
            return np.ascontiguousarray(array[..., 0])
    if array.ndim == 4 and array.shape[0] == 1:
        return _tensor_to_hw_numpy(torch.from_numpy(array[0]), name)
    raise ValueError(f"{name} must have shape [H, W], [1, H, W], or [H, W, 1], got {array.shape}")


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    reconstruct=False,
    render_view=None,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []

    test_times = 1
    for i in range(test_times):
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if idx == 0 and i == 0:
                time1 = time()
            if render_view is None:
                rendering = render(view, gaussians, pipeline, background)
            else:
                rendering = render_view(view)
            if i == test_times-1:
                render_depths.append(rendering["depth"])
                render_images.append(rendering["render"].cpu())
                if name in ["train", "test", "video"]:
                    gt = view.original_image[0:3, :, :]
                    gt_list.append(gt)
                    mask = view.mask
                    mask_list.append(mask)
                    gt_depth = view.original_depth
                    gt_depths.append(gt_depth)

    time2=time()
    # print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    # import pdb; pdb.set_trace()
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image /= 255.0
            torchvision.utils.save_image(image, os.path.join(depth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image /= 255.0
            torchvision.utils.save_image(image, os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"))
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    
    if reconstruct:
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        pipeline_stage = str(
            getattr(hyperparam, "endomoeg_pipeline_stage", "") or ""
        ).strip().lower()
        render_view = None
        if pipeline_stage in {"router", "joint"}:
            gaussians = None
            scene = Scene(
                dataset,
                gaussians,
                load_iteration=None,
                shuffle=False,
                initialize_gaussians=False,
            )
            if pipeline_stage == "joint":
                assembly_bundle_dir = getattr(
                    hyperparam,
                    "endomoeg_joint_output_dir",
                )
                assembly_router_bundle = os.path.join(
                    assembly_bundle_dir,
                    "router.pth",
                )
            else:
                assembly_bundle_dir = getattr(
                    hyperparam,
                    "endomoeg_bundle_dir",
                )
                assembly_router_bundle = (
                    getattr(hyperparam, "endomoeg_router_bundle", "") or None
                )
            assembly = load_frozen_router_assembly(
                assembly_bundle_dir,
                expected_source_path=dataset.source_path,
                device=get_device(),
                minimum_expert_psnr=float(
                    getattr(hyperparam, "endomoeg_min_expert_psnr", 0.0)
                ),
                router_bundle_path=assembly_router_bundle,
            )
            if int(iteration) not in (-1, assembly.iteration):
                raise ValueError(
                    "Requested iteration {} does not match Router bundle "
                    "iteration {}".format(iteration, assembly.iteration)
                )
            infer_iter = assembly.iteration

            def render_view(view):
                return render_frozen_expert_ensemble(
                    view,
                    assembly.ensemble,
                    assembly.router,
                    pipeline,
                    background,
                )
        else:
            gaussians = GaussianModel(dataset.sh_degree, hyperparam)
            scene = Scene(
                dataset,
                gaussians,
                load_iteration=iteration,
                shuffle=False,
            )
            infer_iter = (
                scene.loaded_iter
                if scene.loaded_iter is not None
                else max(0, int(iteration))
            )
        if hasattr(hyperparam, "current_iteration"):
            hyperparam.current_iteration = infer_iter
        if hasattr(hyperparam, "iterations") and hyperparam.iterations < infer_iter:
            hyperparam.iterations = infer_iter

        device = get_device()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        
        if not skip_train:
            render_set(dataset.model_path, "train", infer_iter, scene.getTrainCameras(), gaussians, pipeline, background, reconstruct=not skip_train, render_view=render_view)
        if not skip_test:
            render_set(dataset.model_path, "test", infer_iter, scene.getTestCameras(), gaussians, pipeline, background, reconstruct=not skip_test, render_view=render_view)
        if not skip_video:
            render_set(dataset.model_path, "video", infer_iter, scene.getVideoCameras(), gaussians, pipeline, background, reconstruct=not skip_video, render_view=render_view)

def reconstruct_point_cloud(images, masks, depths, camera_parameters, name):
    if o3d is None:
        raise ImportError("open3d is required for point cloud reconstruction")
    import cv2
    import copy
    output_frame_folder = os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = _tensor_to_hw_numpy(depths[i_frame], "depth").astype(np.float32, copy=False)
        mask = _tensor_to_hw_numpy(masks[i_frame], "mask")
        
        rgb_new = copy.deepcopy(rgb_np)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])

        valid_depth = depth_np[depth_np != 0]
        if valid_depth.size == 0:
            print(f"[WARN] Skip point-cloud reconstruction for frame {i_frame} in {name}: no valid depth values")
            continue

        close_depth = np.percentile(valid_depth, 5)
        inf_depth = np.percentile(valid_depth, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, width / 2),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)
