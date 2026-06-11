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

from __future__ import annotations

import os
import random
import json
from typing import TYPE_CHECKING

import numpy as np
import torch

from utils.system_utils import searchForMaxIteration

if TYPE_CHECKING:
    from arguments import ModelParams
    from scene.gaussian_model import GaussianModel


def __getattr__(name):
    if name == "GaussianModel":
        from scene.gaussian_model import GaussianModel as gaussian_model_cls

        return gaussian_model_cls
    raise AttributeError(f"module 'scene' has no attribute {name!r}")
class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        load_coarse=False,
        initialize_gaussians=True,
    ):
        from scene.dataset import FourDGSdataset
        from scene.dataset_readers import sceneLoadTypeCallbacks
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.mode = getattr(args, "scene_mode", "binocular")
        if self.mode not in {"binocular", "monocular"}:
            raise ValueError(f"Unsupported scene_mode: {self.mode}")

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        
        source_path = os.path.abspath(args.source_path)
        args.source_path = source_path
        poses_bounds_path = os.path.join(source_path, "poses_bounds.npy")
        extra_mark = getattr(args, "extra_mark", None)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Scene source_path does not exist: {source_path}")

        if extra_mark == "endonerf":
            if not os.path.exists(poses_bounds_path):
                raise FileNotFoundError(
                    "EndoNeRF scene requires poses_bounds.npy at "
                    f"{poses_bounds_path}. Check that -s is an absolute scene directory, e.g. /root/3DGS/data/endonerf/<scene>."
                )
            scene_info = sceneLoadTypeCallbacks["endonerf"](source_path, args.white_background, args.eval, scene_mode=self.mode)
            print("Found poses_bounds.npy and extra_mark='endonerf', assuming EndoNeRF data")
        elif os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](source_path, args.white_background, args.eval)
        elif os.path.exists(poses_bounds_path):
            scene_info = sceneLoadTypeCallbacks["dynerf"](source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](source_path, False, args.eval)
        elif os.path.exists(os.path.join(source_path, "point_cloud.obj")) or os.path.exists(os.path.join(source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](source_path, args.white_background, args.eval, scene_mode=self.mode)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            raise AssertionError(
                "Could not recognize scene type. "
                f"source_path={source_path}, extra_mark={extra_mark!r}, "
                f"expected EndoNeRF file={poses_bounds_path}"
            )
                
        self.maxtime = scene_info.maxtime
        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        # # self.cameras_extent = args.camera_extent
        # print("self.cameras_extent is ", self.cameras_extent)
        # self.gaussians._deformation.set_scene_scale(self.cameras_extent)


        

        self.cameras_extent = float(scene_info.nerf_normalization["radius"])
        print("self.cameras_extent is ", self.cameras_extent)

        # For fixed-camera dynamic endoscopic data, camera radius may be zero.
        # Deformation scale should be based on tissue / point-cloud geometry instead.
        pc = scene_info.point_cloud.points

        lo = np.percentile(pc, 5, axis=0)
        hi = np.percentile(pc, 95, axis=0)
        robust_bbox_diag = float(np.linalg.norm(hi - lo))

        xyz_max = pc.max(axis=0)
        xyz_min = pc.min(axis=0)
        bbox_diag = float(np.linalg.norm(xyz_max - xyz_min))

        arg_extent = float(getattr(args, "camera_extent", 0.0) or 0.0)

        if robust_bbox_diag > 1e-8:
            deformation_scale = robust_bbox_diag
        elif bbox_diag > 1e-8:
            deformation_scale = bbox_diag
        elif arg_extent > 1e-8:
            deformation_scale = arg_extent
        elif self.cameras_extent > 1e-8:
            deformation_scale = self.cameras_extent
        else:
            deformation_scale = 1.0

        print("robust_bbox_diag is ", robust_bbox_diag)
        print("bbox_diag is ", bbox_diag)
        print("args.camera_extent is ", arg_extent)
        print("deformation_scale is ", deformation_scale)

        if self.gaussians is not None:
            self.gaussians._deformation.set_scene_scale(deformation_scale)





        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras,args)
        
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if not isinstance(xyz_max, torch.Tensor):
            xyz_max = torch.from_numpy(xyz_max) if isinstance(xyz_max, np.ndarray) else torch.as_tensor(xyz_max)
        if not isinstance(xyz_min, torch.Tensor):
            xyz_min = torch.from_numpy(xyz_min) if isinstance(xyz_min, np.ndarray) else torch.as_tensor(xyz_min)
        if self.gaussians is not None:
            self.gaussians._deformation.set_aabb(xyz_max, xyz_min)

        if initialize_gaussians and self.gaussians is None:
            raise ValueError(
                "Scene requires a GaussianModel when initialize_gaussians=True"
            )
        if not initialize_gaussians:
            return
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            # self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
            self.gaussians.create_from_pcd(scene_info.point_cloud, args.camera_extent, self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
