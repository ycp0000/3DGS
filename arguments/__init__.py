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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.extra_mark = None
        self.camera_extent = None
        self.scene_mode = "binocular"
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_grid=False
        self.no_ds=False
        self.no_dr=False
        self.no_do=True

        self.tracking_type = "original"

        self.K_geo = 3
        self.K_vis = 2
        self.geo_hidden_dim = 64
        self.vis_hidden_dim = 64

        self.use_soft_routing = True
        self.use_topk = False
        self.topk = 2
        self.topk_geo = 2
        self.topk_vis = 1
        self.router_noise_geo = 0.0
        self.router_noise_vis = 0.0

        self.temperature_geo_init = 2.0
        self.temperature_geo_final = 0.7
        self.temperature_vis_init = 2.0
        self.temperature_vis_final = 1.0

        self.max_disp_smooth_ratio = 0.01
        self.max_disp_local_ratio = 0.03
        self.max_disp_shared_ratio = 0.01
        self.max_rot_smooth = 0.05
        self.max_rot_local = 0.10
        self.max_rot_shared = 0.05
        self.max_scale_smooth = 0.05
        self.max_scale_local = 0.10
        self.max_scale_shared = 0.05
        self.max_opacity_delta = 4.0

        self.lambda_balance_geo = 0.005
        self.lambda_balance_vis = 0.001
        self.lambda_route_conf_geo = 0.002
        self.lambda_route_conf_vis = 0.001
        self.lambda_expert_diversity_geo = 0.001
        self.lambda_entropy_geo = 0.001
        self.lambda_entropy_vis = 0.0005
        self.lambda_geo_temp = 0.01
        self.lambda_geo_spatial = 0.01
        self.lambda_vis_sparse = 0.005
        self.lambda_decouple = 0.05

        self.warmup_iters = 1000
        self.enable_shared_only_iter = 1000
        self.enable_smooth_geo_iter = 1000
        self.enable_local_geo_iter = 1500
        self.enable_visibility_iter = 2000
        self.enable_sparse_routing_iter = 1500
        self.enable_route_stability_iter = 1500
        self.enable_decouple_iter = 2200
        self.entropy_end_iter = 2500
        self.enable_visibility = True

        self.current_iteration = 0
        super().__init__(parser, "ModelHiddenParams")
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfilepath = None
    try:
        if args_cmdline.model_path:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        if cfgfilepath is not None:
            print("Config file not found at {}".format(cfgfilepath))
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    for k, v in vars(args_cmdline).items():
        if k not in merged_dict:
            merged_dict[k] = v
    return Namespace(**merged_dict)
