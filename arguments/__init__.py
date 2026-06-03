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
import math
import sys
import os

from utils.device_utils import get_device_str
from utils.params_utils import normalize_legacy_config_keys



def _same_arg_value(left, right):
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right

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
        group = Namespace()
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
        self.data_device = get_device_str()
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

        self.K_geo = 4
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

        self.max_disp_hexplane_ratio = 0.01
        self.max_disp_smooth_ratio = 0.01
        self.max_disp_global_ratio = 0.01
        self.max_disp_local_ratio = 0.03
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
        self.target_geo_static = 0.30
        self.target_geo_smooth = float("nan")
        self.target_geo_hexplane = float("nan")
        self.target_geo_local = 0.20
        self.target_geo_residual_smooth = float("nan")
        self.target_geo_static_stage2 = float("nan")
        self.target_geo_smooth_stage2 = float("nan")
        self.target_geo_hexplane_stage2 = float("nan")
        self.target_vis_stable = 0.85
        self.target_vis_transient = 0.15
        self.target_usage_geo_global = 0.45
        self.target_usage_geo_local = 0.10
        self.target_usage_geo_cut_graph = 0.45
        self.target_usage_vis_stable = 0.85
        self.target_usage_vis_transient = 0.15
        self.sat_threshold = 0.8
        self.lambda_mag_g1_mu = 1e-4
        self.lambda_mag_g2_mu = 2e-5
        self.lambda_mag_g3_mu = float("nan")
        self.lambda_sat_g1_disp = 5e-4
        self.lambda_sat_g2_disp = 1e-4
        self.lambda_sat_g3_disp = float("nan")
        self.lambda_raw_g1_disp = 1e-4
        self.lambda_raw_g2_disp = 1e-4
        self.lambda_raw_g3_disp = float("nan")
        self.lambda_motion_mag_global = 1e-4
        self.lambda_motion_mag_local = 2e-5
        self.lambda_motion_mag_cut_graph = 2e-5

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

        self.stage_global_only_end = -1
        self.stage_graph_bootstrap_end = -1
        self.stage_local_motion_end = -1
        self.stage_visibility_enable_iter = -1
        self.stage_lifecycle_enable_iter = -1
        self.lambda_appearance_reg = 1e-4
        self.lambda_lifecycle_balance = 1e-4
        self.lambda_lifecycle_reg = 1e-4
        self.target_lifecycle_persistent = 0.8
        self.endomoeg_stage = ""
        self.cams_moe_stage = ""
        self.endomoeg_expert_global_end = -1
        self.endomoeg_expert_local_end = -1
        self.endomoeg_expert_full_end = -1
        self.endomoeg_router_only_end = -1
        self.moe_router_hidden_dim = 64

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
    args_defaults = parser.parse_args([])

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

    merged_dict = normalize_legacy_config_keys(vars(args_cfgfile).copy())
    for k, v in vars(args_cmdline).items():
        default_value = getattr(args_defaults, k)
        if k not in merged_dict or not _same_arg_value(v, default_value):
            merged_dict[k] = v
    return Namespace(**merged_dict)
