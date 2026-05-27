ModelParams = dict(
    extra_mark='endonerf',
    camera_extent=10,
)

OptimizationParams = dict(
    coarse_iterations=1000,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.0000016,
    deformation_lr_delay_mult=0.01,
    grid_lr_init=0.0016,
    grid_lr_final=0.000016,
    iterations=9000,
    percent_dense=0.01,
    opacity_reset_interval=3000,
    position_lr_max_steps=9000,
    pruning_interval=3000,
)

ModelHiddenParams = dict(
    tracking_type='cams_gs',
    enable_visibility=True,
    max_disp_smooth_ratio=0.01,
    max_disp_local_ratio=0.03,
    kplanes_config={
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 100],
    },
    multires=[1, 2, 4, 8],
    defor_depth=0,
    net_width=32,
    plane_tv_weight=0,
    time_smoothness_weight=0,
    l1_time_planes=0,
    weight_decay_iteration=0,
    stage_global_only_end=600,
    stage_graph_bootstrap_end=2700,
    stage_local_motion_end=3600,
    stage_visibility_enable_iter=6300,
    stage_lifecycle_enable_iter=7650,
)
