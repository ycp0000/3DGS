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
    densify_until_iter=15000,
)

ModelHiddenParams = dict(
    tracking_type='heterogeneous_moe',
    enable_visibility=True,
    K_geo=4,
    K_vis=2,
    geo_hidden_dim=64,
    vis_hidden_dim=64,
    temperature_geo_init=2.0,
    temperature_geo_final=0.7,
    temperature_vis_init=2.0,
    temperature_vis_final=1.0,
    max_disp_hexplane_ratio=0.01,
    max_disp_smooth_ratio=0.01,
    max_disp_local_ratio=0.03,
    max_rot_shared=0.05,
    max_rot_smooth=0.03,
    max_rot_local=0.08,
    max_scale_shared=0.05,
    max_scale_smooth=0.03,
    max_scale_local=0.08,

    max_opacity_delta=4.0,


    sat_threshold=0.8,



    lambda_balance_geo=0.005,
    lambda_balance_vis=0.001,

    target_geo_static=0.30,
    target_geo_hexplane=0.35,
    target_geo_local=0.20,
    target_geo_residual_smooth=0.15,

    target_geo_static_stage2=0.40,
    target_geo_hexplane_stage2=0.60,

    lambda_route_conf_geo=0.002,
    lambda_route_conf_vis=0.001,
    lambda_expert_diversity_geo=0.001,

    lambda_entropy_geo=0.001,
    lambda_entropy_vis=0.0005,

    lambda_geo_temp=0.003,
    lambda_geo_spatial=0.003,
    lambda_vis_sparse=0.001,
    lambda_decouple=0.01,

    lambda_sat_g1_disp=5e-4,
    lambda_sat_g2_disp=1e-4,



    lambda_mag_g1_mu=1e-4,
    lambda_mag_g2_mu=2e-5,



    lambda_raw_g1_disp=1e-4,
    lambda_raw_g2_disp=1e-4,



    warmup_iters=1000,
    enable_shared_only_iter=1000,
    enable_smooth_geo_iter=2000,
    enable_local_geo_iter=3500,
    enable_visibility_iter=4500,
    enable_sparse_routing_iter=5000,
    enable_route_stability_iter=5000,
    enable_decouple_iter=5500,
    entropy_end_iter=7000,

    use_soft_routing=True,
    use_topk=False,
    topk_geo=2,
    topk_vis=1,
    router_noise_geo=0.0,
    router_noise_vis=0.0,

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
)
