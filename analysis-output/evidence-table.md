# Evidence Table

## End-state metrics

Run | PSNR | SSIM | LPIPS
--- | --- | --- | ---
`cutting_original` | 37.104351 | 0.965340 | 0.083079
`cutting_geo_moe_only` | 31.916224 | 0.928867 | 0.124543
`cutting_hetero_moe` | 32.502380 | 0.934965 | 0.123414

All values are evaluated at the saved checkpoint `ours_9000`.

## Metric deltas vs original baseline

Run | Δ PSNR | Δ SSIM | Δ LPIPS
--- | --- | --- | ---
`cutting_geo_moe_only` | -5.188128 | -0.036473 | +0.041464
`cutting_hetero_moe` | -4.601971 | -0.030375 | +0.040335

## Protocol-critical and motion-capacity settings

Setting | `cutting_original` | `cutting_geo_moe_only` | `cutting_hetero_moe`
--- | --- | --- | ---
`tracking_type` | 'original' | 'heterogeneous_moe' | 'heterogeneous_moe'
`iterations` | 9000 | 9000 | 9000
`coarse_iterations` | 1000 | 2000 | 2000
`pruning_interval` | 3000 | 1000 | 1000
`densify_until_iter` | 15000 | 15000 | 7000
`entropy_end_iter` | 2500 | 2500 | 7000
`enable_visibility` | True | False | True
`enable_visibility_iter` | 2000 | 2000 | 4500
`lambda_geo_spatial` | 0.01 | 0.01 | 0.003
`lambda_geo_temp` | 0.01 | 0.01 | 0.003
`max_disp_hexplane_ratio` | 0.01 | 0.01 | 0.01
`max_disp_local_ratio` | 0.03 | 0.03 | 0.0015
`max_disp_smooth_ratio` | 0.01 | 0.01 | 0.0005
`max_rot_local` | 0.1 | 0.1 | 0.08
`max_rot_shared` | 0.05 | 0.05 | 0.05
`max_rot_smooth` | 0.05 | 0.05 | 0.03
`max_scale_local` | 0.1 | 0.1 | 0.08
`max_scale_shared` | 0.05 | 0.05 | 0.05
`max_scale_smooth` | 0.05 | 0.05 | 0.03
`target_geo_hexplane` | nan | nan | 0.35
`target_geo_local` | 0.2 | 0.2 | 0.2
`target_geo_residual_smooth` | nan | nan | 0.15
`target_vis_stable` | 0.85 | 0.85 | 0.85
`target_vis_transient` | 0.15 | 0.15 | 0.15
