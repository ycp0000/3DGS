# Analysis Report: EndoNeRF Cutting Runs

## Analysis question

What caused the heterogeneous MoE degradation relative to the original baseline in the saved EndoNeRF cutting runs?

## Key findings

### Observation 1: Both MoE variants degraded substantially

The original baseline achieved **PSNR 37.10 / SSIM 0.965 / LPIPS 0.083**.

Both MoE variants degraded:
- Geo-only MoE: **PSNR 31.92 / SSIM 0.929 / LPIPS 0.125** (Δ PSNR -5.19, Δ SSIM -0.036, Δ LPIPS +0.041)
- Hetero MoE: **PSNR 32.50 / SSIM 0.935 / LPIPS 0.123** (Δ PSNR -4.60, Δ SSIM -0.030, Δ LPIPS +0.040)

The hetero MoE run recovered ~0.6 PSNR relative to geo-only MoE, but both remained far below the original baseline.

### Observation 2: Config drift confounds the comparison

The saved runs are **not** fair method-only comparisons. Major config differences include:

**Protocol-critical drift:**
- `coarse_iterations`: original 1000 → geo-only 2000 → hetero 2000
- `pruning_interval`: original 3000 → geo-only 1000 → hetero 1000
- `densify_until_iter`: original 15000 → geo-only 15000 → hetero 7000

**Motion-capacity drift:**
- `max_disp_local_ratio`: original 0.03 → geo-only 0.03 → hetero 0.0015 (95% reduction)
- `max_rot_local`: original 0.1 → geo-only 0.1 → hetero 0.08
- `max_scale_local`: original 0.1 → geo-only 0.1 → hetero 0.08

**Regularization drift:**
- `lambda_geo_spatial`: original 0.01 → geo-only 0.01 → hetero 0.003
- `lambda_geo_temp`: original 0.01 → geo-only 0.01 → hetero 0.003

The hetero MoE run used **drastically reduced motion capacity** (`max_disp_local_ratio` cut by 95%) and **weaker temporal regularization** relative to the other two runs.

### Interpretation

The degradation is **not attributable to the MoE architecture alone** because:

1. The geo-only MoE run already degraded by -5.19 PSNR despite using the same protocol-critical settings as the original baseline.
2. The hetero MoE run used severely constrained motion capacity (`max_disp_local_ratio` 0.0015 vs 0.03), which likely prevented the MoE branch from expressing useful residual motion.
3. The hetero MoE run also used much earlier densification cutoff (`densify_until_iter` 7000 vs 15000), which may have frozen geometry prematurely.

The most defensible interpretation is that **config drift dominated the observed degradation**, not the MoE design itself.

### Constraint

This analysis is bounded by:
- **Single-run end-state observations only**: no repeated seeds, no training curves, no routing logs.
- **Confounded comparison**: config drift prevents isolating the MoE effect.
- **One dataset**: cutting scene only; pulling scene not analyzed.
- **No ablation**: cannot separate motion-capacity effect from MoE-architecture effect.

## Decision

The saved runs **do not support a conclusion** about whether the heterogeneous MoE architecture is fundamentally flawed.

The next action should be:
1. **Run fair-comparison experiments** with matched protocol settings (same `coarse_iterations`, `pruning_interval`, `densify_until_iter`, `max_disp_*`, `lambda_geo_*`) across original / geo-only MoE / hetero MoE.
2. **Use the new CAMS-GS path** instead of trying to fix the old heterogeneous MoE, since CAMS-GS was designed to address the architectural issues identified during the root-cause investigation.
3. **Treat the saved runs as evidence of config sensitivity**, not as evidence of MoE failure.

## What changed in the experimental understanding

Before this analysis, the degradation was attributed to the MoE architecture.

After this analysis, the degradation is attributed to **config drift** (especially motion-capacity reduction and protocol changes), with the MoE architecture effect remaining **unresolved** due to confounding.

The CAMS-GS redesign was motivated by the correct observation that the old MoE path had architectural issues (fake 3-way routing, disconnected visibility/lifecycle), but the saved runs do not provide clean evidence that those issues caused the metric degradation.
