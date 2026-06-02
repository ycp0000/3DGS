# Stats Appendix

## Descriptive statistics

### End-state metrics (iteration 9000)

| Run | PSNR (dB) | SSIM | LPIPS |
|-----|-----------|------|-------|
| `cutting_original` | 37.104 | 0.965 | 0.083 |
| `cutting_geo_moe_only` | 31.916 | 0.929 | 0.125 |
| `cutting_hetero_moe` | 32.502 | 0.935 | 0.123 |

### Deltas vs original baseline

| Run | Δ PSNR | Δ SSIM | Δ LPIPS |
|-----|--------|--------|---------|
| `cutting_geo_moe_only` | -5.188 | -0.036 | +0.041 |
| `cutting_hetero_moe` | -4.602 | -0.030 | +0.040 |

All three metrics moved in the degradation direction for both MoE variants.

## Sample size and unit of analysis

- **n = 1** for each method (single run per config).
- **Unit of analysis**: end-state checkpoint at iteration 9000.
- **No repeated seeds**: cannot compute standard deviation or confidence intervals.
- **No training curves**: cannot assess optimization stability or convergence behavior.

## Inferential tests

**Blocked**: inferential statistics require repeated measures or multiple seeds. With n=1 per method, no significance test is valid.

## Effect sizes

**Blocked**: effect size estimation (Cohen's d, Hedges' g) requires variance estimates, which are unavailable with n=1.

## Assumptions checked

Not applicable: no parametric or non-parametric tests were run due to insufficient sample size.

## Multiple comparison corrections

Not applicable: no hypothesis tests were performed.

## Explicit blockers and limitations

1. **Single-run observations**: All conclusions are descriptive only. No inferential claims are supported.
2. **Config confounding**: The saved runs differ in protocol-critical settings (`coarse_iterations`, `pruning_interval`, `densify_until_iter`) and motion-capacity settings (`max_disp_local_ratio`, `max_rot_*`, `max_scale_*`), preventing isolation of the MoE architecture effect.
3. **No training logs**: Cannot diagnose whether degradation occurred early (optimization failure) or late (overfitting / underfitting).
4. **No routing logs**: Cannot verify whether MoE routing collapsed, balanced, or specialized as intended.
5. **One dataset**: Cutting scene only. Generalization to pulling scene or other EndoNeRF scenes is unknown.

## What can be concluded

With n=1 and confounded configs, the only defensible conclusions are:

- **Observation**: Both MoE variants degraded relative to the original baseline in the saved runs.
- **Observation**: The hetero MoE run used drastically reduced motion capacity (`max_disp_local_ratio` 0.0015 vs 0.03 in the other runs).
- **Observation**: The hetero MoE run used earlier densification cutoff (`densify_until_iter` 7000 vs 15000 in the other runs).

## What cannot be concluded

- Whether the MoE architecture itself caused the degradation (confounded by config drift).
- Whether the degradation is stable across seeds (no repeated runs).
- Whether the degradation generalizes to other scenes (one dataset).
- Whether the degradation is due to optimization failure, capacity mismatch, or routing collapse (no training logs or routing logs).

## Recommendation for future experiments

To isolate the MoE effect, run:
- **Matched protocol**: same `coarse_iterations`, `pruning_interval`, `densify_until_iter` across all methods.
- **Matched motion capacity**: same `max_disp_*`, `max_rot_*`, `max_scale_*` across all methods.
- **Repeated seeds**: at least 3 seeds per method to estimate variance.
- **Training logs**: save loss curves, routing statistics, and motion magnitude over time.
- **Multiple scenes**: cutting + pulling at minimum.

Alternatively, **use the new CAMS-GS path** instead of re-running the old heterogeneous MoE, since CAMS-GS was designed to address the architectural issues identified during the root-cause investigation.
