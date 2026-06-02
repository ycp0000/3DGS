# Figure Catalog

## Figure 1: Main comparison

**Filename**: `figures/figure-01-main-comparison.png` (also `.pdf`)

**Purpose**: Show the end-state metric degradation of both MoE variants relative to the original baseline.

**Data source**: `output/endonerf/*/results.json` for the three saved cutting runs.

**Plotted variables**:
- x-axis: method (original, geo-only MoE, hetero MoE)
- y-axis: PSNR (dB), SSIM, LPIPS
- Three subplots, one per metric

**Error bars**: None (n=1 per method).

**Caption requirements**:
- State that these are single-run end-state observations at iteration 9000.
- State that higher is better for PSNR and SSIM, lower is better for LPIPS.
- State that config drift confounds the comparison (see Figure 2).

**Key observation**:
Both MoE variants degraded substantially:
- Geo-only MoE: -5.19 PSNR, -0.036 SSIM, +0.041 LPIPS
- Hetero MoE: -4.60 PSNR, -0.030 SSIM, +0.040 LPIPS

**Interpretation checklist**:
1. Why does this figure exist? To show that both MoE variants degraded in the saved runs.
2. What exactly should the reader notice? The large PSNR gap (~5 dB) between original and both MoE variants.
3. What does that observation change? It motivates the root-cause investigation, but does not yet explain whether the degradation is due to the MoE architecture or config drift.

**Known caveats**:
- Single-run observations (no error bars).
- Config drift confounds the comparison (see Figure 2 and stats appendix).
- One dataset (cutting scene only).

---

## Figure 2: Config drift

**Filename**: `figures/figure-02-config-drift.png` (also `.pdf`)

**Purpose**: Show the relative config changes between the saved runs to explain why the comparison is confounded.

**Data source**: `output/endonerf/*/cfg_args` for the three saved cutting runs.

**Plotted variables**:
- x-axis: relative change from original (%)
- y-axis: config parameter name
- Two subplots: geo-only MoE vs original, hetero MoE vs original
- Top 12 largest relative changes per subplot

**Error bars**: None (config values are deterministic).

**Caption requirements**:
- State that these are relative changes in protocol-critical and motion-capacity settings.
- State that the hetero MoE run used drastically reduced motion capacity (`max_disp_local_ratio` cut by 95%).
- State that this config drift prevents isolating the MoE architecture effect.

**Key observation**:
The hetero MoE run used:
- 95% reduction in `max_disp_local_ratio` (0.0015 vs 0.03)
- 53% reduction in densification duration (`densify_until_iter` 7000 vs 15000)
- 70% reduction in temporal regularization (`lambda_geo_temp` 0.003 vs 0.01)

**Interpretation checklist**:
1. Why does this figure exist? To show that the saved runs differ in protocol-critical and motion-capacity settings, not just in the MoE architecture.
2. What exactly should the reader notice? The 95% reduction in `max_disp_local_ratio` for the hetero MoE run.
3. What does that observation change? It explains that the degradation cannot be attributed to the MoE architecture alone, because the hetero MoE run used severely constrained motion capacity.

**Known caveats**:
- Relative change is undefined for boolean and NaN values (filtered out).
- Only shows top 12 changes per subplot (full table in `evidence-table.md`).
- Does not show which config changes are protocol-critical vs scientifically interesting (reader must infer from parameter names).

---

## Additional artifacts

**`evidence-table.md`**: Full numeric table with exact metrics and all protocol-critical config settings for the three saved runs. Use this for exact values and detailed config comparison.
