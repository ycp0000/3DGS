# CAMS-GS Result Analysis

## Comparison: Original vs CAMS-GS (cutting scene, iteration 9000)

| Metric | Original | CAMS-GS | Delta | Interpretation |
|--------|----------|---------|-------|----------------|
| PSNR (dB) | 37.093 | 37.055 | -0.038 | Negligible degradation |
| SSIM | 0.9655 | 0.9651 | -0.0004 | Negligible degradation |
| LPIPS | 0.0829 | 0.0824 | -0.0005 | Negligible improvement |

## Statistical interpretation

**Sample size**: n=1 per method (single run, no repeated seeds).

**Effect magnitude**:
- PSNR delta: **0.038 dB** — well below the typical perceptual threshold (~0.5 dB).
- SSIM delta: **0.0004** — negligible (typical meaningful difference is ~0.01).
- LPIPS delta: **0.0005** — negligible (typical meaningful difference is ~0.01).

**Conclusion**: The observed differences are **not meaningful**. CAMS-GS performed **equivalently** to the original baseline within measurement noise.

## What this result means

### Claim status: **NEUTRAL / INCONCLUSIVE**

The CAMS-GS run did **not degrade** relative to the original baseline, which is a positive signal given that:
1. The old heterogeneous MoE runs degraded by **-5.19 PSNR** (geo-only) and **-4.60 PSNR** (hetero).
2. CAMS-GS avoided that catastrophic regression.

However, CAMS-GS also did **not improve** over the original baseline. The deltas are too small to claim any gain.

### Why CAMS-GS didn't improve

Several hypotheses, ordered by likelihood:

#### 1. **CAMS-GS is equivalent to the original baseline in capacity**
- The original baseline already uses a strong k-plane deformation field.
- CAMS-GS adds structured motion decomposition, but if the scene's motion is already well-captured by the baseline, the added structure provides no gain.
- **Test**: Run CAMS-GS on a scene with more complex motion (pulling scene, or a scene with tool occlusion).

#### 2. **CAMS-GS motion capacity is under-tuned**
- The current CAMS-GS preset uses conservative motion bounds inherited from the corrective hetero MoE preset.
- If `max_disp_local_ratio`, `max_rot_*`, `max_scale_*` are too small, CAMS-GS cannot express useful residual motion.
- **Test**: Run an ablation with increased motion capacity (match the original baseline's motion bounds exactly).

#### 3. **CAMS-GS scheduler is misaligned with the 9000-iteration budget**
- The CAMS-GS scheduler has 6 phases: `global_only`, `graph_bootstrap`, `local_motion_only`, `motion_warmup`, `visibility_refine`, `joint_finetune`.
- If the phase transitions are poorly timed for a 9000-iteration run, the model may not have enough iterations to refine the late-stage heads (visibility, lifecycle, appearance).
- **Test**: Inspect the phase boundaries in the CAMS-GS preset and verify they align with the 9000-iteration budget.

#### 4. **CAMS-GS visibility/lifecycle/appearance heads are not helping**
- These heads were designed to address the old MoE's disconnected visibility/lifecycle problem.
- If the original baseline already handles visibility/appearance well through the k-plane field, the added heads provide no gain.
- **Test**: Run an ablation with visibility/lifecycle disabled (`enable_visibility=False`) to isolate the geometry-only CAMS-GS effect.

#### 5. **The cutting scene is too easy**
- The original baseline achieves **PSNR 37.09**, which is already very high for endoscopic reconstruction.
- If the scene has limited motion or is well-conditioned, there may be no room for improvement.
- **Test**: Run CAMS-GS on the pulling scene, which typically has more complex motion.

## Recommended next steps

### Immediate diagnostic experiments (in priority order)

1. **Verify config parity**: Confirm that the CAMS-GS run used the same protocol-critical settings as the original baseline:
   - `coarse_iterations`
   - `pruning_interval`
   - `densify_until_iter`
   - `iterations`
   - `position_lr_max_steps`

2. **Inspect CAMS-GS routing logs** (if available):
   - Check `usage_geo_global`, `usage_geo_local`, `usage_geo_cut_graph` to see if the geometry routing collapsed or balanced.
   - Check `usage_vis_stable`, `usage_vis_transient` to see if visibility routing activated.
   - Check `mean_norm_d_mu`, `mean_norm_d_rot`, `mean_norm_d_scale` to see if CAMS-GS motion magnitude is non-trivial.

3. **Run motion-capacity ablation**: Increase CAMS-GS motion bounds to match the original baseline exactly:
   - `max_disp_local_ratio`: 0.03 (not 0.0015)
   - `max_rot_local`: 0.1 (not 0.08)
   - `max_scale_local`: 0.1 (not 0.08)

4. **Run pulling scene**: Test CAMS-GS on a scene with more complex motion to see if the structured decomposition helps there.

5. **Run geometry-only CAMS-GS ablation**: Disable visibility/lifecycle (`enable_visibility=False`) to isolate the geometry effect.

### Longer-term experiments

- **Repeated seeds**: Run 3 seeds per method to estimate variance and check if the observed deltas are stable.
- **Training curves**: Save loss curves and routing statistics over time to diagnose whether CAMS-GS converges differently from the original baseline.
- **Ablation study**: Systematically ablate CAMS-GS components (cut-graph branch, visibility head, lifecycle head, appearance head) to identify which components contribute to the final result.

## Decision

**Do not claim improvement**. The CAMS-GS result is **equivalent** to the original baseline within measurement noise.

**Do not claim failure**. CAMS-GS avoided the catastrophic degradation observed in the old heterogeneous MoE runs, which suggests the architectural redesign was sound.

**Next action**: Run the immediate diagnostic experiments above to understand why CAMS-GS did not improve, then decide whether to:
- Tune CAMS-GS hyperparameters (motion capacity, scheduler timing),
- Test on a harder scene (pulling),
- Accept equivalence as the result and move to ablation studies for publication.
