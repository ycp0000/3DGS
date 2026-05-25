# EndoMoeGaussian

EndoMoeGaussian extends the original endoscopic dynamic Gaussian Splatting pipeline with a heterogeneous Mixture-of-Experts tracking module designed for deformable surgical scenes.

The current codebase supports three tracking regimes:

- `tracking_type='original'`: original deformation path.
- `tracking_type='split'`: intermediate split-head path.
- `tracking_type='heterogeneous_moe'`: final heterogeneous geometry/visibility MoE path.

The heterogeneous path is the main experimental target in this repository.

## What is implemented now

The current heterogeneous tracking stack is a displacement-first MoE with:

- 4 geometry experts: `static`, `hexplane`, `local`, `smooth`
- 2 visibility experts: `stable`, `transient`
- a staged training scheduler that activates experts progressively
- phase-aware optimizer gating
- adjacent-time temporal regularization from the current batch
- checkpoint metadata checks to prevent architecture-mismatched resume/load
- preset/config compatibility guards for legacy names such as `prune_interval`

A more detailed implementation note is in [HETEROGENEOUS_MOE_ARCHITECTURE.md](HETEROGENEOUS_MOE_ARCHITECTURE.md).

## Important scope note

The corrected heterogeneous implementation is now a **residual augmentation** over the original dynamic deformation path.

That means heterogeneous mode:

- keeps the original deformation backbone active as the baseline floor,
- applies MoE residuals to geometry translation and opacity logits,
- keeps scale and rotation driven by the original path in the current corrective patch.

So in `tracking_type='heterogeneous_moe'`:

- translation = original dynamic translation + MoE residual translation
- opacity logits = original dynamic opacity update + MoE residual opacity delta
- `d_rot = 0` and `d_scale = 0` remain true for the MoE residual branch itself

This change matters for interpretation: the main MoE experiment is no longer a lower-DoF replacement model. It is now a baseline-preserving residual expert layer, which is the fair comparison for asking whether MoE helps.

## Environment setup

Follow the base 3DGS / 4DGaussians dependency setup first.

```bash
git clone https://github.com/ycp0000/3DGS.git
cd 3DGS
git submodule update --init --recursive
conda create -n endomoe python=3.7 -y
conda activate endomoe
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

This repository has been used with:

- Python 3.7-style environment from the original codebase
- PyTorch 1.13.1 + CUDA 11.6 in the original setup

## Dataset preparation

For EndoNeRF-style endoscopic data, use the EndoNeRF directory layout expected by the existing loaders and set:

- `extra_mark='endonerf'`
- `camera_extent=10`

The repository already contains EndoNeRF presets under:

- `arguments/endonerf/`

Useful preset families include:

- baseline original tracking:
  - `arguments/endonerf/cutting_original.py`
  - `arguments/endonerf/pulling_original.py`
- full heterogeneous MoE:
  - `arguments/endonerf/cutting_disentangled_moe.py`
  - `arguments/endonerf/pulling_disentangled_moe.py`
- geo-only heterogeneous ablation:
  - `arguments/endonerf/cutting_geo_moe_only.py`
  - `arguments/endonerf/pulling_geo_moe_only.py`
- dense / sparse routing variants:
  - `arguments/endonerf/*_disentangled_moe_dense.py`
  - `arguments/endonerf/*_disentangled_moe_sparse.py`

## Recommended training workflow

Do **not** jump directly to the final MoE preset and declare success or failure from one run. Use a staged experiment ladder.

### Stage 1: Sanity-check the baseline

Run the original tracking preset first.

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_original" \
  --configs arguments/endonerf/cutting_original.py
```

Why:

- verifies data loading and scene scaling are correct
- gives you the baseline PSNR / SSIM / LPIPS reference
- tells you whether later failures are from MoE logic or from the scene/setup itself

### Stage 2: Check the geometry-only heterogeneous path

Run the geometry-only MoE ablation next.

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_geo_moe_only" \
  --configs arguments/endonerf/cutting_geo_moe_only.py
```

Why:

- isolates whether the geometry routing story is already helping
- separates geometry specialization issues from visibility routing issues

### Stage 3: Run the full heterogeneous MoE

Then run the full geometry + visibility residual MoE.

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_hetero_moe" \
  --configs arguments/endonerf/cutting_disentangled_moe.py
```

For pulling scenes, switch to the matching `pulling_*` preset.

### Stage 4: Compare dense and sparse routing variants

If the full heterogeneous run is stable, compare dense vs sparse routing.

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_hetero_moe_dense" \
  --configs arguments/endonerf/cutting_disentangled_moe_dense.py

python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_hetero_moe_sparse" \
  --configs arguments/endonerf/cutting_disentangled_moe_sparse.py
```

Why:

- tells you whether sparse routing improves specialization or just destabilizes training
- helps diagnose whether the router is benefiting from competition or suffering from premature sparsity

## Output layout

If you pass `--expname`, training writes to:

```text
./output/<expname>/
```

For example:

```text
./output/endonerf/cutting_hetero_moe/
```

This directory stores:

- `cfg_args`
- checkpoints such as `chkpnt*.pth`
- point clouds under `point_cloud/`
- render/eval outputs used later by `render.py` and `metrics.py`

## Rendering workflow

After training, render the saved model.

```bash
python render.py \
  --model_path "output/endonerf/cutting_hetero_moe" \
  --skip_train \
  --configs arguments/endonerf/cutting_disentangled_moe.py
```

Use the preset that matches the run you trained.

Recommended practice:

- render baseline and MoE runs with the same evaluation path
- keep output folders separate and named clearly
- do not compare runs that used mismatched presets or mismatched `tracking_type`

## Metric evaluation workflow

Evaluate one or more trained runs with:

```bash
python metrics.py -m \
  "output/endonerf/cutting_original" \
  "output/endonerf/cutting_geo_moe_only" \
  "output/endonerf/cutting_hetero_moe"
```

Use this to compare:

- baseline original tracking
- geo-only MoE
- full heterogeneous MoE
- dense vs sparse routing variants

## Recommended experiment ladder

Use the following order for each scene:

1. `*_original.py`
2. `*_geo_moe_only.py`
3. `*_disentangled_moe.py`
4. `*_disentangled_moe_dense.py`
5. `*_disentangled_moe_sparse.py`

After the corrective patch above, these presets are intended to share the same coarse/pruning/densification protocol so the comparison isolates architecture rather than training-budget drift.

This is the minimum useful ladder because it answers:

- does heterogeneous geometry help at all?
- does visibility routing add value on top of geometry routing?
- does sparse routing help or hurt?

## What to record for every run

For every experiment, keep a small run sheet with:

- scene name
- preset path
- output directory
- final iteration reached
- best / final PSNR
- SSIM
- LPIPS
- qualitative render notes
- any instability signs
- whether routing collapsed or diversified

For heterogeneous runs, also inspect training logs / saved summaries for:

- `usage_geo_static`
- `usage_geo_hexplane`
- `usage_geo_local`
- `usage_geo_smooth`
- `usage_vis_stable`
- `usage_vis_transient`
- `route_max_prob_geo`
- `route_max_prob_vis`
- `expert_diversity_geo`
- `L_geo_temp`

These are important because headline image metrics alone are not enough to diagnose whether the MoE is actually learning specialization.

## How to analyze results

Do not interpret results with only one question: "Did PSNR go up?"

Use three layers of analysis.

### 1. Metric layer

Compare:

- baseline vs geo-only
- geo-only vs full heterogeneous
- dense vs sparse

Questions:

- Did the full MoE beat the baseline?
- If not, did geo-only help while visibility hurt?
- Did sparse routing improve specialization but reduce image quality?
- Are gains consistent across scenes or only on one scene?

### 2. Routing-behavior layer

Questions:

- Did one geometry expert dominate nearly all points?
- Did the visibility router collapse to `stable`?
- Did the local/smooth experts activate only after their scheduled phases?
- Did route confidence become too sharp too early?
- Did balance targets look incompatible with the scene?

### 3. Failure-mode layer

Questions:

- Are renders sharper but temporally unstable?
- Are metrics flat because translation-only modeling is insufficient?
- Is the MoE over-regularized and under-moving?
- Is one expert saturating at its displacement bound?
- Does the scene simply lack enough motion diversity for the extra MoE capacity to help?

## If the MoE does not improve stably

That is a valid experimental outcome. Do **not** patch the story after one failed run without diagnosis.

Bring back:

- the exact preset used
- the scene name
- final metrics
- baseline metrics
- qualitative render observations
- any routing usage statistics you logged
- whether the failure is:
  - no improvement
  - unstable improvement
  - better metrics but worse temporal behavior
  - better qualitative motion but worse PSNR/LPIPS

Then we can reason about the next step from evidence instead of guessing.

Typical next-step diagnoses include:

- router collapse
- weak local expert capacity
- over-strong temporal or saturation regularization
- visibility routing hurting a scene that needs geometry help more than opacity help
- schedule timing mismatches
- translation-only limitation in scenes where rotation/scale changes matter

## Suggested feedback package for the next iteration

When you want another round of analysis, send back something like this:

```text
Scene: cutting
Preset: arguments/endonerf/cutting_disentangled_moe.py
Run dir: output/endonerf/cutting_hetero_moe
Baseline metrics: PSNR=?, SSIM=?, LPIPS=?
MoE metrics: PSNR=?, SSIM=?, LPIPS=?
Observed routing: geo_static=?, geo_hexplane=?, geo_local=?, geo_smooth=?
Observed visibility: stable=?, transient=?
Qualitative notes: ...
Failure suspicion: ...
```

That will make the next debugging / redesign round much faster and more grounded.

## Verification status of the current code path

The current heterogeneous integration has focused regression coverage for:

- heterogeneous scheduler behavior
- heterogeneous forward output shapes
- sparse visibility routing behavior
- adjacent-time temporal loss
- EndoNeRF preset key validity
- legacy config alias compatibility

Relevant tests:

- `tests/test_disentangled_moe_tracking.py`
- `tests/test_endonerf_presets.py`

## Notes and caveats

- `full_eval.py` is for other benchmark families and is not the main EndoNeRF workflow.
- `render.py` and `metrics.py` should be run with output paths that match the exact preset/run being compared.
- Resume/load safety now checks `tracking_type`, so mismatched checkpoints should fail loudly instead of silently loading the wrong architecture.
- Some old import names remain for compatibility, but the real implementation target is `HeterogeneousMoETracking`.

## Acknowledgement

This repository builds on ideas and code from:

- 3D Gaussian Splatting
- 4DGaussians
- k-planes
- HexPlane
- TiNeuVox

Please also see the original upstream acknowledgements in the project history.
