# EndoMoeGaussian

EndoMoeGaussian now centers on **CAMS-GS**: a Cut-Aware Motion Scaffold Gaussian Splatting path for dynamic endoscopic 3D Gaussian Splatting.

The repository still contains earlier tracking paths for comparison, but the current forward-looking experimental path is:

- `tracking_type='cams_gs'`

**Latest Update (plan.md Strict Compliance)**: The implementation now strictly conforms to the original `plan.md` specification:

- **View-dependent visibility routing**: visibility router receives view direction, camera depth, and screen projection coordinates from the camera, instead of only spatial/temporal features
- **Geometry-visibility decoupling**: visibility router no longer depends on `scaffold_weights` or `cut_gate_values` from the geometry routing path
- **kNN spatial smoothness loss** (`L_geo_spatial`): penalizes motion discrepancies between spatially nearby Gaussians using k-nearest-neighbor graph; activated via `--lambda_geo_spatial 0.01`
- **Camera parameter propagation**: camera information flows from `gaussian_renderer` through `scene/deformation.py` to `models/tracking/cams_gs_tracking.py` and into the visibility module

Supported tracking modes in the codebase are:

- `tracking_type='original'`: original deformation path
- `tracking_type='split'`: split-head intermediate path
- `tracking_type='heterogeneous_moe'`: older residual heterogeneous MoE path
- `tracking_type='cams_gs'`: current Cut-Aware Motion Scaffold Gaussian Splatting path

A detailed implementation note for the current CAMS-GS path is in [CAMS_GS_ARCHITECTURE.md](CAMS_GS_ARCHITECTURE.md).

## What CAMS-GS implements now

The current CAMS-GS path replaces the old per-Gaussian MoE story with a structured tracking design built around:

- a **staged curriculum** from global motion to joint refinement
- a **cut-aware scaffold gate** over `global / local / cut_graph` geometry experts
- a **real 3-branch geometry composition** for translation updates
- a **visibility / appearance head** with view-dependent features (view direction, camera depth, screen projection) that affects opacity and rendered color
- a **lifecycle head** that affects opacity persistence in late training
- **kNN spatial smoothness regularization** (`L_geo_spatial`) to encourage coherent motion among nearby Gaussians
- phase-aware optimizer gating through named tracking parameter groups
- checkpoint metadata validation for architecture-safe restore/load
- adjacent-time temporal regularization from the current batch

The current CAMS-GS implementation is not just an aux-logging branch: the geometry routing, visibility gating, lifecycle gating, and appearance modulation now affect the actual forward/render path.

## High-level CAMS-GS data flow

At a high level, `tracking_type='cams_gs'` runs as follows:

1. `scene/deformation.py` computes the original backbone dynamic deformation state.
2. The CAMS head receives the **backbone-updated** Gaussian state, time features, and camera parameters.
3. `CutGraphGating` predicts scaffold weights and cut-aware gate values.
4. `CAMSGSTracking` converts those into a 3-way geometry routing distribution `pi_geo` over:
   - `global`
   - `local`
   - `cut_graph`
5. `MotionDecomposition` predicts three bounded motion branches and composes them with `pi_geo`.
6. `VisibilityAppearanceHead` predicts:
   - `pi_vis`
   - `visibility_alpha`
   - `appearance_rgb_delta`
   - using **view-dependent features**: view direction, camera depth, screen projection
   - without direct dependence on geometry routing logits/gates
7. `GaussianLifecycleHead` predicts:
   - `lifecycle_probs`
   - `lifecycle_alpha`
8. The CAMS head returns updated geometry / scale / rotation / opacity logits and aux statistics.
9. `gaussian_renderer/__init__.py` uses CAMS outputs during rendering:
   - opacity is already gated inside `models/tracking/cams_gs_tracking.py` through visibility and lifecycle
   - precomputed RGB is modulated by `appearance_rgb_delta`

## Current training curriculum

`models/tracking/cams_gs_tracking.py` defines the staged curriculum through `CAMSGSScheduler`.

The current schedule is:

1. `global_only`
   - train the backbone time path and global motion branch
   - visibility off
2. `graph_bootstrap`
   - activate cut-graph scaffold learning
   - visibility off
3. `local_motion_only`
   - activate local and cut-graph motion refinement
   - visibility off
4. `motion_warmup`
   - continue motion refinement before visibility is enabled
   - visibility off
5. `visibility_refine`
   - enable visibility and appearance learning
   - lifecycle still off
6. `joint_finetune`
   - enable lifecycle together with full CAMS-GS refinement

This schedule matters because it prevents early appearance/lifecycle noise from dominating before geometry routing has stabilized.

## Current EndoNeRF presets

The current CAMS-GS EndoNeRF presets are:

- `arguments/endonerf/cutting_cams_gs.py`
- `arguments/endonerf/pulling_cams_gs.py`

These currently use:

- `tracking_type='cams_gs'`
- `iterations=9000`
- `coarse_iterations=1000`
- `position_lr_max_steps=9000`
- `pruning_interval=3000`
- `camera_extent=10`
- EndoNeRF-style k-plane settings

## Environment setup

Follow the original dependency stack first.

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

This project has been used with the original 3DGS / 4DGaussians-style environment, including PyTorch 1.13.1 + CUDA 11.6 in the earlier setup.

## Dataset setup

For EndoNeRF-style scenes, keep the repository’s expected dataset layout and use:

- `extra_mark='endonerf'`
- `camera_extent=10`

The EndoNeRF presets in `arguments/endonerf/` already set these values.

## Recommended CAMS-GS workflow

Do not judge the method from one run without comparing it against the existing baselines.

### 1. Run the original baseline

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_original" \
  --configs arguments/endonerf/cutting_original.py
```

Why:

- validates the scene/data setup
- gives the baseline PSNR / SSIM / LPIPS reference
- separates CAMS issues from dataset/setup issues

### 2. Run CAMS-GS

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_cams_gs" \
  --configs arguments/endonerf/cutting_cams_gs.py
```

For pulling scenes, switch to:

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/pulling_cams_gs" \
  --configs arguments/endonerf/pulling_cams_gs.py
```

### 3. Run CAMS-GS with kNN spatial smoothness

```bash
python train.py \
  -s <ENDONERF_SCENE_PATH> \
  --expname "endonerf/cutting_cams_gs_spatial" \
  --configs arguments/endonerf/cutting_cams_gs.py \
  --lambda_geo_spatial 0.01
```

Use this when you want the full plan.md-compliant version with spatial motion coherence.

### 4. Render the trained model

```bash
python render.py \
  --model_path "output/endonerf/cutting_cams_gs" \
  --skip_train \
  --configs arguments/endonerf/cutting_cams_gs.py
```

### 5. Evaluate metrics

```bash
python metrics.py -m \
  "output/endonerf/cutting_original" \
  "output/endonerf/cutting_cams_gs"
```

## Output layout

If you pass `--expname`, outputs are written under:

```text
./output/<expname>/
```

For example:

```text
./output/endonerf/cutting_cams_gs/
```

Typical contents include:

- `cfg_args`
- checkpoints such as `chkpnt*.pth`
- point clouds under `point_cloud/`
- render/eval outputs used by `render.py` and `metrics.py`

## What to inspect during CAMS-GS runs

Do not only look at final PSNR.

For CAMS-GS runs, inspect:

- geometry routing:
  - `usage_geo_global`
  - `usage_geo_local`
  - `usage_geo_cut_graph`
  - `route_max_prob_geo`
  - `route_margin_geo`
- visibility routing:
  - `usage_vis_stable`
  - `usage_vis_transient`
  - `route_max_prob_vis`
  - `route_margin_vis`
- motion magnitude:
  - `mean_norm_d_mu`
  - `mean_norm_d_rot`
  - `mean_norm_d_scale`
  - `mean_abs_d_opacity`
- temporal regularization:
  - `L_geo_temp`
  - `temporal_pair_count`
- Patch C late-stage signals:
  - `L_appearance_reg`
  - `L_lifecycle_balance`
  - `L_lifecycle_reg`

These are needed to diagnose whether CAMS-GS is improving because of better motion structure, or simply shifting capacity around.

## How to analyze CAMS-GS results

Use three layers of analysis.

### 1. Metric layer

Compare:

- original vs CAMS-GS
- cutting vs pulling generality

Questions:

- Does CAMS-GS improve PSNR / SSIM / LPIPS consistently?
- Are gains scene-specific or stable across scenes?
- Are metrics improving together, or is one metric trading against temporal plausibility?

### 2. Routing-and-stage layer

Questions:

- Does the geometry routing collapse to one branch?
- Does `cut_graph` activate meaningfully or remain unused?
- Does visibility stay near `stable`, or does `transient` activate in dynamic regions?
- Do the late-stage lifecycle losses become active only during `joint_finetune`?

### 3. Failure-mode layer

Questions:

- Are renders sharper but less temporally coherent?
- Is the cut-graph branch active but not helping metrics?
- Is appearance modulation helping specular/transient structure or just adding noise?
- Is lifecycle gating over-suppressing opacity in hard frames?
- Does the scene need stronger scale/rotation modeling than the current CAMS schedule provides?

## If CAMS-GS does not improve stably

That is still a useful result.

Bring back:

- scene name
- preset path
- output directory
- baseline metrics
- CAMS-GS metrics
- routing usage statistics
- qualitative observations
- whether the failure looks like:
  - router collapse
  - ineffective cut-graph branch
  - unstable visibility/lifecycle gating
  - over-regularized motion
  - improved local appearance but worse global reconstruction

That evidence is enough to decide whether the next step should be:

- scheduler retiming
- stronger cut-graph motion capacity
- weaker lifecycle suppression
- different appearance modulation bounds
- better scale/rotation modeling

## Suggested feedback package

When you want the next debugging or redesign round, send back something like:

```text
Scene: cutting
Preset: arguments/endonerf/cutting_cams_gs.py
Run dir: output/endonerf/cutting_cams_gs
Baseline metrics: PSNR=?, SSIM=?, LPIPS=?
CAMS-GS metrics: PSNR=?, SSIM=?, LPIPS=?
Geometry routing: global=?, local=?, cut_graph=?
Visibility routing: stable=?, transient=?
Late-stage lifecycle stats: persistent=?, transient=?
Qualitative notes: ...
Failure suspicion: ...
```

## Verification status

Current focused regression coverage includes:

- CAMS-GS scheduler behavior
- CAMS-GS parameter-group exposure
- checkpoint metadata compatibility
- train-time aux merging
- cut-graph route affecting geometry output
- visibility / lifecycle / appearance control exposure
- phase-gated Patch C losses
- EndoNeRF preset key validity

Relevant tests:

- `tests/test_disentangled_moe_tracking.py`
- `tests/test_endonerf_presets.py`

## Notes and caveats

- `full_eval.py` is not the main EndoNeRF workflow here.
- `render.py` and `metrics.py` should always be run with the exact preset that matches the trained checkpoint.
- `tracking_type` metadata checks are active, so incompatible checkpoints should fail loudly.
- The older heterogeneous MoE path remains in the repository for comparison, but it is no longer the main architectural story.

## Acknowledgement

This repository builds on ideas and code from:

- 3D Gaussian Splatting
- 4DGaussians
- k-planes
- HexPlane
- TiNeuVox

Please also see the upstream acknowledgements in the original project history.
