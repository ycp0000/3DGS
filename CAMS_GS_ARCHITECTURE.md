# CAMS-GS Architecture

## Goal

`tracking_type='cams_gs'` implements **Cut-Aware Motion Scaffold Gaussian Splatting** for dynamic endoscopic Gaussian Splatting.

The current design goal is to replace the old per-point heterogeneous MoE story with a more structured dynamic model that:

- separates global motion from local and cut-aware motion
- stages geometry learning before visibility/lifecycle refinement
- ties visibility, lifecycle, and appearance heads to real rendered outputs
- preserves the existing outer training/orchestration contracts in `scene/deformation.py`, `train.py`, and `scene/tracking_losses.py`

## Top-level runtime path

At runtime, CAMS-GS follows this sequence:

1. `scene/deformation.py::Deformation.forward_dynamic()` runs the backbone deformation path.
2. The CAMS head receives the backbone-updated Gaussian state:
   - `means3d`
   - `scales`
   - `rotations`
   - `opacity_logits`
   - `time_values`
   - `time_features`
   - `scene_scale`
   - `phase`
3. `CutGraphGating` predicts scaffold structure.
4. `CAMSGSTracking._build_geo_probabilities()` converts scaffold outputs into `pi_geo`.
5. `MotionDecomposition` predicts motion branches and composes them with the routing weights.
6. `VisibilityAppearanceHead` predicts visibility routing and appearance RGB modulation.
7. `GaussianLifecycleHead` predicts lifecycle persistence gating.
8. The CAMS head returns updated Gaussian state plus aux tensors.
9. `gaussian_renderer/__init__.py` uses the returned deformation aux to modulate color in the fine stage, while consuming the already gated opacity returned by the CAMS head.

## Geometry branch

### Geometry experts

`CAMSGSTracking.GEO_EXPERT_NAMES = ('global', 'local', 'cut_graph')`

The current geometry interpretation is:

- `global`: low-capacity time-only motion branch for shared scene motion
- `local`: position-aware local motion branch
- `cut_graph`: position-aware branch intended to specialize on scaffold regions separated by the cut-aware gate

### Scaffold and routing

`models/tracking/cut_graph_gating.py` predicts:

- `scaffold_logits`
- `scaffold_weights`
- `cut_gate_logits`
- `cut_gate_values`

`CAMSGSTracking._build_geo_probabilities()` turns those into:

- `global_mix`
- `local_mix`
- `cut_graph_mix`
- normalized `pi_geo`

The key point is that the current implementation no longer logs a fake 3-way route while using only 2 motion paths. The `cut_graph` branch now has its own motion output in the composed geometry update.

### Motion composition

`models/tracking/motion_decomposition.py` predicts:

- `global_delta`
- `local_delta`
- `cut_graph_delta`

and composes them as:

- `blended_global = global_mix * global_delta`
- `blended_local = local_mix * local_delta`
- `blended_cut_graph = cut_graph_mix * cut_graph_delta`
- `d_mu = blended_global + blended_local + blended_cut_graph`

The module also predicts:

- `d_rot`
- `d_scale`
- `d_opacity_logit`

The current auxiliary outputs include:

- `global_motion`
- `local_motion`
- `cut_graph_motion`
- `d_mu`
- `d_rot`
- `d_scale`
- `d_opacity_logit`

## Visibility and appearance branch

`models/tracking/cams_gs_visibility.py` implements the visibility/appearance head.

It predicts:

- `visibility_logits`
- `pi_vis`
- `visibility_alpha`
- `appearance_offsets`
- `appearance_rgb_delta`

### Visibility behavior

When visibility is enabled, `pi_vis = softmax(visibility_logits)` and:

- `visibility_alpha = pi_vis[:, :1]`

So the first visibility class is the keep/stable visibility mass used for opacity gating.

When visibility is disabled, the head falls back to:

- `pi_vis[:, 0] = 1`
- `visibility_alpha = 1`

so early geometry phases remain unaffected.

### Appearance behavior

`appearance_rgb_delta` is a bounded RGB modulation term derived from the appearance head output.

It is not just logged in aux: during rendering, it is applied to the precomputed RGB of deformed points in the fine stage.

## Lifecycle branch

`models/tracking/cams_gs_lifecycle.py` implements the lifecycle head.

It predicts:

- `lifecycle_logits`
- `lifecycle_probs`
- `lifecycle_alpha`

The class convention is now aligned across rendering and losses:

- column `0` = persistent / alive class
- column `1` = transient / off class
- `lifecycle_alpha = lifecycle_probs[:, :1]`

Outside `joint_finetune`, lifecycle is kept neutral:

- `lifecycle_probs[:, 0] = 1`
- `lifecycle_alpha = 1`

This prevents early phases from being unintentionally suppressed by lifecycle gating.

## Opacity and color integration

The current CAMS-GS implementation now affects the actual rendered state.

### Opacity integration

Inside `models/tracking/cams_gs_tracking.py`, when opacity learning is enabled:

- `opacity_scale = visibility_alpha * lifecycle_alpha`
- `opacity_logits_out = motion_state['opacity_logits'] + logit(clamp(opacity_scale, 1e-4, 1 - 1e-4))`

This means the returned opacity logits already include the visibility/lifecycle gate, with the gate clamped before the logit transform for numerical stability.

If opacity updates are disabled (`no_do=True`), CAMS-GS now preserves the original opacity path instead of silently changing opacity through late-stage gates.

### Color integration

Inside `gaussian_renderer/__init__.py`, fine-stage rendering now reads:

- `appearance_rgb_delta`

from deformation aux and applies it to the precomputed RGB of deformed points.

When `compute_cov3D_python=True`, the renderer now recomputes covariance from the deformed `scales` and `rotations` after CAMS updates, so scale/rotation changes remain visible in that path as well.

So appearance learning is now connected to actual image formation rather than only self-regularized aux tensors.

## Scheduler

`models/tracking/cams_gs_tracking.py::CAMSGSScheduler` currently defines this curriculum:

1. `global_only`
2. `graph_bootstrap`
3. `local_motion_only`
4. `motion_warmup`
5. `visibility_refine`
6. `joint_finetune`

### Phase meaning

- `global_only`
  - backbone + global motion only
- `graph_bootstrap`
  - scaffold/cut-graph learning begins
- `local_motion_only`
  - local and cut-graph geometry branches become trainable
- `motion_warmup`
  - geometry continues refining before visibility turns on
- `visibility_refine`
  - visibility + appearance active
- `joint_finetune`
  - lifecycle active in addition to the full CAMS stack

This phase structure is important because Patch C losses are phase-gated to match it.

## Loss surface

`scene/tracking_losses.py` remains the public tracking-loss entrypoint.

CAMS-GS currently plugs into it through deformation aux.

### Existing CAMS-relevant losses

- geometry/visibility usage logging
- geometry/visibility balance losses
- route confidence metrics
- adjacent-time temporal regularization
- `L_appearance_reg`
- `L_lifecycle_balance`
- `L_lifecycle_reg`

### Phase gating

The current Patch C contract is:

- `motion_warmup`
  - no appearance loss
  - no lifecycle loss
- `visibility_refine`
  - appearance loss may be active
  - lifecycle losses remain off
- `joint_finetune`
  - appearance + lifecycle losses may be active

## Train-time aux merge

`train.py` now safely merges deformation aux even when the aux dictionary contains non-tensor metadata such as:

- `tracking_phase_name`

This was required once phase-aware Patch C losses started depending on aux metadata.

## Optimizer groups

The CAMS-GS path exposes named parameter groups through `scene/deformation.py` and `scene/gaussian_model.py`.

The important groups are:

- `tracking_base_deformation`
- `tracking_base_grid`
- `tracking_time_encoder`
- `tracking_motion_global`
- `tracking_motion_local`
- `tracking_cut_graph`
- `tracking_visibility`
- `tracking_appearance`
- `tracking_lifecycle`

These are turned on/off by `TrackingPhase` rather than by changing the outer optimizer contract.

## Metadata and compatibility

CAMS-GS checkpoints currently use:

- `tracking_type='cams_gs'`
- `tracking_arch_version='cams_gs_v2'`

Restore/load checks reject incompatible or legacy metadata so the wrong architecture cannot be resumed silently.

## Current practical interpretation

The current CAMS-GS implementation should be understood as:

- original deformation backbone retained as the shared base path
- structured 3-branch geometry refinement on top of the backbone-updated state
- visibility and lifecycle converted into real opacity gating
- appearance converted into real RGB modulation in the renderer
- stage-aware optimization that delays late semantic heads until geometry has stabilized

That is the architecture the repository now actually trains and evaluates.
