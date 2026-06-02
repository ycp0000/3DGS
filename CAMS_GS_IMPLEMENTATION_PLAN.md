# CAMS-GS Implementation Plan

## Objective

Replace the current `tracking_type='heterogeneous_moe'` geometry-routing design with a new structured dynamic module:

**CAMS-GS = Cut-Aware Motion Scaffold Gaussian Splatting**

The redesign is meant to fix the failure mode exposed by the saved EndoNeRF runs: broad geometry degradation begins already in the geometry-only MoE variant, which indicates that the main problem is not visibility modeling but the dynamic geometry representation itself.

The core change is to stop treating motion assignment as a per-Gaussian expert-routing problem and instead model deformable tissue dynamics through a sparse motion scaffold with explicit global/local decomposition, cut-aware connectivity, Gaussian lifecycle control, and decoupled visibility/appearance refinement.

## Why This Replaces the Current MoE Design

The current heterogeneous MoE path is weak in exactly the place where the experiments failed:

- geometry is decided per Gaussian through routing rather than through a shared motion structure,
- motion specialization is learned indirectly via router pressure instead of explicit tissue motion constraints,
- newly exposed or disappearing structures must be forced into coordinate deformation and opacity routing,
- topology changes caused by pulling/cutting are not represented as connectivity changes.

CAMS-GS addresses those points directly:

- the scaffold represents shared motion carriers,
- cut-awareness changes edge influence rather than merely penalizing outputs,
- local motion is expressed relative to scaffold motion,
- lifecycle handles appearance/disappearance without overloading the deformation field,
- visibility stays decoupled from geometry.

## Proposed Public Surface

### New tracking type

Add a new tracking mode:

- `tracking_type='cams_gs'`

Keep the existing modes unchanged:

- `original`
- `split`
- `heterogeneous_moe`

This keeps the current repo usable for baseline, prior MoE ablations, and the new CAMS-GS path.

### New architecture version

Add a new checkpoint architecture version string:

- `cams_gs_v1`

This should be emitted through `get_tracking_arch_version()` and validated in the same way current heterogeneous metadata is validated.

## File-Level Plan

## 1. Keep `scene/deformation.py` as the orchestration layer

### Role after refactor

`scene/deformation.py` should remain responsible for:

- time encoding hookup,
- original dynamic backbone query,
- switching on `tracking_type`,
- delegating to the selected tracking head,
- exposing aux outputs,
- exposing parameter groups,
- exposing phase state,
- exposing checkpoint architecture version.

### Required changes

- Extend the tracking mode normalization to accept `cams_gs`.
- Add `self.cams_head` and `self.scheduler` wiring for the new mode.
- Keep `_forward_original(...)` intact as a reusable baseline/global-motion feature source.
- In `forward_dynamic(...)`, create a dedicated `cams_gs` branch.
- Continue returning `(pts, scales, rotations, opacity)` to avoid destabilizing the renderer and outer training loop.

### Important design choice

CAMS-GS should **not** be implemented as a thin residual wrapper around the old MoE head. It should be a new tracking path with its own structure. The legacy backbone may still be reused as a feature provider or initialization prior, but the CAMS-GS motion decision must come from the scaffold decomposition rather than router composition.

## 2. Add new tracking modules under `models/tracking/`

Create a new top-level file:

- `models/tracking/cams_gs_tracking.py`

This should define the main tracking head:

- `CAMSGSTracking`

That file should orchestrate the following submodules.

### 2.1 `models/tracking/motion_decomposition.py`

Classes:

- `GlobalMotionField`
- `LocalMotionField`
- `MotionComposer`

Responsibilities:

- produce a low-frequency scene-wide motion prior from canonical position and time features,
- produce scaffold-conditioned local residual motion,
- combine global and local motion into final translation, scale, and rotation deltas.

Recommended formulation:

- `global motion`: low-frequency, smooth, wide-support tissue motion,
- `local motion`: scaffold-node-conditioned residual that handles instrument-contact and local tissue deformation,
- `final motion = global + weighted local + optional anchor residual`.

This avoids forcing all dynamics into either a shared field or pure local experts.

### 2.2 `models/tracking/cut_graph_gating.py`

Classes:

- `MotionScaffoldBuilder`
- `CutAwareGraphGating`
- `ScaffoldInfluenceProjector`

Responsibilities:

- define or update scaffold nodes from canonical Gaussian support,
- infer scaffold-to-Gaussian influence weights,
- infer edge attenuation / gating that weakens motion transfer across likely cut boundaries,
- expose diagnostics such as edge utilization, gate entropy, and disconnected-component counts.

Important constraint:

Cut-awareness must affect **connectivity or influence propagation**, not just a scalar loss.

### 2.3 `models/tracking/gaussian_lifecycle.py`

Classes:

- `GaussianLifecycleHead`
- `LifecycleState`

Responsibilities:

- score whether a Gaussian remains persistent, should be deactivated, or should be marked as newly exposed/transient,
- produce bounded lifecycle logits or masks for training-time supervision/regularization,
- emit aux metrics for activation ratios and churn.

Scope limit for first implementation:

The first CAMS-GS patch should stop short of changing the global densify/prune engine. Instead, lifecycle should initially act through soft state variables and loss terms, then be wired to pruning/spawn logic in a second stage if the first stage behaves well.

### 2.4 `models/tracking/appearance_visibility.py`

Classes:

- `VisibilityTransitionHead`
- `AppearanceResidualHead`

Responsibilities:

- model transience/visibility separately from geometry motion,
- condition visibility on composed motion and lifecycle state,
- keep opacity/appearance corrections bounded and diagnostically visible.

This preserves the good intuition from the previous redesign: geometry and visibility should remain decoupled.

### 2.5 `models/tracking/tracking_phase.py`

Classes:

- `TrackingPhase`
- `CAMSGSScheduler`

Responsibilities:

- define phase trainability,
- define per-group LR scaling,
- define which CAMS-GS submodules are active at each training stage.

This can either live in its own file or replace/generalize the existing `TrackingPhase`/`HeterogeneousMoEScheduler` definitions if that refactor remains clean.

## 3. Update `models/tracking/__init__.py`

Export:

- `CAMSGSTracking`
- `CAMSGSScheduler`

Do not remove:

- `HeterogeneousMoETracking`
- `HeterogeneousMoEScheduler`
- `SplitTrackingHead`

Reason:

The repo still needs the earlier architectures for baselines and ablations.

## 4. Restructure tracking-loss logic

### Keep the outer entrypoint

Retain:

- `scene/tracking_losses.py::compute_tracking_losses(...)`

This keeps `train.py` stable.

### Split CAMS-specific logic into helper files

Recommended new files:

- `scene/tracking_losses_motion.py`
- `scene/tracking_losses_graph.py`
- `scene/tracking_losses_lifecycle.py`
- `scene/tracking_losses_visibility.py`

Then let `scene/tracking_losses.py` remain the aggregation layer.

### New loss families

#### Motion decomposition losses

- global smoothness loss,
- local residual energy loss,
- scaffold-relative motion consistency,
- temporal velocity regularization on composed motion.

#### Cut-graph losses

- edge sparsity / entropy regularization,
- motion discontinuity encouragement where cut gates open,
- within-component motion coherence.

#### Lifecycle losses

- persistence regularization for stable Gaussians,
- bounded transition loss for transient/revealed structures,
- churn penalty to prevent collapse into aggressive activate/deactivate oscillation.

#### Visibility / appearance losses

- transient visibility balance,
- opacity sparsity for non-transient states,
- optional decoupling penalty between geometry magnitude and opacity change.

## 5. Preserve optimizer ownership in `scene/gaussian_model.py`

`scene/gaussian_model.py` should remain the owner of optimizer construction and LR scheduling.

### Required parameter-group boundaries

CAMS-GS should expose these named groups:

- `tracking_time_encoder`
- `tracking_base_grid`
- `tracking_base_deformation`
- `tracking_motion_global`
- `tracking_motion_local`
- `tracking_cut_graph`
- `tracking_visibility`
- `tracking_appearance`
- `tracking_lifecycle`

### Scheduler constraint

Do not add new LR schedule families.

Keep only:

- `grid`
- `deformation`
- existing `xyz` / `none`

Mapping:

- `tracking_base_grid` -> `grid`
- every other CAMS-GS tracking group -> `deformation`

Trainability should be controlled through:

- `phase.is_group_trainable(name)`
- `phase.lr_scale_for_group(name)`

That preserves the current `GaussianModel.update_learning_rate(...)` contract.

## 6. Add CAMS-GS metadata safety to checkpoint/load paths

Update both save/restore routes in `scene/gaussian_model.py`:

- `capture()` / `restore()`
- `save_deformation()` / `load_model()`

Required behavior:

- save `tracking_type='cams_gs'` and `tracking_arch_version='cams_gs_v1'`,
- reject mismatched CAMS-GS vs non-CAMS-GS checkpoints,
- reject missing or stale CAMS-GS metadata when strict compatibility matters.

## 7. Extend the argument surface

## `arguments/__init__.py`

Add a new CAMS-GS argument block rather than overloading the old MoE-only names.

### Suggested CAMS-GS arguments

#### Scaffold structure

- `scaffold_node_count`
- `scaffold_knn_k`
- `scaffold_update_interval`
- `scaffold_feature_dim`

#### Motion decomposition

- `global_motion_hidden_dim`
- `local_motion_hidden_dim`
- `max_disp_global_ratio`
- `max_disp_local_ratio`
- `max_rot_global`
- `max_rot_local`
- `max_scale_global`
- `max_scale_local`

#### Cut-aware gating

- `cut_gate_temperature_init`
- `cut_gate_temperature_final`
- `cut_gate_sparsity_weight`
- `cut_disconnect_threshold`

#### Lifecycle

- `enable_lifecycle`
- `lifecycle_hidden_dim`
- `lambda_lifecycle_persist`
- `lambda_lifecycle_churn`
- `lambda_lifecycle_transient`

#### Visibility / appearance

- `lambda_visibility_balance`
- `lambda_visibility_sparse`
- `lambda_appearance_residual`

#### Phase schedule

- `stage_global_only_end`
- `stage_graph_bootstrap_end`
- `stage_local_motion_end`
- `stage_visibility_enable_iter`
- `stage_lifecycle_enable_iter`
- `stage_joint_finetune_end`

### Compatibility note

Keep legacy normalization untouched for current presets. CAMS-GS should use its own clean names rather than forcing old MoE aliases to mean new things.

## 8. Add new EndoNeRF presets

Create new presets alongside the existing ones, for example:

- `arguments/endonerf/cutting_cams_gs.py`
- `arguments/endonerf/pulling_cams_gs.py`
- optional ablations:
  - `cutting_cams_gs_no_cut_graph.py`
  - `cutting_cams_gs_no_lifecycle.py`
  - `cutting_cams_gs_global_only.py`

### Fair-comparison rules

The main CAMS-GS presets should match the baseline on protocol-critical settings:

- `coarse_iterations`
- `iterations`
- `position_lr_max_steps`
- `pruning_interval`
- `densify_until_iter`

Do not repeat the earlier confound where the proposed method used a stricter training protocol than the baseline.

## 9. Expected forward data flow

The intended runtime path is:

1. `deform_network._encode_time(...)` produces `time_features`.
2. `Deformation.forward_dynamic(...)` prepares canonical/backbone features.
3. `CAMSGSTracking.forward(...)` receives:
   - canonical or backbone-conditioned `means3d`
   - `scales`
   - `rotations`
   - `opacity_logits`
   - `time_values`
   - `time_features`
   - `scene_scale`
   - `phase`
4. CAMS-GS builds or queries scaffold motion context.
5. Global motion is predicted.
6. Local scaffold-conditioned motion is predicted.
7. Cut-aware graph gating modulates motion propagation.
8. Motion is composed into final geometric updates.
9. Lifecycle state and visibility/appearance corrections are predicted.
10. The module returns:
   - final `means3d_t`
   - final `scales_t`
   - final `rotations_t`
   - final `opacity_logits_t`
   - aux diagnostics.

## 10. Phase schedule

Recommended training curriculum:

### Phase 1: `global_only`

Trainable groups:

- `tracking_time_encoder`
- `tracking_base_grid`
- `tracking_base_deformation`
- `tracking_motion_global`

Purpose:

- learn a stable low-frequency tissue motion prior before graph decisions become active.

### Phase 2: `graph_bootstrap`

Trainable groups:

- previous groups
- `tracking_cut_graph`

Purpose:

- learn stable scaffold assignment and cut-aware attenuation without full local-motion freedom.

### Phase 3: `local_motion_only`

Trainable groups:

- `tracking_motion_local`
- `tracking_cut_graph`
- `tracking_motion_global` with reduced LR

Purpose:

- let local deformation specialize around the scaffold after global motion is already stable.

### Phase 4: `visibility_enable`

Trainable groups:

- previous groups
- `tracking_visibility`
- `tracking_appearance`

Purpose:

- add transience/appearance refinement only after geometry is not collapsing.

### Phase 5: `lifecycle_enable`

Trainable groups:

- previous groups
- `tracking_lifecycle`

Purpose:

- allow appearance/disappearance reasoning after the motion scaffold is meaningful.

### Phase 6: `joint_finetune`

Trainable groups:

- all CAMS-GS groups

LR policy:

- keep global motion and base backbone at lower LR,
- keep graph and local motion moderately active,
- keep lifecycle and visibility heads bounded by smaller phase scales to avoid late collapse.

## 11. Aux outputs required for training and diagnostics

CAMS-GS should emit enough aux data to keep `train.py` logging and `scene/tracking_losses.py` supervision explicit.

### Required aux tensors

- `d_mu`
- `d_rot`
- `d_scale`
- `d_opacity_logit`
- `global_d_mu`
- `local_d_mu`
- `scaffold_weights`
- `cut_gate_values`
- `lifecycle_logits`
- `visibility_logits`
- `component_ids` or equivalent connectivity summary
- `entropy_cut_graph` or equivalent gate uncertainty metric

### Optional scalar diagnostics

- scaffold node utilization,
- active edge ratio,
- disconnected component count,
- lifecycle transient fraction,
- motion decomposition energy ratio `||local|| / ||global||`.

## 12. Test plan

## Extend `tests/test_disentangled_moe_tracking.py`

Add CAMS-GS-focused coverage for:

- `tracking_type='cams_gs'` dispatch works,
- parameter groups include every required CAMS-GS prefix,
- phase gating keeps base groups live and correctly stages new groups,
- checkpoint metadata writes and rejects incompatible CAMS-GS payloads,
- zero-initialized CAMS-GS path is numerically stable at initialization,
- cut-graph and lifecycle aux outputs exist with expected shapes.

## Extend `tests/test_endonerf_presets.py`

Add coverage for:

- main CAMS-GS presets match baseline protocol-critical knobs,
- CAMS-GS presets only use known parser keys,
- CAMS-GS ablation presets differ only in intended knobs.

## Optional new tests

If the file becomes too crowded, create:

- `tests/test_cams_gs_tracking.py`
- `tests/test_cams_gs_presets.py`

## 13. Staged implementation order

### Patch A: public plumbing

- add `tracking_type='cams_gs'`
- add scheduler skeleton
- add placeholder head
- add checkpoint version string
- add empty parameter groups
- add tests for dispatch and metadata

Success criterion:

- repo runs and tests pass without yet changing science behavior.

### Patch B: motion scaffold core

- implement scaffold builder
- implement global/local motion decomposition
- wire cut-aware gating
- return structured aux outputs
- add motion/graph tests

Success criterion:

- forward path is stable and phase wiring works end to end.

### Patch C: visibility and lifecycle

- add visibility/appearance head
- add lifecycle head
- integrate losses
- add tests for aux outputs and phase progression

Success criterion:

- full CAMS-GS path trains without outer-loop changes.

### Patch D: presets and documentation

- add `cutting_cams_gs.py` and `pulling_cams_gs.py`
- add ablation presets
- update `README.md`
- replace or supersede `HETEROGENEOUS_MOE_ARCHITECTURE.md` with CAMS-GS documentation

Success criterion:

- experiment entrypoints and architecture docs match the actual code.

## 14. Verification plan

After each patch:

1. Run targeted unit tests.
2. Run `pytest tests/test_disentangled_moe_tracking.py tests/test_endonerf_presets.py` or the CAMS-GS split equivalents.
3. Run `python -m compileall scene models tests arguments`.
4. Inspect optimizer param-group names and phase gating behavior.
5. Before experiments, confirm CAMS-GS presets preserve fair protocol parity against `*_original.py`.

## 15. Reviewer-facing ablation ladder

The first ablations should be:

- baseline `original`
- `cams_gs_global_only`
- `cams_gs_no_cut_graph`
- `cams_gs_no_lifecycle`
- full `cams_gs`

This ladder directly answers the likely reviewer questions:

- does structure help over the baseline,
- does cut-aware connectivity matter,
- does lifecycle matter,
- is the gain coming from geometry structure or merely from extra visibility parameters.

## Final recommendation

Implement CAMS-GS as a **new tracking path** rather than mutating the current heterogeneous MoE head again. The repository already has the right outer abstractions for:

- phase-aware training,
- optimizer-group ownership,
- tracking aux aggregation,
- preset validation,
- checkpoint metadata safety.

The correct engineering move is therefore not another incremental MoE patch, but a structured replacement inside the existing orchestration contract.