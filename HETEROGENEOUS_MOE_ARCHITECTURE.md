# Heterogeneous MoE Tracking Architecture

## Goal

`tracking_type='heterogeneous_moe'` replaces the old single-backbone dynamic head with a true heterogeneous Mixture-of-Experts tracking module. The design follows the MoE-GS-style idea that different motion patterns should be modeled by different experts instead of forcing all dynamics through one shared deformation head.

The implemented objective is:

- use different geometry experts for different motion regimes,
- decouple geometry routing from visibility/opacity routing,
- train experts with an explicit stage schedule instead of activating everything from iteration 0,
- keep motion bounded by scene scale so expert capacity has a clear physical meaning,
- regularize motion using adjacent-time consistency from the current batch instead of previous optimizer-step state.

## Top-Level Data Flow

In heterogeneous mode the runtime path is:

1. `scene/deformation.py` selects `tracking_mode='hetero_moe'`.
2. `Deformation.forward_dynamic()` obtains the current `TrackingPhase` from `HeterogeneousMoEScheduler`.
3. The time encoder provides `time_features`; the heterogeneous head receives:
   - `means3d`
   - `scales`
   - `rotations`
   - `opacity_logits`
   - `time_values`
   - `time_features`
   - `scene_scale`
   - `phase`
4. `HeterogeneousMoETracking.forward()` computes geometry routing, geometry-expert outputs, visibility routing, and visibility-expert outputs.
5. The module returns:
   - translated means `means3d_t`
   - unchanged `scales`
   - unchanged `rotations`
   - updated `opacity_logits_t`
   - auxiliary routing and regularization statistics

In the current implementation, heterogeneous mode predicts translation and opacity changes only. `d_rot` is zero and `d_scale` is zero by design.

## Geometry Branch

### Geometry experts

`HeterogeneousMoETracking.GEO_EXPERT_NAMES = ('static', 'hexplane', 'local', 'smooth')`

Each expert models a different motion prior:

- `static`: zero-motion expert for stable tissue points.
- `hexplane`: global spatio-temporal deformation expert driven by a dedicated `HexPlaneField` and an MLP head.
- `local`: local residual motion expert driven by an MLP on normalized position, time features, scale, and opacity context.
- `smooth`: low-capacity residual expert that captures small smooth corrections.

### Geometry routing

The geometry router sees:

- normalized 3D position,
- time features from the time encoder,
- Gaussian scales,
- opacity logits.

It outputs a 4-way routing distribution `pi_geo` over `static / hexplane / local / smooth`.

Routing supports:

- temperature-controlled soft routing,
- optional sparse top-k routing,
- forced one-hot routing during warmup phases.

### Geometry output

Each active expert predicts a bounded translation field. The final geometry update is the weighted sum of expert outputs:

- `d_mu = sum_k pi_geo[k] * d_mu_k`
- `means3d_t = means3d + d_mu`

Bounding is scene-scale-aware:

- `hexplane` motion is bounded by `max_disp_hexplane_ratio * scene_scale`
- `local` motion is bounded by `max_disp_local_ratio * scene_scale`
- `smooth` motion is bounded by `max_disp_smooth_ratio * scene_scale`

This keeps each expert's capacity interpretable and prevents router collapse through unbounded residual magnitude.

## Visibility Branch

### Visibility experts

`HeterogeneousMoETracking.VIS_EXPERT_NAMES = ('stable', 'transient')`

The visibility branch is intentionally smaller than the geometry branch:

- `stable`: zero-opacity-change expert for persistent points.
- `transient`: bounded opacity-logit residual expert for occlusion / appearance transients.

### Visibility routing

Visibility routing is conditioned on post-geometry context, not only raw inputs. The visibility router uses:

- translated normalized position,
- detached geometry displacement `d_mu`,
- displacement norm,
- opacity logits,
- detached geometry routing `pi_geo`,
- time features.

This means the visibility decision depends on what kind of motion the geometry branch already inferred.

### Visibility output

The visibility branch predicts an opacity-logit delta:

- `d_opacity = sum_j pi_vis[j] * d_opacity_j`
- `opacity_logits_t = opacity_update(opacity_logits, d_opacity)`

The transient branch is bounded by `max_opacity_delta`.

## Stage Scheduler

The scheduler is defined by `TrackingPhase` and `HeterogeneousMoEScheduler`.

A phase specifies:

- active geometry/visibility expert counts,
- routing temperatures,
- whether sparse routing is enabled,
- forced experts for warmup phases,
- which optimizer parameter groups are trainable,
- per-group LR scaling.

The current training curriculum is:

1. `hexplane_only`
   - force geometry to `hexplane`
   - train time encoder + hexplane geometry path
2. `smooth_only`
   - force geometry to `smooth`
   - train time encoder + smooth path
3. `local_only`
   - force geometry to `local`
   - train time encoder + local path
4. `router_only`
   - activate full geometry routing before the final joint stage
5. `joint_finetune`
   - full geometry routing
   - optional visibility routing
   - optional sparse top-k routing

This schedule is important because it prevents early router noise from starving experts before they have learned a useful specialization.

## Optimizer / Parameter Groups

The heterogeneous head exposes named parameter groups for phase-aware optimization. Training can selectively enable:

- time encoder parameters,
- hexplane grid parameters,
- hexplane MLP parameters,
- local expert parameters,
- smooth expert parameters,
- geometry router parameters,
- visibility router/expert parameters.

`train.py` now asks the deformation module for the active phase and updates learning rates with phase context, so the schedule affects both trainability and effective step size.

## Losses and Regularization

`scene/tracking_losses.py` is architecture-aware and no longer assumes the old 3-expert layout.

### Geometry routing losses

The loss code logs usage for named experts and applies:

- geometry balance loss against named target distributions,
- route-confidence loss,
- expert-diversity loss,
- per-expert usage metrics.

Forced single-expert warmup phases skip balance/confidence penalties so the schedule does not fight itself.

### Visibility routing losses

The visibility branch applies:

- visibility balance loss against `stable / transient` targets,
- visibility route-confidence loss,
- sparsity pressure on opacity deltas,
- optional geometry/visibility decoupling loss through transient routing.

### Temporal regularization

Temporal smoothing is based on adjacent times inside the current batch:

- stack `d_mu_sequence` across sampled views,
- sort by `time_sequence`,
- compute finite-difference velocity,
- penalize velocity energy.

This replaces the previous-step-state heuristic and makes the regularizer correspond to real temporal continuity.

### Expert-capacity regularization

The geometry branch also logs and regularizes expert-specific displacement behavior:

- weighted displacement norm,
- weighted displacement ratio relative to its capacity,
- saturation penalty when an expert persistently pushes against its bound.

The active preset knobs are:

- `lambda_mag_g1_mu`, `lambda_mag_g2_mu`, `lambda_mag_g3_mu`
- `lambda_sat_g1_disp`, `lambda_sat_g2_disp`, `lambda_sat_g3_disp`
- `lambda_raw_g1_disp`, `lambda_raw_g2_disp`, `lambda_raw_g3_disp`

These correspond to `hexplane`, `local`, and `smooth` geometry regularization strength.

## Checkpoint / Resume Safety

`scene/gaussian_model.py` now stores deformation metadata with `tracking_type` and validates it during restore/load. This prevents loading an incompatible checkpoint into a different tracking architecture.

## Public API Surface

The tracking package exports:

- `TrackingPhase`
- `HeterogeneousMoEScheduler`
- `HeterogeneousMoETracking`
- `SplitTrackingHead`
- `shape_debug_check`

`DisentangledMoETracking` is currently kept as an alias to `HeterogeneousMoETracking` so existing imports continue to resolve.

## Final Practical Interpretation

The final heterogeneous tracking stack should be understood as:

- a scene-scale-bounded geometry MoE,
- a smaller visibility MoE conditioned on geometry outcome,
- a staged optimization schedule that teaches experts before teaching the router,
- a temporally grounded motion regularizer,
- a displacement-first design where rotation and scale are intentionally left unchanged in heterogeneous mode.

That is the architecture that the current codebase actually trains and evaluates.
