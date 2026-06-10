# EndoMoeGaussian Engineering Plan

## 1. Target Model

The canonical Gaussian cloud produced by coarse reconstruction remains the immutable reference state. The dynamic model is decomposed as:

`G_t = BaseHexPlane(G_0, t) + Sum_k Router_k(G_0, t, view) * ResidualExpert_k(G_0, BaseHexPlane, t, view)`.

The shared base is the original EndoGaussian multi-resolution 4D HexPlane deformation, but its output heads are initialized to exact identity. This preserves the baseline function class without destroying the static checkpoint at fine-stage step 1.

Three heterogeneous residual experts operate over the shared base:

1. `global_smooth`: low-frequency time-conditioned translation with conservative rotation/scale.
2. `tissue_local`: spatial-temporal residual for non-rigid tissue motion with neighborhood coherence.
3. `tool_contact`: cut-graph and view-conditioned residual for instruments, contact boundaries, occlusion, and appearance changes.

Experts produce complete Gaussian proposals, but their residuals are zero initialized. They do not contain nested routers.

## 2. Routing

The Gaussian router produces a stable per-Gaussian prior from canonical position, shared-base motion, opacity, temporal encoding, and expert residual magnitudes. It uses soft top-2 routing during training.

The pixel router composes rendered expert proposals in image space. Its inputs include expert RGB, depth, alpha, projected motion, Gaussian prior logits, coverage, and view direction. The final weights are:

`softmax(pixel_logits + log(gaussian_prior + eps))`.

Load balancing prevents dead experts, but fixed usage targets are warm-up priors only. The final router is optimized by image reconstruction and counterfactual expert advantage, not forced to match arbitrary global ratios.

## 3. Training Protocol

### Stage A: Shared Base
- Load the static coarse checkpoint.
- Train only identity-safe HexPlane base deformation.
- Use fixed absolute boundaries and full learning rate age local to this stage.
- Validate that step-1 PSNR remains close to the coarse result.

### Stage B: Expert Pretraining
- Clone the frozen shared base into each expert training context.
- Train each residual expert independently with a forced route and its own temporal encoder.
- Save one checkpoint per expert.
- Reject an expert if it does not outperform the shared base on its specialization mask or counterfactual subset.

### Stage C: Router Training
- Load and freeze all expert checkpoints.
- Train Gaussian prior and pixel router only.
- Compare learned routing against uniform blending, oracle best expert, and every single expert.

### Stage D: Joint Fine-Tuning
- Unfreeze residual heads only, not the canonical cloud or shared base initially.
- Use expert LR at 0.05-0.10 of router LR.
- Unfreeze visibility/appearance only after router validation stabilizes.

## 4. Visibility and Lifecycle

Visibility and lifecycle must affect rendered alpha even when the geometry router uses Gaussian-level blending. Both heads use identity-safe initialization:

- visibility alpha starts at one;
- lifecycle persistence starts near one;
- transient appearance starts at zero.

Appearance residual is multiplied by transient probability. Lifecycle and visibility are supervised through rendered RGB/depth/alpha, not only auxiliary balance losses.

## 5. Regularization

- Temporal regularization samples adjacent timestamps in the same optimization batch.
- Spatial coherence uses an actual KNN index operator, not `distCUDA2` scalar distances.
- Motion magnitude is normalized by scene scale and phase.
- Rotation/scale penalties are expert-specific and activated only after expert warm-up.
- Router confidence is scheduled: entropy-preserving early, decisive late.
- Fixed usage targets decay to weak priors after router warm-up.

## 6. Instrumentation

Log fixed-view train/test PSNR, SSIM, LPIPS, RGB/depth/TV loss components, point count, each optimizer-group LR and gradient norm, route entropy/margin, per-expert motion, residual saturation, alpha statistics, and counterfactual expert PSNR.

## 7. Implementation Order

1. Absolute stages and per-group local LR.
2. Complete diagnostics and fixed-view validation.
3. Identity-safe HexPlane base.
4. Heterogeneous residual expert interfaces.
5. Independent expert checkpoint workflow.
6. Gaussian-prior router.
7. Pixel router and image-space composition.
8. Visibility/lifecycle photometric integration.
9. Temporal/spatial regularization.
10. Full ablation and reproducibility scripts.

## 8. Required Experiments

- EndoGaussian baseline under the identical data and evaluation protocol.
- Identity-safe shared base only.
- Shared base plus each individual expert.
- Gaussian router without pixel router.
- Pixel router without Gaussian prior.
- Full dual-router method.
- No visibility, no lifecycle, no temporal loss, and no spatial loss ablations.
- At least three seeds for the final comparison.

## 9. Success Criteria

- No fine-stage PSNR collapse.
- Shared-base-only result matches the EndoGaussian baseline within 0.3 dB.
- Each accepted expert improves its designated subset.
- Learned routing exceeds the best single expert.
- Full model improves PSNR/SSIM/LPIPS without introducing temporal flicker or black frames.
