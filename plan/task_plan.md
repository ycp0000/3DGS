# Task Plan: EndoMoeGaussian Engineering Redesign

## Goal
Build a theoretically consistent EndoMoeGaussian pipeline that preserves the EndoGaussian HexPlane baseline, trains heterogeneous experts without gradient starvation, and learns image-supervised routing for dynamic endoscopic reconstruction.

## Phases
- [x] Phase 1: Audit TensorBoard and current gradient paths
- [x] Phase 2: Define target architecture and training contracts
- [x] Phase 3: Rebuild stage scheduler and group-local learning rates
- [x] Phase 4: Restore identity-safe shared HexPlane deformation
- [x] Phase 5: Implement heterogeneous residual experts
- [x] Phase 6: Implement Gaussian-prior and pixel-space routing
- [x] Phase 7: Repair visibility, lifecycle, and temporal supervision
- [x] Phase 8: Add diagnostics, tests, scripts, and documentation
- [ ] Phase 9: Run baseline, ablation, and full-model experiments

## Key Questions
1. Does every enabled module receive a measurable photometric gradient?
2. Does every expert improve over the frozen shared base on its intended region?
3. Does routing improve validation PSNR instead of only satisfying usage targets?
4. Does the full method retain or exceed the EndoGaussian baseline?

## Decisions Made
- Keep `cams_gs` as a non-MoE diagnostic baseline; develop the final method under `cams_gs_moe`.
- Use a zero-output-initialized HexPlane deformation as the shared dynamic base.
- Experts predict bounded residuals over the shared base, not unrelated absolute states.
- Train experts independently with frozen shared/base components before router training.
- Use Gaussian routing as a prior and pixel routing as the final image-space composition mechanism.
- Use absolute stage boundaries and per-group local learning-rate ages.
- Require fixed-view PSNR, gradient norms, and expert counterfactual metrics before accepting a stage.

## Errors Encountered
- Independent Codex review timed out twice on 2026-06-10; local code and TensorBoard audit remains the source of truth.

## Status
**Currently in Phase 9** - implementation is ready for server-side baseline, ablation, and full-model experiments.
