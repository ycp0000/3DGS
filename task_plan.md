# Task Plan: CAMS-GS Pixel-Space Routing Refactor

## Goal
Redesign CAMS-GS so Gaussian-space experts remain proposal generators while the decisive routing/fusion moves into pixel space, informed by MoE-GS-style renderer-side routing.

## Phases
- [ ] Phase 1: Expose expert-wise proposals and Gaussian prior routing in aux
- [ ] Phase 2: Add renderer-side expert evidence maps and pixel router plumbing
- [ ] Phase 3: Switch tracking losses/logging/presets to pixel-space routing
- [ ] Phase 4: Review, verify, and prepare experiment-ready presets

## Key Questions
1. Which current CAMS outputs must remain stable so training/rendering contracts do not break?
2. What is the smallest non-breaking step toward pixel-space routing?
3. Which existing tests should be extended first to lock the new aux contract?

## Decisions Made
- Preserve `tracking_type='cams_gs'` externally and change internals incrementally.
- Keep `scene/deformation.py -> gaussian_renderer -> train.py -> scene/tracking_losses.py` as the main integration seam.
- Implement geometry-side pixel routing before visibility-side pixel routing.

## Errors Encountered
- `pyright` is not installed in the current environment, so static type verification could not run during earlier verification.

## Status
**Currently in Phase 1** - Writing tests and plumbing for expert-wise CAMS proposal tensors and Gaussian prior routing before touching final renderer composition.
