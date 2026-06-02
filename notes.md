# Notes: CAMS-GS Pixel-Space Routing Refactor

## Sources

### Source 1: MoE-GS public paper/project/code
- URL: https://openreview.net/forum?id=WrEQFwWCdT
- URL: https://paper.pnu-cvsp.com/MoE-GS
- URL: https://github.com/cvsp-lab/MoE-GS
- Key points:
  - The core idea is not just multiple experts, but moving decisive routing closer to image-space supervision.
  - Public code structure indicates per-Gaussian signals are splatted/rasterized into per-pixel weight maps in the renderer.
  - Router training uses image reconstruction to supervise pixel-space blending.

### Source 2: Current EndoMoeGaussian routing path
- Local files:
  - `scene/deformation.py`
  - `models/tracking/cams_gs_tracking.py`
  - `gaussian_renderer/__init__.py`
  - `scene/tracking_losses.py`
- Key points:
  - Current CAMS routing is decided at the Gaussian / motion-branch level before rendering.
  - Renderer currently consumes only final deformed Gaussian states plus `appearance_rgb_delta`.
  - Losses/logging already flow through `deformation_aux`, which is the safest compatibility seam for the redesign.

## Synthesized Findings

### Architectural shift
- Keep global/local/cut-graph and visibility/lifecycle modules as proposal generators.
- Stop treating `pi_geo` / `pi_vis` as final gates.
- Expose expert-wise proposals and Gaussian prior routing in aux first.
- Move final routing/fusion into renderer-side pixel-space composition.

### Contract constraints to preserve
- Preserve `tracking_type='cams_gs'` externally.
- Preserve `TrackingPhase` / scheduler semantics.
- Preserve optimizer-group naming and aux-driven loss integration where possible.
- Preserve render() return keys and checkpoint metadata strictness.
