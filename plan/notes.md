# Notes: EndoMoeGaussian Redesign

## TensorBoard Evidence
- Latest run: `output/last`, 15000 fine iterations, 52 scalar tags.
- Fine L1 phase means: `0.03321`, `0.02964`, `0.02369`, `0.01901`, `0.01774`, `0.01725`.
- Last-1000 L1 improved from `0.01814` at 9000 iterations to `0.01716` at 15000.
- Last-1000 total loss worsened from `0.05046` to `0.06809`.
- Local plus cut motion magnitude increased from about `0.223` to `0.495`.
- Rotation and scale residual norms remained near `0.10`.
- Visibility transient usage ended near `0.046` despite target `0.15`.
- Lifecycle persistent usage ended near `0.563` despite target `0.8`.
- Opacity residual remained exactly zero.

## Confirmed Engineering Faults
- `cutting_cams_gs.py` runs `cams_gs`, not the independent-expert `cams_gs_moe` path.
- `graph_bootstrap` trains cut-graph parameters while `active_geo=1` masks their rendered output.
- Fractional stages delay newly enabled modules when total iterations increase.
- Learning-rate decay still reaches its floor at step 9000 in a 15000-step run.
- CAMS paths use the canonical identity state instead of the EndoGaussian HexPlane dynamic backbone.
- With `no_do=True` and pixel routing disabled, visibility/lifecycle have no reliable opacity-supervised image path.
- Temporal loss needs multiple temporal samples but default training samples one camera.
- Spatial loss assumes a neighbor-index output from `distCUDA2` that the extension does not provide.
- Current `cams_gs_moe` experts are nested copies of the same CAMS module and share the outer time encoder.

## Target Gradient Contract
- Shared base: photometric, depth, temporal, and spatial losses.
- Global expert: photometric residual and magnitude regularization.
- Tissue-local expert: photometric residual, neighborhood coherence, and bounded rotation/scale.
- Tool/contact expert: photometric residual, motion-boundary consistency, visibility, and appearance.
- Gaussian router: load balance, confidence schedule, and expert counterfactual advantage.
- Pixel router: final RGB/depth loss and spatial smoothness.
- Lifecycle: opacity-rendered photometric loss plus identity-safe persistence prior.

## Acceptance Gates
- Fine step 1 PSNR must remain within 0.5 dB of the coarse checkpoint.
- No training phase may contain trainable parameters with zero rendered contribution.
- Every active expert must have non-zero gradient norm and measurable counterfactual image contribution.
- Validation PSNR must be logged on a fixed camera subset every 500 steps.
- Router-only training must improve validation PSNR over uniform and best-single-expert baselines.
