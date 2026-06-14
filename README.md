# EndoMoeGaussian

EndoMoeGaussian is an endoscopy-adapted dynamic Gaussian Splatting system built on EndoGaussian. It keeps a high-quality EndoGaussian dynamic reconstruction as an always-on Global anchor, then learns two heterogeneous residual experts for tissue-local deformation and transient tool-contact content.

The old single-model `cams_gs_moe` continuous curriculum and the old three-independent-reconstructor design remain historical ablations. The recommended implementation uses a strict Global-to-residual lineage and versioned identity-safe bundles.

## Method

### Stage 1: static canonical reconstruction

- Runs the EndoGaussian coarse/static reconstruction.
- Saves `canonical.pth`.
- The canonical fingerprint is shared by all subsequent experts.

### Stage 2: Global dynamic anchor

- Restores `canonical.pth`.
- Trains the official EndoGaussian deformation backbone and canonical Gaussian parameters.
- Saves the quality-gated `global.pth`.
- Preserves raw EndoGaussian residual semantics and the official 64-channel HexPlane output.

### Stage 3: heterogeneous residual experts

Both residual experts restore the optimized canonical cloud and dynamic backbone from `global.pth`. Canonical parameters, the Global deformation field, HexPlane, and outer time encoder are frozen.

- `local`: a SC-GS/MoSca-inspired sparse SE(3) motion scaffold with farthest-point control nodes, bounded surface-local node offsets/radii, surface-aware KNN skinning, learnable sparse node gates, absolute spatial support, identity-preserving Dual Quaternion Blending, ARAP regularization, and temporal acceleration regularization. It refines position and rotation only.
- `contact`: an STG/HyperNeRF-inspired auxiliary spacetime Gaussian bank with persistent tissue-parent bindings, bounded second-order trajectories, multiple temporal RBF charts, learned lifecycle duration, and projected tool-boundary supervision. It models transient contact surfaces without forcing the canonical cloud to explain appearance/disappearance.

The residual modules start as exact identities. Residual training uses a frozen Global render as a teacher: full-image reconstruction keeps gradients on every valid pixel, high-error regions receive bounded extra boosting, already-correct regions are preserved by distillation, and a smooth pixel-wise no-regret barrier penalizes regressions. Binocular inverse-depth supervision is also teacher-relative. Legacy TV, DSSIM, LPIPS, and monocular Pearson depth losses are disabled in residual stages because they bypass the Global quality anchor. A low-LR warm-up, refinement-only gradient clipping, exact fixed-view RGB/depth parity checks, and best-state rollback prevent Local or Contact bundles from finishing below the Global anchor.

All structural motion bounds are explicit preset parameters. They are written into `cfg_args` and expert bundles, including Local node offset/radius limits and Contact spatial, velocity, acceleration, rotation, scale, and duration limits.

### Stage 4: frozen residual Router

All experts are frozen. The Router alone is optimized using:

- Global always-on reconstruction
- independent Local and Contact Gaussian gates that do not sum to one
- Gaussian features from canonical position, Local/Contact residual motion, opacity change, aggregated Contact child activity, view direction, and time
- exact-zero forward gates with sigmoid surrogate gradients
- Gaussian-state composition before one final rasterization
- detached incremental-gain supervision relative to Global
- gate sparsity and no-regret penalties
- fixed-view `G`, `G+L`, `G+C`, `G+L+C`, and per-pixel oracle headroom checks

The Router stage fails immediately if residual experts provide insufficient oracle headroom or if either Router parameter branch receives no gradient.

### Optional Stage 5: controlled joint fine-tuning

This is a conservative optional stage, not a requirement of the main method:

- canonical geometry, appearance, opacity, scale, rotation, and topology remain frozen
- the Global anchor and its deformation field remain frozen
- the Router uses small learning rates
- local/contact experts may update only their role-specific refinement modules
- parameter-anchor loss and gradient clipping constrain drift
- the result is saved only if the Router and every individual expert pass fixed-view PSNR non-degradation gates

Joint fine-tuning writes a new assembly directory and never overwrites the frozen expert/Router bundles.

## Identity and safety contracts

Expert bundle format version 6 and Router bundle version 6 bind:

- absolute dataset path
- source canonical fingerprint
- complete expert-state fingerprint
- canonical topology fingerprint
- point count
- tracking architecture version
- fixed-view validation metrics
- Router architecture and exact expert manifest

The complete expert-state fingerprint includes deformation weights and spatial context. A same-topology expert with different dynamic weights is rejected.
Bundles produced by earlier independent-expert or pre-bounded residual architectures must be retrained; they are intentionally rejected instead of being silently migrated.

The version-6 residual protocol changes Local state, residual objectives, depth protection, and start-parity validation. Use a new bundle directory and rerun canonical, Global, Local, Contact, and Router stages; do not reuse version-5 expert or Router bundles.

Other enforced contracts:

- EndoNeRF paths must be absolute.
- EndoNeRF scenes require `poses_bounds.npy`.
- Presets set `extra_mark='endonerf'`.
- NumPy/Torch AABB boundaries are converted explicitly.
- Router and Joint use camera-only `Scene` objects; they never initialize a dummy Gaussian cloud.
- Public depth remains single-channel `[1, H, W]`.

## Environment

```bash
git submodule update --init --recursive
conda create -n endomoe python=3.7 -y
conda activate endomoe
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The current implementation retains Python 3.7 syntax compatibility checks.

## Dataset

Use an absolute EndoNeRF scene path:

```text
/root/3DGS/data/endonerf/cutting
```

The scene directory must contain:

```text
poses_bounds.npy
```

Do not use `root/3DGS/...`; the leading `/` is required.

## Presets

- `arguments/endonerf/cutting_endomoeg.py`
- `arguments/endonerf/pulling_endomoeg.py`

The presets are stage-neutral. Pipeline stage, expert role, bundle directory, and output directory are supplied explicitly on the command line.

## Complete cutting workflow

Adjust `SOURCE`, `RUN_ROOT`, and `MIN_EXPERT_PSNR` for the server. Set `MIN_EXPERT_PSNR` close to the original EndoGaussian fixed-view test PSNR. Local/Contact training will not start unless `global.pth` passes this gate.

```bash
cd /root/3DGS

SOURCE=/root/3DGS/data/endonerf/cutting
RUN_ROOT=/root/autodl-tmp/endomoeg/cutting
BUNDLES=$RUN_ROOT/bundles
CONFIG=arguments/endonerf/cutting_endomoeg.py
MIN_EXPERT_PSNR=36.8

mkdir -p "$RUN_ROOT" "$BUNDLES"
```

### 1. Static canonical

```bash
python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/01_canonical" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage canonical \
  --endomoeg_bundle_dir "$BUNDLES"
```

Expected bundle:

```text
$BUNDLES/canonical.pth
```

### 2. Global expert

```bash
python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/02_global" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage expert \
  --endomoeg_expert_role global \
  --endomoeg_bundle_dir "$BUNDLES" \
  --endomoeg_canonical_bundle "$BUNDLES/canonical.pth"
```

### 3. Local expert

```bash
python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/03_local" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage expert \
  --endomoeg_expert_role local \
  --endomoeg_bundle_dir "$BUNDLES" \
  --endomoeg_canonical_bundle "$BUNDLES/canonical.pth" \
  --endomoeg_min_expert_psnr "$MIN_EXPERT_PSNR"
```

### 4. Contact expert

```bash
python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/04_contact" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage expert \
  --endomoeg_expert_role contact \
  --endomoeg_bundle_dir "$BUNDLES" \
  --endomoeg_canonical_bundle "$BUNDLES/canonical.pth" \
  --endomoeg_min_expert_psnr "$MIN_EXPERT_PSNR"
```

Before Router training, confirm:

```text
$BUNDLES/global.pth
$BUNDLES/local.pth
$BUNDLES/contact.pth
```

### 5. Frozen residual Router

```bash
python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/05_router" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage router \
  --endomoeg_stage_iterations 4000 \
  --endomoeg_bundle_dir "$BUNDLES" \
  --endomoeg_min_expert_psnr "$MIN_EXPERT_PSNR"
```

Expected bundle:

```text
$BUNDLES/router.pth
```

Before optimization, TensorBoard must show at least the configured `0.3 dB` `oracle - global` headroom. If this check fails, improve the Local/Contact experts instead of forcing Router training.

### 6. Optional controlled Joint

```bash
JOINT_BUNDLES=$BUNDLES/joint

python train.py \
  -s "$SOURCE" \
  --model_path "$RUN_ROOT/06_joint" \
  --configs "$CONFIG" \
  --endomoeg_pipeline_stage joint \
  --endomoeg_stage_iterations 1000 \
  --endomoeg_bundle_dir "$BUNDLES" \
  --endomoeg_router_bundle "$BUNDLES/router.pth" \
  --endomoeg_joint_output_dir "$JOINT_BUNDLES" \
  --endomoeg_min_expert_psnr "$MIN_EXPERT_PSNR"
```

The Joint assembly is:

```text
$JOINT_BUNDLES/global.pth
$JOINT_BUNDLES/local.pth
$JOINT_BUNDLES/contact.pth
$JOINT_BUNDLES/router.pth
```

## Pulling workflow

Use the same commands with:

```bash
SOURCE=/root/3DGS/data/endonerf/pulling
RUN_ROOT=/root/autodl-tmp/endomoeg/pulling
CONFIG=arguments/endonerf/pulling_endomoeg.py
```

## TensorBoard

```bash
tensorboard --logdir /root/autodl-tmp/endomoeg --port 6006
```

Important expert metrics:

- `fine/validation/test/psnr`
- `fine/validation/test/ssim`
- `fine/validation/test/lpips`
- training loss and role-specific tracking statistics
- `fine/tracking/losses/L_scaffold_arap`
- `fine/tracking/losses/L_scaffold_acceleration`
- `fine/tracking/losses/L_scaffold_gate_sparsity`
- `fine/tracking/losses/L_residual_boost`
- `fine/tracking/losses/L_residual_reconstruction`
- `fine/tracking/losses/L_residual_preserve`
- `fine/tracking/losses/L_residual_no_regret`
- `fine/tracking/losses/L_residual_total`
- `fine/tracking/losses/L_residual_depth`
- `fine/tracking/losses/L_contact_bank_sparsity`
- `fine/tracking/losses/L_contact_bank_locality`
- `fine/tracking/stats/scaffold_node_translation_norm`
- `fine/tracking/stats/scaffold_point_gate_mean`
- `fine/tracking/stats/scaffold_spatial_support_mean`
- `fine/tracking/stats/residual_teacher_error`
- `fine/tracking/stats/residual_candidate_error`
- `fine/tracking/stats/residual_teacher_psnr`
- `fine/tracking/stats/residual_psnr_delta`
- `fine/tracking/stats/residual_regressed_fraction`
- `fine/tracking/stats/residual_depth_regressed_fraction`
- `fine/tracking/stats/residual_grad_norm_before_clip`
- `fine/tracking/stats/lr_group_tracking_expert_refinement`
- `fine/tracking/stats/contact_bank_temporal_activity`
- `fine/tracking/stats/contact_bank_boundary_support`
- `fine/residual/global_baseline_psnr`
- `fine/residual/best_psnr`
- `fine/residual/parity_rgb_max_abs`
- `fine/residual/parity_depth_max_abs`

Important Router metrics:

- `router/headroom/psnr_global`
- `router/headroom/psnr_local`
- `router/headroom/psnr_contact`
- `router/headroom/psnr_full`
- `router/headroom/psnr_oracle`
- `router/train/L_total`
- `router/train/L_router_reconstruction`
- `router/train/L_router_no_regret`
- `router/train/L_router_gate_sparsity`
- `router/train/L_router_gain_local`
- `router/train/L_router_gain_contact`
- `router/train/router_usage_local`
- `router/train/router_usage_contact`
- `router/train/router_target_local`
- `router/train/router_target_contact`
- `router/gradients/grad_norm_router_base_gates`
- `router/gradients/grad_norm_router_feature_mlp`
- `router/validation/test/psnr`

Important Joint metrics:

- `joint/train/L_total`
- `joint/train/L_anchor`
- `joint/gradients/grad_norm_joint_expert_local`
- `joint/gradients/grad_norm_joint_expert_contact`
- `joint/validation/test/psnr`

Stop and inspect the run if:

- an expert is below the original EndoGaussian quality level
- any Router gradient norm is zero or non-finite
- oracle headroom is below the configured threshold
- Local/Contact targets indicate gain but the corresponding gate remains zero
- rendered frames become black
- PSNR shows abrupt frame-dependent drops

## Render

Render the frozen residual Router:

```bash
python render.py \
  --model_path "$RUN_ROOT/05_router" \
  --skip_train \
  --configs "$CONFIG"
```

Render the optional Joint assembly:

```bash
python render.py \
  --model_path "$RUN_ROOT/06_joint" \
  --skip_train \
  --configs "$CONFIG"
```

`render.py` reads the saved `cfg_args`. Router runs load Stage 3 bundles; Joint runs load the new Joint bundle directory.

## Evaluate

```bash
python metrics.py -m "$RUN_ROOT/05_router"
python metrics.py -m "$RUN_ROOT/06_joint"
```

Always compare against:

- original EndoGaussian
- CAMS-GS
- Global anchor and each residual candidate
- frozen residual Router
- optional Joint

Do not claim an improvement from training-batch PSNR alone; use fixed-view test metrics and rendered videos.

## Output layout

```text
/root/autodl-tmp/endomoeg/cutting/
├── 01_canonical/
├── 02_global/
├── 03_local/
├── 04_contact/
├── 05_router/
├── 06_joint/
└── bundles/
    ├── canonical.pth
    ├── global.pth
    ├── local.pth
    ├── contact.pth
    ├── router.pth
    └── joint/
        ├── global.pth
        ├── local.pth
        ├── contact.pth
        └── router.pth
```

Every training output directory contains `cfg_args` and TensorBoard event files.

## Baselines and ablations

Recommended comparisons:

1. original EndoGaussian
2. CAMS-GS
3. Global EndoGaussian anchor
4. Global + Local scaffold
5. Global + Contact spacetime bank
6. Global + Local + Contact without learned gates
7. frozen residual Router
8. controlled Joint
9. Local without DQB/ARAP
10. Contact without temporal charts/tool-boundary supervision
11. Router without incremental-gain supervision
12. Router without no-regret loss
13. Router without headroom fail-fast
14. legacy independent-expert Router
15. legacy continuous `cams_gs_moe`

Legacy independent-expert and residual-component checkpoints are intentionally incompatible with the version-6 heterogeneous residual pipeline.

## Tests

Focused tests cover:

- absolute EndoNeRF paths and loader selection
- NumPy/Torch AABB boundaries
- Global parity, Local scaffold, Contact spacetime-bank architecture and gradients
- canonical/expert/Router bundle identity
- full deformation-state fingerprinting
- Global-to-residual lineage and frozen-parameter contracts
- exact-zero residual gates and Gaussian-state composition
- real Router gradient flow through composite and gate rasterization
- incremental-gain, no-regret, and headroom fail-fast behavior
- camera-only Router/Joint rendering
- controlled Joint trainable-parameter whitelist and quality gates
- Python 3.7 syntax compatibility

Run:

```bash
python -m pytest tests -q
```
