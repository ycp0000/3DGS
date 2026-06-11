# EndoMoeGaussian

EndoMoeGaussian is an endoscopy-adapted dynamic Gaussian Splatting system built on EndoGaussian. Its primary method follows the MoE-GS training principle that every expert must first become a complete scene reconstructor before a separate Router learns how to combine them.

The old single-model `cams_gs_moe` continuous curriculum remains only as a historical ablation. The recommended implementation uses complete, independently optimized experts and versioned identity-safe bundles.

## Method

### Stage 1: static canonical reconstruction

- Runs the EndoGaussian coarse/static reconstruction.
- Saves `canonical.pth`.
- The canonical fingerprint is shared by all subsequent experts.

### Stage 2: three complete dynamic experts

Every expert owns an independent:

- `GaussianModel`
- canonical Gaussian cloud and topology
- appearance parameters
- HexPlane deformation field
- optimizer trajectory and densification history

The roles are:

- `global`: complete EndoGaussian deformation backbone for stable full-scene motion.
- `local`: complete backbone plus a tissue-local refinement field for non-rigid local deformation.
- `contact`: complete backbone plus tool-contact, visibility, lifecycle, and appearance refinement.

All three experts start from the same Stage 1 canonical bundle, but they are trained and saved independently as `global.pth`, `local.pth`, and `contact.pth`.

### Stage 3: frozen volume-aware Router

All experts are frozen. The Router alone is optimized using:

- per-Gaussian learnable logits for each independent expert topology
- shared Gaussian features from position, motion, view direction, opacity, scale, time, and role embedding
- differentiable volumetric feature splatting
- a pixel-space residual Router
- oracle-error distillation from detached per-expert reconstruction errors
- anti-starvation minimum-usage regularization
- dense routing followed by soft top-2 inference

The training loop fails immediately if Gaussian logits, Gaussian feature MLP, or pixel Router receive no gradient.

### Optional Stage 4: controlled joint fine-tuning

This is a conservative fourth stage, not a requirement of the main method:

- canonical geometry, appearance, opacity, scale, rotation, and topology remain frozen
- the Router uses small learning rates
- the global expert may update its deformation field at a very low learning rate
- local/contact experts may update only their role-specific refinement modules
- parameter-anchor loss and gradient clipping constrain drift
- the result is saved only if the Router and every individual expert pass fixed-view PSNR non-degradation gates

Stage 4 writes a new assembly directory and never overwrites Stage 2/3 bundles.

## Identity and safety contracts

Bundle format version 2 binds:

- absolute dataset path
- source canonical fingerprint
- complete expert-state fingerprint
- canonical topology fingerprint
- point count
- tracking architecture version
- fixed-view validation metrics
- Router architecture and exact expert manifest

The complete expert-state fingerprint includes deformation weights and spatial context. A same-topology expert with different dynamic weights is rejected.

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

Adjust `SOURCE`, `RUN_ROOT`, and `MIN_EXPERT_PSNR` for the server. Set `MIN_EXPERT_PSNR` from the original EndoGaussian fixed-view test PSNR, allowing only a small tolerance.

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
  --endomoeg_canonical_bundle "$BUNDLES/canonical.pth"
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
  --endomoeg_canonical_bundle "$BUNDLES/canonical.pth"
```

Before Router training, confirm:

```text
$BUNDLES/global.pth
$BUNDLES/local.pth
$BUNDLES/contact.pth
```

### 5. Frozen Router

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

Important Stage 2 metrics:

- `fine/validation/test/psnr`
- `fine/validation/test/ssim`
- `fine/validation/test/lpips`
- training loss and role-specific tracking statistics

Important Stage 3 metrics:

- `router/train/L_total`
- `router/train/L_router_reconstruction`
- `router/train/L_router_oracle`
- `router/train/L_router_starvation`
- `router/train/router_usage_global`
- `router/train/router_usage_local`
- `router/train/router_usage_contact`
- `router/train/pixel_residual_abs_mean`
- `router/gradients/grad_norm_router_gaussian_logits`
- `router/gradients/grad_norm_router_feature_mlp`
- `router/gradients/grad_norm_router_pixel`
- `router/validation/test/psnr`

Important Stage 4 metrics:

- `joint/train/L_total`
- `joint/train/L_anchor`
- `joint/gradients/grad_norm_joint_expert_global`
- `joint/gradients/grad_norm_joint_expert_local`
- `joint/gradients/grad_norm_joint_expert_contact`
- `joint/validation/test/psnr`

Stop and inspect the run if:

- an expert is below the original EndoGaussian quality level
- any Router gradient norm is zero or non-finite
- one expert usage remains exactly zero
- rendered frames become black
- PSNR shows abrupt frame-dependent drops

## Render

Render the Stage 3 Router:

```bash
python render.py \
  --model_path "$RUN_ROOT/05_router" \
  --skip_train \
  --configs "$CONFIG"
```

Render the Stage 4 Joint assembly:

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
- each independent expert
- frozen Router
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
3. complete global expert
4. complete local expert
5. complete contact expert
6. frozen volume-aware Router
7. controlled Joint
8. Router without pixel residual
9. Router without oracle-error distillation
10. Router without anti-starvation
11. legacy continuous `cams_gs_moe` ablation

The legacy residual-component checkpoints are intentionally incompatible with the version-2 complete-expert pipeline.

## Tests

Focused tests cover:

- absolute EndoNeRF paths and loader selection
- NumPy/Torch AABB boundaries
- complete expert architecture and gradients
- canonical/expert/Router bundle identity
- full deformation-state fingerprinting
- frozen expert and Router contracts
- real Router gradient flow through rasterized routing features
- camera-only Router/Joint rendering
- controlled Joint trainable-parameter whitelist and quality gates
- Python 3.7 syntax compatibility

Run:

```bash
python -m pytest tests -q
```
