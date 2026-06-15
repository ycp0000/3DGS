"""Refresh ``validation_metrics`` on an existing EndoMoe expert bundle.

Why this exists
---------------

The training loop used to mutate Gaussian topology on the final
iteration (densify / prune), *after* the last call to
``training_report``. The bundle was written immediately after the
loop ended, so ``validation_metrics`` recorded the pre-prune model
while ``expert_state`` recorded the post-prune model. Any consumer
that re-renders the bundle (notably the Stage 3 baseline parity gate
``endomoeg_residual_max_baseline_psnr_drop``) saw a much lower PSNR
than the bundle advertised and aborted with an opaque error message.

The structural fix lives in ``train.py``:

* the final iteration is now treated as topology-frozen,
* every bundle write re-evaluates metrics on the exact state being
  persisted via ``measure_bundle_metrics``.

That fix protects every *future* run. This script repairs runs whose
bundles were already written before the fix landed, without retraining
Stage 2.

Usage
-----

::

    python tools/refresh_bundle_metrics.py \\
        --bundle /root/autodl-tmp/endomoeg/cutting/bundles/global.pth \\
        --source /root/3DGS/data/endonerf/cutting

The script never modifies ``expert_state`` and never touches the
fingerprint fields. It only rewrites the ``validation_metrics`` block
to match what the saved state actually renders today on the standard
fixed-view subset.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from typing import Optional

import torch

# Allow the script to be run as ``python scripts/refresh_bundle_metrics.py``
# from the repository root without an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.endomoeg.ensemble import freeze_gaussian_model  # noqa: E402
from models.endomoeg.expert_bundle import (  # noqa: E402
    load_expert_bundle,
    save_bundle,
)
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.eval_utils import evaluate_fixed_view_metrics  # noqa: E402

try:
    import lpips as external_lpips  # type: ignore
except ImportError:  # pragma: no cover - environment-specific
    external_lpips = None

if external_lpips is None:
    from lpipsPyTorch import LPIPS as LocalLPIPS  # type: ignore
else:
    LocalLPIPS = None  # type: ignore


def _build_args_namespace(payload: dict, source_path: str) -> Namespace:
    config = payload.get("config") or {}
    model_params = config.get("model_params") or {}
    hidden_params = config.get("hidden_params") or {}
    if not model_params or not hidden_params:
        raise ValueError(
            "Bundle is missing reconstruction config; cannot rebuild "
            "GaussianModel for re-evaluation"
        )
    merged = {**model_params, **hidden_params}
    merged.setdefault("source_path", source_path)
    merged["source_path"] = os.path.abspath(merged["source_path"])
    merged.setdefault("images", "images")
    merged.setdefault("eval", True)
    merged.setdefault("white_background", False)
    merged.setdefault("model_path", "/tmp/endomoeg_refresh")
    merged["sh_degree"] = int(model_params.get("sh_degree", 3))
    merged.setdefault("render_process", False)
    return Namespace(**merged)


class _StubPipe:
    debug = False
    convert_SHs_python = False
    compute_cov3D_python = False


def _resolve_lpips(device: torch.device):
    if external_lpips is not None:
        return external_lpips.LPIPS(net="vgg").to(device)
    if LocalLPIPS is not None:
        return LocalLPIPS(net_type="vgg").to(device)
    return None


def refresh_bundle(
    bundle_path: str,
    source_path: str,
    expected_role: Optional[str] = None,
    output_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    bundle_path = os.path.abspath(bundle_path)
    payload = load_expert_bundle(
        bundle_path,
        map_location="cpu",
        expected_role=expected_role,
    )
    role = payload.get("role")
    print(f"[refresh] Loaded {role} bundle from {bundle_path}")

    args = _build_args_namespace(payload, source_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussians = GaussianModel(args.sh_degree, args)
    scene = Scene(args, gaussians, load_coarse=None)
    gaussians.restore_expert_state(payload["expert_state"], training_args=None)
    freeze_gaussian_model(gaussians)

    pipe = _StubPipe()
    background = torch.tensor(
        [1.0, 1.0, 1.0] if getattr(args, "white_background", False) else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    lpips_model = _resolve_lpips(device)

    metrics = evaluate_fixed_view_metrics(
        scene,
        gaussians,
        pipe,
        background,
        stage="fine",
        view_count=4,
        lpips_model=lpips_model,
        splits=("test",),
    )
    refreshed = metrics.get("test", {})
    if "psnr" not in refreshed:
        raise RuntimeError(
            "Re-evaluation produced no PSNR. Either the test split is empty "
            "or the bundle does not match the requested source path."
        )

    old_metrics = payload.get("validation_metrics") or {}
    print(
        "[refresh] Old validation_metrics: {}".format(
            {k: round(float(v), 4) for k, v in old_metrics.items()}
        )
    )
    print(
        "[refresh] New validation_metrics: {}".format(
            {k: round(float(v), 4) for k, v in refreshed.items()}
        )
    )

    if dry_run:
        print("[refresh] --dry-run set; not writing bundle")
        return refreshed

    target_path = os.path.abspath(output_path) if output_path else bundle_path
    payload["validation_metrics"] = refreshed
    save_bundle(target_path, payload)
    print(f"[refresh] Wrote refreshed bundle to {target_path}")
    return refreshed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh validation_metrics on an EndoMoe expert bundle "
        "to match the model state it actually persists."
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Absolute path to the expert bundle (.pth) to refresh.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Absolute path to the dataset the bundle was trained on.",
    )
    parser.add_argument(
        "--expected-role",
        default=None,
        choices=("global", "local", "contact"),
        help="If set, fail unless the bundle role matches.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional alternate output path. Defaults to overwriting --bundle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Re-evaluate and print metrics without writing the bundle back.",
    )
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    refresh_bundle(
        bundle_path=args.bundle,
        source_path=args.source,
        expected_role=args.expected_role,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
