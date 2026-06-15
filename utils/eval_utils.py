"""Evaluation helpers shared by training, bundling, Router, and Joint stages.

The single most important contract in this module is that
``evaluate_fixed_view_metrics`` is a stateless, deterministic measurement
of the *current* model on the same fixed test views that all bundle gates
use. Every bundle write must call this helper *after* topology updates
have settled and *on the same model state* it is about to persist. This
prevents the "metric reflects pre-prune model, state reflects post-prune
model" class of bug that turns Stage 3 into an opaque RuntimeError.
"""

from typing import Any, Dict, Optional, Sequence, Tuple, TypeVar

import torch


CameraT = TypeVar("CameraT")


def select_fixed_views(
    cameras: Sequence[CameraT],
    count: int = 4,
) -> Tuple[CameraT, ...]:
    camera_count = len(cameras)
    if camera_count == 0 or count <= 0:
        return ()
    if camera_count <= count:
        return tuple(cameras)
    indices = [
        round(index * (camera_count - 1) / (count - 1))
        for index in range(count)
    ]
    return tuple(cameras[index] for index in indices)


@torch.no_grad()
def evaluate_fixed_view_metrics(
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    *,
    stage: str = "fine",
    view_count: int = 4,
    lpips_model: Optional[Any] = None,
    splits: Tuple[str, ...] = ("test",),
) -> Dict[str, Dict[str, float]]:
    """Render fixed views with the current model and return masked metrics.

    The caller controls *which* model is evaluated by passing it in as
    ``gaussians``. The function renders the same fixed views that the
    rest of the pipeline uses (``select_fixed_views(..., count=4)``),
    applies the project-standard masked PSNR / SSIM / LPIPS, and returns
    a flat ``{split: {metric_name: value}}`` mapping.

    This is the single source of truth that all bundle writes must use.
    Do not maintain a parallel measurement path.
    """
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import l1_loss, lpips_loss, ssim

    if not splits:
        return {}

    device = background.device

    split_camera_sources = {
        "test": getattr(scene, "getTestCameras", None),
        "train": getattr(scene, "getTrainCameras", None),
    }

    results: Dict[str, Dict[str, float]] = {}
    for split_name in splits:
        getter = split_camera_sources.get(split_name)
        if getter is None:
            continue
        cameras = select_fixed_views(getter(), count=view_count)
        if not cameras:
            continue
        totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0, "lpips": 0.0}
        for viewpoint in cameras:
            package = render(
                viewpoint,
                gaussians,
                pipe,
                background,
                stage=stage,
                update_deformation_stats=False,
            )
            image = package["render"].clamp(0.0, 1.0).unsqueeze(0)
            ground_truth = (
                viewpoint.original_image.to(device).float().clamp(0.0, 1.0).unsqueeze(0)
            )
            mask = viewpoint.mask.to(device).unsqueeze(0)
            masked_image = image * mask
            masked_gt = ground_truth * mask

            totals["l1"] += float(l1_loss(image, ground_truth, mask).item())
            totals["psnr"] += float(
                psnr(image, ground_truth, mask).mean().item()
            )
            totals["ssim"] += float(ssim(masked_image, masked_gt).item())
            if lpips_model is not None:
                totals["lpips"] += float(
                    lpips_loss(masked_image, masked_gt, lpips_model).item()
                )
        camera_count = float(len(cameras))
        results[split_name] = {
            name: value / camera_count for name, value in totals.items()
        }
    return results


def measure_bundle_metrics(
    scene,
    gaussians,
    pipe,
    background: torch.Tensor,
    lpips_model: Optional[Any] = None,
) -> Dict[str, float]:
    """Re-evaluate fixed-view metrics on the *current* model state.

    Every bundle write must obtain ``validation_metrics`` through this
    helper, after the training loop has finished and after best-state
    restoration, on the model that will actually be persisted. Calling
    it earlier (e.g. relying on a metric collected mid-loop) creates a
    metric/state mismatch that downstream parity gates surface as an
    opaque RuntimeError two stages later.
    """
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
    return metrics.get("test", {})
