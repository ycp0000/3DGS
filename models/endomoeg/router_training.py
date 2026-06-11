from collections import OrderedDict
from random import randint

import torch
from tqdm import tqdm

from gaussian_renderer import render, rasterize_endomoeg_routing_features
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.eval_utils import select_fixed_views

from .ensemble import FrozenExpertEnsemble
from .expert_bundle import EXPERT_ROLES
from .router import EndoMoeVolumeAwareRouter, compute_router_losses
from .router_bundle import build_router_bundle, save_router_bundle


def _gradient_norm(parameters):
    squared = None
    has_gradient = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        has_gradient = True
        value = parameter.grad.detach().float().square().sum()
        squared = value if squared is None else squared + value
    if not has_gradient:
        return 0.0
    return float(squared.sqrt().item())


def collect_router_gradient_metrics(router):
    return {
        "grad_norm_router_gaussian_logits": _gradient_norm(
            router.base_logits.parameters()
        ),
        "grad_norm_router_feature_mlp": _gradient_norm(
            router.gaussian_feature_mlp.parameters()
        ),
        "grad_norm_router_pixel": _gradient_norm(
            router.pixel_router.parameters()
        ),
    }


def assert_router_gradient_contract(metrics):
    missing = [
        name
        for name, value in metrics.items()
        if not torch.isfinite(torch.tensor(value)) or float(value) <= 0.0
    ]
    if missing:
        raise RuntimeError(
            "EndoMoe Router gradient contract failed: {}".format(
                ", ".join(missing)
            )
        )


def render_frozen_expert_ensemble(
    viewpoint,
    ensemble,
    router,
    pipe,
    background,
    top_k=None,
):
    expert_images = []
    expert_depths = []
    gaussian_priors = []
    projected_motions = []
    coverage_maps = []
    per_expert = OrderedDict()
    for role, expert in ensemble:
        package = render(
            viewpoint,
            expert,
            pipe,
            background,
            stage="fine",
            update_deformation_stats=False,
            return_routing_state=True,
        )
        routing_state = package["routing_state"]
        gaussian_logits = router.gaussian_logits(
            role,
            routing_state,
            viewpoint,
            time_value=float(viewpoint.time),
        )
        routing_maps = rasterize_endomoeg_routing_features(
            viewpoint,
            expert,
            pipe,
            routing_state,
            gaussian_logits,
        )
        expert_images.append(package["render"])
        expert_depths.append(package["depth"].squeeze(0))
        gaussian_priors.append(routing_maps["gaussian_prior"])
        projected_motions.append(routing_maps["projected_motion"])
        coverage_maps.append(routing_maps["coverage"])
        per_expert[role] = {
            "render": package["render"],
            "depth": package["depth"],
            "routing_state": routing_state,
            "gaussian_logits": gaussian_logits,
            "routing_maps": routing_maps,
        }

    expert_rgb = torch.stack(expert_images, dim=0)
    expert_depth = torch.stack(expert_depths, dim=0)
    gaussian_prior = torch.stack(gaussian_priors, dim=0)
    projected_motion = torch.stack(projected_motions, dim=0)
    coverage = torch.stack(coverage_maps, dim=0) > 1e-8
    weights, residual_logits = router.route_pixels(
        expert_rgb=expert_rgb,
        expert_depth=expert_depth,
        gaussian_prior=gaussian_prior,
        projected_motion=projected_motion,
        coverage=coverage,
        top_k=top_k,
    )
    blended_image = (expert_rgb * weights.unsqueeze(1)).sum(dim=0)
    blended_depth = (expert_depth * weights).sum(dim=0, keepdim=True)
    return {
        "render": blended_image,
        "depth": blended_depth,
        "weights": weights,
        "pixel_router_residual_logits": residual_logits,
        "expert_rgb": expert_rgb,
        "expert_depth": expert_depth,
        "gaussian_prior": gaussian_prior,
        "projected_motion": projected_motion,
        "coverage": coverage,
        "per_expert": per_expert,
    }


def evaluate_frozen_router(
    scene,
    ensemble,
    router,
    pipe,
    background,
    top_k,
):
    cameras = select_fixed_views(scene.getTestCameras(), count=4)
    if not cameras:
        return {}
    totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    device = background.device
    with torch.no_grad():
        for viewpoint in cameras:
            output = render_frozen_expert_ensemble(
                viewpoint,
                ensemble,
                router,
                pipe,
                background,
                top_k=top_k,
            )
            image = output["render"].clamp(0.0, 1.0).unsqueeze(0)
            ground_truth = viewpoint.original_image.to(device).float().unsqueeze(0)
            mask = viewpoint.mask.to(device).unsqueeze(0)
            totals["l1"] += float(l1_loss(image, ground_truth, mask).item())
            totals["psnr"] += float(
                psnr(image, ground_truth, mask).mean().item()
            )
            totals["ssim"] += float(
                ssim(image * mask, ground_truth * mask).item()
            )
    return {
        name: value / float(len(cameras))
        for name, value in totals.items()
    }


def train_frozen_router(
    dataset,
    hyper,
    opt,
    pipe,
    scene,
    testing_iterations,
    tb_writer,
    config,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    ensemble = FrozenExpertEnsemble.load(
        hyper.endomoeg_bundle_dir,
        minimum_psnr=float(hyper.endomoeg_min_expert_psnr),
        device=device,
        expected_source_path=dataset.source_path,
    )
    ensemble.assert_frozen()
    router = EndoMoeVolumeAwareRouter(
        ensemble.point_counts(),
        gaussian_hidden_dim=int(hyper.moe_router_hidden_dim),
        pixel_hidden_dim=int(hyper.moe_pixel_router_hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": router.base_logits.parameters(),
                "lr": float(hyper.endomoeg_router_gaussian_lr),
                "name": "router_gaussian_logits",
            },
            {
                "params": list(router.role_embedding.parameters())
                + list(router.gaussian_feature_mlp.parameters()),
                "lr": float(hyper.endomoeg_router_feature_lr),
                "name": "router_feature_mlp",
            },
            {
                "params": router.pixel_router.parameters(),
                "lr": float(hyper.endomoeg_router_pixel_lr),
                "name": "router_pixel",
            },
        ],
        eps=1e-15,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    train_cameras = scene.getTrainCameras()
    final_metrics = {}
    progress = tqdm(range(1, int(opt.iterations) + 1), desc="Router training")
    for iteration in progress:
        viewpoint = train_cameras[randint(0, len(train_cameras) - 1)]
        sparse_start = min(
            max(float(hyper.endomoeg_router_sparse_start), 0.0),
            1.0,
        )
        top_k = (
            2
            if iteration / max(int(opt.iterations), 1) >= sparse_start
            else None
        )
        output = render_frozen_expert_ensemble(
            viewpoint,
            ensemble,
            router,
            pipe,
            background,
            top_k=top_k,
        )
        ground_truth = viewpoint.original_image.to(device).float()
        mask = viewpoint.mask.to(device)
        blended, _, loss_dict = compute_router_losses(
            output["weights"],
            output["expert_rgb"],
            ground_truth,
            mask=mask,
            oracle_temperature=float(
                hyper.endomoeg_router_oracle_temperature
            ),
            lambda_oracle=float(hyper.endomoeg_router_lambda_oracle),
            lambda_starvation=float(
                hyper.endomoeg_router_lambda_starvation
            ),
        )
        dssim_loss = 1.0 - ssim(
            (blended * mask).unsqueeze(0),
            (ground_truth * mask).unsqueeze(0),
        )
        total_loss = (
            loss_dict["L_router_total"]
            + float(hyper.endomoeg_router_lambda_dssim) * dssim_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_metrics = collect_router_gradient_metrics(router)
        warmup = int(hyper.endomoeg_router_gradient_warmup)
        if iteration == max(warmup, 1):
            assert_router_gradient_contract(gradient_metrics)
        optimizer.step()
        ensemble.assert_frozen()

        if iteration % 10 == 0:
            progress.set_postfix(
                loss="{:.6f}".format(float(total_loss.detach().item())),
                psnr="{:.2f}".format(
                    float(
                        psnr(
                            blended.unsqueeze(0),
                            ground_truth.unsqueeze(0),
                            mask.unsqueeze(0),
                        )
                        .mean()
                        .item()
                    )
                ),
                topk="dense" if top_k is None else str(top_k),
            )
        if tb_writer is not None:
            tb_writer.add_scalar(
                "router/train/L_total",
                float(total_loss.detach().item()),
                iteration,
            )
            tb_writer.add_scalar(
                "router/train/L_dssim",
                float(dssim_loss.detach().item()),
                iteration,
            )
            tb_writer.add_scalar(
                "router/train/pixel_residual_abs_mean",
                float(
                    output["pixel_router_residual_logits"]
                    .detach()
                    .abs()
                    .mean()
                    .item()
                ),
                iteration,
            )
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    tb_writer.add_scalar(
                        "router/train/{}".format(name),
                        float(value.detach().item()),
                        iteration,
                    )
            for name, value in gradient_metrics.items():
                tb_writer.add_scalar(
                    "router/gradients/{}".format(name),
                    value,
                    iteration,
                )

        if iteration in testing_iterations:
            final_metrics = evaluate_frozen_router(
                scene,
                ensemble,
                router,
                pipe,
                background,
                top_k=top_k,
            )
            if final_metrics:
                print(
                    "\n[ITER {}] Router validation: L1 {:.6f} "
                    "PSNR {:.3f} SSIM {:.4f}".format(
                        iteration,
                        final_metrics["l1"],
                        final_metrics["psnr"],
                        final_metrics["ssim"],
                    )
                )
                if tb_writer is not None:
                    for name, value in final_metrics.items():
                        tb_writer.add_scalar(
                            "router/validation/test/{}".format(name),
                            value,
                            iteration,
                        )

    if "psnr" not in final_metrics:
        raise RuntimeError(
            "Final Router fixed-view validation PSNR was not computed"
        )
    payload = build_router_bundle(
        router,
        ensemble,
        iteration=opt.iterations,
        config=config,
        validation_metrics=final_metrics,
    )
    path = save_router_bundle(
        "{}/router.pth".format(hyper.endomoeg_bundle_dir),
        payload,
    )
    print("[EndoMoe] Saved frozen-expert Router bundle to {}".format(path))
    return router, ensemble, final_metrics
