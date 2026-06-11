from collections import OrderedDict
from random import randint

import torch
from tqdm import tqdm

from gaussian_renderer import (
    rasterize_endomoeg_composite_state,
    rasterize_endomoeg_routing_features,
    render,
)
from utils.eval_utils import select_fixed_views
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

from .ensemble import FrozenExpertEnsemble
from .router import (
    RESIDUAL_ROLES,
    EndoMoeVolumeAwareRouter,
    compose_residual_gaussian_state,
    compute_router_losses,
)
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
        "grad_norm_router_base_gates": _gradient_norm((router.base_gates,)),
        "grad_norm_router_feature_mlp": _gradient_norm(
            router.gaussian_feature_mlp.parameters()
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


def _render_expert_states(viewpoint, ensemble, pipe, background):
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
        per_expert[role] = {
            "expert": expert,
            "render": package["render"],
            "depth": package["depth"],
            "routing_state": package["routing_state"],
        }
    return per_expert


def _compose_and_render(
    viewpoint,
    per_expert,
    gates,
    pipe,
    background,
):
    state = compose_residual_gaussian_state(
        per_expert["global"]["routing_state"],
        per_expert["local"]["routing_state"],
        per_expert["contact"]["routing_state"],
        gates,
    )
    package = rasterize_endomoeg_composite_state(
        viewpoint,
        per_expert["global"]["expert"],
        pipe,
        background,
        state,
    )
    package["composite_state"] = state
    return package


def render_frozen_expert_ensemble(
    viewpoint,
    ensemble,
    router,
    pipe,
    background,
):
    per_expert = _render_expert_states(
        viewpoint,
        ensemble,
        pipe,
        background,
    )
    global_state = per_expert["global"]["routing_state"]
    local_state = per_expert["local"]["routing_state"]
    contact_state = per_expert["contact"]["routing_state"]
    gates, raw_gates = router.residual_gates(
        global_state,
        local_state,
        contact_state,
        viewpoint,
        time_value=float(viewpoint.time),
    )
    composite = _compose_and_render(
        viewpoint,
        per_expert,
        gates,
        pipe,
        background,
    )
    gate_maps = []
    for index in range(len(RESIDUAL_ROLES)):
        gate_render = rasterize_endomoeg_routing_features(
            viewpoint,
            per_expert["global"]["expert"],
            pipe,
            global_state,
            gates[:, index],
            probabilities=True,
        )
        gate_maps.append(gate_render["gaussian_prior"].clamp(0.0, 1.0))
    return {
        "render": composite["render"],
        "depth": composite["depth"],
        "gates": gates,
        "raw_gates": raw_gates,
        "gate_maps": torch.stack(gate_maps, dim=0),
        "candidate_rgb": {
            "global": per_expert["global"]["render"],
            "local": per_expert["local"]["render"],
            "contact": per_expert["contact"]["render"],
        },
        "per_expert": per_expert,
        "composite_state": composite["composite_state"],
    }


def _masked_psnr(image, ground_truth, mask):
    return float(
        psnr(
            image.unsqueeze(0),
            ground_truth.unsqueeze(0),
            mask.unsqueeze(0),
        )
        .mean()
        .item()
    )


def evaluate_router_headroom(scene, ensemble, pipe, background):
    cameras = select_fixed_views(scene.getTestCameras(), count=4)
    if not cameras:
        return {}
    totals = OrderedDict(
        (name, 0.0)
        for name in ("global", "local", "contact", "full", "oracle")
    )
    with torch.no_grad():
        for viewpoint in cameras:
            per_expert = _render_expert_states(
                viewpoint,
                ensemble,
                pipe,
                background,
            )
            parent_count = int(
                per_expert["global"]["routing_state"]["base_point_count"]
            )
            full = _compose_and_render(
                viewpoint,
                per_expert,
                torch.ones(
                    parent_count,
                    2,
                    device=background.device,
                ),
                pipe,
                background,
            )["render"]
            ground_truth = viewpoint.original_image.to(background.device).float()
            mask = viewpoint.mask.to(background.device)
            candidates = torch.stack(
                (
                    per_expert["global"]["render"],
                    per_expert["local"]["render"],
                    per_expert["contact"]["render"],
                    full,
                ),
                dim=0,
            )
            errors = (candidates - ground_truth.unsqueeze(0)).abs().mean(dim=1)
            oracle_indices = errors.argmin(dim=0)
            oracle = torch.gather(
                candidates,
                0,
                oracle_indices.unsqueeze(0).unsqueeze(1).expand(
                    1,
                    3,
                    -1,
                    -1,
                ),
            ).squeeze(0)
            named_images = {
                "global": candidates[0],
                "local": candidates[1],
                "contact": candidates[2],
                "full": candidates[3],
                "oracle": oracle,
            }
            for name, image in named_images.items():
                totals[name] += _masked_psnr(image, ground_truth, mask)
    return {
        name: value / float(len(cameras))
        for name, value in totals.items()
    }


def assert_router_headroom(metrics, minimum_gain):
    if not metrics:
        raise RuntimeError("Router headroom requires fixed test views")
    gain = float(metrics["oracle"]) - float(metrics["global"])
    if gain < float(minimum_gain):
        raise RuntimeError(
            "Residual experts provide only {:.4f} dB oracle headroom; "
            "required {:.4f} dB".format(gain, float(minimum_gain))
        )


def evaluate_frozen_router(
    scene,
    ensemble,
    router,
    pipe,
    background,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": (router.base_gates,),
                "lr": float(hyper.endomoeg_router_gaussian_lr),
                "name": "router_base_gates",
            },
            {
                "params": router.gaussian_feature_mlp.parameters(),
                "lr": float(hyper.endomoeg_router_feature_lr),
                "name": "router_feature_mlp",
            },
        ],
        eps=1e-15,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    headroom = evaluate_router_headroom(
        scene,
        ensemble,
        pipe,
        background,
    )
    assert_router_headroom(
        headroom,
        minimum_gain=float(hyper.endomoeg_min_oracle_headroom),
    )
    if tb_writer is not None:
        for name, value in headroom.items():
            tb_writer.add_scalar(
                "router/headroom/psnr_{}".format(name),
                value,
                0,
            )
    train_cameras = scene.getTrainCameras()
    final_metrics = {}
    progress = tqdm(range(1, int(opt.iterations) + 1), desc="Router training")
    for iteration in progress:
        viewpoint = train_cameras[randint(0, len(train_cameras) - 1)]
        output = render_frozen_expert_ensemble(
            viewpoint,
            ensemble,
            router,
            pipe,
            background,
        )
        ground_truth = viewpoint.original_image.to(device).float()
        mask = viewpoint.mask.to(device)
        loss_dict = compute_router_losses(
            output["render"],
            output["candidate_rgb"]["global"],
            {
                role: output["candidate_rgb"][role]
                for role in RESIDUAL_ROLES
            },
            output["gates"],
            output["gate_maps"],
            ground_truth,
            mask=mask,
            gain_temperature=float(hyper.endomoeg_router_gain_temperature),
            lambda_gain=float(hyper.endomoeg_router_lambda_gain),
            lambda_sparsity=float(hyper.endomoeg_router_lambda_sparsity),
            lambda_no_regret=float(hyper.endomoeg_router_lambda_no_regret),
        )
        dssim_loss = 1.0 - ssim(
            (output["render"] * mask).unsqueeze(0),
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
                    _masked_psnr(output["render"], ground_truth, mask)
                ),
                gL="{:.3f}".format(
                    float(output["gates"][:, 0].mean().detach().item())
                ),
                gC="{:.3f}".format(
                    float(output["gates"][:, 1].mean().detach().item())
                ),
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
            )
            if tb_writer is not None:
                for name, value in final_metrics.items():
                    tb_writer.add_scalar(
                        "router/validation/test/{}".format(name),
                        value,
                        iteration,
                    )

    if not final_metrics:
        final_metrics = evaluate_frozen_router(
            scene,
            ensemble,
            router,
            pipe,
            background,
        )
    payload = build_router_bundle(
        router,
        ensemble,
        iteration=int(opt.iterations),
        config=config,
        validation_metrics=final_metrics,
    )
    save_router_bundle(
        hyper.endomoeg_router_bundle,
        payload,
    )
    return router, final_metrics
