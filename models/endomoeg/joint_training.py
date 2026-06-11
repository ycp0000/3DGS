import os
from collections import OrderedDict
from random import randint

import torch
from tqdm import tqdm

from gaussian_renderer import render
from utils.eval_utils import select_fixed_views
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

from .ensemble import freeze_gaussian_model
from .expert_bundle import EXPERT_ROLES, build_expert_bundle, save_bundle
from .inference import load_frozen_router_assembly
from .router import compute_router_losses
from .router_bundle import build_router_bundle, save_router_bundle
from .router_training import (
    assert_router_gradient_contract,
    collect_router_gradient_metrics,
    evaluate_frozen_router,
    render_frozen_expert_ensemble,
)


_CANONICAL_PARAMETER_NAMES = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
)


def _parameter_list(parameters):
    return [parameter for parameter in parameters]


def configure_joint_trainable_parameters(assembly, hyper):
    router = assembly.router
    ensemble = assembly.ensemble
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    router.train()

    groups = [
        {
            "params": _parameter_list(router.base_logits.parameters()),
            "lr": float(hyper.endomoeg_joint_router_gaussian_lr),
            "name": "joint_router_gaussian_logits",
        },
        {
            "params": _parameter_list(router.role_embedding.parameters())
            + _parameter_list(router.gaussian_feature_mlp.parameters()),
            "lr": float(hyper.endomoeg_joint_router_feature_lr),
            "name": "joint_router_feature_mlp",
        },
        {
            "params": _parameter_list(router.pixel_router.parameters()),
            "lr": float(hyper.endomoeg_joint_router_pixel_lr),
            "name": "joint_router_pixel",
        },
    ]
    expert_groups = OrderedDict()
    for role, expert in ensemble:
        for name in _CANONICAL_PARAMETER_NAMES:
            getattr(expert, name).requires_grad_(False)
        for parameter in expert._deformation.parameters():
            parameter.requires_grad_(False)
        expert._deformation.eval()

        if role == "global":
            parameters = _parameter_list(expert._deformation.parameters())
            for parameter in parameters:
                parameter.requires_grad_(True)
            expert._deformation.train()
            learning_rate = float(
                hyper.endomoeg_joint_global_deformation_lr
            )
            group_name = "joint_expert_global_deformation"
        else:
            refinement = (
                expert._deformation.deformation_net.complete_expert_head.refinement
            )
            if refinement is None:
                raise RuntimeError(
                    "Joint expert '{}' has no refinement module".format(role)
                )
            parameters = _parameter_list(refinement.parameters())
            for parameter in parameters:
                parameter.requires_grad_(True)
            refinement.train()
            learning_rate = float(hyper.endomoeg_joint_refinement_lr)
            group_name = "joint_expert_{}_refinement".format(role)
        if not parameters:
            raise RuntimeError(
                "Joint expert '{}' has no trainable parameters".format(role)
            )
        expert_groups[role] = parameters
        groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "name": group_name,
            }
        )
    assert_joint_trainable_contract(assembly)
    return groups, expert_groups


def assert_joint_trainable_contract(assembly):
    for role, expert in assembly.ensemble:
        canonical_trainable = [
            name
            for name in _CANONICAL_PARAMETER_NAMES
            if getattr(expert, name).requires_grad
        ]
        if canonical_trainable:
            raise RuntimeError(
                "Joint stage must freeze canonical expert parameters: "
                "{}".format(", ".join(canonical_trainable))
            )
        trainable_deformation = [
            name
            for name, parameter in expert._deformation.named_parameters()
            if parameter.requires_grad
        ]
        if role == "global":
            if not trainable_deformation:
                raise RuntimeError(
                    "Joint global expert deformation is not trainable"
                )
        else:
            invalid = [
                name
                for name in trainable_deformation
                if "complete_expert_head.refinement" not in name
            ]
            if invalid or not trainable_deformation:
                raise RuntimeError(
                    "Joint '{}' expert may train refinement only".format(role)
                )


def capture_parameter_anchors(optimizer_groups):
    anchors = OrderedDict()
    for group in optimizer_groups:
        anchors[group["name"]] = [
            parameter.detach().clone()
            for parameter in group["params"]
        ]
    return anchors


def parameter_anchor_loss(optimizer_groups, anchors):
    total = None
    element_count = 0
    for group in optimizer_groups:
        group_anchors = anchors[group["name"]]
        for parameter, anchor in zip(group["params"], group_anchors):
            value = (parameter - anchor).square().sum()
            total = value if total is None else total + value
            element_count += parameter.numel()
    if total is None or element_count == 0:
        raise RuntimeError("Joint parameter anchor set is empty")
    return total / float(element_count)


def collect_joint_expert_gradient_metrics(expert_groups):
    metrics = OrderedDict()
    for role, parameters in expert_groups.items():
        squared = None
        has_gradient = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            has_gradient = True
            value = parameter.grad.detach().float().square().sum()
            squared = value if squared is None else squared + value
        metrics["grad_norm_joint_expert_{}".format(role)] = (
            float(squared.sqrt().item()) if has_gradient else 0.0
        )
    return metrics


def assert_joint_expert_gradient_contract(metrics):
    invalid = [
        name
        for name, value in metrics.items()
        if not torch.isfinite(torch.tensor(value)) or float(value) <= 0.0
    ]
    if invalid:
        raise RuntimeError(
            "EndoMoe joint expert gradient contract failed: {}".format(
                ", ".join(invalid)
            )
        )


def evaluate_individual_experts(scene, ensemble, pipe, background):
    cameras = select_fixed_views(scene.getTestCameras(), count=4)
    if not cameras:
        return {}
    device = background.device
    results = OrderedDict()
    with torch.no_grad():
        for role, expert in ensemble:
            totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
            for viewpoint in cameras:
                package = render(
                    viewpoint,
                    expert,
                    pipe,
                    background,
                    stage="fine",
                    update_deformation_stats=False,
                )
                image = package["render"].clamp(0.0, 1.0).unsqueeze(0)
                ground_truth = (
                    viewpoint.original_image.to(device).float().unsqueeze(0)
                )
                mask = viewpoint.mask.to(device).unsqueeze(0)
                totals["l1"] += float(
                    l1_loss(image, ground_truth, mask).item()
                )
                totals["psnr"] += float(
                    psnr(image, ground_truth, mask).mean().item()
                )
                totals["ssim"] += float(
                    ssim(image * mask, ground_truth * mask).item()
                )
            results[role] = {
                name: value / float(len(cameras))
                for name, value in totals.items()
            }
    return results


def _assert_joint_quality_gate(
    assembly,
    ensemble_metrics,
    expert_metrics,
    max_psnr_drop,
):
    baseline_router_psnr = float(
        assembly.payload["validation_metrics"]["psnr"]
    )
    if float(ensemble_metrics["psnr"]) < baseline_router_psnr - max_psnr_drop:
        raise RuntimeError(
            "Joint Router PSNR {:.4f} degraded below parent {:.4f}".format(
                float(ensemble_metrics["psnr"]),
                baseline_router_psnr,
            )
        )
    for role in EXPERT_ROLES:
        baseline = float(
            assembly.ensemble.payloads[role]["validation_metrics"]["psnr"]
        )
        current = float(expert_metrics[role]["psnr"])
        if current < baseline - max_psnr_drop:
            raise RuntimeError(
                "Joint '{}' expert PSNR {:.4f} degraded below parent "
                "{:.4f}".format(role, current, baseline)
            )


def _save_joint_assembly(
    assembly,
    output_dir,
    iteration,
    config,
    ensemble_metrics,
    expert_metrics,
):
    os.makedirs(output_dir, exist_ok=True)
    updated_payloads = OrderedDict()
    for role, expert in assembly.ensemble:
        parent_payload = assembly.ensemble.payloads[role]
        expert_config = dict(parent_payload.get("config") or {})
        expert_config["joint_finetune"] = {
            "parent_expert_state_fingerprint": parent_payload[
                "expert_state_fingerprint"
            ],
            "parent_router_bundle": assembly.bundle_path,
        }
        payload = build_expert_bundle(
            expert,
            role=role,
            source_canonical_fingerprint=(
                assembly.ensemble.source_canonical_fingerprint
            ),
            iteration=iteration,
            config=expert_config,
            validation_metrics=expert_metrics[role],
        )
        save_bundle(
            os.path.join(output_dir, "{}.pth".format(role)),
            payload,
        )
        updated_payloads[role] = payload

    for _, expert in assembly.ensemble:
        freeze_gaussian_model(expert)
    for parameter in assembly.router.parameters():
        parameter.requires_grad_(False)
    assembly.router.eval()
    assembly.ensemble.payloads = updated_payloads
    assembly.ensemble.assert_frozen()

    router_config = dict(config or {})
    router_config["joint_finetune"] = {
        "parent_router_bundle": assembly.bundle_path,
        "parent_expert_state_fingerprints": {
            role: assembly.payload["expert_manifest"][role][
                "expert_state_fingerprint"
            ]
            for role in EXPERT_ROLES
        },
    }
    router_payload = build_router_bundle(
        assembly.router,
        assembly.ensemble,
        iteration=iteration,
        config=router_config,
        validation_metrics=ensemble_metrics,
        inference_top_k=assembly.top_k,
    )
    router_path = save_router_bundle(
        os.path.join(output_dir, "router.pth"),
        router_payload,
    )
    return router_path


def train_controlled_joint(
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
    assembly = load_frozen_router_assembly(
        hyper.endomoeg_bundle_dir,
        expected_source_path=dataset.source_path,
        device=device,
        minimum_expert_psnr=float(hyper.endomoeg_min_expert_psnr),
        router_bundle_path=(
            getattr(hyper, "endomoeg_router_bundle", "") or None
        ),
    )
    optimizer_groups, expert_groups = configure_joint_trainable_parameters(
        assembly,
        hyper,
    )
    anchors = capture_parameter_anchors(optimizer_groups)
    optimizer = torch.optim.Adam(
        optimizer_groups,
        eps=1e-15,
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    train_cameras = scene.getTrainCameras()
    progress = tqdm(
        range(1, int(opt.iterations) + 1),
        desc="Controlled joint fine-tuning",
    )
    final_metrics = {}
    for iteration in progress:
        viewpoint = train_cameras[randint(0, len(train_cameras) - 1)]
        sparse_start = min(
            max(float(hyper.endomoeg_joint_sparse_start), 0.0),
            1.0,
        )
        top_k = (
            assembly.top_k
            if iteration / max(int(opt.iterations), 1) >= sparse_start
            else None
        )
        output = render_frozen_expert_ensemble(
            viewpoint,
            assembly.ensemble,
            assembly.router,
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
        anchor_loss = parameter_anchor_loss(optimizer_groups, anchors)
        total_loss = (
            loss_dict["L_router_total"]
            + float(hyper.endomoeg_router_lambda_dssim) * dssim_loss
            + float(hyper.endomoeg_joint_anchor_lambda) * anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        router_gradients = collect_router_gradient_metrics(assembly.router)
        expert_gradients = collect_joint_expert_gradient_metrics(expert_groups)
        warmup = max(int(hyper.endomoeg_router_gradient_warmup), 1)
        if iteration == warmup:
            assert_router_gradient_contract(router_gradients)
            assert_joint_expert_gradient_contract(expert_gradients)
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer_groups
                for parameter in group["params"]
            ],
            max_norm=float(hyper.endomoeg_joint_gradient_clip),
        )
        optimizer.step()
        assert_joint_trainable_contract(assembly)

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
            )
        if tb_writer is not None:
            tb_writer.add_scalar(
                "joint/train/L_total",
                float(total_loss.detach().item()),
                iteration,
            )
            tb_writer.add_scalar(
                "joint/train/L_anchor",
                float(anchor_loss.detach().item()),
                iteration,
            )
            tb_writer.add_scalar(
                "joint/train/L_dssim",
                float(dssim_loss.detach().item()),
                iteration,
            )
            for name, value in loss_dict.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    tb_writer.add_scalar(
                        "joint/train/{}".format(name),
                        float(value.detach().item()),
                        iteration,
                    )
            for name, value in list(router_gradients.items()) + list(
                expert_gradients.items()
            ):
                tb_writer.add_scalar(
                    "joint/gradients/{}".format(name),
                    value,
                    iteration,
                )

        if iteration in testing_iterations:
            final_metrics = evaluate_frozen_router(
                scene,
                assembly.ensemble,
                assembly.router,
                pipe,
                background,
                top_k=assembly.top_k,
            )
            if tb_writer is not None:
                for name, value in final_metrics.items():
                    tb_writer.add_scalar(
                        "joint/validation/test/{}".format(name),
                        value,
                        iteration,
                    )

    final_metrics = evaluate_frozen_router(
        scene,
        assembly.ensemble,
        assembly.router,
        pipe,
        background,
        top_k=assembly.top_k,
    )
    expert_metrics = evaluate_individual_experts(
        scene,
        assembly.ensemble,
        pipe,
        background,
    )
    if "psnr" not in final_metrics or tuple(expert_metrics.keys()) != EXPERT_ROLES:
        raise RuntimeError("Joint final fixed-view validation is incomplete")
    _assert_joint_quality_gate(
        assembly,
        final_metrics,
        expert_metrics,
        max_psnr_drop=float(hyper.endomoeg_joint_max_psnr_drop),
    )
    router_path = _save_joint_assembly(
        assembly,
        output_dir=hyper.endomoeg_joint_output_dir,
        iteration=opt.iterations,
        config=config,
        ensemble_metrics=final_metrics,
        expert_metrics=expert_metrics,
    )
    print("[EndoMoe] Saved controlled joint assembly to {}".format(router_path))
    return assembly, final_metrics, expert_metrics
