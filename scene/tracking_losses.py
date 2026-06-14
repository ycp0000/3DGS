from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch


DEFAULT_GEO_EXPERT_NAMES = ("static", "hexplane", "local", "smooth")
DEFAULT_VIS_EXPERT_NAMES = ("stable", "transient")


def _safe_mean(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return value.mean()


def _get_float_arg(args, name: str, default: float) -> float:
    value = getattr(args, name, default)
    if value is None:
        return float(default)
    value = float(value)
    if not math.isfinite(value):
        return float(default)
    return value


def _get_aux_tensor(aux: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
    value = aux.get(name)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _get_aux_scale(
    aux: Dict[str, torch.Tensor],
    name: str,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = aux.get(name, 1.0)
    if isinstance(value, torch.Tensor):
        value = _safe_mean(value).to(device=reference.device, dtype=reference.dtype)
    else:
        value = reference.new_tensor(float(value))
    return value.clamp(0.0, 1.0)


def _add_scaffold_regularization(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
) -> None:
    terms = (
        ("scaffold_arap", "L_scaffold_arap", "lambda_scaffold_arap", 1e-2),
        (
            "scaffold_acceleration",
            "L_scaffold_acceleration",
            "lambda_scaffold_acceleration",
            1e-3,
        ),
        (
            "scaffold_node_offset",
            "L_scaffold_node_offset",
            "lambda_scaffold_node_offset",
            1e-3,
        ),
        (
            "scaffold_gate_sparsity",
            "L_scaffold_gate_sparsity",
            "lambda_scaffold_gate_sparsity",
            1e-3,
        ),
    )
    for raw_name, loss_name, weight_name, default_weight in terms:
        raw_value = _get_aux_tensor(aux, raw_name)
        if raw_value is None:
            continue
        mean_value = _safe_mean(raw_value)
        losses[raw_name] = mean_value.detach()
        losses[loss_name] = mean_value * _get_float_arg(
            args,
            weight_name,
            default_weight,
        )
    node_translation = _get_aux_tensor(
        aux,
        "scaffold_node_translation_norm",
    )
    if node_translation is not None:
        losses["scaffold_node_translation_norm"] = _safe_mean(
            node_translation
        ).detach()
    mean_radius = _get_aux_tensor(aux, "scaffold_mean_radius")
    if mean_radius is not None:
        losses["scaffold_mean_radius"] = _safe_mean(mean_radius).detach()
    for name in (
        "scaffold_point_gate_mean",
        "scaffold_spatial_support_mean",
    ):
        value = _get_aux_tensor(aux, name)
        if value is not None:
            losses[name] = _safe_mean(value).detach()


def _add_contact_bank_regularization(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
) -> None:
    terms = (
        (
            "contact_bank_sparsity",
            "L_contact_bank_sparsity",
            "lambda_contact_bank_sparsity",
            1e-4,
        ),
        (
            "contact_bank_locality",
            "L_contact_bank_locality",
            "lambda_contact_bank_locality",
            1e-3,
        ),
        (
            "contact_bank_acceleration",
            "L_contact_bank_acceleration",
            "lambda_contact_bank_acceleration",
            1e-3,
        ),
        (
            "contact_bank_spatial_offset",
            "L_contact_bank_spatial_offset",
            "lambda_contact_bank_spatial_offset",
            1e-3,
        ),
        (
            "contact_bank_duration",
            "L_contact_bank_duration",
            "lambda_contact_bank_duration",
            1e-4,
        ),
    )
    for raw_name, loss_name, weight_name, default_weight in terms:
        raw_value = _get_aux_tensor(aux, raw_name)
        if raw_value is None:
            continue
        mean_value = _safe_mean(raw_value)
        losses[raw_name] = mean_value.detach()
        losses[loss_name] = mean_value * _get_float_arg(
            args,
            weight_name,
            default_weight,
        )
    temporal_rbf = _get_aux_tensor(aux, "auxiliary_temporal_rbf")
    if temporal_rbf is not None:
        losses["contact_bank_temporal_activity"] = _safe_mean(
            temporal_rbf
        ).detach()
    contact_target = _get_aux_tensor(aux, "auxiliary_contact_target")
    if contact_target is not None:
        losses["contact_bank_boundary_support"] = _safe_mean(
            contact_target
        ).detach()


def _apply_complete_expert_loss_contract(
    losses: Dict[str, torch.Tensor],
    expert_role,
) -> None:
    role = str(expert_role or "").strip().lower()
    if role not in {"global", "local", "contact"}:
        return
    allowed_prefixes = {
        "global": (),
        "local": ("L_scaffold_",),
        "contact": ("L_contact_bank_", "L_vis_sparse"),
    }[role]
    for name, value in tuple(losses.items()):
        if not name.startswith("L_"):
            continue
        if any(name.startswith(prefix) for prefix in allowed_prefixes):
            continue
        losses[name] = value * 0.0


def _resolve_expert_names(
    names: Sequence[str],
    count: int,
    defaults: Sequence[str],
    prefix: str,
) -> Tuple[str, ...]:
    if len(names) >= count:
        return tuple(names[:count])

    resolved = list(names)
    default_iter = iter(defaults)
    while len(resolved) < count:
        fallback = next(default_iter, f"{prefix}_{len(resolved)}")
        if fallback in resolved:
            fallback = f"{prefix}_{len(resolved)}"
        resolved.append(fallback)
    return tuple(resolved)


def _normalize_target(target: torch.Tensor) -> torch.Tensor:
    target = target.clamp_min(1e-6)
    return target / target.sum()


def _build_geo_target(
    args,
    active_geo_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    legacy_static = _get_float_arg(args, "target_geo_static", 0.30)
    legacy_smooth = _get_float_arg(args, "target_geo_smooth", 0.50)
    legacy_local = _get_float_arg(args, "target_geo_local", 0.20)

    target_map = {
        "static": legacy_static,
        "hexplane": _get_float_arg(args, "target_geo_hexplane", legacy_smooth * 0.7),
        "local": legacy_local,
        "smooth": _get_float_arg(args, "target_geo_residual_smooth", legacy_smooth * 0.3),
        "global": 1.0,
        "cut_graph": 1.0,
    }

    if tuple(active_geo_names) == ("static", "hexplane"):
        values = [
            _get_float_arg(args, "target_geo_static_stage2", legacy_static),
            _get_float_arg(
                args,
                "target_geo_hexplane_stage2",
                _get_float_arg(args, "target_geo_smooth_stage2", legacy_smooth),
            ),
        ]
    else:
        use_usage_targets = any(name in {"global", "cut_graph"} for name in active_geo_names)
        if use_usage_targets:
            values = [
                _get_float_arg(args, f"target_usage_geo_{name}", target_map.get(name, 1.0))
                for name in active_geo_names
            ]
        else:
            values = [target_map.get(name, 1.0) for name in active_geo_names]

    target = torch.tensor(values, device=device, dtype=dtype)
    return _normalize_target(target)


def _build_vis_target(
    args,
    active_vis_names: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tuple(active_vis_names[:2]) == ("stable", "transient"):
        values = [
            _get_float_arg(args, "target_vis_stable", 0.85),
            _get_float_arg(args, "target_vis_transient", 0.15),
        ][: len(active_vis_names)]
        target = torch.tensor(values, device=device, dtype=dtype)
        return _normalize_target(target)

    return torch.full((len(active_vis_names),), 1.0 / max(len(active_vis_names), 1), device=device, dtype=dtype)


def _accumulate_weighted_loss(
    losses: Dict[str, torch.Tensor],
    total_name: str,
    metric_name: str,
    metric_value: torch.Tensor,
    weight: float,
) -> None:
    losses[metric_name] = metric_value.detach()
    if weight == 0.0:
        return

    weighted = metric_value * weight
    losses[total_name] = weighted if total_name not in losses else losses[total_name] + weighted


def _add_geo_spatial_loss(losses: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor], args) -> None:
    """
    Compute spatial smoothness loss using kNN neighbors.

    Penalizes motion discrepancies between spatially nearby Gaussians to encourage
    coherent motion in local regions.
    """
    lambda_geo_spatial = _get_float_arg(args, "lambda_geo_spatial", 0.0)
    if lambda_geo_spatial <= 0:
        return

    d_mu = _get_aux_tensor(aux, "d_mu")
    means3d_canonical = _get_aux_tensor(aux, "means3d_canonical")

    if d_mu is None or means3d_canonical is None:
        return

    if d_mu.numel() == 0 or means3d_canonical.shape[0] < 2:
        return

    point_count = means3d_canonical.shape[0]
    sample_size = max(2, int(getattr(args, "geo_spatial_sample_size", 2048)))
    sample_count = min(point_count, sample_size)
    if sample_count < point_count:
        sample_indices = torch.randperm(
            point_count,
            device=means3d_canonical.device,
        )[:sample_count]
        sample_points = means3d_canonical[sample_indices]
        sample_motion = d_mu[sample_indices]
    else:
        sample_points = means3d_canonical
        sample_motion = d_mu

    scene_scale = _get_aux_tensor(aux, "scene_scale")
    if scene_scale is None:
        scale = sample_motion.new_tensor(1.0)
    else:
        scale = scene_scale.abs().reshape(()).clamp_min(1e-6)
    normalized_motion = sample_motion / scale

    neighbor_count = min(
        max(1, int(getattr(args, "geo_spatial_k", 8))),
        sample_count - 1,
    )
    chunk_size = max(1, int(getattr(args, "geo_spatial_chunk_size", 256)))
    weighted_roughness = normalized_motion.new_zeros(())
    total_weight = normalized_motion.new_zeros(())

    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        with torch.no_grad():
            squared_distances = torch.cdist(
                sample_points[start:end].detach().float(),
                sample_points.detach().float(),
            ).square()
            row_indices = torch.arange(end - start, device=squared_distances.device)
            squared_distances[row_indices, start + row_indices] = float("inf")
            neighbor_sq_distances, neighbor_indices = torch.topk(
                squared_distances,
                k=neighbor_count,
                dim=-1,
                largest=False,
                sorted=False,
            )
            finite_distances = neighbor_sq_distances[
                torch.isfinite(neighbor_sq_distances)
            ]
            bandwidth = finite_distances.median().clamp_min(1e-8)
            spatial_weights = torch.exp(
                -neighbor_sq_distances / bandwidth
            ).to(dtype=normalized_motion.dtype)

        neighbor_motion = normalized_motion[neighbor_indices]
        query_motion = normalized_motion[start:end].unsqueeze(1)
        robust_motion_difference = torch.sqrt(
            (neighbor_motion - query_motion).square().sum(dim=-1) + 1e-8
        )
        weighted_roughness = weighted_roughness + (
            spatial_weights * robust_motion_difference
        ).sum()
        total_weight = total_weight + spatial_weights.sum()

    spatial_roughness = weighted_roughness / total_weight.clamp_min(1e-8)
    losses["geo_spatial_roughness"] = spatial_roughness.detach()
    losses["geo_spatial_sample_count"] = spatial_roughness.new_tensor(
        float(sample_count)
    )
    losses["geo_spatial_neighbor_count"] = spatial_roughness.new_tensor(
        float(neighbor_count)
    )
    losses["L_geo_spatial"] = spatial_roughness * lambda_geo_spatial


def _add_temporal_regularization(losses: Dict[str, torch.Tensor], aux: Dict[str, torch.Tensor], args) -> None:
    d_mu_sequence = _get_aux_tensor(aux, "d_mu_sequence")
    time_sequence = _get_aux_tensor(aux, "time_sequence")
    if d_mu_sequence is None or time_sequence is None:
        return
    if d_mu_sequence.ndim != 3 or d_mu_sequence.shape[0] < 2:
        return

    time_sequence = time_sequence.reshape(-1)
    if time_sequence.numel() != d_mu_sequence.shape[0]:
        return

    order = torch.argsort(time_sequence)
    d_mu_sorted = d_mu_sequence[order]
    time_sorted = time_sequence[order]
    time_delta = time_sorted[1:] - time_sorted[:-1]
    valid = time_delta.abs() > 1e-6

    losses["temporal_pair_count"] = valid.to(dtype=d_mu_sequence.dtype).sum()
    if not bool(valid.any()):
        return

    delta_mu = d_mu_sorted[1:] - d_mu_sorted[:-1]
    velocity = delta_mu[valid] / time_delta[valid].abs().view(-1, 1, 1).clamp_min(1e-4)
    velocity_energy = velocity.square().mean()
    losses["geo_temp_velocity"] = velocity_energy.detach()
    losses["L_geo_temp"] = velocity_energy * _get_float_arg(args, "lambda_geo_temp", 0.0)


def _add_geo_expert_regularization(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
    geo_expert_names: Sequence[str],
) -> None:
    if "hexplane" not in geo_expert_names and "local" not in geo_expert_names and "smooth" not in geo_expert_names:
        return

    expert_specs = (
        (
            "hexplane",
            _get_float_arg(args, "lambda_mag_g1_mu", 1e-4),
            _get_float_arg(args, "lambda_sat_g1_disp", 5e-4),
            _get_float_arg(args, "lambda_raw_g1_disp", 1e-4),
        ),
        (
            "local",
            _get_float_arg(args, "lambda_mag_g2_mu", 2e-5),
            _get_float_arg(args, "lambda_sat_g2_disp", 1e-4),
            _get_float_arg(args, "lambda_raw_g2_disp", 1e-4),
        ),
        (
            "smooth",
            _get_float_arg(args, "lambda_mag_g3_mu", _get_float_arg(args, "lambda_mag_g2_mu", 2e-5)),
            _get_float_arg(args, "lambda_sat_g3_disp", _get_float_arg(args, "lambda_sat_g2_disp", 1e-4)),
            _get_float_arg(args, "lambda_raw_g3_disp", _get_float_arg(args, "lambda_raw_g2_disp", 1e-4)),
        ),
    )

    for expert_name, mag_weight, sat_weight, raw_weight in expert_specs:
        if expert_name not in geo_expert_names:
            continue

        disp_norm = _get_aux_tensor(aux, f"geo_weighted_disp_norm_{expert_name}")
        disp_ratio = _get_aux_tensor(aux, f"geo_weighted_disp_ratio_{expert_name}")
        saturation = _get_aux_tensor(aux, f"geo_saturation_{expert_name}")

        if disp_norm is not None:
            disp_norm_mean = _safe_mean(disp_norm)
            _accumulate_weighted_loss(
                losses,
                total_name="L_raw_geo",
                metric_name=f"geo_disp_norm_{expert_name}",
                metric_value=disp_norm_mean,
                weight=raw_weight,
            )

        if disp_ratio is not None:
            disp_ratio_mean = _safe_mean(disp_ratio)
            _accumulate_weighted_loss(
                losses,
                total_name="L_mag_geo",
                metric_name=f"geo_disp_ratio_{expert_name}",
                metric_value=disp_ratio_mean,
                weight=mag_weight,
            )

        if saturation is not None:
            saturation_mean = _safe_mean(saturation)
            _accumulate_weighted_loss(
                losses,
                total_name="L_sat_geo",
                metric_name=f"geo_saturation_{expert_name}",
                metric_value=saturation_mean,
                weight=sat_weight,
            )


def _add_cams_motion_magnitude_loss(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
) -> None:
    global_motion_norm = _get_aux_tensor(aux, "global_motion_norm")
    local_motion_norm = _get_aux_tensor(aux, "local_motion_norm")
    cut_graph_motion_norm = _get_aux_tensor(aux, "cut_graph_motion_norm")

    if global_motion_norm is not None:
        global_mag = _safe_mean(global_motion_norm)
        losses["global_motion_magnitude"] = global_mag.detach()
        _accumulate_weighted_loss(
            losses,
            total_name="L_motion_mag",
            metric_name="global_motion_mag_loss",
            metric_value=global_mag,
            weight=_get_float_arg(args, "lambda_motion_mag_global", 1e-4),
        )

    if local_motion_norm is not None:
        local_mag = _safe_mean(local_motion_norm)
        losses["local_motion_magnitude"] = local_mag.detach()
        _accumulate_weighted_loss(
            losses,
            total_name="L_motion_mag",
            metric_name="local_motion_mag_loss",
            metric_value=local_mag,
            weight=_get_float_arg(args, "lambda_motion_mag_local", 2e-5),
        )

    if cut_graph_motion_norm is not None:
        cut_graph_mag = _safe_mean(cut_graph_motion_norm)
        losses["cut_graph_motion_magnitude"] = cut_graph_mag.detach()
        _accumulate_weighted_loss(
            losses,
            total_name="L_motion_mag",
            metric_name="cut_graph_motion_mag_loss",
            metric_value=cut_graph_mag,
            weight=_get_float_arg(args, "lambda_motion_mag_cut_graph", 2e-5),
        )


def _add_cams_patch_c_losses(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
    phase: Optional[str],
) -> None:
    appearance_offsets = _get_aux_tensor(aux, "appearance_offsets")
    visibility_alpha = _get_aux_tensor(aux, "visibility_alpha")
    transient_probability = _get_aux_tensor(aux, "transient_probability")
    lifecycle_probs = _get_aux_tensor(aux, "lifecycle_probs")
    lifecycle_logits = _get_aux_tensor(aux, "lifecycle_logits")
    appearance_phases = {
        "visibility_refine",
        "joint_finetune",
        "moe_expert_full",
        "moe_joint_finetune",
    }
    lifecycle_phases = {
        "joint_finetune",
        "moe_expert_full",
        "moe_joint_finetune",
    }

    if appearance_offsets is not None and phase in appearance_phases:
        appearance_energy = appearance_offsets.square().mean()
        losses["appearance_offset_energy"] = appearance_energy.detach()
        losses["L_appearance_reg"] = appearance_energy * _get_float_arg(args, "lambda_appearance_reg", 1e-4)

    if visibility_alpha is not None and phase in lifecycle_phases:
        occlusion_probability = 1.0 - visibility_alpha.clamp(0.0, 1.0)
        mean_occlusion = occlusion_probability.mean()
        losses["mean_visibility_alpha"] = visibility_alpha.mean().detach()
        losses["mean_occlusion_probability"] = mean_occlusion.detach()
        losses["L_visibility_occlusion"] = (
            mean_occlusion
            * _get_float_arg(args, "lambda_visibility_occlusion", 1e-4)
        )

    if transient_probability is not None and phase in lifecycle_phases:
        mean_transient = transient_probability.clamp(0.0, 1.0).mean()
        losses["mean_transient_probability"] = mean_transient.detach()
        losses["L_transient_sparse"] = (
            mean_transient
            * _get_float_arg(args, "lambda_transient_sparse", 1e-4)
        )

    if lifecycle_probs is not None and phase in lifecycle_phases:
        persistent_prob = lifecycle_probs[:, :1]
        transient_prob = lifecycle_probs[:, 1:2] if lifecycle_probs.shape[-1] > 1 else 1.0 - persistent_prob
        lifecycle_balance = (persistent_prob.mean() - _get_float_arg(args, "target_lifecycle_persistent", 0.8)) ** 2
        losses["mean_lifecycle_persistent"] = persistent_prob.mean().detach()
        losses["mean_lifecycle_transient"] = transient_prob.mean().detach()
        losses["L_lifecycle_balance"] = lifecycle_balance * _get_float_arg(args, "lambda_lifecycle_balance", 1e-4)

    if lifecycle_logits is not None and lifecycle_probs is not None and phase in lifecycle_phases:
        transient_prob = lifecycle_probs[:, 1:2] if lifecycle_probs.shape[-1] > 1 else 1.0 - lifecycle_probs[:, :1]
        lifecycle_transient = transient_prob.mean()
        losses["lifecycle_transient_probability"] = lifecycle_transient.detach()
        losses["L_lifecycle_reg"] = (
            lifecycle_transient
            * _get_float_arg(args, "lambda_lifecycle_reg", 1e-4)
        )


def compute_tracking_losses(
    aux: Dict[str, torch.Tensor],
    iteration: int,
    args,
    prev_d_mu: Optional[torch.Tensor],
    active_geo: int,
    active_vis: int,
    enable_visibility: bool,
    geo_expert_names: Sequence[str],
    vis_expert_names: Sequence[str],
    force_geo_expert: Optional[str] = None,
    force_vis_expert: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    del prev_d_mu

    pixel_routing_weights = aux.get("pixel_routing_weights")
    pixel_expert_coverage = aux.get("pixel_expert_coverage")
    pixel_router_residual_logits = aux.get("pixel_router_residual_logits")
    pi_geo = aux.get("pi_geo")
    pi_vis = aux.get("pi_vis")

    d_mu = aux.get("d_mu")
    d_rot = aux.get("d_rot")
    d_scale = aux.get("d_scale")
    d_opacity_logit = aux.get("d_opacity_logit")
    entropy_geo = aux.get("entropy_geo")
    entropy_vis = aux.get("entropy_vis")

    losses: Dict[str, torch.Tensor] = {}
    if isinstance(pixel_router_residual_logits, torch.Tensor):
        losses["pixel_router_residual_abs_mean"] = (
            pixel_router_residual_logits.abs().mean().detach()
        )
        losses["pixel_router_residual_abs_max"] = (
            pixel_router_residual_logits.abs().amax().detach()
        )

    pixel_usage_geo = None
    if pixel_routing_weights is not None:
        if (
            isinstance(pixel_expert_coverage, torch.Tensor)
            and pixel_expert_coverage.shape == pixel_routing_weights.shape
        ):
            covered_pixels = pixel_expert_coverage.sum(dim=0) > 1e-8
        else:
            covered_pixels = pixel_routing_weights.sum(dim=0) > 0
        if covered_pixels.any():
            covered_weights = pixel_routing_weights[:, covered_pixels]
            pixel_usage_geo = covered_weights.mean(dim=1)
            losses["pixel_route_entropy_geo"] = -(
                covered_weights.clamp_min(1e-8)
                * covered_weights.clamp_min(1e-8).log()
            ).sum(dim=0).mean().detach()
            losses["pixel_route_max_prob_geo"] = (
                covered_weights.max(dim=0).values.mean().detach()
            )
            losses["pixel_route_covered_fraction"] = covered_pixels.to(
                dtype=pixel_routing_weights.dtype
            ).mean().detach()
        else:
            pixel_usage_geo = torch.zeros(
                (pixel_routing_weights.shape[0],),
                device=pixel_routing_weights.device,
                dtype=pixel_routing_weights.dtype,
            )
            losses["pixel_route_entropy_geo"] = pixel_routing_weights.new_zeros(())
            losses["pixel_route_max_prob_geo"] = pixel_routing_weights.new_zeros(())
            losses["pixel_route_covered_fraction"] = pixel_routing_weights.new_zeros(())

    if pi_geo is not None:
        usage_geo = pi_geo.mean(dim=0)
    else:
        usage_geo = pixel_usage_geo

    if usage_geo is not None:
        resolved_geo_names = _resolve_expert_names(
            geo_expert_names,
            usage_geo.numel(),
            DEFAULT_GEO_EXPERT_NAMES,
            prefix="geo",
        )
        for index, expert_name in enumerate(resolved_geo_names):
            losses[f"usage_geo_{expert_name}"] = usage_geo[index]

        dense_pi_geo = aux.get("gaussian_pi_geo_dense")
        if (
            isinstance(pi_geo, torch.Tensor)
            and pi_geo.ndim == 2
            and isinstance(dense_pi_geo, torch.Tensor)
            and dense_pi_geo.shape == pi_geo.shape
        ):
            dense_usage_geo = dense_pi_geo.mean(dim=0)
            for index, expert_name in enumerate(resolved_geo_names):
                losses[f"dense_usage_geo_{expert_name}"] = dense_usage_geo[index].detach()
                losses[f"route_coverage_geo_{expert_name}"] = (
                    pi_geo[:, index] > 1e-8
                ).to(dtype=pi_geo.dtype).mean().detach()

            sparse_probability = pi_geo.clamp_min(0.0)
            dense_probability = dense_pi_geo.clamp_min(0.0)
            losses["route_zero_fraction_geo"] = (pi_geo <= 1e-8).to(dtype=pi_geo.dtype).mean().detach()
            losses["route_effective_experts_geo"] = (
                1.0 / sparse_probability.square().sum(dim=-1).clamp_min(1e-8)
            ).mean().detach()
            losses["route_dense_effective_experts_geo"] = (
                1.0 / dense_probability.square().sum(dim=-1).clamp_min(1e-8)
            ).mean().detach()
            losses["route_sparse_dense_l1_geo"] = (
                sparse_probability - dense_probability
            ).abs().mean().detach()

        if pixel_usage_geo is not None:
            pixel_geo_names = _resolve_expert_names(
                geo_expert_names,
                pixel_usage_geo.numel(),
                DEFAULT_GEO_EXPERT_NAMES,
                prefix="geo",
            )
            for index, expert_name in enumerate(pixel_geo_names):
                losses[f"pixel_usage_geo_{expert_name}"] = pixel_usage_geo[index].detach()
                if (
                    isinstance(pixel_expert_coverage, torch.Tensor)
                    and pixel_expert_coverage.shape[0] > index
                ):
                    losses[f"pixel_coverage_geo_{expert_name}"] = (
                        pixel_expert_coverage[index] > 1e-8
                    ).to(dtype=pixel_expert_coverage.dtype).mean().detach()

        if force_geo_expert is None:
            active_geo = max(1, min(int(active_geo), usage_geo.numel()))
            active_geo_names = resolved_geo_names[:active_geo]
            active_usage_geo = usage_geo[:active_geo]
            target_geo = _build_geo_target(args, active_geo_names, usage_geo.device, usage_geo.dtype)

            for expert_name, target_value in zip(active_geo_names, target_geo):
                losses[f"target_usage_geo_{expert_name}"] = target_value.detach()

            route_balance_scale = _get_aux_scale(aux, "route_balance_scale", usage_geo)
            route_confidence_scale = _get_aux_scale(aux, "route_confidence_scale", usage_geo)
            losses["route_balance_scale"] = route_balance_scale.detach()
            losses["route_confidence_scale"] = route_confidence_scale.detach()
            losses["L_balance_geo"] = (
                ((active_usage_geo - target_geo) ** 2).sum()
                * _get_float_arg(args, "lambda_balance_geo", 0.0)
                * route_balance_scale
            )

            route_max_prob_geo = aux.get("route_max_prob_geo")
            route_margin_geo = aux.get("route_margin_geo")
            if route_max_prob_geo is not None:
                route_max_prob_geo = _safe_mean(route_max_prob_geo)
                losses["route_max_prob_geo"] = route_max_prob_geo.detach()
                if active_geo > 1:
                    losses["L_route_conf_geo"] = (
                        (1.0 - route_max_prob_geo)
                        * _get_float_arg(args, "lambda_route_conf_geo", 0.0)
                        * route_confidence_scale
                    )
            if route_margin_geo is not None:
                losses["route_margin_geo"] = _safe_mean(route_margin_geo).detach()
        else:
            losses["L_balance_geo"] = torch.zeros((), device=usage_geo.device, dtype=usage_geo.dtype)

        _add_geo_expert_regularization(losses, aux, args, resolved_geo_names)
        _add_cams_motion_magnitude_loss(losses, aux, args)
        _add_cams_patch_c_losses(losses, aux, args, aux.get("tracking_phase_name"))

    if pi_vis is not None:
        usage_vis = pi_vis.mean(dim=0)
        resolved_vis_names = _resolve_expert_names(
            vis_expert_names,
            usage_vis.numel(),
            DEFAULT_VIS_EXPERT_NAMES,
            prefix="vis",
        )
        for index, expert_name in enumerate(resolved_vis_names):
            losses[f"usage_vis_{expert_name}"] = usage_vis[index]

        active_vis = max(1, min(int(active_vis), usage_vis.numel()))
        active_vis_names = resolved_vis_names[:active_vis]
        if force_vis_expert is None and active_vis > 1 and enable_visibility:
            target_vis = _build_vis_target(args, active_vis_names, usage_vis.device, usage_vis.dtype)
            for expert_name, target_value in zip(active_vis_names, target_vis):
                losses[f"target_usage_vis_{expert_name}"] = target_value.detach()
            losses["L_balance_vis"] = ((usage_vis[:active_vis] - target_vis) ** 2).sum() * _get_float_arg(args, "lambda_balance_vis", 0.0)
        else:
            losses["L_balance_vis"] = torch.zeros((), device=usage_vis.device, dtype=usage_vis.dtype)

        route_max_prob_vis = aux.get("route_max_prob_vis")
        route_margin_vis = aux.get("route_margin_vis")
        if route_max_prob_vis is not None:
            route_max_prob_vis = _safe_mean(route_max_prob_vis)
            losses["route_max_prob_vis"] = route_max_prob_vis.detach()
            if force_vis_expert is None and active_vis > 1 and enable_visibility:
                losses["L_route_conf_vis"] = (1.0 - route_max_prob_vis) * _get_float_arg(args, "lambda_route_conf_vis", 0.0)
        if route_margin_vis is not None:
            losses["route_margin_vis"] = _safe_mean(route_margin_vis).detach()

    if entropy_geo is not None and entropy_vis is not None and iteration <= getattr(args, "entropy_end_iter", 0):
        losses["L_entropy"] = -args.lambda_entropy_geo * entropy_geo - args.lambda_entropy_vis * entropy_vis

    if d_mu is not None:
        losses["mean_norm_d_mu"] = _safe_mean(torch.norm(d_mu, dim=-1)).detach()
        _add_temporal_regularization(losses, aux, args)
        _add_geo_spatial_loss(losses, aux, args)
        _add_scaffold_regularization(losses, aux, args)
        _add_contact_bank_regularization(losses, aux, args)

    if d_rot is not None:
        losses["mean_norm_d_rot"] = _safe_mean(torch.norm(d_rot, dim=-1)).detach()
    if d_scale is not None:
        losses["mean_norm_d_scale"] = _safe_mean(torch.norm(d_scale, dim=-1)).detach()

    if d_opacity_logit is not None:
        mean_abs_opacity = _safe_mean(torch.abs(d_opacity_logit))
        losses["mean_abs_d_opacity"] = mean_abs_opacity.detach()
        losses["L_vis_sparse"] = mean_abs_opacity * _get_float_arg(args, "lambda_vis_sparse", 0.0)

    if (
        pi_vis is not None
        and d_mu is not None
        and pi_vis.shape[-1] > 1
        and active_vis > 1
        and enable_visibility
        and iteration >= getattr(args, "enable_decouple_iter", 0)
    ):
        resolved_vis_names = _resolve_expert_names(
            vis_expert_names,
            pi_vis.shape[-1],
            DEFAULT_VIS_EXPERT_NAMES,
            prefix="vis",
        )
        if "transient" in resolved_vis_names:
            transient_index = resolved_vis_names.index("transient")
            pi_transient = pi_vis[:, transient_index]
            d_mu_sq = (d_mu ** 2).sum(dim=-1)
            losses["mean_pi_vis_transient"] = _safe_mean(pi_transient).detach()
            losses["L_decouple"] = (pi_transient * d_mu_sq).mean() * _get_float_arg(args, "lambda_decouple", 0.0)
    elif pi_vis is not None and pi_vis.shape[-1] > 1:
        resolved_vis_names = _resolve_expert_names(
            vis_expert_names,
            pi_vis.shape[-1],
            DEFAULT_VIS_EXPERT_NAMES,
            prefix="vis",
        )
        if "transient" in resolved_vis_names:
            transient_index = resolved_vis_names.index("transient")
            losses["mean_pi_vis_transient"] = _safe_mean(pi_vis[:, transient_index]).detach()

    expert_diversity_geo = aux.get("expert_diversity_geo")
    if expert_diversity_geo is not None:
        expert_diversity_geo = _safe_mean(expert_diversity_geo)
        losses["expert_diversity_geo"] = expert_diversity_geo.detach()
        losses["L_expert_diversity_geo"] = expert_diversity_geo * _get_float_arg(args, "lambda_expert_diversity_geo", 0.0)

    if entropy_geo is not None:
        losses["entropy_geo"] = entropy_geo.detach()
    if entropy_vis is not None:
        losses["entropy_vis"] = entropy_vis.detach()

    _apply_complete_expert_loss_contract(
        losses,
        aux.get("expert_role"),
    )
    return losses
