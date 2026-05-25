from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch


def _safe_mean(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return torch.zeros((), device=value.device, dtype=value.dtype)
    return value.mean()


def _get_float_arg(args, name: str, default: float) -> float:
    value = getattr(args, name, default)
    if value is None:
        return float(default)
    return float(value)


def _get_aux_tensor(
    aux: Dict[str, torch.Tensor],
    names: Sequence[str],
) -> Optional[torch.Tensor]:
    for name in names:
        value = aux.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _add_aux_regularization(
    losses: Dict[str, torch.Tensor],
    aux: Dict[str, torch.Tensor],
    args,
    loss_name: str,
    specs: Iterable[tuple[Sequence[str], str, float, str]],
) -> None:
    total: Optional[torch.Tensor] = None

    for key_aliases, lambda_name, default_lambda, metric_name in specs:
        value = _get_aux_tensor(aux, key_aliases)
        if value is None:
            continue

        mean_value = _safe_mean(value)
        weight = _get_float_arg(args, lambda_name, default_lambda)
        losses[metric_name] = mean_value.detach()

        if weight == 0.0:
            weighted_term = mean_value * 0.0
            losses[f"{metric_name}_weighted"] = weighted_term.detach()
            continue

        term = mean_value * weight
        losses[f"{metric_name}_weighted"] = term.detach()
        total = term if total is None else total + term

    if total is not None:
        losses[loss_name] = total


def compute_tracking_losses(
    aux: Dict[str, torch.Tensor],
    iteration: int,
    args,
    prev_d_mu: torch.Tensor | None,
    active_geo: int,
    active_vis: int,
    enable_visibility: bool,
) -> Dict[str, torch.Tensor]:
    pi_geo = aux.get("pi_geo")
    pi_vis = aux.get("pi_vis")

    d_mu = aux.get("d_mu")
    d_rot = aux.get("d_rot")
    d_scale = aux.get("d_scale")
    d_opacity_logit = aux.get("d_opacity_logit")
    entropy_geo = aux.get("entropy_geo")
    entropy_vis = aux.get("entropy_vis")

    losses: Dict[str, torch.Tensor] = {}

    if pi_geo is not None:
        usage_geo = pi_geo.mean(dim=0)
        losses["usage_geo_static"] = usage_geo[0]
        if usage_geo.numel() > 1:
            losses["usage_geo_smooth"] = usage_geo[1]
        if usage_geo.numel() > 2:
            losses["usage_geo_local"] = usage_geo[2]

        active_geo = max(1, min(int(active_geo), usage_geo.numel()))
        active_usage_geo = usage_geo[:active_geo]

        if active_geo == 3:
            target_geo = torch.tensor(
                [
                    _get_float_arg(args, "target_geo_static", 0.30),
                    _get_float_arg(args, "target_geo_smooth", 0.50),
                    _get_float_arg(args, "target_geo_local", 0.20),
                ],
                device=usage_geo.device,
                dtype=usage_geo.dtype,
            )
            target_geo = target_geo / target_geo.sum()
        elif active_geo == 2:
            target_geo = torch.tensor(
                [
                    _get_float_arg(args, "target_geo_static_stage2", 0.40),
                    _get_float_arg(args, "target_geo_smooth_stage2", 0.60),
                ],
                device=usage_geo.device,
                dtype=usage_geo.dtype,
            )
            target_geo = target_geo / target_geo.sum()
        else:
            target_geo = torch.ones_like(active_usage_geo) / max(active_geo, 1)

        losses["target_usage_geo_static"] = target_geo[0].detach()
        if target_geo.numel() > 1:
            losses["target_usage_geo_smooth"] = target_geo[1].detach()
        if target_geo.numel() > 2:
            losses["target_usage_geo_local"] = target_geo[2].detach()

        losses["L_balance_geo"] = ((active_usage_geo - target_geo) ** 2).sum() * _get_float_arg(args, "lambda_balance_geo", 0.0)

        route_max_prob_geo = aux.get("route_max_prob_geo")
        route_margin_geo = aux.get("route_margin_geo")
        if route_max_prob_geo is not None:
            losses["route_max_prob_geo"] = _safe_mean(route_max_prob_geo)
            if active_geo > 1:
                losses["L_route_conf_geo"] = (1.0 - _safe_mean(route_max_prob_geo)) * _get_float_arg(args, "lambda_route_conf_geo", 0.0)
        if route_margin_geo is not None:
            losses["route_margin_geo"] = _safe_mean(route_margin_geo)

    if pi_vis is not None:
        usage_vis = pi_vis.mean(dim=0)
        losses["usage_vis_stable"] = usage_vis[0]
        if usage_vis.numel() > 1:
            losses["usage_vis_transient"] = usage_vis[1]

        active_vis = max(1, min(int(active_vis), usage_vis.numel()))
        if active_vis > 1 and enable_visibility:
            target_vis = torch.tensor([0.85, 0.15], device=usage_vis.device, dtype=usage_vis.dtype)
            target_vis = target_vis[:active_vis]
            target_vis = target_vis / target_vis.sum()
            losses["L_balance_vis"] = ((usage_vis[:active_vis] - target_vis) ** 2).sum() * _get_float_arg(args, "lambda_balance_vis", 0.0)
        else:
            losses["L_balance_vis"] = torch.zeros((), device=usage_vis.device, dtype=usage_vis.dtype)

        route_max_prob_vis = aux.get("route_max_prob_vis")
        route_margin_vis = aux.get("route_margin_vis")
        if route_max_prob_vis is not None:
            losses["route_max_prob_vis"] = _safe_mean(route_max_prob_vis)
            if active_vis > 1 and enable_visibility:
                losses["L_route_conf_vis"] = (1.0 - _safe_mean(route_max_prob_vis)) * _get_float_arg(args, "lambda_route_conf_vis", 0.0)
        if route_margin_vis is not None:
            losses["route_margin_vis"] = _safe_mean(route_margin_vis)

    if entropy_geo is not None and entropy_vis is not None and iteration <= args.entropy_end_iter:
        losses["L_entropy"] = -args.lambda_entropy_geo * entropy_geo - args.lambda_entropy_vis * entropy_vis

    if d_mu is not None:
        losses["mean_norm_d_mu"] = _safe_mean(torch.norm(d_mu, dim=-1))
        if prev_d_mu is not None and prev_d_mu.shape == d_mu.shape:
            losses["L_geo_temp"] = ((d_mu - prev_d_mu) ** 2).mean() * _get_float_arg(args, "lambda_geo_temp", 0.0)
        centered = d_mu - d_mu.mean(dim=0, keepdim=True)
        losses["L_geo_spatial"] = (centered ** 2).mean() * _get_float_arg(args, "lambda_geo_spatial", 0.0)

    if d_rot is not None:
        losses["mean_norm_d_rot"] = _safe_mean(torch.norm(d_rot, dim=-1))
    if d_scale is not None:
        losses["mean_norm_d_scale"] = _safe_mean(torch.norm(d_scale, dim=-1))

    if d_opacity_logit is not None:
        losses["mean_abs_d_opacity"] = _safe_mean(torch.abs(d_opacity_logit))
        losses["L_vis_sparse"] = _safe_mean(torch.abs(d_opacity_logit)) * _get_float_arg(args, "lambda_vis_sparse", 0.0)

    if (
        pi_vis is not None
        and d_mu is not None
        and pi_vis.shape[-1] > 1
        and active_vis > 1
        and enable_visibility
        and iteration >= args.enable_decouple_iter
    ):
        pi_transient = pi_vis[:, 1]
        d_mu_sq = (d_mu ** 2).sum(dim=-1)
        losses["mean_pi_vis_transient"] = _safe_mean(pi_transient)
        losses["L_decouple"] = (pi_transient * d_mu_sq).mean() * _get_float_arg(args, "lambda_decouple", 0.0)
    elif pi_vis is not None and pi_vis.shape[-1] > 1:
        losses["mean_pi_vis_transient"] = _safe_mean(pi_vis[:, 1])

    expert_diversity_geo = aux.get("expert_diversity_geo")
    if expert_diversity_geo is not None:
        losses["expert_diversity_geo"] = _safe_mean(expert_diversity_geo)
        losses["L_expert_diversity_geo"] = _safe_mean(expert_diversity_geo) * _get_float_arg(args, "lambda_expert_diversity_geo", 0.0)

    _add_aux_regularization(
        losses=losses,
        aux=aux,
        args=args,
        loss_name="L_sat_geo",
        specs=[
            (("loss_geo_e1_sat_disp", "loss_geo_e1_loss_sat_disp"), "lambda_sat_g1_disp", 5e-4, "sat_geo_e1_disp"),
            (("loss_geo_e1_sat_rot", "loss_geo_e1_loss_sat_rot"), "lambda_sat_g1_rot", 5e-4, "sat_geo_e1_rot"),
            (("loss_geo_e1_sat_scl", "loss_geo_e1_loss_sat_scl"), "lambda_sat_g1_scl", 0.0, "sat_geo_e1_scl"),
            (("loss_geo_e2_sat_disp", "loss_geo_e2_loss_sat_disp"), "lambda_sat_g2_disp", 1e-4, "sat_geo_e2_disp"),
            (("loss_geo_e2_sat_rot", "loss_geo_e2_loss_sat_rot"), "lambda_sat_g2_rot", 1e-4, "sat_geo_e2_rot"),
            (("loss_geo_e2_sat_scl", "loss_geo_e2_loss_sat_scl"), "lambda_sat_g2_scl", 0.0, "sat_geo_e2_scl"),
        ],
    )

    _add_aux_regularization(
        losses=losses,
        aux=aux,
        args=args,
        loss_name="L_mag_geo",
        specs=[
            (("loss_geo_e1_mag_disp", "loss_geo_e1_mag_mu"), "lambda_mag_g1_mu", 1e-4, "mag_geo_e1_mu"),
            (("loss_geo_e1_mag_rot",), "lambda_mag_g1_rot", 1e-4, "mag_geo_e1_rot"),
            (("loss_geo_e1_mag_scl",), "lambda_mag_g1_scl", 0.0, "mag_geo_e1_scl"),
            (("loss_geo_e2_mag_disp", "loss_geo_e2_mag_mu"), "lambda_mag_g2_mu", 2e-5, "mag_geo_e2_mu"),
            (("loss_geo_e2_mag_rot",), "lambda_mag_g2_rot", 2e-5, "mag_geo_e2_rot"),
            (("loss_geo_e2_mag_scl",), "lambda_mag_g2_scl", 0.0, "mag_geo_e2_scl"),
        ],
    )

    _add_aux_regularization(
        losses=losses,
        aux=aux,
        args=args,
        loss_name="L_raw_geo",
        specs=[
            (("loss_geo_e1_raw_disp",), "lambda_raw_g1_disp", 1e-4, "raw_geo_e1_disp"),
            (("loss_geo_e1_raw_rot",), "lambda_raw_g1_rot", 1e-4, "raw_geo_e1_rot"),
            (("loss_geo_e1_raw_scl",), "lambda_raw_g1_scl", 0.0, "raw_geo_e1_scl"),
            (("loss_geo_e2_raw_disp",), "lambda_raw_g2_disp", 1e-4, "raw_geo_e2_disp"),
            (("loss_geo_e2_raw_rot",), "lambda_raw_g2_rot", 5e-5, "raw_geo_e2_rot"),
            (("loss_geo_e2_raw_scl",), "lambda_raw_g2_scl", 0.0, "raw_geo_e2_scl"),
        ],
    )

    if entropy_geo is not None:
        losses["entropy_geo"] = entropy_geo
    if entropy_vis is not None:
        losses["entropy_vis"] = entropy_vis

    return losses
