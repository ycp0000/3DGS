from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _canonical_mask(
    mask: Optional[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    value = mask.to(device=reference.device)
    if value.ndim == reference.ndim - 1:
        value = value.unsqueeze(1)
    if value.shape != reference.shape:
        value = value.expand_as(reference)
    return value.bool()


def _masked_weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    effective_weight = weight * mask.to(dtype=value.dtype)
    denominator = effective_weight.sum().clamp_min(1.0)
    masked_value = torch.where(
        mask,
        value,
        torch.zeros_like(value),
    )
    return (masked_value * effective_weight).sum() / denominator


def _assert_finite_on_mask(
    name: str,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    expanded_mask = mask.expand_as(value)
    if not torch.isfinite(value[expanded_mask]).all():
        raise FloatingPointError(
            "{} contains NaN or Inf in the valid residual region".format(name)
        )


def compute_residual_boosting_losses(
    candidate: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    support: Optional[torch.Tensor] = None,
    hard_quantile: float = 0.7,
    reconstruction_weight: float = 1.0,
    boost_weight: float = 0.25,
    preserve_weight: float = 1.0,
    no_regret_weight: float = 1.0,
    no_regret_margin: float = 0.0,
    no_regret_temperature: float = 0.01,
) -> Dict[str, torch.Tensor]:
    if candidate.shape != teacher.shape or candidate.shape != target.shape:
        raise ValueError(
            "candidate, teacher, and target must share the same shape"
        )
    if candidate.ndim != 4:
        raise ValueError("residual boosting expects [B, C, H, W] tensors")
    if not 0.0 <= float(hard_quantile) <= 1.0:
        raise ValueError("hard_quantile must be in [0, 1]")
    if float(no_regret_temperature) <= 0.0:
        raise ValueError("no_regret_temperature must be positive")

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    teacher_error = (teacher - target).abs().mean(dim=1, keepdim=True)
    valid_mask = _canonical_mask(mask, candidate_error)
    if support is None:
        support_weight = valid_mask.to(dtype=candidate.dtype)
        preserve_region_weight = None
        no_regret_weight_map = valid_mask.to(dtype=candidate.dtype)
    else:
        support_value = support.to(device=candidate.device, dtype=candidate.dtype)
        if support_value.ndim == candidate_error.ndim - 1:
            support_value = support_value.unsqueeze(1)
        if support_value.shape != candidate_error.shape:
            support_value = support_value.expand_as(candidate_error)
        support_weight = support_value.detach().clamp(0.0, 1.0)
        support_weight = support_weight * valid_mask.to(dtype=candidate.dtype)
        preserve_region_weight = (
            (1.0 - support_weight) * valid_mask.to(dtype=candidate.dtype)
        )
        no_regret_weight_map = preserve_region_weight
    _assert_finite_on_mask("candidate", candidate, valid_mask)
    _assert_finite_on_mask("teacher", teacher, valid_mask)
    _assert_finite_on_mask("target", target, valid_mask)

    hard_mask = torch.zeros_like(valid_mask)
    detached_teacher_error = teacher_error.detach()
    for batch_index in range(candidate.shape[0]):
        valid_values = detached_teacher_error[batch_index][
            valid_mask[batch_index]
        ]
        if valid_values.numel() == 0:
            continue
        threshold = torch.quantile(valid_values, float(hard_quantile))
        hard_mask[batch_index] = (
            detached_teacher_error[batch_index] >= threshold
        ) & valid_mask[batch_index]

    if support is None:
        preserve_mask = valid_mask & ~hard_mask
        preserve_region_weight = preserve_mask.to(dtype=candidate.dtype)
        reconstruction_weight_map = valid_mask.to(dtype=candidate.dtype)
        hard_weight = hard_mask.to(dtype=candidate.dtype)
    else:
        reconstruction_weight_map = support_weight
        hard_weight = hard_mask.to(dtype=candidate.dtype) * support_weight
    valid_weight = valid_mask.to(dtype=candidate.dtype)

    reconstruction_loss = _masked_weighted_mean(
        candidate_error,
        reconstruction_weight_map,
        valid_mask,
    )
    boost_loss = _masked_weighted_mean(
        candidate_error,
        hard_weight,
        valid_mask,
    )
    teacher_distance = (candidate - teacher).abs().mean(
        dim=1,
        keepdim=True,
    )
    preserve_loss = _masked_weighted_mean(
        teacher_distance,
        preserve_region_weight,
        valid_mask,
    )
    no_regret_delta = (
        candidate_error
        - teacher_error.detach()
        - float(no_regret_margin)
    )
    temperature = float(no_regret_temperature)
    no_regret_excess = torch.relu(
        no_regret_delta
    )
    if support is None:
        no_regret = F.softplus(
            no_regret_delta / temperature
        ) * temperature
    else:
        no_regret = no_regret_excess
    no_regret_loss = _masked_weighted_mean(
        no_regret,
        no_regret_weight_map,
        valid_mask,
    )
    total = (
        float(reconstruction_weight) * reconstruction_loss
        + float(boost_weight) * boost_loss
        + float(preserve_weight) * preserve_loss
        + float(no_regret_weight) * no_regret_loss
    )
    valid_count = valid_weight.sum().clamp_min(1.0)
    return {
        "L_residual_reconstruction": (
            reconstruction_loss * float(reconstruction_weight)
        ),
        "L_residual_boost": boost_loss * float(boost_weight),
        "L_residual_preserve": preserve_loss * float(preserve_weight),
        "L_residual_no_regret": no_regret_loss * float(no_regret_weight),
        "L_residual_total": total,
        "residual_hard_fraction": hard_weight.sum().detach() / valid_count,
        "residual_support_fraction": (
            support_weight.sum().detach() / valid_count
        ),
        "residual_teacher_error": _masked_weighted_mean(
            teacher_error.detach(),
            valid_weight,
            valid_mask,
        ).detach(),
        "residual_candidate_error": _masked_weighted_mean(
            candidate_error.detach(),
            valid_weight,
            valid_mask,
        ).detach(),
        "residual_regressed_fraction": _masked_weighted_mean(
            (no_regret_excess.detach() > 0.0).to(candidate.dtype),
            valid_weight,
            valid_mask,
        ).detach(),
        "residual_mean_regret": _masked_weighted_mean(
            no_regret_excess.detach(),
            valid_weight,
            valid_mask,
        ).detach(),
    }
