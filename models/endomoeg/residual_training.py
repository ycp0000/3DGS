from typing import Dict, Optional

import torch


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
    return (value * effective_weight).sum() / denominator


def compute_residual_boosting_losses(
    candidate: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    hard_quantile: float = 0.7,
    preserve_weight: float = 1.0,
    no_regret_weight: float = 1.0,
    no_regret_margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if candidate.shape != teacher.shape or candidate.shape != target.shape:
        raise ValueError(
            "candidate, teacher, and target must share the same shape"
        )
    if candidate.ndim != 4:
        raise ValueError("residual boosting expects [B, C, H, W] tensors")
    if not 0.0 <= float(hard_quantile) <= 1.0:
        raise ValueError("hard_quantile must be in [0, 1]")

    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    teacher_error = (teacher - target).abs().mean(dim=1, keepdim=True)
    valid_mask = _canonical_mask(mask, candidate_error)

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

    preserve_mask = valid_mask & ~hard_mask
    hard_weight = hard_mask.to(dtype=candidate.dtype)
    preserve_region_weight = preserve_mask.to(dtype=candidate.dtype)
    valid_weight = valid_mask.to(dtype=candidate.dtype)

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
    no_regret = torch.relu(
        candidate_error
        - teacher_error.detach()
        - float(no_regret_margin)
    )
    no_regret_loss = _masked_weighted_mean(
        no_regret,
        valid_weight,
        valid_mask,
    )
    total = (
        boost_loss
        + float(preserve_weight) * preserve_loss
        + float(no_regret_weight) * no_regret_loss
    )
    valid_count = valid_weight.sum().clamp_min(1.0)
    return {
        "L_residual_boost": boost_loss,
        "L_residual_preserve": preserve_loss * float(preserve_weight),
        "L_residual_no_regret": no_regret_loss * float(no_regret_weight),
        "L_residual_total": total,
        "residual_hard_fraction": hard_weight.sum().detach() / valid_count,
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
    }
