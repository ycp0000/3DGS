from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tracking.cams_gs_moe_tracking import PixelSpaceRouter

from .expert_bundle import EXPERT_ROLES


DEFAULT_MINIMUM_USAGE = {
    "global": 0.10,
    "local": 0.10,
    "contact": 0.02,
}


def _masked_mean(value, mask):
    if mask is None:
        return value.mean()
    mask_value = mask.to(device=value.device, dtype=value.dtype)
    while mask_value.ndim < value.ndim:
        mask_value = mask_value.unsqueeze(0)
    expanded = mask_value.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def oracle_routing_targets(
    expert_rgb,
    ground_truth,
    mask=None,
    temperature=0.05,
):
    if expert_rgb.ndim != 4 or expert_rgb.shape[1] != 3:
        raise ValueError("expert_rgb must have shape [experts, 3, H, W]")
    if ground_truth.ndim != 3 or ground_truth.shape[0] != 3:
        raise ValueError("ground_truth must have shape [3, H, W]")
    error = (expert_rgb.detach() - ground_truth.detach().unsqueeze(0)).abs().mean(
        dim=1
    )
    targets = torch.softmax(
        -error / max(float(temperature), 1e-6),
        dim=0,
    )
    if mask is not None:
        valid = mask.to(device=targets.device, dtype=torch.bool)
        if valid.ndim == 3:
            valid = valid.squeeze(0)
        if valid.shape != targets.shape[1:]:
            raise ValueError("mask shape does not match expert images")
        uniform = torch.full_like(targets, 1.0 / targets.shape[0])
        targets = torch.where(valid.unsqueeze(0), targets, uniform)
    return targets


def sparsify_router_weights(weights, top_k=None):
    if top_k is None or int(top_k) >= weights.shape[0]:
        return weights
    top_k = max(int(top_k), 1)
    values, indices = torch.topk(weights, k=top_k, dim=0)
    sparse = torch.zeros_like(weights).scatter(0, indices, values)
    sparse = sparse / sparse.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return sparse.detach() - weights.detach() + weights


def compute_router_losses(
    weights,
    expert_rgb,
    ground_truth,
    mask=None,
    oracle_temperature=0.05,
    lambda_oracle=1.0,
    lambda_starvation=0.01,
    minimum_usage=None,
):
    if weights.shape != expert_rgb.shape[:1] + expert_rgb.shape[2:]:
        raise ValueError("weights must have shape [experts, H, W]")
    targets = oracle_routing_targets(
        expert_rgb,
        ground_truth,
        mask=mask,
        temperature=oracle_temperature,
    )
    blended = (expert_rgb * weights.unsqueeze(1)).sum(dim=0)
    reconstruction = _masked_mean(
        (blended - ground_truth).abs(),
        mask,
    )
    oracle_ce = _masked_mean(
        -(targets * weights.clamp_min(1e-8).log()).sum(dim=0),
        mask,
    )

    usage_mask = None
    if mask is not None:
        usage_mask = mask.squeeze(0) if mask.ndim == 3 else mask
    usage = torch.stack(
        [_masked_mean(weights[index], usage_mask) for index in range(weights.shape[0])]
    )
    floors = minimum_usage or DEFAULT_MINIMUM_USAGE
    floor_tensor = weights.new_tensor(
        [float(floors[role]) for role in EXPERT_ROLES]
    )
    starvation = F.relu(floor_tensor - usage).square().sum()
    entropy = _masked_mean(
        -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=0),
        usage_mask,
    )
    losses = {
        "L_router_reconstruction": reconstruction,
        "L_router_oracle": oracle_ce * float(lambda_oracle),
        "L_router_starvation": starvation * float(lambda_starvation),
        "router_entropy": entropy.detach(),
        "oracle_target_entropy": _masked_mean(
            -(targets.clamp_min(1e-8) * targets.clamp_min(1e-8).log()).sum(
                dim=0
            ),
            usage_mask,
        ).detach(),
    }
    for index, role in enumerate(EXPERT_ROLES):
        losses["router_usage_{}".format(role)] = usage[index].detach()
        losses["oracle_usage_{}".format(role)] = _masked_mean(
            targets[index],
            usage_mask,
        ).detach()
    losses["L_router_total"] = (
        losses["L_router_reconstruction"]
        + losses["L_router_oracle"]
        + losses["L_router_starvation"]
    )
    return blended, targets, losses


class EndoMoeVolumeAwareRouter(nn.Module):
    def __init__(
        self,
        point_counts,
        gaussian_hidden_dim=64,
        pixel_hidden_dim=32,
    ):
        super().__init__()
        ordered_counts = OrderedDict(
            (role, int(point_counts[role]))
            for role in EXPERT_ROLES
        )
        if any(count <= 0 for count in ordered_counts.values()):
            raise ValueError("Every EndoMoe expert must contain Gaussians")
        self.point_counts = ordered_counts
        self.base_logits = nn.ParameterDict(
            {
                role: nn.Parameter(torch.zeros(count, 1))
                for role, count in ordered_counts.items()
            }
        )
        self.role_embedding = nn.Embedding(len(EXPERT_ROLES), 4)
        self.gaussian_feature_mlp = nn.Sequential(
            nn.Linear(16, gaussian_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(gaussian_hidden_dim, gaussian_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(gaussian_hidden_dim, 1),
        )
        self.pixel_router = PixelSpaceRouter(pixel_hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.role_embedding.weight, mean=0.0, std=0.02)
        linear_layers = [
            module
            for module in self.gaussian_feature_mlp
            if isinstance(module, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)
        self.pixel_router.reset_parameters()
        for parameter in self.base_logits.values():
            nn.init.zeros_(parameter)

    def gaussian_logits(self, role, routing_state, camera, time_value):
        normalized_role = str(role).lower()
        if normalized_role not in EXPERT_ROLES:
            raise ValueError("Unsupported router role: {}".format(role))
        canonical = routing_state["canonical_xyz"]
        deformed = routing_state["means3d"]
        if canonical.shape[0] != self.point_counts[normalized_role]:
            raise ValueError(
                "Router point count for '{}' does not match expert state".format(
                    normalized_role
                )
            )
        xyz_min = canonical.amin(dim=0, keepdim=True)
        xyz_max = canonical.amax(dim=0, keepdim=True)
        normalized_xyz = (
            (canonical - xyz_min)
            / (xyz_max - xyz_min).clamp_min(1e-6)
            * 2.0
            - 1.0
        )
        scene_scale = float(routing_state.get("scene_scale", 1.0) or 1.0)
        normalized_motion = (
            routing_state["motion"] / max(abs(scene_scale), 1e-6)
        ).clamp(-1.0, 1.0)
        camera_center = camera.camera_center.to(
            device=deformed.device,
            dtype=deformed.dtype,
        )
        view_direction = deformed - camera_center.unsqueeze(0)
        view_direction = view_direction / view_direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        opacity = routing_state["opacity"].clamp(1e-6, 1.0 - 1e-6)
        scales = routing_state.get("scales")
        if scales is None:
            scale_feature = deformed.new_zeros((deformed.shape[0], 1))
        else:
            scale_feature = scales.clamp_min(1e-8).log().mean(
                dim=-1,
                keepdim=True,
            )
        time_feature = deformed.new_full(
            (deformed.shape[0], 1),
            float(time_value),
        )
        role_index = EXPERT_ROLES.index(normalized_role)
        role_features = self.role_embedding(
            torch.full(
                (deformed.shape[0],),
                role_index,
                device=deformed.device,
                dtype=torch.long,
            )
        )
        features = torch.cat(
            (
                normalized_xyz,
                normalized_motion,
                view_direction,
                opacity,
                scale_feature,
                time_feature,
                role_features,
            ),
            dim=-1,
        )
        residual = self.gaussian_feature_mlp(features)
        return self.base_logits[normalized_role] + residual

    def route_pixels(
        self,
        expert_rgb,
        expert_depth,
        gaussian_prior,
        projected_motion,
        coverage,
        top_k=None,
    ):
        fallback = gaussian_prior.mean(dim=(1, 2), keepdim=True).clamp_min(0.0)
        if fallback.sum() <= 1e-8:
            fallback = torch.ones_like(fallback)
        fallback = fallback / fallback.sum().clamp_min(1e-8)
        weights, residual_logits = self.pixel_router(
            expert_rgb=expert_rgb,
            expert_depth=expert_depth,
            gaussian_prior=gaussian_prior,
            projected_motion=projected_motion,
            coverage=coverage,
            fallback_prior=fallback,
        )
        return sparsify_router_weights(weights, top_k), residual_logits
