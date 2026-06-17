from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motion_scaffold import (
    _axis_angle_to_quaternion,
    _farthest_point_indices,
    _quaternion_multiply,
)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _reset_mlp(module: nn.Sequential, zero_output: bool = False) -> None:
    linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    if zero_output:
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value).clamp_min(1e-8))


class ContactSpacetimeExpert(nn.Module):
    def __init__(
        self,
        anchor_count: int = 512,
        chart_count: int = 3,
        hidden_dim: int = 64,
        time_frequencies: int = 4,
        max_parent_opacity_delta: float = 4.0,
        max_parent_rgb_delta: float = 0.1,
        max_child_rgb_delta: float = 0.2,
        max_spatial_offset_ratio: float = 0.02,
        max_velocity_ratio: float = 0.05,
        max_acceleration_ratio: float = 0.05,
        max_rotation_radians: float = 0.5,
        max_scale_delta: float = 0.1,
        initial_duration: float = 0.15,
    ) -> None:
        super().__init__()
        self.anchor_count = int(anchor_count)
        self.chart_count = int(chart_count)
        self.child_capacity = self.anchor_count * self.chart_count
        self.time_frequencies = int(time_frequencies)
        self.max_parent_opacity_delta = float(max_parent_opacity_delta)
        self.max_parent_rgb_delta = float(max_parent_rgb_delta)
        self.max_child_rgb_delta = float(max_child_rgb_delta)
        self.max_spatial_offset_ratio = float(max_spatial_offset_ratio)
        self.max_velocity_ratio = float(max_velocity_ratio)
        self.max_acceleration_ratio = float(max_acceleration_ratio)
        self.max_rotation_radians = float(max_rotation_radians)
        self.max_scale_delta = float(max_scale_delta)
        time_dim = 1 + 2 * self.time_frequencies
        self.parent_trunk = _build_mlp(3 + time_dim + 1, hidden_dim, hidden_dim)
        self.parent_opacity_head = nn.Linear(hidden_dim, 1)
        self.parent_appearance_head = nn.Linear(hidden_dim, 3)
        self.parent_support_head = nn.Linear(hidden_dim, 1)

        self.child_spatial_offset = nn.Parameter(
            torch.zeros(self.child_capacity, 3)
        )
        self.child_velocity = nn.Parameter(torch.zeros(self.child_capacity, 3))
        self.child_acceleration = nn.Parameter(
            torch.zeros(self.child_capacity, 3)
        )
        self.child_rotation_velocity = nn.Parameter(
            torch.zeros(self.child_capacity, 3)
        )
        self.child_scale_delta = nn.Parameter(
            torch.zeros(self.child_capacity, 3)
        )
        self.child_amplitude_raw = nn.Parameter(
            torch.zeros(self.child_capacity, 1)
        )
        self.child_rgb_delta = nn.Parameter(
            torch.zeros(self.child_capacity, 3)
        )
        self.child_center_offset = nn.Parameter(
            torch.zeros(self.child_capacity, 1)
        )
        initial_duration_tensor = torch.tensor(
            max(float(initial_duration) - 0.02, 1e-4)
        )
        duration_logit = _inverse_softplus(initial_duration_tensor)
        self.child_log_duration = nn.Parameter(
            torch.full(
                (self.child_capacity, 1),
                float(duration_logit.item()),
            )
        )
        self.register_buffer(
            "anchor_positions",
            torch.zeros(self.anchor_count, 3),
        )
        self.register_buffer(
            "anchor_parent_indices",
            torch.zeros(self.anchor_count, dtype=torch.long),
        )
        self.register_buffer(
            "child_anchor_indices",
            torch.zeros(self.child_capacity, dtype=torch.long),
        )
        self.register_buffer(
            "child_chart_centers",
            torch.zeros(self.child_capacity, 1),
        )
        self.register_buffer("active_anchors", torch.zeros((), dtype=torch.long))
        self.register_buffer("active_children", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_mlp(self.parent_trunk)
        for head in (
            self.parent_opacity_head,
            self.parent_appearance_head,
            self.parent_support_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(self.parent_support_head.bias, -4.0)
        for parameter in (
            self.child_spatial_offset,
            self.child_velocity,
            self.child_acceleration,
            self.child_rotation_velocity,
            self.child_scale_delta,
            self.child_amplitude_raw,
            self.child_rgb_delta,
            self.child_center_offset,
        ):
            nn.init.zeros_(parameter)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {"tracking_expert_refinement": self.parameters()}

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.xyz_max.copy_(
            xyz_max.detach().to(self.xyz_max.device, self.xyz_max.dtype).reshape(3)
        )
        self.xyz_min.copy_(
            xyz_min.detach().to(self.xyz_min.device, self.xyz_min.dtype).reshape(3)
        )

    def initialize_from_canonical(
        self,
        canonical_means: torch.Tensor,
        canonical_rotations: Optional[torch.Tensor] = None,
    ) -> None:
        del canonical_rotations
        if bool(self.initialized.item()):
            return
        with torch.no_grad():
            selected = _farthest_point_indices(
                canonical_means.detach(),
                self.anchor_count,
            )
            active_anchors = int(selected.numel())
            self.anchor_positions[:active_anchors].copy_(
                canonical_means.detach()[selected]
            )
            self.anchor_parent_indices[:active_anchors].copy_(selected)
            chart_centers = torch.linspace(
                0.2,
                0.8,
                self.chart_count,
                device=canonical_means.device,
                dtype=canonical_means.dtype,
            )
            child_count = active_anchors * self.chart_count
            anchor_indices = torch.arange(
                active_anchors,
                device=canonical_means.device,
            ).repeat_interleave(self.chart_count)
            centers = chart_centers.repeat(active_anchors).unsqueeze(-1)
            self.child_anchor_indices[:child_count].copy_(anchor_indices)
            self.child_chart_centers[:child_count].copy_(centers)
            self.active_anchors.fill_(active_anchors)
            self.active_children.fill_(child_count)
            self.initialized.fill_(True)

    def _normalize_xyz(self, means3d: torch.Tensor) -> torch.Tensor:
        xyz_max = self.xyz_max.to(means3d.device, means3d.dtype)
        xyz_min = self.xyz_min.to(means3d.device, means3d.dtype)
        extent = (xyz_max - xyz_min).clamp_min(1e-6)
        return ((means3d - xyz_min) / extent) * 2.0 - 1.0

    def _encode_time(self, time_values: torch.Tensor) -> torch.Tensor:
        time_value = time_values[:, :1]
        frequencies = 2.0 ** torch.arange(
            self.time_frequencies,
            device=time_value.device,
            dtype=time_value.dtype,
        )
        angles = time_value * frequencies.view(1, -1) * torch.pi
        return torch.cat(
            (time_value, torch.sin(angles), torch.cos(angles)),
            dim=-1,
        )

    @staticmethod
    def _camera_boundary_support(
        means3d: torch.Tensor,
        camera: object = None,
    ) -> torch.Tensor:
        fallback = means3d.new_zeros((means3d.shape[0], 1))
        if camera is None or not hasattr(camera, "mask"):
            return fallback
        try:
            mask = camera.mask.to(means3d.device, means3d.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
            invalid = 1.0 - mask.clamp(0.0, 1.0)
            dilated = F.max_pool2d(invalid, kernel_size=9, stride=1, padding=4)
            boundary = (dilated - invalid).clamp(0.0, 1.0)
            homogeneous = torch.cat(
                (means3d, torch.ones_like(means3d[:, :1])),
                dim=-1,
            )
            projection = camera.full_proj_transform.to(
                means3d.device,
                means3d.dtype,
            )
            clip = homogeneous @ projection.T
            screen = clip[:, :2] / clip[:, 3:4].abs().clamp_min(1e-6)
            grid = screen.view(1, -1, 1, 2)
            sampled = F.grid_sample(
                boundary,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            return sampled.view(-1, 1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return fallback

    @staticmethod
    def _straight_through_unit_gate(raw: torch.Tensor) -> torch.Tensor:
        forward_value = raw.clamp(0.0, 1.0)
        surrogate = torch.sigmoid(raw)
        return forward_value.detach() + surrogate - surrogate.detach()

    def forward(
        self,
        canonical_means3d: torch.Tensor,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        if not bool(self.initialized.item()):
            raise RuntimeError(
                "Contact spacetime bank must be initialized from canonical "
                "Gaussians before optimizer construction"
            )
        normalized_scale = torch.as_tensor(
            scene_scale,
            device=means3d.device,
            dtype=means3d.dtype,
        ).reshape(()).abs().clamp_min(1e-6)
        boundary_support = self._camera_boundary_support(means3d, camera)
        parent_features = torch.cat(
            (
                self._normalize_xyz(canonical_means3d),
                self._encode_time(time_values),
                boundary_support,
            ),
            dim=-1,
        )
        parent_hidden = self.parent_trunk(parent_features)
        support_logit = self.parent_support_head(parent_hidden)
        contact_support = torch.sigmoid(support_logit + 2.0 * boundary_support)
        d_opacity = (
            torch.tanh(self.parent_opacity_head(parent_hidden))
            * self.max_parent_opacity_delta
            * contact_support
        )
        parent_rgb_delta = (
            torch.tanh(self.parent_appearance_head(parent_hidden))
            * self.max_parent_rgb_delta
            * contact_support
        )

        child_count = int(self.active_children.item())
        child_anchor = self.child_anchor_indices[:child_count]
        child_parent = self.anchor_parent_indices[: int(
            self.active_anchors.item()
        )][child_anchor]
        time_value = time_values[:, :1].mean().reshape(1, 1)
        centers = (
            self.child_chart_centers[:child_count]
            + 0.1 * torch.tanh(self.child_center_offset[:child_count])
        ).clamp(0.0, 1.0)
        duration = (
            F.softplus(self.child_log_duration[:child_count]) + 0.02
        ).clamp_max(0.5)
        delta_time = time_value - centers
        temporal_rbf = torch.exp(
            -0.5 * (delta_time / duration.clamp_min(1e-4)).square()
        )
        amplitude = self._straight_through_unit_gate(
            self.child_amplitude_raw[:child_count]
        )
        parent_alpha = torch.sigmoid(opacity_logits[child_parent])
        child_alpha = (parent_alpha * amplitude * temporal_rbf).clamp(0.0, 1.0)
        spatial_offset = torch.tanh(
            self.child_spatial_offset[:child_count]
        ) * (self.max_spatial_offset_ratio * normalized_scale)
        velocity = torch.tanh(
            self.child_velocity[:child_count]
        ) * (self.max_velocity_ratio * normalized_scale)
        acceleration = torch.tanh(
            self.child_acceleration[:child_count]
        ) * (self.max_acceleration_ratio * normalized_scale)
        rotation_velocity = torch.tanh(
            self.child_rotation_velocity[:child_count]
        ) * self.max_rotation_radians
        scale_delta = torch.tanh(
            self.child_scale_delta[:child_count]
        ) * self.max_scale_delta
        child_canonical_means = (
            canonical_means3d[child_parent]
            + spatial_offset
        )
        child_means = (
            means3d[child_parent]
            + spatial_offset
            + velocity * delta_time
            + 0.5
            * acceleration
            * delta_time.square()
        )
        rotation_delta = _axis_angle_to_quaternion(
            rotation_velocity * delta_time
        )
        child_rotations = F.normalize(
            _quaternion_multiply(rotation_delta, rotations[child_parent]),
            dim=-1,
        )
        child_scales = (
            scales[child_parent] + scale_delta
        )
        child_rgb_delta = self.max_child_rgb_delta * torch.tanh(
            self.child_rgb_delta[:child_count]
        )
        anchor_boundary_support = self._camera_boundary_support(
            child_means,
            camera,
        )
        active_child_weight = amplitude * temporal_rbf
        auxiliary_support = (
            active_child_weight * anchor_boundary_support.detach()
        ).clamp(0.0, 1.0)
        transient_probability = contact_support
        pi_vis = torch.cat(
            (1.0 - transient_probability, transient_probability),
            dim=-1,
        )
        lifecycle_logits = torch.cat(
            (torch.zeros_like(support_logit), support_logit),
            dim=-1,
        )
        lifecycle_probs = torch.softmax(lifecycle_logits, dim=-1)
        return {
            "means3d": means3d,
            "scales": scales,
            "rotations": rotations,
            "opacity_logits": opacity_logits + d_opacity,
            "residual_support": contact_support.detach(),
            "d_mu": torch.zeros_like(means3d),
            "d_scale": torch.zeros_like(scales),
            "d_rot": means3d.new_zeros((means3d.shape[0], 3)),
            "d_opacity_logit": d_opacity,
            "appearance_offsets": parent_rgb_delta,
            "appearance_rgb_delta": parent_rgb_delta,
            "visibility_alpha": torch.sigmoid(d_opacity),
            "visibility_logits": lifecycle_logits,
            "transient_probability": transient_probability,
            "pi_vis": pi_vis,
            "entropy_vis": -(
                pi_vis.clamp_min(1e-8) * pi_vis.clamp_min(1e-8).log()
            ).sum(dim=-1).mean(),
            "route_max_prob_vis": pi_vis.max(dim=-1).values,
            "route_margin_vis": (pi_vis[:, 0] - pi_vis[:, 1]).abs(),
            "lifecycle_logits": lifecycle_logits,
            "lifecycle_probs": lifecycle_probs,
            "lifecycle_alpha": lifecycle_probs[:, :1],
            "auxiliary_means3d": child_means,
            "auxiliary_canonical_means3d": child_canonical_means,
            "auxiliary_scales": child_scales,
            "auxiliary_rotations": child_rotations,
            "auxiliary_opacity": child_alpha,
            "auxiliary_parent_indices": child_parent,
            "auxiliary_rgb_delta": child_rgb_delta,
            "auxiliary_residual_support": auxiliary_support.detach(),
            "auxiliary_temporal_rbf": temporal_rbf,
            "auxiliary_amplitude": amplitude,
            "auxiliary_contact_target": anchor_boundary_support,
            "contact_parent_support": contact_support,
            "contact_bank_sparsity": amplitude.square().mean(),
            "contact_bank_locality": (
                active_child_weight
                * (1.0 - anchor_boundary_support.detach())
            ).mean(),
            "contact_bank_acceleration": (
                acceleration / normalized_scale
            ).square().mean(),
            "contact_bank_spatial_offset": (
                spatial_offset / normalized_scale
            ).square().mean(),
            "contact_bank_duration": duration.mean(),
        }
