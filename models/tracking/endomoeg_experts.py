from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _reset_mlp(module: nn.Sequential, zero_output: bool = False, output_bias: float = 0.0) -> None:
    linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    if zero_output and linear_layers:
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.constant_(linear_layers[-1].bias, output_bias)


def _apply_quaternion_delta(
    rotations: torch.Tensor,
    delta_xyz: torch.Tensor,
) -> torch.Tensor:
    delta_norm_sq = delta_xyz.square().sum(dim=-1, keepdim=True)
    delta_w = torch.sqrt(torch.clamp(1.0 - delta_norm_sq, min=1e-8))
    delta_quat = torch.cat((delta_w, delta_xyz), dim=-1)
    rw, rx, ry, rz = rotations.unbind(dim=-1)
    dw, dx, dy, dz = delta_quat.unbind(dim=-1)
    updated = torch.stack(
        (
            rw * dw - rx * dx - ry * dy - rz * dz,
            rw * dx + rx * dw + ry * dz - rz * dy,
            rw * dy - rx * dz + ry * dw + rz * dx,
            rw * dz + rx * dy - ry * dx + rz * dw,
        ),
        dim=-1,
    )
    return F.normalize(updated, dim=-1)


class EndoMoeResidualExpert(nn.Module):
    def __init__(self, time_feature_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.hidden_dim = int(hidden_dim or max(32, self.time_feature_dim))
        self.time_encoder = _build_mlp(1, self.hidden_dim, self.time_feature_dim)
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.xyz_max.copy_(xyz_max.detach().to(self.xyz_max.device, self.xyz_max.dtype).reshape(3))
        self.xyz_min.copy_(xyz_min.detach().to(self.xyz_min.device, self.xyz_min.dtype).reshape(3))

    def iter_regularized_grids(self):
        return iter(())

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {"residual": self.parameters()}

    def _encode_time(self, time_values: torch.Tensor) -> torch.Tensor:
        return self.time_encoder(time_values[:, :1])

    def _normalize_xyz(self, means3d: torch.Tensor) -> torch.Tensor:
        xyz_max = self.xyz_max.to(device=means3d.device, dtype=means3d.dtype)
        xyz_min = self.xyz_min.to(device=means3d.device, dtype=means3d.dtype)
        extent = (xyz_max - xyz_min).clamp_min(1e-6)
        return (((means3d - xyz_min) / extent) * 2.0 - 1.0).clamp(-2.0, 2.0)

    @staticmethod
    def _identity_aux(reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        count = reference.shape[0]
        zeros_rgb = reference.new_zeros((count, 3))
        ones_alpha = reference.new_ones((count, 1))
        pi_vis = reference.new_zeros((count, 2))
        pi_vis[:, 0] = 1.0
        lifecycle_probs = pi_vis.clone()
        return {
            "appearance_offsets": zeros_rgb,
            "appearance_rgb_delta": zeros_rgb,
            "visibility_alpha": ones_alpha,
            "transient_probability": reference.new_zeros((count, 1)),
            "visibility_logits": reference.new_zeros((count, 2)),
            "pi_vis": pi_vis,
            "entropy_vis": reference.new_zeros(()),
            "route_max_prob_vis": reference.new_ones((count,)),
            "route_margin_vis": reference.new_ones((count,)),
            "route_top1_vis_mean": reference.new_ones(()),
            "lifecycle_logits": reference.new_zeros((count, 2)),
            "lifecycle_probs": lifecycle_probs,
            "lifecycle_alpha": ones_alpha,
        }


class GlobalSmoothExpert(EndoMoeResidualExpert):
    def __init__(
        self,
        time_feature_dim: int,
        max_disp_ratio: float = 0.01,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__(time_feature_dim, hidden_dim)
        self.max_disp_ratio = float(max_disp_ratio)
        self.motion_head = _build_mlp(self.time_feature_dim, self.hidden_dim, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_mlp(self.time_encoder)
        _reset_mlp(self.motion_head, zero_output=True)

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        del camera
        scale = torch.as_tensor(scene_scale, device=means3d.device, dtype=means3d.dtype).reshape(()).abs().clamp_min(1e-6)
        time_features = self._encode_time(time_values)
        d_mu = torch.tanh(self.motion_head(time_features)) * (self.max_disp_ratio * scale)
        zeros_scale = torch.zeros_like(scales)
        zeros_rot = means3d.new_zeros((means3d.shape[0], 3))
        zeros_opacity = torch.zeros_like(opacity_logits)
        output = {
            "means3d": means3d + d_mu,
            "scales": scales,
            "rotations": rotations,
            "opacity_logits": opacity_logits,
            "d_mu": d_mu,
            "d_scale": zeros_scale,
            "d_rot": zeros_rot,
            "d_opacity_logit": zeros_opacity,
        }
        output.update(self._identity_aux(means3d))
        return output


class TissueLocalExpert(EndoMoeResidualExpert):
    def __init__(
        self,
        time_feature_dim: int,
        max_disp_ratio: float = 0.03,
        max_rot_delta: float = 0.05,
        max_scale_delta: float = 0.05,
        hidden_dim: Optional[int] = None,
        enable_rotation: bool = True,
        enable_scale: bool = True,
    ) -> None:
        super().__init__(time_feature_dim, hidden_dim)
        self.max_disp_ratio = float(max_disp_ratio)
        self.max_rot_delta = float(max_rot_delta)
        self.max_scale_delta = float(max_scale_delta)
        self.enable_rotation = bool(enable_rotation)
        self.enable_scale = bool(enable_scale)
        spatial_dim = 7
        self.trunk = _build_mlp(self.time_feature_dim + spatial_dim, self.hidden_dim, self.hidden_dim)
        self.motion_head = nn.Linear(self.hidden_dim, 3)
        self.rotation_head = nn.Linear(self.hidden_dim, 3)
        self.scale_head = nn.Linear(self.hidden_dim, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_mlp(self.time_encoder)
        _reset_mlp(self.trunk)
        for head in (self.motion_head, self.rotation_head, self.scale_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _build_features(self, means3d: torch.Tensor, time_values: torch.Tensor) -> torch.Tensor:
        xyz = self._normalize_xyz(means3d)
        spatial = torch.cat((xyz, xyz.square(), xyz.norm(dim=-1, keepdim=True)), dim=-1)
        return torch.cat((self._encode_time(time_values), spatial), dim=-1)

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        del camera
        scale = torch.as_tensor(scene_scale, device=means3d.device, dtype=means3d.dtype).reshape(()).abs().clamp_min(1e-6)
        hidden = self.trunk(self._build_features(means3d, time_values))
        d_mu = torch.tanh(self.motion_head(hidden)) * (self.max_disp_ratio * scale)

        if self.enable_rotation:
            d_rot = torch.tanh(self.rotation_head(hidden)) * self.max_rot_delta
            rotations_out = _apply_quaternion_delta(rotations, d_rot * 0.5)
        else:
            d_rot = means3d.new_zeros((means3d.shape[0], 3))
            rotations_out = rotations

        if self.enable_scale:
            d_scale = torch.tanh(self.scale_head(hidden)) * self.max_scale_delta
            scales_out = scales + d_scale
        else:
            d_scale = torch.zeros_like(scales)
            scales_out = scales

        output = {
            "means3d": means3d + d_mu,
            "scales": scales_out,
            "rotations": rotations_out,
            "opacity_logits": opacity_logits,
            "d_mu": d_mu,
            "d_scale": d_scale,
            "d_rot": d_rot,
            "d_opacity_logit": torch.zeros_like(opacity_logits),
        }
        output.update(self._identity_aux(means3d))
        return output


class ToolContactExpert(TissueLocalExpert):
    def __init__(
        self,
        time_feature_dim: int,
        max_disp_ratio: float = 0.03,
        max_rot_delta: float = 0.05,
        max_scale_delta: float = 0.05,
        max_opacity_delta: float = 4.0,
        hidden_dim: Optional[int] = None,
        enable_rotation: bool = True,
        enable_scale: bool = True,
        enable_opacity: bool = True,
    ) -> None:
        EndoMoeResidualExpert.__init__(self, time_feature_dim, hidden_dim)
        self.max_disp_ratio = float(max_disp_ratio)
        self.max_rot_delta = float(max_rot_delta)
        self.max_scale_delta = float(max_scale_delta)
        self.max_opacity_delta = float(max_opacity_delta)
        self.enable_rotation = bool(enable_rotation)
        self.enable_scale = bool(enable_scale)
        self.enable_opacity = bool(enable_opacity)
        feature_dim = self.time_feature_dim + 14
        self.trunk = _build_mlp(feature_dim, self.hidden_dim, self.hidden_dim)
        self.motion_head = nn.Linear(self.hidden_dim, 3)
        self.rotation_head = nn.Linear(self.hidden_dim, 3)
        self.scale_head = nn.Linear(self.hidden_dim, 3)
        self.opacity_head = nn.Linear(self.hidden_dim, 1)
        self.visibility_head = nn.Linear(self.hidden_dim, 1)
        self.transient_head = nn.Linear(self.hidden_dim, 1)
        self.appearance_head = nn.Linear(self.hidden_dim, 3)
        self.lifecycle_head = nn.Linear(self.hidden_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_mlp(self.time_encoder)
        _reset_mlp(self.trunk)
        for head in (
            self.motion_head,
            self.rotation_head,
            self.scale_head,
            self.opacity_head,
            self.appearance_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.visibility_head.weight)
        nn.init.constant_(self.visibility_head.bias, 12.0)
        nn.init.zeros_(self.transient_head.weight)
        nn.init.constant_(self.transient_head.bias, -12.0)
        nn.init.zeros_(self.lifecycle_head.weight)
        with torch.no_grad():
            self.lifecycle_head.bias.copy_(torch.tensor((12.0, -12.0), dtype=self.lifecycle_head.bias.dtype))

    @staticmethod
    def _camera_features(means3d: torch.Tensor, camera: object = None) -> torch.Tensor:
        count = means3d.shape[0]
        fallback = means3d.new_zeros((count, 6))
        if camera is None:
            return fallback
        try:
            camera_center = camera.camera_center.to(means3d.device, means3d.dtype)
            view_direction = means3d - camera_center.unsqueeze(0)
            view_direction = view_direction / view_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            homogeneous = torch.cat((means3d, torch.ones_like(means3d[:, :1])), dim=-1)
            view_matrix = camera.world_view_transform.to(means3d.device, means3d.dtype)
            camera_depth = (homogeneous @ view_matrix.T)[:, 2:3]
            projection = camera.full_proj_transform.to(means3d.device, means3d.dtype)
            clip = homogeneous @ projection.T
            screen = clip[:, :2] / clip[:, 3:4].abs().clamp_min(1e-6)
            return torch.cat((view_direction, camera_depth, screen), dim=-1)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return fallback

    def _build_features(
        self,
        means3d: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        camera: object = None,
    ) -> torch.Tensor:
        xyz = self._normalize_xyz(means3d)
        spatial = torch.cat((xyz, xyz.square(), xyz.norm(dim=-1, keepdim=True)), dim=-1)
        return torch.cat(
            (
                self._encode_time(time_values),
                spatial,
                opacity_logits,
                self._camera_features(means3d, camera),
            ),
            dim=-1,
        )

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        scale = torch.as_tensor(scene_scale, device=means3d.device, dtype=means3d.dtype).reshape(()).abs().clamp_min(1e-6)
        hidden = self.trunk(self._build_features(means3d, opacity_logits, time_values, camera))
        d_mu = torch.tanh(self.motion_head(hidden)) * (self.max_disp_ratio * scale)

        if self.enable_rotation:
            d_rot = torch.tanh(self.rotation_head(hidden)) * self.max_rot_delta
            rotations_out = _apply_quaternion_delta(rotations, d_rot * 0.5)
        else:
            d_rot = means3d.new_zeros((means3d.shape[0], 3))
            rotations_out = rotations

        if self.enable_scale:
            d_scale = torch.tanh(self.scale_head(hidden)) * self.max_scale_delta
            scales_out = scales + d_scale
        else:
            d_scale = torch.zeros_like(scales)
            scales_out = scales

        visibility_logit = self.visibility_head(hidden)
        visibility_alpha = torch.sigmoid(visibility_logit)
        transient_logit = self.transient_head(hidden)
        transient_prob = torch.sigmoid(transient_logit)
        lifecycle_logits = self.lifecycle_head(hidden)
        lifecycle_probs = torch.softmax(lifecycle_logits, dim=-1)
        lifecycle_alpha = lifecycle_probs[:, :1]
        opacity_gate = (visibility_alpha * lifecycle_alpha).clamp(1e-6, 1.0)
        initial_visibility = torch.sigmoid(opacity_gate.new_tensor(12.0))
        initial_lifecycle = torch.softmax(opacity_gate.new_tensor((12.0, -12.0)), dim=0)[0]
        initial_opacity_gate = (initial_visibility * initial_lifecycle).clamp_min(1e-6)
        relative_opacity_gate = (opacity_gate / initial_opacity_gate).clamp(1e-4, 1.0)

        if self.enable_opacity:
            d_opacity = torch.tanh(self.opacity_head(hidden)) * self.max_opacity_delta
        else:
            d_opacity = torch.zeros_like(opacity_logits)
        dynamic_alpha = torch.sigmoid(opacity_logits + d_opacity) * relative_opacity_gate
        opacity_out = torch.logit(dynamic_alpha.clamp(1e-6, 1.0 - 1e-6))
        effective_d_opacity = opacity_out - opacity_logits

        appearance_offsets = self.appearance_head(hidden)
        appearance_rgb_delta = transient_prob * (0.1 * torch.tanh(appearance_offsets))
        pi_vis = torch.cat((1.0 - transient_prob, transient_prob), dim=-1)
        entropy_vis = -(pi_vis.clamp_min(1e-8) * pi_vis.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_max_prob_vis = pi_vis.max(dim=-1).values
        route_margin_vis = (pi_vis[:, 0] - pi_vis[:, 1]).abs()

        return {
            "means3d": means3d + d_mu,
            "scales": scales_out,
            "rotations": rotations_out,
            "opacity_logits": opacity_out,
            "d_mu": d_mu,
            "d_scale": d_scale,
            "d_rot": d_rot,
            "d_opacity_logit": effective_d_opacity,
            "raw_d_opacity_logit": d_opacity,
            "appearance_offsets": appearance_offsets,
            "appearance_rgb_delta": appearance_rgb_delta,
            "visibility_alpha": visibility_alpha,
            "visibility_logits": torch.cat((torch.zeros_like(visibility_logit), visibility_logit), dim=-1),
            "transient_logit": transient_logit,
            "transient_probability": transient_prob,
            "pi_vis": pi_vis,
            "entropy_vis": entropy_vis,
            "route_max_prob_vis": route_max_prob_vis,
            "route_margin_vis": route_margin_vis,
            "route_top1_vis_mean": route_max_prob_vis.mean(),
            "lifecycle_logits": lifecycle_logits,
            "lifecycle_probs": lifecycle_probs,
            "lifecycle_alpha": lifecycle_alpha,
        }
