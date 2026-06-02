from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heterogeneous_moe_tracking import TrackingPhase


class MotionDecomposition(nn.Module):
    def __init__(
        self,
        time_feature_dim: int,
        *,
        max_disp_global_ratio: float = 0.01,
        max_disp_local_ratio: float = 0.03,
        max_rot_delta: float = 0.05,
        max_scale_delta: float = 0.05,
        max_opacity_delta: float = 4.0,
        enable_scale: bool = True,
        enable_rotation: bool = True,
        enable_opacity: bool = True,
    ) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.local_feature_dim = self.time_feature_dim + 9
        hidden_dim = max(16, self.time_feature_dim)
        self.max_disp_global_ratio = float(max_disp_global_ratio)
        self.max_disp_local_ratio = float(max_disp_local_ratio)
        self.max_rot_delta = float(max_rot_delta)
        self.max_scale_delta = float(max_scale_delta)
        self.max_opacity_delta = float(max_opacity_delta)
        self.enable_scale = bool(enable_scale)
        self.enable_rotation = bool(enable_rotation)
        self.enable_opacity = bool(enable_opacity)

        self.global_motion = nn.Linear(self.time_feature_dim, 3)
        self.local_motion = nn.Sequential(
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.cut_graph_motion = nn.Sequential(
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.opacity_head = nn.Sequential(
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.global_motion.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.global_motion.bias)
        for module in (self.local_motion, self.cut_graph_motion, self.rotation_head, self.scale_head, self.opacity_head):
            linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(linear_layers[-1].bias)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "tracking_motion_global": list(self.global_motion.parameters()),
            "tracking_motion_local": list(self.local_motion.parameters())
            + list(self.cut_graph_motion.parameters())
            + list(self.rotation_head.parameters())
            + list(self.scale_head.parameters())
            + list(self.opacity_head.parameters()),
        }

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.xyz_max.copy_(xyz_max.detach().to(self.xyz_max.device, self.xyz_max.dtype).reshape(3))
        self.xyz_min.copy_(xyz_min.detach().to(self.xyz_min.device, self.xyz_min.dtype).reshape(3))

    def iter_regularized_grids(self):
        return iter(())

    def _normalize_xyz(self, means3d: torch.Tensor) -> torch.Tensor:
        xyz_max = self.xyz_max.to(device=means3d.device, dtype=means3d.dtype)
        xyz_min = self.xyz_min.to(device=means3d.device, dtype=means3d.dtype)
        extent = (xyz_max - xyz_min).clamp_min(1e-6)
        xyz_norm = ((means3d - xyz_min) / extent) * 2.0 - 1.0
        return xyz_norm.clamp(-2.0, 2.0)

    def _build_local_features(self, means3d: torch.Tensor, time_features: torch.Tensor, gating_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        xyz_norm = gating_state.get("xyz_norm")
        if xyz_norm is None:
            xyz_norm = self._normalize_xyz(means3d)
        scaffold_weights = gating_state["scaffold_weights"]
        cut_gate_values = gating_state["cut_gate_values"]
        return torch.cat((time_features, xyz_norm, scaffold_weights, cut_gate_values), dim=-1)

    def _apply_quaternion_delta(self, rotations: torch.Tensor, d_rot: torch.Tensor) -> torch.Tensor:
        delta_xyz = torch.tanh(d_rot) * self.max_rot_delta * 0.5
        delta_w = torch.sqrt(torch.clamp(1.0 - (delta_xyz ** 2).sum(dim=-1, keepdim=True), min=1e-8))
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

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_features: torch.Tensor,
        scene_scale: torch.Tensor,
        gating_state: Dict[str, torch.Tensor],
        phase: TrackingPhase,
    ) -> Dict[str, torch.Tensor]:
        scale = torch.as_tensor(scene_scale, device=means3d.device, dtype=means3d.dtype).reshape(()).abs().clamp_min(1e-6)

        global_delta = torch.tanh(self.global_motion(time_features)) * (self.max_disp_global_ratio * scale)
        local_features = self._build_local_features(means3d, time_features, gating_state)
        local_delta = torch.tanh(self.local_motion(local_features)) * (self.max_disp_local_ratio * scale)
        cut_graph_delta = torch.tanh(self.cut_graph_motion(local_features)) * (self.max_disp_local_ratio * scale)
        global_mix = gating_state.get("global_mix")
        local_mix = gating_state.get("local_mix")
        cut_graph_mix = gating_state.get("cut_graph_mix")
        if global_mix is None or local_mix is None or cut_graph_mix is None:
            mixes = torch.zeros((means3d.shape[0], 3), device=means3d.device, dtype=means3d.dtype)
            mixes[:, 0] = 1.0
        else:
            mixes = torch.cat((global_mix, local_mix, cut_graph_mix), dim=-1)

        active_geo = max(1, min(int(getattr(phase, "active_geo", mixes.shape[-1])), mixes.shape[-1]))
        active_mask = torch.zeros_like(mixes)
        active_mask[:, :active_geo] = 1.0
        mixes = mixes * active_mask

        mixes_sum = mixes.sum(dim=-1, keepdim=True)
        fallback = torch.zeros_like(mixes)
        fallback[:, 0] = 1.0
        mixes = torch.where(mixes_sum > 1e-8, mixes / mixes_sum.clamp_min(1e-8), fallback)

        global_mix = mixes[:, 0:1]
        local_mix = mixes[:, 1:2]
        cut_graph_mix = mixes[:, 2:3]

        blended_global = global_delta * global_mix
        blended_local = local_delta * local_mix
        blended_cut_graph = cut_graph_delta * cut_graph_mix
        d_mu = blended_global + blended_local + blended_cut_graph
        geo_expert_d_mu = torch.stack((global_delta, local_delta, cut_graph_delta), dim=1)

        raw_d_rot = self.rotation_head(local_features)
        d_rot = torch.tanh(raw_d_rot) * self.max_rot_delta
        if self.enable_rotation:
            rotations_out = self._apply_quaternion_delta(rotations, raw_d_rot)
        else:
            d_rot = torch.zeros((means3d.shape[0], 3), device=means3d.device, dtype=means3d.dtype)
            rotations_out = rotations

        if self.enable_scale:
            d_scale = torch.tanh(self.scale_head(local_features)) * self.max_scale_delta
            scales_out = scales + d_scale
        else:
            d_scale = torch.zeros_like(scales)
            scales_out = scales

        if self.enable_opacity:
            d_opacity_logit = torch.tanh(self.opacity_head(local_features)) * self.max_opacity_delta
            opacity_out = opacity_logits + d_opacity_logit
        else:
            d_opacity_logit = torch.zeros_like(opacity_logits)
            opacity_out = opacity_logits

        geo_expert_means3d = torch.stack(
            (
                means3d + global_delta,
                means3d + local_delta,
                means3d + cut_graph_delta,
            ),
            dim=1,
        )
        geo_expert_scales = scales_out.unsqueeze(1).expand(-1, 3, -1)
        geo_expert_rotations = rotations_out.unsqueeze(1).expand(-1, 3, -1)
        geo_expert_opacity_logits = opacity_out.unsqueeze(1).expand(-1, 3, -1)

        return {
            "means3d": means3d + d_mu,
            "scales": scales_out,
            "rotations": rotations_out,
            "opacity_logits": opacity_out,
            "d_mu": d_mu,
            "d_rot": d_rot,
            "d_scale": d_scale,
            "d_opacity_logit": d_opacity_logit,
            "global_motion": blended_global,
            "local_motion": blended_local,
            "cut_graph_motion": blended_cut_graph,
            "geo_expert_d_mu": geo_expert_d_mu,
            "geo_expert_means3d": geo_expert_means3d,
            "geo_expert_scales": geo_expert_scales,
            "geo_expert_rotations": geo_expert_rotations,
            "geo_expert_opacity_logits": geo_expert_opacity_logits,
        }
