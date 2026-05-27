from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .heterogeneous_moe_tracking import TrackingPhase


class CutGraphGating(nn.Module):
    def __init__(self, time_feature_dim: int) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.feature_dim = self.time_feature_dim + 8
        hidden_dim = max(16, self.time_feature_dim)
        self.scaffold_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.cut_gate_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.scaffold_head, self.cut_gate_head):
            linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(linear_layers[-1].bias)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "tracking_cut_graph": list(self.scaffold_head.parameters()) + list(self.cut_gate_head.parameters()),
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

    def _build_features(self, means3d: torch.Tensor, time_values: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        xyz_norm = self._normalize_xyz(means3d)
        radial = torch.norm(xyz_norm, dim=-1, keepdim=True)
        xyz_sq = xyz_norm.square()
        return torch.cat((time_features, time_values, xyz_norm, radial, xyz_sq), dim=-1)

    def forward(
        self,
        means3d: torch.Tensor,
        time_values: torch.Tensor,
        time_features: torch.Tensor,
        phase: TrackingPhase,
    ) -> Dict[str, torch.Tensor]:
        del phase
        gating_features = self._build_features(means3d, time_values, time_features)
        scaffold_logits = self.scaffold_head(gating_features)
        cut_gate_logits = self.cut_gate_head(gating_features)
        scaffold_weights = torch.softmax(scaffold_logits, dim=-1)
        cut_gate_values = torch.sigmoid(cut_gate_logits)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": scaffold_weights,
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": cut_gate_values,
            "xyz_norm": self._normalize_xyz(means3d),
        }
