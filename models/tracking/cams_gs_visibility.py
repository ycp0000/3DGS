from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .heterogeneous_moe_tracking import TrackingPhase


class VisibilityAppearanceHead(nn.Module):
    def __init__(self, time_feature_dim: int) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        hidden_dim = max(16, self.time_feature_dim)
        feature_dim = self.time_feature_dim + 6

        self.visibility_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.appearance_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.visibility_head, self.appearance_head):
            linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(linear_layers[-1].bias)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "tracking_visibility": list(self.visibility_head.parameters()),
            "tracking_appearance": list(self.appearance_head.parameters()),
        }

    def iter_regularized_grids(self):
        return iter(())

    def _build_features(
        self,
        time_features: torch.Tensor,
        gating_state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.cat(
            (
                time_features,
                gating_state["scaffold_weights"],
                gating_state["cut_gate_values"],
            ),
            dim=-1,
        )

    def forward(
        self,
        time_features: torch.Tensor,
        gating_state: Dict[str, torch.Tensor],
        phase: TrackingPhase,
    ) -> Dict[str, torch.Tensor]:
        features = self._build_features(time_features, gating_state)
        visibility_logits = self.visibility_head(features)
        appearance_offsets = self.appearance_head(features)
        appearance_rgb_delta = 0.1 * torch.tanh(appearance_offsets)

        if phase.enable_visibility:
            pi_vis = torch.softmax(visibility_logits, dim=-1)
            visibility_alpha = pi_vis[:, :1]
        else:
            pi_vis = torch.zeros_like(visibility_logits)
            pi_vis[:, 0] = 1.0
            visibility_alpha = torch.ones((visibility_logits.shape[0], 1), device=visibility_logits.device, dtype=visibility_logits.dtype)

        entropy_vis = -(pi_vis.clamp_min(1e-8) * pi_vis.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_max_prob_vis = pi_vis.max(dim=-1).values
        top2_vis = torch.topk(pi_vis, k=min(2, pi_vis.shape[-1]), dim=-1).values
        if top2_vis.shape[-1] > 1:
            route_margin_vis = top2_vis[:, 0] - top2_vis[:, 1]
        else:
            route_margin_vis = top2_vis[:, 0]

        return {
            "visibility_logits": visibility_logits,
            "appearance_offsets": appearance_offsets,
            "appearance_rgb_delta": appearance_rgb_delta,
            "pi_vis": pi_vis,
            "visibility_alpha": visibility_alpha,
            "entropy_vis": entropy_vis,
            "route_max_prob_vis": route_max_prob_vis,
            "route_margin_vis": route_margin_vis,
            "route_top1_vis_mean": route_max_prob_vis.mean(),
            "vis_expert_rgb_delta": torch.stack(
                (
                    torch.zeros_like(appearance_rgb_delta),
                    appearance_rgb_delta,
                ),
                dim=1,
            ),
            "vis_expert_visibility_alpha": torch.stack(
                (
                    torch.ones_like(visibility_alpha),
                    torch.zeros_like(visibility_alpha),
                ),
                dim=1,
            ),
        }
