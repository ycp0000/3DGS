from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .heterogeneous_moe_tracking import TrackingPhase


class GaussianLifecycleHead(nn.Module):
    def __init__(self, time_feature_dim: int) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        hidden_dim = max(16, self.time_feature_dim)
        feature_dim = self.time_feature_dim + 6

        self.lifecycle_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [layer for layer in self.lifecycle_head if isinstance(layer, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(linear_layers[-1].bias)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "tracking_lifecycle": list(self.lifecycle_head.parameters()),
        }

    def iter_regularized_grids(self):
        return iter(())

    def forward(
        self,
        time_features: torch.Tensor,
        gating_state: Dict[str, torch.Tensor],
        phase: TrackingPhase,
    ) -> Dict[str, torch.Tensor]:
        features = torch.cat(
            (
                time_features,
                gating_state["scaffold_weights"],
                gating_state["cut_gate_values"],
            ),
            dim=-1,
        )
        lifecycle_logits = self.lifecycle_head(features)
        if phase.name == "joint_finetune":
            lifecycle_probs = torch.softmax(lifecycle_logits, dim=-1)
            lifecycle_alpha = lifecycle_probs[:, 0:1]
        else:
            lifecycle_probs = torch.zeros_like(lifecycle_logits)
            lifecycle_probs[:, 0] = 1.0
            lifecycle_alpha = torch.ones((lifecycle_logits.shape[0], 1), device=lifecycle_logits.device, dtype=lifecycle_logits.dtype)
        return {
            "lifecycle_logits": lifecycle_logits,
            "lifecycle_probs": lifecycle_probs,
            "lifecycle_alpha": lifecycle_alpha,
        }
