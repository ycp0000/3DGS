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
        feature_dim = self.time_feature_dim + 2

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

    def _compute_view_direction(self, means3d: torch.Tensor, camera_center: torch.Tensor) -> torch.Tensor:
        """Compute normalized view direction vectors."""
        dir_vec = means3d - camera_center.unsqueeze(0)
        return dir_vec / dir_vec.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def _compute_camera_depth(self, means3d: torch.Tensor, world_view_transform: torch.Tensor) -> torch.Tensor:
        """Compute depth in camera space."""
        means3d_homogeneous = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=-1)
        view_coords = means3d_homogeneous @ world_view_transform.T
        return view_coords[:, 2:3]

    def _compute_screen_projection(self, means3d: torch.Tensor, full_proj_transform: torch.Tensor,
                                     image_width: int, image_height: int) -> torch.Tensor:
        """Compute normalized screen coordinates."""
        means3d_homogeneous = torch.cat([means3d, torch.ones_like(means3d[:, :1])], dim=-1)
        points_proj = means3d_homogeneous @ full_proj_transform.T
        points_proj = points_proj / points_proj[:, 3:4].clamp_min(1e-6)
        screen_x = points_proj[:, 0] / image_width * 2 - 1
        screen_y = points_proj[:, 1] / image_height * 2 - 1
        return torch.stack([screen_x, screen_y], dim=-1)

    def _build_features(
        self,
        time_features: torch.Tensor,
        means3d: torch.Tensor,
        opacity: torch.Tensor,
        d_mu_norm: torch.Tensor,
        camera: object = None,
    ) -> torch.Tensor:
        """Build view-dependent visibility router features."""
        features = [time_features, opacity, d_mu_norm]

        if camera is not None:
            try:
                camera_center = camera.camera_center.to(means3d.device, means3d.dtype)
                view_dir = self._compute_view_direction(means3d, camera_center)
                features.append(view_dir)

                world_view_transform = camera.world_view_transform.to(means3d.device, means3d.dtype)
                cam_depth = self._compute_camera_depth(means3d, world_view_transform)
                features.append(cam_depth)

                full_proj_transform = camera.full_proj_transform.to(means3d.device, means3d.dtype)
                screen_proj = self._compute_screen_projection(
                    means3d, full_proj_transform,
                    int(camera.image_width), int(camera.image_height)
                )
                features.append(screen_proj)
            except Exception:
                pass

        return torch.cat(features, dim=-1)

    def forward(
        self,
        time_features: torch.Tensor,
        means3d: torch.Tensor,
        opacity: torch.Tensor,
        d_mu: torch.Tensor,
        phase: TrackingPhase,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        d_mu_norm = d_mu.norm(dim=-1, keepdim=True)
        features = self._build_features(time_features, means3d, opacity, d_mu_norm, camera)
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
