from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cams_gs_tracking import CAMSGSTracking
from .heterogeneous_moe_tracking import TrackingPhase


def _fixed_phase(name: str, active_geo: int, active_vis: int, enable_visibility: bool) -> TrackingPhase:
    return TrackingPhase(
        name=name,
        active_geo=active_geo,
        active_vis=active_vis,
        enable_visibility=enable_visibility,
        temperature_geo=1.0,
        temperature_vis=1.0,
        use_sparse_geo=False,
        use_sparse_vis=False,
        topk_geo=1,
        topk_vis=1,
    )


class EndoMoEGaussianScheduler:
    def __init__(self, args) -> None:
        self.args = args

    def build(self, iteration: int, total_iterations: int) -> TrackingPhase:
        explicit_stage = getattr(self.args, "endomoeg_stage", None) or getattr(self.args, "cams_moe_stage", None)
        if explicit_stage:
            return self._phase_for_stage(str(explicit_stage))

        total_iterations = max(int(total_iterations), 1)
        expert_global_end = self._resolve_stage("endomoeg_expert_global_end", total_iterations, 0.20)
        expert_local_end = self._resolve_stage("endomoeg_expert_local_end", total_iterations, 0.40)
        expert_full_end = self._resolve_stage("endomoeg_expert_full_end", total_iterations, 0.60)
        router_only_end = self._resolve_stage("endomoeg_router_only_end", total_iterations, 0.85)

        expert_global_end = max(1, expert_global_end)
        expert_local_end = max(expert_global_end + 1, expert_local_end)
        expert_full_end = max(expert_local_end + 1, expert_full_end)
        router_only_end = max(expert_full_end + 1, router_only_end)

        if iteration < expert_global_end:
            return self._phase_for_stage("expert_global")
        if iteration < expert_local_end:
            return self._phase_for_stage("expert_local")
        if iteration < expert_full_end:
            return self._phase_for_stage("expert_full")
        if iteration < router_only_end:
            return self._phase_for_stage("router_only")
        return self._phase_for_stage("joint_finetune")

    def _resolve_stage(self, name: str, total_iterations: int, fraction: float) -> int:
        value = int(getattr(self.args, name, 0) or 0)
        if value > 0:
            return value
        return max(1, int(round(total_iterations * fraction)))

    def _phase_for_stage(self, stage: str) -> TrackingPhase:
        stage = stage.lower()
        if stage in {"expert_global", "global"}:
            return TrackingPhase(
                name="moe_expert_global",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="global",
                trainable_group_prefixes=("tracking_time_encoder", "tracking_expert_global"),
            )
        if stage in {"expert_local", "local"}:
            return TrackingPhase(
                name="moe_expert_local",
                active_geo=3,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="local",
                trainable_group_prefixes=("tracking_time_encoder", "tracking_expert_local"),
            )
        if stage in {"expert_full", "full"}:
            return TrackingPhase(
                name="moe_expert_full",
                active_geo=3,
                active_vis=2,
                enable_visibility=True,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="full",
                trainable_group_prefixes=("tracking_time_encoder", "tracking_expert_full"),
            )
        if stage in {"router", "router_only"}:
            return TrackingPhase(
                name="moe_router_only",
                active_geo=3,
                active_vis=2,
                enable_visibility=True,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                trainable_group_prefixes=("tracking_moe_router",),
            )
        return TrackingPhase(
            name="moe_joint_finetune",
            active_geo=3,
            active_vis=2,
            enable_visibility=True,
            temperature_geo=1.0,
            temperature_vis=1.0,
            use_sparse_geo=False,
            use_sparse_vis=False,
            topk_geo=1,
            topk_vis=1,
            trainable_group_prefixes=(
                "tracking_time_encoder",
                "tracking_moe_router",
                "tracking_expert_global",
                "tracking_expert_local",
                "tracking_expert_full",
            ),
            group_lr_scales={
                "tracking_expert_global": 0.1,
                "tracking_expert_local": 0.1,
                "tracking_expert_full": 0.1,
            },
        )


class VolumeAwareGaussianRouter(nn.Module):
    def __init__(self, time_feature_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        hidden_dim = int(hidden_dim or max(32, self.time_feature_dim))
        feature_dim = self.time_feature_dim + 3 + 1 + 3
        self.router = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [layer for layer in self.router if isinstance(layer, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(linear_layers[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(linear_layers[-1].bias)

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.xyz_max.copy_(xyz_max.detach().to(self.xyz_max.device, self.xyz_max.dtype).reshape(3))
        self.xyz_min.copy_(xyz_min.detach().to(self.xyz_min.device, self.xyz_min.dtype).reshape(3))

    def _normalize_xyz(self, means3d: torch.Tensor) -> torch.Tensor:
        xyz_max = self.xyz_max.to(device=means3d.device, dtype=means3d.dtype)
        xyz_min = self.xyz_min.to(device=means3d.device, dtype=means3d.dtype)
        extent = (xyz_max - xyz_min).clamp_min(1e-6)
        return (((means3d - xyz_min) / extent) * 2.0 - 1.0).clamp(-2.0, 2.0)

    def forward(
        self,
        means3d: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_features: torch.Tensor,
        expert_d_mu: torch.Tensor,
        phase: TrackingPhase,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_norms = expert_d_mu.norm(dim=-1)
        features = torch.cat((time_features, self._normalize_xyz(means3d), opacity_logits, motion_norms), dim=-1)
        logits = self.router(features)
        weights = torch.softmax(logits / max(float(phase.temperature_geo), 1e-6), dim=-1)

        forced = (phase.force_geo_expert or "").lower()
        forced_index = {"global": 0, "local": 1, "full": 2}.get(forced)
        if forced_index is not None:
            weights = torch.zeros_like(weights)
            weights[:, forced_index] = 1.0

        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_max_prob = weights.max(dim=-1).values
        top2 = torch.topk(weights, k=2, dim=-1).values
        route_margin = top2[:, 0] - top2[:, 1]
        return weights, entropy, route_max_prob, route_margin


class CAMSGSMoETracking(nn.Module):
    GEO_EXPERT_NAMES = ("global", "local", "full")
    VIS_EXPERT_NAMES = ("stable", "transient")

    def __init__(self, time_feature_dim: int, **kwargs) -> None:
        super().__init__()
        self.expert_global = CAMSGSTracking(time_feature_dim, **kwargs)
        self.expert_local = CAMSGSTracking(time_feature_dim, **kwargs)
        self.expert_full = CAMSGSTracking(time_feature_dim, **kwargs)
        self.router = VolumeAwareGaussianRouter(time_feature_dim, kwargs.get("moe_router_hidden_dim"))

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups: Dict[str, Iterable[nn.Parameter]] = {
            "tracking_moe_router": self.router.parameters(),
        }
        for expert_name, expert in self._experts().items():
            for group_name, params in expert.named_parameter_groups().items():
                groups[f"tracking_expert_{expert_name}_{group_name}"] = params
        return groups

    def _experts(self) -> Dict[str, CAMSGSTracking]:
        return {
            "global": self.expert_global,
            "local": self.expert_local,
            "full": self.expert_full,
        }

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.router.set_aabb(xyz_max, xyz_min)
        for expert in self._experts().values():
            expert.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        for expert in self._experts().values():
            yield from expert.iter_regularized_grids()

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        for expert in self._experts().values():
            expert.reset_parameters()

    def _run_experts(
        self,
        kwargs: Dict[str, Union[torch.Tensor, object]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        expert_phases = (
            _fixed_phase("endo_moe_global_expert", active_geo=1, active_vis=1, enable_visibility=False),
            _fixed_phase("endo_moe_local_expert", active_geo=3, active_vis=1, enable_visibility=False),
            _fixed_phase("endo_moe_full_expert", active_geo=3, active_vis=2, enable_visibility=True),
        )
        outputs = []
        for expert, expert_phase in zip(self._experts().values(), expert_phases):
            outputs.append(expert(phase=expert_phase, **kwargs))
        return outputs

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        time_features: torch.Tensor,
        scene_scale: torch.Tensor,
        phase: TrackingPhase,
        camera: object = None,
    ):
        kwargs = {
            "means3d": means3d,
            "scales": scales,
            "rotations": rotations,
            "opacity_logits": opacity_logits,
            "time_values": time_values,
            "time_features": time_features,
            "scene_scale": scene_scale,
            "camera": camera,
        }
        outputs = self._run_experts(kwargs)
        expert_means = torch.stack([output[0] for output in outputs], dim=1)
        expert_scales = torch.stack([output[1] for output in outputs], dim=1)
        expert_rotations = torch.stack([output[2] for output in outputs], dim=1)
        expert_opacity = torch.stack([output[3] for output in outputs], dim=1)
        expert_aux = [output[4] for output in outputs]
        expert_d_mu = expert_means - means3d.unsqueeze(1)

        pi_geo, entropy_geo, route_max_prob_geo, route_margin_geo = self.router(
            means3d=means3d,
            opacity_logits=opacity_logits,
            time_features=time_features,
            expert_d_mu=expert_d_mu,
            phase=phase,
        )
        weights = pi_geo.unsqueeze(-1)
        means_out = (expert_means * weights).sum(dim=1)
        scales_out = (expert_scales * weights).sum(dim=1)
        rotations_out = F.normalize((expert_rotations * weights).sum(dim=1), dim=-1)
        opacity_out = (expert_opacity * weights).sum(dim=1)

        d_mu = means_out - means3d
        d_scale = scales_out - scales
        d_rot = rotations_out[:, 1:] - rotations[:, 1:]
        d_opacity = opacity_out - opacity_logits

        vis_rgb_delta = torch.stack([aux["appearance_rgb_delta"] for aux in expert_aux], dim=1)
        vis_alpha = torch.stack([aux["visibility_alpha"] for aux in expert_aux], dim=1)
        lifecycle_alpha = torch.stack([aux["lifecycle_alpha"] for aux in expert_aux], dim=1)

        aux = {
            "means3d_canonical": means3d,
            "d_mu": d_mu,
            "d_rot": d_rot,
            "d_scale": d_scale,
            "d_opacity_logit": d_opacity,
            "global_motion": expert_d_mu[:, 0] * pi_geo[:, 0:1],
            "local_motion": expert_d_mu[:, 1] * pi_geo[:, 1:2],
            "cut_graph_motion": expert_d_mu[:, 2] * pi_geo[:, 2:3],
            "global_motion_norm": (expert_d_mu[:, 0] * pi_geo[:, 0:1]).norm(dim=-1, keepdim=True),
            "local_motion_norm": (expert_d_mu[:, 1] * pi_geo[:, 1:2]).norm(dim=-1, keepdim=True),
            "cut_graph_motion_norm": (expert_d_mu[:, 2] * pi_geo[:, 2:3]).norm(dim=-1, keepdim=True),
            "geo_expert_d_mu": expert_d_mu,
            "geo_expert_means3d": expert_means,
            "geo_expert_scales": expert_scales,
            "geo_expert_rotations": expert_rotations,
            "geo_expert_opacity_logits": expert_opacity,
            "pi_geo": pi_geo,
            "gaussian_pi_geo_prior": pi_geo,
            "pi_vis": expert_aux[-1]["pi_vis"],
            "gaussian_pi_vis_prior": expert_aux[-1]["pi_vis"],
            "entropy_geo": entropy_geo,
            "entropy_vis": expert_aux[-1]["entropy_vis"],
            "route_max_prob_geo": route_max_prob_geo,
            "route_margin_geo": route_margin_geo,
            "route_top1_geo_mean": route_max_prob_geo.mean(),
            "route_max_prob_vis": expert_aux[-1]["route_max_prob_vis"],
            "route_margin_vis": expert_aux[-1]["route_margin_vis"],
            "route_top1_vis_mean": expert_aux[-1]["route_top1_vis_mean"],
            "visibility_logits": expert_aux[-1]["visibility_logits"],
            "visibility_alpha": (vis_alpha * weights).sum(dim=1),
            "appearance_offsets": expert_aux[-1]["appearance_offsets"],
            "appearance_rgb_delta": (vis_rgb_delta * weights).sum(dim=1),
            "vis_expert_rgb_delta": vis_rgb_delta,
            "vis_expert_visibility_alpha": vis_alpha,
            "lifecycle_logits": expert_aux[-1]["lifecycle_logits"],
            "lifecycle_probs": expert_aux[-1]["lifecycle_probs"],
            "lifecycle_alpha": (lifecycle_alpha * weights).sum(dim=1),
            "lifecycle_expert_alpha": lifecycle_alpha,
            "tracking_phase_name": phase.name,
            "moe_router_logits": torch.log(pi_geo.clamp_min(1e-8)),
        }
        return means_out, scales_out, rotations_out, opacity_out, aux
