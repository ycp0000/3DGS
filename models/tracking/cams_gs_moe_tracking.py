from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .endomoeg_experts import GlobalSmoothExpert, TissueLocalExpert, ToolContactExpert
from .heterogeneous_moe_tracking import TrackingPhase


_ENDOMOEG_STAGE_ALIASES = {
    "expert_global": "expert_global",
    "global": "expert_global",
    "expert_local": "expert_local",
    "local": "expert_local",
    "expert_full": "expert_full",
    "full": "expert_full",
    "router": "router_only",
    "router_only": "router_only",
    "joint": "joint_finetune",
    "joint_finetune": "joint_finetune",
}

_ENDOMOEG_STAGE_COMPONENTS = {
    "expert_global": (),
    "expert_local": ("shared_base",),
    "expert_full": ("shared_base",),
    "router_only": ("shared_base", "global", "local", "full"),
    "joint_finetune": ("shared_base", "global", "local", "full", "router"),
}


def normalize_endomoeg_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if not normalized:
        return ""
    if normalized not in _ENDOMOEG_STAGE_ALIASES:
        raise ValueError(f"Unsupported EndoMoe stage: {stage}")
    return _ENDOMOEG_STAGE_ALIASES[normalized]


def required_endomoeg_components(stage: str) -> Tuple[str, ...]:
    normalized = normalize_endomoeg_stage(stage)
    if not normalized:
        return ()
    return _ENDOMOEG_STAGE_COMPONENTS[normalized]


class EndoMoEGaussianScheduler:
    def __init__(self, args) -> None:
        self.args = args

    def build(self, iteration: int, total_iterations: int) -> TrackingPhase:
        explicit_stage = getattr(self.args, "endomoeg_stage", None) or getattr(self.args, "cams_moe_stage", None)
        if explicit_stage:
            return self._phase_for_stage(
                normalize_endomoeg_stage(str(explicit_stage)),
                iteration=iteration,
                stage_start=0,
                stage_end=total_iterations,
            )

        total_iterations = max(int(total_iterations), 1)
        expert_global_end = self._resolve_stage("endomoeg_expert_global_end", total_iterations, 2000)
        expert_local_end = self._resolve_stage("endomoeg_expert_local_end", total_iterations, 4500)
        expert_full_end = self._resolve_stage("endomoeg_expert_full_end", total_iterations, 7500)
        router_only_end = self._resolve_stage("endomoeg_router_only_end", total_iterations, 11000)

        expert_global_end = max(1, expert_global_end)
        expert_local_end = max(expert_global_end + 1, expert_local_end)
        expert_full_end = max(expert_local_end + 1, expert_full_end)
        router_only_end = max(expert_full_end + 1, router_only_end)

        if iteration < expert_global_end:
            return self._phase_for_stage("expert_global", iteration, 0, expert_global_end)
        if iteration < expert_local_end:
            return self._phase_for_stage("expert_local", iteration, expert_global_end, expert_local_end)
        if iteration < expert_full_end:
            return self._phase_for_stage("expert_full", iteration, expert_local_end, expert_full_end)
        if iteration < router_only_end:
            return self._phase_for_stage("router_only", iteration, expert_full_end, router_only_end)
        return self._phase_for_stage("joint_finetune", iteration, router_only_end, total_iterations)

    def _resolve_stage(self, name: str, total_iterations: int, absolute_default: int) -> int:
        value = int(getattr(self.args, name, 0) or 0)
        if value > 0:
            return min(value, total_iterations)
        return min(max(1, int(absolute_default)), total_iterations)

    @staticmethod
    def _stage_progress(iteration: int, stage_start: int, stage_end: int) -> float:
        return min(max((int(iteration) - int(stage_start)) / max(int(stage_end) - int(stage_start), 1), 0.0), 1.0)

    def _router_temperature(self, progress: float) -> float:
        temperature_init = float(getattr(self.args, "temperature_geo_init", 2.0))
        temperature_final = float(getattr(self.args, "temperature_geo_final", 0.7))
        return temperature_init * (1.0 - progress) + temperature_final * progress

    def _phase_for_stage(
        self,
        stage: str,
        iteration: int = 0,
        stage_start: int = 0,
        stage_end: int = 1,
    ) -> TrackingPhase:
        stage = normalize_endomoeg_stage(stage)
        progress = self._stage_progress(iteration, stage_start, stage_end)
        if stage == "expert_global":
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
                trainable_group_prefixes=(
                    "tracking_shared_base",
                    "tracking_expert_global",
                ),
                group_schedule_progress={
                    "tracking_shared_base": progress,
                    "tracking_expert_global": progress,
                },
            )
        if stage == "expert_local":
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
                trainable_group_prefixes=("tracking_expert_local",),
                group_schedule_progress={"tracking_expert_local": progress},
            )
        if stage == "expert_full":
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
                trainable_group_prefixes=("tracking_expert_full",),
                group_schedule_progress={"tracking_expert_full": progress},
            )
        if stage == "router_only":
            sparse_start = min(
                max(float(getattr(self.args, "endomoeg_router_sparse_start", 0.25)), 0.0),
                1.0,
            )
            balance_final_scale = min(
                max(float(getattr(self.args, "endomoeg_router_balance_final_scale", 0.10)), 0.0),
                1.0,
            )
            balance_scale = 1.0 * (1.0 - progress) + balance_final_scale * progress
            confidence_scale = min(
                max((progress - sparse_start) / max(1.0 - sparse_start, 1e-6), 0.0),
                1.0,
            )
            return TrackingPhase(
                name="moe_router_only",
                active_geo=3,
                active_vis=2,
                enable_visibility=True,
                temperature_geo=self._router_temperature(progress),
                temperature_vis=1.0,
                use_sparse_geo=progress >= sparse_start,
                use_sparse_vis=False,
                topk_geo=2,
                topk_vis=1,
                trainable_group_prefixes=("tracking_time_encoder", "tracking_moe_router"),
                group_schedule_progress={
                    "tracking_time_encoder": progress,
                    "tracking_moe_router": progress,
                },
                route_balance_scale=balance_scale,
                route_confidence_scale=confidence_scale,
            )
        if stage == "joint_finetune":
            return TrackingPhase(
                name="moe_joint_finetune",
                active_geo=3,
                active_vis=2,
                enable_visibility=True,
                temperature_geo=float(getattr(self.args, "temperature_geo_final", 0.7)),
                temperature_vis=1.0,
                use_sparse_geo=True,
                use_sparse_vis=False,
                topk_geo=2,
                topk_vis=1,
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_moe_router",
                    "tracking_expert_global",
                    "tracking_expert_local",
                    "tracking_expert_full",
                ),
                group_lr_scales={
                    "tracking_time_encoder": 0.1,
                    "tracking_expert_global": 0.1,
                    "tracking_expert_local": 0.1,
                    "tracking_expert_full": 0.1,
                },
                group_schedule_progress={
                    "tracking_time_encoder": progress,
                    "tracking_moe_router": progress,
                    "tracking_expert_global": progress,
                    "tracking_expert_local": progress,
                    "tracking_expert_full": progress,
                },
                route_balance_scale=min(
                    max(float(getattr(self.args, "endomoeg_joint_balance_scale", 0.05)), 0.0),
                    1.0,
                ),
                route_confidence_scale=1.0,
            )
        raise ValueError(f"Unsupported EndoMoe stage: {stage}")


class VolumeAwareGaussianRouter(nn.Module):
    def __init__(self, time_feature_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        hidden_dim = int(hidden_dim or max(32, self.time_feature_dim))
        feature_dim = self.time_feature_dim + 3 + 3 + 1 + 3
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
        canonical_means3d: torch.Tensor,
        shared_base_d_mu: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_features: torch.Tensor,
        expert_d_mu: torch.Tensor,
        scene_scale: torch.Tensor,
        phase: TrackingPhase,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.as_tensor(
            scene_scale,
            device=canonical_means3d.device,
            dtype=canonical_means3d.dtype,
        ).reshape(()).abs().clamp_min(1e-6)
        normalized_base_motion = shared_base_d_mu / scale
        normalized_expert_motion = expert_d_mu.norm(dim=-1) / scale
        opacity_probability = torch.sigmoid(opacity_logits)
        features = torch.cat(
            (
                time_features,
                self._normalize_xyz(canonical_means3d),
                normalized_base_motion,
                opacity_probability,
                normalized_expert_motion,
            ),
            dim=-1,
        )
        logits = self.router(features)
        dense_weights = torch.softmax(logits / max(float(phase.temperature_geo), 1e-6), dim=-1)
        weights = dense_weights

        forced = (phase.force_geo_expert or "").lower()
        forced_index = {"global": 0, "local": 1, "full": 2}.get(forced)
        if forced_index is not None:
            weights = torch.zeros_like(weights)
            weights[:, forced_index] = 1.0
        elif phase.use_sparse_geo:
            topk = min(max(int(phase.topk_geo), 1), weights.shape[-1])
            topk_values, topk_indices = torch.topk(weights, k=topk, dim=-1)
            weights = torch.zeros_like(weights).scatter(-1, topk_indices, topk_values)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_max_prob = weights.max(dim=-1).values
        top2 = torch.topk(weights, k=2, dim=-1).values
        route_margin = top2[:, 0] - top2[:, 1]
        return weights, dense_weights, logits, entropy, route_max_prob, route_margin


class PixelSpaceRouter(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.score_network = nn.Sequential(
            nn.Conv2d(10, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        conv_layers = [module for module in self.score_network if isinstance(module, nn.Conv2d)]
        for layer in conv_layers[:-1]:
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(conv_layers[-1].weight)
        nn.init.zeros_(conv_layers[-1].bias)

    def forward(
        self,
        expert_rgb: torch.Tensor,
        expert_depth: torch.Tensor,
        gaussian_prior: torch.Tensor,
        projected_motion: torch.Tensor,
        coverage: torch.Tensor,
        fallback_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if expert_rgb.ndim != 4:
            raise ValueError("expert_rgb must have shape [experts, 3, height, width]")
        if expert_depth.ndim != 3:
            raise ValueError("expert_depth must have shape [experts, height, width]")

        prior = gaussian_prior.clamp_min(0.0)
        eligible = coverage & (prior > 1e-8)
        prior = torch.where(eligible, prior, torch.zeros_like(prior))
        prior_sum = prior.sum(dim=0, keepdim=True)
        normalized_prior = prior / prior_sum.clamp_min(1e-8)
        fallback = fallback_prior.to(device=prior.device, dtype=prior.dtype)
        normalized_prior = torch.where(
            (prior_sum > 1e-8).expand_as(normalized_prior),
            normalized_prior,
            fallback.expand_as(normalized_prior),
        )

        rgb_context = (expert_rgb * normalized_prior.unsqueeze(1)).sum(dim=0, keepdim=True)
        rgb_delta = expert_rgb - rgb_context
        depth_context = (expert_depth * normalized_prior).sum(dim=0, keepdim=True)
        depth_scale = depth_context.abs().clamp_min(1e-3)
        relative_depth = ((expert_depth - depth_context) / depth_scale).clamp(-10.0, 10.0)
        motion_scale = projected_motion.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
        normalized_motion = (projected_motion / motion_scale).clamp(0.0, 1.0)

        features = torch.cat(
            (
                expert_rgb,
                rgb_delta,
                relative_depth.unsqueeze(1),
                normalized_prior.unsqueeze(1),
                normalized_motion.unsqueeze(1),
                coverage.to(dtype=expert_rgb.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        residual_logits = self.score_network(features).squeeze(1)
        logits = residual_logits + normalized_prior.clamp_min(1e-8).log()
        logits = torch.where(
            eligible,
            logits,
            torch.full_like(logits, -1e9),
        )
        weights = torch.softmax(logits, dim=0)
        weights = torch.where(
            eligible.any(dim=0, keepdim=True).expand_as(weights),
            weights,
            fallback.expand_as(weights),
        )
        return weights, residual_logits


class CAMSGSMoETracking(nn.Module):
    GEO_EXPERT_NAMES = ("global", "local", "full")
    VIS_EXPERT_NAMES = ("stable", "transient")

    def __init__(self, time_feature_dim: int, **kwargs) -> None:
        super().__init__()
        hidden_dim = kwargs.get("moe_expert_hidden_dim")
        self.expert_global = GlobalSmoothExpert(
            time_feature_dim,
            max_disp_ratio=kwargs.get("max_disp_global_ratio", 0.01),
            hidden_dim=hidden_dim,
        )
        self.expert_local = TissueLocalExpert(
            time_feature_dim,
            max_disp_ratio=kwargs.get("max_disp_local_ratio", 0.03),
            max_rot_delta=kwargs.get("max_rot_delta", 0.05),
            max_scale_delta=kwargs.get("max_scale_delta", 0.05),
            hidden_dim=hidden_dim,
            enable_rotation=kwargs.get("enable_rotation", True),
            enable_scale=kwargs.get("enable_scale", True),
        )
        self.expert_full = ToolContactExpert(
            time_feature_dim,
            max_disp_ratio=kwargs.get("max_disp_local_ratio", 0.03),
            max_rot_delta=kwargs.get("max_rot_delta", 0.05),
            max_scale_delta=kwargs.get("max_scale_delta", 0.05),
            max_opacity_delta=kwargs.get("max_opacity_delta", 4.0),
            hidden_dim=hidden_dim,
            enable_rotation=kwargs.get("enable_rotation", True),
            enable_scale=kwargs.get("enable_scale", True),
            enable_opacity=kwargs.get("enable_opacity", True),
        )
        self.router = VolumeAwareGaussianRouter(time_feature_dim, kwargs.get("moe_router_hidden_dim"))
        self.pixel_router = PixelSpaceRouter(kwargs.get("moe_pixel_router_hidden_dim", 32))

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups: Dict[str, Iterable[nn.Parameter]] = {
            "tracking_moe_router_gaussian": self.router.parameters(),
            "tracking_moe_router_pixel": self.pixel_router.parameters(),
        }
        for expert_name, expert in self._experts().items():
            for group_name, params in expert.named_parameter_groups().items():
                groups[f"tracking_expert_{expert_name}_{group_name}"] = params
        return groups

    def _experts(self) -> Dict[str, nn.Module]:
        return {
            "global": self.expert_global,
            "local": self.expert_local,
            "full": self.expert_full,
        }

    def component_state_dict(self, component: str):
        component = str(component).lower()
        if component == "router":
            return {
                "gaussian_router": self.router.state_dict(),
                "pixel_router": self.pixel_router.state_dict(),
            }
        experts = self._experts()
        if component not in experts:
            raise ValueError(f"Unsupported EndoMoe component: {component}")
        return experts[component].state_dict()

    def load_component_state_dict(self, component: str, state_dict) -> None:
        component = str(component).lower()
        if component == "router":
            self.router.load_state_dict(state_dict["gaussian_router"])
            self.pixel_router.load_state_dict(state_dict["pixel_router"])
            return
        experts = self._experts()
        if component not in experts:
            raise ValueError(f"Unsupported EndoMoe component: {component}")
        experts[component].load_state_dict(state_dict)

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.router.set_aabb(xyz_max, xyz_min)
        for expert in self._experts().values():
            expert.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        for expert in self._experts().values():
            yield from expert.iter_regularized_grids()

    def reset_parameters(self) -> None:
        self.router.reset_parameters()
        self.pixel_router.reset_parameters()
        for expert in self._experts().values():
            expert.reset_parameters()

    def route_pixels(
        self,
        expert_rgb: torch.Tensor,
        expert_depth: torch.Tensor,
        gaussian_prior: torch.Tensor,
        projected_motion: torch.Tensor,
        coverage: torch.Tensor,
        fallback_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pixel_router(
            expert_rgb=expert_rgb,
            expert_depth=expert_depth,
            gaussian_prior=gaussian_prior,
            projected_motion=projected_motion,
            coverage=coverage,
            fallback_prior=fallback_prior,
        )

    def _run_experts(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ):
        return [
            expert(
                means3d=means3d,
                scales=scales,
                rotations=rotations,
                opacity_logits=opacity_logits,
                time_values=time_values,
                scene_scale=scene_scale,
                camera=camera,
            )
            for expert in self._experts().values()
        ]

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
        canonical_means3d: Optional[torch.Tensor] = None,
    ):
        if canonical_means3d is None:
            canonical_means3d = means3d
        expert_outputs = self._run_experts(
            means3d=means3d,
            scales=scales,
            rotations=rotations,
            opacity_logits=opacity_logits,
            time_values=time_values,
            scene_scale=scene_scale,
            camera=camera,
        )
        expert_means = torch.stack([output["means3d"] for output in expert_outputs], dim=1)
        expert_scales = torch.stack([output["scales"] for output in expert_outputs], dim=1)
        expert_rotations = torch.stack([output["rotations"] for output in expert_outputs], dim=1)
        expert_opacity = torch.stack([output["opacity_logits"] for output in expert_outputs], dim=1)
        expert_d_mu = expert_means - means3d.unsqueeze(1)

        shared_base_d_mu = means3d - canonical_means3d
        pi_geo, dense_pi_geo, router_logits, entropy_geo, route_max_prob_geo, route_margin_geo = self.router(
            canonical_means3d=canonical_means3d,
            shared_base_d_mu=shared_base_d_mu,
            opacity_logits=opacity_logits,
            time_features=time_features,
            expert_d_mu=expert_d_mu,
            scene_scale=scene_scale,
            phase=phase,
        )
        weights = pi_geo.unsqueeze(-1)
        means_out = (expert_means * weights).sum(dim=1)
        scales_out = (expert_scales * weights).sum(dim=1)
        rotations_out = F.normalize((expert_rotations * weights).sum(dim=1), dim=-1)
        opacity_out = (expert_opacity * weights).sum(dim=1)

        residual_d_mu = means_out - means3d
        d_mu = means_out - canonical_means3d
        d_scale = scales_out - scales
        d_rot = rotations_out[:, 1:] - rotations[:, 1:]
        d_opacity = opacity_out - opacity_logits

        vis_rgb_delta = torch.stack([output["appearance_rgb_delta"] for output in expert_outputs], dim=1)
        vis_alpha = torch.stack([output["visibility_alpha"] for output in expert_outputs], dim=1)
        transient_probability = torch.stack(
            [output["transient_probability"] for output in expert_outputs],
            dim=1,
        )
        lifecycle_alpha = torch.stack([output["lifecycle_alpha"] for output in expert_outputs], dim=1)
        full_output = expert_outputs[-1]

        aux = {
            "means3d_canonical": canonical_means3d,
            "shared_base_means3d": means3d,
            "shared_base_d_mu": shared_base_d_mu,
            "scene_scale": torch.as_tensor(
                scene_scale,
                device=means3d.device,
                dtype=means3d.dtype,
            ).reshape(()),
            "d_mu": d_mu,
            "d_mu_residual": residual_d_mu,
            "d_rot": d_rot,
            "d_scale": d_scale,
            "d_opacity_logit": d_opacity,
            "global_motion": expert_d_mu[:, 0] * pi_geo[:, 0:1],
            "local_motion": expert_d_mu[:, 1] * pi_geo[:, 1:2],
            "cut_graph_motion": expert_d_mu[:, 2] * pi_geo[:, 2:3],
            "full_motion": expert_d_mu[:, 2] * pi_geo[:, 2:3],
            "global_motion_norm": (expert_d_mu[:, 0] * pi_geo[:, 0:1]).norm(dim=-1, keepdim=True),
            "local_motion_norm": (expert_d_mu[:, 1] * pi_geo[:, 1:2]).norm(dim=-1, keepdim=True),
            "cut_graph_motion_norm": (expert_d_mu[:, 2] * pi_geo[:, 2:3]).norm(dim=-1, keepdim=True),
            "full_motion_norm": (expert_d_mu[:, 2] * pi_geo[:, 2:3]).norm(dim=-1, keepdim=True),
            "geo_expert_d_mu": expert_d_mu,
            "geo_expert_means3d": expert_means,
            "geo_expert_scales": expert_scales,
            "geo_expert_rotations": expert_rotations,
            "geo_expert_opacity_logits": expert_opacity,
            "pi_geo": pi_geo,
            "gaussian_pi_geo_prior": pi_geo,
            "gaussian_pi_geo_dense": dense_pi_geo,
            "pi_vis": full_output["pi_vis"],
            "gaussian_pi_vis_prior": full_output["pi_vis"],
            "entropy_geo": entropy_geo,
            "entropy_vis": full_output["entropy_vis"],
            "route_max_prob_geo": route_max_prob_geo,
            "route_margin_geo": route_margin_geo,
            "route_top1_geo_mean": route_max_prob_geo.mean(),
            "route_max_prob_vis": full_output["route_max_prob_vis"],
            "route_margin_vis": full_output["route_margin_vis"],
            "route_top1_vis_mean": full_output["route_top1_vis_mean"],
            "visibility_logits": full_output["visibility_logits"],
            "visibility_alpha": (vis_alpha * weights).sum(dim=1),
            "transient_probability": full_output["transient_probability"],
            "appearance_offsets": full_output["appearance_offsets"],
            "appearance_rgb_delta": (vis_rgb_delta * weights).sum(dim=1),
            "vis_expert_rgb_delta": vis_rgb_delta,
            "vis_expert_visibility_alpha": vis_alpha,
            "vis_expert_transient_probability": transient_probability,
            "lifecycle_logits": full_output["lifecycle_logits"],
            "lifecycle_probs": full_output["lifecycle_probs"],
            "lifecycle_alpha": (lifecycle_alpha * weights).sum(dim=1),
            "lifecycle_expert_alpha": lifecycle_alpha,
            "expert_opacity_includes_visibility": True,
            "tracking_phase_name": phase.name,
            "moe_router_logits": router_logits,
            "router_sparse_active": means3d.new_tensor(float(phase.use_sparse_geo)),
            "router_topk_geo": means3d.new_tensor(float(phase.topk_geo)),
            "route_balance_scale": means3d.new_tensor(float(phase.route_balance_scale)),
            "route_confidence_scale": means3d.new_tensor(float(phase.route_confidence_scale)),
        }
        return means_out, scales_out, rotations_out, opacity_out, aux
