from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _build_router_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _init_mlp_small_output(
    mlp: nn.Sequential,
    final_std: float = 1e-4,
    final_bias: float = 0.0,
) -> None:
    linear_layers = [module for module in mlp if isinstance(module, nn.Linear)]
    if not linear_layers:
        return

    for layer in linear_layers[:-1]:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    final_layer = linear_layers[-1]
    nn.init.normal_(final_layer.weight, mean=0.0, std=final_std)
    nn.init.constant_(final_layer.bias, final_bias)


def _init_router_mlp(mlp: nn.Sequential, final_bias: torch.Tensor) -> None:
    linear_layers = [module for module in mlp if isinstance(module, nn.Linear)]
    if not linear_layers:
        return

    for layer in linear_layers[:-1]:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    final_layer = linear_layers[-1]
    nn.init.normal_(final_layer.weight, mean=0.0, std=1e-4)
    with torch.no_grad():
        final_layer.bias.copy_(final_bias.to(final_layer.bias.device, final_layer.bias.dtype))


def _scene_scale_tensor(
    scene_scale: Union[torch.Tensor, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    scale = torch.as_tensor(scene_scale, device=reference.device, dtype=reference.dtype).reshape(())
    return scale.abs().clamp_min(1e-6)


@dataclass(frozen=True)
class TrackingPhase:
    name: str
    active_geo: int
    active_vis: int
    enable_visibility: bool
    temperature_geo: float
    temperature_vis: float
    use_sparse_geo: bool
    use_sparse_vis: bool
    topk_geo: int
    topk_vis: int
    force_geo_expert: Optional[str] = None
    force_vis_expert: Optional[str] = None
    trainable_group_prefixes: Tuple[str, ...] = ()
    frozen_group_prefixes: Tuple[str, ...] = ()
    group_lr_scales: Dict[str, float] = field(default_factory=dict)
    group_schedule_progress: Dict[str, float] = field(default_factory=dict)
    route_balance_scale: float = 1.0
    route_confidence_scale: float = 1.0

    def is_group_trainable(self, group_name: str) -> bool:
        if any(
            group_name.startswith(prefix)
            for prefix in self.frozen_group_prefixes
        ):
            return False
        if group_name in {
            "always",
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
            "tracking_base_deformation",
            "tracking_base_grid",
        }:
            return True
        return any(group_name.startswith(prefix) for prefix in self.trainable_group_prefixes)

    def lr_scale_for_group(self, group_name: str) -> float:
        if group_name in {
            "always",
            "xyz",
            "f_dc",
            "f_rest",
            "opacity",
            "scaling",
            "rotation",
            "tracking_base_deformation",
            "tracking_base_grid",
        }:
            return 1.0
        for prefix, scale in self.group_lr_scales.items():
            if group_name.startswith(prefix):
                return float(scale)
        return 1.0 if self.is_group_trainable(group_name) else 0.0

    def schedule_progress_for_group(self, group_name: str) -> Optional[float]:
        for prefix, progress in self.group_schedule_progress.items():
            if group_name.startswith(prefix):
                return min(max(float(progress), 0.0), 1.0)
        return None


class HeterogeneousMoEScheduler:
    def __init__(self, args) -> None:
        self.args = args

    def build(self, iteration: int, total_iterations: int) -> TrackingPhase:
        total_iterations = max(int(total_iterations), 1)
        progress = min(max(iteration / total_iterations, 0.0), 1.0)
        temperature_geo = (
            getattr(self.args, "temperature_geo_init", 2.0) * (1.0 - progress)
            + getattr(self.args, "temperature_geo_final", 0.7) * progress
        )
        temperature_vis = (
            getattr(self.args, "temperature_vis_init", 2.0) * (1.0 - progress)
            + getattr(self.args, "temperature_vis_final", 1.0) * progress
        )

        def _fractional_default(fraction: float) -> int:
            return max(1, int(round(total_iterations * fraction)))

        stage_hexplane_only_end = int(
            getattr(
                self.args,
                "stage_hexplane_only_end",
                getattr(self.args, "enable_shared_only_iter", _fractional_default(0.15)),
            )
        )
        stage_smooth_only_end = int(
            getattr(
                self.args,
                "stage_smooth_only_end",
                getattr(self.args, "enable_smooth_geo_iter", _fractional_default(0.30)),
            )
        )
        stage_local_only_end = int(
            getattr(
                self.args,
                "stage_local_only_end",
                getattr(self.args, "enable_local_geo_iter", _fractional_default(0.45)),
            )
        )
        stage_router_only_end = int(
            getattr(
                self.args,
                "stage_router_only_end",
                getattr(
                    self.args,
                    "enable_route_stability_iter",
                    getattr(self.args, "enable_sparse_routing_iter", _fractional_default(0.60)),
                ),
            )
        )

        stage_hexplane_only_end = max(1, stage_hexplane_only_end)
        stage_smooth_only_end = max(stage_hexplane_only_end + 1, stage_smooth_only_end)
        stage_local_only_end = max(stage_smooth_only_end + 1, stage_local_only_end)
        stage_router_only_end = max(stage_local_only_end + 1, stage_router_only_end)

        enable_visibility_iter = int(getattr(self.args, "enable_visibility_iter", stage_local_only_end))
        enable_sparse_routing_iter = int(getattr(self.args, "enable_sparse_routing_iter", stage_router_only_end))

        if iteration < stage_hexplane_only_end:
            return TrackingPhase(
                name="hexplane_only",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=temperature_geo,
                temperature_vis=temperature_vis,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="hexplane",
                force_vis_expert="stable",
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_geo_hexplane_grid",
                    "tracking_geo_hexplane_mlp",
                ),
            )

        if iteration < stage_smooth_only_end:
            return TrackingPhase(
                name="smooth_only",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=temperature_geo,
                temperature_vis=temperature_vis,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="smooth",
                force_vis_expert="stable",
                trainable_group_prefixes=("tracking_time_encoder", "tracking_geo_smooth"),
            )

        if iteration < stage_local_only_end:
            return TrackingPhase(
                name="local_only",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=temperature_geo,
                temperature_vis=temperature_vis,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                force_geo_expert="local",
                force_vis_expert="stable",
                trainable_group_prefixes=("tracking_time_encoder", "tracking_geo_local"),
            )

        if iteration < stage_router_only_end:
            return TrackingPhase(
                name="router_only",
                active_geo=4,
                active_vis=2,
                enable_visibility=bool(getattr(self.args, "enable_visibility", True) and iteration >= enable_visibility_iter),
                temperature_geo=temperature_geo,
                temperature_vis=temperature_vis,
                use_sparse_geo=bool(getattr(self.args, "use_topk", False) and iteration >= enable_sparse_routing_iter),
                use_sparse_vis=bool(getattr(self.args, "use_topk", False) and iteration >= enable_sparse_routing_iter),
                topk_geo=int(getattr(self.args, "topk_geo", 2)),
                topk_vis=int(getattr(self.args, "topk_vis", 1)),
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_geo_router",
                    "tracking_vis_router",
                    "tracking_vis_transient",
                ),
            )

        return TrackingPhase(
            name="joint_finetune",
            active_geo=4,
            active_vis=2,
            enable_visibility=bool(getattr(self.args, "enable_visibility", True) and iteration >= enable_visibility_iter),
            temperature_geo=temperature_geo,
            temperature_vis=temperature_vis,
            use_sparse_geo=bool(getattr(self.args, "use_topk", False) and iteration >= enable_sparse_routing_iter),
            use_sparse_vis=bool(getattr(self.args, "use_topk", False) and iteration >= enable_sparse_routing_iter),
            topk_geo=int(getattr(self.args, "topk_geo", 2)),
            topk_vis=int(getattr(self.args, "topk_vis", 1)),
            trainable_group_prefixes=(
                "tracking_time_encoder",
                "tracking_geo_router",
                "tracking_vis_router",
                "tracking_geo_hexplane_grid",
                "tracking_geo_hexplane_mlp",
                "tracking_geo_local",
                "tracking_geo_smooth",
                "tracking_vis_transient",
            ),
            group_lr_scales={
                "tracking_time_encoder": 1.0,
                "tracking_geo_router": 1.0,
                "tracking_vis_router": 1.0,
                "tracking_geo_hexplane": 0.1,
                "tracking_geo_local": 0.1,
                "tracking_geo_smooth": 0.1,
                "tracking_vis_transient": 0.1,
            },
        )



class _BoundedTranslationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_disp_ratio: float) -> None:
        super().__init__()
        self.net = _build_mlp(in_dim, hidden_dim, 3)
        self.max_disp_ratio = float(max_disp_ratio)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_mlp_small_output(self.net)

    def forward(self, features: torch.Tensor, scene_scale: torch.Tensor) -> torch.Tensor:
        raw = self.net(features)
        return torch.tanh(raw) * (self.max_disp_ratio * scene_scale)


class _TransientOpacityHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_opacity_delta: float) -> None:
        super().__init__()
        self.net = _build_mlp(in_dim, hidden_dim, 1)
        self.max_opacity_delta = float(max_opacity_delta)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_mlp_small_output(self.net)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features)) * self.max_opacity_delta


class SplitTrackingHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        geo_hidden_dim: int,
        vis_hidden_dim: int,
        max_disp_smooth_ratio: float,
        max_rot_smooth: float,
        max_scale_smooth: float,
        max_opacity_delta: float,
    ) -> None:
        del max_rot_smooth, max_scale_smooth
        super().__init__()
        self.geo_head = _BoundedTranslationHead(feature_dim, geo_hidden_dim, max_disp_smooth_ratio)
        self.vis_head = _TransientOpacityHead(feature_dim, vis_hidden_dim, max_opacity_delta)

    def reset_parameters(self) -> None:
        self.geo_head.reset_parameters()
        self.vis_head.reset_parameters()

    def forward(
        self,
        features: torch.Tensor,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        scene_scale: torch.Tensor,
    ):
        d_mu = self.geo_head(features, _scene_scale_tensor(scene_scale, means3d))
        d_opacity = self.vis_head(features)
        opacity = torch.sigmoid(opacity_logits).clamp(1e-6, 1.0 - 1e-6)
        opacity_logits_t = torch.logit(torch.sigmoid(torch.logit(opacity) + d_opacity).clamp(1e-6, 1.0 - 1e-6))
        aux = {
            "d_mu": d_mu,
            "d_rot": torch.zeros_like(means3d),
            "d_scale": torch.zeros_like(scales),
            "d_opacity_logit": d_opacity,
            "pi_geo": torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
            "pi_vis": torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
            "entropy_geo": torch.zeros((), device=features.device, dtype=features.dtype),
            "entropy_vis": torch.zeros((), device=features.device, dtype=features.dtype),
        }
        return means3d + d_mu, scales, rotations, opacity_logits_t, aux


class _ZeroStaticExpert(nn.Module):
    def forward(self, means3d: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros_like(means3d)


class _LocalMLPGeometryExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_disp_ratio: float) -> None:
        super().__init__()
        self.head = _BoundedTranslationHead(in_dim, hidden_dim, max_disp_ratio)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    def forward(self, features: torch.Tensor, scene_scale: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.head(features, scene_scale)


class _HexPlaneGeometryExpert(nn.Module):
    def __init__(
        self,
        bounds: float,
        planeconfig,
        multires,
        time_feature_dim: int,
        hidden_dim: int,
        max_disp_ratio: float,
    ) -> None:
        super().__init__()
        from scene.hexplane import HexPlaneField

        self.field = HexPlaneField(bounds, planeconfig, multires)
        decoder_in_dim = self.field.feat_dim + 3 + time_feature_dim + 3 + 1
        self.head = _BoundedTranslationHead(decoder_in_dim, hidden_dim, max_disp_ratio)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()

    def set_aabb(self, xyz_max, xyz_min) -> None:
        self.field.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self) -> Iterable[nn.ParameterList]:
        yield from self.field.grids

    def forward(
        self,
        means3d: torch.Tensor,
        xyz_norm: torch.Tensor,
        time_values: torch.Tensor,
        time_features: torch.Tensor,
        scales: torch.Tensor,
        opacity_logits: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> torch.Tensor:
        field_features = self.field(means3d, time_values)
        features = torch.cat([field_features, xyz_norm, time_features, scales, opacity_logits], dim=-1)
        return self.head(features, scene_scale)


class HeterogeneousMoETracking(nn.Module):
    GEO_EXPERT_NAMES = ("static", "hexplane", "local", "smooth")
    VIS_EXPERT_NAMES = ("stable", "transient")

    def __init__(
        self,
        time_feature_dim: int,
        geo_hidden_dim: int,
        vis_hidden_dim: int,
        bounds: float,
        planeconfig,
        multires,
        max_disp_hexplane_ratio: float,
        max_disp_local_ratio: float,
        max_disp_smooth_ratio: float,
        max_opacity_delta: float,
        sat_threshold: float = 0.8,
        use_soft_routing: bool = True,
        use_topk: bool = False,
        topk_geo: int = 2,
        topk_vis: int = 1,
        router_noise_geo: float = 0.0,
        router_noise_vis: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.use_soft_routing = bool(use_soft_routing)
        self.use_topk = bool(use_topk)
        self.topk_geo = int(topk_geo)
        self.topk_vis = int(topk_vis)
        self.router_noise_geo = float(router_noise_geo)
        self.router_noise_vis = float(router_noise_vis)
        self.sat_threshold = float(sat_threshold)

        geo_router_in_dim = 3 + self.time_feature_dim + 3 + 1
        vis_router_in_dim = 3 + 3 + 1 + 1 + len(self.GEO_EXPERT_NAMES) + self.time_feature_dim
        expert_feature_dim = 3 + self.time_feature_dim + 3 + 1

        self.geometry_router = _build_router_mlp(geo_router_in_dim, geo_hidden_dim, len(self.GEO_EXPERT_NAMES))
        self.visibility_router = _build_router_mlp(vis_router_in_dim, vis_hidden_dim, len(self.VIS_EXPERT_NAMES))

        self.geo_experts = nn.ModuleDict(
            {
                "static": _ZeroStaticExpert(),
                "hexplane": _HexPlaneGeometryExpert(
                    bounds=bounds,
                    planeconfig=planeconfig,
                    multires=multires,
                    time_feature_dim=self.time_feature_dim,
                    hidden_dim=geo_hidden_dim,
                    max_disp_ratio=max_disp_hexplane_ratio,
                ),
                "local": _LocalMLPGeometryExpert(expert_feature_dim, geo_hidden_dim, max_disp_local_ratio),
                "smooth": _LocalMLPGeometryExpert(expert_feature_dim, max(32, geo_hidden_dim // 2), max_disp_smooth_ratio),
            }
        )
        self.visibility_experts = nn.ModuleDict(
            {
                "transient": _TransientOpacityHead(vis_router_in_dim, vis_hidden_dim, max_opacity_delta),
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_router_mlp(self.geometry_router, final_bias=torch.tensor([2.0, 0.0, 0.0, 0.0]))
        _init_router_mlp(self.visibility_router, final_bias=torch.tensor([2.0, -2.0]))
        for expert in self.geo_experts.values():
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters()
        for expert in self.visibility_experts.values():
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters()

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "tracking_geo_router": self.geometry_router.parameters(),
            "tracking_vis_router": self.visibility_router.parameters(),
            "tracking_geo_hexplane_grid": self.geo_experts["hexplane"].field.parameters(),
            "tracking_geo_hexplane_mlp": self.geo_experts["hexplane"].head.parameters(),
            "tracking_geo_local": self.geo_experts["local"].parameters(),
            "tracking_geo_smooth": self.geo_experts["smooth"].parameters(),
            "tracking_vis_transient": self.visibility_experts["transient"].parameters(),
        }

    def set_aabb(self, xyz_max, xyz_min) -> None:
        for expert in self.geo_experts.values():
            if hasattr(expert, "set_aabb"):
                expert.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self) -> Iterable[nn.ParameterList]:
        for expert in self.geo_experts.values():
            if hasattr(expert, "iter_regularized_grids"):
                yield from expert.iter_regularized_grids()

    def _route(
        self,
        logits: torch.Tensor,
        temperature: float,
        active_count: int,
        use_sparse: bool,
        topk: int,
        noise_scale: float,
    ) -> torch.Tensor:
        logits = logits.clamp(-15.0, 15.0)
        active_count = max(1, min(int(active_count), logits.shape[-1]))
        if self.training and noise_scale > 0.0:
            logits = logits + torch.randn_like(logits) * noise_scale
        if active_count < logits.shape[-1]:
            active_mask = torch.full_like(logits, -1e9)
            active_mask[:, :active_count] = 0.0
            logits = logits + active_mask
        if use_sparse and active_count > 1:
            keep = max(1, min(int(topk), active_count))
            if keep < active_count:
                topk_idx = torch.topk(logits[:, :active_count], k=keep, dim=-1).indices
                sparse_mask = torch.full_like(logits[:, :active_count], -1e9)
                sparse_mask.scatter_(1, topk_idx, 0.0)
                logits = torch.cat([logits[:, :active_count] + sparse_mask, logits[:, active_count:]], dim=-1)
        elif not self.use_soft_routing and active_count > 1:
            top1_idx = torch.argmax(logits[:, :active_count], dim=-1, keepdim=True)
            hard_mask = torch.full_like(logits[:, :active_count], -1e9)
            hard_mask.scatter_(1, top1_idx, 0.0)
            logits = torch.cat([logits[:, :active_count] + hard_mask, logits[:, active_count:]], dim=-1)
        return torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)

    def _one_hot_route(self, batch_size: int, expert_name: str, names: Tuple[str, ...], reference: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros(batch_size, len(names), device=reference.device, dtype=reference.dtype)
        weights[:, names.index(expert_name)] = 1.0
        return weights

    def _route_stats(self, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        topk = min(2, weights.shape[-1])
        values, indices = torch.topk(weights, k=topk, dim=-1)
        margin = values[:, 0] if topk == 1 else values[:, 0] - values[:, 1]
        return {
            "max_prob": values[:, 0].mean(),
            "margin": margin.mean(),
            "top1_index_mean": indices[:, 0].float().mean(),
        }

    def _expert_diversity(self, geo_stack: torch.Tensor) -> torch.Tensor:
        dynamic = geo_stack[:, 1:, :]
        if dynamic.shape[1] < 2:
            return torch.zeros((), device=geo_stack.device, dtype=geo_stack.dtype)
        similarities = []
        for i in range(dynamic.shape[1]):
            for j in range(i + 1, dynamic.shape[1]):
                a = dynamic[:, i, :]
                b = dynamic[:, j, :]
                denom = torch.norm(a, dim=-1).clamp_min(1e-6) * torch.norm(b, dim=-1).clamp_min(1e-6)
                similarities.append(((a * b).sum(dim=-1) / denom).abs().mean())
        return torch.stack(similarities).mean() if similarities else torch.zeros((), device=geo_stack.device, dtype=geo_stack.dtype)

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
    ):
        scene_scale = _scene_scale_tensor(scene_scale, means3d)
        xyz_norm = means3d / scene_scale
        geo_features = torch.cat([xyz_norm, time_features, scales, opacity_logits], dim=-1)

        if phase.force_geo_expert is not None:
            pi_geo = self._one_hot_route(means3d.shape[0], phase.force_geo_expert, self.GEO_EXPERT_NAMES, means3d)
            logits_geo = None
        else:
            logits_geo = self.geometry_router(geo_features)
            pi_geo = self._route(
                logits_geo,
                phase.temperature_geo,
                phase.active_geo,
                use_sparse=phase.use_sparse_geo,
                topk=phase.topk_geo,
                noise_scale=self.router_noise_geo,
            )

        geo_outputs = []
        for name in self.GEO_EXPERT_NAMES:
            if phase.force_geo_expert is not None and name != phase.force_geo_expert:
                geo_outputs.append(torch.zeros_like(means3d))
                continue
            expert = self.geo_experts[name]
            if name == "hexplane":
                geo_outputs.append(
                    expert(
                        means3d=means3d,
                        xyz_norm=xyz_norm,
                        time_values=time_values,
                        time_features=time_features,
                        scales=scales,
                        opacity_logits=opacity_logits,
                        scene_scale=scene_scale,
                    )
                )
            elif name == "static":
                geo_outputs.append(expert(means3d))
            else:
                geo_outputs.append(expert(features=geo_features, scene_scale=scene_scale))
        geo_stack = torch.stack(geo_outputs, dim=1)
        d_mu = (pi_geo.unsqueeze(-1) * geo_stack).sum(dim=1)
        means3d_t = means3d + d_mu

        xyz_t_norm = means3d_t / scene_scale
        d_mu_detached = d_mu.detach()
        pi_geo_detached = pi_geo.detach()
        disp_norm = torch.norm(d_mu_detached, dim=-1, keepdim=True)
        vis_features = torch.cat([xyz_t_norm, d_mu_detached, disp_norm, opacity_logits, pi_geo_detached, time_features], dim=-1)

        if not phase.enable_visibility or phase.force_vis_expert == "stable":
            pi_vis = self._one_hot_route(means3d.shape[0], "stable", self.VIS_EXPERT_NAMES, means3d)
            logits_vis = None
            d_opacity = torch.zeros_like(opacity_logits)
        else:
            logits_vis = self.visibility_router(vis_features)
            pi_vis = self._route(
                logits_vis,
                phase.temperature_vis,
                phase.active_vis,
                use_sparse=phase.use_sparse_vis,
                topk=phase.topk_vis,
                noise_scale=self.router_noise_vis,
            )
            transient_delta = self.visibility_experts["transient"](vis_features)
            vis_stack = torch.stack([torch.zeros_like(transient_delta), transient_delta], dim=1)
            d_opacity = (pi_vis.unsqueeze(-1) * vis_stack).sum(dim=1)

        opacity = torch.sigmoid(opacity_logits).clamp(1e-6, 1.0 - 1e-6)
        opacity_logits_t = torch.logit(torch.sigmoid(torch.logit(opacity) + d_opacity).clamp(1e-6, 1.0 - 1e-6))

        entropy_geo = -(pi_geo * torch.log(pi_geo + 1e-8)).sum(dim=-1).mean()
        entropy_vis = -(pi_vis * torch.log(pi_vis + 1e-8)).sum(dim=-1).mean()
        geo_stats = self._route_stats(pi_geo)
        vis_stats = self._route_stats(pi_vis)
        expert_diversity_geo = self._expert_diversity(geo_stack)

        aux = {
            "pi_geo": pi_geo,
            "pi_vis": pi_vis,
            "d_mu": d_mu,
            "d_rot": torch.zeros(means3d.shape[0], 3, device=means3d.device, dtype=means3d.dtype),
            "d_scale": torch.zeros_like(scales),
            "d_opacity_logit": d_opacity,
            "entropy_geo": entropy_geo,
            "entropy_vis": entropy_vis,
            "route_max_prob_geo": geo_stats["max_prob"],
            "route_margin_geo": geo_stats["margin"],
            "route_top1_geo_mean": geo_stats["top1_index_mean"],
            "route_max_prob_vis": vis_stats["max_prob"],
            "route_margin_vis": vis_stats["margin"],
            "route_top1_vis_mean": vis_stats["top1_index_mean"],
            "expert_diversity_geo": expert_diversity_geo,
        }
        for expert_index, expert_name in enumerate(self.GEO_EXPERT_NAMES):
            disp_norm_per_expert = torch.norm(geo_stack[:, expert_index, :], dim=-1)
            aux[f"geo_weighted_disp_norm_{expert_name}"] = (pi_geo[:, expert_index] * disp_norm_per_expert).mean()
            aux[f"geo_usage_{expert_name}"] = pi_geo[:, expert_index].mean()

            if expert_name == "static":
                aux[f"geo_weighted_disp_ratio_{expert_name}"] = torch.zeros((), device=means3d.device, dtype=means3d.dtype)
                aux[f"geo_saturation_{expert_name}"] = torch.zeros((), device=means3d.device, dtype=means3d.dtype)
                continue

            expert = self.geo_experts[expert_name]
            max_ratio = float(expert.head.max_disp_ratio)
            max_norm = scene_scale * max_ratio * (3.0 ** 0.5)
            disp_ratio = disp_norm_per_expert / max_norm.clamp_min(1e-6)
            aux[f"geo_weighted_disp_ratio_{expert_name}"] = (pi_geo[:, expert_index] * disp_ratio).mean()
            aux[f"geo_saturation_{expert_name}"] = (
                pi_geo[:, expert_index] * torch.relu(disp_ratio - self.sat_threshold).pow(2)
            ).mean()
        for vis_index, vis_name in enumerate(self.VIS_EXPERT_NAMES):
            aux[f"vis_usage_{vis_name}"] = pi_vis[:, vis_index].mean()
        return means3d_t, scales, rotations, opacity_logits_t, aux


@torch.no_grad()
def shape_debug_check(device: torch.device = torch.device("cpu")) -> Dict[str, bool]:
    n = 32
    time_feature_dim = 8
    model = HeterogeneousMoETracking(
        time_feature_dim=time_feature_dim,
        geo_hidden_dim=16,
        vis_hidden_dim=16,
        bounds=1.6,
        planeconfig={
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 8,
            "resolution": [16, 16, 16, 8],
        },
        multires=[1],
        max_disp_hexplane_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_disp_smooth_ratio=0.005,
        max_opacity_delta=4.0,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    ).to(device)
    phase = TrackingPhase(
        name="joint_finetune",
        active_geo=4,
        active_vis=2,
        enable_visibility=True,
        temperature_geo=1.0,
        temperature_vis=1.0,
        use_sparse_geo=True,
        use_sparse_vis=True,
        topk_geo=2,
        topk_vis=1,
        trainable_group_prefixes=("tracking_",),
    )
    means3d_t, scales_t, rotations_t, opacity_t, aux = model(
        means3d=torch.randn(n, 3, device=device),
        scales=torch.randn(n, 3, device=device),
        rotations=torch.randn(n, 4, device=device),
        opacity_logits=torch.randn(n, 1, device=device),
        time_values=torch.rand(n, 1, device=device),
        time_features=torch.randn(n, time_feature_dim, device=device),
        scene_scale=torch.tensor(1.0, device=device),
        phase=phase,
    )
    return {
        "mu_shape": means3d_t.shape == (n, 3),
        "scale_shape": scales_t.shape == (n, 3),
        "rotation_shape": rotations_t.shape == (n, 4),
        "opacity_shape": opacity_t.shape == (n, 1),
        "pi_geo_shape": aux["pi_geo"].shape == (n, 4),
        "pi_vis_shape": aux["pi_vis"].shape == (n, 2),
        "pi_geo_sum": torch.allclose(aux["pi_geo"].sum(dim=-1), torch.ones(n, device=device), atol=1e-4),
        "pi_vis_sum": torch.allclose(aux["pi_vis"].sum(dim=-1), torch.ones(n, device=device), atol=1e-4),
        "finite_outputs": bool(
            torch.isfinite(means3d_t).all().item()
            and torch.isfinite(scales_t).all().item()
            and torch.isfinite(rotations_t).all().item()
            and torch.isfinite(opacity_t).all().item()
        ),
    }
