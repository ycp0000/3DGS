from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from .cams_gs_lifecycle import GaussianLifecycleHead
from .cams_gs_visibility import VisibilityAppearanceHead
from .cut_graph_gating import CutGraphGating
from .heterogeneous_moe_tracking import TrackingPhase
from .motion_decomposition import MotionDecomposition


class CAMSGSScheduler:
    def __init__(self, args) -> None:
        self.args = args

    def build(self, iteration: int, total_iterations: int) -> TrackingPhase:
        total_iterations = max(int(total_iterations), 1)

        def _fractional_default(fraction: float) -> int:
            return max(1, int(round(total_iterations * fraction)))

        def _resolve_stage(name: str, fraction: float) -> int:
            raw_value = getattr(self.args, name, None)
            if raw_value is None:
                return _fractional_default(fraction)
            value = int(raw_value)
            if value <= 0:
                return _fractional_default(fraction)
            return value

        stage_global_only_end = _resolve_stage("stage_global_only_end", 0.15)
        stage_graph_bootstrap_end = _resolve_stage("stage_graph_bootstrap_end", 0.30)
        stage_local_motion_end = _resolve_stage("stage_local_motion_end", 0.45)
        stage_visibility_enable_iter = _resolve_stage("stage_visibility_enable_iter", 0.70)
        stage_lifecycle_enable_iter = _resolve_stage("stage_lifecycle_enable_iter", 0.85)

        stage_global_only_end = max(1, stage_global_only_end)
        stage_graph_bootstrap_end = max(stage_global_only_end + 1, stage_graph_bootstrap_end)
        stage_local_motion_end = max(stage_graph_bootstrap_end + 1, stage_local_motion_end)
        stage_visibility_enable_iter = max(stage_local_motion_end + 1, stage_visibility_enable_iter)
        stage_lifecycle_enable_iter = max(stage_visibility_enable_iter + 1, stage_lifecycle_enable_iter)

        if iteration < stage_global_only_end:
            return TrackingPhase(
                name="global_only",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_base_deformation",
                    "tracking_base_grid",
                    "tracking_motion_global",
                ),
            )

        if iteration < stage_graph_bootstrap_end:
            return TrackingPhase(
                name="graph_bootstrap",
                active_geo=1,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_motion_global",
                    "tracking_cut_graph",
                ),
            )

        if iteration < stage_local_motion_end:
            return TrackingPhase(
                name="local_motion_only",
                active_geo=3,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_motion_global",
                    "tracking_motion_local",
                    "tracking_cut_graph",
                ),
                group_lr_scales={"tracking_motion_global": 0.1},
            )

        if iteration < stage_visibility_enable_iter:
            return TrackingPhase(
                name="motion_warmup",
                active_geo=3,
                active_vis=1,
                enable_visibility=False,
                temperature_geo=1.0,
                temperature_vis=1.0,
                use_sparse_geo=False,
                use_sparse_vis=False,
                topk_geo=1,
                topk_vis=1,
                trainable_group_prefixes=(
                    "tracking_time_encoder",
                    "tracking_motion_global",
                    "tracking_motion_local",
                    "tracking_cut_graph",
                ),
                group_lr_scales={"tracking_motion_global": 0.25},
            )

        if iteration < stage_lifecycle_enable_iter:
            return TrackingPhase(
                name="visibility_refine",
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
                    "tracking_motion_global",
                    "tracking_motion_local",
                    "tracking_cut_graph",
                    "tracking_visibility",
                    "tracking_appearance",
                ),
                group_lr_scales={"tracking_motion_global": 0.25},
            )

        return TrackingPhase(
            name="joint_finetune",
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
                "tracking_motion_global",
                "tracking_motion_local",
                "tracking_cut_graph",
                "tracking_visibility",
                "tracking_appearance",
                "tracking_lifecycle",
            ),
            group_lr_scales={"tracking_motion_global": 0.5},
        )


class CAMSGSTracking(nn.Module):
    GEO_EXPERT_NAMES = ("global", "local", "cut_graph")
    VIS_EXPERT_NAMES = ("stable", "transient")

    def __init__(self, time_feature_dim: int, **kwargs) -> None:
        super().__init__()
        self.time_feature_dim = int(time_feature_dim)
        self.cut_graph = CutGraphGating(self.time_feature_dim)
        self.motion = MotionDecomposition(
            self.time_feature_dim,
            max_disp_global_ratio=float(kwargs.get("max_disp_global_ratio", 0.01)),
            max_disp_local_ratio=float(kwargs.get("max_disp_local_ratio", 0.03)),
            max_rot_delta=float(kwargs.get("max_rot_local", kwargs.get("max_rot_smooth", 0.05))),
            max_scale_delta=float(kwargs.get("max_scale_local", kwargs.get("max_scale_smooth", 0.05))),
            max_opacity_delta=float(kwargs.get("max_opacity_delta", 4.0)),
            enable_scale=bool(kwargs.get("enable_scale", True)),
            enable_rotation=bool(kwargs.get("enable_rotation", True)),
            enable_opacity=bool(kwargs.get("enable_opacity", True)),
        )
        self.visibility = VisibilityAppearanceHead(self.time_feature_dim)
        self.lifecycle = GaussianLifecycleHead(self.time_feature_dim)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups = dict(self.motion.named_parameter_groups())
        groups.update(self.cut_graph.named_parameter_groups())
        groups.update(self.visibility.named_parameter_groups())
        groups.update(self.lifecycle.named_parameter_groups())
        return groups

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.cut_graph.set_aabb(xyz_max, xyz_min)
        self.motion.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        yield from self.cut_graph.iter_regularized_grids()
        yield from self.motion.iter_regularized_grids()
        yield from self.visibility.iter_regularized_grids()
        yield from self.lifecycle.iter_regularized_grids()

    def reset_parameters(self) -> None:
        self.cut_graph.reset_parameters()
        self.motion.reset_parameters()
        self.visibility.reset_parameters()
        self.lifecycle.reset_parameters()

    def _build_geo_probabilities(self, gating_state: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scaffold_weights = gating_state["scaffold_weights"]
        cut_gate_values = gating_state["cut_gate_values"][:, :1]
        global_mix = scaffold_weights[:, :1]
        local_mix = scaffold_weights[:, 1:2] * cut_gate_values
        cut_graph_mix = scaffold_weights[:, 2:3] * (1.0 - cut_gate_values)
        pi_geo = torch.cat((global_mix, local_mix, cut_graph_mix), dim=-1)
        pi_geo = pi_geo / pi_geo.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gating_state["local_mix"] = pi_geo[:, 1:2]
        gating_state["global_mix"] = pi_geo[:, 0:1]
        gating_state["cut_graph_mix"] = pi_geo[:, 2:3]
        entropy_geo = -(pi_geo.clamp_min(1e-8) * pi_geo.clamp_min(1e-8).log()).sum(dim=-1).mean()
        route_max_prob_geo = pi_geo.max(dim=-1).values
        top2_geo = torch.topk(pi_geo, k=min(2, pi_geo.shape[-1]), dim=-1).values
        route_margin_geo = top2_geo[:, 0] - top2_geo[:, 1]
        return pi_geo, entropy_geo, route_max_prob_geo, route_margin_geo

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
        gating_state = self.cut_graph(
            means3d=means3d,
            time_values=time_values,
            time_features=time_features,
            phase=phase,
        )
        pi_geo, entropy_geo, route_max_prob_geo, route_margin_geo = self._build_geo_probabilities(gating_state)
        motion_state = self.motion(
            means3d=means3d,
            scales=scales,
            rotations=rotations,
            opacity_logits=opacity_logits,
            time_features=time_features,
            scene_scale=scene_scale,
            gating_state=gating_state,
            phase=phase,
        )
        visibility_state = self.visibility(
            time_features=time_features,
            gating_state=gating_state,
            phase=phase,
        )
        lifecycle_state = self.lifecycle(
            time_features=time_features,
            gating_state=gating_state,
            phase=phase,
        )
        aux = {
            "d_mu": motion_state["d_mu"],
            "d_rot": motion_state["d_rot"],
            "d_scale": motion_state["d_scale"],
            "d_opacity_logit": motion_state["d_opacity_logit"],
            "global_motion": motion_state["global_motion"],
            "local_motion": motion_state["local_motion"],
            "cut_graph_motion": motion_state["cut_graph_motion"],
            "geo_expert_d_mu": motion_state["geo_expert_d_mu"],
            "geo_expert_means3d": motion_state["geo_expert_means3d"],
            "geo_expert_scales": motion_state["geo_expert_scales"],
            "geo_expert_rotations": motion_state["geo_expert_rotations"],
            "geo_expert_opacity_logits": motion_state["geo_expert_opacity_logits"],
            "scaffold_logits": gating_state["scaffold_logits"],
            "scaffold_weights": gating_state["scaffold_weights"],
            "cut_gate_logits": gating_state["cut_gate_logits"],
            "cut_gate_values": gating_state["cut_gate_values"],
            "pi_geo": pi_geo,
            "gaussian_pi_geo_prior": pi_geo,
            "pi_vis": visibility_state["pi_vis"],
            "gaussian_pi_vis_prior": visibility_state["pi_vis"],
            "entropy_geo": entropy_geo,
            "entropy_vis": visibility_state["entropy_vis"],
            "route_max_prob_geo": route_max_prob_geo,
            "route_margin_geo": route_margin_geo,
            "route_top1_geo_mean": route_max_prob_geo.mean(),
            "route_max_prob_vis": visibility_state["route_max_prob_vis"],
            "route_margin_vis": visibility_state["route_margin_vis"],
            "route_top1_vis_mean": visibility_state["route_top1_vis_mean"],
            "visibility_logits": visibility_state["visibility_logits"],
            "visibility_alpha": visibility_state["visibility_alpha"],
            "appearance_offsets": visibility_state["appearance_offsets"],
            "appearance_rgb_delta": visibility_state["appearance_rgb_delta"],
            "vis_expert_rgb_delta": visibility_state["vis_expert_rgb_delta"],
            "vis_expert_visibility_alpha": visibility_state["vis_expert_visibility_alpha"],
            "lifecycle_logits": lifecycle_state["lifecycle_logits"],
            "lifecycle_probs": lifecycle_state["lifecycle_probs"],
            "lifecycle_alpha": lifecycle_state["lifecycle_alpha"],
            "lifecycle_expert_alpha": lifecycle_state["lifecycle_expert_alpha"],
            "tracking_phase_name": phase.name,
        }
        if self.motion.enable_opacity:
            opacity_scale = visibility_state["visibility_alpha"] * lifecycle_state["lifecycle_alpha"]
            opacity_logits_out = motion_state["opacity_logits"] + torch.logit(opacity_scale.clamp(1e-4, 1.0 - 1e-4))
        else:
            opacity_logits_out = motion_state["opacity_logits"]
        return (
            motion_state["means3d"],
            motion_state["scales"],
            motion_state["rotations"],
            opacity_logits_out,
            aux,
        )
