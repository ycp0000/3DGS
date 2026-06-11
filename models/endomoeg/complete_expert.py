from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from models.tracking.endomoeg_experts import (
    TissueLocalExpert,
    ToolContactExpert,
)
from models.tracking.heterogeneous_moe_tracking import TrackingPhase


COMPLETE_EXPERT_ROLES = ("global", "local", "contact")


class CompleteExpertScheduler:
    def __init__(self, role):
        normalized_role = str(role).strip().lower()
        if normalized_role not in COMPLETE_EXPERT_ROLES:
            raise ValueError("Unsupported complete expert role: {}".format(role))
        self.role = normalized_role

    def build(self, iteration, total_iterations):
        del iteration, total_iterations
        contact = self.role == "contact"
        return TrackingPhase(
            name="endomoeg_expert_{}".format(self.role),
            active_geo=1,
            active_vis=2 if contact else 1,
            enable_visibility=contact,
            temperature_geo=1.0,
            temperature_vis=1.0,
            use_sparse_geo=False,
            use_sparse_vis=False,
            topk_geo=1,
            topk_vis=1,
            force_geo_expert=self.role,
            force_vis_expert=None,
            trainable_group_prefixes=(
                "tracking_time_encoder",
                "tracking_expert_refinement",
            ),
        )


class CompleteEndoMoeExpert(nn.Module):
    def __init__(
        self,
        role,
        time_feature_dim,
        hidden_dim=64,
        max_disp_local_ratio=0.03,
        max_rot_delta=0.05,
        max_scale_delta=0.05,
        max_opacity_delta=4.0,
        enable_rotation=True,
        enable_scale=True,
        enable_opacity=True,
    ):
        super().__init__()
        normalized_role = str(role).strip().lower()
        if normalized_role not in COMPLETE_EXPERT_ROLES:
            raise ValueError("Unsupported complete expert role: {}".format(role))
        self.role = normalized_role
        self.refinement = None
        if self.role == "local":
            self.refinement = TissueLocalExpert(
                time_feature_dim=time_feature_dim,
                hidden_dim=hidden_dim,
                max_disp_ratio=max_disp_local_ratio,
                max_rot_delta=max_rot_delta,
                max_scale_delta=max_scale_delta,
                enable_rotation=enable_rotation,
                enable_scale=enable_scale,
            )
        elif self.role == "contact":
            self.refinement = ToolContactExpert(
                time_feature_dim=time_feature_dim,
                hidden_dim=hidden_dim,
                max_disp_ratio=max_disp_local_ratio,
                max_rot_delta=max_rot_delta,
                max_scale_delta=max_scale_delta,
                max_opacity_delta=max_opacity_delta,
                enable_rotation=enable_rotation,
                enable_scale=enable_scale,
                enable_opacity=enable_opacity,
            )

    def named_parameter_groups(self):
        if self.refinement is None:
            return {}
        return {
            "tracking_expert_refinement": self.refinement.parameters(),
        }

    def set_aabb(self, xyz_max, xyz_min):
        if self.refinement is not None:
            self.refinement.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        return iter(())

    def reset_parameters(self):
        if self.refinement is not None:
            self.refinement.reset_parameters()

    @staticmethod
    def _identity_refinement(
        means3d,
        scales,
        rotations,
        opacity_logits,
    ):
        count = means3d.shape[0]
        pi_vis = means3d.new_zeros((count, 2))
        pi_vis[:, 0] = 1.0
        return {
            "means3d": means3d,
            "scales": scales,
            "rotations": rotations,
            "opacity_logits": opacity_logits,
            "d_mu": torch.zeros_like(means3d),
            "d_scale": torch.zeros_like(scales),
            "d_rot": means3d.new_zeros((count, 3)),
            "d_opacity_logit": torch.zeros_like(opacity_logits),
            "appearance_offsets": means3d.new_zeros((count, 3)),
            "appearance_rgb_delta": means3d.new_zeros((count, 3)),
            "visibility_alpha": means3d.new_ones((count, 1)),
            "visibility_logits": means3d.new_zeros((count, 2)),
            "transient_probability": means3d.new_zeros((count, 1)),
            "pi_vis": pi_vis,
            "entropy_vis": means3d.new_zeros(()),
            "route_max_prob_vis": means3d.new_ones((count,)),
            "route_margin_vis": means3d.new_ones((count,)),
            "route_top1_vis_mean": means3d.new_ones(()),
            "lifecycle_logits": means3d.new_zeros((count, 2)),
            "lifecycle_probs": pi_vis,
            "lifecycle_alpha": means3d.new_ones((count, 1)),
        }

    def forward(
        self,
        canonical_means3d,
        canonical_scales,
        canonical_rotations,
        canonical_opacity,
        base_means3d,
        base_scales,
        base_rotations,
        base_opacity,
        time_values,
        scene_scale,
        camera=None,
    ):
        if self.refinement is None:
            refined = self._identity_refinement(
                base_means3d,
                base_scales,
                base_rotations,
                base_opacity,
            )
        else:
            refined = self.refinement(
                means3d=base_means3d,
                scales=base_scales,
                rotations=base_rotations,
                opacity_logits=base_opacity,
                time_values=time_values,
                scene_scale=scene_scale,
                camera=camera,
            )

        means_out = refined["means3d"]
        scales_out = refined["scales"]
        rotations_out = refined["rotations"]
        opacity_out = refined["opacity_logits"]
        count = means_out.shape[0]
        pi_geo = means_out.new_ones((count, 1))
        backbone_d_mu = base_means3d - canonical_means3d
        refinement_d_mu = means_out - base_means3d
        total_d_mu = means_out - canonical_means3d
        normalized_scene_scale = torch.as_tensor(
            scene_scale,
            device=means_out.device,
            dtype=means_out.dtype,
        ).reshape(()).abs().clamp_min(1e-6)
        aux = dict(refined)
        aux.update(
            {
                "expert_role": self.role,
                "means3d_canonical": canonical_means3d,
                "shared_base_means3d": base_means3d,
                "shared_base_d_mu": backbone_d_mu,
                "d_mu_backbone": backbone_d_mu,
                "d_mu_refinement": refinement_d_mu,
                "d_mu": total_d_mu,
                "d_scale": scales_out - canonical_scales,
                "d_opacity_logit": opacity_out - canonical_opacity,
                "pi_geo": pi_geo,
                "entropy_geo": means_out.new_zeros(()),
                "route_max_prob_geo": means_out.new_ones((count,)),
                "route_margin_geo": means_out.new_ones((count,)),
                "route_top1_geo_mean": means_out.new_ones(()),
                "tracking_phase_name": "endomoeg_expert_{}".format(self.role),
                "expert_opacity_includes_visibility": self.role == "contact",
            }
        )
        aux["global_motion_norm"] = (
            backbone_d_mu.norm(dim=-1) / normalized_scene_scale
        )
        if self.role == "local":
            aux["local_motion_norm"] = (
                refinement_d_mu.norm(dim=-1) / normalized_scene_scale
            )
        elif self.role == "contact":
            aux["cut_graph_motion_norm"] = (
                refinement_d_mu.norm(dim=-1) / normalized_scene_scale
            )
        return means_out, scales_out, rotations_out, opacity_out, aux
