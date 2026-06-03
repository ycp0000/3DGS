from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.tracking import (
    CAMSGSScheduler,
    CAMSGSTracking,
    HeterogeneousMoEScheduler,
    HeterogeneousMoETracking,
    SplitTrackingHead,
    TrackingPhase,
)
from scene.hexplane import HexPlaneField


class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=None, args=None):
        super().__init__()
        skips = [] if skips is None else skips
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.args = args

        tracking_mode = getattr(args, "tracking_type", "original").lower()
        if tracking_mode in {"disentangled", "disentangled_moe", "heterogeneous", "heterogeneous_moe"}:
            tracking_mode = "hetero_moe"
        if tracking_mode not in {"original", "split", "hetero_moe", "cams_gs"}:
            raise ValueError(f"Unsupported tracking_type: {tracking_mode}")
        self.tracking_mode = tracking_mode
        self.use_backbone = self.tracking_mode in {"original", "split", "hetero_moe", "cams_gs"}
        self.no_grid = bool(getattr(args, "no_grid", False) and self.use_backbone)

        self.grid: Optional[HexPlaneField] = None
        self.feature_out: Optional[nn.Sequential] = None
        self.pos_deform: Optional[nn.Sequential] = None
        self.scales_deform: Optional[nn.Sequential] = None
        self.rotations_deform: Optional[nn.Sequential] = None
        self.opacity_deform: Optional[nn.Sequential] = None

        if self.use_backbone:
            self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()

        self.latest_aux: Dict[str, torch.Tensor] = {}
        self.scene_scale = torch.tensor(float(getattr(args, "camera_extent", 1.0) or 1.0))
        self.prev_d_mu: Optional[torch.Tensor] = None
        self.latest_d_mu: Optional[torch.Tensor] = None
        self.current_phase: Optional[TrackingPhase] = None

        self.split_head: Optional[SplitTrackingHead] = None
        self.heterogeneous_head: Optional[HeterogeneousMoETracking] = None
        self.cams_head: Optional[CAMSGSTracking] = None
        self.scheduler = None

        if self.tracking_mode == "split":
            self.split_head = SplitTrackingHead(
                feature_dim=self.W,
                geo_hidden_dim=getattr(args, "geo_hidden_dim", 64),
                vis_hidden_dim=getattr(args, "vis_hidden_dim", 64),
                max_disp_smooth_ratio=getattr(args, "max_disp_smooth_ratio", 0.01),
                max_rot_smooth=getattr(args, "max_rot_smooth", 0.05),
                max_scale_smooth=getattr(args, "max_scale_smooth", 0.05),
                max_opacity_delta=getattr(args, "max_opacity_delta", 4.0),
            )
        elif self.tracking_mode == "hetero_moe":
            self.scheduler = HeterogeneousMoEScheduler(args)
            self.heterogeneous_head = HeterogeneousMoETracking(
                time_feature_dim=getattr(args, "timenet_output", 32),
                geo_hidden_dim=getattr(args, "geo_hidden_dim", 64),
                vis_hidden_dim=getattr(args, "vis_hidden_dim", 64),
                bounds=getattr(args, "bounds", 1.6),
                planeconfig=getattr(args, "kplanes_config"),
                multires=getattr(args, "multires"),
                max_disp_hexplane_ratio=getattr(args, "max_disp_hexplane_ratio", getattr(args, "max_disp_smooth_ratio", 0.01)),
                max_disp_local_ratio=getattr(args, "max_disp_local_ratio", 0.03),
                max_disp_smooth_ratio=getattr(args, "max_disp_smooth_ratio", 0.005),
                max_opacity_delta=getattr(args, "max_opacity_delta", 4.0),
                sat_threshold=getattr(args, "sat_threshold", 0.8),
                use_soft_routing=getattr(args, "use_soft_routing", True),
                use_topk=getattr(args, "use_topk", False),
                topk_geo=getattr(args, "topk_geo", getattr(args, "topk", 2)),
                topk_vis=getattr(args, "topk_vis", 1),
                router_noise_geo=getattr(args, "router_noise_geo", 0.0),
                router_noise_vis=getattr(args, "router_noise_vis", 0.0),
            )
        elif self.tracking_mode == "cams_gs":
            self.scheduler = CAMSGSScheduler(args)
            self.cams_head = CAMSGSTracking(
                time_feature_dim=getattr(args, "timenet_output", 32),
                max_disp_global_ratio=getattr(args, "max_disp_global_ratio", getattr(args, "max_disp_smooth_ratio", 0.01)),
                max_disp_local_ratio=getattr(args, "max_disp_local_ratio", 0.03),
                max_rot_local=getattr(args, "max_rot_local", getattr(args, "max_rot_smooth", 0.05)),
                max_scale_local=getattr(args, "max_scale_local", getattr(args, "max_scale_smooth", 0.05)),
                max_opacity_delta=getattr(args, "max_opacity_delta", 4.0),
                enable_scale=not bool(getattr(args, "no_ds", False)),
                enable_rotation=not bool(getattr(args, "no_dr", False)),
                enable_opacity=not bool(getattr(args, "no_do", False)),
            )

    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = nn.Sequential(nn.Linear(4, self.W), nn.ReLU())
        else:
            assert self.grid is not None
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim, self.W)]
            for _ in range(self.D - 1):
                self.feature_out.append(nn.ReLU())
                self.feature_out.append(nn.Linear(self.W, self.W))
            self.feature_out = nn.Sequential(*self.feature_out)

        return (
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4)),
            nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1)),
        )

    def query_time(self, rays_pts_emb, time_emb):
        if self.feature_out is None:
            raise RuntimeError("Backbone query requested when backbone is unavailable")
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], dim=-1)
        else:
            assert self.grid is not None
            h = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
        return self.feature_out(h)

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, time_emb=None, time_features=None, camera=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb, time_features, camera)

    def forward_static(self, rays_pts_emb):
        if not self.use_backbone:
            return rays_pts_emb[:, :3]
        zeros = torch.zeros((rays_pts_emb.shape[0], 1), device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
        hidden = self.query_time(rays_pts_emb, zeros).float()
        assert self.pos_deform is not None
        dx = self.pos_deform(hidden)
        return rays_pts_emb[:, :3] + dx

    def get_tracking_phase(self) -> Optional[TrackingPhase]:
        if self.scheduler is None:
            return None
        if self.current_phase is not None:
            return self.current_phase
        iteration = int(getattr(self.args, "current_iteration", 0))
        total_iterations = int(getattr(self.args, "iterations", 30000))
        return self.scheduler.build(iteration, total_iterations)

    def set_tracking_phase(self, phase: Optional[TrackingPhase]) -> None:
        self.current_phase = phase

    def _forward_original(self, hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb):
        max_disp_ratio = getattr(self.args, "max_disp_smooth_ratio", 0.01)
        max_rot = getattr(self.args, "max_rot_smooth", 0.05)
        max_scale = getattr(self.args, "max_scale_smooth", 0.05)
        max_opacity_delta = getattr(self.args, "max_opacity_delta", 4.0)
        scene_scale = self.scene_scale.to(hidden.device, hidden.dtype).reshape(()).abs().clamp_min(1e-6)

        assert self.pos_deform is not None
        assert self.scales_deform is not None
        assert self.rotations_deform is not None
        assert self.opacity_deform is not None

        dx = torch.tanh(self.pos_deform(hidden)) * (max_disp_ratio * scene_scale)
        pts = rays_pts_emb[:, :3] + dx

        if self.args.no_ds or scales_emb is None:
            scales = scales_emb[:, :3] if scales_emb is not None else torch.zeros_like(rays_pts_emb[:, :3])
        else:
            ds = torch.tanh(self.scales_deform(hidden)) * max_scale
            scales = scales_emb[:, :3] + ds

        if self.args.no_dr or rotations_emb is None:
            rotations = rotations_emb[:, :4] if rotations_emb is not None else torch.zeros(rays_pts_emb.shape[0], 4, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
            if rotations_emb is None:
                rotations[:, 0] = 1.0
        else:
            dr = torch.tanh(self.rotations_deform(hidden)) * max_rot
            rotations = rotations_emb[:, :4] + dr

        if self.args.no_do or opacity_emb is None:
            opacity = opacity_emb[:, :1] if opacity_emb is not None else torch.zeros(rays_pts_emb.shape[0], 1, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
        else:
            do = torch.tanh(self.opacity_deform(hidden)) * max_opacity_delta
            opacity = opacity_emb[:, :1] + do
        self.latest_aux = {}
        return pts, scales, rotations, opacity

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb, time_features, camera=None):
        if self.tracking_mode == "original":
            hidden = self.query_time(rays_pts_emb, time_emb).float()
            return self._forward_original(hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb)

        scene_scale = self.scene_scale.to(rays_pts_emb.device, rays_pts_emb.dtype).reshape(()).abs().clamp_min(1e-6)

        if self.tracking_mode == "split":
            if self.split_head is None:
                raise RuntimeError("split tracking mode selected but split head is not initialized")
            hidden = self.query_time(rays_pts_emb, time_emb).float()
            pts, scales, rotations, opacity, aux = self.split_head(
                hidden,
                rays_pts_emb[:, :3],
                scales_emb[:, :3],
                rotations_emb[:, :4],
                opacity_emb[:, :1],
                scene_scale,
            )
            self.latest_aux = aux
            self.latest_d_mu = aux["d_mu"].detach()
            return pts, scales, rotations, opacity

        if self.heterogeneous_head is None and self.cams_head is None:
            raise RuntimeError(f"{self.tracking_mode} tracking mode selected but head is not initialized")
        if time_features is None:
            raise RuntimeError(f"{self.tracking_mode} tracking requires time_features from the time encoder")

        hidden = self.query_time(rays_pts_emb, time_emb).float()
        if self.tracking_mode == "cams_gs":
            base_pts = rays_pts_emb[:, :3]
            base_scales = scales_emb[:, :3]
            base_rotations = rotations_emb[:, :4]
            base_opacity = opacity_emb[:, :1]
        else:
            base_pts, base_scales, base_rotations, base_opacity = self._forward_original(
                hidden,
                rays_pts_emb,
                scales_emb,
                rotations_emb,
                opacity_emb,
            )

        phase = self.get_tracking_phase()
        if phase is None:
            raise RuntimeError(f"Tracking phase is unavailable for {self.tracking_mode} mode")

        tracking_head = self.heterogeneous_head if self.tracking_mode == "hetero_moe" else self.cams_head
        assert tracking_head is not None
        pts, scales, rotations, opacity, aux = tracking_head(
            means3d=base_pts,
            scales=base_scales,
            rotations=base_rotations,
            opacity_logits=base_opacity,
            time_values=time_emb[:, :1],
            time_features=time_features,
            scene_scale=scene_scale,
            phase=phase,
            camera=camera,
        )
        self.latest_aux = aux
        self.latest_d_mu = aux["d_mu"].detach()
        return pts, scales, rotations, opacity

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def _get_backbone_mlp_parameters(self):
        modules = (
            self.feature_out,
            self.pos_deform,
            self.scales_deform,
            self.rotations_deform,
            self.opacity_deform,
        )
        parameters = []
        for module in modules:
            if module is not None:
                parameters.extend(list(module.parameters()))
        return parameters

    def get_grid_parameters(self):
        if self.grid is None:
            return []
        return list(self.grid.parameters())

    def get_tracking_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups: Dict[str, Iterable[nn.Parameter]] = {}
        if self.tracking_mode in {"hetero_moe", "cams_gs"} and self.use_backbone:
            backbone_parameters = self._get_backbone_mlp_parameters()
            if backbone_parameters:
                groups["tracking_base_deformation"] = backbone_parameters
            grid_parameters = self.get_grid_parameters()
            if grid_parameters:
                groups["tracking_base_grid"] = grid_parameters
        if self.heterogeneous_head is not None:
            groups.update(self.heterogeneous_head.named_parameter_groups())
        if self.cams_head is not None:
            groups.update(self.cams_head.named_parameter_groups())
        return groups

    def set_aabb(self, xyz_max, xyz_min) -> None:
        if self.grid is not None:
            self.grid.set_aabb(xyz_max, xyz_min)
        if self.heterogeneous_head is not None:
            self.heterogeneous_head.set_aabb(xyz_max, xyz_min)
        if self.cams_head is not None:
            self.cams_head.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        if self.grid is not None:
            for grids in self.grid.grids:
                yield grids
        if self.heterogeneous_head is not None:
            yield from self.heterogeneous_head.iter_regularized_grids()
        if self.cams_head is not None:
            yield from self.cams_head.iter_regularized_grids()

    def set_scene_scale(self, scale: float) -> None:
        scale = float(scale)
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            return
        self.scene_scale = torch.tensor(scale, dtype=torch.float32)

    def get_aux_outputs(self) -> Dict[str, torch.Tensor]:
        return self.latest_aux

    def get_previous_d_mu(self) -> Optional[torch.Tensor]:
        return self.prev_d_mu

    def get_latest_d_mu(self) -> Optional[torch.Tensor]:
        return self.latest_d_mu

    def commit_previous_d_mu(self, value: Optional[torch.Tensor] = None) -> None:
        if value is not None:
            self.prev_d_mu = value.detach().clone()
            return
        if self.latest_d_mu is not None:
            self.prev_d_mu = self.latest_d_mu.detach().clone()

    def reset_tracking_parameters(self) -> None:
        if self.split_head is not None:
            self.split_head.reset_parameters()
        if self.heterogeneous_head is not None:
            self.heterogeneous_head.reset_parameters()
        if self.cams_head is not None:
            self.cams_head.reset_parameters()

    def get_expert_names(self):
        if self.heterogeneous_head is not None:
            return (self.heterogeneous_head.GEO_EXPERT_NAMES, self.heterogeneous_head.VIS_EXPERT_NAMES)
        if self.cams_head is not None:
            return (self.cams_head.GEO_EXPERT_NAMES, self.cams_head.VIS_EXPERT_NAMES)
        return (("single",), ("single",))

    def get_tracking_arch_version(self) -> str:
        if self.tracking_mode == "hetero_moe":
            return "hetero_residual_v2"
        if self.tracking_mode == "cams_gs":
            return "cams_gs_v2"
        if self.tracking_mode == "split":
            return "split_v1"
        return "original_v1"


class deform_network(nn.Module):
    def __init__(self, args):
        super().__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2 * timebase_pe + 1

        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width),
            nn.ReLU(),
            nn.Linear(timenet_width, timenet_output),
        )

        self.deformation_net = Deformation(
            W=net_width,
            D=defor_depth,
            input_ch=(4 + 3) + ((4 + 3) * args.scale_rotation_pe) * 2,
            input_ch_time=timenet_output,
            args=args,
        )

        self.register_buffer("time_poc", torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.apply(initialize_weights)
        self.deformation_net.reset_tracking_parameters()

    def _encode_time(self, times_sel: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if times_sel is None:
            return None
        base = times_sel[:, :1]
        if self.time_poc.numel() == 0:
            encoded = base
        else:
            scaled = base * self.time_poc.view(1, -1)
            encoded = torch.cat([base, torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.timenet(encoded)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        return self.forward_static(point)

    def forward_static(self, points):
        return self.deformation_net(points)

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        time_features = self._encode_time(times_sel)
        return self.deformation_net(
            point,
            scales,
            rotations,
            opacity,
            times_sel,
            time_features=time_features,
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

    def get_tracking_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        groups = self.deformation_net.get_tracking_parameter_groups()
        if groups:
            groups = dict(groups)
            groups["tracking_time_encoder"] = self.timenet.parameters()
        return groups

    def get_optimizer_param_groups(self, training_args, spatial_lr_scale: float):
        tracking_groups = self.get_tracking_parameter_groups()
        if not tracking_groups:
            return [
                {
                    "params": list(self.get_mlp_parameters()),
                    "lr": training_args.deformation_lr_init * spatial_lr_scale,
                    "name": "deformation",
                    "schedule": "deformation",
                    "phase_lr_scale": 1.0,
                },
                {
                    "params": list(self.get_grid_parameters()),
                    "lr": training_args.grid_lr_init * spatial_lr_scale,
                    "name": "grid",
                    "schedule": "grid",
                    "phase_lr_scale": 1.0,
                },
            ]

        groups = []
        for name, params_iter in tracking_groups.items():
            params = list(params_iter)
            if not params:
                continue
            schedule = "grid" if "grid" in name else "deformation"
            base_lr = training_args.grid_lr_init if schedule == "grid" else training_args.deformation_lr_init
            groups.append(
                {
                    "params": params,
                    "lr": base_lr * spatial_lr_scale,
                    "name": name,
                    "schedule": schedule,
                    "phase_lr_scale": 1.0,
                }
            )
        return groups

    def set_tracking_phase(self, phase: Optional[TrackingPhase]) -> None:
        self.deformation_net.set_tracking_phase(phase)

    def get_tracking_phase(self) -> Optional[TrackingPhase]:
        return self.deformation_net.get_tracking_phase()

    def get_aux_outputs(self) -> Dict[str, torch.Tensor]:
        return self.deformation_net.get_aux_outputs()

    def get_previous_d_mu(self) -> Optional[torch.Tensor]:
        return self.deformation_net.get_previous_d_mu()

    def get_latest_d_mu(self) -> Optional[torch.Tensor]:
        return self.deformation_net.get_latest_d_mu()

    def commit_previous_d_mu(self, value: Optional[torch.Tensor] = None) -> None:
        self.deformation_net.commit_previous_d_mu(value)

    def set_scene_scale(self, scale: float) -> None:
        self.deformation_net.set_scene_scale(scale)

    def set_aabb(self, xyz_max, xyz_min) -> None:
        self.deformation_net.set_aabb(xyz_max, xyz_min)

    def iter_regularized_grids(self):
        yield from self.deformation_net.iter_regularized_grids()

    def get_expert_names(self):
        return self.deformation_net.get_expert_names()


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            init.zeros_(module.bias)
