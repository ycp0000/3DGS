from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField
from models.tracking import DisentangledMoETracking, SplitTrackingHead
import numpy as np
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        self.args = args
        tracking_mode = getattr(args, "tracking_type", "original").lower()
        if tracking_mode in {"disentangled", "disentangled_moe"}:
            tracking_mode = "disentangled_moe"
        if tracking_mode not in {"original", "split", "disentangled_moe"}:
            raise ValueError(f"Unsupported tracking_type: {tracking_mode}")
        self.tracking_mode = tracking_mode

        self.latest_aux: Dict[str, torch.Tensor] = {}
        self.scene_scale = torch.tensor(float(getattr(args, "camera_extent", 1.0) or 1.0))
        self.prev_d_mu: Optional[torch.Tensor] = None
        self.latest_d_mu: Optional[torch.Tensor] = None

        self.split_head: Optional[SplitTrackingHead] = None
        self.disentangled_head: Optional[DisentangledMoETracking] = None

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
        elif self.tracking_mode == "disentangled_moe":
            self.disentangled_head = DisentangledMoETracking(
                feature_dim=self.W,
                geo_router_in_dim=self.W + 3 + 1 + 3 + 1,
                vis_router_in_dim=self.W + 3 + 1 + 1,
                geo_hidden_dim=getattr(args, "geo_hidden_dim", 64),
                vis_hidden_dim=getattr(args, "vis_hidden_dim", 64),
                k_geo=getattr(args, "K_geo", 3),
                k_vis=getattr(args, "K_vis", 2),
                max_disp_smooth_ratio=getattr(args, "max_disp_smooth_ratio", 0.01),
                max_disp_local_ratio=getattr(args, "max_disp_local_ratio", 0.001),
                max_rot_smooth=getattr(args, "max_rot_smooth", 0.05),
                max_rot_local=getattr(args, "max_rot_local", 0.10),
                max_scale_smooth=getattr(args, "max_scale_smooth", 0.05),
                max_scale_local=getattr(args, "max_scale_local", 0.10),
                max_opacity_delta=getattr(args, "max_opacity_delta", 4.0),
                raw_scale_disp=getattr(args, "raw_scale_disp", 0.05),
                raw_scale_rot=getattr(args, "raw_scale_rot", 0.05),
                raw_scale_scale=getattr(args, "raw_scale_scale", 0.05),
                raw_scale_opacity=getattr(args, "raw_scale_opacity", 0.05),
                sat_threshold=getattr(args, "sat_threshold", 0.8),
                raw_limit=getattr(args, "raw_limit", 1.1),
                freeze_scale_branch=getattr(args, "freeze_scale_branch", True),
                scale_branch_multiplier=getattr(args, "scale_branch_multiplier", 0.0),
                freeze_rot_branch=getattr(args, "freeze_rot_branch", False),
                rot_branch_multiplier=getattr(args, "rot_branch_multiplier", 1.0),
                max_disp_shared_ratio=getattr(args, "max_disp_shared_ratio", getattr(args, "max_disp_smooth_ratio", 0.01)),
                max_rot_shared=getattr(args, "max_rot_shared", getattr(args, "max_rot_smooth", 0.05)),
                max_scale_shared=getattr(args, "max_scale_shared", getattr(args, "max_scale_smooth", 0.05)),
                use_soft_routing=getattr(args, "use_soft_routing", True),
                use_topk=getattr(args, "use_topk", False),
                topk_geo=getattr(args, "topk_geo", getattr(args, "topk", 2)),
                topk_vis=getattr(args, "topk_vis", 1),
                router_noise_geo=getattr(args, "router_noise_geo", 0.0),
                router_noise_vis=getattr(args, "router_noise_vis", 0.0),
            )
        
    def create_net(self):
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            h = grid_feature
        h = self.feature_out(h)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        if self.no_grid:
            zeros = torch.zeros((rays_pts_emb.shape[0], 1), device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
            h = torch.cat([rays_pts_emb[:, :3], zeros], dim=-1)
        else:
            zeros = torch.zeros((rays_pts_emb.shape[0], 1), device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
            h = self.grid(rays_pts_emb[:, :3], zeros)
        h = self.feature_out(h).float()
        dx = self.pos_deform(h)
        return rays_pts_emb[:, :3] + dx

    def _set_stage_context(self, time_emb: torch.Tensor) -> Dict[str, float]:
        del time_emb
        iteration = int(getattr(self.args, "current_iteration", 0))
        warmup_iters = int(getattr(self.args, "warmup_iters", 1000))
        enable_smooth_geo_iter = int(getattr(self.args, "enable_smooth_geo_iter", warmup_iters))
        enable_local_geo_iter = int(getattr(self.args, "enable_local_geo_iter", 1500))
        enable_visibility_iter = int(getattr(self.args, "enable_visibility_iter", 2000))
        enable_shared_only_iter = int(getattr(self.args, "enable_shared_only_iter", warmup_iters))
        enable_sparse_routing_iter = int(getattr(self.args, "enable_sparse_routing_iter", enable_local_geo_iter))
        enable_route_stability_iter = int(getattr(self.args, "enable_route_stability_iter", enable_sparse_routing_iter))

        if iteration < enable_smooth_geo_iter:
            active_geo = 1
        elif iteration < enable_local_geo_iter:
            active_geo = 2
        else:
            active_geo = getattr(self.args, "K_geo", 3)

        active_vis = 1 if iteration < enable_visibility_iter else getattr(self.args, "K_vis", 2)

        max_iter = max(1, int(getattr(self.args, "iterations", 30000)))
        progress = min(max(iteration / max_iter, 0.0), 1.0)
        temperature_geo = (
            getattr(self.args, "temperature_geo_init", 2.0) * (1.0 - progress)
            + getattr(self.args, "temperature_geo_final", 0.7) * progress
        )
        temperature_vis = (
            getattr(self.args, "temperature_vis_init", 2.0) * (1.0 - progress)
            + getattr(self.args, "temperature_vis_final", 1.0) * progress
        )

        shared_only = iteration < enable_shared_only_iter
        geo_residual_gate = 0.0 if shared_only else min(
            1.0,
            max(0.0, (iteration - enable_shared_only_iter) / max(1, enable_local_geo_iter - enable_shared_only_iter))
        )
        vis_residual_gate = 1.0 if iteration >= enable_visibility_iter else 0.0
        use_sparse_geo = bool(getattr(self.args, "use_topk", False) and iteration >= enable_sparse_routing_iter)
        use_sparse_vis = bool(getattr(self.args, "use_topk", False) and iteration >= enable_visibility_iter)

        return {
            "active_geo": int(active_geo),
            "active_vis": int(active_vis),
            "temperature_geo": float(temperature_geo),
            "temperature_vis": float(temperature_vis),
            "enable_visibility": bool(getattr(self.args, "enable_visibility", True) and iteration >= enable_visibility_iter),
            "shared_only": bool(shared_only),
            "geo_residual_gate": float(geo_residual_gate),
            "vis_residual_gate": float(vis_residual_gate),
            "use_sparse_geo": bool(use_sparse_geo),
            "use_sparse_vis": bool(use_sparse_vis),
            "topk_geo": int(getattr(self.args, "topk_geo", getattr(self.args, "topk", 2))),
            "topk_vis": int(getattr(self.args, "topk_vis", 1)),
            "route_stability_active": bool(iteration >= enable_route_stability_iter),
        }

    def _forward_original(self, hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb):
        # Bounded output for numerical stability (aligned with split/MoE modes)
        max_disp_ratio = getattr(self.args, "max_disp_smooth_ratio", 0.01)
        max_rot = getattr(self.args, "max_rot_smooth", 0.05)
        max_scale = getattr(self.args, "max_scale_smooth", 0.05)
        max_opacity_delta = getattr(self.args, "max_opacity_delta", 4.0)

        scene_scale = self.scene_scale.to(hidden.device, hidden.dtype).reshape(())
        if (not torch.isfinite(scene_scale)) or scene_scale.abs() < 1e-8:
            scene_scale = torch.tensor(100.0, device=hidden.device, dtype=hidden.dtype)

        dx_raw = self.pos_deform(hidden)
        dx = torch.tanh(dx_raw) * (max_disp_ratio * scene_scale)
        pts = rays_pts_emb[:, :3] + dx

        if self.args.no_ds or scales_emb is None:
            scales = scales_emb[:, :3] if scales_emb is not None else torch.zeros_like(rays_pts_emb[:, :3])
        else:
            ds_raw = self.scales_deform(hidden)
            ds = torch.tanh(ds_raw) * max_scale
            scales = scales_emb[:, :3] + ds

        if self.args.no_dr or rotations_emb is None:
            rotations = rotations_emb[:, :4] if rotations_emb is not None else torch.zeros(rays_pts_emb.shape[0], 4, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
            if rotations_emb is None:
                rotations[:, 0] = 1.0
        else:
            dr_raw = self.rotations_deform(hidden)
            dr = torch.tanh(dr_raw) * max_rot
            rotations = rotations_emb[:, :4] + dr

        if self.args.no_do or opacity_emb is None:
            opacity = opacity_emb[:, :1] if opacity_emb is not None else torch.zeros(rays_pts_emb.shape[0], 1, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
        else:
            do_raw = self.opacity_deform(hidden)
            do = torch.tanh(do_raw) * max_opacity_delta
            opacity = opacity_emb[:, :1] + do
        self.latest_aux = {}
        return pts, scales, rotations, opacity

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()

        if self.tracking_mode == "original":
            return self._forward_original(hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb)

        # scene_scale = self.scene_scale.to(hidden.device, hidden.dtype)

        scene_scale = self.scene_scale.to(hidden.device, hidden.dtype).reshape(())

        if (not torch.isfinite(scene_scale)) or scene_scale.abs() < 1e-8:
            with torch.no_grad():
                xyz = rays_pts_emb[:, :3].detach()
                lo = torch.quantile(xyz, 0.05, dim=0)
                hi = torch.quantile(xyz, 0.95, dim=0)
                fallback_scale = (hi - lo).norm().clamp_min(1e-6)

            print("[WARN] scene_scale is invalid in forward_dynamic; using fallback_scale =", float(fallback_scale))
            scene_scale = fallback_scale



        if self.tracking_mode == "split":
            if self.split_head is None:
                raise RuntimeError("split tracking mode selected but split head is not initialized")
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

        if self.disentangled_head is None:
            raise RuntimeError("disentangled_moe tracking mode selected but head is not initialized")

        stage_ctx = self._set_stage_context(time_emb)
        pts, scales, rotations, opacity, aux = self.disentangled_head(
            hidden,
            rays_pts_emb[:, :3],
            scales_emb[:, :3],
            rotations_emb[:, :4],
            opacity_emb[:, :1],
            time_emb[:, :1],
            scene_scale,
            temperature_geo=stage_ctx["temperature_geo"],
            temperature_vis=stage_ctx["temperature_vis"],
            active_geo=stage_ctx["active_geo"],
            active_vis=stage_ctx["active_vis"],
            enable_visibility=stage_ctx["enable_visibility"],
            use_sparse_geo=stage_ctx["use_sparse_geo"],
            use_sparse_vis=stage_ctx["use_sparse_vis"],
            topk_geo=stage_ctx["topk_geo"],
            topk_vis=stage_ctx["topk_vis"],
            geo_residual_gate=stage_ctx["geo_residual_gate"],
            vis_residual_gate=stage_ctx["vis_residual_gate"],
        )
        self.latest_aux = aux
        self.latest_d_mu = aux["d_mu"].detach()
        return pts, scales, rotations, opacity
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        return list(self.grid.parameters())

    def set_scene_scale(self, scale: float) -> None:
        scale = float(scale)

        if not np.isfinite(scale) or abs(scale) < 1e-8:
            print(
                f"[WARN] Invalid deformation scene_scale={scale}. "
                f"Keep previous scene_scale={float(self.scene_scale)}"
            )
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
        if self.disentangled_head is not None:
            self.disentangled_head.reset_parameters()


class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))
        
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args)
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        self.deformation_net.reset_tracking_parameters()
    
    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        means3D, scales, rotations, opacity = self.deformation_net( point,
                                                scales,
                                                rotations,
                                                opacity,
                                                times_sel)
        return means3D, scales, rotations, opacity
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

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

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.zeros_(m.bias)
