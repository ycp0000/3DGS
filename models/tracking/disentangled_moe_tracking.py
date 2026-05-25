from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _init_mlp_small_output(
    mlp: nn.Sequential,
    hidden_gain: float = 1.0,
    final_std: float = 1e-4,
    final_bias: float = 0.0,
) -> None:
    linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
    if not linear_layers:
        return

    for layer in linear_layers[:-1]:
        nn.init.xavier_uniform_(layer.weight, gain=hidden_gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    final = linear_layers[-1]
    nn.init.normal_(final.weight, mean=0.0, std=final_std)
    if final.bias is not None:
        nn.init.constant_(final.bias, final_bias)


def _init_router_mlp(
    mlp: nn.Sequential,
    final_bias: Optional[torch.Tensor] = None,
    hidden_gain: float = 1.0,
    final_std: float = 1e-4,
) -> None:
    linear_layers = [m for m in mlp if isinstance(m, nn.Linear)]
    if not linear_layers:
        return

    for layer in linear_layers[:-1]:
        nn.init.xavier_uniform_(layer.weight, gain=hidden_gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    final = linear_layers[-1]
    nn.init.normal_(final.weight, mean=0.0, std=final_std)
    if final.bias is not None:
        if final_bias is None:
            nn.init.zeros_(final.bias)
        else:
            with torch.no_grad():
                final.bias.copy_(final_bias.to(final.bias.device, final.bias.dtype))


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


def _build_router_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )


class _ZeroGeoExpert(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], 9, device=features.device, dtype=features.dtype)


class _BoundedGeoExpert(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        max_disp_ratio: float,
        max_rot: float,
        max_scale: float,
        raw_scale_disp: float = 0.05,
        raw_scale_rot: float = 0.05,
        raw_scale_scale: float = 0.05,
        sat_threshold: float = 0.8,
        raw_limit: float = 1.1,
    ) -> None:
        super().__init__()
        self.disp_net = _build_mlp(in_dim, hidden_dim, 3)
        self.rot_net = _build_mlp(in_dim, hidden_dim, 3)
        self.scl_net = _build_mlp(in_dim, hidden_dim, 3)

        self.max_disp_ratio = float(max_disp_ratio)
        self.max_rot = float(max_rot)
        self.max_scale = float(max_scale)

        self.raw_scale_disp = float(raw_scale_disp)
        self.raw_scale_rot = float(raw_scale_rot)
        self.raw_scale_scale = float(raw_scale_scale)

        self.sat_threshold = float(sat_threshold)
        self.raw_limit = float(raw_limit)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_mlp_small_output(self.disp_net, final_std=1e-4, final_bias=0.0)
        _init_mlp_small_output(self.rot_net, final_std=1e-4, final_bias=0.0)
        _init_mlp_small_output(self.scl_net, final_std=1e-4, final_bias=0.0)

    def forward(
        self,
        features: torch.Tensor,
        scene_scale: torch.Tensor,
        return_debug: bool = False,
    ):
        raw_disp_unscaled = self.disp_net(features)
        raw_rot_unscaled = self.rot_net(features)
        raw_scl_unscaled = self.scl_net(features)

        raw_disp = self.raw_scale_disp * raw_disp_unscaled
        raw_rot = self.raw_scale_rot * raw_rot_unscaled
        raw_scl = self.raw_scale_scale * raw_scl_unscaled

        disp_scale_factor = self.max_disp_ratio * scene_scale
        tanh_disp = torch.tanh(raw_disp)
        tanh_rot = torch.tanh(raw_rot)
        tanh_scl = torch.tanh(raw_scl)

        disp = tanh_disp * disp_scale_factor
        rot = tanh_rot * self.max_rot
        scl = tanh_scl * self.max_scale
        out = torch.cat([disp, rot, scl], dim=-1)

        if not return_debug:
            return out

        loss_sat_disp_per_point = torch.relu(tanh_disp.abs() - self.sat_threshold).pow(2).mean(dim=-1)
        loss_sat_rot_per_point = torch.relu(tanh_rot.abs() - self.sat_threshold).pow(2).mean(dim=-1)
        loss_sat_scl_per_point = torch.relu(tanh_scl.abs() - self.sat_threshold).pow(2).mean(dim=-1)

        disp_den = torch.as_tensor(disp_scale_factor, device=features.device, dtype=features.dtype).clamp_min(1e-8)
        rot_den = torch.tensor(float(self.max_rot), device=features.device, dtype=features.dtype).clamp_min(1e-8)
        scl_den = torch.tensor(float(self.max_scale), device=features.device, dtype=features.dtype).clamp_min(1e-8)

        loss_mag_disp_per_point = (disp / disp_den).pow(2).mean(dim=-1)
        loss_mag_rot_per_point = (rot / rot_den).pow(2).mean(dim=-1)
        loss_mag_scl_per_point = (scl / scl_den).pow(2).mean(dim=-1)

        loss_raw_disp_per_point = torch.relu(raw_disp.abs() - self.raw_limit).pow(2).mean(dim=-1)
        loss_raw_rot_per_point = torch.relu(raw_rot.abs() - self.raw_limit).pow(2).mean(dim=-1)
        loss_raw_scl_per_point = torch.relu(raw_scl.abs() - self.raw_limit).pow(2).mean(dim=-1)

        debug = {
            "bounded_disp_norm_mean": torch.norm(disp.detach(), dim=-1).mean(),
            "bounded_disp_norm_max": torch.norm(disp.detach(), dim=-1).max(),
            "bounded_rot_norm_mean": torch.norm(rot.detach(), dim=-1).mean(),
            "bounded_rot_norm_max": torch.norm(rot.detach(), dim=-1).max(),
            "bounded_scl_norm_mean": torch.norm(scl.detach(), dim=-1).mean(),
            "bounded_scl_norm_max": torch.norm(scl.detach(), dim=-1).max(),
            "raw_disp_norm_mean": torch.norm(raw_disp.detach(), dim=-1).mean(),
            "raw_rot_norm_mean": torch.norm(raw_rot.detach(), dim=-1).mean(),
            "raw_scl_norm_mean": torch.norm(raw_scl.detach(), dim=-1).mean(),
            "loss_sat_disp_per_point": loss_sat_disp_per_point,
            "loss_sat_rot_per_point": loss_sat_rot_per_point,
            "loss_sat_scl_per_point": loss_sat_scl_per_point,
            "loss_mag_disp_per_point": loss_mag_disp_per_point,
            "loss_mag_rot_per_point": loss_mag_rot_per_point,
            "loss_mag_scl_per_point": loss_mag_scl_per_point,
            "loss_raw_disp_per_point": loss_raw_disp_per_point,
            "loss_raw_rot_per_point": loss_raw_rot_per_point,
            "loss_raw_scl_per_point": loss_raw_scl_per_point,
        }
        return out, debug


class _ZeroVisExpert(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(features.shape[0], 1, device=features.device, dtype=features.dtype)


class _TransientVisExpert(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        max_opacity_delta: float,
        raw_scale_opacity: float = 0.05,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(in_dim, hidden_dim, 1)
        self.max_opacity_delta = float(max_opacity_delta)
        self.raw_scale_opacity = float(raw_scale_opacity)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_mlp_small_output(self.net, final_std=1e-4, final_bias=0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.raw_scale_opacity * self.net(features)
        return torch.tanh(raw) * self.max_opacity_delta


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
        super().__init__()
        self.geo_head = _BoundedGeoExpert(
            in_dim=feature_dim,
            hidden_dim=geo_hidden_dim,
            max_disp_ratio=max_disp_smooth_ratio,
            max_rot=max_rot_smooth,
            max_scale=max_scale_smooth,
        )
        self.vis_head = _TransientVisExpert(
            in_dim=feature_dim,
            hidden_dim=vis_hidden_dim,
            max_opacity_delta=max_opacity_delta,
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        geo_delta = self.geo_head(features, scene_scale)
        d_mu = geo_delta[:, 0:3]
        d_rot = geo_delta[:, 3:6]
        d_scale = geo_delta[:, 6:9]

        means_t = means3d + d_mu
        scales_t = scales + d_scale
        rotations_t = rotations.clone()
        rotations_t[:, 1:4] = rotations_t[:, 1:4] + d_rot

        d_opacity = self.vis_head(features)
        opacity_current = torch.sigmoid(opacity_logits).clamp(1e-6, 1.0 - 1e-6)
        opacity_t = torch.sigmoid(torch.logit(opacity_current) + d_opacity)
        opacity_logits_t = torch.logit(opacity_t.clamp(1e-6, 1.0 - 1e-6))

        aux = {
            "d_mu": d_mu,
            "d_opacity_logit": d_opacity,
            "pi_geo": torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
            "pi_vis": torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype),
            "entropy_geo": torch.zeros((), device=features.device, dtype=features.dtype),
            "entropy_vis": torch.zeros((), device=features.device, dtype=features.dtype),
        }
        return means_t, scales_t, rotations_t, opacity_logits_t, aux


class DisentangledMoETracking(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        geo_router_in_dim: int,
        vis_router_in_dim: int,
        geo_hidden_dim: int,
        vis_hidden_dim: int,
        k_geo: int,
        k_vis: int,
        max_disp_smooth_ratio: float,
        max_disp_local_ratio: float,
        max_rot_smooth: float,
        max_rot_local: float,
        max_scale_smooth: float,
        max_scale_local: float,
        max_opacity_delta: float,
        raw_scale_disp: float = 0.05,
        raw_scale_rot: float = 0.05,
        raw_scale_scale: float = 0.05,
        raw_scale_opacity: float = 0.05,
        sat_threshold: float = 0.8,
        freeze_scale_branch: bool = True,
        scale_branch_multiplier: float = 0.0,
        raw_limit: float = 1.1,
        freeze_rot_branch: bool = False,
        rot_branch_multiplier: float = 1.0,
        max_disp_shared_ratio: Optional[float] = None,
        max_rot_shared: Optional[float] = None,
        max_scale_shared: Optional[float] = None,
        use_soft_routing: bool = True,
        use_topk: bool = False,
        topk_geo: Optional[int] = None,
        topk_vis: Optional[int] = None,
        router_noise_geo: float = 0.0,
        router_noise_vis: float = 0.0,
    ) -> None:
        super().__init__()

        if k_geo != 3:
            raise ValueError("DisentangledMoETracking currently requires K_geo=3 (static/smooth/local).")
        if k_vis != 2:
            raise ValueError("DisentangledMoETracking currently requires K_vis=2 (stable/transient).")

        self.k_geo = int(k_geo)
        self.k_vis = int(k_vis)
        self.freeze_scale_branch = bool(freeze_scale_branch)
        self.scale_branch_multiplier = float(scale_branch_multiplier)
        self.freeze_rot_branch = bool(freeze_rot_branch)
        self.rot_branch_multiplier = float(rot_branch_multiplier)

        self.use_soft_routing = bool(use_soft_routing)
        self.use_topk = bool(use_topk)
        self.topk_geo = int(topk_geo if topk_geo is not None else min(2, k_geo))
        self.topk_vis = int(topk_vis if topk_vis is not None else 1)
        self.router_noise_geo = float(router_noise_geo)
        self.router_noise_vis = float(router_noise_vis)

        self.geometry_router = _build_router_mlp(geo_router_in_dim, geo_hidden_dim, k_geo)
        self.visibility_router = _build_router_mlp(vis_router_in_dim, vis_hidden_dim, k_vis)

        self.shared_geo_head = _BoundedGeoExpert(
            in_dim=feature_dim,
            hidden_dim=geo_hidden_dim,
            max_disp_ratio=float(max_disp_shared_ratio if max_disp_shared_ratio is not None else max_disp_smooth_ratio),
            max_rot=float(max_rot_shared if max_rot_shared is not None else max_rot_smooth),
            max_scale=float(max_scale_shared if max_scale_shared is not None else max_scale_smooth),
            raw_scale_disp=raw_scale_disp,
            raw_scale_rot=raw_scale_rot,
            raw_scale_scale=raw_scale_scale,
            sat_threshold=sat_threshold,
            raw_limit=raw_limit,
        )

        self.geo_experts = nn.ModuleList([
            _ZeroGeoExpert(),
            _BoundedGeoExpert(
                in_dim=feature_dim,
                hidden_dim=geo_hidden_dim,
                max_disp_ratio=max_disp_smooth_ratio,
                max_rot=max_rot_smooth,
                max_scale=max_scale_smooth,
                raw_scale_disp=raw_scale_disp,
                raw_scale_rot=raw_scale_rot,
                raw_scale_scale=raw_scale_scale,
                sat_threshold=sat_threshold,
                raw_limit=raw_limit,
            ),
            _BoundedGeoExpert(
                in_dim=feature_dim,
                hidden_dim=geo_hidden_dim,
                max_disp_ratio=max_disp_local_ratio,
                max_rot=max_rot_local,
                max_scale=max_scale_local,
                raw_scale_disp=raw_scale_disp,
                raw_scale_rot=raw_scale_rot,
                raw_scale_scale=raw_scale_scale,
                sat_threshold=sat_threshold,
                raw_limit=raw_limit,
            ),
        ])

        self.vis_experts = nn.ModuleList([
            _ZeroVisExpert(),
            _TransientVisExpert(
                in_dim=vis_router_in_dim,
                hidden_dim=vis_hidden_dim,
                max_opacity_delta=max_opacity_delta,
                raw_scale_opacity=raw_scale_opacity,
            ),
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_router_mlp(self.geometry_router, final_bias=torch.zeros(self.k_geo), final_std=1e-4)
        _init_router_mlp(self.visibility_router, final_bias=torch.tensor([2.0, -2.0]), final_std=1e-4)
        self.shared_geo_head.reset_parameters()
        for expert in self.geo_experts:
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters()
        for expert in self.vis_experts:
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters()

    def _build_geometry_router_features(
        self,
        features: torch.Tensor,
        means3d: torch.Tensor,
        time_emb: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([features, means3d, time_emb, scales, opacity], dim=-1)

    def _build_visibility_router_features(
        self,
        features: torch.Tensor,
        means3d_t: torch.Tensor,
        opacity: torch.Tensor,
        d_mu: torch.Tensor,
    ) -> torch.Tensor:
        d_mu_norm = torch.norm(d_mu, dim=-1, keepdim=True)
        return torch.cat([features, means3d_t, opacity, d_mu_norm], dim=-1)

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

        if not self.use_soft_routing and not use_sparse and active_count > 1:
            top1_idx = torch.argmax(logits[:, :active_count], dim=-1, keepdim=True)
            hard_mask = torch.full_like(logits[:, :active_count], -1e9)
            hard_mask.scatter_(1, top1_idx, 0.0)
            logits = torch.cat([logits[:, :active_count] + hard_mask, logits[:, active_count:]], dim=-1)

        return torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)

    def _weighted_expert_loss(
        self,
        geo_loss_terms: Dict[str, torch.Tensor],
        pi_geo: torch.Tensor,
        expert_idx: int,
        key: str,
        features: torch.Tensor,
    ) -> torch.Tensor:
        value = geo_loss_terms.get(f"geo_e{expert_idx}_{key}")
        if value is None:
            return torch.zeros((), device=features.device, dtype=features.dtype)
        if value.dim() == 0:
            return value
        pi_k = pi_geo[:, expert_idx].detach()
        return (pi_k * value).mean()

    def _route_stats(self, pi: torch.Tensor) -> Dict[str, torch.Tensor]:
        topk = min(2, pi.shape[-1])
        values, indices = torch.topk(pi, k=topk, dim=-1)
        max_prob = values[:, 0]
        margin = max_prob if topk == 1 else (values[:, 0] - values[:, 1])
        return {
            "max_prob": max_prob.mean(),
            "margin": margin.mean(),
            "top1_index_mean": indices[:, 0].float().mean(),
        }

    def forward(
        self,
        features: torch.Tensor,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_emb: torch.Tensor,
        scene_scale: torch.Tensor,
        temperature_geo: float,
        temperature_vis: float,
        active_geo: int,
        active_vis: int,
        enable_visibility: bool,
        camera: Optional[object] = None,
        tool_mask: Optional[torch.Tensor] = None,
        use_sparse_geo: Optional[bool] = None,
        use_sparse_vis: Optional[bool] = None,
        topk_geo: Optional[int] = None,
        topk_vis: Optional[int] = None,
        geo_residual_gate: float = 1.0,
        vis_residual_gate: float = 1.0,
        router_noise_geo: Optional[float] = None,
        router_noise_vis: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del camera, tool_mask

        opacity = torch.sigmoid(opacity_logits)
        use_sparse_geo = self.use_topk if use_sparse_geo is None else bool(use_sparse_geo)
        use_sparse_vis = self.use_topk if use_sparse_vis is None else bool(use_sparse_vis)
        topk_geo = self.topk_geo if topk_geo is None else int(topk_geo)
        topk_vis = self.topk_vis if topk_vis is None else int(topk_vis)
        router_noise_geo = self.router_noise_geo if router_noise_geo is None else float(router_noise_geo)
        router_noise_vis = self.router_noise_vis if router_noise_vis is None else float(router_noise_vis)

        z_geo = self._build_geometry_router_features(features, means3d, time_emb, scales, opacity)
        logits_geo = self.geometry_router(z_geo)
        pi_geo = self._route(
            logits_geo,
            temperature_geo,
            active_geo,
            use_sparse=use_sparse_geo,
            topk=topk_geo,
            noise_scale=router_noise_geo,
        )

        shared_geo, shared_dbg = self.shared_geo_head(features, scene_scale, return_debug=True)

        geo_outputs = []
        geo_debug: Dict[str, torch.Tensor] = {}
        geo_loss_terms: Dict[str, torch.Tensor] = {}
        for idx, expert in enumerate(self.geo_experts):
            if idx == 0:
                out = expert(features)
            else:
                out, dbg = expert(features, scene_scale, return_debug=True)
                for key, value in dbg.items():
                    if key.startswith("loss_"):
                        geo_loss_terms[f"geo_e{idx}_{key}"] = value
                    else:
                        geo_debug[f"dbg_geo_e{idx}_{key}"] = value
            geo_outputs.append(out)

        geo_stack = torch.stack(geo_outputs, dim=1)
        geo_residual = (pi_geo.unsqueeze(-1) * geo_stack).sum(dim=1)
        geo_weight = float(geo_residual_gate)
        geo_total = shared_geo + geo_weight * geo_residual

        d_mu = geo_total[:, 0:3]
        d_rot_raw = geo_total[:, 3:6]
        d_scale_raw = geo_total[:, 6:9]

        if self.freeze_rot_branch:
            d_rot = torch.zeros_like(d_rot_raw)
        else:
            d_rot = self.rot_branch_multiplier * d_rot_raw

        if self.freeze_scale_branch:
            d_scale = torch.zeros_like(d_scale_raw)
        else:
            d_scale = self.scale_branch_multiplier * d_scale_raw

        means_t = means3d + d_mu
        scales_t = scales + d_scale
        rotations_t = rotations.clone()
        rotations_t[:, 1:4] = rotations_t[:, 1:4] + d_rot

        z_vis = self._build_visibility_router_features(features, means_t.detach(), opacity.detach(), d_mu.detach())
        logits_vis = self.visibility_router(z_vis)
        vis_active_count = active_vis if enable_visibility else 1
        pi_vis = self._route(
            logits_vis,
            temperature_vis,
            vis_active_count,
            use_sparse=use_sparse_vis and enable_visibility,
            topk=topk_vis,
            noise_scale=router_noise_vis,
        )

        vis_outputs = [self.vis_experts[0](z_vis), self.vis_experts[1](z_vis)]
        vis_stack = torch.stack(vis_outputs, dim=1)
        vis_residual = (pi_vis.unsqueeze(-1) * vis_stack).sum(dim=1)
        d_opacity_logit = float(vis_residual_gate) * vis_residual if enable_visibility else torch.zeros_like(vis_residual)

        opacity_current = opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity_t = torch.sigmoid(torch.logit(opacity_current) + d_opacity_logit)
        opacity_logits_t = torch.logit(opacity_t.clamp(1e-6, 1.0 - 1e-6))

        entropy_geo = -(pi_geo * torch.log(pi_geo + 1e-8)).sum(dim=-1).mean()
        entropy_vis = -(pi_vis * torch.log(pi_vis + 1e-8)).sum(dim=-1).mean()

        geo_stats = self._route_stats(pi_geo)
        vis_stats = self._route_stats(pi_vis)

        smooth_disp = geo_stack[:, 1, 0:3]
        local_disp = geo_stack[:, 2, 0:3]
        smooth_den = torch.norm(smooth_disp, dim=-1).clamp_min(1e-6)
        local_den = torch.norm(local_disp, dim=-1).clamp_min(1e-6)
        expert_diversity_geo = (
            ((smooth_disp * local_disp).sum(dim=-1) / (smooth_den * local_den)).abs()
        ).mean()

        geo_reg_aux = {
            "loss_geo_e1_sat_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_sat_disp_per_point", features),
            "loss_geo_e1_sat_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_sat_rot_per_point", features),
            "loss_geo_e1_sat_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_sat_scl_per_point", features),
            "loss_geo_e1_mag_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_mag_disp_per_point", features),
            "loss_geo_e1_mag_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_mag_rot_per_point", features),
            "loss_geo_e1_mag_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_mag_scl_per_point", features),
            "loss_geo_e1_raw_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_raw_disp_per_point", features),
            "loss_geo_e1_raw_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_raw_rot_per_point", features),
            "loss_geo_e1_raw_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 1, "loss_raw_scl_per_point", features),
            "loss_geo_e2_sat_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_sat_disp_per_point", features),
            "loss_geo_e2_sat_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_sat_rot_per_point", features),
            "loss_geo_e2_sat_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_sat_scl_per_point", features),
            "loss_geo_e2_mag_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_mag_disp_per_point", features),
            "loss_geo_e2_mag_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_mag_rot_per_point", features),
            "loss_geo_e2_mag_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_mag_scl_per_point", features),
            "loss_geo_e2_raw_disp": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_raw_disp_per_point", features),
            "loss_geo_e2_raw_rot": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_raw_rot_per_point", features),
            "loss_geo_e2_raw_scl": self._weighted_expert_loss(geo_loss_terms, pi_geo, 2, "loss_raw_scl_per_point", features),
        }

        aux = {
            "pi_geo": pi_geo,
            "pi_vis": pi_vis,
            "d_mu": d_mu,
            "d_rot": d_rot,
            "d_scale": d_scale,
            "d_opacity_logit": d_opacity_logit,
            "entropy_geo": entropy_geo,
            "entropy_vis": entropy_vis,
            "route_max_prob_geo": geo_stats["max_prob"],
            "route_margin_geo": geo_stats["margin"],
            "route_top1_geo_mean": geo_stats["top1_index_mean"],
            "route_max_prob_vis": vis_stats["max_prob"],
            "route_margin_vis": vis_stats["margin"],
            "route_top1_vis_mean": vis_stats["top1_index_mean"],
            "expert_diversity_geo": expert_diversity_geo,
            "geo_residual_gate": torch.tensor(float(geo_residual_gate), device=features.device, dtype=features.dtype),
            "vis_residual_gate": torch.tensor(float(vis_residual_gate), device=features.device, dtype=features.dtype),
            "dbg_geo_logits_mean": logits_geo.detach().mean(),
            "dbg_geo_logits_std": logits_geo.detach().std(),
            "dbg_vis_logits_mean": logits_vis.detach().mean(),
            "dbg_vis_logits_std": logits_vis.detach().std(),
            "dbg_geo_route_max_prob": geo_stats["max_prob"].detach(),
            "dbg_geo_route_margin": geo_stats["margin"].detach(),
            "dbg_vis_route_max_prob": vis_stats["max_prob"].detach(),
            "dbg_vis_route_margin": vis_stats["margin"].detach(),
            "dbg_shared_geo_disp_norm_mean": torch.norm(shared_geo[:, 0:3].detach(), dim=-1).mean(),
            "dbg_shared_geo_rot_norm_mean": torch.norm(shared_geo[:, 3:6].detach(), dim=-1).mean(),
            "dbg_shared_geo_scl_norm_mean": torch.norm(shared_geo[:, 6:9].detach(), dim=-1).mean(),
            "dbg_geo_residual_disp_norm_mean": torch.norm(geo_residual[:, 0:3].detach(), dim=-1).mean(),
            "dbg_geo_residual_rot_norm_mean": torch.norm(geo_residual[:, 3:6].detach(), dim=-1).mean(),
            "dbg_geo_residual_scl_norm_mean": torch.norm(geo_residual[:, 6:9].detach(), dim=-1).mean(),
            "dbg_scene_scale": torch.as_tensor(scene_scale).detach().float().mean(),
            "dbg_rot_branch_frozen": torch.tensor(float(self.freeze_rot_branch), device=features.device, dtype=features.dtype),
            "dbg_scale_branch_frozen": torch.tensor(float(self.freeze_scale_branch), device=features.device, dtype=features.dtype),
            "dbg_expert_diversity_geo": expert_diversity_geo.detach(),
            "dbg_shared_bounded_disp_norm_mean": shared_dbg["bounded_disp_norm_mean"],
            "dbg_shared_bounded_rot_norm_mean": shared_dbg["bounded_rot_norm_mean"],
            "dbg_shared_bounded_scl_norm_mean": shared_dbg["bounded_scl_norm_mean"],
        }
        aux.update(geo_debug)
        aux.update(geo_reg_aux)
        for key, value in geo_reg_aux.items():
            aux[f"dbg_{key}"] = value.detach()
        return means_t, scales_t, rotations_t, opacity_logits_t, aux


@torch.no_grad()
def shape_debug_check(device: torch.device = torch.device("cpu")) -> Dict[str, bool]:
    n = 128
    f_dim = 64
    model = DisentangledMoETracking(
        feature_dim=f_dim,
        geo_router_in_dim=f_dim + 3 + 1 + 3 + 1,
        vis_router_in_dim=f_dim + 3 + 1 + 1,
        geo_hidden_dim=64,
        vis_hidden_dim=64,
        k_geo=3,
        k_vis=2,
        max_disp_smooth_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_rot_smooth=0.05,
        max_rot_local=0.10,
        max_scale_smooth=0.05,
        max_scale_local=0.10,
        max_opacity_delta=4.0,
        max_disp_shared_ratio=0.01,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    ).to(device)

    features = torch.randn(n, f_dim, device=device)
    mu = torch.randn(n, 3, device=device)
    sc = torch.randn(n, 3, device=device)
    rot = torch.randn(n, 4, device=device)
    op_logit = torch.randn(n, 1, device=device)
    t = torch.rand(n, 1, device=device)
    scene_scale = torch.tensor(1.0, device=device)

    mu_t, sc_t, rot_t, op_t, aux = model(
        features,
        mu,
        sc,
        rot,
        op_logit,
        t,
        scene_scale,
        temperature_geo=1.5,
        temperature_vis=1.5,
        active_geo=3,
        active_vis=2,
        enable_visibility=True,
        use_sparse_geo=True,
        use_sparse_vis=True,
        topk_geo=2,
        topk_vis=1,
        geo_residual_gate=1.0,
        vis_residual_gate=1.0,
    )

    checks = {
        "mu_shape": mu_t.shape == (n, 3),
        "scale_shape": sc_t.shape == (n, 3),
        "rotation_shape": rot_t.shape == rot.shape,
        "opacity_shape": op_t.shape == (n, 1),
        "pi_geo_shape": aux["pi_geo"].shape == (n, 3),
        "pi_vis_shape": aux["pi_vis"].shape == (n, 2),
        "pi_geo_sum1": torch.allclose(aux["pi_geo"].sum(dim=-1), torch.ones(n, device=device), atol=1e-4),
        "pi_vis_sum1": torch.allclose(aux["pi_vis"].sum(dim=-1), torch.ones(n, device=device), atol=1e-4),
        "opacity_in_range": bool(((torch.sigmoid(op_t) >= 0.0) & (torch.sigmoid(op_t) <= 1.0)).all().item()),
        "no_nan_inf": bool(
            torch.isfinite(mu_t).all().item()
            and torch.isfinite(sc_t).all().item()
            and torch.isfinite(rot_t).all().item()
            and torch.isfinite(op_t).all().item()
        ),
    }
    return checks
