import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.tracking import (
    DisentangledMoETracking,
    HeterogeneousMoEScheduler,
    TrackingPhase,
    shape_debug_check,
)

_TRACKING_LOSSES_SPEC = importlib.util.spec_from_file_location(
    "tracking_losses_module",
    ROOT / "scene" / "tracking_losses.py",
)
tracking_losses_module = importlib.util.module_from_spec(_TRACKING_LOSSES_SPEC)
assert _TRACKING_LOSSES_SPEC.loader is not None
_TRACKING_LOSSES_SPEC.loader.exec_module(tracking_losses_module)
compute_tracking_losses = tracking_losses_module.compute_tracking_losses


PLANE_CONFIG = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [16, 16, 16, 8],
}


def _build_model() -> DisentangledMoETracking:
    return DisentangledMoETracking(
        time_feature_dim=8,
        geo_hidden_dim=16,
        vis_hidden_dim=16,
        bounds=1.6,
        planeconfig=PLANE_CONFIG,
        multires=[1],
        max_disp_hexplane_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_disp_smooth_ratio=0.005,
        max_opacity_delta=4.0,
        use_soft_routing=True,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    )


def _build_phase(use_sparse: bool = True) -> TrackingPhase:
    return TrackingPhase(
        name="joint_finetune",
        active_geo=4,
        active_vis=2,
        enable_visibility=True,
        temperature_geo=1.0,
        temperature_vis=1.0,
        use_sparse_geo=use_sparse,
        use_sparse_vis=use_sparse,
        topk_geo=2,
        topk_vis=1,
        trainable_group_prefixes=("tracking_",),
    )


def _build_scheduler_args() -> SimpleNamespace:
    return SimpleNamespace(
        temperature_geo_init=2.0,
        temperature_geo_final=0.7,
        temperature_vis_init=2.0,
        temperature_vis_final=1.0,
        enable_shared_only_iter=1000,
        enable_smooth_geo_iter=2000,
        enable_local_geo_iter=3500,
        enable_visibility_iter=4500,
        enable_sparse_routing_iter=5000,
        enable_route_stability_iter=5000,
        enable_visibility=True,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    )


def _build_loss_args() -> SimpleNamespace:
    return SimpleNamespace(
        lambda_balance_geo=0.0,
        lambda_balance_vis=0.0,
        lambda_route_conf_geo=0.0,
        lambda_route_conf_vis=0.0,
        lambda_expert_diversity_geo=0.0,
        lambda_entropy_geo=0.0,
        lambda_entropy_vis=0.0,
        entropy_end_iter=0,
        lambda_geo_temp=1.0,
        lambda_vis_sparse=0.0,
        enable_decouple_iter=999999,
        lambda_decouple=0.0,
        target_geo_static=0.30,
        target_geo_hexplane=0.35,
        target_geo_local=0.20,
        target_geo_residual_smooth=0.15,
        target_geo_static_stage2=0.40,
        target_geo_hexplane_stage2=0.60,
        target_vis_stable=0.85,
        target_vis_transient=0.15,
        lambda_mag_g1_mu=0.0,
        lambda_mag_g2_mu=0.0,
        lambda_mag_g3_mu=0.0,
        lambda_sat_g1_disp=0.0,
        lambda_sat_g2_disp=0.0,
        lambda_sat_g3_disp=0.0,
        lambda_raw_g1_disp=0.0,
        lambda_raw_g2_disp=0.0,
        lambda_raw_g3_disp=0.0,
    )


def test_scheduler_respects_legacy_stage_knobs_and_trains_time_encoder():
    scheduler = HeterogeneousMoEScheduler(_build_scheduler_args())

    hexplane_phase = scheduler.build(500, 9000)
    assert hexplane_phase.name == "hexplane_only"
    assert hexplane_phase.force_geo_expert == "hexplane"
    assert hexplane_phase.active_geo == 1
    assert hexplane_phase.is_group_trainable("tracking_time_encoder")

    smooth_phase = scheduler.build(1500, 9000)
    assert smooth_phase.name == "smooth_only"
    assert smooth_phase.force_geo_expert == "smooth"
    assert smooth_phase.active_geo == 1
    assert smooth_phase.is_group_trainable("tracking_time_encoder")

    local_phase = scheduler.build(2500, 9000)
    assert local_phase.name == "local_only"
    assert local_phase.force_geo_expert == "local"
    assert local_phase.active_geo == 1
    assert local_phase.is_group_trainable("tracking_time_encoder")

    router_phase = scheduler.build(4700, 9000)
    assert router_phase.name == "router_only"
    assert router_phase.force_geo_expert is None
    assert router_phase.active_geo == 4
    assert router_phase.enable_visibility
    assert router_phase.is_group_trainable("tracking_time_encoder")


def test_disentangled_moe_shape_debug():
    checks = shape_debug_check(device=torch.device("cpu"))
    assert all(checks.values()), checks


def test_disentangled_branch_outputs_are_decoupled():
    n = 16
    model = _build_model()
    phase = _build_phase(use_sparse=True)

    mu_t, scale_t, rot_t, op_t, aux = model(
        means3d=torch.randn(n, 3),
        scales=torch.randn(n, 3),
        rotations=torch.randn(n, 4),
        opacity_logits=torch.randn(n, 1),
        time_values=torch.rand(n, 1),
        time_features=torch.randn(n, 8),
        scene_scale=torch.tensor(1.0),
        phase=phase,
    )

    assert aux["d_mu"].shape == (n, 3)
    assert aux["d_opacity_logit"].shape == (n, 1)
    assert aux["pi_geo"].shape == (n, 4)
    assert aux["pi_vis"].shape == (n, 2)
    assert "expert_diversity_geo" in aux
    assert "route_max_prob_geo" in aux
    assert "geo_usage_hexplane" in aux
    assert "geo_usage_local" in aux
    assert "geo_usage_smooth" in aux
    assert mu_t.shape == (n, 3)
    assert scale_t.shape == (n, 3)
    assert rot_t.shape == (n, 4)
    assert op_t.shape == (n, 1)


def test_topk_visibility_router_is_sparse():
    n = 32
    model = _build_model()
    phase = _build_phase(use_sparse=True)

    outputs = model(
        means3d=torch.randn(n, 3),
        scales=torch.randn(n, 3),
        rotations=torch.randn(n, 4),
        opacity_logits=torch.randn(n, 1),
        time_values=torch.rand(n, 1),
        time_features=torch.randn(n, 8),
        scene_scale=torch.tensor(1.0),
        phase=phase,
    )
    aux = outputs[-1]

    nonzero_vis = (aux["pi_vis"] > 1e-6).sum(dim=-1)
    assert torch.all(nonzero_vis == 1)


def test_tracking_losses_use_adjacent_time_sequence_not_prev_step_state():
    args = _build_loss_args()
    aux = {
        "pi_geo": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "pi_vis": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "d_mu": torch.zeros(2, 3),
        "d_mu_sequence": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
            ]
        ),
        "time_sequence": torch.tensor([0.0, 0.5]),
        "entropy_geo": torch.tensor(0.0),
        "entropy_vis": torch.tensor(0.0),
    }

    losses = compute_tracking_losses(
        aux=aux,
        iteration=10,
        args=args,
        prev_d_mu=torch.full((2, 3), 99.0),
        active_geo=4,
        active_vis=2,
        enable_visibility=True,
        geo_expert_names=("static", "hexplane", "local", "smooth"),
        vis_expert_names=("stable", "transient"),
    )

    expected = torch.tensor((0.2 / 0.5) ** 2 / 3.0)
    assert torch.allclose(losses["L_geo_temp"], expected, atol=1e-6)
    assert torch.equal(losses["temporal_pair_count"], torch.tensor(1.0))
