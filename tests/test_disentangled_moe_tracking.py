import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.tracking.disentangled_moe_tracking import DisentangledMoETracking, shape_debug_check


def test_disentangled_moe_shape_debug():
    checks = shape_debug_check(device=torch.device("cpu"))
    assert all(checks.values()), checks


def test_disentangled_branch_outputs_are_decoupled():
    n = 16
    f_dim = 8
    model = DisentangledMoETracking(
        feature_dim=f_dim,
        geo_router_in_dim=f_dim + 3 + 1 + 3 + 1,
        vis_router_in_dim=f_dim + 3 + 1 + 1,
        geo_hidden_dim=16,
        vis_hidden_dim=16,
        k_geo=3,
        k_vis=2,
        max_disp_smooth_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_rot_smooth=0.05,
        max_rot_local=0.1,
        max_scale_smooth=0.05,
        max_scale_local=0.1,
        max_opacity_delta=4.0,
        max_disp_shared_ratio=0.01,
        max_rot_shared=0.05,
        max_scale_shared=0.05,
        use_soft_routing=True,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    )
    features = torch.randn(n, f_dim)
    mu = torch.randn(n, 3)
    scale = torch.randn(n, 3)
    rot = torch.randn(n, 4)
    op = torch.randn(n, 1)
    t = torch.rand(n, 1)

    mu_t, scale_t, rot_t, op_t, aux = model(
        features,
        mu,
        scale,
        rot,
        op,
        t,
        torch.tensor(1.0),
        temperature_geo=1.2,
        temperature_vis=1.2,
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

    assert aux["d_mu"].shape == (n, 3)
    assert aux["d_opacity_logit"].shape == (n, 1)
    assert aux["pi_geo"].shape == (n, 3)
    assert aux["pi_vis"].shape == (n, 2)
    assert "expert_diversity_geo" in aux
    assert "route_max_prob_geo" in aux
    assert mu_t.shape == mu.shape
    assert scale_t.shape == scale.shape
    assert rot_t.shape == rot.shape
    assert op_t.shape == op.shape


def test_topk_visibility_router_is_sparse():
    n = 32
    f_dim = 8
    model = DisentangledMoETracking(
        feature_dim=f_dim,
        geo_router_in_dim=f_dim + 3 + 1 + 3 + 1,
        vis_router_in_dim=f_dim + 3 + 1 + 1,
        geo_hidden_dim=16,
        vis_hidden_dim=16,
        k_geo=3,
        k_vis=2,
        max_disp_smooth_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_rot_smooth=0.05,
        max_rot_local=0.1,
        max_scale_smooth=0.05,
        max_scale_local=0.1,
        max_opacity_delta=4.0,
        use_topk=True,
        topk_geo=2,
        topk_vis=1,
    )

    outputs = model(
        torch.randn(n, f_dim),
        torch.randn(n, 3),
        torch.randn(n, 3),
        torch.randn(n, 4),
        torch.randn(n, 1),
        torch.rand(n, 1),
        torch.tensor(1.0),
        temperature_geo=1.0,
        temperature_vis=1.0,
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
    aux = outputs[-1]

    nonzero_vis = (aux["pi_vis"] > 1e-6).sum(dim=-1)
    assert torch.all(nonzero_vis == 1)


def test_invalid_k_values_raise():
    try:
        DisentangledMoETracking(
            feature_dim=8,
            geo_router_in_dim=8 + 3 + 1 + 3 + 1,
            vis_router_in_dim=8 + 3 + 1 + 1,
            geo_hidden_dim=16,
            vis_hidden_dim=16,
            k_geo=4,
            k_vis=2,
            max_disp_smooth_ratio=0.01,
            max_disp_local_ratio=0.03,
            max_rot_smooth=0.05,
            max_rot_local=0.1,
            max_scale_smooth=0.05,
            max_scale_local=0.1,
            max_opacity_delta=4.0,
        )
        assert False, "Expected ValueError for unsupported K_geo"
    except ValueError:
        assert True
