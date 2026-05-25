import importlib.util
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

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

_DEFORMATION_SPEC = importlib.util.spec_from_file_location(
    "deformation_module",
    ROOT / "scene" / "deformation.py",
)
deformation_module = importlib.util.module_from_spec(_DEFORMATION_SPEC)
assert _DEFORMATION_SPEC.loader is not None
_DEFORMATION_SPEC.loader.exec_module(deformation_module)
Deformation = deformation_module.Deformation
deform_network = deformation_module.deform_network

if "simple_knn._C" not in sys.modules:
    simple_knn_module = ModuleType("simple_knn")
    simple_knn_c_module = ModuleType("simple_knn._C")

    def _dist_cuda2_unavailable(*args, **kwargs):
        raise RuntimeError("distCUDA2 is unavailable in unit tests")

    simple_knn_c_module.distCUDA2 = _dist_cuda2_unavailable
    simple_knn_module._C = simple_knn_c_module
    sys.modules.setdefault("simple_knn", simple_knn_module)
    sys.modules["simple_knn._C"] = simple_knn_c_module

_GAUSSIAN_MODEL_SPEC = importlib.util.spec_from_file_location(
    "gaussian_model_module",
    ROOT / "scene" / "gaussian_model.py",
)
gaussian_model_module = importlib.util.module_from_spec(_GAUSSIAN_MODEL_SPEC)
assert _GAUSSIAN_MODEL_SPEC.loader is not None
gaussian_model_loader = _GAUSSIAN_MODEL_SPEC.loader
assert gaussian_model_loader is not None
gaussian_model_loader.exec_module(gaussian_model_module)
GaussianModel = gaussian_model_module.GaussianModel

PLANE_CONFIG = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [16, 16, 16, 8],
}


class _DeformationStateStub:
    def __init__(self, tracking_mode: str = "hetero_moe") -> None:
        self.tracking_mode = tracking_mode

    def get_tracking_arch_version(self) -> str:
        if self.tracking_mode == "hetero_moe":
            return "hetero_residual_v2"
        if self.tracking_mode == "split":
            return "split_v1"
        return "original_v1"


class _DeformationWrapperStub:
    def __init__(self, tracking_mode: str = "hetero_moe") -> None:
        self.deformation_net = _DeformationStateStub(tracking_mode=tracking_mode)

    def state_dict(self):
        return {"dummy": torch.tensor([1.0])}


class _GaussianSaveStub:
    def __init__(self, tracking_mode: str = "hetero_moe") -> None:
        self._deformation = _DeformationWrapperStub(tracking_mode=tracking_mode)
        self._deformation_table = torch.ones(1, dtype=torch.bool)
        self._deformation_accum = torch.zeros(1, 3)


class _DeformationLoadWrapperStub(_DeformationWrapperStub):
    def __init__(self, tracking_mode: str = "hetero_moe") -> None:
        super().__init__(tracking_mode=tracking_mode)
        self.loaded_state_dict = None
        self.loaded_device = None

    def load_state_dict(self, state_dict) -> None:
        self.loaded_state_dict = state_dict

    def to(self, device: str):
        self.loaded_device = device
        return self


class _GaussianTrainingStub:
    def __init__(self, tracking_mode: str = "hetero_moe") -> None:
        self._xyz = nn.Parameter(torch.zeros(1, 3))
        self._features_dc = nn.Parameter(torch.zeros(1, 1, 3))
        self._features_rest = nn.Parameter(torch.zeros(1, 1, 3))
        self._opacity = nn.Parameter(torch.zeros(1, 1))
        self._scaling = nn.Parameter(torch.zeros(1, 3))
        self._rotation = nn.Parameter(torch.zeros(1, 4))
        self._deformation = _DeformationLoadWrapperStub(tracking_mode=tracking_mode)
        self.optimizer = None
        self.spatial_lr_scale = 1.0
        self.percent_dense = 0.0
        self.xyz_gradient_accum = None
        self.denom = None
        self.active_sh_degree = None
        self.max_radii2D = torch.zeros(1)
        self.training_setup_calls = []
        self.optimizer_load_state_dict_calls = []

    @property
    def get_xyz(self):
        return self._xyz

    def training_setup(self, training_args) -> None:
        self.training_setup_calls.append(training_args)
        self.optimizer = SimpleNamespace(load_state_dict=lambda state: self.optimizer_load_state_dict_calls.append(state))



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


class _ResidualHeadStub(nn.Module):
    def __init__(self, delta_mu: torch.Tensor, delta_opacity: torch.Tensor) -> None:
        super().__init__()
        self.delta_mu = delta_mu
        self.delta_opacity = delta_opacity
        self.last_inputs = None

    def forward(
        self,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        **kwargs,
    ):
        self.last_inputs = {
            "means3d": means3d.clone(),
            "scales": scales.clone(),
            "rotations": rotations.clone(),
            "opacity_logits": opacity_logits.clone(),
        }
        aux = {
            "pi_geo": torch.ones(means3d.shape[0], 4, device=means3d.device, dtype=means3d.dtype) / 4.0,
            "pi_vis": torch.tensor([[1.0, 0.0]], device=means3d.device, dtype=means3d.dtype).repeat(means3d.shape[0], 1),
            "d_mu": self.delta_mu,
            "d_rot": torch.zeros(means3d.shape[0], 3, device=means3d.device, dtype=means3d.dtype),
            "d_scale": torch.zeros_like(scales),
            "d_opacity_logit": self.delta_opacity,
            "entropy_geo": torch.zeros((), device=means3d.device, dtype=means3d.dtype),
            "entropy_vis": torch.zeros((), device=means3d.device, dtype=means3d.dtype),
            "route_max_prob_geo": torch.tensor(1.0, device=means3d.device, dtype=means3d.dtype),
            "route_margin_geo": torch.tensor(1.0, device=means3d.device, dtype=means3d.dtype),
            "route_top1_geo_mean": torch.tensor(0.0, device=means3d.device, dtype=means3d.dtype),
            "route_max_prob_vis": torch.tensor(1.0, device=means3d.device, dtype=means3d.dtype),
            "route_margin_vis": torch.tensor(1.0, device=means3d.device, dtype=means3d.dtype),
            "route_top1_vis_mean": torch.tensor(0.0, device=means3d.device, dtype=means3d.dtype),
            "expert_diversity_geo": torch.zeros((), device=means3d.device, dtype=means3d.dtype),
        }
        return means3d + self.delta_mu, scales, rotations, opacity_logits + self.delta_opacity, aux


def _build_deformation_args() -> SimpleNamespace:
    scheduler_args = _build_scheduler_args()
    return SimpleNamespace(
        tracking_type="heterogeneous_moe",
        no_grid=False,
        bounds=1.6,
        kplanes_config=PLANE_CONFIG,
        multires=[1],
        geo_hidden_dim=16,
        vis_hidden_dim=16,
        timenet_output=8,
        camera_extent=1.0,
        use_soft_routing=True,
        use_topk=False,
        topk_geo=2,
        topk_vis=1,
        router_noise_geo=0.0,
        router_noise_vis=0.0,
        no_ds=False,
        no_dr=False,
        no_do=False,
        max_disp_smooth_ratio=0.01,
        max_rot_smooth=0.05,
        max_scale_smooth=0.05,
        max_opacity_delta=4.0,
        current_iteration=0,
        iterations=9000,
        temperature_geo_init=scheduler_args.temperature_geo_init,
        temperature_geo_final=scheduler_args.temperature_geo_final,
        temperature_vis_init=scheduler_args.temperature_vis_init,
        temperature_vis_final=scheduler_args.temperature_vis_final,
        enable_shared_only_iter=scheduler_args.enable_shared_only_iter,
        enable_smooth_geo_iter=scheduler_args.enable_smooth_geo_iter,
        enable_local_geo_iter=scheduler_args.enable_local_geo_iter,
        enable_visibility_iter=scheduler_args.enable_visibility_iter,
        enable_sparse_routing_iter=scheduler_args.enable_sparse_routing_iter,
        enable_route_stability_iter=scheduler_args.enable_route_stability_iter,
        enable_visibility=scheduler_args.enable_visibility,
        net_width=8,
        defor_depth=1,
        timebase_pe=1,
        timenet_width=8,
        scale_rotation_pe=0,
    )


def _build_deformation_model() -> Deformation:
    return Deformation(D=1, W=8, args=_build_deformation_args())


def _build_training_args() -> SimpleNamespace:
    return SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.1,
        position_lr_final=0.1,
        position_lr_delay_mult=1.0,
        position_lr_max_steps=9000,
        feature_lr=0.01,
        opacity_lr=0.01,
        scaling_lr=0.01,
        rotation_lr=0.01,
        deformation_lr_init=0.05,
        deformation_lr_final=0.05,
        deformation_lr_delay_mult=1.0,
        grid_lr_init=0.02,
        grid_lr_final=0.02,
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
    assert router_phase.is_group_trainable("tracking_base_deformation")
    assert router_phase.lr_scale_for_group("tracking_base_grid") == 1.0


def test_heterogeneous_tracking_groups_include_base_path_when_backbone_enabled():
    model = _build_deformation_model()
    groups = model.get_tracking_parameter_groups()

    assert "tracking_base_deformation" in groups
    assert "tracking_base_grid" in groups
    assert "tracking_geo_router" in groups


def test_heterogeneous_save_deformation_writes_arch_version(tmp_path):
    stub = _GaussianSaveStub()
    save_deformation = GaussianModel.save_deformation.__get__(stub, _GaussianSaveStub)

    save_deformation(tmp_path)

    metadata = torch.load(tmp_path / "deformation_meta.pth", map_location="cpu")
    assert metadata["tracking_type"] == "hetero_moe"
    assert metadata["tracking_arch_version"] == "hetero_residual_v2"


def test_heterogeneous_moe_zero_residual_matches_original_dynamic_output():
    model = _build_deformation_model()
    phase = _build_phase(use_sparse=False)
    model.set_tracking_phase(phase)

    def _fake_query_time(self, rays_pts_emb, time_emb):
        return torch.zeros(rays_pts_emb.shape[0], self.W, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)

    def _fake_forward_original(self, hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb):
        del hidden
        return (
            rays_pts_emb[:, :3] + 1.0,
            scales_emb[:, :3] + 2.0,
            rotations_emb[:, :4] + 3.0,
            opacity_emb[:, :1] + 4.0,
        )

    model.query_time = MethodType(_fake_query_time, model)
    model._forward_original = MethodType(_fake_forward_original, model)
    zero_mu = torch.zeros(5, 3)
    zero_opacity = torch.zeros(5, 1)
    model.heterogeneous_head = _ResidualHeadStub(zero_mu, zero_opacity)

    points = torch.randn(5, 3)
    scales = torch.randn(5, 3)
    rotations = torch.randn(5, 4)
    opacity = torch.randn(5, 1)
    times = torch.rand(5, 1)
    time_features = torch.randn(5, 8)

    pts_t, scales_t, rotations_t, opacity_t = model.forward_dynamic(
        points,
        scales,
        rotations,
        opacity,
        times,
        time_features,
    )

    assert torch.allclose(pts_t, points + 1.0)
    assert torch.allclose(scales_t, scales + 2.0)
    assert torch.allclose(rotations_t, rotations + 3.0)
    assert torch.allclose(opacity_t, opacity + 4.0)
    assert torch.allclose(model.heterogeneous_head.last_inputs["means3d"], points + 1.0)
    assert torch.allclose(model.heterogeneous_head.last_inputs["opacity_logits"], opacity + 4.0)
    assert torch.allclose(model.get_aux_outputs()["d_mu"], zero_mu)
    assert torch.allclose(model.get_latest_d_mu(), zero_mu)


def test_heterogeneous_moe_applies_residual_on_top_of_original_output():
    model = _build_deformation_model()
    phase = _build_phase(use_sparse=False)
    model.set_tracking_phase(phase)

    def _fake_query_time(self, rays_pts_emb, time_emb):
        return torch.zeros(rays_pts_emb.shape[0], self.W, device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)

    def _fake_forward_original(self, hidden, rays_pts_emb, scales_emb, rotations_emb, opacity_emb):
        del hidden
        return (
            rays_pts_emb[:, :3] - 0.5,
            scales_emb[:, :3] + 0.25,
            rotations_emb[:, :4] - 0.75,
            opacity_emb[:, :1] + 0.5,
        )

    model.query_time = MethodType(_fake_query_time, model)
    model._forward_original = MethodType(_fake_forward_original, model)
    delta_mu = torch.full((4, 3), 0.2)
    delta_opacity = torch.full((4, 1), -0.3)
    model.heterogeneous_head = _ResidualHeadStub(delta_mu, delta_opacity)

    points = torch.randn(4, 3)
    scales = torch.randn(4, 3)
    rotations = torch.randn(4, 4)
    opacity = torch.randn(4, 1)
    times = torch.rand(4, 1)
    time_features = torch.randn(4, 8)

    pts_t, scales_t, rotations_t, opacity_t = model.forward_dynamic(
        points,
        scales,
        rotations,
        opacity,
        times,
        time_features,
    )

    expected_base_pts = points - 0.5
    expected_base_scales = scales + 0.25
    expected_base_rotations = rotations - 0.75
    expected_base_opacity = opacity + 0.5

    assert torch.allclose(model.heterogeneous_head.last_inputs["means3d"], expected_base_pts)
    assert torch.allclose(model.heterogeneous_head.last_inputs["scales"], expected_base_scales)
    assert torch.allclose(model.heterogeneous_head.last_inputs["rotations"], expected_base_rotations)
    assert torch.allclose(model.heterogeneous_head.last_inputs["opacity_logits"], expected_base_opacity)
    assert torch.allclose(pts_t, expected_base_pts + delta_mu)
    assert torch.allclose(scales_t, expected_base_scales)
    assert torch.allclose(rotations_t, expected_base_rotations)
    assert torch.allclose(opacity_t, expected_base_opacity + delta_opacity)
    assert torch.allclose(model.get_aux_outputs()["d_mu"], delta_mu)
    assert torch.allclose(model.get_latest_d_mu(), delta_mu)


def test_load_model_rejects_legacy_heterogeneous_metadata(tmp_path, monkeypatch):
    torch.save({"dummy": torch.tensor([1.0])}, tmp_path / "deformation.pth")
    torch.save({"tracking_type": "hetero_moe"}, tmp_path / "deformation_meta.pth")

    stub = SimpleNamespace(
        _deformation=_DeformationLoadWrapperStub(tracking_mode="hetero_moe"),
        _deformation_table=torch.empty(0),
        _deformation_accum=torch.empty(0),
        _xyz=torch.zeros(1, 3),
        max_radii2D=torch.empty(0),
    )
    stub.get_xyz = stub._xyz

    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {"dummy": torch.tensor([1.0])} if str(path).endswith("deformation.pth") else {"tracking_type": "hetero_moe"})

    with pytest.raises(ValueError, match="predates the residual-MoE architecture update"):
        GaussianModel.load_model(stub, tmp_path)


def test_restore_accepts_current_heterogeneous_checkpoint_metadata():
    stub = _GaussianTrainingStub(tracking_mode="hetero_moe")
    training_args = SimpleNamespace(percent_dense=0.25)
    model_args = (
        0,
        torch.zeros(1, 3),
        {"dummy": torch.tensor([1.0])},
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 3),
        torch.zeros(1, 4),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        {"state": "ok"},
        0.5,
        1.0,
        {"tracking_type": "hetero_moe", "tracking_arch_version": "hetero_residual_v2"},
    )

    GaussianModel.restore(stub, model_args, training_args)

    assert stub._deformation.loaded_state_dict == {"dummy": torch.tensor([1.0])}
    assert stub.training_setup_calls == [training_args]
    assert stub.optimizer_load_state_dict_calls == [{"state": "ok"}]
    assert stub.percent_dense == 0.5


def test_restore_rejects_legacy_heterogeneous_checkpoint_without_arch_version():
    stub = _GaussianTrainingStub(tracking_mode="hetero_moe")
    training_args = SimpleNamespace(percent_dense=0.25)
    legacy_model_args = (
        0,
        torch.zeros(1, 3),
        {"dummy": torch.tensor([1.0])},
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 3),
        torch.zeros(1, 4),
        torch.zeros(1, 1),
        torch.zeros(1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        {"state": "ok"},
        0.5,
        1.0,
        {"tracking_type": "hetero_moe"},
    )

    with pytest.raises(ValueError, match="predates the residual-MoE architecture update"):
        GaussianModel.restore(stub, legacy_model_args, training_args)


def test_optimizer_phase_wiring_keeps_base_groups_live_and_stages_moe_groups():
    network = deform_network(_build_deformation_args())
    tracking_groups = network.get_tracking_parameter_groups()

    assert "tracking_base_deformation" in tracking_groups
    assert "tracking_base_grid" in tracking_groups
    assert "tracking_geo_router" in tracking_groups

    optimizer_groups = network.get_optimizer_param_groups(_build_training_args(), spatial_lr_scale=1.0)
    groups_by_name = {group["name"]: group for group in optimizer_groups}

    for name in ("tracking_base_deformation", "tracking_base_grid"):
        assert name in groups_by_name

    scheduler = HeterogeneousMoEScheduler(_build_scheduler_args())

    hexplane_phase = scheduler.build(500, 9000)
    assert hexplane_phase.is_group_trainable("tracking_base_deformation")
    assert hexplane_phase.lr_scale_for_group("tracking_base_grid") == 1.0
    assert not hexplane_phase.is_group_trainable("tracking_geo_router")
    assert not hexplane_phase.is_group_trainable("tracking_vis_router")
    assert not hexplane_phase.is_group_trainable("tracking_geo_local")
    assert not hexplane_phase.is_group_trainable("tracking_geo_smooth")

    joint_phase = scheduler.build(7000, 9000)
    assert joint_phase.is_group_trainable("tracking_geo_router")
    assert joint_phase.is_group_trainable("tracking_vis_router")
    assert joint_phase.is_group_trainable("tracking_geo_local")
    assert joint_phase.is_group_trainable("tracking_geo_smooth")
    assert joint_phase.lr_scale_for_group("tracking_geo_hexplane_grid") == 0.1
    assert joint_phase.lr_scale_for_group("tracking_vis_transient") == 0.1


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
