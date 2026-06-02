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
    CAMSGSScheduler,
    CAMSGSTracking,
    DisentangledMoETracking,
    HeterogeneousMoEScheduler,
    TrackingPhase,
    shape_debug_check,
)
from models.tracking.cut_graph_gating import CutGraphGating
from models.tracking.motion_decomposition import MotionDecomposition

from utils.device_utils import _CPUEventStub, get_device, get_device_str, safe_cuda_event

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


def test_device_utils_respects_environment_variable(monkeypatch):
    monkeypatch.setenv("ENDOGAUSSIAN_DEVICE", "cpu")
    assert get_device().type == "cpu"
    assert get_device_str() == "cpu"


def test_device_utils_force_cpu_overrides_cuda(monkeypatch):
    monkeypatch.delenv("ENDOGAUSSIAN_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device(force_cpu=True).type == "cpu"
    assert get_device_str(force_cpu=True) == "cpu"


def test_safe_cuda_event_returns_cpu_stub_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    event_a = safe_cuda_event(enable_timing=True)
    event_b = safe_cuda_event(enable_timing=True)
    assert isinstance(event_a, _CPUEventStub)
    event_a.record()
    event_b.record()
    assert event_a.elapsed_time(event_b) >= 0.0


def _quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternions_equivalent(lhs: torch.Tensor, rhs: torch.Tensor, atol: float = 1e-6) -> bool:
    lhs_norm = torch.nn.functional.normalize(lhs, dim=-1)
    rhs_norm = torch.nn.functional.normalize(rhs, dim=-1)
    alignment = torch.abs((lhs_norm * rhs_norm).sum(dim=-1))
    return bool(torch.allclose(alignment, torch.ones_like(alignment), atol=atol))


def _load_gaussian_renderer_module(monkeypatch, rasterizer_cls):
    raster_module = ModuleType("diff_gaussian_rasterization")

    class _RasterSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    scene_module = ModuleType("scene")
    scene_gaussian_model_module = ModuleType("scene.gaussian_model")
    scene_gaussian_model_module.GaussianModel = object
    scene_module.gaussian_model = scene_gaussian_model_module
    raster_module.GaussianRasterizationSettings = _RasterSettings
    raster_module.GaussianRasterizer = rasterizer_cls
    monkeypatch.setitem(sys.modules, "scene", scene_module)
    monkeypatch.setitem(sys.modules, "scene.gaussian_model", scene_gaussian_model_module)
    monkeypatch.setitem(sys.modules, "diff_gaussian_rasterization", raster_module)

    spec = importlib.util.spec_from_file_location(
        "gaussian_renderer_test_module",
        ROOT / "gaussian_renderer" / "__init__.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    original_zeros_like = torch.zeros_like
    monkeypatch.setattr(
        module.torch,
        "zeros_like",
        lambda input, **kwargs: original_zeros_like(
            input,
            dtype=kwargs.get("dtype", input.dtype),
            requires_grad=kwargs.get("requires_grad", False),
        ),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)
    return module


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
        if self.tracking_mode == "cams_gs":
            return "cams_gs_v2"
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


def test_deformation_accepts_tracking_type_cams_gs():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"

    model = Deformation(D=1, W=8, args=args)

    assert model.tracking_mode == "cams_gs"
    assert model.scheduler is not None
    assert model.cams_head is not None


def test_cams_gs_tracking_groups_expose_patch_c_optimizer_groups():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    model = Deformation(D=1, W=8, args=args)
    groups = model.get_tracking_parameter_groups()

    expected_present = {
        "tracking_base_deformation",
        "tracking_base_grid",
        "tracking_motion_global",
        "tracking_motion_local",
        "tracking_cut_graph",
        "tracking_visibility",
        "tracking_appearance",
        "tracking_lifecycle",
    }
    assert expected_present.issubset(groups.keys())


def test_cams_gs_optimizer_groups_keep_base_and_time_paths_live():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    network = deform_network(args)

    optimizer_groups = network.get_optimizer_param_groups(_build_training_args(), spatial_lr_scale=1.0)
    group_names = {group["name"] for group in optimizer_groups}

    assert {
        "tracking_base_deformation",
        "tracking_base_grid",
        "tracking_time_encoder",
        "tracking_motion_global",
        "tracking_motion_local",
        "tracking_cut_graph",
        "tracking_visibility",
        "tracking_appearance",
        "tracking_lifecycle",
    }.issubset(group_names)

    groups_by_name = {group["name"]: group for group in optimizer_groups}
    for name in ("tracking_motion_global", "tracking_motion_local", "tracking_cut_graph"):
        params = groups_by_name[name]["params"]
        assert params
        assert all(isinstance(param, nn.Parameter) for param in params)
        assert all(param.requires_grad for param in params)


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


def test_cams_gs_save_deformation_writes_arch_version(tmp_path):
    stub = _GaussianSaveStub(tracking_mode="cams_gs")
    save_deformation = GaussianModel.save_deformation.__get__(stub, _GaussianSaveStub)

    save_deformation(tmp_path)

    metadata = torch.load(tmp_path / "deformation_meta.pth", map_location="cpu")
    assert metadata["tracking_type"] == "cams_gs"
    assert metadata["tracking_arch_version"] == "cams_gs_v2"


def test_restore_accepts_current_cams_gs_checkpoint_metadata():
    stub = _GaussianTrainingStub(tracking_mode="cams_gs")
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
        {"tracking_type": "cams_gs", "tracking_arch_version": "cams_gs_v2"},
    )

    GaussianModel.restore(stub, model_args, training_args)

    assert stub._deformation.loaded_state_dict == {"dummy": torch.tensor([1.0])}
    assert stub.training_setup_calls == [training_args]
    assert stub.optimizer_load_state_dict_calls == [{"state": "ok"}]


def test_restore_rejects_legacy_cams_gs_checkpoint_without_arch_version():
    stub = _GaussianTrainingStub(tracking_mode="cams_gs")
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
        {"tracking_type": "cams_gs"},
    )

    with pytest.raises(ValueError, match="predates the CAMS-GS architecture update"):
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


def test_cams_gs_scheduler_builds_patch_c_phase_gating():
    scheduler = CAMSGSScheduler(_build_scheduler_args())

    early_phase = scheduler.build(500, 9000)
    assert early_phase.name == "global_only"
    assert early_phase.is_group_trainable("tracking_base_deformation")
    assert early_phase.is_group_trainable("tracking_motion_global")
    assert not early_phase.is_group_trainable("tracking_motion_local")
    assert not early_phase.enable_visibility
    assert early_phase.lr_scale_for_group("tracking_base_grid") == 1.0

    local_phase = scheduler.build(3000, 9000)
    assert local_phase.name == "local_motion_only"
    assert local_phase.is_group_trainable("tracking_motion_local")
    assert local_phase.is_group_trainable("tracking_cut_graph")
    assert local_phase.is_group_trainable("tracking_motion_global")
    assert not local_phase.is_group_trainable("tracking_visibility")
    assert not local_phase.enable_visibility
    assert local_phase.lr_scale_for_group("tracking_motion_global") == 0.1

    warmup_phase = scheduler.build(5000, 9000)
    assert warmup_phase.name == "motion_warmup"
    assert warmup_phase.is_group_trainable("tracking_motion_local")
    assert warmup_phase.is_group_trainable("tracking_cut_graph")
    assert not warmup_phase.is_group_trainable("tracking_visibility")
    assert not warmup_phase.enable_visibility
    assert warmup_phase.lr_scale_for_group("tracking_motion_global") == 0.25

    visibility_phase = scheduler.build(7000, 9000)
    assert visibility_phase.name == "visibility_refine"
    assert visibility_phase.enable_visibility
    assert visibility_phase.is_group_trainable("tracking_visibility")
    assert visibility_phase.is_group_trainable("tracking_appearance")
    assert not visibility_phase.is_group_trainable("tracking_lifecycle")

    late_phase = scheduler.build(8000, 9000)
    assert late_phase.name == "joint_finetune"
    assert late_phase.enable_visibility
    assert late_phase.is_group_trainable("tracking_motion_local")
    assert late_phase.is_group_trainable("tracking_cut_graph")
    assert late_phase.is_group_trainable("tracking_visibility")
    assert late_phase.is_group_trainable("tracking_appearance")
    assert late_phase.is_group_trainable("tracking_lifecycle")


def test_cams_cut_graph_gating_depends_on_spatial_position():
    gating = CutGraphGating(time_feature_dim=8)
    gating.set_aabb(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([-1.0, -1.0, -1.0]),
    )

    means_a = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)
    means_b = torch.tensor([[0.75, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    time_values = torch.zeros(2, 1)
    time_features = torch.zeros(2, 8)
    phase = TrackingPhase(
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
    )

    out_a = gating(means_a, time_values, time_features, phase)
    out_b = gating(means_b, time_values, time_features, phase)

    assert not torch.allclose(out_a["scaffold_logits"], out_b["scaffold_logits"])
    assert not torch.allclose(out_a["cut_gate_logits"], out_b["cut_gate_logits"])


def test_cams_local_motion_depends_on_spatial_position():
    motion = MotionDecomposition(time_feature_dim=8)
    motion.set_aabb(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([-1.0, -1.0, -1.0]),
    )

    means_a = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=torch.float32)
    means_b = torch.tensor([[0.75, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    opacity = torch.zeros(2, 1)
    time_features = torch.zeros(2, 8)
    gating_state = {
        "scaffold_weights": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "cut_gate_values": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "global_mix": torch.zeros(2, 1),
        "local_mix": torch.ones(2, 1),
        "cut_graph_mix": torch.zeros(2, 1),
    }
    phase = TrackingPhase(
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
    )

    out_a = motion(means_a, scales, rotations, opacity, time_features, torch.tensor(1.0), gating_state, phase)
    out_b = motion(means_b, scales, rotations, opacity, time_features, torch.tensor(1.0), gating_state, phase)

    assert not torch.allclose(out_a["local_motion"], out_b["local_motion"])
    assert not torch.allclose(out_a["d_mu"], out_b["d_mu"])


def test_motion_decomposition_global_only_masks_non_global_mixes():
    motion = MotionDecomposition(time_feature_dim=8)
    means = torch.zeros(2, 3)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    opacity = torch.zeros(2, 1)
    time_features = torch.randn(2, 8)
    gating_state = {
        "scaffold_weights": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "cut_gate_values": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        "global_mix": torch.full((2, 1), 0.2),
        "local_mix": torch.full((2, 1), 0.3),
        "cut_graph_mix": torch.full((2, 1), 0.5),
    }
    phase = TrackingPhase(
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
    )

    outputs = motion(means, scales, rotations, opacity, time_features, torch.tensor(1.0), gating_state, phase)

    assert torch.allclose(outputs["local_motion"], torch.zeros_like(outputs["local_motion"]), atol=1e-6)
    assert torch.allclose(outputs["cut_graph_motion"], torch.zeros_like(outputs["cut_graph_motion"]), atol=1e-6)
    assert torch.allclose(outputs["d_mu"], outputs["global_motion"], atol=1e-6)


def test_cams_cut_graph_route_changes_rendered_geometry():
    model = CAMSGSTracking(time_feature_dim=8)
    phase = TrackingPhase(
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
    )

    def _fake_cut_graph_local(*args, **kwargs):
        means3d = kwargs["means3d"]
        count = means3d.shape[0]
        scaffold_logits = torch.full((count, 3), -20.0)
        scaffold_logits[:, 1] = 20.0
        cut_gate_logits = torch.full((count, 3), 20.0)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means3d),
        }

    def _fake_cut_graph_cut(*args, **kwargs):
        means3d = kwargs["means3d"]
        count = means3d.shape[0]
        scaffold_logits = torch.full((count, 3), -20.0)
        scaffold_logits[:, 2] = 20.0
        cut_gate_logits = torch.full((count, 3), -20.0)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means3d),
        }

    means3d = torch.zeros(2, 3)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacity = torch.zeros(2, 1)
    time_values = torch.zeros(2, 1)
    time_features = torch.randn(2, 8)
    scene_scale = torch.tensor(1.0)

    model.cut_graph.forward = _fake_cut_graph_local
    local_outputs = model(means3d, scales, rotations, opacity, time_values, time_features, scene_scale, phase)
    local_pts = local_outputs[0]
    local_aux = local_outputs[-1]

    model.cut_graph.forward = _fake_cut_graph_cut
    cut_outputs = model(means3d, scales, rotations, opacity, time_values, time_features, scene_scale, phase)
    cut_pts = cut_outputs[0]
    cut_aux = cut_outputs[-1]

    assert torch.allclose(local_aux["pi_geo"], torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-4)
    assert torch.allclose(cut_aux["pi_geo"], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), atol=1e-4)
    assert not torch.allclose(local_pts, cut_pts)




def test_cams_global_only_phase_disables_local_and_cut_graph_contributions():
    model = CAMSGSTracking(time_feature_dim=8)
    phase = TrackingPhase(
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
    )

    means3d = torch.zeros(2, 3)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacity = torch.zeros(2, 1)
    time_values = torch.zeros(2, 1)
    time_features = torch.randn(2, 8)
    scene_scale = torch.tensor(1.0)

    def _fake_cut_graph(*args, **kwargs):
        means = kwargs["means3d"]
        scaffold_logits = torch.zeros(means.shape[0], 3)
        cut_gate_logits = torch.zeros(means.shape[0], 3)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means),
        }

    model.cut_graph.forward = _fake_cut_graph

    def _fake_motion(*args, **kwargs):
        means = kwargs["means3d"]
        gating_state = kwargs["gating_state"]
        global_delta = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=means.dtype)
        local_delta = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=means.dtype)
        cut_delta = torch.tensor([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=means.dtype)
        global_mix = gating_state["global_mix"]
        local_mix = gating_state["local_mix"]
        cut_mix = gating_state["cut_graph_mix"]
        blended_global = global_delta * global_mix
        blended_local = local_delta * local_mix
        blended_cut = cut_delta * cut_mix
        d_mu = blended_global + blended_local + blended_cut
        return {
            "means3d": means + d_mu,
            "scales": kwargs["scales"],
            "rotations": kwargs["rotations"],
            "opacity_logits": kwargs["opacity_logits"],
            "d_mu": d_mu,
            "d_rot": torch.zeros((means.shape[0], 3), dtype=means.dtype),
            "d_scale": torch.zeros_like(kwargs["scales"]),
            "d_opacity_logit": torch.zeros_like(kwargs["opacity_logits"]),
            "global_motion": blended_global,
            "local_motion": blended_local,
            "cut_graph_motion": blended_cut,
            "geo_expert_d_mu": torch.stack((global_delta, local_delta, cut_delta), dim=1),
            "geo_expert_means3d": torch.stack((means + global_delta, means + local_delta, means + cut_delta), dim=1),
            "geo_expert_scales": kwargs["scales"].unsqueeze(1).expand(-1, 3, -1),
            "geo_expert_rotations": kwargs["rotations"].unsqueeze(1).expand(-1, 3, -1),
            "geo_expert_opacity_logits": kwargs["opacity_logits"].unsqueeze(1).expand(-1, 3, -1),
        }

    model.motion.forward = _fake_motion

    pts, _, _, _, aux = model(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=time_values,
        time_features=time_features,
        scene_scale=scene_scale,
        phase=phase,
    )

    assert torch.allclose(aux["pi_geo"], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)
    assert torch.allclose(aux["d_mu"], torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)
    assert torch.allclose(aux["local_motion"], torch.zeros_like(aux["local_motion"]), atol=1e-5)
    assert torch.allclose(aux["cut_graph_motion"], torch.zeros_like(aux["cut_graph_motion"]), atol=1e-5)
    assert torch.allclose(pts, means3d + torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), atol=1e-5)


def test_cams_open_phase_allows_non_global_experts_to_change_output():
    model = CAMSGSTracking(time_feature_dim=8)
    phase = TrackingPhase(
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
    )

    means3d = torch.zeros(2, 3)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacity = torch.zeros(2, 1)
    time_values = torch.zeros(2, 1)
    time_features = torch.randn(2, 8)
    scene_scale = torch.tensor(1.0)

    def _fake_motion(*args, **kwargs):
        means = kwargs["means3d"]
        gating_state = kwargs["gating_state"]
        global_delta = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=means.dtype)
        local_delta = torch.tensor([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=means.dtype)
        cut_delta = torch.tensor([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=means.dtype)
        global_mix = gating_state["global_mix"]
        local_mix = gating_state["local_mix"]
        cut_mix = gating_state["cut_graph_mix"]
        blended_global = global_delta * global_mix
        blended_local = local_delta * local_mix
        blended_cut = cut_delta * cut_mix
        d_mu = blended_global + blended_local + blended_cut
        return {
            "means3d": means + d_mu,
            "scales": kwargs["scales"],
            "rotations": kwargs["rotations"],
            "opacity_logits": kwargs["opacity_logits"],
            "d_mu": d_mu,
            "d_rot": torch.zeros((means.shape[0], 3), dtype=means.dtype),
            "d_scale": torch.zeros_like(kwargs["scales"]),
            "d_opacity_logit": torch.zeros_like(kwargs["opacity_logits"]),
            "global_motion": blended_global,
            "local_motion": blended_local,
            "cut_graph_motion": blended_cut,
            "geo_expert_d_mu": torch.stack((global_delta, local_delta, cut_delta), dim=1),
            "geo_expert_means3d": torch.stack((means + global_delta, means + local_delta, means + cut_delta), dim=1),
            "geo_expert_scales": kwargs["scales"].unsqueeze(1).expand(-1, 3, -1),
            "geo_expert_rotations": kwargs["rotations"].unsqueeze(1).expand(-1, 3, -1),
            "geo_expert_opacity_logits": kwargs["opacity_logits"].unsqueeze(1).expand(-1, 3, -1),
        }

    model.motion.forward = _fake_motion

    def _fake_local(*args, **kwargs):
        means = kwargs["means3d"]
        scaffold_logits = torch.full((means.shape[0], 3), -20.0)
        scaffold_logits[:, 1] = 20.0
        cut_gate_logits = torch.full((means.shape[0], 3), 20.0)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means),
        }

    def _fake_cut(*args, **kwargs):
        means = kwargs["means3d"]
        scaffold_logits = torch.full((means.shape[0], 3), -20.0)
        scaffold_logits[:, 2] = 20.0
        cut_gate_logits = torch.full((means.shape[0], 3), -20.0)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means),
        }

    model.cut_graph.forward = _fake_local
    local_pts, _, _, _, local_aux = model(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=time_values,
        time_features=time_features,
        scene_scale=scene_scale,
        phase=phase,
    )

    model.cut_graph.forward = _fake_cut
    cut_pts, _, _, _, cut_aux = model(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=time_values,
        time_features=time_features,
        scene_scale=scene_scale,
        phase=phase,
    )

    assert torch.allclose(local_aux["pi_geo"], torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), atol=1e-4)
    assert torch.allclose(cut_aux["pi_geo"], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]), atol=1e-4)
    assert not torch.allclose(local_pts, cut_pts)


def test_cams_gs_emits_expert_proposals_and_gaussian_priors_for_pixel_router():
    model = CAMSGSTracking(time_feature_dim=8)
    phase = TrackingPhase(
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
    )
    n = 5
    means3d = torch.randn(n, 3)
    scales = torch.randn(n, 3)
    rotations = torch.randn(n, 4)
    rotations[:, 0] = 1.0
    opacity = torch.randn(n, 1)
    time_values = torch.rand(n, 1)
    time_features = torch.randn(n, 8)
    scene_scale = torch.tensor(1.0)

    _, _, _, _, aux = model(
        means3d=means3d,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=time_values,
        time_features=time_features,
        scene_scale=scene_scale,
        phase=phase,
    )

    assert torch.allclose(aux["gaussian_pi_geo_prior"], aux["pi_geo"])
    assert torch.allclose(aux["gaussian_pi_vis_prior"], aux["pi_vis"])
    assert aux["geo_expert_means3d"].shape == (n, 3, 3)
    assert aux["geo_expert_scales"].shape == (n, 3, 3)
    assert aux["geo_expert_rotations"].shape == (n, 3, 4)
    assert aux["geo_expert_opacity_logits"].shape == (n, 3, 1)
    assert aux["geo_expert_d_mu"].shape == (n, 3, 3)
    assert aux["vis_expert_rgb_delta"].shape == (n, 2, 3)
    assert aux["vis_expert_visibility_alpha"].shape == (n, 2, 1)
    assert aux["lifecycle_expert_alpha"].shape == (n, 2, 1)
    assert torch.allclose(aux["lifecycle_expert_alpha"].sum(dim=1), torch.ones(n, 1), atol=1e-5)


def test_cams_visibility_and_lifecycle_change_opacity_outputs():
    model = CAMSGSTracking(time_feature_dim=8)
    phase = TrackingPhase(
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
    )
    means3d = torch.zeros(2, 3)
    scales = torch.zeros(2, 3)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    opacity = torch.zeros(2, 1)
    time_values = torch.zeros(2, 1)
    time_features = torch.randn(2, 8)
    scene_scale = torch.tensor(1.0)

    def _base_cut_graph(*args, **kwargs):
        means = kwargs["means3d"]
        scaffold_logits = torch.zeros(means.shape[0], 3)
        cut_gate_logits = torch.zeros(means.shape[0], 3)
        return {
            "scaffold_logits": scaffold_logits,
            "scaffold_weights": torch.softmax(scaffold_logits, dim=-1),
            "cut_gate_logits": cut_gate_logits,
            "cut_gate_values": torch.sigmoid(cut_gate_logits),
            "xyz_norm": torch.zeros_like(means),
        }

    model.cut_graph.forward = _base_cut_graph

    def _visibility_open(*args, **kwargs):
        count = kwargs["time_features"].shape[0]
        zeros_rgb = torch.zeros(count, 3)
        return {
            "visibility_logits": torch.tensor([[12.0, -12.0]]).repeat(count, 1),
            "appearance_offsets": zeros_rgb,
            "appearance_rgb_delta": zeros_rgb,
            "pi_vis": torch.tensor([[1.0, 0.0]]).repeat(count, 1),
            "visibility_alpha": torch.ones(count, 1),
            "entropy_vis": torch.zeros(()),
            "route_max_prob_vis": torch.ones(count),
            "route_margin_vis": torch.ones(count),
            "route_top1_vis_mean": torch.tensor(1.0),
            "vis_expert_rgb_delta": torch.stack((zeros_rgb, zeros_rgb), dim=1),
            "vis_expert_visibility_alpha": torch.tensor([[[1.0], [0.0]]]).repeat(count, 1, 1),
        }

    def _visibility_closed(*args, **kwargs):
        count = kwargs["time_features"].shape[0]
        zeros_rgb = torch.zeros(count, 3)
        return {
            "visibility_logits": torch.tensor([[-12.0, 12.0]]).repeat(count, 1),
            "appearance_offsets": zeros_rgb,
            "appearance_rgb_delta": zeros_rgb,
            "pi_vis": torch.tensor([[0.0, 1.0]]).repeat(count, 1),
            "visibility_alpha": torch.zeros(count, 1),
            "entropy_vis": torch.zeros(()),
            "route_max_prob_vis": torch.ones(count),
            "route_margin_vis": torch.ones(count),
            "route_top1_vis_mean": torch.tensor(1.0),
            "vis_expert_rgb_delta": torch.stack((zeros_rgb, zeros_rgb), dim=1),
            "vis_expert_visibility_alpha": torch.tensor([[[1.0], [0.0]]]).repeat(count, 1, 1),
        }

    def _lifecycle_alive(*args, **kwargs):
        count = kwargs["time_features"].shape[0]
        probs = torch.tensor([[1.0, 0.0]]).repeat(count, 1)
        return {
            "lifecycle_logits": torch.tensor([[12.0, -12.0]]).repeat(count, 1),
            "lifecycle_probs": probs,
            "lifecycle_alpha": torch.ones(count, 1),
            "lifecycle_expert_alpha": probs.unsqueeze(-1),
        }

    def _lifecycle_dead(*args, **kwargs):
        count = kwargs["time_features"].shape[0]
        probs = torch.tensor([[0.0, 1.0]]).repeat(count, 1)
        return {
            "lifecycle_logits": torch.tensor([[-12.0, 12.0]]).repeat(count, 1),
            "lifecycle_probs": probs,
            "lifecycle_alpha": torch.zeros(count, 1),
            "lifecycle_expert_alpha": probs.unsqueeze(-1),
        }

    model.visibility.forward = _visibility_open
    model.lifecycle.forward = _lifecycle_alive
    open_outputs = model(means3d, scales, rotations, opacity, time_values, time_features, scene_scale, phase)

    model.visibility.forward = _visibility_closed
    model.lifecycle.forward = _lifecycle_dead
    closed_outputs = model(means3d, scales, rotations, opacity, time_values, time_features, scene_scale, phase)

    assert not torch.allclose(open_outputs[3], closed_outputs[3])


def test_motion_decomposition_zero_rotation_delta_preserves_orientation():
    motion = MotionDecomposition(time_feature_dim=8)
    rotations = torch.nn.functional.normalize(torch.tensor([[0.6, -0.2, 0.4, 0.5]]), dim=-1)

    updated = motion._apply_quaternion_delta(rotations, torch.zeros(1, 3))

    assert _quaternions_equivalent(updated, rotations)


def test_motion_decomposition_rotation_delta_matches_quaternion_composition():
    motion = MotionDecomposition(time_feature_dim=8)
    rotations = torch.nn.functional.normalize(torch.tensor([[0.7, 0.1, -0.3, 0.6]]), dim=-1)
    raw_d_rot = torch.tensor([[0.03, -0.02, 0.04]])
    delta_xyz = torch.tanh(raw_d_rot) * motion.max_rot_delta * 0.5
    delta_w = torch.sqrt(torch.clamp(1.0 - (delta_xyz ** 2).sum(dim=-1, keepdim=True), min=1e-8))
    delta_quat = torch.cat((delta_w, delta_xyz), dim=-1)
    expected = _quaternion_multiply(rotations, delta_quat)

    updated = motion._apply_quaternion_delta(rotations, raw_d_rot)

    assert _quaternions_equivalent(updated, expected)


def test_renderer_recomputes_covariance_after_cams_deformation(monkeypatch):
    class _FakeRasterizer:
        last_kwargs = None

        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            type(self).last_kwargs = kwargs
            count = kwargs["means3D"].shape[0]
            return torch.zeros(1, 1, 1), torch.ones(count), torch.zeros(count)

    renderer = _load_gaussian_renderer_module(monkeypatch, _FakeRasterizer)
    covariance_calls = {}

    class _FakeDeformation:
        def __call__(self, means3d, scales, rotations, opacity, time):
            return means3d + 1.0, scales + 3.0, rotations + 5.0, opacity + 7.0

        def get_aux_outputs(self):
            return {}

    class _FakePointCloud:
        def __init__(self):
            self.get_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            self._opacity = torch.tensor([[0.1], [0.2]])
            self._scaling = torch.tensor([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]])
            self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.2, 0.3]])
            self._deformation_table = torch.tensor([True, False])
            self._deformation_accum = torch.zeros(2, 3)
            self._deformation = _FakeDeformation()
            self.active_sh_degree = 0
            self.max_sh_degree = 0
            self.scaling_activation = lambda value: value + 10.0
            self.rotation_activation = lambda value: value + 20.0
            self.opacity_activation = lambda value: value
            self.get_features = torch.zeros(2, 1, 3)
            self.get_covariance_calls = 0

        def get_covariance(self, scaling_modifier=1.0):
            self.get_covariance_calls += 1
            return torch.tensor([[999.0]])

        def covariance_activation(self, scales, scaling_modifier, rotations):
            covariance_calls["scales"] = scales.clone()
            covariance_calls["rotations"] = rotations.clone()
            covariance_calls["scaling_modifier"] = scaling_modifier
            return torch.cat((scales, rotations[:, :3]), dim=-1)

    point_cloud = _FakePointCloud()
    camera = SimpleNamespace(
        FoVx=0.5,
        FoVy=0.5,
        image_height=4,
        image_width=4,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        time=0.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=True, convert_SHs_python=False, debug=False)

    renderer.render(camera, point_cloud, pipe, torch.zeros(3), override_color=torch.zeros(2, 3), stage="fine")

    expected_scales = torch.tensor([[14.0, 14.5, 15.0], [12.5, 13.0, 13.5]])
    expected_rotations = torch.tensor([[26.0, 25.0, 25.0, 25.0], [20.8, 20.1, 20.2, 20.3]])
    assert point_cloud.get_covariance_calls == 0
    assert torch.allclose(covariance_calls["scales"], expected_scales)
    assert torch.allclose(covariance_calls["rotations"], expected_rotations)
    assert covariance_calls["scaling_modifier"] == 1.0
    assert _FakeRasterizer.last_kwargs["scales"] is None
    assert _FakeRasterizer.last_kwargs["rotations"] is None
    assert torch.allclose(
        _FakeRasterizer.last_kwargs["cov3D_precomp"],
        torch.cat((expected_scales, expected_rotations[:, :3]), dim=-1),
    )


def test_renderer_applies_appearance_delta_and_opacity_gate_to_rasterizer_inputs(monkeypatch):
    class _FakeRasterizer:
        last_kwargs = None

        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            type(self).last_kwargs = kwargs
            count = kwargs["means3D"].shape[0]
            return torch.zeros(1, 1, 1), torch.ones(count), torch.zeros(count)

    renderer = _load_gaussian_renderer_module(monkeypatch, _FakeRasterizer)

    class _FakeDeformation:
        def __call__(self, means3d, scales, rotations, opacity, time):
            return means3d + 1.0, scales + 2.0, rotations + 3.0, opacity + 4.0

        def get_aux_outputs(self):
            return {
                "appearance_rgb_delta": torch.tensor([[0.3, -0.1, 0.2]]),
                "visibility_alpha": torch.ones(1, 1),
                "lifecycle_alpha": torch.ones(1, 1),
            }

    class _FakePointCloud:
        def __init__(self):
            self.get_xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            self._opacity = torch.tensor([[0.1], [0.2]])
            self._scaling = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
            self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.2, 0.3]])
            self._deformation_table = torch.tensor([True, False])
            self._deformation_accum = torch.zeros(2, 3)
            self._deformation = _FakeDeformation()
            self.active_sh_degree = 0
            self.max_sh_degree = 0
            self.scaling_activation = lambda value: value
            self.rotation_activation = lambda value: value
            self.opacity_activation = lambda value: value
            self.get_features = torch.zeros(2, 1, 3)

    point_cloud = _FakePointCloud()
    camera = SimpleNamespace(
        FoVx=0.5,
        FoVy=0.5,
        image_height=4,
        image_width=4,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        time=0.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    base_colors = torch.tensor([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]])

    outputs = renderer.render(camera, point_cloud, pipe, torch.zeros(3), override_color=base_colors, stage="fine")

    assert torch.allclose(_FakeRasterizer.last_kwargs["colors_precomp"][0], torch.tensor([0.5, 0.1, 0.4]))
    assert torch.allclose(_FakeRasterizer.last_kwargs["colors_precomp"][1], base_colors[1])
    assert torch.allclose(_FakeRasterizer.last_kwargs["opacities"], torch.tensor([[4.1], [0.2]]))
    assert torch.allclose(outputs["deformation_aux"]["appearance_rgb_delta"], torch.tensor([[0.3, -0.1, 0.2]]))


def test_renderer_pixel_routing_preserves_expert_appearance_and_opacity_controls(monkeypatch):
    class _FakeRasterizer:
        calls = []

        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            type(self).calls.append(kwargs)
            count = kwargs["means3D"].shape[0]
            color_value = 0.0
            if kwargs.get("colors_precomp") is not None:
                color_value = float(kwargs["colors_precomp"][0, 0].item())
            opacity_value = float(kwargs["opacities"][0, 0].item()) if kwargs.get("opacities") is not None else 0.0
            render_value = torch.full((3, 1, 1), color_value + opacity_value)
            return render_value, torch.ones(count), torch.zeros(1, 1)

    renderer = _load_gaussian_renderer_module(monkeypatch, _FakeRasterizer)

    class _FakeDeformation:
        def __call__(self, means3d, scales, rotations, opacity, time):
            return means3d + 1.0, scales + 2.0, rotations + 3.0, opacity + 4.0

        def get_aux_outputs(self):
            return {
                "geo_expert_means3d": torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
                "geo_expert_scales": torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]),
                "geo_expert_rotations": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
                "geo_expert_opacity_logits": torch.tensor([[[4.1], [4.1]]]),
                "gaussian_pi_geo_prior": torch.tensor([[3.0, -3.0]]),
                "appearance_rgb_delta": torch.tensor([[0.3, -0.1, 0.2]]),
                "vis_expert_rgb_delta": torch.tensor([[[0.0, 0.0, 0.0], [-0.2, 0.1, -0.1]]]),
                "vis_expert_visibility_alpha": torch.tensor([[[1.0], [0.0]]]),
                "lifecycle_expert_alpha": torch.tensor([[[1.0], [1.0]]]),
                "visibility_alpha": torch.ones(1, 1),
                "lifecycle_alpha": torch.ones(1, 1),
            }

    class _FakePointCloud:
        def __init__(self):
            self.get_xyz = torch.tensor([[0.0, 0.0, 0.0]])
            self._opacity = torch.tensor([[0.1]])
            self._scaling = torch.tensor([[1.0, 1.0, 1.0]])
            self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self._deformation_table = torch.tensor([True])
            self._deformation_accum = torch.zeros(1, 3)
            self._deformation = _FakeDeformation()
            self.active_sh_degree = 0
            self.max_sh_degree = 0
            self.scaling_activation = lambda value: value
            self.rotation_activation = lambda value: value
            self.opacity_activation = lambda value: value
            self.get_features = torch.zeros(1, 1, 3)

    point_cloud = _FakePointCloud()
    camera = SimpleNamespace(
        FoVx=0.5,
        FoVy=0.5,
        image_height=1,
        image_width=1,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        time=0.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    base_colors = torch.tensor([[0.2, 0.2, 0.2]])

    outputs = renderer.render(camera, point_cloud, pipe, torch.zeros(3), override_color=base_colors, stage="fine")

    expert_render_calls = [
        call
        for call in _FakeRasterizer.calls
        if call.get("colors_precomp") is not None
        and call["colors_precomp"].shape == (1, 3)
        and float(call["colors_precomp"][0, 0].item()) <= 1.0
    ]
    assert torch.allclose(expert_render_calls[0]["colors_precomp"][0], torch.tensor([0.5, 0.1, 0.4]))
    assert torch.allclose(expert_render_calls[1]["colors_precomp"][0], torch.tensor([0.3, 0.2, 0.3]))
    assert torch.allclose(expert_render_calls[0]["opacities"], torch.tensor([[4.1]]))
    assert torch.allclose(expert_render_calls[1]["opacities"], torch.tensor([[0.0]]))
    assert outputs["deformation_aux"]["pixel_routing_weights"].shape == (2, 1, 1)


def test_tracking_losses_use_covered_pixel_routing_weights_only():
    args = _build_loss_args()
    aux = {
        "pixel_routing_weights": torch.tensor(
            [
                [[0.9, 0.0], [0.0, 0.0]],
                [[0.1, 0.0], [0.0, 0.0]],
            ]
        ),
        "pi_vis": torch.tensor([[1.0, 0.0]]),
        "d_mu": torch.zeros(1, 3),
    }

    losses = compute_tracking_losses(
        aux=aux,
        iteration=10,
        args=args,
        prev_d_mu=None,
        active_geo=2,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("global", "local"),
        vis_expert_names=("stable", "transient"),
    )

    assert torch.allclose(losses["usage_geo_global"], torch.tensor(0.9))
    assert torch.allclose(losses["usage_geo_local"], torch.tensor(0.1))


def test_renderer_pixel_routing_masks_uncovered_experts_and_aggregates_radii(monkeypatch):
    class _FakeRasterizer:
        calls = []

        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            type(self).calls.append(kwargs)
            mean_x = float(kwargs["means3D"][0, 0].item())
            color_value = float(kwargs["colors_precomp"][0, 0].item()) if kwargs.get("colors_precomp") is not None else 0.0
            render_value = torch.full((3, 1, 1), color_value)
            radii = torch.tensor([0.0]) if mean_x < 15.0 else torch.tensor([2.0])
            return render_value, radii, torch.zeros(1, 1)

    renderer = _load_gaussian_renderer_module(monkeypatch, _FakeRasterizer)

    class _FakeDeformation:
        def __call__(self, means3d, scales, rotations, opacity, time):
            return means3d + 1.0, scales, rotations, opacity

        def get_aux_outputs(self):
            return {
                "geo_expert_means3d": torch.tensor([[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]),
                "geo_expert_scales": torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
                "geo_expert_rotations": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
                "geo_expert_opacity_logits": torch.tensor([[[1.0], [1.0]]]),
                "gaussian_pi_geo_prior": torch.tensor([[1.0, 0.0]]),
                "vis_expert_rgb_delta": torch.zeros(1, 2, 3),
                "vis_expert_visibility_alpha": torch.ones(1, 2, 1),
                "lifecycle_expert_alpha": torch.ones(1, 2, 1),
                "visibility_alpha": torch.ones(1, 1),
                "lifecycle_alpha": torch.ones(1, 1),
            }

    class _FakePointCloud:
        def __init__(self):
            self.get_xyz = torch.tensor([[0.0, 0.0, 0.0]])
            self._opacity = torch.tensor([[0.1]])
            self._scaling = torch.tensor([[1.0, 1.0, 1.0]])
            self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self._deformation_table = torch.tensor([True])
            self._deformation_accum = torch.zeros(1, 3)
            self._deformation = _FakeDeformation()
            self.active_sh_degree = 0
            self.max_sh_degree = 0
            self.scaling_activation = lambda value: value
            self.rotation_activation = lambda value: value
            self.opacity_activation = lambda value: value
            self.get_features = torch.zeros(1, 1, 3)

    point_cloud = _FakePointCloud()
    camera = SimpleNamespace(
        FoVx=0.5,
        FoVy=0.5,
        image_height=1,
        image_width=1,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        time=0.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    base_colors = torch.tensor([[0.2, 0.2, 0.2]])

    outputs = renderer.render(camera, point_cloud, pipe, torch.zeros(3), override_color=base_colors, stage="fine")

    assert torch.allclose(outputs["render"], torch.full((3, 1, 1), 0.2))
    assert torch.allclose(outputs["radii"], torch.tensor([2.0]))
    assert torch.allclose(outputs["deformation_aux"]["pixel_routing_weights"], torch.tensor([[[1.0]], [[0.0]]]))


def test_cams_visibility_head_exposes_render_affecting_controls():
    head = model = CAMSGSTracking(time_feature_dim=8).visibility
    phase = TrackingPhase(
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
    )
    time_features = torch.randn(4, 8)
    means3d = torch.randn(4, 3)
    opacity = torch.randn(4, 1)
    d_mu = torch.randn(4, 3)

    outputs = head(time_features, means3d, opacity, d_mu, phase)

    assert "visibility_alpha" in outputs
    assert "appearance_rgb_delta" in outputs
    assert outputs["visibility_alpha"].shape == (4, 1)
    assert outputs["appearance_rgb_delta"].shape == (4, 3)


def test_cams_lifecycle_head_exposes_render_affecting_controls():
    head = CAMSGSTracking(time_feature_dim=8).lifecycle
    phase = TrackingPhase(
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
    )
    gating_state = {
        "scaffold_weights": torch.softmax(torch.randn(4, 3), dim=-1),
        "cut_gate_values": torch.sigmoid(torch.randn(4, 3)),
    }
    outputs = head(torch.randn(4, 8), gating_state, phase)

    assert "lifecycle_alpha" in outputs
    assert outputs["lifecycle_alpha"].shape == (4, 1)


def test_cams_forward_dynamic_respects_no_ds_no_do_no_dr_end_to_end():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    args.no_ds = True
    args.no_do = True
    args.no_dr = True
    model = Deformation(D=1, W=8, args=args)
    phase = model.scheduler.build(3000, 9000)
    model.set_tracking_phase(phase)

    points = torch.randn(4, 3)
    scales = torch.randn(4, 3)
    rotations = torch.randn(4, 4)
    rotations[:, 0] = 1.0
    opacity = torch.randn(4, 1)
    times = torch.rand(4, 1)
    time_features = torch.randn(4, 8)

    _, scales_t, rotations_t, opacity_t = model.forward_dynamic(
        points,
        scales,
        rotations,
        opacity,
        times,
        time_features,
    )

    assert torch.allclose(scales_t, scales)
    assert torch.allclose(rotations_t, rotations)
    assert torch.allclose(opacity_t, opacity)


def test_cams_lifecycle_class_semantics_match_balance_target():
    head = CAMSGSTracking(time_feature_dim=8).lifecycle
    phase = TrackingPhase(
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
    )
    gating_state = {
        "scaffold_weights": torch.softmax(torch.randn(4, 3), dim=-1),
        "cut_gate_values": torch.sigmoid(torch.randn(4, 3)),
    }
    outputs = head(torch.randn(4, 8), gating_state, phase)

    assert torch.allclose(outputs["lifecycle_alpha"], outputs["lifecycle_probs"][:, :1])


def test_cams_optimizer_groups_cover_output_affecting_motion_parameters():
    motion = MotionDecomposition(time_feature_dim=8)
    groups = motion.named_parameter_groups()
    grouped_params = {id(param) for params in groups.values() for param in params}

    for module in (
        motion.global_motion,
        motion.local_motion,
        motion.rotation_head,
        motion.scale_head,
        motion.opacity_head,
    ):
        for param in module.parameters():
            assert id(param) in grouped_params


def test_cams_gs_uses_local_rotation_and_scale_caps():
    tracking = CAMSGSTracking(
        time_feature_dim=8,
        max_rot_local=0.10,
        max_scale_local=0.10,
        max_rot_smooth=0.05,
        max_scale_smooth=0.05,
    )

    assert tracking.motion.max_rot_delta == 0.10
    assert tracking.motion.max_scale_delta == 0.10


def test_deformation_wires_cams_gs_local_rotation_and_scale_caps():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    args.max_rot_local = 0.10
    args.max_scale_local = 0.10
    args.max_rot_smooth = 0.01
    args.max_scale_smooth = 0.01

    model = Deformation(D=1, W=8, args=args)

    assert model.cams_head is not None
    assert model.cams_head.motion.max_rot_delta == 0.10
    assert model.cams_head.motion.max_scale_delta == 0.10


def test_cams_gs_forward_emits_patch_c_aux_and_supports_tracking_losses():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    model = Deformation(D=1, W=8, args=args)
    phase = model.scheduler.build(8000, 9000)
    model.set_tracking_phase(phase)

    points = torch.randn(6, 3)
    scales = torch.randn(6, 3)
    rotations = torch.randn(6, 4)
    opacity = torch.randn(6, 1)
    times = torch.rand(6, 1)
    time_features = torch.randn(6, 8)

    pts_t, scales_t, rotations_t, opacity_t = model.forward_dynamic(
        points,
        scales,
        rotations,
        opacity,
        times,
        time_features,
    )
    aux = model.get_aux_outputs()

    assert pts_t.shape == points.shape
    assert scales_t.shape == scales.shape
    assert rotations_t.shape == rotations.shape
    assert opacity_t.shape == opacity.shape
    assert aux["d_mu"].shape == points.shape
    assert aux["d_rot"].shape == (points.shape[0], 3)
    assert aux["d_scale"].shape == scales.shape
    assert aux["d_opacity_logit"].shape == opacity.shape
    assert aux["global_motion"].shape == points.shape
    assert aux["local_motion"].shape == points.shape
    assert aux["cut_graph_motion"].shape == points.shape
    assert aux["geo_expert_d_mu"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES), 3)
    assert aux["geo_expert_means3d"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES), 3)
    assert aux["geo_expert_scales"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES), 3)
    assert aux["geo_expert_rotations"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES), 4)
    assert aux["geo_expert_opacity_logits"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES), 1)
    assert aux["gaussian_pi_geo_prior"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES))
    assert aux["gaussian_pi_vis_prior"].shape == (points.shape[0], len(model.cams_head.VIS_EXPERT_NAMES))
    assert aux["visibility_alpha"].shape == (points.shape[0], 1)
    assert aux["appearance_rgb_delta"].shape == (points.shape[0], 3)
    assert aux["vis_expert_rgb_delta"].shape == (points.shape[0], len(model.cams_head.VIS_EXPERT_NAMES), 3)
    assert aux["vis_expert_visibility_alpha"].shape == (points.shape[0], len(model.cams_head.VIS_EXPERT_NAMES), 1)
    assert aux["lifecycle_alpha"].shape == (points.shape[0], 1)
    assert aux["lifecycle_expert_alpha"].shape == (points.shape[0], 2, 1)
    assert aux["scaffold_weights"].shape == (points.shape[0], 3)
    assert aux["cut_gate_logits"].shape == (points.shape[0], 3)
    assert aux["cut_gate_values"].shape == (points.shape[0], 3)
    assert aux["pi_geo"].shape == (points.shape[0], len(model.cams_head.GEO_EXPERT_NAMES))
    assert aux["pi_vis"].shape == (points.shape[0], len(model.cams_head.VIS_EXPERT_NAMES))
    assert aux["visibility_logits"].shape[0] == points.shape[0]
    assert aux["appearance_offsets"].shape[0] == points.shape[0]
    assert aux["lifecycle_logits"].shape[0] == points.shape[0]
    assert aux["tracking_phase_name"] == "joint_finetune"
    assert torch.is_tensor(aux["entropy_geo"])
    assert torch.is_tensor(aux["entropy_vis"])

    loss_args = _build_loss_args()
    loss_dict = compute_tracking_losses(
        aux=aux,
        iteration=8000,
        args=loss_args,
        prev_d_mu=None,
        active_geo=phase.active_geo,
        active_vis=phase.active_vis,
        enable_visibility=phase.enable_visibility,
        geo_expert_names=model.cams_head.GEO_EXPERT_NAMES,
        vis_expert_names=model.cams_head.VIS_EXPERT_NAMES,
    )

    assert "usage_geo_global" in loss_dict
    assert "usage_vis_stable" in loss_dict
    assert "entropy_geo" in loss_dict
    assert "entropy_vis" in loss_dict
    assert "mean_norm_d_mu" in loss_dict
    assert "mean_abs_d_opacity" in loss_dict


def test_cams_patch_c_losses_are_phase_gated():
    args = _build_loss_args()
    aux = {
        "pi_geo": torch.tensor([[0.5, 0.3, 0.2]]),
        "pi_vis": torch.tensor([[1.0, 0.0]]),
        "d_mu": torch.zeros(1, 3),
        "appearance_offsets": torch.ones(1, 3),
        "lifecycle_logits": torch.ones(1, 2),
        "lifecycle_probs": torch.tensor([[0.8, 0.2]]),
        "tracking_phase_name": "motion_warmup",
    }
    losses = compute_tracking_losses(
        aux=aux,
        iteration=10,
        args=args,
        prev_d_mu=None,
        active_geo=3,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("global", "local", "cut_graph"),
        vis_expert_names=("stable", "transient"),
    )

    assert "L_appearance_reg" not in losses
    assert "L_lifecycle_balance" not in losses
    assert "L_lifecycle_reg" not in losses


@pytest.mark.parametrize(
    ("disabled_flag", "output_name", "aux_name"),
    [
        ("no_ds", "scales", "d_scale"),
        ("no_dr", "rotations", "d_rot"),
        ("no_do", "opacity", "d_opacity_logit"),
    ],
)
def test_cams_forward_dynamic_respects_individual_disable_flags(disabled_flag, output_name, aux_name):
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    setattr(args, disabled_flag, True)
    model = Deformation(D=1, W=8, args=args)
    phase = model.scheduler.build(3000, 9000)
    model.set_tracking_phase(phase)

    points = torch.randn(4, 3)
    scales = torch.randn(4, 3)
    rotations = torch.randn(4, 4)
    rotations[:, 0] = 1.0
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
    del pts_t
    aux = model.get_aux_outputs()

    outputs = {
        "scales": (scales_t, scales),
        "rotations": (rotations_t, rotations),
        "opacity": (opacity_t, opacity),
    }
    actual, expected = outputs[output_name]
    assert torch.allclose(actual, expected)
    assert torch.allclose(aux[aux_name], torch.zeros_like(aux[aux_name]))




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


def test_cams_patch_c_losses_propagate_gradients_to_appearance_and_lifecycle():
    args = _build_deformation_args()
    args.tracking_type = "cams_gs"
    model = Deformation(D=1, W=8, args=args)
    phase = model.scheduler.build(8000, 9000)
    model.set_tracking_phase(phase)

    points = torch.randn(6, 3)
    scales = torch.randn(6, 3)
    rotations = torch.randn(6, 4)
    rotations[:, 0] = 1.0
    opacity = torch.randn(6, 1)
    times = torch.rand(6, 1)
    time_features = torch.randn(6, 8)

    model.forward_dynamic(points, scales, rotations, opacity, times, time_features)
    aux = model.get_aux_outputs()
    loss_args = _build_loss_args()
    loss_args.lambda_appearance_reg = 1e-3
    loss_args.lambda_lifecycle_balance = 1e-3
    loss_args.lambda_lifecycle_reg = 1e-3
    loss_dict = compute_tracking_losses(
        aux=aux,
        iteration=8000,
        args=loss_args,
        prev_d_mu=None,
        active_geo=phase.active_geo,
        active_vis=phase.active_vis,
        enable_visibility=phase.enable_visibility,
        geo_expert_names=model.cams_head.GEO_EXPERT_NAMES,
        vis_expert_names=model.cams_head.VIS_EXPERT_NAMES,
    )
    total_loss = sum(value for name, value in loss_dict.items() if name.startswith("L_") and torch.is_tensor(value))
    total_loss.backward()

    appearance_grads = [param.grad for param in model.cams_head.visibility.appearance_head.parameters()]
    lifecycle_grads = [param.grad for param in model.cams_head.lifecycle.lifecycle_head.parameters()]
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in appearance_grads)
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in lifecycle_grads)


def test_train_aux_merge_handles_phase_metadata_and_tensor_values():
    deformation_aux_list = [
        {
            "pi_geo": torch.tensor([[0.6, 0.3, 0.1]]),
            "d_mu": torch.tensor([[0.1, 0.0, 0.0]]),
            "tracking_phase_name": "visibility_refine",
        },
        {
            "pi_geo": torch.tensor([[0.3, 0.4, 0.3]]),
            "d_mu": torch.tensor([[0.2, 0.0, 0.0]]),
            "tracking_phase_name": "visibility_refine",
        },
    ]

    merged_aux = {}
    for key in deformation_aux_list[0].keys():
        values = [a[key] for a in deformation_aux_list if key in a]
        if not values:
            continue
        if not torch.is_tensor(values[0]):
            merged_aux[key] = values[0]
            continue
        if values[0].dim() == 0:
            merged_aux[key] = torch.stack(values).mean()
        else:
            merged_aux[key] = torch.cat(values, dim=0)

    assert merged_aux["tracking_phase_name"] == "visibility_refine"
    assert merged_aux["pi_geo"].shape == (2, 3)
    assert merged_aux["d_mu"].shape == (2, 3)


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
