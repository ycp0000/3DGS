import importlib.util
import math
import sys
from argparse import ArgumentParser, Namespace
from collections import UserDict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arguments import (
    ModelHiddenParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from models.tracking.cams_gs_tracking import CAMSGSScheduler
from train import (
    allows_gaussian_topology_updates,
    clip_residual_refinement_gradients,
    normalize_endomoeg_pipeline_stage,
    should_apply_color_refinement,
    validate_global_anchor_config,
    validate_residual_depth_shapes,
    validate_endomoeg_pipeline_args,
)
from utils.params_utils import merge_hparams

ENDONERF_PRESET_DIR = ROOT / "arguments" / "endonerf"
_TRACKING_LOSSES_SPEC = importlib.util.spec_from_file_location(
    "tracking_losses_module",
    ROOT / "scene" / "tracking_losses.py",
)
tracking_losses_module = importlib.util.module_from_spec(_TRACKING_LOSSES_SPEC)
assert _TRACKING_LOSSES_SPEC.loader is not None
_TRACKING_LOSSES_SPEC.loader.exec_module(tracking_losses_module)
_build_geo_target = tracking_losses_module._build_geo_target
_get_float_arg = tracking_losses_module._get_float_arg


def _build_parser():
    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    ModelHiddenParams(parser)
    PipelineParams(parser)
    return parser


def _build_default_args():
    return _build_parser().parse_args([])


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_legacy_fallbacks(args):
    device = torch.device("cpu")
    dtype = torch.float32

    target = _build_geo_target(
        args,
        ("static", "hexplane", "local", "smooth"),
        device=device,
        dtype=dtype,
    )
    expected = torch.tensor([0.25, 0.63, 0.20, 0.27], device=device, dtype=dtype)
    expected = expected / expected.sum()
    assert torch.allclose(target, expected, atol=1e-6)

    stage2_target = _build_geo_target(
        args,
        ("static", "hexplane"),
        device=device,
        dtype=dtype,
    )
    expected_stage2 = torch.tensor([0.25, 0.90], device=device, dtype=dtype)
    expected_stage2 = expected_stage2 / expected_stage2.sum()
    assert torch.allclose(stage2_target, expected_stage2, atol=1e-6)

    assert abs(
        _get_float_arg(args, "lambda_mag_g3_mu", _get_float_arg(args, "lambda_mag_g2_mu", 2e-5))
        - 0.123
    ) < 1e-9
    assert abs(
        _get_float_arg(args, "lambda_sat_g3_disp", _get_float_arg(args, "lambda_sat_g2_disp", 1e-4))
        - 0.456
    ) < 1e-9
    assert abs(
        _get_float_arg(args, "lambda_raw_g3_disp", _get_float_arg(args, "lambda_raw_g2_disp", 1e-4))
        - 0.789
    ) < 1e-9



def _load_preset_args(preset_name: str):
    module = _load_module(ENDONERF_PRESET_DIR / preset_name)
    return merge_hparams(
        _build_default_args(),
        {
            "ModelParams": getattr(module, "ModelParams", {}),
            "OptimizationParams": getattr(module, "OptimizationParams", {}),
            "ModelHiddenParams": getattr(module, "ModelHiddenParams", {}),
            "PipelineParams": getattr(module, "PipelineParams", {}),
        },
    )



def test_endonerf_presets_only_use_known_parser_keys():
    args = _build_default_args()

    for preset_path in sorted(ENDONERF_PRESET_DIR.glob("*.py")):
        module = _load_module(preset_path)
        for section_name in (
            "ModelParams",
            "OptimizationParams",
            "ModelHiddenParams",
            "PipelineParams",
        ):
            section = getattr(module, section_name, None)
            if not isinstance(section, dict):
                continue
            unknown_keys = sorted(key for key in section if not hasattr(args, key))
            assert not unknown_keys, (
                f"{preset_path.name} contains unsupported {section_name} keys: {unknown_keys}"
            )



def test_merge_hparams_accepts_mapping_config_sections():
    args = _build_default_args()
    config = {
        "ModelParams": UserDict({"extra_mark": "endonerf", "camera_extent": 10}),
        "ModelHiddenParams": UserDict({"tracking_type": "cams_gs"}),
    }

    merged = merge_hparams(args, config)

    assert merged.extra_mark == "endonerf"
    assert merged.camera_extent == 10
    assert merged.tracking_type == "cams_gs"



def test_merge_hparams_keeps_endonerf_pruning_and_tracking_regularizers():
    args = _build_default_args()
    module = _load_module(ENDONERF_PRESET_DIR / "cutting_disentangled_moe.py")

    merged = merge_hparams(
        args,
        {
            "OptimizationParams": module.OptimizationParams,
            "ModelHiddenParams": module.ModelHiddenParams,
        },
    )

    assert merged.pruning_interval == 3000
    assert merged.densify_until_iter == 15000
    assert merged.lambda_mag_g1_mu == 1e-4
    assert merged.lambda_mag_g2_mu == 2e-5
    assert merged.lambda_sat_g1_disp == 5e-4
    assert merged.lambda_sat_g2_disp == 1e-4
    assert merged.lambda_raw_g1_disp == 1e-4
    assert merged.lambda_raw_g2_disp == 1e-4



def test_merge_hparams_maps_legacy_aliases_and_preserves_fallbacks():
    args = _build_default_args()

    merged = merge_hparams(
        args,
        {
            "OptimizationParams": {"prune_interval": 321},
            "ModelHiddenParams": {
                "max_disp_shared_ratio": 0.125,
                "target_geo_static": 0.25,
                "target_geo_smooth": 0.9,
                "lambda_mag_g2_mu": 0.123,
                "lambda_sat_g2_disp": 0.456,
                "lambda_raw_g2_disp": 0.789,
            },
        },
    )

    assert merged.pruning_interval == 321
    assert merged.max_disp_hexplane_ratio == 0.125
    assert merged.target_geo_smooth == 0.9
    _assert_legacy_fallbacks(merged)



def test_get_combined_args_maps_legacy_cfg_args_aliases(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg_args = (
        f"Namespace(model_path={str(model_path)!r}, prune_interval=321, "
        "max_disp_shared_ratio=0.125, target_geo_static=0.25, "
        "target_geo_smooth=0.9, lambda_mag_g2_mu=0.123, "
        "lambda_sat_g2_disp=0.456, lambda_raw_g2_disp=0.789)"
    )
    (model_path / "cfg_args").write_text(cfg_args, encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--model_path", str(model_path)])
    merged = get_combined_args(_build_parser())

    assert merged.pruning_interval == 321
    assert merged.max_disp_hexplane_ratio == 0.125
    assert merged.target_geo_smooth == 0.9
    _assert_legacy_fallbacks(merged)


def test_get_combined_args_reads_cfg_args_with_nan_values(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg_args = (
        f"Namespace(model_path={str(model_path)!r}, target_geo_smooth=nan, "
        "target_geo_hexplane=inf)"
    )
    (model_path / "cfg_args").write_text(cfg_args, encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["prog", "--model_path", str(model_path)])
    merged = get_combined_args(_build_parser())

    assert math.isnan(merged.target_geo_smooth)
    assert math.isinf(merged.target_geo_hexplane)



def test_get_combined_args_allows_cli_overrides_on_top_of_legacy_cfg_args(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    cfg_args = (
        f"Namespace(model_path={str(model_path)!r}, prune_interval=321, "
        "max_disp_shared_ratio=0.125, target_geo_static=0.25, "
        "target_geo_smooth=0.9, lambda_mag_g2_mu=0.123, "
        "lambda_sat_g2_disp=0.456, lambda_raw_g2_disp=0.789)"
    )
    (model_path / "cfg_args").write_text(cfg_args, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--model_path", str(model_path), "--pruning_interval", "999"],
    )
    merged = get_combined_args(_build_parser())

    assert merged.pruning_interval == 999
    assert merged.max_disp_hexplane_ratio == 0.125
    assert merged.target_geo_smooth == 0.9
    _assert_legacy_fallbacks(merged)



def test_main_heterogeneous_presets_match_baseline_protocol_and_motion_caps():
    for scene_name in ("cutting", "pulling"):
        baseline = _load_preset_args(f"{scene_name}_original.py")
        for variant in (
            f"{scene_name}_geo_moe_only.py",
            f"{scene_name}_disentangled_moe.py",
            f"{scene_name}_disentangled_moe_dense.py",
            f"{scene_name}_disentangled_moe_sparse.py",
        ):
            candidate = _load_preset_args(variant)
            assert candidate.coarse_iterations == baseline.coarse_iterations
            assert candidate.pruning_interval == baseline.pruning_interval
            assert candidate.densify_until_iter == baseline.densify_until_iter
            assert candidate.iterations == baseline.iterations
            assert candidate.position_lr_max_steps == baseline.position_lr_max_steps
            assert candidate.max_disp_smooth_ratio == 0.01
            assert candidate.max_disp_local_ratio == 0.03



def test_get_float_arg_treats_non_finite_values_as_unset():
    args = type("Args", (), {"value": float("inf")})()
    assert _get_float_arg(args, "value", 1.23) == 1.23

    args.value = -float("inf")
    assert _get_float_arg(args, "value", 4.56) == 4.56


def test_cams_gs_presets_use_known_parser_keys_and_tracking_type():
    for scene_name in ("cutting", "pulling"):
        baseline = _load_preset_args(f"{scene_name}_original.py")
        candidate = _load_preset_args(f"{scene_name}_cams_gs.py")
        assert candidate.tracking_type == "cams_gs"
        assert candidate.coarse_iterations == baseline.coarse_iterations
        assert candidate.pruning_interval == baseline.pruning_interval
        assert candidate.iterations == baseline.iterations
        assert candidate.position_lr_max_steps == baseline.position_lr_max_steps
        assert candidate.max_disp_smooth_ratio == 0.01
        assert candidate.max_disp_local_ratio == 0.03


def test_endomoeg_presets_use_complete_expert_pipeline_defaults():
    for scene_name in ("cutting", "pulling"):
        baseline = _load_preset_args(f"{scene_name}_original.py")
        candidate = _load_preset_args(f"{scene_name}_endomoeg.py")
        assert candidate.tracking_type == "original"
        assert candidate.endomoeg_pipeline_stage == ""
        assert candidate.extra_mark == "endonerf"
        assert candidate.coarse_iterations == baseline.coarse_iterations
        assert candidate.pruning_interval == baseline.pruning_interval
        assert candidate.iterations == baseline.iterations
        assert candidate.position_lr_max_steps == baseline.position_lr_max_steps
        assert candidate.endomoeg_expert_hidden_dim == 64
        assert candidate.endomoeg_scaffold_max_node_offset_ratio == pytest.approx(
            0.02
        )
        assert candidate.endomoeg_scaffold_max_radius_scale == pytest.approx(4.0)
        assert candidate.endomoeg_scaffold_initial_gate_probability == pytest.approx(
            0.05
        )
        assert candidate.lambda_scaffold_gate_sparsity == pytest.approx(1e-3)
        assert candidate.endomoeg_residual_hard_quantile == pytest.approx(0.7)
        assert candidate.endomoeg_residual_reconstruction_weight == pytest.approx(
            1.0
        )
        assert candidate.endomoeg_residual_boost_weight == pytest.approx(0.25)
        assert candidate.endomoeg_residual_preserve_weight == pytest.approx(1.0)
        assert candidate.endomoeg_residual_no_regret_weight == pytest.approx(2.0)
        assert candidate.endomoeg_residual_no_regret_temperature == pytest.approx(
            0.01
        )
        assert candidate.endomoeg_residual_depth_weight == pytest.approx(0.05)
        assert candidate.endomoeg_residual_lr_scale == pytest.approx(0.01)
        assert candidate.endomoeg_residual_warmup_iterations == 500
        assert candidate.endomoeg_residual_gradient_clip == pytest.approx(0.05)
        assert candidate.endomoeg_residual_max_baseline_psnr_drop == pytest.approx(
            0.05
        )
        assert candidate.endomoeg_residual_render_parity_tolerance == pytest.approx(
            1e-5
        )
        assert candidate.endomoeg_contact_max_spatial_offset_ratio == pytest.approx(
            0.02
        )
        assert candidate.endomoeg_contact_max_velocity_ratio == pytest.approx(0.05)
        assert candidate.endomoeg_contact_max_acceleration_ratio == pytest.approx(
            0.05
        )
        assert candidate.endomoeg_contact_max_rotation_radians == pytest.approx(
            0.5
        )
        assert candidate.endomoeg_contact_max_scale_delta == pytest.approx(0.1)
        assert candidate.endomoeg_contact_initial_duration == pytest.approx(0.15)
        assert candidate.kplanes_config["output_coordinate_dim"] == 64
        assert candidate.endomoeg_min_oracle_headroom == pytest.approx(0.3)
        assert candidate.endomoeg_router_gain_temperature == pytest.approx(0.02)
        assert candidate.endomoeg_router_lambda_gain == pytest.approx(0.1)
        assert candidate.endomoeg_router_lambda_sparsity == pytest.approx(1e-3)
        assert candidate.endomoeg_router_lambda_no_regret == pytest.approx(0.5)
        assert candidate.endomoeg_router_gradient_warmup == 20
        assert candidate.endomoeg_joint_anchor_lambda == pytest.approx(1e-3)
        assert candidate.endomoeg_joint_max_psnr_drop == pytest.approx(0.05)


def test_residual_experts_disable_canonical_topology_updates():
    global_hyper = type(
        "Hyper",
        (),
        {
            "endomoeg_pipeline_stage": "expert",
            "endomoeg_expert_role": "global",
        },
    )()
    local_hyper = type(
        "Hyper",
        (),
        {
            "endomoeg_pipeline_stage": "expert",
            "endomoeg_expert_role": "local",
        },
    )()
    contact_hyper = type(
        "Hyper",
        (),
        {
            "endomoeg_pipeline_stage": "expert",
            "endomoeg_expert_role": "contact",
        },
    )()

    assert allows_gaussian_topology_updates("coarse", local_hyper)
    assert allows_gaussian_topology_updates("fine", global_hyper)
    assert not allows_gaussian_topology_updates("fine", local_hyper)
    assert not allows_gaussian_topology_updates("fine", contact_hyper)


def test_residual_experts_disable_legacy_color_outlier_mask_refinement():
    teacher = object()

    assert should_apply_color_refinement(999, residual_teacher=None)
    assert not should_apply_color_refinement(1000, residual_teacher=None)
    assert not should_apply_color_refinement(1, residual_teacher=teacher)


def test_residual_depth_shape_contract_rejects_silent_broadcasting():
    candidate = torch.zeros(2, 1, 8, 10)
    teacher = torch.zeros_like(candidate)
    target = torch.zeros_like(candidate)

    validate_residual_depth_shapes(candidate, teacher, target)

    with pytest.raises(ValueError, match="must match"):
        validate_residual_depth_shapes(
            candidate,
            teacher[:1],
            target,
        )
    with pytest.raises(ValueError, match=r"\[B, 1, H, W\]"):
        validate_residual_depth_shapes(
            candidate,
            teacher,
            target.squeeze(1),
        )


def test_endomoeg_preset_preserves_cli_pipeline_stage_and_role(tmp_path):
    args = _build_default_args()
    args.endomoeg_pipeline_stage = "expert"
    args.endomoeg_expert_role = "local"
    args.endomoeg_bundle_dir = str((tmp_path / "bundles").resolve())
    args.endomoeg_min_expert_psnr = 30.0
    module = _load_module(ENDONERF_PRESET_DIR / "cutting_endomoeg.py")

    merged = merge_hparams(
        args,
        {
            "ModelParams": module.ModelParams,
            "OptimizationParams": module.OptimizationParams,
            "ModelHiddenParams": module.ModelHiddenParams,
        },
    )
    validated = validate_endomoeg_pipeline_args(merged)

    assert validated.endomoeg_pipeline_stage == "expert"
    assert validated.endomoeg_expert_role == "local"
    assert validated.tracking_type == "endomoeg_expert"


def test_endomoeg_component_loading_arguments_have_safe_defaults():
    args = _build_default_args()
    assert args.endomoeg_component_dir == ""
    assert args.endomoeg_component_output_dir == ""
    assert args.endomoeg_strict_component_loading is True
    assert args.endomoeg_stage_iterations == -1


def test_residual_gradient_clipping_only_touches_refinement_group():
    refinement = torch.nn.Parameter(torch.tensor((3.0, 4.0)))
    frozen = torch.nn.Parameter(torch.tensor((1.0, 2.0)))
    refinement.grad = torch.tensor((3.0, 4.0))
    frozen.grad = torch.tensor((7.0, 8.0))
    optimizer = torch.optim.Adam(
        [
            {
                "params": [refinement],
                "name": "tracking_expert_refinement",
            },
            {
                "params": [frozen],
                "name": "tracking_base_deformation",
            },
        ],
        lr=1e-3,
    )
    phase = type(
        "Phase",
        (),
        {
            "is_group_trainable": staticmethod(
                lambda name: name == "tracking_expert_refinement"
            )
        },
    )()

    norm = clip_residual_refinement_gradients(
        optimizer,
        phase,
        max_norm=1.0,
    )

    assert norm.item() == pytest.approx(5.0)
    assert refinement.grad.norm().item() == pytest.approx(1.0)
    assert torch.equal(frozen.grad, torch.tensor((7.0, 8.0)))


def test_global_anchor_config_rejects_base_deformation_mismatch():
    hyper = Namespace(
        no_grid=False,
        no_ds=False,
        no_dr=False,
        no_do=False,
        no_dshs=False,
        apply_rotation=False,
        kplanes_config={"resolution": [64, 64, 64, 100]},
        multires=[1, 2, 4, 8],
        defor_depth=0,
        net_width=32,
        timebase_pe=4,
        timenet_width=64,
        timenet_output=32,
        scale_rotation_pe=2,
        opacity_pe=2,
        bounds=1.6,
    )
    payload = {
        "config": {
            "hidden_params": dict(vars(hyper)),
        }
    }

    validate_global_anchor_config(hyper, payload)
    payload["config"]["hidden_params"]["net_width"] = 64

    with pytest.raises(ValueError, match="net_width"):
        validate_global_anchor_config(hyper, payload)


def test_endomoeg_pipeline_requires_absolute_bundle_paths(tmp_path):
    args = _build_default_args()
    args.endomoeg_pipeline_stage = "expert"
    args.endomoeg_expert_role = "local"
    args.endomoeg_bundle_dir = "relative/bundles"
    args.endomoeg_min_expert_psnr = 30.0

    with pytest.raises(ValueError, match="absolute"):
        validate_endomoeg_pipeline_args(args)

    args.endomoeg_bundle_dir = str(tmp_path.resolve())
    validated = validate_endomoeg_pipeline_args(args)

    assert validated.tracking_type == "endomoeg_expert"
    assert validated.endomoeg_expert_role == "local"
    assert validated.endomoeg_canonical_bundle == str(
        (tmp_path / "canonical.pth").resolve()
    )


def test_residual_expert_requires_positive_global_anchor_quality_gate(tmp_path):
    args = _build_default_args()
    args.endomoeg_pipeline_stage = "expert"
    args.endomoeg_expert_role = "contact"
    args.endomoeg_bundle_dir = str(tmp_path.resolve())

    with pytest.raises(ValueError, match="Global anchor"):
        validate_endomoeg_pipeline_args(args)


def test_endomoeg_pipeline_rejects_unknown_stage_and_role(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_endomoeg_pipeline_stage("continuous")

    args = _build_default_args()
    args.endomoeg_pipeline_stage = "expert"
    args.endomoeg_expert_role = "unknown"
    args.endomoeg_bundle_dir = str(tmp_path.resolve())
    with pytest.raises(ValueError, match="expert_role"):
        validate_endomoeg_pipeline_args(args)


def test_endomoeg_joint_uses_separate_absolute_output_directory(tmp_path):
    bundle_dir = (tmp_path / "bundles").resolve()
    args = _build_default_args()
    args.endomoeg_pipeline_stage = "joint"
    args.endomoeg_bundle_dir = str(bundle_dir)
    args.endomoeg_min_expert_psnr = 35.0

    validated = validate_endomoeg_pipeline_args(args)

    assert validated.tracking_type == "original"
    assert validated.endomoeg_joint_output_dir == str(
        (bundle_dir / "joint").resolve()
    )

    overwrite_args = _build_default_args()
    overwrite_args.endomoeg_pipeline_stage = "joint"
    overwrite_args.endomoeg_bundle_dir = str(bundle_dir)
    overwrite_args.endomoeg_joint_output_dir = str(bundle_dir)
    overwrite_args.endomoeg_min_expert_psnr = 35.0
    with pytest.raises(ValueError, match="must not overwrite"):
        validate_endomoeg_pipeline_args(overwrite_args)

    relative_router_args = _build_default_args()
    relative_router_args.endomoeg_pipeline_stage = "joint"
    relative_router_args.endomoeg_bundle_dir = str(bundle_dir)
    relative_router_args.endomoeg_router_bundle = "relative/router.pth"
    relative_router_args.endomoeg_min_expert_psnr = 35.0
    with pytest.raises(ValueError, match="router_bundle"):
        validate_endomoeg_pipeline_args(relative_router_args)

    missing_gate_args = _build_default_args()
    missing_gate_args.endomoeg_pipeline_stage = "router"
    missing_gate_args.endomoeg_bundle_dir = str(bundle_dir)
    with pytest.raises(ValueError, match="positive"):
        validate_endomoeg_pipeline_args(missing_gate_args)


def test_cams_gs_early_phases_preset_has_explicit_stage_boundaries():
    candidate = _load_preset_args("cutting_cams_gs_early_phases.py")
    assert candidate.tracking_type == "cams_gs"
    assert candidate.stage_global_only_end == 600
    assert candidate.stage_graph_bootstrap_end == 2700
    assert candidate.stage_local_motion_end == 3600
    assert candidate.stage_visibility_enable_iter == 6300
    assert candidate.stage_lifecycle_enable_iter == 7650

    scheduler = CAMSGSScheduler(candidate)
    assert scheduler.build(599, candidate.iterations).name == "global_only"
    assert scheduler.build(600, candidate.iterations).name == "graph_bootstrap"
    assert scheduler.build(2699, candidate.iterations).name == "graph_bootstrap"
    assert scheduler.build(2700, candidate.iterations).name == "local_motion_only"
    assert scheduler.build(3599, candidate.iterations).name == "local_motion_only"
    assert scheduler.build(3600, candidate.iterations).name == "motion_warmup"
    assert scheduler.build(6299, candidate.iterations).name == "motion_warmup"
    assert scheduler.build(6300, candidate.iterations).name == "visibility_refine"
    assert scheduler.build(7649, candidate.iterations).name == "visibility_refine"
    assert scheduler.build(7650, candidate.iterations).name == "joint_finetune"
