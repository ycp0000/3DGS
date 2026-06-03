import importlib.util
import sys
from argparse import ArgumentParser
from collections import UserDict
from pathlib import Path

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
