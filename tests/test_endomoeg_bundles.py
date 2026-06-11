from types import SimpleNamespace
import copy

import pytest
import torch
import torch.nn as nn

from models.endomoeg.expert_bundle import (
    build_canonical_bundle,
    build_expert_bundle,
    canonical_fingerprint,
    expert_fingerprint,
    load_canonical_bundle,
    load_expert_bundle,
    save_bundle,
    validate_expert_bundle,
)
from models.endomoeg.ensemble import (
    FrozenExpertEnsemble,
    assert_gaussian_model_frozen,
    freeze_gaussian_model,
)
from models.endomoeg.router_bundle import (
    load_router_bundle,
    save_router_bundle,
)
from scene.gaussian_model import GaussianModel


class _FakeDeformation(nn.Module):
    def __init__(self, tracking_type="original", arch_version="original_v1"):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.deformation_net = SimpleNamespace(
            tracking_mode=tracking_type,
            get_tracking_arch_version=lambda: arch_version,
            scene_scale=torch.tensor(10.0),
        )
        self.restored_scene_scale = None
        self.restored_aabb = None

    def set_scene_scale(self, value):
        self.restored_scene_scale = float(value)

    def set_aabb(self, xyz_max, xyz_min):
        self.restored_aabb = (
            torch.as_tensor(xyz_max).clone(),
            torch.as_tensor(xyz_min).clone(),
        )


def _build_gaussian_stub(
    tracking_type="original",
    arch_version="original_v1",
    point_count=3,
):
    model = GaussianModel.__new__(GaussianModel)
    model.active_sh_degree = 2
    model.max_sh_degree = 3
    model._xyz = nn.Parameter(
        torch.arange(point_count * 3, dtype=torch.float32).reshape(point_count, 3)
    )
    model._features_dc = nn.Parameter(torch.zeros(point_count, 3, 1))
    model._features_rest = nn.Parameter(torch.zeros(point_count, 3, 15))
    model._scaling = nn.Parameter(torch.zeros(point_count, 3))
    model._rotation = nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(point_count, 1)
    )
    model._opacity = nn.Parameter(torch.zeros(point_count, 1))
    model._deformation_table = torch.ones(point_count, dtype=torch.bool)
    model._deformation_accum = torch.full((point_count, 3), 0.25)
    model.max_radii2D = torch.zeros(point_count)
    model.xyz_gradient_accum = torch.zeros(point_count, 1)
    model.denom = torch.zeros(point_count, 1)
    model.percent_dense = 0.01
    model.spatial_lr_scale = 10.0
    model.optimizer = None
    model._deformation = _FakeDeformation(tracking_type, arch_version)
    return model


def test_canonical_bundle_round_trip_and_fingerprint(tmp_path):
    model = _build_gaussian_stub()
    payload = build_canonical_bundle(
        model,
        iteration=1000,
        config={"source_path": "/root/3DGS/data/endonerf/cutting"},
    )
    path = save_bundle(tmp_path / "canonical.pth", payload)
    loaded = load_canonical_bundle(path)

    assert loaded["iteration"] == 1000
    assert loaded["canonical_fingerprint"] == canonical_fingerprint(
        loaded["canonical_state"]
    )
    assert loaded["canonical_state"]["xyz"].device.type == "cpu"


def test_complete_expert_bundle_rejects_legacy_and_source_mismatch(tmp_path):
    model = _build_gaussian_stub(
        tracking_type="endomoeg_expert",
        arch_version="endomoeg_complete_global_v1",
    )
    canonical = build_canonical_bundle(model, iteration=1000)
    payload = build_expert_bundle(
        model,
        role="global",
        source_canonical_fingerprint=canonical["canonical_fingerprint"],
        iteration=9000,
        validation_metrics={"psnr": 38.5},
    )
    path = save_bundle(tmp_path / "global.pth", payload)

    loaded = load_expert_bundle(
        path,
        expected_role="global",
        expected_source_fingerprint=canonical["canonical_fingerprint"],
        minimum_psnr=37.0,
    )
    assert loaded["point_count"] == 3
    assert loaded["validation_metrics"]["psnr"] == pytest.approx(38.5)
    assert loaded["expert_state_fingerprint"] == expert_fingerprint(
        loaded["expert_state"]
    )

    with pytest.raises(ValueError, match="source canonical fingerprint"):
        load_expert_bundle(
            path,
            expected_source_fingerprint="different-canonical",
        )

    with pytest.raises(ValueError, match="Legacy residual component"):
        validate_expert_bundle(
            {
                "format": "legacy_component",
                "component": "global",
                "state_dict": {},
            }
        )

    tampered = copy.deepcopy(payload)
    tampered["expert_state"]["deformation"]["weight"].add_(1.0)
    with pytest.raises(ValueError, match="full-state fingerprint"):
        validate_expert_bundle(tampered)

    wrong_architecture = copy.deepcopy(payload)
    wrong_architecture["tracking_arch_version"] = "endomoeg_complete_global_v0"
    with pytest.raises(ValueError, match="requires tracking architecture"):
        validate_expert_bundle(wrong_architecture)


def test_complete_expert_state_round_trip_preserves_topology_and_deformation():
    source = _build_gaussian_stub(
        tracking_type="original",
        arch_version="original_v1",
        point_count=4,
    )
    source._deformation.weight.data.fill_(7.0)
    state = source.capture_expert_state()

    restored = _build_gaussian_stub(
        tracking_type="original",
        arch_version="original_v1",
        point_count=1,
    )
    restored.restore_expert_state(state)

    assert restored._xyz.shape == (4, 3)
    assert torch.equal(restored._xyz, source._xyz)
    assert torch.equal(restored._deformation_table, source._deformation_table)
    assert torch.equal(restored._deformation_accum, source._deformation_accum)
    assert torch.equal(restored._deformation.weight, source._deformation.weight)
    assert restored._deformation.restored_scene_scale == pytest.approx(10.0)
    assert torch.equal(
        restored._deformation.restored_aabb[0],
        source._xyz.detach().amax(dim=0),
    )
    assert torch.equal(
        restored._deformation.restored_aabb[1],
        source._xyz.detach().amin(dim=0),
    )


def test_expert_state_restore_rejects_architecture_mismatch():
    source = _build_gaussian_stub(
        tracking_type="original",
        arch_version="original_v1",
    )
    restored = _build_gaussian_stub(
        tracking_type="original",
        arch_version="different_v2",
    )

    with pytest.raises(ValueError, match="architecture"):
        restored.restore_expert_state(source.capture_expert_state())


def test_freeze_gaussian_model_disables_all_expert_gradients():
    model = _build_gaussian_stub()

    freeze_gaussian_model(model)
    assert_gaussian_model_frozen(model, "global")

    assert model.optimizer is None
    assert not model._xyz.requires_grad
    assert not model._features_dc.requires_grad
    assert all(
        not parameter.requires_grad
        for parameter in model._deformation.parameters()
    )


def test_assert_gaussian_model_frozen_detects_trainable_state():
    model = _build_gaussian_stub()

    with pytest.raises(RuntimeError, match="still has trainable"):
        assert_gaussian_model_frozen(model, "local")


def test_frozen_ensemble_requires_exact_global_canonical_lineage():
    experts = []
    payloads = []
    for role in ("global", "local", "contact"):
        model = freeze_gaussian_model(_build_gaussian_stub())
        experts.append((role, model))
        payloads.append(
            (
                role,
                {
                    "trained_canonical_fingerprint": "global-trained",
                    "expert_state": {
                        "canonical": {"active_sh_degree": 2},
                    },
                },
            )
        )

    FrozenExpertEnsemble(
        experts,
        payloads,
        source_canonical_fingerprint="source",
    )

    mismatched_payloads = copy.deepcopy(payloads)
    mismatched_payloads[1][1]["trained_canonical_fingerprint"] = "different"
    with pytest.raises(ValueError, match="trained Global canonical"):
        FrozenExpertEnsemble(
            experts,
            mismatched_payloads,
            source_canonical_fingerprint="source",
        )

    mismatched_sh = copy.deepcopy(payloads)
    mismatched_sh[2][1]["expert_state"]["canonical"]["active_sh_degree"] = 3
    with pytest.raises(ValueError, match="active SH degree"):
        FrozenExpertEnsemble(
            experts,
            mismatched_sh,
            source_canonical_fingerprint="source",
        )


def test_bundle_io_rejects_relative_paths():
    with pytest.raises(ValueError, match="absolute"):
        save_bundle("relative/canonical.pth", {})
    with pytest.raises(ValueError, match="absolute"):
        load_canonical_bundle("relative/canonical.pth")
    with pytest.raises(ValueError, match="absolute"):
        save_router_bundle("relative/router.pth", {})
    with pytest.raises(ValueError, match="absolute"):
        load_router_bundle("relative/router.pth")
