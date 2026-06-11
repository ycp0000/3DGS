from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import models.endomoeg.joint_training as joint_training
from models.endomoeg.joint_training import (
    _assert_joint_quality_gate,
    assert_joint_trainable_contract,
    capture_parameter_anchors,
    configure_joint_trainable_parameters,
    parameter_anchor_loss,
)
from models.endomoeg.router import EndoMoeVolumeAwareRouter


class _FakeCompleteHead(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.refinement = (
            nn.Linear(2, 2) if role in {"local", "contact"} else None
        )


class _FakeDeformationNet(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.complete_expert_head = _FakeCompleteHead(role)


class _FakeDeformation(nn.Module):
    def __init__(self, role):
        super().__init__()
        self.timenet = nn.Linear(2, 2)
        self.deformation_net = _FakeDeformationNet(role)


class _FakeExpert(nn.Module):
    def __init__(self, role):
        super().__init__()
        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ):
            setattr(self, name, nn.Parameter(torch.ones(2, 2)))
        self._deformation = _FakeDeformation(role)


class _FakeEnsemble:
    def __init__(self):
        self.experts = {
            role: _FakeExpert(role)
            for role in ("global", "local", "contact")
        }
        self.payloads = {
            role: {
                "expert_state_fingerprint": "{}-parent".format(role),
                "validation_metrics": {"psnr": 38.0},
                "config": {},
            }
            for role in self.experts
        }
        self.source_canonical_fingerprint = "shared-canonical"

    def __iter__(self):
        return iter(self.experts.items())

    def assert_frozen(self):
        for expert in self.experts.values():
            assert all(
                not parameter.requires_grad
                for parameter in expert.parameters()
            )


def _assembly():
    return SimpleNamespace(
        router=EndoMoeVolumeAwareRouter(
            {"global": 2, "local": 2, "contact": 2},
            gaussian_hidden_dim=8,
        ),
        ensemble=_FakeEnsemble(),
        payload={
            "validation_metrics": {"psnr": 39.0},
            "expert_manifest": {
                role: {
                    "expert_state_fingerprint": "{}-parent".format(role)
                }
                for role in ("global", "local", "contact")
            },
        },
        bundle_path="/absolute/parent/router.pth",
    )


def _hyper():
    return SimpleNamespace(
        endomoeg_joint_router_gaussian_lr=5e-4,
        endomoeg_joint_router_feature_lr=1e-4,
        endomoeg_joint_refinement_lr=5e-6,
    )


def test_joint_trainable_contract_freezes_canonical_and_role_backbones():
    assembly = _assembly()
    groups, expert_groups = configure_joint_trainable_parameters(
        assembly,
        _hyper(),
    )

    assert len(groups) == 4
    assert [group["name"] for group in groups[:2]] == [
        "joint_router_base_gates",
        "joint_router_feature_mlp",
    ]
    assert set(expert_groups) == {"local", "contact"}
    assert all(
        not getattr(expert, "_xyz").requires_grad
        for _, expert in assembly.ensemble
    )
    global_expert = assembly.ensemble.experts["global"]
    assert all(
        not parameter.requires_grad
        for parameter in global_expert._deformation.parameters()
    )
    for role in ("local", "contact"):
        expert = assembly.ensemble.experts[role]
        assert not expert._deformation.deformation_net.backbone.weight.requires_grad
        assert (
            expert._deformation.deformation_net.complete_expert_head
            .refinement.weight.requires_grad
        )
    assert_joint_trainable_contract(assembly)


def test_joint_anchor_penalizes_parameter_drift():
    assembly = _assembly()
    groups, _ = configure_joint_trainable_parameters(assembly, _hyper())
    anchors = capture_parameter_anchors(groups)

    assert parameter_anchor_loss(groups, anchors).item() == pytest.approx(0.0)
    groups[0]["params"][0].data.add_(1.0)
    assert parameter_anchor_loss(groups, anchors).item() > 0.0


def test_joint_quality_gate_protects_router_and_each_expert():
    assembly = _assembly()
    expert_metrics = {
        role: {"psnr": 38.0}
        for role in ("global", "local", "contact")
    }
    _assert_joint_quality_gate(
        assembly,
        ensemble_metrics={"psnr": 39.0},
        expert_metrics=expert_metrics,
        max_psnr_drop=0.05,
    )

    degraded = dict(expert_metrics)
    degraded["contact"] = {"psnr": 37.0}
    with pytest.raises(RuntimeError, match="contact"):
        _assert_joint_quality_gate(
            assembly,
            ensemble_metrics={"psnr": 39.0},
            expert_metrics=degraded,
            max_psnr_drop=0.05,
        )

    with pytest.raises(RuntimeError, match="Router PSNR"):
        _assert_joint_quality_gate(
            assembly,
            ensemble_metrics={"psnr": 38.0},
            expert_metrics=expert_metrics,
            max_psnr_drop=0.05,
        )


def test_joint_save_rebinds_router_to_updated_expert_states(
    monkeypatch,
    tmp_path,
):
    assembly = _assembly()
    saved_paths = []
    captured = {}

    def fake_build_expert_bundle(
        expert,
        role,
        source_canonical_fingerprint,
        iteration,
        config,
        validation_metrics,
    ):
        del expert, iteration, config
        assert source_canonical_fingerprint == "shared-canonical"
        return {
            "expert_state_fingerprint": "{}-joint".format(role),
            "trained_canonical_fingerprint": "{}-canonical".format(role),
            "point_count": 2,
            "tracking_arch_version": "{}-arch".format(role),
            "validation_metrics": validation_metrics,
        }

    def fake_build_router_bundle(router, ensemble, **kwargs):
        del router
        captured["fingerprints"] = {
            role: payload["expert_state_fingerprint"]
            for role, payload in ensemble.payloads.items()
        }
        captured["router_kwargs"] = kwargs
        return {"router": "joint"}

    monkeypatch.setattr(
        joint_training,
        "build_expert_bundle",
        fake_build_expert_bundle,
    )
    monkeypatch.setattr(
        joint_training,
        "build_router_bundle",
        fake_build_router_bundle,
    )
    monkeypatch.setattr(
        joint_training,
        "save_bundle",
        lambda path, payload: saved_paths.append((path, payload)) or path,
    )
    monkeypatch.setattr(
        joint_training,
        "save_router_bundle",
        lambda path, payload: saved_paths.append((path, payload)) or path,
    )
    output_dir = str((tmp_path / "joint").resolve())
    metrics = {
        role: {"psnr": 38.1}
        for role in ("global", "local", "contact")
    }

    router_path = joint_training._save_joint_assembly(
        assembly,
        output_dir=output_dir,
        iteration=500,
        config={"stage": "joint"},
        ensemble_metrics={"psnr": 39.1},
        expert_metrics=metrics,
    )

    assert router_path.endswith("router.pth")
    assert captured["fingerprints"] == {
        "global": "global-joint",
        "local": "local-joint",
        "contact": "contact-joint",
    }
    assert "inference_top_k" not in captured["router_kwargs"]
    assert len(saved_paths) == 4
    assert all(
        not parameter.requires_grad
        for parameter in assembly.router.parameters()
    )
