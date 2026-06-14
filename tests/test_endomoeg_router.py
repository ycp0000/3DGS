from types import SimpleNamespace

import pytest
import torch

from models.endomoeg import inference, router_training
from models.endomoeg.router import (
    EndoMoeVolumeAwareRouter,
    compose_residual_gaussian_state,
    compute_router_losses,
    incremental_gain_targets,
)
from models.endomoeg.router_bundle import (
    build_router_bundle,
    validate_router_bundle,
)


def _routing_state(
    point_count=3,
    mean_offset=0.0,
    opacity=0.5,
    color=0.25,
):
    canonical = torch.linspace(-1.0, 1.0, point_count * 3).reshape(
        point_count,
        3,
    )
    return {
        "canonical_xyz": canonical,
        "means3d": canonical + float(mean_offset),
        "motion": torch.full_like(canonical, float(mean_offset)),
        "opacity": torch.full((point_count, 1), float(opacity)),
        "scales": torch.full((point_count, 3), 0.1),
        "rotations": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]]
        ).repeat(point_count, 1),
        "colors": torch.full((point_count, 3), float(color)),
        "scene_scale": 10.0,
        "base_point_count": point_count,
        "auxiliary_point_count": 0,
    }


def _camera(time=0.5):
    return SimpleNamespace(
        camera_center=torch.zeros(3),
        time=float(time),
    )


def test_residual_router_requires_shared_parent_cloud_size():
    with pytest.raises(ValueError, match="identical parent point counts"):
        EndoMoeVolumeAwareRouter(
            {"global": 3, "local": 4, "contact": 3},
            gaussian_hidden_dim=16,
        )


def test_residual_gates_start_exactly_zero_and_receive_gradients():
    router = EndoMoeVolumeAwareRouter(
        {"global": 4, "local": 4, "contact": 4},
        gaussian_hidden_dim=16,
    )
    gates, raw_gates = router.residual_gates(
        _routing_state(4, mean_offset=0.0),
        _routing_state(4, mean_offset=0.1),
        _routing_state(4, mean_offset=-0.1),
        _camera(),
        time_value=0.5,
    )

    assert torch.count_nonzero(gates).item() == 0
    assert torch.count_nonzero(raw_gates).item() == 0
    gates.sum().backward()
    assert router.base_gates.grad is not None
    assert torch.count_nonzero(router.base_gates.grad).item() == 8
    final_layer = router.gaussian_feature_mlp[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad).item() > 0


def test_contact_auxiliary_activity_is_visible_to_router_features():
    router = EndoMoeVolumeAwareRouter(
        {"global": 2, "local": 2, "contact": 2},
        gaussian_hidden_dim=1,
    )
    global_state = _routing_state(2)
    local_state = _routing_state(2)
    inactive_contact = _routing_state(2)
    active_contact = _routing_state(2)
    active_contact["opacity"] = torch.cat(
        (active_contact["opacity"], torch.tensor(((0.8,),))),
        dim=0,
    )
    active_contact["auxiliary_point_count"] = 1
    active_contact["auxiliary_parent_indices"] = torch.tensor((1,))
    with torch.no_grad():
        first, second, output = (
            router.gaussian_feature_mlp[0],
            router.gaussian_feature_mlp[2],
            router.gaussian_feature_mlp[4],
        )
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, -5] = 1.0
        second.weight.fill_(1.0)
        second.bias.zero_()
        output.weight.zero_()
        output.bias.zero_()
        output.weight[1, 0] = 1.0

    inactive_gates, _ = router.residual_gates(
        global_state,
        local_state,
        inactive_contact,
        _camera(),
        time_value=0.5,
    )
    active_gates, _ = router.residual_gates(
        global_state,
        local_state,
        active_contact,
        _camera(),
        time_value=0.5,
    )

    assert inactive_gates[1, 1].item() == pytest.approx(0.0)
    assert active_gates[1, 1].item() > 0.0
    assert active_gates[0, 1].item() == pytest.approx(0.0)


def test_incremental_gain_targets_prefer_candidate_that_improves_global():
    ground_truth = torch.zeros(3, 2, 2)
    global_rgb = torch.full_like(ground_truth, 0.4)
    improved = torch.full_like(ground_truth, 0.1)
    degraded = torch.full_like(ground_truth, 0.8)

    improved_target = incremental_gain_targets(
        global_rgb,
        improved,
        ground_truth,
        temperature=0.02,
    )
    degraded_target = incremental_gain_targets(
        global_rgb,
        degraded,
        ground_truth,
        temperature=0.02,
    )

    assert torch.all(improved_target > 0.5)
    assert torch.all(degraded_target < 0.5)


def test_zero_residual_gates_reproduce_global_state_exactly():
    global_state = _routing_state(3, mean_offset=0.0, opacity=0.4, color=1.2)
    local_state = _routing_state(3, mean_offset=0.3, opacity=0.9, color=0.9)
    contact_state = _routing_state(
        3,
        mean_offset=-0.2,
        opacity=0.8,
        color=0.7,
    )

    composed = compose_residual_gaussian_state(
        global_state,
        local_state,
        contact_state,
        torch.zeros(3, 2),
    )

    for name in ("means3d", "scales", "rotations", "opacity", "colors"):
        assert torch.equal(composed[name], global_state[name])


def test_local_and_contact_gates_compose_parent_and_auxiliary_states():
    global_state = _routing_state(2, mean_offset=0.0, opacity=0.4, color=0.2)
    local_state = _routing_state(2, mean_offset=0.5, opacity=0.4, color=0.2)
    contact_state = _routing_state(2, mean_offset=0.25, opacity=0.8, color=0.6)
    contact_state["means3d"] = torch.cat(
        (contact_state["means3d"], torch.tensor([[2.0, 2.0, 2.0]])),
        dim=0,
    )
    contact_state["scales"] = torch.cat(
        (contact_state["scales"], torch.full((1, 3), 0.05)),
        dim=0,
    )
    contact_state["rotations"] = torch.cat(
        (
            contact_state["rotations"],
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ),
        dim=0,
    )
    contact_state["opacity"] = torch.cat(
        (contact_state["opacity"], torch.tensor([[0.9]])),
        dim=0,
    )
    contact_state["colors"] = torch.cat(
        (contact_state["colors"], torch.tensor([[0.9, 0.8, 0.7]])),
        dim=0,
    )
    contact_state["auxiliary_point_count"] = 1
    contact_state["auxiliary_parent_indices"] = torch.tensor([1])
    gates = torch.tensor(((1.0, 0.0), (0.5, 1.0)))

    composed = compose_residual_gaussian_state(
        global_state,
        local_state,
        contact_state,
        gates,
    )

    assert torch.allclose(
        composed["means3d"][0],
        local_state["means3d"][0],
    )
    expected_parent = (
        global_state["means3d"][1]
        + 0.5
        * (local_state["means3d"][1] - global_state["means3d"][1])
        + (contact_state["means3d"][1] - global_state["means3d"][1])
    )
    assert torch.allclose(composed["means3d"][1], expected_parent)
    expected_child = (
        contact_state["means3d"][2]
        + 0.5
        * (local_state["means3d"][1] - global_state["means3d"][1])
    )
    assert torch.allclose(composed["means3d"][2], expected_child)
    assert composed["opacity"][2].item() == pytest.approx(0.9)
    assert composed["auxiliary_point_count"] == 1


def test_contact_gate_does_not_cancel_local_parent_rotation():
    global_state = _routing_state(1)
    local_state = _routing_state(1)
    contact_state = _routing_state(1)
    half_angle = torch.tensor(torch.pi / 8.0)
    local_rotation = torch.tensor(
        [[torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]]
    )
    local_state["rotations"] = local_rotation
    contact_state["means3d"] = torch.cat(
        (contact_state["means3d"], torch.tensor([[0.0, 0.0, 0.0]])),
        dim=0,
    )
    contact_state["scales"] = torch.cat(
        (contact_state["scales"], torch.full((1, 3), 0.05)),
        dim=0,
    )
    contact_state["rotations"] = torch.cat(
        (contact_state["rotations"], contact_state["rotations"]),
        dim=0,
    )
    contact_state["opacity"] = torch.cat(
        (contact_state["opacity"], torch.ones(1, 1)),
        dim=0,
    )
    contact_state["colors"] = torch.cat(
        (contact_state["colors"], torch.ones(1, 3)),
        dim=0,
    )
    contact_state["auxiliary_point_count"] = 1
    contact_state["auxiliary_parent_indices"] = torch.tensor([0])

    composed = compose_residual_gaussian_state(
        global_state,
        local_state,
        contact_state,
        torch.ones(1, 2),
    )

    assert torch.allclose(
        composed["rotations"][0],
        local_rotation[0],
        atol=1e-6,
    )
    assert torch.allclose(
        composed["rotations"][1],
        local_rotation[0],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "mask_transform",
    (
        lambda mask: mask,
        lambda mask: mask.unsqueeze(0),
        lambda mask: mask.unsqueeze(-1),
    ),
)
def test_router_losses_accept_masks_and_penalize_regret(mask_transform):
    ground_truth = torch.zeros(3, 1, 2)
    global_rgb = torch.full_like(ground_truth, 0.2)
    composite_rgb = torch.full_like(ground_truth, 0.4, requires_grad=True)
    candidate_rgb = {
        "local": torch.full_like(ground_truth, 0.1),
        "contact": torch.full_like(ground_truth, 0.3),
    }
    gates = torch.full((3, 2), 0.25, requires_grad=True)
    gate_maps = torch.full((2, 1, 2), 0.25, requires_grad=True)
    mask = mask_transform(torch.tensor(((1.0, 0.0),)))

    losses = compute_router_losses(
        composite_rgb,
        global_rgb,
        candidate_rgb,
        gates,
        gate_maps,
        ground_truth,
        mask=mask,
        lambda_no_regret=1.0,
    )

    assert losses["L_router_no_regret"].item() > 0
    assert losses["router_target_local"].item() > 0.5
    assert losses["router_target_contact"].item() < 0.5
    assert all(
        torch.isfinite(value)
        for value in losses.values()
        if torch.is_tensor(value)
    )
    losses["L_router_total"].backward()
    assert composite_rgb.grad is not None
    assert gates.grad is not None
    assert gate_maps.grad is not None


def test_gain_supervision_opens_exact_zero_local_gate_without_sparse_bias():
    ground_truth = torch.zeros(3, 1, 1)
    global_rgb = torch.full_like(ground_truth, 0.4)
    composite_rgb = global_rgb.clone().requires_grad_(True)
    candidate_rgb = {
        "local": torch.full_like(ground_truth, 0.1),
        "contact": torch.full_like(ground_truth, 0.8),
    }
    gates = torch.zeros(2, 2, requires_grad=True)
    gate_maps = torch.zeros(2, 1, 1, requires_grad=True)

    losses = compute_router_losses(
        composite_rgb,
        global_rgb,
        candidate_rgb,
        gates,
        gate_maps,
        ground_truth,
        lambda_sparsity=1.0,
    )
    losses["L_router_total"].backward()

    assert gate_maps.grad[0].item() < 0.0
    assert gate_maps.grad[1].item() > 0.0
    assert torch.count_nonzero(gates.grad).item() == 0


def _expert_payloads(point_count=2):
    architectures = {
        "global": "endomoeg_complete_global_v1",
        "local": "endomoeg_complete_local_v5",
        "contact": "endomoeg_complete_contact_v4",
    }
    return {
        role: {
            "expert_state_fingerprint": "{}-full-state".format(role),
            "trained_canonical_fingerprint": "{}-fingerprint".format(role),
            "point_count": point_count,
            "tracking_arch_version": architectures[role],
            "validation_metrics": {"psnr": 38.0},
        }
        for role in ("global", "local", "contact")
    }


def test_router_bundle_binds_residual_experts_without_top_k_protocol():
    router = EndoMoeVolumeAwareRouter(
        {"global": 2, "local": 2, "contact": 2},
        gaussian_hidden_dim=8,
    )
    payloads = _expert_payloads()
    ensemble = SimpleNamespace(
        payloads=payloads,
        source_canonical_fingerprint="shared-source",
    )
    bundle = build_router_bundle(
        router,
        ensemble,
        iteration=4000,
        validation_metrics={"psnr": 39.0},
    )

    validate_router_bundle(bundle, ensemble=ensemble)
    assert "inference_top_k" not in bundle
    assert tuple(bundle["point_counts"].values()) == (2, 2, 2)

    version_five = dict(bundle)
    version_five["version"] = 5
    with pytest.raises(ValueError, match="Unsupported Router bundle version"):
        validate_router_bundle(version_five, ensemble=ensemble)

    replaced = _expert_payloads()
    replaced["local"]["expert_state_fingerprint"] = "replacement"
    with pytest.raises(ValueError, match="full-state fingerprint mismatch"):
        validate_router_bundle(
            bundle,
            ensemble=SimpleNamespace(
                payloads=replaced,
                source_canonical_fingerprint="shared-source",
            ),
        )


def test_frozen_ensemble_render_keeps_residual_gradient_chain(monkeypatch):
    point_count = 2
    router = EndoMoeVolumeAwareRouter(
        {"global": point_count, "local": point_count, "contact": point_count},
        gaussian_hidden_dim=8,
    )
    experts = [
        (role, SimpleNamespace(role=role, active_sh_degree=0))
        for role in ("global", "local", "contact")
    ]
    offsets = {"global": 0.0, "local": 0.4, "contact": -0.2}

    def fake_render(
        viewpoint,
        expert,
        pipe,
        background,
        stage,
        update_deformation_stats,
        return_routing_state,
    ):
        del (
            viewpoint,
            pipe,
            background,
            stage,
            update_deformation_stats,
            return_routing_state,
        )
        state = _routing_state(
            point_count,
            mean_offset=offsets[expert.role],
        )
        return {
            "render": torch.full((3, 2, 2), offsets[expert.role] + 0.5),
            "depth": torch.ones(1, 2, 2),
            "routing_state": state,
        }

    def fake_composite(
        viewpoint,
        expert,
        pipe,
        background,
        composite_state,
    ):
        del viewpoint, expert, pipe, background
        value = composite_state["means3d"].mean()
        return {
            "render": value.expand(3, 2, 2),
            "depth": value.expand(1, 2, 2),
        }

    def fake_gate_projection(
        viewpoint,
        expert,
        pipe,
        routing_state,
        values,
        probabilities,
    ):
        del viewpoint, expert, pipe, routing_state
        assert probabilities is True
        return {"gaussian_prior": values.mean().expand(2, 2)}

    monkeypatch.setattr(router_training, "render", fake_render)
    monkeypatch.setattr(
        router_training,
        "rasterize_endomoeg_composite_state",
        fake_composite,
    )
    monkeypatch.setattr(
        router_training,
        "rasterize_endomoeg_routing_features",
        fake_gate_projection,
    )
    output = router_training.render_frozen_expert_ensemble(
        _camera(),
        experts,
        router,
        SimpleNamespace(),
        torch.zeros(3),
    )
    output["render"].mean().backward()
    metrics = router_training.collect_router_gradient_metrics(router)

    assert output["render"].shape == (3, 2, 2)
    assert output["gate_maps"].shape == (2, 2, 2)
    router_training.assert_router_gradient_contract(metrics)


def test_router_gradient_contract_rejects_disconnected_branch():
    with pytest.raises(RuntimeError, match="feature_mlp"):
        router_training.assert_router_gradient_contract(
            {
                "grad_norm_router_base_gates": 1.0,
                "grad_norm_router_feature_mlp": 0.0,
            }
        )


def test_router_headroom_requires_real_incremental_capacity():
    router_training.assert_router_headroom(
        {"global": 38.0, "oracle": 38.5},
        minimum_gain=0.3,
    )
    with pytest.raises(RuntimeError, match="oracle headroom"):
        router_training.assert_router_headroom(
            {"global": 38.0, "oracle": 38.1},
            minimum_gain=0.3,
        )


def test_frozen_router_assembly_uses_saved_architecture_and_freezes(
    monkeypatch,
    tmp_path,
):
    point_counts = {"global": 2, "local": 2, "contact": 2}
    trained_router = EndoMoeVolumeAwareRouter(
        point_counts,
        gaussian_hidden_dim=12,
    )
    payload = {
        "iteration": 4000,
        "config": {
            "hidden_params": {
                "moe_router_hidden_dim": 12,
            }
        },
        "router_state": trained_router.state_dict(),
    }
    fake_ensemble = SimpleNamespace(
        point_counts=lambda: point_counts,
        assert_frozen=lambda: None,
    )
    monkeypatch.setattr(
        inference,
        "FrozenExpertEnsemble",
        SimpleNamespace(load=lambda *args, **kwargs: fake_ensemble),
    )
    monkeypatch.setattr(
        inference,
        "load_router_bundle",
        lambda *args, **kwargs: payload,
    )
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    (bundle_dir / "router.pth").touch()
    source_path = tmp_path / "scene"
    source_path.mkdir()

    assembly = inference.load_frozen_router_assembly(
        str(bundle_dir.resolve()),
        expected_source_path=str(source_path.resolve()),
        device="cpu",
    )

    assert assembly.iteration == 4000
    assert assembly.router.parent_count == 2
    assert assembly.router.gaussian_feature_mlp[0].out_features == 12
    assert all(
        not parameter.requires_grad
        for parameter in assembly.router.parameters()
    )


def test_frozen_router_assembly_rejects_relative_paths(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        inference.resolve_router_bundle_path("relative/bundles")

    with pytest.raises(ValueError, match="source_path"):
        inference.load_frozen_router_assembly(
            str(tmp_path.resolve()),
            expected_source_path="relative/scene",
            device="cpu",
        )
