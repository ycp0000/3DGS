from types import SimpleNamespace

import pytest
import torch

from models.endomoeg.router import (
    EndoMoeVolumeAwareRouter,
    compute_router_losses,
    oracle_routing_targets,
    sparsify_router_weights,
)
from models.endomoeg.router_bundle import (
    build_router_bundle,
    validate_router_bundle,
)
from models.endomoeg import inference, router_training


def _routing_state(point_count):
    canonical = torch.linspace(-1.0, 1.0, point_count * 3).reshape(
        point_count,
        3,
    )
    return {
        "canonical_xyz": canonical,
        "means3d": canonical + 0.05,
        "motion": torch.full_like(canonical, 0.05),
        "opacity": torch.full((point_count, 1), 0.5),
        "scales": torch.full((point_count, 3), 0.1),
        "scene_scale": 10.0,
    }


def test_volume_router_supports_different_expert_point_counts():
    router = EndoMoeVolumeAwareRouter(
        {"global": 3, "local": 5, "contact": 7},
        gaussian_hidden_dim=16,
        pixel_hidden_dim=8,
    )
    camera = SimpleNamespace(camera_center=torch.zeros(3))

    for role, count in (("global", 3), ("local", 5), ("contact", 7)):
        logits = router.gaussian_logits(
            role,
            _routing_state(count),
            camera,
            time_value=0.25,
        )
        assert logits.shape == (count, 1)


def test_gaussian_router_parameters_receive_gradients():
    router = EndoMoeVolumeAwareRouter(
        {"global": 4, "local": 4, "contact": 4},
        gaussian_hidden_dim=16,
        pixel_hidden_dim=8,
    )
    logits = router.gaussian_logits(
        "local",
        _routing_state(4),
        SimpleNamespace(camera_center=torch.zeros(3)),
        time_value=0.5,
    )
    logits.square().mean().backward()

    assert router.base_logits["local"].grad is not None
    assert any(
        parameter.grad is not None
        for parameter in router.gaussian_feature_mlp.parameters()
    )


def test_pixel_router_and_top2_weights_receive_photometric_gradient():
    router = EndoMoeVolumeAwareRouter(
        {"global": 2, "local": 2, "contact": 2},
        gaussian_hidden_dim=8,
        pixel_hidden_dim=8,
    )
    expert_rgb = torch.rand(3, 3, 4, 5)
    weights, residual = router.route_pixels(
        expert_rgb=expert_rgb,
        expert_depth=torch.rand(3, 4, 5) + 0.5,
        gaussian_prior=torch.rand(3, 4, 5) + 0.1,
        projected_motion=torch.rand(3, 4, 5),
        coverage=torch.ones(3, 4, 5, dtype=torch.bool),
        top_k=2,
    )
    blended = (expert_rgb * weights.unsqueeze(1)).sum(dim=0)
    blended.square().mean().backward()

    assert weights.shape == (3, 4, 5)
    assert torch.allclose(weights.sum(dim=0), torch.ones(4, 5), atol=1e-6)
    assert (weights > 0).sum(dim=0).max().item() <= 2
    assert residual.shape == (3, 4, 5)
    final_layer = [
        module
        for module in router.pixel_router.score_network
        if isinstance(module, torch.nn.Conv2d)
    ][-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad).item() > 0


def test_oracle_targets_prefer_low_error_expert():
    ground_truth = torch.zeros(3, 2, 2)
    expert_rgb = torch.stack(
        (
            torch.zeros_like(ground_truth),
            torch.full_like(ground_truth, 0.5),
            torch.ones_like(ground_truth),
        ),
        dim=0,
    )
    targets = oracle_routing_targets(
        expert_rgb,
        ground_truth,
        temperature=0.05,
    )

    assert torch.all(targets[0] > targets[1])
    assert torch.all(targets[1] > targets[2])


def test_router_losses_penalize_starvation_without_uniform_target():
    ground_truth = torch.zeros(3, 2, 2)
    expert_rgb = torch.stack(
        (
            torch.zeros_like(ground_truth),
            torch.full_like(ground_truth, 0.2),
            torch.full_like(ground_truth, 0.4),
        ),
        dim=0,
    )
    collapsed = torch.zeros(3, 2, 2)
    collapsed[0] = 1.0
    _, _, losses = compute_router_losses(
        collapsed,
        expert_rgb,
        ground_truth,
        lambda_starvation=1.0,
    )

    assert losses["L_router_starvation"].item() > 0
    assert losses["router_usage_global"].item() == 1.0
    assert losses["router_usage_contact"].item() == 0.0


@pytest.mark.parametrize(
    "mask_transform",
    (
        lambda mask: mask,
        lambda mask: mask.unsqueeze(0),
        lambda mask: mask.unsqueeze(-1),
    ),
)
def test_router_losses_accept_supported_spatial_mask_layouts(mask_transform):
    ground_truth = torch.zeros(3, 1, 2)
    expert_rgb = torch.stack(
        (
            torch.zeros_like(ground_truth),
            torch.full_like(ground_truth, 0.2),
            torch.full_like(ground_truth, 0.4),
        ),
        dim=0,
    )
    weights = torch.full((3, 1, 2), 1.0 / 3.0)
    mask = mask_transform(
        torch.tensor(
            (
                (1.0, 0.0),
            )
        )
    )

    blended, targets, losses = compute_router_losses(
        weights,
        expert_rgb,
        ground_truth,
        mask=mask,
    )

    assert blended.shape == ground_truth.shape
    assert targets.shape == weights.shape
    assert all(torch.isfinite(value) for value in losses.values())


def test_sparsify_router_weights_preserves_dense_mode():
    weights = torch.softmax(torch.randn(3, 2, 2), dim=0)

    assert torch.equal(sparsify_router_weights(weights, None), weights)
    assert torch.equal(sparsify_router_weights(weights, 3), weights)


def test_sparse_router_uses_straight_through_dense_gradients():
    logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    weights = torch.softmax(logits, dim=0).reshape(3, 1, 1)
    sparse = sparsify_router_weights(weights, top_k=1)

    assert torch.count_nonzero(sparse).item() == 1
    coefficients = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1)
    (sparse * coefficients).sum().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 3


def test_router_bundle_binds_exact_expert_fingerprints_and_counts():
    router = EndoMoeVolumeAwareRouter(
        {"global": 2, "local": 3, "contact": 4},
        gaussian_hidden_dim=8,
        pixel_hidden_dim=8,
    )
    payloads = {}
    for role, count in (("global", 2), ("local", 3), ("contact", 4)):
        payloads[role] = {
            "expert_state_fingerprint": "{}-full-state".format(role),
            "trained_canonical_fingerprint": "{}-fingerprint".format(role),
            "point_count": count,
            "tracking_arch_version": "endomoeg_complete_{}_v1".format(role),
            "validation_metrics": {"psnr": 38.0},
        }
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
    assert bundle["point_counts"]["contact"] == 4

    changed_payloads = dict(payloads)
    changed_payloads["local"] = dict(payloads["local"])
    changed_payloads["local"]["expert_state_fingerprint"] = "replacement"
    changed_ensemble = SimpleNamespace(
        payloads=changed_payloads,
        source_canonical_fingerprint="shared-source",
    )
    try:
        validate_router_bundle(bundle, ensemble=changed_ensemble)
    except ValueError as exc:
        assert "full-state fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("Router accepted a replaced expert")

    architecture_payloads = dict(payloads)
    architecture_payloads["contact"] = dict(payloads["contact"])
    architecture_payloads["contact"]["tracking_arch_version"] = "replacement"
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_router_bundle(
            bundle,
            ensemble=SimpleNamespace(
                payloads=architecture_payloads,
                source_canonical_fingerprint="shared-source",
            ),
        )

    metric_payloads = dict(payloads)
    metric_payloads["global"] = dict(payloads["global"])
    metric_payloads["global"]["validation_metrics"] = {"psnr": 37.0}
    with pytest.raises(ValueError, match="validation PSNR mismatch"):
        validate_router_bundle(
            bundle,
            ensemble=SimpleNamespace(
                payloads=metric_payloads,
                source_canonical_fingerprint="shared-source",
            ),
        )


def test_frozen_ensemble_render_keeps_full_router_gradient_chain(monkeypatch):
    point_counts = {"global": 2, "local": 3, "contact": 4}
    router = EndoMoeVolumeAwareRouter(
        point_counts,
        gaussian_hidden_dim=8,
        pixel_hidden_dim=8,
    )
    experts = [
        (role, SimpleNamespace(role=role, active_sh_degree=0))
        for role in ("global", "local", "contact")
    ]
    ensemble = SimpleNamespace(__iter__=lambda self: iter(experts))
    role_values = {"global": 0.1, "local": 0.5, "contact": 0.9}

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
        count = point_counts[expert.role]
        state = _routing_state(count)
        state.update(
            {
                "means2d": torch.zeros(count, 3),
                "rotations": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]]
                ).repeat(count, 1),
                "covariance": None,
            }
        )
        value = role_values[expert.role]
        return {
            "render": torch.full((3, 2, 2), value),
            "depth": torch.full((1, 2, 2), value + 1.0),
            "routing_state": state,
        }

    def fake_rasterize(viewpoint, expert, pipe, routing_state, logits):
        del viewpoint, expert, pipe, routing_state
        prior = torch.sigmoid(logits).mean().expand(2, 2)
        return {
            "gaussian_prior": prior,
            "projected_motion": torch.full((2, 2), 0.1),
            "coverage": torch.ones(2, 2),
            "depth": torch.ones(1, 2, 2),
        }

    monkeypatch.setattr(router_training, "render", fake_render)
    monkeypatch.setattr(
        router_training,
        "rasterize_endomoeg_routing_features",
        fake_rasterize,
    )
    camera = SimpleNamespace(time=0.5, camera_center=torch.zeros(3))
    output = router_training.render_frozen_expert_ensemble(
        camera,
        experts,
        router,
        SimpleNamespace(),
        torch.zeros(3),
        top_k=None,
    )
    ground_truth = torch.full((3, 2, 2), 0.3)
    blended, _, losses = compute_router_losses(
        output["weights"],
        output["expert_rgb"],
        ground_truth,
    )
    losses["L_router_total"].backward()
    metrics = router_training.collect_router_gradient_metrics(router)

    assert blended.shape == (3, 2, 2)
    router_training.assert_router_gradient_contract(metrics)


def test_router_gradient_contract_rejects_disconnected_branch():
    metrics = {
        "grad_norm_router_gaussian_logits": 1.0,
        "grad_norm_router_feature_mlp": 0.5,
        "grad_norm_router_pixel": 0.0,
    }
    try:
        router_training.assert_router_gradient_contract(metrics)
    except RuntimeError as exc:
        assert "router_pixel" in str(exc)
    else:
        raise AssertionError("Disconnected pixel Router was accepted")


def test_frozen_router_assembly_uses_saved_architecture_and_freezes(
    monkeypatch,
    tmp_path,
):
    point_counts = {"global": 2, "local": 3, "contact": 4}
    trained_router = EndoMoeVolumeAwareRouter(
        point_counts,
        gaussian_hidden_dim=12,
        pixel_hidden_dim=10,
    )
    payload = {
        "iteration": 4000,
        "inference_top_k": 2,
        "config": {
            "hidden_params": {
                "moe_router_hidden_dim": 12,
                "moe_pixel_router_hidden_dim": 10,
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
    assert assembly.top_k == 2
    assert all(
        not parameter.requires_grad
        for parameter in assembly.router.parameters()
    )
    assert assembly.router.pixel_router.score_network[0].out_channels == 10


def test_frozen_router_assembly_rejects_relative_paths(tmp_path):
    try:
        inference.resolve_router_bundle_path("relative/bundles")
    except ValueError as exc:
        assert "absolute" in str(exc)
    else:
        raise AssertionError("Relative Router bundle directory was accepted")

    absolute_bundle_dir = tmp_path.resolve()
    try:
        inference.load_frozen_router_assembly(
            str(absolute_bundle_dir),
            expected_source_path="relative/scene",
            device="cpu",
        )
    except ValueError as exc:
        assert "source_path" in str(exc)
    else:
        raise AssertionError("Relative source_path was accepted")
