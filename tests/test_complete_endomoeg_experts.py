from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.endomoeg.complete_expert import (
    CompleteEndoMoeExpert,
    CompleteExpertScheduler,
)
from scene.deformation import deform_network


def _canonical_inputs(count=5):
    means = torch.randn(count, 3)
    scales = torch.zeros(count, 3)
    rotations = torch.zeros(count, 4)
    rotations[:, 0] = 1.0
    opacity = torch.zeros(count, 1)
    return means, scales, rotations, opacity


def _deformation_args(role):
    return SimpleNamespace(
        tracking_type="endomoeg_expert",
        endomoeg_expert_role=role,
        endomoeg_expert_hidden_dim=16,
        net_width=16,
        timebase_pe=2,
        defor_depth=1,
        timenet_width=16,
        timenet_output=8,
        scale_rotation_pe=0,
        no_grid=True,
        no_ds=False,
        no_dr=False,
        no_do=False,
        bounds=1.6,
        kplanes_config={
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 8,
            "resolution": [8, 8, 8, 4],
        },
        multires=[1],
        camera_extent=1.0,
        max_disp_smooth_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_rot_smooth=0.05,
        max_rot_local=0.05,
        max_scale_smooth=0.05,
        max_scale_local=0.05,
        max_opacity_delta=4.0,
        current_iteration=0,
        iterations=100,
    )


@pytest.mark.parametrize("role", ("global", "local", "contact"))
def test_complete_expert_roles_preserve_full_scene_output_contract(role):
    means, scales, rotations, opacity = _canonical_inputs()
    expert = CompleteEndoMoeExpert(
        role=role,
        time_feature_dim=8,
        hidden_dim=16,
    )
    outputs = expert(
        canonical_means3d=means,
        canonical_scales=scales,
        canonical_rotations=rotations,
        canonical_opacity=opacity,
        base_means3d=means + 0.1,
        base_scales=scales,
        base_rotations=rotations,
        base_opacity=opacity,
        time_values=torch.rand(means.shape[0], 1),
        scene_scale=torch.tensor(10.0),
    )
    means_out, scales_out, rotations_out, opacity_out, aux = outputs

    assert means_out.shape == means.shape
    assert scales_out.shape == scales.shape
    assert rotations_out.shape == rotations.shape
    assert opacity_out.shape == opacity.shape
    assert aux["expert_role"] == role
    assert aux["pi_geo"].shape == (means.shape[0], 1)
    assert torch.allclose(aux["pi_geo"], torch.ones_like(aux["pi_geo"]))
    assert torch.allclose(aux["d_mu"], means_out - means)
    assert aux["global_motion_norm"].shape == (means.shape[0],)
    if role == "local":
        assert aux["local_motion_norm"].shape == (means.shape[0],)
        assert "cut_graph_motion_norm" not in aux
    elif role == "contact":
        assert aux["cut_graph_motion_norm"].shape == (means.shape[0],)
        assert "local_motion_norm" not in aux
    else:
        assert "local_motion_norm" not in aux
        assert "cut_graph_motion_norm" not in aux


def test_global_expert_uses_backbone_without_time_only_residual():
    means, scales, rotations, opacity = _canonical_inputs()
    base_means = means + torch.tensor([0.1, -0.2, 0.3])
    expert = CompleteEndoMoeExpert(role="global", time_feature_dim=8)

    means_out, _, _, _, aux = expert(
        canonical_means3d=means,
        canonical_scales=scales,
        canonical_rotations=rotations,
        canonical_opacity=opacity,
        base_means3d=base_means,
        base_scales=scales,
        base_rotations=rotations,
        base_opacity=opacity,
        time_values=torch.rand(means.shape[0], 1),
        scene_scale=torch.tensor(10.0),
    )

    assert torch.equal(means_out, base_means)
    assert torch.count_nonzero(aux["d_mu_refinement"]).item() == 0
    assert expert.named_parameter_groups() == {}


def test_local_and_contact_refinement_parameters_receive_gradients():
    for role in ("local", "contact"):
        means, scales, rotations, opacity = _canonical_inputs()
        expert = CompleteEndoMoeExpert(
            role=role,
            time_feature_dim=8,
            hidden_dim=16,
        )
        means_out, scales_out, rotations_out, opacity_out, _ = expert(
            canonical_means3d=means,
            canonical_scales=scales,
            canonical_rotations=rotations,
            canonical_opacity=opacity,
            base_means3d=means,
            base_scales=scales,
            base_rotations=rotations,
            base_opacity=opacity,
            time_values=torch.rand(means.shape[0], 1),
            scene_scale=torch.tensor(10.0),
        )
        loss = (
            means_out.square().mean()
            + scales_out.square().mean()
            + rotations_out.square().mean()
            + opacity_out.square().mean()
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in expert.parameters()
            if parameter.requires_grad
        ]
        assert any(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in gradients
        )


def test_deform_network_builds_complete_expert_and_reports_role_version():
    model = deform_network(_deformation_args("local"))

    assert model.deformation_net.tracking_mode == "endomoeg_expert"
    assert model.deformation_net.complete_expert_head.role == "local"
    assert (
        model.deformation_net.get_tracking_arch_version()
        == "endomoeg_complete_local_v1"
    )
    assert "tracking_expert_refinement" in model.get_tracking_parameter_groups()


@pytest.mark.parametrize("role", ("global", "local", "contact"))
def test_complete_expert_dynamic_start_is_exactly_canonical(role):
    network = deform_network(_deformation_args(role))
    deformation = network.deformation_net
    for head in (
        deformation.pos_deform,
        deformation.scales_deform,
        deformation.rotations_deform,
        deformation.opacity_deform,
    ):
        linear_layers = [
            module for module in head.modules() if isinstance(module, nn.Linear)
        ]
        assert torch.count_nonzero(linear_layers[-1].weight).item() == 0
        assert torch.count_nonzero(linear_layers[-1].bias).item() == 0

    means, scales, rotations, opacity = _canonical_inputs()
    outputs = network(
        means,
        scales,
        rotations,
        opacity,
        torch.rand(means.shape[0], 1),
    )

    assert torch.allclose(outputs[0], means)
    assert torch.allclose(outputs[1], scales)
    assert torch.allclose(outputs[2], rotations)
    assert torch.allclose(outputs[3], opacity)


def test_complete_expert_scheduler_enables_visibility_only_for_contact():
    local_phase = CompleteExpertScheduler("local").build(10, 100)
    contact_phase = CompleteExpertScheduler("contact").build(10, 100)

    assert local_phase.force_geo_expert == "local"
    assert not local_phase.enable_visibility
    assert local_phase.is_group_trainable("tracking_time_encoder")
    assert local_phase.is_group_trainable("tracking_expert_refinement")
    assert contact_phase.force_geo_expert == "contact"
    assert contact_phase.enable_visibility
    assert contact_phase.active_vis == 2
