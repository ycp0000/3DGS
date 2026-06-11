from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models.endomoeg.complete_expert import (
    CompleteEndoMoeExpert,
    CompleteExpertScheduler,
)
from models.endomoeg.motion_scaffold import MotionScaffoldLocalExpert
from scene.deformation import deform_network
from scene.gaussian_model import GaussianModel
from scene.tracking_losses import compute_tracking_losses


def _canonical_inputs(count=5):
    means = torch.randn(count, 3)
    scales = torch.zeros(count, 3)
    rotations = torch.zeros(count, 4)
    rotations[:, 0] = 1.0
    opacity = torch.zeros(count, 1)
    return means, scales, rotations, opacity


def _initialize_expert(expert, means, rotations):
    expert.initialize_from_canonical(means, rotations)


def _assign_gaussian_state(model, point_count=6):
    means, scales, rotations, opacity = _canonical_inputs(point_count)
    model.active_sh_degree = 0
    model._xyz = nn.Parameter(means.clone())
    model._features_dc = nn.Parameter(torch.zeros(point_count, 3, 1))
    model._features_rest = nn.Parameter(torch.zeros(point_count, 3, 15))
    model._scaling = nn.Parameter(scales.clone())
    model._rotation = nn.Parameter(rotations.clone())
    model._opacity = nn.Parameter(opacity.clone())
    model._deformation_table = torch.ones(point_count, dtype=torch.bool)
    model._deformation_accum = torch.zeros(point_count, 3)
    model.max_radii2D = torch.zeros(point_count)
    model.xyz_gradient_accum = torch.zeros(point_count, 1)
    model.denom = torch.zeros(point_count, 1)
    model.percent_dense = 0.01
    model.spatial_lr_scale = 1.0
    model._deformation.set_scene_scale(2.0)
    model._deformation.set_aabb(means.amax(dim=0), means.amin(dim=0))


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
    _initialize_expert(expert, means, rotations)
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
    _initialize_expert(expert, means, rotations)

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


def test_endogaussian_backbone_uses_unbounded_raw_residuals():
    network = deform_network(_deformation_args("global"))
    deformation = network.deformation_net
    hidden = torch.zeros(2, deformation.W)
    means, scales, rotations, opacity = _canonical_inputs(count=2)
    expected = {
        deformation.pos_deform: torch.tensor((0.4, -0.3, 0.2)),
        deformation.scales_deform: torch.tensor((0.2, -0.1, 0.3)),
        deformation.rotations_deform: torch.tensor((0.4, -0.2, 0.1, 0.3)),
        deformation.opacity_deform: torch.tensor((5.0,)),
    }
    with torch.no_grad():
        for head, bias in expected.items():
            linear_layers = [
                module for module in head.modules() if isinstance(module, nn.Linear)
            ]
            linear_layers[-1].weight.zero_()
            linear_layers[-1].bias.copy_(bias)

    outputs = deformation._forward_original(
        hidden,
        means,
        scales,
        rotations,
        opacity,
    )

    assert torch.allclose(outputs[0], means + expected[deformation.pos_deform])
    assert torch.allclose(outputs[1], scales + expected[deformation.scales_deform])
    assert torch.allclose(
        outputs[2],
        rotations + expected[deformation.rotations_deform],
    )
    assert torch.allclose(
        outputs[3],
        opacity + expected[deformation.opacity_deform],
    )


def test_local_and_contact_refinement_parameters_receive_gradients():
    for role in ("local", "contact"):
        means, scales, rotations, opacity = _canonical_inputs()
        expert = CompleteEndoMoeExpert(
            role=role,
            time_feature_dim=8,
            hidden_dim=16,
        )
        _initialize_expert(expert, means, rotations)
        means_out, scales_out, rotations_out, opacity_out, aux = expert(
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
        if "auxiliary_opacity" in aux:
            loss = loss + aux["auxiliary_opacity"].mean()
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
        == "endomoeg_complete_local_v3"
    )
    assert "tracking_expert_refinement" in model.get_tracking_parameter_groups()


def test_complete_expert_construction_uses_configured_geometry_bounds():
    local_args = _deformation_args("local")
    local_args.endomoeg_scaffold_max_node_offset_ratio = 0.012
    local_args.endomoeg_scaffold_max_radius_scale = 2.5
    local_network = deform_network(local_args)
    local_refinement = (
        local_network.deformation_net.complete_expert_head.refinement
    )

    assert local_refinement.max_node_offset_ratio == pytest.approx(0.012)
    assert local_refinement.max_radius_scale == pytest.approx(2.5)

    contact_args = _deformation_args("contact")
    contact_args.endomoeg_contact_max_spatial_offset_ratio = 0.011
    contact_args.endomoeg_contact_max_velocity_ratio = 0.021
    contact_args.endomoeg_contact_max_acceleration_ratio = 0.031
    contact_args.endomoeg_contact_max_rotation_radians = 0.41
    contact_args.endomoeg_contact_max_scale_delta = 0.09
    contact_args.endomoeg_contact_initial_duration = 0.12
    contact_network = deform_network(contact_args)
    contact_refinement = (
        contact_network.deformation_net.complete_expert_head.refinement
    )

    assert contact_refinement.max_spatial_offset_ratio == pytest.approx(0.011)
    assert contact_refinement.max_velocity_ratio == pytest.approx(0.021)
    assert contact_refinement.max_acceleration_ratio == pytest.approx(0.031)
    assert contact_refinement.max_rotation_radians == pytest.approx(0.41)
    assert contact_refinement.max_scale_delta == pytest.approx(0.09)
    duration = torch.nn.functional.softplus(
        contact_refinement.child_log_duration
    ) + 0.02
    assert duration.mean().item() == pytest.approx(0.12)


def test_residual_expert_transplants_global_anchor_without_overwriting_refinement():
    global_model = GaussianModel(3, _deformation_args("global"))
    _assign_gaussian_state(global_model)
    with torch.no_grad():
        global_model._deformation.deformation_net.pos_deform[-1].bias.fill_(0.7)
        global_model._features_dc.fill_(0.25)
    global_state = global_model.capture_expert_state()

    local_model = GaussianModel(3, _deformation_args("local"))
    _assign_gaussian_state(local_model, point_count=2)
    refinement_before = {
        name: value.detach().clone()
        for name, value in local_model._deformation.state_dict().items()
        if "complete_expert_head.refinement" in name
    }

    local_model.restore_global_anchor_state(global_state)

    assert torch.equal(local_model._xyz, global_model._xyz)
    assert torch.equal(local_model._features_dc, global_model._features_dc)
    assert torch.equal(
        local_model._deformation.deformation_net.pos_deform[-1].bias,
        global_model._deformation.deformation_net.pos_deform[-1].bias,
    )
    local_state = local_model._deformation.state_dict()
    for name, value in refinement_before.items():
        assert torch.equal(local_state[name], value)


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
    network.initialize_tracking_state(means, rotations)
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


def test_local_motion_scaffold_identity_and_rigid_transform_contract():
    means, scales, rotations, opacity = _canonical_inputs(count=12)
    scaffold = MotionScaffoldLocalExpert(
        node_count=6,
        knn=4,
        hidden_dim=16,
        max_translation_ratio=0.1,
        max_rotation_radians=0.4,
    )
    scaffold.set_aabb(means.amax(dim=0), means.amin(dim=0))
    scaffold.initialize_from_canonical(means, rotations)
    times = torch.full((means.shape[0], 1), 0.5)

    identity = scaffold(
        canonical_means3d=means,
        canonical_rotations3d=rotations,
        means3d=means,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=times,
        scene_scale=torch.tensor(1.0),
    )

    assert torch.allclose(identity["means3d"], means, atol=1e-6)
    assert torch.allclose(identity["rotations"], rotations, atol=1e-6)
    assert identity["scaffold_arap"].item() == pytest.approx(0.0, abs=1e-8)

    with torch.no_grad():
        last_linear = [
            layer
            for layer in scaffold.trajectory
            if isinstance(layer, nn.Linear)
        ][-1]
        last_linear.bias.copy_(
            torch.tensor((0.2, -0.1, 0.15, 0.0, 0.0, 0.0))
        )
    transformed = scaffold(
        canonical_means3d=means,
        canonical_rotations3d=rotations,
        means3d=means,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=times,
        scene_scale=torch.tensor(1.0),
    )

    assert not torch.allclose(transformed["means3d"], means)
    assert torch.allclose(
        torch.cdist(transformed["means3d"], transformed["means3d"]),
        torch.cdist(means, means),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        transformed["rotations"].norm(dim=-1),
        torch.ones(means.shape[0]),
        atol=1e-6,
    )


def test_local_motion_scaffold_bounds_node_geometry():
    means, scales, rotations, opacity = _canonical_inputs(count=12)
    scaffold = MotionScaffoldLocalExpert(
        node_count=6,
        knn=4,
        hidden_dim=16,
        max_node_offset_ratio=0.02,
        max_radius_scale=2.0,
    )
    scaffold.set_aabb(means.amax(dim=0), means.amin(dim=0))
    scaffold.initialize_from_canonical(means, rotations)
    with torch.no_grad():
        scaffold.node_offsets.fill_(100.0)
        scaffold.node_log_radius_scale.fill_(100.0)

    output = scaffold(
        canonical_means3d=means,
        canonical_rotations3d=rotations,
        means3d=means,
        scales=scales,
        rotations=rotations,
        opacity_logits=opacity,
        time_values=torch.full((means.shape[0], 1), 0.5),
        scene_scale=torch.tensor(2.0),
    )
    active = int(scaffold.active_nodes.item())
    bounded_offsets = (
        torch.tanh(scaffold.node_offsets[:active]) * 0.04
    )
    bounded_radii = scaffold.node_base_radii[:active] * 2.0

    assert bounded_offsets.abs().max().item() == pytest.approx(0.04)
    assert output["scaffold_mean_radius"].item() == pytest.approx(
        bounded_radii.mean().item()
    )
    assert torch.isfinite(output["means3d"]).all()
    assert torch.isfinite(output["rotations"]).all()


def test_local_scaffold_regularization_is_part_of_tracking_objective():
    args = SimpleNamespace(
        lambda_scaffold_arap=0.1,
        lambda_scaffold_acceleration=0.2,
        lambda_scaffold_node_offset=0.3,
    )
    aux = {
        "d_mu": torch.zeros(2, 3),
        "pi_geo": torch.ones(2, 1),
        "scaffold_arap": torch.tensor(2.0),
        "scaffold_acceleration": torch.tensor(3.0),
        "scaffold_node_offset": torch.tensor(4.0),
        "scaffold_node_translation_norm": torch.tensor((0.1, 0.2)),
        "scaffold_mean_radius": torch.tensor(0.5),
    }

    losses = compute_tracking_losses(
        aux=aux,
        iteration=1,
        args=args,
        prev_d_mu=None,
        active_geo=1,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("local",),
        vis_expert_names=("stable",),
        force_geo_expert="local",
    )

    assert losses["L_scaffold_arap"].item() == pytest.approx(0.2)
    assert losses["L_scaffold_acceleration"].item() == pytest.approx(0.6)
    assert losses["L_scaffold_node_offset"].item() == pytest.approx(1.2)
    assert losses["scaffold_node_translation_norm"].item() == pytest.approx(0.15)


def test_contact_spacetime_bank_starts_invisible_but_receives_gradient():
    means, scales, rotations, opacity = _canonical_inputs(count=10)
    expert = CompleteEndoMoeExpert(
        role="contact",
        time_feature_dim=8,
        hidden_dim=16,
        contact_anchor_count=4,
        contact_chart_count=3,
    )
    expert.set_aabb(means.amax(dim=0), means.amin(dim=0))
    expert.initialize_from_canonical(means, rotations)
    times = torch.full((means.shape[0], 1), 0.5)

    means_out, _, _, opacity_out, aux = expert(
        canonical_means3d=means,
        canonical_scales=scales,
        canonical_rotations=rotations,
        canonical_opacity=opacity,
        base_means3d=means,
        base_scales=scales,
        base_rotations=rotations,
        base_opacity=opacity,
        time_values=times,
        scene_scale=torch.tensor(1.0),
    )

    assert torch.allclose(means_out, means)
    assert torch.allclose(opacity_out, opacity)
    assert aux["auxiliary_means3d"].shape == (12, 3)
    assert torch.count_nonzero(aux["auxiliary_opacity"]).item() == 0
    assert torch.equal(
        aux["auxiliary_parent_indices"],
        expert.refinement.anchor_parent_indices[
            expert.refinement.child_anchor_indices[:12]
        ],
    )
    assert aux["auxiliary_temporal_rbf"][1].item() == pytest.approx(
        1.0,
        abs=1e-6,
    )
    aux["auxiliary_opacity"].sum().backward()
    amplitude_gradient = expert.refinement.child_amplitude_raw.grad
    assert amplitude_gradient is not None
    assert torch.count_nonzero(amplitude_gradient).item() > 0


def test_contact_spacetime_bank_bounds_raw_motion_parameters():
    means, scales, rotations, opacity = _canonical_inputs(count=10)
    expert = CompleteEndoMoeExpert(
        role="contact",
        time_feature_dim=8,
        hidden_dim=16,
        contact_anchor_count=4,
        contact_chart_count=3,
    )
    expert.set_aabb(means.amax(dim=0), means.amin(dim=0))
    expert.initialize_from_canonical(means, rotations)
    with torch.no_grad():
        expert.refinement.child_spatial_offset.fill_(100.0)
        expert.refinement.child_velocity.fill_(100.0)
        expert.refinement.child_acceleration.fill_(100.0)
        expert.refinement.child_rotation_velocity.fill_(100.0)
        expert.refinement.child_scale_delta.fill_(100.0)

    _, _, _, _, aux = expert(
        canonical_means3d=means,
        canonical_scales=scales,
        canonical_rotations=rotations,
        canonical_opacity=opacity,
        base_means3d=means,
        base_scales=scales,
        base_rotations=rotations,
        base_opacity=opacity,
        time_values=torch.ones(means.shape[0], 1),
        scene_scale=torch.tensor(2.0),
    )
    child_parent = aux["auxiliary_parent_indices"]
    canonical_offset = (
        aux["auxiliary_canonical_means3d"] - means[child_parent]
    )
    scale_delta = aux["auxiliary_scales"] - scales[child_parent]

    assert canonical_offset.abs().max().item() == pytest.approx(0.04, abs=1e-6)
    assert scale_delta.abs().max().item() == pytest.approx(0.1, abs=1e-6)
    assert torch.isfinite(aux["auxiliary_means3d"]).all()
    assert torch.isfinite(aux["auxiliary_rotations"]).all()
    assert torch.allclose(
        aux["auxiliary_rotations"].norm(dim=-1),
        torch.ones(aux["auxiliary_rotations"].shape[0]),
        atol=1e-6,
    )


def test_contact_bank_regularization_is_part_of_tracking_objective():
    args = SimpleNamespace(
        lambda_contact_bank_sparsity=0.1,
        lambda_contact_bank_locality=0.2,
        lambda_contact_bank_acceleration=0.3,
        lambda_contact_bank_spatial_offset=0.4,
        lambda_contact_bank_duration=0.5,
    )
    aux = {
        "d_mu": torch.zeros(2, 3),
        "pi_geo": torch.ones(2, 1),
        "contact_bank_sparsity": torch.tensor(1.0),
        "contact_bank_locality": torch.tensor(2.0),
        "contact_bank_acceleration": torch.tensor(3.0),
        "contact_bank_spatial_offset": torch.tensor(4.0),
        "contact_bank_duration": torch.tensor(5.0),
        "auxiliary_temporal_rbf": torch.tensor((0.2, 0.4)),
        "auxiliary_contact_target": torch.tensor((0.1, 0.5)),
    }

    losses = compute_tracking_losses(
        aux=aux,
        iteration=1,
        args=args,
        prev_d_mu=None,
        active_geo=1,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("contact",),
        vis_expert_names=("stable",),
        force_geo_expert="contact",
    )

    assert losses["L_contact_bank_sparsity"].item() == pytest.approx(0.1)
    assert losses["L_contact_bank_locality"].item() == pytest.approx(0.4)
    assert losses["L_contact_bank_acceleration"].item() == pytest.approx(0.9)
    assert losses["L_contact_bank_spatial_offset"].item() == pytest.approx(1.6)
    assert losses["L_contact_bank_duration"].item() == pytest.approx(2.5)
    assert losses["contact_bank_temporal_activity"].item() == pytest.approx(0.3)
    assert losses["contact_bank_boundary_support"].item() == pytest.approx(0.3)


def test_complete_experts_use_role_specific_regularization_only():
    shared_args = SimpleNamespace(
        lambda_geo_temp=1.0,
        lambda_geo_spatial=1.0,
        lambda_motion_mag_global=1.0,
        lambda_motion_mag_local=1.0,
        lambda_balance_vis=1.0,
        lambda_route_conf_vis=1.0,
        lambda_entropy_geo=1.0,
        lambda_entropy_vis=1.0,
        entropy_end_iter=100,
        lambda_vis_sparse=1.0,
        lambda_decouple=1.0,
        enable_decouple_iter=0,
        lambda_scaffold_arap=2.0,
        lambda_contact_bank_sparsity=3.0,
    )
    common = {
        "d_mu": torch.ones(2, 3),
        "d_mu_sequence": torch.stack(
            (torch.zeros(2, 3), torch.ones(2, 3)),
            dim=0,
        ),
        "time_sequence": torch.tensor((0.0, 1.0)),
        "means3d_canonical": torch.randn(2, 3),
        "pi_geo": torch.ones(2, 1),
        "pi_vis": torch.full((2, 2), 0.5),
        "entropy_geo": torch.tensor(0.5),
        "entropy_vis": torch.tensor(0.5),
        "route_max_prob_vis": torch.full((2,), 0.5),
        "d_opacity_logit": torch.ones(2, 1),
        "global_motion_norm": torch.ones(2),
        "local_motion_norm": torch.ones(2),
        "transient_probability": torch.full((2, 1), 0.5),
    }

    global_losses = compute_tracking_losses(
        aux=dict(common, expert_role="global"),
        iteration=1,
        args=shared_args,
        prev_d_mu=None,
        active_geo=1,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("global",),
        vis_expert_names=("stable",),
        force_geo_expert="global",
    )
    assert all(
        value.item() == pytest.approx(0.0)
        for name, value in global_losses.items()
        if name.startswith("L_")
    )

    local_losses = compute_tracking_losses(
        aux=dict(common, expert_role="local", scaffold_arap=torch.tensor(2.0)),
        iteration=1,
        args=shared_args,
        prev_d_mu=None,
        active_geo=1,
        active_vis=1,
        enable_visibility=False,
        geo_expert_names=("local",),
        vis_expert_names=("stable",),
        force_geo_expert="local",
    )
    assert local_losses["L_scaffold_arap"].item() == pytest.approx(4.0)
    assert local_losses["L_geo_temp"].item() == pytest.approx(0.0)
    assert local_losses["L_motion_mag"].item() == pytest.approx(0.0)

    contact_losses = compute_tracking_losses(
        aux=dict(
            common,
            expert_role="contact",
            contact_bank_sparsity=torch.tensor(2.0),
        ),
        iteration=1,
        args=shared_args,
        prev_d_mu=None,
        active_geo=1,
        active_vis=2,
        enable_visibility=True,
        geo_expert_names=("contact",),
        vis_expert_names=("stable", "transient"),
        force_geo_expert="contact",
    )
    assert contact_losses["L_contact_bank_sparsity"].item() == pytest.approx(6.0)
    assert contact_losses["L_balance_vis"].item() == pytest.approx(0.0)
    assert contact_losses["L_entropy"].item() == pytest.approx(0.0)
    assert contact_losses["L_decouple"].item() == pytest.approx(0.0)
    assert contact_losses["L_vis_sparse"].item() > 0.0


def test_complete_expert_scheduler_enables_visibility_only_for_contact():
    local_phase = CompleteExpertScheduler("local").build(10, 100)
    contact_phase = CompleteExpertScheduler("contact").build(10, 100)

    assert local_phase.force_geo_expert == "local"
    assert not local_phase.enable_visibility
    assert not local_phase.is_group_trainable("tracking_time_encoder")
    assert not local_phase.is_group_trainable("tracking_base_deformation")
    assert not local_phase.is_group_trainable("tracking_base_grid")
    assert not local_phase.is_group_trainable("xyz")
    assert not local_phase.is_group_trainable("f_dc")
    assert local_phase.is_group_trainable("tracking_expert_refinement")
    assert contact_phase.force_geo_expert == "contact"
    assert contact_phase.enable_visibility
    assert contact_phase.active_vis == 2
    assert not contact_phase.is_group_trainable("tracking_time_encoder")
    assert not contact_phase.is_group_trainable("opacity")
    assert contact_phase.is_group_trainable("tracking_expert_refinement")
