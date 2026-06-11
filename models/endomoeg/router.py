from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


RESIDUAL_ROLES = ("local", "contact")


def _contact_parent_activity(contact_state, parent_count):
    activity = contact_state["opacity"].new_zeros((parent_count, 1))
    child_count = int(contact_state.get("auxiliary_point_count", 0))
    if child_count <= 0:
        return activity
    child_slice = slice(parent_count, parent_count + child_count)
    child_parent = contact_state["auxiliary_parent_indices"].long()
    if child_parent.shape[0] != child_count:
        raise ValueError(
            "Contact auxiliary parent count does not match child opacity count"
        )
    activity.index_add_(
        0,
        child_parent,
        contact_state["opacity"][child_slice].clamp_min(0.0),
    )
    return 1.0 - torch.exp(-activity)


def _canonicalize_spatial_mask(mask, spatial_shape, device, dtype):
    mask_value = mask.to(device=device, dtype=dtype)
    expected_shape = tuple(spatial_shape)
    if tuple(mask_value.shape) == (1,) + expected_shape:
        mask_value = mask_value[0]
    elif tuple(mask_value.shape) == expected_shape + (1,):
        mask_value = mask_value[..., 0]
    if mask_value.ndim != 2 or tuple(mask_value.shape) != expected_shape:
        raise ValueError(
            "mask must have shape [H, W], [1, H, W], or [H, W, 1]; "
            "expected spatial shape {}, got {}".format(
                expected_shape,
                tuple(mask.shape),
            )
        )
    return mask_value


def _masked_mean(value, mask):
    if mask is None:
        return value.mean()
    mask_value = _canonicalize_spatial_mask(
        mask,
        value.shape[-2:],
        value.device,
        value.dtype,
    )
    while mask_value.ndim < value.ndim:
        mask_value = mask_value.unsqueeze(0)
    expanded = mask_value.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def _nlerp_quaternion(source, target, gate):
    aligned_target = torch.where(
        (source * target).sum(dim=-1, keepdim=True) < 0,
        -target,
        target,
    )
    return F.normalize(
        source + gate * (aligned_target - source),
        dim=-1,
    )


def _quaternion_conjugate(quaternion):
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quaternion_multiply(left, right):
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _apply_rotation_residual(current, source, target, gate):
    aligned_target = torch.where(
        (source * target).sum(dim=-1, keepdim=True) < 0,
        -target,
        target,
    )
    delta = F.normalize(
        _quaternion_multiply(
            aligned_target,
            _quaternion_conjugate(source),
        ),
        dim=-1,
    )
    identity = torch.zeros_like(delta)
    identity[..., 0] = 1.0
    gated_delta = _nlerp_quaternion(identity, delta, gate)
    return F.normalize(
        _quaternion_multiply(gated_delta, current),
        dim=-1,
    )


def incremental_gain_targets(
    global_rgb,
    candidate_rgb,
    ground_truth,
    temperature=0.02,
):
    global_error = (global_rgb.detach() - ground_truth.detach()).abs().mean(
        dim=0
    )
    candidate_error = (
        candidate_rgb.detach() - ground_truth.detach()
    ).abs().mean(dim=0)
    return torch.sigmoid(
        (global_error - candidate_error) / max(float(temperature), 1e-6)
    )


def compose_residual_gaussian_state(
    global_state,
    local_state,
    contact_state,
    gates,
):
    if gates.ndim != 2 or gates.shape[1] != 2:
        raise ValueError("gates must have shape [N, 2]")
    parent_count = int(global_state["base_point_count"])
    if gates.shape[0] != parent_count:
        raise ValueError("gate count does not match parent Gaussian count")
    for role, state in (
        ("local", local_state),
        ("contact", contact_state),
    ):
        if int(state["base_point_count"]) != parent_count:
            raise ValueError(
                "{} parent point count does not match Global".format(role)
            )
        if not torch.allclose(
            global_state["canonical_xyz"][:parent_count],
            state["canonical_xyz"][:parent_count],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "{} canonical parent cloud does not match Global".format(role)
            )

    local_gate = gates[:, 0:1]
    contact_gate = gates[:, 1:2]
    global_means = global_state["means3d"][:parent_count]
    local_means = local_state["means3d"][:parent_count]
    contact_means = contact_state["means3d"][:parent_count]
    parent_means = (
        global_means
        + local_gate * (local_means - global_means)
        + contact_gate * (contact_means - global_means)
    )
    global_scales = global_state["scales"][:parent_count]
    parent_scales = (
        global_scales
        + local_gate
        * (local_state["scales"][:parent_count] - global_scales)
        + contact_gate
        * (contact_state["scales"][:parent_count] - global_scales)
    )
    global_rotations = global_state["rotations"][:parent_count]
    parent_rotations = _apply_rotation_residual(
        global_rotations,
        global_rotations,
        local_state["rotations"][:parent_count],
        local_gate,
    )
    parent_rotations = _apply_rotation_residual(
        parent_rotations,
        global_rotations,
        contact_state["rotations"][:parent_count],
        contact_gate,
    )
    global_opacity = global_state["opacity"][:parent_count]
    parent_opacity = (
        global_opacity
        + contact_gate
        * (contact_state["opacity"][:parent_count] - global_opacity)
    ).clamp(0.0, 1.0)
    global_colors = global_state["colors"][:parent_count]
    parent_colors = (
        global_colors
        + contact_gate
        * (contact_state["colors"][:parent_count] - global_colors)
    ).clamp_min(0.0)

    child_count = int(contact_state.get("auxiliary_point_count", 0))
    if child_count <= 0:
        return {
            "means3d": parent_means,
            "scales": parent_scales,
            "rotations": parent_rotations,
            "opacity": parent_opacity,
            "colors": parent_colors,
            "base_point_count": parent_count,
            "auxiliary_point_count": 0,
        }
    child_slice = slice(parent_count, parent_count + child_count)
    child_parent = contact_state["auxiliary_parent_indices"].long()
    child_local_offset = (
        local_means[child_parent] - global_means[child_parent]
    )
    child_means = (
        contact_state["means3d"][child_slice]
        + local_gate[child_parent] * child_local_offset
    )
    child_opacity = (
        contact_state["opacity"][child_slice]
        * contact_gate[child_parent]
    ).clamp(0.0, 1.0)
    contact_parent_rotations = contact_state["rotations"][:parent_count]
    child_relative_rotations = F.normalize(
        _quaternion_multiply(
            contact_state["rotations"][child_slice],
            _quaternion_conjugate(contact_parent_rotations[child_parent]),
        ),
        dim=-1,
    )
    child_rotations = F.normalize(
        _quaternion_multiply(
            child_relative_rotations,
            parent_rotations[child_parent],
        ),
        dim=-1,
    )
    return {
        "means3d": torch.cat((parent_means, child_means), dim=0),
        "scales": torch.cat(
            (parent_scales, contact_state["scales"][child_slice]),
            dim=0,
        ),
        "rotations": torch.cat(
            (
                parent_rotations,
                child_rotations,
            ),
            dim=0,
        ),
        "opacity": torch.cat((parent_opacity, child_opacity), dim=0),
        "colors": torch.cat(
            (parent_colors, contact_state["colors"][child_slice]),
            dim=0,
        ),
        "base_point_count": parent_count,
        "auxiliary_point_count": child_count,
    }


def compute_router_losses(
    composite_rgb,
    global_rgb,
    candidate_rgb,
    gates,
    gate_maps,
    ground_truth,
    mask=None,
    gain_temperature=0.02,
    lambda_gain=0.1,
    lambda_sparsity=1e-3,
    lambda_no_regret=0.5,
):
    reconstruction = _masked_mean(
        (composite_rgb - ground_truth).abs(),
        mask,
    )
    global_error = (global_rgb.detach() - ground_truth.detach()).abs().mean(
        dim=0
    )
    composite_error = (composite_rgb - ground_truth).abs().mean(dim=0)
    no_regret = _masked_mean(
        F.relu(composite_error - global_error),
        mask,
    )
    gain_losses = []
    losses = {
        "L_router_reconstruction": reconstruction,
        "L_router_no_regret": no_regret * float(lambda_no_regret),
        "L_router_gate_sparsity": gates.square().mean()
        * float(lambda_sparsity),
    }
    for index, role in enumerate(RESIDUAL_ROLES):
        target = incremental_gain_targets(
            global_rgb,
            candidate_rgb[role],
            ground_truth,
            temperature=gain_temperature,
        )
        gate_map = gate_maps[index]
        safe_gate_map = gate_map.clamp(1e-6, 1.0 - 1e-6)
        safe_gate_map = gate_map + (safe_gate_map - gate_map).detach()
        gain_loss = _masked_mean(
            F.binary_cross_entropy(
                safe_gate_map,
                target,
                reduction="none",
            ),
            mask,
        )
        gain_losses.append(gain_loss)
        losses["router_usage_{}".format(role)] = gates[:, index].mean().detach()
        losses["router_target_{}".format(role)] = _masked_mean(
            target,
            mask,
        ).detach()
        losses["L_router_gain_{}".format(role)] = (
            gain_loss * float(lambda_gain)
        )
    losses["L_router_total"] = (
        losses["L_router_reconstruction"]
        + losses["L_router_no_regret"]
        + losses["L_router_gate_sparsity"]
        + sum(
            losses["L_router_gain_{}".format(role)]
            for role in RESIDUAL_ROLES
        )
    )
    return losses


class EndoMoeVolumeAwareRouter(nn.Module):
    def __init__(
        self,
        point_counts,
        gaussian_hidden_dim=64,
    ):
        super().__init__()
        ordered_counts = OrderedDict(
            (role, int(point_counts[role]))
            for role in ("global", "local", "contact")
        )
        if any(count <= 0 for count in ordered_counts.values()):
            raise ValueError("Every EndoMoe expert must contain Gaussians")
        if len(set(ordered_counts.values())) != 1:
            raise ValueError(
                "Residual Router requires identical parent point counts"
            )
        self.point_counts = ordered_counts
        self.parent_count = ordered_counts["global"]
        self.base_gates = nn.Parameter(
            torch.zeros(self.parent_count, len(RESIDUAL_ROLES))
        )
        self.gaussian_feature_mlp = nn.Sequential(
            nn.Linear(18, gaussian_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(gaussian_hidden_dim, gaussian_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(gaussian_hidden_dim, len(RESIDUAL_ROLES)),
        )
        self.reset_parameters()

    def reset_parameters(self):
        linear_layers = [
            module
            for module in self.gaussian_feature_mlp
            if isinstance(module, nn.Linear)
        ]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)
        nn.init.zeros_(self.base_gates)

    @staticmethod
    def _straight_through_gates(raw_gates):
        forward_value = raw_gates.clamp(0.0, 1.0)
        surrogate = torch.sigmoid(raw_gates)
        return forward_value.detach() + surrogate - surrogate.detach()

    def residual_gates(
        self,
        global_state,
        local_state,
        contact_state,
        camera,
        time_value,
    ):
        parent_count = self.parent_count
        canonical = global_state["canonical_xyz"][:parent_count]
        xyz_min = canonical.amin(dim=0, keepdim=True)
        xyz_max = canonical.amax(dim=0, keepdim=True)
        normalized_xyz = (
            (canonical - xyz_min)
            / (xyz_max - xyz_min).clamp_min(1e-6)
            * 2.0
            - 1.0
        )
        global_means = global_state["means3d"][:parent_count]
        scene_scale = float(global_state.get("scene_scale", 1.0) or 1.0)
        local_residual = (
            local_state["means3d"][:parent_count] - global_means
        ) / max(abs(scene_scale), 1e-6)
        contact_residual = (
            contact_state["means3d"][:parent_count] - global_means
        ) / max(abs(scene_scale), 1e-6)
        local_norm = local_residual.norm(dim=-1, keepdim=True)
        contact_norm = contact_residual.norm(dim=-1, keepdim=True)
        opacity_delta = torch.cat(
            (
                local_state["opacity"][:parent_count]
                - global_state["opacity"][:parent_count],
                contact_state["opacity"][:parent_count]
                - global_state["opacity"][:parent_count],
            ),
            dim=-1,
        )
        contact_activity = _contact_parent_activity(
            contact_state,
            parent_count,
        )
        camera_center = camera.camera_center.to(
            global_means.device,
            global_means.dtype,
        )
        view_direction = global_means - camera_center.unsqueeze(0)
        view_direction = view_direction / view_direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        time_feature = global_means.new_full(
            (parent_count, 1),
            float(time_value),
        )
        features = torch.cat(
            (
                normalized_xyz,
                local_residual,
                contact_residual,
                local_norm,
                contact_norm,
                opacity_delta,
                contact_activity,
                view_direction,
                time_feature,
            ),
            dim=-1,
        )
        raw_gates = self.base_gates + self.gaussian_feature_mlp(features)
        return self._straight_through_gates(raw_gates), raw_gates
