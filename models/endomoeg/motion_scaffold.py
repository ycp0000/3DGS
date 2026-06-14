from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def _zero_last_linear(module: nn.Sequential) -> None:
    linear_layers = [layer for layer in module if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    nn.init.zeros_(linear_layers[-1].weight)
    nn.init.zeros_(linear_layers[-1].bias)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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


def _quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _rotate_vectors(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(quaternion, dim=-1)
    vector_quaternion = torch.cat(
        (torch.zeros_like(vectors[..., :1]), vectors),
        dim=-1,
    )
    return _quaternion_multiply(
        _quaternion_multiply(normalized, vector_quaternion),
        _quaternion_conjugate(normalized),
    )[..., 1:]


def _axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return F.normalize(
        torch.cat((torch.cos(half_angle), axis_angle * scale), dim=-1),
        dim=-1,
    )


def _dual_quaternion(
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    translation_quaternion = torch.cat(
        (torch.zeros_like(translation[..., :1]), translation),
        dim=-1,
    )
    dual = 0.5 * _quaternion_multiply(translation_quaternion, rotation)
    return rotation, dual


def _blend_dual_quaternions(
    real: torch.Tensor,
    dual: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference = real[:, :1]
    signs = torch.where(
        (real * reference).sum(dim=-1, keepdim=True) < 0,
        -torch.ones_like(real[..., :1]),
        torch.ones_like(real[..., :1]),
    )
    weighted_real = (weights.unsqueeze(-1) * real * signs).sum(dim=1)
    weighted_dual = (weights.unsqueeze(-1) * dual * signs).sum(dim=1)
    norm = weighted_real.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    blended_real = weighted_real / norm
    blended_dual = weighted_dual / norm
    blended_dual = blended_dual - blended_real * (
        blended_real * blended_dual
    ).sum(dim=-1, keepdim=True)
    return blended_real, blended_dual


def _dual_quaternion_transform(
    points: torch.Tensor,
    real: torch.Tensor,
    dual: torch.Tensor,
) -> torch.Tensor:
    translation = 2.0 * _quaternion_multiply(
        dual,
        _quaternion_conjugate(real),
    )[..., 1:]
    return _rotate_vectors(real, points) + translation


def _farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    point_count = int(points.shape[0])
    if point_count == 0:
        raise ValueError("Cannot initialize a motion scaffold from zero points")
    count = max(1, min(int(count), point_count))
    center = points.mean(dim=0, keepdim=True)
    first = (points - center).square().sum(dim=-1).argmax()
    selected = torch.empty(count, device=points.device, dtype=torch.long)
    selected[0] = first
    minimum_distance = (points - points[first]).square().sum(dim=-1)
    for index in range(1, count):
        next_index = minimum_distance.argmax()
        selected[index] = next_index
        distance = (points - points[next_index]).square().sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
    return selected


class MotionScaffoldLocalExpert(nn.Module):
    def __init__(
        self,
        node_count: int = 256,
        knn: int = 4,
        hidden_dim: int = 64,
        time_frequencies: int = 4,
        max_translation_ratio: float = 0.03,
        max_rotation_radians: float = 0.35,
        max_node_offset_ratio: float = 0.02,
        max_radius_scale: float = 4.0,
        initial_gate_probability: float = 0.05,
        normal_weight: float = 0.25,
        temporal_step: float = 1.0 / 155.0,
    ) -> None:
        super().__init__()
        self.node_count = int(node_count)
        self.knn = int(knn)
        self.time_frequencies = int(time_frequencies)
        self.max_translation_ratio = float(max_translation_ratio)
        self.max_rotation_radians = float(max_rotation_radians)
        self.max_node_offset_ratio = float(max_node_offset_ratio)
        self.max_radius_scale = float(max_radius_scale)
        self.initial_gate_probability = float(initial_gate_probability)
        if self.max_node_offset_ratio < 0.0:
            raise ValueError("max_node_offset_ratio must be non-negative")
        if self.max_radius_scale < 1.0:
            raise ValueError("max_radius_scale must be at least 1")
        if not 0.0 < self.initial_gate_probability < 1.0:
            raise ValueError("initial_gate_probability must be in (0, 1)")
        self.normal_weight = float(normal_weight)
        self.temporal_step = float(temporal_step)
        time_dim = 1 + 2 * self.time_frequencies
        self.trajectory = _build_mlp(3 + time_dim, hidden_dim, 6)
        self.node_offsets = nn.Parameter(torch.zeros(self.node_count, 3))
        self.node_log_radius_scale = nn.Parameter(torch.zeros(self.node_count))
        self.node_gate_logits = nn.Parameter(torch.empty(self.node_count))
        self.register_buffer(
            "node_positions",
            torch.zeros(self.node_count, 3),
        )
        self.register_buffer(
            "node_base_radii",
            torch.ones(self.node_count),
        )
        self.register_buffer(
            "node_normals",
            torch.zeros(self.node_count, 3),
        )
        self.register_buffer("active_nodes", torch.zeros((), dtype=torch.long))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("xyz_max", torch.ones(3), persistent=False)
        self.register_buffer("xyz_min", -torch.ones(3), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _zero_last_linear(self.trajectory)
        nn.init.zeros_(self.node_offsets)
        nn.init.zeros_(self.node_log_radius_scale)
        initial_logit = torch.logit(
            torch.tensor(self.initial_gate_probability)
        ).item()
        nn.init.constant_(self.node_gate_logits, initial_logit)

    def named_parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {"tracking_expert_refinement": self.parameters()}

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        self.xyz_max.copy_(
            xyz_max.detach().to(self.xyz_max.device, self.xyz_max.dtype).reshape(3)
        )
        self.xyz_min.copy_(
            xyz_min.detach().to(self.xyz_min.device, self.xyz_min.dtype).reshape(3)
        )

    def initialize_from_canonical(
        self,
        canonical_means: torch.Tensor,
        canonical_rotations: Optional[torch.Tensor] = None,
    ) -> None:
        if bool(self.initialized.item()):
            return
        with torch.no_grad():
            selected = _farthest_point_indices(
                canonical_means.detach(),
                self.node_count,
            )
            active = int(selected.numel())
            selected_positions = canonical_means.detach()[selected]
            self.node_positions[:active].copy_(selected_positions)
            if canonical_rotations is None:
                selected_normals = torch.zeros_like(selected_positions)
                selected_normals[:, 2] = 1.0
            else:
                local_z = selected_positions.new_zeros((active, 3))
                local_z[:, 2] = 1.0
                selected_normals = _rotate_vectors(
                    canonical_rotations.detach()[selected],
                    local_z,
                )
            self.node_normals[:active].copy_(
                F.normalize(selected_normals, dim=-1)
            )
            if active > 1:
                distances = torch.cdist(
                    selected_positions,
                    selected_positions,
                )
                distances.fill_diagonal_(float("inf"))
                base_radii = distances.amin(dim=1).clamp_min(1e-4)
            else:
                base_radius = (
                    canonical_means.detach().amax(dim=0)
                    - canonical_means.detach().amin(dim=0)
                ).norm().clamp_min(1e-4)
                base_radii = base_radius.expand(active)
            self.node_base_radii[:active].copy_(base_radii)
            self.active_nodes.fill_(active)
            self.initialized.fill_(True)

    def _normalized_nodes(self, nodes: torch.Tensor) -> torch.Tensor:
        xyz_max = self.xyz_max.to(nodes.device, nodes.dtype)
        xyz_min = self.xyz_min.to(nodes.device, nodes.dtype)
        extent = (xyz_max - xyz_min).clamp_min(1e-6)
        return ((nodes - xyz_min) / extent) * 2.0 - 1.0

    def _encode_time(self, time_value: torch.Tensor) -> torch.Tensor:
        frequencies = 2.0 ** torch.arange(
            self.time_frequencies,
            device=time_value.device,
            dtype=time_value.dtype,
        )
        angles = time_value * frequencies.view(1, -1) * torch.pi
        return torch.cat(
            (time_value, torch.sin(angles), torch.cos(angles)),
            dim=-1,
        )

    def _node_motion(
        self,
        nodes: torch.Tensor,
        time_value: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_time = self._encode_time(time_value).expand(nodes.shape[0], -1)
        raw = self.trajectory(
            torch.cat((self._normalized_nodes(nodes), encoded_time), dim=-1)
        )
        translation = torch.tanh(raw[:, :3]) * (
            self.max_translation_ratio * scene_scale
        )
        axis_angle = torch.tanh(raw[:, 3:]) * self.max_rotation_radians
        return translation, axis_angle

    def _surface_aware_neighbors(
        self,
        canonical_means: torch.Tensor,
        canonical_rotations: torch.Tensor,
        nodes: torch.Tensor,
        node_normals: torch.Tensor,
        radii: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = torch.cdist(canonical_means, nodes)
        local_z = canonical_means.new_zeros(canonical_means.shape)
        local_z[:, 2] = 1.0
        point_normals = _rotate_vectors(canonical_rotations, local_z)
        normal_cost = 1.0 - torch.matmul(
            F.normalize(point_normals, dim=-1),
            F.normalize(node_normals, dim=-1).transpose(0, 1),
        ).abs()
        metric = distances + self.normal_weight * radii.median() * normal_cost
        neighbor_count = min(self.knn, int(nodes.shape[0]))
        _, indices = torch.topk(
            metric,
            k=neighbor_count,
            dim=1,
            largest=False,
        )
        selected_distances = distances.gather(1, indices)
        selected_radii = radii[indices].clamp_min(1e-5)
        unnormalized = torch.exp(
            -0.5 * (selected_distances / selected_radii).square()
        )
        weights = unnormalized / unnormalized.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        support = unnormalized.amax(dim=1, keepdim=True)
        return indices, weights, support

    @staticmethod
    def _node_base_positions(
        nodes: torch.Tensor,
        canonical_means: torch.Tensor,
        base_means: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        global_displacement = base_means - canonical_means
        node_count = int(nodes.shape[0])
        accumulated = nodes.new_zeros((node_count, 3))
        denominator = nodes.new_zeros((node_count, 1))
        flat_indices = indices.reshape(-1)
        flat_weights = weights.reshape(-1, 1)
        repeated_displacement = global_displacement.unsqueeze(1).expand(
            -1,
            indices.shape[1],
            -1,
        )
        accumulated.index_add_(
            0,
            flat_indices,
            (repeated_displacement.reshape(-1, 3) * flat_weights),
        )
        denominator.index_add_(0, flat_indices, flat_weights)
        return nodes + accumulated / denominator.clamp_min(1e-8)

    def _regularization(
        self,
        nodes: torch.Tensor,
        node_base: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        axis_angle: torch.Tensor,
        time_value: torch.Tensor,
        scene_scale: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if nodes.shape[0] < 2:
            zero = nodes.new_zeros(())
            return {
                "scaffold_arap": zero,
                "scaffold_acceleration": zero,
            }
        distances = torch.cdist(nodes, nodes)
        distances.fill_diagonal_(float("inf"))
        neighbor_count = min(4, int(nodes.shape[0]) - 1)
        neighbor_indices = torch.topk(
            distances,
            k=neighbor_count,
            dim=1,
            largest=False,
        ).indices
        source = node_base.unsqueeze(1)
        target = node_base[neighbor_indices]
        deformed_source = (node_base + translation).unsqueeze(1)
        deformed_target = (node_base + translation)[neighbor_indices]
        predicted_edges = deformed_target - deformed_source
        rigid_edges = _rotate_vectors(
            rotation.unsqueeze(1).expand(-1, neighbor_count, -1),
            target - source,
        )
        arap = (
            (predicted_edges - rigid_edges) / scene_scale
        ).square().sum(dim=-1).mean()

        step = nodes.new_tensor(self.temporal_step)
        previous_translation, previous_axis = self._node_motion(
            nodes,
            (time_value - step).clamp(0.0, 1.0),
            scene_scale,
        )
        next_translation, next_axis = self._node_motion(
            nodes,
            (time_value + step).clamp(0.0, 1.0),
            scene_scale,
        )
        acceleration = (
            (
                next_translation
                - 2.0 * translation
                + previous_translation
            )
            / scene_scale
        ).square().mean()
        acceleration = acceleration + (
            next_axis - 2.0 * axis_angle + previous_axis
        ).square().mean()
        return {
            "scaffold_arap": arap,
            "scaffold_acceleration": acceleration,
        }

    def forward(
        self,
        canonical_means3d: torch.Tensor,
        canonical_rotations3d: torch.Tensor,
        means3d: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        time_values: torch.Tensor,
        scene_scale: torch.Tensor,
        camera: object = None,
    ) -> Dict[str, torch.Tensor]:
        del camera
        if not bool(self.initialized.item()):
            raise RuntimeError(
                "Motion scaffold must be initialized from canonical Gaussians "
                "before optimizer construction"
            )
        active = int(self.active_nodes.item())
        normalized_scale = torch.as_tensor(
            scene_scale,
            device=means3d.device,
            dtype=means3d.dtype,
        ).reshape(()).abs().clamp_min(1e-6)
        node_offset = torch.tanh(self.node_offsets[:active]) * (
            self.max_node_offset_ratio * normalized_scale
        )
        nodes = self.node_positions[:active] + node_offset
        node_normals = self.node_normals[:active]
        radius_log_bound = means3d.new_tensor(self.max_radius_scale).log()
        radius_scale = torch.exp(
            torch.tanh(self.node_log_radius_scale[:active]) * radius_log_bound
        )
        radii = self.node_base_radii[:active] * radius_scale
        time_value = time_values[:, :1].mean(dim=0, keepdim=True)
        translation, axis_angle = self._node_motion(
            nodes,
            time_value,
            normalized_scale,
        )
        node_rotation = _axis_angle_to_quaternion(axis_angle)
        indices, weights, spatial_support = self._surface_aware_neighbors(
            canonical_means3d,
            canonical_rotations3d,
            nodes,
            node_normals,
            radii,
        )
        node_base = self._node_base_positions(
            nodes,
            canonical_means3d,
            means3d,
            indices,
            weights,
        )
        global_translation = (
            node_base
            + translation
            - _rotate_vectors(node_rotation, node_base)
        )
        real, dual = _dual_quaternion(node_rotation, global_translation)
        blended_real, blended_dual = _blend_dual_quaternions(
            real[indices],
            dual[indices],
            weights,
        )
        transformed_means = _dual_quaternion_transform(
            means3d,
            blended_real,
            blended_dual,
        )
        node_gates = torch.sigmoid(self.node_gate_logits[:active])
        point_gate = (
            (weights * node_gates[indices]).sum(dim=1, keepdim=True)
            * spatial_support
        ).clamp(0.0, 1.0)
        transformed_means = means3d + point_gate * (
            transformed_means - means3d
        )
        identity_rotation = torch.zeros_like(blended_real)
        identity_rotation[:, 0] = 1.0
        aligned_rotation = torch.where(
            blended_real[:, :1] < 0.0,
            -blended_real,
            blended_real,
        )
        gated_rotation = F.normalize(
            (1.0 - point_gate) * identity_rotation
            + point_gate * aligned_rotation,
            dim=-1,
        )
        transformed_rotations = _quaternion_multiply(
            gated_rotation,
            rotations,
        )
        regularization = self._regularization(
            nodes,
            node_base,
            translation,
            node_rotation,
            axis_angle,
            time_value,
            normalized_scale,
        )
        d_mu = transformed_means - means3d
        output = {
            "means3d": transformed_means,
            "scales": scales,
            "rotations": transformed_rotations,
            "opacity_logits": opacity_logits,
            "d_mu": d_mu,
            "d_scale": torch.zeros_like(scales),
            "d_rot": axis_angle[indices[:, 0]] * point_gate,
            "d_opacity_logit": torch.zeros_like(opacity_logits),
            "appearance_offsets": means3d.new_zeros((means3d.shape[0], 3)),
            "appearance_rgb_delta": means3d.new_zeros((means3d.shape[0], 3)),
            "visibility_alpha": means3d.new_ones((means3d.shape[0], 1)),
            "transient_probability": means3d.new_zeros((means3d.shape[0], 1)),
            "scaffold_node_translation_norm": (
                translation.norm(dim=-1) / normalized_scale
            ),
            "scaffold_node_offset": (
                node_offset / normalized_scale
            ).square().mean(),
            "scaffold_mean_radius": radii.mean(),
            "scaffold_gate_sparsity": node_gates.mean(),
            "scaffold_point_gate_mean": point_gate.mean(),
            "scaffold_spatial_support_mean": spatial_support.mean(),
        }
        output.update(regularization)
        return output
