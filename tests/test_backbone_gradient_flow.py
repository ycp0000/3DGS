"""Backbone gradient-flow regression for the EndoMoe Global anchor.

Guards the contract restored in commit "fix(model): stop zero-init of
endomoeg_expert backbone heads": the Global anchor must train its
HexPlane / feature_out / deformation heads end-to-end. Zero-initialising
the last Linear layer of the deformation heads silently severs the
gradient path through the upstream backbone:

    dL/d(input) = dL/d(output) @ W = 0   if W is zero

If anything in the future reintroduces that zero-init for the
``endomoeg_expert`` Global role, the backbone becomes effectively
untrainable and Stage 2 caps at the canonical reconstruction PSNR. The
failure is invisible at the loss-curve level (``Ll1`` still drops a
little because the canonical Gaussian parameters themselves can move),
but the dynamic component of the model never materialises and Stage 4
later aborts with ``oracle headroom == 0``.

These tests do not rely on the rasterizer or CUDA; they call the
deformation network directly and inspect ``.grad`` after a synthetic
backward.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_deformation_module = _load_module("deformation_module_grad", "scene/deformation.py")
Deformation = _deformation_module.Deformation
deform_network = _deformation_module.deform_network


_PLANE_CONFIG = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 8,
    "resolution": [16, 16, 16, 8],
}


def _global_args() -> SimpleNamespace:
    return SimpleNamespace(
        tracking_type="endomoeg_expert",
        endomoeg_expert_role="global",
        endomoeg_expert_hidden_dim=16,
        no_grid=False,
        bounds=1.6,
        kplanes_config=_PLANE_CONFIG,
        multires=[1],
        timenet_output=8,
        camera_extent=1.0,
        no_ds=False,
        no_dr=False,
        no_do=False,
        max_disp_smooth_ratio=0.01,
        max_disp_local_ratio=0.03,
        max_rot_smooth=0.05,
        max_rot_local=0.05,
        max_scale_smooth=0.05,
        max_scale_local=0.05,
        max_opacity_delta=4.0,
        current_iteration=0,
        iterations=9000,
        net_width=8,
        defor_depth=1,
        timebase_pe=1,
        timenet_width=8,
        scale_rotation_pe=0,
        endomoeg_residual_lr_scale=0.01,
        endomoeg_residual_warmup_iterations=500,
    )


def _residual_args(role: str) -> SimpleNamespace:
    args = _global_args()
    args.endomoeg_expert_role = role
    return args


def _make_inputs(args, point_count: int = 8):
    torch.manual_seed(0)
    point = torch.randn(point_count, 3, requires_grad=False)
    times = torch.rand(point_count, 1, requires_grad=False)
    scales = torch.zeros(point_count, 3, requires_grad=False)
    rotations = torch.zeros(point_count, 4, requires_grad=False)
    rotations[:, 0] = 1.0
    opacity = torch.zeros(point_count, 1, requires_grad=False)
    return point, scales, rotations, opacity, times


def _initialize_residual_state(network: nn.Module, point: torch.Tensor, rotations: torch.Tensor) -> None:
    """Mirror the production order: residual expert refinement modules
    are initialised from canonical Gaussians before the first forward.
    """
    network.initialize_tracking_state(point, rotations)


def _backbone_grad_norms(network: nn.Module) -> dict:
    inner = network.deformation_net
    norms = {}

    def _norm(module: nn.Module) -> float:
        total = 0.0
        any_grad = False
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            any_grad = True
            total += float(parameter.grad.detach().square().sum().item())
        return total**0.5 if any_grad else 0.0

    norms["grid"] = _norm(inner.grid)
    norms["feature_out"] = _norm(inner.feature_out)
    norms["pos_deform"] = _norm(inner.pos_deform)
    norms["scales_deform"] = _norm(inner.scales_deform)
    norms["rotations_deform"] = _norm(inner.rotations_deform)
    norms["opacity_deform"] = _norm(inner.opacity_deform)
    return norms


def _run_forward_backward(network: nn.Module, args, *, initialize_residual: bool = False) -> None:
    point, scales, rotations, opacity, times = _make_inputs(args)
    if initialize_residual:
        # ``set_aabb`` and the refinement initialiser run during the
        # production training_setup; reproduce that order here so the
        # forward path is well-defined.
        xyz_max = point.detach().amax(dim=0)
        xyz_min = point.detach().amin(dim=0)
        network.set_aabb(xyz_max, xyz_min)
        _initialize_residual_state(network, point, rotations)
    pts, scales_out, rotations_out, opacity_out = network(
        point=point,
        scales=scales,
        rotations=rotations,
        opacity=opacity,
        times_sel=times,
    )
    loss = (
        pts.square().mean()
        + scales_out.square().mean()
        + rotations_out.square().mean()
        + opacity_out.square().mean()
    )
    loss.backward()


def test_global_role_propagates_gradients_through_full_backbone():
    """The Global anchor must learn end-to-end. HexPlane, feature_out,
    and every deformation head must receive non-zero gradients on the
    very first forward / backward pass, otherwise Stage 2 cannot reach
    the original EndoGaussian PSNR no matter how long it trains.
    """
    args = _global_args()
    network = deform_network(args)
    _run_forward_backward(network, args)
    norms = _backbone_grad_norms(network)
    severed = [name for name, value in norms.items() if value == 0.0]
    assert not severed, (
        "Global anchor backbone has zero gradients on: {}. "
        "Last-Linear zero-init severs gradient flow through HexPlane "
        "and feature_out. See scene/deformation.py::"
        "reset_backbone_to_identity.".format(severed)
    )


@pytest.mark.parametrize("role", ["local", "contact"])
def test_residual_roles_keep_backbone_gradients_alive(role):
    """Local/Contact residual experts must also keep the deformation
    backbone gradient-alive at construction time. The Global anchor is
    transplanted later by ``restore_global_anchor_state``, but at
    construction time the model still has to be a well-formed
    differentiable graph; otherwise any code path that runs forward
    before the transplant (sanity checks, optimizer setup) silently
    breaks.
    """
    args = _residual_args(role)
    network = deform_network(args)
    _run_forward_backward(network, args, initialize_residual=True)
    norms = _backbone_grad_norms(network)
    severed = [name for name, value in norms.items() if value == 0.0]
    assert not severed, (
        "Residual expert role={} has zero gradients on: {}".format(
            role, severed
        )
    )


def test_legacy_cams_gs_moe_keeps_identity_init():
    """``cams_gs_moe`` is the historical multi-expert routing path. Its
    last-Linear zero-init is intentional because every expert must
    start from the canonical state for the residual routing maths to
    hold. Guard against accidental removal of that behaviour.
    """
    args = _global_args()
    args.tracking_type = "cams_gs_moe"
    args.endomoeg_expert_role = ""
    # cams_gs_moe wires extra config; we only need to verify the
    # zero-init path is still triggered by the helper itself.
    inner = Deformation(D=1, W=8, args=args)
    inner.reset_backbone_to_identity()
    for head in (
        inner.pos_deform,
        inner.scales_deform,
        inner.rotations_deform,
        inner.opacity_deform,
    ):
        if head is None:
            continue
        last_linear = [
            module for module in head.modules() if isinstance(module, nn.Linear)
        ][-1]
        assert torch.equal(last_linear.weight, torch.zeros_like(last_linear.weight))
        assert torch.equal(last_linear.bias, torch.zeros_like(last_linear.bias))
