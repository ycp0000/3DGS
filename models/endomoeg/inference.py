import os

import torch

from .ensemble import FrozenExpertEnsemble
from .router import EndoMoeVolumeAwareRouter
from .router_bundle import load_router_bundle


class FrozenRouterAssembly:
    def __init__(self, ensemble, router, payload, bundle_path):
        self.ensemble = ensemble
        self.router = router
        self.payload = payload
        self.bundle_path = bundle_path
        self.iteration = int(payload["iteration"])
        self.assert_frozen()

    def assert_frozen(self):
        self.ensemble.assert_frozen()
        trainable = [
            name
            for name, parameter in self.router.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(
                "Inference Router still has trainable parameters: {}".format(
                    ", ".join(trainable)
                )
            )


def resolve_router_bundle_path(bundle_dir, router_bundle_path=None):
    bundle_dir_value = os.fspath(bundle_dir)
    if not os.path.isabs(bundle_dir_value):
        raise ValueError("EndoMoe bundle directory must be absolute")
    if router_bundle_path:
        router_path_value = os.fspath(router_bundle_path)
        if not os.path.isabs(router_path_value):
            raise ValueError("EndoMoe Router bundle path must be absolute")
        resolved = os.path.abspath(router_path_value)
    else:
        resolved = os.path.join(
            os.path.abspath(bundle_dir_value),
            "router.pth",
        )
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            "EndoMoe Router bundle does not exist: {}".format(resolved)
        )
    return resolved


def load_frozen_router_assembly(
    bundle_dir,
    expected_source_path,
    device,
    minimum_expert_psnr=0.0,
    router_bundle_path=None,
):
    source_path_value = os.fspath(expected_source_path)
    if not os.path.isabs(source_path_value):
        raise ValueError("EndoMoe source_path must be absolute")
    resolved_router_path = resolve_router_bundle_path(
        bundle_dir,
        router_bundle_path=router_bundle_path,
    )
    ensemble = FrozenExpertEnsemble.load(
        bundle_dir,
        minimum_psnr=float(minimum_expert_psnr),
        device=device,
        expected_source_path=os.path.abspath(source_path_value),
    )
    payload = load_router_bundle(
        resolved_router_path,
        map_location="cpu",
        ensemble=ensemble,
    )
    hidden_params = (payload.get("config") or {}).get("hidden_params") or {}
    if not hidden_params:
        raise ValueError(
            "Router bundle is missing the hidden-parameter architecture config"
        )
    router = EndoMoeVolumeAwareRouter(
        ensemble.point_counts(),
        gaussian_hidden_dim=int(hidden_params["moe_router_hidden_dim"]),
    ).to(torch.device(device))
    router.load_state_dict(payload["router_state"], strict=True)
    router.eval()
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    return FrozenRouterAssembly(
        ensemble=ensemble,
        router=router,
        payload=payload,
        bundle_path=resolved_router_path,
    )
