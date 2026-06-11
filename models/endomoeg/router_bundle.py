import os
from collections import OrderedDict
from typing import Mapping

import torch

from .expert_bundle import EXPERT_ROLES


ROUTER_BUNDLE_FORMAT = "endomoeg_frozen_expert_router_bundle"
ROUTER_BUNDLE_VERSION = 2
ROUTER_ARCHITECTURE_VERSION = "endomoeg_volume_aware_router_v1"


def _absolute_path(path):
    path_value = os.fspath(path)
    if not os.path.isabs(path_value):
        raise ValueError("Router bundle path must be absolute")
    return os.path.abspath(path_value)


def _cpu_state_dict(state_dict):
    return OrderedDict(
        (
            name,
            value.detach().cpu().clone()
            if torch.is_tensor(value)
            else value,
        )
        for name, value in state_dict.items()
    )


def build_router_bundle(
    router,
    ensemble,
    iteration,
    config=None,
    validation_metrics=None,
    inference_top_k=2,
):
    if inference_top_k is not None:
        inference_top_k = int(inference_top_k)
        if inference_top_k < 1 or inference_top_k > len(EXPERT_ROLES):
            raise ValueError("inference_top_k must be between 1 and 3")
    manifest = OrderedDict()
    for role in EXPERT_ROLES:
        payload = ensemble.payloads[role]
        manifest[role] = {
            "expert_state_fingerprint": payload[
                "expert_state_fingerprint"
            ],
            "trained_canonical_fingerprint": payload[
                "trained_canonical_fingerprint"
            ],
            "point_count": int(payload["point_count"]),
            "tracking_arch_version": payload["tracking_arch_version"],
            "validation_psnr": float(
                payload["validation_metrics"]["psnr"]
            ),
        }
    return {
        "format": ROUTER_BUNDLE_FORMAT,
        "version": ROUTER_BUNDLE_VERSION,
        "architecture_version": ROUTER_ARCHITECTURE_VERSION,
        "iteration": int(iteration),
        "source_canonical_fingerprint": (
            ensemble.source_canonical_fingerprint
        ),
        "expert_manifest": manifest,
        "point_counts": OrderedDict(router.point_counts),
        "router_state": _cpu_state_dict(router.state_dict()),
        "inference_top_k": inference_top_k,
        "config": dict(config or {}),
        "validation_metrics": dict(validation_metrics or {}),
    }


def validate_router_bundle(payload, ensemble=None):
    if not isinstance(payload, Mapping):
        raise ValueError("Router bundle payload must be a mapping")
    if payload.get("format") != ROUTER_BUNDLE_FORMAT:
        raise ValueError("Not an EndoMoe frozen-expert Router bundle")
    if int(payload.get("version", -1)) != ROUTER_BUNDLE_VERSION:
        raise ValueError(
            "Unsupported Router bundle version: {}".format(
                payload.get("version")
            )
        )
    if payload.get("architecture_version") != ROUTER_ARCHITECTURE_VERSION:
        raise ValueError(
            "Unsupported Router architecture: {}".format(
                payload.get("architecture_version")
            )
        )
    manifest = payload.get("expert_manifest")
    point_counts = payload.get("point_counts")
    if not isinstance(manifest, Mapping) or not isinstance(
        point_counts,
        Mapping,
    ):
        raise ValueError("Router bundle is missing expert manifest")
    if tuple(manifest.keys()) != EXPERT_ROLES:
        raise ValueError("Router expert manifest order is invalid")
    if tuple(point_counts.keys()) != EXPERT_ROLES:
        raise ValueError("Router point-count order is invalid")
    for role in EXPERT_ROLES:
        required_fields = (
            "expert_state_fingerprint",
            "trained_canonical_fingerprint",
            "point_count",
            "tracking_arch_version",
            "validation_psnr",
        )
        missing_fields = [
            name for name in required_fields if name not in manifest[role]
        ]
        if missing_fields:
            raise ValueError(
                "Router manifest for '{}' is missing: {}".format(
                    role,
                    ", ".join(missing_fields),
                )
            )
        if int(manifest[role]["point_count"]) != int(point_counts[role]):
            raise ValueError(
                "Router point count does not match manifest for '{}'".format(
                    role
                )
            )
    if not isinstance(payload.get("router_state"), Mapping):
        raise ValueError("Router bundle is missing router_state")
    inference_top_k = payload.get("inference_top_k")
    if inference_top_k is not None:
        inference_top_k = int(inference_top_k)
        if inference_top_k < 1 or inference_top_k > len(EXPERT_ROLES):
            raise ValueError("Router inference_top_k must be between 1 and 3")

    if ensemble is not None:
        if (
            payload.get("source_canonical_fingerprint")
            != ensemble.source_canonical_fingerprint
        ):
            raise ValueError(
                "Router source canonical fingerprint does not match experts"
            )
        for role in EXPERT_ROLES:
            expert_payload = ensemble.payloads[role]
            if (
                manifest[role]["expert_state_fingerprint"]
                != expert_payload["expert_state_fingerprint"]
            ):
                raise ValueError(
                    "Router expert full-state fingerprint mismatch for "
                    "'{}'".format(role)
                )
            if (
                manifest[role]["trained_canonical_fingerprint"]
                != expert_payload["trained_canonical_fingerprint"]
            ):
                raise ValueError(
                    "Router expert canonical fingerprint mismatch for "
                    "'{}'".format(role)
                )
            if int(point_counts[role]) != int(expert_payload["point_count"]):
                raise ValueError(
                    "Router expert point-count mismatch for '{}'".format(role)
                )
            if (
                manifest[role]["tracking_arch_version"]
                != expert_payload["tracking_arch_version"]
            ):
                raise ValueError(
                    "Router expert architecture mismatch for '{}'".format(role)
                )
            if float(manifest[role]["validation_psnr"]) != float(
                expert_payload["validation_metrics"]["psnr"]
            ):
                raise ValueError(
                    "Router expert validation PSNR mismatch for '{}'".format(
                        role
                    )
                )
    return payload


def save_router_bundle(path, payload):
    resolved = _absolute_path(path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    torch.save(dict(payload), resolved)
    return resolved


def load_router_bundle(path, map_location="cpu", ensemble=None):
    resolved = _absolute_path(path)
    payload = torch.load(resolved, map_location=map_location)
    return validate_router_bundle(payload, ensemble=ensemble)
