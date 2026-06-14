import hashlib
import os
from typing import Any, Dict, Mapping, Optional

import torch


CANONICAL_BUNDLE_FORMAT = "endomoeg_canonical_bundle"
EXPERT_BUNDLE_FORMAT = "endomoeg_complete_expert_bundle"
CANONICAL_BUNDLE_VERSION = 1
EXPERT_BUNDLE_VERSION = 6
EXPERT_ARCHITECTURE_VERSION = "endomoeg_heterogeneous_residual_expert_v6"
EXPERT_ROLES = ("global", "local", "contact")
EXPERT_TRACKING_ARCHITECTURES = {
    "global": "endomoeg_complete_global_v1",
    "local": "endomoeg_complete_local_v5",
    "contact": "endomoeg_complete_contact_v4",
}


def _absolute_path(path):
    path_value = os.fspath(path)
    if not os.path.isabs(path_value):
        raise ValueError("Bundle path must be absolute")
    return os.path.abspath(path_value)


def _update_hash_from_tensor(digest, name, value):
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())


def _update_hash_from_value(digest, name, value):
    if torch.is_tensor(value):
        _update_hash_from_tensor(digest, name, value)
        return
    digest.update(name.encode("utf-8"))
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value.keys(), key=str):
            _update_hash_from_value(
                digest,
                "{}.{}".format(name, key),
                value[key],
            )
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("utf-8"))
        for index, item in enumerate(value):
            _update_hash_from_value(
                digest,
                "{}[{}]".format(name, index),
                item,
            )
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def canonical_fingerprint(canonical_state):
    required = (
        "xyz",
        "features_dc",
        "features_rest",
        "scaling",
        "rotation",
        "opacity",
        "deformation_table",
    )
    missing = [name for name in required if name not in canonical_state]
    if missing:
        raise ValueError(
            "Cannot fingerprint canonical state; missing fields: {}".format(
                ", ".join(missing)
            )
        )
    digest = hashlib.sha256()
    for name in required:
        _update_hash_from_tensor(digest, name, canonical_state[name])
    return digest.hexdigest()


def expert_fingerprint(expert_state):
    required = (
        "canonical",
        "deformation",
        "deformation_accum",
        "tracking_type",
        "tracking_arch_version",
        "spatial_context",
    )
    missing = [name for name in required if name not in expert_state]
    if missing:
        raise ValueError(
            "Cannot fingerprint expert state; missing fields: {}".format(
                ", ".join(missing)
            )
        )
    digest = hashlib.sha256()
    for name in required:
        _update_hash_from_value(digest, name, expert_state[name])
    return digest.hexdigest()


def build_canonical_bundle(gaussians, iteration, config=None):
    canonical_state = gaussians.capture_canonical_state()
    return {
        "format": CANONICAL_BUNDLE_FORMAT,
        "version": CANONICAL_BUNDLE_VERSION,
        "iteration": int(iteration),
        "canonical_fingerprint": canonical_fingerprint(canonical_state),
        "canonical_state": canonical_state,
        "config": dict(config or {}),
    }


def build_expert_bundle(
    gaussians,
    role,
    source_canonical_fingerprint,
    iteration,
    config=None,
    validation_metrics=None,
):
    normalized_role = str(role).strip().lower()
    if normalized_role not in EXPERT_ROLES:
        raise ValueError(
            "Unsupported EndoMoe expert role '{}'; expected one of {}".format(
                role,
                ", ".join(EXPERT_ROLES),
            )
        )
    if not source_canonical_fingerprint:
        raise ValueError("source_canonical_fingerprint is required")

    expert_state = gaussians.capture_expert_state()
    return {
        "format": EXPERT_BUNDLE_FORMAT,
        "version": EXPERT_BUNDLE_VERSION,
        "architecture_version": EXPERT_ARCHITECTURE_VERSION,
        "role": normalized_role,
        "iteration": int(iteration),
        "source_canonical_fingerprint": str(source_canonical_fingerprint),
        "trained_canonical_fingerprint": canonical_fingerprint(
            expert_state["canonical"]
        ),
        "expert_state_fingerprint": expert_fingerprint(expert_state),
        "point_count": int(expert_state["canonical"]["xyz"].shape[0]),
        "tracking_type": expert_state["tracking_type"],
        "tracking_arch_version": expert_state["tracking_arch_version"],
        "expert_state": expert_state,
        "config": dict(config or {}),
        "validation_metrics": dict(validation_metrics or {}),
    }


def validate_canonical_bundle(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Canonical bundle payload must be a mapping")
    if payload.get("format") != CANONICAL_BUNDLE_FORMAT:
        raise ValueError("Not an EndoMoe canonical bundle")
    if int(payload.get("version", -1)) != CANONICAL_BUNDLE_VERSION:
        raise ValueError(
            "Unsupported canonical bundle version: {}".format(
                payload.get("version")
            )
        )
    canonical_state = payload.get("canonical_state")
    if not isinstance(canonical_state, Mapping):
        raise ValueError("Canonical bundle is missing canonical_state")
    actual_fingerprint = canonical_fingerprint(canonical_state)
    if payload.get("canonical_fingerprint") != actual_fingerprint:
        raise ValueError("Canonical bundle fingerprint mismatch")
    return payload


def validate_expert_bundle(
    payload,
    expected_role=None,
    expected_source_fingerprint=None,
    minimum_psnr=None,
):
    if not isinstance(payload, Mapping):
        raise ValueError("Expert bundle payload must be a mapping")
    if payload.get("format") != EXPERT_BUNDLE_FORMAT:
        raise ValueError(
            "Not a complete EndoMoe expert bundle. "
            "Legacy residual component checkpoints are not accepted."
        )
    if int(payload.get("version", -1)) != EXPERT_BUNDLE_VERSION:
        raise ValueError(
            "Unsupported expert bundle version: {}".format(payload.get("version"))
        )
    if payload.get("architecture_version") != EXPERT_ARCHITECTURE_VERSION:
        raise ValueError(
            "Unsupported expert architecture: {}".format(
                payload.get("architecture_version")
            )
        )

    role = str(payload.get("role", "")).lower()
    if role not in EXPERT_ROLES:
        raise ValueError("Expert bundle has an invalid role: {}".format(role))
    if expected_role is not None and role != str(expected_role).lower():
        raise ValueError(
            "Expected expert role '{}', got '{}'".format(expected_role, role)
        )

    source_fingerprint = payload.get("source_canonical_fingerprint")
    if not source_fingerprint:
        raise ValueError("Expert bundle is missing source canonical fingerprint")
    if (
        expected_source_fingerprint is not None
        and source_fingerprint != expected_source_fingerprint
    ):
        raise ValueError(
            "Expert source canonical fingerprint does not match the ensemble"
        )

    expert_state = payload.get("expert_state")
    if not isinstance(expert_state, Mapping):
        raise ValueError("Expert bundle is missing expert_state")
    expected_tracking_arch = EXPERT_TRACKING_ARCHITECTURES[role]
    if payload.get("tracking_type") != "endomoeg_expert":
        raise ValueError("Expert bundle has an invalid tracking type")
    if expert_state.get("tracking_type") != payload.get("tracking_type"):
        raise ValueError("Expert bundle tracking type does not match expert state")
    if payload.get("tracking_arch_version") != expected_tracking_arch:
        raise ValueError(
            "Expert role '{}' requires tracking architecture '{}', got '{}'".format(
                role,
                expected_tracking_arch,
                payload.get("tracking_arch_version"),
            )
        )
    if expert_state.get("tracking_arch_version") != expected_tracking_arch:
        raise ValueError(
            "Expert state architecture does not match role '{}'".format(role)
        )
    canonical_state = expert_state.get("canonical")
    if not isinstance(canonical_state, Mapping):
        raise ValueError("Expert bundle is missing its trained canonical state")
    actual_fingerprint = canonical_fingerprint(canonical_state)
    if payload.get("trained_canonical_fingerprint") != actual_fingerprint:
        raise ValueError("Expert trained canonical fingerprint mismatch")
    actual_expert_fingerprint = expert_fingerprint(expert_state)
    if payload.get("expert_state_fingerprint") != actual_expert_fingerprint:
        raise ValueError("Expert full-state fingerprint mismatch")
    if int(payload.get("point_count", -1)) != int(
        canonical_state["xyz"].shape[0]
    ):
        raise ValueError("Expert bundle point_count does not match expert state")

    if minimum_psnr is not None:
        metrics = payload.get("validation_metrics") or {}
        if "psnr" not in metrics:
            raise ValueError(
                "Expert bundle has no validation PSNR for quality gating"
            )
        if float(metrics["psnr"]) < float(minimum_psnr):
            raise ValueError(
                "Expert PSNR {:.4f} is below required {:.4f}".format(
                    float(metrics["psnr"]),
                    float(minimum_psnr),
                )
            )
    return payload


def save_bundle(path, payload):
    resolved = _absolute_path(path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    torch.save(dict(payload), resolved)
    return resolved


def load_canonical_bundle(path, map_location="cpu"):
    payload = torch.load(_absolute_path(path), map_location=map_location)
    return validate_canonical_bundle(payload)


def load_expert_bundle(
    path,
    map_location="cpu",
    expected_role=None,
    expected_source_fingerprint=None,
    minimum_psnr=None,
):
    payload = torch.load(_absolute_path(path), map_location=map_location)
    return validate_expert_bundle(
        payload,
        expected_role=expected_role,
        expected_source_fingerprint=expected_source_fingerprint,
        minimum_psnr=minimum_psnr,
    )
