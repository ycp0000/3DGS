import os
from argparse import Namespace
from collections import OrderedDict

import torch

from .expert_bundle import EXPERT_ROLES, load_expert_bundle


def freeze_gaussian_model(model):
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            value.requires_grad_(False)
    for parameter in model._deformation.parameters():
        parameter.requires_grad_(False)
    model._deformation.eval()
    model.optimizer = None
    return model


def assert_gaussian_model_frozen(model, role):
    trainable = []
    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
    ):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor) and value.requires_grad:
            trainable.append(name)
    trainable.extend(
        "deformation.{}".format(name)
        for name, parameter in model._deformation.named_parameters()
        if parameter.requires_grad
    )
    if trainable:
        raise RuntimeError(
            "Frozen expert '{}' still has trainable parameters: {}".format(
                role,
                ", ".join(trainable),
            )
        )


class FrozenExpertEnsemble:
    def __init__(self, experts, payloads, source_canonical_fingerprint):
        self.experts = OrderedDict(experts)
        self.payloads = OrderedDict(payloads)
        self.source_canonical_fingerprint = source_canonical_fingerprint
        if tuple(self.experts.keys()) != EXPERT_ROLES:
            raise ValueError(
                "Frozen expert order must be {}".format(", ".join(EXPERT_ROLES))
            )
        reference_payload = self.payloads["global"]
        reference_fingerprint = reference_payload[
            "trained_canonical_fingerprint"
        ]
        reference_sh_degree = int(
            reference_payload["expert_state"]["canonical"]["active_sh_degree"]
        )
        for role in EXPERT_ROLES[1:]:
            payload = self.payloads[role]
            if payload["trained_canonical_fingerprint"] != reference_fingerprint:
                raise ValueError(
                    "Residual expert '{}' does not share the trained Global "
                    "canonical state".format(role)
                )
            active_sh_degree = int(
                payload["expert_state"]["canonical"]["active_sh_degree"]
            )
            if active_sh_degree != reference_sh_degree:
                raise ValueError(
                    "Residual expert '{}' active SH degree does not match "
                    "Global".format(role)
                )
        for role, model in self.experts.items():
            assert_gaussian_model_frozen(model, role)

    @classmethod
    def load(
        cls,
        bundle_dir,
        minimum_psnr=0.0,
        device=None,
        expected_source_path=None,
    ):
        from scene.gaussian_model import GaussianModel

        bundle_dir_value = os.fspath(bundle_dir)
        if not os.path.isabs(bundle_dir_value):
            raise ValueError("Expert bundle directory must be absolute")
        resolved_dir = os.path.abspath(bundle_dir_value)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        experts = []
        payloads = []
        source_fingerprint = None
        for role in EXPERT_ROLES:
            payload = load_expert_bundle(
                os.path.join(resolved_dir, "{}.pth".format(role)),
                map_location="cpu",
                expected_role=role,
                expected_source_fingerprint=source_fingerprint,
                minimum_psnr=minimum_psnr,
            )
            if source_fingerprint is None:
                source_fingerprint = payload["source_canonical_fingerprint"]

            config = payload.get("config") or {}
            model_params = config.get("model_params") or {}
            hidden_params = config.get("hidden_params") or {}
            if not model_params or not hidden_params:
                raise ValueError(
                    "Expert '{}' bundle is missing reconstruction config".format(
                        role
                    )
                )
            if expected_source_path is not None:
                saved_source = os.path.abspath(
                    os.fspath(
                        config.get("source_path")
                        or model_params.get("source_path")
                        or ""
                    )
                )
                current_source = os.path.abspath(
                    os.fspath(expected_source_path)
                )
                if saved_source != current_source:
                    raise ValueError(
                        "Expert '{}' source_path '{}' does not match '{}'".format(
                            role,
                            saved_source,
                            current_source,
                        )
                    )
            if hidden_params.get("tracking_type") != "endomoeg_expert":
                raise ValueError(
                    "Expert '{}' was not trained as a complete expert".format(role)
                )
            if hidden_params.get("endomoeg_expert_role") != role:
                raise ValueError(
                    "Expert '{}' config role does not match its bundle".format(role)
                )

            model = GaussianModel(
                int(model_params.get("sh_degree", 3)),
                Namespace(**hidden_params),
            )
            model._deformation = model._deformation.to(device)
            model.restore_expert_state(
                payload["expert_state"],
                training_args=None,
            )
            freeze_gaussian_model(model)
            experts.append((role, model))
            payloads.append((role, payload))

        return cls(
            experts=experts,
            payloads=payloads,
            source_canonical_fingerprint=source_fingerprint,
        )

    def __len__(self):
        return len(self.experts)

    def __iter__(self):
        return iter(self.experts.items())

    def point_counts(self):
        return {
            role: int(model.get_xyz.shape[0])
            for role, model in self.experts.items()
        }

    def validation_psnr(self):
        return {
            role: float(payload["validation_metrics"]["psnr"])
            for role, payload in self.payloads.items()
        }

    def assert_frozen(self):
        for role, model in self.experts.items():
            assert_gaussian_model_frozen(model, role)
