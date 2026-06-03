from collections.abc import Mapping


LEGACY_KEY_ALIASES = {
    "max_disp_shared_ratio": "max_disp_hexplane_ratio",
    "prune_interval": "pruning_interval",
}


def normalize_legacy_config_keys(values):
    normalized = {}
    for key, value in values.items():
        target_key = LEGACY_KEY_ALIASES.get(key, key)
        if target_key != key and target_key in values:
            continue
        normalized[target_key] = value
    return normalized


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        section = config.get(param)
        if not isinstance(section, Mapping):
            continue
        for key, value in normalize_legacy_config_keys(section).items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args
