from typing import Dict

import torch


def collect_optimizer_group_metrics(optimizer) -> Dict[str, torch.Tensor]:
    metrics: Dict[str, torch.Tensor] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group_{group_index}"))
        params = tuple(group.get("params", ()))
        reference = next(
            (parameter for parameter in params if torch.is_tensor(parameter)),
            None,
        )
        if reference is None:
            continue

        total_parameter_count = sum(parameter.numel() for parameter in params)
        gradient_parameter_count = 0
        gradient_norm_squared = reference.new_zeros((), dtype=torch.float32)
        for parameter in params:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            gradient_parameter_count += parameter.numel()
            gradient_norm_squared = gradient_norm_squared + gradient.square().sum()

        metrics[f"lr_group_{group_name}"] = reference.new_tensor(
            float(group.get("lr", 0.0))
        )
        metrics[f"grad_norm_group_{group_name}"] = gradient_norm_squared.sqrt()
        metrics[f"grad_coverage_group_{group_name}"] = reference.new_tensor(
            gradient_parameter_count / max(total_parameter_count, 1)
        )
    return metrics
