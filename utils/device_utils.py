from __future__ import annotations

import os
import time

import torch


class _CPUEventStub:
    def __init__(self, enable_timing: bool = True) -> None:
        self.enable_timing = enable_timing
        self._timestamp: float | None = None

    def record(self) -> None:
        if self.enable_timing:
            self._timestamp = time.perf_counter()

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, end_event: "_CPUEventStub") -> float:
        if not self.enable_timing or self._timestamp is None or end_event._timestamp is None:
            return 0.0
        return max(0.0, (end_event._timestamp - self._timestamp) * 1000.0)


def _normalize_device_name(device_name: str | None) -> str | None:
    if device_name is None:
        return None
    normalized = str(device_name).strip().lower()
    return normalized or None


def get_device(force_cpu: bool | None = None) -> torch.device:
    if force_cpu is True:
        return torch.device("cpu")

    configured = _normalize_device_name(os.environ.get("ENDOGAUSSIAN_DEVICE"))
    if configured == "cpu":
        return torch.device("cpu")
    if configured == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_str(force_cpu: bool | None = None) -> str:
    return str(get_device(force_cpu=force_cpu))


def safe_cuda_event(enable_timing: bool = True):
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=enable_timing)
    return _CPUEventStub(enable_timing=enable_timing)
