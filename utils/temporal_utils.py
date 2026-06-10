from bisect import bisect_left
from typing import Iterable, Optional, Tuple


def sorted_unique_times(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(sorted({float(value) for value in values}))


def nearest_adjacent_time(
    current_time: float,
    available_times: Iterable[float],
    tolerance: float = 1e-8,
) -> Optional[float]:
    times = sorted_unique_times(available_times)
    if len(times) < 2:
        return None

    current_time = float(current_time)
    insertion = bisect_left(times, current_time)
    candidates = []
    for index in (insertion - 1, insertion, insertion + 1):
        if 0 <= index < len(times):
            candidate = times[index]
            if abs(candidate - current_time) > tolerance:
                candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - current_time))
