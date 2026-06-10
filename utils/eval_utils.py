from typing import Sequence, TypeVar


CameraT = TypeVar("CameraT")


def select_fixed_views(
    cameras: Sequence[CameraT],
    count: int = 4,
) -> tuple[CameraT, ...]:
    camera_count = len(cameras)
    if camera_count == 0 or count <= 0:
        return ()
    if camera_count <= count:
        return tuple(cameras)
    indices = [
        round(index * (camera_count - 1) / (count - 1))
        for index in range(count)
    ]
    return tuple(cameras[index] for index in indices)
