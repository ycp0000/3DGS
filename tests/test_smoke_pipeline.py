import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.device_utils import get_device, get_device_str


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_device_utils_default_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("ENDOGAUSSIAN_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device().type == "cpu"
    assert get_device_str() == "cpu"


def test_render_module_uses_cpu_background_when_cuda_unavailable(monkeypatch):
    class _FakeRasterizer:
        last_kwargs = None

        def __init__(self, raster_settings):
            self.raster_settings = raster_settings

        def __call__(self, **kwargs):
            type(self).last_kwargs = kwargs
            count = kwargs["means3D"].shape[0]
            return torch.zeros(1, 2, 2), torch.ones(count), torch.zeros(2, 2)

    raster_module = ModuleType("diff_gaussian_rasterization")

    class _RasterSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    scene_module = ModuleType("scene")
    scene_gaussian_model_module = ModuleType("scene.gaussian_model")
    scene_gaussian_model_module.GaussianModel = object
    scene_module.gaussian_model = scene_gaussian_model_module
    raster_module.GaussianRasterizationSettings = _RasterSettings
    raster_module.GaussianRasterizer = _FakeRasterizer
    monkeypatch.setitem(sys.modules, "scene", scene_module)
    monkeypatch.setitem(sys.modules, "scene.gaussian_model", scene_gaussian_model_module)
    monkeypatch.setitem(sys.modules, "diff_gaussian_rasterization", raster_module)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    renderer = _load_module("gaussian_renderer_smoke", "gaussian_renderer/__init__.py")

    class _FakeDeformation:
        def __call__(self, means3d, scales, rotations, opacity, time):
            return means3d, scales, rotations, opacity

        def get_aux_outputs(self):
            return {}

    point_cloud = SimpleNamespace(
        get_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
        _opacity=torch.tensor([[0.1]]),
        _scaling=torch.tensor([[1.0, 1.0, 1.0]]),
        _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        _deformation_table=torch.tensor([True]),
        _deformation_accum=torch.zeros(1, 3),
        _deformation=_FakeDeformation(),
        active_sh_degree=0,
        max_sh_degree=0,
        scaling_activation=lambda value: value,
        rotation_activation=lambda value: value,
        opacity_activation=lambda value: value,
        get_features=torch.zeros(1, 1, 3),
    )
    camera = SimpleNamespace(
        FoVx=0.5,
        FoVy=0.5,
        image_height=2,
        image_width=2,
        world_view_transform=torch.eye(4),
        full_proj_transform=torch.eye(4),
        camera_center=torch.zeros(3),
        time=0.0,
    )
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    outputs = renderer.render(camera, point_cloud, pipe, torch.zeros(3), override_color=torch.zeros(1, 3), stage="fine")

    assert outputs["render"].shape == (1, 2, 2)
    assert outputs["depth"].shape == (1, 2, 2)
    assert _FakeRasterizer.last_kwargs["means3D"].device.type == "cpu"


def test_metrics_helpers_run_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    metrics_module = _load_module("metrics_smoke", "metrics.py")
    tensor = metrics_module.array2tensor([[1.0, 2.0]])
    assert tensor.device.type == "cpu"


@pytest.mark.parametrize("script_path", ["train.py", "render.py", "metrics.py"])
def test_core_entrypoints_import_without_gpu(monkeypatch, script_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)
    if "simple_knn._C" not in sys.modules:
        simple_knn_module = ModuleType("simple_knn")
        simple_knn_c_module = ModuleType("simple_knn._C")
        simple_knn_c_module.distCUDA2 = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("distCUDA2 unavailable"))
        simple_knn_module._C = simple_knn_c_module
        sys.modules.setdefault("simple_knn", simple_knn_module)
        sys.modules["simple_knn._C"] = simple_knn_c_module
    _load_module(f"smoke_{Path(script_path).stem}", script_path)
