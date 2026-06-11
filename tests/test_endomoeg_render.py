from types import SimpleNamespace

import pytest
import torch

import render as render_module


class _FakeScene:
    def __init__(
        self,
        dataset,
        gaussians,
        load_iteration=None,
        shuffle=False,
        initialize_gaussians=True,
    ):
        assert gaussians is None
        assert initialize_gaussians is False
        del load_iteration, shuffle
        self.loaded_iter = None
        self._view = SimpleNamespace(name="test-view")
        dataset.source_path = str(dataset.source_path)

    def getTrainCameras(self):
        return [self._view]

    def getTestCameras(self):
        return [self._view]

    def getVideoCameras(self):
        return [self._view]


def _router_render_args(tmp_path):
    source_path = tmp_path / "scene"
    source_path.mkdir()
    model_path = tmp_path / "output"
    model_path.mkdir()
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    dataset = SimpleNamespace(
        sh_degree=3,
        source_path=str(source_path.resolve()),
        model_path=str(model_path.resolve()),
        white_background=False,
    )
    hyper = SimpleNamespace(
        endomoeg_pipeline_stage="router",
        endomoeg_bundle_dir=str(bundle_dir.resolve()),
        endomoeg_router_bundle="",
        endomoeg_min_expert_psnr=35.0,
        current_iteration=0,
        iterations=100,
    )
    return dataset, hyper


def test_render_sets_routes_through_frozen_router_assembly(
    monkeypatch,
    tmp_path,
):
    dataset, hyper = _router_render_args(tmp_path)
    assembly = SimpleNamespace(
        ensemble=object(),
        router=object(),
        iteration=4000,
    )
    calls = {}
    monkeypatch.setattr(
        render_module,
        "GaussianModel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Router render must not create a GaussianModel")
        ),
    )
    monkeypatch.setattr(render_module, "Scene", _FakeScene)
    monkeypatch.setattr(render_module, "get_device", lambda: torch.device("cpu"))

    def fake_load(bundle_dir, **kwargs):
        calls["bundle_dir"] = bundle_dir
        calls["load_kwargs"] = kwargs
        return assembly

    def fake_ensemble_render(
        view,
        ensemble,
        router,
        pipeline,
        background,
    ):
        calls["ensemble_render"] = {
            "view": view,
            "ensemble": ensemble,
            "router": router,
            "pipeline": pipeline,
            "background": background,
        }
        return {
            "render": torch.zeros(3, 2, 2),
            "depth": torch.ones(1, 2, 2),
        }

    def fake_render_set(
        model_path,
        name,
        iteration,
        views,
        gaussians,
        pipeline,
        background,
        reconstruct=False,
        render_view=None,
    ):
        del model_path, gaussians, pipeline, background, reconstruct
        calls["render_set"] = (name, iteration)
        assert render_view is not None
        output = render_view(views[0])
        assert output["render"].shape == (3, 2, 2)

    monkeypatch.setattr(
        render_module,
        "load_frozen_router_assembly",
        fake_load,
    )
    monkeypatch.setattr(
        render_module,
        "render_frozen_expert_ensemble",
        fake_ensemble_render,
    )
    monkeypatch.setattr(render_module, "render_set", fake_render_set)

    render_module.render_sets(
        dataset,
        hyper,
        iteration=-1,
        pipeline=SimpleNamespace(),
        skip_train=True,
        skip_test=False,
        skip_video=True,
    )

    assert calls["render_set"] == ("test", 4000)
    assert calls["load_kwargs"]["expected_source_path"] == dataset.source_path
    assert hyper.current_iteration == 4000
    assert hyper.iterations == 4000


def test_render_sets_rejects_router_iteration_mismatch(
    monkeypatch,
    tmp_path,
):
    dataset, hyper = _router_render_args(tmp_path)
    monkeypatch.setattr(
        render_module,
        "GaussianModel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Router render must not create a GaussianModel")
        ),
    )
    monkeypatch.setattr(render_module, "Scene", _FakeScene)
    monkeypatch.setattr(render_module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        render_module,
        "load_frozen_router_assembly",
        lambda *args, **kwargs: SimpleNamespace(
            ensemble=object(),
            router=object(),
            iteration=4000,
        ),
    )

    with pytest.raises(ValueError, match="does not match Router bundle"):
        render_module.render_sets(
            dataset,
            hyper,
            iteration=3000,
            pipeline=SimpleNamespace(),
            skip_train=True,
            skip_test=True,
            skip_video=True,
        )


def test_render_sets_loads_joint_output_assembly(monkeypatch, tmp_path):
    dataset, hyper = _router_render_args(tmp_path)
    joint_dir = tmp_path / "joint-bundles"
    joint_dir.mkdir()
    hyper.endomoeg_pipeline_stage = "joint"
    hyper.endomoeg_joint_output_dir = str(joint_dir.resolve())
    calls = {}
    monkeypatch.setattr(
        render_module,
        "GaussianModel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Joint render must not create a GaussianModel")
        ),
    )
    monkeypatch.setattr(render_module, "Scene", _FakeScene)
    monkeypatch.setattr(render_module, "get_device", lambda: torch.device("cpu"))

    def fake_load(bundle_dir, **kwargs):
        calls["bundle_dir"] = bundle_dir
        calls["router_bundle_path"] = kwargs["router_bundle_path"]
        return SimpleNamespace(
            ensemble=object(),
            router=object(),
            iteration=500,
        )

    monkeypatch.setattr(
        render_module,
        "load_frozen_router_assembly",
        fake_load,
    )

    render_module.render_sets(
        dataset,
        hyper,
        iteration=-1,
        pipeline=SimpleNamespace(),
        skip_train=True,
        skip_test=True,
        skip_video=True,
    )

    assert calls["bundle_dir"] == str(joint_dir.resolve())
    assert calls["router_bundle_path"] == str(
        (joint_dir / "router.pth").resolve()
    )
