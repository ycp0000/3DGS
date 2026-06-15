"""Bundle metric coherence regression.

Guards the contract added in commit "fix(model): bundle metric capture
must follow state": every metric written to a bundle must be measured
on the *same* model state that is persisted, not on a stale snapshot
from earlier in the training loop.

The historical failure mode was:

1. ``training_report`` measures PSNR on the live model at iteration K.
2. The training loop then runs ``densify`` / ``prune`` at iteration K
   (because the loop did not treat the final iteration as
   topology-frozen).
3. ``capture_expert_state`` stores the post-prune model.
4. ``build_expert_bundle(... validation_metrics=stale_metrics ...)``
   writes the pre-prune metric next to the post-prune state.

Stage 3 then reloads the bundle, re-renders, gets a much lower PSNR,
and aborts via ``endomoeg_residual_max_baseline_psnr_drop``. The user
sees a parity error that reads as "Local training is broken" but the
true cause sits two stages upstream.

These tests assert two invariants on the new code path:

* ``evaluate_fixed_view_metrics`` is a deterministic pure function of
  the current model state and the supplied views.
* The metric-capture helper used by bundle writes returns identical
  values for a model and for any round-tripped copy of that same
  model, regardless of mutations applied to the *original* live model
  after the round-trip.

The tests intentionally bypass the real CUDA renderer by patching the
render entry point with a small Python stub that depends only on the
Gaussian model parameters and the camera. This keeps the regression
fast and CPU-only, while still exercising the exact code path used by
``measure_bundle_metrics``.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def _make_camera(image: torch.Tensor, mask: torch.Tensor):
    return SimpleNamespace(
        original_image=image,
        mask=mask,
    )


class _DummyScene:
    def __init__(self, test_cameras, train_cameras=None):
        self._test = list(test_cameras)
        self._train = list(train_cameras or test_cameras)

    def getTestCameras(self):
        return self._test

    def getTrainCameras(self):
        return self._train


class _DummyGaussians(nn.Module):
    """Minimum surface area: a single per-pixel scalar bias.

    The render stub returns ``ground_truth + bias``; this lets us drive
    PSNR deterministically by mutating ``bias`` and verify that the
    metric helper reflects only the parameters present in the model
    object passed to it.
    """

    def __init__(self, bias: float, image_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.full(image_shape, float(bias)))

    def state_snapshot(self):
        return {"bias": self.bias.detach().clone()}

    def load_state_snapshot(self, snapshot):
        with torch.no_grad():
            self.bias.copy_(snapshot["bias"])


def _install_render_stub(monkeypatch):
    """Patch ``gaussian_renderer.render`` for the duration of the test.

    The stub must be installed before ``utils.eval_utils`` is imported,
    since ``evaluate_fixed_view_metrics`` performs a local import. The
    ``render`` is not part of any test's assertion surface; it just
    needs to be deterministic in the model parameter ``bias``.
    """

    fake_module = types.ModuleType("gaussian_renderer")

    def fake_render(viewpoint, gaussians, pipe, background, **_):
        bias = gaussians.bias
        gt = viewpoint.original_image.to(bias.device).float()
        return {
            "render": (gt + bias).clamp(0.0, 1.0),
        }

    fake_module.render = fake_render
    monkeypatch.setitem(sys.modules, "gaussian_renderer", fake_module)


@pytest.fixture
def coherence_setup(monkeypatch):
    _install_render_stub(monkeypatch)
    image_shape = (3, 4, 4)
    image = torch.full(image_shape, 0.5)
    mask = torch.ones((1, 4, 4))
    cameras = [_make_camera(image, mask) for _ in range(2)]
    scene = _DummyScene(test_cameras=cameras)
    background = torch.zeros(3)
    pipe = SimpleNamespace(debug=False)
    return scene, pipe, background, image_shape


def test_metric_helper_is_pure_function_of_model_state(coherence_setup):
    """If two models hold the same parameters they must produce the
    same metrics, even if one of them was mutated *after* a snapshot
    was taken.
    """
    from utils.eval_utils import evaluate_fixed_view_metrics

    scene, pipe, background, image_shape = coherence_setup
    model = _DummyGaussians(bias=0.05, image_shape=image_shape)

    pristine = evaluate_fixed_view_metrics(
        scene, model, pipe, background, splits=("test",)
    )
    snapshot = model.state_snapshot()

    # Mutate the live model the same way ``densify`` / ``prune`` does
    # at the final iteration: in place, after the metric was taken.
    with torch.no_grad():
        model.bias.add_(0.4)
    perturbed = evaluate_fixed_view_metrics(
        scene, model, pipe, background, splits=("test",)
    )
    assert pristine["test"]["psnr"] != perturbed["test"]["psnr"]

    # Restoring the snapshot must restore the metrics exactly.
    model.load_state_snapshot(snapshot)
    restored = evaluate_fixed_view_metrics(
        scene, model, pipe, background, splits=("test",)
    )
    assert restored["test"]["psnr"] == pytest.approx(pristine["test"]["psnr"])
    assert restored["test"]["l1"] == pytest.approx(pristine["test"]["l1"])


def test_bundle_metric_capture_reflects_post_mutation_state(coherence_setup):
    """The bundle-write helper must report the *current* model, not a
    cached metric from a previous call.
    """
    from utils.eval_utils import measure_bundle_metrics

    scene, pipe, background, image_shape = coherence_setup
    model = _DummyGaussians(bias=0.0, image_shape=image_shape)

    before = measure_bundle_metrics(scene, model, pipe, background)
    with torch.no_grad():
        model.bias.add_(0.25)
    after = measure_bundle_metrics(scene, model, pipe, background)

    # Mutating the live model between two helper calls must change the
    # reported metric. Without this, a stale call site would silently
    # write the wrong PSNR into a bundle.
    assert before["psnr"] != pytest.approx(after["psnr"])
    assert before["psnr"] > after["psnr"]


def test_select_fixed_views_is_stable_under_call_order():
    """``measure_bundle_metrics`` selects the same fixed views the
    bundle-loading parity check selects. If these two ever diverge,
    Stage 3's parity gate becomes meaningless. Assert deterministic
    selection so this stays true under future refactors.
    """
    from utils.eval_utils import select_fixed_views

    cameras = list(range(20))
    selected_first = select_fixed_views(cameras, count=4)
    selected_second = select_fixed_views(cameras, count=4)
    assert selected_first == selected_second
    assert len(selected_first) == 4
    assert selected_first[0] == cameras[0]
    assert selected_first[-1] == cameras[-1]
