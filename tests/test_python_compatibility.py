import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ENDOMOE_RUNTIME_FILES = (
    "arguments/__init__.py",
    "arguments/endonerf/cutting_endomoeg.py",
    "arguments/endonerf/pulling_endomoeg.py",
    "gaussian_renderer/__init__.py",
    "models/endomoeg/__init__.py",
    "models/endomoeg/complete_expert.py",
    "models/endomoeg/ensemble.py",
    "models/endomoeg/expert_bundle.py",
    "models/endomoeg/inference.py",
    "models/endomoeg/joint_training.py",
    "models/endomoeg/router.py",
    "models/endomoeg/router_bundle.py",
    "models/endomoeg/router_training.py",
    "models/tracking/__init__.py",
    "models/tracking/cams_gs_moe_tracking.py",
    "models/tracking/endomoeg_experts.py",
    "models/tracking/heterogeneous_moe_tracking.py",
    "scene/deformation.py",
    "scene/gaussian_model.py",
    "scene/tracking_losses.py",
    "train.py",
    "utils/eval_utils.py",
    "utils/optimizer_utils.py",
    "utils/temporal_utils.py",
)

PEP585_BUILTINS = {
    "dict",
    "frozenset",
    "list",
    "set",
    "tuple",
    "type",
}


def _iter_annotations(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (
                list(getattr(node.args, "posonlyargs", ()))
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                if argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def test_endomoe_runtime_annotations_support_python_37():
    violations = []
    for relative_path in ENDOMOE_RUNTIME_FILES:
        path = ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for annotation in _iter_annotations(tree):
            for node in ast.walk(annotation):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in PEP585_BUILTINS
                ):
                    violations.append(
                        "{}:{} uses PEP 585 {}".format(
                            relative_path,
                            node.lineno,
                            node.value.id,
                        )
                    )
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                    violations.append(
                        "{}:{} uses PEP 604 union".format(
                            relative_path,
                            node.lineno,
                        )
                    )

    assert not violations, "\n".join(violations)
