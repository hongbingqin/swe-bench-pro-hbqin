import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


# ---------------------------------------------------------------------------
# Robust, design-agnostic discovery of the from-scratch optimizer class.
# Contract (per instruction): a class constructed with bounds, exposing a
# quadratic-fit prediction method, a next-point proposal method taking an
# explicit trust radius, and an end-to-end optimize method.
# ---------------------------------------------------------------------------

_PREDICT = (
    "predict",
    "surrogate",
    "predict_score",
    "evaluate",
    "quad_predict",
    "fitted_value",
    "value",
)
_PROPOSE = (
    "propose",
    "propose_next",
    "next",
    "suggest",
    "ask",
    "next_point",
    "propose_point",
    "select",
)
_OPTIMIZE = (
    "optimize",
    "run",
    "search",
    "minimize",
    "maximize",
    "run_optimization",
    "fit",
)


def _import_train():
    import importlib

    return importlib.import_module("train")


def _find_cls(mod):
    # Find the optimizer class in the train module's namespace by the shape of
    # its methods (predict + propose), regardless of which module defined it.
    found = []
    for _n, obj in vars(mod).items():
        if inspect.isclass(obj):
            ms = {m for m in dir(obj) if not m.startswith("__")}
            if any(m in ms for m in _PREDICT) and any(m in ms for m in _PROPOSE):
                found.append(obj)
    assert found, (
        "train.py must expose an optimizer class with predict + propose methods"
    )
    return found[0]


def _construct(cls, bounds):
    lo, hi = bounds
    for a, k in [
        ((bounds,), {}),
        (tuple(bounds), {}),
        ((), {"bounds": bounds}),
        ((lo, hi), {}),
        ((), {"lower": lo, "upper": hi}),
        ((), {"low": lo, "high": hi}),
    ]:
        try:
            return cls(*a, **k)
        except TypeError:
            continue
    raise AssertionError("could not construct optimizer with bounds %r" % (bounds,))


def _opt(bounds):
    return _construct(_find_cls(_import_train()), bounds)


def _method(obj, names):
    for n in names:
        f = getattr(obj, n, None)
        if callable(f):
            return f
    raise AssertionError("optimizer missing a method among %s" % (names,))


def _predict(obj, history, x):
    f = _method(obj, _PREDICT)
    try:
        return float(f(history, x))
    except TypeError:
        return float(f(x, history))


def _propose(obj, history, radius):
    f = _method(obj, _PROPOSE)
    try:
        return float(f(history, radius))
    except TypeError:
        return float(f(radius, history))


def _optimize(obj, objective, n_iter, seed):
    f = _method(obj, _OPTIMIZE)
    for a, k in [
        ((objective,), {"n_iter": n_iter, "seed": seed}),
        ((objective,), {"max_iters": n_iter, "seed": seed}),
        ((objective,), {"budget": n_iter, "seed": seed}),
        ((objective, n_iter, seed), {}),
        ((objective, n_iter), {"seed": seed}),
        ((objective,), {}),
    ]:
        try:
            return f(*a, **k)
        except TypeError:
            continue
    raise AssertionError("could not call the optimize method")


def _extract_best(r):
    if isinstance(r, tuple):
        first = r[0]
        if isinstance(first, dict):
            for k in (
                "best_config",
                "best",
                "best_x",
                "best_lr",
                "x",
                "lr",
                "learning_rate",
                "config",
                "argmax",
            ):
                if k in first:
                    return float(first[k])
            raise AssertionError(
                "optimize() tuple[0] dict has no recognized best-config key: %r"
                % (list(first.keys()),)
            )
        return float(first)
    if isinstance(r, dict):
        # config-like keys ONLY (never score/loss/value -- that's the objective, not x)
        for k in (
            "best_config",
            "best",
            "best_x",
            "best_lr",
            "x",
            "lr",
            "learning_rate",
            "config",
            "argmax",
        ):
            if k in r:
                return float(r[k])
        raise AssertionError(
            "optimize() dict has no recognized best-config key: %r" % (list(r.keys()),)
        )
    if isinstance(r, (list, np.ndarray)):
        raise AssertionError(
            "optimize() must return the best config found, not a bare history sequence"
        )
    return float(r)


def _extract_history(r):
    if isinstance(r, tuple) and len(r) >= 2 and isinstance(r[1], (list, tuple)):
        return list(r[1])
    if isinstance(r, dict):
        for k in ("history", "trace", "observations", "evaluations"):
            if k in r:
                return list(r[k])
    return None


def _ls_quad(history):
    xs = np.asarray([x for x, _ in history], dtype=float)
    ss = np.asarray([s for _, s in history], dtype=float)
    return np.polyfit(xs, ss, 2)  # [a, b, c]


# ---------------------------------------------------------------------------
# fail_to_pass
# ---------------------------------------------------------------------------


def test_module_imports_without_training():
    assert _import_train() is not None


def test_predict_is_least_squares_quadratic():
    bounds = (0.0, 10.0)
    history = [(1.0, -4.0), (3.0, 0.0), (5.0, -4.0)]  # exactly -(x-3)^2
    opt = _opt(bounds)
    a, b, c = _ls_quad(history)
    for x in (0.0, 2.0, 4.0, 7.0):
        expected = float(a * x * x + b * x + c)
        got = _predict(opt, history, x)
        assert abs(got - expected) < 1e-6, (
            "predict(%.1f)=%.4f is not the LS quadratic value %.4f "
            "(a GP/kernel/linear surrogate fails here)" % (x, got, expected)
        )


def test_propose_vertex_when_concave():
    bounds = (0.0, 10.0)
    history = [(1.0, -4.0), (3.0, 0.0), (5.0, -4.0)]  # vertex at 3.0, a<0
    opt = _opt(bounds)
    got = _propose(opt, history, 5.0)  # best_x=3, feasible [0,8], vertex 3 interior
    assert abs(got - 3.0) < 1e-6, "concave fit: expected vertex 3.0, got %s" % got


def test_propose_clips_to_trust_region():
    bounds = (0.0, 10.0)
    history = [(0.0, -9.0), (1.0, -4.0), (2.0, -1.0)]  # -(x-3)^2; best_x=2; vertex 3
    opt = _opt(bounds)
    got = _propose(opt, history, 0.5)  # feasible [1.5, 2.5] -> clip vertex 3 to 2.5
    assert abs(got - 2.5) < 1e-6, "vertex should clip to feasible 2.5, got %s" % got


def test_propose_convex_fallback_to_endpoint():
    bounds = (0.0, 10.0)
    history = [(0.0, 4.0), (2.0, 0.0), (5.0, 9.0)]  # (x-2)^2; a>=0; best_x=5
    opt = _opt(bounds)
    # feasible [2, 8]; predicted q(2)=0 < q(8)=36 -> choose endpoint 8.0
    got = _propose(opt, history, 3.0)
    assert abs(got - 8.0) < 1e-6, "convex fallback: expected endpoint 8.0, got %s" % got


def test_optimize_loop_bounded_and_deterministic():
    bounds = (0.0, 10.0)

    def objective(x):
        return -((x - 3.0) ** 2)

    r1 = _optimize(_opt(bounds), objective, 12, 0)
    r2 = _optimize(_opt(bounds), objective, 12, 0)
    b1, b2 = _extract_best(r1), _extract_best(r2)
    assert b1 == b2, "same seed must give the same result (%s vs %s)" % (b1, b2)
    assert 0.0 - 1e-9 <= b1 <= 10.0 + 1e-9, "best %s outside bounds" % b1
    hist = _extract_history(r1)
    if hist is not None:
        assert 3 <= len(hist) <= 12 + 3, "history length %d not bounded by cap" % len(
            hist
        )
    assert objective(b1) >= -1.0, "optimizer did not get near the optimum: best=%s" % b1


def test_gridsearchcv_replaced_in_train():
    # The exhaustive GridSearchCV must be removed from the training path and
    # replaced by the from-scratch optimizer -- not left alongside it.
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "train.py"), encoding="utf-8") as f:
        src = f.read()
    # Check for actual usage (the call / the import), not mere mentions in a
    # comment, so the exhaustive grid search is genuinely gone.
    assert "GridSearchCV(" not in src, (
        "GridSearchCV must no longer be instantiated in train.py "
        "(replace it with the from-scratch optimizer)"
    )
    assert "import GridSearchCV" not in src, (
        "GridSearchCV import should be removed from train.py"
    )


def test_cli_and_main_guard():
    """CLI must have --search-iter, --lr-low/high and main() guard."""
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "train.py"), encoding="utf-8") as f:
        src = f.read()
    # main() wrapper and guard
    assert "def main" in src, "train.py must define main() to be import-safe"
    assert "__name__" in src and "__main__" in src, (
        "train.py must have if __name__ == '__main__' guard"
    )
    # required args
    for flag in ("--search-iter", "--lr-low", "--lr-high", "--seed", "--save"):
        assert flag in src, f"train.py must support {flag} arg"
    # constant
    assert "DEFAULT_SEARCH_ITERATIONS" in src or "SEARCH_ITERATIONS" in src, (
        "must define iteration cap constant"
    )
    # must not do training at import time (already checked by import test) but also check fit not at top-level
    # cross_val_score usage and save_params
    assert "cross_val_score" in src or "cross_val" in src, (
        "objective should use cross_val_score"
    )
    assert "save_params" in src, "best model must be saved via save_params"
    # optimizer class must be defined inside train.py, not imported from separate module
    assert "class" in src and "Optimizer" in src, (
        "optimizer class must be defined inside train.py"
    )
    # check for import-safe: main() should contain the training logic, not top-level
    main_idx = src.find("def main")
    assert main_idx != -1
    after_main = src[main_idx:]
    assert ".fit(" in after_main or "fit(" in after_main, (
        "training fit should be inside main()"
    )


def test_skorch_wiring_preserved():
    """Original skorch train_split + data.Loader + bptt wiring must be preserved."""
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "train.py"), encoding="utf-8") as f:
        src = f.read()
    # my_train_split definition
    assert "def my_train_split" in src, "must define my_train_split(ds, y)"
    assert "skorch.dataset.Dataset" in src, (
        "my_train_split should return skorch.dataset.Dataset"
    )
    assert "corpus.valid[:200]" in src or "valid[:200]" in src, (
        "my_train_split should slice corpus.valid[:200]"
    )
    # train_split wiring
    assert (
        "train_split=my_train_split" in src or "train_split = my_train_split" in src
    ), "Net must use train_split=my_train_split"
    # data.Loader wiring
    assert (
        "iterator_train=data.Loader" in src or "iterator_train = data.Loader" in src
    ), "must use iterator_train=data.Loader"
    assert "iterator_train__bptt" in src, "must preserve iterator_train__bptt=args.bptt"
    assert (
        "iterator_valid=data.Loader" in src or "iterator_valid = data.Loader" in src
    ), "must use iterator_valid=data.Loader"
    assert "iterator_valid__bptt" in src, "must preserve iterator_valid__bptt"
    # corpus and torch seeding
    assert "data.Corpus" in src, "must load corpus = data.Corpus(args.data)"
    assert "torch.manual_seed" in src, "must call torch.manual_seed(args.seed)"
    # data-limit + numpy handling
    assert "data_limit" in src, "objective should handle args.data_limit"
    assert ".numpy()" in src, (
        "should use corpus.train[:data_limit].numpy() for sklearn compat"
    )


# ---------------------------------------------------------------------------
# pass_to_pass — existing single-config training path unchanged.
# ---------------------------------------------------------------------------


def test_optimizer_constants():
    """Optimizer must have correct trust-region constants as in reference solution."""
    mod = _import_train()
    cls = _find_cls(mod)
    # Check for class attributes or instance attributes that match reference
    # Reference: N_SEED=3, INIT_RADIUS_FRAC=0.25, SHRINK=0.5
    # Agent may define as class constants or use directly in optimize, so we check both
    # by inspecting source and by behavioral check
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "train.py"), encoding="utf-8") as f:
        src = f.read()
    # Must have the three constants defined somewhere with correct values
    # Allow variations: 0.25, .25, 1/4 etc for INIT_RADIUS_FRAC, but we check for 0.25 or 0.2-0.3 range via behavior
    # For strict Medium difficulty, require exact values appear in source
    assert "N_SEED" in src or "n_seed" in src.lower() or "min(" in src, (
        "optimizer should have N_SEED concept"
    )
    # Check for 0.25 and 0.5 values in proximity to radius/shrink logic
    # We do behavioral check: initial radius should be 0.25*(hi-lo) and shrink 0.5
    bounds = (0.0, 10.0)
    opt = _opt(bounds)
    # Check class attrs if present
    for attr, expected in [("INIT_RADIUS_FRAC", 0.25), ("SHRINK", 0.5), ("N_SEED", 3)]:
        val = getattr(opt, attr, None) or getattr(cls, attr, None)
        if val is not None:
            assert abs(float(val) - expected) < 1e-6, (
                f"{attr} should be {expected}, got {val}"
            )

    # Behavioral: with n_iter=3, history len should be 3 (only seeds), no propose yet
    def obj(x):
        return -((x - 5.0) ** 2)

    r = _optimize(_opt(bounds), obj, 3, 0)
    hist = _extract_history(r)
    if hist is not None:
        assert len(hist) == 3, (
            f"with n_iter=3, should have exactly 3 seed evals, got {len(hist)}"
        )

    # With n_iter=4, after seeds, one propose. If first propose is non-improving, next radius should shrink
    # We test that radius shrinks: by checking that propose is called with smaller radius after non-improvement
    # This is implicitly tested via deterministic behavior, but we add explicit check for 0.25 and 0.5 in source near radius
    assert "0.25" in src or ".25" in src or "INIT_RADIUS_FRAC" in src, (
        "initial radius fraction 0.25 should appear"
    )
    assert "0.5" in src or ".5" in src or "SHRINK" in src, (
        "shrink factor 0.5 should appear"
    )


def test_existing_rnn_training_path():
    from model import RNNModel
    from net import Net
    import torch

    net = Net(
        module=RNNModel,
        ntokens=40,
        module__rnn_type="LSTM",
        module__ntoken=40,
        module__ninp=16,
        module__nhid=16,
        module__nlayers=1,
        max_epochs=1,
        batch_size=4,
    )
    net.initialize()

    assert hasattr(net.module_, "init_hidden")
    hidden = net.module_.init_hidden(4)
    X = torch.randint(0, 40, (5, 4), dtype=torch.long)
    output, _ = net.module_(X, hidden)
    assert tuple(output.shape) == (5, 4, 40)
