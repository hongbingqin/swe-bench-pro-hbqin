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

_PREDICT = ("predict", "surrogate", "predict_score", "evaluate", "quad_predict",
            "fitted_value", "value")
_PROPOSE = ("propose", "propose_next", "next", "suggest", "ask", "next_point",
            "propose_point", "select")
_OPTIMIZE = ("optimize", "run", "search", "minimize", "maximize",
             "run_optimization", "fit")


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
    assert found, "train.py must expose an optimizer class with predict + propose methods"
    return found[0]


def _construct(cls, bounds):
    lo, hi = bounds
    for a, k in [((bounds,), {}), (tuple(bounds), {}), ((), {"bounds": bounds}),
                 ((lo, hi), {}), ((), {"lower": lo, "upper": hi}),
                 ((), {"low": lo, "high": hi})]:
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
    for a, k in [((objective,), {"n_iter": n_iter, "seed": seed}),
                 ((objective,), {"max_iters": n_iter, "seed": seed}),
                 ((objective,), {"budget": n_iter, "seed": seed}),
                 ((objective, n_iter, seed), {}),
                 ((objective, n_iter), {"seed": seed})]:
        try:
            return f(*a, **k)
        except TypeError:
            continue
    raise AssertionError("could not call the optimize method")


def _extract_best(r):
    if isinstance(r, tuple):
        return float(r[0])
    if isinstance(r, dict):
        for k in ("best_config", "best", "best_x", "x", "argmax"):
            if k in r:
                return float(r[k])
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
    return np.polyfit(xs, ss, 2)   # [a, b, c]


# ---------------------------------------------------------------------------
# fail_to_pass
# ---------------------------------------------------------------------------

def test_module_imports_without_training():
    assert _import_train() is not None


def test_predict_is_least_squares_quadratic():
    bounds = (0.0, 10.0)
    history = [(1.0, -4.0), (3.0, 0.0), (5.0, -4.0)]   # exactly -(x-3)^2
    opt = _opt(bounds)
    a, b, c = _ls_quad(history)
    for x in (0.0, 2.0, 4.0, 7.0):
        expected = float(a * x * x + b * x + c)
        got = _predict(opt, history, x)
        assert abs(got - expected) < 1e-6, (
            "predict(%.1f)=%.4f is not the LS quadratic value %.4f "
            "(a GP/kernel/linear surrogate fails here)" % (x, got, expected))


def test_propose_vertex_when_concave():
    bounds = (0.0, 10.0)
    history = [(1.0, -4.0), (3.0, 0.0), (5.0, -4.0)]   # vertex at 3.0, a<0
    opt = _opt(bounds)
    got = _propose(opt, history, 5.0)   # best_x=3, feasible [0,8], vertex 3 interior
    assert abs(got - 3.0) < 1e-6, "concave fit: expected vertex 3.0, got %s" % got


def test_propose_clips_to_trust_region():
    bounds = (0.0, 10.0)
    history = [(0.0, -9.0), (1.0, -4.0), (2.0, -1.0)]  # -(x-3)^2; best_x=2; vertex 3
    opt = _opt(bounds)
    got = _propose(opt, history, 0.5)   # feasible [1.5, 2.5] -> clip vertex 3 to 2.5
    assert abs(got - 2.5) < 1e-6, "vertex should clip to feasible 2.5, got %s" % got


def test_propose_convex_fallback_to_endpoint():
    bounds = (0.0, 10.0)
    history = [(0.0, 4.0), (2.0, 0.0), (5.0, 9.0)]     # (x-2)^2; a>=0; best_x=5
    opt = _opt(bounds)
    # feasible [2, 8]; predicted q(2)=0 < q(8)=36 -> choose endpoint 8.0
    got = _propose(opt, history, 3.0)
    assert abs(got - 8.0) < 1e-6, "convex fallback: expected endpoint 8.0, got %s" % got


def test_optimize_loop_bounded_and_deterministic():
    bounds = (0.0, 10.0)

    def objective(x):
        return -(x - 3.0) ** 2

    r1 = _optimize(_opt(bounds), objective, 12, 0)
    r2 = _optimize(_opt(bounds), objective, 12, 0)
    b1, b2 = _extract_best(r1), _extract_best(r2)
    assert b1 == b2, "same seed must give the same result (%s vs %s)" % (b1, b2)
    assert 0.0 - 1e-9 <= b1 <= 10.0 + 1e-9, "best %s outside bounds" % b1
    hist = _extract_history(r1)
    if hist is not None:
        assert 3 <= len(hist) <= 12 + 3, "history length %d not bounded by cap" % len(hist)
    assert objective(b1) >= -1.0, "optimizer did not get near the optimum: best=%s" % b1


# ---------------------------------------------------------------------------
# pass_to_pass — existing single-config training path unchanged.
# ---------------------------------------------------------------------------

def test_existing_rnn_training_path():
    from model import RNNModel
    from net import Net
    import torch

    net = Net(
        module=RNNModel,
        ntokens=40,
        module__rnn_type='LSTM',
        module__ntoken=40,
        module__ninp=16,
        module__nhid=16,
        module__nlayers=1,
        max_epochs=1,
        batch_size=4,
    )
    net.initialize()

    assert hasattr(net.module_, 'init_hidden')
    hidden = net.module_.init_hidden(4)
    X = torch.randint(0, 40, (5, 4), dtype=torch.long)
    output, _ = net.module_(X, hidden)
    assert tuple(output.shape) == (5, 4, 40)
