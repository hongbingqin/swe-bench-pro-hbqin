import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


# ---------------------------------------------------------------------------
# Robust, design-agnostic discovery of the from-scratch optimizer.
#
# instruction.md requires an optimizer *class* constructed with the bounds,
# exposing a surrogate-prediction method, a next-pick proposal method, and an
# end-to-end optimize method. Names are not prescribed, so we find the class by
# the shape of its methods and bind constructor/method arguments flexibly.
# ---------------------------------------------------------------------------

_PREDICT = ("predict", "surrogate", "surrogate_predict", "predict_score",
            "estimate", "posterior_mean", "predict_mean", "mean")
_PROPOSE = ("propose", "propose_next", "next", "suggest", "ask", "select",
            "propose_candidate", "next_point", "acquire", "choose")
_OPTIMIZE = ("optimize", "run", "search", "maximize", "minimize",
             "run_optimization", "fit")


def _import_train():
    import importlib
    return importlib.import_module("train")


def _find_optimizer_class(mod):
    found = []
    for _name, obj in vars(mod).items():
        if inspect.isclass(obj) and getattr(obj, "__module__", None) == mod.__name__:
            methods = {m for m in dir(obj) if not m.startswith("__")}
            if any(m in methods for m in _PREDICT) and any(m in methods for m in _PROPOSE):
                found.append(obj)
    assert found, (
        "train.py must define an optimizer class exposing surrogate-prediction "
        "and next-pick proposal methods")
    return found[0]


def _construct(cls, bounds):
    lo, hi = bounds
    attempts = [((bounds,), {}), (tuple(bounds), {}), ((), {"bounds": bounds}),
                ((lo, hi), {}), ((), {"lower": lo, "upper": hi}),
                ((), {"low": lo, "high": hi})]
    for args, kwargs in attempts:
        try:
            return cls(*args, **kwargs)
        except TypeError:
            continue
    raise AssertionError("could not construct optimizer with bounds %r" % (bounds,))


def _method(obj, names):
    for n in names:
        m = getattr(obj, n, None)
        if callable(m):
            return m
    raise AssertionError("optimizer is missing a method among %s" % (names,))


def _opt(bounds):
    cls = _find_optimizer_class(_import_train())
    return _construct(cls, bounds)


def _call_predict(obj, history, x):
    m = _method(obj, _PREDICT)
    try:
        return float(m(history, x))
    except TypeError:
        return float(m(x, history))


def _call_propose(obj, history, candidates):
    m = _method(obj, _PROPOSE)
    try:
        return m(history, candidates)
    except TypeError:
        return m(candidates, history)


def _call_optimize(obj, objective, n_iter, seed):
    m = _method(obj, _OPTIMIZE)
    for args, kwargs in [((objective, n_iter, seed), {}),
                         ((objective,), {"n_iter": n_iter, "seed": seed}),
                         ((objective, n_iter), {"seed": seed}),
                         ((objective,), {"budget": n_iter, "seed": seed})]:
        try:
            return m(*args, **kwargs)
        except TypeError:
            continue
    raise AssertionError("could not call the optimize method")


def _extract_best(result):
    if isinstance(result, tuple):
        return float(result[0])
    if isinstance(result, dict):
        for k in ("best_config", "best", "x", "best_x", "argmax"):
            if k in result:
                return float(result[k])
    return float(result)


def _extract_history(result):
    if isinstance(result, tuple) and len(result) >= 2:
        h = result[1]
        if isinstance(h, (list, tuple)):
            return list(h)
    if isinstance(result, dict):
        for k in ("history", "trace", "observations", "evaluations"):
            if k in result:
                return list(result[k])
    return None


# Oracle NW surrogate (mirrors the spec) used to compute expected values.
def _nw(history, x, h):
    xs = np.asarray([c for c, _ in history], dtype=float)
    ss = np.asarray([s for _, s in history], dtype=float)
    w = np.exp(-((x - xs) ** 2) / (2.0 * h ** 2))
    total = float(w.sum())
    if total < 1e-12:
        return float(ss.mean())
    return float((w * ss).sum() / total)


# ---------------------------------------------------------------------------
# fail_to_pass
# ---------------------------------------------------------------------------

def test_module_imports_without_training():
    train = _import_train()
    assert train is not None


def test_surrogate_is_nadaraya_watson():
    bounds = (0.0, 10.0)        # h = 0.1 * 10 = 1.0
    h = 0.1 * (bounds[1] - bounds[0])
    history = [(1.0, 0.0), (2.0, 1.0), (8.0, 0.0)]
    opt = _opt(bounds)
    for x in (1.5, 2.5, 5.0):
        expected = _nw(history, x, h)
        got = _call_predict(opt, history, x)
        assert abs(got - expected) < 1e-6, (
            "surrogate at x=%s: expected NW %.6f, got %.6f "
            "(not a Nadaraya-Watson kernel average)" % (x, expected, got))


def test_surrogate_zero_weight_fallback():
    bounds = (0.0, 1.0)         # h = 0.1
    history = [(0.0, 5.0), (0.0, 7.0)]
    opt = _opt(bounds)
    # x is ~10 bandwidths away -> all weights underflow -> fallback = mean(scores)
    got = _call_predict(opt, history, 1.0)
    assert abs(got - 6.0) < 1e-6, (
        "zero-weight fallback should return the mean of observed scores (6.0), "
        "got %.6f" % got)


def test_propose_maximizes_acquisition():
    bounds = (0.0, 10.0)        # h = 1.0
    h = 0.1 * (bounds[1] - bounds[0])
    history = [(1.0, 0.0), (2.0, 1.0), (8.0, 0.0)]
    candidates = [1.0, 2.0, 5.0, 8.0, 9.5]

    def nearest(x):
        return min(abs(x - c) for c, _ in history)

    acq = [_nw(history, c, h) + 0.2 * nearest(c) for c in candidates]
    expected = candidates[int(np.argmax(acq))]   # argmax -> smallest index on ties

    opt = _opt(bounds)
    got = _call_propose(opt, history, candidates)
    assert abs(float(got) - expected) < 1e-9, (
        "propose should maximize mean + 0.2*exploration -> %.3f, got %s"
        % (expected, got))


def test_optimize_loop_bounded_and_deterministic():
    # End-to-end loop mechanics, kept robust to the seeding RNG choice (the
    # exact NW/acquisition discrimination lives in the formula tests above).
    bounds = (0.0, 10.0)

    def objective(x):
        return -(x - 3.0) ** 2     # smooth bowl, maximized at x = 3.0 (score 0)

    r1 = _call_optimize(_opt(bounds), objective, 12, 0)
    r2 = _call_optimize(_opt(bounds), objective, 12, 0)
    b1, b2 = _extract_best(r1), _extract_best(r2)

    # Deterministic: same seed -> same result.
    assert b1 == b2, "same seed must give the same result (got %s vs %s)" % (b1, b2)

    # Returned config is within the bounds.
    assert 0.0 - 1e-9 <= b1 <= 10.0 + 1e-9, "best config %s outside bounds" % b1

    # The cap bounds the work (tolerant of whether the 3 seed points count
    # toward the cap): the loop must not run unbounded (e.g. the whole grid).
    hist = _extract_history(r1)
    if hist is not None:
        assert 3 <= len(hist) <= 12 + 3, "history length %d not bounded by the cap" % len(hist)

    # It actually optimizes on a smooth objective (loose band: any correct
    # history-guided optimizer converges well within this, regardless of RNG;
    # rules out a non-optimizing/garbage result).
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
