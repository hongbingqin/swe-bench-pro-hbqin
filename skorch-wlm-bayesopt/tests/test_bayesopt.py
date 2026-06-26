import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


# ---------------------------------------------------------------------------
# Robust, design-agnostic discovery + invocation of the search factory.
#
# instruction.md requires the search to be "constructible/invokable in
# isolation" but does not prescribe the factory's name OR its parameter
# order/names. We therefore (1) find the factory by any reasonable name and
# (2) bind our arguments to its parameters by flexible name-matching, falling
# back to positional order for unmatched required parameters. This accepts any
# reasonable factory shape rather than one hard-coded signature.
# ---------------------------------------------------------------------------

_FACTORY_NAMES = [
    "build_search", "make_search", "build_bayes_search", "make_bayes_search",
    "build_bayesian_search", "build_optimizer", "make_optimizer", "build_hpo",
    "get_search", "create_search", "bayes_search", "build_bayessearch",
    "bayesian_search", "build_bayes_opt", "make_bayesian_search",
    "build_hyperparameter_search", "build_param_search", "search_factory",
    "build_hyperparam_search", "create_bayes_search",
]

_EST_NAMES = {"estimator", "model", "net", "est", "base_estimator", "clf",
              "regressor", "classifier", "learner", "pipeline"}
_SPACE_NAMES = {"search_spaces", "search_space", "spaces", "space",
                "param_space", "param_spaces", "param_distributions",
                "param_grid", "params", "parameters", "distributions",
                "search", "grid", "param_dist", "dimensions"}
_ITER_NAMES = {"n_iter", "n_iterations", "max_iter", "max_iters", "iterations",
               "n_calls", "budget", "cap", "max_evals", "n_trials", "num_iter",
               "iteration_cap", "n_points"}
_SEED_NAMES = {"random_state", "seed", "rng", "random_seed"}


def _import_train():
    import importlib
    return importlib.import_module("train")


def _get_factory(train_module):
    for name in _FACTORY_NAMES:
        fn = getattr(train_module, name, None)
        if callable(fn) and not isinstance(fn, type):
            return fn
    raise AssertionError(
        "train.py must expose an importable, callable factory that builds the "
        "search in isolation. Looked for one of: %s" % ", ".join(_FACTORY_NAMES))


def _invoke_factory(factory, estimator, space, n_iter, random_state):
    sig = inspect.signature(factory)
    params = [p for p in sig.parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD,
                            p.KEYWORD_ONLY)]
    used = set()
    kwargs = {}

    def bind(nameset, value):
        for p in params:
            if p.name in nameset and p.name not in used:
                used.add(p.name)
                kwargs[p.name] = value
                return True
        return False

    got_est = bind(_EST_NAMES, estimator)
    got_space = bind(_SPACE_NAMES, space)
    bind(_ITER_NAMES, n_iter)
    bind(_SEED_NAMES, random_state)

    # Fallback: assign unmatched REQUIRED params positionally (estimator first,
    # then space) so factories with unconventional names still work.
    required = [p for p in params
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                and p.name not in used]
    if not got_est and required:
        p = required.pop(0)
        kwargs[p.name] = estimator
        used.add(p.name)
    if not got_space and required:
        p = required.pop(0)
        kwargs[p.name] = space
        used.add(p.name)

    return factory(**kwargs)


def _build(estimator, space, n_iter, random_state=0):
    train = _import_train()
    factory = _get_factory(train)
    return _invoke_factory(factory, estimator, space, n_iter, random_state)


def _cheap_estimator():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=0)


def _cheap_data():
    from sklearn.datasets import make_classification
    return make_classification(n_samples=60, n_features=5, random_state=0)


def _wide_space():
    from skopt.space import Integer
    return {"min_samples_split": Integer(2, 40)}  # 39 integer points


def _n_evaluated(search):
    cv = getattr(search, "cv_results_", None)
    if cv is not None and "params" in cv:
        return len(cv["params"])
    raise AssertionError(
        "search exposes no cv_results_['params'] to count evaluated configs")


# ---------------------------------------------------------------------------
# fail_to_pass
# ---------------------------------------------------------------------------

def test_train_module_imports_without_training():
    # The heavy training code must be guarded so importing the module does not
    # parse argv, load the corpus, or train.
    train = _import_train()
    assert train is not None


def test_search_respects_iteration_cap():
    n_iter = 5
    search = _build(_cheap_estimator(), _wide_space(), n_iter, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    n = _n_evaluated(search)
    assert 2 <= n <= n_iter, (
        "expected 2..%d configurations evaluated, got %d" % (n_iter, n))


def test_search_is_not_exhaustive_grid():
    # 39-point integer space, cap of 5 -> a capped search must evaluate far
    # fewer than the full grid.
    search = _build(_cheap_estimator(), _wide_space(), 5, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    assert _n_evaluated(search) < 39


def test_search_surfaces_best_config_and_score():
    search = _build(_cheap_estimator(), _wide_space(), 5, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert "min_samples_split" in dict(search.best_params_)
    assert np.isfinite(search.best_score_)


# ---------------------------------------------------------------------------
# pass_to_pass — existing single-config training path is unchanged.
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
