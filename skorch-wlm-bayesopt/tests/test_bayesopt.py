import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — design-agnostic discovery of the search factory.
#
# instruction.md describes the search as "constructible/invokable in isolation"
# but does not prescribe a name, so we accept any reasonable factory name and
# verify it by behavior.
# ---------------------------------------------------------------------------

_FACTORY_NAMES = [
    "build_search", "make_search", "build_bayes_search", "make_bayes_search",
    "build_bayesian_search", "build_optimizer", "make_optimizer", "build_hpo",
    "get_search", "create_search", "bayes_search", "build_bayessearch",
]


def _import_train():
    import importlib
    train = importlib.import_module("train")
    return train


def _get_factory(train_module):
    for name in _FACTORY_NAMES:
        fn = getattr(train_module, name, None)
        if callable(fn):
            return fn
    raise AssertionError(
        "train.py must expose a callable factory that builds the search in "
        "isolation (looked for one of: %s)" % ", ".join(_FACTORY_NAMES))


def _cheap_estimator():
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(random_state=0)


def _cheap_data():
    from sklearn.datasets import make_classification
    return make_classification(n_samples=60, n_features=5, random_state=0)


def _wide_space():
    # An integer space far wider than the iteration cap, so an exhaustive grid
    # would evaluate many more points than a capped Bayesian search.
    from skopt.space import Integer
    return {"min_samples_split": Integer(2, 40)}  # 39 integer points


# ---------------------------------------------------------------------------
# fail_to_pass
# ---------------------------------------------------------------------------

def test_train_module_imports_without_training():
    # The heavy training code must be guarded so the module can be imported in
    # isolation: importing it must not parse argv, load the corpus, or train.
    train = _import_train()
    assert train is not None


def test_search_respects_iteration_cap():
    train = _import_train()
    factory = _get_factory(train)
    n_iter = 5
    search = factory(_cheap_estimator(), _wide_space(),
                     n_iter=n_iter, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    n_evaluated = len(search.cv_results_["params"])
    assert n_evaluated == n_iter, (
        "expected exactly %d configurations evaluated, got %d"
        % (n_iter, n_evaluated))


def test_search_is_not_exhaustive_grid():
    # With an integer space of 39 points and a cap of 5, a capped Bayesian
    # search must evaluate far fewer configurations than the full grid.
    train = _import_train()
    factory = _get_factory(train)
    search = factory(_cheap_estimator(), _wide_space(),
                     n_iter=5, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    full_grid = 39
    assert len(search.cv_results_["params"]) < full_grid


def test_search_surfaces_best_config_and_score():
    train = _import_train()
    factory = _get_factory(train)
    search = factory(_cheap_estimator(), _wide_space(),
                     n_iter=5, random_state=0)
    X, y = _cheap_data()
    search.fit(X, y)
    assert hasattr(search, "best_params_")
    assert hasattr(search, "best_score_")
    assert hasattr(search, "best_estimator_")
    assert set(search.best_params_.keys()) <= {"min_samples_split"}
    assert np.isfinite(search.best_score_)


# ---------------------------------------------------------------------------
# pass_to_pass — the existing single-config training path is unchanged.
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
