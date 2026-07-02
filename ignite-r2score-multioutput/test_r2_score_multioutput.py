import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from ignite.metrics.regression import R2Score


def _to_np(res):
    if isinstance(res, torch.Tensor):
        return res.cpu().numpy()
    return res


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
@pytest.mark.parametrize("K", [2, 3, 5])
def test_multioutput_single_update(multioutput, K):
    torch.manual_seed(42)
    size = 60
    y = torch.rand(size, K)
    y_pred = torch.rand(size, K)

    m = R2Score(multioutput=multioutput)
    m.update((y_pred, y))
    res = _to_np(m.compute())

    expected = r2_score(y.numpy(), y_pred.numpy(), multioutput=multioutput)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
@pytest.mark.parametrize("K", [2, 4])
def test_multioutput_accumulation(multioutput, K):
    torch.manual_seed(7)
    size = 105
    y = torch.rand(size, K)
    y_pred = torch.rand(size, K)

    m = R2Score(multioutput=multioutput)
    batch_size = 16
    n_iters = size // batch_size + 1
    for i in range(n_iters):
        idx = i * batch_size
        yb = y[idx : idx + batch_size]
        if yb.shape[0] == 0:
            continue
        m.update((y_pred[idx : idx + batch_size], yb))
    res = _to_np(m.compute())

    expected = r2_score(y.numpy(), y_pred.numpy(), multioutput=multioutput)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
def test_multioutput_constant_target(multioutput):
    # One target is constant (SS_tot == 0). Must replicate sklearn's force_finite handling.
    y = torch.tensor(
        [[3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    y_pred = torch.tensor(
        [[3.0, 1.1], [3.0, 2.1], [2.9, 2.8], [3.05, 4.2]],
        dtype=torch.float32,
    )

    m = R2Score(multioutput=multioutput)
    m.update((y_pred, y))
    res = _to_np(m.compute())

    expected = r2_score(y.numpy(), y_pred.numpy(), multioutput=multioutput)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
def test_multioutput_constant_target_perfect(multioutput):
    # Constant target with a perfect prediction -> sklearn returns 1.0 for that output.
    y = torch.tensor(
        [[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]],
        dtype=torch.float32,
    )
    y_pred = torch.tensor(
        [[3.0, 1.1], [3.0, 2.1], [3.0, 2.8]],
        dtype=torch.float32,
    )

    m = R2Score(multioutput=multioutput)
    m.update((y_pred, y))
    res = _to_np(m.compute())

    expected = r2_score(y.numpy(), y_pred.numpy(), multioutput=multioutput)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-6)


def test_multioutput_raw_values_shape():
    torch.manual_seed(3)
    K = 4
    y = torch.rand(30, K)
    y_pred = torch.rand(30, K)

    m = R2Score(multioutput="raw_values")
    m.update((y_pred, y))
    res = m.compute()

    assert isinstance(res, torch.Tensor)
    assert res.shape == (K,)


@pytest.mark.parametrize("multioutput", ["uniform_average", "variance_weighted"])
def test_multioutput_scalar_return_type(multioutput):
    torch.manual_seed(5)
    y = torch.rand(40, 3)
    y_pred = torch.rand(40, 3)

    m = R2Score(multioutput=multioutput)
    m.update((y_pred, y))
    res = m.compute()

    assert isinstance(res, float)


def test_backward_compat_single_output_default():
    # Single-output (N,) with default multioutput returns the same scalar float
    # as the base behaviour and matches sklearn default.
    torch.manual_seed(42)
    size = 51
    y = torch.rand(size)
    y_pred = torch.rand(size)

    m = R2Score()  # default 'uniform_average'
    m.update((y_pred, y))
    res = m.compute()

    assert isinstance(res, float)
    expected = r2_score(y.numpy(), y_pred.numpy())
    assert res == pytest.approx(expected)


def test_backward_compat_single_output_2d():
    # Single-output as (N, 1) also returns a scalar float matching sklearn.
    torch.manual_seed(1)
    size = 80
    y = torch.rand(size, 1)
    y_pred = torch.rand(size, 1)

    m = R2Score()
    batch_size = 16
    n_iters = size // batch_size
    for i in range(n_iters):
        idx = i * batch_size
        m.update((y_pred[idx : idx + batch_size], y[idx : idx + batch_size]))
    res = m.compute()

    assert isinstance(res, float)
    expected = r2_score(y.numpy(), y_pred.numpy())
    assert res == pytest.approx(expected)
