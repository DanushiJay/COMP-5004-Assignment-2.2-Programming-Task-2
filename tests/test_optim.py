"""Tests for numcompute.optim."""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian, line_search


def square_sum(x):
    return np.sum(x**2)


def test_grad_of_square_matches_analytical():
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(grad(square_sum, x), 2 * x, atol=1e-6)


def test_grad_forward_matches_analytical_loosely():
    x = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(grad(square_sum, x, method="forward"), 2 * x, atol=1e-3)


def test_central_more_accurate_than_forward():
    x = np.array([0.5, 1.5])
    f = lambda v: np.sum(v**3)
    exact = 3 * x**2
    err_c = np.max(np.abs(grad(f, x, h=1e-3) - exact))
    err_f = np.max(np.abs(grad(f, x, h=1e-3, method="forward") - exact))
    assert err_c < err_f


def test_grad_mixed_function_hand_computed():
    # f(x, y) = x*y + y**2 at (2, 3): df/dx = y = 3, df/dy = x + 2y = 8
    f = lambda v: v[0] * v[1] + v[1] ** 2
    np.testing.assert_allclose(grad(f, [2.0, 3.0]), [3.0, 8.0], atol=1e-6)


def test_grad_does_not_modify_input():
    x = np.array([1.0, 2.0])
    grad(square_sum, x)
    np.testing.assert_array_equal(x, [1.0, 2.0])


def test_grad_integer_input_is_handled():
    np.testing.assert_allclose(grad(square_sum, np.array([1, 2])), [2.0, 4.0], atol=1e-6)


def test_grad_invalid_method_raises():
    with pytest.raises(ValueError):
        grad(square_sum, [1.0], method="backward")


def test_grad_non_1d_x_raises():
    with pytest.raises(ValueError):
        grad(square_sum, np.ones((2, 2)))


def test_grad_non_positive_h_raises():
    with pytest.raises(ValueError):
        grad(square_sum, [1.0], h=0)


def test_jacobian_shape_n_to_m():
    F = lambda v: np.array([v[0] + v[1], v[0] * v[2], v[1] ** 2, np.sum(v)])
    J = jacobian(F, np.array([1.0, 2.0, 3.0]))
    assert J.shape == (4, 3)


def test_jacobian_values_hand_computed():
    # F(x, y) = [x*y, x + y] at (2, 3) -> [[3, 2], [1, 1]]
    F = lambda v: np.array([v[0] * v[1], v[0] + v[1]])
    np.testing.assert_allclose(jacobian(F, [2.0, 3.0]), [[3.0, 2.0], [1.0, 1.0]], atol=1e-6)


def test_jacobian_forward_close_to_central():
    F = lambda v: np.array([v[0] ** 2, v[0] * v[1]])
    x = [1.0, 2.0]
    np.testing.assert_allclose(
        jacobian(F, x, method="forward"), jacobian(F, x), atol=1e-3
    )


def test_jacobian_non_1d_output_raises():
    with pytest.raises(ValueError):
        jacobian(lambda v: np.ones((2, 2)), [1.0, 2.0])


def test_line_search_accepts_full_step_when_sufficient():
    x = np.array([1.0, 1.0])
    g = 2 * x
    # d = -g/2 steps straight to the minimum, so alpha=1 is accepted
    assert line_search(square_sum, x, -g / 2, grad_x=g) == 1.0


def test_line_search_backtracks_on_large_step():
    x = np.array([1.0])
    g = 2 * x
    alpha = line_search(square_sum, x, -g, grad_x=g, alpha0=4.0)
    assert alpha < 4.0
    assert square_sum(x - alpha * g) <= square_sum(x) - 1e-4 * alpha * 4.0


def test_line_search_estimates_gradient_when_omitted():
    alpha = line_search(square_sum, np.array([2.0, -1.0]), np.array([-1.0, 1.0]))
    assert 0 < alpha <= 1.0


def test_line_search_non_descent_direction_raises():
    with pytest.raises(ValueError):
        line_search(square_sum, np.array([1.0]), np.array([1.0]))


def test_line_search_shape_mismatch_raises():
    with pytest.raises(ValueError):
        line_search(square_sum, np.array([1.0, 2.0]), np.array([1.0]))
