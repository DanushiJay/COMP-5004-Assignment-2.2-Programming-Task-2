"""Finite-difference gradients, Jacobians and a backtracking line search.

Why h = 1e-5: with central differences, truncation error shrinks like h**2
while round-off error grows like eps / h (eps is about 1e-16 for float64).
The two roughly cancel near eps**(1/3), which is about 1e-5. Forward
differences prefer a smaller h (about 1e-8) but share the same default.
h is an absolute step, so it is not scale-invariant: for very large or very
small x, pass an h of a similar magnitude.
"""

import numpy as np

_METHODS = ("central", "forward")


def _validate(x, h, method):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D x of shape (n,), got {x.shape}")
    if x.size == 0:
        raise ValueError("x must not be empty")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("x must be numeric")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not h > 0:
        raise ValueError(f"h must be positive, got {h}")
    return x.astype(float)


def grad(f, x, h=1e-5, method="central"):
    """Finite-difference gradient of a scalar function.

    Parameters
    ----------
    f : callable
        ``f(x) -> float`` where ``x`` has shape ``(n,)``.
    x : array_like, shape (n,)
        Point at which to evaluate the gradient.
    h : float, default 1e-5
        Absolute step size (not scale-invariant; see module docstring).
    method : {'central', 'forward'}, default 'central'
        ``'central'``: ``(f(x+h e_i) - f(x-h e_i)) / 2h``, error O(h^2).
        ``'forward'``: ``(f(x+h e_i) - f(x)) / h``, error O(h).

    Returns
    -------
    ndarray, shape (n,)
        Estimated gradient.

    Shapes
    ------
    ``x`` is ``(n,)``; output is ``(n,)``.

    Raises
    ------
    ValueError
        If ``x`` is not a non-empty 1-D array, ``h <= 0`` or ``method`` is unknown.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time: 2n evaluations of ``f`` (central) or n + 1 (forward); space O(n).
    The loop is over coordinates, with one ``f`` call per coordinate.
    """
    x = _validate(x, h, method)
    g = np.empty_like(x)
    fx = float(f(x)) if method == "forward" else None
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        if method == "central":
            g[i] = (float(f(x + step)) - float(f(x - step))) / (2 * h)
        else:
            g[i] = (float(f(x + step)) - fx) / h
    return g


def jacobian(F, x, h=1e-5, method="central"):
    """Finite-difference Jacobian of a vector-valued function.

    Parameters
    ----------
    F : callable
        ``F(x) -> array_like`` of shape ``(m,)`` for ``x`` of shape ``(n,)``.
    x : array_like, shape (n,)
        Point at which to evaluate the Jacobian.
    h : float, default 1e-5
        Absolute step size (not scale-invariant; see module docstring).
    method : {'central', 'forward'}, default 'central'
        Difference scheme, as in `grad`.

    Returns
    -------
    ndarray, shape (m, n)
        ``J[i, j] = dF_i / dx_j``.

    Shapes
    ------
    ``x`` is ``(n,)``; output is ``(m, n)``.

    Raises
    ------
    ValueError
        If ``x`` is invalid, ``h <= 0``, ``method`` is unknown, or ``F`` does
        not return a 1-D array.
    TypeError
        If ``x`` is not numeric.

    Notes
    -----
    Time: 2n evaluations of ``F`` (central) or n + 1 (forward); space O(m n).
    """
    x = _validate(x, h, method)
    f0 = np.atleast_1d(np.asarray(F(x), dtype=float))
    if f0.ndim != 1:
        raise ValueError(f"F must return a 1-D array, got shape {f0.shape}")
    J = np.empty((f0.size, x.size))
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = h
        if method == "central":
            hi = np.atleast_1d(np.asarray(F(x + step), dtype=float))
            lo = np.atleast_1d(np.asarray(F(x - step), dtype=float))
            J[:, j] = (hi - lo) / (2 * h)
        else:
            hi = np.atleast_1d(np.asarray(F(x + step), dtype=float))
            J[:, j] = (hi - f0) / h
    return J


def line_search(f, x, direction, grad_x=None, alpha0=1.0, c=1e-4, rho=0.5, max_iter=50):
    """Backtracking line search satisfying the Armijo sufficient-decrease condition.

    Finds a step ``alpha`` with ``f(x + alpha d) <= f(x) + c * alpha * (grad . d)``
    by starting at ``alpha0`` and multiplying by ``rho`` until the condition holds.

    Parameters
    ----------
    f : callable
        ``f(x) -> float`` for ``x`` of shape ``(n,)``.
    x : array_like, shape (n,)
        Current point.
    direction : array_like, shape (n,)
        Search direction ``d``; must be a descent direction.
    grad_x : array_like, shape (n,), optional
        Gradient of ``f`` at ``x``. Estimated with `grad` if omitted.
    alpha0 : float, default 1.0
        Initial step length (> 0).
    c : float, default 1e-4
        Sufficient-decrease constant in ``(0, 1)``.
    rho : float, default 0.5
        Shrink factor in ``(0, 1)``.
    max_iter : int, default 50
        Maximum number of shrink steps.

    Returns
    -------
    float
        Accepted step length ``alpha``.

    Shapes
    ------
    ``x``, ``direction`` and ``grad_x`` are all ``(n,)``; output is a scalar.

    Raises
    ------
    ValueError
        If shapes differ, parameters are out of range, or ``direction`` is not
        a descent direction (``grad . d >= 0``).
    RuntimeError
        If no acceptable step is found within ``max_iter`` shrinks.

    Notes
    -----
    Time: at most ``max_iter + 1`` evaluations of ``f`` (plus the gradient
    estimate if ``grad_x`` is omitted); space O(n).
    """
    x = _validate(x, 1e-5, "central")
    d = np.asarray(direction, dtype=float)
    if d.shape != x.shape:
        raise ValueError(f"Expected direction shape {x.shape}, got {d.shape}")
    if not (0 < c < 1 and 0 < rho < 1 and alpha0 > 0):
        raise ValueError("Require 0 < c < 1, 0 < rho < 1 and alpha0 > 0")
    g = grad(f, x) if grad_x is None else np.asarray(grad_x, dtype=float)
    if g.shape != x.shape:
        raise ValueError(f"Expected grad_x shape {x.shape}, got {g.shape}")
    slope = float(g @ d)
    if slope >= 0:
        raise ValueError("direction is not a descent direction (grad . d >= 0)")
    fx = float(f(x))
    alpha = float(alpha0)
    for _ in range(max_iter):
        if float(f(x + alpha * d)) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    raise RuntimeError(f"line search did not converge in {max_iter} iterations")
