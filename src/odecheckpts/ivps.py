"""Initial value problems."""

import jax.numpy as jnp
from diffeqzoo import ivps, backend
import jax


def logistic():
    if not backend.has_been_selected:
        backend.select("jax")
    f, u0, time_span, args = ivps.logistic()

    def vf(u, *, t, p):
        return f(u, *p)

    u0 = jnp.atleast_1d(u0)
    return vf, (u0,), time_span, args


def rigid_body(*, time_span=(0.0, 10.0)):
    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, time_span, args = ivps.rigid_body(time_span=time_span)

    def vf(u, *, t, p):
        return f(u, *p)

    return vf, (u0,), time_span, args


def pleiades_1st():
    if not backend.has_been_selected:
        backend.select("jax")

    f, u0, time_span, args = pleiades_2nd()

    @jax.jit
    def vf(u, *, t, p):
        x, dx = jnp.split(u, 2)
        ddx = f(x, dx, t=t, p=p)
        return jnp.concatenate([dx, ddx])

    return vf, (jnp.concatenate(u0),), time_span, args


def pleiades_2nd():
    if not backend.has_been_selected:
        backend.select("jax")

    return _pleiades()


def _pleiades():
    # fmt: off
    u0 = jnp.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
        ]
    )
    du0 = jnp.asarray(
        [
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on
    t0, t1 = 0.0, 3.0

    @jax.jit
    def vf(u, du, *, t, p):  # noqa: ARG001
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)

        # rij = jnp.where(rij == 0.0, 1.0, rij)

        mj = jnp.arange(1, 8)[None, :]
        ddx = jnp.sum(jnp.nan_to_num(mj * (xj - xi) / rij), axis=1)
        ddy = jnp.sum(jnp.nan_to_num(mj * (yj - yi) / rij), axis=1)
        return jnp.concatenate((ddx, ddy))

    return vf, (u0, du0), (t0, t1), ()


def neural_ode_mlp(*, layer_sizes: tuple):
    if not backend.has_been_selected:
        backend.select("jax")

    _, u0, time_span, args = ivps.neural_ode_mlp(layer_sizes=layer_sizes)

    def vf(u, *, t, p):
        return _mlp(*p, jnp.concatenate([u, t[None]]))

    u0 = jnp.atleast_1d(u0)
    return vf, (u0,), time_span, args


def _mlp(params, inputs):
    # A multi-layer perceptron, i.e. a fully-connected neural network.
    # Taken from: http://implicit-layers-tutorial.org/neural_odes/
    for w, b in params:
        outputs = jnp.dot(inputs, w) + b
        inputs = jax.nn.tanh(outputs)
    return outputs


def brusselator(N=20, t0=0.0, tmax=10.0):
    """Brusselator as in https://uk.mathworks.com/help/matlab/math/solve-stiff-odes.html.
    N=20 is the same default as in Matlab.
    """
    alpha = 1.0 / 50.0
    const = alpha * (N + 1) ** 2
    weights = jnp.array([1.0, -2.0, 1.0])

    def f(y, *, t, p, n=N, w=weights, c=const):
        """Evaluate the Brusselator RHS via jnp.convolve, which is equivalent to multiplication with a banded matrix."""
        u, v = y[:n], y[n:]

        # Compute (1, -2, 1)-weighted average with boundary behaviour as in the Matlab link above.
        u_pad = jnp.array([1.0])
        v_pad = jnp.array([3.0])
        u_ = jnp.concatenate([u_pad, u, u_pad])
        v_ = jnp.concatenate([v_pad, v, v_pad])
        conv_u = jnp.convolve(u_, w, mode="valid")
        conv_v = jnp.convolve(v_, w, mode="valid")

        u_new = 1.0 + u**2 * v - 4 * u + c * conv_u
        v_new = 3 * u - u**2 * v + c * conv_v
        return jnp.concatenate([u_new, v_new])

    u0 = jnp.arange(1, N + 1) / N + 1
    v0 = 3.0 * jnp.ones(N)
    y0 = jnp.concatenate([u0, v0])

    return f, (y0,), (t0, tmax), ()
