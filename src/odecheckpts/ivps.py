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
