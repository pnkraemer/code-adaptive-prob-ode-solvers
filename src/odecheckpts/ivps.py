"""Initial value problems."""

import jax.numpy as jnp
from diffeqzoo import ivps, backend


def logistic():
    backend.select("jax")
    f, u0, time_span, args = ivps.logistic()

    def vf(u, t, p):
        return f(u, *p)

    u0 = jnp.atleast_1d(u0)
    return vf, u0, time_span, args


def rigid_body(*, time_span=(0.0, 10.0)):
    backend.select("jax")
    f, u0, time_span, args = ivps.rigid_body(time_span=time_span)

    def vf(u, t, p):
        return f(u, *p)

    return vf, u0, time_span, args
