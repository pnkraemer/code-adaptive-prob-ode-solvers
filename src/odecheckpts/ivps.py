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
