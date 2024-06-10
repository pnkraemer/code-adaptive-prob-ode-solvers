"""Initial value problems."""

from diffeqzoo import ivps, backend


def logistic():
    backend.select("jax")
    f, *others = ivps.logistic()

    def vf(u, t, p):
        return f(u, *p)

    return vf, *others
