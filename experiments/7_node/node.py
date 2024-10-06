"""Fit the stiffness-constant of VdP with adaptive steps."""

import functools

import jax
import equinox as eqx
import jax.numpy as jnp
from probdiffeq.impl import impl
from probdiffeq import ivpsolvers, taylor, ivpsolve, stats

from probdiffeq.backend import control_flow as cfl


class VanDerPol(eqx.Module):
    _mu: jax.Array

    def __init__(self, mu):
        self._mu = jnp.log10(mu)

    @property
    def mu(self):
        return 10.0**self._mu

    def __call__(self, y, ydot, *, t):  # noqa: ARG001
        return self.mu * (ydot * (1 - y**2) - y)


def main():
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(1,))

    vdp = VanDerPol(mu=10**1)
    u0s = jnp.asarray([[2.0], [0.0]])
    t0, t1 = 0.0, 6.3
    save_at = jnp.linspace(t0, t1)

    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=1000)
    with cfl.context_overwrite_while_loop(loop):
        sol = solve(vdp, u0s, save_at=save_at)
        loss = log_likelihood(save_at=save_at, u=sol.u)

        loss = jax.jit(jax.value_and_grad(loss))
        print(loss(vdp, u0s))
        for _ in range(10):
            print(loss(vdp, u0s))


def log_likelihood(save_at, u):
    def loss(ode, u0s):
        solution = solve(ode, u0s, save_at=save_at)
        std = 1e-8 * jnp.ones_like(save_at)
        lml = stats.log_marginal_likelihood(
            u, standard_deviation=std, posterior=solution.posterior
        )
        return -lml

    return loss


def solve(ode, u0s, save_at):
    # Set up the solver
    num = 4
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    solver = ivpsolvers.solver(strategy)

    # Set up the initial condition
    t0 = save_at[0]
    tcoeffs = taylor.odejet_padded_scan(
        lambda *y: ode(*y, t=t0), [u0s[0], u0s[1]], num=num - 1
    )
    output_scale = 1.0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    tol = 1e-3
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    return ivpsolve.solve_adaptive_save_at(
        ode, init, save_at=save_at, dt0=0.1, adaptive_solver=adaptive_solver
    )


if __name__ == "__main__":
    main()
