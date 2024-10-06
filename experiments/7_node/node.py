"""Fit the stiffness-constant of VdP with adaptive steps."""

import functools

import jax
import equinox as eqx
import jax.numpy as jnp
from probdiffeq.impl import impl
from probdiffeq import ivpsolvers, taylor, ivpsolve, stats
import optax

from probdiffeq.backend import control_flow as cfl
import tqdm


class VanDerPol(eqx.Module):
    _mu: jax.Array

    def __init__(self, mu):
        self._mu = jnp.sqrt(mu)

    @property
    def mu(self):
        return self._mu**2

    def __call__(self, y, ydot, *, t):  # noqa: ARG001
        return self.mu * (ydot * (1 - y**2) - y)


def main():
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(1,))

    vdp = VanDerPol(mu=10**1)
    u0s = jnp.asarray([[2.0], [0.0]])
    t0, t1 = 0.0, 3.15
    save_at = jnp.linspace(t0, t1)

    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=1000)
    with cfl.context_overwrite_while_loop(loop):
        truth = solve(vdp, u0s, save_at=save_at)
        loss = log_likelihood(save_at=save_at, u=truth.u)

        # next: make things scale to stiff equations.
        # idea 1): more data
        # idea 2): different parametrisation of VdP.
        #  E.g. Lienhard transform: https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
        # idea 3): Add forcing term to have 2 parameters which makes learning more interesting
        # idea 4): ???

        vdp = VanDerPol(mu=1.0)
        loss = jax.jit(jax.value_and_grad(loss))

        optimizer = optax.adabelief(1e-1)
        opt_state = optimizer.init(vdp)
        progressbar = tqdm.tqdm(range(1000))
        progressbar.set_description(f"loss: {1.0:.2e}, mu={vdp.mu:.3f}")
        for _ in progressbar:
            val, grads = loss(vdp, u0s)

            updates, opt_state = optimizer.update(grads, opt_state)
            vdp = optax.apply_updates(vdp, updates)
            progressbar.set_description(f"loss: {val:.2e}, mu={vdp.mu:.3f}")


def log_likelihood(save_at, u):
    def loss(ode, u0s):
        solution = solve(ode, u0s, save_at=save_at)
        eps = jnp.sqrt(jnp.sqrt(jnp.finfo(u).eps))
        std = eps * jnp.ones_like(save_at)
        lml = stats.log_marginal_likelihood(
            u, standard_deviation=std, posterior=solution.posterior
        )
        return -lml

    return loss


def solve(ode, u0s, save_at):
    # Set up the solver
    num = 3
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

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
