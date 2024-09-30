"""Train a neural ODE with ProbDiffEq and Optax."""

import equinox
import jax
import jax.flatten_util
import jax.numpy as jnp
from probdiffeq.backend import control_flow
from probdiffeq import ivpsolvers, taylor, ivpsolve
from probdiffeq.impl import impl

from odecheckpts import ivps

import matplotlib.pyplot as plt


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    vf, u0, (t0, t1) = ivps.van_der_pol()
    solve = make_solve(vf, num_derivs=5, ode_order=2, tol=1e-5)

    # Build the noise model
    key = jax.random.PRNGKey(1)
    std = 0.1

    # Generate the truth
    save_at_data = jnp.linspace(t0, t1, endpoint=True, num=10)
    solution = solve(u0, save_at=save_at_data)

    # Generate the data
    key, subkey = jax.random.split(key, num=2)
    noise = std * jax.random.normal(subkey, shape=solution.u.shape)
    data = solution.u + noise

    # Compute the truth (again, but at higher resolution for plotting)
    save_at_plot = jnp.linspace(t0, t1, endpoint=True, num=200)
    solution_plot = solve(u0, save_at=save_at_plot)

    # Create an initial guess
    key, subkey = jax.random.split(key, num=2)
    u0_random = tree_random_like(subkey, u0)
    guess_plot = solve(u0_random, save_at=save_at_plot)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(save_at_plot, *solution_plot.u.T, label="Truth", alpha=0.1, color="C0")
    ax.plot(save_at_plot, *guess_plot.u.T, label="Initial guess", color="C1")

    ax.errorbar(save_at_data, *data.T, yerr=std * 3, linestyle="None", color="C0")

    plt.legend()

    plt.show()


def make_solve(vf, *, num_derivs: int, ode_order: int, tol: float):
    impl.select("dense", ode_shape=(1,))
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
    ts1 = ivpsolvers.correction_ts1(ode_order=ode_order)
    strategy = ivpsolvers.strategy_filter(ibm, ts1)
    solver = ivpsolvers.solver(strategy)
    ctrl = ivpsolve.control_proportional_integral()

    def solve(init, save_at):
        # Set up the initial condition
        t0, _t1 = save_at[0], save_at[-1]
        num = num_derivs + 1 - ode_order
        tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
        output_scale = jnp.ones((), dtype=float)
        init = solver.initial_condition(tcoeffs, output_scale)

        asolver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
        with control_flow.context_overwrite_while_loop(while_loop_func):
            solution = ivpsolve.solve_adaptive_save_at(
                vf, init, save_at=save_at, dt0=0.01, adaptive_solver=asolver
            )

        return solution

    return solve


def tree_random_like(key, tree):
    flat, unflatten = jax.flatten_util.ravel_pytree(tree)
    random = 10 * jax.random.normal(key, shape=flat.shape, dtype=flat.dtype)
    return unflatten(random)


def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=10_000)


if __name__ == "__main__":
    main()
