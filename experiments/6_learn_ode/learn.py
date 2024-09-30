"""Train a neural ODE with ProbDiffEq and Optax."""

import equinox
import jax
import jax.flatten_util
import jax.numpy as jnp
from probdiffeq.backend import control_flow
from probdiffeq import ivpsolvers, taylor, ivpsolve
from probdiffeq.impl import impl

from odecheckpts import ivps


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # Parameters
    num_epochs = 10

    vf, u0, (t0, t1) = ivps.van_der_pol()
    save_at = jnp.linspace(t0, t1, endpoint=True, num=10)
    solve = make_solve(vf, save_at=save_at, num_derivs=5, ode_order=2, tol=1e-2)
    solution = solve(u0)

    print(solution)
    assert False


def make_solve(vf, *, save_at, num_derivs: int, ode_order: int, tol: float):
    impl.select("dense", ode_shape=(1,))
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
    ts1 = ivpsolvers.correction_ts1(ode_order=ode_order)
    strategy = ivpsolvers.strategy_filter(ibm, ts1)
    solver = ivpsolvers.solver(strategy)
    ctrl = ivpsolve.control_proportional_integral()

    # Set up the initial condition
    t0, t1 = save_at[0], save_at[-1]
    num = num_derivs + 1 - ode_order

    def solve(init):
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


def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


if __name__ == "__main__":
    main()
