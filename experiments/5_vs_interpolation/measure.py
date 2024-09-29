import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl
import matplotlib.pyplot as plt
from odecheckpts import ivps


class IVPSolution(NamedTuple):
    grid: jax.Array
    solution: jax.Array

    @property
    def steps(self):
        return jnp.diff(self.grid)

    @property
    def num_steps(self):
        return len(self.steps)


class TimeOutput(NamedTuple):
    runtime: float
    output: IVPSolution


def jit_and_time(fun: Callable) -> Callable:
    fun = jax.jit(fun)

    def fun_wrapped():
        out = fun()
        out.grid.block_until_ready()
        out.solution.block_until_ready()

        t0 = time.perf_counter()
        out = fun()
        out.grid.block_until_ready()
        out.solution.block_until_ready()
        t1 = time.perf_counter()
        return TimeOutput(runtime=t1 - t0, output=out)

    return fun_wrapped


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # Set up the IVP
    vf, (u0, du0), (t0, t1) = ivps.three_body_restricted()

    # Set up the solver
    impl.select("blockdiag", ode_shape=(2,))
    num = 3
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts0(ode_order=2)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

    # Set up the initial condition
    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), [u0, du0], num=num - 1)
    output_scale = jnp.ones((2,), dtype=float)
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    tol = 1e-3
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
    )
    baseline = IVPSolution(grid=solution.t, solution=solution.u)
    print(baseline)

    plt.plot(*baseline.solution.T)
    plt.show()


if __name__ == "__main__":
    main()
