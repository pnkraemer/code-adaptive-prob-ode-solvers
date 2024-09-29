import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl

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


class RunnerCheckpt:
    def __init__(self, vf, init, tspan, /, *, ode_order: int, num_derivs: int):
        self.vf = vf

        ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
        ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
        strategy = ivpsolvers.strategy_filter(ibm, ts0)
        self.solver = ivpsolvers.solver_dynamic(strategy)
        self.ctrl = ivpsolve.control_proportional_integral()

        # Set up the initial condition
        t0, t1 = tspan
        num = num_derivs + 1 - ode_order
        tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
        output_scale = jnp.ones((2,), dtype=float)
        self.init = self.solver.initial_condition(tcoeffs, output_scale)

        self.solve = None

    def prepare(self, *, tol, save_at):
        solve = functools.partial(self._solve, tol=tol, save_at=save_at)
        self.solve = jax.jit(solve)
        return self.solve()

    def _solve(self, *, tol, save_at):
        asolver = ivpsolve.adaptive(self.solver, atol=tol, rtol=tol, control=self.ctrl)
        solution = ivpsolve.solve_adaptive_save_at(
            self.vf, self.init, save_at=save_at, dt0=0.01, adaptive_solver=asolver
        )
        return IVPSolution(grid=solution.t, solution=solution.u)

    def runtime(self):
        t0 = time.perf_counter()
        sol = self.solve()
        sol.grid.block_until_ready()
        sol.solution.block_until_ready()
        t1 = time.perf_counter()
        return t1 - t0


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # Set up the IVP
    ivp = ivps.three_body_restricted()

    # Set up the solver
    impl.select("blockdiag", ode_shape=(2,))
    baseline = solve_baseline(*ivp, tol=1e-3, ode_order=2, num_derivs=3)
    # plt.plot(*baseline.solution.T)
    # plt.show()

    save_at = jnp.linspace(jnp.amin(baseline.grid), jnp.amax(baseline.grid))
    checkpoint = RunnerCheckpt(*ivp, ode_order=2, num_derivs=3)
    tols = 10.0 ** (-jnp.arange(3, 8, step=1))
    for tol in tols:
        _solution = checkpoint.prepare(tol=tol, save_at=save_at)
        runtime = checkpoint.runtime()
        print(tol, runtime)


def solve_baseline(vf, init, tspan, /, *, tol: float, ode_order: int, num_derivs: int):
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
    ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

    # Set up the initial condition
    t0, t1 = tspan
    num = num_derivs + 1 - ode_order
    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
    output_scale = jnp.ones((2,), dtype=float)
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
    )
    return IVPSolution(grid=solution.t, solution=solution.u)


if __name__ == "__main__":
    main()
