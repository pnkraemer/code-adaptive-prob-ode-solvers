import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
from probdiffeq import ivpsolve, ivpsolvers, taylor, stats
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


class Runner:
    def __init__(self, vf, init, tspan, /, *, ode_order: int, num_derivs: int, which):
        self.vf = vf

        ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
        ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
        if which == "filter":
            strategy = ivpsolvers.strategy_filter(ibm, ts0)
        elif which == "fixedpoint":
            strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
        else:
            raise ValueError
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
        cts = []
        for _ in range(3):
            t0 = time.perf_counter()
            sol = self.solve()
            sol.grid.block_until_ready()
            sol.solution.block_until_ready()
            t1 = time.perf_counter()
            cts.append(t1 - t0)
        return min(cts)


class RunnerTextbook:
    def __init__(self, vf, init, tspan, /, *, ode_order: int, num_derivs: int):
        self.vf = vf

        ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
        ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
        strategy = ivpsolvers.strategy_smoother(ibm, ts0)
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
        small_value = 1e-10
        t0 = save_at[0] - small_value
        t1 = save_at[-1] + small_value
        adaptive = self._solve_adaptive(tol=tol, t0=t0, t1=t1)
        print(len(adaptive.grid))
        solve = functools.partial(self._solve, grid=adaptive.grid, save_at=save_at)
        self.solve = jax.jit(solve)
        return self.solve()

    def _solve_adaptive(self, *, tol, t0, t1):
        asolver = ivpsolve.adaptive(self.solver, atol=tol, rtol=tol, control=self.ctrl)
        solution = ivpsolve.solve_adaptive_save_every_step(
            self.vf, self.init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=asolver
        )
        return IVPSolution(grid=solution.t, solution=solution.u)

    def _solve(self, grid, save_at):
        sol = ivpsolve.solve_fixed_grid(
            self.vf, self.init, grid=grid, solver=self.solver
        )
        dense, _ = stats.offgrid_marginals_searchsorted(
            ts=save_at, solution=sol, solver=self.solver
        )
        return IVPSolution(grid=save_at, solution=dense)

    def runtime(self):
        cts = []
        for _ in range(3):
            t0 = time.perf_counter()
            sol = self.solve()
            sol.grid.block_until_ready()
            sol.solution.block_until_ready()
            t1 = time.perf_counter()
            cts.append(t1 - t0)
        return min(cts)


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # Set up the IVP
    ivp = ivps.three_body_restricted()

    # Set up the solver
    impl.select("blockdiag", ode_shape=(2,))
    baseline = solve_baseline(*ivp, tol=1e-3, ode_order=2, num_derivs=5)
    # plt.plot(*baseline.solution.T)
    # plt.show()

    checkpoint_fixpt = Runner(*ivp, ode_order=2, num_derivs=5, which="fixedpoint")
    textbook = RunnerTextbook(*ivp, ode_order=2, num_derivs=5)

    save_at = jnp.linspace(jnp.amin(baseline.grid), jnp.amax(baseline.grid), num=50)
    reference = checkpoint_fixpt.prepare(tol=1e-11, save_at=save_at)

    tols = 10.0 ** (-jnp.arange(1, 11, step=2))
    for alg in [textbook, checkpoint_fixpt]:
        for tol in tols:
            approximation = alg.prepare(tol=tol, save_at=save_at)
            runtime = alg.runtime()
            accuracy = error(approximation.solution, reference.solution)
            print(f"tol={tol:.0e}, time={runtime:.3f}s, acc={accuracy:.2e}")
        print()


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


def error(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(b.size)


if __name__ == "__main__":
    main()
