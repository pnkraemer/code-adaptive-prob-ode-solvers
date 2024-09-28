import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl


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

    def vf(y, ydot, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 10**3 * (ydot * (1 - y**2) - y)

    u0 = jnp.asarray([2.0])
    du0 = jnp.asarray([0.0])
    t0, t1 = 0.0, 6.3

    # Set up the solver
    impl.select("dense", ode_shape=(1,))
    num = 4
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

    # Set up the initial condition
    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), [u0, du0], num=num - 1)
    output_scale = 1.0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    tol = 1e-3
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
    )
    baseline = IVPSolution(grid=solution.t, solution=solution.u)

    # Determine how many steps the accurate fixed-step solver should take
    min_step = jnp.amin(baseline.steps)
    required_steps = ((t1 - t0) / min_step).astype(int)

    # Benchmark the solvers

    @jit_and_time
    def solve_adaptive():
        sl = ivpsolve.solve_fixed_grid(vf, init, grid=baseline.grid, solver=solver)
        return IVPSolution(sl.t, sl.u)

    @jit_and_time
    def solve_fixed_inaccurate():
        grid = jnp.linspace(t0, t1, num=len(baseline.grid), endpoint=True)
        sl = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)
        return IVPSolution(sl.t, sl.u)

    @jit_and_time
    def solve_fixed_accurate():
        grid = jnp.linspace(t0, t1, num=required_steps, endpoint=True)
        sl = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)
        return IVPSolution(sl.t, sl.u)

    # Assert that the inaccurate solution is actually bad

    fixed_inaccurate = solve_fixed_inaccurate()
    assert jnp.any(jnp.isnan(fixed_inaccurate.output.solution))

    # Run and time the accurate solutions

    adaptive = solve_adaptive()
    fixed_accurate = solve_fixed_accurate()

    # Save all important arrays to files
    filename = str(__file__)
    name = filename.replace(".py", "_baseline_grid.npy")
    jnp.save(name, baseline.grid)
    name = filename.replace(".py", "_baseline_solution.npy")
    jnp.save(name, baseline.solution)
    name = filename.replace(".py", "_grid_adaptive.npy")
    jnp.save(name, adaptive.output.grid)
    name = filename.replace(".py", "_grid_fixed_accurate.npy")
    jnp.save(name, fixed_accurate.output.grid)
    name = filename.replace(".py", "_grid_fixed_inaccurate.npy")
    jnp.save(name, fixed_inaccurate.output.grid)
    name = filename.replace(".py", "_runtime_adaptive.npy")
    jnp.save(name, adaptive.runtime)
    name = filename.replace(".py", "_runtime_fixed_accurate.npy")
    jnp.save(name, fixed_accurate.runtime)
    name = filename.replace(".py", "_runtime_fixed_inaccurate.npy")
    jnp.save(name, fixed_inaccurate.runtime)


if __name__ == "__main__":
    main()
