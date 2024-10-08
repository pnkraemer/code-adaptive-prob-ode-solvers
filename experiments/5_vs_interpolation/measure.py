import functools
import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.impl import impl

from odecheckpts import ivps
import os


class IVPSolution(NamedTuple):
    grid: jax.Array
    solution: jax.Array

    @property
    def steps(self):
        return jnp.diff(self.grid)

    @property
    def num_steps(self):
        return len(self.steps)


class RunnerCheckpoint:
    name = "ATS (ours)"

    def __init__(
        self,
        vf,
        init,
        tspan,
        /,
        *,
        ode_order: int,
        num_derivs: int,
        num_samples: int,
    ):
        self.vf = vf
        self.num_samples = num_samples

        ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
        ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
        strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
        self.solver = ivpsolvers.solver(strategy)
        self.ctrl = ivpsolve.control_proportional_integral()

        # Set up the initial condition
        t0, t1 = tspan
        num = num_derivs + 1 - ode_order
        tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
        output_scale = jnp.ones((), dtype=float)
        self.init = self.solver.initial_condition(tcoeffs, output_scale)

        self.solve = None

    def prepare_and_solve(self, *, tol, save_at):
        solve = functools.partial(self._solve, tol=tol, save_at=save_at)
        self.solve = jax.jit(solve)
        return self.solve()

    def _solve(self, *, tol, save_at):
        asolver = ivpsolve.adaptive(self.solver, atol=tol, rtol=tol, control=self.ctrl)
        solution = ivpsolve.solve_adaptive_save_at(
            self.vf, self.init, save_at=save_at, dt0=0.01, adaptive_solver=asolver
        )
        # Sample from the posterior
        key = jax.random.PRNGKey(1)
        posterior = stats.markov_select_terminal(solution.posterior)
        (qoi, samples), (init, _) = stats.markov_sample(
            key, posterior, shape=(self.num_samples,), reverse=True
        )
        qoi = jnp.concatenate([qoi, init[..., None, :]], axis=-2)
        return IVPSolution(grid=save_at, solution=qoi.mean(axis=0))


class RunnerTextbook:
    name = "AS"

    def __init__(
        self,
        vf,
        init,
        tspan,
        /,
        *,
        ode_order: int,
        num_derivs: int,
        num_samples: int,
    ):
        self.vf = vf
        self.num_samples = num_samples

        ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
        ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
        strategy = ivpsolvers.strategy_smoother(ibm, ts0)
        self.solver = ivpsolvers.solver(strategy)
        self.ctrl = ivpsolve.control_proportional_integral()

        # Set up the initial condition
        t0, t1 = tspan
        num = num_derivs + 1 - ode_order
        tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
        output_scale = jnp.ones((), dtype=float)
        self.init = self.solver.initial_condition(tcoeffs, output_scale)

        self.solve = None

    def prepare_and_solve(self, *, tol, save_at):
        t0 = save_at[0]
        t1 = save_at[-1]
        adaptive = self._solve_adaptive(tol=tol, t0=t0, t1=t1)

        # Add the save_at points the adaptive solution
        #  to emulate a "tstops" argument
        grid = jnp.union1d(adaptive.grid, save_at)
        grid = jnp.sort(grid)

        solve = functools.partial(self._solve, grid=grid, save_at=save_at)
        self.solve = jax.jit(solve)
        return self.solve()

    def _solve_adaptive(self, *, tol, t0, t1):
        asolver = ivpsolve.adaptive(self.solver, atol=tol, rtol=tol, control=self.ctrl)
        solution = ivpsolve.solve_adaptive_save_every_step(
            self.vf, self.init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=asolver
        )
        return IVPSolution(grid=solution.t, solution=solution.u)

    def _solve(self, grid, save_at):
        solution = ivpsolve.solve_fixed_grid(
            self.vf, self.init, grid=grid, solver=self.solver
        )
        # Sample from the posterior
        key = jax.random.PRNGKey(1)
        posterior = stats.markov_select_terminal(solution.posterior)
        (qoi, samples), (init, _) = stats.markov_sample(
            key, posterior, shape=(self.num_samples,), reverse=True
        )
        qoi = jnp.concatenate([qoi, init[..., None, :]], axis=-2)

        # Find indices of save_at in grid
        _, _, indices = jnp.intersect1d(
            save_at, grid, size=len(save_at), return_indices=True
        )
        return IVPSolution(grid=save_at, solution=qoi[:, indices, :].mean(axis=0))


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    # Set up the IVP
    # ivp = ivps.van_der_pol(mu=1000)
    ivp = ivps.three_body_restricted()

    # Set up the solver
    impl.select("isotropic", ode_shape=(2,))
    baseline = solve_baseline(*ivp, tol=1e-7, ode_order=2, num_derivs=3)

    results = {}
    num_samples = [5, 50, 500]
    i = 1
    for n in num_samples:
        tols = [10.0 ** (-4.0), 10.0 ** (-7.0), 10.0 ** (-10.0)]
        for tol in tols:
            checkpoint = RunnerCheckpoint(
                *ivp, ode_order=2, num_derivs=4, num_samples=n
            )
            textbook = RunnerTextbook(*ivp, ode_order=2, num_derivs=4, num_samples=n)

            results[i] = {}
            results[i]["K"] = n
            results[i]["tol."] = tol

            for alg in [textbook, checkpoint]:
                save_at = jnp.linspace(ivp[2][0], ivp[2][-1])
                reference = checkpoint.prepare_and_solve(tol=1e-12, save_at=save_at)

                approximation = alg.prepare_and_solve(tol=tol, save_at=save_at)
                tm = runtime(alg.solve, num_runs=3)

                accuracy = error(approximation.solution, reference.solution)

                results[i][f"{alg.name} (time)"] = tm
                results[i][f"{alg.name} (accuracy)"] = accuracy

                print(
                    f"alg={alg.name}, K={n}, tol={tol:.0e}, time={tm:.3f}s, acc={accuracy:.2e}"
                )
            i += 1

        print()

    filename = os.path.dirname(__file__) + "/data"
    jnp.save(f"{filename}_results.npy", results, allow_pickle=True)
    jnp.save(f"{filename}_solution.npy", baseline.solution, allow_pickle=True)
    print(f"Saved to {filename}")


def solve_baseline(vf, init, tspan, /, *, tol: float, ode_order: int, num_derivs: int):
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
    ts0 = ivpsolvers.correction_ts0(ode_order=ode_order)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver(strategy)

    # Set up the initial condition
    t0, t1 = tspan
    num = num_derivs + 1 - ode_order
    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num)
    output_scale = jnp.ones((), dtype=float)
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
    )
    return IVPSolution(grid=solution.t, solution=solution.u)


def runtime(function: Callable, num_runs: int):
    cts = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sol = function()
        sol.grid.block_until_ready()
        sol.solution.block_until_ready()
        t1 = time.perf_counter()
        cts.append(t1 - t0)
    return min(cts)


def error(a, b):
    return jnp.linalg.norm(a - b) / jnp.sqrt(b.size)


if __name__ == "__main__":
    main()
