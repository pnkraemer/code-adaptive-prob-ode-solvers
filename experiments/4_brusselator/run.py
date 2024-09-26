import matplotlib.pyplot as plt
import warnings
import jax
import jax.flatten_util

from odecheckpts import ivps
from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl

import jax.numpy as jnp
from odecheckpts import exp_util

import time

import os


def main():
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)
    plt.rcParams.update(exp_util.plot_params())

    results_checkpoint = {
        "N": [],
        "runtime": [],
        "memory": [],
        "ts": [],
        "ys": [],
        "num_steps": [],
    }
    results_textbook = {
        "N": [],
        "runtime": [],
        "memory": [],
        "ts": [],
        "ys": [],
        "num_steps": [],
    }

    Nranges = [10, 30, 50, 70]  # with N>=50, the pcolormesh will even look decent.
    # Nranges = [2, 4, 6, 8, 10]
    for N in Nranges:
        # Set up the problem
        vf, u0, (t0, t1), params = ivps.brusselator(N=N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            impl.select("dense", ode_shape=(2 * N,))

        # Set up the solver
        num = 4
        tol = 1e-8
        ctrl = ivpsolve.control_proportional_integral()
        ibm = ivpsolvers.prior_ibm(num_derivatives=num)
        ts0 = ivpsolvers.correction_ts1(ode_order=1)
        strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
        solver = ivpsolvers.solver_dynamic(strategy)
        adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)

        # Set up the initial condition. Use it to estimate the memory demands.
        tcoeffs = taylor.odejet_unroll(lambda *y: vf(*y, t=t0, p=params), u0, num=num)
        output_scale = 1.0  # or any other value with the same shape
        init = solver.initial_condition(tcoeffs, output_scale)

        # 8 GB of memory / size of initial condition.
        # Note: adaptive steps carry around three copies
        # (step_from, interpolate_from, and the current state)
        print(f"\nFor N={N}:")
        num_gb = 8
        num_copies = 3
        size_init = num_copies * jax.flatten_util.ravel_pytree(init)[0].nbytes
        ncopies = num_gb * 1024**3 / size_init
        msg = f"\t{int(ncopies):,} copies of the initial condition fit in memory."
        print(msg)

        # Use simulate_terminal_values to see how many steps will be taken
        # without crashing the machine.
        jax.clear_caches()
        solve = jax.jit(
            ivpsolve.solve_adaptive_terminal_values,
            static_argnums=0,
            static_argnames=["adaptive_solver"],
        )
        count0 = time.perf_counter()
        solution = solve(
            vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
        )
        solution.u.block_until_ready()
        total_memory = solution.num_steps * size_init / 1024**2
        count1 = time.perf_counter() - count0
        msg = f"\tBaseline: {int(solution.num_steps):,} steps ({int(total_memory):,} MB) in {count1:.1f}s"
        print(msg)

        # not 8000 because other processes must run as well (but this limit isn't tight)...
        if total_memory < 7000:
            strategy_ = ivpsolvers.strategy_smoother(ibm, ts0)
            solver_ = ivpsolvers.solver_dynamic(strategy_)
            adaptive_solver_ = ivpsolve.adaptive(
                solver_, atol=tol, rtol=tol, control=ctrl
            )
            jax.clear_caches()
            count0 = time.perf_counter()
            solution = ivpsolve.solve_adaptive_save_every_step(
                vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver_
            )
            solution.u.block_until_ready()
            count1 = time.perf_counter() - count0
            print(f"\tTextbook solver: {count1:.1f}s")
            results_textbook["N"].append(N)
            results_textbook["runtime"].append(count1)
            results_textbook["memory"].append(total_memory)
            results_textbook["ts"].append(solution.t)
            results_textbook["ys"].append(solution.u)
            results_textbook["num_steps"].append(jnp.amax(solution.num_steps))

        save_at = jnp.linspace(t0, t1, num=100)
        count0 = time.perf_counter()
        jax.clear_caches()
        solve = jax.jit(
            ivpsolve.solve_adaptive_save_at,
            static_argnums=0,
            static_argnames=["adaptive_solver"],
        )
        solution = solve(
            vf, init, save_at=save_at, dt0=0.01, adaptive_solver=adaptive_solver
        )
        solution.u.block_until_ready()
        count1 = time.perf_counter() - count0
        print(f"\tCheckpoint solver: {count1:.1f}s")
        results_checkpoint["N"].append(N)
        results_checkpoint["runtime"].append(count1)
        results_checkpoint["memory"].append(len(save_at) * size_init / 1024**2)
        results_checkpoint["ts"].append(solution.t)
        results_checkpoint["ys"].append(solution.u)
        results_checkpoint["num_steps"].append(jnp.amax(solution.num_steps))
    print()

    jnp.save(
        os.path.dirname(__file__) + "/data_checkpoint.npy",
        results_checkpoint,
        allow_pickle=True,
    )
    jnp.save(
        os.path.dirname(__file__) + "/data_textbook.npy",
        results_textbook,
        allow_pickle=True,
    )


if __name__ == "__main__":
    main()
