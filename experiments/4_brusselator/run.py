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


def main():
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)
    plt.rcParams.update(exp_util.plot_params())

    # Simulate once to get plotting code
    # for N in [5, 10, 20, 50, 100, 200]:
    #     vf, u0, (t0, t1), params = ivps.brusselator(N=N)
    #
    #     # Set up the solver
    #     impl.select("dense", ode_shape=(2 * N,))
    #     num = 4
    #     ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    #     ts0 = ivpsolvers.correction_ts1(ode_order=1)
    #     strategy = ivpsolvers.strategy_filter(ibm, ts0)
    #     solver = ivpsolvers.solver_dynamic(strategy)
    #
    #     # Set up the initial condition
    #     tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0, p=params), u0, num=num)
    #     output_scale = 1.0  # or any other value with the same shape
    #     init = solver.initial_condition(tcoeffs, output_scale)
    #
    #     print(N, 8 * 1024**3 / jax.flatten_util.ravel_pytree(init)[0].nbytes, "copies fit in memory")

    for N in [2, 4, 8, 16, 32, 64]:
        vf, u0, (t0, t1), params = ivps.brusselator(N=N)

        # Set up the solver
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            impl.select("dense", ode_shape=(2 * N,))

        num = 4
        tol = 1e-8
        ctrl = ivpsolve.control_proportional_integral()
        ibm = ivpsolvers.prior_ibm(num_derivatives=num)
        ts0 = ivpsolvers.correction_ts1(ode_order=1)
        strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
        solver = ivpsolvers.solver_dynamic(strategy)
        adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)

        # Set up the initial condition
        tcoeffs = taylor.odejet_unroll(lambda *y: vf(*y, t=t0, p=params), u0, num=num)
        output_scale = 1.0  # or any other value with the same shape
        init = solver.initial_condition(tcoeffs, output_scale)

        # 8 GB of memory / size of initial condition.
        # Note: adaptive steps carry around three copies
        print(f"\nFor N={N}:")
        size_init = 3 * jax.flatten_util.ravel_pytree(init)[0].nbytes
        ncopies = 8 * 1024**3 / size_init
        msg = f"\t{int(ncopies):,} copies of the initial condition fit in memory."
        print(msg)

        # Use simulate_terminal_values to see how many steps will be taken
        # without crashing the machine.
        count0 = time.perf_counter()
        solution = ivpsolve.solve_adaptive_terminal_values(
            vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
        )
        solution.u.block_until_ready()
        ncopies = solution.num_steps * size_init / 1024**2
        count1 = time.perf_counter() - count0
        msg = f"\t{int(solution.num_steps):,} steps ({int(ncopies):,} MB) were taken in {count1:.1f}s"
        print(msg)

        if ncopies < 4000:
            strategy_ = ivpsolvers.strategy_smoother(ibm, ts0)
            solver_ = ivpsolvers.solver_dynamic(strategy_)
            adaptive_solver_ = ivpsolve.adaptive(
                solver_, atol=tol, rtol=tol, control=ctrl
            )
            count0 = time.perf_counter()
            solution = ivpsolve.solve_adaptive_save_every_step(
                vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver_
            )
            solution.u.block_until_ready()
            count1 = time.perf_counter() - count0
            print(f"\tAdaptive solver takes {count1:.1f}s")

        save_at = jnp.linspace(t0, t1, num=100)
        count0 = time.perf_counter()
        solution = ivpsolve.solve_adaptive_save_at(
            vf, init, save_at=save_at, dt0=0.01, adaptive_solver=adaptive_solver
        )
        solution.u.block_until_ready()
        count1 = time.perf_counter() - count0
        print(f"\tCheckpointed solver takes {count1:.1f}s")
    print("\n")
    assert False

    # Compute a baseline solution
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.01, adaptive_solver=adaptive_solver
    )
    ts, ys = solution.t, solution.u

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(u0[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)

    fig, ax = plt.subplots(figsize=(3.25, 2.5), dpi=200)
    ax.contourf(Xs, Ts, Us)

    ax.set_xlabel("Space $x$")
    ax.set_ylabel("Time $t$ (steps are marked)")
    ax.set_title(f"Brusselator: $N={len(ts):,}$ adaptive steps")
    ax.set_yticks(ts[::20])
    ax.set_yticklabels(())
    ax.tick_params(which="both", direction="out")
    plt.show()

    #
    # fig = plt.figure(figsize=(3.25, 2.5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(Xs, Ts, Us)
    # ax.set_xlabel("Space $X$")
    # ax.set_ylabel("Time $t$")
    # ax.set_zlabel("Solution $u$")

    plt.show()


if __name__ == "__main__":
    main()
