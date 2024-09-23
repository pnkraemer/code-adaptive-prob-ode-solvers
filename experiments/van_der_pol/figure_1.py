"""Solve the logistic equation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import contextlib

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl
from tueplots import axes

plt.rcParams.update(axes.legend())


@contextlib.contextmanager
def time_function():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(1,))

    vf, (u0, du0), (t0, t1) = problem_van_der_pol()
    init, solver = solver_ts1(vf=vf, init=(u0, du0), t0=t0)

    # Compute the grid to-be-used for the adaptive solver
    grid_adaptive, sol_adaptive = solve_adaptive(
        vf=vf, init=init, time_span=(t0, t1), solver=solver, tol=1e-3
    )

    # Assemble the fixed-step reference grids
    num_steps = ((t1 - t0) / jnp.amin(jnp.diff(grid_adaptive))).astype(int)
    grid_fixed_accurate = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
    grid_fixed_inaccurate = jnp.linspace(t0, t1, num=len(grid_adaptive), endpoint=True)

    # Benchmark the solvers
    with time_function() as stopwatch:
        sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid_adaptive, solver=solver)
        sol.u.block_until_ready()

    time_adaptive = stopwatch()

    with time_function() as stopwatch:
        sol = ivpsolve.solve_fixed_grid(
            vf, init, grid=grid_fixed_accurate, solver=solver
        )
        sol.u.block_until_ready()

    time_fixed_accurate = stopwatch()

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.semilogy(
        grid_adaptive[:-1],
        jnp.diff(grid_adaptive),
        linestyle="solid",
        marker="None",
        markersize=1,
        color="C0",
        label=f"$N$={len(grid_adaptive):,} adaptive steps run in {time_adaptive:.1f} sec",
    )
    ax.semilogy(
        grid_fixed_inaccurate[:-1],
        jnp.diff(grid_fixed_inaccurate),
        linestyle="dotted",
        marker="None",
        color="gray",
        label=f"$N$={len(grid_fixed_inaccurate):,} fixed steps contain NaNs",
    )
    ax.semilogy(
        grid_fixed_accurate[:-1],
        jnp.diff(grid_fixed_accurate),
        linestyle="dashed",
        marker="None",
        color="C1",
        label=f"$N$={len(grid_fixed_accurate):,} fixed steps run in {time_fixed_accurate:.1f} sec",
    )
    ax.legend(loc="upper left", edgecolor="white", handlelength=1.5, fontsize="small")
    ax.set_xlabel(r"ODE domain (time $t$)")
    ax.set_ylabel(r"Step-size $\Delta t$")
    ax.set_ylim((2e-6, 5e1))
    ax.set_xlim((-0.1, 6.4))
    ax.set_xticks((0, 1, 2, 3, 4, 5, 6))

    axin1 = ax.inset_axes([0.8, 0.725, 0.175, 0.175])
    axin1.set_title("VdP solution", fontsize="small")
    axin1.set_xticks((0.0, 3.0, 6.0))
    axin1.set_yticks((-2, 2))
    axin1.plot(grid_adaptive, sol_adaptive, color="black", linewidth=0.75)
    axin1.set_xlim((0.0, 6.3))

    filename = str(__file__)
    filename = filename.replace("experiments/", "figures/")
    filename = filename.replace(".py", ".pdf")
    plt.savefig(filename)
    plt.show()


def problem_van_der_pol():
    def vf(y, ydot, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 10**3 * (ydot * (1 - y**2) - y)

    u0 = jnp.asarray([2.0])
    du0 = jnp.asarray([0.0])
    t0, t1 = 0.0, 6.3
    return vf, (u0, du0), (t0, t1)


def solver_ts1(*, vf, init, t0):
    num = 4
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num - 1)
    output_scale = 1.0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)
    return init, solver


def solve_adaptive(*, vf, init, time_span, solver, tol):
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)

    dt0 = 0.01
    t0, t1 = time_span
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
    )
    return solution.t, solution.u


if __name__ == "__main__":
    main()
