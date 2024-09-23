"""Solve the logistic equation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl


def main():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    impl.select("dense", ode_shape=(1,))

    vf, (u0, du0), (t0, t1) = problem_van_der_pol()
    init, solver = solver_ts1(vf, (u0, du0), t0)

    # Pre-compile and compute error
    _, asol_u = solve_adaptive(vf, init, (t0, t1), solver, tol=1e-3)

    # Execute and benchmark
    time_ = time.perf_counter()
    asol_t, asol_u = solve_adaptive(vf, init, (t0, t1), solver, tol=1e-3)
    asol_u.block_until_ready()
    asol_u.block_until_ready()
    atime = time.perf_counter() - time_

    # Pre-compile
    dtmin = jnp.amin(jnp.diff(asol_t))
    _ = solve_fixed(vf, init, (t0, t1), solver, dtmin)

    # Execute and benchmark
    time_ = time.perf_counter()
    fsol_t, fsol_u = solve_fixed(vf, init, (t0, t1), solver, dtmin)
    fsol_t.block_until_ready()
    fsol_u.block_until_ready()
    ftime = time.perf_counter() - time_

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.semilogy(
        asol_t[:-1],
        jnp.diff(asol_t),
        linestyle="-",
        markersize=0.25,
        color="C0",
    )
    ax.semilogy(
        fsol_t[:-1],
        jnp.diff(fsol_t),
        linestyle="-",
        markersize=0.25,
        color="C1",
    )
    ax.annotate(
        f"Runtime: {atime:.1f} s",
        xy=(1.5, 2e-2),
        xytext=(0.5, 2e-1),
        arrowprops=dict(arrowstyle="->", color="C0"),
        color="C0",
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    ax.annotate(
        f"{len(asol_t)} steps",
        xy=(2.75, 4e-2),
        xytext=(1.5, 1e1),
        arrowprops=dict(arrowstyle="->", color="C0"),
        color="C0",
        horizontalalignment="left",
        verticalalignment="bottom",
    )
    ax.annotate(
        f"Runtime: {ftime:.1f} s",
        xy=(1.25, 6e-6),
        xytext=(2, 1e-6),
        arrowprops=dict(arrowstyle="->", color="C1"),
        color="C1",
        horizontalalignment="right",
        verticalalignment="top",
    )
    ax.annotate(
        f"{len(fsol_t)} steps",
        xy=(3.15, 6e-6),
        xytext=(4, 1e-6),
        arrowprops=dict(arrowstyle="->", color="C1"),
        color="C1",
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.set_xlabel(r"ODE domain (time $t$)")
    ax.set_ylabel(r"Step-size $\Delta t$")
    ax.set_ylim((1e-7, 1e2))
    ax.set_xlim((-0.1, 6.4))
    axin1 = ax.inset_axes([0.8, 0.725, 0.175, 0.175])
    axin1.set_title("Van der Pol", fontsize="small")
    axin1.plot(asol_t, asol_u, color="black", linewidth=0.75)
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


def solver_ts1(vf, init, t0):
    num = 4
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_dynamic(strategy)

    tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), init, num=num - 1)
    output_scale = 1.0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)
    return init, solver


def solve_adaptive(vf, init, time_span, solver, tol):
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)

    dt0 = 0.01
    t0, t1 = time_span
    solution = ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
    )
    return solution.t, solution.u


def solve_fixed(vf, init, time_span, solver, dt):
    t0, t1 = time_span
    num_steps = ((t1 - t0) / dt).astype(int)

    grid = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
    sol = ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)
    return grid, sol.u


if __name__ == "__main__":
    main()
