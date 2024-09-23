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

    ref = solve_adaptive(vf, init, (t0, t1), solver, tol=1e-5)

    # Pre-compile and compute error
    asol = solve_adaptive(vf, init, (t0, t1), solver, tol=1e-3)
    aerror = jnp.linalg.norm(ref.u[-1] - asol.u[-1])

    # Execute and benchmark
    time_ = time.perf_counter()
    asol = solve_adaptive(vf, init, (t0, t1), solver, tol=1e-3)
    asol.u.block_until_ready()
    atime = time.perf_counter() - time_

    # Pre-compile
    dtmin = jnp.amin(jnp.diff(asol.t)) * 2
    fsol = solve_fixed(vf, init, (t0, t1), solver, dtmin)
    ferror = jnp.linalg.norm(fsol.u[-1] - ref.u[-1])

    # Execute and benchmark
    time_ = time.perf_counter()
    fsol = solve_fixed(vf, init, (t0, t1), solver, dtmin)
    fsol.u.block_until_ready()
    ftime = time.perf_counter() - time_

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.semilogy(
        asol.t[:-1],
        jnp.diff(asol.t),
        linestyle="-",
        marker=".",
        color="C0",
        label=f"{atime:.1f} s, {aerror:.0e} e",
    )
    ax.semilogy(
        asol.t[:-1],
        dtmin * jnp.ones_like(asol.t[:-1]),
        linestyle="-",
        marker=".",
        color="C1",
        label=f"{ftime:.1f} s, {ferror:.0e} e",
    )
    ax.legend()

    ax.annotate(f"Runtime: {atime:.1f} s", (asol.t[0], jnp.diff(asol.t)[0]), color="C0")
    ax.annotate(f"Error: {aerror:.0e}", (asol.t[-1], jnp.diff(asol.t)[-1]), color="C0")
    ax.set_xlabel(r"Time $t$ ($\rightarrow$ input of ODE)")
    ax.set_ylabel(r"Step-size $\Delta t$")
    # axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])
    # axin1.plot(asol.t, asol.u)
    plt.show()


def problem_van_der_pol():
    def vf(y, ydot, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 10**2 * (ydot * (1 - y**2) - y)

    u0 = jnp.asarray([2.0])
    du0 = jnp.asarray([0.0])
    t0, t1 = 0.0, 6.1
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
    return solution


def solve_fixed(vf, init, time_span, solver, dt):
    t0, t1 = time_span
    num_steps = ((t1 - t0) / dt).astype(int)
    grid = jnp.linspace(t0, t1, num=num_steps, endpoint=True)
    return ivpsolve.solve_fixed_grid(vf, init, grid=grid, solver=solver)


if __name__ == "__main__":
    main()
