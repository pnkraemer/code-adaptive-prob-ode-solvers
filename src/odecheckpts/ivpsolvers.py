"""Solution routines for initial value problems."""

import functools
import warnings

import diffrax
import jax
import jax.numpy as jnp
import scipy.integrate
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.impl import impl


def solve(
    method: str,
    vf,
    u0_like,
    /,
    save_at,
    *,
    dt0,
    atol,
    rtol,
    ode_order=1,
    calibrate="dynamic",
):
    # Select a state-space model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        implementation = "isotropic"
        impl.select(implementation, ode_shape=u0_like.shape)

    num_derivatives = int(method[-1])
    if method[:3] == "ts0":
        correction = ivpsolvers.correction_ts0(ode_order=ode_order)
    else:
        raise ValueError

    # Build a solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivatives)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, correction)

    if calibrate == "dynamic":
        solver = ivpsolvers.solver_dynamic(strategy)
    elif calibrate == "none":
        solver = ivpsolvers.solver(strategy)
    else:
        raise ValueError

    control = ivpsolve.control_proportional_integral()
    asolver = ivpsolve.adaptive(solver, atol=atol, rtol=rtol, control=control)

    def solve_(u0: tuple, p, output_scale=1.0):
        if not isinstance(u0, tuple):
            raise ValueError("Tuple expected.")

        def vf_wrapped(*y, t):
            return vf(*y, t=t, p=p)

        # Initial state
        t0 = save_at[0]
        vf_auto = functools.partial(vf_wrapped, t=t0)
        tcoeffs = taylor.odejet_padded_scan(
            vf_auto, u0, num=num_derivatives + 1 - ode_order
        )
        init = solver.initial_condition(tcoeffs, output_scale=output_scale)

        # Solve
        sol = ivpsolve.solve_adaptive_save_at(
            vf_wrapped,
            init,
            save_at=save_at,
            dt0=dt0,
            adaptive_solver=asolver,
        )

        # Marginalise
        markov_seq_posterior = stats.markov_select_terminal(sol.posterior)
        margs_posterior = stats.markov_marginals(markov_seq_posterior, reverse=True)

        # Stack the initial state into the solution
        mean = jnp.concatenate(
            [margs_posterior.mean, sol.posterior.init.mean[[-1], ...]]
        )
        # Select the QOI
        aux = {"solution": sol, "u0_solve": sol.u}
        return jax.vmap(impl.hidden_model.qoi_from_sample)(mean), aux

    return solve_


def solve_via_interpolate(method: str, vf, u0_like, /, save_at, *, dt0, atol, rtol):
    small_value = 1e-6
    # Select a state-space model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        implementation = "isotropic"
        impl.select(implementation, ode_shape=u0_like.shape)

    num_derivatives = int(method[-1])
    if method[:3] == "ts0":
        correction = ivpsolvers.correction_ts0()
    else:
        raise ValueError

    # Build a solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivatives)
    strategy = ivpsolvers.strategy_smoother(ibm, correction)
    solver = ivpsolvers.solver_dynamic(strategy)
    control = ivpsolve.control_proportional_integral()
    asolver = ivpsolve.adaptive(solver, atol=atol, rtol=rtol, control=control)

    offgrid_marginals = jax.jit(stats.offgrid_marginals_searchsorted)

    def solve_(u0: tuple, p, output_scale=1.0):
        if not isinstance(u0, tuple):
            raise ValueError("Tuple expected.")

        def vf_wrapped(*y, t):
            return vf(*y, t=t, p=p)

        # Initial state
        t0 = save_at[0]
        vf_auto = functools.partial(vf_wrapped, t=t0)
        tcoeffs = taylor.odejet_padded_scan(vf_auto, u0, num=num_derivatives)
        init = solver.initial_condition(tcoeffs, output_scale=output_scale)

        # Solve
        sol = ivpsolve.solve_adaptive_save_every_step(
            vf_wrapped,
            init,
            # Small perturbation so that all save_at values
            # are in the *interior* of the domain.
            t0=save_at[0] - small_value,
            t1=save_at[-1] + small_value,
            dt0=dt0,
            adaptive_solver=asolver,
        )

        dense, _ = offgrid_marginals(ts=save_at, solution=sol, solver=solver)

        return dense, {"solution": sol, "u0_solve": sol.u}

    return solve_


def solve_diffrax(
    method: str, vf, _u0_like, /, save_at, *, dt0, atol, rtol, ode_order=1
):
    if method == "tsit5":
        solver = diffrax.Tsit5()
    elif method == "bosh3":
        solver = diffrax.Bosh3()
    elif method == "dopri5":
        solver = diffrax.Dopri5()
    elif method == "dopri8":
        solver = diffrax.Dopri8()
    else:
        raise ValueError
    term = diffrax.ODETerm(lambda t, y, args: vf(y, t=t, p=args))
    controller = diffrax.PIDController(atol=atol, rtol=rtol)
    saveat = diffrax.SaveAt(t0=False, t1=False, ts=save_at)

    def solve_(u0: tuple, p):
        if not isinstance(u0, tuple):
            raise ValueError("Tuple expected.")
        (init,) = u0
        sol = diffrax.diffeqsolve(
            term,
            y0=init,
            args=p,
            t0=save_at[0],
            t1=save_at[-1],
            saveat=saveat,
            stepsize_controller=controller,
            dt0=dt0,
            max_steps=None,
            solver=solver,
        )
        if ode_order == 1:
            u = sol.ys
        elif ode_order == 2:
            d = len(sol.ys[0])
            u = sol.ys[:, : d // 2]
        else:
            raise ValueError
        return u, {"solution": sol, "u0_solve": sol.ys}

    return solve_


def asolve_scipy(method: str, vf, /, time_span, *, atol, rtol):
    def solve_(u0: tuple, p):
        if not isinstance(u0, tuple):
            raise ValueError("Tuple expected.")

        def vf_scipy(t, y):
            return vf(y, t=t, p=p)

        (y0,) = u0
        solution = scipy.integrate.solve_ivp(
            vf_scipy, y0=y0, t_span=time_span, atol=atol, rtol=rtol, method=method
        )
        return solution.t, solution.y.T

    return solve_
