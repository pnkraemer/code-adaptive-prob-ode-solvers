"""Solution routines for initial value problems."""

import functools
import warnings

import diffrax
import jax
import jax.numpy as jnp
import scipy.integrate

from probdiffeq import adaptive, controls, ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated, markov, solution
from probdiffeq.solvers.strategies import fixedpoint, smoothers
from probdiffeq.solvers.strategies.components import priors, corrections
from probdiffeq.taylor import autodiff


def solve(method: str, vf, u0_like, /, save_at, *, dt0, atol, rtol):
    # Select a state-space model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        implementation = "isotropic"
        impl.select(implementation, ode_shape=u0_like.shape)

    num_derivatives = int(method[-1])
    if method[:3] == "ts0":
        correction = corrections.ts0()
    else:
        raise ValueError

    # Build a solver
    ibm = priors.ibm_adaptive(num_derivatives=num_derivatives)
    strategy = fixedpoint.fixedpoint_adaptive(ibm, correction)
    solver = calibrated.dynamic(strategy)
    control = controls.proportional_integral()
    asolver = adaptive.adaptive(solver, atol=atol, rtol=rtol, control=control)

    def solve_(u0, p):
        def vf_wrapped(y, /, *, t):
            return vf(y, t, p)

        # Initial state
        t0 = save_at[0]
        vf_auto = functools.partial(vf_wrapped, t=t0)
        tcoeffs = autodiff.taylor_mode_scan(vf_auto, (u0,), num=num_derivatives)
        output_scale = 1.0 * jnp.ones((2,)) if implementation == "blockdiag" else 1.0
        init = solver.initial_condition(tcoeffs, output_scale=output_scale)

        # Solve
        sol = ivpsolve.solve_and_save_at(
            vf_wrapped,
            init,
            save_at=save_at,
            dt0=dt0,
            adaptive_solver=asolver,
        )

        # Marginalise
        markov_seq_posterior = markov.select_terminal(sol.posterior)
        margs_posterior = markov.marginals(markov_seq_posterior, reverse=True)

        # Stack the initial state into the solution
        mean = jnp.concatenate(
            [margs_posterior.mean, sol.posterior.init.mean[[-1], ...]]
        )
        # Select the QOI
        return jax.vmap(impl.hidden_model.qoi_from_sample)(mean), {
            "solution": sol,
            "u0_solve": sol.u,
        }

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
        correction = corrections.ts0()
    else:
        raise ValueError

    # Build a solver
    ibm = priors.ibm_adaptive(num_derivatives=num_derivatives)
    strategy = smoothers.smoother_adaptive(ibm, correction)
    solver = calibrated.dynamic(strategy)
    control = controls.proportional_integral()
    asolver = adaptive.adaptive(solver, atol=atol, rtol=rtol, control=control)

    offgrid_marginals = jax.jit(solution.offgrid_marginals_searchsorted)

    def solve_(u0, p):
        def vf_wrapped(y, /, *, t):
            return vf(y, t, p)

        # Initial state
        t0 = save_at[0]
        vf_auto = functools.partial(vf_wrapped, t=t0)
        tcoeffs = autodiff.taylor_mode_scan(vf_auto, (u0,), num=num_derivatives)
        output_scale = (
            1.0 * jnp.ones(u0_like.shape) if implementation == "blockdiag" else 1.0
        )
        init = solver.initial_condition(tcoeffs, output_scale=output_scale)

        # Solve
        sol = ivpsolve.solve_and_save_every_step(
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


def solve_diffrax(method: str, vf, _u0_like, /, save_at, *, dt0, atol, rtol):
    if method == "tsit5":
        solver = diffrax.Tsit5()
    elif method == "bosh3":
        solver = diffrax.Bosh3()
    elif method == "dopri8":
        solver = diffrax.Dopri8()
    else:
        raise ValueError
    term = diffrax.ODETerm(lambda t, y, args: vf(y, t, args))
    controller = diffrax.PIDController(atol=atol, rtol=rtol)
    saveat = diffrax.SaveAt(t0=False, t1=False, ts=save_at)

    def solve_(u0, p):
        sol = diffrax.diffeqsolve(
            term,
            y0=u0,
            args=p,
            t0=save_at[0],
            t1=save_at[-1],
            saveat=saveat,
            stepsize_controller=controller,
            dt0=dt0,
            max_steps=None,
            solver=solver,
        )
        return sol.ys, {"solution": sol, "u0_solve": sol.ys}

    return solve_


def asolve_scipy(method: str, vf, /, time_span, *, atol, rtol):
    def solve_(u0, p):
        def vf_scipy(t, y):
            return vf(y, t, p)

        solution = scipy.integrate.solve_ivp(
            vf_scipy, y0=u0, t_span=time_span, atol=atol, rtol=rtol, method=method
        )
        return solution.t, solution.y.T

    return solve_
