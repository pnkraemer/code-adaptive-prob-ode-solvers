import argparse
import functools
import os
import statistics
import timeit
import warnings
from typing import Callable

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import adaptive, controls, ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated, markov
from probdiffeq.solvers.strategies import fixedpoint
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from probdiffeq.util.doc_util import info

# todo: move the different solvers to the src
# todo: simplify this script a bit
# todo: add a solver that does save_every_step + offgrid_marginals


def set_jax_config() -> None:
    """Set JAX and other external libraries up."""
    # x64 precision
    jax.config.update("jax_enable_x64", True)


def print_library_info() -> None:
    """Print the environment info for this benchmark."""
    info.print_info()
    print("\n------------------------------------------\n")


def parse_arguments() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def tolerances_from_args(arguments: argparse.Namespace, /) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(arguments.start, arguments.stop, step=1.0)


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        _ = fun()
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def solver_probdiffeq(
    num_derivatives: int, implementation, correction, save_at
) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @jax.jit
    def vf_probdiffeq(y, /, *, t, p1=-2.0, p2=1.25, p3=-0.5):
        r"""Rigid body dynamics without external forces."""
        return jnp.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])

    u0 = jnp.asarray((1.0, 0.0, 0.9))
    t0, _t1 = (0.0, 50.0)

    @jax.jit
    def param_to_solution(tol):
        impl.select(implementation, ode_shape=(3,))

        # Build a solver
        ibm = priors.ibm_adaptive(num_derivatives=num_derivatives)
        strategy = fixedpoint.fixedpoint_adaptive(ibm, correction())
        solver = calibrated.dynamic(strategy)
        control = controls.proportional_integral()
        adaptive_solver = adaptive.adaptive(
            solver, atol=1e-2 * tol, rtol=tol, control=control
        )

        # Initial state
        vf_auto = functools.partial(vf_probdiffeq, t=t0)
        tcoeffs = autodiff.taylor_mode_scan(vf_auto, (u0,), num=num_derivatives)
        output_scale = 1.0 * jnp.ones((2,)) if implementation == "blockdiag" else 1.0
        init = solver.initial_condition(tcoeffs, output_scale=output_scale)

        # Solve
        dt0 = timestep.initial(vf_auto, (u0,))
        solution = ivpsolve.solve_and_save_at(
            vf_probdiffeq,
            init,
            save_at=save_at,
            dt0=dt0,
            adaptive_solver=adaptive_solver,
        )

        markov_seq_posterior = markov.select_terminal(solution.posterior)
        margs_posterior = markov.marginals(markov_seq_posterior, reverse=True)

        mean = jnp.concatenate(
            [margs_posterior.mean, solution.posterior.init.mean[[-1], ...]]
        )
        return jax.vmap(impl.hidden_model.qoi_from_sample)(mean)

    return lambda t: param_to_solution(t).block_until_ready()


def solver_diffrax(*, solver, save_at) -> Callable:
    """Construct a solver that wraps Diffrax' solution routines."""

    @diffrax.ODETerm
    @jax.jit
    def vf_diffrax(_t, y, _args, p1=-2.0, p2=1.25, p3=-0.5):
        return jnp.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])

    u0 = jnp.asarray((1.0, 0.0, 0.9))
    t0, t1 = (0.0, 50.0)

    @jax.jit
    def param_to_solution(tol):
        tol_ = 1e-2 * tol
        controller = diffrax.PIDController(atol=1e-3 * tol_, rtol=tol_)
        saveat = diffrax.SaveAt(t0=False, t1=False, ts=save_at)
        solution = diffrax.diffeqsolve(
            vf_diffrax,
            y0=u0,
            t0=t0,
            t1=t1,
            saveat=saveat,
            stepsize_controller=controller,
            dt0=None,
            max_steps=None,
            solver=solver,
        )
        return solution.ys

    return lambda t: param_to_solution(t).block_until_ready()


def solver_scipy(*, method: str, save_at) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, y, *, p1=-2.0, p2=1.25, p3=-0.5):
        r"""Rigid body dynamics without external forces."""
        return np.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])

    u0 = jnp.asarray((1.0, 0.0, 0.9))
    time_span = np.asarray([0.0, 50.0])

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            vf_scipy,
            y0=u0,
            t_span=time_span,
            t_eval=save_at,
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return solution.y.T

    return param_to_solution


def plot_ivp_solution():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, y, *, p1=-2.0, p2=1.25, p3=-0.5):
        r"""Rigid body dynamics without external forces."""
        return np.asarray([p1 * y[1] * y[2], p2 * y[0] * y[2], p3 * y[0] * y[1]])

    u0 = jnp.asarray((1.0, 0.0, 0.9))
    time_span = np.asarray([0.0, 50.0])

    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def rmse_relative(expected: jax.Array, *, nugget=1e-5) -> Callable:
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        error_relative = error_absolute / jnp.abs(nugget + expected)
        return jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

    return rmse


def workprec(fun, *, precision_fun: Callable, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function to a parameter-to-workprecision function.

    Turn a function param->solution into a function

    (param1, param2, ...)->(workprecision1, workprecision2, ...)

    where workprecisionX is a dictionary with keys "work" and "precision".
    """

    def parameter_list_to_workprecision(list_of_args, /):
        works_min = []
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg))
            times = timeit_fun(lambda: fun(arg))  # noqa: B023

            precisions.append(precision)
            works_min.append(min(times))
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "length_of_longest_vector": jnp.ones_like(jnp.asarray(precisions)),
            "work_min": jnp.asarray(works_min),
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


if __name__ == "__main__":
    # Set up all the configs
    set_jax_config()
    print_library_info()

    # Simulate once to get plotting code
    ts, ys = plot_ivp_solution()

    # If we change the probdiffeq-impl halfway through a script, a warning is raised.
    # But for this benchmark, such a change is on purpose.
    warnings.filterwarnings("ignore")

    # Read configuration from command line
    args = parse_arguments()
    tolerances = tolerances_from_args(args)
    timeit_fun = timeit_fun_from_args(args)

    # Save-at:
    xs = jnp.linspace(jnp.amin(ts), jnp.amax(ts), num=5)

    # Assemble algorithms
    ts0, ts1 = corrections.ts0, corrections.ts1
    ts0_2 = solver_probdiffeq(2, correction=ts0, implementation="isotropic", save_at=xs)
    ts0_4 = solver_probdiffeq(4, correction=ts0, implementation="isotropic", save_at=xs)
    algorithms = {
        # r"ProbDiffEq: TS0($1$)": ts0_1,
        r"TS0($2$)": ts0_2,
        r"TS0($4$)": ts0_4,
        "Bosh3()": solver_diffrax(solver=diffrax.Bosh3(), save_at=xs),
        "Dopri5()": solver_diffrax(solver=diffrax.Dopri5(), save_at=xs),
    }

    # Compute a reference solution
    reference = solver_scipy(method="LSODA", save_at=xs)(1e-15)
    precision_fun = rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        jnp.save(os.path.dirname(__file__) + "/plot_ts.npy", ts)
        jnp.save(os.path.dirname(__file__) + "/plot_ys.npy", ys)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")
