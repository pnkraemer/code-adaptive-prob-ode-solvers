import argparse
import os
import statistics
import timeit
from typing import Callable

import jax
import jax.numpy as jnp
import tqdm

from odecheckpts import ivps, ivpsolvers

# todo: to make offgrid_marginals() fair, split save_at from t0 and t1
#  (save_at must be in the interior)
# todo: give the interpolate() methods the first half of tolerances (to save time...)
# todo: run with ts0_2
# todo: return lengths of the vectors


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
    n0, n1 = arguments.start, arguments.stop
    n1_short = n0 + (1 + n1 - n0) * 2 // 3
    tols_ = 0.1 ** jnp.arange(n0, n1, step=1.0)
    tols_short_ = 0.1 ** jnp.arange(n0, n1_short, step=1.0)
    return tols_short_, tols_


def timeit_fun_from_args(arguments: argparse.Namespace, /) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        _ = fun()
        return list(timeit.repeat(fun, number=1, repeat=arguments.repeats))

    return timer


def rmse_absolute(expected: jax.Array) -> Callable:
    """Compute the relative RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        # error_relative = error_absolute / jnp.abs(nugget + expected)
        return jnp.linalg.norm(error_absolute) / jnp.sqrt(error_absolute.size)

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
        lengths = []
        for arg in tqdm.tqdm(list_of_args, leave=False):
            sol, aux = fun(arg)

            precision = precision_fun(sol)
            length = len(aux["u0_solve"])
            times = timeit_fun(lambda: fun(arg)[0].block_until_ready())  # noqa: B023

            lengths.append(length)
            precisions.append(precision)
            works_min.append(min(times))
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "list_of_args": list_of_args,
            "length_of_longest_vector": jnp.asarray(lengths),
            "work_min": jnp.asarray(works_min),
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


if __name__ == "__main__":
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to get plotting code
    vf, u0, tspan, params = ivps.pleiades_1st()
    solve = ivpsolvers.asolve_scipy("LSODA", vf, tspan, atol=1e-13, rtol=1e-13)
    ts, ys = solve(u0, params)

    vf_2nd, u0_2nd, tspan, params = ivps.pleiades_2nd()

    # # If we change the probdiffeq-impl halfway through a script, a warning is raised.

    # Read configuration from command line
    args = parse_arguments()
    tols_short, tols = tolerances_from_args(args)
    time = timeit_fun_from_args(args)

    # Save-at:
    xs = jnp.linspace(jnp.amin(ts), jnp.amax(ts), num=50)
    dt0 = 0.1

    # Assemble algorithms

    @jax.jit
    def ts0_3(tol):
        tol *= 10
        u0_like = u0_2nd[0]
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve(
            "ts0-3",
            vf_2nd,
            u0_like,
            save_at=xs,
            dt0=dt0,
            atol=atol,
            rtol=rtol,
            ode_order=2,
        )
        return fun(u0_2nd, params)

    @jax.jit
    def ts0_8(tol):
        tol *= 10
        u0_like = u0_2nd[0]
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve(
            "ts0-8",
            vf_2nd,
            u0_like,
            save_at=xs,
            dt0=dt0,
            atol=atol,
            rtol=rtol,
            ode_order=2,
        )
        return fun(u0_2nd, params)

    @jax.jit
    def ts0_5(tol):
        tol *= 10
        u0_like = u0_2nd[0]
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve(
            "ts0-5",
            vf_2nd,
            u0_like,
            save_at=xs,
            dt0=dt0,
            atol=atol,
            rtol=rtol,
            ode_order=2,
        )
        return fun(u0_2nd, params)

    @jax.jit
    def bosh3(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "bosh3", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol, ode_order=2
        )
        return fun(u0, params)

    @jax.jit
    def dopri5(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "dopri5",
            vf,
            u0_like,
            save_at=xs,
            dt0=dt0,
            atol=atol,
            rtol=rtol,
            ode_order=2,
        )
        return fun(u0, params)

    @jax.jit
    def tsit5(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "tsit5", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol, ode_order=2
        )
        return fun(u0, params)

    @jax.jit
    def dopri8(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "dopri8",
            vf,
            u0_like,
            save_at=xs,
            dt0=dt0,
            atol=atol,
            rtol=rtol,
            ode_order=2,
        )
        return fun(u0, params)

    algorithms = {
        "TS0(3)": (tols, ts0_3),
        "TS0(5)": (tols, ts0_5),
        "TS0(8)": (tols, ts0_8),
        "Bosh3()": (tols, bosh3),
        "Tsit5()": (tols, tsit5),
        "Dopri8()": (tols, dopri8),
    }

    # Compute a reference solution
    reference, _ = dopri5(1e-15)
    precision = rmse_absolute(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, (tols, algo) in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision, timeit_fun=time)
        results[label] = param_to_wp(tols)

    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        jnp.save(os.path.dirname(__file__) + "/plot_ts.npy", ts)
        jnp.save(os.path.dirname(__file__) + "/plot_ys.npy", ys)
        jnp.save(os.path.dirname(__file__) + "/plot_timeseries.npy", xs)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")
