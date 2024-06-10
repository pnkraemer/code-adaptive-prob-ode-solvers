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
    return 0.1 ** jnp.arange(arguments.start, arguments.stop, step=1.0)


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
    jax.config.update("jax_enable_x64", True)

    # Simulate once to get plotting code
    vf, u0, tspan, params = ivps.rigid_body(time_span=(0.0, 50.0))
    solve = ivpsolvers.asolve_scipy("LSODA", vf, tspan, atol=1e-13, rtol=1e-13)
    ts, ys = solve(u0, params)

    # # If we change the probdiffeq-impl halfway through a script, a warning is raised.
    # # But for this benchmark, such a change is on purpose.
    # warnings.filterwarnings("ignore")

    # Read configuration from command line
    args = parse_arguments()
    tolerances = tolerances_from_args(args)
    time = timeit_fun_from_args(args)

    # Save-at:
    xs = jnp.linspace(jnp.amin(ts), jnp.amax(ts), num=5)
    dt0 = jnp.amax(ts) - jnp.amin(ts)

    # Assemble algorithms

    @jax.jit
    def ts0_2(tol):
        tol *= 100
        u0_like = u0
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve(
            "ts0-2", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    @jax.jit
    def ts0_4(tol):
        tol *= 100
        u0_like = u0
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve(
            "ts0-4", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    def ts0_2_interp(tol):
        if tol < 1e-8:
            tol = 1e-3
        tol *= 100

        u0_like = u0
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve_via_interpolate(
            "ts0-2", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    def ts0_4_interp(tol):
        if tol < 1e-8:
            tol = 1e-3
        tol *= 100

        u0_like = u0
        atol, rtol = 1e-3 * tol, tol
        fun = ivpsolvers.solve_via_interpolate(
            "ts0-4", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    @jax.jit
    def bosh3(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "bosh3", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    @jax.jit
    def tsit5(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "tsit5", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    @jax.jit
    def dopri8(tol):
        atol, rtol = 1e-3 * tol, tol
        u0_like = u0
        fun = ivpsolvers.solve_diffrax(
            "dopri8", vf, u0_like, save_at=xs, dt0=dt0, atol=atol, rtol=rtol
        )
        return fun(u0, params)[0]

    algorithms = {
        "TS0(2) (interp., can't jit)": ts0_2_interp,
        "TS0(4) (interp., can't jit)": ts0_4_interp,
        "TS0(2)": ts0_2,
        "TS0(4)": ts0_4,
        "Bosh3()": bosh3,
        "Tsit5()": tsit5,
    }

    # Compute a reference solution
    reference = dopri8(1e-15)
    precision = rmse_absolute(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision, timeit_fun=time)
        results[label] = param_to_wp(tolerances)

    # Save results
    if args.save:
        jnp.save(os.path.dirname(__file__) + "/results.npy", results)
        jnp.save(os.path.dirname(__file__) + "/plot_ts.npy", ts)
        jnp.save(os.path.dirname(__file__) + "/plot_ys.npy", ys)
        print("\nSaving successful.\n")
    else:
        print("\nSkipped saving.\n")
