"""Test the solve-and-save-at functionality."""

import functools

import jax.numpy as jnp
import pytest_cases
from odecheckpts import ivpsolvers, ivps


@pytest_cases.parametrize("m0", ["ts0-2", "ts0-4"])
@pytest_cases.parametrize("m1", ["bosh3", "tsit5"])
def case_solvers_checkpoint_versus_diffrax(m0: str, m1: str):
    solver = functools.partial(ivpsolvers.solve, m0)
    solver_diffrax = functools.partial(ivpsolvers.solve_diffrax, m1)
    return solver, solver_diffrax


@pytest_cases.parametrize("m0", ["ts0-2", "ts0-4"])
@pytest_cases.parametrize("m1", ["bosh3", "tsit5"])
def case_solvers_interpolate_versus_diffrax(m0, m1):
    solver = functools.partial(ivpsolvers.solve_via_interpolate, m0)
    solver_diffrax = functools.partial(ivpsolvers.solve_diffrax, m1)
    return solver, solver_diffrax


def case_ivp_1d_logistic():
    return ivps.logistic()


@pytest_cases.parametrize_with_cases("solvers", cases=".", prefix="case_solvers_")
@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="case_ivp_")
def test_two_solvers_return_the_same_solution(solvers: tuple, ivp):
    vf, u0, time_span, args = ivp
    solver1, solver2 = solvers

    dt0 = 0.1
    atol, rtol = 1e-3, 1e-3
    save_at = jnp.linspace(*time_span, num=5)
    u0_like = u0  # infer shapes etc.

    solve1 = solver1(vf, u0_like, save_at, dt0=dt0, atol=atol, rtol=rtol)
    solution1, aux = solve1(u0, args)

    solve2 = solver2(vf, u0_like, save_at, dt0=dt0, atol=atol, rtol=rtol)
    solution2, aux = solve2(u0, args)

    assert jnp.allclose(solution1, solution2, atol=jnp.sqrt(atol), rtol=jnp.sqrt(rtol))
