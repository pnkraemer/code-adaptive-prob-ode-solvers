"""Test the solve-and-save-at functionality."""

import jax.numpy as jnp
import pytest_cases
from odecheckpts import ivpsolvers, ivps


def case_solvers_probdiffeq_versus_diffrax():
    solver = ivpsolvers.solve
    solver_diffrax = ivpsolvers.solve_diffrax
    return solver, solver_diffrax


def case_ivp_1d_logistic():
    return ivps.logistic()


@pytest_cases.parametrize_with_cases("solvers", cases=".", prefix="case_solvers_")
@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="case_ivp_")
def test_two_solvers_return_the_same_solution(solvers: tuple, ivp):
    vf, u0, time_span, args = ivp
    solver1, solver2 = solvers

    dt0 = 0.1
    atol, rtol = 1e-4, 1e-4
    save_at = jnp.linspace(*time_span, num=5)
    u0_like = u0  # infer shapes etc.

    solve1 = solver1(vf, u0_like, save_at, dt0=dt0, atol=atol, rtol=rtol)
    solution1, aux = solve1(u0, args)

    solve2 = solver2(vf, u0_like, save_at, dt0=dt0, atol=atol, rtol=rtol)
    solution2, aux = solve2(u0, args)

    assert jnp.allclose(solution1, solution2, atol=atol, rtol=rtol)
