"""Test the solve-and-save-at functionality."""

import pytest_cases
from odecheckpts import ivpsolve, ivps


def case_solvers_probdiffeq_and_diffrax():
    solver = ivpsolve.solve_and_save_at
    solver_diffrax = ivpsolve.solve_and_save_at_diffrax
    return solver, solver_diffrax


def case_ivp_rigid_body():
    return ivps.rigid_body()


@pytest_cases.parametrize_with_cases("solvers", cases=".", prefix="case_solvers_")
@pytest_cases.parametrize_with_cases("ivp", cases=".", prefix="case_ivp_")
def test_two_solvers_return_the_same_solution(solvers: tuple, ivp):
    print(solvers)
    print(ivp)
    assert False
    pass
