import jax
import jax.numpy as jnp
import optax

from probdiffeq.solvers import solution
from typing import Callable


def loss(solver: Callable, unflatten: Callable):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss_fn(params, X, y):
        """Loss function: log-marginal likelihood of the data."""
        u0, p, output_scale, standard_deviation = unflatten(params)
        sol, info = solver(u0, p, output_scale=output_scale)
        posterior = info["solution"].posterior
        #
        # tcoeffs = (*u0, vf(*u0, t=t0, p=p))
        # init = solver.initial_condition(tcoeffs, output_scale=output_scale)
        #
        # vf_p = functools.partial(vf, p=p)
        # sol = ivpsolve.solve_fixed_grid(
        #     vf_p, init, grid=grid, solver=solver
        # )

        observation_std = jnp.ones_like(X) * standard_deviation
        marginal_likelihood = solution.log_marginal_likelihood(
            y[:, None], standard_deviation=observation_std, posterior=posterior
        )
        return -1 * marginal_likelihood

    return loss_fn


def update(optimizer, loss_fn, /):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update_fn(params, opt_state, X, y):
        """Update the optimiser state."""
        loss_val, grads = jax.value_and_grad(loss_fn)(params, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss_val}

    return update_fn
