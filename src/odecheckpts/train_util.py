from typing import Callable

import jax
import jax.numpy as jnp
import optax
from probdiffeq import stats


def loss(solver: Callable, unflatten: Callable):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss_fn(params, *, X, y, stdev, scale, u0):
        """Loss function: log-marginal likelihood of the data."""
        # _u0, p, _output_scale, _standard_deviation = unflatten(params)
        (p,) = unflatten(params)

        sol, info = solver(u0, p, output_scale=scale)
        posterior = info["solution"].posterior

        observation_std = jnp.ones_like(X) * stdev
        marginal_likelihood = stats.log_marginal_likelihood(
            y[:, None], standard_deviation=observation_std, posterior=posterior
        )
        return -1 * marginal_likelihood

    return loss_fn


def update(optimizer, loss_fn, /):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update_fn(params, opt_state, **kwargs):
        """Update the optimiser state."""
        loss_val, grads = jax.value_and_grad(loss_fn)(params, **kwargs)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss_val}

    return update_fn
