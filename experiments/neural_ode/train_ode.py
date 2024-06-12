"""Train a neural ODE with ProbDiffEq and Optax."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm
import os
import equinox

from odecheckpts import train_util, exp_util, ivps, ivpsolvers
from probdiffeq.backend import control_flow
import jax

PLOT_PARAMS = exp_util.plot_params()


def main():
    plt.rcParams.update(PLOT_PARAMS)

    jax.config.update("jax_enable_x64", True)

    # todo: minibatch the data?
    # Todo: why is the sine curve easy with fixed steps but harder with adaptive steps?
    # Todo: improve the MLP construction
    # Todo: decide which parameters to optimise:
    #   output scale would also be great, regression noise too
    # Todo: train a Runge-Kutta method for comparison
    # Todo: run for different seeds (probably wait until successful)
    # Todo: verify that the log-marginal likelihood actually sees all data

    # Parameters
    num_epochs = 10_000

    # Initialise the figure
    layout = [["data", "trained", "loss", "loss"]]
    fig, axes = plt.subplot_mosaic(layout, figsize=(8, 2), constrained_layout=True)

    # Create and plot the data
    grid = jnp.linspace(0, 1, num=10)
    data = jnp.sin(2 * jnp.pi * grid)
    for lbl in ["trained", "data"]:
        axes[lbl].plot(
            grid, data, "-", linewidth=5, alpha=0.5, label="Data", color="C1"
        )

    # Create the neural ODE to be fitted
    vf, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))

    # Make an ODE solver
    solve = ivpsolvers.solve(
        "ts0-4", vf, u0, save_at=grid, dt0=1.0, atol=1e-1, rtol=1e-1, calibrate="none"
    )

    # Plot the initial guess
    u, _ = solve((u0,), f_args, output_scale=1.0)
    axes["data"].plot(grid, u, ".-", label="Initial estimate", color="C0")

    # Initialise the optimisation problem
    p_ = [f_args]
    p, unflatten = jax.flatten_util.ravel_pytree(p_)
    loss_fn = train_util.loss(solver=solve, unflatten=unflatten)
    optim = optax.adam(learning_rate=1e-2)
    update = train_util.update(optim, loss_fn)

    # Make a reverse-mode differentiable while-loop

    context_compute_gradient = control_flow.context_overwrite_while_loop(
        while_loop_func
    )

    # Train
    with context_compute_gradient:
        losses = []
        state = optim.init(p)
        progressbar = tqdm.tqdm(range(num_epochs))
        loss_value = loss_fn(p, X=grid, y=data, u0=(u0,), stdev=1e-2, scale=1.0)
        losses.append(loss_value)
        progressbar.set_description(f"Loss: {loss_value:.4e}")
        for _ in progressbar:
            try:
                p, state, info = update(
                    p, state, X=grid, y=data, u0=(u0,), stdev=1e-2, scale=1.0
                )

                loss_value = info["loss"]
                losses.append(loss_value)
                progressbar.set_description(f"Loss: {loss_value:.4e}")
            except KeyboardInterrupt:
                break

    # Plot the loss-curve
    losses = jnp.asarray(losses)
    axes["loss"].plot(losses)
    axes["loss"].set_yscale("symlog")

    # Plot the final guess
    (p_,) = unflatten(p)
    u, _ = solve((u0,), p_, output_scale=1.0)
    axes["trained"].plot(grid, u, ".-", label="Final guess", color="C0")

    # Label everything
    axes["data"].set_title("Before training", fontsize="medium")
    axes["trained"].set_title("After training", fontsize="medium")
    axes["loss"].set_title("Loss evolution", fontsize="medium")
    for lbl in ["trained", "data"]:
        axes[lbl].set_xlabel("Time $t$")
        axes[lbl].set_ylabel("State $y$")
        axes[lbl].legend(fontsize="x-small")
    axes["loss"].set_xlabel("Epoch $i$")
    axes["loss"].set_ylabel(r"Loss $\rho$")

    # Show the plot
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()


def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


if __name__ == "__main__":
    main()
