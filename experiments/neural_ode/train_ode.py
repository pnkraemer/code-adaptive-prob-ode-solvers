"""Train a neural ODE with ProbDiffEq and Optax."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

import equinox

from odecheckpts import train_util, ivps, ivpsolvers
from probdiffeq.backend import control_flow

# Todo: Train the prior diffusion and observation noise, too
# Todo: Use solve_and_save_at instead of solve_fixed_step
# Todo: move the Neural ODE implementation to the src?
#

# Parameters
num_epochs = 100

# Initialise the figure
layout = [["data", "trained", "loss", "loss"]]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 2), constrained_layout=True)


# Create and plot the data
grid = jnp.linspace(0, 1, num=50)
data = jnp.sin(5 * jnp.pi * grid)
for lbl in ["trained", "data"]:
    axes[lbl].plot(grid, data, "-", linewidth=5, alpha=0.5, label="Data", color="C1")


# Create the neural ODE to be fitted
vf, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))

# Make an ODE solver
solve = ivpsolvers.solve(
    "ts0-1", vf, u0, save_at=grid, dt0=0.1, atol=1e-2, rtol=1e-2, calibrate="none"
)


# Plot the initial guess
u, _ = solve((u0,), f_args, output_scale=1.0)
axes["data"].plot(grid, u, ".-", label="Initial estimate", color="C0")


# Initialise the optimisation problem
p_ = [(u0,), f_args, 1.0, 1e-2]
p, unflatten = jax.flatten_util.ravel_pytree(p_)
loss_fn = train_util.loss(solver=solve, unflatten=unflatten)
optim = optax.adam(learning_rate=1e-1)
update_fn = train_util.update(optim, loss_fn)

# Make a reverse-mode differentiable while-loop


def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


context_compute_gradient = control_flow.context_overwrite_while_loop(while_loop_func)

# Train
with context_compute_gradient:
    losses = []
    state = optim.init(p)
    progressbar = tqdm.tqdm(range(num_epochs))
    loss_value = loss_fn(p, grid, data)
    losses.append(loss_value)
    progressbar.set_description(f"Loss: {loss_value:.1f}")
    for _ in progressbar:
        p, state, info = update_fn(p, state, grid, data)

        loss_value = info["loss"]
        losses.append(loss_value)
        progressbar.set_description(f"Loss: {loss_value:.1f}")
        print(unflatten(p))

# Plot the loss-curve
losses = jnp.asarray(losses)
axes["loss"].plot(losses)

# Plot the final guess
*final, _ = unflatten(p)
u, _ = solve(*final)
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
plt.show()
