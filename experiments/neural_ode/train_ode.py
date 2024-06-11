"""Train a neural ODE with ProbDiffEq and Optax."""

import jax.flatten_util
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

from odecheckpts import train_util, ivps, ivpsolvers
from probdiffeq import ivpsolve
from probdiffeq.impl import impl


# Todo: Train the prior diffusion and observation noise, too
# Todo: Use solve_and_save_at instead of solve_fixed_step
# Todo: move the Neural ODE implementation to the src?
#

num_epochs = 5_000

impl.select("isotropic", ode_shape=(1,))

grid = jnp.linspace(0, 1, num=10)
data = jnp.sin(5 * jnp.pi * grid)

layout = [["data", "trained", "loss", "loss"]]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 2), constrained_layout=True)


vf, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))

# Make a solver
solve = ivpsolvers.solve(
    "ts0-1", vf, u0, save_at=grid, dt0=0.1, atol=1e-2, rtol=1e-2, calibrate="none"
)
u, _ = solve((u0,), f_args, output_scale=1.0)


#
# ibm = priors.ibm_adaptive(num_derivatives=1)
# ts0 = corrections.ts0()
# strategy = smoothers.smoother_adaptive(ibm, ts0)
# solver_ts0 = uncalibrated.solver(strategy)
#
# tcoeffs = (u0, vf(u0, t=t0, p=f_args))
# init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)
#
# sol = ivpsolve.solve_fixed_grid(
#     lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0
# )

axes["data"].plot(grid, u, ".-", label="Initial estimate", color="C0")
axes["data"].plot(grid, data, "-", linewidth=5, alpha=0.5, label="Data", color="C1")
axes["data"].legend(fontsize="x-small")

p_ = [(u0,), f_args, 1.0, 1e-2]
p, unflatten = jax.flatten_util.ravel_pytree(p_)

loss_fn = train_util.loss(solver=solve, unflatten=unflatten)
optim = optax.adam(learning_rate=2e-2)
update_fn = train_util.update(optim, loss_fn)


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

losses = jnp.asarray(losses)

axes["trained"].plot(sol.t, data, "-", linewidth=5, alpha=0.5, label="Data", color="C1")
tcoeffs = (u0, vf(u0, t=t0, p=p))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=p), init, grid=grid, solver=solver_ts0
)


axes["trained"].plot(sol.t, sol.u, ".-", label="Final guess", color="C0")

tcoeffs = (u0, vf(u0, t=t0, p=f_args))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0
)


axes["trained"].legend(fontsize="x-small")

axes["loss"].plot(losses)


# Label everything
axes["data"].set_title("Before training", fontsize="medium")
axes["trained"].set_title("After training", fontsize="medium")
axes["loss"].set_title("Loss evolution", fontsize="medium")

for lbl in ["trained", "data"]:
    axes[lbl].set_xlabel("Time $t$")
    axes[lbl].set_ylabel("State $y$")

axes["loss"].set_xlabel("Epoch $i$")
axes["loss"].set_ylabel(r"Loss $\rho$")
plt.show()
