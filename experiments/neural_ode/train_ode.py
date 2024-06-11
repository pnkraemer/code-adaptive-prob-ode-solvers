"""Train a neural ODE with ProbDiffEq and Optax."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffeqzoo import backend, ivps
import tqdm

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import solution, uncalibrated
from probdiffeq.solvers.strategies import smoothers
from probdiffeq.solvers.strategies.components import corrections, priors

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

num_epochs = 10_000

impl.select("isotropic", ode_shape=(1,))

grid = jnp.linspace(0, 1, num=100)
data = jnp.sin(5 * jnp.pi * grid)

layout = [["data", "trained", "loss", "loss"]]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 2), constrained_layout=True)


def build_loss_fn(vf, initial_values, solver, *, standard_deviation=1e-2):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss_fn(parameters):
        """Loss function: log-marginal likelihood of the data."""
        tcoeffs = (*initial_values, vf(*initial_values, t=t0, p=parameters))
        init = solver.initial_condition(tcoeffs, output_scale=1.0)

        sol = ivpsolve.solve_fixed_grid(
            lambda *a, **kw: vf(*a, **kw, p=parameters), init, grid=grid, solver=solver
        )

        observation_std = jnp.ones_like(grid) * standard_deviation
        marginal_likelihood = solution.log_marginal_likelihood(
            data[:, None], standard_deviation=observation_std, posterior=sol.posterior
        )
        return -1 * marginal_likelihood

    return loss_fn


def build_update_fn(*, optimizer, loss_fn):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update(params, opt_state):
        """Update the optimiser state."""
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update


f, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))


@jax.jit
def vf(y, *, t, p):
    """Evaluate the MLP."""
    return f(y, t, *p)


# Make a solver
ibm = priors.ibm_adaptive(num_derivatives=1)
ts0 = corrections.ts0()
strategy = smoothers.smoother_adaptive(ibm, ts0)
solver_ts0 = uncalibrated.solver(strategy)

tcoeffs = (u0, vf(u0, t=t0, p=f_args))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0
)

axes["data"].plot(sol.t, sol.u, ".-", label="Initial estimate", color="C0")
axes["data"].plot(sol.t, data, "-", linewidth=5, alpha=0.5, label="Data", color="C1")
axes["data"].legend(fontsize="x-small")


loss_fn = build_loss_fn(vf=vf, initial_values=(u0,), solver=solver_ts0)
optim = optax.adam(learning_rate=2e-2)
update_fn = build_update_fn(optimizer=optim, loss_fn=loss_fn)

losses = []
p = f_args
state = optim.init(p)
progressbar = tqdm.tqdm(range(num_epochs))
loss_value = loss_fn(p)
losses.append(loss_value)
progressbar.set_description(f"Loss: {loss_value:.1f}")
for _ in progressbar:
    p, state = update_fn(p, state)
    loss_value = loss_fn(p)
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
# axes["trained"].plot(sol.t, sol.u, ".-", label="Initial guess")


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
