"""Fit an ODE parameter using adaptive steps.

What's the point?
- Show that adaptive solvers work with Equinox and optax
- We can learn an ODE that fixed steps struggle with (a-priori at least)
-

"""

import functools
import matplotlib.pyplot as plt
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import control_flow as cfl
from probdiffeq.impl import impl


class VanDerPol(eqx.Module):
    _params: jax.Array

    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        return self._params

    def __call__(self, u, *, t):
        x, y = u
        xdot = y
        ydot = self.params * (1 - x**2) * y - x
        return jnp.asarray([xdot, ydot])


def main():
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(2,))

    # Set up the problem
    t0, t1 = 0.0, 3.15
    save_at = jnp.linspace(t0, t1, num=100)
    params_true = jnp.asarray(1000.0)
    vdp_true = VanDerPol(params_true)

    sol = solve(vdp_true, jnp.array([2.0, 0.0]), save_at=save_at)
    plt.plot(save_at, sol.u)
    plt.show()

    assert False

    @jax.vmap
    def generate_data(u0):
        noise = jax.random.normal(jax.random.PRNGKey(4), shape=(len(save_at), 2))
        return solve(vdp_true, u0, save_at=save_at).u + 1e-3 * noise

    num_data = 16
    u0s = jax.random.uniform(jax.random.PRNGKey(1), shape=(num_data, 2))
    truths = generate_data(u0s)

    loss = log_likelihood(save_at=save_at)

    # Use Equinox's bounded while loop for differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Set up the optimizer
        loss = jax.jit(jax.value_and_grad(loss))
        optimizer = optax.sgd(1e-5)
        print(f"Truth: {loss(vdp_true, u0s, truths)[0]:.2e}, p={vdp_true.params}")

        # Initialise the optimizer
        r0 = jnp.asarray(900.0)
        vdp = VanDerPol(r0)
        opt_state = optimizer.init(vdp)

        # Run the training loop
        num_epochs = 256
        data = dataloader(u0s, truths, num_epochs=num_epochs, num_batches=4)
        for idx, (inputs, outputs) in enumerate(data):
            val, grads = loss(vdp, inputs, outputs)
            updates, opt_state = optimizer.update(grads, opt_state)
            vdp = optax.apply_updates(vdp, updates)

            print(f"{grads._params:.1e}")
            label = f"{idx}/{num_epochs} | loss: {val:.2e}, p={vdp.params}"
            print(label)


def dataloader(inputs, outputs, /, *, num_epochs, num_batches):
    idx = jnp.arange(len(inputs))
    key = jax.random.PRNGKey(4123)
    for _ in range(num_epochs // num_batches):
        key, subkey = jax.random.split(key, num=2)
        idx = jax.random.permutation(subkey, idx)
        for i in idx.reshape((num_batches, -1)):
            yield inputs[i], outputs[i]


def log_likelihood(save_at):
    def loss(ode, u0s, truths):
        losses = loss_single(ode, u0s, truths)
        return jnp.mean(losses)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def loss_single(ode, u0, truth):
        solution = solve(ode, u0, save_at=save_at)

        std = 1e-3 * jnp.ones_like(save_at)

        lml = stats.log_marginal_likelihood
        lml = functools.partial(lml, posterior=solution.posterior)
        lml = functools.partial(lml, standard_deviation=std)
        return -lml(truth)

    return loss


def solve(ode, u0s, save_at):
    # any high-accuracy solution works. why? loss-landscape (zoom in!)
    num = 4
    tol = 1e-8

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts1 = ivpsolvers.correction_ts1(ode_order=1)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts1)
    solver = ivpsolvers.solver(strategy)  # why? plot loss-landscape

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf, [u0s], num=num)
    output_scale = 1e0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)

    # return ivpsolve.solve_fixed_grid(ode, init, grid=save_at, solver=solver)

    # Compute a baseline solution
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    return ivpsolve.solve_adaptive_save_at(
        ode, init, save_at=save_at, dt0=0.1, adaptive_solver=adaptive_solver
    )


if __name__ == "__main__":
    main()
