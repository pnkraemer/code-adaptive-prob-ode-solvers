"""Fit an ODE parameter using adaptive steps.

What's the point?
- Show that adaptive solvers work with Equinox and optax
- We can learn an ODE that fixed steps struggle with (a-priori at least)
-

"""

import functools
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


class NeuralODE(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2, out_size=1, width_size=32, depth=2, activation=jnp.tanh, key=key
        )

    def __call__(self, u, *, t):
        x, y = u
        xdot = y
        ydot = self.mlp(u)
        return jnp.concatenate([xdot[None], ydot], axis=0)


def main(num_data=1, std=1e-3, num_epochs=500, num_batches=1):
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(2,))

    # Random number generation
    key = jax.random.PRNGKey(1)

    # Set up the problem
    t0, t1 = 0.0, 1
    save_at = jnp.linspace(t0, t1, num=10)

    # Sample data
    key, *subkeys = jax.random.split(key, num=4)
    mu = jax.random.uniform(subkeys[0], shape=())
    vdp = VanDerPol(mu)
    generate = generate_data(vdp, save_at=save_at, key=subkeys[1], std=std)
    data_in = jax.random.uniform(subkeys[2], shape=(num_data, 2))
    data_out = jax.vmap(generate)(data_in)
    print(f"Truth: {mu:.3f}")

    # Set up the optimizer
    loss = log_likelihood(save_at=save_at, std=std)
    print(f"True loss: {loss(vdp, data_in, data_out):.2e}")
    loss = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    optimizer = optax.adam(1e-3)

    # Use Equinox's bounded while loop for reverse-differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Initialise the optimizer
        key, subkey = jax.random.split(key, num=2)
        model = NeuralODE(subkey)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Run the training loop
        data = dataloader(data_in, data_out, num_batches=num_batches)
        for idx, (inputs, outputs) in zip(range(num_epochs), data):
            val, grads = loss(model, inputs, outputs)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            label = f"{idx}/{num_epochs} | loss: {val:.2e}"
            print(label)


def generate_data(model_true, *, save_at, key, std):
    def generate(u0):
        noise = jax.random.normal(key, shape=(len(save_at), 2))
        return solve(model_true, u0, save_at=save_at).u + std * noise

    return generate


def dataloader(inputs, outputs, /, *, num_batches):
    assert len(inputs) % num_batches == 0, (len(inputs), num_batches)

    idx = jnp.arange(len(inputs))
    key = jax.random.PRNGKey(4123)
    while True:
        key, subkey = jax.random.split(key, num=2)
        idx = jax.random.permutation(subkey, idx)
        for i in idx.reshape((num_batches, -1)):
            yield inputs[i], outputs[i]


def log_likelihood(*, save_at, std):
    def loss(ode, data_in, data_out):
        losses = loss_single(ode, data_in, data_out)
        return jnp.mean(losses)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def loss_single(ode, u0, truth):
        solution = solve(ode, u0, save_at=save_at)
        std_vec = std * jnp.ones_like(save_at)

        lml = stats.log_marginal_likelihood
        lml = functools.partial(lml, posterior=solution.posterior)
        lml = functools.partial(lml, standard_deviation=std_vec)
        return -lml(truth)

    return loss


def solve(ode, data_in, save_at):
    # Any relatively-high-accuracy solution works.
    # Why? Plot the loss-landscape
    num = 3
    tol = 1e-4

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts1 = ivpsolvers.correction_ts1(ode_order=1)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts1)
    solver = ivpsolvers.solver(strategy)  # why? plot loss-landscape

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf, [data_in], num=num)
    output_scale = 1e0  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)

    # Build the solver and solve
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    solve_fun = functools.partial(ivpsolve.solve_adaptive_save_at, save_at=save_at)
    solve_fun = functools.partial(solve_fun, dt0=0.1)
    solve_fun = functools.partial(solve_fun, adaptive_solver=adaptive_solver)
    return solve_fun(ode, init)


if __name__ == "__main__":
    main()
