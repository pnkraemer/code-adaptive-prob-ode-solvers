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
    mlp: eqx.Module

    def __init__(self, key, *, data_size, width_size, depth):
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )

    @property
    def params(self):
        return self._params

    def __call__(self, u, *, t):
        x, y = u
        xdot = y
        ydot = self.params * (1 - x**2) * y - x
        return jnp.asarray([xdot, ydot])


def main(num_data=16, std=1e-3, num_epochs=256, num_batches=4):
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(2,))

    # Random number generation
    key = jax.random.PRNGKey(1)

    # Set up the problem
    t0, t1 = 0.0, 3.15
    save_at = jnp.linspace(t0, t1, num=10)

    # Sample data
    key, subkey = jax.random.split(key, num=2)
    mu = jax.random.uniform(key, shape=())
    model_true = VanDerPol(mu)
    key, subkey = jax.random.split(key, num=2)
    generate = generate_data(model_true, save_at=save_at, key=subkey, std=std)
    print(f"Truth: {mu:.3f}")

    key, subkey = jax.random.split(key, num=2)
    u0s = jax.random.uniform(subkey, shape=(num_data, 2))
    truths = jax.vmap(generate)(u0s)

    # Set up the optimizer
    loss = log_likelihood(save_at=save_at, std=std)
    loss = jax.jit(jax.value_and_grad(loss))
    optimizer = optax.adabelief(1e-2)

    # Use Equinox's bounded while loop for reverse-differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Initialise the optimizer
        r0 = jnp.asarray(0.1)
        vdp = VanDerPol(r0)
        opt_state = optimizer.init(vdp)

        # Run the training loop
        data = dataloader(u0s, truths, num_epochs=num_epochs, num_batches=num_batches)
        for idx, (inputs, outputs) in enumerate(data):
            val, grads = loss(vdp, inputs, outputs)
            updates, opt_state = optimizer.update(grads, opt_state)
            vdp = optax.apply_updates(vdp, updates)

            label = f"{idx}/{num_epochs} | loss: {val:.2e}, p={vdp.params:.3f}"
            print(label)


def generate_data(model_true, *, save_at, key, std):
    def generate(u0):
        noise = jax.random.normal(key, shape=(len(save_at), 2))
        return solve(model_true, u0, save_at=save_at).u + std * noise

    return generate


def dataloader(inputs, outputs, /, *, num_epochs, num_batches):
    idx = jnp.arange(len(inputs))
    key = jax.random.PRNGKey(4123)
    for _ in range(num_epochs // num_batches):
        key, subkey = jax.random.split(key, num=2)
        idx = jax.random.permutation(subkey, idx)
        for i in idx.reshape((num_batches, -1)):
            yield inputs[i], outputs[i]


def log_likelihood(*, save_at, std):
    def loss(ode, u0s, truths):
        losses = loss_single(ode, u0s, truths)
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


def solve(ode, u0s, save_at):
    # any high-accuracy solution works. why? loss-landscape (zoom in!)
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
    tcoeffs = taylor.odejet_padded_scan(vf, [u0s], num=num)
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
