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
import matplotlib.pyplot as plt


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
            in_size=2,
            out_size=1,
            width_size=32,
            depth=2,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, u, *, t):
        x, y = u
        xdot = y
        ydot = self.mlp(u)
        return jnp.concatenate([xdot[None], ydot], axis=0)


def main(seed=1, num_data=1, std=0.5, num_epochs=1_000, num_batches=1, lr=1e-2):
    jax.config.update("jax_enable_x64", True)
    impl.select("blockdiag", ode_shape=(2,))

    # Random number generation
    key = jax.random.PRNGKey(seed)

    # Set up the problem
    t0, t1 = 0.0, 2 * 6.3
    save_at = jnp.linspace(t0, t1, num=10)

    # Sample data
    key, *subkeys = jax.random.split(key, num=4)
    mu = 10 * jax.random.uniform(subkeys[0], shape=())
    vdp = VanDerPol(mu)
    generate = generate_data(vdp, save_at=save_at, key=subkeys[1], std=std)
    data_in = jax.random.uniform(subkeys[2], shape=(num_data, 2))
    data_out = jax.vmap(generate)(data_in)
    print(f"Truth: {mu:.3f}")

    # Set up the optimizer
    pn_loss = pn_loss_function(save_at=save_at, std=std)
    print(f"True pn_loss: {pn_loss(vdp, data_in, data_out):.2e}")
    pn_loss = eqx.filter_jit(eqx.filter_value_and_grad(pn_loss))
    optimizer = optax.adam(lr)

    # Use Equinox's bounded while loop for reverse-differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Initialise the optimizer
        key, subkey = jax.random.split(key, num=2)
        pn_model_before = NeuralODE(subkey)
        pn_model = pn_model_before
        pn_opt_state = optimizer.init(eqx.filter(pn_model, eqx.is_inexact_array))

        # Run the training loop
        try:
            key, subkey = jax.random.split(key, num=2)
            data = dataloader(data_in, data_out, key=subkey, num_batches=num_batches)
            for idx, (inputs, outputs) in zip(range(num_epochs), data):
                val, grads = pn_loss(pn_model, inputs, outputs)
                updates, pn_opt_state = optimizer.update(grads, pn_opt_state)
                pn_model = eqx.apply_updates(pn_model, updates)

                label = f"{idx}/{num_epochs} | pn_loss: {val:.2e}"
                print(label)
        except KeyboardInterrupt:
            pass

    # Plot before and after (at a finer resolution)
    save_at_plot = jnp.linspace(save_at[0], save_at[-1], num=100)
    before = pn_solve(pn_model_before, data_in[0], save_at=save_at_plot)
    after = pn_solve(pn_model, data_in[0], save_at=save_at_plot)
    plt.plot(save_at_plot, before.u, color="C0", label="Before")
    plt.plot(save_at_plot, after.u, color="C1", label="After")
    plt.plot(save_at, data_out[0], "x", color="C2", label="Data")
    plt.legend()
    plt.show()


def generate_data(model_true, *, save_at, key, std):
    def generate(u0):
        noise = jax.random.normal(key, shape=(len(save_at), 2))
        return pn_solve(model_true, u0, save_at=save_at).u + std * noise

    return generate


def dataloader(inputs, outputs, /, *, key, num_batches):
    assert len(inputs) % num_batches == 0, (len(inputs), num_batches)

    idx = jnp.arange(len(inputs))
    while True:
        key, subkey = jax.random.split(key, num=2)
        idx = jax.random.permutation(subkey, idx)
        for i in idx.reshape((num_batches, -1)):
            yield inputs[i], outputs[i]


def pn_loss_function(*, save_at, std):
    def pn_loss(ode, data_in, data_out):
        pn_losses = pn_loss_single(ode, data_in, data_out)
        return jnp.mean(pn_losses)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def pn_loss_single(ode, u0, truth):
        solution = pn_solve(ode, u0, save_at=save_at)
        std_vec = std * jnp.ones_like(save_at)

        lml = stats.log_marginal_likelihood
        lml = functools.partial(lml, posterior=solution.posterior)
        lml = functools.partial(lml, standard_deviation=std_vec)
        return -lml(truth)

    return pn_loss


def pn_solve(ode, data_in, save_at):
    # Any relatively-high-accuracy solution works.
    # Why? Plot the pn_loss-landscape
    num = 3
    tol = 1e-4

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts0(ode_order=1)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    pn_solver = ivpsolvers.solver(strategy)

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf, [data_in], num=num)
    output_scale = jnp.ones((2,))
    init = pn_solver.initial_condition(tcoeffs, output_scale)

    # Build the pn_solver and solve
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(pn_solver, atol=tol, rtol=tol, control=ctrl)
    solve_fun = functools.partial(ivpsolve.solve_adaptive_save_at, save_at=save_at)
    solve_fun = functools.partial(solve_fun, dt0=0.1)
    solve_fun = functools.partial(solve_fun, adaptive_solver=adaptive_solver)
    return solve_fun(ode, init)


if __name__ == "__main__":
    main()
