"""Fit an ODE parameter using adaptive steps.

What's the point?
- Show that adaptive solvers work with Equinox and optax
- We can learn an ODE that fixed steps struggle with (a-priori at least)
-

"""

import diffrax
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
            in_size=2,
            out_size=2,
            width_size=32,
            depth=2,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, u, *, t):
        return self.mlp(u)
        # x, y = u
        # xdot = y
        # ydot = self.mlp(u)
        # return jnp.concatenate([xdot[None], ydot], axis=0)


def main(seed, num_data=1, std=0.5, num_epochs=150, num_batches=1, lr=1e-3):
    # Random number generation
    key = jax.random.PRNGKey(seed)

    # Set up the problem
    t0, t1 = 0.0, 6.3
    save_at = jnp.linspace(t0, t1, num=20)

    # Sample data
    key, *subkeys = jax.random.split(key, num=3)
    vdp = VanDerPol(10.0)
    generate = generate_data(vdp, save_at=save_at, key=subkeys[0], std=std)
    data_in = jax.random.uniform(subkeys[1], shape=(num_data, 2))
    data_out = jax.vmap(generate)(data_in)

    # Set up the optimizer
    pn_loss = pn_loss_function(save_at=save_at, std=std)
    pn_loss = eqx.filter_jit(eqx.filter_value_and_grad(pn_loss))

    rk_loss = rk_loss_function(save_at=save_at)
    rk_loss = eqx.filter_jit(eqx.filter_value_and_grad(rk_loss))

    # Use Equinox's bounded while loop for reverse-differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Initialise the optimizer
        key, subkey = jax.random.split(key, num=2)
        model_before = NeuralODE(subkey)
        pn_model, rk_model = model_before, model_before

        optimizer = optax.adam(lr)
        pn_opt_state = optimizer.init(eqx.filter(pn_model, eqx.is_inexact_array))
        rk_opt_state = optimizer.init(eqx.filter(rk_model, eqx.is_inexact_array))

        rk_best = (rk_model, 10_000)
        pn_best = (pn_model, 10_000)

        # Run the training loop
        try:
            key, subkey = jax.random.split(key, num=2)
            data = dataloader(data_in, data_out, key=subkey, num_batches=num_batches)
            for idx, (inputs, outputs) in zip(range(num_epochs), data):
                pn_val, pn_grads = pn_loss(pn_model, inputs, outputs)
                pn_updates, pn_opt_state = optimizer.update(pn_grads, pn_opt_state)
                pn_model = eqx.apply_updates(pn_model, pn_updates)
                if pn_val < pn_best[1]:
                    pn_best = (pn_model, pn_val)

                rk_val, rk_grads = rk_loss(rk_model, inputs, outputs)
                rk_updates, rk_opt_state = optimizer.update(rk_grads, rk_opt_state)
                rk_model = eqx.apply_updates(rk_model, rk_updates)
                if rk_val < rk_best[1]:
                    rk_best = (rk_model, rk_val)

                label = f"{idx}/{num_epochs} | pn_loss: {pn_val:.2e} | rk_loss: {rk_val:.2e}"
                print(label)

        except KeyboardInterrupt:
            pass

    # Evaluate the test losses
    key, subkey = jax.random.split(key, num=2)
    save_at_test = jnp.sort(jax.random.uniform(key, shape=(100,)))
    save_at_test *= save_at[1] - save_at[0]
    save_at_test += save_at[0]
    test_loss_rk = rk_loss_function(save_at=save_at_test)
    test_loss_pn = pn_loss_function(save_at=save_at_test, std=std)
    test_data_in = data_in
    test_data_out = jax.vmap(rk_solve, in_axes=(None, 0, None))(
        vdp, test_data_in, save_at_test
    ).ys
    rk_rk = test_loss_rk(rk_best[0], test_data_in, test_data_out)
    rk_pn = test_loss_rk(pn_best[0], test_data_in, test_data_out)
    pn_rk = test_loss_pn(rk_best[0], test_data_in, test_data_out)
    pn_pn = test_loss_pn(pn_best[0], test_data_in, test_data_out)
    print()
    print(f"RK loss (MSE): \n\tRK={rk_rk:.4e} \n\tPN={rk_pn:.4e}")
    print(f"PN loss (LML): \n\tRK={pn_rk:.4e} \n\tPN={pn_pn:.4e}")
    print()

    # todo: save the estimates and plot?
    # todo: save the losses and plot?
    #
    #
    # # Plot before and after (at a finer resolution)
    # save_at_plot = jnp.linspace(save_at[0], save_at[-1], num=100)
    # before = pn_solve(model_before, data_in[0], save_at=save_at_plot).u
    # truth = pn_solve(vdp, data_in[0], save_at=save_at_plot).u
    #
    # rk_after = rk_solve(rk_best[0], data_in[0], save_at=save_at_plot).ys
    # pn_after = pn_solve(pn_best[0], data_in[0], save_at=save_at_plot).u

    # # todo: use the loss functions instead of
    # print("RK error:", jnp.linalg.norm(rk_after - truth) / jnp.sqrt(truth.size))
    # print("PN error:", jnp.linalg.norm(pn_after - truth) / jnp.sqrt(truth.size))
    #
    # plt.plot(save_at_plot, before, color="C0", label="Before")
    # plt.plot(save_at_plot, rk_after, color="C1", alpha=0.5, label="After (RK)")
    # plt.plot(save_at_plot, pn_after, color="C2", alpha=0.5, label="After (PN)")
    # plt.plot(save_at_plot, truth, "-", color="black", label="Truth", zorder=0)
    # plt.plot(save_at, data_out[0], "x", color="black", label="Data", zorder=0)
    #
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # plt.show()


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
        posterior = stats.calibrate(solution.posterior, solution.output_scale)

        std_vec = std * jnp.ones_like(save_at)

        lml = stats.log_marginal_likelihood
        lml = functools.partial(lml, posterior=posterior)
        lml = functools.partial(lml, standard_deviation=std_vec)
        return -lml(truth)

    return pn_loss


def rk_loss_function(*, save_at):
    def rk_loss(ode, data_in, data_out):
        pn_losses = rk_loss_single(ode, data_in, data_out)
        return jnp.mean(pn_losses)

    @functools.partial(jax.vmap, in_axes=(None, 0, 0))
    def rk_loss_single(ode, u0, truth):
        solution = rk_solve(ode, u0, save_at=save_at)
        return jnp.mean(jnp.square(solution.ys - truth))

    return rk_loss


def rk_solve(ode, data_in, save_at):
    def func(t, y, args):
        return ode(y, t=t)

    rtol = 1e-3
    atol = 1e-6

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        diffrax.Tsit5(),
        t0=save_at[0],
        t1=save_at[-1],
        dt0=0.1,
        y0=data_in,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=save_at),
    )
    return solution


def pn_solve(ode, data_in, save_at):
    # Any relatively-high-accuracy solution works.
    # Why? Plot the pn_loss-landscape
    num = 4
    rtol = 1e-3
    atol = 1e-6

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts0(ode_order=1)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    pn_solver = ivpsolvers.solver_mle(strategy)

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf, [data_in], num=num)
    output_scale = jnp.ones(())
    init = pn_solver.initial_condition(tcoeffs, output_scale)

    # Build the pn_solver and solve
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(pn_solver, atol=atol, rtol=rtol, control=ctrl)
    solve_fun = functools.partial(ivpsolve.solve_adaptive_save_at, save_at=save_at)
    solve_fun = functools.partial(solve_fun, dt0=0.1)
    solve_fun = functools.partial(solve_fun, adaptive_solver=adaptive_solver)
    return solve_fun(ode, init)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    impl.select("isotropic", ode_shape=(2,))

    for rng in [1, 2, 3]:
        main(rng)
