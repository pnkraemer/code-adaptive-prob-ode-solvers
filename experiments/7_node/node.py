"""Fit an ODE parameter using adaptive steps.

What's the point?
- Show that adaptive solvers work with Equinox and optax
- We can learn an ODE that fixed steps struggle with (a-priori at least)
-

"""

import functools
import os

import diffrax
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
    sigma: jax.Array  # is it hacky to include it here? After all, it's the model...

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            width_size=32,
            depth=2,
            activation=jnp.tanh,
            key=key,
        )
        self.sigma = jnp.asarray(4.0)

    def __call__(self, u, *, t):
        return self.mlp(u)


def main():
    jax.config.update("jax_enable_x64", True)
    impl.select("isotropic", ode_shape=(2,))

    # todo: verify the robustness of the results.

    # Use Equinox's bounded while loop for reverse-differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=100)
    with cfl.context_overwrite_while_loop(loop):
        # Use different seeds.
        # Discard the "unsuccessful" ones later
        for rng in [1, 2, 3]:
            s = 0.1
            losses, plots = run(rng, std=s)
            filename = os.path.dirname(__file__) + "/data_losses"
            jnp.save(f"{filename}_rng_{rng}_std_{s}.npy", losses, allow_pickle=True)

            print()
            print(losses)
            print()

            filename = os.path.dirname(__file__) + "/data_plots"
            jnp.save(f"{filename}_rng_{rng}_std_{s}.npy", plots, allow_pickle=True)


def run(seed, std, num_epochs=500, lr=1e-3):
    # Random number generation
    key = jax.random.PRNGKey(seed)

    # Set up the problem
    t0, t1 = 0.0, 6.3 * 3
    save_at = jnp.linspace(t0, t1, num=20)
    vdp = VanDerPol(1.0)
    u0 = jnp.array([2.0, 0.0])

    # Sample data
    generate = generate_data(vdp, u0=u0, std=std)
    key, subkey = jax.random.split(key, num=2)
    data = generate(key=subkey, save_at=save_at)

    # Set up the loss function(s)
    pn_loss = pn_loss_function(u0=u0, std=std)
    pn_loss_grad = eqx.filter_jit(eqx.filter_value_and_grad(pn_loss))
    rk_loss = rk_loss_function(u0=u0)
    rk_loss_grad = eqx.filter_jit(eqx.filter_value_and_grad(rk_loss))
    optimizer = optax.adam(lr)

    # Initialise the optimizer
    key, subkey = jax.random.split(key, num=2)
    model_before = NeuralODE(subkey)
    pn_model, rk_model = model_before, model_before
    pn_opt_state = optimizer.init(eqx.filter(pn_model, eqx.is_inexact_array))
    rk_opt_state = optimizer.init(eqx.filter(rk_model, eqx.is_inexact_array))

    # Store the best-so-far results. '10_000' is a dummy for 'large loss'.
    rk_best = (rk_model, 10_000)
    pn_best = (pn_model, 10_000)

    try:
        # Run the training loop
        for idx in range(num_epochs):
            pn_val, pn_grads = pn_loss_grad(pn_model, save_at=save_at, data=data)
            pn_updates, pn_opt_state = optimizer.update(pn_grads, pn_opt_state)
            pn_model = eqx.apply_updates(pn_model, pn_updates)
            if pn_val < pn_best[1]:
                pn_best = (pn_model, pn_val)

            rk_val, rk_grads = rk_loss_grad(rk_model, save_at=save_at, data=data)
            rk_updates, rk_opt_state = optimizer.update(rk_grads, rk_opt_state)
            rk_model = eqx.apply_updates(rk_model, rk_updates)
            if rk_val < rk_best[1]:
                rk_best = (rk_model, rk_val)

            # Print every K-th iteration
            if idx % 20 == 0:
                label = f"{idx}/{num_epochs} | pn_loss (n-lml): {pn_val:.2e} | rk_loss (mse): {rk_val:.2e}"
                print(label)
    except KeyboardInterrupt:
        pass

    print(rk_model.sigma)
    print(pn_model.sigma)

    # Evaluate the test losses

    key, subkey = jax.random.split(key, num=2)
    save_at_test = jax.random.uniform(key, shape=(98,))
    save_at_test = jnp.concatenate([save_at_test, save_at[0][None], save_at[-1][None]])
    save_at_test = jnp.sort(save_at_test)
    save_at_test *= save_at[1] - save_at[0]
    save_at_test += save_at[0]

    key, subkey = jax.random.split(key, num=2)
    data_test = generate(subkey, save_at_test)

    rk_rk = rk_loss(rk_best[0], save_at=save_at_test, data=data_test)
    rk_pn = rk_loss(pn_best[0], save_at=save_at_test, data=data_test)
    pn_rk = pn_loss(rk_best[0], save_at=save_at_test, data=data_test)
    pn_pn = pn_loss(pn_best[0], save_at=save_at_test, data=data_test)

    mses = {"RK-learned": rk_rk, "PN-learned": rk_pn}
    lmls = {"RK-learned": pn_rk, "PN-learned": pn_pn}
    losses = {r"MSE ($\downarrow$)": mses, r"N-LML ($\downarrow$)": lmls}

    save_at = jnp.linspace(save_at[0], save_at[-1], num=100)
    truth = rk_solve(vdp, u0=u0, save_at=save_at).ys
    before = rk_solve(model_before, u0=u0, save_at=save_at).ys
    rk = rk_solve(rk_best[0], u0=u0, save_at=save_at).ys
    pn = pn_solve(pn_best[0], u0=u0, save_at=save_at).u
    plots = {"ts": save_at, "truth": truth, "before": before, "rk": rk, "pn": pn}
    return losses, plots


def generate_data(model_true, *, u0, std):
    def generate(key, save_at):
        noise = jax.random.normal(key, shape=(len(save_at), 2))
        return rk_solve(model_true, u0=u0, save_at=save_at).ys + std * noise

    return generate


def rk_loss_function(*, u0):
    def rk_loss_single(ode, *, save_at, data):
        solution = rk_solve(ode, u0=u0, save_at=save_at)
        return jnp.mean(jnp.square(solution.ys - data))

    return rk_loss_single


def rk_solve(ode, *, u0, save_at):
    def func(t, y, args):  # noqa: ARG001
        return ode(y, t=t)

    rtol = 1e-3
    atol = 1e-6

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(func),
        diffrax.Tsit5(),
        t0=save_at[0],
        t1=save_at[-1],
        dt0=0.1,
        y0=u0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=save_at),
    )
    return solution


def pn_loss_function(*, u0, std):
    def pn_loss_single(ode, *, save_at, data):
        solution = pn_solve(ode, u0=u0, save_at=save_at)
        posterior = solution.posterior

        std_vec = std * jnp.ones_like(save_at)

        lml = stats.log_marginal_likelihood
        lml = functools.partial(lml, posterior=posterior)
        lml = functools.partial(lml, standard_deviation=std_vec)
        return -lml(data)

    return pn_loss_single


def pn_solve(ode, *, u0, save_at):
    # Any relatively-high-accuracy solution works.
    # Why? Plot the pn_loss-landscape
    num = 4
    rtol = 1e-3
    atol = 1e-6

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts0(ode_order=1)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    pn_solver = ivpsolvers.solver(strategy)

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf, [u0], num=num)
    output_scale = 10.0**ode.sigma * jnp.ones(())
    init = pn_solver.initial_condition(tcoeffs, output_scale)

    # Build the pn_solver and solve
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(pn_solver, atol=atol, rtol=rtol, control=ctrl)
    solve_fun = functools.partial(ivpsolve.solve_adaptive_save_at, save_at=save_at)
    solve_fun = functools.partial(solve_fun, dt0=0.1)
    solve_fun = functools.partial(solve_fun, adaptive_solver=adaptive_solver)

    return solve_fun(ode, init)


if __name__ == "__main__":
    main()
