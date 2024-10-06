import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import control_flow
from probdiffeq.impl import impl


def main(
    dataset_size=16,
    batch_size=1,
    lr_strategy=(1e-1, 1e-1),
    steps_strategy=(100, 100),
    length_strategy=(1.0, 1),
    width_size=4,
    depth=2,
    seed=5678,
    plot=True,
    print_every=1,
    data_resolution=100,
    mode="probdiffeq",
):
    jax.config.update("jax_enable_x64", True)

    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key, resolution=data_resolution)
    _, length_size, data_size = ys.shape

    impl.select("isotropic", ode_shape=ys[0, 0, ...].shape)
    model = NeuralODE(data_size, width_size, depth, key=model_key, mode=mode)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        if mode == "diffrax":
            y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
            return jnp.mean((yi - y_pred.ys) ** 2)
        if mode == "probdiffeq":
            y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])

            stds = jnp.ones_like(ti) * model.std
            lmls = jax.vmap(
                lambda a, b, c: stats.log_marginal_likelihood(
                    a, standard_deviation=b, posterior=c
                ),
                in_axes=(0, None, 0),
            )(yi, stds, y_pred.posterior)
            return -lmls.mean()

        raise ValueError

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, grads

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            # start = time.time()
            loss, model, opt_state, grads = make_step(_ts, yi, model, opt_state)
            # model, opt_state = run_lbfgs(model, loss, max_iter=2)
            # end = time.time()
            print("std", model.std, "sigma", model.sigma)
            print("std", grads.std, "sigma", grads.sigma)
            print()
            #
            # if (step % print_every) == 0 or step == steps - 1:
            #     print(
            #         f"Step: {step}, Loss: {loss:.3e}, Time: {(end - start):.3e}, Sigma: {model.sigma:.3e}, Std: {model.std:.3e}"
            #     )

        if plot:
            plt.plot(ts, ys[0, :, 0], "x", c="dodgerblue", label="Real")
            plt.plot(ts, ys[0, :, 1], "x", c="dodgerblue")
            if mode == "diffrax":
                model_y = model(ts, ys[0, 0]).ys
            elif mode == "probdiffeq":
                model_y = model(ts, ys[0, 0]).u
            plt.plot(ts, model_y[:, 0], ".", c="crimson", label="Model")
            plt.plot(ts, model_y[:, 1], ".", c="crimson")
            plt.legend()
            plt.tight_layout()
            # plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
            plt.show()

    return ts, ys, model


def get_data(dataset_size, *, key, resolution):
    ts = jnp.linspace(0, 10, num=resolution)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys


def _get_data(ts, *, key):
    y0 = jr.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class NeuralODE(eqx.Module):
    func: Func
    mode: str

    # Trainable solver-parameters
    _sigma: jax.Array
    _std: jax.Array

    def __init__(self, data_size, width_size, depth, *, key, mode, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.func = Func(data_size, width_size, depth, key=key)

        self._sigma = 1e20 * jnp.ones((), dtype=float)
        self._std = -5 * jnp.ones((), dtype=float)

    @property
    def sigma(self):
        return 1e10
        # return jax.lax.stop_gradient(self._sigma)

    @property
    def std(self):
        return 1e-3
        # return self._std

    def __call__(self, ts, y0):
        atol, rtol = 1e-3, 1e-3
        if self.mode == "diffrax":
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return solution
        if self.mode == "probdiffeq":
            num_derivs = 4
            ode_order = 1
            ibm = ivpsolvers.prior_ibm(num_derivatives=num_derivs)
            ts1 = ivpsolvers.correction_ts0(ode_order=ode_order)
            strategy = ivpsolvers.strategy_fixedpoint(ibm, ts1)
            solver = ivpsolvers.solver(strategy)
            ctrl = ivpsolve.control_proportional_integral()

            # Set up the initial condition
            t0 = ts[0]
            num = num_derivs + 1 - ode_order
            tcoeffs = taylor.odejet_unroll(
                lambda y: self.func(t0, y, args=()), [y0], num=num
            )
            output_scale = self.sigma * jnp.ones((), dtype=float)
            init = solver.initial_condition(tcoeffs, output_scale)

            asolver = ivpsolve.adaptive(solver, atol=atol, rtol=rtol, control=ctrl)

            def vf(y, *, t):
                return self.func(t, y, args=())

            with control_flow.context_overwrite_while_loop(while_loop_func):
                solution = ivpsolve.solve_adaptive_save_at(
                    vf, init, save_at=ts, dt0=1.0, adaptive_solver=asolver
                )

            return solution

        raise ValueError


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def while_loop_func(*a, **kw):
    return eqx.internal.while_loop(*a, **kw, kind="bounded", max_steps=1_000)


if __name__ == "__main__":
    main()
