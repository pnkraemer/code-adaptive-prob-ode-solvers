"""Estimate ODE paramaters with ProbDiffEq and BlackJAX."""

import functools
from typing import Callable

import blackjax
import equinox as eqx
import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.impl import impl
from probdiffeq.util.doc_util import notebook

from odecheckpts import exp_util


def main(num_samples=150, num_steps_warmup=200):
    # x64 precision
    jax.config.update("jax_enable_x64", True)

    # CPU
    jax.config.update("jax_platform_name", "cpu")

    # IVP examples in JAX
    if not backend.has_been_selected:
        backend.select("jax")

    # Nice-looking plots
    plt.rcParams.update(notebook.plot_style())
    plt.rcParams.update(notebook.plot_sizes())

    impl.select("isotropic", ode_shape=(2,))

    # Create a problem and an initial guess

    # next: replicate https://proceedings.mlr.press/v216/ott23a/ott23a-supp.pdf
    model_true, (t0, t1) = model_ivp()

    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key, num=2)
    model = exp_util.tree_random_like(subkey, model_true)

    # Visualise the initial guess and the data
    save_at = jnp.linspace(t0, t1, num=250, endpoint=True)
    solve_save_at = solver_adaptive(t0, save_at=save_at)

    fig, ax = plt.subplots(figsize=(5, 3))

    data_kwargs = {"alpha": 0.5, "color": "gray"}
    ax.annotate("Data", (13.0, 30.0), **data_kwargs)
    sol = solve_save_at(model_true)
    ax = plot_solution(sol, ax=ax, **data_kwargs)

    guess_kwargs = {"color": "C3"}
    ax.annotate("Initial guess", (7.5, 20.0), **guess_kwargs)
    sol = solve_save_at(model)
    ax = plot_solution(sol, ax=ax, **guess_kwargs)
    plt.show()

    # Define the fixed-step solver

    ts = jnp.linspace(t0, t1, endpoint=True, num=100)
    solve_fixed = solver_fixed(t0, grid=ts)
    data = solve_fixed(model_true).u

    # Define the loss function. For now, use
    # fixed steps for reverse-mode differentiability:

    # mean, cov = model.prior()
    log_M = log_posterior_fun(data=data, solve_fixed=solve_fixed)

    # Build a BlackJax sampler:
    key, subkey = jax.random.split(key, num=2)

    # Warmup
    warmup = blackjax.window_adaptation(blackjax.nuts, log_M, progress_bar=True)
    warmup_results, _ = warmup.run(subkey, model, num_steps=num_steps_warmup)
    initial_state = warmup_results.state
    step_size = warmup_results.parameters["step_size"]
    inverse_mass_matrix = warmup_results.parameters["inverse_mass_matrix"]
    nuts_kernel = blackjax.nuts(
        logdensity_fn=log_M,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # Inference loop
    key, subkey = jax.random.split(key, 2)
    states = inference_loop(
        subkey, kernel=nuts_kernel, initial_state=initial_state, num_samples=num_samples
    )

    solution_samples = jax.vmap(solve_save_at)(states.position)

    # Visualise the initial guess and the data

    fig, ax = plt.subplots()

    sample_kwargs = {"color": "C0"}
    ax.annotate("Samples", (2.75, 31.0), **sample_kwargs)
    for sol in solution_samples:
        ax = plot_solution(sol, ax=ax, linewidth=0.1, alpha=0.75, **sample_kwargs)

    data_kwargs = {"color": "gray"}
    ax.annotate("Data", (18.25, 40.0), **data_kwargs)
    sol = solve_save_at(model_true)
    ax = plot_solution(sol, ax=ax, linewidth=4, alpha=0.5, **data_kwargs)

    guess_kwargs = {"color": "gray"}
    ax.annotate("Initial guess", (6.0, 12.0), **guess_kwargs)
    sol = solve_save_at(model)
    ax = plot_solution(sol, ax=ax, linestyle="dashed", alpha=0.75, **guess_kwargs)
    plt.show()

    plt.title("Posterior samples (parameter space)")
    plt.plot(
        states.position.u0[:, 0], states.position.u0[:, 1], "o", alpha=0.5, markersize=4
    )
    plt.plot(model_true.u0[0], model_true.u0[1], "P", label="Truth", markersize=8)
    plt.plot(model.u0[0], model.u0[1], "P", label="Initial guess", markersize=8)
    plt.legend()
    plt.show()
    #
    # xlim = 17, jnp.amax(states.position.u0[:, 0]) + 0.5
    # ylim = 17, jnp.amax(states.position.u0[:, 1]) + 0.5
    #
    # xs = jnp.linspace(*xlim, endpoint=True, num=300)
    # ys = jnp.linspace(*ylim, endpoint=True, num=300)
    # Xs, Ys = jnp.meshgrid(xs, ys)
    #
    # Thetas = jnp.stack((Xs, Ys))
    # log_M_vmapped_x = jax.vmap(log_M, in_axes=-1, out_axes=-1)
    # log_M_vmapped = jax.vmap(log_M_vmapped_x, in_axes=-1, out_axes=-1)
    # Zs = log_M_vmapped(Thetas)
    #
    # fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3))
    #
    # ax_samples, ax_heatmap = ax
    #
    # fig.suptitle("Posterior samples (parameter space)")
    # ax_samples.plot(
    #     states.position.u0[:, 0], states.position.u0[:, 1], ".", alpha=0.5, markersize=4
    # )
    # ax_samples.plot(
    #     model_true.u0[0], model_true.u0[1], "P", label="Truth", markersize=8
    # )
    # ax_samples.plot(model.u0[0], model.u0[1], "P", label="Initial guess", markersize=8)
    # ax_samples.legend()
    # im = ax_heatmap.contourf(Xs, Ys, jnp.exp(Zs), cmap="cividis", alpha=0.8)
    # plt.colorbar(im)
    # plt.show()


class ODE(eqx.Module):
    _u0: jax.Array
    _args: jax.Array = eqx.field(static=True)
    _unravel: Callable = eqx.field(static=True)
    vf: Callable = eqx.field(static=True)

    def __init__(self, u0, args, vf):
        self._u0 = jnp.sqrt(u0)
        self._args, self._unravel = jax.flatten_util.ravel_pytree(args)
        self.vf = vf

    @property
    def u0(self):
        return self._u0**2
        # return jax.nn.softplus(self._u0)

    @property
    def args(self):
        return self._unravel(self._args)

    @jax.jit
    def __call__(self, y, *, t):
        return self.vf(y, *self.args)

    def prior(self):
        mean = self.u0
        cov = jnp.eye(len(self.u0)) * 30  # fairly uninformed prior
        return mean, cov


def model_ivp():
    vf, u0, (t0, t1), params = ivps.lotka_volterra()
    return ODE(vf=vf, u0=u0, args=params), (t0, t1)


def plot_solution(sol, *, ax, marker=".", **plotting_kwargs):
    """Plot the IVP solution."""
    for d in [0, 1]:
        ax.plot(sol.t, sol.u[:, d], marker="None", **plotting_kwargs)
        ax.plot(sol.t[0], sol.u[0, d], marker=marker, **plotting_kwargs)
        ax.plot(sol.t[-1], sol.u[-1, d], marker=marker, **plotting_kwargs)
    return ax


def solver_fixed(t0, *, grid):
    def solve_fixed(model):
        """Evaluate the parameter-to-solution map, solving on a fixed grid."""
        # Create a probabilistic solver
        ibm = ivpsolvers.prior_ibm(num_derivatives=2)
        ts0 = ivpsolvers.correction_ts0()
        strategy = ivpsolvers.strategy_filter(ibm, ts0)
        solver = ivpsolvers.solver(strategy)

        tcoeffs = taylor.odejet_padded_scan(
            lambda y: model(y, t=t0), (model.u0,), num=2
        )
        output_scale = 10.0
        init = solver.initial_condition(tcoeffs, output_scale)

        sol = ivpsolve.solve_fixed_grid(model, init, grid=grid, solver=solver)
        return sol[-1]

    return jax.jit(solve_fixed)


def solver_adaptive(t0, *, save_at):
    def solve(model: ODE):
        """Evaluate the parameter-to-solution map, solving on an adaptive grid."""
        # Create a probabilistic solver
        ibm = ivpsolvers.prior_ibm(num_derivatives=2)
        ts0 = ivpsolvers.correction_ts0()
        strategy = ivpsolvers.strategy_filter(ibm, ts0)
        solver = ivpsolvers.solver(strategy)
        adaptive_solver = ivpsolve.adaptive(solver)

        tcoeffs = taylor.odejet_padded_scan(
            lambda y: model(y, t=t0), (model.u0,), num=2
        )
        output_scale = 10.0
        init = solver.initial_condition(tcoeffs, output_scale)
        return ivpsolve.solve_adaptive_save_at(
            model, init, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1
        )

    return jax.jit(solve)


def log_posterior_fun(data, solve_fixed, obs_stdev=0.1):
    @jax.jit
    def logpost(model):
        """Evaluate the logposterior-function of the data."""

        y_T = solve_fixed(model)
        lml = stats.log_marginal_likelihood_terminal_values
        logpdf_data = lml(data, standard_deviation=obs_stdev, posterior=y_T.posterior)

        mean, cov = model.prior()
        pdf = jax.scipy.stats.multivariate_normal.logpdf
        logpdf_prior = pdf(model.u0, mean=mean, cov=cov)
        return logpdf_data + logpdf_prior

    return logpost


@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run BlackJAX' inference loop."""

    def one_step(state, rng_key):
        state, _ = kernel.step(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


if __name__ == "__main__":
    main()
