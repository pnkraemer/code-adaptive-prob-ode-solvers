"""Estimate ODE paramaters with ProbDiffEq and BlackJAX."""

import functools
from typing import Callable, Any

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
    # x64 precision & CPU
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    # Nice-looking plots
    plt.rcParams.update(notebook.plot_style())
    plt.rcParams.update(notebook.plot_sizes())

    # Create a problem and an initial guess
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key, num=2)
    model_true, (t0, t1) = model_ivp()
    model = exp_util.tree_random_like(subkey, model_true)

    # Visualise the initial guess and the data
    grid_plot = jnp.linspace(t0, t1, num=250, endpoint=True)
    grid_data = jnp.linspace(t0, t1, num=100, endpoint=True)
    solver = Solver(grid_plot=grid_plot, grid_data=grid_data, ode_shape=(2,))

    # Make a figure
    kwargs = {"Data": {"alpha": 0.5, "color": "gray"}, "Initial guess": {"color": "C3"}}
    figure = Figure(figsize=(5, 3), kwargs=kwargs)

    data = solver.solve_data(model_true)
    figure.plot_solution(data, label="Data")

    guess = solver.solve_data(model)
    figure.plot_solution(guess, label="Initial guess")

    figure.legend()
    figure.show()

    log_M = log_posterior_fun(data=data, solver=solver)

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

    print(states)
    assert False
    solution_samples = jax.vmap(solver.solve_data)(states.position)
    # Visualise the initial guess and the data

    sample_kwargs = {"color": "C0", "label": "Samples"}
    for sol in solution_samples:
        figure.plot_solution(sol, linewidth=0.1, alpha=0.75, **sample_kwargs)

    data_kwargs = {"color": "gray", "label": "Data"}
    sol = solve_save_at(model_true)
    ax = figure.plot_solution(sol, ax=ax, linewidth=4, alpha=0.5, **data_kwargs)

    guess_kwargs = {"color": "gray", "label": "Initial guess"}
    sol = solve_save_at(model)
    ax = figure.plot_solution(
        sol, ax=ax, linestyle="dashed", alpha=0.75, **guess_kwargs
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()

    plt.title("Posterior samples (parameter space)")
    plt.plot(
        states.position.u0[:, 0],
        states.position.u0[:, 1],
        "o",
        alpha=0.5,
        markersize=4,
        label="Samples",
    )
    plt.plot(model_true.u0[0], model_true.u0[1], "P", label="Truth", markersize=8)
    plt.plot(model.u0[0], model.u0[1], "P", label="Initial guess", markersize=8)
    plt.legend()
    plt.show()


class Figure(eqx.Module):
    fig: Any
    ax: Any
    kwargs: dict

    def __init__(self, figsize, kwargs: dict):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.kwargs = kwargs

    def plot_solution(self, sol, *, label, marker="."):
        kwargs = self.kwargs[label]
        for d in [0, 1]:
            self.ax.plot(sol.t, sol.u[:, d], marker="None", label=label, **kwargs)
            self.ax.plot(sol.t[0], sol.u[0, d], marker=marker, **kwargs)
            self.ax.plot(sol.t[-1], sol.u[-1, d], marker=marker, **kwargs)

    def legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    def show(self):
        plt.show()


class Solver(eqx.Module):
    grid_plot: jax.Array
    grid_data: jax.Array

    def __init__(self, grid_plot, grid_data, ode_shape):
        # Set up the solver
        impl.select("isotropic", ode_shape=ode_shape)

        self.grid_plot = grid_plot
        self.grid_data = grid_data

    @property
    def solver(self):
        # Create a probabilistic solver
        ibm = ivpsolvers.prior_ibm(num_derivatives=2)
        ts0 = ivpsolvers.correction_ts0()
        strategy = ivpsolvers.strategy_filter(ibm, ts0)
        return ivpsolvers.solver(strategy)

    def solve_data(self, model):
        adaptive_solver = ivpsolve.adaptive(self.solver)
        tcoeffs = model.tcoeffs(num=2, t0=self.grid_data[0])
        output_scale = 10.0
        init = self.solver.initial_condition(tcoeffs, output_scale)
        return ivpsolve.solve_adaptive_save_at(
            model,
            init,
            save_at=self.grid_data,
            adaptive_solver=adaptive_solver,
            dt0=0.1,
        )

    def solve_fixed(self, model):
        tcoeffs = model.tcoeffs(t0=self.grid_data[0], num=2)
        output_scale = 10.0
        init = self.solver.initial_condition(tcoeffs, output_scale)
        sol = ivpsolve.solve_fixed_grid(
            model, init, grid=self.grid_data, solver=self.solver
        )

        return sol[-1]


class ODE(eqx.Module):
    """Parametrised ODE."""

    vf: Callable = eqx.field(static=True)
    _u0: jax.Array
    _args: jax.Array = eqx.field(static=True)
    _unravel: Callable = eqx.field(static=True)

    def __init__(self, *, u0, args, vf):
        self._u0 = jnp.sqrt(u0)
        self._args, self._unravel = jax.flatten_util.ravel_pytree(args)
        self.vf = vf

    @property
    def u0(self):
        return self._u0**2

    @property
    def args(self):
        return self._unravel(self._args)

    def tcoeffs(self, num, t0):
        tfun = functools.partial(taylor.odejet_padded_scan, num=num)
        return tfun(lambda y: self(y, t=t0), (self.u0,))

    @jax.jit
    def __call__(self, y, *, t):
        return self.vf(y, *self.args)

    def logpdf_prior(self):
        mean = jnp.zeros_like(self.u0)
        cov = jnp.eye(len(self.u0)) * 30  # fairly uninformed prior

        pdf = jax.scipy.stats.multivariate_normal.logpdf
        return pdf(self.u0, mean=mean, cov=cov)


def model_ivp():
    # IVP examples in JAX
    if not backend.has_been_selected:
        backend.select("jax")

    vf, u0, (t0, t1), params = ivps.lotka_volterra()
    return ODE(vf=vf, u0=u0, args=params), (t0, t1)


def log_posterior_fun(data, solver, obs_stdev=0.1):
    @jax.jit
    def logpost(model):
        """Evaluate the logposterior-function of the data."""
        y_T = solver.solve_fixed(model)
        lml = stats.log_marginal_likelihood_terminal_values
        logpdf_data = lml(
            data.u[-1], standard_deviation=obs_stdev, posterior=y_T.posterior
        )

        logpdf_prior = model.logpdf_prior()
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
