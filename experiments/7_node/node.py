"""Fit the stiffness-constant of VdP with adaptive steps."""

# next: make things scale to stiff equations. Ie, choose the setup from fig. 1 and learn mu
# 1. plot the loss landscape to find a reasonable setup (data, tolerances, solver, etc.)
# 2. prove that fixed steps ain't gonna work
# idea -1): plot the loss landscape
# idea 0): see whether fixed steps would've worked. Or try adaptive with crazy small tolerances
# idea 1): more data
# idea 2): different parametrisation of VdP.
#  E.g. Lienhard transform: https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
# idea 3): Add forcing term to have 2 parameters which makes learning more interesting
# idea 4): ???

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
from probdiffeq import ivpsolve, ivpsolvers, stats, taylor
from probdiffeq.backend import control_flow as cfl
from probdiffeq.impl import impl
import matplotlib.pyplot as plt


class VanDerPol(eqx.Module):
    _mu: jax.Array

    def __init__(self, mu):
        self._mu = mu

    @property
    def mu(self):
        return self._mu

    def __call__(self, y, ydot, *, t):  # noqa: ARG001
        return self.mu * (ydot * (1 - y**2) - y)


def main():
    jax.config.update("jax_enable_x64", True)
    impl.select("dense", ode_shape=(1,))

    # Set up the problem
    mu_true = 10**3
    vdp = VanDerPol(mu=mu_true)
    u0s = jnp.asarray([[2.0], [0.0]])
    t0, t1 = 0.0, 3.15  # half a time-domain; if not, the loss gets flat
    save_at = jnp.linspace(t0, t1, num=50)
    truth = solve(vdp, u0s, save_at=save_at)
    loss = log_likelihood(save_at=save_at, u=truth.u)

    # Use Equinox's bounded while loop for differentiability
    loop = functools.partial(eqx.internal.while_loop, kind="bounded", max_steps=1_000)
    with cfl.context_overwrite_while_loop(loop):
        # Plot the loss landscape to verify the setup
        # Both zoomed and not-zoomed
        mu_init = 10**1.5
        mus_zoomed_in = jnp.linspace(mu_init * 0.999, mu_init * 1.001, num=100)
        mus_zoomed_out = jnp.linspace(mu_init / 4, mu_true * 2, num=100)
        for mus in [mus_zoomed_in, mus_zoomed_out]:
            vdps = jax.vmap(VanDerPol)(mus)
            losses = jax.jit(jax.vmap(loss, in_axes=(0, None)))(vdps, u0s)
            plt.plot(vdps._mu, losses, ".")
            plt.axvline(mu_init)
            # plt.axvline(mu_true)
            plt.show()

        # Set up the optimizer
        loss = jax.jit(jax.value_and_grad(loss))
        optimizer = optax.adam(1e-3)

        # Initialise the optimizer
        vdp = VanDerPol(mu=mu_init)
        opt_state = optimizer.init(vdp)
        val, grads = loss(vdp, u0s)

        # Run the training loop
        progressbar = tqdm.tqdm(range(1000))
        label = f"loss: {val:.2e}, mu={vdp.mu:.3f}, grad={grads.mu}"
        progressbar.set_description(label)
        for _ in progressbar:
            val, grads = loss(vdp, u0s)
            updates, opt_state = optimizer.update(grads, opt_state)
            vdp = optax.apply_updates(vdp, updates)

            label = f"loss: {val:.2e}, mu={vdp.mu:.3f}, grad={grads.mu}"
            progressbar.set_description(label)


def log_likelihood(save_at, u):
    def loss(ode, u0s):
        solution = solve(ode, u0s, save_at=save_at)
        eps = jnp.sqrt(jnp.sqrt(jnp.finfo(u).eps))
        std = eps * jnp.ones_like(save_at)
        lml = stats.log_marginal_likelihood(
            u, standard_deviation=std, posterior=solution.posterior
        )
        return -lml

    return loss


def solve(ode, u0s, save_at):
    # any high-accuracy solution works. why? loss-landscape (zoom in!)
    num = 5
    tol = 1e-10

    # Set up the solver
    ibm = ivpsolvers.prior_ibm(num_derivatives=num)
    ts0 = ivpsolvers.correction_ts1(ode_order=2)
    strategy = ivpsolvers.strategy_fixedpoint(ibm, ts0)
    solver = ivpsolvers.solver(strategy)  # why? plot loss-landscape

    # Set up the initial condition
    t0 = save_at[0]
    vf = functools.partial(ode, t=t0)
    init = [u0s[0], u0s[1]]
    tcoeffs = taylor.odejet_padded_scan(vf, init, num=num - 1)
    output_scale = 1e10  # or any other value with the same shape
    init = solver.initial_condition(tcoeffs, output_scale)

    # Compute a baseline solution
    ctrl = ivpsolve.control_proportional_integral()
    adaptive_solver = ivpsolve.adaptive(solver, atol=tol, rtol=tol, control=ctrl)
    return ivpsolve.solve_adaptive_save_at(
        ode, init, save_at=save_at, dt0=0.1, adaptive_solver=adaptive_solver
    )


if __name__ == "__main__":
    main()
