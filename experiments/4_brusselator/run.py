import matplotlib.pyplot as plt

import jax

from odecheckpts import ivps, ivpsolvers

import jax.numpy as jnp
from odecheckpts import exp_util


def main():
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)
    plt.rcParams.update(exp_util.plot_params())

    # Simulate once to get plotting code
    N = 50
    vf, u0, tspan, params = ivps.brusselator(N=N)
    solve = ivpsolvers.asolve_scipy("LSODA", vf, tspan, atol=1e-13, rtol=1e-13)
    ts, ys = solve(u0, params)

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(u0[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)

    plt.subplots(figsize=(3.25, 2.5), dpi=200)
    plt.contourf(Xs, Ts, Us)
    plt.xlabel("Space $x$")
    plt.ylabel("Time $t$")
    plt.title(f"Brusselator: $N={len(ts):,}$ adaptive steps")
    plt.show()

    #
    # fig = plt.figure(figsize=(3.25, 2.5))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(Xs, Ts, Us)
    # ax.set_xlabel("Space $X$")
    # ax.set_ylabel("Time $t$")
    # ax.set_zlabel("Solution $u$")

    plt.show()


if __name__ == "__main__":
    main()
