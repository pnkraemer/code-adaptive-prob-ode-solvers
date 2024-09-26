import matplotlib.pyplot as plt

import jax

from odecheckpts import ivps, ivpsolvers


def main():
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to get plotting code
    vf, u0, tspan, params = ivps.brusselator()
    solve = ivpsolvers.asolve_scipy("LSODA", vf, tspan, atol=1e-13, rtol=1e-13)
    ts, ys = solve(u0, params)

    plt.plot(ts, ys)
    plt.show()


if __name__ == "__main__":
    main()
