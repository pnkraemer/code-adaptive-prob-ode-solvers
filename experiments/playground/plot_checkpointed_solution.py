"""Plot the probabilistic ODE solution."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from odecheckpts import ivps, ivpsolvers

vf, u0, time_span, args = ivps.rigid_body()

dt0 = 0.1
atol, rtol = 1e-4, 1e-4
save_at = jnp.linspace(*time_span, num=100)
u0_like = u0  # infer shapes etc.


solve = ivpsolvers.solve_and_save_at(
    vf, u0_like, save_at, dt0=dt0, atol=atol, rtol=rtol
)
solve = jax.jit(solve)
u, solution_full = solve(u0, args)

plt.plot(save_at, u)
plt.show()
