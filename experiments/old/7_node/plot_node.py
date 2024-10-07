import jax.numpy as jnp
import os
import matplotlib.pyplot as plt

rng = 1
s = 0.0001
filename = os.path.dirname(__file__) + "/data_losses"
losses = jnp.load(f"{filename}_rng_{rng}_std_{s}.npy", allow_pickle=True).item()

filename = os.path.dirname(__file__) + "/data_plots"
plots = jnp.load(f"{filename}_rng_{rng}_std_{s}.npy", allow_pickle=True).item()
print(losses)


plt.plot(plots["ins"], plots["outs"], "o", color="black", label="Data")
plt.plot(plots["ts"], plots["truth"], color="gray", label="Truth")
plt.plot(plots["ts"], plots["before"], color="gray", linestyle="dotted", label="Before")
plt.plot(plots["ts"], plots["rk"], color="C0", label="Runge-Kutta")
plt.plot(plots["ts"], plots["pn"], color="C1", label="Prob.-Num.")
plt.legend()
plt.show()
