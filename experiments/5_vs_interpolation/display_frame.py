import os
import jax.numpy as jnp
import pandas as pd

filename = os.path.dirname(__file__) + "/data"
results_dict = jnp.load(f"{filename}_results.npy", allow_pickle=True).item()
solution = jnp.load(f"{filename}_solution.npy", allow_pickle=True)
#
#
# plt.plot(*solution.T)
# plt.show()
#

print(results_dict)
print()

results = pd.DataFrame.from_dict(results_dict)
print(results.T)
print()
print(results.T.to_latex())
