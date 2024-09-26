import os
import jax

import jax.numpy as jnp
import matplotlib.pyplot as plt

import dataclasses

from odecheckpts import exp_util

raise RuntimeError(
    "in the data generation, prefix the data files with data_* instead of vdp_*"
)


@dataclasses.dataclass
class Data:
    runtime: float
    grid: jax.Array

    @property
    def steps(self):
        return jnp.diff(self.grid)

    @property
    def num_steps(self):
        return len(self.steps)

    @classmethod
    def load(cls, filename, key: str):
        xs = jnp.load(filename.replace(".py", f"_grid_{key}.npy"))
        t = jnp.load(filename.replace(".py", f"_runtime_{key}.npy"))
        return cls(t, xs)


def main():
    # Otherwise, the small step-sizes are clipped during "jnp.diff".
    jax.config.update("jax_enable_x64", True)

    # Load the data
    filename = str(__file__)
    filename = filename.replace("_plot", "")
    baseline_grid = jnp.load(filename.replace(".py", "_baseline_grid.npy"))
    baseline_solution = jnp.load(filename.replace(".py", "_baseline_solution.npy"))
    adaptive = Data.load(filename, "adaptive")
    fixed_accurate = Data.load(filename, "fixed_accurate")
    fixed_inaccurate = Data.load(filename, "fixed_inaccurate")

    # Make all plots look similar
    plt.rcParams.update(exp_util.plot_params())
    fig, ax = plt.subplots(dpi=100, figsize=(3.25, 2.0))  # 3.25: half-page

    # Plot the three curves
    label = f"$N={adaptive.num_steps:,}$ adaptive steps take {adaptive.runtime:.2f}s"
    ax.semilogy(adaptive.grid[:-1], adaptive.steps, linestyle="solid", label=label)
    label = rf"$N={fixed_inaccurate.num_steps:,}$ fixed steps yield NaNs"
    ax.semilogy(
        fixed_inaccurate.grid[:-1],
        fixed_inaccurate.steps,
        linestyle="dashed",
        label=label,
    )
    label = f"$N={fixed_accurate.num_steps:,}$ fixed steps take {fixed_accurate.runtime:.2f}s"
    ax.semilogy(
        fixed_accurate.grid[:-1], fixed_accurate.steps, linestyle="dotted", label=label
    )

    # Style legend, axes, etc.
    ax.legend(loc="upper left", fontsize="xx-small")
    ax.set_ylabel(r"Step-size $\Delta t$")
    ax.set_ylim((4e-6, 5e0))
    ax.set_xlabel(r"ODE domain (time $t$)")
    ax.set_title("a) Step-size evolution during the simulation")

    # Insert the ODE solution
    axin1 = ax.inset_axes([0.8, 0.75, 0.2, 0.25])
    axin1.set_title("b) VdP sol.", fontsize="x-small", x=0.5, y=0.5)
    axin1.set_ylim((-4, 6))
    axin1.set_yticks((-2, 2))
    axin1.plot(baseline_grid, baseline_solution, color="black")

    # Make the xticks match between both plots
    for a in [ax, axin1]:
        a.set_xlim((-0.1, 6.4))
        a.set_xticks((0, 2, 4, 6))

    # Save the plot to a file
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
