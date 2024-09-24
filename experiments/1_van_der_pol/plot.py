import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tueplots import axes
import dataclasses


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
    plt.rcParams.update(axes.legend())

    filename = str(__file__)
    filename = filename.replace("plot", "vdp")

    baseline_grid = jnp.load(filename.replace(".py", "_baseline_grid.npy"))
    baseline_solution = jnp.load(filename.replace(".py", "_baseline_solution.npy"))

    adaptive = Data.load(filename, "adaptive")
    fixed_accurate = Data.load(filename, "fixed_accurate")
    fixed_inaccurate = Data.load(filename, "fixed_inaccurate")

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.semilogy(
        adaptive.grid[:-1],
        adaptive.steps,
        linestyle="solid",
        marker="None",
        markersize=1,
        color="C0",
        label=f"$N$={adaptive.num_steps:,} adaptive steps take {adaptive.runtime:.1f}s",
    )
    ax.semilogy(
        fixed_inaccurate.grid[:-1],
        fixed_inaccurate.steps,
        linestyle="dotted",
        marker="None",
        color="gray",
        label=rf"$N$={fixed_inaccurate.num_steps:,} fixed, evenly spaced steps yield NaNs",
    )
    ax.semilogy(
        fixed_accurate.grid[:-1],
        fixed_accurate.steps,
        linestyle="dashed",
        marker="None",
        color="C1",
        label=f"$N$={fixed_accurate.num_steps:,} fixed, evenly spaced steps take {fixed_accurate.runtime:.1f}s",
    )
    ax.legend(loc="upper left", edgecolor="white", handlelength=1.1, fontsize="small")
    ax.set_xlabel(r"ODE domain (time $t$)")
    ax.set_ylabel(r"Step-size $\Delta t$")
    ax.set_ylim((4e-6, 5e0))
    ax.set_xlim((-0.1, 6.4))
    ax.set_xticks((0, 1, 2, 3, 4, 5, 6))

    axin1 = ax.inset_axes([0.8, 0.75, 0.2, 0.25])
    axin1.set_title("VdP solution", fontsize="small", x=0.5, y=0.65)
    axin1.set_xlim((-0.1, 6.5))
    axin1.set_xticks((0.0, 3.0, 6.0))
    axin1.set_yticks((-2.0, 2.0))
    axin1.set_ylim((-3, 5))
    axin1.set_yticks((-2, 2))
    axin1.tick_params(axis="x", direction="in")
    axin1.tick_params(axis="y", direction="in")
    axin1.plot(baseline_grid, baseline_solution, color="black", linewidth=0.75)

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")

    filename = str(__file__)
    filename = filename.replace("experiments/", "figures/")
    filename = filename.replace(".py", ".pdf")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
