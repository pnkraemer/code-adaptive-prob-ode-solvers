import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tueplots import axes, bundles
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
    jax.config.update("jax_enable_x64", True)
    plt.rcParams.update(axes.tick_direction(x="in", y="in"))
    plt.rcParams.update(axes.lines(tick_minor_base_ratio=0.0))
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(
        bundles.aistats2025(column="half", ncols=1, nrows=1, family="serif")
    )

    filename = str(__file__)
    filename = filename.replace("_plot", "")

    baseline_grid = jnp.load(filename.replace(".py", "_baseline_grid.npy"))
    baseline_solution = jnp.load(filename.replace(".py", "_baseline_solution.npy"))

    adaptive = Data.load(filename, "adaptive")
    fixed_accurate = Data.load(filename, "fixed_accurate")
    fixed_inaccurate = Data.load(filename, "fixed_inaccurate")

    fig, ax = plt.subplots(dpi=200)
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
    ax.legend(loc="upper left", fontsize="xx-small")
    ax.set_ylabel(r"Step-size $\Delta t$")
    ax.set_ylim((4e-6, 5e0))
    ax.set_xlabel(r"ODE domain (which is the time $t$)")

    axin1 = ax.inset_axes([0.8, 0.75, 0.2, 0.25])
    axin1.set_title("VdP sol.", fontsize="x-small", x=0.5, y=0.57)
    axin1.set_ylim((-4, 6))
    axin1.set_yticks((-2, 2))
    axin1.plot(baseline_grid, baseline_solution, color="black", linewidth=0.75)

    for a in [ax, axin1]:
        a.set_xlim((-0.1, 6.4))
        a.set_xticks((0, 2, 4, 6))

    filename = str(__file__)
    filename = filename.replace("experiments/", "figures/")
    filename = filename.replace(".py", ".pdf")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    main()
