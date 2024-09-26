import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from odecheckpts import exp_util

TODO = "\nTODO: Rename all saved data into data_* so the directory is clean.\n"
PLOT_PARAMS = exp_util.plot_params()
STYLE = exp_util.style_harder()


def main():
    plt.rcParams.update(PLOT_PARAMS)
    fig, axes = plt.subplots(figsize=(3.25, 2.0), dpi=150)

    results = load_results()
    _ts, ys = load_solution()

    _ = plot_results(axes, results)

    axes_in = axes.inset_axes([0.75, 0.65, 0.25, 0.35])
    _ = plot_solution(axes_in, ys)

    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()


def load_results():
    """Load the results from a file."""
    data = jnp.load(os.path.dirname(__file__) + "/data_results.npy", allow_pickle=True)
    return data[()]


def load_solution():
    """Load the solution-to-be-plotted from a file."""
    ts = jnp.load(os.path.dirname(__file__) + "/data_ts.npy")
    ys = jnp.load(os.path.dirname(__file__) + "/data_ys.npy")
    return ts, ys


def plot_results(axis, results):
    axis.set_title("a) Work versus precision")
    for label, wp in results.items():
        precision = wp["precision"]
        work = wp["work_min"]

        axis.loglog(
            precision,
            work,
            label=STYLE.label(label),
            color=STYLE.color(label),
            marker=STYLE.marker(label),
            linestyle=STYLE.linestyle(label),
            zorder=STYLE.zorder(label),
        )

    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid()
    axis.set_ylim((3e-4, 1e0))
    axis.legend(ncols=2, loc="lower left", fontsize="x-small")
    return axis


def plot_solution(axis, ys):
    axis.set_title("b) Pleiades sol.", fontsize="x-small", x=0.5, y=0.65)

    for i in range(7):
        axis.plot(ys[:, i], ys[:, 7 + i], color="black")
        axis.plot(ys[[0], i], ys[[0], 7 + i], marker=".", markersize=1, color="black")
        axis.plot(
            ys[[-1], i],
            ys[[-1], 7 + i],
            marker="*",
            color="black",
            markersize=3,
        )
    axis.set_xlim((-4, 4))
    axis.set_ylim((-7, 12))
    axis.set_xticks(())
    axis.set_yticks(())
    # axis.set_xlabel("$x$-coordinate")
    # axis.set_ylabel("$y$-coordinate")
    return axis


if __name__ == "__main__":
    main()
