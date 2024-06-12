import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from odecheckpts import exp_util

TODO = "\nTODO: Rename all saved data into data_* so the directory is clean.\n"
PLOT_PARAMS = exp_util.plot_params()
STYLE = exp_util.style_harder()


def main():
    plt.rcParams.update(PLOT_PARAMS)
    layout = [["solution", "benchmark", "benchmark"]]
    fig, axes = plt.subplot_mosaic(layout, figsize=(8, 3), dpi=200)

    results = load_results()
    _ts, ys = load_solution()

    _ = plot_results(axes["benchmark"], results)
    _ = plot_solution(axes["solution"], ys)

    axes["solution"].set_title(r"\bf a.", loc="left", fontweight="bold")
    axes["benchmark"].set_title(r"\bf b.", loc="left", fontweight="bold")

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
    axis.set_title("Work versus precision")
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
        )

    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid()
    axis.legend(ncols=2)
    return axis


def plot_solution(axis, ys):
    axis.set_title("Pleiades problem")

    axis.plot(ys[:, :7], ys[:, 7:14], color="black")
    axis.plot(ys[[0], :7], ys[[0], 7:14], marker=".", color="black")
    axis.plot(
        ys[[-1], :7],
        ys[[-1], 7:14],
        marker="*",
        color="black",
    )

    axis.set_xlabel("$x$-coordinate")
    axis.set_ylabel("$y$-coordinate")
    return axis


if __name__ == "__main__":
    main()
