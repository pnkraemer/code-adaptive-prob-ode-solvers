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

    plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
    plt.show()


def load_results():
    """Load the results from a file."""
    return jnp.load(os.path.dirname(__file__) + "/results.npy", allow_pickle=True)[()]


def load_solution():
    """Load the solution-to-be-plotted from a file."""
    ts = jnp.load(os.path.dirname(__file__) + "/plot_ts.npy")
    ys = jnp.load(os.path.dirname(__file__) + "/plot_ys.npy")
    return ts, ys


def load_timeseries():
    return jnp.load(os.path.dirname(__file__) + "/plot_timeseries.npy")


def plot_results(axis, results):
    axis.set_title("Work versus precision")
    for label, wp in results.items():
        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        range_lower, range_upper = work_mean - work_std, work_mean + work_std

        axis.loglog(
            precision,
            work_mean,
            label=STYLE.label(label),
            color=STYLE.color(label),
            marker=STYLE.marker(label),
            linestyle=STYLE.linestyle(label),
        )

        axis.fill_between(
            precision,
            range_lower,
            range_upper,
            color=STYLE.color(label),
            alpha=STYLE.alpha_fill_between(label),
        )

    # axis.set_ylim((1.1e-5, 1e1))
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
