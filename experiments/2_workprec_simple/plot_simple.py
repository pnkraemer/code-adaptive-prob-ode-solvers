import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from odecheckpts import exp_util
from tueplots import bundles

PLOT_PARAMS = exp_util.plot_params()
STYLE = exp_util.style_simple()


def main():
    """Load and plot the results."""
    plt.rcParams.update(bundles.probnum2025(column="full"))
    plt.rcParams.update(PLOT_PARAMS)

    results = load_results()
    ts, ys = load_solution()
    checkpoints = load_checkpoints()

    # Create a figure
    layout = [
        ["solution", "solution", "solution", "solution", "solution"],
        ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
        ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
        ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
    ]
    fig, axes = plt.subplot_mosaic(layout, dpi=200)

    # Plot each set of results
    _ = plot_solution(axes["solution"], ts, ys, checkpoints)
    _ = plot_results(axes["benchmark"], results)
    _ = plot_results_error_vs_length(axes["error_vs_length"], results)

    for axis in [axes["benchmark"], axes["error_vs_length"]]:
        axis.set_xticks([1e-0, 1e-2, 1e-4, 1e-6, 1e-8])

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


def load_checkpoints():
    return jnp.load(os.path.dirname(__file__) + "/data_checkpoints.npy")


def plot_solution(axis, ts, ys, checkpoints, yscale="linear"):
    axis.set_title("a) Rigid body problem: three-dimensional solution")
    for linestyle, y in zip(["solid", "dashed", "dotted"], ys.T):
        axis.plot(ts, y, linestyle=linestyle, color="black")

    # for t in checkpoints:
    #     axis.axvline(t, linestyle="dotted", color="black")

    axis.set_xlim((-0.1, 50.1))
    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution")
    axis.set_yscale(yscale)
    return axis


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("b) Work versus precision")
    for label, wp in results.items():
        precision = wp["precision"]
        work = wp["work_min"]

        axis.loglog(
            precision,
            work,
            marker=STYLE.marker(label),
            linestyle=STYLE.linestyle(label),
            label=STYLE.label(label),
            color=STYLE.color(label),
            alpha=STYLE.alpha_line(label),
        )

    axis.set_ylim((1.1e-5, 1e1))
    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid()
    axis.legend(loc="lower center", ncols=3, fontsize="x-small")
    return axis


def plot_results_error_vs_length(axis, results):
    axis.set_title("c) Memory requirements")
    for label, wp in results.items():
        # Only plot the probabilistic solvers because
        # Runge-Kutta methods' checkpointing is well understood
        # if "TS" in label:
        if True:
            axis.loglog(
                wp["precision"],
                wp["length_of_longest_vector"],
                label=STYLE.label(label),
                marker=STYLE.marker(label),
                color=STYLE.color(label),
                linestyle=STYLE.linestyle(label),
                alpha=STYLE.alpha_line(label),
                zorder=STYLE.zorder(label),
            )

    axis.set_yticks((1e0, 1e1, 1e2, 1e3, 1e4))
    axis.legend(ncols=1, fontsize="x-small")

    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Length of the solution vector")
    axis.grid()
    return axis


if __name__ == "__main__":
    main()
