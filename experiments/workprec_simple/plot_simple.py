import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from odecheckpts import exp_util

PLOT_PARAMS = exp_util.plot_params()
STYLE = exp_util.style_simple()


def main():
    """Load and plot the results."""
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
    fig, axes = plt.subplot_mosaic(layout, figsize=(8, 4.5), dpi=100)

    # Plot each set of results
    _ = plot_results(axes["benchmark"], results)
    _ = plot_results_error_vs_length(axes["error_vs_length"], results)
    _ = plot_solution(axes["solution"], ts, ys, checkpoints)

    # Add subplot-labels so the figures can be referenced in the text
    axes["solution"].set_title(r"\bf a.", loc="left", fontweight="bold")
    axes["benchmark"].set_title(r"\bf b.", loc="left", fontweight="bold")
    axes["error_vs_length"].set_title(r"\bf c.", loc="left", fontweight="bold")

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


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("Work versus precision")
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
    axis.legend(loc="lower center", ncols=3)
    return axis


def plot_results_error_vs_length(axis, results):
    axis.set_title("Memory requirements")
    for label, wp in results.items():
        # Only plot the probabilistic solvers because
        # Runge-Kutta methods' checkpointing is well understood
        if "TS" in label:
            axis.loglog(
                wp["precision"],
                wp["length_of_longest_vector"],
                label=STYLE.label(label),
                marker=STYLE.marker(label),
                color=STYLE.color(label),
                linestyle=STYLE.linestyle(label),
                alpha=STYLE.alpha_line(label),
            )

    axis.set_ylim((0.9, 1e4))
    axis.legend(ncols=1)

    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Length of the solution vector")
    axis.grid()
    return axis


def plot_solution(axis, ts, ys, checkpoints, yscale="linear"):
    axis.set_title("Rigid body problem")
    for linestyle, y in zip(["solid", "dashed", "dotted"], ys.T):
        axis.plot(ts, y, linestyle=linestyle, color="black")

    for t in checkpoints:
        axis.axvline(t, linestyle="dotted", color="black")

    axis.set_xlim((jnp.amin(ts) - 0.5, jnp.amax(ts) + 0.5))
    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


if __name__ == "__main__":
    main()
