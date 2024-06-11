import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from odecheckpts import exp_util

#
# rc = {
#     "font.size": 10,
#     "axes.titlesize": "medium",
#     "legend.fontsize": "small",
#     "legend.frameon": True,
#     "legend.facecolor": "white",
#     "legend.edgecolor": "black",
#     "legend.fancybox": False,
#     "lines.linewidth": 0.75,
#     "axes.linewidth": 0.5,
#     "markers.fillstyle": "none",
# }
#

plt.rcParams.update(exp_util.plot_params())
style = exp_util.style_rigid_body()


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
    """Plot the results."""
    axis.set_title("Work versus precision")
    for label, wp in results.items():
        if "interp" in label:
            label = label.replace(") (interp.", ", interp.")

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])

        axis.loglog(
            precision,
            work_mean,
            marker=style.marker(label),
            linestyle=style.linestyle(label),
            label=style.label(label),
            color=style.color(label),
        )

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(
            precision,
            range_lower,
            range_upper,
            alpha=style.alpha_fill_between(label),
            color=style.color(label),
        )

    axis.set_ylim((1.1e-5, 1e1))
    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid()
    legend = axis.legend(
        ncols=3,
        # handlelength=2.6,
        # loc="lower center",
    )
    legend.get_frame().set_linewidth(0.5)
    return axis


def plot_results_error_vs_length(axis, results):
    """Plot the results."""
    axis.set_title("Memory requirements")
    for label, wp in results.items():
        if "TS" in label:
            precision = wp["precision"]
            length = wp["length_of_longest_vector"]

            axis.loglog(
                precision,
                length,
                label=style.label(label),
                marker=style.marker(label),
                color=style.color(label),
                linestyle=style.linestyle(label),
                alpha=0.9,
            )

    axis.set_ylim((0.9, 1e4))
    legend = axis.legend(
        # handlelength=3.0,
        ncols=1,
        # loc="center left",
    )
    legend.get_frame().set_linewidth(0.5)

    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Length of solution vector")
    axis.grid()
    return axis


def plot_solution(axis, ts, ys, timeseries, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("Rigid body problem")
    for colour, y in zip(["black", "darkslategray", "midnightblue"], ys.T):
        axis.plot(ts, y, linewidth=0.75, linestyle="solid", color=colour, alpha=0.8)

    for t in timeseries:
        axis.axvline(t, linestyle="dotted", color="black")

    axis.set_xlim((jnp.amin(ts) - 0.5, jnp.amax(ts) + 0.5))
    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


layout = [
    ["solution", "solution", "solution", "solution", "solution"],
    ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
    ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
    ["benchmark", "benchmark", "benchmark", "error_vs_length", "error_vs_length"],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 5), constrained_layout=True, dpi=100)


results = load_results()
ts, ys = load_solution()
timeseries = load_timeseries()

_ = plot_results(axes["benchmark"], results)
_ = plot_results_error_vs_length(axes["error_vs_length"], results)
_ = plot_solution(axes["solution"], ts, ys, timeseries)


axes["solution"].set_title("a.", loc="left", fontweight="bold")
axes["benchmark"].set_title("b.", loc="left", fontweight="bold")
axes["error_vs_length"].set_title("c.", loc="left", fontweight="bold")

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
plt.show()
