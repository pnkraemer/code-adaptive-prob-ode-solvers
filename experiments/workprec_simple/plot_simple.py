import os
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


def choose_style(label):
    """Choose a plotting style for a given algorithm."""
    if "int" in label:
        style = {"linestyle": "dashed"}
    elif "TS" in label:
        style = {"linestyle": "solid"}
    elif "SciPy" in label:
        style = {"linestyle": "dashed"}
    elif "()" in label:
        style = {"linestyle": "dotted"}

    if "2" in label or "eun" in label:
        style["color"] = "C0"
    if "4" in label or "opri" in label:
        style["color"] = "C1"

    return style


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("Work versus precision", fontsize="medium")
    for label, wp in results.items():
        if "interp" in label:
            label = label.replace(") (interp.", ", interp")
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])

        axis.loglog(precision, work_mean, marker=".", label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_ylim((1.1e-5, 1e1))
    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid(linestyle="dotted")
    axis.legend(
        ncols=3,
        handlelength=2.6,
        loc="lower center",
        facecolor="ghostwhite",
        edgecolor="black",
        fontsize="x-small",
    )
    return axis


def plot_results_error_vs_length(axis, results):
    """Plot the results."""
    axis.set_title("Memory requirements", fontsize="medium")
    for label, wp in results.items():
        if "TS" in label:
            style = choose_style(label)

            precision = wp["precision"]
            length = wp["length_of_longest_vector"]

            axis.loglog(precision, length, marker=".", label=label, alpha=0.9, **style)

    axis.set_ylim((0.9, 1e4))
    axis.legend(
        handlelength=3.0,
        ncols=2,
        loc="center left",
        facecolor="ghostwhite",
        edgecolor="black",
        fontsize="x-small",
    )
    axis.set_xlabel("Time-series error (RMSE)")
    axis.set_ylabel("Length: solution vector")
    axis.grid(linestyle="dotted")
    return axis


def plot_solution(axis, ts, ys, timeseries, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("Rigid body problem", fontsize="medium")
    for colour, y in zip(["black", "darkgreen", "darkred"], ys.T):
        axis.plot(ts, y, linestyle="solid", color=colour, alpha=0.8)

    for t in timeseries:
        axis.axvline(t, linestyle="dotted", color="black")

    axis.set_xlim((jnp.amin(ts) - 0.5, jnp.amax(ts) + 0.5))
    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


layout = [
    ["solution", "solution", "solution", "solution"],
    ["benchmark", "benchmark", "error_vs_length", "error_vs_length"],
    ["benchmark", "benchmark", "error_vs_length", "error_vs_length"],
    ["benchmark", "benchmark", "error_vs_length", "error_vs_length"],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 4), constrained_layout=True, dpi=100)


results = load_results()
ts, ys = load_solution()
timeseries = load_timeseries()

_ = plot_results(axes["benchmark"], results)
_ = plot_results_error_vs_length(axes["error_vs_length"], results)
_ = plot_solution(axes["solution"], ts, ys, timeseries)

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
plt.show()
