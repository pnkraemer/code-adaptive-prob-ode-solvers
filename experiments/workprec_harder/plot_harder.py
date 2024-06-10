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


def plot_solution(axis, ts, ys):
    """Plot the IVP solution."""
    axis.set_title("Henon-Heiles problem", fontsize="medium")
    for colour, y in zip(["black", "darkgreen", "darkred"], ys.T):
        axis.plot(ts, y, linewidth=0.75, linestyle="solid", color=colour, alpha=0.8)
    return axis


layout = [["solution", "benchmark"]]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 3), constrained_layout=True, dpi=100)


results = load_results()
ts, ys = load_solution()
timeseries = load_timeseries()

_ = plot_results(axes["benchmark"], results)
_ = plot_solution(axes["solution"], ts, ys)


axes["solution"].set_title("a.", loc="left", fontsize="medium", fontweight="bold")
axes["benchmark"].set_title("b.", loc="left", fontsize="medium", fontweight="bold")

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
plt.show()
