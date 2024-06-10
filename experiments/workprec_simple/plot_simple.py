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


def choose_style(label):
    """Choose a plotting style for a given algorithm."""
    if "interp" in label:
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
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("Work vs precision")
    for label, wp in results.items():
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_xlabel("Time-series error (avg. rel. RMSE)")
    axis.set_ylabel("Wall time (s)")
    axis.grid(linestyle="dotted")
    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="x-small")
    return axis


def plot_results_error_vs_length(axis, results):
    """Plot the results."""
    axis.set_title("Memory requirements")
    for label, wp in results.items():
        if "TS" in label:
            style = choose_style(label)

            precision = wp["precision"]
            length = wp["length_of_longest_vector"]
            axis.semilogx(precision, length, label=label, **style)

    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="x-small")
    axis.set_xlabel("Time-series error (avg. rel. RMSE)")
    axis.set_ylabel("Length of the solution vector")
    axis.grid(linestyle="dotted")
    return axis


def plot_solution(axis, ts, ys, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("Rigid body problem")
    axis.plot(ts, ys, color="darkslategray")
    axis.set_xlim((jnp.amin(ts), jnp.amax(ts)))
    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


layout = [
    ["solution", "solution", "solution", "solution", "solution", "solution"],
    [
        "benchmark",
        "benchmark",
        "benchmark",
        "error_vs_length",
        "error_vs_length",
        "error_vs_length",
    ],
    [
        "benchmark",
        "benchmark",
        "benchmark",
        "error_vs_length",
        "error_vs_length",
        "error_vs_length",
    ],
    [
        "benchmark",
        "benchmark",
        "benchmark",
        "error_vs_length",
        "error_vs_length",
        "error_vs_length",
    ],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 4), constrained_layout=True, dpi=300)


results = load_results()
ts, ys = load_solution()

_ = plot_results(axes["benchmark"], results)
_ = plot_results_error_vs_length(axes["error_vs_length"], results)
_ = plot_solution(axes["solution"], ts, ys)

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
