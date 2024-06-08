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
    if "TS" in label:
        style = {"linestyle": "solid"}
    if "SciPy" in label:
        style = {"linestyle": "dashed"}
    if "()" in label:
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
    # axis.set_title("Work-precision")
    for label, wp in results.items():
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_xlabel("Time-series error (requires 'solve_and_save_at')")
    axis.set_ylabel("Wall time (s)")
    axis.grid(linestyle="dotted")
    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="small")
    # axis.set_ylim((1e-5, 1e1))
    return axis


def plot_results_past(axis, results):
    """Plot the results."""
    # axis.set_title("Work-precision")
    for label, wp in results.items():
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, **style)

    axis.set_xlabel("Terminal-value error (prior work)")
    axis.set_ylabel("Wall time (s)")
    axis.grid(linestyle="dotted")
    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="small")
    # axis.set_ylim((1e-5, 1e1))
    return axis


def plot_solution(axis, ts, ys, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("Rigid body problem")
    axis.plot(ts, ys, color="darkslategray")
    #
    # axis.plot(ts, ys[:, 0], linestyle="solid", marker="None", label="Predators")
    # axis.plot(ts, ys[:, 1], linestyle="dashed", marker="None", label="Prey")

    # axis.set_ylim((-1, 27))
    axis.set_xlim((jnp.amin(ts), jnp.amax(ts)))
    # axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="small")

    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


layout = [
    ["solution", "solution", "solution", "solution"],
    ["benchmark_past", "benchmark_past", "benchmark", "benchmark"],
    ["benchmark_past", "benchmark_past", "benchmark", "benchmark"],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 5), constrained_layout=True, dpi=300)


results = load_results()
ts, ys = load_solution()

_ = plot_results_past(axes["benchmark_past"], results)
_ = plot_results(axes["benchmark"], results)
_ = plot_solution(axes["solution"], ts, ys)

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
