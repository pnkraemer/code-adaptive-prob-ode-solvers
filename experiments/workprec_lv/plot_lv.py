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
    if "ProbDiffEq" in label:
        return {"linestyle": "solid"}
    if "SciPy" in label:
        return {"linestyle": "dashed"}
    if "iffrax" in label:
        return {"linestyle": "dotted"}
    msg = f"Label {label} unknown."
    raise ValueError(msg)


def plot_results(axis, results):
    """Plot the results."""
    axis.set_title("Work-precision")
    for idx, (label, wp) in enumerate(results.items()):
        style = choose_style(label)

        precision = wp["precision"]
        work_mean, work_std = (wp["work_mean"], wp["work_std"])
        axis.loglog(precision, work_mean, label=label, color=f"C{idx}", **style)

        range_lower, range_upper = work_mean - work_std, work_mean + work_std
        axis.fill_between(precision, range_lower, range_upper, alpha=0.3, color=f"C{idx}", **style)

    axis.set_xlabel("Precision [relative RMSE]")
    axis.set_ylabel("Work [wall time, s]")
    axis.grid(linestyle="dotted")
    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="small")
    # axis.set_ylim((1e-5, 1e1))
    return axis


def plot_solution(axis, ts, ys, yscale="linear"):
    """Plot the IVP solution."""
    axis.set_title("ODE model: Lotka-Volterra")

    axis.plot(ts, ys[:, 0], linestyle="solid", marker="None", label="Predators")
    axis.plot(ts, ys[:, 1], linestyle="dashed", marker="None", label="Prey")

    # axis.set_ylim((-1, 27))
    axis.set_xlim((jnp.amin(ts), jnp.amax(ts)))
    axis.legend(facecolor="ghostwhite", edgecolor="black", fontsize="small")

    axis.set_xlabel("Time $t$")
    axis.set_ylabel("Solution $y$")
    axis.set_yscale(yscale)
    return axis


layout = [
    ["benchmark", "solution"],
    ["benchmark", "solution"],
]
fig, axes = plt.subplot_mosaic(layout, figsize=(8, 3), constrained_layout=True, dpi=300)


results = load_results()
ts, ys = load_solution()

_ = plot_results(axes["benchmark"], results)
_ = plot_solution(axes["solution"], ts, ys)

plt.savefig(os.path.dirname(__file__) + "/figure.pdf")
