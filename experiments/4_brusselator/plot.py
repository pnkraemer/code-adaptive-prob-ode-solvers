import matplotlib.pyplot as plt


import jax.numpy as jnp
from odecheckpts import exp_util


import os


def load(str) -> dict:
    return jnp.load(os.path.dirname(__file__) + f"/{str}.npy", allow_pickle=True).item()


if __name__ == "__main__":
    plt.rcParams.update(exp_util.plot_params())
    checkpoint = load("data_checkpoint")
    textbook = load("data_textbook")

    # Prepare the Figure
    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)

    # Choose one of the resolutions to be plotted
    idx = checkpoint["N"].index(64)
    num_steps = checkpoint["num_steps"][idx]
    ts = checkpoint["ts"][idx]
    ys = checkpoint["ys"][idx]

    # Plot it into the Brusselator figure
    msg = f"a) Brusselator: ${len(ts):,}$ target pts., ${int(jnp.amax(num_steps)):,}$ compute pts."
    ax["brusselator"].set_title(msg)
    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)
    ax["brusselator"].pcolormesh(Xs, Ts, Us)
    ax["brusselator"].set_xlabel("Space dimension")
    ax["brusselator"].set_ylabel("Time dimension $t$")
    ax["brusselator"].tick_params(which="both", direction="out")

    # Mark it with a vertical line in the complexity figure
    ax["complexity"].set_title("b) Memory consumption vs. problem size")
    N, mem = checkpoint["N"][idx], checkpoint["memory"][idx]
    ax["complexity"].axvline(N, color="black", linestyle="dotted")
    ax["complexity"].annotate(
        "Used for Figure a)",
        xy=(N, 3 * mem),
        rotation=90 * 3,
        color="black",
        fontsize="x-small",
    )

    # Plot the memory usage
    Ns, mems = textbook["N"], textbook["memory"]
    ax["complexity"].plot(
        Ns,
        mems,
        marker=".",
        label="Previous SoTA",
        color="C1",
    )
    Ns, mems = checkpoint["N"], checkpoint["memory"]
    ax["complexity"].plot(
        Ns,
        mems,
        marker=".",
        label="Our code",
        color="C0",
    )

    # Circle some of the memory markers for the checkpointer
    stride = 1
    Ns = checkpoint["N"][::stride]
    runs = checkpoint["runtime"][::stride]
    mems = checkpoint["memory"][::stride]
    for n, t, m in zip(Ns, runs, mems):
        ax["complexity"].plot(n, m, "s", markersize=10, color="C0")
        ax["complexity"].annotate(
            f"{t:.1f}s",
            xy=(n, m),
            xytext=(n * 1.1, 0.35 * m),
            color="C0",
            fontsize="x-small",
        )

    # Circle some of the memory markers for the checkpointer
    Ns = textbook["N"][::stride]
    runs = textbook["runtime"][::stride]
    mems = textbook["memory"][::stride]
    for n, t, m in zip(Ns, runs, mems):
        ax["complexity"].plot(n, m, "s", markersize=10, color="C1")
        # if n % 3 == 0:
        ax["complexity"].annotate(
            f"{t:.1f}s", xy=(n, m), xytext=(n, 0.35 * m), color="C1", fontsize="x-small"
        )

    # Mark the predicted lines with a failure
    for n in checkpoint["N"][len(checkpoint["runtime"]) :]:
        ax["complexity"].plot(n, 8_000, "x", color="C1")
        ax["complexity"].annotate(
            "fail", xy=(n * 1.1, 12_000), fontsize="x-small", color="C1"
        )

    # Annotate the machine limit
    ax["complexity"].axhline(8_000, color="black", linestyle="dotted")
    ax["complexity"].annotate(
        r"$\approx$ 8 GB (Machine limit)",
        xy=(10, 9_000),
        color="black",
        fontsize="small",
        zorder=0,
    )

    # Adjust the x- and y-limits and some other formats
    ax["complexity"].set_xlabel("Problem size $d$")
    ax["complexity"].set_ylabel("Memory consumption (MB)")
    ax["complexity"].set_ylim((1.5e-1, 40_000))
    ax["complexity"].set_xlim((0.6 * checkpoint["N"][0], checkpoint["N"][-1] * 3))
    ax["complexity"].set_xscale("log", base=2)
    ax["complexity"].set_yscale("log", base=2)
    ax["complexity"].legend(fontsize="small", loc="lower right")

    # Save the figure
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()
