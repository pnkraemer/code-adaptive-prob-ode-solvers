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

    num_steps = checkpoint["num_steps"][-1]
    ts = checkpoint["ts"][-1]
    ys = checkpoint["ys"][-1]
    # ts, ys = solution.t, solution.u

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)

    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)

    msg = f"a) Brusselator: ${len(ts):,}$ target pts., ${int(jnp.amax(num_steps)):,}$ compute pts."
    ax["brusselator"].set_title(msg)
    ax["brusselator"].pcolormesh(Xs, Ts, Us)
    ax["brusselator"].set_xlabel("Space dimension")
    ax["brusselator"].set_ylabel("Time dimension $t$")
    ax["brusselator"].tick_params(which="both", direction="out")

    ax["complexity"].set_title("b) Memory consumption vs. problem size")

    for n, t, m in zip(checkpoint["N"], checkpoint["runtime"], checkpoint["memory"]):
        ax["complexity"].semilogy(n, m, "^", color="C0")
        ax["complexity"].annotate(
            f"{t:.1f}s", xy=(n, m), xytext=(n, 0.35 * m), color="C0", fontsize="small"
        )

    for n, t, m in zip(textbook["N"], textbook["runtime"], textbook["memory"]):
        ax["complexity"].semilogy(n, m, "o", color="C1")
        ax["complexity"].annotate(
            f"{t:.1f}s", xy=(n, m), xytext=(n, 1.5 * m), color="C1", fontsize="small"
        )

    ax["complexity"].set_xlabel("Problem size $d$")
    ax["complexity"].set_ylabel("Memory consumption (MB)")
    ax["complexity"].set_ylim((2e-1, 40_000))

    ax["complexity"].axvline(checkpoint["N"][-1], color="black", linestyle="dotted")
    ax["complexity"].axhline(8_000, color="black", linestyle="dotted")
    ax["complexity"].annotate(
        r"Machine limit ($\approx$ 8 GB)",
        xy=(10, 9_000),
        color="black",
        fontsize="small",
        zorder=0,
    )
    ax["complexity"].annotate(
        "Used for Figure a)",
        xy=(checkpoint["N"][-1], 3e-1),
        rotation=90 * 3,
        color="black",
        fontsize="small",
    )
    ax["complexity"].semilogy(
        checkpoint["N"], checkpoint["memory"], marker="None", label="Our code"
    )
    ax["complexity"].semilogy(
        textbook["N"], textbook["memory"], marker="None", label="Fully adaptive"
    )
    ax["complexity"].legend()
    plt.show()
