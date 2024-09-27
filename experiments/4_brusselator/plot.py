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

    num_steps = checkpoint["num_steps"][-2]
    ts = checkpoint["ts"][-2]
    ys = checkpoint["ys"][-2]
    # ts, ys = solution.t, solution.u

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)

    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)

    print(textbook["N"])
    msg = f"a) Brusselator: ${len(ts):,}$ target pts., ${int(jnp.amax(num_steps)):,}$ compute pts."
    ax["brusselator"].set_title(msg)
    ax["brusselator"].pcolormesh(Xs, Ts, Us)
    ax["brusselator"].set_xlabel("Space dimension")
    ax["brusselator"].set_ylabel("Time dimension $t$")
    ax["brusselator"].tick_params(which="both", direction="out")

    ax["complexity"].set_title("b) Memory consumption vs. problem size")

    for n, t, m in zip(
        checkpoint["N"][::2], checkpoint["runtime"][::2], checkpoint["memory"][::2]
    ):
        # if n % 3 == 0:
        ax["complexity"].plot(n, m, "s", markersize=10, color="C0")
        ax["complexity"].annotate(
            f"{t:.1f}s",
            xy=(n, m),
            xytext=(n * 1.1, 0.35 * m),
            color="C0",
            fontsize="x-small",
        )

    for n, t, m in zip(
        textbook["N"][::2], textbook["runtime"][::2], textbook["memory"][::2]
    ):
        ax["complexity"].plot(n, m, "s", markersize=10, color="C1")
        # if n % 3 == 0:
        ax["complexity"].annotate(
            f"{t:.1f}s", xy=(n, m), xytext=(n, 0.35 * m), color="C1", fontsize="x-small"
        )
    for n in checkpoint["N"][-3:]:
        ax["complexity"].plot(n, 8_000, "x", color="C1")
        ax["complexity"].annotate(
            "fail", xy=(n * 1.1, 12_000), fontsize="x-small", color="C1"
        )

    ax["complexity"].set_xlabel("Problem size $d$")
    ax["complexity"].set_ylabel("Memory consumption (MB)")
    ax["complexity"].set_ylim((1.5e-1, 40_000))

    ax["complexity"].axhline(8_000, color="black", linestyle="dotted")
    ax["complexity"].annotate(
        r"$\approx$ 8 GB (Machine limit)",
        xy=(10, 9_000),
        color="black",
        fontsize="small",
        zorder=0,
    )
    # ax["complexity"].axhline(2_000, color="black", linestyle="dotted")
    # ax["complexity"].annotate(
    #     r"$\approx$ GB",
    #     xy=(10, 4_400),
    #     color="black",
    #     fontsize="small",
    #     zorder=0,
    # )

    ax["complexity"].set_xlim((0.6 * checkpoint["N"][0], checkpoint["N"][-1] * 3))

    ax["complexity"].axvline(checkpoint["N"][-2], color="black", linestyle="dotted")
    ax["complexity"].annotate(
        "Used for Figure a)",
        xy=(checkpoint["N"][-2], 3 * checkpoint["memory"][-2]),
        rotation=90 * 3,
        color="black",
        fontsize="x-small",
    )

    ax["complexity"].plot(
        textbook["N"], textbook["memory"], marker=".", label="Previous SoTA", color="C1"
    )
    ax["complexity"].plot(
        checkpoint["N"], checkpoint["memory"], marker=".", label="Our code", color="C0"
    )
    ax["complexity"].set_xscale("log", base=2)
    ax["complexity"].set_yscale("log", base=2)
    ax["complexity"].legend(fontsize="small", loc="lower right")
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()
