import matplotlib.pyplot as plt


import jax.numpy as jnp
from odecheckpts import exp_util


import os


def load(str) -> dict:
    return jnp.load(os.path.dirname(__file__) + f"/{str}.npy", allow_pickle=True).item()


if __name__ == "__main__":
    plt.rcParams.update(exp_util.plot_params())
    results_checkpoint = load("data_checkpoint")
    results_textbook = load("data_textbook")

    num_steps = results_checkpoint["num_steps"][-1]
    ts = results_checkpoint["ts"][-1]
    ys = results_checkpoint["ys"][-1]
    # ts, ys = solution.t, solution.u

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)

    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)
    ax["brusselator"].pcolormesh(Xs, Ts, Us)
    ax["brusselator"].set_xlabel("Space dimension")
    ax["brusselator"].set_ylabel("Time dimension $t$")
    msg = f"$N={len(ts):,}$ target points, $M={int(jnp.amax(num_steps)):,}$ compute points"
    ax["brusselator"].set_title(msg)
    ax["brusselator"].tick_params(which="both", direction="out")

    ax["complexity"].axvline(
        results_checkpoint["N"][-1], color="black", linestyle="dotted"
    )
    ax["complexity"].axhline(8_000, color="black", linestyle="dotted")
    ax["complexity"].semilogy(results_textbook["N"], results_textbook["memory"], "^")
    ax["complexity"].semilogy(
        results_checkpoint["N"], results_checkpoint["memory"], "o"
    )
    plt.show()
