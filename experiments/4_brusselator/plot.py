import dataclasses
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from odecheckpts import exp_util


@dataclasses.dataclass
class Data:
    Ns: list
    memory: list
    runtime: list
    num_steps: list
    ts: list
    ys: list

    def __getitem__(self, item):
        return Data(
            Ns=self.Ns[item],
            memory=self.memory[item],
            runtime=self.runtime[item],
            num_steps=self.num_steps[item],
            ts=self.ts[item],
            ys=self.ys[item],
        )

    @classmethod
    def load(cls, string, /) -> dict:
        filename = os.path.dirname(__file__) + f"/{string}.npy"
        data = jnp.load(filename, allow_pickle=True).item()

        Ns, mem, tm, n = data["N"], data["memory"], data["runtime"], data["num_steps"]
        ts, ys = data["ts"], data["ys"]
        return cls(
            Ns=Ns[2:],
            memory=mem[2:],
            runtime=tm[2:],
            num_steps=n[2:],
            ts=ts[2:],
            ys=ys[2:],
        )

    def plot_pcolormesh(self, axis, /, *, resolution: int):
        idx = self.Ns.index(resolution)
        num_steps = self.num_steps[idx]
        ts = self.ts[idx]
        ys = self.ys[idx]

        title = "a) Brusselator:"
        title += f" ${len(ts):,}$ target pts."
        title += f", ${int(num_steps):,}$ compute pts."
        axis.set_title(title)

        axis.set_xlabel("Space dimension")
        axis.set_ylabel("Time dimension $t$")
        axis.tick_params(which="both", direction="out")

        Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
        xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
        Xs, Ts = jnp.meshgrid(xs, ts)
        axis.pcolormesh(Xs, Ts, Us)

    def plot_mark_vline_resolution(self, axis, /, *, resolution: int):
        idx = self.Ns.index(resolution)
        N, mem = self.Ns[idx], self.memory[idx]
        axis.axvline(N, color="black", linestyle="dotted")

        text = "Figure a)"
        xy = (N, max(self.memory))
        kwargs = {"rotation": 90 * 3, "color": "black", "fontsize": "x-small"}
        axis.annotate(text, xy=xy, **kwargs)

    def plot_mark_hline_machine_limit(self, axis, /, *, where: int, text: str):
        axis.axhline(where, color="black", linestyle="dotted")
        xy = (min(self.Ns), where * 1.2)
        axis.annotate(text, xy=xy, color="black", fontsize="small")

    def plot_curve_memory(
        self, axis, /, *, color: str, label: str, linestyle="solid", marker="."
    ):
        axis.plot(
            self.Ns,
            self.memory,
            marker=marker,
            label=label,
            color=color,
            linestyle=linestyle,
        )

    def plot_annotate_runtime(self, axis, /, *, color: str, stride: int):
        Ns = self.Ns[::stride]
        runs = self.runtime[::stride]
        mems = self.memory[::stride]
        for n, t, m in zip(Ns, runs, mems):
            axis.plot(n, m, "s", markersize=10, color=color)
            axis.annotate(
                f"{t:.1f}s",
                xy=(n, m),
                xytext=(n * 1.1, 0.35 * m),
                color=color,
                fontsize="x-small",
            )

    def plot_annotate_failures(self, axis, /, *, color: str):
        for n, m in zip(self.Ns[len(self.runtime) :], self.memory[len(self.runtime) :]):
            xy = (1.1 * n, 1.1 * m)
            text = f"{(m/1024):.1f} GB"
            axis.annotate(text, xy=xy, fontsize="x-small", color=color)


def main():
    plt.rcParams.update(exp_util.plot_params())
    checkpoint = Data.load("data_checkpoint")
    textbook = Data.load("data_textbook")

    # Prepare the Figure
    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)

    # Plot a bunch of stuff
    checkpoint.plot_pcolormesh(ax["brusselator"], resolution=90)
    checkpoint.plot_mark_vline_resolution(ax["complexity"], resolution=90)

    checkpoint.plot_curve_memory(ax["complexity"], color="C0", label="Ours")
    checkpoint.plot_annotate_runtime(ax["complexity"], color="C0", stride=2)

    # todo: plot the predicted runtime for the failures
    nsuccess = len(textbook.runtime)
    textbook[:nsuccess].plot_curve_memory(
        ax["complexity"], color="C1", label="Prev. SotA"
    )
    textbook[nsuccess:].plot_curve_memory(
        ax["complexity"],
        color="gray",
        label="Prev. SotA (estimated)",
        linestyle="None",
        marker="x",
    )
    text = "SotA exceeds capacity"
    textbook.plot_mark_hline_machine_limit(
        ax["complexity"], where=textbook.memory[nsuccess - 1], text=text
    )
    textbook.plot_annotate_runtime(ax["complexity"], color="C1", stride=2)
    textbook.plot_annotate_failures(ax["complexity"], color="gray")

    # Adjust the x- and y-limits and some other formats
    ax["complexity"].set_title("b) Memory consumption vs. problem size")
    ax["complexity"].set_xlabel("Problem size $d$")
    ax["complexity"].set_ylabel("Memory consumption (MB)")
    # ax["complexity"].set_ylim((1.5e-1, 40_000))
    # ax["complexity"].set_xlim((2, 2**10))
    ax["complexity"].set_xscale("log", base=2)
    ax["complexity"].set_yscale("log", base=2)
    ax["complexity"].legend(fontsize="x-small")

    # Save the figure
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
