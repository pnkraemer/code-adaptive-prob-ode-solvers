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

    def __len__(self):
        return len(self.Ns)

    def __getitem__(self, item):
        return Data(
            Ns=self.Ns[item],
            memory=self.memory[item],
            runtime=self.runtime[item],
            num_steps=self.num_steps[item],
        )

    @classmethod
    def load(cls, string, /, skip: int):
        filename = os.path.dirname(__file__) + f"/{string}.npy"
        data = jnp.load(filename, allow_pickle=True).item()

        Ns, mem, tm, n = data["N"], data["memory"], data["runtime"], data["num_steps"]
        return cls(
            Ns=Ns[skip:], memory=mem[skip:], runtime=tm[skip:], num_steps=n[skip:]
        )

    def plot_mark_vline_resolution(self, axis, /, *, resolution: int):
        idx = self.Ns.index(resolution)
        N = self.Ns[idx]
        axis.axvline(N, color="black", linestyle="dotted")

        text = "Figure a)"
        xy = (N, 40_000)
        kwargs = {"rotation": 90 * 3, "color": "black", "fontsize": "x-small"}
        axis.annotate(text, xy=xy, **kwargs)

    def plot_mark_hline_capacity(self, axis, /, *, text: str):
        axis.axhline(max(self.memory), color="black", linestyle="dotted")

        xy = (3, max(self.memory) * 1.2)
        axis.annotate(text, xy=xy, color="black", fontsize="small")

    def plot_curve_memory(self, axis, /, **mpl_kwargs):
        axis.plot(self.Ns, self.memory, **mpl_kwargs)

    def plot_annotate_max_memory(self, axis, /, **mpl_kwargs):
        n, m = self.Ns[-1], self.memory[-1]
        xy = (n, m)
        xytext = (n * 1.1, m * 1.8)
        text = f"{(m):.2f} MB"
        axis.annotate(text, xy=xy, xytext=xytext, fontsize="x-small", **mpl_kwargs)
        axis.plot(n, m, "s", markersize=10, **mpl_kwargs)

    def plot_annotate_runtime(
        self, axis, /, *, annotate: str, stride: int, **mpl_kwargs
    ):
        Ns = self.Ns[::stride]
        runs = self.runtime[::stride]
        mems = self.memory[::stride]
        for n, t, m in zip(Ns, runs, mems):
            if annotate == "upper":
                xytext = (n / 1.8, m * 2)
            else:
                xytext = (n * 1.1, m / 4)
            axis.annotate(
                f"{t:.1f} sec",
                xy=(n, m),
                xytext=xytext,
                fontsize="x-small",
                **mpl_kwargs,
            )
            axis.plot(n, m, "s", markersize=10, **mpl_kwargs)

    def plot_annotate_failures(self, axis, /, *, color: str):
        for n, m in zip(self.Ns[len(self.runtime) :], self.memory[len(self.runtime) :]):
            xy = (1.2 * n, 1.1 * m)
            text = f"{(m/1024):.0f} GB (est.)"
            axis.annotate(text, xy=xy, fontsize="x-small", color=color)


def load_meshgrid(string, resolution):
    filename = os.path.dirname(__file__) + f"/{string}.npy"
    data = jnp.load(filename, allow_pickle=True).item()

    idx = data["N"].index(resolution)
    num_steps = data["num_steps"][idx]
    ts = data["ts"][idx]
    ys = data["ys"][idx]

    Us, Vs = jnp.split(ys, axis=1, indices_or_sections=2)
    xs = jnp.linspace(0, 1, endpoint=True, num=len(ys[0]) // 2)
    Xs, Ts = jnp.meshgrid(xs, ts)
    return Xs, Ts, Us, num_steps


def plot_pcolormesh(axis, /, Xs, Ts, Us, *, num_steps):
    title = "a) Brusselator solution:"
    title += f" ${len(Ts):,}$ target points"
    # title += f", ${int(num_steps):,}$ compute pts."
    axis.set_title(title)

    axis.set_xlabel("Space dimension")
    axis.set_ylabel("Time dimension")
    axis.tick_params(which="both", direction="out")
    axis.pcolormesh(Xs, Ts, Us)


def main():
    plt.rcParams.update(exp_util.plot_params())
    checkpoint = Data.load("data_checkpoint", skip=1)
    textbook = Data.load("data_textbook", skip=1)

    # Prepare the Figure
    layout = [["brusselator", "complexity"]]
    fig, ax = plt.subplot_mosaic(layout, figsize=(6.75, 2.5), dpi=150)

    Xs, Ts, Us, num_steps = load_meshgrid("data_checkpoint", resolution=128)
    plot_pcolormesh(ax["brusselator"], Xs, Ts, Us, num_steps=num_steps)

    # Plot the checkpointing info
    checkpoint.plot_mark_vline_resolution(ax["complexity"], resolution=128)
    checkpoint.plot_curve_memory(ax["complexity"], marker=".", color="C0", label="Ours")
    checkpoint.plot_annotate_runtime(
        ax["complexity"], annotate="lower", color="C0", stride=2
    )
    checkpoint.plot_annotate_max_memory(ax["complexity"], color="C0")

    # Split the current SotA into "good" and "bad"
    nsuccess = len(textbook.runtime)
    textbook_good, textbook_bad = textbook[:nsuccess], textbook[nsuccess:]

    # Plot the "good" results
    textbook_good.plot_curve_memory(
        ax["complexity"], marker=".", color="C1", label="Prev. SotA"
    )
    textbook_good.plot_annotate_runtime(
        ax["complexity"], annotate="upper", color="C1", stride=2
    )

    ax["complexity"].axhline(8 * 1024, color="black", linestyle="dotted")
    xy = (3, 10_000)
    ax["complexity"].annotate(
        "8 GB (machine capacity)", xy=xy, color="black", fontsize="x-small"
    )

    # textbook_good.plot_mark_hline_capacity(ax["complexity"], text="SotA max. capacity")
    textbook_bad.plot_curve_memory(
        ax["complexity"],
        color="C1",
        label="Prev. SotA (failed)",
        linestyle="None",
        marker="x",
    )
    textbook_bad.plot_annotate_failures(ax["complexity"], color="C1")

    # Adjust the x- and y-limits and some other formats
    ax["complexity"].set_title("b) Memory consumption vs. problem size")
    ax["complexity"].set_xlabel("Problem size $d$")
    ax["complexity"].set_ylabel("Memory consumption")
    ax["complexity"].set_ylim((1.2e-1, 2_000_000))
    ax["complexity"].set_xlim((2, 1.25 * 2**11))
    ax["complexity"].set_xscale("log", base=2)
    ax["complexity"].set_yscale("log", base=2)
    ax["complexity"].legend(fontsize="x-small")

    ax["complexity"].set_yticks([1024**i for i in range(3)])
    ax["complexity"].set_yticklabels([f"1 {m}" for m in ["MB", "GB", "TB"]])

    # Save the figure
    plt.savefig(f"./figures/{os.path.basename(os.path.dirname(__file__))}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
