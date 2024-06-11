from tueplots import axes
import dataclasses

from typing import Callable


def plot_to_pgf(filename, X, y):
    if filename[-4:] != ".dat":
        filename = f"{filename}.dat"
    with open(filename, "w") as file1:
        for x_, y_ in zip(X, y):
            file1.write(f"{x_} {y_}\n")


def plot_params():
    fontsize = fontsize_uniform(10)
    axes_lines = axes.lines(base_width=0.35)
    axes_legend = axes.legend()
    axes_grid = axes.grid()
    axes_ticks = axes.tick_direction(x="in", y="in")
    return {
        "markers.fillstyle": "none",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{cmbright}",
        **fontsize,
        **axes_lines,
        **axes_legend,
        **axes_grid,
        **axes_ticks,
    }


def fontsize_uniform(base):
    return {
        "font.size": base,
        "axes.labelsize": "medium",
        "legend.fontsize": "small",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "axes.titlesize": "medium",
    }


@dataclasses.dataclass
class Style:
    marker: Callable[[str], str]
    label: Callable[[str], str]
    color: Callable[[str], str]
    linestyle: Callable[[str], str]
    alpha_fill_between: Callable[[str], float]


def style_rigid_body():
    def marker(string, /):
        if "can't" in string.lower() or "interp" in string.lower():
            return "^"
        if "bosh3" in string.lower():
            return "s"
        if "tsit5" in string.lower():
            return "s"
        return "o"

    def label(string, /):
        if "()" in string:
            string = string.replace("()", "")
        if "interp." in string:
            string = string.replace("interp.", "interpolate")
        return string

    def color(string, /):
        if "2" in string:
            return "black"
        if "4" in string:
            return "darkorange"
        if "bosh3" in string.lower():
            return "black"
        if "tsit5" in string.lower():
            return "darkorange"
        return "black"

    def alpha_fill_between(_string):
        return 0.1

    def linestyle(string, /):
        if "bosh3" in string.lower():
            return "dotted"
        if "tsit5" in string.lower():
            return "dotted"
        if "can't" in string.lower():
            return "dashed"

        return "solid"

    return Style(
        marker=marker,
        label=label,
        color=color,
        alpha_fill_between=alpha_fill_between,
        linestyle=linestyle,
    )
