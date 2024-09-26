from tueplots import axes
import dataclasses

from typing import Callable


def plot_params():
    fontsize = fontsize_uniform(9)
    axes_lines = axes.lines(base_width=0.35)
    axes_legend = axes.legend()
    axes_grid = axes.grid()
    axes_ticks = axes.tick_direction(x="in", y="in")
    return {
        "markers.fillstyle": "none",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{cmbright}",
        "figure.constrained_layout.use": True,
        "lines.markeredgewidth": 0.5,
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
        "axes.titlesize": "medium",
        "legend.fontsize": "small",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
    }


@dataclasses.dataclass
class Style:
    marker: Callable[[str], str]
    label: Callable[[str], str]
    color: Callable[[str], str]
    linestyle: Callable[[str], str]
    alpha_line: Callable[[str], float]
    alpha_fill_between: Callable[[str], float]
    zorder: Callable[[str], int]


def style_harder():
    def label(string, /):
        if "()" in string:
            string = string.replace("()", "")
        if "probdiffeq" in string:
            string = string.replace(" via probdiffeq", "")
        if "diffrax" in string:
            string = string.replace(" via diffrax", "")
        return string

    def marker(string, /):
        if "diffrax" in string.lower():
            return "P"
        if "prob" in string.lower():
            return "o"

    def linestyle(string, /):
        if "2" in string.lower() or "3" in string.lower():
            return "dotted"
        if "4" in string.lower() or "5" in string.lower():
            return "solid"
        if "7" in string.lower() or "8" in string.lower():
            return "dashed"
        raise ValueError(string)

    return Style(
        marker=marker,
        label=label,
        color=lambda _s: "black",
        linestyle=linestyle,
        alpha_line=None,
        alpha_fill_between=lambda _s: 0.0,
    )


def style_simple():
    def marker(string, /):
        if "step" in string.lower():
            return "^"
        if "loop" in string.lower():
            return "o"
        if "diffrax" in string.lower():
            return "s"

        raise ValueError(string)

    def color(string, /):
        if "step" in string.lower():
            return "C0"
        if "loop" in string.lower():
            return "C1"
        if "diffrax" in string.lower():
            return "C2"

        raise ValueError(string)

    def zorder(string, /):
        if "step" in string.lower():
            return 1
        if "loop" in string.lower():
            return 2
        if "diffrax" in string.lower():
            return 0

        raise ValueError(string)

    def label(string, /):
        if "()" in string:
            string = string.replace("()", "")
        if "probdiffeq" in string:
            string = string.replace("via probdiffeq", "")
        if "diffrax" in string:
            string = string.replace("via diffrax", "")
        if "TS" in string:
            string = string.replace("TS0", "Prob")
        if "can't" in string:
            string = string.replace("can't", "no")
        return string.capitalize()

    def linestyle(string, /):
        if "2" in string.lower():
            return "dotted"
        if "3" in string.lower():
            return "dotted"

        if "4" in string.lower():
            return "solid"
        if "5" in string.lower():
            return "solid"

        raise ValueError

    return Style(
        marker=marker,
        label=label,
        color=color,
        alpha_fill_between=lambda _s: 0.0,
        linestyle=linestyle,
        alpha_line=lambda _s: 0.99,
        zorder=zorder,
    )
