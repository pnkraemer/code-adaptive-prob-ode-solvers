from tueplots import axes
import dataclasses

from typing import Callable


def plot_params():
    fontsize = fontsize_uniform(11)
    axes_lines = axes.lines(base_width=0.25)
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


def style_harder():
    def label(string, /):
        if "()" in string:
            string = string.replace("()", "")
        if "interp." in string:
            string = string.replace("interp.", "interpolate")
        return string

    def color(string, /):
        if "3" in string:
            return "black"
        if "5" in string:
            return "darkorange"
        if "8" in string:
            return "steelblue"
        return "black"

    def marker(string, /):
        if "bosh3" in string.lower():
            return "P"
        if "tsit5" in string.lower():
            return "P"
        if "dopri8" in string.lower():
            return "P"
        return "o"

    def linestyle(string, /):
        if "bosh3" in string.lower():
            return "dotted"
        if "tsit5" in string.lower():
            return "dotted"
        if "dopri8" in string.lower():
            return "dotted"

    return Style(
        marker=marker,
        label=label,
        color=color,
        linestyle=linestyle,
        alpha_line=None,
        alpha_fill_between=lambda _s: 0.1,
    )


def style_simple():
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
        alpha_line=lambda s: 0.9,
    )
