# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""matplotlib in the course's style, in light and dark alike.

The learner writes ordinary matplotlib (`import matplotlib.pyplot as plt`, `plt.plot(...)`). When pyplot is
imported, the course sets its style and its way of showing a figure: as an SVG drawn into the page, whose
colours are the course's tokens (var(--ea-*)), so the same figure follows the light and the dark theme.
matplotlib itself only knows fixed colours, so the style uses stand-in colours that `render` swaps for tokens.

`describe` reads back what a figure shows (the kind of chart, its values, its title and axis labels), for the
checks of plotting exercises.
"""
from __future__ import annotations

import html
import io
import re
import sys

# Stand-in colours: matplotlib draws with these, and `render` swaps them for the course's tokens.
_TOKENS = {
    "#0a0b01": "var(--ea-ink)",
    "#0a0b02": "var(--ea-ink-2)",
    "#0a0b03": "var(--ea-ink-3)",
    "#0a0b04": "var(--ea-line)",
    "#0a0b05": "var(--ea-data)",
    "#0a0b06": "var(--ea-axis)",
    "#0a0b07": "var(--ea-page)",
}
INK, INK_2, INK_3, LINE, DATA, AXIS, PAGE = _TOKENS
# The roles of the style guide: a tone for lines and points, and a darker one for text in a chart.
_ROLES = {
    "#0a0c01": "var(--ea-pred)", "#0a0c02": "var(--ea-pred-text)",
    "#0a0c03": "var(--ea-param)", "#0a0c04": "var(--ea-param-text)",
    "#0a0c05": "var(--ea-residual)", "#0a0c06": "var(--ea-residual-text)",
    "#0a0c07": "var(--ea-loss)", "#0a0c08": "var(--ea-loss-text)",
    "#0a0c09": "var(--ea-grad)", "#0a0c0a": "var(--ea-grad-text)",
    "#0a0c0b": "var(--ea-hyper)", "#0a0c0c": "var(--ea-hyper-text)",
    "#0a0c0d": "var(--ea-hidden)", "#0a0c0e": "var(--ea-hidden-text)",
}
(PRED, PRED_TEXT, PARAM, PARAM_TEXT, RESIDUAL, RESIDUAL_TEXT, LOSS, LOSS_TEXT,
 GRAD, GRAD_TEXT, HYPER, HYPER_TEXT, HIDDEN, HIDDEN_TEXT) = _ROLES
_TOKENS.update(_ROLES)


def style() -> None:
    """The course's look for every figure (style guide 1 and 3): a calm horizontal grid, no box, the title on the
    left, tick labels in ink-3. Plain data has the data role: every series is grey, and a second or third series
    differs by line style, not by colour. Points have a ring in the page colour."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "figure.figsize": (6.4, 3.6), "figure.dpi": 100,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.edgecolor": AXIS,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_3, "ytick.labelcolor": INK_3,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": LINE, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=[DATA] * 3) + cycler(linestyle=["-", "--", ":"]),
        "lines.color": DATA, "patch.facecolor": DATA, "boxplot.boxprops.color": INK_2,
        "font.family": "sans-serif", "font.sans-serif": ["Inter", "DejaVu Sans"], "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.titleweight": 600, "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "lines.linewidth": 2.2, "lines.markersize": 7, "lines.solid_capstyle": "round",
        "scatter.edgecolors": PAGE, "lines.markeredgecolor": PAGE, "lines.markeredgewidth": 1.2, "patch.edgecolor": "none",
        "legend.frameon": False,
        "svg.fonttype": "none",   # text stays text, in the course's own font
    })


def render(fig, alt: str = "") -> str:
    """A figure as HTML: an SVG at its own size (never wider than the page), with the course's tokens for its
    colours and font, and `alt` as its description for screen readers."""
    for ax in fig.axes:   # an image shows its own pixels: grid lines over it would cover them
        if ax.images:
            ax.grid(False)
            for side in ("top", "right"):   # a frame on all four sides shows where the image ends, also on white or black
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(ax.spines["left"].get_edgecolor())
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    svg = buf.getvalue()
    svg = svg[svg.find("<svg"):]                                  # no XML header or doctype
    for hex_, token in _TOKENS.items():
        svg = re.sub(re.escape(hex_), token, svg, flags=re.IGNORECASE)
    svg = re.sub(r"font-family:[^;\"]*", "font-family:var(--ea-font-ui)", svg)
    svg = re.sub(r"font: ([\d.]+)px '[^']*'", r"font-size: \1px; font-family: var(--ea-font-ui)", svg)
    size = re.search(r'<svg[^>]*? width="([\d.]+)pt"', svg)
    width = f"{float(size.group(1)) * 4 / 3:.0f}px" if size else "100%"   # points to pixels
    label = f' role="img" aria-label="{html.escape(alt, quote=True)}"' if alt else ""
    svg = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', lambda m: f'<svg{m.group(1)}{label} style="width:{width};max-width:100%;height:auto"', svg, count=1)
    return f'<figure class="ea-figure ea-plot" style="margin:0.25rem 0 0.75rem">{svg}</figure>'


def _show(canvas):
    return "text/html", render(canvas.figure)


def install() -> bool:
    """Show every matplotlib figure the course's way, also with `plt.show()`: marimo hands them to this.
    The style uses stand-in colours that only `render` turns into the course's colours, so it is set only when
    the renderer could be put in place; otherwise figures keep marimo's own look. True when it worked."""
    try:
        from marimo._output import mpl as marimo_mpl
        if not hasattr(marimo_mpl, "_render_figure_mimebundle"):
            return False
        marimo_mpl._render_figure_mimebundle = _show
    except Exception:   # outside marimo, such as in a plain script: matplotlib shows figures itself
        return False
    style()
    return True


def when_imported() -> None:
    """Install the course's way of showing figures once matplotlib is imported (or now, if it already is).
    marimo sets up matplotlib itself when it is imported, and then applies its theme: the course takes that
    last step over, so its own style and renderer come after marimo's and are not overwritten."""
    try:
        from marimo._output.formatters.matplotlib_formatters import MatplotlibFormatter
        MatplotlibFormatter.apply_theme = lambda self, theme: install()
    except Exception:   # an older or newer marimo: figures then look as marimo makes them
        pass
    if "matplotlib.pyplot" in sys.modules:
        install()


# --- reading a figure back, for checks --------------------------------------------------------------------

def _num(v):
    """A value from a figure as plain Python: a whole number, a float, or a text."""
    if isinstance(v, str):
        return str(v)
    if v is None:   # a point the code drew as None, such as a moment that was never reached: not a number
        return float("nan")
    v = float(v)
    return int(v) if v.is_integer() else v


def describe(fig=None) -> dict:
    """What the current figure (or `fig`) shows, on its first axes: per chart, in the order of the calls,
    its kind ("line", "scatter", "bar", or "barh" for bars on their side), its x and y values, and the title
    and axis labels; and the scale of each axis, "linear" or "log". Names along a categorical axis are read as
    texts, numbers as numbers."""
    import matplotlib.pyplot as plt
    from matplotlib.category import StrCategoryConverter
    from matplotlib.collections import PathCollection
    from matplotlib.container import BarContainer

    fig = fig or plt.gcf()
    out = {"charts": [], "title": "", "xlabel": "", "ylabel": "", "xscale": "linear", "yscale": "linear"}
    if not fig.axes:
        return out
    ax = fig.axes[0]
    title = " ".join(t for t in (ax.get_title(loc=loc) for loc in ("left", "center", "right")) if t)
    out.update(title=title, xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(), xscale=ax.get_xscale(), yscale=ax.get_yscale())

    def names(axis, values):
        """Positions on an axis with names on it (such as `plt.bar(["a", "b"], ...)`) as those names."""
        if not isinstance(axis.get_converter(), StrCategoryConverter):
            return [_num(v) for v in values]
        mapping = {round(float(pos), 6): name for name, pos in axis.units._mapping.items()}
        return [mapping.get(round(float(v), 6), _num(v)) for v in values]

    charts = []   # (order in which it was drawn, chart)
    for line in ax.get_lines():
        kind = "line" if line.get_linestyle() not in ("None", "", " ") else "scatter"
        charts.append((ax.get_children().index(line), {"kind": kind, "x": [_num(x) for x in line.get_xdata()], "y": [_num(y) for y in line.get_ydata()]}))
    for coll in ax.collections:
        if isinstance(coll, PathCollection) and len(coll.get_offsets()):
            offsets = coll.get_offsets()
            charts.append((ax.get_children().index(coll), {"kind": "scatter", "x": names(ax.xaxis, [p[0] for p in offsets]),
                                                            "y": [_num(p[1]) for p in offsets]}))
    for cont in ax.containers:
        if isinstance(cont, BarContainer) and cont.patches:
            first = ax.get_children().index(cont.patches[0])
            if getattr(cont, "orientation", "vertical") == "horizontal":
                charts.append((first, {"kind": "barh", "x": names(ax.yaxis, [p.get_y() + p.get_height() / 2 for p in cont.patches]),
                                       "y": [_num(p.get_width()) for p in cont.patches]}))
            else:
                charts.append((first, {"kind": "bar", "x": names(ax.xaxis, [p.get_x() + p.get_width() / 2 for p in cont.patches]),
                                       "y": [_num(p.get_height()) for p in cont.patches]}))
    out["charts"] = [c for _, c in sorted(charts, key=lambda c: c[0])]
    return out


def alt_text(described: dict) -> str:
    """A short description of a chart for screen readers, from `describe`: its kind, title and size."""
    kinds = {"line": "Line chart", "scatter": "Scatter plot", "bar": "Bar chart", "barh": "Bar chart"}
    if not described["charts"]:
        return "Empty chart"
    first = described["charts"][0]
    text = kinds.get(first["kind"], "Chart")
    if described["title"]:
        text += f": {described['title']}"
    return text + f", {len(first['y'])} values"
