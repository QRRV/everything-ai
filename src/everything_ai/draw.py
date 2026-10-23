# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Small figures drawn as SVG, in the course's colours, for lessons before the learner meets
a plotting library. Every colour is a token (var(--ea-*)), so a figure follows the light and
dark theme. Roles follow the style guide: orange is what is learned, purple counts mistakes.
Code shows without ligatures, so `>=` looks the way you type it."""
from __future__ import annotations

import html

import marimo as mo

_FONT = "font-family:var(--ea-font-ui)"


def _svg(width: int, height: int, title: str, body: str, *, natural: bool = False, drag: str = "") -> mo.Html:
    """A figure that fills the column up to 40rem; with `natural`, never wider than its own width, so a small
    figure keeps its text at the size it was drawn. `drag` is the axis a play block drags along, as chart.figure
    writes it: "left right low high log"."""
    widest = f"min(40rem,{width}px)" if natural else "40rem"
    along = f' data-drag="{drag}"' if drag else ""
    return mo.Html(
        f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{widest}">'
        f'<svg viewBox="0 0 {width} {height}" role="img"{along} style="width:100%;height:auto;{_FONT};font-variant-ligatures:none;font-feature-settings:&quot;calt&quot; 0,&quot;liga&quot; 0">'
        f'<title>{html.escape(title)}</title>{body}</svg></figure>')


def threshold_line(points, threshold=None, *, lo: int, hi: int, axis: str, yes: str, no: str,
                   rule: str = "", wrong: str = "", title: str = "", drag: bool = False) -> mo.Html:
    """Examples on a number line: a filled dot is a yes, an open dot a no. With a threshold,
    the rule is "yes below the threshold"; the examples it gets wrong get a purple ring. With `drag`, a play block
    (L.explore) sets the threshold where you drag along the line. Its text is 16 px, so that it stays 11 px or more in
    a play block at 700 px."""
    W, H, L, R = 640, 150, 30, 610

    def x(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    parts = [f'<line x1="{L}" y1="92" x2="{R}" y2="92" style="stroke:var(--ea-axis);stroke-width:1.5"/>']
    for t in range(lo, hi + 1):
        if t % 5 == 0:
            parts.append(f'<line x1="{x(t):.1f}" y1="92" x2="{x(t):.1f}" y2="98" style="stroke:var(--ea-axis)"/>'
                         f'<text x="{x(t):.1f}" y="114" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:middle">{t}</text>')
    parts.append(f'<text x="{R}" y="138" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(axis)}</text>')
    if threshold is not None:
        tx = x(threshold)
        parts.append(f'<rect x="{L}" y="40" width="{tx - L:.1f}" height="52" style="fill:var(--ea-param-wash);opacity:0.7"/>'
                     f'<line x1="{tx:.1f}" y1="30" x2="{tx:.1f}" y2="98" style="stroke:var(--ea-param);stroke-width:2.5"/>')
        if rule:
            anchor = "start" if threshold < (lo + hi) / 2 else "end"
            dx = 6 if anchor == "start" else -6
            parts.append(f'<text x="{tx + dx:.1f}" y="24" style="font-size:16px;font-weight:600;fill:var(--ea-param-text);'
                         f'text-anchor:{anchor}">{html.escape(rule)}</text>')
    seen: dict[float, int] = {}
    for v, is_yes in points:
        k = seen.get(v, 0)
        seen[v] = k + 1
        cx, cy = x(v), 70 - 18 * k
        if threshold is not None and (v < threshold) != is_yes:
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="11" style="fill:none;stroke:var(--ea-loss);stroke-width:2.5"/>')
        style = "fill:var(--ea-ink)" if is_yes else "fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:2"
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="6" style="{style}"/>')
    legend = (f'<circle cx="{L + 6}" cy="134" r="5" style="fill:var(--ea-ink)"/>'
              f'<text x="{L + 16}" y="138" style="font-size:16px;fill:var(--ea-ink)">{html.escape(yes)}</text>'
              f'<circle cx="{L + 96}" cy="134" r="5" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:2"/>'
              f'<text x="{L + 106}" y="138" style="font-size:16px;fill:var(--ea-ink)">{html.escape(no)}</text>')
    if threshold is not None and wrong:
        legend += (f'<circle cx="{L + 196}" cy="134" r="7" style="fill:none;stroke:var(--ea-loss);stroke-width:2.5"/>'
                   f'<text x="{L + 210}" y="138" style="font-size:16px;fill:var(--ea-loss-text)">{html.escape(wrong)}</text>')
    return _svg(W, H, title or axis, "".join(parts) + legend, drag=f"{L} {R} {lo} {hi} 0" if drag else "")


def mistakes_per_rule(counts: dict[int, int], *, axis: str, value: str, best: str = "", title: str = "") -> mo.Html:
    """A bar per candidate rule: how many mistakes it makes. The rules with the fewest are orange. Its text is 16 px, as
    in threshold_line, which it stands beside."""
    W, H, L, R, B, T = 640, 200, 40, 610, 150, 30
    keys = sorted(counts)
    top = max(max(counts.values()), 1)
    fewest = min(counts.values())
    step = (R - L) / len(keys)
    parts = [f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" style="stroke:var(--ea-axis);stroke-width:1.5"/>',
             f'<text x="{L - 8}" y="{T + 4}" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:end">{top}</text>',
             f'<text x="{L - 8}" y="{B + 4}" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:end">0</text>',
             f'<text x="{L}" y="{T - 12}" style="font-size:16px;fill:var(--ea-loss-text)">{html.escape(value)}</text>']
    for i, k in enumerate(keys):
        h = (B - T) * counts[k] / top
        bx = L + i * step + 1.5
        colour = "var(--ea-param)" if counts[k] == fewest else "var(--ea-loss)"
        opacity = "1" if counts[k] == fewest else "0.55"
        parts.append(f'<rect x="{bx:.1f}" y="{B - max(h, 4):.1f}" width="{step - 3:.1f}" height="{max(h, 4):.1f}" '
                     f'rx="1.5" style="fill:{colour};opacity:{opacity}"/>')
        if k % 5 == 0:
            parts.append(f'<text x="{bx + (step - 3) / 2:.1f}" y="{B + 18}" style="font-size:16px;fill:var(--ea-ink-2);'
                         f'text-anchor:middle">{k}</text>')
    parts.append(f'<text x="{R}" y="{B + 42}" style="font-size:16px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(axis)}</text>')
    if best:
        parts.append(f'<text x="{L}" y="{B + 42}" style="font-size:16px;font-weight:600;fill:var(--ea-param-text)">{html.escape(best)}</text>')
    return _svg(W, H, title or value, "".join(parts))


def route_grid(rows: list[str], cols: list[str], counts: dict[tuple[int, int], int], *, unit: tuple[str, str],
               arrows: int = 0, title: str = "") -> mo.Html:
    """A grid of the route: a row per level, a column per branch, and in each cell how many lessons
    lie there. A fuller cell is a little darker. The first `arrows` columns get an arrow between them,
    for the order in which the route goes through them; the columns after that stand apart, dashed."""
    W, LW, CH, GAP, RG = 640, 104, 62, 22, 10
    HEAD = 30 if title else 0        # the title, visible above the grid
    TOP = HEAD + 34
    cw = (W - LW - GAP * (len(cols) - 1)) / len(cols)
    most = max(counts.values() or [1])

    def cx(j):
        return LW + j * (cw + GAP)

    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for j, name in enumerate(cols):
        parts.append(f'<text x="{cx(j) + cw / 2:.1f}" y="{HEAD + 20}" style="font-size:14px;font-weight:650;fill:var(--ea-ink);'
                     f'text-anchor:middle">{html.escape(name)}</text>')
    for i, name in enumerate(rows):
        y = TOP + i * (CH + RG)
        parts.append(f'<text x="0" y="{y + CH / 2 + 5:.1f}" style="font-size:14px;font-weight:650;fill:var(--ea-ink)">'
                     f'{html.escape(name)}</text>')
        for j in range(len(cols)):
            n = counts.get((i, j), 0)
            side = j >= arrows > 0
            fill = 0.03 + 0.17 * n / most
            dash = "stroke-dasharray:4 3;" if side else ""
            parts.append(f'<rect x="{cx(j):.1f}" y="{y}" width="{cw:.1f}" height="{CH}" rx="6" style="fill:var(--ea-ink);'
                         f'fill-opacity:{fill:.3f};stroke:var(--ea-ink-3);stroke-width:1.2;{dash}"/>')
            word = unit[0] if n == 1 else unit[1]
            parts.append(f'<text x="{cx(j) + cw / 2:.1f}" y="{y + 30}" style="font-size:21px;font-weight:700;fill:var(--ea-ink);'
                         f'text-anchor:middle">{n}</text>'
                         f'<text x="{cx(j) + cw / 2:.1f}" y="{y + 49}" style="font-size:12.5px;fill:var(--ea-ink-2);'
                         f'text-anchor:middle">{html.escape(word)}</text>')
            if j + 1 < arrows:
                ax, ay = cx(j) + cw + GAP / 2, y + CH / 2
                parts.append(f'<path d="M{ax - 6:.1f} {ay} h11 M{ax + 1:.1f} {ay - 4} l4 4 l-4 4" '
                             f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round"/>')
    H = TOP + len(rows) * (CH + RG) - RG + 2
    return _svg(W, H, title or ", ".join(rows), "".join(parts))


def maze_grids(panels: list[dict], *, alt: str, cell: int = 30) -> mo.Html:
    """Mazes as grids of cells, one panel per maze, side by side; in a narrow window, such as beside an editor, they go
    one below the other before their text gets too small, as chart.figure does. Each panel is a dict:
      rows      the maze as strings: '#' is a wall, any other sign a cell you can walk on; 'S' and 'G' are written in
                their cell, the start and the goal
      title     the words above the panel (optional)
      done      cells (row, column) with a grey fill, a step of the data colour: the states a search expanded
      frontier  cells with a dashed ring: the states on the frontier
      numbers   {cell: text} written in the cell, in the text colour of `role` ("loss" for steps, "pred" for estimates)
      path      the cells of a path, drawn as a line through their middles
      mark      one cell with a thick ink ring: the state just expanded
      index     True: the row numbers on the left and the column numbers on top, as the code counts them
    Walls are data, grey; the path is ink. `alt` says what the figure shows, for a screen reader."""
    def at(r, c, left, top):
        return left + c * cell, top + r * cell

    size = max(11, round(cell * 0.4))   # the text in a cell
    svgs, widest = [], 0
    for p in panels:
        rows = p["rows"]
        left = 18 if p.get("index") else 1
        top = (24 if p.get("title") else 0) + (16 if p.get("index") else 1)
        W, H = left + len(rows[0]) * cell + 1, top + len(rows) * cell + 2
        widest = max(widest, W)
        done, frontier, numbers = set(p.get("done", ())), set(p.get("frontier", ())), p.get("numbers", {})
        tone = f"var(--ea-{p['role']}-text)" if p.get("role") else "var(--ea-ink)"
        parts = []
        if p.get("title"):
            parts.append(f'<text x="1" y="15" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(p["title"])}</text>')
        if p.get("index"):
            for c in range(len(rows[0])):
                x, _ = at(0, c, left, top)
                parts.append(f'<text x="{x + cell / 2:.1f}" y="{top - 4}" style="font-size:11px;fill:var(--ea-ink-3);text-anchor:middle">{c}</text>')
            for r in range(len(rows)):
                _, y = at(r, 0, left, top)
                parts.append(f'<text x="{left - 5}" y="{y + cell / 2 + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-3);text-anchor:end">{r}</text>')
        for r, row in enumerate(rows):
            for c, sign in enumerate(row):
                x, y = at(r, c, left, top)
                if sign == "#":
                    parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:var(--ea-data);stroke:var(--ea-data);stroke-width:1"/>')
                    continue
                fill = "color-mix(in srgb, var(--ea-data) 30%, var(--ea-page))" if (r, c) in done else "var(--ea-page)"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
                if (r, c) in frontier:
                    parts.append(f'<rect x="{x + 3}" y="{y + 3}" width="{cell - 6}" height="{cell - 6}" rx="3" '
                                 f'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.4;stroke-dasharray:3 2.5"/>')
        path = p.get("path") or []
        if len(path) > 1:
            points = " ".join(f"{at(r, c, left, top)[0] + cell / 2:.1f},{at(r, c, left, top)[1] + cell / 2:.1f}" for r, c in path)
            parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-ink);stroke-width:{max(2.5, cell / 10):.1f};'
                         f'stroke-linecap:round;stroke-linejoin:round;opacity:0.85"/>')
        if p.get("mark") is not None:
            x, y = at(*p["mark"], left, top)
            parts.append(f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cell - 3}" height="{cell - 3}" style="fill:none;stroke:var(--ea-ink);stroke-width:2.6"/>')
        halo = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
        for r, row in enumerate(rows):
            for c, sign in enumerate(row):
                x, y = at(r, c, left, top)
                if sign in "SG":
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + size * 0.36:.1f}" style="font-size:{size + 1}px;'
                                 f'font-weight:700;fill:var(--ea-ink);text-anchor:middle;{halo}">{sign}</text>')
                elif (r, c) in numbers:
                    parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + size * 0.36:.1f}" style="font-size:{size}px;'
                                 f'font-weight:600;fill:{tone};text-anchor:middle;{halo}">{html.escape(str(numbers[(r, c)]))}</text>')
        # its smallest text is 11 px: the panel never shrinks, and in a narrow window it goes below the one before
        svgs.append(f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:1 1 0;min-width:min(100%,{W}px);'
                    f'max-width:{W}px;height:auto;{_FONT};font-variant-numeric:tabular-nums">{"".join(parts)}</svg>')
    gap = 16
    return mo.Html(f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;'
                   f'max-width:min(100%,{widest * len(panels) + gap * (len(panels) - 1)}px);display:flex;flex-wrap:wrap;'
                   f'gap:0.5rem {gap}px;align-items:flex-start">{"".join(svgs)}</figure>')


def _board(board: str, x: float, y: float, cell: float, new: int | None = None, index: bool = False) -> str:
    """One tic-tac-toe board of 3 by 3 squares, its top left corner at (x, y): X and O in ink, the square just played
    with a grey fill, and with `index` the number of each empty square in grey, as the code counts them."""
    parts, size = [], max(11, round(cell * 0.78))
    for i, sign in enumerate(board):
        sx, sy = x + (i % 3) * cell, y + (i // 3) * cell
        fill = "color-mix(in srgb, var(--ea-data) 30%, var(--ea-page))" if i == new else "var(--ea-page)"
        parts.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
        if sign in "XO":
            parts.append(f'<text x="{sx + cell / 2:.1f}" y="{sy + cell / 2 + size * 0.36:.1f}" style="font-size:{size}px;'
                         f'font-weight:600;fill:var(--ea-ink);text-anchor:middle">{sign}</text>')
        elif index:
            parts.append(f'<text x="{sx + cell / 2:.1f}" y="{sy + cell / 2 + 4:.1f}" style="font-size:11px;'
                         f'fill:var(--ea-ink-3);text-anchor:middle">{i}</text>')
    return "".join(parts)


def boards(panels: list[dict], *, alt: str, cell: int = 30) -> mo.Html:
    """Tic-tac-toe boards side by side; in a narrow window they go one below the other before their text gets too
    small, as maze_grids does. Each panel is a dict:
      board   9 signs, row by row: X, O, or '.' for an empty square
      title   the words above the board (optional)
      index   True: the number of each empty square in it, as the code counts them, 0 to 8 row by row
      new     the square just played, with a grey fill (optional)
    X and O are ink, the lines grey: no role colours. `alt` says what the figure shows, for a screen reader."""
    svgs, widest = [], 0
    for p in panels:
        top = 22 if p.get("title") else 1
        W, H = max(3 * cell + 2, 8 * len(p.get("title", ""))), top + 3 * cell + 2
        widest = max(widest, W)
        parts = []
        if p.get("title"):
            parts.append(f'<text x="1" y="15" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(p["title"])}</text>')
        parts.append(_board(p["board"], 1, top, cell, p.get("new"), p.get("index", False)))
        svgs.append(f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:0 1 auto;width:{W}px;max-width:100%;'
                    f'height:auto;{_FONT}">{"".join(parts)}</svg>')
    gap = 20
    return mo.Html(f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;'
                   f'max-width:min(100%,{widest * len(panels) + gap * (len(panels) - 1)}px);display:flex;flex-wrap:wrap;'
                   f'gap:0.5rem {gap}px;align-items:flex-start">{"".join(svgs)}</figure>')


def game_tree(nodes: list[dict], rows: list[str], *, alt: str, cell: int = 15, slot: int = 64) -> mo.Html:
    """A game tree, top down. Each node is a dict:
      parent  the index of its parent in `nodes`, or None for the top; a child comes after its parent, and the
              children of one node are drawn left to right in the order they come
      board   a position of tic-tac-toe, 9 signs row by row, drawn as a small board (optional)
      text    the words in a box, for a node without a board (optional)
      name    a small label above the node, such as the move that leads to it (optional)
      new     the square just played, with a grey fill (optional)
      value   the text under the node, such as "+1", in bold ink; "?" in grey, for a value not known yet
      ring    "mark": a thick ink ring, the node whose value just came back; "wait": a dashed ring, a call that waits
              for the values below it
      chosen  True: the line from its parent is thick, the move the player there takes
    Every node without children gets a slot of `slot` px of its own; a parent sits above the middle of its children.
    `rows` are the words on the left of each level, such as "X to move". Lines are grey, signs and values ink: no role
    colours. `alt` says what the figure shows, for a screen reader."""
    kids = {i: [] for i in range(len(nodes))}
    depth = {}
    for i, n in enumerate(nodes):
        if n.get("parent") is None:
            depth[i] = 0
        else:
            kids[n["parent"]].append(i)
            depth[i] = depth[n["parent"]] + 1
    xs, order = {}, []

    def place(i):
        if not kids[i]:
            xs[i] = len(order) * slot + slot / 2
            order.append(i)
        else:
            for k in kids[i]:
                place(k)
            xs[i] = (xs[kids[i][0]] + xs[kids[i][-1]]) / 2
    for i in range(len(nodes)):
        if depth[i] == 0:
            place(i)
    left = max(0, round(max((len(r) for r in rows), default=0) * 6.4)) + 10   # room for the words of each level
    shape = 3 * cell   # a board, or a box of the same width
    box_h = 22
    has_board = any(n.get("board") for n in nodes)
    node_h = shape if has_board else box_h
    named = any(n.get("name") for n in nodes)
    below = 22 if any(n.get("value") for n in nodes) else 4   # room under a node for its value, clear of a ring
    level_h = node_h + below + 28 + (14 if named else 0)
    top = 4 + (14 if named else 0)
    W = left + len(order) * slot
    H = top + (max(depth.values()) + 1) * level_h - 26
    parts = []

    def y_of(i):
        return top + depth[i] * level_h

    for d, words in enumerate(rows):
        parts.append(f'<text x="0" y="{top + d * level_h + node_h / 2 + 4:.1f}" style="font-size:11.5px;fill:var(--ea-ink-2)">'
                     f'{html.escape(words)}</text>')
    for i, n in enumerate(nodes):   # the lines first, so that the boards lie on top
        if n.get("parent") is None:
            continue
        p = n["parent"]
        x1, y1 = left + xs[p], y_of(p) + node_h + below
        x2, y2 = left + xs[i], y_of(i) - (14 if n.get("name") else 3)
        wide, ink = ("2.6", "var(--ea-ink)") if n.get("chosen") else ("1.2", "var(--ea-ink-3)")
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="stroke:{ink};stroke-width:{wide};stroke-linecap:round"/>')
    for i, n in enumerate(nodes):
        cx, y = left + xs[i], y_of(i)
        x = cx - shape / 2
        if n.get("board"):
            parts.append(_board(n["board"], x, y, cell, n.get("new")))
        else:
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{shape}" height="{box_h}" rx="4" style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1"/>')
            if n.get("text"):
                parts.append(f'<text x="{cx:.1f}" y="{y + box_h / 2 + 4.5:.1f}" style="font-size:13px;font-weight:600;'
                             f'fill:var(--ea-ink);text-anchor:middle">{html.escape(n["text"])}</text>')
        if n.get("name"):
            parts.append(f'<text x="{cx:.1f}" y="{y - 4:.1f}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:middle">'
                         f'{html.escape(n["name"])}</text>')
        ring = n.get("ring")
        if ring:
            style = ("stroke:var(--ea-ink);stroke-width:2.6" if ring == "mark"
                     else "stroke:var(--ea-ink-2);stroke-width:1.4;stroke-dasharray:3 2.5")
            parts.append(f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="{shape + 8}" height="{node_h + 8}" rx="4" style="fill:none;{style}"/>')
        value = n.get("value", "")
        if value:
            known = value != "?"
            look = "font-weight:700;fill:var(--ea-ink)" if known else "fill:var(--ea-ink-3)"
            parts.append(f'<text x="{cx:.1f}" y="{y + node_h + 19:.1f}" style="font-size:13px;{look};text-anchor:middle">'
                         f'{html.escape(value)}</text>')
    return mo.Html(f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;'
                   f'max-width:min(100%,{W}px)"><svg viewBox="0 0 {W} {H:.0f}" aria-hidden="true" style="width:100%;height:auto;'
                   f'{_FONT};font-variant-numeric:tabular-nums">{"".join(parts)}</svg></figure>')


# where a triangle of each action sits in its cell, as (x, y) from the middle, in parts of half a cell
_TOWARDS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def grid_world(panels: list[dict], *, alt: str, cell: int = 50, decimal: str = ".") -> mo.Html:
    """Grid worlds of reinforcement learning, one panel per world, side by side; in a narrow window they go one below
    the other before their text gets too small, as maze_grids does. Each panel is a dict:
      rows      the world as strings: '.' a cell to stand on, 'S' the start, 'G' the goal and 'X' a pit; the goal and a
                pit show their reward, +1 and −1, the start a small S in its corner
      title     the words above the panel (optional)
      values    {cell: number} written in the middle of the cell with two decimals, in the text colour of `role`
                ("pred" for the values of value iteration, "param" for what Q-learning learned)
      arrows    {cell: [actions]}: for each action ("up", "down", "left", "right") a small ink triangle at that side of
                the cell, the action of the policy there
      path      the cells of an episode, drawn as a line through each cell a little below its number; a step that
                stayed in its cell, into the edge, shows as a dot
      mark      one cell with a thick ink ring, such as where the agent is
      index     True: the row numbers on the left and the column numbers on top, as the code counts them
    Pits are data, a grey fill; the goal has a light grey one. `decimal` is the decimal mark: "," in Dutch. `alt` says
    what the figure shows, for a screen reader."""
    size = max(11, round(cell * 0.25))   # the text in a cell: a number of 4 signs leaves room for a triangle each side
    halo = "paint-order:stroke;stroke:var(--ea-page);stroke-width:3px;stroke-linejoin:round"
    svgs, widest = [], 0
    for p in panels:
        rows = p["rows"]
        left = 18 if p.get("index") else 1
        top = (24 if p.get("title") else 0) + (16 if p.get("index") else 1)
        W, H = left + len(rows[0]) * cell + 1, top + len(rows) * cell + 2
        widest = max(widest, W)
        values, arrows = p.get("values", {}), p.get("arrows", {})
        tone = f"var(--ea-{p['role']}-text)" if p.get("role") else "var(--ea-ink)"
        parts = []
        if p.get("title"):
            parts.append(f'<text x="1" y="15" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(p["title"])}</text>')
        if p.get("index"):
            for c in range(len(rows[0])):
                parts.append(f'<text x="{left + c * cell + cell / 2:.1f}" y="{top - 4}" style="font-size:11px;fill:var(--ea-ink-3);text-anchor:middle">{c}</text>')
            for r in range(len(rows)):
                parts.append(f'<text x="{left - 5}" y="{top + r * cell + cell / 2 + 4:.1f}" style="font-size:11px;fill:var(--ea-ink-3);text-anchor:end">{r}</text>')
        for r, row in enumerate(rows):
            for c, sign in enumerate(row):
                x, y = left + c * cell, top + r * cell
                fill = {"X": "var(--ea-data)", "G": "var(--ea-data-wash)"}.get(sign, "var(--ea-page)")
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
        path = p.get("path") or []
        low = cell * 0.28   # the way runs a little below the middle of each cell, so that it does not cross the number
        if len(path) > 1:
            points = " ".join(f"{left + c * cell + cell / 2:.1f},{top + r * cell + cell / 2 + low:.1f}" for r, c in path)
            parts.append(f'<polyline points="{points}" style="fill:none;stroke:var(--ea-ink);stroke-width:{max(2.0, cell / 18):.1f};'
                         f'stroke-linecap:round;stroke-linejoin:round;opacity:0.45"/>')
        for a, b in zip(path, path[1:]):
            if a == b:   # into the edge: the agent stayed where it was
                parts.append(f'<circle cx="{left + a[1] * cell + cell / 2:.1f}" cy="{top + a[0] * cell + cell / 2 + low:.1f}" r="{max(3.0, cell / 12):.1f}" '
                             f'style="fill:var(--ea-ink);opacity:0.45"/>')
        if p.get("mark") is not None:
            r, c = p["mark"]
            parts.append(f'<rect x="{left + c * cell + 1.5}" y="{top + r * cell + 1.5}" width="{cell - 3}" height="{cell - 3}" '
                         f'style="fill:none;stroke:var(--ea-ink);stroke-width:2.6"/>')
        for r, row in enumerate(rows):
            for c, sign in enumerate(row):
                x, y = left + c * cell, top + r * cell
                mx, my = x + cell / 2, y + cell / 2
                if sign in "GX":
                    words = "+1" if sign == "G" else "−1"
                    ink = "var(--ea-ink)" if sign == "G" else "var(--ea-page)"
                    parts.append(f'<text x="{mx:.1f}" y="{my + size * 0.36:.1f}" style="font-size:{size + 1}px;font-weight:700;'
                                 f'fill:{ink};text-anchor:middle">{words}</text>')
                    continue
                if sign == "S":
                    parts.append(f'<text x="{x + 4}" y="{y + 12}" style="font-size:11px;font-weight:700;fill:var(--ea-ink-2)">S</text>')
                for action in arrows.get((r, c), ()):
                    dx, dy = _TOWARDS[action]
                    tip = (mx + dx * (cell / 2 - 2), my + dy * (cell / 2 - 2))
                    base = (mx + dx * (cell / 2 - 8), my + dy * (cell / 2 - 8))
                    side = (dy * 4.5, dx * 4.5)
                    corners = [tip, (base[0] + side[0], base[1] + side[1]), (base[0] - side[0], base[1] - side[1])]
                    parts.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in corners)
                                 + '" style="fill:var(--ea-ink-2)"/>')
                if (r, c) in values:
                    text = f"{values[(r, c)]:.2f}"
                    if float(text) == 0:
                        text = text.lstrip("-")
                    text = text.replace("-", "−").replace(".", decimal)
                    parts.append(f'<text x="{mx:.1f}" y="{my + size * 0.36:.1f}" style="font-size:{size}px;font-weight:600;'
                                 f'fill:{tone};text-anchor:middle;{halo}">{text}</text>')
        # its smallest text is 11 px: the panel never shrinks, and in a narrow window it goes below the one before
        svgs.append(f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="flex:1 1 0;min-width:min(100%,{W}px);'
                    f'max-width:{W}px;height:auto;{_FONT};font-variant-numeric:tabular-nums">{"".join(parts)}</svg>')
    gap = 16
    return mo.Html(f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;'
                   f'max-width:min(100%,{widest * len(panels) + gap * (len(panels) - 1)}px);display:flex;flex-wrap:wrap;'
                   f'gap:0.5rem {gap}px;align-items:flex-start">{"".join(svgs)}</figure>')


def next_word_bars(prefix: str, options: list[tuple[str, float]], *, picked: int | None = 0, picked_label: str = "",
                   title: str = "") -> mo.Html:
    """What a language model sees after some text: a bar per possible next word, as long as how likely
    that word is. The bars are what the model puts out, so they are blue (prediction); the word it
    writes gets the full colour and a label."""
    W, LW, ROW, BW = 640, 120, 30, 300
    HEAD = 30 if title else 0        # the title, visible above the figure
    TOP = HEAD + 52
    top = max(share for _, share in options) or 1
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    parts.append(f'<text x="0" y="{HEAD + 22}" style="font-size:16px;font-style:italic;fill:var(--ea-ink)">{html.escape(prefix)} …</text>')
    for i, (word, share) in enumerate(options):
        y = TOP + i * ROW
        w = BW * share / top
        full = i == picked
        parts.append(f'<text x="{LW - 10}" y="{y + 15}" style="font-size:15px;font-weight:{650 if full else 450};'
                     f'fill:var(--ea-ink);text-anchor:end">{html.escape(word)}</text>'
                     f'<rect x="{LW}" y="{y + 2}" width="{max(w, 3):.1f}" height="18" rx="3" '
                     f'style="fill:var(--ea-pred);opacity:{1 if full else 0.45}"/>'
                     f'<text x="{LW + max(w, 3) + 8:.1f}" y="{y + 16}" style="font-size:13px;fill:var(--ea-pred-text)">'
                     f'{round(share * 100)}%</text>')
        if full and picked_label:
            parts.append(f'<text x="{LW + max(w, 3) + 52:.1f}" y="{y + 16}" style="font-size:13px;font-weight:600;'
                         f'fill:var(--ea-pred-text)">← {html.escape(picked_label)}</text>')
    H = TOP + len(options) * ROW + 4
    return _svg(W, H, title or prefix, "".join(parts))





def timeline(years: list[int], winters: list[tuple[int, int]], wave: list[tuple[int, float]], *,
             start: int, end: int, winter_label: str, wave_label: str, title: str = "") -> mo.Html:
    """A timeline of AI: a sketched line for how much hope and money there was (a sketch, not a measurement),
    grey bands for the winters, and numbered events on the axis; the lesson lists what each number is below
    the figure, so long labels never collide. Ink only: nothing here is a role of a model."""
    W, L, R = 640, 14, 540
    TOP = 30 if title else 6
    AX = TOP + 150                       # the time axis
    HI, LO = TOP + 30, TOP + 130         # where the wave is at its highest and lowest

    def x(year):
        return L + (year - start) / (end - start) * (R - L)

    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for a, b in winters:
        parts.append(f'<rect x="{x(a):.1f}" y="{TOP + 8}" width="{x(b) - x(a):.1f}" height="{AX - TOP - 8}" '
                     f'style="fill:var(--ea-ink);fill-opacity:0.08"/>'
                     f'<text x="{(x(a) + x(b)) / 2:.1f}" y="{TOP + 22}" style="font-size:12px;fill:var(--ea-ink-2);'
                     f'text-anchor:middle">{html.escape(winter_label)}</text>')
    pts = " ".join(f"{x(yr):.1f},{LO - (LO - HI) * v:.1f}" for yr, v in wave)
    parts.append(f'<polyline points="{pts}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:2;stroke-linejoin:round"/>')
    last_year, last_v = wave[-1]
    parts.append(f'<text x="{x(last_year) + 8:.1f}" y="{LO - (LO - HI) * last_v + 4:.1f}" style="font-size:12px;'
                 f'fill:var(--ea-ink-2)">{html.escape(wave_label)}</text>')
    parts.append(f'<line x1="{L}" y1="{AX}" x2="{R}" y2="{AX}" style="stroke:var(--ea-axis);stroke-width:1.5"/>')
    for decade in range((start // 10 + 1) * 10, end + 1, 10):
        parts.append(f'<text x="{x(decade):.1f}" y="{AX - 6}" style="font-size:11px;fill:var(--ea-ink-3);'
                     f'text-anchor:middle">{decade}</text>')
    for i, year in enumerate(years):
        ex = x(year)
        cy = AX + 18 + (i % 2) * 22        # neighbours alternate between two rows, so close years stay apart
        parts.append(f'<line x1="{ex:.1f}" y1="{AX}" x2="{ex:.1f}" y2="{cy - 9}" style="stroke:var(--ea-ink-3);stroke-width:1"/>'
                     f'<circle cx="{ex:.1f}" cy="{AX}" r="4" style="fill:var(--ea-ink)"/>'
                     f'<circle cx="{ex:.1f}" cy="{cy}" r="9" style="fill:var(--ea-page);stroke:var(--ea-ink);stroke-width:1.3"/>'
                     f'<text x="{ex:.1f}" y="{cy + 4}" style="font-size:11.5px;font-weight:650;fill:var(--ea-ink);'
                     f'text-anchor:middle">{i + 1}</text>')
    H = AX + 18 + 22 + 14
    return _svg(W, H, title or wave_label, "".join(parts))


def flow(nodes: list[tuple[str, list[str]]], forward: list[str], back: list[str], *, title: str = "") -> mo.Html:
    """Boxes in a row with arrows between neighbours: a forward arrow with its label above the middle,
    and, where `back` has a label, an arrow back with its label below. Labels may have line breaks (\\n).
    For schemes of who talks to whom; ink only, because nothing here is a role of a model."""
    W, GAP, BH = 720, 124, 104
    n = len(nodes)
    bw = (W - GAP * (n - 1)) / n
    TOP = 34 if title else 4
    mid = TOP + BH / 2

    def bx(i):
        return i * (bw + GAP)

    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for i, (name, lines) in enumerate(nodes):
        x0 = bx(i)
        parts.append(f'<rect x="{x0:.1f}" y="{TOP}" width="{bw:.1f}" height="{BH}" rx="8" '
                     f'style="fill:var(--ea-ink);fill-opacity:0.04;stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                     f'<text x="{x0 + bw / 2:.1f}" y="{TOP + 26}" style="font-size:14.5px;font-weight:650;fill:var(--ea-ink);'
                     f'text-anchor:middle">{html.escape(name)}</text>')
        for k, line in enumerate(lines):
            parts.append(f'<text x="{x0 + bw / 2:.1f}" y="{TOP + 48 + k * 15}" style="font-size:12px;fill:var(--ea-ink-2);'
                         f'text-anchor:middle">{html.escape(line)}</text>')
    arrow = "style=\"fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round\""
    for i in range(n - 1):
        a, b = bx(i) + bw + 6, bx(i + 1) - 6
        c = (a + b) / 2
        y = mid - 10
        parts.append(f'<path d="M{a:.1f} {y:.1f} H{b:.1f} M{b - 6:.1f} {y - 5:.1f} L{b:.1f} {y:.1f} L{b - 6:.1f} {y + 5:.1f}" {arrow}/>')
        for k, line in enumerate(reversed(forward[i].split("\n"))):
            parts.append(f'<text x="{c:.1f}" y="{y - 8 - k * 14:.1f}" style="font-size:12px;fill:var(--ea-ink);'
                         f'text-anchor:middle">{html.escape(line)}</text>')
        if i < len(back) and back[i]:
            y2 = mid + 10
            parts.append(f'<path d="M{b:.1f} {y2:.1f} H{a:.1f} M{a + 6:.1f} {y2 - 5:.1f} L{a:.1f} {y2:.1f} L{a + 6:.1f} {y2 + 5:.1f}" {arrow}/>')
            for k, line in enumerate(back[i].split("\n")):
                parts.append(f'<text x="{c:.1f}" y="{y2 + 18 + k * 14:.1f}" style="font-size:12px;fill:var(--ea-ink);'
                             f'text-anchor:middle">{html.escape(line)}</text>')
    H = TOP + BH + 18
    return _svg(W, H, title or " · ".join(name for name, _ in nodes), "".join(parts))



def program_trace(lines: list[str], shown: list[str], *, code_label: str, output_label: str,
                  title: str = "", names: list[dict] | None = None, names_label: str = "") -> mo.Html:
    """A program and what each line does, one row per line, read from top to bottom: the code,
    what that line adds under the cell (`shown[i]`; Python works it out), and with `names`, every
    name and its value after that line (`names[i]`), the new or changed one in ink, the rest grey.
    A fixed picture: everything is visible at once. Ink only: running code is not a role of a model."""
    GAP, LH, PAD = 20, 30, 12
    if names is None:
        W, cols = 640, [(0, 360), (380, 260)]
        labels = [code_label, output_label]
    else:
        W, cols = 680, [(0, 210), (230, 300), (550, 130)]
        labels = [code_label, names_label, output_label]
    TOP = (30 if title else 0) + 26
    mono = "font-family:var(--ea-font-code);font-size:13px;white-space:pre"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for (cx, _), label in zip(cols, labels):
        parts.append(f'<text x="{cx + PAD}" y="{TOP - 8}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink-2)">'
                     f'{html.escape(label)}</text>')
    box_h = len(lines) * LH
    parts.append(f'<rect x="{cols[0][0]}" y="{TOP}" width="{cols[0][1]}" height="{box_h}" rx="6" style="fill:var(--ea-inset)"/>')
    for cx, cw in cols[1:]:
        parts.append(f'<rect x="{cx}" y="{TOP}" width="{cw}" height="{box_h}" rx="6" '
                     f'style="fill:none;stroke:var(--ea-line);stroke-width:1.2"/>')
    before: dict = {}
    for i, line in enumerate(lines):
        y = TOP + i * LH + 20
        if i:
            parts.append(f'<line x1="0" y1="{TOP + i * LH}" x2="{W}" y2="{TOP + i * LH}" '
                         f'style="stroke:var(--ea-line);stroke-width:0.8;stroke-dasharray:3 3"/>')
        parts.append(f'<text x="{PAD}" y="{y}" style="{mono};fill:var(--ea-ink-3)">{i + 1}</text>'
                     f'<text x="{PAD + 22}" y="{y}" style="{mono};fill:var(--ea-ink)">{html.escape(line)}</text>')
        if names is not None:
            spans = []
            for name, value in names[i].items():
                new = name not in before or before[name] != value
                spans.append(f'<tspan style="fill:var(--ea-ink{"" if new else "-3"});font-weight:{600 if new else 400}">'
                             f'{html.escape(f"{name} = {value!r}")}</tspan>')
            parts.append(f'<text x="{cols[1][0] + PAD}" y="{y}" style="{mono}">{"<tspan>  </tspan>".join(spans)}</text>')
            before = dict(names[i])
        out = shown[i].rstrip("\n").splitlines()
        parts.append(f'<text x="{cols[-1][0] + PAD}" y="{y}" style="{mono};fill:var(--ea-ink)">'
                     f'{html.escape(" / ".join(out))}</text>')
    H = TOP + box_h + 2
    return _svg(W, H, title or code_label, "".join(parts))


def notebook_cells(panels, *, title: str = "") -> mo.Html:
    """Notebook cells as boxes, in panels side by side. A panel is (heading, cells, flows, verdict):
    `cells` the code of each cell (lines split with \\n), `flows` the name that goes from each cell to the
    next one (None for no arrow), `verdict` one line under the panel. Ink only: names are not a role."""
    W, GAP, LH, PAD, CELL_GAP = 720, 24, 20, 10, 34
    pw = (W - GAP * (len(panels) - 1)) / len(panels)
    TOP = (30 if title else 0) + 24
    mono = "font-family:var(--ea-font-code);font-size:13px;white-space:pre"
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    parts, bottom = [], TOP
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for p, (heading, cells, flows, verdict) in enumerate(panels):
        x = p * (pw + GAP)
        parts.append(f'<text x="{x:.1f}" y="{TOP - 8}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink-2)">'
                     f'{html.escape(heading)}</text>')
        y = TOP
        for c, code in enumerate(cells):
            lines = code.split("\n")
            h = len(lines) * LH + 2 * PAD - 4
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{pw:.1f}" height="{h}" rx="6" style="fill:var(--ea-inset)"/>')
            for k, line in enumerate(lines):
                parts.append(f'<text x="{x + PAD:.1f}" y="{y + PAD + 11 + k * LH}" style="{mono};fill:var(--ea-ink)">'
                             f'{html.escape(line)}</text>')
            y += h
            if c < len(cells) - 1:
                name = flows[c] if c < len(flows) else None
                if name:
                    cx = x + 28
                    parts.append(f'<path d="M{cx:.1f} {y + 4} V{y + CELL_GAP - 4} M{cx - 5:.1f} {y + CELL_GAP - 10} '
                                 f'L{cx:.1f} {y + CELL_GAP - 4} L{cx + 5:.1f} {y + CELL_GAP - 10}" {arrow}/>'
                                 f'<text x="{cx + 12:.1f}" y="{y + CELL_GAP / 2 + 5:.1f}" style="{mono};font-size:12.5px;'
                                 f'fill:var(--ea-ink-2)">{html.escape(name)}</text>')
                y += CELL_GAP
        parts.append(f'<text x="{x:.1f}" y="{y + 24}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(verdict)}</text>')
        bottom = max(bottom, y + 34)
    return _svg(W, bottom, title or panels[0][0], "".join(parts))


def string_boxes(text: str, *, index_label: str, back_label: str, title: str = "") -> mo.Html:
    """A string as a row of boxes, one character each, with its place counted from 0 above
    and counted back from the end below. A space gets a visible box too. Ink only."""
    BW, BH, GAP = 46, 46, 6
    n = len(text)
    LEFT = 110
    W = max(LEFT + n * (BW + GAP), 420)
    TOP = (30 if title else 0) + 26
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    parts.append(f'<text x="0" y="{TOP - 8}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(index_label)}</text>'
                 f'<text x="0" y="{TOP + BH + 26}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(back_label)}</text>')
    for i, ch in enumerate(text):
        x = LEFT + i * (BW + GAP)
        cx = x + BW / 2
        parts.append(f'<text x="{cx}" y="{TOP - 8}" style="{mono};font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{i}</text>'
                     f'<rect x="{x}" y="{TOP}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>'
                     f'<text x="{cx}" y="{TOP + BH / 2 + 7}" style="{mono};font-size:20px;fill:var(--ea-ink);text-anchor:middle;'
                     f'white-space:pre">{html.escape(ch if ch != " " else "␣")}</text>'
                     f'<text x="{cx}" y="{TOP + BH + 26}" style="{mono};font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{i - n}</text>')
    return _svg(W, TOP + BH + 36, title or text, "".join(parts))


def if_else(condition: str, yes: list[str], no: list[str], after: list[str], *, yes_label: str, no_label: str,
            title: str = "") -> mo.Html:
    """An if with an else as two paths: the line with the condition on top, the lines for True on the
    left and for False on the right, and the lines after the if at the bottom, where both paths meet.
    Ink only: a choice in a program is not a role of a model."""
    W, LH, PAD = 640, 20, 10
    TOP = (30 if title else 0) + 4
    mono = "font-family:var(--ea-font-code);font-size:13px;white-space:pre"
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')

    def box(x, y, w, lines):
        h = len(lines) * LH + 2 * PAD - 4
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
        for k, line in enumerate(lines):
            parts.append(f'<text x="{x + PAD}" y="{y + PAD + 11 + k * LH}" style="{mono};fill:var(--ea-ink)">{html.escape(line)}</text>')
        return h

    def down(x1, y1, x2, y2, label="", anchor="end"):
        parts.append(f'<path d="M{x1} {y1} L{x2} {y2 - 2}" {arrow}/>')
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / norm, dy / norm
        for sx in (1, -1):   # the two sides of the arrowhead
            px, py = x2 - 8 * ux + sx * 4.5 * uy, y2 - 2 - 8 * uy - sx * 4.5 * ux
            parts.append(f'<path d="M{px:.1f} {py:.1f} L{x2} {y2 - 2}" {arrow}/>')
        if label:
            lx = (x1 + x2) / 2 + (-10 if anchor == "end" else 10)
            parts.append(f'<text x="{lx:.1f}" y="{(y1 + y2) / 2 + 4:.1f}" style="font-size:13px;font-weight:600;'
                         f'fill:var(--ea-ink);text-anchor:{anchor}">{html.escape(label)}</text>')

    cw = 260
    ch = box((W - cw) / 2, TOP, cw, [condition])
    y2 = TOP + ch + 60
    bw = 270
    lh = box(20, y2, bw, yes)
    rh = box(W - 20 - bw, y2, bw, no)
    down(W / 2 - 40, TOP + ch, 20 + bw / 2, y2, yes_label, "end")
    down(W / 2 + 40, TOP + ch, W - 20 - bw / 2, y2, no_label, "start")
    y3 = y2 + max(lh, rh) + 50
    down(20 + bw / 2, y2 + lh, W / 2 - 40, y3)
    down(W - 20 - bw / 2, y2 + rh, W / 2 + 40, y3)
    ah = box((W - cw) / 2, y3, cw, after)
    return _svg(W, y3 + ah + 4, title or condition, "".join(parts))


def list_boxes(items: list[str], *, index_label: str, back_label: str = "", span: tuple[int, int] | None = None,
               span_label: str = "", title: str = "") -> mo.Html:
    """A list as a row of boxes, one item each, with its index above, and with `back_label` its index counted
    back from the end below. With `span` (start, stop), a bracket under that marks the slice from start up to,
    not including, stop. Ink only."""
    GAP = 6
    BW = max(46, 14 + 9 * max(len(s) for s in items))
    BH = 46
    LEFT = 110
    n = len(items)
    W = max(LEFT + n * (BW + GAP), 420)
    TOP = (30 if title else 0) + 26
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    parts.append(f'<text x="0" y="{TOP - 8}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(index_label)}</text>')
    for i, item in enumerate(items):
        x = LEFT + i * (BW + GAP)
        cx = x + BW / 2
        inside = span is not None and span[0] <= i < span[1]
        parts.append(f'<text x="{cx}" y="{TOP - 8}" style="{mono};font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{i}</text>'
                     f'<rect x="{x}" y="{TOP}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-inset);'
                     f'stroke:var(--ea-{"ink" if inside else "line"});stroke-width:{2 if inside else 1}"/>'
                     f'<text x="{cx}" y="{TOP + BH / 2 + 6}" style="{mono};font-size:17px;fill:var(--ea-ink);text-anchor:middle;'
                     f'white-space:pre">{html.escape(item)}</text>')
    H = TOP + BH + 8
    if back_label:
        parts.append(f'<text x="0" y="{TOP + BH + 22}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(back_label)}</text>')
        for i in range(n):
            cx = LEFT + i * (BW + GAP) + BW / 2
            parts.append(f'<text x="{cx}" y="{TOP + BH + 22}" style="{mono};font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">{i - n}</text>')
        H = TOP + BH + 30
    if span is not None:
        a = LEFT + span[0] * (BW + GAP)
        b = LEFT + span[1] * (BW + GAP) - GAP
        y = H + 2
        parts.append(f'<path d="M{a} {y} V{y + 8} H{b} V{y}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/>'
                     f'<text x="{(a + b) / 2}" y="{y + 28}" style="{mono};font-size:13.5px;fill:var(--ea-ink);'
                     f'text-anchor:middle">{html.escape(span_label)}</text>')
        H = y + 36
    return _svg(W, H, title or ", ".join(items), "".join(parts))


def array_grids(arrays: list[tuple], *, indices: bool = False, title: str = "", box_width: int = 46,
                block: int = 0) -> mo.Html:
    """Arrays side by side, each as a grid of boxes: one row of boxes per row of the array (a vector is one row).
    Each entry is (label, rows, captions) or (label, rows, captions, marked): its label above, its caption lines
    below, the first in code, such as its shape or what selects the marked boxes. `marked` is a set of
    (row, column): those boxes are drawn in ink, the others go grey. With `indices`, the column numbers stand
    above each grid and the row numbers to its left. `box_width` fits longer numbers, such as 0.4375. With `block`,
    a wider gap after every `block` rows and columns, so blocks of `block` by `block` stand apart, as in pooling.
    Ink only."""
    BW, BH, GAP, SPACE = box_width, 40, 4, 56
    apart = (lambda i: (i // block) * 10) if block else (lambda i: 0)   # the extra gap before row or column i
    LEFT = 22 if indices else 0
    TOP = (30 if title else 0) + 26 + (20 if indices else 0)
    mono = "font-family:var(--ea-font-code)"
    parts, x0, H = [], 1, 0
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for label, rows, captions, *rest in arrays:
        marked = rest[0] if rest else None
        cols = max(len(r) for r in rows)
        gx = x0 + LEFT
        w = LEFT + cols * (BW + GAP) - GAP + apart(cols - 1)
        parts.append(f'<text x="{x0}" y="{TOP - (30 if indices else 10)}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
        if indices:
            for j in range(cols):
                parts.append(f'<text x="{gx + j * (BW + GAP) + apart(j) + BW / 2}" y="{TOP - 8}" style="{mono};font-size:12px;fill:var(--ea-ink-2);'
                             f'text-anchor:middle">{j}</text>')
        for i, row in enumerate(rows):
            y = TOP + i * (BH + GAP) + apart(i)
            if indices:
                parts.append(f'<text x="{x0 + 6}" y="{y + BH / 2 + 4}" style="{mono};font-size:12px;fill:var(--ea-ink-2)">{i}</text>')
            for j, value in enumerate(row):
                x = gx + j * (BW + GAP) + apart(j)
                on = marked is None or (i, j) in marked
                stroke = "stroke:var(--ea-ink);stroke-width:2" if marked is not None and on else "stroke:var(--ea-line);stroke-width:1"
                ink = "var(--ea-ink)" if on else "var(--ea-ink-2)"
                weight = ";font-weight:600" if marked is not None and on else ""
                parts.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="5" style="fill:var(--ea-inset);{stroke}"/>'
                             f'<text x="{x + BW / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:15px;fill:{ink};text-anchor:middle{weight}">'
                             f'{html.escape(value)}</text>')
        y = TOP + len(rows) * (BH + GAP) + apart(len(rows) - 1) + 16
        for k, line in enumerate(captions):
            style = f"{mono};font-size:14px;fill:var(--ea-ink)" if k == 0 else "font-size:12.5px;fill:var(--ea-ink-2)"
            parts.append(f'<text x="{x0}" y="{y + 20 * k}" style="{style}">{html.escape(line)}</text>')
        width = max(w, 8 * max([len(label)] + [len(c) for c in captions]))
        H = max(H, y + 20 * (len(captions) - 1) + 8)
        x0 += width + SPACE
    # For a screen reader: the title, then each array with its numbers row by row, and its captions.
    alt = "; ".join(f"{a[0]}: {' / '.join(' '.join(r) for r in a[1])}, {', '.join(a[2])}" for a in arrays)
    return _svg(int(x0 - SPACE + 2), H, f"{title}. {alt}" if title else alt, "".join(parts), natural=True)


def axis_results(rows: list[list[str]], down: list[str], across: list[str], *, down_label: str, across_label: str,
                 title: str = "") -> mo.Html:
    """A matrix as a grid, with what an operation gives along each axis: under the grid, one result per column
    (`down`, axis 0, an arrow going down the rows), and to its right, one result per row (`across`, axis 1, an
    arrow going along each row). The results are in dashed boxes: they come out of the grid. Ink only."""
    BW, BH, GAP = 46, 40, 4
    cols = len(rows[0])
    TOP = (30 if title else 0) + 8
    mono = "font-family:var(--ea-font-code)"
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    gw = cols * (BW + GAP) - GAP
    gh = len(rows) * (BH + GAP) - GAP
    AX = gw + 44                       # where the results of each row stand
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')

    def box(x, y, value, dashed=False):
        style = "fill:none;stroke:var(--ea-ink);stroke-width:1.5;stroke-dasharray:4 3" if dashed else "fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"
        return (f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="5" style="{style}"/>'
                f'<text x="{x + BW / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:15px;fill:var(--ea-ink);text-anchor:middle">{html.escape(value)}</text>')

    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            parts.append(box(1 + j * (BW + GAP), TOP + i * (BH + GAP), value))
        y = TOP + i * (BH + GAP) + BH / 2
        parts.append(f'<path d="M{gw + 8} {y} H{AX - 8} M{AX - 13} {y - 4} L{AX - 8} {y} L{AX - 13} {y + 4}" {arrow}/>')
        parts.append(box(AX, TOP + i * (BH + GAP), across[i], dashed=True))
    parts.append(f'<text x="{AX + BW + 12}" y="{TOP + gh / 2 + 5}" style="{mono};font-size:13px;fill:var(--ea-ink)">{html.escape(across_label)}</text>')
    by = TOP + gh + 34
    for j in range(cols):
        x = 1 + j * (BW + GAP) + BW / 2
        parts.append(f'<path d="M{x} {TOP + gh + 6} V{by - 6} M{x - 4} {by - 11} L{x} {by - 6} L{x + 4} {by - 11}" {arrow}/>')
        parts.append(box(1 + j * (BW + GAP), by, down[j], dashed=True))
    parts.append(f'<text x="1" y="{by + BH + 22}" style="{mono};font-size:13px;fill:var(--ea-ink)">{html.escape(down_label)}</text>')
    W = max(AX + BW + 16 + 8 * len(across_label), 8 * len(down_label) + 8)
    H = by + BH + 30
    alt = f"{title}. " if title else ""
    alt += f"a matrix with rows {' / '.join(' '.join(r) for r in rows)}; {down_label}: {' '.join(down)}; {across_label}: {' '.join(across)}"
    return _svg(int(W), int(H), alt, "".join(parts), natural=True)


def data_tables(panels: list[tuple[str, list[str], list[list[str]], set, str]], *, title: str = "", alt: str = "") -> mo.Html:
    """Tables side by side, as a DataFrame shows: the column names above, the row labels to the left, and the
    values in boxes. Each panel is (label, column names, rows, marked, caption): the boxes in `marked`, a set of
    (row, column), are what a selection picks, in ink with a thick border; the rest goes gray. The caption, in
    code, says which selection. `alt` describes the picture for a screen reader, in the lesson's language. Ink only."""
    BH, GAP, SPACE, LEFT = 28, 3, 44, 18
    mono = "font-family:var(--ea-font-code)"
    TOP = (30 if title else 0) + 40
    parts, x0, H = [], 1, 0
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for label, header, rows, marked, caption in panels:
        widths = [max(40, 12 + 8 * max([len(header[j])] + [len(r[j]) for r in rows])) for j in range(len(header))]
        parts.append(f'<text x="{x0}" y="{TOP - 26}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
        x = x0 + LEFT
        for j, name in enumerate(header):
            on = any(c == j for _, c in marked)
            parts.append(f'<text x="{x + widths[j] / 2}" y="{TOP - 7}" style="{mono};font-size:12.5px;'
                         f'fill:var(--ea-{"ink" if on else "ink-2"});text-anchor:middle;font-weight:600">{html.escape(name)}</text>')
            x += widths[j] + GAP
        for i, row in enumerate(rows):
            y = TOP + i * (BH + GAP)
            kept = any(r == i for r, _ in marked) and len({c for _, c in marked}) > 1   # a row that is kept whole
            parts.append(f'<text x="{x0 + 2}" y="{y + BH / 2 + 4}" style="{mono};font-size:11.5px;fill:var(--ea-{"ink" if kept else "ink-3"})">{i}</text>')
            x = x0 + LEFT
            for j, value in enumerate(row):
                on = (i, j) in marked
                stroke = "stroke:var(--ea-ink);stroke-width:2" if on else "stroke:var(--ea-line);stroke-width:1"
                parts.append(f'<rect x="{x}" y="{y}" width="{widths[j]}" height="{BH}" rx="4" style="fill:var(--ea-inset);{stroke}"/>'
                             f'<text x="{x + widths[j] / 2}" y="{y + BH / 2 + 4.5}" style="{mono};font-size:13px;'
                             f'fill:var(--ea-{"ink" if on else "ink-2"});text-anchor:middle">{html.escape(value)}</text>')
                x += widths[j] + GAP
        y = TOP + len(rows) * (BH + GAP) + 18
        parts.append(f'<text x="{x0}" y="{y}" style="{mono};font-size:13.5px;fill:var(--ea-ink)">{html.escape(caption)}</text>')
        H = max(H, y + 8)
        w = LEFT + sum(widths) + GAP * (len(widths) - 1)
        x0 += max(w, 8 * len(caption)) + SPACE
    return _svg(int(x0 - SPACE + 2), int(H), alt or title, "".join(parts), natural=True)


def text_numbers(text: str, labels: tuple[str, str, str], *, title: str = "") -> mo.Html:
    """A text three times, one row each: its characters, the code point of each (`ord`), and its bytes in UTF-8.
    A character that takes more than one byte is as wide as its bytes, with a thick border, so you see which
    bytes belong to it. `labels` name the three rows, in the lesson's language. Ink only."""
    BW, BH, GAP, RG = 46, 34, 6, 14
    LEFT = 8 * max(len(label) for label in labels) + 16
    TOP = (30 if title else 0) + 4
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for r, label in enumerate(labels):
        parts.append(f'<text x="0" y="{TOP + r * (BH + RG) + BH / 2 + 4.5}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')

    def box(x, y, w, value, wide):
        stroke = "stroke:var(--ea-ink);stroke-width:2" if wide else "stroke:var(--ea-line);stroke-width:1"
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="5" style="fill:var(--ea-inset);{stroke}"/>'
                     f'<text x="{x + w / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:15px;fill:var(--ea-ink);'
                     f'text-anchor:middle;white-space:pre">{html.escape(value)}</text>')

    x, said = LEFT, []
    for ch in text:
        data = list(ch.encode("utf-8"))
        w = len(data) * (BW + GAP) - GAP
        wide = len(data) > 1
        box(x, TOP, w, ch, wide)
        box(x, TOP + BH + RG, w, str(ord(ch)), wide)
        for k, b in enumerate(data):
            box(x + k * (BW + GAP), TOP + 2 * (BH + RG), BW, str(b), wide)
        said.append(f"{ch}: {labels[1]} {ord(ch)}, {labels[2]} {' '.join(map(str, data))}")
        x += w + GAP
    H = TOP + 3 * (BH + RG) - RG + 4
    # For a screen reader: the title, then each character with its code point and its bytes.
    alt = "; ".join(said)
    return _svg(int(x - GAP + 2), int(H), f"{title}. {alt}" if title else alt, "".join(parts), natural=True)


def token_rows(rows: list[tuple[str, list[tuple[str, int | None]]]], *, title: str = "", alt: str = "") -> mo.Html:
    """Texts cut into tokens: for each text a label, then its tokens as boxes, with the id of each token under its
    box, or no ids in a row where every id is None. A space inside a token shows as ␣. Boxes wrap onto a new line where the
    row would be wider than the figure. Ink only: tokens are input."""
    W, GAP, BH, LINE = 380, 5, 30, 56
    mono = "font-family:var(--ea-font-code)"
    parts = []
    y = 0
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
        y = 30
    for label, tokens in rows:
        line = LINE if any(tid is not None for _, tid in tokens) else 40
        parts.append(f'<text x="0" y="{y + 14}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
        y += 22
        x = 0
        for piece, tid in tokens:
            shown = piece.replace(" ", "\u2423")
            bw = max(30, 12 + 9 * len(shown))
            if x + bw > W:
                x, y = 0, y + line
            parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{BH}" rx="5" style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"/>'
                         f'<text x="{x + bw / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:15px;fill:var(--ea-ink);'
                         f'text-anchor:middle;white-space:pre">{html.escape(shown)}</text>')
            if tid is not None:
                parts.append(f'<text x="{x + bw / 2}" y="{y + BH + 15}" style="{mono};font-size:12px;fill:var(--ea-ink-2);'
                             f'text-anchor:middle">{tid}</text>')
            x += bw + GAP
        y += line + 6
    return _svg(W, y - 4, alt or title, "".join(parts), natural=True)


def look_grids(panels: list[tuple[str, list[list[bool]]]], tokens: list[str], *, caption: str, title: str = "",
               alt: str = "") -> mo.Html:
    """Who looks at whom with attention, one grid per panel: row i is the token that looks, column j a token it may
    look at; a filled box means yes. Each panel is (its title, rows of True and False). The filled boxes are in the
    colour for values inside the model (hidden); the tokens and labels are ink."""
    n = len(tokens)
    CELL, GAP = 24, 3
    LEFT = 10 + 8 * max(len(t) for t in tokens)
    top = (30 if title else 0) + 22
    grid_top = top + 20 + 8 * max(len(t) for t in tokens)
    PW = LEFT + n * (CELL + GAP)
    SPACE = 22
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for p, (label, rows) in enumerate(panels):
        x0 = p * (PW + SPACE)
        parts.append(f'<text x="{x0}" y="{top}" style="font-size:13.5px;font-weight:600;fill:var(--ea-ink)">{html.escape(label)}</text>')
        for j, tok in enumerate(tokens):   # the columns: the tokens looked at, written upward
            cx = x0 + LEFT + j * (CELL + GAP) + CELL / 2 + 4
            parts.append(f'<text x="{cx}" y="{grid_top - 6}" transform="rotate(-90 {cx} {grid_top - 6})" '
                         f'style="{mono};font-size:12px;fill:var(--ea-ink-2)">{html.escape(tok)}</text>')
        for i, tok in enumerate(tokens):
            y = grid_top + i * (CELL + GAP)
            parts.append(f'<text x="{x0 + LEFT - 6}" y="{y + CELL / 2 + 4}" style="{mono};font-size:12px;fill:var(--ea-ink-2);'
                         f'text-anchor:end">{html.escape(tok)}</text>')
            for j in range(n):
                x = x0 + LEFT + j * (CELL + GAP)
                fill = "var(--ea-hidden)" if rows[i][j] else "var(--ea-inset)"
                stroke = "var(--ea-hidden)" if rows[i][j] else "var(--ea-line)"
                parts.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="3" style="fill:{fill};stroke:{stroke};stroke-width:1"/>')
    H = grid_top + n * (CELL + GAP) + 18
    parts.append(f'<text x="0" y="{H - 2}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(caption)}</text>')
    W = len(panels) * (PW + SPACE) - SPACE
    return _svg(int(W), int(H + 2), alt or title, "".join(parts), natural=True)


def hbars(rows: list[tuple[str, float]], *, top: float, title: str = "", note: str = "", alt: str = "") -> mo.Html:
    """One bar per row, from 0 to its value on a scale up to `top`, with its label above it and its value at its end.
    `note` stands under the bars. Ink only: for a measurement such as the number of right answers."""
    W, BAR, ROW = 300, 16, 44
    parts = []
    y = 0
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
        y = 30
    for label, value in rows:
        w = max(2.0, W * value / top)
        parts.append(f'<text x="0" y="{y + 13}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(label)}</text>'
                     f'<rect x="0" y="{y + 19}" width="{w:.1f}" height="{BAR}" rx="2" style="fill:var(--ea-ink)"/>'
                     f'<text x="{w + 6:.1f}" y="{y + 32}" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{value:g}</text>')
        y += ROW
    if note:
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(note)}</text>')
        y += 18
    return _svg(380, y + 4, alt or title, "".join(parts), natural=True)


def conv_grids(n: int, k: int, s: int, p: int, *, labels: dict) -> mo.Html:
    """A convolution on an n by n image with a frame of p zeros, a k by k filter and stride s. Left: the padded
    input, its frame of zeros grey; the filter at its first place, and dashed at its second, in the colour of what a
    model learns (param). Right: the output, with the boxes those two places fill in the colour for values inside the
    model (hidden). `labels` has "input", "output" and "caption", and may have "alt"."""
    m = n + 2 * p
    out = (n + 2 * p - k) // s + 1
    C, G = 22, 2
    TOP = 26
    parts = [f'<text x="0" y="16" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(labels["input"])}</text>']
    for r in range(m):
        for c in range(m):
            pad = r < p or c < p or r >= p + n or c >= p + n
            fill = "var(--ea-page)" if pad else "var(--ea-inset)"
            stroke = "var(--ea-line)" if pad else "var(--ea-ink-3)"
            dash = ";stroke-dasharray:2 2" if pad else ""
            parts.append(f'<rect x="{c * C}" y="{TOP + r * C}" width="{C - G}" height="{C - G}" rx="2" '
                         f'style="fill:{fill};stroke:{stroke};stroke-width:1{dash}"/>')
            if pad:
                parts.append(f'<text x="{c * C + (C - G) / 2}" y="{TOP + r * C + 14}" style="font-size:10px;fill:var(--ea-ink-3);'
                             f'text-anchor:middle">0</text>')
    for i, dashed in ((0, False), (1, True)):   # the filter at output (0, 0) and at (0, 1)
        if i >= out:
            continue
        x, y = i * s * C - 1, TOP - 1
        style = "fill:none;stroke:var(--ea-param);stroke-width:2.5" + (";stroke-dasharray:5 3" if dashed else "")
        parts.append(f'<rect x="{x}" y="{y}" width="{k * C - G + 2}" height="{k * C - G + 2}" rx="3" style="{style}"/>')
    x0 = m * C + 34
    parts.append(f'<text x="{x0}" y="16" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(labels["output"])}</text>')
    for r in range(out):
        for c in range(out):
            filled = r == 0 and c in (0, 1)
            fill = "var(--ea-hidden-wash)" if filled else "var(--ea-inset)"
            stroke = "var(--ea-hidden)" if filled else "var(--ea-ink-3)"
            parts.append(f'<rect x="{x0 + c * C}" y="{TOP + r * C}" width="{C - G}" height="{C - G}" rx="2" '
                         f'style="fill:{fill};stroke:{stroke};stroke-width:{1.5 if filled else 1}"/>')
    H = TOP + m * C + 22
    parts.append(f'<text x="0" y="{H - 4}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(labels["caption"])}</text>')
    W = max(x0 + out * C, 8 * len(labels["caption"])) + 2
    body = '<g transform="translate(2 0)">' + "".join(parts) + "</g>"   # room for the left edge of the first frame
    return _svg(int(W), int(H), labels.get("alt", labels["caption"]), body, natural=True)


def dict_boxes(pairs: list[tuple[str, str]], *, key_label: str, value_label: str, mark: int | None = None,
               mark_label: str = "", title: str = "") -> mo.Html:
    """A dictionary as rows of pairs: each key in a box on the left, an arrow, and its value in a box on the
    right. With `mark`, that row is drawn in ink, with `mark_label` beside it: the lookup. Ink only."""
    BH, GAP, AR = 40, 10, 64
    KW = max(96, 24 + 10 * max(len(k) for k, _ in pairs))
    VW = max(72, 24 + 10 * max(len(v) for _, v in pairs))
    VX = KW + AR
    TOP = (30 if title else 0) + 24
    mono = "font-family:var(--ea-font-code)"
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    W = max(VX + VW + (24 + 9 * len(mark_label) if mark_label else 0), 420)
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    parts.append(f'<text x="{KW / 2}" y="{TOP - 8}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(key_label)}</text>'
                 f'<text x="{VX + VW / 2}" y="{TOP - 8}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(value_label)}</text>')
    for i, (key, value) in enumerate(pairs):
        y = TOP + i * (BH + GAP)
        cy = y + BH / 2
        marked = i == mark
        stroke = f'stroke:var(--ea-{"ink" if marked else "line"});stroke-width:{2 if marked else 1}'
        for x, w, s in ((1, KW - 2, key), (VX, VW - 1, value)):
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="6" style="fill:var(--ea-inset);{stroke}"/>'
                         f'<text x="{x + w / 2}" y="{cy + 6}" style="{mono};font-size:16px;fill:var(--ea-ink);text-anchor:middle;'
                         f'white-space:pre">{html.escape(s)}</text>')
        x1, x2 = KW + 10, VX - 10
        parts.append(f'<path d="M{x1} {cy} H{x2}" {arrow}/><path d="M{x2 - 7} {cy - 4.5} L{x2} {cy} L{x2 - 7} {cy + 4.5}" {arrow}/>')
        if marked and mark_label:
            parts.append(f'<text x="{VX + VW + 16}" y="{cy + 5}" style="{mono};font-size:13.5px;fill:var(--ea-ink)">{html.escape(mark_label)}</text>')
    H = TOP + len(pairs) * (BH + GAP)
    return _svg(W, H, title or ", ".join(f"{k}: {v}" for k, v in pairs), "".join(parts))


def value_rows(rows: list[tuple[str, list[str], str, bool]], *, title: str = "") -> mo.Html:
    """The same values kept in several ways, one row each: the code on the left, the values in boxes,
    and a short note on the right. A row with `ordered` False (a set) has its boxes in a dashed bag,
    without places: it has no order. Ink only."""
    BW, BH, GAP, RG = 46, 40, 6, 18
    LEFT = 10 + 8 * max(len(code) for code, _, _, _ in rows) + 16
    n = max(len(items) for _, items, _, _ in rows)
    NOTE = LEFT + n * (BW + GAP) + 16
    W = max(NOTE + 8 * max(len(note) for _, _, note, _ in rows), 420)
    TOP = (30 if title else 0) + 8
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for r, (code, items, note, ordered) in enumerate(rows):
        y = TOP + r * (BH + RG)
        cy = y + BH / 2
        parts.append(f'<text x="0" y="{cy + 5}" style="{mono};font-size:13.5px;fill:var(--ea-ink);white-space:pre">{html.escape(code)}</text>')
        if not ordered:
            w = len(items) * (BW + GAP) - GAP + 12
            parts.append(f'<rect x="{LEFT - 6}" y="{y - 6}" width="{w}" height="{BH + 12}" rx="14" '
                         f'style="fill:none;stroke:var(--ea-ink-2);stroke-dasharray:4 4"/>')
        for i, item in enumerate(items):
            x = LEFT + i * (BW + GAP)
            parts.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>'
                         f'<text x="{x + BW / 2}" y="{cy + 6}" style="{mono};font-size:16px;fill:var(--ea-ink);text-anchor:middle;'
                         f'white-space:pre">{html.escape(item)}</text>')
        parts.append(f'<text x="{NOTE}" y="{cy + 5}" style="font-size:13px;fill:var(--ea-ink-2)">{html.escape(note)}</text>')
    H = TOP + len(rows) * (BH + RG) - RG + 8
    # For a screen reader: the title, then each row with its values and its note.
    alt = "; ".join(f"{code}: {' '.join(items)}" + (f", {note}" if note else "") for code, items, note, _ in rows)
    return _svg(W, H, f"{title}. {alt}" if title else alt, "".join(parts))


def comprehension_parts(loop_lines: list[tuple[str, str]], parts: list[tuple[str, str]], labels: dict[str, str], *,
                        loop_label: str, line_label: str, title: str = "") -> mo.Html:
    """A loop that builds a list, and the same list built in one line. `loop_lines` and `parts` are (code, mark)
    pairs; a mark such as "1" puts a numbered circle next to that loop line and under that part of the one-line
    version, and `labels` says what each number is. Ink only: a way of writing code is not a role of a model."""
    W, LH, PAD, FS = 640, 26, 12, 16
    CW = 0.6 * FS   # the code font is monospaced: every character is 0.6 em wide
    TOP = (30 if title else 0) + 4
    mono = "font-family:var(--ea-font-code)"
    parts_svg = []
    if title:
        parts_svg.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')

    def circle(cx, cy, mark):
        parts_svg.append(f'<circle cx="{cx}" cy="{cy}" r="10" style="fill:var(--ea-page);stroke:var(--ea-ink-2)"/>'
                         f'<text x="{cx}" y="{cy + 4.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(mark)}</text>')

    # the loop, in a box, with the numbers on the right
    parts_svg.append(f'<text x="0" y="{TOP + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(loop_label)}</text>')
    by = TOP + 22
    bw = max(len(code) for code, _ in loop_lines) * 8.4 + 2 * PAD + 40
    bh = len(loop_lines) * LH + 2 * PAD - 6
    parts_svg.append(f'<rect x="0" y="{by}" width="{bw}" height="{bh}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
    for k, (code, mark) in enumerate(loop_lines):
        y = by + PAD + 10 + k * LH
        parts_svg.append(f'<text x="{PAD}" y="{y}" style="{mono};font-size:14px;fill:var(--ea-ink);white-space:pre" '
                         f'xml:space="preserve">{html.escape(code)}</text>')
        if mark:
            circle(bw - 18, y - 5, mark)
    # the same list in one line, each part placed by its characters, with a bracket and a number under it
    ly = by + bh + 34
    parts_svg.append(f'<text x="0" y="{ly}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(line_label)}</text>')
    cy = ly + 30
    x = 0.0
    for code, mark in parts:
        w = len(code) * CW
        if code.strip():
            parts_svg.append(f'<text x="{x:.1f}" y="{cy}" textLength="{w:.1f}" lengthAdjust="spacingAndGlyphs" '
                             f'style="{mono};font-size:{FS}px;fill:var(--ea-ink);white-space:pre" xml:space="preserve">{html.escape(code)}</text>')
        if mark:
            y0 = cy + 8
            parts_svg.append(f'<path d="M{x + 2:.1f} {y0} V{y0 + 6} H{x + w - 2:.1f} V{y0}" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/>')
            circle(x + w / 2, y0 + 20, mark)
        x += w
    # what each number means
    gy = cy + 62
    for k, (mark, text) in enumerate(labels.items()):
        y = gy + k * 26
        circle(10, y - 4, mark)
        parts_svg.append(f'<text x="28" y="{y}" style="font-size:13.5px;fill:var(--ea-ink)">{html.escape(text)}</text>')
    H = gy + (len(labels) - 1) * 26 + 12
    return _svg(W, H, title or "".join(code for code, _ in parts), "".join(parts_svg))


def marked_lines(blocks: list[tuple[str, list[tuple[str, str]]]], labels: dict[str, str], *, title: str = "") -> mo.Html:
    """Code boxes, each with a label above it, such as a cell and the traceback under it. A line with a mark
    such as "1" gets a numbered circle on its right, and `labels` says what each number means. Ink only."""
    LH, PAD, CW = 22, 12, 8.4   # code at 14 px: every character is 0.6 em wide
    width = max(len(code) for _, lines in blocks for code, _ in lines) * CW + 2 * PAD + 40
    W = max(width, 600)   # room on the right, so short code is not drawn larger than other figures
    TOP = (30 if title else 0) + 4
    mono = "font-family:var(--ea-font-code);font-size:14px;white-space:pre"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')

    def circle(cx, cy, mark):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="9.5" style="fill:var(--ea-page);stroke:var(--ea-ink-2)"/>'
                     f'<text x="{cx}" y="{cy + 4.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(mark)}</text>')

    y = TOP
    for label, lines in blocks:
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(label)}</text>')
        by = y + 20
        bh = len(lines) * LH + 2 * PAD - 6
        parts.append(f'<rect x="0" y="{by}" width="{width}" height="{bh}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
        for k, (code, mark) in enumerate(lines):
            ly = by + PAD + 11 + k * LH
            parts.append(f'<text x="{PAD}" y="{ly}" style="{mono};fill:var(--ea-ink)" xml:space="preserve">{html.escape(code)}</text>')
            if mark:
                circle(width - 18, ly - 5, mark)
        y = by + bh + 16
    for k, (mark, text) in enumerate(labels.items()):
        ly = y + 14 + k * 26
        circle(10, ly - 4, mark)
        parts.append(f'<text x="28" y="{ly}" style="font-size:13.5px;fill:var(--ea-ink)">{html.escape(text)}</text>')
    H = y + 14 + (len(labels) - 1) * 26 + 10
    return _svg(int(W), int(H), title or label, "".join(parts))


def class_objects(class_lines: list[str], objects: list[tuple[str, list[str]]], *, class_label: str, objects_label: str,
                  title: str = "") -> mo.Html:
    """A class as one box (the plan: its name and its methods), and below it the objects made from it, each a box
    with its own data. Arrows run from the class to each object. Ink only."""
    W, LH, PAD = 640, 20, 10
    mono = "font-family:var(--ea-font-code);font-size:13px;white-space:pre"
    arrow = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    TOP = (30 if title else 0) + 4
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')

    def box(x, y, w, lines, bold_first=False):
        h = len(lines) * LH + 2 * PAD - 4
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
        for k, line in enumerate(lines):
            weight = ";font-weight:600" if bold_first and k == 0 else ""
            parts.append(f'<text x="{x + PAD}" y="{y + PAD + 11 + k * LH}" style="{mono};fill:var(--ea-ink){weight}" '
                         f'xml:space="preserve">{html.escape(line)}</text>')
        return h

    cw = max(220, 16 + 8 * max(len(line) for line in class_lines))
    cx = (W - cw) / 2
    parts.append(f'<text x="{cx}" y="{TOP + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(class_label)}</text>')
    ch = box(cx, TOP + 20, cw, class_lines, bold_first=True)
    y2 = TOP + 20 + ch + 70
    n = len(objects)
    gap = 20
    ow = (W - gap * (n - 1)) / n
    parts.append(f'<text x="0" y="{y2 - 10}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(objects_label)}</text>')
    heights = []
    for i, (head, lines) in enumerate(objects):
        ox = i * (ow + gap)
        heights.append(box(ox, y2, ow, [head] + lines, bold_first=True))
        x1, y1, x2 = W / 2 + (i - (n - 1) / 2) * 40, TOP + 20 + ch, ox + ow / 2
        # the arrows end above the label of the objects, so they never touch it
        parts.append(f'<path d="M{x1} {y1} L{x2} {y2 - 30}" {arrow}/>'
                     f'<path d="M{x2 - 4.5} {y2 - 38} L{x2} {y2 - 30} L{x2 + 4.5} {y2 - 38}" {arrow}/>')
    H = y2 + max(heights) + 4
    return _svg(W, int(H), title or class_lines[0], "".join(parts))


def project_steps(steps: list[str], current: int, *, here_label: str, title: str = "") -> mo.Html:
    """Where a project stands: its steps as a column of boxes, top to bottom, joined by arrows. The step of today
    is in ink, filled, with `here_label` beside it; the other steps are grey. Ink only. The figure is as wide as
    its longest line, and no wider."""
    BH, GAP = 34, 18
    BW = max(300, 24 + 8 * max(len(s) for s in steps))
    TOP = (30 if title else 0) + 4
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    for i, step in enumerate(steps):
        y = TOP + i * (BH + GAP)
        now = i == current
        stroke = "stroke:var(--ea-ink);stroke-width:2" if now else "stroke:var(--ea-line);stroke-width:1"
        fill = "var(--ea-inset)" if now else "none"
        ink = "var(--ea-ink)" if now else "var(--ea-ink-2)"
        weight = ";font-weight:600" if now else ""
        parts.append(f'<rect x="1" y="{y}" width="{BW}" height="{BH}" rx="6" style="fill:{fill};{stroke}"/>'
                     f'<text x="14" y="{y + BH / 2 + 5}" style="font-size:13.5px;fill:{ink}{weight}">{html.escape(step)}</text>')
        if now:
            parts.append(f'<text x="{BW + 14}" y="{y + BH / 2 + 5}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(here_label)}</text>')
        if i < len(steps) - 1:
            x, y1, y2 = 24, y + BH + 3, y + BH + GAP - 3
            parts.append(f'<path d="M{x} {y1} V{y2} M{x - 3.5} {y2 - 4} L{x} {y2} L{x + 3.5} {y2 - 4}" {arrow}/>')
    H = TOP + len(steps) * (BH + GAP) - GAP + 4
    W = max(BW + 24 + 8 * len(here_label), 9 * len(title))
    return _svg(W, H, title or " · ".join(steps), "".join(parts), natural=True)


def computational_graph(nodes: list[tuple], edges: list[tuple], *, title: str = "", alt: str = "",
                        down: bool = False, later: dict[str, str] | None = None) -> mo.Html:
    """A calculation as a graph, left to right: one box per value, with what it is above its number, and arrows from
    the values that go into it. Each node is (key, column, row, label, value, gradient, role): `gradient` is the
    text of its slope, drawn below the value in the grad colour, or None while the backward pass has not reached it;
    `role` colours the label by the style guide (param, pred, loss and so on), or None for ink. Each edge is (from, to,
    local): `local` is the local slope of the edge, written next to it in the grad colour, or None. With `down`, the
    graph runs from top to bottom, a column of nodes becoming a row, at the size it was drawn: a calculation of many
    steps then stays readable in a narrow column, such as a play block at 700 px. `later` gives a text in grey for a
    node the backward pass has not reached yet, in the place of its slope, such as the step at which it comes."""
    BW, BH, CW, RH = 124, 70, 184, 96
    TOP = (30 if title else 0) + 6
    mono = "font-family:var(--ea-font-code)"
    if down:   # a column of the graph is a row here, 116 px apart, and its rows sit 150 px apart side by side
        at = {key: (row * 150 + 1, TOP + col * 116) for key, col, row, *_ in nodes}
    else:
        at = {key: (col * CW + 1, TOP + row * RH) for key, col, row, *_ in nodes}
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    arrow = 'style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"'
    for a, b, local in edges:
        (ax, ay), (bx, by) = at[a], at[b]
        if down:
            x1, y1, x2, y2 = ax + BW / 2, ay + BH, bx + BW / 2, by - 3
        else:
            x1, y1, x2, y2 = ax + BW, ay + BH / 2, bx - 3, by + BH / 2
        dx, dy = x2 - x1, y2 - y1
        d = max((dx * dx + dy * dy) ** 0.5, 1)
        ux, uy = dx / d, dy / d
        parts.append(f'<path d="M{x1:.1f} {y1:.1f} L{x2:.1f} {y2:.1f} M{x2 - 7 * ux - 4 * uy:.1f} {y2 - 7 * uy + 4 * ux:.1f} '
                     f'L{x2:.1f} {y2:.1f} L{x2 - 7 * ux + 4 * uy:.1f} {y2 - 7 * uy - 4 * ux:.1f}" {arrow}/>')
        if local and down:
            # in the free row between two rows of boxes: beside the middle of the arrow, on the side it does not lean to
            cx, cy = x1 + 0.5 * dx, y1 + 0.5 * dy
            side = -1 if dx > 1 else 1
            off = 8 + 12 * abs(dx) / max(abs(dy), 1)   # a sloping arrow passes the text's height further out: keep clear of it
            parts.append(f'<text x="{cx + side * off:.1f}" y="{cy + 4:.1f}" style="font-size:12px;fill:var(--ea-grad-text);'
                         f'text-anchor:{"end" if side < 0 else "start"}">{html.escape(local)}</text>')
        elif local:
            # beside the middle of the arrow, on its upper side, far enough that a sloping line never runs through the text
            nx, ny = uy, -ux
            far = abs(nx) * 3.6 * len(local) + abs(ny) * 5 + 4
            cx, cy = x1 + 0.5 * dx + nx * far, y1 + 0.5 * dy + ny * far
            parts.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" style="font-size:12px;fill:var(--ea-grad-text);'
                         f'text-anchor:middle">{html.escape(local)}</text>')
    for key, col, row, label, value, gradient, role in nodes:
        x, y = at[key]
        tone = f"var(--ea-{role}-text)" if role else "var(--ea-ink)"
        parts.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="8" style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1.2"/>'
                     f'<text x="{x + BW / 2}" y="{y + 20}" style="font-size:13.5px;font-weight:600;fill:{tone};text-anchor:middle">{html.escape(label)}</text>'
                     f'<text x="{x + BW / 2}" y="{y + 41}" style="{mono};font-size:15px;fill:var(--ea-ink);text-anchor:middle">{html.escape(value)}</text>')
        if gradient:
            parts.append(f'<text x="{x + BW / 2}" y="{y + 61}" style="font-size:12.5px;fill:var(--ea-grad-text);text-anchor:middle">{html.escape(gradient)}</text>')
        elif later and key in later:
            parts.append(f'<text x="{x + BW / 2}" y="{y + 61}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(later[key])}</text>')
    width = max(x for x, _ in at.values()) + BW + 1
    height = max(y for _, y in at.values()) + BH + 4
    return _svg(int(width), int(height), f"{title}. {alt}" if title else alt, "".join(parts), natural=down)


def network_diagram(sizes: list[int], captions: list[str], *, title: str = "", alt: str = "") -> mo.Html:
    """A network drawn as columns of circles, one column per layer and one circle per neuron, with a line from every
    circle to every circle of the next column. The inputs are grey (data), the hidden layers in the colour for values
    inside the model (hidden), the outputs blue (pred). `captions` stand under the columns, one per column."""
    assert len(captions) == len(sizes), "one caption per column"
    CW, R, GAP = 150, 13, 40
    TOP = (30 if title else 0) + 12
    tallest = max(sizes)
    height = TOP + tallest * GAP + 46
    xs = [40 + i * CW for i in range(len(sizes))]

    def ys(n: int) -> list[float]:
        top = TOP + (tallest - n) * GAP / 2 + GAP / 2
        return [top + k * GAP for k in range(n)]

    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for i in range(len(sizes) - 1):
        for y1 in ys(sizes[i]):
            for y2 in ys(sizes[i + 1]):
                parts.append(f'<line x1="{xs[i] + R}" y1="{y1:.1f}" x2="{xs[i + 1] - R}" y2="{y2:.1f}" style="stroke:var(--ea-ink-3);stroke-width:0.9"/>')
    for i, n in enumerate(sizes):
        role = "data" if i == 0 else "pred" if i == len(sizes) - 1 else "hidden"
        for y in ys(n):
            parts.append(f'<circle cx="{xs[i]}" cy="{y:.1f}" r="{R}" style="fill:var(--ea-{role}-wash);stroke:var(--ea-{role});stroke-width:1.8"/>')
        parts.append(f'<text x="{xs[i]}" y="{height - 12}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(captions[i])}</text>')
    width = xs[-1] + 80
    return _svg(int(width), int(height), f"{title}. {alt}" if title else alt, "".join(parts), natural=True)


def feature_maps(stages: list[tuple[str, str, int, int | None, str]], steps: list[str], *, title: str = "",
                 alt: str = "") -> mo.Html:
    """What one image looks like on its way through a CNN, from top to bottom. Each stage is a stack of squares, one
    per channel (at most 6 drawn), as wide as its height and width; a vector, such as the logits, is a row of small
    boxes (at most 10 drawn: a longer one has dots in the second-to-last place). Beside each stage its shape and a note; between two stages an arrow with the step that leads from one to
    the next. `stages` holds (shape, note, channels, size, role), with size None for a vector and role "data" (the
    input, grey), "hidden" (values inside the model) or "pred" (the output); `steps` has one label per arrow."""
    assert len(steps) == len(stages) - 1, "one step between every two stages"
    SCALE, OFF, MOST, ARROW, BOX, ROW = 2, 5, 6, 30, 7, 10
    widest = max(size for _, _, _, size, _ in stages if size is not None)
    TEXT_X = 2 + widest * SCALE + (MOST - 1) * OFF + 22
    mid = 2 + widest * SCALE / 2
    mono = "font-family:var(--ea-font-code)"
    y = (30 if title else 0) + 4
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    texts = []
    for i, (shape, note, channels, size, role) in enumerate(stages):
        fill = f"fill:var(--ea-{role}-wash);stroke:var(--ea-{role});stroke-width:1.5"
        if size is None:
            for j in range(min(channels, ROW)):   # a longer vector: dots in the second-to-last place
                if channels > ROW and j == ROW - 2:
                    parts.append(f'<text x="{2 + j * (BOX + 2) + BOX / 2}" y="{y + 12 + BOX}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">…</text>')
                else:
                    parts.append(f'<rect x="{2 + j * (BOX + 2)}" y="{y + 12}" width="{BOX}" height="{BOX}" rx="1.5" style="{fill}"/>')
            height = BOX + 24
        else:
            n, side = min(channels, MOST), size * SCALE
            for j in range(n - 1, -1, -1):   # the back square first: each next one sits lower and to the left
                parts.append(f'<rect x="{2 + j * OFF}" y="{y + (n - 1 - j) * OFF}" width="{side}" height="{side}" rx="2" style="{fill}"/>')
            height = side + (n - 1) * OFF
        middle = y + max(height, 36) / 2
        texts += [shape, note]
        parts.append(f'<text x="{TEXT_X}" y="{middle - 3:.1f}" style="{mono};font-size:14px;fill:var(--ea-ink)">{html.escape(shape)}</text>'
                     f'<text x="{TEXT_X}" y="{middle + 14:.1f}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(note)}</text>')
        y += max(height, 36)
        if i < len(steps):
            texts.append(steps[i])
            parts.append(f'<line x1="{mid}" y1="{y + 5}" x2="{mid}" y2="{y + ARROW - 5}" style="stroke:var(--ea-ink-3);stroke-width:1.5"/>'
                         f'<path d="M{mid - 4} {y + ARROW - 10} L{mid} {y + ARROW - 5} L{mid + 4} {y + ARROW - 10}" style="fill:none;stroke:var(--ea-ink-3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round"/>'
                         f'<text x="{TEXT_X}" y="{y + ARROW / 2 + 4}" style="font-size:12.5px;fill:var(--ea-ink-2);font-style:italic">{html.escape(steps[i])}</text>')
            y += ARROW
    width = TEXT_X + 7.5 * max(len(t) for t in texts) + 4
    return _svg(int(width), int(y + 4), f"{title}. {alt}" if title else alt, "".join(parts), natural=True)


def weight_grid(words: list[str], weights, *, row_label: str, col_label: str, decimal: str = ".", title: str = "",
                alt: str = "") -> mo.Html:
    """A square matrix of weights, such as attention weights: one row and one column per word, the words beside and
    above. Every box shows its weight with two decimals and is filled in the colour for values inside the model
    (hidden), the stronger the larger the weight, so that a row reads as how much its word takes from each word.
    `row_label` stands above the words on the left, `col_label` above the columns."""
    n = len(words)
    BW, BH, GAP = 54, 36, 4
    LEFT = 12 + 8 * max(len(w) for w in words) + 8
    TOP = (30 if title else 0) + 40
    mono = "font-family:var(--ea-font-code)"
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    parts.append(f'<text x="0" y="{TOP - 24}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(row_label)}</text>'
                 f'<text x="{LEFT}" y="{TOP - 24}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(col_label)}</text>')
    for j, word in enumerate(words):
        parts.append(f'<text x="{LEFT + j * (BW + GAP) + BW / 2}" y="{TOP - 7}" style="font-size:13px;fill:var(--ea-ink);text-anchor:middle">{html.escape(word)}</text>')
    for i, word in enumerate(words):
        y = TOP + i * (BH + GAP)
        parts.append(f'<text x="0" y="{y + BH / 2 + 4}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(word)}</text>')
        for j in range(n):
            w = float(weights[i][j])
            x = LEFT + j * (BW + GAP)
            parts.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="4" style="fill:var(--ea-hidden);fill-opacity:{0.06 + 0.6 * w:.3f};stroke:var(--ea-line);stroke-width:1"/>'
                         f'<text x="{x + BW / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:14px;fill:var(--ea-ink);text-anchor:middle">{f"{w:.2f}".replace(".", decimal)}</text>')
    width = max(LEFT + n * (BW + GAP), 8 * len(col_label) + LEFT, 8 * len(title))
    return _svg(int(width), int(TOP + n * (BH + GAP) + 4), f"{title}. {alt}" if title else alt, "".join(parts), natural=True)
