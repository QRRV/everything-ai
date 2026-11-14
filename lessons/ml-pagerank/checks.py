# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-pagerank: the karate club of Zachary (1977), 34 members and 78 pairs of friends, with four
centralities for every member (degree, closeness, betweenness and PageRank) and a picture of the club with every
member as large as their centrality; a small made-up web of pages with links, for PageRank with directions; and the
functions closeness and pagerank."""
import contextlib
import functools
import html
import io
import math
import warnings

from everything_ai import check as checking, yours
from everything_ai.check import FunctionBox, Mistake

# The 78 pairs of members who also met outside the club, members 0 to 33 (Zachary 1977, as networkx gives them).
EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17),
         (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7),
         (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10),
         (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32),
         (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
         (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33),
         (30, 32), (30, 33), (31, 32), (31, 33), (32, 33)]
MEMBERS = 34
# The members who went with the instructor, member 0, when the club split; the other 17 went with the administrator, 33.
INSTRUCTOR_SIDE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21}
D = 0.85   # the chance that the random walker follows an edge; else it jumps to a random node

# Where every member is drawn, from 0 to 1 across and down: the layout of Kamada and Kawai, made once with networkx.
AT = {0: (0.49, 0.62), 1: (0.32, 0.57), 2: (0.54, 0.43), 3: (0.61, 0.56), 4: (0.65, 0.76), 5: (0.64, 0.82),
      6: (0.46, 0.82), 7: (0.73, 0.55), 8: (0.30, 0.43), 9: (0.53, 0.27), 10: (0.82, 0.73), 11: (0.29, 0.79),
      12: (0.87, 0.65), 13: (0.39, 0.46), 14: (0.00, 0.25), 15: (0.03, 0.19), 16: (0.62, 1.00), 17: (0.21, 0.72),
      18: (0.10, 0.13), 19: (0.21, 0.47), 20: (0.19, 0.08), 21: (0.12, 0.68), 22: (0.32, 0.07), 23: (0.69, 0.14),
      24: (1.00, 0.26), 25: (0.98, 0.34), 26: (0.42, 0.00), 27: (0.73, 0.27), 28: (0.64, 0.26), 29: (0.53, 0.08),
      30: (0.11, 0.38), 31: (0.65, 0.39), 32: (0.25, 0.27), 33: (0.40, 0.20)}

# A small made-up web: every page and the pages it links to. Everyone links to A; A links only to X; D and E, which no
# page links to, also link to Y.
WEB = {"A": {"X"}, "B": {"A"}, "C": {"A"}, "D": {"A", "Y"}, "E": {"A", "Y"}, "F": {"A"}, "X": {"A"}, "Y": {"A"}}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def club() -> dict:
    """The club as a graph without directions, made anew at every call, as the worked example makes it: every member
    and the list of their friends. Every pair is in EDGES once, so a list holds no friend twice."""
    out = {m: [] for m in range(MEMBERS)}
    for a, b in EDGES:
        out[a].append(b)
        out[b].append(a)
    return out


def steps_from(graph, start) -> dict:
    """The fewest steps from start to every node it can reach, by breadth-first search: flights_from of
    ml-network-data."""
    from collections import deque
    steps = {start: 0}
    frontier = deque([start])
    while frontier:
        node = frontier.popleft()
        for nxt in graph[node]:
            if nxt not in steps:
                steps[nxt] = steps[node] + 1
                frontier.append(nxt)
    return steps


def closeness(graph, node) -> float:
    """The course's closeness: the number of other nodes node reaches, divided by the sum of the steps to them."""
    steps = steps_from(graph, node)
    total = 0
    for other in steps:
        total = total + steps[other]
    return (len(steps) - 1) / total


def betweenness(graph) -> dict:
    """For every node, on how many shortest paths between two other nodes it lies, a path counting for its share when
    two nodes have several shortest paths; every pair once (Brandes 2001, without directions)."""
    from collections import deque
    out = {v: 0.0 for v in graph}
    for s in graph:
        order, before = [], {v: [] for v in graph}
        paths, dist = {v: 0 for v in graph}, {s: 0}
        paths[s] = 1
        frontier = deque([s])
        while frontier:
            v = frontier.popleft()
            order.append(v)
            for w in graph[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    frontier.append(w)
                if dist[w] == dist[v] + 1:
                    paths[w] += paths[v]
                    before[w].append(v)
        share = {v: 0.0 for v in graph}
        for w in reversed(order):
            for v in before[w]:
                share[v] += paths[v] / paths[w] * (1 + share[w])
            if w != s:
                out[w] += share[w]
    return {v: x / 2 for v, x in out.items()}   # every pair was counted from both ends


def pagerank(graph, d, rounds) -> dict:
    """The course's pagerank: every node starts at 1/n; every round, every node passes d times its score on, in equal
    parts, to the nodes in graph[node], and every node gets (1 - d) / n for the jumps."""
    n = len(graph)
    scores = {node: 1 / n for node in graph}
    for _ in range(rounds):
        new = {node: (1 - d) / n for node in graph}
        for node in graph:
            share = d * scores[node] / len(graph[node])
            for nxt in graph[node]:
                new[nxt] = new[nxt] + share
        scores = new
    return scores


@functools.lru_cache(maxsize=1)
def measures() -> dict:
    """The four centralities of every member: "degree", "closeness", "betweenness" and "pagerank"."""
    g = club()
    return {"degree": {m: len(g[m]) for m in g}, "closeness": {m: closeness(g, m) for m in g},
            "betweenness": betweenness(g), "pagerank": pagerank(g, D, 100)}


KINDS = ["degree", "closeness", "betweenness", "pagerank"]   # the measures of the slider, 1 to 4


def ranking(values: dict) -> list:
    """The members from the highest value down; equal values by member number."""
    return sorted(values, key=lambda m: (-values[m], m))


def place(values: dict, m) -> int:
    """The place of member m: 1 more than the number of members with a higher value, so equal values share a place."""
    return 1 + sum(values[x] > values[m] for x in values)


def top(values: dict, k: int = 5) -> list:
    """The k highest members, and the members with the same value as the k-th, at most 8."""
    order = ranking(values)
    return [m for m in order if values[m] >= values[order[k - 1]]][:8]


def _amount(x: float, lang: str) -> str:
    """A score of the learner's with 3 decimals, or "more than 1,000" when it got out of hand."""
    return number(x, lang, 3) if abs(x) < 1000 else {"en": "more than 1,000", "nl": "meer dan 1.000"}[lang]


def _value(kind: str, x: float, lang: str) -> str:
    return number(x, lang, {"degree": 0, "closeness": 3, "betweenness": 1, "pagerank": 3}[kind])


def club_svg(values: dict, label: list, alt: str) -> str:
    """The club with every member as a circle, the larger the higher its value, from the lowest value to the highest:
    filled when the member went with the instructor, open when with the administrator. The members in `label` get
    their number beside them."""
    W, H, pad = 460, 360, 26
    lo = min(values.values())
    span = (max(values.values()) - lo) or 1.0

    def at(m):
        x, y = AT[m]
        return pad + x * (W - 2 * pad), pad + y * (H - 2 * pad)

    def radius(m):
        return 3.0 + 13.0 * math.sqrt((values[m] - lo) / span)

    parts = []
    for a, b in EDGES:
        (x0, y0), (x1, y1) = at(a), at(b)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" style="stroke:var(--ea-ink-3);stroke-width:1;opacity:0.55"/>')
    for m in sorted(range(MEMBERS), key=lambda m: -radius(m)):   # the large ones first, so that small ones stay seen
        x, y = at(m)
        look = ("fill:color-mix(in srgb, var(--ea-data) 80%, var(--ea-page));stroke:var(--ea-page);stroke-width:1.2"
                if m in INSTRUCTOR_SIDE else "fill:var(--ea-page);stroke:var(--ea-data);stroke-width:1.6")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius(m):.1f}" style="{look}"/>')
    for m in label:
        x, y = at(m)
        r = radius(m)
        parts.append(f'<text x="{x + r + 3:.1f}" y="{y + 4:.1f}" style="font-size:12px;font-weight:600;fill:var(--ea-ink);'
                     f'paint-order:stroke;stroke:var(--ea-page);stroke-width:3px">{m}</text>')
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.75rem;max-width:min(100%,{W}px)">'
            f'<svg viewBox="0 0 {W} {H}" aria-hidden="true" style="width:100%;height:auto;font-family:var(--ea-font-ui);'
            f'font-variant-numeric:tabular-nums">{"".join(parts)}</svg></figure>')


_KIND_WORDS = {
    "degree": {"en": "Degree", "nl": "Graad"},
    "closeness": {"en": "Closeness centrality", "nl": "Closeness centrality"},
    "betweenness": {"en": "Betweenness centrality", "nl": "Betweenness centrality"},
    "pagerank": {"en": "PageRank", "nl": "PageRank"},
}
_KIND_NOTE = {
    "degree": {"en": "the number of friends", "nl": "het aantal vrienden"},
    "closeness": {"en": "the other 33 members divided by the sum of the steps to them", "nl": "de 33 andere leden gedeeld door de som van de stappen naar hen"},
    "betweenness": {"en": "the shortest paths between two other members that run through the member", "nl": "de kortste paden tussen twee andere leden die door het lid lopen"},
    "pagerank": {"en": "the share of the steps a random walker spends at the member", "nl": "het deel van de stappen dat een willekeurige wandelaar bij het lid doorbrengt"},
}


def said(i: int, lang: str) -> str:
    """The sentence under the picture of measure i (1 to 4) of the slider."""
    kind = KINDS[i - 1]
    values = measures()[kind]
    best = ", ".join(f"{m} ({_value(kind, values[m], lang)})" for m in top(values))
    p1, p31 = place(values, 1), place(values, 31)
    if lang == "nl":
        return (f"{_KIND_WORDS[kind]['nl']}: {_KIND_NOTE[kind]['nl']}. De hoogste: {best}. Lid 1, met 9 vrienden, "
                f"staat {p1}e; lid 31, met 6, staat {p31}e.")
    return (f"{_KIND_WORDS[kind]['en']}: {_KIND_NOTE[kind]['en']}. The highest: {best}. Member 1, with 9 friends, "
            f"is {_ordinal(p1)}; member 31, with 6, is {_ordinal(p31)}.")


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def picture(i: int, lang: str) -> str:
    kind = KINDS[i - 1]
    values = measures()[kind]
    name = _KIND_WORDS[kind][lang] if kind == "pagerank" else _KIND_WORDS[kind][lang].lower()
    alt = ({"en": f"The karate club, 34 members as circles joined by their friendships: the higher their {name}, the larger the circle. ",
            "nl": f"De karateclub, 34 leden als cirkels verbonden door hun vriendschappen: hoe hoger hun {name}, hoe groter de cirkel. "}[lang]
           + said(i, lang))
    return club_svg({m: float(v) for m, v in values.items()}, list(dict.fromkeys(top(values) + [1, 31])), alt)


# Where the pages of the small web are drawn, in steps of 64 pixels.
WEB_AT = {"B": (0.5, 0.5), "C": (0.5, 1.6), "F": (0.5, 2.7), "A": (2.5, 1.6), "X": (4.6, 1.6), "Y": (4.6, 3.1),
          "D": (1.9, 3.4), "E": (3.5, 2.75)}


def web_picture(lang: str):
    """The small web as arrows: from every page to the pages it links to."""
    from everything_ai import draw
    edges = [(a, b) for a in sorted(WEB) for b in sorted(WEB[a])]
    alt = {"en": "Eight pages with arrows for their links. B, C, D, E, F, X and Y each link to A; A links only to X. D and E, to which no page links, also link to Y.",
           "nl": "Acht pagina's met pijlen voor hun links. B, C, D, E, F, X en Y linken elk naar A; A linkt alleen naar X. D en E, waar geen pagina naar linkt, linken ook naar Y."}[lang]
    title = {"en": "The small web", "nl": "Het kleine web"}[lang]
    return draw.graph_panels([{"at": WEB_AT, "edges": edges, "directed": True, "title": title}], alt=alt)


# The lesson is done when these are correct. "web" is the choice question in the lesson.
REQUIRED = ["closeness", "pagerank", "web"]


class _Val:
    """A test value that the check's messages write by its name, such as `club`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


_CLUB = _Val("club", club())
_LINE = _Val("line", {"P": {"Q"}, "Q": {"P", "R"}, "R": {"Q", "S"}, "S": {"R"}})
_STAR = _Val("star", {"X": {"P", "Q", "R", "S", "T"}, "P": {"X"}, "Q": {"X"}, "R": {"X"}, "S": {"X"}, "T": {"X"}})
_RING = _Val("ring", {"A": {"B", "F"}, "B": {"A", "C"}, "C": {"B", "D"}, "D": {"C", "E"}, "E": {"D", "F"}, "F": {"E", "A"}})
_WEB = _Val("web", WEB)
_TWO = _Val("two_parts", {"A": {"B"}, "B": {"A"}, "C": {"D", "E"}, "D": {"C", "E"}, "E": {"C", "D"}})


def _shown(v):
    if isinstance(v, dict) and len(v) > 8:
        return {"en": f"a dictionary of {len(v)} scores", "nl": f"een dictionary van {len(v)} scores"}
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {round(x, 4) if isinstance(x, float) else x!r}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return repr(round(v, 4))
    return repr(v)


def _same(got, want) -> bool:
    """The same scores, each within a small margin: a right answer may add in another order."""
    if isinstance(want, dict):
        return isinstance(got, dict) and set(got) == set(want) and all(checking.same_value(got[k], want[k]) for k in want)
    return checking.same_value(got, want)


_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


def _run(seen, *args):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    if not callable(fn):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in args)))
    except Exception:
        return None


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v)


# --- exercise 1: closeness centrality --------------------------------------------------------------------------------

_STEPS = ("from collections import deque\n\n\ndef steps_from(graph, start):\n    steps = {start: 0}\n"
          "    frontier = deque([start])\n    while len(frontier) > 0:\n        node = frontier.popleft()\n"
          "        for nxt in graph[node]:\n            if nxt not in steps:\n                steps[nxt] = steps[node] + 1\n"
          "                frontier.append(nxt)\n    return steps\n\n\n")


def _c(add: str = "total + steps[other]", back: str = "(len(steps) - 1) / total") -> str:
    return (_STEPS + "def closeness(graph, node):\n    steps = steps_from(graph, node)\n    total = 0\n"
            f"    for other in steps:\n        total = {add}\n    return {back}\n")


_C_START = _c("...", "...")
_C = _c()

_CL_WORDS = {
    "title": {"en": "Your closeness for every member of the club", "nl": "Jouw closeness voor elk lid van de club"},
    "right": {"en": "The higher your closeness, the larger the circle. Member 0, the instructor, is the closest to all others, 0.569: on average 1.76 steps to each. Member 31, with 6 friends, is 4th, 0.541; member 1, with 9, only 9th, 0.485.",
              "nl": "Hoe hoger jouw closeness, hoe groter de cirkel. Lid 0, de instructeur, ligt het dichtst bij alle anderen, 0,569: gemiddeld 1,76 stappen naar elk. Lid 31, met 6 vrienden, staat 4e, 0,541; lid 1, met 9, pas 9e, 0,485."},
    "said": {"en": "The higher your closeness, the larger the circle. Yours puts member {first} first, with {value}. The course's puts member 0 first, with 0.569: 33 other members divided by 58 steps.",
             "nl": "Hoe hoger jouw closeness, hoe groter de cirkel. Die van jou zet lid {first} bovenaan, met {value}. Die van de course zet lid 0 bovenaan, met 0,569: 33 andere leden gedeeld door 58 stappen."},
}


def _cl_picture(seen) -> str:
    lang = seen.lang
    g = club()
    values = {}
    for m in g:
        v = _number(_run(seen, g, m))
        if v is None:
            return ""
        values[m] = v
    if max(values.values()) <= 0:
        return ""
    words_ = {k: v[lang] for k, v in _CL_WORDS.items()}
    order = ranking(values)
    right = all(checking.same_value(values[m], measures()["closeness"][m]) for m in g)
    text = words_["right"] if right else words_["said"].format(first=order[0], value=_amount(values[order[0]], lang))
    svg = club_svg(values, list(dict.fromkeys(top(values) + [1, 31])), text)
    return yours.figure(words_["title"], svg + _key(text), alt=text)


EXERCISES = {
    "closeness": FunctionBox(
        name="closeness",
        picture=_cl_picture,
        function="closeness",
        start=_C_START,
        solution=_C,
        show=_shown,
        compare=_same,
        visible=[(_CLUB, 0)],
        hidden=[(_CLUB, 16), (_LINE, "P"), (_LINE, "Q"), (_STAR, "X"), (_STAR, "P"), (_RING, "A"), (_TWO, "C")],
        first=[_DOTS],
        mistakes=[
            Mistake("mean", wrong=_c(back="total / (len(steps) - 1)"),
                    en="`{call}` gives {got}: the average number of steps to the others. Few steps should give a high closeness, so turn it around: the number of others divided by `total`.",
                    nl="`{call}` geeft {got}: het gemiddelde aantal stappen naar de anderen. Weinig stappen moet een hoge closeness geven, dus draai het om: het aantal anderen gedeeld door `total`."),
            Mistake("itself", wrong=_c(back="len(steps) / total"),
                    en="`{call}` gives {got}: `len(steps)` counts `node` itself too, at 0 steps. Divide the number of others, `len(steps) - 1`, by `total`.",
                    nl="`{call}` geeft {got}: `len(steps)` telt `node` zelf ook mee, op 0 stappen. Deel het aantal anderen, `len(steps) - 1`, door `total`."),
            Mistake("one", wrong=_c(back="1 / total"),
                    en="`{call}` gives {got}: 1 divided by the sum of the steps. That depends on how many others there are; divide the number of others, `len(steps) - 1`, by `total`, so that it is 1 divided by the average.",
                    nl="`{call}` geeft {got}: 1 gedeeld door de som van de stappen. Dat hangt af van hoeveel anderen er zijn; deel het aantal anderen, `len(steps) - 1`, door `total`, dan is het 1 gedeeld door het gemiddelde."),
            Mistake("all-nodes", wrong=_c(back="(len(graph) - 1) / total"),
                    en="`{call}` gives {got}: `len(graph) - 1` counts every other node of the graph, also the ones `node` cannot reach. Count the nodes in `steps`: `len(steps) - 1`.",
                    nl="`{call}` geeft {got}: `len(graph) - 1` telt elke andere node van de graaf, ook de nodes die `node` niet kan bereiken. Tel de nodes in `steps`: `len(steps) - 1`."),
            Mistake("count", wrong=_c(add="total + 1"),
                    en="`{call}` gives {got}: `total + 1` counts the members, not their steps. Add the steps to `other`: `steps[other]`.",
                    nl="`{call}` geeft {got}: `total + 1` telt de leden, niet hun stappen. Tel de stappen naar `other` op: `steps[other]`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: add up the steps to every other node, and divide the number of others by that sum.",
                      "nl": "Vervang de twee `...`: tel de stappen naar elke andere node op, en deel het aantal anderen door die som."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(graph, node)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(graph, node)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. `total` is the sum of the steps from `node` to every node it reaches, and the closeness is the number of others divided by `total`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. `total` is de som van de stappen van `node` naar elke node die hij bereikt, en de closeness is het aantal anderen gedeeld door `total`."},
            "almost": {"en": "Right for member 0 of the club. But `{call}` gives {got}, and it should give {want}. Does your function use `graph` and `node`, and nothing of the club?",
                       "nl": "Goed voor lid 0 van de club. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `graph` en `node`, en niets van de club?"},
            "correct": {"en": "Your function is right in all {n} cases. Member 0 reaches the 33 others in 58 steps together: closeness 33 / 58 = 0.569, the highest of the club.",
                        "nl": "Je functie klopt in alle {n} gevallen. Lid 0 bereikt de 33 anderen in samen 58 stappen: closeness 33 / 58 = 0,569, de hoogste van de club."},
        },
    ),
}


# --- exercise 2: PageRank by power iteration ---------------------------------------------------------------------------

_P_HEAD = ("def pagerank(graph, d, rounds):\n    n = len(graph)\n    scores = {}\n    for node in graph:\n"
           "        scores[node] = 1 / n\n    for _ in range(rounds):\n        new = {}\n        for node in graph:\n"
           "            new[node] = (1 - d) / n\n        for node in graph:\n")


def _p(share: str = "d * scores[node] / len(graph[node])", add: str = "new[nxt] + share") -> str:
    return (_P_HEAD + f"            share = {share}\n            for nxt in graph[node]:\n                new[nxt] = {add}\n"
            "        scores = new\n    return scores\n")


_P_START = _p("...", "...")
_P = _p()

_PR_WORDS = {
    "title": {"en": "Your PageRank for the club, after 100 rounds", "nl": "Jouw PageRank voor de club, na 100 rondes"},
    "right": {"en": "The higher your PageRank, the larger the circle. Member 33 gets 0.101 and member 0 0.097, close to their shares of the degrees, 17 and 16 of 156 ends of edges. The 34 scores add up to 1: what one member passes on, another gets.",
              "nl": "Hoe hoger jouw PageRank, hoe groter de cirkel. Lid 33 krijgt 0,101 en lid 0 0,097, dicht bij hun deel van de graden, 17 en 16 van 156 uiteinden van edges. De 34 scores tellen op tot 1: wat het ene lid doorgeeft, krijgt een ander."},
    "said": {"en": "The higher your score, the larger the circle. Yours add up to {total}, and put member {first} first, with {value}. PageRank adds up to 1, and puts member 33 first, with 0.101.",
             "nl": "Hoe hoger jouw score, hoe groter de cirkel. Die van jou tellen op tot {total}, en zetten lid {first} bovenaan, met {value}. PageRank telt op tot 1, en zet lid 33 bovenaan, met 0,101."},
}


def _pr_picture(seen) -> str:
    lang = seen.lang
    got = _run(seen, club(), D, 100)
    if not isinstance(got, dict) or set(got) != set(range(MEMBERS)):
        return ""
    values = {m: _number(v) for m, v in got.items()}
    if any(v is None for v in values.values()) or max(values.values()) <= 0:
        return ""
    words_ = {k: v[lang] for k, v in _PR_WORDS.items()}
    order = ranking(values)
    if _same(got, measures()["pagerank"]):
        text = words_["right"]
    else:
        text = words_["said"].format(total=_amount(sum(values.values()), lang), first=order[0],
                                     value=_amount(values[order[0]], lang))
    svg = club_svg(values, list(dict.fromkeys(top(values) + [1, 31])), text)
    return yours.figure(words_["title"], svg + _key(text), alt=text)


EXERCISES["pagerank"] = FunctionBox(
    name="pagerank",
    picture=_pr_picture,
    function="pagerank",
    start=_P_START,
    solution=_P,
    show=_shown,
    compare=_same,
    visible=[(_CLUB, D, 100)],
    hidden=[(_WEB, D, 100), (_RING, D, 20), (_STAR, 0.5, 30), (_LINE, D, 3), (_CLUB, 1.0, 2)],
    first=[_DOTS],
    mistakes=[
        Mistake("no-d", wrong=_p(share="scores[node] / len(graph[node])"),
                en="`{call}` gives {got}: every node passes on all of its score, and every node also gets `(1 - d) / n` for the jumps, so the scores add up to more than 1. A node passes on `d` times its score; the rest goes to the jumps.",
                nl="`{call}` geeft {got}: elke node geeft zijn hele score door, en elke node krijgt ook nog `(1 - d) / n` voor de sprongen, dus de scores tellen op tot meer dan 1. Een node geeft `d` keer zijn score door; de rest gaat naar de sprongen."),
        Mistake("not-split", wrong=_p(share="d * scores[node]"),
                en="`{call}` gives {got}: every neighbor gets the whole `d * scores[node]`, so a node with 9 neighbors passes on 9 times what it has. Split it in equal parts: divide by the number of neighbors, `len(graph[node])`.",
                nl="`{call}` geeft {got}: elke buur krijgt de hele `d * scores[node]`, dus een node met 9 buren geeft 9 keer door wat hij heeft. Verdeel het in gelijke delen: deel door het aantal buren, `len(graph[node])`."),
        Mistake("overwrite", wrong=_p(add="share"),
                en="`{call}` gives {got}: `new[nxt] = share` throws away what `nxt` already got, from the jumps and from other nodes. Add the share to it: `new[nxt] + share`.",
                nl="`{call}` geeft {got}: `new[nxt] = share` gooit weg wat `nxt` al kreeg, van de sprongen en van andere nodes. Tel het deel erbij op: `new[nxt] + share`."),
        Mistake("old-scores", wrong=_p(add="scores[nxt] + share"),
                en="`{call}` gives {got}: `scores[nxt] + share` starts from the old score of `nxt` instead of from what it gets this round. Add the share to `new[nxt]`.",
                nl="`{call}` geeft {got}: `scores[nxt] + share` begint bij de oude score van `nxt` in plaats van bij wat hij deze ronde krijgt. Tel het deel op bij `new[nxt]`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the share that `node` passes to each node in `graph[node]`, and the new score of `nxt` with that share.",
                  "nl": "Vervang de twee `...`: het deel dat `node` doorgeeft aan elke node in `graph[node]`, en de nieuwe score van `nxt` met dat deel."},
        "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(graph, d, rounds)`.",
                      "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(graph, d, rounds)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Every round, a node passes `d` times its score on, split in equal parts over the nodes in `graph[node]`, and each of them adds that part to its new score.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Elke ronde geeft een node `d` keer zijn score door, in gelijke delen verdeeld over de nodes in `graph[node]`, en elk van hen telt dat deel op bij zijn nieuwe score."},
        "almost": {"en": "Right for the club. But `{call}` gives {got}, and it should give {want}. Does your function use `graph`, `d` and `rounds`, and nothing of the club?",
                   "nl": "Goed voor de club. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `graph`, `d` en `rounds`, en niets van de club?"},
        "correct": {"en": "Your function is right in all {n} cases. On the club, member 33 gets 0.101 and member 0 0.097; on the small web, page X, with one link from A, gets 0.411, and Y, with two links, 0.035.",
                    "nl": "Je functie klopt in alle {n} gevallen. Op de club krijgt lid 33 0,101 en lid 0 0,097; op het kleine web krijgt pagina X, met één link van A, 0,411, en Y, met twee links, 0,035."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("closeness", _C_START, "empty", None),
    ("closeness", _c(back="..."), "wrong", "dots-left"),
    ("closeness", _C, "correct", None),
    ("closeness", _c(back="(len(graph) - 1) / total"), "wrong", "all-nodes"),
    ("closeness", _c(back="1 / (total / (len(steps) - 1))"), "correct", None),
    ("closeness", _c(back="total / (len(steps) - 1)"), "wrong", "mean"),
    ("closeness", _c(back="len(steps) / total"), "wrong", "itself"),
    ("closeness", _c(back="1 / total"), "wrong", "one"),
    ("closeness", _c(add="total + 1"), "wrong", "count"),
    ("closeness", _STEPS + "def closeness(graph, node):\n    return 33 / 58\n", "almost", None),
    ("pagerank", _P_START, "empty", None),
    ("pagerank", _p(add="..."), "wrong", "dots-left"),
    ("pagerank", _P, "correct", None),
    ("pagerank", _p(share="d * (scores[node] / len(graph[node]))", add="share + new[nxt]"), "correct", None),
    ("pagerank", _p(share="scores[node] / len(graph[node])"), "wrong", "no-d"),
    ("pagerank", _p(share="d * scores[node]"), "wrong", "not-split"),
    ("pagerank", _p(add="share"), "wrong", "overwrite"),
    ("pagerank", _p(add="scores[nxt] + share"), "wrong", "old-scores"),
]
