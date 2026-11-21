# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-what-to-align-to: aligned with what? The learner writes contradictions, which counts the
clicks that go against the ranking a person states, and circle_of_three, which finds preferences that go round in a
circle of three, as in Russell's example. A demo goes through Gabriel's six targets of alignment on one made-up
request."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

# made up: the ranking a person states, 1 the best, and what the same person clicked when two articles were shown
STATED = {"news": 1, "science": 2, "gossip": 3}
CLICKS = [("gossip", "news"), ("news", "science"), ("gossip", "science"), ("science", "gossip"), ("gossip", "news")]
# Russell's example of preferences that go round: vanilla over bubble gum, bubble gum over pistachio, pistachio over
# vanilla; and with licorice last, which all three beat
RUSSELL = [("vanilla", "bubblegum"), ("bubblegum", "pistachio"), ("pistachio", "vanilla")]
LICORICE = [("vanilla", "licorice"), ("bubblegum", "licorice"), ("pistachio", "licorice")]
TARGETS = [1, 2, 3, 4, 5, 6]   # the stops of the demo's slider: the six targets in Gabriel's list


def contradictions(stated, clicks):
    """What the learner writes in Exercise 1: how many clicks go against the stated ranking."""
    count = 0
    for clicked, skipped in clicks:
        if stated[clicked] > stated[skipped]:
            count = count + 1
    return count


def circle_of_three(prefs):
    """What the learner writes in Exercise 2: whether three options go round in a circle, a over b, b over d, d over a."""
    for a, b in prefs:
        for c, d in prefs:
            if b == c and (d, a) in prefs:
                return True
    return False


def running(stated, clicks) -> list:
    """After each click, how many clicks so far went against the stated ranking."""
    return [contradictions(stated, clicks[:k + 1]) for k in range(len(clicks))]


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def show_value(v) -> str:
    """A result as the check shows it: True, False and None as Python writes them, a whole number as it is, anything
    else as Python shows it, cut short when it is long."""
    text = repr(v)
    return text if len(text) <= 200 else text[:197] + "..."


def same_answer(got, want) -> bool:
    """Whether the learner's answer is the course's: the same whole number, or the same True or False. A count as
    True or a float such as 3.0 does not count, nor does 1 for True."""
    if isinstance(want, bool):
        return isinstance(got, bool) and got == want
    if isinstance(want, int):
        return isinstance(got, int) and not isinstance(got, bool) and got == want
    return type(got) is type(want) and got == want   # two wrong versions that both give None, say


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a call that stopped with an error


def _all(fn, calls: list) -> list:
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check:
    a list of the results, with _STOPPED for each call that stops with an error; the Timeout passes on."""
    def one(args):
        try:
            return fn(*(copy.deepcopy(a) for a in args))
        except Timeout:
            raise
        except Exception:   # noqa: BLE001   the check above says what went wrong
            return _STOPPED

    with contextlib.redirect_stdout(io.StringIO()):
        return with_time_limit(lambda: [one(args) for args in calls])


def _ranking(stated: dict) -> str:
    """A stated ranking, best first: news, science, gossip."""
    return ", ".join(sorted(stated, key=lambda k: stated[k]))


def _clicks(clicks: list, over: str) -> str:
    """Clicks as the picture writes them: gossip over news, news over science."""
    return ", ".join(f"{c} {over} {s}" for c, s in clicks) if clicks else "–"


def _prefs(prefs: list) -> str:
    """Preferences as the picture writes them: vanilla > bubblegum, bubblegum > pistachio."""
    return ", ".join(f"{a} > {b}" for a, b in prefs) if prefs else "–"


# --- the demo --------------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Six targets for one request",
           "request": "Late at night, someone asks a news assistant: “Give me something to read.”",
           "said": "Target {k}, {name}: “{definition}” An assistant aligned with it would {action}",
           "note": "The request and what the assistant would do are made up for this lesson; the six targets and "
                   "their definitions are Gabriel's. The person in the example clicks gossip most, as in Exercise 1.",
           "targets": [
               ("instructions", "the agent does what I instruct it to do.",
                "show the first text it finds: something to read, as asked."),
               ("expressed intentions", "the agent does what I intend it to do.",
                "show a short article to read before sleep, which is what the request means at this hour."),
               ("revealed preferences", "the agent does what my behaviour reveals I prefer.",
                "show gossip, since that is what the person clicks most."),
               ("informed preferences or desires",
                "the agent does what I would want it to do if I were rational and informed.",
                "show the storm warning for tomorrow morning, which the person would want to read if they knew of it."),
               ("interest or well-being",
                "the agent does what is in my interest, or what is best for me, objectively speaking.",
                "show one short, calm piece and suggest stopping there, since sleep is what is best for the person now."),
               ("values", "the agent does what it morally ought to do, as defined by the individual or society.",
                "show a reliable article, not a misleading one that would get more clicks."),
           ]},
    "nl": {"title": "Zes doelen voor één verzoek",
           "request": "Laat op de avond vraagt iemand een nieuwsassistent: geef me iets om te lezen.",
           "said": "Bij doel {k}, {name}, doet de AI volgens Gabriel {definition} Afgestemd op dit doel zou een assistent "
                   "{action}",
           "note": "Het verzoek en wat de assistent zou doen, heeft deze les zelf verzonnen; de zes doelen en wat ze "
                   "betekenen, komen van Gabriel. De persoon in het voorbeeld klikt het vaakst op roddels, net als in "
                   "oefening 1.",
           "targets": [
               ("instructies", "wat de persoon opdraagt.",
                "de eerste tekst laten zien die de assistent vindt: iets om te lezen, zoals gevraagd."),
               ("uitgesproken bedoelingen", "wat de persoon bedoelt.",
                "een kort stuk laten zien om voor het slapen te lezen, want dat bedoelt het verzoek op dit uur."),
               ("geopenbaarde voorkeuren", "wat de persoon verkiest, te zien aan wat die persoon doet.",
                "roddels laten zien, want daar klikt de persoon het vaakst op."),
               ("geïnformeerde voorkeuren of verlangens",
                "wat de persoon zou willen als die rationeel en goed geïnformeerd was.",
                "de stormwaarschuwing voor morgenochtend laten zien: wie ervan wist, zou die willen lezen."),
               ("belang of welzijn", "wat objectief gezien in het belang van de persoon is, of het beste is.",
                "één kort, rustig stuk laten zien en voorstellen daarna te stoppen, want slapen is nu het beste voor "
                "de persoon."),
               ("waarden", "wat moreel juist is, zoals de persoon of de samenleving dat opvat.",
                "een betrouwbaar stuk laten zien, en geen misleidend stuk dat meer klikken zou krijgen."),
           ]},
}


def demo_said(k: int, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    name, definition, action = w["targets"][k - 1]
    return w["said"].format(k=k, name=name, definition=definition, action=action)


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: Gabriel's six targets as a column of rows, the chosen one outlined, with what an assistant aligned
    with it would do for the made-up request."""
    k = min(TARGETS, key=lambda t: abs(t - float(v)))
    w = _DEMO_WORDS[lang]
    rows = []
    for i, (name, _definition, _action) in enumerate(w["targets"], start=1):
        on = i == k
        style = ("border:2px solid var(--ea-ink);color:var(--ea-ink);font-weight:650" if on else
                 "border:1px solid var(--ea-line);color:var(--ea-ink-3)")
        rows.append(f'<div style="{style};border-radius:6px;padding:0.25rem 0.6rem;margin:0.2rem 0;'
                    f'font-family:var(--ea-font-ui);font-size:0.86rem">{i}. {html.escape(name)}</div>')
    said = demo_said(k, lang)
    body = (_note(w["request"]) + '<div style="max-width:420px;margin:0.4rem 0">' + "".join(rows) + "</div>"
            + f'<div style="font-family:var(--ea-font-ui);font-size:0.86rem;color:var(--ea-ink)">{html.escape(said)}</div>'
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"] + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

_CONTRA_WORDS = {
    "en": {"title": "Your count of clicks against the stated ranking",
           "x": "click", "y": "against the ranking so far", "all": "all", "over": "over",
           "head": ["ranking (best first)", "clicks", "the course's", "yours"], "stopped": "error",
           "note": "The chart is the first case of the check: the stated ranking news, science, gossip, and five "
                   "clicks. Each bar is the number of clicks so far that went against the ranking; on the right, the "
                   "course's count is a dot and yours a ring. The table is every case; a dashed row is one where your "
                   "function gives something other than the course's."},
    "nl": {"title": "Jouw telling van klikken tegen de uitgesproken rangorde in",
           "x": "klik", "y": "tot nu toe tegen de rangorde in", "all": "alle", "over": "boven",
           "head": ["rangorde (beste eerst)", "klikken", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De grafiek is het eerste geval van de check: de uitgesproken rangorde news, science, gossip, en "
                   "vijf klikken. Elke balk is het aantal klikken tot nu toe dat tegen de rangorde in ging; rechts is "
                   "de telling van de course een stip en de jouwe een ring. De tabel bevat elk geval; een gestippelde "
                   "rij is er een waar jouw functie iets anders geeft dan die van de course."},
}

OTHER = {"gossip": 1, "news": 2, "science": 3}   # made up: a person who says gossip is best
CONTRA_CASES = [
    (STATED, CLICKS),
    (STATED, [("news", "gossip"), ("gossip", "science")]),
    (STATED, [("science", "news")]),
    (STATED, [("gossip", "news"), ("gossip", "science"), ("science", "news")]),
    (OTHER, [("gossip", "news"), ("science", "news"), ("news", "science")]),
    (STATED, [("news", "science"), ("science", "gossip")]),
]


def contradictions_picture(seen) -> str:
    """The first case: a bar per click with the count so far, the course's count as a dot and the learner's as a
    ring; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CONTRA_WORDS[seen.lang]
    got = _all(fn, CONTRA_CASES)
    rows = []
    for i, case in enumerate(CONTRA_CASES):
        want = contradictions(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_answer(got[i], want)
        rows.append(([(html.escape(_ranking(case[0])), ""), (html.escape(_clicks(case[1], w["over"])), ""),
                      (_code(show_value(want)), ""), (shown_, "font-weight:600")], _OFF if off else ""))
    so_far = running(STATED, CLICKS)
    n = len(CLICKS)
    want = contradictions(STATED, CLICKS)
    panel = chart.Panel(x=(0.4, n + 1.6), y=(0, n + 0.6), xlabel=w["x"], ylabel=w["y"],
                        xticks=list(range(1, n + 2)), xtick_labels=[str(k) for k in range(1, n + 1)] + [w["all"]],
                        yticks=list(range(0, n + 1)))
    for k, c in enumerate(so_far, start=1):
        panel.rect(k - 0.3, 0, k + 0.3, c, role="ink-3")
    panel.points([n + 1], [want], role="ink", r=3)
    mine = got[0]
    if isinstance(mine, int) and not isinstance(mine, bool) and 0 <= mine <= n + 0.6:
        panel.points([n + 1], [mine], role="ink", r=7, hollow=True)
    svg = chart.figure([panel], alt=w["title"], width=380, height=240)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_CIRCLE_WORDS = {
    "en": {"title": "Your answer for Russell's three flavors",
           "course": "the course's answer", "yours": "your answer",
           "head": ["preferences (better > worse)", "the course's", "yours"], "stopped": "error",
           "alt": "Three flavors with an arrow from each preferred one to the one it beats: vanilla to bubblegum, "
                  "bubblegum to pistachio, and pistachio to vanilla, a circle.",
           "note": "The drawing is the first case of the check, Russell's example: an arrow points from the flavor "
                   "preferred to the one it beats, and the three arrows go round. The table is every case; a dashed "
                   "row is one where your function gives something other than the course's."},
    "nl": {"title": "Jouw antwoord voor de drie smaken van Russell",
           "course": "antwoord van de course", "yours": "jouw antwoord",
           "head": ["voorkeuren (beter > slechter)", "van de course", "de jouwe"], "stopped": "fout",
           "alt": "Drie smaken met een pijl van elke verkozen smaak naar de smaak die ze verslaat: vanilla naar "
                  "bubblegum, bubblegum naar pistachio, en pistachio naar vanilla, een kring.",
           "note": "De tekening is het eerste geval van de check, het voorbeeld van Russell: een pijl wijst van de "
                   "verkozen smaak naar de smaak die ze verslaat, en de drie pijlen lopen rond. De tabel bevat elk "
                   "geval; een gestippelde rij is er een waar jouw functie iets anders geeft dan die van de course."},
}

CIRCLE_CASES = [
    (RUSSELL,),
    ([("vanilla", "bubblegum"), ("bubblegum", "pistachio"), ("vanilla", "pistachio")],),
    (LICORICE,),
    ([("vanilla", "licorice")] + RUSSELL,),
    ([("tea", "coffee"), ("coffee", "tea")],),
    ([],),
]


def _graph(prefs: list, alt: str, lines: list) -> str:
    """The options of `prefs` on an ellipse, each in a box, with an arrow from the better to the worse of each pair,
    and below it the lines of text in `lines`. Ink only: no option is a role of a model."""
    names = []
    for a, b in prefs:
        for x in (a, b):
            if x not in names:
                names.append(x)
    W, cx, cy, rx, ry, bh = 380, 190, 104, 118, 66, 26
    place = {}
    parts = []
    for i, name in enumerate(names):
        angle = -math.pi / 2 + 2 * math.pi * i / max(len(names), 1)
        place[name] = (cx + rx * math.cos(angle), cy + ry * math.sin(angle), 4.2 + 3.9 * len(name))
    line = 'style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round"'
    for a, b in prefs:
        (x0, y0, w0), (x1, y1, w1) = place[a], place[b]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length

        def out(w, ux=ux, uy=uy):
            """How far from a box's middle the arrow leaves it, along (ux, uy), plus a small gap."""
            return min(w / max(abs(ux), 1e-9), (bh / 2) / max(abs(uy), 1e-9)) + 5

        sx, sy = x0 + ux * out(w0 + 4), y0 + uy * out(w0 + 4)
        ex, ey = x1 - ux * out(w1 + 4), y1 - uy * out(w1 + 4)
        parts.append(f'<path d="M{sx:.1f} {sy:.1f} L{ex:.1f} {ey:.1f}" {line}/>'
                     f'<path d="M{ex - 8 * ux - 5 * uy:.1f} {ey - 8 * uy + 5 * ux:.1f} L{ex:.1f} {ey:.1f} '
                     f'L{ex - 8 * ux + 5 * uy:.1f} {ey - 8 * uy - 5 * ux:.1f}" {line}/>')
    for name, (x, y, w) in place.items():
        parts.append(f'<rect x="{x - w - 4:.1f}" y="{y - bh / 2:.1f}" width="{2 * w + 8:.1f}" height="{bh}" rx="6" '
                     f'style="fill:var(--ea-page);stroke:var(--ea-ink-3);stroke-width:1.2"/>'
                     f'<text x="{x:.1f}" y="{y + 5:.1f}" style="font-size:14px;fill:var(--ea-ink);text-anchor:middle">'
                     f'{html.escape(name)}</text>')
    y = cy + ry + bh + 14
    for text in lines:
        parts.append(f'<text x="{cx}" y="{y}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(text)}</text>')
        y += 18
    H = y - 4
    return (f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;max-width:{W}px;'
            f'font-family:var(--ea-font-ui)"><title>{html.escape(alt)}</title>{"".join(parts)}</svg>')


def circle_picture(seen) -> str:
    """The first case: Russell's three flavors with an arrow from each preferred flavor to the one it beats, the
    course's answer and the learner's below it; every case in a table."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _CIRCLE_WORDS[seen.lang]
    got = _all(fn, CIRCLE_CASES)
    rows = []
    for i, case in enumerate(CIRCLE_CASES):
        want = circle_of_three(*case)
        if got[i] is _STOPPED:
            shown_, off = html.escape(w["stopped"]), True
        else:
            shown_, off = _code(show_value(got[i])), not same_answer(got[i], want)
        rows.append(([(html.escape(_prefs(case[0])), ""), (_code(show_value(want)), ""),
                      (shown_, "font-weight:600")], _OFF if off else ""))
    mine = w["stopped"] if got[0] is _STOPPED else show_value(got[0])
    svg = _graph(RUSSELL, w["alt"], [f'{w["course"]}: {show_value(circle_of_three(RUSSELL))}',
                                     f'{w["yours"]}: {mine}'])
    body = svg + _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["contradictions", "circle", "letter", "train"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: contradictions; the first case is the made-up ranking and five clicks, then two clicks, one click, three
# clicks that all go against it, another person's ranking, and two clicks that both follow it
_N_START = "def contradictions(stated, clicks):\n    ...\n"
_N = ("def contradictions(stated, clicks):\n"
      "    count = 0\n"
      "    for clicked, skipped in clicks:\n"
      "        if stated[clicked] > stated[skipped]:\n"
      "            count = count + 1\n"
      "    return count\n")
_N_AGREE = _N.replace("stated[clicked] > stated[skipped]", "stated[clicked] < stated[skipped]")
_N_EVERY = _N.replace("        if stated[clicked] > stated[skipped]:\n            count = count + 1\n",
                      "        count = count + 1\n")
_N_RESET = ("def contradictions(stated, clicks):\n"
            "    for clicked, skipped in clicks:\n"
            "        count = 0\n"
            "        if stated[clicked] > stated[skipped]:\n"
            "            count = count + 1\n"
            "    return count\n")
_N_FIRST = _N.replace("    return count\n", "        return count\n")
_N_WORDS = _N.replace("stated[clicked] > stated[skipped]", "clicked > skipped")
_N_TUPLE = ("def contradictions(stated, clicks):\n"
            "    count = 0\n"
            "    for clicked in clicks:\n"
            "        if stated[clicked] > 1:\n"
            "            count = count + 1\n"
            "    return count\n")

# exercise 2: circle_of_three; the first case is Russell's three flavors, then the same flavors in order, licorice last,
# the circle after another pair, a pair both ways, and no preferences at all
_R_START = "def circle_of_three(prefs):\n    ...\n"
_R = ("def circle_of_three(prefs):\n"
      "    for a, b in prefs:\n"
      "        for c, d in prefs:\n"
      "            if b == c and (d, a) in prefs:\n"
      "                return True\n"
      "    return False\n")
_R_CHAIN = _R.replace("if b == c and (d, a) in prefs:", "if b == c:")
_R_DIRECTION = _R.replace("(d, a) in prefs", "(a, d) in prefs")
_R_EARLY = _R.replace("                return True\n    return False\n", "                return True\n        return False\n")
_R_NONE = _R.replace("    return False\n", "")
_R_NOCHAIN = _R.replace("if b == c and (d, a) in prefs:", "if (d, a) in prefs:")
_R_PAIR = _R.replace("if b == c and (d, a) in prefs:", "if (b, a) in prefs:")


_NAMED = [(STATED, "stated"), (CLICKS, "clicks")]   # the data of the worked example, which the texts call by name


def _lit(value) -> str:
    """A value the way the lesson writes it in code: the worked example's data by name, texts in double quotes, also
    inside a pair, so that a call reads `contradictions(stated, [("news", "gossip")])`."""
    for data, name in _NAMED:
        if type(value) is type(data) and value == data:
            return name
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_lit(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in value.items()) + "}"
    return repr(value)


class PairBox(FunctionBox):
    """A FunctionBox whose messages write a call the way the lesson does: `contradictions(stated, clicks)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_lit(a) for a in case)})"


EXERCISES = {
    "contradictions": PairBox(
        name="contradictions",
        picture=contradictions_picture,
        function="contradictions",
        start=_N_START,
        solution=_N,
        visible=[CONTRA_CASES[0]],
        hidden=CONTRA_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_answer,
        mistakes=[
            Mistake("tuple", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], tuple),
                    en="You look up a whole click in `stated`, but each click is a pair of two options, and `stated` only knows single options. Take each pair apart in the loop into `clicked` and `skipped`.",
                    nl="Je zoekt een hele klik op in `stated`, maar elke klik is een paar van twee opties, en `stated` kent alleen losse opties. Haal elk paar in de loop uit elkaar in `clicked` en `skipped`."),
            Mistake("agree", wrong=_N_AGREE,
                    en="`{call}` gives {got}, and it should give {want}: you count the clicks that follow the stated ranking. A click goes against it when the place of `clicked` is a larger number than the place of `skipped`: place 1 is the best.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de klikken die de uitgesproken rangorde volgen. Een klik gaat ertegenin als de plaats van `clicked` een groter getal is dan de plaats van `skipped`: plaats 1 is de beste."),
            Mistake("every", wrong=_N_EVERY,
                    en="`{call}` gives {got}, and it should give {want}: you count every click, but only a click against the ranking counts. Put `count = count + 1` under an `if` that compares the places of the two options.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt elke klik, maar alleen een klik tegen de rangorde in telt. Zet `count = count + 1` onder een `if` die de plaatsen van de twee opties vergelijkt."),
            Mistake("reset", wrong=_N_RESET,
                    en="`{call}` gives {got}, and it should give {want}: `count = 0` is inside the loop, so it starts again at 0 for every click and only the last click is left. Put it before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `count = 0` staat in de loop, dus de telling begint bij elke klik opnieuw bij 0 en alleen de laatste klik blijft over. Zet het voor de loop."),
            Mistake("first", wrong=_N_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first click. Put `return count` after the loop, as far in as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste klik. Zet `return count` na de loop, even ver ingesprongen als `for`."),
            Mistake("words", wrong=_N_WORDS,
                    en="`{call}` gives {got}, and it should give {want}: you compare the names of the two options, and Python compares texts from A to Z. Compare their places: look each name up in `stated`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vergelijkt de namen van de twee opties, en Python vergelijkt teksten van A tot Z. Vergelijk hun plaatsen: zoek elke naam op in `stated`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start `count` at 0, go through the clicks, add 1 for each click against the stated ranking, and return `count`.",
                      "nl": "Vervang de `...`: laat `count` beginnen bij 0, loop door de klikken, tel 1 op bij elke klik tegen de uitgesproken rangorde in, en geef `count` terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start `count` at 0. For each click, take it apart into `clicked` and `skipped`, look up both places in `stated`, and add 1 when the place of `clicked` is the larger number. Return `count` after the loop.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Laat `count` beginnen bij 0. Haal elke klik uit elkaar in `clicked` en `skipped`, zoek beide plaatsen op in `stated`, en tel 1 op als de plaats van `clicked` het grotere getal is. Geef `count` na de loop terug."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `stated` and `clicks` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `stated` en `clicks` uit de aanroep?"},
            "correct": {"en": "Your `contradictions` is right in all {n} cases. Under the check, your ring lies around the course's dot.",
                        "nl": "Je `contradictions` klopt in alle {n} gevallen. Onder de check ligt jouw ring om de stip van de course."},
        },
    ),
    "circle": PairBox(
        name="circle",
        picture=circle_picture,
        function="circle_of_three",
        start=_R_START,
        solution=_R,
        visible=[CIRCLE_CASES[0]],
        hidden=CIRCLE_CASES[1:],
        first=[_DOTS],
        show=show_value,
        compare=same_answer,
        mistakes=[
            Mistake("chain", wrong=_R_CHAIN,
                    en="`{call}` gives {got}, and it should give {want}: you return `True` as soon as one pair follows another, as in a over b and b over d. That is a chain, which can be perfectly consistent. A circle also needs the way back: the pair `(d, a)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft `True` zodra het ene paar op het andere volgt, zoals a boven b en b boven d. Dat is een keten, en die kan prima kloppen. Een kring heeft ook de weg terug nodig: het paar `(d, a)`."),
            Mistake("direction", wrong=_R_DIRECTION,
                    en="`{call}` gives {got}, and it should give {want}: you look for the pair `(a, d)`, a over d. That is what a consistent ranking does. In a circle, the third pair goes back: d over a.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zoekt het paar `(a, d)`, a boven d. Dat doet een rangorde die klopt. In een kring gaat het derde paar terug: d boven a."),
            Mistake("early", wrong=_R_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return False` is inside the first loop, so the function gives up after the first pair `(a, b)`. Put it after both loops, as far in as the first `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return False` staat in de eerste loop, dus de functie geeft het op na het eerste paar `(a, b)`. Zet het na beide loops, even ver ingesprongen als de eerste `for`."),
            Mistake("nochain", wrong=_R_NOCHAIN,
                    en="`{call}` gives {got}, and it should give {want}: you only check whether a pair leads back from `d` to `a`. The second pair also has to start where the first one ends: `b` and `c` must be the same option.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je kijkt alleen of er een paar terugleidt van `d` naar `a`. Het tweede paar moet ook beginnen waar het eerste eindigt: `b` en `c` moeten dezelfde optie zijn."),
            Mistake("pair", wrong=_R_PAIR,
                    en="`{call}` gives {got}, and it should give {want}: you look for a pair that also appears the other way round, a over b and b over a. The circle of this exercise has three options: a over b, b over d and d over a.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zoekt een paar dat ook andersom voorkomt, a boven b en b boven a. De kring van deze oefening heeft drie opties: a boven b, b boven d en d boven a."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through every pair `(a, b)`, and for each go through every pair `(c, d)`; return `True` when they go round with a third pair, and `False` after both loops.",
                      "nl": "Vervang de `...`: loop door elk paar `(a, b)`, en loop daarbinnen door elk paar `(c, d)`; geef `True` als ze met een derde paar rondgaan, en `False` na beide loops."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Go through every pair `(a, b)` in `prefs`, and inside that through every pair `(c, d)`. When `b` and `c` are the same option and the pair that goes back from `d` to `a` is also in `prefs`, return `True`. After both loops, return `False`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Loop door elk paar `(a, b)` in `prefs`, en daarbinnen door elk paar `(c, d)`. Als `b` en `c` dezelfde optie zijn en het paar dat van `d` terug naar `a` gaat ook in `prefs` staat, geef dan `True`. Geef na beide loops `False`."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `prefs` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie wel de `prefs` uit de aanroep?"},
            "correct": {"en": "Your `circle_of_three` is right in all {n} cases. Under the drawing, your answer is the course's.",
                        "nl": "Je `circle_of_three` klopt in alle {n} gevallen. Onder de tekening is jouw antwoord dat van de course."},
        },
    ),
}

CASES = [
    ("contradictions", _N_START, "empty", None),
    ("contradictions", _N, "correct", None),
    ("contradictions", "def contradictions(stated, clicks):\n    return sum(1 for c, s in clicks if stated[c] > stated[s])\n", "correct", None),
    ("contradictions", "def contradictions(stated, clicks):\n    count = 0\n    for click in clicks:\n        if stated[click[0]] > stated[click[1]]:\n            count += 1\n    return count\n", "correct", None),
    ("contradictions", _N_TUPLE, "error", "tuple"),
    ("contradictions", _N_AGREE, "wrong", "agree"),
    ("contradictions", _N_EVERY, "wrong", "every"),
    ("contradictions", _N_RESET, "wrong", "reset"),
    ("contradictions", _N_FIRST, "wrong", "first"),
    ("contradictions", _N_WORDS, "wrong", "words"),
    ("contradictions", "def contradictions(stated, clicks):\n    return 3\n", "almost", None),
    ("circle", _R_START, "empty", None),
    ("circle", _R, "correct", None),
    ("circle", "def circle_of_three(prefs):\n    for a, b in prefs:\n        for c, d in prefs:\n            for e, f in prefs:\n                if b == c and d == e and f == a:\n                    return True\n    return False\n", "correct", None),
    ("circle", "def circle_of_three(prefs):\n    return any(b == c and (d, a) in prefs for a, b in prefs for c, d in prefs)\n", "correct", None),
    ("circle", _R_CHAIN, "wrong", "chain"),
    ("circle", _R_DIRECTION, "wrong", "direction"),
    ("circle", _R_EARLY, "wrong", "early"),
    ("circle", _R_NONE, "wrong", "unreached"),
    ("circle", _R_NOCHAIN, "wrong", "nochain"),
    ("circle", _R_PAIR, "wrong", "pair"),
    ("circle", "def circle_of_three(prefs):\n    return True\n", "almost", None),
]
