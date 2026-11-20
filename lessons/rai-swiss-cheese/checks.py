# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-swiss-cheese: safeguards in a row. The learner writes gets_through, the holes that line up
in three safeguards (the Swiss cheese model), and top_event, a fault tree with an AND gate over safeguards that miss
independently and an OR gate with a shared trick that fools them all. A demo puts one to five safeguards in a row."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

MISS = 0.1            # the chance that one safeguard misses an ordinary harmful request
SHARED = 0.01         # the share of harmful requests that use a trick which fools every safeguard
REQUESTS = 1000000    # the harmful requests the demo and the guess count
LAYERS = [1, 2, 3, 4, 5]   # the demo's slider: how many safeguards stand in a row


def gets_through(first, second, third):
    """What the learner writes in Exercise 1: the holes of `first` that the second and the third safeguard have too, in
    the order of `first`."""
    through = []
    for hole in first:
        if hole in second and hole in third:
            through.append(hole)
    return through


def top_event(misses, shared):
    """What the learner writes in Exercise 2: every safeguard misses (AND, independent), or the shared trick (OR)."""
    all_miss = 1
    for miss in misses:
        all_miss = all_miss * miss
    return 1 - (1 - shared) * (1 - all_miss)


def nines(p: float) -> float:
    """The nines of reliability of a chance of failure p, as in the lesson rai-nines."""
    return -math.log10(p)


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_CHIP = ("display:inline-block;margin:0.1rem 0.2rem 0.1rem 0;padding:0 0.3rem;border:1px solid var(--ea-line);"
         "border-radius:3px;font-family:var(--ea-font-code);font-size:0.76rem;white-space:nowrap")
_LINED_UP = ";background:var(--ea-loss-wash);border-color:var(--ea-loss);color:var(--ea-loss-text)"


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


def _number(v):
    """v as a float, or None for something that is not a number, such as True or a text."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _value(v) -> str:
    """A result as the check's messages and the tables show it: a number with at most six decimals, a tiny one in
    e-notation, else what Python shows, cut short when it is long."""
    f = _number(v)
    if f is None or f != f or f in (float("inf"), float("-inf")):
        text = repr(v) if f is None else str(f)
        return text if len(text) <= 40 else text[:37] + "..."
    if f != 0 and abs(f) < 1e-6:
        return f"{f:.3g}"
    text = f"{f:.6f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _prose(x: float, lang: str, digits: int = 6) -> str:
    """A number in a picture's words: at most `digits` decimals, with a decimal comma in Dutch."""
    text = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


def _whole(v, lang: str) -> str:
    """A whole number as the text writes it: 1,000 in English, 1000 in Dutch, which uses a dot only from 10.000."""
    n = int(round(float(v)))
    if lang == "en":
        return f"{n:,}"
    return f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n)


def _percent(x: float, lang: str, digits: int = 2) -> str:
    """A chance as a percentage with at most `digits` decimals: 0.01 as 1%."""
    text = f"{x * 100:.{digits}f}".rstrip("0").rstrip(".") + "%"
    return text.replace(".", ",") if lang == "nl" else text


_DEMO_WORDS = {
    "en": {"title": "{n} {safeguards} in a row: what gets through",
           "one": "safeguard", "many": "safeguards", "miss_one": "that misses 10%", "miss_many": "that each miss 10%",
           "head": ["", "get through, of 1,000,000", "nines"],
           "rows": ["without the shared trick", "with the shared trick, 1%"],
           "said": "With {n} {safeguards} {miss}, {alone} of 1,000,000 harmful requests get through without the shared "
                   "trick, and {shared} when 1% of the requests use a trick that fools every safeguard.",
           "x": "safeguards in a row", "y": "nines",
           "note": "Each safeguard misses 10% of the ordinary harmful requests, independently of the others. With the "
                   "shared trick, 1% of the requests fool every safeguard at once. The dots mark the number on the "
                   "slider; the nines are rounded down to two decimals."},
    "nl": {"title": "{n} {safeguards} achter elkaar: wat erdoor komt",
           "one": "safeguard", "many": "safeguards", "miss_one": "die 10% mist", "miss_many": "die elk 10% missen",
           "head": ["", "komen erdoor, van 1.000.000", "negens"],
           "rows": ["zonder de gedeelde truc", "met de gedeelde truc, 1%"],
           "said": "Met {n} {safeguards} {miss}, komen {alone} van de 1.000.000 schadelijke verzoeken erdoor zonder de "
                   "gedeelde truc, en {shared} als 1% van de verzoeken een truc gebruikt die elke safeguard om de tuin "
                   "leidt.",
           "x": "safeguards achter elkaar", "y": "negens",
           "note": "Elke safeguard mist 10% van de gewone schadelijke verzoeken, onafhankelijk van de andere. Met de "
                   "gedeelde truc leidt 1% van de verzoeken alle safeguards tegelijk om de tuin. De stippen staan bij het "
                   "aantal op de schuif; de negens zijn naar beneden afgerond op twee decimalen."},
}


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: with n safeguards that each miss 10% independently, how many of 1,000,000 harmful requests get
    through, with and without a shared trick that fools them all, and how many nines that is."""
    n = min(LAYERS, key=lambda m: abs(m - float(v)))
    w = _DEMO_WORDS[lang]
    alone, shared = MISS ** n, top_event([MISS] * n, SHARED)
    rows = [([(html.escape(name), ""), (_code(_whole(REQUESTS * p, lang)), "text-align:right"),
              (_code(_prose(math.floor(round(nines(p), 6) * 100) / 100, lang, 2)), "text-align:right")], "")
            for name, p in zip(w["rows"], (alone, shared))]
    panel = chart.Panel(x=(1, 5), y=(0, 5.6), xlabel=w["x"], ylabel=w["y"], xticks=LAYERS, yticks=[0, 1, 2, 3, 4, 5])
    panel.line(LAYERS, [nines(MISS ** k) for k in LAYERS], role="ink-3")
    panel.line(LAYERS, [nines(top_event([MISS] * k, SHARED)) for k in LAYERS], role="loss")
    panel.points([n, n], [nines(alone), nines(shared)], role="loss")
    for y, name, role in ((5.2, 0, "ink-3"), (4.6, 1, "loss")):   # a key at the top left, where no line runs
        panel.line([1.08, 1.32], [y, y], role=role)
        panel.text(1.4, y, w["rows"][name], role=role, anchor="start")
    word = w["one"] if n == 1 else w["many"]
    title = w["title"].format(n=n, safeguards=word)
    said = w["said"].format(n=n, safeguards=word, miss=w["miss_one"] if n == 1 else w["miss_many"], alone=_whole(REQUESTS * alone, lang),
                            shared=_whole(REQUESTS * shared, lang))
    body = (_table(rows, [html.escape(h) for h in w["head"]])
            + chart.figure([panel], alt=title + ". " + said, width=360, height=240) + _note(w["note"]))
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _holes(holes: list, lined_up: list, none: str) -> str:
    """A safeguard's holes as chips; the holes that line up through all three are filled."""
    if not holes:
        return f'<span style="{_UI}">{html.escape(none)}</span>'
    return "".join(f'<span style="{_CHIP}{_LINED_UP if h in lined_up else ""}">{html.escape(str(h))}</span>'
                   for h in holes)


_HOLE_WORDS = {
    "en": {"title": "Your gets_through beside the course's",
           "head": ["first", "second", "third", "the course's", "yours"],
           "none": "none", "stopped": "error",
           "note": "Each list is a safeguard's holes: the kinds of trick it misses. A filled hole is in all three lists, "
                   "so a request with that trick gets through every safeguard. A dashed row is one where your "
                   "gets_through gives something other than the course's."},
    "nl": {"title": "Jouw gets_through naast die van de course",
           "head": ["first", "second", "third", "van de course", "de jouwe"],
           "none": "geen", "stopped": "fout",
           "note": "Elke lijst bevat de gaten van een safeguard: de soorten trucs die de safeguard mist. Een gevuld gat "
                   "staat in alle drie de lijsten, dus een verzoek met die truc komt door elke safeguard. Een "
                   "gestippelde rij is er een waar jouw gets_through iets anders geeft dan die van de course."},
}


def holes_picture(seen) -> str:
    """The cases of the check: the holes of the three safeguards, with the ones that line up filled, the course's list
    and the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _HOLE_WORDS[seen.lang]
    got = _all(fn, HOLE_CASES)
    rows = []
    for i, (first, second, third) in enumerate(HOLE_CASES):
        want = gets_through(first, second, third)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        elif isinstance(got[i], list) and all(isinstance(h, str) for h in got[i]):
            shown, off = _holes(got[i], want, w["none"]), got[i] != want
        else:
            shown, off = _code(_value(got[i])), True
        rows.append(([(_holes(first, want, w["none"]), ""), (_holes(second, want, w["none"]), ""),
                      (_holes(third, want, w["none"]), ""), (_holes(want, want, w["none"]), ""), (shown, "")],
                     _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_TREE_WORDS = {
    "en": {"title": "Your top_event in the fault tree, beside the course's",
           "top": "a harmful request gets through", "or": "OR", "and": "AND",
           "shared": "shared trick", "layer": "safeguard {i}", "misses": "misses {p}", "all": "all miss: {p}",
           "values": "the course's {want} · yours {got}",
           "head": ["misses", "shared", "all miss", "the course's", "yours"], "stopped": "error",
           "note": "The tree is the first case of the check. Under the AND gate, the chances that every safeguard "
                   "misses multiply; at the OR gate, the top event happens when the shared trick or the AND gate "
                   "happens. A dashed row is one where your top_event gives something other than the course's."},
    "nl": {"title": "Jouw top_event in de foutenboom, naast die van de course",
           "top": "een schadelijk verzoek komt erdoor", "or": "OF", "and": "EN",
           "shared": "gedeelde truc", "layer": "safeguard {i}", "misses": "mist {p}", "all": "elk mist: {p}",
           "values": "van de course {want} · de jouwe {got}",
           "head": ["misses", "shared", "elk mist", "van de course", "de jouwe"], "stopped": "fout",
           "note": "De boom is het eerste geval van de check. Onder de EN-poort vermenigvuldigen de kansen dat elke "
                   "safeguard mist; bij de OF-poort gebeurt de topgebeurtenis als de gedeelde truc of de EN-poort "
                   "gebeurt. Een gestippelde rij is er een waar jouw top_event iets anders geeft dan die van de "
                   "course."},
}


def _tree(misses: list, shared: float, got, lang: str) -> str:
    """The fault tree of one case: the top event over an OR gate, with the shared trick on one side and an AND gate
    over the safeguards on the other; the top box gives the course's chance and the learner's."""
    w = _TREE_WORDS[lang]
    ink, line, page = "var(--ea-ink)", "var(--ea-ink-3)", "var(--ea-page)"
    out = []

    def box(x, y, width, top, bottom, role=ink):
        out.append(f'<rect x="{x - width / 2:.1f}" y="{y - 19}" width="{width}" height="38" rx="4" '
                   f'style="fill:{page};stroke:{role};stroke-width:1.2"/>')
        out.append(f'<text x="{x:.1f}" y="{y - 3}" style="font-size:12px;fill:{ink};text-anchor:middle">'
                   f'{html.escape(top)}</text>')
        out.append(f'<text x="{x:.1f}" y="{y + 12}" style="font-size:11.5px;fill:var(--ea-ink-2);text-anchor:middle;'
                   f'font-family:var(--ea-font-code)">{html.escape(bottom)}</text>')

    def gate(x, y, label):
        out.append(f'<rect x="{x - 22}" y="{y - 11}" width="44" height="22" rx="11" '
                   f'style="fill:{page};stroke:{ink};stroke-width:1.2"/>')
        out.append(f'<text x="{x}" y="{y + 4}" style="font-size:11px;font-weight:600;fill:{ink};text-anchor:middle">'
                   f'{html.escape(label)}</text>')

    def link(x1, y1, x2, y2):
        out.append(f'<line x1="{x1:.1f}" y1="{y1}" x2="{x2:.1f}" y2="{y2}" style="stroke:{line};stroke-width:1"/>')

    all_miss = 1
    for m in misses:
        all_miss = all_miss * m
    want = top_event(misses, shared)
    yours_ = w["stopped"] if got is _STOPPED else (_prose(got, lang) if _number(got) is not None else _value(got))
    off = got is _STOPPED or _number(got) is None or not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-9)
    k = len(misses)
    xs = [400 + (i - (k - 1) / 2) * 92 for i in range(k)]
    link(270, 45, 270, 67)
    link(270, 89, 140, 111)
    link(270, 89, 400, 119)
    for x in xs:
        link(400, 141, x, 171)
    box(270, 26, 300, w["top"], w["values"].format(want=_prose(want, lang), got=yours_),
        role="var(--ea-loss)" if off else ink)
    gate(270, 78, w["or"])
    box(140, 130, 150, w["shared"], _percent(shared, lang))
    gate(400, 130, w["and"])
    out.append(f'<text x="428" y="134" style="font-size:11px;fill:var(--ea-ink-2)">'
               f'{html.escape(w["all"].format(p=_prose(all_miss, lang)))}</text>')
    for i, (x, m) in enumerate(zip(xs, misses)):
        box(x, 190, 86, w["layer"].format(i=i + 1), w["misses"].format(p=_percent(m, lang)))
    return (f'<svg viewBox="0 0 545 212" role="img" aria-label="{html.escape(w["title"])}" '
            f'style="width:100%;max-width:545px;height:auto;font-family:var(--ea-font-ui)">{"".join(out)}</svg>')


def tree_picture(seen) -> str:
    """The fault tree of the first case with the learner's chance at the top, and every case of the check: the misses,
    the shared trick, the chance that all miss, the course's top event and the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _TREE_WORDS[seen.lang]
    got = _all(fn, TOP_CASES)
    rows = []
    for i, (misses, shared) in enumerate(TOP_CASES):
        want = top_event(misses, shared)
        all_miss = 1
        for m in misses:
            all_miss = all_miss * m
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown = _code(_value(got[i]))
            off = _number(got[i]) is None or not math.isclose(got[i], want, rel_tol=1e-6, abs_tol=1e-9)
        rows.append(([(_code(repr(misses)), ""), (_code(_value(shared)), "text-align:right"),
                      (_code(_value(all_miss)), "text-align:right"), (_code(_value(want)), "text-align:right"),
                      (shown, "text-align:right;font-weight:600")], _OFF if off else ""))
    first = _STOPPED if got is _STOPPED else got[0]
    body = (_tree(*TOP_CASES[0], first, seen.lang) + _table(rows, [html.escape(h) for h in w["head"]])
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["gets_through", "top_event", "common_cause", "missing_layer"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: gets_through; the first case has one hole that lines up, and one that is in two of the three
HOLE_CASES = [(["role play", "base64", "prefix"], ["base64", "role play", "many-shot"], ["role play", "other language"]),
              (["base64", "prefix"], ["prefix"], ["base64"]),
              (["role play", "base64"], ["base64", "role play"], ["base64", "role play", "prefix"]),
              ([], ["base64"], ["base64"]),
              (["many-shot"], ["many-shot"], ["many-shot"]),
              (["prefix", "many-shot", "base64"], ["many-shot", "base64"], ["other language", "prefix"]),
              (["other language", "base64"], ["base64", "other language"], ["other language", "base64"]),
              (["base64"], [], ["base64"])]
_G_START = "def gets_through(first, second, third):\n    ...\n"
_G = ("def gets_through(first, second, third):\n"
      "    through = []\n"
      "    for hole in first:\n"
      "        if hole in second and hole in third:\n"
      "            through.append(hole)\n"
      "    return through\n")
_G_OR = _G.replace("if hole in second and hole in third:", "if hole in second or hole in third:")
_G_TWO = _G.replace("if hole in second and hole in third:", "if hole in second:")
_G_NOT = _G.replace("if hole in second and hole in third:", "if hole not in second and hole not in third:")
_G_ORDER = _G.replace("    for hole in first:\n        if hole in second and hole in third:\n",
                      "    for hole in third:\n        if hole in first and hole in second:\n")
_G_EARLY = _G.replace("    return through\n", "        return through\n")
_G_ALL = _G.replace("        if hole in second and hole in third:\n            through.append(hole)\n",
                    "        through.append(hole)\n")

# exercise 2: top_event; the first case is the textbook's three safeguards with a shared trick for 1% of the requests
TOP_CASES = [([0.1, 0.1, 0.1], 0.01), ([0.1, 0.1, 0.1], 0), ([0.5], 0.5), ([0.2, 0.5], 0), ([], 0.02), ([0.1, 0.1], 1),
             ([0.3, 0.2, 0.1, 0.5], 0.05), ([0.9], 0.2)]
_T_START = "def top_event(misses, shared):\n    ...\n"
_T = ("def top_event(misses, shared):\n"
      "    all_miss = 1\n"
      "    for miss in misses:\n"
      "        all_miss = all_miss * miss\n"
      "    return 1 - (1 - shared) * (1 - all_miss)\n")
_T_SUM = _T.replace("    all_miss = 1\n", "    all_miss = 0\n").replace("all_miss * miss", "all_miss + miss")
_T_RARE = _T.replace("return 1 - (1 - shared) * (1 - all_miss)", "return shared + all_miss")
_T_NO_OR = _T.replace("return 1 - (1 - shared) * (1 - all_miss)", "return all_miss")
_T_PRODUCT = _T.replace("return 1 - (1 - shared) * (1 - all_miss)", "return shared * all_miss")
_T_ZERO = _T.replace("    all_miss = 1\n", "    all_miss = 0\n")
_T_ANY = ("def top_event(misses, shared):\n"
          "    all_pass = 1\n"
          "    for miss in misses:\n"
          "        all_pass = all_pass * (1 - miss)\n"
          "    return 1 - (1 - shared) * all_pass\n")


EXERCISES = {
    "gets_through": FunctionBox(
        name="gets_through",
        picture=holes_picture,
        function="gets_through",
        start=_G_START,
        solution=_G,
        visible=[HOLE_CASES[0]],
        hidden=HOLE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("or", wrong=_G_OR,
                    en="`{call}` gives {got}, and it should give {want}: you let a hole through when it is in the second or in the third list. A request with that trick still meets the safeguard that has no such hole: it gets through only when the hole is in all three lists.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je laat een gat door als het in de tweede of in de derde lijst staat. Een verzoek met die truc komt dan nog langs de safeguard die zo'n gat niet heeft: het komt er alleen door als het gat in alle drie de lijsten staat."),
            Mistake("two", wrong=_G_TWO,
                    en="`{call}` gives {got}, and it should give {want}: you only look at `second`. The request still meets the third safeguard: the hole must be in `third` too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je kijkt alleen naar `second`. Het verzoek komt daarna nog langs de derde safeguard: het gat moet ook in `third` staan."),
            Mistake("not", wrong=_G_NOT,
                    en="`{call}` gives {got}, and it should give {want}: you keep the holes that are in neither of the other lists. The other safeguards catch those; the requests that get through use a hole that is in all three lists.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je houdt de gaten over die in geen van de andere lijsten staan. Die vangen de andere safeguards op; de verzoeken die erdoor komen, gebruiken een gat dat in alle drie de lijsten staat."),
            Mistake("order", wrong=_G_ORDER,
                    en="`{call}` gives {got}, and it should give {want}: the right holes, but in another order. Go through `first`, so that they come in its order.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de goede gaten, maar in een andere volgorde. Loop door `first`, zodat ze in die volgorde komen."),
            Mistake("early", wrong=_G_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so the function stops after the first hole. Put `return through` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na het eerste gat. Zet `return through` na de loop, op gelijke hoogte met `for`."),
            Mistake("all", wrong=_G_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you return every hole of the first safeguard. The second and the third catch most of them: keep only the holes that the other two lists have too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alle gaten van de eerste safeguard terug. De tweede en de derde vangen de meeste daarvan op: houd alleen de gaten over die ook in de andere twee lijsten staan."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go through `first`, keep each hole that is also in `second` and in `third`, and return the list of those holes.",
                      "nl": "Vervang de `...`: loop door `first`, houd elk gat over dat ook in `second` en in `third` staat, en geef de lijst met die gaten terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Start with an empty list `through`. For each `hole` in `first` that is in both other lists, add it with `append`; return `through` after the loop.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Begin met een lege lijst `through`. Voeg elk `hole` uit `first` dat in beide andere lijsten staat toe met `append`; geef na de loop `through` terug."},
            "almost": {"en": "Right for the first safeguards. But `{call}` gives {got}, and it should give {want}. Does your function look at the three lists it gets?",
                       "nl": "Goed voor de eerste safeguards. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie naar de drie lijsten die ze krijgt?"},
            "correct": {"en": "Your `gets_through` is right in all {n} cases. Under the check, the holes that line up are filled, with your list beside the course's.",
                        "nl": "Je `gets_through` klopt in alle {n} gevallen. Onder de check zijn de gaten gevuld die op één lijn liggen, met jouw lijst naast die van de course."},
        },
    ),
    "top_event": FunctionBox(
        name="top_event",
        picture=tree_picture,
        function="top_event",
        start=_T_START,
        solution=_T,
        visible=[TOP_CASES[0]],
        hidden=TOP_CASES[1:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("sum", wrong=_T_SUM,
                    en="`{call}` gives {got}, and it should give {want}: you add the chances that the safeguards miss. A request gets through all of them only when every one misses: with independent safeguards, those chances multiply. Start `all_miss` at 1 and multiply it by each `miss`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de kansen op dat de safeguards missen. Een verzoek komt alleen door allemaal als elke safeguard mist: bij onafhankelijke safeguards vermenigvuldigen die kansen. Begin `all_miss` op 1 en vermenigvuldig met elke `miss`."),
            Mistake("rare-event", wrong=_T_RARE,
                    en="`{call}` gives {got}, and it should give {want}: you add the two chances at the OR gate. That is close when both are small, but counts twice the requests where both happen: with `[0.5]` and `0.5` it gives 1, as if every request got through. The top event does not happen only when neither branch happens; take one minus the chance of that.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je telt de twee kansen bij de OF-poort op. Dat komt in de buurt als ze allebei klein zijn, maar telt de verzoeken waarbij allebei gebeuren dubbel: met `[0.5]` en `0.5` geeft het 1, alsof elk verzoek erdoor kwam. De topgebeurtenis gebeurt alleen niet als geen van beide takken gebeurt; neem één min de kans daarop."),
            Mistake("no-or", wrong=_T_NO_OR,
                    en="`{call}` gives {got}, and it should give {want}: you give only the chance that every safeguard misses. The top event also happens when the shared trick fools them all: combine the two at the OR gate.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je geeft alleen de kans dat elke safeguard mist. De topgebeurtenis gebeurt ook als de gedeelde truc ze allemaal om de tuin leidt: combineer de twee bij de OF-poort."),
            Mistake("product", wrong=_T_PRODUCT,
                    en="`{call}` gives {got}, and it should give {want}: you multiply the chance of the shared trick by the chance that every safeguard misses. That is an AND, both at once. The top event happens when either branch happens: an OR gate.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt de kans op de gedeelde truc met de kans dat elke safeguard mist. Dat is een EN, allebei tegelijk. De topgebeurtenis gebeurt als een van beide takken gebeurt: een OF-poort."),
            Mistake("zero", wrong=_T_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: `all_miss` starts at 0, so multiplying keeps it 0. Start it at 1: before any safeguard, every request still gets through.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `all_miss` begint op 0, dus vermenigvuldigen houdt het op 0. Begin op 1: voor de eerste safeguard komt nog elk verzoek erdoor."),
            Mistake("any", wrong=_T_ANY,
                    en="`{call}` gives {got}, and it should give {want}: you compute the chance that at least one safeguard misses. A request gets through only when every safeguard misses it: multiply the chances that they miss.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je rekent de kans uit dat minstens één safeguard mist. Een verzoek komt er alleen door als elke safeguard het mist: vermenigvuldig de kansen dat ze missen."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: multiply the chances in `misses` into `all_miss`, then combine `all_miss` and `shared` at the OR gate and return that chance.",
                      "nl": "Vervang de `...`: vermenigvuldig de kansen in `misses` tot `all_miss`, combineer dan `all_miss` en `shared` bij de OF-poort en geef die kans terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Under the AND gate, multiply the chances that the safeguards miss. At the OR gate, the top event does not happen only when neither the shared trick nor the AND gate happens: return one minus that chance.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vermenigvuldig onder de EN-poort de kansen dat de safeguards missen. Bij de OF-poort gebeurt de topgebeurtenis alleen niet als de gedeelde truc en de EN-poort allebei niet gebeuren: geef één min die kans terug."},
            "almost": {"en": "Right for the first tree. But `{call}` gives {got}, and it should give {want}. Does your function use the `misses` and the `shared` it gets?",
                       "nl": "Goed voor de eerste boom. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `misses` en de `shared` die ze krijgt?"},
            "correct": {"en": "Your `top_event` is right in all {n} cases. Under the check, the fault tree of the first case carries your chance at the top.",
                        "nl": "Je `top_event` klopt in alle {n} gevallen. Onder de check staat jouw kans bovenaan de foutenboom van het eerste geval."},
        },
    ),
}

CASES = [
    ("gets_through", _G_START, "empty", None),
    ("gets_through", _G, "correct", None),
    ("gets_through", "def gets_through(first, second, third):\n    return [hole for hole in first if hole in second if hole in third]\n", "correct", None),
    ("gets_through", "def gets_through(first, second, third):\n    through = []\n    for hole in first:\n        if hole in second:\n            if hole in third:\n                through += [hole]\n    return through\n", "correct", None),
    ("gets_through", _G_OR, "wrong", "or"),
    ("gets_through", _G_TWO, "wrong", "two"),
    ("gets_through", _G_NOT, "wrong", "not"),
    ("gets_through", _G_ORDER, "wrong", "order"),
    ("gets_through", _G_EARLY, "wrong", "early"),
    ("gets_through", _G_ALL, "wrong", "all"),
    ("gets_through", "def gets_through(first, second, third):\n    return [\"role play\"]\n", "almost", None),
    ("top_event", _T_START, "empty", None),
    ("top_event", _T, "correct", None),
    ("top_event", "import math\n\n\ndef top_event(misses, shared):\n    all_miss = math.prod(misses)\n    return shared + all_miss - shared * all_miss\n", "correct", None),
    ("top_event", _T_SUM, "wrong", "sum"),
    ("top_event", _T_RARE, "wrong", "rare-event"),
    ("top_event", _T_NO_OR, "wrong", "no-or"),
    ("top_event", _T_PRODUCT, "wrong", "product"),
    ("top_event", _T_ZERO, "wrong", "zero"),
    ("top_event", _T_ANY, "wrong", "any"),
    ("top_event", "def top_event(misses, shared):\n    return 0.01099\n", "almost", None),
]
