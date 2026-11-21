# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-meaning-and-truth: decide whether every thing of a list is in another list (every), find
the things that do something to a given thing (doers), and decide whether a sentence is true in a small world from the
meanings of its parts (is_true). The world of four animals, the second world NIGHT, the sentences and the 64 worlds of
the demo are the course's own; the trees are written as lists in the course's own short notation."""
import contextlib
import copy
import html
import io

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the course's world ----------------------------------------------------------------------------------------------

# The words of the world in both languages. A property is a list of things; a relation a list of pairs (doer, done).
WORD = {
    "en": {"cat": "cat", "dog": "dog", "mouse": "mouse", "unicorn": "unicorn", "sleeps": "sleeps", "chases": "chases",
           "every": "every", "some": "some"},
    "nl": {"cat": "kat", "dog": "hond", "mouse": "muis", "unicorn": "eenhoorn", "sleeps": "slaapt",
           "chases": "achtervolgt", "every": "elke", "some": "een"},
}
THINGS = ["Tom", "Mia", "Rex", "Pip"]
KIND = {"Tom": "cat", "Mia": "cat", "Rex": "dog", "Pip": "mouse"}


def _world(sleeps: list, chases: list, lang: str) -> dict:
    w = WORD[lang]
    out = {w[k]: [t for t in THINGS if KIND[t] == k] for k in ("cat", "dog", "mouse")}
    out[w["sleeps"]] = list(sleeps)
    out[w["chases"]] = [tuple(p) for p in chases]
    return out


# WORLD: Mia and Rex sleep; Tom chases Pip and Rex chases Tom. NIGHT: Tom, Mia and Rex sleep, and Pip chases Tom.
WORLD = {lang: _world(["Mia", "Rex"], [("Tom", "Pip"), ("Rex", "Tom")], lang) for lang in WORD}
NIGHT = {lang: _world(["Tom", "Mia", "Rex"], [("Pip", "Tom")], lang) for lang in WORD}

# The five sentences of Exercise 3: subject, verb, object (None for a verb without object).
_PLAN = [("Mia", "sleeps", None), ("Tom", "sleeps", None), ("Rex", "chases", "Tom"), ("Tom", "chases", "Rex"),
         ("Pip", "chases", "Tom")]


def _sentence(subject: str, verb: str, obj, lang: str) -> list:
    vp = ["VP", WORD[lang][verb]] + ([["NP", obj]] if obj else [])
    return ["S", ["NP", subject], vp]


SENTENCES = {lang: [_sentence(*p, lang) for p in _PLAN] for lang in WORD}


def words_of(tree) -> list:
    if isinstance(tree, str):
        return [tree]
    out = []
    for part in tree[1:]:
        out = out + words_of(part)
    return out


def text_of(tree) -> str:
    return " ".join(words_of(tree))


# --- the course's functions ------------------------------------------------------------------------------------------

def some(things, group):
    inside = [thing for thing in things if thing in group]
    return len(inside) > 0


def every(things, group):
    inside = [thing for thing in things if thing in group]
    return len(inside) == len(things)


def doers(pairs, target):
    return [pair[0] for pair in pairs if pair[1] == target]


def is_true(tree, world):
    vp = tree[2]
    if len(vp) == 2:
        group = world[vp[1]]
    else:
        group = doers(world[vp[1]], vp[2][1])
    return tree[1][1] in group


def meaning_of_vp(vp, world):
    """The meaning of a verb phrase: the list of things it is true of."""
    if len(vp) == 2:
        return list(world[vp[1]])
    return doers(world[vp[1]], vp[2][1])


def truth(tree, world, lang: str) -> bool:
    """The course's own evaluator, also for a subject with *every* or *some*: the demo uses it."""
    np, group = tree[1], meaning_of_vp(tree[2], world)
    if len(np) == 2:
        return np[1] in group
    things = world.get(np[2], [])
    return every(things, group) if np[1] == WORD[lang]["every"] else some(things, group)


# --- the demo: six facts, 64 worlds ----------------------------------------------------------------------------------

FACTS = [("sleeps", "Tom"), ("sleeps", "Mia"), ("sleeps", "Rex"), ("sleeps", "Pip"),
         ("chases", ("Tom", "Pip")), ("chases", ("Rex", "Tom"))]
FACTS_ON = [False, True, True, False, True, True]   # the facts of WORLD


def fact_text(fact, lang: str) -> str:
    verb, arg = fact
    if verb == "sleeps":
        return f'{arg} {WORD[lang]["sleeps"]}'
    return f'{arg[0]} {WORD[lang]["chases"]} {arg[1]}'


def world_of(on: list, lang: str) -> dict:
    sleeps = [f[1] for f, o in zip(FACTS, on) if o and f[0] == "sleeps"]
    chases = [f[1] for f, o in zip(FACTS, on) if o and f[0] == "chases"]
    return _world(sleeps, chases, lang)


def all_worlds() -> list:
    """The 64 ways to switch the six facts on and off; world k has fact i on when bit i of k is 1."""
    return [[bool(k >> i & 1) for i in range(len(FACTS))] for k in range(2 ** len(FACTS))]


_DEMO_PLAN = [
    ("Mia", None, "sleeps", None),
    ("Pip", None, "sleeps", None),
    ("cat", "every", "sleeps", None),
    ("cat", "some", "sleeps", None),
    ("mouse", "every", "sleeps", None),
    ("Rex", None, "chases", "Tom"),
]
DEMO_LABELS = {
    "en": ["Mia sleeps", "Pip sleeps", "Every cat sleeps", "Some cat sleeps", "Every mouse sleeps", "Rex chases Tom"],
    "nl": ["Mia slaapt", "Pip slaapt", "Elke kat slaapt", "Er slaapt een kat", "Elke muis slaapt", "Rex achtervolgt Tom"],
}


def demo_tree(k: int, lang: str) -> list:
    head, det, verb, obj = _DEMO_PLAN[k]
    w = WORD[lang]
    np = ["NP", head] if det is None else ["NP", w[det], w[head]]
    vp = ["VP", w[verb]] + ([["NP", obj]] if obj else [])
    return ["S", np, vp]


def true_in(k: int, lang: str) -> list:
    """For demo sentence k: in which of the 64 worlds it is true."""
    tree = demo_tree(k, lang)
    return [truth(tree, world_of(on, lang), lang) for on in all_worlds()]


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = ";color:var(--ea-loss-text)"
_CODE = ";font-family:var(--ea-font-code);font-size:0.78rem"
_CAPTION = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin:0 0 0.2rem"

_KIND_WORDS = {"en": {"cat": "cat", "dog": "dog", "mouse": "mouse"}, "nl": {"cat": "kat", "dog": "hond", "mouse": "muis"}}
_SVG_IDS = [0]


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(head: list, rows: list) -> str:
    """rows: (cells, off); a cell is (text, is_code)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = ""
    for cells, off in rows:
        body += "<tr>" + "".join(
            f'<td style="{_CELL}{_CODE if code else ""}{_OFF if off else ""}">{html.escape(text)}</td>'
            for text, code in cells) + "</tr>"
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def world_svg(world: dict, lang: str, marked=(), alt: str = "") -> str:
    """The four animals as cards, a sleep label under each one that sleeps, and an arrow for each pair of chases: to
    the right above the cards, to the left below them. Cards in `marked` get a bold border."""
    w = WORD[lang]
    sleeps, chases = world.get(w["sleeps"], []), world.get(w["chases"], [])
    _SVG_IDS[0] += 1
    mid = f"ea-arrow-{_SVG_IDS[0]}"
    card_w, card_h, gap, left, top = 104, 50, 34, 10, 66
    x = {t: left + i * (card_w + gap) for i, t in enumerate(THINGS)}
    width = left * 2 + len(THINGS) * card_w + (len(THINGS) - 1) * gap
    height = top + card_h + 96
    parts = [f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>']
    for pair in chases:
        if not (isinstance(pair, tuple) and len(pair) == 2 and pair[0] in x and pair[1] in x and pair[0] != pair[1]):
            continue
        a, b = x[pair[0]] + card_w / 2, x[pair[1]] + card_w / 2
        span = abs(b - a)
        if b > a:   # to the right: above the cards
            y0, cy, ly = top - 2, top - 26 - span * 0.12, top - 18 - span * 0.06
        else:       # to the left: below the cards, under the sleep labels
            y0, cy, ly = top + card_h + 26, top + card_h + 58 + span * 0.12, top + card_h + 54 + span * 0.06
        parts.append(f'<path d="M{a + (8 if b > a else -8):.1f},{y0:.1f} Q{(a + b) / 2:.1f},{cy:.1f} '
                     f'{b - (8 if b > a else -8):.1f},{y0:.1f}" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1.6" '
                     f'marker-end="url(#{mid})"/>')
        parts.append(f'<text x="{(a + b) / 2:.1f}" y="{ly:.1f}" style="font-family:var(--ea-font-ui);font-size:12px;'
                     f'fill:var(--ea-ink-3);text-anchor:middle">{html.escape(w["chases"])}</text>')
    for t in THINGS:
        bold = t in marked
        stroke = "stroke:var(--ea-ink);stroke-width:2.4" if bold else "stroke:var(--ea-line);stroke-width:1.2"
        parts.append(f'<rect x="{x[t]}" y="{top}" width="{card_w}" height="{card_h}" rx="8" '
                     f'style="fill:var(--ea-inset);{stroke}"/>')
        parts.append(f'<text x="{x[t] + card_w / 2}" y="{top + 21}" style="font-family:var(--ea-font-ui);font-size:15px;'
                     f'font-weight:600;fill:var(--ea-ink);text-anchor:middle">{t}</text>')
        parts.append(f'<text x="{x[t] + card_w / 2}" y="{top + 39}" style="font-family:var(--ea-font-ui);font-size:12px;'
                     f'fill:var(--ea-ink-3);text-anchor:middle">{html.escape(_KIND_WORDS[lang][KIND[t]])}</text>')
        if t in sleeps:
            parts.append(f'<text x="{x[t] + card_w / 2}" y="{top + card_h + 17}" style="font-family:var(--ea-font-code);'
                         f'font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(w["sleeps"])}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;max-width:{width}px;height:auto">'
            f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg>')


def describe(world: dict, lang: str) -> str:
    """The world in words, for the alt text of a picture."""
    w = WORD[lang]
    kinds = ", ".join(f"{t} ({_KIND_WORDS[lang][KIND[t]]})" for t in THINGS)
    sleeps = ", ".join(world.get(w["sleeps"], [])) or "-"
    chases = ", ".join(f"{a} → {b}" for a, b in world.get(w["chases"], [])) or "-"
    return f'{kinds}. {w["sleeps"]}: {sleeps}. {w["chases"]}: {chases}.'


def _show(value) -> str:
    """A value the way Python shows it, with the strings of a list in double quotes."""
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, tuple):
        return "(" + ", ".join(_show(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    return repr(value)


_IDEA_WORDS = {
    "en": {"title": "A small world: four animals, who sleeps and who chases whom",
           "head": ["Word", "Its meaning in this world"],
           "legend": "A name stands for one animal. A noun and a verb without object stand for a list of animals; a verb "
                     "with an object for a list of pairs, with the doer first."},
    "nl": {"title": "Een kleine wereld: vier dieren, wie er slaapt en wie wie achtervolgt",
           "head": ["Woord", "De betekenis in deze wereld"],
           "legend": "Een naam staat voor één dier. Een zelfstandig naamwoord en een werkwoord zonder lijdend voorwerp "
                     "staan voor een lijst dieren; een werkwoord met een lijdend voorwerp voor een lijst paren, met voorop "
                     "wie de handeling uitvoert."},
}


def idea_picture(lang: str) -> str:
    w, world = _IDEA_WORDS[lang], WORLD[lang]
    rows = [((("Mia", False), (_show("Mia"), True)), False)]
    for k in ("cat", "sleeps", "chases"):
        rows.append((((WORD[lang][k], False), (_show(world[WORD[lang][k]]), True)), False))
    alt = f'{w["title"]}. {describe(world, lang)} {w["legend"]}'
    body = world_svg(world, lang, alt=describe(world, lang)) + _table(w["head"], rows) + _note(w["legend"])
    return yours.figure(w["title"], body, alt=alt)


# the tree of *Rex chases Tom* with the meaning of each part: the idea of compositionality
def _node(tree, world, lang):
    """A tree as nodes for drawing: [label, children, is_word, meaning]."""
    if isinstance(tree, str):
        return [tree, [], True, ""]
    kids = [_node(p, world, lang) for p in tree[1:]]
    if tree[0] == "VP":
        meaning = _show(meaning_of_vp(tree, world))
    elif tree[0] == "NP" and len(tree) == 2:
        meaning = _show(tree[1])
    elif tree[0] == "S":
        meaning = repr(truth(tree, world, lang))
    else:
        meaning = ""
    return [tree[0], kids, False, meaning]


def _leaves(node, out: list) -> list:
    if node[2] or not node[1]:
        out.append(node)
    for k in node[1]:
        _leaves(k, out)
    return out


def _depth(node) -> int:
    return 0 if node[2] else 1 + max((_depth(k) for k in node[1]), default=0)


def meaning_tree_svg(tree, world, lang: str, alt: str) -> str:
    """A tree with, under each symbol, its meaning in the world: a name, a list, or True and False at the top."""
    root = _node(tree, world, lang)
    leaves = _leaves(root, [])
    bottom = _depth(root)
    row = 58

    def width(node) -> float:
        return max(40.0, 7.6 * max(len(node[0]), len(node[3])) + 18)

    xs, x = {}, 6.0
    for leaf in leaves:
        xs[id(leaf)] = x + width(leaf) / 2
        x += width(leaf) + 6
    spot = {}

    def place(node, depth):
        y = 18 + (bottom if node[2] else depth) * row
        if node[1]:
            kids = [place(k, depth + 1) for k in node[1]]
            nx = (kids[0][0] + kids[-1][0]) / 2
        else:
            nx = xs[id(node)]
        spot[id(node)] = (nx, y)
        return nx, y

    place(root, 0)
    total = max(x + 2, max(s[0] for s in spot.values()) + 90)
    parts = []

    def draw(node):
        nx, y = spot[id(node)]
        for k in node[1]:
            kx, ky = spot[id(k)]
            parts.append(f'<line x1="{nx:.1f}" y1="{y + (22 if node[3] else 6):.1f}" x2="{kx:.1f}" y2="{ky - 14:.1f}" '
                         f'style="stroke:var(--ea-ink-3);stroke-width:1.2"/>')
            draw(k)
        style = ("font-family:var(--ea-font-ui);font-size:15px;fill:var(--ea-ink);font-weight:600" if node[2] else
                 "font-family:var(--ea-font-code);font-size:13px;fill:var(--ea-ink-2)")
        parts.append(f'<text x="{nx:.1f}" y="{y:.1f}" style="{style};text-anchor:middle">{html.escape(node[0])}</text>')
        if node[3]:
            parts.append(f'<text x="{nx:.1f}" y="{y + 16:.1f}" style="font-family:var(--ea-font-code);font-size:12px;'
                         f'fill:var(--ea-ink);text-anchor:middle">{html.escape(node[3])}</text>')

    draw(root)
    height = 18 + bottom * row + 10
    return (f'<svg viewBox="0 0 {total:.0f} {height:.0f}" role="img" style="width:100%;max-width:{total:.0f}px;'
            f'height:auto"><title>{html.escape(alt)}</title>{"".join(parts)}</svg>')


_PARTS_WORDS = {
    "en": {"title": "The meaning of the whole from the meanings of its parts: Rex chases Tom",
           "legend": "Under each part, its meaning in the world: the VP chases Tom is the list of the animals that chase "
                     "Tom, and the sentence is true when Rex is in that list."},
    "nl": {"title": "De betekenis van het geheel uit die van de delen: Rex achtervolgt Tom",
           "legend": "Onder elk deel staat de betekenis in de wereld: de VP achtervolgt Tom is de lijst van de dieren die "
                     "Tom achtervolgen, en de zin is waar als Rex in die lijst staat."},
}


def parts_picture(lang: str) -> str:
    w = _PARTS_WORDS[lang]
    tree = SENTENCES[lang][2]
    alt = f'{w["title"]}. {w["legend"]}'
    return yours.figure(w["title"], meaning_tree_svg(tree, WORLD[lang], lang, alt) + _note(w["legend"]), alt=alt)


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function within the time limit of a check; _STOPPED when it stops with an error, and the Timeout
    passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _short(value, limit: int = 40) -> str:
    text = _show(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _rows(fn, cases, lang, w):
    """Call the learner's function on each case: (rows, note at the end, all right). A case is (cells, args, want)."""
    rows, end, all_right = [], "", True
    for cells, args, want in cases:
        try:
            got = _call(fn, *args)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(what=cells[0][0])
            break
        right = type(got) is type(want) and got == want
        all_right = all_right and right
        rows.append((list(cells) + [(_short(got), True), (_show(want), True)], not right))
    return rows, end, all_right


_EVERY_WORDS = {
    "en": {"title": "Your every, at work: is each thing of the first list in the second?",
           "head": ["Sentence", "things", "group", "Yours", "Right"],
           "fine": "Your every gives the right answer for each sentence, also for every unicorn sleeps: in this world "
                   "there are no unicorns, and then none of them is awake.",
           "differs": "Where a row is in purple, your every gives another answer.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw every aan het werk: zit elk ding uit de eerste lijst in de tweede?",
           "head": ["Zin", "things", "group", "Jouw antwoord", "Goed"],
           "fine": "Je every geeft bij elke zin het goede antwoord, ook bij elke eenhoorn slaapt: in deze wereld zijn er "
                   "geen eenhoorns, en dan is er ook geen enkele wakker.",
           "differs": "Waar een rij paars is, geeft je every een ander antwoord.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def every_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _EVERY_WORDS[seen.lang]
    world, word = WORLD[lang], WORD[lang]
    cases = []
    for noun in ("cat", "dog", "mouse", "unicorn"):
        things = world.get(word[noun], [])
        group = world[word["sleeps"]]
        sentence = f'{word["every"]} {word[noun]} {word["sleeps"]}'
        cases.append((((sentence, False), (_show(things), True), (_show(group), True)), (things, group),
                      every(things, group)))
    rows, end, all_right = _rows(fn, cases, lang, w)
    after = end or (w["fine"] if all_right else w["differs"])
    alt = f'{w["title"]}. ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {after}"
    return yours.figure(w["title"], _table(w["head"], rows) + _note(after), alt=alt)


_DOERS_WORDS = {
    "en": {"title": "Your doers, at work: who chases whom in the world of the lesson",
           "head": ["Phrase", "Yours", "Right"],
           "phrase": "chases {t}",
           "fine": "Your doers finds for each animal who chases it. The cards with a bold border are the ones that chase Tom: "
                   "the meaning of the VP chases Tom.",
           "differs": "Where a row is in purple, your doers finds other animals. The cards with a bold border are the ones your "
                      "function finds for chases Tom.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw doers aan het werk: wie wie achtervolgt in de wereld van de les",
           "head": ["Woordgroep", "Jouw antwoord", "Goed"],
           "phrase": "achtervolgt {t}",
           "fine": "Je doers vindt bij elk dier wie het achtervolgt. De kaarten met een dikke rand zijn de dieren die Tom achtervolgen: "
                   "de betekenis van de VP achtervolgt Tom.",
           "differs": "Waar een rij paars is, vindt je doers andere dieren. De kaarten met een dikke rand zijn de dieren die je functie "
                      "vindt bij achtervolgt Tom.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def doers_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _DOERS_WORDS[seen.lang]
    world = WORLD[lang]
    pairs = world[WORD[lang]["chases"]]
    cases = [(((w["phrase"].format(t=t), False),), (pairs, t), doers(pairs, t)) for t in THINGS]
    rows, end, all_right = _rows(fn, cases, lang, w)
    try:
        found = _call(fn, pairs, "Tom")
    except Timeout:
        found = _STOPPED
    marked = [t for t in THINGS if isinstance(found, list) and t in found]
    after = end or (w["fine"] if all_right else w["differs"])
    alt = f'{w["title"]}. {describe(world, lang)} ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {after}"
    body = world_svg(world, lang, marked=marked, alt=describe(world, lang)) + _table(w["head"], rows) + _note(after)
    return yours.figure(w["title"], body, alt=alt)


_TRUE_WORDS = {
    "en": {"title": "Your is_true in two worlds: WORLD and NIGHT",
           "head": ["Sentence", "WORLD: yours", "right", "NIGHT: yours", "right"],
           "night": "NIGHT: Tom, Mia and Rex sleep, and Pip chases Tom.",
           "fine": "Your is_true is right for each sentence in both worlds. The sentences stay the same; the world "
                   "decides whether each one is true.",
           "differs": "Where a row is in purple, your is_true gives another answer in at least one of the two worlds.",
           "error": "Your function stops with an error at: {what}.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw is_true in twee werelden: WORLD en NIGHT",
           "head": ["Zin", "WORLD: jouw antwoord", "goed", "NIGHT: jouw antwoord", "goed"],
           "night": "NIGHT: Tom, Mia en Rex slapen, en Pip achtervolgt Tom.",
           "fine": "Je is_true klopt bij elke zin in beide werelden. De zinnen blijven hetzelfde; de wereld beslist of "
                   "elke zin waar is.",
           "differs": "Waar een rij paars is, geeft je is_true in minstens één van de twee werelden een ander antwoord.",
           "error": "Je functie stopt met een fout bij: {what}.",
           "slow": "Je functie doet er te lang over."},
}


def true_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    lang, w = seen.lang, _TRUE_WORDS[seen.lang]
    rows, end, all_right = [], "", True
    for tree in SENTENCES[lang]:
        cells, off = [(text_of(tree), False)], False
        for world in (WORLD[lang], NIGHT[lang]):
            try:
                got = _call(fn, tree, world)
            except Timeout:
                end = w["slow"]
                break
            if got is _STOPPED:
                end = w["error"].format(what=text_of(tree))
                break
            want = is_true(tree, world)
            off = off or not (type(got) is bool and got == want)
            cells += [(_short(got), True), (repr(want), True)]
        if end:
            break
        all_right = all_right and not off
        rows.append((cells, off))
    after = end or (w["fine"] if all_right else w["differs"])
    alt = f'{w["title"]}. {w["night"]} ' + "; ".join(" | ".join(c[0] for c in r[0]) for r in rows) + f". {after}"
    return yours.figure(w["title"], _table(w["head"], rows) + _note(w["night"]) + _note(after), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "Your world, and the 64 worlds of six facts",
           "now": "In your world", "true": "true", "false": "false",
           "grid": "{sentence}: true in {k} of the 64 worlds",
           "how": "Each square is one world. From left to right, the columns switch whether Tom, Mia and Rex sleep; "
                  "from top to bottom, the rows switch whether Pip sleeps, whether Tom chases Pip and whether Rex chases "
                  "Tom. A filled square is a world in which the sentence is true; the ringed square is your world.",
           "on": "on", "off": "off"},
    "nl": {"title": "Jouw wereld, en de 64 werelden van zes feiten",
           "now": "In jouw wereld", "true": "waar", "false": "onwaar",
           "grid": "{sentence}: waar in {k} van de 64 werelden",
           "how": "Elk vakje is één wereld. Van links naar rechts schakelen de kolommen of Tom, Mia en Rex slapen; van "
                  "boven naar beneden schakelen de rijen of Pip slaapt, of Tom Pip achtervolgt en of Rex Tom "
                  "achtervolgt. Een gevuld vakje is een wereld waarin de zin waar is; het vakje met de ring is jouw "
                  "wereld.",
           "on": "aan", "off": "uit"},
}


def demo_picture(on: list, k: int, lang: str) -> str:
    """The world of the switches, the six sentences in it, and the 64 worlds of sentence k."""
    w = _DEMO_WORDS[lang]
    on = [bool(v) for v in on][:len(FACTS)] + [False] * max(0, len(FACTS) - len(on))
    world = world_of(on, lang)
    labels = DEMO_LABELS[lang]
    lines = []
    for i, label in enumerate(labels):
        value = truth(demo_tree(i, lang), world, lang)
        weight = "font-weight:600;" if i == k else ""
        lines.append(f'<div style="{weight}font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink)">'
                     f'{html.escape(label)}: <span style="color:var(--ea-ink-2)">'
                     f'{html.escape(w["true"] if value else w["false"])}</span></div>')
    now = sum(1 << i for i, o in enumerate(on) if o)
    truths = true_in(k, lang)
    cells = []
    for idx, value in enumerate(truths):
        facts = [fact_text(f, lang) for f, o in zip(FACTS, all_worlds()[idx]) if o]
        tip = "; ".join(facts) or "-"
        fill = "var(--ea-ink)" if value else "var(--ea-inset)"
        ring = "outline:3px solid var(--ea-ink-2);outline-offset:2px;" if idx == now else ""
        cells.append(f'<div title="{html.escape(tip)}" style="width:22px;height:22px;border-radius:4px;background:{fill};'
                     f'border:1px solid var(--ea-line);{ring}"></div>')
    grid = ('<div style="display:grid;grid-template-columns:repeat(8,22px);gap:5px;margin:0.4rem 0 0.2rem 4px">'
            + "".join(cells) + "</div>")
    head = w["grid"].format(sentence=labels[k], k=sum(truths))
    alt = (f'{w["title"]}. {describe(world, lang)} ' + "; ".join(
        f'{label}: {w["true"] if truth(demo_tree(i, lang), world, lang) else w["false"]}' for i, label in enumerate(labels))
        + f'. {head}. {w["how"]}')
    body = (world_svg(world, lang, alt=describe(world, lang))
            + f'<p style="{_CAPTION}">{html.escape(w["now"])}</p>' + "".join(lines)
            + f'<p style="{_CAPTION};margin-top:0.8rem">{html.escape(head)}</p>' + grid + _note(w["how"]))
    return yours.figure(w["title"], body, alt=alt)


def demo_facts(lang: str) -> list:
    """The switches of the demo: (label, on in WORLD)."""
    return [(fact_text(f, lang), o) for f, o in zip(FACTS, FACTS_ON)]


def demo_options(lang: str) -> dict:
    return {label: i for i, label in enumerate(DEMO_LABELS[lang])}


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["every", "doers", "is_true", "idiom"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


class DoersArguments(Exception):
    """The course's doers, as the check gives it to is_true, got something other than pairs and a name."""


def _given_doers(pairs, target):
    if not isinstance(target, str) or not isinstance(pairs, (list, tuple)) or not all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in pairs):
        raise DoersArguments("doers takes a list of pairs first and then a name")
    return doers(pairs, target)


def _doers_arguments(e) -> bool:
    return isinstance(e, DoersArguments)


_WORLD_WORDS = sorted({v for lang in WORD for k, v in WORD[lang].items() if k in ("sleeps", "chases")} | set(THINGS))


def _writes_words(got, text) -> bool:
    """The code writes a word of the world itself, as text between quotes."""
    code = "\n".join(line.split("#")[0] for line in (text or "").splitlines())
    return any(q + w + q in code for w in _WORLD_WORDS for q in "\"'")


# exercise 1: every
EVERY_CASES = [
    (["Tom", "Mia"], ["Mia", "Rex"]),
    (["Mia"], ["Mia", "Rex"]),
    ([], ["Mia", "Rex"]),
    (["Tom", "Mia"], ["Tom", "Mia", "Rex"]),
    (["Rex"], []),
    (["Tom", "Mia"], ["Mia"]),
    (["Pip", "Tom", "Mia"], ["Mia", "Pip", "Rex", "Tom"]),
]
_E_START = "def every(things, group):\n    ...\n"
_E = ("def every(things, group):\n"
      "    inside = [thing for thing in things if thing in group]\n"
      "    return len(inside) == len(things)\n")
_E_SOME = _E.replace("len(inside) == len(things)", "len(inside) > 0")
_E_GROUP = _E.replace("len(inside) == len(things)", "len(inside) == len(group)")
_E_COUNT = _E.replace("len(inside) == len(things)", "len(inside)")
_E_EMPTY = _E.replace("len(inside) == len(things)", "len(inside) > 0 and len(inside) == len(things)")
_E_TURNED = ("def every(things, group):\n"
             "    inside = [thing for thing in group if thing in things]\n"
             "    return len(inside) == len(group)\n")

# exercise 2: doers
_CHASES = [("Tom", "Pip"), ("Rex", "Tom")]
DOERS_CASES = [
    (_CHASES, "Tom"),
    (_CHASES, "Pip"),
    (_CHASES, "Mia"),
    ([("Rex", "Tom"), ("Mia", "Tom"), ("Tom", "Pip")], "Tom"),
    ([("Pip", "Tom")], "Tom"),
    ([("Tom", "Pip"), ("Mia", "Rex")], "Rex"),
    ([], "Rex"),
]
_D_START = "def doers(pairs, target):\n    ...\n"
_D = ("def doers(pairs, target):\n"
      "    found = []\n"
      "    for pair in pairs:\n"
      "        if pair[1] == target:\n"
      "            found.append(pair[0])\n"
      "    return found\n")
_D_DONE = _D.replace("found.append(pair[0])", "found.append(pair[1])")
_D_TURNED = _D.replace("if pair[1] == target:", "if pair[0] == target:").replace("found.append(pair[0])",
                                                                                   "found.append(pair[1])")
_D_WHOLE = _D.replace("found.append(pair[0])", "found.append(pair)")
_D_EARLY = _D.replace("\n    return found\n", "\n        return found\n")

# exercise 3: is_true; the names in a message are those of the lesson's cells
TRUE_CASES = ([(Named(f"SENTENCES[{i}]", t), Named("WORLD", WORLD["en"])) for i, t in enumerate(SENTENCES["en"])]
              + [(Named(f"SENTENCES[{i}]", t), Named("NIGHT", NIGHT["en"])) for i, t in enumerate(SENTENCES["en"])])
_T_START = "def is_true(tree, world):\n    ...\n"
_T = ("def is_true(tree, world):\n"
      "    vp = tree[2]\n"
      "    if len(vp) == 2:\n"
      "        group = world[vp[1]]\n"
      "    else:\n"
      "        group = doers(world[vp[1]], vp[2][1])\n"
      "    return tree[1][1] in group\n")
_T_TURNED = ("def is_true(tree, world):\n"
             "    vp = tree[2]\n"
             "    if len(vp) == 2:\n"
             "        return tree[1][1] in world[vp[1]]\n"
             "    return vp[2][1] in doers(world[vp[1]], tree[1][1])\n")
_T_ANY = _T.replace("group = doers(world[vp[1]], vp[2][1])", "group = [pair[0] for pair in world[vp[1]]]")
_T_PAIRS = ("def is_true(tree, world):\n"
            "    vp = tree[2]\n"
            "    group = world[vp[1]]\n"
            "    return tree[1][1] in group\n")
_T_ONEWAY = ("def is_true(tree, world):\n"
             "    vp = tree[2]\n"
             "    group = doers(world[vp[1]], vp[2][1])\n"
             "    return tree[1][1] in group\n")


# code with words of the world in it, also when it gives the right answers: in the other language it would not
_FIXED = Mistake("fixed", matches=_writes_words,
               en="Your function writes words of the sentences itself. Take each word from `tree` and each list from `world`: then the function works for any sentence of this kind, in any world.",
               nl="Je functie schrijft zelf woorden van de zinnen. Haal elk woord uit `tree` en elke lijst uit `world`: dan werkt de functie voor elke zin van deze soort, in elke wereld.")


EXERCISES = {
    "every": FunctionBox(
        name="every",
        picture=every_picture,
        function="every",
        start=_E_START,
        solution=_E,
        visible=EVERY_CASES[:1],
        hidden=EVERY_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("some", wrong=_E_SOME,
                    en="Your function is `some` again: it gives `True` as soon as one thing is in `group`. `every` needs all of them: `{call}` should give {want}.",
                    nl="Je functie is weer `some`: ze geeft `True` zodra één ding in `group` zit. `every` heeft ze allemaal nodig: `{call}` moet {want} geven."),
            Mistake("group", wrong=_E_GROUP,
                    en="`{call}` gives {got}, and it should give {want}: your function checks whether every thing in `group` is in `things`, the other way round. That is *everything that sleeps is a cat*; *every cat sleeps* asks about the cats, whatever else sleeps too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie kijkt of elk ding uit `group` in `things` zit, andersom. Dat is *alles wat slaapt is een kat*; *elke kat slaapt* gaat over de katten, wat er verder ook slaapt."),
            Mistake("count", wrong=_E_COUNT,
                    en="Your function gives back a number, the length of `inside`. Give back `True` or `False`: compare that length with another.",
                    nl="Je functie geeft een getal terug, de lengte van `inside`. Geef `True` of `False` terug: vergelijk die lengte met een andere."),
            Mistake("empty", wrong=_E_EMPTY,
                    en="Right, except for `{call}`: with no things at all, your function gives {got}. In logic, *every unicorn sleeps* is true in a world without unicorns, since there is no unicorn that does not sleep.",
                    nl="Goed, behalve bij `{call}`: zonder dingen geeft je functie {got}. In de logica is *elke eenhoorn slaapt* waar in een wereld zonder eenhoorns, want er is geen eenhoorn die niet slaapt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back `True` when every thing in `things` is also in `group`, and `False` otherwise.",
                      "nl": "Vervang de `...`: geef `True` terug als elk ding uit `things` ook in `group` zit, en anders `False`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is every thing of the first list in the second?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zit elk ding uit de eerste lijst in de tweede?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use both lists it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie allebei de lijsten die ze krijgt?"},
            "correct": {"en": "Your `every` is right in all {n} cases, also with an empty first list. Look at the picture: your function on four sentences of the world.",
                        "nl": "Je `every` klopt in alle {n} gevallen, ook met een lege eerste lijst. Kijk naar de afbeelding: je functie bij vier zinnen over de wereld."},
        },
    ),
    "doers": FunctionBox(
        name="doers",
        picture=doers_picture,
        function="doers",
        start=_D_START,
        solution=_D,
        visible=DOERS_CASES[:1],
        hidden=DOERS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("done", wrong=_D_DONE,
                    en="`{call}` gives {got}, and it should give {want}: your list holds the thing that is chased, once for each pair it finds. Keep the one who does the chasing.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst bevat het ding dat achtervolgd wordt, één keer voor elk paar dat ze vindt. Bewaar wie er achtervolgt."),
            Mistake("turned", wrong=_D_TURNED,
                    en="`{call}` gives {got}, and it should give {want}: your function finds what `target` chases. It should find who chases `target`: the pairs whose second item is `target`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vindt wat `target` achtervolgt. Ze moet vinden wie `target` achtervolgt: de paren waarvan het tweede item `target` is."),
            Mistake("pair", wrong=_D_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your list holds whole pairs. Keep only the doer, so that the list holds animals, as `WORLD` does for a verb without object.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je lijst bevat hele paren. Bewaar alleen wie de handeling uitvoert, zodat de lijst dieren bevat, zoals `WORLD` bij een werkwoord zonder lijdend voorwerp."),
            Mistake("early", wrong=_D_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first pair, because its `return` is inside the loop. Move the `return` to the left, in line with `for`, so that the loop goes over every pair first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie stopt na het eerste paar, omdat de `return` in de loop staat. Zet de `return` even ver ingesprongen als `for`, zodat de loop eerst elk paar langsgaat."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the doers for `target`, in the order of `pairs`.",
                      "nl": "Vervang de `...`: geef de lijst terug van wie de handeling uitvoert op `target`, in de volgorde van `pairs`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. In which pairs is `target` the second item?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. In welke paren is `target` het tweede item?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `pairs` and the `target` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `pairs` en de `target` die ze krijgt?"},
            "correct": {"en": "Your `doers` is right in all {n} cases. Look at the picture: who chases each animal, and in bold the meaning of *chases Tom*.",
                        "nl": "Je `doers` klopt in alle {n} gevallen. Kijk naar de afbeelding: wie elk dier achtervolgt, en dik de betekenis van *achtervolgt Tom*."},
        },
    ),
    "is_true": FunctionBox(
        name="is_true",
        picture=true_picture,
        function="is_true",
        start=_T_START,
        solution=_T,
        visible=TRUE_CASES[:1],
        hidden=TRUE_CASES[1:],
        first=[_DOTS],
        names={"doers": _given_doers},
        form=[_FIXED],
        mistakes=[
            Mistake("one-way", error=_index_error,
                    en="With `{call}`, your function looks for an item that the list does not have. A VP without object has two items, the symbol and the verb; a VP with an object has three. Tell the two apart by `len(vp)`.",
                    nl="Met `{call}` zoekt je functie een item dat de lijst niet heeft. Een VP zonder lijdend voorwerp heeft twee items, het symbool en het werkwoord; een VP met een lijdend voorwerp heeft er drie. Onderscheid de twee met `len(vp)`."),
            Mistake("turned", wrong=_T_TURNED,
                    en="`{call}` gives {got}, and it should give {want}: your function reads the sentence the other way round, with the object as the doer. Find the doers for the object, and look for the subject among them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie leest de zin andersom, met het lijdend voorwerp als wie de handeling uitvoert. Zoek wie de handeling uitvoert op het lijdend voorwerp, en kijk of het onderwerp daarbij is."),
            Mistake("object", wrong=_T_ANY,
                    en="`{call}` gives {got}, and it should give {want}: for a VP with an object, your function looks at everyone who chases anyone. The object counts too: *chases Rex* holds only those who chase Rex.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een VP met een lijdend voorwerp kijkt je functie naar iedereen die iemand achtervolgt. Het lijdend voorwerp telt mee: *achtervolgt Rex* bevat alleen wie Rex achtervolgt."),
            Mistake("pairs", wrong=_T_PAIRS,
                    en="`{call}` gives {got}, and it should give {want}: for a VP with an object, your function finds the subject in no world. Look for it in a list of animals: the list that `doers` gives for the name in the object.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: bij een VP met een lijdend voorwerp vindt je functie het onderwerp in geen enkele wereld. Zoek het in een lijst dieren: wie de handeling uitvoert volgens `doers`, voor de naam in het lijdend voorwerp."),
            Mistake("doers-args", error=_doers_arguments,
                    en="With `{call}`, your function gives `doers` something it cannot use. `doers` takes the list of pairs of the verb first, and then one name, a string.",
                    nl="Met `{call}` geeft je functie `doers` iets wat die functie niet kan gebruiken. `doers` krijgt eerst de lijst paren van het werkwoord, en daarna één naam, een string."),
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function looks up a key that this world does not have. Take each key from the tree: the verb is in the VP. A key you write yourself fits one language only, and the check also gives the sentences in English.",
                    nl="Met `{call}` zoekt je functie een key op die deze wereld niet heeft. Haal elke key uit de boom: het werkwoord staat in de VP. Een key die je zelf schrijft, past maar bij één taal, en de check geeft de zinnen ook in het Engels."),
            _FIXED,
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: find the list of animals the VP stands for, and give back whether the subject is in it.",
                      "nl": "Vervang de `...`: zoek de lijst dieren waar de VP voor staat, en geef terug of het onderwerp erin zit."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What list does the VP stand for in that world, and is the subject in it?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor welke lijst staat de VP in die wereld, en zit het onderwerp erin?"},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function work for a VP with an object, and in both worlds?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor een VP met een lijdend voorwerp, en in beide werelden?"},
            "correct": {"en": "Your `is_true` is right in all {n} cases: five sentences in two worlds. Look at the picture: the same sentences, and some of them true in one world and false in the other.",
                        "nl": "Je `is_true` klopt in alle {n} gevallen: vijf zinnen in twee werelden. Kijk naar de afbeelding: dezelfde zinnen, en een paar ervan waar in de ene wereld en onwaar in de andere."},
        },
    ),
}

CASES = [
    ("every", _E_START, "empty", None),
    ("every", _E, "correct", None),
    ("every", "def every(things, group):\n    for thing in things:\n        if thing not in group:\n            return False\n    return True\n", "correct", None),
    ("every", "def every(things, group):\n    return [t for t in things if t in group] == things\n", "correct", None),
    ("every", _E_SOME, "wrong", "some"),
    ("every", _E_GROUP, "wrong", "group"),
    ("every", _E_COUNT, "wrong", "count"),
    ("every", _E_EMPTY, "wrong", "empty"),
    ("every", _E_TURNED, "wrong", "group"),
    ("every", "def every(things, group):\n    return False\n", "almost", None),
    ("doers", _D_START, "empty", None),
    ("doers", _D, "correct", None),
    ("doers", "def doers(pairs, target):\n    return [doer for doer, done in pairs if done == target]\n", "correct", None),
    ("doers", _D_DONE, "wrong", "done"),
    ("doers", _D_TURNED, "wrong", "turned"),
    ("doers", _D_WHOLE, "wrong", "pair"),
    ("doers", _D_EARLY, "wrong", "early"),
    ("doers", "def doers(pairs, target):\n    return [\"Rex\"]\n", "almost", None),
    ("is_true", _T_START, "empty", None),
    ("is_true", _T, "correct", None),
    ("is_true", "def is_true(tree, world):\n    subject, vp = tree[1][1], tree[2]\n    if len(vp) == 3:\n        return subject in [p[0] for p in world[vp[1]] if p[1] == vp[2][1]]\n    return subject in world[vp[1]]\n", "correct", None),
    ("is_true", _T_TURNED, "wrong", "turned"),
    ("is_true", _T_ANY, "wrong", "object"),
    ("is_true", _T_PAIRS, "wrong", "pairs"),
    ("is_true", _T_ONEWAY, "error", "one-way"),
    ("is_true", "def is_true(tree, world):\n    if tree[1][1] == \"Mia\":\n        return True\n    return False\n", "wrong", "fixed"),
    ("is_true", "def is_true(tree, world):\n    return True\n", "almost", None),
    ("is_true", _T.replace("vp[2][1])", "vp[2][0])"), "wrong", "pairs"),
    ("is_true", _T.replace("doers(world[vp[1]], vp[2][1])", "doers(vp[2][1], world[vp[1]])"), "error", "doers-args"),
    ("is_true", _T.replace("vp[2][1])", "vp[2])"), "error", "doers-args"),
    ("is_true", _T.replace("group = world[vp[1]]", "group = world[\"slaapt\"]"), "error", "key"),
    ("is_true", _T.replace("if len(vp) == 2:", "if vp[1] == \"sleeps\":"), "wrong", "fixed"),
]
