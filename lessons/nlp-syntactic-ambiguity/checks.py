# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-syntactic-ambiguity: turn the tree of a verb phrase into the other tree of the same words,
count the bracketings of a phrase of n words, sort ambiguous sentences by their kind, and find the phrase whose verb
form leaves one reading. The trees, the sentences and the phrases are the course's own; the trees are written as lists
in the course's own short notation, without the tags above the words."""
import contextlib
import copy
import html
import io
import math

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the course's trees ----------------------------------------------------------------------------------------------

# A tree is a list: its symbol first, then its parts; a word is a string. The sentence is the one of Exercise 2 of
# Phrase structure and endless sentences. In VP_VERB the PP is part of the verb phrase; in VP_NOUN it is part of the
# noun phrase, through the rule NP -> NP PP that this lesson adds.
SUBJECTS = {"en": ["NP", "a", "long", "train"], "nl": ["NP", "een", "lange", "trein"]}
VP_VERB = {
    "en": ["VP", "passed", ["NP", "the", "bridge"], ["PP", "near", "Chicago"]],
    "nl": ["VP", "passeerde", ["NP", "de", "brug"], ["PP", "bij", "Breda"]],
}
VP_NOUN = {
    "en": ["VP", "passed", ["NP", ["NP", "the", "bridge"], ["PP", "near", "Chicago"]]],
    "nl": ["VP", "passeerde", ["NP", ["NP", "de", "brug"], ["PP", "bij", "Breda"]]],
}
# the English words of the two trees, as the Dutch lesson writes them: for what a check message shows
_TO_NL = {"passed": "passeerde", "the": "de", "bridge": "brug", "near": "bij", "Chicago": "Breda"}

# the phrases of the demo: six nouns in English; in Dutch, the parts of one long compound
PHRASES = {
    "en": ["city", "bus", "driver", "union", "pay", "talks"],
    "nl": ["stads", "bus", "chauffeurs", "bond", "loon", "overleg"],
}


def other_tree(vp):
    """The course's solution of Exercise 1."""
    if len(vp) == 4:
        return ["VP", vp[1], ["NP", vp[2], vp[3]]]
    phrase = vp[2]
    return ["VP", vp[1], phrase[1], phrase[2]]


def bracketings(n):
    """The course's solution of Exercise 2: the number of ways to bracket n words, two groups at a time."""
    counts = [0, 1]
    for size in range(2, n + 1):
        total = 0
        for left in range(1, size):
            total = total + counts[left] * counts[size - left]
        counts.append(total)
    return counts[n]


def all_bracketings(words: list) -> list:
    """Every bracketing of the words, as text with square brackets, in a fixed order: the top split from left to
    right."""
    if len(words) == 1:
        return [words[0]]
    out = []
    for left in range(1, len(words)):
        for a in all_bracketings(words[:left]):
            for b in all_bracketings(words[left:]):
                out.append(f"[{a} {b}]")
    return out


def words_of(tree) -> list:
    """The words of a tree, from left to right."""
    if isinstance(tree, str):
        return [tree]
    out = []
    for part in tree[1:]:
        out += words_of(part)
    return out


def is_tree(value, depth: int = 0) -> bool:
    """Whether a value is a tree in the notation of the lesson: a list with a symbol first and at least one part, each a
    word or a tree, at most 8 levels deep and with at most 40 words."""
    if depth > 8 or not isinstance(value, list) or len(value) < 2 or not isinstance(value[0], str):
        return False
    if not all(isinstance(p, str) or is_tree(p, depth + 1) for p in value[1:]):
        return False
    return depth > 0 or len(words_of(value)) <= 40


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = ";color:var(--ea-loss-text)"
_CAPTION = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin:0 0 0.2rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _num(n: int, lang: str) -> str:
    """A whole number with a thousands separator: a comma in English, a dot in Dutch."""
    text = f"{n:,}"
    return text.replace(",", ".") if lang == "nl" else text


def _node(tree):
    """A tree as nodes for drawing: [label, children, is_word]."""
    if isinstance(tree, str):
        return [tree, [], True]
    return [tree[0], [_node(p) for p in tree[1:]], False]


def _leaves(node, out: list) -> list:
    if node[2] or not node[1]:
        out.append(node)
    for k in node[1]:
        _leaves(k, out)
    return out


def _depth(node) -> int:
    return 0 if node[2] else 1 + max((_depth(k) for k in node[1]), default=0)


def tree_svg(tree, alt: str) -> str:
    """A tree: the symbols in grey code letters, the words in ink along the bottom, a line from each symbol to its
    parts. The line into a PP is bold, so the eye finds where the PP hangs."""
    root = _node(tree)
    leaves = _leaves(root, [])
    bottom = _depth(root)
    row = 44

    def width(node) -> float:
        return max(30.0, (8.6 if node[2] else 8.0) * len(node[0]) + 14)

    xs, x = {}, 6.0
    for leaf in leaves:
        xs[id(leaf)] = x + width(leaf) / 2
        x += width(leaf) + 4
    total = x + 2
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
    parts = []

    def draw(node):
        nx, y = spot[id(node)]
        for k in node[1]:
            kx, ky = spot[id(k)]
            bold = not k[2] and k[0] == "PP"
            style = "stroke:var(--ea-ink);stroke-width:2.6" if bold else "stroke:var(--ea-ink-3);stroke-width:1.2"
            parts.append(f'<line x1="{nx:.1f}" y1="{y + 6:.1f}" x2="{kx:.1f}" y2="{ky - 14:.1f}" style="{style}"/>')
            draw(k)
        style = ("font-family:var(--ea-font-ui);font-size:15px;fill:var(--ea-ink);font-weight:600" if node[2] else
                 "font-family:var(--ea-font-code);font-size:13px;fill:var(--ea-ink-2)")
        parts.append(f'<text x="{nx:.1f}" y="{y:.1f}" style="{style};text-anchor:middle">{html.escape(node[0])}</text>')

    draw(root)
    height = 18 + bottom * row + 10
    return (f'<svg viewBox="0 0 {total:.0f} {height:.0f}" role="img" style="width:100%;max-width:{total:.0f}px;'
            f'height:auto"><title>{html.escape(alt)}</title>{"".join(parts)}</svg>')


def _bracketed(tree) -> str:
    if isinstance(tree, str):
        return tree
    return f'[{tree[0]} {" ".join(_bracketed(p) for p in tree[1:])}]'


def _pair(left: tuple, right: tuple) -> str:
    """Two trees side by side, each with its caption: (caption, tree)."""
    cells = []
    for caption, tree in (left, right):
        alt = f"{caption}: {_bracketed(tree)}"
        cells.append(f'<div style="flex:1 1 360px;min-width:0"><p style="{_CAPTION}">{html.escape(caption)}</p>'
                     f'{tree_svg(tree, alt)}</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem 1.6rem;align-items:flex-end">{"".join(cells)}</div>'


_IDEA_WORDS = {
    "en": {"title": "Two trees for one sentence: a long train passed the bridge near Chicago",
           "verb": "The PP is part of the verb phrase: near Chicago says where the train passed",
           "noun": "The PP is part of the noun phrase: near Chicago says which bridge",
           "legend": "The words are the same, in the same order. Only the place of the PP differs: follow the bold line."},
    "nl": {"title": "Twee bomen voor één zin: een lange trein passeerde de brug bij Breda",
           "verb": "De PP hoort bij de werkwoordgroep: bij Breda zegt waar de trein passeerde",
           "noun": "De PP hoort bij de naamwoordgroep: bij Breda zegt welke brug",
           "legend": "De woorden zijn hetzelfde, in dezelfde volgorde. Alleen de plek van de PP verschilt: volg de dikke lijn."},
}


def idea_picture(lang: str) -> str:
    """The two trees of the sentence of The idea."""
    w = _IDEA_WORDS[lang]
    verb = ["S", SUBJECTS[lang], VP_VERB[lang]]
    noun = ["S", SUBJECTS[lang], VP_NOUN[lang]]
    alt = f'{w["title"]}. {w["verb"]}: {_bracketed(verb)}. {w["noun"]}: {_bracketed(noun)}. {w["legend"]}'
    body = _pair((w["verb"], verb), (w["noun"], noun)) + _note(w["legend"])
    return yours.figure(w["title"], body, alt=alt)


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


def _short(value, limit: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


_OTHER_WORDS = {
    "en": {"title": "Your other_tree, drawn: the tree it got, and the tree it gives back",
           "in": "VP_VERB, the tree it got: the PP is part of the verb phrase",
           "out": "What your other_tree gives back",
           "fine": "Your other_tree put the PP into the noun phrase: the same words, in the same order, another tree. "
                   "On its own tree it gives back VP_VERB.",
           "same": "Your other_tree gives back the tree it got: the PP is still part of the verb phrase.",
           "order": "The words of your tree are {got}, and they should stay {want}: a tree of the same sentence keeps "
                    "every word, in its order.",
           "other": "Your tree has the words in their order, but it is not the tree with the PP in the noun phrase.",
           "back": "Your other_tree gives the right tree for VP_VERB, but on that tree it does not give back VP_VERB.",
           "not_tree": "Your function gives back {got}: that is not a tree in the notation of this lesson.",
           "error": "Your function stops with an error on VP_VERB.",
           "slow": "Your function takes too long."},
    "nl": {"title": "Jouw other_tree, getekend: de boom die ze kreeg, en de boom die ze teruggeeft",
           "in": "VP_VERB, de boom die ze kreeg: de PP hoort bij de werkwoordgroep",
           "out": "Wat je other_tree teruggeeft",
           "fine": "Je other_tree zette de PP in de naamwoordgroep: dezelfde woorden, in dezelfde volgorde, een andere "
                   "boom. Op die boom geeft ze VP_VERB terug.",
           "same": "Je other_tree geeft de boom terug die ze kreeg: de PP hoort nog bij de werkwoordgroep.",
           "order": "De woorden van je boom zijn {got}, en dat moeten {want} blijven: een boom van dezelfde zin houdt "
                    "elk woord, in de volgorde van de zin.",
           "other": "Je boom heeft de woorden in de goede volgorde, maar het is niet de boom met de PP in de naamwoordgroep.",
           "back": "Je other_tree geeft de goede boom voor VP_VERB, maar op die boom geeft ze VP_VERB niet terug.",
           "not_tree": "Je functie geeft {got} terug: dat is geen boom in de schrijfwijze van deze les.",
           "error": "Je functie stopt met een fout bij VP_VERB.",
           "slow": "Je functie doet er te lang over."},
}


def other_picture(seen) -> str:
    """VP_VERB of the learner's language and what the learner's other_tree makes of it, side by side."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _OTHER_WORDS[seen.lang]
    vp = VP_VERB[seen.lang]
    try:
        got = _call(fn, vp)
        back = _call(fn, got) if is_tree(got) else None
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"]), alt=f'{w["title"]}. {w["slow"]}')
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=f'{w["title"]}. {w["error"]}')
    if not is_tree(got):
        text = w["not_tree"].format(got=_short(got))
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {text}')
    if got == vp:
        after = w["same"]
    elif words_of(got) != words_of(vp):
        after = w["order"].format(got=" ".join(words_of(got)), want=" ".join(words_of(vp)))
    elif got != VP_NOUN[seen.lang]:
        after = w["other"]
    elif back != vp:
        after = w["back"]
    else:
        after = w["fine"]
    alt = f'{w["title"]}. {w["in"]}: {_bracketed(vp)}. {w["out"]}: {_bracketed(got)}. {after}'
    return yours.figure(w["title"], _pair((w["in"], vp), (w["out"], got)) + _note(after), alt=alt)


_COUNT_WORDS = {
    "en": {"title": "Your bracketings, for 1 to 12 words",
           "head": ["Words", "Your bracketings", "Right", ""],
           "fine": "All twelve are right. The count grows faster and faster: 10 words already have 4,862 bracketings.",
           "differs": "Your function gives another number where the row is purple.",
           "four": "The 5 bracketings of four words:",
           "error": "Your function stops with an error at n = {n}.",
           "slow": "Your function takes too long at n = {n}."},
    "nl": {"title": "Jouw bracketings, voor 1 tot en met 12 woorden",
           "head": ["Woorden", "Jouw bracketings", "Goed", ""],
           "fine": "Alle twaalf kloppen. Het aantal groeit steeds sneller: 10 woorden hebben al 4.862 bracketings.",
           "differs": "Je functie geeft een ander getal waar de rij paars is.",
           "four": "De 5 bracketings van vier delen:",
           "error": "Je functie stopt met een fout bij n = {n}.",
           "slow": "Je functie doet er te lang over bij n = {n}."},
}


def _bar(n: int, top: int) -> str:
    width = 0.3 + 9.0 * math.log10(max(n, 1)) / math.log10(top)
    return (f'<span style="display:inline-block;width:{width:.2f}rem;height:0.6rem;border-radius:2px;'
            f'background:var(--ea-ink-3)"></span>')


def counts_picture(seen) -> str:
    """The learner's counts for 1 to 12 words next to the right ones, and the five bracketings of four words."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _COUNT_WORDS[seen.lang]
    lang = seen.lang
    rows, end, all_right = [], "", True
    top = bracketings(12)
    for n in range(1, 13):
        try:
            got = _call(fn, n)
        except Timeout:
            end = w["slow"].format(n=n)
            break
        if got is _STOPPED:
            end = w["error"].format(n=n)
            break
        want = bracketings(n)
        number = isinstance(got, (int, float)) and not isinstance(got, bool)
        right = number and got == want
        all_right = all_right and right
        whole = number and (isinstance(got, int) or got.is_integer())
        shown = _num(int(got), lang) if whole else _short(got, 24)
        style = _CELL + ("" if right else _OFF)
        rows.append(f'<tr><td style="{style}">{n}</td><td style="{style}">{html.escape(shown)}</td>'
                    f'<td style="{_CELL}">{_num(want, lang)}</td><td style="{_CELL}">{_bar(want, top)}</td></tr>')
    head = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in w["head"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
             f'<tr>{head}</tr>{"".join(rows)}</table></div>')
    after = end or (w["fine"] if all_right else w["differs"])
    four = all_bracketings(PHRASES[lang][:4])
    listed = (f'<div style="{_UI}">{html.escape(w["four"])}</div><div style="font-family:var(--ea-font-code);'
              f'font-size:0.8rem;color:var(--ea-ink-2)">' + "<br>".join(html.escape(b) for b in four) + "</div>")
    alt = f'{w["title"]}. {after} {w["four"]} ' + "; ".join(four)
    return yours.figure(w["title"], table + _note(after) + listed, alt=alt)


_DEMO_WORDS = {
    "en": {"title": "Every bracketing of {phrase}",
           "count": "{n} words: {k} bracketings, listed by their top split, where the outermost brackets divide the phrase, from left to right."},
    "nl": {"title": "Elke bracketing van {phrase}",
           "count": "{n} delen: {k} bracketings, op volgorde van de bovenste splitsing, waar de buitenste haakjes de woordgroep in tweeën delen, van links naar rechts."},
}


def demo_picture(n: int, lang: str) -> str:
    """Every bracketing of the first n words of the demo phrase."""
    w = _DEMO_WORDS[lang]
    words = PHRASES[lang][:n]
    phrase = " ".join(words) if lang == "en" else "".join(words) + " (" + " | ".join(words) + ")"
    found = all_bracketings(words)
    title = w["title"].format(phrase=phrase)
    count = w["count"].format(n=n, k=_num(len(found), lang))
    grid = ('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(15rem,1fr));gap:0.1rem 1.2rem;'
            'font-family:var(--ea-font-code);font-size:0.78rem;color:var(--ea-ink-2);margin-top:0.3rem">'
            + "".join(f"<div>{html.escape(b)}</div>" for b in found) + "</div>")
    alt = f"{title}. {count} " + "; ".join(found)
    return yours.figure(title, _note(count) + grid, alt=alt)


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["other_tree", "bracketings", "kinds", "agree"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


VP_VERB_N = Named("VP_VERB", VP_VERB["en"])
VP_NOUN_N = Named("VP_NOUN", VP_NOUN["en"])


def _code(value) -> str:
    if isinstance(value, str):
        return '"' + value + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_code(v) for v in value) + "]"
    return repr(value)


def _in_dutch(value):
    if isinstance(value, str):
        return _TO_NL.get(value, value)
    if isinstance(value, list):
        return [_in_dutch(v) for v in value]
    return value


def show_tree(value) -> dict:
    """How {got} and {want} show a tree: the words of the lesson's sentence in the learner's language."""
    return {"en": _code(value), "nl": _code(_in_dutch(value))}


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


_SENTENCE_WORDS = [w for pair in _TO_NL.items() for w in pair if w not in ("the", "de")]


def _writes_words(got, text) -> bool:
    """The code writes a word of the lesson's sentence itself, as text between quotes."""
    code = "\n".join(line.split("#")[0] for line in (text or "").splitlines())
    return any(q + w + q in code for w in _SENTENCE_WORDS for q in "\"'")


def _has_tuple(value, depth: int = 0) -> bool:
    if isinstance(value, tuple):
        return True
    return depth < 8 and isinstance(value, list) and any(_has_tuple(v, depth + 1) for v in value)


# exercise 1: other_tree; the cases with tags in the place of the words are the same in both languages
_TAGS_VERB = ["VP", "VERB", ["NP", "DET", "ADJ", "NOUN"], ["PP", "ADP", "DET", "NOUN"]]
_NAMES_VERB = ["VP", "VERB", ["NP", "PROPN"], ["PP", "ADP", "PROPN"]]
OTHER_CASES = [
    (VP_VERB_N,),
    (VP_NOUN_N,),
    (_TAGS_VERB,),
    (other_tree(_TAGS_VERB),),
    (_NAMES_VERB,),
    (other_tree(_NAMES_VERB),),
]
_O_START = "def other_tree(vp):\n    ...\n"
_O = ("def other_tree(vp):\n"
      "    if len(vp) == 4:\n"
      "        return [\"VP\", vp[1], [\"NP\", vp[2], vp[3]]]\n"
      "    phrase = vp[2]\n"
      "    return [\"VP\", vp[1], phrase[1], phrase[2]]\n")
_O_SAME = "def other_tree(vp):\n    return vp\n"
_O_SWAP = _O.replace('["NP", vp[2], vp[3]]', '["NP", vp[3], vp[2]]')
_O_NOLABEL = _O.replace('["NP", vp[2], vp[3]]', '[vp[2], vp[3]]')
_O_STAYS = _O.replace('    return ["VP", vp[1], phrase[1], phrase[2]]\n', '    return ["VP", vp[1], phrase]\n')
_O_ONEWAY = "def other_tree(vp):\n    return [\"VP\", vp[1], [\"NP\", vp[2], vp[3]]]\n"

# exercise 2: bracketings
COUNT_CASES = [(4,), (1,), (2,), (3,), (5,), (7,), (10,), (12,)]
_B_START = "def bracketings(n):\n    ...\n"
_B = ("def bracketings(n):\n"
      "    counts = [0, 1]\n"
      "    for size in range(2, n + 1):\n"
      "        total = 0\n"
      "        for left in range(1, size):\n"
      "            total = total + counts[left] * counts[size - left]\n"
      "        counts.append(total)\n"
      "    return counts[n]\n")
_B_PLUS = _B.replace("counts[left] * counts[size - left]", "counts[left] + counts[size - left]")
_B_LEFT = _B.replace("counts[left] * counts[size - left]", "counts[left] * counts[left]")
_B_TOTAL = _B.replace("    counts = [0, 1]\n", "    counts = [0, 1]\n    total = 0\n").replace("        total = 0\n", "")
_B_LIST = _B.replace("    return counts[n]\n", "    return counts\n")
_B_ONE = _B.replace("    counts = [0, 1]\n", "    counts = [0, 2]\n")


EXERCISES = {
    "other_tree": FunctionBox(
        name="other_tree",
        picture=other_picture,
        function="other_tree",
        start=_O_START,
        solution=_O,
        visible=OTHER_CASES[:1],
        hidden=OTHER_CASES[1:],
        first=[_DOTS],
        show=show_tree,
        mistakes=[
            Mistake("one-way", error=_index_error,
                    en="With `{call}`, your function looks for an item that the list does not have. A VP has four items when the PP is part of the verb phrase, and three when it sits in the noun phrase. Check which part of your function handles which length, and which items each list has.",
                    nl="Met `{call}` zoekt je functie een item dat de lijst niet heeft. Een VP heeft vier items als de PP bij de werkwoordgroep hoort, en drie als die in de naamwoordgroep zit. Kijk welk deel van je functie welke lengte afhandelt, en welke items elke lijst heeft."),
            Mistake("same", wrong=_O_SAME,
                    en="Your function gives back the tree it got. The words stay, but the PP should move: into the noun phrase when the VP has four items, and out of it when the VP has three.",
                    nl="Je functie geeft de boom terug die ze kreeg. De woorden blijven, maar de PP moet verhuizen: de naamwoordgroep in als de VP vier items heeft, en eruit als de VP er drie heeft."),
            Mistake("swap", wrong=_O_SWAP,
                    en="In your new noun phrase, the PP comes before the NP, so the words no longer read in the order of the sentence. Put the NP first, then the PP.",
                    nl="In je nieuwe naamwoordgroep staat de PP vóór de NP, dus de woorden staan niet meer in de volgorde van de zin. Zet eerst de NP, dan de PP."),
            Mistake("no-label", wrong=_O_NOLABEL,
                    en="Your new noun phrase has no symbol: a list for a phrase starts with its symbol, as `[\"PP\", ...]` does.",
                    nl="Je nieuwe naamwoordgroep heeft geen symbool: een lijst voor een woordgroep begint met het symbool, zoals `[\"PP\", ...]`."),
            Mistake("stays", wrong=_O_STAYS,
                    en="For a VP of four items your function is right. For a VP of three, it gives back the same tree: there the PP should come out of the noun phrase, so that it is part of the verb phrase again.",
                    nl="Voor een VP van vier items klopt je functie. Voor een VP van drie geeft ze dezelfde boom terug: daar moet de PP uit de naamwoordgroep komen, zodat die weer bij de werkwoordgroep hoort."),
            Mistake("tuple", matches=lambda got, text: _has_tuple(got),
                    en="Your function gives back a tuple where a list belongs: a tree in this lesson is a list, with square brackets.",
                    nl="Je functie geeft een tuple terug waar een lijst hoort: een boom is in deze les een lijst, met vierkante haken."),
            Mistake("fixed", matches=_writes_words,
                    en="Your function writes words of the sentence itself. The check also gives it trees with other words: build the new tree from the items of `vp`.",
                    nl="Je functie schrijft zelf woorden van de zin. De check geeft je functie ook bomen met andere woorden: bouw de nieuwe boom uit de items van `vp`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the other tree of the same words, for a VP of four items and for a VP of three.",
                      "nl": "Vervang de `...`: geef de andere boom van dezelfde woorden terug, voor een VP van vier items en voor een VP van drie."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. The words stay in their order; only the place of the PP changes.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. De woorden blijven in hun volgorde; alleen de plek van de PP verandert."},
            "almost": {"en": "Right for the first case. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `vp` it gets, also when it has three items?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `vp` die ze krijgt, ook als die drie items heeft?"},
            "correct": {"en": "Your `other_tree` is right in all {n} cases: it moves the PP into the noun phrase and back, and the words stay where they are. Look at the picture: both trees, drawn from your function.",
                        "nl": "Je `other_tree` klopt in alle {n} gevallen: ze zet de PP in de naamwoordgroep en terug, en de woorden blijven waar ze staan. Kijk naar de afbeelding: beide bomen, getekend met je functie."},
        },
    ),
    "bracketings": FunctionBox(
        name="bracketings",
        picture=counts_picture,
        function="bracketings",
        start=_B_START,
        solution=_B,
        visible=COUNT_CASES[:1],
        hidden=COUNT_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function looks past the end of `counts`. The count for `size` words is only in the list once you append it: inside the loops, use counts of fewer words, and let `size` go up to and including `n`.",
                    nl="Met `{call}` kijkt je functie voorbij het eind van `counts`. Het aantal voor `size` woorden staat pas in de lijst als je het toevoegt: gebruik binnen de loops aantallen van minder woorden, en laat `size` lopen tot en met `n`."),
            Mistake("plus", wrong=_B_PLUS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the counts of the two parts. Each bracketing of the left part goes with each bracketing of the right part, so a split gives their product.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie telt de aantallen van de twee delen op. Elke bracketing van het linkerdeel gaat samen met elke bracketing van het rechterdeel, dus een splitsing geeft hun product."),
            Mistake("left-twice", wrong=_B_LEFT,
                    en="`{call}` gives {got}, and it should give {want}: your function multiplies the count of the left part by itself. The right part has the other words of the phrase.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vermenigvuldigt het aantal van het linkerdeel met zichzelf. Het rechterdeel heeft de andere woorden van de woordgroep."),
            Mistake("total", wrong=_B_TOTAL,
                    en="`{call}` gives {got}, and it should give {want}: `total` runs on from one size to the next. Start it at 0 again for each size, inside the outer loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `total` loopt door van de ene grootte naar de volgende. Zet `total` bij elke grootte weer op 0, binnen de buitenste loop."),
            Mistake("list", wrong=_B_LIST,
                    en="Your function gives back the whole list `counts`. Give back only the count for `n` words.",
                    nl="Je functie geeft de hele lijst `counts` terug. Geef alleen het aantal voor `n` woorden terug."),
            Mistake("one", wrong=_B_ONE,
                    en="`{call}` gives {got}, and it should give {want}: one word has one bracketing, the word itself. Start the list with 0 for no words and 1 for one word.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: één woord heeft één bracketing, het woord zelf. Begin de lijst met 0 voor geen woorden en 1 voor één woord."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: fill the list `counts` from small to large, and give back the count for `n` words.",
                      "nl": "Vervang de `...`: vul de lijst `counts` van klein naar groot, en geef het aantal voor `n` woorden terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Three words have 2 bracketings, four words 5.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Drie woorden hebben 2 bracketings, vier woorden 5."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `n` it gets?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `n` die ze krijgt?"},
            "correct": {"en": "Your `bracketings` is right in all {n} cases, from 1 word up to 12. Look at the picture: how fast the count grows.",
                        "nl": "Je `bracketings` klopt in alle {n} gevallen, van 1 woord tot en met 12. Kijk naar de afbeelding: hoe snel het aantal groeit."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "attach": {"en": "Where a phrase attaches", "nl": "Waar een woordgroep aanhaakt"},
            "and": {"en": "What *and* connects", "nl": "Wat *en* verbindt"},
            "role": {"en": "A word with two roles", "nl": "Een woord met twee rollen"},
        },
        items={
            "lens": {"en": "**We watched the bird with the binoculars.**", "nl": "**We bekeken de vogel met de verrekijker.**"},
            "lights": {"en": "**The officer fined the cyclist without lights.**", "nl": "**De agent bekeurde de fietser zonder licht.**"},
            "fish": {"en": "**The menu offers fresh fish and fries.**", "nl": "**Op de kaart staan verse vis en friet.**"},
            "club": {"en": "**The club wants young players and coaches.**", "nl": "**De club zoekt jonge spelers en trainers.**"},
            "union": {"en": "**Union demands rise.**", "nl": "**Verdachte vlucht naar Spanje.**"},
            "washing": {"en": "**Washing machines can be noisy.**", "nl": "**Fietsen zonder bel verboden.**"},
        },
        solution={"lens": "attach", "lights": "attach", "fish": "and", "club": "and", "union": "role", "washing": "role"},
        nudge={
            "lens": {"en": "Who has the binoculars: the bird, or the people watching it?",
                     "nl": "Wie heeft de verrekijker: de vogel, of wie ernaar kijkt?"},
            "lights": {"en": "Who has no lights: the cyclist, or the officer?",
                       "nl": "Wie heeft er geen licht: de fietser, of de agent?"},
            "fish": {"en": "Is only the fish fresh, or the fries too?",
                     "nl": "Is alleen de vis vers, of de friet ook?"},
            "club": {"en": "Are only the players young, or the coaches too?",
                     "nl": "Zijn alleen de spelers jong, of de trainers ook?"},
            "union": {"en": "Is *demands* what the union does, or the thing that rises?",
                      "nl": "Is *vlucht* iets wat een verdachte doet, of een reis?"},
            "washing": {"en": "Is *washing* something you do to machines, or a kind of machine?",
                        "nl": "Is *fietsen* iets wat je doet, of zijn het de fietsen zelf?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the sentences again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zinnen opnieuw uit."},
            "empty": {"en": "Put each bold sentence under the kind of ambiguity it has.",
                      "nl": "Zet elke vetgedrukte zin bij de soort ambiguïteit die ze heeft."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs under another kind.",
                      "nl": "{item} hoort bij een andere soort."},
            "correct": {"en": "All {n} are right. A phrase that can attach to two words, an *and* whose parts can be cut two ways, and a word that can be two parts of speech: each gives the same words two trees.",
                        "nl": "Alle {n} kloppen. Een woordgroep die bij twee woorden kan horen, een *en* waarvan de delen op twee manieren te knippen zijn, en een woord dat twee woordsoorten kan zijn: elk geeft dezelfde woorden twee bomen."},
        },
    ),
}

CASES = [
    ("other_tree", _O_START, "empty", None),
    ("other_tree", _O, "correct", None),
    ("other_tree", "def other_tree(vp):\n    if len(vp) == 3:\n        noun = vp[2]\n        return [vp[0], vp[1], noun[1], noun[2]]\n    return [vp[0], vp[1], [\"NP\", vp[2], vp[3]]]\n", "correct", None),
    ("other_tree", _O_SAME, "wrong", "same"),
    ("other_tree", _O_SWAP, "wrong", "swap"),
    ("other_tree", _O_NOLABEL, "wrong", "no-label"),
    ("other_tree", _O_STAYS, "wrong", "stays"),
    ("other_tree", _O_ONEWAY, "error", "one-way"),
    ("other_tree", "def other_tree(vp):\n    return [\"VP\", \"passed\", [\"NP\", [\"NP\", \"the\", \"bridge\"], [\"PP\", \"near\", \"Chicago\"]]]\n", "wrong", "fixed"),
    ("other_tree", "def other_tree(vp):\n    if len(vp) == 4:\n        return [\"VP\", \"passed\", [\"NP\", [\"NP\", \"the\", \"bridge\"], [\"PP\", \"near\", \"Chicago\"]]]\n    return [\"VP\", \"passed\", [\"NP\", \"the\", \"bridge\"], [\"PP\", \"near\", \"Chicago\"]]\n", "wrong", "fixed"),
    ("other_tree", "def other_tree(vp):\n    if len(vp) == 4:\n        return (\"VP\", vp[1], (\"NP\", vp[2], vp[3]))\n    phrase = vp[2]\n    return (\"VP\", vp[1], phrase[1], phrase[2])\n", "wrong", "tuple"),
    ("other_tree", "def other_tree(vp):\n    if len(vp) == 4:\n        return [\"VP\", vp[1], (\"NP\", vp[2], vp[3])]\n    phrase = vp[2]\n    return [\"VP\", vp[1], phrase[1], phrase[2]]\n", "wrong", "tuple"),
    ("bracketings", _B.replace("    return counts[n]\n", "    return counts[n] * 1.0\n"), "correct", None),
    ("bracketings", _B_START, "empty", None),
    ("bracketings", _B, "correct", None),
    ("bracketings", "def bracketings(n):\n    counts = [1]\n    for k in range(1, n):\n        total = 0\n        for i in range(k):\n            total = total + counts[i] * counts[k - 1 - i]\n        counts.append(total)\n    return counts[n - 1]\n", "correct", None),
    ("bracketings", _B_PLUS, "wrong", "plus"),
    ("bracketings", _B_LEFT, "wrong", "left-twice"),
    ("bracketings", _B_TOTAL, "wrong", "total"),
    ("bracketings", _B_LIST, "wrong", "list"),
    ("bracketings", _B_ONE, "wrong", "one"),
    ("bracketings", _B.replace("range(2, n + 1)", "range(2, n)"), "error", "index"),
    ("bracketings", _B.replace("range(1, size)", "range(1, size + 1)"), "error", "index"),
    ("bracketings", "def bracketings(n):\n    return 5\n", "almost", None),
    ("kinds", {}, "empty", None),
    ("kinds", {"lens": "attach", "lights": "attach", "fish": "and", "club": "and", "union": "role", "washing": "role"},
     "correct", None),
    ("kinds", {"lens": "attach", "fish": "and"}, "wrong", "unfinished"),
    ("kinds", {"union": "attach"}, "wrong", "wrong:union"),
    ("kinds", {"lens": "attach", "lights": "and", "fish": "and", "club": "and", "union": "role", "washing": "role"},
     "wrong", "wrong:lights"),
]
