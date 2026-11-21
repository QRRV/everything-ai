# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-phrase-structure: rewrite the first non-terminal of a row of symbols, derive a sentence
from S with a list of choices, tell constituents from other groups of words, and find the rule that makes a grammar
endless. The grammar is the course's own, after the grammar L0 of Jurafsky and Martin (Speech and Language Processing,
chapter 19), with the same rules in English and in Dutch and words of the course's own."""
import contextlib
import copy
import html
import io
import random

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, with_time_limit

# --- the course's grammar --------------------------------------------------------------------------------------------

# the rules for the groups, the same in both languages
RULES = {
    "S": [["NP", "VP"]],
    "NP": [["DET", "NOM"], ["PROPN"]],
    "NOM": [["NOUN"], ["ADJ", "NOM"]],
    "VP": [["VERB", "NP"], ["VERB", "NP", "PP"]],
    "PP": [["ADP", "NP"]],
}
# the lexicon: each tag with its words; the Dutch nouns are all de-words, so that the adjectives keep their -e
WORDS = {
    "en": {"DET": ["the", "a"], "ADJ": ["late", "long"], "NOUN": ["train", "bridge", "city"],
           "PROPN": ["Boston", "Chicago"], "VERB": ["passed", "left"], "ADP": ["near", "after"]},
    "nl": {"DET": ["de", "een"], "ADJ": ["late", "lange"], "NOUN": ["trein", "brug", "stad"],
           "PROPN": ["Utrecht", "Breda"], "VERB": ["passeerde", "verliet"], "ADP": ["bij", "na"]},
}
GRAMMARS = {lang: RULES | {tag: [[w] for w in words] for tag, words in WORDS[lang].items()} for lang in WORDS}
GRAMMAR_EN, GRAMMAR_NL = GRAMMARS["en"], GRAMMARS["nl"]

# small grammars of the check's own: one rule that comes back to S, and sums with brackets
BRACKETS = {"S": [["a", "S", "b"], ["c"]]}
SUMS = {"S": [["S", "+", "T"], ["T"]], "T": [["x"], ["y"], ["(", "S", ")"]]}

# the choices for the sentence of The idea and for the sentence under Exercise 2
IDEA_CHOICES = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2]                  # the late train left the city
EXAMPLE_CHOICES = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]   # a long train passed the bridge near Chicago
SELF_RULE = ("NOM", ["ADJ", "NOM"])   # the rule that comes back to its own symbol


def first_open(symbols, grammar):
    """The place of the first non-terminal in `symbols`, or None: the function of the worked example."""
    for i in range(len(symbols)):
        if symbols[i] in grammar:
            return i
    return None


def rewrite_first(symbols, grammar, k):
    """The course's solution of Exercise 1."""
    i = first_open(symbols, grammar)
    right = grammar[symbols[i]][k]
    return symbols[:i] + right + symbols[i + 1:]


def derive(grammar, choices):
    """The course's solution of Exercise 2."""
    symbols = ["S"]
    steps = [symbols]
    for k in choices:
        symbols = rewrite_first(symbols, grammar, k)
        steps.append(symbols)
    return steps


def sentence(lang: str, choices) -> str:
    return " ".join(derive(GRAMMARS[lang], choices)[-1])


def without_self_rule(grammar: dict) -> dict:
    """A copy of the grammar without NOM → ADJ NOM."""
    out = copy.deepcopy(grammar)
    left, right = SELF_RULE
    if left in out:
        out[left] = [r for r in out[left] if r != right]
    return out


def sentence_counts(grammar: dict, n_max: int) -> list:
    """How many sentences of 0, 1, ..., n_max words the grammar makes from S: the number of derivations of each length,
    which is the number of sentences when every sentence has one tree, as in the grammar of the lesson. A symbol that is
    not a key is a word. Rules that go round in a circle without words, such as A → B and B → A, are not counted twice."""
    symbols = list(grammar)
    count = {x: [0] * (n_max + 1) for x in symbols}

    def ways(right: list, n: int) -> int:
        row = [1] + [0] * n            # row[a]: ways for the symbols so far to make a words
        for y in right:
            new = [0] * (n + 1)
            for a in range(n + 1):
                if row[a]:
                    for b in range(1, n - a + 1):
                        c = count[y][b] if y in count else int(b == 1)
                        if c:
                            new[a + b] += row[a] * c
            row = new
        return row[n]

    for n in range(1, n_max + 1):
        for _ in range(len(symbols) + 1):   # a rule such as NP → PROPN needs PROPN's count of the same length first
            for x in symbols:
                count[x][n] = sum(ways(r, n) for r in grammar[x])
    return count["S"] if "S" in count else [0] * (n_max + 1)


def loop_symbols(grammar: dict) -> list:
    """The non-terminals that come back to themselves through the rules, directly or through other rules. Only symbols
    that S can reach and that can end in words count, and only rules whose symbols can all end in words."""
    ends = set()                     # the non-terminals that can end in words
    for _ in range(len(grammar) + 1):
        for x in grammar:
            if any(all(y in ends or y not in grammar for y in r) for r in grammar[x]):
                ends.add(x)
    rules = {x: [r for r in grammar[x] if all(y in ends or y not in grammar for y in r)] for x in grammar if x in ends}
    used = {"S"} if "S" in rules else set()
    for _ in range(len(grammar) + 1):
        used |= {y for x in used for r in rules[x] for y in r if y in rules}
    grammar = {x: rules[x] for x in rules if x in used}   # in the order of the grammar
    reach = {x: {y for r in grammar[x] for y in r if y in grammar} for x in grammar}
    for _ in range(len(grammar)):
        for x in grammar:
            reach[x] |= set().union(*(reach[y] for y in reach[x])) if reach[x] else set()
    return [x for x in grammar if x in reach[x]]


def random_sentence(grammar: dict, rng: random.Random, max_words: int):
    """A sentence of at most max_words words, each rule of a symbol equally likely, or None after 200 tries."""
    for _ in range(200):
        symbols, steps = ["S"], 0
        while steps < 60 and len(symbols) <= max_words:
            i = first_open(symbols, grammar)
            if i is None:
                return " ".join(symbols)
            symbols = symbols[:i] + rng.choice(grammar[symbols[i]]) + symbols[i + 1:]
            steps += 1
    return None


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.88rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:2px"
_CODE = "font-family:var(--ea-font-code);font-size:12px"
_CHIP = "display:inline-block;margin:0 0.2rem 0.2rem 0;padding:0.05rem 0.35rem;border-radius:4px"
_SYMBOL = (_CHIP + ";font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink-2);background:var(--ea-inset);"
           "border:1px solid var(--ea-line)")
_NEXT = ";border-bottom:2px solid var(--ea-ink)"
_WORD = _CHIP + ";font-family:var(--ea-font-ui);font-size:0.9rem;color:var(--ea-ink)"
_K = "display:inline-block;width:3.6rem;font-family:var(--ea-font-code);font-size:0.75rem;color:var(--ea-ink-3)"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _num(n: int, lang: str) -> str:
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _is_row(value) -> bool:
    return isinstance(value, list) and all(isinstance(s, str) for s in value)


def _tree(steps: list, grammar: dict):
    """The tree of a derivation: (root, n), where n counts the steps after the first that rewrite the first
    non-terminal of the row before by one of its rules. A node is [symbol, children]; the tree stops at the first step
    that does not fit."""
    root = ["S", []]
    leaves = [root]
    n = 0
    for before, after in zip(steps, steps[1:]):
        i = first_open(before, grammar)
        if i is None or not _is_row(after):
            break
        width = len(after) - len(before) + 1
        right = after[i:i + width]
        if width < 1 or right not in grammar[before[i]] or after[:i] != before[:i] or after[i + width:] != before[i + 1:]:
            break
        kids = [[s, []] for s in right]
        leaves[i][1] = kids
        leaves = leaves[:i] + kids + leaves[i + 1:]
        n += 1
    return root, n


def _depth(node) -> int:
    return 1 + max((_depth(k) for k in node[1]), default=-1)


def _leaves(node, out):
    if node[1]:
        for k in node[1]:
            _leaves(k, out)
    else:
        out.append(node)
    return out


def tree_svg(root, grammar: dict, alt: str) -> str:
    """A parse tree: the non-terminals in grey code letters, the words in ink along the bottom, a line from each symbol
    to its children."""
    leaves = _leaves(root, [])
    bottom = _depth(root)
    row = 46

    def label_width(node) -> float:
        word = node[0] not in grammar
        return max(34.0, (8.6 if word else 8.0) * len(node[0]) + 16)

    # the width of a leaf's column: its own label, and the labels of the symbols right above it with one child
    widths = {id(leaf): label_width(leaf) for leaf in leaves}

    def column(node):
        if not node[1]:
            return node
        if len(node[1]) == 1:
            leaf = column(node[1][0])
            widths[id(leaf)] = max(widths[id(leaf)], label_width(node))
            return leaf
        for k in node[1]:
            column(k)
        return None

    column(root)
    x, xs = 8.0, {}
    for leaf in leaves:
        xs[id(leaf)] = x + widths[id(leaf)] / 2
        x += widths[id(leaf)] + 6
    width = x + 2

    def place(node, depth):
        """x and y of each node; the words stand on the bottom row."""
        word = node[0] not in grammar
        y = 20 + (bottom if word else depth) * row
        if node[1]:
            spots = [place(k, depth + 1) for k in node[1]]
            nx = (spots[0][0] + spots[-1][0]) / 2
        else:
            nx = xs[id(node)]
        node.append((nx, y))
        return nx, y

    place(root, 0)
    parts = []

    def draw(node):
        nx, y = node[2]
        for k in node[1]:
            kx, ky = k[2]
            parts.append(f'<line x1="{nx:.1f}" y1="{y + 6:.1f}" x2="{kx:.1f}" y2="{ky - 15:.1f}" '
                         f'style="stroke:var(--ea-ink-3);stroke-width:1.2"/>')
            draw(k)
        word = node[0] not in grammar
        style = ("font-family:var(--ea-font-ui);font-size:15px;fill:var(--ea-ink);font-weight:600" if word else
                 "font-family:var(--ea-font-code);font-size:13px;fill:var(--ea-ink-2)")
        parts.append(f'<text x="{nx:.1f}" y="{y:.1f}" style="{style};text-anchor:middle">{html.escape(node[0])}</text>')

    draw(root)
    height = 20 + bottom * row + 12
    return (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" style="width:100%;'
            f'max-width:{width:.0f}px;min-width:{0.75 * width:.0f}px;height:auto">'
            f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg></div>')


def _bracketed(node) -> str:
    if not node[1]:
        return node[0]
    return f'[{node[0]} {" ".join(_bracketed(k) for k in node[1])}]'


_IDEA_WORDS = {
    "en": {"title": "Parse tree of the sentence: the late train left the city",
           "legend": "Each symbol covers the words below it: a constituent. Under the NP on the left, NOM comes twice, because the rule "
                     "NOM → ADJ NOM was used, and then NOM → NOUN."},
    "nl": {"title": "Ontleedboom van de zin: de late trein verliet de stad",
           "legend": "Elk symbool bestrijkt de woorden eronder: een constituent. Onder de NP links komt NOM twee keer voor, omdat de regel "
                     "NOM → ADJ NOM werd gebruikt, en daarna NOM → NOUN."},
}


def idea_picture(lang: str) -> str:
    """The parse tree of the sentence of The idea, by the grammar of the lesson."""
    w = _IDEA_WORDS[lang]
    grammar = GRAMMARS[lang]
    root, _ = _tree(derive(grammar, IDEA_CHOICES), grammar)
    alt = f'{w["title"]}: {_bracketed(root)}. {w["legend"]}'
    return yours.figure(w["title"], tree_svg(root, grammar, alt) + _note(w["legend"]), alt=alt)


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


def _row_html(symbols: list, grammar: dict, k=None, off: bool = False) -> str:
    nxt = first_open(symbols, grammar)
    chips = []
    for i, s in enumerate(symbols[:30]):
        if s in grammar:
            chips.append(f'<span style="{_SYMBOL}{_NEXT if i == nxt else ""}">{html.escape(s)}</span>')
        else:
            chips.append(f'<span style="{_WORD}">{html.escape(s)}</span>')
    more = " …" if len(symbols) > 30 else ""
    label = "" if k is None else f"k = {k}"
    frame = f";{_OFF}" if off else ""
    return (f'<div style="margin:0.1rem 0;white-space:nowrap"><span style="{_K}">{label}</span>'
            f'<span style="display:inline-block{frame}">{"".join(chips)}{more}</span></div>')


_STEPS_WORDS = {
    "en": {"title": "Your rewrite_first, step by step, to the sentence: the late train left the city",
           "legend": "Each row after the first is one call of your function, with its k on the left. Grey: "
                     "non-terminals; the underlined one is rewritten next.",
           "fine": "Your 13 steps are the course's. The words appear from left to right, because each step "
                   "rewrites the first non-terminal.",
           "differs": "From the row with the purple dashed frame on, your rows differ from the course's.",
           "error": "Your function stops with an error at step {n}.",
           "not_row": "At step {n}, your function gives back something that is not a list of symbols.",
           "slow": "Your function takes too long on one step."},
    "nl": {"title": "Jouw rewrite_first, stap voor stap, naar de zin: de late trein verliet de stad",
           "legend": "Elke rij na de eerste is één aanroep van je functie, met de k ervan links. Grijs: "
                     "non-terminals; de onderstreepte wordt als volgende herschreven.",
           "fine": "Je 13 stappen zijn die van de course. De woorden verschijnen van links naar rechts, omdat elke "
                   "stap de eerste non-terminal herschrijft.",
           "differs": "Vanaf de rij met het paarse gestreepte kader verschillen je rijen van die van de course.",
           "error": "Je functie stopt bij stap {n} met een fout.",
           "not_row": "Bij stap {n} geeft je functie iets terug dat geen lijst met symbolen is.",
           "slow": "Je functie doet te lang over één stap."},
}


def steps_picture(seen) -> str:
    """The derivation of the sentence of The idea, one row for each call of the learner's rewrite_first."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _STEPS_WORDS[seen.lang]
    grammar = GRAMMARS[seen.lang]
    want = derive(grammar, IDEA_CHOICES)
    rows, ks, end = [["S"]], [None], ""
    for n, k in enumerate(IDEA_CHOICES, start=1):
        if first_open(rows[-1], grammar) is None:
            break
        try:
            got = _call(fn, rows[-1], grammar, k)
        except Timeout:
            end = w["slow"]
            break
        if got is _STOPPED:
            end = w["error"].format(n=n)
            break
        if not _is_row(got):
            end = w["not_row"].format(n=n)
            break
        rows.append(got)
        ks.append(k)
    first_off = next((i for i, r in enumerate(rows) if i >= len(want) or r != want[i]), None)
    body = "".join(_row_html(r, grammar, k, off=i == first_off) for i, (r, k) in enumerate(zip(rows, ks)))
    if first_off is not None:
        after = w["differs"] + (" " + end if end else "")
    elif end:
        after = end
    elif len(rows) != len(want):
        after = w["differs"]
    else:
        after = w["fine"]
    said = "; ".join(" ".join(r) for r in rows)
    return yours.figure(w["title"], f'<div style="overflow-x:auto">{body}</div>' + _note(w["legend"]) + _note(after),
                        alt=f'{w["title"]}: {said}. {after}')


_DERIVE_WORDS = {
    "en": {"title": "The tree of your derivation",
           "fine": "Your {n} rows follow the grammar, and give this tree for {s}: each rewrite gave one symbol its "
                   "children.",
           "stops": "Up to row {n}, each row follows from the row before, and the tree shows that part. Row {m} does "
                    "not: it should be {want}.",
           "start": "Your first row is {got}, and a derivation starts with ['S'].",
           "short": "Your list has {n} rows, and the 17 choices make 18.",
           "error": "Your function stops with an error on the grammar of the lesson.",
           "not_steps": "Your function gives back no list of rows, each a list of symbols.",
           "slow": "Your function takes too long on the grammar of the lesson."},
    "nl": {"title": "De boom van jouw afleiding",
           "fine": "Je {n} rijen volgen de grammatica, en geven deze boom voor {s}: elke herschrijfstap hing kinderen "
                   "onder één symbool.",
           "stops": "Tot en met rij {n} volgt elke rij uit de rij ervoor, en de boom laat dat deel zien. Rij {m} niet: "
                    "die moet {want} zijn.",
           "start": "Je eerste rij is {got}, en een afleiding begint met ['S'].",
           "short": "Je lijst heeft {n} rijen, en de 17 keuzes maken er 18.",
           "error": "Je functie stopt met een fout op de grammatica van de les.",
           "not_steps": "Je functie geeft geen lijst met rijen terug, elk een lijst met symbolen.",
           "slow": "Je functie doet te lang over de grammatica van de les."},
}


def derive_picture(seen) -> str:
    """The tree of the learner's derive on the grammar of the lesson, with the 17 choices of the example."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _DERIVE_WORDS[seen.lang]
    grammar = GRAMMARS[seen.lang]
    want = derive(grammar, EXAMPLE_CHOICES)
    try:
        got = _call(fn, grammar, list(EXAMPLE_CHOICES))
    except Timeout:
        return yours.figure(w["title"], _note(w["slow"]), alt=f'{w["title"]}. {w["slow"]}')
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["error"]), alt=f'{w["title"]}. {w["error"]}')
    if not isinstance(got, list) or not got or not all(_is_row(r) for r in got):
        return yours.figure(w["title"], _note(w["not_steps"]), alt=f'{w["title"]}. {w["not_steps"]}')
    if got[0] != ["S"]:
        text = w["start"].format(got=got[0][:8])
        return yours.figure(w["title"], _note(text), alt=f'{w["title"]}. {text}')
    root, n = _tree(got, grammar)
    if n == len(want) - 1 and len(got) == len(want):
        text = w["fine"].format(n=len(got), s=" ".join(got[-1]))
    elif n < len(got) - 1:
        text = w["stops"].format(n=n + 1, m=n + 2, want=want[n + 1] if n + 1 < len(want) else want[-1])
    else:
        text = w["short"].format(n=len(got))
    alt = f'{w["title"]}: {_bracketed(root)}. {text}'
    return yours.figure(w["title"], tree_svg(root, grammar, alt) + _note(text), alt=alt)


_DEMO_WORDS = {
    "en": {"title": "Sentences of the grammar, by length", "head": ["words", "sentences"],
           "loops": "{loop} comes back to {loop}, so there is no longest sentence: the grammar keeps making longer "
                    "ones.",
           "own": "The table counts trees; with the rules of the lesson, each sentence has one. With rules of "
                  "your own, a sentence with two trees counts twice.",
           "finite": "No rule comes back to its own symbol: the grammar makes {total} sentences, none longer than {n} "
                     "words.",
           "examples": "Three random sentences of at most {n} words:",
           "bad": "The demo cannot read the grammar: GRAMMAR must be a dict, with a list of right-hand sides for each "
                  "key, each a list of symbols.",
           "circle": "The grammar has rules that go round in a circle without words, such as A → B and B → A."},
    "nl": {"title": "Zinnen van de grammatica, naar lengte", "head": ["woorden", "zinnen"],
           "loops": "{loop} komt terug bij {loop}, dus er is geen langste zin: de grammatica blijft langere maken.",
           "own": "De tabel telt bomen; met de regels van de les heeft elke zin er één. Met eigen regels telt een zin "
                  "met twee bomen dubbel.",
           "finite": "Geen regel komt terug bij het eigen symbool: de grammatica maakt {total} zinnen, geen ervan "
                     "langer dan {n} woorden.",
           "examples": "Drie willekeurige zinnen van hoogstens {n} woorden:",
           "bad": "De demo kan de grammatica niet lezen: GRAMMAR moet een dict zijn, met bij elke key een lijst met "
                  "rechterkanten, elk een lijst met symbolen.",
           "circle": "De grammatica heeft regels die zonder woorden in een kring gaan, zoals A → B en B → A."},
}


def _good_grammar(grammar) -> bool:
    return (isinstance(grammar, dict) and 0 < len(grammar) <= 200
            and all(isinstance(x, str) and isinstance(rs, list) and rs and all(_is_row(r) and r for r in rs)
                    for x, rs in grammar.items()))


def _unit_circle(grammar: dict) -> bool:
    """Whether rules of one symbol, such as A → B, go round in a circle."""
    unit = {x: {r[0] for r in grammar[x] if len(r) == 1 and r[0] in grammar} for x in grammar}
    for _ in range(len(grammar)):
        for x in grammar:
            unit[x] |= set().union(*(unit[y] for y in unit[x])) if unit[x] else set()
    return any(x in unit[x] for x in grammar)


def demo_picture(grammar, with_rule: bool, max_words: int, seed: int, lang: str) -> str:
    """How many sentences of each length the grammar of the lesson makes, with or without NOM → ADJ NOM, and three
    random sentences."""
    w = _DEMO_WORDS[lang]
    if not _good_grammar(grammar):
        return yours.figure(w["title"], _note(w["bad"]), alt=f'{w["title"]}. {w["bad"]}')
    g = grammar if with_rule else without_self_rule(grammar)
    if _unit_circle(g):
        return yours.figure(w["title"], _note(w["circle"]), alt=f'{w["title"]}. {w["circle"]}')
    n_max = max(1, min(int(max_words), 16))
    counts = sentence_counts(g, n_max)
    rows = [[(str(n), ""), (_num(c, lang), "")] for n, c in enumerate(counts) if n >= 1 and (c or n >= 3)]
    loops = loop_symbols(g)
    if loops:
        after = w["loops"].format(loop=loops[0])
    else:
        far = sentence_counts(g, 40)
        longest = max((n for n, c in enumerate(far) if c), default=0)
        after = w["finite"].format(total=_num(sum(far), lang), n=longest)
    rng = random.Random(1000 + int(seed))
    examples = [s for s in (random_sentence(g, rng, n_max) for _ in range(3)) if s]
    shown = "".join(f'<div style="font-family:var(--ea-font-ui);font-size:0.92rem;color:var(--ea-ink);'
                    f'margin:0.1rem 0 0 0.6rem">{html.escape(s)}</div>' for s in examples)
    said = "; ".join(f"{n}: {_num(c, lang)}" for n, c in enumerate(counts) if n >= 1)
    if grammar != GRAMMAR_EN and grammar != GRAMMAR_NL:
        after += " " + w["own"]
    body = (_table(rows, w["head"]) + _note(after)
            + (_note(w["examples"].format(n=n_max)) + shown if examples else ""))
    return yours.figure(w["title"], body, alt=f'{w["title"]}: {said}. {after} {" / ".join(examples)}')


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["rewrite_first", "derive", "constituents", "endless"]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


GRAMMAR_N = Named("GRAMMAR", GRAMMAR_EN)   # only cases whose answer has no words: the same in both languages

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

# exercise 1: rewrite_first
REWRITE_CASES = [
    (["NP", "VP"], GRAMMAR_N, 0),
    (["NOM", "VP"], GRAMMAR_N, 1),
    (["a", "S", "b"], BRACKETS, 0),
    (["a", "S", "b"], BRACKETS, 1),
    (["x", "+", "T"], SUMS, 2),
    (["T", "+", "T"], SUMS, 0),
    (["(", "T", ")", "+", "S"], SUMS, 1),
]
_R_START = "def rewrite_first(symbols, grammar, k):\n    ...\n"
_R = ("def rewrite_first(symbols, grammar, k):\n"
      "    i = first_open(symbols, grammar)\n"
      "    right = grammar[symbols[i]][k]\n"
      "    return symbols[:i] + right + symbols[i + 1:]\n")
_R_NESTED = _R.replace("symbols[:i] + right + symbols", "symbols[:i] + [right] + symbols")
_R_KEEP = _R.replace("symbols[i + 1:]", "symbols[i:]")
_R_REST = _R.replace(" + symbols[i + 1:]", "")
_R_BEFORE = _R.replace("symbols[:i] + right", "right")
_R_ZERO = _R.replace("grammar[symbols[i]][k]", "grammar[symbols[i]][0]")
_R_ALL = _R.replace("grammar[symbols[i]][k]", "grammar[symbols[i]]")
_R_START0 = ("def rewrite_first(symbols, grammar, k):\n"
             "    right = grammar[symbols[0]][k]\n"
             "    return right + symbols[1:]\n")
_R_PLACE = _R.replace("grammar[symbols[i]][k]", "grammar[i][k]")

# exercise 2: derive
DERIVE_CASES = [
    (BRACKETS, [0, 0, 1]),
    (SUMS, [0, 1, 0, 2, 1, 1]),
    (BRACKETS, [1]),
    (BRACKETS, []),
    (GRAMMAR_N, [0, 1]),
]
_D_START = "def derive(grammar, choices):\n    ...\n"
_D = ("def derive(grammar, choices):\n"
      "    symbols = [\"S\"]\n"
      "    steps = [symbols]\n"
      "    for k in choices:\n"
      "        symbols = rewrite_first(symbols, grammar, k)\n"
      "        steps.append(symbols)\n"
      "    return steps\n")
_D_NOSTART = _D.replace("steps = [symbols]", "steps = []")
_D_TEXT = _D.replace("steps = [symbols]", 'steps = ["S"]')
_D_FINAL = _D.replace("    return steps\n", "    return symbols\n")
_D_BEFORE = _D.replace("        symbols = rewrite_first(symbols, grammar, k)\n        steps.append(symbols)\n",
                       "        steps.append(symbols)\n        symbols = rewrite_first(symbols, grammar, k)\n")
_D_ZERO = _D.replace("rewrite_first(symbols, grammar, k)", "rewrite_first(symbols, grammar, 0)")
_D_NOASSIGN = _D.replace("        symbols = rewrite_first(", "        rewrite_first(")
_D_EARLY = _D.replace("    return steps\n", "        return steps\n")
_D_RESET = _D.replace("        symbols = rewrite_first(symbols, grammar, k)\n",
                      "        symbols = [\"S\"]\n        symbols = rewrite_first(symbols, grammar, k)\n")


def _key_word(e) -> bool:
    return isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str)


def _key_place(e) -> bool:
    return isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], int)


EXERCISES = {
    "rewrite_first": FunctionBox(
        name="rewrite_first",
        picture=steps_picture,
        function="rewrite_first",
        start=_R_START,
        solution=_R,
        visible=REWRITE_CASES[:1],
        hidden=REWRITE_CASES[1:],
        names={"first_open": first_open},
        first=[_DOTS],
        mistakes=[
            Mistake("word-key", error=_key_word,
                    en="With `{call}`, your function looks up a word in `grammar`, and only non-terminals are keys. Find the place of the first non-terminal with `first_open`, and rewrite the symbol at that place.",
                    nl="Met `{call}` zoekt je functie een woord op in `grammar`, en alleen non-terminals zijn keys. Zoek de plaats van de eerste non-terminal met `first_open`, en herschrijf het symbool op die plaats."),
            Mistake("place-key", error=_key_place,
                    en="With `{call}`, your function looks up the place `i` in `grammar`, and the keys are symbols. Look up the symbol that stands at place `i`.",
                    nl="Met `{call}` zoekt je functie de plaats `i` op in `grammar`, en de keys zijn symbolen. Zoek het symbool op dat op plaats `i` staat."),
            Mistake("nested", wrong=_R_NESTED,
                    en="`{call}` gives {got}, and it should give {want}: the right-hand side is one item in your list, a list inside the list. Glue it in with `+`, so that its symbols stand in the row one by one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de rechterkant is één item in je lijst, een lijst in de lijst. Plak die erin met `+`, zodat de symbolen ervan één voor één in de rij staan."),
            Mistake("keep", wrong=_R_KEEP,
                    en="`{call}` gives {got}, and it should give {want}: the non-terminal you rewrote is still there, after its right-hand side. The right-hand side takes its place, so leave it out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de non-terminal die je herschreef, staat er nog, na de rechterkant. De rechterkant komt in de plaats ervan, dus laat die weg."),
            Mistake("rest", wrong=_R_REST,
                    en="`{call}` gives {got}, and it should give {want}: the symbols after the non-terminal are gone. Keep them, after the right-hand side.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de symbolen na de non-terminal zijn weg. Houd ze, na de rechterkant."),
            Mistake("before", wrong=_R_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: the symbols before the non-terminal are gone. Keep them, before the right-hand side.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de symbolen vóór de non-terminal zijn weg. Houd ze, vóór de rechterkant."),
            Mistake("zero", wrong=_R_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your function always takes the first right-hand side. Take right-hand side number `k`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt altijd de eerste rechterkant. Neem rechterkant nummer `k`."),
            Mistake("all", wrong=_R_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you put in the whole list of right-hand sides of the symbol. Take only the one with number `k`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je zet de hele lijst met rechterkanten van het symbool erin. Neem alleen die met nummer `k`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: find the first non-terminal with `first_open`, and put its right-hand side number `k` in its place.",
                      "nl": "Vervang de `...`: zoek de eerste non-terminal met `first_open`, en zet rechterkant nummer `k` ervan in de plaats ervan."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back a new list: the symbols before the first non-terminal, its right-hand side number `k`, and the symbols after it.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een nieuwe lijst terug: de symbolen vóór de eerste non-terminal, rechterkant nummer `k` ervan, en de symbolen erna."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `symbols`, `grammar` and `k` it gets, also when the first non-terminal is not at the front?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `symbols`, `grammar` en `k` die ze krijgt, ook als de eerste non-terminal niet vooraan staat?"},
            "correct": {"en": "Your `rewrite_first` is right in all {n} cases. Look at the picture: your function, step by step, from S to a sentence.",
                        "nl": "Je `rewrite_first` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie, stap voor stap, van S naar een zin."},
        },
    ),
    "derive": FunctionBox(
        name="derive",
        picture=derive_picture,
        function="derive",
        start=_D_START,
        solution=_D,
        visible=DERIVE_CASES[:1],
        hidden=DERIVE_CASES[1:],
        names={"rewrite_first": rewrite_first, "first_open": first_open},
        first=[_DOTS],
        mistakes=[
            Mistake("no-start", wrong=_D_NOSTART,
                    en="`{call}` gives {got}, and it should give {want}: the first row, `['S']`, is missing. Put it in `steps` before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste rij, `['S']`, ontbreekt. Zet die vóór de loop in `steps`."),
            Mistake("text-start", wrong=_D_TEXT,
                    en="`{call}` gives {got}, and it should give {want}: your first row is the string `'S'`. Each row is a list of symbols, such as `['S']`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je eerste rij is de string `'S'`. Elke rij is een lijst met symbolen, zoals `['S']`."),
            Mistake("final", wrong=_D_FINAL,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back only the last row. Give back `steps`, the list of all rows.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft alleen de laatste rij terug. Geef `steps` terug, de lijst met alle rijen."),
            Mistake("before", wrong=_D_BEFORE,
                    en="`{call}` gives {got}, and it should give {want}: the first row comes twice, and the last row is missing. Append each row after you rewrote it, not before.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de eerste rij staat er twee keer in, en de laatste ontbreekt. Voeg elke rij toe nadat je die herschreef, niet ervoor."),
            Mistake("zero", wrong=_D_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: each step takes the first right-hand side. Give `rewrite_first` the choice `k` of that step.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap neemt de eerste rechterkant. Geef `rewrite_first` de keuze `k` van die stap."),
            Mistake("no-assign", wrong=_D_NOASSIGN,
                    en="`{call}` gives {got}, and it should give {want}: every row stays `['S']`. `rewrite_first` gives back a new list and leaves the old one as it is: give the new list a name.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke rij blijft `['S']`. `rewrite_first` geeft een nieuwe lijst terug en laat de oude zoals die was: geef de nieuwe lijst een naam."),
            Mistake("early", wrong=_D_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back `steps` after the first choice. Move `return steps` to the left, under `for k in ...`, so that it comes after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft `steps` al na de eerste keuze terug. Zet `return steps` meer naar links, onder `for k in ...`, zodat die na de loop komt."),
            Mistake("reset", wrong=_D_RESET,
                    en="`{call}` gives {got}, and it should give {want}: each step starts again from `['S']`. Make the first row once, before the loop, and rewrite the row of the step before.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke stap begint opnieuw bij `['S']`. Maak de eerste rij één keer, vóór de loop, en herschrijf de rij van de stap ervoor."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with `['S']`, rewrite it once for each choice, and keep every row in a list.",
                      "nl": "Vervang de `...`: begin met `['S']`, herschrijf die één keer voor elke keuze, en bewaar elke rij in een lijst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives back the list of rows: `['S']`, then the row after each choice.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft de lijst met rijen terug: `['S']`, en dan de rij na elke keuze."},
            "almost": {"en": "Right for the first case. But `{call}` gives {got}, and it should give {want}. Does your function use the `grammar` and `choices` it gets, also for no choices at all?",
                       "nl": "Goed voor het eerste geval. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `grammar` en `choices` die ze krijgt, ook zonder keuzes?"},
            "correct": {"en": "Your `derive` is right in all {n} cases, also for no choices at all. Look at the picture: the tree of your rows for a longer sentence of the lesson.",
                        "nl": "Je `derive` klopt in alle {n} gevallen, ook zonder keuzes. Kijk naar de afbeelding: de boom van je rijen voor een langere zin van de les."},
        },
    ),
    "constituents": Sort(
        name="constituents",
        groups={
            "yes": {"en": "constituent", "nl": "constituent"},
            "no": {"en": "not a constituent", "nl": "geen constituent"},
        },
        items={
            "subject": {"en": "**the old ferry**", "nl": "**de oude veerboot**"},
            "object": {"en": "**the harbor**", "nl": "**de haven**"},
            "when": {"en": "**before dawn**", "nl": "**vóór zonsopgang**"},
            "det-adj": {"en": "**the old**", "nl": "**de oude**"},
            "noun-verb": {"en": "**ferry left**", "nl": "**veerboot verliet**"},
            "noun-prep": {"en": "**the harbor before**", "nl": "**de haven vóór**"},
        },
        solution={"subject": "yes", "object": "yes", "when": "yes", "det-adj": "no", "noun-verb": "no",
                  "noun-prep": "no"},
        nudge={
            "subject": {"en": "Can one word take the place of these three: *it left the harbor before dawn*?",
                        "nl": "Wat staat er in de zin vóór de persoonsvorm *verliet*?"},
            "object": {"en": "Try *it*: *the old ferry left it before dawn*. And what did the old ferry leave?",
                       "nl": "Probeer *die*: *De oude veerboot verliet die vóór zonsopgang*. En wat verliet de oude veerboot?"},
            "when": {"en": "Can *then* take its place, or can it move to the front: *Before dawn, the old ferry left the harbor*?",
                     "nl": "Kan de groep vooraan: *Vóór zonsopgang verliet de oude veerboot de haven*?"},
            "det-adj": {"en": "Try *it* for these two words: *it ferry left the harbor before dawn*. Is that English?",
                        "nl": "Zet de twee woorden vooraan: *De oude verliet veerboot de haven vóór zonsopgang*. Is dat Nederlands?"},
            "noun-verb": {"en": "Can one word take the place of these two, or can they move together?",
                          "nl": "Kan één woord de plaats van deze twee innemen, of kunnen ze samen naar een andere plek?"},
            "noun-prep": {"en": "Try *it* for these three words: *the old ferry left it dawn*. Is that English?",
                          "nl": "Zet de drie woorden vooraan: *De haven vóór verliet de oude veerboot zonsopgang*. Is dat Nederlands?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the word groups again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de woordgroepen opnieuw uit."},
            "empty": {"en": "Put each group of words under constituent or not a constituent.",
                      "nl": "Zet elke groep woorden onder constituent of geen constituent."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in the other group.",
                      "nl": "{item} hoort in de andere groep."},
            "correct": {"en": "All {n} are right. Each constituent passed a test: one word took its place, it moved to the front, or it answered a question alone.",
                        "nl": "Alle {n} kloppen. Elke constituent doorstond een test: één woord nam de plaats in, de groep kon vooraan, of de groep vormde zonder de rest een antwoord op een vraag."},
        },
    ),
}

CASES = [
    ("rewrite_first", _R_START, "empty", None),
    ("rewrite_first", _R, "correct", None),
    ("rewrite_first", "def rewrite_first(symbols, grammar, k):\n    i = first_open(symbols, grammar)\n    new = symbols[:i]\n    for s in grammar[symbols[i]][k]:\n        new.append(s)\n    return new + symbols[i + 1:]\n", "correct", None),
    ("rewrite_first", "def rewrite_first(symbols, grammar, k):\n    i = first_open(symbols, grammar)\n    symbols[i:i + 1] = grammar[symbols[i]][k]\n    return symbols\n", "correct", None),
    ("rewrite_first", _R_NESTED, "wrong", "nested"),
    ("rewrite_first", "def rewrite_first(symbols, grammar, k):\n    i = first_open(symbols, grammar)\n    symbols[i] = grammar[symbols[i]][k]\n    return symbols\n", "wrong", "nested"),
    ("rewrite_first", _R_KEEP, "wrong", "keep"),
    ("rewrite_first", _R_REST, "wrong", "rest"),
    ("rewrite_first", _R_BEFORE, "wrong", "before"),
    ("rewrite_first", _R_ZERO, "wrong", "zero"),
    ("rewrite_first", _R_ALL, "wrong", "all"),
    ("rewrite_first", _R_START0, "error", "word-key"),
    ("rewrite_first", _R_PLACE, "error", "place-key"),
    ("rewrite_first", "def rewrite_first(symbols, grammar, k):\n    return ['DET', 'NOM', 'VP']\n", "almost", None),
    ("derive", _D_START, "empty", None),
    ("derive", _D, "correct", None),
    ("derive", "def derive(grammar, choices):\n    steps = [['S']]\n    for k in choices:\n        steps.append(rewrite_first(steps[-1], grammar, k))\n    return steps\n", "correct", None),
    ("derive", _D_NOSTART, "wrong", "no-start"),
    ("derive", _D_TEXT, "wrong", "text-start"),
    ("derive", _D_FINAL, "wrong", "final"),
    ("derive", _D_BEFORE, "wrong", "before"),
    ("derive", _D_ZERO, "wrong", "zero"),
    ("derive", _D_NOASSIGN, "wrong", "no-assign"),
    ("derive", _D_EARLY, "wrong", "early"),
    ("derive", _D_RESET, "wrong", "reset"),
    ("derive", "def derive(grammar, choices):\n    return [['S'], ['a', 'S', 'b'], ['a', 'a', 'S', 'b', 'b'], ['a', 'a', 'c', 'b', 'b']]\n", "almost", None),
    ("constituents", {}, "empty", None),
    ("constituents", {"subject": "yes", "object": "yes", "when": "yes", "det-adj": "no", "noun-verb": "no",
                      "noun-prep": "no"}, "correct", None),
    ("constituents", {"subject": "yes", "object": "yes"}, "wrong", "unfinished"),
    ("constituents", {"det-adj": "yes"}, "wrong", "wrong:det-adj"),
    ("constituents", {"subject": "yes", "object": "no", "when": "yes", "det-adj": "no", "noun-verb": "no",
                      "noun-prep": "no"}, "wrong", "wrong:object"),
]
