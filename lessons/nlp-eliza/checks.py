# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-eliza: ELIZA, Joseph Weizenbaum's chatbot of 1966, which answers by rules from a script.
swap swaps the words of the script (I for YOU, MY for YOUR); keyword picks the keyword with the highest rank, the first
one on a tie; reassemble fills the parts of a decomposition into a reassembly rule.

The program is everything_ai.eliza, the course's own version of the program of 1965. The English lesson runs the DOCTOR
script of 1965 (data/eliza-doctor-1965.txt), the Dutch lesson the course's Dutch script (data/course-eliza-nl.txt).
CONVERSATION is the conversation of Weizenbaum's paper as the ELIZA team typed it into the program of 1965, with one
word changed (see CONVERSATION), and its Dutch version; tests/test_eliza.py checks that the course's ELIZA gives every
reply of 1965 (REPLIES_1965)."""
import html
import importlib.util

from everything_ai import eliza, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort

LANGS = ("en", "nl")
_KEY = "nlp-eliza"


def _summarize():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_summarize_for_eliza",
                                                  paths.LESSONS / "nlp-summarize" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _summarize()   # the parts of the pictures and the exercises

# --- the program and its scripts ---------------------------------------------------------------------------------------

SCRIPTS = {"en": eliza.doctor(), "nl": eliza.dutch()}
SWAPS = {lang: s.swaps for lang, s in SCRIPTS.items()}            # every word with its substitute
RANKS = {lang: s.keywords for lang, s in SCRIPTS.items()}         # every keyword with its rank
RULES = {lang: {w: e.rules for w, e in s.entries.items() if e.rules} for lang, s in SCRIPTS.items()}
words, clause, swap, keyword = eliza.words, eliza.clause, eliza.swap, eliza.keyword
decompose = eliza.decompose


def reassemble(template, parts):
    """The reply: each word of template that is a key of parts, such as "3", becomes the words of that part."""
    reply = []
    for word in template:
        if word in parts:
            reply += parts[word]
        else:
            reply.append(word)
    return reply


def numbered(parts) -> dict:
    """The parts of a decomposition as the exercises write them: {"1": [...], "2": [...], ...}."""
    return {str(i): list(p) for i, p in enumerate(parts, 1)}


def new_eliza(lang: str, **steps) -> eliza.Eliza:
    """A fresh ELIZA with the script of the lesson's language."""
    return eliza.Eliza(SCRIPTS[lang if lang in LANGS else "en"], **steps)


# The conversation of Weizenbaum's paper (1966), as the ELIZA team typed it into the program of 1965 for
# etc/running-eliza.txt of rupertl/eliza-ctss: the paper's "I'm" written out as "I am" in line 4, and no comma before
# "but" in line 11. The course changes one word: line 4 starts with "They", where those lines have a pronoun for one
# person. Neither word is in the script, so the reply is the same.
CONVERSATION = {
    "en": ["Men are all alike.",
           "They're always bugging us about something or other.",
           "Well, my boyfriend made me come here.",
           "They say I am depressed much of the time.",
           "It's true. I am unhappy.",
           "I need some help, that much seems certain.",
           "Perhaps I could learn to get along with my mother.",
           "My mother takes care of me.",
           "My father.",
           "You are like my father in some ways.",
           "You are not very aggressive, but I think you don't want me to notice that.",
           "You don't argue with me.",
           "You are afraid of me.",
           "My father is afraid of everybody.",
           "Bullies."],
    "nl": ["Mannen zijn allemaal hetzelfde.",
           "Mannen vallen ons altijd lastig over het een of ander.",
           "Nou, mijn vriend liet me hierheen komen.",
           "Ik ben vaak depressief, wordt er gezegd.",
           "Het is waar. Ik ben ongelukkig.",
           "Ik heb hulp nodig, zoveel is wel zeker.",
           "Misschien kan ik leren beter met mijn moeder om te gaan.",
           "Mijn moeder zorgt voor me.",
           "Mijn vader.",
           "Jij bent in sommige opzichten net als mijn vader.",
           "Je bent niet erg agressief, maar ik denk dat je niet wilt dat ik dat merk.",
           "Je bent nooit boos op me.",
           "Je bent bang voor me.",
           "Mijn vader is bang voor iedereen.",
           "Pestkoppen."],
}

# What the program of 1965 answered, in etc/running-eliza.txt of rupertl/eliza-ctss (commit 3af426b7).
REPLIES_1965 = ("DID YOU THINK THEY MIGHT NOT BE ALL ALIKE",
                "CAN YOU THINK OF A SPECIFIC EXAMPLE",
                "YOUR BOYFRIEND MADE YOU COME HERE",
                "IS IT BECAUSE YOU ARE DEPRESSED MUCH OF THE TIME THAT YOU CAME TO ME",
                "HOW LONG HAVE YOU BEEN UNHAPPY",
                "YOU SAY YOU NEED SOME HELP",
                "YOU DON'T SEEM QUITE CERTAIN",
                "TELL ME MORE ABOUT YOUR FAMILY",
                "WHO ELSE IN YOUR FAMILY",
                "WHAT MAKES YOU THINK I AM LIKE YOUR FATHER IN SOME WAYS",
                "DOES IT PLEASE YOU TO BELIEVE I AM NOT VERY AGGRESSIVE",
                "WHY DO YOU THINK I DON'T ARGUE WITH YOU",
                "DID YOUR PARENTS AM AFRAID OF YOU",
                "REALLY , EVERYONE",
                "DOES THAT HAVE ANYTHING TO DO WITH THE FACT THAT YOUR BOYFRIEND MADE YOU COME HERE")


def run(lang: str, **steps) -> list:
    """Every Step of the lesson's conversation, from a fresh ELIZA."""
    return eliza.conversation(SCRIPTS[lang if lang in LANGS else "en"], CONVERSATION[lang if lang in LANGS else "en"],
                              **steps)


def mirrored(step) -> bool:
    """Whether a reply repeats a piece of what the user typed: a memory, or a reassembly rule that puts in a part with
    words."""
    if step.how == "memory":
        return True
    return step.how == "rule" and any(w.isdigit() and step.parts[int(w) - 1] for w in step.template)


IDEA_LINE = 2   # the line of the conversation that the picture of The idea follows step by step

# --- the pictures ------------------------------------------------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _OUT, _PRED, _LOSS = S._DATA, S._FAINT, S._OUT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num = S._note, S._colors, S._label, S._num
_cell, _table, _short = S._cell, S._table, S._short
_fn, _call, _empty, _SLOW, _STOPPED = S._fn, S._call, S._empty, S._SLOW, S._STOPPED
_WHY_EMPTY = S._WHY_EMPTY
_GONE = ";text-decoration:line-through;opacity:0.65"
_MONO = "font-family:var(--ea-font-code);font-size:0.8rem"


def _chips(ws, style=_CHIP_DATA, marked=(), marked_style=_CHIP_PRED) -> str:
    """Words as chips; the words at the places in `marked` in `marked_style`."""
    if not ws:
        return f'<span style="{_UI}">–</span>'
    return "".join(f'<span style="{marked_style if i in marked else style}">{html.escape(str(w))}</span>'
                   for i, w in enumerate(ws))


def _rule_text(element) -> str:
    if isinstance(element, tuple):
        return "(" + element[0] + " ".join(element[1:]) + ")"
    return str(element)


def _pattern(rule_words) -> str:
    """A rule the way the script writes it: (0 YOUR 0)."""
    return "(" + " ".join(_rule_text(e) for e in rule_words) + ")"


def _code(text_: str) -> str:
    return f'<code style="{_MONO}">{html.escape(text_)}</code>'


_IDEA_TEXT = {
    "en": {"title": "One reply of ELIZA, step by step",
           "typed": "Line 3 of the conversation, as typed:",
           "words": "1. The words, in capitals. ELIZA keeps the first part with a keyword: the part before the comma has none and goes, and the period at the end goes too.",
           "swapped": "2. The script swaps words: {swaps}.",
           "keyword": "3. Words of the script with rules are keywords: {found}. The keyword is {key}.",
           "rank": "{w} (rank {r})",
           "memory": "4. {key} is the MEMORY keyword. The SLIP hash of the last word, {last}, picks memory rule {i} of 4, {rule}, which makes a memory for later:",
           "rule": "5. The first decomposition rule of {key} that fits is {rule}. It cuts the text into parts:",
           "part": "part {i}",
           "reply": "6. Reassembly rule {rule} gives the reply, with the words of part 3 in place of the 3:",
           "gray": "Gray: what was typed, and the words that ELIZA takes from it.",
           "blue": "Blue: what ELIZA makes of them: the swapped words, the keyword, the reply and the memory."},
    "nl": {"title": "Eén antwoord van ELIZA, stap voor stap",
           "typed": "Regel 3 van het gesprek, precies zoals de gebruiker typte:",
           "words": "1. De woorden, in hoofdletters. ELIZA houdt het eerste stuk met een sleutelwoord: het stuk tot de komma heeft er geen en valt weg, en de punt aan het eind ook.",
           "swapped": "2. Het script wisselt woorden om: {swaps}.",
           "keyword": "3. Woorden van het script met regels zijn sleutelwoorden: {found}. Het sleutelwoord is {key}.",
           "rank": "{w} (rang {r})",
           "memory": "4. {key} is het MEMORY-sleutelwoord. De SLIP-hash van het laatste woord, {last}, kiest geheugenregel {i} van 4, {rule}; deze regel maakt een herinnering voor later:",
           "rule": "5. De eerste passende ontledingsregel van {key} is {rule}. Deze regel knipt de tekst in delen:",
           "part": "deel {i}",
           "reply": "6. Opbouwregel {rule} geeft het antwoord, met de woorden van deel 3 op de plaats van de 3:",
           "gray": "Grijs: wat er getypt is, en de woorden die ELIZA daaruit haalt.",
           "blue": "Blauw: wat ELIZA ervan maakt: de omgewisselde woorden, het sleutelwoord, het antwoord en de herinnering."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    s = SCRIPTS[lang]
    step = run(lang)[IDEA_LINE]
    body = _label(w["typed"]) + f'<div style="{_DATA}">{html.escape(step.said)}</div>'
    # 1. the words, with the ones that go struck through
    kept_at = next(i for i in range(len(step.words)) if step.words[i:i + len(step.kept)] == step.kept)
    gone = set(range(len(step.words))) - set(range(kept_at, kept_at + len(step.kept)))
    body += _label(w["words"]) + "<div>" + _chips(step.words, marked=gone, marked_style=_CHIP_DATA + _GONE) + "</div>"
    # 2. the swaps
    changed = [i for i, (a, b) in enumerate(zip(step.kept, step.swapped)) if a != b]
    swaps = ", ".join(f"{step.kept[i]} → {step.swapped[i]}" for i in changed)
    body += _label(w["swapped"].format(swaps=swaps)) + "<div>" + _chips(step.swapped, marked=set(changed)) + "</div>"
    # 3. the keyword
    found = [x for x in step.kept if x in s.keywords]
    body += _label(w["keyword"].format(found=", ".join(w["rank"].format(w=x, r=s.keywords[x]) for x in found),
                                       key=step.keyword))
    body += "<div>" + _chips([step.keyword], style=_CHIP_PRED) + "</div>"
    # 4. the memory
    if step.remembered:
        i = eliza.slip_hash(step.swapped[-1], 2)
        rule = s.memory_rules[i]
        body += _label(w["memory"].format(key=step.keyword, last=step.swapped[-1], i=i + 1,
                                          rule=_pattern(rule.decomposition) + " = " + " ".join(rule.reassemblies[0])))
        body += f'<div style="{_PRED}">{html.escape(" ".join(step.remembered))}</div>'
    # 5. the decomposition
    body += _label(w["rule"].format(key=step.keyword, rule=_pattern(step.rule.decomposition)))
    rows, start = [], 0
    for i, (e, p) in enumerate(zip(step.rule.decomposition, step.parts), 1):
        swapped_here = {j for j in range(len(p)) if start + j in changed}   # the parts are the swapped words in order
        start += len(p)
        rows.append("<tr>" + _cell(html.escape(w["part"].format(i=i)) + " " + _code(_rule_text(e)), _FAINT)
                    + _cell(_chips(p, marked=swapped_here)) + "</tr>")
    body += _table(rows, ["", ""])
    # 6. the reassembly
    body += _label(w["reply"].format(rule=_pattern(step.template)))
    body += f'<div style="{_PRED}">{html.escape(step.text)}</div>'
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


_RUN_TEXT = {
    "en": {"head": ["", "Typed", "ELIZA with your {fn}"],
           "same": "With your {fn}, ELIZA gives the same reply as the program of 1965 in {k} of the {n} lines.",
           "was": "1965: {r}",
           "gray": "Gray: the lines of the conversation, and what the program of 1965 said where your ELIZA says something else.",
           "blue": "Blue: a reply of your ELIZA that is the same as in 1965.",
           "purple": "Purple: a reply, or a step, that differs from the course's ELIZA."},
    "nl": {"head": ["", "Getypt", "ELIZA met je {fn}"],
           "same": "Met je {fn} geeft ELIZA bij {k} van de {n} regels hetzelfde antwoord als de ELIZA van de course.",
           "was": "De course: {r}",
           "gray": "Grijs: de regels van het gesprek, en wat de ELIZA van de course zegt waar jouw ELIZA iets anders zegt.",
           "blue": "Blauw: een antwoord van jouw ELIZA dat hetzelfde is als bij de course.",
           "purple": "Paars: een antwoord, of een stap, dat anders is dan bij de ELIZA van de course."},
}


class _Odd(Exception):
    """A learner's function gave something a step of ELIZA cannot use."""


def _is_words(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _guarded(name: str, fn):
    """The learner's function as a step of ELIZA, stopping with _Odd when it gives what the step cannot use."""
    if name == "swap":
        def step(ws, swaps):
            out = fn(list(ws), dict(swaps))
            if not _is_words(out):
                raise _Odd
            return out
    elif name == "keyword":
        def step(ws, ranks):
            out = fn(list(ws), dict(ranks))
            if out is not None and (not isinstance(out, str) or out not in ranks):
                raise _Odd
            return out
    else:
        def step(template, parts):
            out = fn(list(template), numbered(parts))
            if not _is_words(out):
                raise _Odd
            return out
    return {name: step}


def _conversation_picture(seen, name: str, w: dict, extra) -> str:
    """The conversation with the learner's function as one step of ELIZA: per line what was typed, the step
    (extra(step, want_step) gives HTML or None) and the reply, against the course's ELIZA."""
    lang = seen.lang if seen.lang in LANGS else "en"
    w = w[lang] | _RUN_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        try:
            return run(lang, **_guarded(name, fn))
        except _Odd:
            return None

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if mine is None:
        return _empty(w, "odd")
    want = run(lang)
    rows, same = [], 0
    for i, (got, ref) in enumerate(zip(mine, want), 1):
        good = got.text == ref.text
        same += good
        reply = f'<div style="{_PRED if good else _LOSS}">{html.escape(got.text)}</div>'
        if not good:
            reply += f'<div style="{_UI}">{html.escape(w["was"].format(r=ref.text))}</div>'
        shown = extra(got, ref)
        rows.append("<tr>" + _cell(_num(i, lang), _FAINT, _NUMBER) + _cell(html.escape(got.said), _DATA)
                    + (_cell(shown) if shown is not None else "") + _cell(reply) + "</tr>")
    head = [h.format(fn=name) for h in w["head"]]
    if w.get("step"):
        head = head[:2] + [w["step"]] + head[2:]
    body = _table(rows, head, (0,)) + _note(w["same"].format(fn=name, k=_num(same, lang), n=_num(len(want), lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_SWAP_TEXT = {
    "en": {"title": "The conversation of 1966, with your swap", "step": "Kept words after your swap",
           "kind": "a list of words"},
    "nl": {"title": "Het gesprek, met je swap", "step": "Bewaarde woorden na je swap", "kind": "een lijst woorden"},
}


def _swapped_html(got, ref) -> str:
    n = max(len(got.swapped), len(ref.swapped))
    bad = {i for i in range(n) if i >= len(ref.swapped) or i >= len(got.swapped) or got.swapped[i] != ref.swapped[i]}
    changed = {i for i, (a, b) in enumerate(zip(got.kept, got.swapped)) if a != b} - bad
    out = "".join(f'<span style="{_CHIP_LOSS if i in bad else _CHIP_PRED if i in changed else _CHIP_DATA}">'
                  f"{html.escape(x)}</span>" for i, x in enumerate(got.swapped))
    return out or f'<span style="{_UI}">–</span>'


def swap_picture(seen) -> str:
    return _conversation_picture(seen, "swap", _SWAP_TEXT, _swapped_html)


_KEYWORD_TEXT = {
    "en": {"title": "The conversation of 1966, with your keyword", "step": "Your keyword",
           "none": "none", "kind": "a keyword or None"},
    "nl": {"title": "Het gesprek, met je keyword", "step": "Sleutelwoord uit je keyword", "none": "geen", "kind": "een sleutelwoord of None"},
}


def keyword_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"

    def shown(got, ref):
        text_ = got.keyword if got.keyword is not None else _KEYWORD_TEXT[lang]["none"]
        return f'<span style="{_PRED if got.keyword == ref.keyword else _LOSS}">{html.escape(text_)}</span>'

    return _conversation_picture(seen, "keyword", _KEYWORD_TEXT, shown)


_REASSEMBLE_TEXT = {
    "en": {"title": "The conversation of 1966, with your reassemble", "step": "Reassembly rule",
           "memory": "a memory", "none": "–", "kind": "a list of words"},
    "nl": {"title": "Het gesprek, met je reassemble", "step": "Opbouwregel",
           "memory": "een herinnering", "none": "–", "kind": "een lijst woorden"},
}


def reassemble_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"

    def shown(got, ref):
        if got.how == "memory":
            return f'<span style="{_UI}">{html.escape(_REASSEMBLE_TEXT[lang]["memory"])}</span>'
        if got.template is None:
            return f'<span style="{_UI}">–</span>'
        return _code(_pattern(got.template))

    return _conversation_picture(seen, "reassemble", _REASSEMBLE_TEXT, shown)


_TRY_TEXT = {
    "en": {"title": "Your conversation with ELIZA",
           "greeting": "ELIZA starts with: {g}",
           "rule": "keyword {key} · decomposition {rule} · reassembly {k} of {n}",
           "via": "keyword {key}, which goes on with the rules of {via} · decomposition {rule} · reassembly {k} of {n}",
           "none": "no keyword: a fixed line of NONE, {k} of {n}",
           "memory": "no keyword, and the counter is at 4: the oldest memory",
           "nomatch": "keyword {key}, but no decomposition rule fits: a fixed line of the program itself",
           "remembered": "Remembered for later:",
           "more": "The course answers the first {n} lines.",
           "empty": "Type one line per turn in the box.",
           "gray": "Gray: your lines, and how ELIZA found each reply.",
           "blue": "Blue: ELIZA's replies, and what it remembers."},
    "nl": {"title": "Je gesprek met ELIZA",
           "greeting": "ELIZA begint met: {g}",
           "rule": "sleutelwoord {key} · ontleding {rule} · opbouwregel {k} van {n}",
           "via": "sleutelwoord {key}, dat verdergaat met de regels van {via} · ontleding {rule} · opbouwregel {k} van {n}",
           "none": "geen sleutelwoord: een vaste zin van NONE, {k} van {n}",
           "memory": "geen sleutelwoord, en de teller staat op 4: de oudste herinnering",
           "nomatch": "sleutelwoord {key}, maar geen ontledingsregel past: een vaste zin van het programma zelf",
           "remembered": "Onthouden voor later:",
           "more": "De course beantwoordt de eerste {n} regels.",
           "empty": "Typ in het vak één regel per beurt.",
           "gray": "Grijs: je regels, en hoe ELIZA elk antwoord vond.",
           "blue": "Blauw: de antwoorden van ELIZA, en wat ELIZA onthoudt."},
}
TRY_LINES = 12      # lines of the try-it box that ELIZA answers
TRY_CHARS = 300     # characters of a line that ELIZA reads

TRY_START = {
    "en": "My sister never listens to me.\nI am tired of it.\nThe weather is nice today.\nYou are a computer.\n"
          "Everybody says that.",
    "nl": "Mijn zus luistert nooit naar me.\nIk ben het zat.\nHet weer is mooi vandaag.\nJij bent een computer.\n"
          "Niemand zegt dat.",
}


def _how(step, w) -> str:
    if step.how == "memory":
        return w["memory"]
    if step.how == "nomatch":
        return w["nomatch"].format(key=step.keyword)
    n = len(step.rule.reassemblies)
    k = step.rule.reassemblies.index(step.template) + 1
    if step.how == "none":
        return w["none"].format(k=k, n=n)
    if step.via:
        return w["via"].format(key=step.keyword, via=" → ".join(step.via), rule=_pattern(step.rule.decomposition), k=k, n=n)
    return w["rule"].format(key=step.keyword, rule=_pattern(step.rule.decomposition), k=k, n=n)


def try_picture(lang: str, text_: str = "") -> str:
    lang = lang if lang in LANGS else "en"
    w = _TRY_TEXT[lang]
    lines = [line[:TRY_CHARS] for line in str(text_ or "").splitlines() if line.strip()]
    if not lines:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["title"])
    el = new_eliza(lang)
    body = _label(w["greeting"].format(g=el.greeting))
    for line in lines[:TRY_LINES]:
        step = el.step(line)
        body += (f'<div style="margin-top:0.55rem;{_DATA}">{html.escape(line)}</div>'
                 f'<div style="{_PRED}">{html.escape(step.text)}</div>'
                 + _note(_how(step, w)))
        if step.remembered:
            body += (f'<div style="{_UI}">{html.escape(w["remembered"])} '
                     f'<span style="{_PRED}">{html.escape(" ".join(step.remembered))}</span></div>')
    if len(lines) > TRY_LINES:
        body += _note(w["more"].format(n=_num(TRY_LINES, lang)))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the exercises -----------------------------------------------------------------------------------------------------

_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS


def _show(v):
    return S._show(v)


def _written(v):
    return S._written(v)


def _is_str(got, text_) -> bool:
    return isinstance(got, str)


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


_SW_HEAD = "def swap(words, swaps):\n    new = []\n    for word in words:\n"
_SW_BODY = "        if word in swaps:\n            new.append(swaps[word])\n        else:\n            new.append(word)\n"
_SW_RET = "    return new\n"
_SW = _SW_HEAD + _SW_BODY + _SW_RET
_SW_START = _SW_HEAD + "        ...\n" + _SW_RET
_SW_SAME = _SW_HEAD + "        new.append(word)\n" + _SW_RET
_SW_DROP = _SW_HEAD + "        if word in swaps:\n            new.append(swaps[word])\n" + _SW_RET
_SW_TWICE = ("def swap(words, swaps):\n    new = list(words)\n    for old in swaps:\n        for i in range(len(new)):\n"
             "            if new[i] == old:\n                new[i] = swaps[old]\n    return new\n")
_SW_KEYS = _SW_HEAD + "        if word in swaps:\n            new.append(word)\n        else:\n            new.append(swaps[word])\n" + _SW_RET

_KW_HEAD = "def keyword(words, ranks):\n    best = None\n    best_rank = -1\n    for word in words:\n"
_KW_IF = "        if word in ranks and ranks[word] > best_rank:\n"
_KW_BODY = "            best = word\n            best_rank = ranks[word]\n"
_KW_RET = "    return best\n"
_KW = _KW_HEAD + _KW_IF + _KW_BODY + _KW_RET
_KW_START = _KW_HEAD + "        if ...:\n" + _KW_BODY + _KW_RET
_KW_TIE = _KW_HEAD + "        if word in ranks and ranks[word] >= best_rank:\n" + _KW_BODY + _KW_RET
_KW_FIRST = "def keyword(words, ranks):\n    for word in words:\n        if word in ranks:\n            return word\n    return None\n"
_KW_LAST = "def keyword(words, ranks):\n    best = None\n    for word in words:\n        if word in ranks:\n            best = word\n    return best\n"
_KW_RANK = _KW_HEAD + _KW_IF + _KW_BODY + "    return best_rank\n"
_KW_NEVER = _KW_HEAD + "        if word in ranks and ranks[word] < best_rank:\n" + _KW_BODY + _KW_RET
_KW_ZERO = _KW_HEAD + "        if word in ranks and ranks[word] > 0:\n" + _KW_BODY + _KW_RET

_RE_HEAD = "def reassemble(template, parts):\n    reply = []\n    for word in template:\n        if word in parts:\n"
_RE_PART = "            for part_word in parts[word]:\n                reply.append(part_word)\n"
_RE_ELSE = "        else:\n            reply.append(word)\n"
_RE_RET = "    return reply\n"
_RE = _RE_HEAD + _RE_PART + _RE_ELSE + _RE_RET
_RE_START = _RE_HEAD + "            ...\n" + _RE_ELSE + _RE_RET
_RE_NESTED = _RE_HEAD + "            reply.append(parts[word])\n" + _RE_ELSE + _RE_RET
_RE_NUMBERS = "def reassemble(template, parts):\n    reply = []\n    for word in template:\n        reply.append(word)\n" + _RE_RET
_RE_ONLY = _RE_HEAD + _RE_PART + _RE_RET
_RE_ALL = ("def reassemble(template, parts):\n    reply = []\n    for word in template:\n"
           "        for part_word in parts[word]:\n            reply.append(part_word)\n" + _RE_RET)

# The small cases the exercise texts show; the same in both lessons.
TOY_SWAP = ((["MY", "MOTHER", "TAKES", "CARE", "OF", "ME"], {"I": "YOU", "YOU": "I", "MY": "YOUR", "ME": "YOU"}),
            (["IK", "BEN", "ONGELUKKIG"], {"IK": "JE", "JE": "IK", "BEN": "BENT", "BENT": "BEN"}))
TOY_KEYWORD = ((["MY", "FATHER", "IS", "AFRAID", "OF", "EVERYBODY"], {"MY": 0, "EVERYBODY": 2, "I": 0}),
               (["MIJN", "VADER", "IS", "BANG", "VOOR", "IEDEREEN"], {"MIJN": 0, "IEDEREEN": 2, "IK": 0}))
TOY_REASSEMBLE = ((["WHY", "DO", "YOU", "THINK", "I", "3", "YOU"],
                   {"1": [], "2": ["I"], "3": ["DON'T", "ARGUE", "WITH"], "4": ["YOU"], "5": []}),
                  (["HOE", "LANG", "BEN", "JE", "AL", "4"], {"1": [], "2": ["JE"], "3": ["BENT"], "4": ["ONGELUKKIG"]}))


def _s(lang: str) -> _Val:
    return _Val(f'C.SWAPS["{lang}"]', SWAPS[lang])


def _r(lang: str) -> _Val:
    return _Val(f'C.RANKS["{lang}"]', RANKS[lang])


def _kept(lang: str, i: int) -> list:
    return clause(words(CONVERSATION[lang][i]), RANKS[lang])


REQUIRED = ["swap", "keyword", "reassemble", "does"]

EXERCISES = {
    "swap": FunctionBox(
        name="swap",
        picture=swap_picture,
        function="swap",
        start=_SW_START,
        solution=_SW,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_SWAP],
        hidden=[(_written([]), _written({"I": "YOU"})),
                (_written(["HELLO", "THERE"]), _written({"I": "YOU", "YOU": "I"})),
                (_written(["YOU", "AND", "I"]), _written({"I": "YOU", "YOU": "I"})),
                (_written(["I", "AND", "YOU"]), _written({"YOU": "I", "I": "YOU"})),
                (_written(["MY", "MY", "MY"]), _written({"MY": "YOUR"}))]
               + [(_written(_kept(lang, i)), _s(lang)) for lang in LANGS for i in (9, 11)],
        first=[_DOTS],
        mistakes=[
            Mistake("keys", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: your function looks up a word in `swaps` that is not a key of it. Which words can you look up in `swaps`, and what happens to the other words?",
                    nl="`{call}` stopt met een `KeyError`: je functie zoekt in `swaps` een woord op dat er geen sleutel van is. Welke woorden kun je opzoeken in `swaps`, en wat gebeurt er met de andere woorden?"),
            Mistake("string", matches=_is_str,
                    en="`{call}` gives the string {got}, and it should give the list {want}. The next steps of ELIZA work on a list of words: what does your function give back?",
                    nl="`{call}` geeft de string {got}, en het moet de lijst {want} zijn. De volgende stappen van ELIZA werken met een lijst woorden: wat geeft je functie terug?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of words, as long as `words`. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst woorden, even lang als `words`. Wat geeft je functie terug?"),
            Mistake("unchanged", wrong=_SW_SAME,
                    en="`{call}` gives {got}, and it should give {want}: every word stays as it is. Which words of `words` are keys in `swaps`, and what should they become?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord blijft zoals het is. Welke woorden van `words` zijn sleutels in `swaps`, en wat moeten deze woorden worden?"),
            Mistake("dropped", wrong=_SW_DROP,
                    en="`{call}` gives {got}, and it should give {want}: the words that have no swap are gone. What should happen to a word that is not in `swaps`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de woorden zonder omwisseling zijn weg. Wat moet er gebeuren met een woord dat niet in `swaps` staat?"),
            Mistake("twice", wrong=_SW_TWICE,
                    en="`{call}` gives {got}, and it should give {want}: a word that was swapped is swapped again by a later pair, so a word becomes its substitute and then turns back again. Can your function look up each word once, as it stood in `words`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een woord dat al omgewisseld is, wordt door een later paar opnieuw omgewisseld, dus een woord wordt de vervanger en daarna weer zichzelf. Kan je functie elk woord één keer opzoeken, zoals het in `words` stond?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with an `if` and an `else`: a word that is a key of `swaps` adds its swap to `new`, any other word adds itself.",
                      "nl": "Vervang de `...` door een `if` en een `else`: een woord dat een sleutel van `swaps` is, voegt de omgewisselde vorm toe aan `new`, elk ander woord voegt zichzelf toe."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each word that is a key of `swaps` become the word that goes with it, and does every other word stay, in the same order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wordt elk woord dat een sleutel van `swaps` is het woord dat erbij hoort, en blijft elk ander woord staan, in dezelfde volgorde?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function give an empty list for no words, and look up each word once?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie een lege lijst bij geen woorden, en zoekt de functie elk woord één keer op?"},
            "correct": {"en": "Right: your function swaps words the way the script says. In the picture below, ELIZA uses your swap on the conversation of 1966: which lines have the most swaps, and what does line 13 show about swapping word by word?",
                        "nl": "Goed: je functie wisselt woorden om zoals het script zegt. In het plaatje hieronder gebruikt ELIZA je swap op het gesprek: welke regels hebben de meeste omwisselingen, en wat laat regel 13 zien over woord voor woord omwisselen?"},
        },
    ),
    "keyword": FunctionBox(
        name="keyword",
        picture=keyword_picture,
        function="keyword",
        start=_KW_START,
        solution=_KW,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_KEYWORD],
        hidden=[(_written([]), _written({"MY": 0})),
                (_written(["MEN", "ARE", "ALIKE"]), _written({"I": 0})),
                (_written(["PERHAPS", "I", "COULD", "ASK", "MY", "MOTHER"]), _written({"PERHAPS": 0, "I": 0, "MY": 0})),
                (_written(["I", "ALWAYS", "SAY", "IF"]), _written({"I": 0, "ALWAYS": 1, "IF": 3})),
                (_written(["NOBODY", "LIKES", "EVERYBODY"]), _written({"EVERYBODY": 2, "NOBODY": 2})),
                (_written(["HELLO"]), _written({"HELLO": 0}))]
               + [(_written(_kept(lang, i)), _r(lang)) for lang in LANGS for i in (1, 9, 13, 14)],
        first=[_DOTS],
        mistakes=[
            Mistake("rank", wrong=_KW_RANK,
                    en="`{call}` gives {got}, and it should give {want}: the rank of the keyword, not the keyword itself. Which of the two does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de rang van het sleutelwoord, niet het sleutelwoord zelf. Welke van de twee geeft je functie terug?"),
            Mistake("zero", wrong=_KW_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: your function skips every keyword of rank 0, so words whose keywords all have rank 0 get none. What should the rank of a keyword be higher than, to take the place of `best`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie slaat elk sleutelwoord met rang 0 over, dus woorden met alleen sleutelwoorden van rang 0 krijgen er geen. Waarboven moet de rang van een sleutelwoord liggen om de plaats van `best` in te nemen?"),
            Mistake("tie-last", wrong=_KW_TIE,
                    en="`{call}` gives {got}, and it should give {want}: of keywords with the same rank, your function picks the last one. ELIZA keeps the first: should a keyword with the same rank take its place?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: van sleutelwoorden met dezelfde rang kiest je functie het laatste. ELIZA houdt het eerste: moet een sleutelwoord met dezelfde rang de plaats ervan innemen?"),
            Mistake("first", wrong=_KW_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the first keyword, whatever its rank. Which keyword of the words has the highest rank?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kiest het eerste sleutelwoord, wat de rang ook is. Welk sleutelwoord in de woorden heeft de hoogste rang?"),
            Mistake("last", wrong=_KW_LAST,
                    en="`{call}` gives {got}, and it should give {want}: your function picks the last keyword, whatever its rank. Does it compare the ranks?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie kiest het laatste sleutelwoord, wat de rang ook is. Vergelijkt de functie de rangen?"),
            Mistake("never", wrong=_KW_NEVER,
                    en="`{call}` gives {got}, and it should give {want}: your function finds no keyword, also when the words hold one. When should a word take the place of the best keyword so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vindt geen sleutelwoord, ook als de woorden er een bevatten. Wanneer moet een woord de plaats innemen van het beste sleutelwoord tot nu toe?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the test for a word that should become the best keyword so far: a keyword whose rank is higher than the best rank so far.",
                      "nl": "Vervang de `...` door de test voor een woord dat het beste sleutelwoord tot nu toe moet worden: een sleutelwoord met een hogere rang dan de beste rang tot nu toe."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function give the keyword with the highest rank, the first one on a tie, and `None` when the words hold no keyword?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Geeft je functie het sleutelwoord met de hoogste rang, het eerste bij gelijke rang, en `None` als de woorden geen sleutelwoord bevatten?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does your function give `None` when the words hold no keyword, and keep the first of two keywords with the same rank?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `None` als de woorden geen sleutelwoord bevatten, en houdt de functie het eerste van twee sleutelwoorden met dezelfde rang?"},
            "unreached": {"en": "`{call}` gives `None`, but the words hold a keyword. Which keyword has the highest rank, and does your function get to `return best` with it?",
                          "nl": "`{call}` geeft `None`, maar de woorden bevatten een sleutelwoord. Welk sleutelwoord heeft de hoogste rang, en komt je functie daarmee bij `return best`?"},
            "no_return": {"en": "`{call}` gives `None`, but the words hold a keyword. Does your function end with `return best`?",
                          "nl": "`{call}` geeft `None`, maar de woorden bevatten een sleutelwoord. Eindigt je functie met `return best`?"},
            "correct": {"en": "Right: your function picks the keyword the way ELIZA does. In the picture below, which lines have a keyword with a rank above 0, and which keyword does line 7 get?",
                        "nl": "Goed: je functie kiest het sleutelwoord zoals ELIZA het doet. Kijk in het plaatje hieronder: welke regels hebben een sleutelwoord met een rang boven 0, en welk sleutelwoord krijgt regel 7?"},
        },
    ),
    "reassemble": FunctionBox(
        name="reassemble",
        picture=reassemble_picture,
        function="reassemble",
        start=_RE_START,
        solution=_RE,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_REASSEMBLE],
        hidden=[(_written(["WHO", "ELSE", "IN", "YOUR", "FAMILY", "5"]),
                 _written({"1": [], "2": ["YOUR"], "3": [], "4": ["FATHER"], "5": []})),
                (_written(["REALLY", ",", "2", "3"]), _written({"1": [], "2": ["IF"], "3": ["I", "WERE", "RICH"]})),
                (_written(["PLEASE", "GO", "ON"]), _written({"1": ["I", "AM", "HERE"]})),
                (_written(["YOU", "SAY", "1", "AND", "1"]), _written({"1": ["NO"]})),
                (_written([]), _written({"1": ["A"]})),
                (_written(["YOUR", "3"]), _written({"1": [], "2": ["YOUR"], "3": ["BOYFRIEND", "MADE", "YOU", "COME", "HERE"]})),
                (_written(["JOUW", "4"]), _written({"1": [], "2": ["JOUW"], "3": [], "4": ["VADER"], "5": []}))],
        first=[_DOTS],
        mistakes=[
            Mistake("keys", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: your function looks up a word in `parts` that is not a number of a part. Which words of the template are keys of `parts`, and what happens to the other words?",
                    nl="`{call}` stopt met een `KeyError`: je functie zoekt in `parts` een woord op dat geen nummer van een deel is. Welke woorden van de opbouwregel zijn sleutels van `parts`, en wat gebeurt er met de andere woorden?"),
            Mistake("string", matches=_is_str,
                    en="`{call}` gives the string {got}, and it should give the list {want}. ELIZA joins the words only when it prints the reply: what does your function give back?",
                    nl="`{call}` geeft de string {got}, en het moet de lijst {want} zijn. ELIZA plakt de woorden pas aan elkaar bij het printen van het antwoord: wat geeft je functie terug?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of words. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst woorden. Wat geeft je functie terug?"),
            Mistake("nested", wrong=_RE_NESTED,
                    en="`{call}` gives {got}, and it should give {want}: a part ends up in the reply as a list inside the list. Can the words of the part go into the reply one at a time?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een deel komt als lijst in de lijst van het antwoord terecht. Kunnen de woorden van het deel één voor één in het antwoord?"),
            Mistake("numbers", wrong=_RE_NUMBERS,
                    en="`{call}` gives {got}, and it should give {want}: the numbers of the template are still in the reply. What should a number such as `\"3\"` become?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de nummers van de opbouwregel staan nog in het antwoord. Wat moet een nummer zoals `\"3\"` worden?"),
            Mistake("only-parts", wrong=_RE_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: only the words of the parts are in the reply, and the words of the template itself are gone. What should happen to a word of the template that is not a number?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de woorden van de delen staan in het antwoord, en de woorden van de opbouwregel zelf zijn weg. Wat moet er gebeuren met een woord van de opbouwregel dat geen nummer is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: a number of the template, such as `\"3\"`, adds the words of that part to `reply`, one at a time.",
                      "nl": "Vervang de `...`: een nummer van de opbouwregel, zoals `\"3\"`, voegt de woorden van dat deel toe aan `reply`, één voor één."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each number of the template become the words of its part, and does each other word stay as it is, in the same order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wordt elk nummer van de opbouwregel de woorden van het deel, en blijft elk ander woord zoals het is, in dezelfde volgorde?"},
            "almost": {"en": "Right for the two examples. But `{call}` gives {got}, and it should give {want}. Does an empty part add nothing, and does a number that comes twice add its part twice?",
                       "nl": "Goed voor de twee voorbeelden. Maar `{call}` geeft {got}, en het moet {want} zijn. Voegt een leeg deel niets toe, en voegt een nummer dat twee keer voorkomt het deel twee keer toe?"},
            "correct": {"en": "Right: your function builds replies the way ELIZA does. In the picture below, which replies take no words from the user at all, and where does reply 15 come from?",
                        "nl": "Goed: je functie bouwt antwoorden zoals ELIZA het doet. Kijk in het plaatje hieronder: welke antwoorden nemen helemaal geen woorden van de gebruiker over, en waar komt antwoord 15 vandaan?"},
        },
    ),
}

EXERCISES["does"] = Sort(
    name="does",
    groups={
        "does": {"en": "ELIZA does this", "nl": "ELIZA doet dit"},
        "not": {"en": "ELIZA does not do this", "nl": "ELIZA doet dit niet"},
    },
    items={
        "grammar": {"en": "**Grammar.** It checks that its reply is a correct sentence before it prints it.",
                    "nl": "**Grammatica.** Het controleert of het antwoord een goede zin is, voordat het antwoord geprint wordt."},
        "swap": {"en": "**Turning words around.** It turns *my* into *your* and *me* into *you*.",
                 "nl": "**Woorden omdraaien.** Het programma maakt *jouw* van *mijn* en *jou* van *me*."},
        "refers": {"en": "**Bullies.** It understands that *Bullies.* in line 15 is about the father of line 14.",
                   "nl": "**Pestkoppen.** Het begrijpt dat *Pestkoppen.* in regel 15 over de vader van regel 14 gaat."},
        "rank": {"en": "**One word.** It answers to one word of the input, the keyword with the highest rank.",
                 "nl": "**Eén woord.** Het reageert op één woord van de invoer, het sleutelwoord met de hoogste rang."},
        "boyfriend": {"en": "**Boyfriend.** It knows what a boyfriend is.",
                      "nl": "**Vriend.** Het weet wat een vriend is."},
        "cycle": {"en": "**Variety.** When the same decomposition rule fits again, it takes its next reassembly rule.",
                  "nl": "**Afwisseling.** Als dezelfde ontledingsregel weer past, neemt het programma de volgende opbouwregel ervan."},
        "unhappy": {"en": "**Feelings.** It notices that the user is unhappy.",
                    "nl": "**Gevoelens.** Het merkt dat de gebruiker ongelukkig is."},
        "memory": {"en": "**Memory.** Later in the conversation, it brings back something the user said about *my*.",
                   "nl": "**Geheugen.** Later in het gesprek haalt het iets terug wat de gebruiker over *mijn* zei."},
    },
    solution={"grammar": "not", "swap": "does", "refers": "not", "rank": "does",
              "boyfriend": "not", "cycle": "does", "unhappy": "not", "memory": "does"},
    nudge={
        "grammar": {"en": "Look at reply 13 of the conversation of 1966. Would a program that checks its sentences print it?",
                    "nl": "Kijk naar antwoord 13 van het gesprek. Zou een programma dat de zinnen controleert dit printen?"},
        "swap": {"en": "Which entries of the script have an equals sign, and what happens to such a word in the input?",
                 "nl": "Welke woorden in het script hebben een isgelijkteken erachter, en wat gebeurt er met zo'n woord in de invoer?"},
        "refers": {"en": "*Bullies.* holds no keyword. Where did reply 15 come from, and when was it made?",
                   "nl": "*Pestkoppen.* bevat geen sleutelwoord. Waar kwam antwoord 15 vandaan, en wanneer werd het gemaakt?"},
        "rank": {"en": "In line 14, both *my* and *everybody* are keywords. To which of the two did ELIZA answer, and why?",
                 "nl": "In regel 14 zijn *mijn* en *iedereen* allebei sleutelwoorden. Op welke van de twee reageerde ELIZA, en waarom?"},
        "boyfriend": {"en": "Is *boyfriend* a word of the script? What does ELIZA do with a word that is not?",
                      "nl": "Is *vriend* een woord van het script? Wat doet ELIZA met een woord dat er niet in staat?"},
        "cycle": {"en": "Lines 4 and 5 of the conversation fit the same decomposition rule. Do they get the same reply?",
                  "nl": "Regel 4 en regel 5 van het gesprek passen bij dezelfde ontledingsregel. Krijgen beide zinnen hetzelfde antwoord?"},
        "unhappy": {"en": "What would ELIZA answer to *I am happy*? Which part of the decomposition rule `(0 YOU ARE 0)` looks at what the user feels?",
                    "nl": "Wat zou ELIZA antwoorden op *Ik ben blij*? Welk deel van de ontledingsregel `(0 JE BENT 0)` kijkt naar wat de gebruiker voelt?"},
        "memory": {"en": "What did the MEMORY keyword do with line 3, and when did that come back?",
                   "nl": "Wat deed het MEMORY-sleutelwoord met regel 3, en wanneer kwam dat terug?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Sort each statement into the group that fits.",
                  "nl": "Deel elke uitspraak in bij de groep die klopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other group.",
                  "nl": "{item} hoort bij de andere groep."},
        "correct": {"en": "All {n} are right. Everything ELIZA does is done to words: swapping them, ranking them, cutting them into parts and putting them back, taking turns between replies and keeping a sentence for later. Nothing in it knows what a word means, which is why a sentence about bullies can be answered with a boyfriend, and *DID YOUR PARENTS AM AFRAID OF YOU* gets printed.",
                    "nl": "Alle {n} kloppen. Alles wat ELIZA doet, gebeurt met woorden: omwisselen, een rang geven, in delen knippen en weer samenvoegen, antwoorden afwisselen en een zin bewaren voor later. Niets erin weet wat een woord betekent; zo krijgt een zin over pestkoppen een antwoord over een vriend, en wordt *HEBBEN JE OUDERS BEN BANG VOOR JOU* geprint."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("swap", _SW_START, "empty", None),
    ("swap", _SW, "correct", None),
    ("swap", _SW_HEAD + "        if word in swaps:\n            new.append(...)\n        else:\n            new.append(word)\n"
     + _SW_RET, "wrong", "dots-left"),
    ("swap", _SW_KEYS, "error", "keys"),
    ("swap", _SW + "\n\ndef swap(words, swaps):\n    return ' '.join(words)\n", "wrong", "string"),
    ("swap", "def swap(words, swaps):\n    return len(words)\n", "wrong", "not-list"),
    ("swap", _SW_SAME, "wrong", "unchanged"),
    ("swap", _SW_DROP, "wrong", "dropped"),
    ("swap", _SW_TWICE, "wrong", "twice"),
    ("swap", "def swap(words, swaps):\n    return [swaps[w] if w in swaps else w for w in words]\n", "correct", None),
    ("swap", "def swap(words, swaps):\n    new = []\n    for word in words:\n        if word in swaps:\n"
     "            word = swaps[word]\n        new.append(word)\n    return new\n", "correct", None),
    ("keyword", _KW_START, "empty", None),
    ("keyword", _KW, "correct", None),
    ("keyword", _KW_HEAD + "        if word in ranks and ...:\n" + _KW_BODY + _KW_RET, "wrong", "dots-left"),
    ("keyword", _KW_RANK, "wrong", "rank"),
    ("keyword", _KW_TIE, "wrong", "tie-last"),
    ("keyword", _KW_FIRST, "wrong", "first"),
    ("keyword", _KW_LAST, "wrong", "last"),
    ("keyword", _KW_NEVER, "wrong", "never"),
    ("keyword", _KW_ZERO, "wrong", "zero"),
    ("keyword", "def keyword(words, ranks):\n    best = None\n    for word in words:\n        if word in ranks:\n"
     "            if best is None or ranks[word] > ranks[best]:\n                best = word\n    return best\n", "correct", None),
    ("keyword", "def keyword(words, ranks):\n    found = [w for w in words if w in ranks]\n    if len(found) == 0:\n"
     "        return None\n    top = max(ranks[w] for w in found)\n    for w in found:\n        if ranks[w] == top:\n"
     "            return w\n", "correct", None),
    ("reassemble", _RE_START, "empty", None),
    ("reassemble", _RE, "correct", None),
    ("reassemble", _RE_HEAD + "            for part_word in ...:\n                reply.append(part_word)\n" + _RE_ELSE + _RE_RET,
     "wrong", "dots-left"),
    ("reassemble", _RE_ALL, "error", "keys"),
    ("reassemble", _RE + "\n\ndef reassemble(template, parts):\n    return ' '.join(template)\n", "wrong", "string"),
    ("reassemble", "def reassemble(template, parts):\n    return len(template)\n", "wrong", "not-list"),
    ("reassemble", _RE_NESTED, "wrong", "nested"),
    ("reassemble", _RE_NUMBERS, "wrong", "numbers"),
    ("reassemble", _RE_ONLY, "wrong", "only-parts"),
    ("reassemble", _RE_HEAD + "            reply = reply + parts[word]\n" + _RE_ELSE + _RE_RET, "correct", None),
    ("does", {}, "empty", None),
    ("does", dict(EXERCISES["does"].solution), "correct", None),
    ("does", {"grammar": "not", "swap": "does"}, "wrong", "unfinished"),
] + [("does", {item: "not" if group == "does" else "does"}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["does"].solution.items()]
