# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-annotation: 20 short messages, 12 clear ones from the course's message set (6 spam, 6 ok)
and 8 hard ones, one of them from the set and 7 written for this lesson. Two annotators the course made up, A and B,
label all 20 twice: under guidelines 1 ("spam is a message you do not want") and under guidelines 2 (a rule with
examples). A third, C, says "ok" to everything. The labels are made up to show what happens; the agreement, the
agreement by chance and Cohen's kappa are computed from them. The functions agreement and kappa.
tests/test_data_annotation.py computes the numbers in the lesson again."""
import contextlib
import io
import json
import re
import warnings

from everything_ai import chart, check as checking, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/course-spam-messages.json"   # 60 short messages written by the course: see data/README.md
CLEAR_SPAM = [1, 3, 6, 9, 15, 18]           # positions in the test part of the message set
CLEAR_OK = [0, 4, 5, 8, 10, 14]
HARD_FROM_SET = 11                          # "we won free concert tickets text me now", ok in the set
# The 7 hard messages written for this lesson, in English as in the message set, with a Dutch version for the lesson.
HARD_WRITTEN = [
    ("the shoe shop has a sale this weekend all boots half price", "de schoenwinkel heeft dit weekend uitverkoop alle laarzen voor de halve prijs"),
    ("join our club today and get your first month free", "word vandaag lid van onze club en krijg je eerste maand gratis"),
    ("your school trip money is due friday pay at the office", "het geld voor het schoolreisje moet vrijdag binnen zijn betaal op het kantoor"),
    ("rate your last order and get ten percent off the next one", "beoordeel je laatste bestelling en krijg tien procent korting op de volgende"),
    ("vote for our band in the contest text yes to win a signed album", "stem op onze band in de wedstrijd sms ja en win een gesigneerd album"),
    ("a new cafe opened on your street come and try it", "er is een nieuw café in je straat geopend kom het eens proberen"),
    ("forward this to ten friends for good luck", "stuur dit door naar tien vrienden voor geluk"),
]
HARD_SET_NL = "we hebben gratis concertkaartjes gewonnen sms me nu"

S, O = "spam", "ok"
# The labels of the 20 messages, in this order: 6 clear spam, 6 clear ok, then the 8 hard ones (the one from the set
# first, then the 7 written for this lesson in the order above). Made up by the course.
CLEAR = [S] * 6 + [O] * 6
A1 = CLEAR + [S, S, S, O, S, S, S, S]   # guidelines 1: A calls everything that sells or pushes spam
B1 = CLEAR + [O, O, S, S, O, S, O, O]   # guidelines 1: B calls only what wants money, a payment or a click spam
A2 = CLEAR + [O, O, S, O, S, S, O, O]   # guidelines 2: both follow the rule; ten percent off stays a doubt
B2 = CLEAR + [O, O, S, O, O, S, O, O]
C2 = [O] * 20                           # C says "ok" to everything
CASES_SLIDER = {1: (A1, B1, "B"), 2: (A2, B2, "B"), 3: (A2, C2, "C")}


def messages() -> list:
    """The 20 messages in English, in the order of the labels above."""
    test = json.loads((paths.ROOT / DATA).read_text(encoding="utf-8"))["test"]
    return [test[i][1] for i in CLEAR_SPAM + CLEAR_OK + [HARD_FROM_SET]] + [en for en, _ in HARD_WRITTEN]


def agreement_of(a: list, b: list) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def spam_share(a: list) -> float:
    return a.count(S) / len(a)


def chance_of(a: list, b: list) -> float:
    pa, pb = spam_share(a), spam_share(b)
    return pa * pb + (1 - pa) * (1 - pb)


def kappa_of(a: list, b: list) -> float:
    p_o, p_e = agreement_of(a, b), chance_of(a, b)
    return (p_o - p_e) / (1 - p_e)


def number(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = "0"
    return text.replace(".", ",") if lang == "nl" else text


def share(x: float, lang: str) -> str:
    """A share with up to three decimals and no zeros at the end, such as 0.7 and 0.485."""
    text = f"{x:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", ",") if lang == "nl" else text


def percent(x: float, lang: str) -> str:
    text = f"{100 * x:.1f}".rstrip("0").rstrip(".")
    return (text.replace(".", ",") if lang == "nl" else text) + "%"


def case_said(k: int, lang: str) -> str:
    a, b, name = CASES_SLIDER[k]
    same = round(agreement_of(a, b) * 20)
    p_o, p_e, kap = agreement_of(a, b), chance_of(a, b), kappa_of(a, b)
    sa, sb = a.count(S), b.count(S)
    if lang == "nl":
        head = {1: "Richtlijnen 1: A en B.", 2: "Richtlijnen 2: A en B.", 3: "Richtlijnen 2: A en C, die overal ok zegt."}[k]
        return (f"{head} Ze zijn het eens over {same} van de 20 berichten, {percent(p_o, lang)}. A noemt {sa} berichten spam, "
                f"{name} {sb}; door toeval alleen zouden ze het over {percent(p_e, lang)} eens zijn. Kappa: {number(kap, lang)}.")
    head = {1: "Guidelines 1: A and B.", 2: "Guidelines 2: A and B.", 3: "Guidelines 2: A and C, who says ok to everything."}[k]
    return (f"{head} They agree on {same} of the 20 messages, {percent(p_o, lang)}. A calls {sa} messages spam, "
            f"{name} {sb}; by chance alone they would agree on {percent(p_e, lang)}. Kappa: {number(kap, lang)}.")


_WORDS = {
    "labels": {"en": "The labels of the 20 messages", "nl": "De labels van de 20 berichten"},
    "clear": {"en": "clear", "nl": "duidelijk"},
    "hard": {"en": "hard", "nl": "lastig"},
    "spam": {"en": "filled: spam, open: ok", "nl": "gevuld: spam, open: ok"},
    "measures": {"en": "Agreement", "nl": "Overeenstemming"},
    "agree": {"en": "agree", "nl": "eens"},
    "chance": {"en": "by chance", "nl": "toeval"},
    "kappa": {"en": "kappa", "nl": "kappa"},
}


def words(lang: str) -> dict:
    return {k: v[lang] for k, v in _WORDS.items()}


def _grid(a: list, b: list, names: tuple, w: dict):
    """A row of dots per annotator, a column per message: filled for spam, open for ok; a gray band where they differ."""
    p = chart.Panel((-2.2, 19.7), (-0.6, 1.95), title=w["labels"], xlabel=w["spam"], xticks=[], yticks=[])
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            p.rect(i - 0.45, -0.35, i + 0.45, 1.35, role="ink-3", solid=False)
    for row, labels in ((1, a), (0, b)):
        p.text(-1.3, row, names[1 - row], role="ink-2", anchor="middle")
        for hollow, value in ((False, S), (True, O)):
            xs = [i for i, v in enumerate(labels) if v == value]
            p.points(xs, [row] * len(xs), role="data", r=4.5, hollow=hollow)
    p.line([11.5, 11.5], [-0.5, 1.75], role="ink-3", width=1, dashed=True)
    p.text(5.5, 1.7, w["clear"], role="ink-2", anchor="middle")
    p.text(15.5, 1.7, w["hard"], role="ink-2", anchor="middle")
    return p


def _bars(p_o: float, p_e: float, kap: float, w: dict, lang: str):
    p = chart.Panel((-0.6, 2.6), (min(0.0, kap) - 0.05, 1.15), title=w["measures"], xticks=[0, 1, 2],
                    xtick_labels=[w["agree"], w["chance"], w["kappa"]], yticks=[0, 0.5, 1])
    for i, (v, role) in enumerate(((p_o, "ink-3"), (p_e, "ink-3"), (kap, "ink"))):
        p.rect(i - 0.3, min(0, v), i + 0.3, max(0, v), role=role, solid=True)
        shown = number(v, lang) if i == 2 else share(v, lang)   # kappa with two decimals, as in the text
        p.text(i, max(0, v) + 0.07, shown, role="ink", anchor="middle", halo=True)
    return p


def case_picture(k: int, w: dict, lang: str) -> str:
    """Left: the labels of the two annotators of comparison k, the messages where they differ in a gray band. Right:
    how often they agree, how often they would by chance, and kappa."""
    a, b, name = CASES_SLIDER[k]
    left = _grid(a, b, ("A", name), w)
    right = _bars(agreement_of(a, b), chance_of(a, b), kappa_of(a, b), w, lang)
    return chart.figure([left, right], alt=w["alt"], width=330, height=230)


# The lesson is done when these are correct. "gold" is the choice question in the lesson.
REQUIRED = ["agreement", "kappa", "gold"]


class _Val:
    """A test value that the check's messages write by its name, such as `0.7`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _function(seen_):
    fn = seen_.mine.names.get(seen_.exercise.function) if seen_.mine is not None else None
    return fn if callable(fn) else None


def _quiet(fn, *args):
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*args)), True
    except Exception:
        return None, False


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


def _same_number(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _shown_number(v):
    try:
        text = f"{float(v):.6g}"
    except (TypeError, ValueError):
        return repr(v)[:60]
    return {"en": text, "nl": text.replace(".", ",")}


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _three(fn) -> list | None:
    """What the learner's function gives for the three comparisons of the slider, or None."""
    out = []
    for k in (1, 2, 3):
        value, ok = fn(k)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not ok or value != value or abs(value) > 100:
            return None
        out.append(value)
    return out


def _three_panel(mine: list, want: list, title: str, w: dict, lang: str):
    low = min(0.0, min(mine), min(want)) - 0.05
    high = max(1.15, max(mine) + 0.15)
    p = chart.Panel((-0.6, 2.6), (low, high), title=title, xticks=[0, 1, 2], xtick_labels=[w["one"], w["two"], w["three"]],
                    yticks=None if high > 2 or low < -1 else [t for t in (-0.5, 0, 0.5, 1) if low <= t <= high])
    for i, (m, r) in enumerate(zip(mine, want)):
        p.rect(i - 0.3, min(0, m), i + 0.3, max(0, m), role="ink", solid=True)
        p.text(i, max(0, m) + 0.08, number(m, lang), role="ink", anchor="middle", halo=True)
        if abs(m - r) > 1e-9:   # the right value as a dashed line across the bar
            p.line([i - 0.42, i + 0.42], [r, r], role="ink-2", width=1.4, dashed=True)
    return p


_THREE_WORDS = {
    "one": {"en": "1: A and B", "nl": "1: A en B"},
    "two": {"en": "2: A and B", "nl": "2: A en B"},
    "three": {"en": "2: A and C", "nl": "2: A en C"},
}


# --- exercise 1: how often two annotators agree -------------------------------------------------------------------------

_AG_HEAD = "def agreement(a, b):\n    same = 0\n    for i in range(len(a)):\n"


def _ag(body: str = "        if a[i] == b[i]:\n            same = same + 1\n") -> str:
    return _AG_HEAD + body + "    return same / len(a)\n"


_AG_START = _ag("        ...\n")
_AG = _ag()
_AG_VISIBLE = ([S, O, O, S, O], [S, O, S, S, O])
_AG_HIDDEN = [([O, O, O], [O, O, O]), ([S, O], [O, S]), ([O, O, S, O], [O, S, S, O]), (list(A1), list(B1))]

_AG_WORDS = {
    "title": {"en": "Your agreement for the three comparisons of the slider", "nl": "Jouw overeenstemming voor de drie vergelijkingen van de slider"},
    "bars": {"en": "agreement", "nl": "overeenstemming"},
    "right": {"en": "Your function gives {values}: A and B agree more under guidelines 2, and even C, who says ok to everything, agrees with A on more than half of the messages.",
              "nl": "Je functie geeft {values}: A en B zijn het onder richtlijnen 2 vaker eens, en zelfs C, die overal ok zegt, is het met A over meer dan de helft van de berichten eens."},
    "said": {"en": "Your function gives {values}; the share of messages on which they agree is {want}, the dashed lines.",
             "nl": "Je functie geeft {values}; het deel van de berichten waarover ze het eens zijn is {want}, de gestreepte lijnen."},
}


def _ag_picture(seen_) -> str:
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    mine = _three(lambda k: _quiet(fn, list(CASES_SLIDER[k][0]), list(CASES_SLIDER[k][1])))
    if mine is None:
        return ""
    want = [agreement_of(*CASES_SLIDER[k][:2]) for k in (1, 2, 3)]
    aw = {k: v[lang] for k, v in _AG_WORDS.items()}
    w = {k: v[lang] for k, v in _THREE_WORDS.items()}
    values = ", ".join(number(v, lang) for v in mine)
    if all(abs(m - r) < 1e-9 for m, r in zip(mine, want)):
        text = aw["right"].format(values=values)
    else:
        text = aw["said"].format(values=values, want=", ".join(number(v, lang) for v in want))
    p = _three_panel(mine, want, aw["bars"], w, lang)
    return yours.figure(aw["title"], chart.figure([p], alt=text, width=460, height=230) + _key(text), alt=text)


EXERCISES = {
    "agreement": FunctionBox(
        name="agreement",
        picture=_ag_picture,
        function="agreement",
        start=_AG_START,
        solution=_AG,
        show=_shown_number,
        compare=_same_number,
        visible=[_AG_VISIBLE],
        hidden=_AG_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("disagree", wrong=_ag("        if a[i] != b[i]:\n            same = same + 1\n"),
                    en="`{call}` gives {got}: your function counts the messages on which the two differ. Count the ones on which they give the same label.",
                    nl="`{call}` geeft {got}: je functie telt de berichten waarover de twee verschillen. Tel de berichten waaraan ze hetzelfde label geven."),
            Mistake("spam-both", wrong=_ag("        if a[i] == \"spam\":\n            if b[i] == \"spam\":\n                same = same + 1\n"),
                    en="`{call}` gives {got}: your function counts only the messages both call spam. Both saying ok is agreeing too.",
                    nl="`{call}` geeft {got}: je functie telt alleen de berichten die allebei spam noemen. Allebei ok zeggen is het ook eens zijn."),
            Mistake("whole-lists", wrong=_ag("        if a == b:\n            same = same + 1\n"),
                    en="`{call}` gives {got}: `a == b` compares the two whole lists, which are only the same when they agree on every message. Compare one message at a time: `a[i]` and `b[i]`.",
                    nl="`{call}` geeft {got}: `a == b` vergelijkt de twee hele lijsten, en die zijn alleen hetzelfde als ze het over elk bericht eens zijn. Vergelijk één bericht tegelijk: `a[i]` en `b[i]`."),
            Mistake("always", wrong=_ag("        same = same + 1\n"),
                    en="`{call}` gives {got}: your function counts every message, so the share is always 1. Add 1 only when `a[i]` and `b[i]` are the same.",
                    nl="`{call}` geeft {got}: je functie telt elk bericht, dus het deel is altijd 1. Tel er alleen 1 bij op als `a[i]` en `b[i]` hetzelfde zijn."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with code that adds 1 to `same` for every message on which the two agree.",
                      "nl": "Vervang de `...` door code die 1 bij `same` optelt voor elk bericht waarover de twee het eens zijn."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(a, b)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(a, b)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the share of messages on which `a` and `b` give the same label.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: het deel van de berichten waaraan `a` en `b` hetzelfde label geven."},
            "almost": {"en": "Right for the five messages. But `{call}` gives {got}, and it should give {want}. Does your function look at every message of the lists it gets?",
                       "nl": "Goed voor de vijf berichten. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie naar elk bericht van de lijsten die ze krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases. On the 20 messages under guidelines 1, A and B agree on 0.7 of them.",
                        "nl": "Je functie klopt in alle {n} gevallen. Op de 20 berichten zijn A en B het onder richtlijnen 1 over 0,7 ervan eens."},
        },
    ),
}


# --- exercise 2: Cohen's kappa ------------------------------------------------------------------------------------------

_KA_HEAD = "def kappa(agree, a_spam, b_spam):\n"


def _ka(chance: str = "a_spam * b_spam + (1 - a_spam) * (1 - b_spam)", back: str = "(agree - chance) / (1 - chance)") -> str:
    return _KA_HEAD + f"    chance = {chance}\n    return {back}\n"


_KA_START = _ka("...", "...")
_KA = _ka()
_KA_VISIBLE = (0.7, 0.65, 0.45)
_KA_HIDDEN = [(0.95, 0.45, 0.4), (0.55, 0.45, 0.0), (0.9, 0.1, 0.1), (0.6, 0.5, 0.5), (0.3, 0.5, 0.5)]

_KA_WORDS = {
    "title": {"en": "Your kappa for the three comparisons of the slider", "nl": "Jouw kappa voor de drie vergelijkingen van de slider"},
    "bars": {"en": "kappa", "nl": "kappa"},
    "right": {"en": "Your function gives {values}. Under guidelines 2, A and B use most of the room above chance; C agrees with A on 55% of the messages, but no more often than chance, so C's kappa is 0.",
              "nl": "Je functie geeft {values}. Onder richtlijnen 2 gebruiken A en B het grootste deel van de ruimte boven toeval; C is het met A over 55% van de berichten eens, maar niet vaker dan toeval, dus de kappa van C is 0."},
    "said": {"en": "Your function gives {values}; Cohen's kappa of the three comparisons is {want}, the dashed lines.",
             "nl": "Je functie geeft {values}; de Cohen's kappa van de drie vergelijkingen is {want}, de gestreepte lijnen."},
}


def _ka_picture(seen_) -> str:
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""

    def one(k):
        a, b, _ = CASES_SLIDER[k]
        return _quiet(fn, agreement_of(a, b), spam_share(a), spam_share(b))

    mine = _three(one)
    if mine is None:
        return ""
    want = [kappa_of(*CASES_SLIDER[k][:2]) for k in (1, 2, 3)]
    kw = {k: v[lang] for k, v in _KA_WORDS.items()}
    w = {k: v[lang] for k, v in _THREE_WORDS.items()}
    values = ", ".join(number(v, lang) for v in mine)
    if all(abs(m - r) < 1e-9 for m, r in zip(mine, want)):
        text = kw["right"].format(values=values)
    else:
        text = kw["said"].format(values=values, want=", ".join(number(v, lang) for v in want))
    p = _three_panel(mine, want, kw["bars"], w, lang)
    return yours.figure(kw["title"], chart.figure([p], alt=text, width=460, height=230) + _key(text), alt=text)


EXERCISES["kappa"] = FunctionBox(
    name="kappa",
    picture=_ka_picture,
    function="kappa",
    start=_KA_START,
    solution=_KA,
    show=_shown_number,
    compare=_same_number,
    visible=[_KA_VISIBLE],
    hidden=_KA_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("rounded", matches=lambda got, text: "round(" in text.split("return")[-1],
                en="`{call}` gives {got}: your function rounds kappa. Give kappa itself, without `round`; the lesson rounds only to show it.",
                nl="`{call}` geeft {got}: je functie rondt kappa af. Geef kappa zelf, zonder `round`; de les rondt alleen af om het te laten zien."),
        Mistake("ok-is-rest", syntax=r"1\s*-\s*a_spam\s*\*\s*b_spam",
                en="Your `chance` adds `1 - a_spam * b_spam`, everything except both spam, so it is always 1 and `1 - chance` is 0. Everything except both spam also holds the messages on which they differ. Both ok by chance is the two shares of ok multiplied, as for spam.",
                nl="Je `chance` telt `1 - a_spam * b_spam` op, alles behalve allebei spam, dus die is altijd 1 en `1 - chance` is 0. Alles behalve allebei spam bevat ook de berichten waarover ze verschillen. Allebei ok door toeval is de twee delen ok vermenigvuldigd, zoals bij spam."),
        Mistake("ok-plus", wrong=_ka("a_spam * b_spam + (1 - a_spam) + (1 - b_spam)"),
                en="`{call}` gives {got}: your `chance` adds the two shares of ok, so it comes out above 1, and a probability never does. Both say ok by chance as often as the two shares of ok multiplied, as for spam.",
                nl="`{call}` geeft {got}: je `chance` telt de twee delen ok op, dus die komt boven 1 uit, en een kans doet dat nooit. Allebei ok door toeval gebeurt zo vaak als de twee delen ok vermenigvuldigd, zoals bij spam."),
        Mistake("sign", wrong=_ka(back="(chance - agree) / (1 - chance)"),
                en="`{call}` gives {got}: the right size, but the sign is turned around. Take chance away from the agreement: `agree - chance`.",
                nl="`{call}` geeft {got}: de goede grootte, maar het teken is omgedraaid. Haal toeval van de overeenstemming af: `agree - chance`."),
        Mistake("order", matches=lambda got, text: re.search(r"return\s*agree\s*-\s*chance\s*/", text) is not None,
                en="`{call}` gives {got}: Python divides before it subtracts, so your line divides only `chance`. Put `agree - chance` and `1 - chance` each between parentheses.",
                nl="`{call}` geeft {got}: Python deelt vóór het aftrekt, dus je regel deelt alleen `chance`. Zet `agree - chance` en `1 - chance` elk tussen haakjes."),
        Mistake("spam-only", wrong=_ka("a_spam * b_spam"),
                en="`{call}` gives {got}: your `chance` counts only both saying spam by chance. Both saying ok by chance is agreeing too: add how often that happens.",
                nl="`{call}` geeft {got}: je `chance` telt alleen dat ze allebei toevallig spam zeggen. Allebei toevallig ok zeggen is het ook eens zijn: tel op hoe vaak dat gebeurt."),
        Mistake("half", wrong=_ka("0.5"),
                en="`{call}` gives {got}: your `chance` is always 0.5, as if both said spam half of the time. Chance depends on how often each of them says spam: `a_spam` and `b_spam`.",
                nl="`{call}` geeft {got}: je `chance` is altijd 0,5, alsof ze allebei de helft van de keren spam zeggen. Toeval hangt af van hoe vaak elk van hen spam zegt: `a_spam` en `b_spam`."),
        Mistake("no-room", wrong=_ka(back="agree - chance"),
                en="`{call}` gives {got}: that is how far the agreement lies above chance, but kappa says what part of the room above chance they use. Divide by that room, `1 - chance`.",
                nl="`{call}` geeft {got}: zoveel ligt de overeenstemming boven toeval, maar kappa zegt welk deel van de ruimte boven toeval ze gebruiken. Deel door die ruimte, `1 - chance`."),
        Mistake("over-chance", wrong=_ka(back="(agree - chance) / chance"),
                en="`{call}` gives {got}: your function divides by `chance`. The room above chance is `1 - chance`: from chance up to always agreeing.",
                nl="`{call}` geeft {got}: je functie deelt door `chance`. De ruimte boven toeval is `1 - chance`: van toeval tot altijd eens."),
        Mistake("ratio", wrong=_ka(back="agree / chance"),
                en="`{call}` gives {got}: agreement divided by chance says how many times as often they agree, not what part of the room above chance they use. Kappa compares how far the agreement lies above chance with how far it could lie above chance.",
                nl="`{call}` geeft {got}: overeenstemming gedeeld door toeval zegt hoeveel keer zo vaak ze het eens zijn, niet welk deel van de ruimte boven toeval ze gebruiken. Kappa vergelijkt hoe ver de overeenstemming boven toeval ligt met hoe ver die erboven kan liggen."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: how often the two would agree by chance, and kappa.",
                  "nl": "Vervang de twee `...`: hoe vaak de twee het door toeval eens zouden zijn, en kappa."},
        "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(agree, a_spam, b_spam)`.",
                      "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(agree, a_spam, b_spam)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: chance is both spam plus both ok by chance, and kappa is the agreement above chance divided by the room above chance.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: toeval is allebei spam plus allebei ok door toeval, en kappa is de overeenstemming boven toeval gedeeld door de ruimte boven toeval."},
        "almost": {"en": "Right for guidelines 1. But `{call}` gives {got}, and it should give {want}. Does your function use its three values, and no fixed number such as 0.485?",
                   "nl": "Goed voor richtlijnen 1. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de drie waarden die ze krijgt, en geen vast getal zoals 0,485?"},
        "correct": {"en": "Your function is right in all {n} cases. Under guidelines 1, A and B agree on 0.7 of the messages and would agree on 0.485 by chance: kappa 0.42.",
                    "nl": "Je functie klopt in alle {n} gevallen. Onder richtlijnen 1 zijn A en B het over 0,7 van de 20 berichten eens, en door toeval over 0,485: kappa 0,42."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("agreement", _AG_START, "empty", None),
    ("agreement", _ag("        if a[i] == b[i]:\n            ...\n"), "wrong", "dots-left"),
    ("agreement", _AG, "correct", None),
    ("agreement", _ag("        if b[i] == a[i]:\n            same = same + 1\n"), "correct", None),
    ("agreement", _ag("        if a[i] != b[i]:\n            same = same + 1\n"), "wrong", "disagree"),
    ("agreement", _ag("        if a[i] == \"spam\":\n            if b[i] == \"spam\":\n                same = same + 1\n"), "wrong", "spam-both"),
    ("agreement", _ag("        same = same + 1\n"), "wrong", "always"),
    ("agreement", _ag("        if a == b:\n            same = same + 1\n"), "wrong", "whole-lists"),
    ("kappa", _KA_START, "empty", None),
    ("kappa", _ka(back="..."), "wrong", "dots-left"),
    ("kappa", _KA, "correct", None),
    ("kappa", _ka("a_spam * b_spam + (1 - a_spam) * (1 - b_spam)", "1 - (1 - agree) / (1 - chance)"), "correct", None),
    ("kappa", _ka("a_spam * b_spam"), "wrong", "spam-only"),
    ("kappa", _ka("0.5"), "wrong", "half"),
    ("kappa", _ka(back="agree - chance"), "wrong", "no-room"),
    ("kappa", _ka(back="(agree - chance) / chance"), "wrong", "over-chance"),
    ("kappa", _ka(back="agree / chance"), "wrong", "ratio"),
    ("kappa", _ka(back="round((agree - chance) / (1 - chance), 4)"), "wrong", "rounded"),
    ("kappa", _ka("a_spam * b_spam + (1 - a_spam * b_spam)"), "error", "ok-is-rest"),
    ("kappa", _ka("a_spam * b_spam + (1 - a_spam) + (1 - b_spam)"), "wrong", "ok-plus"),
    ("kappa", _ka(back="(chance - agree) / (1 - chance)"), "wrong", "sign"),
    ("kappa", _ka(back="agree - chance / (1 - chance)"), "wrong", "order"),
    ("kappa", _ka(back="agree - chance / 1 - chance"), "wrong", "order"),
]
