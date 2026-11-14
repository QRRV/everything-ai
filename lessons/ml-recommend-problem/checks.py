# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-recommend-problem: the orders of a British online shop whose customer is known
(data/online-retail-baskets.csv.gz), split by time so that every customer's last order is hidden, the popularity
baseline and picking at random for every k of the slider, and the functions popular and hit_rate."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.shop import popular_for, rows, split  # noqa: F401
from everything_ai.check import FunctionBox, Mistake

KS = [1, 5, 10, 20, 50]                      # the stops of the slider
EXAMPLE = 14905.0                            # the customer of the worked example and of exercise 1


# --- the orders, split by time: everything_ai.shop, shared by the lessons on recommending ---------------------------

@functools.lru_cache(maxsize=1)
def firsts() -> list:
    """For every test customer, where the first of their hidden products stands in the popularity list made for them:
    0 for the top. A product nobody bought before counts as never found."""
    import bisect
    s = split()
    rank = {p: i for i, p in enumerate(s["counts"].index)}
    out = []
    for c, h in s["hidden"].items():
        before = sorted(rank[p] for p in s["seen"][c])
        places = [rank[p] - bisect.bisect_left(before, rank[p]) for p in h if p in rank]
        out.append(min(places) if places else math.inf)
    return out


def hits(k: int) -> int:
    """How many test customers find at least one of their hidden products in the k most popular they have not bought."""
    return sum(1 for f in firsts() if f < k)


def hit_popular(k: int) -> float:
    return hits(k) / len(firsts())


@functools.lru_cache(maxsize=None)
def hit_random(k: int) -> float:
    """The hit rate of picking k products at random from the ones a customer has not bought, as an expected value:
    for every customer, 1 minus the chance that all k picks miss their hidden products."""
    s = split()
    n = len(s["counts"])
    out = 0.0
    for c, h in s["hidden"].items():
        free = n - len(s["seen"][c])
        found = sum(1 for p in h if p in s["counts"].index)
        out += 1 - math.comb(free - found, k) / math.comb(free, k)
    return out / len(s["hidden"])


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def percent(x: float, lang: str, digits: int = 1) -> str:
    return number(100 * x, lang, digits) + "%"


def said(k: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    n = len(firsts())
    if lang == "nl":
        what = "het populairste product dat" if k == 1 else f"de {k} populairste producten die"
        pick = "1 product" if k == 1 else f"{k} producten"
        return (f"k = {k}: met {what} een klant nog niet kocht, vinden {number(hits(k), lang)} van de "
                f"{number(n, lang)} klanten minstens één product van hun laatste bestelling terug, "
                f"{percent(hit_popular(k), lang)}. Kies je willekeurig {pick}, dan gemiddeld {percent(hit_random(k), lang)}.")
    what = "the most popular product" if k == 1 else f"the {k} most popular products"
    return (f"k = {k}: with {what} a customer has not bought, {number(hits(k), lang)} of the {number(n, lang)} customers "
            f"find at least one product of their last order, {percent(hit_popular(k), lang)}. Pick {k} at random, "
            f"and on average {percent(hit_random(k), lang)}.")


# --- the pictures ----------------------------------------------------------------------------------------------------

def picture(k: int, labels: dict) -> str:
    """The hit rate at every k of the slider: the popularity baseline as dots on a line, picking at random as open dots
    on a dashed line, both on a log scale of k, with the k of the slider marked."""
    p = chart.Panel((0.8, 62), (0, 60), title=labels["title"], xlabel="k", ylabel=labels["ylabel"], xlog=True,
                    xticks=KS, xtick_labels=[str(v) for v in KS], yticks=[0, 20, 40, 60])
    p.line([k, k], [0, 60], role="ink-3", width=0.8, dashed=True)
    pop = [100 * hit_popular(v) for v in KS]
    rnd = [100 * hit_random(v) for v in KS]
    p.line(KS, pop, role="ink", width=2.0)
    p.points(KS, pop, role="ink", r=3.5)
    p.line(KS, rnd, role="ink-3", width=1.4, dashed=True)
    p.points(KS, rnd, role="ink-3", r=3.5, hollow=True)
    p.text(48, pop[-1] + 4, labels["popular"], role="ink-2", anchor="end")
    p.text(48, rnd[-1] + 4, labels["random"], role="ink-2", anchor="end")
    return chart.figure([p], alt=labels["alt"], width=460, height=270)


def matrix(labels: dict) -> str:
    """A corner of the user-item matrix: the first 60 test customers against the 120 products most bought before
    their last order. A filled dot: bought before the last order; an open dot: new in the last order, hidden."""
    s = split()
    top = list(s["counts"].index[:120])
    at = {p: i for i, p in enumerate(top)}
    customers = list(s["hidden"])[:60]
    p = chart.Panel((-1, 120), (-1, 60), title=labels["title"], xlabel=labels["xlabel"], ylabel=labels["ylabel"],
                    xticks=[0, 29, 59, 89, 119], xtick_labels=["1", "30", "60", "90", "120"], yticks=[])
    xs, ys, hx, hy = [], [], [], []
    for row, c in enumerate(customers):
        for prod in s["seen"][c]:
            if prod in at:
                xs.append(at[prod]); ys.append(59 - row)
        for prod in s["hidden"][c]:
            if prod in at:
                hx.append(at[prod]); hy.append(59 - row)
    p.points(xs, ys, role="ink", r=1.7)
    p.points(hx, hy, role="ink", r=2.2, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


def matrix_numbers() -> dict:
    """The dots of the corner of the matrix: how many are filled and how many open, out of 60 × 120 cells."""
    s = split()
    top = set(s["counts"].index[:120])
    customers = list(s["hidden"])[:60]
    filled = sum(len(s["seen"][c] & top) for c in customers)
    open_ = sum(len(s["hidden"][c] & top) for c in customers)
    return {"filled": filled, "open": open_, "cells": 60 * 120}


# The lesson is done when these are correct. "implicit" is the choice question in the lesson.
REQUIRED = ["popular", "hit-rate", "implicit"]


class _Val:
    """A test value that the check's messages write by its name, such as `counts`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _shown(v):
    if isinstance(v, float):
        return str(round(v, 3))
    if isinstance(v, list) and len(v) > 4:
        start = repr(v[:2])[:-1] + ", ...]"
        return {"en": f"a list of {len(v)} products, starting {start}", "nl": f"een lijst van {len(v)} producten, beginnend met {start}"}
    return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _call(seen, case: tuple):
    fn = seen.mine.names.get(seen.exercise.function) if seen.mine is not None else None
    if not callable(fn):
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
    except Exception:
        return None


def _key(text: str) -> str:
    return ('<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.15rem">'
            + text + "</div>")


# --- exercise 1: the most popular products a customer has not bought ------------------------------------------------

def _series(name: str, counts: dict) -> _Val:
    import pandas as pd
    return _Val(name, pd.Series(counts, name="count"))


def _counts() -> _Val:
    return _Val("counts", split()["counts"])


def _seen_of(customer: float) -> _Val:
    return _Val("seen", set(split()["seen"][customer]))


_SHOP = {"CAKESTAND": 9, "HEART HOLDER": 8, "BUNTING": 7, "CAKE CASES": 6, "BIRD ORNAMENT": 5, "JUMBO BAG": 4,
         "CAKE TINS": 3, "LUNCH BAG": 2}

_P_HEAD = "def popular(counts, seen, k):\n    top = []\n    for product in counts.index:\n"


def _p(test: str = "product not in seen and len(top) < k", back: str = "top") -> str:
    return _P_HEAD + f"        if {test}:\n            top.append(product)\n    return {back}\n"


_P_START = _p("...")
_P = _p()


def popular(counts, seen, k):
    """The course's popular: the first k products of counts that are not in seen."""
    return [p for p in counts.index if p not in seen][:k]


_PO_WORDS = {
    "title": {"en": "Your popular(counts, seen, 10)", "nl": "Jouw popular(counts, seen, 10)"},
    "xlabel": {"en": "the 14 most popular products, most bought first", "nl": "de 14 populairste producten, meest gekochte eerst"},
    "ylabel": {"en": "customers who bought it", "nl": "klanten die het kochten"},
    "said": {"en": "The 14 most popular products, as bars as high as the number of customers who bought them. Filled: in your list. ×: customer 14905 bought it already. Your list has {n} products, {bought} of them bought already; the course's list has 10, none bought already.",
             "nl": "De 14 populairste producten, als staven zo hoog als het aantal klanten dat ze kocht. Gevuld: in jouw lijst. ×: klant 14905 kocht het al. Jouw lijst heeft {n} producten, waarvan {bought} al gekocht; de lijst van de course heeft er 10, geen enkele al gekocht."},
    "right": {"en": "The 14 most popular products, as bars as high as the number of customers who bought them. Filled: in your list, the course's. ×: customer 14905 bought it already, so your list skips it and goes on to the 12th product. Two of the ten are in the customer's last order: JUMBO BAG RED RETROSPOT and HEART OF WICKER SMALL.",
              "nl": "De 14 populairste producten, als staven zo hoog als het aantal klanten dat ze kocht. Gevuld: in jouw lijst, die van de course. ×: klant 14905 kocht het al, dus je lijst slaat het over en gaat door tot het 12e product. Twee van de tien zitten in de laatste bestelling van de klant: JUMBO BAG RED RETROSPOT en HEART OF WICKER SMALL."},
}


def _po_picture(seen) -> str:
    got = _call(seen, (_counts(), _seen_of(EXAMPLE), 10))
    if not isinstance(got, (list, tuple)) or not got:
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _PO_WORDS.items()}
    s = split()
    top = list(s["counts"].index[:14])
    bought = s["seen"][EXAMPLE]
    mine = {p for p in got if isinstance(p, str)}
    p = chart.Panel((-0.6, 13.6), (0, 760), xlabel=words["xlabel"], ylabel=words["ylabel"], xticks=[0, 4, 9, 13],
                    xtick_labels=["1", "5", "10", "14"], yticks=[0, 200, 400, 600])
    for i, prod in enumerate(top):
        p.rect(i - 0.36, 0, i + 0.36, float(s["counts"][prod]), role="ink" if prod in mine else "ink-3", solid=prod in mine)
        if prod in bought:
            p.text(i, float(s["counts"][prod]) + 30, "×", role="ink", halo=True)
    right = checking.same_value(list(got), popular(s["counts"], bought, 10))
    text = words["right"] if right else words["said"].format(n=len(got), bought=sum(1 for x in got if x in bought))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=250) + _key(text), alt=text)


EXERCISES = {
    "popular": FunctionBox(
        name="popular",
        picture=_po_picture,
        function="popular",
        start=_P_START,
        solution=_P,
        show=_shown,
        visible=[(_counts(), _seen_of(EXAMPLE), 10)],
        hidden=[(_series("small", _SHOP), _Val("bought", {"CAKESTAND", "BUNTING"}), 3),
                (_series("small", _SHOP), _Val("nothing", set()), 4),
                (_series("small", _SHOP), _Val("most", {"CAKESTAND", "HEART HOLDER", "BUNTING", "CAKE CASES", "BIRD ORNAMENT", "JUMBO BAG"}), 3),
                (_series("small", _SHOP), _Val("other", {"TEAPOT"}), 2)],
        first=[_DOTS],
        mistakes=[
            Mistake("bought", wrong=_p("product in seen and len(top) < k"),
                    en="`{call}` gives {got}: products the customer bought already. Recommend the ones they have not bought: `not in seen`.",
                    nl="`{call}` geeft {got}: producten die de klant al kocht. Raad aan wat de klant nog niet kocht: `not in seen`."),
            Mistake("all", wrong=_p("product not in seen"),
                    en="`{call}` gives {got}: every product the customer has not bought. A page has room for `k`: stop adding when `top` has `k` products.",
                    nl="`{call}` geeft {got}: elk product dat de klant nog niet kocht. Er is plek voor `k`: voeg niets meer toe als `top` `k` producten heeft."),
            Mistake("top", wrong=_p("len(top) < k"),
                    en="`{call}` gives {got}, and it should give {want}. Your list is the top of `counts` for everyone, also the products this customer bought already. Skip those: `product not in seen`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Jouw lijst is de top van `counts` voor iedereen, ook de producten die deze klant al kocht. Sla die over: `product not in seen`."),
            Mistake("one-more", wrong=_p("product not in seen and len(top) <= k"),
                    en="`{call}` gives {got}, one product too many. With `len(top) <= k`, the product is added when `top` already has `k`: use `<`.",
                    nl="`{call}` geeft {got}, één product te veel. Met `len(top) <= k` komt het product er nog bij als `top` er al `k` heeft: gebruik `<`."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: when a product goes into `top`.",
                      "nl": "Vervang de `...`: wanneer een product in `top` komt."},
            "arguments": {"en": "The check calls `{call}`, with three values, and the `def` line of your function has a different number of names between its parentheses. Keep `(counts, seen, k)`.",
                          "nl": "De check roept `{call}` aan, met drie waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(counts, seen, k)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does a product go into `top` only when the customer has not bought it and `top` has fewer than `k`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Komt een product alleen in `top` als de klant het nog niet kocht en `top` er minder dan `k` heeft?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `seen` and `k`, and not the products and numbers of the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `seen` en `k`, en niet de producten en getallen van het voorbeeld?"},
            "correct": {"en": "Your function is right in all {n} cases. Customer 14905 bought the second and the third most popular product already, so their ten run on to the 12th. Two of them are in the last order.",
                        "nl": "Je functie klopt in alle {n} gevallen. Klant 14905 kocht het op één na en het op twee na populairste product al, dus loopt het tiental van deze klant door tot het 12e. Twee ervan zitten in de laatste bestelling."},
        },
    ),
}


# --- exercise 2: the hit rate ----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _recommended() -> _Val:
    s = split()
    return _Val("recommended", {c: popular_for(c, 10) for c in s["hidden"]})


@functools.lru_cache(maxsize=1)
def _hidden() -> _Val:
    return _Val("hidden", {c: set(h) for c, h in split()["hidden"].items()})


_H_HEAD = ("def hit_rate(recommended, hidden):\n    hits = 0\n    for customer, products in recommended.items():\n"
           "        found = False\n        for product in products:\n")


def _h(test: str = "product in hidden[customer]", back: str = "hits / len(recommended)") -> str:
    return (_H_HEAD + f"            if {test}:\n                found = True\n        if found:\n            hits = hits + 1\n"
            f"    return {back}\n")


_H_START = _h("...", "...")
_H = _h()


def hit_rate(recommended, hidden):
    """The course's hit_rate: the share of the customers with at least one recommended product among their hidden ones."""
    return sum(1 for c, ps in recommended.items() if any(p in hidden[c] for p in ps)) / len(recommended)


_TWO = _Val("two", {"ann": ["TEAPOT", "BUNTING"], "bob": ["CAKESTAND", "MUG"]})
_TWO_H = _Val("next_two", {"ann": {"BUNTING", "MUG"}, "bob": {"TEAPOT"}})
_FOUR = _Val("four", {1: ["A", "B", "C"], 2: ["A", "B", "C"], 3: ["A", "D", "E"], 4: ["B", "C", "D"]})
_FOUR_H = _Val("next_four", {1: {"C"}, 2: {"F"}, 3: {"A", "E"}, 4: {"G", "H"}})
_ALL = _Val("all_hit", {"x": ["A"], "y": ["B", "C"]})
_ALL_H = _Val("next_all", {"x": {"A"}, "y": {"C"}})

_HR_WORDS = {
    "title": {"en": "Your hit_rate, customer by customer", "nl": "Jouw hit_rate, klant voor klant"},
    "said": {"en": "The first 100 of the 2,498 customers, a square each: filled when your function, given that one customer, says they found something (1.0), open when it says 0.0. Your hit rate over all customers: {mine}; the course's: {want}.",
             "nl": "De eerste 100 van de 2.498 klanten, een vierkantje elk: gevuld als jouw functie, met alleen die klant, zegt dat die iets terugvond (1,0), open als hij 0,0 zegt. Jouw hit rate over alle klanten: {mine}; die van de course: {want}."},
    "right": {"en": "The first 100 of the 2,498 customers, a square each: filled when at least one of their ten recommended products is in their last order. {few} of these 100; over all 2,498 customers, {hits}: a hit rate of {want}.",
              "nl": "De eerste 100 van de 2.498 klanten, een vierkantje elk: gevuld als minstens één van hun tien aanbevolen producten in hun laatste bestelling zit. {few} van deze 100; over alle 2.498 klanten {hits}: een hit rate van {want}."},
}


def _hr_picture(seen) -> str:
    got = _call(seen, (_recommended(), _hidden()))
    try:
        mine = float(got)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(mine):
        return ""
    lang = seen.lang
    words = {k: v[lang] for k, v in _HR_WORDS.items()}
    rec, hid = _recommended().value, _hidden().value
    customers = list(rec)[:100]
    p = chart.Panel((-0.6, 19.6), (-0.6, 4.6), xticks=[], yticks=[])
    few = 0
    for i, c in enumerate(customers):
        one = _call(seen, (_Val("one", {c: rec[c]}), _Val("next_one", {c: hid[c]})))
        filled = isinstance(one, (int, float)) and not isinstance(one, bool) and one == 1
        few += filled
        x, y = i % 20, 4 - i // 20
        p.rect(x - 0.4, y - 0.4, x + 0.4, y + 0.4, role="ink" if filled else "ink-3", solid=filled)
    want = hit_rate(rec, hid)
    if abs(mine - want) < 1e-9:
        text = words["right"].format(few=few, hits=number(hits(10), lang), want=number(want, lang, 3))
    else:
        text = words["said"].format(mine=number(mine, lang, 3), want=number(want, lang, 3))
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=130) + _key(text), alt=text)


EXERCISES["hit-rate"] = FunctionBox(
    name="hit-rate",
    picture=_hr_picture,
    function="hit_rate",
    start=_H_START,
    solution=_H,
    show=_shown,
    visible=[(_recommended(), _hidden())],
    hidden=[(_TWO, _TWO_H), (_FOUR, _FOUR_H), (_ALL, _ALL_H)],
    first=[_DOTS],
    mistakes=[
        Mistake("count", wrong=_h(back="hits"),
                en="`{call}` gives {got}: how many customers found something. The hit rate is a share: divide by the number of customers.",
                nl="`{call}` geeft {got}: zoveel klanten vonden iets. De hit rate is een deel: deel door het aantal klanten."),
        Mistake("own", wrong=_h(test="product in products"),
                en="`{call}` gives {got}: every customer counts. `products` is the list you recommend, so every product is in it. Look in what the customer bought in their last order: `hidden[customer]`.",
                nl="`{call}` geeft {got}: elke klant telt mee. `products` is de lijst die je aanraadt, dus elk product zit erin. Kijk in wat de klant in de laatste bestelling kocht: `hidden[customer]`."),
        Mistake("keys", wrong=_h(test="product in hidden"),
                en="`{call}` gives {got}: no customer counts. `hidden` is a dictionary with a customer as each key, so a product is never in it. The products of one customer are `hidden[customer]`.",
                nl="`{call}` geeft {got}: geen enkele klant telt mee. `hidden` is een dictionary met een klant als elke key, dus een product zit er nooit in. De producten van één klant zijn `hidden[customer]`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: when a recommended product is a hit, and the share that the function gives back.",
                  "nl": "Vervang de twee `...`: wanneer een aanbevolen product raak is, en het deel dat de functie teruggeeft."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(recommended, hidden)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(recommended, hidden)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does a customer count when at least one recommended product is in `hidden[customer]`, and do you divide by the number of customers?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt een klant mee als minstens één aanbevolen product in `hidden[customer]` zit, en deel je door het aantal klanten?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function work for any number of customers?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk aantal klanten?"},
        "correct": {"en": "Your function is right in all {n} cases. The popularity baseline finds at least one product of the last order for 541 of the 2,498 customers: a hit rate of 0.217. A customer with two hits counts once.",
                    "nl": "Je functie klopt in alle {n} gevallen. De popularity baseline vindt voor 541 van de 2.498 klanten minstens één product van de laatste bestelling: een hit rate van 0,217. Een klant met twee hits telt één keer."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("popular", _P_START, "empty", None),
    ("popular", _P, "correct", None),
    ("popular", _p("not product in seen and len(top) < k"), "correct", None),
    ("popular", _p("product not in seen", "top[:k]"), "correct", None),
    ("popular", _p("product in seen and len(top) < k"), "wrong", "bought"),
    ("popular", _p("product not in seen"), "wrong", "all"),
    ("popular", _p("len(top) < k"), "wrong", "top"),
    ("popular", _p("product not in seen and len(top) <= k"), "wrong", "one-more"),
    ("popular", _p("product not in seen and len(top) < 10"), "almost", None),
    ("hit-rate", _H_START, "empty", None),
    ("hit-rate", _h(back="..."), "wrong", "dots-left"),
    ("hit-rate", _H, "correct", None),
    ("hit-rate", _h(back="hits / len(hidden)"), "correct", None),
    ("hit-rate", _h(back="hits"), "wrong", "count"),
    ("hit-rate", _h(test="product in products"), "wrong", "own"),
    ("hit-rate", _h(test="product in hidden"), "wrong", "keys"),
    ("hit-rate", _h(back="hits / 2498"), "almost", None),
]
