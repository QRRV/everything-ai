# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-association-rules: the 19,773 orders of a British online shop (data/online-retail-baskets
.csv.gz), Apriori on them for every min support of the slider, the rules of the frequent pairs with their confidence
and lift, and the functions support and rule."""
import functools
import math

from everything_ai import chart, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/online-retail-baskets.csv.gz"   # see data/README.md
SUPPORTS = [0.01, 0.015, 0.02, 0.03, 0.04]   # the stops of the slider
GREEN = "GREEN REGENCY TEACUP AND SAUCER"
ROSES = "ROSES REGENCY TEACUP AND SAUCER"
PINK = "PINK REGENCY TEACUP AND SAUCER"
HEART = "WHITE HANGING HEART T-LIGHT HOLDER"
JUMBO = "JUMBO BAG RED RETROSPOT"


# --- the orders ------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def orders() -> dict:
    """The orders as a sparse table of 0 and 1, an order per row and a product per column, with the names of the
    products, the number of orders and how many orders hold each product."""
    import numpy as np
    import pandas as pd
    from scipy import sparse
    df = pd.read_csv(paths.ROOT / DATA, dtype=str, keep_default_na=False)
    row = pd.factorize(df["invoice"])[0]
    col, names = pd.factorize(df["product"])
    M = sparse.csr_matrix((np.ones(len(df), dtype=np.float32), (row, col)))
    count = np.asarray(M.sum(axis=0)).ravel().astype(int)
    count.flags.writeable = False
    return {"M": M, "names": list(names), "n": M.shape[0], "count": count, "index": {p: i for i, p in enumerate(names)}}


def baskets() -> list:
    """Every order as a set of products, as the lesson makes them."""
    import pandas as pd
    df = pd.read_csv(paths.ROOT / DATA)
    return list(df.groupby("invoice")["product"].agg(set))


def counts(a: str, b: str) -> tuple:
    """How many orders hold a, b, and both."""
    o = orders()
    i, j = o["index"][a], o["index"][b]
    both = int(o["M"][:, i].multiply(o["M"][:, j]).sum())
    return int(o["count"][i]), int(o["count"][j]), both


@functools.lru_cache(maxsize=None)
def apriori(min_support: float) -> dict:
    """Apriori up to sets of three: the frequent products, how many pairs of them it counts and which are frequent,
    how many sets of three it counts (those whose three pairs are all frequent) and how many are frequent."""
    import numpy as np
    o = orders()
    n, need = o["n"], min_support * o["n"]
    f1 = np.flatnonzero(o["count"] >= need)
    dense = o["M"][:, f1].toarray() > 0
    together = dense.T.astype(np.float32) @ dense.astype(np.float32)
    k = len(f1)
    pairs = [(i, j, int(together[i, j])) for i in range(k) for j in range(i + 1, k) if together[i, j] >= need]
    frequent = {(i, j) for i, j, _ in pairs}
    triples = 0
    kept = []
    for i, j, _ in pairs:
        for m in range(j + 1, k):
            if (i, m) in frequent and (j, m) in frequent:
                triples += 1
                c = int(np.count_nonzero(dense[:, i] & dense[:, j] & dense[:, m]))
                if c >= need:
                    kept.append((f1[i], f1[j], f1[m], c))
    names = o["names"]
    return {"singles": int(k), "pairs_counted": k * (k - 1) // 2,
            "pairs": [(names[f1[i]], names[f1[j]], c) for i, j, c in pairs],
            "triples_counted": triples, "triples": [(names[a], names[b], names[c], m) for a, b, c, m in kept]}


def rules(min_support: float) -> list:
    """Both rules of every frequent pair: (confidence, lift, A, B)."""
    o = orders()
    n, count, index = o["n"], o["count"], o["index"]
    out = []
    for a, b, c in apriori(min_support)["pairs"]:
        for x, y in ((a, b), (b, a)):
            conf = c / count[index[x]]
            out.append((float(conf), float(conf / (count[index[y]] / n)), x, y))
    return out


def all_pairs() -> int:
    n = len(orders()["names"])
    return n * (n - 1) // 2


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def short(x: float, lang: str) -> str:
    """A min support as short as it can be written: 0.01, 0.015; in Dutch with a decimal comma."""
    text = f"{x:g}"
    return text.replace(".", ",") if lang == "nl" else text


def said(min_support: float, lang: str) -> str:
    """The sentence under the picture of the slider."""
    a, r = apriori(min_support), rules(min_support)
    lo, hi = min(x[1] for x in r), max(x[1] for x in r)
    n = lambda x: number(x, lang)
    if lang == "nl":
        lift = f"allebei met een lift van {number(lo, lang, 2)}" if len(r) == 2 and round(lo, 2) == round(hi, 2) else \
            f"met een lift van {number(lo, lang, 2)} tot {number(hi, lang, 2)}"
        return (f"Min support {short(min_support, lang)}: {n(a['singles'])} frequente producten. Apriori telt {n(a['pairs_counted'])} paren "
                f"ervan en houdt er {n(len(a['pairs']))}; het telt {n(a['triples_counted'])} sets van drie en houdt er {n(len(a['triples']))}. "
                f"{n(len(r))} regels, {lift}.")
    lift = f"both with a lift of {number(lo, lang, 2)}" if len(r) == 2 and round(lo, 2) == round(hi, 2) else \
        f"with a lift from {number(lo, lang, 2)} to {number(hi, lang, 2)}"
    return (f"Min support {short(min_support, lang)}: {n(a['singles'])} frequent products. Apriori counts {n(a['pairs_counted'])} pairs "
            f"of them and keeps {n(len(a['pairs']))}; it counts {n(a['triples_counted'])} sets of three and keeps {n(len(a['triples']))}. "
            f"{n(len(r))} rules, {lift}.")


# --- the picture of the slider ---------------------------------------------------------------------------------------

def picture(min_support: float, labels: dict) -> str:
    """Left: for single products, pairs and sets of three, how many sets Apriori counts (open bars) and how many are
    frequent (filled bars), on a log scale, with a dashed mark at all 7,085,730 pairs. Right: every rule of the
    frequent pairs, its confidence against its lift, also on a log scale, from a lift of 1 to 100."""
    o = orders()
    a = apriori(min_support)
    counted = [len(o["names"]), a["pairs_counted"], a["triples_counted"]]
    kept = [a["singles"], len(a["pairs"]), len(a["triples"])]
    left = chart.Panel((-0.6, 2.6), (1, 2e7), title=labels["left"], ylabel=labels["sets"], xticks=[0, 1, 2],
                       xtick_labels=labels["sizes"], ylog=True)
    for i, (c, k) in enumerate(zip(counted, kept)):
        if c > 0:
            left.rect(i - 0.36, 1, i - 0.02, c, role="ink-3")
        if k > 0:
            left.rect(i + 0.02, 1, i + 0.36, k, role="data", solid=True)
        left.text(i + 0.19, max(k, 1) * 2.2, number(k, labels["lang"]), "ink", halo=True)
    left.line([0.64, 1.36], [all_pairs()] * 2, role="ink-3", width=1.5, dashed=True)
    r = rules(min_support)
    right = chart.Panel((0, 1), (1, 100), title=labels["right"], xlabel="confidence", ylabel="lift",
                        xticks=[0, 0.25, 0.5, 0.75, 1], ylog=True)
    if r:
        right.points([x[0] for x in r], [x[1] for x in r], "data", r=2.5)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=270)


# The lesson is done when these are correct. "apriori" is the choice question in the lesson.
REQUIRED = ["support", "rule", "apriori"]


class _Val:
    """A test value that the check's messages write by its name, such as `orders`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _shown(v) -> str:
    if isinstance(v, float):
        return str(round(v, 3))
    if isinstance(v, tuple) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return "(" + ", ".join(str(round(float(x), 3)) for x in v) + ")"
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


# --- exercise 1: support ---------------------------------------------------------------------------------------------

_ORDERS = _Val("orders", [{"GREEN TEACUP", "ROSES TEACUP"}, {"GREEN TEACUP"}, {"ROSES TEACUP", "CAKESTAND"},
                          {"GREEN TEACUP", "ROSES TEACUP", "CAKESTAND"}, {"BUNTING"}, {"ROSES TEACUP"}])
_BOTH = _Val("both", {"GREEN TEACUP", "ROSES TEACUP"})
_ONE = _Val("one", {"BUNTING"})
_NONE = _Val("never", {"BUNTING", "CAKESTAND"})
_THREE = _Val("three", {"GREEN TEACUP", "ROSES TEACUP", "CAKESTAND"})
_EIGHT = _Val("eight", [{"CAKESTAND", "ROSES TEACUP"}, {"CAKESTAND"}, {"CAKESTAND", "ROSES TEACUP", "BUNTING"},
                        {"BUNTING"}, {"ROSES TEACUP"}, {"CAKESTAND", "ROSES TEACUP"}, {"GREEN TEACUP"}, {"CAKESTAND"}])
_STAND = _Val("stand", {"CAKESTAND", "ROSES TEACUP"})

_S_HEAD = "def support(orders, items):\n    count = 0\n    for order in orders:\n"


def _s(test: str = "items <= order", back: str = "count / len(orders)") -> str:
    return _S_HEAD + f"        if {test}:\n            count = count + 1\n    return {back}\n"


_S_START = _s("...", "...")
_S = _s()

EXERCISES = {
    "support": FunctionBox(
        name="support",
        picture=yours.steps(call=0),
        function="support",
        start=_S_START,
        solution=_S,
        show=_shown,
        visible=[(_ORDERS, _BOTH)],
        hidden=[(_ORDERS, _ONE), (_ORDERS, _NONE), (_ORDERS, _THREE), (_EIGHT, _STAND)],
        first=[_DOTS],
        mistakes=[
            Mistake("reversed", wrong=_s(test="order <= items"),
                    en="`{call}` gives {got}, and it should give {want}. Your test asks whether every product of the order is in `items`; turn it around: is every product of `items` in the order?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je test vraagt of elk product van de bestelling in `items` zit; draai hem om: zit elk product van `items` in de bestelling?"),
            Mistake("count", wrong=_s(back="count"),
                    en="`{call}` gives {got}: that is how many orders hold the set. The support is a share: divide by the number of orders.",
                    nl="`{call}` geeft {got}: zoveel bestellingen bevatten de set. De support is een deel: deel door het aantal bestellingen."),
            Mistake("overlap", wrong=_s(test="items & order"),
                    en="`{call}` gives {got}, and it should give {want}. Your test counts an order as soon as it holds one product of `items`; it should hold every one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je test telt een bestelling al als er één product van `items` in zit; er moet elk product in zitten."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: when an order counts, and the share that the function gives back.",
                      "nl": "Vervang de twee `...`: wanneer een bestelling meetelt, en het deel dat de functie teruggeeft."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(orders, items)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(orders, items)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does an order count when every product of `items` is in it, and do you divide by the number of orders?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt een bestelling mee als elk product van `items` erin zit, en deel je door het aantal bestellingen?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `items` and `orders`, and work for any number of orders?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `items` en `orders`, en werkt hij voor elk aantal bestellingen?"},
            "correct": {"en": "Your function is right in all {n} cases. The GREEN TEACUP and the ROSES TEACUP are together in 2 of the 6 orders: a support of 0.333. A set in no order has a support of 0.",
                        "nl": "Je functie klopt in alle {n} gevallen. De GREEN TEACUP en de ROSES TEACUP zitten samen in 2 van de 6 bestellingen: een support van 0,333. Een set in geen enkele bestelling heeft support 0."},
        },
    ),
}


# --- exercise 2: confidence and lift ---------------------------------------------------------------------------------

_R_HEAD = "def rule(n_a, n_b, n_both, n):\n"


def _r(conf: str = "n_both / n_a", lift: str = "confidence / (n_b / n)") -> str:
    return _R_HEAD + f"    confidence = {conf}\n    lift = {lift}\n    return confidence, lift\n"


_R_START = _r("...", "...")
_R = _r()

EXERCISES["rule"] = FunctionBox(
    name="rule",
    picture=yours.steps(call=0),
    function="rule",
    start=_R_START,
    solution=_R,
    show=_shown,
    visible=[(1013, 1065, 767, 19773)],
    hidden=[(10, 20, 5, 100), (50, 50, 10, 100), (40, 25, 10, 100), (20, 20, 20, 100)],
    first=[_DOTS],
    mistakes=[
        Mistake("other-side", wrong=_r(conf="n_both / n_b"),
                en="`{call}` gives {got}, and it should give {want}. Your confidence is the share of the orders with B that hold A. The confidence of A → B looks at the orders with A: `n_both / n_a`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Jouw confidence is het deel van de bestellingen met B dat A bevat. De confidence van A → B kijkt naar de bestellingen met A: `n_both / n_a`."),
        Mistake("support", wrong=_r(conf="n_both / n"),
                en="`{call}` gives {got}, and it should give {want}. `n_both / n` is the support of A and B together. The confidence divides by the orders with A, not by all orders.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `n_both / n` is de support van A en B samen. De confidence deelt door de bestellingen met A, niet door alle bestellingen."),
        Mistake("count-b", wrong=_r(lift="confidence / n_b"),
                en="`{call}` gives {got}, and it should give {want}. Divide the confidence by the support of B, a share: `n_b / n`, not the count `n_b`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Deel de confidence door de support van B, een deel: `n_b / n`, niet het aantal `n_b`."),
        Mistake("support-a", wrong=_r(lift="confidence / (n_a / n)"),
                en="`{call}` gives {got}, and it should give {want}. The lift compares the confidence with the support of B, the product the rule is about: `n_b / n`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. De lift vergelijkt de confidence met de support van B, het product waar de regel over gaat: `n_b / n`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the confidence of A → B, and its lift.",
                  "nl": "Vervang de twee `...`: de confidence van A → B, en de lift ervan."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(n_a, n_b, n_both, n)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(n_a, n_b, n_both, n)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is the confidence the share of the orders with A that hold B too, and the lift the confidence divided by the support of B?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Is de confidence het deel van de bestellingen met A dat ook B bevat, en de lift de confidence gedeeld door de support van B?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use the four counts it gets?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de vier aantallen die hij krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases. GREEN → ROSES: a confidence of 0.757 and a lift of 14.06. With 40, 25 and 10 of 100 orders the lift is exactly 1: B is in a quarter of all orders, and in a quarter of the orders with A, so A makes no difference.",
                    "nl": "Je functie klopt in alle {n} gevallen. GREEN → ROSES: een confidence van 0,757 en een lift van 14,06. Met 40, 25 en 10 van de 100 bestellingen is de lift precies 1: B zit in een kwart van alle bestellingen, en in een kwart van de bestellingen met A, dus A maakt geen verschil."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("support", _S_START, "empty", None),
    ("support", _s(back="..."), "wrong", "dots-left"),
    ("support", _S, "correct", None),
    ("support", _s(test="items.issubset(order)"), "correct", None),
    ("support", _s(test="len(items - order) == 0"), "correct", None),
    ("support", _s(test="order <= items"), "wrong", "reversed"),
    ("support", _s(back="count"), "wrong", "count"),
    ("support", _s(test="items & order"), "wrong", "overlap"),
    ("support", _s(back="count / 6"), "almost", None),
    ("rule", _R_START, "empty", None),
    ("rule", _r(lift="..."), "wrong", "dots-left"),
    ("rule", _R, "correct", None),
    ("rule", _r(lift="n_both * n / (n_a * n_b)"), "correct", None),
    ("rule", _r(conf="n_both / n_b"), "wrong", "other-side"),
    ("rule", _r(conf="n_both / n"), "wrong", "support"),
    ("rule", _r(lift="confidence / n_b"), "wrong", "count-b"),
    ("rule", _r(lift="confidence / (n_a / n)"), "wrong", "support-a"),
    ("rule", _r(conf="767 / 1013", lift="14.057526"), "almost", None),
]
