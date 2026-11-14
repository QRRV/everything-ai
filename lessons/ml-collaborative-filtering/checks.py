# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-collaborative-filtering: the user-item matrix of the orders before every customer's last,
how alike two products are by who bought them, item-item collaborative filtering for every test customer, and how it
compares with content-based filtering and the popularity baseline; and the functions similar_to and cf_scores."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import check as checking, yours
from everything_ai.shop import (bought_columns, collaborative_found, collaborative_for, collaborative_scores,  # noqa: F401
                                content_found, found, item_similarities, matrix, popular_for, rows, split)
from everything_ai.check import FunctionBox, Mistake

EXAMPLE = 13960.0                      # the customer of exercise 2, as in the previous lesson
PRODUCT = "HEART OF WICKER SMALL"      # the product of the worked example, the picture and exercise 1


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


@functools.lru_cache(maxsize=1)
def names() -> list:
    """The products, the most popular first: the columns of the matrix."""
    return list(split()["counts"].index)


def column(product: str) -> int:
    return names().index(product)


@functools.lru_cache(maxsize=1)
def compared() -> dict:
    """At k = 10, how many test customers each recommender finds something for, and at least one of the three."""
    cf, cb, po = collaborative_found(10), content_found(10), found(popular_for, 10)
    return {"n": len(cf), "collaborative": sum(cf.values()), "content": sum(cb.values()), "popular": sum(po.values()),
            "any": sum(1 for c in cf if cf[c] or cb[c] or po[c])}


@functools.lru_cache(maxsize=1)
def counts_found() -> int:
    """The same recommender, but adding up how many customers bought both instead of the cosine similarities: for how
    many test customers one of the ten is hidden."""
    import numpy as np
    s = split()
    R = matrix()
    together = R.T @ R
    hits = 0
    for c, h in s["hidden"].items():
        cols = bought_columns(c)
        score = together[cols].sum(axis=0)
        score[cols] = -np.inf
        hits += any(names()[i] in h for i in np.argsort(-score, kind="stable")[:10])
    return hits


def pairs(product: str, n: int = 5) -> dict:
    """The n other products bought together with `product` by the most customers, with how many, and the n most alike
    by cosine similarity, with the similarity."""
    import numpy as np
    R = matrix()
    j = column(product)
    together = R.T @ R[:, j]
    alike = item_similarities()[j]
    most = [int(i) for i in np.argsort(-together, kind="stable") if i != j][:n]
    near = [int(i) for i in np.argsort(-alike, kind="stable") if i != j][:n]
    return {"together": [(names()[i], int(together[i]), int(R[:, i].sum())) for i in most],
            "alike": [(names()[i], float(alike[i]), int(R[:, i].sum())) for i in near]}


def pairs_picture(product: str, labels: dict, lang: str) -> str:
    """Two lists side by side: the products bought together with `product` by the most customers, and the products most
    alike by cosine similarity; with each, how many customers bought it."""
    p = pairs(product)

    def column_html(title, items, shown):
        out = [f'<div style="font-weight:600;color:var(--ea-ink);margin-bottom:0.2rem">{title}</div>']
        for i, (name, value, buyers) in enumerate(items):
            out.append(f'<div style="display:grid;grid-template-columns:1.3rem minmax(0,1fr) auto;gap:0.4rem;'
                       f'align-items:baseline;padding:0.1rem 0"><span style="color:var(--ea-ink-3)">{i + 1}</span>'
                       f'<span style="color:var(--ea-ink);overflow-wrap:anywhere">{name}<span style="color:var(--ea-ink-3)">'
                       f' · {labels["buyers"].format(n=number(buyers, lang))}</span></span>'
                       f'<span style="color:var(--ea-ink-2)">{shown(value)}</span></div>')
        return '<div style="min-width:0">' + "".join(out) + "</div>"

    left = column_html(labels["together"], p["together"], lambda v: number(v, lang))
    right = column_html(labels["alike"], p["alike"], lambda v: number(v, lang, 3))
    return (f'<figure class="ea-figure" role="img" aria-label="{labels["alt"]}" style="margin:0.4rem 0 0.9rem;max-width:720px">'
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:0.8rem 1.6rem;'
            f'font-family:var(--ea-font-ui);font-size:0.85rem">{left}{right}</div></figure>')


# The lesson is done when these are correct. "new-customer" is the choice question in the lesson.
REQUIRED = ["similar-to", "cf-scores", "new-customer"]


class _Val:
    """A test value that the check's messages write by its name, such as `R`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


def _shown(v):
    import numpy as np
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 6 and np.issubdtype(v.dtype, np.number):
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        return {"en": f"an array of {v.size:,} numbers from {number(lo, 'en', 3)} to {number(hi, 'en', 3)}",
                "nl": f"een array van {number(v.size, 'nl')} getallen van {number(lo, 'nl', 3)} tot {number(hi, 'nl', 3)}"}
    if isinstance(v, np.ndarray):
        return str([round(float(x), 3) for x in v.ravel()]) if np.issubdtype(v.dtype, np.number) else repr(v.tolist())
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


def _rows(rows: list, lang: str, top: float) -> str:
    """A list of products, one row each: a square, filled when the product is in the customer's last order and open
    when it is not; the name; a bar as long as its score; and the score."""
    out = []
    for name, score, mark in rows:
        sign = "■" if mark == "hit" else "□"
        tone = "var(--ea-ink)" if mark else "var(--ea-ink-3)"
        share = max(0.0, min(1.0, score / top)) if top > 0 and math.isfinite(score) else 0.0
        look = "background:var(--ea-ink)" if mark == "hit" else "border:1px solid var(--ea-ink-3)"
        out.append(f'<span style="color:{tone};font-size:1rem;line-height:1">{sign}</span>'
                   f'<span style="color:var(--ea-ink);overflow-wrap:anywhere">{name}</span>'
                   f'<span style="display:block;height:0.7rem;width:{100 * share:.1f}%;{look};box-sizing:border-box"></span>'
                   f'<span style="color:var(--ea-ink-2);text-align:right">{number(score, lang, 3)}</span>')
    return ('<div style="display:grid;grid-template-columns:1.1rem minmax(0,1fr) 6rem 3.6rem;gap:0.2rem 0.6rem;'
            'align-items:center;font-family:var(--ea-font-ui);font-size:0.85rem;max-width:460px;margin:0.3rem 0">'
            + "".join(out) + "</div>")


def _array(name: str, rows_) -> _Val:
    import numpy as np
    return _Val(name, np.array(rows_, dtype=float))


@functools.lru_cache(maxsize=1)
def _R() -> _Val:
    return _Val("R", matrix())


# --- exercise 1: how alike every product is to one product ----------------------------------------------------------

_S_HEAD = ("import numpy as np\n\n\ndef similar_to(R, j):\n    together = R.T @ R[:, j]\n"
           "    lengths = np.linalg.norm(R, axis=0)\n")


def _s(back: str = "together / (lengths * lengths[j])") -> str:
    return _S_HEAD + f"    return {back}\n"


_S_START = _s("...")
_S = _s()


def similar_to(R, j):
    """The course's similar_to: the cosine similarity of column j of R with every column."""
    import numpy as np
    lengths = np.linalg.norm(R, axis=0)
    return R.T @ R[:, j] / (lengths * lengths[j])


_SI_WORDS = {
    "title": {"en": "Your similar_to(R, j) for HEART OF WICKER SMALL", "nl": "Jouw similar_to(R, j) voor HEART OF WICKER SMALL"},
    "said": {"en": "The six other products your scores put first, as bars as long as your score. ■: the product is among the six most alike by cosine similarity; □: not. {same} of your six are. Your highest score is {top}; a cosine similarity is at most 1.",
             "nl": "De zes andere producten die jouw scores bovenaan zetten, als staven zo lang als jouw score. ■: het product hoort bij de zes die er volgens de cosine similarity het meest op lijken; □: niet. {same} van jouw zes doen dat. Jouw hoogste score is {top}; een cosine similarity is hoogstens 1."},
    "right": {"en": "The six products most alike HEART OF WICKER SMALL by who bought them, as bars as long as their cosine similarity. HEART OF WICKER LARGE comes first: 254 of its 368 buyers bought the small one too. WHITE HANGING HEART T-LIGHT HOLDER was bought together with it more often than any product but the large heart, by 216 customers, but it has 642 buyers: 0.414.",
              "nl": "De zes producten die volgens wie ze kocht het meest op HEART OF WICKER SMALL lijken, als staven zo lang als hun cosine similarity. HEART OF WICKER LARGE staat bovenaan: 254 van zijn 368 kopers kochten ook de kleine. WHITE HANGING HEART T-LIGHT HOLDER werd vaker samen gekocht dan elk ander product behalve het grote hartje, door 216 klanten, maar heeft 642 kopers: 0,414."},
}


def _si_picture(seen) -> str:
    import numpy as np
    j = column(PRODUCT)
    got = _call(seen, (_R(), j))
    try:
        mine = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return ""
    if mine.shape != (len(names()),) or not np.isfinite(mine).any():
        return ""
    mine = np.where(np.isfinite(mine), mine, -np.inf)
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _SI_WORDS.items()}
    six = [int(i) for i in np.argsort(-mine, kind="stable") if i != j][:6]
    near = {int(i) for i in [i for i in np.argsort(-item_similarities()[j], kind="stable") if i != j][:6]}
    top = max(float(mine[i]) for i in six)
    rows_ = [(names()[i], float(mine[i]), "hit" if i in near else "") for i in six]
    if checking.same_value(mine, similar_to(matrix(), j)):
        text = words_["right"]
    else:
        text = words_["said"].format(same=sum(1 for i in six if i in near), top=number(top, lang, 3))
    return yours.figure(words_["title"], _rows(rows_, lang, max(top, 1.0)) + _key(text), alt=text)


_POPULAR = _array("popular", [[1, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 0]])

EXERCISES = {
    "similar-to": FunctionBox(
        name="similar-to",
        picture=_si_picture,
        function="similar_to",
        start=_S_START,
        solution=_S,
        show=_shown,
        visible=[(_R(), column(PRODUCT))],
        hidden=[(_array("small", [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]), 0),
                (_array("five", [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 0]]), 2),
                (_POPULAR, 0)],
        first=[_DOTS],
        mistakes=[
            Mistake("together", wrong=_s("together"),
                    en="`{call}` gives {got}: how many customers bought each product together with product `j`. A product that many customers buy anyway, such as WHITE HANGING HEART T-LIGHT HOLDER with 642 buyers, then looks like everything. Divide by the lengths of both columns.",
                    nl="`{call}` geeft {got}: hoeveel klanten elk product samen met product `j` kochten. Een product dat veel klanten toch al kopen, zoals WHITE HANGING HEART T-LIGHT HOLDER met 642 kopers, lijkt dan op alles. Deel door de lengtes van beide kolommen."),
            Mistake("rows-only", wrong=_s("together / lengths"),
                    en="`{call}` gives {got}. The order is right, but these are not cosine similarities: you divide by the length of every column, and not by that of column `j`. Divide by both: `lengths * lengths[j]`.",
                    nl="`{call}` geeft {got}. De volgorde klopt, maar dit zijn geen cosine similarities: je deelt door de lengte van elke kolom, en niet door die van kolom `j`. Deel door allebei: `lengths * lengths[j]`."),
            Mistake("j-only", wrong=_s("together / lengths[j]"),
                    en="`{call}` gives {got}. You divide by the length of column `j` only, the same number for every product, so a popular product still wins. `lengths` holds the length of every column: divide by `lengths * lengths[j]`.",
                    nl="`{call}` geeft {got}. Je deelt alleen door de lengte van kolom `j`, hetzelfde getal voor elk product, dus wint een populair product nog steeds. In `lengths` staat de lengte van elke kolom: deel door `lengths * lengths[j]`."),
            Mistake("brackets", wrong=_s("together / lengths * lengths[j]"),
                    en="`{call}` gives {got}. Python works `together / lengths * lengths[j]` out from left to right: it divides by `lengths` and then multiplies by `lengths[j]`. Put the two lengths between parentheses.",
                    nl="`{call}` geeft {got}. Python rekent `together / lengths * lengths[j]` van links naar rechts uit: het deelt door `lengths` en vermenigvuldigt dan met `lengths[j]`. Zet de twee lengtes tussen haakjes."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: the cosine similarity of column `j` with every column of `R`.",
                      "nl": "Vervang de `...`: de cosine similarity van kolom `j` met elke kolom van `R`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(R, j)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(R, j)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The cosine similarity of two columns is how many customers bought both, divided by the length of the one column times the length of the other.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De cosine similarity van twee kolommen is hoeveel klanten allebei kochten, gedeeld door de lengte van de ene kolom maal de lengte van de andere."},
            "almost": {"en": "Right for HEART OF WICKER SMALL. But `{call}` gives {got}, and it should give {want}. Does your function use `R` and `j`, and nothing of the example?",
                       "nl": "Goed voor HEART OF WICKER SMALL. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `R` en `j`, en niets van het voorbeeld?"},
            "correct": {"en": "Your function is right in all {n} cases. HEART OF WICKER SMALL is most like HEART OF WICKER LARGE, 0.644, and then like the white hearts of wicker: products customers buy together, found without reading a single name.",
                        "nl": "Je functie klopt in alle {n} gevallen. HEART OF WICKER SMALL lijkt het meest op HEART OF WICKER LARGE, 0,644, en dan op de witte rieten hartjes: producten die klanten samen kopen, gevonden zonder één naam te lezen."},
        },
    ),
}


# --- exercise 2: the scores of every product for one customer -------------------------------------------------------

_C_HEAD = _S + "\n\ndef cf_scores(R, bought):\n    scores = np.zeros(R.shape[1])\n    for j in bought:\n"


def _c(add: str = "scores + similar_to(R, j)") -> str:
    return _C_HEAD + f"        scores = {add}\n    return scores\n"


_C_START = _c("...")
_C = _c()


def cf_scores(R, bought):
    """The course's cf_scores: the similarities of every product with each of the bought ones, added up."""
    import numpy as np
    scores = np.zeros(R.shape[1])
    for j in bought:
        scores = scores + similar_to(R, j)
    return scores


@functools.lru_cache(maxsize=1)
def _bought() -> _Val:
    return _Val("bought", bought_columns(EXAMPLE))


_CF_WORDS = {
    "title": {"en": "Your cf_scores for customer 13960", "nl": "Jouw cf_scores voor klant 13960"},
    "said": {"en": "The ten products customer 13960 has not bought that your scores put first, as bars as long as your score. ■: in their last order, {hits} of the 10; □: not. The course's ten hold 2 products of the last order.",
             "nl": "De tien producten die klant 13960 nog niet kocht en die jouw scores bovenaan zetten, als staven zo lang als jouw score. ■: in de laatste bestelling, {hits} van de 10; □: niet. In het tiental van de course zitten 2 producten van de laatste bestelling."},
    "right": {"en": "The ten products customer 13960 has not bought with the highest scores: the similarities with their four products, added up. ■: in their last order, 2 of the 10, LARGE WHITE HEART OF WICKER and WICKER STAR; □: not. Content-based filtering found 3 for this customer.",
              "nl": "De tien producten die klant 13960 nog niet kocht met de hoogste scores: de similarities met hun vier producten, opgeteld. ■: in de laatste bestelling, 2 van de 10, LARGE WHITE HEART OF WICKER en WICKER STAR; □: niet. Content-based filtering vond er voor deze klant 3."},
}


def _cf_picture(seen) -> str:
    import numpy as np
    got = _call(seen, (_R(), _bought()))
    try:
        mine = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return ""
    if mine.shape != (len(names()),) or not np.isfinite(mine).any():
        return ""
    mine = np.where(np.isfinite(mine), mine, -np.inf)
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _CF_WORDS.items()}
    s = split()
    bought, hidden = s["seen"][EXAMPLE], s["hidden"][EXAMPLE]
    ten = [int(i) for i in np.argsort(-mine, kind="stable") if names()[i] not in bought][:10]
    top = max(float(mine[i]) for i in ten)
    rows_ = [(names()[i], float(mine[i]), "hit" if names()[i] in hidden else "") for i in ten]
    if checking.same_value(mine, collaborative_scores(EXAMPLE)):
        text = words_["right"]
    else:
        text = words_["said"].format(hits=sum(1 for _, _, m in rows_ if m))
    return yours.figure(words_["title"], _rows(rows_, lang, top) + _key(text), alt=text)


EXERCISES["cf-scores"] = FunctionBox(
    name="cf-scores",
    picture=_cf_picture,
    function="cf_scores",
    start=_C_START,
    solution=_C,
    show=_shown,
    visible=[(_R(), _bought())],
    hidden=[(_array("small", [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]), _Val("first_two", [0, 1])),
            (_array("five", [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 0]]), _Val("three", [0, 2, 3])),
            (_POPULAR, _Val("one", [2]))],
    first=[_DOTS],
    mistakes=[
        Mistake("last", wrong=_c("similar_to(R, j)"),
                en="`{call}` gives {got}: only the similarities with the last product in `bought`. Every round of the loop throws the scores so far away. Add to them: `scores + ...`.",
                nl="`{call}` geeft {got}: alleen de similarities met het laatste product in `bought`. Elke ronde van de loop gooit de scores tot dan toe weg. Tel erbij op: `scores + ...`."),
        Mistake("counts", wrong=_c("scores + R.T @ R[:, j]"),
                en="`{call}` gives {got}: how often every product was bought together with each bought product, added up. Then the popular products win. Add up the similarities that `similar_to(R, j)` gives.",
                nl="`{call}` geeft {got}: hoe vaak elk product samen met elk gekocht product gekocht werd, opgeteld. Dan winnen de populaire producten. Tel de similarities op die `similar_to(R, j)` geeft."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the `...`: the scores so far, with the similarities with product `j` added.",
                  "nl": "Vervang de `...`: de scores tot nu toe, met de similarities met product `j` erbij opgeteld."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(R, bought)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(R, bought)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every product in `bought` add its similarities to `scores`?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Telt elk product in `bought` zijn similarities op bij `scores`?"},
        "almost": {"en": "Right for customer 13960. But `{call}` gives {got}, and it should give {want}. Does your function use `R` and `bought`, and nothing of the example?",
                   "nl": "Goed voor klant 13960. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `R` en `bought`, en niets van het voorbeeld?"},
        "correct": {"en": "Your function is right in all {n} cases. With the ten highest scores a customer has not bought, collaborative filtering finds something for 800 of the 2,498 customers: 32.0%, against 23.8% for the names and 21.7% for popularity.",
                    "nl": "Je functie klopt in alle {n} gevallen. Met de tien hoogste scores die een klant nog niet kocht, vindt collaborative filtering iets voor 800 van de 2.498 klanten: 32,0%, tegen 23,8% voor de namen en 21,7% voor populariteit."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("similar-to", _S_START, "empty", None),
    ("similar-to", _S, "correct", None),
    ("similar-to", _s("together / lengths / lengths[j]"), "correct", None),
    ("similar-to", _s("together / (lengths * np.linalg.norm(R[:, j]))"), "correct", None),
    ("similar-to", _s("together"), "wrong", "together"),
    ("similar-to", _s("together / lengths"), "wrong", "rows-only"),
    ("similar-to", _s("together / lengths[j]"), "wrong", "j-only"),
    ("similar-to", _s("together / lengths * lengths[j]"), "wrong", "brackets"),
    ("similar-to", _s("together / (lengths * 20.566963801203133)"), "almost", None),
    ("cf-scores", _C_START, "empty", None),
    ("cf-scores", _C, "correct", None),
    ("cf-scores", _c("similar_to(R, j) + scores"), "correct", None),
    ("cf-scores", _c("similar_to(R, j)"), "wrong", "last"),
    ("cf-scores", _c("scores + R.T @ R[:, j]"), "wrong", "counts"),
]
