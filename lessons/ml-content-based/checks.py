# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-content-based: the words of the product names of the online shop as item features, the
profile of a customer, the cosine similarity of every product with it, the ten closest products a customer has not
bought, and how that compares with the popularity baseline of ml-recommend-problem on the same customers and the same
hidden orders; and the functions cosine_scores and recommend."""
import contextlib
import functools
import io
import math
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.shop import (content_for, content_found, content_scores, found, name_vectors,  # noqa: F401
                                popular_for, profile_of, split, words)
from everything_ai.check import FunctionBox, Mistake

EXAMPLE = 13960.0                                      # the customer of the worked example and the exercises
CUSTOMERS = [13960.0, 13242.0, 16356.0, 14905.0, 14875.0]   # the customers of the slider, 1 to 5
NEW = "GREEN HANGING WICKER HEART"                    # a made-up new product, in the choice question


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def percent(x: float, lang: str, digits: int = 1) -> str:
    return number(100 * x, lang, digits) + "%"


@functools.lru_cache(maxsize=1)
def names() -> list:
    """The products a recommender may show, the most popular first: the rows of the features."""
    return list(split()["counts"].index)


@functools.lru_cache(maxsize=1)
def compared() -> dict:
    """At k = 10, how many test customers content-based filtering and the popularity baseline each find something for,
    for how many both do, only one does, or at least one does."""
    c, p = content_found(10), found(popular_for, 10)
    return {"n": len(c), "content": sum(c.values()), "popular": sum(p.values()),
            "both": sum(1 for x in c if c[x] and p[x]), "content_only": sum(1 for x in c if c[x] and not p[x]),
            "popular_only": sum(1 for x in c if p[x] and not c[x]), "either": sum(1 for x in c if c[x] or p[x])}


def profile_words(customer: float, n: int = 8) -> list:
    """The n words that are most often in the profile of a customer, with how often: equal counts in the order of the
    columns."""
    import numpy as np
    p = profile_of(customer)
    word_of = {i: w for w, i in words().items()}
    return [(word_of[int(i)], int(p[i])) for i in np.argsort(-p, kind="stable")[:n] if p[i] > 0]


@functools.lru_cache(maxsize=None)
def tens(customer: float) -> dict:
    """The ten products of content-based filtering and of the popularity baseline for one customer, and for each
    whether it is in their last order."""
    hidden = split()["hidden"][customer]
    c, p = content_for(customer, 10), popular_for(customer, 10)
    return {"content": [(x, x in hidden) for x in c], "popular": [(x, x in hidden) for x in p]}


def new_score(customer: float = EXAMPLE) -> float:
    """The cosine similarity of the made-up new product with the profile of a customer."""
    import numpy as np
    w = words()
    row = np.zeros(len(w))
    for word in NEW.split():
        row[w[word]] = 1
    p = profile_of(customer)
    return float(row @ p / (np.linalg.norm(row) * np.linalg.norm(p)))


def said(customer: float, lang: str) -> str:
    """The sentence under the picture of the slider."""
    s = split()
    bought, new = len(s["seen"][customer]), len(s["hidden"][customer])
    t = tens(customer)
    hc, hp = sum(h for _, h in t["content"]), sum(h for _, h in t["popular"])
    top = profile_words(customer, 3)
    if lang == "nl":
        often = ", ".join(f"{w} {c} keer" for w, c in top)
        return (f"Klant {customer:.0f} kocht {bought} producten vóór de laatste bestelling, en in die laatste bestelling "
                f"zitten er {new} die nieuw zijn voor deze klant. De woorden die het vaakst in het profiel staan: {often}. "
                f"Van de tien producten die het dichtst bij het profiel liggen, zitten er {hc} in de laatste bestelling; "
                f"van de tien populairste {hp}.")
    often = ", ".join(f"{w} {c} time{'' if c == 1 else 's'}" for w, c in top)
    return (f"Customer {customer:.0f} bought {bought} products before their last order, and their last order holds "
            f"{new} that are new to them. The words most often in their profile: {often}. Of the ten products closest "
            f"to the profile, {hc} are in their last order; of the ten most popular, {hp}.")


# --- the pictures ----------------------------------------------------------------------------------------------------

def _word_ticks(top: int) -> list:
    return list(range(0, top + 1)) if top <= 5 else list(range(0, top + 1, 5))


def picture(customer: float, labels: dict) -> str:
    """The slider: the words most often in the profile of a customer as bars, and their ten products of each
    recommender as squares, filled when the product is in their last order."""
    pw = profile_words(customer)
    top = max(c for _, c in pw)
    left = chart.Panel((0, top * 2.1), (-0.6, len(pw) - 0.4), title=labels["words"], xlabel=labels["times"],
                       xticks=_word_ticks(top), yticks=[])
    for i, (w, c) in enumerate(pw):
        y = len(pw) - 1 - i
        left.rect(0, y - 0.32, c, y + 0.32, role="ink", solid=True)
        left.text(c + top * 0.06, y - 0.12, w, role="ink-2", anchor="start")
    t = tens(customer)
    right = chart.Panel((-0.7, 9.7), (-0.8, 3.4), title=labels["ten"], xlabel=labels["place"], xticks=[0, 4, 9],
                        xtick_labels=["1", "5", "10"], yticks=[])
    for row, (key, y) in enumerate((("content", 1.9), ("popular", 0))):
        right.text(-0.6, y + 0.75, labels[key], role="ink-2", anchor="start")
        for x, (_, hit) in enumerate(t[key]):
            right.rect(x - 0.38, y - 0.38, x + 0.38, y + 0.38, role="ink" if hit else "ink-3", solid=hit)
    return chart.figure([left, right], alt=labels["alt"], width=330, height=280)


# The lesson is done when these are correct. "new-product" is the choice question in the lesson.
REQUIRED = ["cosine-scores", "recommend", "new-product"]


class _Val:
    """A test value that the check's messages write by its name, such as `features`; the function gets a copy."""

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


def _rows(rows: list, lang: str, top: float | None = None) -> str:
    """A list of products, one row each: a square, filled when the product is in the customer's last order, open when
    it is not, and a × when they bought it already; the name; and with `top`, a bar as long as its score and the
    score."""
    out = []
    for name, score, mark in rows:
        sign = {"hit": "■", "bought": "×"}.get(mark, "□")
        tone = "var(--ea-ink)" if mark else "var(--ea-ink-3)"
        cells = (f'<span style="color:{tone};font-size:1rem;line-height:1">{sign}</span>'
                 f'<span style="color:var(--ea-ink);overflow-wrap:anywhere">{name}</span>')
        if top is not None:
            share = max(0.0, min(1.0, score / top)) if top > 0 and math.isfinite(score) else 0.0
            look = ("background:var(--ea-ink)" if mark == "hit" else "border:1px solid var(--ea-ink-3)")
            cells += (f'<span style="display:block;height:0.7rem;width:{100 * share:.1f}%;{look};box-sizing:border-box"></span>'
                      f'<span style="color:var(--ea-ink-2);text-align:right">{number(score, lang, 3)}</span>')
        out.append(cells)
    columns = "1.1rem minmax(0,1fr) 6rem 3.2rem" if top is not None else "1.1rem minmax(0,1fr)"
    return (f'<div style="display:grid;grid-template-columns:{columns};gap:0.2rem 0.6rem;align-items:center;'
            f'font-family:var(--ea-font-ui);font-size:0.85rem;max-width:460px;margin:0.3rem 0">' + "".join(out) + "</div>")


# --- exercise 1: the cosine similarity of every product with the profile --------------------------------------------

@functools.lru_cache(maxsize=1)
def _features() -> _Val:
    return _Val("features", name_vectors())


@functools.lru_cache(maxsize=1)
def _profile() -> _Val:
    return _Val("profile", profile_of(EXAMPLE))


def _array(name: str, rows) -> _Val:
    import numpy as np
    return _Val(name, np.array(rows, dtype=float))


_C_HEAD = ("import numpy as np\n\n\ndef cosine_scores(features, profile):\n    dots = features @ profile\n"
           "    lengths = np.linalg.norm(features, axis=1)\n")


def _c(back: str = "dots / (lengths * np.linalg.norm(profile))") -> str:
    return _C_HEAD + f"    return {back}\n"


_C_START = _c("...")
_C = _c()


def cosine_scores(features, profile):
    """The course's cosine_scores: the cosine similarity of every row of features with profile."""
    import numpy as np
    return features @ profile / (np.linalg.norm(features, axis=1) * np.linalg.norm(profile))


_CS_WORDS = {
    "title": {"en": "Your cosine_scores for customer 13960", "nl": "Jouw cosine_scores voor klant 13960"},
    "said": {"en": "The ten products your scores put first that customer 13960 has not bought, as bars as long as your score. ■: in their last order, {hits} of the 10; □: not. Your highest score is {top}. A cosine similarity is at most 1, and the course's ten hold 3 products of the last order.",
             "nl": "De tien producten die klant 13960 nog niet kocht en die jouw scores bovenaan zetten, als staven zo lang als jouw score. ■: in de laatste bestelling, {hits} van de 10; □: niet. Jouw hoogste score is {top}. Een cosine similarity is hoogstens 1, en in het tiental van de course zitten 3 producten van de laatste bestelling."},
    "right": {"en": "The ten products closest to the profile of customer 13960 that they have not bought, as bars as long as their cosine similarity. ■: in their last order; □: not. 3 of the 10 are, the 2nd, the 8th and the 9th: wicker hearts, like the two they bought.",
              "nl": "De tien producten die het dichtst bij het profiel van klant 13960 liggen en die deze klant nog niet kocht, als staven zo lang als hun cosine similarity. ■: in de laatste bestelling; □: niet. Dat zijn er 3 van de 10, de 2e, de 8e en de 9e: rieten hartjes, zoals de twee die de klant kocht."},
}


def _ten_of(scores) -> list:
    """The ten products customer 13960 has not bought with the highest scores; equal scores in the order of names."""
    import numpy as np
    seen = split()["seen"][EXAMPLE]
    out = []
    for i in np.argsort(-scores, kind="stable"):
        if names()[i] not in seen:
            out.append(int(i))
            if len(out) == 10:
                break
    return out


def _cs_picture(seen) -> str:
    import numpy as np
    got = _call(seen, (_features(), _profile()))
    try:
        mine = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return ""
    if mine.shape != (len(names()),) or not np.isfinite(mine).any():
        return ""
    mine = np.where(np.isfinite(mine), mine, -np.inf)
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _CS_WORDS.items()}
    hidden = split()["hidden"][EXAMPLE]
    ten = _ten_of(mine)
    top = max(float(mine[i]) for i in ten)
    rows = [(names()[i], float(mine[i]), "hit" if names()[i] in hidden else "") for i in ten]
    if checking.same_value(mine, cosine_scores(_features().value, _profile().value)):
        text = words_["right"]
    else:
        text = words_["said"].format(hits=sum(1 for _, _, m in rows if m), top=number(top, lang, 3))
    return yours.figure(words_["title"], _rows(rows, lang, max(top, 1.0)) + _key(text), alt=text)


EXERCISES = {
    "cosine-scores": FunctionBox(
        name="cosine-scores",
        picture=_cs_picture,
        function="cosine_scores",
        start=_C_START,
        solution=_C,
        show=_shown,
        visible=[(_features(), _profile())],
        hidden=[(_array("four", [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]]), _array("two_words", [2, 1, 0, 0])),
                (_array("long", [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]]), _array("first_two", [1, 1, 0, 0, 0, 0])),
                (_array("three", [[1, 0, 0], [0, 1, 0], [1, 1, 1]]), _array("third", [0, 0, 3]))],
        first=[_DOTS],
        mistakes=[
            Mistake("dots", wrong=_c("dots"),
                    en="`{call}` gives {got}: dot products, not cosine similarities. A product with a long name shares a word with the profile more easily. Divide by the lengths of both, the rows and the profile.",
                    nl="`{call}` geeft {got}: inproducten, geen cosine similarities. Een product met een lange naam deelt makkelijker een woord met het profiel. Deel door de lengtes van allebei, de rijen en het profiel."),
            Mistake("rows-only", wrong=_c("dots / lengths"),
                    en="`{call}` gives {got}. The order is right, but these are not cosine similarities: you divide by the length of each row, and not by that of the profile. Divide by both: `lengths * np.linalg.norm(profile)`.",
                    nl="`{call}` geeft {got}. De volgorde klopt, maar dit zijn geen cosine similarities: je deelt door de lengte van elke rij, en niet door die van het profiel. Deel door allebei: `lengths * np.linalg.norm(profile)`."),
            Mistake("profile-only", wrong=_c("dots / np.linalg.norm(profile)"),
                    en="`{call}` gives {got}. You divide by the length of the profile, but not by that of each row, so a long name still wins. `lengths` holds the length of every row: divide by `lengths * np.linalg.norm(profile)`.",
                    nl="`{call}` geeft {got}. Je deelt door de lengte van het profiel, maar niet door die van elke rij, dus een lange naam wint nog steeds. In `lengths` staat de lengte van elke rij: deel door `lengths * np.linalg.norm(profile)`."),
            Mistake("brackets", wrong=_c("dots / lengths * np.linalg.norm(profile)"),
                    en="`{call}` gives {got}. Python works `dots / lengths * np.linalg.norm(profile)` out from left to right: it divides by `lengths` and then multiplies by the length of the profile. Put the two lengths between parentheses.",
                    nl="`{call}` geeft {got}. Python rekent `dots / lengths * np.linalg.norm(profile)` van links naar rechts uit: het deelt door `lengths` en vermenigvuldigt dan met de lengte van het profiel. Zet de twee lengtes tussen haakjes."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...`: the cosine similarity of every row of `features` with `profile`.",
                      "nl": "Vervang de `...`: de cosine similarity van elke rij van `features` met `profile`."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(features, profile)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(features, profile)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The cosine similarity is the dot product divided by the length of the row times the length of the profile.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De cosine similarity is het inproduct gedeeld door de lengte van de rij maal de lengte van het profiel."},
            "almost": {"en": "Right for customer 13960. But `{call}` gives {got}, and it should give {want}. Does your function use `features` and `profile`, and nothing of the example?",
                       "nl": "Goed voor klant 13960. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `features` en `profile`, en niets van het voorbeeld?"},
            "correct": {"en": "Your function is right in all {n} cases. For customer 13960, the products they have not bought score from 0 to 0.676; the highest are the two white hearts of wicker, which share HEART, OF, WICKER, and SMALL or LARGE with their profile.",
                        "nl": "Je functie klopt in alle {n} gevallen. Voor klant 13960 scoren de producten die de klant nog niet kocht van 0 tot 0,676; het hoogst de twee witte rieten hartjes, die HEART, OF, WICKER, en SMALL of LARGE met het profiel delen."},
        },
    ),
}


# --- exercise 2: the k closest products a customer has not bought ---------------------------------------------------

@functools.lru_cache(maxsize=1)
def _scores() -> _Val:
    return _Val("scores", content_scores(EXAMPLE))


@functools.lru_cache(maxsize=1)
def _names() -> _Val:
    return _Val("names", names())


def _seen_of(customer: float) -> _Val:
    return _Val("seen", set(split()["seen"][customer]))


_R_HEAD = "import numpy as np\n\n\ndef recommend(scores, names, seen, k):\n    top = []\n"


def _r(order: str = 'np.argsort(-scores, kind="stable")', test: str = "names[i] not in seen and len(top) < k") -> str:
    return _R_HEAD + f"    for i in {order}:\n        if {test}:\n            top.append(names[i])\n    return top\n"


_R_START = _r("...", "...")
_R = _r()


def recommend(scores, names, seen, k):
    """The course's recommend: the k names with the highest scores that are not in seen; equal scores in their order."""
    import numpy as np
    return [names[i] for i in np.argsort(-scores, kind="stable") if names[i] not in seen][:k]


_RE_WORDS = {
    "title": {"en": "Your recommend(scores, names, seen, 10) for customer 13960", "nl": "Jouw recommend(scores, names, seen, 10) voor klant 13960"},
    "said": {"en": "Your list, in its order. ■: in their last order, {hits}; □: not; ×: bought already, {bought}. Your list has {n} products; the course's has 10, with 3 of the last order.",
             "nl": "Jouw lijst, in zijn volgorde. ■: in de laatste bestelling, {hits}; □: niet; ×: al gekocht, {bought}. Jouw lijst heeft {n} producten; die van de course heeft er 10, met 3 van de laatste bestelling."},
    "right": {"en": "Your list, the course's: the ten products closest to the profile of customer 13960 that they have not bought. ■: in their last order, 3 of the 10; □: not. The ten most popular products find none.",
              "nl": "Jouw lijst, die van de course: de tien producten die het dichtst bij het profiel van klant 13960 liggen en die deze klant nog niet kocht. ■: in de laatste bestelling, 3 van de 10; □: niet. De tien populairste producten vinden er geen."},
    "more": {"en": "and {n} more", "nl": "en nog {n}"},
}


def _re_picture(seen) -> str:
    got = _call(seen, (_scores(), _names(), _seen_of(EXAMPLE), 10))
    if not isinstance(got, (list, tuple)) or not got:
        return ""
    lang = seen.lang
    words_ = {k: v[lang] for k, v in _RE_WORDS.items()}
    s = split()
    hidden, bought = s["hidden"][EXAMPLE], s["seen"][EXAMPLE]
    marks = ["hit" if p in hidden else "bought" if p in bought else "" for p in got]
    rows = [(str(p), None, m) for p, m in zip(got[:12], marks[:12])]
    body = _rows(rows, lang)
    if len(got) > 12:
        body += _key(words_["more"].format(n=len(got) - 12))
    if checking.same_value(list(got), recommend(_scores().value, names(), bought, 10)):
        text = words_["right"]
    else:
        text = words_["said"].format(hits=marks.count("hit"), bought=marks.count("bought"), n=len(got))
    return yours.figure(words_["title"], body + _key(text), alt=text)


def _tie_reordered(got) -> bool:
    """Whether `got` is the course's ten for customer 13960 with equal scores in another order: the same score at every
    place, but not the same list."""
    scores, seen = content_scores(EXAMPLE), split()["seen"][EXAMPLE]
    want = recommend(scores, names(), seen, 10)
    at = {n: i for i, n in enumerate(names())}
    if not isinstance(got, list) or len(got) != len(want) or got == want:
        return False
    try:
        return all(g not in seen for g in got) and [scores[at[g]] for g in got] == [scores[at[w]] for w in want]
    except (KeyError, TypeError):
        return False


_SCORES5 = _array("five", [0.2, 0.9, 0.5, 0.9, 0.0])
_TIES = _array("ties", [0.5, 0.1, 0.5, 0.5, 0.3])

EXERCISES["recommend"] = FunctionBox(
    name="recommend",
    picture=_re_picture,
    function="recommend",
    start=_R_START,
    solution=_R,
    show=_shown,
    visible=[(_scores(), _names(), _seen_of(EXAMPLE), 10)],
    hidden=[(_SCORES5, _Val("letters", ["A", "B", "C", "D", "E"]), _Val("bought", {"B"}), 2),
            (_TIES, _Val("letters", ["V", "W", "X", "Y", "Z"]), _Val("nothing", set()), 3),
            (_array("two", [0.3, 0.1]), _Val("two_names", ["P", "Q"]), _Val("bought_p", {"P"}), 3)],
    first=[_DOTS],
    mistakes=[
        Mistake("lowest", wrong=_r(order='np.argsort(scores, kind="stable")'),
                en="`{call}` gives {got}: the products with the lowest scores, the least like the profile. `np.argsort` sorts from small to large. Sort `-scores`: the highest score is then the smallest.",
                nl="`{call}` geeft {got}: de producten met de laagste scores, die het minst op het profiel lijken. `np.argsort` sorteert van klein naar groot. Sorteer `-scores`: de hoogste score is dan de kleinste."),
        Mistake("reversed", wrong=_r(order='np.argsort(scores, kind="stable")[::-1]'),
                en="`{call}` gives {got}, and it should give {want}. Turning the order around puts equal scores the other way around too: the least popular first. Sort `-scores` instead.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Als je de volgorde omdraait, komen gelijke scores ook andersom: de minst populaire eerst. Sorteer in plaats daarvan `-scores`."),
        Mistake("bought", wrong=_r(test="len(top) < k"),
                en="`{call}` gives {got}: also products the customer bought already. Skip those: `names[i] not in seen`.",
                nl="`{call}` geeft {got}: ook producten die de klant al kocht. Sla die over: `names[i] not in seen`."),
        Mistake("all", wrong=_r(test="names[i] not in seen"),
                en="`{call}` gives {got}: every product the customer has not bought. A page has room for `k`: stop adding when `top` has `k` products.",
                nl="`{call}` geeft {got}: elk product dat de klant nog niet kocht. Er is plek voor `k`: voeg niets meer toe als `top` `k` producten heeft."),
        Mistake("stable", matches=lambda got, text: "stable" not in text and _tie_reordered(got),
                en="`{call}` gives {got}, and it should give {want}. The same scores, but equal scores in another order. Many products have equal scores here: names as long, that share as much with the profile. Add `kind=\"stable\"`, so that equal scores keep the order of `names`, the most popular first.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Dezelfde scores, maar gelijke scores in een andere volgorde. Veel producten hebben hier gelijke scores: namen die even lang zijn en evenveel met het profiel delen. Zet er `kind=\"stable\"` bij, dan houden gelijke scores de volgorde van `names`, de populairste eerst."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: the positions of the products from the highest score down, and when a product goes into `top`.",
                  "nl": "Vervang de twee `...`: de posities van de producten van de hoogste score omlaag, en wanneer een product in `top` komt."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(scores, names, seen, k)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(scores, names, seen, k)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you go through the positions from the highest score down, and keep a product only when the customer has not bought it and `top` has fewer than `k`?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Ga je de posities langs van de hoogste score omlaag, en houd je een product alleen als de klant het nog niet kocht en `top` er minder dan `k` heeft?"},
        "almost": {"en": "Right for customer 13960. But `{call}` gives {got}, and it should give {want}. Does your function use `scores`, `names`, `seen` and `k`, and nothing of the example?",
                   "nl": "Goed voor klant 13960. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `scores`, `names`, `seen` en `k`, en niets van het voorbeeld?"},
        "correct": {"en": "Your function is right in all {n} cases. Customer 13960 gets ten hearts, nine of them of wicker, and three are in their last order. Over all 2,498 customers, content-based filtering finds something for 595: 23.8%.",
                    "nl": "Je functie klopt in alle {n} gevallen. Klant 13960 krijgt tien hartjes, waarvan negen van riet, en drie ervan zitten in de laatste bestelling. Over alle 2.498 klanten vindt content-based filtering voor 595 iets: 23,8%."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("cosine-scores", _C_START, "empty", None),
    ("cosine-scores", _C, "correct", None),
    ("cosine-scores", _c("dots / lengths / np.linalg.norm(profile)"), "correct", None),
    ("cosine-scores", _c("features @ profile / (np.linalg.norm(features, axis=1) * np.linalg.norm(profile))"), "correct", None),
    ("cosine-scores", _c("dots"), "wrong", "dots"),
    ("cosine-scores", _c("dots / lengths"), "wrong", "rows-only"),
    ("cosine-scores", _c("dots / np.linalg.norm(profile)"), "wrong", "profile-only"),
    ("cosine-scores", _c("dots / lengths * np.linalg.norm(profile)"), "wrong", "brackets"),
    ("cosine-scores", _c("dots / (lengths * 5.291502622129181)"), "almost", None),
    ("recommend", _R_START, "empty", None),
    ("recommend", _r(test="..."), "wrong", "dots-left"),
    ("recommend", _R, "correct", None),
    ("recommend", _r(test="len(top) < k and names[i] not in seen"), "correct", None),
    ("recommend", _r(order='np.argsort(scores, kind="stable")'), "wrong", "lowest"),
    ("recommend", _r(order='np.argsort(scores, kind="stable")[::-1]'), "wrong", "reversed"),
    ("recommend", _r(test="len(top) < k"), "wrong", "bought"),
    ("recommend", _r(test="names[i] not in seen"), "wrong", "all"),
    ("recommend", _r(order="np.argsort(-scores)"), "wrong", "stable"),
    ("recommend", _r(test="names[i] not in seen and len(top) < 10"), "almost", None),
]
