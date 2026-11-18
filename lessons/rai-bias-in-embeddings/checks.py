# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-bias-in-embeddings: the word lists of three of WEAT's tests (Caliskan, Bryson and
Narayanan 2017), the embeddings Granite Embedding gives each word (data/course-fairness-granite.npz, written by
data/tools/record_course_fairness_granite.py), and the three functions the learner writes: association (how much
closer a word is to one list of words than to another), effect_size (WEAT's effect size) and neutralize (Bolukbasi and
others' step that takes the gender direction out of a word); with the pictures of what each gives on Granite."""
import html
import json
import re

import numpy as np

from everything_ai import lesson as lesson_helpers, paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = paths.ROOT / "data" / "course-fairness-granite.npz"
REPO = "ibm-granite/granite-embedding-107m-multilingual"   # everything_ai.embed.REPO, without importing PyTorch
FILES = [(REPO, "model.safetensors"), (REPO, "tokenizer.json")]
DOWNLOADS = {"Granite Embedding": FILES}   # what the demo downloads, for the line under the header

# --- the word lists ---------------------------------------------------------------------------------------------------
# Three tests of Caliskan, Bryson and Narayanan (Science, 2017), rows 6 to 8 of their Table 1, with the word lists of
# their arXiv draft (arXiv:1608.07187, pp. 5-6); Gonen and Goldberg (2019, footnote 12) give the same lists. X and Y
# are the target words, A and B the attribute words; "glove" is the effect size the paper found in GloVe.
TESTS = {
    "career": {"X": ["John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill"],
               "Y": ["Amy", "Joan", "Lisa", "Sarah", "Diana", "Kate", "Ann", "Donna"],
               "A": ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"],
               "B": ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"],
               "glove": 1.81},
    "math": {"X": ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
             "Y": ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
             "A": ["male", "man", "boy", "brother", "he", "him", "his", "son"],
             "B": ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"],
             "glove": 1.06},
    "science": {"X": ["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"],
                "Y": ["poetry", "art", "Shakespeare", "dance", "literature", "novel", "symphony", "drama"],
                "A": ["brother", "father", "uncle", "grandfather", "son", "he", "his", "him"],
                "B": ["sister", "mother", "aunt", "grandmother", "daughter", "she", "hers", "her"],
                "glove": 1.24},
}
# How the lesson's code names each list, and the gender-neutral lists that neutralize changes (names and words such as
# "he" or "sister" are about gender, so Bolukbasi and others leave them as they are).
LISTS = {"career": ("MEN", "WOMEN", "CAREER", "FAMILY"), "math": ("MATH", "ARTS", "MALE", "FEMALE"),
         "science": ("SCIENCE", "ARTS_2", "MALE_2", "FEMALE_2")}
NEUTRAL_LISTS = ["CAREER", "FAMILY", "MATH", "ARTS", "SCIENCE", "ARTS_2"]
# The ten pairs Bolukbasi, Chang, Zou, Saligrama and Kalai (2016, Figure 2) find the gender direction with, she side
# first, and the ten occupations their Figure 1 puts closest to "she" and to "he" in word2vec.
PAIRS = [("she", "he"), ("her", "his"), ("woman", "man"), ("Mary", "John"), ("herself", "himself"),
         ("daughter", "son"), ("mother", "father"), ("gal", "guy"), ("girl", "boy"), ("female", "male")]
SHE_JOBS = ["homemaker", "nurse", "receptionist", "librarian", "socialite", "hairdresser", "nanny", "bookkeeper",
            "stylist", "housekeeper"]
HE_JOBS = ["maestro", "skipper", "protege", "philosopher", "captain", "architect", "financier", "warrior",
           "broadcaster", "magician"]
# Ethayarajh, Duvenaud and Hirst (2019, p. 1700): one word on each side, with two choices of attribute words.
DOOR = {"X": ["door"], "Y": ["curtain"], "pairs": [(["masculine"], ["feminine"]), (["man"], ["woman"])]}
SENTENCE = "This is {}."   # one of the "semantically bleached" templates of SEAT (May and others 2019, p. 623)


def all_words() -> list:
    """Every word the lesson embeds, each once, in a fixed order: the course's run has one embedding per word."""
    out = []
    for t in TESTS.values():
        for side in "XYAB":
            out += t[side]
    for a, b in PAIRS:
        out += [a, b]
    out += SHE_JOBS + HE_JOBS + DOOR["X"] + DOOR["Y"]
    for a, b in DOOR["pairs"]:
        out += a + b
    return list(dict.fromkeys(out))


def test_words() -> list:
    """The words of the three tests, which the course also embeds in the sentence SENTENCE."""
    return list(dict.fromkeys(w for t in TESTS.values() for side in "XYAB" for w in t[side]))


# --- the course's run of Granite --------------------------------------------------------------------------------------

_RUN: dict = {}


def _unit(rows) -> list:
    """The rows in float64, each scaled to length 1 again: float32 leaves them a hair off, and then the dot product
    and the cosine similarity would differ in the seventh decimal."""
    rows = rows.astype(np.float64)
    return list(rows / np.linalg.norm(rows, axis=1, keepdims=True))


def _run() -> dict:
    if not _RUN:
        with np.load(DATA) as f:
            _RUN["words"] = dict(zip(f["words"].tolist(), _unit(f["vectors"])))
            _RUN["sentences"] = dict(zip(f["sentence_words"].tolist(), _unit(f["sentence_vectors"])))
            _RUN["about"] = json.loads(str(f["about"]))
    return _RUN


def vec() -> dict:
    """Granite's embedding of every word of the lesson, as a one-word text: {word: array of 384 numbers}, each of
    length 1."""
    return _run()["words"]


def sentence_vec() -> dict:
    """Granite's embedding of each word of the three tests in the sentence "This is <word>."."""
    return _run()["sentences"]


class Vectors(list):
    """A list of embeddings that the check and the lesson's code call by a name, such as CAREER, so that a message
    stays short; the learner's function gets a plain list of plain arrays."""

    def __init__(self, items, name: str, words: list | None = None):
        super().__init__(items)
        self.name, self.words = name, words

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [np.array(v) for v in self]


class Named(np.ndarray):
    """One embedding that the check shows by a name, such as vec["John"]; the learner's function gets a plain array."""

    def __repr__(self):
        return getattr(self, "name", None) or np.ndarray.__repr__(np.asarray(self))

    def __deepcopy__(self, memo):
        return np.array(self)


def named(v, name: str) -> Named:
    out = np.asarray(v, dtype=np.float64).view(Named)
    out.name = name
    return out


def lists(V: dict | None = None) -> dict:
    """The word lists of the three tests as Vectors, by the names the lesson uses: MEN, WOMEN, CAREER and so on."""
    V = vec() if V is None else V
    out = {}
    for key, names in LISTS.items():
        for side, name in zip("XYAB", names):
            words = TESTS[key][side]
            out[name] = Vectors([V[w] for w in words], name, words)
    return out


# --- the course's own versions of the three functions -----------------------------------------------------------------

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def association(w, A, B):
    """s(w, A, B) of Caliskan and others: the mean cosine similarity of w with the words of A, minus that with B."""
    to_a = []
    for a in A:
        to_a.append(cosine(w, a))
    to_b = []
    for b in B:
        to_b.append(cosine(w, b))
    return np.mean(to_a) - np.mean(to_b)


def effect_size(X, Y, A, B):
    """WEAT's effect size: the mean association of X minus that of Y, divided by the standard deviation of the
    associations of all target words. np.std as it is, which divides by the number of words: then it is at most 2."""
    s_x = []
    for x in X:
        s_x.append(association(x, A, B))
    s_y = []
    for y in Y:
        s_y.append(association(y, A, B))
    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y)


def neutralize(w, g):
    """Bolukbasi and others' Neutralize for one direction g of length 1: w without its part along g, scaled back to
    length 1."""
    w = w - np.dot(w, g) * g
    return w / np.linalg.norm(w)


def gender_direction(V: dict | None = None):
    """The direction from "he" to "she": the mean of the ten differences she - he, her - his and so on, scaled to
    length 1. Bolukbasi and others take the first principal component of the ten pairs instead; on Granite the two
    point almost the same way (tests/test_rai_bias_in_embeddings.py measures the cosine)."""
    V = vec() if V is None else V
    total = np.zeros(len(V[PAIRS[0][0]]))
    for she, he in PAIRS:
        total = total + (V[she] - V[he])
    return total / np.linalg.norm(total)


def gender_pca(V: dict | None = None):
    """Bolukbasi and others' own way: each pair centred on its mean, and the first principal component of all twenty,
    pointing to the "she" side."""
    V = vec() if V is None else V
    rows = []
    for she, he in PAIRS:
        mid = (V[she] + V[he]) / 2
        rows += [V[she] - mid, V[he] - mid]
    g = np.linalg.svd(np.array(rows), full_matrices=False)[2][0]
    return g if np.dot(g, gender_direction(V)) > 0 else -g


def neutral_words() -> list:
    """The gender-neutral words that Neutralize changes: the target and attribute words of the lists in NEUTRAL_LISTS,
    and the twenty occupations."""
    out = []
    for key, names in LISTS.items():
        for side, name in zip("XYAB", names):
            if name in NEUTRAL_LISTS:
                out += TESTS[key][side]
    return list(dict.fromkeys(out + SHE_JOBS + HE_JOBS))


def neutralized(fn=neutralize, V: dict | None = None) -> dict:
    """Every embedding of the lesson, with the gender-neutral words passed through `fn(w, g)`."""
    V = vec() if V is None else V
    g = gender_direction(V)
    out = dict(V)
    for w in neutral_words():
        out[w] = np.asarray(fn(np.array(V[w]), np.array(g)), dtype=np.float64)
    return out


def effects(V: dict | None = None, fn=effect_size) -> dict:
    """The effect size of each of the three tests, with `fn`."""
    L_ = lists(V)
    return {key: float(fn(*(L_[n] for n in names))) for key, names in LISTS.items()}


def neighbours(V: dict | None = None) -> int:
    """Of the twenty occupations, how many have as their nearest other occupation (by cosine similarity) one from the
    same side of Bolukbasi and others' Figure 1: Gonen and Goldberg's question whether the words still group by their
    old gender, on a small scale."""
    V = vec() if V is None else V
    jobs = SHE_JOBS + HE_JOBS
    same = 0
    for w in jobs:
        best = max((o for o in jobs if o != w), key=lambda o: cosine(V[w], V[o]))
        same += (w in SHE_JOBS) == (best in SHE_JOBS)
    return same


def jobs_effect(V: dict | None = None, fn=effect_size) -> float:
    """The effect size of the ten "she" occupations of Bolukbasi and others' Figure 1 against their ten "he"
    occupations, with the female and male words of the math test as attributes."""
    V = vec() if V is None else V
    return float(fn([V[w] for w in SHE_JOBS], [V[w] for w in HE_JOBS], [V[w] for w in TESTS["math"]["B"]],
                    [V[w] for w in TESTS["math"]["A"]]))


# --- measured on the course's run (tests/test_rai_bias_in_embeddings.py checks every number) ---------------------------

GRANITE = {"career": 1.49, "math": 1.36, "science": 0.96}   # one-word texts
SEAT = {"career": 1.28, "math": 1.30, "science": 0.87}      # in "This is <word>."
AFTER = {"career": 1.15, "math": -0.33, "science": 0.13}    # with the gender-neutral words neutralized
JOBS = (1.11, 0.96)                                          # jobs_effect before and after
NEIGHBOURS = (16, 15)                                        # neighbours before and after
G_PCA = 0.998                                                # cosine of gender_direction and gender_pca


# --- the pictures -----------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _number(x: float, lang: str, digits: int = 2) -> str:
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _mine(seen):
    """The learner's function, or None when the code stopped or the name is not a function."""
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    from everything_ai import check as checking
    return checking.with_time_limit(lambda: fn(*args))


def _ticks(lo: float, hi: float) -> list:
    """Three to six round ticks between lo and hi."""
    span = hi - lo
    for step in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100):
        if span / step <= 6:
            break
    first = np.ceil(lo / step) * step
    return [round(first + i * step, 6) for i in range(int((hi - first) / step + 1e-9) + 1)]


_A_WORDS = {
    "en": {"title": "Your association of each name with career rather than family", "xlabel": "association s(name, CAREER, FAMILY)",
           "rows": {"MEN": "the eight men's names", "WOMEN": "the eight women's names"},
           "mean": "mean {m}",
           "key": "Each dot is one name, placed by your association: to the right, closer to the career words; to the left, "
                  "closer to the family words. The line in each row is its mean.",
           "alt": "Your associations: men's names {men}; women's names {women}."},
    "nl": {"title": "Jouw associatie van elke naam met carrière in plaats van gezin", "xlabel": "associatie s(naam, CAREER, FAMILY)",
           "rows": {"MEN": "de acht mannennamen", "WOMEN": "de acht vrouwennamen"},
           "mean": "gemiddeld {m}",
           "key": "Elke stip is één naam, geplaatst op jouw associatie: naar rechts dichter bij de woorden over carrière, "
                  "naar links dichter bij de woorden over gezin. Het streepje in elke rij is het gemiddelde.",
           "alt": "Jouw associaties: mannennamen {men}; vrouwennamen {women}."},
}


def association_picture(seen) -> str:
    """The learner's association of each of the sixteen names of the career test with CAREER and FAMILY, as dots in
    two rows, with each row's mean."""
    from everything_ai import chart
    fn = _mine(seen)
    if fn is None:
        return ""
    w, L_ = _A_WORDS[seen.lang], lists()
    rows = {}
    try:
        for name in ("MEN", "WOMEN"):
            rows[name] = [float(_call(fn, np.array(v), [np.array(a) for a in L_["CAREER"]],
                                      [np.array(b) for b in L_["FAMILY"]])) for v in L_[name]]
    except Exception:
        return ""
    values = rows["MEN"] + rows["WOMEN"] + [0.0]
    if not all(np.isfinite(values)):
        return ""
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.08 or 0.01
    ticks = _ticks(lo - pad, hi + pad)
    panel = chart.Panel(x=(lo - pad, hi + pad), y=(0, 2.4), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[_number(t, seen.lang, 3) for t in ticks])
    panel.line([0, 0], [0, 2.3], role="ink-3", width=1, dashed=True)
    for i, name in enumerate(("MEN", "WOMEN")):
        y = 1.55 - 1.15 * i
        m = float(np.mean(rows[name]))
        panel.text(lo - pad * 0.9, y + 0.45, w["rows"][name] + ": " + w["mean"].format(m=_number(m, seen.lang, 4)),
                   "ink", anchor="start", halo=True)
        panel.points(rows[name], [y] * 8, role="data", r=4.5, hollow=i == 1)
        panel.line([m, m], [y - 0.25, y + 0.25], role="ink", width=2.4)
    said = {k: ", ".join(_number(v, seen.lang, 4) for v in rows[k]) for k in rows}
    svg = chart.figure([panel], alt=w["alt"].format(men=said["MEN"], women=said["WOMEN"]), width=560, height=230)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


_E_WORDS = {
    "en": {"title": "Your effect size of three of WEAT's tests on Granite", "xlabel": "effect size",
           "rows": {"career": "men's and women's names, career and family", "math": "math and arts, male and female words",
                    "science": "science and arts, male and female words"},
           "key": "Filled: your effect_size on Granite's embeddings of the words. Open: what Caliskan and colleagues found "
                  "in GloVe (2017). An effect size lies between −2 and 2; the dashed lines are those ends.",
           "alt": "Your effect sizes on Granite: {said}. In GloVe: 1.81, 1.06 and 1.24."},
    "nl": {"title": "Jouw effectgrootte bij drie tests van WEAT op Granite", "xlabel": "effectgrootte",
           "rows": {"career": "mannen- en vrouwennamen, carrière en gezin", "math": "wiskunde en kunst, mannelijke en vrouwelijke woorden",
                    "science": "wetenschap en kunst, mannelijke en vrouwelijke woorden"},
           "key": "Gevuld: jouw effect_size op de embeddings van Granite. Open: wat Caliskan en collega's vonden in GloVe "
                  "(2017). Een effectgrootte ligt tussen −2 en 2; de streepjeslijnen zijn die grenzen.",
           "alt": "Jouw effectgroottes op Granite: {said}. In GloVe: 1,81, 1,06 en 1,24."},
}


def _effect_rows(panel, rows: list, lang: str) -> None:
    """Rows of (label, open value or None, filled value), on a panel from −2.2 to 2.2, each with its own label above it."""
    for i, (label, open_, filled) in enumerate(rows):
        y = 0.6 + 1.05 * (len(rows) - 1 - i)
        panel.text(-2.15, y + 0.42, label, "ink", anchor="start", halo=True)
        if open_ is not None and -2.2 <= open_ <= 2.2:
            panel.points([open_], [y], role="data", r=6, hollow=True)
        if open_ is not None and -2.2 <= open_ <= 2.2 and -2.2 <= filled <= 2.2:
            panel.line([open_, filled], [y, y], role="ink-3", width=1.2)
        if -2.2 <= filled <= 2.2:
            panel.points([filled], [y], role="loss", r=5.5)
            panel.text(filled, y - 0.36, _number(filled, lang), "ink-2", halo=True)


def effects_picture(seen) -> str:
    """The learner's effect_size of the three tests on Granite's embeddings, beside what Caliskan and others found in
    GloVe."""
    from everything_ai import chart
    fn = _mine(seen)
    if fn is None:
        return ""
    w = _E_WORDS[seen.lang]
    try:
        got = effects(fn=lambda *a: _call(fn, *(list(np.array(v) for v in x) for x in a)))
    except Exception:
        return ""
    if not all(np.isfinite(list(got.values()))):
        return ""
    panel = chart.Panel(x=(-2.2, 2.2), y=(0, 3.4), xlabel=w["xlabel"], yticks=[], xticks=[-2, -1, 0, 1, 2],
                        xtick_labels=["−2", "−1", "0", "1", "2"])
    for x in (-2, 2):
        panel.line([x, x], [0, 3.3], role="ink-3", width=1, dashed=True)
    panel.line([0, 0], [0, 3.3], role="ink-3", width=1)
    _effect_rows(panel, [(w["rows"][k], TESTS[k]["glove"], got[k]) for k in TESTS], seen.lang)
    said = "; ".join(f'{w["rows"][k]} {_number(got[k], seen.lang)}' for k in TESTS)
    svg = chart.figure([panel], alt=w["alt"].format(said=said), width=560, height=250)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


_N_WORDS = {
    "en": {"title": "WEAT on Granite before and after your neutralize", "xlabel": "effect size",
           "rows": {"career": "names, career and family", "math": "math and arts, male and female words",
                    "science": "science and arts, male and female words",
                    "jobs": "\"she\" and \"he\" occupations, female and male words"},
           "key": "Open: before. Filled: after your neutralize took the gender direction out of every gender-neutral "
                  "word (not out of the men's and women's names, nor out of words such as \"he\" or \"sister\").",
           "along": "Along the gender direction, the neutral words now lie at most {m} from 0.",
           "near": "Occupations whose nearest occupation is from the same side of Bolukbasi and colleagues' list: "
                   "{before} of 20 before, {after} of 20 after.",
           "alt": "Effect sizes before and after your neutralize: {said}."},
    "nl": {"title": "WEAT op Granite voor en na jouw neutralize", "xlabel": "effectgrootte",
           "rows": {"career": "namen, carrière en gezin", "math": "wiskunde en kunst, mannelijke en vrouwelijke woorden",
                    "science": "wetenschap en kunst, mannelijke en vrouwelijke woorden",
                    "jobs": "beroepen bij \"she\" en bij \"he\", vrouwelijke en mannelijke woorden"},
           "key": "Open: ervoor. Gevuld: nadat jouw neutralize de genderrichting uit elk genderneutraal woord haalde "
                  "(niet uit de namen van mannen en vrouwen, en niet uit woorden als \"he\" of \"sister\").",
           "along": "Langs de genderrichting liggen de neutrale woorden nu hooguit {m} van 0.",
           "near": "Beroepen waarvan het dichtstbijzijnde beroep van dezelfde kant van de lijst van Bolukbasi en "
                   "collega's komt: {before} van de 20 ervoor, {after} van de 20 erna.",
           "alt": "Effectgroottes voor en na jouw neutralize: {said}."},
}


def neutralize_picture(seen) -> str:
    """WEAT's effect sizes of the three tests and of the occupations, before and after the learner's neutralize, with
    how far the neutral words still lie along the gender direction and how many occupations keep a neighbour from
    their old side."""
    from everything_ai import chart
    fn = _mine(seen)
    if fn is None:
        return ""
    w = _N_WORDS[seen.lang]
    try:
        after = neutralized(lambda v, g: _call(fn, v, g))
        if any(np.shape(after[x]) != (len(vec()[x]),) for x in neutral_words()):
            return ""
        got = effects(after)
        got["jobs"] = jobs_effect(after)
        near = neighbours(after)
    except Exception:
        return ""
    if not all(np.isfinite(list(got.values()))):
        return ""
    before = effects()
    before["jobs"] = jobs_effect()
    g = gender_direction()
    along = max(abs(float(np.dot(after[x], g))) for x in neutral_words())
    panel = chart.Panel(x=(-2.2, 2.2), y=(0, 4.45), xlabel=w["xlabel"], yticks=[], xticks=[-2, -1, 0, 1, 2],
                        xtick_labels=["−2", "−1", "0", "1", "2"])
    panel.line([0, 0], [0, 4.35], role="ink-3", width=1)
    keys = ["career", "math", "science", "jobs"]
    _effect_rows(panel, [(w["rows"][k], before[k], got[k]) for k in keys], seen.lang)
    said = "; ".join(f'{w["rows"][k]} {_number(before[k], seen.lang)} → {_number(got[k], seen.lang)}' for k in keys)
    svg = chart.figure([panel], alt=w["alt"].format(said=said), width=560, height=300)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    notes = [w["key"], w["along"].format(m=_number(along, seen.lang, 3)),
             w["near"].format(before=neighbours(), after=near)]
    return yours.figure(w["title"], svg + "".join(f'<div style="{_UI}">{html.escape(n)}</div>' for n in notes))


# --- the demo: the learner's own word lists, embedded by Granite -------------------------------------------------------

def size_mb() -> int:
    from everything_ai import hub
    return round(sum(hub.PINNED[pin]["size"] for pin in FILES) / 1e6)


def downloaded() -> bool:
    from everything_ai import hub
    return all(hub.pinned(*pin) is not None for pin in FILES)


def download_or_stop(button, texts: dict) -> None:
    lesson_helpers.pins_or_stop(button, texts, FILES)


MAX_WORDS = 12   # per list in the demo


def words_of(text: str) -> list:
    """The words in a box of the demo: split at commas, without spaces around them, empty ones left out, each once."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out[:MAX_WORDS]


def filled(boxes: dict) -> bool:
    """Whether each of the four boxes of the demo, X, Y, A and B, has at least one word."""
    return all(words_of(boxes[k]) for k in "XYAB")


def demo(boxes: dict, words: dict, lang: str) -> str:
    """The learner's four lists, embedded by Granite one word at a time, as the course's run: each target word's
    association with A and B as a dot with its word, and WEAT's effect size."""
    from everything_ai import chart, embed
    lists_ = {k: words_of(boxes[k]) for k in "XYAB"}
    every = list(dict.fromkeys(lists_["X"] + lists_["Y"] + lists_["A"] + lists_["B"]))
    V = dict(zip(every, embed.embed(every).numpy().astype(np.float64)))
    A, B = [V[a] for a in lists_["A"]], [V[b] for b in lists_["B"]]
    s = {t: float(association(V[t], A, B)) for t in lists_["X"] + lists_["Y"]}
    values = list(s.values())
    d = float(effect_size([V[x] for x in lists_["X"]], [V[y] for y in lists_["Y"]], A, B)) if np.std(values) > 0 else None
    lo, hi = min(values + [0.0]), max(values + [0.0])
    pad = (hi - lo) * 0.12 or 0.01
    ticks = _ticks(lo - pad, hi + pad)
    panel = chart.Panel(x=(lo - pad, hi + pad), y=(0, 3.8), xlabel=words["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[_number(t, lang, 3) for t in ticks])
    panel.line([0, 0], [0, 3.7], role="ink-3", width=1, dashed=True)
    for i, side in enumerate("XY"):
        y = 2.6 - 1.75 * i
        panel.text(lo - pad * 0.9, y + 0.95, words["rows"][side], "ink", anchor="start", halo=True)
        xs = [s[t] for t in lists_[side]]
        panel.points(xs, [y] * len(xs), role="data", r=4.5, hollow=i == 1)
        for t, level in zip(lists_[side], label_levels(xs, lists_[side], hi - lo + 2 * pad)):
            if level is not None:
                panel.text(s[t], y + level, t, "ink-2", halo=True)
    alt = "; ".join(f"{t} {_number(v, lang, 4)}" for t, v in s.items())
    svg = chart.figure([panel], alt=alt, width=560, height=280)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    head = words["effect"].format(d=_number(d, lang)) if d is not None else words["no_spread"]
    every = " · ".join(f"{t} {_number(s[t], lang, 4)}" for t in lists_["X"] + lists_["Y"])
    return yours.figure(words["title"], f'<p style="margin:0 0 0.3rem 0">{html.escape(head)}</p>' + svg
                        + f'<div style="{_UI}">{html.escape(words["every"].format(words=every))}</div>')


_LEVELS = (0.3, -0.34, 0.6, -0.64)   # where a word's label goes, above or below its dot, in the demo's units


def label_levels(xs: list, labels: list, span: float, width_px: float = 470) -> list:
    """For each dot of a row at xs, the first of _LEVELS where its label does not run into a label already placed
    there, from left to right; None when no level has room (the line under the picture still names every word)."""
    per = width_px / span
    ends = [None] * len(_LEVELS)
    out = [None] * len(xs)
    for i in sorted(range(len(xs)), key=lambda k: xs[k]):
        half = (7 * len(labels[i]) + 8) / 2
        middle = xs[i] * per
        for j, level in enumerate(_LEVELS):
            if ends[j] is None or middle - half > ends[j]:
                ends[j], out[i] = middle + half, level
                break
    return out


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["association", "effect", "door", "neutralize", "hidden"]


def _toy(values) -> Named:
    return named(values, "np.array([" + ", ".join(repr(float(v)) for v in values) + "])")


def _toys(rows) -> Vectors:
    vs = [_toy(r) for r in rows]
    return Vectors(vs, "[" + ", ".join(repr(v) for v in vs) + "]")


_L = lists()
_V = vec()
_G = named(gender_direction(), "g")
# the worked example's vectors, and three-dimensional ones whose lengths are not 1
TOY_X, TOY_Y = _toys([[0.8, 0.6], [1.0, 0.0]]), _toys([[0.6, 0.8], [0.0, 1.0]])
TOY_A, TOY_B = _toys([[1.0, 0.0]]), _toys([[0.0, 1.0]])
TOY_W = _toy([3.0, 4.0, 0.0])
TOY_P, TOY_Q = [_toy([1.0, 0.0, 0.0]), _toy([0.0, 2.0, 0.0])], [_toy([0.0, 4.0, 3.0])]


def show_number(v):
    """A number in a message with at most four decimals; anything else as Python writes it."""
    if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
        text = f"{round(float(v), 4):g}"
        return {"en": f"`{text}`", "nl": f"`{text}`"}
    text = f"`{v!r}`"
    return {"en": text, "nl": text}


def show_vector(v):
    """An embedding in a message: a short array in full, a long one by its size, its length and how it starts."""
    if isinstance(v, np.ndarray) and v.ndim == 1 and np.issubdtype(v.dtype, np.number):
        if len(v) <= 3:
            text = "`np.array([" + ", ".join(f"{round(float(x), 4):g}" for x in v) + "])`"
            return {"en": text, "nl": text}
        start = "`[" + ", ".join(f"{float(x):.4f}" for x in v[:3]) + ", ...]`"
        length = f"{float(np.linalg.norm(v)):.4f}"
        return {"en": f"an array of {len(v)} numbers, of length {length}, that starts with {start}",
                "nl": f"een array van {len(v)} getallen, met lengte {length.replace('.', ',')}, die begint met {start}"}
    text = f"`{v!r}`"
    return {"en": text, "nl": text}


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: association
_A_HEAD = ("import numpy as np\n\n\n"
           "def cosine(u, v):\n"
           "    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))\n\n\n"
           "def association(w, A, B):\n"
           "    to_a = []\n"
           "    for a in A:\n"
           "        to_a.append(cosine(w, a))\n")
_A_LOOP = ("    to_b = []\n"
           "    for b in B:\n"
           "        to_b.append(cosine(w, b))\n")
_A_RET = "    return np.mean(to_a) - np.mean(to_b)\n"
_A_START = _A_HEAD + "    ...\n"
_A = _A_HEAD + _A_LOOP + _A_RET


def _a(loop: str = _A_LOOP, ret: str = _A_RET) -> str:
    return _A_HEAD + loop + ret


# exercise 2: effect_size
_E_HEAD = ("import numpy as np\n\n\n"
           "def effect_size(X, Y, A, B):\n"
           "    s_x = []\n"
           "    for x in X:\n"
           "        s_x.append(association(x, A, B))\n"
           "    s_y = []\n"
           "    for y in Y:\n"
           "        s_y.append(association(y, A, B))\n")
_E_RET = "    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y)\n"
_E_START = _E_HEAD + "    ...\n"
_E = _E_HEAD + _E_RET

# exercise 3: neutralize
_N_HEAD = ("import numpy as np\n\n\n"
           "def neutralize(w, g):\n"
           "    along = np.dot(w, g)\n")
_N_BODY = ("    w = w - along * g\n"
           "    return w / np.linalg.norm(w)\n")
_N_START = _N_HEAD + "    ...\n"
_N = _N_HEAD + _N_BODY

_DOT_TEXT = {
    "en": "`{call}` gives {got}, and it should give {want}. Your function uses `np.dot` where it should use `cosine`. Granite's embeddings have length 1, so there the two are the same; in this case the vectors are longer, and the dot product grows with their length. Use `cosine(w, b)`.",
    "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Je functie gebruikt `np.dot` waar `cosine` hoort. De embeddings van Granite hebben lengte 1, dus daar zijn de twee gelijk; in dit geval zijn de vectoren langer, en het inproduct groeit mee met hun lengte. Gebruik `cosine(w, b)`.",
}

EXERCISES = {
    "association": FunctionBox(
        name="association",
        picture=association_picture,
        function="association",
        start=_A_START,
        solution=_A,
        show=show_number,
        visible=[(named(_V["John"], 'vec["John"]'), _L["CAREER"], _L["FAMILY"])],
        hidden=[(named(_V["Amy"], 'vec["Amy"]'), _L["CAREER"], _L["FAMILY"]),
                (named(_V["math"], 'vec["math"]'), _L["MALE"], _L["FEMALE"]),
                (named(_V["poetry"], 'vec["poetry"]'), _L["MALE"], _L["FEMALE"]),
                (TOY_W, TOY_P, TOY_Q), (TOY_W, TOY_Q, TOY_P)],
        first=[_DOTS],
        mistakes=[
            Mistake("flipped", wrong=_a(ret="    return np.mean(to_b) - np.mean(to_a)\n"),
                    en="`{call}` gives {got}, and it should give {want}: the same number with the other sign. You subtract the closeness to `A` from that to `B`. WEAT does it the other way round, so that a word closer to `A` gets a positive association: `np.mean(to_a) - np.mean(to_b)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hetzelfde getal met het andere teken. Je trekt de nabijheid tot `A` af van die tot `B`. WEAT doet het andersom, zodat een woord dat dichter bij `A` ligt een positieve associatie krijgt: `np.mean(to_a) - np.mean(to_b)`."),
            Mistake("sum", wrong=_a(ret="    return np.sum(to_a) - np.sum(to_b)\n"),
                    en="`{call}` gives {got}, and it should give {want}. You add up the cosine similarities. WEAT takes their mean, so that a longer list does not weigh more: `np.mean(to_a) - np.mean(to_b)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt de cosine similarities op. WEAT neemt hun gemiddelde, zodat een langere lijst niet zwaarder weegt: `np.mean(to_a) - np.mean(to_b)`."),
            Mistake("only-a", wrong=_a(loop="", ret="    return np.mean(to_a)\n"),
                    en="`{call}` gives {got}, and it should give {want}. That is how close `w` is to `A`. The association compares: subtract how close `w` is to `B`, with a second loop over `B`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is hoe dicht `w` bij `A` ligt. De associatie vergelijkt: trek af hoe dicht `w` bij `B` ligt, met een tweede loop over `B`."),
            Mistake("dot", wrong=_a(loop=_A_LOOP.replace("cosine(w, b)", "np.dot(w, b)")), en=_DOT_TEXT["en"], nl=_DOT_TEXT["nl"]),
            Mistake("dot-both", wrong=_a(loop=_A_LOOP.replace("cosine(w, b)", "np.dot(w, b)")).replace("cosine(w, a)", "np.dot(w, a)"),
                    en=_DOT_TEXT["en"].replace("Use `cosine(w, b)`.", "Use `cosine(w, a)` and `cosine(w, b)`."),
                    nl=_DOT_TEXT["nl"].replace("Gebruik `cosine(w, b)`.", "Gebruik `cosine(w, a)` en `cosine(w, b)`.")),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the rest: the cosine similarity of `w` with each word of `B`, in a list `to_b`, and then the mean of `to_a` minus the mean of `to_b`.",
                      "nl": "Vervang de `...` door de rest: de cosine similarity van `w` met elk woord van `B`, in een lijst `to_b`, en dan het gemiddelde van `to_a` min het gemiddelde van `to_b`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The association is the mean cosine similarity of `w` with the words of `A`, minus that with the words of `B`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De associatie is de gemiddelde cosine similarity van `w` met de woorden van `A`, min die met de woorden van `B`."},
            "almost": {"en": "Right for `association(vec[\"John\"], CAREER, FAMILY)`. But `{call}` gives {got}, and it should give {want}. Does your function use `w`, `A` and `B`, and no numbers of its own?",
                       "nl": "Goed voor `association(vec[\"John\"], CAREER, FAMILY)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `w`, `A` en `B`, en geen eigen getallen?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does `association` end with `return` and the mean of `to_a` minus the mean of `to_b`, as far indented as the loops?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt `association` met `return` en het gemiddelde van `to_a` min het gemiddelde van `to_b`, even ver ingesprongen als de loops?"},
            "correct": {"en": "Your `association` is right in all {n} cases. The picture places all sixteen names of the test with it.",
                        "nl": "Je `association` klopt in alle {n} gevallen. Het plaatje zet er alle zestien namen van de test mee neer."},
        },
    ),
    "effect": FunctionBox(
        name="effect",
        picture=effects_picture,
        function="effect_size",
        start=_E_START,
        solution=_E,
        names={"association": association},
        show=show_number,
        visible=[(_L["MEN"], _L["WOMEN"], _L["CAREER"], _L["FAMILY"])],
        hidden=[(_L["MATH"], _L["ARTS"], _L["MALE"], _L["FEMALE"]),
                (_L["SCIENCE"], _L["ARTS_2"], _L["MALE_2"], _L["FEMALE_2"]),
                (_L["WOMEN"], _L["MEN"], _L["CAREER"], _L["FAMILY"]),
                (TOY_X, TOY_Y, TOY_A, TOY_B)],
        first=[_DOTS],
        mistakes=[
            Mistake("sample", wrong=_E_HEAD + "    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y, ddof=1)\n",
                    en="`{call}` gives {got}, and it should give {want}. With `ddof=1`, `np.std` divides by one word fewer. Caliskan and colleagues do not say which of the two they use; this lesson uses `np.std` as it is, so that an effect size can reach 2 exactly, as one word a side shows. Leave out `ddof=1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Met `ddof=1` deelt `np.std` door één woord minder. Caliskan en collega's zeggen niet welke van de twee ze gebruiken; deze les gebruikt `np.std` zoals hij is, zodat een effectgrootte precies 2 kan worden, zoals één woord per kant laat zien. Laat `ddof=1` weg."),
            Mistake("sums", wrong=_E_HEAD + "    return (np.sum(s_x) - np.sum(s_y)) / np.std(s_x + s_y)\n",
                    en="`{call}` gives {got}, and it should give {want}. You add up the associations of each list. The effect size takes their means, so that it does not grow with the number of words: `np.mean(s_x) - np.mean(s_y)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt de associaties van elke lijst op. De effectgrootte neemt hun gemiddelden, zodat hij niet groeit met het aantal woorden: `np.mean(s_x) - np.mean(s_y)`."),
            Mistake("no-std", wrong=_E_HEAD + "    return np.mean(s_x) - np.mean(s_y)\n",
                    en="`{call}` gives {got}, and it should give {want}. That is the difference of the two means. The effect size divides it by the standard deviation of all associations, `np.std(s_x + s_y)`, so that it says how far apart the two lists are compared with how much the words differ anyway.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het verschil van de twee gemiddelden. De effectgrootte deelt het door de standaardafwijking van alle associaties, `np.std(s_x + s_y)`, zodat hij zegt hoe ver de twee lijsten uit elkaar liggen vergeleken met hoeveel de woorden toch al verschillen."),
            Mistake("std-x", wrong=_E_HEAD + "    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x)\n",
                    en="`{call}` gives {got}, and it should give {want}. You divide by the standard deviation of `s_x` alone. WEAT takes it over the associations of all target words, of `X` and `Y` together: `np.std(s_x + s_y)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je deelt door de standaardafwijking van alleen `s_x`. WEAT neemt die over de associaties van alle doelwoorden, van `X` en `Y` samen: `np.std(s_x + s_y)`."),
            Mistake("brackets", wrong=_E_HEAD + "    return np.mean(s_x) - np.mean(s_y) / np.std(s_x + s_y)\n",
                    en="`{call}` gives {got}, and it should give {want}. Python divides before it subtracts, so your code divides only `np.mean(s_y)`. Put the difference in brackets: `(np.mean(s_x) - np.mean(s_y)) / ...`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Python deelt voordat het aftrekt, dus je code deelt alleen `np.mean(s_y)`. Zet het verschil tussen haakjes: `(np.mean(s_x) - np.mean(s_y)) / ...`."),
            Mistake("flipped", wrong=_E_HEAD + "    return (np.mean(s_y) - np.mean(s_x)) / np.std(s_x + s_y)\n",
                    en="`{call}` gives {got}, and it should give {want}: the same number with the other sign. WEAT takes the mean of `X` minus that of `Y`, so that a positive effect size means: `X` lies closer to `A`, and `Y` to `B`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: hetzelfde getal met het andere teken. WEAT neemt het gemiddelde van `X` min dat van `Y`, zodat een positieve effectgrootte betekent: `X` ligt dichter bij `A`, en `Y` bij `B`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one line: the mean of `s_x` minus the mean of `s_y`, divided by the standard deviation of all associations together.",
                      "nl": "Vervang de `...` door één regel: het gemiddelde van `s_x` min het gemiddelde van `s_y`, gedeeld door de standaardafwijking van alle associaties samen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Follow the worked example: the mean association of `X` minus that of `Y`, divided by the standard deviation of all of them, `np.std(s_x + s_y)`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Volg het uitgewerkte voorbeeld: de gemiddelde associatie van `X` min die van `Y`, gedeeld door de standaardafwijking van allemaal, `np.std(s_x + s_y)`."},
            "almost": {"en": "Right for `effect_size(MEN, WOMEN, CAREER, FAMILY)`. But `{call}` gives {got}, and it should give {want}. Does your function use `s_x` and `s_y`, and no numbers of its own?",
                       "nl": "Goed voor `effect_size(MEN, WOMEN, CAREER, FAMILY)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `s_x` en `s_y`, en geen eigen getallen?"},
            "correct": {"en": "Your `effect_size` is right in all {n} cases. The picture gives the three tests on Granite with it.",
                        "nl": "Je `effect_size` klopt in alle {n} gevallen. Het plaatje geeft er de drie tests op Granite mee."},
        },
    ),
    "neutralize": FunctionBox(
        name="neutralize",
        picture=neutralize_picture,
        function="neutralize",
        start=_N_START,
        solution=_N,
        show=show_vector,
        visible=[(named(_V["nurse"], 'vec["nurse"]'), _G)],
        hidden=[(named(_V["math"], 'vec["math"]'), _G), (named(_V["captain"], 'vec["captain"]'), _G),
                (_toy([0.6, 0.0, 0.8]), _toy([0.0, 0.0, 1.0])), (TOY_W, _toy([1.0, 0.0, 0.0]))],
        first=[_DOTS],
        mistakes=[
            Mistake("no-scale", wrong=_N_HEAD + "    return w - along * g\n",
                    en="`{call}` gives {got}, and it should give {want}. Taking the part along `g` out makes `w` shorter. Neutralize then scales it back to length 1, as every embedding of Granite has: divide by `np.linalg.norm(w)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Als je het deel langs `g` eruit haalt, wordt `w` korter. Neutralize schaalt hem daarna terug naar lengte 1, zoals elke embedding van Granite heeft: deel door `np.linalg.norm(w)`."),
            Mistake("plus", wrong=_N_HEAD + "    w = w + along * g\n    return w / np.linalg.norm(w)\n",
                    en="`{call}` gives {got}, and it should give {want}. You add the part along `g`, so the word moves further along the gender direction. Take it out: `w - along * g`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt het deel langs `g` erbij op, dus het woord schuift verder langs de genderrichting. Haal het eruit: `w - along * g`."),
            Mistake("along-only", wrong=_N_HEAD + "    return along * g\n",
                    en="`{call}` gives {got}, and it should give {want}. That is the part of `w` along `g`, the part Neutralize takes out. Give back the rest: `w - along * g`, scaled to length 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het deel van `w` langs `g`, het deel dat Neutralize eruit haalt. Geef de rest terug: `w - along * g`, geschaald naar lengte 1."),
            Mistake("scale-only", wrong=_N_HEAD + "    return w / np.linalg.norm(w)\n",
                    en="`{call}` gives {got}, and it should give {want}. Your function scales `w` to length 1, and the part along `g` is still in it. Take that out first: `w = w - along * g`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie schaalt `w` naar lengte 1, en het deel langs `g` zit er nog in. Haal dat er eerst uit: `w = w - along * g`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: take the part along `g`, `along * g`, out of `w`, and give `w` back scaled to length 1.",
                      "nl": "Vervang de `...` door twee regels: haal het deel langs `g`, `along * g`, uit `w`, en geef `w` terug geschaald naar lengte 1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Neutralize takes the part along `g` out of `w`, `w - along * g`, and scales the rest to length 1.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neutralize haalt het deel langs `g` uit `w`, `w - along * g`, en schaalt de rest naar lengte 1."},
            "almost": {"en": "Right for `neutralize(vec[\"nurse\"], g)`. But `{call}` gives {got}, and it should give {want}. Does your function use `w` and `g`, and no numbers of its own?",
                       "nl": "Goed voor `neutralize(vec[\"nurse\"], g)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `w` en `g`, en geen eigen getallen?"},
            "correct": {"en": "Your `neutralize` is right in all {n} cases. The picture takes the gender direction out of every gender-neutral word with it, and measures again.",
                        "nl": "Je `neutralize` klopt in alle {n} gevallen. Het plaatje haalt er de genderrichting mee uit elk genderneutraal woord, en meet opnieuw."},
        },
    ),
}

_JOHN = float(association(_V["John"], _L["CAREER"], _L["FAMILY"]))
_MEN_WOMEN = float(effect_size(_L["MEN"], _L["WOMEN"], _L["CAREER"], _L["FAMILY"]))

CASES = [
    ("association", _A_START, "empty", None),
    ("association", _A_HEAD + "    to_b = []\n    ...\n" + _A_RET, "wrong", "dots-left"),
    ("association", _A, "correct", None),
    ("association", _a(ret="    return sum(to_a) / len(to_a) - sum(to_b) / len(to_b)\n"), "correct", None),
    ("association", _a(loop="    to_b = []\n    for b in B:\n        to_b.append(np.dot(w, b) / (np.linalg.norm(w) * np.linalg.norm(b)))\n"),
     "correct", None),
    ("association", _a(ret="    return np.mean(to_b) - np.mean(to_a)\n"), "wrong", "flipped"),
    ("association", _a(ret="    return np.sum(to_a) - np.sum(to_b)\n"), "wrong", "sum"),
    ("association", _a(loop="", ret="    return np.mean(to_a)\n"), "wrong", "only-a"),
    ("association", _a(loop=_A_LOOP.replace("cosine(w, b)", "np.dot(w, b)")), "wrong", "dot"),
    ("association", _a(loop=_A_LOOP.replace("cosine(w, b)", "np.dot(w, b)")).replace("cosine(w, a)", "np.dot(w, a)"),
     "wrong", "dot-both"),
    ("association", _a(ret="    return np.mean(to_a) + np.mean(to_b)\n"), "wrong", None),
    ("association", _a(ret="    return\n"), "wrong", "unreached"),
    ("association", _a(ret=f"    return {_JOHN!r}\n"), "almost", None),
    ("effect", _E_START, "empty", None),
    ("effect", _E_HEAD.replace("        s_y.append(association(y, A, B))\n", "        ...\n") + _E_RET, "wrong", "dots-left"),
    ("effect", _E, "correct", None),
    ("effect", _E_HEAD + "    both = s_x + s_y\n    return (np.mean(s_x) - np.mean(s_y)) / np.std(both)\n", "correct", None),
    ("effect", _E_HEAD + "    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y, ddof=1)\n", "wrong", "sample"),
    ("effect", _E_HEAD + "    return (np.sum(s_x) - np.sum(s_y)) / np.std(s_x + s_y)\n", "wrong", "sums"),
    ("effect", _E_HEAD + "    return np.mean(s_x) - np.mean(s_y)\n", "wrong", "no-std"),
    ("effect", _E_HEAD + "    return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x)\n", "wrong", "std-x"),
    ("effect", _E_HEAD + "    return np.mean(s_x) - np.mean(s_y) / np.std(s_x + s_y)\n", "wrong", "brackets"),
    ("effect", _E_HEAD + "    return (np.mean(s_y) - np.mean(s_x)) / np.std(s_x + s_y)\n", "wrong", "flipped"),
    ("effect", _E_HEAD + "    return np.std(s_x + s_y)\n", "wrong", None),
    ("effect", _E_HEAD + "    return\n", "wrong", "no-return"),
    ("effect", _E_HEAD + f"    return {_MEN_WOMEN!r}\n", "almost", None),
    ("neutralize", _N_START, "empty", None),
    ("neutralize", _N_HEAD + "    w = w - along * g\n    ...\n", "wrong", "dots-left"),
    ("neutralize", _N, "correct", None),
    ("neutralize", _N_HEAD + "    rest = w - np.dot(w, g) * g\n    return rest / np.linalg.norm(rest)\n", "correct", None),
    ("neutralize", _N_HEAD + "    return w - along * g\n", "wrong", "no-scale"),
    ("neutralize", _N_HEAD + "    w = w + along * g\n    return w / np.linalg.norm(w)\n", "wrong", "plus"),
    ("neutralize", _N_HEAD + "    return along * g\n", "wrong", "along-only"),
    ("neutralize", _N_HEAD + "    return w / np.linalg.norm(w)\n", "wrong", "scale-only"),
    ("neutralize", _N_HEAD + "    w = w - along\n    return w / np.linalg.norm(w)\n", "wrong", None),
    ("neutralize", _N_HEAD + "    w = w - along * g\n", "wrong", "no-return"),
    ("neutralize", _N_HEAD + "    return neutral\n", "error", None),
    ("neutralize", _N_HEAD + "    return np.array(" + repr([float(x) for x in neutralize(_V["nurse"], _G)]) + ")\n", "almost", None),
]
