# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-llm-or-small-model: a function that trains a small model (TF-IDF and logistic regression) on sorted
questions of customers of a bank, and one that works out the macro F1; the course's measurements on the 120 test
questions of data/banking77-small.json (the learning curve, the speed, the price), against Qwen3's recorded run of
build-ai-services (data/course-apis-qwen.json), their pictures, and the demo with the learner's own small model."""
import html
import json
import math
import time

import numpy as np

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, kept, run_code

DATA = paths.ROOT / "data" / "course-apis-qwen.json"
BANKING = paths.ROOT / "data" / "banking77-small.json"
PRICE_IN, PRICE_OUT = 3.0, 15.0   # made-up prices, in dollars per million tokens, as in build-tokens-prices
SIZES = (2, 5, 10, 20, 30, 50)    # sorted questions per label, for the learning curve


def banking() -> dict:
    return json.loads(BANKING.read_text(encoding="utf-8"))


LABELS = tuple(banking()["labels"])


class Named(list):
    """A list of the lesson's questions or labels, written in the check's messages by its name, such as texts[:60]."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


def train() -> tuple[Named, Named]:
    b = banking()["train"]
    return Named([t for _, t in b], "texts"), Named([lab for lab, _ in b], "labels")


def test() -> tuple[Named, Named]:
    b = banking()["test"]
    return Named([t for _, t in b], "test_texts"), Named([lab for lab, _ in b], "test_labels")


def first(n: int) -> tuple[list, list]:
    """The first n training questions of each label, in the order of the file, and their labels."""
    texts, labels, count = [], [], {lab: 0 for lab in LABELS}
    for lab, t in banking()["train"]:
        if count[lab] < n:
            texts.append(t)
            labels.append(lab)
            count[lab] += 1
    return texts, labels


def qwen_answers() -> list:
    """Qwen3's answers to the 120 test questions in the course's run of build-ai-services."""
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-ai-services"]["test"]["answers"]


def _parse(answer, labels):
    best = None
    for label in labels:
        if label in answer.lower() and (best is None or len(label) > len(best)):
            best = label
    return best


def qwen_labels() -> Named:
    """The labels Qwen3 gave, read as build-ai-services reads them; "none" where an answer had no label."""
    return Named([_parse(a["text"], LABELS) or "none" for a in qwen_answers()], "qwen_labels")


# --- the course's small model and its measurements ----------------------------------------------------------------------

SOLUTION_TRAIN = '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_small(texts, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    return vectorizer, model
'''


def _course_train():
    return run_code(SOLUTION_TRAIN).names["train_small"]


def predict(pair, texts) -> list:
    vectorizer, model = pair
    return [str(p) for p in model.predict(vectorizer.transform(list(texts)))]


def _f1(true, predicted, labels=LABELS) -> float:
    true, predicted = np.array(true), np.array(predicted)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (predicted == label))
        fp = np.sum((true != label) & (predicted == label))
        fn = np.sum((true == label) & (predicted != label))
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def curve() -> list[tuple[int, float]]:
    """The macro F1 on the 120 test questions of the course's small model, trained on the first n questions of each
    label, for each n of SIZES."""
    def make():
        test_texts, test_labels = test()
        fn = _course_train()
        return [(n, _f1(test_labels, predict(fn(*first(n)), test_texts))) for n in SIZES]
    return kept(("build-llm-or-small-model", "curve"), make)


def facts() -> dict:
    """The numbers the lesson gives: the macro F1 and right answers of Qwen3 and of the small model on all 50 per label,
    Qwen3's seconds and tokens, and the made-up price of a million messages."""
    def make():
        test_texts, test_labels = test()
        q = qwen_labels()
        small = predict(_course_train()(*train()), test_texts)
        a = qwen_answers()
        tokens_in = sum(x["prompt_tokens"] for x in a) / len(a)
        tokens_out = sum(x["tokens"] for x in a) / len(a)
        return {"qwen_f1": _f1(test_labels, q), "qwen_right": sum(x == y for x, y in zip(q, test_labels)),
                "small_f1": _f1(test_labels, small), "small_right": sum(x == y for x, y in zip(small, test_labels)),
                "qwen_seconds": sum(x["seconds"] for x in a), "tokens_in": tokens_in, "tokens_out": tokens_out,
                "million": 1e6 * (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6}
    return kept(("build-llm-or-small-model", "facts"), make)


def size() -> dict:
    """The course's small model, trained on the 300 questions: the words it knows, and its size as Python saves it."""
    import pickle
    pair = _course_train()(*train())
    return {"words": len(pair[0].vocabulary_), "weights": pair[1].coef_.size, "bytes": len(pickle.dumps(pair))}


def speed() -> dict:
    """How long the course's small model takes on this computer: training on the 300 questions, and sorting the 120."""
    fn = _course_train()
    texts, labels = train()
    start = time.perf_counter()
    pair = fn(texts, labels)
    trained = time.perf_counter() - start
    start = time.perf_counter()
    predict(pair, test()[0])
    return {"train": trained, "sort": time.perf_counter() - start}


def _nr(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


_CURVE = {
    "en": {"title": "Macro F1 on the 120 test questions", "x": "sorted questions per label the small model learned from",
           "small": "small model (TF-IDF and logistic regression)", "qwen": "Qwen3-0.6B with a prompt, no examples",
           "alt": "A line chart: the macro F1 of the small model rises with the number of sorted questions per label it learned from, from below Qwen3's horizontal line with 2 and 5 per label to above it from 10 per label on."},
    "nl": {"title": "Macro F1 op de 120 testvragen", "x": "gesorteerde vragen per label waarvan het kleine model leerde",
           "small": "klein model (TF-IDF en logistische regressie)", "qwen": "Qwen3-0.6B met een prompt, zonder voorbeelden",
           "alt": "Een lijngrafiek: de macro F1 van het kleine model stijgt met het aantal gesorteerde vragen per label waarvan het leerde, van onder de horizontale lijn van Qwen3 bij 2 en 5 per label tot erboven vanaf 10 per label."},
}


def curve_svg(points: list[tuple[int, float]], qwen: float, lang: str, mine: tuple[int, float] | None = None) -> str:
    """The learning curve: the small model's macro F1 for each number of questions per label, Qwen3's as a dashed line,
    and with `mine` a dot of the learner's own model."""
    w = _CURVE[lang]
    W, H, left, right, top, bottom = 600, 308, 52, 20, 18, 82
    lo, hi = 0.5, 1.0
    xmax = max(n for n, _ in points)

    def X(n):
        return left + (W - left - right) * n / xmax

    def Y(v):
        return top + (H - top - bottom) * (hi - v) / (hi - lo)
    grid = ""
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        grid += (f'<line x1="{left}" x2="{W - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--ea-line)" stroke-width="0.6"/>'
                 f'<text x="{left - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--ea-ink-2)">{_nr(v, lang, 1)}</text>')
    for n, _ in points:
        grid += f'<text x="{X(n):.1f}" y="{H - bottom + 16}" text-anchor="middle" font-size="11" fill="var(--ea-ink-2)">{n}</text>'
    path = " ".join(f"{'M' if i == 0 else 'L'} {X(n):.1f} {Y(v):.1f}" for i, (n, v) in enumerate(points))
    dots = "".join(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="3.5" fill="var(--ea-pred)"/>' for n, v in points)
    if mine is not None:
        n, v = mine
        dots += (f'<circle cx="{X(n):.1f}" cy="{Y(max(lo, v)):.1f}" r="7" fill="none" stroke="var(--ea-ink)" stroke-width="1.6"/>')
    e = html.escape
    svg = (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{e(w["alt"])}" style="max-width:{W}px;font-family:var(--ea-font-ui)">'
           + grid
           + f'<line x1="{left}" x2="{W - right}" y1="{Y(qwen):.1f}" y2="{Y(qwen):.1f}" stroke="var(--ea-ink-2)" stroke-width="1.4" stroke-dasharray="5 4"/>'
           + f'<path d="{path}" fill="none" stroke="var(--ea-pred)" stroke-width="2"/>' + dots
           + f'<text x="{(left + W - right) / 2}" y="{H - bottom + 34}" text-anchor="middle" font-size="12" fill="var(--ea-ink-2)">{e(w["x"])}</text>'
           + f'<line x1="{left}" x2="{left + 22}" y1="{H - 28}" y2="{H - 28}" stroke="var(--ea-pred)" stroke-width="2"/>'
           + f'<text x="{left + 28}" y="{H - 24}" font-size="12" fill="var(--ea-ink)">{e(w["small"])}</text>'
           + f'<line x1="{left}" x2="{left + 22}" y1="{H - 8}" y2="{H - 8}" stroke="var(--ea-ink-2)" stroke-width="1.4" stroke-dasharray="5 4"/>'
           + f'<text x="{left + 28}" y="{H - 4}" font-size="12" fill="var(--ea-ink)">{e(w["qwen"])}</text>'
           + '</svg>')
    return svg


def curve_picture(lang: str) -> str:
    w = _CURVE[lang]
    return yours.figure(w["title"], curve_svg(curve(), facts()["qwen_f1"], lang), alt=w["alt"])


# --- exercise 1: train the small model ----------------------------------------------------------------------------------

_START_TRAIN = '''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_small(texts, labels):
    vectorizer = TfidfVectorizer()
    X = ...
    model = LogisticRegression(max_iter=1000)
    ...
    return vectorizer, model
'''


class _Cannot:
    """What the check gets when it cannot sort the test questions with what train_small gave back."""

    def __init__(self, why: str):
        self.why = why

    def __eq__(self, other):
        return isinstance(other, _Cannot) and other.why == self.why

    def __hash__(self):
        return hash(self.why)

    def __repr__(self):
        return f"<{self.why}>"


def _sorted_test(value):
    """What the check compares: the labels the learner's model gives the 120 test questions."""
    if not (isinstance(value, tuple) and len(value) == 2):
        return _Cannot("not-pair")
    vectorizer, model = value
    if hasattr(vectorizer, "predict") and not hasattr(vectorizer, "transform") and hasattr(model, "transform"):
        return _Cannot("swapped")
    try:
        X = vectorizer.transform(list(test()[0]))
    except Exception as e:   # the learner's vectorizer
        return _Cannot("vectorizer-" + type(e).__name__)
    try:
        return [str(p) for p in model.predict(X)]
    except Exception as e:   # the learner's model
        return _Cannot("model-" + type(e).__name__)


def _summary(value):
    if isinstance(value, _Cannot):
        return {"en": "something the check cannot sort the test questions with", "nl": "iets waarmee de check de testvragen niet kan sorteren"}
    right = sum(p == t for p, t in zip(value, test()[1]))
    return {"en": f"a model that sorts {right} of the 120 test questions right", "nl": f"een model dat {right} van de 120 testvragen goed sorteert"}


def train_picture(seen) -> str:
    """The learner's model, trained on the 300 questions: its macro F1 on the 120 test questions beside Qwen3's, on the
    course's learning curve."""
    run = seen.mine
    fn = run.names.get("train_small") if run is not None else None
    if not callable(fn):
        return ""
    title = {"en": "Your small model on the 120 test questions", "nl": "Je kleine model op de 120 testvragen"}[seen.lang]
    try:
        start = time.perf_counter()
        pair = fn(*train())
        took = time.perf_counter() - start
        got = predict(pair, test()[0])
    except Exception:   # the learner's code
        return ""
    f1 = _f1(test()[1], got)
    note = {"en": "Your model, trained on all 50 questions per label in {s} seconds (the circle): macro F1 {f1}, against Qwen3's {q}.",
            "nl": "Je model, getraind op alle 50 vragen per label in {s} seconden (de cirkel): macro F1 {f1}, tegenover {q} van Qwen3."}[seen.lang]
    note = note.format(s=_nr(took, seen.lang, 2), f1=_nr(f1, seen.lang), q=_nr(facts()["qwen_f1"], seen.lang))
    body = curve_svg(curve(), facts()["qwen_f1"], seen.lang, mine=(50, f1)) + (
        f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(note)}</div>')
    return yours.figure(title, body, alt=title)


# --- exercise 2: the macro F1 -------------------------------------------------------------------------------------------

_START_F1 = '''import numpy as np


def macro_f1(true, predicted, labels):
    true, predicted = np.array(true), np.array(predicted)
    scores = []
    for label in labels:
        tp = np.sum((true == label) & (predicted == label))
        fp = np.sum((true != label) & (predicted == label))
        fn = np.sum((true == label) & (predicted != label))
        scores.append(...)
    return ...
'''
SOLUTION_F1 = _START_F1.replace("scores.append(...)", "scores.append(2 * tp / (2 * tp + fp + fn))").replace(
    "    return ...", "    return np.mean(scores)")
_ACCURACY = SOLUTION_F1.replace("    return np.mean(scores)", "    return np.mean(true == predicted)")
_SUM = SOLUTION_F1.replace("    return np.mean(scores)", "    return np.sum(scores)")
_ONE = SOLUTION_F1.replace("    return np.mean(scores)", "        return np.mean(scores)")
_PLAIN = SOLUTION_F1.replace("2 * tp / (2 * tp + fp + fn)", "(tp / (tp + fp) + tp / (tp + fn)) / 2")
_HARMONIC = SOLUTION_F1.replace("        scores.append(2 * tp / (2 * tp + fp + fn))",
                                "        p, r = tp / (tp + fp), tp / (tp + fn)\n        scores.append(2 * p * r / (p + r))")


def _nan(got, text) -> bool:
    try:
        return math.isnan(float(got))
    except (TypeError, ValueError):
        return False


def _four(v) -> str:
    """A score as the messages show it: four decimals, without float noise."""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return repr(v)


def f1_picture(seen) -> str:
    """The learner's macro_f1 on Qwen3's labels and on the course's small model's."""
    run = seen.mine
    fn = run.names.get("macro_f1") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your macro_f1 on the 120 test questions", "Qwen3 with a prompt", "the small model, 50 per label"),
         "nl": ("Je macro_f1 op de 120 testvragen", "Qwen3 met een prompt", "het kleine model, 50 per label")}[seen.lang]
    test_texts, test_labels = test()
    rows = []
    try:
        for name, got in ((w[1], qwen_labels()), (w[2], predict(_course_train()(*train()), test_texts))):
            rows.append((name, float(fn(list(test_labels), list(got), list(LABELS)))))
    except Exception:   # the learner's code
        return ""
    body = ""
    for name, v in rows:
        width = max(0.0, min(1.0, v)) * 100 if v == v else 0
        body += (f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;font-family:var(--ea-font-ui);font-size:12px">'
                 f'<span style="width:12rem;color:var(--ea-ink-2)">{html.escape(name)}</span>'
                 f'<span style="flex:1;max-width:18rem;background:var(--ea-inset);border-radius:3px;height:12px">'
                 f'<span style="display:block;width:{width:.1f}%;height:12px;border-radius:3px;background:var(--ea-ink-2)"></span></span>'
                 f'<span style="color:var(--ea-ink);font-family:var(--ea-font-code)">{_nr(v, seen.lang)}</span></div>')
    return yours.figure(w[0], body, alt=w[0])


_TINY = (Named(["a", "a", "b", "c"], '["a", "a", "b", "c"]'), Named(["a", "b", "b", "b"], '["a", "b", "b", "b"]'),
         Named(["a", "b", "c"], '["a", "b", "c"]'))


def _cases_f1():
    test_texts, test_labels = test()
    small = Named(predict(_course_train()(*train()), test_texts), "small_labels")
    labels = Named(list(LABELS), "LABELS")
    return [(test_labels, qwen_labels(), labels), (test_labels, small, labels), (test_labels, test_labels, labels)]


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "train_small": FunctionBox(
        name="train_small",
        picture=train_picture,
        function="train_small",
        start=_START_TRAIN,
        solution=SOLUTION_TRAIN,
        visible=[train()],
        hidden=[(Named(first(10)[0], "texts_10"), Named(first(10)[1], "labels_10"))],
        collect=_sorted_test,
        show=_summary,
        time_limit=30.0,
        mistakes=[
            Mistake("not-fitted", wrong=SOLUTION_TRAIN.replace("    model.fit(X, labels)\n", ""),
                    en="`{call}` gives a model that has not learned anything yet: the check cannot sort with it. Train it on `X` and `labels`, with `.fit`.",
                    nl="`{call}` geeft een model dat nog niets geleerd heeft: de check kan er niet mee sorteren. Train het op `X` en `labels`, met `.fit`."),
            Mistake("raw-texts", error=lambda e: isinstance(e, ValueError) and "string" in str(e).lower(),
                    en="With `{call}`, `LogisticRegression` gets texts, and it learns from numbers. Give it `X`, the TF-IDF vectors that `vectorizer.fit_transform(texts)` made.",
                    nl="Met `{call}` krijgt `LogisticRegression` teksten, en het leert van getallen. Geef het `X`, de TF-IDF-vectoren die `vectorizer.fit_transform(texts)` maakte."),
            Mistake("not-pair", matches=lambda got, text: got == _Cannot("not-pair"),
                    en="`{call}` gives back something else than the two things the check needs: the vectorizer and the model, as `return vectorizer, model`.",
                    nl="`{call}` geeft iets anders terug dan de twee dingen die de check nodig heeft: de vectorizer en het model, als `return vectorizer, model`."),
            Mistake("swapped", matches=lambda got, text: got == _Cannot("swapped"),
                    en="`{call}` gives back the model first and the vectorizer second. The check takes them in the order of the task: `return vectorizer, model`.",
                    nl="`{call}` geeft eerst het model terug en dan de vectorizer. De check neemt ze in de volgorde van de opdracht: `return vectorizer, model`."),
            Mistake("vectorizer-not-fitted", matches=lambda got, text: got == _Cannot("vectorizer-NotFittedError"),
                    en="`{call}` gives a vectorizer that has not seen any texts yet, so it cannot turn the test questions into vectors. Make `X` with `vectorizer.fit_transform(texts)`: that learns the words and makes the vectors in one go.",
                    nl="`{call}` geeft een vectorizer die nog geen teksten gezien heeft, dus hij kan de testvragen niet in vectoren omzetten. Maak `X` met `vectorizer.fit_transform(texts)`: dat leert de woorden en maakt de vectoren in één keer."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: `X`, the TF-IDF vectors of the texts, and the line that trains the model on them.",
                      "nl": "Vervang de twee `...`: `X`, de TF-IDF-vectoren van de teksten, en de regel die het model erop traint."},
            "correct": {"en": "Right: your `train_small` learns the words of the questions, turns them into vectors, and trains a logistic regression on them, in a fraction of a second. The picture shows how it does against Qwen3.",
                        "nl": "Goed: je `train_small` leert de woorden van de vragen, maakt er vectoren van, en traint er een logistische regressie op, in een fractie van een seconde. Het plaatje laat zien hoe hij het doet tegenover Qwen3."},
            "wrong": {"en": "`{call}` gives {got}, and the course's gives {want}.",
                      "nl": "`{call}` geeft {got}, en die van de course geeft {want}."},
            "almost": {"en": "Right for the 300 questions. But `{call}` gives {got}, and the course's gives {want}.",
                       "nl": "Goed voor de 300 vragen. Maar `{call}` geeft {got}, en die van de course geeft {want}."},
        },
    ),
    "macro_f1": FunctionBox(
        name="macro_f1",
        picture=f1_picture,
        function="macro_f1",
        start=_START_F1,
        solution=SOLUTION_F1,
        visible=[_TINY],
        hidden=_cases_f1(),
        show=_four,
        mistakes=[
            Mistake("accuracy", wrong=_ACCURACY,
                    en="`{call}` gives `{got}`: that is the part of all answers that is right, the accuracy. The macro F1 is the mean of the F1 of each label: work out each label's F1 in the loop, and give back their mean.",
                    nl="`{call}` geeft `{got}`: dat is het deel van alle antwoorden dat goed is, de accuracy. De macro F1 is het gemiddelde van de F1 van elk label: reken in de loop de F1 van elk label uit, en geef hun gemiddelde terug."),
            Mistake("sum", wrong=_SUM,
                    en="`{call}` gives `{got}`, more than 1: that is the sum of the F1 scores of the labels. Give back their mean.",
                    nl="`{call}` geeft `{got}`, meer dan 1: dat is de som van de F1 scores van de labels. Geef hun gemiddelde terug."),
            Mistake("one", wrong=_ONE,
                    en="`{call}` gives `{got}`: the F1 of the first label only, because the `return` is inside the loop. Move it back, so it comes after the loop.",
                    nl="`{call}` geeft `{got}`: alleen de F1 van het eerste label, omdat de `return` in de loop staat. Zet hem terug, zodat hij na de loop komt."),
            Mistake("plain-mean", wrong=_PLAIN,
                    en="`{call}` gives `{got}`: you take the plain mean of precision and recall. The F1 is their harmonic mean, `2 * tp / (2 * tp + fp + fn)`.",
                    nl="`{call}` geeft `{got}`: je neemt het gewone gemiddelde van precision en recall. De F1 is hun harmonisch gemiddelde, `2 * tp / (2 * tp + fp + fn)`."),
            Mistake("no-precision", matches=_nan,
                    en="`{call}` gives `nan`: a label the model never gives has no precision, because that is 0 divided by 0. Work out the F1 straight from the counts, `2 * tp / (2 * tp + fp + fn)`: it gives 0 for such a label, and the same as `2 * p * r / (p + r)` for the others.",
                    nl="`{call}` geeft `nan`: een label dat het model nooit geeft, heeft geen precision, want dat is 0 gedeeld door 0. Reken de F1 direct uit de tellingen uit, `2 * tp / (2 * tp + fp + fn)`: dat geeft 0 voor zo'n label, en voor de andere hetzelfde als `2 * p * r / (p + r)`."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Replace the two `...`: the F1 of the label from `tp`, `fp` and `fn`, and the mean of all scores.",
                      "nl": "Vervang de twee `...`: de F1 van het label uit `tp`, `fp` en `fn`, en het gemiddelde van alle scores."},
            "correct": {"en": "Right: your `macro_f1` works out the F1 of each label, and gives back their mean, so every label counts as much, however often it comes up.",
                        "nl": "Goed: je `macro_f1` rekent de F1 van elk label uit, en geeft hun gemiddelde terug, zodat elk label even zwaar telt, hoe vaak het ook voorkomt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["train_small", "macro_f1", "when-llm", "test-set"]

CASES = [
    ("train_small", _START_TRAIN, "empty", None),
    ("train_small", SOLUTION_TRAIN, "correct", None),
    ("train_small", SOLUTION_TRAIN.replace("    model.fit(X, labels)\n", ""), "wrong", "not-fitted"),
    ("train_small", SOLUTION_TRAIN.replace("model.fit(X, labels)", "model.fit(texts, labels)"), "error", "raw-texts"),
    ("train_small", SOLUTION_TRAIN.replace("return vectorizer, model", "return model"), "wrong", "not-pair"),
    ("train_small", SOLUTION_TRAIN.replace("return vectorizer, model", "return model, vectorizer"), "wrong", "swapped"),
    ("train_small", SOLUTION_TRAIN.replace("X = vectorizer.fit_transform(texts)", "X = TfidfVectorizer().fit_transform(texts)"),
     "wrong", "vectorizer-not-fitted"),
    ("macro_f1", _START_F1, "empty", None),
    ("macro_f1", SOLUTION_F1, "correct", None),
    ("macro_f1", _ACCURACY, "wrong", "accuracy"),
    ("macro_f1", _SUM, "wrong", "sum"),
    ("macro_f1", _ONE, "wrong", "one"),
    ("macro_f1", _PLAIN, "wrong", "plain-mean"),
    ("macro_f1", _HARMONIC, "wrong", "no-precision"),
]


# --- the demo: the learner's small model, on fewer or more sorted questions ------------------------------------------------

def learner_train(text):
    """The learner's train_small from exercise 1, when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["train_small"].check(text).state != "correct":
        return None
    return run_code(text).names.get("train_small")


def demo(train_small, n: int, message: str, words: dict, lang: str) -> str:
    """The learner's train_small on the first n questions of each label: its macro F1 on the 120 test questions, on the
    course's curve, and the label it gives the learner's own message."""
    texts, labels = first(n)
    start = time.perf_counter()
    try:
        pair = train_small(texts, labels)
        got = predict(pair, test()[0])
        mine = predict(pair, [message])[0] if message.strip() else None
    except Exception as e:   # the learner's code
        return f"<p><em>{html.escape(words['failed'].format(why=f'{type(e).__name__}: {e}'))}</em></p>"
    took = time.perf_counter() - start
    f1 = _f1(test()[1], got)
    note = words["note"].format(n=n, total=6 * n, f1=_nr(f1, lang), q=_nr(facts()["qwen_f1"], lang), s=_nr(took, lang, 2))
    said = "" if mine is None else (f'<div style="font-family:var(--ea-font-ui);font-size:13px;margin-top:0.4rem;color:var(--ea-ink)">'
                                    f'{html.escape(words["label"])} <span style="font-family:var(--ea-font-code);color:var(--ea-pred-text);font-weight:600">{html.escape(mine)}</span></div>')
    body = (curve_svg(curve(), facts()["qwen_f1"], lang, mine=(n, f1))
            + f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(note)}</div>' + said)
    return yours.figure(words["title"], body, alt=words["title"])
