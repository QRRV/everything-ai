# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-competitions: a made-up competition with 1,000 test examples, 300 of them on the public
leaderboard and 700 on the private one, and submissions that guess every label with a coin flip, made with NumPy's
generator and seed 0 as the worked example makes them; the best public score of the first k submissions and the
private score of the same submission; the majority vote of the submissions that scored above 50% on the public part;
and the functions best_of and vote. The numbers in capitals were measured once; tests/test_ml_competitions.py
measures them again, also those of the real models on the traffic of ml-gbdt-default."""
import contextlib
import functools
import io
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

N, PUBLIC, SUBMISSIONS = 1000, 300, 1000     # test examples, of which on the public leaderboard; submissions made
SLIDER = [1, 10, 100, 200, 1000]             # the slider: how many of the submissions are sent in
# For the first k submissions: which one is best on the public part, its public score and its private score, and the
# best private score among them.
BEST = {1: (0, 0.493, 0.514, 0.514), 10: (9, 0.537, 0.489, 0.540), 100: (70, 0.567, 0.517, 0.541),
        200: (115, 0.570, 0.486, 0.564), 1000: (767, 0.613, 0.544, 0.567)}
# The majority vote of the first k submissions that scored above 50% on the public part: how many there are, and the
# public and private score of the vote.
VOTE = {10: (7, 0.550, 0.530), 100: (46, 0.633, 0.497), 200: (100, 0.670, 0.509), 1000: (467, 0.840, 0.499)}

# The real models: 24 settings of HistGradientBoostingRegressor on the traffic of ml-gbdt-default, with the column for
# the holidays, scored by MAE on the 15,246 test hours, split at random into a public and a private part, 200 times
# for each size of the public part: how often the best on the public part is also the best on the private part, and
# how much higher its private MAE is than its public MAE, on average.
REAL_SETTINGS = 24
SPLITS = 200                # random splits of the test hours for every size of the public part
TEST_HOURS, VAL_HOURS = 15_246, 7_837   # the test hours of ml-gbdt-default, and its validation hours of 2016
# The worked example: the mean public and private score of all 1,000 submissions; of the first 200, how many score
# exactly 0.5 and how many above it on the public part, and the place of the winner on the private part, tied with how
# many in all; the standard deviation of a coin's score on 300 and on 700 examples, in percentage points.
MEANS = (0.499, 0.5)
AT_HALF_200, ABOVE_HALF_200 = 2, 100
PRIVATE_PLACE, TIED = 159, 3
SPREAD = {300: 2.9, 700: 1.9}
REAL = {100: (0.10, 6.5), 300: (0.32, 3.4), 1000: (0.55, 1.4), 4500: (0.75, -0.5)}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def percent(x: float, lang: str) -> str:
    return number(100 * x, lang, 1) + "%"


@functools.lru_cache(maxsize=1)
def competition():
    """The competition of the worked example: the hidden labels, the positions of the public and the private part, and
    1,000 submissions of coin flips."""
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, N)
    order = rng.permutation(N)
    public, private = order[:PUBLIC], order[PUBLIC:]
    submissions = rng.integers(0, 2, (SUBMISSIONS, N))
    return y, public, private, submissions


@functools.lru_cache(maxsize=1)
def scores():
    """The public and the private score, the share right, of every submission."""
    y, public, private, submissions = competition()
    right = submissions == y
    return right[:, public].mean(axis=1), right[:, private].mean(axis=1)


def course_vote(submissions, public_scores):
    """The course's vote: the majority of the submissions with a public score above 0.5, for every example."""
    lucky = submissions[public_scores > 0.5]
    votes = lucky.mean(axis=0)
    return votes > 0.5


def vote_scores(prediction) -> tuple:
    """The public and the private score of one prediction for every example."""
    y, public, private, _ = competition()
    right = prediction == y
    return float(right[public].mean()), float(right[private].mean())


# --- the slider -----------------------------------------------------------------------------------------------------

def _jitter(i: int) -> float:
    return ((i * 7919) % 101) / 101 - 0.5


def strip_picture(k: int, labels: dict, mark: float | None = None) -> str:
    """Every one of the first k submissions as a dot on two rows, its public score above and its private score below,
    with the best on the public part and the line to its private score drawn thick. `mark`, a private score, adds a
    ring on the private row."""
    import numpy as np
    pub, priv = scores()
    b = int(np.argmax(pub[:k]))
    p = chart.Panel((0.40, 0.65), (-0.6, 1.6), title=labels["title"], xlabel=labels["x"], xticks=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
                    xtick_labels=["40%", "45%", "50%", "55%", "60%", "65%"], yticks=[])
    p.line([0.5, 0.5], [-0.6, 1.6], role="ink-3", width=0.8, dashed=True)
    r = 3.0 if k <= 200 else 2.0
    xs = [float(v) for v in pub[:k]]
    p.points(xs, [1 + 0.36 * _jitter(i) for i in range(k)], role="ink-3", r=r)
    p.points([float(v) for v in priv[:k]], [0.36 * _jitter(i) for i in range(k)], role="ink-3", r=r)
    p.line([float(pub[b]), float(priv[b])], [1 + 0.36 * _jitter(b), 0.36 * _jitter(b)], role="ink", width=2.2)
    p.points([float(pub[b])], [1 + 0.36 * _jitter(b)], role="ink", r=5.5)
    p.points([float(priv[b])], [0.36 * _jitter(b)], role="ink", r=5.5)
    if mark is not None:
        p.points([mark], [0.36 * _jitter(b)], role="ink", r=8, hollow=True)
    p.text(0.405, 1.42, labels["public"], role="ink-2", anchor="start")
    p.text(0.405, 0.42, labels["private"], role="ink-2", anchor="start")
    return chart.figure([p], alt=labels["alt"], width=460, height=250)


def strip_said(k: int, lang: str) -> str:
    """The sentence under the picture of the slider."""
    _, pub, priv, best_priv = BEST[k]
    if lang == "nl":
        if k == 1:
            return (f"Eén inzending: {percent(pub, lang)} goed op het publieke deel, {percent(priv, lang)} op het "
                    f"private deel. Een munt per voorbeeld zit rond de 50%.")
        return (f"De beste van {number(k, lang)} op het publieke deel: {percent(pub, lang)} goed. Dezelfde inzending op "
                f"het private deel: {percent(priv, lang)}. De beste op het private deel had {percent(best_priv, lang)}, "
                f"maar dat kon niemand zien.")
    if k == 1:
        return (f"One submission: {percent(pub, lang)} right on the public part, {percent(priv, lang)} on the private "
                f"part. A coin per example is around 50%.")
    return (f"The best of {number(k, lang)} on the public part: {percent(pub, lang)} right. The same submission on the "
            f"private part: {percent(priv, lang)}. The best on the private part had {percent(best_priv, lang)}, but "
            f"nobody could see that.")


# The lesson is done when these are correct. "final-pick" is the choice question in the lesson.
REQUIRED = ["best-of", "vote", "final-pick"]


class _Val:
    """A test value that the check's messages write by its name, such as `public_scores`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value)


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


_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: any("..." in line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _arrays(name_pub: str, name_priv: str, pub, priv):
    import numpy as np
    return _Val(name_pub, np.array(pub)), _Val(name_priv, np.array(priv))


# --- exercise 1: the best on the public part -------------------------------------------------------------------------

_PUB200, _PRIV200 = (_Val(n, v) for n, v in zip(["public_scores", "private_scores"], (s[:200] for s in scores())))
_SMALL = _arrays("public_small", "private_small", [0.52, 0.61, 0.48, 0.55], [0.50, 0.47, 0.53, 0.58])
_FIRST = _arrays("public_first", "private_first", [0.70, 0.60, 0.65], [0.66, 0.68, 0.61])
_LAST = _arrays("public_last", "private_last", [0.40, 0.45, 0.50, 0.58, 0.62], [0.52, 0.49, 0.51, 0.47, 0.55])

_B_HEAD = "import numpy as np\n\n\ndef best_of(public_scores, private_scores):\n"


def _b(best: str = "np.argmax(public_scores)", back: str = "private_scores[best]") -> str:
    return _B_HEAD + f"    best = {best}\n    return {back}\n"


_B_START = _b("...", "...")
_B = _b()


def _close(got, want) -> bool:
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False


def _shown_score(v):
    import numbers

    import numpy as np
    if isinstance(v, (numbers.Integral, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    if isinstance(v, np.ndarray) and v.ndim > 0:
        return {"en": f"an array of {number(v.size, 'en')} values", "nl": f"een array met {number(v.size, 'nl')} waarden"}
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return repr(v)[:60]


_BO_WORDS = {
    "title": {"en": "Your pick among the first 200 submissions", "nl": "Jouw keuze uit de eerste 200 inzendingen"},
    "x": {"en": "share right", "nl": "aandeel goed"},
    "public": {"en": "public part", "nl": "publieke deel"},
    "private": {"en": "private part", "nl": "private deel"},
    "right": {"en": "Every dot is one of the first 200 submissions, its public score above and its private score below. Your function picks the dark one: 57.0% right on the public part, the best of all 200, and 48.6% on the private part, a little worse than a coin. Its lead on the public part was luck.",
              "nl": "Elke stip is een van de eerste 200 inzendingen, de publieke score boven en de private score onder. Jouw functie kiest de donkere: 57,0% goed op het publieke deel, de beste van alle 200, en 48,6% op het private deel, iets slechter dan een munt. Die voorsprong op het publieke deel was geluk."},
    "said": {"en": "Every dot is one of the first 200 submissions, its public score above and its private score below. Your function gives {got}, the ring; the course's function picks the dark one, the best on the public part, and gives its private score, 48.6%.",
             "nl": "Elke stip is een van de eerste 200 inzendingen, de publieke score boven en de private score onder. Jouw functie geeft {got}, de ring; die van de course kiest de donkere, de beste op het publieke deel, en geeft de private score daarvan: 48,6%."},
    "said_off": {"en": "Every dot is one of the first 200 submissions, its public score above and its private score below. Your function gives {got}; the course's function picks the dark one, the best on the public part, and gives its private score, 48.6%.",
                 "nl": "Elke stip is een van de eerste 200 inzendingen, de publieke score boven en de private score onder. Jouw functie geeft {got}; die van de course kiest de donkere, de beste op het publieke deel, en geeft de private score daarvan: 48,6%."},
}


def _bo_picture(seen) -> str:
    lang = seen.lang
    got = _call(seen, (_PUB200, _PRIV200))
    try:
        value = float(got)
    except (TypeError, ValueError):
        return ""
    words = {k: v[lang] for k, v in _BO_WORDS.items()}
    if _close(value, scores()[1][BEST[200][0]]):
        text, mark = words["right"], None
    else:
        mark = value if 0.40 <= value <= 0.65 else None
        shown = percent(value, lang) if 0 <= value <= 1 else _shown_score(got)
        text = words["said" if mark is not None else "said_off"].format(got=shown)
    labels = {"title": "", "x": words["x"], "public": words["public"], "private": words["private"], "alt": text}
    return yours.figure(words["title"], strip_picture(200, labels, mark) + _key(text), alt=text)


EXERCISES = {
    "best-of": FunctionBox(
        name="best-of",
        picture=_bo_picture,
        function="best_of",
        start=_B_START,
        solution=_B,
        show=_shown_score,
        compare=_close,
        visible=[(_PUB200, _PRIV200)],
        hidden=[_SMALL, _FIRST, _LAST],
        first=[_DOTS],
        mistakes=[
            Mistake("best-private", wrong=_b("np.argmax(private_scores)"),
                    en="`{call}` gives {got}: the best score on the private part. That is the one you want, but nobody can see it until the competition ends. Pick by what you can see, the public scores: `np.argmax(public_scores)`.",
                    nl="`{call}` geeft {got}: de beste score op het private deel. Die wil je hebben, maar niemand ziet die voordat de wedstrijd afloopt. Kies op wat je kunt zien, de publieke scores: `np.argmax(public_scores)`."),
            Mistake("public-back", wrong=_b(back="public_scores[best]"),
                    en="`{call}` gives {got}: the public score of the best submission, the number on the leaderboard. The question is what that same submission scores on the private part: `private_scores[best]`.",
                    nl="`{call}` geeft {got}: de publieke score van de beste inzending, het getal op het leaderboard. De vraag is wat diezelfde inzending op het private deel scoort: `private_scores[best]`."),
            Mistake("argmin", wrong=_b("np.argmin(public_scores)"),
                    en="`{call}` gives {got}: `np.argmin` gives the position of the lowest score, the worst submission. A share right is better when it is higher: `np.argmax`.",
                    nl="`{call}` geeft {got}: `np.argmin` geeft de plek van de laagste score, de slechtste inzending. Een aandeel goed is beter als het hoger is: `np.argmax`."),
            Mistake("position", wrong=_b(back="best"),
                    en="`{call}` gives {got}: the position of the leader in the list, not a score. Look up what is at that position in `private_scores`.",
                    nl="`{call}` geeft {got}: de plek van de leider in de lijst, geen score. Zoek op wat er op die plek in `private_scores` staat."),
            Mistake("max-value", error=lambda e: isinstance(e, IndexError) and "only integers" in str(e),
                    en="With `{call}`, your function stops with an error: `best` is a score, such as 0.57, and a score is no position in `private_scores`. `max` gives the best score itself; `np.argmax` gives its position.",
                    nl="Met `{call}` stopt je functie met een fout: `best` is een score, zoals 0,57, en een score is geen plek in `private_scores`. `max` geeft de beste score zelf; `np.argmax` geeft zijn plek."),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the two `...`: which submission is best on the public part, and what it scores on the private part.",
                      "nl": "Vervang de twee `...`: welke inzending het best is op het publieke deel, en wat die op het private deel scoort."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(public_scores, private_scores)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(public_scores, private_scores)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the private score of the submission with the highest public score.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: de private score van de inzending met de hoogste publieke score."},
            "almost": {"en": "Right for the competition. But `{call}` gives {got}, and it should give {want}. Does your function use its two values, and nothing of the competition?",
                       "nl": "Goed voor de wedstrijd. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie zijn twee waarden, en niets van de wedstrijd?"},
            "correct": {"en": "Your function is right in all {n} cases. The best of 200 coin flips scores 57.0% on the public part and 48.6% on the private part.",
                        "nl": "Je functie klopt in alle {n} gevallen. De beste van 200 keer munten gooien scoort 57,0% op het publieke deel en 48,6% op het private deel."},
        },
    ),
}


# --- exercise 2: the vote of the lucky ones --------------------------------------------------------------------------

_SUBS200 = _Val("submissions", competition()[3][:200])
_PUBS200 = _Val("public_scores", scores()[0][:200])


def _made_vote(seed: int, k: int, n: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    subs = rng.integers(0, 2, (k, n))
    pub = np.round(rng.uniform(0.35, 0.65, k), 3)
    pub[0] = 0.5                         # exactly 0.5 is not above it
    return _Val("submissions_" + str(seed), subs), _Val("public_" + str(seed), pub)


_V1 = _made_vote(1, 9, 12)
_V2 = _made_vote(2, 15, 8)
_V3 = _made_vote(3, 6, 20)

_V_HEAD = "import numpy as np\n\n\ndef vote(submissions, public_scores):\n"


def _v(lucky: str = "submissions[public_scores > 0.5]", votes: str = "lucky.mean(axis=0)", back: str = "votes > 0.5") -> str:
    return _V_HEAD + f"    lucky = {lucky}\n    votes = {votes}\n    return {back}\n"


_V_START = _v("submissions[...]", "...")
_V = _v()


def _as_votes(v):
    """A list of 0 and 1, or None when v is not a sequence of 0 and 1 or of True and False."""
    import numpy as np
    try:
        values = np.asarray(v)
    except Exception:
        return None
    if values.ndim != 1 or values.dtype.kind not in "biu":
        return None
    if not set(np.unique(values).tolist()) <= {0, 1}:
        return None
    return [int(x) for x in values]


def _same_votes(got, want) -> bool:
    mine = _as_votes(got)
    return mine is not None and mine == _as_votes(want)


def _shown_votes(v):
    votes = _as_votes(v)
    if votes is None:
        try:
            import numpy as np
            a = np.asarray(v)
            return {"en": f"an array of shape {a.shape}", "nl": f"een array met shape {a.shape}"}
        except Exception:
            return repr(v)[:60]
    if len(votes) > 12:
        return {"en": f"{number(len(votes), 'en')} votes, {number(sum(votes), 'en')} of them 1",
                "nl": f"{number(len(votes), 'nl')} stemmen, waarvan {number(sum(votes), 'nl')} op 1"}
    return "[" + ", ".join(str(x) for x in votes) + "]"


_VO_WORDS = {
    "title": {"en": "Your vote on the public and the private part", "nl": "Jouw stemming op het publieke en het private deel"},
    "x": {"en": "share right", "nl": "aandeel goed"},
    "public": {"en": "public part", "nl": "publieke deel"},
    "private": {"en": "private part", "nl": "private deel"},
    "best": {"en": "best single", "nl": "beste losse"},
    "yours": {"en": "your vote", "nl": "jouw stemming"},
    "right": {"en": "Your vote of the 100 submissions that scored above 50% on the public part gets 67.0% right there, far above the best single one, 57.0%. On the private part it gets 50.9%: a coin. In a competition it would top the public leaderboard and drop on the private one. With all 1,000 submissions the vote gets 84.0% on the public part, and 49.9% on the private part.",
              "nl": "Jouw stemming van de 100 inzendingen die boven 50% scoorden op het publieke deel haalt daar 67,0% goed, ver boven de beste losse inzending, 57,0%. Op het private deel haalt die 50,9%: een munt. In een wedstrijd zou hij bovenaan het publieke leaderboard staan en zakken op het private. Met alle 1.000 inzendingen haalt de stemming 84,0% op het publieke deel, en 49,9% op het private deel."},
    "said": {"en": "Your vote gets {pub} right on the public part and {priv} on the private part; the course's vote gets 67.0% and 50.9%.",
             "nl": "Jouw stemming haalt {pub} goed op het publieke deel en {priv} op het private deel; die van de course haalt 67,0% en 50,9%."},
}


def _vo_picture(seen) -> str:
    lang = seen.lang
    got = _call(seen, (_SUBS200, _PUBS200))
    votes = _as_votes(got)
    if votes is None or len(votes) != N:
        return ""
    import numpy as np
    words = {k: v[lang] for k, v in _VO_WORDS.items()}
    pub, priv = vote_scores(np.array(votes))
    if _same_votes(votes, course_vote(_SUBS200.value, _PUBS200.value)):
        text = words["right"]
    else:
        text = words["said"].format(pub=percent(pub, lang), priv=percent(priv, lang))
    p = chart.Panel((0.30, 0.90), (-0.6, 1.6), xlabel=words["x"], xticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    xtick_labels=["30%", "40%", "50%", "60%", "70%", "80%", "90%"], yticks=[])
    p.line([0.5, 0.5], [-0.6, 1.6], role="ink-3", width=0.8, dashed=True)
    _, best_pub, best_priv, _ = BEST[200]
    p.points([best_pub], [1], role="ink-3", r=5)
    p.points([best_priv], [0], role="ink-3", r=5)
    p.line([best_pub, best_priv], [1, 0], role="ink-3", width=1.2, dashed=True)
    if 0.30 <= pub <= 0.90 and 0.30 <= priv <= 0.90:
        p.line([pub, priv], [1, 0], role="ink", width=2.2)
        p.points([pub, priv], [1, 0], role="ink", r=5.5)
    p.text(0.305, 1.42, words["public"], role="ink-2", anchor="start")
    p.text(0.305, 0.42, words["private"], role="ink-2", anchor="start")
    key = f'<span style="color:var(--ea-ink)">●</span> {words["yours"]} &nbsp; <span style="color:var(--ea-ink-3)">●</span> {words["best"]}'
    return yours.figure(words["title"], chart.figure([p], alt=text, width=460, height=220) + _key(key) + _key(text), alt=text)


EXERCISES["vote"] = FunctionBox(
    name="vote",
    picture=_vo_picture,
    function="vote",
    start=_V_START,
    solution=_V,
    show=_shown_votes,
    compare=_same_votes,
    visible=[(_SUBS200, _PUBS200)],
    hidden=[_V1, _V2, _V3],
    first=[_DOTS],
    mistakes=[
        Mistake("all", wrong=_v("submissions"),
                en="`{call}` gives {got}: the vote of all submissions, lucky or not. None of them knows anything, so whatever that vote scores, on either part, is luck. Keep only the rows with a public score above 0.5: `submissions[public_scores > 0.5]`.",
                nl="`{call}` geeft {got}: de stemming van alle inzendingen, met of zonder geluk. Geen van hen weet iets, dus wat die stemming ook scoort, op welk deel ook, is geluk. Houd alleen de rijen met een publieke score boven 0,5: `submissions[public_scores > 0.5]`."),
        Mistake("unlucky", wrong=_v("submissions[public_scores < 0.5]"),
                en="`{call}` gives {got}: the vote of the unlucky ones, below 0.5, which is wrong on the public part more often than right. Keep the rows above 0.5: `public_scores > 0.5`.",
                nl="`{call}` geeft {got}: de stemming van de inzendingen zonder geluk, onder 0,5, en die is op het publieke deel vaker fout dan goed. Houd de rijen boven 0,5: `public_scores > 0.5`."),
        Mistake("at-half", wrong=_v("submissions[public_scores >= 0.5]"),
                en="`{call}` gives {got}: with `>=`, the submissions with a public score of exactly 0.5 vote too, and they did no better than a coin. Two of the 200 score exactly 0.5. Keep those above 0.5: `>`.",
                nl="`{call}` geeft {got}: met `>=` stemmen ook de inzendingen met een publieke score van precies 0,5 mee, en die deden het niet beter dan een munt. Twee van de 200 scoren precies 0,5. Houd die boven 0,5: `>`."),
        Mistake("sum", wrong=_v(votes="lucky.sum(axis=0)"),
                en="`{call}` gives {got}: `sum` counts the 1s, so `votes > 0.5` is True as soon as one of them says 1. The share of 1s is the mean: `lucky.mean(axis=0)`.",
                nl="`{call}` geeft {got}: `sum` telt de enen, dus `votes > 0.5` is True zodra één van hen 1 zegt. Het aandeel enen is het gemiddelde: `lucky.mean(axis=0)`."),
        Mistake("percent", wrong=_v("submissions[public_scores > 50]"),
                en="`{call}` gives {got}: the public scores are shares between 0 and 1, such as 0.57, so no score is above 50 and nobody votes. Compare with 0.5.",
                nl="`{call}` geeft {got}: de publieke scores zijn aandelen tussen 0 en 1, zoals 0,57, dus geen score ligt boven 50 en niemand stemt. Vergelijk met 0,5."),
        Mistake("rows", wrong=_v(votes="lucky.mean(axis=1)"),
                en="`{call}` gives {got}: with `axis=1` the mean runs along a row, one number per submission. The vote is per example, down the columns: `axis=0`.",
                nl="`{call}` geeft {got}: met `axis=1` loopt het gemiddelde langs een rij, één getal per inzending. De stemming is per voorbeeld, langs de kolommen: `axis=0`."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the two `...`: which rows are lucky, and the share of 1 among them for every example.",
                  "nl": "Vervang de twee `...`: welke rijen geluk hadden, en het aandeel 1 onder hen voor elk voorbeeld."},
        "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(submissions, public_scores)`.",
                      "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(submissions, public_scores)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: for every example, whether more than half of the submissions with a public score above 0.5 say 1.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: voor elk voorbeeld of meer dan de helft van de inzendingen met een publieke score boven 0,5 er 1 zegt."},
        "almost": {"en": "Right for the competition. But `{call}` gives {got}, and it should give {want}. Does your function use its two values, and nothing of the competition?",
                   "nl": "Goed voor de wedstrijd. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie zijn twee waarden, en niets van de wedstrijd?"},
        "correct": {"en": "Your function is right in all {n} cases. The vote of the 100 lucky ones gets 67.0% on the public part and 50.9% on the private part.",
                    "nl": "Je functie klopt in alle {n} gevallen. De stemming van de 100 met geluk haalt 67,0% op het publieke deel en 50,9% op het private deel."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("best-of", _B_START, "empty", None),
    ("best-of", _b(back="..."), "wrong", "dots-left"),
    ("best-of", _B, "correct", None),
    ("best-of", _b("int(np.argmax(public_scores))", "float(private_scores[best])"), "correct", None),
    ("best-of", _b("list(public_scores).index(max(public_scores))"), "correct", None),
    ("best-of", _b("np.argmax(private_scores)"), "wrong", "best-private"),
    ("best-of", _b(back="public_scores[best]"), "wrong", "public-back"),
    ("best-of", _b("np.argmin(public_scores)"), "wrong", "argmin"),
    ("best-of", _b("max(public_scores)"), "error", "max-value"),
    ("best-of", _b(back="best"), "wrong", "position"),
    ("vote", _V_START, "empty", None),
    ("vote", _v(votes="..."), "wrong", "dots-left"),
    ("vote", _V, "correct", None),
    ("vote", _v(back="(votes > 0.5).astype(int)"), "correct", None),
    ("vote", _v("submissions"), "wrong", "all"),
    ("vote", _v("submissions[public_scores < 0.5]"), "wrong", "unlucky"),
    ("vote", _v(votes="lucky.mean(axis=1)"), "wrong", "rows"),
    ("vote", _v("submissions[public_scores >= 0.5]"), "wrong", "at-half"),
    ("vote", _v(votes="lucky.sum(axis=0)"), "wrong", "sum"),
    ("vote", _v("submissions[public_scores > 50]"), "wrong", "percent"),
    ("vote", _v(votes="np.mean(lucky, axis=0)"), "correct", None),
]
