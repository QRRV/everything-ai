# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-randomized-response: a made-up survey of 10,000 people on a question they would rather
not answer (TRUTH, with 2,000 true yes answers), the two coins every person flips (FIRST and SECOND, 1 for heads),
the answers they give (ANSWERS), and eight people with every combination of truth and coins (FEW_TRUTH, FEW_FIRST,
FEW_SECOND). The three functions the learner writes: answer (what each person says), estimate (the true share of yes,
from the answers) and epsilon (how much privacy a first coin that says "tell the truth" with chance t gives); with the
pictures of what the learner's functions give."""
import html
import math
import re

import numpy as np

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, _show_rounded

N = 10000       # people in the made-up survey
YES = 2000      # of them, the people whose true answer is yes
SEED = 2026


class Named:
    """An array the check calls by the name the lesson gives it, such as TRUTH, so that a message stays short. It is
    made the first time a check needs it, and the learner's code gets its own copy."""

    def __init__(self, make, name: str):
        self.make, self.name, self._value = make, name, None

    @property
    def value(self) -> np.ndarray:
        if self._value is None:
            self._value = self.make()
        return self._value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.value.copy()


def answer(truth, first, second):
    return (1 - first) * truth + first * second


def estimate(answers):
    return 2 * answers.mean() - 0.5


def epsilon(t):
    yes_if_yes = t + (1 - t) / 2
    yes_if_no = (1 - t) / 2
    return np.log(yes_if_yes / yes_if_no)


def _survey():
    """The made-up survey: exactly 2,000 true yes answers among 10,000 people, in a random order, and two fair coins
    for every person."""
    rng = np.random.default_rng(SEED)
    truth = rng.permutation(np.array([1] * YES + [0] * (N - YES)))
    return truth, rng.integers(0, 2, size=N), rng.integers(0, 2, size=N)


_SURVEY: list = []


def survey() -> tuple:
    if not _SURVEY:
        _SURVEY.extend(_survey())
    return tuple(_SURVEY)


TRUTH = Named(lambda: survey()[0], "TRUTH")
FIRST = Named(lambda: survey()[1], "FIRST")
SECOND = Named(lambda: survey()[2], "SECOND")
ANSWERS = Named(lambda: answer(*survey()), "ANSWERS")

# eight people, one for each combination of truth, first coin and second coin
FEW_TRUTH = Named(lambda: np.array([1, 1, 1, 1, 0, 0, 0, 0]), "FEW_TRUTH")
FEW_FIRST = Named(lambda: np.array([0, 0, 1, 1, 0, 0, 1, 1]), "FEW_FIRST")
FEW_SECOND = Named(lambda: np.array([0, 1, 0, 1, 0, 1, 0, 1]), "FEW_SECOND")

# the surveys of the picture under estimate: 200 surveys of each size, with a true share of 0.2 drawn person by person
SIZES = [100, 1000, 10000]
SURVEYS = 200
SHARE = 0.2
_MANY: dict = {}


def many(n: int) -> list:
    """200 made-up surveys of n people: for each, the true answers and the answers with two coins."""
    if n not in _MANY:
        out = []
        for k in range(SURVEYS):
            rng = np.random.default_rng(1000 * n + k)
            truth = (rng.random(n) < SHARE) * 1
            first, second = rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)
            out.append((truth, answer(truth, first, second)))
        _MANY[n] = out
    return _MANY[n]


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{html.escape(c)}</div>'
                    for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list) -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:34rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _grouped(n: int, lang: str) -> str:
    """A whole number as the lesson's language writes it: 10,000 in English, 10.000 in Dutch (a dot from five digits)."""
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


def _decimal(v: float, lang: str, digits: int) -> str:
    text = f"{v:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _percent(v: float, lang: str) -> str:
    """A share from 0 to 1 as a percentage: 75.0% or 75,0%. A number outside 0 to 1 is no share, so it is shown as it
    is."""
    text = f"{100 * v:.1f}%" if 0 <= v <= 1 else f"{v:g}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _limited(work):
    from everything_ai import check as checking
    return checking.with_time_limit(work)


def _array(v):
    """What a function gave back as a NumPy array of numbers, or None."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    return a if a.ndim == 1 else None


def _number(v):
    try:
        if isinstance(v, bool):
            return None
        out = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


_ANSWER_WORDS = {
    "en": {"title": "Your answer(truth, first, second) on eight people and the whole survey",
           "head": ["person", "truth", "first coin", "second coin", "your answer"],
           "yes": "yes", "no": "no", "heads": "heads", "tails": "tails",
           "split": "Of the {y} people whose truth is yes, your answer says yes for {a}; of the {n} whose truth is "
                    "no, for {b}.",
           "key": "FEW_TRUTH, FEW_FIRST and FEW_SECOND: eight made-up people, one for each combination of truth and "
                  "coins.",
           "below": "Below them, your answer on the 10,000 people of the survey.",
           "wrong": "A dashed frame marks where your answer differs from the course's."},
    "nl": {"title": "Jouw answer(truth, first, second) op acht mensen en de hele enquête",
           "head": ["persoon", "waarheid", "eerste munt", "tweede munt", "jouw antwoord"],
           "yes": "ja", "no": "nee", "heads": "kop", "tails": "munt",
           "split": "Van de {y} mensen bij wie de waarheid ja is, geeft jouw functie ja voor {a}; van de {n} bij wie "
                    "de waarheid nee is, voor {b}.",
           "key": "FEW_TRUTH, FEW_FIRST en FEW_SECOND: acht verzonnen mensen, één voor elke combinatie van waarheid "
                  "en munten.",
           "below": "Daaronder jouw functie answer op de 10.000 mensen van de enquête.",
           "wrong": "Een gestreept kader wijst aan waar jouw antwoord verschilt van dat van de course."},
}


def _bar(label: str, v: float, lang: str) -> str:
    width = max(0.0, min(1.0, v))
    return (f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.2rem 0;font-family:var(--ea-font-ui);'
            f'font-size:0.85rem;color:var(--ea-ink)"><span style="flex:0 0 11rem">{html.escape(label)}</span>'
            f'<span style="flex:1;height:0.7rem;background:var(--ea-line);border-radius:3px;max-width:14rem">'
            f'<span style="display:block;height:100%;width:{100 * width:.1f}%;background:var(--ea-ink-2);border-radius:3px">'
            f'</span></span><span style="min-width:3.4rem;text-align:right">{_percent(v, lang)}</span></div>')


def answer_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ANSWER_WORDS[seen.lang]
    few = [FEW_TRUTH.value, FEW_FIRST.value, FEW_SECOND.value]
    try:
        got = _array(_limited(lambda: fn(*(a.copy() for a in few))))
    except Exception:
        got = None
    want = answer(*few)
    rows, wrong_any = [], False
    widths = [2, 2, 3, 3, 3]
    for i in range(len(want)):
        mine = None if got is None or len(got) != len(want) else got[i]
        wrong = mine is None or mine != want[i]
        wrong_any |= wrong
        shown = ("–" if mine is None else w["yes"] if mine == 1 else w["no"] if mine == 0 else
                 html.escape(_decimal(mine, seen.lang, 2)))
        rows.append(_row([str(i + 1), w["yes"] if few[0][i] else w["no"], w["heads"] if few[1][i] else w["tails"],
                          w["heads"] if few[2][i] else w["tails"], shown], _WRONG if wrong else _PLAIN, widths))
    body = _table(w["head"], rows, widths)
    truth, first, second = survey()
    try:
        whole = _array(_limited(lambda: fn(truth.copy(), first.copy(), second.copy())))
    except Exception:
        whole = None
    note = w["key"]
    if whole is not None and len(whole) == N:
        a, b = float(whole[truth == 1].mean()), float(whole[truth == 0].mean())
        labels = {"en": (f"truth yes ({_grouped(YES, 'en')})", f"truth no ({_grouped(N - YES, 'en')})"),
                  "nl": (f"waarheid ja ({_grouped(YES, 'nl')})", f"waarheid nee ({_grouped(N - YES, 'nl')})")}
        body += _bar(labels[seen.lang][0], a, seen.lang) + _bar(labels[seen.lang][1], b, seen.lang)
        note += " " + w["below"] + " " + w["split"].format(y=_grouped(YES, seen.lang), n=_grouped(N - YES, seen.lang),
                                        a=_percent(a, seen.lang), b=_percent(b, seen.lang))
    if wrong_any:
        note += " " + w["wrong"]
    return yours.figure(w["title"], body + f'<div style="{_UI}">{html.escape(note)}</div>')


_ESTIMATE_WORDS = {
    "en": {"title": "Your estimate(answers) on 200 made-up surveys of each size",
           "rows": {100: "coins, 100 answers", 1000: "coins, 1,000 answers", 10000: "coins, 10,000 answers",
                    "straight": "asked straight, 1,000 answers"},
           "x": "estimated share of yes",
           "key": "Each dot is one survey in which each person's truth is yes with chance 0.2, the dashed line. In the three rows with coins, a "
                  "dot is what your estimate gives on its answers; in the bottom row, the plain share of yes of "
                  "1,000 people who all tell the truth.",
           "said": "The standard deviation of your estimates: {a} with 100 answers, {b} with 1,000 and {c} with "
                   "10,000. Asked straight, 1,000 answers give {d}.",
           "wrong": "Your estimate differs from the course's on some of these surveys."},
    "nl": {"title": "Jouw estimate(answers) op 200 verzonnen enquêtes van elke grootte",
           "rows": {100: "munten, 100 antwoorden", 1000: "munten, 1000 antwoorden",
                    10000: "munten, 10.000 antwoorden", "straight": "rechtstreeks gevraagd, 1000 antwoorden"},
           "x": "geschat aandeel ja",
           "key": "Elke stip is één enquête waarin de waarheid van elke persoon met kans 0,2 ja is, de gestreepte lijn. In de drie rijen met "
                  "munten is een stip wat jouw functie estimate geeft op de antwoorden; in de onderste rij het gewone aandeel "
                  "ja van 1000 mensen die allemaal de waarheid zeggen.",
           "said": "De standaardafwijking van jouw schattingen: {a} met 100 antwoorden, {b} met 1000 en {c} met "
                   "10.000. Rechtstreeks gevraagd geven 1000 antwoorden {d}.",
           "wrong": "Jouw functie estimate verschilt bij sommige van deze enquêtes van die van de course."},
}

LO, HI = -0.1, 0.6


def estimate_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ESTIMATE_WORDS[seen.lang]

    def run():
        out = {}
        for n in SIZES:
            values = []
            for _, answers in many(n):
                try:
                    values.append(_number(fn(answers.copy())))
                except Exception:
                    values.append(None)
            out[n] = values
        return out
    try:
        got = _limited(run)
    except Exception:
        return ""
    straight = [float(truth.mean()) for truth, _ in many(1000)]
    panel = chart.Panel(x=(LO, HI), y=(0, 4.3), xlabel=w["x"], yticks=[])
    jitter = np.random.default_rng(7).uniform(-0.22, 0.22, size=SURVEYS)
    places = [("straight", 0.45, straight)] + [(n, 1.45 + i, got[n]) for i, n in enumerate(reversed(SIZES))]
    wrong_any = False
    for key, y, values in places:
        xs, ys = [], []
        for v, j in zip(values, jitter):
            if v is not None and LO <= v <= HI:
                xs.append(v)
                ys.append(y + j)
        panel.points(xs, ys, role="ink-2", r=2.4, hollow=key == "straight")
        panel.text(LO, y + 0.42, w["rows"][key], role="ink-2", anchor="start", halo=True)
    for n in SIZES:
        want = [estimate(answers) for _, answers in many(n)]
        wrong_any |= any(v is None or abs(v - x) > 1e-9 for v, x in zip(got[n], want))
    panel.line([SHARE, SHARE], [0, 4.3], role="ink", width=1.4, dashed=True)
    figure = chart.figure([panel], alt=w["title"], width=480, height=300)
    if seen.lang == "nl":
        figure = chart.with_decimal_comma(figure)
    note = w["key"]
    if wrong_any:
        note += " " + w["wrong"]
    else:
        sd = {n: float(np.std([v for v in got[n]])) for n in SIZES}
        note += " " + w["said"].format(a=_decimal(sd[100], seen.lang, 3), b=_decimal(sd[1000], seen.lang, 3),
                                       c=_decimal(sd[10000], seen.lang, 4), d=_decimal(float(np.std(straight)),
                                                                                        seen.lang, 3))
    return yours.figure(w["title"], figure + f'<div style="{_UI}">{html.escape(note)}</div>')


_EPSILON_WORDS = {
    "en": {"title": "Your epsilon(t): how much one answer may say",
           "x": "t: chance that the first coin says: tell the truth", "y": "ε",
           "said": "Your epsilon(0.5) gives {a}; epsilon(0.9) gives {b}; epsilon(0) gives {c}."},
    "nl": {"title": "Jouw epsilon(t): hoeveel één antwoord mag zeggen",
           "x": "t: kans dat de eerste munt zegt: zeg de waarheid", "y": "ε",
           "said": "Jouw epsilon(0.5) geeft {a}; epsilon(0.9) geeft {b}; epsilon(0) geeft {c}."},
}


def epsilon_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _EPSILON_WORDS[seen.lang]
    curve = yours.curve("epsilon", [i / 100 for i in range(0, 96)], role="ink",
                        title=yours.Text(en=_EPSILON_WORDS["en"]["title"], nl=_EPSILON_WORDS["nl"]["title"]),
                        xlabel=yours.Text(en=_EPSILON_WORDS["en"]["x"], nl=_EPSILON_WORDS["nl"]["x"]),
                        ylabel="ε", y=(0, 4))(seen)
    values = []
    for t in (0.5, 0.9, 0.0):
        try:
            values.append(_number(_limited(lambda t=t: fn(t))))
        except Exception:
            values.append(None)
    shown = ["–" if v is None else _decimal(v, seen.lang, 2) for v in values]
    note = w["said"].format(a=shown[0], b=shown[1], c=shown[2])
    return curve + f'<div style="{_UI}">{html.escape(note)}</div>'


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["answer", "estimate", "epsilon", "a-yes", "every-day"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: answer
_R_HEAD = ("def answer(truth, first, second):\n"
           "    # tails (0) on the first coin: the truth; heads (1): what the second coin says\n")
_R_START = _R_HEAD + "    return ...\n"
_R = _R_HEAD + "    return (1 - first) * truth + first * second\n"
_R_SWAP = _R_HEAD + "    return first * truth + (1 - first) * second\n"
_R_TRUTH = _R_HEAD + "    return truth\n"
_R_FORCED = _R_HEAD + "    return (1 - first) * truth\n"
_R_PLUS = _R_HEAD + "    return truth + first * second\n"
_R_COINS = _R_HEAD + "    return first * second\n"

# exercise 2: estimate
_E_HEAD = ("def estimate(answers):\n"
           "    # the true share of yes, from the answers with two coins\n")
_E_START = _E_HEAD + "    return ...\n"
_E = _E_HEAD + "    return 2 * answers.mean() - 0.5\n"
_E_SHARE = _E_HEAD + "    return answers.mean()\n"
_E_DOUBLE = _E_HEAD + "    return 2 * answers.mean()\n"
_E_ONE = _E_HEAD + "    return 2 * (answers.mean() - 0.5)\n"
_E_QUARTER = _E_HEAD + "    return answers.mean() - 0.25\n"
_E_SUM = _E_HEAD + "    return 2 * answers.sum() - 0.5\n"

# exercise 3: epsilon
_P_HEAD = ("import numpy as np\n\n\n"
           "def epsilon(t):\n"
           "    # the log of how many times as likely a yes is for someone whose truth is yes, as for a no\n")
_P_START = _P_HEAD + "    yes_if_yes = ...\n    yes_if_no = ...\n    return ...\n"
_P_YES = "    yes_if_yes = t + (1 - t) / 2\n"
_P_NO = "    yes_if_no = (1 - t) / 2\n"
_P = _P_HEAD + _P_YES + _P_NO + "    return np.log(yes_if_yes / yes_if_no)\n"
_P_RATIO = _P_HEAD + _P_YES + _P_NO + "    return yes_if_yes / yes_if_no\n"
_P_LOG10 = _P_HEAD + _P_YES + _P_NO + "    return np.log10(yes_if_yes / yes_if_no)\n"
_P_FLIP = _P_HEAD + _P_YES + _P_NO + "    return np.log(yes_if_no / yes_if_yes)\n"
_P_TRUTH = _P_HEAD + "    yes_if_yes = t\n" + _P_NO + "    return np.log(yes_if_yes / yes_if_no)\n"
_P_HALF = _P_HEAD + _P_YES + "    yes_if_no = 1 - t\n" + "    return np.log(yes_if_yes / yes_if_no)\n"

EXERCISES = {
    "answer": FunctionBox(
        name="answer",
        picture=answer_picture,
        function="answer",
        start=_R_START,
        solution=_R,
        visible=[(FEW_TRUTH, FEW_FIRST, FEW_SECOND)],
        hidden=[(TRUTH, FIRST, SECOND), (np.array([1, 0, 1, 0]), np.array([1, 1, 1, 1]), np.array([1, 1, 0, 0])),
                (np.array([1, 0, 1, 0]), np.array([0, 0, 0, 0]), np.array([1, 1, 0, 0]))],
        first=[_DOTS],
        mistakes=[
            Mistake("if", error=lambda e: isinstance(e, ValueError) and "truth value" in str(e),
                    en="With `{call}`, your function stops: something such as `if` or `and` needs one True or False, and a comparison with an array, such as `first == 0`, gives one for every person. Work with whole arrays instead: `1 - first` is 1 where the first coin shows tails (0), and 0 where it shows heads (1).",
                    nl="Met `{call}` stopt je functie: zoiets als `if` of `and` heeft één True of False nodig, en een vergelijking met een array, zoals `first == 0`, geeft er één per persoon. Reken liever met hele arrays: `1 - first` is 1 waar de eerste munt op munt (0) valt, en 0 waar die op kop (1) valt."),
            Mistake("swap", wrong=_R_SWAP,
                    en="`{call}` gives {got}, and it should give {want}. Your code tells the truth on heads. In the recipe it is the other way round: tails (0) on the first coin gives the truth, heads (1) hands the answer to the second coin.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code zegt de waarheid bij kop. In het recept is het andersom: valt de eerste munt op munt (0), dan geeft dat de waarheid; valt die op kop (1), dan beslist de tweede munt."),
            Mistake("truth", wrong=_R_TRUTH,
                    en="`{call}` gives {got}: the truth itself, as if nobody flipped a coin. On heads (1) on the first coin, the answer is what the second coin says.",
                    nl="`{call}` geeft {got}: de waarheid zelf, alsof niemand een munt opgooide. Valt de eerste munt op kop (1), dan is het antwoord wat de tweede munt zegt."),
            Mistake("forced", wrong=_R_FORCED,
                    en="`{call}` gives {got}, and it should give {want}. On heads (1) on the first coin your answer is always 0, but then the second coin decides: add `first * second`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Valt de eerste munt op kop (1), dan is je antwoord altijd 0, maar dan beslist de tweede munt: tel er `first * second` bij op."),
            Mistake("plus", wrong=_R_PLUS,
                    en="`{call}` gives {got}, and it should give {want}. Your answer can be 2: on heads (1) on the first coin, the truth still counts. Multiply the truth by `(1 - first)`, which is 0 on heads.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je antwoord kan 2 zijn: valt de eerste munt op kop (1), dan telt de waarheid nog mee. Vermenigvuldig de waarheid met `(1 - first)`, dat bij kop 0 is."),
            Mistake("coins", wrong=_R_COINS,
                    en="`{call}` gives {got}, and it should give {want}. On tails (0) on the first coin your answer is always 0, but then the person tells the truth: add `(1 - first) * truth`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Valt de eerste munt op munt (0), dan is je antwoord altijd 0, maar dan zegt de persoon de waarheid: tel er `(1 - first) * truth` bij op."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the answers: the truth where `first` is 0, and `second` where `first` is 1.",
                      "nl": "Vervang de `...` door de antwoorden: de waarheid waar `first` 0 is, en `second` waar `first` 1 is."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where `first` is 0, the answer is `truth`; where it is 1, the answer is `second`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Waar `first` 0 is, is het antwoord `truth`; waar het 1 is, is het antwoord `second`."},
            "almost": {"en": "Right for `answer(FEW_TRUTH, FEW_FIRST, FEW_SECOND)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `truth`, the `first` and the `second` it gets?",
                       "nl": "Goed voor `answer(FEW_TRUTH, FEW_FIRST, FEW_SECOND)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `truth`, de `first` en de `second` die hij krijgt?"},
            "correct": {"en": "Your `answer` is right in all {n} cases. The picture runs it on the eight people and on the whole survey.",
                        "nl": "Je `answer` klopt in alle {n} gevallen. Het plaatje laat hem lopen op de acht mensen en op de hele enquête."},
        },
    ),
    "estimate": FunctionBox(
        name="estimate",
        picture=estimate_picture,
        function="estimate",
        show=lambda v: _show_rounded(v, 4),
        start=_E_START,
        solution=_E,
        visible=[(ANSWERS,)],
        hidden=[(np.array([1, 1, 0, 1, 0, 0, 0, 1]),), (np.array([0, 0, 0, 0]),), (np.array([1, 1, 1, 1]),),
                (np.array([1, 0, 0, 0]),)],
        first=[_DOTS],
        mistakes=[
            Mistake("share", wrong=_E_SHARE,
                    en="`{call}` gives {got}: the share of yes among the answers. The coins push that share toward one half, so undo them: twice the share, minus 0.5.",
                    nl="`{call}` geeft {got}: het aandeel ja in de antwoorden. De munten duwen dat aandeel naar de helft, dus maak ze ongedaan: twee keer het aandeel, min 0,5."),
            Mistake("double", wrong=_E_DOUBLE,
                    en="`{call}` gives {got}, and it should give {want}. Doubling is half of it: the forced yes answers add a quarter, which doubling makes one half. Take that 0.5 off.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Verdubbelen is de helft van het werk: de gedwongen ja-antwoorden voegen een kwart toe, en dat wordt na verdubbelen een half. Haal die 0,5 eraf."),
            Mistake("one", wrong=_E_ONE,
                    en="`{call}` gives {got}, and it should give {want}. In the end, your code takes 1 off twice the share; the estimate takes off only 0.5. Double the share first, then take 0.5 off.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je code haalt uiteindelijk 1 af van twee keer het aandeel; de schatting haalt er maar 0,5 af. Verdubbel eerst het aandeel, en haal er dan 0,5 af."),
            Mistake("quarter", wrong=_E_QUARTER,
                    en="`{call}` gives {got}, and it should give {want}. You take off the quarter of forced yes answers, but a true yes is said only by the half who tell the truth: double what is left.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je haalt het kwart gedwongen ja-antwoorden eraf, maar een echte ja zegt alleen de helft die de waarheid vertelt: verdubbel wat er overblijft."),
            Mistake("sum", wrong=_E_SUM,
                    en="`{call}` gives {got}: `.sum()` counts the yes answers. The estimate works on their share: `.mean()`.",
                    nl="`{call}` geeft {got}: `.sum()` telt de ja-antwoorden. De schatting werkt met hun aandeel: `.mean()`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with twice the share of yes among `answers`, minus 0.5.",
                      "nl": "Vervang de `...` door twee keer het aandeel ja in `answers`, min 0,5."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Take the share of yes among `answers`, double it, and take 0.5 off.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Neem het aandeel ja in `answers`, verdubbel het, en haal er 0,5 af."},
            "almost": {"en": "Right for `estimate(ANSWERS)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `answers` it gets?",
                       "nl": "Goed voor `estimate(ANSWERS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `answers` die hij krijgt?"},
            "correct": {"en": "Your `estimate` is right in all {n} cases. The picture runs it on 600 made-up surveys.",
                        "nl": "Je `estimate` klopt in alle {n} gevallen. Het plaatje laat hem lopen op 600 verzonnen enquêtes."},
        },
    ),
    "epsilon": FunctionBox(
        name="epsilon",
        picture=epsilon_picture,
        function="epsilon",
        show=lambda v: _show_rounded(v, 4),
        start=_P_START,
        solution=_P,
        visible=[(0.5,)],
        hidden=[(0.0,), (0.25,), (0.75,), (0.9,)],
        first=[_DOTS],
        mistakes=[
            Mistake("turned", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="With `{call}`, your function divides by zero. At t = 0 the first coin never says \"tell the truth\", so the second coin decides every answer and both chances of a yes are 1/2. Is your `t` the chance that the second coin decides? Here it is the chance that the first coin says \"tell the truth\"; the second coin decides with chance `1 - t`.",
                    nl="Met `{call}` deelt je functie door nul. Bij t = 0 zegt de eerste munt nooit \"zeg de waarheid\", dus beslist de tweede munt elk antwoord en zijn beide kansen op ja 1/2. Is jouw `t` de kans dat de tweede munt beslist? Hier is het de kans dat de eerste munt \"zeg de waarheid\" zegt; de tweede munt beslist met kans `1 - t`."),
            Mistake("ratio", wrong=_P_RATIO,
                    en="`{call}` gives {got}: how many times as likely a yes is. ε is the log of that number: `np.log(...)`.",
                    nl="`{call}` geeft {got}: hoeveel keer zo waarschijnlijk een ja is. ε is de log van dat getal: `np.log(...)`."),
            Mistake("log10", wrong=_P_LOG10,
                    en="`{call}` gives {got}, and it should give {want}. That is the log with base 10. ε uses the log with base e: `np.log`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is de log met grondtal 10. ε gebruikt de log met grondtal e: `np.log`."),
            Mistake("flip", wrong=_P_FLIP,
                    en="`{call}` gives {got}, and it should give {want}. Your fraction is upside down: the chance of a yes for someone whose truth is yes goes on top.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je breuk staat op zijn kop: de kans op ja voor iemand bij wie de waarheid ja is, staat boven."),
            Mistake("truth", wrong=_P_TRUTH,
                    en="`{call}` gives {got}, and it should give {want}. Someone whose truth is yes also says yes when the coins force a yes. Add that chance: `(1 - t) / 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Iemand bij wie de waarheid ja is, zegt ook ja als de munten een ja afdwingen. Tel die kans erbij op: `(1 - t) / 2`."),
            Mistake("half", wrong=_P_HALF,
                    en="`{call}` gives {got}, and it should give {want}. When the first coin does not say \"tell the truth\", the second coin says yes only half of the time: `(1 - t) / 2`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Als de eerste munt niet \"zeg de waarheid\" zegt, zegt de tweede munt maar de helft van de keren ja: `(1 - t) / 2`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the three `...`: the chance of a yes for someone whose truth is yes, the same for a no, and the log of their ratio.",
                      "nl": "Vervang de drie `...`: de kans op ja voor iemand bij wie de waarheid ja is, hetzelfde voor nee, en de log van hun verhouding."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A yes comes from the truth with chance `t`, or from the coins with chance `(1 - t) / 2`; for a no, only from the coins. ε is the log of the ratio.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een ja komt van de waarheid met kans `t`, of van de munten met kans `(1 - t) / 2`; bij nee alleen van de munten. ε is de log van de verhouding."},
            "almost": {"en": "Right for `epsilon(0.5)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `t` it gets?",
                       "nl": "Goed voor `epsilon(0.5)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `t` die hij krijgt?"},
            "correct": {"en": "Your `epsilon` is right in all {n} cases. The picture draws it for t from 0 to 0.95.",
                        "nl": "Je `epsilon` klopt in alle {n} gevallen. Het plaatje tekent hem voor t van 0 tot 0,95."},
        },
    ),
}

CASES = [
    ("answer", _R_START, "empty", None),
    ("answer", _R, "correct", None),
    ("answer", _R_HEAD + "    return truth * (1 - first) + second * first\n", "correct", None),
    ("answer", "import numpy as np\n\n\n" + _R_HEAD + "    return np.where(first == 1, second, truth)\n", "correct", None),
    ("answer", _R_HEAD + "    if first == 0:\n        return truth\n    return second\n", "error", "if"),
    ("answer", _R_SWAP, "wrong", "swap"),
    ("answer", _R_TRUTH, "wrong", "truth"),
    ("answer", _R_FORCED, "wrong", "forced"),
    ("answer", _R_PLUS, "wrong", "plus"),
    ("answer", _R_COINS, "wrong", "coins"),
    ("answer", _R_HEAD + "    return (1 - first) * truth + first * (1 - second)\n", "wrong", None),
    ("estimate", _E_START, "empty", None),
    ("estimate", _E, "correct", None),
    ("estimate", _E_HEAD + "    return (answers.mean() - 0.25) * 2\n", "correct", None),
    ("estimate", _E_HEAD + "    return 2 * answers.sum() / len(answers) - 0.5\n", "correct", None),
    ("estimate", _E_SHARE, "wrong", "share"),
    ("estimate", _E_HEAD + "    return answers.sum() / len(answers)\n", "wrong", "share"),
    ("estimate", _E_DOUBLE, "wrong", "double"),
    ("estimate", _E_ONE, "wrong", "one"),
    ("estimate", _E_QUARTER, "wrong", "quarter"),
    ("estimate", _E_SUM, "wrong", "sum"),
    ("estimate", _E_HEAD + "    return 2 * answers.mean() + 0.5\n", "wrong", None),
    ("epsilon", _P_START, "empty", None),
    ("epsilon", _P, "correct", None),
    ("epsilon", _P_HEAD + "    return np.log((1 + t) / (1 - t))\n", "correct", None),
    ("epsilon", _P_HEAD + "    yes_if_yes = (1 - t) + t / 2\n    yes_if_no = t / 2\n    return np.log(yes_if_yes / yes_if_no)\n", "error", "turned"),
    ("epsilon", _P_RATIO, "wrong", "ratio"),
    ("epsilon", _P_LOG10, "wrong", "log10"),
    ("epsilon", _P_FLIP, "wrong", "flip"),
    ("epsilon", _P_TRUTH, "wrong", "truth"),
    ("epsilon", _P_HALF, "wrong", "half"),
    ("epsilon", _P_HEAD + _P_YES + _P_NO + "    return np.log(yes_if_yes)\n", "wrong", None),
]
