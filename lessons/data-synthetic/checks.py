# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-synthetic: the 10,000 runs of AI4I 2020, a dataset its maker simulated to look like
real maintenance data. The first 3000 runs are the log: not one heat failure, because the difference between process
and air temperature stays above 8.6 K there. The next 2000 runs, a warmer stretch, hold all 115 heat failures:
the test. The lesson simulates 1000 runs, a temperature difference between 7.5 and 9.5 K and a speed taken from the
log, labels them with a rule, and trains k-NN (k = 5) on the log and the simulated runs together. The slider moves the
speed limit of the rule; the check of exercise 1 compares a rule for power failures with AI4I's own labels. The
functions power_failure and between. tests/test_data_synthetic.py computes the numbers in the lesson again."""
import contextlib
import functools
import io
import math
import re
import warnings

from everything_ai import chart, check as checking, yours
from everything_ai.check import FunctionBox, Mistake

DATA = "data/ai4i-2020.csv.gz"   # AI4I 2020 Predictive Maintenance (CC BY 4.0): see data/README.md
N_LOG = 3000                     # the first 3000 runs: the log, without a heat failure
TEST = (3000, 5000)              # the next 2000 runs: the test, with all 115 heat failures
K = 5                            # an odd k: two classes cannot tie
LOW, HIGH = 7.5, 9.5             # the simulated temperature differences, in K
N_SIM = 1000
SIM_SEED = 4                     # seed 4 gives a typical result (see the tests)
OTHER_SEED = 104                 # the 1000 simulated test runs
DIFF_LIMIT, SPEED_LIMIT = 8.6, 1380   # AI4I's rule for a heat failure (Matzka 2020)
LIMITS = [1260, 1300, 1340, 1380, 1420, 1460, 1500]   # the stops of the slider
POWER_LOW, POWER_HIGH = 3500, 9000    # AI4I's rule for a power failure, in watts
WIDE = 1460                      # the colleague's speed limit of exercise 3


@functools.lru_cache(maxsize=1)
def _data():
    """The columns the lesson uses, in the order of the runs: the temperature difference, the speed, the torque, and
    the labels for a heat failure and a power failure. The difference is not rounded: AI4I made its labels from the
    difference as Python computes it, so 309.1 - 300.5 = 8.599999999999966 counts as below 8.6 K."""
    import numpy as np
    import pandas as pd
    from everything_ai import paths
    df = pd.read_csv(paths.ROOT / DATA)
    temps = np.array(df[["Process temperature [K]", "Air temperature [K]"]])
    return {"diff": temps[:, 0] - temps[:, 1], "speed": np.array(df["Rotational speed [rpm]"]),
            "torque": np.array(df["Torque [Nm]"]), "heat": np.array(df["HDF"]), "power": np.array(df["PWF"])}


def runs(diff, speed):
    """One row for every run: its temperature difference and its speed."""
    import numpy as np
    out = np.zeros((len(diff), 2))
    out[:, 0] = diff
    out[:, 1] = speed
    return out


def log():
    d = _data()
    return runs(d["diff"][:N_LOG], d["speed"][:N_LOG]), d["heat"][:N_LOG]


def test():
    d = _data()
    a, b = TEST
    return runs(d["diff"][a:b], d["speed"][a:b]), d["heat"][a:b]


def knn(train, labels, queries, k: int = K):
    """k-NN as the lesson's code does it: standardized with the mean and standard deviation of `train`, the k nearest
    by np.argsort with kind="stable", 1 when most of them failed. Only the runs that can be among the k nearest are
    sorted, so equally far ones keep the order np.argsort gives them; the distances are the lesson's own."""
    import numpy as np
    mean, std = train.mean(axis=0), train.std(axis=0)
    A, B = (train - mean) / std, (queries - mean) / std
    out = []
    for i in range(0, len(B), 250):
        dists = np.linalg.norm(A[None, :, :] - B[i:i + 250, None, :], axis=2)
        kth = np.partition(dists, k - 1, axis=1)[:, k - 1]
        for row, limit in zip(dists, kth):
            near = np.flatnonzero(row <= limit)
            near = near[np.argsort(row[near], kind="stable")][:k]
            out.append(int(np.mean(labels[near]) > 0.5))
    return np.array(out)


def heat_rule(diff, speed, speed_limit: int = SPEED_LIMIT):
    """1 for a heat failure: a temperature difference below 8.6 K and a speed below the limit."""
    import numpy as np
    return ((np.asarray(diff) < DIFF_LIMIT) & (np.asarray(speed) < speed_limit)).astype(float)


def simulate(seed: int = SIM_SEED, n: int = N_SIM, speed_limit: int = SPEED_LIMIT, low: float = LOW):
    """n simulated runs as the lesson's code makes them: first n temperature differences between 7.5 and 9.5 K, then
    n speeds picked from the log at random positions, labeled with the rule. Gives (the runs, their labels)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    diff = low + (HIGH - low) * rng.random(n)
    speed = _data()["speed"][:N_LOG][rng.integers(0, N_LOG, n)]
    return runs(diff, speed), heat_rule(diff, speed, speed_limit)


def counts(predicted, real) -> tuple:
    """(missed failures, false alarms)."""
    import numpy as np
    return int(np.sum(real[predicted == 0])), int(np.sum(predicted[real == 0]))


@functools.lru_cache(maxsize=None)
def result(speed_limit: int = SPEED_LIMIT, seed: int = SIM_SEED, low: float = LOW) -> dict:
    """Train k-NN on the log and 1000 simulated runs labeled with the given speed limit; test it on the next 2000 real
    runs and on 1000 more simulated runs, labeled with the same rule."""
    import numpy as np
    L, ly = log()
    S, sy = simulate(seed, N_SIM, speed_limit, low)
    train, labels = np.concatenate([L, S]), np.concatenate([ly, sy])
    T, ty = test()
    O, oy = simulate(OTHER_SEED, N_SIM, speed_limit)
    return {"made": int(np.sum(sy)), "real": counts(knn(train, labels, T), ty),
            "other": counts(knn(train, labels, O), oy), "other_made": int(np.sum(oy))}


@functools.lru_cache(maxsize=1)
def log_only() -> tuple:
    L, ly = log()
    T, ty = test()
    return counts(knn(L, ly, T), ty)


def said(limit: int, lang: str) -> str:
    """What the slider says at a speed limit."""
    r = result(limit)
    m, a = r["real"]
    sm, sa = r["other"]
    gap = abs(limit - SPEED_LIMIT)
    if lang == "nl":
        def alarm(x):
            return "geen vals alarm" if x == 0 else f"{x} keer vals alarm"
        text = (f"Met de grens op {limit} rpm heeft {r['made']} van de 1000 gesimuleerde metingen een hittestoring. Op de "
                f"volgende 2000 echte metingen mist k-NN er {m} van de 115 en geeft het {alarm(a)}; op 1000 andere "
                f"gesimuleerde metingen, met labels van dezelfde regel, mist het er {sm} van de {r['other_made']} en geeft "
                f"het {alarm(sa)}.")
        if limit == SPEED_LIMIT:
            return text + " Met deze grens is AI4I gemaakt."
        return text + f" AI4I is gemaakt met 1380 rpm, {gap} rpm {'hoger' if limit < SPEED_LIMIT else 'lager'}."
    def alarms(x):
        return "no false alarms" if x == 0 else f"{x} false alarms"
    text = (f"With the limit at {limit} rpm, {r['made']} of the 1000 simulated runs have a heat failure. On the next 2000 "
            f"real runs, k-NN misses {m} of the 115 and raises {alarms(a)}; on 1000 other simulated runs, labeled by "
            f"the same rule, it misses {sm} of {r['other_made']} and raises {alarms(sa)}.")
    if limit == SPEED_LIMIT:
        return text + " AI4I was made with this limit."
    return text + f" AI4I was made with 1380 rpm, {gap} rpm {'higher' if limit < SPEED_LIMIT else 'lower'}."


_WORDS = {
    "real": {"en": "On 2000 real runs", "nl": "Op 2000 echte metingen"},
    "other": {"en": "On 1000 simulated runs", "nl": "Op 1000 gesimuleerde metingen"},
    "limit": {"en": "speed limit of the rule (rpm)", "nl": "toerentalgrens van de regel (rpm)"},
    "missed": {"en": "missed", "nl": "gemist"},
    "alarms": {"en": "false alarms", "nl": "vals alarm"},
    "both": {"en": "missed and false alarms: both low", "nl": "gemist en vals alarm: allebei laag"},
}


def words(lang: str) -> dict:
    return {k: v[lang] for k, v in _WORDS.items()}


def picture(limit: int, w: dict, lang: str) -> str:
    """Left: the failures k-NN misses and the false alarms it raises on the next 2000 real runs, for every speed limit
    of the slider; right: the same on 1000 other simulated runs, labeled by the same rule. One scale for both, and a
    dot at the chosen limit. Drawn as SVG text, once for every stop of the slider (L.explore)."""
    panels = []
    for key in ("real", "other"):
        p = chart.Panel((1240, 1520), (0, 280), title=w[key], xlabel=w["limit"], xticks=[1300, 1380, 1460],
                        yticks=[0, 50, 100, 150, 200, 250])
        missed = [result(x)[key][0] for x in LIMITS]
        alarms = [result(x)[key][1] for x in LIMITS]
        p.line(LIMITS, missed, role="loss", width=2.4)
        p.line(LIMITS, alarms, role="loss", width=1.4, dashed=True)
        i = LIMITS.index(limit)
        p.points([limit, limit], [missed[i], alarms[i]], role="loss", r=5.5, hollow=True)
        if key == "real":
            p.text(1268, missed[0] + 16, w["missed"], role="loss", anchor="start", halo=True)
            p.text(1478, alarms[-1] - 22, w["alarms"], role="loss", anchor="end", halo=True)
        else:
            p.text(1262, 40, w["both"], role="loss", anchor="start", halo=True)
        panels.append(p)
    return chart.figure(panels, alt=w["alt"], width=330, height=250)


# The lesson is done when these are correct. "test-on" is the choice question in the lesson.
REQUIRED = ["power", "between", "test-on"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _shown(v):
    """What a function gave back, as the lesson writes it: a whole number as it is, a float with up to six digits,
    several numbers as a list in Python's own way, with a point in both languages."""
    import numpy as np
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, list):
        return "[" + ", ".join(_float(x) if isinstance(x, float) else str(x) for x in v) + "]"
    if isinstance(v, (bool, int)) or v is None:
        return str(v)
    if isinstance(v, float):
        text = _float(v)
        return {"en": text, "nl": text.replace(".", ",")}
    return repr(v)[:60]


def _float(x: float) -> str:
    text = f"{x:.6g}"
    return text + ".0" if text.lstrip("-").isdigit() else text


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


# --- exercise 1: a rule for power failures ------------------------------------------------------------------------------

_W_HEAD = "import math\n\n\ndef power_failure(torque, speed):\n    power = torque * speed * 2 * math.pi / 60   # in watts\n"


def _pw(body: str) -> str:
    return _W_HEAD + "".join("    " + line + "\n" for line in body.split("\n"))


_W_START = _pw("...")
_W = _pw("if power < 3500:\n    return 1\nif power > 9000:\n    return 1\nreturn 0")
_W_VISIBLE = (20.0, 1500)
_W_HIDDEN = [(40.0, 1500), (65.0, 1400), (30.0, 1100), (35.0, 1000), (70.0, 1200), (55.0, 1700), (12.0, 2800)]


def _is_label(v) -> bool:
    """A label the check accepts: 0 or 1, as a whole number, a float or True and False."""
    import numpy as np
    if isinstance(v, (bool, np.bool_)):
        return True
    return isinstance(v, (int, float, np.integer, np.floating)) and v in (0, 1)


def _same_label(got, want) -> bool:
    return _is_label(got) and got == want


def power_rule(torque, speed):
    """1 for a power failure, AI4I's own rule: a power below 3500 W or above 9000 W."""
    import numpy as np
    power = np.asarray(torque) * np.asarray(speed) * 2 * math.pi / 60
    return ((power < POWER_LOW) | (power > POWER_HIGH)).astype(int)


def shown_runs():
    """The runs in the picture of exercise 1: AI4I's 95 power failures and every tenth of the other runs, 991 more."""
    import numpy as np
    d = _data()
    fail = np.flatnonzero(d["power"] == 1)
    rest = np.flatnonzero(d["power"] == 0)[::10]
    return np.concatenate([fail, rest])


_P_WORDS = {
    "title": {"en": "Your rule on the 10,000 runs of AI4I", "nl": "Jouw regel op de 10.000 metingen van AI4I"},
    "plane": {"en": "Runs by torque and speed", "nl": "Metingen naar koppel en toerental"},
    "x": {"en": "torque (Nm)", "nl": "koppel (Nm)"},
    "y": {"en": "speed (rpm)", "nl": "toerental (rpm)"},
    "key": {"en": "Blue: a run your rule calls a power failure; gray: the rest; a purple ring: your rule and AI4I's label differ. Dashed: 3500 W and 9000 W. Shown are AI4I's 95 power failures and every tenth other run.",
            "nl": "Blauw: een meting die jouw regel een vermogensstoring noemt; grijs: de rest; een paarse ring: jouw regel en het label van AI4I verschillen. Gestreept: 3500 W en 9000 W. Je ziet de 95 vermogensstoringen van AI4I en elke tiende andere meting."},
    "said": {"en": "Your rule gives {yes} of the 10,000 runs a power failure, and the same label as AI4I for {same} of the 10,000.",
             "nl": "Jouw regel geeft {yes} van de 10.000 metingen een vermogensstoring, en hetzelfde label als AI4I bij {same} van de 10.000."},
}


def _labels_of(fn):
    """What the learner's rule gives each of the 10,000 runs, as 0 and 1; None if it is no label somewhere."""
    d = _data()

    def all_of_them():
        return [fn(float(t), int(v)) for t, v in zip(d["torque"], d["speed"])]

    got, ok = _quiet(all_of_them)
    if not ok or got is None or any(not _is_label(g) for g in got):
        return None
    return [int(g) for g in got]


def _pw_picture(seen_) -> str:
    import numpy as np
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    mine = _labels_of(fn)
    if mine is None:
        return ""
    mine = np.array(mine)
    d = _data()
    w = {k: v[lang] for k, v in _P_WORDS.items()}
    p = chart.Panel((0, 80), (1100, 2950), title=w["plane"], xlabel=w["x"], ylabel=w["y"], xticks=[0, 20, 40, 60, 80],
                    yticks=[1200, 1600, 2000, 2400, 2800])
    for watts in (POWER_LOW, POWER_HIGH):
        ts = [t / 2 for t in range(2, 161) if 1100 <= watts * 60 / (2 * math.pi * t / 2) <= 2950]
        p.line(ts, [watts * 60 / (2 * math.pi * t) for t in ts], role="ink-2", width=1, dashed=True)
    show = shown_runs()
    no = [j for j in show if mine[j] == 0]
    yes = [j for j in show if mine[j] == 1]
    p.points(d["torque"][no], d["speed"][no], role="ink-3", r=1.8)
    p.points(d["torque"][yes], d["speed"][yes], role="pred", r=2.6)
    off = [j for j in show if mine[j] != d["power"][j]]
    if off:
        p.points(d["torque"][off], d["speed"][off], role="loss", r=5, hollow=True)
    p.text(26.5, 1170, "3500 W", role="ink-2", anchor="end", halo=True)
    p.text(31.5, 2800, "9000 W", role="ink-2", anchor="start", halo=True)
    text = w["said"].format(yes=f"{int(np.sum(mine)):,}".replace(",", "." if lang == "nl" else ","),
                            same=f"{int(np.sum(mine == d['power'])):,}".replace(",", "." if lang == "nl" else ","))
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=300) + _key(w["key"] + " " + text), alt=text)


EXERCISES = {
    "power": FunctionBox(
        name="power",
        picture=_pw_picture,
        function="power_failure",
        start=_W_START,
        solution=_W,
        show=_shown,
        compare=_same_label,
        visible=[_W_VISIBLE],
        hidden=_W_HIDDEN,
        first=[_DOTS],
        mistakes=[
            Mistake("low-only", wrong=_pw("if power < 3500:\n    return 1\nreturn 0"),
                    en="`{call}` gives {got}, and it should give {want}: your rule only knows too little power. Too much power, above 9000 W, is a power failure too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je regel kent alleen te weinig vermogen. Te veel vermogen, boven 9000 W, is ook een vermogensstoring."),
            Mistake("high-only", wrong=_pw("if power > 9000:\n    return 1\nreturn 0"),
                    en="`{call}` gives {got}, and it should give {want}: your rule only knows too much power. Too little power, below 3500 W, is a power failure too.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je regel kent alleen te veel vermogen. Te weinig vermogen, onder 3500 W, is ook een vermogensstoring."),
            Mistake("inside", wrong=_pw("if power > 3500:\n    if power < 9000:\n        return 1\nreturn 0"),
                    en="`{call}` gives {got}, and it should give {want}: your rule calls a failure between 3500 and 9000 W, where the machine runs well. A failure is outside: below 3500 W, or above 9000 W.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je regel noemt een storing tussen 3500 en 9000 W, waar de machine goed draait. Een storing ligt erbuiten: onder 3500 W, of boven 9000 W."),
            Mistake("both", wrong=_pw("if power < 3500:\n    if power > 9000:\n        return 1\nreturn 0"),
                    en="`{call}` gives {got}, and it should give {want}: an `if` inside an `if` asks for both at once, and no power is below 3500 W and above 9000 W together. Two `if`s one after the other each give a failure.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een `if` binnen een `if` vraagt om allebei tegelijk, en geen vermogen is tegelijk onder 3500 W en boven 9000 W. Twee `if`s na elkaar geven elk een storing."),
            Mistake("always", wrong=_pw("return 1"),
                    en="`{call}` gives {got}, and it should give {want}: your rule calls every run in the check a power failure. Look at each limit: is it in watts, and does `<` or `>` point the right way, below 3500 W and above 9000 W?",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je regel noemt elke meting in de check een vermogensstoring. Kijk naar elke grens: staat die in watt, en wijst `<` of `>` de goede kant op, onder 3500 W en boven 9000 W?"),
        ],
        texts=_SAME | {
            "empty": {"en": "Replace the `...` with your rule: 1 for a power below 3500 W or above 9000 W, and 0 otherwise.",
                      "nl": "Vervang de `...` door je regel: 1 bij een vermogen onder 3500 W of boven 9000 W, en anders 0."},
            "arguments": {"en": "The check calls `{call}`, with two values, and the `def` line of your function has a different number of names between its parentheses. Keep `(torque, speed)`.",
                          "nl": "De check roept `{call}` aan, met twee waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(torque, speed)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: 1 when the power is below 3500 W or above 9000 W, and 0 otherwise.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: 1 als het vermogen onder 3500 W of boven 9000 W is, en anders 0."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `power`, for both limits?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `power`, voor allebei de grenzen?"},
            "correct": {"en": "Your rule is right in all {n} cases. The example runs at 3142 W, too little: a power failure.",
                        "nl": "Je regel klopt in alle {n} gevallen. Het voorbeeld draait op 3142 W, te weinig: een vermogensstoring."},
        },
    ),
}


# --- exercise 2: numbers between two limits -----------------------------------------------------------------------------

class _Rng:
    """A generator the check's messages write as `np.random.default_rng(3)`; the function gets a new one, so every
    call draws the same numbers."""

    def __init__(self, seed: int):
        self.seed = seed

    def __repr__(self):
        return f"np.random.default_rng({self.seed})"

    def __deepcopy__(self, memo):
        import numpy as np
        return np.random.default_rng(self.seed)


_B_HEAD = "import numpy as np\n\n\ndef between(rng, low, high, n):\n"


def _bw(back: str) -> str:
    return _B_HEAD + f"    return {back}\n"


_B_START = _bw("...")
_B = _bw("low + (high - low) * rng.random(n)")
_B_VISIBLE = (_Rng(1), 7.5, 9.5, 3)
_B_HIDDEN = [(_Rng(2), 1100, 2900, 4), (_Rng(3), 0, 1, 3), (_Rng(4), -2, 2, 5), (_Rng(5), 10, 12, 1), (_Rng(6), 7.5, 9.5, 6)]


def _same_numbers(got, want) -> bool:
    import numpy as np
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w))


_B_WORDS = {
    "title": {"en": "Your simulated temperature differences", "nl": "Jouw gesimuleerde temperatuurverschillen"},
    "bars": {"en": "Runs per temperature difference", "nl": "Metingen per temperatuurverschil"},
    "x": {"en": "process minus air temperature (K)", "nl": "procestemperatuur min luchttemperatuur (K)"},
    "rule": {"en": "8.6 K", "nl": "8,6 K"},
    "key": {"en": "Filled: the 3000 runs of the log; open: the 1000 differences of <code>between(rng, 7.5, 9.5, 1000)</code>, with the lesson's seed 4.",
            "nl": "Gevuld: de 3000 metingen van het logboek; open: de 1000 verschillen van <code>between(rng, 7.5, 9.5, 1000)</code>, met seed 4 van de les."},
    "outside": {"en": " {o} of them lie outside the chart, below 7 K or above 13 K.", "nl": " Daarvan liggen er {o} buiten de grafiek, onder 7 K of boven 13 K."},
    "said": {"en": "{a} of your 1000 temperature differences are below 8.6 K, where the rule can give a heat failure; in the log, none is.",
             "nl": "{a} van jouw 1000 temperatuurverschillen liggen onder 8,6 K, waar de regel een hittestoring kan geven; in het logboek niet één."},
}


def _bw_picture(seen_) -> str:
    import numpy as np
    lang = seen_.lang
    fn = _function(seen_)
    if fn is None:
        return ""
    got, ok = _quiet(fn, np.random.default_rng(SIM_SEED), LOW, HIGH, N_SIM)
    try:
        values = np.asarray(got, dtype=float) if ok else None
    except (TypeError, ValueError):
        values = None
    if values is None or values.shape != (N_SIM,) or not np.all(np.isfinite(values)):
        return ""
    w = {k: v[lang] for k, v in _B_WORDS.items()}
    edges = [7.0 + 0.2 * b for b in range(31)]           # 7.0 to 13.0 K, with 8.6 K on an edge
    logged = _data()["diff"][:N_LOG]
    bars = []
    for a, b in zip(edges[:-1], edges[1:]):
        bars.append((a, b, int(np.sum((logged >= a) & (logged < b))), int(np.sum((values >= a) & (values < b)))))
    top = max(max(n1, n2) for _, _, n1, n2 in bars)
    high = 50 * (top // 50 + 2)
    p = chart.Panel((7.0, 13.0), (0, high), title=w["bars"], xlabel=w["x"], xticks=[7, 8, 9, 10, 11, 12, 13],
                    yticks=list(range(0, high, 100 if high > 300 else 50)))
    for a, b, nl_, ny in bars:
        if nl_:
            p.rect(a + 0.01, 0, (a + b) / 2, nl_, role="data", solid=True)   # the log on the left half of the bin
        if ny:
            p.rect((a + b) / 2, 0, b - 0.01, ny, role="data")                 # yours on the right half
    p.line([DIFF_LIMIT, DIFF_LIMIT], [0, high], role="ink-2", width=1, dashed=True)
    p.text(DIFF_LIMIT - 0.08, high * 0.92, w["rule"], role="ink-2", anchor="end", halo=True)
    text = w["said"].format(a=int(np.sum(values < DIFF_LIMIT)))
    outside = int(np.sum((values < 7.0) | (values >= 13.0)))
    if outside:
        text += w["outside"].format(o=outside)
    return yours.figure(w["title"], chart.figure([p], alt=text, width=460, height=240) + _key(w["key"] + " " + text), alt=text)


EXERCISES["between"] = FunctionBox(
    name="between",
    picture=_bw_picture,
    function="between",
    start=_B_START,
    solution=_B,
    show=_shown,
    compare=_same_numbers,
    visible=[_B_VISIBLE],
    hidden=_B_HIDDEN,
    first=[_DOTS],
    mistakes=[
        Mistake("one-number", matches=lambda got, text: _one_number(got),
                en="`{call}` gives {got}: one number. `rng.random(n)` gives `n` of them at once, and the arithmetic then works on each.",
                nl="`{call}` geeft {got}: één getal. `rng.random(n)` geeft er `n` tegelijk, en het rekenwerk werkt dan op elk."),
        Mistake("no-low", wrong=_bw("(high - low) * rng.random(n)"),
                en="`{call}` gives {got}, and it should give {want}: your numbers start at 0, not at `low`. Add `low`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je getallen beginnen bij 0, niet bij `low`. Tel `low` erbij op."),
        Mistake("times-high", wrong=_bw("low + high * rng.random(n)"),
                en="`{call}` gives {got}, and it should give {want}: `high * rng.random(n)` is as wide as `high` itself, so your numbers go up to `low + high`. The width is `high - low`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: `high * rng.random(n)` is zo breed als `high` zelf, dus je getallen gaan tot `low + high`. De breedte is `high - low`."),
        Mistake("high-only", wrong=_bw("high * rng.random(n)"),
                en="`{call}` gives {got}, and it should give {want}: your numbers go from 0 to `high`. Start at `low`, and take the width `high - low`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je getallen gaan van 0 tot `high`. Begin bij `low`, en neem de breedte `high - low`."),
        Mistake("whole", wrong=_bw("rng.integers(low, high, n)"),
                en="`{call}` gives {got}, and it should give {want}: `rng.integers` gives whole numbers. A temperature difference can be 8.27 K: take `rng.random(n)`, numbers between 0 and 1, and stretch them.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: `rng.integers` geeft gehele getallen. Een temperatuurverschil kan 8,27 K zijn: neem `rng.random(n)`, getallen tussen 0 en 1, en rek ze op."),
        Mistake("from-top", wrong=_bw("high - (high - low) * rng.random(n)"),
                en="`{call}` gives {got}, and it should give {want}: your numbers are between `low` and `high` too, but counted down from `high`. Count up from `low`, so that your simulation draws the same numbers as the lesson's.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je getallen liggen ook tussen `low` en `high`, maar geteld vanaf `high` naar beneden. Tel op vanaf `low`, zodat je simulatie dezelfde getallen trekt als die van de les."),
    ],
    texts=_SAME | {
        "empty": {"en": "Replace the `...` with `n` numbers between `low` and `high`, from `rng.random(n)`.",
                  "nl": "Vervang de `...` door `n` getallen tussen `low` en `high`, uit `rng.random(n)`."},
        "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(rng, low, high, n)`.",
                      "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(rng, low, high, n)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}: `n` numbers between `low` and `high`, each `low` plus a share of the width `high - low`.",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn: `n` getallen tussen `low` en `high`, elk `low` plus een deel van de breedte `high - low`."},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does your function use `rng`, `low`, `high` and `n`, the values it gets?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie `rng`, `low`, `high` en `n`, de waarden die ze krijgt?"},
        "correct": {"en": "Your function is right in all {n} cases. Every number is `low` plus a random share of the width: the simulation of the lesson draws its temperature differences this way.",
                    "nl": "Je functie klopt in alle {n} gevallen. Elk getal is `low` plus een willekeurig deel van de breedte: zo trekt de simulatie van de les haar temperatuurverschillen."},
    },
)


def _one_number(got) -> bool:
    import numpy as np
    try:
        return np.ndim(got) == 0 and isinstance(float(got), float)
    except (TypeError, ValueError):
        return False


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("power", _W_START, "empty", None),
    ("power", _pw("if power < 3500:\n    return 1\n..."), "wrong", "dots-left"),
    ("power", _W, "correct", None),
    ("power", _pw("if power < 3500:\n    return True\nif power > 9000:\n    return True\nreturn False"), "correct", None),
    ("power", _pw("if power < 3500:\n    return 1\nelif power > 9000:\n    return 1\nelse:\n    return 0"), "correct", None),
    ("power", _pw("failure = 0\nif power < 3500:\n    failure = 1\nif power > 9000:\n    failure = 1\nreturn failure"), "correct", None),
    ("power", _pw("if power >= 3500:\n    if power <= 9000:\n        return 0\nreturn 1"), "correct", None),
    ("power", _pw("if power < 3500:\n    return 1\nif power > 9000:\n    return 1"), "wrong", "unreached"),
    ("power", _pw("if power < 3500:\n    return 1\nreturn 0"), "wrong", "low-only"),
    ("power", _pw("if power > 9000:\n    return 1\nreturn 0"), "wrong", "high-only"),
    ("power", _pw("if power > 3500:\n    if power < 9000:\n        return 1\nreturn 0"), "wrong", "inside"),
    ("power", _pw("if power < 3500:\n    if power > 9000:\n        return 1\nreturn 0"), "wrong", "both"),
    ("power", _pw("if power < 3.5:\n    return 1\nif power > 9:\n    return 1\nreturn 0"), "wrong", "always"),
    ("power", _pw("if power > 3500:\n    return 1\nif power < 9000:\n    return 1\nreturn 0"), "wrong", "always"),
    ("between", _B_START, "empty", None),
    ("between", _B, "correct", None),
    ("between", _bw("rng.random(n) * (high - low) + low"), "correct", None),
    ("between", _bw("rng.uniform(low, high, n)"), "correct", None),
    ("between", _B_HEAD + "    width = high - low\n    return low + width * rng.random(n)\n", "correct", None),
    ("between", _bw("low + (high - low) * rng.random()"), "wrong", "one-number"),
    ("between", _bw("(high - low) * rng.random(n)"), "wrong", "no-low"),
    ("between", _bw("low + high * rng.random(n)"), "wrong", "times-high"),
    ("between", _bw("high * rng.random(n)"), "wrong", "high-only"),
    ("between", _bw("rng.integers(low, high, n)"), "wrong", "whole"),
    ("between", _bw("high - (high - low) * rng.random(n)"), "wrong", "from-top"),
]
