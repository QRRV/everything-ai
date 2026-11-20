# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-reliability: robust, resilient, available, predictable. The course's own softmax
regression of the project lesson ml-digits-linear, trained the same way on the same 6,000 QMNIST digits, and the 1,000
test digits; three made-up agents with the success rate of each of their ten tasks. The learner writes
noisy_accuracy, the share of test digits the model still reads right after noise is added, and pass_hat, the chance
that all k tries at a task succeed, averaged over the tasks (Yao and colleagues, tau-bench, 2024)."""
import contextlib
import copy
import functools
import html
import io
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
K, LR, STEPS = 10, 1.0, 100       # as in ml-digits-linear: ten digits, learning rate 1, 100 steps

# Hendrycks and Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations" (ICLR
# 2019): AlexNet's top-1 error on clean ImageNet (Table 1) and on Gaussian noise, averaged over five severities (p. 13)
ALEXNET_CLEAN_ERROR, ALEXNET_NOISE_ERROR = 43.5, 88.6
CORRUPTIONS, SEVERITIES = 15, 5

# Yao and colleagues, "tau-bench" (2024), p. 2: gpt-4o on tau-retail, about 61% of the tasks on one try, and about 25%
# for pass^8
RETAIL_PASS_1, RETAIL_PASS_8 = 0.61, 0.25

# Three made-up agents, each with the success rate of each of its ten tasks; all three succeed on 60% of the tries
STEADY = [0.6] * 10
SPLIT = [1.0] * 6 + [0.0] * 4
AGENT = [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.0]
AGENTS = {"STEADY": STEADY, "AGENT": AGENT, "SPLIT": SPLIT}
KS = [1, 2, 4, 8]

SHIFTS = [0, 1, 2, 3, 4, 5]          # the demo: pixels to the right
SIGMAS = [0.0, 0.25, 0.5, 1.0]       # the picture under the check of noisy_accuracy


@functools.lru_cache(maxsize=1)
def _data():
    """The training images and labels and the test images and labels; every pixel divided by 255, so from 0 to 1."""
    import numpy as np
    d = np.load(paths.ROOT / DATA)
    return d["train_images"] / 255, d["train_labels"], d["test_images"] / 255, d["test_labels"]


@functools.lru_cache(maxsize=1)
def model():
    """The course's softmax regression, trained as in ml-digits-linear: W (10 by 784) and b (10)."""
    import numpy as np
    X, y, _, _ = _data()
    W, b = np.zeros((K, X.shape[1])), np.zeros(K)
    for _ in range(STEPS):
        Z = X @ W.T + b
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        diff = E / E.sum(axis=1, keepdims=True) - np.eye(K)[y]
        W, b = W - LR * diff.T @ X / len(y), b - LR * diff.mean(axis=0)
    return W, b


def test_set():
    """The 1,000 test digits, one row of 784 pixels each, and their labels."""
    _, _, X, y = _data()
    return X, y


def accuracy(X, y) -> float:
    """The share of the images in X the course's model reads right."""
    import numpy as np
    W, b = model()
    return float(np.mean(np.argmax(X @ W.T + b, axis=1) == y))


def shifted(X, pixels: int):
    """Every image of X moved `pixels` to the right; the columns that come in on the left are black (0)."""
    import numpy as np
    images = np.asarray(X).reshape(-1, 28, 28)
    out = np.zeros_like(images)
    out[:, :, pixels:] = images[:, :, :28 - pixels]
    return out.reshape(len(images), 784)


@functools.lru_cache(maxsize=1)
def shift_accuracies() -> tuple:
    X, y = test_set()
    return tuple(accuracy(shifted(X, s), y) for s in SHIFTS)


def noisy_accuracy(W, b, X, y, sigma):
    """What the learner writes in Exercise 1."""
    import numpy as np
    rng = np.random.default_rng(0)
    noisy = X + rng.normal(0, sigma, size=X.shape)
    predicted = np.argmax(noisy @ W.T + b, axis=1)
    return np.mean(predicted == y)


def pass_hat(rates, k):
    """What the learner writes in Exercise 2: for each task, the chance that all k tries succeed, averaged."""
    total = 0
    for p in rates:
        total = total + p ** k
    return total / len(rates)


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _shown(v) -> str:
    """A result as the pictures show it: a number with at most four decimals, else what Python shows."""
    import numpy as np
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        text = repr(v)
        return text if len(text) <= 40 else text[:37] + "..."
    return f"{float(v):.4f}".rstrip("0").rstrip(".") if float(v) == float(v) else "nan"


def _same(a, b) -> bool:
    import numpy as np
    try:
        return bool(abs(float(a) - float(b)) <= 1e-9 + 1e-6 * abs(float(b))) and not isinstance(a, (bool, np.bool_))
    except (TypeError, ValueError):
        return False


_DEMO_WORDS = {
    "en": {"title": "The test digits, {n} pixels to the right", "title_1": "The test digits, 1 pixel to the right",
           "title_0": "The test digits, not shifted",
           "said": "Shifted {n} pixels to the right, the model reads {k} of the 1,000 test digits right, against {k0} "
                   "not shifted.",
           "said_1": "Shifted 1 pixel to the right, the model reads {k} of the 1,000 test digits right, against {k0} "
                     "not shifted.",
           "said_0": "Not shifted, the model reads {k0} of the 1,000 test digits right.",
           "chart": "Test digits read right, of 1,000, for each shift in pixels",
           "note": "Under each digit: what the model says, and in brackets the right digit where it is wrong. The bar "
                   "of the shift on the slider stands out."},
    "nl": {"title": "De testcijfers, {n} pixels naar rechts", "title_1": "De testcijfers, 1 pixel naar rechts",
           "title_0": "De testcijfers, niet geschoven",
           "said": "{n} pixels naar rechts geschoven leest het model {k} van de 1000 testcijfers goed, tegen {k0} niet "
                   "geschoven.",
           "said_1": "1 pixel naar rechts geschoven leest het model {k} van de 1000 testcijfers goed, tegen {k0} niet "
                     "geschoven.",
           "said_0": "Niet geschoven leest het model {k0} van de 1000 testcijfers goed.",
           "chart": "Testcijfers goed gelezen, van de 1000, per aantal pixels naar rechts",
           "note": "Onder elk cijfer: wat het model zegt, en tussen haakjes het goede cijfer als het fout is. De balk "
                   "van het aantal pixels op de schuif valt op."},
}


def _bars(values: list, now: int, label: str) -> str:
    """Bars of the number read right for each shift; the one on the slider in ink, the others in the line color."""
    W, H, left, base, top = 360, 150, 34, 120, 14
    step = (W - left - 8) / len(values)
    parts = [f'<line x1="{left}" y1="{base}" x2="{W - 4}" y2="{base}" style="stroke:var(--ea-axis)"/>']
    for tick in (0, 500, 1000):
        y = base - tick / 1000 * (base - top)
        parts.append(f'<text x="{left - 5}" y="{y + 4:.1f}" style="font-size:10.5px;fill:var(--ea-ink-2);'
                     f'text-anchor:end">{tick}</text>')
        if tick:
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - 4}" y2="{y:.1f}" style="stroke:var(--ea-line);'
                         f'stroke-dasharray:2 3"/>')
    for i, v in enumerate(values):
        x = left + i * step + step * 0.18
        h = v / 1000 * (base - top)
        fill = "var(--ea-ink)" if i == now else "var(--ea-line)"
        parts.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{step * 0.64:.1f}" height="{h:.1f}" '
                     f'style="fill:{fill}"/>')
        weight = "font-weight:600;" if i == now else ""
        parts.append(f'<text x="{x + step * 0.32:.1f}" y="{base - h - 4:.1f}" style="font-size:10.5px;{weight}'
                     f'fill:var(--ea-ink);text-anchor:middle">{v}</text>')
        parts.append(f'<text x="{x + step * 0.32:.1f}" y="{base + 15}" style="font-size:11px;fill:var(--ea-ink-2);'
                     f'text-anchor:middle">{i}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(label)}" style="width:{W}px;'
            f'max-width:100%;height:auto;display:block;font-family:var(--ea-font-ui)">{"".join(parts)}</svg>')


def demo_frame(pixels, lang: str) -> tuple[str, str]:
    """The demo: the first eight test digits moved `pixels` to the right with what the course's model says, and the
    number of test digits it reads right for every shift."""
    import numpy as np
    n = int(min(SHIFTS, key=lambda s: abs(s - float(pixels))))
    w = _DEMO_WORDS[lang]
    X, y = test_set()
    W, b = model()
    moved = shifted(X[:8], n)
    said_digits = np.argmax(moved @ W.T + b, axis=1)
    counts = [round(a * 1000) for a in shift_accuracies()]
    title = w["title_0"] if n == 0 else (w["title_1"] if n == 1 else w["title"].format(n=n))
    rows = yours.guesses([m.reshape(28, 28) for m in moved], [int(d) for d in said_digits], [int(d) for d in y[:8]],
                         alt=title, size=56)
    key = "said_0" if n == 0 else ("said_1" if n == 1 else "said")
    said = w[key].format(n=n, k=counts[n], k0=counts[0])
    body = rows + f'<div style="{_UI}">{html.escape(w["chart"])}</div>' + _bars(counts, n, w["chart"]) + _note(w["note"])
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


_NOISY_WORDS = {
    "en": {"title": "Your noisy_accuracy beside the course's",
           "head": ["noise σ", "the course's", "yours"], "stopped": "error",
           "images": "The first six test digits with noise of σ = 0.5, from the generator with seed 0",
           "note": "A dashed row is one where your noisy_accuracy gives something other than the course's."},
    "nl": {"title": "Jouw noisy_accuracy naast die van de course",
           "head": ["ruis σ", "van de course", "de jouwe"], "stopped": "fout",
           "images": "De eerste zes testcijfers met ruis van σ = 0,5, uit de generator met seed 0",
           "note": "Een gestippelde rij is er een waar jouw noisy_accuracy iets anders geeft dan die van de course."},
}


def _decimal(text: str, lang: str) -> str:
    return text.replace(".", ",") if lang == "nl" else text


def noisy_picture(seen) -> str:
    """The course's accuracy under noise of four strengths beside the learner's, and six noisy test digits."""
    import numpy as np
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _NOISY_WORDS[seen.lang]
    W, b = model()
    X, y = test_set()
    got = _all(fn, [(W, b, X, y, s) for s in SIGMAS])
    rows = []
    for i, s in enumerate(SIGMAS):
        want = noisy_accuracy(W, b, X, y, s)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_decimal(_shown(got[i]), seen.lang)), not _same(got[i], want)
        rows.append(([(_decimal(f"{s:g}", seen.lang), ""), (_code(_decimal(_shown(want), seen.lang)), ""),
                      (shown, "font-weight:600")], _OFF if off else ""))
    noisy = X[:6] + np.random.default_rng(0).normal(0, 0.5, size=X.shape)[:6]
    images = yours.guesses([np.clip(m, 0, 1).reshape(28, 28) for m in noisy], [""] * 6, [""] * 6,
                           alt=w["images"], size=48)
    body = (_table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"]) + f'<div style="{_UI}">'
            f'{html.escape(w["images"])}</div>' + images)
    return yours.figure(w["title"], body, alt=w["title"])


_PASS_WORDS = {
    "en": {"title": "pass^k of three agents, yours beside the course's",
           "head": ["agent", ""] + [f"k = {k}" for k in KS], "course": "the course's", "yours": "yours",
           "stopped": "error",
           "names": {"STEADY": "STEADY: every task 60%", "AGENT": "AGENT: from 100% to 0%",
                     "SPLIT": "SPLIT: six tasks always, four never"},
           "note": "All three succeed on 60% of the tries. A dashed cell is one where your pass_hat gives something "
                   "other than the course's."},
    "nl": {"title": "pass^k van drie agents, de jouwe naast die van de course",
           "head": ["agent", ""] + [f"k = {k}" for k in KS], "course": "van de course", "yours": "de jouwe",
           "stopped": "fout",
           "names": {"STEADY": "STEADY: elke taak 60%", "AGENT": "AGENT: van 100% tot 0%",
                     "SPLIT": "SPLIT: zes taken altijd, vier nooit"},
           "note": "Alle drie slagen in 60% van de pogingen. Een gestippelde cel is er een waar jouw pass_hat iets "
                   "anders geeft dan die van de course."},
}


def pass_picture(seen) -> str:
    """For the three agents and k = 1, 2, 4 and 8: the course's pass^k and the learner's under it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PASS_WORDS[seen.lang]
    calls = [(AGENTS[name], k) for name in AGENTS for k in KS]
    got = _all(fn, calls)
    rows = []
    for a, name in enumerate(AGENTS):
        wants = [pass_hat(AGENTS[name], k) for k in KS]
        rows.append(([(html.escape(w["names"][name]), ""), (html.escape(w["course"]), "color:var(--ea-ink-2)")]
                     + [(_code(_decimal(_shown(v), seen.lang)), "") for v in wants], ""))
        cells = [("", ""), (html.escape(w["yours"]), "font-weight:600")]
        for j, want in enumerate(wants):
            if got is _STOPPED:
                cells.append((html.escape(w["stopped"]), _OFF))
            else:
                v = got[a * len(KS) + j]
                cells.append((_code(_decimal(_shown(v), seen.lang)), "font-weight:600" + ("" if _same(v, want)
                                                                                         else ";" + _OFF)))
        rows.append((cells, ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["noisy_accuracy", "pass_hat", "test", "uptime"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(AGENT|STEADY|SPLIT|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `AGENT` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `AGENT` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


class _Lazy:
    """The course's model or test set, made when a check first needs it, so the lesson opens without training."""

    def __init__(self, name: str, get):
        self.name, self.get = name, get

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.get(), memo)


_W = _Lazy("W", lambda: model()[0])
_B = _Lazy("b", lambda: model()[1])
_X = _Lazy("X", lambda: test_set()[0])
_Y = _Lazy("y", lambda: test_set()[1])
_SMALL = ([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.2], [0.1, 0.9], [0.8, 0.3], [0.4, 0.6]], [0, 1, 1, 1])


def _small(sigma, bias=(0.0, 0.0)):
    """A model of two classes and four images of two pixels, each argument shown as the code that makes it."""
    import numpy as np
    W, X, y = _SMALL
    return tuple(Named(f"np.array({v!r})", np.array(v)) for v in (W, list(bias), X, y)) + (sigma,)


# exercise 1: noisy_accuracy; the first case is the course's model on the 1,000 test digits with noise of 0.5
NOISY_CASES = [(_W, _B, _X, _Y, 0.5), (_W, _B, _X, _Y, 0.0), (_W, _B, _X, _Y, 1.0), (_W, _B, _X, _Y, 0.25),
               _small(0.0), _small(0.0, (0.0, 0.6)), _small(0.3)]
_N_HEAD = "import numpy as np\n\n\n"
_N_START = _N_HEAD + "def noisy_accuracy(W, b, X, y, sigma):\n    ...\n"
_N = (_N_HEAD + "def noisy_accuracy(W, b, X, y, sigma):\n"
      "    rng = np.random.default_rng(0)\n"
      "    noisy = X + rng.normal(0, sigma, size=X.shape)\n"
      "    predicted = np.argmax(noisy @ W.T + b, axis=1)\n"
      "    return np.mean(predicted == y)\n")
_N_CLEAN = _N.replace("noisy @ W.T + b", "X @ W.T + b")
_N_UNIFORM = _N.replace("rng.normal(0, sigma, size=X.shape)", "sigma * rng.random(size=X.shape)")
_N_COUNT = _N.replace("    return np.mean(predicted == y)\n", "    return np.sum(predicted == y)\n")
_N_NOBIAS = _N.replace("noisy @ W.T + b", "noisy @ W.T")
_N_VARIANCE = _N.replace("rng.normal(0, sigma, size=X.shape)", "rng.normal(0, sigma ** 2, size=X.shape)")

# exercise 2: pass_hat; the first case is the made-up AGENT with k = 8
PASS_CASES = [(Named("AGENT", AGENT), 8), (Named("STEADY", STEADY), 8), (Named("SPLIT", SPLIT), 8),
              (Named("AGENT", AGENT), 1), ([0.9, 0.5], 2), ([1.0], 5), ([0.5, 0.5, 0.0, 1.0], 3)]
_P_START = "def pass_hat(rates, k):\n    ...\n"
_P = ("def pass_hat(rates, k):\n"
      "    total = 0\n"
      "    for p in rates:\n"
      "        total = total + p ** k\n"
      "    return total / len(rates)\n")
_P_MEAN_FIRST = _P.replace("        total = total + p ** k\n    return total / len(rates)\n",
                           "        total = total + p\n    return (total / len(rates)) ** k\n")
_P_ONCE = _P.replace("total + p ** k", "total + p")
_P_TIMES = _P.replace("total + p ** k", "total + p * k")
_P_SUM = _P.replace("    return total / len(rates)\n", "    return total\n")
_P_EARLY = _P.replace("    return total / len(rates)\n", "        return total / len(rates)\n")


EXERCISES = {
    "noisy_accuracy": FunctionBox(
        name="noisy_accuracy",
        picture=noisy_picture,
        function="noisy_accuracy",
        start=_N_START,
        solution=_N,
        visible=[NOISY_CASES[0]],
        hidden=NOISY_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("clean", wrong=_N_CLEAN,
                    en="`{call}` gives {got}, and it should give {want}: that is the accuracy on the clean digits. Compute the logits of the noisy images.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de accuracy op de schone cijfers. Reken de logits uit van de beelden met ruis."),
            Mistake("uniform", wrong=_N_UNIFORM,
                    en="`{call}` gives {got}, and it should give {want}: `rng.random` gives numbers from 0 to 1, all positive, so every pixel gets lighter. Noise around 0 comes from `rng.normal`, with mean 0 and standard deviation `sigma`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `rng.random` geeft getallen van 0 tot 1, allemaal positief, dus elke pixel wordt lichter. Ruis rond 0 komt van `rng.normal`, met gemiddelde 0 en standaardafwijking `sigma`."),
            Mistake("count", wrong=_N_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of digits read right. Return the share: the mean of the `True`s and `False`s.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het aantal goed gelezen cijfers. Geef het deel terug: het gemiddelde van de `True`'s en `False`'s."),
            Mistake("bias", wrong=_N_NOBIAS,
                    en="`{call}` gives {got}, and it should give {want}: the logits leave out the bias terms. Add `b`, as in the worked example.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de logits laten de bias terms weg. Tel `b` erbij op, zoals in het uitgewerkte voorbeeld."),
            Mistake("variance", wrong=_N_VARIANCE,
                    en="`{call}` gives {got}, and it should give {want}: your noise has standard deviation `sigma ** 2`. Give `rng.normal` the standard deviation itself, `sigma`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je ruis heeft standaardafwijking `sigma ** 2`. Geef `rng.normal` de standaardafwijking zelf, `sigma`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: make a generator with seed 0, add noise with standard deviation `sigma` to `X`, let the model say a digit for every noisy image, and return the share it gets right.",
                      "nl": "Vervang de `...`: maak een generator met seed 0, tel ruis met standaardafwijking `sigma` op bij `X`, laat het model voor elk beeld met ruis een cijfer zeggen, en geef het deel terug dat het goed heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Use `np.random.default_rng(0)` and its `normal(0, sigma, size=X.shape)`, take the largest logit in each row, and return the mean of the `True`s and `False`s.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Gebruik `np.random.default_rng(0)` en de `normal(0, sigma, size=X.shape)` ervan, neem in elke rij de grootste logit, en geef het gemiddelde van de `True`'s en `False`'s terug."},
            "almost": {"en": "Right for the course's model with noise of 0.5. But `{call}` gives {got}, and it should give {want}. Does your function use the `W`, `b`, `X`, `y` and `sigma` it gets?",
                       "nl": "Goed voor het model van de course met ruis van 0,5. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `W`, `b`, `X`, `y` en `sigma` die ze krijgt?"},
            "correct": {"en": "Your `noisy_accuracy` is right in all {n} cases. Under the check, your numbers stand beside the course's for four strengths of noise.",
                        "nl": "Je `noisy_accuracy` klopt in alle {n} gevallen. Onder de check staan jouw getallen naast die van de course, voor vier sterktes van ruis."},
        },
    ),
    "pass_hat": FunctionBox(
        name="pass_hat",
        picture=pass_picture,
        function="pass_hat",
        start=_P_START,
        solution=_P,
        visible=[PASS_CASES[0]],
        hidden=PASS_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("mean-first", wrong=_P_MEAN_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: you take the power of the average rate. Take the power of each task's rate first, then the average.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de macht van het gemiddelde. Neem eerst de macht van het percentage van elke taak, dan het gemiddelde."),
            Mistake("once", wrong=_P_ONCE,
                    en="`{call}` gives {got}, and it should give {want}: that is the average rate, the chance for one try. For all `k` tries, each rate goes to the power `k`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het gemiddelde percentage, de kans bij één poging. Voor alle `k` pogingen gaat elk percentage tot de macht `k`."),
            Mistake("times", wrong=_P_TIMES,
                    en="`{call}` gives {got}, and it should give {want}: you multiply each rate by `k`. All `k` tries must succeed: the rate times itself, `k` times.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vermenigvuldigt elk percentage met `k`. Alle `k` pogingen moeten slagen: het percentage `k` keer met zichzelf vermenigvuldigd."),
            Mistake("sum", wrong=_P_SUM,
                    en="`{call}` gives {got}, and it should give {want}: that is the sum over the tasks. Divide it by the number of tasks.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is de som over de taken. Deel die door het aantal taken."),
            Mistake("early", wrong=_P_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your `return` is inside the loop, so the function stops after the first task. Move `return` back, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `return` staat in de loop, dus de functie stopt na de eerste taak. Zet `return` terug, op gelijke hoogte met `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for each task, the chance that all `k` tries succeed; add them up and divide by the number of tasks.",
                      "nl": "Vervang de `...`: voor elke taak de kans dat alle `k` pogingen slagen; tel ze op en deel door het aantal taken."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Add up each task's rate to the power `k`, and divide by the number of tasks.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel het percentage van elke taak tot de macht `k` op, en deel door het aantal taken."},
            "almost": {"en": "Right for AGENT with k = 8. But `{call}` gives {got}, and it should give {want}. Does your function use the `rates` and `k` it gets?",
                       "nl": "Goed voor AGENT met k = 8. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `rates` en `k` die ze krijgt?"},
            "correct": {"en": "Your `pass_hat` is right in all {n} cases. Under the check, your numbers stand under the course's for three agents.",
                        "nl": "Je `pass_hat` klopt in alle {n} gevallen. Onder de check staan jouw getallen onder die van de course, voor drie agents."},
        },
    ),
}

CASES = [
    ("noisy_accuracy", _N_START, "empty", None),
    ("noisy_accuracy", _N, "correct", None),
    ("noisy_accuracy", _N_HEAD + "def noisy_accuracy(W, b, X, y, sigma):\n    rng = np.random.default_rng(0)\n    noisy = X + sigma * rng.normal(size=X.shape)\n    logits = noisy @ W.T + b\n    right = logits.argmax(axis=1) == y\n    return right.mean()\n", "correct", None),
    ("noisy_accuracy", _N_CLEAN, "wrong", "clean"),
    ("noisy_accuracy", _N_UNIFORM, "wrong", "uniform"),
    ("noisy_accuracy", _N_COUNT, "wrong", "count"),
    ("noisy_accuracy", _N_NOBIAS, "wrong", "bias"),
    ("noisy_accuracy", _N_VARIANCE, "wrong", "variance"),
    ("noisy_accuracy", _N_HEAD + "def noisy_accuracy(W, b, X, y, sigma):\n    return 0.82\n", "almost", None),
    ("noisy_accuracy", _N_HEAD + "def noisy_accuracy(W, b, X, y, sigma):\n    return C.accuracy(X, y)\n", "error", "lesson-names"),
    ("pass_hat", _P_START, "empty", None),
    ("pass_hat", _P, "correct", None),
    ("pass_hat", "def pass_hat(rates, k):\n    chances = []\n    for p in rates:\n        chances.append(p ** k)\n    return sum(chances) / len(chances)\n", "correct", None),
    ("pass_hat", _P_MEAN_FIRST, "wrong", "mean-first"),
    ("pass_hat", _P_ONCE, "wrong", "once"),
    ("pass_hat", _P_TIMES, "wrong", "times"),
    ("pass_hat", _P_SUM, "wrong", "sum"),
    ("pass_hat", _P_EARLY, "wrong", "early"),
    ("pass_hat", "def pass_hat(rates, k):\n    return 0.267665796\n", "almost", None),
    ("pass_hat", "def pass_hat(rates, k):\n    return len(AGENT)\n", "error", "lesson-names"),
]
