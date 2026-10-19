# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson ml-digits-linear: cross_entropy_gradients and fit for softmax regression, each saved
in the learner's project (work/projects/nn/) when right; the handwritten digits in the picture; and the demo that
trains the learner's own model on 6,000 digits of QMNIST and measures it on 1,000 others."""
import functools
import re

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, Timeout, run_code, with_time_limit

DATA = "data/qmnist-small.npz"   # the first 6,000 training and 1,000 test images of QMNIST: see data/README.md
K, LR, STEPS = 10, 1.0, 100       # ten digits; the demo trains with a learning rate of 1, for 100 steps


@functools.lru_cache(maxsize=1)
def data():
    """The training images and labels and the test images and labels; every pixel divided by 255, so from 0 to 1."""
    import numpy as np
    d = np.load(paths.ROOT / DATA)
    return d["train_images"] / 255, d["train_labels"], d["test_images"] / 255, d["test_labels"]


def _slopes(X, y, W, b):
    """The course's own slopes of the cross-entropy, so that a wrong module of the learner shows up."""
    import numpy as np
    X, y, W, b = np.array(X, dtype=float), np.array(y), np.array(W, dtype=float), np.array(b, dtype=float)
    Z = X @ W.T + b
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    diff = E / E.sum(axis=1, keepdims=True) - np.eye(len(b))[y]
    return diff.T @ X / len(y), diff.mean(axis=0)


def _fit(X, y, k, lr, steps):
    import numpy as np
    X = np.array(X, dtype=float)
    W, b = np.zeros((k, X.shape[1])), np.zeros(k)
    for _ in range(steps):
        gW, gb = _slopes(X, y, W, b)
        W, b = W - lr * gW, b - lr * gb
    return W, b


# --- the pictures: images of 28 by 28 as squares in a few shades of their role, so they follow the theme ---------

_SHADES = [0.3, 0.55, 0.8, 1.0]


def _image(img, x0: float, y0: float, size: float, signed: bool, top: float) -> str:
    """One image as SVG: every pixel a square of `size`, in one of four shades. Unsigned (0 to 1): the data role, so a
    digit is dark on a light page and light on a dark one. Signed (weights): orange for positive, slate for negative,
    the page for 0, as the style guide's scale with a sign."""
    import numpy as np
    img = np.asarray(img, dtype=float).reshape(28, 28)
    paths_ = {}
    for r in range(28):
        c = 0
        while c < 28:
            v = img[r, c] / top
            level = min(3, int(abs(v) * 4)) if abs(v) >= 0.1 else -1
            key = (level, v > 0) if level >= 0 else None
            run = 1
            while c + run < 28 and key is not None:
                w = img[r, c + run] / top
                nxt = (min(3, int(abs(w) * 4)), w > 0) if abs(w) >= 0.1 else None
                if nxt != key:
                    break
                run += 1
            if key is not None:
                paths_.setdefault(key, []).append(f"M{x0 + c * size:g} {y0 + r * size:g}h{run * size:g}v{size:g}h{-run * size:g}z")
            c += run
    out = [f'<rect x="{x0 - 0.5:g}" y="{y0 - 0.5:g}" width="{28 * size + 1:g}" height="{28 * size + 1:g}" '
           f'style="fill:none;stroke:var(--ea-line);stroke-width:1"/>']
    for (level, positive), parts in sorted(paths_.items()):
        colour = ("var(--ea-param)" if positive else "var(--ea-ink-3)") if signed else "var(--ea-data)"
        out.append(f'<path d="{"".join(parts)}" style="fill:{colour};fill-opacity:{_SHADES[level]}"/>')
    return "".join(out)


def images_picture(images, captions: list[str], title: str, alt: str, signed: bool = False) -> str:
    """Images of 28 by 28 in rows of five, each with its caption below; with `signed`, weights with one scale for all."""
    import html
    import numpy as np
    size, gap, top_y = 3, 18, (30 if title else 4)
    top = float(np.max(np.abs(images))) if signed else 1.0
    box = 28 * size
    parts = []
    if title:
        parts.append(f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
    for i, (img, caption) in enumerate(zip(images, captions)):
        x, y = 1 + (i % 5) * (box + gap), top_y + (i // 5) * (box + 34)
        parts.append(_image(img, x, y, size, signed, top))
        parts.append(f'<text x="{x + box / 2:g}" y="{y + box + 20:g}" style="font-size:13px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(caption)}</text>')
    rows = (len(captions) + 4) // 5
    width, height = 5 * box + 4 * gap + 2, top_y + rows * (box + 34)
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{width}px)">'
            f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>{"".join(parts)}</svg></figure>')


def examples():
    """The first image of each digit in the training set, 0 to 9."""
    import numpy as np
    X, y, _, _ = data()
    return [X[int(np.argmax(y == k))] for k in range(K)]


# --- the demo ---------------------------------------------------------------------------------------------------

def gradients_ok() -> bool | None:
    """None when cross_entropy_gradients is not saved yet; else whether the saved one still gives the right slopes."""
    if not (projects.folder() / "nn" / "softmax_gradients.py").exists():
        return None
    for case in [([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 2, 1], [[0.5, -1.0], [0.0, 0.2], [0.3, 0.3]], [0.1, 0.0, -0.1]),
                 ([[0.5, 1.0, 0.0], [1.0, 0.0, 2.0]], [1, 0], [[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]], [0.0, 0.5])]:
        run = run_code(f"from nn import softmax_gradients\nsoftmax_gradients.cross_entropy_gradients{case!r}")
        if run.error is not None or run.timeout or not _close_pair(run.value, _slopes(*case)):
            return False
    return True


def trained():
    """For the demo: None when fit is not saved yet; "gradients" when its module is broken; False when the saved fit
    no longer trains right, or stops with an error; "slow" when it takes more than a minute on the 6,000 digits; else
    (W, b) as arrays after the learner's own fit on them."""
    if not (projects.folder() / "nn" / "softmax.py").exists():
        return None
    if gradients_ok() is not True:
        return "gradients"
    small = ([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], 3, 0.5, 20)
    run = run_code(f"from nn import softmax\nsoftmax.fit{small!r}")
    if run.error is not None or run.timeout or not _close_pair(run.value, _fit(*small)):
        return False
    X, y, _, _ = data()
    import numpy as np
    try:
        fit = run.names["softmax"].fit
        W, b = with_time_limit(lambda: fit(X, y, K, LR, STEPS), 60)
    except Timeout:
        return "slow"
    except Exception:
        return False
    return np.asarray(W, dtype=float), np.asarray(b, dtype=float)   # also when the learner's fit gives back lists


def predictions(W, b):
    """The digit with the largest logit, for every test image."""
    _, _, X_test, _ = data()
    return (X_test @ W.T + b).argmax(axis=1)


def accuracy(W, b) -> float:
    import numpy as np
    return float(np.mean(predictions(W, b) == data()[3]))


def baseline() -> tuple[int, float]:
    """The most common digit in the training set, and the accuracy on the test set of always predicting it."""
    import numpy as np
    _, y, _, y_test = data()
    digit = int(np.bincount(y).argmax())
    return digit, float(np.mean(y_test == digit))


def confusion(W, b):
    """The confusion matrix on the test set: rows the true digit, columns the digit the model says."""
    import numpy as np
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(data()[3], predictions(W, b)):
        cm[t, p] += 1
    return cm


def most_confused(cm) -> tuple[int, int, int]:
    """(true digit, digit said, how often): the largest count off the diagonal; the first one on a tie."""
    best = (0, 1, -1)
    for i in range(K):
        for j in range(K):
            if i != j and cm[i, j] > best[2]:
                best = (i, j, int(cm[i, j]))
    return best


def confusion_picture(cm, marked: tuple[int, int], labels: dict) -> str:
    """The confusion matrix as a grid of counts, with the row and column numbers; `marked` in bold, the rest grey.
    Under it the caption in code, then each of `labels["lines"]` on a line of its own, so it stays narrow on a phone."""
    from everything_ai import draw
    rows = [[str(int(v)) for v in row] for row in cm]
    return draw.array_grids([(labels["label"], rows, [labels["caption"], *labels["lines"]], {marked})], indices=True,
                            title=labels["title"]).text


# The lesson is done when these are correct. "how-many" is the choice question in the lesson.
REQUIRED = ["how-many", "cross-entropy-gradients", "fit-softmax"]


def _close_pair(got, want) -> bool:
    """(slopes or weights for W, one number per class for b), the same shapes and numbers as the course's."""
    import numpy as np
    try:
        gW, gb = got
        wW, wb = want
        pairs = [(np.asarray(gW, dtype=float), np.asarray(wW, dtype=float)), (np.asarray(gb, dtype=float), np.asarray(wb, dtype=float))]
    except (TypeError, ValueError, OverflowError):
        return False
    return all(g.shape == w.shape and np.allclose(g, w, rtol=1e-6, atol=1e-9, equal_nan=True) for g, w in pairs)


def _plain(v) -> str:
    """A value as the lessons write it: lists and numbers to 4 decimals, never array(...) or np.float64(...)."""
    import numpy as np
    if isinstance(v, tuple):
        return "(" + ", ".join(_plain(x) for x in v) + ")"
    try:
        return str((np.round(np.asarray(v, dtype=float), 4) + 0.0).tolist())
    except (TypeError, ValueError):
        return repr(v)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own lines.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen regels.")

_G_HEAD = ("import numpy as np\n\n\ndef cross_entropy_gradients(X, y, W, b):\n    X = np.array(X)\n    y = np.array(y)\n"
           "    W = np.array(W)\n    b = np.array(b)\n")
_G_START = _G_HEAD + "    ...\n    return grad_W, grad_b\n"
_LOGITS = "    Z = X @ W.T + b\n"
_SOFTMAX = "    E = np.exp(Z - Z.max(axis=1, keepdims=True))\n    P = E / E.sum(axis=1, keepdims=True)\n"
_DIFF = "    diff = P - np.eye(len(b))[y]\n"
_SLOPES = "    grad_W = diff.T @ X / len(y)\n    grad_b = diff.mean(axis=0)\n"
_G = _G_HEAD + _LOGITS + _SOFTMAX + _DIFF + _SLOPES + "    return grad_W, grad_b"


def _g(logits=_LOGITS, softmax=_SOFTMAX, diff=_DIFF, slopes=_SLOPES, end="    return grad_W, grad_b") -> str:
    """The solution with one part changed."""
    return _G_HEAD + logits + softmax + diff + slopes + end


_FIT_HEAD = ("import numpy as np\nfrom nn import softmax_gradients\n\n\ndef fit(X, y, K, lr, steps):\n    X = np.array(X)\n    y = np.array(y)\n"
             "    W = np.zeros((K, X.shape[1]))\n    b = np.zeros(K)\n    for step in range(steps):\n")
_FIT_START = _FIT_HEAD + "        ...\n    return W, b\n"
_FIT_BODY = ("        grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W = W - lr * grad_W\n"
             "        b = b - lr * grad_b\n    return W, b")
_FIT = _FIT_HEAD + _FIT_BODY
# What the check compares with: the course's own slopes, not the learner's module, so a wrong module shows up.
_COURSE_SLOPES = ("def _slopes(X, y, W, b):\n    Z = X @ W.T + b\n    E = np.exp(Z - Z.max(axis=1, keepdims=True))\n"
                  "    diff = E / E.sum(axis=1, keepdims=True) - np.eye(len(b))[y]\n    return diff.T @ X / len(y), diff.mean(axis=0)\n\n\n")
_FIT_COURSE = ("import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1]
               + _FIT_BODY.replace("softmax_gradients.cross_entropy_gradients(X, y, W, b)", "_slopes(X, y, W, b)"))


def _fit_course(body: str) -> str:
    """A wrong loop, with the course's own slopes."""
    return "import numpy as np\n\n\n" + _COURSE_SLOPES + _FIT_HEAD.split("\n\n\n", 1)[1] + body


# A rounded answer would show the same numbers as got and want: say what it is instead.
_ROUNDED = Mistake("rounded", matches=lambda got, text: re.search(r"\bround\s*\(", text) is not None,
                   en="`{call}` gives `{got}`: rounded numbers. The check compares more decimals than it shows, so give the numbers back as they come out, without `round`.",
                   nl="`{call}` geeft `{got}`: afgeronde getallen. De check vergelijkt meer decimalen dan hij laat zien, dus geef de getallen terug zoals ze uit de berekening komen, zonder `round`.")

_REDO = {"en": " Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.",
         "nl": " Heb je dat bestand veranderd? Verwijder het, en laat oefening 2 opnieuw nakijken: die check bewaart je functie dan opnieuw."}

_VISIBLE = ([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], [[0, 0], [0, 0], [0, 0]], [0, 0, 0])

EXERCISES = {
    "cross-entropy-gradients": FunctionBox(
        name="cross-entropy-gradients",
        function="cross_entropy_gradients",
        start=_G_START,
        solution=_G,
        visible=[_VISIBLE],
        hidden=[([[0.5, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 1.0, 1.0]], [1, 0, 2], [[0.2, -0.1, 0.4], [0.0, 0.3, -0.2], [-0.5, 0.1, 0.1]], [0.1, -0.2, 0.0]),
                ([[1.0, 2.0]], [1], [[0.5, -0.5], [1.0, 0.0]], [0.0, 0.5]),
                ([[1, 0], [0, 1]], [0, 1], [[1000, 0], [0, 1000], [500, 500]], [0, 0, 0]),
                ([[0.2, 0.9], [0.5, 0.4], [0.8, 0.1], [0.3, 0.7], [0.6, 0.6]], [3, 0, 1, 3, 2],
                 [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6], [0.0, 0.7]], [0.2, 0.0, -0.1, 0.3])],
        compare=_close_pair,
        show=_plain,
        save_as=("nn", "softmax_gradients"),
        first=[_DOTS],
        mistakes=[
            Mistake("x-at-w", syntax=r"\bX\s*@\s*W\b(?!\s*\.T)",
                    en="Python stops: `X @ W` puts (n, d) before (K, d), and the inner numbers differ. `W` has one row per class; each logit is the dot product of an example with a row of `W`: `X @ W.T`.",
                    nl="Python stopt: `X @ W` zet (n, d) voor (K, d), en de binnenste getallen verschillen. `W` heeft één rij per klasse; elke logit is het inproduct van een voorbeeld met een rij van `W`: `X @ W.T`."),
            Mistake("recipe-names", error=lambda e: isinstance(e, NameError) and re.search(r"name '(K|n)' is not", str(e)) is not None,
                    en="With `{call}`, Python does not know `K` or `n`: in the recipe they are the number of classes and the number of examples. In your function, those are `len(b)` and `len(y)`.",
                    nl="Met `{call}` kent Python `K` of `n` niet: in het recept zijn dat het aantal klassen en het aantal voorbeelden. In je functie zijn dat `len(b)` en `len(y)`."),
            Mistake("no-keepdims", syntax=r"\b(max|sum)\((\s*\w+\s*,)?\s*(axis\s*=\s*)?1\s*\)",
                    en="Python stops: with `axis=1` the largest logit or the sum of each row has shape (n,), and `Z` has shape (n, K): they do not fit. Keep one number per row as a column: `keepdims=True`.",
                    nl="Python stopt: met `axis=1` heeft de grootste logit of de som van elke rij shape (n,), en `Z` heeft shape (n, K): die passen niet. Houd één getal per rij als kolom: `keepdims=True`."),
            Mistake("no-one-hot", syntax=r"\bP\s*-\s*y\b",
                    en="Python stops: `P - y` takes the label, one number, off a whole row of K probabilities, and the shapes (n, K) and (n,) do not fit. Make the labels one-hot first: `np.eye(len(b))[y]`, one row per example.",
                    nl="Python stopt: `P - y` haalt het label, één getal, van een hele rij van K kansen af, en de shapes (n, K) en (n,) passen niet. Maak de labels eerst one-hot: `np.eye(len(b))[y]`, één rij per voorbeeld."),
            Mistake("eye-size", syntax=r"np\.eye\(\s*(\d+|len\(\s*[yX]\s*\)|[yX]\.shape\[0\])\s*\)",
                    en="Python stops on the line `{line}`: the one-hot labels need one column per class, as many as `P` has. The number of classes is `len(b)`: `np.eye(len(b))[y]`.",
                    nl="Python stopt op de regel `{line}`: de one-hot labels hebben één kolom per klasse nodig, evenveel als `P` heeft. Het aantal klassen is `len(b)`: `np.eye(len(b))[y]`."),
            Mistake("diff-at-x", syntax=r"\bdiff\s*@\s*X\b",
                    en="Python stops: `diff @ X` puts (n, K) before (n, d), and the inner numbers differ. Turn `diff` first: `diff.T @ X` has shape (K, d), the shape of `W`.",
                    nl="Python stopt: `diff @ X` zet (n, K) voor (n, d), en de binnenste getallen verschillen. Draai `diff` eerst: `diff.T @ X` heeft shape (K, d), de shape van `W`."),
            Mistake("one-value", matches=lambda got, text: not isinstance(got, (tuple, list)),
                    en="`{call}` gives `{got}`: one thing, and the check expects two, the slopes for `W` and the slopes for `b`. Give both back: `return grad_W, grad_b`.",
                    nl="`{call}` geeft `{got}`: één ding, en de check verwacht er twee, de hellingen voor `W` en de hellingen voor `b`. Geef ze allebei terug: `return grad_W, grad_b`."),
            _ROUNDED,
            Mistake("x-t-at-diff", wrong=_g(slopes="    grad_W = X.T @ diff / len(y)\n    grad_b = diff.mean(axis=0)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your slopes for `W` have one row per feature, the transpose of `W`, which has one row per class. Turn the other one: `diff.T @ X`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hellingen voor `W` hebben één rij per feature, zoals de getransponeerde van `W`; `W` zelf heeft één rij per klasse. Draai de andere: `diff.T @ X`."),
            Mistake("columns", wrong=_g(softmax="    E = np.exp(Z - Z.max(axis=0, keepdims=True))\n    P = E / E.sum(axis=0, keepdims=True)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your probabilities add up to 1 in each column, over the examples. Softmax goes over the classes of one example: along each row, `axis=1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kansen tellen per kolom op tot 1, over de voorbeelden. Softmax gaat over de klassen van één voorbeeld: langs elke rij, `axis=1`."),
            Mistake("no-max", wrong=_g(softmax="    E = np.exp(Z)\n    P = E / E.sum(axis=1, keepdims=True)\n"),
                    en="Right for the example. But `{call}` gives `{got}`, and it should give `{want}`: with logits of 1000, `np.exp` is too big for a float, and `inf` divided by `inf` is `nan`. Take the largest logit of each row off first.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: bij logits van 1000 is `np.exp` te groot voor een float, en `inf` gedeeld door `inf` is `nan`. Haal eerst de grootste logit van elke rij eraf.",
                    state="almost"),
            Mistake("sign", wrong=_g(diff="    diff = np.eye(len(b))[y] - P\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the right size, the wrong sign. The slopes for the logits are the probabilities minus the one-hot labels: `P - np.eye(len(b))[y]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de goede grootte, het verkeerde teken. De hellingen voor de logits zijn de kansen min de one-hot labels: `P - np.eye(len(b))[y]`."),
            Mistake("no-softmax", wrong=_g(softmax="", diff="    diff = Z - np.eye(len(b))[y]\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the logits minus the one-hot labels, not the probabilities. Put each row of logits through softmax first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de logits min de one-hot labels, niet de kansen. Haal elke rij logits eerst door softmax."),
            Mistake("sum", wrong=_g(slopes="    grad_W = diff.T @ X\n    grad_b = diff.sum(axis=0)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you add up the slopes of the examples. The cross-entropy is a mean over the examples, so its slopes are means too: divide by `len(y)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je telt de hellingen van de voorbeelden op. De cross-entropy is een gemiddelde over de voorbeelden, dus zijn hellingen zijn dat ook: deel door `len(y)`."),
            Mistake("b-rows", wrong=_g(slopes="    grad_W = diff.T @ X / len(y)\n    grad_b = diff.mean(axis=1)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `grad_b` has one number per example. `b` has one bias term per class, so take the mean down each column: `axis=0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `grad_b` heeft één getal per voorbeeld. `b` heeft één bias term per klasse, dus neem het gemiddelde langs elke kolom: `axis=0`."),
            Mistake("b-one", wrong=_g(slopes="    grad_W = diff.T @ X / len(y)\n    grad_b = diff.mean()\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `grad_b` is one number, the mean of all differences. `b` has one bias term per class, so take the mean down each column: `axis=0`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `grad_b` is één getal, het gemiddelde van alle verschillen. `b` heeft één bias term per klasse, dus neem het gemiddelde langs elke kolom: `axis=0`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with steps 1 to 4 of the recipe: the logits `Z`, the probabilities `P`, the differences `diff`, and the slopes `grad_W` and `grad_b`.",
                      "nl": "Vervang de `...` door stap 1 tot en met 4 van het recept: de logits `Z`, de kansen `P`, de verschillen `diff`, en de hellingen `grad_W` en `grad_b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with four values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, W, b)`.",
                          "nl": "De check roept `{call}` aan, met vier waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, W, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you follow steps 1 to 4 of the recipe: the logits, softmax along each row, the probabilities minus the one-hot labels, and then the slopes?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Volg je stap 1 tot en met 4 van het recept: de logits, softmax langs elke rij, de kansen min de one-hot labels, en dan de hellingen?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `X`, `y`, `W` and `b` themselves, for any number of examples, features and classes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `X`, `y`, `W` en `b` zelf, voor elk aantal voorbeelden, features en klassen?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: exercise 3 imports it from there.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: oefening 3 haalt hem daar binnen."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
    "fit-softmax": FunctionBox(
        name="fit-softmax",
        function="fit",
        start=_FIT_START,
        solution=_FIT,
        visible=[([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], 3, 0.5, 100)],
        hidden=[([[2.0]], [0], 2, 0.1, 3), ([[1], [2], [3], [4]], [0, 0, 1, 1], 2, 0.2, 200),
                ([[0.5, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1, 2, 0, 1], 3, 0.1, 300), ([[1, 2], [2, 1]], [1, 0], 2, 0.05, 0),
                ([[1, 0], [0, 1]], [0, 3], 4, 0.5, 20), ([[1, 0], [0, 1], [1, 1], [0, 0]], [0, 2, 2, 1], 3, 0.2, 50)],
        compare=_close_pair,
        show=_plain,
        save_as=("nn", "softmax"),
        first=[
            _DOTS,
            Mistake("not-saved-yet", state="empty", matches=lambda got, text: gradients_ok() is None,
                    en="Your `cross_entropy_gradients` is not in your project yet. The check of exercise 2 saves it, as `work/projects/nn/softmax_gradients.py`, once it is right. Finish exercise 2 first; this check then runs again by itself.",
                    nl="Je `cross_entropy_gradients` staat nog niet in je project. De check van oefening 2 bewaart hem, als `work/projects/nn/softmax_gradients.py`, zodra hij klopt. Maak eerst oefening 2 af; deze check draait daarna vanzelf opnieuw."),
            Mistake("bad-gradients", matches=lambda got, text: gradients_ok() is False,
                    en="Your loop may be right, but the `cross_entropy_gradients` in `work/projects/nn/softmax_gradients.py` no longer gives the right slopes, or stops with an error." + _REDO["en"],
                    nl="Je loop kan goed zijn, maar de `cross_entropy_gradients` in `work/projects/nn/softmax_gradients.py` geeft niet meer de goede hellingen, of stopt met een fout." + _REDO["nl"]),
        ],
        form=[Mistake("own-module", state="almost", matches=lambda got, text: re.search(r"\bcross_entropy_gradients\s*\(", text) is None,
                      en="Right, but you worked out the slopes here again. The point is to use your own function from your project: `softmax_gradients.cross_entropy_gradients(X, y, W, b)`.",
                      nl="Goed, maar je rekent de hellingen hier opnieuw uit. Het gaat erom dat je je eigen functie uit je project gebruikt: `softmax_gradients.cross_entropy_gradients(X, y, W, b)`.")],
        mistakes=[
            Mistake("bare-name", error=lambda e: isinstance(e, NameError) and "cross_entropy_gradients" in str(e),
                    en="With `{call}`, Python does not know `cross_entropy_gradients` on its own: it lives in the module you imported. Call it through the module: `softmax_gradients.cross_entropy_gradients(X, y, W, b)`.",
                    nl="Met `{call}` kent Python `cross_entropy_gradients` niet los: hij woont in de module die je binnenhaalde. Roep hem aan via de module: `softmax_gradients.cross_entropy_gradients(X, y, W, b)`."),
            _ROUNDED,
            Mistake("uphill", wrong=_fit_course("        grad_W, grad_b = _slopes(X, y, W, b)\n        W = W + lr * grad_W\n        b = b + lr * grad_b\n    return W, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your steps go uphill, so the loss grows. An update step goes against the slope: minus.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stappen gaan bergop, dus de loss groeit. Een update-stap gaat tegen de helling in: min."),
            Mistake("no-lr", wrong=_fit_course("        grad_W, grad_b = _slopes(X, y, W, b)\n        W = W - grad_W\n        b = b - grad_b\n    return W, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: without the learning rate, every step is the whole slope. Multiply each slope by `lr`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: zonder learning rate is elke stap de hele helling. Vermenigvuldig elke helling met `lr`."),
            Mistake("only-w", wrong=_fit_course("        grad_W, grad_b = _slopes(X, y, W, b)\n        W = W - lr * grad_W\n    return W, b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `b` stays all 0. Update `b` too, with its own slopes: `b = b - lr * grad_b`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `b` blijft helemaal 0. Doe ook een update-stap voor `b`, met zijn eigen hellingen: `b = b - lr * grad_b`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one update step: the slopes from your `softmax_gradients.cross_entropy_gradients`, and then the new `W` and `b`.",
                      "nl": "Vervang de `...` door één update-stap: de hellingen uit je `softmax_gradients.cross_entropy_gradients`, en dan de nieuwe `W` en `b`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with five values, and the `def` line of your function has a different number of names between its parentheses. Keep `(X, y, K, lr, steps)`.",
                          "nl": "De check roept `{call}` aan, met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(X, y, K, lr, steps)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does each round get both slopes from `softmax_gradients.cross_entropy_gradients`, and move `W` and `b` against them, by `lr` times each slope?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Haalt elke ronde beide hellingen uit `softmax_gradients.cross_entropy_gradients`, en schuift hij `W` en `b` ertegenin, met `lr` keer elke helling?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you use `lr` and `steps` themselves?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruik je `lr` en `steps` zelf?"},
            "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`. The demo below trains it on 6,000 handwritten digits.",
                        "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`. De demo hieronder traint hem op 6000 handgeschreven cijfers."},
            "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
# The tests check every solution first, so cross_entropy_gradients is saved before the fit cases run.
CASES = [
    ("cross-entropy-gradients", _G_START, "empty", None),
    ("cross-entropy-gradients", _G, "correct", None),
    ("cross-entropy-gradients", _g(logits="    Z = X @ W.T + b\n    Z = Z - Z.max(axis=1, keepdims=True)\n", softmax="    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)\n"), "correct", None),
    ("cross-entropy-gradients", _g(diff="    diff = P - np.eye(W.shape[0])[y]\n", slopes="    grad_W = (X.T @ diff).T / len(y)\n    grad_b = np.mean(diff, axis=0)\n"), "correct", None),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z - Z.max())\n    P = E / E.sum(axis=1, keepdims=True)\n"), "correct", None),
    ("cross-entropy-gradients", _g(logits="    Z = X @ W + b\n"), "error", "x-at-w"),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z - Z.max(axis=1))\n    P = E / E.sum(axis=1)\n"), "error", "no-keepdims"),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z - Z.max(axis=1, keepdims=True))\n    P = E / np.sum(E, axis=1)\n"), "error", "no-keepdims"),
    ("cross-entropy-gradients", _g(diff="    diff = P - y\n"), "error", "no-one-hot"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff @ X / len(y)\n    grad_b = diff.mean(axis=0)\n"), "error", "diff-at-x"),
    ("cross-entropy-gradients", _g(end="    return grad_W"), "wrong", "one-value"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = X.T @ diff / len(y)\n    grad_b = diff.mean(axis=0)\n"), "wrong", "x-t-at-diff"),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z - Z.max(axis=0, keepdims=True))\n    P = E / E.sum(axis=0, keepdims=True)\n"), "wrong", "columns"),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z)\n    P = E / E.sum(axis=1, keepdims=True)\n"), "almost", "no-max"),
    ("cross-entropy-gradients", _g(diff="    diff = np.eye(len(b))[y] - P\n"), "wrong", "sign"),
    ("cross-entropy-gradients", _g(softmax="", diff="    diff = Z - np.eye(len(b))[y]\n"), "wrong", "no-softmax"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff.T @ X\n    grad_b = diff.sum(axis=0)\n"), "wrong", "sum"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff.T @ X / len(y)\n    grad_b = diff.mean(axis=1)\n"), "wrong", "b-rows"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff.T @ X / len(y)\n    grad_b = diff.mean()\n"), "wrong", "b-one"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff.T @ X / 4\n    grad_b = diff.sum(axis=0) / 4\n"), "almost", None),
    ("cross-entropy-gradients", _g(diff="    diff = P - np.eye(K)[y]\n"), "error", "recipe-names"),
    ("cross-entropy-gradients", _g(slopes="    grad_W = diff.T @ X / n\n    grad_b = diff.mean(axis=0)\n"), "error", "recipe-names"),
    ("cross-entropy-gradients", _g(softmax="    E = np.exp(Z - Z.max(1))\n    P = E / E.sum(1)\n"), "error", "no-keepdims"),
    ("cross-entropy-gradients", _g(diff="    diff = P - np.eye(len(y))[y]\n"), "error", "eye-size"),
    ("cross-entropy-gradients", _g(diff="    diff = P - np.eye(3)[y]\n"), "error", "eye-size"),
    ("cross-entropy-gradients", _g(end="    return np.round(grad_W, 4), np.round(grad_b, 4)"), "wrong", "rounded"),
    ("fit-softmax", _FIT_START, "empty", None),
    ("fit-softmax", _FIT, "correct", None),
    ("fit-softmax", _FIT_HEAD + "        gW, gb = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W -= lr * gW\n        b -= lr * gb\n    return W, b", "correct", None),
    ("fit-softmax", _FIT_HEAD + "        grad_W, grad_b = cross_entropy_gradients(X, y, W, b)\n        W = W - lr * grad_W\n        b = b - lr * grad_b\n    return W, b", "error", "bare-name"),
    ("fit-softmax", _FIT_HEAD + "        grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W = W + lr * grad_W\n        b = b + lr * grad_b\n    return W, b", "wrong", "uphill"),
    ("fit-softmax", _FIT_HEAD + "        grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W = W - grad_W\n        b = b - grad_b\n    return W, b", "wrong", "no-lr"),
    ("fit-softmax", _FIT_HEAD + "        grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W = W - lr * grad_W\n    return W, b", "wrong", "only-w"),
    ("fit-softmax", _FIT_COURSE, "almost", "own-module"),
    ("fit-softmax", _FIT_HEAD + _FIT_BODY.replace("    return W, b", "    return np.round(W, 4), np.round(b, 4)"), "wrong", "rounded"),
    ("fit-softmax", _FIT_HEAD + "        grad_W, grad_b = softmax_gradients.cross_entropy_gradients(X, y, W, b)\n        W = W - 0.5 * grad_W\n        b = b - 0.5 * grad_b\n    return W, b", "almost", None),
]
