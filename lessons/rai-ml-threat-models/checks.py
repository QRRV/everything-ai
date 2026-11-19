# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-ml-threat-models: an evasion attack on a small model the course trains itself, a logistic
regression that tells a stop sign from a speed limit sign in made-up pictures of 8 by 8 pixels. The three things the
learner writes: fgsm (the fast gradient sign method, white box), smallest_eps (how far every pixel has to move) and
guess_signs (the same direction found with questions only, black box); with the pictures of what they give, and the
demo that compares a step along the signs of the weights with a random step of the same size."""
import copy
import html
import re

import numpy as np

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, with_time_limit


class _Shown(np.ndarray):
    """A picture or a vector of weights that a check message writes as the lesson names it, such as STOP. The
    learner's code gets a plain copy."""

    def __repr__(self):
        name = getattr(self, "shown", None)
        return name if name else repr(np.asarray(self))

    def __array_finalize__(self, obj):
        self.shown = None          # a slice of W is not W

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        out = np.asarray(obj)      # what NumPy computes from W, such as 0.1 * W or np.sum(W), is a plain array
        return out[()] if return_scalar else out

    def __deepcopy__(self, memo):
        return np.array(self)


class _Number(float):
    """A number that a check message writes by its name in the lesson, such as B. The learner's code gets a plain
    float."""

    def __new__(cls, value, name: str):
        n = super().__new__(cls, value)
        n.shown = name
        return n

    def __repr__(self):
        return self.shown

    def __deepcopy__(self, memo):
        return float(self)


def _named(values, name: str) -> np.ndarray:
    v = np.array(values, dtype=float).view(_Shown)
    v.shown = name
    return v


# --- the made-up pictures: two signs on 8 by 8 pixels, each pixel from 0 to 1 ------------------------------------------

_OCTAGON = ["..XXXX..", ".XXXXXX.", "XXXXXXXX", "XXXXXXXX", "XXXXXXXX", "XXXXXXXX", ".XXXXXX.", "..XXXX.."]
_INSIDE = ["........", "..XXXX..", ".XXXXXX.", ".XXXXXX.", ".XXXXXX.", ".XXXXXX.", "..XXXX..", "........"]


def _mask(rows: list) -> np.ndarray:
    return np.array([[c == "X" for c in r] for r in rows], dtype=float)


_band = np.zeros((8, 8))
_band[3:5, 1:7] = 1                       # the letters across a stop sign, lower than the rest
_digits = np.zeros((8, 8))                # the figures inside a speed limit sign
_digits[2:6, 2] = 1
_digits[2:6, 4] = 1
_digits[2, 5] = _digits[5, 5] = _digits[3, 4] = 1
STOP_SHAPE = 0.1 + 0.7 * _mask(_OCTAGON) - 0.4 * _band
SPEED_SHAPE = 0.1 + 0.7 * (_mask(_OCTAGON) - _mask(_INSIDE)) + 0.5 * _digits * _mask(_INSIDE)


def _make(rng, shape: np.ndarray, n: int) -> np.ndarray:
    """n pictures of one sign: a little stronger or weaker each time, with noise, between 0.02 and 0.98."""
    out = [np.clip(shape * rng.uniform(0.85, 1.1) + rng.normal(0, 0.06, (8, 8)), 0.02, 0.98).ravel() for _ in range(n)]
    return np.array(out)


def _train(X: np.ndarray, y: np.ndarray, steps: int):
    """Logistic regression with gradient descent on the log loss, with a small L2 penalty on the weights."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w = w - 0.5 * ((p - y) @ X / len(y) + 0.01 * w)
        b = b - 0.5 * float(np.mean(p - y))
    return w, b


_rng = np.random.default_rng(7)
_X_TRAIN = np.vstack([_make(_rng, STOP_SHAPE, 100), _make(_rng, SPEED_SHAPE, 100)])
_Y_TRAIN = np.r_[np.zeros(100), np.ones(100)]
_X_TEST = np.vstack([_make(_rng, STOP_SHAPE, 100), _make(_rng, SPEED_SHAPE, 100)])
_Y_TEST = np.r_[np.zeros(100), np.ones(100)]
_w, _b = _train(_X_TRAIN, _Y_TRAIN, 400)
W = _named(_w, "W")                       # the 64 weights of the model: class 1 is a speed limit sign, class 0 a stop sign
B = _Number(_b, "B")                      # its bias term
_w2, _b2 = _train(_X_TRAIN[::5], _Y_TRAIN[::5], 60)
W2 = _named(_w2, "W2")                    # a second model, trained on fewer pictures for fewer steps
B2 = _Number(_b2, "B2")
STOP = _named(_X_TEST[0], "STOP")         # a stop sign the model has not seen
STOP_2 = _named(_X_TEST[1], "STOP_2")     # another one
TEST_ACCURACY = float(np.mean((_X_TEST @ _w + _b > 0) == _Y_TEST))
RANDOM_SIGNS = np.where(np.random.default_rng(3).random(64) < 0.5, -1.0, 1.0)   # up or down at random, for comparison


def logit(image, w=None, b=None) -> float:
    w, b = (W, B) if w is None else (w, b)
    return float(np.dot(np.asarray(w), np.asarray(image, dtype=float)) + float(b))


def predict(image, w=None, b=None) -> float:
    """The probability that the model gives to a speed limit sign."""
    return float(1 / (1 + np.exp(-logit(image, w, b))))


def fgsm(image, w, eps):
    return image + eps * np.sign(w)


def smallest_eps(image, w, b):
    z = np.dot(w, image) + b
    return -z / np.sum(abs(w))


def guess_signs(ask, image, delta):
    base = ask(image)
    signs = []
    for i in range(len(image)):
        bumped = np.array(image)
        bumped[i] = bumped[i] + delta
        if ask(bumped) > base:
            signs.append(1)
        else:
            signs.append(-1)
    return np.array(signs)


class Ask:
    """The model as a black box: it takes a picture of 64 pixels and gives back the probability of a speed limit
    sign, nothing else. It counts how often it is asked. A check message writes it as ask; the learner's code gets a
    fresh one."""

    def __init__(self, w, b, name: str = "ask"):
        self._w, self._b, self.name, self.count = np.array(w), b, name, 0

    def __call__(self, picture):
        self.count += 1
        return predict(np.asarray(picture, dtype=float), self._w, self._b)

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Ask(self._w, self._b, self.name)


ASK = Ask(W, B, "ask")
ASK_2 = Ask(W2, B2, "ask_2")
EPS_STOP = float(smallest_eps(STOP, W, B))      # how far every pixel of STOP has to move: about 0.21
QUESTIONS = 1 + len(STOP)                       # what guess_signs asks: once for the picture, once per pixel


# --- pictures -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.78rem;color:var(--ea-ink-2)"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem;color:var(--ea-ink)"


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _frame(wrong: bool) -> str:
    return "outline:1.5px dashed var(--ea-loss);outline-offset:2px" if wrong else ""


_STOPPED = object()   # what a call gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _as_picture(v):
    """v as 64 numbers, or None when it is not a picture."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    return a.ravel() if a.size == 64 and np.all(np.isfinite(a)) else None


def _pixels(image, label: str, wrong: bool = False) -> str:
    """A picture of 8 by 8 pixels: the higher the pixel, the more its cell stands out from the page; past 0 or 1 drawn
    as 0 or 1."""
    cells = []
    for r in range(8):
        row = "".join(f'<td style="width:0.95rem;height:0.95rem;padding:0;background:color-mix(in srgb, var(--ea-data) '
                      f'{round(100 * min(1.0, max(0.0, float(image[8 * r + c]))))}%, var(--ea-page))"></td>'
                      for c in range(8))
        cells.append(f"<tr>{row}</tr>")
    table = (f'<table style="border-collapse:collapse;border:1px solid var(--ea-line);{_frame(wrong)}">'
             f'{"".join(cells)}</table>')
    return (f'<div style="display:inline-block;vertical-align:top;margin:0 0.9rem 0.4rem 0">{table}'
            f'<div style="{_LABEL};max-width:9rem;margin-top:0.2rem">{label}</div></div>')


def _signs(signs, label: str, other=None) -> str:
    """8 by 8 cells with + and −: the sign of each weight, in the color of what a model learns. With other, a cell
    whose sign differs from it gets a dashed purple frame."""
    cells = []
    for r in range(8):
        row = ""
        for c in range(8):
            s = float(signs[8 * r + c])
            mark = "+" if s > 0 else ("−" if s < 0 else "0")
            off = other is not None and np.sign(s) != np.sign(float(other[8 * r + c]))
            frame = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px;" if off else ""
            row += (f'<td style="width:0.95rem;height:0.95rem;padding:0;text-align:center;font-size:0.7rem;line-height:1;'
                    f'{frame}color:var(--ea-param-text);font-family:var(--ea-font-code)">{mark}</td>')
        cells.append(f"<tr>{row}</tr>")
    table = f'<table style="border-collapse:collapse;border:1px solid var(--ea-line)">{"".join(cells)}</table>'
    return (f'<div style="display:inline-block;vertical-align:top;margin:0 0.9rem 0.4rem 0">{table}'
            f'<div style="{_LABEL};max-width:9rem;margin-top:0.2rem">{label}</div></div>')


_WORDS = {
    "en": {"stop": "stop sign", "speed": "speed limit sign", "says": "P(speed limit) = {p}: the model says {what}",
           "signs": "the sign of each weight", "plus": "plus {eps} × the signs"},
    "nl": {"stop": "stopbord", "speed": "snelheidsbord", "says": "P(snelheidsbord) = {p}: het model zegt {what}",
           "signs": "het teken van elke weight", "plus": "plus {eps} × de tekens"},
}


def _p_text(p: float, lang: str) -> str:
    text = f"{p:.3f}"
    return text.replace(".", ",") if lang == "nl" else text


def _says(image, lang: str, w=None, b=None) -> str:
    """What the model says about a picture; a speed limit sign in purple, as that is the model's mistake here."""
    t = _WORDS[lang]
    p = predict(image, w, b)
    what = t["speed"] if p > 0.5 else t["stop"]
    color = "color:var(--ea-loss-text);font-weight:600" if p > 0.5 else ""
    return f'<span style="{color}">{html.escape(t["says"].format(p=_p_text(p, lang), what=what))}</span>'


def _eps_text(eps: float, lang: str) -> str:
    text = f"{eps:.2f}"
    return text.replace(".", ",") if lang == "nl" else text


_IDEA_WORDS = {
    "en": {"title": "A stop sign, the signs of the weights, and the two together",
           "said": "Left, a stop sign the model has not seen. Middle, the sign of each of the model's 64 weights: + where "
                   "a higher pixel pushes towards a speed limit sign. Right, every pixel moved by {eps} in the direction "
                   "of its sign. The sign still looks like a stop sign; the model now reads a speed limit sign."},
    "nl": {"title": "Een stopbord, de tekens van de weights, en die twee samen",
           "said": "Links een stopbord dat het model niet eerder zag. In het midden het teken van elk van de 64 "
                   "weights van het model: + waar een hogere pixel richting snelheidsbord duwt. Rechts is elke "
                   "pixel {eps} opgeschoven in de richting van het teken van die pixel. Het bord lijkt nog steeds een "
                   "stopbord; het model leest nu een snelheidsbord."},
}

IDEA_EPS = 0.22


def idea_picture(lang: str) -> str:
    w, t = _IDEA_WORDS[lang], _WORDS[lang]
    after = fgsm(np.array(STOP), np.array(W), IDEA_EPS)
    body = (_pixels(STOP, _says(STOP, lang)) + _signs(np.sign(W), html.escape(t["signs"]))
            + _pixels(after, html.escape(t["plus"].format(eps=_eps_text(IDEA_EPS, lang))) + "<br>" + _says(after, lang)))
    return yours.figure(w["title"], f"<div>{body}</div>" + _note([w["said"].format(eps=_eps_text(IDEA_EPS, lang))]))


# --- the picture under exercise 1: fgsm --------------------------------------------------------------------------------

_FGSM_EPS = (0.1, 0.25)
_FGSM_WORDS = {
    "en": {"title": "Your fgsm on the stop sign STOP",
           "said": "What your fgsm makes of STOP with eps = 0.1 and eps = 0.25, and what the model says about each.",
           "fine": "The same pictures as the course's fgsm.",
           "wrong": "A dashed purple frame marks a picture that differs from what the course's fgsm makes.",
           "stops": "Your fgsm stops with an error on fgsm(STOP, W, {eps}): the check above says where.",
           "off": "Your fgsm does not give 64 pixels for fgsm(STOP, W, {eps}), so there is no picture to draw."},
    "nl": {"title": "Jouw fgsm op het stopbord STOP",
           "said": "Wat jouw fgsm van STOP maakt met eps = 0,1 en eps = 0,25, en wat het model over elk plaatje zegt.",
           "fine": "Dezelfde plaatjes als bij de fgsm van de course.",
           "wrong": "Een gestreept paars kader wijst een plaatje aan dat verschilt van wat de fgsm van de course maakt.",
           "stops": "Jouw fgsm stopt met een fout bij fgsm(STOP, W, {eps}): de check hierboven zegt waar.",
           "off": "Jouw fgsm geeft geen 64 pixels voor fgsm(STOP, W, {eps}), dus er is geen plaatje om te tekenen."},
}


def _short(v) -> str:
    try:
        text = repr(v)
    except Exception:
        text = "?"
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= 50 else text[:47] + "..."


def _fn(seen, name: str):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(name)
    return fn if callable(fn) else None


def fgsm_picture(seen) -> str:
    fn = _fn(seen, "fgsm")
    if fn is None:
        return ""
    w = _FGSM_WORDS[seen.lang]
    parts, wrong_any = [_pixels(STOP, "STOP<br>" + _says(STOP, seen.lang))], False
    for eps in _FGSM_EPS:
        shown = f"{eps:g}".replace(".", ",") if seen.lang == "nl" else f"{eps:g}"
        got = _call(fn, STOP, W, eps)
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(eps=eps)]))
        picture = _as_picture(got)
        if picture is None:
            return yours.figure(w["title"], _note([w["off"].format(eps=eps)]))
        wrong = not _same_picture(picture, fgsm(np.array(STOP), np.array(W), eps))
        wrong_any |= wrong
        parts.append(_pixels(picture, f"eps = {shown}<br>" + _says(picture, seen.lang), wrong))
    return yours.figure(w["title"], f"<div>{''.join(parts)}</div>"
                        + _note([w["said"], w["wrong"] if wrong_any else w["fine"]]))


# --- the picture under exercise 2: smallest_eps ------------------------------------------------------------------------

_EPS_WORDS = {
    "en": {"title": "Your smallest_eps for STOP, and just below and above it",
           "said": "Your smallest_eps(STOP, W, B) gives {eps}. Left, fgsm with 0.01 less; right, with 0.01 more.",
           "fine": "The same as the course's smallest_eps: below it the model still reads a stop sign, above it a "
                   "speed limit sign.",
           "wrong": "The course's smallest_eps gives {want}: at your eps, the model's answer does not turn over between "
                    "left and right.",
           "stops": "Your smallest_eps stops with an error on smallest_eps(STOP, W, B): the check above says where.",
           "off": "Your smallest_eps gives {got} for smallest_eps(STOP, W, B); the picture needs a number above 0.01.",
           "none": "Your smallest_eps gives {got} for smallest_eps(STOP, W, B); the picture needs one number."},
    "nl": {"title": "Jouw smallest_eps voor STOP, en net eronder en erboven",
           "said": "Jouw smallest_eps(STOP, W, B) geeft {eps}. Links fgsm met 0,01 minder; rechts met 0,01 meer.",
           "fine": "Hetzelfde als de smallest_eps van de course: eronder leest het model nog een stopbord, erboven een "
                   "snelheidsbord.",
           "wrong": "De smallest_eps van de course geeft {want}: bij jouw eps slaat het antwoord van het model tussen links "
                    "en rechts niet om.",
           "stops": "Jouw smallest_eps stopt met een fout bij smallest_eps(STOP, W, B): de check hierboven zegt waar.",
           "off": "Jouw smallest_eps geeft {got} voor smallest_eps(STOP, W, B); het plaatje heeft een getal boven 0,01 nodig.",
           "none": "Jouw smallest_eps geeft {got} voor smallest_eps(STOP, W, B); het plaatje heeft één getal nodig."},
}


def _number(v):
    try:
        x = float(np.asarray(v, dtype=float))
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _three(x: float, lang: str) -> str:
    text = f"{x:.3f}"
    return text.replace(".", ",") if lang == "nl" else text


def eps_picture(seen) -> str:
    fn = _fn(seen, "smallest_eps")
    if fn is None:
        return ""
    w = _EPS_WORDS[seen.lang]
    got = _call(fn, STOP, W, B)
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    eps = _number(got)
    if eps is None:
        return yours.figure(w["title"], _note([w["none"].format(got=_short(got))]))
    if eps <= 0.01:
        return yours.figure(w["title"], _note([w["off"].format(got=_three(eps, seen.lang).replace("-", "−"))]))
    pics = []
    for e in (eps - 0.01, eps + 0.01):
        after = fgsm(np.array(STOP), np.array(W), e)
        pics.append(_pixels(after, f"eps = {_three(e, seen.lang)}<br>" + _says(after, seen.lang)))
    below, above = (predict(fgsm(np.array(STOP), np.array(W), e)) for e in (eps - 0.01, eps + 0.01))
    right = below < 0.5 < above
    said = w["fine"] if right else w["wrong"].format(want=_three(EPS_STOP, seen.lang))
    return yours.figure(w["title"], f"<div>{''.join(pics)}</div>"
                        + _note([w["said"].format(eps=_three(eps, seen.lang)), said]))


# --- the picture under exercise 3: guess_signs -------------------------------------------------------------------------

_SIGN_WORDS = {
    "en": {"title": "Your guess_signs, with only the answers of ask",
           "said": "Left, the signs your guess_signs(ask, STOP, 0.01) found; right, the signs of the model's weights, "
                   "which an attacker with a black box never sees. ask was asked {q} times.",
           "fine": "All 64 signs are right: the same direction as in Exercise 1, without the weights.",
           "wrong": "{n} of the 64 signs are right; a dashed purple frame marks each one that is not.",
           "found": "found with questions", "true": "the signs of the weights",
           "stops": "Your guess_signs stops with an error on guess_signs(ask, STOP, 0.01): the check above says where.",
           "off": "Your guess_signs does not give 64 signs for guess_signs(ask, STOP, 0.01), so there is no picture to draw."},
    "nl": {"title": "Jouw guess_signs, met alleen de antwoorden van ask",
           "said": "Links de tekens die jouw guess_signs(ask, STOP, 0.01) vond; rechts de tekens van de weights van het "
                   "model, die een aanvaller met een black box nooit ziet. ask kreeg {q} vragen.",
           "fine": "Alle 64 tekens kloppen: dezelfde richting als in oefening 1, zonder de weights.",
           "wrong": "{n} van de 64 tekens kloppen; een gestreept paars kader wijst elk teken aan dat niet klopt.",
           "found": "gevonden met vragen", "true": "de tekens van de weights",
           "stops": "Jouw guess_signs stopt met een fout bij guess_signs(ask, STOP, 0.01): de check hierboven zegt waar.",
           "off": "Jouw guess_signs geeft geen 64 tekens voor guess_signs(ask, STOP, 0.01), dus er is geen plaatje om te tekenen."},
}


def signs_picture(seen) -> str:
    fn = _fn(seen, "guess_signs")
    if fn is None:
        return ""
    w = _SIGN_WORDS[seen.lang]
    ask = Ask(W, B, "ask")
    try:
        got = with_time_limit(lambda: fn(ask, np.array(STOP), 0.01))
    except Exception:
        return yours.figure(w["title"], _note([w["stops"]]))
    signs = _as_picture(got)
    if signs is None:
        return yours.figure(w["title"], _note([w["off"]]))
    truth = np.sign(np.array(W))
    n = int(np.sum(np.sign(signs) == truth))
    body = _signs(signs, html.escape(w["found"]), truth) + _signs(truth, html.escape(w["true"]))
    return yours.figure(w["title"], f"<div>{body}</div>"
                        + _note([w["said"].format(q=ask.count), w["fine"] if n == 64 else w["wrong"].format(n=n)]))


# --- the demo: along the signs, or at random -------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "Every pixel moved by {eps}: along the signs of the weights, or at random",
           "along": "along the signs", "random": "up or down at random",
           "said": "Both pictures change every pixel by the same amount. Only the direction differs: along the signs, "
                   "every small step pushes the logit the same way, and the steps add up."},
    "nl": {"title": "Elke pixel {eps} opgeschoven: langs de tekens van de weights, of willekeurig",
           "along": "langs de tekens", "random": "willekeurig omhoog of omlaag",
           "said": "Beide plaatjes veranderen elke pixel even veel. Alleen de richting verschilt: langs de tekens duwt "
                   "elke kleine stap de logit dezelfde kant op, en tellen de stappen op."},
}


def demo_picture(eps, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    eps = float(eps)
    along = np.array(STOP) + eps * np.sign(np.array(W))
    random = np.array(STOP) + eps * RANDOM_SIGNS
    body = (_pixels(STOP, "STOP<br>" + _says(STOP, lang))
            + _pixels(along, html.escape(w["along"]) + "<br>" + _says(along, lang))
            + _pixels(random, html.escape(w["random"]) + "<br>" + _says(random, lang)))
    return yours.figure(w["title"].format(eps=_eps_text(eps, lang)), f"<div>{body}</div>" + _note([w["said"]]))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["fgsm", "smallest-eps", "signs", "goal", "knowledge"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    try:
        g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    except (TypeError, ValueError):
        return False
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-6, atol=1e-9))


def _same_picture(got, want) -> bool:
    """fgsm as the lesson writes it, or with every pixel kept between 0 and 1, as a real attack also does."""
    return _close(got, want) or _close(got, np.clip(np.asarray(want, dtype=float), 0, 1))


def _value(v) -> str:
    """{got} and {want}: a number with four decimals; more numbers as the first three of them."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return _short(v)
    if a.ndim == 0:
        return f"{float(a):.4f}"
    flat = a.ravel()
    return "np.array([" + ", ".join(f"{x:.4f}" for x in flat[:3]) + (", ...])" if flat.size > 3 else "])")


def _lesson_name(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(W|W2|B|B2|STOP|STOP_2)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_lesson_name,
                 en="With `{call}`, your function uses a name of the lesson, such as `W` or `STOP`, and the box does not know it. Use the names between the parentheses of the `def` line: the check calls your function with other pictures and another model too.",
                 nl="Met `{call}` gebruikt je functie een naam uit de les, zoals `W` of `STOP`, en die kent het vak niet. Gebruik de namen tussen de haakjes van de `def`-regel: de check roept je functie ook aan met andere plaatjes en een ander model.")

# exercise 1: fgsm
_F_HEAD = ("def fgsm(image, w, eps):\n"
           "    # move every pixel of image by eps, in the direction that raises the logit of the model with weights w\n")
_F_START = _F_HEAD + "    ...\n"
_F = _F_HEAD + "    return image + eps * np.sign(w)\n"
_F_MINUS = _F_HEAD + "    return image - eps * np.sign(w)\n"
_F_GRAD = _F_HEAD + "    return image + eps * w\n"
_F_NOEPS = _F_HEAD + "    return image + np.sign(w)\n"
_F_IMAGE = _F_HEAD + "    return image + eps * np.sign(image)\n"
_F_DOT = _F_HEAD + "    return np.dot(image, eps * np.sign(w))\n"


def _one_number(got, text) -> bool:
    return np.ndim(got) == 0 and _number(got) is not None


# exercise 2: smallest_eps
_E_HEAD = ("def smallest_eps(image, w, b):\n"
           "    # the eps at which fgsm(image, w, eps) brings the logit of the model exactly to 0\n")
_E_START = _E_HEAD + "    ...\n"
_E = _E_HEAD + ("    z = np.dot(w, image) + b\n"
                "    return -z / np.sum(abs(w))\n")
_E_NOB = _E_HEAD + "    return -np.dot(w, image) / np.sum(abs(w))\n"
_E_NOABS = _E_HEAD + "    return -(np.dot(w, image) + b) / np.sum(w)\n"
_E_SIGN = _E_HEAD + "    return (np.dot(w, image) + b) / np.sum(abs(w))\n"
_E_SQUARE = _E_HEAD + "    return -(np.dot(w, image) + b) / np.sum(w * w)\n"
_E_MANY = _E_HEAD + "    return -(np.dot(w, image) + b) / abs(w)\n"


def _many(got, text) -> bool:
    return np.ndim(got) > 0


# exercise 3: guess_signs
_G_HEAD = ("def guess_signs(ask, image, delta):\n"
           "    # ask(picture) gives only the probability of a speed limit sign; find the sign of each pixel's weight\n")
_G_START = _G_HEAD + "    ...\n"
_G_BODY = ("    base = ask(image)\n"
           "    signs = []\n"
           "    for i in range(len(image)):\n"
           "        bumped = np.array(image)\n"
           "        bumped[i] = bumped[i] + delta\n"
           "        if ask(bumped) > base:\n"
           "            signs.append(1)\n"
           "        else:\n"
           "            signs.append(-1)\n"
           "    return np.array(signs)\n")
_G = _G_HEAD + _G_BODY


def _g(old: str, new: str) -> str:
    assert _G_BODY.count(old) == 1, old
    return _G_HEAD + _G_BODY.replace(old, new)


_G_ALIAS = _g("bumped = np.array(image)", "bumped = image")
_G_FLIP = _g("if ask(bumped) > base:", "if ask(bumped) < base:")
_G_BOOL = _g("        if ask(bumped) > base:\n            signs.append(1)\n        else:\n            signs.append(-1)\n",
             "        signs.append(ask(bumped) > base)\n")
_G_ONE = _g("            signs.append(-1)", "            signs.append(1)")


def _no_copy(got, text) -> bool:
    """The new picture is the image itself, so each step keeps the pixels the steps before it moved."""
    code = "\n".join(line.split("#", 1)[0] for line in str(text).splitlines())
    return re.search(r"^\s*(\w+)\s*=\s*image(\[:\])?\s*$", code, re.MULTILINE) is not None


def _zero_one(got, text) -> bool:
    try:
        a = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    return a.size == 64 and set(np.unique(a)) == {0.0, 1.0}


_EPS_SHOWN = {lang: _three(EPS_STOP, lang) for lang in ("en", "nl")}

EXERCISES = {
    "fgsm": FunctionBox(
        name="fgsm",
        picture=fgsm_picture,
        function="fgsm",
        names={"np": np},
        start=_F_START,
        solution=_F,
        visible=[(STOP, W, 0.1)],
        hidden=[(STOP, W, 0.25), (STOP_2, W, 0.05), (STOP, W2, 0.1)],
        compare=_same_picture,
        show=_value,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("one-number", matches=_one_number,
                    en="`{call}` gives one number, and it should give a picture: 64 pixels, each moved by `eps`. Start from `image` and add to it, so all 64 stay.",
                    nl="`{call}` geeft één getal, en het moet een plaatje zijn: 64 pixels, elk opgeschoven met `eps`. Begin bij `image` en tel daarbij op, zodat ze alle 64 blijven."),
            Mistake("minus", wrong=_F_MINUS,
                    en="`{call}` moves every pixel the wrong way: the logit goes down, and the model is even surer that it sees a stop sign. Where a weight is positive, a higher pixel raises the logit.",
                    nl="`{call}` schuift elke pixel de verkeerde kant op: de logit gaat omlaag, en het model is nog zekerder dat het een stopbord ziet. Waar een weight positief is, wordt de logit hoger als de pixel omhoog gaat."),
            Mistake("grad", wrong=_F_GRAD,
                    en="`{call}` moves each pixel by `eps` times its weight, so a pixel with a small weight hardly moves. FGSM moves every pixel by the same `eps`: only the direction comes from the weight, + or −.",
                    nl="`{call}` schuift elke pixel op met `eps` keer de weight van die pixel, dus een pixel met een kleine weight beweegt bijna niet. FGSM schuift elke pixel even ver op, `eps`: alleen de richting komt van de weight, + of −."),
            Mistake("no-eps", wrong=_F_NOEPS,
                    en="`{call}` moves every pixel by a whole 1, the full range from 0 to 1. Multiply the direction by `eps`, so each pixel moves only a little.",
                    nl="`{call}` schuift elke pixel een hele 1 op, de hele weg van 0 tot 1. Vermenigvuldig de richting met `eps`, zodat elke pixel maar een beetje beweegt."),
            Mistake("image-sign", wrong=_F_IMAGE,
                    en="`{call}` takes its direction from the picture itself, and every pixel is positive, so it only moves pixels up. The direction comes from the model: the signs of its weights `w`.",
                    nl="`{call}` haalt de richting uit het plaatje zelf, en elke pixel is positief, dus elke pixel gaat alleen omhoog. De richting komt van het model: de tekens van de weights `w`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the picture with every pixel moved by `eps`, up where its weight is positive and down where it is negative.",
                      "nl": "Vervang de `...`: geef het plaatje terug met elke pixel `eps` opgeschoven, omhoog waar de weight positief is en omlaag waar die negatief is."},
            "wrong": {"en": "`{call}` gives other pixels than it should. Every pixel moves by the same `eps`, in the direction of the sign of its weight; the picture below marks what differs.",
                      "nl": "`{call}` geeft andere pixels dan zou moeten. Elke pixel schuift even ver op, `eps`, in de richting van het teken van de weight; het plaatje hieronder wijst aan wat verschilt."},
            "almost": {"en": "Right for `fgsm(STOP, W, 0.1)`. But `{call}` gives other pixels than it should. Does your function use the `image`, `w` and `eps` it gets?",
                       "nl": "Goed voor `fgsm(STOP, W, 0.1)`. Maar `{call}` geeft andere pixels dan zou moeten. Gebruikt je functie de `image`, `w` en `eps` die ze krijgt?"},
            "correct": {"en": "Your `fgsm` is right in all {n} cases. The picture shows what it makes of STOP: with eps = 0.25, the model reads a speed limit sign.",
                        "nl": "Je `fgsm` klopt in alle {n} gevallen. Het plaatje laat zien wat ze van STOP maakt: met eps = 0,25 leest het model een snelheidsbord."},
        },
    ),
    "smallest-eps": FunctionBox(
        name="smallest-eps",
        picture=eps_picture,
        function="smallest_eps",
        names={"np": np},
        start=_E_START,
        solution=_E,
        visible=[(STOP, W, B)],
        hidden=[(STOP_2, W, B), (STOP, W2, B2)],
        compare=_close,
        show=_value,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("many", matches=_many,
                    en="`{call}` gives more than one number: one for each pixel. `eps` is one number, the same for all 64 pixels. Add up over all pixels with `np.sum`.",
                    nl="`{call}` geeft meer dan één getal: één voor elke pixel. `eps` is één getal, hetzelfde voor alle 64 pixels. Tel op over alle pixels met `np.sum`."),
            Mistake("no-b", wrong=_E_NOB,
                    en="`{call}` gives `{got}`, and it should give `{want}`. The logit is the dot product plus the bias term `b`: leave out `b`, and the step brings the dot product to 0, not the logit.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. De logit is het inproduct plus de bias term `b`: laat je `b` weg, dan brengt de stap het inproduct op 0, niet de logit."),
            Mistake("no-abs", wrong=_E_NOABS,
                    en="`{call}` gives `{got}`, and it should give `{want}`. Each pixel moves the logit by `eps` times the size of its weight, whatever its sign: negative weights push the same way as positive ones, as the pixel moves down there. Add up `abs(w)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Elke pixel verschuift de logit met `eps` keer de grootte van de weight, welk teken die ook heeft: negatieve weights duwen dezelfde kant op als positieve, want daar schuift de pixel omlaag. Tel `abs(w)` op."),
            Mistake("sign", wrong=_E_SIGN,
                    en="`{call}` gives `{got}`, and it should give `{want}`. A stop sign has a logit below 0, so the logit has to go up by minus the logit: put a minus in front of it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Een stopbord heeft een logit onder 0, dus de logit moet omhoog met min de logit: zet er een min voor."),
            Mistake("square", wrong=_E_SQUARE,
                    en="`{call}` gives `{got}`, and it should give `{want}`. With fgsm, each pixel moves by `eps`, so the logit moves by `eps` times the size of the weight, `abs(w)`, not by the square of the weight.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Met fgsm schuift elke pixel `eps` op, dus de logit schuift met `eps` keer de grootte van de weight, `abs(w)`, niet met het kwadraat van de weight."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: work out the logit of `image`, and how much one step of fgsm with eps = 1 would raise it.",
                      "nl": "Vervang de `...`: reken de logit van `image` uit, en hoeveel één stap fgsm met eps = 1 die zou verhogen."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. fgsm with `eps` raises the logit by `eps` times the sum of the sizes of the weights; the logit of `image` has to reach 0.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. fgsm met `eps` verhoogt de logit met `eps` keer de som van de groottes van de weights; de logit van `image` moet 0 worden."},
            "almost": {"en": "Right for `smallest_eps(STOP, W, B)`. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use the `image`, `w` and `b` it gets?",
                       "nl": "Goed voor `smallest_eps(STOP, W, B)`. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie de `image`, `w` en `b` die ze krijgt?"},
            "correct": {"en": f"Your `smallest_eps` is right in all {{n}} cases: STOP needs {_EPS_SHOWN['en']} per pixel. The picture tries 0.01 less and 0.01 more.",
                        "nl": f"Je `smallest_eps` klopt in alle {{n}} gevallen: STOP heeft {_EPS_SHOWN['nl']} per pixel nodig. Het plaatje probeert 0,01 minder en 0,01 meer."},
        },
    ),
    "signs": FunctionBox(
        name="signs",
        picture=signs_picture,
        function="guess_signs",
        names={"np": np},
        start=_G_START,
        solution=_G,
        visible=[(ASK, STOP, 0.01)],
        hidden=[(ASK, STOP_2, 0.01), (ASK_2, STOP, 0.05)],
        compare=_close,
        show=_value,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("ones-zeros", matches=_zero_one,
                    en="`{call}` gives 1 and 0, or True and False. A sign is 1 or −1: keep −1 for a pixel when raising it does not give a higher probability.",
                    nl="`{call}` geeft 1 en 0, of True en False. Een teken is 1 of −1: bewaar −1 voor een pixel als die ophogen geen hogere kans geeft."),
            Mistake("no-copy", matches=_no_copy,
                    en="`{call}` gives other signs than it should. Your new picture is the image itself, not a copy: each step changes `image`, so it keeps the pixels the steps before it moved, and the answers no longer belong to one pixel. Make a copy for each pixel with `np.array(image)`.",
                    nl="`{call}` geeft andere tekens dan zou moeten. Je nieuwe plaatje is het plaatje zelf, geen kopie: elke stap verandert `image`, dus houdt het de pixels die de stappen ervoor verschoven, en horen de antwoorden niet meer bij één pixel. Maak voor elke pixel een kopie met `np.array(image)`."),
            Mistake("flip", wrong=_G_FLIP,
                    en="`{call}` gives every sign the wrong way round. When raising a pixel makes the probability of a speed limit sign go up, its weight is positive.",
                    nl="`{call}` geeft elk teken andersom. Als een pixel ophogen de kans op een snelheidsbord laat stijgen, is de weight van die pixel positief."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: ask once about `image`, then for each pixel about a copy with that one pixel a bit higher, and keep 1 or −1.",
                      "nl": "Vervang de `...`: vraag één keer naar `image`, en dan voor elke pixel naar een kopie waarin alleen die pixel iets hoger is, en bewaar 1 of −1."},
            "wrong": {"en": "`{call}` gives other signs than it should. For each pixel: 1 when the copy with that one pixel raised gets a higher probability than `image` itself, −1 when not. The picture below marks each sign that differs.",
                      "nl": "`{call}` geeft andere tekens dan zou moeten. Voor elke pixel: 1 als de kopie met die ene pixel opgehoogd een hogere kans krijgt dan `image` zelf, −1 als dat niet zo is. Het plaatje hieronder wijst elk teken aan dat verschilt."},
            "almost": {"en": "Right for `guess_signs(ask, STOP, 0.01)`. But `{call}` gives other signs than it should. Does your function use the `ask`, `image` and `delta` it gets?",
                       "nl": "Goed voor `guess_signs(ask, STOP, 0.01)`. Maar `{call}` geeft andere tekens dan zou moeten. Gebruikt je functie de `ask`, `image` en `delta` die ze krijgt?"},
            "correct": {"en": "Your `guess_signs` is right in all {n} cases: it finds the sign of all 64 weights from the answers alone. The picture counts the questions.",
                        "nl": "Je `guess_signs` klopt in alle {n} gevallen: ze vindt het teken van alle 64 weights uit alleen de antwoorden. Het plaatje telt de vragen."},
        },
    ),
}

CASES = [
    ("fgsm", _F_START, "empty", None),
    ("fgsm", _F, "correct", None),
    ("fgsm", _F_HEAD + "    step = eps * np.sign(w)\n    return image + step\n", "correct", None),
    ("fgsm", _F_HEAD + "    return image + eps * (w > 0) - eps * (w < 0)\n", "correct", None),
    ("fgsm", _F_HEAD + "    return np.clip(image + eps * np.sign(w), 0, 1)\n", "correct", None),
    ("fgsm", _F_MINUS, "wrong", "minus"),
    ("fgsm", _F_GRAD, "wrong", "grad"),
    ("fgsm", _F_NOEPS, "wrong", "no-eps"),
    ("fgsm", _F_IMAGE, "wrong", "image-sign"),
    ("fgsm", _F_DOT, "wrong", "one-number"),
    ("fgsm", _F_HEAD + "    return image + 0.1 * np.sign(w)\n", "almost", None),
    ("fgsm", _F_HEAD + "    return image + eps * np.sign(W)\n", "error", "lesson-names"),
    ("fgsm", _F_HEAD + "    return image * (1 + eps)\n", "wrong", None),
    ("smallest-eps", _E_START, "empty", None),
    ("smallest-eps", _E, "correct", None),
    ("smallest-eps", _E_HEAD + "    return -(np.dot(w, image) + b) / np.sum(np.abs(w))\n", "correct", None),
    ("smallest-eps", _E_NOB, "wrong", "no-b"),
    ("smallest-eps", _E_NOABS, "wrong", "no-abs"),
    ("smallest-eps", _E_SIGN, "wrong", "sign"),
    ("smallest-eps", _E_SQUARE, "wrong", "square"),
    ("smallest-eps", _E_MANY, "wrong", "many"),
    ("smallest-eps", _E_HEAD + f"    return {EPS_STOP!r}\n", "almost", None),
    ("smallest-eps", _E_HEAD + "    return -(np.dot(W, image) + b) / np.sum(abs(W))\n", "error", "lesson-names"),
    ("smallest-eps", _E_HEAD + "    return -(np.dot(w, image) + b)\n", "wrong", None),
    ("signs", _G_START, "empty", None),
    ("signs", _G, "correct", None),
    ("signs", _G_HEAD + "    signs = []\n    for i in range(len(image)):\n        bumped = image.copy()\n"
              "        bumped[i] += delta\n        if ask(bumped) > ask(image):\n            signs.append(1)\n"
              "        else:\n            signs.append(-1)\n    return signs\n", "correct", None),
    ("signs", _G_ALIAS, "wrong", "no-copy"),
    ("signs", _G_FLIP, "wrong", "flip"),
    ("signs", _G_BOOL, "wrong", "ones-zeros"),
    ("signs", _G_ONE, "wrong", None),
]
