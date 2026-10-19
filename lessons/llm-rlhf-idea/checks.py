# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-rlhf-idea: penalized_reward, the reward of an answer minus the KL penalty of RLHF for it;
best_policy, the policy that maximizes reward minus beta times the KL divergence from the reference model; and a
choice. The demo is four answers to one instruction, with their probabilities under the course's SFT model and the
rewards of a flawed reward model that counts the requested words: a slider for beta shows the policy, its reward, its
KL divergence, and when it turns to the answer that games the reward."""
import math
import warnings

from everything_ai.check import FunctionBox, Mistake

WORDS = ["kite", "park"]
# Four answers to "Write a story with the words: kite and park.", the same length but the last.
ANSWERS = {
    "good": "Tom took his red kite to the park. The wind was strong, and the kite flew high over the trees. Tom laughed and ran after it.",
    "one": "Tom took his red kite to the hill. The wind was strong, and the kite flew high over the trees. Tom laughed and ran after it.",
    "none": "Tom took his red ball to the hill. The wind was strong, and the ball flew high over the trees. Tom laughed and ran after it.",
    "hack": "The kite in the park, the kite in the park, the kite in the park, the kite in the park, the kite in the park.",
}
BETAS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)   # the values of the slider


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def reward(text: str) -> int:
    """The flawed reward model of the demo: how often the requested words stand in the answer. It learned that an
    answer that uses the words is better, and so that more of them is better still."""
    words = [w.strip(".,!?;:\"'").lower() for w in text.split()]
    return sum(words.count(w) for w in WORDS)


def reference() -> dict:
    """The probability of each answer under the course's SFT model, out of these four: exp of its log-probability,
    divided by their sum."""
    top = max(MEASURED["logp"].values())
    weights = {k: math.exp(v - top) for k, v in MEASURED["logp"].items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def kl(p: dict, q: dict) -> float:
    """The KL divergence of p from q in nats, with the natural log as for log-probabilities."""
    return sum(p[k] * math.log(p[k] / q[k]) for k in p if p[k] > 0)


def policy(beta: float, best_policy) -> dict | str:
    """The learner's best policy over the four answers at `beta`, as a dict; "policy" when it stops with an error or
    gives no probabilities."""
    ref = reference()
    keys = list(ANSWERS)
    try:
        with warnings.catch_warnings():   # NumPy's "overflow in exp" of a float32 version: the note explains, not Python
            warnings.simplefilter("ignore", RuntimeWarning)
            out = [float(v) for v in best_policy([ref[k] for k in keys], [reward(ANSWERS[k]) for k in keys], beta)]
    except Exception:
        return "policy"
    if len(out) != len(keys) or any(not (v >= 0 and math.isfinite(v)) for v in out) or abs(sum(out) - 1) > 1e-6:
        return "policy"
    return dict(zip(keys, out))


def bars_picture(ref: dict, pol: dict, labels: dict):
    """Two bars per answer: the probability under the reference model, in grey, and under the policy, in the colour of
    what a model learns."""
    import html
    from everything_ai import draw
    W, row, bar = 380, 46, 14
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    left, width = 150, W - 150 - 44
    y = 34
    for k in ANSWERS:
        parts.append(f'<text x="0" y="{y + 18}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(labels[k])}</text>')
        for i, (value, fill, stroke) in enumerate(((ref[k], "--ea-data-wash", "--ea-line"), (pol[k], "--ea-param-wash", "--ea-param"))):
            yy = y + i * (bar + 2)
            parts.append(f'<rect x="{left}" y="{yy}" width="{max(1.0, width * value):.1f}" height="{bar}" style="fill:var({fill});stroke:var({stroke});stroke-width:1"/>'
                         f'<text x="{left + max(1.0, width * value) + 4:.1f}" y="{yy + 11}" style="font-size:11px;fill:var(--ea-ink-2)">{labels["pct"](value)}</text>')
        y += row
    parts.append(f'<rect x="{left}" y="{y + 2}" width="12" height="10" style="fill:var(--ea-data-wash);stroke:var(--ea-line)"/>'
                 f'<text x="{left + 16}" y="{y + 11}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["ref"])}</text>'
                 f'<rect x="{left + 140}" y="{y + 2}" width="12" height="10" style="fill:var(--ea-param-wash);stroke:var(--ea-param)"/>'
                 f'<text x="{left + 156}" y="{y + 11}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["policy"])}</text>')
    return draw._svg(W, y + 18, labels["alt"], "".join(parts), natural=True)


# --- measured: the log-probability of each answer after its instruction under the course's SFT model,
# data/course-sft.pt, the end-of-text included; on the CPU ---
MEASURED = {
    "logp": {"good": -135.358, "one": -139.358, "none": -138.946, "hack": -153.283},
}
# --- end of measured ---

REQUIRED = ["penalized-reward", "best-policy", "small-beta"]


def _texts() -> dict:
    return {
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    }


_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...") or line.strip() == "return ..."
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _shown(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_shown(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(float(f"{v:.4g}"))
    return repr(v)


def _close(got, want) -> bool:
    try:
        if isinstance(want, (list, tuple)):
            return len(got) == len(want) and all(math.isclose(float(g), float(w), rel_tol=1e-6, abs_tol=1e-12) for g, w in zip(got, want))
        return math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


# --- exercise 1: penalized_reward --------------------------------------------------------------------------------------

_R_HEAD = "def penalized_reward(reward, logp, logp_ref, beta):\n"
_R = _R_HEAD + "    return reward - beta * (logp - logp_ref)\n"
_R_START = _R_HEAD + "    # the reward, minus beta times how much more likely the model makes the answer than the reference model\n    return ...\n"

# --- exercise 2: best_policy -------------------------------------------------------------------------------------------

_P_HEAD = "import math\n\n\ndef best_policy(ref, rewards, beta):\n"
_P = (_P_HEAD + "    weights = []\n    for p, r in zip(ref, rewards):\n        weights.append(p * math.exp(r / beta))\n"
      "    total = sum(weights)\n    return [w / total for w in weights]\n")
_P_START = (_P_HEAD + "    weights = []\n    for p, r in zip(ref, rewards):\n"
            "        # the probability of the reference model, times e to the reward over beta\n        ...\n"
            "    # divided by their sum, so that they add up to 1\n    total = sum(weights)\n    return ...\n")
_P_NO_REF = _P.replace("weights.append(p * math.exp(r / beta))", "weights.append(math.exp(r / beta))")
_P_TIMES = _P.replace("math.exp(r / beta)", "math.exp(r * beta)")
_P_RAW = _P.replace("    return [w / total for w in weights]\n", "    return weights\n")
_P_LINEAR = _P.replace("weights.append(p * math.exp(r / beta))", "weights.append(p * r / beta)")
_P_FLOAT32 = ("import torch\n\n\ndef best_policy(ref, rewards, beta):\n"
              "    weights = torch.tensor(ref) * torch.exp(torch.tensor(rewards) / beta)\n    return (weights / weights.sum()).tolist()\n")


def _not_finite(got) -> bool:
    try:
        return any(not math.isfinite(float(v)) for v in got)
    except (TypeError, ValueError):
        return False

EXERCISES = {
    "penalized-reward": FunctionBox(
        name="penalized-reward",
        function="penalized_reward",
        start=_R_START,
        solution=_R,
        visible=[(3.0, -135.0, -138.0, 0.2)],
        hidden=[(0.0, -10.0, -10.0, 1.0), (2.0, -20.0, -15.0, 0.1), (5.0, -3.0, -4.0, 0.0), (1.0, -7.5, -9.0, 2.0)],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("sign", wrong=_R_HEAD + "    return reward + beta * (logp - logp_ref)\n",
                    en=_CALL + ": your function gives more, not less, when the model makes the answer more likely than the reference model does. The penalty is `beta * (logp - logp_ref)`, and it is taken off the reward.",
                    nl=_CALL_NL + ": je functie geeft meer, niet minder, als het model het antwoord waarschijnlijker maakt dan het reference model. De straf is `beta * (logp - logp_ref)`, en die gaat van de reward af."),
            Mistake("no-beta", wrong=_R_HEAD + "    return reward - (logp - logp_ref)\n",
                    en=_CALL + ": your penalty leaves out `beta`, which says how much the penalty weighs against the reward.",
                    nl=_CALL_NL + ": je straf laat `beta` weg, en die zegt hoe zwaar de straf weegt tegen de reward."),
            Mistake("abs", wrong=_R_HEAD + "    return reward - beta * abs(logp - logp_ref)\n",
                    en=_CALL + ": the penalty for one answer can be below 0. When the model makes an answer less likely than the reference model does, `logp - logp_ref` is negative, and the answer gets more than its reward. Only the mean over the answers of the model, the KL divergence, is never below 0.",
                    nl=_CALL_NL + ": de straf voor één antwoord kan onder 0 liggen. Maakt het model een antwoord minder waarschijnlijk dan het reference model, dan is `logp - logp_ref` negatief, en krijgt het antwoord meer dan zijn reward. Alleen het gemiddelde over de antwoorden van het model, de KL-divergentie, is nooit negatief."),
            Mistake("never-below-zero", wrong=_R_HEAD + "    return reward - beta * max(0, logp - logp_ref)\n",
                    en=_CALL + ": the penalty for one answer can be below 0. When the model makes an answer less likely than the reference model does, `logp - logp_ref` is negative, and the answer gets more than its reward. Only the mean over the answers of the model, the KL divergence, is never below 0.",
                    nl=_CALL_NL + ": de straf voor één antwoord kan onder 0 liggen. Maakt het model een antwoord minder waarschijnlijk dan het reference model, dan is `logp - logp_ref` negatief, en krijgt het antwoord meer dan zijn reward. Alleen het gemiddelde over de antwoorden van het model, de KL-divergentie, is nooit negatief."),
        ],
        texts={
            **_texts(),
            "empty": {"en": "Replace the `...`: the reward minus `beta` times the difference of the two log-probabilities.",
                      "nl": "Vervang de `...`: de reward min `beta` keer het verschil van de twee log-kansen."},
            "wrong": {"en": _CALL + ". The reward, minus `beta` times the log-probability of the model minus that of the reference model?",
                      "nl": _CALL_NL + ". De reward, min `beta` keer de log-kans van het model min die van het reference model?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use all four values, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie alle vier de waarden, en geen vaste getallen?"},
            "correct": {"en": "Your function gives what RLHF rewards: the reward of the answer, less `beta` for every nat by which the model makes the answer more likely than the reference model does.",
                        "nl": "Je functie geeft wat RLHF beloont: de reward van het antwoord, min `beta` voor elke nat die het model het antwoord waarschijnlijker maakt dan het reference model."},
        },
    ),
    "best-policy": FunctionBox(
        name="best-policy",
        function="best_policy",
        start=_P_START,
        solution=_P,
        visible=[([0.5, 0.3, 0.2], [1.0, 0.0, 2.0], 1.0)],
        hidden=[([0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.0, 0.0], 1.0), ([0.9, 0.1], [0.0, 3.0], 0.5),
                ([0.6, 0.4], [2.0, 1.0], 10.0), ([0.7, 0.2, 0.1], [0.0, 1.0, 1.0], 0.25), ([0.5, 0.5], [12.0, 11.5], 0.125)],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-ref", wrong=_P_NO_REF,
                    en=_CALL + ": your policy forgets the reference model, so it is the softmax of `r / beta` alone, and the KL penalty has nothing to hold it to. Multiply every `math.exp(r / beta)` by the probability `p` of the reference model.",
                    nl=_CALL_NL + ": je policy vergeet het reference model, dus is hij de softmax van alleen `r / beta`, en heeft de KL penalty niets om hem bij te houden. Vermenigvuldig elke `math.exp(r / beta)` met de kans `p` van het reference model."),
            Mistake("times-beta", wrong=_P_TIMES,
                    en=_CALL + ": you multiply the reward by `beta`. A bigger `beta` is a heavier penalty, so it must make the rewards count less: divide, `r / beta`.",
                    nl=_CALL_NL + ": je vermenigvuldigt de reward met `beta`. Een grotere `beta` is een zwaardere straf, dus moet hij de rewards minder laten tellen: deel, `r / beta`."),
            Mistake("not-normalized", wrong=_P_RAW,
                    en=_CALL + ": these weights do not add up to 1, so they are not probabilities yet. Divide each by their sum.",
                    nl=_CALL_NL + ": deze getallen tellen niet op tot 1, dus zijn het nog geen kansen. Deel elk door hun som."),
            Mistake("linear", wrong=_P_LINEAR,
                    en=_CALL + ": the reward goes into an exponent, `math.exp(r / beta)`: one more reward multiplies a probability by the same factor, whatever the reward was.",
                    nl=_CALL_NL + ": de reward gaat in een exponent, `math.exp(r / beta)`: één reward meer vermenigvuldigt een kans met dezelfde factor, wat de reward ook was."),
            Mistake("overflow", matches=lambda got, text: _not_finite(got),
                    en=_CALL + ": somewhere a weight became `inf` or `nan`, and then dividing gives no probabilities. This call needs e to the power 96, and the demo e to the power 200: a float32 tensor or array stops at about e to the power 88. A Python float with `math.exp`, as at the start of the box, goes up to about e to the power 709.",
                    nl=_CALL_NL + ": ergens werd een getal `inf` of `nan`, en dan geeft delen geen kansen. Deze aanroep heeft e tot de macht 96 nodig, en de demo e tot de macht 200: een float32-tensor of -array houdt op bij ongeveer e tot de macht 88. Een Python-float met `math.exp`, zoals aan het begin van het vak, gaat tot ongeveer e tot de macht 709."),
        ],
        texts={
            **_texts(),
            "empty": {"en": "Replace the two `...`: a weight for every answer, and the weights divided by their sum.",
                      "nl": "Vervang de twee `...`: een getal voor elk antwoord, en die getallen gedeeld door hun som."},
            "wrong": {"en": _CALL + ". For every answer the probability of the reference model times e to the reward over `beta`, and then divided by the sum of all of them?",
                      "nl": _CALL_NL + ". Voor elk antwoord de kans van het reference model keer e tot de reward gedeeld door `beta`, en dan gedeeld door de som van allemaal?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Equal rewards, another `beta`, and more answers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gelijke rewards, een andere `beta`, en meer antwoorden?"},
            "correct": {"en": "Your policy is the reference model, tilted towards the answers with more reward: the more, the smaller `beta` is.",
                        "nl": "Je policy is het reference model, gekanteld naar de antwoorden met meer reward: des te meer naarmate `beta` kleiner is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("penalized-reward", _R_START, "empty", None),
    ("penalized-reward", _R, "correct", None),
    ("penalized-reward", _R_HEAD + "    penalty = beta * logp - beta * logp_ref\n    return reward - penalty\n", "correct", None),
    ("penalized-reward", _R_HEAD + "    return reward + beta * (logp - logp_ref)\n", "wrong", "sign"),
    ("penalized-reward", _R_HEAD + "    return reward - beta * (logp_ref - logp)\n", "wrong", "sign"),
    ("penalized-reward", _R_HEAD + "    return reward - (logp - logp_ref)\n", "wrong", "no-beta"),
    ("penalized-reward", _R_HEAD + "    return reward - beta * abs(logp - logp_ref)\n", "wrong", "abs"),
    ("penalized-reward", _R_HEAD + "    return reward - beta * max(0, logp - logp_ref)\n", "wrong", "never-below-zero"),
    ("penalized-reward", _R_HEAD + "    return beta * (logp - logp_ref)\n", "wrong", None),
    ("best-policy", _P_START, "empty", None),
    ("best-policy", _P, "correct", None),
    ("best-policy", _P_HEAD + "    logs = [math.log(p) + r / beta for p, r in zip(ref, rewards)]\n    top = max(logs)\n"
                    "    weights = [math.exp(v - top) for v in logs]\n    return [w / sum(weights) for w in weights]\n", "correct", None),
    ("best-policy", _P_NO_REF, "wrong", "no-ref"),
    ("best-policy", _P_TIMES, "wrong", "times-beta"),
    ("best-policy", _P_RAW, "wrong", "not-normalized"),
    ("best-policy", _P_LINEAR, "wrong", "linear"),
    ("best-policy", _P_FLOAT32, "wrong", "overflow"),
]
