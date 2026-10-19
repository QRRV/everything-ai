# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-dpo-loss: answer_logprob, the log-probability of the tokens of an answer from the logits
of a language model; dpo_loss, the DPO loss of preference pairs from four log-probabilities each; and a choice. The
demo trains the course's SFT model for STEPS steps of DPO on one preference pair, the story with both requested words
against the same story with "hill" in place of "park", with the learner's two functions, and shows the loss, how far
each answer moved from the reference model, and the probability of the word where the two answers differ."""
import importlib.util
import math
import time

from everything_ai import paths, projects
from everything_ai.check import Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _checks("llm-rlhf-idea", "everything_ai_checks_llm_rlhf_idea_for_dpo")
T = _checks("llm-sft", "everything_ai_checks_llm_sft_for_dpo")
S = T.S
BUILDS_ON = T.BUILDS_ON   # the model and the tokenizer of the course's SFT model
projects.builds_on(*BUILDS_ON)
SAVES = (("llm", "answer_logprob", "llm-dpo-loss", "answer-logprob"), ("llm", "dpo_loss", "llm-dpo-loss", "dpo-loss"))

WORDS = R.WORDS
CHOSEN, REJECTED = R.ANSWERS["good"], R.ANSWERS["one"]   # they differ in one word: "park" or "hill"
CONTEXT = "Tom took his red kite to the"                 # what both answers say before that word
NEXT = ("park", "hill")
BETAS = (0.05, 0.1, 0.2, 0.5, 1.0)   # the values of the slider
STEPS, LR = 20, 2e-5                 # AdamW on the whole model, one pair a step


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def setup() -> dict | str:
    """The course's SFT model, frozen, with the token ids of the instruction, the two answers and the words where they
    differ; "run" when a module of the learner's that it needs stops with an error."""
    tools = T.tools()
    if tools == "run":
        return "run"
    try:
        model, info, _ = T.load("course")
    except Exception:
        return "run"
    merges = {tuple(p): 256 + i for i, p in enumerate(info["merges"])}
    encode = tools["encode_fast"]
    prompt = encode(T.instruction(WORDS), merges)
    end = 256 + len(merges)
    for p in model.parameters():
        p.requires_grad_(False)
    return {"model": model, "start": len(prompt),
            "chosen": prompt + encode(CHOSEN, merges) + [end], "rejected": prompt + encode(REJECTED, merges) + [end],
            "context": prompt + encode(CONTEXT, merges), "next": {w: encode(" " + w, merges) for w in NEXT}}


def _course(name: str):
    return run_code(EXERCISES[name].solution).names[name.replace("-", "_")]


def word_probs(model, s: dict) -> dict:
    """The probability of each word of NEXT right after CONTEXT: the product of the probabilities of its tokens."""
    import torch
    out = {}
    with torch.no_grad():
        for w, toks in s["next"].items():
            ids = s["context"] + toks
            logits = model(torch.tensor([ids[:-1]]))[0]
            out[w] = math.exp(float(_course("answer-logprob")(logits, ids, len(s["context"]))))
    return out


def run(beta: float, answer_logprob, dpo_loss, s: dict | None = None) -> dict | str:
    """STEPS steps of DPO with AdamW on a copy of the course's SFT model, on the one pair, with the learner's functions.
    Per step the loss and how far each answer moved from the reference model, in nats; the probability of the two
    words before and after. "logprob" or "loss" when that function stops with an error or gives no tensor to train
    with; "run" when the model cannot be made."""
    import copy
    import torch
    s = setup() if s is None else s
    if s == "run":
        return "run"
    ref = s["model"]

    def logp(model, ids):
        return answer_logprob(model(torch.tensor([ids[:-1]]))[0], ids, s["start"])

    try:
        with torch.no_grad():   # the reference model is frozen: its two log-probabilities once, before the first step
            ref_c, ref_r = float(logp(ref, s["chosen"])), float(logp(ref, s["rejected"]))
    except Exception:
        return "logprob"
    policy = copy.deepcopy(ref)
    for p in policy.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    history, t0 = [], time.time()
    for step in range(STEPS + 1):
        try:
            pc, pr = logp(policy, s["chosen"]), logp(policy, s["rejected"])
            if not (isinstance(pc, torch.Tensor) and pc.requires_grad):
                return "logprob"
        except Exception:
            return "logprob"
        try:
            loss = dpo_loss(pc.reshape(1), pr.reshape(1), torch.tensor([ref_c]), torch.tensor([ref_r]), beta)
            if not (isinstance(loss, torch.Tensor) and loss.requires_grad and loss.numel() == 1 and math.isfinite(float(loss.detach()))):
                return "loss"
        except Exception:
            return "loss"
        history.append((float(loss.detach()), float(pc.detach()) - ref_c, float(pr.detach()) - ref_r))
        if step == STEPS:
            break
        opt.zero_grad()
        try:
            loss.backward()
        except Exception:
            return "loss"
        opt.step()
    policy.eval()
    return {"history": history, "before": word_probs(ref, s), "after": word_probs(policy, s), "seconds": time.time() - t0}


def demo(beta: float, answer_logprob=None, dpo_loss=None) -> tuple:
    """The run of the demo with the learner's functions, None for one that has not passed its check yet; where one of
    them stops on the model, the course's stands in. The result of run, and which of the learner's stood down."""
    s = setup()
    fns = {"logprob": answer_logprob or _course("answer-logprob"), "loss": dpo_loss or _course("dpo-loss")}
    failed = []
    while True:
        r = run(beta, fns["logprob"], fns["loss"], s)
        if not isinstance(r, str) or r not in fns:
            return r, failed
        if r in failed:   # it stopped again, with the course's own in its place: the course's two together
            return run(beta, _course("answer-logprob"), _course("dpo-loss"), s), failed
        failed.append(r)
        fns[r] = _course("answer-logprob" if r == "logprob" else "dpo-loss")


def percent(p: float, lang: str) -> str:
    """A probability as a percentage: one decimal from 0.1% on, three below it, and below 0.001% only that."""
    x = 100 * p
    if x >= 0.1:
        return f"{number(x, 1, lang)}%"
    if x >= 0.001:
        return f"{number(x, 3, lang)}%"
    return f"< {number(0.001, 3, lang)}%"


def loss_picture(beta: float, points: list, labels: dict) -> str:
    """The DPO loss of one pair against its margin at `beta`, in the colour of loss, with ln 2 dashed and a dot and
    its words for every (margin, words) of `points`."""
    from everything_ai import chart
    far = 40
    ms = [-far + 2 * far * i / 160 for i in range(161)]
    loss = lambda m: math.log1p(math.exp(-beta * m))   # -log sigmoid(beta m)
    p = chart.Panel((-far, far), (0, 4.5), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line([-far, far], [math.log(2)] * 2, "ink-3", 1.2, dashed=True)
    p.line(ms, [loss(m) for m in ms], "loss")
    for m, words in points:
        p.points([m], [loss(m)], "loss", 5)
        p.text(m + (2 if m >= 0 else -2), loss(m) + (0.6 if m >= 0 else -0.55), words, "ink", anchor="start" if m >= 0 else "end")   # below the curve on its steep side
    svg = chart.figure([p], alt=labels["alt"], width=560, height=300)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def curves_picture(history: list, labels: dict) -> str:
    """Two panels over the steps: the loss, in the colour of loss, with ln 2 dashed; and how far the chosen and the
    rejected answer moved from the reference model, in the colour of what a model learns, the rejected one dashed."""
    from everything_ai import chart
    steps = list(range(len(history)))
    losses = [h[0] for h in history]
    p1 = chart.Panel((0, len(history) - 1), (0, 0.8), title=labels["loss_title"], xlabel=labels["x"], ylabel=labels["loss"])
    p1.line([0, len(history) - 1], [math.log(2)] * 2, "ink-3", 1.2, dashed=True)
    p1.line(steps, losses, "loss")
    lo = min(-10, 10 * math.floor((min(h[2] for h in history) - 5) / 10))   # room below the end of the rejected line
    hi = max(10, 5 * math.ceil(max(h[1] for h in history) / 5))
    p2 = chart.Panel((0, len(history) - 1), (lo, hi), title=labels["moved_title"], xlabel=labels["x"], ylabel=labels["moved"])
    p2.line([0, len(history) - 1], [0, 0], "ink-3", 1.2, dashed=True)
    p2.line(steps, [h[1] for h in history], "param")
    p2.line(steps, [h[2] for h in history], "param", dashed=True)
    p2.text(len(history) - 1, history[-1][1] + 0.07 * (hi - lo), labels["chosen"], "ink", anchor="end")
    p2.text(len(history) - 1, history[-1][2] - 0.08 * (hi - lo), labels["rejected"], "ink", anchor="end")
    svg = chart.figure([p1, p2], alt=labels["alt"], width=280, height=260)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- measured: STEPS steps of DPO on the course's SFT model, data/course-sft.pt, with the course's functions, on the
# CPU; for every beta the loss, the nats each answer moved, the probability of "park" and "hill" after them, and the
# seconds at beta 0.1 with one thread ---
MEASURED = {
    "before": {"park": 0.003352, "hill": 5.877e-05},
    "betas": {
        0.05: {"loss": 0.1064, "chosen": 6.17, "rejected": -37.56, "after": {"park": 0.9353, "hill": 2.055e-13}},
        0.1: {"loss": 0.0221, "chosen": 6.46, "rejected": -31.55, "after": {"park": 0.8666, "hill": 3.193e-12}},
        0.2: {"loss": 0.0027, "chosen": 6.71, "rejected": -22.91, "after": {"park": 0.5477, "hill": 2.033e-10}},
        0.5: {"loss": 0.0001, "chosen": 5.59, "rejected": -14.07, "after": {"park": 0.09928, "hill": 1.47e-08}},
        1.0: {"loss": 0.0, "chosen": 4.57, "rejected": -10.14, "after": {"park": 0.03659, "hill": 1.137e-07}},
    },
    "seconds": 4.2,
}
# --- end of measured ---

REQUIRED = ["answer-logprob", "dpo-loss", "no-z"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_SCENARIO = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`"
_SCENARIO_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    try:
        if isinstance(want, (list, tuple)):
            return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
        if isinstance(got, (str, bytes)):
            return False
        return abs(float(got) - float(want)) <= 1e-4 * max(1.0, abs(float(want)))   # also float32, as in torch
    except (TypeError, ValueError):
        return False


# --- exercise 1: answer_logprob ---------------------------------------------------------------------------------------

_A_HEAD = "import torch\n\n\ndef answer_logprob(logits, ids, start):\n"
_A = (_A_HEAD + "    logp = torch.log_softmax(logits, dim=-1)\n    total = 0\n"
      "    for t in range(start, len(ids)):\n        total = total + logp[t - 1, ids[t]]\n    return total\n")
_A_START = (_A_HEAD + "    # the log-probability of every token of the vocabulary, at every place\n    logp = ...\n    total = 0\n"
            "    for t in range(start, len(ids)):\n        # add the log-probability the model gave token t\n        ...\n    return total\n")
_A_SETUP = ("import torch\ntorch.manual_seed(0)\n"
            "logits = torch.randn(5, 7)   # a model's logits for ids[:-1]: 5 places, a vocabulary of 7\n"
            "ids = [3, 1, 4, 1, 5, 2]\n")
_A_SCENARIOS = [
    # the lesson's example: an instruction of 2 tokens, then an answer of 4
    _A_SETUP + "round(float(answer_logprob(logits, ids, 2)), 4)",
    # an instruction of 1 token
    _A_SETUP + "round(float(answer_logprob(logits, ids, 1)), 4)",
    # an answer of one token, the last
    _A_SETUP + "round(float(answer_logprob(logits, ids, 5)), 4)",
    # another sequence, with bigger logits
    "import torch\ntorch.manual_seed(1)\nlogits = 3 * torch.randn(7, 11)\nids = [10, 0, 2, 9, 9, 4, 7, 1]\n"
    "round(float(answer_logprob(logits, ids, 4)), 4)",
    # DPO trains through it: the gradient has to reach the logits
    "import torch\ntorch.manual_seed(0)\nlogits = torch.randn(5, 7, requires_grad=True)\nids = [3, 1, 4, 1, 5, 2]\n"
    "answer_logprob(logits, ids, 2).backward()\n[round(float(v), 4) for v in logits.grad[:, 5]]",
]
_A_SAME_PLACE = (_A_HEAD + "    logp = torch.log_softmax(logits, dim=-1)\n    total = 0\n"
                 "    for t in range(start - 1, len(ids) - 1):\n        total = total + logp[t, ids[t]]\n    return total\n")
_A_ALL = _A.replace("range(start, len(ids))", "range(1, len(ids))")
_A_LATE = _A.replace("range(start, len(ids))", "range(start + 1, len(ids))")
_A_LOGITS = _A.replace("torch.log_softmax(logits, dim=-1)", "logits")
_A_PROBS = _A.replace("torch.log_softmax(logits, dim=-1)", "torch.softmax(logits, dim=-1)")
_A_MEAN = _A.replace("    return total\n", "    return total / (len(ids) - start)\n")
_A_ITEM = _A.replace("    return total\n", "    return total.item()\n")

# --- exercise 2: dpo_loss ---------------------------------------------------------------------------------------------

_D_HEAD = "import torch.nn.functional as F\n\n\ndef dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):\n"
_D = (_D_HEAD + "    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)\n"
      "    return -F.logsigmoid(beta * margin).mean()\n")
_D_START = (_D_HEAD + "    # how much more the policy gained on the reference model with the chosen answer than with the rejected one\n"
            "    margin = ...\n    # minus the log of the Bradley-Terry probability, as a mean over the pairs\n    return ...\n")
_D_SCENARIOS = [
    # the lesson's example: one pair, with a margin of 3, so that every term shows in the loss
    "import torch\nloss = dpo_loss(torch.tensor([-134.0]), torch.tensor([-141.0]), torch.tensor([-135.358]), torch.tensor([-139.358]), 0.1)\n"
    "round(float(loss), 4)",
    # the policy is still the reference model
    "import torch\nloss = dpo_loss(torch.tensor([-12.0, -40.0]), torch.tensor([-9.0, -41.0]), torch.tensor([-12.0, -40.0]), torch.tensor([-9.0, -41.0]), 0.5)\n"
    "round(float(loss), 4)",
    # three pairs, another beta; one of them the policy has the wrong way round
    "import torch\nloss = dpo_loss(torch.tensor([-20.0, -31.0, -8.0]), torch.tensor([-26.0, -30.0, -9.5]), torch.tensor([-22.0, -30.0, -8.0]), torch.tensor([-24.0, -33.0, -9.0]), 0.5)\n"
    "round(float(loss), 4)",
    # DPO trains through it: the gradient for the policy's log-probabilities
    "import torch\npc = torch.tensor([-20.0, -31.0, -8.0], requires_grad=True)\npr = torch.tensor([-26.0, -30.0, -9.5], requires_grad=True)\n"
    "dpo_loss(pc, pr, torch.tensor([-22.0, -30.0, -8.0]), torch.tensor([-24.0, -33.0, -9.0]), 0.5).backward()\n"
    "[round(float(v), 4) for v in list(pc.grad) + list(pr.grad)]",
]
_D_NO_REF = _D.replace("(policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)", "policy_chosen - policy_rejected")
_D_SWAPPED = _D.replace("(policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)", "(policy_rejected - ref_rejected) - (policy_chosen - ref_chosen)")
_D_PLUS = _D.replace("return -F.logsigmoid", "return F.logsigmoid")
_D_BETA_OUT = _D.replace("-F.logsigmoid(beta * margin).mean()", "-beta * F.logsigmoid(margin).mean()")
_D_NO_BETA = _D.replace("-F.logsigmoid(beta * margin).mean()", "-F.logsigmoid(margin).mean()")
_D_SUM = _D.replace(".mean()", ".sum()")
_D_NO_LOG = _D.replace("-F.logsigmoid(beta * margin).mean()", "(1 - torch.sigmoid(beta * margin)).mean()").replace(
    "import torch.nn.functional as F\n", "import torch\nimport torch.nn.functional as F\n")
_D_ITEM = _D.replace(".mean()\n", ".mean().item()\n")


def _no_backward(e: BaseException) -> bool:
    return isinstance(e, AttributeError) and "has no attribute 'backward'" in str(e)


def _in_check_lines(e: BaseException) -> bool:
    """The error comes from the check's own lines after the learner's function, not from inside it."""
    import traceback
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<your code>"]
    return bool(frames) and frames[-1].name == "<module>"


def _many_numbers(e: BaseException) -> bool:
    """float() of what the learner's function gave back, a tensor with more than one number."""
    return (isinstance(e, (RuntimeError, ValueError)) and _in_check_lines(e)
            and ("cannot be converted to Scalar" in str(e) or "only one element tensors" in str(e)))


def _no_graph(e: BaseException) -> bool:
    """.backward() on what the learner's function gave back, a tensor the gradient cannot go back through."""
    return isinstance(e, RuntimeError) and "does not require grad" in str(e) and _in_check_lines(e)


EXERCISES = {
    "answer-logprob": ScenarioBox(
        name="answer-logprob",
        save_as=("llm", "answer_logprob"),
        start=_A_START,
        solution=_A,
        scenarios=_A_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-shift", error=lambda e: isinstance(e, IndexError) and "out of bounds for dimension 0" in str(e),
                    en="Your function asks for a row of `logits` that is not there. The logits are those of `ids[:-1]`: one row fewer than `ids`, as nothing comes after the last token. Row `t - 1` holds what the model predicted for token `t`.",
                    nl="Je functie vraagt een rij van `logits` die er niet is. De logits zijn die van `ids[:-1]`: één rij minder dan `ids`, want na het laatste token komt niets. Rij `t - 1` bevat wat het model voorspelde voor token `t`."),
            Mistake("same-place", wrong=_A_SAME_PLACE,
                    en=_SCENARIO + ": your function takes, at every place, the probability of the token that stands there already. The logits at a place predict the *next* token: row `t - 1` is the prediction for token `t`.",
                    nl=_SCENARIO_NL + ": je functie neemt op elke plaats de kans van het token dat daar al staat. De logits op een plaats voorspellen het *volgende* token: rij `t - 1` is de voorspelling voor token `t`."),
            Mistake("prompt-too", wrong=_A_ALL,
                    en=_SCENARIO + ": your function adds the tokens of the instruction too. Only the answer counts, from token `start` on: the instruction is the same for both answers of a pair.",
                    nl=_SCENARIO_NL + ": je functie telt de tokens van de instructie ook mee. Alleen het antwoord telt, vanaf token `start`: de instructie is voor beide antwoorden van een paar dezelfde."),
            Mistake("late", wrong=_A_LATE,
                    en=_SCENARIO + ": your function skips the first token of the answer, token `start`.",
                    nl=_SCENARIO_NL + ": je functie slaat het eerste token van het antwoord over, token `start`."),
            Mistake("logits", wrong=_A_LOGITS,
                    en=_SCENARIO + ": your function adds logits. A logit is no log-probability: `torch.log_softmax(logits, dim=-1)` turns every row into log-probabilities, which add up to 1 as probabilities.",
                    nl=_SCENARIO_NL + ": je functie telt logits op. Een logit is geen log-kans: `torch.log_softmax(logits, dim=-1)` maakt van elke rij log-kansen, die als kansen optellen tot 1."),
            Mistake("probs", wrong=_A_PROBS,
                    en=_SCENARIO + ": your function adds probabilities. The probability of an answer is the *product* of those of its tokens, and its log is the sum of their logs: take `torch.log_softmax`.",
                    nl=_SCENARIO_NL + ": je functie telt kansen op. De kans van een antwoord is het *product* van die van zijn tokens, en de log ervan de som van hun logs: neem `torch.log_softmax`."),
            Mistake("mean", wrong=_A_MEAN,
                    en=_SCENARIO + ": your function gives the mean per token. The log-probability of the whole answer is the sum.",
                    nl=_SCENARIO_NL + ": je functie geeft het gemiddelde per token. De log-kans van het hele antwoord is de som."),
            Mistake("item", error=_no_backward,
                    en="Your function gives back a plain number, and DPO trains through it: `.backward()` needs the tensor. Give back `total` as it is, without `.item()` or `float()`.",
                    nl="Je functie geeft een los getal terug, en DPO traint erdoorheen: `.backward()` heeft de tensor nodig. Geef `total` terug zoals hij is, zonder `.item()` of `float()`."),
            Mistake("many", error=_many_numbers,
                    en="Your function gives back a tensor with more than one number, and `float(...)` needs one. The log-probability of an answer is one number: the sum over its tokens.",
                    nl="Je functie geeft een tensor met meer dan één getal terug, en `float(...)` heeft er één nodig. De log-kans van een antwoord is één getal: de som over zijn tokens."),
            Mistake("detached", error=_no_graph,
                    en="The gradient cannot go back through what your function gives back: it was made apart from `logits`, as with `.detach()`, `torch.tensor(...)` or `torch.no_grad()`. DPO trains through it: give back what you worked out from `logits`, as it is.",
                    nl="De gradient kan niet terug door wat je functie teruggeeft: het is los van `logits` gemaakt, zoals met `.detach()`, `torch.tensor(...)` of `torch.no_grad()`. DPO traint erdoorheen: geef terug wat je uit `logits` uitrekende, zoals het is."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the log-probabilities of the logits, and the one of every token of the answer added to `total`.",
                      "nl": "Vervang de twee `...`: de log-kansen van de logits, en die van elk token van het antwoord opgeteld bij `total`."},
            "wrong": {"en": _SCENARIO + ". For every token of the answer, from `start` on, the log-probability that the row before it gave it?",
                      "nl": _SCENARIO_NL + ". Voor elk token van het antwoord, vanaf `start`, de log-kans die de rij ervoor het gaf?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Another `start`, other logits, and a gradient that reaches the logits?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Een andere `start`, andere logits, en een gradient die de logits bereikt?"},
            "correct": {"en": "Your function gives the log-probability of the answer after its instruction: the sum, over its tokens, of the log-probability the model gave each one. It is in your project now, as `{saved}`.",
                        "nl": "Je functie geeft de log-kans van het antwoord na zijn instructie: de som, over zijn tokens, van de log-kans die het model elk ervan gaf. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `answer_logprob` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `answer_logprob` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "dpo-loss": ScenarioBox(
        name="dpo-loss",
        save_as=("llm", "dpo_loss"),
        start=_D_START,
        solution=_D,
        scenarios=_D_SCENARIOS,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("no-ref", wrong=_D_NO_REF,
                    en=_SCENARIO + ": your margin leaves out the reference model. DPO looks at how much the policy has *moved*: `policy_chosen - ref_chosen` for the chosen answer, and the same for the rejected one.",
                    nl=_SCENARIO_NL + ": je marge laat het reference model weg. DPO kijkt naar hoeveel de policy is *verschoven*: `policy_chosen - ref_chosen` voor het gekozen antwoord, en hetzelfde voor het afgewezen antwoord."),
            Mistake("swapped", wrong=_D_SWAPPED,
                    en=_SCENARIO + ": your loss gets small when the policy moves towards the *rejected* answer. The margin, or the sign in front of it, is the wrong way round: the margin starts with the chosen answer, what it gained minus what the rejected one gained, and the loss is small when the margin is big.",
                    nl=_SCENARIO_NL + ": je loss wordt klein als de policy naar het *afgewezen* antwoord schuift. De marge, of het teken ervoor, staat andersom: de marge begint met het gekozen antwoord, wat dat won min wat het afgewezen antwoord won, en de loss is klein als de marge groot is."),
            Mistake("plus", wrong=_D_PLUS,
                    en=_SCENARIO + ": your loss is below 0. The log of a probability is at most 0, so the loss is *minus* it: `-F.logsigmoid(...)`.",
                    nl=_SCENARIO_NL + ": je loss is kleiner dan 0. De log van een kans is hooguit 0, dus de loss is *min* die log: `-F.logsigmoid(...)`."),
            Mistake("beta-out", wrong=_D_BETA_OUT,
                    en=_SCENARIO + ": `beta` multiplies the loss, not the margin. It belongs inside: `F.logsigmoid(beta * margin)`, as the rewards in the Bradley-Terry probability are `beta` times the log-ratios.",
                    nl=_SCENARIO_NL + ": `beta` vermenigvuldigt de loss, niet de marge. Hij hoort erbinnen: `F.logsigmoid(beta * margin)`, want de rewards in de kans van Bradley-Terry zijn `beta` keer de log-verhoudingen."),
            Mistake("no-beta", wrong=_D_NO_BETA,
                    en=_SCENARIO + ": the loss leaves out `beta`. The rewards in the Bradley-Terry probability are `beta` times the log-ratios: `F.logsigmoid(beta * margin)`.",
                    nl=_SCENARIO_NL + ": de loss laat `beta` weg. De rewards in de kans van Bradley-Terry zijn `beta` keer de log-verhoudingen: `F.logsigmoid(beta * margin)`."),
            Mistake("sum", wrong=_D_SUM,
                    en=_SCENARIO + ": your loss is the sum over the pairs. Take the mean, `.mean()`, so that the loss of a batch does not grow with the number of pairs in it.",
                    nl=_SCENARIO_NL + ": je loss is de som over de paren. Neem het gemiddelde, `.mean()`, zodat de loss van een batch niet groeit met het aantal paren erin."),
            Mistake("no-log", wrong=_D_NO_LOG,
                    en=_SCENARIO + ": your loss is no log of the probability. The loss of a pair is minus the log of its Bradley-Terry probability, as in the fit of the Bradley-Terry model: `-F.logsigmoid(beta * margin)`.",
                    nl=_SCENARIO_NL + ": je loss is geen log van de kans. De loss van een paar is min de log van zijn kans volgens Bradley-Terry, net als bij het fitten van het Bradley-Terry model: `-F.logsigmoid(beta * margin)`."),
            Mistake("item", error=_no_backward,
                    en="Your function gives back a plain number, and DPO trains with it: `.backward()` needs the tensor. Give back the mean as it is, without `.item()` or `float()`.",
                    nl="Je functie geeft een los getal terug, en DPO traint ermee: `.backward()` heeft de tensor nodig. Geef het gemiddelde terug zoals het is, zonder `.item()` of `float()`."),
            Mistake("many", error=_many_numbers,
                    en="Your function gives back a tensor with more than one number, a loss for every pair perhaps, and `float(...)` needs one. The loss of a batch is one number: the mean over its pairs.",
                    nl="Je functie geeft een tensor met meer dan één getal terug, misschien een loss per paar, en `float(...)` heeft er één nodig. De loss van een batch is één getal: het gemiddelde over zijn paren."),
            Mistake("detached", error=_no_graph,
                    en="The gradient cannot go back through what your function gives back: it was made apart from the log-probabilities, as with `.detach()`, `torch.tensor(...)` or `torch.no_grad()`. DPO trains with it: give back the mean as it is.",
                    nl="De gradient kan niet terug door wat je functie teruggeeft: het is los van de log-kansen gemaakt, zoals met `.detach()`, `torch.tensor(...)` of `torch.no_grad()`. DPO traint ermee: geef het gemiddelde terug zoals het is."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the margin of the four log-probabilities, and the loss of it.",
                      "nl": "Vervang de twee `...`: de marge van de vier log-kansen, en de loss ervan."},
            "wrong": {"en": _SCENARIO + ". What the chosen answer gained on the reference model, minus what the rejected one gained, times `beta`, and then minus the log of its sigmoid, as a mean?",
                      "nl": _SCENARIO_NL + ". Wat het gekozen antwoord won op het reference model, min wat het afgewezen antwoord won, keer `beta`, en dan min de log van de sigmoid ervan, als gemiddelde?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. More pairs, another `beta`, and a gradient for every log-probability of the policy?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Meer paren, een andere `beta`, en een gradient voor elke log-kans van de policy?"},
            "correct": {"en": "Your function gives the DPO loss: small when the policy has moved towards the chosen answers and away from the rejected ones, compared with the reference model. It is in your project now, as `{saved}`.",
                        "nl": "Je functie geeft de loss van DPO: klein als de policy naar de gekozen antwoorden is geschoven en weg van de afgewezen antwoorden, vergeleken met het reference model. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `dpo_loss` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `dpo_loss` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("answer-logprob", _A_START, "empty", None),
    ("answer-logprob", _A, "correct", None),
    ("answer-logprob", _A_HEAD + "    logp = torch.log_softmax(logits, dim=-1)\n"
                       "    return logp[torch.arange(start - 1, len(ids) - 1), ids[start:]].sum()\n", "correct", None),
    ("answer-logprob", _A_HEAD + "    import torch.nn.functional as F\n    total = 0\n    for t in range(start, len(ids)):\n"
                       "        total = total - F.cross_entropy(logits[t - 1:t], torch.tensor([ids[t]]))\n    return total\n", "correct", None),
    ("answer-logprob", _A.replace("logp[t - 1, ids[t]]", "logp[t, ids[t]]"), "error", "no-shift"),
    ("answer-logprob", _A_SAME_PLACE, "wrong", "same-place"),
    ("answer-logprob", _A_ALL, "wrong", "prompt-too"),
    ("answer-logprob", _A_LATE, "wrong", "late"),
    ("answer-logprob", _A_LOGITS, "wrong", "logits"),
    ("answer-logprob", _A_PROBS, "wrong", "probs"),
    ("answer-logprob", _A_MEAN, "wrong", "mean"),
    ("answer-logprob", _A_ITEM, "error", "item"),
    ("answer-logprob", _A_HEAD + "    logp = torch.log_softmax(logits, dim=-1)\n"
                       "    return logp[torch.arange(start - 1, len(ids) - 1), ids[start:]]\n", "error", "many"),
    ("answer-logprob", _A.replace("    return total\n", "    return total.detach()\n"), "error", "detached"),
    ("dpo-loss", _D_START, "empty", None),
    ("dpo-loss", _D, "correct", None),
    ("dpo-loss", "import torch\n\n\ndef dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):\n"
                 "    chosen = beta * (policy_chosen - ref_chosen)\n    rejected = beta * (policy_rejected - ref_rejected)\n"
                 "    return -torch.log(torch.sigmoid(chosen - rejected)).mean()\n", "correct", None),
    ("dpo-loss", _D_HEAD + "    margin = policy_chosen - policy_rejected - ref_chosen + ref_rejected\n"
                 "    return F.softplus(-beta * margin).mean()\n", "correct", None),
    ("dpo-loss", _D_NO_REF, "wrong", "no-ref"),
    ("dpo-loss", _D_SWAPPED, "wrong", "swapped"),
    ("dpo-loss", _D_PLUS, "wrong", "plus"),
    ("dpo-loss", _D_BETA_OUT, "wrong", "beta-out"),
    ("dpo-loss", _D_NO_BETA, "wrong", "no-beta"),
    ("dpo-loss", _D_SUM, "wrong", "sum"),
    ("dpo-loss", _D_NO_LOG, "wrong", "no-log"),
    ("dpo-loss", _D_ITEM, "error", "item"),
    ("dpo-loss", _D.replace(".mean()", ""), "error", "many"),
    ("dpo-loss", _D.replace(".mean()\n", ".mean().detach()\n"), "error", "detached"),
    ("dpo-loss", _D.replace("-F.logsigmoid(beta * margin)", "-F.logsigmoid(-beta * margin)"), "wrong", "swapped"),
]
