# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-rlvr: advantages, the reward of every answer of a group minus the group's mean, as GRPO
uses it; and policy_loss, the loss whose step makes answers with a positive advantage more likely and the others less.
The demo trains the learner's model of llm-dpo (or the course's) with GRPO on instructions to write a story without a
word, with a program as its reward, and measures it before and after on instructions it never trained on: without a
word, and with words, the kind of instruction DPO trained it on."""
import importlib.util
import math
import random
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _checks("llm-eval-chat", "everything_ai_checks_llm_eval_chat_for_rlvr")   # its program checks, as the course's code
D, T, P = E.D, E.T, E.T.P
BUILDS_ON = D.BUILDS_ON
projects.builds_on(*BUILDS_ON)

G = 8                     # answers per instruction: one group
PROMPTS = 4               # instructions per step
LR = 2e-3                 # AdamW on the adapters
STEPS = {"cpu": 10, "gpu": 25}   # steps of the demo: with its two measures, on a CPU about seven minutes with 2 threads
GUARD = 3                 # sentences a story needs at least, so that an empty or one-line story earns nothing
MEASURE = 20              # instructions of each kind the demo measures on, before and after
_AVOIDS = run_code(E._A).names["avoids"]
_SENTENCES = run_code(E._S).names["sentences"]


def number(x: float, digits: int, lang: str) -> str:
    return P.number(x, digits, lang)


def prompt(word: str) -> str:
    """The instruction to write a story without `word`, in the course's template."""
    return f"Write a story without the word: {word}.\nStory:\n"


def excerpt(story: str, n: int) -> str:
    """A story on one line, for a list: its first `n` characters, with … only when there is more."""
    flat = " ".join(story.split())
    return flat if len(flat) <= n else flat[:n].rstrip() + "…"


def held_out() -> list[str]:
    """The words of the instructions the demo measures on: those of llm-eval-chat's prompts without a word, which no
    model of the course trained on."""
    return [p[1][0] for p in E.prompts() if p[0] == "without"]


def train_words() -> list[str]:
    """The words GRPO trains on: those of llm-sft's list that are not held out."""
    held = set(held_out())
    return [w for w in sorted(T.WORDS) if w not in held]


def verify(story: str, word: str) -> float:
    """The reward, by a program: 1 when the story leaves `word` out and has at least GUARD sentences, else 0."""
    return float(_AVOIDS(story, word) and _SENTENCES(story) >= GUARD)


# --- the course's run: GRPO on the course's model of llm-dpo, 25 steps, on the CPU (about 15 minutes with 2 threads);
# measured on all 50 held-out instructions without a word, and the 100 test instructions with words of llm-sft ---
COURSE_RUN = {
    "rewards": [0.47, 0.12, 0.31, 0.34, 0.16, 0.53, 0.44, 0.59, 0.59, 0.41, 0.66, 0.44, 0.78, 0.56, 0.44, 0.56, 0.53, 0.72,
                0.78, 0.94, 0.69, 0.88, 0.66, 0.88, 0.91],
    "without": {"before": 18, "after": 41, "total": 50},
    "with": {"before": 60, "after": 28, "total": 100},
}
# An older, simpler way to learn from the program: keep the answers that pass (4 to each of the 91 training
# words) and fine-tune on them, as STaR and expert iteration do; 100 steps of SFT on the adapters, measured as above.
COURSE_KEEP = {"kept": 125, "tried": 364, "without": {"before": 18, "after": 16, "total": 50}}
# Two remedies for what the run took down, 25 steps each: a KL penalty with beta 0.2 to the model before the run, and
# half of every step's instructions with words, rewarded by words_used.
COURSE_REMEDIES = {"kl": {"without": 18, "with": 47}, "mix": {"without": 28, "with": 39}}
# --- end of the course's run ---


def _logprobs(model, prompt_ids: list, answer_ids: list):
    """The log-probability of every token of an answer after its prompt, with gradient."""
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    x = torch.tensor([prompt_ids + answer_ids[:-1]], device=device)
    with P._on(device.type):
        logits = model(x)[0, len(prompt_ids) - 1:]
    return F.log_softmax(logits.float(), dim=-1).gather(1, torch.tensor(answer_ids, device=device)[:, None]).squeeze(1)


def _floats(values, n: int) -> list | None:
    try:
        out = [float(v) for v in values]
    except Exception:
        return None
    return out if len(out) == n and all(math.isfinite(v) for v in out) else None


def measure(model, merges: dict, encode_fast, decode, words_used, show=None) -> dict:
    """How many of the first MEASURE held-out instructions without a word the model's story leaves the word out of,
    and how many of the first MEASURE test instructions with words of llm-sft it uses every word in."""
    end = 256 + len(merges)
    out = {"without": 0, "with": 0, "total": MEASURE}
    for k, w in enumerate(held_out()[:MEASURE]):
        ids, _ = T.answer(model, encode_fast(prompt(w), merges), end, seed=k)
        out["without"] += _AVOIDS(decode(ids, merges), w)
        if show:
            show({"phase": "measure", "done": k + 1, "total": 2 * MEASURE})
    for k, words in enumerate(T.tests()[:MEASURE]):
        ids, _ = T.answer(model, encode_fast(T.instruction(list(words)), merges), end, seed=k)
        out["with"] += bool(words_used(decode(ids, merges), list(words)))
        if show:
            show({"phase": "measure", "done": MEASURE + k + 1, "total": 2 * MEASURE})
    return out


def run(advantages, policy_loss, show, steps: int | None = None) -> dict | str:
    """The demo: the learner's model of llm-dpo (or the course's), measured, then trained with GRPO for `steps` steps
    (STEPS of the device when None) with the learner's advantages and policy_loss, then measured again. A string when
    it stops: "run" (a module of the kit), "advantages" or "loss" (the learner's function), "memory"."""
    import torch
    begin = time.perf_counter()   # the seconds of the whole demo, both measures too
    try:
        model, info, where = D.load()
        t = run_code("from llm.fast_encode import encode_fast\nfrom llm.decode import decode\nfrom llm.words_used import words_used").names
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
        end = 256 + len(merges)
        device = P.where()
        model.to(device["device"])
        before = measure(model, merges, t["encode_fast"], t["decode"], t["words_used"], show)
    except Exception:
        return "run"
    steps = STEPS[device["kind"]] if steps is None else steps
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    rng = random.Random(0)
    rewards, samples, start = [], [], time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        got = []
        for i, w in enumerate(rng.sample(train_words(), PROMPTS)):
            p = t["encode_fast"](prompt(w), merges)
            outs = []
            for g in range(G):
                ids, ended = T.answer(model, p, end, seed=step * 1000 + i * 50 + g)
                outs.append(ids + ([end] if ended else []))
            r = [verify(t["decode"]([j for j in ids if j != end], merges), w) for ids in outs]
            got += r
            try:
                adv = _floats(advantages(list(r)), G)
            except Exception:
                adv = None
            if adv is None:
                return "advantages"
            if step == steps - 1 and i == 0:
                samples = [(w, t["decode"]([j for j in ids if j != end], merges), rr, a) for ids, rr, a in zip(outs[:3], r, adv)]
            if not any(adv):
                continue
            model.train()
            try:
                lps = [_logprobs(model, p, ids) for ids in outs if ids]
                kept = [a for ids, a in zip(outs, adv) if ids]
                loss = policy_loss(lps, kept)
                if not isinstance(loss, torch.Tensor) or loss.dim() != 0 or not torch.isfinite(loss) or not loss.requires_grad:
                    return "loss"
                (loss / PROMPTS).backward()
            except Exception as e:
                return "memory" if "out of memory" in str(e).lower() else "loss"
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        model.eval()
        rewards.append(sum(got) / len(got))
        show({"phase": "train", "step": step + 1, "steps": steps, "reward": rewards[-1], "seconds": time.perf_counter() - start})
    try:
        after = measure(model, merges, t["encode_fast"], t["decode"], t["words_used"], show)
    except Exception:
        return "run"
    return {"source": where, "device": device["name"], "steps": steps, "rewards": rewards, "before": before, "after": after,
            "samples": samples, "seconds": time.perf_counter() - begin}


def rewards_picture(rewards: list, labels: dict, course: list | None = None) -> str:
    """The mean reward of every step, as a line in ink: a reward is what the run aims for, not a mistake it counts. With
    `course`, the course's run of 25 steps as a dashed grey line under it."""
    from everything_ai import chart
    n = max(len(rewards), len(course or []))
    p = chart.Panel((0.5, n + 0.5), (0, 1), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    yticks=[0, 0.25, 0.5, 0.75, 1])
    if course:
        p.line(list(range(1, len(course) + 1)), course, "ink-3", 1.4, dashed=True)
        p.text(n, 0.06, labels["course"], "ink-2", "end")   # bottom right, away from both lines
    if rewards:
        p.line(list(range(1, len(rewards) + 1)), rewards, "ink", 2.0)
        p.points(list(range(1, len(rewards) + 1)), rewards, "ink", r=3)
    return chart.figure([p], alt=labels["alt"], width=360, height=240)


# --- exercise 1: advantages ----------------------------------------------------------------------------------------------

_A_HEAD = ("def advantages(rewards):\n"
           "    # the rewards of the answers of one group: how much better each did than the group's mean\n")
_A_CODE = _A_HEAD + "    mean = sum(rewards) / len(rewards)\n    return [r - mean for r in rewards]\n"
_A_START = _A_HEAD + "    mean = ...\n    return ...\n"
_A_WRONG = {
    "raw": _A_HEAD + "    return [float(r) for r in rewards]\n",
    "max": _A_HEAD + "    best = max(rewards)\n    return [r - best for r in rewards]\n",
    "sign": _A_HEAD + "    mean = sum(rewards) / len(rewards)\n    return [mean - r for r in rewards]\n",
    "spread": _A_HEAD + "    mean = sum(rewards) / len(rewards)\n    spread = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5\n"
                        "    return [(r - mean) / spread for r in rewards]\n",
}


def _rounded(v) -> str:
    """Advantages for the messages, their numbers rounded to 4 decimals: 0.5774, not 0.5773502691896258."""
    try:
        values = [float(x) for x in v]
    except (TypeError, ValueError):
        return repr(v)
    return "[" + ", ".join(repr(round(x, 4) + 0.0) for x in values) + "]"


def _divided_by_spread(got, text) -> bool:
    """Every advantage of the group of 4 divided by the spread of its rewards, as full GRPO does: the spread over n
    (as written out, or statistics.pstdev) or over n - 1 (statistics.stdev, torch.std), with or without a small eps."""
    try:
        values = [float(x) for x in got]
    except (TypeError, ValueError):
        return False
    r = [1.0, 1.0, 0.0, 1.0]
    mean = sum(r) / len(r)
    for n in (len(r), len(r) - 1):
        spread = (sum((x - mean) ** 2 for x in r) / n) ** 0.5
        if len(values) == len(r) and all(abs(a - (x - mean) / spread) < 1e-3 for a, x in zip(values, r)):
            return True
    return False

# --- exercise 2: policy_loss ---------------------------------------------------------------------------------------------

_L_HEAD = ("import torch\n\n\n"
           "def policy_loss(logprobs, advantages):\n"
           "    # logprobs: for every answer of a group, the log-probabilities of its tokens (a tensor)\n"
           "    # advantages: for every answer, its advantage\n"
           "    # the loss whose step makes answers with a positive advantage more likely, and the others less:\n"
           "    # minus the mean, over the answers, of advantage times the mean log-probability of the tokens\n")
_L_CODE = (_L_HEAD + "    terms = []\n    for i in range(len(logprobs)):\n        terms.append(advantages[i] * logprobs[i].mean())\n"
           "    return -torch.stack(terms).mean()\n")
_L_START = (_L_HEAD + "    terms = []\n    for i in range(len(logprobs)):\n"
            "        # the advantage times the mean log-probability of this answer's tokens\n        ...\n"
            "    # minus their mean\n    return ...\n")
_L_WRONG = {
    "plus": _L_CODE.replace("    return -torch.stack(terms).mean()\n", "    return torch.stack(terms).mean()\n"),
    "sum": _L_CODE.replace("logprobs[i].mean()", "logprobs[i].sum()"),
    "pooled": (_L_HEAD + "    weighted = torch.cat([a * lp for lp, a in zip(logprobs, advantages)])\n    return -weighted.mean()\n"),
    "no-advantage": _L_CODE.replace("terms.append(advantages[i] * logprobs[i].mean())", "terms.append(logprobs[i].mean())"),
}
_LP = "import torch\nlogprobs = [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5, -0.5, -1.0, -2.0])]\n"
_L_SCENARIOS = [
    # the lesson's example: two answers, one better than the group and one worse
    _LP + "loss = policy_loss(logprobs, [0.5, -0.5])\nround(float(loss), 4)",
    # answers of different lengths count the same: the mean over each answer's tokens
    _LP + "loss = policy_loss(logprobs, [1.0, 1.0])\nround(float(loss), 4)",
    # a group of three: every answer counts, and the mean is over all of them
    "import torch\nlogprobs = [torch.tensor([-1.0]), torch.tensor([-2.0, -2.0]), torch.tensor([-0.5, -1.5, -1.0])]\n"
    "loss = policy_loss(logprobs, [1.0, -0.5, -0.5])\nround(float(loss), 4)",
    # a step lowers the loss by making the better answer more likely: its gradient
    "import torch\nlp = torch.tensor([-1.0, -2.0], requires_grad=True)\nbad = torch.tensor([-1.5, -1.5], requires_grad=True)\n"
    "policy_loss([lp, bad], [0.5, -0.5]).backward()\n[round(float(v), 4) for v in lp.grad]",
    # an advantage of zero adds nothing
    _LP + "loss = policy_loss(logprobs, [0.0, 0.0])\nround(float(loss), 4)",
]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip().endswith("...") for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_CALL = ("`{call}` gives `{got}`, and it should give `{want}`", "`{call}` geeft `{got}`, en dat moet `{want}` zijn")
_SAY = ("With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`",
        "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn")

EXERCISES = {
    "advantages": FunctionBox(
        name="advantages",
        function="advantages",
        start=_A_START,
        solution=_A_CODE,
        visible=[([1.0, 1.0, 0.0, 1.0],)],   # _divided_by_spread works it out for this group
        hidden=[([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],), ([1.0, 1.0, 1.0],), ([0.0, 0.0],), ([0.25, 0.75, 0.5, 0.5],)],
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("raw", wrong=_A_WRONG["raw"],
                    en=_CALL[0] + ": you give back the rewards themselves. Then every answer that passes is pushed up, also when every answer of the group passes, and those that fail are left as they were. Take the mean of the group off every reward: what counts is how much better an answer did than the others to the same instruction.",
                    nl=_CALL[1] + ": je geeft de rewards zelf terug. Dan wordt elk antwoord dat slaagt omhooggeduwd, ook als elk antwoord van de groep slaagt, en blijven de antwoorden die zakken zoals ze waren. Trek het gemiddelde van de groep van elke reward af: wat telt, is hoeveel beter een antwoord het deed dan de andere op dezelfde instructie."),
            Mistake("max", wrong=_A_WRONG["max"],
                    en=_CALL[0] + ": you take off the best reward of the group, so no answer gets a positive advantage, and the best ones get 0: none is pushed up. Take off the mean of the group.",
                    nl=_CALL[1] + ": je trekt de beste reward van de groep af, dus geen antwoord krijgt een positieve advantage, en de beste krijgen 0: geen enkel wordt omhooggeduwd. Trek het gemiddelde van de groep af."),
            Mistake("sign", wrong=_A_WRONG["sign"],
                    en=_CALL[0] + ": your advantages have the wrong sign: the answers that did better than the group get a negative one. It is the reward minus the mean: `r - mean`.",
                    nl=_CALL[1] + ": je advantages hebben het verkeerde teken: de antwoorden die het beter deden dan de groep krijgen een negatief. Het is de reward min het gemiddelde: `r - mean`."),
            Mistake("spread", matches=_divided_by_spread,
                    en=_CALL[0] + ": you divide by the spread of the group's rewards. Full GRPO does that too, but here an advantage is just the reward minus the group's mean, as in the formula of *The idea*: the run below trains with that. Leave the division out.",
                    nl=_CALL[1] + ": je deelt door de spreiding van de rewards van de groep. Volledige GRPO doet dat ook, maar hier is een advantage gewoon de reward min het gemiddelde van de groep, zoals in de formule van *Het idee*: de run hieronder traint daarmee. Laat het delen weg."),
        ],
        texts={
            **E._TEXTS,
            "empty": {"en": "Replace the two `...`: the mean of the rewards, and every reward minus it.",
                      "nl": "Vervang de twee `...`: het gemiddelde van de rewards, en elke reward min dat gemiddelde."},
            "wrong": {"en": _CALL[0] + ". Does every answer get its reward minus the mean reward of its group?",
                      "nl": _CALL[1] + ". Krijgt elk antwoord zijn reward min de gemiddelde reward van zijn groep?"},
            "almost": {"en": "Right for the group of 4. But " + _CALL[0] + ". Does it work for a group of another size, and for a group whose answers all get the same reward?",
                       "nl": "Goed voor de groep van 4. Maar " + _CALL[1] + ". Werkt hij voor een groep van een andere grootte, en voor een groep waarvan alle antwoorden dezelfde reward krijgen?"},
            "correct": {"en": "Your `advantages` says how much better every answer did than its group. A group whose answers all pass, or all fail, gets advantages of 0: it teaches nothing.",
                        "nl": "Je `advantages` zegt hoeveel beter elk antwoord het deed dan zijn groep. Een groep waarvan alle antwoorden slagen, of alle zakken, krijgt advantages van 0: daar leert het model niets van."},
        },
    ),
    "policy-loss": ScenarioBox(
        name="policy-loss",
        start=_L_START,
        solution=_L_CODE,
        scenarios=_L_SCENARIOS,
        digits=4,
        first=[_DOTS],
        mistakes=[
            Mistake("plus", wrong=_L_WRONG["plus"],
                    en=_SAY[0] + ": your loss has the wrong sign. A step lowers the loss; to make answers with a positive advantage more likely, the loss is minus the mean of advantage times log-probability.",
                    nl=_SAY[1] + ": je loss heeft het verkeerde teken. Een stap verlaagt de loss; om antwoorden met een positieve advantage waarschijnlijker te maken, is de loss min het gemiddelde van advantage maal log-kans."),
            Mistake("sum", wrong=_L_WRONG["sum"],
                    en=_SAY[0] + ": you add up the log-probabilities of an answer's tokens. Then a long answer weighs more than a short one with the same advantage, and the run learns to make answers longer or shorter. Take their mean: `logprobs[i].mean()`.",
                    nl=_SAY[1] + ": je telt de log-kansen van de tokens van een antwoord op. Dan weegt een lang antwoord zwaarder dan een kort met dezelfde advantage, en leert de run antwoorden langer of korter te maken. Neem hun gemiddelde: `logprobs[i].mean()`."),
            Mistake("pooled", wrong=_L_WRONG["pooled"],
                    en=_SAY[0] + ": you take the mean over all tokens of all answers at once, so a long answer counts as many times as it has tokens. Take the mean of each answer's tokens first, then the mean over the answers.",
                    nl=_SAY[1] + ": je neemt het gemiddelde over alle tokens van alle antwoorden tegelijk, dus een lang antwoord telt zo vaak mee als het tokens heeft. Neem eerst het gemiddelde van de tokens van elk antwoord, dan het gemiddelde over de antwoorden."),
            Mistake("no-graph", error=lambda e: isinstance(e, RuntimeError) and "does not require grad" in str(e)
                    or isinstance(e, AttributeError) and "has no attribute 'backward'" in str(e),
                    en="With the lines of the check after your code, Python stops at `.backward()`: nothing in your loss leads back to the log-probabilities, so it has no gradient and a step could not change the model. Keep every term a tensor, and make one tensor of them with `torch.stack(terms)`, not with `torch.tensor`, `float` or `.item()`.",
                    nl="Met de regels van de check na je code stopt Python bij `.backward()`: niets in je loss leidt terug naar de log-kansen, dus heeft hij geen gradient en kan een stap het model niet veranderen. Houd elke term een tensor, en maak er één tensor van met `torch.stack(terms)`, niet met `torch.tensor`, `float` of `.item()`."),
            Mistake("no-advantage", wrong=_L_WRONG["no-advantage"],
                    en=_SAY[0] + ": your loss leaves out the advantages, so every answer is made more likely, the good and the bad: that is SFT on the model's own answers. Multiply each answer's mean log-probability by its advantage.",
                    nl=_SAY[1] + ": je loss laat de advantages weg, dus wordt elk antwoord waarschijnlijker gemaakt, het goede en het slechte: dat is SFT op de eigen antwoorden van het model. Vermenigvuldig de gemiddelde log-kans van elk antwoord met zijn advantage."),
        ],
        texts={
            **E._TEXTS,
            "empty": {"en": "Replace the two `...`: for every answer, its advantage times the mean of its log-probabilities; then minus the mean of those.",
                      "nl": "Vervang de twee `...`: voor elk antwoord zijn advantage maal het gemiddelde van zijn log-kansen; dan min het gemiddelde daarvan."},
            "wrong": {"en": _SAY[0] + ". Is the loss minus the mean, over the answers, of advantage times the mean log-probability of the answer's tokens?",
                      "nl": _SAY[1] + ". Is de loss min het gemiddelde, over de antwoorden, van advantage maal de gemiddelde log-kans van de tokens van het antwoord?"},
            "almost": {"en": "Right for the first example, with advantages 0.5 and -0.5. But " + _SAY[0][0].lower() + _SAY[0][1:] + ". Does it work for a group of another size, do answers of different lengths count the same, and does the gradient push the better answer up?",
                       "nl": "Goed voor het eerste voorbeeld, met advantages 0,5 en -0,5. Maar " + _SAY[1][0].lower() + _SAY[1][1:] + ". Werkt hij voor een groep van een andere grootte, tellen antwoorden van verschillende lengte even zwaar, en duwt de gradient het betere antwoord omhoog?"},
            "correct": {"en": "Your `policy_loss` pushes every answer up or down by its advantage, with every answer counting the same whatever its length. Below, it trains your model.",
                        "nl": "Je `policy_loss` duwt elk antwoord omhoog of omlaag naar zijn advantage, en elk antwoord telt even zwaar, hoe lang het ook is. Hieronder traint hij je model."},
        },
    ),
}

# The lesson is done when these are correct. "hack" is a choice question, "trade" the open question at the end.
REQUIRED = ["advantages", "policy-loss", "hack", "trade"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
CASES = [
    ("advantages", _A_START, "empty", None),
    ("advantages", _A_CODE, "correct", None),
    ("advantages", _A_WRONG["raw"], "wrong", "raw"),
    ("advantages", _A_WRONG["max"], "wrong", "max"),
    ("advantages", _A_WRONG["sign"], "wrong", "sign"),
    ("advantages", _A_WRONG["spread"], "wrong", "spread"),
    ("advantages", _A_WRONG["spread"].replace("/ spread for", "/ (spread + 1e-4) for"), "wrong", "spread"),
    ("advantages", _A_HEAD + "    m = sum(rewards) / len(rewards)\n    out = []\n    for r in rewards:\n        out.append(r - m)\n    return out\n", "correct", None),
    ("policy-loss", _L_START, "empty", None),
    ("policy-loss", _L_CODE, "correct", None),
    ("policy-loss", _L_WRONG["plus"], "wrong", "plus"),
    ("policy-loss", _L_WRONG["sum"], "wrong", "sum"),
    ("policy-loss", _L_WRONG["pooled"], "wrong", "pooled"),
    ("policy-loss", _L_WRONG["no-advantage"], "wrong", "no-advantage"),
    ("policy-loss", _L_HEAD + "    return -sum(a * lp.mean() for lp, a in zip(logprobs, advantages)) / len(logprobs)\n", "correct", None),
    ("policy-loss", _L_CODE.replace("torch.stack(terms).mean()", "torch.stack(terms).sum() / 2"), "almost", None),
    ("policy-loss", _L_CODE.replace("-torch.stack(terms)", "-torch.tensor(terms)"), "error", "no-graph"),
    ("policy-loss", _L_CODE.replace("return -torch.stack(terms).mean()", "return -float(torch.stack(terms).mean())"), "error", "no-graph"),
    ("policy-loss", _L_CODE.replace("logprobs[i].mean())", "logprobs[i].mean().detach())"), "error", "no-graph"),
]
