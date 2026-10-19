# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson llm-dpo: dpo_pairs, the preference pairs a checker gives among the answers to each
instruction; reference_logprob, the log-probability of an answer under the reference model, the policy with its LoRA
adapters switched off; dpo_step, one step of DPO on a batch of pairs; and a choice. The demo puts adapters on the
learner's SFT model of llm-sft (or the course's), makes preference pairs from the answers the course's SFT model wrote
to 1,000 instructions (data/course-dpo-answers.json) with the learner's words_used and make_pairs, trains the
adapters with DPO for one pass over the pairs, and lets the model answer the test instructions of llm-sft again."""
import importlib.util
import json
import math
import random
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code, same_value


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LO = _checks("llm-lora", "everything_ai_checks_llm_lora_for_dpo")
PP = _checks("llm-preference-pairs", "everything_ai_checks_llm_preference_pairs_for_dpo")
D = _checks("llm-dpo-loss", "everything_ai_checks_llm_dpo_loss_for_dpo")
T, S = LO.T, LO.S
SAVES = (("llm", "dpo_pairs", "llm-dpo", "dpo-pairs"), ("llm", "reference_logprob", "llm-dpo", "reference-logprob"),
         ("llm", "dpo_step", "llm-dpo", "dpo-step"))
BUILDS_ON = LO.BUILDS_ON + PP.SAVES + D.SAVES + SAVES   # dpo_pairs imports preference_pairs, dpo_step the rest
projects.builds_on(*BUILDS_ON)

ANSWERS = "data/course-dpo-answers.json"   # the course's SFT model's answers, K to an instruction
RANK, ALPHA = 8, 16    # the adapters, as the course measured LoRA
BETA = 0.1             # the weight of the KL penalty, as in the DPO paper
LR = 1e-3              # AdamW on the adapters
BATCH = 8              # pairs per step
MOVED = 100            # the pairs on which the demo measures how far the answers moved: the first ones
NAME = "dpo"           # the adapters in work/projects/llm/: dpo.pt, with dpo.json
COURSE = "data/course-dpo"   # the course's adapters on its own SFT model, .pt and .json


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def items() -> list:
    """The instructions of data/course-dpo-answers.json: for each its requested words, the answers of the course's SFT
    model, and whether each answer stopped by itself."""
    return json.loads((paths.ROOT / ANSWERS).read_text(encoding="utf-8"))["items"]


def tools() -> dict | str:
    """The learner's modules the demo needs, or "run" when one stops with an error."""
    ran = run_code("from llm.fast_encode import encode_fast\nfrom llm.decode import decode\nfrom llm.words_used import words_used\n"
                   "from llm.add_lora import add_lora\nfrom llm.dpo_pairs import dpo_pairs\nfrom llm.dpo_step import dpo_step\n"
                   "from llm.answer_logprob import answer_logprob\nfrom llm.reference_logprob import reference_logprob")
    return "run" if ran.error is not None else ran.names


def _pairs_ok(pairs) -> bool:
    return isinstance(pairs, list) and all(isinstance(p, (tuple, list)) and len(p) == 3 and isinstance(p[0], (tuple, list))
                                            and isinstance(p[1], str) and isinstance(p[2], str) for p in pairs)


def examples(pairs: list, merges: dict, encode_fast, ended: dict) -> list:
    """(chosen ids, rejected ids, start) for every pair: the instruction and the answer, with an end-of-text after
    an answer that stopped by itself; the answer starts after the instruction."""
    end = 256 + len(merges)
    out = []
    for words, chosen, rejected in pairs:
        prompt = encode_fast(T.instruction(list(words)), merges)
        c = prompt + encode_fast(chosen, merges) + ([end] if ended.get(chosen, True) else [])
        r = prompt + encode_fast(rejected, merges) + ([end] if ended.get(rejected, True) else [])
        out.append((c, r, len(prompt)))
    return out


def before(source: str) -> dict:
    """What the SFT model did on the test instructions before DPO: the learner's run of llm-sft when DPO starts from
    their model, else the course's model's own."""
    info = T.saved() if source == "yours" else T.course_info()
    return {"used": info["after"]["used"], "total": info["after"]["total"], "hits": info["hits"]["after"]}


def run(show) -> dict | str:
    """The demo: adapters of RANK on the learner's SFT model (or the course's), preference pairs from the course's
    answers by the learner's dpo_pairs, one pass of the learner's dpo_step over them in a shuffled order, and the
    answers to the test instructions after. The adapters are kept in work/projects/llm/dpo.pt. A string when it stops:
    "run", "pairs", "lora", "step", "memory" or "diverged"."""
    import torch
    t = tools()
    if t == "run":
        return "run"
    try:
        model, info, source = T.load()
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
        made = items()
    except Exception:
        return "run"
    try:
        pairs = t["dpo_pairs"](made)
    except Exception:
        return "pairs"
    if not _pairs_ok(pairs) or not pairs:
        return "pairs"
    show({"phase": "pairs", "pairs": len(pairs), "instructions": len(made)})   # before the ids: encoding them takes a while
    ended = {a: e for it in made for a, e in zip(it["answers"], it["ended"])}
    data = examples(pairs, merges, t["encode_fast"], ended)
    try:
        torch.manual_seed(0)   # the same random A on every run
        layers = t["add_lora"](model, RANK, ALPHA)
    except Exception:
        return "lora"
    w = T.P.where()
    model.to(w["device"]).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    order = list(range(len(data)))
    random.Random(0).shuffle(order)
    steps = math.ceil(len(data) / BATCH)
    losses, start, last = [], time.perf_counter(), 0.0
    try:
        for step in range(steps):
            batch = [data[i] for i in order[step * BATCH:(step + 1) * BATCH]]
            with T.P._on(w["device"]), torch.device(w["device"]):   # the tensors the learner makes land on the device too
                loss = t["dpo_step"](model, layers, opt, batch, BETA)
            if not isinstance(loss, (int, float)) or isinstance(loss, bool):
                return "step"
            if not math.isfinite(loss):
                return "diverged"
            losses.append(loss)
            now = time.perf_counter()
            if now - last > 1 or step == steps - 1:
                last = now
                show({"phase": "steps", "step": step + 1, "steps": steps, "loss": sum(losses[-25:]) / len(losses[-25:]),
                      "seconds": now - start})
    except torch.OutOfMemoryError:
        return "memory"
    except Exception:
        return "step"
    seconds = time.perf_counter() - start
    model.eval()
    try:   # how far the chosen and the rejected answers moved from the reference model, in nats, on average
        moved = {"chosen": 0.0, "rejected": 0.0}
        probe = data[:MOVED]
        with torch.no_grad(), torch.device(w["device"]):
            for c, r, s in probe:
                for key, ids in (("chosen", c), ("rejected", r)):
                    policy = float(t["answer_logprob"](model(torch.tensor([ids[:-1]]))[0], ids, s))
                    moved[key] += (policy - float(t["reference_logprob"](model, layers, ids, s))) / len(probe)
    except Exception:
        return "step"
    after = T.rate(model, merges, t["words_used"], t["encode_fast"], t["decode"], show, "after")
    if isinstance(after, str):
        return "run"
    result = {"model": model, "source": source, "sft_sha256": sft_sha256(source), "pairs": len(data), "instructions": len(made),
              "steps": steps, "moved": moved,
              "seconds": seconds, "device": w["name"], "losses": losses, "after": after, "before": before(source),
              "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    save(result)
    return result


def save(result: dict) -> None:
    """The adapters in work/projects/llm/dpo.pt, and the run in dpo.json: also, for every test instruction in order, 1
    when the answer used every word and 0 when not."""
    import torch
    folder = projects.folder() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    adapters = {k: v.detach().cpu() for k, v in result["model"].state_dict().items() if k.endswith((".A", ".B"))}
    torch.save(adapters, folder / f"{NAME}.pt")
    (folder / f"{NAME}.json").write_text(json.dumps(keep(result), indent=1) + "\n", encoding="utf-8")


def sft_sha256(source: str) -> str:
    """The SHA-256 of the SFT model the adapters sit on: the learner's work/projects/llm/sft.pt, or data/course-sft.pt."""
    import hashlib
    path = projects.folder() / "llm" / f"{T.NAME}.pt" if source == "yours" else (paths.ROOT / T.COURSE).with_suffix(".pt")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(x: float, digits: int, lang: str) -> str:
    """A number with its sign, also +: how far something moved."""
    return ("+" if x >= 0 else "") + number(x, digits, lang)


def keep(result: dict) -> dict:
    """What dpo.json and data/course-dpo.json hold of a run."""
    every = max(1, math.ceil(len(result["losses"]) / 50))   # at most 50 of them: every `every`th step
    return {"base": result["source"], "sft_sha256": result["sft_sha256"], "rank": RANK, "alpha": ALPHA, "beta": BETA, "lr": LR, "batch": BATCH,
            "pairs": result["pairs"], "instructions": result["instructions"], "steps": result["steps"],
            "moved": {k: round(v, 2) for k, v in result["moved"].items()},
            "trainable": result["trainable"], "seconds": round(result["seconds"], 1), "device": result["device"],
            "losses": [round(v, 4) for v in result["losses"][::every]], "every": every,
            "after": {k: result["after"][k] for k in ("used", "stopped", "total")},
            "before": {k: result["before"][k] for k in ("used", "total")},
            "hits": "".join("1" if h else "0" for h in result["after"]["hits"]),
            "example": example(result["before"]["hits"], result["after"])}


def example(before_hits: str, after: dict) -> dict:
    """The first test instruction that the model did after DPO and not before, with its answer; the first one when
    there is none."""
    k = next((k for k, (b, a) in enumerate(zip(before_hits, after["hits"])) if a and b == "0"), None)
    better = k is not None
    k = k if better else 0
    return {"words": after["words"][k], "after": after["answers"][k], "better": better}


def saved() -> dict | None:
    folder = projects.folder() / "llm"
    try:
        info = json.loads((folder / f"{NAME}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if (folder / f"{NAME}.pt").is_file() else None


def course_info() -> dict:
    return json.loads((paths.ROOT / COURSE).with_suffix(".json").read_text(encoding="utf-8"))


def load(source: str | None = None) -> tuple:
    """The model after DPO the lessons after this one go on with: the learner's SFT model with their adapters of
    work/projects/llm/dpo.pt when they are there and fit it (or "yours" is asked), else the course's SFT model with
    the course's adapters; with its information and where it comes from, "yours" or "course"."""
    import torch
    info = saved() if source in (None, "yours") else None
    try:   # the learner's adapters only on the SFT model they were trained on
        if info is not None and info.get("sft_sha256") != sft_sha256(info["base"]):
            info = None
    except OSError:
        info = None
    base = info["base"] if info is not None else "course"
    model, sft_info, _ = T.load(base)
    ran = run_code("from llm.add_lora import add_lora")
    torch.manual_seed(0)
    ran.names["add_lora"](model, RANK, ALPHA)
    if info is not None:
        where, state = "yours", projects.folder() / "llm" / f"{NAME}.pt"
    else:
        where, info, state = "course", course_info(), (paths.ROOT / COURSE).with_suffix(".pt")
    model.load_state_dict({k: v.float() for k, v in torch.load(state, map_location="cpu", weights_only=True).items()}, strict=False)
    return model.eval(), {**sft_info, "dpo": info}, where


def losses_picture(losses: list, every: int, labels: dict) -> str:
    """The DPO loss along the run, of every `every`th step, as the mean of the last five, in the colour of loss, with
    ln 2 dashed."""
    from everything_ai import chart
    smooth = [sum(losses[max(0, i - 4):i + 1]) / len(losses[max(0, i - 4):i + 1]) for i in range(len(losses))]   # the alt text ends at smooth[-1]
    xs = [1 + i * every for i in range(len(losses))]
    p = chart.Panel((1, xs[-1]), (0, 0.8), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    p.line([1, xs[-1]], [math.log(2)] * 2, "ink-3", 1.2, dashed=True)
    p.text(xs[-1], math.log(2) + 0.04, "ln 2", "ink", anchor="end")
    p.line(xs, smooth, "loss")
    svg = chart.figure([p], alt=labels["alt"], width=560, height=280)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- measured: the course's run of the demo with its own solutions, on the CPU, one thread; "yours" on its model of
# the demo run of llm-sft (on the demo run of llm-pretrain-capstone, TinyStories, so not shared: only its numbers),
# "course" on the course's own SFT model, data/course-sft.pt (its adapters are data/course-dpo.pt) ---
MEASURED = {
    "yours": {"base": "yours", "pairs": 1569, "instructions": 1000, "steps": 197, "trainable": 43776, "moved": {"chosen": 3.03, "rejected": -24.23}, "after": {"used": 64, "stopped": 100, "total": 100}, "before": {"used": 29, "total": 100}, "hits": "1010011111101000101100011010111000101010111011110110101111101011101111101010100110101111110111101101", "example": {"words": ["fish"], "after": "Tom and Mia liked to play with toys. They had a big white red fish that they had to pick up the frog and crawl around the frog and they ran to the fish. The frog liked to swim and catch the fish and crawl around.\n\"Look, I can fish you!\" Tom said to him rolling the fish and crawled up to the fish.", "better": True}, "losses": [0.6931, 0.682, 0.6533, 0.664, 0.6258, 0.4969, 0.6302, 0.5125, 0.4982, 0.5158, 0.663, 0.4086, 0.3789, 0.557, 0.6197, 0.7019, 0.5357, 0.6042, 0.6159, 0.459, 0.6671, 0.4185, 0.3672, 0.4567, 0.3833, 0.4335, 0.4021, 0.6892, 0.3945, 0.3812, 0.3685, 0.8159, 0.2187, 0.3812, 0.1674, 0.1946, 0.3116, 0.3934, 0.1045, 0.6754, 0.5302, 0.2779, 0.4864, 0.7212, 0.3669, 0.2541, 0.1699, 0.2035, 0.2417, 0.7826], "every": 4, "seconds": 167.0, "other_seconds": 116.5, "mixed": 470},
    "course": {"base": "course", "pairs": 1569, "instructions": 1000, "steps": 197, "trainable": 175104, "moved": {"chosen": -11.14, "rejected": -50.26}, "after": {"used": 60, "stopped": 100, "total": 100}, "before": {"used": 24, "total": 100}, "hits": "1011101110101001111100101110101011101011101110101100101111100010101011100000100011100111101010101111", "example": {"words": ["park"], "after": "In a bright, sunny day, three friends named Timmy, Lucy, and Max were exploring their favorite park bench near the park. One fifth, they enjoyed sandwiches, picnics, and some cool puddles. As they was setting up Mrs. Johnson's barrison, Max asked, \"Mrs.", "better": True}, "losses": [0.6931, 0.6817, 0.6765, 0.5907, 0.6781, 0.749, 0.6723, 0.5318, 0.5774, 0.5035, 0.5495, 0.5263, 0.5749, 0.491, 0.4464, 0.4376, 0.5304, 0.4218, 0.3696, 0.489, 0.4676, 0.457, 0.2683, 0.3534, 0.3104, 0.4335, 0.3138, 0.4915, 0.4734, 0.3784, 0.3721, 0.3913, 0.139, 0.3585, 0.1318, 0.2443, 0.3642, 0.1878, 0.1973, 0.3473, 0.3349, 0.0838, 0.2156, 0.2238, 0.2569, 0.2528, 0.1003, 0.1724, 0.1832, 0.1398], "every": 4, "seconds": 537.1, "other_seconds": 248.1, "mixed": 470},
}
# --- end of measured ---

REQUIRED = ["dpo-pairs", "reference-logprob", "dpo-step", "why-off"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL
_SCENARIO = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`"
_SCENARIO_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn"


def _close(got, want) -> bool:
    """Numbers up to float32, also inside lists and tuples; names and other values as they are."""
    if isinstance(want, (list, tuple)):
        return isinstance(got, (list, tuple)) and len(got) == len(want) and all(_close(g, w) for g, w in zip(got, want))
    if isinstance(want, str) or isinstance(got, str):
        return got == want
    return D._close(got, want)


# --- exercise 1: dpo_pairs ---------------------------------------------------------------------------------------------

_P_HEAD = "from llm.preference_pairs import make_pairs\nfrom llm.words_used import words_used\n\n\ndef dpo_pairs(items):\n"
_P = (_P_HEAD + "    pairs = []\n    for item in items:\n"
      "        hits = [words_used(answer, item[\"words\"]) for answer in item[\"answers\"]]\n"
      "        for chosen, rejected in make_pairs(item[\"answers\"], hits):\n"
      "            pairs.append((item[\"words\"], chosen, rejected))\n    return pairs\n")
_P_START = (_P_HEAD + "    pairs = []\n    for item in items:\n"
            "        # for every answer: does it use every requested word?\n        hits = ...\n"
            "        # every chosen and rejected answer of this instruction, with its words\n        ...\n    return pairs\n")
_ITEMS = [{"words": ["kite"], "answers": ["A kite flew.", "A dog ran.", "The kite fell."]},
          {"words": ["ball", "park"], "answers": ["The ball rolled in the park.", "The ball rolled.", "A park."]}]
_ITEMS_MORE = [{"words": ["cat"], "answers": ["A cat sat.", "The cat ran!"]},                      # all pass: no pairs
               {"words": ["sun"], "answers": ["It was dark.", "Rain fell."]},                      # all fail: no pairs
               {"words": ["tree", "bird"], "answers": ["A bird in a tree.", "A tree.", "Birds sang.", "The tree, the bird."]}]
_ITEMS_WORDS = [{"words": ["sun"], "answers": ["The Sun rose.", "We lived in Sunnyville.", "It was sunny."]}]   # only words_used tells
_P_BY_HAND = _P.replace("words_used(answer, item[\"words\"])", "all(w in answer.lower() for w in item[\"words\"])")
_P_NO_WORDS = _P.replace("pairs.append((item[\"words\"], chosen, rejected))", "pairs.append((chosen, rejected))")
_P_SWAPPED = _P.replace("pairs.append((item[\"words\"], chosen, rejected))", "pairs.append((item[\"words\"], rejected, chosen))")
_P_EARLY = _P.replace("            pairs.append((item[\"words\"], chosen, rejected))\n    return pairs\n",
                      "            pairs.append((item[\"words\"], chosen, rejected))\n        return pairs\n")

def _triples(pairs) -> list | None:
    """(words, chosen, rejected) as tuples, in order; None when it is not a list of them, each a tuple or a list of three
    with the words a list or a tuple of strings."""
    if not isinstance(pairs, list) or not all(isinstance(p, (tuple, list)) and len(p) == 3 and isinstance(p[0], (tuple, list))
                                               and all(isinstance(x, str) for x in (*p[0], p[1], p[2])) for p in pairs):
        return None
    return sorted((tuple(w), c, r) for w, c, r in pairs)


def _same_triples(got, want) -> bool:
    """The same pairs, in any order: the demo shuffles them anyway."""
    a, b = _triples(got), _triples(want)
    return a == b if a is not None and b is not None else same_value(got, want)


# --- exercise 2: reference_logprob -------------------------------------------------------------------------------------

_R_HEAD = "import torch\nfrom llm.answer_logprob import answer_logprob\n\n\ndef reference_logprob(model, layers, ids, start):\n"
_R_OFF = "    for layer in layers:\n        layer.on = False\n"
_R_CALC = ("    with torch.no_grad():\n        logits = model(torch.tensor([ids[:-1]]))[0]\n"
           "        logp = answer_logprob(logits, ids, start)\n")
_R_ON = "    for layer in layers:\n        layer.on = True\n"
_R = _R_HEAD + _R_OFF + _R_CALC + _R_ON + "    return logp\n"
_R_START = (_R_HEAD + "    # the reference model: the policy with its adapters off\n    ...\n" + _R_CALC
            + "    # and the policy again\n    ...\n    return logp\n")
_R_SETUP = ("import torch\nfrom llm.modern_gpt import ModernGPT\nfrom llm.add_lora import add_lora\nfrom llm.answer_logprob import answer_logprob\n\n"
            "torch.manual_seed(0)\nmodel = ModernGPT(40, 16, 2, 1, 2)\nids = [3, 1, 4, 1, 5, 9, 2, 6]\n"
            "with torch.no_grad():\n    sft = float(answer_logprob(model(torch.tensor([ids[:-1]]))[0], ids, 3))\n"
            "layers = add_lora(model, 2, 4)\nwith torch.no_grad():   # adapters that have learned something\n"
            "    for layer in layers:\n        layer.B.normal_(0, 0.5)\n")
_POLICY = "float(answer_logprob(model(torch.tensor([ids[:-1]]))[0], ids, 3))"
_R_SCENARIOS = [
    # the lesson's example: the SFT model's log-probability, although the adapters have learned
    _R_SETUP + "round(float(reference_logprob(model, layers, ids, 3)), 4), round(sft, 4)",
    # afterwards the adapters are on again: the policy is still the policy
    _R_SETUP + f"policy = {_POLICY}\nreference_logprob(model, layers, ids, 3)\nround({_POLICY}, 4), round(policy, 4)",
    # another answer, from another place
    _R_SETUP + "round(float(reference_logprob(model, layers, ids, 6)), 4)",
    # afterwards every switch is on, and B is what it learned
    _R_SETUP + "reference_logprob(model, layers, ids, 3)\n[layer.on for layer in layers][:2], round(float(sum(layer.B.abs().sum() for layer in layers)), 4)",
]
_R_NOT_OFF = _R_HEAD + _R_CALC + "    return logp\n"
_R_NOT_ON = _R_HEAD + _R_OFF + _R_CALC + "    return logp\n"
_R_ZERO_B = _R_HEAD + "    for layer in layers:\n        layer.B.data.zero_()\n" + _R_CALC + "    return logp\n"
_R_FIRST = _R_HEAD + "    layers[0].on = False\n" + _R_CALC + "    layers[0].on = True\n    return logp\n"

# --- exercise 3: dpo_step ----------------------------------------------------------------------------------------------

_S_HEAD = ("import torch\nfrom llm.answer_logprob import answer_logprob\nfrom llm.dpo_loss import dpo_loss\n"
           "from llm.reference_logprob import reference_logprob\n\n\ndef dpo_step(model, layers, opt, batch, beta):\n"
           "    pc, pr, rc, rr = [], [], [], []\n    for chosen, rejected, start in batch:\n"
           "        pc.append(answer_logprob(model(torch.tensor([chosen[:-1]]))[0], chosen, start))\n")
_S_REST = ("        pr.append(answer_logprob(model(torch.tensor([rejected[:-1]]))[0], rejected, start))\n"
           "        rc.append(float(reference_logprob(model, layers, chosen, start)))\n"
           "        rr.append(float(reference_logprob(model, layers, rejected, start)))\n")
_S_LOSS = "    loss = dpo_loss(torch.stack(pc), torch.stack(pr), torch.tensor(rc), torch.tensor(rr), beta)\n"
_S_OPT = "    opt.zero_grad()\n    loss.backward()\n    opt.step()\n"
_S = _S_HEAD + _S_REST + _S_LOSS + _S_OPT + "    return loss.item()\n"
_S_START = (_S_HEAD + "        # the same for the rejected answer, and both under the reference model, as numbers\n        ...\n"
            + _S_LOSS + "    # one step of the optimizer, as in every training loop\n    ...\n    return loss.item()\n")
_B_MOVED = "round(float(sum(layer.B.abs().sum() for layer in layers)), 4)"   # how far B moved from zero, over all adapters
_S_SETUP = ("import torch\nfrom llm.modern_gpt import ModernGPT\nfrom llm.add_lora import add_lora\n\n"
            "torch.manual_seed(0)\nmodel = ModernGPT(40, 16, 2, 1, 2)\nlayers = add_lora(model, 2, 4)\n"
            "opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)\n"
            "batch = [([3, 1, 4, 1, 5, 9, 2], [3, 1, 4, 8, 8, 8, 2], 3), ([7, 2, 5, 5, 6, 2], [7, 2, 5, 9, 2], 3)]\n")
_S_SCENARIOS = [
    # the lesson's example: three steps on the same batch, and how far B moved from zero
    _S_SETUP + f"[round(dpo_step(model, layers, opt, batch, 0.5), 4) for _ in range(3)], {_B_MOVED}",
    # the adapters learn, and only they
    _S_SETUP + "before = {name: p.detach().clone() for name, p in model.named_parameters()}\n"
               "dpo_step(model, layers, opt, batch, 0.5)\ndpo_step(model, layers, opt, batch, 0.5)\n"
               "sorted({name.rsplit('.', 1)[-1] for name, p in model.named_parameters() if not torch.equal(p, before[name])})",
    # a batch of one pair, another beta, five steps
    _S_SETUP + "[round(dpo_step(model, layers, opt, batch[:1], 0.1), 4) for _ in range(5)]",
]
_S_NO_ZERO = _S.replace("    opt.zero_grad()\n", "")
_S_NO_STEP = _S.replace("    opt.step()\n", "")
_S_POLICY_REF = _S.replace("        rc.append(float(reference_logprob(model, layers, chosen, start)))\n        rr.append(float(reference_logprob(model, layers, rejected, start)))\n",
                           "        rc.append(float(answer_logprob(model(torch.tensor([chosen[:-1]]))[0], chosen, start)))\n"
                           "        rr.append(float(answer_logprob(model(torch.tensor([rejected[:-1]]))[0], rejected, start)))\n")
_S_SWAPPED = _S.replace("dpo_loss(torch.stack(pc), torch.stack(pr), torch.tensor(rc), torch.tensor(rr), beta)",
                        "dpo_loss(torch.stack(pr), torch.stack(pc), torch.tensor(rr), torch.tensor(rc), beta)")


EXERCISES = {
    "dpo-pairs": FunctionBox(
        name="dpo-pairs",
        function="dpo_pairs",
        start=_P_START,
        solution=_P,
        visible=[(_ITEMS,)],
        hidden=[(_ITEMS_MORE,), ([],), (_ITEMS + _ITEMS_MORE,), (_ITEMS_WORDS,)],
        compare=_same_triples,
        save_as=("llm", "dpo_pairs"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-words", wrong=_P_NO_WORDS,
                    en=_CALL + ": your pairs leave out the requested words. The pair needs its instruction, to write it in front of both answers: `(item[\"words\"], chosen, rejected)`.",
                    nl=_CALL_NL + ": je paren laten de gevraagde woorden weg. Het paar heeft zijn instructie nodig, om die voor beide antwoorden te zetten: `(item[\"words\"], chosen, rejected)`."),
            Mistake("swapped", wrong=_P_SWAPPED,
                    en=_CALL + ": in your pairs the answer that fails the checker comes first. First the chosen answer, then the rejected one, as `make_pairs` gives them.",
                    nl=_CALL_NL + ": in je paren staat het antwoord dat zakt voor de checker vooraan. Eerst het gekozen antwoord, dan het afgewezen antwoord, zoals `make_pairs` ze geeft."),
            Mistake("by-hand", wrong=_P_BY_HAND,
                    en=_CALL + ": your verdicts count a word that is only part of another, as *sun* in *Sunnyville*. Take the verdict on every answer from your `words_used`: it compares whole words.",
                    nl=_CALL_NL + ": je oordelen tellen een woord dat alleen een deel van een ander is, zoals *sun* in *Sunnyville*. Neem het oordeel over elk antwoord van je `words_used`: die vergelijkt hele woorden."),
            Mistake("early", wrong=_P_EARLY,
                    en=_CALL + ": your function stops after the first instruction: the `return` is inside the loop. Move it out, to the level of `pairs = []`.",
                    nl=_CALL_NL + ": je functie stopt na de eerste instructie: de `return` staat in de loop. Zet hem erbuiten, op het niveau van `pairs = []`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the checker's verdict on every answer, and a pair for every chosen and rejected answer.",
                      "nl": "Vervang de twee `...`: het oordeel van de checker over elk antwoord, en een paar voor elk gekozen en afgewezen antwoord."},
            "wrong": {"en": _CALL + ". For every instruction: `words_used` on each answer, `make_pairs` on the answers and those verdicts, and every pair with the instruction's words in front?",
                      "nl": _CALL_NL + ". Voor elke instructie: `words_used` op elk antwoord, `make_pairs` op de antwoorden en die oordelen, en elk paar met de woorden van de instructie ervoor?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Instructions where every answer passes, or none, no instructions at all, and the verdicts of `words_used` on every answer?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Instructies waar elk antwoord slaagt, of geen, helemaal geen instructies, en de oordelen van `words_used` over elk antwoord?"},
            "correct": {"en": "Your function turns the checker's verdicts on the answers to every instruction into preference pairs, each with its words. It is in your project now, as `{saved}`.",
                        "nl": "Je functie maakt van de oordelen van de checker over de antwoorden op elke instructie preference pairs, elk met zijn woorden. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `dpo_pairs` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `dpo_pairs` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "reference-logprob": ScenarioBox(
        name="reference-logprob",
        start=_R_START,
        solution=_R,
        scenarios=_R_SCENARIOS,
        compare=_close,
        save_as=("llm", "reference_logprob"),
        first=[_DOTS],
        mistakes=[
            Mistake("not-off", wrong=_R_NOT_OFF,
                    en=_SCENARIO + ": your function gives the log-probability of the policy, adapters and all. The reference model is the model without them: set `layer.on = False` for every layer of `layers` first.",
                    nl=_SCENARIO_NL + ": je functie geeft de log-kans van de policy, met adapters en al. Het reference model is het model zonder: zet eerst `layer.on = False` voor elke layer van `layers`."),
            Mistake("first-only", wrong=_R_FIRST,
                    en=_SCENARIO + ": your function switches off only the first adapter. Every layer of `layers` has one: switch them all off, and all on again.",
                    nl=_SCENARIO_NL + ": je functie zet alleen de eerste adapter uit. Elke layer van `layers` heeft er een: zet ze allemaal uit, en allemaal weer aan."),
            Mistake("zero-b", wrong=_R_ZERO_B,
                    en=_SCENARIO + ": your function sets `B` to zero. That gives the reference model once, but the adapters lose what they learned. Switch them off with `layer.on = False` instead, and on again afterwards.",
                    nl=_SCENARIO_NL + ": je functie zet `B` op nul. Dat geeft één keer het reference model, maar de adapters zijn kwijt wat ze leerden. Zet ze in plaats daarvan uit met `layer.on = False`, en daarna weer aan."),
            Mistake("not-on", wrong=_R_NOT_ON,
                    en=_SCENARIO + ": after your function the adapters stay off, so the policy is the reference model from then on, and the adapters get no gradient. Switch them on again before the `return`.",
                    nl=_SCENARIO_NL + ": na je functie blijven de adapters uit, dus is de policy vanaf dan het reference model, en krijgen de adapters geen gradient. Zet ze weer aan voor de `return`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: every adapter off before the model runs, and on again after.",
                      "nl": "Vervang de twee `...`: elke adapter uit voordat het model draait, en daarna weer aan."},
            "wrong": {"en": _SCENARIO + ". Every adapter of `layers` off, then the log-probability, and every adapter on again?",
                      "nl": _SCENARIO_NL + ". Elke adapter van `layers` uit, dan de log-kans, en elke adapter weer aan?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Are the adapters on again afterwards, and does `start` count?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Staan de adapters daarna weer aan, en telt `start` mee?"},
            "correct": {"en": "Your function gives the log-probability under the reference model, without a second model in memory: the SFT model is still there, under the adapters. It is in your project now, as `{saved}`.",
                        "nl": "Je functie geeft de log-kans onder het reference model, zonder een tweede model in het geheugen: het SFT-model is er nog, onder de adapters. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `reference_logprob` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `reference_logprob` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "dpo-step": ScenarioBox(
        name="dpo-step",
        start=_S_START,
        solution=_S,
        scenarios=_S_SCENARIOS,
        compare=_close,
        save_as=("llm", "dpo_step"),
        first=[_DOTS],
        mistakes=[
            Mistake("tensor", error=lambda e: isinstance(e, TypeError) and "__round__" in str(e),
                    en="With the lines of the check after your code, Python stops: your `dpo_step` gives back a tensor, and the check, like the demo, needs a number. Give back `loss.item()`.",
                    nl="Met de regels van de check na je code stopt Python: je `dpo_step` geeft een tensor terug, en de check heeft, net als de demo, een getal nodig. Geef `loss.item()` terug."),
            Mistake("no-graph", error=lambda e: isinstance(e, RuntimeError) and "does not require grad" in str(e),
                    en="With the lines of the check after your code, Python stops at `loss.backward()`: nothing in your loss leads back to the adapters. Keep the policy's log-probabilities as tensors, and make one tensor of each list with `torch.stack`, not with `torch.tensor` or `float`.",
                    nl="Met de regels van de check na je code stopt Python bij `loss.backward()`: niets in je loss leidt terug naar de adapters. Houd de log-kansen van de policy als tensors, en maak van elke lijst één tensor met `torch.stack`, niet met `torch.tensor` of `float`."),
            Mistake("no-step", wrong=_S_NO_STEP,
                    en=_SCENARIO + ": the loss stays at 0.6931, ln 2, and `B` stays zero: the optimizer takes no step with the gradients. End with `opt.zero_grad()`, `loss.backward()` and `opt.step()`, in that order.",
                    nl=_SCENARIO_NL + ": de loss blijft 0,6931, ln 2, en `B` blijft nul: de optimizer zet geen stap met de gradients. Sluit af met `opt.zero_grad()`, `loss.backward()` en `opt.step()`, in die volgorde."),
            Mistake("no-zero", wrong=_S_NO_ZERO,
                    en=_SCENARIO + ": the gradients of every step add up, because nothing sets them back to zero. Start the optimizer's part with `opt.zero_grad()`.",
                    nl=_SCENARIO_NL + ": de gradients van elke stap tellen op, want niets zet ze terug op nul. Begin het deel van de optimizer met `opt.zero_grad()`."),
            Mistake("policy-ref", wrong=_S_POLICY_REF,
                    en=_SCENARIO + ": your reference is the policy itself, adapters on: at the first step that is the same, as B is still zero, but after it the margin is always 0 and the loss stays at ln 2. Take the reference model's with your `reference_logprob`.",
                    nl=_SCENARIO_NL + ": je reference is de policy zelf, met de adapters aan: bij de eerste stap is dat hetzelfde, want B is nog nul, maar daarna is de marge altijd 0 en blijft de loss ln 2. Neem die van het reference model, met je `reference_logprob`."),
            Mistake("swapped", wrong=_S_SWAPPED,
                    en=_SCENARIO + ": the loss of your steps is that of the pairs the other way round, so DPO pushes towards the rejected answers. `dpo_loss` takes the policy's chosen and rejected first, then the reference model's, in the same order.",
                    nl=_SCENARIO_NL + ": de loss van je stappen is die van de paren andersom, dus duwt DPO naar de afgewezen antwoorden. `dpo_loss` neemt eerst het gekozen en het afgewezen antwoord van de policy, dan die van het reference model, in dezelfde volgorde."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the three other log-probabilities of every pair, and the step of the optimizer.",
                      "nl": "Vervang de twee `...`: de drie andere log-kansen van elk paar, en de stap van de optimizer."},
            "wrong": {"en": _SCENARIO + ". The rejected answer under the policy, both under the reference model as numbers, and then `opt.zero_grad()`, `loss.backward()` and `opt.step()`?",
                      "nl": _SCENARIO_NL + ". Het afgewezen antwoord onder de policy, beide onder het reference model als getallen, en dan `opt.zero_grad()`, `loss.backward()` en `opt.step()`?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Do only the adapters learn, and does another batch and `beta` work?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Leren alleen de adapters, en werkt een andere batch en `beta`?"},
            "correct": {"en": "Your function takes one step of DPO: the policy against the reference model on every pair, and a step of the adapters towards bigger margins. It is in your project now, as `{saved}`.",
                        "nl": "Je functie zet één stap DPO: de policy tegen het reference model op elk paar, en een stap van de adapters naar grotere marges. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `dpo_step` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `dpo_step` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("dpo-pairs", _P_START, "empty", None),
    ("dpo-pairs", _P, "correct", None),
    ("dpo-pairs", _P_HEAD + "    pairs = []\n    for item in items:\n        words, answers = item[\"words\"], item[\"answers\"]\n"
                  "        hits = []\n        for answer in answers:\n            hits.append(words_used(answer, words))\n"
                  "        pairs += [(words, c, r) for c, r in make_pairs(answers, hits)]\n    return pairs\n", "correct", None),
    ("dpo-pairs", _P_NO_WORDS, "wrong", "no-words"),
    ("dpo-pairs", _P_SWAPPED, "wrong", "swapped"),
    ("dpo-pairs", _P_EARLY, "wrong", "early"),
    ("dpo-pairs", _P_BY_HAND, "wrong", "by-hand"),
    ("dpo-pairs", _P.replace("pairs.append((item[\"words\"], chosen, rejected))", "pairs.append([item[\"words\"], chosen, rejected])"), "correct", None),
    ("dpo-pairs", _P.replace("pairs.append((item[\"words\"], chosen, rejected))", "pairs.append((\" and \".join(item[\"words\"]), chosen, rejected))"), "wrong", None),
    ("reference-logprob", _R_START, "empty", None),
    ("reference-logprob", _R, "correct", None),
    ("reference-logprob", _R_HEAD + _R_OFF + _R_CALC + _R_ON + "    return float(logp)\n", "correct", None),
    ("reference-logprob", _R_NOT_OFF, "wrong", "not-off"),
    ("reference-logprob", _R_FIRST, "wrong", "first-only"),
    ("reference-logprob", _R_NOT_ON, "wrong", "not-on"),
    ("reference-logprob", _R_ZERO_B, "wrong", "zero-b"),
    ("dpo-step", _S_START, "empty", None),
    ("dpo-step", _S, "correct", None),
    ("dpo-step", _S_NO_STEP, "wrong", "no-step"),
    ("dpo-step", _S_NO_ZERO, "wrong", "no-zero"),
    ("dpo-step", _S_POLICY_REF, "wrong", "policy-ref"),
    ("dpo-step", _S_SWAPPED, "wrong", "swapped"),
    ("dpo-step", _S.replace(_S_OPT, "    loss.backward()\n    opt.step()\n    opt.zero_grad()\n"), "correct", None),
    ("dpo-step", _S.replace(_S_OPT, "    loss.backward()\n    opt.zero_grad()\n    opt.step()\n"), "wrong", "no-step"),
    ("dpo-step", _S.replace("    return loss.item()\n", "    return loss\n"), "error", "tensor"),
    ("dpo-step", _S.replace("torch.stack(pc), torch.stack(pr)", "torch.tensor(pc), torch.tensor(pr)"), "error", "no-graph"),
]
