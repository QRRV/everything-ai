# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson llm-same-on-open-model: rename, which turns the names in Qwen3's file into those of the
course's Qwen3 with ModernGPT's names; lora_parameters, the numbers add_lora trains on a model, from its config.json;
and a choice. The demo loads Qwen3-0.6B-Base (everything_ai.qwen3) with the learner's rename, and trains it with the
learner's own modules of the llm project: add_lora, sft_example, sft_batch and pretrain_step for SFT on short questions
of OASST1 with their best answer (data/oasst-short.json); merge_lora to fold those adapters into the model; then
add_lora again and dpo_step for DPO on short pairs of HelpSteer2 (data/helpsteer2-short.json). It measures before and
after, on questions and pairs it never trained on: the loss on the people's answers, how many answers stop by
themselves with <|endoftext|>, and on how many pairs DPO's reward agrees with the people."""
import importlib.util
import json
import math
import random
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DP = _checks("llm-dpo", "everything_ai_checks_llm_dpo_for_open_model")
T, S = DP.T, DP.S
BUILDS_ON = DP.BUILDS_ON   # add_lora, merge_lora, sft_example, sft_batch, pretrain_step and dpo_step, with what they import
projects.builds_on(*BUILDS_ON)

OASST, HELPSTEER2 = "data/oasst-short.json", "data/helpsteer2-short.json"
COURSE = "data/course-open-model.json"   # the course's run, at the settings of a GPU, and what it measured
RANK, ALPHA = 16, 32      # the adapters: rank 16 on every linear layer of the blocks, as the course measured
SFT = {"lr": 4e-4, "steps": {"cpu": 30, "gpu": 150}, "batch": {"cpu": 2, "gpu": 4}}
DPO = {"lr": 1e-4, "beta": 0.1, "steps": {"cpu": 10, "gpu": 60}, "batch": {"cpu": 2, "gpu": 4}}
MEASURE = {"cpu": {"loss": 20, "stops": 5, "pairs": 20}, "gpu": {"loss": 50, "stops": 50, "pairs": 100}}
NEW = 256                 # tokens an answer may take while measuring; the longest answer of the people is 192
SHOWN = 3                 # held-out questions whose answers the demo shows
START, TURN = "<|im_start|>", "<|im_end|>"   # the tokens of Qwen's chat template around a turn
END = "<|endoftext|>"     # the end of an answer: the one special token the base model learned; the end id of SFT


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def prompt(question: str) -> str:
    """A question in Qwen's chat template, with the generation prompt: the opening of the assistant's turn."""
    return f"{START}user\n{question}{TURN}\n{START}assistant\n"


def oasst() -> dict:
    return json.loads((paths.ROOT / OASST).read_text(encoding="utf-8"))


def helpsteer2() -> dict:
    return json.loads((paths.ROOT / HELPSTEER2).read_text(encoding="utf-8"))


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def header() -> dict:
    from everything_ai import qwen3
    return qwen3.header()


def model_names() -> dict:
    from everything_ai import qwen3
    return qwen3.model_names()


def _capstone() -> dict:
    """The sizes of the course's model of llm-pretrain-capstone, the model the learner's lessons trained."""
    info = json.loads((paths.ROOT / "data" / "course-capstone.json").read_text(encoding="utf-8"))
    vocab, d, heads, kv_heads, layers = info["sizes"]
    return {"params": info["params"], "tokens": info["tokens"], "vocab": vocab, "d": d, "heads": heads, "kv_heads": kv_heads,
            "layers": layers}


CAPSTONE = _capstone()
QWEN3_TOKENS = 36e12   # the tokens Qwen3 was pretrained on: about 36 trillion (Qwen3 Technical Report, 2025)


def excerpt(text: str, n: int) -> str:
    """A text on one line, for a list in Markdown: its first `n` characters, cut at a space, with … only when there is
    more; the characters Markdown would read as formatting are written as themselves."""
    flat = " ".join(text.split())
    if len(flat) > n:
        cut = flat[:n]
        flat = (cut[:cut.rfind(" ")] if " " in cut else cut).rstrip(" ,;:") + "…"
    for ch in "\\`*_[]<>#|~":
        flat = flat.replace(ch, "\\" + ch)
    return flat


def sizes_picture(labels: dict) -> str:
    """Two bar charts on a log scale, side by side: the parameters of the learner's model and of Qwen3-0.6B, in the
    colour of what a model learns, and the tokens each trained on, in ink."""
    from everything_ai import chart
    params = [CAPSTONE["params"], sum(math.prod(t["shape"]) for t in header()["tensors"].values())]
    tokens = [CAPSTONE["tokens"], QWEN3_TOKENS]
    panels = []
    for title, values, lo, hi, role in ((labels["title"][0], params, 1e6, 1e9, "param"),
                                        (labels["title"][1], tokens, 1e6, 1e14, "ink")):
        p = chart.Panel((lo, hi), (-0.5, 1.5), title=title, xlog=True, yticks=[])
        for row, (name, v) in enumerate(zip(labels["rows"], values)):
            y = 1 - row
            p.rect(lo, y - 0.2, v, y + 0.2, role)
            p.text(lo * 1.3, y + 0.46, name, "ink-2", "start")
        panels.append(p)
    return chart.figure(panels, alt=labels["alt"], width=300, height=170)


# --- the model in the lesson: downloaded once ------------------------------------------------------------------------

def downloaded() -> bool:
    from everything_ai import qwen3
    return qwen3.downloaded()


def download_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until Qwen3 is downloaded; a click downloads it, with the
    progress in the cell. `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned", "no_room" and
    "failed" ({why}), as in lm-peek-real-llm."""
    import errno
    import html
    import urllib.error

    import marimo as mo

    from everything_ai import hub, qwen3
    failed = None
    if not downloaded() and button.value:
        total = round(sum(hub.PINNED[(qwen3.REPO, f)]["size"] for f in qwen3.FILES) / 1e6)
        shown = [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 20 != shown[0] // 20:   # every 20 MB
                shown[0] = mb
                mo.output.replace(mo.md(texts["busy"].format(done=mb, total=total)))
        try:
            qwen3.download(progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        mo.stop(True, mo.vstack(([mo.md(failed)] if failed else []) + [mo.md(texts["ask"]), button]))


# --- measuring -----------------------------------------------------------------------------------------------------

def _autocast(device: str):
    """bfloat16 for the matrix products on a GPU, as the trainer of gpt-train-chars does; nothing on the CPU."""
    import contextlib

    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def answer_loss(model, pairs: list, enc, end: int) -> float:
    """The mean loss, per token, on the people's answers to `pairs` (question, answer), after each question in the
    chat template: the answer and its <|endoftext|>, as SFT trains on them."""
    import torch
    import torch.nn.functional as F
    from everything_ai import qwen3
    device = next(model.parameters()).device
    total, n = 0.0, 0
    with torch.no_grad(), _autocast(device.type):
        for q, a in pairs:
            p, ids = qwen3.encode(enc, prompt(q)), qwen3.encode(enc, a) + [end]
            x = torch.tensor([p + ids[:-1]], device=device)
            logits = model(x)[0, len(p) - 1:].float()
            total += float(F.cross_entropy(logits, torch.tensor(ids, device=device), reduction="sum"))
            n += len(ids)
    return total / n


def answer(model, question: str, enc, end: int) -> tuple:
    """The model's answer to a question, greedily, with the KV cache: its text, and whether it stopped by itself with
    <|endoftext|> within NEW tokens."""
    import torch
    from everything_ai import qwen3
    device = next(model.parameters()).device
    caches = [{} for _ in model.blocks]
    out = []
    with torch.no_grad(), _autocast(device.type):
        logits = model(torch.tensor([qwen3.encode(enc, prompt(question))], device=device), caches)[0, -1]
        for _ in range(NEW):
            j = int(logits.argmax())
            if j == end:
                return enc.decode(out), True
            out.append(j)
            logits = model(torch.tensor([[j]], device=device), caches)[0, -1]
    return enc.decode(out), False


def answers(model, questions: list, enc, end: int, show, phase: str) -> dict:
    """The answers to `questions`, and how many stopped by themselves."""
    got = []
    for k, q in enumerate(questions):
        got.append(answer(model, q, enc, end))
        show({"phase": phase, "done": k + 1, "total": len(questions)})
    return {"stopped": sum(s for _, s in got), "total": len(got), "texts": [t for t, _ in got[:SHOWN]],
            "ended": [s for _, s in got[:SHOWN]]}


def agreement(model, layers, pairs: list, answer_logprob, reference_logprob, show) -> dict:
    """On how many of `pairs` (start, chosen ids, rejected ids) DPO's reward agrees with the people: the chosen
    answer's log-probability rose more, from the reference model to the policy, than the rejected one's."""
    import torch
    device = next(model.parameters()).device
    agree = 0
    with torch.no_grad(), _autocast(device.type), torch.device(device.type):
        for k, (chosen, rejected, start) in enumerate(pairs):
            margin = 0.0
            for ids, sign in ((chosen, 1), (rejected, -1)):
                policy = float(answer_logprob(model(torch.tensor([ids[:-1]]))[0].float(), ids, start))
                margin += sign * (policy - float(reference_logprob(model, layers, ids, start)))
            agree += margin > 0
            show({"phase": "agree", "done": k + 1, "total": len(pairs)})
    return {"agree": agree, "pairs": len(pairs)}


# --- the demo --------------------------------------------------------------------------------------------------------

def tools() -> dict | str:
    """The learner's modules the demo needs, or "run" when one stops with an error."""
    ran = run_code("from llm.add_lora import add_lora\nfrom llm.merge_lora import merge_lora\n"
                   "from llm.sft_example import sft_example\nfrom llm.sft_batch import sft_batch\n"
                   "from llm.pretrain_step import pretrain_step\nfrom llm.dpo_step import dpo_step\n"
                   "from llm.answer_logprob import answer_logprob\nfrom llm.reference_logprob import reference_logprob")
    return "run" if ran.error is not None else ran.names


def _merge(model, merge_lora) -> None:
    """Fold every LoRALinear of the blocks into a plain linear layer, with the learner's merge_lora."""
    for block in model.blocks:
        for part, names in ((block.attention, ("query", "key", "value", "out")), (block.mlp, ("gate", "up", "down"))):
            for name in names:
                layer = getattr(part, name)
                if hasattr(layer, "base"):
                    setattr(part, name, merge_lora(layer))


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(rename, show, kind: str | None = None) -> dict | str:
    """The demo: Qwen3-0.6B-Base loaded with the learner's rename, measured; SFT with adapters of the learner's
    add_lora, sft_example, sft_batch and pretrain_step; measured; the adapters merged with merge_lora; DPO with new
    adapters and dpo_step; measured. `kind` is "cpu" or "gpu", the settings of SFT, DPO and MEASURE: those of the
    device when None. A string when it stops: "names" (rename), "run" (a module of the kit, or loading), "lora"
    (add_lora), "sft" (sft_example, sft_batch or pretrain_step), "merge" (merge_lora), "dpo" (dpo_step), "memory"."""
    import torch
    from everything_ai import qwen3
    t = tools()
    if isinstance(t, str):
        return t
    try:
        qwen3.check_names(rename)
    except Exception:
        return "names"
    w = T.P.where()
    kind = kind or w["kind"]
    device = w["device"]
    seconds, begin = {}, time.perf_counter()
    show({"phase": "load"})
    try:
        model = qwen3.load(rename).to(device)
        enc = qwen3.tokenizer()
        end = enc.encode_single_token(END)
    except Exception as e:
        return "memory" if "out of memory" in str(e).lower() else "run"
    m = MEASURE[kind]
    sft_data, dpo_data = oasst(), helpsteer2()
    questions = [q for q, _ in sft_data["held"]]
    result = {"device": w["name"], "kind": kind, "questions": questions[:SHOWN]}
    try:
        result["base"] = {"loss": answer_loss(model, sft_data["held"][:m["loss"]], enc, end),
                          **answers(model, questions[:m["stops"]], enc, end, show, "base")}
    except torch.OutOfMemoryError:
        return "memory"
    seconds["base"] = time.perf_counter() - begin

    # SFT with the learner's adapters and steps
    start = time.perf_counter()
    try:
        torch.manual_seed(0)   # the same random A on every run
        with torch.device(device):
            layers = t["add_lora"](model, RANK, ALPHA)
    except Exception:
        return "lora"
    if trainable(model) == 0:
        return "lora"
    result["trainable"] = trainable(model)
    try:
        examples = [t["sft_example"](qwen3.encode(enc, prompt(q)), qwen3.encode(enc, a), end) for q, a in sft_data["train"]]
    except Exception:
        return "sft"
    order = list(range(len(examples)))
    random.Random(0).shuffle(order)
    steps, batch = SFT["steps"][kind], SFT["batch"][kind]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=SFT["lr"], weight_decay=0.0)
    model.train()
    losses = []
    try:
        for step in range(steps):
            chosen = [examples[order[(step * batch + k) % len(order)]] for k in range(batch)]
            x, y = t["sft_batch"](chosen, end)
            with torch.device(device):
                loss = t["pretrain_step"](model, opt, x.to(device), y.to(device), SFT["lr"] * S.multiplier(step, steps),
                                          device, device == "cuda")
            if not math.isfinite(float(loss)):
                return "sft"
            losses.append(float(loss))
            show({"phase": "sft", "step": step + 1, "steps": steps, "loss": losses[-1], "seconds": time.perf_counter() - start})
    except torch.OutOfMemoryError:
        return "memory"
    except Exception:
        return "sft"
    model.eval()
    seconds["sft"] = time.perf_counter() - start
    result["sft_losses"] = losses
    try:
        result["sft"] = {"loss": answer_loss(model, sft_data["held"][:m["loss"]], enc, end),
                         **answers(model, questions[:m["stops"]], enc, end, show, "sft")}
    except torch.OutOfMemoryError:
        return "memory"

    # the SFT adapters into the weights; then DPO with new adapters, whose reference is the model after SFT
    try:
        with torch.device(device):
            _merge(model, t["merge_lora"])
    except Exception:
        return "merge"
    model.requires_grad_(False)   # a merged layer is a new nn.Linear, whose weight would train; add_lora freezes it too
    start = time.perf_counter()
    try:
        torch.manual_seed(0)
        with torch.device(device):
            layers = t["add_lora"](model, RANK, ALPHA)
    except Exception:
        return "lora"

    def pair(q, chosen, rejected):
        p = qwen3.encode(enc, prompt(q))
        return p + qwen3.encode(enc, chosen) + [end], p + qwen3.encode(enc, rejected) + [end], len(p)

    train = [pair(q, c, r) for q, c, r, _ in dpo_data["train"]]
    held = [pair(q, c, r) for q, c, r, _ in dpo_data["held"][:m["pairs"]]]
    order = list(range(len(train)))
    random.Random(0).shuffle(order)
    steps, batch = DPO["steps"][kind], DPO["batch"][kind]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=DPO["lr"], weight_decay=0.0)
    model.train()
    losses = []
    try:
        for step in range(steps):
            chosen = [train[order[(step * batch + k) % len(order)]] for k in range(batch)]
            # without autocast: under it every forward keeps a bfloat16 copy of each frozen weight for the backward
            # pass, 1.2 GB, and a step of DPO keeps eight forwards; in float32 the frozen weights are kept as they are
            with torch.device(device):
                loss = t["dpo_step"](model, layers, opt, chosen, DPO["beta"])
            if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
                return "dpo"
            losses.append(float(loss))
            show({"phase": "dpo", "step": step + 1, "steps": steps, "loss": losses[-1], "seconds": time.perf_counter() - start})
    except torch.OutOfMemoryError:
        return "memory"
    except Exception:
        return "dpo"
    model.eval()
    seconds["dpo"] = time.perf_counter() - start
    result["dpo_losses"] = losses
    seen = [train[i] for i in order[:m["pairs"]]]   # pairs it trained on: the first of its order, all in its steps
    try:
        result["dpo"] = {**agreement(model, layers, held, t["answer_logprob"], t["reference_logprob"], show),
                         "seen": agreement(model, layers, seen, t["answer_logprob"], t["reference_logprob"], show),
                         **answers(model, questions[:m["stops"]], enc, end, show, "dpo")}
    except torch.OutOfMemoryError:
        return "memory"
    except Exception:
        return "dpo"
    result["steps"] = {"sft": SFT["steps"][kind], "dpo": DPO["steps"][kind]}
    result["seconds"] = {**seconds, "all": time.perf_counter() - begin}
    return result


# --- exercise 1: rename ----------------------------------------------------------------------------------------------

_R_HEAD = "# a part of a name in Qwen3's file, and the same part in the course's model\nPARTS = {\n"
_R_GIVEN = ('    "embed_tokens": "tok",\n    "layers": "blocks",\n    "input_layernorm": "norm1",\n'
            '    "post_attention_layernorm": "norm2",\n    "self_attn": "attention",\n    "q_proj": "query",\n')
_R_REST = ('    "k_proj": "key",\n    "v_proj": "value",\n    "o_proj": "out",\n    "gate_proj": "gate",\n'
           '    "up_proj": "up",\n    "down_proj": "down",\n')
_R_DEF = ("}\n\n\ndef rename(name):\n"
          "    # \"model.layers.0.self_attn.q_proj.weight\" becomes \"blocks.0.attention.query.weight\"\n")
_R_BODY = ("    new = []\n    for part in name.split(\".\")[1:]:\n        if part in PARTS:\n            new.append(PARTS[part])\n"
           "        else:\n            new.append(part)\n    return \".\".join(new)\n")
_R_CODE = _R_HEAD + _R_GIVEN + _R_REST + _R_DEF + _R_BODY
_R_START = (_R_HEAD + _R_GIVEN + "    # and the six other linear layers of attention and the MLP\n" + _R_DEF
            + "    # every part after the first, model, by its name in the course's model: the same name when it is not in PARTS\n"
              "    ...\n")
_R_WRONG = {
    "model": _R_CODE.replace('name.split(".")[1:]', 'name.split(".")'),
    "proj": _R_HEAD + _R_GIVEN + _R_DEF + _R_BODY,
    "no-dots": _R_CODE.replace('return ".".join(new)', 'return "".join(new)'),
}
_NAMES = ["model.layers.27.mlp.down_proj.weight", "model.layers.3.self_attn.o_proj.weight", "model.layers.5.self_attn.k_norm.weight",
          "model.layers.12.post_attention_layernorm.weight", "model.embed_tokens.weight", "model.norm.weight",
          "model.layers.9.self_attn.v_proj.weight", "model.layers.1.mlp.gate_proj.weight", "model.layers.2.mlp.up_proj.weight",
          "model.layers.4.self_attn.k_proj.weight", "model.layers.6.input_layernorm.weight"]

# --- exercise 2: lora_parameters -------------------------------------------------------------------------------------

QWEN3_06B = {"vocab_size": 151936, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
             "intermediate_size": 3072, "num_hidden_layers": 28}
QWEN3_17B = {"vocab_size": 151936, "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
             "intermediate_size": 6144, "num_hidden_layers": 28}
QWEN3_4B = {"vocab_size": 151936, "hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 9728, "num_hidden_layers": 36}
_TINY = {"vocab_size": 1257, "hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
         "intermediate_size": 170, "num_hidden_layers": 2}
_P_HEAD = ("def lora_parameters(config, r):\n"
           "    # how many numbers add_lora trains, at rank r, on a model with this config.json\n"
           "    d = config[\"hidden_size\"]\n")
_P_SIZES = ("    q = config[\"num_attention_heads\"] * config[\"head_dim\"]\n"
            "    kv = config[\"num_key_value_heads\"] * config[\"head_dim\"]\n")
_P_MID = ("    hidden = config[\"intermediate_size\"]\n"
          "    # (inputs, outputs) of the seven linear layers of a block: query, key, value, out, gate, up, down\n")
_P_LAYERS = "    layers = [(d, q), (d, kv), (d, kv), (q, d), (d, hidden), (d, hidden), (hidden, d)]\n"
_P_LOOP = "    per_block = 0\n    for n_in, n_out in layers:\n"
_P_TERM = "        per_block = per_block + r * n_in + n_out * r\n"
_P_END = "    return config[\"num_hidden_layers\"] * per_block\n"


def _p(sizes: str = _P_SIZES, layers: str = _P_LAYERS, term: str = _P_TERM, end: str = _P_END) -> str:
    return _P_HEAD + sizes + _P_MID + layers + _P_LOOP + term + end


_P_CODE = _p()
_P_START = (_P_HEAD + "    # the outputs of query, and of key and value: every head is head_dim numbers wide\n    q = ...\n    kv = ...\n"
            + _P_MID + "    layers = ...\n" + _P_LOOP + "        # a LoRALinear trains A, r by its inputs, and B, its outputs by r\n"
            "        ...\n    return ...\n")
_P_WRONG = {
    "head-dim": _p(sizes="    q = d\n    kv = config[\"num_key_value_heads\"] * int(d / config[\"num_attention_heads\"])\n"),
    "attention-only": _p(layers="    layers = [(d, q), (d, kv), (d, kv), (q, d)]\n"),
    "only-a": _p(term="        per_block = per_block + r * n_in\n"),
    "full": _p(term="        per_block = per_block + n_in * n_out\n"),
    "one-block": _p(end="    return per_block\n"),
    "mlp-width": _p(sizes=_P_SIZES + "    config = dict(config, intermediate_size=int(8 * d / 3))\n"),
}

_DOTS = S._DOTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL


def _starts_model(got, text) -> bool:
    return isinstance(got, str) and got.startswith("model.")


def _proj_left(got, text) -> bool:
    return isinstance(got, str) and "_proj" in got


def _no_dots(got, text) -> bool:
    return isinstance(got, str) and "." not in got


def _a_list(got, text) -> bool:
    return isinstance(got, (list, tuple))


EXERCISES = {
    "rename": FunctionBox(
        name="rename",
        function="rename",
        start=_R_START,
        solution=_R_CODE,
        visible=[("model.layers.7.self_attn.q_proj.weight",)],
        hidden=[(n,) for n in _NAMES],
        first=[_DOTS],
        mistakes=[
            Mistake("list", matches=_a_list,
                    en=_CALL + ": a list of parts. Join them into one name again, with a dot between every two: `\".\".join(...)`.",
                    nl=_CALL_NL + ": een lijst van delen. Voeg ze weer samen tot één naam, met een punt tussen elke twee: `\".\".join(...)`."),
            Mistake("model", matches=_starts_model,
                    en=_CALL + ": the name still starts with `model`. The course's model has no part of that name: its names start with `blocks`, `tok` or `norm`. Leave out the first part of the name.",
                    nl=_CALL_NL + ": de naam begint nog met `model`. Het model van de course heeft geen deel met die naam: zijn namen beginnen met `blocks`, `tok` of `norm`. Laat het eerste deel van de naam weg."),
            Mistake("proj", matches=_proj_left,
                    en=_CALL + ": a part that ends in `_proj` stays as it was. Add the six other linear layers to `PARTS`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`, each with the name of the same layer in the course's model: the worked example lists both.",
                    nl=_CALL_NL + ": een deel dat op `_proj` eindigt, blijft zoals het was. Zet de zes andere lineaire layers in `PARTS`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` en `down_proj`, elk met de naam van dezelfde layer in het model van de course: het uitgewerkte voorbeeld geeft ze allebei."),
            Mistake("no-dots", matches=_no_dots,
                    en=_CALL + ": the parts are joined with no dots between them. Join them with `\".\".join(...)`.",
                    nl=_CALL_NL + ": de delen staan aan elkaar, zonder punten ertussen. Voeg ze samen met `\".\".join(...)`."),
        ],
        texts={
            **S._TEXTS,
            "empty": {"en": "Add the six other linear layers to `PARTS`, and replace the `...` in `rename`: every part of the name after the first, by its name in the course's model.",
                      "nl": "Zet de zes andere lineaire layers in `PARTS`, en vervang de `...` in `rename`: elk deel van de naam na het eerste, met zijn naam in het model van de course."},
            "wrong": {"en": _CALL + ". Look at the two lists in the worked example: which part of a name changes, and into what?",
                      "nl": _CALL_NL + ". Kijk naar de twee lijsten in het uitgewerkte voorbeeld: welk deel van een naam verandert er, en waarin?"},
            "almost": {"en": "Right for the query of block 7. But " + _CALL + ". Does it work for the other layers, the norms, and the names outside the blocks?",
                       "nl": "Goed voor de query van blok 7. Maar " + _CALL_NL + ". Werkt hij ook voor de andere layers, de norms, en de namen buiten de blokken?"},
            "correct": {"en": "Every kind of name in Qwen3's file now becomes the name of the same tensor in the course's model. The demo below loads Qwen3 with your `rename`, and first checks all 310 names and their shapes.",
                        "nl": "Elke soort naam in het bestand van Qwen3 wordt nu de naam van dezelfde tensor in het model van de course. De demo hieronder laadt Qwen3 met je `rename`, en controleert eerst alle 310 namen en hun shapes."},
        },
    ),
    "lora-parameters": FunctionBox(
        name="lora-parameters",
        function="lora_parameters",
        start=_P_START,
        solution=_P_CODE,
        visible=[(QWEN3_06B, 16)],
        hidden=[(QWEN3_17B, 8), (QWEN3_4B, 16), (_TINY, 4), (QWEN3_06B, 1)],
        first=[_DOTS],
        mistakes=[
            Mistake("head-dim", wrong=_P_WRONG["head-dim"],
                    en=_CALL + ": you take a head as `hidden_size / num_attention_heads` wide, as in your ModernGPT. Qwen3 says how wide a head is itself, in `head_dim`: 128, so that its 16 heads of queries are 2048 numbers, twice `hidden_size`.",
                    nl=_CALL_NL + ": je neemt een head `hidden_size / num_attention_heads` breed, zoals in je ModernGPT. Qwen3 zegt zelf hoe breed een head is, in `head_dim`: 128, zodat zijn 16 heads aan queries 2048 getallen zijn, twee keer `hidden_size`."),
            Mistake("attention-only", wrong=_P_WRONG["attention-only"],
                    en=_CALL + ": you count the four layers of attention only. Your `add_lora` also puts adapters on the three layers of the MLP: `gate`, `up` and `down`.",
                    nl=_CALL_NL + ": je telt alleen de vier layers van attention. Je `add_lora` zet ook adapters op de drie layers van de MLP: `gate`, `up` en `down`."),
            Mistake("only-a", wrong=_P_WRONG["only-a"],
                    en=_CALL + ": you count only `A`, r by the inputs. A LoRALinear trains `B` too, the outputs by r.",
                    nl=_CALL_NL + ": je telt alleen `A`, r bij de inputs. Een LoRALinear traint ook `B`, de outputs bij r."),
            Mistake("full", wrong=_P_WRONG["full"],
                    en=_CALL + ": you count every weight of the layers, inputs times outputs: what full fine-tuning trains. A LoRALinear trains only `A` and `B`: r times the inputs, and the outputs times r.",
                    nl=_CALL_NL + ": je telt elke weight van de layers, inputs maal outputs: wat full fine-tuning traint. Een LoRALinear traint alleen `A` en `B`: r maal de inputs, en de outputs maal r."),
            Mistake("mlp-width", wrong=_P_WRONG["mlp-width"],
                    en=_CALL + ": you take the MLP as `int(8 * d / 3)` wide, as in your ModernGPT. Qwen3 says how wide its MLP is itself, in `intermediate_size`: 3072.",
                    nl=_CALL_NL + ": je neemt de MLP `int(8 * d / 3)` breed, zoals in je ModernGPT. Qwen3 zegt zelf hoe breed zijn MLP is, in `intermediate_size`: 3072."),
            Mistake("one-block", wrong=_P_WRONG["one-block"],
                    en=_CALL + ": that is one block. The model has `num_hidden_layers` of them, each with its own adapters.",
                    nl=_CALL_NL + ": dat is één blok. Het model heeft er `num_hidden_layers`, elk met zijn eigen adapters."),
        ],
        texts={
            **S._TEXTS,
            "empty": {"en": "Replace the `...`: the outputs of query, key and value, the seven layers with their inputs and outputs, what each adds, and the total.",
                      "nl": "Vervang de `...`: de outputs van query, key en value, de zeven layers met hun inputs en outputs, wat elk toevoegt, en het totaal."},
            "wrong": {"en": _CALL + ". Is it, over every block, r times (inputs + outputs) for each of the seven layers?",
                      "nl": _CALL_NL + ". Is het, over elk blok, r maal (inputs + outputs) voor elk van de zeven layers?"},
            "almost": {"en": "Right for Qwen3-0.6B at rank 16. But " + _CALL + ". Does it work for another config and another rank?",
                       "nl": "Goed voor Qwen3-0.6B bij rang 16. Maar " + _CALL_NL + ". Werkt hij ook voor een andere config en een andere rang?"},
            "correct": {"en": "Your `lora_parameters` reads the sizes from a config.json. At rank 16, add_lora trains 10,092,544 numbers of Qwen3-0.6B's 596,049,920: 1.7 percent. The demo below counts them in the model too.",
                        "nl": "Je `lora_parameters` leest de maten uit een config.json. Bij rang 16 traint add_lora 10.092.544 getallen van de 596.049.920 van Qwen3-0.6B: 1,7 procent. De demo hieronder telt ze ook in het model."},
        },
    ),
}

# The lesson is done when these are correct. "end-id" is a choice question, "results" the open question at the end.
REQUIRED = ["rename", "lora-parameters", "end-id", "results"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
CASES = [
    ("rename", _R_START, "empty", None),
    ("rename", _R_CODE, "correct", None),
    ("rename", _R_WRONG["model"], "wrong", "model"),
    ("rename", _R_WRONG["proj"], "wrong", "proj"),
    ("rename", _R_WRONG["no-dots"], "wrong", "no-dots"),
    ("rename", _R_CODE.replace('return ".".join(new)', 'return "/".join(new)'), "wrong", "no-dots"),
    ("rename", _R_CODE.replace('    return ".".join(new)\n', "    return new\n"), "wrong", "list"),
    ("rename", _R_HEAD + _R_GIVEN + _R_REST + _R_DEF + "    out = []\n    for part in name.split(\".\")[1:]:\n"
     "        out.append(PARTS.get(part, part))\n    return \".\".join(out)\n", "correct", None),
    ("lora-parameters", _P_START, "empty", None),
    ("lora-parameters", _P_CODE, "correct", None),
    ("lora-parameters", _P_WRONG["head-dim"], "wrong", "head-dim"),
    ("lora-parameters", _P_WRONG["attention-only"], "wrong", "attention-only"),
    ("lora-parameters", _P_WRONG["only-a"], "wrong", "only-a"),
    ("lora-parameters", _P_WRONG["full"], "wrong", "full"),
    ("lora-parameters", _P_WRONG["one-block"], "wrong", "one-block"),
    ("lora-parameters", _p(term="        per_block = per_block + r * (n_in + n_out)\n"), "correct", None),
]
