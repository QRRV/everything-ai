# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-open-recipes: features, which parts of nanochat's GPT the learner's ModernGPT already has;
size, the width, heads and parameters nanochat gives a model of a depth; lr_multiplier, nanochat's schedule of the
learning rate; and a choice. The demo trains the learner's ModernGPT, sized by nanochat's rule, for a few hundred steps
with the learner's lr_multiplier, on the shards of llm-data-pipeline. The lesson quotes nanochat (github.com/karpathy/
nanochat, MIT, by Andrej Karpathy) at the commit below; it downloads nothing of it."""
import importlib.util
import json
import math
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, Sort, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _checks("llm-modern-gpt", "everything_ai_checks_llm_modern_gpt_for_recipes")
D = _checks("llm-data-pipeline", "everything_ai_checks_llm_data_pipeline_for_recipes")
BUILDS_ON = M.BUILDS_ON + tuple(m for m in D.BUILDS_ON if m[1] in ("shards", "token_windows"))
projects.builds_on(*BUILDS_ON)

COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"   # nanochat as the lesson quotes it, of 3 July 2026
ASPECT = 64          # nanochat: model_dim = depth * aspect_ratio, rounded up to a multiple of the head size
HEAD_DIM = 128       # nanochat: the target size of an attention head
B_REF = 2 ** 19      # nanochat: the batch of its reference model d12, in tokens
POWER = 0.383        # nanochat: the batch grows as the tokens to the power 0.383 (the Power Lines paper)

# The speedrun of runs/speedrun.sh: depth 24, 8 tokens per scaling parameter, a vocabulary of 2 ** 15.
SPEEDRUN = {"depth": 24, "ratio": 8, "vocab": 2 ** 15}

# The demo: the learner's ModernGPT at nanochat's rule for DEPTH, on windows of BLOCK tokens of the shards.
DEPTH, BLOCK, BATCH, STEPS = 4, 256, 16, 300
PEAK = 3e-3                                                # AdamW's learning rate at the top of the schedule
SCHEDULE = {"warmup": 40, "warmdown": 0.65, "final": 0.05}  # nanochat's defaults in scripts/base_train.py


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def size_of(depth: int, vocab: int) -> tuple:
    """nanochat's rule: the width, the heads, and about its scaling parameters (12 L d^2 + V d)."""
    dim = math.ceil(depth * ASPECT / HEAD_DIM) * HEAD_DIM
    return dim, dim // HEAD_DIM, 12 * depth * dim * dim + vocab * dim


def speedrun() -> dict:
    """The worked example: the speedrun's model, tokens, batch and steps, by nanochat's rules."""
    dim, heads, params = size_of(SPEEDRUN["depth"], SPEEDRUN["vocab"])
    tokens = SPEEDRUN["ratio"] * params
    ref = SPEEDRUN["ratio"] * size_of(12, SPEEDRUN["vocab"])[2]
    batch = 2 ** round(math.log2(B_REF * (tokens / ref) ** POWER))
    return {"dim": dim, "heads": heads, "params": params, "tokens": tokens, "batch": batch, "steps": tokens // batch}


def run(lr_multiplier, show) -> dict | str:
    """The demo: the learner's ModernGPT at nanochat's size for DEPTH, trained for STEPS steps of BATCH windows of
    BLOCK tokens of the shards of llm-data-pipeline (written first with the course's versions when they are missing),
    with AdamW at PEAK times the learner's lr_multiplier. `show` gets the step and the loss every 10 steps, and the loss
    of each of those steps ("fresh"). "run" when a module of the learner stops with an error; else the losses, and the
    validation loss per token and per character."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    tools = run_code("from llm.modern_gpt import ModernGPT\nfrom llm.token_windows import TokenWindows")
    if tools.error is not None:
        return "run"
    if not (D.folder() / "info.json").exists():
        written = D.write_demo(run_code(D.EXERCISES["write-shards"].solution).names["write_shards"])
        if not isinstance(written, dict):
            return "run"
    info = json.loads((D.folder() / "info.json").read_text(encoding="utf-8"))
    files = [str(D.folder() / name) for name in info["files"]]
    dtype = np.dtype(info["dtype"])
    train = tools.names["TokenWindows"](files[:-1], BLOCK, dtype)
    val = tools.names["TokenWindows"](files[-1:], BLOCK, dtype)
    dim, heads, _ = size_of(DEPTH, D.B.VOCAB)
    device = M.plan()["device"]
    torch.manual_seed(0)
    model = tools.names["ModernGPT"](D.B.VOCAB, dim, heads, heads, DEPTH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.1)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(0))
    batches = iter(loader)
    losses, start = [], time.perf_counter()
    for step in range(STEPS):
        try:
            xb, yb = next(batches)
        except StopIteration:
            batches = iter(loader)
            xb, yb = next(batches)
        try:
            factor = float(lr_multiplier(step, STEPS, SCHEDULE["warmup"], SCHEDULE["warmdown"], SCHEDULE["final"]))
        except Exception:
            return "run"
        for group in opt.param_groups:
            group["lr"] = PEAK * factor
        with M.on_device(device):
            logits = model(xb.to(device))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.to(device).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if step % 10 == 9:
            show({"step": step + 1, "steps": STEPS, "loss": sum(losses[-10:]) / 10, "seconds": time.perf_counter() - start,
                  "fresh": list(zip(range(step - 8, step + 2), losses[-10:]))})   # its last 10 steps, for the live chart
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(DataLoader(val, batch_size=32)):
            if i == 20:
                break
            with M.on_device(device):
                logits = model(xb.to(device))
            total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.to(device).reshape(-1), reduction="sum").item()
            count += yb.numel()
    per_token = total / count
    bytes_per_token = D.MEASURED["chars"] / info["ids"]   # TinyStories here is ASCII: a character is a byte
    curve = [sum(losses[i:i + 10]) / 10 for i in range(0, STEPS, 10)]
    return {"curve": curve, "val": per_token, "per_char": per_token / bytes_per_token, "device": device,
            "seconds": time.perf_counter() - start, "params": sum(p.numel() for p in model.parameters()), "dim": dim, "heads": heads}


def nanochat_multiplier(step: int, steps: int) -> float:
    """nanochat's schedule with its defaults, for the picture (the course's solution of exercise 3)."""
    warmdown = round(SCHEDULE["warmdown"] * steps)
    if step < SCHEDULE["warmup"]:
        return (step + 1) / SCHEDULE["warmup"]
    if step <= steps - warmdown:
        return 1.0
    progress = (steps - step) / warmdown
    return progress + (1 - progress) * SCHEDULE["final"]


def cosine_multiplier(step: int, steps: int) -> float:
    """A warmup of the same length, then a half cosine down to 0, as in opt-lr-schedules: for the picture."""
    warmup = SCHEDULE["warmup"]
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))


def schedule_picture(labels: dict) -> str:
    """The learning rate, as a share of its peak, over the STEPS steps of the demo: nanochat's schedule solid, a cosine
    schedule dashed; both hyperparameters, so both in that role."""
    from everything_ai import chart
    xs = list(range(STEPS + 1))
    panel = chart.Panel((0, STEPS), (0, 1.12), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    panel.line(xs, [cosine_multiplier(i, STEPS) for i in xs], "hyper", 1.6, dashed=True)
    panel.line(xs, [nanochat_multiplier(i, STEPS) for i in xs], "hyper", 2.4)
    panel.text(70, 1.05, labels["nanochat"], "hyper", "start")
    panel.text(196, 0.12, labels["cosine"], "hyper", "end")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=230)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


def course_curve() -> list:
    """The course's run as (step, loss): the mean of every 10 steps, at the last of them, as curve_picture draws it."""
    return [(10 * (k + 1), v) for k, v in enumerate(MEASURED["curve"])]


def curve_picture(curves: list, labels: dict) -> str:
    """The training loss per token, averaged over every 10 steps: `curves` is a list of (values, dashed), the learner's
    run solid and the course's dashed."""
    from everything_ai import chart
    top = max(max(v) for v, _ in curves)
    panel = chart.Panel((0, STEPS), (2.0, math.ceil(top)), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    for values, dashed in curves:
        xs = [10 * (k + 1) for k in range(len(values))]
        panel.line(xs, values, "loss", 1.6 if dashed else 2.4, dashed=dashed)
    panel.text(STEPS - 4, math.ceil(top) - 0.4, labels["key"], "loss", "end")
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=240)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- measured: the course's run of the demo with its own versions and merge rules, on the CPU, 2 threads ---
# Its validation loss per token and per character, the seconds, the parameters, and the training loss every 10 steps.
MEASURED = {"val": 2.4149, "per_char": 0.823, "seconds": 232, "params": 3791849,
            "curve": [6.943, 5.392, 4.328, 3.798, 3.614, 3.502, 3.327, 3.244, 3.255, 3.177, 3.059, 3.036, 2.978, 2.923, 2.867,
                      2.853, 2.791, 2.748, 2.757, 2.691, 2.638, 2.553, 2.605, 2.576, 2.499, 2.517, 2.438, 2.445, 2.467, 2.449]}
# The course's GPT on tokens of gpt-train-bpe, CPU plan, seed 0: 10 epochs, its best validation loss per character.
BPE_RUN = {"minutes": 34.9, "per_char": 0.737, "epochs": 10}
# --- end of measured ---

REQUIRED = ["features", "size", "lr-multiplier", "recipe"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"
_SORT_TEXTS = {
    "missing": {"en": "I cannot see your choices. Run the cell with the parts again.",
                "nl": "Ik zie je keuzes niet. Voer de cel met de onderdelen opnieuw uit."},
    "empty": {"en": "Pick a place for each part.", "nl": "Kies voor elk onderdeel een plek."},
    "unfinished": {"en": "Everything you picked so far is right. Pick a place for the rest too ({n} to go).",
                   "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een plek (nog {n})."},
    "wrong": {"en": "{item} goes somewhere else.", "nl": "{item} hoort ergens anders."},
    "correct": {"en": "All {n} are in place. Your ModernGPT already has the skeleton of nanochat's GPT; nanochat adds changes that proved worth keeping in its own runs and those of others, and a new optimizer.",
                "nl": "Alle {n} staan goed. Je ModernGPT heeft het geraamte van de GPT van nanochat al; nanochat voegt veranderingen toe die in zijn eigen runs en die van anderen de moeite waard bleken, en een nieuwe optimizer."},
}

# --- exercise 2: size ------------------------------------------------------------------------------------------------

_S_HEAD = "import math\n\n\ndef size(depth, vocab):\n"
_S_DIM = "    dim = math.ceil(depth * 64 / 128) * 128\n"
_S_REST = "    heads = int(dim / 128)\n    params = 12 * depth * dim * dim + vocab * dim\n    return dim, heads, params\n"
_S_START = (_S_HEAD + "    # depth times 64, rounded up to a multiple of 128\n    dim = ...\n"
            "    # one head per 128 numbers\n    heads = ...\n"
            "    # 12 d^2 per block, and the output head of vocab times d\n    params = ...\n    return dim, heads, params\n")


def _s(dim: str = _S_DIM, rest: str = _S_REST) -> str:
    return _S_HEAD + dim + rest


def _rounded(v) -> str:
    """A value for the messages of lr_multiplier: a float rounded to 4 decimals, so 0.118 and not 0.11800000000000001."""
    if isinstance(v, (bool, int, str)):
        return repr(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


# --- exercise 3: lr_multiplier ---------------------------------------------------------------------------------------

_L_BODY = ("def lr_multiplier(step, steps, warmup, warmdown_ratio, final):\n    warmdown = round(warmdown_ratio * steps)\n"
           "    if step < warmup:\n        return (step + 1) / warmup\n    if step <= steps - warmdown:\n        return 1.0\n"
           "    progress = (steps - step) / warmdown\n    return progress + (1 - progress) * final\n")
_L_START = ("def lr_multiplier(step, steps, warmup, warmdown_ratio, final):\n    warmdown = round(warmdown_ratio * steps)\n"
            "    # the warmup: from 1 / warmup at step 0 up to 1\n    ...\n"
            "    # then 1, up to and including step `steps - warmdown`\n    ...\n"
            "    # the warmdown: in a straight line from 1 down to `final` at the end\n    progress = ...\n"
            "    return progress + (1 - progress) * final\n")

_CASES_SIZE = [(5, 1257), (4, 1257), (1, 50257), (26, 65536), (3, 0)]
_CASES_LR = [(30, 100, 20, 0.5, 0.1), (50, 100, 20, 0.5, 0.1), (51, 100, 20, 0.5, 0.1), (99, 100, 20, 0.5, 0.1),
             (0, 100, 20, 0.5, 0.1), (19, 100, 20, 0.5, 0.1), (200, 300, 40, 0.65, 0.05), (105, 300, 40, 0.65, 0.05), (90, 101, 10, 0.65, 0.05)]

EXERCISES = {
    "features": Sort(
        name="features",
        groups={
            "yours": {"en": "Your ModernGPT has it too", "nl": "Je ModernGPT heeft het ook"},
            "new": {"en": "New in nanochat", "nl": "Nieuw in nanochat"},
        },
        items={
            "rope": {"en": "**RoPE**: rotary embeddings, and no table of places", "nl": "**RoPE**: rotary embeddings, en geen tabel van plekken"},
            "qknorm": {"en": "**QK-norm**: the queries and keys are normalized before attention", "nl": "**QK-norm**: de queries en keys worden genormaliseerd vóór de attention"},
            "rms": {"en": "**RMSNorm** instead of layer normalization", "nl": "**RMSNorm** in plaats van layer normalization"},
            "relu2": {"en": "**ReLU squared** in the MLP: ReLU, then squared", "nl": "**ReLU in het kwadraat** in de MLP: eerst ReLU, dan in het kwadraat"},
            "gqa": {"en": "**Grouped-query attention**, as an option: fewer heads for keys and values than for queries", "nl": "**Grouped-query attention**, als optie: minder heads voor keys en values dan voor queries"},
            "muon": {"en": "**Muon**, an optimizer for the weight matrices of the blocks; AdamW for the rest", "nl": "**Muon**, een optimizer voor de weight-matrices van de blocks; AdamW voor de rest"},
            "head": {"en": "**An output head of its own**, not the table of token embeddings used again", "nl": "**Een eigen output head**, niet nog een keer de tabel van token-embeddings"},
            "window": {"en": "**Sliding-window attention** in three of every four blocks: a token sees only the last quarter of the context, 512 of 2,048 tokens", "nl": "**Sliding-window attention** in drie van elke vier blocks: een token ziet alleen het laatste kwart van de context, 512 van de 2.048 tokens"},
            "bos": {"en": "**A BOS token** at the start of every document, instead of end-of-text after it", "nl": "**Een BOS-token** aan het begin van elk document, in plaats van end-of-text erna"},
        },
        solution={"rope": "yours", "qknorm": "new", "rms": "yours", "relu2": "new", "gqa": "yours", "muon": "new",
                  "head": "yours", "window": "new", "bos": "new"},
        nudge={
            "rope": {"en": "Which change of [Project: upgrade your GPT to a modern LLM](/course/open/llm-modern-gpt?lang=en) turns queries and keys by their place?",
                     "nl": "Welke verandering uit [Project: je GPT een modern LLM maken](/course/open/llm-modern-gpt?lang=nl) draait queries en keys naar hun plek?"},
            "qknorm": {"en": "Does your attention normalize the queries and keys, or only the input of the block?",
                       "nl": "Normaliseert je attention de queries en keys, of alleen de input van het block?"},
            "rms": {"en": "Which normalization did you write for your ModernGPT?", "nl": "Welke normalisatie schreef je voor je ModernGPT?"},
            "relu2": {"en": "Which activation does the MLP of your ModernGPT use?", "nl": "Welke activation function gebruikt de MLP van je ModernGPT?"},
            "gqa": {"en": "How many key and value heads does your ModernGPT have, next to its query heads?", "nl": "Hoeveel key- en value-heads heeft je ModernGPT, naast zijn query heads?"},
            "muon": {"en": "Which optimizer trains your ModernGPT?", "nl": "Welke optimizer traint je ModernGPT?"},
            "head": {"en": "Is the output head of your GPT an `nn.Linear` of its own, or the embedding table?", "nl": "Is de output head van je GPT een eigen `nn.Linear`, of de embedding-tabel?"},
            "window": {"en": "Does a token in your ModernGPT see every token before it in the window?", "nl": "Ziet een token in je ModernGPT elk token ervoor in het venster?"},
            "bos": {"en": "Where does your tokenizer put its special token between two stories?", "nl": "Waar zet je tokenizer zijn special token tussen twee verhalen?"},
        },
        texts=_SORT_TEXTS,
    ),
    "size": FunctionBox(
        name="size",
        function="size",
        start=_S_START,
        solution=_s(),
        visible=[(20, 32768)],
        hidden=_CASES_SIZE,
        first=[_DOTS],
        mistakes=[
            Mistake("no-round", wrong=_s(dim="    dim = depth * 64\n"),
                    en=_CALL + ": `depth * 64` is not always a multiple of 128, so the heads do not come out whole. Round it up first: `math.ceil(depth * 64 / 128) * 128`.",
                    nl=_CALL_NL + ": `depth * 64` is niet altijd een veelvoud van 128, dus de heads komen niet heel uit. Rond het eerst naar boven af: `math.ceil(depth * 64 / 128) * 128`."),
            Mistake("round-down", wrong=_s(dim="    dim = depth * 64 // 128 * 128\n"),
                    en=_CALL + ": you round down, so an odd depth gets a narrower model, and depth 1 none at all. nanochat rounds up: `math.ceil(depth * 64 / 128) * 128`.",
                    nl=_CALL_NL + ": je rondt naar beneden af, dus een oneven diepte krijgt een smaller model, en diepte 1 helemaal geen. nanochat rondt naar boven af: `math.ceil(depth * 64 / 128) * 128`."),
            Mistake("no-head", wrong=_s(rest=_S_REST.replace(" + vocab * dim", "")),
                    en=_CALL + ": your count leaves out the output head, `vocab * dim` weights. nanochat counts it: the matrices of the blocks and the head.",
                    nl=_CALL_NL + ": je telling laat de output head weg, `vocab * dim` weights. nanochat telt hem mee: de matrices van de blocks en de head."),
            Mistake("two-tables", wrong=_s(rest=_S_REST.replace(" + vocab * dim", " + 2 * vocab * dim")),
                    en=_CALL + ": you also count the table of token embeddings. nanochat's scaling parameters leave it out: they are the matrices of the blocks and the head, the count that in its runs best predicted how the loss falls as the model grows.",
                    nl=_CALL_NL + ": je telt de tabel van token-embeddings ook mee. De scaling-parameters van nanochat laten hem weg: het zijn de matrices van de blocks en de head, de telling die in zijn runs het best voorspelde hoe de loss daalt als het model groeit."),
            Mistake("attention-only", wrong=_s(rest=_S_REST.replace("12 * depth", "4 * depth")),
                    en=_CALL + ": 4 d^2 per block are the queries, keys, values and output of attention. The MLP adds 8 d^2: a matrix from d to 4d and one back. 12 d^2 per block.",
                    nl=_CALL_NL + ": 4 d^2 per block zijn de queries, keys, values en output van de attention. De MLP voegt 8 d^2 toe: een matrix van d naar 4d en een terug. 12 d^2 per block."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the width rounded up to a multiple of 128, the heads, and the parameters.",
                      "nl": "Vervang de drie `...`: de breedte, naar boven afgerond op een veelvoud van 128, de heads, en de parameters."},
            "wrong": {"en": _CALL + ". `depth * 64` rounded up to a multiple of 128, one head per 128, and `12 * depth * dim * dim + vocab * dim`?",
                      "nl": _CALL_NL + ". `depth * 64` naar boven afgerond op een veelvoud van 128, één head per 128, en `12 * depth * dim * dim + vocab * dim`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function round up, and use `depth` and `vocab`, and no fixed numbers for them?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Rondt je functie naar boven af, en gebruikt hij `depth` en `vocab`, en geen vaste getallen daarvoor?"},
            "correct": {"en": "Your function sizes a model the way nanochat does: one number, the depth, sets the width, the heads and the parameters.",
                        "nl": "Je functie bepaalt de maat van een model zoals nanochat dat doet: één getal, de diepte, legt de breedte, de heads en de parameters vast."},
        },
    ),
    "lr-multiplier": FunctionBox(
        name="lr-multiplier",
        function="lr_multiplier",
        start=_L_START,
        solution=_L_BODY,
        visible=[(10, 100, 20, 0.5, 0.1)],
        hidden=_CASES_LR,
        show=_rounded,
        first=[_DOTS],
        mistakes=[
            Mistake("warmup-zero", wrong=_L_BODY.replace("return (step + 1) / warmup", "return step / warmup"),
                    en=_CALL + ": at step 0 your learning rate is 0, so the first step changes nothing. nanochat counts the step itself: `(step + 1) / warmup`.",
                    nl=_CALL_NL + ": bij stap 0 is je learning rate 0, dus de eerste stap verandert niets. nanochat telt de stap zelf mee: `(step + 1) / warmup`."),
            Mistake("to-zero", wrong=_L_BODY.replace("    return progress + (1 - progress) * final\n", "    return progress\n"),
                    en=_CALL + ": your warmdown goes to 0, and nanochat's to `final` times the peak. Mix the two: `progress + (1 - progress) * final`.",
                    nl=_CALL_NL + ": je warmdown gaat naar 0, en die van nanochat naar `final` keer de piek. Meng de twee: `progress + (1 - progress) * final`."),
            Mistake("upside-down", wrong=_L_BODY.replace("progress = (steps - step) / warmdown", "progress = (step - (steps - warmdown)) / warmdown"),
                    en=_CALL + ": your progress grows during the warmdown, so the learning rate goes up again. It is the share of the warmdown that is still left: `(steps - step) / warmdown`.",
                    nl=_CALL_NL + ": je voortgang groeit tijdens de warmdown, dus de learning rate gaat weer omhoog. Het is het deel van de warmdown dat nog over is: `(steps - step) / warmdown`."),
            Mistake("cosine", wrong=_L_BODY.replace("    progress = (steps - step) / warmdown\n",
                                                    "    import math\n    progress = 0.5 * (1 + math.cos(math.pi * (1 - (steps - step) / warmdown)))\n"),
                    en=_CALL + ": your warmdown follows a half cosine, as in [Learning-rate schedules](/course/open/opt-lr-schedules?lang=en). nanochat goes down in a straight line: `(steps - step) / warmdown`.",
                    nl=_CALL_NL + ": je warmdown volgt een halve cosinus, zoals in [Learning-rate schedules](/course/open/opt-lr-schedules?lang=nl). nanochat gaat in een rechte lijn omlaag: `(steps - step) / warmdown`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the warmup, the plateau, and the progress of the warmdown.",
                      "nl": "Vervang de drie `...`: de warmup, het plateau, en de voortgang van de warmdown."},
            "wrong": {"en": _CALL + ". From `1 / warmup` at step 0 up to 1 in the warmup, 1 up to and including step `steps - warmdown`, and then from 1 in a straight line to `final`?",
                      "nl": _CALL_NL + ". Van `1 / warmup` bij stap 0 tot 1 in de warmup, 1 tot en met stap `steps - warmdown`, en dan van 1 in een rechte lijn naar `final`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function also give the plateau and the warmdown: 1 up to and including step `steps - warmdown`, then in a straight line down to `final`?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Geeft je functie ook het plateau en de warmdown: 1 tot en met stap `steps - warmdown`, en dan in een rechte lijn omlaag naar `final`?"},
            "correct": {"en": "Your function gives nanochat's schedule: up in a straight line, a plateau at the peak, and down in a straight line to a small fraction of the peak.",
                        "nl": "Je functie geeft het schema van nanochat: in een rechte lijn omhoog, een plateau op de piek, en in een rechte lijn omlaag naar een klein deel van de piek."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("features", {"rope": "yours", "qknorm": "new", "rms": "yours", "relu2": "new", "gqa": "yours", "muon": "new",
                  "head": "yours", "window": "new", "bos": "new"}, "correct", None),
    ("features", {"rope": "yours", "qknorm": "yours"}, "wrong", "wrong:qknorm"),
    ("features", {"rope": "yours"}, "wrong", "unfinished"),
    ("size", _S_START, "empty", None),
    ("size", _s(), "correct", None),
    ("size", _S_HEAD + "    dim = -(-depth * 64 // 128) * 128\n" + _S_REST, "correct", None),
    ("size", _s(dim="    dim = depth * 64\n"), "wrong", "no-round"),
    ("size", _s(dim="    dim = depth * 64 // 128 * 128\n"), "wrong", "round-down"),
    ("size", _s(rest=_S_REST.replace(" + vocab * dim", "")), "wrong", "no-head"),
    ("size", _s(rest=_S_REST.replace(" + vocab * dim", " + 2 * vocab * dim")), "wrong", "two-tables"),
    ("size", _s(rest=_S_REST.replace("12 * depth", "4 * depth")), "wrong", "attention-only"),
    ("lr-multiplier", _L_START, "empty", None),
    ("lr-multiplier", _L_BODY, "correct", None),
    ("lr-multiplier", _L_BODY.replace("return (step + 1) / warmup", "return step / warmup"), "wrong", "warmup-zero"),
    ("lr-multiplier", _L_BODY.replace("if step <= steps - warmdown", "if step < steps - warmdown"), "correct", None),
    ("lr-multiplier", _L_BODY.replace("    return progress + (1 - progress) * final\n", "    return progress\n"), "wrong", "to-zero"),
    ("lr-multiplier", _L_BODY.replace("progress = (steps - step) / warmdown", "progress = (step - (steps - warmdown)) / warmdown"), "wrong", "upside-down"),
    ("lr-multiplier", _L_BODY.replace("    progress = (steps - step) / warmdown\n", "    import math\n    progress = 0.5 * (1 + math.cos(math.pi * (1 - (steps - step) / warmdown)))\n"), "wrong", "cosine"),
]
