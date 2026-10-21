# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-reasoning-from-scratch: steps_answer, the answer to a multiplication with its steps written
out; read_product, the checker that reads the product a model wrote; and two open questions. The demo trains the
learner's ModernGPT from scratch twice on multiplications of numbers of 1 to 3 digits, the same model, steps, data and
seed: once on the answer alone and once on the answer with its steps, with the learner's sft_example, sft_batch and
pretrain_step of the project, and measures both on 200 held-out pairs of 2, 3 and 4 digits each. The course's runs of
the demo, three seeds on each plan (data/course-reasoning-runs.json), stand in until the learner runs it, and its
models of seed 0 on the CPU plan (data/course-arith-steps.pt and data/course-arith-plain.pt) answer in their place."""
import importlib.util
import json
import math
import random
import re
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SFT = _checks("llm-sft", "everything_ai_checks_llm_sft_for_reasoning")
P, S = SFT.P, SFT.S   # the capstone (the device, the step of the run) and the scaling law (the schedule)
# Only what the demo imports: the ModernGPT and its parts, the step of the capstone, and the example and batch of SFT.
BUILDS_ON = tuple(m for m in SFT.BUILDS_ON + SFT.SAVES
                  if m[1] in ("modern_parts", "modern_attention", "modern_gpt", "pretrain_step", "sft_example", "sft_batch"))
projects.builds_on(*BUILDS_ON)
SAVES = (("llm", "steps_answer", "llm-reasoning-from-scratch", "steps-answer"),
         ("llm", "read_product", "llm-reasoning-from-scratch", "read-product"))

CHARS = "".join(sorted("0123456789*+=,>"))   # the characters, sorted: character i has id i, as in gpt-train-chars
END = len(CHARS)                  # the id of the end-of-text, after the characters
SIZES = [END + 1, 128, 4, 2, 4]   # the ModernGPT: vocabulary, d, query heads, key and value heads, blocks
TRAIN = (1, 2, 3)                 # digits per number in training: both numbers of a pair have as many
TESTS = (2, 3, 4)                 # held out, HELD pairs each; 4 digits is longer than any number in training
HELD = 200
SEED_TEST = 1234                  # the held-out pairs, as the research report drew them (its multiply.py)
LR, WEIGHT_DECAY, BETAS = 2e-3, 0.1, (0.9, 0.98)   # AdamW, as in the research report, at LR times the course's schedule
BATCH = 64                        # pairs per step, on every device
PROBE = 20                        # steps of each format the probe times, each on a new model
RUNS = {"cpu": [("steps", 1800), ("plain", 1800)],
        "gpu": [("steps", 1800), ("plain", 1800), ("plain", 7400)]}   # a GPU also runs answer only for about the time
#                                 the CPU takes for the run with steps: four times as many steps
SNAPSHOTS = 5                     # moments of a run after its start, at every fifth: for the replay of the run
SNAP = 50                         # the first SNAP held-out 3x3 pairs that each moment measures
NAME = "arith"                    # the learner's run in work/projects/llm/: arith_steps.pt, arith_plain.pt, arith.json
COURSE = "data/course-reasoning-runs.json"   # the course's runs of the demo
COURSE_MODELS = {"steps": "data/course-arith-steps.pt", "plain": "data/course-arith-plain.pt"}   # seed 0, CPU plan


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def listed(values: list, lang: str) -> str:
    """Numbers in a sentence: 150, 152 and 160, or in Dutch 150, 152 en 160."""
    words = [number(v, 0, lang) if isinstance(v, (int, float)) else str(v) for v in values]
    if len(words) < 2:
        return "".join(words)
    return ", ".join(words[:-1]) + (" and " if lang == "en" else " en ") + words[-1]


# --- the data ----------------------------------------------------------------------------------------------------------

def draw(n: int, rng: random.Random) -> int:
    """A number of n digits: from 10 ** (n - 1) to 10 ** n - 1, and from 0 to 9 for one digit."""
    return rng.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1)


def held_out() -> dict:
    """HELD pairs of every length of TESTS, drawn as the research report drew them: 1 digit first, which is not held
    out (there are only 100 such pairs), then 2, 3 and 4 digits, from one random generator."""
    rng = random.Random(SEED_TEST)
    out = {n: [(draw(n, rng), draw(n, rng)) for _ in range(HELD)] for n in (1, 2, 3, 4)}
    return {n: out[n] for n in TESTS}


HELD_OUT = held_out()
_HELD_SET = {pair for pairs in HELD_OUT.values() for pair in pairs}


def pair(rng: random.Random) -> tuple:
    """A training pair: as many digits for both numbers, 1, 2 or 3, never a held-out pair."""
    n = rng.choice(TRAIN)
    while True:
        a, b = draw(n, rng), draw(n, rng)
        if (a, b) not in _HELD_SET:
            return a, b


def prompt(a: int, b: int) -> str:
    return f"{a}*{b}="


def encode(text: str) -> list:
    """The ids of a text: character i of CHARS has id i. A ValueError for any other character."""
    return [CHARS.index(c) for c in text]


def decode(ids: list) -> str:
    """The text of ids, up to the first end-of-text."""
    out = []
    for i in ids:
        if i == END:
            break
        out.append(CHARS[i])
    return "".join(out)


def answer(a: int, b: int, fmt: str, steps_answer) -> str:
    """The answer the training data has for a * b: the product alone ("plain"), or with its steps ("steps")."""
    return steps_answer(a, b) if fmt == "steps" else f"{a * b}"


# --- the solutions, which the course's runs use ------------------------------------------------------------------------

_A_HEAD = "def steps_answer(a, b):\n"
_A_DIGITS = "    digits = [ord(c) - ord(\"0\") for c in reversed(str(b))]\n"
_A_CODE = (_A_HEAD + "    # the digits of b, from the right, as numbers\n" + _A_DIGITS +
           "    # 1. a partial product for every digit\n"
           "    parts = []\n"
           "    for d in digits:\n"
           "        parts.append(f\"{a}*{d}={a * d}\")\n"
           "    # 2. the running sums: each next partial product times 10 ** i\n"
           "    total = a * digits[0]\n"
           "    sums = []\n"
           "    for i in range(1, len(digits)):\n"
           "        add = a * digits[i] * 10 ** i\n"
           "        sums.append(f\"{total}+{add}={total + add}\")\n"
           "        total = total + add\n"
           "    # 3. the partial products, the sums, and the product\n"
           "    return \",\".join(parts) + \">\" + \",\".join(sums) + \">\" + f\"{total}\"\n")

_R_HEAD = "def read_product(text):\n"
_R_CODE = (_R_HEAD + "    last = text.split(\">\")[-1]\n"
           "    if last.isdigit():\n"
           "        return last\n"
           "    return None\n")


PARAMS = S.M.n_params(*SIZES)   # the parameters of the demo's ModernGPT, as llm-modern-gpt counts them
START_LOSS = math.log(SIZES[0])   # the loss of a new model that gives every token the same chance


def course_functions() -> tuple:
    """The course's steps_answer and read_product."""
    return run_code(_A_CODE).names["steps_answer"], run_code(_R_CODE).names["read_product"]


# --- the demo ------------------------------------------------------------------------------------------------------------

def tools() -> dict | str:
    """The learner's modules the demo needs, or "run" when one stops with an error."""
    ran = run_code("from llm.modern_gpt import ModernGPT\nfrom llm.pretrain_step import pretrain_step\n"
                   "from llm.sft_example import sft_example\nfrom llm.sft_batch import sft_batch")
    return "run" if ran.error is not None else ran.names


def where() -> dict:
    return P.where()


def build(ModernGPT):
    return ModernGPT(*SIZES)


def _example(a: int, b: int, fmt: str, steps_answer, sft_example) -> tuple:
    """One pair as an example of SFT: the ids of its prompt and its answer, with the end-of-text, and their labels."""
    return sft_example(encode(prompt(a, b)), encode(answer(a, b, fmt, steps_answer)), END)


def longest(n: int, fmt: str, steps_answer) -> int:
    """The tokens an answer to an n-digit pair may take: the longest answer of that length, its end-of-text and one
    more, as in the research report."""
    big = 10 ** n - 1
    return len(answer(big, big, fmt, steps_answer)) + 2


def greedy(model, pairs: list, new: int) -> list:
    """What the model writes after the prompt of every pair, greedily, all at once with its caches: the prompts of
    pairs of one length have one length, so the batch needs no padding. The text of each, up to its end-of-text."""
    import torch
    device = next(model.parameters()).device
    x = torch.tensor([encode(prompt(a, b)) for a, b in pairs], device=device)
    caches = [{} for _ in model.blocks]
    model.eval()
    out = []
    with torch.no_grad(), P._on(device.type):
        logits = model(x, caches)
        done = torch.zeros(len(pairs), dtype=torch.bool, device=device)
        for _ in range(new):
            nxt = logits[:, -1].argmax(-1)
            out.append(nxt)
            done = done | (nxt == END)
            if bool(done.all()):
                break
            logits = model(nxt[:, None], caches)
    rows = torch.stack(out, dim=1).tolist() if out else [[] for _ in pairs]
    return [decode(r) for r in rows]


def measure(model, fmt: str, steps_answer, read_product, n: int, pairs: list | None = None) -> dict | str:
    """The model's greedy answers to the held-out pairs of n digits (or `pairs`), each read by the learner's
    read_product and compared with the product as text, as the research report compared it: how many are right, the
    texts, the tokens the answers took with their end-of-text, and the seconds. "read" when read_product stops with an
    error."""
    pairs = HELD_OUT[n] if pairs is None else pairs
    start = time.perf_counter()
    texts = greedy(model, pairs, longest(n, fmt, steps_answer))
    seconds = time.perf_counter() - start
    right = 0
    for (a, b), text in zip(pairs, texts):
        try:
            got = read_product(text)
        except Exception:
            return "read"
        right += got is not None and str(got) == f"{a * b}"
    return {"right": right, "total": len(pairs), "texts": texts, "tokens": sum(len(t) + 1 for t in texts),
            "seconds": seconds}


def train(fmt: str, steps: int, seed: int, steps_answer, read_product, t: dict, show, snap: bool = True) -> dict | str:
    """Train a new ModernGPT on `fmt` for `steps` steps with the learner's sft_example, sft_batch and pretrain_step:
    AdamW at LR times the schedule of llm-open-recipes, BATCH pairs a step drawn with `seed`, never a held-out pair.
    `show` hears a step every second, with every (step, loss) since it last heard ("fresh"), for a live chart. With
    `snap`, before the first step and after every steps / SNAPSHOTS steps the model answers the first SNAP held-out
    3x3 pairs, for the replay of the run. A string when it stops: "steps" (steps_answer), "example" (sft_example),
    "batch" (sft_batch), "run" (pretrain_step or the model), "read" (read_product), "memory", "diverged"."""
    import torch
    w = where()
    torch.manual_seed(seed)
    try:
        model = build(t["ModernGPT"]).to(w["device"])
    except Exception:
        return "run"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)
    rng = random.Random(seed)
    marks = {round(steps * k / SNAPSHOTS) for k in range(1, SNAPSHOTS + 1)}
    snapshots, losses, recent, fresh, seconds, tokens, first = [], [], [], [], 0.0, 0, None
    last_show = time.perf_counter()

    def moment(step):
        m = measure(model, fmt, steps_answer, read_product, 3, HELD_OUT[3][:SNAP])
        if isinstance(m, str):
            return m
        model.train()
        return {"step": step, "right": m["right"], "total": m["total"], "text": m["texts"][0]}

    if snap:
        at_start = moment(0)
        if isinstance(at_start, str):
            return at_start
        snapshots.append(at_start)
    model.train()
    for step in range(steps):
        start = time.perf_counter()
        made = []
        for _ in range(BATCH):
            a, b = pair(rng)
            try:
                text = answer(a, b, fmt, steps_answer)
                ids = encode(prompt(a, b)), encode(text)
            except Exception:
                return "steps"
            try:
                made.append(t["sft_example"](ids[0], ids[1], END))
            except Exception:
                return "example"
        try:
            x, y = t["sft_batch"](made, END)
        except Exception:
            return "batch"
        try:
            with P._on(w["device"]):
                loss = float(t["pretrain_step"](model, opt, x.to(w["device"]), y.to(w["device"]),
                                                LR * S.multiplier(step, steps), w["device"], False))
        except Exception as e:
            return "memory" if "out of memory" in str(e).lower() else "run"
        if not math.isfinite(loss):
            return "diverged"
        P._sync(w["device"])
        seconds += time.perf_counter() - start
        tokens += sum(len(ids) - 1 for ids, _ in made)   # the tokens the model reads, without the padding
        recent.append(loss)
        fresh.append((step + 1, loss))
        first = loss if step == 0 else first
        if (step + 1) % 50 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": sum(recent) / len(recent)})
            recent = []
        if snap and step + 1 in marks:
            m = moment(step + 1)
            if isinstance(m, str):
                return m
            snapshots.append(m)
        if time.perf_counter() - last_show > 1 or step + 1 == steps:
            show({"phase": "train", "format": fmt, "step": step + 1, "steps": steps, "loss": loss, "seconds": seconds,
                  "fresh": fresh})
            fresh, last_show = [], time.perf_counter()
    model.eval()
    return {"model": model, "format": fmt, "steps": steps, "seed": seed, "seconds": seconds, "tokens": tokens,
            "first": first, "losses": losses, "snapshots": snapshots, "device": w["device"], "name": w["name"]}


def evaluate(model, fmt: str, steps_answer, read_product) -> dict | str:
    """The model on all held-out pairs of every length of TESTS: per length its right answers, the first three texts,
    and the answer tokens and seconds."""
    out = {}
    for n in TESTS:
        m = measure(model, fmt, steps_answer, read_product, n)
        if isinstance(m, str):
            return m
        out[str(n)] = {"right": m["right"], "total": m["total"], "texts": m["texts"][:3], "tokens": m["tokens"],
                       "seconds": round(m["seconds"], 2)}
    return out


def run_one(fmt: str, steps: int, seed: int, steps_answer, read_product, t: dict, show, snap: bool = True) -> dict | str:
    """One run of the demo: train, then measure on the held-out pairs."""
    r = train(fmt, steps, seed, steps_answer, read_product, t, show, snap)
    if isinstance(r, str):
        return r
    show({"phase": "measure", "format": fmt, "steps": steps})
    acc = evaluate(r["model"], fmt, steps_answer, read_product)
    if isinstance(acc, str):
        return acc
    return {**r, "accuracy": acc}


def kept(r: dict) -> dict:
    """What a run keeps in its JSON: everything but the model, the seconds rounded."""
    out = {k: v for k, v in r.items() if k != "model"}
    out["seconds"] = round(out["seconds"], 1)
    out["losses"] = [{"step": l["step"], "loss": round(l["loss"], 4)} for l in out["losses"]]
    return out


def probe(steps_answer, t: dict, kind: str | None = None) -> dict | str:
    """How long the runs of the plan take on this computer: PROBE steps of each format on a new model; the seconds a
    step, and the minutes of all runs of the plan with their measuring. A string as train gives when it stops."""
    kind = kind or where()["kind"]
    per = {}
    for fmt in dict.fromkeys(f for f, _ in RUNS[kind]):
        r = train(fmt, PROBE, 0, steps_answer, lambda text: None, t, lambda d: None, snap=False)
        if isinstance(r, str):
            return r
        per[fmt] = r["seconds"] / PROBE
    minutes = sum(steps * per[fmt] for fmt, steps in RUNS[kind]) / 60 * (1 + MEASURING)
    return {"seconds": per, "minutes": minutes, "kind": kind}


MEASURING = 0.04   # what the moments of the replay and the measuring after a run add to its time, about


def run(steps_answer, read_product, show, kind: str | None = None) -> dict | str:
    """The demo: every run of the plan of this device (RUNS), one after the other, with seed 0, then the runs are kept
    in work/projects/llm/. `kind` is "cpu" or "gpu": that of the device when None. `show` hears {"phase": "start",
    "format", "steps", "i", "n"} before a run, its steps as train gives them, and {"phase": "measure"} after its
    training. A string when it stops, as train gives it."""
    t = tools()
    if t == "run":
        return "run"
    kind = kind or where()["kind"]
    begin = time.perf_counter()
    runs = []
    for i, (fmt, steps) in enumerate(RUNS[kind], 1):
        show({"phase": "start", "format": fmt, "steps": steps, "i": i, "n": len(RUNS[kind])})
        r = run_one(fmt, steps, 0, steps_answer, read_product, t, show)
        if isinstance(r, str):
            return r
        runs.append(r)
    result = {"kind": kind, "runs": runs, "device": runs[0]["device"], "name": runs[0]["name"],
              "minutes": (time.perf_counter() - begin) / 60}
    save(result)
    return result


def save(result: dict) -> None:
    """The two models of equal steps in work/projects/llm/arith_steps.pt and arith_plain.pt, and every run without its
    model in arith.json, for the replay and for asking the models."""
    import torch
    folder = projects.folder() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    for r in pair_of(result):
        torch.save({k: v.detach().cpu() for k, v in r["model"].state_dict().items()}, folder / f"{NAME}_{r['format']}.pt")
    keep = {"kind": result["kind"], "device": result["device"], "name": result["name"],
            "minutes": round(result["minutes"], 1), "runs": [kept(r) for r in result["runs"]]}
    (folder / f"{NAME}.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")


def pair_of(result: dict) -> list:
    """The run with steps and the run of answer only of the same steps: the first two runs of every plan."""
    return result["runs"][:2]


def saved() -> dict | None:
    """The learner's runs, from work/projects/llm/arith.json, when both models are there too; else None."""
    folder = projects.folder() / "llm"
    try:
        info = json.loads((folder / f"{NAME}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if all((folder / f"{NAME}_{fmt}.pt").is_file() for fmt in ("steps", "plain")) else None


def load(fmt: str, source: str | None = None) -> tuple:
    """The model of `fmt` that the lesson asks: the learner's when there is one (or "yours" is asked), else the
    course's of seed 0 on the CPU plan; with where it comes from, "yours" or "course"."""
    import torch
    ran = run_code("from llm.modern_gpt import ModernGPT")
    folder = projects.folder() / "llm"
    mine = folder / f"{NAME}_{fmt}.pt"
    if source in (None, "yours") and saved() is not None:
        where_, path = "yours", mine
    else:
        where_, path = "course", paths.ROOT / COURSE_MODELS[fmt]
    model = build(ran.names["ModernGPT"])
    model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()})
    return model.eval(), where_


def asked(text: str) -> tuple | None:
    """Two numbers someone typed, such as 123*456 or 123 x 456, of 1 to 4 digits each; else None."""
    m = re.fullmatch(r"\s*(\d{1,4})\s*[*x×]\s*(\d{1,4})\s*", text or "")
    return (int(m[1]), int(m[2])) if m else None


def ask(pair_: tuple, read_product, steps_answer) -> list:
    """What both models write for one pair, greedily, with what read_product reads in it: (format, text, read, right,
    where the model comes from)."""
    a, b = pair_
    n = max(len(str(a)), len(str(b)))
    out = []
    for fmt in ("steps", "plain"):
        model, where_ = load(fmt)
        text = greedy(model, [pair_], longest(n, fmt, steps_answer))[0]
        try:
            got = read_product(text)
        except Exception:
            got = None
        out.append((fmt, text, got, got is not None and str(got) == f"{a * b}", where_))
    return out


# --- the course's runs ---------------------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


SAME = {("gpu", "steps", 1800): "cpu", ("gpu", "plain", 1800): "cpu"}   # the runs of the GPU plan that the CPU plan has


def course_runs(plan: str, fmt: str, steps: int) -> list:
    """The course's runs of one format and number of steps on a plan, in the order of their seeds. The runs of 1,800
    steps of the GPU plan are those of the CPU plan: the same code, seeds and steps, so the course recorded them once,
    and the GPU plan's third run on the same CPU with the same threads."""
    plan = SAME.get((plan, fmt, steps), plan)
    return sorted((r for r in course()["runs"] if (r["plan"], r["format"], r["steps"]) == (plan, fmt, steps)),
                  key=lambda r: r["seed"])


def course_result(plan: str, seed: int = 0) -> dict:
    """The course's runs of one seed on a plan, in the shape of the demo's result, for the replay and the table."""
    runs = [course_runs(plan, fmt, steps)[seed] for fmt, steps in RUNS[plan]]
    return {"kind": plan, "runs": runs, "device": "cpu", "name": "CPU"}


def rights(plan: str, fmt: str, steps: int, n: int) -> list:
    """The right answers of 200 on n-digit pairs of the course's seeds of one run."""
    return [r["accuracy"][str(n)]["right"] for r in course_runs(plan, fmt, steps)]


def answer_tokens(r: dict) -> float:
    """The tokens an answer of a run took on average, its end-of-text included, over all held-out pairs."""
    return sum(a["tokens"] for a in r["accuracy"].values()) / sum(a["total"] for a in r["accuracy"].values())


def per_step(plan: str, fmt: str, steps: int) -> float:
    """The seconds a training step of the course's runs took, on average over its seeds."""
    found = course_runs(plan, fmt, steps)
    return sum(r["seconds"] / r["steps"] for r in found) / len(found)


def plan_minutes(plan: str) -> float:
    """The minutes all runs of a plan took on the course's CPU, wall time with their measuring, on average over the
    seeds."""
    total = 0.0
    for fmt, steps in RUNS[plan]:
        found = course_runs(plan, fmt, steps)
        total += sum(r["wall_seconds"] for r in found) / len(found)
    return total / 60


def threads(plan: str) -> int:
    """The threads of the course's CPU that the last run of a plan was recorded with."""
    return course_runs(plan, *RUNS[plan][-1])[0]["threads"]


# --- the pictures --------------------------------------------------------------------------------------------------------

PAIRS = [(7, 8), (17, 21), (64, 50), (99, 99), (169, 186), (123, 405), (999, 999), (8785, 1613)]   # the play block


def tokens_frame(k: int, labels: dict) -> tuple:
    """The k-th pair of PAIRS (from 1) in both formats, as the tokens the model gets: the prompt in the colour of
    data, which the loss does not count, and the answer with its end-of-text in the colour of a prediction, which
    the model learns to write; and the number of answer tokens of each. `labels`: title ({a}, {b}), plain, steps,
    tokens ({n}), end, alt."""
    import html
    steps_answer, _ = course_functions()
    a, b = PAIRS[k - 1]
    W, BW, GAP, BH = 460, 13, 2, 22
    mono = "font-family:var(--ea-font-code)"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">'
             f'{html.escape(labels["title"].format(a=a, b=b))}</text>']
    y = 30
    counts = {}
    for fmt in ("plain", "steps"):
        shown = answer(a, b, fmt, steps_answer)
        counts[fmt] = len(shown) + 1
        parts.append(f'<text x="0" y="{y + 14}" style="font-size:13px;fill:var(--ea-ink-2)">'
                     f'{html.escape(labels[fmt])} · {html.escape(labels["tokens"].format(n=counts[fmt]))}</text>')
        y += 22
        x = 0
        cells = [(c, "data") for c in prompt(a, b)] + [(c, "pred") for c in shown] + [(labels["end"], "pred")]
        for c, role in cells:
            w = BW if len(c) == 1 else 12 + 8 * len(c)
            if x + w > W:
                x, y = 0, y + BH + 4
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{BH}" rx="3" '
                         f'style="fill:var(--ea-{role}-wash);stroke:var(--ea-{role});stroke-width:1"/>'
                         f'<text x="{x + w / 2}" y="{y + BH / 2 + 4.5}" style="{mono};font-size:13px;'
                         f'fill:var(--ea-{role}-text);text-anchor:middle">{html.escape(c)}</text>')
            x += w + GAP
        y += BH + 18
    return draw_svg(W, y, labels["alt"], "".join(parts)), counts


def draw_svg(width: int, height: int, alt: str, body: str) -> str:
    """An SVG as text, for play.Explore, in the frame of draw._svg."""
    from everything_ai import draw
    return draw._svg(width, height, alt, body, natural=True).text


def formats_frame(k: int, labels: dict) -> tuple:
    """For L.explore: the picture of tokens_frame and the sentence under it."""
    svg, counts = tokens_frame(k, labels)
    a, b = PAIRS[k - 1]
    n = max(len(str(a)), len(str(b)))
    parts = len(str(b))
    return svg, labels["said"].format(a=a, b=b, product=whole(a * b, labels.get("lang", "en")), plain=counts["plain"],
                                      steps=counts["steps"], parts=labels["parts" if parts > 1 else "part"].format(n=parts), n=n)


def shown(text: str) -> str:
    """What a model wrote, as code in the sentence under a picture, with a place to break the line (a zero-width space)
    after every comma and every >, and after 24 characters without one, so that a long answer wraps in a narrow window."""
    if not text:
        return "…"
    out, run = [], 0
    for c in text:
        out.append(c)
        run += 1
        if c in ",>" or run == 24:
            out.append("\u200b")
            run = 0
    return "`" + "".join(out) + "`"


def whole(x: int, lang: str) -> str:
    """A whole number in a sentence: with a separator from five digits on, 12,800 and 12.800, and 3200 without one."""
    return number(x, 0, lang) if abs(x) >= 10000 else str(x)


def replay_frame(result: dict, i: int, labels: dict) -> tuple:
    """Moment i (from 0) of the two runs of equal steps of a result, for the replay of the run (L.explore): on the
    left the loss of both along the run, the mean of every 50 steps, with a dashed line at the moment; on the right
    the right answers to the first SNAP held-out 3x3 pairs at every moment so far, in ink, solid with steps and
    dashed answer only; under it what both wrote for the first 3x3 pair, and at the last moment three answers of the
    model with steps to 4x4 pairs. `labels`: title_loss, title_right, x, y_loss, y_right, alt, lang, start, after
    ({step}), right ({steps}, {plain}, {n}), steps, plain, longer."""
    from everything_ai import chart
    two = pair_of(result)
    lang = labels.get("lang", "en")
    step = two[0]["snapshots"][i]["step"]
    total = two[0]["steps"]
    top = math.ceil(max(l["loss"] for r in two for l in r["losses"]) * 2) / 2
    left = chart.Panel((0, total), (0, top), title=labels["title_loss"], xlabel=labels["x"], ylabel=labels["y_loss"])
    for r, dashed in zip(two, (False, True)):
        left.line([l["step"] for l in r["losses"]], [l["loss"] for l in r["losses"]], "loss", 1.6 if dashed else 2.2, dashed=dashed)
    left.line([step, step], [0, top], "ink-3", 1.2, dashed=True)
    right = chart.Panel((0, total), (0, SNAP), title=labels["title_right"], xlabel=labels["x"], ylabel=labels["y_right"])
    for r, dashed in zip(two, (False, True)):
        done = [s for s in r["snapshots"] if s["step"] <= step]
        right.line([s["step"] for s in done], [s["right"] for s in done], "ink", 1.6 if dashed else 2.2, dashed=dashed)
        right.points([s["step"] for s in done], [s["right"] for s in done], "ink", r=3, hollow=dashed)
    right.line([step, step], [0, SNAP], "ink-3", 1.2, dashed=True)
    svg = chart.figure([left, right], alt=labels["alt"], width=330, height=220)
    when = labels["start"] if step == 0 else labels["after"].format(step=number(step, 0, lang))
    snaps = [r["snapshots"][i] for r in two]
    a, b = HELD_OUT[3][0]
    said = (f"**{when}** · " + labels["right"].format(steps=snaps[0]["right"], plain=snaps[1]["right"], n=SNAP) +
            f"\n\n{a} × {b} = {whole(a * b, lang)}. {labels['steps']}: {shown(snaps[0]['text'])}  \n{labels['plain']}: {shown(snaps[1]['text'])}")
    if i == len(two[0]["snapshots"]) - 1:
        fours = two[0]["accuracy"]["4"]["texts"]
        said += "\n\n" + labels["longer"] + "  \n" + "  \n".join(
            f"{x} × {y}: {shown(t)}" for (x, y), t in zip(HELD_OUT[4], fours))
    return svg, said


def live_reference(plan: str, fmt: str, steps: int) -> list:
    """The loss of the course's run of seed 0 of one format on a plan, as (step, loss), for the live chart."""
    return [(l["step"], l["loss"]) for l in course_runs(plan, fmt, steps)[0]["losses"]]


def first_part(text: str) -> str:
    """What a model wrote for the first partial product of an answer with steps: the number after its first =."""
    return text.split(",")[0].split("=")[-1]


# GRPO on a model with steps of the research report (docs/onderzoek/reports/Reasoning in de course.md, C1): its
# multiply.py with the report's schedule, seed 0, then tiny/grpo_tiny.py for 6 minutes at three learning rates. Right of
# 200 3x3 pairs greedily, in percent, before and after; one sample at temperature 1 and pass@8 before.
RL = {"before": 78.0, "steps": 52, "1e-4": 77.0, "3e-4": 67.5, "1e-3": 0.0, "one": 66.1, "pass8": 93}


# --- measured: what the course's runs say (from data/course-reasoning-runs.json) -----------------------------------------

def _measured() -> dict:
    out = {}
    for plan in RUNS:
        out[plan] = {f"{fmt}-{steps}": {str(n): rights(plan, fmt, steps, n) for n in TESTS} for fmt, steps in RUNS[plan]}
    return out


MEASURED = _measured()


# --- the exercises -------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
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


class _Nothing:
    """None as an answer of read_product: a right outcome of its own, not a function that forgot its return, so the
    check compares it as a value and shows it as `None`."""

    def __repr__(self):
        return "None"


NOTHING = _Nothing()


def _kept(value):
    return NOTHING if value is None else value


def _same_product(got, want) -> bool:
    """The same digits as text; a number counts as its text, so 357 is "357"."""
    got, want = (None if v is NOTHING else v for v in (got, want))
    return (got is None) == (want is None) and (got is None or str(got) == str(want))


class _TextBox(FunctionBox):
    """A check whose cases can be long texts: a message names a case by its beginning and its end."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_short(a) for a in case)})"


def _short(value) -> str:
    if isinstance(value, str):
        text = value if len(value) <= 60 else value[:24] + " … " + value[-30:]
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return repr(value)


# --- exercise 1: steps_answer ---------------------------------------------------------------------------------------

_A_START = (_A_HEAD + "    # the digits of b, from the right, as numbers\n" + _A_DIGITS +
            "    # 1. a partial product for every digit\n"
            "    parts = []\n"
            "    for d in digits:\n"
            "        ...\n"
            "    # 2. the running sums: each next partial product times 10 ** i\n"
            "    total = a * digits[0]\n"
            "    sums = []\n"
            "    for i in range(1, len(digits)):\n"
            "        ...\n"
            "    # 3. the partial products, the sums, and the product\n"
            "    return ...\n")
_A_WRONG = {
    "left": _A_CODE.replace("reversed(str(b))", "str(b)"),
    "no-shift": _A_CODE.replace("a * digits[i] * 10 ** i", "a * digits[i]"),
    "shifted-parts": _A_CODE.replace("    for d in digits:\n        parts.append(f\"{a}*{d}={a * d}\")\n",
                                     "    for i in range(len(digits)):\n        parts.append(f\"{a}*{digits[i]}={a * digits[i] * 10 ** i}\")\n"),
    "from-zero": _A_CODE.replace("    total = a * digits[0]\n", "    total = 0\n").replace("range(1, len(digits))", "range(len(digits))"),
    "no-product": _A_CODE.replace(" + \">\" + f\"{total}\"\n", "\n"),
    "end": _A_CODE.replace(" + f\"{total}\"\n", " + f\"{total}\" + \"\\n\"\n"),
    "spaces": _A_CODE.replace("\",\".join(parts)", "\", \".join(parts)").replace("\",\".join(sums)", "\", \".join(sums)"),
    "one-digit": _A_CODE.replace("    return \",\".join(parts) + \">\" + \",\".join(sums) + \">\" + f\"{total}\"\n",
                                 "    text = \",\".join(parts)\n    if sums:\n        text = text + \">\" + \",\".join(sums)\n"
                                 "    return text + \">\" + f\"{total}\"\n"),
}
_A_VISIBLE = [(17, 21)]
_A_HIDDEN = [(7, 1), (12, 34), (404, 760), (123, 405), (999, 999), (5, 38), (90, 10), (3141, 5926), (1000, 1)]

# --- exercise 2: read_product ---------------------------------------------------------------------------------------

_R_START = (_R_HEAD + "    # the part after the last >, or the whole text when it has none\n"
            "    last = ...\n"
            "    # that part if it is all digits, else None\n"
            "    ...\n")
_R_WRONG = {
    "first": _R_CODE.replace("[-1]", "[1]"),
    "not-digits": _R_HEAD + "    return text.split(\">\")[-1]\n",
    "equals": _R_CODE.replace("text.split(\">\")[-1]", "text.split(\"=\")[-1]"),
}
_R_LONG = "855*3=3965,855*6=5130,855*1=855>3965+51300=55265,55265+85500=140765>140765"   # the course's model for 8785 x 1613
_R_VISIBLE = [("17*1=17,17*2=34>17+340=357>357",)]
_R_HIDDEN = [("357",), ("7*1=7>>7",), ("",), ("12*3=36,12*1=12>36+120=",), ("12*3=36>",), ("35*2",), ("0",), (_R_LONG,)]

EXERCISES = {
    "steps-answer": _TextBox(
        name="steps-answer",
        function="steps_answer",
        start=_A_START,
        solution=_A_CODE,
        visible=_A_VISIBLE,
        hidden=_A_HIDDEN,
        save_as=("llm", "steps_answer"),
        first=[_DOTS],
        mistakes=[
            Mistake("left", wrong=_A_WRONG["left"],
                    en=_CALL + ": you take the digits of `b` from the left. Then the first partial product belongs to the highest place, and the shifts in the sums no longer fit. Keep the line the box starts with: `reversed(str(b))` gives them from the right.",
                    nl=_CALL_NL + ": je neemt de cijfers van `b` van links. Dan hoort het eerste deelproduct bij de hoogste plek, en passen de verschuivingen in de sommen niet meer. Laat de regel staan waarmee het vak begint: `reversed(str(b))` geeft ze van rechts."),
            Mistake("no-shift", wrong=_A_WRONG["no-shift"],
                    en=_CALL + ": you add the partial products without their place, so the product comes out wrong. The digit at place `i` counts `10 ** i` times: add `a * digits[i] * 10 ** i`.",
                    nl=_CALL_NL + ": je telt de deelproducten op zonder hun plek, dus het product klopt niet. Het cijfer op plek `i` telt `10 ** i` keer: tel `a * digits[i] * 10 ** i` op."),
            Mistake("shifted-parts", wrong=_A_WRONG["shifted-parts"], state="almost",
                    en=_CALL + ". Your product is right, but you shift the partial products themselves. The course writes each partial product as the model can check it, `17*2=34`, and the shift in the sum after it, `17+340=357`: each step a small one.",
                    nl=_CALL_NL + ". Je product klopt, maar je verschuift de deelproducten zelf. De course schrijft elk deelproduct zoals het model het kan narekenen, `17*2=34`, en de verschuiving in de som erna, `17+340=357`: elke stap een kleine."),
            Mistake("from-zero", wrong=_A_WRONG["from-zero"],
                    en=_CALL + ": your sums start at 0, so there is one sum too many, `0+17=17`. Start the total at the first partial product, `a * digits[0]`, as the box does, and add from the second digit on.",
                    nl=_CALL_NL + ": je sommen beginnen bij 0, dus er is één som te veel, `0+17=17`. Begin het totaal bij het eerste deelproduct, `a * digits[0]`, zoals het vak doet, en tel op vanaf het tweede cijfer."),
            Mistake("no-product", wrong=_A_WRONG["no-product"],
                    en=_CALL + ": your answer ends with the sums, without `>` and the product. The product at the end is what the checker of exercise 2 reads: after the last `>`.",
                    nl=_CALL_NL + ": je antwoord eindigt met de sommen, zonder `>` en het product. Het product aan het eind is wat de checker van oefening 2 leest: na de laatste `>`."),
            Mistake("end", matches=lambda got, text: isinstance(got, str) and got != got.rstrip(),
                    en=_CALL + ": your answer ends with a new line or a space. Give only the steps and the product: `sft_example` puts the end-of-text after it.",
                    nl=_CALL_NL + ": je antwoord eindigt met een nieuwe regel of een spatie. Geef alleen de stappen en het product: `sft_example` zet het end-of-text erachter."),
            Mistake("spaces", matches=lambda got, text: isinstance(got, str) and ", " in got,
                    en=_CALL + ": there is a space after your commas. A space is no character of the model: join with `\",\"`, without a space.",
                    nl=_CALL_NL + ": er staat een spatie na je komma's. Een spatie is geen teken van het model: voeg samen met `\",\"`, zonder spatie."),
            Mistake("one-digit", wrong=_A_WRONG["one-digit"], state="almost",
                    en=_CALL + ". With a `b` of one digit there are no sums, but the course still writes both `>`: `7*1=7>>7`. Then every answer has the same three parts, and the product is always after the second `>`.",
                    nl=_CALL_NL + ". Met een `b` van één cijfer zijn er geen sommen, maar de course schrijft toch beide `>`: `7*1=7>>7`. Dan heeft elk antwoord dezelfde drie delen, en staat het product altijd na de tweede `>`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: a partial product for every digit, a sum for every digit after the first, and the answer as one text.",
                      "nl": "Vervang de drie `...`: een deelproduct voor elk cijfer, een som voor elk cijfer na het eerste, en het antwoord als één tekst."},
            "wrong": {"en": _CALL + ". The partial products `a*d=...` from the right, joined with commas; `>`; the running sums `total+add=...`, joined with commas; `>`; and the product?",
                      "nl": _CALL_NL + ". De deelproducten `a*d=...` van rechts, met komma's ertussen; `>`; de lopende sommen `totaal+erbij=...`, met komma's ertussen; `>`; en het product?"},
            "almost": {"en": "Right for 17 × 21. But " + _CALL[0].lower() + _CALL[1:] + ". A `b` of one digit, a zero among the digits, a `b` of four digits?",
                       "nl": "Goed voor 17 × 21. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Een `b` van één cijfer, een nul tussen de cijfers, een `b` van vier cijfers?"},
            "correct": {"en": "Your answers write every step of the multiplication, in the format the course trains on, also with zeros and with one digit. It is in your project now, as `{saved}`.",
                        "nl": "Je antwoorden schrijven elke stap van de vermenigvuldiging, in de vorm waarop de course traint, ook met nullen en met één cijfer. Het staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `steps_answer` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `steps_answer` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "read-product": _TextBox(
        name="read-product",
        function="read_product",
        start=_R_START,
        solution=_R_CODE,
        visible=_R_VISIBLE,
        hidden=_R_HIDDEN,
        compare=_same_product,
        collect=_kept,
        save_as=("llm", "read_product"),
        first=[_DOTS],
        mistakes=[
            Mistake("int", error=lambda e: isinstance(e, ValueError) and "int()" in str(e),
                    en="With `{call}`, your function stops with an error: `int` cannot make a number of a text that is not all digits. Check with `.isdigit()` first, and give `None` when it is not; the product can stay text.",
                    nl="Met `{call}` stopt je functie met een fout: `int` kan geen getal maken van een tekst die niet alleen uit cijfers bestaat. Controleer eerst met `.isdigit()`, en geef `None` als dat niet zo is; het product mag tekst blijven."),
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: it asks for a part of the text that is not there. An answer of the model without steps has no `>` at all, and a broken one can be empty. `text.split(\">\")[-1]` is always there, and `.isdigit()` works on an empty text too.",
                    nl="Met `{call}` stopt je functie met een fout: hij vraagt om een deel van de tekst dat er niet is. Een antwoord van het model zonder stappen heeft helemaal geen `>`, en een kapot antwoord kan leeg zijn. `text.split(\">\")[-1]` is er altijd, en `.isdigit()` werkt ook op een lege tekst."),
            Mistake("first", wrong=_R_WRONG["first"],
                    en=_CALL + ": you read the part after the first `>`: the running sums. The product comes after the last one: `text.split(\">\")[-1]`.",
                    nl=_CALL_NL + ": je leest het deel na de eerste `>`: de lopende sommen. Het product staat na de laatste: `text.split(\">\")[-1]`."),
            Mistake("not-digits", wrong=_R_WRONG["not-digits"],
                    en=_CALL + ": you give the last part also when it is not a number. A model that stops halfway writes such a part; a checker must say *no answer*, `None`, or a broken answer counts as one. Give the part only when `.isdigit()` says so.",
                    nl=_CALL_NL + ": je geeft het laatste deel ook als het geen getal is. Een model dat halverwege stopt, schrijft zo'n deel; een checker moet *geen antwoord* zeggen, `None`, anders telt een kapot antwoord als antwoord. Geef het deel alleen als `.isdigit()` dat zegt."),
            Mistake("equals", wrong=_R_WRONG["equals"],
                    en=_CALL + ": you read the part after the last `=`. With steps, that is the result of the last sum, `>` and the product, such as `357>357`; the product alone comes after the last `>`.",
                    nl=_CALL_NL + ": je leest het deel na de laatste `=`. Met stappen is dat de uitkomst van de laatste som, `>` en het product, zoals `357>357`; het product alleen staat na de laatste `>`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the part after the last `>`, and that part only when it is all digits, else `None`.",
                      "nl": "Vervang de twee `...`: het deel na de laatste `>`, en dat deel alleen als het alleen uit cijfers bestaat, anders `None`."},
            "wrong": {"en": _CALL + ". Is it the part after the last `>` (the whole text when there is none), and `None` when that part is not all digits?",
                      "nl": _CALL_NL + ". Is het het deel na de laatste `>` (de hele tekst als die er niet is), en `None` als dat deel niet alleen uit cijfers bestaat?"},
            "almost": {"en": "Right for the answer with steps. But " + _CALL[0].lower() + _CALL[1:] + ". An answer without steps, an empty text, an answer that stops halfway?",
                       "nl": "Goed voor het antwoord met stappen. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Een antwoord zonder stappen, een lege tekst, een antwoord dat halverwege stopt?"},
            "correct": {"en": "Your checker reads the product of both formats, and says `None` when there is none, so that a broken answer counts as wrong. It is in your project now, as `{saved}`.",
                        "nl": "Je checker leest het product van beide vormen, en zegt `None` als er geen is, zodat een kapot antwoord als fout telt. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `read_product` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `read_product` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# The lesson is done when these are correct; "longer" and "fair" are the open questions after the demo.
REQUIRED = ["steps-answer", "read-product", "longer", "fair"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("steps-answer", _A_START, "empty", None),
    ("steps-answer", _A_CODE, "correct", None),
    *[("steps-answer", code, "almost" if key in ("shifted-parts", "one-digit") else "wrong", key) for key, code in _A_WRONG.items()],
    ("steps-answer", _A_HEAD + _A_DIGITS + "    parts = [f\"{a}*{d}={a * d}\" for d in digits]\n"
     "    sums = []\n    total = a * digits[0]\n    i = 1\n    for d in digits[1:]:\n"
     "        sums.append(f\"{total}+{a * d * 10 ** i}={total + a * d * 10 ** i}\")\n"
     "        total += a * d * 10 ** i\n        i += 1\n"
     "    return \",\".join(parts) + \">\" + \",\".join(sums) + \">\" + str(a * b)\n", "correct", None),
    ("steps-answer", _A_CODE.replace(" + f\"{total}\"\n", " + f\"{a + b}\"\n"), "wrong", None),
    ("read-product", _R_START, "empty", None),
    ("read-product", _R_CODE, "correct", None),
    *[("read-product", code, "wrong", key) for key, code in _R_WRONG.items()],
    ("read-product", _R_HEAD + "    if \">\" in text:\n        text = text.split(\">\")[-1]\n"
     "    if text.isdigit():\n        return int(text)\n    return None\n", "correct", None),
    ("read-product", _R_HEAD + "    last = text.split(\">\")[-1]\n    return int(last)\n", "error", "int"),
    ("read-product", _R_HEAD + "    return text.split(\">\")[2]\n", "error", "index"),
    ("read-product", _R_CODE.replace("    return None\n", "    return \"\"\n"), "almost", None),
]

# A run with steps that gets at least this many of 200 held-out pairs right at 2x2 and 3x3 has learned as the course's
# runs did: the lowest of the course's three seeds (188 and 149 on both plans, whose runs with steps are the same), less a
# margin for another computer, another PyTorch, and float rounding on a GPU; the answer alone never got more than 37.
THRESHOLD = {"cpu": {"2": 170, "3": 120}, "gpu": {"2": 170, "3": 120}}


def _download_line() -> dict:
    """The line under the header: what the lesson needs, and how long its demo took on the course's CPU."""
    m = {lang: number(plan_minutes("cpu"), 0, lang) for lang in ("en", "nl")}
    return {
        "en": f"Works on every computer; nothing is downloaded. The demo trains two small models from scratch: {m['en']} minutes on the course's CPU with 2 threads, and a probe in the lesson says how long on yours. Without the time, the course's runs and models stand in.",
        "nl": f"Werkt op elke computer; er wordt niets gedownload. De demo traint twee kleine modellen van nul: {m['nl']} minuten op de CPU van de course met 2 threads, en een meting in de les zegt hoe lang op de jouwe. Zonder die tijd nemen de runs en modellen van de course het over.",
    }


DOWNLOAD_LINE = _download_line()
