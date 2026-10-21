# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-pretrain-capstone: plan_run, the depth and the tokens of a run that spends a compute budget
compute-optimally, with the learner's compute_optimal; pretrain_step, one step of the run: the learning rate of the
schedule, the forward pass under autocast when that is faster, and gradient clipping; and a choice. The demos download
more stories and turn them into shards, measure this computer, plan the run, pretrain the learner's ModernGPT and let it
write a story. The course's run on its CPU stands in until the learner runs theirs. The course's own checkpoint, trained
longer on stories of Cosmopedia by data/tools/train_course_capstone.py, is what later lessons fall back on."""
import importlib.util
import json
import math
import re
import time

from everything_ai import hub, paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _checks("llm-fit-scaling-law", "everything_ai_checks_llm_fit_scaling_law_for_capstone")
M, D = S.M, S.D
B = M.B
BUILDS_ON = S.BUILDS_ON + S.SAVES
projects.builds_on(*BUILDS_ON)

WIDTH, HEAD, BLOCK, PEAK = S.WIDTH, S.HEAD, S.BLOCK, S.PEAK   # the family of llm-fit-scaling-law, and its learning rate
DEPTHS = tuple(range(2, 17))   # the depths plan_run chooses from
BATCH = {"cpu": 16, "gpu": 64}  # windows per mini-batch: a GPU does more at once
MINUTES = 15                    # the time budget of the run
TOKENS_PER_PARAM = S.CHINCHILLA
PROBE, PROBE_STEPS = 6, 8       # the depth that measures this computer, and its timed steps (after two to warm up)
EVALS = 10                      # times the run measures its validation loss on QUICK windows; at the end, on all
QUICK = 64
STORIES = ("datasets/roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt")   # the first 200 MB: see hub.PINNED
CHUNK = 4_000_000               # characters encoded at a time
FOLDER = "tokens_capstone"      # the shards of the run, in work/projects/llm/
NAME = "capstone"               # the run's model in work/projects/llm/: capstone.pt, with capstone.json
COURSE = "data/course-capstone"  # the course's model, .pt and .json
START, LENGTH, TAU = "Once upon a time", 200, 0.8
SNAP = 60                       # tokens of the story the model writes at every evaluation, for the replay of the run
DOWNLOAD_LINE = {
    "en": "Works on every computer. The first 200 MB of the training stories of TinyStories are downloaded once from Hugging Face when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. De eerste 200 MB van de trainingsverhalen van TinyStories worden één keer gedownload van Hugging Face als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def family(depth: int) -> tuple:
    """The width and the heads of the model of `depth` in the family: WIDTH times the depth, a head per HEAD numbers."""
    return WIDTH * depth, depth


def build(ModernGPT, depth: int, vocab: int):
    d, h = family(depth)
    return ModernGPT(vocab, d, h, h, depth)


def sizes(ModernGPT, vocab: int = B.VOCAB) -> dict:
    """The parameters of every depth of DEPTHS, counted on PyTorch's meta device, which keeps no numbers."""
    import torch
    with torch.device("meta"):
        return {depth: sum(p.numel() for p in build(ModernGPT, depth, vocab).parameters()) for depth in DEPTHS}


# --- the solutions, which the course's own runs use ---------------------------------------------------------------------

_PLAN = ("from llm.compute_optimal import compute_optimal\n\n\n"
         "def plan_run(flops_per_second, seconds, sizes, tokens_per_param):\n"
         "    compute = flops_per_second * seconds\n"
         "    params, _ = compute_optimal(compute, tokens_per_param)\n"
         "    best = None\n"
         "    for depth in sizes:\n"
         "        if best is None or abs(sizes[depth] - params) < abs(sizes[best] - params):\n"
         "            best = depth\n"
         "    return best, int(compute / (6 * sizes[best]))\n")

_STEP_HEAD = "import torch\nimport torch.nn.functional as F\n\n\ndef pretrain_step(model, opt, x, y, lr, device_type, mixed):\n"
_STEP = (_STEP_HEAD +
         "    for group in opt.param_groups:\n        group[\"lr\"] = lr\n"
         "    opt.zero_grad()\n"
         "    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed):\n"
         "        logits = model(x)\n"
         "        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))\n"
         "    loss.backward()\n"
         "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
         "    opt.step()\n"
         "    return loss.item()\n")


def course_step():
    return run_code(_STEP).names["pretrain_step"]


# --- the data ----------------------------------------------------------------------------------------------------------

def folder():
    return projects.folder() / "llm" / FOLDER


def shards_info() -> dict | None:
    """What prepare() wrote, when its shards are all there; else None."""
    try:
        info = json.loads((folder() / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if all((folder() / f).is_file() for f in info.get("files", [])) else None


def _digest(merges: dict) -> str:
    """A fingerprint of merge rules, so that shards of other rules are noticed."""
    import hashlib
    return hashlib.sha256(json.dumps(sorted(merges, key=merges.get)).encode()).hexdigest()[:16]


def stories_text(path) -> list:
    """The stories of the downloaded beginning of TinyStories, without the last one, which the cut may have broken."""
    text = path.read_bytes().decode("utf-8", errors="ignore")
    return [s.strip() for s in text.split(B.END) if s.strip()][:-1]


def prepare(show, stories: list | None = None, merges: dict | None = None, source: str | None = None) -> dict | str:
    """Download the first 200 MB of TinyStories once, encode its stories with the learner's encode_fast and the merge
    rules of D.tokenizer(), CHUNK characters at a time, and write each part with the learner's write_shards into
    work/projects/llm/tokens_capstone/. Shards of the same merge rules are kept. `show` hears the progress. "download"
    when the download fails, "run" when a module of the learner stops with an error; else what was written.
    The course's script passes its own stories and merge rules."""
    run = run_code("from llm.fast_encode import encode_fast\nfrom llm.shards import write_shards")
    if run.error is not None:
        return "run"
    if merges is None:
        tok = D.tokenizer()
        merges, source = tok["merges"], tok["source"]
    have = shards_info()
    if have is not None and have.get("digest") == _digest(merges) and stories is None:
        return have
    if stories is None:
        try:
            path = hub.download(*STORIES, progress=lambda done, total: show({"phase": "download", "done": done, "total": total}))
        except OSError:
            return "download"
        stories = stories_text(path)
    out = folder()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("part_*/shard_*.bin"):
        old.unlink()
    (out / "info.json").unlink(missing_ok=True)
    files, ids_total, chars, start, part, i = [], 0, 0, time.perf_counter(), 0, 0
    while i < len(stories):
        chunk = []
        size = 0
        while i < len(stories) and size < CHUNK:
            chunk.append(stories[i])
            size += len(stories[i]) + len(B.END)
            i += 1
        text = B.END.join(chunk) + B.END
        try:
            ids = run.names["encode_fast"](text, merges)
            where = out / f"part_{part:03d}"
            where.mkdir(exist_ok=True)
            written = run.names["write_shards"](ids, B.VOCAB, str(where), D.SHARD)
        except Exception:
            return "run"
        files += [f"part_{part:03d}/" + p.replace("\\", "/").rsplit("/", 1)[-1] for p in written]
        ids_total += len(ids)
        chars += len(text)
        part += 1
        show({"phase": "encode", "stories": i, "total": len(stories), "ids": ids_total, "seconds": time.perf_counter() - start})
    info = {"files": files, "ids": ids_total, "chars": chars, "stories": len(stories), "merges": source,
            "digest": _digest(merges), "dtype": D.DTYPE, "seconds": time.perf_counter() - start}
    (out / "info.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
    return info


# --- measuring and planning ------------------------------------------------------------------------------------------

def where() -> dict:
    """The device of the run and its name, as in gpt-train-bpe; "gpu" or "cpu" for the mini-batch."""
    p = M.plan()
    return {"device": p["device"], "name": p["name"], "kind": "cpu" if p["device"] == "cpu" else "gpu"}


def _sync(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


_on = M.on_device   # the tensors the forward pass makes, such as the causal mask, on the device of the run


def peak(device: str, dtype_name: str = "float32") -> float:
    """The FLOP/s of the fastest of five multiplications of two 2048 x 2048 matrices in `dtype_name`: about the most
    this device does in that format, as in sys-profiling-mfu."""
    import torch
    n, dtype = 2048, getattr(torch, dtype_name)
    a, b = torch.randn(n, n, device=device, dtype=dtype), torch.randn(n, n, device=device, dtype=dtype)
    c = torch.matmul(a, b)
    _sync(device)
    best = float("inf")
    for _ in range(5):
        t = time.perf_counter()
        torch.matmul(a, b, out=c)
        _sync(device)
        best = min(best, time.perf_counter() - t)
    return 2 * n ** 3 / best


def flops_per_token(params: int, depth: int) -> int:
    """6N for the weights plus the attention scores, 6 L T d, as in gpt-count and sys-profiling-mfu."""
    return 6 * params + 6 * depth * BLOCK * family(depth)[0]


def measure(step_fn, show) -> dict | str:
    """Time PROBE_STEPS steps of the learner's pretrain_step on the model of depth PROBE, in FP32 and under autocast,
    and keep the faster. Its tokens per second times 6N are the FLOP/s a plan can count on. "run" when the learner's
    code stops with an error."""
    import torch
    ran = run_code("from llm.modern_gpt import ModernGPT")
    if ran.error is not None:
        return "run"
    w = where()
    batch = BATCH[w["kind"]]
    torch.manual_seed(0)
    x = torch.randint(0, B.VOCAB, (batch, BLOCK), device=w["device"])
    y = torch.randint(0, B.VOCAB, (batch, BLOCK), device=w["device"])
    speed = {}
    for mixed in (False, True):
        try:
            model = build(ran.names["ModernGPT"], PROBE, B.VOCAB).to(w["device"])
            opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.1)
            with _on(w["device"]):
                for _ in range(2):
                    step_fn(model, opt, x, y, PEAK, w["device"], mixed)
                _sync(w["device"])
                start = time.perf_counter()
                for _ in range(PROBE_STEPS):
                    step_fn(model, opt, x, y, PEAK, w["device"], mixed)
                _sync(w["device"])
            speed[mixed] = batch * BLOCK * PROBE_STEPS / (time.perf_counter() - start)
        except Exception:
            if not mixed:
                return "run"
        show({"phase": "measure", "mixed": mixed})
    params = sum(p.numel() for p in model.parameters())
    mixed = speed.get(True, 0) > speed[False]
    return {**w, "mixed": mixed, "tps_fp32": speed[False], "tps_bf16": speed.get(True), "probe_params": params,
            "flops": speed[mixed] * 6 * params}


# --- the run ---------------------------------------------------------------------------------------------------------

def _windows(tools, info: dict):
    import numpy as np
    files = [str(folder() / f) for f in info["files"]]
    return tools["TokenWindows"](files, BLOCK, np.dtype(info["dtype"]))


def _loss(model, windows, device: str, mixed: bool, count: int | None = None) -> float:
    """The mean loss per token on the first `count` windows (all when None), in batches of 32."""
    import torch
    total, n = 0.0, 0
    count = len(windows) if count is None else min(count, len(windows))
    model.eval()
    with torch.no_grad():
        for start in range(0, count, 32):
            pairs = [windows[i] for i in range(start, min(start + 32, count))]
            xb = torch.stack([p[0] for p in pairs]).to(device)
            yb = torch.stack([p[1] for p in pairs]).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=mixed), _on(device):
                logits = model(xb)
            total += torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), yb.reshape(-1),
                                                       reduction="sum").item()
            n += yb.numel()
    model.train()
    return total / n


class _Orders:
    """The shuffled order of each epoch, made once."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.made = n, seed, {}

    def __call__(self, k: int) -> int:
        import torch
        epoch, at = divmod(k, self.n)
        if epoch not in self.made:
            self.made = {epoch: torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed * 1000 + epoch))}
        return int(self.made[epoch][at])


def train(depth: int, tokens: int, step_fn, mixed: bool, show, *, train_windows=None, val_windows=None,
          vocab: int = B.VOCAB, seed: int = 0, state_file=None, save_every: int = 0, snap=None) -> dict | str:
    """Pretrain the learner's ModernGPT of `depth` for `tokens` tokens with their pretrain_step: AdamW at PEAK times
    the schedule of llm-open-recipes, mini-batches of BATCH windows, the validation loss on QUICK windows EVALS times and
    on all at the end. `show` hears every step that ends a second and every evaluation. With `state_file`, the model, the
    optimizer and the step are kept there every `save_every` steps, and a run starts again from it. With `snap(model)`,
    every evaluation also keeps what the model writes then ("story"), for the replay of the run. "run" when a module
    of the learner stops with an error; else the model and what the run measured."""
    import torch
    ran = run_code("from llm.modern_gpt import ModernGPT\nfrom llm.token_windows import TokenWindows")
    if ran.error is not None:
        return "run"
    if train_windows is None:
        info = shards_info()
        if info is None:
            return "data"
        tok = D.tokenizer()
        if info.get("digest") != _digest(tok["merges"]):   # the merge rules changed after the stories were encoded
            return "merges"
        pipeline = D.folder() / "info.json"   # validation shards of other merge rules: S._shards writes them again
        try:
            if json.loads(pipeline.read_text(encoding="utf-8")).get("merges") != tok["source"]:
                pipeline.unlink()
        except (OSError, ValueError):
            pass
        train_windows = _windows(ran.names, info)
        data = S._shards(ran.names)
        if isinstance(data, str):
            return data
        val_windows = data[1]
    w = where()
    batch = BATCH[w["kind"]]
    torch.manual_seed(seed)
    model = build(ran.names["ModernGPT"], depth, vocab).to(w["device"])
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=0.1)
    steps = max(1, round(tokens / (batch * BLOCK)))
    every = max(1, steps // EVALS)
    history, step, seconds = [], 0, 0.0
    state = torch.load(state_file, map_location=w["device"], weights_only=False) if state_file is not None and state_file.exists() else None
    if state is not None and (state.get("depth"), state.get("steps"), state.get("seed")) == (depth, steps, seed):   # the same run
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        history, step, seconds = state["history"], state["step"], state["seconds"]
        for e in history:   # the curve of the part already done
            show({"phase": "eval", **e, "steps": steps})
    order = _Orders(len(train_windows), seed)
    model.train()
    recent, last_show = [], time.perf_counter()
    while step < steps:
        start = time.perf_counter()
        pairs = [train_windows[order(k)] for k in range(step * batch, (step + 1) * batch)]
        xb = torch.stack([p[0] for p in pairs]).to(w["device"])
        yb = torch.stack([p[1] for p in pairs]).to(w["device"])
        try:
            with _on(w["device"]):
                loss = float(step_fn(model, opt, xb, yb, PEAK * S.multiplier(step, steps), w["device"], mixed))
        except Exception as e:
            return "memory" if "out of memory" in str(e).lower() else "run"
        if not math.isfinite(loss):
            return "diverged"
        _sync(w["device"])
        seconds += time.perf_counter() - start
        recent.append(loss)
        step += 1
        if step % every == 0 or step == steps:
            val = _loss(model, val_windows, w["device"], mixed, QUICK)
            history.append({"step": step, "tokens": step * batch * BLOCK, "train": sum(recent) / len(recent), "val": val,
                            "seconds": seconds})
            if snap is not None:   # what the model writes now; its own random numbers, so the training is the same
                try:
                    history[-1]["story"] = snap(model)
                except Exception:
                    history[-1]["story"] = None
            recent = []
            show({"phase": "eval", **history[-1], "steps": steps})
        elif time.perf_counter() - last_show > 1:
            show({"phase": "train", "step": step, "steps": steps, "loss": loss, "seconds": seconds})
            last_show = time.perf_counter()
        if state_file is not None and save_every and step % save_every == 0 and step < steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "history": history, "step": step,
                        "seconds": seconds, "depth": depth, "steps": steps, "seed": seed}, state_file)
    val = _loss(model, val_windows, w["device"], mixed)
    tps = steps * batch * BLOCK / seconds
    top = peak(w["device"], "bfloat16" if mixed else "float32")
    return {"model": model, "depth": depth, "params": params, "tokens": steps * batch * BLOCK, "steps": steps,
            "batch": batch, "val": val, "history": history, "seconds": seconds, "tps": tps, "peak": top,
            "mfu": tps * flops_per_token(params, depth) / top, **w, "mixed": mixed}


def write(model, merges: dict, encode, decode, seed: int = 0, start: str = START, length: int | None = None) -> str:
    """A story of LENGTH tokens (or `length`) from `start`, drawn at temperature TAU with the model's caches, as in
    llm-modern-gpt; it ends early at an end-of-text."""
    import torch
    end_id = 256 + len(merges)
    ids = encode(start, merges)
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    caches = [{} for _ in model.blocks]
    model.eval()
    out = list(ids)
    with torch.no_grad():
        with _on(device.type):
            logits = model(torch.tensor([ids], device=device), caches)
        for _ in range(LENGTH if length is None else length):
            p = torch.softmax(logits[0, -1].double().cpu() / TAU, dim=0)
            j = int(torch.multinomial(p, 1, generator=g))
            if j == end_id:
                break
            out.append(j)
            with _on(device.type):
                logits = model(torch.tensor([[j]], device=device), caches)
    model.train()
    return decode(out, merges)


def saved() -> dict | None:
    """The learner's run, from work/projects/llm/capstone.json, when its model is there too; else None."""
    folder_ = projects.folder() / "llm"
    try:
        info = json.loads((folder_ / f"{NAME}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if (folder_ / f"{NAME}.pt").is_file() else None


def save(result: dict, merges: dict, story: str) -> None:
    """Keep the learner's model in work/projects/llm/capstone.pt, and its sizes, merge rules, run and story in
    capstone.json, for the lessons after this one."""
    import torch
    folder_ = projects.folder() / "llm"
    folder_.mkdir(parents=True, exist_ok=True)
    torch.save(result["model"].state_dict(), folder_ / f"{NAME}.pt")
    d, h = family(result["depth"])
    keep = {k: v for k, v in result.items() if k != "model"}
    keep.update({"sizes": [B.VOCAB, d, h, h, result["depth"]], "merges": [list(p) for p in sorted(merges, key=merges.get)],
                 "story": story})
    (folder_ / f"{NAME}.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")


def load(source: str | None = None) -> tuple:
    """The model the lessons after this one go on with: the learner's capstone when there is one (or "yours" is
    asked), else the course's; with its information and where it comes from, "yours" or "course"."""
    import torch
    ran = run_code("from llm.modern_gpt import ModernGPT")
    info = saved() if source in (None, "yours") else None
    if info is not None:
        where_, state_path = "yours", projects.folder() / "llm" / f"{NAME}.pt"
    else:
        where_ = "course"
        info = json.loads((paths.ROOT / COURSE).with_suffix(".json").read_text(encoding="utf-8"))
        state_path = (paths.ROOT / COURSE).with_suffix(".pt")
    vocab, d, h, kv, depth = info["sizes"]
    model = ran.names["ModernGPT"](vocab, d, h, kv, depth)
    model.load_state_dict({k: v.float() for k, v in torch.load(state_path, map_location="cpu", weights_only=True).items()})
    return model.eval(), info, where_


PLAN, STATE = "capstone_plan.json", "capstone_state.pt"   # in work/projects/llm/


def save_plan(measured: dict, plan_run) -> dict | str:
    """The plan of the learner's plan_run for MINUTES of what the probe measured, kept in
    work/projects/llm/capstone_plan.json. "run" when their code stops with an error or gives no depth of the family."""
    ran = run_code("from llm.modern_gpt import ModernGPT")
    if ran.error is not None:
        return "run"
    table = sizes(ran.names["ModernGPT"])
    try:
        depth, tokens = plan_run(measured["flops"], MINUTES * 60, table, TOKENS_PER_PARAM)
        depth, tokens = int(depth), int(tokens)
    except Exception:
        return "run"
    if depth not in table or tokens < 1:
        return "run"
    batch = BATCH[measured["kind"]]
    plan = {k: measured[k] for k in ("device", "name", "kind", "mixed", "tps_fp32", "tps_bf16", "flops")}
    plan.update({"tps": measured["tps_bf16"] if measured["mixed"] else measured["tps_fp32"],
                 "compute": measured["flops"] * MINUTES * 60, "depth": depth, "params": table[depth], "tokens": tokens,
                 "batch": batch, "steps": max(1, round(tokens / (batch * BLOCK)))})
    path = projects.folder() / "llm" / PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    return plan


def saved_plan() -> dict | None:
    try:
        return json.loads((projects.folder() / "llm" / PLAN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_plan(plan: dict, step_fn, show) -> dict | str:
    """Pretrain the plan with the learner's step, going on from a state the same plan left; then let the model write
    and keep it. A string when it stops: "data", "merges", "memory", "run" or "diverged"."""
    state = projects.folder() / "llm" / STATE
    tok = D.tokenizer()
    tools = run_code("from llm.encode import encode\nfrom llm.decode import decode")
    if tools.error is not None:
        return "run"
    result = train(plan["depth"], plan["tokens"], step_fn, plan["mixed"], show, state_file=state,
                   save_every=max(1, plan["steps"] // EVALS),
                   snap=lambda m: write(m, tok["merges"], tools.names["encode"], tools.names["decode"], length=SNAP))
    if isinstance(result, str):
        return result
    try:
        story = write(result["model"], tok["merges"], tools.names["encode"], tools.names["decode"])
    except Exception:
        return "run"
    save(result, tok["merges"], story)
    state.unlink(missing_ok=True)
    return {**result, "story": story}


def quote(story: str) -> str:
    """A story as a quote in Markdown, with nothing in it that Markdown or KaTeX would read as a sign, and no spaces in
    front of a line that would make it code."""
    safe = re.sub(r"([\\`*_$<>#\[\]|])", r"\\\1", story.strip())
    return "\n>\n".join("> " + part.strip() for part in re.split(r"\n+", safe) if part.strip())


def written(text: str) -> str:
    """What a model wrote, in Markdown for the sentence under a picture: every line in italics on its own line, in the
    serif of a quote (<i>, as play.Explore writes it), with nothing in it that Markdown or KaTeX would read as a sign,
    and no spaces in front that would make it code: also no link from an address (: and .), no ^superscript^ and
    no ++key++."""
    signs = {"&": "&amp;", "<": "&lt;", ">": "&gt;", **{c: f"&#{ord(c)};" for c in "\\`*_$[]|~#^+:."}}
    lines = ["".join(signs.get(c, c) for c in line.strip()) for line in text.split("\n") if line.strip()]
    return "  \n".join(f"<i>{line}</i>" for line in lines) or "…"


def moments(history: list) -> list:
    """The evaluations of a run at which the model wrote something, for its replay. A run evaluates every
    steps // EVALS steps and at its last step, so that last evaluation can come a few steps after the one before
    (4,690 and 4,696 in the course's run): the model has hardly changed and writes the same. Then the one before is
    left out, so that the replay ends with the model the run keeps, whose story the lesson shows in full."""
    ks = [k for k, e in enumerate(history) if e.get("story") is not None]
    if len(ks) > 2:
        gap = history[1]["step"] - history[0]["step"]   # of the run, also when a story is missing in between
        if history[ks[-1]]["step"] - history[ks[-2]]["step"] < gap / 2:   # the last comes less than half a gap later
            del ks[-2]
    return ks


def replay_frame(history: list, k: int, labels: dict) -> tuple:
    """One evaluation of a run, for its replay (L.explore): the validation loss against the tokens seen, on a log scale,
    as curve_picture draws it, with a dashed line at evaluation `k`; and what the model wrote then, from START.
    `labels`: title, x, y, alt, lang, and "after" ({tokens}, in millions), "val"."""
    from everything_ai import chart
    e = history[k]
    xs, ys = [h["tokens"] for h in history], [h["val"] for h in history]
    low, high = math.floor(min(ys) * 4 - 0.5) / 4, math.ceil(max(ys) * 4 + 0.5) / 4
    left, right = 10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs)))   # whole decades
    p = chart.Panel((left, right), (low, high), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True)
    p.line(xs, ys, "loss", 2.2)
    p.line([e["tokens"], e["tokens"]], [low, high], "ink-2", 1.2, dashed=True)
    p.points([e["tokens"]], [e["val"]], "loss", r=4)
    svg = chart.figure([p], alt=labels["alt"], width=360, height=220)
    lang = labels.get("lang", "en")
    said = (f"**{labels['after'].format(tokens=number(e['tokens'] / 1e6, 1, lang))}** · {labels['val']} {number(e['val'], 3, lang)}"
            f"\n\n{written(e['story'])}")
    return svg, said


def course_info() -> dict:
    return json.loads((paths.ROOT / COURSE).with_suffix(".json").read_text(encoding="utf-8"))


def curve_picture(runs: list, reference: float, labels: dict) -> str:
    """The validation loss per token against the tokens seen, on a log scale: each run of `runs` (a history and
    whether to dash it) as a line with dots (loss), and the loss of depth 6 of llm-fit-scaling-law as a thin line."""
    from everything_ai import chart
    xs = [e["tokens"] for history, _ in runs for e in history]
    ys = [e["val"] for history, _ in runs for e in history] + [reference]
    low, high = math.floor(min(ys) * 4 - 0.5) / 4, math.ceil(max(ys) * 4 + 0.5) / 4
    left, right = 10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs)))   # whole decades
    panel = chart.Panel((left, right), (low, high), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xlog=True)
    panel.line([left, right], [reference, reference], "ink-3", 1.2, dashed=True)
    # the name of the reference in the top right corner, where no run comes, with a piece of its dashed line
    panel.text(right / 1.1, high - (high - low) * 0.08, "- - " + labels["reference"], "ink-3", "end")
    for history, dashed in runs:
        panel.line([e["tokens"] for e in history], [e["val"] for e in history], "loss", 1.6 if dashed else 2.4, dashed=dashed)
        if not dashed:
            panel.points([e["tokens"] for e in history], [e["val"] for e in history], "loss", 3)
    svg = chart.figure([panel], alt=labels["alt"], width=330, height=240)
    return chart.with_decimal_comma(svg) if labels.get("lang") == "nl" else svg


# --- measured: the course's run of the demos with its own versions and merge rules, on the CPU, 2 threads; the stories
# of the run come from the same plan run again with run_plan's snap, with the same losses at every evaluation ---
MEASURED = {
    "data": {"stories": 244110, "ids": 67962215, "chars": 199331945, "seconds": 201.4},
    "measure": {"tps_fp32": 3914, "tps_bf16": 1839, "flops": 73778578974.06494, "probe_params": 3141801},
    "plan": {"depth": 3, "params": 575337, "tokens": 19235312, "steps": 4696},
    "run": {"depth": 3, "params": 575337, "tokens": 19234816, "steps": 4696, "batch": 16, "val": 1.8338, "seconds": 1134.7, "tps": 16951, "peak": 284770000000.0, "mfu": 0.2318, "device": "cpu", "name": "CPU",
            "history": [
                {"step": 469, "tokens": 1921024, "train": 3.3041, "val": 2.6295, "seconds": 112.8,
                 "story": "Once upon a time, in a small town, there was a little boy named Tim. Tom was a little girl who loved to fur. He loved other room and play with his friends.\nOne day, Tom saw a new toy in the yard. \"May"},
                {"step": 938, "tokens": 3842048, "train": 2.446, "val": 2.345, "seconds": 223.1,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy boy named Tim.\nOne day, Tim and his friend, Sue, Sue, came. They played together with their toys toys. Tim and Sue was"},
                {"step": 1407, "tokens": 5763072, "train": 2.2704, "val": 2.2341, "seconds": 339.1,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy car. Tim was very safe for a big boy. He had a fragile in.\nOne day, Tim saw a new toy car named Brown. \"W"},
                {"step": 1876, "tokens": 7684096, "train": 2.1701, "val": 2.1239, "seconds": 449.4,
                 "story": "Once upon a time, together, there was a happy cat who loved to play in the sun. The cat and the garden were very happy.\nOne day, the cat met a cat. It was a big, shiny thing. The cat was cat"},
                {"step": 2345, "tokens": 9605120, "train": 2.0919, "val": 2.076, "seconds": 565.1,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy car named Bobo.\nOne day, a big boy named Tim saw his toy car in the fox. He wanted to play in the ground. He saw a small cat"},
                {"step": 2814, "tokens": 11526144, "train": 2.0382, "val": 2.0218, "seconds": 680.5,
                 "story": "Once upon a time, together, there was a happy cat who loved to play in the sand.\nOne sunny day, the cat went to the sandwiches. The sand cat and the boy was wet and in the sky. It was big and small. The"},
                {"step": 3283, "tokens": 13447168, "train": 1.9868, "val": 1.9793, "seconds": 794.6,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy boy named Tim.\nOne day, Tim saw a small boy named Tim. He wanted to play with it. His mom said, \"No, what is it,\" said the"},
                {"step": 3752, "tokens": 15368192, "train": 1.9427, "val": 1.9337, "seconds": 907.2,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy boy named Tim.\nOne day, Tim went to the park to play. He saw his friend, Sam, sending the boy named Tim. Tim saw a cat"},
                {"step": 4221, "tokens": 17289216, "train": 1.9017, "val": 1.8917, "seconds": 1019.9,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy boy named Tim.\nOne day, Tim saw a little boy named Tim. Tim wanted to play with the boy, so he thought, \"Max, can I have water"},
                {"step": 4690, "tokens": 19210240, "train": 1.8689, "val": 1.8604, "seconds": 1133.3,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy bear named Lily. They would play together every day.\nOne day, Tim saw a big, funny cat. It was a hard day, but a big anxious! \"Let"},
                {"step": 4696, "tokens": 19234816, "train": 1.8687, "val": 1.8599, "seconds": 1134.7,
                 "story": "Once upon a time, there was a little boy named Tim. Tim had a toy bear named Lily. They would play together every day.\nOne day, Tim saw a big, funny cat. It was a hard day, but a big anxious! \"Let"},
            ]},
    "story": "Once upon a time, there was a little boy named Tim. Tim had a toy bear named Lily. They would play together every day.\nOne day, Tim saw a big, funny cat. It was a hard day, but a big anxious! \"Let's play in the park!\" Tim thought about it and played with the big ball. He liked to explore it and Ziggy would stay on it. From that day on, When it was time to go home, Lily saw a big picture of a big box. They played together all day long and had lots of fun.",
}
# --- end of measured ---


# --- the checks --------------------------------------------------------------------------------------------------------

REQUIRED = ["plan-run", "pretrain-step", "why-measure"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL
_Named = S._Named

_PLAN_START = ("from llm.compute_optimal import compute_optimal\n\n\n"
               "def plan_run(flops_per_second, seconds, sizes, tokens_per_param):\n"
               "    # the compute of the budget, and the compute-optimal parameters of your compute_optimal\n    ...\n"
               "    # the depth of `sizes` whose parameters come closest\n    ...\n"
               "    # that depth, and the tokens that spend the whole budget on its parameters\n    return ...\n")
_GPU = {6: 3141801, 7: 4782537, 8: 6938601, 9: 9691305, 10: 13100457}
_SMALL = {2: 260649, 3: 575337, 4: 1110633, 5: 1940457, 6: 3141801}
_BELOW = _PLAN.replace("        if best is None or abs(sizes[depth] - params) < abs(sizes[best] - params):\n",
                       "        if sizes[depth] <= params and (best is None or sizes[depth] > sizes[best]):\n")


def _plan_close(got, want) -> bool:
    """The same depth, and the same tokens up to rounding to a whole number."""
    return (isinstance(got, (tuple, list)) and len(got) == 2 and got[0] == want[0]
            and S._close_whole(got[1], want[1]))


_S_SETUP = ("import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Linear(4, 3)\n"
            "seen = []  # the format the layer calculates in\nhook = model.register_forward_hook(lambda m, i, o: seen.append(str(o.dtype)))\n")
_S_XY = "x, y = torch.randn(2, 5, 4), torch.randint(0, 3, (2, 5))  # 2 sequences of 5 positions, as a GPT gets them\n"
_S_SCENARIOS = [
    # the lesson's example: under autocast the layer calculates in BF16, the weights stay FP32, and the step uses lr
    _S_SETUP + "opt = torch.optim.AdamW(model.parameters(), lr=0.1)\n" + _S_XY +
    "pretrain_step(model, opt, x, y, 0.01, \"cpu\", True)\n[seen[0], str(model.weight.dtype), opt.param_groups[0][\"lr\"]]",
    # without mixed, plain FP32
    _S_SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n" + _S_XY +
    "pretrain_step(model, opt, x, y, 0.01, \"cpu\", False)\nseen[0]",
    # the learning rate of the step, not the optimizer's own: a small gradient, not clipped
    _S_SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\nbefore = model.weight.detach().clone()\n" +
    "x, y = torch.randn(2, 5, 4) * 0.1, torch.randint(0, 3, (2, 5))\n"
    "pretrain_step(model, opt, x, y, 0.5, \"cpu\", False)\nround((model.weight - before).abs().sum().item(), 5)",
    # a huge gradient is clipped to a norm of 1: with SGD at a learning rate of 1, every weight together moves by 1
    _S_SETUP + "opt = torch.optim.SGD(model.parameters(), lr=1.0)\nbefore = [p.detach().clone() for p in model.parameters()]\n"
    "x, y = torch.randn(2, 5, 4) * 1000, torch.randint(0, 3, (2, 5))\npretrain_step(model, opt, x, y, 1.0, \"cpu\", False)\n"
    "round(sum(((p - b) ** 2).sum() for p, b in zip(model.parameters(), before)).item() ** 0.5, 4)",
    # the loss it gives back, calculated in FP32 from the BF16 logits
    _S_SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n" + _S_XY +
    "loss = pretrain_step(model, opt, x, y, 0.1, \"cpu\", True)\nround(float(loss), 4)",
    # two steps in a row, each from zero gradients
    _S_SETUP + "opt = torch.optim.SGD(model.parameters(), lr=0.1)\n" + _S_XY +
    "first = pretrain_step(model, opt, x, y, 0.1, \"cpu\", False)\nsecond = pretrain_step(model, opt, x, y, 0.1, \"cpu\", False)\n"
    "round(model.weight.abs().sum().item(), 4)",
    # every group of the optimizer gets the learning rate of the step
    _S_SETUP + "opt = torch.optim.SGD([{\"params\": [model.weight]}, {\"params\": [model.bias], \"lr\": 0.2}], lr=0.1)\n" + _S_XY +
    "pretrain_step(model, opt, x, y, 0.05, \"cpu\", False)\n[group[\"lr\"] for group in opt.param_groups]",
]
_STEP_START = (_STEP_HEAD + "    # the learning rate of this step, for every group of the optimizer\n    ...\n"
               "    opt.zero_grad()\n"
               "    # the forward pass and the loss, under autocast in BF16 only when mixed is True\n    ...\n"
               "    loss.backward()\n"
               "    # clip the gradients to a norm of at most 1.0\n    ...\n"
               "    opt.step()\n    return loss.item()\n")
_NO_LR = _STEP.replace("    for group in opt.param_groups:\n        group[\"lr\"] = lr\n", "")
_NO_CLIP = _STEP.replace("    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n", "")
_ALWAYS = _STEP.replace(", enabled=mixed", "")
_NEVER = _STEP.replace("    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed):\n"
                       "        logits = model(x)\n        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))\n",
                       "    logits = model(x)\n    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))\n")
_LOSS_OUT = _STEP.replace("        logits = model(x)\n        loss = F.cross_entropy(",
                          "        logits = model(x)\n    loss = F.cross_entropy(")
_NO_ZERO = _STEP.replace("    opt.zero_grad()\n", "")
_NO_RESHAPE = _STEP.replace("F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))", "F.cross_entropy(logits, y)")


def _formats(got) -> bool:
    return isinstance(got, list) and len(got) == 3 and all(isinstance(v, str) for v in got[:2])


EXERCISES = {
    "plan-run": FunctionBox(
        name="plan-run",
        function="plan_run",
        start=_PLAN_START,
        solution=_PLAN,
        visible=[(_Named("8e12"), 900, _GPU, 20)],
        hidden=[(_Named("7e10"), 900, _SMALL, 20), (_Named("1.3e11"), 900, _SMALL, 20),
                (_Named("2e13"), 1800, {8: 6938601, 10: 13100457, 12: 22214505}, 10)],
        show=S._rounded,
        compare=_plan_close,
        save_as=("llm", "plan_run"),
        first=[_DOTS],
        mistakes=[
            Mistake("optimal-tokens", wrong=_PLAN.replace("    params, _ = compute_optimal(", "    params, tokens = compute_optimal(")
                    .replace("    return best, int(compute / (6 * sizes[best]))\n", "    return best, int(tokens)\n"),
                    en=_CALL + ": the depth is right, but its tokens are those of the compute-optimal size, and the model you chose has another size. Spend the whole budget on its own parameters: `compute / (6 * sizes[best])`.",
                    nl=_CALL_NL + ": de diepte klopt, maar de tokens zijn die van de compute-optimal grootte, en het model dat je koos heeft een andere grootte. Besteed het hele budget aan zijn eigen parameters: `compute / (6 * sizes[best])`."),
            Mistake("below", wrong=_BELOW,
                    en=_CALL + ": your function takes the biggest model that is not above the compute-optimal size. Take the one that comes closest, above or below: the smallest `abs(sizes[depth] - params)`.",
                    nl=_CALL_NL + ": je functie neemt het grootste model dat niet boven de compute-optimal grootte uitkomt. Neem het model dat er het dichtst bij zit, erboven of eronder: de kleinste `abs(sizes[depth] - params)`."),
            Mistake("no-seconds", wrong=_PLAN.replace("    compute = flops_per_second * seconds\n", "    compute = flops_per_second\n"),
                    en=_CALL + ": your budget is the FLOPs of one second. The compute of the whole budget is `flops_per_second * seconds`.",
                    nl=_CALL_NL + ": je budget is de FLOPs van één seconde. De compute van het hele budget is `flops_per_second * seconds`."),
            Mistake("params-back", wrong=_PLAN.replace("    return best, int(", "    return sizes[best], int("),
                    en=_CALL + ": the first number is the parameters of the model, not its depth. Give back the depth, the key of `sizes`.",
                    nl=_CALL_NL + ": het eerste getal is het aantal parameters van het model, niet zijn diepte. Geef de diepte terug, de key van `sizes`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the compute and its compute-optimal parameters, the closest depth, and that depth with its tokens.",
                      "nl": "Vervang de drie `...`: de compute en zijn compute-optimal parameters, de dichtstbijzijnde diepte, en die diepte met zijn tokens."},
            "wrong": {"en": _CALL + ". The compute is `flops_per_second * seconds`; your `compute_optimal` gives the parameters; then the depth whose parameters come closest, and the tokens that spend the whole compute on them?",
                      "nl": _CALL_NL + ". De compute is `flops_per_second * seconds`; je `compute_optimal` geeft de parameters; dan de diepte waarvan de parameters het dichtst bij komen, en de tokens die de hele compute eraan besteden?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use all four arguments, and no fixed numbers?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie alle vier de argumenten, en geen vaste getallen?"},
            "correct": {"en": "Your function plans a run that spends a time budget compute-optimally. It is in your project now, as `{saved}`.",
                        "nl": "Je functie plant een run die een tijdsbudget compute-optimal besteedt. Hij staat nu in je project, als `{saved}`."},
        },
    ),
    "pretrain-step": ScenarioBox(
        name="pretrain-step",
        start=_STEP_START,
        solution=_STEP,
        scenarios=_S_SCENARIOS,
        save_as=("llm", "pretrain_step"),
        first=[_DOTS],
        mistakes=[
            Mistake("no-lr", wrong=_NO_LR,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your step keeps the learning rate the optimizer started with. Set it first: `group[\"lr\"] = lr` for every `group` of `opt.param_groups`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je stap houdt de learning rate waarmee de optimizer begon. Zet hem eerst: `group[\"lr\"] = lr` voor elke `group` van `opt.param_groups`."),
            Mistake("never-mixed", wrong=_NEVER,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe layer calculates in `torch.float32`, also when `mixed` is True. Put the forward pass and the loss under `with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed):`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\nrekent de layer in `torch.float32`, ook als `mixed` True is. Zet de forward pass en de loss onder `with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed):`."),
            Mistake("always-mixed", wrong=_ALWAYS,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe layer calculates in `torch.bfloat16`, also when `mixed` is False. Let `mixed` switch autocast on and off: `enabled=mixed`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\nrekent de layer in `torch.bfloat16`, ook als `mixed` False is. Laat `mixed` autocast aan- en uitzetten: `enabled=mixed`."),
            Mistake("no-clip", wrong=_NO_CLIP,
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: a huge gradient moves your weights all the way. Clip the gradients between `loss.backward()` and `opt.step()`: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: een enorme gradient verschuift je weights helemaal mee. Clip de gradients tussen `loss.backward()` en `opt.step()`: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`."),
            Mistake("model-converted", matches=lambda got, text: _formats(got) and got[1] != "torch.float32",
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`: the weights of your model are no longer FP32, so a small update rounds away. Leave the model and `x` as they are: autocast makes the BF16 copies itself.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`: de weights van je model zijn geen FP32 meer, dus een kleine update rondt weg. Laat het model en `x` zoals ze zijn: autocast maakt de BF16-kopieën zelf."),
            Mistake("no-zero-grad", wrong=_NO_ZERO, state="almost",
                    en="Your second step adds its gradients to those of the first, because nothing sets them back to 0. Keep `opt.zero_grad()` before the forward pass.",
                    nl="Je tweede stap telt zijn gradients op bij die van de eerste, omdat niets ze terugzet naar 0. Houd `opt.zero_grad()` voor de forward pass."),
            Mistake("loss-bf16", wrong=_LOSS_OUT, state="almost",
                    en="With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`: your loss has only the few digits of BF16. Is it calculated outside the `with` block, from the BF16 logits, or turned into BF16? Calculate it inside the block and give it back as it is: there autocast calculates it in FP32.",
                    nl="Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn: je loss heeft maar de paar cijfers van BF16. Wordt hij buiten het `with`-blok uitgerekend, uit de BF16-logits, of omgezet naar BF16? Reken hem binnen het blok uit en geef hem terug zoals hij is: daar rekent autocast hem in FP32 uit."),
            Mistake("shapes", error=lambda e: isinstance(e, (RuntimeError, ValueError)) and ("Expected target size" in str(e) or "Expected input batch_size" in str(e)),
                    en="Python stops: `F.cross_entropy` gets logits and targets whose shapes do not fit together. A GPT gives logits of shape (batch, positions, vocabulary) and targets of shape (batch, positions): give the logits as `logits.reshape(-1, logits.shape[-1])` and the targets as `y.reshape(-1)`, one row per position.",
                    nl="Python stopt: `F.cross_entropy` krijgt logits en targets waarvan de shapes niet bij elkaar passen. Een GPT geeft logits met shape (batch, posities, vocabulary) en targets met shape (batch, posities): geef de logits als `logits.reshape(-1, logits.shape[-1])` en de targets als `y.reshape(-1)`, één rij per positie."),
        ],
        form=[
            Mistake("cpu-only", state="almost",
                    matches=lambda got, text: re.search(r"""autocast\(\s*(?:device_type\s*=\s*)?["']cpu["']|torch\.cpu\.amp""", text) is not None,
                    en="Right here, on the CPU. But your step always asks autocast for the CPU, so on a GPU the layer would calculate in FP32. Pass `device_type` on: `torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed)`.",
                    nl="Goed hier, op de CPU. Maar je stap vraagt autocast altijd voor de CPU, dus op een GPU zou de layer in FP32 rekenen. Geef `device_type` door: `torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=mixed)`."),
            Mistake("no-dtype", state="almost", matches=lambda got, text: "autocast(" in text and "bfloat16" not in text,
                    en="Right here, on the CPU, where autocast picks BF16 by itself. On an NVIDIA GPU it picks FP16, which needs loss scaling. Say what you want: `dtype=torch.bfloat16`.",
                    nl="Goed hier, op de CPU, waar autocast zelf BF16 kiest. Op een NVIDIA-GPU kiest hij FP16, en dat heeft loss scaling nodig. Zeg wat je wilt: `dtype=torch.bfloat16`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the three `...`: the learning rate, the forward pass and the loss under autocast, and the clipping.",
                      "nl": "Vervang de drie `...`: de learning rate, de forward pass en de loss onder autocast, en het clippen."},
            "wrong": {"en": "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does your step set `lr`, run the forward pass and the loss under autocast with `enabled=mixed`, and clip the gradients to 1.0 between backward and step?",
                      "nl": "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Zet je stap `lr`, draait hij de forward pass en de loss onder autocast met `enabled=mixed`, en clipt hij de gradients op 1,0 tussen backward en step?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`. Does your step set `lr`, switch autocast with `mixed`, clip the gradients, and start from zero gradients?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn. Zet je stap `lr`, schakelt hij autocast met `mixed`, clipt hij de gradients, en begint hij met gradients van nul?"},
            "correct": {"en": "Your step is right in all {n} cases: the learning rate of the schedule, BF16 only when `mixed` asks for it, a loss in FP32, and clipped gradients. It is in your project now, as `{saved}`.",
                        "nl": "Je stap klopt in alle {n} gevallen: de learning rate van het schema, BF16 alleen als `mixed` erom vraagt, een loss in FP32, en geclipte gradients. Hij staat nu in je project, als `{saved}`."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("plan-run", _PLAN_START, "empty", None),
    ("plan-run", _PLAN, "correct", None),
    ("plan-run", _PLAN.replace("    return best, int(compute / (6 * sizes[best]))\n", "    return best, compute / (6 * sizes[best])\n"), "correct", None),
    ("plan-run", _PLAN.replace("    params, _ = compute_optimal(", "    params, tokens = compute_optimal(")
     .replace("    return best, int(compute / (6 * sizes[best]))\n", "    return best, int(tokens)\n"), "wrong", "optimal-tokens"),
    ("plan-run", _BELOW, "wrong", "below"),
    ("plan-run", _PLAN.replace("    compute = flops_per_second * seconds\n", "    compute = flops_per_second\n"), "wrong", "no-seconds"),
    ("plan-run", _PLAN.replace("    return best, int(", "    return sizes[best], int("), "wrong", "params-back"),
    ("pretrain-step", _STEP_START, "empty", None),
    ("pretrain-step", _STEP, "correct", None),
    ("pretrain-step", _STEP.replace("    return loss.item()\n", "    return loss\n"), "correct", None),
    ("pretrain-step", _STEP.replace("y.reshape(-1)", "y.flatten()").replace("logits.reshape(-1, logits.shape[-1])", "logits.flatten(0, 1)"), "correct", None),
    ("pretrain-step", _NO_LR, "wrong", "no-lr"),
    ("pretrain-step", _NEVER, "wrong", "never-mixed"),
    ("pretrain-step", _ALWAYS, "wrong", "always-mixed"),
    ("pretrain-step", _NO_CLIP, "wrong", "no-clip"),
    ("pretrain-step", _NO_CLIP.replace("    opt.step()\n", "    opt.step()\n    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"), "wrong", "no-clip"),
    ("pretrain-step", _STEP.replace("    opt.zero_grad()\n", "    opt.zero_grad()\n    model.to(torch.bfloat16)\n"), "wrong", "model-converted"),
    ("pretrain-step", _NO_ZERO, "almost", "no-zero-grad"),
    ("pretrain-step", _LOSS_OUT, "almost", "loss-bf16"),
    ("pretrain-step", _NO_RESHAPE, "error", "shapes"),
    ("pretrain-step", _STEP.replace("device_type=device_type", "device_type=\"cpu\""), "almost", "cpu-only"),
    ("pretrain-step", _STEP.replace(", dtype=torch.bfloat16", ""), "almost", "no-dtype"),
]
