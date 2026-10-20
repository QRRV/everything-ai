# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-sft: sft_example, an instruction and its answer as one example whose labels count only the
answer and the end-of-text; sft_batch, examples of different lengths as one batch of inputs and targets, shifted by one
and padded; words_used, whether a story uses every word an instruction asks for; and a choice. The demo writes
instructions for stories of TinyStories, lets the learner's model of llm-pretrain-capstone (or the course's) answer
held-out instructions, fine-tunes it with the learner's functions and pretrain_step, and lets it answer them again. The
course's own SFT model, data/course-sft.pt, made with the same code from stories of Cosmopedia by
data/tools/train_course_sft.py, is what later lessons fall back on."""
import importlib.util
import json
import math
import random
import re
import time

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _checks("llm-pretrain-capstone", "everything_ai_checks_llm_pretrain_capstone_for_sft")
S = P.S
B = P.B
BUILDS_ON = P.BUILDS_ON + (("llm", "pretrain_step", "llm-pretrain-capstone", "pretrain-step"),)
projects.builds_on(*BUILDS_ON)
SAVES = (("llm", "sft_example", "llm-sft", "sft-example"), ("llm", "sft_batch", "llm-sft", "sft-batch"),
         ("llm", "words_used", "llm-sft", "words-used"))

DATA = "data/tinystories-small.txt"
HELD = 400          # the last stories of DATA: held out, as in llm-eval-base; the test instructions come from them
MAX_LEN = 128       # ids per example at most: the instruction, the answer and the end-of-text
ROOM = 28           # ids an instruction may take, so that the answer gets the rest: the longest of WORDS take 28
TESTS = 100         # held-out instructions, every other one with one word, the others with two
STEPS, LR = 1000, 1e-3   # SFT steps, and their learning rate: a third of the pretraining's PEAK, with its schedule
BATCH = P.BATCH     # examples per mini-batch, on a CPU and on a GPU
LENGTH, TAU = P.LENGTH, P.TAU   # an answer of at most LENGTH tokens, drawn at temperature TAU
NAME = "sft"        # the fine-tuned model in work/projects/llm/: sft.pt, with sft.json
COURSE = "data/course-sft"   # the course's SFT model, .pt and .json
# Nouns the instructions ask for: things and places that the stories of TinyStories often name, in the singular.
WORDS = tuple(sorted("""
apple bag balloon ball barn basket bathroom beach bear bed bee bench bird boat bone book bottle bowl box branch bridge
bug bus bush butterfly button cage cake car castle cat cave chair closet cloud couch cow dog doll door duck egg farm
fence field fire fish floor flower forest fountain fridge frog fruit garage garden gate grass hill hole house island jar
jungle key kitchen kite kitten ladder lake leaf library mirror monster moon mountain mouse mud museum nest ocean oven
paper park path phone picture pie pillow playground pond pool pot puddle puzzle rabbit rain river road rock rocket roof
room rope sand sea sheep shelf shop sink sky slide snow snowman soup squirrel star stick stone store storm street sun
swing table tent tower town toy train tree truck tunnel turtle village wall water wind window yard zoo
""".split()))
PUNCTUATION = ".,!?;:\"'"   # what words_used strips from the ends of a word
ENDS = (".", "!", "?", ".\"", "!\"", "?\"")   # a sentence ends with a token that ends in one of these


def number(x: float, digits: int, lang: str) -> str:
    return S.number(x, digits, lang)


def instruction(words: list) -> str:
    """The instruction for a story with `words`, in the course's template: its last line, "Story:", is the generation
    prompt, and the answer starts on the line after it."""
    if len(words) == 1:
        return f"Write a story with the word: {words[0]}.\nStory:\n"
    return f"Write a story with the words: {' and '.join(words)}.\nStory:\n"


def found(story: str) -> set:
    """The words of a story in small letters, without the punctuation around them, as the course's words_used sees them."""
    return {w.strip(PUNCTUATION) for w in story.lower().split()}


def stories() -> list:
    return [s.strip() for s in (paths.ROOT / DATA).read_text(encoding="utf-8").split(B.END) if s.strip()]


def tests() -> list:
    """TESTS lists of requested words from the HELD last stories: every other one with one word, the others with two,
    drawn from the words of WORDS that the story uses."""
    rng = random.Random(1)
    out = []
    for story in stories()[-HELD:]:
        chosen = sorted(w for w in found(story) if w in WORDS)
        if len(chosen) >= 2:
            out.append(rng.sample(chosen, 2 if len(out) % 2 else 1))
        if len(out) == TESTS:
            break
    return out


def pairs(texts: list, merges: dict, encode_fast, decode, seed: int = 0) -> list:
    """Instructions and answers from stories: each story cut after the last sentence that fits in MAX_LEN - ROOM - 1
    ids, and one or two words of WORDS that the cut story uses, drawn at random. A tuple (words, the ids of the
    instruction, the ids of the answer) per story that has such a word. The stories are encoded all at once, with an
    end-of-text between every two, as in pretraining."""
    end = 256 + len(merges)
    ids = encode_fast(B.END.join(texts) + B.END, merges)
    pieces = {}
    rng = random.Random(seed)
    out, story = [], []
    for i in ids:
        if i != end:
            story.append(i)
            continue
        cut = None
        for k in range(min(len(story), MAX_LEN - ROOM - 1)):
            if k + 1 < len(story) and not _piece(pieces, story[k + 1], merges, decode)[:1].isspace():
                continue   # ."A" and the like: the sentence goes on
            if _piece(pieces, story[k], merges, decode).rstrip().endswith(ENDS):
                cut = k + 1
        if cut is not None:
            answer = story[:cut]
            chosen = sorted(w for w in found(decode(answer, merges)) if w in WORDS)
            if chosen:
                words = rng.sample(chosen, min(len(chosen), rng.choice((1, 2))))
                prompt = encode_fast(instruction(words), merges)
                if len(prompt) <= ROOM:   # so that the whole example fits in MAX_LEN
                    out.append((words, prompt, answer))
        story = []
    return out


def _piece(cache: dict, i: int, merges: dict, decode) -> str:
    if i not in cache:
        cache[i] = decode([i], merges)
    return cache[i]


PICTURE = (["kite"], "Tom flew his kite.")   # the example of the picture


def example_picture(labels: dict):
    """PICTURE as an example in the template, cut into the tokens of the model the demo fine-tunes."""
    info = P.saved() or P.course_info()
    merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
    t = run_code("from llm.encode import encode\nfrom llm.decode import decode").names
    words, story = PICTURE
    return tokens_picture(t["encode"](instruction(words), merges), t["encode"](story, merges), 256 + len(merges), merges,
                          t["decode"], labels)


def tokens_picture(prompt: list, answer: list, end_id: int, merges: dict, decode, labels: dict):
    """An example cut into tokens: the ids, and under them the labels, -100 for the instruction. Ink only: tokens are
    input; the labels that count in the loss are in the colour of the loss."""
    import html
    from everything_ai import draw
    ids = prompt + answer + [end_id]
    marks = [-100] * len(prompt) + answer + [end_id]
    W, GAP, BH = 380, 5, 30
    mono = "font-family:var(--ea-font-code)"
    parts = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    x, y = 0, 30
    for i, mark in zip(ids, marks):
        shown = labels["end"] if i == end_id else decode([i], merges).replace(" ", "␣").replace("\n", "↵")
        bw = max(38, 12 + 9 * len(shown))
        if x + bw > W:
            x, y = 0, y + 62
        counted = mark != -100
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{BH}" rx="5" style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"/>'
                     f'<text x="{x + bw / 2}" y="{y + BH / 2 + 5}" style="{mono};font-size:15px;fill:var(--ea-ink);'
                     f'text-anchor:middle;white-space:pre">{html.escape(shown)}</text>'
                     f'<text x="{x + bw / 2}" y="{y + BH + 16}" style="{mono};font-size:12px;'
                     f'fill:var({"--ea-loss-text" if counted else "--ea-ink-3"});font-weight:{600 if counted else 400};'
                     f'text-anchor:middle">{mark}</text>')
        x += bw + GAP
    return draw._svg(W, y + BH + 24, labels["alt"], "".join(parts), natural=True)


# --- the demo ----------------------------------------------------------------------------------------------------------

def tools() -> dict | str:
    """The learner's modules the demo needs, or "run" when one stops with an error."""
    ran = run_code("from llm.fast_encode import encode_fast\nfrom llm.encode import encode\nfrom llm.decode import decode\n"
                   "from llm.pretrain_step import pretrain_step")
    return "run" if ran.error is not None else ran.names


def answer(model, prompt: list, end_id: int, seed: int) -> tuple:
    """An answer of at most LENGTH tokens to the ids of an instruction, drawn at temperature TAU with the model's
    caches; the ids, and whether it stopped by itself at an end-of-text."""
    import torch
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    caches = [{} for _ in model.blocks]
    model.eval()
    out = []
    with torch.no_grad():
        with P._on(device.type):
            logits = model(torch.tensor([prompt], device=device), caches)
        for _ in range(LENGTH):
            p = torch.softmax(logits[0, -1].double().cpu() / TAU, dim=0)
            j = int(torch.multinomial(p, 1, generator=g))
            if j == end_id:
                return out, True
            out.append(j)
            with P._on(device.type):
                logits = model(torch.tensor([[j]], device=device), caches)
    return out, False


def rate(model, merges: dict, words_used, encode_fast, decode, show, phase: str) -> dict | str:
    """The model's answers to the TESTS held-out instructions: how many use every requested word, by the learner's
    words_used, and how many stop by themselves. "words" when words_used stops with an error or gives no bool."""
    end = 256 + len(merges)
    asked = tests()
    used = stopped = 0
    answers, hits = [], []
    start = time.perf_counter()
    for k, words in enumerate(asked):
        ids, ended = answer(model, encode_fast(instruction(words), merges), end, seed=k)
        text = decode(ids, merges)
        try:
            ok = words_used(text, list(words))
        except Exception:
            return "words"
        if not isinstance(ok, bool):
            return "words"
        used += ok
        stopped += ended
        answers.append(text)
        hits.append(ok)
        show({"phase": phase, "done": k + 1, "total": len(asked), "used": used})
    return {"used": used, "stopped": stopped, "total": len(asked), "answers": answers, "hits": hits, "words": asked,
            "seconds": time.perf_counter() - start}


SNAPSHOTS = 5   # moments of the run, after the start, at which tune lets the model answer: for the replay of the run


def tune(model, examples: list, sft_batch, step_fn, end_id: int, show, steps: int | None = None, seed: int = 0,
         lr: float | None = None, snap=None) -> dict | str:
    """Fine-tune `model` on `examples` with the learner's sft_batch and pretrain_step for `steps` steps (STEPS when
    None): AdamW on the parameters that need a gradient, at `lr` (LR when None) times the schedule of
    llm-open-recipes, mini-batches of BATCH examples in a shuffled order, from the start again when they run out.
    `show` hears a step every second, with every (step, loss) since it last heard ("fresh"), for a live chart. With
    `snap(step, model)`, the model answers before the first step and after every steps / SNAPSHOTS steps, for the
    replay of the run ("snapshots"). "batch" when sft_batch stops with an error, "run" when pretrain_step does,
    "memory" when the device runs out of it, "diverged" when the loss is no number."""
    import torch
    steps = STEPS if steps is None else steps
    lr = LR if lr is None else lr
    w = P.where()
    model.to(w["device"]).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    batch = BATCH[w["kind"]]
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    plan = P.saved_plan()
    mixed = bool(plan and plan.get("mixed") and plan.get("device") == w["device"])
    losses, recent, fresh, seconds, last_show = [], [], [], 0.0, time.perf_counter()
    marks = {round(steps * k / SNAPSHOTS) for k in range(1, SNAPSHOTS + 1)}
    snapshots = [] if snap is None else [snap(0, model)]
    model.train()
    for step in range(steps):
        start = time.perf_counter()
        chosen = [examples[order[(step * batch + k) % len(order)]] for k in range(batch)]
        try:
            x, y = sft_batch(chosen, end_id)
        except Exception:
            return "batch"
        try:
            with P._on(w["device"]):
                loss = float(step_fn(model, opt, x.to(w["device"]), y.to(w["device"]), lr * S.multiplier(step, steps), w["device"], mixed))
        except Exception as e:
            return "memory" if "out of memory" in str(e).lower() else "run"
        if not math.isfinite(loss):
            return "diverged"
        P._sync(w["device"])
        seconds += time.perf_counter() - start
        recent.append(loss)
        fresh.append((step + 1, loss))
        if snap is not None and step + 1 in marks:
            snapshots.append(snap(step + 1, model))
            model.train()
        if (step + 1) % 50 == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": sum(recent) / len(recent)})
            recent = []
        if time.perf_counter() - last_show > 1 or step + 1 == steps:
            show({"phase": "train", "step": step + 1, "steps": steps, "loss": loss, "seconds": seconds, "fresh": fresh})
            fresh, last_show = [], time.perf_counter()
    model.eval()
    return {"steps": steps, "seconds": seconds, "losses": losses, "device": w["device"], "name": w["name"], "mixed": mixed,
            "snapshots": snapshots}


def snapper(merges: dict, words_used, encode_fast, decode):
    """For the replay of a run: snap(step, model), which lets the model answer the first test instruction as rate
    answers it, with the same seed at every moment, and says whether the answer used every word and stopped by itself."""
    end = 256 + len(merges)
    words = list(tests()[0])
    asked = encode_fast(instruction(words), merges)

    def snap(step, model):
        ids, ended = answer(model, asked, end, seed=0)
        text = decode(ids, merges)
        try:
            used = bool(words_used(text, list(words)))   # a copy, as in rate: a words_used may empty the list it gets
        except Exception:
            used = False
        return {"step": step, "words": list(words), "text": text, "ended": ended, "used": used}
    return snap


def run(sft_example, sft_batch, words_used, show) -> dict | str:
    """The demo: instructions for the stories, the model's answers to the held-out instructions before SFT, SFT,
    and its answers after; then the model is kept in work/projects/llm/sft.pt. A string when it stops: "run",
    "example", "batch", "words", "memory" or "diverged"."""
    import torch
    t = tools()
    if t == "run":
        return "run"
    try:
        model, info, source = P.load()
        merges = {tuple(pair): 256 + i for i, pair in enumerate(info["merges"])}
        end = 256 + len(merges)
        made = pairs(stories()[:-HELD], merges, t["encode_fast"], t["decode"])
    except Exception:
        return "run"
    try:
        examples = [sft_example(list(prompt), list(answer_), end) for _, prompt, answer_ in made]
        if any(len(ids) != len(labels) for ids, labels in examples):
            return "example"
    except Exception:
        return "example"
    show({"phase": "pairs", "pairs": len(examples)})
    model.to(P.where()["device"])
    before = rate(model, merges, words_used, t["encode_fast"], t["decode"], show, "before")
    if isinstance(before, str):
        return before
    torch.manual_seed(0)
    tuned = tune(model, examples, sft_batch, t["pretrain_step"], end, show, snap=snapper(merges, words_used, t["encode_fast"], t["decode"]))
    if isinstance(tuned, str):
        return tuned
    after = rate(model, merges, words_used, t["encode_fast"], t["decode"], show, "after")
    if isinstance(after, str):
        return after
    result = {"model": model, "source": source, "pairs": len(examples), "params": sum(p.numel() for p in model.parameters()),
              "before": before, "after": after, **tuned}
    save(result, info)
    return result


def save(result: dict, info: dict) -> None:
    """The fine-tuned model in work/projects/llm/sft.pt, and what the run measured in sft.json, for the lessons after
    this one: also, for every test instruction in order, 1 when the answer used every word and 0 when not."""
    import torch
    folder = projects.folder() / "llm"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in result["model"].state_dict().items()}, folder / f"{NAME}.pt")
    keep = {"sizes": info["sizes"], "merges": info["merges"], "base": result["source"], "params": result["params"],
            "pairs": result["pairs"], "steps": result["steps"], "lr": LR, "seconds": round(result["seconds"], 1),
            "device": result["device"], "losses": result["losses"], "snapshots": result.get("snapshots", []),
            **{when: {k: result[when][k] for k in ("used", "stopped", "total")} for when in ("before", "after")},
            "example": example(result["before"], result["after"]),
            "hits": {when: "".join("1" if h else "0" for h in result[when]["hits"]) for when in ("before", "after")}}
    (folder / f"{NAME}.json").write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")


def replay_frame(run: dict, step: int, labels: dict, reference: list | None = None) -> tuple:
    """One moment of a run, for the replay of the run (L.explore): the loss along the run, the mean of every 50 steps,
    with a line at `step`; and what the model wrote at that moment to the first test instruction. `labels`: title,
    x, y, alt, lang (for the numbers), and the words of the sentence under it: "after" ({step}), "start", "used",
    "not_used", "stopped", "not_stopped". With `reference`, another run's loss as (step, loss), it is drawn thin and
    dashed beside it, with labels["reference"] at its end, on the side away from this run."""
    from everything_ai import chart, live
    snap = next(s for s in run["snapshots"] if s["step"] == step)
    xs, ys = [l["step"] for l in run["losses"]], [l["loss"] for l in run["losses"]]
    top = math.ceil(max(ys + [v for _, v in reference or []]) * 2) / 2
    p = chart.Panel((0, run["steps"]), (0, top), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"])
    if reference:
        p.line([x for x, _ in reference], [v for _, v in reference], "ink-2", 1.4, dashed=True)
        above = ys[-1] < reference[-1][1]   # its name on the side away from this run
        at_x, at_y = live.name_at(reference, labels["reference"], above, run["steps"], (0, top))
        p.text(at_x - 0.03 * run["steps"], at_y, labels["reference"], "ink-2", "end")   # clear of the line at the last moment
    p.line(xs, ys, "loss", 2.2)
    p.line([step, step], [0, top], "ink-2", 1.2, dashed=True)
    done = [(x, y) for x, y in zip(xs, ys) if x <= step]
    if done:
        p.points([done[-1][0]], [done[-1][1]], "loss", r=4)
    svg = chart.figure([p], alt=labels["alt"], width=360, height=220)
    when = labels["start"] if step == 0 else labels["after"].format(step=number(step, 0, labels.get("lang", "en")))
    verdict = f"{labels['used'] if snap['used'] else labels['not_used']}; {labels['stopped'] if snap['ended'] else labels['not_stopped']}"
    return svg, f"**{when}** · {verdict}\n\n{written(snap['text'])}"


def written(text: str) -> str:
    """What a model wrote, in Markdown for the sentence under a picture: every line in italics on its own line, in the
    serif of a quote (<i>, as play.Explore writes it), with nothing in it that Markdown or KaTeX would read as a sign,
    and no spaces in front that would make it code: also no link from an address (: and .), no ^superscript^ and
    no ++key++."""
    signs = {"&": "&amp;", "<": "&lt;", ">": "&gt;", **{c: f"&#{ord(c)};" for c in "\\`*_$[]|~#^+:."}}
    lines = ["".join(signs.get(c, c) for c in line.strip()) for line in text.split("\n") if line.strip()]
    return "  \n".join(f"<i>{line}</i>" for line in lines) or "…"


def example(before: dict, after: dict) -> dict:
    """The first held-out instruction that the model did after SFT and not before, with both answers; the first
    instruction when there is none."""
    k = next((k for k, (b, a) in enumerate(zip(before["hits"], after["hits"])) if a and not b), None)
    better = k is not None
    k = k if better else 0
    return {"words": after["words"][k], "before": before["answers"][k], "after": after["answers"][k], "better": better}


def saved() -> dict | None:
    """The learner's SFT run, from work/projects/llm/sft.json, when its model is there too; else None."""
    folder = projects.folder() / "llm"
    try:
        info = json.loads((folder / f"{NAME}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if (folder / f"{NAME}.pt").is_file() else None


def load(source: str | None = None) -> tuple:
    """The SFT model the lessons after this one go on with: the learner's when there is one (or "yours" is asked),
    else the course's; with its information and where it comes from, "yours" or "course"."""
    import torch
    ran = run_code("from llm.modern_gpt import ModernGPT")
    info = saved() if source in (None, "yours") else None
    if info is not None:
        where, state_path = "yours", projects.folder() / "llm" / f"{NAME}.pt"
    else:
        where = "course"
        info = course_info()
        state_path = (paths.ROOT / COURSE).with_suffix(".pt")
    vocab, d, h, kv, depth = info["sizes"]
    model = ran.names["ModernGPT"](vocab, d, h, kv, depth)
    model.load_state_dict({k: v.float() for k, v in torch.load(state_path, map_location="cpu", weights_only=True).items()})
    return model.eval(), info, where


def course_info() -> dict:
    return json.loads((paths.ROOT / COURSE).with_suffix(".json").read_text(encoding="utf-8"))


def write_for(model, merges: dict, words: list, encode_fast, decode, seed: int = 0) -> tuple:
    """The model's answer to the instruction for `words`, and whether it stopped by itself."""
    ids, ended = answer(model, encode_fast(instruction(words), merges), 256 + len(merges), seed)
    return decode(ids, merges), ended


def asked_words(text: str) -> list:
    """The words someone typed for an instruction: in small letters, without punctuation, at most two."""
    return [w for w in re.findall(r"[a-z]+", text.lower())][:2]


# --- measured: the course's run of the demo with its own solutions, on the CPU, one thread; "seconds" the SFT steps,
# "other_seconds" the examples and the answers before and after ---
# "yours": on the course's model of the demo of llm-pretrain-capstone (TinyStories, so not shared: only its numbers).
MEASURED = {
    "yours": {"pairs": 5252, "params": 575337, "steps": STEPS, "seconds": 197.9, "other_seconds": 111.0,
              "before": {"used": 9, "stopped": 85, "total": TESTS}, "after": {"used": 29, "stopped": 100, "total": TESTS},
              "example": {"words": ["park"], "before": "They are very kind and understanding. They are happy to be friends again. They have each other's sail. They can eat it again.", "after": "Once upon a time, there was a little girl named Lisa. She was a brave boy who loved to visit her. Every day, she would start his paws to the park with her new toy.\nOne day, Lisa saw a pretty patch of popcorn. She wanted to help her. She tried to push it, but it was too big.", "better": True},
              # for every test instruction in order, 1 when the answer used every word
              "hits": {"before": "0000101000000000100000001000000000000000001000000000000000100000000010000000000010000000100000000000",
                       "after": "1000001011000000000000110110000000000010000100101010100000101000101011001010100100100010100000101000"}},
    # the same run with sft_example without the end-of-text (_E_NO_END), on two threads: the answers after SFT that
    # stopped by themselves
    "no-end": {"stopped": 1, "total": TESTS},
}
# --- end of measured ---

REQUIRED = ["sft-example", "sft-batch", "words-used", "no-end"]

_DOTS = S._DOTS
_TEXTS = S._TEXTS
_CALL, _CALL_NL = S._CALL, S._CALL_NL
_SCENARIO = "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe last line gives `{got}`, and it should give `{want}`"
_SCENARIO_NL = "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\ngeeft de laatste regel `{got}`, en dat moet `{want}` zijn"


def _pair(got, want) -> bool:
    """Two lists, ids and labels, as a tuple or a list of two."""
    return isinstance(got, (tuple, list)) and len(got) == 2 and all(isinstance(g, list) for g in got) and list(got) == list(want)


def _lengths_differ(got, text) -> bool:
    return isinstance(got, (tuple, list)) and len(got) == 2 and all(isinstance(g, list) for g in got) and len(got[0]) != len(got[1])


# --- exercise 1: sft_example ---------------------------------------------------------------------------------------------

_E_HEAD = "def sft_example(prompt_ids, answer_ids, end_id):\n"
_E = (_E_HEAD + "    ids = prompt_ids + answer_ids + [end_id]\n"
      "    labels = [-100] * len(prompt_ids) + answer_ids + [end_id]\n"
      "    return ids, labels\n")
_E_START = (_E_HEAD + "    # the whole example: the instruction, the answer and the end-of-text\n    ids = ...\n"
            "    # a label for every id: -100 for the instruction, else the id itself\n    labels = ...\n"
            "    return ids, labels\n")
_E_NO_MASK = _E.replace("[-100] * len(prompt_ids) + answer_ids", "prompt_ids + answer_ids")
_E_NO_END = _E.replace(" + [end_id]\n", "\n")
_E_END_MASKED = _E.replace("answer_ids + [end_id]\n    return", "answer_ids + [-100]\n    return")
_E_ZERO = _E.replace("[-100] * len(prompt_ids)", "[0] * len(prompt_ids)")
_E_REVERSED = _E.replace("[-100] * len(prompt_ids) + answer_ids", "prompt_ids + [-100] * len(answer_ids)")

# --- exercise 2: sft_batch -----------------------------------------------------------------------------------------------

_T_HEAD = "import torch\n\n\ndef sft_batch(examples, end_id):\n"
_T = (_T_HEAD + "    n = max(len(ids) for ids, labels in examples) - 1\n"
      "    x, y = [], []\n"
      "    for ids, labels in examples:\n"
      "        pad = n - (len(ids) - 1)\n"
      "        x.append(ids[:-1] + [end_id] * pad)\n"
      "        y.append(labels[1:] + [-100] * pad)\n"
      "    return torch.tensor(x), torch.tensor(y)\n")
_T_START = (_T_HEAD + "    # the inputs of the longest example: all its ids but the last\n    n = ...\n"
            "    x, y = [], []\n"
            "    for ids, labels in examples:\n"
            "        # the inputs and the targets, shifted by one and padded to n\n        ...\n"
            "    return torch.tensor(x), torch.tensor(y)\n")
_T_SETUP = ("examples = [([4, 6, 1, 8, 2, 0], [-100, -100, -100, 8, 2, 0]),\n"
            "            ([4, 7, 5, 0], [-100, -100, 5, 0])]\n")
_T_SCENARIOS = [
    # the lesson's example: two examples of other lengths
    _T_SETUP + "x, y = sft_batch(examples, 0)\nx.tolist(), y.tolist()",
    # whole numbers, as an embedding and cross_entropy take them
    _T_SETUP + "x, y = sft_batch(examples, 0)\nstr(x.dtype), str(y.dtype)",
    # one example: nothing to pad
    "x, y = sft_batch([([4, 1, 7, 9], [-100, 1, 7, 9])], 9)\nx.tolist(), y.tolist()",
    # three examples, the longest last, and another end-of-text
    "x, y = sft_batch([([3, 4, 9], [-100, 4, 9]), ([2, 9], [-100, 9]), ([3, 3, 5, 6, 9], [-100, -100, 5, 6, 9])], 9)\n"
    "x.tolist(), y.tolist()",
]
_T_NO_SHIFT = (_T_HEAD + "    n = max(len(ids) for ids, labels in examples)\n"
               "    x, y = [], []\n"
               "    for ids, labels in examples:\n"
               "        pad = n - len(ids)\n"
               "        x.append(ids + [end_id] * pad)\n"
               "        y.append(labels + [-100] * pad)\n"
               "    return torch.tensor(x), torch.tensor(y)\n")
_T_PAD_LABELS = _T.replace("labels[1:] + [-100] * pad", "labels[1:] + [end_id] * pad")
_T_BACKWARDS = _T.replace("ids[:-1] + [end_id]", "ids[1:] + [end_id]").replace("labels[1:] + [-100]", "labels[:-1] + [-100]")
_T_FRONT = _T.replace("ids[:-1] + [end_id] * pad", "[end_id] * pad + ids[:-1]").replace("labels[1:] + [-100] * pad", "[-100] * pad + labels[1:]")
_T_ONE_MORE = _T.replace("    n = max(len(ids) for ids, labels in examples) - 1\n", "    n = max(len(ids) for ids, labels in examples)\n")
_T_RAGGED = (_T_HEAD + "    x, y = [], []\n"
             "    for ids, labels in examples:\n"
             "        x.append(ids[:-1])\n"
             "        y.append(labels[1:])\n"
             "    return torch.tensor(x), torch.tensor(y)\n")

# --- exercise 3: words_used ----------------------------------------------------------------------------------------------

_W_HEAD = "def words_used(story, words):\n"
_W = (_W_HEAD + "    found = set()\n"
      "    for word in story.lower().split():\n"
      "        found.add(word.strip(\".,!?;:\\\"'\"))\n"
      "    for word in words:\n"
      "        if word not in found:\n"
      "            return False\n"
      "    return True\n")
_W_START = (_W_HEAD + "    # the words of the story, in small letters, without the punctuation around them\n    found = ...\n"
            "    # True only when every requested word is among them\n    ...\n")
_W_SUBSTRING = (_W_HEAD + "    for word in words:\n"
                "        if word not in story.lower():\n"
                "            return False\n"
                "    return True\n")
_W_CASE = _W.replace("story.lower().split()", "story.split()")
_W_STRIP = _W.replace("found.add(word.strip(\".,!?;:\\\"'\"))", "found.add(word)")
_W_ANY = (_W_HEAD + "    found = set()\n"
          "    for word in story.lower().split():\n"
          "        found.add(word.strip(\".,!?;:\\\"'\"))\n"
          "    for word in words:\n"
          "        if word in found:\n"
          "            return True\n"
          "    return False\n")

EXERCISES = {
    "sft-example": FunctionBox(
        name="sft-example",
        function="sft_example",
        start=_E_START,
        solution=_E,
        visible=[([31, 8, 14, 2], [50, 51, 52], 99)],
        hidden=[([1], [2], 3), ([4, 4], [], 5), ([], [6, 7], 0), ([10, 11, 12], [13, 13], 1256)],
        compare=_pair,
        save_as=("llm", "sft_example"),
        first=[_DOTS],
        mistakes=[
            Mistake("int-end", error=lambda e: isinstance(e, TypeError) and "can only concatenate list" in str(e),
                    en="`end_id` is one number, and `+` only puts lists together. Put it in a list of one: `[end_id]`.",
                    nl="`end_id` is één getal, en `+` zet alleen lijsten aan elkaar. Zet het in een lijst van één: `[end_id]`."),
            Mistake("lengths", matches=_lengths_differ,
                    en=_CALL + ": your `ids` and `labels` are not equally long. Every id needs one label, also the end-of-text: as many `-100` as `prompt_ids` has ids, then the answer and the end.",
                    nl=_CALL_NL + ": je `ids` en `labels` zijn niet even lang. Elk id heeft één label nodig, ook het end-of-text: zoveel keer `-100` als `prompt_ids` ids heeft, en dan het antwoord en het einde."),
            Mistake("no-mask", wrong=_E_NO_MASK,
                    en=_CALL + ": your labels are the ids themselves, so the loss also counts the instruction, and the model learns to write instructions as well. Give every id of the instruction the label `-100`: `[-100] * len(prompt_ids)`.",
                    nl=_CALL_NL + ": je labels zijn de ids zelf, dus de loss telt ook de instructie, en het model leert ook instructies schrijven. Geef elk id van de instructie het label `-100`: `[-100] * len(prompt_ids)`."),
            Mistake("no-end", wrong=_E_NO_END,
                    en=_CALL + ": the example stops after the answer, without the end-of-text. Then the model never learns to stop, and writes on after its story. Add `[end_id]` to the ids and to the labels.",
                    nl=_CALL_NL + ": het voorbeeld stopt na het antwoord, zonder het end-of-text. Dan leert het model nooit stoppen, en schrijft het door na zijn verhaal. Zet `[end_id]` achter de ids en achter de labels."),
            Mistake("end-masked", wrong=_E_END_MASKED,
                    en=_CALL + ": the end-of-text is in the ids, but its label is `-100`, so the loss does not count it and the model never learns to stop. Its label is `end_id` itself.",
                    nl=_CALL_NL + ": het end-of-text staat in de ids, maar zijn label is `-100`, dus de loss telt het niet en het model leert nooit stoppen. Zijn label is `end_id` zelf."),
            Mistake("zero", wrong=_E_ZERO,
                    en=_CALL + ": `0` is the id of a token too, so the model learns to predict that token at every place of the instruction. The label that `F.cross_entropy` skips is `-100`.",
                    nl=_CALL_NL + ": `0` is ook het id van een token, dus het model leert dat token te voorspellen op elke plek van de instructie. Het label dat `F.cross_entropy` overslaat is `-100`."),
            Mistake("reversed", wrong=_E_REVERSED,
                    en=_CALL + ": you hide the answer and count the instruction: the model would learn to write instructions, not answers. `-100` belongs under the instruction, `[-100] * len(prompt_ids)`, and the answer keeps its ids.",
                    nl=_CALL_NL + ": je verbergt het antwoord en telt de instructie: het model zou instructies leren schrijven, geen antwoorden. `-100` hoort onder de instructie, `[-100] * len(prompt_ids)`, en het antwoord houdt zijn ids."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the ids of the whole example, and a label for every id.",
                      "nl": "Vervang de twee `...`: de ids van het hele voorbeeld, en een label voor elk id."},
            "wrong": {"en": _CALL + ". The ids of the instruction, the answer and `end_id`, and as labels `-100` for every id of the instruction, then the answer and `end_id`?",
                      "nl": _CALL_NL + ". De ids van de instructie, het antwoord en `end_id`, en als labels `-100` voor elk id van de instructie, en dan het antwoord en `end_id`?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function work for an empty answer, or an example without an instruction?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Werkt je functie ook voor een leeg antwoord, of een voorbeeld zonder instructie?"},
            "correct": {"en": "Your example holds the instruction, the answer and the end-of-text, and its labels count only the answer and the end. It is in your project now, as `{saved}`.",
                        "nl": "Je voorbeeld bevat de instructie, het antwoord en het end-of-text, en de labels tellen alleen het antwoord en het einde. Het staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `sft_example` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `sft_example` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "sft-batch": ScenarioBox(
        name="sft-batch",
        start=_T_START,
        solution=_T,
        scenarios=_T_SCENARIOS,
        save_as=("llm", "sft_batch"),
        first=[_DOTS],
        mistakes=[
            Mistake("ragged", error=lambda e: isinstance(e, ValueError) and "expected sequence of length" in str(e),
                    en="`torch.tensor` gets rows of different lengths: a tensor is a rectangle. Pad every row to `n`: the inputs with `end_id`, the targets with `-100`.",
                    nl="`torch.tensor` krijgt rijen van verschillende lengtes: een tensor is een rechthoek. Vul elke rij aan tot `n`: de inputs met `end_id`, de doelen met `-100`."),
            Mistake("lists", error=lambda e: isinstance(e, AttributeError) and re.search(r"'(list|tuple)' object has no attribute 'tolist'", str(e)) is not None,
                    en="Your `sft_batch` gives back a list where the check needs a tensor: `.tolist()` only works on a tensor. Give back `torch.tensor(x), torch.tensor(y)`.",
                    nl="Je `sft_batch` geeft een lijst terug waar de check een tensor nodig heeft: `.tolist()` werkt alleen op een tensor. Geef `torch.tensor(x), torch.tensor(y)` terug."),
            Mistake("no-shift", wrong=_T_NO_SHIFT,
                    en=_SCENARIO + ": the target at every place is the id at that place itself, so the model only learns to copy its input. Shift by one: the inputs are the ids without the last, `ids[:-1]`, the targets the labels without the first, `labels[1:]`.",
                    nl=_SCENARIO_NL + ": het doel op elke plek is het id op die plek zelf, dus het model leert alleen zijn input kopiëren. Schuif één op: de inputs zijn de ids zonder de laatste, `ids[:-1]`, de doelen de labels zonder de eerste, `labels[1:]`."),
            Mistake("pad-labels", wrong=_T_PAD_LABELS,
                    en=_SCENARIO + ": the padding of the targets is `end_id`, so the loss counts it, and the model learns to predict an end-of-text after the end. Pad the targets with `-100`, which the loss skips.",
                    nl=_SCENARIO_NL + ": de padding van de doelen is `end_id`, dus de loss telt die mee, en het model leert een end-of-text na het einde te voorspellen. Vul de doelen aan met `-100`, dat de loss overslaat."),
            Mistake("backwards", wrong=_T_BACKWARDS,
                    en=_SCENARIO + ": you shifted the wrong way, so the model predicts every token from the one after it. The inputs are `ids[:-1]`, the targets `labels[1:]`.",
                    nl=_SCENARIO_NL + ": je schoof de verkeerde kant op, dus het model voorspelt elk token uit het token erna. De inputs zijn `ids[:-1]`, de doelen `labels[1:]`."),
            Mistake("front", wrong=_T_FRONT,
                    en=_SCENARIO + ": you pad at the front. Then the real tokens of a short row look at the padding before them, through the causal mask, and those of a long row do not. Pad at the end: `ids[:-1] + [end_id] * pad`.",
                    nl=_SCENARIO_NL + ": je vult vooraan aan. Dan kijken de echte tokens van een korte rij door het causal mask naar de padding ervoor, en die van een lange rij niet. Vul achteraan aan: `ids[:-1] + [end_id] * pad`."),
            Mistake("one-more", wrong=_T_ONE_MORE,
                    en=_SCENARIO + ": every row has one place too many, only padding. The longest example has `len(ids) - 1` inputs: its last id is only a target.",
                    nl=_SCENARIO_NL + ": elke rij heeft één plek te veel, alleen padding. Het langste voorbeeld heeft `len(ids) - 1` inputs: zijn laatste id is alleen een doel."),
            Mistake("dtype", matches=lambda got, text: isinstance(got, tuple) and len(got) == 2 and all(isinstance(g, str) and g.startswith("torch.") for g in got),
                    en=_SCENARIO + ": the model takes whole numbers as inputs, and `F.cross_entropy` takes its targets as whole numbers of the kind `torch.int64`. `torch.tensor` makes that kind by itself from lists of whole numbers, without a `dtype`.",
                    nl=_SCENARIO_NL + ": het model neemt gehele getallen als inputs, en `F.cross_entropy` neemt zijn doelen als gehele getallen van de soort `torch.int64`. `torch.tensor` maakt die soort vanzelf van lijsten met gehele getallen, zonder `dtype`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the inputs of the longest example, and in the loop the two rows to add.",
                      "nl": "Vervang de twee `...`: de inputs van het langste voorbeeld, en in de loop de twee rijen om toe te voegen."},
            "wrong": {"en": _SCENARIO + ". For every example the ids without the last and the labels without the first, the inputs padded with `end_id` and the targets with `-100`, at the end, to `n`?",
                      "nl": _SCENARIO_NL + ". Voor elk voorbeeld de ids zonder de laatste en de labels zonder de eerste, de inputs aangevuld met `end_id` en de doelen met `-100`, achteraan, tot `n`?"},
            "almost": {"en": "Right for the first example. But " + _SCENARIO[0].lower() + _SCENARIO[1:] + ". Does your function use `end_id`, and the length of the longest example in `examples`?",
                       "nl": "Goed voor het eerste voorbeeld. Maar " + _SCENARIO_NL[0].lower() + _SCENARIO_NL[1:] + ". Gebruikt je functie `end_id`, en de lengte van het langste voorbeeld in `examples`?"},
            "correct": {"en": "Your batch has a row per example: the inputs, and the targets one place on, padded so that the loss skips the padding. It is in your project now, as `{saved}`.",
                        "nl": "Je batch heeft een rij per voorbeeld: de inputs, en de doelen één plek verder, aangevuld zo dat de loss de padding overslaat. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `sft_batch` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `sft_batch` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "words-used": FunctionBox(
        name="words-used",
        function="words_used",
        start=_W_START,
        solution=_W,
        visible=[("Tom flew his kite in the park.", ["kite", "park"])],
        hidden=[("The cat sat on the mat.", ["cat", "dog"]), ("The dog chased a caterpillar.", ["cat"]),
                ("Park benches are nice.", ["park"]), ("\"Look, a kite!\" said Ben.", ["kite"]),
                ("The ducks swam in the pond.", ["duck", "pond"]), ("Mia went home.", []),
                ("Sam saw a frog. The frog jumped.", ["frog"]), ("Ben shouted, \"Park!\"", ["park"])],
        python_values=True,
        save_as=("llm", "words_used"),
        first=[_DOTS],
        mistakes=[
            Mistake("substring", wrong=_W_SUBSTRING,
                    en=_CALL + ": you look for the word anywhere in the text, so `cat` is found in *caterpillar* and `duck` in *ducks*. Look among the whole words of the story: split it with `.split()`.",
                    nl=_CALL_NL + ": je zoekt het woord overal in de tekst, dus `cat` wordt gevonden in *caterpillar* en `duck` in *ducks*. Zoek tussen de hele woorden van het verhaal: splits het met `.split()`."),
            Mistake("case", wrong=_W_CASE,
                    en=_CALL + ": at the start of a sentence a word has a capital, *Park*, and the requested words are in small letters. Make the story small first: `story.lower()`.",
                    nl=_CALL_NL + ": aan het begin van een zin heeft een woord een hoofdletter, *Park*, en de gevraagde woorden zijn in kleine letters. Maak het verhaal eerst klein: `story.lower()`."),
            Mistake("punctuation", wrong=_W_STRIP,
                    en=_CALL + ": after `.split()`, punctuation sticks to the words: `park.` is not `park`. Strip it from both ends of every word: `word.strip(\".,!?;:\\\"'\")`.",
                    nl=_CALL_NL + ": na `.split()` plakken leestekens aan de woorden: `park.` is niet `park`. Haal ze van beide kanten van elk woord: `word.strip(\".,!?;:\\\"'\")`."),
            Mistake("any", wrong=_W_ANY,
                    en=_CALL + ": your function says True as soon as one requested word is there. The instruction asks for every word: give False as soon as one is missing, and True only after the loop.",
                    nl=_CALL_NL + ": je functie zegt True zodra één gevraagd woord erin staat. De instructie vraagt om elk woord: geef False zodra er één ontbreekt, en True pas na de loop."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: the words of the story, and whether every requested word is among them.",
                      "nl": "Vervang de twee `...`: de woorden van het verhaal, en of elk gevraagd woord ertussen staat."},
            "wrong": {"en": _CALL + ". The whole words of the story in small letters, without punctuation at their ends, and True only when every requested word is among them?",
                      "nl": _CALL_NL + ". De hele woorden van het verhaal in kleine letters, zonder leestekens aan hun uiteinden, en True alleen als elk gevraagd woord ertussen staat?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". A word with a capital or with punctuation before or after it, a word inside another, a missing word, and no requested word at all?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Een woord met een hoofdletter of met een leesteken ervoor of erachter, een woord binnen een ander, een ontbrekend woord, en helemaal geen gevraagd woord?"},
            "correct": {"en": "Your function says whether a story does what the instruction asks: a check a program can do, without a person reading. It is in your project now, as `{saved}`.",
                        "nl": "Je functie zegt of een verhaal doet wat de instructie vraagt: een check die een programma kan doen, zonder dat iemand meeleest. Hij staat nu in je project, als `{saved}`."},
            "kept": {"en": "Your `words_used` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `words_used` is goed. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("sft-example", _E_START, "empty", None),
    ("sft-example", _E, "correct", None),
    ("sft-example", _E_HEAD + "    labels = [-100] * len(prompt_ids)\n    for i in answer_ids + [end_id]:\n        labels.append(i)\n"
                    "    return prompt_ids + answer_ids + [end_id], labels\n", "correct", None),
    ("sft-example", _E.replace("    return ids, labels\n", "    return [ids, labels]\n"), "correct", None),
    ("sft-example", _E_NO_MASK, "wrong", "no-mask"),
    ("sft-example", _E_NO_END, "wrong", "no-end"),
    ("sft-example", _E_END_MASKED, "wrong", "end-masked"),
    ("sft-example", _E_ZERO, "wrong", "zero"),
    ("sft-example", _E_REVERSED, "wrong", "reversed"),
    ("sft-example", _E.replace("[-100] * len(prompt_ids)", "[-100] * (len(prompt_ids) + 1)"), "wrong", "lengths"),
    ("sft-example", _E.replace("answer_ids + [end_id]\n    return", "answer_ids\n    return"), "wrong", "lengths"),
    ("sft-example", _E.replace("prompt_ids + answer_ids + [end_id]", "prompt_ids + answer_ids + end_id"), "error", "int-end"),
    ("sft-batch", _T_START, "empty", None),
    ("sft-batch", _T, "correct", None),
    ("sft-batch", _T_HEAD + "    n = max(len(ids) for ids, _ in examples)\n"
                  "    x = [ids[:-1] + [end_id] * (n - len(ids)) for ids, _ in examples]\n"
                  "    y = [labels[1:] + [-100] * (n - len(labels)) for _, labels in examples]\n"
                  "    return torch.tensor(x), torch.tensor(y)\n", "correct", None),
    ("sft-batch", _T_NO_SHIFT, "wrong", "no-shift"),
    ("sft-batch", _T_PAD_LABELS, "wrong", "pad-labels"),
    ("sft-batch", _T_BACKWARDS, "wrong", "backwards"),
    ("sft-batch", _T_FRONT, "wrong", "front"),
    ("sft-batch", _T_ONE_MORE, "wrong", "one-more"),
    ("sft-batch", _T_RAGGED, "error", "ragged"),
    ("sft-batch", _T.replace("[end_id] * pad", "[0] * pad"), "almost", None),
    ("sft-batch", _T.replace("return torch.tensor(x), torch.tensor(y)", "return x, y"), "error", "lists"),
    ("sft-batch", _T.replace("torch.tensor(x), torch.tensor(y)", "torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)"), "wrong", "dtype"),
    ("words-used", _W_START, "empty", None),
    ("words-used", _W, "correct", None),
    ("words-used", _W_HEAD + "    found = {w.strip(\".,!?;:\\\"'\") for w in story.lower().split()}\n"
                   "    return set(words) <= found\n", "correct", None),
    ("words-used", _W_SUBSTRING, "wrong", "substring"),
    ("words-used", _W_CASE, "wrong", "case"),
    ("words-used", _W_STRIP, "wrong", "punctuation"),
    ("words-used", _W_ANY, "wrong", "any"),
    ("words-used", _W.replace("word.strip(", "word.rstrip("), "almost", None),
]
