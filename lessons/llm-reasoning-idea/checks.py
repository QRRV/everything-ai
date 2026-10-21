# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-reasoning-idea: few_shot_prompt, a prompt of worked examples of GSM8K with or without
their steps; read_answer, the checker that reads a model's final number; majority_vote, the most common answer of a
few; and two choices and an open question. The play blocks show the course's run (data/course-reasoning-qwen.json):
every question with its direct answer and its chain of thought, and voting and pass@k for k samples. The demo runs
Qwen3-0.6B-Base (everything_ai.qwen3, loaded with the course's rename of llm-same-on-open-model) on the first questions
of the GSM8K test set with the learner's prompt and checker: four examples with only the answer against the same four
with their steps, greedy. On a GPU it also draws samples and votes; on a CPU the votes come from the course's run."""
import json
import re
import time

from everything_ai import paths
from everything_ai.check import Code, FunctionBox, Mistake

GSM8K = "data/gsm8k-sample.jsonl"            # the first 4 train and the first 30 test problems of GSM8K (MIT)
COURSE = "data/course-reasoning-qwen.json"   # the course's run on the 30, made with this file's code
SHOTS = 4                                # worked examples in the prompt, as the Qwen3 report does for GSM8K
QUESTIONS = {"cpu": 10, "gpu": 30}       # test questions of the demo
VOTE = {"cpu": 0, "gpu": 8}              # samples per question for the vote in the demo; on a CPU, the course's run
SAMPLES = 8                              # samples per question in the course's run
TEMPERATURE = 0.7                        # of the samples
NEW = {"direct": 24, "steps": 256}       # tokens an answer may take: the direct answer, and one with steps
STOPS = ("\n\nQuestion", "\n\n\n")       # an answer ends where the model starts a question of its own, or at END
END = "<|endoftext|>"
ZERO_SHOT = " Let's think step by step."   # Kojima and others (2022): steps without examples
QWEN3_REPORT = 59.59                     # Qwen3-0.6B-Base on all 1,319 test problems, 4-shot with steps (Qwen3 Technical Report, 2025, table 8)


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


# --- the data --------------------------------------------------------------------------------------------------------

def _problem(row: dict) -> dict:
    """A problem of GSM8K as the lesson uses it: the question, the worked steps without GSM8K's calculator notes
    (<<16-3-4=9>>), and the answer key as text, the digits of the number after ####."""
    steps, final = row["answer"].split("####")
    return {"question": row["question"].strip(), "steps": re.sub(r"<<[^>]*>>", "", steps).strip(),
            "answer": final.strip().replace(",", "")}


def problems(split: str) -> list:
    """The problems of `split`, "train" (the examples) or "test" (the questions), in the order of GSM8K."""
    rows = [json.loads(line) for line in (paths.ROOT / GSM8K).read_text(encoding="utf-8").splitlines() if line]
    return [_problem(r) for r in rows if r["split"] == split]


def examples() -> list:
    return problems("train")[:SHOTS]


def course() -> dict:
    """The course's run, with every test question's text beside its answer key."""
    run = json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))
    tests = problems("test")
    for q in run["questions"]:
        q["question"] = tests[q["k"]]["question"]
    return run


def zero_shot_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:{ZERO_SHOT}"


# --- the model: Qwen3-0.6B-Base, with the course's code ---------------------------------------------------------------

# The course's rename of llm-same-on-open-model (its exercise 1), so that this lesson loads Qwen3 without that lesson;
# a test checks that the two give the same names.
PARTS = {"embed_tokens": "tok", "layers": "blocks", "input_layernorm": "norm1", "post_attention_layernorm": "norm2",
         "self_attn": "attention", "q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "out",
         "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def rename(name: str) -> str:
    return ".".join(PARTS.get(part, part) for part in name.split(".")[1:])


def downloaded() -> bool:
    from everything_ai import qwen3
    return qwen3.downloaded()


def download_or_stop(button, texts: dict) -> None:
    """Stop the cell with `texts["ask"]` and the button until Qwen3 is downloaded; a click downloads it, with the
    progress in the cell. `texts` has "ask", "busy" ({done} and {total} in MB), "no_net", "not_pinned", "no_room" and
    "failed" ({why}), as in llm-same-on-open-model, which downloads the same files."""
    import errno
    import html
    import urllib.error

    import marimo as mo

    from everything_ai import hub, qwen3
    failed = None
    if not downloaded() and button.value:
        total = round(sum(hub.PINNED[(qwen3.REPO, f)]["size"] for f in qwen3.FILES) / 1e6)
        shown_ = [-1]

        def progress(done, size):
            mb = int(done / 1e6)
            if mb // 20 != shown_[0] // 20:   # every 20 MB
                shown_[0] = mb
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


def where() -> dict:
    """The device of the demo and its name; "gpu" or "cpu" for the settings."""
    import torch
    if torch.cuda.is_available():
        return {"device": "cuda", "name": torch.cuda.get_device_name(0), "kind": "gpu"}
    return {"device": "cpu", "name": "CPU", "kind": "cpu"}


def load(device: str = "cpu"):
    """Qwen3-0.6B-Base in float32 on `device`, and its tokenizer."""
    from everything_ai import qwen3
    return qwen3.load(rename).to(device), qwen3.tokenizer()


def _cut(text: str) -> str:
    for s in STOPS:
        text = text.split(s)[0]
    return text


def generate(model, enc, prompt: str, new: int, n: int = 1, temperature: float = 0.0, seed: int = 0) -> list:
    """`n` answers to `prompt`, each a dict: its text (up to where it stops), its tokens (the forward passes after the
    prompt) and its seconds (from the start until it stopped). Greedy when temperature is 0, else sampled with
    `seed`. The whole prompt goes into an empty KV cache at once, so the course's Attention masks it (it applies the
    causal mask only while the cache is empty); then one new token at a time, for all n answers together."""
    import torch
    from everything_ai import qwen3
    device = next(model.parameters()).device
    end = enc.encode_single_token(END)
    g = torch.Generator(device=device).manual_seed(seed)
    start = time.perf_counter()
    caches = [{} for _ in model.blocks]
    out = [[] for _ in range(n)]
    seconds = [0.0] * n
    done = [False] * n
    with torch.no_grad():
        logits = model(torch.tensor([qwen3.encode(enc, prompt)], device=device), caches)[:, -1]
        if n > 1:
            for c in caches:
                c["k"], c["v"] = c["k"].expand(n, -1, -1, -1), c["v"].expand(n, -1, -1, -1)
            logits = logits.expand(n, -1)
        for _ in range(new):
            if temperature == 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits.float() / temperature, dim=-1), 1, generator=g)[:, 0]
            for i, j in enumerate(nxt.tolist()):
                if not done[i]:
                    out[i].append(j)
                    if j == end or any(s in enc.decode(out[i]) for s in STOPS):
                        done[i], seconds[i] = True, time.perf_counter() - start
            if all(done):
                break
            logits = model(nxt[:, None], caches)[:, -1]
    now = time.perf_counter() - start
    return [{"text": _cut(enc.decode([j for j in ids if j != end])), "tokens": len(ids),
             "seconds": round(seconds[i] if done[i] else now, 2)} for i, ids in enumerate(out)]


# --- a run, scored: the course's or the learner's ---------------------------------------------------------------------

def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _right(got, key: str) -> bool:
    return got is not None and str(got) == key


def scores(run: dict, read_answer, majority_vote) -> dict:
    """What a run of the course says, read with `read_answer` and voted with `majority_vote` (the learner's, or the
    course's): of its questions, how many each prompt gets right; voting and pass@k for k = 1 to SAMPLES; the mean
    tokens and seconds of an answer; and the same for the first QUESTIONS["cpu"], the demo on a CPU."""
    qs = run["questions"]
    right = {kind: [_right(_safe(read_answer, q[kind]["text"]), q["answer"]) for q in qs] for kind in ("direct", "steps", "zero")}
    reads = [[_safe(read_answer, v["text"]) for v in q["votes"]] for q in qs]
    vote = [sum(_right(_safe(majority_vote, r[:k]), q["answer"]) for r, q in zip(reads, qs)) for k in range(1, SAMPLES + 1)]
    passes = [sum(any(_right(x, q["answer"]) for x in r[:k]) for r, q in zip(reads, qs)) for k in range(1, SAMPLES + 1)]
    n, first = len(qs), QUESTIONS["cpu"]
    mean = lambda xs: sum(xs) / len(xs)
    return {"n": n, **{kind: sum(r) for kind, r in right.items()}, "right": right, "vote": vote, "pass": passes,
            "first": {"n": first, "direct": sum(right["direct"][:first]), "steps": sum(right["steps"][:first])},
            "samples": mean([sum(_right(x, q["answer"]) for x in r) / len(r) for r, q in zip(reads, qs)]) * n,
            "tokens": {kind: mean([q[kind]["tokens"] for q in qs]) for kind in ("direct", "steps", "zero")}
            | {"sample": mean([v["tokens"] for q in qs for v in q["votes"]])},
            "seconds": {kind: mean([q[kind]["seconds"] for q in qs]) for kind in ("direct", "steps", "zero")}
            | {"votes": mean([max(v["seconds"] for v in q["votes"]) for q in qs])}}


def course_functions() -> tuple:
    """The course's read_answer and majority_vote, from its solutions."""
    from everything_ai.check import run_code
    return run_code(_R_CODE).names["read_answer"], run_code(_V_CODE).names["majority_vote"]


def phrase_only(run: dict) -> int:
    """How many chains of thought of the run are right when only the number after "The answer is" counts."""
    from everything_ai.check import run_code
    read = run_code(_R_WRONG["no-fallback"]).names["read_answer"]
    return sum(_right(_safe(read, q["steps"]["text"]), q["answer"]) for q in run["questions"])


def _counter(run: dict) -> int:
    """The first question (from 0) of the course's run whose direct answer is right and whose chain of thought is
    wrong, by the course's checker: the one the text after the first play block names."""
    right = SCORES["right"]
    return next(i for i in range(len(run["questions"])) if right["direct"][i] and not right["steps"][i])


def _example(run: dict) -> dict:
    """The vote of the samples of question 1 of the course's run, for the worked example: what the course's checker read
    in each, the answer that comes most often and how often."""
    read, vote = course_functions()
    got = [_safe(read, v["text"]) for v in run["questions"][0]["votes"]]
    top = vote(got)
    return {"read": got, "top": top, "count": got.count(top)}


def run_scores(run: dict) -> dict:
    """The numbers of the table after the demo, for a run of the demo (its answers carry what the learner's checker
    read and whether it was right) or for the course's run (read with the course's checker): right answers direct,
    with steps and by vote (None without samples), and the mean tokens and seconds of an answer."""
    read, vote = course_functions()
    qs = run["questions"]

    def right(q, kind):
        a = q[kind]
        return a["right"] if "right" in a else _right(_safe(read, a["text"]), q["answer"])

    votes = None
    if all("votes" in q for q in qs):
        votes = sum(_right(q["vote"] if "vote" in q else _safe(vote, [_safe(read, v["text"]) for v in q["votes"]]), q["answer"])
                    for q in qs)
    mean = lambda xs: sum(xs) / len(xs)
    return {"n": len(qs), "direct": sum(right(q, "direct") for q in qs), "steps": sum(right(q, "steps") for q in qs),
            "vote": votes, **{key: {kind: mean([q[kind][key] for q in qs]) for kind in ("direct", "steps")}
                              for key in ("tokens", "seconds")}}


def shown(text: str, empty: str = "—") -> str:
    """A model's text in a Markdown line of the lesson as it is: the characters Markdown or KaTeX would read as
    formatting are written as themselves ($ too), a line that looks like an item of a list stays a line, and every
    line break stays one; `empty` for a text without a word."""
    lines = []
    for line in text.strip().split("\n"):
        for ch in "\\`*_[]<>#|~$":
            line = line.replace(ch, "\\" + ch)
        line = re.sub(r"^(\s*\d+)\.", r"\1\\.", line)
        line = re.sub(r"^(\s*)([-+])", r"\1\\\2", line)
        lines.append(line)
    return "  \n".join(lines) if text.strip() else empty


def answer_frame(run: dict, i: int, read_answer, labels: dict) -> tuple:
    """Question i (from 1) of a run, for the play block: the tokens of its direct answer and of its chain of thought as
    two bars, in the colour of what a model writes, each with what `read_answer` read and ✓ or ✗; under it the question,
    the answer key, and both answers. `labels`: title ({i}), x, alt, direct, steps, tokens, read, nothing, key,
    question ({i}, {n}). The x-axis runs to NEW["steps"], the most an answer may take, so that every question has the
    same scale."""
    from everything_ai import chart
    q = run["questions"][i - 1]
    p = chart.Panel((0, NEW["steps"]), (-0.5, 1.6), title=labels["title"].format(i=i), xlabel=labels["x"], yticks=[])
    parts = []
    for row, kind in enumerate(("direct", "steps")):
        a = q[kind]
        got = _safe(read_answer, a["text"])
        ok = _right(got, q["answer"])
        y = 1 - row
        p.rect(0, y - 0.22, a["tokens"], y + 0.22, "pred")
        mark = "✓" if ok else "✗"
        p.text(3, y + 0.5, f"{labels[kind]} · {a['tokens']} {labels['tokens']} · {mark}", "pred", "start")
        read = labels["nothing"] if got is None else str(got)
        parts.append(f"**{labels[kind]}** · {a['tokens']} {labels['tokens']} · {labels['read']} {read} {mark}  \n{shown(a['text'], labels['nothing'])}")
    svg = chart.figure([p], alt=labels["alt"], width=460, height=190)
    said = (f"**{labels['question'].format(i=i, n=len(run['questions']))}** · {labels['key']} **{q['answer']}**  \n"
            f"*{shown(q['question'])}*\n\n" + "\n\n".join(parts))
    return svg, said


MARGIN = 1.3   # room right of k = SAMPLES on the x-axis of the votes chart, for the names of its two lines


def _free_place(s: dict, label: str) -> tuple:
    """Where the name of the greedy line goes: just above or below it, where it is furthest from the two other lines.
    The name is about 0.4 of the x-axis wide for 23 letters (about 7 px a letter on a plot of about 390 px)."""
    n, width = s["n"], len(label) * 7 / 390 * (SAMPLES + MARGIN)

    def at(line, x):   # a line through k = 1 to SAMPLES, straight between its points
        x = min(max(x, 1), SAMPLES)
        i = min(int(x), SAMPLES - 1)
        return line[i - 1] + (line[i] - line[i - 1]) * (x - i)

    best = None
    for x0 in [0.6 + 0.25 * j for j in range(int((SAMPLES - 0.4 - width) / 0.25) + 1)]:
        for y in (s["steps"] + 0.05 * n, s["steps"] - 0.08 * n):
            room = min(abs(at(line, x0 + f * width) - y) for line in (s["vote"], s["pass"]) for f in (0, 0.25, 0.5, 0.75, 1))
            if best is None or room > best[0]:
                best = (room, x0, y)
    return best[1], best[2]


def votes_frame(s: dict, k: int, labels: dict) -> tuple:
    """Voting and pass@k for k samples of every question of the course's run, from `scores`: the votes that are right
    as a solid line, pass@k dashed, both in ink (they count right answers); the greedy chain of thought as a thin line;
    a dashed line at the chosen k in the colour of a setting you choose. The names of voting and pass@k stand right of
    k = SAMPLES, so that the line at k never crosses them. `labels`: title, x, y, vote, pass, greedy, alt, said ({k},
    {vote}, {passes}, {greedy}, {n}, {tokens})."""
    from everything_ai import chart
    n, ks = s["n"], list(range(1, SAMPLES + 1))
    p = chart.Panel((0.5, SAMPLES + 0.5 + MARGIN), (0, n), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=ks)
    p.line([0.5, SAMPLES + 0.5], [s["steps"], s["steps"]], "ink-3", 1.2)
    p.line([k, k], [0, n], "hyper", 1.4, dashed=True)
    p.line(ks, s["vote"], "ink", 2.2)
    p.points(ks, s["vote"], "ink", r=3)
    p.line(ks, s["pass"], "ink", 2.2, dashed=True)
    p.points(ks, s["pass"], "ink", r=3, hollow=True)
    p.text(SAMPLES + 0.2, s["pass"][-1], labels["pass"], "ink", "start")
    low = min(s["vote"][-1], s["steps"])
    p.text(SAMPLES + 0.2, low - 0.06 * n, labels["vote"], "ink", "start")
    x, y = _free_place(s, labels["greedy"])
    p.text(x, y, labels["greedy"], "ink-2", "start")
    svg = chart.figure([p], alt=labels["alt"], width=460, height=260)
    tokens = round(k * s["tokens"]["sample"], -1)
    return svg, labels["said"].format(k=k, vote=s["vote"][k - 1], passes=s["pass"][k - 1], greedy=s["steps"], n=n,
                                      tokens=number(tokens, 0, labels.get("lang", "en")))


# --- the demo --------------------------------------------------------------------------------------------------------

def run(few_shot_prompt, read_answer, majority_vote, show, kind: str | None = None) -> dict | str:
    """The demo: Qwen3-0.6B-Base on the first QUESTIONS test questions, with the learner's prompt (four examples, with
    only the answer and with their steps), greedy, every answer read by the learner's read_answer; with VOTE samples
    at TEMPERATURE per question, voted with majority_vote, when VOTE of the device is more than 0. `kind` is "cpu" or
    "gpu": that of the device when None. `show` hears {"phase": "load"}, then {"phase": "answer", "i", "n", "kind"}
    before an answer and {"phase": "answered", "i", "n", "kind", "tokens", "seconds", "right"} after it, and
    {"phase": "question", "i", "n", "direct", "steps"} after each question with the right answers so far. A string
    when it stops: "prompt" (few_shot_prompt), "read" (read_answer), "vote" (majority_vote), "run" (loading), "memory"."""
    import torch
    w = where()
    kind = kind or w["kind"]
    begin = time.perf_counter()
    shots, tests = examples(), problems("test")[:QUESTIONS[kind]]
    prompts = []
    for p in tests:
        two = [_safe(few_shot_prompt, shots, p["question"], s) for s in (False, True)]
        if not all(isinstance(x, str) and x for x in two):
            return "prompt"
        prompts.append(two)
    show({"phase": "load"})
    try:
        model, enc = load(w["device"])
    except Exception as e:
        return "memory" if "out of memory" in str(e).lower() else "run"
    result = {"device": w["name"], "kind": kind, "questions": []}
    so_far = {"direct": 0, "steps": 0}
    for i, (p, two) in enumerate(zip(tests, prompts), 1):
        q = {"k": i - 1, "answer": p["answer"], "question": p["question"]}
        for kind_, prompt in zip(("direct", "steps"), two):
            show({"phase": "answer", "i": i, "n": len(tests), "kind": kind_})
            try:
                a = generate(model, enc, prompt, NEW[kind_])[0]
            except torch.OutOfMemoryError:
                return "memory"
            try:
                a["read"] = read_answer(a["text"])
            except Exception:
                return "read"
            a["right"] = _right(a["read"], p["answer"])
            so_far[kind_] += a["right"]
            q[kind_] = a
            show({"phase": "answered", "i": i, "n": len(tests), "kind": kind_, "tokens": a["tokens"],
                  "seconds": a["seconds"], "right": a["right"]})
        if VOTE[kind]:
            show({"phase": "answer", "i": i, "n": len(tests), "kind": "votes"})
            try:
                q["votes"] = generate(model, enc, two[1], NEW["steps"], n=VOTE[kind], temperature=TEMPERATURE, seed=i - 1)
            except torch.OutOfMemoryError:
                return "memory"
            try:
                q["vote"] = majority_vote([read_answer(v["text"]) for v in q["votes"]])
            except Exception:
                return "vote"
        result["questions"].append(q)
        show({"phase": "question", "i": i, "n": len(tests), "direct": so_far["direct"], "steps": so_far["steps"]})
    result["seconds"] = time.perf_counter() - begin
    return result


def so_far(run: dict, kind: str, n: int) -> list:
    """(question, right so far) of the first n questions of a run, read with the course's checker: the course's line in
    the live chart."""
    read, _ = course_functions()
    out, total = [], 0
    for i, q in enumerate(run["questions"][:n], 1):
        total += _right(_safe(read, q[kind]["text"]), q["answer"])
        out.append((i, total))
    return out


# --- exercise 1: few_shot_prompt --------------------------------------------------------------------------------------

_F_HEAD = "def few_shot_prompt(examples, question, steps):\n"
_F_LOOP = "    parts = []\n    for example in examples:\n"
_F_STEPS = ("        if steps:\n"
            "            answer = example[\"steps\"] + \"\\nThe answer is \" + example[\"answer\"] + \".\"\n"
            "        else:\n"
            "            answer = \"The answer is \" + example[\"answer\"] + \".\"\n")
_F_PART = "        parts.append(\"Question: \" + example[\"question\"] + \"\\nAnswer: \" + answer + \"\\n\\n\")\n"
_F_END = "    parts.append(\"Question: \" + question + \"\\nAnswer:\")\n    return \"\".join(parts)\n"


def _f(steps: str = _F_STEPS, part: str = _F_PART, end: str = _F_END) -> str:
    return _F_HEAD + _F_LOOP + steps + part + end


_F_CODE = _f()
_F_START = (_F_HEAD
            + "    # every example: \"Question: \", its question, a new line, \"Answer: \", with steps its steps and a new line,\n"
              "    # then \"The answer is \", its answer and a full stop; a blank line after it\n"
              "    ...\n"
              "    # the new question in the same way, up to \"Answer:\"\n"
              "    ...\n")
_F_WRONG = {
    "answer-first": _f(steps=_F_STEPS.replace("example[\"steps\"] + \"\\nThe answer is \" + example[\"answer\"] + \".\"",
                                              "\"The answer is \" + example[\"answer\"] + \".\\n\" + example[\"steps\"]")),
    "no-answer-end": _f(end="    parts.append(\"Question: \" + question)\n    return \"\".join(parts)\n"),
    "no-blank": _f(part=_F_PART.replace("\"\\n\\n\")", "\"\\n\")")),
    "no-phrase": _f(steps=("        if steps:\n            answer = example[\"steps\"]\n"
                           "        else:\n            answer = example[\"answer\"]\n")),
    "steps-always": _f(steps="        answer = example[\"steps\"] + \"\\nThe answer is \" + example[\"answer\"] + \".\"\n"),
    "space-end": _f(end="    parts.append(\"Question: \" + question + \"\\nAnswer: \")\n    return \"\".join(parts)\n"),
}

_ONE = [{"question": "Tom has 3 apples and buys 4 more. How many apples does he have?",
         "steps": "Tom has 3 + 4 = 7 apples.", "answer": "7"}]
_TWO = [{"question": "A pen costs $2. How much do 5 pens cost?", "steps": "5 pens cost 5 * 2 = $10.", "answer": "10"},
        {"question": "Sam reads 12 pages a day for 3 days, then 5 more. How many pages does he read?",
         "steps": "In 3 days he reads 12 * 3 = 36 pages.\nWith 5 more, he reads 36 + 5 = 41 pages.\nSo he reads 41 pages.",
         "answer": "41"}]


def _f_cases() -> tuple:
    shots, first = examples(), problems("test")[0]["question"]
    visible = [(shots, first, True)]
    hidden = [(shots, first, False), (_ONE, "Ann has 9 cats and gives 2 away. How many are left?", True),
              (_ONE, "Ann has 9 cats and gives 2 away. How many are left?", False), (_TWO, "What is 12 times 12?", True)]
    return visible, hidden


_F_VISIBLE, _F_HIDDEN = _f_cases()


def _named(examples_: list) -> str:
    """How a message names the examples of a case: the lesson's own, or a short list of its own."""
    if examples_ == examples():
        return "examples"
    return f"[{len(examples_)} example{'s' if len(examples_) > 1 else ''}]"


def _first_difference(got: str, want: str) -> dict:
    """The first line where two prompts differ: its number, and the line in each, as code; or the end of the shorter."""
    a, b = got.split("\n"), want.split("\n")
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else None
        y = b[i] if i < len(b) else None
        if x != y:
            return {"line": i + 1, "got": _line(x), "want": _line(y)}
    return {"line": 1, "got": _line(got), "want": _line(want)}


def _line(text) -> dict | str:
    if text is None:
        return {"en": "no line", "nl": "geen regel"}
    shown_ = text if len(text) <= 90 else text[:60] + " … " + text[-25:]
    return "`" + Code('"' + shown_.replace("\\", "\\\\").replace('"', '\\"') + '"') + "`"


class _PromptBox(FunctionBox):
    """The check of few_shot_prompt: a prompt is long, so a message names its case in a few words, and shows only the
    first line that differs."""

    def _call_text(self, case) -> str:
        examples_, question, steps = case
        q = question if len(question) <= 40 else question[:36] + " …"
        return f"few_shot_prompt({_named(examples_)}, \"{q}\", {steps})"

    def message(self, key: str, **values):
        if key in ("wrong", "almost") and isinstance(values.get("got"), str) and isinstance(values.get("want"), str):
            values = {**values, **_first_difference(values["got"], values["want"])}
        return super().message(key, **values)


def _not_text(got, text) -> bool:
    return not isinstance(got, str)


# --- exercise 2: read_answer ------------------------------------------------------------------------------------------

_R_NUMBER = ("def number(word):\n"
             "    # the word as a number of digits, or None: without its commas, and without $ % . : ; ! ? * ( ) at its ends\n"
             "    word = \"\".join(word.split(\",\")).strip(\"$%.:;!?*()\")\n"
             "    if word.isdigit():\n"
             "        return word\n"
             "    return None\n\n\n")
_R_HEAD = ("def read_answer(text):\n"
           "    # the number after the first \"The answer is\"; without that phrase, the last word that is a number\n")
_R_PHRASE = ("    if \"The answer is\" in text:\n"
             "        for word in text.split(\"The answer is\")[1].split():\n"
             "            if number(word) is not None:\n"
             "                return number(word)\n"
             "        return None\n")
_R_LAST = ("    for word in reversed(text.split()):\n"
           "        if number(word) is not None:\n"
           "            return number(word)\n"
           "    return None\n")


def _r(numbers: str = _R_NUMBER, phrase: str = _R_PHRASE, last: str = _R_LAST) -> str:
    return numbers + _R_HEAD + phrase + last


_R_CODE = _r()
_R_START = (_R_NUMBER.split("    word =")[0] + "    ...\n\n\n" + _R_HEAD + "    ...\n")
_R_WRONG = {
    "last-phrase": _r(phrase=_R_PHRASE.replace("split(\"The answer is\")[1]", "split(\"The answer is\")[-1]")),
    "no-fallback": _r(last="    return None\n"),
    "first-number": _r(last=_R_LAST.replace("reversed(text.split())", "text.split()")),
    "comma": _r(numbers=_R_NUMBER.replace("\"\".join(word.split(\",\"))", "word")),
    "dollar": _r(numbers=_R_NUMBER.replace("strip(\"$%.:;!?*()\")", "strip(\"%.:;!?*()\")")),
    "minus": _r(numbers=_R_NUMBER.replace("strip(\"$%.:;!?*()\")", "strip(\"$%.:;!?*()-\")")),
    "index": _r(last="    words = []\n    for word in text.split():\n        if number(word) is not None:\n"
                     "            words.append(number(word))\n    return words[-1]\n"),
}


def _r_cases() -> tuple:
    """The texts the check reads: answers of Qwen3 from the course's run, and a few made for the check; last, every
    other chain of thought of the run, so that a read_answer that passes scores the run as the course's does (the
    message of "correct" says so)."""
    run = course()
    texts = [q["steps"]["text"] for q in run["questions"]] + [q["direct"]["text"] for q in run["questions"][:3]]
    with_phrase = next(t for t in texts if "The answer is" in t)
    without = [t for t in texts if "The answer is" not in t][:3]
    made = ["The answer is 18 dollars.\n\nQuestion: Ann buys 3 pens. The answer is 20.",
            "The answer is $1,250.", "P = $19.50 / 0.75\nP = $26.", "First she pays $5, then $12.\nSo she pays $17 in all.",
            "So the answer is **42**.", "She has none left.", "", "The answer is 7 more than 3, so 10 in all.",
            "The answer is not given: Tom forgot to say.", "So she needs 40 - 60 = -20 cups.\nThe answer is -20."]
    rest = [t for t in texts if t != with_phrase and t not in without]
    return [(with_phrase,)], [(t,) for t in without + made + rest]


class _Nothing:
    """None as an answer of read_answer or majority_vote: a right outcome of its own, not a function that forgot its
    return, so the check compares it as a value and shows it as `None`."""

    def __repr__(self):
        return "None"


NOTHING = _Nothing()


def _kept(value):
    return NOTHING if value is None else value


def _same_answer(got, want) -> bool:
    """The same number of digits as text; a number counts as its text, so 18 is "18"."""
    got, want = (None if v is NOTHING else v for v in (got, want))
    return (got is None) == (want is None) and (got is None or str(got) == str(want))


def _same_reading(got, want) -> bool:
    """As _same_answer; and where the course reads nothing in a negative number, such as -20, reading it with its
    sign counts as right too: no answer key of the course's questions is negative, so it changes no score."""
    signed = isinstance(got, str) and got[:1] == "-" and got[1:].isdigit()
    return _same_answer(got, want) or (want is NOTHING and signed)


class _TextBox(FunctionBox):
    """A check whose cases are long texts: a message names a case by its beginning and its end."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(_short(a) for a in case)})"


def _short(value) -> str:
    if isinstance(value, str):
        text = value if len(value) <= 70 else value[:30] + " … " + value[-36:]
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_short(v) for v in value) + "]"
    return repr(value)


_R_VISIBLE, _R_HIDDEN = _r_cases()

# --- exercise 3: majority_vote ----------------------------------------------------------------------------------------

_V_HEAD = "def majority_vote(answers):\n"
_V_COUNT = ("    counts = {}\n"
            "    for answer in answers:\n"
            "        if answer is not None:\n"
            "            if answer in counts:\n"
            "                counts[answer] = counts[answer] + 1\n"
            "            else:\n"
            "                counts[answer] = 1\n")
_V_PICK = ("    best = None\n"
           "    for answer in answers:\n"
           "        if answer is not None and (best is None or counts[answer] > counts[best]):\n"
           "            best = answer\n"
           "    return best\n")


def _v(count: str = _V_COUNT, pick: str = _V_PICK) -> str:
    return _V_HEAD + count + pick


_V_CODE = _v()
_V_START = (_V_HEAD + "    # how often each answer comes, without None\n    counts = {}\n    ...\n"
            "    # the answer that comes most often; of answers that come as often, the first in the list; None if there is none\n"
            "    ...\n")
_V_WRONG = {
    "counts-none": _v(count=_V_COUNT.replace("        if answer is not None:\n            if answer in counts:\n"
                                             "                counts[answer] = counts[answer] + 1\n            else:\n"
                                             "                counts[answer] = 1\n",
                                             "        if answer in counts:\n            counts[answer] = counts[answer] + 1\n"
                                             "        else:\n            counts[answer] = 1\n"),
                      pick=_V_PICK.replace("answer is not None and (best is None or counts[answer] > counts[best])",
                                           "best is None or counts[answer] > counts[best]")),
    "tie-last": _v(pick=_V_PICK.replace("counts[answer] > counts[best]", "counts[answer] >= counts[best]")),
    "count": _v(pick=_V_PICK.replace("    return best\n", "    if best is None:\n        return None\n    return counts[best]\n")),
    "first": _v(pick="    for answer in answers:\n        if answer is not None:\n            return answer\n    return None\n"),
}


def _v_cases() -> tuple:
    """The groups the check votes on: a few made for it, and the answers the course's checker read in every group of 8
    samples of the course's run."""
    read, _ = course_functions()
    groups = [[_safe(read, v["text"]) for v in q["votes"]] for q in course()["questions"] if "votes" in q]
    made = [["18", "26", "18", None], ["180"] * 7 + ["540"], ["2", "2", "3", "3"], [None, None, "5"], ["5", None, None],
            [], [None], ["7", "9", "9"]]
    return [(made[0],)], [(g,) for g in made[1:] + groups]


_V_VISIBLE, _V_HIDDEN = _v_cases()

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
_DIFF = "With `{call}`, line {line} of your prompt is {got}, and it should be {want}"
_DIFF_NL = "Met `{call}` is regel {line} van je prompt {got}, en dat moet {want} zijn"


def _course_scores() -> dict:
    read, vote = course_functions()
    return scores(course(), read, vote)


SCORES = _course_scores()          # the course's run, read and voted with the course's own solutions
PHRASE_ONLY = phrase_only(course())   # its chains of thought that are right when only the phrase counts
COUNTER = _counter(course())           # the first question with the direct answer right and the chain of thought wrong
FIRST_VOTE = _example(course())        # the vote of question 1, for the worked example
# the chains of thought of the first questions of the demo that the course's checker reads as nothing
UNREAD = [i + 1 for i, q in enumerate(course()["questions"][:QUESTIONS["cpu"]])
          if _safe(course_functions()[0], q["steps"]["text"]) is None]
# zero-shot answers that, without any example, still write "The answer is"
ZERO_STYLE = {"phrase": sum("The answer is" in q["zero"]["text"] for q in course()["questions"])}
DEMO = course().get("demo")            # the demo measured on the course's CPU: minutes, memory, threads


def _download_line() -> dict:
    """The line under the header: what the lesson downloads, and what its demo took on the course's CPU."""
    m = {lang: number(DEMO["minutes"], 1, lang) if DEMO else "?" for lang in ("en", "nl")}
    gb = {lang: number(DEMO["memory_gb"], 1, lang) if DEMO else "?" for lang in ("en", "nl")}
    return {
        "en": f"Works on every computer. Qwen3-0.6B-Base (1.2 GB, Apache-2.0) is downloaded once from Hugging Face when you click its button in the lesson, the same files as in The same steps on an open model; on the course's CPU its demo then took {m['en']} minutes and {gb['en']} GB of memory. The exercises and the course's run work without it.",
        "nl": f"Werkt op elke computer. Qwen3-0.6B-Base (1,2 GB, Apache-2.0) wordt één keer gedownload van Hugging Face als je in de les op de knop ervoor klikt, dezelfde bestanden als in Dezelfde stappen op een open model; op de CPU van de course duurde de demo daarna {m['nl']} minuten, met {gb['nl']} GB geheugen. De oefeningen en de run van de course werken ook zonder.",
    }


DOWNLOAD_LINE = _download_line()

EXERCISES = {
    "few-shot-prompt": _PromptBox(
        name="few-shot-prompt",
        function="few_shot_prompt",
        start=_F_START,
        solution=_F_CODE,
        visible=_F_VISIBLE,
        hidden=_F_HIDDEN,
        show=lambda v: v,
        first=[_DOTS],
        mistakes=[
            Mistake("not-text", matches=_not_text,
                    en="With `{call}`, your function gives no text back. Join the parts into one text and give that back: `\"\".join(parts)`.",
                    nl="Met `{call}` geeft je functie geen tekst terug. Voeg de delen samen tot één tekst en geef die terug: `\"\".join(parts)`."),
            Mistake("answer-first", wrong=_F_WRONG["answer-first"],
                    en="With `{call}`, your examples give `The answer is` before their steps. The model copies that: it would write its answer before any step, and the steps could no longer help it. Put the steps first, then the answer.",
                    nl="Met `{call}` geven je voorbeelden `The answer is` vóór hun stappen. Het model doet dat na: het zou zijn antwoord schrijven vóór elke stap, en dan kunnen de stappen het niet meer helpen. Zet eerst de stappen, dan het antwoord."),
            Mistake("no-answer-end", wrong=_F_WRONG["no-answer-end"],
                    en="With `{call}`, your prompt ends with the question. Then the model may go on writing the question, or a new one. End it with a new line and `Answer:`, as every example has, so that what comes next is the answer.",
                    nl="Met `{call}` eindigt je prompt met de vraag. Dan kan het model de vraag verder schrijven, of een nieuwe. Eindig met een nieuwe regel en `Answer:`, zoals elk voorbeeld heeft, zodat wat erna komt het antwoord is."),
            Mistake("no-blank", wrong=_F_WRONG["no-blank"],
                    en="With `{call}`, there is no blank line between your examples: each ends with one new line, `\\n`, where it should end with two, `\\n\\n`. The blank line is where an answer ends, and the demo stops the model where it starts a new question after one.",
                    nl="Met `{call}` staat er geen lege regel tussen je voorbeelden: elk eindigt met één nieuwe regel, `\\n`, waar het er twee moeten zijn, `\\n\\n`. De lege regel is waar een antwoord ophoudt, en de demo stopt het model waar het daarna een nieuwe vraag begint."),
            Mistake("no-phrase", wrong=_F_WRONG["no-phrase"],
                    en="With `{call}`, your examples do not end with `The answer is`, the answer and a full stop. Your checker in exercise 2 looks for that phrase, and examples without it teach the model not to write it.",
                    nl="Met `{call}` eindigen je voorbeelden niet met `The answer is`, het antwoord en een punt. Je checker in oefening 2 zoekt naar die woorden, en voorbeelden zonder leren het model ze niet te schrijven."),
            Mistake("steps-always", wrong=_F_WRONG["steps-always"],
                    en="With `{call}`, your examples have their steps also when `steps` is `False`. The direct prompt shows only `Answer: The answer is` and the answer: that is the difference the demo measures.",
                    nl="Met `{call}` hebben je voorbeelden hun stappen ook als `steps` `False` is. De directe prompt laat alleen `Answer: The answer is` en het antwoord zien: dat verschil meet de demo."),
            Mistake("space-end", wrong=_F_WRONG["space-end"],
                    en="With `{call}`, your prompt ends with `\"Answer: \"`, with a space. A word token starts with its space, such as `\" The\"`, so after a space at the end the model must write a word without its space, which it hardly ever saw. End with `Answer:`.",
                    nl="Met `{call}` eindigt je prompt met `\"Answer: \"`, met een spatie. Een token van een woord begint met zijn spatie, zoals `\" The\"`, dus na een spatie aan het eind moet het model een woord zonder zijn spatie schrijven, en dat zag het bijna nooit. Eindig met `Answer:`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: every example in the layout of the task, and then the new question up to `Answer:`.",
                      "nl": "Vervang de `...`: elk voorbeeld in de vorm van de opdracht, en dan de nieuwe vraag tot en met `Answer:`."},
            "wrong": {"en": _DIFF + ". Compare it sign by sign with the layout in the task.",
                      "nl": _DIFF_NL + ". Vergelijk hem teken voor teken met de vorm in de opdracht."},
            "almost": {"en": "Right with the lesson's examples and steps. But " + _DIFF[0].lower() + _DIFF[1:] + ". Does it work without steps, with one example, and with steps of several lines?",
                       "nl": "Goed met de voorbeelden van de les en hun stappen. Maar " + _DIFF_NL[0].lower() + _DIFF_NL[1:] + ". Werkt hij ook zonder stappen, met één voorbeeld, en met stappen van meer regels?"},
            "correct": {"en": "Your prompts have the layout of the examples, with and without steps, and end where the answer begins. The demo below gives Qwen3 your prompts.",
                        "nl": "Je prompts hebben de vorm van de voorbeelden, met en zonder stappen, en eindigen waar het antwoord begint. De demo hieronder geeft Qwen3 je prompts."},
        },
    ),
    "read-answer": _TextBox(
        name="read-answer",
        function="read_answer",
        start=_R_START,
        solution=_R_CODE,
        visible=_R_VISIBLE,
        hidden=_R_HIDDEN,
        compare=_same_reading,
        collect=_kept,
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an error: it asks for a place in a list that is not there. That happens with a text without the phrase, or without any number. Then give `None`.",
                    nl="Met `{call}` stopt je functie met een fout: hij vraagt om een plek in een lijst die er niet is. Dat gebeurt bij een tekst zonder `The answer is`, of zonder getal. Geef dan `None`."),
            Mistake("last-phrase", wrong=_R_WRONG["last-phrase"],
                    en=_CALL + ": you read the number after the *last* `The answer is`. A base model often goes on with a question of its own and answers that too; its answer to this question is the first.",
                    nl=_CALL_NL + ": je leest het getal na de *laatste* `The answer is`. Een base model gaat vaak door met een eigen vraag en beantwoordt die ook; zijn antwoord op deze vraag is het eerste."),
            Mistake("no-fallback", wrong=_R_WRONG["no-fallback"],
                    en=_CALL + ": without `The answer is` you give `None`. Some answers end with a sentence such as *So she pays 17 dollars.*: then take the last word that is a number.",
                    nl=_CALL_NL + ": zonder `The answer is` geef je `None`. Sommige antwoorden eindigen met een zin als *So she pays 17 dollars.*: neem dan het laatste woord dat een getal is."),
            Mistake("first-number", wrong=_R_WRONG["first-number"],
                    en=_CALL + ": without the phrase you take the first number. The first numbers of an answer are mostly those of the question; its result comes at the end. Walk the words from the end, with `reversed`.",
                    nl=_CALL_NL + ": zonder `The answer is` neem je het eerste getal. De eerste getallen van een antwoord zijn meestal die van de vraag; de uitkomst staat aan het eind. Loop de woorden van achteren af, met `reversed`."),
            Mistake("comma", wrong=_R_WRONG["comma"],
                    en=_CALL + ": a number with a comma, such as `1,250`, is not all digits until the comma is out. Take the commas out of every word: `\"\".join(word.split(\",\"))`.",
                    nl=_CALL_NL + ": een getal met een komma, zoals `1,250`, bestaat pas uit cijfers als de komma eruit is. Haal de komma's uit elk woord: `\"\".join(word.split(\",\"))`."),
            Mistake("minus", wrong=_R_WRONG["minus"],
                    en=_CALL + ": you take the minus sign off, so `-20` becomes `20`, another number. Leave it on: then the word is not all digits and reads as `None`, which counts as wrong, as `-20` should for a question whose answer is 20.",
                    nl=_CALL_NL + ": je haalt het minteken weg, zodat `-20` `20` wordt, een ander getal. Laat het staan: dan bestaat het woord niet alleen uit cijfers en lees je `None`, en dat telt als fout, zoals `-20` dat hoort te doen bij een vraag met antwoord 20."),
            Mistake("dollar", wrong=_R_WRONG["dollar"],
                    en=_CALL + ": a word that starts with a dollar sign is not all digits yet. Take that sign off its ends too, with the full stops and the other signs.",
                    nl=_CALL_NL + ": een woord dat met een dollarteken begint, bestaat nog niet alleen uit cijfers. Haal dat teken ook van de uiteinden, met de punten en de andere tekens."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: `number` cleans one word and says whether it is a number, and `read_answer` uses it with and without the phrase.",
                      "nl": "Vervang de `...`: `number` maakt één woord schoon en zegt of het een getal is, en `read_answer` gebruikt het met en zonder `The answer is`."},
            "wrong": {"en": _CALL + ". Is it the first number after the first `The answer is`, or else the last word that is a number?",
                      "nl": _CALL_NL + ". Is het het eerste getal na de eerste `The answer is`, of anders het laatste woord dat een getal is?"},
            "almost": {"en": "Right for the answer with steps to question 1 of the course's run. But " + _CALL[0].lower() + _CALL[1:] + ". Answers without the phrase, a question of its own, commas, dollars, and a text without a number?",
                       "nl": "Goed voor het antwoord met stappen op vraag 1 van de run van de course. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Antwoorden zonder `The answer is`, een eigen vraag, komma's, dollars, en een tekst zonder getal?"},
            "correct": {"en": f"With your `read_answer`, the course's {SCORES['n']} chains of thought have {SCORES['steps']} right answers; counting only the number after `The answer is`, {PHRASE_ONLY}. A score is only as good as its checker: say its rules when you report one.",
                        "nl": f"Met je `read_answer` hebben de {SCORES['n']} chains of thought van de course {SCORES['steps']} goede antwoorden; tel je alleen het getal na `The answer is`, dan {PHRASE_ONLY}. Een score is maar zo goed als zijn checker: noem de regels als je er een meldt."},
        },
    ),
    "majority-vote": _TextBox(
        name="majority-vote",
        function="majority_vote",
        start=_V_START,
        solution=_V_CODE,
        visible=_V_VISIBLE,
        hidden=_V_HIDDEN,
        compare=_same_answer,
        collect=_kept,
        first=[_DOTS],
        mistakes=[
            Mistake("counts-none", wrong=_V_WRONG["counts-none"],
                    en=_CALL + ": you count `None` as an answer. `None` means the checker found no number: it is no vote. Leave it out of the counts.",
                    nl=_CALL_NL + ": je telt `None` als antwoord. `None` betekent dat de checker geen getal vond: dat is geen stem. Laat het buiten de telling."),
            Mistake("tie-last", wrong=_V_WRONG["tie-last"],
                    en=_CALL + ": of two answers that come as often, you give the last. Give the first in the list: keep an answer only when it comes *more* often than the one you have.",
                    nl=_CALL_NL + ": van twee antwoorden die even vaak komen, geef je het laatste. Geef het eerste in de lijst: neem een antwoord alleen als het *vaker* komt dan het antwoord dat je hebt."),
            Mistake("count", wrong=_V_WRONG["count"],
                    en=_CALL + ": you give how often the winner comes. Give the answer itself.",
                    nl=_CALL_NL + ": je geeft hoe vaak de winnaar komt. Geef het antwoord zelf."),
            Mistake("first", wrong=_V_WRONG["first"],
                    en=_CALL + ": you give the first answer, without counting. The vote is the answer that comes most often.",
                    nl=_CALL_NL + ": je geeft het eerste antwoord, zonder te tellen. De stemming is het antwoord dat het vaakst komt."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: count every answer that is not `None` in `counts`, then walk the list and keep the first that comes most often.",
                      "nl": "Vervang de `...`: tel elk antwoord dat niet `None` is in `counts`, en loop dan de lijst af en houd het eerste dat het vaakst komt."},
            "wrong": {"en": _CALL + ". Is it the answer that comes most often, without `None`, and the first of a tie?",
                      "nl": _CALL_NL + ". Is het het antwoord dat het vaakst komt, zonder `None`, en het eerste bij gelijkspel?"},
            "almost": {"en": "Right for the group of the task. But " + _CALL[0].lower() + _CALL[1:] + ". A tie, a group with `None`, an empty group, and the course's groups of 8?",
                       "nl": "Goed voor de groep van de opdracht. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gelijkspel, een groep met `None`, een lege groep, en de groepen van 8 van de course?"},
            "correct": {"en": f"Your vote gives the course's own on all its {SCORES['n']} groups of {SAMPLES}: {SCORES['vote'][-1]} right. The block below votes with your functions.",
                        "nl": f"Je stemming geeft op alle {SCORES['n']} groepen van {SAMPLES} hetzelfde als die van de course: {SCORES['vote'][-1]} goed. Het blok hieronder stemt met je functies."},
        },
    ),
}

# The lesson is done when these are correct. "quick-..." are the quick check; "faithful" the open question at the end.
REQUIRED = ["few-shot-prompt", "read-answer", "majority-vote", "faithful"]

# What the tests feed each check, and what it must say: (exercise, what the learner gave, state, known mistake).
CASES = [
    ("few-shot-prompt", _F_START, "empty", None),
    ("few-shot-prompt", _F_CODE, "correct", None),
    *[("few-shot-prompt", code, "wrong", key) for key, code in _F_WRONG.items()],
    ("few-shot-prompt", _F_CODE.replace("    return \"\".join(parts)\n", "    return parts\n"), "wrong", "not-text"),
    ("few-shot-prompt", _F_HEAD + "    text = \"\"\n    for example in examples:\n        text += f\"Question: {example['question']}\\nAnswer: \"\n"
     "        if steps:\n            text += example[\"steps\"] + \"\\n\"\n        text += f\"The answer is {example['answer']}.\\n\\n\"\n"
     "    return text + f\"Question: {question}\\nAnswer:\"\n", "correct", None),
    ("few-shot-prompt", _f(part=_F_PART.replace("\"Question: \"", "\"Q: \"")), "wrong", None),
    ("read-answer", _R_START, "empty", None),
    ("read-answer", _R_CODE, "correct", None),
    ("read-answer", _R_CODE.replace("    if word.isdigit():\n", "    if word.isdigit() or (word[:1] == \"-\" and word[1:].isdigit()):\n"),
     "correct", None),
    *[("read-answer", code, "error" if key == "index" else "wrong", key) for key, code in _R_WRONG.items()],
    ("read-answer", _R_CODE.replace("                return number(word)\n        return None\n",
                                    "                return number(word)\n        return \"\"\n"), "almost", None),
    ("majority-vote", _V_START, "empty", None),
    ("majority-vote", _V_CODE, "correct", None),
    *[("majority-vote", code, "wrong", key) for key, code in _V_WRONG.items()],
    ("majority-vote", _V_HEAD + "    best, most = None, 0\n    for answer in answers:\n        n = 0\n        for other in answers:\n"
     "            if other == answer:\n                n = n + 1\n        if answer is not None and n > most:\n"
     "            best, most = answer, n\n    return best\n", "correct", None),
]
