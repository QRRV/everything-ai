# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-apis-qwen.json, data/course-dev-prompts-qwen.json, data/course-rag-qwen.json,
data/course-agents-qwen.json and data/course-production-qwen.json, the course's runs of Qwen3-0.6B (the chat model) for
the modules *Calling models* (apis), *Prompts for developers* (prompts), *RAG* (rag), *Agents and tools* (agents) and
*Evaluating and shipping* (productie) of the domain bouwen, one file for each module. Each run goes through the course's own server (everything_ai.serve) with the
standard library's urllib, the way a learner's code calls it: in float32 with the course's code and KV cache,
non-thinking unless asked, sampled as Qwen3 advises for each mode (everything_ai.chat.SAMPLING, and THINKING for an
ask with "think"). Each lesson's checks.py has
RECORD, {key: ask}, and an ask is one of

    {"messages": [...], "max_tokens": n, "seeds": k}   answered with the seeds 0 to k - 1, as a stream, which gives the
        seconds until the first piece of the answer ("first") and until its end ("seconds"); with "think": true in
        thinking mode, and with "think" or "raw" without a stream, which keeps the whole JSON the server sent ("raw");
        "temperature", "top_p", "stop", "response_format", "logprobs" and "top_logprobs" go into the request as they
        are, and an ask with "logprobs" goes without a stream and keeps the logprobs of every token ("logprobs");
        messages that end with one of the assistant are a prefill; with "retry": n, checks.py's follow_up(text) reads each answer
        (follow_up(text, feedback) when the ask has a "feedback" text to fill in), and while it gives a message and fewer than n tries were made, the answer and that message (as the user's) are
        added and the model is asked again with the same seed: such an answer is {"seed": s, "tries": [...]}; with
        "then": n, checks.py's then(key, answers) gives the next ask from the answers so far, such as a request built
        from what a plan gave, or None, and it is asked with the same seed, at most n asks in all: such an answer is
        {"seed": s, "chain": [...]}
    {"messages": [...], "tools": [...], "steps": n, "max_tokens": n, "seeds": k}   a question with tools, without a
        stream: the answer's tool calls are run by checks.py's tool_results(message, ask), which gives a message with the
        role tool for each, and the answer and those results go back to the model, at most n requests in all;
        "tool_choice" goes with the first request only, the later ones are "auto"; "temperature", "top_p" and
        "parallel_tool_calls" go with each. Such an answer is {"seed": s, "steps": [...]}, each step with its "tool_calls" as the server gave them
        and the "results" of the tools. When checks.py has new_run(ask), it is called before each such run, so that tools
        that keep state, such as a browser on a shop, start from the beginning
    {"turns": [...], "max_tokens": n, "seeds": k}      a chat: in each turn the user's next message, after the
        messages so far and the model's answers to them, every turn with the same seed; "messages" before the first
        turn, such as a system prompt, go first
    {"speed": "float32" | "bfloat16", "messages": [...], "max_tokens": n}   the chat model in that dtype on one thread,
        without the server: how many tokens it writes per second
    {"count": [text, ...]}                              how many tokens each text has, with Qwen3's tokenizer
    {"image": name, "question": q, "max_tokens": n, "seeds": k}   a question about data/photos/<name>.png, sent as a
        data URL with the server's hosted tool detect_objects, without a stream; it keeps what the tool found
        ("hosted_tools"). The detector is loaded before the first such ask, so that its time is not in the seconds.
    {"labels": side, "system": prompt, "max_tokens": n}  every question of that side of data/banking77-small.json,
        after the system prompt, greedy (temperature 0): the text of each answer, its finish reason, and its seconds
        and tokens
    {"each": [text, ...], "system": prompt, "max_tokens": n}   each text as the user's message after the system
        prompt, the same way
    {"sequence": [{"messages": [...], "cache": bool}, ...], "max_tokens": n, "seeds": k}   requests sent one after
        the other without a stream, k times over, the k-th time with seed k - 1; "cache": false sends
        "cache_prompt": false, so the server computes the whole prompt again. For each request: its seconds, and how
        many of its prompt tokens came from the server's kept cache ("cached_tokens"). An answer is {"seed": s,
        "steps": [...]}

An ask whose content changed is asked again from the start, one with more seeds only for the new seeds. Timings depend
on how busy the computer is: `--again lesson:key ...` asks those again, for a run on a quiet computer. The file is
written after every answer. Run from the course folder, with the chat model downloaded
(everything_ai.qwen3.download_chat()), for all lessons or the ones named:

    uv run python data/tools/record_course_apis_qwen.py [lesson-id ...] [--again lesson:key ...]
"""
import base64
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3, serve  # noqa: E402

FILES = {   # the file of each module, and its lessons
    "course-apis-qwen.json": ["build-tokens-prices", "build-call-api", "build-project-call", "build-streaming",
                              "build-choose-model", "build-where-models-run", "build-multimodal-apis", "build-ai-services"],
    "course-dev-prompts-qwen.json": ["build-sampling-api", "build-system-prompt", "build-structured-output",
                                     "build-prompt-templates", "build-project-prompts", "build-model-specific",
                                     "build-defensive-prompts", "build-self-consistency", "build-context-engineering",
                                     "build-prompt-caching"],
    "course-rag-qwen.json": ["build-rag-idea", "build-rag-first", "build-grounded-answers", "build-project-rag",
                             "build-reranking", "build-query-rewriting", "build-rag-eval"],
    "course-agents-qwen.json": ["build-tool-calling", "build-tool-design", "build-workflow-or-agent", "build-workflow-patterns",
                                "build-agent-loop", "build-plan-execute", "build-agent-memory", "build-browser-use",
                                "build-mcp", "build-agent-extensions", "build-kinds-of-agents", "build-agent-history",
                                "build-agent-frameworks"],
    "course-production-qwen.json": ["build-error-analysis", "build-eval-set", "build-judges"],
}
LESSONS = [lesson for lessons in FILES.values() for lesson in lessons]
THREADS = 2   # the server's threads; a speed ask runs on one


def _checks(lesson: str):
    spec = importlib.util.spec_from_file_location(lesson.replace("-", "_"), paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cpu() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _code() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _post(url: str, body: dict):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def _whole(url: str, body: dict) -> dict:
    start = time.perf_counter()
    with _post(url, body) as r:
        raw = json.loads(r.read())
    seconds = time.perf_counter() - start
    choice, usage = raw["choices"][0], raw["usage"]
    out = {"text": choice["message"]["content"] or "", "finish": choice["finish_reason"],
           "prompt_tokens": usage["prompt_tokens"], "tokens": usage["completion_tokens"], "seconds": round(seconds, 2)}
    if "reasoning_content" in choice["message"]:
        out["reasoning"] = choice["message"]["reasoning_content"]
    if "prompt_tokens_details" in usage:
        out["cached_tokens"] = usage["prompt_tokens_details"]["cached_tokens"]
    if choice.get("logprobs"):
        out["logprobs"] = choice["logprobs"]["content"]
    return out, raw


def _stream(url: str, body: dict) -> dict:
    """The answer as a stream: its text, the seconds until the first piece and until the end, the second at which each
    piece came in ("times"), and the first and last lines of the stream as the server sent them ("lines")."""
    start, first, pieces, times, lines, usage, finish = time.perf_counter(), None, [], [], [], None, None
    with _post(url, {**body, "stream": True, "stream_options": {"include_usage": True}}) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            lines.append(line)
            if line == "data: [DONE]":
                break
            chunk = json.loads(line[len("data: "):])
            usage = chunk.get("usage") or usage
            for c in chunk["choices"]:
                piece = c["delta"].get("content")
                if piece:
                    now = time.perf_counter() - start
                    first = first if first is not None else now
                    pieces.append(piece)
                    times.append(round(now, 2))
                finish = c.get("finish_reason") or finish
    return {"text": "".join(pieces), "finish": finish, "prompt_tokens": usage["prompt_tokens"], "tokens": usage["completion_tokens"],
            "first": round(first, 2) if first is not None else None, "seconds": round(time.perf_counter() - start, 2),
            "pieces": len(pieces), "times": times, "lines": lines[:6] + lines[-4:] if len(lines) > 10 else lines}


def _image(url: str, ask: dict, seed: int) -> dict:
    data = (paths.ROOT / "data" / "photos" / f"{ask['image']}.png").read_bytes()
    content = [{"type": "text", "text": ask["question"]},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}]
    body = {"model": serve.MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": ask["max_tokens"],
            "seed": seed, "tools": [{"type": serve.DETECT}]}
    a, raw = _whole(url, body)
    a["hosted_tools"] = raw["hosted_tools"]
    return a


def _each(url: str, ask: dict, texts: list) -> list:
    out = []
    for text in texts:
        body = {"model": serve.MODEL, "max_tokens": ask["max_tokens"], "temperature": 0,
                "messages": [{"role": "system", "content": ask["system"]}, {"role": "user", "content": text}]}
        a, _ = _whole(url, body)
        out.append({"text": a["text"], "finish": a["finish"], "seconds": a["seconds"], "prompt_tokens": a["prompt_tokens"],
                    "tokens": a["tokens"]})
    return out


def _labels(url: str, ask: dict) -> list:
    rows = json.loads((paths.ROOT / "data" / "banking77-small.json").read_text(encoding="utf-8"))[ask["labels"]]
    return [{"label": label, **a} for (label, _), a in zip(rows, _each(url, ask, [text for _, text in rows]))]


def _answer(url: str, ask: dict, messages: list, seed: int) -> dict:
    if "image" in ask:
        return _image(url, ask, seed)
    body = {"model": serve.MODEL, "messages": messages, "max_tokens": ask["max_tokens"], "seed": seed,
            **{k: ask[k] for k in ("temperature", "top_p", "stop", "response_format", "logprobs", "top_logprobs") if k in ask}}
    if ask.get("think"):
        body["chat_template_kwargs"] = {"enable_thinking": True}
    if ask.get("think") or ask.get("raw") or ask.get("logprobs"):
        a, raw = _whole(url, body)
        if ask.get("raw"):
            a["raw"] = raw
        return a
    return _stream(url, body)


def _steps(url: str, ask: dict, seed: int, C) -> list:
    """A question with tools: each answer, with its tool calls and their results, until an answer calls no tool or
    ask["steps"] requests were made."""
    messages, steps = list(ask["messages"]), []
    if hasattr(C, "new_run"):
        C.new_run(ask)
    for n in range(ask["steps"]):
        body = {"model": serve.MODEL, "messages": messages, "max_tokens": ask["max_tokens"], "seed": seed, "tools": ask["tools"],
                **{k: ask[k] for k in ("temperature", "top_p", "parallel_tool_calls") if k in ask}}
        if n == 0 and "tool_choice" in ask:
            body["tool_choice"] = ask["tool_choice"]
        a, raw = _whole(url, body)
        message = raw["choices"][0]["message"]
        a["tool_calls"] = message.get("tool_calls") or []
        results = C.tool_results(message, ask) if a["tool_calls"] else []
        a["results"] = [r["content"] for r in results]
        steps.append(a)
        if not a["tool_calls"]:
            break
        messages = messages + [message] + results
    return steps


def _speed(ask: dict) -> dict:
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[ask["speed"]]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = chat.Chat(model=qwen3.load(repo=qwen3.CHAT, dtype=dtype))
        a = model.answer(ask["messages"], max_tokens=ask["max_tokens"], seed=0)
    finally:
        torch.set_num_threads(threads)
    return {"text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"], "finish": a["finish"],
            "per_second": round(a["tokens"] / a["seconds"], 2)}


def _opened(name: str, load_seconds: float) -> dict:
    path = paths.ROOT / "data" / name
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"], "dtype": "float32",
        "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__,
        "parts": {}}
    out["sampling"] = {"normal": chat.SAMPLING, "thinking": chat.THINKING}
    out["load_seconds"] = load_seconds
    return out


def main(lessons: list, again: set) -> None:
    torch.set_num_threads(THREADS)
    t = time.perf_counter()
    running = serve.start(serve.Model(chat.Chat()))
    load_seconds = round(time.perf_counter() - t, 1)
    opened = {}
    warm = [False]
    for lesson in lessons:
        name = next(n for n, ls in FILES.items() if lesson in ls)
        out = opened.setdefault(name, _opened(name, load_seconds))

        def save(*said, out=out, name=name):
            out["code"] = _code()
            (paths.ROOT / "data" / name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(*said, flush=True)
        C = _checks(lesson)
        part = out["parts"].setdefault(lesson, {})
        for key in list(part):
            if key not in C.RECORD:
                del part[key]
        for key, ask in C.RECORD.items():
            same = {k: v for k, v in ask.items() if k != "seeds"}
            entry = part.get(key)
            if entry is None or entry["ask"] != same or f"{lesson}:{key}" in again:
                entry = part[key] = {"ask": same, "answers": []}
            if "count" in ask:
                if not entry["answers"]:
                    enc = qwen3.tokenizer()
                    entry["answers"] = [{"text": t, "tokens": len(qwen3.encode(enc, t))} for t in ask["count"]]
                    save(lesson, key, [a["tokens"] for a in entry["answers"]])
                continue
            if "speed" in ask:
                if not entry["answers"]:
                    entry["answers"].append(_speed(ask))
                    save(lesson, key, entry["answers"][-1]["per_second"], "tokens/s")
                continue
            if "labels" in ask or "each" in ask:
                if not entry["answers"]:
                    entry["answers"] = _labels(running.url, ask) if "labels" in ask else _each(running.url, ask, ask["each"])
                    save(lesson, key, len(entry["answers"]), "answers")
                continue
            if "image" in ask and not warm[0]:   # load the detector before the first image ask is timed
                from PIL import Image
                running.backend.detect(Image.new("RGB", (64, 64)))
                warm[0] = True
            for seed in range(len(entry["answers"]), ask["seeds"]):
                if "tools" in ask:
                    steps = _steps(running.url, ask, seed, C)
                    entry["answers"].append({"seed": seed, "steps": steps})
                    save(lesson, key, seed, [(len(a["tool_calls"]), repr(a["text"][:50])) for a in steps])
                elif "sequence" in ask:
                    steps = []
                    for step in ask["sequence"]:
                        body = {"model": serve.MODEL, "messages": step["messages"], "max_tokens": ask["max_tokens"],
                                "seed": seed, "cache_prompt": step.get("cache", True)}
                        steps.append(_whole(running.url, body)[0])
                    entry["answers"].append({"seed": seed, "steps": steps})
                    save(lesson, key, seed, [(a["cached_tokens"], a["seconds"]) for a in steps])
                elif "turns" in ask:
                    messages, turns = list(ask.get("messages", [])), []
                    for said in ask["turns"]:
                        messages.append({"role": "user", "content": said})
                        a = _answer(running.url, ask, messages, seed)
                        turns.append(a)
                        messages.append({"role": "assistant", "content": a["text"]})
                    entry["answers"].append({"seed": seed, "turns": turns})
                    save(lesson, key, seed, [a["prompt_tokens"] for a in turns])
                elif "retry" in ask:
                    messages, tries = list(ask["messages"]), [_answer(running.url, ask, ask["messages"], seed)]

                    def told(text, ask=ask, C=C):   # the lesson's follow-up, with the feedback text of the ask if it has one
                        return C.follow_up(text, ask["feedback"]) if "feedback" in ask else C.follow_up(text)
                    said = told(tries[-1]["text"])
                    while said is not None and len(tries) < ask["retry"]:
                        messages += [{"role": "assistant", "content": tries[-1]["text"]}, {"role": "user", "content": said}]
                        tries.append(_answer(running.url, ask, messages, seed))
                        said = told(tries[-1]["text"])
                    entry["answers"].append({"seed": seed, "tries": tries})
                    save(lesson, key, seed, len(tries), [repr(a["text"][:60]) for a in tries])
                elif "then" in ask:
                    chain = [_answer(running.url, ask, ask["messages"], seed)]
                    while len(chain) < ask["then"]:
                        nxt = C.then(key, chain)
                        if nxt is None:
                            break
                        chain.append(_answer(running.url, nxt, nxt["messages"], seed))
                    entry["answers"].append({"seed": seed, "chain": chain})
                    save(lesson, key, seed, [(a["tokens"], repr(a["text"][:50])) for a in chain])
                else:
                    a = _answer(running.url, ask, ask.get("messages"), seed)
                    entry["answers"].append({"seed": seed, **a})
                    save(lesson, key, seed, a["tokens"], a["finish"], a.get("first"), repr(a["text"][:70]))
    running.stop()


if __name__ == "__main__":
    args = sys.argv[1:]
    again = set(args[args.index("--again") + 1:]) if "--again" in args else set()
    named = args[:args.index("--again")] if "--again" in args else args
    main(named or LESSONS, again)
