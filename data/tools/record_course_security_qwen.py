# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-security-qwen.json, the course's runs of Qwen3-0.6B (the chat model) for the module *security*
of the domain verantwoord: in float32 with the course's code and KV cache. Each lesson's checks.py has ASKS, {key:
{"messages": [...], "max_tokens": n, "stop": [...], "thinking": bool, "seeds": n, "temperature": t, "tools":
[...]}}, and SEEDS; every ask is answered with the seeds 0 to its "seeds" - 1 (SEEDS when it has none), sampled as
Qwen3 advises for its mode (everything_ai.chat.SAMPLING without thinking, chat.THINKING with it), or greedy when its
"temperature" is 0; with "tools", the model reads those tools (everything_ai.chat.tool) and may call them, and each
answer keeps its calls as everything_ai.chat.parse reads them. Each ask is kept with its messages, so the file says
what was asked. With "stop", the model stops as soon as its answer holds one of those texts. An ask whose messages,
max_tokens, stop, thinking, temperature or tools changed is asked again from the start. The file is written after
every answer, and a run that was stopped goes on where it was. Run from the course folder, with the chat model
downloaded (everything_ai.qwen3.download_chat()), for all lessons or the ones named:

    uv run python data/tools/record_course_security_qwen.py [lesson-id ...]
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3  # noqa: E402

OUT = paths.ROOT / "data" / "course-security-qwen.json"
LESSONS = ["rai-prompt-injection", "rai-indirect-injection"]
FIELDS = ("messages", "max_tokens", "stop", "thinking")


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
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main(lessons: list) -> None:
    t = time.perf_counter()
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"], "dtype": "float32",
        "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(), "torch": torch.__version__,
        "sampling": chat.SAMPLING, "thinking": chat.THINKING, "parts": {}}
    out["load_seconds"] = round(time.perf_counter() - t, 1)
    for lesson in lessons:
        C = _checks(lesson)
        part = out["parts"].setdefault(lesson, {})
        for key in list(part):
            if key not in C.ASKS:
                del part[key]
        for key, ask in C.ASKS.items():
            want = {f: ask.get(f, [] if f == "stop" else False) for f in FIELDS}
            for f in ("temperature", "tools"):   # kept only when the ask sets it, so that the other asks stay as they were
                if f in ask:
                    want[f] = ask[f]
            entry = part.get(key)
            if entry is None or any(entry.get(f) != want.get(f) for f in (*FIELDS, "temperature", "tools")):
                entry = part[key] = {**want, "answers": []}
            sampling = dict(chat.THINKING if want["thinking"] else chat.SAMPLING)
            if "temperature" in want:
                sampling["temperature"] = want["temperature"]
            for seed in range(len(entry["answers"]), ask.get("seeds", C.SEEDS)):
                a = model.answer(want["messages"], want.get("tools"), max_tokens=want["max_tokens"], seed=seed, thinking=want["thinking"],
                                 stop=want["stop"] or None, **sampling)
                entry["answers"].append({"seed": seed, "text": a["text"], "tokens": a["tokens"],
                                         "seconds": a["seconds"], "finish": a["finish"]})
                if want.get("tools"):   # the tool calls as everything_ai.chat.parse reads them from the text
                    entry["answers"][-1]["calls"] = a["message"]["tool_calls"]
                out["code"] = _code()
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                print(lesson, key, seed, a["tokens"], a["finish"], repr(a["text"][-60:]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or LESSONS)
