# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-work-qwen.json, the course's runs of Qwen3-0.6B (the chat model) for the module *AI at work*
(werk) of the domain bouwen: in float32 with the course's code and KV cache, non-thinking, sampled as Qwen3 advises
(everything_ai.chat.SAMPLING). Each lesson's checks.py has ASKS, {key: {"messages": [...], "max_tokens": n}}, and
SEEDS; every ask is answered SEEDS times, with the seeds 0 to SEEDS - 1, and kept with its messages, so the file says
what was asked. An ask whose messages or max_tokens changed is asked again from the start. The file is written after
every answer, and a run that was stopped goes on where it was. Run from the course folder, with the chat model
downloaded (everything_ai.qwen3.download_chat()), for all lessons or the ones named:

    uv run python data/tools/record_course_work_qwen.py [lesson-id ...]
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

OUT = paths.ROOT / "data" / "course-work-qwen.json"
LESSONS = ["build-ai-at-work", "build-research-and-learn", "build-make-without-code", "build-coding-workflow",
           "build-ai-explains-code", "build-check-ai-code"]


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
        "sampling": chat.SAMPLING, "parts": {}}
    out["load_seconds"] = round(time.perf_counter() - t, 1)
    for lesson in lessons:
        C = _checks(lesson)
        part = out["parts"].setdefault(lesson, {})
        for key in list(part):
            if key not in C.ASKS:
                del part[key]
        for key, ask in C.ASKS.items():
            entry = part.get(key)
            if entry is None or entry["messages"] != ask["messages"] or entry["max_tokens"] != ask["max_tokens"]:
                entry = part[key] = {"messages": ask["messages"], "max_tokens": ask["max_tokens"], "answers": []}
            for seed in range(len(entry["answers"]), C.SEEDS):
                a = model.answer(ask["messages"], max_tokens=ask["max_tokens"], seed=seed)
                entry["answers"].append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"],
                                         "seconds": a["seconds"], "finish": a["finish"]})
                out["code"] = _code()
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                print(lesson, key, seed, a["tokens"], a["finish"], repr(a["message"]["content"][:80]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or LESSONS)
