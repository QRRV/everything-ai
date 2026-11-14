# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-prompts-qwen.json, the course's run of start-prompt-basics: Qwen3-0.6B (the chat model), in
float32 with the course's code and KV cache, non-thinking, sampled as Qwen3 advises (everything_ai.chat.SAMPLING), on
the lesson's three questions, each asked in the lesson's seven ways (checks.messages), SEEDS times with the seeds 0 to
SEEDS - 1; every answer with its text, tokens, seconds and how it ended. The file is written after every answer, and a
run that was stopped goes on where it was. Run from the course folder, with the chat model downloaded
(everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_prompts_qwen.py
"""
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3  # noqa: E402

OUT = paths.ROOT / "data" / "course-prompts-qwen.json"


def _lesson():
    spec = importlib.util.spec_from_file_location("prompts", paths.LESSONS / "start-prompt-basics" / "checks.py")
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


def main() -> None:
    C = _lesson()
    t = time.perf_counter()
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"], "dtype": "float32",
        "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(), "torch": torch.__version__,
        "code": _code(), "sampling": chat.SAMPLING, "max_tokens": C.MAX_TOKENS, "seeds": C.SEEDS, "answers": {}}
    out["load_seconds"] = round(time.perf_counter() - t, 1)
    for q, (question, _) in C.QUESTIONS.items():
        for way in C.WAYS:
            done = out["answers"].setdefault(q, {}).setdefault(way, [])
            for seed in range(len(done), C.SEEDS):
                a = model.answer(C.messages(way, question), max_tokens=C.MAX_TOKENS, seed=seed)
                done.append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"],
                             "finish": a["finish"]})
                OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                print(q, way, seed, C.read(a["message"]["content"]), a["tokens"], a["seconds"], flush=True)


if __name__ == "__main__":
    main()
