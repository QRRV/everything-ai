# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-moderation-qwen.json, the course's run for the lesson rai-toxicity-moderation of the domain
verantwoord: Qwen3-0.6B, the chat model, in float32 with the course's code, as a moderator. For every question of the
lesson's QUESTIONS and every text of its TEXTS, the model gets the question about the text (checks.messages, without
thinking) and writes one token; the file keeps, from the 20 most likely first tokens, the probability of the answer
A, of the answer B, and the lesson's score (checks.score_from: the share of B in A and B together), with the five
most likely tokens. The run is greedy and the same on every run on the same machine, so the whole file is written
again each time. Run from the course folder, with the chat model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_moderation_qwen.py
"""
import importlib.util
import json
import math
import os
import platform
import subprocess
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3  # noqa: E402

LESSON = "rai-toxicity-moderation"
OUT = paths.ROOT / "data" / "course-moderation-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("rai_toxicity_moderation", paths.LESSONS / LESSON / "checks.py")
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
    C = _checks()
    t = time.perf_counter()
    bot = chat.Chat()
    out = {"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
           "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
           "torch": torch.__version__, "load_seconds": round(time.perf_counter() - t, 1),
           "sampling": {"temperature": 0, "max_tokens": 1, "logprobs": C.TOP, "thinking": False}, "questions": {}}
    for key, question in C.QUESTIONS.items():
        part = out["questions"][key] = {"question": question, "texts": {}}
        for group, texts in C.TEXTS.items():
            for text in texts:
                a = bot.answer(C.messages(text, key), max_tokens=1, temperature=0, logprobs=C.TOP)
                top = a["logprobs"][0]["top_logprobs"]
                p = C.chances(top)
                part["texts"][text] = {"group": group, "score": C.score_from(top), "a": round(p["a"], 6),
                                       "b": round(p["b"], 6),
                                       "top": [[e["token"], round(e["logprob"], 4)] for e in top[:5]]}
                print(key, group, part["texts"][text]["score"], repr(text), flush=True)
    out["seconds"] = round(time.perf_counter() - t, 1)
    out["code"] = _code()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("seconds", out["seconds"], flush=True)


if __name__ == "__main__":
    main()
