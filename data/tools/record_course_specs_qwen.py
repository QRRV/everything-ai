# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-specs-qwen.json, the course's run for the lesson rai-model-specs of the domain verantwoord:
Qwen3-0.6B, the chat model, in float32 with the course's code, answers every prompt of the lesson's PROMPTS (four per
rule of the lesson's spec) once for each seed of its SEEDS, without thinking, sampled as Qwen3's model card advises
for non-thinking mode (everything_ai.chat.SAMPLING), with at most MAX_TOKENS tokens. The file keeps every answer with
its seed, the number of tokens and whether the model ended the answer itself ("stop") or reached the limit
("length"). Sampling with a seed gives the same answers on the same machine; the whole file is written again each
time. Run from the course folder, with the chat model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_specs_qwen.py
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

LESSON = "rai-model-specs"
OUT = paths.ROOT / "data" / "course-specs-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("rai_model_specs", paths.LESSONS / LESSON / "checks.py")
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
           "sampling": {**chat.SAMPLING, "max_tokens": C.MAX_TOKENS, "seeds": C.SEEDS, "thinking": False},
           "prompts": {}}
    for rule, prompts in C.PROMPTS.items():
        for prompt in prompts:
            part = out["prompts"][prompt] = {"rule": rule, "answers": []}
            for seed in C.SEEDS:
                a = bot.answer(C.messages(prompt), max_tokens=C.MAX_TOKENS, seed=seed)
                part["answers"].append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"],
                                        "finish": a["finish"]})
                print(rule, seed, a["tokens"], a["finish"], repr(prompt), repr(a["message"]["content"][:80]), flush=True)
    out["seconds"] = round(time.perf_counter() - t, 1)
    out["code"] = _code()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("seconds", out["seconds"], flush=True)


if __name__ == "__main__":
    main()
