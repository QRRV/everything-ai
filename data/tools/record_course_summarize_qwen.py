# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-summarize-qwen.json, the course's run of nlp-summarize: Qwen3-0.6B (the chat model), in float32
with the course's code, non-thinking and greedy (temperature 0), asked to summarize each of the lesson's six abstracts
(checks.ABSTRACTS) in one sentence (checks.summarize_messages), with at most checks.MAX_TOKENS tokens. Every answer with
its text, tokens, seconds and how it ended, under abstracts[key]; each keeps the SHA-256 of its prompt, and an abstract
whose prompt changed is asked again. Run from the course folder, with the chat model downloaded
(everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_summarize_qwen.py
"""
import hashlib
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

LESSON = "nlp-summarize"
OUT = paths.ROOT / "data" / "course-summarize-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("nlp_summarize", paths.LESSONS / LESSON / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt_sha(messages: list) -> str:
    """The SHA-256 of a prompt: the text of its one message."""
    return hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()


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
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "temperature": 0, "thinking": False, "max_tokens": C.MAX_TOKENS,
                "instruction": C.INSTRUCTION, "load_seconds": round(time.perf_counter() - t, 1)})
    done = out.setdefault("abstracts", {})
    for key in C.ABSTRACTS:
        messages = C.summarize_messages(key)
        sha = prompt_sha(messages)
        if done.get(key, {}).get("sha256") == sha:
            continue
        a = model.answer(messages, max_tokens=C.MAX_TOKENS, temperature=0)
        done[key] = {"sha256": sha, "text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"],
                     "finish": a["finish"]}
        out["code"] = _code()
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(key, a["finish"], repr(a["message"]["content"]), flush=True)


if __name__ == "__main__":
    main()
