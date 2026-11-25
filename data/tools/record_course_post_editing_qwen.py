# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-post-editing-qwen.json, the course's run of nlp-post-editing: Qwen3-0.6B (the chat model), in
float32 with the course's code, non-thinking and greedy (temperature 0), asked to translate into English each Dutch
sentence of Tatoeba that nlp-classic-mt measures on and that has more than one English translation by people (the
3,707 of nlp-bleu), with the prompt PROMPT; every translation with its text, tokens, seconds and how it ended, under
translations[id]. The prompt keeps its SHA-256: when it changes, every sentence is asked again. The file is written
every 50 sentences, and a run that was stopped goes on where it was. Run from the course folder, with the chat model
downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_post_editing_qwen.py
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

LESSON = "nlp-post-editing"
OUT = paths.ROOT / "data" / "course-post-editing-qwen.json"
PROMPT = "Translate this Dutch sentence into English. Reply with only the English translation.\n\n{dutch}"
MAX_TOKENS = 64


def _classic_mt():
    spec = importlib.util.spec_from_file_location("nlp_classic_mt", paths.LESSONS / "nlp-classic-mt" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sentences() -> list:
    """(id, Dutch text) of the sentences of the run, in the order of nlp-classic-mt."""
    return [(sid, text) for sid, text, es in _classic_mt()._covered() if len(es) > 1]


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
    t = time.perf_counter()
    model = chat.Chat()
    sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if out.get("prompt_sha256") != sha:
        out = {}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "temperature": 0, "thinking": False, "max_tokens": MAX_TOKENS,
                "prompt": PROMPT, "prompt_sha256": sha, "load_seconds": round(time.perf_counter() - t, 1)})
    done = out.setdefault("translations", {})
    todo = [(sid, text) for sid, text in sentences() if str(sid) not in done]
    for k, (sid, text) in enumerate(todo, start=1):
        a = model.answer([{"role": "user", "content": PROMPT.format(dutch=text)}], max_tokens=MAX_TOKENS, temperature=0)
        done[str(sid)] = {"text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"],
                          "finish": a["finish"]}
        if k % 50 == 0 or k == len(todo):
            out["code"] = _code()
            OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(k, len(todo), sid, repr(text), repr(a["message"]["content"]), flush=True)


if __name__ == "__main__":
    main()
