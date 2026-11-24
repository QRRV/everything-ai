# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-qa-kinds-qwen.json, the course's run of nlp-qa-kinds: Qwen3-0.6B (the chat model), in float32
with the course's code and KV cache, non-thinking, sampled as Qwen3 advises (everything_ai.chat.SAMPLING), on the
lesson's twelve questions in English and in Dutch, each asked without a text (closed-book) and with each of the six
texts of the lesson's collection (checks.prompt), with the seeds 0 to SEEDS - 1; every answer with its text, tokens,
prompt tokens, seconds and how it ended, under answers[lang][question]["closed"] and answers[lang][question]["texts"]
[text], one per seed. Each prompt keeps its SHA-256: an answer whose prompt changed is asked again. The file is
written after every answer, and a run that was stopped goes on where it was. Run from the course folder, with the
chat model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_qa_kinds_qwen.py
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

LESSON = "nlp-qa-kinds"
OUT = paths.ROOT / "data" / "course-qa-kinds-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("nlp_qa_kinds", paths.LESSONS / LESSON / "checks.py")
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


def _slot(slot: dict, sha: str) -> dict:
    """The answers of one prompt, emptied when the prompt changed."""
    if slot.get("prompt_sha256") != sha:
        slot.clear()
        slot.update({"prompt_sha256": sha, "seeds": []})
    return slot


def main() -> None:
    C = _checks()
    t = time.perf_counter()
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"answers": {}}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "sampling": chat.SAMPLING, "max_tokens": C.MAX_TOKENS, "seeds": C.SEEDS,
                "load_seconds": round(time.perf_counter() - t, 1)})
    for lang in C.LANGS:
        questions = out["answers"].setdefault(lang, [])
        del questions[len(C.QUESTIONS[lang]):]
        while len(questions) < len(C.QUESTIONS[lang]):
            questions.append({"closed": {}, "texts": []})
        for i, q in enumerate(questions):
            del q["texts"][len(C.TEXTS[lang]):]
            while len(q["texts"]) < len(C.TEXTS[lang]):
                q["texts"].append({})
        for seed in range(C.SEEDS):
            for i, q in enumerate(questions):
                for k in [None, *range(len(C.TEXTS[lang]))]:
                    text = C.prompt(lang, i, k)
                    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    slot = _slot(q["closed"] if k is None else q["texts"][k], sha)
                    if len(slot["seeds"]) > seed:
                        continue
                    a = model.answer([{"role": "user", "content": text}], max_tokens=C.MAX_TOKENS, seed=seed)
                    slot["seeds"].append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"],
                                          "prompt_tokens": a["prompt_tokens"], "seconds": a["seconds"],
                                          "finish": a["finish"]})
                    out["code"] = _code()
                    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                    print(lang, seed, i, k, a["tokens"], a["seconds"], repr(a["message"]["content"][:90]), flush=True)


if __name__ == "__main__":
    main()
