# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-extraction-qwen.json, the course's run of nlp-extraction-with-llms: Qwen3-0.6B (the chat model),
in float32 with the course's code and KV cache, non-thinking, sampled as Qwen3 advises (everything_ai.chat.SAMPLING),
on the lesson's twelve notices in English and in Dutch, each asked in the lesson's three ways (checks.WAYS: without
examples, with the two examples, and with the two examples and the whole newsletter), with the seeds 0 to SEEDS - 1;
every answer with its text, tokens, prompt tokens, seconds and how it ended, under answers[lang][way][notice], one per
seed. In the way long the twelve prompts of a seed start the same, so the KV cache of that start is kept and used
again (cached_tokens). Each notice keeps the SHA-256 of its prompt: an answer whose prompt changed is asked again.
The file is written after every answer, and a run that was stopped goes on where it was. Run from the course folder,
with the chat model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_extraction_qwen.py
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

LESSON = "nlp-extraction-with-llms"
OUT = paths.ROOT / "data" / "course-extraction-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("nlp_extraction_with_llms", paths.LESSONS / LESSON / "checks.py")
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
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"answers": {}}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "sampling": chat.SAMPLING, "max_tokens": C.MAX_TOKENS, "seeds": C.SEEDS,
                "load_seconds": round(time.perf_counter() - t, 1)})
    for lang in C.LANGS:
        for way in C.WAYS:
            notices = out["answers"].setdefault(lang, {}).setdefault(way, [])
            del notices[len(C.NOTICES[lang]):]
            for i in range(len(C.NOTICES[lang])):
                sha = hashlib.sha256(C.prompt(way, lang, i).encode("utf-8")).hexdigest()
                if i == len(notices):
                    notices.append({"prompt_sha256": sha, "seeds": []})
                if notices[i]["prompt_sha256"] != sha:
                    notices[i] = {"prompt_sha256": sha, "seeds": []}
            for seed in range(C.SEEDS):
                for i in range(len(C.NOTICES[lang])):
                    done = notices[i]["seeds"]
                    if len(done) > seed:
                        continue
                    a = model.answer([{"role": "user", "content": C.prompt(way, lang, i)}], max_tokens=C.MAX_TOKENS,
                                     seed=seed, cache=way == "long")
                    done.append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"],
                                 "prompt_tokens": a["prompt_tokens"], "cached_tokens": a.get("cached_tokens", 0),
                                 "seconds": a["seconds"], "finish": a["finish"]})
                    out["code"] = _code()
                    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
                    print(lang, way, seed, i, a["tokens"], a["seconds"], repr(a["message"]["content"][:80]), flush=True)


if __name__ == "__main__":
    main()
