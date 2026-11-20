# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-detection-qwen.json, the course's run for the lesson rai-ai-text-detection of the domain
verantwoord: Qwen3-0.6B, the chat model, in float32 with the course's code and KV cache. For every question of the
lesson's QUESTIONS, the model answers the question alone (without thinking, seed 0, sampled as Qwen3 advises,
everything_ai.chat.SAMPLING, at most MAX_TOKENS tokens), and the file keeps that answer next to the answer a person
wrote in data/oasst-short.json. Then the same model scores, with the lesson's own score, both answers alone and both
answers after the question in the chat template, and every text of FAMOUS alone: the log probability of each token.
A question whose text changed is asked again; questions and texts that are gone are dropped. The file is written
after every question, and a run that was stopped goes on where it was. Run from the course folder, with the chat
model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_detection_qwen.py
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

LESSON = "rai-ai-text-detection"
OUT = paths.ROOT / "data" / "course-detection-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("rai_ai_text_detection", paths.LESSONS / LESSON / "checks.py")
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
    people = {q: a for q, a in (lambda d: d["held"] + d["train"])(
        json.loads((paths.ROOT / "data" / "oasst-short.json").read_text(encoding="utf-8")))}
    missing = [q for q in C.QUESTIONS if q not in people]
    if missing:
        raise SystemExit(f"not in data/oasst-short.json: {missing}")
    t = time.perf_counter()
    model = qwen3.load(repo=qwen3.CHAT)
    enc = qwen3.tokenizer()
    bot = chat.Chat(model=model, enc=enc)
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "sampling": chat.SAMPLING, "seed": 0, "max_tokens": C.MAX_TOKENS,
                "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(), "torch": torch.__version__,
                "load_seconds": round(time.perf_counter() - t, 1)})
    items = out.setdefault("items", [])
    items[:] = [x for x in items if x["question"] in C.QUESTIONS and x.get("max_tokens") == C.MAX_TOKENS]
    done = {x["question"] for x in items}
    for q in C.QUESTIONS:
        if q in done:
            continue
        messages = [{"role": "user", "content": q}]
        a = bot.answer(messages, max_tokens=C.MAX_TOKENS, seed=0, thinking=False, **chat.SAMPLING)
        context = chat.prompt(messages, None, False)
        human, ai = people[q], a["text"].strip()
        items.append({"question": q, "max_tokens": C.MAX_TOKENS, "human": human,
                      "ai": ai, "ai_tokens": a["tokens"], "ai_finish": a["finish"], "ai_seconds": a["seconds"],
                      "scores": {"human": C.score(model, enc, human), "ai": C.score(model, enc, ai),
                                 "human_after_question": C.score(model, enc, human, context),
                                 "ai_after_question": C.score(model, enc, ai, context)}})
        items.sort(key=lambda x: C.QUESTIONS.index(x["question"]))
        out["code"] = _code()
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(len(items), a["tokens"], a["finish"], repr(ai[:70]), flush=True)
    out["famous"] = {key: {"text": text, "score": C.score(model, enc, text)} for key, text in C.FAMOUS.items()}
    out["code"] = _code()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("famous", len(out["famous"]), "seconds", round(time.perf_counter() - t, 1), flush=True)


if __name__ == "__main__":
    main()
