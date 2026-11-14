# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-chains-qwen.json, the course's run of start-prompt-chains: Qwen3-0.6B (the chat model), in float32
with the course's code and KV cache, non-thinking, sampled as Qwen3 advises (everything_ai.chat.SAMPLING), on the
lesson's four experiments (checks.py), SEEDS times each with the seeds 0 to SEEDS - 1:

- "chain": the shopping sum in one prompt, and in a chain of two prompts (the amounts for 12 people, then the cost);
- "verify": five primes between 80 and 100 as a first draft, then a fresh check of every number it names, asked plainly
  ("checks") and worked out step by step ("steps");
- "source": the closing time on Saturday without the source, with it, and a question the source does not answer;
- "refine": the first answers of start-prompt-basics (data/course-prompts-qwen.json, the ways "plain" and "number"),
  each followed by the request to check it. Run that one after record_course_prompts_qwen.py.

Every answer with its text, tokens and seconds. The file is written after every seed, and a run that was stopped goes
on where it was. Run from the course folder, with the chat model downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_chains_qwen.py
    uv run python data/tools/record_course_chains_qwen.py refine
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3  # noqa: E402

OUT = paths.ROOT / "data" / "course-chains-qwen.json"
PROMPTS = paths.ROOT / "data" / "course-prompts-qwen.json"


def _lesson(lesson_id: str):
    spec = importlib.util.spec_from_file_location(lesson_id.replace("-", "_"), paths.LESSONS / lesson_id / "checks.py")
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


def _answer(model, messages, seed, max_tokens):
    a = model.answer(messages, max_tokens=max_tokens, seed=seed)
    return {"text": a["message"]["content"], "tokens": a["tokens"], "max_tokens": max_tokens, "seconds": a["seconds"], "finish": a["finish"]}


def main(parts) -> None:
    C = _lesson("start-prompt-chains")
    model = chat.Chat()
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"], "dtype": "float32",
        "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(), "torch": torch.__version__,
        "code": _code(), "sampling": chat.SAMPLING, "max_tokens": C.MAX_TOKENS, "one_max_tokens": C.ONE_TOKENS, "check_max_tokens": C.CHECK_TOKENS, "seeds": C.SEEDS}

    def save():
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if "chain" in parts:
        done = out.setdefault("chain", [])
        for seed in range(len(done), C.SEEDS):
            one = _answer(model, C.one_prompt(), seed, C.ONE_TOKENS)
            first = _answer(model, C.chain_first(), seed, C.MAX_TOKENS)
            second = _answer(model, C.chain_second(C.amounts_of(first["text"])), seed, C.MAX_TOKENS)
            done.append({"seed": seed, "one": one, "first": first, "second": second})
            save()
            print("chain", seed, C.read(one["text"]), C.scaled_right(first["text"]), C.read(second["text"]), flush=True)
    if "one" in parts:   # the one prompt again, for runs recorded with another limit of tokens
        out["one_max_tokens"] = C.ONE_TOKENS
        for row in out["chain"]:
            if row["one"].get("max_tokens") != C.ONE_TOKENS:
                row["one"] = _answer(model, C.one_prompt(), row["seed"], C.ONE_TOKENS)
                save()
                print("one", row["seed"], C.read(row["one"]["text"]), row["one"]["finish"], flush=True)
    if "verify" in parts:
        out["check_max_tokens"] = C.CHECK_TOKENS
        done = out.setdefault("verify", [])
        for seed in range(len(done), C.SEEDS):
            draft = _answer(model, C.draft_primes(), seed, C.MAX_TOKENS)
            checks = {str(n): _answer(model, C.verify(n), seed, C.MAX_TOKENS) for n in C.listed(draft["text"])}
            steps = {str(n): _answer(model, C.verify_steps(n), seed, C.CHECK_TOKENS) for n in C.listed(draft["text"])}
            done.append({"seed": seed, "draft": draft, "checks": checks, "steps": steps})
            save()
            print("verify", seed, C.listed(draft["text"]), {n: C.said_yes(a["text"]) for n, a in checks.items()},
                  {n: C.said_yes(a["text"]) for n, a in steps.items()}, flush=True)
    if "source" in parts:
        done = out.setdefault("source", [])
        for seed in range(len(done), C.SEEDS):
            row = {"seed": seed, "without": _answer(model, C.ungrounded(C.ASK_SATURDAY), seed, C.MAX_TOKENS),
                   "with": _answer(model, C.grounded(C.ASK_SATURDAY), seed, C.MAX_TOKENS),
                   "not_said": _answer(model, C.grounded(C.ASK_MAIL), seed, C.MAX_TOKENS)}
            done.append(row)
            save()
            print("source", seed, C.says_two(row["without"]["text"]), C.says_two(row["with"]["text"]), repr(row["not_said"]["text"][:80]), flush=True)
    if "refine" in parts:
        P = _lesson("start-prompt-basics")
        prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))["answers"]
        done = out.setdefault("refine", {})
        for q, (question, answer) in P.QUESTIONS.items():
            entry = done.setdefault(q, {"question": question, "answer": answer})
            for way in C.REFINE_WAYS:
                rows = entry.setdefault(way, [])
                for seed in range(len(rows), C.SEEDS):
                    first = prompts[q][way][seed]["text"]
                    rows.append({"seed": seed, "first": first,
                                 "checked": _answer(model, C.check_own(P.messages(way, question)[-1]["content"], first), seed, C.MAX_TOKENS)})
                    save()
                    print("refine", q, way, seed, P.read(first), C.read(rows[-1]["checked"]["text"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["chain", "verify", "source"])
