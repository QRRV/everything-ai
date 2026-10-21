# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-reasoning-qwen.json, the course's run of llm-reasoning-idea: Qwen3-0.6B-Base, in float32 with
the course's code and KV cache, on the 30 test questions of data/gsm8k-sample.jsonl, with the lesson's own prompt
(the course's few_shot_prompt, four worked examples), stop rule and generation (checks.generate). For every question:
the direct answer (examples with only the answer), the chain of thought (examples with their steps), the zero-shot
answer ("Let's think step by step.", no examples), all greedy, and SAMPLES answers with steps at TEMPERATURE for the
vote, drawn together with the question's number as seed; each answer with its text, tokens and seconds. The greedy
answers come first, for all questions, then the samples. The file is written after every question, and a run that
was stopped goes on where it was. With --demo, it runs the lesson's demo instead, as a learner with the course's
solutions would on a CPU (checks.run, kind "cpu"), and adds how long it took end to end, its memory, its threads, and
how many of its answers are the same text as the course's run: run that in a process of its own, after the run. Run
from the course folder, with Qwen3 downloaded (everything_ai.qwen3.download()):

    uv run python data/tools/record_course_reasoning_qwen.py
    uv run python data/tools/record_course_reasoning_qwen.py --demo
"""
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import hub, paths, qwen3  # noqa: E402
from everything_ai.check import run_code  # noqa: E402

OUT = paths.ROOT / "data" / "course-reasoning-qwen.json"


def _lesson():
    spec = importlib.util.spec_from_file_location("reasoning", paths.LESSONS / "llm-reasoning-idea" / "checks.py")
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


def demo() -> None:
    """The demo end to end, as the lesson runs it on a CPU, with the course's solutions."""
    C = _lesson()
    run = json.loads(OUT.read_text(encoding="utf-8"))
    names = {**run_code(C._F_CODE).names, **run_code(C._R_CODE).names, **run_code(C._V_CODE).names}
    start = time.perf_counter()
    r = C.run(names["few_shot_prompt"], names["read_answer"], names["majority_vote"],
              lambda d: print(d, flush=True) if d["phase"] == "question" else None, kind="cpu")
    if isinstance(r, str):
        sys.exit(f"the demo stopped: {r}")
    same = sum(q[kind]["text"] == c[kind]["text"] for q, c in zip(r["questions"], run["questions"]) for kind in ("direct", "steps"))
    run["demo"] = {"questions": len(r["questions"]), "minutes": round((time.perf_counter() - start) / 60, 1),
                   "memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
                   "threads": torch.get_num_threads(), "same_text": same, "answers": 2 * len(r["questions"]),
                   "direct": sum(q["direct"]["right"] for q in r["questions"]), "steps": sum(q["steps"]["right"] for q in r["questions"])}
    OUT.write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(run["demo"], flush=True)


def main() -> None:
    C = _lesson()
    few_shot_prompt = run_code(C._F_CODE).names["few_shot_prompt"]
    tests, shots = C.problems("test"), C.examples()
    run = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "model": qwen3.REPO, "revision": hub.PINNED[(qwen3.REPO, "model.safetensors")].get("revision", ""),
        "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
        "torch": torch.__version__, "code": _code(), "shots": C.SHOTS, "new": C.NEW, "stops": list(C.STOPS),
        "samples": C.SAMPLES, "temperature": C.TEMPERATURE,
        "questions": [{"k": k, "answer": p["answer"]} for k, p in enumerate(tests)]}
    begin = time.perf_counter()
    t = time.perf_counter()
    model, enc = C.load()
    run["load_seconds"] = round(time.perf_counter() - t, 1)
    for k, p in enumerate(tests):
        q = run["questions"][k]
        if "zero" in q:
            continue
        q["direct"] = C.generate(model, enc, few_shot_prompt(shots, p["question"], False), C.NEW["direct"])[0]
        q["steps"] = C.generate(model, enc, few_shot_prompt(shots, p["question"], True), C.NEW["steps"])[0]
        q["zero"] = C.generate(model, enc, C.zero_shot_prompt(p["question"]), C.NEW["steps"])[0]
        OUT.write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(k, "direct", q["direct"]["tokens"], q["direct"]["seconds"], "steps", q["steps"]["tokens"], q["steps"]["seconds"],
              "zero", q["zero"]["tokens"], q["zero"]["seconds"], round(time.perf_counter() - begin), flush=True)
    for k, p in enumerate(tests):
        q = run["questions"][k]
        if "votes" in q:
            continue
        q["votes"] = C.generate(model, enc, few_shot_prompt(shots, p["question"], True), C.NEW["steps"], n=C.SAMPLES,
                                temperature=C.TEMPERATURE, seed=k)
        OUT.write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(k, "votes", [v["tokens"] for v in q["votes"]], max(v["seconds"] for v in q["votes"]),
              round(time.perf_counter() - begin), flush=True)


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
