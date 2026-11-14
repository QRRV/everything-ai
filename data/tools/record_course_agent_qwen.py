# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-agent-qwen.json, the course's runs of agent-idea: Qwen3-0.6B (the chat model), in float32 with
the course's code and KV cache, on the lesson's pantry task. First the task alone, without tools or instructions,
with WITHOUT seeds; then RUNS runs of the course's run_agent (the solution of exercise 1) with the lesson's `ask`
(checks.qwen_ask: the system message, the two tools, non-thinking mode, the seed of each question), each with its
messages and the tokens and seconds of every message of the model. Run from the course folder, with the chat model
downloaded (everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_agent_qwen.py
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
from everything_ai.check import run_code  # noqa: E402

OUT = paths.ROOT / "data" / "course-agent-qwen.json"
WITHOUT, RUNS = 6, 6


def _lesson():
    spec = importlib.util.spec_from_file_location("agent", paths.LESSONS / "agent-idea" / "checks.py")
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
    run_agent = run_code(C.EXERCISES["run-agent"].solution).names["run_agent"]
    t = time.perf_counter()
    model = chat.Chat()
    out = {"model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"], "dtype": "float32",
           "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(), "torch": torch.__version__,
           "code": _code(), "load_seconds": round(time.perf_counter() - t, 1), "sampling": chat.SAMPLING,
           "thinking": C.THINKING, "max_tokens": C.MAX_TOKENS, "task": C.QWEN_TASK, "system": C.SYSTEM,
           "without_tools": [], "runs": []}
    for seed in range(WITHOUT):
        a = model.answer([{"role": "user", "content": C.QWEN_TASK}], max_tokens=C.MAX_TOKENS, seed=seed, thinking=C.THINKING)
        out["without_tools"].append({"seed": seed, "text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"],
                                     "finish": a["finish"]})
        print("without tools", seed, a["tokens"], repr(a["text"][:120]), flush=True)
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for seed in range(RUNS):
        run = C.run_agent_with_qwen(run_agent, model, seed)
        out["runs"].append(run)
        print("run", seed, [c["name"] for m in run["messages"] if m["role"] == "assistant" for c in m["tool_calls"]],
              repr(run["messages"][-1].get("content", "")[:120]), flush=True)
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
