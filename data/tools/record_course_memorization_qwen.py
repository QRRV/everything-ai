# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-memorization-qwen.json, the course's run for the lesson rai-memorization of the domain
verantwoord: Qwen3-0.6B-Base and Qwen3-0.6B (the chat model), in float32 with the course's code and KV cache. For
every text in the lesson's TEXTS and every k in its KS, each model gets the first k words of the text alone, without
a chat template, and writes NEW tokens greedily, with the lesson's own go_on; the file keeps each start with what
the model wrote, so it says what was asked. A run whose start or NEW changed is made again; runs of texts that are
gone are dropped. The file is written after every run, and a run that was stopped goes on where it was. Run from the
course folder, with both models downloaded (everything_ai.qwen3.download() and download_chat()):

    uv run python data/tools/record_course_memorization_qwen.py
"""
import gc
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import hub, paths, qwen3  # noqa: E402

LESSON = "rai-memorization"
OUT = paths.ROOT / "data" / "course-memorization-qwen.json"
REPOS = {"base": qwen3.REPO, "chat": qwen3.CHAT}


def _checks():
    spec = importlib.util.spec_from_file_location("rai_memorization", paths.LESSONS / LESSON / "checks.py")
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
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"runs": {}}
    out.update({"lesson": LESSON, "dtype": "float32", "new_tokens": C.NEW, "ks": C.KS,
                "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__})
    out.setdefault("models", {})
    enc = qwen3.tokenizer()
    for name, repo in REPOS.items():
        want = {f"{key}|{k}": C.split_text(text, k, C.N)[0] for key, text in C.TEXTS.items() for k in C.KS}
        runs = out["runs"].setdefault(name, {})
        for key in list(runs):
            if key not in want or runs[key]["start"] != want[key] or runs[key].get("new") != C.NEW:
                del runs[key]
        if all(key in runs for key in want):
            continue
        t = time.perf_counter()
        model = qwen3.load(repo=repo)
        out["models"][name] = {"model": repo, "revision": hub.PINNED[(repo, "model.safetensors")]["revision"],
                               "load_seconds": round(time.perf_counter() - t, 1)}
        for key, start in want.items():
            if key in runs:
                continue
            a = C.go_on(model, enc, start, C.NEW)
            runs[key] = {"start": start, "new": C.NEW, **a}
            out["runs"][name] = dict(sorted(runs.items(), key=lambda kv: list(want).index(kv[0])))
            runs = out["runs"][name]
            out["code"] = _code()
            OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(name, key, a["tokens"], a["finish"], repr(a["text"][:60]), flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
