# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-embeddings-granite.json, the course's runs of Granite Embedding 107M multilingual (the
embedding model) for the lessons of the domain bouwen. Each run goes through the course's own server
(everything_ai.serve, the endpoint /v1/embeddings) with the standard library's urllib, the way a learner's code
calls it. Each lesson's checks.py has EMBED, {key: ask}, and an ask is one of

    {"texts": [...]}                 the embeddings of the texts in one request: the answer as the server sent it with
        the first four numbers of each embedding ("start"), the tokens of each text ("tokens"), the length of each
        embedding ("lengths"), the dot product of every two ("dots", a table), and the seconds of the request
    {"speed": [...], "repeat": n}    the seconds of one request with all the texts, and of one request per text for
        the same texts, each the median of n tries

An ask whose content changed is asked again; `--again lesson:key ...` asks those again, for a run on a quiet computer.
Run from the course folder, with Granite Embedding downloaded (everything_ai.embed.download()), for all lessons or the
ones named:

    uv run python data/tools/record_course_embeddings.py [lesson-id ...] [--again lesson:key ...]
"""
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import embed, hub, paths, serve  # noqa: E402

OUT = paths.ROOT / "data" / "course-embeddings-granite.json"
LESSONS = ["build-embeddings-api"]
THREADS = 2   # the server's threads, as for the chat model's runs


def _checks(lesson: str):
    spec = importlib.util.spec_from_file_location(lesson.replace("-", "_"), paths.LESSONS / lesson / "checks.py")
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


def _post(url: str, texts: list) -> tuple[dict, float]:
    body = {"model": serve.EMBEDDER, "input": texts}
    req = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())
    return raw, time.perf_counter() - start


def _texts(url: str, texts: list) -> dict:
    raw, seconds = _post(url, texts)
    vectors = [d["embedding"] for d in sorted(raw["data"], key=lambda d: d["index"])]
    t = torch.tensor(vectors, dtype=torch.float64)
    return {"start": [v[:4] for v in vectors], "tokens": [len(embed.encode(x)) for x in texts],
            "prompt_tokens": raw["usage"]["prompt_tokens"], "lengths": [round(x, 6) for x in t.norm(dim=1).tolist()],
            "dots": [[round(x, 6) for x in row] for row in (t @ t.T).tolist()], "seconds": round(seconds, 3)}


def _speed(url: str, texts: list, repeat: int) -> dict:
    together = statistics.median(_post(url, texts)[1] for _ in range(repeat))
    apart = statistics.median(sum(_post(url, [x])[1] for x in texts) for _ in range(repeat))
    return {"together": round(together, 3), "apart": round(apart, 3)}


def main(lessons: list, again: set) -> None:
    torch.set_num_threads(THREADS)
    running = serve.start(serve.Model())
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"parts": {}}
    out.update({"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"], "dtype": "float32",
                "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__})
    t = time.perf_counter()
    _post(running.url, ["warm up"])   # load the model before any request is timed
    out["load_seconds"] = round(time.perf_counter() - t, 2)
    out["parts"] = dict(out.pop("parts"))   # keep the parts last in the file

    def save(*said):
        out["code"] = _code()
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(*said, flush=True)
    for lesson in lessons:
        C = _checks(lesson)
        part = out["parts"].setdefault(lesson, {})
        for key in list(part):
            if key not in C.EMBED:
                del part[key]
        for key, ask in C.EMBED.items():
            entry = part.get(key)
            if entry is not None and entry["ask"] == ask and f"{lesson}:{key}" not in again:
                continue
            if "texts" in ask:
                part[key] = {"ask": ask, **_texts(running.url, ask["texts"])}
                save(lesson, key, part[key]["prompt_tokens"], "tokens", part[key]["seconds"], "s")
            else:
                part[key] = {"ask": ask, **_speed(running.url, ask["speed"], ask["repeat"])}
                save(lesson, key, part[key])
    running.stop()


if __name__ == "__main__":
    args = sys.argv[1:]
    again = set(args[args.index("--again") + 1:]) if "--again" in args else set()
    named = args[:args.index("--again")] if "--again" in args else args
    main(named or LESSONS, again)
