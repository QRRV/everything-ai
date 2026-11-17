# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-banking-granite.npz: the embeddings of Granite Embedding 107M multilingual (the embedding model)
of the 120 test questions of data/banking77-small.json, for build-cost-latency-cache. The questions go through the
course's own server (everything_ai.serve, the endpoint /v1/embeddings) with the standard library's urllib, the way a
learner's code sends them: all 120 in one request for the embeddings, and then each question again in a request of its
own, timed, as a semantic cache embeds each question that comes in.

The file holds `test` (one row of 384 numbers per question, in order) as float16 to keep it small, every row of length
1 before it was rounded, and `seconds` (the seconds of each question's own request, in order). `about` is a JSON text
with the model, its revision, the SHA-256 of banking77-small.json, the seconds of the request with all 120, the commit
of this code, the machine, threads and PyTorch. Run it again only on purpose, from the course folder, with Granite
Embedding downloaded (everything_ai.embed.download()):

    uv run python data/tools/record_course_banking_granite.py
"""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from everything_ai import embed, hub, paths, serve  # noqa: E402

BANKING = paths.ROOT / "data" / "banking77-small.json"
OUT = paths.ROOT / "data" / "course-banking-granite.npz"
THREADS = 2   # the server's threads, as for the chat model's runs


def _code() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def _cpu() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _embed(url: str, texts: list) -> tuple[np.ndarray, float]:
    body = {"model": serve.EMBEDDER, "input": texts}
    req = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())
    seconds = time.perf_counter() - t
    return np.array([d["embedding"] for d in sorted(raw["data"], key=lambda d: d["index"])], dtype=np.float32), seconds


def main() -> None:
    torch.set_num_threads(THREADS)
    questions = [text for _, text in json.loads(BANKING.read_text(encoding="utf-8"))["test"]]
    running = serve.start(serve.Model())
    try:
        _embed(running.url, ["warm up"])   # load the model before any request is timed
        test, together = _embed(running.url, questions)
        seconds = np.array([_embed(running.url, [q])[1] for q in questions], dtype=np.float32)
    finally:
        running.stop()
    lengths = np.linalg.norm(test, axis=1)
    assert test.shape == (len(questions), serve.DIMS) and np.allclose(lengths, 1, atol=1e-4), (lengths.min(), lengths.max())
    about = {"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"],
             "banking": hashlib.sha256(BANKING.read_bytes()).hexdigest(), "together_seconds": round(together, 3),
             "code": _code(), "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__}
    np.savez_compressed(OUT, about=np.array(json.dumps(about)), test=test.astype(np.float16), seconds=seconds)
    print("written", OUT, test.shape, "one question: median", round(float(np.median(seconds)), 4), "s")


if __name__ == "__main__":
    main()
