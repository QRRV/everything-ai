# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-rewrites-granite.npz: the embeddings of Granite Embedding 107M multilingual (the embedding
model) of every text the lesson build-query-rewriting searches with, for the 24 questions about the course that a
point answers: each follow-up question, the follow-up after its earlier question, Qwen3's rewrite of it, the short
question, Qwen3's passage to search with (HyDE) and the versions of the short question it wrote, as the course's
versions() reads them (lessons/build-query-rewriting/checks.py, to_embed()). Qwen3's texts come from
data/course-rag-qwen.json, so run data/tools/record_course_apis_qwen.py build-query-rewriting first. The texts go
through the course's own server (everything_ai.serve, the endpoint /v1/embeddings) with the standard library's urllib,
64 at a time.

The file holds `texts`, a JSON list of the texts, and `vectors` (one row of 384 per text, in that order) as float16;
every row had length 1 before it was rounded. `about` is a JSON text with the model, its revision, the commit of this
code and the machine. Run it again only on purpose, from the course folder, with Granite Embedding downloaded
(everything_ai.embed.download()):

    uv run python data/tools/record_course_rewrites_granite.py
"""
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from everything_ai import embed, hub, paths, serve  # noqa: E402

OUT = paths.ROOT / "data" / "course-rewrites-granite.npz"
BATCH = 64
THREADS = 2


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


def _embed(url: str, texts: list) -> np.ndarray:
    out = []
    for first in range(0, len(texts), BATCH):
        body = {"model": serve.EMBEDDER, "input": texts[first:first + BATCH]}
        req = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = json.loads(r.read())
        out += [d["embedding"] for d in sorted(raw["data"], key=lambda d: d["index"])]
        print(first + len(body["input"]), "of", len(texts), flush=True)
    return np.array(out, dtype=np.float32)


def main() -> None:
    spec = importlib.util.spec_from_file_location("build_query_rewriting", paths.LESSONS / "build-query-rewriting" / "checks.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    texts = checks.to_embed()
    torch.set_num_threads(THREADS)
    running = serve.start(serve.Model())
    try:
        vectors = _embed(running.url, texts)
    finally:
        running.stop()
    lengths = np.linalg.norm(vectors, axis=1)
    assert np.allclose(lengths, 1, atol=1e-4), (lengths.min(), lengths.max())
    about = {"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"], "code": _code(),
             "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__}
    np.savez_compressed(OUT, about=np.array(json.dumps(about)), texts=np.array(json.dumps(texts, ensure_ascii=False)),
                        vectors=vectors.astype(np.float16))
    print("written", OUT, len(texts), "texts")


if __name__ == "__main__":
    main()
