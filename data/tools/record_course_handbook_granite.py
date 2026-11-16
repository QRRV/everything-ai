# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-handbook-granite.npz: the embeddings of Granite Embedding 107M multilingual (the embedding model)
of the chunks of the handbook of Spoke & Chain (data/documents/spoke-and-chain.md, cut by everything_ai.summaries
into chunks of 80 words that overlap by 20) and of the English questions about it
(data/documents/spoke-and-chain-questions.json), for build-project-rag. The texts go through the course's own server
(everything_ai.serve, the endpoint /v1/embeddings) with the standard library's urllib, the way a learner's code sends
them: the chunks in one request, as the learner's make_library sends up to 64, and the questions in another.

The file holds `chunks` and `questions` (one row of 384 numbers per text, in order) as float16 to keep it small; every
row had length 1 before it was rounded. `about` is a JSON text with the model, its revision, the SHA-256 of the
handbook and of the questions, the commit of this code, the machine, threads and PyTorch. Run it again only on
purpose, from the course folder, with Granite Embedding downloaded (everything_ai.embed.download()):

    uv run python data/tools/record_course_handbook_granite.py
"""
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from everything_ai import embed, hub, serve, summaries  # noqa: E402

OUT = summaries.HANDBOOK_VECTORS
THREADS = 2


def _code() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=summaries.paths.ROOT, capture_output=True, text=True).stdout.strip()
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
    body = {"model": serve.EMBEDDER, "input": texts}
    req = urllib.request.Request(url + "/embeddings", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = json.loads(r.read())
    return np.array([d["embedding"] for d in sorted(raw["data"], key=lambda d: d["index"])], dtype=np.float32)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    torch.set_num_threads(THREADS)
    running = serve.start(serve.Model())
    try:
        arrays = {"chunks": _embed(running.url, summaries.handbook_chunks()),
                  "questions": _embed(running.url, [q["text"]["en"] for q in summaries.handbook_questions()])}
    finally:
        running.stop()
    for name, a in arrays.items():
        lengths = np.linalg.norm(a, axis=1)
        assert np.allclose(lengths, 1, atol=1e-4), (name, lengths.min(), lengths.max())
    about = {"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"],
             "handbook": _sha(summaries.HANDBOOK), "questions": _sha(summaries.HANDBOOK_QUESTIONS), "code": _code(),
             "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__}
    np.savez_compressed(OUT, about=np.array(json.dumps(about)), **{k: v.astype(np.float16) for k, v in arrays.items()})
    print("written", OUT, {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
