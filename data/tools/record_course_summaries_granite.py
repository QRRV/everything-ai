# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-summaries-granite.npz: the embeddings of Granite Embedding 107M multilingual (the embedding
model) of every point and every question of data/course-summaries.json, in English and in Dutch, for the lessons on
RAG of the domain bouwen that search by meaning. The texts go through the course's own server (everything_ai.serve,
the endpoint /v1/embeddings) with the standard library's urllib, the way a learner's code sends them, 64 at a time.

The file holds `points_en`, `points_nl` (1,266 by 384) and `questions_en`, `questions_nl` (28 by 384), in the order of
course-summaries.json, as float16 to keep it small; every row had length 1 before it was rounded. `about` is a JSON
text with the model, its revision, the commit course-summaries.json was made from, the commit of this code, and the
seconds it took on the machine it names, with Granite loaded: all English points in requests of 64 ("points"), and
each English question alone, one request each, as a learner's search sends it ("question", the median).
Run it again only on purpose, from the course folder, with Granite Embedding downloaded
(everything_ai.embed.download()):

    uv run python data/tools/record_course_summaries_granite.py
"""
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
import urllib.request

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from everything_ai import embed, hub, paths, serve, summaries  # noqa: E402

OUT = paths.ROOT / "data" / "course-summaries-granite.npz"
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
    torch.set_num_threads(THREADS)
    running = serve.start(serve.Model())
    try:
        _embed(running.url, ["Granite loads at the first request."])
        arrays, seconds = {}, {}
        for lang in ("en", "nl"):
            start = time.perf_counter()
            arrays[f"points_{lang}"] = _embed(running.url, summaries.points(lang))
            if lang == "en":
                seconds["points"] = round(time.perf_counter() - start, 2)
            arrays[f"questions_{lang}"] = _embed(running.url, [q["text"][lang] for q in summaries.questions()])
        alone = []
        for q in summaries.questions():
            start = time.perf_counter()
            one = _embed(running.url, [q["text"]["en"]])
            alone.append(time.perf_counter() - start)
            assert abs(float(one[0] @ arrays["questions_en"][len(alone) - 1]) - 1) < 1e-3
        seconds["question"] = round(statistics.median(alone), 3)
    finally:
        running.stop()
    for name, a in arrays.items():
        lengths = np.linalg.norm(a, axis=1)
        assert np.allclose(lengths, 1, atol=1e-4), (name, lengths.min(), lengths.max())
    about = {"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"],
             "made_from": summaries.made_from(), "code": _code(), "seconds": seconds,
             "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": THREADS, "torch": torch.__version__}
    np.savez_compressed(OUT, about=np.array(json.dumps(about)), **{k: v.astype(np.float16) for k, v in arrays.items()})
    print("written", OUT)


if __name__ == "__main__":
    main()
