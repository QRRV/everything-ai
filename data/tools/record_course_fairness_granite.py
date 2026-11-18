# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-fairness-granite.npz: the embeddings of Granite Embedding 107M multilingual (the embedding model)
of every word rai-bias-in-embeddings uses, for its tests of bias in word embeddings. The words and the sentence come
from the lesson's checks.py: all_words() and test_words(). Each word is embedded as a text of its own, with the
course's code (everything_ai.embed), the way the lesson's demo embeds the learner's words, and each word of the three
tests also in the sentence "This is <word>." (SENTENCE).

The file holds `words` and `vectors` (one row of 384 numbers per word, in that order, float32, each of length 1),
`sentence_words` and `sentence_vectors` (the same for the sentences), and `about`, a JSON text with the model, its
revision, the commit of this code, the machine, threads, PyTorch and the seconds it took. Run it again only on
purpose, from the course folder, with Granite Embedding downloaded (everything_ai.embed.download()):

    uv run python data/tools/record_course_fairness_granite.py
"""
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from everything_ai import embed, hub, paths  # noqa: E402

OUT = paths.ROOT / "data" / "course-fairness-granite.npz"
LESSON = "rai-bias-in-embeddings"


def _checks():
    spec = importlib.util.spec_from_file_location(LESSON.replace("-", "_"), paths.LESSONS / LESSON / "checks.py")
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
    words, tests = C.all_words(), C.test_words()
    t = time.perf_counter()
    vectors = embed.embed(words).numpy().astype(np.float32)
    sentences = embed.embed([C.SENTENCE.format(w) for w in tests]).numpy().astype(np.float32)
    about = {"model": embed.REPO, "revision": hub.PINNED[(embed.REPO, "model.safetensors")]["revision"],
             "dtype": "float32", "pooling": "CLS", "sentence": C.SENTENCE, "code": _code(),
             "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
             "torch": torch.__version__, "seconds": round(time.perf_counter() - t, 1)}
    np.savez_compressed(OUT, words=np.array(words), vectors=vectors, sentence_words=np.array(tests),
                        sentence_vectors=sentences, about=np.array(json.dumps(about)))
    print(len(words), "words,", len(tests), "sentences,", about["seconds"], "seconds")


if __name__ == "__main__":
    main()
