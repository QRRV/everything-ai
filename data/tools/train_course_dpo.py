# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Trains the course's own DPO adapters: the ones llm-dpo and the lessons after it fall back on. It runs the demo of
llm-dpo with the course's versions of every module the learner writes, on the course's SFT model, data/course-sft.pt,
with the preference pairs of data/course-dpo-answers.json, on two threads. Run from the course folder:

    uv run python data/tools/train_course_dpo.py

It writes the adapters in float16 to data/course-dpo.pt, and the run and the answers after to data/course-dpo.json."""
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402

OUT = paths.ROOT / "data" / "course-dpo.pt"
INFO = paths.ROOT / "data" / "course-dpo.json"


def main():
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("dpo_checks", paths.LESSONS / "llm-dpo" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    last = [0.0]

    def show(p):
        if time.perf_counter() - last[0] > 60:
            last[0] = time.perf_counter()
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}, flush=True)

    t0 = time.perf_counter()
    r = C.run(show)
    total = time.perf_counter() - t0
    assert isinstance(r, dict) and r["source"] == "course", r
    folder = projects.folder() / "llm"
    adapters = torch.load(folder / f"{C.NAME}.pt", map_location="cpu", weights_only=True)
    torch.save({k: v.half() for k, v in adapters.items()}, OUT)
    keep = {**json.loads((folder / f"{C.NAME}.json").read_text(encoding="utf-8")), "total_seconds": round(total, 1), "threads": 2, "torch": torch.__version__,
            "answers": C.ANSWERS, "answers_sha256": hashlib.sha256((paths.ROOT / C.ANSWERS).read_bytes()).hexdigest()}
    INFO.write_text(json.dumps(keep, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(os.environ["EAI_WORK"], ignore_errors=True)
    print(f"before {keep['before']['used']} of {keep['before']['total']}, after {keep['after']['used']} of {keep['after']['total']}")
    print(OUT.name, hashlib.sha256(OUT.read_bytes()).hexdigest())
    print(INFO.name, hashlib.sha256(INFO.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
