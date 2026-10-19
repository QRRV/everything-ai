# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Records one run of the ablation of repro-guided-ablation: the run_seed of repro-guided-figure (the solution of its
exercise) with the course's versions of every module of the kit and the paper's settings, but with the weight decay
and the fraction of the table given, with the seed given (0 when none is), the steps of repro-guided-figure, on one
thread. From the course folder:

    uv run python data/tools/record_ablation.py 0.0 0.3 > ablation-0.0-0.3.json
    uv run python data/tools/record_ablation.py 0.0 0.9 2 > ablation-0.0-0.9-seed-2.json

prints the run as one line of JSON with its weight decay and fraction, and the seconds per step on stderr."""
import importlib.util
import json
import os
import sys
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, projects  # noqa: E402
from everything_ai.check import run_code  # noqa: E402


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("figure_checks", paths.LESSONS / "repro-guided-figure" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    projects.builds_on(*C.BUILDS_ON)
    projects.on_path()
    projects.use_course_versions()   # every module of the kit, as the course wrote it
    run_seed = run_code(C.EXERCISES["run-seed"].solution).names["run_seed"]
    weight_decay, fraction = float(sys.argv[1]), float(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    settings = {**C.PAPER, "weight_decay": weight_decay, "fraction": fraction}
    start = time.perf_counter()
    log = run_seed(settings, seed, C.STEPS, progress=lambda log: print(weight_decay, fraction, log["step"][-1], log["train"][-1], log["val"][-1], file=sys.stderr, flush=True))
    print(f"seconds per step: {(time.perf_counter() - start) / C.STEPS:.3f}", file=sys.stderr)
    print(json.dumps({"weight_decay": weight_decay, "fraction": fraction, "seed": log["seed"], "step": log["step"],
                      "train": [round(v, 4) for v in log["train"]], "val": [round(v, 4) for v in log["val"]]}))


if __name__ == "__main__":
    main()
