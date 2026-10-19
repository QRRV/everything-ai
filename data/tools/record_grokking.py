# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Records the course's runs of repro-guided-figure, data/course-grokking.jsonl: the lesson's own run_seed (the
solution of its exercise) with the course's versions of every module of the kit, the paper's settings at the
reproduction's fraction, STEPS steps, on one thread. One process per seed, from the course folder:

    uv run python data/tools/record_grokking.py 0 > grokking-0.json

prints the run as one line of JSON, and the seconds per step on stderr; the lines of the seeds 0, 1 and 2, in that
order, are data/course-grokking.jsonl. With --mirror it trains seed 0 up to MIRROR["step"] only, and prints the
validation accuracy there on the equations whose mirror b + a is in the training data, and on the others. A run takes
about 23 minutes on the course's CPU."""
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


def lesson_checks():
    spec = importlib.util.spec_from_file_location("figure_checks", paths.LESSONS / "repro-guided-figure" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


def main():
    torch.set_num_threads(1)
    C = lesson_checks()
    projects.builds_on(*C.BUILDS_ON)
    projects.on_path()
    projects.use_course_versions()   # every module of the kit, as the course wrote it
    names = run_code(C.EXERCISES["run-seed"].solution).names
    settings = {**C.PAPER, "fraction": C.FRACTION}
    if sys.argv[1] == "--mirror":
        step, seen = C.MIRROR["step"], {}
        train, val = names["dataset"](settings["p"], settings["fraction"], 0)
        have = {(a, b) for a, b, _ in train}
        mirror = torch.tensor([(b, a) in have for a, b, _ in val])
        accuracy = names["accuracy"]

        def measured(model, x, y):   # the lesson's accuracy, and at the step, also on the two parts of validation
            if len(x) == len(val) and seen.get("calls", 0) == seen.get("at"):
                for part, rows in (("mirrored", mirror), ("others", ~mirror)):
                    seen[part] = accuracy(model, x[rows], y[rows])
            seen["calls"] = seen.get("calls", 0) + 1
            return accuracy(model, x, y)

        seen["at"] = 2 * names["log_steps"](step, 20).index(step) + 1   # the call for validation at that step
        names["accuracy"] = measured
        log = names["run_seed"](settings, 0, step)
        print(json.dumps({"step": step, "share": round(mirror.float().mean().item(), 4), "mirrored": round(seen["mirrored"], 4),
                          "others": round(seen["others"], 4), "val": log["val"][-1]}))
        return
    seed = int(sys.argv[1])
    start = time.perf_counter()
    log = names["run_seed"](settings, seed, C.STEPS, progress=lambda log: print(seed, log["step"][-1], log["train"][-1], log["val"][-1], file=sys.stderr, flush=True))
    print(f"seconds per step: {(time.perf_counter() - start) / C.STEPS:.3f}", file=sys.stderr)
    print(json.dumps({k: [round(v, 4) for v in log[k]] if k in ("train", "val") else log[k] for k in log}))


if __name__ == "__main__":
    main()
