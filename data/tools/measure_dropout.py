# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Measures the course's experiment for the dropout paper, the solution of the experiment of repro-own-build, for its
play block: the test error with dropout ("claim") and without ("other"), seeds 0, 1 and 2, after each of the budgets
of BUDGETS steps, every run a run of its own, on one thread. From the course folder:

    uv run python data/tools/measure_dropout.py

prints RUNS and SHORT_RUN as lessons/repro-own-build/checks.py has them. With --ablation, it measures the two ablations of
repro-own-ablation instead, at the longest budget: dropout in the hidden layers only, and at the input only, and prints
their runs as ABLATIONS in lessons/repro-own-ablation/checks.py has them."""
import importlib.util
import os
import sys
import tempfile

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own

import torch  # noqa: E402

from everything_ai import paths  # noqa: E402
from everything_ai.check import run_code  # noqa: E402


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("build_checks", paths.LESSONS / "repro-own-build" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    names = run_code(C.E).names
    run = names["run"]
    if "--ablation" in sys.argv:
        names["SETTINGS"].update({"no-input": (0.0, 0.5), "no-hidden": (0.2, 0.0)})   # the dict make_model reads
        print({s: [run(s, seed, C.BUDGETS[-1]) for seed in C.SEEDS] for s in ("no-input", "no-hidden")})
        return
    runs = {steps: {s: [run(s, seed, steps) for seed in C.SEEDS] for s in C.SETTINGS} for steps in C.BUDGETS}
    print(runs)
    print({s: run(s, 0, C.STEPS) for s in C.SETTINGS})   # SHORT_RUN


if __name__ == "__main__":
    sys.exit(main())
