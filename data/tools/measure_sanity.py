# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Measures the two runs that repro-guided-sanity shows: the course's ModernGPT with the paper's settings
(lessons/repro-guided-build/checks.py, PAPER) learning one batch by heart, the first 512 training equations of
addition modulo 97 at the fraction 0.3 and seed 0, with AdamW at the paper's learning rate and betas but without
weight decay, on one thread. One run reads its guess for c at the place of =, as it should; the other at the place
of a, the bug the lesson plants. Run from the course folder:

    uv run python data/tools/measure_sanity.py

It prints the loss on the batch before each of the first 150 steps, and after the last, for both runs: the numbers
in RUNS of the lesson's checks. It takes about half a minute on the course's CPU."""
import os
import tempfile

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai.lesson import lesson  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(1)
    C = lesson("repro-guided-sanity", "en").checks
    for place in ("right", "wrong"):
        print(place, [round(v, 4) for v in C.course_run(place)])
