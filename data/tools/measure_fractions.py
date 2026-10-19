# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Measures what repro-guided-build says about the training data fraction: the course's ModernGPT with the paper's
settings (lessons/repro-guided-build/checks.py, PAPER) on addition modulo 97, seed 0, at each fraction given, on one
thread. It logs the accuracy on all training and all validation equations every 100 steps, and stops when validation
reaches 99% or after the budget. Run from the course folder:

    uv run python data/tools/measure_fractions.py 0.9 0.5 0.3

It prints, per fraction, the first logged step at which training and validation reach 99% (t and v), and the
seconds per step: the numbers in FRACTIONS and SECONDS_PER_STEP of the lesson's checks. A run at 0.3 takes about
half an hour on the course's CPU."""
import importlib.util
import json
import os
import sys
import tempfile
import time

os.environ["EAI_WORK"] = tempfile.mkdtemp()   # a project of its own, with only the course's versions in it

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from everything_ai import paths, projects  # noqa: E402
from everything_ai.check import run_code  # noqa: E402

STEPS = 20_000   # the most steps per run here; the paper's budget is 10^5


def run(C, ModernGPT, fraction: float) -> dict:
    s = C.PAPER
    dataset = run_code(C.EXERCISES["dataset"].solution).names["dataset"]
    train, val = dataset(s["p"], fraction, 0)
    xt, yt = C.tokens(train, s["p"])
    xv, yv = C.tokens(val, s["p"])
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    model = ModernGPT(s["p"] + 2, s["width"], s["heads"], s["heads"], s["layers"])
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"], betas=s["betas"])
    warm = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / s["warmup"]))
    batch = min(s["batch"], len(train) // 2)

    def accuracy(x, y):
        with torch.no_grad():
            return (model(x)[:, -1].argmax(-1) == y).float().mean().item()

    t = v = None
    start = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train), (batch,), generator=g)
        loss = F.cross_entropy(model(xt[idx])[:, -1], yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        warm.step()
        if step % 100 == 0:
            tr, va = accuracy(xt, yt), accuracy(xv, yv)
            print(fraction, step, round(tr, 4), round(va, 4), file=sys.stderr, flush=True)
            if t is None and tr >= 0.99:
                t = step
            if v is None and va >= 0.99:
                v = step
                break
    return {"t": t, "v": v, "seconds_per_step": (time.perf_counter() - start) / step}


def main():
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("build_checks", paths.LESSONS / "repro-guided-build" / "checks.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    ModernGPT = run_code(projects.course_code("llm", "modern_gpt")).names["ModernGPT"]
    out = {float(f): run(C, ModernGPT, float(f)) for f in sys.argv[1:]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
