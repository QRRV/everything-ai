# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Measures how far one changed bit carries, for the guess of res-share-work: the run_seed of repro-guided-figure (the
solution of its exercise) with the course's versions of every module of the kit and the paper's settings, seed 0, the
steps of repro-guided-figure, on one thread, but with the last bit of the first starting weight of the model moved one
step up. From the course folder:

    uv run python data/tools/measure_one_bit.py

prints t, v and the validation accuracy at the step where the course's seed 0 reaches 99%, as ONE_BIT in
lessons/res-share-work/checks.py has them."""
import importlib.util
import os
import sys
import tempfile

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
    names = run_code(C.EXERCISES["run-seed"].solution).names
    Model = names["ModernGPT"]

    class OneBit(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                w = next(iter(self.parameters())).view(-1)
                w[0] = torch.nextafter(w[0], torch.tensor(1.0))   # the next float32 above it: one bit

    names["ModernGPT"] = OneBit
    log = names["run_seed"]({**C.PAPER, "fraction": C.FRACTION}, 0, C.STEPS)
    before = C.read(C.course_runs()[0])["v"]
    r = C.read(log)
    print({"t": r["t"], "v": r["v"], "v_before": before, "val_at": round(log["val"][log["step"].index(before)], 4)})


if __name__ == "__main__":
    sys.exit(main())
