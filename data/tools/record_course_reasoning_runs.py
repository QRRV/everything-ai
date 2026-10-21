# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes the course's runs of llm-reasoning-from-scratch into data/course-reasoning-runs.json: the lesson's own demo
run (checks.run_one), with the course's versions of every module the learner writes (an empty work folder, so the
course's ModernGPT, sft_example, sft_batch and pretrain_step stand in) and the lesson's solutions of steps_answer and
read_product. One run per call: the plan ("cpu" or "gpu"), the format ("steps" or "plain"), the steps and the seed.
The GPU plan's runs of 1,800 steps are those of the CPU plan (checks.SAME), so of the GPU plan only its third run is
recorded. The run replaces one of the same plan, format, steps and seed in the file. The runs of seed 0 of the CPU plan of 1,800
steps also keep their model as data/course-arith-steps.pt and data/course-arith-plain.pt, in float32. Run from the
course folder, one run at a time (the course: with 2 threads, also for the GPU plan's third run, on a CPU that other
work shared: with 4 threads a step took ten times as long there):

    OMP_NUM_THREADS=2 uv run python data/tools/record_course_reasoning_runs.py cpu steps 1800 0
    OMP_NUM_THREADS=2 uv run python data/tools/record_course_reasoning_runs.py gpu plain 7400 0
"""
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import paths  # noqa: E402

OUT = paths.ROOT / "data" / "course-reasoning-runs.json"


def _lesson():
    spec = importlib.util.spec_from_file_location("reasoning_runs", paths.LESSONS / "llm-reasoning-from-scratch" / "checks.py")
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
    plan, fmt, steps, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    C = _lesson()
    assert (fmt, steps) in C.RUNS[plan] and (plan, fmt, steps) not in C.SAME, (plan, fmt, steps)
    t = C.tools()
    assert t != "run", "the course's modules stop with an error"
    steps_answer, read_product = C.course_functions()
    begin = time.perf_counter()

    def show(d):
        if d["phase"] == "train" and (d["step"] % 100 < len(d["fresh"]) or d["step"] == d["steps"]):
            print(json.dumps({"step": d["step"], "loss": round(d["loss"], 4), "seconds": round(d["seconds"], 1)}), flush=True)

    r = C.run_one(fmt, steps, seed, steps_answer, read_product, t, show)
    assert isinstance(r, dict), r
    if plan == "cpu" and steps == 1800 and seed == 0:
        torch.save({k: v.detach().cpu().float() for k, v in r["model"].state_dict().items()},
                   paths.ROOT / C.COURSE_MODELS[fmt])
    kept = {"plan": plan, **C.kept(r), "threads": torch.get_num_threads(), "wall_seconds": round(time.perf_counter() - begin, 1)}
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "machine": f"{_cpu()}, {os.cpu_count()} cores", "torch": torch.__version__, "runs": []}
    data["code"] = _code()
    data["runs"] = [x for x in data["runs"] if (x["plan"], x["format"], x["steps"], x["seed"]) != (plan, fmt, steps, seed)] + [kept]
    order = {(p, f, s): i for p in C.RUNS for i, (f, s) in enumerate(C.RUNS[p])}
    data["runs"].sort(key=lambda x: (x["plan"] != "cpu", order[(x["plan"], x["format"], x["steps"])], x["seed"]))
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print("RESULT", json.dumps({k: kept[k] for k in ("plan", "format", "steps", "seed", "seconds", "wall_seconds")}),
          json.dumps({n: a["right"] for n, a in kept["accuracy"].items()}),
          [s["right"] for s in kept["snapshots"]], flush=True)


if __name__ == "__main__":
    main()
