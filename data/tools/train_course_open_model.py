# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-open-model.json, the course's run of llm-same-on-open-model: Qwen3-0.6B-Base trained with the
course's own versions of the learner's modules, at the settings of a GPU (the lesson's checks.run with kind "gpu"),
with what it measured; the loss on the same answers with the wrong layout of RoPE, for the lesson's guess; and how
many rows of the embedding have exactly the numbers of <|im_end|>: tokens the base model never trained. With
--cpu, it runs the demo at the settings of a CPU instead and adds how long that took and how much memory it used:
run that in a process of its own, before or after. Run from the course folder, with Qwen3 downloaded
(everything_ai.qwen3.download()):

    uv run python data/tools/train_course_open_model.py
    uv run python data/tools/train_course_open_model.py --cpu
"""
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())   # a project of its own, with only the course's versions in it

import torch  # noqa: E402

from everything_ai import paths, qwen3  # noqa: E402
from everything_ai.check import run_code  # noqa: E402

OUT = paths.ROOT / "data" / "course-open-model.json"


def _lesson():
    spec = importlib.util.spec_from_file_location("open_model", paths.LESSONS / "llm-same-on-open-model" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _show(d: dict) -> None:
    if d.get("step") is not None and d["step"] % 10 == 0:
        print(d["phase"], d["step"], "of", d["steps"], round(d["loss"], 3), round(d["seconds"]), "s", flush=True)
    elif d.get("done") is not None and d["done"] == d["total"]:
        print(d["phase"], "measured", d["total"], flush=True)


def main(cpu: bool) -> None:
    C = _lesson()
    rename = run_code(C._R_CODE).names["rename"]
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if cpu:
        start = time.perf_counter()
        r = C.run(rename, _show, kind="cpu")
        if isinstance(r, str):
            sys.exit(f"the run stopped: {r}")
        old["cpu"] = {"minutes": round((time.perf_counter() - start) / 60, 1),
                      "memory_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
                      "threads": torch.get_num_threads(), **{k: r[k] for k in ("base", "sft", "dpo", "steps")}}
    else:
        r = C.run(rename, _show, kind="gpu")
        if isinstance(r, str):
            sys.exit(f"the run stopped: {r}")
        enc = qwen3.tokenizer()
        end = enc.encode_single_token(C.END)
        wrong = qwen3.load(rename, rope=qwen3.rope_pairs)
        held = C.oasst()["held"][:C.MEASURE["gpu"]["loss"]]
        old.update({k: v for k, v in r.items()})
        old["rope"] = {"half": r["base"]["loss"], "pairs": C.answer_loss(wrong, held, enc, end)}
        old["measured"] = C.MEASURE["gpu"]
        old["params"] = sum(p.numel() for p in wrong.parameters())
        emb = wrong.tok.weight
        same = (emb == emb[enc.encode_single_token(C.TURN)]).all(dim=1)   # rows with exactly the numbers of <|im_end|>
        old["untrained"] = {"tokens": int(same.sum()), "special": int(same[end:enc.n_vocab].sum()),   # of the 26 special tokens
                            "unused": int(same[enc.n_vocab:].sum()), "text": int(same[:end].sum()),   # rows no token uses; text tokens
                            "has_start": bool(same[enc.encode_single_token(C.START)]),
                            "norm": round(float(emb[enc.encode_single_token(C.TURN)].norm()), 2),
                            "end_norm": round(float(emb[end].norm()), 2), "mean_norm": round(float(emb[:end].norm(dim=1).mean()), 2)}
        old["sft_examples"], old["dpo_pairs"] = len(C.oasst()["train"]), len(C.helpsteer2()["train"])
        old["threads"] = torch.get_num_threads()
        old["torch"] = torch.__version__
    OUT.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main("--cpu" in sys.argv[1:])
