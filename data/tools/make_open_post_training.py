# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes the open data llm-same-on-open-model trains Qwen3-0.6B-Base on: short English questions with their best
answer from OpenAssistant Conversations (OASST1, Apache-2.0), in data/oasst-short.json, and short single-turn pairs
of answers with a clear human preference from HelpSteer2 (CC-BY-4.0), in data/helpsteer2-short.json. Short means
short in Qwen3's tokens, so that a step stays small: a question of at most Q_MAX tokens, answers of at most A_MAX
(OASST, at least A_MIN) or H_MAX (HelpSteer2). HELD of each are held out: chosen by the SHA-256 of their question, so
that the same question never lands on both sides. Download the two files at the commits below and run, from the
course folder, with Qwen3's tokenizer downloaded (everything_ai.qwen3.download()):

    uv run python data/tools/make_open_post_training.py 2023-04-12_oasst_ready.trees.jsonl.gz preference.jsonl.gz
"""
import gzip
import hashlib
import json
import sys

from everything_ai import paths, qwen3

OASST = ("https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/fdf72ae0827c1cda404aff25b6603abec9e3399b/"
         "2023-04-12_oasst_ready.trees.jsonl.gz", "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7")
HELPSTEER2 = ("https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/990b2711a36180dd19d9c94b8627844866f8982a/"
              "preference/preference.jsonl.gz", "a5cd48600fb7a330cf0ccc8f59051e24e8f236907c379f42eff1ba18da55204b")
Q_MAX, A_MIN, A_MAX, H_MAX = 64, 16, 192, 224
HELD = {"oasst": 50, "helpsteer2": 100}
TOXIC = 0.05   # OASST's detoxify score of a message: at most this


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _order(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def _clean(m: dict) -> bool:
    toxic = (m.get("detoxify") or {}).get("toxicity", 0.0)
    return not m.get("deleted") and m.get("review_result") is not False and toxic <= TOXIC


def oasst(path: str, n) -> list:
    """(question, answer) for every English tree: its first question, and the assistant's answer that people ranked
    first, both clean and short."""
    out, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)["prompt"]
            if p.get("lang") != "en" or not _clean(p) or p["text"].strip() in seen:
                continue
            best = [r for r in p.get("replies", []) if r["role"] == "assistant" and r.get("rank") == 0 and _clean(r)]
            if not best:
                continue
            q, a = p["text"].strip(), best[0]["text"].strip()
            if n(q) <= Q_MAX and A_MIN <= n(a) <= A_MAX:
                out.append((q, a))
                seen.add(q)
    return out


def helpsteer2(path: str, n) -> list:
    """(question, chosen, rejected, strength) for every single-turn pair whose preference is at least "better"
    (|strength| >= 2; positive: the second answer is better), with a short question and short answers."""
    out, seen = [], set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["prompt"].strip()
            if "<extra_id_1>" in q or abs(r["preference_strength"]) < 2 or q in seen:   # a conversation of more turns
                continue
            a, b = r["response_1"].strip(), r["response_2"].strip()
            chosen, rejected = (b, a) if r["preference_strength"] > 0 else (a, b)
            if n(q) <= Q_MAX + 32 and n(chosen) <= H_MAX and n(rejected) <= H_MAX and chosen != rejected:
                out.append((q, chosen, rejected, abs(r["preference_strength"])))
                seen.add(q)
    return out


def main(oasst_path: str, helpsteer2_path: str) -> None:
    for path, (address, sha) in [(oasst_path, OASST), (helpsteer2_path, HELPSTEER2)]:
        if _sha(path) != sha:
            sys.exit(f"{path} is not the file of {address}")
    enc = qwen3.tokenizer()

    def n(text: str) -> int:
        return len(qwen3.encode(enc, text))

    for name, rows, address, license, keys in [
        ("oasst", oasst(oasst_path, n), OASST[0], "Apache-2.0", ["question", "answer"]),
        ("helpsteer2", helpsteer2(helpsteer2_path, n), HELPSTEER2[0], "CC-BY-4.0", ["question", "chosen", "rejected", "strength"]),
    ]:
        rows.sort(key=lambda r: _order(r[0]))
        held, train = rows[:HELD[name]], rows[HELD[name]:]
        out = {"source": address, "license": license, "keys": keys, "held": held, "train": train}
        target = paths.ROOT / "data" / f"{name}-short.json"
        target.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
        print(target.name, "held", len(held), "train", len(train), round(target.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
