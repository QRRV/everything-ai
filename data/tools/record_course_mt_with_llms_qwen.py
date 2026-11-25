# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-mt-with-llms-qwen.json, the course's run of nlp-mt-with-llms: Qwen3-0.6B (the chat model), in
float32 with the course's code, non-thinking and greedy (temperature 0), on the lesson's twelve Dutch texts
(checks.DOCUMENTS), each translated into English in the lesson's three ways (checks.WAYS): alone, each sentence in a
chat of its own (checks.alone_messages); chat, sentence by sentence in one chat that keeps every earlier message
(checks.chat_messages); and document, the whole text in one prompt with numbered lines (checks.document_messages).
Every answer with its text, tokens, seconds and how it ended, under documents[i][way] (a list of one answer per
sentence for alone and chat, one answer for document). Each text keeps the SHA-256 of its sentences and of the two
instructions: a text whose prompts changed is asked again. The file is written after every text, and a run that was
stopped goes on where it was. Run from the course folder, with the chat model downloaded
(everything_ai.qwen3.download_chat()):

    uv run python data/tools/record_course_mt_with_llms_qwen.py
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time

os.environ.setdefault("EAI_WORK", tempfile.mkdtemp())

import torch  # noqa: E402

from everything_ai import chat, hub, paths, qwen3  # noqa: E402

LESSON = "nlp-mt-with-llms"
OUT = paths.ROOT / "data" / "course-mt-with-llms-qwen.json"


def _checks():
    spec = importlib.util.spec_from_file_location("nlp_mt_with_llms", paths.LESSONS / LESSON / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def text_sha(C, sentences) -> str:
    """The SHA-256 of a text's prompts: the two instructions and the sentences, one per line."""
    return hashlib.sha256("\n".join([C.INSTRUCTION, C.DOCUMENT_INSTRUCTION, *sentences]).encode("utf-8")).hexdigest()


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
    C = _checks()
    t = time.perf_counter()
    model = chat.Chat()

    def ask(messages: list, max_tokens: int) -> dict:
        a = model.answer(messages, max_tokens=max_tokens, temperature=0)
        return {"text": a["message"]["content"], "tokens": a["tokens"], "seconds": a["seconds"], "finish": a["finish"]}

    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out.update({"lesson": LESSON, "model": qwen3.CHAT, "revision": hub.PINNED[(qwen3.CHAT, "model.safetensors")]["revision"],
                "dtype": "float32", "machine": f"{_cpu()}, {os.cpu_count()} cores", "threads": torch.get_num_threads(),
                "torch": torch.__version__, "temperature": 0, "thinking": False, "max_tokens": C.MAX_TOKENS,
                "document_tokens": C.DOCUMENT_TOKENS, "instruction": C.INSTRUCTION,
                "document_instruction": C.DOCUMENT_INSTRUCTION, "load_seconds": round(time.perf_counter() - t, 1)})
    done = out.setdefault("documents", [])
    for i, d in enumerate(C.DOCUMENTS):
        sha = text_sha(C, d["nl"])
        if i < len(done) and done[i].get("sha256") == sha:
            continue
        rec = {"place": d["place"], "sha256": sha, "alone": [ask(C.alone_messages(s), C.MAX_TOKENS) for s in d["nl"]],
               "chat": []}
        for k in range(len(d["nl"])):
            answers = [a["text"] for a in rec["chat"]]
            rec["chat"].append(ask(C.chat_messages(C.INSTRUCTION, d["nl"], answers), C.MAX_TOKENS))
        rec["document"] = ask(C.document_messages(d["nl"]), C.DOCUMENT_TOKENS)
        done[i:i + 1] = [rec]
        out["code"] = _code()
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(i, d["place"], repr(rec["document"]["text"][:80]), flush=True)


if __name__ == "__main__":
    main()
