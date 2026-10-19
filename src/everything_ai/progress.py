# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Progress and settings, stored as small JSON files in the learner's work folder. No account, nothing online.

Every lesson runs in its own process, so each lesson writes its own file (work/progress/<id>.json);
two lessons open at the same time never overwrite each other.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from everything_ai import paths

DEFAULT_SETTINGS = {"lang": "en", "theme": "system", "extras": False}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data) -> None:
    """Write via a temporary file and a rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# --- settings ----------------------------------------------------------------------

def settings() -> dict:
    return {**DEFAULT_SETTINGS, **_read(paths.SETTINGS, {})}


def save_settings(**changes) -> dict:
    current = settings()
    current.update({k: v for k, v in changes.items() if k in DEFAULT_SETTINGS})
    _write(paths.SETTINGS, current)
    return current


# --- progress ----------------------------------------------------------------------
# One file per lesson:
# {"exercises": {"sum": {"state": "correct", "solution_seen": false, "text": "3 + 7"}},
#  "answers": {"explain": {...}}, "done": true, "opened": 1760000000}

def _file(lesson_id: str) -> Path:
    return paths.PROGRESS / f"{lesson_id}.json"


def lesson(lesson_id: str) -> dict:
    return _read(_file(lesson_id), {})


def all_lessons() -> dict[str, dict]:
    if not paths.PROGRESS.is_dir():
        return {}
    return {f.stem: _read(f, {}) for f in paths.PROGRESS.glob("*.json")}


def update(lesson_id: str, fn) -> dict:
    data = lesson(lesson_id)
    fn(data)
    _write(_file(lesson_id), data)
    return data


def mark_opened(lesson_id: str, lang: str) -> None:
    def _set(d):
        d["opened"] = int(time.time())
        d["lang"] = lang
    update(lesson_id, _set)


def record_exercise(lesson_id: str, name: str, state: str, required: list[str], text: str | None = None) -> None:
    """Remember the latest state of one exercise. The lesson is done once every required exercise is correct."""
    def _set(d):
        ex = d.setdefault("exercises", {}).setdefault(name, {})
        ex["state"] = state
        if text is not None:
            ex["text"] = text
        if state == "correct":
            ex["ever_correct"] = True
        done = all(d["exercises"].get(r, {}).get("ever_correct") for r in required)
        if done and not d.get("done"):
            d["done"] = True
            d["done_at"] = int(time.time())
    update(lesson_id, _set)


def record_solution_seen(lesson_id: str, name: str) -> None:
    def _set(d):
        d.setdefault("exercises", {}).setdefault(name, {})["solution_seen"] = True
    update(lesson_id, _set)


def record_answer(lesson_id: str, key: str, value) -> None:
    """Answers that are not graded: a guess, a choice, an explanation and the points ticked."""
    def _set(d):
        d.setdefault("answers", {})[key] = value
    update(lesson_id, _set)


def saved_text(lesson_id: str, name: str) -> str | None:
    return lesson(lesson_id).get("exercises", {}).get(name, {}).get("text")


def saved_answer(lesson_id: str, key: str):
    return lesson(lesson_id).get("answers", {}).get(key)
