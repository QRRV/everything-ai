# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The learner works in a copy of each lesson (work/lessons/<id>/<lang>.py), never in the course's own file.

marimo saves every change to the open file. Working in a copy means an update of the
course never overwrites your work, and your work never blocks an update:
- the first time you open a lesson, it is copied;
- a newer course version replaces your copy only while you have not changed it yet;
- "start over" puts the course version back. Your progress stays.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from everything_ai import course, paths


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(lesson_id: str, lang: str) -> Path:
    return course.lesson_dir(lesson_id) / f"{lang}.py"


def copy_path(lesson_id: str, lang: str) -> Path:
    return paths.WORK_LESSONS / lesson_id / f"{lang}.py"


def _base_file(lesson_id: str) -> Path:
    return paths.WORK_LESSONS / lesson_id / ".base.json"


def _bases(lesson_id: str) -> dict:
    try:
        return json.loads(_base_file(lesson_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _copy(lesson_id: str, lang: str) -> None:
    src, dst = source(lesson_id, lang), copy_path(lesson_id, lang)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    bases = _bases(lesson_id)
    bases[lang] = _hash(src)          # the version you started from
    _base_file(lesson_id).write_text(json.dumps(bases), encoding="utf-8")


def state(lesson_id: str, lang: str) -> str:
    """'none' (no copy yet), 'fresh' (your copy is the course version), 'changed' (you worked in it),
    'update' (you worked in it and the course has a newer version)."""
    dst = copy_path(lesson_id, lang)
    if not dst.is_file():
        return "none"
    base = _bases(lesson_id).get(lang)
    current = _hash(dst)
    newest = _hash(source(lesson_id, lang))
    if current == newest:
        return "fresh"
    if current == base:
        return "fresh"                 # untouched copy of an older version: ensure() will refresh it
    return "update" if base != newest else "changed"


def ensure(lesson_id: str, lang: str) -> Path:
    """The path of your copy, made or refreshed when that is safe."""
    dst = copy_path(lesson_id, lang)
    if not dst.is_file():
        _copy(lesson_id, lang)
    elif _hash(dst) == _bases(lesson_id).get(lang) and _hash(dst) != _hash(source(lesson_id, lang)):
        _copy(lesson_id, lang)         # you have not changed it: take the newer course version
    return dst


def reset(lesson_id: str) -> None:
    """Start the lesson over: both languages back to the course version."""
    for lang in course.LANGS:
        if source(lesson_id, lang).is_file():
            _copy(lesson_id, lang)


def file_key(lesson_id: str, lang: str) -> str:
    """How marimo names the file: relative to the folder it serves (work/)."""
    return copy_path(lesson_id, lang).relative_to(paths.WORK).as_posix()
