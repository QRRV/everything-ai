# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course as data: lessons, routes, branches, goals and fixed phrases.

Everything here is read from files in the repository, so the start page, the lesson
header and the tests all count and name things the same way.
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

from everything_ai import paths

try:
    _Loader = yaml.CSafeLoader
except AttributeError:  # PyYAML without the C extension
    _Loader = yaml.SafeLoader

LANGS = ("en", "nl")


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


@functools.cache
def lessons() -> dict[str, dict]:
    """Every lesson in the route data, by id. A lesson without its own `tak` takes the file's."""
    out = {}
    for f in sorted((paths.COURSE / "route" / "lessen").glob("*.yaml")):
        data = _load(f)
        for lesson in data["lessons"]:
            lesson.setdefault("tak", data.get("tak"))
            lesson.setdefault("requires", [])
            out[lesson["id"]] = lesson
    return out


@functools.cache
def _routes() -> dict:
    return _load(paths.COURSE / "route" / "routes.yaml")


MAIN = "hoofdroute"


@functools.cache
def route_steps(route_id: str = MAIN) -> list[dict]:
    """A route in order: [{"id": ..., "extra": bool}, ...]."""
    route = next(r for r in _routes()["routes"] if r["id"] == route_id)
    steps = []
    for step in route["steps"]:
        if isinstance(step, str):
            steps.append({"id": step, "extra": False})
        else:
            steps.append({"id": step["id"], "extra": bool(step.get("extra"))})
    return steps


def main_route() -> list[dict]:
    """The main route in order: [{"id": ..., "extra": bool}, ...]."""
    return route_steps(MAIN)


def _names(r: dict) -> dict:
    return {"id": r["id"], "title": {"en": r["title"], "nl": r["title_nl"]},
            "short": {"en": r.get("short", r["title"]), "nl": r.get("short_nl", r["title_nl"])}}


def main_route_names() -> dict:
    """The id, title and short name of the first route, the one from zero to your own LLM."""
    return _names(next(r for r in _routes()["routes"] if r["id"] == MAIN))


def branch_routes() -> list[dict]:
    """The routes of branches that start after a step of the first route, such as beeld after the basics."""
    return [{**_names(r), "tak": r.get("tak"), "after": r.get("after")}
            for r in _routes()["routes"] if r["id"] != MAIN]


def route_ids() -> list[str]:
    return [r["id"] for r in _routes()["routes"]]


def route_for(lesson_id: str) -> str:
    """The route a lesson belongs to: the main route when it is in it, else the first branch route that has it."""
    for rid in route_ids():
        if any(s["id"] == lesson_id for s in route_steps(rid)):
            return rid
    return MAIN


def milestones() -> list[dict]:
    route = next(r for r in _routes()["routes"] if r["id"] == MAIN)
    return route.get("milestones", [])


def numbering(extras: bool = False, route_id: str = MAIN) -> tuple[dict[str, int], int]:
    """The one count a learner sees: place in the route, without extra lessons unless switched on."""
    steps = [s for s in route_steps(route_id) if extras or not s["extra"]]
    return {s["id"]: i + 1 for i, s in enumerate(steps)}, len(steps)


def chain(lesson_id: str) -> set[str]:
    """Every lesson this lesson builds on, directly or further back."""
    all_lessons = lessons()
    seen: set[str] = set()
    todo = list(all_lessons[lesson_id]["requires"])
    while todo:
        lid = todo.pop()
        if lid not in seen and lid in all_lessons:
            seen.add(lid)
            todo.extend(all_lessons[lid]["requires"])
    return seen


def goal_route(lesson_id: str) -> list[str]:
    """The smallest set of lessons that reaches a goal, in main-route order."""
    need = chain(lesson_id) | {lesson_id}
    order = {s["id"]: i for i, s in enumerate(main_route())}
    return sorted(need, key=lambda lid: order.get(lid, len(order)))


def goals() -> list[dict]:
    return _routes().get("goals", [])


@functools.cache
def branches() -> list[dict]:
    return _load(paths.COURSE / "route" / "takken.yaml")["takken"]


@functools.cache
def display() -> dict:
    return _load(paths.COURSE / "display.yaml")


@functools.cache
def _phrases() -> dict[str, dict]:
    return {p["id"]: p for p in _load(paths.COURSE / "phrases.yaml")["phrases"]}


def phrase(key: str, lang: str, **values) -> str:
    """A fixed phrase of the course (course/phrases.yaml), in one language."""
    text = _phrases()[key][lang]
    return text.format(**values) if values else text


def level_name(level: str, lang: str) -> str:
    return phrase(f"level-{level}", lang)


def module_name(module: str, lang: str) -> str:
    names = display()["modules"].get(module)
    return names[lang] if names else module


# --- built lessons ---------------------------------------------------------------

def lesson_dir(lesson_id: str) -> Path:
    return paths.LESSONS / lesson_id


def is_built(lesson_id: str) -> bool:
    d = lesson_dir(lesson_id)
    return all((d / f"{lang}.py").is_file() for lang in LANGS) and (d / "lesson.yaml").is_file()


def built_lessons() -> list[str]:
    if not paths.LESSONS.is_dir():
        return []
    return sorted(d.name for d in paths.LESSONS.iterdir() if d.is_dir() and is_built(d.name))


def lesson_meta(lesson_id: str) -> dict:
    """What a built lesson says about itself in both languages: title, goal, minutes, what you can do after."""
    return _load(lesson_dir(lesson_id) / "lesson.yaml")


def title(lesson_id: str, lang: str) -> str:
    """The title in this language for a built lesson; the English working title from the route otherwise."""
    if is_built(lesson_id):
        return lesson_meta(lesson_id)["title"][lang]
    return lessons()[lesson_id]["title"]
