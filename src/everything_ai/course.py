# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The course as data: lessons, routes, the parts of the course (the kern, the foundation and the domains), goals
and fixed phrases.

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
    """Every lesson in the route data, by id, with its place in the course: `domein` and `module`."""
    out = {}
    for f in sorted((paths.COURSE / "route" / "lessen").glob("*.yaml")):
        data = _load(f)
        for lesson in data["lessons"]:
            lesson.setdefault("requires", [])
            out[lesson["id"]] = lesson
    return out


@functools.cache
def _routes() -> dict:
    return _load(paths.COURSE / "route" / "routes.yaml")


KERN = "kern"   # the first route: the start everyone takes before choosing a domain (besluit 1.18)


@functools.cache
def route_steps(route_id: str = KERN) -> list[dict]:
    """A route in order: [{"id": ..., "extra": bool}, ...]."""
    route = next(r for r in _routes()["routes"] if r["id"] == route_id)
    steps = []
    for step in route["steps"]:
        if isinstance(step, str):
            steps.append({"id": step, "extra": False})
        else:
            steps.append({"id": step["id"], "extra": bool(step.get("extra"))})
    return steps


def kern_route() -> list[dict]:
    """The kern in order: [{"id": ..., "extra": bool}, ...]."""
    return route_steps(KERN)


def _names(r: dict) -> dict:
    return {"id": r["id"], "title": {"en": r["title"], "nl": r["title_nl"]},
            "short": {"en": r.get("short", r["title"]), "nl": r.get("short_nl", r["title_nl"])}}


def kern_names() -> dict:
    """The id, title and short name of the kern."""
    return _names(next(r for r in _routes()["routes"] if r["id"] == KERN))


def domain_routes() -> list[dict]:
    """The routes of the domains. Each starts after the kern: the core steps of the kern come first."""
    return [{**_names(r), "domein": r["domein"]} for r in _routes()["routes"] if r["id"] != KERN]


def route_ids() -> list[str]:
    return [r["id"] for r in _routes()["routes"]]


def route_for(lesson_id: str) -> str:
    """The route a lesson is numbered in: the route of its own domain when that has one, else the first route that
    has it (the kern for a lesson of the kern, the first domain that borrows it for a lesson of a shared module)."""
    domein = lessons()[lesson_id]["domein"]
    for r in domain_routes():
        if r["domein"] == domein:
            return r["id"]
    for rid in route_ids():
        if any(s["id"] == lesson_id for s in route_steps(rid)):
            return rid
    return KERN


def route_of(lesson_id: str, chosen: str | None = None) -> str:
    """The route a learner sees a lesson in: the route they chose when it has the lesson (a shared lesson that several
    domains borrow), else the route the lesson is numbered in."""
    if chosen in route_ids() and any(s["id"] == lesson_id for s in route_steps(chosen)):
        return chosen
    return route_for(lesson_id)


def milestones(route_id: str = KERN) -> list[dict]:
    route = next(r for r in _routes()["routes"] if r["id"] == route_id)
    return route.get("milestones", [])


def numbering(extras: bool = False, route_id: str = KERN) -> tuple[dict[str, int], int]:
    """The one count a learner sees: place in the route, without extra lessons unless switched on."""
    steps = [s for s in route_steps(route_id) if extras or not s["extra"]]
    return {s["id"]: i + 1 for i, s in enumerate(steps)}, len(steps)


@functools.cache
def order() -> tuple[str, ...]:
    """The whole course in order: the kern, then the routes of the domains, every lesson at its first place."""
    out: dict[str, None] = {}
    for rid in route_ids():
        for s in route_steps(rid):
            out.setdefault(s["id"])
    return tuple(out)


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
    """The smallest set of lessons that reaches a goal, in the order of the course."""
    need = chain(lesson_id) | {lesson_id}
    pos = {lid: i for i, lid in enumerate(order())}
    return sorted(need, key=lambda lid: pos.get(lid, len(pos)))


def goals() -> list[dict]:
    return _routes().get("goals", [])


@functools.cache
def structure() -> dict:
    """The parts of the course (route/domeinen.yaml): the foundation, whose kern modules everyone takes first and
    whose shared modules a domain pulls in, and the domains, in the groups the start page shows them in."""
    return _load(paths.COURSE / "route" / "domeinen.yaml")


def domains() -> list[str]:
    return [d["id"] for d in structure()["domeinen"]]


def groups() -> list[dict]:
    """The groups of domains as the start page shows them: [{"id": ..., "domains": [...]}, ...]."""
    return [{"id": g["id"], "domains": list(g["domeinen"])} for g in structure()["groepen"]]


def part(lesson_id: str) -> str:
    """Where a lesson belongs: "kern", "fundament" (a shared module of the foundation) or the id of its domain."""
    les = lessons()[lesson_id]
    if les["domein"] != "fundament":
        return les["domein"]
    kern = {m["id"] for m in structure()["fundament"]["modules"] if m.get("kern")}
    return "kern" if les["module"] in kern else "fundament"


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


def module_key(lesson_id: str) -> str:
    """The key of a lesson's module in display.yaml: domein/module."""
    les = lessons()[lesson_id]
    return f"{les['domein']}/{les['module']}"


def module_name(lesson_id: str, lang: str) -> str:
    """The name of a lesson's module (its chapter) in one language."""
    names = display()["modules"].get(module_key(lesson_id))
    return names[lang] if names else lessons()[lesson_id]["module"]


def part_name(part_id: str, lang: str) -> str:
    """The name of a part of the course (kern, fundament or a domain) in one language."""
    return display()["parts"][part_id][lang]


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
