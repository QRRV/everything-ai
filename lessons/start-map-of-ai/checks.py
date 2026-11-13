# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-map-of-ai, and the numbers about the course that the lesson shows.

The numbers come from the route data itself (route/domeinen.yaml, route/routes.yaml and the lessons), so the lesson
stays right when the course grows."""
from collections import Counter

from everything_ai import course
from everything_ai.check import Sort

LEVELS = ["basis", "gevorderd", "expert"]
N_DOMAINS = len(course.domains())
N_GROUPS = len(course.groups())
CORE = [s["id"] for s in course.kern_route() if not s["extra"]]   # the core without its extra lessons


def core_levels() -> dict[str, int]:
    """The lessons of the core per level: all at the basics level."""
    c = Counter(course.lessons()[lid]["level"] for lid in CORE)
    return {lvl: c.get(lvl, 0) for lvl in LEVELS}


def domain_lessons(domain: str) -> int:
    """How many lessons of a domain are ready: built lessons whose part is that domain."""
    return sum(1 for lid in course.built_lessons() if lid in course.lessons() and course.part(lid) == domain)


def map_groups(lang: str, words: tuple[str, str, str]) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """The groups of domains for the figure: each group's name and, per domain, its name, how many lessons it has
    (words: one lesson, n lessons, still to come) and whether it has any."""
    out = []
    for g in course.groups():
        domains = []
        for d in g["domains"]:
            n = domain_lessons(d)
            line = words[0] if n == 1 else words[1].format(n=n) if n else words[2]
            domains.append((course.part_name(d, lang), line, n > 0))
        out.append((course.display()["groups"][g["id"]][lang], domains))
    return out


def goal(goal_id: str) -> dict:
    """What a goal route takes: how many lessons, where the goal sits along the core and the route of its domain
    (without extra lessons), and its lessons per level and per part of the course."""
    g = next(x for x in course.goals() if x["id"] == goal_id)
    route = course.goal_route(g["goal"])
    rid = course.route_for(g["goal"])
    path = CORE + ([s["id"] for s in course.route_steps(rid) if not s["extra"] and s["id"] not in CORE]
                   if rid != course.KERN else [])
    levels = Counter(course.lessons()[lid]["level"] for lid in route)
    parts = Counter(course.part(lid) for lid in route)
    return {
        "lessons": len(route),
        "place": path.index(g["goal"]) + 1,
        "route": rid,
        "levels": {lvl: levels.get(lvl, 0) for lvl in LEVELS},
        "core": parts.get("kern", 0),
        "shared": parts.get("fundament", 0),
        "domain": len(route) - parts.get("kern", 0) - parts.get("fundament", 0),
    }


# The lesson is done when these are correct. "pick-route" and "foundation" are choice questions in the lesson.
REQUIRED = ["pick-route", "kinds", "foundation"]

EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "domain": {"en": "Domain", "nl": "Domein"},
            "level": {"en": "Level", "nl": "Niveau"},
            "route": {"en": "Route", "nl": "Route"},
        },
        items={
            "vision": {"en": "**Computer vision.** Lessons about pictures and video, with a project of its own.",
                       "nl": "**Computer vision.** Lessen over beelden en video, met een eigen project."},
            "intermediate": {"en": "**Intermediate.** The label on the lessons where you build a GPT.",
                             "nl": "**Gevorderd.** Het label op de lessen waarin je een GPT bouwt."},
            "attention-goal": {"en": "**I want to understand attention.** Only the lessons this goal builds on, one after the other.",
                               "nl": "**Ik wil attention snappen.** Alleen de lessen waar dit doel op voortbouwt, na elkaar."},
            "speech": {"en": "**Speech, audio and music.** A part of AI that builds on the core.",
                       "nl": "**Spraak, audio en muziek.** Een deel van AI dat op de kern voortbouwt."},
            "expert": {"en": "**Expert.** The label on the lessons where you reproduce a paper on your own.",
                       "nl": "**Expert.** Het label op de lessen waarin je zelfstandig een paper nabouwt."},
            "core": {"en": "**The core.** The lessons everyone takes first, one after the other, from your first line of Python to a first language model.",
                     "nl": "**De kern.** De lessen die iedereen eerst doet, na elkaar, van je eerste regel Python tot een eerste language model."},
        },
        solution={"vision": "domain", "intermediate": "level", "attention-goal": "route",
                  "speech": "domain", "expert": "level", "core": "route"},
        nudge={
            "vision": {"en": "Is it a part of AI, a label for how advanced a lesson is, or an order of lessons?",
                       "nl": "Is het een deel van AI, een label voor hoe gevorderd een les is, of een volgorde van lessen?"},
            "intermediate": {"en": "Does it say which part of AI a lesson is about, or how advanced the lesson is?",
                             "nl": "Zegt het over welk deel van AI een les gaat, of hoe gevorderd de les is?"},
            "attention-goal": {"en": "Is it a part of AI, or a list of lessons to take one after the other?",
                         "nl": "Is het een deel van AI, of een lijst lessen die je na elkaar doet?"},
            "speech": {"en": "Does it say how advanced a lesson is, or which part of AI it is about?",
                       "nl": "Zegt het hoe gevorderd een les is, of over welk deel van AI hij gaat?"},
            "expert": {"en": "Does it say which part of AI a lesson is about, or how advanced the lesson is?",
                       "nl": "Zegt het over welk deel van AI een les gaat, of hoe gevorderd de les is?"},
            "core": {"en": "Is it one part of AI, or an order of lessons through several parts?",
                     "nl": "Is het één deel van AI, of een volgorde van lessen door meerdere delen heen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the items again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de onderdelen opnieuw uit."},
            "empty": {"en": "Pick a group for each item.",
                      "nl": "Kies voor elk onderdeel een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are in the right group: two domains, two levels and two routes.",
                        "nl": "Alle {n} staan in de goede groep: twee domeinen, twee niveaus en twee routes."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"vision": "domain", "intermediate": "level", "attention-goal": "route", "speech": "domain", "expert": "level", "core": "route"}
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _RIGHT, "correct", None),
    ("kinds", {"vision": "domain", "intermediate": "level"}, "wrong", "unfinished"),
    ("kinds", {**_RIGHT, "intermediate": "domain"}, "wrong", "wrong:intermediate"),
    ("kinds", {**_RIGHT, "attention-goal": "domain"}, "wrong", "wrong:attention-goal"),
    ("kinds", {**_RIGHT, "core": "level", "speech": "route"}, "wrong", "wrong:speech"),
]
