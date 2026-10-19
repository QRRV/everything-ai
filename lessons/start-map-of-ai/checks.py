# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson start-map-of-ai, and the numbers about the route that the lesson shows.

The numbers come from the route data itself, so the lesson stays right when the route changes."""
from collections import Counter

from everything_ai import course
from everything_ai.check import Sort

LEVELS = ["basis", "gevorderd", "expert"]
MAIN_BRANCHES = ["stam", "taal", "onderzoek"]   # the foundation, language and research: most of the main route
N_BRANCHES = len(course.branches())


def counts() -> dict[tuple[str, str], int]:
    """Lessons of the main route (without extra lessons) per level and branch. The few lessons
    from other branches are counted together as "other"."""
    lessons = course.lessons()
    out: Counter = Counter()
    for step in course.main_route():
        if not step["extra"]:
            les = lessons[step["id"]]
            tak = les["tak"] if les["tak"] in MAIN_BRANCHES else "other"
            out[(les["level"], tak)] += 1
    return dict(out)


def goal(goal_id: str) -> dict:
    """What a goal route takes: how many lessons, where the goal sits in the main route, and per level and branch."""
    g = next(x for x in course.goals() if x["id"] == goal_id)
    route = course.goal_route(g["goal"])
    numbers, _ = course.numbering()
    lessons = course.lessons()
    levels = Counter(lessons[lid]["level"] for lid in route)
    return {
        "lessons": len(route),
        "place": numbers[g["goal"]],
        "levels": {lvl: levels.get(lvl, 0) for lvl in LEVELS},
        "foundation": sum(1 for lid in route if lessons[lid]["tak"] == "stam"),
    }


def grid() -> dict[tuple[int, int], int]:
    """The counts as the figure in the lesson draws them: a row per level, a column per branch, "other" last."""
    c = counts()
    return {(i, j): c.get((lvl, tak), 0) for i, lvl in enumerate(LEVELS)
            for j, tak in enumerate(MAIN_BRANCHES + ["other"])}


# The lesson is done when these are correct. "pick-route" and "foundation" are choice questions in the lesson.
REQUIRED = ["pick-route", "kinds", "foundation"]

EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "branch": {"en": "Branch", "nl": "Tak"},
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
            "speech": {"en": "**Speech, audio and music.** A part of AI that grows from the foundation.",
                       "nl": "**Spraak, audio en muziek.** Een deel van AI dat uit het fundament groeit."},
            "expert": {"en": "**Expert.** The label on the lessons where you reproduce a paper on your own.",
                       "nl": "**Expert.** Het label op de lessen waarin je zelfstandig een paper nabouwt."},
            "main": {"en": "**The main route.** From your first line of Python to a model of your own and a reproduced paper.",
                     "nl": "**De hoofdroute.** Van je eerste regel Python tot een eigen model en een nagebouwde paper."},
        },
        solution={"vision": "branch", "intermediate": "level", "attention-goal": "route",
                  "speech": "branch", "expert": "level", "main": "route"},
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
            "main": {"en": "Is it one part of AI, or an order of lessons through several parts?",
                     "nl": "Is het één deel van AI, of een volgorde van lessen door meerdere delen heen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the items again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de onderdelen opnieuw uit."},
            "empty": {"en": "Pick a group for each item.",
                      "nl": "Kies voor elk onderdeel een groep."},
            "unfinished": {"en": "Everything you picked so far is right. Pick a group for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest een groep (nog {n})."},
            "wrong": {"en": "\u201c{item}\u201d is in the wrong group.",
                      "nl": "\u2018{item}\u2019 staat in de verkeerde groep."},
            "correct": {"en": "All {n} are in the right group: two branches, two levels and two routes.",
                        "nl": "Alle {n} staan in de goede groep: twee takken, twee niveaus en twee routes."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"vision": "branch", "intermediate": "level", "attention-goal": "route", "speech": "branch", "expert": "level", "main": "route"}
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _RIGHT, "correct", None),
    ("kinds", {"vision": "branch", "intermediate": "level"}, "wrong", "unfinished"),
    ("kinds", {**_RIGHT, "intermediate": "branch"}, "wrong", "wrong:intermediate"),
    ("kinds", {**_RIGHT, "attention-goal": "branch"}, "wrong", "wrong:attention-goal"),
    ("kinds", {**_RIGHT, "main": "level", "speech": "route"}, "wrong", "wrong:speech"),
]
