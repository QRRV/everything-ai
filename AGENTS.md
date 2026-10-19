# Everything AI: notes for AI agents

This file is for an AI assistant that a learner points at this repository, such as Claude Code, Codex or Cursor. It says how the course is put together and how to help a learner without taking the learning away from them. `CLAUDE.md` points here.

## What this repository is

An interactive course that teaches AI from a first line of Python to your own small language model and a reproduced paper. It runs locally: `uv run everything-ai` starts the course in the browser. Lessons are [marimo](https://marimo.io) notebooks with exercises that are checked straight away. Every lesson exists in English (`en.py`) and Dutch (`nl.py`).

| Path | What is there |
|---|---|
| `lessons/<id>/en.py`, `nl.py` | the course's own lesson, one file per language, with identical code |
| `lessons/<id>/checks.py` | what is right in each exercise, the known mistakes, every check message in both languages, and `CASES` the tests run |
| `lessons/<id>/lesson.yaml` | title, goal, minutes, summary and "You can now" of the lesson |
| `course/route/` | the route: every lesson with its level, module, prerequisites (`requires`) and the ideas it introduces |
| `course/phrases.yaml` | fixed words the course uses in the same place everywhere |
| `src/everything_ai/` | the app: start page, lesson helper (`lesson.py`), checks (`check.py`), widgets, progress |
| `data/` | datasets the course hosts itself, such as `tiny-shakespeare.txt`; `data/README.md` says where each comes from |
| `work/` | **the learner's own work**, not in git: their copies of lessons, progress and settings, and their project code in `work/projects/` |

## Helping a learner

- **Find out where they are.** Their copy of a lesson is `work/lessons/<id>/<lang>.py`; what they did is in `work/progress/<id>.json` (each exercise's last state, whether it was ever correct, hints opened). Their language is `lang` in `work/settings.json`.
- **Answer in their language.** In Dutch, keep AI jargon in English as the course does (neural network, loss, layer, training), and use the established Dutch word for maths (afgeleide, residu).
- **Do not give the answer away.** Learning happens when the learner finds it. Explain the idea with another example, ask a question that points the way, or point to the lesson's hints, which open one at a time. Every lesson has the solution as the last hint; if the learner asks for the answer outright, that is their choice.
- **Stay within what they have learned.** The route says which lessons come before this one (`requires` in `course/route/lessen/*.yaml`) and what each lesson introduces. Do not lean on ideas or Python they have not met yet; if you need one, say that a later lesson covers it.
- **Read the check, not just the code.** The message the learner saw comes from `lessons/<id>/checks.py`. It looks at the outcome of their answer, not at how they wrote it, so a different correct answer counts as correct.
- **Do not edit the course to help.** Change nothing in `lessons/` or `course/`. Change the learner's copy in `work/` only when they ask you to. "Start over" on the start page puts a lesson back the way the course made it.
- **Something wrong in the course?** Suggest the "Report it on GitHub" link at the bottom of the lesson, with what happened.

