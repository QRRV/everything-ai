# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Where everything lives. The course runs from a clone of the repository (`uv run everything-ai`)."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]                 # the repository: src/everything_ai -> src -> root
LESSONS = ROOT / "lessons"                # the course's lessons, one folder per lesson id
COURSE = ROOT / "course"                  # route data and fixed phrases
WEB = PACKAGE / "web"                     # start page and lesson styles

# The learner's own work: copies of lessons, progress and settings. Not in git.
# EAI_WORK lets tests use a temporary folder. The app itself needs work/ inside the repository:
# marimo looks for the course's settings ([tool.marimo] in pyproject.toml) next to the notebook.
WORK = Path(os.environ.get("EAI_WORK", ROOT / "work")).resolve()
WORK_LESSONS = WORK / "lessons"
PROGRESS = WORK / "progress"
SETTINGS = WORK / "settings.json"
SYSTEM = WORK / "system.json"
