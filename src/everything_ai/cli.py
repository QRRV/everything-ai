# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""`uv run everything-ai`: start the course and open it in your browser.

The course is marimo's own editor, serving your copies of the lessons (work/), with the
course's start page in front of it (server.py). Everything runs on your computer; the
browser only shows it. Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import sys

from everything_ai import paths, progress


def _prepare() -> None:
    paths.WORK_LESSONS.mkdir(parents=True, exist_ok=True)
    paths.PROGRESS.mkdir(parents=True, exist_ok=True)
    paths.SYSTEM.unlink(missing_ok=True)      # look at the computer again at every start
    # marimo keeps its own settings (such as the theme) in the first .marimo.toml it finds from the
    # folder it starts in. One in the course folder keeps the course's settings out of your own marimo setup.
    config = paths.ROOT / ".marimo.toml"
    if not config.exists():
        theme = progress.settings()["theme"]
        config.write_text(f'[display]\ntheme = "{theme}"\n', encoding="utf-8")


def main() -> None:
    _prepare()
    os.chdir(paths.ROOT)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    args = [
        "edit", str(paths.WORK),
        "--skip-update-check",      # marimo updates come with the course, never on their own
        "--no-sandbox",
        "--session-ttl", "600",     # a lesson keeps running 10 minutes after you close its tab
    ]
    if os.environ.get("EAI_PORT"):
        args += ["--port", os.environ["EAI_PORT"]]
    if os.environ.get("EAI_HEADLESS"):
        args += ["--headless"]
    if os.environ.get("EAI_TOKEN"):
        args += ["--token-password", os.environ["EAI_TOKEN"]]
    print("Everything AI is starting. Your browser opens by itself; if not, open the link below.")
    print("Stop the course with Ctrl+C.\n")
    from marimo._cli.cli import main as marimo_main
    sys.exit(marimo_main(args=args, prog_name="everything-ai"))
