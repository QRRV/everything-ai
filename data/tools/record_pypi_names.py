# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Writes data/course-pypi-names.json: for each package name that the lesson rai-package-squatting needs, whether PyPI
had a project of that name on the day of the run. The names are the ones Qwen3-0.6B gave in the course's run (the
part rai-package-squatting of data/course-security-qwen.json, read with the lesson's names_in) and the lesson's own
examples (NAMES in its checks.py), each normalized as PyPI compares names. For each it asks PyPI's JSON API,
GET https://pypi.org/pypi/<name>/json: an answer 200 means PyPI had the project, 404 that it had none; anything else
stops the run. It only reads; nothing is downloaded or installed. Run from the course folder:

    uv run python data/tools/record_pypi_names.py
"""
import datetime
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "src")
from everything_ai import paths  # noqa: E402

OUT = paths.ROOT / "data" / "course-pypi-names.json"
LESSON = "rai-package-squatting"
API = "https://pypi.org/pypi/{name}/json"
VALID = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")   # a normalized name: letters, digits and single hyphens inside


def _checks():
    spec = importlib.util.spec_from_file_location("checks", paths.LESSONS / LESSON / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def on_pypi(name: str) -> bool:
    if not VALID.match(name):
        raise ValueError(f"not a normalized package name: {name!r}")
    try:
        with urllib.request.urlopen(API.format(name=name), timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def main() -> None:
    C = _checks()
    names = sorted(set(C.recorded_names()) | {C.normalize(n) for n in C.NAMES})
    out = {"date": datetime.date.today().isoformat(), "source": API, "names": {}}
    for name in names:
        out["names"][name] = on_pypi(name)
        print(name, out["names"][name], flush=True)
        time.sleep(0.5)   # gently: one request at a time
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
