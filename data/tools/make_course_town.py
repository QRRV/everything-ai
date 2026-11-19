# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Make the made-up town of the lesson rai-linkage-attacks: three tables of people who do not exist.

    python data/tools/make_course_town.py

Writes, next to this folder's parent:

- data/course-town-people.csv.gz: the whole town, 30,000 adults aged 18 to 90 on 1 January 2024, as only the course
  knows it: `name`, `zip` (one of three ZIP codes), `birth` (a date, YYYY-MM-DD), `sex` (`female` or `male`),
  `voter` (`True` for the 65% on the voter list) and `visit` (the id of their visit in the released table, or empty);
- data/course-town-voters.csv.gz: the voter list anyone can get: `name`, `zip`, `birth` and `sex` of every voter,
  sorted by name;
- data/course-town-visits.csv.gz: the released table of a made-up health insurer: `visit` (V0001 to V1000), `zip`,
  `birth`, `sex` and `diagnosis` of 1,000 people of the town, without names, sorted by visit.

Everything is drawn with numpy's default generator and seed 2026: ages with fewer people over 65, a birth date
spread evenly over the days of the birth year, ZIP codes with 40%, 35% and 25% of the town, sex half and half, the
voter list as a random 65% (about the share of Cambridge's adults on the voter list Sweeney used, after Barth-Jones
2012), the visits as 1,000 people drawn at random. Names are a first name, an initial and a surname from short lists,
each name used once; the first names are ones given to people of any sex. The files are written without a time in
their gzip header, so the same numpy gives the same bytes."""
import csv
import datetime as dt
import gzip
import io
from pathlib import Path

import numpy as np

SEED = 2026
PEOPLE = 30_000
VOTERS = 0.65
VISITS = 1_000
ZIPS = {"30101": 0.40, "30102": 0.35, "30103": 0.25}
FIRST = ["Alex", "Ari", "Avery", "Blair", "Cameron", "Casey", "Charlie", "Dakota", "Dani", "Drew", "Eden", "Eli",
         "Elliot", "Emery", "Finley", "Frankie", "Harper", "Hayden", "Jamie", "Jesse", "Jo", "Jordan", "Jules", "Kai",
         "Kendall", "Kim", "Kit", "Lee", "Logan", "Lou", "Marley", "Max", "Mika", "Morgan", "Nico", "Noa", "Parker",
         "Pat", "Peyton", "Reese", "Remy", "Riley", "Robin", "Rowan", "Sage", "Sam", "Sasha", "Shay", "Sky", "Taylor",
         "Toni", "Wren"]
START = ["Ash", "Birch", "Brook", "Clay", "Cole", "Elm", "Fern", "Glen", "Hart", "Hazel", "Holm", "Kings", "Lang",
         "Lark", "Mill", "Moss", "North", "Oak", "Pine", "Reed", "Rose", "Rush", "Stone", "Thorn", "West", "Wood"]
END = ["by", "dale", "field", "ford", "ham", "hurst", "ley", "more", "stead", "ton", "well", "wick"]
DIAGNOSES = {"flu": 0.22, "back pain": 0.16, "asthma": 0.12, "high blood pressure": 0.12, "allergy": 0.1,
             "migraine": 0.1, "broken arm": 0.08, "diabetes": 0.1}


def town():
    rng = np.random.default_rng(SEED)
    ages = np.arange(18, 91)
    weight = np.where(ages < 30, 1.3, np.where(ages < 65, 1.0, np.where(ages < 80, 0.7, 0.3)))
    age = rng.choice(ages, size=PEOPLE, p=weight / weight.sum())
    births = []
    for a in age:
        year = 2023 - int(a)
        first = dt.date(year, 1, 1)
        days = (dt.date(year + 1, 1, 1) - first).days
        births.append((first + dt.timedelta(days=int(rng.integers(days)))).isoformat())
    zips = rng.choice(list(ZIPS), size=PEOPLE, p=list(ZIPS.values()))
    sex = rng.choice(["female", "male"], size=PEOPLE)
    surnames = [s + e for s in START for e in END if s[-1] != e[0]]   # no Elmmore
    initials = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    every = len(FIRST) * len(initials) * len(surnames)
    picks = rng.choice(every, size=PEOPLE, replace=False)
    names = []
    for p in picks:
        f, rest = divmod(int(p), len(initials) * len(surnames))
        i, s = divmod(rest, len(surnames))
        names.append(f"{FIRST[f]} {initials[i]}. {surnames[s]}")
    voter = rng.random(PEOPLE) < VOTERS
    visit = [""] * PEOPLE
    for number, who in enumerate(rng.choice(PEOPLE, size=VISITS, replace=False), start=1):
        visit[int(who)] = f"V{number:04d}"
    diagnosis = rng.choice(list(DIAGNOSES), size=PEOPLE, p=list(DIAGNOSES.values()))
    return [{"name": n, "zip": z, "birth": b, "sex": s, "voter": bool(v), "visit": w, "diagnosis": d}
            for n, z, b, s, v, w, d in zip(names, zips, births, sex, voter, visit, diagnosis)]


def write(path: Path, header: list, rows: list) -> None:
    text = io.StringIO()
    out = csv.writer(text, lineterminator="\n")
    out.writerow(header)
    out.writerows(rows)
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as g:
        g.write(text.getvalue().encode("utf-8"))


def main() -> None:
    data = Path(__file__).resolve().parents[1]
    people = town()
    write(data / "course-town-people.csv.gz", ["name", "zip", "birth", "sex", "voter", "visit"],
          [[p["name"], p["zip"], p["birth"], p["sex"], p["voter"], p["visit"]]
           for p in sorted(people, key=lambda p: (p["zip"], p["name"]))])
    write(data / "course-town-voters.csv.gz", ["name", "zip", "birth", "sex"],
          [[p["name"], p["zip"], p["birth"], p["sex"]] for p in sorted(people, key=lambda p: p["name"]) if p["voter"]])
    write(data / "course-town-visits.csv.gz", ["visit", "zip", "birth", "sex", "diagnosis"],
          [[p["visit"], p["zip"], p["birth"], p["sex"], p["diagnosis"]]
           for p in sorted((p for p in people if p["visit"]), key=lambda p: p["visit"])])


if __name__ == "__main__":
    main()
