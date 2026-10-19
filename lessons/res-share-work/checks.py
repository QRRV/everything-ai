# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-share-work: checksums, the SHA-256 of every file of a shared folder in the form sha256sum
writes and checks; the sort of what goes into an open artifact, what you link to instead, and what stays out; and a
choice of the license for the code. The demo puts the learner's reproduction together as one folder that someone
else can rerun, work/projects/repro/share/: the kit, the logs and the report, the versions it ran with, a script that
runs the key figure again, a license, a README and the checksums."""
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import date
from importlib import metadata

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, Sort, _show_rounded


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _checks("repro-guided-report", "everything_ai_checks_repro_guided_report_for_share")
G, STEPS, number = R.G, R.STEPS, R.number
# Every module of the reproduction kit, the ones it imports first: the folder holds them all, so that it runs alone.
KIT = R.BUILDS_ON + (("repro", "results", "repro-guided-report", "result-rows"), ("repro", "report", "repro-guided-report", "report"))
projects.builds_on(*KIT)
SHARE = "share"   # the folder, in work/projects/repro/
# The logs the reproduction wrote, in work/projects/repro/, and the course's runs that stand in where the learner's are
# missing, as they did in the report; the comparison over five seeds always used the course's runs.
LOGS = {"runs.jsonl": G.RUNS_FILE, "ablation.jsonl": R.A.RUNS_FILE}
SEEDS_LOG = "seeds.jsonl"
# --- measured: seed 0 of the key figure again, 12,000 steps on one thread, with the last bit of the first starting weight
# changed (data/tools/measure_one_bit.py), against the course's seed 0 in data/course-grokking.jsonl ---
ONE_BIT = {"v": 8913, "v_before": 7943, "val_at": 0.3569, "val_before": 0.997}
# --- end of measured ---
REPORT = R.REPORT


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- exercise 1: checksums ---------------------------------------------------------------------------------------------

_C_HEAD = ("import hashlib\n\n\n"
           "def checksums(files):\n"
           "    # one line for each file, in the order of sorted(files): the SHA-256 of its text as 64 hex digits, two\n"
           "    # spaces, and its name, as sha256sum writes them; files maps each name to its text\n"
           "    lines = []\n")
_C_LOOP = ("    for name in sorted(files):\n"
           "        lines.append(hashlib.sha256(files[name].encode()).hexdigest() + \"  \" + name)\n")
_C_TAIL = "    return lines\n"
_C_START = _C_HEAD + "    # a line for each name, in sorted order\n    ...\n" + _C_TAIL
_C = _C_HEAD + _C_LOOP + _C_TAIL
_C_ORDER = _C.replace("    for name in sorted(files):\n", "    for name in files:\n")
_C_NAME = _C.replace("hashlib.sha256(files[name].encode())", "hashlib.sha256(name.encode())")
_C_SHORT = _C.replace(".hexdigest() + ", ".hexdigest()[:8] + ")
_C_ONE_SPACE = _C.replace('+ "  " +', '+ " " +')

_C_VISIBLE = ({"plan.py": "THRESHOLD = 0.99\n", "README.md": "# My reproduction\n"},)
_C_EN, _C_NL = "`{call}` gives `{got}`, and it should give `{want}`", "`{call}` geeft `{got}`, en dat moet `{want}` zijn"


def _show_lines(v) -> str:
    return repr(v) if isinstance(v, list) else _show_rounded(v, 4)


# --- exercise 2: what goes into the folder -----------------------------------------------------------------------------

_ITEMS = {
    "kit": {"en": "The code of your kit, also the modules of your GPT it imports", "nl": "De code van je kit, ook de modules van je GPT die hij importeert"},
    "paper": {"en": "The PDF of the grokking paper", "nl": "De pdf van de grokking-paper"},
    "versions": {"en": "The versions of Python, PyTorch and NumPy the runs used", "nl": "De versies van Python, PyTorch en NumPy waarmee de runs draaiden"},
    "token": {"en": "The file with the login token of your Hugging Face account", "nl": "Het bestand met het login-token van je account bij Hugging Face"},
    "logs": {"en": "Your experiment logs, with the seed and the accuracies of every run", "nl": "Je experimentlogboeken, met de seed en de accuracies van elke run"},
    "dataset": {"en": "A dataset of someone else's, whose license does not let you pass it on", "nl": "Een dataset van iemand anders, waarvan de licentie niet toestaat hem door te geven"},
    "report": {"en": "Your report, with its negative results", "nl": "Je verslag, met zijn negatieve resultaten"},
    "progress": {"en": "Your progress in the course, `work/progress/`", "nl": "Je voortgang in de course, `work/progress/`"},
    "rerun": {"en": "A script that runs the key figure again with your settings and seeds", "nl": "Een script dat de belangrijkste figuur opnieuw draait met je instellingen en seeds"},
}
_GROUPS = {
    "share": {"en": "In the folder", "nl": "In de map"},
    "link": {"en": "A link to it, not the file", "nl": "Een link ernaartoe, niet het bestand"},
    "out": {"en": "Stays out", "nl": "Blijft erbuiten"},
}
_SOLUTION = {"kit": "share", "paper": "link", "versions": "share", "token": "out", "logs": "share", "dataset": "link",
             "report": "share", "progress": "out", "rerun": "share"}

EXERCISES = {
    "checksums": FunctionBox(
        name="checksums",
        function="checksums",
        start=_C_START,
        solution=_C,
        visible=[_C_VISIBLE],
        hidden=[({},), ({"b.txt": "same\n", "a.txt": "same\n"},), ({"repro/plan.py": "DELAY = 10\n", "LICENSE": "MIT License\n"},)],
        show=_show_lines,
        first=[G._DOTS],
        mistakes=[
            Mistake("order", wrong=_C_ORDER,
                    en=_C_EN + ": the same lines, in the order of `files`. Then two people with the same files can get another list, and the list itself another checksum. Loop over `sorted(files)`.",
                    nl=_C_NL + ": dezelfde regels, in de volgorde van `files`. Dan kunnen twee mensen met dezelfde bestanden een andere lijst krijgen, en de lijst zelf een andere checksum. Loop over `sorted(files)`."),
            Mistake("name", wrong=_C_NAME,
                    en=_C_EN + ": you take the hash of the name, so a file whose text changed keeps its line. Take the hash of its text: `files[name].encode()`.",
                    nl=_C_NL + ": je neemt de hash van de naam, dus houdt een bestand waarvan de tekst veranderde zijn regel. Neem de hash van zijn tekst: `files[name].encode()`."),
            Mistake("one-space", wrong=_C_ONE_SPACE,
                    en="Your lines have one space between the hash and the name, where `sha256sum` writes two: the second space says that the file was read as text. Put two spaces between them, not one.",
                    nl="Je regels hebben één spatie tussen de hash en de naam, waar `sha256sum` er twee schrijft: de tweede spatie zegt dat het bestand als tekst gelezen is. Zet er twee spaties tussen, niet één."),
            Mistake("short", wrong=_C_SHORT,
                    en=_C_EN + ": your hashes are 8 hex digits, like the code version in your experiment log. To tell versions apart that is enough; to check that a file is the same, it takes all 64, and `sha256sum` wants them all.",
                    nl=_C_NL + ": je hashes zijn 8 hextekens, zoals de codeversie in je experimentlogboek. Om versies uit elkaar te houden is dat genoeg; om te controleren dat een bestand hetzelfde is, zijn alle 64 nodig, en `sha256sum` wil ze allemaal."),
        ],
        texts={
            "empty": {"en": "Replace the `...`: for each name in `sorted(files)`, the SHA-256 of its text, `.hexdigest()`, two spaces and the name, added to `lines`.",
                      "nl": "Vervang de `...`: voor elke naam in `sorted(files)` de SHA-256 van zijn tekst, `.hexdigest()`, twee spaties en de naam, toegevoegd aan `lines`."},
            "arguments": {"en": "The check calls `{call}`, with one dictionary of files, and the `def` line of your `checksums` has another number of names between its parentheses. Keep `(files)`.",
                          "nl": "De check roept `{call}` aan, met één dictionary met bestanden, en de `def`-regel van je `checksums` heeft een ander aantal namen tussen de haakjes. Houd `(files)`."},
            "wrong": {"en": _C_EN + ". Does each line hold the 64 hex digits of the SHA-256 of the file's text, two spaces and its name, in the order of `sorted(files)`, added to `lines` and not only printed?",
                      "nl": _C_NL + ". Heeft elke regel de 64 hextekens van de SHA-256 van de tekst van het bestand, twee spaties en zijn naam, in de volgorde van `sorted(files)`, toegevoegd aan `lines` en niet alleen geprint?"},
            "almost": {"en": "Your `checksums` gives the course's lines for the first example, but `{call}` gives `{got}`, and it should give `{want}`. Does it work for no files, for two files with the same text, and for a name with a folder in it?",
                       "nl": "Je `checksums` geeft de regels van de course voor het eerste voorbeeld, maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij zonder bestanden, voor twee bestanden met dezelfde tekst, en voor een naam met een map erin?"},
            "correct": {"en": "Your `checksums` gives every file a line that `sha256sum -c` can check: whoever gets your folder sees at once whether a file changed on the way.",
                        "nl": "Je `checksums` geeft elk bestand een regel die `sha256sum -c` kan controleren: wie je map krijgt, ziet meteen of er onderweg een bestand veranderde."},
        },
    ),
    "what-goes": Sort(
        name="what-goes",
        items=_ITEMS,
        groups=_GROUPS,
        solution=_SOLUTION,
        nudge={
            "kit": {"en": "Could someone run your figure again without it?", "nl": "Kan iemand je figuur zonder dit opnieuw draaien?"},
            "paper": {"en": "Whose is the PDF, and may you hand it out?", "nl": "Van wie is de pdf, en mag jij hem uitdelen?"},
            "versions": {"en": "What does the same seed give with another version of PyTorch?", "nl": "Wat geeft dezelfde seed met een andere versie van PyTorch?"},
            "token": {"en": "What could someone do with it?", "nl": "Wat kan iemand ermee?"},
            "logs": {"en": "How would someone else check your numbers without them?", "nl": "Hoe controleert iemand anders je getallen zonder ze?"},
            "dataset": {"en": "What does its license let you do, and what does the reader need?", "nl": "Wat laat zijn licentie je doen, en wat heeft de lezer nodig?"},
            "report": {"en": "Who reads the folder first, and what do they want to know?", "nl": "Wie leest de map als eerste, en wat wil die weten?"},
            "progress": {"en": "Does someone need it to run your reproduction?", "nl": "Heeft iemand het nodig om je reproductie te draaien?"},
            "rerun": {"en": "How does someone start, the first time?", "nl": "Hoe begint iemand, de eerste keer?"},
        },
        texts={
            "correct": {"en": "Each item is where it belongs. The folder holds all that is needed to run it again; what is not yours to hand out gets a link; secrets and what nobody needs stay out.",
                        "nl": "Elk item staat waar het hoort. De map bevat alles wat nodig is om het opnieuw te draaien; wat niet van jou is om uit te delen, krijgt een link; geheimen en wat niemand nodig heeft, blijven erbuiten."},
            "missing": {"en": "I cannot see your choices. Run the cell with the items again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de items opnieuw uit."},
            "empty": {"en": "Put each item in a group: in the folder, a link to it, or stays out.",
                      "nl": "Zet elk item in een groep: in de map, een link ernaartoe, of blijft erbuiten."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "\"{item}\" is not in its group.", "nl": "\"{item}\" staat niet in zijn groep."},
        },
    ),
}

# The lesson is done when these are correct. "license" is the choice question in the lesson.
REQUIRED = ["checksums", "what-goes", "license"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("checksums", _C_START, "empty", None),
    ("checksums", _C, "correct", None),
    ("checksums", _C_ORDER, "wrong", "order"),
    ("checksums", _C_NAME, "wrong", "name"),
    ("checksums", _C_SHORT, "wrong", "short"),
    ("checksums", _C_ONE_SPACE, "wrong", "one-space"),
    # other ways to write it that give the same lines
    ("checksums", _C_HEAD.replace("    lines = []\n", "") + "    return [hashlib.sha256(files[n].encode()).hexdigest() + \"  \" + n for n in sorted(files)]\n", "correct", None),
    ("checksums", _C.replace("hashlib.sha256(files[name].encode()).hexdigest() + \"  \" + name", "\"%s  %s\" % (hashlib.sha256(files[name].encode()).hexdigest(), name)"), "correct", None),
    ("checksums", _C.replace("sorted(files)", "sorted(files.keys())"), "correct", None),
]


# --- the demo: the folder ----------------------------------------------------------------------------------------------

def _course_checksums():
    names = {}
    exec(_C, names)
    return names["checksums"]


def checksums_of(code: str):
    """The learner's checksums from the text of their code box, or None when it does not run."""
    try:
        names = {}
        exec(code, names)
        return names.get("checksums")
    except Exception:
        return None


def versions() -> dict:
    """The versions the runs of this computer use."""
    return {"Python": platform.python_version(), "PyTorch": metadata.version("torch"), "NumPy": metadata.version("numpy")}


_RERUN = '''# Runs the key figure of this reproduction again: seeds {seeds} with SETTINGS, {steps} steps each, on one CPU thread,
# into rerun.jsonl. Then compare rerun.jsonl with runs.jsonl, run by run: README.md says what to expect.
import torch

from repro.seeds import run_seeds
from repro.settings import SETTINGS

torch.set_num_threads(1)
run_seeds(SETTINGS, {seeds_list}, {steps}, "rerun.jsonl")
'''

_README = '''# {title}

A reproduction of Power and others (2022), *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*,
arXiv:2201.02177, made with the Everything AI course.{found}

## What is here

{tree}

## Run it again

With Python {Python}, PyTorch {PyTorch} and NumPy {NumPy} (`pip install -r requirements.txt`), from this folder:

    python rerun.py

It runs the key figure again on one CPU thread, seeds {seeds}, {steps} steps each, into `rerun.jsonl`; on the course's
computer, one seed took about {minutes} minutes that way. With these versions on one CPU thread, the accuracies usually
match those in `runs.jsonl` that ran the same way, to its 4 decimals. A GPU, other threads, another CPU or other versions
round some sums differently, and that small difference grows while the model trains: during the jump, the accuracies
of two such runs can be far apart, and the step at which validation reaches 99% can move by a logged step.

The data needs no download: `repro/data.py` makes every equation of addition modulo 97, and the seed splits them into
training and validation data.

## Check the files

    sha256sum -c SHA256SUMS

## License

The code is under the MIT license, in `LICENSE`.{own}
{course_note}'''

_MIT = '''MIT License

Copyright (c) {year} {name}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


def _and(names: list) -> str:
    """`a`, `b` and `c`."""
    shown = [f"`{n}`" for n in names]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def folder_files(name: str) -> dict:
    """The files of the shared folder, by their path in it: every module of the kit (the learner's, or the course's
    where theirs is missing), the logs (the course's runs where the learner's are missing, and always for the comparison
    over seeds, as in the report), the report if the learner has one, the script that runs the key figure again, the
    versions, the license with the learner's name, and a README. Gives {"files": ..., "course": the modules that are the
    course's, "course_logs": the logs that are the course's runs, "missing": the report when the learner has none yet}."""
    projects.builds_on(*KIT)   # also when another lesson said what it builds on since
    name = name.strip() or "the author"
    src = projects.folder()
    files, course = {}, []
    for package, module, _, _ in KIT:
        mine = src / package / f"{module}.py"
        files[f"{package}/__init__.py"] = ""
        if mine.is_file():
            text = mine.read_text(encoding="utf-8")
            if projects._MARK in text:   # the course's header and fingerprint above the code: not part of the work
                text = text.split(projects._MARK, 1)[1].split("\n", 1)[1]
            files[f"{package}/{module}.py"] = text
        else:
            files[f"{package}/{module}.py"] = projects.course_code(package, module) + "\n"
            course.append(f"{package}/{module}.py")
    course_logs, own = [], []
    for log, course_file in LOGS.items():
        p = src / "repro" / log
        if p.is_file():
            files[log] = p.read_text(encoding="utf-8")
            own.append(log)
        else:
            files[log] = course_file.read_text(encoding="utf-8")
            course_logs.append(log)
    files[SEEDS_LOG] = R.C.RUNS_FILE.read_text(encoding="utf-8")
    course_logs.append(SEEDS_LOG)
    missing = []
    if (src / "repro" / REPORT).is_file():
        files[REPORT] = (src / "repro" / REPORT).read_text(encoding="utf-8")
        own.insert(0, REPORT)
    else:
        missing.append(REPORT)
    v = versions()
    seeds = ", ".join(str(s) for s in G.SEEDS)
    files["rerun.py"] = _RERUN.format(seeds=seeds, seeds_list=json.dumps(G.SEEDS), steps=STEPS)
    files["requirements.txt"] = "".join(f"{pkg}=={v[key].split('+')[0]}\n" for pkg, key in (("torch", "PyTorch"), ("numpy", "NumPy")))   # no +cpu or +cu130
    files["LICENSE"] = _MIT.format(year=date.today().year, name=name)
    tree = "\n".join(f"- `{p}`" for p in sorted(files) if not p.endswith("__init__.py")) + "\n- `README.md`, this file\n- `SHA256SUMS`, the checksums of all the others"
    note = ("\nThe modules " + ", ".join(f"`{p}`" for p in course) + " are the course's own versions, under MIT-0.\n") if course else ""
    note += (f"\nThe runs in {_and(course_logs)} are the course's own, by Everything AI, under CC BY 4.0"
             + (", as the report uses them" if REPORT in files else "") + ".\n")
    found = " What it found is in `report.md`." if REPORT in files else ""
    mine = (f" {_and(own)} {'is' if len(own) == 1 else 'are'} under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/): "
            f"share and change {'it' if len(own) == 1 else 'them'}, and name {name}.") if own else ""
    files["README.md"] = _README.format(title=R.TITLE, tree=tree, seeds=seeds, steps=f"{STEPS:,}", found=found, own=mine,
                                        minutes=round(STEPS * G.SECONDS_PER_STEP / 60), name=name, course_note=note, **v)
    return {"files": files, "course": course, "course_logs": course_logs, "missing": missing}


def build(name: str, checksums_of=None) -> dict:
    """Writes the shared folder, work/projects/repro/share/, anew: the files of folder_files and SHA256SUMS, made with
    checksums_of (the learner's checksums, or the course's). Gives back where, the files, and what folder_files says."""
    out = folder_files(name)
    files = dict(out["files"])
    try:
        sums = checksums_of(files)
        assert isinstance(sums, list) and sums == _course_checksums()(files)
        mine = True
    except Exception:
        sums, mine = _course_checksums()(files), False
    files["SHA256SUMS"] = "\n".join(sums) + "\n"
    target = projects.folder() / "repro" / SHARE
    if target.exists():
        import shutil
        shutil.rmtree(target)
    for path, text in files.items():
        p = target / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return {**out, "files": files, "where": f"work/projects/repro/{SHARE}/", "mine": mine}
