# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-experiment-log: log_run, which adds one run as a JSON line to the end of a log file, and
find_runs, which picks the runs with some settings out of a log. And the course's own runs of res-hypothesis-control
and res-seeds-variance, written as a log."""
import hashlib
import importlib.util
import json
import re
import tempfile
from pathlib import Path

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, ScenarioBox, _as_code, run_code


def _hypothesis_control():
    """The checks of res-hypothesis-control, whose small GPT made the runs in this lesson's log."""
    spec = importlib.util.spec_from_file_location("everything_ai_checks_hypothesis_control_for_log",
                                                  paths.LESSONS / "res-hypothesis-control" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _hypothesis_control()


def version_of(code: str) -> str:
    """The code version of a text: the first 8 characters of its SHA-256 hash, as in the worked example."""
    return hashlib.sha256(code.encode()).hexdigest()[:8]


def project_version() -> tuple:
    """The code version of the learner's GPT project: all .py files in work/projects/llm/ and work/projects/nn/, the
    code a run of their GPT trains with (its Trainer is in nn), in the order of their names; and how many files that
    is. (None, 0) while llm/ has none yet."""
    files = []
    for package in ("llm", "nn"):
        folder = projects.folder() / package
        files += sorted(folder.glob("*.py")) if folder.is_dir() else []
    if not any(f.parent.name == "llm" for f in files):
        return None, 0
    return version_of("".join(f"{f.parent.name}/{f.name}\n" + f.read_text(encoding="utf-8") for f in files)), len(files)


def scratch_path(name: str = "runs.jsonl") -> str:
    """A path to a file that does not exist yet, in a new folder in the computer's temporary folder: the worked
    example writes its log there, never in the learner's project."""
    return str(Path(tempfile.mkdtemp(prefix="everything-ai-")) / name)


class _Log(list):
    """The course's log: a list of runs that a check message calls `log`, instead of writing out all 16 runs. The
    learner's code gets a plain list."""

    def __repr__(self):
        return "log"

    def __deepcopy__(self, memo):
        return json.loads(json.dumps(list(self)))


# The course's runs of the small GPT of res-hypothesis-control: the validation loss after 6 epochs, on a CPU with
# 1 thread. Seeds 0 to 2 of each variant come from res-hypothesis-control, seeds 3 and 4 of the control run and of
# warmup=0 from res-seeds-variance; the same runs as the REFERENCE numbers of those two lessons.
_LOSSES = {
    "control": [2.3902, 2.3924, 2.4174, 2.3938, 2.4012],
    "lr=0.003": [2.0827, 2.0987, 2.0850],
    "warmup=0": [2.3866, 2.3841, 2.4115, 2.3916, 2.3962],
    "layers=4": [2.3335, 2.3245, 2.3374],
}
_CHANGES = {"control": {}, "lr=0.003": {"lr": 0.003}, "warmup=0": {"warmup": 0}, "layers=4": {"layers": 4}}
SETTINGS = {k: v for k, v in H.CONTROL.items() if k != "seed"}   # the seed has its own place in a line
# All runs ran with the same code, the course's run of res-hypothesis-control, so they share one code version: the
# version_of its checks.py when this log was made. Fixed here, as a line of a log is: an edit of that file later
# must not change the version of runs that were already made.
CODE = "4f2af156"


def _course_log() -> list:
    """The runs in the order the course made them: seeds 0 to 2 of all four, then seeds 3 and 4 of the control run
    and of warmup=0."""
    out = []
    for seed in range(5):
        for name in ("control", "lr=0.003", "warmup=0", "layers=4"):
            if seed < len(_LOSSES[name]):
                out.append({"code": CODE, "settings": {**SETTINGS, **_CHANGES[name]}, "seed": seed, "val_loss": _LOSSES[name][seed]})
    return out


LOG = _Log(_course_log())

# The questions of the demo: which runs, as the settings the learner's find_runs gets.
QUESTIONS = {
    "control": {"lr": 0.001, "warmup": 100, "layers": 2},
    "lr=0.003": {"lr": 0.003},
    "warmup=0": {"warmup": 0},
    "layers=4": {"layers": 4},
}


def as_code(value) -> str:
    """A dictionary the way the lessons write it in code: `{"warmup": 0}`."""
    return _as_code(value)


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


def runs_table(runs: list, lang: str) -> str:
    """The runs as a Markdown table: seed, validation loss, code version."""
    head = {"en": "| seed | validation loss | code version |\n|---|---|---|", "nl": "| seed | validation loss | codeversie |\n|---|---|---|"}[lang]
    rows = [f"| {r['seed']} | {number(r['val_loss'], 4, lang)} | `{r['code']}` |" for r in runs]
    return head + "\n" + "\n".join(rows)


_PARTS_COLOURS = {"code": "var(--ea-ink)", "settings": "var(--ea-ink)", "seed": "var(--ea-ink)", "val_loss": "var(--ea-loss)"}


def line_picture(labels: dict) -> str:
    """One line of the log, with each of its four parts named underneath: the result is a loss, so purple; the
    rest is ink. HTML, so the line wraps in a narrow window."""
    import html
    run = {"code": CODE, "settings": {"lr": 0.001, "warmup": 100, "layers": 2, "...": "..."}, "seed": 0, "val_loss": 2.3902}
    parts = []
    keys = list(run)
    for i, key in enumerate(keys):
        shown = json.dumps(run[key]).replace('"...": "..."', "...")
        text_ = ("{" if i == 0 else "") + f"{json.dumps(key)}: {shown}" + ("" if i == len(keys) - 1 else ",")
        close = '<span style="color:var(--ea-ink)">}</span>' if i == len(keys) - 1 else ""   # the brace of the whole line: ink
        parts.append(f'<span style="display:inline-block;margin:0 0.3rem 0.4rem 0;vertical-align:top">'
                     f'<code style="color:{_PARTS_COLOURS[key]};background:var(--ea-inset);padding:0.15rem 0.3rem;border-radius:4px">'
                     f'{html.escape(text_)}{close}</code><br>'
                     f'<span style="font-size:0.8rem;color:var(--ea-ink-2)">{html.escape(labels[key])}</span></span>')
    body = "".join(parts)
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(labels["alt"])}" '
            f'style="margin:0.25rem 0 0.75rem;max-width:640px;font-family:var(--ea-font-ui);font-size:0.875rem;line-height:1.5">'
            f'<div style="font-size:1rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.5rem">{html.escape(labels["title"])}</div>'
            f'<div>{body}</div>'
            f'<div style="font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.25rem">{html.escape(labels["under"])}</div></figure>')


# --- exercise 1: log_run -----------------------------------------------------------------------------------------------------

_HEAD = "import json\n\n\ndef log_run(path, settings, seed, val_loss, version):\n"
_START = (_HEAD + "    # the record of one run, with the keys \"code\", \"settings\", \"seed\" and \"val_loss\"\n    run = ...\n"
          "    # one JSON line at the end of the file at path\n    ...\n")


def _l(mode: str = '"a"', line: str = 'json.dumps(run) + "\\n"', run: str = '{"code": version, "settings": settings, "seed": seed, "val_loss": val_loss}') -> str:
    return (_HEAD + f"    run = {run}\n    with open(path, {mode}) as f:\n        f.write({line})\n")


_NEW = ('import json, os, tempfile\n# a new file in a new temporary folder\nfolder = tempfile.TemporaryDirectory()\n'
        'path = os.path.join(folder.name, "runs.jsonl")\n')
_READ = "[json.loads(line) for line in open(path)]"
_SCENARIOS = [
    _NEW + 'log_run(path, {"lr": 0.001, "warmup": 100}, 0, 2.3902, "4f9a1c2e")\n'
           'log_run(path, {"lr": 0.001, "warmup": 0}, 0, 2.3866, "4f9a1c2e")\n' + _READ,
    _NEW + 'with open(path, "w") as f:\n    f.write(\'{"code": "0b7d55e1", "settings": {"lr": 0.003}, "seed": 2, "val_loss": 2.085}\\n\')\n'
           'log_run(path, {"lr": 0.001}, 1, 2.3924, "4f9a1c2e")\n' + _READ,
    _NEW + 'log_run(path, {"layers": 4, "d": 64}, 3, 2.3374, "b81d07a3")\n' + _READ,
]
_SCENARIO = "With the lines of the first example after your code, the log holds `{got}`, and it should hold `{want}`"
_SCENARIO_NL = "Met de regels van het eerste voorbeeld na je code bevat het logboek `{got}`, en dat moet `{want}` zijn"
_KEYS = {"code", "settings", "seed", "val_loss"}


def _other_keys(got, text) -> bool:
    return isinstance(got, list) and any(isinstance(r, dict) and set(r) != _KEYS for r in got)


_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_INDENT = Mistake("indent", matches=lambda got, text: re.search(r"\bindent\s*=", text) is not None,
                  en="`indent` spreads one run over several lines of the file, and a log has one run per line. Leave `indent` out: `json.dumps(run)`.",
                  nl="`indent` spreidt één run over meerdere regels van het bestand, en een logboek heeft één run per regel. Laat `indent` weg: `json.dumps(run)`.")


def _json_error(e, what: str) -> bool:
    """Reading the log back stopped at a line: "extra", two runs on one line; "empty", an empty line; "quotes", a
    dictionary the way Python shows it, with single quotes; "other", any other line that is not JSON."""
    if not isinstance(e, json.JSONDecodeError):
        return False
    two = e.msg == "Extra data" and e.doc[e.pos:].lstrip().startswith("{")   # a second run right after the first
    kind = ("extra" if two else "empty" if not e.doc.strip()
            else "quotes" if e.msg.startswith("Expecting property name enclosed in double quotes") else "other")
    return kind == what


def _in_log_run(e) -> bool:
    """The error happened inside the learner's log_run, not in the check's own lines after it."""
    name, tb = None, e.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<your code>":
            name = tb.tb_frame.f_code.co_name
        tb = tb.tb_next
    return name == "log_run"


EXERCISES = {
    "log-run": ScenarioBox(
        name="log-run",
        start=_START,
        solution=_l(),
        scenarios=_SCENARIOS,
        first=[_DOTS, _INDENT],
        mistakes=[
            Mistake("no-newline", error=lambda e: _json_error(e, "extra"),
                    en="Reading your log back stops: two runs stand on one line, because each line needs a newline at its end. Write `json.dumps(run) + \"\\n\"`.",
                    nl="Je logboek teruglezen stopt: twee runs staan op één regel, omdat elke regel aan het eind een regeleinde nodig heeft. Schrijf `json.dumps(run) + \"\\n\"`."),
            Mistake("not-json", error=lambda e: _json_error(e, "quotes"),
                    en="Reading your log back stops: a line of it is not JSON. It has single quotes, the way Python shows a dictionary, and JSON needs double quotes. Turn the dictionary into a line with `json.dumps(run)`.",
                    nl="Je logboek teruglezen stopt: een regel ervan is geen JSON. Er staan enkele aanhalingstekens in, zoals Python een dictionary laat zien, en JSON heeft dubbele nodig. Maak van de dictionary een regel met `json.dumps(run)`."),
            Mistake("empty-line", error=lambda e: _json_error(e, "empty"),
                    en="Reading your log back stops: your log has an empty line, and an empty line is not JSON. Write each run as `json.dumps(run) + \"\\n\"`: one newline, at the end of its line.",
                    nl="Je logboek teruglezen stopt: er staat een lege regel in, en een lege regel is geen JSON. Schrijf elke run als `json.dumps(run) + \"\\n\"`: één regeleinde, aan het eind van zijn regel."),
            Mistake("other-json", error=lambda e: _json_error(e, "other"),
                    en="Reading your log back stops: a line of it is not JSON. Turn the dictionary into a line with `json.dumps(run)`, and add nothing to it but `\"\\n\"`.",
                    nl="Je logboek teruglezen stopt: een regel ervan is geen JSON. Maak van de dictionary een regel met `json.dumps(run)`, en zet er niets bij dan `\"\\n\"`."),
            Mistake("read-mode", error=lambda e: isinstance(e, FileNotFoundError) and _in_log_run(e),
                    en="Your `open` needs the file to be there already, and a new log does not exist yet. Open it with `\"a\"`: that adds to the end, and makes the file when it is not there yet.",
                    nl="Je `open` heeft het bestand al nodig, en een nieuw logboek bestaat nog niet. Open het met `\"a\"`: dat voegt toe aan het eind, en maakt het bestand als het er nog niet is."),
            Mistake("loads", error=lambda e: isinstance(e, TypeError) and "JSON object must be str" in str(e),
                    en="`json.loads` turns a line of text into a dictionary, and you give it the dictionary. To turn the dictionary into a line, use `json.dumps(run)`.",
                    nl="`json.loads` maakt van een regel tekst een dictionary, en jij geeft hem de dictionary. Om van de dictionary een regel te maken, gebruik je `json.dumps(run)`."),
            Mistake("arguments", error=lambda e: isinstance(e, TypeError) and re.search(r"log_run\(\) (missing|takes)", str(e)) is not None,
                    en="The check calls `log_run` with five values, and the `def` line of your function has another number of names between its parentheses. Keep `(path, settings, seed, val_loss, version)`.",
                    nl="De check roept `log_run` aan met vijf waarden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, settings, seed, val_loss, version)`."),
            Mistake("dict-write", error=lambda e: isinstance(e, TypeError) and "write() argument must be str" in str(e),
                    en="`write` takes text, and you give it the dictionary itself. Turn it into one line of text first: `f.write(json.dumps(run) + \"\\n\")`.",
                    nl="`write` neemt tekst, en jij geeft hem de dictionary zelf. Maak er eerst één regel tekst van: `f.write(json.dumps(run) + \"\\n\")`."),
            Mistake("overwrite", wrong=_l(mode='"w"'),
                    en=_SCENARIO + ": only the last run is left, because `\"w\"` empties the file each time you open it. Open it with `\"a\"`, which adds to the end.",
                    nl=_SCENARIO_NL + ": alleen de laatste run is over, omdat `\"w\"` het bestand leegmaakt elke keer dat je het opent. Open het met `\"a\"`, dat aan het eind toevoegt."),
            Mistake("keys", matches=_other_keys,
                    en=_SCENARIO + ". Each line has exactly the four keys `\"code\"`, `\"settings\"`, `\"seed\"` and `\"val_loss\"`, with the version, the settings, the seed and the validation loss as their values.",
                    nl=_SCENARIO_NL + ". Elke regel heeft precies de vier keys `\"code\"`, `\"settings\"`, `\"seed\"` en `\"val_loss\"`, met de versie, de instellingen, de seed en de validation loss als hun waarden."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the two `...`: the first with a dictionary with the keys `\"code\"`, `\"settings\"`, `\"seed\"` and `\"val_loss\"`; the second with the lines that add it to the end of the file as one JSON line.",
                      "nl": "Vervang de twee `...`: de eerste door een dictionary met de keys `\"code\"`, `\"settings\"`, `\"seed\"` en `\"val_loss\"`; de tweede door de regels die hem als één JSON-regel aan het eind van het bestand toevoegen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "wrong": {"en": "With these lines after your code:\n\n```python\n{scenario}\n```\n\nthe log holds `{got}`, and it should hold `{want}`. Does each run become one line, `json.dumps(run) + \"\\n\"`, added to the end of the file with `\"a\"`?",
                      "nl": "Met deze regels na je code:\n\n```python\n{scenario}\n```\n\nbevat het logboek `{got}`, en dat moet `{want}` zijn. Wordt elke run één regel, `json.dumps(run) + \"\\n\"`, die met `\"a\"` aan het eind van het bestand komt?"},
            "almost": {"en": "Right for the first example. But with these lines after your code:\n\n```python\n{scenario}\n```\n\nthe log holds `{got}`, and it should hold `{want}`. Does a line that was already in the file stay there, and does each line hold the values your function gets?",
                       "nl": "Goed voor het eerste voorbeeld. Maar met deze regels na je code:\n\n```python\n{scenario}\n```\n\nbevat het logboek `{got}`, en dat moet `{want}` zijn. Blijft een regel die al in het bestand stond daar staan, en bevat elke regel de waarden die je functie krijgt?"},
            "correct": {"en": "Your function is right in all {n} cases: each run becomes one JSON line at the end of the log, and the lines that were there stay.",
                        "nl": "Je functie klopt in alle {n} gevallen: elke run wordt één JSON-regel aan het eind van het logboek, en de regels die er al stonden blijven staan."},
        },
    ),
}


# --- exercise 2: find_runs ---------------------------------------------------------------------------------------------------

_FHEAD = "def find_runs(log, wanted):\n"
_FSTART = (_FHEAD + "    found = []\n    for run in log:\n        # keep the run when its settings have the value of wanted for every key of wanted\n"
           "        ...\n    return found\n")


def _f(test: str = "run[\"settings\"][key] != wanted[key]", keep: str = "found.append(run)", start: str = "True",
       flip: str = "False") -> str:
    return (_FHEAD + "    found = []\n    for run in log:\n        same = " + start + "\n        for key in wanted:\n"
            f"            if {test}:\n                same = {flip}\n        if same:\n            {keep}\n    return found\n")


_ANY = (_FHEAD + "    found = []\n    for run in log:\n        for key in wanted:\n            if run[\"settings\"][key] == wanted[key]:\n"
        "                found.append(run)\n                break\n    return found\n")
_FIRST = (_FHEAD + "    for run in log:\n        if all(run[\"settings\"][key] == wanted[key] for key in wanted):\n            return [run]\n    return []\n")
# `same = True` once, before the loop over the runs, instead of at the start of each run
_ONCE = (_FHEAD + "    found = []\n    same = True\n    for run in log:\n        for key in wanted:\n            if run[\"settings\"][key] != wanted[key]:\n"
         "                same = False\n        if same:\n            found.append(run)\n    return found\n")


_LISTED = 5   # up to this many runs, a message names their validation losses; more runs, only how many


def _show_runs(v):
    """Runs the way the lessons write them in a check message: how many, and up to _LISTED of them their validation
    losses, since every run of the course's log ended at another one. A longer list only by its number: the right
    answers hold at most 5 runs, except all 16 of the log."""
    if isinstance(v, list) and all(isinstance(r, dict) and "val_loss" in r for r in v):
        if not v:
            return {"en": "no runs, `[]`", "nl": "geen runs, `[]`"}
        one = len(v) == 1
        if len(v) > _LISTED:
            return {"en": f"{len(v)} runs", "nl": f"{len(v)} runs"}
        losses = [r["val_loss"] for r in v]
        en = ", ".join(number(x, 4, "en") for x in losses[:-1]) + (" and " if len(losses) > 1 else "") + number(losses[-1], 4, "en")
        nl = "; ".join(number(x, 4, "nl") for x in losses[:-1]) + (" en " if len(losses) > 1 else "") + number(losses[-1], 4, "nl")
        return {"en": f"{len(v)} run{'' if one else 's'} (validation loss {en})",
                "nl": f"{len(v)} run{'' if one else 's'} (validation loss {nl})"}
    if isinstance(v, list) and v and all(isinstance(r, dict) and "val_loss" not in r for r in v):
        return {"en": f"a list of {len(v)} dictionaries without a `\"val_loss\"`", "nl": f"een lijst van {len(v)} dictionaries zonder `\"val_loss\"`"}
    return {"en": f"`{v!r}`", "nl": f"`{v!r}`"}


EXERCISES["find-runs"] = FunctionBox(
    name="find-runs",
    function="find_runs",
    start=_FSTART,
    solution=_f(),
    visible=[(LOG, {"warmup": 0})],
    hidden=[(LOG, {"lr": 0.003}), (LOG, {"lr": 0.001, "warmup": 0}), (LOG, {"layers": 3}), (LOG, {}),
            (LOG, {"warmup": 0, "layers": 4}), (LOG, {"lr": 0.001, "warmup": 100, "layers": 2})],
    show=_show_runs,
    first=[_DOTS],
    mistakes=[
        Mistake("exact", wrong=_FHEAD + "    found = []\n    for run in log:\n        if run[\"settings\"] == wanted:\n            found.append(run)\n    return found\n",
                en="`{call}` gives {got}, and it should give {want}. You keep a run only when its settings are exactly `wanted`, and `wanted` holds only the settings you ask about. Compare each key of `wanted` on its own: `run[\"settings\"][key] == wanted[key]`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je houdt een run alleen als zijn instellingen precies `wanted` zijn, en `wanted` bevat alleen de instellingen waar je naar vraagt. Vergelijk elke key van `wanted` apart: `run[\"settings\"][key] == wanted[key]`."),
        Mistake("any", wrong=_ANY,
                en="`{call}` gives {got}, and it should give {want}. You keep a run when one key of `wanted` has its value, and a run must match every key. Keep it only when no key of `wanted` has another value.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je houdt een run als één key van `wanted` zijn waarde heeft, en een run moet bij elke key kloppen. Houd hem alleen als geen enkele key van `wanted` een andere waarde heeft."),
        Mistake("first-only", wrong=_FIRST,
                en="`{call}` gives {got}, and it should give {want}. Your function stops at the first run that fits; it should go through the whole log and give back every run that fits.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie stopt bij de eerste run die past; hij moet het hele logboek doorlopen en elke run teruggeven die past."),
        Mistake("same-once", wrong=_ONCE,
                en="`{call}` gives {got}, and it should give {want}. `same = True` stands before the loop over the runs, so once one run does not fit, `same` stays `False` for every run after it. Start each run with `same = True`, inside `for run in log:`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. `same = True` staat vóór de loop over de runs, dus zodra één run niet past, blijft `same` `False` voor elke run daarna. Begin elke run met `same = True`, binnen `for run in log:`."),
        Mistake("loss-only", wrong=_f(keep="found.append(run[\"val_loss\"])"),
                en="`{call}` gives {got}, and it should give {want}. You give back only the validation loss of each run; give back the whole run, with its settings and its seed too: `found.append(run)`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je geeft alleen de validation loss van elke run terug; geef de hele run terug, ook met zijn instellingen en zijn seed: `found.append(run)`."),
        Mistake("settings-only", wrong=_f(keep="found.append(run[\"settings\"])"),
                en="`{call}` gives {got}, and it should give {want}. You give back only the settings of each run; give back the whole run, with its seed and its validation loss: `found.append(run)`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je geeft alleen de instellingen van elke run terug; geef de hele run terug, met zijn seed en zijn validation loss: `found.append(run)`."),
        Mistake("key-only", wrong=_f(test="key not in run[\"settings\"]"),
                en="`{call}` gives {got}, and it should give {want}. You check that each key of `wanted` is in the settings, and every run has them all. Check its value: `run[\"settings\"][key] != wanted[key]`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je kijkt of elke key van `wanted` in de instellingen staat, en elke run heeft ze allemaal. Kijk naar de waarde: `run[\"settings\"][key] != wanted[key]`."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with lines that add `run` to `found` when its settings have the value of `wanted` for every key of `wanted`.",
                  "nl": "Vervang de `...` door regels die `run` aan `found` toevoegen als zijn instellingen voor elke key van `wanted` de waarde van `wanted` hebben."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with the log and the settings you ask about, and the `def` line of your function has another number of names between its parentheses. Keep `(log, wanted)`.",
                      "nl": "De check roept `{call}` aan, met het logboek en de instellingen waar je naar vraagt, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(log, wanted)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you keep each run whose settings have the value of `wanted` for every key of `wanted`, in the order of the log?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houd je elke run waarvan de instellingen voor elke key van `wanted` de waarde van `wanted` hebben, in de volgorde van het logboek?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Do you use the keys and values of `wanted`, also when it has more than one key, or none?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruik je de keys en waarden van `wanted`, ook als het meer dan één key heeft, of geen?"},
        "correct": {"en": "Your function is right in all {n} cases: it finds every run with the settings you ask about, in the order of the log.",
                    "nl": "Je functie klopt in alle {n} gevallen: hij vindt elke run met de instellingen waar je naar vraagt, in de volgorde van het logboek."},
    },
)

# The lesson is done when these are correct. "all-settings" is the choice question, "code-version" the open question.
REQUIRED = ["log-run", "find-runs", "all-settings", "code-version"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("log-run", _START, "empty", None),
    ("log-run", _l(), "correct", None),
    ("log-run", _HEAD + "    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n"
     "    with open(path, \"a\") as f:\n        json.dump(run, f)\n        f.write(\"\\n\")\n", "correct", None),
    ("log-run", _HEAD + "    run = dict(code=version, settings=settings, seed=seed, val_loss=val_loss)\n"
     "    f = open(path, \"a\")\n    f.write(json.dumps(run) + \"\\n\")\n    f.close()\n", "correct", None),
    ("log-run", _l(mode='"w"'), "wrong", "overwrite"),
    ("log-run", _l(line="json.dumps(run)"), "error", "no-newline"),
    ("log-run", _l(line='str(run) + "\\n"'), "error", "not-json"),
    ("log-run", _l(line="run"), "error", "dict-write"),
    ("log-run", _l(line='json.dumps(run, indent=1) + "\\n"'), "wrong", "indent"),
    ("log-run", _l(run='{"version": version, "settings": settings, "seed": seed, "val_loss": val_loss}'), "wrong", "keys"),
    ("log-run", _l(run='{"code": version, "settings": settings, "val_loss": val_loss}'), "wrong", "keys"),
    ("log-run", _START.replace("    run = ...\n", "    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n"), "wrong", "dots-left"),
    ("log-run", _l(line='f"{run}\\n"'), "error", "not-json"),
    ("log-run", _l(line='json.dumps(run) + "\\n\\n"'), "error", "empty-line"),
    ("log-run", _l(line='"\\n" + json.dumps(run)'), "error", "empty-line"),
    ("log-run", _l(line='json.dumps(run) + ";\\n"'), "error", "other-json"),
    ("log-run", _l(line='json.dumps(run) + "/n"'), "error", "other-json"),
    ("log-run", _l(mode='"r"'), "error", "read-mode"),
    ("log-run", _HEAD + "    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n"
     "    with open(path) as f:\n        f.write(json.dumps(run) + \"\\n\")\n", "error", "read-mode"),
    ("log-run", _l(line='json.loads(run) + "\\n"'), "error", "loads"),
    ("log-run", _HEAD.replace("version)", "version, epochs)") + "    run = {\"code\": version, \"settings\": settings, \"seed\": seed, \"val_loss\": val_loss}\n"
     "    with open(path, \"a\") as f:\n        f.write(json.dumps(run) + \"\\n\")\n", "error", "arguments"),
    ("log-run", _l(run='{"code": "4f9a1c2e", "settings": settings, "seed": seed, "val_loss": val_loss}'), "almost", None),
    ("find-runs", _FSTART, "empty", None),
    ("find-runs", _f(), "correct", None),
    ("find-runs", _FHEAD + "    found = []\n    for run in log:\n        if all(run[\"settings\"][key] == wanted[key] for key in wanted):\n"
     "            found.append(run)\n    return found\n", "correct", None),
    ("find-runs", _FHEAD + "    return [run for run in log if all(run[\"settings\"].get(k) == v for k, v in wanted.items())]\n", "correct", None),
    ("find-runs", _FHEAD + "    found = []\n    for run in log:\n        if run[\"settings\"] == wanted:\n            found.append(run)\n    return found\n", "wrong", "exact"),
    ("find-runs", _ANY, "wrong", "any"),
    ("find-runs", _FIRST, "wrong", "first-only"),
    ("find-runs", _f(keep="found.append(run[\"settings\"])"), "wrong", "settings-only"),
    ("find-runs", _f(keep="found.append(run[\"val_loss\"])"), "wrong", "loss-only"),
    ("find-runs", _ONCE, "wrong", "same-once"),
    ("find-runs", _f(test="key not in run[\"settings\"]"), "wrong", "key-only"),
    ("find-runs", _FSTART.replace("        ...\n", "        if run[\"settings\"][\"warmup\"] == 0:\n            found.append(run)\n"), "almost", None),
]


def find_runs_of(text: str | None):
    """The learner's find_runs from the code of their box, when it is right, else the course's: (function, theirs)."""
    if text:
        ran = run_code(text)
        fn = ran.names.get("find_runs") if ran.error is None else None
        if callable(fn):
            return fn, True
    return run_code(_f()).names["find_runs"], False
