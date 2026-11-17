# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-code-sandbox: code an agent wrote, run in a separate process with a time limit, in a new empty folder,
without the environment variables of the program that starts it, and with the course's guard: an audit hook that
stops the network, files outside the folder and other programs. Eight pieces of code that a small model could write,
run with and without the guard on the learner's own computer, in a table. The exercises: `run`, which runs code that
way, and `result_text`, which turns what came back into the text the model reads. Their pictures, and a box to try the
learner's sandbox on code of their own."""
import copy
import html
import http.server
import os
import subprocess
import sys
import tempfile
import threading

from everything_ai import paths, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, kept, run_code

PYTHON = sys.executable

# The course's guard: Python code that runs before the agent's code, in the same process. It adds an audit hook
# (sys.addaudithook), which Python calls before it opens a file, makes a connection or starts a program. The hook lets
# files in the folder through, and reading Python's own files, so that imports work; the rest stops with a
# PermissionError. It is a check within Python, not a wall: code made to get out can get around it.
GUARD = '''import sys


def _guard():
    import os
    here = os.getcwd()
    python = {os.path.realpath(p) for p in
              (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
    files = {"open", "os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir",
             "os.mkdir", "os.chmod", "os.truncate", "os.utime", "os.link", "os.symlink",
             "os.chdir", "shutil.rmtree", "shutil.copyfile", "shutil.move"}
    two = {"os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.move"}
    programs = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                "os.fork", "os.forkpty", "os.kill", "os.startfile"}

    def inside(path, folders):
        real = os.path.realpath(path)
        return any(real == f or real.startswith(f + os.sep) for f in folders)

    def reads(event, args):
        if event != "open":
            return event in ("os.listdir", "os.scandir")
        mode, flags = args[1], args[2] or 0
        writes = os.O_WRONLY | os.O_RDWR | os.O_CREAT
        return (mode is None or not any(c in mode for c in "wax+")) and not flags & writes

    def hook(event, args):
        if event.startswith("socket."):
            raise PermissionError("the sandbox does not allow the network")
        if event in programs or event.startswith("ctypes."):
            raise PermissionError("the sandbox does not allow other programs")
        if event in files:
            for path in args[:2] if event in two else args[:1]:
                if path is None:
                    path = "."
                if not isinstance(path, (str, bytes, os.PathLike)):
                    continue
                path = os.fsdecode(path)
                if inside(path, {here}) or (reads(event, args) and inside(path, python)):
                    continue
                raise PermissionError("the sandbox does not allow files outside its folder")
    sys.addaudithook(hook)


sys.dont_write_bytecode = True
_guard()
del _guard
'''

SECONDS = 3   # the time limit of the course's table: enough for any row but the loop, also on a slow computer

# --- the course's sandbox, with and without the guard --------------------------------------------------------------------

EXAMPLE = paths.WORK / "sandbox-example"   # a folder outside the sandbox, with a key that is not real and a shopping list
KEY_TEXT = "EXAMPLE_API_KEY=sk-example-not-a-real-key"
SHOPPING_TEXT = "milk, eggs, flour"


KEYS = EXAMPLE / "keys.txt"
SHOPPING = EXAMPLE / "shopping.txt"


def example_files() -> dict:
    """The two files outside the sandbox, made again as they were: their full paths."""
    EXAMPLE.mkdir(parents=True, exist_ok=True)
    KEYS.write_text(KEY_TEXT + "\n", encoding="utf-8")
    SHOPPING.write_text(SHOPPING_TEXT + "\n", encoding="utf-8")
    return {"KEYS": str(KEYS), "SHOPPING": str(SHOPPING)}


def sandbox(code: str, seconds: float, guard: bool = True) -> dict:
    """The course's sandbox: the code in a separate process, in a new empty folder, without environment variables, with
    a time limit, and with the guard or without it. What came back, and the files in the folder at the end."""
    command = [PYTHON, "-I", "-c", (GUARD if guard else "") + code]
    with tempfile.TemporaryDirectory() as folder:
        try:
            done = subprocess.run(command, cwd=folder, env={}, capture_output=True, text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            return {"stopped": True, "returncode": None, "stdout": "", "stderr": "", "last": "", "files": sorted(os.listdir(folder))}
        return {"stopped": False, "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
                "last": last_line(done.stderr), "files": sorted(os.listdir(folder))}


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def last_line(text: str) -> str:
    lines = (text or "").strip().split("\n")
    return lines[-1] if lines else ""


class _Server(http.server.BaseHTTPRequestHandler):
    """A server on the learner's own computer, in place of one on the internet: it keeps what it gets."""
    got: list = []

    def do_POST(self):
        size = int(self.headers.get("Content-Length") or 0)
        _Server.got.append(self.rfile.read(size).decode("utf-8", "replace"))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"thanks")

    def log_message(self, *args):   # quiet
        pass


SHOP = ('need = {"flour": 250, "milk": 450, "eggs": 110, "butter": 25}\n'
        'have = {"flour": 1000, "milk": 200, "eggs": 0, "butter": 250}\n'
        'buy = [item + " " + str(need[item] - have[item]) + " g" for item in need if have[item] < need[item]]\n'
        'print(", ".join(buy))\n')
TYPO = 'have = {"flour": 1000, "milk": 200}\nprint(450 - have["mlk"])\n'
LONG_LOOP = 'total = 0\nfor n in range(10 ** 10):\n    total = total + n\nprint(total)\n'
NOTES = 'open("notes.txt", "w").write("Buy milk and eggs.")\n'
DELETE = 'import os\nos.remove(SHOPPING)\n'
READ_KEYS = 'print(open(KEYS).read())\n'
SEND = 'import urllib.request\nurllib.request.urlopen(URL, data=b"my notes")\n'
PROGRAM = 'import subprocess\nimport sys\nsubprocess.run([sys.executable, "-c", "print(\'outside\')"])\n'

ROWS = [   # (key, the agent's code as the table shows it, what it does in words)
    ("shop", SHOP, {"en": "works out the shopping list", "nl": "rekent de boodschappenlijst uit"}),
    ("typo", TYPO, {"en": "has a typo in a name", "nl": "heeft een tikfout in een naam"}),
    ("loop", LONG_LOOP, {"en": "adds up ten billion numbers, which takes hours", "nl": "telt tien miljard getallen op, wat uren duurt"}),
    ("notes", NOTES, {"en": "writes a file", "nl": "schrijft een bestand"}),
    ("delete", DELETE, {"en": "deletes your shopping list, by its full path", "nl": "verwijdert je boodschappenlijst, via het volledige pad"}),
    ("keys", READ_KEYS, {"en": "reads your key file, by its full path", "nl": "leest je bestand met keys, via het volledige pad"}),
    ("send", SEND, {"en": "sends your notes to a server", "nl": "stuurt je notities naar een server"}),
    ("program", PROGRAM, {"en": "starts another program", "nl": "start een ander programma"}),
]
OUTSIDE = ["delete", "keys", "send", "program"]   # the rows that reach outside the folder


def _measure() -> dict:
    """Every row run with and without the guard, with the files outside made again before each run and the course's
    server listening: the outcome of each run, and whether it reached outside the folder."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/upload"
    out = {}
    try:
        for key, code, _ in ROWS:
            for guard in (False, True):
                where = example_files()
                _Server.got = []
                given = {"KEYS": where["KEYS"], "SHOPPING": where["SHOPPING"], "URL": url}
                names = "".join(f"{k} = {v!r}\n" for k, v in given.items() if k in code)   # only the names the code uses
                o = sandbox(names + code, SECONDS, guard)
                o["reached"] = {"delete": not os.path.exists(where["SHOPPING"]),
                                "keys": KEY_TEXT in o["stdout"],
                                "send": bool(_Server.got),
                                "program": "outside" in o["stdout"]}.get(key, False)
                out[f"{key}-{'guard' if guard else 'plain'}"] = o
    finally:
        server.shutdown()
        server.server_close()
        example_files()   # the shopping list back as it was
    return out


def measured() -> dict:
    return kept("build-code-sandbox:table", _measure)


def findings() -> dict:
    m = measured()
    keys = [k for k, _, _ in ROWS]
    return {
        "rows": len(ROWS),
        "seconds": SECONDS,
        "plain_reached": [k for k in keys if m[f"{k}-plain"]["reached"]],
        "guard_reached": [k for k in keys if m[f"{k}-guard"]["reached"]],
        "stopped": [k for k in keys if m[f"{k}-plain"]["stopped"] and m[f"{k}-guard"]["stopped"]],
        "guard_errors": [k for k in OUTSIDE if "the sandbox does not allow" in m[f"{k}-guard"]["last"]],
        "notes_files": m["notes-plain"]["files"],
    }


_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.25rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


_TABLE_WORDS = {
    "en": {"title": "Eight pieces of code an agent could write, in a sandbox without and with the guard", "code": "the agent's code",
           "plain": "separate process, time limit, empty folder", "guard": "with the course's guard as well",
           "printed": "printed {v}", "nothing": "ran, and printed nothing", "error": "stopped with an error: {v}",
           "stopped": "stopped after {s} seconds", "file": "wrote {v} in the empty folder, which was then thrown away",
           "delete": "deleted your shopping list (the course put it back)", "keys": "printed your key",
           "send": "reached the server, which got your notes", "program": "started another program, which printed {v}",
           "names": "In the code, KEYS and SHOPPING are the full paths of two files in your work folder, outside the empty folder, and URL is the address of a server the course starts on your own computer, in place of one on the internet."},
    "nl": {"title": "Acht stukken code die een agent zou kunnen schrijven, in een sandbox zonder en met de bewaker", "code": "de code van de agent",
           "plain": "apart proces, tijdslimiet, lege map", "guard": "ook met de bewaker van de course",
           "printed": "printte {v}", "nothing": "draaide, en printte niets", "error": "stopte met een fout: {v}",
           "stopped": "gestopt na {s} seconden", "file": "schreef {v} in de lege map, die daarna werd weggegooid",
           "delete": "verwijderde je boodschappenlijst (de course zette hem terug)", "keys": "printte je key",
           "send": "bereikte de server, die je notities kreeg", "program": "startte een ander programma, dat {v} printte",
           "names": "In de code zijn KEYS en SHOPPING de volledige paden van twee bestanden in je werkmap, buiten de lege map, en URL is het adres van een server die de course op je eigen computer start, in plaats van een op internet."},
}


def outcome(key: str, o: dict) -> tuple:
    """What one run did: the words for it in _TABLE_WORDS, and the value that goes into them, such as what it printed."""
    if o["stopped"]:
        return "stopped", ""
    if o["reached"]:
        return key, o["stdout"].strip() if key == "program" else ""
    if o["returncode"] != 0:
        return "error", o["last"]
    if o["files"]:
        return "file", ", ".join(o["files"])
    return ("printed", o["stdout"].strip()) if o["stdout"].strip() else ("nothing", "")


def outcome_text(key: str, o: dict, lang: str) -> str:
    """What one run did, in words, with the value in backticks."""
    words, value = outcome(key, o)
    return _TABLE_WORDS[lang][words].format(v=f"`{value}`", s=SECONDS)


def _outcome_html(key: str, o: dict, lang: str, color: str) -> str:
    words, value = outcome(key, o)
    text = html.escape(_TABLE_WORDS[lang][words].replace("{s}", str(SECONDS)))
    return text.replace("{v}", _code(value, color))


def table_picture(lang: str) -> str:
    """Each piece of code with what it did in the sandbox without the guard, and with it."""
    w, m = _TABLE_WORDS[lang], measured()
    rows = []
    for key, code, what in ROWS:
        cells = []
        for side in ("plain", "guard"):
            o = m[f"{key}-{side}"]
            color = "var(--ea-loss-text)" if o["reached"] else "var(--ea-ink)"
            weight = 600 if o["reached"] else 400
            cells.append(f'<td style="{_CELL};{_UI};color:{color};font-weight:{weight}">{_outcome_html(key, o, lang, color)}</td>')
        rows.append(f'<tr><td style="{_CELL}"><div style="{_UI};font-size:13px;margin-bottom:0.15rem">{html.escape(what[lang])}</div>'
                    f'{_code(code.strip())}</td>{"".join(cells)}</tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["code"], w["plain"], w["guard"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             f'<p style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.4rem 0 0">{html.escape(w["names"])}</p>')
    return yours.figure(w["title"], table, alt=w["title"])


_LAYER_WORDS = {
    "en": {"title": "The course's sandbox, layer by layer", "computer": "your computer: your files, your keys, the internet",
           "loop": "your program, with the agent loop", "code": "the code the model wrote, as the arguments of a tool call",
           "process": "a separate process, with a time limit and without your environment variables",
           "guard": "the guard: no network, no files outside the folder, no other programs", "folder": "a new, empty folder",
           "agent": "the agent's code runs here", "back": "back to the model: the output, or the last line of an error",
           "alt": "Nested boxes: your computer holds your program; the code goes from it into a separate process, within the guard, in an empty folder; what comes out goes back to the model."},
    "nl": {"title": "De sandbox van de course, laag voor laag", "computer": "je computer: je bestanden, je keys, internet",
           "loop": "je programma, met de agent-loop", "code": "de code die het model schreef, als argumenten van een tool call",
           "process": "een apart proces, met een tijdslimiet en zonder je environment variables",
           "guard": "de bewaker: geen netwerk, geen bestanden buiten de map, geen andere programma's", "folder": "een nieuwe, lege map",
           "agent": "hier draait de code van de agent", "back": "terug naar het model: de output, of de laatste regel van een fout",
           "alt": "Geneste vakken: je computer bevat je programma; de code gaat daaruit naar een apart proces, binnen de bewaker, in een lege map; wat eruit komt, gaat terug naar het model."},
}


def layers_picture(lang: str) -> str:
    """The layers of the sandbox as nested boxes, with the code going in and the text coming back."""
    w = _LAYER_WORDS[lang]

    def box(label: str, inner: str, border: str, color: str = "var(--ea-ink-2)") -> str:
        return (f'<div style="border:{border};border-radius:6px;padding:0.45rem 0.6rem;margin-top:0.35rem">'
                f'<div style="{_UI};font-size:12px;color:{color}">{html.escape(label)}</div>{inner}</div>')
    arrow = (f'<div style="{_UI};font-size:12px;margin:0.35rem 0 0 0.6rem;color:{{c}}">{{t}}</div>')
    core = box(w["folder"], f'<div style="{_UI};font-size:13px;color:var(--ea-pred-text);margin-top:0.2rem">{html.escape(w["agent"])}</div>',
               "1px solid var(--ea-line)")
    sandbox = box(w["process"], box(w["guard"], core, "1px dashed var(--ea-ink-2)"), "2px solid var(--ea-ink)", "var(--ea-ink)")
    body = box(w["computer"], box(w["loop"], "", "1px solid var(--ea-line)")
               + arrow.format(c="var(--ea-pred-text)", t="↓ " + html.escape(w["code"]))
               + sandbox
               + arrow.format(c="var(--ea-data-text)", t="↑ " + html.escape(w["back"])), "1px solid var(--ea-line)")
    return yours.figure(w["title"], body, alt=w["alt"])


def measured_text(lang: str) -> str:
    """What the table shows, in the lesson's language, from the runs on this computer."""
    f = findings()
    plain, guard = f["plain_reached"], f["guard_reached"]
    every = sorted(plain) == sorted(OUTSIDE)
    if lang == "nl":
        reach = (f"Zonder bewaker deden {len(plain)} van de {f['rows']} stukken code buiten de lege map wat ze probeerden"
                 f"{': verwijderen, lezen, iets versturen en een programma starten' if every else ''}. Een apart proces in een "
                 f"lege map houdt daar niets van tegen: de map bepaalt alleen waar een kort pad zoals `notes.txt` terechtkomt, "
                 f"en het proces mag alles wat jij mag. ")
        stop = (f"Met de bewaker stopten {'ze alle' if f['guard_errors'] == plain else 'er'} {len(f['guard_errors'])} met een fout van de bewaker, en "
                f"{'kwam er geen enkele buiten de map' if not guard else f'kwamen er nog {len(guard)} buiten de map'}. ")
        return (reach + stop + f"De loop van uren stopte in beide gevallen na {SECONDS} seconden, de tikfout gaf een fout die het "
                f"model kan lezen, en het bestand van de agent kwam in de lege map terecht, die na de run weg was.")
    reach = (f"Without the guard, {len(plain)} of the {f['rows']} pieces of code did outside the empty folder what they tried"
             f"{': deleting, reading, sending and starting a program' if every else ''}. A separate process in an empty folder "
             f"stops none of that: the folder only decides where a short path such as `notes.txt` ends up, and the process "
             f"may do whatever you may. ")
    stop = (f"With the guard, {'all ' if f['guard_errors'] == plain else ''}{len(f['guard_errors'])} stopped with an error from the guard, and "
            f"{'none got outside the folder' if not guard else f'{len(guard)} still got outside the folder'}. ")
    return (reach + stop + f"The loop of hours was stopped after {SECONDS} seconds both times, the typo gave an error the model "
            f"can read, and the agent's file landed in the empty folder, which was gone after the run.")


def worked_text(lang: str) -> str:
    """The worked example: the typo, run in the sandbox with the guard, step by step, from the run on this computer."""
    o = measured()["typo-guard"]
    if lang == "nl":
        return (f"### Uitgewerkt voorbeeld: een tikfout\n\n"
                f"De code van de agent voor de boodschappenlijst heeft `have[\"mlk\"]` waar hij `have[\"milk\"]` bedoelt.\n\n"
                f"1. **Je programma krijgt de code** als argumenten van een tool call, en start een apart proces: Python met "
                f"`-I`, de bewaker en de code, in een nieuwe lege map, zonder je environment variables, met een limiet van "
                f"{SECONDS} seconden.\n"
                f"2. **De code stopt met een fout.** `subprocess.run` geeft returncode {o['returncode']} terug, niets in "
                f"`stdout`, en in `stderr` een traceback, waarvan de laatste regel zegt wat er misging: "
                f"`{o['last']}`.\n"
                f"3. **De map wordt weggegooid, en je programma gaat door:** een fout in de code van de agent is geen fout in "
                f"de jouwe.\n"
                f"4. **Het model krijgt één regel terug:** `Error: {o['last']}`. Daarmee kan het de code opnieuw schrijven, "
                f"met `milk`.\n\n"
                f"Dit kwam in `stderr` terecht; het regelnummer telt de regels van de bewaker mee:\n\n"
                f"```text\n{o['stderr'].strip()}\n```")
    return (f"### Worked example: a typo\n\n"
            f"The agent's code for the shopping list has `have[\"mlk\"]` where it means `have[\"milk\"]`.\n\n"
            f"1. **Your program gets the code** as the arguments of a tool call, and starts a separate process: Python with "
            f"`-I`, the guard and the code, in a new empty folder, without your environment variables, with a limit of "
            f"{SECONDS} seconds.\n"
            f"2. **The code stops with an error.** `subprocess.run` gives back returncode {o['returncode']}, nothing in "
            f"`stdout`, and in `stderr` a traceback, whose last line says what went wrong: `{o['last']}`.\n"
            f"3. **The folder is thrown away, and your program goes on:** an error in the agent's code is not an error in "
            f"yours.\n"
            f"4. **The model gets back one line:** `Error: {o['last']}`. With it, it can write the code again, with `milk`.\n\n"
            f"This is what came in `stderr`; the line number counts the lines of the guard too:\n\n"
            f"```text\n{o['stderr'].strip()}\n```")


# --- exercise 1: run code in a separate process --------------------------------------------------------------------------

START_RUN = '''import subprocess
import tempfile


def run(code, seconds):
    command = [PYTHON, "-I", "-c", GUARD + code]
    with tempfile.TemporaryDirectory() as folder:
        ...
'''
SOLUTION_RUN = START_RUN.replace('''        ...
''', '''        try:
            return subprocess.run(command, cwd=folder, env={}, capture_output=True,
                                  text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            return None
''')
SANDBOX = {"PYTHON": PYTHON, "GUARD": GUARD}

EXAMPLE_KEY = "EXAMPLE_API_KEY"   # a variable the check puts in the environment while it runs, as a key would be there


class Code(str):
    """A piece of code that a check's message names by a short name instead of writing it all out."""

    def __new__(cls, name: str, text: str):
        s = super().__new__(cls, text)
        s.name = name
        return s

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


C_SHOP = Code("shop_code", SHOP)
C_TYPO = Code("typo_code", TYPO)
C_FOLDER = Code("folder_code", 'import os\nprint(os.listdir("."))\n')
C_ENV = Code("env_code", f'import os\nprint(os.environ.get("{EXAMPLE_KEY}"))\n')
C_SLOW = Code("slow_code", 'import time\ntime.sleep(1.2)\nprint("done")\n')
C_KEYS = Code("keys_code", f"print(open({str(KEYS)!r}).read())\n")


def summary(done):
    """What counts in what run gives back: None, or the returncode, the output and the last line of the error."""
    if done is None or not isinstance(done, subprocess.CompletedProcess):
        return done
    return {"returncode": done.returncode, "stdout": done.stdout, "last": last_line(done.stderr) if isinstance(done.stderr, str) else done.stderr}


def _shown_summary(v):
    if not isinstance(v, dict):
        return {"en": repr(v), "nl": repr(v)}
    return {"en": f"returncode {v['returncode']}, stdout {v['stdout']!r}, last line of stderr {v['last']!r}",
            "nl": f"returncode {v['returncode']}, stdout {v['stdout']!r}, laatste regel van stderr {v['last']!r}"}


_RUN_NOW: dict = {}   # the case the check runs now: RunBox._run sets it, so that a mistake knows which case it judges


class RunBox(FunctionBox):
    """A FunctionBox for run: while it runs a case, a key that is not real is in the environment, as keys often are,
    and the files outside the sandbox are there as they were."""

    def _run(self, fn, case):
        _RUN_NOW["case"] = case
        example_files()
        before = os.environ.get(EXAMPLE_KEY)
        os.environ[EXAMPLE_KEY] = "sk-example-not-a-real-key"
        try:
            return summary(fn(*case))
        finally:
            if before is None:
                os.environ.pop(EXAMPLE_KEY, None)
            else:
                os.environ[EXAMPLE_KEY] = before

    def _call_text(self, case) -> str:
        return f"run({case[0]!r}, {case[1]})"


RUN_CASES = [(C_SHOP, 2), (C_TYPO, 2), (C_FOLDER, 2), (C_ENV, 2), (C_KEYS, 2), (C_SLOW, 0.5)]


def _now_is(code: Code) -> bool:
    return _RUN_NOW.get("case", (None,))[0] is code


def _no_timeout(got, text) -> bool:
    return _now_is(C_SLOW) and isinstance(got, dict) and got.get("stdout") == "done\n"


def _env(got, text) -> bool:
    return _now_is(C_ENV) and isinstance(got, dict) and "sk-example" in str(got.get("stdout"))


def _no_folder(got, text) -> bool:
    return _now_is(C_FOLDER) and isinstance(got, dict) and got.get("returncode") == 0 and got.get("stdout") != "[]\n"


def _no_guard(got, text) -> bool:
    return _now_is(C_KEYS) and isinstance(got, dict) and KEY_TEXT in str(got.get("stdout"))


def _not_captured(got, text) -> bool:
    return isinstance(got, dict) and got.get("stdout") is None


def _bytes(got, text) -> bool:
    return isinstance(got, dict) and isinstance(got.get("stdout"), bytes)


def _timeout_value(got, text) -> bool:
    return _now_is(C_SLOW) and got is not None and not isinstance(got, dict)


_RUN_WORDS = {
    "en": {"title": "Your run on the check's code", "code": "the code", "got": "what your run gave back", "want": "the course's run",
           "none": "None: the time limit stopped it", "error": "stops with {e}"},
    "nl": {"title": "Je run op de code van de check", "code": "de code", "got": "wat je run teruggaf", "want": "de run van de course",
           "none": "None: de tijdslimiet stopte hem", "error": "stopt met {e}"},
}


def _summary_html(v, w) -> str:
    if v is None:
        return _code(w["none"], "var(--ea-ink-2)")
    if not isinstance(v, dict):
        return _code(repr(v))
    parts = [f"returncode {v['returncode']}"]
    if v["stdout"]:
        parts.append(f"stdout {v['stdout']!r}")
    if v["last"]:
        parts.append(f"stderr … {v['last']!r}")
    return _code(", ".join(parts))


def run_picture(seen) -> str:
    """What the learner's run gave back for each piece of code of the check, beside the course's run."""
    run_ = seen.run
    fn = run_.names.get("run") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _RUN_WORDS[seen.lang]
    box = EXERCISES["run"]
    wants = seen.remember("results", lambda: [box._run(box._fn(SOLUTION_RUN), case) for case in RUN_CASES])
    rows = []
    for case, want in zip(RUN_CASES, wants):
        try:
            mine = _summary_html(box._run(fn, case), w)
        except Exception as e:   # noqa: BLE001 - the learner's code
            mine = _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)")
        want = _summary_html(want, w)
        code = f"{case[0].name}, {case[1]} s:\n{case[0].strip()}"
        rows.append(f'<tr><td style="{_CELL}">{_code(code)}</td><td style="{_CELL}">{mine}</td><td style="{_CELL}">{want}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["code"], w["got"], w["want"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 2: what the model reads -----------------------------------------------------------------------------------

LIMIT = 1000   # characters of output the model gets back at most

START_TEXT = '''def result_text(done, seconds):
    if done is None:
        return "Stopped: the code was still running after " + str(seconds) + " seconds."
    return done.stdout
'''
SOLUTION_TEXT = START_TEXT.replace('''    return done.stdout
''', '''    if done.returncode != 0:
        lines = done.stderr.strip().split("\\n")
        return "Error: " + lines[-1]
    if len(done.stdout) > LIMIT:
        cut = len(done.stdout) - LIMIT
        return done.stdout[:LIMIT] + "\\n(cut off: " + str(cut) + " more characters)"
    return done.stdout
''')
TEXT_NAMES = {"LIMIT": LIMIT}


class Done(subprocess.CompletedProcess):
    """What subprocess.run gives back, which a check's message names by a short name."""

    def __init__(self, name: str, returncode: int, stdout: str, stderr: str):
        super().__init__([PYTHON, "-I", "-c", "..."], returncode, stdout, stderr)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Done(self.name, self.returncode, self.stdout, self.stderr)


TRACEBACK = ('Traceback (most recent call last):\n  File "<string>", line 59, in <module>\nKeyError: \'mlk\'\n')
D_SHOP = Done("done_shop", 0, "milk 250 g, eggs 110 g\n", "")
D_TYPO = Done("done_typo", 1, "", TRACEBACK)
D_LONG = Done("done_long", 0, "".join(f"line {i}\n" for i in range(1, 501)), "")
D_EMPTY = Done("done_quiet", 0, "", "")
D_EXIT = Done("done_exit", 2, "half a list\n", "Error: no recipe for waffles\n")
TEXT_CASES = [(D_SHOP, 2), (None, 2), (D_TYPO, 2), (D_LONG, 2), (D_EMPTY, 2), (D_EXIT, 5), (None, 5)]

_NOW: dict = {}   # the case the check runs now: TextBox._run sets it, so that the comparison knows what it judges


def _right_text(got, want) -> bool:
    """Right when the text says what the model needs: how long it ran, the last line of an error and not the whole
    traceback, the output as it is, or the output cut to LIMIT with how many characters were cut."""
    done, seconds = _NOW.get("case", (None, None))
    if not isinstance(got, str):
        return False
    if done is None:
        return str(seconds) in got
    if done.returncode != 0:
        return last_line(done.stderr) in got and "Traceback" not in got and len(got) <= 200
    if len(done.stdout) > LIMIT:
        return got.startswith(done.stdout[:LIMIT]) and len(got) <= LIMIT + 200 and str(len(done.stdout) - LIMIT) in got[LIMIT:]
    return got == done.stdout


class TextBox(FunctionBox):
    def _run(self, fn, case):
        _NOW["case"] = case
        return fn(*copy.deepcopy(case))

    def _call_text(self, case) -> str:
        return f"result_text({case[0]!r}, {case[1]})"


def _now(kind: str):
    def test(got, text) -> bool:
        done, seconds = _NOW.get("case", (None, None))
        if not isinstance(got, str):
            return False
        return {
            "whole-stderr": done is not None and done.returncode != 0 and "Traceback" in got,
            "no-error": done is not None and done.returncode != 0 and got == done.stdout,
            "not-cut": done is not None and done.returncode == 0 and len(done.stdout) > LIMIT and got == done.stdout,
            "no-strip": done is not None and done.returncode != 0 and got.rstrip().endswith(":") and len(got) <= 20,
            "no-note": done is not None and done.returncode == 0 and len(done.stdout) > LIMIT and got == done.stdout[:LIMIT],
        }[kind]
    return test


def _short(v) -> str:
    return repr(v) if not isinstance(v, str) or len(v) <= 90 else repr(v[:60] + " … " + v[-20:])


_TEXT_WORDS = {
    "en": {"title": "What the model reads back from your result_text", "done": "what came back", "reads": "what the model reads",
           "error": "stops with {e}", "chars": "{n} characters"},
    "nl": {"title": "Wat het model terugleest uit je result_text", "done": "wat er terugkwam", "reads": "wat het model leest",
           "error": "stopt met {e}", "chars": "{n} tekens"},
}


def _done_html(done, seconds, lang: str) -> str:
    if done is None:
        return _code({"en": f"None, after a limit of {seconds} seconds", "nl": f"None, na een limiet van {seconds} seconden"}[lang])
    parts = [f"returncode {done.returncode}"]
    if done.stdout:
        parts.append(f"stdout {_short(done.stdout)}" + (f" ({_TEXT_WORDS[lang]['chars'].format(n=len(done.stdout))})" if len(done.stdout) > 90 else ""))
    if done.stderr:
        parts.append(f"stderr {_short(done.stderr)}")
    return _code(", ".join(parts))


def text_picture(seen) -> str:
    """For each case of the check: what came back from the process, and what the learner's function lets the model read."""
    run_ = seen.run
    fn = run_.names.get("result_text") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _TEXT_WORDS[seen.lang]
    rows = []
    for done, seconds in TEXT_CASES[:5]:
        try:
            got = fn(copy.deepcopy(done), seconds)
            _NOW["case"] = (done, seconds)
            ok = _right_text(got, None)
            mine = _code(_short(got), "var(--ea-pred-text)" if ok else "var(--ea-loss-text)")
        except Exception as e:   # noqa: BLE001 - the learner's code
            mine = _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)")
        rows.append(f'<tr><td style="{_CELL}">{_done_html(done, seconds, seen.lang)}</td><td style="{_CELL}">{mine}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["done"], w["reads"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- the exercises ----------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _same(got, want) -> bool:
    return got == want


EXERCISES = {
    "run": RunBox(
        name="run",
        picture=run_picture,
        function="run",
        start=START_RUN,
        solution=SOLUTION_RUN,
        names=SANDBOX,
        visible=RUN_CASES[:1],
        hidden=RUN_CASES[1:],
        compare=_same,
        show=_shown_summary,
        mistakes=[
            Mistake("not-captured", matches=_not_captured,
                    en="With `{call}`, your run gives back {got}: the output went to the screen, not into the result. Give `subprocess.run` `capture_output=True`, so that `stdout` and `stderr` come back.",
                    nl="Met `{call}` geeft je run {got} terug: de output ging naar het scherm, niet naar de uitkomst. Geef `subprocess.run` `capture_output=True` mee, dan komen `stdout` en `stderr` terug."),
            Mistake("bytes", matches=_bytes,
                    en="With `{call}`, the output comes back as bytes: {got}. Give `subprocess.run` `text=True`, so that it comes back as text.",
                    nl="Met `{call}` komt de output terug als bytes: {got}. Geef `subprocess.run` `text=True` mee, dan komt hij terug als tekst."),
            Mistake("no-timeout", matches=_no_timeout,
                    en="With `{call}`, your run waited until the code was done, though it should have stopped it after half a second. Give `subprocess.run` the time limit: `timeout=seconds`.",
                    nl="Met `{call}` wachtte je run tot de code klaar was, terwijl hij hem na een halve seconde had moeten stoppen. Geef `subprocess.run` de tijdslimiet mee: `timeout=seconds`."),
            Mistake("timeout-value", matches=_timeout_value,
                    en="With `{call}`, the time limit stopped the code, and your run gives back {got}. Give back `None` then, so that the code that calls `run` can tell.",
                    nl="Met `{call}` stopte de tijdslimiet de code, en je run geeft {got} terug. Geef dan `None` terug, zodat de code die `run` aanroept het kan zien."),
            Mistake("env", matches=_env,
                    en="With `{call}`, the code printed a key from your environment variables: {got}. Give `subprocess.run` `env={{}}`, so that the process gets none of yours.",
                    nl="Met `{call}` printte de code een key uit je environment variables: {got}. Geef `subprocess.run` `env={{}}` mee, dan krijgt het proces er geen van jou."),
            Mistake("no-folder", matches=_no_folder,
                    en="With `{call}`, the code sees the files of the folder your program runs in: {got}. Give `subprocess.run` `cwd=folder`, so that it starts in the new, empty folder.",
                    nl="Met `{call}` ziet de code de bestanden van de map waarin je programma draait: {got}. Geef `subprocess.run` `cwd=folder` mee, dan begint hij in de nieuwe, lege map."),
            Mistake("no-guard", matches=_no_guard,
                    en="With `{call}`, the code read a file outside the folder: {got}. Run `command` as the box made it, with `GUARD` before the code.",
                    nl="Met `{call}` las de code een bestand buiten de map: {got}. Voer `command` uit zoals het vak hem maakte, met `GUARD` vóór de code."),
            Mistake("uncaught", error=lambda e: isinstance(e, subprocess.TimeoutExpired),
                    en="With `{call}`, the time limit stopped the code, and `subprocess.run` raised `subprocess.TimeoutExpired`, which stopped your run too. Catch it: put the call in `try:`, and give back `None` under `except subprocess.TimeoutExpired:`.",
                    nl="Met `{call}` stopte de tijdslimiet de code, en `subprocess.run` gaf `subprocess.TimeoutExpired`, waardoor ook je run stopte. Vang die op: zet de aanroep in `try:`, en geef `None` terug onder `except subprocess.TimeoutExpired:`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The box makes the command and the folder, but runs nothing yet. In the `with` block, run `command` with `subprocess.run`, and give back what it gives, or `None` when the time limit stops it.",
                      "nl": "Het vak maakt de opdracht en de map, maar voert nog niets uit. Voer in het `with`-blok `command` uit met `subprocess.run`, en geef terug wat dat geeft, of `None` als de tijdslimiet hem stopt."},
            "wrong": {"en": "With `{call}`, your run gives back {got}, and the course's run {want}. Run `command` in `folder`, without your environment variables, with the output as text and a time limit of `seconds`.",
                      "nl": "Met `{call}` geeft je run {got} terug, en de run van de course {want}. Voer `command` uit in `folder`, zonder je environment variables, met de output als tekst en een tijdslimiet van `seconds`."},
            "almost": {"en": "Right for the shopping list. But with `{call}`, your run gives back {got}, and the course's run {want}. Compare them in the picture.",
                       "nl": "Goed voor de boodschappenlijst. Maar met `{call}` geeft je run {got} terug, en de run van de course {want}. Vergelijk ze in het plaatje."},
            "no_function": {"en": "Your box does not make a function `run`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `run`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `run` is a sandbox: a separate process in a new empty folder, without your environment variables, with the guard, and stopped at the time limit. Right for all {n} pieces of code of the check.",
                        "nl": "Je `run` is een sandbox: een apart proces in een nieuwe lege map, zonder je environment variables, met de bewaker, en gestopt bij de tijdslimiet. Goed voor alle {n} stukken code van de check."},
        },
    ),
    "result_text": TextBox(
        name="result_text",
        picture=text_picture,
        function="result_text",
        start=START_TEXT,
        solution=SOLUTION_TEXT,
        names=TEXT_NAMES,
        visible=TEXT_CASES[:1],
        hidden=TEXT_CASES[1:],
        compare=_right_text,
        show=_short,
        mistakes=[
            Mistake("whole-stderr", matches=_now("whole-stderr"),
                    en="With `{call}`, the model gets the whole traceback: {got}. Its last line says what went wrong; send only that, after `\"Error: \"`.",
                    nl="Met `{call}` krijgt het model de hele traceback: {got}. De laatste regel zegt wat er misging; stuur alleen die, na `\"Error: \"`."),
            Mistake("no-error", matches=_now("no-error"),
                    en="With `{call}`, the code stopped with an error, and your function gives back its output: {got}. When `done.returncode` is not 0, send the last line of `done.stderr` instead.",
                    nl="Met `{call}` stopte de code met een fout, en je functie geeft de output terug: {got}. Stuur als `done.returncode` niet 0 is de laatste regel van `done.stderr`."),
            Mistake("not-cut", matches=_now("not-cut"),
                    en="With `{call}`, the output has more than `LIMIT` characters, and your function sends it all. Cut it to the first `LIMIT`, so that one print in a loop cannot fill the model's context.",
                    nl="Met `{call}` heeft de output meer dan `LIMIT` tekens, en je functie stuurt hem helemaal. Knip hem af op de eerste `LIMIT`, zodat één print in een loop de context van het model niet kan vullen."),
            Mistake("no-strip", matches=_now("no-strip"),
                    en="With `{call}`, the model gets {got}: `done.stderr` ends with a new line, so the last piece after `split` is empty. Take the new line at the end off first, with `.strip()`.",
                    nl="Met `{call}` krijgt het model {got}: `done.stderr` eindigt op een nieuwe regel, dus het laatste stuk na `split` is leeg. Haal de nieuwe regel aan het eind er eerst af, met `.strip()`."),
            Mistake("no-note", matches=_now("no-note"),
                    en="With `{call}`, your function cuts the output, but the model cannot tell. Add a line that says how many characters were cut.",
                    nl="Met `{call}` knipt je functie de output af, maar het model kan dat niet zien. Voeg een regel toe die zegt hoeveel tekens er weg zijn."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The function only handles the time limit yet. Add a text for an error, with the last line of `done.stderr`, and cut an output that is longer than `LIMIT`.",
                      "nl": "De functie regelt nog alleen de tijdslimiet. Voeg een tekst toe voor een fout, met de laatste regel van `done.stderr`, en knip een output af die langer is dan `LIMIT`."},
            "wrong": {"en": "With `{call}`, your function gives back {got}. The model needs the output as it is, or the last line of an error, or the output cut to `LIMIT` characters with how many were cut.",
                      "nl": "Met `{call}` geeft je functie {got} terug. Het model heeft de output nodig zoals hij is, of de laatste regel van een fout, of de output afgeknipt op `LIMIT` tekens met hoeveel er weg zijn."},
            "almost": {"en": "Right for the shopping list. But with `{call}`, your function gives back {got}. Does it handle an error and a long output, and give back any other output as it is?",
                       "nl": "Goed voor de boodschappenlijst. Maar met `{call}` geeft je functie {got} terug. Werkt hij ook bij een fout en een lange output, en geeft hij elke andere output terug zoals hij is?"},
            "no_function": {"en": "Your box does not make a function `result_text`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `result_text`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `result_text` gives the model what it can use: the output, cut at `LIMIT` with a note, the last line of an error, or how long the code ran before it was stopped. Right for all {n} cases.",
                        "nl": "Je `result_text` geeft het model wat het kan gebruiken: de output, afgeknipt bij `LIMIT` met een regel erover, de laatste regel van een fout, of hoe lang de code liep voordat hij werd gestopt. Goed bij alle {n} gevallen."},
        },
    ),
}

REQUIRED = ["run", "result_text", "guard-wall"]

CASES = [
    ("run", START_RUN, "empty", None),
    ("run", "", "empty", None),
    ("run", SOLUTION_RUN, "correct", None),
    ("run", SOLUTION_RUN.replace("capture_output=True,\n                                  text=True,", "stdout=subprocess.PIPE,\n                                  stderr=subprocess.PIPE, text=True,"), "correct", None),
    ("run", SOLUTION_RUN.replace("env={}, capture_output=True,", "env={},"), "wrong", "not-captured"),
    ("run", SOLUTION_RUN.replace("\n                                  text=True,", ""), "wrong", "bytes"),
    ("run", SOLUTION_RUN.replace(", timeout=seconds", ""), "wrong", "no-timeout"),
    ("run", SOLUTION_RUN.replace("            return None\n", '            return "too slow"\n'), "wrong", "timeout-value"),
    ("run", SOLUTION_RUN.replace(" env={},", ""), "wrong", "env"),
    ("run", SOLUTION_RUN.replace(" cwd=folder,", ""), "wrong", "no-folder"),
    ("run", SOLUTION_RUN.replace("GUARD + code", "code"), "wrong", "no-guard"),
    ("run", SOLUTION_RUN.replace(", timeout=seconds", ", timeout=5"), "wrong", "no-timeout"),
    ("run", SOLUTION_RUN.replace("            return subprocess.run(", "            done = subprocess.run(").replace(
        "timeout=seconds)\n", "timeout=seconds)\n            return done.stdout\n"), "wrong", None),
    ("run", SOLUTION_RUN.replace("        except subprocess.TimeoutExpired:\n            return None\n", "        except ValueError:\n            return None\n"), "error", "uncaught"),
    ("result_text", START_TEXT, "empty", None),
    ("result_text", "", "empty", None),
    ("result_text", SOLUTION_TEXT, "correct", None),
    ("result_text", START_TEXT.replace('''    return done.stdout
''', '''    if done.returncode != 0:
        return "The code stopped: " + done.stderr.split("\\n")[-2]
    if len(done.stdout) > LIMIT:
        return done.stdout[:LIMIT] + " ... and " + str(len(done.stdout) - LIMIT) + " characters more"
    return done.stdout
'''), "correct", None),
    ("result_text", SOLUTION_TEXT.replace('return "Error: " + lines[-1]', 'return "Error: " + done.stderr'), "wrong", "whole-stderr"),
    ("result_text", SOLUTION_TEXT.replace('''    if done.returncode != 0:
        lines = done.stderr.strip().split("\\n")
        return "Error: " + lines[-1]
''', ""), "wrong", "no-error"),
    ("result_text", SOLUTION_TEXT.replace("    if len(done.stdout) > LIMIT:\n", "    if len(done.stdout) > LIMIT * 10:\n"), "wrong", "not-cut"),
    ("result_text", SOLUTION_TEXT.replace(' + "\\n(cut off: " + str(cut) + " more characters)"', ""), "wrong", "no-note"),
    ("result_text", SOLUTION_TEXT.replace("done.stderr.strip().split", "done.stderr.split"), "wrong", "no-strip"),
    ("result_text", SOLUTION_TEXT.replace('''        return done.stdout[:LIMIT] + "\\n(cut off: " + str(cut) + " more characters)"
''', '''        done.stdout = done.stdout[:LIMIT] + "\\n(cut off: " + str(cut) + " more characters)"
'''), "correct", None),
    ("result_text", SOLUTION_TEXT.replace("    return done.stdout\n", '    return done.stdout + "(done)"\n'), "wrong", None),
    ("result_text", SOLUTION_TEXT.replace(' + str(seconds) + " seconds."', ' + "a while."'), "almost", None),
]


# --- try it: the learner's sandbox on code of their own -------------------------------------------------------------------

def learner_run(text):
    """The learner's run from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["run"].check(text).state != "correct":
        return None
    return run_code(text, dict(SANDBOX)).names.get("run")


def learner_text(text):
    """The learner's result_text from exercise 2, when it is right; else the course's."""
    if text is not MISSING and text is not None and EXERCISES["result_text"].check(text).state == "correct":
        fn = run_code(text, dict(TEXT_NAMES)).names.get("result_text")
        if callable(fn):
            return fn
    return run_code(SOLUTION_TEXT, dict(TEXT_NAMES)).names["result_text"]


def tried(run_fn, text_fn, code: str, seconds: float, words: dict) -> str:
    """The learner's own code through their sandbox: what came back, and what the model would read."""
    try:
        done = run_fn(code, seconds)
        reads = text_fn(done, seconds)
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(words["title"], _code(words["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=words["title"])
    body = (f'<p style="{_UI};font-size:13px;margin:0 0 0.3rem">{html.escape(words["reads"])}</p>'
            f'<div style="background:var(--ea-inset);padding:0.4rem 0.6rem;border-radius:4px">{_code(str(reads), "var(--ea-data-text)")}</div>')
    return yours.figure(words["title"], body, alt=words["title"])
