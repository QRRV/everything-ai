# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-report: markdown_table, the table of the kit's report at the precision the metric
has, saved as own/table.py; and the report of the learner's own reproduction: the parts they write, Claim, Setup,
Differences and Not checked, and Results, which the course writes from their kit with their own table, put together
with report of Guided reproduction 7 and saved as own/REPORT.md. The demo puts the files that go public in
own/MANIFEST.txt, with the SHA-256 of each, as sha256sum writes them."""
import hashlib
import importlib.util
import re
from dataclasses import dataclass, field

from everything_ai import paths, projects
from everything_ai.check import Exercise, FunctionBox, Mistake, Result, _fill, run_code


def _checks(lesson: str, name: str):
    spec = importlib.util.spec_from_file_location(name, paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


K = _checks("repro-own-compare", "everything_ai_checks_repro_own_compare_for_report")
GR = _checks("repro-guided-report", "everything_ai_checks_repro_guided_report_for_own")
A, F, B, P = K.A, K.F, K.B, K.P
short, whole, signed = B.short, B.whole, A.signed
SECTIONS = GR.SECTIONS            # Claim, Setup, Results, Differences, Not checked: the kit's report
WRITTEN = ["Claim", "Setup", "Differences", "Not checked"]   # the parts the learner writes; Results comes from the kit
REPORT = "REPORT.md"
# The kit so far: the experiment of Your reproduction 3, and the report of Guided reproduction 7; where one is missing
# or broken, the course's version stands in.
BUILDS_ON = A.BUILDS_ON + (("repro", "report", "repro-guided-report", "report"),)
projects.builds_on(*BUILDS_ON)


# --- exercise 1: the table ---------------------------------------------------------------------------------------------

_T_HEAD = ("def markdown_table(head, rows, digits):\n"
           "    # a table in Markdown: a line with the names of head, a line with --- for each\n"
           "    # column, and a line for each row; every number rounded to `digits` decimals.\n"
           "    # Each line starts and ends with |, and has \" | \" between its cells\n")
_T_BODY = ("    lines = [\"| \" + \" | \".join(head) + \" |\", \"|\" + \"---|\" * len(head)]\n"
           "    for row in rows:\n"
           "        cells = [str(round(x, digits)) for x in row]\n"
           "        lines.append(\"| \" + \" | \".join(cells) + \" |\")\n"
           "    return \"\\n\".join(lines)\n")
_T_START = _T_HEAD + "    # 1. the line of names and the line of ---\n    ...\n    # 2. a line for each row\n    ...\n"
T_CODE = _T_HEAD + _T_BODY
_T_WRONG = {
    "no-rule": T_CODE.replace("    lines = [\"| \" + \" | \".join(head) + \" |\", \"|\" + \"---|\" * len(head)]\n",
                              "    lines = [\"| \" + \" | \".join(head) + \" |\"]\n"),
    "one-rule": T_CODE.replace("\"|\" + \"---|\" * len(head)", "\"|---|\""),
    "no-round": T_CODE.replace("cells = [str(round(x, digits)) for x in row]", "cells = [str(x) for x in row]"),
    "blank-lines": T_CODE.replace("    return \"\\n\".join(lines)\n", "    return \"\\n\\n\".join(lines)\n"),
}
_HEAD = ["seed", "with dropout", "without dropout", "gain"]


def _same_table(got, want) -> bool:
    """The same table, also with spaces around the --- of the second line, as "| --- | --- |": Markdown draws both."""
    def rule(t):
        lines = t.split("\n")
        if len(lines) > 1:
            lines[1] = lines[1].replace(" ", "")
        return "\n".join(lines)
    return isinstance(got, str) and isinstance(want, str) and rule(got) == rule(want)
_VISIBLE = (_HEAD, [[0, 12.5, 12.9, 0.40000000000000036], [1, 12.9, 13.6, 0.6999999999999993]], 1)

EXERCISES = {
    "table": FunctionBox(
        name="table",
        function="markdown_table",
        start=_T_START,
        solution=T_CODE,
        visible=[_VISIBLE],
        hidden=[(["seed", "rank 8", "full", "gain"], [[0, 81.25, 80.53, 0.72], [1, 80.04, 79.33333, 0.70667], [2, 82.08, 80.01, 2.07]], 2),
                (["seed", "a", "b"], [], 1),
                (["budget", "loss"], [[100, 2.302585], [12800, 0.418713]], 3)],
        compare=_same_table,
        python_values=True,
        save_as=("own", "table"),
        first=[B._DOTS],
        mistakes=[
            Mistake("no-rule", wrong=_T_WRONG["no-rule"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: there is no line of `---` under the names. Without it, Markdown shows no table, only lines with bars in them. Add `\"|\" + \"---|\" * len(head)` right after the names.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: er staat geen regel `---` onder de namen. Zonder die regel laat Markdown geen tabel zien, alleen regels met strepen erin. Zet `\"|\" + \"---|\" * len(head)` meteen na de namen."),
            Mistake("one-rule", wrong=_T_WRONG["one-rule"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: the line of `---` needs one `---|` for each column. Repeat it: `\"|\" + \"---|\" * len(head)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de regel `---` heeft één `---|` nodig voor elke kolom. Herhaal hem: `\"|\" + \"---|\" * len(head)`."),
            Mistake("no-round", wrong=_T_WRONG["no-round"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: your numbers have all the decimals Python keeps, such as `0.40000000000000036`, the noise of subtracting two floats. Round each to `digits` decimals: `round(x, digits)`. The test error of 1000 pictures moves in steps of 0.1: more decimals say nothing.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je getallen hebben alle decimalen die Python bewaart, zoals `0.40000000000000036`, de ruis van twee floats van elkaar aftrekken. Rond elk af op `digits` decimalen: `round(x, digits)`. De testfout van 1000 afbeeldingen beweegt in stappen van 0,1: meer decimalen zeggen niets."),
            Mistake("blank-lines", wrong=_T_WRONG["blank-lines"],
                    en="`{call}` gives `{got}`, and it should give `{want}`: an empty line between the lines of a table breaks it into pieces in Markdown. Join the lines with one newline: `\"\\n\".join(lines)`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: een lege regel tussen de regels van een tabel breekt hem in Markdown in stukken. Voeg de regels samen met één regeleinde: `\"\\n\".join(lines)`."),
        ],
        texts={
            **F._KEEP,
            "empty": {"en": "Replace both `...`: first `lines` with the line of names and the line of `---`; then a line for each row, with every number rounded, and the lines joined by newlines.",
                      "nl": "Vervang beide `...`: eerst `lines` met de regel met namen en de regel met `---`; dan een regel voor elke rij, met elk getal afgerond, en de regels samengevoegd met regeleinden."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does it start with the line of names and the line of `---`, round every number to `digits` decimals, and put `\" | \"` between the cells and `|` at both ends?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Begint hij met de regel met namen en de regel met `---`, rondt hij elk getal af op `digits` decimalen, en zet hij `\" | \"` tussen de cellen en `|` aan beide kanten?"},
            "almost": {"en": "Your `markdown_table` is right for the lesson's example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for another number of columns, for no rows, and for another number of `digits`?",
                       "nl": "Je `markdown_table` klopt voor het voorbeeld van de les. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor een ander aantal kolommen, voor geen rijen, en voor een ander aantal `digits`?"},
            "kept": {"en": "Your `markdown_table` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `markdown_table` klopt. Je hebt `{saved}` zelf veranderd, dus laat de check dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
            "correct": {"en": "Your `markdown_table` writes a table at the precision your metric has. It is in your kit now, as `{saved}`, and the Results of your report use it.",
                        "nl": "Je `markdown_table` schrijft een tabel met de precisie die je metric heeft. Hij zit nu in je kit, als `{saved}`, en de Results van je verslag gebruiken hem."},
        },
    ),
}


# --- the Results, from the kit ------------------------------------------------------------------------------------------

def _table(head, rows, digits) -> str:
    """The table by the learner's markdown_table from own/table.py, or by the course's when it is not there, stops or
    gives back no text."""
    import importlib
    projects.on_path()
    projects.forget()
    try:
        out = importlib.import_module("own.table").markdown_table(head, rows, digits)
        if isinstance(out, str):
            return out
    except Exception:
        pass
    finally:
        projects.forget()
    return run_code(T_CODE).names["markdown_table"](head, rows, digits)


def digits_of(results: list[float], most: int = 4) -> int:
    """The decimals the metric has: the fewest, up to `most`, that write every result of the runs as it is. The
    course's run gives its test error with one decimal, so 1 for it."""
    for d in range(most + 1):
        if all(round(x, d) == x for x in results):
            return d
    return most


def _english(plan: dict) -> dict:
    """The plan with its words in English: for a paper of the course's list, a plan the course wrote in Dutch gets the
    words of its English plan, so that an English report has no Dutch in it."""
    if plan.get("paper") not in P.PLANS:
        return plan
    en, nl = P.course_plan(plan["paper"], "en"), P.course_plan(plan["paper"], "nl")
    out = dict(plan)
    for k in ("claim", "setup", "metric", "settings", "shown_in"):
        if plan.get(k) == nl[k]:
            out[k] = en[k]
    return out


def start_parts() -> dict:
    """What the form starts with from the plan: its claim and its setup, in English."""
    plan = P.saved_plan()
    if not plan:
        return {}
    plan = _english(plan)
    return {"Claim": str(plan.get("claim") or ""), "Setup": str(plan.get("setup") or "")}


def results_text() -> str | None:
    """The Results section, in English, from the learner's kit: the rule of the plan and its verdict at the budget, the
    runs of every seed as a table, the ablation and the share it keeps, and the comparison with the paper. None when
    the main result is not in the kit yet."""
    plan, budget, lines = P.saved_plan(), F.saved_budget(), F.my_log()
    if not plan or not budget or not lines:
        return None
    steps, n = budget["steps"], int(plan["seeds"])
    a, b = P._settings(plan)
    res = F.course_results_at(lines, steps, plan)
    if len(res[a]) < n or len(res[b]) < n:
        return None
    direction = plan.get("direction", "lower")
    g = K.gains(res[a][:n], res[b][:n], direction)
    rows = [[i, res[a][i], res[b][i], g[i]] for i in range(n)]
    d = digits_of(res[a][:n] + res[b][:n])   # the precision of the metric, as the runs give it
    said = F.my_verdict(lines, plan, steps) or F.course_verdict(lines, plan, steps)   # the plan's rule, by the course's code when the kit's stops
    words = _english(plan)
    wa, wb = P._settings(words)
    out = [f"The rule of the plan: {P.rule_text(words, 'en')} At {steps:,} steps, it says: **{said}**.", _table(["seed", wa, wb, "gain"], rows, d),
           f"Every run of the ladder of budgets is in `runs.jsonl`, one line per run."]
    reading = A.my_reading(plan, steps)
    if reading and reading.get("kept") not in (None, "error"):
        part = _part()
        share = round(100 * reading["kept"])
        out.append(f"The ablation{', without ' + part if part else ''}: it keeps {signed(share, 'en')}% of the gain; by the rule of one half, fixed "
                   f"before its runs, the result {'depends' if A.depends(reading['kept']) else 'does not depend'} on it.")
    else:
        out.append(f"*The ablation at {steps:,} steps is not in the kit yet.*")
    comp = K.my_comparison()
    numbers = K.saved_numbers()
    if isinstance(comp, dict) and numbers:
        p, s = comp["points"], comp["share"]
        where = str(numbers.get("where") or "").strip().rstrip(".")
        where = K.PAPER["where"].rstrip(".") if where == K.PAPER_WHERE["nl"].rstrip(".") else where
        out.append(f"Against the paper: {where + '. ' if where else ''}Its gain is {comp['paper_gain']:.{d + 1}f}, or {comp['paper_share']:.1f}% of its "
                   f"number for *{wb}*. The mean gain of the seeds is {p['mean']:.{d + 1}f} (95% bootstrap interval {p['low']:.{d + 1}f} to "
                   f"{p['high']:.{d + 1}f}): the difference is {'real' if p['real'] else 'not shown'}; as a share, {s['mean']:.1f}% ({s['low']:.1f}% "
                   f"to {s['high']:.1f}%): {'real' if s['real'] else 'not shown'}.")
    else:
        out.append("*The comparison with the paper is not in the kit yet.*")
    return "\n\n".join(out)


def course_results_text() -> str:
    """The Results section of the course's own report: its main result, its ablation and its comparison, from the
    course's runs, the way results_text writes the learner's."""
    plan = P.course_plan("dropout", "en")
    claim, other = K.course_results()
    g = K.gains(claim, other)
    table = run_code(T_CODE).names["markdown_table"](["seed", *plan["settings"], "gain"], [[i, claim[i], other[i], g[i]] for i in range(len(g))], 1)
    verdict = F.course_verdict(F.course_log(), plan)
    share = round(100 * A.course_kept(A.COURSE_ABLATION))
    mean, low, high = K.interval(g)
    smean, slow, shigh = K.interval(K.shares(claim, other))
    return "\n\n".join([
        f"The rule of the plan: {P.rule_text(plan, 'en')} At {F.COURSE_STEPS:,} steps, it says: **{verdict}**.", table,
        "Every run of the ladder of budgets is in `runs.jsonl`, one line per run.",
        f"The ablation, without the dropout at the input: it keeps {signed(share, 'en')}% of the gain; by the rule of one half, fixed before its runs, "
        f"the result depends on it.",
        f"Against the paper: {K.PAPER['where'].rstrip('.')}. Its gain is {K.paper_gain(K.PAPER):.2f}, or {K.paper_share(K.PAPER):.1f}% of its error without "
        f"dropout. The mean gain of the seeds is {mean:.2f} (95% bootstrap interval {low:.2f} to {high:.2f}): the difference is real; as a share, "
        f"{smean:.1f}% ({slow:.1f}% to {shigh:.1f}%): real."])


def _part() -> str:
    """PART of the learner's own/ablation.py, read from its text: the name of the part their ablation takes out."""
    path = projects.folder() / "own" / "ablation.py"
    try:
        found = re.search(r"^PART\s*=\s*[\"'](.+?)[\"']", path.read_text(encoding="utf-8"), re.MULTILINE)
        return found.group(1) if found else ""
    except OSError:
        return ""


def _report_fn():
    """report(title, parts) of the kit, Guided reproduction 7's: the learner's, or the course's in its place."""
    return run_code("from repro.report import report").names["report"]


# --- exercise 2: the report ---------------------------------------------------------------------------------------------

def report_path():
    return projects.folder() / "own" / REPORT


def parts_path():
    return projects.folder() / "own" / "report_parts.json"


def save_parts(value: dict) -> None:
    """The parts the learner wrote, so that the form shows them again and the button can write the report anew."""
    import json
    path = parts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: (value.get(k) or "").strip() for k in ["title"] + WRITTEN}
    path.write_text(json.dumps(keep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def saved_parts() -> dict | None:
    import json
    try:
        p = json.loads(parts_path().read_text(encoding="utf-8"))
        return p if isinstance(p, dict) else None
    except (OSError, ValueError):
        return None


def write_report(value: dict, results: str | None = None) -> str | None:
    """Puts the report together, the learner's parts and the Results from their kit (or `results`), with the kit's
    report, and saves it as own/REPORT.md. Gives back the Results, or None when the main result is not in the kit yet.
    The kit's report may stop with an error: that is for the caller."""
    results = results_text() if results is None else results
    parts = {name: (value.get(name) or "").strip() for name in WRITTEN}
    parts["Results"] = results or ""
    text = _report_fn()((value.get("title") or "").strip(), parts)
    path = report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return results


def report_sections(text: str) -> tuple[str, dict]:
    """The title of a report in Markdown and its parts, by the names of its "## " headings, in their order: to show
    the learner's report in tabs, as the lesson shows the course's."""
    title, *rest = re.split(r"(?m)^## ", text)
    parts = {}
    for block in rest:
        name, _, body = block.partition("\n")
        parts[name.strip()] = body.strip()
    return title.strip().removeprefix("# ").strip(), parts


def _words(text) -> int:
    return len(re.findall(r"\w+", text or ""))


@dataclass
class ReportReview(Exercise):
    """The parts of the report the learner writes, from the lesson's form: {"title", "Claim", "Setup", "Differences",
    "Not checked"}. Once each part says something, the report is put together with the kit's report and the Results
    from the kit, and saved as own/REPORT.md. With no main result in the kit yet, it is saved with its Results not
    written, and the check says so."""
    start: dict = field(default_factory=dict)
    solution: dict = field(default_factory=dict)

    def check(self, value) -> Result:
        if not isinstance(value, dict):
            return Result("error", self.message("missing"))
        start = start_parts()   # the claim and the setup of the plan, as the form starts with them
        if not any((value.get(k) or "").strip() not in ("", start.get(k, "").strip()) for k in ["title"] + WRITTEN):
            return Result("empty", self.message("empty"))
        if _words(value.get("title")) < 3:
            return self._mistake("title")
        for name in WRITTEN:
            if _words(value.get(name)) < 8:
                return self._mistake("short", part={"en": name, "nl": name})
        save_parts(value)
        results = results_text()
        try:
            write_report(value, results or "")
        except Exception:
            return self._mistake("report")
        path = report_path()
        if results is None:   # the writing is done; the Results come with the runs
            return Result("correct", self.message("no-results", saved=projects.shown(path)))
        return Result("correct", self.message("correct", saved=projects.shown(path)))

    def _mistake(self, key: str, **values) -> Result:
        m = next(m for m in self.mistakes if m.key == key)
        return Result(m.state, _fill(m.text, **values), mistake=key)


_R7 = {"en": "[Guided reproduction 7](/course/open/repro-guided-report?lang=en)", "nl": "[Begeleide reproductie 7](/course/open/repro-guided-report?lang=nl)"}
COURSE_PARTS = {
    "title": "Dropout (Srivastava and others, 2014), reproduced on Fashion-MNIST",
    "Claim": "Srivastava and others (2014): dropout, switching off neurons at random during training, reduces overfitting: a network trained with it has a clearly lower test error than without. The paper shows it on MNIST, among others: the best networks without dropout reach about 1.60%; with dropout, another network of the same kind reaches 1.35%.",
    "Setup": "The same network in both settings: a multilayer perceptron of 784, 512, 512 and 10 units with ReLU, on the first 6000 training pictures of Fashion-MNIST, tested on 1000. SGD with momentum 0.9 and a learning rate of 0.05 with a cosine decay over the steps of each run, batches of 100. With dropout: 0.2 at the input and 0.5 in the hidden layers, as in the paper; without: none. Seeds 0, 1 and 2; a budget of 12,800 steps, fixed before the runs, and its seven halvings.",
    "Differences": "On purpose: Fashion-MNIST instead of MNIST, 6000 training pictures instead of 60,000, a smaller network with ReLU where the paper's two networks have logistic units, and no max-norm constraint, so that every run fits on a laptop. The paper's two MNIST numbers belong to two different networks, without a spread over runs; mine are three seeds of one network. My errors are eight to nine times the paper's, so the gain is larger in points and smaller as a share of the error.",
    "Not checked": "MNIST itself, and the paper's own networks and training. Larger networks, such as the paper's of 1024 to 8192 units a layer, with which it reaches its lowest errors. More seeds: with three, the interval of the gain cannot reach past the smallest and the largest gain of the seeds. Other budgets than the ladder of 12,800 steps.",
}

EXERCISES["report"] = ReportReview(
    name="report",
    start={"title": "", "Claim": "", "Setup": "", "Differences": "", "Not checked": ""},
    solution=COURSE_PARTS,
    mistakes=[
        Mistake("title",
                en="Give your report a title of a few words: the paper, with its authors and year, and what you reproduced it on.",
                nl="Geef je verslag een titel van een paar woorden: de paper, met zijn auteurs en jaar, en waarop je hem reproduceerde."),
        Mistake("short",
                en="The part *{part}* says too little yet. Write at least a sentence or two: a reader who has only your report should be able to follow it.",
                nl="Het deel *{part}* zegt nog te weinig. Schrijf minstens een zin of twee: een lezer die alleen je verslag heeft, moet het kunnen volgen."),
        Mistake("report", state="error",
                en="The `report` of your kit stops with an error, `work/projects/repro/report.py` of " + _R7["en"] + ". Did you change it after its check? Then check it again there.",
                nl="De `report` van je kit stopt met een fout, `work/projects/repro/report.py` van " + _R7["nl"] + ". Heb je hem veranderd na zijn check? Check hem daar dan opnieuw."),
    ],
    texts={
        "missing": {"en": "I cannot see the form. Run the cell with the form again.",
                    "nl": "Ik zie het formulier niet. Voer de cel met het formulier opnieuw uit."},
        "empty": {"en": "Write the title and the four parts of your report that the kit cannot write for you: Claim, Setup, Differences and Not checked.",
                  "nl": "Schrijf de titel en de vier delen van je verslag die de kit niet voor je kan schrijven: Claim, Setup, Differences en Not checked."},
        "no-results": {"en": "Your report is saved, as `{saved}`, without its Results yet: the runs of your main result at your budget are not in your kit. Run them in [Your reproduction 5](/course/open/repro-own-figure?lang=en), and press the button below: then the report gets them.",
                       "nl": "Je verslag is bewaard, als `{saved}`, nog zonder zijn Results: de runs van je hoofdresultaat bij je budget zitten niet in je kit. Draai ze in [Je reproductie 5](/course/open/repro-own-figure?lang=nl), en druk op de knop hieronder: dan krijgt het verslag ze."},
        "correct": {"en": "Your report is in your kit now, as `{saved}`: your four parts, and the Results from what your kit has now.",
                    "nl": "Je verslag zit nu in je kit, als `{saved}`: je vier delen, en de Results uit wat je kit nu heeft."},
    },
)

REQUIRED = ["table", "report", "to-authors"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("table", _T_START, "empty", None),
    ("table", T_CODE, "correct", None),
] + [("table", code, "wrong", key) for key, code in _T_WRONG.items()] + [
    ("report", None, "error", None),
    ("report", {"title": "", "Claim": "", "Setup": "", "Differences": "", "Not checked": ""}, "empty", None),
    ("report", {**COURSE_PARTS, "title": "Dropout"}, "wrong", "title"),
    ("report", {**COURSE_PARTS, "Not checked": "Nothing."}, "wrong", "short"),
]


# --- the demo: what goes public ---------------------------------------------------------------------------------------

SHARED = ["experiment.py", "contract.py", "sanity.py", "plan.py", "runner.py", "results.py", "ablation.py", "kept.py", "compare.py", "table.py",
          "paper.json", "plan.json", "budget.json", "paper_numbers.json", "runs.jsonl", "ablation.jsonl", "report_parts.json", REPORT]


def manifest() -> list[str] | str | None:
    """Writes the report anew from the parts the learner saved, with the Results their kit has now, and then the files
    of the reproduction that go public, from work/projects/own/, each as a line of its SHA-256 and its name, as
    sha256sum writes them, in the order of sorted names; saved as own/MANIFEST.txt. None when there are no parts yet,
    "report" when the kit's report stops."""
    folder = projects.folder() / "own"
    parts = saved_parts()
    if parts is None:
        return None
    try:
        write_report(parts)
    except Exception:
        return "report"
    files = {name: (folder / name).read_bytes() for name in SHARED if (folder / name).is_file()}   # the bytes, as sha256sum reads them
    lines = [hashlib.sha256(files[name]).hexdigest() + "  " + name for name in sorted(files)]
    (folder / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
