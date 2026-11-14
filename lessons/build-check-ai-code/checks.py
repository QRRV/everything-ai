# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-check-ai-code, and the course's run of Qwen3-0.6B writing two small functions
(data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS); what the 16 functions give,
their picture, and the exercise in which the learner writes tests that catch a bug."""
import ast
import html
import json
import re
from dataclasses import dataclass

from everything_ai import paths, yours
from everything_ai.check import MISSING, BugFix, CodeBox, Mistake, Result, Sort, is_empty_text, run_code

COURSE = "data/course-work-qwen.json"
SEEDS = 8
MAX_TOKENS = 300
REQUESTS = {
    "vowels": "Write a Python function count_vowels(text) that counts the vowels (a, e, i, o, u) in a text. Reply with the code only.",
    "palindrome": ("Write a Python function is_palindrome(text) that says whether a text reads the same backwards, "
                   "ignoring capitals and spaces. Reply with the code only."),
}
ASKS = {k: {"messages": [{"role": "user", "content": r}], "max_tokens": MAX_TOKENS} for k, r in REQUESTS.items()}


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-check-ai-code"]


# --- what the 16 functions give ------------------------------------------------------------------------------------------

FUNCTIONS = {"vowels": "count_vowels", "palindrome": "is_palindrome"}
INPUTS = {"vowels": ["hello", "Apple", "AEIOU", "", "rhythm", "Banana bread"],
          "palindrome": ["racecar", "Taco cat", "Never odd or even", "hello", "", "Ab ba"]}
WANT = {"vowels": [2, 2, 5, 0, 0, 5], "palindrome": [True, True, True, False, True, True]}
# What each function gives for INPUTS, in that order. tests/test_build_check_ai_code.py runs every function again.
OUTPUTS = {
    **{f"vowels:{s}": [2, 1, 0, 0, 0, 5] for s in range(8)},
    "palindrome:0": [True, False, False, False, True, True], "palindrome:1": [True, True, True, False, True, True],
    "palindrome:2": [True, True, True, False, True, True], "palindrome:3": [True, True, True, False, True, True],
    "palindrome:4": [True, True, True, False, True, True], "palindrome:5": [True, False, False, False, True, True],
    "palindrome:6": [True, True, True, False, True, True], "palindrome:7": [True, True, True, False, True, True],
}
# All eight count_vowels are this one, character for character.
QWEN_VOWELS = "def count_vowels(text):\n    vowels = {'a', 'e', 'i', 'o', 'u'}\n    return sum(1 for char in text if char in vowels)\n"
RIGHT_VOWELS = QWEN_VOWELS.replace("'u'}", "'u', 'A', 'E', 'I', 'O', 'U'}")


def program(key: str) -> str:
    """The code of one function: the first Python block of its answer."""
    kind, seed = key.split(":")
    return re.search(r"```python\n(.*?)```", course()[kind]["answers"][int(seed)]["text"], re.S).group(1)


def oks(key: str) -> list:
    kind = key.split(":")[0]
    return [got == want for got, want in zip(OUTPUTS[key], WANT[kind])]


def right(kind: str, j: int) -> int:
    return sum(oks(f"{kind}:{s}")[j] for s in range(SEEDS))


def findings() -> dict:
    keep = [s for s in range(SEEDS) if "c == ' '" in program(f"palindrome:{s}")]
    return {
        "same_vowels": sum(program(f"vowels:{s}") == QWEN_VOWELS for s in range(SEEDS)),
        "vowels_right": sum(all(oks(f"vowels:{s}")) for s in range(SEEDS)),
        "keeps_spaces": keep,
        "says_removes": sum("Remove all non-alphanumeric characters" in program(f"palindrome:{s}") for s in keep),
        "palindrome_right": sum(all(oks(f"palindrome:{s}")) for s in range(SEEDS)),
        "lucky": right("palindrome", 5),
    }


def _shown(v, lang: str) -> str:
    if isinstance(v, bool):
        return {True: {"en": "yes", "nl": "ja"}, False: {"en": "no", "nl": "nee"}}[v][lang]
    return str(v)


_HEADS = {"vowels": {"en": "count_vowels: how many vowels?", "nl": "count_vowels: hoeveel klinkers?"},
          "palindrome": {"en": "is_palindrome: the same backwards?", "nl": "is_palindrome: achterstevoren hetzelfde?"}}


def checks_grid(lang: str) -> str:
    """For each function, its six inputs as rows: the input, the right answer, and how many of the eight versions
    Qwen3 wrote give it; a row where some got it wrong in the colour of mistakes, on its wash."""
    X1, X2, RH = 190, 300, 22
    of = {"en": "of", "nl": "van"}[lang]
    heads = {"en": ["input", "right", "Qwen3 right"], "nl": ["invoer", "goed", "Qwen3 goed"]}[lang]
    parts, y = [], 0
    for kind in FUNCTIONS:
        parts.append(f'<text x="0" y="{y + 16}" style="font-size:13px;fill:var(--ea-ink);font-weight:600;font-family:var(--ea-font-code)">'
                     f'{_HEADS[kind][lang]}</text>')
        y += 24
        for x, h in zip((0, X1, X2), heads):
            parts.append(f'<text x="{x}" y="{y + 12}" style="font-size:11px;fill:var(--ea-ink-2)">{h}</text>')
        y += 18
        for j, (given, want) in enumerate(zip(INPUTS[kind], WANT[kind])):
            k = right(kind, j)
            shown = html.escape(repr(given)).replace("&#x27;", '"').replace("'", '"')
            parts.append(f'<text x="0" y="{y + 15}" style="font-size:12px;fill:var(--ea-ink);font-family:var(--ea-font-code);white-space:pre">{shown}</text>'
                         f'<text x="{X1}" y="{y + 15}" style="font-size:12px;fill:var(--ea-ink)">{_shown(want, lang)}</text>')
            if k < SEEDS:
                parts.append(f'<rect x="{X2 - 6}" y="{y + 2}" width="76" height="{RH - 4}" rx="3" style="fill:var(--ea-loss-wash)"/>')
            style = "fill:var(--ea-loss-text);font-weight:600" if k < SEEDS else "fill:var(--ea-ink)"
            parts.append(f'<text x="{X2}" y="{y + 15}" style="font-size:12px;{style};font-variant-numeric:tabular-nums">{k} {of} 8</text>')
            y += RH
        y += 14
    W = X2 + 80
    title = {"en": "How many of the eight versions of each function give the right answer", "nl": "Hoeveel van de acht versies van elke functie het goede antwoord geven"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:26rem">'
            f'<svg viewBox="0 0 {W} {y}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(parts)}</svg></figure>')


def grid_after(lang: str) -> str:
    F = findings()
    assert F == {"same_vowels": 8, "vowels_right": 0, "keeps_spaces": [0, 5], "says_removes": 2, "palindrome_right": 6, "lucky": 8}
    if lang == "nl":
        return ("Alle acht keer schreef Qwen3 precies dezelfde count_vowels, en alle acht tellen alleen kleine letters: "
                "\"Apple\" geeft 1 en \"AEIOU\" 0. Een test met alleen kleine letters, zoals \"hello\", vangt dat nooit. "
                f"Van de acht is_palindrome laten er {len(F['keeps_spaces'])} de spaties staan, terwijl het commentaar erboven zegt "
                "*Remove all non-alphanumeric characters* (haal alles weg wat geen letter of cijfer is): \"Taco cat\" en \"Never odd "
                "or even\" geven bij hen nee. En \"Ab ba\" geeft bij alle acht ja, ook bij die twee: de spatie staat precies in het "
                "midden, dus de test slaagt bij toeval.")
    return ("All eight times, Qwen3 wrote exactly the same count_vowels, and all eight count small letters only: \"Apple\" "
            "gives 1 and \"AEIOU\" 0. A test with only small letters, such as \"hello\", never catches that. Of the eight "
            f"is_palindrome, {len(F['keeps_spaces'])} keep the spaces, while the comment above says *Remove all non-alphanumeric "
            "characters*: \"Taco cat\" and \"Never odd or even\" give no for them. And \"Ab ba\" gives yes for all eight, those "
            "two too: the space is exactly in the middle, so the test passes by luck.")


def program_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th function (1 to 16): its code, and what it gives for the six inputs, each marked right or wrong."""
    key = list(OUTPUTS)[int(i) - 1]
    kind, seed = key.split(":")
    rows = "".join(
        f'<tr><td style="font-family:var(--ea-font-code);padding:0 0.8rem 0 0;white-space:pre">{html.escape(repr(given))}</td>'
        f'<td style="padding:0 0.6rem 0 0;color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"};font-weight:{400 if ok else 600}">'
        f'{_shown(got, lang)}</td><td style="color:var(--ea-loss-text)">{"" if ok else "✗"}</td></tr>'
        for given, got, ok in zip(INPUTS[kind], OUTPUTS[key], oks(key)))
    body = (f'<div style="max-width:42rem;font-family:var(--ea-font-ui)">'
            f'<pre style="font-family:var(--ea-font-code);font-size:0.8rem;background:var(--ea-inset);padding:0.5rem 0.7rem;'
            f'border-radius:6px;overflow-x:auto;margin:0 0 0.4rem">{html.escape(program(key).rstrip())}</pre>'
            f'<table style="font-size:0.85rem;border-collapse:collapse">{rows}</table></div>')
    n = sum(oks(key))
    said = {"en": f"**{FUNCTIONS[kind]}, version {int(seed) + 1} of 8:** {n} of the 6 right.",
            "nl": f"**{FUNCTIONS[kind]}, versie {int(seed) + 1} van 8:** {n} van de 6 goed."}[lang]
    return body, said


# --- the exercises -------------------------------------------------------------------------------------------------------

def _function(code: str, name: str):
    names = {}
    exec(code, names)
    return names[name]


def _assert_lines(text: str) -> list:
    return [line.strip() for line in text.splitlines() if re.match(r"\s*assert\b", line)]


def _one_line(code: str) -> str:
    return " ".join(part.strip() for part in code.splitlines() if part.strip())


def _statement_at(text: str, line: int | None) -> str:
    """The whole statement that holds `line`, on one line: also an assert written over several lines."""
    if not line:
        return ""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.splitlines()[line - 1].strip() if 0 < line <= len(text.splitlines()) else ""
    inner = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and not isinstance(n, ast.FunctionDef)
             and n.lineno <= line <= (n.end_lineno or n.lineno)]
    if not inner:
        return ""
    node = min(inner, key=lambda n: (n.end_lineno or n.lineno) - n.lineno)
    return _one_line(ast.get_source_segment(text, node) or "")


def _tests_called(text: str) -> str:
    """The text, with a call after it of each test function it defines, as pytest would call them: `def test_...():`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    calls = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test") and not n.args.args]
    return text.rstrip("\n") + "\n" + "".join(f"{c}()\n" for c in calls) if calls else text


def _unchecked(text: str, name: str) -> str:
    """A statement that calls the function but checks nothing, such as `count_vowels("Apple") == 2` without `assert`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == name
                                              for c in ast.walk(node)):
            return _one_line(ast.get_source_segment(text, node) or "")
    return ""


def _passes(code: str, fn, name: str) -> bool | None:
    """Whether a piece of the tests passes for fn: True, False when an assert fails, None when it stops otherwise."""
    run = run_code(code, {name: fn})
    if run.error is None and not run.timeout:
        return True
    return False if isinstance(run.error, AssertionError) else None


def tests_picture(seen) -> str:
    """Each test the learner wrote (an assert, or a test function), and whether it passes for a right count_vowels and
    for Qwen3's. A test runs after the other lines above it, such as `n = count_vowels("Apple")`."""
    if not isinstance(seen.text, str):
        return ""
    try:
        tree = ast.parse(seen.text)
    except SyntaxError:
        return ""
    tests, before = [], []
    for node in tree.body:
        code = ast.get_source_segment(seen.text, node) or ""
        if isinstance(node, ast.Assert):
            tests.append((_one_line(code), "\n".join(before + [code])))
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test") and not node.args.args:
            tests.append((f"{node.name}()", "\n".join(before + [code, f"{node.name}()"])))
        else:
            before.append(code)
    if not tests:
        return ""
    lang = seen.lang
    marks = {True: {"en": "passes", "nl": "slaagt"}, False: {"en": "fails", "nl": "faalt"}, None: {"en": "error", "nl": "fout"}}
    right_fn, qwen_fn = _function(RIGHT_VOWELS, "count_vowels"), _function(QWEN_VOWELS, "count_vowels")
    rows = []
    for shown, code in tests[:8]:
        cells = []
        for fn in (right_fn, qwen_fn):
            p = _passes(code, fn, "count_vowels")
            colour = "var(--ea-ink)" if p is True else "var(--ea-loss-text)"   # where a mistake shows: in the test or in the code
            cells.append(f'<td style="padding:0.1rem 0.8rem 0.1rem 0;color:{colour};font-weight:600">{marks[p][lang]}</td>')
        rows.append(f'<tr><td style="font-family:var(--ea-font-code);font-size:0.8rem;padding:0.1rem 0.8rem 0.1rem 0;white-space:pre">'
                    f'{html.escape(shown)}</td>{"".join(cells)}</tr>')
    head = {"en": ["your test", "right count_vowels", "Qwen3's count_vowels"], "nl": ["je test", "goede count_vowels", "count_vowels van Qwen3"]}[lang]
    table = (f'<table style="font-size:0.85rem;border-collapse:collapse"><tr>'
             + "".join(f'<th style="text-align:left;font-weight:400;color:var(--ea-ink-2);padding:0 0.8rem 0.2rem 0">{h}</th>' for h in head)
             + f'</tr>{"".join(rows)}</table>')
    title = {"en": "Your tests on two versions of count_vowels", "nl": "Je tests op twee versies van count_vowels"}[lang]
    return yours.figure(title, table)


@dataclass
class TestsBox(CodeBox):
    """The learner writes tests of a function: assert lines, or test functions as pytest has them (`def test_...():`),
    which the check calls. It runs them twice: on a right version, where every one must pass, and on a version with a
    bug, where at least one must fail. Texts: missing, empty, cleared, no_assert, fails_right ({line}), unchecked
    ({line}), misses, slow, correct ({line})."""
    function: str = ""
    right: str = ""                    # the right version, as code text
    buggy: str = ""                    # the version with the bug, as code text

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start):
            cleared = text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.mistakes:
            found = re.search(m.syntax, text, flags=re.MULTILINE) if m.syntax else None
            if found:
                line = text[text.rfind("\n", 0, found.start()) + 1:].split("\n", 1)[0].strip()
                return Result("error", {k: v.replace("{line}", line) for k, v in m.text.items()}, mistake=m.key)
        if not _assert_lines(text):
            return Result("wrong", self.message("no_assert"), mistake="no-assert")
        code = _tests_called(text)
        good = run_code(code, {self.function: _function(self.right, self.function)})
        if good.timeout:
            return Result("slow", self.message("slow"))
        if isinstance(good.error, AssertionError):
            return Result("wrong", self.message("fails_right", line=_statement_at(text, good.line)), mistake="fails-right")
        if good.error is not None:
            return Result("error", self._plain_error(good, text))
        bad = run_code(code, {self.function: _function(self.buggy, self.function)})
        if bad.timeout:
            return Result("slow", self.message("slow"))
        if not isinstance(bad.error, AssertionError):
            loose = _unchecked(text, self.function)
            if loose:
                return Result("wrong", self.message("unchecked", line=loose), mistake="not-asserted")
            return Result("wrong", self.message("misses"), mistake="misses-bug")
        return Result("correct", self.message("correct", line=_statement_at(text, bad.line)))


_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("one-equals", syntax=r"^[ \t]*assert\b(?![^\n]*(?:==|[<>!]=))[^\n]*\)[ \t]*=[^=\n]*$",
            en="The line `{line}` has one `=`. That puts a value in a name; to ask whether two things are equal, Python uses `==`.",
            nl="De regel `{line}` heeft één `=`. Daarmee zet je een waarde in een naam; om te vragen of twee dingen gelijk zijn, gebruikt Python `==`."),
]
_START_TESTS = 'assert count_vowels("hello") == 2\n'
_SOLUTION_TESTS = 'assert count_vowels("hello") == 2\nassert count_vowels("Apple") == 2\n'

EXERCISES = {
    "catch-vowels": TestsBox(
        name="catch-vowels",
        picture=tests_picture,
        function="count_vowels",
        right=RIGHT_VOWELS,
        buggy=QWEN_VOWELS,
        start=_START_TESTS,
        solution=_SOLUTION_TESTS,
        mistakes=_SYNTAX,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "This one test passes for Qwen3's count_vowels too. Add a test, a line with `assert`, that passes for a right count_vowels and fails for Qwen3's.",
                      "nl": "Deze ene test slaagt ook voor de count_vowels van Qwen3. Voeg een test toe, een regel met `assert`, die slaagt voor een goede count_vowels en faalt voor die van Qwen3."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first test again.",
                        "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste test."},
            "no_assert": {"en": "Your box has no line with `assert`. A test here is one line: `assert`, then a call of `count_vowels`, `==`, and the number it must give.",
                          "nl": "Je vak heeft geen regel met `assert`. Een test is hier één regel: `assert`, dan een aanroep van `count_vowels`, `==`, en het getal dat hij moet geven."},
            "fails_right": {"en": "The test `{line}` fails for a right count_vowels too, so the test itself is wrong. Count the vowels in that text again, by hand.",
                            "nl": "De test `{line}` faalt ook voor een goede count_vowels, dus de test zelf klopt niet. Tel de klinkers in die tekst nog eens, met de hand."},
            "unchecked": {"en": "The line `{line}` runs count_vowels, but checks nothing: only a line that starts with `assert` stops when the answer is wrong. Put `assert` in front of it.",
                          "nl": "De regel `{line}` voert count_vowels uit, maar controleert niets: alleen een regel die met `assert` begint, stopt als het antwoord niet klopt. Zet er `assert` voor."},
            "misses": {"en": "All your tests pass for a right count_vowels, and they all pass for Qwen3's too: they do not catch its mistake. Look at the table above the exercise: which inputs did all eight versions get wrong?",
                       "nl": "Al je tests slagen voor een goede count_vowels, en ze slagen ook allemaal voor die van Qwen3: ze vangen zijn fout niet. Kijk naar de tabel boven de oefening: welke invoer hadden alle acht versies fout?"},
            "slow": {"en": "Your tests are still not done after a few seconds. Does something keep going forever?",
                     "nl": "Je tests zijn na een paar seconden nog niet klaar. Gaat er iets eindeloos door?"},
            "correct": {"en": "Your test `{line}` fails for Qwen3's count_vowels and passes for a right one: a test that catches the bug. Keep it: after every change to the function, it runs again.",
                        "nl": "Je test `{line}` faalt voor de count_vowels van Qwen3 en slaagt voor een goede: een test die de bug vangt. Bewaar hem: na elke verandering aan de functie draait hij opnieuw."},
        },
    ),
    "fix-vowels": BugFix(
        name="fix-vowels",
        picture=yours.steps(call=0),
        function="count_vowels",
        start=QWEN_VOWELS,
        solution=RIGHT_VOWELS,
        visible=[("Apple",)],
        hidden=[("AEIOU",), ("hello",), ("",), ("rhythm",), ("Banana bread",), ("Ice cream",)],
        mistakes=[
            Mistake("only-capitals", wrong=QWEN_VOWELS.replace("{'a', 'e', 'i', 'o', 'u'}", "{'A', 'E', 'I', 'O', 'U'}"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: now it counts only the capitals. Count the small vowels and the capitals.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: nu telt hij alleen de hoofdletters. Tel de kleine klinkers en de hoofdletters."),
        ],
        texts={
            "missing": {"en": "I cannot find `count_vowels`. Run the cell with the function again; if it shows an error, fix that first.",
                        "nl": "Ik vind `count_vowels` niet. Voer de cel met de functie opnieuw uit; laat die een fout zien, los die dan eerst op."},
            "empty_error": {"en": "This function has a bug. The check calls `{call}`, and Python stops with this error:\n\n```text\n{traceback}\n```",
                            "nl": "Deze functie heeft een bug. De check roept `{call}` aan, en Python stopt met deze fout:\n\n```text\n{traceback}\n```"},
            "empty_wrong": {"en": "This is Qwen3's count_vowels, and it gives no error: `{call}` gives `{got}`, and it should give `{want}`. Fix it in the cell above, and run the cell.",
                            "nl": "Dit is de count_vowels van Qwen3, en hij geeft geen fout: `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Verbeter hem in de cel hierboven, en voer de cel uit."},
            "error": {"en": "With `{call}`, your function stops with an error:\n\n```text\n{traceback}\n```\n\nRead the last line first, then the line of your cell just above it.",
                      "nl": "Met `{call}` stopt je functie met een fout:\n\n```text\n{traceback}\n```\n\nLees eerst de laatste regel, en dan de regel uit je cel er net boven."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: A and e. Which letters does `vowels` hold?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: A en e. Welke letters zitten er in `vowels`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function count every vowel, small and capital?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Telt je functie elke klinker, klein en hoofdletter?"},
            "correct": {"en": "Bug fixed: your count_vowels is right in all {n} cases, capitals included. A test like `assert count_vowels(\"Apple\") == 2` passes now too.",
                        "nl": "Bug opgelost: je count_vowels klopt in alle {n} gevallen, hoofdletters ook. Een test zoals `assert count_vowels(\"Apple\") == 2` slaagt nu ook."},
        },
    ),
    "which-check": Sort(
        name="which-check",
        groups={
            "test": {"en": "A test", "nl": "Een test"},
            "review": {"en": "A review", "nl": "Een review"},
            "static": {"en": "Static analysis", "nl": "Static analysis"},
        },
        items={
            "apple": {"en": "**Apple.** `count_vowels(\"Apple\")` gives 1.",
                      "nl": "**Apple.** `count_vowels(\"Apple\")` geeft 1."},
            "unused": {"en": "**Never used.** A name gets a value on line 3, and no line ever uses it.",
                       "nl": "**Nooit gebruikt.** Een naam krijgt op regel 3 een waarde, en geen enkele regel gebruikt hem."},
            "more": {"en": "**More than asked.** The function also removes commas and dots, which the spec did not ask for.",
                     "nl": "**Meer dan gevraagd.** De functie haalt ook komma's en punten weg, waar de spec niet om vroeg."},
            "taco": {"en": "**Taco cat.** When you run it, `is_palindrome(\"Taco cat\")` gives False.",
                     "nl": "**Taco cat.** Als je hem uitvoert, geeft `is_palindrome(\"Taco cat\")` False."},
            "undefined": {"en": "**Unknown name.** A line uses a name Python does not know, in an `if` that no test reaches.",
                          "nl": "**Onbekende naam.** Een regel gebruikt een naam die Python niet kent, in een `if` waar geen test komt."},
            "why": {"en": "**37.** The code works, but nobody can tell why the limit is 37.",
                    "nl": "**37.** De code werkt, maar niemand kan zien waarom de grens 37 is."},
        },
        solution={"apple": "test", "unused": "static", "more": "review", "taco": "test", "undefined": "static", "why": "review"},
        nudge={
            "apple": {"en": "Does running the function on one input show it?", "nl": "Zie je het als je de functie op één invoer uitvoert?"},
            "taco": {"en": "Does running the function on one input show it?", "nl": "Zie je het als je de functie op één invoer uitvoert?"},
            "unused": {"en": "Can a tool see it by reading the code, without running it and without knowing what it is for?",
                       "nl": "Kan een tool het zien door de code te lezen, zonder hem uit te voeren en zonder te weten waar hij voor is?"},
            "undefined": {"en": "Can a tool see it by reading the code, without running it and without knowing what it is for?",
                          "nl": "Kan een tool het zien door de code te lezen, zonder hem uit te voeren en zonder te weten waar hij voor is?"},
            "more": {"en": "Does it take a person who knows what the code is for?", "nl": "Is er iemand voor nodig die weet waar de code voor is?"},
            "why": {"en": "Does it take a person who knows what the code is for?", "nl": "Is er iemand voor nodig die weet waar de code voor is?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six problems again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes problemen opnieuw uit."},
            "empty": {"en": "Pick for each of the six problems the check that finds it.",
                      "nl": "Kies voor elk van de zes problemen de controle die het vindt."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. A test runs the code on a case and sees a wrong answer; static analysis reads the code without running it and finds what is wrong whatever it is for; a review takes a person who knows what it is for. Each finds what the others miss, so you use all three.",
                        "nl": "Alle {n} kloppen. Een test voert de code uit op een geval en ziet een verkeerd antwoord; static analysis leest de code zonder hem uit te voeren en vindt wat fout is, waar hij ook voor is; een review vraagt iemand die weet waar hij voor is. Elk vindt wat de andere missen, dus je gebruikt alle drie."},
        },
    ),
}

# The lesson is done when these are correct.
REQUIRED = ["catch-vowels", "fix-vowels", "which-check"]

# What the tests feed each check, and what it must say: (exercise, what the learner left, state, known mistake).
_WHICH = {"apple": "test", "unused": "static", "more": "review", "taco": "test", "undefined": "static", "why": "review"}
CASES = [
    ("catch-vowels", _START_TESTS, "empty", None),
    ("catch-vowels", "", "empty", None),
    ("catch-vowels", _SOLUTION_TESTS, "correct", None),
    ("catch-vowels", 'assert count_vowels("AEIOU") == 5\n', "correct", None),
    ("catch-vowels", 'assert count_vowels("hello") == 2\nassert count_vowels("rhythm") == 0\n', "wrong", "misses-bug"),
    ("catch-vowels", 'assert count_vowels("Apple") == 1\n', "wrong", "fails-right"),
    ("catch-vowels", 'print(count_vowels("Apple"))\n', "wrong", "no-assert"),
    ("catch-vowels", 'assert count_vowels("Apple") = 2\n', "error", "one-equals"),
    ("catch-vowels", _START_TESTS + 'count_vowels("Apple") == 2\n', "wrong", "not-asserted"),
    ("catch-vowels", 'n = count_vowels("Apple")\nassert n == 2\n', "correct", None),
    ("catch-vowels", 'assert (count_vowels("Apple")\n        == 2)\n', "correct", None),
    ("catch-vowels", 'def test_apple():\n    assert count_vowels("Apple") == 2\n', "correct", None),
    ("catch-vowels", 'assert count_vowels(text="Apple") == 2\n', "correct", None),
    ("catch-vowels", 'assert count_vowels(text="Apple")\n', "wrong", "misses-bug"),
    ("fix-vowels", RIGHT_VOWELS, "correct", None),
    ("fix-vowels", QWEN_VOWELS.replace("char in text", "char in text.lower()"), "correct", None),
    ("fix-vowels", QWEN_VOWELS.replace("{'a', 'e', 'i', 'o', 'u'}", "{'A', 'E', 'I', 'O', 'U'}"), "wrong", "only-capitals"),
    ("fix-vowels", QWEN_VOWELS.replace("'u'}", "'u', 'A'}"), "almost", None),
    ("which-check", {}, "empty", None),
    ("which-check", _WHICH, "correct", None),
    ("which-check", {"apple": "test"}, "wrong", "unfinished"),
    ("which-check", {**_WHICH, "unused": "test"}, "wrong", "wrong:unused"),
    ("which-check", {**_WHICH, "more": "static"}, "wrong", "wrong:more"),
    ("which-check", {**_WHICH, "taco": "review"}, "wrong", "wrong:taco"),
]
