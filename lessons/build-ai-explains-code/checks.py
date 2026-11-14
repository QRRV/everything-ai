# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-ai-explains-code, and the course's run of Qwen3-0.6B explaining a function and renaming
its names (data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS); what the 24 answers
say, and their picture."""
import html
import json
import re

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

COURSE = "data/course-work-qwen.json"
SEEDS = 8
AVERAGE = ("def average_without_extremes(scores):\n    scores = sorted(scores)\n    middle = scores[1:-1]\n"
           "    return sum(middle) / len(scores)\n")
COUNT = ("def f(a, b):\n    c = 0\n    for d in a:\n        if d > b:\n            c = c + 1\n    return c\n")


def _code(text: str) -> str:
    return "```python\n" + text + "```"


ASKS = {
    "explain": {"messages": [{"role": "user", "content": "Explain in two sentences what this function returns.\n\n" + _code(AVERAGE)}],
                "max_tokens": 150},
    "value": {"messages": [{"role": "user", "content": "What does average_without_extremes([1, 2, 3, 4, 100]) return? "
                                                        "Reply with the number only.\n\n" + _code(AVERAGE)}],
              "max_tokens": 300},
    "rename": {"messages": [{"role": "user", "content": "Give the function and its variables clear names. "
                                                         "Reply with the code only.\n\n" + _code(COUNT)}],
               "max_tokens": 200},
}


def course() -> dict:
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-ai-explains-code"]


# --- what the 24 answers say ---------------------------------------------------------------------------------------------

CASE = [1, 2, 3, 4, 100]
# The final answer of each "value" run, as it wrote it: the last number in its answer (tests check them against the texts).
VALUES = ["3", "3", "1.25", "3", "1", "10/3", "3", "1.8"]
_DIVISION = re.compile(r"divid|len\(scores\)|number of (all )?(the )?scores|\bfive\b|\b5\b")


def answer(key: str, seed: int) -> str:
    return course()[key]["answers"][seed]["text"]


def final(text: str) -> str:
    """The last number in an answer, such as 3, 1.25 or 10/3."""
    return re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", text)[-1]


def returns() -> float:
    """What average_without_extremes really returns for CASE."""
    names = {}
    exec(AVERAGE, names)
    return names["average_without_extremes"](list(CASE))


def sees_bug(seed: int) -> bool:
    """Whether an explanation says that the function divides by the number of all the scores."""
    return bool(_DIVISION.search(answer("explain", seed)))


def renamed(seed: int) -> bool:
    """Whether a rename answer gives code in which f, a, b, c or d has another name."""
    m = re.search(r"```python\n(.*?)```", answer("rename", seed), re.S)
    return bool(m) and "def f(a, b):" not in m.group(1)


def says_clear(seed: int) -> bool:
    """Whether a rename answer says the code has clear names, while it gives the same code back."""
    return "clear names" in answer("rename", seed) and not renamed(seed) and "def f(a, b):" in answer("rename", seed)


def findings() -> dict:
    right = f"{returns():g}"
    return {
        "returns": right,
        "sees_bug": sum(sees_bug(s) for s in range(SEEDS)),
        "right": sum(v == right for v in VALUES),
        "three": VALUES.count("3"),
        "renamed": sum(renamed(s) for s in range(SEEDS)),
        "says_clear": sum(says_clear(s) for s in range(SEEDS)),
    }


_ROWS = {
    "explain": {"en": "Explain what it returns: does the explanation see the bug?", "nl": "Leg uit wat hij teruggeeft: ziet de uitleg de bug?"},
    "value": {"en": "What does it return for [1, 2, 3, 4, 100]? Right: 1.8", "nl": "Wat geeft hij terug voor [1, 2, 3, 4, 100]? Goed: 1,8"},
    "rename": {"en": "Give it clear names: did it rename anything?", "nl": "Geef hem duidelijke namen: hernoemde het iets?"},
}
_NO = {"en": "no", "nl": "nee"}


def _cells(key: str, lang: str) -> list:
    """(text, right) for each of the eight runs of one question."""
    if key == "explain":
        return [(_NO[lang] if not sees_bug(s) else {"en": "yes", "nl": "ja"}[lang], sees_bug(s)) for s in range(SEEDS)]
    if key == "value":
        right = findings()["returns"]
        return [(v.replace(".", ",") if lang == "nl" else v, v == right) for v in VALUES]
    return [(_NO[lang] if not renamed(s) else {"en": "yes", "nl": "ja"}[lang], renamed(s)) for s in range(SEEDS)]


def runs_grid(lang: str) -> str:
    """The three questions as rows and the eight runs as cells; a wrong answer in the colour of mistakes, on its wash,
    the one right answer outlined in ink."""
    CW, CH, X0 = 64, 26, 0
    parts = []
    for s in range(SEEDS):
        parts.append(f'<text x="{X0 + s * CW + CW / 2}" y="14" text-anchor="middle" style="font-size:11px;fill:var(--ea-ink-2)">'
                     f'{ {"en": "run", "nl": "run"}[lang] } {s + 1}</text>')
    y = 24
    for key in _ROWS:
        parts.append(f'<text x="0" y="{y + 14}" style="font-size:12px;fill:var(--ea-ink);font-weight:600">{html.escape(_ROWS[key][lang])}</text>')
        y += 22
        for s, (text, ok) in enumerate(_cells(key, lang)):
            x = X0 + s * CW
            if not ok:
                parts.append(f'<rect x="{x + 6}" y="{y}" width="{CW - 12}" height="{CH}" rx="3" style="fill:var(--ea-loss-wash)"/>')
                style = "fill:var(--ea-loss-text)"
            else:
                parts.append(f'<rect x="{x + 6}" y="{y}" width="{CW - 12}" height="{CH}" rx="3" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/>')
                style = "fill:var(--ea-ink);font-weight:700"
            parts.append(f'<text x="{x + CW / 2}" y="{y + 17}" text-anchor="middle" style="font-size:12px;{style};'
                         f'font-variant-numeric:tabular-nums">{html.escape(text)}</text>')
        y += CH + 14
    W = X0 + SEEDS * CW
    title = {"en": "What Qwen3 answered in eight runs of each question", "nl": "Wat Qwen3 antwoordde in acht runs van elke vraag"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:34rem">'
            f'<svg viewBox="0 0 {W} {y}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(parts)}</svg></figure>')


def grid_after(lang: str) -> str:
    F = findings()
    assert F == {"returns": "1.8", "sees_bug": 0, "right": 1, "three": 4, "renamed": 0, "says_clear": 2}
    assert "9 / 3 = 3" in answer("value", 6) and "len(scores)" in answer("value", 6)   # it wrote the right line, and still divided by 3
    if lang == "nl":
        return (f"Alle acht uitleggen zeggen dat de functie het gemiddelde geeft van de scores zonder de eerste en de laatste. "
                f"Geen enkele ziet dat de laatste regel deelt door het aantal van álle scores, vijf, en niet door de drie in het "
                f"midden. Gevraagd wat hij teruggeeft voor [1, 2, 3, 4, 100], zei Qwen3 {F['three']} keer 3, het getal dat bij de "
                f"uitleg past; maar één run gaf 1,8, wat de functie echt teruggeeft. Eén run schreef zelfs "
                f"`sum(middle) / len(scores) = 9 / 3 = 3` op: de goede regel, en toch gedeeld door 3. En gevraagd om duidelijke "
                f"namen, veranderde geen van de acht een naam; {F['says_clear']} zeiden dat de code duidelijke namen *heeft*, en "
                f"gaven hem terug met dezelfde namen.")
    return (f"All eight explanations say that the function gives the average of the scores without the first and the last. Not "
            f"one sees that the last line divides by the number of *all* the scores, five, not by the three in the middle. Asked "
            f"what it returns for [1, 2, 3, 4, 100], Qwen3 said 3 in {F['three']} runs, the number that fits the explanation; only "
            f"one run gave 1.8, what the function really returns. One run even wrote `sum(middle) / len(scores) = 9 / 3 = 3`: "
            f"the right line, and still divided by 3. And asked for clear names, none of the eight changed a name; "
            f"{F['says_clear']} said that the code *has* clear names, and gave it back with the same names.")


def answer_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th answer (1 to 24): eight explanations, eight values, eight renames."""
    keys = list(_ROWS)
    key, seed = keys[(int(i) - 1) // SEEDS], (int(i) - 1) % SEEDS
    body = (f'<pre style="font-family:var(--ea-font-code);font-size:0.8rem;background:var(--ea-inset);padding:0.5rem 0.7rem;'
            f'border-radius:6px;margin:0;max-width:42rem;max-height:20rem;overflow:auto;white-space:pre-wrap">'
            f'{html.escape(answer(key, seed).strip())}</pre>')
    run = {"en": f"run {seed + 1} of 8", "nl": f"run {seed + 1} van 8"}[lang]
    if key == "explain":
        said = {"en": f"**Explanation, {run}:** it does not say that the function divides by all five scores.",
                "nl": f"**Uitleg, {run}:** die zegt niet dat de functie door alle vijf de scores deelt."}[lang]
    elif key == "value":
        v = VALUES[seed]
        shown = v.replace(".", ",") if lang == "nl" else v
        if v == findings()["returns"]:
            said = {"en": f"**Answer, {run}:** {shown}, what the function really returns.",
                    "nl": f"**Antwoord, {run}:** {shown}, wat de functie echt teruggeeft."}[lang]
        else:
            said = {"en": f"**Answer, {run}:** {shown}; the function returns 1.8.",
                    "nl": f"**Antwoord, {run}:** {shown}; de functie geeft 1,8 terug."}[lang]
    else:
        clear = says_clear(seed)
        said = {"en": f"**Rename, {run}:** no name changed" + ("; it says the code has clear names." if clear else "."),
                "nl": f"**Hernoemen, {run}:** geen naam veranderd" + ("; het zegt dat de code duidelijke namen heeft." if clear else ".")}[lang]
    return body, said


# --- the exercises -------------------------------------------------------------------------------------------------------

_AVERAGE_RIGHT = AVERAGE.replace("len(scores)", "len(middle)")
_COUNT_RIGHT = ("def count_above(numbers, limit):\n    count = 0\n    for number in numbers:\n        if number > limit:\n"
                "            count = count + 1\n    return count\n")
_SYNTAX = [
    Mistake("curly-quote", syntax=r"[“”‘’„]",
            en="Your code has a curly quote, like `“`. Python only reads straight quotes, like `\"`: type the quotes again in the box.",
            nl="Je code heeft een gekruld aanhalingsteken, zoals `“`. Python leest alleen rechte, zoals `\"`: typ de aanhalingstekens opnieuw in het vak."),
    Mistake("colon-missing", syntax=r"^[ \t]*(def\b[^:\n]*|for\b[^:\n]*|if\b[^:\n]*)$",
            en="The line `{line}` must end with a colon, `:`. The colon tells Python that the indented lines below belong to it.",
            nl="De regel `{line}` moet eindigen op een dubbele punt, `:`. Die vertelt Python dat de ingesprongen regels eronder erbij horen."),
]

EXERCISES = {
    "fix-average": FunctionBox(
        name="fix-average",
        picture=yours.steps(call=0),
        function="average_without_extremes",
        start=AVERAGE,
        solution=_AVERAGE_RIGHT,
        visible=[(list(CASE),)],
        hidden=[([10, 20, 30],), ([5, 1, 9, 7],), ([3, 3, 3, 3],), ([2.5, 1.0, 4.0, 3.5],)],
        mistakes=_SYNTAX + [
            Mistake("minus-one", wrong=AVERAGE.replace("len(scores)", "(len(scores) - 1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. You divide by one less than all the scores, but two scores went: the first and the last. What is the number of scores that `middle` holds?",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Je deelt door één minder dan alle scores, maar er gingen er twee af: de eerste en de laatste. Hoeveel scores zitten er in `middle`?"),
            Mistake("sum-only", wrong=AVERAGE.replace(" / len(scores)", ""),
                    en="`{call}` gives `{got}`: the sum of the middle scores. An average is that sum divided by how many there are.",
                    nl="`{call}` geeft `{got}`: de som van de middelste scores. Een gemiddelde is die som gedeeld door hoeveel het er zijn."),
            Mistake("no-sort", state="almost", wrong=_AVERAGE_RIGHT.replace("    scores = sorted(scores)\n", ""),
                    en="Right for the example, which was already in order. But `{call}` gives `{got}`, and it should give `{want}`: without `sorted`, the first and the last are not the smallest and the largest. Keep the line with `sorted`.",
                    nl="Goed voor het voorbeeld, dat al op volgorde stond. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn: zonder `sorted` zijn de eerste en de laatste niet de kleinste en de grootste. Laat de regel met `sorted` staan."),
            Mistake("fixed", state="almost", wrong="def average_without_extremes(scores):\n    return 3\n",
                    en="Your function always gives `3`, the answer for the example. Work it out from `scores`, so it is right for every list.",
                    nl="Je functie geeft altijd `3`, het antwoord voor het voorbeeld. Reken het uit met `scores`, dan klopt hij voor elke lijst."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "This is still the function that Qwen3 explained: for `[1, 2, 3, 4, 100]` it gives `1.8`. Change one thing, so that it gives the average of the middle scores: `3.0`.",
                      "nl": "Dit is nog de functie die Qwen3 uitlegde: voor `[1, 2, 3, 4, 100]` geeft hij `1.8`. Verander één ding, zodat hij het gemiddelde van de middelste scores geeft: `3.0`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the function again.",
                        "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de functie."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the middle is `[2, 3, 4]`, and 9 divided by 3 is 3. Which number does the last line divide by?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: het midden is `[2, 3, 4]`, en 9 gedeeld door 3 is 3. Door welk getal deelt de laatste regel?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work out the average of the middle scores for every list?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Rekent je functie voor elke lijst het gemiddelde van de middelste scores uit?"},
            "no_function": {"en": "Your box does not make a function `average_without_extremes`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `average_without_extremes`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Right in all {n} cases: now the function does what the explanation said. The explanation was right about what it was meant to do, and wrong about what it did; one case worked out by hand showed the difference.",
                        "nl": "Goed in alle {n} gevallen: nu doet de functie wat de uitleg zei. De uitleg klopte over wat hij moest doen, en niet over wat hij deed; één geval met de hand uitgerekend liet het verschil zien."},
        },
    ),
    "rename-count": FunctionBox(
        name="rename-count",
        picture=yours.steps(call=0),
        function="count_above",
        start=COUNT,
        solution=_COUNT_RIGHT,
        visible=[([3, 8, 5, 10], 4)],
        hidden=[([], 1), ([1, 2, 3], 3), ([5, 5, 6], 5), ([7], 2)],
        mistakes=_SYNTAX + [
            Mistake("at-least", state="almost", wrong=_COUNT_RIGHT.replace("number > limit", "number >= limit"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function now also counts numbers equal to the limit. A rename changes names, never what the code does: keep `>`.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: je functie telt nu ook getallen die gelijk zijn aan de grens. Hernoemen verandert namen, nooit wat de code doet: laat `>` staan."),
            Mistake("below", wrong=_COUNT_RIGHT.replace("number > limit", "number < limit"),
                    en="`{call}` gives `{got}`: your function counts the numbers below the limit. A rename changes names, never what the code does: keep the comparison as it was.",
                    nl="`{call}` geeft `{got}`: je functie telt de getallen onder de grens. Hernoemen verandert namen, nooit wat de code doet: laat de vergelijking zoals hij was."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "The box still has the function `f`. Give it the name `count_above`, and give its arguments and variables names that say what they hold.",
                      "nl": "In het vak staat nog de functie `f`. Geef hem de naam `count_above`, en geef zijn argumenten en variabelen namen die zeggen wat erin zit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the function again.",
                        "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de functie."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: 8, 5 and 10 are above 4. A rename changes names only; compare each line of your function with the same line of `f`.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: 8, 5 en 10 liggen boven 4. Hernoemen verandert alleen namen; vergelijk elke regel van je functie met dezelfde regel van `f`."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does every line still do what the same line of `f` did?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Doet elke regel nog wat dezelfde regel van `f` deed?"},
            "no_function": {"en": "Your box does not make a function `count_above` yet. Change the name after `def`: `def count_above(...)`.",
                            "nl": "Je vak maakt nog geen functie `count_above`. Verander de naam na `def`: `def count_above(...)`."},
            "correct": {"en": "Right in all {n} cases: `count_above` does exactly what `f` did. The check only sees what your function gives; whether the names are clear, you judge yourself. That is what a rename is for: it reads better, and gives nothing else.",
                        "nl": "Goed in alle {n} gevallen: `count_above` doet precies wat `f` deed. De check ziet alleen wat je functie geeft; of de namen duidelijk zijn, beoordeel je zelf. Daar is hernoemen voor: het leest beter, en geeft niets anders."},
        },
    ),
}

# The lesson is done when these are correct. "read-average" is a choice question in the lesson.
REQUIRED = ["read-average", "fix-average", "rename-count"]

# What the tests feed each check, and what it must say: (exercise, the learner's box, state, known mistake).
CASES = [
    ("fix-average", AVERAGE, "empty", None),
    ("fix-average", _AVERAGE_RIGHT, "correct", None),
    ("fix-average", AVERAGE.replace("len(scores)", "(len(scores) - 2)"), "correct", None),
    ("fix-average", AVERAGE.replace("len(scores)", "3"), "almost", None),
    ("fix-average", AVERAGE.replace("len(scores)", "(len(scores) - 1)"), "wrong", "minus-one"),
    ("fix-average", AVERAGE.replace(" / len(scores)", ""), "wrong", "sum-only"),
    ("fix-average", _AVERAGE_RIGHT.replace("    scores = sorted(scores)\n", ""), "almost", "no-sort"),
    ("fix-average", "def average_without_extremes(scores):\n    return 3\n", "almost", "fixed"),
    ("fix-average", _AVERAGE_RIGHT.replace("(scores):", "(scores)"), "error", "colon-missing"),
    ("rename-count", COUNT, "empty", None),
    ("rename-count", _COUNT_RIGHT, "correct", None),
    ("rename-count", COUNT.replace("def f(", "def count_above("), "correct", None),
    ("rename-count", _COUNT_RIGHT.replace("number > limit", "number >= limit"), "almost", "at-least"),
    ("rename-count", _COUNT_RIGHT.replace("number > limit", "number < limit"), "wrong", "below"),
    ("rename-count", COUNT.replace("a, b", "numbers, limit"), "wrong", "no-function"),
]
