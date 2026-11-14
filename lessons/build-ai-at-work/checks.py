# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-ai-at-work, and the course's run of Qwen3-0.6B on ten small tasks that look about as easy
as each other to a person (data/course-work-qwen.json, written by data/tools/record_course_work_qwen.py from ASKS):
how each answer is marked right or wrong, the picture of the jagged frontier, the frames of the slider, and every
number the texts say about the run."""
import html
import json
import re

from everything_ai import paths
from everything_ai.check import Sort

COURSE = "data/course-work-qwen.json"
SEEDS = 8
MAX_TOKENS = 150
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _numbers(text: str) -> list:
    return [x.replace(",", "") for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def _last_number(text: str) -> str | None:
    found = _numbers(text)
    return found[-1] if found else None


def _letters(text: str) -> bool:
    return _last_number(text) == "3" or (not _numbers(text) and re.search(r"\bthree\b", text, re.I) is not None)


def _reverse(text: str) -> bool:
    return "edanomel" in re.sub(r"[^a-z]", "", text.lower())


def _capital(text: str) -> bool:
    return "canberra" in text.lower()


def _rhymes(text: str) -> bool:
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text)}
    return len({w for w in words if w.endswith("at") and w not in ("cat", "what")}) >= 3


def first_sentence(text: str) -> str:
    """The sentence the model wrote, without quotes, stars and anything after its first line with letters."""
    line = next((l for l in text.splitlines() if re.search(r"[A-Za-z]", l)), "")
    return line.strip().strip('"*“”').strip()


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", first_sentence(text)))


def _eight(text: str) -> bool:
    return words(text) == 8


def sentence(text: str) -> str:
    """The sentence the model wrote for the colleague: the last part between double quotes, when there is one, so that
    a line before it such as *Here's a polite sentence to move the meeting from Monday to Tuesday:* does not count."""
    quoted = re.findall(r'"([^"]+)"', text)
    return quoted[-1] if quoted else text


def _email(text: str) -> bool:
    low = sentence(text).lower()
    return "monday" in low and "tuesday" in low


def _compare(text: str) -> bool:
    return _last_number(text) == "9.9"


def _multiply(text: str) -> bool:
    return _last_number(text) == "1081"


def _weekday(text: str) -> bool:
    found = re.findall("|".join(_WEEKDAYS), text.lower())
    return bool(found) and found[-1] == "monday"


def _list(text: str) -> bool:
    items = [l.strip() for l in text.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+\S", l)]
    joined = " ".join(items).lower()
    return len(items) == 3 and all(w in joined for w in ("milk", "bread", "apple"))


# The ten tasks, in the order of the table: the prompt the model got, as written to Qwen3 (it writes English best),
# a short name in both languages, and how its answer is marked.
TASKS = {
    "list": ("Turn this into a shopping list, one item per line: we need milk, two loaves of bread and some apples.",
             {"en": "Shopping list from a sentence", "nl": "Boodschappenlijst uit een zin"}, _list),
    "capital": ("What is the capital of Australia? Reply with the city only.",
                {"en": "Capital of Australia", "nl": "Hoofdstad van Australië"}, _capital),
    "multiply": ("What is 23 × 47? Reply with the number only.",
                 {"en": "23 × 47", "nl": "23 × 47"}, _multiply),
    "letters": ("How many times does the letter r appear in the word strawberry? Reply with the number only.",
                {"en": "How many r's in strawberry", "nl": "Hoeveel r'en in strawberry"}, _letters),
    "email": ("Write one polite sentence asking a colleague to move our meeting from Monday to Tuesday.",
              {"en": "Polite sentence: meeting to Tuesday", "nl": "Beleefde zin: overleg naar dinsdag"}, _email),
    "reverse": ("Write the word lemonade backwards, letter by letter, as one word. Reply with that word only.",
                {"en": "Lemonade backwards", "nl": "Lemonade achterstevoren"}, _reverse),
    "compare": ("Which number is larger: 9.11 or 9.9? Reply with the number only.",
                {"en": "Larger: 9.11 or 9.9", "nl": "Groter: 9,11 of 9,9"}, _compare),
    "rhymes": ("Give three English words that rhyme with cat. Reply with the three words only, separated by commas.",
               {"en": "Three rhymes on cat", "nl": "Drie rijmwoorden op cat"}, _rhymes),
    "weekday": ("What day is three days after Friday? Reply with the day only.",
                {"en": "Three days after Friday", "nl": "Drie dagen na vrijdag"}, _weekday),
    "eight": ("Write one sentence of exactly eight words about the sea. Reply with the sentence only.",
              {"en": "A sentence of exactly eight words", "nl": "Een zin van precies acht woorden"}, _eight),
}
ASKS = {k: {"messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOKENS} for k, (prompt, _, _) in TASKS.items()}


def course() -> dict:
    """The course's run of this lesson: {task: {"messages", "max_tokens", "answers": [{"seed", "text", "tokens",
    "seconds", "finish"}, ...]}}."""
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))["parts"]["build-ai-at-work"]


def marks(task: str) -> list:
    """Right or wrong, for each of the task's answers in seed order."""
    return [TASKS[task][2](a["text"]) for a in course()[task]["answers"]]


def right(task: str) -> int:
    return sum(marks(task))


def name(task: str, lang: str) -> str:
    return TASKS[task][1][lang]


def frontier(lang: str) -> str:
    """The ten tasks, each with its eight answers as eight squares: a filled square in the colour of what a model
    writes for a right answer, an open square in the colour of mistakes for a wrong one, and the count beside them."""
    R = course()
    rows, ROW, W = [], 30, 600
    for i, task in enumerate(TASKS):
        y = 6 + i * ROW
        got = marks(task)
        rows.append(f'<text x="0" y="{y + 16}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(name(task, lang))}</text>')
        for j, ok in enumerate(got):
            x = 300 + j * 26
            style = "fill:var(--ea-pred);stroke:var(--ea-pred)" if ok else "fill:none;stroke:var(--ea-loss);stroke-width:2"
            rows.append(f'<rect x="{x}" y="{y + 3}" width="18" height="18" rx="3" style="{style}"/>')
        rows.append(f'<text x="{300 + 8 * 26 + 8}" y="{y + 16}" style="font-size:13px;fill:var(--ea-ink-2);'
                    f'font-variant-numeric:tabular-nums">{sum(got)} / {len(R[task]["answers"])}</text>')
    H = 12 + len(TASKS) * ROW
    title = {"en": "Ten small tasks, eight runs each: right (filled) or wrong (open)",
             "nl": "Tien kleine taken, elk acht runs: goed (vol) of fout (open)"}[lang]
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:38rem">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{title}</title>{"".join(rows)}</svg></figure>')


def task_frame(i: int, lang: str) -> tuple[str, str]:
    """The i-th task (1 to 10): its prompt and its eight answers, each marked right or wrong; then the sentence."""
    task = list(TASKS)[int(i) - 1]
    answers = course()[task]["answers"]
    got = marks(task)
    mark = {True: ("✓", "var(--ea-pred)"), False: ("✗", "var(--ea-loss)")}
    head = {"en": "Prompt", "nl": "Prompt"}[lang]
    cells = "".join(
        f'<div style="display:flex;gap:0.5rem;align-items:baseline;padding:0.15rem 0;border-top:1px solid var(--ea-line)">'
        f'<span style="color:{mark[ok][1]};font-weight:700;width:1rem">{mark[ok][0]}</span>'
        f'<span style="font-size:0.9rem;max-height:3.2rem;overflow:hidden">{html.escape(a["text"].strip())}</span></div>'
        for a, ok in zip(answers, got))
    body = (f'<div style="max-width:40rem;font-family:var(--ea-font-ui)">'
            f'<p style="margin:0 0 0.4rem"><b>{head}:</b> <i>{html.escape(TASKS[task][0])}</i></p>{cells}</div>')
    n = sum(got)
    if lang == "nl":
        said = f"**{name(task, lang)}:** goed in {n} van de {len(got)} runs."
    else:
        said = f"**{name(task, lang)}:** right in {n} of {len(got)} runs."
    return body, said


def counts() -> dict:
    return {task: right(task) for task in TASKS}


def frontier_after(lang: str) -> str:
    """What the picture shows, from the run."""
    got = counts()
    n = len(course()["list"]["answers"])
    always = [t for t in TASKS if got[t] == n]
    never = [t for t in TASKS if got[t] == 0]
    assert {"multiply", "compare"} <= set(always) and {"capital", "letters", "reverse"} <= set(never)
    capitals = [a["text"].strip().rstrip(".") for a in course()["capital"]["answers"]]
    assert set(capitals) == {"Sydney", "New South Wales", "Queensland"}
    states = sum(c != "Sydney" for c in capitals)
    rhymes = sum("cat, cat, cat" in a["text"].lower() for a in course()["rhymes"]["answers"])
    assert rhymes == n
    email = right("email")
    assert 0 < email < n and all("monday" not in sentence(a["text"]).lower()
                                   for a, ok in zip(course()["email"]["answers"], marks("email")) if not ok)
    if lang == "nl":
        return (f"Qwen3 rekende 23 × 47 in alle {n} runs goed uit, maar de hoofdstad van Australië noemde het geen enkele keer: "
                f"{n - states} keer schreef het Sydney, en {states} keer een deelstaat, New South Wales of Queensland. Het telde "
                f"de r'en in *strawberry* nooit goed, schreef *lemonade* nooit achterstevoren, en gaf als drie rijmwoorden op "
                f"*cat* in elke run *cat, cat, cat*. Voor een mens is 23 × 47 de moeilijkste van die vijf; voor het model is het "
                f"juist andersom. En de beleefde zin, misschien de makkelijkste taak van allemaal, noemde maar in {email} van de "
                f"{n} runs allebei de dagen: in de andere ontbrak *Monday*.")
    return (f"Qwen3 worked out 23 × 47 right in all {n} runs, but it never named the capital of Australia: {n - states} times "
            f"it wrote Sydney, and {states} times a state, New South Wales or Queensland. It never counted the r's in "
            f"*strawberry* right, never wrote *lemonade* backwards, and in every run its three rhymes on *cat* were "
            f"*cat, cat, cat*. For a person, 23 × 47 is the hardest of those five; for the model it is the other way around. "
            f"And the polite sentence, perhaps the easiest task of all, named both days in only {email} of the {n} runs: in "
            f"the others, Monday was missing.")


# The lesson is done when these are correct. "frontier-qwen" and "outside-study" are choice questions in the lesson.
REQUIRED = ["centaur-or-cyborg", "frontier-qwen", "outside-study"]

_CENTAUR = {"en": "Does the AI do a whole part on its own, which the person checks as a whole afterwards?",
            "nl": "Doet de AI een heel deel alleen, dat de persoon daarna als geheel nakijkt?"}
_CYBORG = {"en": "Do the person and the AI take turns, sentence by sentence or step by step?",
           "nl": "Wisselen de persoon en de AI elkaar af, zin voor zin of stap voor stap?"}

EXERCISES = {
    "centaur-or-cyborg": Sort(
        name="centaur-or-cyborg",
        groups={
            "centaur": {"en": "Centaur", "nl": "Centaur"},
            "cyborg": {"en": "Cyborg", "nl": "Cyborg"},
        },
        items={
            "minutes": {"en": "**Minutes.** A secretary gives the assistant the notes of a meeting, gets the minutes back, and reads them through once before sending them.",
                        "nl": "**Notulen.** Een secretaris geeft de assistent de aantekeningen van een vergadering, krijgt de notulen terug en leest ze één keer door voordat ze de deur uit gaan."},
            "sentence": {"en": "**Sentence by sentence.** A copywriter writes an advert with the assistant: they write a line, ask for three better versions, change one word, and ask again.",
                         "nl": "**Zin voor zin.** Een tekstschrijver schrijft een advertentie samen met de assistent: een regel schrijven, om drie betere versies vragen, één woord veranderen en opnieuw vragen."},
            "translate": {"en": "**Translation.** A sales team has the assistant translate a product sheet into German, and a German colleague checks it before it goes out.",
                          "nl": "**Vertaling.** Een verkoopteam laat de assistent een productblad in het Duits vertalen, en een Duitse collega controleert het voordat het de deur uit gaat."},
            "argue": {"en": "**Sparring.** A planner thinks out loud in a chat: each idea goes in, the assistant pokes holes in it, the planner adjusts, and so on for an hour.",
                      "nl": "**Sparren.** Een planner denkt hardop in een chat: elk idee gaat erin, de assistent schiet er gaten in, de planner past het aan, en zo een uur lang."},
            "split": {"en": "**Split.** A teacher writes the questions of a test alone, and lets the assistant make the answer key, which the teacher then checks.",
                      "nl": "**Verdeeld.** Een docent schrijft de vragen van een toets zelf, en laat de assistent het antwoordmodel maken, dat de docent daarna nakijkt."},
            "code": {"en": "**Line by line.** A programmer types, accepts a suggestion for the next line, changes it, types on, and asks the chat why a line fails.",
                     "nl": "**Regel voor regel.** Een programmeur typt, neemt een voorstel voor de volgende regel over, past het aan, typt verder en vraagt de chat waarom een regel misgaat."},
        },
        solution={"minutes": "centaur", "sentence": "cyborg", "translate": "centaur", "argue": "cyborg", "split": "centaur", "code": "cyborg"},
        nudge={"minutes": _CENTAUR, "translate": _CENTAUR, "split": _CENTAUR, "sentence": _CYBORG, "argue": _CYBORG, "code": _CYBORG},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six ways of working again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes manieren van werken opnieuw uit."},
            "empty": {"en": "Pick centaur or cyborg for each of the six.",
                      "nl": "Kies voor elk van de zes centaur of cyborg."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. The centaurs hand over a whole part and check it as a whole; the cyborgs go back and forth so often that you can no longer say which part is whose. Most people do both, depending on the task.",
                        "nl": "Alle {n} kloppen. De centauren geven een heel deel uit handen en kijken het als geheel na; de cyborgs gaan zo vaak heen en weer dat je niet meer kunt zeggen welk deel van wie is. De meeste mensen doen allebei, afhankelijk van de taak."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"minutes": "centaur", "sentence": "cyborg", "translate": "centaur", "argue": "cyborg", "split": "centaur", "code": "cyborg"}
CASES = [
    ("centaur-or-cyborg", {}, "empty", None),
    ("centaur-or-cyborg", _RIGHT, "correct", None),
    ("centaur-or-cyborg", {"minutes": "centaur", "sentence": "cyborg"}, "wrong", "unfinished"),
    ("centaur-or-cyborg", {**_RIGHT, "translate": "cyborg"}, "wrong", "wrong:translate"),
    ("centaur-or-cyborg", {**_RIGHT, "code": "centaur"}, "wrong", "wrong:code"),
    ("centaur-or-cyborg", {**_RIGHT, "argue": "centaur", "split": "cyborg"}, "wrong", "wrong:argue"),
]
