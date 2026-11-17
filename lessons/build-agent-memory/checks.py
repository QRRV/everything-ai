# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-memory: what an agent remembers. Short-term memory is the conversation in the request; long-term
memory is notes the agent saves with a tool, in a file, which the next conversation reads into its system prompt. The
course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): six things a user says about themselves, each in a
conversation where the model can save a note with the tool remember, and six questions that need them, asked without
memory, later in the same conversation, and in a new conversation with the saved notes. What they show, their pictures,
the exercises (the tool remember, and the first messages of a new conversation with the notes) and the demo on Qwen3."""
import html
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"

SYSTEM = "You are a kitchen assistant. Answer in one or two short sentences."
TOLD = ("You are a kitchen assistant. When the user tells you something about themselves that will still matter in later "
        "conversations, save it with the tool remember, as a short note. Answer in one or two short sentences.")
NOTES_HEAD = "Notes from earlier conversations with the user:"
REMEMBER = {"type": "function", "function": {
    "name": "remember", "description": "Save a short note about the user, for later conversations.",
    "parameters": {"type": "object", "properties": {"note": {"type": "string", "description": "the note, such as: The user has no oven."}},
                   "required": ["note"]}}}
REMEMBER_JSON = json.dumps(REMEMBER, indent=2)   # the tool as the lesson shows it
NAMED = {"type": "function", "function": {"name": "remember"}}   # tool_choice that makes the model call remember


def _starts_no(t: str) -> bool:
    return re.match(r"\W*no\b", t) is not None


FACTS = [   # (what the user says, the question that needs it, whether an answer is right)
    ("I am allergic to nuts.", "Can I make pasta with pesto tonight? Answer yes or no first.", lambda t: _starts_no(t) and "nut" in t),
    ("I do not have an oven.", "Can I bake an apple pie tonight? Answer yes or no first.", lambda t: _starts_no(t) and "oven" in t),
    ("I do not eat eggs.", "Should I make an omelette for breakfast? Answer yes or no first.", lambda t: _starts_no(t) and "egg" in t),
    ("I always cook for four people.", "For how many people do I cook?", lambda t: "four" in t or re.search(r"\b4\b", t) is not None),
    ("My favorite dish is rice pudding.", "What is my favorite dish?", lambda t: "rice pudding" in t),
    ("I do my shopping on Saturdays.", "On which day do I do my shopping?", lambda t: "saturday" in t),
]


def right(i: int, text: str | None) -> bool:
    return bool(text) and FACTS[i][2](text.strip().lower())


def memory_system(notes: list) -> str:
    """The system prompt of a new conversation: the course's, with the notes under their heading when there are any."""
    return SYSTEM if not notes else SYSTEM + "\n\n" + NOTES_HEAD + "\n" + "\n".join(notes)


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: the result of each call of the tool remember, "Saved.", as a message with the role tool."""
    return [{"role": "tool", "tool_call_id": c["id"],
             "content": "Saved." if c["function"]["name"] == "remember" else "Error: no tool " + c["function"]["name"]}
            for c in message.get("tool_calls") or []]


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"].get("build-agent-memory", {})


def told_steps(i: int, kind: str = "told") -> list:
    """The conversation in which the user said fact i: with tool_choice auto ("told"), or naming remember ("saved")."""
    return course()[f"{kind}-{i}"]["answers"][0]["steps"]


def notes_of(steps: list) -> list:
    """The notes a run saved: the note of each call of remember whose arguments are JSON with a note."""
    out = []
    for st in steps:
        for c in st["tool_calls"]:
            try:
                arguments = json.loads(c["function"]["arguments"])
            except ValueError:
                continue
            if c["function"]["name"] == "remember" and isinstance(arguments, dict) and str(arguments.get("note", "")).strip():
                out.append(str(arguments["note"]).strip())
    return out


def saved_notes() -> list:
    """Every note the six runs with tool_choice naming remember saved, in the order of the facts: the notes file of a
    new conversation."""
    return [n for i in range(len(FACTS)) for n in notes_of(told_steps(i, "saved"))]


def asked(text: str) -> list:
    return [{"role": "user", "content": text}]


RECORD = {}
for _i, (_fact, _question, _) in enumerate(FACTS):
    RECORD[f"told-{_i}"] = {"messages": [{"role": "system", "content": TOLD}] + asked(_fact), "tools": [REMEMBER],
                            "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}
    RECORD[f"saved-{_i}"] = {"messages": [{"role": "system", "content": TOLD}] + asked(_fact), "tools": [REMEMBER],
                             "tool_choice": NAMED, "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}
    RECORD[f"none-{_i}"] = {"messages": [{"role": "system", "content": SYSTEM}] + asked(_question), "max_tokens": 80,
                            "temperature": 0, "seeds": 1}
    RECORD[f"short-{_i}"] = {"messages": [{"role": "system", "content": SYSTEM}], "turns": [_fact, _question],
                             "max_tokens": 80, "temperature": 0, "seeds": 1}
if DATA.exists() and all(f"saved-{_i}" in course() for _i in range(len(FACTS))):   # the second pass of the recorder
    _system = memory_system(saved_notes())
    for _i, (_fact, _question, _) in enumerate(FACTS):
        RECORD[f"long-{_i}"] = {"messages": [{"role": "system", "content": _system}] + asked(_question), "max_tokens": 80,
                                "temperature": 0, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def answer(kind: str, i: int) -> dict:
    """The answer to question i: without memory ("none"), later in the same conversation ("short"), or in a new
    conversation with the saved notes ("long")."""
    a = course()[f"{kind}-{i}"]["answers"][0]
    return a["turns"][-1] if kind == "short" else a


def wrote_instead(i: int) -> str | None:
    """How the run with tool_choice auto wrote the note into its answer instead of calling remember: "json" (the call as
    JSON text), "tags" (between <remember> tags), "remember" (after "Remember:"), "other", or None when it called it."""
    steps = told_steps(i)
    if notes_of(steps):
        return None
    text = steps[0]["text"].strip()
    return ("json" if text.startswith("{") and '"remember"' in text else "tags" if "<remember>" in text else
            "remember" if text.lower().startswith("remember:") else "other")


def findings() -> dict:
    n = len(FACTS)
    ways = [wrote_instead(i) for i in range(n)]
    out = {"facts": n, "called": [i for i in range(n) if ways[i] is None],
           "ways": {w: [i for i in range(n) if ways[i] == w] for w in ("json", "tags", "remember", "other")},
           "saved": [i for i in range(n) if notes_of(told_steps(i, "saved"))], "notes": saved_notes()}
    for kind in ("none", "short", "long"):
        out[kind] = [i for i in range(n) if right(i, answer(kind, i)["text"])]
        out[kind + "_prompt"] = sum(answer(kind, i)["prompt_tokens"] for i in range(n))
    out["first_person"] = [k for k, note in enumerate(out["notes"]) if re.match(r"(I|My)\b", note)]
    out["yes_no"] = [i for i, (_, q, _) in enumerate(FACTS) if q.endswith("Answer yes or no first.")]
    out["long_yes"] = [i for i in out["yes_no"] if re.match(r"\W*yes\b", answer("long", i)["text"].lower())]
    return out


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the pictures of the idea and of the measurement ------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _message(role: str, text: str) -> str:
    """One message of a request: its role, and its text in the color of who wrote it."""
    color = {"system": "var(--ea-ink-2)", "user": "var(--ea-ink)", "assistant": "var(--ea-pred-text)",
             "notes": "var(--ea-data-text)", "tool": "var(--ea-data-text)"}[role]
    shown = "system" if role == "notes" else role
    return (f'<div style="border-left:3px solid {color};padding:0.15rem 0.5rem;margin:0.25rem 0">'
            f'<div style="{_UI};font-size:11px;color:var(--ea-ink-2)">{shown}</div>{_code(text, color)}</div>')


_IDEA_WORDS = {
    "en": {"title": "What the model gets to read, three ways", "none": "No memory: a new conversation",
           "short": "Short-term memory: the same conversation", "long": "Long-term memory: a new conversation with the notes",
           "earlier": "An earlier conversation saved the notes with the tool remember, in a file; your code reads them into the system prompt.",
           "alt": "Three requests with the same last question. Without memory, the request holds only the system prompt and the question. With short-term memory, it holds the whole conversation so far, with what the user said earlier. With long-term memory, a new conversation's system prompt holds the notes an earlier conversation saved in a file."},
    "nl": {"title": "Wat het model te lezen krijgt, op drie manieren", "none": "Geen geheugen: een nieuw gesprek",
           "short": "Kortetermijngeheugen: hetzelfde gesprek", "long": "Langetermijngeheugen: een nieuw gesprek met de notities",
           "earlier": "Een eerder gesprek bewaarde de notities met de tool remember, in een bestand; je code leest ze in de system prompt.",
           "alt": "Drie requests met dezelfde laatste vraag. Zonder geheugen staat in het request alleen de system prompt en de vraag. Met kortetermijngeheugen staat het hele gesprek tot nu toe erin, met wat de gebruiker eerder zei. Met langetermijngeheugen staan in de system prompt van een nieuw gesprek de notities die een eerder gesprek in een bestand bewaarde."},
}

IDEA = 0   # the fact of the picture of the idea: the nut allergy


def idea_picture(lang: str) -> str:
    """The three requests for the same question: without memory, in the same conversation, and with the notes."""
    w = _IDEA_WORDS[lang]
    fact, question, _ = FACTS[IDEA]
    note = "The user is allergic to nuts."
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin-bottom:0.2rem"
    cols = [
        (w["none"], _message("system", SYSTEM) + _message("user", question), ""),
        (w["short"], _message("system", SYSTEM) + _message("user", fact) + _message("assistant", "…") + _message("user", question), ""),
        (w["long"], _message("notes", memory_system([note])) + _message("user", question),
         f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(w["earlier"])}</div>'),
    ]
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:1rem">'
            + "".join(f'<div style="min-width:0"><div style="{head}">{html.escape(t)}</div>{m}{extra}</div>' for t, m, extra in cols)
            + "</div>")
    return yours.figure(w["title"], body, alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "Six things a user said, and a question that needs each", "said": "what the user said",
           "note": "the note Qwen3 saved", "none": "no memory", "short": "same conversation", "long": "new, with the notes",
           "no_note": "no note", "right": "right", "wrong": "not right", "total": "{k} of {n}", "tokens": "{k} prompt tokens"},
    "nl": {"title": "Zes dingen die een gebruiker zei, en een vraag die elk nodig heeft", "said": "wat de gebruiker zei",
           "note": "de notitie die Qwen3 bewaarde", "none": "geen geheugen", "short": "hetzelfde gesprek", "long": "nieuw, met de notities",
           "no_note": "geen notitie", "right": "goed", "wrong": "niet goed", "total": "{k} van de {n}", "tokens": "{k} prompt tokens"},
}


def _mark(ok: bool, w: dict) -> str:
    color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
    return f'<span style="{_UI};font-size:12px;color:{color};font-weight:{400 if ok else 600}">{w["right"] if ok else w["wrong"]}</span>'


def table_picture(lang: str) -> str:
    """Each fact with the note the model saved, and whether each of the three answers was right."""
    w, f = _TABLE_WORDS[lang], findings()
    rows = []
    for i, (fact, question, _) in enumerate(FACTS):
        notes = notes_of(told_steps(i, "saved"))
        note = _code("\n".join(notes), "var(--ea-data-text)") if notes else _mark(False, {**w, "wrong": w["no_note"]})
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(fact)}<br><span style="color:var(--ea-ink-2)">{html.escape(question)}</span></td>'
                    f'<td style="{_CELL}">{note}</td>'
                    + "".join(f'<td style="{_CELL}">{_mark(i in f[k], w)}</td>' for k in ("none", "short", "long")) + "</tr>")
    n = f["facts"]
    foot = (f'<tr><td style="{_CELL}"></td><td style="{_CELL}"></td>'
            + "".join(f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f[k]), n=n)}<br>'
                      f'<span style="font-weight:400;color:var(--ea-ink-2)">{w["tokens"].format(k=number(f[k + "_prompt"], lang))}</span></td>'
                      for k in ("none", "short", "long")) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{w[h]}</th>' for h in ("said", "note", "none", "short", "long"))
             + f'</tr></thead><tbody>{"".join(rows)}{foot}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_ASKED = {"en": {3: "how many people the user cooks for", 4: "the favorite dish",
                 5: "the shopping day"},
          "nl": {3: "voor hoeveel mensen de gebruiker kookt", 4: "het lievelingsgerecht", 5: "de dag van de boodschappen"}}


def _times(k: int, lang: str) -> str:
    if lang == "nl":
        return {1: "één keer", 2: "twee keer"}.get(k, f"{k} keer")
    return {1: "once", 2: "twice"}.get(k, f"{k} times")


def _listed(items: list, lang: str) -> str:
    last = " en " if lang == "nl" else " and "
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + last + items[-1]


def measured_text(lang: str) -> str:
    """What the runs show, in the lesson's language, with the measured numbers."""
    f = findings()
    n, ways = f["facts"], f["ways"]
    same = f["short"] == f["long"]
    about = _listed([_ASKED[lang][i] for i in f["long"]], lang)
    if lang == "nl":
        recall = (f"Direct nadat de gebruiker het zei, in hetzelfde gesprek, klopten er {len(f['short'])} van de {n}, en in een nieuw gesprek met de "
                  f"notities {len(f['long'])}" + (f", beide keren bij dezelfde vragen: {about}." if same else f": {about}."))
        return (f"Met `tool_choice` op `\"auto\"` riep Qwen3 `remember` in {len(f['called'])} van de {n} gesprekken aan. Het "
                f"schreef de notitie in plaats daarvan in zijn antwoord: {_times(len(ways['json']), 'nl')} als de call in "
                f"JSON-tekst, {_times(len(ways['tags']), 'nl')} tussen tags `<remember>`, en {_times(len(ways['remember']), 'nl')} "
                f"na *Remember:*. Daar krijgt je programma geen tool call van, dus er wordt niets bewaard. Met `tool_choice` die "
                f"`remember` noemt, bewaarde het in alle {len(f['saved'])} een notitie: steeds de zin van de gebruiker, woord "
                f"voor woord, zoals *{f['notes'][0]}*\n\n"
                f"Zonder geheugen klopten {len(f['none'])} van de {n} antwoorden. {recall} Bij de {len(f['yes_no'])} vragen met "
                f"ja of nee telde een antwoord alleen als het nee zei en de reden noemde. Met de notities zei Qwen3 ja tegen "
                f"pasta with pesto ondanks de notenallergie, en tegen de apple pie zonder oven; bij de omelette zei het alleen *"
                f"{answer('long', 2)['text']}* De notities brachten de feiten terug, maar het model moet ze ook gebruiken, en "
                f"dat ging hier mis. De notities kostten {number(f['long_prompt'], 'nl')} tokens aan prompts voor de {n} vragen, "
                f"tegen {number(f['none_prompt'], 'nl')} zonder.\n\n"
                f"De notities staan in de eerste persoon, zoals de gebruiker ze zei: *{f['notes'][0]}* In de system prompt van "
                f"een nieuw gesprek zegt niets meer wie *I* is. Een notitie moet zeggen over wie ze gaat, zoals het voorbeeld in "
                f"de beschrijving van de tool: *The user has no oven.*")
    recall = (f"Right after the user said it, in the same conversation, {len(f['short'])} of the {n} were right, and in a new conversation with the "
              f"notes {len(f['long'])}" + (f", both times for the same questions: {about}." if same else f": {about}."))
    return (f"With `tool_choice` on `\"auto\"`, Qwen3 called `remember` in {len(f['called'])} of the {n} conversations. It "
            f"wrote the note into its answer instead: {_times(len(ways['json']), 'en')} as the call in JSON text, "
            f"{_times(len(ways['tags']), 'en')} between `<remember>` tags, and {_times(len(ways['remember']), 'en')} after "
            f"*Remember:*. Your program gets no tool call from that, so nothing is saved. With `tool_choice` naming "
            f"`remember`, it saved a note in all {len(f['saved'])}: each time the user's own sentence, word for word, such as "
            f"*{f['notes'][0]}*\n\n"
            f"Without memory, {len(f['none'])} of the {n} answers were right. {recall} For the {len(f['yes_no'])} yes-or-no "
            f"questions, an answer counted only when it said no and named the reason. With the notes, Qwen3 said yes to pasta "
            f"with pesto despite the nut allergy, and to the apple pie without an oven; for the omelette, it said only "
            f"*{answer('long', 2)['text']}* The notes brought the facts back, but the model still has to use them, and that "
            f"went wrong here. The notes cost {number(f['long_prompt'], 'en')} prompt tokens for the {n} questions, against "
            f"{number(f['none_prompt'], 'en')} without them.\n\n"
            f"The notes are in the first person, as the user said them: *{f['notes'][0]}* In the system prompt of a new "
            f"conversation, nothing says any more who *I* is. A note should say who it is about, as the example in the "
            f"tool's description does: *The user has no oven.*")


WORKED = 3   # the fact of the worked example: cooking for four people


def _call_text(steps: list) -> str:
    c = steps[0]["tool_calls"][0]["function"]
    return f"{c['name']}({c['arguments']})"


def worked_text(lang: str) -> str:
    """The worked example: one fact from the first conversation, through the notes file, to the answer in a new one."""
    fact, question, _ = FACTS[WORKED]
    steps = told_steps(WORKED, "saved")
    after = steps[-1]["text"]
    none, long = answer("none", WORKED), answer("long", WORKED)
    n = len(saved_notes())
    if lang == "nl":
        return (f"### Uitgewerkt voorbeeld\n\nFeit {WORKED + 1}: de gebruiker zegt *{fact}* Met `tool_choice` die `remember` "
                f"noemt, is het eerste antwoord een call: `{_call_text(steps)}`. Je programma voert hem uit: de notitie komt "
                f"aan het eind van het bestand met notities, en het resultaat *Saved.* gaat terug, waarna Qwen3 antwoordt: "
                f"*{after}* In een nieuw gesprek leest je code het bestand en zet de {n} notities in de system prompt, onder "
                f"*{NOTES_HEAD}* De vraag *{question}* krijgt nu: *{long['text']}* Dat request had {long['prompt_tokens']} "
                f"tokens aan prompt; zonder de notities waren het er {none['prompt_tokens']}, en was het antwoord "
                f"*{none['text']}*")
    return (f"### Worked example\n\nFact {WORKED + 1}: the user says *{fact}* With `tool_choice` naming `remember`, the first "
            f"answer is a call: `{_call_text(steps)}`. Your program runs it: the note goes at the end of the notes file, and "
            f"the result *Saved.* goes back, after which Qwen3 answers: *{after}* In a new conversation, your code reads the "
            f"file and puts the {n} notes in the system prompt, under *{NOTES_HEAD}* The question *{question}* now gets: "
            f"*{long['text']}* That request had {long['prompt_tokens']} prompt tokens; without the notes it had "
            f"{none['prompt_tokens']}, and the answer was *{none['text']}*")


_WORKED_WORDS = {
    "en": {"title": "One note, from the first conversation to a new one", "first": "1. The first conversation",
           "file": "2. The notes file, after all {n} conversations", "new": "3. A new conversation",
           "without": "Without the notes, the same question got:",
           "alt": "Three steps. In the first conversation, the user tells a fact and the model calls the tool remember with it as the note. The notes file then holds every saved note, one per line. A new conversation starts with the notes in its system prompt, and the model answers the question that needs the fact."},
    "nl": {"title": "Eén notitie, van het eerste gesprek naar een nieuw", "first": "1. Het eerste gesprek",
           "file": "2. Het bestand met notities, na alle {n} gesprekken", "new": "3. Een nieuw gesprek",
           "without": "Zonder de notities kreeg dezelfde vraag:",
           "alt": "Drie stappen. In het eerste gesprek vertelt de gebruiker een feit, en het model roept de tool remember aan met dat feit als notitie. Het bestand met notities bevat daarna elke bewaarde notitie, één per regel. Een nieuw gesprek begint met de notities in zijn system prompt, en het model beantwoordt de vraag die het feit nodig heeft."},
}


def worked_picture(i: int, lang: str) -> str:
    """One fact on its way: the call of remember in the first conversation, the notes file, and the new conversation."""
    w = _WORKED_WORDS[lang]
    fact, question, _ = FACTS[i]
    steps = told_steps(i, "saved")
    notes = saved_notes()
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin:0.5rem 0 0.2rem"
    first = (_message("system", TOLD) + _message("user", fact) + _message("assistant", _call_text(steps))
             + _message("tool", "Saved.") + _message("assistant", steps[-1]["text"]))
    file_ = (f'<pre style="margin:0;border:1px solid var(--ea-line);border-radius:4px;padding:0.3rem 0.5rem;'
             f'white-space:pre-wrap">{_code(chr(10).join(notes), "var(--ea-data-text)")}</pre>')
    new = (_message("notes", memory_system(notes)) + _message("user", question)
           + _message("assistant", answer("long", i)["text"])
           + f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.3rem">{html.escape(w["without"])} '
           f'{_code(answer("none", i)["text"], "var(--ea-pred-text)")}</div>')
    body = (f'<div style="{head}">{w["first"]}</div>{first}'
            f'<div style="{head}">{html.escape(w["file"].format(n=len(FACTS)))}</div>{file_}'
            f'<div style="{head}">{w["new"]}</div>{new}')
    return yours.figure(w["title"], body, alt=w["alt"])


# --- exercise 1: the tool remember ------------------------------------------------------------------------------------

CHECK_NOTES = paths.WORK / "agent-memory" / "check" / "notes.txt"   # the notes file of the checks
DEMO_NOTES = paths.WORK / "agent-memory" / "notes.txt"             # the notes file of the demo, which stays

START_REMEMBER = '''def remember(arguments):
    note = arguments["note"]
    ...
    return "Saved."
'''
SOLUTION_REMEMBER = START_REMEMBER.replace('''    ...
''', '''    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(note + "\\n")
''')


def _file(path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _set_file(path, text: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text, encoding="utf-8")


class Told(list):
    """Calls of remember, one after the other, on a notes file with these lines beforehand (None: no file yet)."""

    def __init__(self, name: str, calls: list, before: str | None):
        super().__init__(calls)
        self.name, self.before = name, before

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


def _note(text: str) -> dict:
    return {"note": text}


T_TWO = Told("two_notes", [_note("The user is allergic to nuts."), _note("The user has no oven.")], None)
T_MORE = Told("one_more_note", [_note("The user shops on Saturdays.")], "The user does not eat eggs.\nThe user cooks for four people.\n")
T_FIRST = Told("first_note", [_note("My favorite dish is rice pudding.")], None)
REMEMBER_CASES = [(T_TWO,), (T_MORE,), (T_FIRST,)]


class RememberBox(FunctionBox):
    """A FunctionBox for remember: each case starts from its own notes file, calls the learner's remember once for each
    note, and gives what the last call gave back and the file afterwards."""

    def _run(self, fn, case):
        told = case[0]
        _set_file(CHECK_NOTES, told.before)
        returned = None
        for arguments in told:
            returned = fn(dict(arguments))
        return {"returned": returned, "file": _file(CHECK_NOTES)}

    def _call_text(self, case) -> str:
        return "; ".join(f"remember({json.dumps(a)})" for a in case[0])


def _lines(text) -> list | None:
    return None if text is None else [line.strip() for line in str(text).splitlines() if line.strip()]


def _same_file(got, want) -> bool:
    """The same notes, line for line, whatever the empty lines; and the same text given back."""
    return (isinstance(got, dict) and got.get("returned") == want["returned"]
            and _lines(got.get("file")) == _lines(want["file"]))


def _file_shown(v) -> str:
    if not isinstance(v, dict):
        return repr(v)
    return f"{v.get('returned')!r}, file {v.get('file')!r}"


_OVERWRITE = SOLUTION_REMEMBER.replace('open(NOTES, "a"', 'open(NOTES, "w"')
_NO_NEWLINE = SOLUTION_REMEMBER.replace('f.write(note + "\\n")', "f.write(note)")
_WHOLE = SOLUTION_REMEMBER.replace('f.write(note + "\\n")', 'f.write(str(arguments) + "\\n")')
_READ_MODE = SOLUTION_REMEMBER.replace('open(NOTES, "a", encoding="utf-8")', 'open(NOTES, encoding="utf-8")')
MEMORY = {"NOTES": str(CHECK_NOTES), "SYSTEM": SYSTEM, "NOTES_HEAD": NOTES_HEAD}

_REMEMBER_WORDS = {
    "en": {"title": "The notes file after your remember", "before": "before", "after": "after your calls",
           "course": "after the course's", "empty": "(no file yet)", "error": "Your remember stops with {e}."},
    "nl": {"title": "Het bestand met notities na je remember", "before": "vooraf", "after": "na je calls",
           "course": "na die van de course", "empty": "(nog geen bestand)", "error": "Je remember stopt met {e}."},
}


def _file_box(label: str, text: str | None, color: str, w: dict) -> str:
    shown = w["empty"] if text is None else text
    return (f'<div style="min-width:0"><div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(label)}</div>'
            f'<pre style="margin:0.2rem 0;padding:0.3rem 0.5rem;border:1px solid var(--ea-line);border-radius:4px;'
            f'white-space:pre-wrap">{_code(shown, color)}</pre></div>')


def remember_picture(seen) -> str:
    """The notes file of the case with two notes: before, after the learner's calls, and after the course's."""
    run_ = seen.run
    fn = run_.names.get("remember") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _REMEMBER_WORDS[seen.lang]
    box = EXERCISES["remember"]
    want = box._run(box._fn(SOLUTION_REMEMBER), (T_TWO,))
    try:
        got = box._run(fn, (T_TWO,))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    ok = _same_file(got, want)
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:0.8rem">'
            f'{_file_box(w["before"], T_TWO.before, "var(--ea-ink-2)", w)}'
            f'{_file_box(w["after"], got["file"], "var(--ea-data-text)" if ok else "var(--ea-loss-text)", w)}'
            + ("" if ok else _file_box(w["course"], want["file"], "var(--ea-data-text)", w)) + "</div>")
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: the first messages of a new conversation ---------------------------------------------------------------

START_FIRST = '''def first_messages(question):
    system = SYSTEM
    ...
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]
'''
SOLUTION_FIRST = START_FIRST.replace('''    ...
''', '''    try:
        with open(NOTES, encoding="utf-8") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        notes = ""
    if notes:
        system = SYSTEM + "\\n\\n" + NOTES_HEAD + "\\n" + notes
''')


class Asked(str):
    """A question, asked when the notes file has these lines (None: no file yet)."""

    def __new__(cls, name: str, question: str, notes: str | None):
        s = super().__new__(cls, question)
        s.name, s.notes = name, notes
        return s

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


A_PESTO = Asked("pesto_question", FACTS[0][1], "The user is allergic to nuts.\nThe user has no oven.\n")
A_NO_FILE = Asked("question_without_notes_file", FACTS[5][1], None)
A_EMPTY = Asked("question_with_empty_notes_file", FACTS[3][1], "")
A_ONE = Asked("question_with_one_note", FACTS[4][1], "The user's favorite dish is rice pudding.\n")
FIRST_CASES = [(A_PESTO,), (A_NO_FILE,), (A_EMPTY,), (A_ONE,)]


class FirstBox(FunctionBox):
    """A FunctionBox for first_messages: each case first writes its own notes file, or takes it away."""

    def _run(self, fn, case):
        _set_file(CHECK_NOTES, case[0].notes)
        return fn(str(case[0]))

    def _call_text(self, case) -> str:
        return f"first_messages({case[0]!r})"


def _flat(m) -> tuple | None:
    if not isinstance(m, dict):
        return None
    return m.get("role"), None if m.get("content") is None else "\n".join(_lines(m.get("content")))


def _same_messages(got, want) -> bool:
    """The same messages, with each content the same line for line, whatever the empty lines and spaces at the ends."""
    return isinstance(got, list) and len(got) == len(want) and all(_flat(g) == _flat(x) for g, x in zip(got, want))


def _no_file(e) -> bool:
    return isinstance(e, FileNotFoundError)


def _in_question(got, text) -> bool:
    return (isinstance(got, list) and len(got) == 2 and isinstance(got[0], dict) and isinstance(got[1], dict)
            and got[0].get("content") == SYSTEM and "allergic to nuts" in str(got[1].get("content")))


_NO_HEAD = SOLUTION_FIRST.replace('SYSTEM + "\\n\\n" + NOTES_HEAD + "\\n" + notes', 'SYSTEM + "\\n\\n" + notes')
_ALWAYS_HEAD = SOLUTION_FIRST.replace('''    if notes:
        system = SYSTEM''', '''    system = SYSTEM''')
_NO_TRY = SOLUTION_FIRST.replace('''    try:
        with open(NOTES, encoding="utf-8") as f:
            notes = f.read().strip()
    except FileNotFoundError:
        notes = ""
''', '''    with open(NOTES, encoding="utf-8") as f:
        notes = f.read().strip()
''')

_FIRST_WORDS = {
    "en": {"title": "The first messages your function builds, for the question about pesto", "error": "Your first_messages stops with {e}.",
           "course": "The course's messages:"},
    "nl": {"title": "De eerste berichten die je functie maakt, voor de vraag over pesto", "error": "Je first_messages stopt met {e}.",
           "course": "De berichten van de course:"},
}


def _messages_html(messages) -> str:
    if not isinstance(messages, list):
        return _code(repr(messages), "var(--ea-loss-text)")
    out = []
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role"))
            role = "notes" if role == "system" and NOTES_HEAD in str(m.get("content")) else role
            out.append(_message(role if role in ("system", "user", "assistant", "notes") else "user", str(m.get("content"))))
        else:
            out.append(_code(repr(m), "var(--ea-loss-text)"))
    return "".join(out)


def first_picture(seen) -> str:
    """The messages the learner's first_messages builds for the question about pesto, beside the course's when they differ."""
    run_ = seen.run
    fn = run_.names.get("first_messages") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _FIRST_WORDS[seen.lang]
    box = EXERCISES["first-messages"]
    want = box._run(box._fn(SOLUTION_FIRST), (A_PESTO,))
    try:
        got = box._run(fn, (A_PESTO,))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    body = _messages_html(got)
    if not _same_messages(got, want):
        body += f'<p style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.5rem 0 0">{html.escape(w["course"])}</p>' + _messages_html(want)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _read_mode(e) -> bool:
    """A file opened to read: there is no file yet, or it cannot be written to."""
    return isinstance(e, FileNotFoundError) or (type(e).__name__ == "UnsupportedOperation" and "not writable" in str(e))


EXERCISES = {
    "remember": RememberBox(
        name="remember",
        picture=remember_picture,
        function="remember",
        start=START_REMEMBER,
        solution=SOLUTION_REMEMBER,
        names=MEMORY,
        visible=REMEMBER_CASES[:1],
        hidden=REMEMBER_CASES[1:],
        compare=_same_file,
        show=_file_shown,
        mistakes=[
            Mistake("read-mode", error=_read_mode,
                    en="With `{call}`, your code opens the notes file to read it, the way `open` does without a mode. Open it to add to: `open(NOTES, \"a\", encoding=\"utf-8\")`.",
                    nl="Met `{call}` opent je code het bestand met notities om te lezen, zoals `open` doet zonder modus. Open het om aan toe te voegen: `open(NOTES, \"a\", encoding=\"utf-8\")`."),
            Mistake("overwrite", wrong=_OVERWRITE,
                    en="With `{call}`, the second note took the place of the first: `\"w\"` empties the file each time it opens it. Open it with `\"a\"`, so that each note goes at the end.",
                    nl="Met `{call}` nam de tweede notitie de plaats in van de eerste: `\"w\"` maakt het bestand leeg elke keer dat het hem opent. Open het met `\"a\"`, dan komt elke notitie aan het eind."),
            Mistake("no-newline", wrong=_NO_NEWLINE,
                    en="With `{call}`, the two notes ended up on one line: `f.write` adds no newline by itself. Write `note + \"\\n\"`, so that each note is a line of its own.",
                    nl="Met `{call}` kwamen de twee notities op één regel: `f.write` zet zelf geen nieuwe regel. Schrijf `note + \"\\n\"`, dan is elke notitie een eigen regel."),
            Mistake("whole", wrong=_WHOLE,
                    en="With `{call}`, the file holds the whole dictionary of arguments. Write only the note: `note`, which the first line takes out of it.",
                    nl="Met `{call}` staat de hele dictionary met argumenten in het bestand. Schrijf alleen de notitie: `note`, die de eerste regel eruit haalt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The function gives back \"Saved.\", but saves nothing yet. Open the notes file `NOTES` to add to, and write the note on a line of its own.",
                      "nl": "De functie geeft \"Saved.\" terug, maar bewaart nog niets. Open het bestand met notities `NOTES` om aan toe te voegen, en schrijf de notitie op een eigen regel."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}. Compare the files in the picture.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn. Vergelijk de bestanden in het plaatje."},
            "almost": {"en": "Right for two notes in a new file. But with `{call}`, your function gives back {got}, and it should be {want}. Does it keep the notes that were already in the file?",
                       "nl": "Goed voor twee notities in een nieuw bestand. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn. Houdt hij de notities die al in het bestand stonden?"},
            "correct": {"en": "Your `remember` is the agent's long-term memory: each note goes at the end of the file, on a line of its own, and stays there after the conversation ends. Right for all {n} cases of the check.",
                        "nl": "Je `remember` is het langetermijngeheugen van de agent: elke notitie komt aan het eind van het bestand, op een eigen regel, en blijft daar als het gesprek voorbij is. Goed bij alle {n} gevallen van de check."},
        },
    ),
    "first-messages": FirstBox(
        name="first-messages",
        picture=first_picture,
        function="first_messages",
        start=START_FIRST,
        solution=SOLUTION_FIRST,
        names=MEMORY,
        visible=FIRST_CASES[:1],
        hidden=FIRST_CASES[1:],
        compare=_same_messages,
        mistakes=[
            Mistake("no-file", error=_no_file,
                    en="With `{call}`, there is no notes file yet, and your function stops with `FileNotFoundError`. A new user has no notes: catch it with `try` and `except FileNotFoundError:`, and go on without notes.",
                    nl="Met `{call}` is er nog geen bestand met notities, en je functie stopt met `FileNotFoundError`. Een nieuwe gebruiker heeft geen notities: vang het op met `try` en `except FileNotFoundError:`, en ga verder zonder notities."),
            Mistake("no-head", wrong=_NO_HEAD,
                    en="With `{call}`, the notes are in the system prompt without the line that says what they are. Put `NOTES_HEAD` right before them, so that the model reads them as notes from earlier conversations.",
                    nl="Met `{call}` staan de notities in de system prompt zonder de regel die zegt wat ze zijn. Zet `NOTES_HEAD` er direct vóór, dan leest het model ze als notities uit eerdere gesprekken."),
            Mistake("always-head", wrong=_ALWAYS_HEAD,
                    en="With `{call}`, there are no notes, and your system prompt still has the line about notes. Add the line and the notes only when there are any: `if notes:`.",
                    nl="Met `{call}` zijn er geen notities, en je system prompt heeft toch de regel over notities. Voeg de regel en de notities alleen toe als die er zijn: `if notes:`."),
            Mistake("in-question", matches=_in_question,
                    en="With `{call}`, the notes are in the user's message. Put them in the system prompt: that holds what always applies, and the user's message stays the question.",
                    nl="Met `{call}` staan de notities in het bericht van de gebruiker. Zet ze in de system prompt: daar staat wat altijd geldt, en het bericht van de gebruiker blijft de vraag."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The function gives back the system prompt and the question, without the notes. Read the notes file `NOTES`, and when it has notes, add `NOTES_HEAD` and the notes to the system prompt.",
                      "nl": "De functie geeft de system prompt en de vraag terug, zonder de notities. Lees het bestand met notities `NOTES`, en als er notities in staan, voeg dan `NOTES_HEAD` en de notities toe aan de system prompt."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should be `{want}`. Compare the messages in the picture.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Vergelijk de berichten in het plaatje."},
            "almost": {"en": "Right for the question about pesto. But with `{call}`, your function gives back `{got}`, and it should be `{want}`. Does it work when there is no notes file yet, or an empty one?",
                       "nl": "Goed voor de vraag over pesto. Maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Werkt het ook als er nog geen bestand met notities is, of een leeg?"},
            "correct": {"en": "Your `first_messages` starts a new conversation with what the agent remembers: the notes from the file in the system prompt, under their heading, and nothing extra for a user without notes. Right for all {n} cases of the check.",
                        "nl": "Je `first_messages` begint een nieuw gesprek met wat de agent onthoudt: de notities uit het bestand in de system prompt, onder hun kop, en niets extra voor een gebruiker zonder notities. Goed bij alle {n} gevallen van de check."},
        },
    ),
}

REQUIRED = ["remember", "first-messages", "stale-note"]

CASES = [
    ("remember", START_REMEMBER, "empty", None),
    ("remember", "", "empty", None),
    ("remember", SOLUTION_REMEMBER, "correct", None),
    ("remember", _READ_MODE, "error", "read-mode"),
    ("remember", _OVERWRITE, "wrong", "overwrite"),
    ("remember", _NO_NEWLINE, "wrong", "no-newline"),
    ("remember", _WHOLE, "wrong", "whole"),
    ("remember", SOLUTION_REMEMBER.replace('f.write(note + "\\n")', 'f.write("\\n" + note)'), "correct", None),
    ("remember", SOLUTION_REMEMBER.replace('f.write(note + "\\n")', "print(note, file=f)"), "correct", None),
    ("first-messages", START_FIRST, "empty", None),
    ("first-messages", "", "empty", None),
    ("first-messages", SOLUTION_FIRST, "correct", None),
    ("first-messages", _NO_TRY, "error", "no-file"),
    ("first-messages", _NO_HEAD, "wrong", "no-head"),
    ("first-messages", _ALWAYS_HEAD, "wrong", "always-head"),
    ("first-messages", SOLUTION_FIRST.replace('system = SYSTEM + "\\n\\n" + NOTES_HEAD + "\\n" + notes',
                                              'question = NOTES_HEAD + "\\n" + notes + "\\n\\n" + question'), "wrong", "in-question"),
    ("first-messages", SOLUTION_FIRST.replace('SYSTEM + "\\n\\n" + NOTES_HEAD', 'SYSTEM + "\\n" + NOTES_HEAD'), "correct", None),
    ("first-messages", SOLUTION_FIRST.replace("notes = f.read().strip()", "notes = f.read()"), "correct", None),
]


# --- the demo: Qwen3 with the learner's remember and first_messages ---------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


DEMO = {"NOTES": str(DEMO_NOTES), "SYSTEM": SYSTEM, "NOTES_HEAD": NOTES_HEAD}


def learner_fn(key: str, text):
    """The learner's function from an exercise, when it is right, with the demo's notes file; else None."""
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    DEMO_NOTES.parent.mkdir(parents=True, exist_ok=True)
    return run_code(text, dict(DEMO)).names.get(ex.function)


def demo_notes() -> str | None:
    return _file(DEMO_NOTES)


def forget() -> None:
    """Take the demo's notes file away: the agent then starts with no notes again."""
    DEMO_NOTES.unlink(missing_ok=True)


def _post(url: str, messages: list, tools: list | None = None, choice: dict | None = None) -> dict:
    import urllib.request
    body = {"model": serve.MODEL, "messages": messages, "max_tokens": 120, "temperature": 0}
    if tools:
        body["tools"] = tools
    if choice:
        body["tool_choice"] = choice
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["choices"][0]["message"]


def _notes_box(words: dict) -> str:
    text = demo_notes()
    return (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.5rem">{html.escape(words["file"])}</div>'
            f'<pre style="margin:0.2rem 0;white-space:pre-wrap">{_code(words["no_file"] if text is None else text, "var(--ea-data-text)")}</pre>')


def tell(remember_fn, said: str, words: dict) -> str:
    """Something the learner says to Qwen3 with the tool remember: each call runs the learner's remember, and the model
    then answers; the notes file afterwards."""
    try:
        running = serve.model_server()
        messages = [{"role": "system", "content": TOLD}, {"role": "user", "content": said}]
        message = _post(running.url, messages, [REMEMBER], NAMED)
    except Exception as e:   # the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        raise
    lines, results = [], []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
            result = remember_fn(arguments) if c["function"]["name"] == "remember" else "Error: no tool " + c["function"]["name"]
        except Exception as e:   # noqa: BLE001 - a call from a model, run by the learner's code
            result = f"Error: {type(e).__name__}: {e}"
        lines.append(_code(f"{c['function']['name']}({c['function']['arguments']}) = {result}", "var(--ea-pred-text)"))
        results.append({"role": "tool", "tool_call_id": c["id"], "content": str(result)})
    if results:
        message = _post(running.url, messages + [message] + results, [REMEMBER])
    body = ("".join(f"<div>{x}</div>" for x in lines) if lines else
            f'<p style="{_UI};font-size:13px;color:var(--ea-ink-2)">{html.escape(words["no_call"])}</p>')
    body += _message("assistant", message.get("content") or "") + _notes_box(words)
    return yours.figure(words["title"], body, alt=words["title"])


def ask_new(first_fn, question: str, words: dict) -> str:
    """A question in a new conversation: the messages the learner's first_messages builds, and Qwen3's answer."""
    try:
        messages = first_fn(question)
        running = serve.model_server()
        message = _post(running.url, messages)
    except Exception as e:   # noqa: BLE001 - the learner's code, or the real server, which may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        return f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(words["error"].format(e=f"{type(e).__name__}: {e}"))}</p>'
    body = _messages_html(messages) + _message("assistant", message.get("content") or "")
    return yours.figure(words["title"], body, alt=words["title"])
