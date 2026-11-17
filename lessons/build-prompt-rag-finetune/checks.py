# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-prompt-rag-finetune: the three ways to change what an app's model does (the prompt, RAG and
fine-tuning) and the data a provider fine-tunes on. The course's recorded replies of Qwen3-0.6B to the 120 test questions
of the online bank (data/course-production-qwen.json), with version 1 and version 2 of the prompt, what the eval set of
build-eval-set says about each and the notes the course wrote by hand (build-error-analysis, build-eval-online). Two
functions to write (`examples`, the training examples a provider takes, and `pair`, a preference between two replies),
two choices, and the pictures."""
import copy
import html
import importlib.util
import json

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _load(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_prompt_rag_finetune",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("build-eval-online")   # the notes on the replies of version 2
V = O.V                          # the eval set, version 2 of the prompt and the replies of both versions
E = V.E                          # the 120 test questions and version 1


class Named(list):
    """A list of the lesson's, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [copy.deepcopy(x, memo) for x in self]


class NamedDict(dict):
    """A dictionary of the lesson's, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return {k: copy.deepcopy(v, memo) for k, v in self.items()}


class NamedText(str):
    """A text of the lesson's, written in the check's messages by its name."""

    def __new__(cls, text: str, name: str):
        s = super().__new__(cls, text)
        s.name = name
        return s

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return str(self)


SYSTEM = V.SYSTEM_V2   # the system prompt of version 2: the six topics with their facts, and the form of the answer
QUESTIONS = Named(E.QUESTIONS, "QUESTIONS")


def _replies(version: str) -> list:
    passed = V.run(version)["all"]
    return [{"reply": text, "passed": bool(p), "notes": list(n)} for text, p, n in zip(V.texts(version), passed, O.NOTES[version])]


V1 = Named(_replies("v1"), "V1")   # per test question: the reply of version 1, whether it passed the eval set, the course's notes
V2 = Named(_replies("v2"), "V2")   # the same for version 2


# --- the course's solutions, and what they give on the recordings -----------------------------------------------------

def examples(questions, replies):
    out = []
    for i in range(len(questions)):
        r = replies[i]
        if r["passed"] and r["notes"] == []:
            out.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": r["reply"]},
            ]})
    return out


def pair(question, a, b):
    if a["passed"] and not b["passed"]:
        return (question, a["reply"], b["reply"])
    if b["passed"] and not a["passed"]:
        return (question, b["reply"], a["reply"])
    return None


def findings() -> dict:
    n = len(QUESTIONS)
    passed = [i for i in range(n) if V2[i]["passed"]]
    pairs = [pair(QUESTIONS[i], V1[i], V2[i]) for i in range(n)]
    v2_better = [i for i in range(n) if pairs[i] is not None and pairs[i][1] == V2[i]["reply"]]
    v1_better = [i for i in range(n) if pairs[i] is not None and pairs[i][1] == V1[i]["reply"]]
    better_made_up = [i for i in v2_better if "made-up" in V2[i]["notes"]] + [i for i in v1_better if "made-up" in V1[i]["notes"]]
    return {
        "questions": n,
        "passed": {"v1": sum(r["passed"] for r in V1), "v2": len(passed)},
        "made_up": {"v1": sum("made-up" in r["notes"] for r in V1), "v2": sum("made-up" in r["notes"] for r in V2)},
        "passed_made_up": sum("made-up" in V2[i]["notes"] for i in passed),
        "passed_noted": sum(bool(V2[i]["notes"]) for i in passed),
        "examples": len(examples(QUESTIONS, V2)),
        "pairs": len(v2_better) + len(v1_better), "v2_better": len(v2_better), "v1_better": len(v1_better),
        "both": sum(V1[i]["passed"] and V2[i]["passed"] for i in range(n)),
        "neither": sum(not V1[i]["passed"] and not V2[i]["passed"] for i in range(n)),
        "better_made_up": len(better_made_up),
        "topics": len(V.FACTS_BY_TOPIC), "system_chars": len(SYSTEM),
        "fact_sentences": sum(facts.count(". ") + 1 for facts in V.FACTS_BY_TOPIC.values()),
    }


def number(x: float, lang: str) -> str:
    text = f"{x:,}"
    return text.replace(",", ".") if lang == "nl" else text


# --- pictures ----------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"

_WORDS = {
    "en": {"flow_title": "What to change first",
           "flow_alt": "A flow. Something goes wrong: first a better prompt, measured on the eval set. Still wrong? If the model lacks facts, RAG. If it lacks a behavior, fine-tuning, on examples, on preferences or with a grader.",
           "start": "something goes wrong", "prompt": "1. a better prompt", "measure": "measure on the eval set",
           "still": "still wrong?", "facts": "facts are missing", "behavior": "a behavior is missing",
           "rag": "2. RAG", "rag_sub": "look the facts up", "tune": "3. fine-tuning", "tune_sub": "examples, preferences or a grader",
           "example_title": "One training example; in the file it is one line",
           "example_alt": "The example as JSON: the system prompt, the customer's question with the role user, and the reply with the role assistant.",
           "system_cut": "… ({n} characters)",
           "ex_title": "Your examples(QUESTIONS, V2) on the 120 replies of version 2",
           "ex_alt": "A square for each of the 120 replies of version 2: filled when your function made it a training example, with a dark ring when the course's notes say the reply made something up.",
           "ex_key": ["a training example", "made something up, says the course"],
           "ex_sum": "Your function makes {k} examples of the {n} replies; {m} of them made something up, says the course.",
           "ex_none": "Your function makes no examples.", "ex_failed": "Your function gives no list of examples.",
           "ex_first": "The first line of the file:",
           "pair_title": "Your pair(question, a, b) on the 120 questions, with a = V1[i] and b = V2[i]",
           "pair_alt": "A square for each of the 120 questions: filled in one color when your function prefers the reply of version 2, in another when it prefers version 1, and empty when it makes no pair.",
           "pair_key": ["version 2 is better", "version 1 is better", "no pair"],
           "pair_sum": "Your function makes {k} pairs: in {b} the reply of version 2 is the better one, in {a} the reply of version 1.",
           "pair_failed": "For {f} of the {n} questions your function gives neither None nor a (question, better, worse)."},
    "nl": {"flow_title": "Wat je eerst verandert",
           "flow_alt": "Een stroom. Er gaat iets mis: eerst een betere prompt, gemeten op de eval set. Nog steeds fout? Mist het model feiten, dan RAG. Mist het een gedrag, dan fine-tuning, op voorbeelden, op voorkeuren of met een grader.",
           "start": "er gaat iets mis", "prompt": "1. een betere prompt", "measure": "meten op de eval set",
           "still": "nog steeds fout?", "facts": "er ontbreken feiten", "behavior": "er ontbreekt een gedrag",
           "rag": "2. RAG", "rag_sub": "de feiten opzoeken", "tune": "3. fine-tuning", "tune_sub": "voorbeelden, voorkeuren of een grader",
           "example_title": "Eén trainingsvoorbeeld; in het bestand is het één regel",
           "example_alt": "Het voorbeeld als JSON: de system prompt, de vraag van de klant met de rol user, en het antwoord met de rol assistant.",
           "system_cut": "… ({n} tekens)",
           "ex_title": "Jouw examples(QUESTIONS, V2) op de 120 antwoorden van versie 2",
           "ex_alt": "Een vakje voor elk van de 120 antwoorden van versie 2: gevuld als je functie er een trainingsvoorbeeld van maakte, met een donkere rand als de notities van de course zeggen dat het antwoord iets verzon.",
           "ex_key": ["een trainingsvoorbeeld", "verzon iets, zegt de course"],
           "ex_sum": "Je functie maakt {k} voorbeelden van de {n} antwoorden; {m} daarvan verzonnen iets, zegt de course.",
           "ex_none": "Je functie maakt geen voorbeelden.", "ex_failed": "Je functie geeft geen lijst met voorbeelden.",
           "ex_first": "De eerste regel van het bestand:",
           "pair_title": "Jouw pair(question, a, b) op de 120 vragen, met a = V1[i] en b = V2[i]",
           "pair_alt": "Een vakje voor elk van de 120 vragen: gevuld in de ene kleur als je functie het antwoord van versie 2 verkiest, in een andere als hij versie 1 verkiest, en leeg als hij geen paar maakt.",
           "pair_key": ["versie 2 is beter", "versie 1 is beter", "geen paar"],
           "pair_sum": "Je functie maakt {k} paren: bij {b} is het antwoord van versie 2 het betere, bij {a} het antwoord van versie 1.",
           "pair_failed": "Bij {f} van de {n} vragen geeft je functie geen None en geen (question, better, worse)."},
}


def _text(x: float, y: float, s: str, size: int = 12, color: str = "var(--ea-ink)", anchor: str = "middle") -> str:
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{_UI};font-size:{size}px;fill:{color}">'
            f'{html.escape(s)}</text>')


def _box(x: float, y: float, w: float, h: float, fill: str = "none", stroke: str = "var(--ea-ink-2)") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" style="fill:{fill};stroke:{stroke};stroke-width:1"/>'


def _arrow(path: str, end: tuple, color: str = "var(--ea-ink-2)", down: bool = False) -> str:
    x, y = end
    head = f"M{x - 4},{y - 4} l4,4 l4,-4 z" if down else f"M{x - 4},{y - 4} l4,4 l-4,4 z"
    return (f'<path d="{path}" style="fill:none;stroke:{color};stroke-width:1"/>'
            f'<path d="{head}" style="fill:{color}"/>')


def flow_picture(lang: str) -> str:
    """The idea: a better prompt first, measured; then RAG for missing facts, fine-tuning for a missing behavior."""
    w = _WORDS[lang]
    parts = [
        _box(4, 12, 150, 44), _text(79, 38, w["start"]),
        _arrow("M154,34 H178", (182, 34)),
        _box(182, 6, 180, 56, "var(--ea-inset)"), _text(272, 30, w["prompt"]), _text(272, 48, w["measure"], 11, "var(--ea-ink-2)"),
        _arrow("M362,34 H386", (390, 34)),
        _text(440, 38, w["still"], 12, "var(--ea-ink)"),
        _arrow("M440,48 V78 H300 V112", (300, 116), down=True), _text(292, 100, w["facts"], 11, "var(--ea-ink-2)", "end"),
        _arrow("M440,78 H560 V112", (560, 116), down=True), _text(552, 100, w["behavior"], 11, "var(--ea-ink-2)", "end"),
        _box(182, 116, 236, 48), _text(300, 136, w["rag"]), _text(300, 154, w["rag_sub"], 11, "var(--ea-ink-2)"),
        _box(442, 116, 236, 48, "var(--ea-data-wash)", "var(--ea-data)"), _text(560, 136, w["tune"]),
        _text(560, 154, w["tune_sub"], 11, "var(--ea-ink-2)"),
    ]
    width = 684
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} 170" width="100%" style="min-width:600px;max-width:{width}px" '
           f'role="img" aria-label="{html.escape(w["flow_alt"])}">{"".join(parts)}</svg></div>')
    return yours.figure(w["flow_title"], svg, alt=w["flow_alt"])


def _line_of(example: dict, lang: str) -> str:
    """An example as a line of the file, with the long system prompt cut short so the line can be read."""
    shown = copy.deepcopy(example)
    for m in shown.get("messages", []) if isinstance(shown, dict) else []:
        if isinstance(m, dict) and m.get("role") == "system" and isinstance(m.get("content"), str) and len(m["content"]) > 60:
            m["content"] = m["content"][:60] + _WORDS[lang]["system_cut"].format(n=number(len(m["content"]), lang))
    return json.dumps(shown, ensure_ascii=False)


def _code_block(text: str) -> str:
    return (f'<div style="{_MONO};font-size:12px;color:var(--ea-ink);white-space:pre-wrap;overflow-wrap:anywhere;'
            f'background:var(--ea-inset);padding:0.4rem 0.6rem;border-radius:4px">{html.escape(text)}</div>')


def example_picture(i: int, lang: str) -> str:
    """The training example of question i, as the course writes it."""
    w = _WORDS[lang]
    shown = json.loads(_line_of(examples([QUESTIONS[i]], [V2[i]])[0], lang))
    return yours.figure(w["example_title"], _code_block(json.dumps(shown, ensure_ascii=False, indent=2)), alt=w["example_alt"])


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def _user_of(example) -> str | None:
    if not isinstance(example, dict) or not isinstance(example.get("messages"), list):
        return None
    return next((m.get("content") for m in example["messages"] if isinstance(m, dict) and m.get("role") == "user"), None)


def examples_picture(seen) -> str:
    fn = _mine(seen, "examples")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    got = _safe(fn, list(QUESTIONS), copy.deepcopy(list(V2)))
    if not isinstance(got, list):
        return yours.figure(w["ex_title"], f'<div style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(w["ex_failed"])}</div>')
    # which question each example is for, in order
    kept, k = [False] * len(QUESTIONS), 0
    for e in got:
        user = _user_of(e)
        while k < len(QUESTIONS) and QUESTIONS[k] != user:
            k += 1
        if k < len(QUESTIONS):
            kept[k] = True
            k += 1
    made_up = ["made-up" in r["notes"] for r in V2]
    m = sum(a and b for a, b in zip(kept, made_up))
    body = (V._squares(kept, ring=made_up, fill="var(--ea-data)")
            + V._key([("var(--ea-data)", "var(--ea-line)", w["ex_key"][0]), ("var(--ea-page)", "var(--ea-ink)", w["ex_key"][1])]))
    said = w["ex_sum"].format(k=len(got), n=len(QUESTIONS), m=m) if got else w["ex_none"]
    body += f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.35rem">{html.escape(said)}</div>'
    if got:
        body += (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.35rem 0 0.15rem">{html.escape(w["ex_first"])}</div>'
                 + _code_block(_line_of(got[0], seen.lang) if isinstance(got[0], dict) else repr(got[0])))
    return yours.figure(w["ex_title"], f'<div role="img" aria-label="{html.escape(w["ex_alt"])}">{body}</div>')


def pairs_picture(seen) -> str:
    fn = _mine(seen, "pair")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    marks2, marks1, failed = [], [], 0
    for i in range(len(QUESTIONS)):
        got = _safe(fn, QUESTIONS[i], copy.deepcopy(V1[i]), copy.deepcopy(V2[i]))
        better = got[1] if isinstance(got, tuple) and len(got) == 3 else None
        if got is not None and better is None:
            failed += 1
        marks2.append(better is not None and better == V2[i]["reply"])
        marks1.append(better is not None and better == V1[i]["reply"])
    # one strip, a color per version that gave the better reply
    size, gap, per_row = 11, 3, 40
    parts = []
    for i in range(len(QUESTIONS)):
        x, y = (i % per_row) * (size + gap) + 2, (i // per_row) * (size + gap) + 2
        fill = "var(--ea-group-3)" if marks2[i] else ("var(--ea-group-2)" if marks1[i] else "var(--ea-page)")
        parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="2" style="fill:{fill};stroke:var(--ea-line);stroke-width:1"/>')
    wd = per_row * (size + gap) + 4
    hd = ((len(QUESTIONS) + per_row - 1) // per_row) * (size + gap) + 4
    strip = f'<svg viewBox="0 0 {wd} {hd}" width="100%" style="max-width:{wd}px" aria-hidden="true">{"".join(parts)}</svg>'
    key = V._key([("var(--ea-group-3)", "var(--ea-line)", w["pair_key"][0]), ("var(--ea-group-2)", "var(--ea-line)", w["pair_key"][1]),
                  ("var(--ea-page)", "var(--ea-line)", w["pair_key"][2])])
    if failed:
        said, color = w["pair_failed"].format(f=failed, n=len(QUESTIONS)), "var(--ea-loss-text)"
    else:
        said, color = w["pair_sum"].format(k=sum(marks1) + sum(marks2), b=sum(marks2), a=sum(marks1)), "var(--ea-ink)"
    body = strip + key + f'<div style="{_UI};font-size:13px;color:{color};margin-top:0.35rem">{html.escape(said)}</div>'
    return yours.figure(w["pair_title"], f'<div role="img" aria-label="{html.escape(w["pair_alt"])}">{body}</div>')


# --- exercises -----------------------------------------------------------------------------------------------------------

START_EXAMPLES = '''def examples(questions, replies):
    out = []
    for i in range(len(questions)):
        r = replies[i]
        out.append({"messages": [
            {"role": "user", "content": questions[i]},
        ]})
    return out
'''
SOLUTION_EXAMPLES = '''def examples(questions, replies):
    out = []
    for i in range(len(questions)):
        r = replies[i]
        if r["passed"] and r["notes"] == []:
            out.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": r["reply"]},
            ]})
    return out
'''
_EX_PASSED = SOLUTION_EXAMPLES.replace('''        if r["passed"] and r["notes"] == []:''', '''        if r["passed"]:''')
_EX_ALL = SOLUTION_EXAMPLES.replace('''        if r["passed"] and r["notes"] == []:''', '''        if True:''')
_EX_NO_SYSTEM = SOLUTION_EXAMPLES.replace('''                {"role": "system", "content": SYSTEM},
''', "")
_EX_WHOLE = SOLUTION_EXAMPLES.replace('''"content": r["reply"]}''', '''"content": r}''')
_EX_NO_REPLY = SOLUTION_EXAMPLES.replace('''                {"role": "assistant", "content": r["reply"]},
''', "")

START_PAIR = '''def pair(question, a, b):
    return None
'''
SOLUTION_PAIR = '''def pair(question, a, b):
    if a["passed"] and not b["passed"]:
        return (question, a["reply"], b["reply"])
    if b["passed"] and not a["passed"]:
        return (question, b["reply"], a["reply"])
    return None
'''
_PAIR_ONE_WAY = SOLUTION_PAIR.replace('''    if b["passed"] and not a["passed"]:
        return (question, b["reply"], a["reply"])
''', "")
_PAIR_WORSE_FIRST = SOLUTION_PAIR.replace('''        return (question, a["reply"], b["reply"])''', '''        return (question, b["reply"], a["reply"])''', 1).replace(
    '''        return (question, b["reply"], a["reply"])
    return None''', '''        return (question, a["reply"], b["reply"])
    return None''')
_PAIR_BOTH = SOLUTION_PAIR.replace('''    if a["passed"] and not b["passed"]:''', '''    if a["passed"]:''')

_NAMES = {"SYSTEM": SYSTEM, "QUESTIONS": QUESTIONS, "V1": V1, "V2": V2}

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


_ALMOST_CODE = {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."}
_ALMOST_WORDS = {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}.",
                 "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en het zou {want} moeten zijn."}


def _texts(name: str, empty: dict, wrong: dict, correct: dict, almost: dict = _ALMOST_CODE) -> dict:
    return {
        "missing": {"en": f"Run the cell with `{name}` first.", "nl": f"Voer eerst de cel met `{name}` uit."},
        "cleared": _CLEARED,
        "empty": empty,
        "wrong": wrong,
        "almost": almost,
        "no_function": {"en": f"Your box does not make a function `{name}`. Keep the `def` line as it was.",
                        "nl": f"Je vak maakt geen functie `{name}`. Laat de `def`-regel staan zoals hij was."},
        "correct": correct,
    }


class _Shown:
    """A value as the check's messages write it, without the quotes of a text."""

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other):
        return isinstance(other, _Shown) and other.text == self.text

    def __hash__(self):
        return hash(self.text)

    def __repr__(self):
        return self.text


def _shown_examples(v):
    """A list of examples as the number of examples, the questions they are for, and the roles of the first one."""
    if not isinstance(v, list):
        return repr(v)
    users = [_user_of(e) for e in v]
    which = ", ".join(str(QUESTIONS.index(u)) if u in QUESTIONS else "?" for u in users)
    roles = ([m.get("role") if isinstance(m, dict) else "?" for m in v[0]["messages"]]
             if v and isinstance(v[0], dict) and isinstance(v[0].get("messages"), list) else None)
    en = f"{len(v)} example{'' if len(v) == 1 else 's'}" + (f", for the questions {which}" if v else "")
    nl = f"{len(v)} voorbeeld{'' if len(v) == 1 else 'en'}" + (f", voor de vragen {which}" if v else "")
    if roles is not None:
        en += f", the first with the roles {', '.join(map(str, roles))}"
        nl += f", het eerste met de rollen {', '.join(map(str, roles))}"
    return {"en": en, "nl": nl}


class _PairBox(FunctionBox):
    """Shows what pair gives by name: the question and the two replies as `question`, `a["reply"]` and `b["reply"]`,
    so a message does not have to quote two long replies; None stays None."""

    def _run(self, fn, case):
        question, a, b = case
        got = fn(*(copy.deepcopy(x) for x in case))
        if got is None:
            return _Shown("None")
        if isinstance(got, tuple):
            names = {str(question): "question", a["reply"]: 'a["reply"]', b["reply"]: 'b["reply"]'}
            return _Shown("(" + ", ".join(names.get(x, repr(x)) if isinstance(x, str) else repr(x) for x in got) + ")")
        return _Shown(repr(got))


def _q(i: int) -> NamedText:
    return NamedText(QUESTIONS[i], f"QUESTIONS[{i}]")


def _v(version: list, i: int) -> NamedDict:
    return NamedDict(version[i], f"{version.name}[{i}]")


EXERCISES = {
    "examples": FunctionBox(
        name="examples",
        picture=examples_picture,
        function="examples",
        start=START_EXAMPLES,
        solution=SOLUTION_EXAMPLES,
        visible=[(Named(QUESTIONS[:8], "QUESTIONS[:8]"), Named(V2[:8], "V2[:8]"))],
        hidden=[(Named(QUESTIONS[8:16], "QUESTIONS[8:16]"), Named(V2[8:16], "V2[8:16]")),
                (Named(QUESTIONS[:8], "QUESTIONS[:8]"), Named(V1[:8], "V1[:8]")),
                (Named(QUESTIONS[40:60], "QUESTIONS[40:60]"), Named(V2[40:60], "V2[40:60]")),
                (Named([], "no_questions"), Named([], "no_replies"))],
        show=_shown_examples,
        mistakes=[
            Mistake("passed-only", wrong=_EX_PASSED,
                    en="`{call}` gives {got}: you also keep replies the course wrote a note about, such as a reply that made something up. The eval set does not see that; an example teaches the model what it shows. Keep a reply only when it passed and `r[\"notes\"] == []`.",
                    nl="`{call}` geeft {got}: je houdt ook antwoorden waar de course een notitie bij schreef, zoals een antwoord dat iets verzon. De eval set ziet dat niet; een voorbeeld leert het model wat het laat zien. Houd een antwoord alleen als het door de eval set kwam en `r[\"notes\"] == []`."),
            Mistake("all", wrong=_EX_ALL,
                    en="`{call}` gives {got}: every reply becomes an example, also the ones that failed the eval set. Keep a reply only when it passed and has no note.",
                    nl="`{call}` geeft {got}: elk antwoord wordt een voorbeeld, ook de antwoorden die niet door de eval set kwamen. Houd een antwoord alleen als het erdoor kwam en geen notitie heeft."),
            Mistake("no-system", wrong=_EX_NO_SYSTEM,
                    en="`{call}` gives {got}: the examples have no system prompt. The app sends `SYSTEM` with every question, so the examples need it too, as the first message, with the role `system`.",
                    nl="`{call}` geeft {got}: de voorbeelden hebben geen system prompt. De app stuurt `SYSTEM` mee met elke vraag, dus de voorbeelden hebben hem ook nodig, als eerste bericht, met de rol `system`."),
            Mistake("whole-dict", wrong=_EX_WHOLE,
                    en="`{call}` gives {got}: the assistant message holds the whole dictionary `r`, with `\"passed\"` and `\"notes\"` in it. The model should learn only the text of the reply, which `r` holds under `\"reply\"`.",
                    nl="`{call}` geeft {got}: in het assistant-bericht staat de hele dictionary `r`, met `\"passed\"` en `\"notes\"` erin. Het model moet alleen de tekst van het antwoord leren, die in `r` onder `\"reply\"` staat."),
            Mistake("no-reply", wrong=_EX_NO_REPLY,
                    en="`{call}` gives {got}: the examples have no reply. The reply is what the model learns to write: add it last, with the role `assistant`.",
                    nl="`{call}` geeft {got}: de voorbeelden hebben geen antwoord. Het antwoord is wat het model leert schrijven: zet het als laatste, met de rol `assistant`."),
        ],
        texts=_texts("examples",
                     empty={"en": "Your `examples` gives every question without a system prompt and without a reply. Put `SYSTEM` before the question and the reply after it, and keep only replies that passed and have no note.",
                            "nl": "Je `examples` geeft elke vraag zonder system prompt en zonder antwoord. Zet `SYSTEM` voor de vraag en het antwoord erna, en houd alleen antwoorden die door de eval set kwamen en geen notitie hebben."},
                     wrong={"en": "`{call}` gives {got}, and it should give {want}: for each reply that passed and has no note, one example with the messages system, user and assistant.",
                            "nl": "`{call}` geeft {got}, en het zou {want} moeten zijn: voor elk antwoord dat door de eval set kwam en geen notitie heeft, één voorbeeld met de berichten system, user en assistant."},
                     correct={"en": "Your `examples` is right in all {n} cases. In the picture, how many replies did the eval set pass that you left out?",
                              "nl": "Je `examples` klopt in alle {n} gevallen. Hoeveel antwoorden liet je in het plaatje weg die wel door de eval set kwamen?"},
                     almost=_ALMOST_WORDS),
        names=_NAMES,
    ),
    "pair": _PairBox(
        name="pair",
        picture=pairs_picture,
        function="pair",
        start=START_PAIR,
        solution=SOLUTION_PAIR,
        visible=[(_q(0), _v(V1, 0), _v(V2, 0))],
        hidden=[(_q(1), _v(V1, 1), _v(V2, 1)), (_q(2), _v(V1, 2), _v(V2, 2)), (_q(9), _v(V1, 9), _v(V2, 9)),
                (_q(5), _v(V1, 5), _v(V2, 5)), (_q(3), _v(V1, 3), _v(V2, 3)), (_q(0), _v(V2, 0), _v(V1, 0))],
        mistakes=[
            Mistake("one-way", wrong=_PAIR_ONE_WAY,
                    en="`{call}` gives `{got}`: you make a pair only when `a` is the better reply. When `b` passed and `a` did not, `b` is the better one, so it comes right after the question.",
                    nl="`{call}` geeft `{got}`: je maakt alleen een paar als `a` het betere antwoord is. Als `b` erdoor kwam en `a` niet, is `b` het betere, dus komt het direct na de vraag."),
            Mistake("worse-first", wrong=_PAIR_WORSE_FIRST,
                    en="`{call}` gives `{got}`: the worse reply comes first. A pair is the question, then the better reply, then the worse one.",
                    nl="`{call}` geeft `{got}`: het slechtere antwoord staat voorop. Een paar is de vraag, dan het betere antwoord, dan het slechtere."),
            Mistake("both", wrong=_PAIR_BOTH,
                    en="`{call}` gives `{got}`: you make a pair when both replies passed. Then neither is better on the eval set, and the pair would teach a preference no one measured: give `None`.",
                    nl="`{call}` geeft `{got}`: je maakt een paar als beide antwoorden erdoor kwamen. Dan is geen van beide beter op de eval set, en het paar zou een voorkeur leren die niemand mat: geef `None`."),
        ],
        texts=_texts("pair",
                     empty={"en": "Your `pair` never makes a pair. When exactly one of the two replies passed, give the question, the reply that passed and the one that did not.",
                            "nl": "Je `pair` maakt nooit een paar. Als precies één van de twee antwoorden erdoor kwam, geef dan de vraag, het antwoord dat erdoor kwam en het antwoord dat dat niet deed."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: a pair only when exactly one reply passed, with the better reply first.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: alleen een paar als precies één antwoord erdoor kwam, met het betere antwoord voorop."},
                     correct={"en": "Your `pair` is right in all {n} cases. In how many of the pairs in the picture is version 1 the better one?",
                              "nl": "Je `pair` klopt in alle {n} gevallen. Bij hoeveel paren in het plaatje is versie 1 de betere?"}),
        names=_NAMES,
    ),
}

REQUIRED = ["examples", "pair", "first-change", "eval-leak"]

CASES = [
    ("examples", START_EXAMPLES, "empty", None),
    ("examples", "", "empty", None),
    ("examples", SOLUTION_EXAMPLES, "correct", None),
    ("examples", '''def examples(questions, replies):
    out = []
    for question, r in zip(questions, replies):
        if not r["passed"] or r["notes"]:
            continue
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        messages.append({"role": "assistant", "content": r["reply"]})
        out.append({"messages": messages})
    return out
''', "correct", None),
    ("examples", _EX_PASSED, "wrong", "passed-only"),
    ("examples", _EX_ALL, "wrong", "all"),
    ("examples", _EX_NO_SYSTEM, "wrong", "no-system"),
    ("examples", _EX_NO_REPLY, "wrong", "no-reply"),
    ("examples", _EX_WHOLE, "wrong", "whole-dict"),
    ("pair", START_PAIR, "empty", None),
    ("pair", "", "empty", None),
    ("pair", SOLUTION_PAIR, "correct", None),
    ("pair", '''def pair(question, a, b):
    if a["passed"] == b["passed"]:
        return None
    if a["passed"]:
        return (question, a["reply"], b["reply"])
    return (question, b["reply"], a["reply"])
''', "correct", None),
    ("pair", _PAIR_ONE_WAY, "wrong", "one-way"),
    ("pair", _PAIR_WORSE_FIRST, "wrong", "worse-first"),
    ("pair", _PAIR_BOTH, "wrong", "both"),
]
