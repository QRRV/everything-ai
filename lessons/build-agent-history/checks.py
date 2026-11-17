# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-history: before today's agents. MRKL (2022) sent each question to a module, HuggingGPT (2023)
let a model call other models, and AutoGPT and BabyAGI (2023) let a model set its own tasks. The course's own small
BabyAGI: a list of tasks toward an objective, of which the model does the first, then writes new tasks from the result.
The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): three objectives, five rounds each, once with
every new task added to the list, as BabyAGI did, and once with a list that skips repeats and holds at most five tasks.
What they show, their pictures, the exercises (the model's lines as tasks, and adding tasks to the list) and the demo."""
import copy
import html
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"
ROUNDS = 5   # each round: the model does the first task, then writes new tasks
LIMIT = 5    # the most tasks the kept list holds
OBJECTIVES = [
    "Plan a picnic in the park for four people on Saturday.",
    "Start growing herbs on a balcony.",
    "Cook dinner for six guests, one of whom eats no meat.",
]
FIRST = ("You plan the tasks toward an objective.\nObjective: {objective}\n\nWrite the first tasks toward this objective, one "
         "per line, at most 3. Write only the tasks, without numbers.")
DO = ("You do one task toward an objective.\nObjective: {objective}\nTask: {task}\nWhat was done so far:\n{done}\n\nDo the "
      "task. Answer in at most three sentences.")
NEXT = ("You plan the tasks toward an objective.\nObjective: {objective}\nThe last task was: {task}\nIts result: {result}\n"
        "Tasks still to do:\n{queue}\n\nWrite new tasks that are still needed to reach the objective, one per line, at most 3. "
        "Write only the tasks, without numbers. If the objective is reached, write DONE.")
WAYS = ("plain", "kept")   # every new task added, as BabyAGI did; or the list of exercise 2
MAX_TOKENS = 150


# --- the exercises' functions, as the course writes them: the runs use them -----------------------------------------------

START_PARSE = '''def parse_tasks(text):
    tasks = []
    for line in text.split("\\n"):
        ...
    return tasks
'''
SOLUTION_PARSE = START_PARSE.replace('''        ...
''', '''        line = line.strip().lstrip("-*0123456789.) ").rstrip(".")
        if line not in ["", "DONE", "Done"]:
            tasks.append(line)
''')

START_ADD = '''def add_tasks(queue, new, done, limit):
    for task in new:
        ...
    return queue
'''
SOLUTION_ADD = START_ADD.replace('''        ...
''', '''        if len(queue) < limit and task not in queue and task not in done:
            queue.append(task)
''')


def _fn(code: str, name: str):
    return run_code(code, {}).names[name]


def parse_tasks(text: str) -> list:
    return _fn(SOLUTION_PARSE, "parse_tasks")(text)


def add_tasks(queue: list, new: list, done: list, limit: int) -> list:
    return _fn(SOLUTION_ADD, "add_tasks")(queue, new, done, limit)


# --- the course's small BabyAGI ------------------------------------------------------------------------------------------------

def _lines(items: list) -> str:
    return "\n".join("- " + t for t in items) or "(nothing yet)"


def ask(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}], "max_tokens": MAX_TOKENS, "temperature": 0}


def state(way: str, texts: list, parse=None, add=None) -> dict:
    """The list after the model's texts so far: the first is its first tasks, then each round its result and its new
    tasks. The tasks done with their results, the list, every task it proposed, and whether it wrote DONE."""
    parse = parse or parse_tasks
    add = add or add_tasks
    queue, done, results, proposed, said_done = [], [], [], [], False
    for k, text in enumerate(texts):
        if k % 2 == 1:   # the result of the first task in the list
            done.append(queue.pop(0))
            results.append(text.strip())
            continue
        new = parse(text)
        said_done = said_done or "DONE" in text
        proposed.append(new)
        queue = queue + new if way == "plain" else add(queue, new, done, LIMIT)
    return {"queue": queue, "done": done, "results": results, "proposed": proposed, "said_done": said_done}


def next_ask(way: str, o: str, texts: list, parse=None, add=None):
    """The request toward objective o after the model's texts so far, or None when the list is empty or the rounds are over."""
    s = state(way, texts, parse, add)
    if len(texts) % 2 == 1:   # after new tasks: do the first one
        if not s["queue"] or len(s["done"]) >= ROUNDS:
            return None
        done = "\n".join(f"- {t}: {r}" for t, r in zip(s["done"], s["results"])) or "(nothing yet)"
        return ask(DO.format(objective=o, task=s["queue"][0], done=done))
    return ask(NEXT.format(objective=o, task=s["done"][-1], result=s["results"][-1], queue=_lines(s["queue"])))


RECORD = {f"{way}-{i}": {**ask(FIRST.format(objective=o)), "then": 1 + 2 * ROUNDS, "seeds": 1}
          for way in WAYS for i, o in enumerate(OBJECTIVES)}


def then(key: str, chain: list):
    """For the recorder: the next request of the run, from the model's texts so far."""
    way, i = key.split("-")
    return next_ask(way, OBJECTIVES[int(i)], [a["text"] for a in chain])


# --- what the runs show ------------------------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-agent-history"]


def texts(way: str, i: int) -> list:
    return [a["text"] for a in course()[f"{way}-{i}"]["answers"][0]["chain"]]


def _norm(task: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", task.lower()).strip()


def rounds(way: str, i: int) -> list:
    """Each round of a run: the task done, its result, the new tasks the model wrote, and the list after them."""
    t = texts(way, i)
    out = []
    for k in range(1, len(t), 2):
        s = state(way, t[:k + 2])
        out.append({"task": s["done"][-1], "result": s["results"][-1], "new": s["proposed"][-1] if k + 1 < len(t) else [],
                    "queue": list(s["queue"]) if k + 1 < len(t) else None})
    return out


def repeats(way: str, i: int) -> int:
    """How many new tasks the model wrote that were, apart from capitals and punctuation, a task already done or already
    in the list."""
    t, n = texts(way, i), 0
    for k in range(2, len(t), 2):
        before = state(way, t[:k])
        seen = {_norm(x) for x in before["queue"] + before["done"]}
        for task in parse_tasks(t[k]):
            n += _norm(task) in seen
            seen.add(_norm(task))
    return n


def done_again(way: str, i: int) -> int:
    """How many tasks the model did that, apart from capitals and punctuation, it had done before in the same run."""
    seen, n = set(), 0
    for task in state(way, texts(way, i))["done"]:
        n += _norm(task) in seen
        seen.add(_norm(task))
    return n


def slipped_in(i: int) -> list:
    """The new tasks that the list of exercise 2 let in, while apart from capitals and punctuation they were a task
    already done or already in the list."""
    t, out = texts("kept", i), []
    for k in range(2, len(t), 2):
        before, after = state("kept", t[:k]), state("kept", t[:k + 1])
        seen = {_norm(x) for x in before["queue"] + before["done"]}
        out += [x for x in after["queue"] if x not in before["queue"] and _norm(x) in seen]
    return out


def findings() -> dict:
    out = {"rounds": ROUNDS, "limit": LIMIT, "objectives": len(OBJECTIVES)}
    out["first"] = [len(parse_tasks(texts("plain", i)[0])) for i in range(len(OBJECTIVES))]
    out["slipped"] = [x for i in range(len(OBJECTIVES)) for x in slipped_in(i)]
    for way in WAYS:
        s = [state(way, texts(way, i)) for i in range(len(OBJECTIVES))]
        out[way + "_done"] = [len(x["done"]) for x in s]
        out[way + "_left"] = [len(x["queue"]) for x in s]
        out[way + "_proposed"] = [sum(len(p) for p in x["proposed"]) for x in s]
        out[way + "_repeats"] = [repeats(way, i) for i in range(len(OBJECTIVES))]
        out[way + "_said_done"] = [i for i in range(len(OBJECTIVES)) if s[i]["said_done"]]
        out[way + "_again"] = [done_again(way, i) for i in range(len(OBJECTIVES))]
    return out


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?]) ", text.strip())[0]


def _and(items: list, lang: str) -> str:
    items = [str(x) for x in items]
    return ", ".join(items[:-1]) + (" and " if lang == "en" else " en ") + items[-1]


def _again(k: int, lang: str) -> str:
    """How many tasks the model did a second time."""
    if lang == "en":
        return "no task a second time" if k == 0 else f"{k} tasks a second time"
    return "geen taak een tweede keer" if k == 0 else f"{k} taken een tweede keer"


def _slipped(k: int, lang: str) -> str:
    """The repeats that differed only in punctuation and still got in, as the start of a sentence."""
    if lang == "en":
        return ("One repeat that differed only in punctuation still got in: a task done before, written again without "
                "its \"!\"" if k == 1 else f"Still, {k} repeats that differed only in capitals or punctuation got in")
    return ("Eén herhaling die alleen in leestekens verschilde, kwam er toch in: een taak die al gedaan was, opnieuw "
            "geschreven zonder zijn \"!\"" if k == 1 else
            f"Toch kwamen er {k} herhalingen in die alleen in hoofdletters of leestekens verschilden")


def measured_text(lang: str) -> str:
    f = findings()
    n = f["objectives"]
    values = dict(r=f["rounds"], pl=_and(f["plain_left"], lang), first=f["first"][0], slipped=_slipped(len(f["slipped"]), lang), pp=sum(f["plain_proposed"]), pr=sum(f["plain_repeats"]),
                  pa=sum(f["plain_again"]), done=sum(f["plain_done"]), kl=_and(f["kept_left"], lang),
                  ka=_again(sum(f["kept_again"]), lang), sd=len(f["plain_said_done"]) + len(f["kept_said_done"]),
                  runs=2 * n, example=_first_sentence(texts("kept", 0)[1]))
    if lang == "en":
        return ("With every new task added, each list ended longer than it began: it started with {first} tasks, and after {r} "
                "rounds it held {pl}. Of the {pp} new "
                "tasks the model wrote, {pr} were a task already done or already in the list, and {pa} of the {done} tasks it "
                "did were ones it had done before in the same run. With the list of exercise 2, {kl} tasks were left, and the "
                "model did {ka}: `add_tasks` skipped every repeat written word for word. {slipped}.\n\nQwen3 wrote DONE in "
                "{sd} of the {runs} runs, yet every run went on until round {r}: the list still held tasks, and nothing in "
                "the loop ends a run on the model's word. And each result is only text the model wrote, such as *{example}* "
                "In this loop, doing a task means writing that it is done.").format(**values)
    return ("Met elke nieuwe taak erbij eindigde elke lijst langer dan hij begon: hij begon met {first} taken, en na {r} "
            "rondes stonden er {pl} in. Van de {pp} nieuwe "
            "taken die het model schreef, waren er {pr} een taak die al gedaan was of al in de lijst stond, en {pa} van de "
            "{done} taken die het deed, had het in dezelfde run al eens gedaan. Met de lijst van oefening 2 bleven er {kl} "
            "taken over, en deed het model {ka}: `add_tasks` sloeg elke herhaling over die woord voor woord hetzelfde "
            "was. {slipped}.\n\nQwen3 schreef DONE in {sd} van de {runs} runs, en toch ging elke run door tot ronde {r}: er stonden "
            "nog taken in de lijst, en niets in de loop stopt een run op het woord van het model. En elk resultaat is alleen "
            "tekst die het model schreef, zoals *{example}* In deze loop betekent een taak doen: schrijven dat hij gedaan "
            "is.").format(**values)


def worked_run() -> tuple:
    """The run of the worked example: every new task added, on the objective whose list grew longest."""
    left = findings()["plain_left"]
    return ("plain", left.index(max(left)))


def worked_text(lang: str) -> str:
    way, i = worked_run()
    s = state(way, texts(way, i))
    most = max(s["queue"], key=s["queue"].count)
    kept = state("kept", texts("kept", i))
    values = dict(o=OBJECTIVES[i], first=len(parse_tasks(texts(way, i)[0])), r=ROUNDS, n=len(s["queue"]), most=most,
                  k=s["queue"].count(most), again=done_again(way, i), kn=len(kept["queue"]))
    if lang == "en":
        return ("### Worked example: a list that keeps growing\n\nThe objective is *{o}* The model starts with {first} "
                "tasks. Each round it does the first task in the list, and then writes the tasks it thinks are still needed. "
                "Most of those are already in the list, but with every new task added, they go in again. After {r} rounds "
                "the list holds {n} tasks, {k} of them *{most}*, and the model did {again} tasks twice. With the "
                "`add_tasks` of exercise 2, the same objective ended with {kn} tasks in the list.").format(**values)
    return ("### Uitgewerkt voorbeeld: een lijst die blijft groeien\n\nHet doel is *{o}* Het model begint met {first} "
            "taken. Elke ronde doet het de eerste taak uit de lijst, en schrijft daarna de taken die het nog nodig vindt. "
            "De meeste staan al in de lijst, maar met elke nieuwe taak erbij gaan ze er opnieuw in. Na {r} rondes staan er {n} "
            "taken in de lijst, {k} keer *{most}*, en deed het model {again} taken twee keer. Met de `add_tasks` van "
            "oefening 2 eindigde hetzelfde doel met {kn} taken in de lijst.").format(**values)


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the pictures ------------------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _label(text: str) -> str:
    return f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.4rem 0 0.15rem">{html.escape(text)}</div>'


def _pre(inner: str) -> str:
    return f'<div style="background:var(--ea-inset);border-radius:4px;padding:0.35rem 0.55rem;margin:0.1rem 0">{inner}</div>'


_HISTORY = {
    "en": {"title": "Before today's agents: what each design did, and what stayed of it",
           "head": ("year", "design", "what it did", "what stayed"),
           "rows": [("2022", "MRKL", "a router sends each question to a module, such as a calculator or a search engine; the model answers the rest",
                     "tools for what a model does badly"),
                    ("2022", "ReAct", "the model writes a thought and an action in text, and reads what came of it",
                     "the loop of call and result"),
                    ("2023", "HuggingGPT", "a chat model plans the steps of a task and gives each to another model from Hugging Face",
                     "models as tools of a model"),
                    ("2023", "AutoGPT, BabyAGI", "given only an objective, the model writes its own tasks, does them and adds new ones, often with no person checking each step",
                     "little: they looped and wandered off; what came of it is the step limit and a person who checks"),
                    ("2023", "function calling", "the APIs take tools with a schema, and give calls as JSON",
                     "what this module uses")]},
    "nl": {"title": "Vóór de agents van nu: wat elk ontwerp deed, en wat ervan bleef",
           "head": ("jaar", "ontwerp", "wat het deed", "wat bleef"),
           "rows": [("2022", "MRKL", "een router stuurt elke vraag naar een module, zoals een rekenmachine of een zoekmachine; het model beantwoordt de rest",
                     "tools voor wat een model slecht kan"),
                    ("2022", "ReAct", "het model schrijft een gedachte en een actie als tekst, en leest wat ervan kwam",
                     "de loop van call en resultaat"),
                    ("2023", "HuggingGPT", "een chat model plant de stappen van een taak en geeft elke stap aan een ander model van Hugging Face",
                     "modellen als tools van een model"),
                    ("2023", "AutoGPT, BabyAGI", "met alleen een doel schrijft het model zijn eigen taken, doet ze en voegt nieuwe toe, vaak zonder dat iemand elke stap controleert",
                     "weinig: ze draaiden rondjes en dwaalden af; wat eruit voortkwam is de stapgrens en een mens die controleert"),
                    ("2023", "function calling", "de API's nemen tools met een schema, en geven calls als JSON",
                     "wat deze module gebruikt")]},
}


def history_picture(lang: str) -> str:
    w = _HISTORY[lang]
    head = "".join(f'<th style="{_HEAD}">{h}</th>' for h in w["head"])
    rows = "".join(f'<tr><td style="{_CELL};{_UI};color:var(--ea-ink-2)">{y}</td><td style="{_CELL};{_UI};font-weight:600">{html.escape(n)}</td>'
                   f'<td style="{_CELL};{_UI}">{html.escape(d)}</td><td style="{_CELL};{_UI};color:var(--ea-ink-2)">{html.escape(s)}</td></tr>'
                   for y, n, d, s in w["rows"])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>{head}</tr></thead>'
             f'<tbody>{rows}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_TABLE_WORDS = {
    "en": {"title": "Qwen3-0.6B as a small BabyAGI: three objectives, {r} rounds each",
           "objective": "objective", "plain": "every new task added", "kept": "the list of exercise 2, at most {l}",
           "done": "tasks done", "proposed": "new tasks written", "repeats": "of them repeats", "left": "left in the list",
           "said": "wrote DONE", "yes": "yes", "no": "no"},
    "nl": {"title": "Qwen3-0.6B als kleine BabyAGI: drie doelen, elk {r} rondes",
           "objective": "doel", "plain": "elke nieuwe taak erbij", "kept": "de lijst van oefening 2, hoogstens {l}",
           "done": "taken gedaan", "proposed": "nieuwe taken geschreven", "repeats": "daarvan herhalingen", "left": "over in de lijst",
           "said": "schreef DONE", "yes": "ja", "no": "nee"},
}


def table_picture(lang: str) -> str:
    w, f = _TABLE_WORDS[lang], findings()
    cols = ("done", "proposed", "repeats", "left", "said")
    head = f'<th style="{_HEAD}">{w["objective"]}</th><th style="{_HEAD}"></th>' + "".join(f'<th style="{_HEAD}">{w[c]}</th>' for c in cols)
    rows = []
    for i, o in enumerate(OBJECTIVES):
        for k, way in enumerate(WAYS):
            values = [f[way + "_done"][i], f[way + "_proposed"][i], f[way + "_repeats"][i], f[way + "_left"][i],
                      w["yes"] if i in f[way + "_said_done"] else w["no"]]
            first = (f'<td style="{_CELL};{_UI}" rowspan="2">{html.escape(o)}</td>' if k == 0 else "")
            label = w[way].format(l=LIMIT)
            rows.append(f'<tr>{first}<td style="{_CELL};{_UI};color:var(--ea-ink-2)">{html.escape(label)}</td>'
                        + "".join(f'<td style="{_CELL};{_UI}">{v}</td>' for v in values) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>{head}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"].format(r=ROUNDS), table, alt=w["title"].format(r=ROUNDS))


_WORKED_WORDS = {
    "en": {"title": "The list, round by round", "round": "round {k}: the model did", "new": "new tasks it wrote",
           "list": "the list after them", "empty": "(empty)", "first": "first tasks"},
    "nl": {"title": "De lijst, ronde na ronde", "round": "ronde {k}: het model deed", "new": "nieuwe taken die het schreef",
           "list": "de lijst daarna", "empty": "(leeg)", "first": "eerste taken"},
}


def worked_picture(way: str, i: int, lang: str) -> str:
    w = _WORKED_WORDS[lang]
    t = texts(way, i)
    parts = [_label(w["first"]) + _pre(_code("\n".join(parse_tasks(t[0])) or w["empty"]))]
    for k, r in enumerate(rounds(way, i), 1):
        body = _code(r["task"], "var(--ea-pred-text)")
        if r["queue"] is not None:
            body += (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(w["new"])}</div>'
                     + _code("\n".join(r["new"]) or w["empty"])
                     + f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin-top:0.2rem">{html.escape(w["list"])} ({len(r["queue"])})</div>'
                     + _code("\n".join(r["queue"]) or w["empty"], "var(--ea-ink-2)"))
        parts.append(_label(w["round"].format(k=k)) + _pre(body))
    return yours.figure(w["title"], "".join(parts), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------------

PARSE_CASES = [
    ("Buy bread and cheese\nPick a spot in the park\nAsk the friends who is coming",),
    ("1. Check the weather for Saturday\n2. Buy drinks\n",),
    ("- Water the herbs\n\n- Put the pots in the sun\n",),
    ("DONE",),
    ("  Buy candles  \n* Set the table\n3) Light the candles",),
    ("- Set up the picnic area.  \n- Arrange the food and drinks.  \nDONE.",),
    ("1. Buy bread.\n- Done.",),
]
_PARSE_PLAIN = SOLUTION_PARSE.replace('.lstrip("-*0123456789.) ")', "")
_PARSE_EMPTY = SOLUTION_PARSE.replace('"", ', "")
_PARSE_DONE = SOLUTION_PARSE.replace(', "DONE", "Done"', "")
_PARSE_DOT = SOLUTION_PARSE.replace('.rstrip(".")', "")

ADD_CASES = [
    (["Pick a spot", "Invite friends"], ["Buy food", "Pack a blanket", "Buy drinks"], ["Buy food"], LIMIT),
    (["Pick a spot"], ["Pick a spot", "Buy drinks"], [], LIMIT),
    (["Buy pots", "Buy soil", "Buy seeds", "Pick a sunny spot"], ["Water the herbs", "Buy a watering can", "Label the pots"], [], LIMIT),
    ([], ["Buy food", "Buy food"], [], LIMIT),
    (["Set the table"], ["Light the candles", "Buy candles"], ["Buy candles"], 2),
]
_ADD_NO_LIMIT = SOLUTION_ADD.replace("len(queue) < limit and ", "")
_ADD_NO_DONE = SOLUTION_ADD.replace(" and task not in done", "")
_ADD_NO_QUEUE = SOLUTION_ADD.replace(" and task not in queue", "")
_ADD_LE = SOLUTION_ADD.replace("len(queue) < limit", "len(queue) <= limit")


def _learner(seen, name: str):
    run_ = seen.run
    fn = run_.names.get(name) if run_ is not None and run_.error is None else None
    return fn if callable(fn) else None


_STOPPED = object()


def _call(fn, *args):
    try:
        return fn(*copy.deepcopy(args))
    except Exception:   # noqa: BLE001 - the learner's code
        return _STOPPED


_PARSE_WORDS = {
    "en": {"title": "Your parse_tasks on what the model wrote", "text": "what the model wrote, line by line",
           "yours": "your tasks", "course": "the course's tasks", "error": "your function stopped with an error"},
    "nl": {"title": "Je parse_tasks op wat het model schreef", "text": "wat het model schreef, regel voor regel",
           "yours": "jouw taken", "course": "de taken van de course", "error": "je functie stopte met een fout"},
}


def parse_picture(seen) -> str:
    fn = _learner(seen, "parse_tasks")
    if fn is None:
        return ""
    w = _PARSE_WORDS[seen.lang]
    text = PARSE_CASES[4][0]
    got, want = _call(fn, text), parse_tasks(text)
    ok = got == want
    shown = w["error"] if got is _STOPPED else "\n".join(repr(t) for t in got) if isinstance(got, list) else repr(got)
    body = (_label(w["text"]) + _pre(_code(text.replace(" ", "·"), "var(--ea-ink-2)"))
            + _label(w["yours"]) + _pre(_code(shown, "var(--ea-ink)" if ok else "var(--ea-loss-text)")))
    if not ok:
        body += _label(w["course"]) + _pre(_code("\n".join(repr(t) for t in want)))
    return yours.figure(w["title"], body, alt=w["title"])


_ADD_WORDS = {
    "en": {"title": "Your add_tasks on the first case: each new task, and what became of it", "before": "the list before",
           "task": "new task", "course": "the course", "yours": "your list", "added": "added", "done": "skipped: already done",
           "queued": "skipped: already in the list", "full": "skipped: the list is full", "after": "the list after your add_tasks",
           "want": "the course's list", "error": "your function stopped with an error", "in_yours": "in your list", "not": "not in it"},
    "nl": {"title": "Je add_tasks op het eerste geval: elke nieuwe taak, en wat ermee gebeurde", "before": "de lijst ervoor",
           "task": "nieuwe taak", "course": "de course", "yours": "jouw lijst", "added": "toegevoegd", "done": "overgeslagen: al gedaan",
           "queued": "overgeslagen: staat al in de lijst", "full": "overgeslagen: de lijst is vol", "after": "de lijst na je add_tasks",
           "want": "de lijst van de course", "error": "je functie stopte met een fout", "in_yours": "in jouw lijst", "not": "niet erin"},
}


def add_picture(seen) -> str:
    fn = _learner(seen, "add_tasks")
    if fn is None:
        return ""
    w = _ADD_WORDS[seen.lang]
    queue, new, done, limit = ADD_CASES[0]
    got, want = _call(fn, queue, new, done, limit), add_tasks(list(queue), new, done, limit)
    rows, sim = [], list(queue)
    for task in new:
        why = "done" if task in done else "queued" if task in sim else "full" if len(sim) >= limit else "added"
        if why == "added":
            sim.append(task)
        mine = isinstance(got, list) and task in got
        rows.append(f'<tr><td style="{_CELL}">{_code(task)}</td><td style="{_CELL};{_UI};color:var(--ea-ink-2)">{w[why]}</td>'
                    f'<td style="{_CELL};{_UI};color:{"var(--ea-ink)" if mine == (why == "added") else "var(--ea-loss-text)"}">'
                    f'{w["in_yours"] if mine else w["not"]}</td></tr>')
    table = (f'<table style="border-collapse:collapse;width:100%"><thead><tr><th style="{_HEAD}">{w["task"]}</th>'
             f'<th style="{_HEAD}">{w["course"]}</th><th style="{_HEAD}">{w["yours"]}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    ok = got == want
    shown = w["error"] if got is _STOPPED else repr(got)
    body = (_label(w["before"]) + _pre(_code(repr(queue), "var(--ea-ink-2)")) + table
            + _label(w["after"]) + _pre(_code(shown, "var(--ea-ink)" if ok else "var(--ea-loss-text)")))
    if not ok:
        body += _label(w["want"]) + _pre(_code(repr(want)))
    return yours.figure(w["title"], body, alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "parse": FunctionBox(
        name="parse",
        picture=parse_picture,
        function="parse_tasks",
        start=START_PARSE,
        solution=SOLUTION_PARSE,
        visible=PARSE_CASES[:1],
        hidden=PARSE_CASES[1:],
        mistakes=[
            Mistake("plain", wrong=_PARSE_PLAIN,
                    en="With `{call}`, your tasks still start with a number or a dash: {got}. Take those off the start of each line too, with `.lstrip(\"-*0123456789.) \")`.",
                    nl="Met `{call}` beginnen je taken nog met een nummer of een streepje: {got}. Haal die ook van het begin van elke regel af, met `.lstrip(\"-*0123456789.) \")`."),
            Mistake("empty", wrong=_PARSE_EMPTY,
                    en="With `{call}`, your list holds an empty task: {got}. An empty line is no task; skip it.",
                    nl="Met `{call}` staat er een lege taak in je lijst: {got}. Een lege regel is geen taak; sla hem over."),
            Mistake("done", wrong=_PARSE_DONE,
                    en="With `{call}`, your list holds `'DONE'` as a task. DONE is the model saying it is finished; skip that line.",
                    nl="Met `{call}` staat `'DONE'` als taak in je lijst. DONE is het model dat zegt dat het klaar is; sla die regel over."),
            Mistake("dot", wrong=_PARSE_DOT,
                    en="With `{call}`, your function gives back {got}: the model wrote its lines with a dot at the end. Take that dot off too, with `.rstrip(\".\")`, so that `DONE.` becomes `DONE`.",
                    nl="Met `{call}` geeft je functie {got} terug: het model schreef zijn regels met een punt aan het eind. Haal die punt ook af, met `.rstrip(\".\")`, zodat `DONE.` `DONE` wordt."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your function gives back an empty list for every text. For each line, take off the spaces and what a list puts in front, and add the line to `tasks` when something is left that is not DONE.",
                      "nl": "Je functie geeft voor elke tekst een lege lijst terug. Haal bij elke regel de spaties af en wat een lijst ervoor zet, en voeg de regel toe aan `tasks` als er iets overblijft dat geen DONE is."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "almost": {"en": "Right for three plain lines. But with `{call}`, your function gives back {got}, and it should be {want}.",
                       "nl": "Goed voor drie gewone regels. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "correct": {"en": "Your `parse_tasks` turns what the model wrote into tasks: one per line, without numbers, dashes or a dot at the end, without empty lines, and without DONE or Done. Right for all {n} cases of the check.",
                        "nl": "Je `parse_tasks` maakt van wat het model schreef taken: één per regel, zonder nummers, streepjes of een punt aan het eind, zonder lege regels, en zonder DONE of Done. Goed bij alle {n} gevallen van de check."},
        },
    ),
    "add": FunctionBox(
        name="add",
        picture=add_picture,
        function="add_tasks",
        start=START_ADD,
        solution=SOLUTION_ADD,
        visible=ADD_CASES[:1],
        hidden=ADD_CASES[1:],
        mistakes=[
            Mistake("no-limit", wrong=_ADD_NO_LIMIT,
                    en="With `{call}`, your list grows past `limit`: {got}. Add a task only while the list holds fewer than `limit`.",
                    nl="Met `{call}` groeit je lijst voorbij `limit`: {got}. Voeg alleen een taak toe zolang de lijst er minder dan `limit` heeft."),
            Mistake("no-done", wrong=_ADD_NO_DONE,
                    en="With `{call}`, your list takes a task that was already done: {got}. Skip a task that is in `done`, or the agent does it again.",
                    nl="Met `{call}` neemt je lijst een taak op die al gedaan is: {got}. Sla een taak over die in `done` staat, anders doet de agent hem nog een keer."),
            Mistake("no-queue", wrong=_ADD_NO_QUEUE,
                    en="With `{call}`, your list holds the same task twice: {got}. Skip a task that is already in `queue`.",
                    nl="Met `{call}` staat dezelfde taak twee keer in je lijst: {got}. Sla een taak over die al in `queue` staat."),
            Mistake("le", wrong=_ADD_LE,
                    en="With `{call}`, your list ends up with one task more than `limit`: {got}. Add only while `len(queue) < limit`; with `<=`, one more gets in.",
                    nl="Met `{call}` eindigt je lijst met één taak meer dan `limit`: {got}. Voeg alleen toe zolang `len(queue) < limit`; met `<=` komt er één bij te veel."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your function gives back the list as it was. Add each new task that is not done yet, not in the list yet, while the list has room.",
                      "nl": "Je functie geeft de lijst terug zoals hij was. Voeg elke nieuwe taak toe die nog niet gedaan is en nog niet in de lijst staat, zolang er plek is."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}. The picture shows each new task.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn. Het plaatje laat elke nieuwe taak zien."},
            "almost": {"en": "Right for the first case. But with `{call}`, your function gives back {got}, and it should be {want}.",
                       "nl": "Goed voor het eerste geval. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "correct": {"en": "Your `add_tasks` keeps the list in bounds: no task twice, nothing that was already done, and never more than `limit`. Right for all {n} cases of the check.",
                        "nl": "Je `add_tasks` houdt de lijst binnen de perken: geen taak twee keer, niets wat al gedaan is, en nooit meer dan `limit`. Goed bij alle {n} gevallen van de check."},
        },
    ),
}

REQUIRED = ["parse", "add", "stayed"]

CASES = [
    ("parse", START_PARSE, "empty", None),
    ("parse", "", "empty", None),
    ("parse", SOLUTION_PARSE, "correct", None),
    ("parse", _PARSE_PLAIN, "wrong", "plain"),
    ("parse", _PARSE_EMPTY, "wrong", "empty"),
    ("parse", _PARSE_DONE, "wrong", "done"),
    ("parse", _PARSE_DOT, "wrong", "dot"),
    ("add", START_ADD, "empty", None),
    ("add", SOLUTION_ADD, "correct", None),
    ("add", _ADD_NO_LIMIT, "wrong", "no-limit"),
    ("add", _ADD_NO_DONE, "wrong", "no-done"),
    ("add", _ADD_NO_QUEUE, "wrong", "no-queue"),
    ("add", _ADD_LE, "wrong", "le"),
]


# --- the demo: your small BabyAGI on Qwen3 ------------------------------------------------------------------------------------

DEMO_ROUNDS = 3


def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts_: dict) -> None:
    chat.download_or_stop(button, texts_)


def learner_fn(key: str, text):
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, {}).names.get(ex.function)


def _post(url: str, request: dict) -> str:
    import urllib.request
    body = {"model": serve.MODEL, **request}
    r = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"] or ""


def baby_agi(objective: str, parse, add, words: dict) -> str:
    """The course's small BabyAGI on an objective of the learner's, with the learner's parse_tasks and add_tasks, for
    DEMO_ROUNDS rounds."""
    parts = []
    try:
        running = serve.model_server()
        made = [_post(running.url, ask(FIRST.format(objective=objective)))]
        while len(made) < 1 + 2 * DEMO_ROUNDS:
            nxt = next_ask("kept", objective, made, parse, add)
            if nxt is None:
                break
            made.append(_post(running.url, nxt))
        s = state("kept", made, parse, add)
    except Exception as e:   # the real server, or the learner's functions
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        return f"<p><em>{html.escape(words['stopped'].format(e=f'{type(e).__name__}: {e}'))}</em></p>"
    parts.append(_label(words["first"]) + _pre(_code("\n".join(s["proposed"][0]) or words["empty"])))
    for k, (task, result) in enumerate(zip(s["done"], s["results"]), 1):
        new = s["proposed"][k] if k < len(s["proposed"]) else []
        parts.append(_label(words["round"].format(k=k)) + _pre(
            _code(task, "var(--ea-pred-text)") + f'<div style="{_UI};font-size:13px;margin:0.15rem 0">{html.escape(result)}</div>'
            + (f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(words["new"])}</div>' + _code("\n".join(new) or words["empty"])
               if k < len(s["proposed"]) else "")))
    parts.append(_label(words["left"].format(n=len(s["queue"]))) + _pre(_code("\n".join(s["queue"]) or words["empty"], "var(--ea-ink-2)")))
    return yours.figure(words["title"], "".join(parts), alt=words["title"])
