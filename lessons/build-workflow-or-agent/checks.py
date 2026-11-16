# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-workflow-or-agent: a kitchen with six recipes and a pantry, and twelve questions of what to buy. The
course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json) answer each question two ways: as an agent with the
tools recipe and pantry, which chooses its own steps, and as a workflow, in which the code takes the steps and the
model only picks the dish and writes the answer. What they show, their pictures, the exercises (the workflow's code
step, and the whole workflow against a stand-in of the server) and the demo with the learner's own workflow."""
import copy
import html
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case, call
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"

# --- the kitchen: six recipes and a pantry ------------------------------------------------------------------------------

RECIPES = {
    "pancakes": {"flour": 250, "milk": 450, "eggs": 110, "butter": 25},
    "tomato soup": {"tomatoes": 800, "onion": 150, "butter": 30, "stock": 500},
    "omelette": {"eggs": 165, "milk": 50, "cheese": 40, "butter": 10},
    "rice pudding": {"rice": 100, "milk": 750, "sugar": 60},
    "apple pie": {"flour": 300, "butter": 200, "sugar": 150, "apples": 800},
    "pasta with pesto": {"pasta": 400, "basil": 50, "cheese": 60, "pine nuts": 30},
}
PANTRY = {"flour": 1000, "milk": 200, "eggs": 0, "butter": 250, "tomatoes": 400, "onion": 150, "stock": 1000,
          "rice": 500, "sugar": 100, "apples": 300, "pasta": 500, "cheese": 50, "basil": 0, "pine nuts": 0}
DISHES = list(RECIPES)


class Unknown(KeyError):
    """A dish the kitchen has no recipe for: the course's own lookup error, so that a learner's KeyError stays theirs."""


def recipe(dish: str) -> dict:
    """The recipe of a dish, {ingredient: grams}; a dish without a recipe raises Unknown."""
    if dish not in RECIPES:
        raise Unknown(dish)
    return dict(RECIPES[dish])


def pantry(item: str) -> int:
    """How many grams of an ingredient are in the pantry: 0 for one that is not there."""
    return PANTRY.get(item, 0)


def to_buy(dish: str) -> dict:
    """What to buy for a dish: each ingredient the pantry has too little of, with the grams that are missing."""
    need = recipe(dish)
    return {item: need[item] - pantry(item) for item in need if pantry(item) < need[item]}


def as_text(buy: dict) -> str:
    """The shopping list as the workflow gives it to the model: "milk 250 g, eggs 110 g", or "nothing"."""
    return ", ".join(f"{item} {grams} g" for item, grams in buy.items()) or "nothing"


# --- the two ways: an agent with two tools, and a workflow ----------------------------------------------------------------

def _tool(name: str, description: str, argument: str, about: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {argument: {"type": "string", "description": about}},
                       "required": [argument]}}}


TOOLS = [_tool("recipe", "The ingredients of a dish, with the grams of each.", "dish", "the name of the dish, such as pancakes"),
         _tool("pantry", "How many grams of an ingredient are in the pantry now.", "item", "the name of the ingredient, such as milk")]
SYSTEM = ("You are a kitchen assistant with two tools. Use the tools to look up the recipe and how much of each ingredient "
          "is in the pantry. Do not ask the user: the tools have the answers. When you know enough, answer with what to buy, "
          "with the grams of each.")
STEPS = 8   # the agent's step limit: requests in one run


def recipe_tool(arguments: dict) -> str:
    """The tool recipe: the recipe as JSON text, or a sentence that there is none."""
    dish = str(arguments.get("dish", "")).lower().strip()
    return json.dumps(RECIPES[dish]) if dish in RECIPES else f"There is no recipe for {arguments.get('dish')}."


def pantry_tool(arguments: dict) -> str:
    """The tool pantry: the grams of an ingredient as text, such as "200 grams"."""
    return f"{pantry(str(arguments.get('item', '')).lower().strip())} grams"


FUNCTIONS = {"recipe": recipe_tool, "pantry": pantry_tool}


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder and the demo: the result of each tool call in an answer, as a message with the role tool and the
    id of its call; arguments that are not JSON, or a tool that does not exist, get an error as text."""
    out = []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            arguments = None
        fn = FUNCTIONS.get(c["function"]["name"])
        result = ("Error: no tool " + c["function"]["name"] if fn is None else
                  "Error: the arguments are not a JSON object" if not isinstance(arguments, dict) else fn(arguments))
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


PICK = ("Which dish does this question ask about? Answer with one of: " + ", ".join(DISHES) + ". Write only the name.\n\n"
        "Question: {question}")
WRITE = "Write one short sentence for the user that says what to buy to make {dish}: {items}."
SORRY = "Sorry, I only know these dishes: " + ", ".join(DISHES) + "."

_NAMED = {"omelette": "an omelette", "apple pie": "an apple pie"}
QUESTIONS = []   # (question, dish)
for _d in DISHES:
    QUESTIONS.append((f"What do I have to buy to make {_NAMED.get(_d, _d)}?", _d))
    QUESTIONS.append((f"I want to make {_NAMED.get(_d, _d)} tonight. What is missing from my pantry?", _d))


def asked(text: str) -> list:
    return [{"role": "user", "content": text}]


RECORD = {}
for _i, (_q, _d) in enumerate(QUESTIONS):
    RECORD[f"agent-{_i}"] = {"messages": [{"role": "system", "content": SYSTEM}] + asked(_q), "tools": TOOLS,
                             "max_tokens": 200, "temperature": 0, "steps": STEPS, "seeds": 1}
    RECORD[f"pick-{_i}"] = {"messages": asked(PICK.format(question=_q)), "max_tokens": 20, "temperature": 0, "seeds": 1}
for _j, _d in enumerate(DISHES):
    RECORD[f"write-{_j}"] = {"messages": asked(WRITE.format(dish=_d, items=as_text(to_buy(_d)))), "max_tokens": 80,
                             "temperature": 0, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-workflow-or-agent"]


def run(key: str) -> dict:
    return course()[key]["answers"][0]


def agent_steps(i: int) -> list:
    return run(f"agent-{i}")["steps"]


def calls_of(steps: list) -> list:
    """The tool calls of an agent's run, in order: (name, arguments as a dict, or the text when it is not JSON)."""
    out = []
    for s in steps:
        for c in s["tool_calls"]:
            try:
                arguments = json.loads(c["function"]["arguments"])
            except ValueError:
                arguments = c["function"]["arguments"]
            out.append((c["function"]["name"], arguments))
    return out


def final_of(steps: list) -> str | None:
    """The agent's answer: the text of its last step, or None when the step limit stopped it while it still called tools."""
    return None if steps[-1]["tool_calls"] else steps[-1]["text"]


_STEM = {"tomatoes": "tomato", "eggs": "egg", "apples": "apple", "pine nuts": "pine nut", "onion": "onion"}
_SPLIT = re.compile(r"\n|;|,\s|\.\s|\band\b")
_KEPT = re.compile(r"\b(?:have|has|enough|already|pantry|no need)\b", re.I)


def _parts(text: str) -> list:
    return [p.strip().lower() for p in _SPLIT.split(text) if p.strip()]


def right_answer(text: str | None, dish: str) -> bool:
    """Right when every ingredient to buy is named with its grams in one part of the answer (parts end at a new line,
    a comma, a full stop or "and"), and no ingredient the pantry has enough of is named there with an amount to buy."""
    if not text:
        return False
    parts, want = _parts(text), to_buy(dish)
    for item, grams in want.items():
        stem = _STEM.get(item, item)
        if not any(stem in p and re.search(rf"(?<!\d){grams}(?!\d)", p) for p in parts):
            return False
    for item in recipe(dish):
        if item in want:
            continue
        stem = _STEM.get(item, item)
        if any(stem in p and re.search(r"\d", p) and not _KEPT.search(p) for p in parts):
            return False
    return True


def echoes_recipe(text: str | None, dish: str) -> bool:
    """True when an answer names every ingredient of the recipe with the grams of the recipe, as if nothing were there."""
    if not text:
        return False
    parts = _parts(text)
    return all(any(_STEM.get(item, item) in p and re.search(rf"(?<!\d){grams}(?!\d)", p) for p in parts)
               for item, grams in recipe(dish).items())


def picked(i: int) -> str:
    """The dish the workflow's code reads from the model's pick: its answer without spaces, in small letters."""
    return run(f"pick-{i}")["text"].strip().lower()


def write_run(i: int) -> dict | None:
    """The workflow's second request for question i: the sentence for the dish the model picked, or None when the code
    stopped there because the kitchen does not have that dish."""
    dish = picked(i)
    return run(f"write-{DISHES.index(dish)}") if dish in DISHES else None


def workflow_right(i: int) -> bool:
    return picked(i) == QUESTIONS[i][1] and right_answer(write_run(i)["text"], QUESTIONS[i][1])


def findings() -> dict:
    n = len(QUESTIONS)
    agents = [agent_steps(i) for i in range(n)]
    agent_right = [i for i in range(n) if right_answer(final_of(agents[i]), QUESTIONS[i][1])]
    looked = [i for i in range(n) if any(name == "pantry" for name, _ in calls_of(agents[i]))]
    every = [i for i in range(n) if {a.get("item") for name, a in calls_of(agents[i]) if name == "pantry" and isinstance(a, dict)}
             >= set(recipe(QUESTIONS[i][1]))]
    unfinished = [i for i in range(n) if final_of(agents[i]) is None]
    wf = [[r for r in (run(f"pick-{i}"), write_run(i)) if r is not None] for i in range(n)]   # the requests it sent
    return {
        "questions": n, "dishes": len(DISHES), "steps": STEPS,
        "agent_right": agent_right, "agent_looked": looked, "agent_every": every, "agent_unfinished": unfinished,
        "agent_echo": [i for i in range(n) if echoes_recipe(final_of(agents[i]), QUESTIONS[i][1])],
        "agent_requests": sum(len(a) for a in agents),
        "agent_prompt": sum(s["prompt_tokens"] for a in agents for s in a),
        "agent_tokens": sum(s["tokens"] for a in agents for s in a),
        "agent_seconds": sum(s["seconds"] for a in agents for s in a),
        "picked_right": [i for i in range(n) if picked(i) == QUESTIONS[i][1]],
        "picked_other": [i for i in range(n) if picked(i) != QUESTIONS[i][1] and picked(i) in DISHES],   # a wrong dish that exists
        "workflow_right": [i for i in range(n) if workflow_right(i)],
        "workflow_requests": sum(len(rs) for rs in wf),
        "workflow_prompt": sum(r["prompt_tokens"] for rs in wf for r in rs),
        "workflow_tokens": sum(r["tokens"] for rs in wf for r in rs),
        "workflow_seconds": sum(r["seconds"] for rs in wf for r in rs),
    }


WORKED = 0   # the question of the worked example: pancakes


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_workflow",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def agent_idea_line(lang: str) -> str:
    """What Qwen3 did in agent-idea's runs on the pancakes, in one or two sentences."""
    A = _lesson_checks("agent-idea")
    k = A._counts(A.course())
    n, looked, right = k["n"], k["pantry"], k["right"]
    if lang == "en":
        first = (f"In all {n} runs of the course it looked up the recipe first" if k["recipe_first"] == n else
                 f"In {k['recipe_first']} of {n} runs of the course it looked up the recipe first")
        pantry_ = "and it never looked in the pantry" if looked == 0 else f"and it looked in the pantry in {looked}"
        answers = "none of its answers was right" if right == 0 else f"{right} of its {n} answers were right"
        return f"{first}, {pantry_}: {answers}."
    first = (f"In alle {n} runs van de course zocht het eerst het recept op" if k["recipe_first"] == n else
             f"In {k['recipe_first']} van de {n} runs van de course zocht het eerst het recept op")
    pantry_ = "en keek het nooit in de voorraadkast" if looked == 0 else f"en keek het in {looked} ervan in de voorraadkast"
    answers = "geen van zijn antwoorden klopte" if right == 0 else f"{right} van zijn {n} antwoorden klopten"
    return f"{first}, {pantry_}: {answers}."


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


def _many(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def measured_text(lang: str) -> str:
    """What the runs show, in the lesson's language, with the measured numbers."""
    f = findings()
    n = f["questions"]
    a_right, w_right = len(f["agent_right"]), len(f["workflow_right"])
    other = [(QUESTIONS[i][1], picked(i)) for i in f["picked_other"]]
    if lang == "nl":
        limit = ("Geen run haalde de stappenlimiet." if not f["agent_unfinished"] else
                 f"De stappenlimiet stopte {_many(len(f['agent_unfinished']), 'run', 'runs')}.")
        wrong = "".join(f" Bij een vraag over `{d}` koos het `{p}`: dat gerecht bestaat, dus de controle liet het door, en de workflow "
                        f"rekende verder met het verkeerde recept." for d, p in other)
        same = (f"Beide manieren stuurden {f['agent_requests']} requests" if f["agent_requests"] == f["workflow_requests"] else
                f"De agent stuurde {f['agent_requests']} requests, de workflow {f['workflow_requests']}")
        return (f"Het antwoord van de agent klopte bij {a_right} van de {n} vragen, dat van de workflow bij {w_right}. "
                f"De agent keek bij {_many(len(f['agent_looked']), 'vraag', 'vragen')} in de voorraadkast, en "
                + ("nooit" if not f["agent_every"] else f"bij {len(f['agent_every'])}")
                + f" voor elk ingrediënt van het recept; in {len(f['agent_echo'])} van zijn {n} antwoorden "
                f"gaf hij de grammen van het hele recept. {limit} In de workflow noemde het model bij {len(f['picked_right'])} "
                f"van de {n} vragen het goede gerecht.{wrong} {same}, maar de prompts van de agent telden "
                f"{number(f['agent_prompt'], 'nl')} tokens tegen {number(f['workflow_prompt'], 'nl')}, omdat elk request van de "
                f"agent de system prompt, de twee tools en alles wat er al terugkwam meestuurt. De agent deed er "
                f"{number(f['agent_seconds'], 'nl', 1)} seconden over, de workflow {number(f['workflow_seconds'], 'nl', 1)}.")
    limit = ("No run reached the step limit." if not f["agent_unfinished"] else
             f"The step limit stopped {_many(len(f['agent_unfinished']), 'run', 'runs')}.")
    wrong = "".join(f" For a question about `{d}`, it picked `{p}`: that dish exists, so the check let it through, and the "
                    f"workflow went on with the wrong recipe." for d, p in other)
    same = (f"Both ways sent {f['agent_requests']} requests" if f["agent_requests"] == f["workflow_requests"] else
            f"The agent sent {f['agent_requests']} requests, the workflow {f['workflow_requests']}")
    return (f"The agent's answer was right for {a_right} of the {n} questions, the workflow's for {w_right}. "
            f"The agent looked in the pantry for {_many(len(f['agent_looked']), 'question', 'questions')}, and "
            + ("never for every ingredient of the recipe" if not f["agent_every"] else
               f"for every ingredient of the recipe for {len(f['agent_every'])}")
            + f"; in {len(f['agent_echo'])} of its {n} answers, it gave the "
            f"grams of the whole recipe. {limit} In the workflow, the model named the right dish for {len(f['picked_right'])} "
            f"of the {n} questions.{wrong} {same}, but the agent's prompts held {number(f['agent_prompt'], 'en')} tokens against "
            f"{number(f['workflow_prompt'], 'en')}, since each of its requests carries the system prompt, the two tools and "
            f"what came back so far. The agent took {number(f['agent_seconds'], 'en', 1)} seconds, the workflow "
            f"{number(f['workflow_seconds'], 'en', 1)}.")


def worked_text(lang: str) -> str:
    """The worked example: the workflow's two requests on one question, from the recorded runs."""
    q, dish = QUESTIONS[WORKED]
    pick, write = run(f"pick-{WORKED}"), write_run(WORKED)
    prompt = pick["prompt_tokens"] + write["prompt_tokens"]
    tokens = pick["tokens"] + write["tokens"]
    seconds = pick["seconds"] + write["seconds"]
    items = as_text(to_buy(dish))
    answer = write["text"].strip()
    if lang == "nl":
        return (f"### Uitgewerkt voorbeeld: de workflow op één vraag\n\n"
                f"De vraag is *{q}*\n\n"
                f"1. **Request 1** stuurt `PICK` met de vraag erin. Qwen3 antwoordt `{pick['text']!r}`.\n"
                f"2. **Je code** maakt daar `{picked(WORKED)!r}` van met `.strip().lower()`, vindt het in `DISHES`, en rekent met "
                f"`to_buy` uit wat er ontbreekt: `{items}`.\n"
                f"3. **Request 2** stuurt `WRITE`, ingevuld met het gerecht en die lijst. Qwen3 schrijft: *{answer}*\n\n"
                f"Twee requests, {number(prompt, 'nl')} tokens aan prompts en {tokens} geschreven tokens, "
                f"{number(seconds, 'nl', 1)} seconden. De grammen komen uit je code; het model zette ze alleen in een zin.")
    return (f"### Worked example: the workflow on one question\n\n"
            f"The question is *{q}*\n\n"
            f"1. **Request 1** sends `PICK` with the question in it. Qwen3 answers `{pick['text']!r}`.\n"
            f"2. **Your code** makes `{picked(WORKED)!r}` of it with `.strip().lower()`, finds it in `DISHES`, and works out what is "
            f"missing with `to_buy`: `{items}`.\n"
            f"3. **Request 2** sends `WRITE`, filled in with the dish and that list. Qwen3 writes: *{answer}*\n\n"
            f"Two requests, {number(prompt, 'en')} prompt tokens and {tokens} tokens written, {number(seconds, 'en', 1)} seconds. "
            f"The grams come from your code; the model only put them in a sentence.")


# --- the pictures ----------------------------------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"
_UI = "font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _box(text: str, role: str) -> str:
    """One step of a flow: the model's steps in the color of what a model makes, the code's in ink."""
    color = {"model": "var(--ea-pred)", "code": "var(--ea-ink-3)", "tool": "var(--ea-data)"}[role]
    return (f'<div style="border:1.5px solid {color};border-radius:6px;padding:0.3rem 0.5rem;{_UI};font-size:12px;'
            f'color:var(--ea-ink);background:var(--ea-page);min-width:0">{text}</div>')


def _arrow() -> str:
    return f'<div style="{_UI};font-size:13px;color:var(--ea-ink-3);text-align:center">↓</div>'


_FLOW_WORDS = {
    "en": {"title": "The same question, two ways", "workflow": "Workflow: your code fixes the steps",
           "agent": "Agent: the model chooses the steps",
           "pick": "<b>model</b>: which dish does the question ask about?", "recipe": "<b>code</b>: <code>recipe(dish)</code>",
           "pantry": "<b>code</b>: <code>pantry(item)</code> for each ingredient", "subtract": "<b>code</b>: what is missing",
           "write": "<b>model</b>: write the answer", "ask": "<b>model</b>: a tool call, or the answer",
           "run": "<b>your program</b>: run the call, send back its result",
           "loop": "again, until the model answers or the step limit stops it",
           "alt": "Left, a workflow: the model picks the dish, the code looks up the recipe and the pantry and works out what is missing, and the model writes the answer. Right, an agent: the model asks for a tool, the program runs it, and that repeats until the model answers."},
    "nl": {"title": "Dezelfde vraag, op twee manieren", "workflow": "Workflow: je code legt de stappen vast",
           "agent": "Agent: het model kiest de stappen",
           "pick": "<b>model</b>: over welk gerecht gaat de vraag?", "recipe": "<b>code</b>: <code>recipe(dish)</code>",
           "pantry": "<b>code</b>: <code>pantry(item)</code> voor elk ingrediënt", "subtract": "<b>code</b>: wat ontbreekt er",
           "write": "<b>model</b>: schrijf het antwoord", "ask": "<b>model</b>: een tool call, of het antwoord",
           "run": "<b>je programma</b>: voer de call uit, stuur het resultaat terug",
           "loop": "opnieuw, tot het model antwoordt of de stappenlimiet het stopt",
           "alt": "Links een workflow: het model kiest het gerecht, de code zoekt het recept en de voorraadkast op en rekent uit wat er ontbreekt, en het model schrijft het antwoord. Rechts een agent: het model vraagt om een tool, het programma voert hem uit, en dat herhaalt zich tot het model antwoordt."},
}


def flow_picture(lang: str) -> str:
    """The workflow beside the agent: the steps of each, and who takes them."""
    w = _FLOW_WORDS[lang]
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin-bottom:0.3rem"
    left = "".join([_box(w["pick"], "model"), _arrow(), _box(w["recipe"], "code"), _arrow(), _box(w["pantry"], "code"),
                    _arrow(), _box(w["subtract"], "code"), _arrow(), _box(w["write"], "model")])
    right = "".join([_box(w["ask"], "model"), _arrow(), _box(w["run"], "tool"),
                     f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);text-align:center;margin-top:0.3rem">↻ {w["loop"]}</div>'])
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:1rem">'
            f'<div style="min-width:0"><div style="{head}">{w["workflow"]}</div>{left}</div>'
            f'<div style="min-width:0"><div style="{head}">{w["agent"]}</div>{right}</div></div>')
    return yours.figure(w["title"], body, alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "Twelve questions, as an agent and as a workflow", "question": "the question", "agent": "agent: its calls",
           "workflow": "workflow: the dish it picked", "right": "right", "wrong": "not right", "stopped": "stopped at the step limit",
           "total": "{k} of {n} right", "none": "no call", "times": "{name} ×{k}"},
    "nl": {"title": "Twaalf vragen, als agent en als workflow", "question": "de vraag", "agent": "agent: zijn calls",
           "workflow": "workflow: het gerecht dat hij koos", "right": "goed", "wrong": "niet goed", "stopped": "gestopt bij de stappenlimiet",
           "total": "{k} van de {n} goed", "none": "geen call", "times": "{name} ×{k}"},
}


def _calls_line(steps: list, w: dict) -> str:
    names = [name for name, _ in calls_of(steps)]
    if not names:
        return w["none"]
    out, k = [], 1
    for j, name in enumerate(names):
        if j + 1 < len(names) and names[j + 1] == name:
            k += 1
            continue
        out.append(w["times"].format(name=name, k=k) if k > 1 else name)
        k = 1
    return ", ".join(out)


def _mark(ok: bool, w: dict, stopped: bool = False) -> str:
    color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
    word = w["right"] if ok else (w["stopped"] if stopped else w["wrong"])
    return f'<span style="{_UI};font-size:12px;color:{color};font-weight:{400 if ok else 600}">{word}</span>'


def table_picture(lang: str) -> str:
    """Each question with the agent's calls and the dish the workflow picked, and whether each answer was right."""
    w, f = _TABLE_WORDS[lang], findings()
    rows = []
    for i, (q, dish) in enumerate(QUESTIONS):
        steps = agent_steps(i)
        agent = f'{_code(_calls_line(steps, w), "var(--ea-pred-text)")}<br>{_mark(i in f["agent_right"], w, final_of(steps) is None)}'
        flow = f'{_code(picked(i), "var(--ea-pred-text)")}<br>{_mark(i in f["workflow_right"], w)}'
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(q)}</td><td style="{_CELL}">{agent}</td><td style="{_CELL}">{flow}</td></tr>')
    n = f["questions"]
    foot = (f'<tr><td style="{_CELL}"></td><td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f["agent_right"]), n=n)}</td>'
            f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f["workflow_right"]), n=n)}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["agent"], w["workflow"]))
             + f'</tr></thead><tbody>{"".join(rows)}{foot}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 1: the workflow's own step, in code ----------------------------------------------------------------------

START_TO_BUY = '''def to_buy(dish):
    need = recipe(dish)
    buy = {}
    for item in need:
        ...
    return buy
'''
SOLUTION_TO_BUY = START_TO_BUY.replace('''        ...
''', '''        have = pantry(item)
        if have < need[item]:
            buy[item] = need[item] - have
''')
KITCHEN = {"recipe": recipe, "pantry": pantry}

_WRONG_ALL = START_TO_BUY.replace("        ...\n", "        buy[item] = need[item]\n")
_WRONG_WHOLE = START_TO_BUY.replace("        ...\n", "        if pantry(item) < need[item]:\n            buy[item] = need[item]\n")
_WRONG_EVERY = START_TO_BUY.replace("        ...\n", "        buy[item] = need[item] - pantry(item)\n")
_WRONG_HAVE = START_TO_BUY.replace("        ...\n", "        if pantry(item) < need[item]:\n            buy[item] = pantry(item)\n")
_WRONG_EQUAL = START_TO_BUY.replace("        ...\n", "        if pantry(item) <= need[item]:\n            buy[item] = need[item] - pantry(item)\n")

_TO_BUY_WORDS = {
    "en": {"title": "Your to_buy for each dish", "dish": "dish", "yours": "your list", "want": "what is missing",
           "same": "same", "differs": "differs"},
    "nl": {"title": "Je to_buy voor elk gerecht", "dish": "gerecht", "yours": "je lijst", "want": "wat er ontbreekt",
           "same": "gelijk", "differs": "anders"},
}


def to_buy_picture(seen) -> str:
    """The learner's to_buy on each dish, beside what is really missing."""
    w = _TO_BUY_WORDS[seen.lang]
    run_ = seen.run
    fn = run_.names.get("to_buy") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    rows = []
    for dish in DISHES:
        try:
            got = fn(dish)
        except Exception as e:   # noqa: BLE001 - the learner's code
            got = e
        want = to_buy(dish)
        same = got == want
        shown = f"{type(got).__name__}: {got}" if isinstance(got, Exception) else repr(got)
        color = "var(--ea-ink)" if same else "var(--ea-loss-text)"
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(dish)}</td><td style="{_CELL}">{_code(shown, color)}</td>'
                    f'<td style="{_CELL}">{_code(repr(want))}</td>'
                    f'<td style="{_CELL};{_UI};font-size:12px;color:{color}">{w["same"] if same else w["differs"]}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["dish"], w["yours"], w["want"], ""))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- exercise 2: the whole workflow, against a stand-in of the server -------------------------------------------------

START_SHOPPING = '''import json
import urllib.request


def ask(url, prompt):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80, "temperature": 0}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]


def shopping(url, question):
    dish = ask(url, PICK.format(question=question)).strip().lower()
    ...
'''
SOLUTION_SHOPPING = START_SHOPPING.replace('''    ...
''', '''    if dish not in DISHES:
        return SORRY
    buy = to_buy(dish)
    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))
''')
WORKFLOW = {"PICK": PICK, "WRITE": WRITE, "DISHES": DISHES, "SORRY": SORRY, "to_buy": to_buy, "as_text": as_text}

Q_PANCAKES = QUESTIONS[0][0]
Q_RICE = QUESTIONS[7][0]
Q_PIZZA = "What do I have to buy to make a pizza?"
A_PANCAKES = "Buy 250 grams of milk and 110 grams of eggs."
A_RICE = "You need to buy 550 grams of milk."


def _sent(log: list) -> list:
    return [e for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _prompt(entry) -> str | None:
    messages = entry["body"].get("messages")
    try:
        return messages[-1]["content"]
    except (TypeError, IndexError, KeyError):
        return None


def _judge_shopping(question: str, dish: str | None, want: str):
    def judge(got, running):
        sent = _sent(running.log)
        if not sent:
            return None   # the box says that no request reached the server
        if _prompt(sent[0]) != PICK.format(question=question):
            return "pick"
        if dish is None:   # a dish the kitchen does not have: no second request, and SORRY
            return "asked-on" if len(sent) > 1 else (None if got == SORRY else "sorry")
        if len(sent) < 2:
            return "not-written"
        if _prompt(sent[1]) != WRITE.format(dish=dish, items=as_text(to_buy(dish))):
            return "write"
        if len(sent) > 2:
            return "requests"
        return None if got == want else "return"
    return judge


def _stand_in(answers: dict):
    return lambda: serve.StandIn(answers)


SHOPPING_CASES = [
    Case(_stand_in({PICK.format(question=Q_PANCAKES): "Pancakes",
                    WRITE.format(dish="pancakes", items=as_text(to_buy("pancakes"))): A_PANCAKES}),
         (URL, Q_PANCAKES), _judge_shopping(Q_PANCAKES, "pancakes", A_PANCAKES)),
    Case(_stand_in({PICK.format(question=Q_PIZZA): "pizza"}), (URL, Q_PIZZA), _judge_shopping(Q_PIZZA, None, SORRY)),
    Case(_stand_in({PICK.format(question=Q_RICE): " Rice pudding\n",
                    WRITE.format(dish="rice pudding", items=as_text(to_buy("rice pudding"))): A_RICE}),
         (URL, Q_RICE), _judge_shopping(Q_RICE, "rice pudding", A_RICE)),
]

_STEPS_WORDS = {
    "en": {"title": "What your workflow did", "request": "Request {n}: your workflow asked", "answer": "The server answered:",
           "code": "Then your code: dish {dish!r}, to buy {buy}", "gave": "Your function gave back: {v}",
           "error": "Your function stopped with {e}.", "none": "No request reached the server.",
           "status": "The server answered with status {s}: {m}", "memory": "The server ran out of memory. Close other programs, and try again."},
    "nl": {"title": "Wat je workflow deed", "request": "Request {n}: je workflow vroeg", "answer": "De server antwoordde:",
           "code": "Daarna je code: gerecht {dish!r}, te kopen {buy}", "gave": "Je functie gaf terug: {v}",
           "error": "Je functie stopte met {e}.", "none": "Er kwam geen request aan bij de server.",
           "status": "De server antwoordde met status {s}: {m}", "memory": "De server had geen geheugen meer. Sluit andere programma's, en probeer het opnieuw."},
}


def _pre(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<pre style="margin:0.2rem 0 0.6rem;padding:0.45rem 0.6rem;background:var(--ea-inset);border-radius:6px;'
            f'font-family:var(--ea-font-code);font-size:12px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere;'
            f'color:{color}">{html.escape(text)}</pre>')


def _label(text: str) -> str:
    return f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(text)}</div>'


def steps_html(log: list, words: dict, got=None, error=None) -> str:
    """The requests a workflow sent with what came back, the code's step between them, and what the function gave."""
    sent = _sent(log)
    parts = [] if sent else [f'<p style="{_UI};font-size:13px">{html.escape(words["none"])}</p>']
    for n, e in enumerate(sent, 1):
        parts.append(_label(words["request"].format(n=n)) + _pre(str(_prompt(e))))
        answer = e.get("answer")
        if e.get("status", 200) != 200 or not isinstance(answer, dict) or "choices" not in answer:
            message = (answer or {}).get("error", {}).get("message", "") if isinstance(answer, dict) else ""
            parts.append(_label(words["status"].format(s=e.get("status"), m=message)))
            continue
        text = answer["choices"][0]["message"].get("content") or ""
        parts.append(_label(words["answer"]) + _pre(text, "var(--ea-pred-text)"))
        if n == 1:
            dish = text.strip().lower()
            if dish in RECIPES:
                parts.append(_label(words["code"].format(dish=dish, buy=as_text(to_buy(dish)))))
    if error is not None:
        parts.append(f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(words["error"].format(e=type(error).__name__ + ": " + str(error)))}</p>')
    elif sent:
        parts.append(f'<p style="{_UI};font-size:13px">{html.escape(words["gave"].format(v=repr(got)))}</p>')
    return "".join(parts)


def shopping_picture(seen) -> str:
    """What the learner's workflow sent to the stand-in on the question about pancakes, and what it gave back."""
    run_ = seen.run
    fn = run_.names.get("shopping") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    got, error, running = call(fn, SHOPPING_CASES[0])
    words = _STEPS_WORDS[seen.lang]
    return yours.figure(words["title"], steps_html(running.log, words, got, error), alt=words["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "to-buy": FunctionBox(
        name="to-buy",
        picture=to_buy_picture,
        function="to_buy",
        start=START_TO_BUY,
        solution=SOLUTION_TO_BUY,
        names=KITCHEN,
        visible=[("pancakes",)],
        hidden=[(d,) for d in DISHES[1:]],
        mistakes=[
            Mistake("all-needed", wrong=_WRONG_ALL,
                    en="With `{call}`, your function gives back the whole recipe: `{got}`. Look in the pantry first with `pantry(item)`, and put an ingredient in `buy` only when the pantry has too little of it.",
                    nl="Met `{call}` geeft je functie het hele recept terug: `{got}`. Kijk eerst in de voorraadkast met `pantry(item)`, en zet een ingrediënt alleen in `buy` als er te weinig van in de voorraadkast is."),
            Mistake("whole", wrong=_WRONG_WHOLE,
                    en="With `{call}`, your function gives back `{got}`: the right ingredients, but each with the grams of the recipe. Buy only what is missing: the grams of the recipe minus what the pantry has.",
                    nl="Met `{call}` geeft je functie `{got}` terug: de goede ingrediënten, maar elk met de grammen van het recept. Koop alleen wat ontbreekt: de grammen van het recept min wat de voorraadkast heeft."),
            Mistake("every", wrong=_WRONG_EVERY,
                    en="With `{call}`, your function gives back `{got}`: every ingredient, also the ones the pantry has enough of, with 0 or less. Put an ingredient in `buy` only when the pantry has fewer grams than the recipe needs.",
                    nl="Met `{call}` geeft je functie `{got}` terug: elk ingrediënt, ook die waarvan de voorraadkast genoeg heeft, met 0 of minder. Zet een ingrediënt alleen in `buy` als de voorraadkast minder grammen heeft dan het recept nodig heeft."),
            Mistake("have", wrong=_WRONG_HAVE,
                    en="With `{call}`, your function gives back `{got}`: the grams the pantry has. What is missing is what the recipe needs minus what the pantry has.",
                    nl="Met `{call}` geeft je functie `{got}` terug: de grammen die de voorraadkast heeft. Wat ontbreekt, is wat het recept nodig heeft min wat de voorraadkast heeft."),
            Mistake("equal", wrong=_WRONG_EQUAL,
                    en="With `{call}`, your function gives back `{got}`, with an ingredient of 0 grams: the pantry has exactly as much as the recipe needs. Use `<` instead of `<=`, so that such an ingredient stays off the list.",
                    nl="Met `{call}` geeft je functie `{got}` terug, met een ingrediënt van 0 gram: de voorraadkast heeft precies zoveel als het recept nodig heeft. Gebruik `<` in plaats van `<=`, dan blijft zo'n ingrediënt van de lijst af."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop does nothing yet, so `buy` stays empty. For each `item` of the recipe, look up what the pantry has with `pantry(item)`, and when that is less than `need[item]`, put the missing grams in `buy`.",
                      "nl": "De loop doet nog niets, dus `buy` blijft leeg. Zoek voor elk `item` van het recept op wat de voorraadkast heeft met `pantry(item)`, en zet de ontbrekende grammen in `buy` als dat minder is dan `need[item]`."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should give back `{want}`. Compare what the pantry has with what the recipe needs, and keep only the difference of what is short.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Vergelijk wat de voorraadkast heeft met wat het recept nodig heeft, en houd alleen het verschil over van wat er tekortkomt."},
            "almost": {"en": "Right for pancakes, but with `{call}` your function gives back `{got}`, and it should give back `{want}`. Does your code work for every ingredient of every recipe?",
                       "nl": "Goed voor `pancakes`, maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Werkt je code voor elk ingrediënt van elk recept?"},
            "correct": {"en": "Your `to_buy` takes the same steps for every dish: the recipe, the pantry for each ingredient, and what is missing. Right for all {n} dishes, and the model was not needed for any of it.",
                        "nl": "Je `to_buy` zet voor elk gerecht dezelfde stappen: het recept, de voorraadkast voor elk ingrediënt, en wat er ontbreekt. Goed voor alle {n} gerechten, en het model was er nergens voor nodig."},
        },
    ),
    "shopping": ApiBox(
        name="shopping",
        picture=shopping_picture,
        function="shopping",
        start=START_SHOPPING,
        solution=SOLUTION_SHOPPING,
        names=WORKFLOW,
        cases=SHOPPING_CASES,
        mistakes=[
            Mistake("no-gate", error=lambda e: isinstance(e, Unknown),
                    en="With `{call}`, the model picked a dish the kitchen has no recipe for, and your code went on with it: `recipe` could not find it. Check first that `dish` is in `DISHES`, and give back `SORRY` when it is not.",
                    nl="Met `{call}` koos het model een gerecht waar de keuken geen recept voor heeft, en je code ging ermee verder: `recipe` kon het niet vinden. Controleer eerst of `dish` in `DISHES` staat, en geef `SORRY` terug als dat niet zo is."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The workflow stops after the first step, so it gives back nothing. Check the dish, work out what to buy with `to_buy`, and ask the model to write the answer with `WRITE`.",
                      "nl": "De workflow stopt na de eerste stap, dus hij geeft niets terug. Controleer het gerecht, reken uit wat er gekocht moet worden met `to_buy`, en vraag het model het antwoord te schrijven met `WRITE`."},
            "pick": {"en": "With `{call}`, the first request does not ask which dish the question is about. Keep the first line of the box as it is: it asks with `PICK`.",
                     "nl": "Met `{call}` vraagt het eerste request niet over welk gerecht de vraag gaat. Laat de eerste regel van het vak staan zoals hij is: die vraagt het met `PICK`."},
            "asked-on": {"en": "With `{call}`, the model's dish is not in `DISHES`, and your workflow still asked the model again. Stop there: give back `SORRY` at once.",
                         "nl": "Met `{call}` staat het gerecht van het model niet in `DISHES`, en je workflow vroeg het model toch opnieuw. Stop daar: geef meteen `SORRY` terug."},
            "sorry": {"en": "With `{call}`, the model's dish is not in `DISHES`, and your function gives back `{got}`. Give back `SORRY`, so that the user hears which dishes the kitchen knows.",
                      "nl": "Met `{call}` staat het gerecht van het model niet in `DISHES`, en je functie geeft `{got}` terug. Geef `SORRY` terug, zodat de gebruiker hoort welke gerechten de keuken kent."},
            "not-written": {"en": "With `{call}`, your workflow did not ask the model a second time. After `to_buy`, ask it to write the answer: `ask(url, WRITE.format(dish=dish, items=as_text(buy)))`.",
                            "nl": "Met `{call}` vroeg je workflow het model geen tweede keer. Vraag het na `to_buy` het antwoord te schrijven: `ask(url, WRITE.format(dish=dish, items=as_text(buy)))`."},
            "write": {"en": "With `{call}`, the second request is not the prompt the workflow should send. Fill in `WRITE` with the dish and with the list from `as_text(to_buy(dish))`.",
                      "nl": "Met `{call}` is het tweede request niet de prompt die de workflow zou moeten sturen. Vul `WRITE` in met het gerecht en met de lijst uit `as_text(to_buy(dish))`."},
            "requests": {"en": "With `{call}`, your workflow sent more than two requests. A workflow has a fixed number of steps: one request to pick the dish, and one to write the answer.",
                         "nl": "Met `{call}` stuurde je workflow meer dan twee requests. Een workflow heeft een vast aantal stappen: één request om het gerecht te kiezen, en één om het antwoord te schrijven."},
            "return": {"en": "With `{call}`, your function gives back `{got}`. Give back the model's answer to the second request: the sentence for the user.",
                       "nl": "Met `{call}` geeft je functie `{got}` terug. Geef het antwoord van het model op het tweede request terug: de zin voor de gebruiker."},
            "correct": {"en": "Your workflow takes its steps in a fixed order: the model picks the dish, your code checks it and works out what to buy, and the model writes the sentence. Right for all {n} questions of the check, a dish the kitchen does not know included.",
                        "nl": "Je workflow zet zijn stappen in een vaste volgorde: het model kiest het gerecht, je code controleert het en rekent uit wat er gekocht moet worden, en het model schrijft de zin. Goed bij alle {n} vragen van de check, ook bij een gerecht dat de keuken niet kent."},
        },
    ),
}

REQUIRED = ["to-buy", "shopping", "agent-task"]

_GATE = '''    if dish not in DISHES:
        return SORRY
'''
CASES = [
    ("to-buy", START_TO_BUY, "empty", None),
    ("to-buy", "", "empty", None),
    ("to-buy", SOLUTION_TO_BUY, "correct", None),
    ("to-buy", START_TO_BUY.replace("        ...\n", "        if pantry(item) < need[item]:\n            buy[item] = need[item] - pantry(item)\n"), "correct", None),
    ("to-buy", _WRONG_ALL, "wrong", "all-needed"),
    ("to-buy", _WRONG_WHOLE, "wrong", "whole"),
    ("to-buy", _WRONG_EVERY, "wrong", "every"),
    ("to-buy", _WRONG_HAVE, "wrong", "have"),
    ("to-buy", _WRONG_EQUAL, "wrong", "equal"),
    ("to-buy", START_TO_BUY.replace("        ...\n", "        if pantry(item) < need[item]:\n            buy[item] = need[item] - pantry(item) + 1\n"), "wrong", None),
    ("to-buy", START_TO_BUY.replace("        ...\n", "        if dish == \"pancakes\":\n            buy = {\"milk\": 250, \"eggs\": 110}\n"), "almost", None),
    ("shopping", START_SHOPPING, "empty", None),
    ("shopping", "", "empty", None),
    ("shopping", SOLUTION_SHOPPING, "correct", None),
    ("shopping", SOLUTION_SHOPPING.replace(_GATE, ""), "error", "no-gate"),
    ("shopping", SOLUTION_SHOPPING.replace('''    dish = ask(url, PICK.format(question=question)).strip().lower()''',
                                           '''    dish = ask(url, question).strip().lower()'''), "wrong", "pick"),
    ("shopping", SOLUTION_SHOPPING.replace(_GATE, '''    if dish not in DISHES:
        return ask(url, SORRY)
'''), "wrong", "asked-on"),
    ("shopping", SOLUTION_SHOPPING.replace("        return SORRY\n", "        return None\n"), "wrong", "sorry"),
    ("shopping", SOLUTION_SHOPPING.replace("    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))\n", "    return as_text(buy)\n"), "wrong", "not-written"),
    ("shopping", SOLUTION_SHOPPING.replace("items=as_text(buy)", "items=buy"), "wrong", "write"),
    ("shopping", SOLUTION_SHOPPING.replace("    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))\n",
                                           "    text = ask(url, WRITE.format(dish=dish, items=as_text(buy)))\n    return ask(url, text)\n"), "wrong", "requests"),
    ("shopping", SOLUTION_SHOPPING.replace("    return ask(url, WRITE.format(dish=dish, items=as_text(buy)))\n",
                                           "    ask(url, WRITE.format(dish=dish, items=as_text(buy)))\n    return buy\n"), "wrong", "return"),
]


# --- the demo: the learner's workflow with Qwen3 ------------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_shopping(text):
    """The learner's workflow from exercise 2, when it is right; else None."""
    ex = EXERCISES["shopping"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, dict(WORKFLOW)).names.get("shopping")


def demo(shopping_fn, question: str, words: dict, lang: str) -> str:
    """The learner's workflow on a question, against the course's server with Qwen3 behind it: each request, what came
    back, the code's step between them, and what the workflow gave back."""
    running, before = None, 0
    try:
        running = serve.model_server()
        before = len(running.log)
        got, error = shopping_fn(running.url, question), None
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        got, error = None, e
    log = running.log[before:] if running is not None else []
    return yours.figure(words["title"], steps_html(copy.deepcopy(log), words, got, error), alt=words["title"])
