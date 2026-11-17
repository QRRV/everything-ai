# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-plan-execute: an agent that writes its whole plan first, as JSON, after which the code runs the steps
and asks the model once more for the answer (plan-and-execute, in the form of ReWOO: a step can use the result of an
earlier step by its number). The kitchen of build-workflow-or-agent, with three tools for the plan: recipe, pantry and
missing. The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): the twelve questions as a plan, and as
an agent loop with the same three tools. What they show, their pictures, the exercises (run a plan, and find the steps
that can run at the same time, as LLMCompiler does) and the demo on Qwen3."""
import copy
import html
import importlib.util
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_plan_execute",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _lesson_checks("build-workflow-or-agent")   # the kitchen, the twelve questions and how an answer is judged
QUESTIONS, STEPS, right_answer, Unknown = W.QUESTIONS, W.STEPS, W.right_answer, W.Unknown

# --- the three tools of the plan ------------------------------------------------------------------------------------------


def recipe(arguments: dict) -> dict:
    """The ingredients of a dish, with the grams of each."""
    return W.recipe(str(arguments["dish"]).lower().strip())


def pantry(arguments: dict) -> dict:
    """How many grams of each ingredient are in the pantry: of a list, or of the ingredients of a recipe; a text counts
    as one ingredient."""
    items = arguments["items"]
    return {item: W.pantry(item) for item in ([items] if isinstance(items, str) else items)}


def missing(arguments: dict) -> dict:
    """Each ingredient of need that have has too little of, with the grams that are missing."""
    need, have = arguments["need"], arguments["have"]
    return {item: need[item] - have.get(item, 0) for item in need if have.get(item, 0) < need[item]}


FUNCTIONS = {"recipe": recipe, "pantry": pantry, "missing": missing}

PLAN = ("Make a plan to answer the question with these tools:\n"
        "- recipe(dish): the ingredients of a dish, with the grams of each\n"
        "- pantry(items): how many grams of each of these ingredients are in the pantry\n"
        "- missing(need, have): each ingredient of need that have has too little of, with the grams that are missing\n"
        "Write the plan as JSON: {{\"steps\": [...]}}, each step {{\"tool\": ..., \"args\": {{...}}}}. To use the result of an "
        "earlier step, write # and its number: \"#1\" is the result of step 1.\n"
        "Example, for \"What do I need to buy for waffles?\":\n"
        "{{\"steps\": [{{\"tool\": \"recipe\", \"args\": {{\"dish\": \"waffles\"}}}}, "
        "{{\"tool\": \"pantry\", \"args\": {{\"items\": \"#1\"}}}}, "
        "{{\"tool\": \"missing\", \"args\": {{\"need\": \"#1\", \"have\": \"#2\"}}}}]}}\n"
        "Write only the JSON.\n\nQuestion: {question}")
SOLVE = ("Answer the question in one short sentence for the user, with the grams. Use only these results.\n\n"
         "Question: {question}\n\nResults:\n{results}")

TWO = [("What do I have to buy to make pancakes and pasta with pesto?", ("pancakes", "pasta with pesto")),
       ("What do I have to buy to make tomato soup and rice pudding?", ("tomato soup", "rice pudding")),
       ("What do I have to buy to make an omelette and an apple pie?", ("omelette", "apple pie"))]
ASKED = [(q, (d,)) for q, d in QUESTIONS] + TWO   # each question with its dishes: the twelve, then three with two dishes


def _tool(name: str, description: str, properties: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(properties)}}}


LOOP_TOOLS = [   # the same three tools for the agent loop, which has to copy each result into its next call itself
    _tool("recipe", "The ingredients of a dish, with the grams of each.",
          {"dish": {"type": "string", "description": "the name of the dish, such as pancakes"}}),
    _tool("pantry", "How many grams of each of these ingredients are in the pantry.",
          {"items": {"type": "array", "items": {"type": "string"}, "description": "the names of the ingredients to look up"}}),
    _tool("missing", "Each ingredient of need that have has too little of, with the grams that are missing.",
          {"need": {"type": "object", "description": "the grams of each ingredient that are needed, as the tool recipe gives them"},
           "have": {"type": "object", "description": "the grams of each ingredient that are there, as the tool pantry gives them"}}),
]
LOOP_SYSTEM = ("You are a kitchen assistant with three tools. Use the tools to look up the recipe, how much of each "
               "ingredient is in the pantry, and what is missing. Do not ask the user: the tools have the answers. When you "
               "know enough, answer with what to buy, with the grams of each.")


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: the result of each tool call of the agent loop as JSON text, as a message with the role tool
    and the id of its call; arguments that are not JSON, or a call that fails, get an error as text."""
    out = []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
            result = json.dumps(FUNCTIONS[c["function"]["name"]](arguments))
        except Exception as e:   # noqa: BLE001 - a call from a model: a tool that does not exist, a wrong argument
            result = f"Error: {type(e).__name__}: {e}"
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


def asked(text: str) -> list:
    return [{"role": "user", "content": text}]


def read_plan(text: str):
    """The plan in a model's text, as a dictionary with a list of steps, or None when it is not one."""
    try:
        plan = json.loads(text)
    except ValueError:
        return None
    return plan if isinstance(plan, dict) and isinstance(plan.get("steps"), list) else None


def solve_ask(question: str, plan: dict) -> dict:
    """The second request: the question and each step's result, for the model to write the answer from."""
    return {"messages": asked(SOLVE.format(question=question, results="\n".join(execute(plan)[1]))),
            "max_tokens": 100, "temperature": 0}


def then(key: str, chain: list):
    """For the recorder: after the plan, the request for the answer, built from what the plan's steps gave; None when
    the model wrote no plan the code can read."""
    i = int(key.split("-")[1])
    plan = read_plan(chain[0]["text"])
    return solve_ask(ASKED[i][0], plan) if len(chain) == 1 and plan is not None else None


def _shown(value) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def execute(plan) -> tuple:
    """The course's executor, for any plan a model writes: each step's result by its number ("#1", ...), and its lines
    for the solver. A step that cannot run gets an error as its result, and the steps after it still run."""
    results, lines = {}, []
    steps = plan.get("steps") if isinstance(plan, dict) else None
    for step in steps if isinstance(steps, list) else []:
        number = "#" + str(len(results) + 1)
        try:
            args = {name: results[value] if isinstance(value, str) and value in results else value
                    for name, value in step["args"].items()}
            result = FUNCTIONS[step["tool"]](args)
        except Exception as e:   # noqa: BLE001 - a plan from a model: a tool that does not exist, a wrong argument
            result = f"Error: {type(e).__name__}: {e}"
        results[number] = result
        tool = step.get("tool") if isinstance(step, dict) else None
        given = step.get("args") if isinstance(step, dict) else None
        lines.append(f"{number} {tool}({json.dumps(given)}) = {_shown(result)}")
    return results, lines


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-plan-execute"]


def chain_of(i: int) -> list:
    """The requests of question i as a plan: the plan, and the answer when the code could read the plan."""
    return course()[f"plan-{i}"]["answers"][0]["chain"]


def plan_of(i: int):
    return read_plan(chain_of(i)[0]["text"])


def answer_of(i: int) -> str | None:
    c = chain_of(i)
    return c[1]["text"] if len(c) > 1 else None


def given_of(i: int) -> dict:
    """The answer of question i when the code ran the course's own plan and asked the model only for the answer."""
    return course()[f"given-{i}"]["answers"][0]


def loop_steps(i: int) -> list:
    return course()[f"loop-{i}"]["answers"][0]["steps"]


def want_of(dishes) -> dict:
    """What to buy for all the dishes of a question."""
    out = {}
    for d in dishes:
        out.update(W.to_buy(d))
    return out


def computed(plan) -> dict:
    """What a plan worked out to buy: the results of its steps with the tool missing, together."""
    out = {}
    if plan is None:
        return out
    results = execute(plan)[0]
    for k, step in enumerate(plan["steps"], 1):
        if isinstance(step, dict) and step.get("tool") == "missing" and isinstance(results[f"#{k}"], dict):
            out.update(results[f"#{k}"])
    return out


def right_for(text: str | None, dishes) -> bool:
    return bool(text) and all(right_answer(text, d) for d in dishes)


def failed_steps(plan) -> int:
    """How many steps of a plan gave an error when the code ran them."""
    return 0 if plan is None else sum(isinstance(r, str) and r.startswith("Error:") for r in execute(plan)[0].values())


def findings() -> dict:
    n = len(ASKED)
    plans = [plan_of(i) for i in range(n)]
    chains = [chain_of(i) for i in range(n)]
    loops = [loop_steps(i) for i in range(n)]
    givens = [given_of(i) for i in range(n)]
    two = range(len(QUESTIONS), n)
    return {
        "questions": n, "single": len(QUESTIONS), "double": len(TWO), "steps": STEPS,
        "read": [i for i in range(n) if plans[i] is not None],
        "failed": [i for i in range(n) if failed_steps(plans[i])],
        "computed": [i for i in range(n) if plans[i] is not None and computed(plans[i]) == want_of(ASKED[i][1])],
        "plan_right": [i for i in range(n) if right_for(answer_of(i), ASKED[i][1])],
        "plan_requests": sum(len(c) for c in chains),
        "plan_prompt": sum(r["prompt_tokens"] for c in chains for r in c),
        "plan_tokens": sum(r["tokens"] for c in chains for r in c),
        "plan_seconds": sum(r["seconds"] for c in chains for r in c),
        "given_right": [i for i in range(n) if right_for(givens[i]["text"], ASKED[i][1])],
        "given_prompt": sum(g["prompt_tokens"] for g in givens),
        "given_tokens": sum(g["tokens"] for g in givens),
        "given_seconds": sum(g["seconds"] for g in givens),
        "loop_right": [i for i in range(n) if right_for(W.final_of(loops[i]), ASKED[i][1])],
        "loop_unfinished": [i for i in range(n) if W.final_of(loops[i]) is None],
        "loop_requests": sum(len(s) for s in loops),
        "loop_prompt": sum(r["prompt_tokens"] for s in loops for r in s),
        "loop_tokens": sum(r["tokens"] for s in loops for r in s),
        "loop_seconds": sum(r["seconds"] for s in loops for r in s),
        "loop_calls": sum(len(W.calls_of(s)) for s in loops),
        "one_recipe": [i for i in range(n) if plans[i] is not None and tools_of(plans[i]) == ["recipe"]],
        "one_missing": [i for i in range(n) if plans[i] is not None and tools_of(plans[i]) == ["missing"]
                        and not uses_results(plans[i])],
        "refs": [i for i in range(n) if plans[i] is not None and uses_results(plans[i])],
        "given_no_number": [i for i in range(n) if not re.search(r"\d", givens[i]["text"])],
        "loop_one_pantry": [i for i in range(n) if [name for name, _ in W.calls_of(loops[i])] == ["pantry"]],
        "loop_recipe_items": [i for i in range(n) if any(
            name == "pantry" and set(args.get("items") or []) == {x for d in ASKED[i][1] for x in W.RECIPES[d]}
            for name, args in W.calls_of(loops[i]))],
    }


def tools_of(plan) -> list:
    return [st.get("tool") if isinstance(st, dict) else None for st in plan["steps"]]


def uses_results(plan) -> bool:
    """Whether a step of the plan uses the result of an earlier step: a value such as "#1"."""
    return any(isinstance(v, str) and re.fullmatch(r"#\d+", v) for st in plan["steps"] if isinstance(st, dict)
               and isinstance(st.get("args"), dict) for v in st["args"].values())


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


QUOTED = 0          # the answer from the course's plan that gave the recipe instead of what to buy: pancakes
NO_NUMBER = 7       # an answer from the course's plan without a number: rice pudding
WORKED = 12         # the question of the worked example: pancakes and pasta with pesto, the demo's question


def measured_text(lang: str) -> str:
    """What the runs show, in the lesson's language, with the measured numbers."""
    f = findings()
    n, half = f["questions"], f["single"] // 2
    wf = W.findings()
    w_right, w_n = len(wf["workflow_right"]), wf["questions"]
    quoted, plain = given_of(QUOTED)["text"], given_of(NO_NUMBER)["text"]
    if lang == "nl":
        return (f"Qwen3 schreef bij alle {len(f['read'])} vragen JSON die de code kon lezen, maar geen enkel plan rekende uit "
                f"wat er gekocht moest worden. Bij de {len(f['one_recipe'])} vragen *What do I have to buy …* over één gerecht "
                f"was het plan één stap: het recept. Bij {len(f['one_missing'])} van de {half} vragen *What is missing …* was "
                f"het één stap `missing`, met de naam van het gerecht als `need` in plaats van een resultaat, en die stap stopte "
                f"met een fout. Alleen {len(f['refs'])} plannen gebruikten het nummer van een resultaat, zoals `\"#1\"`. Zo klopte "
                f"het antwoord bij {len(f['plan_right'])} van de {n} vragen.\n\n"
                f"Met het plan van de course draaide elke stap, en in de resultaten stond precies wat er gekocht moest worden. "
                f"Toch schreef Qwen3 daaruit maar bij {len(f['given_right'])} van de {n} het goede antwoord. Bij "
                f"{len(f['given_no_number'])} vragen stond er geen enkel getal in zijn antwoord, zoals *{plain}* Bij pancakes "
                f"gaf het het recept: *{quoted}* Als agent loop riep het bij elke vraag één tool aan, steeds `pantry` met "
                f"ingrediënten die het gokte, nooit `recipe` of `missing`, en klopte het bij {len(f['loop_right'])}. Alles samen: "
                f"als eigen plan {f['plan_requests']} requests, {number(f['plan_prompt'], 'nl')} tokens aan prompts en "
                f"{number(f['plan_seconds'], 'nl', 1)} seconden; met het plan van de course {n} requests, "
                f"{number(f['given_prompt'], 'nl')} tokens en {number(f['given_seconds'], 'nl', 1)} seconden; als agent loop "
                f"{f['loop_requests']} requests, {number(f['loop_prompt'], 'nl')} tokens en {number(f['loop_seconds'], 'nl', 1)} "
                f"seconden.\n\n"
                f"Elke fout in de runs kwam uit wat het model schreef: de code van de course voerde elke stap uit zoals het plan zei, en elk "
                f"plan stond er om te lezen vóór er iets draaide. In [Een workflow of een agent?](/course/open/build-workflow-or-agent?lang=nl), "
                f"waar de code ook uitrekende wat er gekocht moest worden en het model alleen de zin schreef, klopten {w_right} "
                f"van de {w_n} antwoorden. Qwen3-0.6B kon voor deze taak het plan niet schrijven, en ook het antwoord uit de "
                f"resultaten ging meestal mis. De course mat alleen dit ene, kleine model.")
    return (f"Qwen3 wrote JSON the code could read for all {len(f['read'])} questions, but no plan worked out what to buy. For "
            f"the {len(f['one_recipe'])} questions *What do I have to buy …* about one dish, the plan was one step: the recipe. "
            f"For {len(f['one_missing'])} of the {half} questions *What is missing …*, it was one step `missing`, with the name "
            f"of the dish as `need` instead of a result, and that step stopped with an error. Only {len(f['refs'])} plans used "
            f"the number of a result, such as `\"#1\"`. So the answer was right for {len(f['plan_right'])} of the {n} questions.\n\n"
            f"With the course's plan, every step ran, and the results held exactly what to buy. Even so, Qwen3 wrote the right "
            f"answer from them for only {len(f['given_right'])} of the {n}. For {len(f['given_no_number'])} questions, its "
            f"answer had no number at all, such as *{plain}* For pancakes, it gave the recipe: *{quoted}* As an agent loop, it "
            f"called one tool for each question, each time `pantry` with ingredients it guessed, never `recipe` or `missing`, and "
            f"was right for {len(f['loop_right'])}. In all: as its own plan {f['plan_requests']} requests, "
            f"{number(f['plan_prompt'], 'en')} prompt tokens and {number(f['plan_seconds'], 'en', 1)} seconds; with the course's "
            f"plan {n} requests, {number(f['given_prompt'], 'en')} tokens and {number(f['given_seconds'], 'en', 1)} seconds; as "
            f"an agent loop {f['loop_requests']} requests, {number(f['loop_prompt'], 'en')} tokens and "
            f"{number(f['loop_seconds'], 'en', 1)} seconds.\n\n"
            f"Every error in the runs came from what the model wrote: the course's code carried out each step as the plan said, and "
            f"each plan was there to read before anything ran. In [A workflow or an agent?](/course/open/build-workflow-or-agent?lang=en), "
            f"where code also worked out what to buy and the model only wrote the sentence, {w_right} of the {w_n} answers were "
            f"right. Qwen3-0.6B could not write the plan for this task, and the answer from the results mostly went wrong too. "
            f"The course measured only this one small model.")


def worked_text(lang: str) -> str:
    """The worked example: the plan Qwen3 wrote for two dishes, read before it runs, and what running it gave."""
    plan, q = plan_of(WORKED), ASKED[WORKED][0]
    results = execute(plan)[0]
    k = len(plan["steps"])
    wrong_dish = plan["steps"][1]["args"]["dish"]
    error2 = results["#2"].removeprefix("Error: ")
    answer, given = answer_of(WORKED), given_of(WORKED)["text"]
    course = course_plan(ASKED[WORKED][1])
    waves = max(course_waves(course).values())
    if lang == "nl":
        return (f"### Uitgewerkt voorbeeld\n\nVraag {WORKED + 1}, *{q}* Het plan van Qwen3 heeft {k} stappen: het recept van "
                f"`pancakes`, het recept van `{wrong_dish}`, de voorraadkast voor `\"#1\"`, en `missing` met `\"#1\"` en "
                f"`\"#2\"`. Wie het leest vóór het draait, ziet drie fouten. De keuken heeft geen gerecht `{wrong_dish}`, alleen "
                f"`pasta with pesto`. Stap 4 vergelijkt het recept van pancakes met stap 2, het andere recept, in plaats van met "
                f"stap 3, de voorraadkast. En voor het tweede gerecht is er geen stap die uitrekent wat er ontbreekt. De code "
                f"voerde het uit zoals het er stond: stap 2 stopte met *{error2}*, en stap 4, die die fouttekst als `have` "
                f"kreeg, stopte ook. Uit deze resultaten schreef Qwen3: *{answer}* Met het plan van de course, het plan voor "
                f"twee gerechten uit oefening 2, met {len(course['steps'])} stappen in {waves} golven, schreef het: *{given}*")
    return (f"### Worked example\n\nQuestion {WORKED + 1}, *{q}* Qwen3's plan has {k} steps: the recipe of `pancakes`, the "
            f"recipe of `{wrong_dish}`, the pantry for `\"#1\"`, and `missing` with `\"#1\"` and `\"#2\"`. Read before it runs, "
            f"it shows three mistakes. The kitchen has no dish `{wrong_dish}`, only `pasta with pesto`. Step 4 compares the "
            f"recipe of pancakes with step 2, the other recipe, instead of with step 3, the pantry. And for the second dish, no "
            f"step works out what is missing. The code ran it as written: step 2 stopped with *{error2}*, and step 4, which got "
            f"that error text as `have`, stopped too. From these results, Qwen3 wrote: *{answer}* With the course's plan, the "
            f"plan for two dishes of exercise 2, with {len(course['steps'])} steps in {waves} waves, it wrote: *{given}*")


# --- exercise 1: run a plan whose steps use earlier results -----------------------------------------------------------

START_RUN = '''def run_plan(plan):
    results = {}
    for step in plan["steps"]:
        arguments = {}
        for name, value in step["args"].items():
            ...
        results["#" + str(len(results) + 1)] = FUNCTIONS[step["tool"]](arguments)
    return results
'''
SOLUTION_RUN = START_RUN.replace('''            ...
''', '''            if isinstance(value, str) and value in results:
                value = results[value]
            arguments[name] = value
''')
PLANNING = {"FUNCTIONS": FUNCTIONS}


class Plan(dict):
    """A plan, which a check's message names by a short name instead of writing it all out."""

    def __init__(self, name: str, steps: list):
        super().__init__(steps=steps)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Plan(self.name, copy.deepcopy(self["steps"], memo))


def _step(tool: str, **args) -> dict:
    return {"tool": tool, "args": args}


def plan_for(dish: str) -> list:
    """The steps of the course's plan for one dish: its recipe, the pantry for its ingredients, and what is missing."""
    return [_step("recipe", dish=dish), _step("pantry", items="#1"), _step("missing", need="#1", have="#2")]


def course_plan(dishes) -> dict:
    """The course's own plan for the dishes of a question: first every recipe, then the pantry for each, then what is
    missing for each."""
    k = len(dishes)
    return {"steps": [_step("recipe", dish=d) for d in dishes] + [_step("pantry", items=f"#{j + 1}") for j in range(k)]
            + [_step("missing", need=f"#{j + 1}", have=f"#{k + j + 1}") for j in range(k)]}


RECORD = {}
for _i, (_q, _ds) in enumerate(ASKED):
    RECORD[f"given-{_i}"] = dict(solve_ask(_q, course_plan(_ds)), seeds=1)
    RECORD[f"plan-{_i}"] = {"messages": asked(PLAN.format(question=_q)), "response_format": {"type": "json_object"},
                            "max_tokens": 300, "temperature": 0, "seeds": 1, "then": 2}
    RECORD[f"loop-{_i}"] = {"messages": [{"role": "system", "content": LOOP_SYSTEM}] + asked(_q), "tools": LOOP_TOOLS,
                            "max_tokens": 200, "temperature": 0, "steps": STEPS, "seeds": 1}


P_PANCAKES = Plan("plan_pancakes", plan_for("pancakes"))
P_TWO = Plan("plan_two_dishes", [_step("recipe", dish="pancakes"), _step("recipe", dish="pasta with pesto"),
                                 _step("pantry", items="#1"), _step("pantry", items="#2"),
                                 _step("missing", need="#1", have="#3"), _step("missing", need="#2", have="#4")])
P_ONE = Plan("plan_one_step", [_step("recipe", dish="omelette")])
P_LATE = Plan("plan_later_use", [_step("recipe", dish="apple pie"), _step("recipe", dish="rice pudding"),
                                 _step("pantry", items="#1"), _step("missing", need="#1", have="#3")])
P_SWAPPED = Plan("plan_args_swapped", [_step("recipe", dish="omelette"), _step("pantry", items="#1"),
                                       {"tool": "missing", "args": {"have": "#2", "need": "#1"}}])
P_COMPARE = Plan("plan_compare_recipes", [_step("recipe", dish="pancakes"), _step("recipe", dish="omelette"),
                                          _step("missing", need="#1", have="#2")])
P_LIST = Plan("plan_with_a_list", [_step("pantry", items=["milk", "eggs"]), _step("recipe", dish="pancakes"),
                                   _step("missing", need="#2", have="#1")])
RUN_CASES = [(P_PANCAKES,), (P_TWO,), (P_ONE,), (P_LATE,), (P_SWAPPED,), (P_COMPARE,), (P_LIST,)]


def course_run(plan) -> dict:
    return execute(plan)[0]


_NO_LOOKUP = SOLUTION_RUN.replace('''            if isinstance(value, str) and value in results:
                value = results[value]
''', "")
_LAST_ONLY = SOLUTION_RUN.replace('''                value = results[value]
''', '''                value = list(results.values())[-1]
''')
_NO_TEXT_RUN = SOLUTION_RUN.replace("if isinstance(value, str) and value in results:", "if value in results:")

_RUN_WORDS = {
    "en": {"title": "Your run_plan on the plan for pancakes", "step": "step", "gave": "result", "error": "stops with {e}",
           "same": "as the course's", "differs": "the course's: {w}"},
    "nl": {"title": "Je run_plan op het plan voor pancakes", "step": "stap", "gave": "resultaat", "error": "stopt met {e}",
           "same": "zoals die van de course", "differs": "die van de course: {w}"},
}

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def step_text(number: str, step: dict) -> str:
    """A step as one line: #2 pantry(items=#1)."""
    args = ", ".join(f"{k}={v if isinstance(v, str) and v.startswith('#') else json.dumps(v)}" for k, v in step["args"].items())
    return f"{number} {step['tool']}({args})"


def plan_rows(plan, got: dict | None, want: dict, w: dict) -> str:
    rows = []
    for k, step in enumerate(plan["steps"], 1):
        number = f"#{k}"
        theirs = want.get(number)
        if got is None:
            mine = ""
        else:
            value = got.get(number, "–") if isinstance(got, dict) else "–"
            ok = value == theirs
            note = "" if ok else (f'<div style="{_UI};font-size:12px;color:var(--ea-loss-text)">'
                                  f'{html.escape(w["differs"].format(w=_shown(theirs)))}</div>')
            mine = _code(_shown(value), "var(--ea-data-text)" if ok else "var(--ea-loss-text)") + note
        rows.append(f'<tr><td style="{_CELL}">{_code(step_text(number, step), "var(--ea-pred-text)")}</td>'
                    f'<td style="{_CELL}">{mine if got is not None else _code(_shown(theirs), "var(--ea-data-text)")}</td></tr>')
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
            f'<th style="{_HEAD}">{w["step"]}</th><th style="{_HEAD}">{w["gave"]}</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def run_picture(seen) -> str:
    """The plan for pancakes, step by step, with what the learner's run_plan gave for each step beside the course's."""
    run_ = seen.run
    fn = run_.names.get("run_plan") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _RUN_WORDS[seen.lang]
    try:
        got = fn(copy.deepcopy(P_PANCAKES))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    return yours.figure(w["title"], plan_rows(P_PANCAKES, got, course_run(P_PANCAKES), w), alt=w["title"])


# --- exercise 2: the steps that can run at the same time --------------------------------------------------------------

START_WAVES = '''def waves(plan):
    wave = {}
    for step in plan["steps"]:
        w = 1
        for name, value in step["args"].items():
            ...
        wave["#" + str(len(wave) + 1)] = w
    return wave
'''
SOLUTION_WAVES = START_WAVES.replace('''            ...
''', '''            if isinstance(value, str) and value in wave:
                w = max(w, wave[value] + 1)
''')


def course_waves(plan) -> dict:
    """The wave of each step of any plan a model writes: 1 for a step that uses no earlier result, else one more than the
    latest wave of the results it uses."""
    wave = {}
    for step in plan["steps"]:
        w = 1
        args = step.get("args") if isinstance(step, dict) else None
        for value in args.values() if isinstance(args, dict) else []:
            if isinstance(value, str) and value in wave:
                w = max(w, wave[value] + 1)
        wave["#" + str(len(wave) + 1)] = w
    return wave


WAVE_CASES = [(P_TWO,), (P_PANCAKES,), (P_ONE,), (P_LATE,), (P_SWAPPED,), (P_COMPARE,), (P_LIST,)]
_NO_TEXT_WAVES = SOLUTION_WAVES.replace("if isinstance(value, str) and value in wave:", "if value in wave:")
_LAST_REF = SOLUTION_WAVES.replace("w = max(w, wave[value] + 1)", "w = wave[value] + 1")
_COUNT = SOLUTION_WAVES.replace("w = max(w, wave[value] + 1)", "w = w + 1")
_NO_PLUS = SOLUTION_WAVES.replace("w = max(w, wave[value] + 1)", "w = max(w, wave[value])")

_WAVE_WORDS = {
    "en": {"title": "Your waves for the plan for two dishes", "wave": "wave {k}", "error": "stops with {e}",
           "differs": "The course's waves: {w}"},
    "nl": {"title": "Je golven voor het plan voor twee gerechten", "wave": "golf {k}", "error": "stopt met {e}",
           "differs": "De golven van de course: {w}"},
}


def waves_html(plan, wave: dict, w: dict) -> str:
    """The steps of a plan in columns, one column per wave: the steps of a column can run at the same time."""
    numbers = [f"#{k}" for k in range(1, len(plan["steps"]) + 1)]
    levels = sorted({wave.get(n) for n in numbers if isinstance(wave.get(n), int)})
    cols = []
    for level in levels:
        boxes = "".join(f'<div style="border:1px solid var(--ea-line);border-radius:4px;padding:0.2rem 0.4rem;margin:0.2rem 0">'
                        f'{_code(step_text(n, plan["steps"][int(n[1:]) - 1]), "var(--ea-pred-text)")}</div>'
                        for n in numbers if wave.get(n) == level)
        cols.append(f'<div style="flex:1 1 10rem;min-width:9rem"><div style="{_UI};font-size:12px;color:var(--ea-ink-2)">'
                    f'{html.escape(w["wave"].format(k=level))}</div>{boxes}</div>')
    return f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem">{"".join(cols)}</div>'


def waves_picture(seen) -> str:
    """The plan for two dishes, in the waves the learner's function found, with the course's waves where they differ."""
    run_ = seen.run
    fn = run_.names.get("waves") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _WAVE_WORDS[seen.lang]
    try:
        got = fn(copy.deepcopy(P_TWO))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    if not isinstance(got, dict):
        return yours.figure(w["title"], _code(repr(got)), alt=w["title"])
    want = course_waves(P_TWO)
    note = "" if got == want else f'<p style="{_UI};font-size:12px;color:var(--ea-loss-text)">{html.escape(w["differs"].format(w=want))}</p>'
    return yours.figure(w["title"], waves_html(P_TWO, got, w) + note, alt=w["title"])


# --- the pictures of the idea and of the measurement ------------------------------------------------------------------

def _box(text: str, role: str) -> str:
    """One step of a flow: the model's steps in the color of what a model makes, the code's in ink, a tool's in the
    color of data."""
    color = {"model": "var(--ea-pred)", "code": "var(--ea-ink-3)", "tool": "var(--ea-data)"}[role]
    return (f'<div style="border:1.5px solid {color};border-radius:6px;padding:0.3rem 0.5rem;{_UI};font-size:12px;'
            f'color:var(--ea-ink);background:var(--ea-page);min-width:0">{text}</div>')


def _arrow() -> str:
    return f'<div style="{_UI};font-size:13px;color:var(--ea-ink-3);text-align:center">↓</div>'


_FLOW_WORDS = {
    "en": {"title": "One step at a time, or the plan first", "loop": "Agent loop: a request for every step",
           "plan": "Plan first: two requests", "ask": "<b>model</b>: a tool call, or the answer",
           "run": "<b>your program</b>: run the call, send back its result",
           "again": "again, with every result so far in the request, until the model answers",
           "write": "<b>model</b>: write the whole plan, each step with the numbers of the results it uses",
           "steps": "<b>your program</b>: run step 1, 2, 3, …, each with the results it names",
           "solve": "<b>model</b>: write the answer from the question and the results",
           "alt": "Left, an agent loop: the model asks for a tool, the program runs it and sends back the result, and that repeats until the model answers. Right, plan first: the model writes the whole plan in one request, the program runs every step, and a second request writes the answer from the results."},
    "nl": {"title": "Stap voor stap, of eerst het plan", "loop": "Agent loop: een request voor elke stap",
           "plan": "Eerst het plan: twee requests", "ask": "<b>model</b>: een tool call, of het antwoord",
           "run": "<b>je programma</b>: voer de call uit, stuur het resultaat terug",
           "again": "opnieuw, met elk resultaat tot nu toe in het request, tot het model antwoordt",
           "write": "<b>model</b>: schrijf het hele plan, elke stap met de nummers van de resultaten die hij gebruikt",
           "steps": "<b>je programma</b>: voer stap 1, 2, 3, … uit, elk met de resultaten die hij noemt",
           "solve": "<b>model</b>: schrijf het antwoord uit de vraag en de resultaten",
           "alt": "Links een agent loop: het model vraagt om een tool, het programma voert hem uit en stuurt het resultaat terug, en dat herhaalt zich tot het model antwoordt. Rechts eerst het plan: het model schrijft het hele plan in één request, het programma voert elke stap uit, en een tweede request schrijft het antwoord uit de resultaten."},
}


def flow_picture(lang: str) -> str:
    """The agent loop beside plan first: the steps of each, and who takes them."""
    w = _FLOW_WORDS[lang]
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin-bottom:0.3rem"
    left = "".join([_box(w["ask"], "model"), _arrow(), _box(w["run"], "tool"),
                    f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);text-align:center;margin-top:0.3rem">↻ {w["again"]}</div>'])
    right = "".join([_box(w["write"], "model"), _arrow(), _box(w["steps"], "tool"), _arrow(), _box(w["solve"], "model")])
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:1rem">'
            f'<div style="min-width:0"><div style="{head}">{w["loop"]}</div>{left}</div>'
            f'<div style="min-width:0"><div style="{head}">{w["plan"]}</div>{right}</div></div>')
    return yours.figure(w["title"], body, alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "{n} questions, as a plan and as an agent loop", "question": "the question", "plan": "Qwen3's plan: its steps",
           "given": "the course's plan", "loop": "agent loop: its calls", "right": "right", "wrong": "not right", "stopped": "stopped at the step limit",
           "unread": "no plan the code could read", "total": "{k} of {n} right", "none": "no call", "times": "{name} ×{k}",
           "requests": "{k} requests"},
    "nl": {"title": "{n} vragen, als plan en als agent loop", "question": "de vraag", "plan": "plan van Qwen3: zijn stappen",
           "given": "plan van de course", "loop": "agent loop: zijn calls", "right": "goed", "wrong": "niet goed", "stopped": "gestopt bij de stappenlimiet",
           "unread": "geen plan dat de code kon lezen", "total": "{k} van de {n} goed", "none": "geen call", "times": "{name} ×{k}",
           "requests": "{k} requests"},
}


def names_line(names: list, w: dict) -> str:
    """Tool names in order, with a run of the same name as one: recipe ×2, pantry ×2, missing ×2."""
    if not names:
        return w["none"]
    out, k = [], 1
    for j, name in enumerate(names):
        if j + 1 < len(names) and names[j + 1] == name:
            k += 1
            continue
        out.append(w["times"].format(name=name, k=k) if k > 1 else str(name))
        k = 1
    return ", ".join(out)


def _mark(ok: bool, w: dict, why: str = "") -> str:
    color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
    word = w["right"] if ok else (why or w["wrong"])
    return f'<span style="{_UI};font-size:12px;color:{color};font-weight:{400 if ok else 600}">{html.escape(word)}</span>'


def table_picture(lang: str) -> str:
    """Each question with the plan's steps and the loop's calls, and whether each answer was right."""
    w, f = _TABLE_WORDS[lang], findings()
    rows = []
    for i, (q, _) in enumerate(ASKED):
        plan = plan_of(i)
        if plan is None:
            mine = _mark(False, w, w["unread"])
        else:
            tools = [st.get("tool") if isinstance(st, dict) else "?" for st in plan["steps"]]
            mine = f'{_code(names_line(tools, w), "var(--ea-pred-text)")}<br>{_mark(i in f["plan_right"], w)}'
        steps = loop_steps(i)
        loop = (f'{_code(names_line([name for name, _ in W.calls_of(steps)], w), "var(--ea-pred-text)")}<br>'
                f'{_mark(i in f["loop_right"], w, w["stopped"] if i in f["loop_unfinished"] else "")}')
        given = _mark(i in f["given_right"], w)
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(q)}</td><td style="{_CELL}">{mine}</td>'
                    f'<td style="{_CELL}">{given}</td><td style="{_CELL}">{loop}</td></tr>')
    n = f["questions"]
    foot = (f'<tr><td style="{_CELL}"></td>'
            f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f["plan_right"]), n=n)}<br>'
            f'<span style="font-weight:400;color:var(--ea-ink-2)">{w["requests"].format(k=f["plan_requests"])}</span></td>'
            f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f["given_right"]), n=n)}<br>'
            f'<span style="font-weight:400;color:var(--ea-ink-2)">{w["requests"].format(k=n)}</span></td>'
            f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f["loop_right"]), n=n)}<br>'
            f'<span style="font-weight:400;color:var(--ea-ink-2)">{w["requests"].format(k=f["loop_requests"])}</span></td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["question"], w["plan"], w["given"], w["loop"]))
             + f'</tr></thead><tbody>{"".join(rows)}{foot}</tbody></table></div>')
    title = w["title"].format(n=n)
    return yours.figure(title, table, alt=title)


_WORKED_WORDS = {
    "en": {"title": "Question {k} as a plan, step by step", "wrote": "1. The plan Qwen3 wrote", "ran": "2. What the code ran",
           "answered": "3. The answer Qwen3 wrote from the results", "step": "step", "gave": "result"},
    "nl": {"title": "Vraag {k} als plan, stap voor stap", "wrote": "1. Het plan dat Qwen3 schreef", "ran": "2. Wat de code uitvoerde",
           "answered": "3. Het antwoord dat Qwen3 uit de resultaten schreef", "step": "stap", "gave": "resultaat"},
}


def worked_picture(i: int, lang: str) -> str:
    """One recorded question as a plan: the JSON the model wrote, each step with its result, and the answer."""
    w = _WORKED_WORDS[lang]
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin:0.5rem 0 0.2rem"
    plan = plan_of(i)
    body = (f'<div style="{head}">{w["wrote"]}</div>'
            f'<pre style="margin:0;white-space:pre-wrap">{_code(chain_of(i)[0]["text"], "var(--ea-pred-text)")}</pre>')
    if plan is not None:
        body += f'<div style="{head}">{w["ran"]}</div>' + plan_rows(plan, None, execute(plan)[0], w)
        body += f'<div style="{head}">{w["answered"]}</div>' + _code(answer_of(i) or "", "var(--ea-pred-text)")
    title = w["title"].format(k=i + 1)
    return yours.figure(title, body, alt=title)


# --- the exercises ------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


def _value(v) -> str:
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return repr(v)


def _str_get(e) -> bool:
    return isinstance(e, AttributeError) and "'str' object has no attribute 'get'" in str(e)


_LOOKUP_ALL = SOLUTION_RUN.replace('''            if isinstance(value, str) and value in results:
                value = results[value]
            arguments[name] = value
''', '''            arguments[name] = results[value]
''')


def _unhashable(e) -> bool:
    return isinstance(e, TypeError) and "unhashable" in str(e)
_NO_ARGUMENTS = SOLUTION_RUN.replace("            arguments[name] = value\n", "")

EXERCISES = {
    "run-plan": FunctionBox(
        name="run-plan",
        picture=run_picture,
        function="run_plan",
        start=START_RUN,
        solution=SOLUTION_RUN,
        names=PLANNING,
        visible=RUN_CASES[:1],
        hidden=RUN_CASES[1:],
        show=_value,
        mistakes=[
            Mistake("no-lookup", error=_str_get,
                    en="With `{call}`, the tool `missing` got the text `\"#2\"` as `have`, not the result of step 2. When a value is the number of an earlier step, give the tool that step's result: `results[value]`.",
                    nl="Met `{call}` kreeg de tool `missing` de tekst `\"#2\"` als `have`, niet het resultaat van stap 2. Is een waarde het nummer van een eerdere stap, geef de tool dan het resultaat van die stap: `results[value]`."),
            Mistake("lookup-all", error=lambda e: isinstance(e, KeyError) and e.args == ("pancakes",),
                    en="With `{call}`, your code looks up every value in `results`, also `\"pancakes\"`, which is the name of a dish and not the number of a step. Look a value up only when it is in `results`: `if value in results:`.",
                    nl="Met `{call}` zoekt je code elke waarde op in `results`, ook `\"pancakes\"`, en dat is de naam van een gerecht, niet het nummer van een stap. Zoek een waarde alleen op als hij in `results` staat: `if value in results:`."),
            Mistake("no-arguments", error=lambda e: isinstance(e, KeyError) and e.args == ("dish",),
                    en="With `{call}`, the tool `recipe` found no `\"dish\"` in its arguments: `arguments` stays empty. Put each value in `arguments` under its name: `arguments[name] = value`.",
                    nl="Met `{call}` vond de tool `recipe` geen `\"dish\"` in zijn argumenten: `arguments` blijft leeg. Zet elke waarde in `arguments` onder zijn naam: `arguments[name] = value`."),
            Mistake("list", error=_unhashable,
                    en="With `{call}`, a value is a list, `[\"milk\", \"eggs\"]`, and `value in results` stops with an error: a list cannot be a key of a dictionary. Only a text can be the number of a step, so look first whether `value` is a text: `isinstance(value, str)`.",
                    nl="Met `{call}` is een waarde een lijst, `[\"milk\", \"eggs\"]`, en `value in results` stopt met een fout: een lijst kan geen key van een dictionary zijn. Alleen een tekst kan het nummer van een stap zijn, dus kijk eerst of `value` een tekst is: `isinstance(value, str)`."),
            Mistake("last-only", wrong=_LAST_ONLY,
                    en="With `{call}`, a number gets the result of the last step so far, but `\"#1\"` is the result of step 1, wherever it is used. Look the result up by its number: `results[value]`.",
                    nl="Met `{call}` krijgt een nummer het resultaat van de laatste stap tot nu toe, maar `\"#1\"` is het resultaat van stap 1, waar het ook gebruikt wordt. Zoek het resultaat op met zijn nummer: `results[value]`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop goes through the arguments of each step, but puts nothing in `arguments` yet. A value that is the number of an earlier step becomes that step's result; every value then goes into `arguments` under its name.",
                      "nl": "De loop gaat door de argumenten van elke stap, maar zet nog niets in `arguments`. Een waarde die het nummer van een eerdere stap is, wordt het resultaat van die stap; daarna gaat elke waarde onder zijn naam in `arguments`."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should be `{want}`. Compare them step by step in the picture.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Vergelijk ze stap voor stap in het plaatje."},
            "almost": {"en": "Right for the plan for pancakes. But with `{call}`, your function gives back `{got}`, and it should be `{want}`. Does it work for a step that uses a result from further back, for arguments in another order, and for a list as a value?",
                       "nl": "Goed voor het plan voor pancakes. Maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Werkt het ook voor een stap die een resultaat van verder terug gebruikt, voor argumenten in een andere volgorde, en voor een lijst als waarde?"},
            "correct": {"en": "Your `run_plan` is the executor of plan-and-execute: it runs each step with the tool the plan names, and gives a step the result of an earlier step wherever the plan writes its number. Right for all {n} plans of the check.",
                        "nl": "Je `run_plan` is de uitvoerder van plan-and-execute: hij voert elke stap uit met de tool die het plan noemt, en geeft een stap het resultaat van een eerdere stap overal waar het plan zijn nummer schrijft. Goed bij alle {n} plannen van de check."},
        },
    ),
    "waves": FunctionBox(
        name="waves",
        picture=waves_picture,
        function="waves",
        start=START_WAVES,
        solution=SOLUTION_WAVES,
        visible=WAVE_CASES[:1],
        hidden=WAVE_CASES[1:],
        show=_value,
        mistakes=[
            Mistake("list", error=_unhashable,
                    en="With `{call}`, a value is a list, `[\"milk\", \"eggs\"]`, and `value in wave` stops with an error: a list cannot be a key of a dictionary. Only a text can be the number of a step, so look first whether `value` is a text: `isinstance(value, str)`.",
                    nl="Met `{call}` is een waarde een lijst, `[\"milk\", \"eggs\"]`, en `value in wave` stopt met een fout: een lijst kan geen key van een dictionary zijn. Alleen een tekst kan het nummer van een stap zijn, dus kijk eerst of `value` een tekst is: `isinstance(value, str)`."),
            Mistake("no-plus", wrong=_NO_PLUS,
                    en="With `{call}`, a step lands in the same wave as a result it uses. It can only start once that result is there: in the wave after it, `wave[value] + 1`.",
                    nl="Met `{call}` komt een stap in dezelfde golf als een resultaat dat hij gebruikt. Hij kan pas beginnen als dat resultaat er is: in de golf erna, `wave[value] + 1`."),
            Mistake("last-ref", wrong=_LAST_REF,
                    en="With `{call}`, a step that uses two results gets the wave after the one it reads last, not after the latest of the two. A step can only start when all its results are there: keep the largest, with `max`.",
                    nl="Met `{call}` krijgt een stap die twee resultaten gebruikt de golf na het resultaat dat hij als laatste leest, niet na het laatste van de twee. Een stap kan pas beginnen als al zijn resultaten er zijn: houd de grootste, met `max`."),
            Mistake("count", wrong=_COUNT,
                    en="With `{call}`, your waves count how many results a step uses. But two results of wave 1 are both there after wave 1: a step's wave is one more than the latest wave it waits for.",
                    nl="Met `{call}` tellen je golven hoeveel resultaten een stap gebruikt. Maar twee resultaten uit golf 1 zijn er allebei na golf 1: de golf van een stap is één meer dan de laatste golf waarop hij wacht."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop starts each step in wave 1, and changes nothing yet. For each value that is the number of an earlier step, the step has to wait for that step: its wave becomes at least one more.",
                      "nl": "De loop zet elke stap in golf 1, en verandert nog niets. Voor elke waarde die het nummer van een eerdere stap is, moet de stap op die stap wachten: zijn golf wordt minstens één meer."},
            "wrong": {"en": "With `{call}`, your waves are `{got}`, and they should be `{want}`. Compare them in the picture.",
                      "nl": "Met `{call}` zijn je golven `{got}`, en het zou `{want}` moeten zijn. Vergelijk ze in het plaatje."},
            "almost": {"en": "Right for the plan for two dishes. But with `{call}`, your waves are `{got}`, and they should be `{want}`. Does it work for a step that uses two results of the same wave, for arguments in another order, and for a list as a value?",
                       "nl": "Goed voor het plan voor twee gerechten. Maar met `{call}` zijn je golven `{got}`, en het zou `{want}` moeten zijn. Werkt het ook voor een stap die twee resultaten uit dezelfde golf gebruikt, voor argumenten in een andere volgorde, en voor een lijst als waarde?"},
            "correct": {"en": "Your `waves` finds the steps that can run at the same time, as LLMCompiler does: a step waits only for the results it uses. Right for all {n} plans of the check.",
                        "nl": "Je `waves` vindt de stappen die tegelijk kunnen draaien, zoals LLMCompiler doet: een stap wacht alleen op de resultaten die hij gebruikt. Goed bij alle {n} plannen van de check."},
        },
    ),
}

REQUIRED = ["run-plan", "waves", "plan-or-loop"]

CASES = [
    ("run-plan", START_RUN, "empty", None),
    ("run-plan", "", "empty", None),
    ("run-plan", SOLUTION_RUN, "correct", None),
    ("run-plan", _NO_LOOKUP, "error", "no-lookup"),
    ("run-plan", _LOOKUP_ALL, "error", "lookup-all"),
    ("run-plan", _NO_ARGUMENTS, "error", "no-arguments"),
    ("run-plan", _LAST_ONLY, "wrong", "last-only"),
    ("run-plan", _NO_TEXT_RUN, "error", "list"),
    ("run-plan", SOLUTION_RUN.replace('''            if isinstance(value, str) and value in results:
                value = results[value]
            arguments[name] = value
''', '''            arguments[name] = results.get(value, value) if isinstance(value, str) else value
'''), "correct", None),
    ("run-plan", SOLUTION_RUN.replace("value in results:", "value.startswith(\"#\"):"), "correct", None),
    ("waves", START_WAVES, "empty", None),
    ("waves", "", "empty", None),
    ("waves", SOLUTION_WAVES, "correct", None),
    ("waves", _NO_PLUS, "wrong", "no-plus"),
    ("waves", _LAST_REF, "wrong", "last-ref"),
    ("waves", _COUNT, "wrong", "count"),
    ("waves", _NO_TEXT_WAVES, "error", "list"),
    ("waves", SOLUTION_WAVES.replace('''            if isinstance(value, str) and value in wave:
                w = max(w, wave[value] + 1)
''', '''            if isinstance(value, str) and value in wave and wave[value] + 1 > w:
                w = wave[value] + 1
'''), "correct", None),
]


# --- the demo: a plan of Qwen3, run by the learner's run_plan ---------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def _safe(tool):
    """A tool for the demo that gives back its error as text, as the course's executor does, so that the steps after
    it still run."""
    def run(arguments):
        try:
            return tool(arguments)
        except Exception as e:   # noqa: BLE001 - a plan from a model: a dish the kitchen does not have, a wrong argument
            return f"Error: {type(e).__name__}: {e}"
    return run


DEMO_FUNCTIONS = {name: _safe(tool) for name, tool in FUNCTIONS.items()}


def learner_fn(key: str, text):
    """The learner's function from an exercise, when it is right; else None. In the demo, run_plan gets the tools that
    give back an error as text."""
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    names = dict(ex.names, FUNCTIONS=DEMO_FUNCTIONS) if key == "run-plan" else dict(ex.names)
    return run_code(text, names).names.get(ex.function)


def _post(url: str, messages: list, max_tokens: int, json_mode: bool = False) -> str:
    import urllib.request
    body = {"model": serve.MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"] or ""


def demo(run_fn, waves_fn, question: str, words: dict, lang: str) -> str:
    """A question to Qwen3 as a plan: the plan it writes, the learner's run_plan on it, the learner's waves when that
    exercise is right too, and the answer Qwen3 writes from the results."""
    head = f"{_UI};font-size:12px;font-weight:600;color:var(--ea-ink-2);margin:0.5rem 0 0.2rem"
    try:
        running = serve.model_server()
        text = _post(running.url, asked(PLAN.format(question=question)), 300, json_mode=True)
    except Exception as e:   # the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        raise
    body = (f'<div style="{head}">{html.escape(words["wrote"])}</div>'
            f'<pre style="margin:0;white-space:pre-wrap">{_code(text, "var(--ea-pred-text)")}</pre>')
    plan = read_plan(text)
    if plan is None:
        body += f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(words["unread"])}</p>'
        return yours.figure(words["title"], body, alt=words["title"])
    try:
        results = run_fn(copy.deepcopy(plan))
    except Exception as e:   # noqa: BLE001 - the learner's code, on a plan a model wrote
        body += (f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">'
                 f'{html.escape(words["error"].format(e=f"{type(e).__name__}: {e}"))}</p>')
        return yours.figure(words["title"], body, alt=words["title"])
    if waves_fn is not None:
        try:
            body += f'<div style="{head}">{html.escape(words["waves"])}</div>' + waves_html(plan, waves_fn(copy.deepcopy(plan)), words)
        except Exception:   # noqa: BLE001 - the learner's code; the waves are extra
            pass
    lines = [f"{n} {step.get('tool')}({json.dumps(step.get('args'))}) = {_shown(r)}"
             for (n, r), step in zip(results.items(), plan["steps"])] if isinstance(results, dict) else []
    body += f'<div style="{head}">{html.escape(words["ran"])}</div>' + "".join(
        f'<div>{_code(line, "var(--ea-data-text)")}</div>' for line in lines)
    answer = _post(running.url, asked(SOLVE.format(question=question, results="\n".join(lines))), 100)
    body += f'<div style="{head}">{html.escape(words["answered"])}</div>' + _code(answer, "var(--ea-pred-text)")
    return yours.figure(words["title"], body, alt=words["title"])
