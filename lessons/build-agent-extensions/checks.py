# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-agent-extensions: AGENTS.md, skills, OpenAPI and plugins. Six skills of the course's kitchen, each a
folder with a SKILL.md (data/skills/), whose name and description an agent always sees and whose instructions it loads
with a tool only when a question needs them. The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json):
twelve questions, two per skill, asked without skills, with the list of skills and the tool load_skill, and with all
six skills in the system prompt. What they show, their pictures, the exercises (the name and description of a skill
read from its SKILL.md, and tools made from the OpenAPI description of a web API) and the demo, with a skill the
learner writes."""
import copy
import html
import json
import re

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"
SKILLS_DIR = paths.ROOT / "data" / "skills"
NAMES = ["cup-measures", "scale-recipe", "food-storage", "oven-settings", "allergens", "in-season"]


def skill_text(name: str) -> str:
    return (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def split_skill(text: str) -> tuple:
    """The course's reading of a SKILL.md: its frontmatter as {key: value}, and its instructions, the text after it."""
    lines = text.split("\n")
    end = lines.index("---", 1)
    front = {}
    for line in lines[1:end]:
        if ": " in line and not line.startswith(" "):
            key, value = line.split(": ", 1)
            front[key.strip()] = value.strip()
    return front, "\n".join(lines[end + 1:]).strip()


def index_line(front: dict) -> str:
    """A skill as the agent always sees it: one line with its name and description."""
    return "- " + front["name"] + ": " + front["description"]


SKILLS = {name: split_skill(skill_text(name)) for name in NAMES}   # name: (frontmatter, instructions)

# --- three ways to give an agent the skills -------------------------------------------------------------------------------

BASE = "You are a kitchen assistant. Answer in one or two short sentences."
INDEX_HEAD = ("You have skills: instructions for some kinds of questions. When a question matches a skill, first load "
              "it with the tool load_skill, then answer as it says. Your skills:")
ALL_HEAD = "Follow these instructions where they apply."


def index_system(fronts: list) -> str:
    """The system prompt with the list of skills: a line per skill."""
    return BASE + "\n\n" + INDEX_HEAD + "\n" + "\n".join(index_line(f) for f in fronts)


def all_system() -> str:
    """The system prompt with every skill's instructions in full."""
    return BASE + "\n\n" + ALL_HEAD + "\n\n" + "\n\n".join(SKILLS[n][1] for n in NAMES)


def load_tool(names: list) -> dict:
    return {"type": "function", "function": {
        "name": "load_skill", "description": "Load the instructions of one of your skills.",
        "parameters": {"type": "object", "properties": {"name": {
            "type": "string", "enum": list(names), "description": "the name of the skill, such as " + names[0]}},
            "required": ["name"]}}}


INDEX = index_system([SKILLS[n][0] for n in NAMES])
LOAD = load_tool(NAMES)


def load_skill(arguments, skills: dict | None = None) -> str:
    """The tool load_skill: the instructions of the skill with that name, or an error that names the skills."""
    skills = SKILLS if skills is None else skills
    name = arguments.get("name") if isinstance(arguments, dict) else None
    if name not in skills:
        return "Error: there is no skill " + str(name) + ". The skills are: " + ", ".join(skills) + "."
    return skills[name][1]


def not_a_tool(name: str, helpful: bool, skills: dict | None = None) -> str:
    """The result of a call of a tool that is not there. With `helpful`, a skill's name called as a tool gets an error
    that says how to load the skill instead."""
    skills = SKILLS if skills is None else skills
    if helpful and name in skills:
        return f"Error: {name} is a skill, not a tool. Load its instructions with the tool load_skill, with the name {name}."
    return "Error: no tool " + name


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: run each call of load_skill, and give back its result as a message with the role tool; a call
    of another tool gets an error, which says how to load a skill when the ask has "helpful"."""
    out = []
    helpful = bool((ask or {}).get("helpful"))
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            arguments = None
        name = c["function"]["name"]
        result = load_skill(arguments) if name == "load_skill" else not_a_tool(name, helpful)
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


def _says(number: str):
    return lambda t: re.search(rf"(?<![\d.]){number}(?![\d])", t) is not None


def _starts(word: str):
    return lambda t: re.match(rf"\W*{word}\b", t) is not None


QUESTIONS = [   # (question, the skill it needs, whether an answer is right: lower case in)
    ("How many grams is 3 cups of flour?", "cup-measures", _says("375")),
    ("A recipe asks for 2 tablespoons of butter. How many grams is that?", "cup-measures", _says("28")),
    ("The pancake recipe uses 450 grams of milk. How much milk do I need for 6 people?", "scale-recipe", _says("675")),
    ("The apple pie recipe uses 300 grams of flour. How much flour do I need for 2 people?", "scale-recipe", _says("150")),
    ("I cooked rice yesterday. How many days does cooked rice keep in the fridge?", "food-storage",
     lambda t: re.search(r"(?<![\d.])(1|one) day\b", t) is not None),
    ("How many days does soup keep in the fridge?", "food-storage", lambda t: re.search(r"(?<![\d.])(3|three) days\b", t) is not None),
    ("The recipe says 180 degrees. What gas mark is that?", "oven-settings", lambda t: re.search(r"gas mark (is )?4\b|\b4\b", t) is not None),
    ("The apple pie bakes at 200 degrees. What should I set my fan oven to?", "oven-settings", _says("180")),
    ("Is the tomato soup safe for someone with a celery allergy? Answer yes or no first.", "allergens", _starts("no")),
    ("Can someone with an egg allergy eat the rice pudding? Answer yes or no first.", "allergens", _starts("yes")),
    ("Are strawberries in season in the Netherlands in December? Answer yes or no first.", "in-season", _starts("no")),
    ("Is asparagus in season in the Netherlands in May? Answer yes or no first.", "in-season", _starts("yes")),
]


def right(i: int, text: str | None) -> bool:
    return bool(text) and QUESTIONS[i][2](text.strip().lower())


def asked(text: str, system: str) -> list:
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


RECORD = {}
for _i, (_q, _, _) in enumerate(QUESTIONS):
    RECORD[f"none-{_i}"] = {"messages": asked(_q, BASE), "max_tokens": 120, "temperature": 0, "seeds": 1}
    RECORD[f"index-{_i}"] = {"messages": asked(_q, INDEX), "tools": [LOAD], "max_tokens": 160, "temperature": 0,
                             "steps": 3, "seeds": 1}
    RECORD[f"all-{_i}"] = {"messages": asked(_q, all_system()), "max_tokens": 120, "temperature": 0, "seeds": 1}
    RECORD[f"helpful-{_i}"] = {"messages": asked(_q, INDEX), "tools": [LOAD], "max_tokens": 160, "temperature": 0,
                               "steps": 4, "helpful": True, "seeds": 1}


# --- exercise 1: a skill's name and description, from its SKILL.md -------------------------------------------------------

START_FRONT = '''def front(text):
    lines = text.split("\\n")
    end = lines.index("---", 1)   # the line that closes the frontmatter
    found = {}
    for line in lines[1:end]:
        ...
    return found
'''
SOLUTION_FRONT = START_FRONT.replace('''        ...
''', '''        if ": " in line:
            key, value = line.split(": ", 1)
            found[key] = value.strip()
''')


class Text(str):
    """A text of a check, which its message names by a short name instead of writing it all out."""

    def __new__(cls, name: str, content: str):
        obj = super().__new__(cls, content)
        obj.name = name
        return obj

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Text(self.name, str(self))


TEXT_CUPS = Text("cup_measures", skill_text("cup-measures"))
TEXT_LICENSE = Text("skill_with_license", """---
name: weekly-menu
description: Plan the dinners of a week: one dish a day, from the dishes of this kitchen. Use when the user asks what to cook this week.
license: CC-BY-SA-4.0
---

# A menu for the week

Pick seven dishes; no dish twice in a row.
name: not-this-one
""")
TEXT_SPACES = Text("skill_with_spaces", "---\nname: tea-times\ndescription: How long to let each kind of tea steep.   \n---\n\nGreen tea: 2 minutes.\n")
FRONT_VISIBLE = [(TEXT_CUPS,)]
FRONT_HIDDEN = [(TEXT_LICENSE,), (TEXT_SPACES,)]

_SPLIT_ALL = SOLUTION_FRONT.replace('key, value = line.split(": ", 1)', 'key, value = line.split(": ")[:2]')
_ALL_LINES = SOLUTION_FRONT.replace("for line in lines[1:end]:", "for line in lines:")
_NO_STRIP = SOLUTION_FRONT.replace("found[key] = value.strip()", "found[key] = value")


# --- exercise 2: tools from the OpenAPI description of a web API ---------------------------------------------------------

def _param(name: str, where: str, required: bool, about: str, schema: dict) -> dict:
    p = {"name": name, "in": where, "description": about, "schema": schema}
    if required:
        p["required"] = True
    return p


def _op(op_id: str, summary: str, params: list, answer: str) -> dict:
    out = {"operationId": op_id, "summary": summary}
    if params:
        out["parameters"] = params
    out["responses"] = {"200": {"description": answer}}
    return out


PRODUCT = _param("product", "path", True, "the product, as search_products names it, such as milk", {"type": "string"})
SHOP_API = {
    "openapi": "3.1.0",
    "info": {"title": "Corner shop", "version": "1.0"},
    "servers": [{"url": "https://shop.example/api"}],
    "paths": {
        "/products": {"get": _op("search_products", "Find products by a word in their name.",
                                 [_param("q", "query", True, "a word in the product's name, such as milk", {"type": "string"})],
                                 "The products whose name has the word, with their prices.")},
        "/cart/{product}": {
            "put": _op("add_to_cart", "Put a product in the cart, or change how many there are.",
                       [PRODUCT, _param("count", "query", False, "how many, such as 2",
                                        {"type": "integer", "minimum": 1, "default": 1})], "The cart after the change."),
            "delete": _op("remove_from_cart", "Take a product out of the cart.", [PRODUCT], "The cart after the change."),
        },
        "/cart": {"get": _op("show_cart", "What is in the cart now, with the total.", [], "The cart.")},
    },
}
SHOP_API_JSON = json.dumps(SHOP_API, indent=2)   # the specification as the lesson shows it
WEATHER_API = {
    "openapi": "3.1.0",
    "info": {"title": "Weather", "version": "2.1"},
    "paths": {"/forecast/{city}": {"get": _op("forecast", "The weather in a city for the next days.", [
        _param("city", "path", True, "the name of the city, such as Utrecht", {"type": "string"}),
        _param("days", "query", False, "how many days, from 1 to 7", {"type": "integer", "minimum": 1, "maximum": 7}),
        _param("units", "query", False, "metric or imperial", {"type": "string", "enum": ["metric", "imperial"]})],
        "The forecast, a day per line.")}},
}

START_OPENAPI = '''def openapi_tools(spec):
    tools = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            properties = {}
            required = []
            ...
            parameters = {"type": "object", "properties": properties, "required": required}
            tools.append({"type": "function", "function": {
                "name": operation["operationId"], "description": operation["summary"],
                "parameters": parameters}})
    return tools
'''
SOLUTION_OPENAPI = START_OPENAPI.replace('''            ...
''', '''            for p in operation.get("parameters", []):
                schema = dict(p["schema"])
                schema["description"] = p["description"]
                properties[p["name"]] = schema
                if p.get("required"):
                    required.append(p["name"])
''')


class Spec(dict):
    """An OpenAPI description of a check, which its message names by a short name instead of writing it all out."""

    def __init__(self, name: str, content: dict):
        super().__init__(content)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Spec(self.name, copy.deepcopy(dict(self), memo))


OPENAPI_VISIBLE = [(Spec("shop_api", SHOP_API),)]
OPENAPI_HIDDEN = [(Spec("weather_api", WEATHER_API),)]

_NO_REQUIRED = SOLUTION_OPENAPI.replace('''                if p.get("required"):
                    required.append(p["name"])
''', "")
_ALL_REQUIRED = SOLUTION_OPENAPI.replace('''                if p.get("required"):
                    required.append(p["name"])
''', '''                required.append(p["name"])
''')
_NO_DESCRIPTION = SOLUTION_OPENAPI.replace('''                schema = dict(p["schema"])
                schema["description"] = p["description"]
                properties[p["name"]] = schema
''', '''                properties[p["name"]] = p["schema"]
''')
_NO_GET = SOLUTION_OPENAPI.replace('for p in operation.get("parameters", []):', 'for p in operation["parameters"]:')
_REQUIRED_KEY = SOLUTION_OPENAPI.replace('if p.get("required"):', 'if p["required"]:')


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- comparing and showing what the functions give back -----------------------------------------------------------------

def _same_front(got, want) -> bool:
    return isinstance(got, dict) and got == want


def _front_shown(v) -> str:
    if not isinstance(v, dict):
        return repr(v)
    return "{" + ", ".join(f"{k!r}: {(x[:24] + '…') if isinstance(x, str) and len(x) > 25 else x!r}" for k, x in v.items()) + "}"


def _norm_tool(t):
    """A tool with its list of required parameters in a fixed order: the order of that list does not matter."""
    if not isinstance(t, dict) or not isinstance(t.get("function"), dict):
        return t
    f = dict(t["function"])
    params = f.get("parameters")
    if isinstance(params, dict) and isinstance(params.get("required"), list):
        f["parameters"] = {**params, "required": sorted(map(str, params["required"]))}
    return {**t, "function": f}


def _same_tools(got, want) -> bool:
    return isinstance(got, list) and len(got) == len(want) and all(_norm_tool(g) == _norm_tool(w) for g, w in zip(got, want))


def _signature(t) -> str:
    """A tool in a few words: its name, and each parameter, with * when it is required and ? when it has no description."""
    try:
        f = t["function"]
        props = f["parameters"]["properties"]
        need = f["parameters"].get("required", [])
        params = ", ".join(name + ("*" if name in need else "") + ("" if isinstance(x, dict) and x.get("description") else "?")
                           for name, x in props.items())
        return f"{f['name']}({params})"
    except (KeyError, TypeError, AttributeError):
        return repr(t)[:60]


def _tools_shown(v) -> str:
    if not isinstance(v, list):
        return repr(v)[:80]
    return "[" + ", ".join(_signature(t) for t in v) + "]"


def _missing_key(key: str):
    return lambda e: isinstance(e, KeyError) and e.args == (key,)


# --- the pictures of the exercises --------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _label(text: str) -> str:
    return f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.1rem 0 0.2rem">{html.escape(text)}</div>'


def _pre(inner: str) -> str:
    return (f'<pre style="margin:0.2rem 0;padding:0.3rem 0.5rem;border:1px solid var(--ea-line);border-radius:4px;'
            f'white-space:pre-wrap">{inner}</pre>')


def skill_html(text: str, lines: int | None = None) -> str:
    """A SKILL.md with its frontmatter in the color of what the agent always sees, and its instructions in grey; with
    `lines`, only that many lines of the instructions."""
    rows = text.rstrip("\n").split("\n")
    try:
        end = rows.index("---", 1)
    except ValueError:
        return _code(text, "var(--ea-ink-2)")
    body = rows[end + 1:]
    more = lines is not None and len(body) > lines
    body = body[:lines] if more else body
    return (_code("\n".join(rows[:end + 1]), "var(--ea-data-text)") + "\n" + _code("\n".join(body) + ("\n…" if more else ""), "var(--ea-ink-3)"))


_FRONT_WORDS = {
    "en": {"title": "The SKILL.md of cup-measures, and what your front reads from it", "file": "SKILL.md",
           "got": "your front gives", "course": "the course's front gives", "line": "the line the agent always sees, with your front:",
           "no_line": "Without a name and a description, there is no line for the agent to see.",
           "error": "Your front stops with {e}."},
    "nl": {"title": "De SKILL.md van cup-measures, en wat je front eruit leest", "file": "SKILL.md",
           "got": "je front geeft", "course": "de front van de course geeft", "line": "de regel die de agent altijd ziet, met je front:",
           "no_line": "Zonder name en description is er geen regel die de agent kan zien.",
           "error": "Je front stopt met {e}."},
}


def _dict_html(d, color: str) -> str:
    if not isinstance(d, dict):
        return _code(repr(d), "var(--ea-loss-text)")
    rows = "".join(f'<tr><td style="{_CELL}">{_code(repr(k), "var(--ea-ink-2)")}</td><td style="{_CELL}">{_code(repr(v), color)}</td></tr>'
                   for k, v in d.items())
    return f'<table style="border-collapse:collapse;width:100%"><tbody>{rows}</tbody></table>' if rows else _code("{}", color)


def front_picture(seen) -> str:
    """The SKILL.md of the visible case, the dictionary the learner's front gives, and the index line made of it."""
    run_ = seen.run
    fn = run_.names.get("front") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _FRONT_WORDS[seen.lang]
    box = EXERCISES["front"]
    want = box._run(box._fn(SOLUTION_FRONT), (TEXT_CUPS,))
    try:
        got = box._run(fn, (TEXT_CUPS,))
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    ok = _same_front(got, want)
    right_side = _label(w["got"]) + _dict_html(got, "var(--ea-data-text)" if ok else "var(--ea-loss-text)")
    if not ok:
        right_side += _label(w["course"]) + _dict_html(want, "var(--ea-data-text)")
    if isinstance(got, dict) and isinstance(got.get("name"), str) and isinstance(got.get("description"), str):
        right_side += _label(w["line"]) + _pre(_code(index_line(got), "var(--ea-ink)"))
    else:
        right_side += f'<p style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(w["no_line"])}</p>'
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:0.8rem">'
            f'<div style="min-width:0">{_label(w["file"])}{_pre(skill_html(str(TEXT_CUPS), 4))}</div>'
            f'<div style="min-width:0">{right_side}</div></div>')
    return yours.figure(w["title"], body, alt=w["title"])


_OPENAPI_WORDS = {
    "en": {"title": "The corner shop's web API, and the tools your openapi_tools makes of it", "api": "in the OpenAPI description",
           "tool": "your tool", "course": "the course's tool", "required": "required", "optional": "optional",
           "no_description": "no description", "missing": "(no tool)", "error": "Your openapi_tools stops with {e}."},
    "nl": {"title": "De web-API van de buurtwinkel, en de tools die je openapi_tools ervan maakt", "api": "in de OpenAPI-beschrijving",
           "tool": "je tool", "course": "de tool van de course", "required": "verplicht", "optional": "niet verplicht",
           "no_description": "geen description", "missing": "(geen tool)", "error": "Je openapi_tools stopt met {e}."},
}


def _tool_html(t, w: dict, color: str) -> str:
    try:
        f = t["function"]
        props = f["parameters"]["properties"]
        need = f["parameters"].get("required", [])
    except (KeyError, TypeError, AttributeError):
        return _code(repr(t), "var(--ea-loss-text)")
    rows = []
    for name, x in props.items():
        about = x.get("description") if isinstance(x, dict) else None
        kind = x.get("type", "") if isinstance(x, dict) else ""
        rows.append(f'<div>{_code(name, color)} {_code(str(kind), "var(--ea-ink-2)")} '
                    f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2)">{w["required"] if name in need else w["optional"]}'
                    f'{"" if about else " · " + w["no_description"]}</span></div>')
    return (f'<div>{_code(str(f.get("name")), color)}</div>'
            f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(str(f.get("description")))}</div>' + "".join(rows))


def openapi_picture(seen) -> str:
    """Each operation of the corner shop's API beside the tool the learner's function makes of it, and the course's
    tool where they differ."""
    run_ = seen.run
    fn = run_.names.get("openapi_tools") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _OPENAPI_WORDS[seen.lang]
    box = EXERCISES["openapi"]
    case = OPENAPI_VISIBLE[0]
    want = box._run(box._fn(SOLUTION_OPENAPI), case)
    try:
        got = box._run(fn, case)
    except Exception as e:   # noqa: BLE001 - the learner's code
        return yours.figure(w["title"], _code(w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"), alt=w["title"])
    got = got if isinstance(got, list) else []
    rows = []
    k = 0
    for path, operations in SHOP_API["paths"].items():
        for method, op in operations.items():
            mine = got[k] if k < len(got) else None
            ok = mine is not None and _norm_tool(mine) == _norm_tool(want[k])
            cells = [f'{_code(method.upper() + " " + path, "var(--ea-ink)")}<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">'
                     f'{html.escape(op["operationId"])}</div>',
                     _tool_html(mine, w, "var(--ea-data-text)" if ok else "var(--ea-loss-text)") if mine is not None else
                     f'<span style="{_UI};font-size:12px;color:var(--ea-loss-text)">{w["missing"]}</span>']
            if not ok:
                cells.append(_tool_html(want[k], w, "var(--ea-data-text)"))
            rows.append((cells, ok))
            k += 1
    wrong_any = any(not ok for _, ok in rows)
    heads = [w["api"], w["tool"]] + ([w["course"]] if wrong_any else [])
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in heads) + "</tr></thead><tbody>"
             + "".join("<tr>" + "".join(f'<td style="{_CELL}">{c}</td>' for c in cells)
                       + ("" if not wrong_any or len(cells) == 3 else f'<td style="{_CELL}"></td>') + "</tr>" for cells, _ in rows)
             + "</tbody></table></div>")
    return yours.figure(w["title"], table, alt=w["title"])


# --- the exercises ------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

class NamedBox(FunctionBox):
    """A FunctionBox whose messages name each case by its short name, such as `front(cup_measures)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(repr(a) for a in case)})"


EXERCISES = {
    "front": NamedBox(
        name="front",
        picture=front_picture,
        function="front",
        start=START_FRONT,
        solution=SOLUTION_FRONT,
        visible=FRONT_VISIBLE,
        hidden=FRONT_HIDDEN,
        compare=_same_front,
        show=_front_shown,
        mistakes=[
            Mistake("split-all", wrong=_SPLIT_ALL,
                    en="With `{call}`, the description stops at its own `: `. `line.split(\": \")` cuts at every `: `; cut only at the first, with `line.split(\": \", 1)`.",
                    nl="Met `{call}` houdt de description op bij zijn eigen `: `. `line.split(\": \")` knipt bij elke `: `; knip alleen bij de eerste, met `line.split(\": \", 1)`."),
            Mistake("all-lines", wrong=_ALL_LINES,
                    en="With `{call}`, a line from the instructions under the frontmatter changed the name. Read only the lines between the two `---` lines: `lines[1:end]`.",
                    nl="Met `{call}` veranderde een regel uit de instructies onder de frontmatter de name. Lees alleen de regels tussen de twee `---`-regels: `lines[1:end]`."),
            Mistake("no-strip", wrong=_NO_STRIP,
                    en="With `{call}`, the description keeps the spaces at its end. `.strip()` takes them off.",
                    nl="Met `{call}` houdt de description de spaties aan het eind. `.strip()` haalt ze eraf."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The function gives back an empty dictionary. For each line with `: ` in it, put the part before it in `found` as the key, and the part after it as the value.",
                      "nl": "De functie geeft een lege dictionary terug. Zet voor elke regel met `: ` erin het deel ervoor in `found` als key, en het deel erna als value."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should be `{want}`. Compare in the picture.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Vergelijk in het plaatje."},
            "almost": {"en": "Right for cup-measures. But with `{call}`, your function gives back `{got}`, and it should be `{want}`.",
                       "nl": "Goed voor cup-measures. Maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn."},
            "correct": {"en": "Your `front` reads the frontmatter of a skill: every key between the two `---` lines, with its value. Its name and description are what the agent always sees; the rest waits until the agent loads it. Right for all {n} cases of the check.",
                        "nl": "Je `front` leest de frontmatter van een skill: elke key tussen de twee `---`-regels, met zijn value. De name en description ervan ziet de agent altijd; de rest wacht tot de agent het laadt. Goed bij alle {n} gevallen van de check."},
        },
    ),
    "openapi": NamedBox(
        name="openapi",
        picture=openapi_picture,
        function="openapi_tools",
        start=START_OPENAPI,
        solution=SOLUTION_OPENAPI,
        visible=OPENAPI_VISIBLE,
        hidden=OPENAPI_HIDDEN,
        compare=_same_tools,
        show=_tools_shown,
        mistakes=[
            Mistake("no-get", error=_missing_key("parameters"),
                    en="With `{call}`, your function stops at an operation without parameters, such as `show_cart`: it has no key `\"parameters\"`. `operation.get(\"parameters\", [])` gives an empty list then.",
                    nl="Met `{call}` stopt je functie bij een operation zonder parameters, zoals `show_cart`: die heeft geen key `\"parameters\"`. `operation.get(\"parameters\", [])` geeft dan een lege lijst."),
            Mistake("required-key", error=_missing_key("required"),
                    en="With `{call}`, your function stops at an optional parameter, such as `count`: it has no key `\"required\"`. `p.get(\"required\")` gives `None` then, which counts as false.",
                    nl="Met `{call}` stopt je functie bij een parameter die niet verplicht is, zoals `count`: die heeft geen key `\"required\"`. `p.get(\"required\")` geeft dan `None`, en dat telt als onwaar."),
            Mistake("no-required", wrong=_NO_REQUIRED,
                    en="With `{call}`, your tools have no required parameters, so the model may leave out `q` and `product`. Add the name of each parameter with `\"required\": true` to `required`.",
                    nl="Met `{call}` hebben je tools geen verplichte parameters, dus het model mag `q` en `product` weglaten. Voeg de naam van elke parameter met `\"required\": true` toe aan `required`."),
            Mistake("all-required", wrong=_ALL_REQUIRED,
                    en="With `{call}`, every parameter is required in your tools, also `count`, which the API does not need. Add a name to `required` only when its parameter says `\"required\": true`.",
                    nl="Met `{call}` is in je tools elke parameter verplicht, ook `count`, die de API niet nodig heeft. Voeg een naam alleen aan `required` toe als zijn parameter `\"required\": true` zegt."),
            Mistake("no-description", wrong=_NO_DESCRIPTION,
                    en="With `{call}`, the parameters in your tools have no description, and the model then has only their names to go on. Copy the schema, and give it the parameter's `\"description\"`.",
                    nl="Met `{call}` hebben de parameters in je tools geen description, en het model heeft dan alleen hun namen. Kopieer het schema, en geef het de `\"description\"` van de parameter."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your tools have names and descriptions, but no parameters yet. For each parameter of the operation, put its schema with its description in `properties`, and its name in `required` when it is required.",
                      "nl": "Je tools hebben namen en descriptions, maar nog geen parameters. Zet voor elke parameter van de operation zijn schema met zijn description in `properties`, en zijn naam in `required` als hij verplicht is."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should be `{want}` (* means required, ? no description). The picture shows which tools differ.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn (* betekent verplicht, ? geen description). Het plaatje laat zien welke tools verschillen."},
            "almost": {"en": "Right for the corner shop. But with `{call}`, your function gives back `{got}`, and it should be `{want}` (* means required, ? no description). Does each schema keep all of its own keys, such as `enum` and `maximum`?",
                       "nl": "Goed voor de buurtwinkel. Maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn (* betekent verplicht, ? geen description). Houdt elk schema al zijn eigen keys, zoals `enum` en `maximum`?"},
            "correct": {"en": "Your `openapi_tools` turns the description of a web API into tools a model can call: an operation becomes a tool, its parameters the tool's parameters, with their schemas, their descriptions and which are required. Right for all {n} cases of the check.",
                        "nl": "Je `openapi_tools` maakt van de beschrijving van een web-API tools die een model kan aanroepen: een operation wordt een tool, zijn parameters de parameters van de tool, met hun schema's, hun descriptions en welke verplicht zijn. Goed bij alle {n} gevallen van de check."},
        },
    ),
}

REQUIRED = ["front", "openapi", "which-extension"]

CASES = [
    ("front", START_FRONT, "empty", None),
    ("front", "", "empty", None),
    ("front", SOLUTION_FRONT, "correct", None),
    ("front", _SPLIT_ALL, "wrong", "split-all"),
    ("front", _ALL_LINES, "wrong", "all-lines"),
    ("front", _NO_STRIP, "wrong", "no-strip"),
    ("front", SOLUTION_FRONT.replace('key, value = line.split(": ", 1)', 'key, _, value = line.partition(":")'), "correct", None),
    ("front", SOLUTION_FRONT.replace('if ": " in line:\n            key, value = line.split(": ", 1)',
                                     'if ":" in line:\n            key, value = line.split(":", 1)'), "correct", None),
    ("openapi", START_OPENAPI, "empty", None),
    ("openapi", "", "empty", None),
    ("openapi", SOLUTION_OPENAPI, "correct", None),
    ("openapi", _NO_GET, "error", "no-get"),
    ("openapi", _REQUIRED_KEY, "error", "required-key"),
    ("openapi", _NO_REQUIRED, "wrong", "no-required"),
    ("openapi", _ALL_REQUIRED, "wrong", "all-required"),
    ("openapi", _NO_DESCRIPTION, "wrong", "no-description"),
    ("openapi", SOLUTION_OPENAPI.replace('                schema = dict(p["schema"])\n                schema["description"] = p["description"]\n',
                                         '                schema = {**p["schema"], "description": p["description"]}\n'), "correct", None),
]


# --- the demo: Qwen3 with the six skills and one of the learner's own ----------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


DEMO_SKILL = """---
name: bread-rising
description: How long each dough of this kitchen rises. Use when the user asks how long to let dough rise.
---

# Letting dough rise

| dough | first rise | second rise |
|---|---|---|
| white bread | 60 minutes | 45 minutes |
| pizza | 90 minutes | none |
| cinnamon buns | 60 minutes | 30 minutes |

Let the dough rise in a warm place, under a damp cloth.
"""
DEMO_QUESTION = "How long should pizza dough rise?"
NAME_RULE = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def learner_fn(key: str, text):
    """The learner's function from an exercise, when it is right; else None."""
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text, {}).names.get(ex.function)


def own_skill(front_fn, text: str):
    """The learner's skill read with their front: (name, frontmatter, instructions), or (None, the problem's key, None)."""
    try:
        found = front_fn(text)
    except Exception:   # noqa: BLE001 - the learner's code, on the learner's text
        return None, "unreadable", None
    if not text.startswith("---\n") or not isinstance(found, dict):
        return None, "unreadable", None
    name, about = found.get("name"), found.get("description")
    if not isinstance(name, str) or not isinstance(about, str) or not about.strip():
        return None, "no_name", None
    if not NAME_RULE.fullmatch(name) or len(name) > 64:
        return None, "bad_name", None
    if name in SKILLS:
        return None, "taken", None
    return name, found, split_skill(text)[1]


def _post(url: str, messages: list, tools: list) -> dict:
    import urllib.request
    body = {"model": serve.MODEL, "messages": messages, "tools": tools, "max_tokens": 160, "temperature": 0}
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())["choices"][0]["message"]


def _message(role: str, text: str) -> str:
    color = {"system": "var(--ea-ink-2)", "user": "var(--ea-ink)", "assistant": "var(--ea-pred-text)",
             "tool": "var(--ea-data-text)"}[role]
    return (f'<div style="border-left:3px solid {color};padding:0.15rem 0.5rem;margin:0.25rem 0">'
            f'<div style="{_UI};font-size:11px;color:var(--ea-ink-2)">{role}</div>{_code(text, color)}</div>')


def _first_lines(text: str, k: int = 4) -> str:
    rows = text.strip().split("\n")
    return "\n".join(rows[:k]) + ("\n…" if len(rows) > k else "")


def ask_with_skill(front_fn, text: str, question: str, words: dict) -> str:
    """The learner's question to Qwen3 with the six skills and the learner's own in the list: each call of load_skill,
    and the answer."""
    name, found, body = own_skill(front_fn, text)
    if name is None:
        return f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(words[found])}</p>'
    skills = {**SKILLS, name: (found, body)}
    tools = [load_tool(list(skills))]
    messages = asked(question, index_system([f for f, _ in skills.values()]))
    parts = [_label(words["line"]) + _pre(_code(index_line(found), "var(--ea-ink)"))]
    try:
        running = serve.model_server()
        for _ in range(3):
            message = _post(running.url, messages, tools)
            calls = message.get("tool_calls") or []
            if not calls:
                break
            messages = messages + [message]
            for c in calls:
                try:
                    arguments = json.loads(c["function"]["arguments"])
                except ValueError:
                    arguments = None
                result = (load_skill(arguments, skills) if c["function"]["name"] == "load_skill" else
                          not_a_tool(c["function"]["name"], True, skills))
                parts.append(_message("assistant", f"{c['function']['name']}({c['function']['arguments']})"))
                parts.append(_message("tool", _first_lines(result)))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    except Exception as e:   # the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        raise
    if len(parts) == 1:
        parts.append(f'<p style="{_UI};font-size:13px;color:var(--ea-ink-2)">{html.escape(words["no_load"])}</p>')
    parts.append(_message("assistant", message.get("content") or ""))
    return yours.figure(words["title"], "".join(parts), alt=words["title"])


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"].get("build-agent-extensions", {})


def steps(kind: str, i: int) -> list:
    """The requests of question i: without skills ("none"), with the list and load_skill ("index"), or with every skill
    in the system prompt ("all")."""
    a = course()[f"{kind}-{i}"]["answers"][0]
    return a["steps"] if "steps" in a else [a]


def final(kind: str, i: int) -> str:
    return steps(kind, i)[-1]["text"]


def calls(i: int, kind: str = "index") -> list:
    """Every tool call of the run of question i with the list of skills, with the plain errors ("index") or the helpful
    ones ("helpful")."""
    return [c for st in steps(kind, i) for c in st["tool_calls"]]


def loaded(i: int, kind: str = "index") -> list:
    """The skills the run of question i loaded with load_skill."""
    out = []
    for c in calls(i, kind):
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            continue
        if c["function"]["name"] == "load_skill" and isinstance(arguments, dict) and arguments.get("name") in SKILLS:
            out.append(arguments["name"])
    return out


def findings() -> dict:
    n = len(QUESTIONS)
    out = {"questions": n, "skills": len(NAMES)}
    for kind in ("none", "index", "all", "helpful"):
        out[kind] = [i for i in range(n) if right(i, final(kind, i))]
        out[kind + "_prompt"] = round(sum(steps(kind, i)[0]["prompt_tokens"] for i in range(n)) / n)
    for kind in ("index", "helpful"):
        out[kind + "_loaded"] = [i for i in range(n) if QUESTIONS[i][1] in loaded(i, kind)]
        out[kind + "_loaded_other"] = [i for i in range(n) if loaded(i, kind) and QUESTIONS[i][1] not in loaded(i, kind)]
        out[kind + "_as_tool"] = [i for i in range(n) if any(c["function"]["name"] in SKILLS for c in calls(i, kind))]
        out[kind + "_no_call"] = [i for i in range(n) if not calls(i, kind)]
        out[kind + "_loaded_right"] = [i for i in out[kind + "_loaded"] if i in out[kind]]
        out[kind + "_total"] = round(sum(sum(st["prompt_tokens"] for st in steps(kind, i)) for i in range(n)) / n)
    out["yes_no"] = [i for i, (q, _, _) in enumerate(QUESTIONS) if q.endswith("Answer yes or no first.")]
    out["all_yes"] = [i for i in out["yes_no"] if re.match(r"\W*yes\b", final("all", i).lower())]
    out["all_numbers"] = [i for i in out["all"] if i not in out["yes_no"]]
    out["helpful_stuck"] = [i for i in out["helpful_as_tool"] if not loaded(i, "helpful") and not final("helpful", i).strip()]
    out["helpful_steps"] = RECORD["helpful-0"]["steps"]
    return out


# --- the pictures of the idea and of the measurement ------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "What the agent always sees, and what it loads", "always": "always in the system prompt: a line per skill",
           "loads": "loaded only when a question needs it", "call": "the model calls load_skill(\"cup-measures\"), and gets:",
           "alt": "Left, the system prompt with one line for each of the six skills: its name and description. Right, the instructions of one skill, cup-measures, which the agent gets only after it calls load_skill with that name."},
    "nl": {"title": "Wat de agent altijd ziet, en wat hij laadt", "always": "altijd in de system prompt: een regel per skill",
           "loads": "pas geladen als een vraag het nodig heeft", "call": "het model roept load_skill(\"cup-measures\") aan, en krijgt:",
           "alt": "Links de system prompt met één regel voor elk van de zes skills: de naam en de beschrijving. Rechts de instructies van één skill, cup-measures, die de agent pas krijgt nadat hij load_skill met die naam aanroept."},
}


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    lines = "\n".join(index_line(SKILLS[n][0]) for n in NAMES)
    body = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:0.8rem">'
            f'<div style="min-width:0">{_label(w["always"])}{_pre(_code(lines, "var(--ea-ink)"))}</div>'
            f'<div style="min-width:0">{_label(w["loads"])}<div style="{_UI};font-size:12px;color:var(--ea-pred-text);margin:0.2rem 0">'
            f'{html.escape(w["call"])}</div>{_pre(_code(SKILLS["cup-measures"][1], "var(--ea-data-text)"))}</div></div>')
    return yours.figure(w["title"], body, alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "Twelve kitchen questions: without skills, with the list of skills, with the list and a helpful error, and with all of them",
           "question": "question", "skill": "needs", "none": "no skills", "index": "list + load_skill",
           "helpful": "list, helpful error", "all": "all in the prompt",
           "right": "right", "wrong": "not right", "loaded": "loaded {s}", "as_tool": "called {s} as a tool",
           "tool_then": "called {s} as a tool, then loaded it", "no_call": "no call",
           "total": "{k} of {n}", "tokens": "{k} prompt tokens"},
    "nl": {"title": "Twaalf vragen over de keuken: zonder skills, met de lijst van skills, met de lijst en een behulpzame fout, en met alle skills",
           "question": "vraag", "skill": "nodig", "none": "geen skills", "index": "lijst + load_skill",
           "helpful": "lijst, behulpzame fout", "all": "alles in de prompt",
           "right": "goed", "wrong": "niet goed", "loaded": "laadde {s}", "as_tool": "riep {s} aan als tool",
           "tool_then": "riep {s} aan als tool, en laadde hem toen", "no_call": "geen call",
           "total": "{k} van de {n}", "tokens": "{k} tokens aan prompt"},
}


def _mark(ok: bool, w: dict) -> str:
    color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
    return f'<span style="{_UI};font-size:12px;color:{color};font-weight:{400 if ok else 600}">{w["right"] if ok else w["wrong"]}</span>'


def _what_it_did(i: int, w: dict, kind: str = "index") -> str:
    names = loaded(i, kind)
    tools = [c["function"]["name"] for c in calls(i, kind) if c["function"]["name"] in SKILLS]
    if names and tools == names[:1]:
        return w["tool_then"].format(s=names[0])
    if names:
        return w["loaded"].format(s=", ".join(dict.fromkeys(names)))
    if tools:
        return w["as_tool"].format(s=", ".join(dict.fromkeys(tools)))
    return w["no_call"] if not calls(i, kind) else ", ".join(dict.fromkeys(c["function"]["name"] for c in calls(i, kind)))


def table_picture(lang: str) -> str:
    w, f = _TABLE_WORDS[lang], findings()
    rows = []
    for i, (q, skill, _) in enumerate(QUESTIONS):
        did = {k: f'<div style="{_UI};font-size:11px;color:var(--ea-ink-2)">{html.escape(_what_it_did(i, w, k))}</div>'
               for k in ("index", "helpful")}
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(q)}</td><td style="{_CELL}">{_code(skill, "var(--ea-ink-2)")}</td>'
                    f'<td style="{_CELL}">{_mark(i in f["none"], w)}</td><td style="{_CELL}">{_mark(i in f["index"], w)}{did["index"]}</td>'
                    f'<td style="{_CELL}">{_mark(i in f["helpful"], w)}{did["helpful"]}</td>'
                    f'<td style="{_CELL}">{_mark(i in f["all"], w)}</td></tr>')
    n = f["questions"]
    foot = (f'<tr><td style="{_CELL}"></td><td style="{_CELL}"></td>'
            + "".join(f'<td style="{_CELL};{_UI};font-weight:600">{w["total"].format(k=len(f[k]), n=n)}<br>'
                      f'<span style="font-weight:400;color:var(--ea-ink-2)">{w["tokens"].format(k=number(f[k + "_prompt"], lang))}</span></td>'
                      for k in ("none", "index", "helpful", "all")) + "</tr>")
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{w[h]}</th>' for h in ("question", "skill", "none", "index", "helpful", "all"))
             + f'</tr></thead><tbody>{"".join(rows)}{foot}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


# --- the texts under the table and of the worked example -----------------------------------------------------------------

def _of(k: int, n: int, lang: str) -> str:
    return f"{k} of the {n}" if lang == "en" else f"{k} van de {n}"


def first_tool_call(i: int, kind: str) -> str:
    """The first call of a skill's name as a tool in a run, as the model wrote it: such as cup-measures({"flour": "3"})."""
    c = next(c for c in calls(i, kind) if c["function"]["name"] in SKILLS)
    return f'{c["function"]["name"]}({c["function"]["arguments"]})'


def measured_text(lang: str) -> str:
    f = findings()
    n, k = f["questions"], f["skills"]
    example = first_tool_call(f["index_as_tool"][0], "index")
    loaded_wrong = len(f["helpful_loaded"]) - len(f["helpful_loaded_right"])
    if lang == "en":
        parts = [
            f"Without skills, Qwen3 answered {_of(len(f['none']), n, lang)} questions right: it does not know this kitchen's "
            f"numbers, so it guessed. With all {k} skills in the system prompt, it answered {len(f['all'])} right, and each request "
            f"carried {number(f['all_prompt'], lang)} prompt tokens instead of {number(f['none_prompt'], lang)}. Not all of that was "
            f"reading: to the {len(f['yes_no'])} questions that ask for yes or no first, it answered yes "
            + ("every time" if f["all_yes"] == f["yes_no"] else f"{len(f['all_yes'])} times") + ", also where the skill says no; "
            f"of the questions with a number, it got {len(f['all_numbers'])} of {n - len(f['yes_no'])} right.",
            f"With the list and `load_skill`, it did what the list did not ask: in {_of(len(f['index_as_tool']), n, lang)} questions "
            f"it called a skill's name as if it were a tool, such as `{example}`, and it never loaded a skill. It answered "
            f"{len(f['index'])} right. The error it got said only that there is no such tool.",
            f"With an error that says what to do, as in [Tools an agent can use well](/course/open/build-tool-design?lang=en), the "
            f"model loaded the right skill after that error in {len(f['helpful_loaded'])} of those {len(f['helpful_as_tool'])} "
            f"questions, and answered {len(f['helpful_loaded_right'])} of those {len(f['helpful_loaded'])} right; in the other "
            f"{loaded_wrong} it had the instructions and still worked out a wrong number. In the {len(f['helpful_stuck'])} questions "
            f"where it did not load the skill, it called the same skill as a tool again and again, until the step limit of "
            f"{f['helpful_steps']} requests. In all it answered {_of(len(f['helpful']), n, lang)} right. The extra steps cost: "
            f"{number(f['helpful_total'], lang)} prompt tokens per question in all, against {number(f['index_total'], lang)} with "
            f"the plain error.",
            f"Per request, the list costs less than all the skills: {number(f['index_prompt'], lang)} prompt tokens instead of "
            f"{number(f['all_prompt'], lang)}. Per question it cost more here, because of the extra requests: "
            f"{number(f['index_total'], lang)} prompt tokens with the plain error and {number(f['helpful_total'], lang)} with the "
            f"helpful one, against {number(f['all_prompt'], lang)}. Since a line is shorter than a skill's instructions, the list pays "
            f"off once there are many skills, and only with a model that loads a skill when a question needs it; Qwen3-0.6B did that "
            f"only after an error told it how.",
        ]
    else:
        parts = [
            f"Zonder skills beantwoordde Qwen3 {_of(len(f['none']), n, lang)} vragen goed: het kent de getallen van deze keuken "
            f"niet, dus het gokte. Met alle {k} skills in de system prompt beantwoordde het er {len(f['all'])} goed, en had elk "
            f"request {number(f['all_prompt'], lang)} tokens aan prompt in plaats van {number(f['none_prompt'], lang)}. Niet alles "
            f"daarvan was lezen: op de {len(f['yes_no'])} vragen die eerst om ja of nee vragen, antwoordde het "
            + ("elke keer" if f["all_yes"] == f["yes_no"] else f"{len(f['all_yes'])} keer") + " yes, ook waar de skill nee zegt; "
            f"van de vragen met een getal had het er {len(f['all_numbers'])} van de {n - len(f['yes_no'])} goed.",
            f"Met de lijst en `load_skill` deed het iets wat de lijst niet vroeg: bij {_of(len(f['index_as_tool']), n, lang)} vragen "
            f"riep het de naam van een skill aan alsof het een tool was, zoals `{example}`, en het laadde nooit een skill. Het "
            f"beantwoordde er {len(f['index'])} goed. De fout die het kreeg, zei alleen dat er geen tool met die naam is.",
            f"Met een fout die zegt wat het moet doen, zoals in [Tools die een agent goed kan gebruiken](/course/open/build-tool-design?lang=nl), "
            f"laadde het model na die fout de goede skill bij {len(f['helpful_loaded'])} van die {len(f['helpful_as_tool'])} vragen, "
            f"en beantwoordde het er daarvan {len(f['helpful_loaded_right'])} goed; bij de andere {loaded_wrong} had het de "
            f"instructies en rekende het toch een verkeerd getal uit. Bij de {len(f['helpful_stuck'])} vragen waarbij het de skill "
            f"niet laadde, riep het steeds weer dezelfde skill aan als tool, tot de stapgrens van {f['helpful_steps']} requests. In "
            f"totaal had het er {_of(len(f['helpful']), n, lang)} goed. De extra stappen kostten wat: "
            f"{number(f['helpful_total'], lang)} tokens aan prompt per vraag in totaal, tegen {number(f['index_total'], lang)} met "
            f"de kale fout.",
            f"Per request kost de lijst minder dan alle skills: {number(f['index_prompt'], lang)} tokens aan prompt in plaats van "
            f"{number(f['all_prompt'], lang)}. Per vraag kostte hij hier meer, door de extra requests: "
            f"{number(f['index_total'], lang)} tokens aan prompt met de kale fout en {number(f['helpful_total'], lang)} met de "
            f"behulpzame, tegen {number(f['all_prompt'], lang)}. Omdat een regel korter is dan de instructies van een skill, loont de "
            f"lijst pas bij veel skills, en alleen met een model dat een skill laadt als een vraag hem nodig heeft; Qwen3-0.6B deed "
            f"dat alleen nadat een fout zei hoe.",
        ]
    return "\n\n".join(parts)


def worked_question() -> int:
    """The first question that the run with the helpful error answered right after loading its skill."""
    return findings()["helpful_loaded_right"][0]


_WORKED_WORDS = {
    "en": {"title": "Question {k} with the list of skills: the plain error, and the helpful one", "index": "with the plain error",
           "helpful": "with the helpful error", "answer": "the answer", "right": "right", "wrong": "not right"},
    "nl": {"title": "Vraag {k} met de lijst van skills: de kale fout, en de behulpzame", "index": "met de kale fout",
           "helpful": "met de behulpzame fout", "answer": "het antwoord", "right": "goed", "wrong": "niet goed"},
}


def _run_html(i: int, kind: str, w: dict) -> str:
    parts = []
    for st in steps(kind, i):
        for c, result in zip(st["tool_calls"], st.get("results") or []):
            parts.append(_message("assistant", f'{c["function"]["name"]}({c["function"]["arguments"]})'))
            parts.append(_message("tool", _first_lines(result)))
    text = final(kind, i)
    parts.append(_message("assistant", text.strip()))
    parts.append(_mark(right(i, text), w))
    return "".join(parts)


def worked_picture(lang: str) -> str:
    w, i = _WORKED_WORDS[lang], worked_question()
    body = (f'{_label(QUESTIONS[i][0])}<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:0.8rem">'
            + "".join(f'<div style="min-width:0">{_label(w[kind])}{_run_html(i, kind, w)}</div>' for kind in ("index", "helpful"))
            + "</div>")
    return yours.figure(w["title"].format(k=i + 1), body, alt=w["title"].format(k=i + 1))


def worked_text(lang: str) -> str:
    i = worked_question()
    q, skill, _ = QUESTIONS[i]
    first = first_tool_call(i, "index")
    plain = next(r for st in steps("index", i) for r in st.get("results") or [])
    helpful = next(r for st in steps("helpful", i) for r in st.get("results") or [])
    if lang == "en":
        return (f"### Worked example\n\nTake question {i + 1}, *{q}* It needs the skill `{skill}`. In both runs, the model's first "
                f"call was `{first}`: the name of the skill, called as a tool. With the plain error, `{plain}`, it gave up and "
                f"answered without the skill. With the helpful error, `{helpful}`, its next call was "
                f"`load_skill({{\"name\": \"{skill}\"}})`; it got the table of the skill, and answered `{final('helpful', i).strip()}`, "
                f"which is right. The model is the same, and so is the list; only the error changed.")
    return (f"### Uitgewerkt voorbeeld\n\nNeem vraag {i + 1}, *{q}* Die heeft de skill `{skill}` nodig. In beide runs was de "
            f"eerste call van het model `{first}`: de naam van de skill, aangeroepen als tool. Met de kale fout, `{plain}`, gaf het "
            f"op en antwoordde het zonder de skill. Met de behulpzame fout, `{helpful}`, was de volgende call "
            f"`load_skill({{\"name\": \"{skill}\"}})`; het kreeg de tabel van de skill, en antwoordde `{final('helpful', i).strip()}`, "
            f"en dat is goed. Het model is hetzelfde, en de lijst ook; alleen de fout veranderde.")
