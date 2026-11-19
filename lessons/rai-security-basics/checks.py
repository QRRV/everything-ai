# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-security-basics: threats against the member file of the made-up library from the module
privacy, sorted with STRIDE, and an attack tree for reading that file, with Schneier's own tree for opening a safe
(1999) beside it. The three functions the learner writes: at_risk (how many threats break each security property),
cost (what the cheapest attack under a node of a tree costs) and cheapest (its steps); with the pictures of what they
give, and the demo that adds defenses to the tree. All threats, steps and costs of the library are made up."""
import copy
import html
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit


class Named(list):
    """A list the check calls by the name the lesson gives it, such as THREATS, so that a message stays short. The
    learner's code gets its own copy, a plain list."""

    def __init__(self, values, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(list(self), memo)


class Node(dict):
    """A threat, or a node of an attack tree, which the check calls by its name in the lesson, such as TREE or
    LOGIN["children"][0]. The learner's code gets its own copy, made of plain dicts and lists."""

    def __init__(self, values: dict, name: str):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


# --- STRIDE and the properties it breaks ---------------------------------------------------------------------------------

# the six kinds of threat of STRIDE, each with the property that guards against it, as in the MSDN article of 2006
# (Hernan, Lambert, Ostwald and Shostack, Figure 3)
PROPERTY = {"spoofing": "authentication", "tampering": "integrity", "repudiation": "non-repudiation",
            "information disclosure": "confidentiality", "denial of service": "availability",
            "elevation of privilege": "authorization"}
ORDER = ["confidentiality", "integrity", "availability", "authentication", "non-repudiation", "authorization"]
CIA = ORDER[:3]

# threats against the made-up library, each with its kind of STRIDE threat
_THREATS = [
    ("Someone logs in to the staff portal with the password of Daan Smit", "spoofing"),
    ("A member changes the return dates of their own loans in the database", "tampering"),
    ("A staff member waives a fine and later says it was someone else", "repudiation"),
    ("The backup of October 2026 lies unencrypted in a shared folder", "information disclosure"),
    ("A flood of requests keeps the catalog website down for a day", "denial of service"),
    ("A volunteer's desk account can open the files of the data team", "elevation of privilege"),
    ("Someone changes the opening hours on the website", "tampering"),
    ("The survey of 2024 is e-mailed to the wrong address", "information disclosure"),
]
THREATS = Named([Node({"what": what, "stride": kind}, f"THREATS[{i}]") for i, (what, kind) in enumerate(_THREATS)],
                "THREATS")


# --- the attack trees ----------------------------------------------------------------------------------------------------

def _leaf(name: str, cost: int, possible: bool) -> dict:
    return {"name": name, "kind": "leaf", "cost": cost, "possible": possible}


def _node(name: str, kind: str, children: list) -> dict:
    return {"name": name, "kind": kind, "children": children}


# the attack tree for reading the member file of the made-up library: costs in euros, what the attacker would spend;
# "possible": whether an attacker from outside, with little money, could take that step
_TREE = _node("Read the member file", "or", [
    _node("Log in as a staff member", "and", [
        _node("Get a password", "or", [
            _leaf("Guess the password", 100, True),
            _leaf("Phish the password", 300, True),
            _leaf("Watch it being typed at the desk", 150, True),
        ]),
        _node("Get the login code", "or", [
            _leaf("Phish the code too", 400, True),
            _leaf("Steal the phone", 1000, True),
        ]),
    ]),
    _node("Steal the backup", "and", [
        _leaf("Get into the IT room", 50, True),
        _leaf("Break the encryption", 20000, False),
    ]),
    _leaf("Inject SQL in the catalog search", 600, True),
    _leaf("Bribe a staff member", 2000, False),
])

# Schneier's tree for opening a safe (1999, Figure 4, costs in thousands of dollars; "possible" as in Figure 1)
_SAFE = _node("Open Safe", "or", [
    _leaf("Pick Lock", 30, False),
    _node("Learn Combo", "or", [
        _leaf("Find Written Combo", 75, False),
        _node("Get Combo From Target", "or", [
            _leaf("Threaten", 60, False),
            _leaf("Blackmail", 100, False),
            _node("Eavesdrop", "and", [
                _leaf("Listen to Conversation", 20, True),
                _leaf("Get Target to State Combo", 40, False),
            ]),
            _leaf("Bribe", 20, True),
        ]),
    ]),
    _leaf("Cut Open Safe", 10, True),
    _leaf("Install Improperly", 100, False),
])


def _named(tree: dict, name: str, aliases: dict) -> Node:
    """The tree as Nodes, each named by its path from the nearest node the lesson gives a name, such as
    LOGIN["children"][0]."""
    name = aliases.get(name, name)
    values = dict(tree)
    if "children" in tree:
        values["children"] = [_named(child, f'{name}["children"][{i}]', aliases)
                              for i, child in enumerate(tree["children"])]
    return Node(values, name)


_ALIASES = {'TREE["children"][0]': "LOGIN", 'LOGIN["children"][0]': "PASSWORD", 'LOGIN["children"][1]': "CODE",
            'TREE["children"][1]': "BACKUP"}
TREE = _named(_TREE, "TREE", _ALIASES)
LOGIN = TREE["children"][0]
PASSWORD = LOGIN["children"][0]
CODE = LOGIN["children"][1]
BACKUP = TREE["children"][1]
SQL = TREE["children"][2]
SAFE = _named(_SAFE, "SAFE", {})


def nodes(tree: dict, depth: int = 0) -> list:
    """Every node of a tree with its depth, top to bottom."""
    found = [(tree, depth)]
    for child in tree.get("children", []):
        found += nodes(child, depth + 1)
    return found


def leaves(tree: dict) -> list:
    return [n for n, _ in nodes(tree) if n["kind"] == "leaf"]


# --- the course's own functions -----------------------------------------------------------------------------------------

_A_HEAD = ("def at_risk(threats):\n"
           "    # PROPERTY gives, for each kind of STRIDE threat, the property it breaks\n"
           "    counts = {}\n"
           "    for threat in threats:\n")
_A_START = _A_HEAD + "        ...\n    return counts\n"
_A = _A_HEAD + ("        prop = PROPERTY[threat[\"stride\"]]\n"
                "        if prop in counts:\n"
                "            counts[prop] = counts[prop] + 1\n"
                "        else:\n"
                "            counts[prop] = 1\n") + "    return counts\n"
at_risk = run_code(_A, {"PROPERTY": PROPERTY}).names["at_risk"]

_C_HEAD = ("def cost(node):\n"
           "    # a leaf has a \"cost\"; an \"and\" or \"or\" node has a list of \"children\"\n"
           "    if node[\"kind\"] == \"leaf\":\n"
           "        return node[\"cost\"]\n")
_C_START = _C_HEAD + "    ...\n"
_C = _C_HEAD + ("    total = 0\n"
                "    best = cost(node[\"children\"][0])\n"
                "    for child in node[\"children\"]:\n"
                "        total = total + cost(child)\n"
                "        if cost(child) < best:\n"
                "            best = cost(child)\n"
                "    if node[\"kind\"] == \"and\":\n"
                "        return total\n"
                "    return best\n")
cost = run_code(_C).names["cost"]

_P_HEAD = ("def cheapest(node):\n"
           "    # cost(node) is ready to use; give back the names of the steps of the cheapest attack\n"
           "    if node[\"kind\"] == \"leaf\":\n"
           "        return [node[\"name\"]]\n"
           "    if node[\"kind\"] == \"and\":\n"
           "        steps = []\n"
           "        for child in node[\"children\"]:\n"
           "            for step in cheapest(child):\n"
           "                steps.append(step)\n"
           "        return steps\n")
_P_START = _P_HEAD + "    ...\n"
_P = _P_HEAD + ("    best = node[\"children\"][0]\n"
                "    for child in node[\"children\"]:\n"
                "        if cost(child) < cost(best):\n"
                "            best = child\n"
                "    return cheapest(best)\n")
cheapest = run_code(_P, {"cost": cost}).names["cheapest"]


def possible(node) -> bool:
    """Schneier's rule for possible and impossible: an OR node is possible when one of its children is, an AND node
    when all of them are."""
    if node["kind"] == "leaf":
        return node["possible"]
    count = sum(possible(child) for child in node["children"])
    return count == len(node["children"]) if node["kind"] == "and" else count > 0


# --- the defenses of the demo -------------------------------------------------------------------------------------------

# each defense makes steps more expensive, or (None) takes a step away
DEFENSES = {
    "long": {"Guess the password": 5000},
    "training": {"Phish the password": 1500, "Phish the code too": 1500},
    "placeholders": {"Inject SQL in the catalog search": None},
    "screens": {"Watch it being typed at the desk": 1200},
}
DEFENSE_LABELS = {
    "en": {"Long passwords": "long", "Phishing training": "training",
           "Placeholders in the search (next lesson)": "placeholders", "Screens turned away at the desk": "screens"},
    "nl": {"Lange wachtwoorden": "long", "Training tegen phishing": "training",
           "Placeholders in de zoekfunctie (volgende les)": "placeholders", "Schermen weggedraaid bij de balie": "screens"},
}


def defended(keys) -> dict:
    """The library's tree, as plain dicts, with the chosen defenses."""
    changes = {}
    for key in keys or []:
        changes.update(DEFENSES[key])

    def apply(node):
        if node["kind"] == "leaf":
            out = dict(node)
            if node["name"] in changes:
                if changes[node["name"]] is None:
                    return None
                out["cost"] = changes[node["name"]]
            return out
        children = [c for c in (apply(child) for child in node["children"]) if c is not None]
        return {"name": node["name"], "kind": node["kind"], "children": children}

    return apply(copy.deepcopy(dict(TREE)))


# --- formatting -------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CODE = "font-family:var(--ea-font-code);font-size:0.8rem"
_LABEL = "font-family:var(--ea-font-code);font-size:0.72rem;color:var(--ea-ink-2)"


def _note(parts: list) -> str:
    return f'<div style="{_UI}">{html.escape(" ".join(p for p in parts if p))}</div>'


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _shown(v) -> str:
    """A value a function gave back, the way Python shows it, kept short."""
    try:
        text = repr(v)
    except Exception:
        text = "?"
    return text if len(text) <= 60 else text[:57] + "..."


_STOPPED = object()   # what a function gave back when it stopped with an error


def _call(fn, *args):
    try:
        return with_time_limit(lambda: fn(*copy.deepcopy(args)))
    except Exception:
        return _STOPPED


def _plain(node) -> dict:
    return copy.deepcopy(dict(node))


def euros(n, lang: str) -> str:
    """€500, €1,150 in English; €500, €1150 and €20.000 in Dutch."""
    if not isinstance(n, int) or isinstance(n, bool):
        return _shown(n)
    if lang == "nl":
        return "€" + (f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n))
    return f"€{n:,}"


def _frame(wrong: bool) -> str:
    return "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid transparent"


def _square(filled: bool, wrong: bool = False) -> str:
    fill = "background:var(--ea-ink)" if filled else "background:transparent"
    edge = "border:1.5px dashed var(--ea-loss)" if wrong else "border:1px solid var(--ea-line)"
    return f'<div style="width:0.9rem;height:0.9rem;border-radius:2px;{fill};{edge}"></div>'


_ROW = ("display:flex;gap:0.5rem;align-items:center;padding:0.15rem 0.4rem;margin:0.1rem 0;border-radius:4px;"
        "font-family:var(--ea-font-ui);font-size:0.85rem;")


def _tree_row(node: dict, depth: int, mark: str, right: str, wrong: bool, faded: bool = False) -> str:
    """One node of a tree as a row: an optional mark (a square), the name indented by its depth with OR or AND for an
    inner node, and on the right a cost or so."""
    kind = "" if node["kind"] == "leaf" else (f'<span style="{_LABEL};margin-left:0.4rem">'
                                              f'{node["kind"].upper()}</span>')
    color = "color:var(--ea-ink-3);" if faded else "color:var(--ea-ink);"
    weight = "font-weight:600;" if node["kind"] != "leaf" else ""
    return (f'<div style="{_ROW}{color}{_frame(wrong)}">'
            f'<div style="flex:0 0 1rem;display:flex;justify-content:center">{mark}</div>'
            f'<div style="flex:1;min-width:0;padding-left:{depth * 1.1:.1f}rem;overflow-wrap:anywhere;{weight}">'
            f'{html.escape(node["name"])}{kind}</div>'
            f'<div style="flex:0 0 auto;text-align:right;font-variant-numeric:tabular-nums">{right}</div></div>')


def _tree_box(rows: list) -> str:
    return f'<div style="max-width:30rem;margin-top:0.3rem">{"".join(rows)}</div>'


# --- the picture in The idea: the library's tree ------------------------------------------------------------------------

_TREE_WORDS = {
    "en": {"title": "The attack tree for the member file",
           "said": "OR: one of the children is enough. AND: every child is needed. Beside each step, what it would cost "
                   "the attacker, in euros; the costs are made up."},
    "nl": {"title": "De aanvalsboom voor het ledenbestand",
           "said": "OR: een van de kinderen is genoeg. AND: elk kind is nodig. Naast elke stap wat die de aanvaller zou "
                   "kosten, in euro's; de bedragen zijn verzonnen."},
}


def tree_picture(lang: str) -> str:
    w = _TREE_WORDS[lang]
    rows = [_tree_row(node, depth, "", euros(node["cost"], lang) if node["kind"] == "leaf" else "", False)
            for node, depth in nodes(TREE)]
    return yours.figure(w["title"], _tree_box(rows) + _note([w["said"]]))


# --- the picture under exercise 1: at_risk -----------------------------------------------------------------------------

_RISK_WORDS = {
    "en": {"title": "Your at_risk: which properties the threats against the library break",
           "said": "One bar per property, as long as the count your at_risk gives for THREATS. The first three are the "
                   "CIA triad; STRIDE adds the other three.",
           "fine": "The same counts as the course's at_risk.",
           "wrong": "A dashed purple frame marks a property where it differs from the course's at_risk.",
           "extra": "It also counts keys that are not properties: {keys}.",
           "off": "Your at_risk gives {got}; the picture draws a dict of counts.",
           "stops": "Your at_risk stops with an error on THREATS: the check above says where."},
    "nl": {"title": "Jouw at_risk: welke eigenschappen de dreigingen tegen de bibliotheek breken",
           "said": "Eén balk per eigenschap, zo lang als het aantal dat jouw at_risk geeft voor THREATS. De eerste drie "
                   "zijn de CIA-triade; STRIDE voegt de andere drie toe.",
           "fine": "Dezelfde aantallen als bij de at_risk van de course.",
           "wrong": "Een gestreept paars kader wijst een eigenschap aan waar hij verschilt van de at_risk van de course.",
           "extra": "Hij telt ook keys die geen eigenschap zijn: {keys}.",
           "off": "Jouw at_risk geeft {got}; het plaatje tekent een dict met aantallen.",
           "stops": "Jouw at_risk stopt met een fout bij THREATS: de check hierboven zegt waar."},
}


def risk_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _RISK_WORDS[seen.lang]
    got = _call(fn, list(_plain(t) for t in THREATS))
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, dict) or not all(isinstance(v, int) and not isinstance(v, bool) for v in got.values()):
        return yours.figure(w["title"], _note([w["off"].format(got=_shown(got))]))
    want = at_risk([_plain(t) for t in THREATS])
    top = max([v for v in got.values()] + [1])
    rows, wrong_any = [], False
    for prop in ORDER:
        n = got.get(prop, 0)
        wrong = n != want.get(prop, 0)
        wrong_any |= wrong
        color = "var(--ea-ink)" if prop in CIA else "var(--ea-ink-3)"
        width = max(0.0, min(n, top)) / top * 100
        rows.append(f'<div style="display:flex;gap:0.5rem;align-items:center;padding:0.15rem 0.3rem;margin:0.1rem 0;'
                    f'border-radius:4px;{_frame(wrong)}">'
                    f'<div style="flex:0 0 8.5rem;{_LABEL}">{html.escape(prop)}</div>'
                    f'<div style="flex:1;min-width:0"><div style="height:0.7rem;width:{width:.1f}%;background:{color};'
                    f'border-radius:2px"></div></div>'
                    f'<div style="flex:0 0 1.5rem;text-align:right;font-family:var(--ea-font-ui);font-size:0.85rem;'
                    f'color:var(--ea-ink)">{n}</div></div>')
    extra = [k for k in got if k not in ORDER]
    notes = [w["said"], w["wrong"] if wrong_any else w["fine"]]
    if extra:
        notes.append(w["extra"].format(keys=", ".join(_shown(k) for k in extra[:4]) + (", …" if len(extra) > 4 else "")))
    return yours.figure(w["title"], f'<div style="max-width:28rem">{"".join(rows)}</div>' + _note(notes))


# --- the picture under exercise 2: cost --------------------------------------------------------------------------------

_COST_WORDS = {
    "en": {"title": "Your cost: what the cheapest attack under each node costs",
           "said": "Each row is a node of the library's tree, with what your cost gives for it. The goal at the top: "
                   "{goal}.",
           "fine": "The same costs as the course's cost.",
           "wrong": "A dashed purple frame marks a node where it differs from the course's cost.",
           "off": "Your cost gives {got} for {call}; the picture draws a number of euros.",
           "stops": "Your cost stops with an error on {call}: the check above says where."},
    "nl": {"title": "Jouw cost: wat de goedkoopste aanval onder elke node kost",
           "said": "Elke rij is een node van de boom van de bibliotheek, met wat jouw cost ervoor geeft. Het doel "
                   "bovenaan: {goal}.",
           "fine": "Dezelfde kosten als bij de cost van de course.",
           "wrong": "Een gestreept paars kader wijst een node aan waar hij verschilt van de cost van de course.",
           "off": "Jouw cost geeft {got} voor {call}; het plaatje tekent een bedrag in euro's.",
           "stops": "Jouw cost stopt met een fout bij {call}: de check hierboven zegt waar."},
}


def cost_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _COST_WORDS[seen.lang]
    rows, wrong_any, goal = [], False, None
    for node, depth in nodes(TREE):
        got = _call(fn, _plain(node))
        call = f"cost({node!r})"
        if got is _STOPPED:
            return yours.figure(w["title"], _note([w["stops"].format(call=call)]))
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            return yours.figure(w["title"], _note([w["off"].format(got=_shown(got), call=call)]))
        wrong = got != cost(_plain(node))
        wrong_any |= wrong
        goal = got if goal is None else goal
        rows.append(_tree_row(node, depth, "", euros(got, seen.lang), wrong))
    notes = [w["said"].format(goal=euros(goal, seen.lang)), w["wrong"] if wrong_any else w["fine"]]
    return yours.figure(w["title"], _tree_box(rows) + _note(notes))


# --- the picture under exercise 3: cheapest ------------------------------------------------------------------------------

_PATH_WORDS = {
    "en": {"title": "Your cheapest: the steps of the cheapest attack on the member file",
           "said": "Filled: the {k} steps your cheapest gives for TREE, together {total}. Beside each step, its cost.",
           "said1": "Filled: the one step your cheapest gives for TREE, {total}. Beside each step, its cost.",
           "fine": "The same steps as the course's cheapest.",
           "wrong": "A dashed purple frame marks a step where it differs from the course's cheapest.",
           "unknown": "It also gives names that are no step of the tree: {names}.",
           "off": "Your cheapest gives {got}; the picture draws a list of step names.",
           "stops": "Your cheapest stops with an error on TREE: the check above says where."},
    "nl": {"title": "Jouw cheapest: de stappen van de goedkoopste aanval op het ledenbestand",
           "said": "Gevuld: de {k} stappen die jouw cheapest geeft voor TREE, samen {total}. Naast elke stap wat die "
                   "kost.",
           "said1": "Gevuld: de ene stap die jouw cheapest geeft voor TREE, {total}. Naast elke stap wat die kost.",
           "fine": "Dezelfde stappen als bij de cheapest van de course.",
           "wrong": "Een gestreept paars kader wijst een stap aan waar hij verschilt van de cheapest van de course.",
           "unknown": "Hij geeft ook namen die geen stap van de boom zijn: {names}.",
           "off": "Jouw cheapest geeft {got}; het plaatje tekent een lijst namen van stappen.",
           "stops": "Jouw cheapest stopt met een fout bij TREE: de check hierboven zegt waar."},
}


def _path_rows(tree: dict, steps: list, want: list | None, lang: str) -> tuple:
    rows, wrong_any = [], False
    for node, depth in nodes(tree):
        if node["kind"] == "leaf":
            on = node["name"] in steps
            wrong = want is not None and on != (node["name"] in want)
            wrong_any |= wrong
            rows.append(_tree_row(node, depth, _square(on), euros(node["cost"], lang), wrong, faded=not on))
        else:
            rows.append(_tree_row(node, depth, "", "", False))
    return rows, wrong_any


def path_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PATH_WORDS[seen.lang]
    got = _call(fn, _plain(TREE))
    if got is _STOPPED:
        return yours.figure(w["title"], _note([w["stops"]]))
    if not isinstance(got, list):
        return yours.figure(w["title"], _note([w["off"].format(got=_shown(got))]))
    known = {n["name"]: n["cost"] for n in leaves(TREE)}
    rows, wrong_any = _path_rows(TREE, got, cheapest(_plain(TREE)), seen.lang)
    on = [name for name in known if name in got]
    total = euros(sum(known[name] for name in on), seen.lang)
    said = (w["said1"] if len(on) == 1 else w["said"]).format(k=len(on), total=total)
    notes = [said, w["wrong"] if wrong_any else w["fine"]]
    unknown = [s for s in got if s not in known]
    if unknown:
        notes.append(w["unknown"].format(names=", ".join(_shown(s) for s in unknown[:3])))
    return yours.figure(w["title"], _tree_box(rows) + _note(notes))


# --- the demo: defenses --------------------------------------------------------------------------------------------------

_DEMO_WORDS = {
    "en": {"title": "The cheapest attack, with {who}",
           "yours": "your cost and cheapest", "course": "the course's cost and cheapest", "mixed": "your {mine} and the "
           "course's {other}",
           "said": "The cheapest attack now costs {total}: {steps}.",
           "then": ", then ",
           "none": "No defense chosen: this is the tree of the exercises.",
           "chosen": "Defenses: {names}.",
           "off": "{who} gives {got} here; the picture needs a number and a list of step names.",
           "stops": "{who} stops with an error on this tree."},
    "nl": {"title": "De goedkoopste aanval, met {who}",
           "yours": "jouw cost en cheapest", "course": "de cost en cheapest van de course", "mixed": "jouw {mine} en de "
           "{other} van de course",
           "said": "De goedkoopste aanval kost nu {total}: {steps}.",
           "then": ", dan ",
           "none": "Geen verdediging gekozen: dit is de boom van de oefeningen.",
           "chosen": "Verdedigingen: {names}.",
           "off": "{who} geeft hier {got}; het plaatje heeft een getal en een lijst namen van stappen nodig.",
           "stops": "{who} stopt met een fout bij deze boom."},
}


def mine(box_text, correct: bool, function: str, names: dict | None = None):
    """The learner's own function when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run = run_code(box_text, dict(names or {}))
    fn = run.names.get(function) if run.error is None else None
    return fn if callable(fn) else None


def defense_picture(keys, my_cost, my_cheapest, lang: str) -> str:
    """The library's tree with the chosen defenses: what each node costs, and the steps of the cheapest attack, with
    the learner's cost and cheapest where they are right (`my_cost`, `my_cheapest`; None for the course's)."""
    w = _DEMO_WORDS[lang]
    if my_cost is not None and my_cheapest is not None:
        who = w["yours"]
    elif my_cost is None and my_cheapest is None:
        who = w["course"]
    elif my_cost is not None:
        who = w["mixed"].format(mine="cost", other="cheapest")
    else:
        who = w["mixed"].format(mine="cheapest", other="cost")
    fn_cost, fn_path = my_cost or cost, my_cheapest or cheapest
    tree = defended(keys)
    title = w["title"].format(who=who)
    total, steps = _call(fn_cost, tree), _call(fn_path, tree)
    if total is _STOPPED or steps is _STOPPED:
        return yours.figure(title, _note([w["stops"].format(who=who[0].upper() + who[1:])]))
    if not isinstance(total, (int, float)) or isinstance(total, bool) or not isinstance(steps, list):
        shown = _shown(total) if not isinstance(total, (int, float)) else _shown(steps)
        return yours.figure(title, _note([w["off"].format(who=who[0].upper() + who[1:], got=shown)]))
    rows = []
    for node, depth in nodes(tree):
        here = _call(fn_cost, node)
        if node["kind"] == "leaf":
            on = node["name"] in steps
            rows.append(_tree_row(node, depth, _square(on), euros(here, lang), False, faded=not on))
        else:
            rows.append(_tree_row(node, depth, "", euros(here, lang), False))
    labels = {key: label for label, key in DEFENSE_LABELS[lang].items()}
    chosen = [labels[k] for k in DEFENSES if k in (keys or [])]
    notes = [w["said"].format(total=euros(total, lang), steps=w["then"].join(steps)),
             w["chosen"].format(names=", ".join(chosen)) if chosen else w["none"]]
    return yours.figure(title, _tree_box(rows) + _note(notes))


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["at-risk", "cost", "cheapest", "integrity", "and-node"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: at_risk
_A_KIND = _A_HEAD + ("        prop = threat[\"stride\"]\n"
                     "        if prop in counts:\n"
                     "            counts[prop] = counts[prop] + 1\n"
                     "        else:\n"
                     "            counts[prop] = 1\n") + "    return counts\n"
_A_ONE = _A_HEAD + "        prop = PROPERTY[threat[\"stride\"]]\n        counts[prop] = 1\n" + "    return counts\n"
_A_KEY = _A_HEAD + ("        prop = PROPERTY[threat[\"stride\"]]\n"
                    "        counts[prop] = counts[prop] + 1\n") + "    return counts\n"
_A_INSIDE = _A_HEAD + ("        prop = PROPERTY[threat[\"stride\"]]\n"
                       "        if prop in counts:\n"
                       "            counts[prop] = counts[prop] + 1\n"
                       "        else:\n"
                       "            counts[prop] = 1\n") + "        return counts\n"

# exercise 2: cost
_C_LOOP = ("    total = 0\n"
           "    best = cost(node[\"children\"][0])\n"
           "    for child in node[\"children\"]:\n"
           "        total = total + cost(child)\n"
           "        if cost(child) < best:\n"
           "            best = cost(child)\n")
_C_SUM = _C_HEAD + _C_LOOP + "    return total\n"
_C_MIN = _C_HEAD + _C_LOOP + "    return best\n"
_C_SWAP = _C_HEAD + _C_LOOP + "    if node[\"kind\"] == \"and\":\n        return best\n    return total\n"
_C_MAX = _C_HEAD + ("    total = 0\n"
                    "    best = cost(node[\"children\"][0])\n"
                    "    for child in node[\"children\"]:\n"
                    "        total = total + cost(child)\n"
                    "        if cost(child) > best:\n"
                    "            best = cost(child)\n") + "    if node[\"kind\"] == \"and\":\n        return total\n    return best\n"
_C_FLAT = _C_HEAD + ("    total = 0\n"
                     "    best = node[\"children\"][0][\"cost\"]\n"
                     "    for child in node[\"children\"]:\n"
                     "        total = total + child[\"cost\"]\n"
                     "        if child[\"cost\"] < best:\n"
                     "            best = child[\"cost\"]\n") + "    if node[\"kind\"] == \"and\":\n        return total\n    return best\n"
_C_SELF = _C_HEAD + ("    total = 0\n"
                     "    best = cost(node)\n"
                     "    for child in node[\"children\"]:\n"
                     "        total = total + cost(child)\n"
                     "        if cost(child) < best:\n"
                     "            best = cost(child)\n") + "    if node[\"kind\"] == \"and\":\n        return total\n    return best\n"

# exercise 3: cheapest
_P_FIRST = _P_HEAD + "    return cheapest(node[\"children\"][0])\n"
_P_DEAR = _P_HEAD + ("    best = node[\"children\"][0]\n"
                     "    for child in node[\"children\"]:\n"
                     "        if cost(child) > cost(best):\n"
                     "            best = child\n"
                     "    return cheapest(best)\n")
_P_NAME = _P_HEAD + ("    best = node[\"children\"][0]\n"
                     "    for child in node[\"children\"]:\n"
                     "        if cost(child) < cost(best):\n"
                     "            best = child\n"
                     "    return [best[\"name\"]]\n")
_P_ALL = _P_HEAD + ("    steps = []\n"
                    "    for child in node[\"children\"]:\n"
                    "        for step in cheapest(child):\n"
                    "            steps.append(step)\n"
                    "    return steps\n")
_P_TIE = _P_HEAD + ("    best = node[\"children\"][0]\n"
                    "    for child in node[\"children\"]:\n"
                    "        if cost(child) <= cost(best):\n"
                    "            best = child\n"
                    "    return cheapest(best)\n")
_P_FLAT = _P_HEAD + ("    best = node[\"children\"][0]\n"
                     "    for child in node[\"children\"]:\n"
                     "        if child[\"cost\"] < best[\"cost\"]:\n"
                     "            best = child\n"
                     "    return cheapest(best)\n")


def _key_error(e: BaseException) -> bool:
    return isinstance(e, KeyError)


def _recursion_error(e: BaseException) -> bool:
    return isinstance(e, RecursionError)


T = THREATS
_TWO = Named([T[1], T[6]], "[THREATS[1], THREATS[6]]")
_ONE = Named([T[4]], "[THREATS[4]]")
_EMPTY = Named([], "[]")
_FOUR = Named([T[3], T[0], T[7], T[5]], "[THREATS[3], THREATS[0], THREATS[7], THREATS[5]]")
_TINY_AND = Node({"name": "Tiny", "kind": "and", "children": [Node(_leaf("A", 5, True), 'TINY["children"][0]'),
                                                               Node(_leaf("B", 7, True), 'TINY["children"][1]')]},
                 "TINY")
_TIE = Node({"name": "Tie", "kind": "or", "children": [Node(_leaf("A", 5, True), 'TIE["children"][0]'),
                                                       Node(_leaf("B", 5, True), 'TIE["children"][1]')]},
            "TIE")
_TINY_OR = Node({"name": "Tiny", "kind": "or", "children": [Node(_leaf("A", 9, True), 'TINY["children"][0]'),
                                                             Node(_leaf("B", 4, True), 'TINY["children"][1]'),
                                                             Node(_leaf("C", 6, True), 'TINY["children"][2]')]},
                "TINY")

EXERCISES = {
    "at-risk": FunctionBox(
        name="at-risk",
        picture=risk_picture,
        function="at_risk",
        names={"PROPERTY": PROPERTY},
        start=_A_START,
        solution=_A,
        visible=[(THREATS,)],
        hidden=[(_TWO,), (_ONE,), (_EMPTY,), (_FOUR,)],
        first=[_DOTS],
        mistakes=[
            Mistake("key", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a key that is not there. A threat has a `\"what\"` and a `\"stride\"`; `PROPERTY` has the kinds of STRIDE threat as keys; and `counts` has a property as a key only once you have put it there.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een key op die er niet is. Een dreiging heeft een `\"what\"` en een `\"stride\"`; `PROPERTY` heeft de soorten STRIDE-dreiging als keys; en `counts` heeft een eigenschap pas als key als je die erin hebt gezet."),
            Mistake("kind", wrong=_A_KIND,
                    en="`{call}` gives {got}: a count per kind of STRIDE threat. Count per property instead: look up in `PROPERTY` which property each kind breaks.",
                    nl="`{call}` geeft {got}: een aantal per soort STRIDE-dreiging. Tel per eigenschap: zoek in `PROPERTY` op welke eigenschap elke soort breekt."),
            Mistake("one", wrong=_A_ONE,
                    en="`{call}` gives {got}, and it should give {want}. Every count is 1: your function sets a property to 1 each time it sees it. Add 1 when the property is already in `counts`, and start at 1 only when it is not.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Elk aantal is 1: je functie zet een eigenschap elke keer op 1. Tel er 1 bij op als de eigenschap al in `counts` staat, en begin pas bij 1 als die er nog niet in staat."),
            Mistake("inside", wrong=_A_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}. Is your `return` inside the loop? Then the function stops after the first threat. Put it after the loop, at the level of `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Staat je `return` in de loop? Dan stopt de functie na de eerste dreiging. Zet hem na de loop, op de hoogte van `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: in the loop, look up which property the threat breaks, and count that property in `counts`.",
                      "nl": "Vervang de `...`: zoek in de loop op welke eigenschap de dreiging breekt, en tel die eigenschap in `counts`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each threat, look up its property in `PROPERTY`, and count how many threats break each property.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Zoek voor elke dreiging de eigenschap op in `PROPERTY`, en tel hoeveel dreigingen elke eigenschap breken."},
            "almost": {"en": "Right for `at_risk(THREATS)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `threats` it gets?",
                       "nl": "Goed voor `at_risk(THREATS)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `threats` die hij krijgt?"},
            "correct": {"en": "Your `at_risk` is right in all {n} cases. The picture draws its counts for the eight threats.",
                        "nl": "Je `at_risk` klopt in alle {n} gevallen. Het plaatje tekent zijn aantallen voor de acht dreigingen."},
        },
    ),
    "cost": FunctionBox(
        name="cost",
        picture=cost_picture,
        function="cost",
        start=_C_START,
        solution=_C,
        visible=[(TREE,)],
        hidden=[(LOGIN,), (BACKUP,), (PASSWORD,), (SQL,), (SAFE,), (SAFE["children"][1],), (_TINY_AND,), (_TINY_OR,), (_TIE,)],
        first=[_DOTS],
        mistakes=[
            Mistake("self", error=_recursion_error,
                    en="With `{call}`, your function keeps calling itself and never stops, until Python gives up. Does it call `cost` on the node it got? Call it on each child: a child is a smaller piece of the tree.",
                    nl="Met `{call}` blijft je functie zichzelf aanroepen en stopt hij nooit, tot Python het opgeeft. Roept hij `cost` aan op de node die hij kreeg? Roep hem aan op elk kind: een kind is een kleiner stuk van de boom."),
            Mistake("flat", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a key that is not there. Only a leaf has a `\"cost\"`; a child can be an `\"and\"` or `\"or\"` node with children of its own. Let `cost` work out what a child costs.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een key op die er niet is. Alleen een blad heeft een `\"cost\"`; een kind kan een `\"and\"`- of `\"or\"`-node zijn met eigen kinderen. Laat `cost` uitrekenen wat een kind kost."),
            Mistake("sum", wrong=_C_SUM,
                    en="`{call}` gives {got}, and it should give {want}. Your function adds up the children of every node. That is right for an AND node, where the attacker has to take every step. An OR node is a choice: it costs as much as its cheapest child.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt de kinderen van elke node op. Dat klopt voor een AND-node, waar de aanvaller elke stap moet zetten. Een OR-node is een keuze: die kost zoveel als het goedkoopste kind."),
            Mistake("min", wrong=_C_MIN,
                    en="`{call}` gives {got}, and it should give {want}. Your function gives every node the cost of its cheapest child. That is right for an OR node. An AND node needs all its steps: it costs the sum of its children.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie geeft elke node de kosten van het goedkoopste kind. Dat klopt voor een OR-node. Een AND-node heeft al zijn stappen nodig: die kost de som van zijn kinderen."),
            Mistake("swap", wrong=_C_SWAP,
                    en="`{call}` gives {got}, and it should give {want}. Are AND and OR swapped? An AND node costs the sum of its children; an OR node the cost of its cheapest child.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Zijn AND en OR omgewisseld? Een AND-node kost de som van zijn kinderen; een OR-node de kosten van het goedkoopste kind."),
            Mistake("max", wrong=_C_MAX,
                    en="`{call}` gives {got}, and it should give {want}. For an OR node, your function keeps the most expensive child. The attacker picks the cheapest one: keep a child when it costs less than the best so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij een OR-node houdt je functie het duurste kind. De aanvaller kiest het goedkoopste: houd een kind als het minder kost dan het beste tot nu toe."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: go over the children, and give back their total for an AND node and the cost of the cheapest one for an OR node.",
                      "nl": "Vervang de `...`: ga langs de kinderen, en geef hun totaal terug voor een AND-node en de kosten van het goedkoopste voor een OR-node."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A leaf costs its `\"cost\"`, an AND node the sum of what its children cost, an OR node what its cheapest child costs.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een blad kost zijn `\"cost\"`, een AND-node de som van wat zijn kinderen kosten, een OR-node wat het goedkoopste kind kost."},
            "almost": {"en": "Right for `cost(TREE)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `node` it gets?",
                       "nl": "Goed voor `cost(TREE)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `node` die hij krijgt?"},
            "correct": {"en": "Your `cost` is right in all {n} cases, also on Schneier's safe. The picture runs it on every node of the library's tree.",
                        "nl": "Je `cost` klopt in alle {n} gevallen, ook bij de kluis van Schneier. Het plaatje laat hem lopen op elke node van de boom van de bibliotheek."},
        },
    ),
    "cheapest": FunctionBox(
        name="cheapest",
        picture=path_picture,
        function="cheapest",
        names={"cost": cost},
        start=_P_START,
        solution=_P,
        visible=[(TREE,)],
        hidden=[(LOGIN,), (PASSWORD,), (BACKUP,), (SAFE,), (SAFE["children"][1],), (_TINY_OR,), (_TINY_AND,), (_TIE,)],
        first=[_DOTS],
        mistakes=[
            Mistake("flat", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up a key that is not there. Only a leaf has a `\"cost\"`. Ask `cost` what a child costs: it is ready to use.",
                    nl="Met `{call}` stopt je functie met een fout: hij zoekt een key op die er niet is. Alleen een blad heeft een `\"cost\"`. Vraag `cost` wat een kind kost: die staat klaar."),
            Mistake("first", wrong=_P_FIRST,
                    en="`{call}` gives {got}, and it should give {want}. For an OR node, your function always follows the first child. Follow the child that costs least.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij een OR-node volgt je functie altijd het eerste kind. Volg het kind dat het minst kost."),
            Mistake("dear", wrong=_P_DEAR,
                    en="`{call}` gives {got}, and it should give {want}. For an OR node, your function follows the most expensive child. Keep a child when it costs less than the best so far.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij een OR-node volgt je functie het duurste kind. Houd een kind als het minder kost dan het beste tot nu toe."),
            Mistake("name", wrong=_P_NAME,
                    en="`{call}` gives {got}, and it should give {want}. For an OR node, your function gives the name of the cheapest child, which can be a node with steps of its own. Give back the steps under that child: let `cheapest` find them.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij een OR-node geeft je functie de naam van het goedkoopste kind, en dat kan een node zijn met eigen stappen. Geef de stappen onder dat kind terug: laat `cheapest` ze zoeken."),
            Mistake("tie", wrong=_P_TIE,
                    en="`{call}` gives {got}, and it should give {want}. With two equally cheap children, your function follows the last of them. Keep the first: replace the best child only when another one costs less, not when it costs the same.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij twee even goedkope kinderen volgt je functie het laatste. Houd het eerste: vervang het beste kind alleen als een ander minder kost, niet als het evenveel kost."),
            Mistake("all", wrong=_P_ALL,
                    en="`{call}` gives {got}, and it should give {want}. For an OR node, your function gives the steps of every child, as for an AND node. An OR node is a choice: give back only the steps of its cheapest child.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij een OR-node geeft je functie de stappen van elk kind, zoals bij een AND-node. Een OR-node is een keuze: geef alleen de stappen van het goedkoopste kind terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: for an OR node, find the child that costs least, and give back the steps of the cheapest attack under it.",
                      "nl": "Vervang de `...`: zoek bij een OR-node het kind dat het minst kost, en geef de stappen van de goedkoopste aanval daaronder terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For an OR node, give back the steps of the cheapest attack under its cheapest child; with two equally cheap children, the first of them.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef bij een OR-node de stappen van de goedkoopste aanval onder het goedkoopste kind terug; bij twee even goedkope kinderen het eerste daarvan."},
            "almost": {"en": "Right for `cheapest(TREE)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `node` it gets?",
                       "nl": "Goed voor `cheapest(TREE)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `node` die hij krijgt?"},
            "correct": {"en": "Your `cheapest` is right in all {n} cases, also on Schneier's safe. The picture marks its steps in the library's tree.",
                        "nl": "Je `cheapest` klopt in alle {n} gevallen, ook bij de kluis van Schneier. Het plaatje markeert zijn stappen in de boom van de bibliotheek."},
        },
    ),
}

CASES = [
    ("at-risk", _A_START, "empty", None),
    ("at-risk", _A, "correct", None),
    ("at-risk", _A_HEAD + "        counts[PROPERTY[threat[\"stride\"]]] = counts.get(PROPERTY[threat[\"stride\"]], 0) + 1\n    return counts\n", "correct", None),
    ("at-risk", "def at_risk(threats):\n    counts = {}\n    for prop in PROPERTY.values():\n        n = 0\n        for threat in threats:\n            if PROPERTY[threat[\"stride\"]] == prop:\n                n = n + 1\n        if n > 0:\n            counts[prop] = n\n    return counts\n", "correct", None),
    ("at-risk", _A_KEY, "error", "key"),
    ("at-risk", _A_HEAD + "        prop = PROPERTY[threat[\"what\"]]\n        counts[prop] = 1\n    return counts\n", "error", "key"),
    ("at-risk", _A_KIND, "wrong", "kind"),
    ("at-risk", _A_ONE, "wrong", "one"),
    ("at-risk", _A_INSIDE, "wrong", "inside"),
    ("at-risk", "def at_risk(threats):\n    return {\"authentication\": 1, \"integrity\": 2, \"non-repudiation\": 1, \"confidentiality\": 2, \"availability\": 1, \"authorization\": 1}\n", "almost", None),
    ("at-risk", _A_HEAD + "        prop = PROPERTY[threat[\"stride\"]]\n        counts[prop] = 2\n    return counts\n", "wrong", None),
    ("cost", _C_START, "empty", None),
    ("cost", _C, "correct", None),
    ("cost", _C_HEAD + "    costs = []\n    for child in node[\"children\"]:\n        costs.append(cost(child))\n    if node[\"kind\"] == \"and\":\n        return sum(costs)\n    return min(costs)\n", "correct", None),
    ("cost", _C_HEAD + "    if node[\"kind\"] == \"and\":\n        total = 0\n        for child in node[\"children\"]:\n            total = total + cost(child)\n        return total\n    best = None\n    for child in node[\"children\"]:\n        c = cost(child)\n        if best is None or c < best:\n            best = c\n    return best\n", "correct", None),
    ("cost", _C_SELF, "error", "self"),
    ("cost", _C_FLAT, "error", "flat"),
    ("cost", _C_SUM, "wrong", "sum"),
    ("cost", _C_MIN, "wrong", "min"),
    ("cost", _C_SWAP, "wrong", "swap"),
    ("cost", _C_MAX, "wrong", "max"),
    ("cost", "def cost(node):\n    return 500\n", "almost", None),
    ("cost", _C_HEAD + "    return 0\n", "wrong", None),
    ("cheapest", _P_START, "empty", None),
    ("cheapest", _P, "correct", None),
    ("cheapest", _P_HEAD + "    best = None\n    for child in node[\"children\"]:\n        if best is None or cost(child) < cost(best):\n            best = child\n    return cheapest(best)\n", "correct", None),
    ("cheapest", _P_FLAT, "error", "flat"),
    ("cheapest", _P_FIRST, "wrong", "first"),
    ("cheapest", _P_DEAR, "wrong", "dear"),
    ("cheapest", _P_NAME, "wrong", "name"),
    ("cheapest", _P_ALL, "wrong", "all"),
    ("cheapest", "def cheapest(node):\n    return [\"Guess the password\", \"Phish the code too\"]\n", "almost", None),
    ("cheapest", _P_TIE, "wrong", "tie"),
    ("cheapest", _P_HEAD + "    return []\n", "wrong", None),
]
