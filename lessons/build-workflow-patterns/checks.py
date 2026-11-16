# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-workflow-patterns: five shapes of a workflow (steps in a row, a router, parts side by side, an
orchestrator with workers, an evaluator that sends an answer back), shown on the kitchen of build-workflow-or-agent.
The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): a tip for each dish under a word limit, with a
check in code that sends a too long tip back with what is wrong (evaluator-optimizer), and menus of several dishes that
the model splits into its dishes (orchestrator). What they show, their pictures, the exercises (the evaluator's loop,
and the orchestrator's) with stand-ins of the model, and the demo with the learner's loop on Qwen3."""
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
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_patterns",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _lesson_checks("build-workflow-or-agent")   # the kitchen: recipes, pantry, what to buy
DISHES, RECIPES, recipe, pantry, to_buy, as_text = W.DISHES, W.RECIPES, W.recipe, W.pantry, W.to_buy, W.as_text

# --- evaluator-optimizer: a tip under a word limit, checked in code --------------------------------------------------

LIMIT = 8   # words in a tip at most
TIP = "Write one tip for making {dish}, in at most {limit} words. Write only the tip."
FEEDBACK = "Your tip has {n} words. That is too many: write a new, shorter tip of at most {limit} words."
AGAIN = "Your tip has {n} words. Write it again in at most {limit} words."   # the first feedback the course tried
KINDS = {"again": AGAIN, "tip": FEEDBACK}   # the recorded runs of each feedback, by the start of their keys
TRIES = 4   # answers the evaluator reads at most


def words(text: str) -> int:
    """The words of a text, as the check counts them: the parts between spaces."""
    return len(text.split())


def follow_up(text: str, feedback: str = FEEDBACK):
    """The evaluator in code, for the recorder too: None when the tip is short enough, else what to tell the writer."""
    n = words(text)
    return None if n <= LIMIT else feedback.format(n=n, limit=LIMIT)


def asked(text: str) -> list:
    return [{"role": "user", "content": text}]


# --- orchestrator: the model splits a menu into its dishes ---------------------------------------------------------------

PLAN = ('Which dishes does this question ask about? Answer with JSON: {{"dishes": [...]}}, with names from this list: '
        + ", ".join(DISHES) + ".\n\nQuestion: {question}")
MENUS = [
    ("I am making pancakes and tomato soup for dinner. What do I have to buy?", ["pancakes", "tomato soup"]),
    ("This weekend I want to make an omelette, rice pudding and an apple pie. What must I buy?", ["omelette", "rice pudding", "apple pie"]),
    ("What do I need to buy for pasta with pesto and an apple pie?", ["pasta with pesto", "apple pie"]),
    ("Guests are coming: tomato soup first, then pasta with pesto. What is missing?", ["tomato soup", "pasta with pesto"]),
    ("I want pancakes for breakfast and an omelette for lunch. What do I have to get?", ["pancakes", "omelette"]),
    ("A shopping list for rice pudding, pancakes and tomato soup, please.", ["rice pudding", "pancakes", "tomato soup"]),
]

RECORD = {}
for _kind, _feedback in KINDS.items():
    for _j, _d in enumerate(DISHES):
        RECORD[f"{_kind}-{_j}"] = {"messages": asked(TIP.format(dish=_d, limit=LIMIT)), "max_tokens": 60, "temperature": 0,
                                   "retry": TRIES, "feedback": _feedback, "seeds": 1}
for _i, (_q, _) in enumerate(MENUS):
    RECORD[f"plan-{_i}"] = {"messages": asked(PLAN.format(question=_q)), "max_tokens": 60, "temperature": 0,
                            "response_format": {"type": "json_object"}, "seeds": 1}


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-workflow-patterns"]


def run(key: str) -> dict:
    return course()[key]["answers"][0]


def tries(j: int, kind: str = "tip") -> list:
    """The answers of the writer for dish j, in order, until the evaluator let one through or TRIES were read; with the
    course's feedback ("tip"), or with the first one it tried ("again")."""
    return [t["text"].strip() for t in run(f"{kind}-{j}")["tries"]]


def planned(i: int):
    """The dishes the model's plan names for menu i, or None when its answer is not such JSON."""
    try:
        plan = json.loads(run(f"plan-{i}")["text"])
    except ValueError:
        return None
    dishes = plan.get("dishes") if isinstance(plan, dict) else None
    return dishes if isinstance(dishes, list) and all(isinstance(d, str) for d in dishes) else None


def _tips(kind: str) -> dict:
    n = len(DISHES)
    answers = [tries(j, kind) for j in range(n)]
    counts = [[words(t) for t in a] for a in answers]
    return {
        "words": counts,
        "first_ok": [j for j in range(n) if counts[j][0] <= LIMIT],
        "end_ok": [j for j in range(n) if counts[j][-1] <= LIMIT],
        "rounds": [len(c) for c in counts],
        "more": sum(len(c) - 1 for c in counts),   # requests after the first, one per answer sent back
        "copied": [j for j in range(n) if len(answers[j]) > 1 and len(set(answers[j])) == 1],   # the same tip every round
        "seconds": sum(t["seconds"] for j in range(n) for t in run(f"{kind}-{j}")["tries"]),
    }


def findings() -> dict:
    plans = [planned(i) for i in range(len(MENUS))]
    return {
        "dishes": len(DISHES), "limit": LIMIT, "tries": TRIES,
        "again": _tips("again"), "tip": _tips("tip"),
        "menus": len(MENUS),
        "plans_right": [i for i, p in enumerate(plans) if p is not None and sorted(p) == sorted(MENUS[i][1])],
        "plans_json": [i for i, p in enumerate(plans) if p is not None],
    }


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the pictures ----------------------------------------------------------------------------------------------------------

_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"
_UI = "font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _step(text: str, role: str) -> str:
    color = {"model": "var(--ea-pred)", "code": "var(--ea-ink-3)", "data": "var(--ea-data)"}[role]
    return (f'<span style="display:inline-block;border:1.5px solid {color};border-radius:5px;padding:0.1rem 0.4rem;'
            f'{_UI};font-size:12px;color:var(--ea-ink);background:var(--ea-page);margin:0.1rem 0">{text}</span>')


_ARROW = f'<span style="{_UI};font-size:12px;color:var(--ea-ink-3);margin:0 0.25rem">→</span>'

_PATTERN_WORDS = {
    "en": {"title": "Five workflow patterns", "alt": "Five small flows: steps in a row, a router, parts side by side, an orchestrator with workers, and an evaluator that sends an answer back to the writer.",
           "chain": ("Steps in a row", "model", "code: check", "model"),
           "router": ("Router", "model: which kind?", "the path for that kind"),
           "parallel": ("Parts side by side", "model, model, model at once", "code: put together"),
           "orchestrator": ("Orchestrator and workers", "model: which parts?", "a worker per part", "put together"),
           "evaluator": ("Evaluator and optimizer", "model: an answer", "check: good?", "no: back, with why")},
    "nl": {"title": "Vijf workflowpatronen", "alt": "Vijf kleine stromen: stappen na elkaar, een router, delen naast elkaar, een orchestrator met workers, en een evaluator die een antwoord terugstuurt naar de schrijver.",
           "chain": ("Stappen na elkaar", "model", "code: controle", "model"),
           "router": ("Router", "model: welke soort?", "het pad voor die soort"),
           "parallel": ("Delen naast elkaar", "model, model, model tegelijk", "code: samenvoegen"),
           "orchestrator": ("Orchestrator en workers", "model: welke delen?", "een worker per deel", "samenvoegen"),
           "evaluator": ("Evaluator en optimizer", "model: een antwoord", "controle: goed?", "nee: terug, met waarom")},
}
_ROLES = {"chain": ("model", "code", "model"), "router": ("model", "code"), "parallel": ("model", "code"),
          "orchestrator": ("model", "model", "code"), "evaluator": ("model", "code", "code")}


def patterns_picture(lang: str) -> str:
    """The five patterns, each as a small flow of steps, the model's in the color of what a model makes."""
    w = _PATTERN_WORDS[lang]
    rows = []
    for key in ("chain", "router", "parallel", "orchestrator", "evaluator"):
        name, *steps = w[key]
        flow = _ARROW.join(_step(html.escape(s), r) for s, r in zip(steps, _ROLES[key]))
        if key == "evaluator":
            flow += f'<span style="{_UI};font-size:12px;color:var(--ea-ink-3);margin-left:0.3rem">↺</span>'
        rows.append(f'<tr><td style="{_CELL};{_UI};font-weight:600;white-space:nowrap">{html.escape(name)}</td>'
                    f'<td style="{_CELL}">{flow}</td></tr>')
    table = f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><tbody>{"".join(rows)}</tbody></table></div>'
    return yours.figure(w["title"], table, alt=w["alt"])


_TIPS_WORDS = {
    "en": {"title": "A tip of at most {limit} words: each answer, and what the check said", "dish": "dish",
           "again": "back with “Write it again …”", "tip": "back with “… write a new, shorter tip …”",
           "words": "{n} words", "ok": "let through", "back": "sent back", "gave_up": "still too long after {k} answers"},
    "nl": {"title": "Een tip van hooguit {limit} woorden: elk antwoord, en wat de controle zei", "dish": "gerecht",
           "again": "terug met “Write it again …”", "tip": "terug met “… write a new, shorter tip …”",
           "words": "{n} woorden", "ok": "doorgelaten", "back": "teruggestuurd", "gave_up": "na {k} antwoorden nog te lang"},
}


def _answers_cell(answers: list, w: dict) -> str:
    lines = []
    for k, text in enumerate(answers):
        n = words(text)
        ok = n <= LIMIT
        verdict = w["ok"] if ok else (w["back"] if k + 1 < len(answers) or len(answers) < TRIES else w["gave_up"].format(k=len(answers)))
        color = "var(--ea-ink-2)" if ok else "var(--ea-loss-text)"
        lines.append(f'<div>{_code(text, "var(--ea-pred-text)")} <span style="{_UI};font-size:12px;color:{color}">'
                     f'{w["words"].format(n=n)}, {verdict}</span></div>')
    return "".join(lines)


def tips_picture(lang: str) -> str:
    """For each dish, each answer of the writer with its words and whether the check let it through: with the first
    feedback the course tried, and with the course's feedback."""
    w = _TIPS_WORDS[lang]
    rows = [f'<tr><td style="{_CELL};{_UI}">{html.escape(dish)}</td>'
            + "".join(f'<td style="{_CELL}">{_answers_cell(tries(j, kind), w)}</td>' for kind in ("again", "tip")) + "</tr>"
            for j, dish in enumerate(DISHES)]
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["dish"], w["again"], w["tip"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"].format(limit=LIMIT), table, alt=w["title"].format(limit=LIMIT))


_PLANS_WORDS = {
    "en": {"title": "Menus split into their dishes by Qwen3", "menu": "the question", "plan": "the model's plan", "right": "right",
           "wrong": "not right: the menu has {want}", "bad": "no JSON with a list of dishes"},
    "nl": {"title": "Menu's door Qwen3 opgesplitst in hun gerechten", "menu": "de vraag", "plan": "het plan van het model", "right": "goed",
           "wrong": "niet goed: het menu heeft {want}", "bad": "geen JSON met een lijst gerechten"},
}


def plans_picture(lang: str) -> str:
    """Each menu, the dishes the model's plan named, and whether they are the menu's."""
    w, f = _PLANS_WORDS[lang], findings()
    rows = []
    for i, (q, want) in enumerate(MENUS):
        p = planned(i)
        if p is None:
            verdict, color = w["bad"], "var(--ea-loss-text)"
        elif i in f["plans_right"]:
            verdict, color = w["right"], "var(--ea-ink-2)"
        else:
            verdict, color = w["wrong"].format(want=", ".join(want)), "var(--ea-loss-text)"
        shown = run(f"plan-{i}")["text"].strip() if p is None else ", ".join(str(d) for d in p)
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(q)}</td><td style="{_CELL}">{_code(shown, "var(--ea-pred-text)")}<br>'
                    f'<span style="{_UI};font-size:12px;color:{color}">{html.escape(verdict)}</span></td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead><tr>'
             + "".join(f'<th style="{_HEAD}">{h}</th>' for h in (w["menu"], w["plan"]))
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


def tips_text(lang: str) -> str:
    """What the tips show, in the lesson's language, with the measured numbers."""
    f = findings()
    n, a, t = f["dishes"], f["again"], f["tip"]
    first = len(t["first_ok"])
    if lang == "nl":
        return (f"Van de eerste tips voor de {n} gerechten {'was er' if first == 1 else 'waren er'} {first} meteen kort genoeg. Met *Write it again* "
                f"kwamen er {len(a['end_ok'])} van de {n} door binnen {f['tries']} antwoorden: bij {len(a['copied'])} gerechten "
                f"schreef het model elke ronde woord voor woord dezelfde tip terug. Met *write a new, shorter tip* kwamen er "
                f"{len(t['end_ok'])} van de {n} door, en had de loop {t['more']} extra requests nodig in plaats van {a['more']}. De controle was beide keren dezelfde; "
                f"alleen de woorden van de terugkoppeling verschilden. Daarom gebruikt de course de tweede.")
    return (f"Of the first tips for the {n} dishes, {first} {'was' if first == 1 else 'were'} short enough at once. With *Write it again*, "
            f"{len(a['end_ok'])} of the {n} got through within {f['tries']} answers: for {len(a['copied'])} dishes, the model "
            f"wrote the same tip back word for word in every round. With *write a new, shorter tip*, {len(t['end_ok'])} of the "
            f"{n} got through, and the loop needed {t['more']} extra requests instead of {a['more']}. The check was the same both times; only the words of the "
            f"feedback differed. That is why the course uses the second.")


def plans_text(lang: str) -> str:
    """What the plans show, in the lesson's language, with the measured numbers."""
    f = findings()
    m, right, js = f["menus"], len(f["plans_right"]), len(f["plans_json"])
    if lang == "nl":
        found = (f"Alle {m} antwoorden waren JSON, daar zorgt JSON mode voor; elk had een lijst met precies de gerechten van "
                 f"zijn menu."
                 if js == right == m else
                 f"{js} van de {m} antwoorden waren JSON met een lijst; {right} noemden precies de gerechten van het menu.")
        return (f"Voor de orchestrator vroeg de course het model {m} menu's op te splitsen in hun gerechten, als JSON, met JSON "
                f"mode aan. {found}")
    found = (f"All {m} answers were JSON, which JSON mode makes sure of; each held a list of exactly the dishes of its "
             f"menu." if js == right == m else
             f"{js} of the {m} answers were JSON with a list; {right} named exactly the dishes of the menu.")
    return (f"For the orchestrator, the course asked the model to split {m} menus into their dishes, as JSON, with JSON mode "
            f"on. {found}")


WORKED = 0   # the dish of the worked example


def worked_text(lang: str) -> str:
    """The worked example: the evaluator's rounds for one dish, from the recorded run."""
    dish, answers = DISHES[WORKED], tries(WORKED)
    lines = []
    for k, text in enumerate(answers, 1):
        said = follow_up(text)
        if lang == "nl":
            lines.append(f"{k}. Het model schrijft: *{text}* ({words(text)} woorden). "
                         + ("De controle laat de tip door." if said is None else f"De controle stuurt hem terug: *{said}*"))
        else:
            lines.append(f"{k}. The model writes: *{text}* ({words(text)} words). "
                         + ("The check lets the tip through." if said is None else f"The check sends it back: *{said}*"))
    head = ("### Uitgewerkt voorbeeld: de rondes voor {d}\n\nDe evaluator vraagt een tip voor {d} van hooguit {limit} woorden.\n\n"
            if lang == "nl" else
            "### Worked example: the rounds for {d}\n\nThe evaluator asks for a tip for {d} of at most {limit} words.\n\n")
    return head.format(d=f"`{dish}`", limit=LIMIT) + "\n".join(lines)


# --- stand-ins of the model for the exercises ------------------------------------------------------------------------

class Writer:
    """A stand-in for the model as writer: its first answer, and the next one when the messages show it an answer of its
    own (role assistant) followed by why it went back (role user), as a model improves only when it reads both; the
    messages may hold every earlier answer, or only the last one.
    It shows itself by its name, and keeps what it gave back in `gave`."""

    def __init__(self, name: str, answers: list):
        self.name, self.answers, self.gave = name, list(answers), []

    def __call__(self, messages):
        if not isinstance(messages, list):
            raise TypeError("ask takes the list of messages, not " + type(messages).__name__)
        shown = [m for m in messages if isinstance(m, dict)]
        pairs = [a.get("content") for a, b in zip(shown, shown[1:]) if a.get("role") == "assistant" and b.get("role") == "user"]
        k = len(pairs)   # answers it read back, each with why it went back: the whole history, or only the last one
        if pairs and not (k <= len(self.answers) and pairs[-1] == self.answers[k - 1]) and pairs[-1] in self.answers:
            k = self.answers.index(pairs[-1]) + 1
        answer = self.answers[min(k, len(self.answers) - 1)]
        self.gave.append(answer)
        return answer

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Writer(self.name, self.answers)


class Limit:
    """The evaluator of the exercise: None when an answer has at most `limit` words, else what is wrong with it."""

    def __init__(self, limit: int):
        self.limit = limit

    def __call__(self, answer):
        n = len(str(answer).split())
        return None if n <= self.limit else FEEDBACK.format(n=n, limit=self.limit)

    def __repr__(self):
        return f"words_at_most_{self.limit}"


SHORT = "Whisk the batter until smooth."
LONG = "Make sure to whisk the batter very well until it is completely smooth and light."
LONGER = "Always let the batter rest for half an hour before you start to fry the pancakes."

START_IMPROVE = '''def improve(ask, task, check, rounds):
    messages = [{"role": "user", "content": task}]
    for n in range(rounds):
        answer = ask(messages)
        problem = check(answer)
        ...
    return None
'''
SOLUTION_IMPROVE = START_IMPROVE.replace('''        ...
''', '''        if problem is None:
            return answer
        messages = messages + [{"role": "assistant", "content": answer},
                               {"role": "user", "content": problem}]
''')
TASK = TIP.format(dish="pancakes", limit=LIMIT)

_NO_BACK = START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n")
_FIRST = START_IMPROVE.replace("        ...\n", "        return answer\n")
_ONLY_PROBLEM = START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n"
                                      "        messages = messages + [{\"role\": \"user\", \"content\": problem}]\n")
_LAST = START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n"
                              "        messages = messages + [{\"role\": \"assistant\", \"content\": answer},\n"
                              "                               {\"role\": \"user\", \"content\": problem}]\n").replace("    return None\n", "    return answer\n")


IMPROVE_VISIBLE = [(Writer("second_try", [LONG, SHORT]), TASK, Limit(LIMIT), 3)]
IMPROVE_HIDDEN = [
    (Writer("first_try", [SHORT]), TASK, Limit(LIMIT), 3),
    (Writer("third_try", [LONGER, LONG, SHORT]), TASK, Limit(LIMIT), 3),
    (Writer("never_short", [LONG, LONGER, LONG]), TASK, Limit(LIMIT), 3),
    (Writer("too_late", [LONG, LONGER, SHORT]), TASK, Limit(LIMIT), 2),
]


_IMPROVE_WORDS = {
    "en": {"title": "Your improve, round by round", "case": "{writer}, at most {rounds} rounds", "asked": "round {n}: the model wrote",
           "check": "the check: {said}", "ok": "good", "gave": "your improve gave back: {v}", "error": "your improve stopped with {e}",
           "sent": "and got back {k} messages"},
    "nl": {"title": "Je improve, ronde na ronde", "case": "{writer}, hooguit {rounds} rondes", "asked": "ronde {n}: het model schreef",
           "check": "de controle: {said}", "ok": "goed", "gave": "je improve gaf terug: {v}", "error": "je improve stopte met {e}",
           "sent": "en kreeg {k} berichten terug"},
}


def improve_picture(seen) -> str:
    """What the learner's improve did with the writer of the example: each round's answer and what the check said."""
    run_ = seen.run
    fn = run_.names.get("improve") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _IMPROVE_WORDS[seen.lang]
    blocks = []
    for writer, task, check, rounds in IMPROVE_VISIBLE + IMPROVE_HIDDEN[2:3]:
        writer = copy.deepcopy(writer)
        try:
            got, error = fn(writer, task, check, rounds), None
        except Exception as e:   # noqa: BLE001 - the learner's code
            got, error = None, e
        lines = [f'<div style="{_UI};font-size:12px;font-weight:600;margin-top:0.3rem">{html.escape(w["case"].format(writer=writer.name, rounds=rounds))}</div>']
        for n, answer in enumerate(writer.gave, 1):
            said = check(answer)
            lines.append(f'<div style="{_UI};font-size:12px">{html.escape(w["asked"].format(n=n))} {_code(answer, "var(--ea-pred-text)")} '
                         f'<span style="color:{"var(--ea-ink-2)" if said is None else "var(--ea-loss-text)"}">'
                         f'{html.escape(w["check"].format(said=w["ok"] if said is None else said))}</span></div>')
        end = w["error"].format(e=type(error).__name__) if error is not None else w["gave"].format(v=repr(got))
        lines.append(f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(end)}</div>')
        blocks.append("".join(lines))
    return yours.figure(w["title"], "".join(blocks), alt=w["title"])


# --- exercise 2: the orchestrator ----------------------------------------------------------------------------------------

class Planner:
    """A stand-in for the model as orchestrator: whatever it is asked, it answers with its plan as JSON text."""

    def __init__(self, name: str, dishes: list):
        self.name, self.dishes, self.asked = name, list(dishes), []

    def __call__(self, prompt):
        self.asked.append(prompt)
        return json.dumps({"dishes": self.dishes})

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Planner(self.name, self.dishes)


class Worker:
    """The worker of the exercise: the shopping list of one dish, as text."""

    def __call__(self, dish):
        return f"{dish}: {as_text(to_buy(dish))}"

    def __repr__(self):
        return "shopping_list"

    def __deepcopy__(self, memo):
        return Worker()


START_ORCHESTRATE = '''import json


def orchestrate(ask, question, worker):
    plan = json.loads(ask(PLAN.format(question=question)))
    results = []
    ...
    return results
'''
SOLUTION_ORCHESTRATE = START_ORCHESTRATE.replace('''    ...
''', '''    for dish in plan["dishes"]:
        if dish in DISHES:
            results.append(worker(dish))
''')
ORCHESTRA = {"PLAN": PLAN, "DISHES": DISHES}

_ALL = START_ORCHESTRATE.replace("    ...\n", "    for dish in plan[\"dishes\"]:\n        results.append(worker(dish))\n")
_FIRST_ONLY = START_ORCHESTRATE.replace("    ...\n", "    results.append(worker(plan[\"dishes\"][0]))\n")
_PLAN_BACK = START_ORCHESTRATE.replace("    ...\n", "    results = plan[\"dishes\"]\n")

ORCHESTRATE_VISIBLE = [(Planner("planner", ["pancakes", "tomato soup"]), MENUS[0][0], Worker())]
ORCHESTRATE_HIDDEN = [
    (Planner("planner", ["omelette", "rice pudding", "apple pie"]), MENUS[1][0], Worker()),
    (Planner("planner", ["pancakes", "pizza"]), "What do I need for pancakes and a pizza?", Worker()),
    (Planner("planner", []), "What can I make with what I have?", Worker()),
]

_ORCH_WORDS = {
    "en": {"title": "Your orchestrate on the menus of the check", "plan": "the plan: {p}", "gave": "your orchestrate gave back:",
           "error": "it stopped with {e}"},
    "nl": {"title": "Je orchestrate op de menu's van de check", "plan": "het plan: {p}", "gave": "je orchestrate gaf terug:",
           "error": "hij stopte met {e}"},
}


def orchestrate_picture(seen) -> str:
    """The plan of each menu of the check, and what the learner's orchestrate gave back for it."""
    run_ = seen.run
    fn = run_.names.get("orchestrate") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _ORCH_WORDS[seen.lang]
    rows = []
    for planner, question, worker in ORCHESTRATE_VISIBLE + ORCHESTRATE_HIDDEN[1:2]:
        planner = copy.deepcopy(planner)
        try:
            got, error = fn(planner, question, worker), None
        except Exception as e:   # noqa: BLE001 - the learner's code
            got, error = None, e
        out = (_code(w["error"].format(e=type(error).__name__), "var(--ea-loss-text)") if error is not None else
               "<br>".join(_code(str(x), "var(--ea-data-text)") for x in got) if isinstance(got, list) else _code(repr(got)))
        rows.append(f'<tr><td style="{_CELL};{_UI}">{html.escape(question)}<br>{_code(w["plan"].format(p=planner.dishes), "var(--ea-pred-text)")}</td>'
                    f'<td style="{_CELL}">{out}</td></tr>')
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><tbody>{"".join(rows)}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "improve": FunctionBox(
        name="improve",
        picture=improve_picture,
        function="improve",
        start=START_IMPROVE,
        solution=SOLUTION_IMPROVE,
        visible=IMPROVE_VISIBLE,
        hidden=IMPROVE_HIDDEN,
        mistakes=[
            Mistake("first", wrong=_FIRST,
                    en="With `{call}`, your function gives back the first answer, `{got}`, whatever the check says. Give back an answer only when `problem` is `None`.",
                    nl="Met `{call}` geeft je functie het eerste antwoord terug, `{got}`, wat de controle ook zegt. Geef een antwoord alleen terug als `problem` `None` is."),
            Mistake("last", wrong=_LAST,
                    en="With `{call}`, no answer got through the check, and your function gives back the last one anyway: `{got}`. After the last round, give back `None`, as the box did, so that your code knows no answer was good.",
                    nl="Met `{call}` kwam geen antwoord door de controle, en je functie geeft toch het laatste terug: `{got}`. Geef na de laatste ronde `None` terug, zoals het vak deed, zodat je code weet dat geen antwoord goed was."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The loop asks and checks, but does nothing with the check yet. When `problem` is `None`, give back the answer; otherwise add the answer and the problem to `messages`, so that the next round reads them.",
                      "nl": "De loop vraagt en controleert, maar doet nog niets met de controle. Is `problem` `None`, geef dan het antwoord terug; zet anders het antwoord en het probleem in `messages`, zodat de volgende ronde ze leest."},
            "unreached": {"en": "With `{call}`, your function gives back `None`: no answer got through the check, though a later one would have. In each next round, the model must read its own answer, with the role `assistant`, and then what was wrong with it, with the role `user`. Are both in `messages`?",
                          "nl": "Met `{call}` geeft je functie `None` terug: geen antwoord kwam door de controle, terwijl een later antwoord dat wel had gedaan. In elke volgende ronde moet het model zijn eigen antwoord lezen, met de rol `assistant`, en dan wat er mis mee was, met de rol `user`. Staan ze allebei in `messages`?"},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should give back `{want}`. Follow the rounds in the picture: which answer gets through the check, and what does the model read before each round?",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Volg de rondes in het plaatje: welk antwoord komt door de controle, en wat leest het model voor elke ronde?"},
            "almost": {"en": "Right for the example, but with `{call}` your function gives back `{got}`, and it should give back `{want}`. Does it work when the first answer is good at once, and when no answer is?",
                       "nl": "Goed voor het voorbeeld, maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Werkt hij ook als het eerste antwoord meteen goed is, en als geen antwoord dat is?"},
            "correct": {"en": "Your `improve` sends an answer back with what is wrong, until the check lets one through or the rounds run out: right for all {n} writers of the check.",
                        "nl": "Je `improve` stuurt een antwoord terug met wat er mis is, tot de controle er een doorlaat of de rondes op zijn: goed voor alle {n} schrijvers van de check."},
        },
    ),
    "orchestrate": FunctionBox(
        name="orchestrate",
        picture=orchestrate_picture,
        function="orchestrate",
        start=START_ORCHESTRATE,
        solution=SOLUTION_ORCHESTRATE,
        names=ORCHESTRA,
        visible=ORCHESTRATE_VISIBLE,
        hidden=ORCHESTRATE_HIDDEN,
        mistakes=[
            Mistake("first-only", wrong=_FIRST_ONLY,
                    en="With `{call}`, your function gives back `{got}`: the list of the first dish only. The plan decides how many parts there are: loop over all of `plan[\"dishes\"]`.",
                    nl="Met `{call}` geeft je functie `{got}` terug: alleen de lijst van het eerste gerecht. Het plan bepaalt hoeveel delen er zijn: loop over heel `plan[\"dishes\"]`."),
            Mistake("plan-back", wrong=_PLAN_BACK,
                    en="With `{call}`, your function gives back the plan itself, `{got}`. Give each dish of the plan to `worker`, and collect what the worker gives back.",
                    nl="Met `{call}` geeft je functie het plan zelf terug, `{got}`. Geef elk gerecht van het plan aan `worker`, en verzamel wat de worker teruggeeft."),
            Mistake("unknown", error=lambda e: isinstance(e, W.Unknown),
                    en="With `{call}`, the plan names a dish the kitchen has no recipe for, and your function gave it to the worker, which stopped. The plan comes from the model: give the worker only the dishes that are in `DISHES`.",
                    nl="Met `{call}` noemt het plan een gerecht waar de keuken geen recept voor heeft, en je functie gaf het aan de worker, die stopte. Het plan komt van het model: geef de worker alleen de gerechten die in `DISHES` staan."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "The plan is read, but no worker gets a part yet. For each dish of `plan[\"dishes\"]` that is in `DISHES`, add what `worker(dish)` gives to `results`.",
                      "nl": "Het plan is gelezen, maar nog geen worker krijgt een deel. Voeg voor elk gerecht van `plan[\"dishes\"]` dat in `DISHES` staat, toe aan `results` wat `worker(dish)` geeft."},
            "wrong": {"en": "With `{call}`, your function gives back `{got}`, and it should give back `{want}`: one result of the worker for each dish of the plan, in the plan's order.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn: één resultaat van de worker voor elk gerecht van het plan, in de volgorde van het plan."},
            "almost": {"en": "Right for the example, but with `{call}` your function gives back `{got}`, and it should give back `{want}`. Does it work for a plan of three dishes, one with a dish the kitchen does not know, and an empty plan?",
                       "nl": "Goed voor het voorbeeld, maar met `{call}` geeft je functie `{got}` terug, en het zou `{want}` moeten zijn. Werkt hij ook voor een plan van drie gerechten, een met een gerecht dat de keuken niet kent, en een leeg plan?"},
            "correct": {"en": "Your `orchestrate` lets the plan decide how many parts there are, gives each known dish to a worker, and collects the results: right for all {n} plans of the check, the empty one included.",
                        "nl": "Je `orchestrate` laat het plan bepalen hoeveel delen er zijn, geeft elk bekend gerecht aan een worker, en verzamelt de resultaten: goed voor alle {n} plannen van de check, ook het lege."},
        },
    ),
}

REQUIRED = ["improve", "orchestrate", "pattern"]

CASES = [
    ("improve", START_IMPROVE, "empty", None),
    ("improve", "", "empty", None),
    ("improve", SOLUTION_IMPROVE, "correct", None),
    ("improve", START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n"
                                      "        messages.append({\"role\": \"assistant\", \"content\": answer})\n"
                                      "        messages.append({\"role\": \"user\", \"content\": problem})\n"), "correct", None),
    ("improve", START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n"
                                      "        messages = [{\"role\": \"user\", \"content\": task}, {\"role\": \"assistant\", \"content\": answer},\n"
                                      "                    {\"role\": \"user\", \"content\": problem}]\n"), "correct", None),
    ("improve", _FIRST, "wrong", "first"),
    ("improve", _NO_BACK, "wrong", "unreached"),
    ("improve", _ONLY_PROBLEM, "wrong", "unreached"),
    ("improve", _LAST, "wrong", "last"),
    ("improve", SOLUTION_IMPROVE.replace("            return answer\n", "            return answer.upper()\n"), "wrong", None),
    ("improve", START_IMPROVE.replace("        ...\n", "        if problem is None:\n            return answer\n"
                                      "        messages = messages + [{\"role\": \"assistant\", \"content\": answer},\n"
                                      "                               {\"role\": \"user\", \"content\": problem}]\n"
                                      "        if n == 1:\n            return \"stop\"\n"), "almost", None),
    ("orchestrate", START_ORCHESTRATE, "empty", None),
    ("orchestrate", "", "empty", None),
    ("orchestrate", SOLUTION_ORCHESTRATE, "correct", None),
    ("orchestrate", _ALL, "error", "unknown"),
    ("orchestrate", _FIRST_ONLY, "wrong", "first-only"),
    ("orchestrate", _PLAN_BACK, "wrong", "plan-back"),
    ("orchestrate", START_ORCHESTRATE.replace("    ...\n", "    for dish in plan[\"dishes\"]:\n        if dish in DISHES:\n            results.append(dish)\n"), "wrong", None),
    ("orchestrate", START_ORCHESTRATE.replace("    ...\n", "    for dish in plan[\"dishes\"][:2]:\n        if dish in DISHES:\n            results.append(worker(dish))\n"), "almost", None),
]


# --- the demo: the learner's improve with Qwen3 -----------------------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_improve(text):
    """The learner's improve from exercise 1, when it is right; else None."""
    ex = EXERCISES["improve"]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return run_code(text).names.get("improve")


def qwen_ask(url: str):
    """`ask` for improve on the real model: the messages go to the course's server, the answer's text comes back."""
    import urllib.request

    def ask(messages):
        body = {"model": serve.MODEL, "messages": messages, "max_tokens": 60, "temperature": 0}
        request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())["choices"][0]["message"]["content"]
    return ask


def demo(improve_fn, dish: str, words_: dict, lang: str) -> str:
    """The learner's improve on a tip for a dish, with Qwen3 behind the course's server and the course's check."""
    running, before = None, 0
    try:
        running = serve.model_server()
        before = len(running.log)
        got, error = improve_fn(qwen_ask(running.url), TIP.format(dish=dish, limit=LIMIT), Limit(LIMIT), TRIES), None
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words_['memory'])}</em></p>"
        got, error = None, e
    log = running.log[before:] if running is not None else []
    rows = []
    for n, e in enumerate([e for e in log if e["method"] == "POST" and isinstance(e.get("answer"), dict) and "choices" in e["answer"]], 1):
        text = e["answer"]["choices"][0]["message"].get("content") or ""
        said = Limit(LIMIT)(text)
        rows.append(f'<div style="{_UI};font-size:13px">{html.escape(words_["round"].format(n=n))} {_code(text, "var(--ea-pred-text)")} '
                    f'<span style="color:{"var(--ea-ink-2)" if said is None else "var(--ea-loss-text)"}">'
                    f'{html.escape(words_["ok"] if said is None else said)}</span></div>')
    end = words_["error"].format(e=type(error).__name__ + ": " + str(error)) if error is not None else words_["gave"].format(v=repr(got))
    rows.append(f'<p style="{_UI};font-size:13px">{html.escape(end)}</p>')
    return yours.figure(words_["title"], "".join(rows), alt=words_["title"])
