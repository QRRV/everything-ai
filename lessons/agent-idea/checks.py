# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson agent-idea: the agent loop, checked with stand-ins for the language model that read the
messages and answer as a model would, but always the same. The pantry and its two tools, as the course's run of
Qwen3-0.6B used them (data/course-agent-qwen.json, made by data/tools/record_course_agent_qwen.py with this file), and the
picture of the messages a loop collects."""
import html
import json
import re

from everything_ai.check import FunctionBox, Mistake

COURSE = "data/course-agent-qwen.json"   # the course's runs of Qwen3-0.6B as an agent, with the code of this file

# The pantry: two recipes, and what is in the cupboard, in grams.
RECIPES = {"pancakes": {"flour": 250, "milk": 450, "eggs": 110, "butter": 25},
           "tomato soup": {"tomatoes": 800, "onion": 150, "butter": 30, "stock": 500}}
PANTRY = {"flour": 1000, "milk": 200, "eggs": 0, "butter": 250, "tomatoes": 400, "onion": 300, "stock": 1000}
TASKS = {"pancakes": "What do I have to buy to make pancakes?", "soup": "What do I have to buy to make tomato soup?"}
SHOPPING = {"pancakes": {"milk": 250, "eggs": 110}, "soup": {"tomatoes": 400}}   # what each task really needs


def recipe(args: dict) -> str:
    """The tool `recipe`: the ingredients of a dish, as JSON text, with the grams of each."""
    found = RECIPES.get(str(args.get("dish", "")).lower().strip())
    return json.dumps(found) if found else f"There is no recipe for {args.get('dish')}."


def pantry(args: dict) -> str:
    """The tool `pantry`: how many grams of an ingredient are in the cupboard."""
    return f"{PANTRY.get(str(args.get('item', '')).lower().strip(), 0)} grams"


class Tools(dict):
    """The tools of a task: a dict from the name of a tool to its function, which gets the arguments as one dict and
    gives back text. It shows itself by its name, as the check writes the call. A tool the model makes up gives a
    message back instead of a KeyError, as a careful program does."""

    def __init__(self, name: str, tools: dict):
        super().__init__(tools)
        self.name = name

    def __missing__(self, key):
        return lambda args: f"There is no tool {key}."

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Tools(self.name, dict(self))


class StandIn:
    """A stand-in for the language model: `ask(messages)` reads the messages, as a model does, and gives the next
    message of the assistant, {"role": "assistant", "content": ..., "tool_calls": [...]}, always the same for the same
    messages. It shows itself by its name."""

    def __init__(self, name: str, brain):
        self.name, self.brain, self.asked = name, brain, 0

    def __call__(self, messages):
        if not isinstance(messages, list):
            raise TypeError("ask takes the list of messages, not " + type(messages).__name__)
        self.asked += 1
        return self.brain(messages)

    def __repr__(self):
        return self.name


def _say(text: str) -> dict:
    return {"role": "assistant", "content": text, "tool_calls": []}


def _calls(*calls) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"name": n, "arguments": a} for n, a in calls]}


def _results(messages) -> list:
    return [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


def _cook(messages) -> dict:
    """What a model that uses the pantry well does: look up the recipe, then each ingredient, then answer."""
    task = messages[0].get("content", "") if messages and isinstance(messages[0], dict) else ""
    dish = "pancakes" if "pancake" in task else "tomato soup"
    results = _results(messages)
    if not results:
        return _calls(("recipe", {"dish": dish}))
    try:
        needed = json.loads(results[0])
    except (TypeError, ValueError):
        return _say("I could not read the recipe.")
    have = results[1:]
    if not have:
        return _calls(*(("pantry", {"item": item}) for item in needed))
    grams = {}
    for item, text in zip(needed, have):
        try:
            grams[item] = int(str(text).split()[0])
        except (ValueError, IndexError):
            return _say(f"I could not read how much {item} there is.")
    buy = [f"{g - grams.get(item, 0)} grams of {item}" for item, g in needed.items() if grams.get(item, 0) < g]
    return _say("Buy " + " and ".join(buy) + "." if buy else "You have everything you need.")


def _chatty(messages) -> dict:
    return _say("Hello! A chat model without tools just answers. How can I help you?")


def _sums(messages) -> dict:
    """Two sums at once, with a tool for each, then the answer."""
    results = _results(messages)
    if not results:
        return _calls(("add", {"a": 12, "b": 30}), ("add", {"a": 7, "b": 8}))
    return _say(f"12 + 30 = {results[0]} and 7 + 8 = {results[1] if len(results) > 1 else '?'}.")


def _clock_watcher(messages) -> dict:
    """A model that never stops: it asks the time again and again. Only max_steps stops it."""
    return _calls(("clock", {}))


PANTRY_TOOLS = {"recipe": recipe, "pantry": pantry}


def kitchen() -> tuple:
    """The stand-in for the model and the two tools of the lesson's worked example."""
    return StandIn("ask", _cook), Tools("tools", PANTRY_TOOLS)


def cases() -> tuple[list, list]:
    """The example of the lesson, and the other cases of the check, each (ask, tools, task, max_steps)."""
    visible = [(StandIn("ask", _cook), Tools("tools", PANTRY_TOOLS), TASKS["pancakes"], 5)]
    hidden = [(StandIn("ask", _cook), Tools("tools", PANTRY_TOOLS), TASKS["soup"], 5),
              (StandIn("ask_chatbot", _chatty), Tools("no_tools", {}), "Hi there!", 5),
              (StandIn("ask_sums", _sums), Tools("calculator", {"add": lambda args: str(args["a"] + args["b"])}),
               "What are 12 + 30 and 7 + 8?", 4),
              (StandIn("ask_forever", _clock_watcher), Tools("clock", {"clock": lambda args: "12:00"}), "What time is it?", 3)]
    return visible, hidden


def flow(messages) -> str:
    """The messages of a run in one line: who wrote each, and which tools an assistant message calls."""
    if not isinstance(messages, list):
        return repr(messages)
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            parts.append(repr(m)[:40])
            continue
        role = m.get("role", "?")
        calls = [c.get("name", "?") for c in m.get("tool_calls") or [] if isinstance(c, dict)]
        parts.append(f"{role} ({', '.join(calls)})" if calls else str(role))
    return " → ".join(parts)


_START = ("def run_agent(ask, tools, task, max_steps):\n    messages = [{\"role\": \"user\", \"content\": task}]\n"
          "    for step in range(max_steps):\n        reply = ...\n        messages.append(reply)\n"
          "        if reply[\"tool_calls\"] == []:\n            ...\n        for call in reply[\"tool_calls\"]:\n"
          "            result = ...\n            messages.append({\"role\": \"tool\", \"content\": result})\n    return messages\n")
_SOLUTION = (_START.replace("reply = ...", "reply = ask(messages)").replace("            ...\n", "            return messages\n")
             .replace("result = ...", 'result = tools[call["name"]](call["arguments"])'))


def _picture(seen) -> str:
    """The messages the learner's loop collects on the pantry task of the lesson, one row each, beside the course's
    where they differ: the task, each answer of the stand-in with the tools it calls, each tool result."""
    from everything_ai import check as checking, yours
    mine = seen.mine
    fn = mine.names.get("run_agent") if mine else None
    if not callable(fn):
        return ""
    (ask, tools, task, steps), = cases()[0]
    try:
        got = checking.with_time_limit(lambda: fn(ask, tools, task, steps))
    except Exception:
        return ""
    if not isinstance(got, list):
        return ""
    lang = seen.lang
    title = {"en": "The messages your run_agent collects for the pancakes", "nl": "De berichten die jouw run_agent verzamelt voor de pannenkoeken"}[lang]
    return yours.figure(title, messages_html(got, lang), alt=flow(got))


_WHO = {"user": {"en": "task", "nl": "taak"}, "assistant": {"en": "model", "nl": "model"}, "tool": {"en": "tool", "nl": "tool"}}


def messages_html(messages: list, lang: str, limit: int = 14) -> str:
    """Messages as rows: who wrote each (the task, the model, a tool), and its text; the tools an answer of the model
    calls, with their arguments, in code. What a model writes has the colour of a prediction."""
    rows = []
    for m in messages[:limit]:
        if not isinstance(m, dict):
            rows.append(f'<div style="color:var(--ea-ink-2)">{html.escape(repr(m)[:120])}</div>')
            continue
        role = str(m.get("role", "?"))
        model = role == "assistant"
        colour = "var(--ea-pred-text)" if model else "var(--ea-ink)"
        text = html.escape(str(m.get("content") or ""))
        calls = "".join(f'<code style="display:block;color:{colour}">{html.escape(str(c.get("name")))}'
                        f'({html.escape(json.dumps(c.get("arguments"), ensure_ascii=False))})</code>'
                        for c in m.get("tool_calls") or [] if isinstance(c, dict))
        who = _WHO.get(role, {}).get(lang, role)
        rows.append(f'<div style="display:flex;gap:0.6rem;padding:0.3rem 0;border-top:1px solid var(--ea-line)">'
                    f'<span style="flex:0 0 3.6rem;font-family:var(--ea-font-ui);font-size:0.8rem;font-weight:600;'
                    f'color:{colour}">{html.escape(who)}</span>'
                    f'<span style="flex:1 1 auto;font-size:0.88rem;color:{colour};overflow-wrap:anywhere">{text}{calls}</span></div>')
    if len(messages) > limit:
        more = {"en": f"and {len(messages) - limit} more", "nl": f"en nog {len(messages) - limit}"}[lang]
        rows.append(f'<div style="color:var(--ea-ink-2);font-size:0.85rem;padding-top:0.3rem">{more}</div>')
    return f'<div style="font-family:var(--ea-font-ui)">{"".join(rows)}</div>'


# --- the real model: Qwen3-0.6B (chat) with the two tools of the pantry -------------------------------------------

SYSTEM = ("You are a kitchen assistant with two tools. Use the tools to look up the recipe and how much of each ingredient "
          "is in the pantry. Do not ask the user: the tools have the answers. When you know enough, answer with what to buy.")
QWEN_TASK = "I want to make pancakes. Which ingredients do I have to buy, and how many grams of each?"   # spelled out for Qwen3
THINKING = False      # non-thinking mode: no reasoning before the tool calls (everything_ai.chat)
MAX_TOKENS = 300      # per message of the model
SHOWN_RUN = 1         # the run the slider opens at, and whose seed the demo uses


def specs() -> list:
    """The two tools as the model reads them: the name, what it does, and its argument (everything_ai.chat.tool)."""
    from everything_ai import chat
    return [chat.tool("recipe", "The ingredients of a dish, with the grams of each.", dish="the name of the dish"),
            chat.tool("pantry", "How many grams of an ingredient are in the pantry now.", item="the name of the ingredient")]


def prompt_example() -> str:
    """The text Qwen3 reads for the pantry task, up to the start of its answer: the system message with the
    instructions and the tools, and the task."""
    from everything_ai import chat
    return chat.prompt([{"role": "system", "content": SYSTEM}, {"role": "user", "content": QWEN_TASK}], specs(), THINKING)


def course() -> dict:
    """The course's runs of Qwen3-0.6B on the pantry task (data/tools/record_course_agent_qwen.py)."""
    from everything_ai import paths
    return json.loads((paths.ROOT / COURSE).read_text(encoding="utf-8"))


def number(x: float, digits: int, lang: str) -> str:
    text_ = f"{x:,.{digits}f}"
    if lang == "nl":
        text_ = text_.replace(",", "_").replace(".", ",").replace("_", ".")
    return text_


def seconds_of(run: dict) -> float:
    return sum(step["seconds"] for step in run["steps"])


def downloaded() -> bool:
    from everything_ai import chat
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    from everything_ai import chat
    chat.download_or_stop(button, texts)


def qwen_ask(model, seed: int, heard=None):
    """`ask` with Qwen3: every message of the learner's loop goes to the model after the system message, with the two
    tools; its answer comes back as a message, {"role": "assistant", "content": ..., "tool_calls": [...]}. The n-th
    question of a run is sampled with seed 100 * seed + n, so a run is the same every time on the same computer.
    `heard(text, n)` hears what the model has written so far of its n-th message. `ask.steps` keeps the tokens and
    seconds of every message."""
    def ask(messages):
        if not isinstance(messages, list):
            raise TypeError("ask takes the list of messages, not " + type(messages).__name__)
        n = len(ask.steps)
        a = model.answer([{"role": "system", "content": SYSTEM}] + messages, specs(), max_tokens=MAX_TOKENS, seed=100 * seed + n,
                         thinking=THINKING, heard=None if heard is None else (lambda text: heard(text, n)))
        ask.steps.append({"prompt_tokens": a["prompt_tokens"], "tokens": a["tokens"], "seconds": a["seconds"], "finish": a["finish"],
                          "text": a["text"]})
        m = a["message"]
        return {"role": "assistant", "content": m["content"], "tool_calls": m["tool_calls"]}
    ask.steps = []
    return ask


def run_agent_with_qwen(run_agent, model, seed: int, heard=None) -> dict:
    """One run of `run_agent` with Qwen3 on the pancakes, with the real tools: its messages and the steps of the model."""
    ask = qwen_ask(model, seed, heard)
    messages = run_agent(ask, Tools("tools", PANTRY_TOOLS), QWEN_TASK, 6)
    return {"seed": seed, "messages": messages, "steps": ask.steps}


# --- the course's runs, as the lesson tells them ------------------------------------------------------------------------

_AMOUNT = re.compile(r"\d+\s*(?:g\b|grams?\b|cups?\b|tbsp\b|tablespoons?\b|tsp\b|teaspoons?\b|ml\b)", re.I)


def kind_of(text: str) -> str:
    """What an answer without tools does: "invents" when it gives amounts of ingredients, "asks" when it only asks
    back, "other" otherwise."""
    if _AMOUNT.search(text):
        return "invents"
    return "asks" if "?" in text else "other"


def calls_of(run: dict) -> list:
    """The names of the tools a run called, in order."""
    return [c["name"] for m in run["messages"] if m.get("role") == "assistant" for c in m.get("tool_calls") or []]


def final_of(run: dict) -> str | None:
    """The model's answer at the end of a run, or None when the step limit stopped it."""
    last = run["messages"][-1]
    return last.get("content", "") if last.get("role") == "assistant" and not last.get("tool_calls") else None


def verdict(run: dict) -> str:
    """"right" when the run looked in the pantry and answers 250 grams of milk and 110 grams of eggs; "no-pantry" when
    it never looked in the pantry; "unfinished" when the step limit stopped it; "wrong" otherwise."""
    final = final_of(run)
    if final is None:
        return "unfinished"
    if "pantry" not in calls_of(run):
        return "no-pantry"
    lines = final.lower().splitlines()
    right = (any("milk" in x and re.search(r"\b250\b", x) for x in lines) and any("egg" in x and re.search(r"\b110\b", x) for x in lines)
             and not any(w in final.lower() for w in ("flour", "butter")))
    return "right" if right else "wrong"


_NOTHING = re.compile(r"don't need to buy|do not need to buy|no additional|already (?:purchased|have)", re.I)
_HAVE = re.compile(r"pantry|you (?:currently )?have|already", re.I)


def answer_kind(run: dict) -> str:
    """What the answer of a run that never looked in the pantry says: "recipe" when it gives the recipe as the list to
    buy, flour's 250 grams included; "nothing" when it says you need to buy nothing; "other" otherwise."""
    final = final_of(run) or ""
    if _NOTHING.search(final):
        return "nothing"
    return "recipe" if re.search(r"flour\W+250|250\s*g(?:rams)? of flour", final, re.I) else "other"


def _counts(R: dict) -> dict:
    runs = R["runs"]
    return {"n": len(runs), "recipe_first": sum(calls_of(r)[:1] == ["recipe"] for r in runs),
            "pantry": sum("pantry" in calls_of(r) for r in runs), "right": sum(verdict(r) == "right" for r in runs),
            "recipe": sum(answer_kind(r) == "recipe" for r in runs), "nothing": sum(answer_kind(r) == "nothing" for r in runs),
            "claims": sum(bool(_HAVE.search(final_of(r) or "")) for r in runs), "steps": sorted({len(r["steps"]) for r in runs})}


def _times(k: int, n: int, lang: str) -> str:
    if k == n:
        return {"en": f"in all {n} runs", "nl": f"in alle {n} runs"}[lang]
    if k == 0:
        return {"en": "in none of the runs", "nl": "in geen enkele run"}[lang]
    return {"en": f"in {k} of {n} runs", "nl": f"in {k} van de {n} runs"}[lang]


def no_tools_reveal(lang: str) -> str:
    """What the guess reveals; written, as the test holds it to, for answers that all made something up."""
    R = course()
    kinds = [kind_of(a["text"]) for a in R["without_tools"]]
    n, k = len(kinds), kinds.count("invents")
    times = {"en": "every time" if k == n else f"{k} of the {n} times", "nl": "elke keer" if k == n else f"{k} van de {n} keer"}[lang]
    if lang == "en":
        return (f"**It made up an answer {times}:** a list of ingredients with amounts in cups, spoons or grams, as if "
                "that were what you have to buy. It cannot know what is in your pantry, and not once did it say so. A "
                "language model writes a likely answer; without a way to look, a likely answer is all it has.")
    return (f"**Het verzon {times} een antwoord:** een lijst ingrediënten met hoeveelheden in kopjes, lepels of grammen, "
            "alsof je dat moest kopen. Het kan niet weten wat er in je voorraadkast staat, en zei dat geen enkele keer. Een "
            "language model schrijft een waarschijnlijk antwoord; zonder een manier om te kijken, heeft het niets anders.")


def runs_summary(lang: str) -> str:
    """The course's runs in two sentences. Written for what they show, that the model never looked in the pantry: a
    test holds the text to that."""
    c = _counts(course())
    if lang == "en":
        return (f"{_times(c['recipe_first'], c['n'], lang).capitalize()} the model first looked up the recipe, as the "
                "instructions say. It never looked in the pantry, and none of its answers was right.")
    return (f"{_times(c['recipe_first'], c['n'], lang).capitalize()} zocht het model eerst het recept op, zoals de instructies "
            "zeggen. In de voorraadkast keek het nooit, en geen van zijn antwoorden klopte.")


def run_frame(run: dict, i: int, n: int, lang: str) -> tuple:
    """A run of the course on the slider: its messages, and what happened in words."""
    calls, v = calls_of(run), verdict(run)
    named = ", ".join(f"`{c}`" for c in calls)
    if lang == "en":
        did = f"the model called {named}, and then answered." if calls else "the model answered right away, without a tool."
        said = {"no-pantry": {"recipe": "It never looked in the pantry: its answer lists the recipe's grams of flour, milk and eggs as what to buy, though there is plenty of flour, and what it says about the pantry, it made up.",
                              "nothing": "It never looked in the pantry, and still its answer says you need to buy nothing, though there are no eggs.",
                              "other": "It never looked in the pantry, so its answer cannot be right."}[answer_kind(run)],
                "right": "Its answer is right: 250 grams of milk and 110 grams of eggs.",
                "wrong": "It looked in the pantry, but its answer is still not the right list.",
                "unfinished": "It was still calling tools when the step limit stopped it."}[v]
        head = f"Run {i} of {n}, seed {run['seed']}"
        sentence = f"**Run {i}:** {did} {said}"
    else:
        did = f"het model riep {named} aan, en antwoordde toen." if calls else "het model antwoordde meteen, zonder tool."
        said = {"no-pantry": {"recipe": "Het keek nooit in de voorraadkast: zijn antwoord noemt de grammen bloem, melk en eieren uit het recept als wat je moet kopen, terwijl er genoeg bloem is, en wat het over de voorraadkast zegt, heeft het verzonnen.",
                              "nothing": "Het keek nooit in de voorraadkast, en toch zegt zijn antwoord dat je niets hoeft te kopen, terwijl er geen eieren zijn.",
                              "other": "Het keek nooit in de voorraadkast, dus zijn antwoord kan niet kloppen."}[answer_kind(run)],
                "right": "Zijn antwoord klopt: 250 gram melk en 110 gram eieren.",
                "wrong": "Het keek in de voorraadkast, maar zijn antwoord is nog steeds niet de goede lijst.",
                "unfinished": "Het riep nog tools aan toen de stappenlimiet het stopte."}[v]
        head = f"Run {i} van {n}, seed {run['seed']}"
        sentence = f"**Run {i}:** {did} {said}"
    body = (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;font-weight:600;color:var(--ea-ink-2);margin-bottom:0.2rem">'
            f'{html.escape(head)}</div>' + messages_html(run["messages"], lang, limit=20))
    return body, sentence


def runs_after(lang: str) -> str:
    """What the runs show, with their numbers; written, as runs_summary, for runs that never looked in the pantry."""
    c = _counts(course())
    if lang == "en":
        return (f"The loop did its part every time: it ran the tool the model called, and gave the model the result. The model "
                f"is the weak link. It never looked in the pantry, though the instructions say to, and still, "
                f"{_times(c['claims'], c['n'], lang)}, its answer spoke of what you have, as if it had looked. In {c['recipe']} of the {c['n']} "
                f"runs it gave the recipe's grams of flour, milk and eggs as the shopping list, flour too, of which you have 1,000 grams; in "
                f"{c['nothing']} it said you need to buy nothing at all, though there are no eggs. Qwen3-0.6B is a small model, "
                "small enough for your computer; agents in products run on much larger models, which do whole tasks like this "
                "far more often, and still not always. So what an agent did matters as much as its answer, and that is why "
                "`run_agent` gives back every message: in them you see which tools it called, and which it skipped. The "
                "stand-in of exercise 1 does what a model that uses its tools well does.")
    return (f"De loop deed elke keer zijn deel: hij voerde de tool uit die het model aanriep, en gaf het model het resultaat. "
            f"Het model is de zwakke schakel. Het keek nooit in de voorraadkast, hoewel de instructies dat zeggen, en toch had "
            f"zijn antwoord het {_times(c['claims'], c['n'], lang)} over wat je hebt, alsof het gekeken had. In {c['recipe']} "
            f"van de {c['n']} runs gaf het de grammen bloem, melk en eieren uit het recept als boodschappenlijst, ook de bloem, terwijl je daar 1.000 gram van hebt; in "
            f"{c['nothing']} zei het dat je helemaal niets hoeft te kopen, terwijl er geen eieren zijn. Qwen3-0.6B is een klein "
            "model, klein genoeg voor je computer; agents in producten draaien op veel grotere modellen, die zulke hele taken "
            "veel vaker goed doen, en nog steeds niet altijd. Wat een agent deed, is dus net zo belangrijk als zijn antwoord, "
            "en daarom geeft `run_agent` elk bericht terug: daarin zie je welke tools hij aanriep, en welke hij oversloeg. De "
            "stand-in van oefening 1 doet wat een model doet dat zijn tools goed gebruikt.")


class _ModelFailed(Exception):
    """Qwen3 itself failed while the learner's loop asked it: "memory" or "load"."""


def run_live(run_agent, show, words: dict, lang: str):
    """The demo: the learner's run_agent with Qwen3 and the seed of run SHOWN_RUN, with the messages so far and what
    the model writes now shown by `show(html)`, at most five times a second. The run, or what went wrong: "load",
    "memory" or "agent"."""
    import time

    from everything_ai import chat
    show(live_picture({"messages": [], "writing": None, "note": words["loading"]}, words, lang))
    try:
        model = chat.loaded()
    except MemoryError:
        return "memory"
    except Exception:
        return "load"
    so_far = {"messages": [], "shown": 0.0}

    def heard(text, n):
        if time.perf_counter() - so_far["shown"] > 0.2:
            so_far["shown"] = time.perf_counter()
            show(live_picture({"messages": so_far["messages"], "writing": text}, words, lang))

    ask = qwen_ask(model, course()["runs"][SHOWN_RUN - 1]["seed"], heard)

    def watched(messages):
        so_far["messages"] = list(messages) if isinstance(messages, list) else []
        if not isinstance(messages, list):
            return ask(messages)            # the TypeError of the stand-ins: the learner's loop gave no list
        try:
            return ask(messages)
        except Exception as e:              # the model failed, not the learner's loop
            raise _ModelFailed("memory" if isinstance(e, MemoryError) or "out of memory" in str(e).lower() else "load") from e
    try:
        messages = run_agent(watched, Tools("tools", PANTRY_TOOLS), QWEN_TASK, 6)
    except _ModelFailed as e:
        return e.args[0]
    except MemoryError:
        return "memory"
    except Exception:
        return "agent"
    if not isinstance(messages, list):
        return "agent"
    return {"messages": messages, "steps": ask.steps}


def live_picture(run: dict, words: dict, lang: str) -> str:
    """The messages of the learner's run so far, and, while the model writes, its message so far."""
    from everything_ai import yours
    body = messages_html(run["messages"], lang, limit=20)
    if run.get("note"):
        body += f'<div style="padding:0.3rem 0;font-family:var(--ea-font-ui);font-size:0.85rem;color:var(--ea-ink-2)">{html.escape(run["note"])}</div>'
    if run.get("writing") is not None:
        body += (f'<div style="padding:0.3rem 0;border-top:1px solid var(--ea-line);font-family:var(--ea-font-code);font-size:0.85rem;'
                 f'white-space:pre-wrap;color:var(--ea-pred-text)">{html.escape(words["writing"])}\n{html.escape(run["writing"])}</div>')
    return yours.figure(words["title"], body, alt=words["alt"])


# The lesson is done when these are correct. "who-runs" and "rl-agent" are choice questions in the lesson.
REQUIRED = ["run-agent", "who-runs", "rl-agent"]

_V, _H = cases()

EXERCISES = {
    "run-agent": FunctionBox(
        name="run-agent",
        picture=_picture,
        function="run_agent",
        start=_START,
        solution=_SOLUTION,
        visible=_V,
        hidden=_H,
        show=flow,
        mistakes=[
            Mistake("dots-left", syntax=r"^[^#\n]*\.\.\.",
                    en="There is still a `...` in your function. Replace each one with your own code.",
                    nl="Er staat nog een `...` in je functie. Vervang ze door je eigen code."),
            Mistake("ask-task", error=lambda e: isinstance(e, TypeError) and "list of messages" in str(e),
                    en="With `{call}`, your function stops with a `TypeError`: `ask` gets something other than the list `messages`. The model reads the whole conversation so far, so give it `messages`, not only the task.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: `ask` krijgt iets anders dan de lijst `messages`. Het model leest het hele gesprek tot nu toe, dus geef het `messages`, niet alleen de taak."),
            Mistake("unpacked", error=lambda e: isinstance(e, TypeError) and "__call__" not in str(e)
                    and ("unexpected keyword argument" in str(e) or "positional argument" in str(e)),
                    en="With `{call}`, your function stops with a `TypeError`: a tool gets the wrong number of values. Every tool takes one value: the dict `call[\"arguments\"]`.",
                    nl="Met `{call}` stopt je functie met een `TypeError`: een tool krijgt het verkeerde aantal waarden. Elke tool neemt één waarde: de dict `call[\"arguments\"]`."),
            Mistake("whole-call", wrong=_SOLUTION.replace('(call["arguments"])', '(call)'),
                    en="`{call}` gives `{got}`: each tool gets the whole call, with its name. A tool needs only the arguments of the call.",
                    nl="`{call}` geeft `{got}`: elke tool krijgt de hele call, met zijn naam. Een tool heeft alleen de argumenten van de call nodig."),
            Mistake("no-stop", wrong=_SOLUTION.replace("            return messages\n", "            pass\n"),
                    en="`{call}` gives `{got}`: after the model's answer, your loop asks it again. When the answer calls no tool, the task is done: stop right there, and give back the list.",
                    nl="`{call}` geeft `{got}`: na het antwoord van het model vraagt je loop het opnieuw. Als het antwoord geen tool aanroept, is de taak klaar: stop daar meteen, en geef de lijst terug."),
            Mistake("one-more", wrong=_SOLUTION.replace("range(max_steps)", "range(max_steps + 1)"), state="almost",
                    en="`{call}` gives `{got}`: your loop asks the model once more than `max_steps` allows. Ask it at most `max_steps` times.",
                    nl="`{call}` geeft `{got}`: je loop vraagt het model één keer vaker dan `max_steps` toestaat. Vraag het hooguit `max_steps` keer."),
            Mistake("answer-only", matches=lambda got, text: isinstance(got, (str, dict)),
                    en="`{call}` gives one message: only the last answer. Give back the whole list `messages`, so that you can see every step the agent took.",
                    nl="`{call}` geeft één bericht: alleen het laatste antwoord. Geef de hele lijst `messages` terug, dan zie je elke stap die de agent zette."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the three `...`: ask the model for its next message, give back `messages` when it calls no tool, and run each tool it calls.",
                      "nl": "Vervang de drie `...`: vraag het model om zijn volgende bericht, geef `messages` terug als het geen tool aanroept, en voer elke tool uit die het aanroept."},
            "cleared": {"en": "There is no code in the box any more. Start again with `def run_agent(ask, tools, task, max_steps):` and, under it, indented, your lines.",
                        "nl": "Er staat geen code meer in het vak. Begin opnieuw met `def run_agent(ask, tools, task, max_steps):`, met daaronder, ingesprongen, je regels."},
            "slow": {"en": "Your function is still not done after a few seconds. Does your loop stop after `max_steps` questions, also when the model keeps calling tools?",
                     "nl": "Je functie is na een paar seconden nog niet klaar. Stopt je loop na `max_steps` vragen, ook als het model tools blijft aanroepen?"},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every step add the model's message, and after it one tool message for each tool it calls?",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Voegt elke stap het bericht van het model toe, en daarna één toolbericht voor elke tool die het aanroept?"},
            "almost": {"en": "Right for the pancakes. But `{call}` gives `{got}`, and it should give `{want}`. Does your loop work for every model and every set of tools it gets, and stop after `max_steps`?",
                       "nl": "Goed voor de pannenkoeken. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je loop voor elk model en elke set tools die hij krijgt, en stopt hij na `max_steps`?"},
            "correct": {"en": "Your loop is right in all {n} cases: with a model that uses two tools, one that just chats, one that calls two tools at once, and one that never stops, which `max_steps` stops.",
                        "nl": "Je loop klopt in alle {n} gevallen: met een model dat twee tools gebruikt, een dat alleen praat, een dat twee tools tegelijk aanroept, en een dat nooit stopt, dat `max_steps` tegenhoudt."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("run-agent", _START, "empty", None),
    ("run-agent", _SOLUTION, "correct", None),
    ("run-agent", _SOLUTION.replace("for step in range(max_steps):", "step = 0\n    while step < max_steps:\n        step = step + 1"),
     "correct", None),
    ("run-agent", _SOLUTION.replace("            return messages\n", "            pass\n"), "wrong", "no-stop"),
    ("run-agent", _SOLUTION.replace("ask(messages)", "ask(task)"), "error", "ask-task"),
    ("run-agent", _SOLUTION.replace("range(max_steps)", "range(max_steps + 1)"), "almost", "one-more"),
    ("run-agent", _SOLUTION.replace('tools[call["name"]](call["arguments"])', 'tools[call["name"]](**call["arguments"])'), "error", "unpacked"),
    ("run-agent", _SOLUTION.replace("    return messages", "    return messages[-1][\"content\"]"), "wrong", "answer-only"),
    ("run-agent", _SOLUTION.replace('(call["arguments"])', '(call)'), "wrong", "whole-call"),
    ("run-agent", _SOLUTION.replace("            return messages\n", "            return reply\n"), "wrong", "answer-only"),
    ("run-agent", _SOLUTION.replace("ask(messages)", "ask()"), "error", None),
    ("run-agent", _SOLUTION.replace("for step in range(max_steps):", "while True:"), "slow", None),
    ("run-agent", _SOLUTION.replace("        messages.append(reply)\n", ""), "wrong", None),
]
