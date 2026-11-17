# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-human-in-loop: the bank's assistant as an agent that acts for its customers, with seven tools and a
day of ten requests whose steps the course wrote out by hand, as an agent might plan them, with four steps that are
wrong. A person answers the steps the agent asks about: yes, a changed step, or no. Two functions to write
(`needs_approval`, the rule for when to ask, and `run_steps`, which asks and goes on as the person says), two choices,
and the pictures: the tools with the learner's rule, the day with the learner's run, and three ways of asking."""
import copy
import html

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake

# --- the tools, the day and the person ---------------------------------------------------------------------------------

class _Tools(dict):
    """The tools, written in the check's messages by their name."""

    def __repr__(self):
        return "TOOLS"


# what each tool does to the world: costs money, sends something out of the bank, can be undone
TOOLS = _Tools({
    "look_up": {"money": False, "sends": False, "can_undo": True},
    "block_card": {"money": False, "sends": False, "can_undo": True},
    "unblock_card": {"money": False, "sends": False, "can_undo": True},
    "order_card": {"money": True, "sends": False, "can_undo": True},
    "refund": {"money": True, "sends": False, "can_undo": False},
    "send_email": {"money": False, "sends": True, "can_undo": False},
    "close_account": {"money": False, "sends": False, "can_undo": False},
})
DOES = {
    "en": {"look_up": "reads the account: balance, transactions, card", "block_card": "blocks the card; unblock_card undoes it",
           "unblock_card": "unblocks the card", "order_card": "orders a new card, which the bank pays for and delivers; the order can be canceled",
           "refund": "pays an amount back to the customer", "send_email": "sends an email", "close_account": "closes the account for good"},
    "nl": {"look_up": "leest de rekening: saldo, transacties, kaart", "block_card": "blokkeert de kaart; unblock_card maakt het ongedaan",
           "unblock_card": "deblokkeert de kaart", "order_card": "bestelt een nieuwe kaart, die de bank betaalt en bezorgt; de bestelling kan worden geannuleerd",
           "refund": "betaalt een bedrag terug aan de klant", "send_email": "verstuurt een e-mail", "close_account": "sluit de rekening voorgoed"},
}


class Named(list):
    """A list of the lesson's, written in the check's messages by its name."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return [copy.deepcopy(x, memo) for x in self]


def _step(task: int, k: int, tool: str, **arguments) -> dict:
    return {"id": f"{task}.{k}", "tool": tool, "arguments": arguments}


# ten requests of one day, with the steps the course wrote out for each, as an agent might plan them
DAY = [
    {"message": "My card was stolen yesterday.",
     "steps": [_step(1, 1, "look_up", what="card"), _step(1, 2, "block_card"), _step(1, 3, "order_card"),
               _step(1, 4, "send_email", to="customer", about="new card")]},
    {"message": "I was charged twice for the same coffee.",
     "steps": [_step(2, 1, "look_up", what="transactions"), _step(2, 2, "refund", amount=3.40)]},
    {"message": "Please refund the 25 euros for the shoes, they never arrived.",
     "steps": [_step(3, 1, "look_up", what="transactions"), _step(3, 2, "refund", amount=250.00)]},
    {"message": "What is my balance?",
     "steps": [_step(4, 1, "look_up", what="balance")]},
    {"message": "Close my account, I am moving abroad.",
     "steps": [_step(5, 1, "look_up", what="balance"), _step(5, 2, "close_account"), _step(5, 3, "send_email", to="customer", about="account closed")]},
    {"message": "I found my card again, it was in my coat.",
     "steps": [_step(6, 1, "block_card")]},
    {"message": "Send me my statement for October.",
     "steps": [_step(7, 1, "look_up", what="transactions"), _step(7, 2, "send_email", to="customer", about="statement")]},
    {"message": "Why was my transfer declined?",
     "steps": [_step(8, 1, "look_up", what="transactions")]},
    {"message": "Refund the subscription I canceled last month.",
     "steps": [_step(9, 1, "look_up", what="transactions"), _step(9, 2, "refund", amount=12.99), _step(9, 3, "send_email", to="customer", about="refund")]},
    {"message": "Email my receipts to my new address, receipts@pay-fast.example",
     "steps": [_step(10, 1, "send_email", to="receipts@pay-fast.example", about="receipts")]},
]
STEPS = Named([s for t in DAY for s in t["steps"]], "STEPS")

# the wrong steps, as the course wrote them, and what the person answers when asked about them; asked about any other
# step, the person says yes
WRONG = {"3.2": "refund", "5.2": "close", "6.1": "block", "10.1": "address"}
_CHANGED = {"3.2": _step(3, 2, "refund", amount=25.00)}
ANSWERS = {"3.2": _CHANGED["3.2"], "5.2": "no", "6.1": "no", "10.1": "no"}


class Person:
    """Someone at the bank who answers the agent: "yes", "no", or the step as they changed it."""

    def __init__(self, answers: dict, name: str, other: str = "yes"):
        self.answers, self.name, self.other = answers, name, other

    def __call__(self, step):
        key = step.get("id") if isinstance(step, dict) else None
        return copy.deepcopy(self.answers.get(key, self.other))

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self


PERSON = Person(ANSWERS, "PERSON")


def needs_approval(step, tools):
    tool = tools[step["tool"]]
    return tool["money"] or tool["sends"] or not tool["can_undo"]


def run_steps(steps, ask):
    done = []
    asks = 0
    for step in steps:
        if needs_approval(step, TOOLS):
            asks = asks + 1
            answer = ask(step)
            if answer == "no":
                break
            if answer != "yes":
                step = answer
        done.append(step)
    return done, asks


def run_day(when, person=PERSON) -> dict:
    """The day with a way of asking: `when(step)` says whether to ask the person. What ran, how often the person was
    asked, and the wrong steps that ran as the agent planned them."""
    ran, answers = [], []
    for task in DAY:
        for step in task["steps"]:
            if when(step):
                answer = person(step)
                answers.append(answer if answer in ("yes", "no") else "changed")
                if answer == "no":
                    break
                if answer != "yes":
                    step = answer
            ran.append(step)
    planned = {s["id"]: s for s in STEPS}
    wrong_ran = [s["id"] for s in ran if s["id"] in WRONG and s == planned[s["id"]]]
    return {"ran": ran, "asks": len(answers), "answers": {a: answers.count(a) for a in ("yes", "changed", "no")}, "wrong_ran": wrong_ran}


WAYS = {"never": lambda s: False, "rule": lambda s: needs_approval(s, TOOLS), "every": lambda s: True}


def findings() -> dict:
    runs = {k: run_day(w) for k, w in WAYS.items()}
    return {
        "tasks": len(DAY), "steps": len(STEPS), "tools": len(TOOLS), "wrong": len(WRONG),
        "ask_tools": [t for t in TOOLS if needs_approval({"tool": t}, TOOLS)],
        "asks": {k: r["asks"] for k, r in runs.items()}, "wrong_ran": {k: r["wrong_ran"] for k, r in runs.items()},
        "ran": {k: len(r["ran"]) for k, r in runs.items()},
        "answers": {k: r["answers"] for k, r in runs.items()},
    }


# --- pictures ------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_MONO = "font-family:var(--ea-font-code)"

_WORDS = {
    "en": {"gate": ["costs money?", "sends something?", "cannot be undone?"], "step": "a step of the agent", "runs": "runs",
           "ask": "ask a person", "yes": "yes", "change": "changed", "no": "no", "stop": "the agent stops and says a person will reply",
           "any": "one of them: yes", "none": "none: runs at once",
           "flow_title": "Before each step: ask, or run",
           "flow_alt": "A step of the agent goes past three questions: does it cost money, does it send something, can it not be undone. If none is true, the step runs. If one is, a person is asked first: yes runs the step, a changed step runs as changed, and no stops the agent, which tells the customer that a person will reply.",
           "tools_title": "Your needs_approval(step, tools) on the seven tools",
           "tools_alt": "A table with a row for each tool of the bank's agent: what it does, whether it costs money, sends something or can be undone, and what your function says: ask first, or run at once.",
           "tools_head": ["tool", "what it does", "money", "sends", "can undo", "your function"],
           "asks": "ask first", "at_once": "runs at once", "error": "no True or False",
           "tools_sum": "Your function asks before {k} of the {n} tools: {names}.", "tools_none": "Your function asks before none of the {n} tools.", "tools_failed": "For {k} of the {n} tools your function gives no True or False.",
           "day_title": "Your run_steps(steps, ask) on the day, with PERSON",
           "day_alt": "A row for each of the ten messages of the day: the customer's message and its steps, marked by what happened: ran at once, asked and the person said yes, ran as the person changed it, the person said no, or did not run.",
           "key": ["ran at once", "asked: yes", "asked: changed", "asked: no", "did not run"],
           "day_sum": "Your function ran {r} of the {n} steps and asked the person {a} times.", "day_failed": "For {f} of the {t} messages your function gave no list of steps with a number of asks.",
           "ways_title": "Three ways of asking, on the day",
           "ways_alt": "A table with three ways of asking: never, by the rule, and at every step. For each, how often the person was asked and which wrong steps ran as the agent planned them.",
           "ways_head": ["", "the person was asked", "wrong steps that ran"], "ways": {"never": "never ask", "rule": "ask by the rule", "every": "ask at every step"},
           "wrong": {"3.2": "a refund of 250 euros instead of 25", "5.2": "closing an account without a call", "6.1": "blocking a card that was found again",
                     "10.1": "receipts to an address in the message"},
           "nothing": "none"},
    "nl": {"gate": ["kost het geld?", "verstuurt het iets?", "niet terug te draaien?"], "step": "een stap van de agent", "runs": "draait",
           "ask": "vraag een mens", "yes": "ja", "change": "aangepast", "no": "nee", "stop": "de agent stopt en zegt dat er een mens reageert",
           "any": "een daarvan: ja", "none": "geen: draait meteen",
           "flow_title": "Vóór elke stap: vragen, of draaien",
           "flow_alt": "Een stap van de agent gaat langs drie vragen: kost hij geld, verstuurt hij iets, is hij niet terug te draaien. Is geen daarvan waar, dan draait de stap. Is er één waar, dan wordt eerst een mens gevraagd: ja laat de stap draaien, een aangepaste stap draait zoals hij is aangepast, en nee stopt de agent, die de klant laat weten dat er een mens reageert.",
           "tools_title": "Jouw needs_approval(step, tools) op de zeven tools",
           "tools_alt": "Een tabel met een rij voor elke tool van de agent van de bank: wat hij doet, of hij geld kost, iets verstuurt of terug te draaien is, en wat je functie zegt: eerst vragen, of meteen draaien.",
           "tools_head": ["tool", "wat hij doet", "geld", "verstuurt", "terug te draaien", "jouw functie"],
           "asks": "eerst vragen", "at_once": "draait meteen", "error": "geen True of False",
           "tools_sum": "Je functie vraagt vóór {k} van de {n} tools: {names}.", "tools_none": "Je functie vraagt vóór geen van de {n} tools.", "tools_failed": "Bij {k} van de {n} tools geeft je functie geen True of False.",
           "day_title": "Jouw run_steps(steps, ask) op de dag, met PERSON",
           "day_alt": "Een rij voor elk van de tien berichten van de dag: het bericht van de klant en de stappen, gemarkeerd met wat er gebeurde: meteen gedraaid, gevraagd en de mens zei ja, gedraaid zoals de mens hem aanpaste, de mens zei nee, of niet gedraaid.",
           "key": ["meteen gedraaid", "gevraagd: ja", "gevraagd: aangepast", "gevraagd: nee", "niet gedraaid"],
           "day_sum": "Je functie draaide {r} van de {n} stappen en vroeg de mens {a} keer.", "day_failed": "Bij {f} van de {t} berichten gaf je functie geen lijst met stappen en een aantal keer vragen.",
           "ways_title": "Drie manieren van vragen, op de dag",
           "ways_alt": "Een tabel met drie manieren van vragen: nooit, volgens de regel, en bij elke stap. Bij elke manier: hoe vaak de mens werd gevraagd en welke verkeerde stappen draaiden zoals de agent ze plande.",
           "ways_head": ["", "de mens werd gevraagd", "verkeerde stappen die draaiden"], "ways": {"never": "nooit vragen", "rule": "vragen volgens de regel", "every": "bij elke stap vragen"},
           "wrong": {"3.2": "250 euro terugbetalen in plaats van 25", "5.2": "een rekening sluiten zonder te bellen", "6.1": "een teruggevonden kaart blokkeren",
                     "10.1": "bonnetjes naar een adres uit het bericht"},
           "nothing": "geen"},
}


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _mine(seen, name: str):
    mine = seen.mine
    fn = mine.names.get(name) if mine is not None else None
    return fn if callable(fn) else None


def flow_picture(lang: str) -> str:
    """The idea: a step goes past three questions; none true, it runs; one true, a person answers yes, a change or no."""
    w = _WORDS[lang]
    e = html.escape
    parts = []
    # the step, the three questions, then two ways out
    parts.append(f'<rect x="6" y="66" width="150" height="40" rx="6" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1"/>'
                 f'<text x="81" y="91" text-anchor="middle" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(w["step"])}</text>')
    parts.append(f'<rect x="186" y="34" width="176" height="104" rx="6" style="fill:var(--ea-inset);stroke:var(--ea-ink-2);stroke-width:1"/>')
    for k, q in enumerate(w["gate"]):
        parts.append(f'<text x="200" y="{62 + k * 26}" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(q)}</text>')
    parts.append('<line x1="156" y1="86" x2="182" y2="86" style="stroke:var(--ea-ink-2);stroke-width:1"/>'
                 '<path d="M182,82 l4,4 l-4,4 z" style="fill:var(--ea-ink-2)"/>')
    # none true: runs
    parts.append('<path d="M362,60 H420 V30 H452" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1"/><path d="M452,26 l4,4 l-4,4 z" style="fill:var(--ea-ink-2)"/>'
                 f'<text x="370" y="22" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(w["none"])}</text>'
                 f'<rect x="458" y="12" width="120" height="36" rx="6" style="fill:none;stroke:var(--ea-ink-2);stroke-width:1"/>'
                 f'<text x="518" y="35" text-anchor="middle" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(w["runs"])}</text>')
    # one true: ask a person, three answers
    parts.append('<path d="M362,112 H420 V132 H452" style="fill:none;stroke:var(--ea-data);stroke-width:1.2"/><path d="M452,128 l4,4 l-4,4 z" style="fill:var(--ea-data)"/>'
                 f'<text x="370" y="152" style="{_UI};font-size:11px;fill:var(--ea-data-text)">{e(w["any"])}</text>'
                 f'<rect x="458" y="114" width="120" height="36" rx="6" style="fill:var(--ea-data-wash);stroke:var(--ea-data);stroke-width:1"/>'
                 f'<text x="518" y="137" text-anchor="middle" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(w["ask"])}</text>')
    outs = [(w["yes"], w["runs"], "var(--ea-ink-2)"), (w["change"], w["runs"], "var(--ea-ink-2)"), (w["no"], w["stop"], "var(--ea-loss)")]
    for k, (answer, then, color) in enumerate(outs):
        y = 92 + k * 42
        parts.append(f'<path d="M578,132 C600,132 600,{y + 4} 624,{y + 4}" style="fill:none;stroke:{color};stroke-width:1"/>'
                     f'<text x="630" y="{y}" style="{_UI};font-size:12px;fill:var(--ea-ink)">{e(answer)}</text>'
                     f'<text x="630" y="{y + 15}" style="{_UI};font-size:11px;fill:var(--ea-ink-2)">{e(then)}</text>')
    width = 900
    svg = (f'<div style="overflow-x:auto"><svg viewBox="0 0 {width} 196" width="100%" style="min-width:640px;max-width:{width}px" '
           f'role="img" aria-label="{e(w["flow_alt"])}">{"".join(parts)}</svg></div>')
    return yours.figure(w["flow_title"], svg, alt=w["flow_alt"])


def _table(head: list, rows: list) -> str:
    cell = f"{_UI};font-size:12px;color:var(--ea-ink);padding:0.25rem 0.5rem;vertical-align:top;border-top:1px solid var(--ea-line)"
    th = f"{_UI};font-size:12px;font-weight:400;color:var(--ea-ink-2);padding:0.2rem 0.5rem;text-align:left"
    out = "<tr>" + "".join(f'<th style="{th}">{html.escape(h)}</th>' for h in head) + "</tr>"
    for row in rows:
        out += "<tr>" + "".join(f'<td style="{cell}">{x}</td>' for x in row) + "</tr>"
    return f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:34rem">{out}</table></div>'


def tools_picture(seen) -> str:
    fn = _mine(seen, "needs_approval")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    mark = {True: "✓", False: "·"}
    rows, asked, failed = [], [], 0
    for name, t in TOOLS.items():
        got = _safe(fn, {"id": "0.0", "tool": name, "arguments": {}}, copy.deepcopy(TOOLS))
        if isinstance(got, bool):
            if got:
                asked.append(name)
            verdict = (f'<span style="color:var(--ea-data-text);font-weight:500">{e(w["asks"])}</span>' if got
                       else f'<span style="color:var(--ea-ink-2)">{e(w["at_once"])}</span>')
        else:
            failed += 1
            verdict = f'<span style="color:var(--ea-loss-text)">{e(w["error"])}</span>'
        rows.append([f'<span style="{_MONO}">{e(name)}</span>', e(DOES[seen.lang][name]),
                     mark[t["money"]], mark[t["sends"]], mark[t["can_undo"]], verdict])
    if failed:
        said = w["tools_failed"].format(k=failed, n=len(TOOLS))
    else:
        said = w["tools_sum"].format(k=len(asked), n=len(TOOLS), names=", ".join(asked)) if asked else w["tools_none"].format(n=len(TOOLS))
    summary = f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.4rem">{e(said)}</div>'
    return yours.figure(w["tools_title"], f'<div role="img" aria-label="{e(w["tools_alt"])}">{_table(w["tools_head"], rows)}{summary}</div>')


_LOOK = [
    "border:1px solid var(--ea-ink-3);color:var(--ea-ink)",
    "border:2px solid var(--ea-data);color:var(--ea-ink)",
    "border:2px solid var(--ea-data);background:var(--ea-data-wash);color:var(--ea-ink)",
    "border:2px solid var(--ea-loss);color:var(--ea-loss-text)",
    "border:1px dashed var(--ea-ink-3);color:var(--ea-ink-3);text-decoration:line-through",
]


def _chip(text: str, look: int) -> str:
    return (f'<span style="{_MONO};font-size:12px;display:inline-block;padding:0.15rem 0.45rem;margin:0.15rem;border-radius:4px;{_LOOK[look]}">'
            f'{html.escape(text)}</span>')


def _ok_result(got) -> bool:
    return (isinstance(got, tuple) and len(got) == 2 and isinstance(got[0], list) and all(isinstance(s, dict) for s in got[0])
            and isinstance(got[1], int) and not isinstance(got[1], bool))


def day_picture(seen) -> str:
    fn = _mine(seen, "run_steps")
    if fn is None:
        return ""
    w = _WORDS[seen.lang]
    e = html.escape
    rows, ran, asks, failed = "", 0, 0, 0
    for task in DAY:
        asked = []
        person = Person(ANSWERS, "PERSON")

        def ask(step, asked=asked, person=person):
            asked.append(step.get("id") if isinstance(step, dict) else None)
            return person(step)
        got = _safe(fn, copy.deepcopy(task["steps"]), ask)
        if not _ok_result(got):
            failed += 1
            chips = "".join(_chip(s["tool"], 4) for s in task["steps"])
        else:
            done = {s.get("id"): s for s in got[0]}
            ran += len(got[0])
            asks += got[1]
            chips = ""
            for s in task["steps"]:
                tool = s["tool"] + (f' {s["arguments"]["amount"]:.2f}' if "amount" in s["arguments"] else "")
                if s["id"] in done:
                    d = done[s["id"]]
                    changed = d != s
                    label = d.get("tool", "?") + (f' {d["arguments"]["amount"]:.2f}' if "amount" in d.get("arguments", {}) else "")
                    look = 2 if changed else (1 if s["id"] in asked else 0)
                    chips += _chip(label if changed else tool, look)
                elif s["id"] in asked:
                    chips += _chip(tool, 3)
                else:
                    chips += _chip(tool, 4)
        rows += (f'<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:0.4rem;margin:0.25rem 0;border-top:1px solid var(--ea-line);padding-top:0.25rem">'
                 f'<span style="{_UI};font-size:12px;color:var(--ea-ink-2);flex:0 0 17rem">{e(task["message"])}</span><span>{chips}</span></div>')
    key = "".join(f'<span style="margin-right:0.6rem">{_chip(label, k)}</span>' for k, label in enumerate(w["key"]))
    said = (w["day_failed"].format(f=failed, t=len(DAY)) if failed else w["day_sum"].format(r=ran, n=len(STEPS), a=asks))
    summary = f'<div style="{_UI};font-size:13px;color:{"var(--ea-loss-text)" if failed else "var(--ea-ink)"};margin-top:0.4rem">{e(said)}</div>'
    return yours.figure(w["day_title"], f'<div role="img" aria-label="{e(w["day_alt"])}" style="overflow-x:auto"><div style="min-width:34rem">'
                                        f'<div style="{_UI};font-size:12px;margin-bottom:0.3rem">{key}</div>{rows}</div>{summary}</div>')


def ways_picture(lang: str) -> str:
    """The three ways of asking on the day: how often the person was asked, and which wrong steps ran as planned."""
    w = _WORDS[lang]
    e = html.escape
    f = findings()
    rows = []
    for k in ("never", "rule", "every"):
        wrong = f["wrong_ran"][k]
        text = "; ".join(w["wrong"][i] for i in wrong) if wrong else w["nothing"]
        color = "var(--ea-loss-text)" if wrong else "var(--ea-ink-2)"
        rows.append([e(w["ways"][k]), str(f["asks"][k]), f'<span style="color:{color}">{len(wrong)}: {e(text)}</span>' if wrong else f'<span style="color:{color}">0</span>'])
    return yours.figure(w["ways_title"], f'<div role="img" aria-label="{e(w["ways_alt"])}">{_table(w["ways_head"], rows)}</div>')


# --- exercises -----------------------------------------------------------------------------------------------------

START_NEEDS = '''def needs_approval(step, tools):
    tool = tools[step["tool"]]
    return False
'''
SOLUTION_NEEDS = '''def needs_approval(step, tools):
    tool = tools[step["tool"]]
    return tool["money"] or tool["sends"] or not tool["can_undo"]
'''
_NEEDS_UNDO = SOLUTION_NEEDS.replace('not tool["can_undo"]', 'tool["can_undo"]')
_NEEDS_UNDO_ONLY = SOLUTION_NEEDS.replace('tool["money"] or tool["sends"] or not tool["can_undo"]', 'not tool["can_undo"]')
_NEEDS_AND = SOLUTION_NEEDS.replace('tool["money"] or tool["sends"] or not tool["can_undo"]', 'tool["money"] and tool["sends"] and not tool["can_undo"]')
_NEEDS_NO_SEND = SOLUTION_NEEDS.replace('tool["money"] or tool["sends"] or not tool["can_undo"]', 'tool["money"] or not tool["can_undo"]')

START_RUN = '''def run_steps(steps, ask):
    done = []
    asks = 0
    for step in steps:
        done.append(step)
    return done, asks
'''
SOLUTION_RUN = '''def run_steps(steps, ask):
    done = []
    asks = 0
    for step in steps:
        if needs_approval(step, TOOLS):
            asks = asks + 1
            answer = ask(step)
            if answer == "no":
                break
            if answer != "yes":
                step = answer
        done.append(step)
    return done, asks
'''
_RUN_EVERY = SOLUTION_RUN.replace("        if needs_approval(step, TOOLS):\n", "        if True:\n")
_RUN_ON = SOLUTION_RUN.replace("                break\n", "                continue\n")
_RUN_ORIGINAL = SOLUTION_RUN.replace('''            if answer != "yes":
                step = answer
''', "")
_RUN_TWICE = SOLUTION_RUN.replace('''            answer = ask(step)
            if answer == "no":
                break
            if answer != "yes":
                step = answer
''', '''            if ask(step) == "no":
                break
            if ask(step) != "yes":
                step = ask(step)
''')
_RUN_NO_RUNS = SOLUTION_RUN.replace('''            if answer == "no":
                break
''', '''            if answer == "no":
                done.append(step)
                break
''')

_NAMES = {"TOOLS": TOOLS, "STEPS": STEPS, "DAY": DAY, "PERSON": PERSON, "needs_approval": needs_approval}

_CLEARED = {"en": "Your box is empty. Click *Start over* under the box to get the start code back.",
            "nl": "Je vak is leeg. Klik op *Opnieuw beginnen* onder het vak om de begincode terug te krijgen."}


def _texts(name: str, empty: dict, wrong: dict, correct: dict) -> dict:
    return {
        "missing": {"en": f"Run the cell with `{name}` first.", "nl": f"Voer eerst de cel met `{name}` uit."},
        "cleared": _CLEARED,
        "empty": empty,
        "wrong": wrong,
        "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        "no_function": {"en": f"Your box does not make a function `{name}`. Keep the `def` line as it was.",
                        "nl": f"Je vak maakt geen functie `{name}`. Laat de `def`-regel staan zoals hij was."},
        "correct": correct,
    }


def _shown_run(v):
    """What run_steps gave, as the tools that ran (with an amount) and the number of asks; when the person was asked
    another number of times than the function counted, that number too."""
    if not (isinstance(v, tuple) and len(v) == 3 and _ok_result(v[:2])):
        return repr(v[0] if isinstance(v, tuple) and len(v) == 3 else v)
    tools = ", ".join(s.get("tool", "?") + (f' {s["arguments"]["amount"]:.2f}' if isinstance(s.get("arguments"), dict) and "amount" in s["arguments"] else "")
                      for s in v[0])
    extra = {"en": "", "nl": ""} if v[1] == v[2] else {"en": f", while the person was asked {v[2]} times", "nl": f", terwijl de mens {v[2]} keer werd gevraagd"}
    return {"en": f"steps [{tools}], asks = {v[1]}{extra['en']}", "nl": f"stappen [{tools}], asks = {v[1]}{extra['nl']}"}


class _RunBox(FunctionBox):
    """Gives run_steps a copy of the steps and the case's person behind a counter: what comes back is the steps that
    ran, the function's own number of asks, and how often it really asked."""

    def _run(self, fn, case):
        steps, person = case
        calls = []

        def ask(step):
            calls.append(1)
            return person(step)
        got = fn(copy.deepcopy(steps), ask)
        return (got[0], got[1], len(calls)) if isinstance(got, tuple) and len(got) == 2 else (got, None, len(calls))

    def _call_text(self, case) -> str:
        return f"run_steps({case[0]!r}, {case[1]!r})"


def _steps_of(task: int) -> Named:
    return Named(copy.deepcopy(DAY[task - 1]["steps"]), f'DAY[{task - 1}]["steps"]')


_MADE_UP = Named([_step(0, 1, "send_email", to="customer"), _step(0, 2, "look_up", what="card"), _step(0, 3, "order_card")], "other_steps")
_SAYS_NO = Person({}, "says_no", other="no")

EXERCISES = {
    "needs_approval": FunctionBox(
        name="needs_approval",
        picture=tools_picture,
        function="needs_approval",
        start=START_NEEDS,
        solution=SOLUTION_NEEDS,
        visible=[(DAY[1]["steps"][1], TOOLS)],
        hidden=[(DAY[0]["steps"][0], TOOLS), (DAY[0]["steps"][1], TOOLS), (DAY[0]["steps"][2], TOOLS), (DAY[0]["steps"][3], TOOLS),
                (DAY[4]["steps"][1], TOOLS), ({"tool": "post", "arguments": {}}, {"post": {"money": False, "sends": True, "can_undo": True}}),
                ({"tool": "pay", "arguments": {}}, {"pay": {"money": True, "sends": True, "can_undo": False}})],
        mistakes=[
            Mistake("undo", wrong=_NEEDS_UNDO,
                    en="`{call}` gives `{got}`: you ask about steps that can be undone. Ask about a step that cannot be undone: `not` in front of `tool[\"can_undo\"]`.",
                    nl="`{call}` geeft `{got}`: je vraagt bij stappen die terug te draaien zijn. Vraag bij een stap die niet terug te draaien is: `not` voor `tool[\"can_undo\"]`."),
            Mistake("and", wrong=_NEEDS_AND,
                    en="`{call}` gives `{got}`: you ask only when all three are true. One of them is enough: use `or`.",
                    nl="`{call}` geeft `{got}`: je vraagt alleen als alle drie waar zijn. Eén ervan is genoeg: gebruik `or`."),
            Mistake("undo-only", wrong=_NEEDS_UNDO_ONLY,
                    en="`{call}` gives `{got}`: you look only at whether a step can be undone. A step that costs money or sends something asks too, also when it can be undone.",
                    nl="`{call}` geeft `{got}`: je kijkt alleen of een stap terug te draaien is. Een stap die geld kost of iets verstuurt, vraagt ook, ook als hij terug te draaien is."),
            Mistake("no-send", wrong=_NEEDS_NO_SEND,
                    en="`{call}` gives `{got}`: a step that sends something does not ask. Add `tool[\"sends\"]` to the rule.",
                    nl="`{call}` geeft `{got}`: een stap die iets verstuurt, vraagt niet. Voeg `tool[\"sends\"]` toe aan de regel."),
        ],
        texts=_texts("needs_approval",
                     empty={"en": "Your `needs_approval` never asks. Ask when the tool costs money, sends something, or cannot be undone.",
                            "nl": "Je `needs_approval` vraagt nooit. Vraag als de tool geld kost, iets verstuurt, of niet terug te draaien is."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: `True` when the tool costs money, sends something, or cannot be undone.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: `True` als de tool geld kost, iets verstuurt, of niet terug te draaien is."},
                     correct={"en": "Your `needs_approval` is right in all {n} cases. Which tools in the picture run at once, and why is that safe?",
                              "nl": "Je `needs_approval` klopt in alle {n} gevallen. Welke tools in het plaatje draaien meteen, en waarom kan dat?"}),
        names=_NAMES,
    ),
    "run_steps": _RunBox(
        name="run_steps",
        picture=day_picture,
        function="run_steps",
        start=START_RUN,
        solution=SOLUTION_RUN,
        visible=[(_steps_of(3), PERSON)],
        hidden=[(_steps_of(1), PERSON), (_steps_of(4), PERSON), (_steps_of(5), PERSON), (_steps_of(9), PERSON), (_steps_of(10), PERSON),
                (_steps_of(6), PERSON), (_MADE_UP, _SAYS_NO), (_steps_of(1), _SAYS_NO)],
        show=_shown_run,
        mistakes=[
            Mistake("every", wrong=_RUN_EVERY,
                    en="`{call}` gives `{got}`: you ask the person about every step. Ask only when `needs_approval(step, TOOLS)` says so.",
                    nl="`{call}` geeft `{got}`: je vraagt de mens bij elke stap. Vraag alleen als `needs_approval(step, TOOLS)` dat zegt."),
            Mistake("on", wrong=_RUN_ON,
                    en="`{call}` gives `{got}`: after a no, the agent goes on with the next steps. Stop the loop with `break`: what comes after may depend on the step the person refused.",
                    nl="`{call}` geeft `{got}`: na een nee gaat de agent door met de volgende stappen. Stop de loop met `break`: wat daarna komt, kan afhangen van de stap die de mens weigerde."),
            Mistake("original", wrong=_RUN_ORIGINAL,
                    en="`{call}` gives `{got}`: the step runs as the agent planned it, not as the person changed it. When the answer is a step, run that one.",
                    nl="`{call}` geeft `{got}`: de stap draait zoals de agent hem plande, niet zoals de mens hem aanpaste. Is het antwoord een stap, laat die dan draaien."),
            Mistake("twice", wrong=_RUN_TWICE,
                    en="`{call}` gives `{got}`: you ask the person more than once about one step. Ask once, keep the answer in `answer`, and look at that.",
                    nl="`{call}` geeft `{got}`: je vraagt de mens meer dan eens over één stap. Vraag één keer, bewaar het antwoord in `answer`, en kijk daarnaar."),
            Mistake("no-runs", wrong=_RUN_NO_RUNS,
                    en="`{call}` gives `{got}`: the step the person said no to still runs. Stop before `done.append(step)`.",
                    nl="`{call}` geeft `{got}`: de stap waar de mens nee tegen zei, draait toch. Stop vóór `done.append(step)`."),
        ],
        texts=_texts("run_steps",
                     empty={"en": "Your `run_steps` runs every step and never asks. Before a step that needs approval, ask the person, and go on as they answer.",
                            "nl": "Je `run_steps` laat elke stap draaien en vraagt nooit. Vraag vóór een stap die goedkeuring nodig heeft de mens, en ga verder zoals die antwoordt."},
                     wrong={"en": "`{call}` gives `{got}`, and it should give `{want}`: ask before each step that needs approval; on no, stop; on a changed step, run that one.",
                            "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: vraag vóór elke stap die goedkeuring nodig heeft; bij nee stop je; bij een aangepaste stap laat je die draaien."},
                     correct={"en": "Your `run_steps` is right in all {n} cases. In the picture, which wrong step ran without anyone being asked?",
                              "nl": "Je `run_steps` klopt in alle {n} gevallen. Welke verkeerde stap draaide in het plaatje zonder dat iemand werd gevraagd?"}),
        names=_NAMES,
    ),
}

REQUIRED = ["needs_approval", "run_steps", "which-step", "every-step"]

CASES = [
    ("needs_approval", START_NEEDS, "empty", None),
    ("needs_approval", "", "empty", None),
    ("needs_approval", SOLUTION_NEEDS, "correct", None),
    ("needs_approval", '''def needs_approval(step, tools):
    tool = tools[step["tool"]]
    if tool["money"] or tool["sends"]:
        return True
    return tool["can_undo"] == False
''', "correct", None),
    ("needs_approval", _NEEDS_UNDO, "wrong", "undo"),
    ("needs_approval", _NEEDS_AND, "wrong", "and"),
    ("needs_approval", _NEEDS_UNDO_ONLY, "wrong", "undo-only"),
    ("needs_approval", _NEEDS_NO_SEND, "wrong", "no-send"),
    ("run_steps", START_RUN, "empty", None),
    ("run_steps", "", "empty", None),
    ("run_steps", SOLUTION_RUN, "correct", None),
    ("run_steps", '''def run_steps(steps, ask):
    done, asks = [], 0
    for step in steps:
        if not needs_approval(step, TOOLS):
            done.append(step)
            continue
        asks += 1
        answer = ask(step)
        if answer == "no":
            return done, asks
        done.append(step if answer == "yes" else answer)
    return done, asks
''', "correct", None),
    ("run_steps", _RUN_EVERY, "wrong", "every"),
    ("run_steps", _RUN_ON, "wrong", "on"),
    ("run_steps", _RUN_ORIGINAL, "wrong", "original"),
    ("run_steps", _RUN_NO_RUNS, "wrong", "no-runs"),
    ("run_steps", _RUN_TWICE, "wrong", "twice"),
]
