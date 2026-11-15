# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-prompt-caching: a function that counts the tokens two prompts have in common at the start, and one
that puts the time after a shop's long handbook instead of before it, so that the server can keep its work on the
handbook; the course's recorded runs (data/course-dev-prompts-qwen.json): seven requests one after the other, with
the seconds each took and the tokens that came from the server's kept cache; their pictures, and the demo."""
import html
import json
import statistics
import time

from everything_ai import chat, paths, serve, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

HANDBOOK = """You are the assistant of Spoke & Chain, a bike shop. Answer the customer's question briefly, using only the rules below. If the rules do not say, tell the customer to call the shop.

Opening hours. The shop is open from Tuesday to Friday from 9:00 to 18:00, and on Saturday from 9:00 to 17:00. It is closed on Sunday and Monday, and on public holidays. On the last Thursday of each month the shop stays open until 21:00 for the repair evening, where customers can fix their own bike with the shop's tools and a mechanic nearby.

Repairs. A flat tire is fixed while you wait, in about twenty minutes, for 15 euros, the inner tube included. A full service costs 85 euros and takes two working days; it covers the brakes, the gears, the chain, the wheels and a safety check. Brake pads cost 12 euros per wheel, fitted. A new chain costs 35 euros, fitted. A wheel that is bent can often be trued for 20 euros; a new wheel starts at 70 euros. Electric bikes get a service for 120 euros, which includes a test of the battery and an update of the motor's software. Repairs are paid when the bike is collected. A bike that is not collected within thirty days after the repair is finished costs 2 euros a day for storage.

Returns. A product that has not been used can be returned within thirty days with the receipt, for the full price back on the card or account it was paid with. Used products, products made to measure, and inner tubes cannot be returned. A bike can be returned within fourteen days if it has been ridden less than 50 kilometers, minus 10 percent of the price for the inspection.

Warranty. New bikes have five years of warranty on the frame and two years on all other parts, except parts that wear, such as tires, brake pads, chains and cables. Batteries of electric bikes have two years of warranty, or 500 charges, whichever comes first. Warranty only holds with the receipt and when the bike was serviced at least once a year. Damage from a fall, an accident or wrong use is not covered.

Rentals. City bikes can be rented for 15 euros a day or 60 euros a week; electric bikes for 35 euros a day or 150 euros a week. Every rental includes a lock and a helmet. A deposit of 100 euros is paid when the bike is picked up and given back when it is returned undamaged. A rental bike that comes back late costs the price of an extra day. Rental bikes may not be taken abroad, and a lost lock costs 30 euros.

Delivery. Bikes bought in the shop can be delivered at home within 15 kilometers for 25 euros, ready to ride. Parts ordered online arrive within three working days; orders above 50 euros are sent for free. Orders placed before 14:00 on a working day leave the same day.

Payment. The shop takes cards and mobile payments, and cash up to 500 euros. Bikes above 1,000 euros can be paid in twelve monthly parts without interest, after a check of the customer's identity. Gift cards are sold in the shop for any amount from 10 euros, and they are valid for two years.

Bike fitting. A fitting takes an hour and costs 60 euros: a mechanic sets the saddle, the handlebars and the pedals to the customer's body and way of riding. Customers who buy a bike above 1,500 euros get the fitting for free within three months of the purchase.

Trade-in. The shop buys used bikes that are less than eight years old and in working order, for a price the mechanic sets after a check of twenty minutes. The price is paid as a voucher for the shop, which is 10 percent higher than the price in cash. Bikes without a frame number, and bikes with a cracked frame, are not bought.

Children's bikes. Children grow fast, so the shop takes back a children's bike that was bought new in the shop for 40 percent of its price, as a voucher, when a larger one is bought within three years. Children's helmets are fitted for free.

Safety. Every bike that leaves the shop after a sale or a service has working lights, a bell and reflectors. The shop does not sell or fit parts that change an electric bike so that its motor helps above 25 kilometers an hour, as that is not allowed on the road.

Members. A membership costs 30 euros a year. Members get 10 percent off all parts and services, one free flat tire repair a year, and two hours earlier access to the yearly sale in March. A membership cannot be shared and ends after one year unless it is renewed."""
QUESTIONS = ["How much does a full service cost?", "Can I return a helmet I wore once?", "Is the shop open on Monday?"]
NOW = ["Monday 16 November 2026, 09:14", "Monday 16 November 2026, 09:15"]


def asked(question: str, now: str | None = None, first: bool = False) -> list:
    """The messages for a question: the handbook as the system prompt, and the time, when there is one, at the very
    start of it (`first`) or after the question."""
    system = HANDBOOK if now is None or not first else "Now: " + now + "\n\n" + HANDBOOK
    user = question if now is None or first else question + "\n\nNow: " + now
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


SEEDS = 3   # the sequence three times over: its seconds as a median
RECORD = {
    "sequence": {"sequence": [{"messages": asked(QUESTIONS[0]), "cache": False}, {"messages": asked(QUESTIONS[1])},
                              {"messages": asked(QUESTIONS[2])},
                              {"messages": asked(QUESTIONS[0], NOW[0], first=True)},
                              {"messages": asked(QUESTIONS[0], NOW[1], first=True)},
                              {"messages": asked(QUESTIONS[0], NOW[0])}, {"messages": asked(QUESTIONS[0], NOW[1])}],
                 "max_tokens": 1, "seeds": SEEDS},
    "answers": {"sequence": [{"messages": asked(q), "cache": i > 0} for i, q in enumerate(QUESTIONS)],
                "max_tokens": 60, "seeds": 1},
    "same": {"sequence": [{"messages": asked(QUESTIONS[2]), "cache": False}, {"messages": asked(QUESTIONS[2])}],
             "max_tokens": 60, "seeds": 1},
    "handbook": {"count": [HANDBOOK]},
}
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"

# What each request of the sequence was: the question it asked and where the time stood.
STEPS = ["cold", "same-1", "same-2", "first-1", "first-2", "last-1", "last-2"]


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-prompt-caching"]


def steps() -> list:
    """Each request of the sequence: its prompt tokens, the tokens of it that came from the kept cache (the same in
    every round), and its seconds, the median of the rounds."""
    rounds = [a["steps"] for a in course()["sequence"]["answers"]]
    out = []
    for i, name in enumerate(STEPS):
        done = [r[i] for r in rounds]
        assert len({(d["prompt_tokens"], d["cached_tokens"]) for d in done}) == 1
        out.append({"name": name, "prompt": done[0]["prompt_tokens"], "cached": done[0]["cached_tokens"],
                    "seconds": statistics.median(d["seconds"] for d in done), "all": [d["seconds"] for d in done]})
    return out


def findings() -> dict:
    """What the recorded runs show: the tokens of the handbook, and per request its tokens, cached tokens and seconds;
    and the answers to the three questions, the later two with the cache."""
    s = {x["name"]: x for x in steps()}
    answers = course()["answers"]["answers"][0]["steps"]
    return {"handbook": course()["handbook"]["answers"][0]["tokens"], "steps": s, "rounds": len(course()["sequence"]["answers"]),
            "answers": [a["text"] for a in answers], "answer_cached": [a["cached_tokens"] for a in answers],
            "answer_seconds": [a["seconds"] for a in answers], "answer_tokens": [a["tokens"] for a in answers],
            "same": [{k: a[k] for k in ("text", "cached_tokens")} for a in course()["same"]["answers"][0]["steps"]]}


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".") if lang == "nl" else text


# --- the pictures of the recorded runs -------------------------------------------------------------------------------

_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _label(text: str) -> str:
    return f'<div style="{_LABEL};margin:0.2rem 0">{html.escape(text)}</div>'


_SEQ_WORDS = {
    "en": {"title": "Seven requests to the course's server, one after the other", "cached": "from the cache",
           "computed": "computed", "seconds": "{s} s",
           "rows": {"cold": "1. Question A, the cache switched off", "same-1": "2. Question B, the same handbook",
                    "same-2": "3. Question C, the same handbook", "first-1": "4. The time first, then the handbook: 09:14",
                    "first-2": "5. The same, a minute later: 09:15", "last-1": "6. The handbook first, the time after the question: 09:14",
                    "last-2": "7. The same, a minute later: 09:15"},
           "note": "Each bar is the prompt of one request, about {n} tokens. The seconds are the median of {k} rounds on the course's computer, with max_tokens 1, so almost all of it is reading the prompt."},
    "nl": {"title": "Zeven requests naar de server van de course, na elkaar", "cached": "uit de cache",
           "computed": "berekend", "seconds": "{s} s",
           "rows": {"cold": "1. Vraag A, de cache uitgezet", "same-1": "2. Vraag B, hetzelfde handboek",
                    "same-2": "3. Vraag C, hetzelfde handboek", "first-1": "4. Eerst de tijd, dan het handboek: 09:14",
                    "first-2": "5. Hetzelfde, een minuut later: 09:15", "last-1": "6. Eerst het handboek, de tijd na de vraag: 09:14",
                    "last-2": "7. Hetzelfde, een minuut later: 09:15"},
           "note": "Elke balk is de prompt van één request, ongeveer {n} tokens. De seconden zijn de mediaan van {k} rondes op de computer van de course, met max_tokens 1, dus bijna alles is het lezen van de prompt."},
}


def sequence_picture(lang: str) -> str:
    """The seven requests: for each, a bar of its prompt tokens, the part that came from the cache in the wash of the
    data color and the part the server computed in the full color, with a key, the numbers, and its seconds."""
    w = _SEQ_WORDS[lang]
    rows = steps()
    top = max(r["prompt"] for r in rows)
    slowest = max(r["seconds"] for r in rows)
    parts = []
    for r in rows:
        cached, computed = r["cached"], r["prompt"] - r["cached"]
        bar = (f'<div style="display:flex;height:12px;width:{100 * r["prompt"] / top:.1f}%;border-radius:2px;overflow:hidden">'
               f'<div style="width:{100 * cached / r["prompt"]:.2f}%;background:var(--ea-data-wash)"></div>'
               f'<div style="flex:1;background:var(--ea-data)"></div></div>')
        time_bar = (f'<div style="display:flex;align-items:center;gap:0.3rem"><div style="height:8px;width:{100 * r["seconds"] / slowest:.1f}%;'
                    f'min-width:2px;max-width:6rem;background:var(--ea-ink-3)"></div>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink)">{w["seconds"].format(s=number(r["seconds"], lang, 2))}</span></div>')
        parts.append(f'<div style="margin:0.35rem 0"><div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(w["rows"][r["name"]])}</div>'
                     f'<div style="display:grid;grid-template-columns:minmax(0,3fr) minmax(0,1.3fr);gap:0.6rem;align-items:center">'
                     f'<div>{bar}<div style="{_LABEL}">{number(cached, lang, 0)} {w["cached"]} · {number(computed, lang, 0)} {w["computed"]}</div></div>'
                     f'<div>{time_bar}</div></div></div>')
    note = w["note"].format(n=number(rows[0]["prompt"], lang, 0), k=findings()["rounds"])
    swatch = '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{};margin:0 0.3rem 0 0.8rem;vertical-align:-1px"></span>'
    key = (f'<div style="{_LABEL};margin:0.4rem 0 0.2rem">{swatch.format("var(--ea-data-wash)")}{html.escape(w["cached"])}'
           f'{swatch.format("var(--ea-data)")}{html.escape(w["computed"])}</div>')
    return yours.figure(w["title"], "".join(parts) + key + _label(note), alt=w["title"])


def _tokens_html(tokens: list, shared: int, limit: int = 18) -> str:
    """Tokens as small boxes: the shared start light, the first one that differs marked, then the rest (cut off)."""
    out = []
    for i, t in enumerate(tokens[:limit]):
        if i < shared:
            style = "background:var(--ea-data-wash);color:var(--ea-ink)"
        elif i == shared:
            style = "background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
        else:
            style = "background:var(--ea-inset);color:var(--ea-ink-2)"
        out.append(f'<span style="{style};font-family:var(--ea-font-code);font-size:12px;padding:0.05rem 0.25rem;border-radius:3px;margin:0 0.12rem 0.12rem 0;display:inline-block">{html.escape(repr(t))}</span>')
    if len(tokens) > limit:
        out.append(f'<span style="{_LABEL}">… {len(tokens) - limit}</span>')
    return "".join(out)


# --- exercise 1: shared_start ---------------------------------------------------------------------------------------------

START_SHARED = """def shared_start(a, b):
    n = 0
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            n = n + 1
    return n
"""
SOLUTION_SHARED = """def shared_start(a, b):
    n = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return n
        n = n + 1
    return n
"""
SHARED_CASES = [([151644, 8948, 198, 3555, 525], [151644, 8948, 198, 2610, 525]),
                ([5, 8, 2, 9, 4], [5, 8, 3, 9, 4]), ([1, 2, 3], [1, 2, 3]), ([1, 2], [1, 2, 3, 4]),
                ([7, 1, 2], [8, 1, 2]), ([], [3, 4])]

_SHARED_WORDS = {
    "en": {"title": "What your shared_start counted", "a": "the tokens of the first prompt", "b": "the tokens of the second prompt",
           "said": "Your function: {got}", "want": "The server can reuse {want}: the tokens up to the first one that differs (marked)."},
    "nl": {"title": "Wat je shared_start telde", "a": "de tokens van de eerste prompt", "b": "de tokens van de tweede prompt",
           "said": "Je functie: {got}", "want": "De server kan er {want} hergebruiken: de tokens tot het eerste dat verschilt (gemarkeerd)."},
}


def shared_picture(seen) -> str:
    """The two token lists of the second case, the shared start light and the first token that differs marked, what
    the learner's shared_start counted, and what it should count."""
    w = _SHARED_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("shared_start") if mine is not None else None
    if not callable(fn):
        return ""
    a, b = SHARED_CASES[1]
    want = run_code(SOLUTION_SHARED).names["shared_start"](list(a), list(b))
    try:
        got = with_time_limit(lambda: fn(list(a), list(b)))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    color = "var(--ea-ink)" if got == want else "var(--ea-loss-text)"
    body = (_label(w["a"]) + _tokens_html(a, want) + _label(w["b"]) + _tokens_html(b, want)
            + f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:{color};margin-top:0.3rem">{html.escape(w["said"].format(got=repr(got)))}</div>'
            + _label(w["want"].format(want=want)))
    return yours.figure(w["title"], body, alt=w["title"])


# --- exercise 2: messages_for, with the time after the handbook ---------------------------------------------------------------

START_ORDER = '''def messages_for(question, now):
    system = "Now: " + now + "\\n\\n" + HANDBOOK
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]
'''
SOLUTION_ORDER = '''def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": question + "\\n\\nNow: " + now}]
'''
ORDER_CASES = [(QUESTIONS[0], NOW[0]), (QUESTIONS[1], NOW[1]), ("Do you rent out tandems?", "Saturday 21 November 2026, 16:02")]


def _text_of(messages) -> str | None:
    """The messages as one text, without Qwen3's markers: each role and content in turn; None when they are not a
    list of messages."""
    if not isinstance(messages, list) or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages):
        return None
    return "\n".join(f'{m.get("role")}: {m["content"]}' for m in messages)


def _parts(want) -> tuple[str, str]:
    """The question and the time of a case, from the course's messages for it."""
    question, now = want[1]["content"].split("\n\nNow: ")
    return question, now


def cache_friendly(got, want) -> bool:
    """Right: the system prompt starts with the whole handbook, the question is in a message of the user at the end,
    and the time comes after the handbook, so two calls a minute apart share the handbook at the start. The time may
    stand at the end of the system prompt or with the question, as long as it says "Now: " and the time."""
    if got == want:
        return True
    question, now = _parts(want)
    text = _text_of(got)
    if text is None or len(got) < 2 or got[0].get("role") != "system" or got[-1].get("role") != "user":
        return False
    if not got[0]["content"].startswith(HANDBOOK) or question not in got[-1]["content"]:
        return False
    return ("Now: " + now) in text and text.index("Now: " + now) > text.index(HANDBOOK)


def _time_first(got, text) -> bool:
    """The time is still before the handbook."""
    if not isinstance(got, list) or not got or not isinstance(got[0], dict):
        return False
    first = str(got[0].get("content"))
    return got[0].get("role") == "system" and HANDBOOK in first and not first.startswith(HANDBOOK)


def _handbook_after(got, text) -> bool:
    """The handbook is in the message of the user, with the question, instead of in the system prompt."""
    if not isinstance(got, list) or not got or not all(isinstance(m, dict) for m in got):
        return False
    return HANDBOOK in str(got[-1].get("content")) and not (got[0].get("role") == "system" and HANDBOOK in str(got[0].get("content")))


_ORDER_WORDS = {
    "en": {"title": "Your messages for the same question at 09:14 and at 09:15", "shared": "In common at the start: {n} of {k} characters",
           "first": "The first place where they differ:", "good": "The handbook is in common: a server can keep its work on it.",
           "bad": "They differ before the end of the handbook: the server computes it again for every new time."},
    "nl": {"title": "Je berichten voor dezelfde vraag om 09:14 en om 09:15", "shared": "Gemeen aan het begin: {n} van {k} tekens",
           "first": "De eerste plek waar ze verschillen:", "good": "Het handboek is gemeen: een server kan zijn werk eraan bewaren.",
           "bad": "Ze verschillen vóór het einde van het handboek: de server berekent het voor elke nieuwe tijd opnieuw."},
}


def order_picture(seen) -> str:
    """The learner's messages_for for the same question a minute apart, written out as one text: how many characters
    they share at the start, and the place where they first differ, with the text around it."""
    w = _ORDER_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("messages_for") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        one, two = (_text_of(with_time_limit(lambda now=now: fn(QUESTIONS[0], now))) for now in NOW)
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if one is None or two is None:
        return ""
    n = run_code(SOLUTION_SHARED).names["shared_start"](list(one), list(two))
    good = HANDBOOK in one[:n]
    around = one[max(0, n - 30):n + 12]
    at = min(30, n)
    shown = (f'<span style="color:var(--ea-ink-2)">…{html.escape(around[:at])}</span>'
             f'<span style="background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600">{html.escape(around[at:at + 1])}</span>'
             f'<span style="color:var(--ea-ink-2)">{html.escape(around[at + 1:])}…</span>')
    body = (f'<div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(w["shared"].format(n=number(n, seen.lang, 0), k=number(len(one), seen.lang, 0)))}</div>'
            + _label(w["first"]) + f'<div style="font-family:var(--ea-font-code);font-size:12px;white-space:pre-wrap;padding:0.3rem 0.5rem;background:var(--ea-inset);border-radius:4px">{shown}</div>'
            + f'<div style="font-family:var(--ea-font-ui);font-size:13px;margin-top:0.3rem;color:{"var(--ea-ink)" if good else "var(--ea-loss-text)"}">{html.escape(w["good"] if good else w["bad"])}</div>')
    return yours.figure(w["title"], body, alt=w["title"])


def _shown(v) -> str:
    text = _text_of(v) or repr(v)
    return text if len(text) <= 90 else text[:89] + "…"


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "shared_start": FunctionBox(
        name="shared_start",
        picture=shared_picture,
        function="shared_start",
        start=START_SHARED,
        solution=SOLUTION_SHARED,
        visible=[SHARED_CASES[0]],
        hidden=SHARED_CASES[1:],
        mistakes=[
            Mistake("from-one", wrong=SOLUTION_SHARED.replace("            return n\n", "            return n + 1\n"),
                    en="`{call}` gives `{got}`, one too many: it counts the first token that differs as well. Give back `n` as it is when the tokens differ: it already counted the ones before.",
                    nl="`{call}` geeft `{got}`, één te veel: hij telt het eerste token dat verschilt ook mee. Geef `n` terug zoals hij is als de tokens verschillen: hij telde de tokens ervoor al."),
            Mistake("longest", error=lambda e: isinstance(e, IndexError),
                    en="With `{call}`, your function stops with an IndexError: it looks past the end of the shorter list. Go only as far as the shorter of the two lists: `min(len(a), len(b))`.",
                    nl="Met `{call}` stopt je functie met een IndexError: hij kijkt voorbij het einde van de kortste lijst. Ga maar zo ver als de kortste van de twee lijsten: `min(len(a), len(b))`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`shared_start` still counts every place where the tokens are the same, also after the first place where they differ. For the cache only the start counts: stop at the first token that differs.",
                      "nl": "`shared_start` telt nog elke plaats waar de tokens gelijk zijn, ook na de eerste plaats waar ze verschillen. Voor de cache telt alleen het begin: stop bij het eerste token dat verschilt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the tokens up to the first place where the two lists differ.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de tokens tot de eerste plaats waar de twee lijsten verschillen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function also work when the lists are the same, when one is the start of the other, and when one is empty?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie ook als de lijsten gelijk zijn, als de ene het begin van de andere is, en als er een leeg is?"},
            "no_function": {"en": "Your box does not make a function `shared_start`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `shared_start`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `shared_start` is right in all {n} cases: it counts the tokens up to the first one that differs, and stops there.",
                        "nl": "Je `shared_start` klopt in alle {n} gevallen: hij telt de tokens tot het eerste dat verschilt, en stopt daar."},
        },
    ),
    "messages_for": FunctionBox(
        name="messages_for",
        picture=order_picture,
        function="messages_for",
        start=START_ORDER,
        solution=SOLUTION_ORDER,
        names={"HANDBOOK": HANDBOOK},
        visible=[ORDER_CASES[0]],
        hidden=ORDER_CASES[1:],
        compare=cache_friendly,
        show=_shown,
        mistakes=[
            Mistake("time-first", matches=_time_first,
                    en="With `{call}`, the time still comes before the handbook. Every minute the start of the prompt differs, so the server can reuse nothing of the handbook. Put the handbook first, and the time after it.",
                    nl="Met `{call}` staat de tijd nog vóór het handboek. Elke minuut verschilt het begin van de prompt, dus kan de server niets van het handboek hergebruiken. Zet het handboek eerst, en de tijd erachter."),
            Mistake("handbook-last", matches=_handbook_after,
                    en="With `{call}`, the handbook is no longer at the start, in the system prompt: the question comes before it, and the question differs every time. Keep the handbook as the system prompt, and put the time with the question.",
                    nl="Met `{call}` staat het handboek niet meer aan het begin, in de system prompt: de vraag komt ervoor, en de vraag verschilt elke keer. Houd het handboek als system prompt, en zet de tijd bij de vraag."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`messages_for` still puts the time at the very start of the system prompt. Move it to after the handbook: after the question, in the message of the user.",
                      "nl": "`messages_for` zet de tijd nog helemaal aan het begin van de system prompt. Zet hem na het handboek: na de vraag, in het bericht van de gebruiker."},
            "wrong": {"en": "`{call}` gives `{got}`. The system prompt should be the handbook, and the message of the user the question, a blank line, and `Now: ` with the time.",
                      "nl": "`{call}` geeft `{got}`. De system prompt hoort het handboek te zijn, en het bericht van de gebruiker de vraag, een lege regel, en `Now: ` met de tijd."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`. Does your function use the question and the time it gets?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`. Gebruikt je functie de vraag en de tijd die hij krijgt?"},
            "no_function": {"en": "Your box does not make a function `messages_for`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `messages_for`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `messages_for` is right in all {n} cases: the handbook first and the same every time, and the time after it, so a server can keep its work on the handbook.",
                        "nl": "Je `messages_for` klopt in alle {n} gevallen: het handboek eerst en elke keer hetzelfde, en de tijd erachter, zodat een server zijn werk aan het handboek kan bewaren."},
        },
    ),
}

REQUIRED = ["shared_start", "messages_for", "breaks"]

CASES = [
    ("shared_start", START_SHARED, "empty", None),
    ("shared_start", "", "empty", None),
    ("shared_start", SOLUTION_SHARED, "correct", None),
    ("shared_start", """def shared_start(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))
""", "correct", None),
    ("shared_start", SOLUTION_SHARED.replace("            return n\n", "            return n + 1\n"), "wrong", "from-one"),
    ("shared_start", SOLUTION_SHARED.replace("min(len(a), len(b))", "max(len(a), len(b))"), "error", "longest"),
    ("shared_start", SOLUTION_SHARED.replace("        n = n + 1\n    return n\n", "        n = n + 1\n    return 0\n"), "almost", None),
    ("messages_for", START_ORDER, "empty", None),
    ("messages_for", SOLUTION_ORDER, "correct", None),
    ("messages_for", '''def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK + "\\n\\nNow: " + now},
            {"role": "user", "content": question}]
''', "correct", None),
    ("messages_for", '''def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": "Now: " + now + "\\n\\n" + question}]
''', "correct", None),
    ("messages_for", START_ORDER.replace('    system = "Now: " + now + "\\n\\n" + HANDBOOK\n', '    system = "Now: " + now + "\\n" + HANDBOOK\n'), "wrong", "time-first"),
    ("messages_for", '''def messages_for(question, now):
    return [{"role": "user", "content": question + "\\n\\nNow: " + now + "\\n\\n" + HANDBOOK}]
''', "wrong", "handbook-last"),
    ("messages_for", '''def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": question}]
''', "wrong", None),
    ("messages_for", '''def messages_for(question, now):
    return [{"role": "system", "content": HANDBOOK},
            {"role": "user", "content": "How much does a full service cost?\\n\\nNow: Monday 16 November 2026, 09:14"}]
''', "almost", None),
]


# --- the demo: your messages_for, against the course's own server -------------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_messages_for(text):
    """The learner's messages_for from exercise 2, when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["messages_for"].check(text).state != "correct":
        return None
    return run_code(text, {"HANDBOOK": HANDBOOK}).names.get("messages_for")


def _post(url: str, body: dict) -> dict:
    import urllib.request
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def demo(messages_for, questions: list, words: dict, lang: str) -> str:
    """The learner's messages_for for each question, with the time of the computer, sent one after the other to the
    course's server with Qwen3 behind it, the first with the cache switched off: per request its prompt tokens, how
    many came from the cache, its seconds, and the answer."""
    import datetime
    running = serve.model_server()
    rows = []
    for i, question in enumerate(questions):
        now = datetime.datetime.now().strftime("%A %d %B %Y, %H:%M:%S")
        try:
            body = {"model": serve.MODEL, "messages": messages_for(question, now), "max_tokens": 60, "cache_prompt": i > 0}
            t = time.perf_counter()
            answer = _post(running.url, body)
        except Exception as e:   # the learner's code, against the real server; it may have run out of memory
            said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
            rows.append(f"<p><em>{html.escape(said)}</em></p>")
            break
        seconds = number(time.perf_counter() - t, lang, 1)
        usage = answer.get("usage", {})
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        rows.append(f'<div style="margin:0.4rem 0"><div style="font-family:var(--ea-font-ui);font-size:13px;color:var(--ea-ink)">{html.escape(question)}</div>'
                    f'<div style="{_LABEL}">{html.escape(words["done"].format(given=number(usage.get("prompt_tokens", 0), lang, 0), cached=number(cached, lang, 0), seconds=seconds))}</div>'
                    f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-pred-text);white-space:pre-wrap">{html.escape(answer["choices"][0]["message"]["content"])}</div></div>')
    return yours.figure(words["title"], "".join(rows), alt=words["title"])
