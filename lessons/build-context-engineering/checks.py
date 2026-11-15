# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-context-engineering: a function that keeps the system prompt and the last messages of a
conversation, and one that puts a summary of what drops out in the system prompt, against a stand-in of the server;
the course's recorded runs (data/course-dev-prompts-qwen.json): a dinner talk whose first message holds a fact, the
last question asked with the whole talk, with a window of the last messages, with a note in the system prompt, and
with a summary; their pictures, and the demo."""
import copy
import html
import json
import re

from everything_ai import apicheck, chat, paths, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header

SYSTEM = "You help plan dinners. Answer briefly."
TALK = [
    ("user", "Hi! I am planning a dinner for six friends on Saturday. One thing first: I am allergic to nuts."),
    ("assistant", "Thanks for telling me, I will keep your nut allergy in mind. What kind of dinner do you have in mind?"),
    ("user", "Something Italian, I think. What starter could I make?"),
    ("assistant", "A tomato bruschetta is quick and fresh, and you can make it in advance."),
    ("user", "Good idea. And a main course?"),
    ("assistant", "A lasagna works well for six people, and you can bake it before the guests arrive."),
    ("user", "Which wine goes with lasagna?"),
    ("assistant", "A Chianti or another light red wine goes well with it."),
    ("user", "What music should I play?"),
    ("assistant", "Soft Italian jazz or an acoustic playlist keeps the mood relaxed."),
]
LAST = "Last question: how many friends did I say are coming, and what can I not eat? Answer in one sentence."
KEEP = 4   # the window: the last four messages of the conversation, and the new question
NOTE = "Notes about the user: allergic to nuts."
SUMMARIZE = ("Here is a conversation between a user and an assistant:\n\n{talk}\n\nSummarize this conversation in at most "
             "three sentences. Keep every fact about the user that later answers may need.")


def messages(talk) -> list:
    return [{"role": role, "content": text} for role, text in talk]


def transcript(talk: list) -> str:
    """A conversation as text, a line per message: User: ... or Assistant: ..."""
    return "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in talk)


SEEDS = 8
DATA = paths.ROOT / "data" / "course-dev-prompts-qwen.json"
SUMMARY_LINE = "\n\nSummary of the conversation so far: "


def _recorded_summary() -> str | None:
    """The summary Qwen3 wrote of the messages that drop out, as the course recorded it; None before it is recorded."""
    try:
        part = json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-context-engineering"]["summary"]
    except (OSError, KeyError, ValueError):
        return None
    if part["ask"]["messages"][0]["content"] != SUMMARIZE.format(talk=transcript(messages(TALK[:-KEEP]))):
        return None   # the prompt for the summary changed: the recorder asks for it again first
    return part["answers"][0]["text"]


RECORD = {
    "whole": {"messages": [{"role": "system", "content": SYSTEM}] + messages(TALK) + [{"role": "user", "content": LAST}],
              "max_tokens": 60, "seeds": SEEDS},
    "window": {"messages": [{"role": "system", "content": SYSTEM}] + messages(TALK[-KEEP:]) + [{"role": "user", "content": LAST}],
               "max_tokens": 60, "seeds": SEEDS},
    "notes": {"messages": [{"role": "system", "content": SYSTEM + "\n\n" + NOTE}] + messages(TALK[-KEEP:])
              + [{"role": "user", "content": LAST}], "max_tokens": 60, "seeds": SEEDS},
    "summary": {"messages": [{"role": "user", "content": SUMMARIZE.format(talk=transcript(messages(TALK[:-KEEP])))}],
                "max_tokens": 120, "seeds": 1},
}
if _recorded_summary() is not None:   # the window with the summary in the system prompt: asked once the summary is there
    RECORD["compact"] = {"messages": [{"role": "system", "content": SYSTEM + SUMMARY_LINE + _recorded_summary()}]
                         + messages(TALK[-KEEP:]) + [{"role": "user", "content": LAST}], "max_tokens": 60, "seeds": SEEDS}
MODES = ("whole", "window", "notes", "compact")


def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"]["build-context-engineering"]


def answers(mode: str) -> list:
    return course()[mode]["answers"]


def knows(text: str) -> bool:
    """Whether an answer knows the allergy: it names nuts."""
    return re.search(r"\bnuts?\b|\bnut-free\b", text.lower()) is not None


def six(text: str) -> bool:
    """Whether an answer says how many friends are coming: six."""
    return re.search(r"\bsix\b|\b6\b", text.lower()) is not None


def findings() -> dict:
    """What the recorded runs show, per way of asking: how many answers named the nut allergy, and the prompt
    tokens of the request; and the summary Qwen3 wrote of the messages that dropped out."""
    out = {}
    for mode in MODES:
        got = answers(mode)
        out[mode] = {"knows": sum(knows(a["text"]) for a in got), "six": sum(six(a["text"]) for a in got), "runs": len(got),
                     "prompt_tokens": got[0]["prompt_tokens"]}
    out["summary"] = course()["summary"]["answers"][0]["text"]
    out["summary_cost"] = {k: course()["summary"]["answers"][0][k] for k in ("prompt_tokens", "tokens")}
    return out


_MODE_WORDS = {
    "en": {"title": "What Qwen3 still knows of the allergy, eight runs per way of asking",
           "modes": {"whole": "the whole talk", "window": "the last {keep} messages", "notes": "the window and a note",
                     "compact": "the window and a summary"},
           "knows": "named the nut allergy", "forgot": "did not name it", "tokens": "{n} prompt tokens",
           "count": "{k} of {n}"},
    "nl": {"title": "Wat Qwen3 nog weet van de allergie, acht runs per manier van vragen",
           "modes": {"whole": "het hele gesprek", "window": "de laatste {keep} berichten", "notes": "het venster en een notitie",
                     "compact": "het venster en een samenvatting"},
           "knows": "noemde de notenallergie", "forgot": "noemde hem niet", "tokens": "{n} prompt tokens",
           "count": "{k} van de {n}"},
}


def modes_picture(lang: str) -> str:
    """Per way of asking a row: a square per run, in the colour of model output when the answer took the allergy into
    account and as a mistake when it did not, with the answer on hovering; and a bar of the prompt tokens, as input."""
    w = _MODE_WORDS[lang]
    f = findings()
    most = max(f[m]["prompt_tokens"] for m in MODES)
    rows = []
    for mode in MODES:
        cells = "".join(f'<span title="{html.escape(" ".join(a["text"].split())[:120])}" style="display:inline-block;width:14px;height:14px;margin-right:3px;border-radius:2px;'
                        f'background:{"var(--ea-pred)" if knows(a["text"]) else "var(--ea-loss)"}"></span>' for a in answers(mode))
        tokens = f[mode]["prompt_tokens"]
        bar = (f'<span style="display:inline-block;height:10px;width:{round(100 * tokens / most)}px;background:var(--ea-data-wash);'
               f'border:1px solid var(--ea-data);border-radius:2px;vertical-align:middle"></span>')
        rows.append(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.3rem 0.6rem;margin:0.25rem 0">'
                    f'<span style="{_LABEL};min-width:11rem">{html.escape(w["modes"][mode].format(keep=KEEP))}</span>'
                    f'<span>{cells}</span><span style="{_LABEL};min-width:3.5rem">{html.escape(w["count"].format(k=f[mode]["knows"], n=f[mode]["runs"]))}</span>'
                    f'{bar}<span style="{_LABEL}">{html.escape(w["tokens"].format(n=tokens))}</span></div>')
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
                     f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{c})"></span>{html.escape(w[k])}</span>'
                     for k, c in (("knows", "pred"), ("forgot", "loss")))
    return yours.figure(w["title"], "".join(rows) + f'<div style="margin-top:0.3rem">{legend}</div>', alt=w["title"])
_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"


def _label(text: str) -> str:
    return f'<div style="{_LABEL};margin:0.2rem 0">{html.escape(text)}</div>'


# --- exercise 1: window ---------------------------------------------------------------------------------------------------------------

START_WINDOW = '''def window(messages, keep):
    out = []
    for message in messages[len(messages) - keep:]:
        out.append(message)
    return out
'''
SOLUTION_WINDOW = '''def window(messages, keep):
    out = [messages[0]]
    for message in messages[max(1, len(messages) - keep):]:
        out.append(message)
    return out
'''
_SHORT = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello! What are you cooking?"}]
WINDOW_CASES = [([{"role": "system", "content": SYSTEM}] + messages(TALK), KEEP), ([{"role": "system", "content": SYSTEM}] + messages(TALK), 0),
                (list(_SHORT), 1), (list(_SHORT), 3), (list(_SHORT), 5)]
window = run_code(SOLUTION_WINDOW).names["window"]

_WINDOW_WORDS = {
    "en": {"title": "What your window sends of the dinner talk, with keep {keep}", "sent": "sent", "left": "left out",
           "twice": "sent twice", "system": "system prompt"},
    "nl": {"title": "Wat je window van het gesprek over het etentje verstuurt, met keep {keep}", "sent": "verstuurd", "left": "weggelaten",
           "twice": "twee keer verstuurd", "system": "system prompt"},
}


def window_picture(seen) -> str:
    """Every message of the dinner talk as a row, sent or left out by the learner's window with keep 4: what is sent
    in the colour of input data, what is left out grey, and a message sent twice as a mistake."""
    w = _WINDOW_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("window") if mine is not None else None
    if not callable(fn):
        return ""
    talk = [{"role": "system", "content": SYSTEM}] + messages(TALK)
    try:
        got = with_time_limit(lambda: fn([dict(m) for m in talk], KEEP))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    got = got if isinstance(got, list) else []
    rows = []
    for m in talk:
        n = sum(1 for g in got if g == m)
        state = w["twice"] if n > 1 else w["sent"] if n else w["left"]
        style = ("background:var(--ea-loss-wash);color:var(--ea-loss-text)" if n > 1 else
                 "background:var(--ea-data-wash);color:var(--ea-ink)" if n else "background:var(--ea-inset);color:var(--ea-ink-3)")
        role = w["system"] if m["role"] == "system" else m["role"]
        rows.append(f'<div style="display:grid;grid-template-columns:6rem minmax(0,1fr) 7rem;gap:0.5rem;align-items:center;'
                    f'margin:2px 0;padding:0.15rem 0.4rem;border-radius:3px;{style}">'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px">{html.escape(role)}</span>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{html.escape(m["content"])}</span>'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;text-align:right">{html.escape(state)}</span></div>')
    title = w["title"].format(keep=KEEP)
    return yours.figure(title, "".join(rows), alt=title)


# --- exercise 2: compact -----------------------------------------------------------------------------------------------------------

START_COMPACT = ('import json\nimport urllib.request\n\n\ndef compact(url, messages, keep):\n'
                 '    old = messages[1:max(1, len(messages) - keep)]\n'
                 '    body = {"model": "qwen3-0.6b", "max_tokens": 120,\n'
                 '            "messages": [{"role": "user", "content": SUMMARIZE.format(talk=transcript(old))}]}\n'
                 '    data = json.dumps(body).encode()\n'
                 '    request = urllib.request.Request(url + "/chat/completions", data=data,\n'
                 '                                     headers={"Content-Type": "application/json"})\n'
                 '    with urllib.request.urlopen(request) as response:\n'
                 '        answer = json.loads(response.read())\n'
                 '    summary = answer["choices"][0]["message"]["content"]\n'
                 '    return window(messages, keep)\n')
SOLUTION_COMPACT = START_COMPACT.replace(
    '    return window(messages, keep)\n',
    '    out = window(messages, keep)\n'
    '    system = {"role": "system",\n'
    '              "content": messages[0]["content"] + "\\n\\nSummary of the conversation so far: " + summary}\n'
    '    return [system] + out[1:]\n')


def _summary_of(talk: list) -> str:
    return "The stand-in's summary of {n} messages.".format(n=len(talk))


def _compact_stand_in(talk_messages: list, keep: int):
    old = talk_messages[1:max(1, len(talk_messages) - keep)]
    return serve.StandIn({SUMMARIZE.format(talk=transcript(old)): _summary_of(old)})


def _posts(log: list) -> list:
    return [e["body"] for e in log if e["method"] == "POST" and isinstance(e["body"], dict)]


def _judge_compact(talk_messages: list, keep: int):
    old = talk_messages[1:max(1, len(talk_messages) - keep)]
    asked = SUMMARIZE.format(talk=transcript(old))
    want = [{"role": "system", "content": talk_messages[0]["content"] + SUMMARY_LINE + _summary_of(old)}] + window(talk_messages, keep)[1:]

    def judge(got, running):
        sent = _posts(running.log)
        said = [m.get("content") for b in sent for m in (b.get("messages") or []) if isinstance(m, dict)]
        if asked not in said:
            return "asked-what"
        if not isinstance(got, list) or not got:
            return "other"
        if got == window(talk_messages, keep):
            return "no-summary"
        if got[1:] != want[1:]:
            return "window"
        if isinstance(got[0], dict) and _summary_of(old) in str(got[0].get("content")) and got[0] != want[0]:
            return "form"
        return None if got == want else "other"
    return judge


_TALK_MESSAGES = [{"role": "system", "content": SYSTEM}] + messages(TALK)


def _changes_its_list(_, text) -> bool:
    """The function gives back the right messages, but changed the list (or the system message) it got."""
    fn = run_code(text, {"SUMMARIZE": SUMMARIZE, "transcript": transcript, "window": window}).names.get("compact")
    if not callable(fn):
        return False
    talk = copy.deepcopy(_TALK_MESSAGES)
    running = serve.start(_compact_stand_in(_TALK_MESSAGES, KEEP))
    try:
        with_time_limit(lambda: fn(running.url, talk, KEEP))
    except Exception:   # the learner's code; the check itself already said what is wrong with it
        return False
    finally:
        running.stop()
    return talk != _TALK_MESSAGES
COMPACT_CASES = [Case(lambda: _compact_stand_in(_TALK_MESSAGES, KEEP), (URL, _TALK_MESSAGES, KEEP), _judge_compact(_TALK_MESSAGES, KEEP)),
                 Case(lambda: _compact_stand_in(_TALK_MESSAGES[:7], 2), (URL, _TALK_MESSAGES[:7], 2), _judge_compact(_TALK_MESSAGES[:7], 2))]

_COMPACT_WORDS = {
    "en": {"title": "What your compact gives back for the dinner talk, with keep 4", "none": "Your function gave back nothing to send yet.",
           "error": "Your function stopped: {why}"},
    "nl": {"title": "Wat je compact teruggeeft voor het gesprek over het etentje, met keep 4", "none": "Je functie gaf nog niets terug om te versturen.",
           "error": "Je functie stopte: {why}"},
}


def compact_picture(seen) -> str:
    """The messages the learner's compact gives back for the dinner talk with keep 4, against the stand-in: the system
    prompt with what came after it, and each message that goes along."""
    w = _COMPACT_WORDS[seen.lang]
    run = seen.run
    fn = run.names.get("compact") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    got, error, _running = apicheck.call(fn, COMPACT_CASES[0])
    if error is not None:
        return yours.figure(w["title"], f'<div style="{_LABEL}">{html.escape(w["error"].format(why=f"{type(error).__name__}: {error}"))}</div>', alt=w["title"])
    if not isinstance(got, list) or not got:
        return yours.figure(w["title"], f'<div style="{_LABEL}">{html.escape(w["none"])}</div>', alt=w["title"])
    rows = []
    for m in got:
        m = m if isinstance(m, dict) else {"role": "?", "content": repr(m)}
        system = m.get("role") == "system"
        style = "background:var(--ea-data-wash)" if not system else "background:var(--ea-inset)"
        rows.append(f'<div style="margin:2px 0;padding:0.2rem 0.4rem;border-radius:3px;{style}">'
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(str(m.get("role")))}</span> '
                    f'<span style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink);white-space:pre-wrap">{html.escape(str(m.get("content")))}</span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


def _not_bytes(e) -> bool:
    return isinstance(e, TypeError) and "bytes" in str(e)


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "window": FunctionBox(
        name="window",
        picture=window_picture,
        function="window",
        start=START_WINDOW,
        solution=SOLUTION_WINDOW,
        visible=[WINDOW_CASES[0]],
        hidden=WINDOW_CASES[1:],
        mistakes=[
            Mistake("no-system", wrong=START_WINDOW,
                    en="`{call}` gives `{got}`: the system prompt is gone, so the model no longer knows its task. Start the list with `messages[0]`, the system prompt, and add the last `keep` messages after it.",
                    nl="`{call}` geeft `{got}`: de system prompt is weg, dus het model kent zijn taak niet meer. Begin de lijst met `messages[0]`, de system prompt, en zet de laatste `keep` berichten erachter."),
            Mistake("twice", state="almost", wrong=SOLUTION_WINDOW.replace("max(1, len(messages) - keep)", "len(messages) - keep"),
                    en="Right when the conversation is long. But `{call}` gives `{got}`: when `keep` reaches back to the start, the system prompt comes along a second time. Start the slice at 1 or later, with `max(1, len(messages) - keep)`.",
                    nl="Goed als het gesprek lang is. Maar `{call}` geeft `{got}`: als `keep` tot het begin terugreikt, komt de system prompt een tweede keer mee. Laat de slice bij 1 of later beginnen, met `max(1, len(messages) - keep)`."),
            Mistake("minus-zero", state="almost", wrong=SOLUTION_WINDOW.replace("messages[max(1, len(messages) - keep):]", "messages[-keep:]"),
                    en="Right for most cases. But `{call}` gives `{got}`: with `keep` 0, `messages[-0:]` is `messages[0:]`, the whole list. Count from the front instead: `max(1, len(messages) - keep)`.",
                    nl="Goed in de meeste gevallen. Maar `{call}` geeft `{got}`: met `keep` 0 is `messages[-0:]` hetzelfde als `messages[0:]`, de hele lijst. Tel liever vanaf het begin: `max(1, len(messages) - keep)`."),
            Mistake("first", wrong=SOLUTION_WINDOW.replace("messages[max(1, len(messages) - keep):]", "messages[1:1 + keep]"),
                    en="`{call}` gives `{got}`: the first messages after the system prompt, not the last. The window keeps what was said last, so the slice starts at `len(messages) - keep`.",
                    nl="`{call}` geeft `{got}`: de eerste berichten na de system prompt, niet de laatste. Het venster houdt wat het laatst gezegd is, dus de slice begint bij `len(messages) - keep`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`window` keeps the last `keep` messages, but leaves out the system prompt. Put `messages[0]` first, and make sure it never comes along twice.",
                      "nl": "`window` houdt de laatste `keep` berichten, maar laat de system prompt weg. Zet `messages[0]` vooraan, en zorg dat hij nooit twee keer meekomt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: the system prompt, then the last `keep` messages after it.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: de system prompt, en daarna de laatste `keep` berichten."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when `keep` is 0, and when it is larger than the conversation?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als `keep` 0 is, en als hij groter is dan het gesprek?"},
            "no_function": {"en": "Your box does not make a function `window`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `window`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `window` is right in all {n} cases: the system prompt once, and the last `keep` messages after it.",
                        "nl": "Je `window` klopt in alle {n} gevallen: de system prompt één keer, en de laatste `keep` berichten erachter."},
        },
    ),
    "compact": ApiBox(
        name="compact",
        picture=compact_picture,
        function="compact",
        start=START_COMPACT,
        solution=SOLUTION_COMPACT,
        names={"SUMMARIZE": SUMMARIZE, "transcript": transcript, "window": window},
        cases=COMPACT_CASES,
        form=[Mistake("changed", state="almost", matches=_changes_its_list,
                      en="Your function gives back the right messages, but it also changed the messages it got: the system prompt in them now holds the summary. An app keeps that list as the conversation; changed in place, the next summary would land after the last one. Make a new dictionary for the system message, and leave `messages` as it was.",
                      nl="Je functie geeft de goede berichten terug, maar hij veranderde ook de berichten die hij kreeg: de system prompt erin bevat nu de samenvatting. Een app bewaart die lijst als het gesprek; als hij zelf verandert, komt de volgende samenvatting achter de vorige. Maak een nieuwe dictionary voor het system-bericht, en laat `messages` zoals hij was.")],
        mistakes=[
            Mistake("data-text", error=_not_bytes,
                    en="urllib sends bytes, not text. Keep `.encode()` after `json.dumps(body)`, as the box had it.",
                    nl="urllib verstuurt bytes, geen tekst. Laat `.encode()` achter `json.dumps(body)` staan, zoals het in het vak stond."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`compact` asks for a summary, but gives back the window without it. Put the summary at the end of the system prompt, after `\\n\\nSummary of the conversation so far: `.",
                      "nl": "`compact` vraagt om een samenvatting, maar geeft het venster zonder terug. Zet de samenvatting aan het eind van de system prompt, na `\\n\\nSummary of the conversation so far: `."},
            "correct": {"en": "Your `compact` summarizes what drops out of the window, and puts the summary in the system prompt, in front of the last messages.",
                        "nl": "Je `compact` vat samen wat uit het venster valt, en zet de samenvatting in de system prompt, vóór de laatste berichten."},
            "asked-what": {"en": "With `{call}`, the request for the summary is not about the messages that drop out. Keep `old` as the box had it, the messages between the system prompt and the window, and send `SUMMARIZE.format(talk=transcript(old))`.",
                           "nl": "Met `{call}` gaat het request om een samenvatting niet over de berichten die wegvallen. Houd `old` zoals het in het vak stond, de berichten tussen de system prompt en het venster, en stuur `SUMMARIZE.format(talk=transcript(old))`."},
            "no-summary": {"en": "With `{call}`, your function gives back the window without the summary. Make a new system message with the content of `messages[0]`, then `\\n\\nSummary of the conversation so far: ` and the summary.",
                           "nl": "Met `{call}` geeft je functie het venster zonder de samenvatting terug. Maak een nieuw system-bericht met de content van `messages[0]`, dan `\\n\\nSummary of the conversation so far: ` en de samenvatting."},
            "window": {"en": "With `{call}`, the messages after the system prompt are not the last `keep` of the conversation. Keep them as `window` gives them, and change only the system prompt.",
                       "nl": "Met `{call}` zijn de berichten na de system prompt niet de laatste `keep` van het gesprek. Houd ze zoals `window` ze geeft, en verander alleen de system prompt."},
            "form": {"en": "With `{call}`, the summary is in the system prompt, but not in the form the lesson asks: the content of `messages[0]`, then `\\n\\nSummary of the conversation so far: ` and the summary.",
                     "nl": "Met `{call}` staat de samenvatting in de system prompt, maar niet in de vorm die de les vraagt: de content van `messages[0]`, dan `\\n\\nSummary of the conversation so far: ` en de samenvatting."},
            "other": {"en": "With `{call}`, your function gives back `{got}`. Give back a list of messages: the system prompt with the summary, then the last `keep` messages.",
                      "nl": "Met `{call}` geeft je functie `{got}` terug. Geef een lijst berichten terug: de system prompt met de samenvatting, en daarna de laatste `keep` berichten."},
        },
    ),
}

REQUIRED = ["window", "compact", "notes"]

CASES = [
    ("window", START_WINDOW, "empty", None),
    ("window", "", "empty", None),
    ("window", SOLUTION_WINDOW, "correct", None),
    ("window", "def window(messages, keep):\n    start = max(1, len(messages) - keep)\n    return [messages[0]] + messages[start:]\n", "correct", None),
    ("window", "def window(messages, keep):\n    return messages[len(messages) - keep:]\n", "wrong", "no-system"),
    ("window", SOLUTION_WINDOW.replace("max(1, len(messages) - keep)", "len(messages) - keep"), "almost", "twice"),
    ("window", SOLUTION_WINDOW.replace("messages[max(1, len(messages) - keep):]", "messages[-keep:]"), "almost", "minus-zero"),
    ("window", SOLUTION_WINDOW.replace("messages[max(1, len(messages) - keep):]", "messages[1:1 + keep]"), "wrong", "first"),
    ("compact", START_COMPACT, "empty", None),
    ("compact", SOLUTION_COMPACT, "correct", None),
    ("compact", SOLUTION_COMPACT.replace("transcript(old)", "transcript(messages)"), "wrong", "asked-what"),
    ("compact", START_COMPACT.replace("    return window(messages, keep)\n", "    return window(messages, keep) + []\n"), "wrong", "no-summary"),
    ("compact", SOLUTION_COMPACT.replace("    return [system] + out[1:]\n", "    return [system] + messages[1:]\n"), "wrong", "window"),
    ("compact", SOLUTION_COMPACT.replace('"\\n\\nSummary of the conversation so far: "', '" "'), "wrong", "form"),
    ("compact", SOLUTION_COMPACT.replace("    return [system] + out[1:]\n", "    return summary\n"), "wrong", "other"),
    ("compact", SOLUTION_COMPACT.replace("json.dumps(body).encode()", "json.dumps(body)"), "error", "data-text"),
    ("compact", START_COMPACT.replace("    return window(messages, keep)\n",
                                      '    out = window(messages, keep)\n'
                                      '    messages[0]["content"] = messages[0]["content"] + "\\n\\nSummary of the conversation so far: " + summary\n'
                                      '    return out\n'), "almost", "changed"),
]


# --- the demo: your window and compact, against the course's own server -------------------------------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


def learner_compact(window_text, compact_text):
    """The learner's compact from exercise 2 with their own window from exercise 1, when both are right; else None."""
    from everything_ai.check import MISSING
    for name, text in (("window", window_text), ("compact", compact_text)):
        if text is MISSING or text is None or EXERCISES[name].check(text).state != "correct":
            return None
    mine = run_code(window_text).names["window"]
    return run_code(compact_text, {"SUMMARIZE": SUMMARIZE, "transcript": transcript, "window": mine}).names["compact"]


def _post(url: str, body: dict) -> dict:
    import urllib.request
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def demo(compact, keep: int, words: dict, lang: str) -> str:
    """The learner's compact on the dinner talk with a window of `keep` messages, through the course's server with
    Qwen3 behind it; then the last question with what compact gave back: the summary, the answer, whether it took the
    nut allergy, and the prompt tokens."""
    import time
    running = serve.model_server()
    talk = [{"role": "system", "content": SYSTEM}] + messages(TALK)
    start = time.perf_counter()
    try:
        got = compact(running.url, [dict(m) for m in talk], keep)
        answer = _post(running.url, {"model": serve.MODEL, "messages": list(got) + [{"role": "user", "content": LAST}],
                                     "max_tokens": 60, "seed": 0})
    except Exception as e:   # the learner's code, against the real server; it may have run out of memory
        said = words["memory"] if serve.out_of_memory(e) else words["failed"].format(why=f"{type(e).__name__}: {e}")
        return f"<p><em>{html.escape(said)}</em></p>"
    seconds = time.perf_counter() - start
    system = str(got[0].get("content")) if got and isinstance(got[0], dict) else ""
    summary = system.split(SUMMARY_LINE, 1)[1] if SUMMARY_LINE in system else ""
    text = answer["choices"][0]["message"]["content"]
    ok = knows(text)
    code = "font-family:var(--ea-font-code);font-size:13px;white-space:pre-wrap"
    body = (_label(words["summary"]) + f'<div style="{code};color:var(--ea-pred-text)">{html.escape(summary.strip())}</div>'
            + _label(words["answer"]) + f'<div style="{code};color:var(--ea-pred-text)">{html.escape(text.strip())}</div>'
            + f'<div style="font-family:var(--ea-font-ui);font-size:13px;margin-top:0.3rem;color:{"var(--ea-ink)" if ok else "var(--ea-loss-text)"}">{html.escape(words["knows"] if ok else words["forgot"])}</div>'
            + _label(words["tokens"].format(tokens=answer["usage"]["prompt_tokens"], seconds=number(seconds, lang, 0))))
    return yours.figure(words["title"], body, alt=words["title"])


def number(x: float, lang: str, digits: int = 1) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text
