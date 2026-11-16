# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-errors-retries: which status codes are worth another try, and a function that tries again with
exponential backoff, against stand-ins of a server that says no at first (everything_ai.serve.StandIn with `fail`);
the worked example, run against such a stand-in; and the pictures: the course's waits, and the learner's own tries on
a time line, as the stand-in saw them."""
import contextlib
import html
import io
import json
import urllib.error

from everything_ai import apicheck, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

QUESTION = "Name three colors of the rainbow."
ANSWER = "Red, orange and yellow."

# --- the worked example: a server that is busy -----------------------------------------------------------------------

WORKED = '''import json
import urllib.error
import urllib.request

body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}
request = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
except urllib.error.HTTPError as e:
    print(e.code, e.headers["Retry-After"])
    print(json.loads(e.read())["error"]["message"])
'''


def worked() -> str:
    """What the worked example prints against a stand-in that answers its first request with 429, Retry-After 2."""
    running = serve.start(serve.StandIn(fail=[429], retry=2))
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(WORKED, {"url": running.url})
    finally:
        running.stop()
    return out.getvalue()


# --- the picture of the idea: waits that double -------------------------------------------------------------------------

_WAIT_WORDS = {
    "en": {"title": "Four tries, with waits that double", "wait": "wait {s} s", "axis": "seconds after the first try",
           "try": "try {n}", "ok": "200: the answer"},
    "nl": {"title": "Vier pogingen, met wachttijden die verdubbelen", "wait": "wacht {s} s", "axis": "seconden na de eerste poging",
           "try": "poging {n}", "ok": "200: het antwoord"},
}
COURSE_TRIES = [429, 503, 503, 200]   # what the server answers, try after try, in the picture of the idea
COURSE_BASE = 1


def all_waits(tries: int = 10, base: float = 1) -> dict:
    """The waits between `tries` tries that double from `base`: there is no wait after the last try."""
    waits = [base * 2 ** attempt for attempt in range(tries - 1)]
    return {"tries": tries, "last": waits[-1], "total": sum(waits)}


def course_times(base: float = COURSE_BASE, statuses=COURSE_TRIES) -> list:
    """The second at which each try goes out, when the k-th wait is base * 2 ** k."""
    times, t = [], 0.0
    for k in range(len(statuses)):
        times.append(t)
        t += base * 2 ** k
    return times


def _timeline(rows: list, end: float, lang: str, axis: str, wait_label: str, digits: int) -> str:
    """Rows of tries on one time line: each row (label, [(second, status)]), a dot per try in the colour of what it got,
    with the wait between two tries written above."""
    W, L, R, row_h = 720, 110, 30, 62
    scale = (W - L - R) / end

    def x(s):
        return L + s * scale

    def n(v):
        s = f"{v:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:g}"
        return s.replace(".", ",") if lang == "nl" else s
    parts = []
    for r, (label, tries) in enumerate(rows):
        y = 34 + r * row_h
        parts.append(f'<text x="{L - 12}" y="{y + 4}" text-anchor="end" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink)">{html.escape(label)}</text>')
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y}" y2="{y}" style="stroke:var(--ea-line)"/>')
        for i, (t, status) in enumerate(tries):
            if i:
                t0 = tries[i - 1][0]
                parts.append(f'<text x="{(x(t0) + x(t)) / 2:.1f}" y="{y - 10}" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">'
                             f'{html.escape(wait_label.format(s=n(t - t0)))}</text>')
            good = status == 200
            colour = "var(--ea-pred-text)" if good else "var(--ea-loss-text)"
            parts.append(f'<circle cx="{x(t):.1f}" cy="{y}" r="6" style="fill:{colour}"/>')
            parts.append(f'<text x="{x(t):.1f}" y="{y + 22}" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-code);fill:{colour}">{status}</text>')
    y = 34 + len(rows) * row_h - 18
    step = next(k for k in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if end / k <= 8)
    k = 0
    while k * step <= end + 1e-9:
        parts.append(f'<line x1="{x(k * step):.1f}" x2="{x(k * step):.1f}" y1="{y}" y2="{y + 5}" style="stroke:var(--ea-ink-3)"/>')
        parts.append(f'<text x="{x(k * step):.1f}" y="{y + 18}" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{n(k * step)}</text>')
        k += 1
    parts.append(f'<line x1="{L}" x2="{x(end):.1f}" y1="{y}" y2="{y}" style="stroke:var(--ea-ink-3)"/>')
    parts.append(f'<text x="{(L + x(end)) / 2:.1f}" y="{y + 36}" text-anchor="middle" style="font-size:11px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">{html.escape(axis)}</text>')
    return f'<svg viewBox="0 0 {W} {y + 44}" role="img" style="width:100%;max-width:46rem;height:auto">{"".join(parts)}</svg>'


def waits_picture(lang: str) -> str:
    w = _WAIT_WORDS[lang]
    times = course_times()
    svg = _timeline([("", list(zip(times, COURSE_TRIES)))], times[-1] + 1, lang, w["axis"], w["wait"], 0)
    return yours.figure(w["title"], svg, alt=w["title"])


# --- exercise 1: which errors are worth another try --------------------------------------------------------------------

_START_RETRY = '''def should_retry(status):
    return False
'''
SOLUTION_RETRY = '''def should_retry(status):
    return status == 429 or status >= 500
'''
RETRY_CASES = [(429,), (400,), (401,), (404,), (500,), (503,), (403,), (502,), (504,)]


def retry_picture(seen) -> str:
    """Each status code with what the learner's should_retry says, and whether that is right."""
    run = seen.mine
    fn = run.names.get("should_retry") if run is not None else None
    if not callable(fn):
        return ""
    right = run_code(SOLUTION_RETRY).names["should_retry"]
    w = {"en": ("What your should_retry says", "try again", "stop", "error"), "nl": ("Wat je should_retry zegt", "opnieuw", "stoppen", "fout")}[seen.lang]
    cells = []
    for (status,) in sorted(RETRY_CASES):
        try:
            said = bool(fn(status))
            word = w[1] if said else w[2]
            ok = said == right(status)
        except Exception:   # the learner's code
            word, ok = w[3], False
        colour = "var(--ea-ink)" if ok else "var(--ea-loss-text)"
        mark = "" if ok else " ✗"
        cells.append(f'<div style="padding:0.3rem 0.5rem;border:1px solid var(--ea-line);border-radius:6px;min-width:5.5rem">'
                     f'<div style="font-family:var(--ea-font-code);font-size:13px;color:var(--ea-ink)">{status}</div>'
                     f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:{colour}">{html.escape(word + mark)}</div></div>')
    body = f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem">{"".join(cells)}</div>'
    return yours.figure(w[0], body, alt=w[0])


# --- exercise 2: try again, with exponential backoff --------------------------------------------------------------------

class NoRetry(Exception):
    """The learner's should_retry is not right yet."""


def no_retry(status):
    raise NoRetry()


def learner_retry(text):
    """The learner's should_retry from exercise 1, when it is right; else None."""
    if text is MISSING or text is None or EXERCISES["should_retry"].check(text).state != "correct":
        return None
    return run_code(text).names.get("should_retry")


def again_inputs(retry_text) -> dict:
    """What exercise 2 runs with: the learner's should_retry from exercise 1, or one that says to finish it first."""
    return {"should_retry": learner_retry(retry_text) or no_retry}


_START_AGAIN = '''import json
import time
import urllib.error
import urllib.request


def ask_again(url, question, tries, base):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    request = urllib.request.Request(url + "/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.loads(response.read())
            return answer["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            ...
'''
SOLUTION_AGAIN = _START_AGAIN.replace('''            ...
''', '''            if not should_retry(e.code) or attempt == tries - 1:
                raise
            time.sleep(base * 2 ** attempt)
''')
BASE = 0.1   # the first wait in the checks, in seconds: small, so the check is quick, and large beside the time a request takes


def _posts(running) -> list:
    return [e for e in running.log if e["method"] == "POST"]


def _judge_busy(got, running):
    posts = _posts(running)
    if got != ANSWER:
        return "other"
    gaps = [b["time"] - a["time"] for a, b in zip(posts, posts[1:])]
    if len(gaps) == 2 and gaps[0] < 0.8 * BASE:
        return "no-wait"
    if len(gaps) == 2 and gaps[1] < 1.6 * BASE:
        return "no-doubling"
    return None


def _judge_key(got, running):
    if len(_posts(running)) > 1:
        return "retried-401"
    if not isinstance(got, urllib.error.HTTPError):
        return "swallowed"
    return None


def _judge_down(got, running):
    if not isinstance(got, urllib.error.HTTPError):
        return "gave-up-quietly"
    if len(_posts(running)) != 3:
        return "tries"
    return None


AGAIN_CASES = [
    Case(lambda: serve.StandIn({QUESTION: ANSWER}, fail=[429, 503]), (URL, QUESTION, 4, BASE), _judge_busy),
    Case(lambda: serve.StandIn({QUESTION: ANSWER}, fail=[401]), (URL, QUESTION, 4, BASE), _judge_key, expect=urllib.error.HTTPError),
    Case(lambda: serve.StandIn({QUESTION: ANSWER}, fail=[503, 503, 503]), (URL, QUESTION, 3, BASE), _judge_down, expect=urllib.error.HTTPError),
]


def tries_picture(seen) -> str:
    """The learner's ask_again against each stand-in of the check, on one time line: every try as the stand-in saw it,
    with what it answered."""
    run = seen.run
    fn = run.names.get("ask_again") if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    w = {"en": {"title": "Your tries, as the stand-in saw them", "rows": ["429, 503, then OK", "401: wrong key", "503, 503, 503"],
                "wait": "{s} s", "axis": "seconds after the first try"},
         "nl": {"title": "Je pogingen, zoals de stand-in ze zag", "rows": ["429, 503, dan goed", "401: verkeerde key", "503, 503, 503"],
                "wait": "{s} s", "axis": "seconden na de eerste poging"}}[seen.lang]
    rows, end = [], 0.0
    for label, case in zip(w["rows"], AGAIN_CASES):
        _got, _error, running = apicheck.call(fn, case)
        posts = _posts(running)[:8]
        if posts:
            t0 = posts[0]["time"]
            tries = [(p["time"] - t0, p["status"] or 0) for p in posts]
            end = max(end, tries[-1][0])
            rows.append((label, tries))
        else:
            rows.append((label, []))
    end = max(end * 1.1, 4 * BASE)
    return yours.figure(w["title"], _timeline(rows, end, seen.lang, w["axis"], w["wait"], 2), alt=w["title"])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "should_retry": FunctionBox(
        name="should_retry",
        picture=retry_picture,
        function="should_retry",
        start=_START_RETRY,
        solution=SOLUTION_RETRY,
        visible=[(429,)],
        hidden=RETRY_CASES[1:],
        mistakes=[
            Mistake("only-429", wrong="def should_retry(status):\n    return status == 429\n",
                    en="`{call}` gives `{got}`. A status from 500 up is trouble in the server itself, such as `503`, busy: that can be over in a moment, so it is worth another try too.",
                    nl="`{call}` geeft `{got}`. Een status vanaf 500 is een probleem in de server zelf, zoals `503`, bezet: dat kan zo over zijn, dus ook dan is nog een poging de moeite waard."),
            Mistake("all-4xx", wrong="def should_retry(status):\n    return status >= 400\n",
                    en="`{call}` gives `{got}`. A status in the 400s, other than 429, says your request itself is wrong: a bad body, no valid key. The same request gets the same answer, however often you send it.",
                    nl="`{call}` geeft `{got}`. Een status in de 400-reeks, behalve 429, zegt dat je request zelf niet klopt: een foute body, geen geldige key. Hetzelfde request krijgt hetzelfde antwoord, hoe vaak je het ook stuurt."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`should_retry` says no to everything yet. Which statuses can be different a moment later?",
                      "nl": "`should_retry` zegt nog nee tegen alles. Welke statussen kunnen een moment later anders zijn?"},
            "correct": {"en": "Right: your `should_retry` says yes to 429 and to every status from 500 up, and no to the other errors in the 400s, for all {n} codes the check tried.",
                        "nl": "Goed: je `should_retry` zegt ja tegen 429 en tegen elke status vanaf 500, en nee tegen de andere fouten in de 400-reeks, voor alle {n} codes die de check probeerde."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. 429 means too many requests: in a moment, it can be fine.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. 429 betekent te veel requests: over een moment kan het goed gaan."},
            "almost": {"en": "Right for 429. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor 429. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
    "ask_again": ApiBox(
        name="ask_again",
        picture=tries_picture,
        function="ask_again",
        start=_START_AGAIN,
        solution=SOLUTION_AGAIN,
        cases=AGAIN_CASES,
        names={"should_retry": run_code(SOLUTION_RETRY).names["should_retry"]},
        mistakes=[
            Mistake("no-retry", error=lambda e: isinstance(e, NoRetry),
                    en="This exercise uses your `should_retry` from exercise 1. Finish exercise 1 first; this check then runs again by itself.",
                    nl="Deze oefening gebruikt je `should_retry` uit oefening 1. Maak eerst oefening 1 af; deze check draait daarna vanzelf opnieuw."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "Under `except`, nothing happens yet: an error is swallowed, and the loop tries again at once. Decide there: pass the error on, or wait and try again.",
                      "nl": "Onder `except` gebeurt nog niets: een fout verdwijnt, en de loop probeert het meteen opnieuw. Beslis daar: geef de fout door, of wacht en probeer opnieuw."},
            "correct": {"en": "Right: your `ask_again` waits longer after every failed try, passes on an error that will not change, and gives up after the last try, against all {n} stand-ins.",
                        "nl": "Goed: je `ask_again` wacht na elke mislukte poging langer, geeft een fout die niet verandert meteen door, en geeft het op na de laatste poging, tegen alle {n} stand-ins."},
            "slow": {"en": "With `{call}`, your function is still not done after {s:g} seconds. Does it wait `base * 2 ** attempt` seconds, with the `base` it gets? The check gives a small one: 0.1 seconds.",
                     "nl": "Met `{call}` is je functie na {s:g} seconden nog niet klaar. Wacht hij `base * 2 ** attempt` seconden, met de `base` die hij krijgt? De check geeft een kleine: 0,1 seconde."},
            "http-429": {"en": "With `{call}`, the stand-in said 429 to the first try, and your function passed it on at once. 429 is worth another try: ask `should_retry(e.code)` before you pass the error on.",
                         "nl": "Met `{call}` zei de stand-in 429 tegen de eerste poging, en je functie gaf dat meteen door. 429 is nog een poging waard: vraag `should_retry(e.code)` voordat je de fout doorgeeft."},
            "http-503": {"en": "With `{call}`, the stand-in said 429 and then 503, and your function passed the 503 on, with tries left. Try again for every status `should_retry` says yes to, until the last try.",
                         "nl": "Met `{call}` zei de stand-in 429 en daarna 503, en je functie gaf de 503 door terwijl er nog pogingen over waren. Probeer opnieuw bij elke status waar `should_retry` ja op zegt, tot de laatste poging."},
            "other": {"en": "With `{call}`, the stand-in said 429, then 503, then answered, and your function gave `{got}`. After a try that worked, return the text of the answer.",
                      "nl": "Met `{call}` zei de stand-in 429, toen 503, en antwoordde toen, en je functie gaf `{got}`. Return na een poging die lukte de tekst van het antwoord."},
            "no-wait": {"en": "With `{call}`, your function tried again almost at once after the 429. Wait first: `time.sleep(base * 2 ** attempt)`.",
                        "nl": "Met `{call}` probeerde je functie het na de 429 bijna meteen opnieuw. Wacht eerst: `time.sleep(base * 2 ** attempt)`."},
            "no-doubling": {"en": "With `{call}`, your function waited, but the second wait was not longer than the first. With `base * 2 ** attempt`, every wait is twice the one before.",
                            "nl": "Met `{call}` wachtte je functie wel, maar de tweede keer niet langer dan de eerste. Met `base * 2 ** attempt` is elke wachttijd twee keer de vorige."},
            "swallowed": {"en": "With `{call}`, the stand-in said 401, no valid key, and your function gave `{got}`, as if nothing was wrong. Pass such an error on with `raise`, so the code that called it hears what happened.",
                          "nl": "Met `{call}` zei de stand-in 401, geen geldige key, en je functie gaf `{got}`, alsof er niets aan de hand was. Geef zo'n fout door met `raise`, zodat de code die hem aanriep hoort wat er gebeurde."},
            "retried-401": {"en": "With `{call}`, the stand-in said 401, no valid key, and your function tried again. A wrong key stays wrong: pass the error on at once, without waiting.",
                            "nl": "Met `{call}` zei de stand-in 401, geen geldige key, en je functie probeerde het opnieuw. Een verkeerde key blijft verkeerd: geef de fout meteen door, zonder te wachten."},
            "gave-up-quietly": {"en": "With `{call}`, all 3 tries got 503, and your function gave `{got}`, as if nothing was wrong. After the last try, pass the error on with `raise`.",
                                "nl": "Met `{call}` kregen alle 3 pogingen 503, en je functie gaf `{got}`, alsof er niets aan de hand was. Geef na de laatste poging de fout door met `raise`."},
            "tries": {"en": "With `{call}`, the stand-in said 503 every time, and your function did not send exactly 3 tries. With `tries` 3, try 3 times, and then pass the error on.",
                      "nl": "Met `{call}` zei de stand-in elke keer 503, en je functie stuurde niet precies 3 pogingen. Met `tries` 3 probeer je het 3 keer, en geef je daarna de fout door."},
        },
        time_limit=2.0,
    ),
}

REQUIRED = ["should_retry", "ask_again", "crowd"]

CASES = [
    ("should_retry", _START_RETRY, "empty", None),
    ("should_retry", SOLUTION_RETRY, "correct", None),
    ("should_retry", "def should_retry(status):\n    return status == 429\n", "wrong", "only-429"),
    ("should_retry", "def should_retry(status):\n    return status >= 400\n", "wrong", "all-4xx"),
    ("should_retry", "def should_retry(status):\n    return True\n", "wrong", "all-4xx"),
    ("should_retry", "def should_retry(status):\n    return status >= 500\n", "wrong", None),
    ("ask_again", _START_AGAIN, "empty", None),
    ("ask_again", SOLUTION_AGAIN, "correct", None),
    ("ask_again", SOLUTION_AGAIN.replace('if not should_retry(e.code) or attempt == tries - 1:',
                                         'if e.code != 429 or attempt == tries - 1:'), "error", "http-503"),
    ("ask_again", SOLUTION_AGAIN.replace('return answer["choices"][0]["message"]["content"]',
                                         'return answer'), "wrong", "other"),
    ("ask_again", SOLUTION_AGAIN.replace('                raise\n',
                                         '                return None\n'), "wrong", "swallowed"),
    ("ask_again", SOLUTION_AGAIN.replace('for attempt in range(tries):',
                                         'for attempt in range(1, tries):'), "wrong", "tries"),
    ("ask_again", _START_AGAIN.replace("            ...\n", "            raise\n"), "error", "http-429"),
    ("ask_again", SOLUTION_AGAIN.replace("            time.sleep(base * 2 ** attempt)\n", "            continue\n"), "wrong", "no-wait"),
    ("ask_again", SOLUTION_AGAIN.replace("base * 2 ** attempt", "base"), "wrong", "no-doubling"),
    ("ask_again", SOLUTION_AGAIN.replace("not should_retry(e.code) or ", ""), "wrong", "retried-401"),
    ("ask_again", SOLUTION_AGAIN.replace(" or attempt == tries - 1", ""), "wrong", "gave-up-quietly"),
    ("ask_again", SOLUTION_AGAIN.replace("base * 2 ** attempt", "2 ** attempt"), "slow", None),
]
