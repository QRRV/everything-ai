# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-batch-keys-costs: a call that takes its API key from an environment variable, against stand-ins
of a server that want a key (everything_ai.serve.StandIn with `key`); the bill of a list of calls from their usage;
the worked example, the lines of a batch file; and the pictures: the request with its key, and the learner's bill call
by call."""
import ast
import contextlib
import html
import io
import json

from everything_ai import apicheck, serve, yours
from everything_ai.apicheck import URL, ApiBox, Case
from everything_ai.check import FunctionBox, Mistake

ENV = "MODEL_API_KEY"
QUESTION = "Name three colors of the rainbow."
ANSWER = "Red, orange and yellow."
KEYS = ["sk-course-7f3a91", "sk-course-b2e604"]   # the keys of the two stand-ins; each call gets its own in ENV

# --- the worked example: the lines of a batch file ----------------------------------------------------------------------

WORKED = '''import json

questions = ["Name a color.", "Name a fruit."]
lines = []
for i in range(len(questions)):
    request = {"custom_id": f"q{i}", "method": "POST", "url": "/v1/chat/completions",
               "body": {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": questions[i]}], "max_tokens": 50}}
    lines.append(json.dumps(request))
for line in lines:
    print(line)
'''


def worked() -> str:
    """What the worked example prints: one line of JSON per request."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(WORKED, {})
    return out.getvalue()


# --- exercise 1: the key from an environment variable -------------------------------------------------------------------

_START_KEY = '''import json
import urllib.request


def ask(url, question):
    body = {"model": "qwen3-0.6b", "messages": [{"role": "user", "content": question}], "max_tokens": 50}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url + "/chat/completions", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())
    return answer["choices"][0]["message"]["content"]
'''
SOLUTION_KEY = _START_KEY.replace("import json\n", "import json\nimport os\n", 1).replace(
    '''    headers = {"Content-Type": "application/json"}''',
    '''    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MODEL_API_KEY"]}''')


def _judge_key(got, running):
    return None if got == ANSWER else "other"


KEY_CASES = [Case(lambda key=key: serve.StandIn({QUESTION: ANSWER}, key=key), (URL, QUESTION), _judge_key, env={ENV: key}) for key in KEYS]


def key_picture(seen) -> str:
    words = {"en": {"title": "What your ask sent, and what came back", "request": "Your code sent: {method} {path}",
                    "headers": "Headers:", "answer": "The server sent back: status {status}", "stream": "a stream, with {n} pieces",
                    "none": "Your function sent nothing to the server yet."},
             "nl": {"title": "Wat je ask stuurde, en wat er terugkwam", "request": "Je code stuurde: {method} {path}",
                    "headers": "Headers:", "answer": "De server stuurde terug: status {status}", "stream": "een stream, met {n} stukjes",
                    "none": "Je functie stuurde nog niets naar de server."}}[seen.lang]
    return apicheck.exchange(seen, "ask", KEY_CASES[0], words)


def _no_env(e) -> bool:
    """A KeyError from os.environ: the error's last frame is in Python's module os."""
    if not isinstance(e, KeyError):
        return False
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return tb is not None and tb.tb_frame.f_globals.get("__name__") == "os"


def _env_outside(text) -> bool:
    """The box reads os.environ (or os.getenv) when it runs, outside the body of a function: at the top, or in a
    default value of a parameter, which Python works out when it reads the `def`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    found = []

    class Look(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            for part in node.decorator_list + node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                self.visit(part)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            pass

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in ("environ", "getenv"):
                found.append(node)
            self.generic_visit(node)

        def visit_Name(self, node):
            if node.id in ("environ", "getenv"):
                found.append(node)

    Look().visit(tree)
    return bool(found)


# --- exercise 2: the bill -----------------------------------------------------------------------------------------------

_START_BILL = '''def bill(usages, price_in, price_out):
    total = 0
    return total
'''
SOLUTION_BILL = '''def bill(usages, price_in, price_out):
    total = 0
    for usage in usages:
        total = total + usage["prompt_tokens"] * price_in / 1_000_000
        total = total + usage["completion_tokens"] * price_out / 1_000_000
    return total
'''
DAY = [{"prompt_tokens": 1200, "completion_tokens": 300}, {"prompt_tokens": 800, "completion_tokens": 200}]
BILL_CASES = [
    (DAY, 3, 15),
    ([{"prompt_tokens": 19, "completion_tokens": 13}], 3, 15),
    ([], 3, 15),
    ([{"prompt_tokens": 5000, "completion_tokens": 0}, {"prompt_tokens": 0, "completion_tokens": 1000}, {"prompt_tokens": 2500, "completion_tokens": 500}], 0.5, 1.5),
    (DAY * 3, 1, 4),
]


def bill_picture(seen) -> str:
    """The learner's bill for the calls of the example, one call more each time: a bar per call with what it adds."""
    run = seen.mine
    fn = run.names.get("bill") if run is not None else None
    if not callable(fn):
        return ""
    w = {"en": ("Your bill, call by call", "call {i}: {tokens}", "in all: {total}", "tokens in, out", "error"),
         "nl": ("Je rekening, aanroep voor aanroep", "aanroep {i}: {tokens}", "samen: {total}", "tokens erin, eruit", "fout")}[seen.lang]
    usages, price_in, price_out = DAY, 3, 15
    rows, before = [], 0.0
    try:
        totals = [float(fn([dict(u) for u in usages[:k]], price_in, price_out)) for k in range(1, len(usages) + 1)]
    except Exception:   # the learner's code
        return yours.figure(w[0], f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-loss-text)">{w[4]}</div>', alt=w[0])
    top = max([abs(t) for t in totals] + [1e-12])
    for i, (u, t) in enumerate(zip(usages, totals), start=1):
        part = t - before
        before = t
        width = max(0.0, min(100.0, 100 * abs(part) / top))
        label = w[1].format(i=i, tokens=f'{u["prompt_tokens"]} + {u["completion_tokens"]}')
        money = f"${part:.4f}" if seen.lang == "en" else f"${part:.4f}".replace(".", ",")
        rows.append(f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0"><div style="width:9rem;font-family:var(--ea-font-ui);'
                    f'font-size:12px;color:var(--ea-ink)">{html.escape(label)}</div><div style="flex:1"><div style="width:{width:.1f}%;height:12px;'
                    f'background:var(--ea-ink-3);border-radius:2px"></div></div><div style="width:5.5rem;text-align:right;font-family:var(--ea-font-code);'
                    f'font-size:12px;color:var(--ea-ink)">{html.escape(money)}</div></div>')
    total = f"${totals[-1]:.4f}" if seen.lang == "en" else f"${totals[-1]:.4f}".replace(".", ",")
    rows.append(f'<div style="border-top:1px solid var(--ea-line);margin-top:0.3rem;padding-top:0.3rem;text-align:right;font-family:var(--ea-font-code);'
                f'font-size:12px;color:var(--ea-ink)">{html.escape(w[2].format(total=total))}</div>')
    return yours.figure(w[0], "".join(rows), alt=w[0])


_COMMON = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
                "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."},
}

EXERCISES = {
    "key": ApiBox(
        name="key",
        picture=key_picture,
        function="ask",
        start=_START_KEY,
        solution=SOLUTION_KEY,
        cases=KEY_CASES,
        first=[
            Mistake("env-outside", matches=lambda _, text: _env_outside(text),
                    en="Your box reads `os.environ` outside `ask`, as soon as the box runs. The check sets `MODEL_API_KEY` only while it calls `ask`. Read the key inside `ask`, just before you send the request.",
                    nl="Je vak leest `os.environ` buiten `ask`, zodra het vak draait. De check zet `MODEL_API_KEY` pas terwijl hij `ask` aanroept. Lees de key in `ask`, vlak voordat je het request stuurt."),
        ],
        mistakes=[
            Mistake("env-name", error=_no_env,
                    en="With `{call}`, `os.environ` has no such name. The key is in the environment variable `MODEL_API_KEY`, written exactly so.",
                    nl="Met `{call}` heeft `os.environ` die naam niet. De key staat in de environment variable `MODEL_API_KEY`, precies zo geschreven."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, a key you ask for is not in the answer. The text is in `[\"choices\"][0][\"message\"][\"content\"]`, as before.",
                    nl="Met `{call}` staat een key waar je om vraagt niet in het antwoord. De tekst staat in `[\"choices\"][0][\"message\"][\"content\"]`, zoals eerst."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`ask` sends the request without a key yet, and the server wants one. Add the header `Authorization`, with `Bearer `, a space and the key from `os.environ`.",
                      "nl": "`ask` stuurt het request nog zonder key, en de server wil er een. Voeg de header `Authorization` toe, met `Bearer `, een spatie en de key uit `os.environ`."},
            "correct": {"en": "Right: your `ask` takes the key from the environment, and sends it with the request. The check gave it {n} different keys, and both servers answered.",
                        "nl": "Goed: je `ask` haalt de key uit de omgeving, en stuurt hem mee met het request. De check gaf hem {n} verschillende keys, en beide servers antwoordden."},
            "http-401": {"en": "With `{call}`, the server answered 401: \"{message}\" The header must be called `Authorization`, and hold `Bearer `, a space, and the key from `os.environ[\"MODEL_API_KEY\"]`.",
                         "nl": "Met `{call}` antwoordde de server 401: \"{message}\" De header moet `Authorization` heten, en `Bearer ` bevatten, een spatie, en de key uit `os.environ[\"MODEL_API_KEY\"]`."},
            "other": {"en": "With `{call}`, your function gave `{got}`; the stand-in answered *Red, orange and yellow.* Give back the text of the answer, as before.",
                      "nl": "Met `{call}` gaf je functie `{got}`; de stand-in antwoordde *Red, orange and yellow.* Geef de tekst van het antwoord terug, zoals eerst."},
        },
    ),
    "bill": FunctionBox(
        name="bill",
        picture=bill_picture,
        function="bill",
        start=_START_BILL,
        solution=SOLUTION_BILL,
        visible=BILL_CASES[:1],
        hidden=BILL_CASES[1:],
        mistakes=[
            Mistake("per-million", wrong=SOLUTION_BILL.replace(" / 1_000_000", ""),
                    en="`{call}` gives `{got}`, a million times too much: the prices are per million tokens. Divide by `1_000_000`.",
                    nl="`{call}` geeft `{got}`, een miljoen keer te veel: de prijzen zijn per miljoen tokens. Deel door `1_000_000`."),
            Mistake("swapped", wrong=SOLUTION_BILL.replace("* price_in", "* PRICE").replace("* price_out", "* price_in").replace("* PRICE", "* price_out"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. The tokens that go in, `prompt_tokens`, cost `price_in`; the ones that come out, `completion_tokens`, cost `price_out`.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De tokens die erin gaan, `prompt_tokens`, kosten `price_in`; die eruit komen, `completion_tokens`, kosten `price_out`."),
            Mistake("one-price", wrong=SOLUTION_BILL.replace("* price_out", "* price_in"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. The tokens that come out cost `price_out`, usually more than the ones that go in.",
                    nl="`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. De tokens die eruit komen kosten `price_out`, meestal meer dan die erin gaan."),
            Mistake("in-only", wrong=SOLUTION_BILL.replace('        total = total + usage["completion_tokens"] * price_out / 1_000_000\n', ""),
                    en="`{call}` gives `{got}`: only the tokens that went in. The answer costs too: add `completion_tokens` at `price_out`.",
                    nl="`{call}` geeft `{got}`: alleen de tokens die erin gingen. Het antwoord kost ook geld: tel `completion_tokens` erbij, tegen `price_out`."),
            Mistake("last-only", wrong='''def bill(usages, price_in, price_out):
    total = 0
    for usage in usages:
        total = usage["prompt_tokens"] * price_in / 1_000_000 + usage["completion_tokens"] * price_out / 1_000_000
    return total
''',
                    en="`{call}` gives `{got}`: the cost of the last call only. Add each call to `total`, instead of putting it in its place.",
                    nl="`{call}` geeft `{got}`: alleen de kosten van de laatste aanroep. Tel elke aanroep op bij `total`, in plaats van hem te vervangen."),
        ],
        texts={
            **_COMMON,
            "empty": {"en": "`bill` gives 0 yet. Go through `usages`, and add the cost of each call to `total`.",
                      "nl": "`bill` geeft nog 0. Ga door `usages`, en tel de kosten van elke aanroep op bij `total`."},
            "correct": {"en": "Right: your `bill` adds up what goes in and what comes out, each at its own price per million tokens, for all {n} lists the check tried, the empty one too.",
                        "nl": "Goed: je `bill` telt op wat erin gaat en wat eruit komt, elk tegen zijn eigen prijs per miljoen tokens, voor alle {n} lijsten die de check probeerde, ook de lege."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Per call: `prompt_tokens` times `price_in`, plus `completion_tokens` times `price_out`, divided by a million.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Per aanroep: `prompt_tokens` keer `price_in`, plus `completion_tokens` keer `price_out`, gedeeld door een miljoen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`.",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn."},
        },
    ),
}

REQUIRED = ["key", "bill", "batch"]

CASES = [
    ("key", _START_KEY, "empty", None),
    ("key", SOLUTION_KEY, "correct", None),
    ("key", SOLUTION_KEY.replace('"Authorization": "Bearer " +', '"Authorization":'), "error", "http-401"),
    ("key", SOLUTION_KEY.replace('os.environ["MODEL_API_KEY"]', 'os.environ["API_KEY"]'), "error", "env-name"),
    ("key", SOLUTION_KEY.replace("\n\ndef ask", '\nKEY = os.environ["MODEL_API_KEY"]\n\n\ndef ask', 1).replace('os.environ["MODEL_API_KEY"]}', "KEY}"), "wrong", "env-outside"),
    ("key", SOLUTION_KEY.replace("def ask(url, question):", 'def ask(url, question, key=os.environ.get("MODEL_API_KEY")):'), "wrong", "env-outside"),
    ("key", SOLUTION_KEY.replace('answer["choices"][0]', 'answer["choice"][0]'), "error", "key"),
    ("key", SOLUTION_KEY.replace('    return answer["choices"][0]["message"]["content"]', '    return answer'), "wrong", "other"),
    ("bill", _START_BILL, "empty", None),
    ("bill", SOLUTION_BILL, "correct", None),
    ("bill", SOLUTION_BILL.replace(" / 1_000_000", ""), "wrong", "per-million"),
    ("bill", SOLUTION_BILL.replace("* price_out", "* price_in"), "wrong", "one-price"),
    ("bill", SOLUTION_BILL.replace('        total = total + usage["completion_tokens"] * price_out / 1_000_000\n', ""), "wrong", "in-only"),
]
