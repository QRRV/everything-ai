# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-secrets-and-endpoints: the made-up repository of the help assistant of the bike shop
Spoke & Chain, whose history once held two keys of a made-up payment service, PedalPay (COMMITS); a scanner that cuts
every file into words and asks of each word whether it has the form of a key; the addresses a server can listen on;
and the check of a token in the header Authorization, by a server whose token can be missing. The keys and tokens are
made up and readable, so that nobody mistakes them for real ones. The worked example starts the course's own server
on 127.0.0.1, with a token, and sends it three requests."""
import copy
import html
import re
import urllib.error
import urllib.request

from everything_ai import serve, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the keys, the tokens and the shop's repository --------------------------------------------------------------------

PREFIX = "pp_live_"   # every key of the made-up payment service PedalPay starts so, followed by 24 letters and digits
REST = 24
PAY_KEY = PREFIX + "trail500brakes2026saddle"
LOG_KEY = PREFIX + "gravel29wheels4417chain0"
PLACEHOLDER = PREFIX + "your_key_here"
TOKEN = "shop-token-4417-chain"   # the token the shop's own server asks for
OTHER = "spare-token-29-gravel"

COMMITS = [   # the history of the shop's repository, oldest first: each commit with the files it changed (None: removed)
    {"id": "a1c3e07", "message": "Start the help assistant",
     "changes": {"assistant.py": 'MODEL_URL = "http://127.0.0.1:8000/v1"\n'
                                 'SYSTEM = "You are the help assistant of the bike shop Spoke & Chain."\n'}},
    {"id": "b7d2f41", "message": "Add refunds through PedalPay",
     "changes": {"refunds.py": 'PEDALPAY_URL = "https://api.pedalpay.example/v1/refunds"\n'
                               f'PEDALPAY_KEY = "{PAY_KEY}"\n'}},
    {"id": "c90e5a2", "message": "Read the PedalPay key from the environment",
     "changes": {"refunds.py": 'import os\n\nPEDALPAY_URL = "https://api.pedalpay.example/v1/refunds"\n'
                               'PEDALPAY_KEY = os.environ["PEDALPAY_KEY"]\n'}},
    {"id": "d4f81b6", "message": "Explain the setup",
     "changes": {"README.md": f"Set PEDALPAY_KEY before you start, such as: export PEDALPAY_KEY={PLACEHOLDER}\n"}},
    {"id": "e2a6c93", "message": "Notes on a refund that failed",
     "changes": {"notes.txt": "POST https://api.pedalpay.example/v1/refunds\n"
                              f"Authorization: Bearer {LOG_KEY}\n"
                              "answer: 402\n"}},
    {"id": "f5b07d8", "message": "Remove the notes", "changes": {"notes.txt": None}},
    {"id": "0a9e4c1", "message": "Let the shop's laptops use the assistant",
     "changes": {"server.py": 'server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)\n'}},
    {"id": "1c7f2e5", "message": "Listen on 127.0.0.1 again",
     "changes": {"server.py": 'server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)\n'}},
]
SIGNS = "=\"'(),:"   # the scanner cuts a file into words at spaces and at these signs


def words(text: str) -> list:
    """The words of a text as the course's scanner cuts it: at spaces, new lines and each of SIGNS."""
    for sign in SIGNS:
        text = text.replace(sign, " ")
    return text.split()


def looks_like_key(word):
    # True when word has the form of a PedalPay key: pp_live_, then 24 letters and digits
    if not word.startswith("pp_live_"):
        return False
    rest = word[8:]
    return len(rest) == 24 and rest.isalnum()


def scan(text: str, fn=looks_like_key) -> list:
    """The words of text that fn says have the form of a key, in order."""
    return [w for w in words(text) if fn(w) is True]


def latest() -> dict:
    """The files of the repository as they are after the last commit."""
    files = {}
    for c in COMMITS:
        for path, text in c["changes"].items():
            if text is None:
                files.pop(path, None)
            else:
                files[path] = text
    return files


def history_scan(fn=looks_like_key) -> dict:
    """What a scan with fn finds: in each commit, the keys in the files it changed, as (path, key); in the latest
    version of the files; and every key in the whole history, once each, in the order they came."""
    commits = [[(path, k) for path, text in c["changes"].items() if text is not None for k in scan(text, fn)]
               for c in COMMITS]
    now = [(path, k) for path, text in latest().items() for k in scan(text, fn)]
    seen = []
    for found in commits:
        for _, k in found:
            if k not in seen:
                seen.append(k)
    return {"commits": commits, "latest": now, "history": seen}


# --- addresses and tokens ------------------------------------------------------------------------------------------------

def local_only(host):
    # True when a server that listens on host can only be reached from this computer
    if host == "localhost" or host == "::1":
        return True
    return host.startswith("127.")


def authorized(header, key):
    # True when the server has a key, and header is "Bearer " followed by that key
    if key == "":
        return False
    return header == "Bearer " + key


HOSTS = ["127.0.0.1", "localhost", "::1", "127.0.0.2", "0.0.0.0", "", "192.168.1.20", "10.0.0.127"]
_REACH = {"127.0.0.1": "local", "localhost": "local", "::1": "local", "127.0.0.2": "local", "0.0.0.0": "all",
          "": "same", "192.168.1.20": "network", "10.0.0.127": "network"}   # who can reach a server on each address

REQUESTS = [   # (header, the server's key) for the picture of authorized: with the key set, and with it missing
    ("", TOKEN), ("Bearer wrong-token", TOKEN), ("Bearer " + TOKEN, TOKEN), ("Bearer " + TOKEN + "0", TOKEN),
    (TOKEN, TOKEN), ("Bearer " + OTHER, OTHER), ("", ""), ("Bearer ", ""), ("Bearer anything", ""),
]


# --- the worked example: the course's server, with a token -----------------------------------------------------------

class Server:
    """The course's server (everything_ai.serve) on 127.0.0.1, on a free port, that asks for a token."""

    def __init__(self, token: str):
        if not isinstance(token, str) or token == "":
            raise ValueError("start_server needs a token: a server without one would let in a header with nothing after Bearer")
        self._running = serve.start(serve.StandIn(key=token))
        self.url = self._running.url
        self.host = self._running.httpd.server_address[0]

    def stop(self) -> None:
        self._running.stop()


def start_server(token: str) -> Server:
    """Starts the course's server on 127.0.0.1 with the token it asks for."""
    return Server(token)


def status(url: str, header) -> int:
    """The status code of the server's answer to a request for its models, with header as the header Authorization
    (None: no header)."""
    request = urllib.request.Request(url + "/models")
    if header is not None:
        request.add_header("Authorization", header)
    try:
        with urllib.request.urlopen(request, timeout=10) as answer:
            return answer.status
    except urllib.error.HTTPError as e:
        return e.code


def worked_example() -> list:
    """The lines the worked example prints, for the lesson's test."""
    server = start_server(TOKEN)
    try:
        lines = [f"listens on {server.host}"]
        for header in [None, "Bearer wrong-token", "Bearer " + TOKEN]:
            lines.append(f"{header} -> {status(server.url, header)}")
        return lines
    finally:
        server.stop()


# --- the pictures ----------------------------------------------------------------------------------------------------
# Every word of the learner is shown through html.escape.

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"
_LINE = ("font-family:var(--ea-font-code);font-size:12.5px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;"
         "background:var(--ea-inset);border:1px solid var(--ea-line);border-radius:4px;padding:0.3rem 0.5rem;margin:0.15rem 0")


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _mono(text: str) -> str:
    return f'<span style="{_MONO}">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_SURFACE_WORDS = {
    "en": {"title": "The attack surface of the help assistant: four points",
           "head": ["point", "what an attacker can try there"],
           "rows": [("the repository on GitHub", "read a key in the code or in its history"),
                    ("the PedalPay key", "make refunds in the shop's name"),
                    ("the model server on port 8000", "send it requests, when it listens beyond this computer"),
                    ("the help page", "put text in front of the model, as in the lessons before")],
           "alt": "The attack surface of the help assistant: {rows}."},
    "nl": {"title": "Het aanvalsoppervlak van de hulpassistent: vier punten",
           "head": ["punt", "wat een aanvaller daar kan proberen"],
           "rows": [("de repository op GitHub", "een key lezen in de code of in de geschiedenis"),
                    ("de key van PedalPay", "terugbetalingen doen namens de winkel"),
                    ("de modelserver op poort 8000", "er requests naar sturen, als hij buiten deze computer luistert"),
                    ("de hulppagina", "tekst voor het model zetten, zoals in de vorige lessen")],
           "alt": "Het aanvalsoppervlak van de hulpassistent: {rows}."},
}


def surface_picture(lang: str) -> str:
    """The four points of the attack surface of the help assistant, with what an attacker can try at each."""
    w = _SURFACE_WORDS[lang]
    rows = [[(html.escape(p), "font-weight:600"), (html.escape(t), "")] for p, t in w["rows"]]
    return yours.figure(w["title"], _table(rows, w["head"]),
                        alt=w["alt"].format(rows="; ".join(f"{p}: {t}" for p, t in w["rows"])))


_HISTORY_WORDS = {
    "en": {"title": "The shop's repository, commit by commit: what the scanner finds",
           "head": ["commit", "message", "changed", "keys found"], "removed": "{path} removed",
           "none": "none",
           "after": "In the latest version of the files, the scanner finds {n}. In the whole history it finds {h}: "
                    "{where}. Both commits are still in the repository, and git shows them to anyone who can read it.",
           "where": "{key} in {path}, commit {id}", "keys": ("{n} key", "{n} keys", "no keys"),
           "alt": "The commits of the shop's repository with the keys a scanner finds in each: {rows}. {after}"},
    "nl": {"title": "De repository van de winkel, commit voor commit: wat de scanner vindt",
           "head": ["commit", "bericht", "veranderd", "gevonden keys"], "removed": "{path} verwijderd",
           "none": "geen",
           "after": "In de nieuwste versie van de bestanden vindt de scanner er {n}. In de hele geschiedenis vindt "
                    "hij er {h}: {where}. Beide commits staan nog in de repository, en git laat ze zien aan iedereen "
                    "die hem kan lezen.",
           "where": "{key} in {path}, commit {id}", "keys": ("{n}", "{n}", "geen"),
           "alt": "De commits van de repository van de winkel met de keys die een scanner in elk vindt: {rows}. "
                  "{after}"},
}


def _many(n: int, forms: tuple) -> str:
    """The words for n things, from forms: (for one, for any other number, and if given for none), each with {n}
    where the number goes."""
    if n == 0 and len(forms) > 2:
        return forms[2]
    return (forms[0] if n == 1 else forms[1]).format(n=n)


def history_picture(lang: str) -> str:
    """The commits of the shop's repository, with the keys the course's scanner finds in each, and what is left in
    the latest version."""
    w = _HISTORY_WORDS[lang]
    found = history_scan()
    rows, said = [], []
    for c, keys in zip(COMMITS, found["commits"]):
        changed = ", ".join(w["removed"].format(path=p) if t is None else p for p, t in c["changes"].items())
        shown = "<br>".join(_mono(k) for _, k in keys) or html.escape(w["none"])
        rows.append([(_mono(c["id"]), ""), (html.escape(c["message"]), ""), (html.escape(changed), ""),
                     (shown, _OFF if keys else "")])
        said.append(f"{c['id']} {c['message']} ({changed}): " + (", ".join(k for _, k in keys) or w["none"]))
    where = "; ".join(w["where"].format(key=k, path=p, id=c["id"])
                      for c, keys in zip(COMMITS, found["commits"]) for p, k in keys)
    after = w["after"].format(n=_many(len(found["latest"]), w["keys"]), h=_many(len(found["history"]), w["keys"]),
                              where=where)
    body = _table(rows, w["head"]) + _note(after)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), after=after))


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


WORDS = [PAY_KEY, LOG_KEY, PLACEHOLDER, PREFIX + "trail500brakes2026saddl", PREFIX + "trail500brakes2026saddles",
         PREFIX + "trail500-brakes2026saddl", PREFIX + "gravel29wheels44", "PEDALPAY_KEY", "pp_test_trail500brakes2026saddle",
         ""]

_KEY_WORDS = {
    "en": {"title": "Your looks_like_key on tricky words, and on the shop's history",
           "head": ["word", "characters after pp_live_", "your looks_like_key", "the course's"], "error": "error",
           "no_prefix": "no pp_live_ at the start", "empty": "(empty text)",
           "rest": "{n}, {kind}", "alnum": "letters and digits", "other": "with other signs",
           "fine": "The same as the course's looks_like_key on all {k} words.",
           "differs": "It differs from the course's on {k} of the {m} words: they have a dashed frame.",
           "history": "On the shop's history, your looks_like_key finds {h} in the commits, and {n} in the latest "
                      "version.",
           "keys": ("{n} key", "{n} keys", "no keys"),
           "alt": "Tricky words with what your looks_like_key and the course's give: {rows}. {history}"},
    "nl": {"title": "Jouw looks_like_key op lastige woorden, en op de geschiedenis van de winkel",
           "head": ["woord", "tekens na pp_live_", "jouw looks_like_key", "die van de course"], "error": "fout",
           "no_prefix": "geen pp_live_ aan het begin", "empty": "(lege tekst)",
           "rest": "{n}, {kind}", "alnum": "letters en cijfers", "other": "met andere tekens",
           "fine": "Hetzelfde als looks_like_key van de course op alle {k} woorden.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} woorden: die hebben een gestreept kader.",
           "history": "Op de geschiedenis van de winkel vindt jouw looks_like_key {h} in de commits, en {n} in de "
                      "nieuwste versie.",
           "keys": ("{n} key", "{n} keys", "geen keys"),
           "alt": "Lastige woorden met wat jouw looks_like_key en die van de course geven: {rows}. {history}"},
}


def keys_picture(seen) -> str:
    """The learner's looks_like_key on tricky words beside the course's, and what it finds in the shop's history."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _KEY_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for word in WORDS:
        g = _call(fn, word)
        want = looks_like_key(word)
        shown = w["error"] if g is _STOPPED else repr(g)
        differs = g is _STOPPED or g is not want
        k += differs
        rest = word[len(PREFIX):]
        about = (w["rest"].format(n=len(rest), kind=w["alnum"] if rest.isalnum() else w["other"])
                 if word.startswith(PREFIX) else w["no_prefix"])
        name = word if word else w["empty"]
        rows.append([(_mono(name), ""), (html.escape(about), ""), (html.escape(shown), _OFF if differs else ""),
                     (repr(want), "")])
        said.append(f"{name} ({about}) {shown}, {want}")
    found = history_scan(lambda word: _call(fn, word))
    history = w["history"].format(h=_many(len(found["history"]), w["keys"]), n=_many(len(found["latest"]), w["keys"]))
    body = _table(rows, w["head"])
    body += _note(w["differs"].format(k=k, m=len(WORDS)) if k else w["fine"].format(k=len(WORDS)))
    body += _note(history)
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said), history=history))


_LOCAL_WORDS = {
    "en": {"title": "Your local_only on the addresses a server can listen on",
           "head": ["address", "who can reach the server", "your local_only", "the course's"], "error": "error",
           "empty": "(empty text)",
           "local": "programs on this computer only", "all": "every address of this computer, so also the network",
           "same": "the same as 0.0.0.0", "network": "every computer that can reach {host}",
           "fine": "The same as the course's local_only on all {k} addresses.",
           "differs": "It differs from the course's on {k} of the {m} addresses: they have a dashed frame.",
           "alt": "Addresses with who can reach a server on them, and what your local_only and the course's give: "
                  "{rows}."},
    "nl": {"title": "Jouw local_only op de adressen waarop een server kan luisteren",
           "head": ["adres", "wie de server kan bereiken", "jouw local_only", "die van de course"], "error": "fout",
           "empty": "(lege tekst)",
           "local": "alleen programma's op deze computer", "all": "elk adres van deze computer, dus ook het netwerk",
           "same": "hetzelfde als 0.0.0.0", "network": "elke computer die {host} kan bereiken",
           "fine": "Hetzelfde als local_only van de course op alle {k} adressen.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} adressen: die hebben een gestreept "
                      "kader.",
           "alt": "Adressen met wie een server erop kan bereiken, en wat jouw local_only en die van de course geven: "
                  "{rows}."},
}


def local_picture(seen) -> str:
    """The learner's local_only on addresses a server can listen on, beside the course's, with who can reach it."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _LOCAL_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for host in HOSTS:
        g = _call(fn, host)
        want = local_only(host)
        shown = w["error"] if g is _STOPPED else repr(g)
        differs = g is _STOPPED or g is not want
        k += differs
        who = w[_REACH[host]].format(host=host)
        name = host if host else w["empty"]
        rows.append([(_mono(name), ""), (html.escape(who), ""), (html.escape(shown), _OFF if differs else ""),
                     (repr(want), "")])
        said.append(f"{name} ({who}) {shown}, {want}")
    body = _table(rows, w["head"])
    body += _note(w["differs"].format(k=k, m=len(HOSTS)) if k else w["fine"].format(k=len(HOSTS)))
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said)))


_AUTH_WORDS = {
    "en": {"title": "Your authorized as the check of the shop's server",
           "head": ["the server's key", "header Authorization", "your authorized", "the course's", "the server answers"],
           "missing": "\"\" (missing)", "no_header": "\"\" (none)", "error": "error",
           "fine": "The same as the course's authorized on all {k} requests: with a key, only the right header gets "
                   "in; without one, nothing does.",
           "differs": "It differs from the course's on {k} of the {m} requests: they have a dashed frame.",
           "alt": "Requests to the shop's server, with and without a key, with what your authorized and the course's "
                  "give and what the server answers: {rows}."},
    "nl": {"title": "Jouw authorized als de controle van de server van de winkel",
           "head": ["key van de server", "header Authorization", "jouw authorized", "die van de course",
                    "de server antwoordt"],
           "missing": "\"\" (ontbreekt)", "no_header": "\"\" (geen)", "error": "fout",
           "fine": "Hetzelfde als authorized van de course bij alle {k} requests: met een key komt alleen de goede "
                   "header binnen; zonder key niets.",
           "differs": "Het verschilt van die van de course bij {k} van de {m} requests: die hebben een gestreept "
                      "kader.",
           "alt": "Requests aan de server van de winkel, met en zonder key, met wat jouw authorized en die van de "
                  "course geven en wat de server antwoordt: {rows}."},
}


def authorized_picture(seen) -> str:
    """The learner's authorized on requests to the shop's server, with its key set and missing, beside the course's,
    and the status code the server would answer with the learner's check."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _AUTH_WORDS[seen.lang]
    rows, said, k = [], [], 0
    for header, key in REQUESTS:
        g = _call(fn, header, key)
        want = authorized(header, key)
        shown = w["error"] if g is _STOPPED else repr(g)
        differs = g is _STOPPED or g is not want
        k += differs
        answer = "200" if g is True else "401"
        key_shown = f'"{key}"' if key else w["missing"]   # in quotes, so that the space after Bearer shows
        header_shown = f'"{header}"' if header else w["no_header"]
        rows.append([(_mono(key_shown), ""), (_mono(header_shown), ""), (html.escape(shown), _OFF if differs else ""),
                     (repr(want), ""), (answer, "")])
        said.append(f"{key_shown}, {header_shown}: {shown}, {want}, {answer}")
    body = _table(rows, w["head"])
    body += _note(w["differs"].format(k=k, m=len(REQUESTS)) if k else w["fine"].format(k=len(REQUESTS)))
    return yours.figure(w["title"], body, alt=w["alt"].format(rows="; ".join(said)))


# --- the demo: a check before a push ---------------------------------------------------------------------------------

DEMO_TEXT = ('# settings of the refund helper\n'
             'PEDALPAY_URL = "https://api.pedalpay.example/v1/refunds"\n'
             f'PEDALPAY_KEY = "{PAY_KEY}"\n')


def mine(box_text, correct: bool, name: str):
    """The learner's own function `name` when its exercise is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    from everything_ai.check import run_code
    run_ = run_code(box_text, {})
    fn = run_.names.get(name) if run_.error is None else None
    return fn if callable(fn) else None


_PUSH_WORDS = {
    "en": {"title": "Before the push: the scanner on your commit",
           "blocked": "Push blocked: {keys} on {lines}. {advice}", "line": ("line {n}", "lines {n}"),
           "advice": ("Take the key out of the code, and revoke it if it was ever pushed.",
                      "Take the keys out of the code, and revoke them if they were ever pushed."),
           "keys": ("1 key", "{n} keys"), "and": " and ",
           "clear": "No word has the form of a key: the push goes through. The scanner knows only the forms it was "
                    "given.",
           "whose": {True: "With your looks_like_key.", False: "With the course's looks_like_key, until yours is "
                                                               "right."},
           "alt": "Your commit, line by line, with the words that have the form of a key framed. {verdict}"},
    "nl": {"title": "Voor de push: de scanner op je commit",
           "blocked": "Push tegengehouden: {keys} op {lines}. {advice}", "line": ("regel {n}", "regel {n}"),
           "advice": ("Haal de key uit de code, en trek hem in als hij ooit gepusht is.",
                      "Haal de keys uit de code, en trek ze in als ze ooit gepusht zijn."),
           "keys": ("1 key", "{n} keys"), "and": " en ",
           "clear": "Geen woord heeft de vorm van een key: de push gaat door. De scanner kent alleen de vormen die "
                    "hij heeft gekregen.",
           "whose": {True: "Met jouw looks_like_key.", False: "Met looks_like_key van de course, tot de jouwe klopt."},
           "alt": "Je commit, regel voor regel, met de woorden die de vorm van een key hebben in een kader. {verdict}"},
}


_CUT = re.compile("([\\s" + re.escape(SIGNS) + "]+)")   # a line cut as words() cuts it, keeping what is between


def _use(fn):
    """A looks_like_key for the demo: the learner's when it gives True or False, else the course's."""
    def use(word):
        if fn is not None:
            got = _call(fn, word)
            if got is True or got is False:
                return got
        return looks_like_key(word)
    return use


def push_picture(text, lang: str, fn=None) -> str:
    """The learner's commit, line by line, with each word that has the form of a key in a dashed frame, and whether
    a check before the push would let it through."""
    w = _PUSH_WORDS[lang]
    text = text if isinstance(text, str) else ""
    check = _use(fn)
    shown, hits, lines = [], 0, []
    for number, line in enumerate(text.split("\n"), start=1):
        found = 0
        parts = []
        for i, piece in enumerate(_CUT.split(line)):   # odd pieces are the spaces and signs between the words
            if i % 2 == 0 and piece and check(piece) is True:
                found += 1
                parts.append(f'<span style="{_OFF};padding:0 0.1rem">{html.escape(piece)}</span>')
            else:
                parts.append(html.escape(piece))
        shown.append("".join(parts) or "&nbsp;")
        if found:
            hits += found
            lines.append(str(number))
    body = f'<div style="{_LINE}">' + "<br>".join(shown) + "</div>"
    joined = w["and"].join(lines) if len(lines) < 3 else ", ".join(lines[:-1]) + w["and"] + lines[-1]
    where = (w["line"][0] if len(lines) == 1 else w["line"][1]).format(n=joined)
    verdict = (w["blocked"].format(keys=_many(hits, w["keys"]), lines=where, advice=_many(hits, w["advice"]))
               if hits else w["clear"])
    body += _note(verdict) + _note(w["whose"][fn is not None])
    return yours.figure(w["title"], body, alt=w["alt"].format(verdict=verdict))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["looks-like-key", "local-only", "authorized", "reach", "environment"]


class Named:
    """A text the check calls by the name the lesson gives it, such as TOKEN, so that a message stays short. The
    learner's code gets the text itself."""

    def __init__(self, value: str, name: str):
        self.value, self.name = value, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.value


KEY_N = Named(PAY_KEY, "KEY")
TOKEN_N = Named(TOKEN, "TOKEN")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(KEY|TOKEN|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `KEY` or `TOKEN`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `KEY` of `TOKEN`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")


def _show(v) -> str:
    """A value for a check message, as a code span."""
    return f"`{v!r}`"


# exercise 1: looks_like_key
_K_HEAD = ("def looks_like_key(word):\n"
           "    # True when word has the form of a PedalPay key: pp_live_, then 24 letters and digits\n")
_K_PREFIX = ("    if not word.startswith(\"pp_live_\"):\n"
             "        return False\n")
_K_REST = "    rest = word[8:]\n"
_K_RET = "    return len(rest) == 24 and rest.isalnum()\n"
_K_START = _K_HEAD + "    ...\n" + _K_REST + "    return ...\n"
_K = _K_HEAD + _K_PREFIX + _K_REST + _K_RET
_K_ONLY_PREFIX = _K_HEAD + "    return word.startswith(\"pp_live_\")\n"
_K_NO_LENGTH = _K.replace("len(rest) == 24 and rest.isalnum()", "rest.isalnum()")
_K_NO_ALNUM = _K.replace("len(rest) == 24 and rest.isalnum()", "len(rest) == 24")
_K_AT_LEAST = _K.replace("len(rest) == 24", "len(rest) >= 24")
_K_WHOLE = _K.replace("len(rest) == 24", "len(word) == 24")

# exercise 2: local_only
_L_HEAD = ("def local_only(host):\n"
           "    # True when a server that listens on host can only be reached from this computer\n"
           "    if host == \"localhost\" or host == \"::1\":\n"
           "        return True\n")
_L_START = _L_HEAD + "    return ...\n"
_L = _L_HEAD + "    return host.startswith(\"127.\")\n"
_L_EXACT = _L_HEAD + "    return host == \"127.0.0.1\"\n"
_L_CONTAINS = _L_HEAD + "    return \"127\" in host\n"
_L_ZEROS = _L_HEAD + "    return host.startswith(\"127.\") or host == \"0.0.0.0\"\n"
_L_EMPTY = _L_HEAD + "    return host.startswith(\"127.\") or host == \"\"\n"

# exercise 3: authorized
_A_HEAD = ("def authorized(header, key):\n"
           "    # True when the server has a key, and header is \"Bearer \" followed by that key\n")
_A_GUARD = ("    if key == \"\":\n"
            "        return False\n")
_A_RET = "    return header == \"Bearer \" + key\n"
_A_START = _A_HEAD + "    ...\n" + _A_RET
_A = _A_HEAD + _A_GUARD + _A_RET
_A_NO_GUARD = _A_HEAD + _A_RET
_A_OPEN = _A_HEAD + "    return key == \"\" or header == \"Bearer \" + key\n"
_A_CONTAINS = _A_HEAD + _A_GUARD + "    return key in header\n"
_A_NO_BEARER = _A_HEAD + _A_GUARD + "    return header == key\n"

EXERCISES = {
    "looks-like-key": FunctionBox(
        name="looks-like-key",
        picture=keys_picture,
        function="looks_like_key",
        start=_K_START,
        solution=_K,
        visible=[(KEY_N,)],
        hidden=[(PLACEHOLDER,), (LOG_KEY,), (PREFIX + "trail500brakes2026saddl",), (PREFIX + "gravel29wheels44",),
                (PREFIX + "trail500-brakes2026saddl",), (PREFIX + "trail500brakes2026saddles",), ("PEDALPAY_KEY",),
                ("pp_test_trail500brakes2026saddle",), ("",)],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("only-prefix", wrong=_K_ONLY_PREFIX,
                    en="`{call}` gives {got}, and it should give {want}: your `looks_like_key` looks only at the start, and the setup text of the README starts with `pp_live_` too. After `pp_live_` a key has 24 letters and digits.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `looks_like_key` kijkt alleen naar het begin, en de tekst in de README begint ook met `pp_live_`. Na `pp_live_` heeft een key 24 letters en cijfers."),
            Mistake("no-length", wrong=_K_NO_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: after `pp_live_`, this word has 23 letters and digits, and your `looks_like_key` does not count them. A PedalPay key has exactly 24.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na `pp_live_` heeft dit woord 23 letters en cijfers, en jouw `looks_like_key` telt ze niet. Een key van PedalPay heeft er precies 24."),
            Mistake("no-alnum", wrong=_K_NO_ALNUM,
                    en="`{call}` gives {got}, and it should give {want}: this word has 24 characters after `pp_live_`, but one of them is a `-`. Check also that the rest holds only letters and digits.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dit woord heeft 24 tekens na `pp_live_`, maar een ervan is een `-`. Controleer ook dat de rest alleen letters en cijfers bevat."),
            Mistake("at-least", wrong=_K_AT_LEAST,
                    en="`{call}` gives {got}, and it should give {want}: after `pp_live_`, this word has 25 letters and digits, and your `looks_like_key` lets any number from 24 up through. A PedalPay key has exactly 24.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: na `pp_live_` heeft dit woord 25 letters en cijfers, en jouw `looks_like_key` laat elk aantal vanaf 24 door. Een key van PedalPay heeft er precies 24."),
            Mistake("whole", wrong=_K_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}: your `looks_like_key` counts the whole word, `pp_live_` included, and a key is 32 characters long. Count the characters of `rest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `looks_like_key` telt het hele woord, `pp_live_` meegeteld, en een key is 32 tekens lang. Tel de tekens van `rest`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: first give back False when `word` does not start with `pp_live_`; then check the length of `rest` and that it holds only letters and digits.",
                      "nl": "Vervang de twee `...`: geef eerst False terug als `word` niet met `pp_live_` begint; controleer dan de lengte van `rest`, en dat die alleen letters en cijfers bevat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A word has the form of a key when it starts with `pp_live_` and the rest is 24 letters and digits.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een woord heeft de vorm van een key als het met `pp_live_` begint en de rest 24 letters en cijfers is."},
            "almost": {"en": "Right for `looks_like_key(KEY)`. But `{call}` gives {got}, and it should give {want}. Look at the start, the length of the rest and its characters, for every word.",
                       "nl": "Goed voor `looks_like_key(KEY)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar het begin, de lengte van de rest en de tekens erin, bij elk woord."},
            "correct": {"en": "Your `looks_like_key` is right on all {n} words. It finds both keys in the shop's history, and not the setup text of the README.",
                        "nl": "Je `looks_like_key` klopt bij alle {n} woorden. Hij vindt beide keys in de geschiedenis van de winkel, en de tekst in de README niet."},
        },
    ),
    "local-only": FunctionBox(
        name="local-only",
        picture=local_picture,
        function="local_only",
        start=_L_START,
        solution=_L,
        visible=[("127.0.0.1",)],
        hidden=[("localhost",), ("::1",), ("127.0.0.2",), ("0.0.0.0",), ("",), ("192.168.1.20",), ("10.0.0.127",)],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("exact", wrong=_L_EXACT,
                    en="`{call}` gives {got}, and it should give {want}: your `local_only` knows only `127.0.0.1`, and this address is the computer itself too. See *Open doors* in *The idea*.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `local_only` kent alleen `127.0.0.1`, en ook dit adres is de computer zelf. Zie *Open deuren* in *Het idee*."),
            Mistake("zeros", wrong=_L_ZEROS,
                    en="`{call}` gives {got}, and it should give {want}: a server that listens on `0.0.0.0` listens on every address of this computer, also the one the network reaches it on.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een server die op `0.0.0.0` luistert, luistert op elk adres van deze computer, ook op het adres waarop het netwerk hem bereikt."),
            Mistake("empty", wrong=_L_EMPTY,
                    en="`{call}` gives {got}, and it should give {want}: an empty text as the address is not nothing. Python then listens on every address of this computer, as with `0.0.0.0`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een lege tekst als adres is niet niets. Python luistert dan op elk adres van deze computer, net als bij `0.0.0.0`."),
            Mistake("contains", wrong=_L_CONTAINS,
                    en="`{call}` gives {got}, and it should give {want}: your `local_only` looks for 127 anywhere in the address, and this address ends in it. What counts is how the address starts.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `local_only` zoekt 127 ergens in het adres, en dit adres eindigt erop. Wat telt, is hoe het adres begint."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` after `return`: give back whether `host` is one of this computer's own addresses; *Open doors* in *The idea* says which those are.",
                      "nl": "Vervang de `...` na `return`: geef terug of `host` een van de eigen adressen van deze computer is; welke dat zijn, staat bij *Open deuren* in *Het idee*."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Only `localhost`, `::1` and the addresses that RFC 1122 gives to the computer itself keep a server on this computer.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Alleen `localhost`, `::1` en de adressen die RFC 1122 aan de computer zelf geeft, houden een server op deze computer."},
            "almost": {"en": "Right for `local_only(\"127.0.0.1\")`. But `{call}` gives {got}, and it should give {want}. Does your function look at the `host` it gets?",
                       "nl": "Goed voor `local_only(\"127.0.0.1\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijkt je functie naar de `host` die ze krijgt?"},
            "correct": {"en": "Your `local_only` is right on all {n} addresses. `0.0.0.0` and the empty text are not local: they are every address at once.",
                        "nl": "Je `local_only` klopt bij alle {n} adressen. `0.0.0.0` en de lege tekst zijn niet lokaal: dat zijn alle adressen tegelijk."},
        },
    ),
    "authorized": FunctionBox(
        name="authorized",
        picture=authorized_picture,
        function="authorized",
        start=_A_START,
        solution=_A,
        visible=[(Named("Bearer " + TOKEN, '"Bearer " + TOKEN'), TOKEN_N)],
        hidden=[("", TOKEN_N), ("Bearer wrong-token", TOKEN_N), (Named("Bearer " + TOKEN + "0", '"Bearer " + TOKEN + "0"'), TOKEN_N),
                ("Bearer " + OTHER, OTHER), ("Bearer ", ""), ("", ""),
                ("Bearer anything", ""), (TOKEN_N, TOKEN_N)],
        show=_show,
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("no-guard", wrong=_A_NO_GUARD,
                    en="`{call}` gives {got}, and it should give {want}: the server has no key here, as when `SHOP_TOKEN` is not set. Then `\"Bearer \" + key` is only `\"Bearer \"`, and a request with an empty token gets in. A server without a key must refuse every request.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de server heeft hier geen key, zoals wanneer `SHOP_TOKEN` niet gezet is. Dan is `\"Bearer \" + key` alleen `\"Bearer \"`, en komt een request met een leeg token binnen. Een server zonder key moet elk request weigeren."),
            Mistake("open", wrong=_A_OPEN,
                    en="`{call}` gives {got}, and it should give {want}: when the server has no key, your `authorized` lets every request in. Without a key it must let none in.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als de server geen key heeft, laat jouw `authorized` elk request binnen. Zonder key moet hij er geen enkele binnenlaten."),
            Mistake("contains", wrong=_A_CONTAINS,
                    en="`{call}` gives {got}, and it should give {want}: your `authorized` looks for the key anywhere in the header, and this header has a character after it. Compare the whole header.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw `authorized` zoekt de key ergens in de header, en in deze header staat er een teken achter. Vergelijk de hele header."),
            Mistake("no-bearer", wrong=_A_NO_BEARER,
                    en="`{call}` gives {got}, and it should give {want}: the header holds `Bearer `, with a space, before the key. Compare it with `\"Bearer \" + key`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: de header bevat `Bearer `, met een spatie, voor de key. Vergelijk hem met `\"Bearer \" + key`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: what must the function give back when the server has no token, before the last line compares the header?",
                      "nl": "Vervang de `...`: wat moet de functie teruggeven als de server geen token heeft, voordat de laatste regel de header vergelijkt?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. A request gets in only when the server has a key, and the header is exactly `Bearer `, a space and that key.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Een request komt alleen binnen als de server een key heeft, en de header precies `Bearer `, een spatie en die key is."},
            "almost": {"en": "Right for the visible call. But `{call}` gives {got}, and it should give {want}. Look at both the header and the key the function gets.",
                       "nl": "Goed voor de zichtbare aanroep. Maar `{call}` geeft {got}, en dat moet {want} zijn. Kijk naar zowel de header als de key die de functie krijgt."},
            "correct": {"en": "Your `authorized` is right on all {n} requests. With a key, only the right header gets in; without a key, nothing does.",
                        "nl": "Je `authorized` klopt bij alle {n} requests. Met een key komt alleen de goede header binnen; zonder key niets."},
        },
    ),
}

CASES = [
    ("looks-like-key", _K_START, "empty", None),
    ("looks-like-key", _K, "correct", None),
    ("looks-like-key", "def looks_like_key(word):\n    return word[:8] == \"pp_live_\" and len(word) == 32 and word[8:].isalnum()\n", "correct", None),
    ("looks-like-key", _K_ONLY_PREFIX, "wrong", "only-prefix"),
    ("looks-like-key", _K_NO_LENGTH, "wrong", "no-length"),
    ("looks-like-key", _K_NO_ALNUM, "wrong", "no-alnum"),
    ("looks-like-key", _K_AT_LEAST, "wrong", "at-least"),
    ("looks-like-key", _K_WHOLE, "wrong", "whole"),
    ("looks-like-key", "def looks_like_key(word):\n    return False\n", "wrong", None),
    ("looks-like-key", "def looks_like_key(word):\n    return True\n", "almost", None),
    ("looks-like-key", "def looks_like_key(word):\n    return word == KEY\n", "error", "lesson-names"),
    ("local-only", _L_START, "empty", None),
    ("local-only", _L, "correct", None),
    ("local-only", "def local_only(host):\n    return host in (\"localhost\", \"::1\") or host[:4] == \"127.\"\n", "correct", None),
    ("local-only", _L_EXACT, "wrong", "exact"),
    ("local-only", _L_ZEROS, "wrong", "zeros"),
    ("local-only", _L_EMPTY, "wrong", "empty"),
    ("local-only", _L_CONTAINS, "wrong", "contains"),
    ("local-only", "def local_only(host):\n    return False\n", "wrong", None),
    ("local-only", "def local_only(host):\n    return True\n", "almost", None),
    ("local-only", "def local_only(host):\n    return host == KEY\n", "error", "lesson-names"),
    ("authorized", _A_START, "empty", None),
    ("authorized", _A, "correct", None),
    ("authorized", "def authorized(header, key):\n    return key != \"\" and header == f\"Bearer {key}\"\n", "correct", None),
    ("authorized", _A_NO_GUARD, "wrong", "no-guard"),
    ("authorized", _A_OPEN, "wrong", "open"),
    ("authorized", _A_CONTAINS, "wrong", "contains"),
    ("authorized", _A_NO_BEARER, "wrong", "no-bearer"),
    ("authorized", "def authorized(header, key):\n    return False\n", "wrong", None),
    ("authorized", "def authorized(header, key):\n    return True\n", "almost", None),
    ("authorized", "def authorized(header, key):\n    return header == \"Bearer \" + TOKEN\n", "error", "lesson-names"),
]
