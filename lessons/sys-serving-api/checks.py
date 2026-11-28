# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-serving-api: the demand model of the project behind an API. The course trains the course's
model of the project as the lesson *Project: a demand model, tracked and packed* trains h16-lr0.01, and serves it with
Python's ThreadingHTTPServer on 127.0.0.1: POST /predict takes one hour as JSON and answers the bikes the model expects,
GET /health says that the server is up. The learner writes the line of `answer` that turns the model's tensor into a
number JSON can hold, and the check in `handle` that answers a bad request before the model sees it. A sort of status
codes, and a question about who can reach a server on 127.0.0.1.

The data are the hourly counts of Capital Bikeshare in Washington, D.C., in 2011, from the Bike Sharing dataset of
Fanaee-T and Gama (2013, CC BY 4.0, see data/README.md); the requests, the server and the pictures are the course's
own. Every word a learner typed, and every answer of a server, is shown through html.escape."""
import copy
import html
import http.client
import importlib.util
import json
import math
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import torch

from everything_ai import check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept, run_code

LESSON = "sys-serving-api"
LANGS = ("en", "nl")


def _course_demand():
    """The course's own copy of ops/demand.py from the lesson *Project: a demand model, tracked and packed*."""
    path = paths.LESSONS / "sys-project-tracked-model" / "project" / "ops" / "demand.py"
    spec = importlib.util.spec_from_file_location("_course_ops_demand_serving", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demand = _course_demand()

HOURS_2011 = demand.read_hours()                                  # 8,645 hours of 365 days
TRAIN_HOURS = HOURS_2011[~(HOURS_2011["day"] % 5 == 4)]           # the training days of the project
WATCH_DAY = 194                                                   # Thursday 14 July 2011, clear all day
RAIN_DAY = 340                                                    # Wednesday 7 December 2011, light rain at 17:00
FIELDS = ["hr"] + demand.FEATURES                                 # what the body of a request holds: 11 numbers


def _lang(lang) -> str:
    return lang if lang in LANGS else "en"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch, and a real minus sign."""
    out = f"{x:,.{digits}f}"
    out = out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out
    return out.replace("-", "−")


# ======== the course's model of the project ===============================================================================

def _trained():
    """The course's model of the project, trained once, the way the course's train_version trains h16-lr0.01 in the
    lesson *Project: a demand model, tracked and packed*: seed 0, 16 neurons, Adam with a learning rate of 0.01, 30
    epochs in batches of 64 hours, on the training days. On one thread, so that the numbers are the ones the lesson's
    text gives."""
    scale = demand.scaling(TRAIN_HOURS)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x, y = demand.inputs(TRAIN_HOURS, scale), demand.targets(TRAIN_HOURS)
        torch.manual_seed(0)
        model = demand.DemandNet(16)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(30):
            demand.train_epoch(model, opt, x, y, 64)
    finally:
        torch.set_num_threads(threads)
    model.eval()
    return model, scale


def _model():
    """Kept once per Python process, for both languages: a server loads its model once, when it starts."""
    return kept((LESSON, "model"), _trained)


_model()   # trained when the lesson loads, so that no check has to wait for it


def expect_one(request):
    """What the course's model of your project expects for one hour, in bikes: a tensor with one number. `request` is a
    dict with hr and the 10 features, as the body of a request has them; hr may be 8 or 8.0."""
    model, scale = _model()
    hours = pd.DataFrame([{name: request[name] for name in FIELDS}])
    hours["hr"] = hours["hr"].astype("int64")
    with torch.no_grad():
        return model(demand.inputs(hours, scale))[0, 0]


def problem(request) -> str:
    """What is wrong with the body of a request, as a short text for the client; "" for a good request: a JSON object
    with the 11 fields, each a number, and hr a whole number from 0 to 23."""
    if not isinstance(request, dict):
        return "send one JSON object with the fields " + ", ".join(FIELDS)
    missing = [name for name in FIELDS if name not in request]
    if missing:
        return ("missing field: " if len(missing) == 1 else "missing fields: ") + ", ".join(missing)
    for name in FIELDS:
        v = request[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"{name} must be a number, not {json.dumps(v)}"
    hr = request["hr"]
    if hr != int(hr) or not 0 <= hr <= 23:
        return f"hr must be a whole number from 0 to 23, not {json.dumps(hr)}"
    return ""


# ======== the requests of the lesson ======================================================================================

_WHOLE = {"hr", "weekday", "workingday", "holiday", "weathersit", "season", "mnth"}


class _Hour(dict):
    """The body of a request from the lesson's cells: the check's call shows it by its name, and the learner's
    function gets a plain copy."""

    def __init__(self, name: str, values: dict):
        super().__init__(values)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(dict(self), memo)


def _from_data(day: int, hr: int) -> dict:
    row = HOURS_2011[(HOURS_2011["day"] == day) & (HOURS_2011["hr"] == hr)].iloc[0]
    return {name: int(row[name]) if name in _WHOLE else round(float(row[name]), 4) for name in FIELDS}


def rented(day: int, hr: int) -> int:
    """How many bikes were rented in that hour."""
    return int(HOURS_2011[(HOURS_2011["day"] == day) & (HOURS_2011["hr"] == hr)]["cnt"].iloc[0])


DAY = [_Hour(f"C.DAY[{h}]", _from_data(WATCH_DAY, h)) for h in range(24)]   # every hour of 14 July 2011
RAIN = _Hour("C.RAIN", _from_data(RAIN_DAY, 17))                             # 7 December 2011 at 17:00, light rain
HOUR_24 = _Hour("C.HOUR_24", dict(DAY[0], hr=24))                            # midnight written as hour 24
NO_TEMP = _Hour("C.NO_TEMP", {k: v for k, v in DAY[8].items() if k != "temp"})
WARM = _Hour("C.WARM", dict(DAY[8], temp="warm"))


def _defined(code: str, name: str, names: dict):
    space = dict(names)
    exec(code, space)   # noqa: S102   the course's own solution text
    return space[name]


# ======== exercise 1: answer =============================================================================================

_A_START = ("def answer(request):\n"
            "    pred = expect_one(request)   # a tensor with one number: the bikes the model expects\n"
            "    bikes = ...\n"
            "    return {\"bikes\": bikes}\n")


def _a(value: str) -> str:
    return _A_START.replace("    bikes = ...\n", f"    bikes = {value}\n")


_A = _a("round(pred.item())")
_A_TENSOR = _a("pred")
_A_FLOAT = _a("pred.item()")
_A_INT = _a("int(pred.item())")
_A_METHOD = _a("pred.item")

answer = _defined(_A, "answer", {"expect_one": expect_one})   # the course's answer


@dataclass(frozen=True)
class NotJson:
    """What a check sees of a value that json cannot write: the kind of the first part it cannot write."""
    kind: str


def _unwritable(v) -> str:
    """The kind of the first part of v that json cannot write, such as Tensor."""
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                return type(k).__name__
            try:
                json.dumps(x, allow_nan=False)
            except (TypeError, ValueError):
                return _unwritable(x)
    if isinstance(v, (list, tuple)):
        for x in v:
            try:
                json.dumps(x, allow_nan=False)
            except (TypeError, ValueError):
                return _unwritable(x)
    if isinstance(v, float):
        return "NaN" if math.isnan(v) else "infinity"
    return type(v).__name__


def through_json(value):
    """What a client reads of `value` after the server wrote it as JSON, or NotJson when json cannot write it."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        return NotJson(_unwritable(value))


def _json_text(v) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(v)


def _code(text_: str) -> str:
    """A code span for a check message, which goes through Markdown: a text with a backtick in it as escaped HTML, so
    that what a learner's code gives back stays text."""
    return f"`{text_}`" if "`" not in text_ else f"<code>{html.escape(text_)}</code>"


def _shown_answer(v):
    """{got} and {want} of answer: the JSON a client reads, or what json cannot write."""
    if isinstance(v, NotJson):
        return {"en": "a value json cannot write: " + _kind(v.kind, "en"),
                "nl": "een waarde die json niet kan schrijven: " + _kind(v.kind, "nl")}
    return _code(_json_text(v))


def _round_on(kind: str):
    """An error of round() on a value that has no __round__, such as a tensor or a method that was not called."""
    return lambda e: isinstance(e, TypeError) and f"type {kind} doesn't define __round__" in str(e)


# ======== exercise 2: handle =============================================================================================

_H_HEAD = ('def handle(method, path, request):\n'
           '    if path == "/health":\n'
           '        return 200, {"status": "ok"}\n'
           '    if path != "/predict":\n'
           '        return 404, {"error": "no endpoint " + path}\n'
           '    if method != "POST":\n'
           '        return 405, {"error": "send a POST to /predict"}\n'
           '    trouble = problem(request)   # "" for a good request, otherwise what is wrong\n')
_H_END = '    return 200, answer(request)\n'


def _h(body: str) -> str:
    return _H_HEAD + body + _H_END


def _when(status: int, error: str = "trouble") -> str:
    return f'    if trouble != "":\n        return {status}, {{"error": {error}}}\n'


_H_START = _h("    ...\n")
_H = _h(_when(400))
_H_SERVER = _h(_when(500))
_H_OK = _h(_when(200))
_H_422 = _h(_when(422))
_H_FIXED = _h(_when(400, '"bad request"'))
_H_ALWAYS = _h('    return 400, {"error": trouble}\n')
_H_NONE = _h("    pass\n")

handle = _defined(_H, "handle", {"problem": problem, "answer": answer})   # the course's handle


def _own_words(got, text_) -> bool:
    """A bad request answered with the right status code and an error, but in words of its own, and not in what
    problem() says is wrong."""
    if not (isinstance(got, tuple) and len(got) == 2 and got[0] == 400 and isinstance(got[1], dict)):
        return False
    return "error" in got[1] and got[1]["error"] not in {problem(r) for r in (HOUR_24, NO_TEMP, WARM)}


def _from_the_model(e: BaseException) -> bool:
    """An error that came out of the model's code: the request reached expect_one."""
    return any(frame.name == "expect_one" for frame in traceback.extract_tb(e.__traceback__))


def _shown_reply(v):
    """{got} and {want} of handle: the status code and the JSON body."""
    if isinstance(v, tuple) and len(v) == 2:
        status, body = v
        return {"en": f"status {_code(repr(status))} with {_code(_json_text(body))}",
                "nl": f"status {_code(repr(status))} met {_code(_json_text(body))}"}
    return _code(_json_text(v))


def reply(fn, method: str, path: str, request) -> tuple:
    """What the server sends for a request with `fn` as its handler: the status code and the JSON fn gives, or 500
    when fn stops with an error, or gives no status code with something json can write."""
    try:
        status, body = fn(method, path, copy.deepcopy(request))
        text = json.dumps(body, allow_nan=False)
    except check.Timeout:
        raise
    except Exception as e:   # noqa: BLE001   a server answers every request, even when its handler breaks
        return 500, {"error": "the server stopped with an error: " + type(e).__name__}
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        return 500, {"error": "the handler gave no status code"}
    return status, json.loads(text)


def mine(box_text, correct: bool):
    """The learner's own handle when Exercise 2 is right, else None: the demo then uses the course's."""
    if not correct or not isinstance(box_text, str):
        return None
    run_ = run_code(box_text, dict(EXERCISES["handle"].names))
    fn = run_.names.get("handle") if run_.error is None else None
    return fn if callable(fn) else None


# ======== the server =====================================================================================================

_LOCAL = {"127.0.0.1", "localhost", "::1"}   # the names of this computer that a request may be addressed to
MAX_BODY = 10_000                             # bytes of a body at most: one hour is under 200
_REASONS = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
            413: "Content Too Large", 500: "Internal Server Error"}


class Server:
    """Python's ThreadingHTTPServer on 127.0.0.1, on a free port, in a thread of its own, with `handle` as its handler:
    every POST and GET goes through reply(handle, ...). `heard` keeps what it got: the request line, the headers and
    the body of every request."""

    def __init__(self, handle_):
        self.handle = handle_
        self.heard = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self.httpd.daemon_threads = True
        self.address, self.port = self.httpd.server_address[:2]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc) -> None:   # `with start(handle) as server:` stops the server, even after an error
        self.stop()


def _handler(server: Server):

    class Handler(BaseHTTPRequestHandler):
        server_version = "EverythingAI/1"
        sys_version = ""
        protocol_version = "HTTP/1.0"   # the connection closes after each answer

        def log_message(self, *args):   # nothing on the lesson's output
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status, _REASONS.get(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if status == 405:
                self.send_header("Allow", "POST")
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):   # the client stopped waiting
                pass

        def _local(self) -> bool:
            """Whether the request is addressed to this computer by name: a web page elsewhere that makes the
            learner's browser call this server (DNS rebinding) sends its own host name instead."""
            host = (self.headers.get("Host") or "").strip().lower()
            host = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
            if host in _LOCAL:
                return True
            self._send(403, {"error": "this server only answers requests addressed to 127.0.0.1 or localhost"})
            return False

        def _body(self):
            """The body as text, or None when it is too large or its length is not a number."""
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if not 0 <= length <= MAX_BODY:
                return None
            return self.rfile.read(length).decode("utf-8", "replace")

        def _hear(self, body) -> None:
            server.heard.append({"line": self.requestline, "headers": list(self.headers.items()), "body": body})

        def _answer(self, request) -> None:
            path = self.path.split("?", 1)[0]
            self._send(*reply(server.handle, self.command, path, request))

        def do_GET(self):
            self._hear(None)
            if self._local():
                self._answer(None)

        def do_POST(self):
            text = self._body()
            self._hear(text)
            if not self._local():
                return
            if text is None:
                self._send(413, {"error": f"send a body of at most {MAX_BODY} bytes"})
                return
            try:
                request = json.loads(text, parse_constant=_no_constant)
            except ValueError:
                self._send(400, {"error": "the body is not JSON"})
                return
            self._answer(request)

    return Handler


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")   # NaN and Infinity, which Python's json reads, and JSON does not have


def start(handle_) -> Server:
    """Start a server on 127.0.0.1 with `handle_` as its handler, on a free port."""
    return Server(handle_)


_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 127.0.0.1 never goes through a proxy


def send(server: Server, method: str, path: str, body=None) -> tuple:
    """Send `server` a request, as a client does: `body` as JSON. Gives back the status code and the JSON of the
    answer."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(server.url + path, data=data, headers=headers, method=method)
    try:
        with _NO_PROXY.open(request, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        with e:
            return e.code, json.loads(e.read())


def worked_example() -> list:
    """The lines the worked example prints, but the port, for the lesson's test."""
    with start(handle) as server:
        return [send(server, "GET", "/health"), send(server, "POST", "/predict", DAY[8]),
                send(server, "POST", "/forecast", DAY[8])]


def _path(text) -> str:
    """The path a learner typed, as a request line can hold it: no spaces, starting with /."""
    text = str(text or "").strip()[:200]
    text = urllib.parse.quote(text, safe="/?=&%:-._~")
    return text if text.startswith("/") else "/" + text


def exchange(fn, method: str, path: str, body_text: str) -> dict:
    """One request to a server with `fn` as its handler, from start to stop: what the server got, what it sent back,
    and how many milliseconds the client waited."""
    server = start(fn)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        data = str(body_text or "").encode("utf-8") if method == "POST" else None
        headers = {} if data is None else {"Content-Type": "application/json"}
        t = time.perf_counter()
        conn.request(method, _path(path), body=data, headers=headers)
        r = conn.getresponse()
        raw = r.read()
        ms = 1000 * (time.perf_counter() - t)
        conn.close()
        kept_headers = [(k, v) for k, v in r.getheaders() if k in ("Content-Type", "Content-Length", "Allow")]
        return {"heard": server.heard[-1] if server.heard else None, "status": r.status, "reason": r.reason,
                "version": "HTTP/1.0" if r.version == 10 else "HTTP/1.1", "headers": kept_headers,
                "body": raw.decode("utf-8", "replace"), "ms": ms}
    finally:
        server.stop()


DEMO_BODY = json.dumps(dict(DAY[8]), indent=1)   # what the demo's box begins with: 8:00 on 14 July 2011


# ======== numbers and small pieces of pictures ============================================================================

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"
_SVG = "width:100%;max-width:330px;height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums"
_PRE = ("font-family:var(--ea-font-code);font-size:12.5px;line-height:1.45;white-space:pre-wrap;overflow-wrap:anywhere;"
        "background:var(--ea-inset);border:1px solid var(--ea-line);border-radius:4px;padding:0.35rem 0.55rem;"
        "margin:0.15rem 0 0.5rem;color:var(--ea-ink);max-width:var(--ea-measure)")
_CAPTION = "font-family:var(--ea-font-ui);font-size:0.82rem;font-weight:600;color:var(--ea-ink-2);margin-top:0.35rem"
_CELL = ("padding:0.2rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px solid var(--ea-loss);outline-offset:-2px"
_MONO = "font-family:var(--ea-font-code);font-size:12px;overflow-wrap:anywhere"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code."""
    return "".join(f"<code>{part}</code>" if k % 2 else part for k, part in enumerate(html.escape(text_).split("`")))


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _t(x: float, y: float, text_: str, size: float = 12.5, weight: int = 400, fill: str = "var(--ea-ink)",
       anchor: str = "start", font: str = "") -> str:
    family = "font-family:var(--ea-font-code);" if font == "code" else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" style="{family}font-size:{size}px;font-weight:{weight};'
            f'fill:{fill}">{html.escape(text_)}</text>')


def _svg(height: float, alt: str, parts: list) -> str:
    return (f'<svg viewBox="0 0 330 {height:.0f}" role="img" aria-label="{html.escape(alt)}" style="{_SVG}">'
            f'{"".join(parts)}</svg>')


def clock(hr: int) -> str:
    return f"{hr}:00"


# ======== the picture of the idea ========================================================================================

IDEA_TEXT = {
    "en": {"title": "A request over HTTP, and its answer",
           "client": ("client", "truck planner"), "server": ("server", "127.0.0.1:8000"), "model": ("model", "DemandNet"),
           "request": "request", "answer": "answer", "heads": ("the request", "the answer"),
           "labels": {"line": "method and path", "headers": "headers", "body": "body: the input as JSON",
                      "status": "status code", "header": "header", "out": "body: the answer as JSON"},
           "alt": "A client sends a request to a server on 127.0.0.1, port 8000, which runs the model and sends an "
                  "answer back. Below, the request as text: the method POST and the path /predict, two headers, and a "
                  "body with the 11 fields of one hour as JSON. Then the answer: the status code 200 OK, a header, and "
                  "the body with the bikes the model expects as JSON."},
    "nl": {"title": "Een verzoek via HTTP, en het antwoord",
           "client": ("client", "planner"), "server": ("server", "127.0.0.1:8000"), "model": ("model", "DemandNet"),
           "request": "verzoek", "answer": "antwoord", "heads": ("het verzoek", "het antwoord"),
           "labels": {"line": "methode en pad", "headers": "headers", "body": "body: de input als JSON",
                      "status": "statuscode", "header": "header", "out": "body: het antwoord als JSON"},
           "alt": "Een client stuurt een verzoek naar een server op 127.0.0.1, poort 8000, en de server voert het model "
                  "uit en stuurt een antwoord terug. Daaronder het verzoek als tekst: de methode POST en het pad "
                  "/predict, twee headers, en een body met de 11 velden van één uur als JSON. Dan het antwoord: de "
                  "statuscode 200 OK, een header, en de body met het aantal fietsen dat het model verwacht, als JSON."},
}

_BODY_LINES = ['{"hr": 8, "weekday": 4, "workingday": 1,', ' "holiday": 0, "weathersit": 1,',
               ' "temp": 0.64, "atemp": 0.6212, "hum": 0.5,', ' "windspeed": 0.3582, "season": 3, "mnth": 7}']


def _box(x: float, y: float, w: float, name: str, sub: str) -> list:
    return [f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="40" rx="5" style="fill:var(--ea-page);stroke:var(--ea-ink-2);stroke-width:1.2"/>',
            _t(x + w / 2, y + 16, name, 12.5, 600, anchor="middle"), _t(x + w / 2, y + 31, sub, 10.5, 400, "var(--ea-ink-2)", "middle")]


def _arrow(x1: float, x2: float, y: float) -> str:
    """A line from x1 to x2 with a head at x2."""
    d = 6 if x2 > x1 else -6
    return (f'<line x1="{x1:.1f}" y1="{y}" x2="{x2 - d:.1f}" y2="{y}" style="stroke:var(--ea-ink-2);stroke-width:1.3"/>'
            f'<path d="M{x2:.1f},{y} L{x2 - d:.1f},{y - 4} L{x2 - d:.1f},{y + 4} Z" style="fill:var(--ea-ink-2)"/>')


def _block(y: float, head: str, lines: list) -> tuple[list, float]:
    """A block of text as it goes over the wire: (line, label) pairs, the label set on the right."""
    parts = [_t(0, y + 12, head, 12.5, 600)]
    y += 20
    height = 16 * len(lines) + 10
    parts.append(f'<rect x="0" y="{y}" width="330" height="{height}" rx="4" style="fill:var(--ea-inset);stroke:var(--ea-line);stroke-width:1"/>')
    for k, (line, label) in enumerate(lines):
        base = y + 17 + 16 * k
        parts.append(_t(8, base, line, 10.5, 600 if k == 0 else 400, font="code"))
        if label:
            parts.append(_t(322, base, label, 10.5, 400, "var(--ea-ink-3)", "end"))
    return parts, y + height + 10


def idea_picture(lang: str) -> str:
    lang = _lang(lang)
    w = IDEA_TEXT[lang]
    lab = w["labels"]
    parts = [_t(0, 15, w["title"], 14, 600)]
    y = 30
    parts += _box(0, y, 96, *w["client"]) + _box(122, y, 104, *w["server"]) + _box(252, y, 78, *w["model"])
    parts += [_arrow(98, 120, y + 13), _arrow(120, 98, y + 27), _arrow(228, 250, y + 13), _arrow(250, 228, y + 27)]
    parts.append(_t(109, y + 54, w["request"] + " / " + w["answer"], 10.5, 400, "var(--ea-ink-3)", "middle"))
    y += 66
    request = [("POST /predict HTTP/1.1", lab["line"]), ("Host: 127.0.0.1:8000", lab["headers"]),
               ("Content-Type: application/json", ""), ("", lab["body"])] + [(line, "") for line in _BODY_LINES]
    block, y = _block(y, w["heads"][0], request)
    parts += block
    answer_ = [("HTTP/1.0 200 OK", lab["status"]), ("Content-Type: application/json", lab["header"]), ("", lab["out"]),
               ('{"bikes": 431}', "")]
    block, y = _block(y, w["heads"][1], answer_)
    parts += block
    return _svg(y, w["alt"], parts)


# ======== the pictures of the learner's work =============================================================================

_STOPPED = object()   # what _call gives for a function that stopped with an error or took too long
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_):
    try:
        return check.with_time_limit(make_, PICTURE_SECONDS)
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _bikes(v):
    """The number under bikes of what a client read, or None."""
    if not isinstance(v, dict) or "bikes" not in v:
        return None
    b = v["bikes"]
    return b if not isinstance(b, bool) and isinstance(b, (int, float)) and math.isfinite(b) else None


YOURS_TEXT = {
    "en": {"answer": {"title": "Your answer for every hour of 14 July 2011",
                      "intro": "What your `answer(request)` gives for the 24 hours of the day, as a client reads it after the server wrote it as JSON.",
                      "at": "8:00: {json}",
                      "range": "The busiest hour is {hi}, with {hi_n} bikes; the quietest is {lo}, with {lo_n}.",
                      "below": "At {hours}, your answer says fewer than 0 bikes: the server passes on what the model says.",
                      "alt": "A bar for every hour of 14 July 2011: the number of bikes your answer gives, from 0:00 to 23:00.",
                      "not_json": "json cannot write what your `answer` gives for 8:00: {kind}. The server would stop with an error, and the client would get status 500.",
                      "none": "Your `answer` stops with an error, takes too long, or does not give a dict with a number under `bikes`, so nothing can be drawn yet.",
                      "and": " and "},
           "handle": {"title": "Your handle on seven requests",
                      "intro": "What a client gets with your `handle(method, path, request)` in the server: an error in your function becomes status 500, as in a real server.",
                      "head": ("request", "status", "body of the answer"),
                      "tries": {"8": "8:00, all 11 fields", "24": "midnight as hour 24", "temp": "no temp",
                                "warm": "\"warm\" as temp", "get": "a GET", "forecast": "another path", "health": "is the server up?"},
                      "same": "{k} of the 7 requests get the same status code as with the course's handle; a frame marks the others.",
                      "all": "All 7 requests get the same status code as with the course's handle.",
                      "none": "Your `handle` takes too long, or the box makes no function `handle` yet, so nothing can be drawn yet."}},
    "nl": {"answer": {"title": "Je answer voor elk uur van 14 juli 2011",
                      "intro": "Wat je `answer(request)` geeft voor de 24 uren van de dag, zoals een client het leest nadat de server het als JSON schreef.",
                      "at": "8:00: {json}",
                      "range": "Het drukste uur is {hi}, met {hi_n} fietsen; het rustigste is {lo}, met {lo_n}.",
                      "below": "Om {hours} zegt je answer minder dan 0 fietsen: de server geeft door wat het model zegt.",
                      "alt": "Een balk voor elk uur van 14 juli 2011: het aantal fietsen dat je answer geeft, van 0:00 tot 23:00.",
                      "not_json": "json kan niet schrijven wat je `answer` voor 8:00 geeft: {kind}. De server zou stoppen met een fout, en de client zou status 500 krijgen.",
                      "none": "Je `answer` stopt met een fout, duurt te lang, of geeft geen dict met een getal onder `bikes`, dus is er nog geen afbeelding.",
                      "and": " en "},
           "handle": {"title": "Je handle bij zeven verzoeken",
                      "intro": "Wat een client krijgt met je `handle(method, path, request)` in de server: een fout in je functie wordt status 500, net als in een echte server.",
                      "head": ("verzoek", "status", "body van het antwoord"),
                      "tries": {"8": "8:00, alle 11 velden", "24": "middernacht als uur 24", "temp": "geen temp",
                                "warm": "\"warm\" als temp", "get": "een GET", "forecast": "een ander pad", "health": "draait de server?"},
                      "same": "{k} van de 7 verzoeken krijgen dezelfde statuscode als met de handle van de course; een kader markeert de andere.",
                      "all": "Alle 7 verzoeken krijgen dezelfde statuscode als met de handle van de course.",
                      "none": "Je `handle` duurt te lang, of het vak maakt nog geen functie `handle`, dus is er nog geen afbeelding."}},
}

_KINDS = {"en": {"Tensor": "a tensor", "builtin_function_or_method": "a method that was not called",
                 "ellipsis": "the `...` of the start code", "ndarray": "a NumPy array",
                 "float": "a number that is not finite, such as nan"},
          "nl": {"Tensor": "een tensor", "builtin_function_or_method": "een methode die niet werd aangeroepen",
                 "ellipsis": "de `...` van de begincode", "ndarray": "een NumPy-array",
                 "float": "een getal dat niet eindig is, zoals nan"}}


def _kind(kind: str, lang: str) -> str:
    return _KINDS[lang].get(kind, ("a " if lang == "en" else "een ") + kind)


def bars_svg(values: list, lang: str, alt: str) -> str:
    """A bar for every hour of the day: the bikes, with a line at 0 and room below it for numbers under 0."""
    L, R, TOP, H = 30, 326, 8, 150
    hi = max(max(values), 100)
    lo = min(min(values), 0)
    span = hi - lo
    ys = lambda v: TOP + H * (hi - v) / span   # noqa: E731
    step = 100 if hi <= 800 else (500 if hi <= 4000 else 2000)
    parts = []
    tick = math.ceil(lo / step) * step
    while tick <= hi:
        parts.append(f'<line x1="{L}" y1="{ys(tick):.1f}" x2="{R}" y2="{ys(tick):.1f}" style="stroke:var(--ea-line);stroke-width:0.8"/>'
                     + _t(L - 5, ys(tick) + 4, number(tick, 0, lang), 10.5, 400, "var(--ea-ink-3)", "end"))
        tick += step
    width = (R - L) / 24
    for h, v in enumerate(values):
        top, bottom = (ys(v), ys(0)) if v >= 0 else (ys(0), ys(v))
        parts.append(f'<rect x="{L + h * width + 1:.1f}" y="{top:.1f}" width="{width - 2:.1f}" height="{max(bottom - top, 0.8):.1f}" '
                     f'style="fill:{"var(--ea-ink)" if h == 8 else "var(--ea-ink-3)"}"/>')
    parts.append(f'<line x1="{L}" y1="{ys(0):.1f}" x2="{R}" y2="{ys(0):.1f}" style="stroke:var(--ea-ink-2);stroke-width:1"/>')
    for h in (0, 6, 12, 18):
        parts.append(_t(L + (h + 0.5) * width, TOP + H + 16, clock(h), 10.5, 400, "var(--ea-ink-3)", "middle"))
    return _svg(TOP + H + 22, alt, parts)


def answer_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["answer"]
    fn = _fn(seen)
    got = _call(lambda: [through_json(fn(copy.deepcopy(dict(r)))) for r in DAY]) if fn is not None else _STOPPED
    if got is not _STOPPED and isinstance(got[8], NotJson):
        body = _note(w["intro"]) + _note(w["not_json"].format(kind=_kind(got[8].kind, lang)), _LOSS_UI)
        return yours.figure(w["title"], body, alt=w["title"])
    values = [_bikes(v) for v in got] if got is not _STOPPED else [None]
    if any(v is None for v in values):
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    hi = max(range(24), key=lambda h: values[h])
    lo = min(range(24), key=lambda h: values[h])
    show = lambda v: number(v, 0 if v == int(v) else 1, lang)   # noqa: E731
    said = w["range"].format(hi=clock(hi), hi_n=show(values[hi]), lo=clock(lo), lo_n=show(values[lo]))
    below = [clock(h) for h in range(24) if values[h] < 0]
    if below:
        said += " " + w["below"].format(hours=", ".join(below[:-1]) + w["and"] + below[-1] if len(below) > 1 else below[0])
    body = (_note(w["intro"]) + _note(w["at"].format(json="`" + _json_text(got[8]) + "`"))
            + bars_svg(values, lang, w["alt"]) + _note(said))
    return yours.figure(w["title"], body, alt=w["title"])


TRIES = [("8", "POST", "/predict", DAY[8]), ("24", "POST", "/predict", HOUR_24), ("temp", "POST", "/predict", NO_TEMP),
         ("warm", "POST", "/predict", WARM), ("get", "GET", "/predict", DAY[8]), ("forecast", "POST", "/forecast", DAY[8]),
         ("health", "GET", "/health", None)]


def _short(text_: str, most: int = 90) -> str:
    return text_ if len(text_) <= most else text_[:most - 1] + "…"


def handle_picture(seen) -> str:
    lang = _lang(seen.lang)
    w = YOURS_TEXT[lang]["handle"]
    fn = _fn(seen)
    got = _call(lambda: [reply(fn, m, p, r) for _, m, p, r in TRIES]) if fn is not None else _STOPPED
    if got is _STOPPED:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    rows, same = [], 0
    for (key, method, path, request_), (status, body) in zip(TRIES, got):
        right = status == reply(handle, method, path, request_)[0]
        same += right
        request = (f'<span style="{_MONO}">{html.escape(method + " " + path)}</span><br>'
                   f'<span style="color:var(--ea-ink-2)">{html.escape(w["tries"][key])}</span>')
        cell = f'<span style="{_MONO};font-weight:600">{status}</span>'
        rows.append([(request, ""), (cell, "" if right else _OFF),
                     (f'<span style="{_MONO}">{html.escape(_short(_json_text(body)))}</span>', "")])
    head = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in w["head"])
    table = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{c}</td>' for c, s in row) + "</tr>" for row in rows)
    said = w["all"] if same == len(TRIES) else w["same"].format(k=same)
    body = (_note(w["intro"]) + f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{head}</tr>{table}</table></div>' + _note(said))
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the demo: your server, on your own computer ====================================================================

DEMO_TEXT = {
    "en": {"title": "Your request, and what the server sent back", "got": "What the server got",
           "sent": "What it sent back", "mine": "Answered in {ms} ms, with your handle of Exercise 2.",
           "course": "Answered in {ms} ms, with the course's handle, until yours of Exercise 2 is right.",
           "silent": "No answer came within 10 seconds. Does your handle keep going forever for this request?"},
    "nl": {"title": "Je verzoek, en wat de server terugstuurde", "got": "Wat de server kreeg",
           "sent": "Wat de server terugstuurde", "mine": "Beantwoord in {ms} ms, met je handle van Oefening 2.",
           "course": "Beantwoord in {ms} ms, met de handle van de course, tot je eigen handle van Oefening 2 klopt.",
           "silent": "Binnen 10 seconden kwam er geen antwoord. Gaat je handle bij dit verzoek eindeloos door?"},
}


def _pre(text_: str) -> str:
    return f'<pre style="{_PRE}">{html.escape(text_)}</pre>'


def exchange_picture(method: str, path: str, body_text: str, fn, lang: str) -> str:
    """The demo: one request to a real server on 127.0.0.1 with `fn` (or the course's handle), as it went over the
    wire."""
    lang = _lang(lang)
    w = DEMO_TEXT[lang]
    method = method if method in ("POST", "GET") else "POST"
    try:
        out = exchange(fn or handle, method, path, body_text)
    except OSError:   # no answer in time, or the connection broke: a handle that keeps going, say
        return yours.figure(w["title"], _note(w["silent"], _LOSS_UI), alt=w["title"])
    heard = out["heard"] or {"line": "", "headers": [], "body": None}
    got = "\n".join([heard["line"]] + [f"{k}: {v}" for k, v in heard["headers"]])
    if heard["body"] is not None:
        got += "\n\n" + heard["body"]
    sent = "\n".join([f'{out["version"]} {out["status"]} {out["reason"]}'] + [f"{k}: {v}" for k, v in out["headers"]])
    sent += "\n\n" + out["body"]
    said = (w["mine"] if fn is not None else w["course"]).format(ms=number(out["ms"], 1, lang))
    body = (f'<div style="{_CAPTION}">{html.escape(w["got"])}</div>' + _pre(got)
            + f'<div style="{_CAPTION}">{html.escape(w["sent"])}</div>' + _pre(sent) + _note(said))
    return yours.figure(w["title"], body, alt=w["title"])


# ======== the exercises ===================================================================================================

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

CODES = Sort(
    name="codes",
    groups={
        "ok": {"en": "200 OK", "nl": "200 OK"},
        "bad": {"en": "400 Bad Request", "nl": "400 Bad Request"},
        "missing": {"en": "404 Not Found", "nl": "404 Not Found"},
        "broken": {"en": "500 Internal Server Error", "nl": "500 Internal Server Error"},
    },
    items={
        "rush": {"en": "**Rush hour.** POST /predict with all 11 fields, and hr 17.",
                 "nl": "**Spits.** POST /predict met alle 11 velden, en hr 17."},
        "health": {"en": "**Still up?** GET /health, while the server runs.",
                   "nl": "**Nog wakker?** GET /health, terwijl de server draait."},
        "warm": {"en": "**Warm.** POST /predict with \"temp\": \"warm\".",
                 "nl": "**Warm.** POST /predict met \"temp\": \"warm\"."},
        "humidity": {"en": "**No humidity.** POST /predict without the field hum.",
                     "nl": "**Geen luchtvochtigheid.** POST /predict zonder het veld hum."},
        "forecast": {"en": "**Forecast.** POST /forecast, with the same body as at rush hour.",
                     "nl": "**Verwachting.** POST /forecast, met dezelfde body als bij de spits."},
        "capital": {"en": "**A capital.** POST /Predict, with a capital P, and a good body.",
                    "nl": "**Een hoofdletter.** POST /Predict, met een hoofdletter P, en een goede body."},
        "bug": {"en": "**A bug.** POST /predict with a good body, after a colleague added a line to `answer` that divides by zero.",
                "nl": "**Een bug.** POST /predict met een goede body, nadat een collega aan `answer` een regel toevoegde met een deling door nul."},
        "memory": {"en": "**Out of memory.** POST /predict with a good body, while the computer of the server has no memory left.",
                   "nl": "**Geheugen vol.** POST /predict met een goede body, terwijl de computer van de server geen geheugen meer over heeft."},
    },
    solution={"rush": "ok", "health": "ok", "warm": "bad", "humidity": "bad", "forecast": "missing", "capital": "missing",
              "bug": "broken", "memory": "broken"},
    nudge={
        "rush": {"en": "Is anything wrong with this request, or with the server?",
                 "nl": "Is er iets mis met dit verzoek, of met de server?"},
        "health": {"en": "What is /health for, and is anything wrong while the server runs?",
                   "nl": "Waar is /health voor, en is er iets mis zolang de server draait?"},
        "warm": {"en": "Whose mistake is a temperature written as a word: the client's or the server's?",
                 "nl": "Wie maakt de fout bij een temperatuur als woord: de client of de server?"},
        "humidity": {"en": "Can the model work without one of its fields? And who has to change the request?",
                     "nl": "Kan het model werken zonder een van de velden? En wie moet het verzoek aanpassen?"},
        "forecast": {"en": "Does the server have an endpoint /forecast?",
                     "nl": "Heeft de server een endpoint /forecast?"},
        "capital": {"en": "Is /Predict the same path as /predict, letter for letter?",
                    "nl": "Is /Predict hetzelfde pad als /predict, letter voor letter?"},
        "bug": {"en": "The request is good. Whose mistake is a division by zero in the code of the server?",
                "nl": "Het verzoek is goed. Wie maakt de fout bij een deling door nul in de code van de server?"},
        "memory": {"en": "The request is good. Could the same request work later, once the server has memory again?",
                   "nl": "Het verzoek is goed. Kan hetzelfde verzoek later werken, als de server weer geheugen heeft?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the status codes again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de statuscodes opnieuw uit."},
        "empty": {"en": "Put each request under the status code its answer should have.",
                  "nl": "Zet elk verzoek onder de statuscode die het antwoord moet hebben."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "*{item}* should get another status code.",
                  "nl": "*{item}* hoort een andere statuscode te krijgen."},
        "correct": {"en": "All {n} are right. 200 when it went well; 400 when the request is wrong and the client must change it; 404 for a path the server does not have, even with one capital letter; 500 when the server broke on a good request, which may work later, unchanged.",
                    "nl": "Alle {n} kloppen. 200 als het goed ging; 400 als het verzoek fout is en de client het moet aanpassen; 404 voor een pad dat de server niet heeft, ook met één hoofdletter; 500 als de server stukging op een goed verzoek, dat later onveranderd kan werken."},
    },
)

EXERCISES = {
    "answer": FunctionBox(
        name="answer",
        function="answer",
        start=_A_START,
        solution=_A,
        names={"expect_one": expect_one},
        visible=[(DAY[8],)],
        hidden=[(DAY[10],), (DAY[17],), (DAY[12],), (RAIN,)],
        first=[_DOTS],
        collect=through_json,
        show=_shown_answer,
        picture=answer_picture,
        mistakes=[
            Mistake("round-tensor", error=_round_on("Tensor"),
                    en="With `{call}`, `round` gets a tensor, and `round` only works on a plain Python number. Which method of a tensor gives the number inside it?",
                    nl="Met `{call}` krijgt `round` een tensor, en `round` werkt alleen op een gewoon Python-getal. Welke methode van een tensor geeft het getal dat erin zit?"),
            Mistake("parens", error=_round_on("builtin_function_or_method"),
                    en="With `{call}`, `round` gets the method itself, and not the number it gives back: a method only runs with parentheses after it. Did you forget the `()`?",
                    nl="Met `{call}` krijgt `round` de methode zelf, en niet het getal dat de methode teruggeeft: een methode werkt alleen met haakjes erachter. Ben je de `()` vergeten?"),
            Mistake("tensor", wrong=_A_TENSOR,
                    en="With `{call}`, `bikes` is a tensor, and json cannot write a tensor: the server would stop with an error and answer 500. Which method of a tensor gives the number inside it as a plain Python number?",
                    nl="Met `{call}` is `bikes` een tensor, en json kan geen tensor schrijven: de server zou stoppen met een fout en 500 antwoorden. Welke methode van een tensor geeft het getal dat erin zit als gewoon Python-getal?"),
            Mistake("method", wrong=_A_METHOD,
                    en="With `{call}`, `bikes` is the method itself, and not the number it gives back: a method only runs with parentheses after it. Did you forget the `()`?",
                    nl="Met `{call}` is `bikes` de methode zelf, en niet het getal dat de methode teruggeeft: een methode werkt alleen met haakjes erachter. Ben je de `()` vergeten?"),
            Mistake("float", wrong=_A_FLOAT,
                    en="`{call}` gives {got}: a number of bikes with a fraction, which no truck can bring. Which Python function rounds it to a whole number?",
                    nl="`{call}` geeft {got}: een aantal fietsen met cijfers achter de komma, en geen vrachtwagen kan een deel van een fiets brengen. Welke Python-functie rondt het af op een heel getal?"),
            Mistake("int", wrong=_A_INT, state="almost",
                    en="Right for 8:00. But `{call}` gives {got}, and it should give {want}: `int` cuts off everything after the decimal point, so 143.547 bikes becomes 143. Which function rounds to the nearest whole number?",
                    nl="Goed voor 8:00. Maar `{call}` geeft {got}, en het moet {want} zijn: `int` haalt alles na de komma weg, dus wordt 143,547 fietsen 143. Welke functie rondt af op het dichtstbijzijnde hele getal?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with what `bikes` should be: the number in `pred`, as a whole number.",
                      "nl": "Vervang de `...` door wat `bikes` moet zijn: het getal in `pred`, als heel getal."},
            "arguments": {"en": "The check calls `{call}`, with one dict, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(request)`?",
                          "nl": "De check roept `{call}` aan, met één dict, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(request)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `bikes` the number in `pred`, rounded to a whole number?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `bikes` het getal in `pred`, afgerond op een heel getal?"},
            "almost": {"en": "Right for 8:00. But `{call}` gives {got}, and it should give {want}. Does `bikes` come from `pred`, and not from a fixed number?",
                       "nl": "Goed voor 8:00. Maar `{call}` geeft {got}, en het moet {want} zijn. Komt `bikes` uit `pred`, en niet uit een vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return {{\"bikes\": bikes}}`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return {{\"bikes\": bikes}}`?"},
            "correct": {"en": "Right in all {n} cases. At 8:00 on 14 July 2011, the server answers `{{\"bikes\": 431}}`: any program that speaks HTTP can read it, in Python or not. In the picture: what does your answer say at night?",
                        "nl": "Goed in alle {n} gevallen. Om 8:00 op 14 juli 2011 antwoordt de server `{{\"bikes\": 431}}`: elk programma dat HTTP spreekt, kan het lezen, in Python of niet. In de afbeelding: wat zegt je answer 's nachts?"},
        },
    ),
    "handle": FunctionBox(
        name="handle",
        function="handle",
        start=_H_START,
        solution=_H,
        names={"problem": problem, "answer": answer},
        visible=[("POST", "/predict", HOUR_24)],
        hidden=[("POST", "/predict", DAY[8]), ("POST", "/predict", NO_TEMP), ("POST", "/predict", WARM),
                ("GET", "/predict", DAY[8]), ("POST", "/forecast", DAY[8]), ("GET", "/health", None)],
        first=[_DOTS],
        show=_shown_reply,
        picture=handle_picture,
        mistakes=[
            Mistake("no-check", error=_from_the_model,
                    en="With `{call}`, the request reaches the model, and the code of the model stops with an error: a server then answers 500, as if the server broke. Does your check stop every request where `trouble` is not empty, before the model sees it?",
                    nl="Met `{call}` komt het verzoek bij het model, en de code van het model stopt met een fout: een server antwoordt dan 500, alsof de server stukging. Houdt je controle elk verzoek tegen waarbij `trouble` niet leeg is, nog voordat het model het verzoek ziet?"),
            Mistake("server-error", wrong=_H_SERVER,
                    en="`{call}` gives {got}. A status code from 500 on says that the server broke, and that the same request may work later. But here the request is wrong, and it will stay wrong. Which status code tells the client to change the request?",
                    nl="`{call}` geeft {got}. Een statuscode vanaf 500 zegt dat de server stukging, en dat hetzelfde verzoek later kan werken. Maar hier is het verzoek fout, en dat blijft het. Welke statuscode zegt tegen de client dat het verzoek anders moet?"),
            Mistake("ok-error", wrong=_H_OK,
                    en="`{call}` gives {got}: status 200 says that all went well, so a client that looks at the status code takes the error for an answer. Which status code says that the request was wrong?",
                    nl="`{call}` geeft {got}: status 200 zegt dat alles goed ging, dus houdt een client die naar de statuscode kijkt, de fout voor een antwoord. Welke statuscode zegt dat het verzoek fout was?"),
            Mistake("unprocessable", wrong=_H_422,
                    en="`{call}` gives {got}. FastAPI answers 422 when the fields do not fit, and that is a fine choice there; this server answers every bad request with the one general status code for a bad request, as the table of the idea does. Which code is that?",
                    nl="`{call}` geeft {got}. FastAPI antwoordt 422 als de velden niet kloppen, en daar is dat een goede keuze; deze server antwoordt op elk fout verzoek met de ene algemene statuscode voor een fout verzoek, zoals de tabel van het idee doet. Welke code is dat?"),
            Mistake("always", wrong=_H_ALWAYS, state="almost",
                    en="Right for hour 24. But `{call}` gives {got}: a good request gets an error too. Do your two lines answer with an error only when `trouble` is not empty?",
                    nl="Goed voor uur 24. Maar `{call}` geeft {got}: ook een goed verzoek krijgt een fout. Antwoorden je twee regels alleen met een fout als `trouble` niet leeg is?"),
            Mistake("fixed", matches=_own_words,
                    en="`{call}` gives {got}: the right status code, but the error does not say what is wrong with this request, so the client cannot tell what to change. Which text says what is wrong?",
                    nl="`{call}` geeft {got}: de juiste statuscode, alleen zegt de foutmelding niet wat er mis is met dit verzoek, dus weet de client niet wat er anders moet. Welke tekst zegt wat er mis is?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: when `trouble` is not empty, answer with the status code for a bad request and `{{\"error\": trouble}}`.",
                      "nl": "Vervang de `...` door twee regels: als `trouble` niet leeg is, antwoord dan met de statuscode voor een fout verzoek en `{{\"error\": trouble}}`."},
            "arguments": {"en": "The check calls `{call}`, with a method, a path and a request, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(method, path, request)`?",
                          "nl": "De check roept `{call}` aan, met een methode, een pad en een verzoek, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(method, path, request)` houden?"},
            "wrong": {"en": "`{call}` gives {got}. Does `handle` answer a bad request with the status code for a bad request, and `{{\"error\": trouble}}`?",
                      "nl": "`{call}` geeft {got}. Antwoordt `handle` op een fout verzoek met de statuscode voor een fout verzoek, en `{{\"error\": trouble}}`?"},
            "almost": {"en": "Right for hour 24. But `{call}` gives {got}, and it should give {want}. Do the other requests still go through the lines of the course?",
                       "nl": "Goed voor uur 24. Maar `{call}` geeft {got}, en het moet {want} zijn. Gaan de andere verzoeken nog door de regels van de course?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return 200, answer(request)`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return 200, answer(request)`?"},
            "correct": {"en": "Right in all {n} cases. Hour 24 now gets the status code of a bad request, with what is wrong: the client knows what to change, and the model never sees the request. In the picture: which requests reach the model?",
                        "nl": "Goed in alle {n} gevallen. Uur 24 krijgt nu de statuscode van een fout verzoek, met wat er mis is: de client weet wat er anders moet, en het model ziet het verzoek nooit. In de afbeelding: welke verzoeken komen bij het model?"},
        },
    ),
    "codes": CODES,
}

# The lesson is done when these are correct. "reach" is the choice question about 127.0.0.1.
REQUIRED = ["answer", "handle", "codes", "reach"]

_SORTED = dict(CODES.solution)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("answer", _A_START, "empty", None),
    ("answer", "", "empty", None),
    ("answer", _A, "correct", None),
    ("answer", _a("round(float(pred))"), "correct", None),
    ("answer", _a("round(pred.item(), 0)"), "correct", None),
    ("answer", _a("int(pred.item() + 0.5)"), "correct", None),
    ("answer", _A_TENSOR, "wrong", "tensor"),
    ("answer", _A_METHOD, "wrong", "method"),
    ("answer", _A_FLOAT, "wrong", "float"),
    ("answer", _a("float(pred)"), "wrong", "float"),
    ("answer", _A_INT, "almost", "int"),
    ("answer", _a("int(pred)"), "almost", "int"),
    ("answer", _a("round(pred)"), "error", "round-tensor"),
    ("answer", _a("round(pred.item)"), "error", "parens"),
    ("answer", _a("round(...)"), "wrong", "dots-left"),
    ("answer", _A.replace("    return {\"bikes\": bikes}\n", ""), "wrong", "no-return"),
    ("answer", _A.replace("def answer(request):", "def answer():"), "error", "arguments"),
    ("answer", "def answer(request):\n    return {\"bikes\": 431}\n", "almost", None),
    ("answer", _a("str(round(pred.item()))"), "wrong", None),
    ("answer", "x = 1\n", "wrong", "no-function"),
    ("handle", _H_START, "empty", None),
    ("handle", "", "empty", None),
    ("handle", _H, "correct", None),
    ("handle", _h('    if trouble:\n        return 400, {"error": trouble}\n'), "correct", None),
    ("handle", _h('    if len(trouble) > 0:\n        return 400, {"error": trouble}\n'), "correct", None),
    ("handle", _H_NONE, "error", "no-check"),
    ("handle", _h('    if trouble == "":\n        return 400, {"error": trouble}\n'), "error", "no-check"),
    ("handle", _H_SERVER, "wrong", "server-error"),
    ("handle", _H_OK, "wrong", "ok-error"),
    ("handle", _H_422, "wrong", "unprocessable"),
    ("handle", _H_FIXED, "wrong", "fixed"),
    ("handle", _h(_when(400, '"Bad Request: " + path')), "wrong", "fixed"),
    ("handle", _H_ALWAYS, "almost", "always"),
    ("handle", _h('    if trouble != "":\n        ...\n'), "wrong", "dots-left"),
    ("handle", _H.replace(_H_END, ""), "wrong", "unreached"),
    ("handle", _H.replace("def handle(method, path, request):", "def handle(method, path):"), "error", "arguments"),
    ("handle", "def handle(method, path, request):\n    return 400, {\"error\": \"hr must be a whole number from 0 to 23, not 24\"}\n",
     "almost", None),
    ("handle", _h('    if trouble != "":\n        return 400, {"problem": trouble}\n'), "wrong", None),
    ("handle", "x = 1\n", "wrong", "no-function"),
    ("codes", {}, "empty", None),
    ("codes", _SORTED, "correct", None),
    ("codes", {"rush": "ok", "health": "ok"}, "wrong", "unfinished"),
    ("codes", _SORTED | {"rush": "bad"}, "wrong", "wrong:rush"),
    ("codes", _SORTED | {"health": "broken"}, "wrong", "wrong:health"),
    ("codes", _SORTED | {"warm": "broken"}, "wrong", "wrong:warm"),
    ("codes", _SORTED | {"humidity": "missing"}, "wrong", "wrong:humidity"),
    ("codes", _SORTED | {"forecast": "bad"}, "wrong", "wrong:forecast"),
    ("codes", _SORTED | {"capital": "ok"}, "wrong", "wrong:capital"),
    ("codes", _SORTED | {"bug": "bad"}, "wrong", "wrong:bug"),
    ("codes", _SORTED | {"memory": "bad"}, "wrong", "wrong:memory"),
]
