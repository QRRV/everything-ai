# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-mcp: tools through a protocol. The tool convert of build-tool-calling, offered by an MCP server of the
course over stdio: JSON-RPC messages, one per line, in MCP version 2026-07-28, in which every request carries its
protocol version in _meta. The course's recorded runs of Qwen3-0.6B (data/course-agents-qwen.json): the eight
questions of build-tool-calling, asked with the tools of six servers instead of convert alone. What they show, their
pictures, the exercises (the server's answers to tools/list and tools/call, and a client's tools/call) and the demo,
in which the learner's server runs as a program of its own and Qwen3 uses its tool."""
import copy
import html
import importlib.util
import json
import os
import subprocess
import sys

from everything_ai import chat, paths, serve, yours
from everything_ai.check import MISSING, FunctionBox, Mistake, run_code

DOWNLOADS = {"Qwen3-0.6B": chat.FILES}   # what the demo downloads, for the line under the header
DATA = paths.ROOT / "data" / "course-agents-qwen.json"
VERSION = "2026-07-28"   # the version of MCP the lesson teaches: the newest, since July 2026


def _lesson_checks(lesson_id: str):
    """The checks of an earlier lesson, loaded once for this module."""
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}_for_mcp",
                                                  paths.LESSONS / lesson_id / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _lesson_checks("build-tool-calling")   # the tool convert, its description, the eight questions and their runs
QUESTIONS, CONVERT = T.QUESTIONS, T.CONVERT

# --- the course's MCP server: the tool convert, over stdio ------------------------------------------------------------

SERVER_HEAD = '''# An MCP server of Everything AI: the tool convert, over stdio, in MCP version 2026-07-28.
import json
import sys

UNITS = {"km": ("length", 1000.0), "mi": ("length", 1609.344), "kg": ("mass", 1.0), "lb": ("mass", 0.45359237)}


def convert(arguments):
    amount, a, b = arguments.get("amount"), arguments.get("from_unit"), arguments.get("to_unit")
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return "Error: amount must be a number"
    if not all(isinstance(u, str) and u in UNITS for u in (a, b)):
        return "Error: unknown unit " + str(b if isinstance(a, str) and a in UNITS else a)
    if UNITS[a][0] != UNITS[b][0]:
        return "Error: cannot convert " + a + " to " + b
    return f"{round(amount * UNITS[a][1] / UNITS[b][1], 2):.2f}".rstrip("0").rstrip(".") + " " + b


TOOLS = [{
    "name": "convert",
    "description": "Convert an amount from one unit to another: kilometres and miles, or kilograms and pounds.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "the amount to convert, such as 37"},
            "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"], "description": "the unit of the amount"},
            "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb"], "description": "the unit to convert to"},
        },
        "required": ["amount", "from_unit", "to_unit"],
    },
}]
FUNCTIONS = {"convert": convert}
'''
SERVER_TAIL = '''

for line in sys.stdin:   # one JSON-RPC message per line; the answer goes out as one line on stdout
    if not line.strip():
        continue
    try:
        message = json.loads(line)
    except ValueError:
        print(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}), flush=True)
        continue
    answer = handle(message)
    if answer is not None:
        print(json.dumps(answer), flush=True)
'''
_HEAD = run_code(SERVER_HEAD)
SERVER_TOOLS, FUNCTIONS = _HEAD.names["TOOLS"], _HEAD.names["FUNCTIONS"]
SERVER = {"TOOLS": SERVER_TOOLS, "FUNCTIONS": FUNCTIONS}   # what the box of exercise 1 may use

META = {"io.modelcontextprotocol/protocolVersion": VERSION, "io.modelcontextprotocol/clientCapabilities": {}}


def as_tools(listed: list) -> list:
    """The course's client: the tools an MCP server lists, in the form of the chat API's tools."""
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["inputSchema"]}} for t in listed]


# --- more servers: the tools a client with six servers gives the model ---------------------------------------------------

def _mcp(name: str, description: str, properties: dict, required: list | None = None) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties, "required": list(properties) if required is None else required}}


def _s(about: str) -> dict:
    return {"type": "string", "description": about}


OTHER = {   # server: its tools, in MCP's form
    "kitchen": [_mcp("recipe", "The ingredients of a dish, with the grams of each.", {"dish": _s("the name of the dish, such as pancakes")}),
                _mcp("pantry", "How many grams of an ingredient are in the pantry now.", {"item": _s("the name of the ingredient, such as milk")})],
    "calendar": [_mcp("list_events", "The events in the user's calendar on one day.", {"day": _s("the day, such as 2026-11-20")}),
                 _mcp("add_event", "Add an event to the user's calendar.", {"day": _s("the day, such as 2026-11-20"),
                      "time": _s("the time, such as 18:30"), "title": _s("what the event is, such as Dinner with Sam")}),
                 _mcp("delete_event", "Delete an event from the user's calendar.", {"event_id": _s("the id of the event, from list_events")})],
    "notes": [_mcp("search_notes", "Find the user's notes that contain a word.", {"query": _s("the word to look for")}),
              _mcp("read_note", "The whole text of one note.", {"note_id": _s("the id of the note, from search_notes")}),
              _mcp("write_note", "Save a new note.", {"title": _s("the title of the note"), "text": _s("the text of the note")})],
    "shop": [_mcp("search_products", "Find products in the shop by name.", {"query": _s("a word in the product's name, such as milk")}),
             _mcp("add_to_cart", "Put a product in the cart.", {"product": _s("the product's name"),
                  "count": {"type": "integer", "description": "how many, such as 2"}}),
             _mcp("show_cart", "What is in the cart now.", {}),
             _mcp("check_out", "Pay for the cart.", {})],
    "weather": [_mcp("forecast", "The weather in a city for the next days.", {"city": _s("the name of the city"),
                     "days": {"type": "integer", "description": "how many days, from 1 to 7"}})],
}
SERVERS = ["kitchen", "calendar", "units", "notes", "shop", "weather"]   # the order in which the client lists them


def all_listed() -> list:
    """Every tool of the six servers, server by server, in MCP's form."""
    return [t for s in SERVERS for t in (SERVER_TOOLS if s == "units" else OTHER[s])]


MANY = as_tools(all_listed())


def tool_results(message: dict, ask: dict | None = None) -> list:
    """For the recorder: convert runs as in build-tool-calling; the other servers' tools are only descriptions in these
    runs, so a call of one of them gets that as an error."""
    out = []
    for c in message.get("tool_calls") or []:
        try:
            arguments = json.loads(c["function"]["arguments"])
        except ValueError:
            arguments = None
        name = c["function"]["name"]
        result = (("Error: the arguments are not a JSON object" if not isinstance(arguments, dict) else FUNCTIONS[name](arguments))
                  if name in FUNCTIONS else
                  "Error: the course's runs have only the tool convert" if any(t["function"]["name"] == name for t in MANY) else
                  "Error: no tool " + name)
        out.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return out


RECORD = {}
for _i, (_q, *_) in enumerate(QUESTIONS):
    RECORD[f"many-{_i}"] = {"messages": T.asked(_q), "tools": MANY, "max_tokens": 120, "temperature": 0, "steps": 2, "seeds": 1}


# --- exercise 1: the server's answers -------------------------------------------------------------------------------------

START_SERVER = """VERSION = "2026-07-28"
INFO = {"io.modelcontextprotocol/serverInfo": {"name": "units", "version": "1.0"}}


def result(message, content):
    content = dict(content)
    content["resultType"] = "complete"
    content["_meta"] = INFO
    return {"jsonrpc": "2.0", "id": message["id"], "result": content}


def error(message, code, text, data=None):
    problem = {"code": code, "message": text}
    if data is not None:
        problem["data"] = data
    return {"jsonrpc": "2.0", "id": message["id"], "error": problem}


def handle(message):
    if "id" not in message:   # a notification gets no answer
        return None
    params = message.get("params") or {}
    meta = params.get("_meta") or {}
    asked = meta.get("io.modelcontextprotocol/protocolVersion")
    if asked != VERSION:
        return error(message, -32022, "Unsupported protocol version",
                     {"supported": [VERSION], "requested": asked})
    if "io.modelcontextprotocol/clientCapabilities" not in meta:
        return error(message, -32602, "The _meta of the request has no clientCapabilities")
    if message["method"] == "server/discover":
        return result(message, {"supportedVersions": [VERSION], "capabilities": {"tools": {}},
                                "ttlMs": 300000, "cacheScope": "public"})
    if message["method"] == "tools/list":
        ...
    if message["method"] == "tools/call":
        ...
    return error(message, -32601, "Method not found: " + str(message["method"]))
"""
SOLUTION_SERVER = START_SERVER.replace("""    if message["method"] == "tools/list":
        ...
    if message["method"] == "tools/call":
        ...
""", """    if message["method"] == "tools/list":
        return result(message, {"tools": TOOLS, "ttlMs": 300000, "cacheScope": "public"})
    if message["method"] == "tools/call":
        name = params.get("name")
        if name not in FUNCTIONS:
            return error(message, -32602, "Unknown tool: " + str(name))
        text = FUNCTIONS[name](params.get("arguments") or {})
        content = [{"type": "text", "text": text}]
        return result(message, {"content": content, "isError": text.startswith("Error")})
""")


def server_file(code: str) -> str:
    """A whole MCP server as a Python file: the tool, the box's code, and the loop that reads and writes the lines."""
    return SERVER_HEAD + "\n\n" + code.strip() + "\n" + SERVER_TAIL


COURSE_HANDLE = run_code(SOLUTION_SERVER, dict(SERVER)).names["handle"]


def same_answer(got, want) -> bool:
    """Whether a server's answer says what the course's says: the same id; for an error, the same code; for a result,
    resultType "complete", the same tools with caching hints, or the same content and isError (absent counts as false);
    other fields, such as _meta, may differ."""
    if want is None or got is None:
        return got is None and want is None
    if not isinstance(got, dict) or got.get("jsonrpc") != "2.0" or got.get("id") != want.get("id"):
        return False
    if "error" in want:
        return isinstance(got.get("error"), dict) and got["error"].get("code") == want["error"]["code"] and "result" not in got
    r, w = got.get("result"), want["result"]
    if not isinstance(r, dict) or r.get("resultType") != "complete" or "error" in got:
        return False
    if "tools" in w:
        return (r.get("tools") == w["tools"] and isinstance(r.get("ttlMs"), int) and r["ttlMs"] >= 0
                and r.get("cacheScope") in ("public", "private"))
    if "content" in w:
        return r.get("content") == w["content"] and bool(r.get("isError")) == bool(w.get("isError"))
    return all(r.get(k) == v for k, v in w.items() if k != "_meta")


def request(number, method: str, params: dict | None = None, meta: dict | None = META) -> dict:
    """A JSON-RPC request of a client in MCP 2026-07-28; without `meta` when it is None."""
    params = dict(params or {})
    if meta is not None:
        params["_meta"] = meta
    return {"jsonrpc": "2.0", "id": number, "method": method, "params": params}


class Message(dict):
    """A message of a check, which its message names by a short name instead of writing it all out."""

    def __init__(self, name: str, content: dict):
        super().__init__(content)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return Message(self.name, copy.deepcopy(dict(self), memo))


ARGS_37 = {"amount": 37, "from_unit": "km", "to_unit": "mi"}
M_LIST = Message("list_request", request(1, "tools/list"))
M_CALL = Message("call_request", request(2, "tools/call", {"name": "convert", "arguments": ARGS_37}))
M_BAD = Message("call_km_to_kg", request("call-3", "tools/call", {"name": "convert",
                                                                   "arguments": {"amount": 37, "from_unit": "km", "to_unit": "kg"}}))
M_UNKNOWN = Message("call_weather", request(4, "tools/call", {"name": "weather", "arguments": {"city": "Utrecht"}}))
M_NO_ARGS = Message("call_without_arguments", request(5, "tools/call", {"name": "convert"}))
M_DISCOVER = Message("discover_request", request(6, "server/discover"))
M_NOTE = Message("cancelled_notification", {"jsonrpc": "2.0", "method": "notifications/cancelled",
                                            "params": {"requestId": 2, "_meta": META}})
M_OLD = Message("old_initialize", {"jsonrpc": "2.0", "id": 7, "method": "initialize",
                                   "params": {"protocolVersion": "2025-11-25", "capabilities": {}}})
M_OTHER = Message("resources_request", request(8, "resources/list"))
SERVER_CASES = [(M_CALL,), (M_BAD,), (M_UNKNOWN,), (M_NO_ARGS,), (M_DISCOVER,), (M_NOTE,), (M_OLD,), (M_OTHER,)]

_NO_CACHE = SOLUTION_SERVER.replace('{"tools": TOOLS, "ttlMs": 300000, "cacheScope": "public"}', '{"tools": TOOLS}')
_UNKNOWN_RESULT = SOLUTION_SERVER.replace('''            return error(message, -32602, "Unknown tool: " + str(name))''',
                                          '''            text = "Unknown tool: " + str(name)
            return result(message, {"content": [{"type": "text", "text": text}], "isError": True})''')
_NO_IS_ERROR = SOLUTION_SERVER.replace(', "isError": text.startswith("Error")', "")
_ARGS_KEY = SOLUTION_SERVER.replace('params.get("arguments") or {}', 'params["arguments"]')


def _no_cache(got, text) -> bool:
    r = got.get("result") if isinstance(got, dict) else None
    return isinstance(r, dict) and "tools" in r and ("ttlMs" not in r or "cacheScope" not in r)


def _content_text(got, text) -> bool:
    r = got.get("result") if isinstance(got, dict) else None
    return isinstance(r, dict) and isinstance(r.get("content"), str)


# --- exercise 2: a client's tools/call ------------------------------------------------------------------------------------

START_CALL = '''META = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}}


def call_tool(send, number, name, arguments):
    request = ...
    answer = send(request)
    ...
'''
SOLUTION_CALL = START_CALL.replace('''    request = ...
    answer = send(request)
    ...
''', '''    request = {"jsonrpc": "2.0", "id": number, "method": "tools/call",
               "params": {"name": name, "arguments": arguments, "_meta": META}}
    answer = send(request)
    if "error" in answer:
        return "Error: " + answer["error"]["message"]
    texts = [part["text"] for part in answer["result"]["content"] if part["type"] == "text"]
    return "\\n".join(texts)
''')

PHOTO_TOOLS = [{"name": "describe", "description": "Describe one of the course's photos.",
                "inputSchema": {"type": "object", "properties": {"photo": {"type": "string"}}, "required": ["photo"]}}]


def _describe(arguments: dict) -> list:
    return [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
            {"type": "text", "text": "A cup of espresso on a red saucer."}, {"type": "text", "text": "It stands on a wooden table."}]


def photo_handle(message: dict):
    """A second server of the check, whose tool gives an image and two pieces of text."""
    if message.get("method") == "tools/call" and (message.get("params") or {}).get("name") == "describe":
        meta = (message.get("params") or {}).get("_meta") or {}
        if meta.get("io.modelcontextprotocol/protocolVersion") == VERSION and "io.modelcontextprotocol/clientCapabilities" in meta:
            return {"jsonrpc": "2.0", "id": message.get("id"),
                    "result": {"resultType": "complete", "content": _describe(message["params"].get("arguments") or {})}}
    return run_code(SOLUTION_SERVER, {"TOOLS": PHOTO_TOOLS, "FUNCTIONS": {}}).names["handle"](message)


class Send:
    """The check's `send`: it writes the request as one line to a server and reads its answer line, as a client does
    over stdio; a request that is not one JSON-RPC message gets the error a server gives it."""

    def __init__(self, name: str = "send", server: str = "units"):
        self.name, self.server, self.sent = name, server, []

    def __repr__(self):
        return self.name

    def __call__(self, message):
        line = json.dumps(message)   # what goes over the line: text, so no Python objects
        self.sent.append(line)
        message = json.loads(line)
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return {"jsonrpc": "2.0", "id": message.get("id") if isinstance(message, dict) else None,
                    "error": {"code": -32600, "message": "Invalid Request"}}
        answer = (photo_handle if self.server == "photo" else COURSE_HANDLE)(message)
        return None if answer is None else json.loads(json.dumps(answer))


CALL_VISIBLE = [(Send(), 1, "convert", dict(ARGS_37))]
CALL_HIDDEN = [(Send(), 2, "convert", {"amount": 37, "from_unit": "km", "to_unit": "kg"}),
               (Send(), 3, "weather", {"city": "Utrecht"}),
               (Send("send_photos", "photo"), 4, "describe", {"photo": "coffee"})]

_FIRST_PART = SOLUTION_CALL.replace('''    texts = [part["text"] for part in answer["result"]["content"] if part["type"] == "text"]
    return "\\n".join(texts)''', '''    return answer["result"]["content"][0].get("text", "")''')


def _no_id(e) -> bool:
    return isinstance(e, TypeError) and "NoneType" in str(e)


def _result_key(e) -> bool:
    return isinstance(e, KeyError) and e.args == ("result",)


_NO_META = SOLUTION_CALL.replace(', "_meta": META}}', "}}")
_NO_JSONRPC = SOLUTION_CALL.replace('request = {"jsonrpc": "2.0", "id": number,', 'request = {"id": number,')
_NO_ERROR_CHECK = SOLUTION_CALL.replace('''    if "error" in answer:
        return "Error: " + answer["error"]["message"]
''', "")


# --- comparing and showing the answers -----------------------------------------------------------------------------------

def _answer_shown(v) -> str:
    """An answer of a server in one line, without its _meta, and with a list of tools shortened to their names."""
    if not isinstance(v, dict):
        return "None" if v is None else repr(v)
    r = v.get("result")
    if isinstance(r, dict):
        r = {k: x for k, x in r.items() if k != "_meta"}
        if isinstance(r.get("tools"), list):
            r["tools"] = ["<" + str(t.get("name") if isinstance(t, dict) else t) + ">" for t in r["tools"]]
        v = {**v, "result": r}
    return json.dumps(v, ensure_ascii=False, default=repr)


def _missing_key(key: str):
    return lambda e: isinstance(e, KeyError) and e.args == (key,)


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the pictures of the exercises --------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui)"
_CELL = "padding:0.18rem 0.5rem;border-bottom:1px solid var(--ea-line);vertical-align:top;font-size:13px"
_HEAD = f"{_CELL};color:var(--ea-ink-2);font-weight:500;text-align:left;font-family:var(--ea-font-ui)"


def _code(text: str, color: str = "var(--ea-ink)") -> str:
    return (f'<span style="font-family:var(--ea-font-code);font-size:12px;color:{color};white-space:pre-wrap;'
            f'overflow-wrap:anywhere">{html.escape(text)}</span>')


def _label(text: str) -> str:
    return f'<div style="{_UI};font-size:12px;color:var(--ea-ink-2);margin:0.1rem 0 0.2rem">{html.escape(text)}</div>'


def _line(direction: str, text: str, color: str) -> str:
    """One line over stdio: who sends it, and the JSON."""
    return (f'<div style="display:grid;grid-template-columns:7rem 1fr;gap:0.4rem;margin:0.15rem 0;'
            f'border-left:3px solid {color};padding-left:0.4rem"><div style="{_UI};font-size:11px;color:var(--ea-ink-2)">'
            f'{html.escape(direction)}</div><div style="min-width:0">{_code(text, color)}</div></div>')


_SERVER_WORDS = {
    "en": {"title": "Lines over stdio: the client's requests, and your server's answers", "client": "client → server",
           "server": "server → client", "course": "the course's server", "none": "(no answer)",
           "error": "Your handle stops with {e}."},
    "nl": {"title": "Regels over stdio: de requests van de client, en de antwoorden van je server", "client": "client → server",
           "server": "server → client", "course": "server van de course", "none": "(geen antwoord)",
           "error": "Je handle stopt met {e}."},
}


def server_picture(seen) -> str:
    """The list request and two calls of the check, as the lines a client writes, with the answer lines of the
    learner's handle, and the course's where they differ."""
    run_ = seen.run
    fn = run_.names.get("handle") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _SERVER_WORDS[seen.lang]
    rows = []
    for m in (M_LIST, M_CALL, M_BAD):
        want = COURSE_HANDLE(copy.deepcopy(dict(m)))
        rows.append(_line(w["client"], json.dumps(dict(m)), "var(--ea-ink)"))
        try:
            got = fn(copy.deepcopy(dict(m)))
        except Exception as e:   # noqa: BLE001 - the learner's code
            rows.append(_line(w["server"], w["error"].format(e=f"{type(e).__name__}: {e}"), "var(--ea-loss-text)"))
            continue
        ok = same_answer(got, want)
        rows.append(_line(w["server"], w["none"] if got is None else _answer_shown(got),
                          "var(--ea-data-text)" if ok else "var(--ea-loss-text)"))
        if not ok:
            rows.append(_line(w["course"], _answer_shown(want), "var(--ea-data-text)"))
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


_CALL_WORDS = {
    "en": {"title": "What your call_tool sends, what comes back, and what it gives the model", "sent": "your request",
           "answer": "the answer", "gives": "call_tool gives", "course": "the course's gives",
           "nothing": "(your call_tool sent nothing)", "error": "Your call_tool stops with {e}."},
    "nl": {"title": "Wat je call_tool verstuurt, wat er terugkomt, en wat het aan het model geeft", "sent": "je request",
           "answer": "het antwoord", "gives": "call_tool geeft", "course": "die van de course geeft",
           "nothing": "(je call_tool verstuurde niets)", "error": "Je call_tool stopt met {e}."},
}


def call_picture(seen) -> str:
    """The request the learner's call_tool sends for 37 km in miles, the server's answer, and the text it gives back."""
    run_ = seen.run
    fn = run_.names.get("call_tool") if run_ is not None and run_.error is None else None
    if not callable(fn):
        return ""
    w = _CALL_WORDS[seen.lang]
    box = EXERCISES["call"]
    want = box._run(box._fn(SOLUTION_CALL), CALL_VISIBLE[0])
    send = Send()
    failed = None
    try:
        got = fn(send, 1, "convert", dict(ARGS_37))
    except Exception as e:   # noqa: BLE001 - the learner's code
        got, failed = None, f"{type(e).__name__}: {e}"
    rows = []
    if send.sent:
        line = send.sent[0]
        rows.append(_line(w["sent"], line, "var(--ea-ink)"))
        rows.append(_line(w["answer"], _answer_shown(Send()(json.loads(line))), "var(--ea-data-text)"))
    else:
        rows.append(f'<p style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(w["nothing"])}</p>')
    if failed:
        rows.append(_line(w["gives"], w["error"].format(e=failed), "var(--ea-loss-text)"))
    else:
        ok = got == want
        rows.append(_line(w["gives"], repr(got), "var(--ea-data-text)" if ok else "var(--ea-loss-text)"))
        if not ok:
            rows.append(_line(w["course"], repr(want), "var(--ea-data-text)"))
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- the exercises ------------------------------------------------------------------------------------------------------

_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}


class NamedBox(FunctionBox):
    """A FunctionBox whose messages name each case by its short name, such as `handle(list_request)`."""

    def _call_text(self, case) -> str:
        return f"{self.function}({', '.join(repr(a) for a in case)})"


EXERCISES = {
    "server": NamedBox(
        name="server",
        picture=server_picture,
        function="handle",
        start=START_SERVER,
        solution=SOLUTION_SERVER,
        names=SERVER,
        visible=[(M_LIST,)],
        hidden=SERVER_CASES,
        compare=same_answer,
        show=_answer_shown,
        mistakes=[
            Mistake("args-key", error=_missing_key("arguments"),
                    en="With `{call}`, the request has no `\"arguments\"`, and your server stops with `KeyError`. A request may leave them out: `params.get(\"arguments\") or {{}}` gives an empty dictionary then, and the tool says what is missing.",
                    nl="Met `{call}` heeft het request geen `\"arguments\"`, en je server stopt met `KeyError`. Een request mag ze weglaten: `params.get(\"arguments\") or {{}}` geeft dan een lege dictionary, en de tool zegt wat er ontbreekt."),
            Mistake("no-cache", matches=_no_cache,
                    en="With `{call}`, the answer has the tools, but not how long a client may keep them: in MCP 2026-07-28, the result of tools/list has `\"ttlMs\"` (milliseconds) and `\"cacheScope\"`, as the answer to server/discover in the box has.",
                    nl="Met `{call}` heeft het antwoord de tools, maar niet hoelang een client ze mag bewaren: in MCP 2026-07-28 heeft het result van tools/list `\"ttlMs\"` (milliseconden) en `\"cacheScope\"`, zoals het antwoord op server/discover in het vak."),
            Mistake("content-text", matches=_content_text,
                    en="With `{call}`, `\"content\"` is the text itself. In MCP it is a list of parts, each with its type: `[{{\"type\": \"text\", \"text\": text}}]`, so that a tool can also give back an image.",
                    nl="Met `{call}` is `\"content\"` de tekst zelf. In MCP is het een lijst met delen, elk met zijn type: `[{{\"type\": \"text\", \"text\": text}}]`, zodat een tool ook een afbeelding kan teruggeven."),
            Mistake("unknown-result", wrong=_UNKNOWN_RESULT,
                    en="With `{call}`, your server answers a call of a tool it does not have with a result. That is an error in the request itself: answer with `error(message, -32602, ...)`. A result with `\"isError\": True` is for a tool that ran and failed.",
                    nl="Met `{call}` antwoordt je server op een call van een tool die hij niet heeft met een result. Dat is een fout in het request zelf: antwoord met `error(message, -32602, ...)`. Een result met `\"isError\": True` is voor een tool die draaide en misging."),
            Mistake("no-is-error", wrong=_NO_IS_ERROR,
                    en="With `{call}`, convert gives back an error, and your answer does not say so. Add `\"isError\": True` when the text starts with `Error`, so that the client can tell it apart from a result.",
                    nl="Met `{call}` geeft convert een fout terug, en je antwoord zegt dat niet. Zet `\"isError\": True` erbij als de tekst begint met `Error`, dan kan de client het onderscheiden van een uitkomst."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your server does not answer tools/list and tools/call yet: they reach the last line, Method not found. Fill in the two `...`: the list of tools, and a call of a tool.",
                      "nl": "Je server antwoordt nog niet op tools/list en tools/call: die komen bij de laatste regel, Method not found. Vul de twee `...` in: de lijst met tools, en een call van een tool."},
            "wrong": {"en": "With `{call}`, your server answers `{got}`, and it should be `{want}`. Compare the lines in the picture.",
                      "nl": "Met `{call}` antwoordt je server `{got}`, en het zou `{want}` moeten zijn. Vergelijk de regels in het plaatje."},
            "almost": {"en": "Right for the list of tools. But with `{call}`, your server answers `{got}`, and it should be `{want}`.",
                       "nl": "Goed voor de lijst met tools. Maar met `{call}` antwoordt je server `{got}`, en het zou `{want}` moeten zijn."},
            "correct": {"en": "Your `handle` is an MCP server: it lists its tool with its input schema, runs a call and gives back the text as content, says when the tool failed, and answers a request it cannot follow with an error. Right for all {n} cases of the check.",
                        "nl": "Je `handle` is een MCP server: hij geeft zijn tool met het input schema, voert een call uit en geeft de tekst terug als content, zegt wanneer de tool misging, en antwoordt op een request dat hij niet kan volgen met een fout. Goed bij alle {n} gevallen van de check."},
        },
    ),
    "call": NamedBox(
        name="call",
        picture=call_picture,
        function="call_tool",
        start=START_CALL,
        solution=SOLUTION_CALL,
        visible=CALL_VISIBLE,
        hidden=CALL_HIDDEN,
        mistakes=[
            Mistake("no-id", error=_no_id,
                    en="With `{call}`, your request has no `\"id\"`, so the server takes it for a notification and gives no answer: `send` gives back `None`. Give the request `\"id\": number`.",
                    nl="Met `{call}` heeft je request geen `\"id\"`, dus de server houdt het voor een notification en geeft geen antwoord: `send` geeft `None` terug. Geef het request `\"id\": number`."),
            Mistake("result-key", error=_missing_key("result"),
                    en="With `{call}`, the server answers with an error, and an error has no `\"result\"`. Look first whether `\"error\"` is in the answer, and give back `\"Error: \"` and its message then.",
                    nl="Met `{call}` antwoordt de server met een fout, en een fout heeft geen `\"result\"`. Kijk eerst of `\"error\"` in het antwoord zit, en geef dan `\"Error: \"` en de message ervan terug."),
            Mistake("no-jsonrpc", wrong=_NO_JSONRPC,
                    en="With `{call}`, the server answers Invalid Request: every JSON-RPC message starts with `\"jsonrpc\": \"2.0\"`.",
                    nl="Met `{call}` antwoordt de server Invalid Request: elk JSON-RPC-bericht begint met `\"jsonrpc\": \"2.0\"`."),
            Mistake("no-meta", wrong=_NO_META,
                    en="With `{call}`, the server answers Unsupported protocol version: in MCP 2026-07-28, every request says in `\"_meta\"` which version the client speaks. Put `\"_meta\": META` in its params.",
                    nl="Met `{call}` antwoordt de server Unsupported protocol version: in MCP 2026-07-28 zegt elk request in `\"_meta\"` welke versie de client spreekt. Zet `\"_meta\": META` in de params."),
            Mistake("first-part", wrong=_FIRST_PART,
                    en="With `{call}`, your function gives back {got}, and it should be {want}. The content is a list of parts: here an image and two pieces of text. Join the text of every part whose type is `\"text\"`.",
                    nl="Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn. De content is een lijst met delen: hier een afbeelding en twee stukken tekst. Voeg de tekst van elk deel met type `\"text\"` samen."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "Your function sends nothing yet: `request` is `...`. Build the JSON-RPC request for tools/call, with the tool's name and arguments and `META` in its params.",
                      "nl": "Je functie verstuurt nog niets: `request` is `...`. Maak het JSON-RPC-request voor tools/call, met de naam en de arguments van de tool en `META` in de params."},
            "wrong": {"en": "With `{call}`, your function gives back {got}, and it should be {want}. The picture shows what your function sent and what came back.",
                      "nl": "Met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn. Het plaatje laat zien wat je functie verstuurde en wat er terugkwam."},
            "almost": {"en": "Right for 37 km in miles. But with `{call}`, your function gives back {got}, and it should be {want}.",
                       "nl": "Goed voor 37 km in mijlen. Maar met `{call}` geeft je functie {got} terug, en het zou {want} moeten zijn."},
            "correct": {"en": "Your `call_tool` is the client's half: it sends a tools/call request with the protocol version in `_meta`, and turns the answer into the text the model reads, also when it is an error or holds more than text. Right for all {n} cases of the check.",
                        "nl": "Je `call_tool` is de helft van de client: hij verstuurt een tools/call-request met de protocolversie in `_meta`, en maakt van het antwoord de tekst die het model leest, ook als het een fout is of meer dan tekst bevat. Goed bij alle {n} gevallen van de check."},
        },
    ),
}

REQUIRED = ["server", "call", "trust-server"]

CASES = [
    ("server", START_SERVER, "empty", None),
    ("server", "", "empty", None),
    ("server", SOLUTION_SERVER, "correct", None),
    ("server", _ARGS_KEY, "error", "args-key"),
    ("server", _NO_CACHE, "wrong", "no-cache"),
    ("server", SOLUTION_SERVER.replace('content = [{"type": "text", "text": text}]', "content = text"), "wrong", "content-text"),
    ("server", _UNKNOWN_RESULT, "wrong", "unknown-result"),
    ("server", _NO_IS_ERROR, "wrong", "no-is-error"),
    ("server", SOLUTION_SERVER.replace('"isError": text.startswith("Error")', '"isError": "Error" in text'), "correct", None),
    ("call", START_CALL, "empty", None),
    ("call", "", "empty", None),
    ("call", SOLUTION_CALL, "correct", None),
    ("call", SOLUTION_CALL.replace('request = {"jsonrpc": "2.0", "id": number,', 'request = {"jsonrpc": "2.0",'), "error", "no-id"),
    ("call", _NO_ERROR_CHECK, "error", "result-key"),
    ("call", _NO_JSONRPC, "wrong", "no-jsonrpc"),
    ("call", _NO_META, "wrong", "no-meta"),
    ("call", _FIRST_PART, "wrong", "first-part"),
    ("call", SOLUTION_CALL.replace('''    texts = [part["text"] for part in answer["result"]["content"] if part["type"] == "text"]
    return "\\n".join(texts)''', '''    texts = []
    for part in answer["result"]["content"]:
        if part.get("type") == "text":
            texts.append(part["text"])
    return "\\n".join(texts)'''), "correct", None),
]


# --- what the recorded runs show ----------------------------------------------------------------------------------------

def course() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))["parts"].get("build-mcp", {})


def many_steps(i: int) -> list:
    """The run of question i with the tools of the six servers."""
    return course()[f"many-{i}"]["answers"][0]["steps"]


def one_steps(i: int) -> list:
    """The run of question i in build-tool-calling, with convert alone and tool_choice "auto"."""
    return T.run(f"auto-{i}")["steps"]


def _first_call(steps: list):
    calls = steps[0]["tool_calls"]
    return calls[0] if calls else None


def _right_call(call, i: int) -> bool:
    try:
        return call is not None and T.right_call(call, i)
    except ValueError:   # arguments that are not JSON
        return False


def findings() -> dict:
    """With convert alone and with the fourteen tools of six servers: the prompt tokens of the first request, the
    questions in which the model called convert, with the right arguments, and answered right, and its calls of other
    tools."""
    n = len(QUESTIONS)
    out = {"questions": n, "servers": len(SERVERS), "tools": len(MANY), "same_tool": as_tools(SERVER_TOOLS) == T.TOOLS}
    for kind, steps_of in (("one", one_steps), ("many", many_steps)):
        runs = [steps_of(i) for i in range(n)]
        firsts = [_first_call(s) for s in runs]
        out[kind + "_prompt"] = runs[0][0]["prompt_tokens"]
        out[kind + "_called"] = [i for i, c in enumerate(firsts) if c is not None and c["function"]["name"] == "convert"]
        out[kind + "_args"] = [i for i, c in enumerate(firsts) if _right_call(c, i)]
        out[kind + "_right"] = [i for i, s in enumerate(runs) if T.right(s[-1]["text"], i)]
        out[kind + "_other"] = [i for i, c in enumerate(firsts) if c is not None and c["function"]["name"] != "convert"]
        out[kind + "_seconds"] = sum(s[0]["seconds"] for s in runs) / n
    out["other_names"] = sorted({_first_call(many_steps(i))["function"]["name"] for i in out["many_other"]})
    grows = {many_steps(i)[0]["prompt_tokens"] - one_steps(i)[0]["prompt_tokens"] for i in range(n)}
    out["grow"] = grows.pop() if len(grows) == 1 else None   # the same for every question
    return out


# --- the pictures of the idea and of the measurement ------------------------------------------------------------------

_IDEA_WORDS = {
    "en": {"title": "Who says what to whom, for one question", "model": "model", "client": "client", "server": "MCP server",
           "subs": ["", "in the assistant", "a program of its own"],
           "steps": ["the client starts the server as a program, and asks for its tools: tools/list",
                     "the server answers with each tool: name, description and inputSchema",
                     "the client gives the tools to the model, with the question, in the chat API's form",
                     "the model calls convert",
                     "the client sends the call to the server: tools/call",
                     "the server runs convert, and answers with the text as content",
                     "the client gives the text to the model as the tool's result; the model answers"],
           "alt": "Three columns: the model, the client in the assistant, and the MCP server. Seven numbered steps go back and forth: the client asks the server for its tools, gives them to the model, sends the model's call to the server, and gives the server's answer back to the model."},
    "nl": {"title": "Wie zegt wat tegen wie, bij één vraag", "model": "model", "client": "client", "server": "MCP server",
           "subs": ["", "in de assistent", "een eigen programma"],
           "steps": ["de client start de server als programma, en vraagt om zijn tools: tools/list",
                     "de server antwoordt met elke tool: name, description en inputSchema",
                     "de client geeft de tools aan het model, met de vraag, in de vorm van de chat-API",
                     "het model roept convert aan",
                     "de client stuurt de call naar de server: tools/call",
                     "de server voert convert uit, en antwoordt met de tekst als content",
                     "de client geeft de tekst aan het model als resultaat van de tool; het model antwoordt"],
           "alt": "Drie kolommen: het model, de client in de assistent, en de MCP server. Zeven genummerde stappen gaan heen en weer: de client vraagt de server om zijn tools, geeft ze aan het model, stuurt de call van het model naar de server, en geeft het antwoord van de server terug aan het model."},
}
_IDEA_MOVES = [(1, 2), (2, 1), (1, 0), (0, 1), (1, 2), (2, 1), (1, 0)]   # (from column, to column) of each step


def idea_picture(lang: str) -> str:
    """The seven steps of one question between the model, the client and the server: numbered arrows between three
    columns, and each step written out under them."""
    w = _IDEA_WORDS[lang]
    width, col = 600, [70, 300, 530]
    top, step = 58, 30
    height = top + step * len(_IDEA_MOVES) + 6
    colors = {0: "var(--ea-pred)", 1: "var(--ea-ink-2)", 2: "var(--ea-data)"}
    text_colors = {0: "var(--ea-pred-text)", 1: "var(--ea-ink-2)", 2: "var(--ea-data-text)"}
    parts = ['<defs><marker id="mcp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--ea-ink-2)"/></marker></defs>']
    for k, (x, name, sub) in enumerate(zip(col, (w["model"], w["client"], w["server"]), w["subs"])):
        parts.append(f'<text x="{x}" y="16" text-anchor="middle" font-size="13" font-weight="600" fill="{text_colors[k]}" '
                     f'style="{_UI}">{html.escape(name)}</text>')
        if sub:
            parts.append(f'<text x="{x}" y="32" text-anchor="middle" font-size="11" fill="var(--ea-ink-2)" style="{_UI}">'
                         f'{html.escape(sub)}</text>')
        parts.append(f'<line x1="{x}" y1="40" x2="{x}" y2="{height - 2}" stroke="{colors[k]}" stroke-width="2" opacity="0.5"/>')
    for k, (a, b) in enumerate(_IDEA_MOVES):
        y = top + step * k + 8
        x1, x2 = col[a] + (6 if b > a else -6), col[b] + (-6 if b > a else 6)
        parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="var(--ea-ink-2)" stroke-width="1.5" marker-end="url(#mcp-arrow)"/>')
        mid = (col[a] + col[b]) / 2
        parts.append(f'<text x="{mid}" y="{y - 5}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--ea-ink)" '
                     f'style="{_UI}">{k + 1}</text>')
    svg = (f'<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto" role="img" '
           f'aria-label="{html.escape(w["alt"])}">' + "".join(parts) + "</svg>")
    steps = "".join(f'<div style="display:flex;gap:0.5rem;margin:0.1rem 0"><span style="font-weight:600;min-width:1rem">{k}</span>'
                    f'<span>{html.escape(text)}</span></div>' for k, text in enumerate(w["steps"], 1))
    body = svg + f'<div style="{_UI};font-size:13px;color:var(--ea-ink);margin-top:0.4rem">{steps}</div>'
    return yours.figure(w["title"], body, alt=w["alt"])


_TABLE_WORDS = {
    "en": {"title": "Eight questions about units, with convert alone and with the tools of six servers",
           "what": "", "one": "convert alone", "many": "{t} tools of {s} servers", "prompt": "prompt tokens per request",
           "called": "called convert", "args": "with the right arguments", "right": "right answer",
           "other": "called another tool first", "seconds": "seconds for the first answer", "of": "{k} of {n}"},
    "nl": {"title": "Acht vragen over eenheden, met alleen convert en met de tools van zes servers",
           "what": "", "one": "alleen convert", "many": "{t} tools van {s} servers", "prompt": "tokens aan prompt per request",
           "called": "riep convert aan", "args": "met de goede argumenten", "right": "goed antwoord",
           "other": "riep eerst een andere tool aan", "seconds": "seconden voor het eerste antwoord", "of": "{k} van de {n}"},
}


def table_picture(lang: str) -> str:
    w, f = _TABLE_WORDS[lang], findings()
    n = f["questions"]
    rows = [("prompt", number(f["one_prompt"], lang), number(f["many_prompt"], lang)),
            ("called", w["of"].format(k=len(f["one_called"]), n=n), w["of"].format(k=len(f["many_called"]), n=n)),
            ("args", w["of"].format(k=len(f["one_args"]), n=n), w["of"].format(k=len(f["many_args"]), n=n)),
            ("right", w["of"].format(k=len(f["one_right"]), n=n), w["of"].format(k=len(f["many_right"]), n=n)),
            ("other", w["of"].format(k=len(f["one_other"]), n=n), w["of"].format(k=len(f["many_other"]), n=n)),
            ("seconds", number(f["one_seconds"], lang, 1), number(f["many_seconds"], lang, 1))]
    head = (f'<tr><th style="{_HEAD}"></th><th style="{_HEAD}">{w["one"]}</th>'
            f'<th style="{_HEAD}">{w["many"].format(t=f["tools"], s=f["servers"])}</th></tr>')
    body = "".join(f'<tr><td style="{_CELL};{_UI};color:var(--ea-ink-2)">{w[k]}</td><td style="{_CELL};{_UI}">{a}</td>'
                   f'<td style="{_CELL};{_UI}">{b}</td></tr>' for k, a, b in rows)
    table = (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%"><thead>{head}</thead>'
             f'<tbody>{body}</tbody></table></div>')
    return yours.figure(w["title"], table, alt=w["title"])


_WORDS = {"en": {1: "once", 2: "twice"}, "nl": {1: "één keer", 2: "twee keer"}}


def _times(k: int, lang: str) -> str:
    return _WORDS[lang].get(k, f"{k} times" if lang == "en" else f"{k} keer")


def measured_text(lang: str) -> str:
    """What the runs show, in the lesson's language, with the measured numbers."""
    f = findings()
    n = f["questions"]
    alone = [i for i in f["many_right"] if i not in f["many_called"]]
    if lang == "nl":
        return ("Met alleen convert riep Qwen3 hem aan bij {a} van de {n} vragen, elke keer met de goede argumenten, en die {a} "
                "antwoorden waren goed. Met de {t} tools van {s} servers riep het convert maar bij {b} van de {n} aan; de andere "
                "{rest} keer antwoordde het zelf, en {k} van die antwoorden hadden toch het goede getal. Een tool van een andere "
                "server riep het nooit aan. Elk request groeide ook met {g} tokens aan prompt, van {p1} naar {p2} bij de eerste vraag, omdat de beschrijving van elke "
                "tool in elk request gaat. Een client met veel servers geeft een klein model meer om uit te kiezen en meer om te lezen, "
                "en hier koos het vaker voor geen tool. Geef een model dus alleen de tools die een taak nodig heeft.").format(
                    a=len(f["one_called"]), n=n, t=f["tools"], s=f["servers"], b=len(f["many_called"]),
                    rest=n - len(f["many_called"]), k=len(alone), p1=number(f["one_prompt"], lang), p2=number(f["many_prompt"], lang),
                    g=number(f["grow"], lang))
    return ("With convert alone, Qwen3 called it in {a} of the {n} questions, each time with the right arguments, and those {a} "
            "answers were right. With the {t} tools of {s} servers, it called convert in only {b} of the {n}; the other {rest} "
            "times it answered by itself, and {k} of those answers still had the right number. It never called a tool of "
            "another server. Each request also grew by {g} prompt tokens, from {p1} to {p2} for the first question, since every tool's description goes into "
            "every request. A client with many servers gives a small model more to choose from and more to read, and here it "
            "chose no tool more often. So give a model only the tools a task needs.").format(
                a=len(f["one_called"]), n=n, t=f["tools"], s=f["servers"], b=len(f["many_called"]),
                rest=n - len(f["many_called"]), k=len(alone), p1=number(f["one_prompt"], lang), p2=number(f["many_prompt"], lang),
                g=number(f["grow"], lang))


def worked_question() -> int:
    """The question of the worked example: the first one in which Qwen3 called convert among the fourteen tools."""
    return findings()["many_called"][0]


def worked_text(lang: str) -> str:
    i = worked_question()
    call = _first_call(many_steps(i))
    lines = worked_lines(i)
    text = "\n".join(p["text"] for p in lines[1][1]["result"]["content"] if p["type"] == "text")
    answer = many_steps(i)[-1]["text"].strip()
    if lang == "nl":
        return ("### Uitgewerkt voorbeeld\n\nNeem vraag {k}: *{q}* Met de {t} tools was dit de {which} vraag waarbij Qwen3 convert "
                "aanriep, met `{args}`. De client stuurt die call als één regel naar de server: een request met methode "
                "`tools/call`, de naam en de arguments van de tool, en in `_meta` de versie van MCP. De server voert convert uit "
                "en antwoordt met hetzelfde id, en een result met als content één tekstdeel: `{text}`. Die tekst geeft de client "
                "aan het model als resultaat van de tool, en het model antwoordt: *{answer}*").format(
                    k=i + 1, q=QUESTIONS[i][0], t=findings()["tools"], args=call["function"]["arguments"], text=text, answer=answer,
                    which="enige" if len(findings()["many_called"]) == 1 else "eerste")
    return ("### Worked example\n\nTake question {k}: *{q}* With the {t} tools, this was the {which} question in which Qwen3 called "
            "convert, with `{args}`. The client sends that call to the server as one line: a request with the method "
            "`tools/call`, the tool's name and arguments, and in `_meta` the version of MCP. The server runs convert and "
            "answers with the same id, and a result whose content is one part of text: `{text}`. The client gives that text to "
            "the model as the tool's result, and the model answers: *{answer}*").format(
                k=i + 1, q=QUESTIONS[i][0], t=findings()["tools"], args=call["function"]["arguments"], text=text, answer=answer,
                which="only" if len(findings()["many_called"]) == 1 else "first")


def worked_lines(i: int) -> list:
    """The lines over stdio for the model's call in the course's run of question i with the six servers' tools: the
    client's tools/call with the model's arguments, and the course's server's answer."""
    call = _first_call(many_steps(i))
    arguments = json.loads(call["function"]["arguments"])
    req = request(3, "tools/call", {"name": call["function"]["name"], "arguments": arguments})
    return [("client", req), ("server", COURSE_HANDLE(copy.deepcopy(req)))]


_WORKED_WORDS = {
    "en": {"title": "The model's call, as the client sends it to the server, and the answer", "client": "client → server",
           "server": "server → client", "model": "the model's call", "result": "what the model reads"},
    "nl": {"title": "De call van het model, zoals de client hem naar de server stuurt, en het antwoord", "client": "client → server",
           "server": "server → client", "model": "de call van het model", "result": "wat het model leest"},
}


def worked_picture(lang: str) -> str:
    w = _WORKED_WORDS[lang]
    i = worked_question()
    call = _first_call(many_steps(i))
    lines = worked_lines(i)
    answer = lines[1][1]
    text = "\n".join(p["text"] for p in answer["result"]["content"] if p["type"] == "text")
    rows = [_line(w["model"], f"{call['function']['name']}({call['function']['arguments']})", "var(--ea-pred-text)"),
            _line(w["client"], json.dumps(lines[0][1]), "var(--ea-ink)"),
            _line(w["server"], json.dumps(answer), "var(--ea-data-text)"),
            _line(w["result"], text, "var(--ea-data-text)")]
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- the demo: the learner's server as a program of its own, the learner's call_tool, and Qwen3 -------------------------

def downloaded() -> bool:
    return chat.downloaded()


def download_or_stop(button, texts: dict) -> None:
    chat.download_or_stop(button, texts)


DEMO_SERVER = paths.WORK / "projects" / "mcp" / "units_server.py"   # the learner's server, saved as a file of its own


def learner_fn(key: str, text):
    """The learner's code of an exercise when it is right (the code for the server, the function for the client); else None."""
    ex = EXERCISES[key]
    if text is MISSING or text is None or ex.check(text).state != "correct":
        return None
    return text if key == "server" else run_code(text, {}).names.get(ex.function)


def save_server(code: str):
    """Write the learner's server as a whole Python file, and give back its path."""
    DEMO_SERVER.parent.mkdir(parents=True, exist_ok=True)
    DEMO_SERVER.write_text(server_file(code), encoding="utf-8")
    return DEMO_SERVER


class Stdio:
    """The course's client end of a server that runs as a program of its own: each message goes in as one line, and a
    request's answer comes out as one line; it gives up on a server that does not answer within `wait` seconds."""

    def __init__(self, path, wait: float = 10.0):
        import queue
        import threading
        self.wait, self.log, self.failed = wait, [], False   # failed: the server gave no answer, or one that is not JSON
        self.process = subprocess.Popen([sys.executable, "-I", str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=str(path.parent),
                                        env={"PATH": os.environ.get("PATH", "")})
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message):
        import queue
        line = json.dumps(message)
        self.log.append(("client", line))
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()
        if not isinstance(message, dict) or "id" not in message:
            return None
        try:
            answer = self.lines.get(timeout=self.wait)
        except queue.Empty:
            answer = None
        if answer is None:
            self.failed = True
            raise RuntimeError("the server gave no answer")
        self.log.append(("server", answer.strip()))
        try:
            return json.loads(answer)
        except ValueError:
            self.failed = True
            raise

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:   # noqa: BLE001 - a server that does not stop
            self.process.kill()


def _post(url: str, messages: list, tools: list) -> dict:
    import urllib.request
    body = {"model": serve.MODEL, "messages": messages, "tools": tools, "max_tokens": 120, "temperature": 0}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())["choices"][0]["message"]


def _message(role: str, text: str) -> str:
    color = {"user": "var(--ea-ink)", "assistant": "var(--ea-pred-text)", "tool": "var(--ea-data-text)"}[role]
    return (f'<div style="border-left:3px solid {color};padding:0.15rem 0.5rem;margin:0.25rem 0">'
            f'<div style="{_UI};font-size:11px;color:var(--ea-ink-2)">{role}</div>{_code(text, color)}</div>')


def ask_server(code: str, call_fn, question: str, words: dict) -> str:
    """The learner's question to Qwen3, with the tools of the learner's server, which runs as a program of its own;
    each call goes to it through the learner's call_tool. The lines over stdio, and the answer."""
    path = save_server(code)
    server = Stdio(path)
    parts, phase = [], "server"   # an error while listing the tools is the server's; after that, the model's
    try:
        server.send(request(1, "server/discover"))
        listed = server.send(request(2, "tools/list"))
        tools = as_tools(listed["result"]["tools"])
        phase = "model"
        running = serve.model_server()
        messages = T.asked(question)
        message = _post(running.url, messages, tools)
        calls = message.get("tool_calls") or []
        if calls:
            messages = messages + [message]
            for k, c in enumerate(calls):
                try:
                    arguments = json.loads(c["function"]["arguments"])
                    text = call_fn(server.send, 3 + k, c["function"]["name"], arguments)
                except Exception as e:   # noqa: BLE001 - a call from a model, sent by the learner's code
                    text = f"Error: {type(e).__name__}: {e}"
                parts.append(_message("assistant", f"{c['function']['name']}({c['function']['arguments']})"))
                parts.append(_message("tool", str(text)))
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": str(text)})
            message = _post(running.url, messages, tools)
        else:
            parts.append(f'<p style="{_UI};font-size:13px;color:var(--ea-ink-2)">{html.escape(words["no_call"])}</p>')
    except Exception as e:   # the real server may run out of memory; the learner's server may stop
        if serve.out_of_memory(e):
            return f"<p><em>{html.escape(words['memory'])}</em></p>"
        said = words["stopped"] if phase == "server" or server.failed else words["failed"]
        parts.append(f'<p style="{_UI};font-size:13px;color:var(--ea-loss-text)">{html.escape(said.format(e=f"{type(e).__name__}: {e}"))}</p>')
        message = None
    finally:
        server.close()
    log = "".join(_line(words["client"] if who == "client" else words["server"], line,
                        "var(--ea-ink)" if who == "client" else "var(--ea-data-text)") for who, line in server.log)
    body = "".join(parts) + (_message("assistant", message.get("content") or "") if message else "")
    body += f'<details style="margin-top:0.4rem"><summary style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(words["lines"])}</summary>{log}</details>'
    body += f'<p style="{_UI};font-size:12px;color:var(--ea-ink-2)">{html.escape(words["saved"].format(path=str(path)))}</p>'
    return yours.figure(words["title"], body, alt=words["title"])
