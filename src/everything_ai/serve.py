# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""A model behind an API on your own computer: an OpenAI-compatible server on 127.0.0.1, with only the standard
library (http.server and json), for the lessons of domain bouwen. It answers

  GET  /v1/models             the model it serves
  POST /v1/chat/completions   an answer to a list of messages, with model, messages, max_tokens, temperature, top_p,
                              seed, stop, stream (and stream_options.include_usage) and tools

in the form of OpenAI's API, which Ollama, llama.cpp, vLLM and most hosted APIs speak as well: code a learner writes
against this server works against those too, with another URL and a key. Behind the server is a backend: `Model`, the
course's chat model (everything_ai.chat), or `StandIn`, which gives fixed answers at once, and on request also fails,
waits or breaks off. The checks talk to a stand-in, since the model takes seconds per answer and a check has two.

`start(backend)` starts a server in a thread of its own and gives a `Running` with its `url` and a `log` of every
request it got, which a check reads to say what a learner's request lacked."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen3-0.6b"   # the name the server gives the course's chat model
PORT = 8765            # where the lessons start the model's server, when that port is free
_STOP = object()


class Backend:
    """What answers behind the server. `answer(request, heard)` gives {"text", "tool_calls", "finish", "prompt_tokens",
    "tokens"} for a request that is valid; `heard(text)` hears the text so far while it is written, for streaming."""
    model = MODEL
    key = None   # when set, a request needs the header "Authorization: Bearer <key>"

    def before(self, request: dict):
        """Something other than an answer for this request, such as an error, as (status, headers, body); else None."""
        return None

    def answer(self, request: dict, heard) -> dict:
        raise NotImplementedError


class Model(Backend):
    """The course's chat model (everything_ai.chat.loaded(), or `chat` when given), one answer at a time."""

    def __init__(self, chat=None):
        self._chat = chat
        self._lock = threading.Lock()

    def answer(self, request: dict, heard) -> dict:
        from everything_ai import chat as C
        settings = {k: request[k] for k in ("temperature", "top_p", "seed") if request.get(k) is not None}
        stop = request.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        markers = [s for s in stop if s] + ["<tool_call>"]
        # Qwen3's own chat template thinks unless told not to, and so do vLLM and SGLang; this server answers the normal
        # way unless told to think, as thinking takes long on a CPU. A request switches it as on those servers:
        # "chat_template_kwargs": {"enable_thinking": true}. Thinking is sampled as Qwen3 advises for it.
        thinking = bool((request.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        if thinking:
            settings = {**C.THINKING, **settings}

        def shown(text):   # a stream shows the text up to a stop sequence or a tool call, and holds back what may start one
            if thinking:   # the thought is not part of the answer: the stream starts after it
                text = text.split("</think>", 1)[1].lstrip() if "</think>" in text else ""
            cut = [text.index(s) for s in markers if s in text]
            heard(_held(text[:min(cut)] if cut else text, markers))
        with self._lock:
            model = self._chat if self._chat is not None else C.loaded()
            a = model.answer(request["messages"], tools=request.get("tools") or None, thinking=thinking,
                             max_tokens=int(request.get("max_tokens") or 256), heard=shown, **settings)
        message, finish = a["message"], a["finish"]
        thought = C.THOUGHT.match(a["text"].lstrip())
        text = message["content"]
        cut = [text.index(s) for s in stop if s and s in text]
        if cut:   # the model went on past a stop sequence: the answer ends before it
            text, finish = text[:min(cut)], "stop"
        if message["tool_calls"] and not cut:
            finish = "tool_calls"
        out = {"text": text, "tool_calls": [] if cut else message["tool_calls"], "finish": finish,
               "prompt_tokens": a["prompt_tokens"], "tokens": a["tokens"]}
        if thinking:   # as vLLM gives it: the thought apart from the answer; a thought that never ended is all thought
            out["reasoning"] = thought.group(0).strip()[len("<think>"):-len("</think>")].strip() if thought else a["text"].replace("<think>", "", 1).strip()
            if not thought:
                out["text"] = ""
        return out


def _held(text: str, markers: list) -> str:
    """`text` without an end that could be the start of one of `markers`."""
    for n in range(min(len(text), max(len(m) for m in markers)), 0, -1):
        if any(m.startswith(text[-n:]) for m in markers):
            return text[:-n]
    return text


class StandIn(Backend):
    """A server's stand-in with fixed answers, for the checks: `answers` maps the last message of the user to the
    answer; any other question gets `default`, with {question} filled in. Tokens are counted as words. On request:
    `fail`, a list of statuses (such as [429, 503]) that the first requests get, one each, with Retry-After `retry`
    seconds; `delay` seconds before every answer; `key`, an API key that every request must carry; `calls`, a tool
    call per question; `cut`, the number of pieces after which a stream breaks off; `model`, the name of its model."""

    def __init__(self, answers=None, default="The stand-in's answer to: {question}", fail=(), retry=None, delay=0.0,
                 key=None, calls=None, cut=None, model=MODEL):
        self.answers, self.default, self.model = dict(answers or {}), default, model
        self.fail, self.retry, self.delay, self.key = list(fail), retry, delay, key
        self.calls, self.cut = dict(calls or {}), cut
        self.seen = 0

    def before(self, request: dict):
        self.seen += 1
        if self.seen <= len(self.fail):
            status = self.fail[self.seen - 1]
            headers = {"Retry-After": str(self.retry)} if self.retry is not None and status in (429, 503) else {}
            return status, headers, _error(status, _REASONS.get(status, "error"))
        return None

    def answer(self, request: dict, heard) -> dict:
        if self.delay:
            time.sleep(self.delay)
        question = _last_user(request["messages"])
        if question in self.calls:
            name, arguments = self.calls[question]
            return {"text": "", "tool_calls": [{"name": name, "arguments": arguments}], "finish": "tool_calls",
                    "prompt_tokens": _words(request["messages"]), "tokens": 0}
        text = self.answers.get(question, self.default.format(question=question))
        words = text.split(" ")
        limit = request.get("max_tokens")
        finish = "stop"
        if limit is not None and len(words) > int(limit):
            words, finish = words[:int(limit)], "length"
        for i in range(1, len(words) + 1):
            if self.cut is not None and i > self.cut:
                heard(_STOP)
                break
            heard(" ".join(words[:i]))
        return {"text": " ".join(words), "tool_calls": [], "finish": finish,
                "prompt_tokens": _words(request["messages"]), "tokens": len(words)}


_REASONS = {400: "The request is not valid.", 401: "No valid API key was given.", 403: "Not allowed.", 404: "Not found.",
            405: "Use another method.", 415: "The body must be JSON.", 429: "Too many requests: wait a moment and try again.",
            500: "Something went wrong in the server.", 503: "The server is busy: try again later."}
_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
          405: "invalid_request_error", 415: "invalid_request_error", 429: "rate_limit_error", 500: "server_error", 503: "server_error"}
_LOCAL = {"127.0.0.1", "localhost", "::1"}   # the names of this computer that a request may be addressed to
MEMORY = "The server ran out of memory. Close other programs, and try again."


def _memory_full(e: BaseException) -> bool:
    from everything_ai import chat
    return chat.memory_full(e)


def server_error(e: Exception) -> dict:
    """The error a server sent with an HTTPError, as OpenAI's API sends it: {"message": ..., "param": ...}. The body of
    an HTTPError can be read only once, so it is kept on the error."""
    if not hasattr(e, "_said"):
        try:
            said = json.loads(e.read())["error"]
            e._said = said if isinstance(said, dict) and "message" in said else {"message": str(e.reason)}
        except Exception:
            e._said = {"message": str(getattr(e, "reason", e))}
    return e._said


def out_of_memory(e: Exception) -> bool:
    """Whether an HTTPError a client got says the server ran out of memory: a 503 with MEMORY. In a stream the same
    words come in an error event: stream_error(line) == MEMORY."""
    import urllib.error
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 503 and server_error(e).get("message") == MEMORY
    return False


def stream_error(line: str) -> str | None:
    """The message of the error in a line of a stream, `data: {"error": ...}`; else None."""
    if not line.startswith('data: {"error"'):
        return None
    try:
        return str(json.loads(line[len("data: "):])["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return None


def _error(status: int, message: str, param: str | None = None) -> dict:
    """An error as OpenAI's API sends it; `param` names the field of the request that is wrong, when one is."""
    return {"error": {"message": message, "type": _TYPES.get(status, "error"), "param": param, "code": status}}


def _last_user(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _words(messages: list) -> int:
    return sum(len(str(m.get("content") or "").split()) + 3 for m in messages)


def _problem(request) -> str | None:
    """What makes a request for /v1/chat/completions not valid, in OpenAI's words more or less; None when it is."""
    if not isinstance(request, dict):
        return "The body must be a JSON object."
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a list with at least one message."
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            return "Each message needs a 'role': system, user, assistant or tool."
        if m["role"] in ("system", "user") and not isinstance(m.get("content"), str):
            return "A message of the system or the user needs its 'content' as a string."
    if request.get("max_tokens") is not None and (not isinstance(request["max_tokens"], int) or request["max_tokens"] < 1):
        return "'max_tokens' must be a whole number of at least 1."
    return None


class Running:
    """A server that runs: its `url` (ending in /v1, as a client's base_url), the `log` of the requests it got, each
    {"method", "path", "time" (seconds after the server started), "headers", "body" (the JSON, or the text when it was
    not JSON), "status", "answer" (the JSON sent back, or for a stream the number of events with a piece of the
    answer)}, and stop()."""

    def __init__(self, httpd, thread, backend):
        self.httpd, self.thread, self.backend, self.log = httpd, thread, backend, []
        self.started = time.perf_counter()
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _handler(running_ref: list):

    class Handler(BaseHTTPRequestHandler):
        server_version = "EverythingAI/1"
        protocol_version = "HTTP/1.0"   # the connection closes after each answer, so a stream needs no chunks

        def log_message(self, *args):   # nothing on the lesson's output
            pass

        def _send(self, status: int, body: dict, headers=None) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self._entry["status"], self._entry["answer"] = status, body
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):   # the client stopped waiting
                pass

        def _start(self, body):
            running = running_ref[0]
            self._entry = {"method": self.command, "path": self.path.split("?")[0], "time": time.perf_counter() - running.started,
                           "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body, "status": None}
            running.log.append(self._entry)
            return running.backend

        def _local(self) -> bool:
            """Whether the request is addressed to this computer by name: a web page elsewhere that makes the
            learner's browser call this server (DNS rebinding) sends its own host name instead."""
            host = (self.headers.get("Host") or "").strip().lower()
            host = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
            if host in _LOCAL:
                return True
            self._send(403, _error(403, "This server only answers requests addressed to 127.0.0.1 or localhost."))
            return False

        def do_GET(self):
            backend = self._start(None)
            if not self._local() or not self._allowed(backend):
                return
            if self._entry["path"].rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": backend.model, "object": "model", "owned_by": "everything-ai"}]})
            elif self._entry["path"].rstrip("/") == "/v1/chat/completions":   # a Request without data= is a GET
                self._send(405, _error(405, "Use POST for /v1/chat/completions, with the request as JSON in the body."))
            else:
                self._send(404, _error(404, f"There is no {self._entry['path']} here."))

        def _allowed(self, backend) -> bool:
            if backend.key is not None and self.headers.get("Authorization", "") != f"Bearer {backend.key}":
                self._send(401, _error(401, _REASONS[401]))
                return False
            return True

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            raw = self.rfile.read(length).decode("utf-8", "replace") if length > 0 else ""
            try:
                body = json.loads(raw)
            except ValueError:
                body = raw
            backend = self._start(body)
            if not self._local():
                return
            if length < 0:
                return self._send(400, _error(400, "The header Content-Length is not a number."))
            if self._entry["path"].rstrip("/") != "/v1/chat/completions":
                return self._send(404, _error(404, f"There is no {self._entry['path']} here."))
            if not self._allowed(backend):
                return
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if kind != "application/json":   # as real servers do; it also keeps out a plain form that a web page posts
                return self._send(415, _error(415, f"The body must be JSON, with the header Content-Type: application/json; this request said {kind or 'nothing'}."))
            if isinstance(body, str):
                return self._send(400, _error(400, "The body is not valid JSON."))
            problem = _problem(body)
            if problem:
                return self._send(400, _error(400, problem))
            if body.get("model") not in (None, backend.model):
                return self._send(404, _error(404, f"The model '{body.get('model')}' does not exist here; this server has '{backend.model}'.", "model"))
            other = backend.before(body)
            if other is not None:
                status, headers, out = other
                return self._send(status, out, headers)
            ident, created = f"chatcmpl-{len(running_ref[0].log)}", int(time.time())
            if body.get("stream"):
                return self._stream(backend, body, ident, created)
            try:
                a = backend.answer(body, lambda text: None)
            except Exception as e:   # noqa: BLE001 - the client gets the error instead of a closed connection
                if _memory_full(e):
                    return self._send(503, _error(503, MEMORY))
                return self._send(500, _error(500, f"The server had an error: {type(e).__name__}: {e}"))
            self._send(200, _completion(a, backend.model, ident, created))

        def _stream(self, backend, body, ident, created):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._entry["status"] = 200
            sent = [""]
            head = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": backend.model}

            events = [0]   # the number of events with a piece of the answer, for the log

            def event(choice: dict, **more) -> None:
                events[0] += 1 if choice.get("delta", {}).get("content") else 0
                self._entry["answer"] = events[0]
                self.wfile.write(b"data: " + json.dumps({**head, "choices": [choice], **more}, ensure_ascii=False).encode("utf-8") + b"\n\n")
                self.wfile.flush()

            class Broken(Exception):
                pass

            def heard(text):
                if text is _STOP:
                    raise Broken
                if text.endswith("�"):   # half a character so far: wait for the rest of its bytes
                    return
                piece = text[len(sent[0]):] if text.startswith(sent[0]) else ""
                if piece:
                    event({"index": 0, "delta": {"content": piece}, "finish_reason": None})
                    sent[0] = text

            try:
                event({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None})
                try:
                    a = backend.answer(body, heard)
                except Broken:   # the stand-in breaks off: the connection closes without [DONE]
                    return
                except (BrokenPipeError, ConnectionResetError):
                    raise
                except Exception as e:   # noqa: BLE001 - an error event, as OpenAI sends one, and no [DONE]
                    status, message = (503, MEMORY) if _memory_full(e) else (500, f"The server had an error: {type(e).__name__}: {e}")
                    self.wfile.write(b"data: " + json.dumps(_error(status, message)).encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    return
                rest = a["text"][len(sent[0]):] if a["text"].startswith(sent[0]) else ""
                if rest:
                    event({"index": 0, "delta": {"content": rest}, "finish_reason": None})
                if a["tool_calls"]:
                    event({"index": 0, "delta": {"tool_calls": _calls(a["tool_calls"], streamed=True)}, "finish_reason": None})
                event({"index": 0, "delta": {}, "finish_reason": a["finish"]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    self.wfile.write(b"data: " + json.dumps({**head, "choices": [], "usage": _usage(a)}).encode("utf-8") + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self._entry["answer"] = events[0]
            except (BrokenPipeError, ConnectionResetError):   # the client stopped listening
                return

    return Handler


def _calls(calls: list, streamed: bool = False) -> list:
    out = []
    for i, c in enumerate(calls):
        call = {"id": f"call_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
        out.append({"index": i, **call} if streamed else call)
    return out


def _usage(a: dict) -> dict:
    return {"prompt_tokens": a["prompt_tokens"], "completion_tokens": a["tokens"], "total_tokens": a["prompt_tokens"] + a["tokens"]}


def _completion(a: dict, model: str, ident: str, created: int) -> dict:
    message = {"role": "assistant", "content": a["text"] if a["text"] or not a["tool_calls"] else None}
    if a.get("reasoning") is not None:
        message["reasoning_content"] = a["reasoning"]
    if a["tool_calls"]:
        message["tool_calls"] = _calls(a["tool_calls"])
    return {"id": ident, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": a["finish"]}], "usage": _usage(a)}


def start(backend: Backend, port: int = 0) -> Running:
    """A server for `backend` on 127.0.0.1, on `port` (0: a free one), in a thread of its own."""
    ref = []
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(ref))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)   # stops quickly
    running = Running(httpd, thread, backend)
    ref.append(running)
    thread.start()
    return running


_SERVERS: dict = {}


def model_server() -> Running:
    """The server of the course's chat model, started once in a session: on PORT when it is free, else on another."""
    if "model" not in _SERVERS:
        backend = Model()
        try:
            _SERVERS["model"] = start(backend, PORT)
        except OSError:
            _SERVERS["model"] = start(backend, 0)
    return _SERVERS["model"]
