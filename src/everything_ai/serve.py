# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""A model behind an API on your own computer: an OpenAI-compatible server on 127.0.0.1, with only the standard
library (http.server and json), for the lessons of domain bouwen. It answers, in the form of OpenAI's API,

  GET  /v1/models                 the models it serves: the chat model and the embedding model
  POST /v1/chat/completions       an answer to a list of messages, with model, messages, max_tokens, temperature, top_p,
                                  seed, stop, stream (and stream_options.include_usage), tools and tool_choice ("auto",
                                  "none", "required", or one function by name: the server starts the answer with a tool
                                  call, of that function when named, and the model writes the rest); a list of messages
                                  that ends with one of the assistant is a prefill, as llama.cpp's server takes it: the
                                  answer goes on from that text. It keeps the KV cache of the last request, as
                                  llama.cpp's server does (prompt caching): usage.prompt_tokens_details.cached_tokens
                                  says how many tokens of the prompt came from it, and "cache_prompt": false switches
                                  it off for a request. With "logprobs": true (and "top_logprobs": n, up to 20),
                                  without a stream and without thinking, the answer also gives the log probability of
                                  every token it wrote, and of the n most likely tokens at each place, under
                                  choices[0].logprobs.content, as OpenAI's API does
  POST /v1/embeddings             the embeddings of a text or a list of texts, with model, input and encoding_format
                                  ("float", or "base64": the float32 numbers as bytes, little-endian, in base64)
  POST /v1/images/generations     images of a prompt, with model, prompt, n, size and response_format ("b64_json": a
                                  PNG in base64); the course draws the shapes the prompt names (everything_ai.drawing):
                                  no model, so /v1/models does not list it

A message of the user may also be a list of parts, text and images, as a multimodal API takes them: an image as a data
URL, data:image/png;base64,... The course's chat model reads text only, so the server answers such a request with an
error, unless the request asks for the server's own tool, {"type": "detect_objects"}: a hosted tool, which the server
runs itself before the model answers. It lets the course's detector (everything_ai.detr) name the objects on each
image, and gives the model those words in place of the image; the answer says what it found, under "hosted_tools".

Ollama, llama.cpp, vLLM and most hosted APIs speak OpenAI's form as well: code a learner writes against this server
works against those too, with another URL and a key. Behind the server is a backend: `Model`, the
course's chat model (everything_ai.chat) and embedding model (everything_ai.embed), or `StandIn`, which gives fixed
answers and made-up embeddings at once, and on request also fails, waits or breaks off. The checks talk to a stand-in,
since the model takes seconds per answer and a check has two.

`start(backend)` starts a server in a thread of its own and gives a `Running` with its `url` and a `log` of every
request it got, which a check reads to say what a learner's request lacked."""
import base64
import hashlib
import json
import math
import random
import re
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen3-0.6b"   # the name the server gives the course's chat model
EMBEDDER = "granite-embedding-107m-multilingual"   # and its embedding model
IMAGER = "course-shapes"   # and its drawing program, at /v1/images/generations: not a model (everything_ai.drawing)
MAX_IMAGES = 4             # images in one request at most
DIMS = 384             # the numbers in one of its embeddings
MAX_INPUTS = 2048      # texts in one request for embeddings at most, as at OpenAI
PORT = 8765            # where the lessons start the model's server, when that port is free
_STOP = object()


DETECT = "detect_objects"   # the server's hosted tool: the course's detector names the objects on an image
THRESHOLD = 0.9             # the score above which the detector names an object, as in cv-pretrained-detector
MAX_PIXELS = 4096 * 4096    # the largest image the server reads
MAX_TOP_LOGPROBS = 20       # the most likely tokens a request may ask for at each place, as at OpenAI


class Backend:
    """What answers behind the server. `answer(request, heard)` gives {"text", "tool_calls", "finish", "prompt_tokens",
    "tokens"} for a request that is valid; `heard(text)` hears the text so far while it is written, for streaming.
    `detect(image)` gives the kinds and scores of the objects on a PIL image, for the hosted tool. `embed(texts)` gives
    the embedding of each text, a list of `dims` numbers, and the number of tokens of all texts together. `draw(prompt,
    n, size)` gives n PNGs of the shapes the prompt names, each with the shapes in other places."""
    model = MODEL
    embedder = EMBEDDER
    imager = IMAGER
    dims = DIMS
    key = None   # when set, a request needs the header "Authorization: Bearer <key>"

    def before(self, request: dict):
        """Something other than an answer for this request, such as an error, as (status, headers, body); else None."""
        return None

    def answer(self, request: dict, heard) -> dict:
        raise NotImplementedError

    def detect(self, image) -> list[tuple[str, float]]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        raise NotImplementedError

    def draw(self, prompt: str, n: int, size: int) -> list[bytes]:
        from everything_ai import drawing
        return [drawing.draw(prompt, size, seed) for seed in range(n)]


class Model(Backend):
    """The course's chat model (everything_ai.chat.loaded(), or `chat` when given), one answer at a time, and its
    embedding model, Granite Embedding (everything_ai.embed), loaded at the first request for embeddings."""

    def __init__(self, chat=None):
        self._chat = chat
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()   # embeddings do not wait for an answer that is being written
        self._detector = None

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """Granite's embeddings of `texts`, each of length 1, and their tokens, <s> and </s> included; a text longer
        than 512 tokens is cut off, as everything_ai.embed does. It raises FileNotFoundError while Granite is not
        downloaded."""
        from everything_ai import embed as E
        with self._embed_lock:
            ids = [E.encode(t) for t in texts]
            return E.embed_ids(ids).tolist(), sum(len(x) for x in ids)

    def detect(self, image) -> list[tuple[str, float]]:
        """DETR's objects on the image, above THRESHOLD. It raises FileNotFoundError while DETR is not downloaded."""
        from everything_ai import detr
        with self._lock:
            if self._detector is None:
                self._detector = detr.load()
            return detr.found(self._detector, image, THRESHOLD)

    def answer(self, request: dict, heard) -> dict:
        from everything_ai import chat as C
        settings = {k: request[k] for k in ("temperature", "top_p", "seed") if request.get(k) is not None}
        stop = request.get("stop") or []
        stop = [stop] if isinstance(stop, str) else list(stop)
        markers = [s for s in stop if s] + ["<tool_call>"]
        # Qwen3's own chat template thinks unless told not to, and so do vLLM and SGLang; this server answers the normal
        # way unless told to think, as thinking takes long on a CPU. A request switches it as on those servers:
        # "chat_template_kwargs": {"enable_thinking": true}. Thinking is sampled as Qwen3 advises for it.
        messages, tools, forced = request["messages"], request.get("tools") or None, ""
        choice = request.get("tool_choice") or "auto"   # null is the default, as at OpenAI
        if choice == "none" or not tools:   # the model reads no tools, so it answers with text
            tools = None
        elif choice != "auto":   # "required", or one function by name: the answer starts with a tool call, as a prefill
            # no space after the colon: the model writes " {" as one token, as it does in a call of its own
            forced = "<tool_call>\n" + ('{"name": ' + json.dumps(choice["function"]["name"]) + ', "arguments":' if isinstance(choice, dict) else "")
            messages = messages + [{"role": "assistant", "content": forced}]
        thinking = bool((request.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        thinking = thinking and _prefill(messages) is None   # after a prefill Qwen3 does not think
        if thinking:
            settings = {**C.THINKING, **settings}

        def shown(text):   # a stream shows the text up to a stop sequence or a tool call, and holds back what may start one
            if thinking:   # the thought is not part of the answer: the stream starts after it
                text = text.split("</think>", 1)[1].lstrip() if "</think>" in text else ""
            text = forced + text   # after a forced start the model writes a tool call, which a stream does not show
            cut = [text.index(s) for s in markers if s in text]
            heard(_held(text[:min(cut)] if cut else text, markers))
        with self._lock:
            model = self._chat if self._chat is not None else C.loaded()
            if stop and not forced:   # the model stops writing at a stop sequence, so the tokens after it are never written;
                # a forced answer is a tool call, and a stop sequence is for text, so it does not cut the call's arguments
                settings["stop"] = [s for s in stop if s]
            if (request.get("response_format") or {}).get("type") == "json_object":   # JSON mode
                settings["json_object"] = True
            if request.get("logprobs"):
                settings["logprobs"] = int(request.get("top_logprobs") or 0)
            # prompt caching, as llama.cpp's server does it: the KV cache of the last request is kept, and a prompt that
            # starts the same way goes on from it; "cache_prompt": false computes the whole prompt again
            a = model.answer(messages, tools=tools, thinking=thinking,
                             max_tokens=int(request.get("max_tokens") or 256), heard=shown,
                             cache=request.get("cache_prompt", True) is not False, keep=True, **settings)
        message, finish = (C.parse(forced + a["text"]) if forced else a["message"]), a["finish"]
        if tools is None:   # without tools, a tool call the model wrote anyway stays out of the answer
            message = {**message, "tool_calls": []}
        elif message["bad_calls"]:   # a call that cannot be read: no calls, and all it wrote as the answer, as vLLM does
            message = {**message, "tool_calls": [], "content": C.THOUGHT.sub("", forced + a["text"]).strip()}
        thought = C.THOUGHT.match(a["text"].lstrip())
        text = message["content"]
        if _prefill(request["messages"]) is not None and not thinking:   # the answer goes on from the prefill: keep its first space
            text = a["text"] if not message["tool_calls"] else text
        cut = [text.index(s) for s in stop if s and s in text] if not forced else []
        if cut:   # the model went on past a stop sequence: the answer ends before it
            text, finish = text[:min(cut)], "stop"
        if message["tool_calls"] and not cut:
            finish = "tool_calls"
        out = {"text": text, "tool_calls": [] if cut else message["tool_calls"], "finish": finish,
               "prompt_tokens": a["prompt_tokens"], "tokens": a["tokens"], "cached_tokens": a.get("cached_tokens", 0)}
        if a.get("logprobs") is not None:   # every token written, as the model wrote it; none after a stop sequence
            kept, at = a["logprobs"], [a["text"].index(s) for s in stop if s and s in a["text"]]
            if cut and at:   # the tokens that start before the first stop sequence in what the model wrote
                end, size, kept = len(a["text"][:min(at)].encode("utf-8")), 0, []
                for entry in a["logprobs"]:
                    if size >= end:
                        break
                    kept.append(entry)
                    size += len(entry["bytes"])
            out["logprobs"] = kept
        if thinking:   # as vLLM gives it: the thought apart from the answer; a thought that never ended is all thought
            out["reasoning"] = thought.group(0).strip()[len("<think>"):-len("</think>")].strip() if thought else a["text"].replace("<think>", "", 1).strip()
            if not thought:
                out["text"] = ""
        return out


def made_up(text: str, dims: int = DIMS) -> list[float]:
    """The stand-in's embedding of a text: `dims` numbers drawn at random, with the text as the seed, so that a text
    always gets the same ones, and scaled to length 1. They mean nothing: two texts of the same meaning get numbers as
    far apart as any two."""
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    v = [rng.gauss(0.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in v))
    return [round(x / norm, 6) for x in v]


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
    seconds; `delay` seconds before every answer; `key`, an API key that every request must carry; `calls`, the tool
    call per question, (name, arguments), or a list of them for several calls in one answer, made unless the request
    says "tool_choice": "none"; `after`, the answer once the last message is a tool's result, with {result} filled in
    (the results since the last answer, each on a line of its own), or a function of the request that gives it; `cut`, the number of pieces after which a stream breaks off; `model`, the name of its model;
    `objects`, what its detector finds on any image; `vectors`, the embedding of a text, {text: [numbers]}, where any
    other text gets made-up numbers of its own (`made_up`), `dims` of them. Like the server with the model, it ends an
    answer before a stop sequence, and after a prefill it answers only the rest: what comes after the prefill in the
    answer, or the whole answer when it does not start with it. Having no model, it gives no logprobs: a request for them
    gets a 400."""

    def __init__(self, answers=None, default="The stand-in's answer to: {question}", fail=(), retry=None, delay=0.0,
                 key=None, calls=None, cut=None, model=MODEL, objects=(("cup", 0.997),), vectors=None, dims=DIMS,
                 after="The stand-in read the result: {result}"):
        self.answers, self.default, self.model = dict(answers or {}), default, model
        self.fail, self.retry, self.delay, self.key = list(fail), retry, delay, key
        self.calls, self.cut, self.objects, self.after = dict(calls or {}), cut, list(objects), after
        self.vectors, self.dims = {t: list(v) for t, v in (vectors or {}).items()}, dims
        self.seen = 0

    def detect(self, image) -> list[tuple[str, float]]:
        return list(self.objects)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        if self.delay:
            time.sleep(self.delay)
        return ([self.vectors[t] if t in self.vectors else made_up(t, self.dims) for t in texts],
                sum(len(t.split()) for t in texts))

    def before(self, request: dict):
        self.seen += 1
        if self.seen <= len(self.fail):
            status = self.fail[self.seen - 1]
            headers = {"Retry-After": str(self.retry)} if self.retry is not None and status in (429, 503) else {}
            return status, headers, _error(status, _REASONS.get(status, "error"))
        if request.get("logprobs"):
            return 400, {}, _error(400, "The stand-in has no model, so it gives no logprobs: leave out 'logprobs'.", "logprobs")
        return None

    def answer(self, request: dict, heard) -> dict:
        if self.delay:
            time.sleep(self.delay)
        question = _last_user(request["messages"])
        if request["messages"][-1].get("role") == "tool":   # the tools' results came back: the answer reads them
            text = self.after(request) if callable(self.after) else self.after.format(result="\n".join(_results(request["messages"])))
        elif question in self.calls and request.get("tool_choice") != "none" and request.get("tools"):
            calls = self.calls[question]
            calls = calls if isinstance(calls, list) else [calls]
            return {"text": "", "tool_calls": [{"name": n, "arguments": a} for n, a in calls], "finish": "tool_calls",
                    "prompt_tokens": _words(request["messages"]), "tokens": 0}
        else:
            text = self.answers.get(question, self.default.format(question=question))
        start = _prefill(request["messages"])
        if start and text.startswith(start):   # a prefill: the answer goes on after it
            text = text[len(start):]
        stop = request.get("stop") or []
        cut = [text.index(s) for s in ([stop] if isinstance(stop, str) else stop) if isinstance(s, str) and s and s in text]
        if cut:   # the answer ends before a stop sequence
            text = text[:min(cut)]
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


def _prefill(messages: list) -> str | None:
    """The start of the answer when a chat ends with a message of the assistant without tool calls, as llama.cpp's
    server reads it: the model goes on from there, and the answer holds only what it writes after it. Else None."""
    last = messages[-1] if messages else {}
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        return last.get("content") or ""
    return None


def _results(messages: list) -> list:
    """The contents of the tool messages at the end of a chat: the results since the model's last answer."""
    out = []
    for m in reversed(messages):
        if m.get("role") != "tool":
            break
        out.insert(0, str(m.get("content")))
    return out


def _last_user(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _words(messages: list) -> int:
    return sum(len(str(m.get("content") or "").split()) + 3 for m in messages)


def _part_problem(part) -> str | None:
    """What makes one part of a message's content not valid; None when it is."""
    if not isinstance(part, dict) or part.get("type") not in ("text", "image_url", "input_audio"):
        return "Each part of a message's content needs a 'type': text, image_url or input_audio."
    if part["type"] == "text":
        return None if isinstance(part.get("text"), str) else "A part of type text needs its 'text' as a string."
    if part["type"] == "input_audio":
        return "The model hears no audio, and this server has no tool that listens."
    url = part.get("image_url").get("url") if isinstance(part.get("image_url"), dict) else None
    if not isinstance(url, str):
        return 'A part of type image_url needs "image_url": {"url": ...}, with the image as a data URL.'
    if url.startswith(("http://", "https://")):
        return "This server reads an image only as a data URL, data:image/png;base64,..., not from an address."
    if not url.startswith("data:image/") or ";base64," not in url:
        return "The url of an image must be a data URL: data:image/png;base64, and then the image in base64."
    try:
        base64.b64decode(url.split(";base64,", 1)[1], validate=True)
    except ValueError:
        return "The image in the data URL is not valid base64."
    return None


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
        content = m.get("content")
        if m["role"] == "user" and isinstance(content, list):
            for part in content:
                problem = _part_problem(part)
                if problem:
                    return problem
        elif m["role"] in ("system", "user") and not isinstance(content, str):
            return "A message of the system or the user needs its 'content' as a string, or for the user a list of parts."
        elif m["role"] == "assistant" and content is not None and not isinstance(content, str):
            return "A message of the assistant needs its 'content' as a string, or null."
    if request.get("max_tokens") is not None and (not isinstance(request["max_tokens"], int) or request["max_tokens"] < 1):
        return "'max_tokens' must be a whole number of at least 1."
    for name, high in (("temperature", 2), ("top_p", 1)):
        value = request.get(name)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= high):
            return f"'{name}' must be a number from 0 to {high}."
    if request.get("seed") is not None and (not _whole(request["seed"]) or not 0 <= request["seed"] < 2 ** 64):
        return f"'seed' must be a whole number from 0 to {2 ** 64 - 1}."
    stop = request.get("stop")
    if stop is not None and not isinstance(stop, str) and (not isinstance(stop, list) or len(stop) > 4
                                                           or not all(isinstance(x, str) for x in stop)):
        return "'stop' must be a string or a list of at most 4 strings."
    if request.get("logprobs") is not None and not isinstance(request["logprobs"], bool):
        return "'logprobs' must be true or false."
    top = request.get("top_logprobs")
    if top is not None and (not _whole(top) or not 0 <= top <= MAX_TOP_LOGPROBS):
        return f"'top_logprobs' must be a whole number from 0 to {MAX_TOP_LOGPROBS}."
    if top is not None and request.get("logprobs") is not True:
        return "'top_logprobs' needs 'logprobs': true."
    if request.get("logprobs") and request.get("stream"):
        return "This server gives logprobs only without a stream: leave out 'stream' or 'logprobs'."
    if request.get("logprobs") and (request.get("chat_template_kwargs") or {}).get("enable_thinking"):
        return "This server gives logprobs only without thinking."
    form = request.get("response_format")
    if form is not None:
        kind = form.get("type") if isinstance(form, dict) else None
        if kind == "json_schema":
            return ("This server has JSON mode, 'response_format': {'type': 'json_object'}, but no 'json_schema': "
                    "check the answer against your schema in your code.")
        if kind not in ("text", "json_object"):
            return "'response_format' must be {'type': 'text'} or {'type': 'json_object'}."
        if kind == "json_object" and (request.get("chat_template_kwargs") or {}).get("enable_thinking"):
            return "This server has JSON mode only without thinking."
    tools = request.get("tools")
    if tools is not None and (not isinstance(tools, list)
                              or any(not isinstance(t, dict) or t.get("type") not in ("function", DETECT) for t in tools)):
        return f"'tools' must be a list of tools, each of type function or {DETECT}."
    return _tool_problem(request)


def _tool_problem(request: dict):
    """What makes the tools, tool_choice or tool messages of a valid request otherwise not valid, as (text, field);
    None when nothing does. As at OpenAI: a function needs a name; every tool call of the assistant gets its result, a
    message with the role tool and the call's id, right after it, and a tool message answers such a call."""
    functions = [t.get("function") for t in request.get("tools") or [] if t.get("type") == "function"]
    if any(not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"] for f in functions):
        return ("Each tool of type function needs 'function': an object with its 'name', its 'description' and its "
                "'parameters', a JSON schema.", "tools")
    names = [f["name"] for f in functions]
    odd = [n for n in names if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n)]
    if odd:
        return (f"The name of a function may hold only letters, digits, _ and -, at most 64 of them; '{odd[0]}' does "
                "not.", "tools")
    choice = request.get("tool_choice")
    if choice is not None:
        named = (isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict)
                 and isinstance(choice["function"].get("name"), str))
        if choice not in ("none", "auto", "required") and not named:
            return ("'tool_choice' must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": "
                    "{\"name\": ...}}.", "tool_choice")
        if choice not in ("none", "auto") and not names:
            return "'tool_choice' can ask for a tool call only when 'tools' has functions.", "tool_choice"
        if named and choice["function"]["name"] not in names:
            return f"'tool_choice' names the function '{choice['function']['name']}', which is not among the tools.", "tool_choice"
        if choice not in ("none", "auto") and _prefill(request["messages"]) is not None:
            return ("A 'tool_choice' that asks for a tool call starts the answer itself, so the chat cannot end with a "
                    "message of the assistant.", "tool_choice")
    waiting = []   # the ids of the tool calls of the last answer that have no result yet
    for m in request["messages"]:
        if m["role"] == "tool":
            if not isinstance(m.get("tool_call_id"), str):
                return "A message with the role tool needs a 'tool_call_id': the 'id' of the tool call it answers.", "messages"
            if m["tool_call_id"] not in waiting:
                return (f"A message with the role tool answers a tool call in the message of the assistant before it, "
                        f"but there is no call with the id '{m['tool_call_id']}' that still waits for its result.", "messages")
            if not isinstance(m.get("content"), str):
                return ("A message with the role tool needs its 'content' as a string: turn a number or a dict into text "
                        "first, such as with json.dumps.", "messages")
            waiting.remove(m["tool_call_id"])
            continue
        if waiting:
            return (f"Each tool call of the assistant needs its result right after it: a message with the role tool and "
                    f"the call's 'tool_call_id'. No result came for {', '.join(waiting)}.", "messages")
        calls = m.get("tool_calls") if m["role"] == "assistant" else None
        if calls is not None:
            if not isinstance(calls, list) or any(
                    not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("function"), dict)
                    or not isinstance(c["function"].get("name"), str) or not isinstance(c["function"].get("arguments"), str)
                    for c in calls):
                return ("Each tool call in a message of the assistant needs its 'id', and a 'function' with its 'name' "
                        "and its 'arguments' as a JSON string, as the answer gave them.", "messages")
            waiting = [c["id"] for c in calls]
    if waiting:
        return (f"Each tool call of the assistant needs its result right after it: a message with the role tool and the "
                f"call's 'tool_call_id'. No result came for {', '.join(waiting)}.", "messages")
    return None


_KINDS = {list: "a list", dict: "an object", float: "a number", bool: "true or false", type(None): "null"}


def _whole(x) -> bool:
    """Whether x is a whole number, as a token id is; true and false are not."""
    return isinstance(x, int) and not isinstance(x, bool)


def _embed_problem(request) -> tuple[str, str | None] | None:
    """What makes a request for /v1/embeddings not valid, and the field it is about; None when it is valid."""
    if not isinstance(request, dict):
        return "The body must be a JSON object.", None
    if "input" not in request:
        return "The request needs 'input': a text, or a list of texts.", "input"
    texts = request["input"]
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list) or not texts:
        return "'input' must be a text, or a list with at least one text.", "input"
    if len(texts) > MAX_INPUTS:
        return f"'input' may hold at most {MAX_INPUTS:,} texts; this request has {len(texts):,}.", "input"
    for i, t in enumerate(texts):
        if _whole(t) or isinstance(t, list) and t and all(_whole(x) for x in t):
            return "This server takes texts in 'input', not token ids.", "input"
        if not isinstance(t, str):
            return f"Every item of 'input' must be a text; item {i} is {_KINDS.get(type(t), 'not')}.", "input"
        if not t.strip():
            return f"A text in 'input' may not be empty; item {i} is." if len(texts) > 1 else "The text in 'input' may not be empty.", "input"
    if request.get("encoding_format") not in (None, "float", "base64"):
        return "'encoding_format' must be \"float\" or \"base64\".", "encoding_format"
    return None


def _image_problem(request) -> tuple[str, str | None] | None:
    """What makes a request for /v1/images/generations not valid, and the field it is about; None when it is valid."""
    from everything_ai import drawing
    if not isinstance(request, dict):
        return "The body must be a JSON object.", None
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "The request needs 'prompt': a text that describes the image.", "prompt"
    n = request.get("n", 1)
    if not _whole(n) or not 1 <= n <= MAX_IMAGES:
        return f"'n' must be a whole number from 1 to {MAX_IMAGES}.", "n"
    size = request.get("size", "256x256")
    if not isinstance(size, str) or size not in drawing.SIZES:
        sizes = [f'"{k}"' for k in drawing.SIZES]
        return f"'size' must be {', '.join(sizes[:-1])} or {sizes[-1]}.", "size"
    if request.get("response_format") == "url":
        return "This server keeps no images to link to: ask for \"b64_json\", and the image comes in the answer.", "response_format"
    if request.get("response_format") not in (None, "b64_json"):
        return "'response_format' must be \"b64_json\".", "response_format"
    shapes = drawing.read(prompt)
    if not shapes:
        return ("This server is no image model: it draws circles, squares and triangles, in red, blue, green, yellow, "
                "purple or teal. Name some in the prompt, such as \"two red circles and a blue square\"."), "prompt"
    if len(shapes) > drawing.MAX_SHAPES:
        return f"This server draws at most {drawing.MAX_SHAPES} shapes on an image; the prompt names {len(shapes)}.", "prompt"
    return None


def _floats(vector: list[float], encoding: str):
    """An embedding as the answer gives it: a list of numbers, or for "base64" its float32 bytes in base64."""
    if encoding == "base64":
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector


def _images(request: dict) -> list:
    """The images in a request's messages, as the bytes their data URLs hold, in order."""
    return [base64.b64decode(p["image_url"]["url"].split(";base64,", 1)[1])
            for m in request["messages"] if m.get("role") == "user" and isinstance(m.get("content"), list)
            for p in m["content"] if p.get("type") == "image_url"]


def named(found: list) -> str:
    """The detector's objects in words: "cup, dining table, spoon", with a count after a kind found more than once."""
    counts: dict = {}
    for kind, _ in found:
        counts[kind] = counts.get(kind, 0) + 1
    return ", ".join(k if n == 1 else f"{k} ({n})" for k, n in counts.items())


def _parts_or_hosted(request: dict) -> bool:
    """Whether a request has a message whose content is a list of parts, or asks for the hosted tool."""
    return (any(m.get("role") == "user" and isinstance(m.get("content"), list) for m in request["messages"])
            or any(t.get("type") == DETECT for t in request.get("tools") or []))


def _described(request: dict, backend: Backend) -> tuple[dict, list]:
    """The request as the model reads it: every content of parts as one text, with every image replaced by the words of
    the hosted tool, and without the hosted tool among the tools; and what the tool found, for "hosted_tools". It raises
    ValueError for an image the server cannot read."""
    import io
    runs, messages, k = [], [], 0
    for m in request["messages"]:
        if m.get("role") != "user" or not isinstance(m.get("content"), list):
            messages.append(m)
            continue
        texts = []
        for p in m["content"]:
            if p["type"] == "text":
                texts.append(p["text"])
                continue
            k += 1
            from PIL import Image
            try:
                image = Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(";base64,", 1)[1])))
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError(f"Image {k} is larger than this server reads: at most {MAX_PIXELS:,} pixels.")
                image = image.convert("RGB")
            except ValueError:
                raise
            except Exception:   # noqa: BLE001 - Pillow says so in many ways
                raise ValueError(f"Image {k} is not an image this server can read: send a PNG, JPEG, GIF or WebP.") from None
            found = backend.detect(image)
            runs.append({"type": DETECT, "image": k, "found": [{"kind": c, "score": round(s, 3)} for c, s in found]})
            texts.append(f"[Image {k}. The objects the server's detector found on it: {named(found)}.]" if found
                         else f"[Image {k}. The server's detector found no object it knows on it.]")
        messages.append({**m, "content": "\n".join(texts)})
    tools = [t for t in request.get("tools") or [] if t.get("type") == "function"]
    return {**request, "messages": messages, "tools": tools or None}, runs


_POSTS = ("/v1/chat/completions", "/v1/embeddings", "/v1/images/generations")   # what takes a POST


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
            path = self._entry["path"].rstrip("/")
            if path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "everything-ai"}
                                                            for m in (backend.model, backend.embedder)]})
            elif path in _POSTS:   # a Request without data= is a GET
                self._send(405, _error(405, f"Use POST for {path}, with the request as JSON in the body."))
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
            path = self._entry["path"].rstrip("/")
            if path not in _POSTS:
                return self._send(404, _error(404, f"There is no {self._entry['path']} here."))
            if not self._allowed(backend):
                return
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if kind != "application/json":   # as real servers do; it also keeps out a plain form that a web page posts
                return self._send(415, _error(415, f"The body must be JSON, with the header Content-Type: application/json; this request said {kind or 'nothing'}."))
            if isinstance(body, str):
                return self._send(400, _error(400, "The body is not valid JSON."))
            if path == "/v1/embeddings":
                return self._embeddings(backend, body)
            if path == "/v1/images/generations":
                return self._images(backend, body)
            problem = _problem(body)
            if problem:   # a text, or (text, the field it is about)
                return self._send(400, _error(400, *problem) if isinstance(problem, tuple) else _error(400, problem))
            if body.get("model") == backend.embedder:
                return self._send(400, _error(400, f"'{backend.embedder}' is an embedding model: it gives embeddings, it writes no answers. Send texts to /v1/embeddings, or ask '{backend.model}' here.", "model"))
            if body.get("model") == backend.imager:
                return self._send(400, _error(400, f"'{backend.imager}' draws images, it writes no answers. Ask for images at /v1/images/generations, or ask '{backend.model}' here.", "model"))
            if body.get("model") not in (None, backend.model):
                return self._send(404, _error(404, f"The model '{body.get('model')}' does not exist here; this server has '{backend.model}'.", "model"))
            if _images(body) and not any(t.get("type") == DETECT for t in body.get("tools") or []):
                return self._send(400, _error(400, f"The model '{backend.model}' reads text only. To have this server name the objects on each image first, ask for its hosted tool: \"tools\": [{{\"type\": \"{DETECT}\"}}].", "messages"))
            other = backend.before(body)
            if other is not None:
                status, headers, out = other
                return self._send(status, out, headers)
            runs = []
            if _parts_or_hosted(body):
                try:
                    body, runs = _described(body, backend)
                except ValueError as e:
                    return self._send(400, _error(400, str(e), "messages"))
                except FileNotFoundError:
                    return self._send(503, _error(503, "The server's detector is not downloaded yet."))
                except Exception as e:   # noqa: BLE001
                    if _memory_full(e):
                        return self._send(503, _error(503, MEMORY))
                    return self._send(500, _error(500, f"The server's detector had an error: {type(e).__name__}: {e}"))
            ident, created = f"chatcmpl-{len(running_ref[0].log)}", int(time.time())
            if body.get("stream"):
                return self._stream(backend, body, ident, created, runs)
            try:
                a = backend.answer(body, lambda text: None)
            except Exception as e:   # noqa: BLE001 - the client gets the error instead of a closed connection
                if _memory_full(e):
                    return self._send(503, _error(503, MEMORY))
                return self._send(500, _error(500, f"The server had an error: {type(e).__name__}: {e}"))
            out = _completion(a, backend.model, ident, created, _earlier_calls(body["messages"]))
            if runs:
                out["hosted_tools"] = runs
            self._send(200, out)

        def _embeddings(self, backend, body):
            problem = _embed_problem(body)
            if problem:
                return self._send(400, _error(400, *problem))
            model = body.get("model")
            if model in (backend.model, backend.imager):
                what = "is a chat model: it writes answers" if model == backend.model else "draws images"
                return self._send(400, _error(400, f"'{model}' {what}, it gives no embeddings. For embeddings this server has '{backend.embedder}'.", "model"))
            if model not in (None, backend.embedder):
                return self._send(404, _error(404, f"The model '{model}' does not exist here; for embeddings this server has '{backend.embedder}'.", "model"))
            if body.get("dimensions") not in (None, backend.dims):
                return self._send(400, _error(400, f"'{backend.embedder}' gives {backend.dims} numbers per text; it cannot give another number of them.", "dimensions"))
            other = backend.before(body)
            if other is not None:
                status, headers, out = other
                return self._send(status, out, headers)
            texts = [body["input"]] if isinstance(body["input"], str) else body["input"]
            try:
                vectors, tokens = backend.embed(texts)
            except FileNotFoundError:
                return self._send(503, _error(503, "The server's embedding model is not downloaded yet."))
            except Exception as e:   # noqa: BLE001
                if _memory_full(e):
                    return self._send(503, _error(503, MEMORY))
                return self._send(500, _error(500, f"The server had an error: {type(e).__name__}: {e}"))
            encoding = body.get("encoding_format") or "float"
            self._send(200, {"object": "list", "model": backend.embedder,
                             "data": [{"object": "embedding", "index": i, "embedding": _floats(v, encoding)} for i, v in enumerate(vectors)],
                             "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})

        def _images(self, backend, body):
            from everything_ai import drawing
            problem = _image_problem(body)
            if problem:
                return self._send(400, _error(400, *problem))
            model = body.get("model")
            if model in (backend.model, backend.embedder):
                what = "a chat model" if model == backend.model else "an embedding model"
                return self._send(400, _error(400, f"'{model}' is {what}: it draws no images. For images this server has '{backend.imager}'.", "model"))
            if model not in (None, backend.imager):
                return self._send(404, _error(404, f"The model '{model}' does not exist here; for images this server has '{backend.imager}'.", "model"))
            other = backend.before(body)
            if other is not None:
                status, headers, out = other
                return self._send(status, out, headers)
            try:
                images = backend.draw(body["prompt"], body.get("n", 1), drawing.SIZES[body.get("size", "256x256")])
            except Exception as e:   # noqa: BLE001
                if _memory_full(e):
                    return self._send(503, _error(503, MEMORY))
                return self._send(500, _error(500, f"The server had an error: {type(e).__name__}: {e}"))
            drawn = drawing.said(drawing.read(body["prompt"]))
            self._send(200, {"created": int(time.time()),
                             "data": [{"b64_json": base64.b64encode(png).decode(), "revised_prompt": drawn} for png in images]})

        def _stream(self, backend, body, ident, created, runs=()):
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
                event({"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None},
                      **({"hosted_tools": list(runs)} if runs else {}))
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
                    event({"index": 0, "delta": {"tool_calls": _calls(a["tool_calls"], streamed=True, first=_earlier_calls(body["messages"]))},
                           "finish_reason": None})
                event({"index": 0, "delta": {}, "finish_reason": a["finish"]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    self.wfile.write(b"data: " + json.dumps({**head, "choices": [], "usage": _usage(a)}).encode("utf-8") + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self._entry["answer"] = events[0]
            except (BrokenPipeError, ConnectionResetError):   # the client stopped listening
                return

    return Handler


def _calls(calls: list, streamed: bool = False, first: int = 0) -> list:
    """Tool calls as OpenAI's API gives them, each with an id: call_<n>, counting on from `first`, the calls the chat
    had before, so that every call in a chat has an id of its own."""
    out = []
    for i, c in enumerate(calls):
        call = {"id": f"call_{first + i}", "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
        out.append({"index": i, **call} if streamed else call)
    return out


def _usage(a: dict) -> dict:
    out = {"prompt_tokens": a["prompt_tokens"], "completion_tokens": a["tokens"], "total_tokens": a["prompt_tokens"] + a["tokens"]}
    if "cached_tokens" in a:   # as OpenAI gives it: how many of the prompt's tokens came from the kept cache
        out["prompt_tokens_details"] = {"cached_tokens": a["cached_tokens"]}
    return out


def _earlier_calls(messages: list) -> int:
    """The tool calls in the assistant's messages of a chat."""
    return sum(len(m.get("tool_calls") or []) for m in messages if isinstance(m, dict) and m.get("role") == "assistant")


def _completion(a: dict, model: str, ident: str, created: int, first: int = 0) -> dict:
    message = {"role": "assistant", "content": a["text"] if a["text"] or not a["tool_calls"] else None}
    if a.get("reasoning") is not None:
        message["reasoning_content"] = a["reasoning"]
    if a["tool_calls"]:
        message["tool_calls"] = _calls(a["tool_calls"], first=first)
    choice = {"index": 0, "message": message, "finish_reason": a["finish"]}
    if a.get("logprobs") is not None:
        choice["logprobs"] = {"content": a["logprobs"]}
    return {"id": ident, "object": "chat.completion", "created": created, "model": model,
            "choices": [choice], "usage": _usage(a)}


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
