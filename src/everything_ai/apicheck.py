# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""Checking code that calls a model's API, for the lessons of domain bouwen. The learner writes a function in a code
box; the check calls it against a stand-in of the server (everything_ai.serve.StandIn), which answers at once, and
reads what the server got. A function that the server answers with an error hears the server's own words in the
check's message, as it would from a real API. The picture shows the request the learner's code sent and what came
back (`exchange`)."""
import contextlib
import html
import io
import json
import os
import re
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable

from everything_ai import serve, yours
from everything_ai.check import (MISSING, TIME_LIMIT, CodeBox, Result, Run, Timeout, _fill, _ran, _skeleton_def, _tb_frames,
                                 is_empty_text, run_code, with_time_limit)

URL = "<url>"   # in a case's arguments: the stand-in's URL, such as http://127.0.0.1:50123/v1


@dataclass
class Case:
    """One call of the learner's function: against a fresh server for `backend()`, with `args` (URL is the server's
    URL) and the environment variables `env` set during the call. `judge(got, running)` says what is wrong: None when
    the value and the requests the server got are right, else the key of a text of the exercise; its text may use
    {got}, the value the function gave, and {call}. With `expect`, an error type, the call should stop with such an
    error: the judge then gets the error, or the value when there was none."""
    backend: Callable[[], serve.Backend]
    args: tuple
    judge: Callable[[Any, serve.Running], str | None]
    env: dict = field(default_factory=dict)
    shown_args: tuple | None = None   # how the call is written in a message, when not with the server's URL
    expect: type | None = None


def _shown(v) -> str:
    text = repr(v)
    return text if len(text) <= 120 else text[:117] + "..."


def call(fn, case: Case, time_limit: float = TIME_LIMIT):
    """Run `fn` on `case` against its own server: (what it gave, the error it raised or None, the server)."""
    running = serve.start(case.backend())
    old = {k: os.environ.get(k) for k in case.env}
    os.environ.update(case.env)
    try:
        args = tuple(running.url if a == URL else a for a in case.args)
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*args), time_limit), None, running
    except Exception as e:   # also Timeout
        return None, e, running
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        running.stop()


def server_said(e: urllib.error.HTTPError) -> str:
    """The message in the JSON of an error the server sent, or its reason."""
    return str(serve.server_error(e)["message"])


_HTTP = {"en": "With `{call}`, the server answered with status {code}: \"{message}\" urllib turns that into an error, HTTPError.",
         "nl": "Met `{call}` antwoordde de server met status {code}: \"{message}\" urllib maakt daar een fout van, HTTPError."}
_UNREACHABLE = {"en": "With `{call}`, your function could not reach the server ({why}). Does the URL start with the `{url}` the function gets, and is the rest of the address right?",
                "nl": "Met `{call}` kon je functie de server niet bereiken ({why}). Begint het adres met de `{url}` die de functie krijgt, en klopt de rest ervan?"}
_SLOW = {"en": "With `{call}`, your function is still not done after {s:g} seconds. Does it wait for something that never comes?",
         "nl": "Met `{call}` is je functie na {s:g} seconden nog niet klaar. Wacht hij op iets dat nooit komt?"}
_NOT_TO_URL = {"en": "With `{call}`, no request reached the server at `{url}`. Does your function send a request with `urlopen`, to an address it builds from the `{url}` it gets?",
               "nl": "Met `{call}` kwam er geen request aan bij de server op `{url}`. Stuurt je functie een request met `urlopen`, naar een adres dat hij maakt van de `{url}` die hij krijgt?"}


@dataclass
class ApiBox(CodeBox):
    """A function in a code box that calls a model's API, checked on `cases` against a stand-in server. Texts it can
    use: "http-<status>-<param>" for an error status the server sent about a field of the request (such as
    "http-404-model"), "http-<status>" for one about anything else (else a general text with the server's message),
    "unreachable", "slow", "no_function", "not-to-url" (no request reached the server, while the case went wrong), and
    the keys the cases' judges give."""
    function: str = ""
    cases: list[Case] = field(default_factory=list)
    time_limit: float = TIME_LIMIT
    url_name: str = "url"   # the name of the function's argument that gets the server's URL

    def _call_text(self, case: Case) -> dict:
        """The call as a message shows it, per language. Without `shown_args`, the server's URL is written as the function's argument
        for it (`url_name`), and every other argument as Python writes it; `shown_args` are written as they are, and each may be a dict per
        language, such as {"en": "[the conversation]", "nl": "[het gesprek]"}."""
        if case.shown_args is None:
            args = [self.url_name if a == URL else repr(a) for a in case.args]
            return {lang: f"{self.function}({', '.join(args)})" for lang in ("en", "nl")}
        return {lang: f"{self.function}({', '.join(a[lang] if isinstance(a, dict) else str(a) for a in case.shown_args)})"
                for lang in ("en", "nl")}

    def _not_to_url(self, running, c) -> Result | None:
        """When no request reached the server: the text that says so."""
        if running.log:
            return None
        return Result("wrong", _fill(self.texts.get("not-to-url", _NOT_TO_URL), call=c, url=self.url_name), mistake="not-to-url")

    def check(self, text, inputs=None) -> Result:
        if text is MISSING or text is None:
            return Result("error", self.message("missing"))
        if is_empty_text(text, self.start) or _skeleton_def(text, self.function):
            cleared = not text.strip() and text.strip() != self.start.strip() and "cleared" in self.texts
            return Result("empty", self.message("cleared" if cleared else "empty"))
        for m in self.first:
            if m.matches(None, text):
                return Result(m.state, _fill(m.text, call=self.function), mistake=m.key)
        names = dict(self.names if inputs is None else inputs)
        run = run_code(text, names)
        _ran(self, text, names, run)
        if run.timeout:
            return Result("slow", self.message("slow", call=self.function, s=self.time_limit) if "slow" in self.texts else
                          _fill(_SLOW, call=self.function, s=self.time_limit))
        if run.error is not None:
            for m in self.mistakes:
                if m.error is not None and m.error(run.error):
                    return Result("error", _fill(m.text, call=self.function), mistake=m.key)
            return Result("error", self._plain_error(run, text))
        fn = run.names.get(self.function)
        if not callable(fn):
            return Result("wrong", self.message("no_function", function=self.function) if "no_function" in self.texts else {
                "en": f"Your box does not make a function `{self.function}`. Start it with `def {self.function}(...):`, with that exact name.",
                "nl": f"Je vak maakt geen functie `{self.function}`. Begin met `def {self.function}(...):`, met precies die naam."},
                mistake="no-function")
        for case in self.cases:
            got, error, running = call(fn, case, self.time_limit)
            c = self._call_text(case)
            if isinstance(error, Timeout):
                return self._not_to_url(running, c) or Result("slow", _fill(self.texts.get("slow", _SLOW), call=c, s=self.time_limit))
            if case.expect is not None and (error is None or isinstance(error, case.expect)):
                key = case.judge(got if error is None else error, running)
                if key is not None:
                    return self._not_to_url(running, c) or Result("almost" if key == "almost" else "wrong",
                                                                  self.message(key, call=c, got=_shown(got)), mistake=key)
                continue
            if isinstance(error, urllib.error.HTTPError):
                if not running.log:   # another server answered
                    return self._not_to_url(running, c)
                said = serve.server_error(error)
                values = {"call": c, "code": error.code, "message": str(said["message"])}
                for key in ([f"http-{error.code}-{said['param']}"] if said.get("param") else []) + [f"http-{error.code}"]:
                    if key in self.texts:
                        return Result("error", self.message(key, **values), mistake=key)
                return Result("error", _fill(_HTTP, **values), mistake="http")
            if isinstance(error, urllib.error.URLError) and not isinstance(getattr(error, "reason", None), TimeoutError):
                return Result("error", _fill(self.texts.get("unreachable", _UNREACHABLE), call=c, why=str(error.reason), url=self.url_name), mistake="unreachable")
            if error is not None:
                for m in self.mistakes:
                    if m.error is not None and m.error(error):
                        return Result("error", _fill(m.text, call=c), mistake=m.key)
                frames = [fr for fr in _tb_frames(error) if fr[0] == "<your code>"]
                plain = self._plain_error(Run(error=error, line=frames[-1][1] if frames else None), text)
                return Result("error", {"en": f"With `{c['en']}`, your function stops with an error. " + plain["en"],
                                        "nl": f"Met `{c['nl']}` stopt je functie met een fout. " + plain["nl"]})
            key = case.judge(got, running)
            if key is not None:
                return self._not_to_url(running, c) or Result("almost" if key == "almost" else "wrong",
                                                              self.message(key, call=c, got=_shown(got)), mistake=key)
        for m in self.form:
            if m.matches(None, text):
                return Result(m.state, _fill(m.text, call=self.function), mistake=m.key)
        return self._correct(text, n=len(self.cases))


# --- the picture: the request the learner's code sent, and the answer ----------------------------------------------

def _json_html(value, limit: int = 900, answer: bool = False) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > limit:
        text = text[:limit - 3] + "..."
    style = "color:var(--ea-pred-text)" if answer else "color:var(--ea-ink)"
    return (f'<pre style="margin:0.2rem 0 0.6rem;padding:0.5rem 0.6rem;background:var(--ea-inset);border-radius:6px;'
            f'font-family:var(--ea-font-code);font-size:12px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere;{style}">'
            f'{html.escape(text)}</pre>')


def _label(text: str) -> str:
    return f'<div style="font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)">{html.escape(text)}</div>'


def trimmed(value, words: dict | None = None):
    """A request or an answer as a picture shows it: a data URL or a long text in base64, such as an image, cut after
    its start, and a list of more than eight numbers, such as an embedding, cut after four, each with how long it was.
    `words` has "data" and "numbers", with {n}, in the lesson's language, and "thousands", the sign between thousands
    in it ("," or ".")."""
    words = {"data": "… {n} characters in all", "numbers": "… {n} numbers in all", "thousands": ",", **(words or {})}
    sep = words["thousands"]
    if isinstance(value, str) and value.startswith("data:") and len(value) > 80:
        return value[:60] + words["data"].format(n=f"{len(value):,}".replace(",", sep))
    if isinstance(value, str) and len(value) > 200 and re.fullmatch(r"[A-Za-z0-9+/]+=*", value):
        return value[:24] + words["data"].format(n=f"{len(value):,}".replace(",", sep))
    if isinstance(value, list) and len(value) > 8 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return value[:4] + [words["numbers"].format(n=f"{len(value):,}".replace(",", sep))]
    if isinstance(value, list):
        return [trimmed(v, words) for v in value]
    if isinstance(value, dict):
        return {k: trimmed(v, words) for k, v in value.items()}
    return value


def exchange(seen, fn_name: str, case: Case, words: dict, entries: int = 1) -> str:
    """The first `entries` requests the learner's function sent to the stand-in in `case`, each with what came back:
    a picture of their own request. `words` (in the lesson's language) has "title", "request" ({method} and
    {path}), "headers", "answer" ({status}), "stream" ({n}) and "none" (nothing was sent), and may have "data" and
    "numbers" for `trimmed`. "" while the code does not run or makes no such function."""
    run = seen.run
    fn = run.names.get(fn_name) if run is not None and run.error is None else None
    if not callable(fn):
        return ""
    _got, _error, running = call(fn, case)
    if not running.log:
        return yours.figure(words["title"], _label(words["none"]), alt=words["title"])
    body = []
    for entry in running.log[:entries]:
        body.append(_label(words["request"].format(method=entry["method"], path=entry["path"])))
        shown = {k: ("Bearer …" if k == "authorization" else v) for k, v in entry["headers"].items()
                 if k in ("content-type", "authorization")}
        if shown:
            body.append(_label(words["headers"] + " " + ", ".join(f"{k}: {v}" for k, v in shown.items())))
        if entry["body"] not in (None, ""):
            body.append(_json_html(trimmed(entry["body"], words)))
        answer = entry.get("answer")
        body.append(_label(words["answer"].format(status=entry["status"])))
        if isinstance(answer, int):
            body.append(_label(words["stream"].format(n=answer)))
        elif answer is not None:
            body.append(_json_html(trimmed(answer, words), answer=entry["status"] == 200))
    return yours.figure(words["title"], "".join(body), alt=words["title"])

