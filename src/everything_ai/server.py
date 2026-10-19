# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The start page and its small API, served by marimo's own server next to the lessons.

marimo loads this middleware through the entry point in pyproject.toml. It answers:
  /                        the start page (marimo's own file list is never shown)
  /course/static/<file>    the start page's script, styles and fonts
  /course/api/state        the route, your progress and settings, as JSON
  /course/api/settings     POST: language, theme, extra lessons on or off
  /course/api/reset/<id>   POST: start a lesson over (your progress stays)
  /course/open/<id>?lang=  make or refresh your copy of a lesson, then open it
Every other request goes to marimo. Everything here sits behind marimo's login token.
"""
from __future__ import annotations

import html
import json
import mimetypes
import urllib.parse

from starlette.middleware import Middleware

from everything_ai import course, paths, progress, system, workcopy

_STATIC = {"app.js", "app.css", "fonts.css", "kleuren.css"}
_THEMES = {"system", "light", "dark"}


def _authenticated(scope) -> bool:
    auth = scope.get("auth")
    return auth is not None and "read" in getattr(auth, "scopes", [])


async def _send(send, status: int, body: bytes, content_type: str, headers: list | None = None):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type.encode()), (b"cache-control", b"no-store"),
                            *(headers or [])]})
    await send({"type": "http.response.body", "body": body})


async def _json(send, data, status: int = 200):
    await _send(send, status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")


async def _redirect(send, location: str):
    await _send(send, 303, b"", "text/plain", [(b"location", location.encode())])


async def _body(receive) -> dict:
    chunks = []
    while True:
        msg = await receive()
        chunks.append(msg.get("body", b""))
        if not msg.get("more_body"):
            break
    try:
        return json.loads(b"".join(chunks) or b"{}")
    except ValueError:
        return {}


def state() -> dict:
    """Everything the start page shows, in one go."""
    lessons = course.lessons()
    built = set(course.built_lessons())
    done = progress.all_lessons()
    route = []
    for step in course.main_route():
        lid = step["id"]
        lesson = lessons[lid]
        entry = {
            "id": lid, "extra": step["extra"], "level": lesson["level"], "module": lesson["module"],
            "tak": lesson.get("tak"), "requires": lesson["requires"], "built": lid in built,
            "title": {"en": lesson["title"], "nl": None},
            "done": bool(done.get(lid, {}).get("done")), "opened": bool(done.get(lid, {}).get("opened")),
        }
        if lid in built:
            meta = course.lesson_meta(lid)
            entry["title"] = meta["title"]
            entry["minutes"] = meta.get("minutes")
            entry["copy"] = {lang: workcopy.state(lid, lang) for lang in course.LANGS}
        route.append(entry)
    disp = course.display()
    return {
        "route": route,
        "milestones": course.milestones(),
        "modules": disp["modules"],
        "levels": {lvl: {lang: course.level_name(lvl, lang) for lang in course.LANGS}
                   for lvl in ("basis", "gevorderd", "expert")},
        "branches": [{"id": b["id"], "title": {"en": b["title"], "nl": b["title_nl"]},
                      "status": b["status"], "order": b["build_order"],
                      "project": (b.get("project") or {}).get("title_nl")} for b in course.branches()],
        "goals": [{"id": g["id"], "lesson": g["goal"], "title": disp["goals"].get(g["id"], {"en": g["title_nl"], "nl": g["title_nl"]}),
                   "lessons": course.goal_route(g["goal"])} for g in course.goals()],
        "settings": progress.settings(),
        "system": _system(),
    }


def _system() -> dict:
    try:
        return json.loads(paths.SYSTEM.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = system.probe()
        paths.SYSTEM.parent.mkdir(parents=True, exist_ok=True)
        paths.SYSTEM.write_text(json.dumps(info), encoding="utf-8")
        return info


def _close_sessions(scope, lesson_id: str) -> None:
    """Close the lesson's open marimo sessions, so a fresh copy is not overwritten by the old one."""
    try:
        manager = scope["app"].state.session_manager
        targets = {str(workcopy.copy_path(lesson_id, lang).resolve()) for lang in course.LANGS}
        for session_id, session in list(manager.sessions.items()):
            path = getattr(getattr(session, "app_file_manager", None), "path", None)
            if path and str(paths.WORK / path if not str(path).startswith("/") else path) in targets:
                manager.close_session(session_id)
    except Exception:  # marimo internals; never block a reset on them
        pass


def _set_theme(scope, theme: str) -> None:
    """Hand the theme to marimo: it is in marimo's settings file for the course (.marimo.toml in the course folder)."""
    try:
        scope["app"].state.config_manager.save_config({"display": {"theme": theme}})
    except Exception:
        pass


class CourseMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        query = urllib.parse.parse_qs(scope.get("query_string", b"").decode())
        ours = path.startswith("/course/") or (path == "/" and "file" not in query and "access_token" not in query)
        if not ours:
            return await self.app(scope, receive, send)
        if not _authenticated(scope):
            if path == "/":
                return await self.app(scope, receive, send)   # marimo asks for the token
            return await _send(send, 401, b"Not signed in", "text/plain")
        method = scope["method"]

        if path == "/":
            token = html.escape(str(scope["app"].state.session_manager.skew_protection_token))
            page = (paths.WEB / "index.html").read_text(encoding="utf-8").replace(
                "</head>", f'<meta name="marimo-server-token" content="{token}">\n</head>', 1)
            return await _send(send, 200, page.encode(), "text/html; charset=utf-8")
        if path.startswith("/course/static/"):
            name = path.removeprefix("/course/static/")
            if name not in _STATIC:
                return await _send(send, 404, b"Not found", "text/plain")
            kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return await _send(send, 200, (paths.WEB / name).read_bytes(), kind)
        if path == "/course/api/state":
            return await _json(send, state())
        if path == "/course/api/settings" and method == "POST":
            body = await _body(receive)
            changes = {}
            if body.get("lang") in course.LANGS:
                changes["lang"] = body["lang"]
            if body.get("theme") in _THEMES:
                changes["theme"] = body["theme"]
                _set_theme(scope, body["theme"])
            if isinstance(body.get("extras"), bool):
                changes["extras"] = body["extras"]
            return await _json(send, progress.save_settings(**changes))
        if path.startswith("/course/api/reset/") and method == "POST":
            lid = path.removeprefix("/course/api/reset/")
            if not course.is_built(lid):
                return await _json(send, {"error": "unknown lesson"}, 404)
            _close_sessions(scope, lid)
            workcopy.reset(lid)
            return await _json(send, {"ok": True})
        if path.startswith("/course/open/"):
            lid = path.removeprefix("/course/open/")
            lang = (query.get("lang") or [progress.settings()["lang"]])[0]
            if lang not in course.LANGS:
                lang = "en"
            if not course.is_built(lid):
                return await _redirect(send, "/")
            workcopy.ensure(lid, lang)
            return await _redirect(send, "/?" + urllib.parse.urlencode({"file": workcopy.file_key(lid, lang)}))
        return await _send(send, 404, b"Not found", "text/plain")


MIDDLEWARE = Middleware(CourseMiddleware)
