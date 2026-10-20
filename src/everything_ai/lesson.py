# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: MIT
"""The helper every lesson uses: `L = lesson("<id>", "en")` in the lesson's setup cell.

It builds the parts every lesson shares from the route data, so a lesson writer never
writes them by hand: the header (place, title, goal, what you need, what you should know
first), questions, code boxes, checks, hints, the wrap-up and the footer. It also keeps
your progress. The lesson's own text stays in the lesson file; its checks live in
lessons/<id>/checks.py.
"""
from __future__ import annotations

import functools
import html
import importlib.util
import json
import re
import types
import urllib.parse

import marimo as mo

from everything_ai import __version__, chart, check as checking, course, play, plots, progress, projects, style, widgets
from everything_ai.check import MISSING  # noqa: F401  (lessons use L.MISSING)

REPO_URL = "https://github.com/QRRV/everything-ai"

_WORDS = {
    "en": {
        "about_minutes": "about {n} minutes", "extra": "extra", "home": "Start page", "lessons_nav": "Lessons",
        "other_lang": "Nederlands", "first_lesson": "This is the first lesson: you need nothing from before.",
        "works_here": "Works on every computer; nothing is downloaded.",
        "works_paper": "Works on every computer; the only download is the paper, which you get yourself.",
        "works_hub": "Works on every computer. The model is downloaded once from Hugging Face, about 200 kB; without a connection, the course's own copy of it is used.",
        "works_download": "Works on every computer. {name} is downloaded once from Hugging Face, {size} MB, when you click its button in the lesson; after that, the lesson works without a connection. The exercises work without {name} too.",
        "needs_gpu": "Training: about {m} minutes on the recommended GPU; it needs a GPU with 8 GB or more.",
        "gpu_fits": "Your GPU, {name}, can do it; nothing is downloaded.",
        "gpu_mac": "On your Mac it trains on the GPU of the Apple chip, about 4 to 5 times slower than on the recommended GPU; nothing is downloaded.",
        "gpu_cpu": "Your computer has no GPU for it, so a smaller version runs on the CPU instead; nothing is downloaded.",
        "needs_nvidia": "It needs Linux, or WSL2 on Windows, with an NVIDIA GPU, for Triton.",
        "triton_cpu": "Your computer has no NVIDIA GPU, so Triton runs your kernel on the CPU in its interpreter: right, but slow; nothing is downloaded.",
        "triton_none": "Triton does not run on your computer: you read along, and the lesson is done without the exercise that runs a kernel.",
        "task": "Task", "your_guess": "Your guess", "compare": "Compare with a model answer",
        "tick": "Tick each point that is in your answer. Nobody grades this; it is for you.",
        "first_yourself": "Try it yourself first; that is how you learn the most.",
        "lesson_done": "This lesson is done.", "lesson_not_done": "This lesson is done once every check above says Correct.",
        "coming": "coming soon", "report_text": "Something not right, or unclear? Report it",
        "report_link": "on GitHub", "self_tick": "you tick this yourself at the open question",
        "solution_code": "The solution:", "your_answer": "Type your answer here",
        "done_before": "done", "not_done_before": "not done yet",
        "check_name": "The check", "missing_check": "This exercise has no check yet.",
    },
    "nl": {
        "about_minutes": "ongeveer {n} minuten", "extra": "verdieping", "home": "Startpagina", "lessons_nav": "Lessen",
        "other_lang": "English", "first_lesson": "Dit is de eerste les: je hebt niets van eerder nodig.",
        "works_here": "Werkt op elke computer; er wordt niets gedownload.",
        "works_paper": "Werkt op elke computer; je downloadt alleen de paper, en dat doe je zelf.",
        "works_hub": "Werkt op elke computer. Het model wordt één keer gedownload van Hugging Face, ongeveer 200 kB; zonder verbinding gebruikt de course zijn eigen kopie ervan.",
        "works_download": "Werkt op elke computer. {name} wordt één keer gedownload van Hugging Face, {size} MB, als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder {name}.",
        "needs_gpu": "Trainen: ongeveer {m} minuten op de aanbevolen GPU; er is een GPU met 8 GB of meer voor nodig.",
        "gpu_fits": "Je GPU, {name}, kan het; er wordt niets gedownload.",
        "gpu_mac": "Op je Mac traint het op de GPU van de Apple-chip, ongeveer 4 tot 5 keer trager dan op de aanbevolen GPU; er wordt niets gedownload.",
        "gpu_cpu": "Je computer heeft er geen GPU voor, dus draait er een kleinere versie op de CPU; er wordt niets gedownload.",
        "needs_nvidia": "Triton heeft Linux nodig, of WSL2 op Windows, met een NVIDIA-GPU.",
        "triton_cpu": "Je computer heeft geen NVIDIA-GPU, dus draait Triton je kernel op de CPU in zijn interpreter: goed, maar traag; er wordt niets gedownload.",
        "triton_none": "Triton draait niet op je computer: je leest mee, en de les is af zonder de oefening die een kernel draait.",
        "task": "Opdracht", "your_guess": "Jouw gok", "compare": "Vergelijk met een modelantwoord",
        "tick": "Vink aan welke punten in jouw antwoord staan. Niemand beoordeelt dit; het is voor jezelf.",
        "first_yourself": "Probeer het eerst zelf; daar leer je het meest van.",
        "lesson_done": "Deze les is af.", "lesson_not_done": "Deze les is af zodra elke check hierboven Goed zegt.",
        "coming": "komt nog", "report_text": "Klopt er iets niet, of is iets onduidelijk? Meld het",
        "report_link": "op GitHub", "self_tick": "dit vink je zelf af bij de open vraag",
        "solution_code": "De oplossing:", "your_answer": "Typ hier je antwoord",
        "done_before": "af", "not_done_before": "nog niet af",
        "check_name": "De check", "missing_check": "Deze oefening heeft nog geen check.",
    },
}


def _md(text: str) -> str:
    """Markdown to HTML, the way marimo renders it (for text inside our own blocks)."""
    return mo.md(text).text


_WRAP = '<span class="markdown prose dark:prose-invert contents">'


def _inline(text: str) -> str:
    """Markdown to HTML without marimo's prose wrapper, for short text inside a check, an option or a list item.
    One paragraph stays inline; with more blocks, each paragraph becomes a <p> element."""
    h = _md(text)
    if h.startswith(_WRAP) and h.endswith("</span>"):
        h = h[len(_WRAP):-len("</span>")]
    one = re.fullmatch(r'<span class="paragraph">(.*?)</span>', h, flags=re.S)
    if one and '<span class="paragraph">' not in one.group(1):
        return one.group(1)
    # More blocks, such as paragraphs around a code block: each paragraph ends where the next block starts.
    return re.sub(r'<span class="paragraph">(.*?)</span>(?=\n<|$)', r"<p>\1</p>", h, flags=re.S)


def _plain_data(value) -> bool:
    """Can it be kept in the progress file? A name in the notebook can hold anything, such as `...`."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# The note when a lesson uses the course's version of a project module instead of yours (template 1.3).
_COURSE_VERSION = {
    "intro": {
        "en": "For now, this lesson uses the course's version of these project files.",
        "nl": "Deze les gebruikt voorlopig de versie van de course voor deze projectbestanden.",
    },
    "kept": {
        "en": "It leaves your own files as they are.",
        "nl": "Je eigen bestanden laat de les zoals ze zijn.",
    },
    "missing": {
        "en": "Your {path} is not there yet: you write it in {link}.",
        "nl": "Je {path} staat er nog niet: dat schrijf je in {link}.",
    },
    "broken": {
        "en": "Your {path} does not give the results the course expects, or stops with an error: repair it in {link}, and this lesson uses yours again.",
        "nl": "Je {path} geeft niet de uitkomsten die de course verwacht, of stopt met een fout: repareer het in {link}, dan gebruikt deze les weer jouw versie.",
    },
    # a module that an earlier exercise of this same lesson saves
    "missing-here": {
        "en": "Your {path} is not there yet: the check of an exercise in this lesson saves it when it is right.",
        "nl": "Je {path} staat er nog niet: de check van een oefening in deze les slaat het op als die goed is.",
    },
    "broken-here": {
        "en": "Your {path} does not give the results the course expects, or stops with an error: repair it, or delete it and check its exercise in this lesson again.",
        "nl": "Je {path} geeft niet de uitkomsten die de course verwacht, of stopt met een fout: repareer het, of verwijder het en check de oefening ervan in deze les opnieuw.",
    },
}


@functools.cache
def _computer() -> dict:
    """The quick look at this computer from system.py, once per process: it may ask nvidia-smi."""
    from everything_ai import system
    return system.probe()


def _open_url(lesson_id: str, lang: str) -> str:
    return f"/course/open/{lesson_id}?lang={lang}"


class Lesson:
    """One lesson in one language."""

    MISSING = checking.MISSING

    def __init__(self, lesson_id: str, lang: str):
        assert lang in course.LANGS, lang
        self.id, self.lang = lesson_id, lang
        self.route = course.lessons()[lesson_id]
        self.meta = course.lesson_meta(lesson_id)
        self.w = _WORDS[lang]
        self.checks = _load_checks(lesson_id, lang)
        progress.mark_opened(lesson_id, lang)
        projects.on_path()             # your own modules can be imported in every lesson
        projects.provide(lesson_id)    # and a lesson can give you starting files
        plots.when_imported()          # matplotlib figures in the course's style, light and dark

    # --- small pieces ---------------------------------------------------------------

    def number(self, x: float, digits: int = 2) -> str:
        """A number for running text, with the decimal mark of the language and a real minus sign."""
        text = f"{x:.{digits}f}".replace("-", "\u2212")
        return text.replace(".", ",") if self.lang == "nl" else text

    def t(self, text: dict | str) -> str:
        """Pick this lesson's language from {"en": ..., "nl": ...}."""
        return text[self.lang] if isinstance(text, dict) else text

    def phrase(self, key: str, **values) -> str:
        return course.phrase(key, self.lang, **values)

    def task(self, text: str) -> mo.Html:
        """One short assignment: TASK / OPDRACHT, then one sentence."""
        return mo.Html(style.task(self.w["task"], _inline(text)))

    def watch_out(self, text: str) -> mo.Html:
        """A 'watch out' note where a word can be misread."""
        return mo.Html(style.note(_inline(text), title=self.phrase("watch-out") + ":"))

    def note(self, text: str) -> mo.Html:
        return mo.Html(style.note(_inline(text)))

    def course_versions(self) -> mo.Html:
        """A note when a project module this lesson builds on is missing or broken, so the lesson uses the course's
        version for now (template 1.3): which file, and which lesson to go back to. Nothing when all are yours."""
        lines, states = [], set()
        for package, module, state, lesson in projects.stand_ins():
            states.add(state)
            path = f"`{projects.shown(projects.folder() / package / (module + '.py'))}`"
            link = f"[*{course.title(lesson, self.lang)}*]({_open_url(lesson, self.lang)})"
            key = state + "-here" if lesson == self.id else state
            lines.append("- " + _COURSE_VERSION[key][self.lang].format(path=path, link=link))
        if not lines:
            return mo.md("")
        intro = _COURSE_VERSION["intro"][self.lang]
        if "broken" in states:
            intro += " " + _COURSE_VERSION["kept"][self.lang]
        return self.note(intro + "\n\n" + "\n".join(lines))

    def formula(self, tex: str) -> mo.Html:
        """A display formula in KaTeX; `[[param:w]]` colours a symbol by its role, in the light and the dark theme."""
        return mo.Html(style.formula(tex))

    # --- A. the header ------------------------------------------------------------------

    def header(self) -> mo.Html:
        s = progress.settings()
        numbers, total = course.numbering(extras=s["extras"])
        level = course.level_name(self.route["level"], self.lang)
        place = []
        if self.id in numbers:
            place.append(self.phrase("block-place", n=numbers[self.id], total=total))
        place.append(f'<span class="ea-level">{html.escape(level)}</span>')
        place.append(self.w["about_minutes"].format(n=self.meta["minutes"]))
        other = "nl" if self.lang == "en" else "en"
        kicker = (f'<div class="ea-kicker ea-topline">{self._home_button()}'
                  f'<span>{" · ".join(place)}</span>'
                  f'<a href="{_open_url(self.id, other)}" hreflang="{other}">{self.w["other_lang"]}</a></div>')
        goal = f'<p class="ea-goal">{html.escape(self.phrase("block-goal").rstrip(" …"))} {_inline(self.t(self.meta["goal"]))}</p>'
        needs = (f'<p class="ea-needs"><b>{self.phrase("block-needs")}:</b> '
                 f'{self.w["about_minutes"].format(n=self.meta["minutes"])}. {self._needs_text()}</p>')
        # a lesson the route marks as hard says so up front (lesson template, block 3)
        tough = f'<p class="ea-needs ea-tough">{html.escape(self.phrase("block-tough"))}</p>' if self.route.get("hard") else ""
        return mo.Html(kicker + f'<h1 class="ea-title">{html.escape(self.t(self.meta["title"]))}</h1>'
                       + goal + needs + tough + self._prerequisites())

    def _needs_text(self) -> str:
        """What the lesson needs, and what that means on this computer (template 2.3). The course's data is in the
        course folder, so nothing is downloaded."""
        needs = self.route.get("needs", "any")
        if needs not in ("gpu8", "nvidia-linux"):
            # a paper (data id paper-…) is the one thing such a lesson asks you to download, by hand
            paper = any(str(d).startswith("paper-") for d in self.route.get("data", []))
            if getattr(self.checks, "HUB_REPOS", None):   # a lesson that downloads a model of the course from the hub
                return self.w["works_hub"]
            line = getattr(self.checks, "DOWNLOAD_LINE", None)   # a lesson that says in its own words what it downloads
            if line:
                return line[self.lang]
            downloads = getattr(self.checks, "DOWNLOADS", None)   # a lesson that downloads a larger model when asked
            if downloads:
                from everything_ai import hub
                name, pin = next(iter(downloads.items()))
                return self.w["works_download"].format(name=name, size=round(hub.PINNED[pin]["size"] / 1e6))
            return self.w["works_paper" if paper else "works_here"]   # other needs get their own line when their first lesson is built
        gpu = _computer()["gpu"] or {}   # may ask nvidia-smi: only for a lesson that needs a GPU
        if needs == "nvidia-linux":
            if importlib.util.find_spec("triton") is None:
                here = self.w["triton_none"]
            elif gpu.get("kind") == "cuda":
                here = self.w["gpu_fits"].format(name=html.escape(gpu["name"]))
            else:
                here = self.w["triton_cpu"]
            return self.w["needs_nvidia"] + " " + here
        if gpu.get("kind") == "cuda" and (gpu.get("vram_gb") or 0) >= 7:
            here = self.w["gpu_fits"].format(name=html.escape(gpu["name"]))
        elif gpu.get("kind") == "mps":
            here = self.w["gpu_mac"]
        else:
            here = self.w["gpu_cpu"]
        return self.w["needs_gpu"].format(m=self.route.get("compute_minutes", "?")) + " " + here

    def _prerequisites(self) -> str:
        req = self.route.get("requires", [])
        if not req:
            return f'<p class="ea-needs"><b>{self.phrase("block-prerequisites")}:</b> {self.w["first_lesson"]}</p>'
        done = progress.all_lessons()
        items = []
        for rid in req:
            ok = done.get(rid, {}).get("done")
            mark = style.icon("tick") if ok else style.icon("box")
            state = self.w["done_before"] if ok else self.w["not_done_before"]
            items.append(f'<li><a href="{_open_url(rid, self.lang)}">{html.escape(course.title(rid, self.lang))}</a>'
                         f' <span class="ea-prereq-state">{mark}<span class="ea-sr">{state}</span></span></li>')
        return (f'<div class="ea-needs"><b>{self.phrase("block-prerequisites")}:</b>'
                f'<ul class="ea-prereqs">{"".join(items)}</ul></div>')

    # --- questions -------------------------------------------------------------------------

    def _question(self, kind: str, key: str, question: str, options: dict, **extra):
        w = widgets.Question(
            kind=kind, question=_md(question),
            options=[{"key": k, "html": _inline(v)} for k, v in options.items()],
            words={"correct": self.phrase("check-correct"), "not_yet": self.phrase("check-not-yet"),
                   "your_guess": self.w["your_guess"]},
            value=progress.saved_answer(self.id, key) or "", **extra)

        def _save(change, _key=key):
            progress.record_answer(self.id, _key, change["new"])
            if kind == "choice" and _key in self.required:
                state = "correct" if change["new"] == w.correct else "wrong"
                progress.record_exercise(self.id, _key, state, self.required)
        w.observe(_save, names="value")
        w._ea_key = key
        return mo.ui.anywidget(w)

    def guess(self, key: str, question: str, options: dict, reveal: str):
        """Guess first: you commit to an answer, then see what really happens next to it. Never graded."""
        return self._question("guess", key, question, options, reveal=_md(reveal),
                              label=self.phrase("block-guess"))

    def choice(self, key: str, question: str, options: dict, correct: str, why: dict):
        """A choice question. Every option has its own explanation, the right one too."""
        assert correct in options and set(why) == set(options), key
        return self._question("choice", key, question, options, correct=correct,
                              feedback={k: _inline(v) for k, v in why.items()})

    def explain(self, key: str, question: str, model_answer: str, points: list[str]):
        """An open question: you write, then compare with a model answer and tick its points yourself."""
        w = widgets.Explain(question=_md(question), placeholder=self.w["your_answer"],
                            model_answer=_inline(model_answer), points=[_inline(p) for p in points],
                            words={"compare": self.w["compare"], "tick": self.w["tick"]},
                            value=progress.saved_answer(self.id, key) or {})
        w.observe(lambda change: progress.record_answer(self.id, key, change["new"]), names="value")
        w._ea_key = key
        return mo.ui.anywidget(w)

    # --- D. exercises ---------------------------------------------------------------------

    @functools.cached_property
    def required(self) -> list[str]:
        return list(getattr(self.checks, "REQUIRED", []))

    def exercise(self, name: str) -> checking.Exercise:
        return self.checks.EXERCISES[name]

    def code_box(self, name: str, lines: int = 3):
        """A box to type code in. Your text is kept, also when you close the lesson or switch language.
        A box you emptied starts again with the course's start when you open the lesson again."""
        ex = self.exercise(name)
        saved = progress.saved_text(self.id, name)
        return mo.ui.code_editor(value=saved if saved and saved.strip() else ex.start, language="python",
                                 min_height=24 * lines + 16, debounce=600, show_copy_button=False)

    def sort(self, name: str, question: str = ""):
        """Put every item of a Sort exercise in a group. Your choices are kept."""
        ex = self.exercise(name)
        w = widgets.Sort(question=_md(question) if question else "",
                         items=[{"key": k, "html": _inline(self.t(v))} for k, v in ex.items.items()],
                         groups=[{"key": k, "html": _inline(self.t(v))} for k, v in ex.groups.items()],
                         value=self.saved(name, {}))
        w._ea_key = name
        return mo.ui.anywidget(w)

    def slider(self, key: str, label: str, start: float, low: float, high: float, step: float, role: str = "param"):
        """A slider in the colour of its role, with its name and value. Where you left it is kept.
        Its value is `slider.widget.value`; L.check takes the slider itself."""
        value = progress.saved_answer(self.id, key)
        if not isinstance(value, (int, float)):
            value = start
        # a value kept from before stays on the slider, also when its range or steps changed since
        value = round(low + round((min(max(value, low), high) - low) / step) * step, 10)
        w = widgets.Slider(role=role, label=label, min=low, max=high, step=step,
                           value=value, decimal="," if self.lang == "nl" else ".")
        w.observe(lambda change: progress.record_answer(self.id, key, change["new"]), names="value")
        w._ea_key = key
        return mo.ui.anywidget(w)

    def explore(self, key: str, label: str, frame, start: float, low: float | None = None, high: float | None = None,
                step: float | None = None, role: str = "param", task: str | None = None, values: list | None = None):
        """A slider with a picture that follows it at once (play.Explore): `frame(v)` gives, for every stop of the
        slider, the picture as SVG text (from chart.figure) and the sentence under it in Markdown, as in the lesson's
        text, so code and formulas work there too; they are drawn once, up front. In Dutch the numbers along the axes
        get a decimal comma. With `task`, the block opens with it. The stops are low to high in steps of step, or the
        `values` given. Where you left it is kept, as with L.slider; L.check takes the block itself."""
        value = progress.saved_answer(self.id, key)
        if not isinstance(value, (int, float)):
            value = start
        nl = self.lang == "nl"

        def drawn(v):
            out = frame(v)
            svg, said = (out, "") if isinstance(out, str) else out
            return chart.with_decimal_comma(svg) if nl else svg, _inline(said) if said else said

        labels = {"name": label} | ({"task_kick": self.w["task"], "task": _inline(task)} if task else {})
        w = play.Explore(drawn, low=low, high=high, step=step, values=values, value=value, role=role, labels=labels, decimal="," if nl else ".")
        w.observe(lambda change: progress.record_answer(self.id, key, w.value), names="index")
        w._ea_key = key
        return mo.ui.anywidget(w)

    def saved(self, name: str, default=None):
        """What you set last time in an exercise that is not code, such as a slider or a sort."""
        value = progress.saved_answer(self.id, name)
        return default if value is None else value

    def check(self, name: str, value, inputs: tuple | None = None) -> "CheckResult":
        """Check one exercise, remember the outcome, and show the message. For a widget such as
        L.sort, pass the widget itself: marimo's value of a widget is all its state, not the answer.
        `inputs` are the current values of the lesson's names that the exercise uses."""
        ex = self.exercise(name)
        if isinstance(getattr(value, "widget", None), widgets.Sort):
            value = dict(value.widget.value)
        elif isinstance(getattr(value, "widget", None), (widgets.Slider, play.Explore)):
            value = value.widget.value
        try:
            result = ex.check(value) if inputs is None else ex.check(value, inputs)
        except Exception:  # a mistake in the check itself, never the learner's fault
            result = checking.Result("broken", {"en": "", "nl": ""})
        text = value if isinstance(ex, checking.CodeBox) and isinstance(value, str) else None
        if result.state != "broken":
            progress.record_exercise(self.id, name, result.state, self.required, text=text)
            if isinstance(ex, (checking.Value, checking.Sort)) and value is not checking.MISSING and _plain_data(value):
                progress.record_answer(self.id, name, value)
        return CheckResult(self, name, result)

    def hints(self, name: str, hints: list[str], why: str):
        """Hints that each say a bit more, then the solution with its reasoning."""
        ex = self.exercise(name)
        if isinstance(ex.solution, str):
            solution = f'<p>{self.w["solution_code"]}</p><pre>{html.escape(ex.solution)}</pre><p>{_inline(why)}</p>'
        else:   # a slider or a sort: the reasoning says what the solution is
            solution = _md(why)
        opened = progress.lesson(self.id).get("exercises", {}).get(name, {}).get("hints_opened", 0)
        w = widgets.Hints(hints=[_inline(h) for h in hints], solution=solution, opened=opened,
                          words={"hint": self.phrase("hint", n="{n}"), "solution": self.phrase("solution"),
                                 "first_yourself": self.w["first_yourself"]})

        def _save(change):
            def _set(d):
                ex_d = d.setdefault("exercises", {}).setdefault(name, {})
                ex_d["hints_opened"] = change["new"]
                if change["new"] > len(hints):
                    ex_d["solution_seen"] = True
            progress.update(self.id, _set)
        w.observe(_save, names="opened")
        return mo.ui.anywidget(w)

    # --- E. wrap-up and footer -----------------------------------------------------------

    def wrap_up(self, *depends_on) -> mo.Html:
        """Summary and 'You can now', ticked by the checks and answers. Pass the check results and
        question widgets of the lesson, so the list updates the moment one of them changes."""
        p = progress.lesson(self.id)
        exercises, answers = p.get("exercises", {}), dict(p.get("answers", {}))
        for dep in depends_on:   # the newest values, also when the progress file is a moment behind
            w = getattr(dep, "widget", None)
            if w is not None and hasattr(w, "_ea_key"):
                answers[w._ea_key] = w.value

        def ok(by: str) -> bool:
            if by in exercises and exercises[by].get("ever_correct"):
                return True
            answer = answers.get(by)
            if isinstance(answer, dict):
                return bool(answer.get("compared"))
            for dep in depends_on:
                w = getattr(dep, "widget", None)
                if getattr(w, "_ea_key", None) == by and isinstance(w, widgets.Question):
                    return w.kind == "choice" and w.value == w.correct
            return False

        points = "\n".join(f"- {self.t(s)}" for s in self.meta.get("summary", []))
        items = []
        for item in self.meta.get("can_now", []):
            tick = ok(item.get("by"))
            items.append(f'<li class="{"ea-ok" if tick else ""}">{style.icon("tick" if tick else "box")}'
                         f'<span>{_inline(self.t(item))}</span></li>')
        done = bool(p.get("done")) or all(ok(r) for r in self.required)
        status = self.w["lesson_done"] if done else self.w["lesson_not_done"]
        return mo.md(
            f'### {self.phrase("block-summary")}\n\n{points}\n\n'
            f'**{self.phrase("block-you-can-now")}**\n\n'
            f'<ul class="ea-cannow">{"".join(items)}</ul>\n\n'
            f'<p class="ea-needs">{status}</p>\n')

    def _home_button(self) -> str:
        return f'<a class="ea-home" href="/">{style.icon("home")}<span>{self.w["home"]}</span></a>'

    def _neighbours(self) -> tuple[str | None, str | None]:
        """The lessons before and after this one in the route the learner follows (with or without extra lessons)."""
        s = progress.settings()
        steps = [st["id"] for st in course.main_route() if s["extras"] or not st["extra"]]
        if self.id not in steps:
            return None, None
        i = steps.index(self.id)
        return (steps[i - 1] if i > 0 else None), (steps[i + 1] if i + 1 < len(steps) else None)

    def _pager_link(self, lesson_id: str | None, where: str) -> str:
        """One half of the row at the bottom: the previous or the next lesson, a link once it is built."""
        if lesson_id is None:
            return ""
        label = (f'← {self.phrase("block-previous")}' if where == "prev" else f'{self.phrase("block-next")} →')
        title = html.escape(course.title(lesson_id, self.lang))
        if course.is_built(lesson_id):
            return (f'<a class="ea-pager-{where}" href="{_open_url(lesson_id, self.lang)}">'
                    f'<span class="ea-pager-label">{label}</span><span class="ea-pager-title">{title}</span></a>')
        return (f'<div class="ea-pager-{where} ea-pager-coming"><span class="ea-pager-label">{label}</span>'
                f'<span class="ea-pager-title">{title}</span><span class="ea-pager-note">{self.w["coming"]}</span></div>')

    def footer(self) -> mo.Html:
        prev, nxt = self._neighbours()
        parts = []
        reading = self.meta.get("further_reading", [])
        if reading:
            links = "".join(f'<li>{_md(self.t(r))}</li>' for r in reading)
            parts.append(f'<p><b>{self.phrase("block-further-reading")}</b></p><ul>{links}</ul>')
        issue = urllib.parse.urlencode({
            "title": f"[{self.id}] ",
            "body": f"Lesson: {self.id}\nLanguage: {self.lang}\nCourse version: {__version__}\n\nWhat happened:\n",
        })
        parts.append(f'<p class="ea-needs">{self.w["report_text"]} '
                     f'<a href="{REPO_URL}/issues/new?{issue}" target="_blank" rel="noopener">{self.w["report_link"]}</a>.</p>')
        if prev or nxt:
            parts.append(f'<nav class="ea-pager" aria-label="{self.w["lessons_nav"]}">'
                         f'{self._pager_link(prev, "prev")}{self._pager_link(nxt, "next")}</nav>')
        parts.append(f'<p class="ea-footer-home">{self._home_button()}</p>')
        return mo.Html('<div class="ea-footer">' + "".join(parts) + "</div>")


class CheckResult:
    """What a check cell shows. It is also a value, so 'You can now' updates with it."""

    def __init__(self, lesson_: Lesson, name: str, result: checking.Result):
        self.lesson, self.name, self.result = lesson_, name, result
        self.state, self.mistake = result.state, result.mistake

    def _mime_(self):
        L = self.lesson
        words = {"empty": "check-empty", "error": "check-code-error", "wrong": "check-not-yet",
                 "almost": "check-almost", "correct": "check-correct", "slow": "check-too-slow"}
        if self.state == "broken":
            body = html.escape(L.phrase("check-broken"))
            return ("text/html", style.check_box("broken", L.phrase("check-broken").split(".")[0], body.split(".", 1)[1]))
        body = _inline(L.t(self.result.text))
        if self.state == "correct":
            body += " " + html.escape(L.phrase("exercise-done"))
        return ("text/html", style.check_box(self.state, L.phrase(words[self.state]), body) + self.result.extra)


def _load_checks(lesson_id: str, lang: str = "en"):
    """Load lessons/<id>/checks.py from the course, never from the learner's copy, so checks update with the course.
    Each lesson in each language gets its own copy, so a Dutch one can write its charts with decimal commas."""
    path = course.lesson_dir(lesson_id) / "checks.py"
    if not path.is_file():
        return _NoChecks()
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson_id.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if lang == "nl":
        for name, value in list(vars(module).items()):
            if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
                setattr(module, name, _with_commas(value))
    return module


def _with_commas(fn):
    """A function of a checks module that gives a chart with decimal commas on its axes, and anything else as it was."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        return chart.with_decimal_comma(out) if isinstance(out, str) and 'class="ea-tick"' in out else out
    return wrapped


class _NoChecks:
    EXERCISES: dict = {}
    REQUIRED: list = []


@functools.cache
def lesson(lesson_id: str, lang: str) -> Lesson:
    return Lesson(lesson_id, lang)
