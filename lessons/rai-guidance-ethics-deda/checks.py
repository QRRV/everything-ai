# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-guidance-ethics-deda: the share of the questions a team has answered in each theme of a
DEDA session, sorting the options for action of guidance ethics, and the picture of the three steps of a workshop."""
import html
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# A team that goes through questions like DEDA's for a data project, made up by the course: how many questions it
# took up in each theme, and how many it has answered so far.
QUESTIONS = {"source": 8, "algorithm": 6, "responsibility": 4, "communication": 5}
ANSWERED = {"source": 8, "algorithm": 3, "responsibility": 1, "communication": 5}


def done(questions: dict, answered: dict) -> dict:
    """For each theme, the share of its questions answered."""
    return {t: answered[t] / questions[t] for t in questions}


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The three steps of a guidance ethics workshop, side by side; the third step has three places to act. `labels`
    gives "title", "steps" (three titles), "texts" (the first two steps' texts), "places" (three texts) and "alt"."""
    def card(title: str, inner: str) -> str:
        return (f'<div style="min-width:0;border:1px solid var(--ea-line);border-radius:var(--ea-radius-small);padding:0.4rem 0.6rem">'
                f'<div style="{_UI};font-size:0.85rem;font-weight:600;color:var(--ea-ink);margin-bottom:0.2rem">{html.escape(title)}</div>'
                f'{inner}</div>')

    def text(words: str) -> str:
        return f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);line-height:1.35">{html.escape(words)}</div>'

    places = "".join(f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);border-top:1px solid var(--ea-line);'
                     f'padding:0.2rem 0;line-height:1.3">{html.escape(p)}</div>' for p in labels["places"])
    cards = [card(f"1. {labels['steps'][0]}", text(labels["texts"][0])),
             card(f"2. {labels['steps'][1]}", text(labels["texts"][1])),
             card(f"3. {labels['steps'][2]}", places)]
    body = (f'<div role="img" aria-label="{html.escape(labels["alt"])}" style="display:grid;'
            f'grid-template-columns:repeat(auto-fit,minmax(min(100%,170px),1fr));gap:0.8rem">{"".join(cards)}</div>')
    return yours.figure(labels["title"], body, alt=labels["alt"])


def _done_picture(seen) -> str:
    """The learner's shares per theme as bars, with a dashed line at the right share."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict):
        return ""
    got, want = mine.value, done(QUESTIONS, ANSWERED)
    themes = list(QUESTIONS)

    def number(v) -> float:
        try:
            return min(1.0, max(0.0, float(v))) if not isinstance(v, (bool, str)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    panel = chart.Panel(x=(-0.6, len(themes) - 0.4), y=(0, 1.0), ylabel=seen.word({"en": "share answered", "nl": "aandeel beantwoord"}),
                        yticks=[0, 0.25, 0.5, 0.75, 1], xticks=list(range(len(themes))), xtick_labels=themes)
    for i, t in enumerate(themes):
        if number(got.get(t)) > 0:
            panel.rect(i - 0.3, 0, i + 0.3, number(got.get(t)), role="data")
        panel.line([i - 0.38, i + 0.38], [want[t]] * 2, role="ink", width=2, dashed=True)
    title = seen.word({"en": "Your share of answered questions per theme", "nl": "Jouw aandeel beantwoorde vragen per thema"})
    words = {"en": ("your code", "right"), "nl": ("jouw code", "goed")}[seen.lang]
    key = (f'<div style="{_UI};font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;flex-wrap:wrap;margin-top:0.3rem">'
           f'<span><svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           f'fill-opacity:0.18;stroke:var(--ea-data);stroke-width:1.2"/></svg> {words[0]}</span>'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
           f'stroke-width:2;stroke-dasharray:5 5"/></svg> {words[1]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=480, height=240) + key)


# The lesson is done when these are correct. "how" is the choice question in the lesson.
REQUIRED = ["done", "places", "how"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values: dict) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_HEAD = (_line("questions", QUESTIONS) + "\n" + _line("answered", ANSWERED) + "\n\ndone = {}\nfor theme in questions:\n")
_START = _HEAD + "    ...\ndone\n"
_LINE = "    done[theme] = answered[theme] / questions[theme]\n"
_SOLUTION = _HEAD + _LINE + "done"


def _d(body: str) -> str:
    return _HEAD + body + "done"


EXERCISES = {
    "done": CodeBox(
        name="done",
        picture=_done_picture,
        start=_START,
        solution=_SOLUTION,
        # The check also runs the box as if the team had answered other questions.
        keep=[_line("questions", QUESTIONS)],
        others=[{"answered": Code('{"source": 6, "algorithm": 6, "responsibility": 2, "communication": 1}')}],
        first=[_DOTS],
        digits=2,
        mistakes=[
            Mistake("dicts-divided", error=lambda e: isinstance(e, TypeError) and "dict" in str(e),
                    en="Your code divides whole dictionaries. Divide the values for this one theme: `answered[theme]` and `questions[theme]`.",
                    nl="Je code deelt hele dictionaries. Deel de waarden voor dit ene thema: `answered[theme]` en `questions[theme]`."),
            Mistake("turned", wrong=_d("    done[theme] = questions[theme] / answered[theme]\n"),
                    en="Your last line gives `{got}`: the questions divided by the answers. A share of the questions answered is the answered ones divided by all of them.",
                    nl="Je laatste regel geeft `{got}`: de vragen gedeeld door de antwoorden. Het aandeel beantwoorde vragen is het aantal beantwoorde gedeeld door alle vragen."),
            Mistake("open", wrong=_d("    done[theme] = (questions[theme] - answered[theme]) / questions[theme]\n"),
                    en="Your last line gives `{got}`: the share of the questions still open. Turn it around: the share answered.",
                    nl="Je laatste regel geeft `{got}`: het aandeel vragen dat nog open staat. Draai het om: het aandeel beantwoord."),
            Mistake("counts", wrong=_d("    done[theme] = answered[theme]\n"),
                    en="Your last line gives `{got}`: the numbers of answered questions. Themes have different numbers of questions: divide by `questions[theme]` to get a share.",
                    nl="Je laatste regel geeft `{got}`: het aantal beantwoorde vragen. Thema's hebben verschillend veel vragen: deel door `questions[theme]` om een aandeel te krijgen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line that puts under `done[theme]` the answered questions of this theme divided by all its questions.",
                      "nl": "Vervang de `...` door één regel die onder `done[theme]` de beantwoorde vragen van dit thema zet, gedeeld door al zijn vragen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other answers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere antwoorden."},
            "no_value": {"en": "The last line of the box shows nothing. Put `done` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `done` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does your loop put under each theme `answered[theme]` divided by `questions[theme]`?",
                      "nl": "Je laatste regel geeft `{got}`. Zet je loop onder elk thema `answered[theme]` gedeeld door `questions[theme]`?"},
            "other": {"en": "Right for this team. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you divide `answered` by `questions` themselves?",
                      "nl": "Goed voor dit team. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Deel je `answered` zelf door `questions`?"},
            "other_error": {"en": "Right for this team. But with {where}, your code stops with an error. Do you divide `answered` by `questions` themselves?",
                            "nl": "Goed voor dit team. Maar met {where} stopt je code met een fout. Deel je `answered` zelf door `questions`?"},
            "correct": {"en": "`{got}`: the team knows where its data come from and how it will tell people, but has answered only a quarter of the questions about who is responsible. That is the theme to talk about next, before the project starts. The check also tried other answers.",
                        "nl": "`{got}`: het team weet waar zijn data vandaan komen en hoe het mensen gaat informeren, maar heeft pas een kwart van de vragen over wie verantwoordelijk is beantwoord. Over dat thema moet het nu praten, voordat het project begint. De check probeerde ook andere antwoorden."},
        },
    ),
    "places": Sort(
        name="places",
        groups={
            "design": {"en": "In the technology", "nl": "In de technologie"},
            "context": {"en": "In the surroundings", "nl": "In de omgeving"},
            "user": {"en": "In the people who use it", "nl": "Bij de mensen die het gebruiken"},
        },
        items={
            "blur": {"en": "**Blurred.** The camera blurs faces until the police ask for an image.",
                     "nl": "**Vervaagd.** De camera maakt gezichten onherkenbaar tot de politie om een beeld vraagt."},
            "rule": {"en": "**A rule.** A rule says who may look at the images, and a committee checks it every year.",
                     "nl": "**Een regel.** Een regel zegt wie de beelden mag bekijken, en een commissie controleert dat elk jaar."},
            "training": {"en": "**Training.** Officers learn when they may look and when not.",
                         "nl": "**Training.** Agenten leren wanneer ze mogen kijken en wanneer niet."},
            "windows": {"en": "**The square only.** The camera records the square, not the windows of homes.",
                        "nl": "**Alleen het plein.** De camera neemt het plein op, niet de ramen van huizen."},
            "report": {"en": "**A report.** The council publishes how often the images were viewed.",
                       "nl": "**Een verslag.** De gemeenteraad publiceert hoe vaak de beelden bekeken zijn."},
            "feedback": {"en": "**Feedback.** Officers hear every month how many of the camera's warnings were real fights, so they learn how far to trust it.",
                         "nl": "**Terugkoppeling.** Agenten horen elke maand hoeveel waarschuwingen van de camera echte vechtpartijen waren, zodat ze leren hoe ver ze erop kunnen vertrouwen."},
        },
        solution={"blur": "design", "windows": "design", "rule": "context", "report": "context", "training": "user", "feedback": "user"},
        nudge={
            "blur": {"en": "Is this something the camera itself does?", "nl": "Is dit iets wat de camera zelf doet?"},
            "windows": {"en": "Where the camera looks is part of how it is built and set up.", "nl": "Waar de camera kijkt, hoort bij hoe hij gebouwd en ingesteld is."},
            "rule": {"en": "The camera does not change. What around it does?", "nl": "De camera verandert niet. Wat eromheen wel?"},
            "report": {"en": "Who publishes, and is that part of the camera?", "nl": "Wie publiceert er, en hoort dat bij de camera?"},
            "training": {"en": "Who changes here: the camera, the rules or a person?", "nl": "Wie verandert hier: de camera, de regels of een mens?"},
            "feedback": {"en": "Who learns something here, and who uses the camera?", "nl": "Wie leert hier iets, en wie gebruikt de camera?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the options again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de mogelijkheden opnieuw uit."},
            "empty": {"en": "Put each option for action where it acts.",
                      "nl": "Zet elke handelingsoptie bij de plek waar ze werkt."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The option *{item}* acts in another place.",
                      "nl": "De optie *{item}* werkt op een andere plek."},
            "correct": {"en": "All {n} are right. The same worry, residents feeling watched, gets an answer in three places: in the camera, in the rules and reports around it, and in what the officers who use it know.",
                        "nl": "Alle {n} kloppen. Dezelfde zorg, bewoners die zich bekeken voelen, krijgt een antwoord op drie plekken: in de camera, in de regels en verslagen eromheen, en in wat de agenten die hem gebruiken weten."},
        },
    ),
}

_SORTED = {"blur": "design", "windows": "design", "rule": "context", "report": "context", "training": "user", "feedback": "user"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("done", _START, "empty", None),
    ("done", _SOLUTION, "correct", None),
    ("done", _d("    done[theme] = round(answered[theme] / questions[theme], 2)\n"), "correct", None),
    ("done", _d("    done[theme] = questions[theme] / answered[theme]\n"), "wrong", "turned"),
    ("done", _d("    done[theme] = 1 - answered[theme] / questions[theme]\n"), "wrong", "open"),
    ("done", _d("    done[theme] = answered[theme]\n"), "wrong", "counts"),
    ("done", _d("    done[theme] = answered / questions\n"), "error", "dicts-divided"),
    ("done", _d("    done[theme] = answered[theme] / 8\n"), "wrong", None),
    ("done", _HEAD + "    pass\ndone = {'source': 1.0, 'algorithm': 0.5, 'responsibility': 0.25, 'communication': 1.0}\ndone", "almost", "other-value"),
    ("done", _SOLUTION.replace(_line("questions", QUESTIONS), _line("questions", {**QUESTIONS, "algorithm": 3})), "wrong", "changed-line"),
    ("done", _HEAD + "    ...\ndone = {}\ndone", "wrong", "dots-left"),
    ("places", {}, "empty", None),
    ("places", _SORTED, "correct", None),
    ("places", {"blur": "design", "feedback": "user"}, "wrong", "unfinished"),
    ("places", {**_SORTED, "windows": "context"}, "wrong", "wrong:windows"),
    ("places", {**_SORTED, "report": "user"}, "wrong", "wrong:report"),
    ("places", {**_SORTED, "feedback": "context"}, "wrong", "wrong:feedback"),
]
