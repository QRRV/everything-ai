# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-wellbeing: the scores of the top of a video feed on three measures, sorting measures by
the theory of well-being they belong to, and the picture of the feed."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# The feed of the lesson, made up by the course: six videos in the order an app shows them, ranked by the share of
# viewers expected to click. For each: that share, the share who say a week later they were glad they watched, and
# the share who could do something new afterwards, such as fix a bike.
TITLES = ["prank", "gossip", "unboxing", "fix a bike", "cook rice", "first aid"]
CLICKS = [0.62, 0.55, 0.48, 0.21, 0.18, 0.12]
GLAD = [0.3, 0.2, 0.4, 0.8, 0.7, 0.9]
LEARNED = [0.0, 0.0, 0.1, 0.6, 0.5, 0.7]
MEASURES = ["clicks", "glad", "learned"]
TOP = 3


def top_scores(order: list[int] | None = None) -> dict[str, float]:
    """The mean of each measure over the first three videos of the feed, or of the feed in another order."""
    order = list(range(len(TITLES))) if order is None else order
    columns = {"clicks": CLICKS, "glad": GLAD, "learned": LEARNED}
    return {m: sum(columns[m][i] for i in order[:TOP]) / TOP for m in MEASURES}


def by_learned() -> list[int]:
    """The feed ranked by what viewers could do afterwards instead of by clicks."""
    return sorted(range(len(TITLES)), key=lambda i: -LEARNED[i])


def picture(labels: dict[str, str]) -> str:
    """The feed in the app's order: for each video, the share who click and the share who could do something new
    afterwards."""
    from everything_ai import chart
    n = len(TITLES)
    panel = chart.Panel(x=(-0.6, n - 0.4), y=(0, 0.75), ylabel=labels["share"], yticks=[0, 0.25, 0.5, 0.75],
                        xticks=list(range(n)), xtick_labels=labels["titles"], title=labels["title"])
    for i in range(n):
        panel.rect(i - 0.34, 0, i - 0.02, CLICKS[i], "data")
        panel.rect(i + 0.02, 0, i + 0.34, LEARNED[i], "data", solid=True)
    panel.text(-0.3, 0.71, labels["clicks"], role="data", anchor="start")
    panel.text(2.6, 0.71, labels["learned"], role="data", anchor="start")
    return chart.figure([panel], alt=labels["alt"], width=520, height=270)


def _scores_picture(seen) -> str:
    """The learner's score for each measure as a bar, the right score as a dashed line, and the score of the same
    measure for the feed ranked by what people learned as a dot."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict) or not seen.want.has_value:
        return ""
    got, want = mine.value, seen.want.value
    other = top_scores(by_learned())

    def number(v) -> float:
        try:
            return min(1.0, max(0.0, float(v))) if not isinstance(v, (bool, str)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    panel = chart.Panel(x=(-0.6, len(MEASURES) - 0.4), y=(0, 1.0), ylabel=seen.word({"en": "mean of the top three", "nl": "gemiddelde van de top drie"}),
                        yticks=[0, 0.25, 0.5, 0.75, 1], xticks=list(range(len(MEASURES))), xtick_labels=MEASURES)
    for i, m in enumerate(MEASURES):
        if number(got.get(m)) > 0:
            panel.rect(i - 0.3, 0, i + 0.3, number(got.get(m)), role="data")
        panel.line([i - 0.38, i + 0.38], [number(want.get(m))] * 2, role="ink", width=2, dashed=True)
    panel.points(list(range(len(MEASURES))), [other[m] for m in MEASURES], role="ink", r=5, hollow=True)
    title = seen.word({"en": "Your scores for the top three of the feed", "nl": "Jouw scores voor de top drie van de feed"})
    words = {"en": ("your code", "right", "top three by learned"), "nl": ("jouw code", "goed", "top drie op learned")}[seen.lang]
    key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
           f'flex-wrap:wrap;margin-top:0.3rem">'
           f'<span><svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           f'fill-opacity:0.18;stroke:var(--ea-data);stroke-width:1.2"/></svg> {words[0]}</span>'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
           f'stroke-width:2;stroke-dasharray:5 5"/></svg> {words[1]}</span>'
           f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/></svg> '
           f'{words[2]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=360, height=240) + key)


# The lesson is done when these are correct. "goal" is the choice question in the lesson.
REQUIRED = ["scores", "measures", "goal"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str),
                     en="`feed` has no column with that name. Its columns are `\"title\"`, `\"clicks\"`, `\"glad\"` and `\"learned\"`; in the loop, `measure` holds the name of one of them.",
                     nl="`feed` heeft geen kolom met die naam. De kolommen zijn `\"title\"`, `\"clicks\"`, `\"glad\"` en `\"learned\"`; in de loop bevat `measure` de naam van een ervan.")

_LISTS = (f'titles = {TITLES!r}'.replace("'", '"'), f"clicks = {CLICKS!r}", f"glad = {GLAD!r}", f"learned = {LEARNED!r}")
_LEARNED_LINE = _LISTS[3]
_HEAD = ("import pandas as pd\n\n" + "\n".join(_LISTS) + "\n"
         'feed = pd.DataFrame({"title": titles, "clicks": clicks, "glad": glad, "learned": learned})\n\n'
         'scores = {}\nfor measure in ["clicks", "glad", "learned"]:\n')
_START = _HEAD + "    ...\nscores\n"
_LINE = "    scores[measure] = feed.iloc[:3][measure].mean()\n"
_SOLUTION = _HEAD + _LINE + "scores"


def _s(body: str) -> str:
    return _HEAD + body + "scores"


EXERCISES = {
    "scores": CodeBox(
        name="scores",
        picture=_scores_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_LISTS[0]],
        # The check also runs the box with other answers to the question what viewers could do afterwards.
        others=[{"learned": Code("[0.2, 0.1, 0.0, 0.6, 0.5, 0.7]")}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            _NO_COLUMN,
            Mistake("all", wrong=_s("    scores[measure] = feed[measure].mean()\n"),
                    en="Your last line gives `{got}`: the mean over all six videos. Most viewers only see the top of the feed: keep the first three rows with `iloc[:3]` before you take the mean.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde over alle zes video's. De meeste kijkers zien alleen de top van de feed: houd met `iloc[:3]` de eerste drie rijen over voordat je het gemiddelde neemt."),
            Mistake("bottom", wrong=_s("    scores[measure] = feed.iloc[3:][measure].mean()\n"),
                    en="Your last line gives `{got}`: the mean of the last three videos. `iloc[3:]` starts at row 3 and goes to the end; the first three rows are `iloc[:3]`.",
                    nl="Je laatste regel geeft `{got}`: het gemiddelde van de laatste drie video's. `iloc[3:]` begint bij rij 3 en gaat door tot het eind; de eerste drie rijen zijn `iloc[:3]`."),
            Mistake("from-one", wrong=_s("    scores[measure] = feed.iloc[1:4][measure].mean()\n"),
                    en="Your last line gives `{got}`: rows 1, 2 and 3, so the first video is missing. Counting starts at 0: the first three rows are `iloc[:3]`.",
                    nl="Je laatste regel geeft `{got}`: rij 1, 2 en 3, dus de eerste video ontbreekt. Tellen begint bij 0: de eerste drie rijen zijn `iloc[:3]`."),
            Mistake("rounded", wrong=_s("    scores[measure] = round(feed.iloc[:3][measure].mean(), 2)\n"),
                    en="Your last line gives `{got}`: the means rounded to two decimals, so learned shows 0.03 instead of 0.0333. The worked example rounds only to print; keep the whole mean in `scores`.",
                    nl="Je laatste regel geeft `{got}`: de gemiddelden afgerond op twee decimalen, dus learned wordt 0,03 in plaats van 0,0333. Het uitgewerkte voorbeeld rondt alleen af om te printen; houd het hele gemiddelde in `scores`."),
            Mistake("sum", wrong=_s("    scores[measure] = feed.iloc[:3][measure].sum()\n"),
                    en="Your last line gives `{got}`: the three shares added up, which can be more than 1. A score of the top three is their mean, not their sum.",
                    nl="Je laatste regel geeft `{got}`: de drie aandelen opgeteld, en dat kan meer dan 1 zijn. Een score van de top drie is hun gemiddelde, niet hun som."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: put the mean of this measure over the first three videos in `scores`, under the measure.",
                      "nl": "Vervang de `...` door één regel: zet het gemiddelde van deze maat over de eerste drie video's in `scores`, onder de maat."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `scores` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `scores` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each round keep the first three rows of `feed`, and put the mean of the column `measure` in `scores` under that measure?",
                      "nl": "Je laatste regel geeft `{got}`. Houdt elke ronde de eerste drie rijen van `feed` over, en zet hij het gemiddelde van de kolom `measure` in `scores` onder die maat?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out from `feed` itself?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `feed` zelf?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you work it out from `feed` itself?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Reken je het uit met `feed` zelf?"},
            "correct": {"en": "About 0.55 for clicks, 0.3 for glad and 0.03 for learned. Ranked by what people could do afterwards, the top three would score about 0.17, 0.8 and 0.6: fewer clicks, and far more of what viewers are glad about later. The check also tried other numbers.",
                        "nl": "Ongeveer 0,55 voor clicks, 0,3 voor glad en 0,03 voor learned. Gerangschikt op wat mensen daarna konden, zou de top drie ongeveer 0,17, 0,8 en 0,6 scoren: minder klikken, en veel meer van waar kijkers later blij mee zijn. De check probeerde ook andere getallen."},
        },
    ),
    "measures": Sort(
        name="measures",
        groups={
            "hedonism": {"en": "Pleasure (hedonism)", "nl": "Plezier (hedonisme)"},
            "preference": {"en": "Getting what you want", "nl": "Krijgen wat je wilt"},
            "list": {"en": "A list of goods", "nl": "Een lijst van goederen"},
            "capability": {"en": "What people can do and be", "nl": "Wat mensen kunnen doen en zijn"},
        },
        items={
            "smiles": {"en": "**Smiles.** How much viewers smile and laugh while they watch.",
                       "nl": "**Glimlachen.** Hoeveel kijkers glimlachen en lachen terwijl ze kijken."},
            "mood": {"en": "**Mood.** How good people say they feel right after a session.",
                     "nl": "**Stemming.** Hoe goed mensen zeggen dat ze zich voelen direct na een sessie."},
            "clicks": {"en": "**Clicks.** How often people pick a video when the app offers it.",
                       "nl": "**Klikken.** Hoe vaak mensen een video kiezen als de app hem aanbiedt."},
            "asked": {"en": "**Asked for.** How often the app shows the kind of video people said they wanted.",
                      "nl": "**Gevraagd.** Hoe vaak de app het soort video laat zien waar mensen om vroegen."},
            "health": {"en": "**Health and friends.** Whether people sleep enough and see their friends, whatever they want.",
                       "nl": "**Gezondheid en vrienden.** Of mensen genoeg slapen en hun vrienden zien, wat ze ook willen."},
            "knowledge": {"en": "**Knowledge.** How much people know about the world, counted as good for anyone.",
                          "nl": "**Kennis.** Hoeveel mensen weten over de wereld, geteld als goed voor iedereen."},
            "skill": {"en": "**A new skill.** Whether people can now do something they could not do before, such as cook a meal.",
                      "nl": "**Iets nieuws kunnen.** Of mensen nu iets kunnen wat ze eerst niet konden, zoals een maaltijd koken."},
            "join": {"en": "**Taking part.** Whether people are able to join a club or a vote in their town.",
                     "nl": "**Meedoen.** Of mensen in staat zijn mee te doen aan een club of een stemming in hun dorp."},
        },
        solution={"smiles": "hedonism", "mood": "hedonism", "clicks": "preference", "asked": "preference",
                  "health": "list", "knowledge": "list", "skill": "capability", "join": "capability"},
        nudge={
            "smiles": {"en": "Is this about a feeling, a wish, a good on a list, or what someone is able to do?", "nl": "Gaat dit over een gevoel, een wens, een goed op een lijst, of wat iemand kan?"},
            "mood": {"en": "It asks how people feel. Which theory counts feelings?", "nl": "Het vraagt hoe mensen zich voelen. Welke theorie telt gevoelens?"},
            "clicks": {"en": "A click is a choice in the moment. Which theory counts what people choose?", "nl": "Een klik is een keuze van het moment. Welke theorie telt wat mensen kiezen?"},
            "asked": {"en": "It follows what people said they wanted. Is that a feeling, or a wish?", "nl": "Het volgt wat mensen zeiden te willen. Is dat een gevoel, of een wens?"},
            "health": {"en": "*Whatever they want*: which theory counts things as good even when people do not want them?", "nl": "*Wat ze ook willen*: welke theorie telt dingen als goed, ook als mensen ze niet willen?"},
            "knowledge": {"en": "Counted as good for anyone: is that a feeling, a wish, or a good on a list?", "nl": "Geteld als goed voor iedereen: is dat een gevoel, een wens, of een goed op een lijst?"},
            "skill": {"en": "It asks what people are now able to do. Which theory asks that?", "nl": "Het vraagt wat mensen nu kunnen. Welke theorie vraagt dat?"},
            "join": {"en": "*Are able to*: is this about a good on a list, or about a real freedom to act?", "nl": "*In staat zijn*: gaat dit over een goed op een lijst, of over een echte vrijheid om iets te doen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the measures again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de maten opnieuw uit."},
            "empty": {"en": "Put each measure with the theory of well-being it belongs to.",
                      "nl": "Zet elke maat bij de theorie van welzijn waar hij bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The measure *{item}* belongs with another theory.",
                      "nl": "De maat *{item}* hoort bij een andere theorie."},
            "correct": {"en": "All {n} are right. Smiles and mood count feelings, clicks and what people asked for count wishes, health and knowledge are goods on a list, and a new skill and taking part are things people are able to do.",
                        "nl": "Alle {n} kloppen. Glimlachen en stemming tellen gevoelens, klikken en wat mensen vroegen tellen wensen, gezondheid en kennis zijn goederen op een lijst, en iets nieuws kunnen en meedoen zijn dingen die mensen kunnen doen."},
        },
    ),
}

_SORTED = {"smiles": "hedonism", "mood": "hedonism", "clicks": "preference", "asked": "preference",
           "health": "list", "knowledge": "list", "skill": "capability", "join": "capability"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("scores", _START, "empty", None),
    ("scores", _SOLUTION, "correct", None),
    ("scores", _s("    scores[measure] = feed[measure].iloc[:3].mean()\n"), "correct", None),
    ("scores", _s("    top = feed.iloc[:3]\n    scores[measure] = top[measure].sum() / 3\n"), "correct", None),
    ("scores", _s("    scores[measure] = feed.iloc[0:3][measure].mean()\n"), "correct", None),
    ("scores", _s("    scores[measure] = feed[measure].mean()\n"), "wrong", "all"),
    ("scores", _s("    scores[measure] = feed.iloc[3:][measure].mean()\n"), "wrong", "bottom"),
    ("scores", _s("    scores[measure] = feed.iloc[1:4][measure].mean()\n"), "wrong", "from-one"),
    ("scores", _s("    scores[measure] = feed.iloc[:3][measure].sum()\n"), "wrong", "sum"),
    ("scores", _s("    scores[measure] = round(feed.iloc[:3][measure].mean(), 2)\n"), "wrong", "rounded"),
    ("scores", _HEAD + "    ...\nscores = {'clicks': 0.55, 'glad': 0.3, 'learned': 0.0333}\nscores", "wrong", "dots-left"),
    ("scores", _s("    scores = {'clicks': 0.55, 'glad': 0.3, 'learned': 1 / 30}\n"), "almost", "other-value"),
    ("scores", _SOLUTION.replace(_LISTS[0], 'titles = ["a", "b", "c", "d", "e", "f"]'), "wrong", "changed-line"),
    ("scores", _s('    scores[measure] = feed.iloc[:3]["measure"].mean()\n'), "error", "no-column"),
    ("measures", {}, "empty", None),
    ("measures", _SORTED, "correct", None),
    ("measures", {"smiles": "hedonism", "skill": "capability"}, "wrong", "unfinished"),
    ("measures", {**_SORTED, "mood": "preference"}, "wrong", "wrong:mood"),
    ("measures", {**_SORTED, "asked": "hedonism"}, "wrong", "wrong:asked"),
    ("measures", {**_SORTED, "knowledge": "capability"}, "wrong", "wrong:knowledge"),
    ("measures", {**_SORTED, "join": "list"}, "wrong", "wrong:join"),
]
