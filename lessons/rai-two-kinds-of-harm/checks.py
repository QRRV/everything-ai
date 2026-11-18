# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-two-kinds-of-harm: how often an image generator shows a woman for four jobs, sorting
harms into allocative and representational, and the picture of the two kinds of harm side by side."""
import re

from everything_ai.check import Code, CodeBox, Mistake, Sort
from everything_ai import yours

# The worked example and the left panel of the picture: 20 loan applications, made up by the course.
LOANS = {"A": (12, 9), "B": (8, 3)}   # group: (applications, yes)
# The right panel: Kay, Matuszek and Munson (2015). Of the Google image results for "CEO", 11% showed a woman, while
# 27% of the CEOs in the US were women (US Bureau of Labor Statistics, as the paper gives it).
CEO_IMAGES, CEO_REAL = 0.11, 0.27

# Exercise 1: ten pictures an image generator made for each of four jobs, and the share of women in each job.
# Both are made up by the course, and the lesson says so.
JOBS = ["nurse", "engineer", "teacher", "CEO"]
WORKFORCE = {"nurse": 0.85, "engineer": 0.25, "teacher": 0.7, "CEO": 0.27}


def picture(labels: dict[str, str]) -> str:
    """Two panels: the share of each group that gets a yes for a loan, and the share of women in the image results
    for CEO next to the share among real CEOs."""
    from everything_ai import chart
    loans = chart.Panel((0, 3), (0, 1.0), title=labels["loans"], ylabel=labels["loans_y"], xticks=[])
    for i, (group, (n, yes)) in enumerate(LOANS.items()):
        loans.rect(i + 1 - 0.3, 0, i + 1 + 0.3, yes / n, "data")
        loans.text(i + 1, yes / n + 0.05, group)
    ceo = chart.Panel((0, 3), (0, 0.5), title=labels["ceo"], ylabel=labels["ceo_y"], xticks=[])
    for i, (key, share) in enumerate((("images", CEO_IMAGES), ("real", CEO_REAL))):
        ceo.rect(i + 1 - 0.3, 0, i + 1 + 0.3, share, "data")
        ceo.text(i + 1, share + 0.025, labels[key])
    return chart.figure([loans, ceo], alt=labels["alt"], width=300, height=260)


def _shares_picture(seen) -> str:
    """The learner's share of women per job as a bar, the right share as a dashed line over it, and the share of
    women in the workforce as a dot: where the bar goes past the dot, the pictures make the stereotype stronger."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, dict) or not seen.want.has_value:
        return ""
    got, want = mine.value, seen.want.value

    def number(v) -> float:
        try:
            return min(1.0, max(0.0, float(v))) if not isinstance(v, (bool, str)) else 0.0
        except (TypeError, ValueError):
            return 0.0
    panel = chart.Panel(x=(-0.6, len(JOBS) - 0.4), y=(0, 1.08), ylabel=seen.word({"en": "share of women", "nl": "aandeel vrouwen"}),
                        yticks=[0, 0.25, 0.5, 0.75, 1], xticks=list(range(len(JOBS))), xtick_labels=JOBS)
    for i, job in enumerate(JOBS):
        if number(got.get(job)) > 0:
            panel.rect(i - 0.34, 0, i + 0.34, number(got.get(job)), role="data")
        panel.line([i - 0.42, i + 0.42], [number(want.get(job))] * 2, role="ink", width=2, dashed=True)
    panel.points(list(range(len(JOBS))), [WORKFORCE[j] for j in JOBS], role="ink", r=5, hollow=True)
    title = seen.word({"en": "Your share of women in the pictures, per job", "nl": "Jouw aandeel vrouwen op de afbeeldingen, per beroep"})
    words = {"en": ("your code", "right", "share of women in the job"), "nl": ("jouw code", "goed", "aandeel vrouwen in het beroep")}[seen.lang]
    key = (f'<div style="font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);display:flex;gap:1.2rem;'
           f'flex-wrap:wrap;margin-top:0.3rem">'
           f'<span><svg width="14" height="12" aria-hidden="true"><rect x="1" y="1" width="12" height="10" style="fill:var(--ea-data);'
           f'fill-opacity:0.18;stroke:var(--ea-data);stroke-width:1.2"/></svg> {words[0]}</span>'
           f'<span><svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" style="stroke:var(--ea-ink);'
           f'stroke-width:2;stroke-dasharray:5 5"/></svg> {words[1]}</span>'
           f'<span><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:none;stroke:var(--ea-ink);stroke-width:1.5"/></svg> '
           f'{words[2]}</span></div>')
    return yours.figure(title, chart.figure([panel], alt=title, width=360, height=240) + key)


# The lesson is done when these are correct. "look-for" is the choice question in the lesson.
REQUIRED = ["shares", "kinds", "look-for"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str),
                     en="`images` has no column with that name. Its columns are `\"job\"` and `\"woman\"`, in small letters, each between quotes.",
                     nl="`images` heeft geen kolom met die naam. De kolommen zijn `\"job\"` en `\"woman\"`, in kleine letters, elk tussen aanhalingstekens.")
_ROWS_MEAN = Mistake("rows-mean", error=lambda e: isinstance(e, TypeError) and ("reduction" in str(e) or "to numeric" in str(e)),
                     en="`.mean()` of whole rows also takes the column `job`, and that holds words, not numbers. Take the mean of the column `woman` only.",
                     nl="`.mean()` van hele rijen neemt ook de kolom `job` mee, en daarin staan woorden, geen getallen. Neem alleen het gemiddelde van de kolom `woman`.")

_JOBS_LINE = 'jobs = ["nurse"] * 10 + ["engineer"] * 10 + ["teacher"] * 10 + ["CEO"] * 10'
_WOMAN_LINE = "woman = [1] * 10 + [1] + [0] * 9 + [1] * 9 + [0] + [1] + [0] * 9"
_HEAD = (f"import pandas as pd\n\n{_JOBS_LINE}\n{_WOMAN_LINE}\n"
         'images = pd.DataFrame({"job": jobs, "woman": woman})\n\nshares = {}\nfor job in ["nurse", "engineer", "teacher", "CEO"]:\n')
_START = _HEAD + "    ...\nshares\n"
_LINE = '    shares[job] = images[images["job"] == job]["woman"].mean()\n'
_SOLUTION = _HEAD + _LINE + "shares"


def _s(body: str) -> str:
    return _HEAD + body + "shares"


def _values(got) -> list[float] | None:
    try:
        return [float(got[j]) for j in JOBS]
    except (TypeError, KeyError, ValueError, IndexError):
        return None


def _all_same(got, text) -> bool:
    """The same share four times: every round took the mean over all pictures."""
    v = _values(got)
    return v is not None and max(v) - min(v) < 1e-9


def _of_men(got, text) -> bool:
    """One minus the right share in every job: the share of men."""
    v = _values(got)
    return v is not None and all(abs(a - b) < 1e-9 for a, b in zip(v, [0.0, 0.9, 0.1, 0.9]))


EXERCISES = {
    "shares": CodeBox(
        name="shares",
        picture=_shares_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_JOBS_LINE],
        # The check also runs the box with other pictures: numbers typed in by hand do not change with them.
        others=[{"woman": Code("[1] * 7 + [0] * 3 + [1] * 3 + [0] * 7 + [1] * 5 + [0] * 5 + [1] * 2 + [0] * 8")}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            _NO_COLUMN,
            _ROWS_MEAN,
            Mistake("overall", matches=_all_same,
                    en="Your last line gives `{got}`: four times the same share, that of all 40 pictures. In each round, keep only the pictures of that job with a mask, and take the mean of their `woman`.",
                    nl="Je laatste regel geeft `{got}`: vier keer hetzelfde aandeel, dat van alle 40 afbeeldingen. Houd in elke ronde met een mask alleen de afbeeldingen van dat beroep over, en neem het gemiddelde van hun `woman`."),
            Mistake("count", wrong=_s('    shares[job] = images[images["job"] == job]["woman"].sum()\n'),
                    en="Your last line gives `{got}`: how many pictures of each job show a woman. A share is that number divided by the pictures of the job: the mean of `woman`, not its sum.",
                    nl="Je laatste regel geeft `{got}`: hoeveel afbeeldingen van elk beroep een vrouw tonen. Een aandeel is dat aantal gedeeld door de afbeeldingen van het beroep: het gemiddelde van `woman`, niet de som."),
            Mistake("of-men", matches=_of_men,
                    en="Your last line gives `{got}`: the share of men in each job. `woman` is 1 for a picture of a woman, so its mean is already the share of women.",
                    nl="Je laatste regel geeft `{got}`: het aandeel mannen in elk beroep. `woman` is 1 voor een afbeelding van een vrouw, dus het gemiddelde ervan is al het aandeel vrouwen."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with one line: put the share of women among the pictures of this job in `shares`, under the job.",
                      "nl": "Vervang de `...` door één regel: zet het aandeel vrouwen op de afbeeldingen van dit beroep in `shares`, onder het beroep."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries other pictures by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf andere afbeeldingen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `shares` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `shares` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does each round keep the pictures of one job with a mask, and put the mean of their `woman` in `shares` under that job?",
                      "nl": "Je laatste regel geeft `{got}`. Houdt elke ronde met een mask de afbeeldingen van één beroep over, en zet hij het gemiddelde van hun `woman` in `shares` onder dat beroep?"},
            "other": {"en": "Right for these pictures. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you work it out with `images` itself?",
                      "nl": "Goed voor deze afbeeldingen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Reken je het uit met `images` zelf?"},
            "other_error": {"en": "Right for these pictures. But with {where}, your code stops with an error. Do you work it out with `images` itself?",
                            "nl": "Goed voor deze afbeeldingen. Maar met {where} stopt je code met een fout. Reken je het uit met `images` zelf?"},
            "correct": {"en": "1.0 for nurse, 0.1 for engineer, 0.9 for teacher and 0.1 for CEO. In the workforce of this exercise the shares are 0.85, 0.25, 0.7 and 0.27: in every job the pictures go further than the world, towards the stereotype. The check also tried other pictures.",
                        "nl": "1,0 voor nurse, 0,1 voor engineer, 0,9 voor teacher en 0,1 voor CEO. In de beroepsbevolking van deze oefening zijn de aandelen 0,85, 0,25, 0,7 en 0,27: in elk beroep gaan de afbeeldingen verder dan de wereld, naar het stereotype toe. De check probeerde ook andere afbeeldingen."},
        },
    ),
    "kinds": Sort(
        name="kinds",
        groups={
            "allocative": {"en": "Allocative: what the system hands out", "nl": "Allocatief: wat het systeem uitdeelt"},
            "representational": {"en": "Representational: how the system shows people", "nl": "Representatief: hoe het systeem mensen laat zien"},
        },
        items={
            "loan": {"en": "**A loan.** A bank's model turns down people from one part of town more often than people with the same income elsewhere.",
                     "nl": "**Een lening.** Het model van een bank wijst mensen uit één wijk vaker af dan mensen met hetzelfde inkomen elders."},
            "checks": {"en": "**Extra checks.** A fraud model picks people with a second nationality far more often for an extra check.",
                       "nl": "**Extra controles.** Een fraudemodel kiest mensen met een tweede nationaliteit veel vaker voor een extra controle."},
            "ad": {"en": "**A job ad.** An ad for a well-paid job is shown to men far more often than to women.",
                   "nl": "**Een vacature.** Een advertentie voor een goedbetaalde baan wordt veel vaker aan mannen dan aan vrouwen getoond."},
            "translation": {"en": "**A translation.** A translation app turns two Turkish sentences without gender into *he is a doctor* and *she is a nurse*.",
                            "nl": "**Een vertaling.** Een vertaalapp maakt van twee Turkse zinnen zonder geslacht *hij is dokter* en *zij is verpleegkundige*."},
            "autocomplete": {"en": "**Autocomplete.** A search box finishes a question about a religion with an insult.",
                             "nl": "**Aanvullen.** Een zoekvak maakt een vraag over een geloof af met een belediging."},
            "ceo": {"en": "**CEOs.** An image generator draws every CEO as a man in a suit.",
                    "nl": "**CEO's.** Een beeldgenerator tekent elke CEO als een man in pak."},
        },
        solution={"loan": "allocative", "checks": "allocative", "ad": "allocative",
                  "translation": "representational", "autocomplete": "representational", "ceo": "representational"},
        nudge={
            "loan": {"en": "Does someone get something here, or not get it?",
                     "nl": "Krijgt iemand hier iets, of juist niet?"},
            "checks": {"en": "An extra check is not a gift, but it is still handed out to some people and not to others. Who gets it?",
                       "nl": "Een extra controle is geen cadeau, maar hij wordt wel aan sommige mensen uitgedeeld en aan anderen niet. Wie krijgt hem?"},
            "ad": {"en": "The ad is a chance at a job. Who gets that chance, and who never sees it?",
                   "nl": "De advertentie is een kans op een baan. Wie krijgt die kans, en wie ziet hem nooit?"},
            "translation": {"en": "Is anyone given or refused something? Or is it about how doctors and nurses are pictured?",
                            "nl": "Krijgt iemand iets, of wordt iemand iets geweigerd? Of gaat het om hoe dokters en verpleegkundigen worden neergezet?"},
            "autocomplete": {"en": "Nobody is refused anything. What does the search box do to how a group is seen?",
                             "nl": "Niemand wordt iets geweigerd. Wat doet het zoekvak met hoe een groep wordt gezien?"},
            "ceo": {"en": "A drawing decides nothing about anyone. What does it say about who can be a CEO?",
                    "nl": "Een tekening beslist over niemand iets. Wat zegt hij over wie CEO kan zijn?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the cases again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gevallen opnieuw uit."},
            "empty": {"en": "Put each case with the kind of harm it does.",
                      "nl": "Zet elk geval bij de soort schade die het doet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The case *{item}* belongs with the other kind of harm.",
                      "nl": "Het geval *{item}* hoort bij de andere soort schade."},
            "correct": {"en": "All {n} are right. A loan, a check and an ad are handed out, so you can count per group who gets them; a translation, a search box and a drawing show people, and harm them without deciding anything.",
                        "nl": "Alle {n} kloppen. Een lening, een controle en een vacature worden uitgedeeld, dus je kunt per groep tellen wie ze krijgt; een vertaling, een zoekvak en een tekening laten mensen zien, en doen schade zonder iets te beslissen."},
        },
    ),
}

_SORTED = {"loan": "allocative", "checks": "allocative", "ad": "allocative",
           "translation": "representational", "autocomplete": "representational", "ceo": "representational"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("shares", _START, "empty", None),
    ("shares", _SOLUTION, "correct", None),
    ("shares", _s('    rows = images[images["job"] == job]\n    shares[job] = rows["woman"].mean()\n'), "correct", None),
    ("shares", _s('    shares[job] = images["woman"][images["job"] == job].mean()\n'), "correct", None),
    ("shares", _s('    shares[job] = images[images["job"] == job]["woman"].sum() / len(images[images["job"] == job])\n'), "correct", None),
    ("shares", _s('    shares[job] = images["woman"].mean()\n'), "wrong", "overall"),
    ("shares", _s('    shares[job] = images[images["job"] == job]["woman"].sum()\n'), "wrong", "count"),
    ("shares", _s('    shares[job] = 1 - images[images["job"] == job]["woman"].mean()\n'), "wrong", "of-men"),
    ("shares", _HEAD + "    ...\nshares = {'nurse': 1.0, 'engineer': 0.1, 'teacher': 0.9, 'CEO': 0.1}\nshares", "wrong", "dots-left"),
    ("shares", _s("    shares = {'nurse': 1.0, 'engineer': 0.1, 'teacher': 0.9, 'CEO': 0.1}\n"), "almost", "other-value"),
    ("shares", _SOLUTION.replace(_JOBS_LINE, 'jobs = ["nurse"] * 10 + ["engineer"] * 10 + ["teacher"] * 10 + ["pilot"] * 10'), "wrong", "changed-line"),
    ("shares", _s('    shares[job] = images[images["jobs"] == job]["woman"].mean()\n'), "error", "no-column"),
    ("shares", _s('    shares[job] = images[images["job"] == job].mean()\n'), "error", "rows-mean"),
    ("kinds", {}, "empty", None),
    ("kinds", _SORTED, "correct", None),
    ("kinds", {"loan": "allocative", "ceo": "representational"}, "wrong", "unfinished"),
    ("kinds", {**_SORTED, "checks": "representational"}, "wrong", "wrong:checks"),
    ("kinds", {**_SORTED, "ad": "representational"}, "wrong", "wrong:ad"),
    ("kinds", {**_SORTED, "translation": "allocative"}, "wrong", "wrong:translation"),
]
