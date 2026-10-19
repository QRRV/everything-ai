# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson repro-own-choose: fits, whether a paper of the course's list, or one of the learner's own,
fits the learner's computer and the hours they have. The cards of the papers on the list, in both languages; which
of them fit each kind of computer; and the learner's choice, saved as work/projects/own/paper.json for the lessons of
their own reproduction."""
import functools
import json
import re

import yaml

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, _show_rounded

LEVELS = ["any", "gpu8", "rec"]   # what a paper needs, from least to most, as in the route data
CHOICE = "paper.json"             # the learner's choice, in work/projects/own/

# The upper end of each estimate in course/route/papers.yaml, in hours on the recommended spec, and what the learner
# measures: the cards of the list. The claims follow each paper's abstract (for the ResNet paper its Figure 1, the
# degradation experiment); the course has not measured these estimates yet ("measured: false" in papers.yaml).
HOURS = {"induction-heads": 2, "emergence-mirage": 4, "reversal-curse": 2, "lora-rank": 4, "random-labels": 3,
         "lottery-ticket": 2, "dropout": 1, "resnet-degradation": 2}
CARDS = {
    "induction-heads": {
        "claim": {"en": "Induction heads, attention heads that complete [A][B] … [A] with [B], form in a small transformer at the same point in training where in-context learning suddenly gets better, visible as a bump in the training loss.",
                  "nl": "Induction heads, attention heads die [A][B] … [A] aanvullen met [B], ontstaan in een kleine transformer op hetzelfde moment in de training waarop in-context learning plotseling beter wordt, te zien als een bult in de training loss."},
        "measure": {"en": "the training loss over steps, and how well each head copies a repeated token", "nl": "de training loss over de stappen, en hoe goed elke head een herhaald token kopieert"}},
    "emergence-mirage": {
        "claim": {"en": "Abilities that seem to appear suddenly as models grow appear because of the metric: with a continuous metric, the same models get better smoothly.",
                  "nl": "Vaardigheden die plotseling lijken te verschijnen als modellen groeien, verschijnen door de metric: met een continue metric worden dezelfde modellen geleidelijk beter."},
        "measure": {"en": "one task on models of several sizes, with an all-or-nothing metric and a continuous one", "nl": "één taak op modellen van verschillende grootte, met een alles-of-niets-metric en een continue"}},
    "reversal-curse": {
        "claim": {"en": "A model fine-tuned on \"A is B\" does not learn \"B is A\": asked the other way round, it is no more likely to give the right name than a random one.",
                  "nl": "Een model dat gefinetuned is op \"A is B\", leert niet \"B is A\": andersom gevraagd, geeft het de juiste naam niet vaker dan een willekeurige."},
        "measure": {"en": "the likelihood of the right name and of a random one, for questions in both directions, after fine-tuning on made-up facts", "nl": "de likelihood van de juiste naam en van een willekeurige, bij vragen in beide richtingen, na finetunen op verzonnen feiten"}},
    "lora-rank": {
        "claim": {"en": "Fine-tuning a low-rank update to frozen weights does as well as fine-tuning all of them, or better, with far fewer weights to train.",
                  "nl": "Een low-rank update finetunen op bevroren weights doet het even goed als alle weights finetunen, of beter, met veel minder weights om te trainen."},
        "measure": {"en": "the score of one task after LoRA at a few ranks, and after full fine-tuning", "nl": "de score op één taak na LoRA bij een paar rangen, en na volledig finetunen"}},
    "random-labels": {
        "claim": {"en": "A standard network for images easily fits training data with random labels, so its capacity and regularization alone do not explain why it generalizes.",
                  "nl": "Een gewoon netwerk voor afbeeldingen leert training data met willekeurige labels moeiteloos uit het hoofd, dus verklaren zijn capaciteit en regularisatie alleen niet waarom het generaliseert."},
        "measure": {"en": "the training loss over steps with the true labels, with random labels, and with random pictures", "nl": "de training loss over de stappen met de echte labels, met willekeurige labels en met willekeurige afbeeldingen"}},
    "lottery-ticket": {
        "claim": {"en": "A large network holds a subnetwork of less than 10 to 20% of its weights that, trained alone from the same starting weights, reaches about the same test accuracy in about as many steps.",
                  "nl": "Een groot netwerk bevat een subnetwerk van minder dan 10 tot 20% van zijn weights dat, alleen getraind vanaf dezelfde startwaarden, ongeveer dezelfde test-accuracy haalt in ongeveer evenveel stappen."},
        "measure": {"en": "the test accuracy after a few rounds of pruning, taking out the smallest weights, from the same starting weights and from new random ones", "nl": "de test-accuracy na een paar rondes pruning, waarin je de kleinste weights weghaalt, vanaf dezelfde startwaarden en vanaf nieuwe willekeurige"}},
    "dropout": {
        "claim": {"en": "Dropout, switching off neurons at random during training, reduces overfitting: a network trained with it has a clearly lower test error than without.",
                  "nl": "Dropout, tijdens het trainen willekeurig neuronen uitzetten, vermindert overfitting: een netwerk dat ermee getraind is, heeft een duidelijk lagere testfout dan zonder."},
        "measure": {"en": "the test error of the same network with and without dropout", "nl": "de testfout van hetzelfde netwerk met en zonder dropout"}},
    "resnet-degradation": {
        "claim": {"en": "A deeper plain network can have a higher training error than a shallower one; with residual connections, the deeper one trains better.",
                  "nl": "Een dieper gewoon netwerk kan een hogere trainingsfout hebben dan een ondieper; met residual connections traint het diepere beter."},
        "measure": {"en": "the training error over steps of a shallow and a deep network, plain and with residual connections", "nl": "de trainingsfout over de stappen van een ondiep en een diep netwerk, gewoon en met residual connections"}},
}

# The data of the papers, by their id in course/route/bronnen.yaml, with the license it gives there.
DATA = {
    "tinystories": {"en": "TinyStories (CDLA-Sharing-1.0)", "nl": "TinyStories (CDLA-Sharing-1.0)"},
    "fashion-mnist": {"en": "Fashion-MNIST (MIT)", "nl": "Fashion-MNIST (MIT)"},
    "cifar-10-download-only": {"en": "CIFAR-10 for the optional night run (no license stated: you download it yourself)",
                               "nl": "CIFAR-10 voor de nachtrun die je kunt overslaan (zonder licentie: je downloadt het zelf)"},
    "synthetic-fictional-facts": {"en": "made-up facts your own code writes", "nl": "verzonnen feiten die je eigen code schrijft"},
    "model-qwen3-0.6b-base": {"en": "the model Qwen3-0.6B-Base (Apache-2.0, downloaded)", "nl": "het model Qwen3-0.6B-Base (Apache-2.0, gedownload)"},
    "oasst": {"en": "OpenAssistant conversations, oasst (Apache-2.0)", "nl": "gesprekken van OpenAssistant, oasst (Apache-2.0)"},
}


def data_text(paper: dict, lang: str) -> str:
    return "; ".join(DATA[d][lang] for d in paper["data"])


def builds_on_text(paper: dict, lang: str) -> str:
    """The lessons a paper builds on, by their titles, each a link."""
    from everything_ai import course
    return ", ".join(f"[{course.title(i, lang)}](/course/open/{i}?lang={lang})" for i in paper["builds_on"])


@functools.cache
def papers() -> list[dict]:
    """The list of the course, course/route/papers.yaml, with the hours of HOURS on every paper."""
    data = yaml.safe_load((paths.ROOT / "course" / "route" / "papers.yaml").read_text(encoding="utf-8"))
    return [{**p, "hours": HOURS[p["id"]]} for p in data["own_choice"]]


@functools.cache
def machine() -> str:
    """The level of this computer: "rec" with an NVIDIA card of 12 GB or more, "gpu8" with one of 7 GB or more, else
    "any". A Mac counts as "any": the estimates are for an NVIDIA card. Once per process: it may ask nvidia-smi."""
    from everything_ai import system
    gpu = system.probe()["gpu"] or {}
    vram = gpu.get("vram_gb") or 0
    if gpu.get("kind") == "cuda" and vram >= 11.5:
        return "rec"
    if gpu.get("kind") == "cuda" and vram >= 7:
        return "gpu8"
    return "any"


def course_fits(paper: dict, level: str, hours: float) -> bool:
    return LEVELS.index(paper["needs"]) <= LEVELS.index(level) and paper["hours"] <= hours


def level_fits(paper: dict, level: str) -> bool:
    return LEVELS.index(paper["needs"]) <= LEVELS.index(level)


def plain(text: str) -> str:
    """A text of the learner's as Markdown shows it as it is: a $ would start a formula."""
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")


def fits_of(code: str):
    """The learner's fits from the text of their code box, or None when it does not run."""
    try:
        names = {}
        exec(code, names)
        return names.get("fits")
    except Exception:
        return None


EVENING = 4   # hours: one evening
# A short name for each paper, for the table of the play block, which has to fit beside an editor.
SHORT = {"induction-heads": "Induction heads", "emergence-mirage": "Emergence, a mirage?", "reversal-curse": "The reversal curse",
         "lora-rank": "LoRA", "random-labels": "Random labels", "lottery-ticket": "The lottery ticket", "dropout": "Dropout",
         "resnet-degradation": "ResNet's degradation"}


def number(x: float, digits: int, lang: str) -> str:
    s = f"{x:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".") if lang == "nl" else s


def levels_picture(margin: float, mine: str, labels: dict) -> str:
    """The list as a table: for each paper its short name and estimate, and for each level of computer whether the
    estimate times `margin` fits one evening of EVENING hours there: labels["yes"], labels["more"] when the level is too
    low, or labels["long"]; the learner's level, `mine`, marked with labels["yours"], and the papers that fit it first."""
    import html
    hours = EVENING / margin
    rows = sorted(papers(), key=lambda p: (not course_fits(p, mine, hours), p["hours"], p["title"]))
    pad = "padding:3px 10px 3px 0;vertical-align:top"
    head = "".join(f"<th style='{pad};text-align:left;white-space:nowrap'>{labels[lv]}" + (f"<br>{labels['yours']}" if lv == mine else "") + "</th>" for lv in LEVELS)
    out = [f"<table class='ea-cards' style='border-collapse:collapse'><thead><tr><th style='{pad};text-align:left'>{labels['paper']}</th>"
           f"<th style='{pad};text-align:right'>{labels['hours']}</th>{head}</tr></thead><tbody>"]
    for p in rows:
        cells = []
        for lv in LEVELS:
            word = labels["yes"] if course_fits(p, lv, hours) else labels["more"] if LEVELS.index(p["needs"]) > LEVELS.index(lv) else labels["long"]
            weight = ";font-weight:600" if lv == mine else ""
            cells.append(f"<td style='{pad};white-space:nowrap{weight}'>{word}</td>")
        out.append(f"<tr style='opacity:{1 if course_fits(p, mine, hours) else 0.6}'><td style='{pad}'>{html.escape(SHORT[p['id']])}</td>"
                   f"<td style='{pad};text-align:right'>{p['hours']}</td>{''.join(cells)}</tr>")
    out.append("</tbody></table>")
    return "".join(out)


# --- exercise 1: fits --------------------------------------------------------------------------------------------------

_F_HEAD = ("LEVELS = [\"any\", \"gpu8\", \"rec\"]   # what a paper needs, from least to most\n\n\n"
           "def fits(paper, machine, hours):\n"
           "    # True when the paper needs no more than the level of machine, a name of LEVELS, and its estimate\n"
           "    # paper[\"hours\"] is no more than the hours you have; paper[\"needs\"] is a name of LEVELS too\n")
_F_BODY = "    return LEVELS.index(paper[\"needs\"]) <= LEVELS.index(machine) and paper[\"hours\"] <= hours\n"
_F_START = _F_HEAD + "    ...\n"
_F = _F_HEAD + _F_BODY
_F_SAME = _F.replace("LEVELS.index(paper[\"needs\"]) <= LEVELS.index(machine)", "paper[\"needs\"] == machine")
_F_HOURS = _F.replace(" and paper[\"hours\"] <= hours", "")
_F_STRICT = _F.replace("paper[\"hours\"] <= hours", "paper[\"hours\"] < hours")
_F_STRICT_LEVEL = _F.replace("<= LEVELS.index(machine)", "< LEVELS.index(machine)")
_F_BACK = _F.replace("LEVELS.index(paper[\"needs\"]) <= LEVELS.index(machine)", "LEVELS.index(paper[\"needs\"]) >= LEVELS.index(machine)")

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace each one with your own code.",
                nl="Er staat nog een `...` in je code. Vervang ze allemaal door je eigen code.")
_F_EN, _F_NL = "`{call}` gives `{got}`, and it should give `{want}`", "`{call}` geeft `{got}`, en dat moet `{want}` zijn"

EXERCISES = {
    "fits": FunctionBox(
        name="fits",
        function="fits",
        start=_F_START,
        solution=_F,
        visible=[({"needs": "any", "hours": 2}, "gpu8", 3)],
        hidden=[({"needs": "rec", "hours": 2}, "gpu8", 8), ({"needs": "gpu8", "hours": 4}, "rec", 4), ({"needs": "any", "hours": 1}, "any", 1),
                ({"needs": "any", "hours": 6}, "rec", 4), ({"needs": "gpu8", "hours": 1}, "any", 10)],
        show=lambda v: _show_rounded(v, 4),
        first=[_DOTS],
        mistakes=[
            Mistake("same", wrong=_F_SAME,
                    en=_F_EN + ": your `fits` lets a paper fit only a machine of exactly the level it needs. A machine that has more runs it too: compare their places in `LEVELS`, with `LEVELS.index(...)`.",
                    nl=_F_NL + ": je `fits` laat een paper alleen passen op een machine van precies het niveau dat hij nodig heeft. Een machine die meer heeft, draait hem ook: vergelijk hun plaatsen in `LEVELS`, met `LEVELS.index(...)`."),
            Mistake("back", wrong=_F_BACK,
                    en=_F_EN + ": the levels are the wrong way round: your machine has to have at least what the paper needs, so the paper's place in `LEVELS` is at most the machine's.",
                    nl=_F_NL + ": de niveaus staan andersom: je machine moet minstens hebben wat de paper nodig heeft, dus is de plaats van de paper in `LEVELS` hooguit die van de machine."),
            Mistake("strict-level", wrong=_F_STRICT_LEVEL, state="almost",
                    en="Your `fits` gives the course's answer for the first example. But `{call}` gives `{got}`, and it should give `{want}`: a machine of exactly the level a paper needs runs it too. Compare the places with `<=`.",
                    nl="Je `fits` geeft het antwoord van de course voor het eerste voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: een machine van precies het niveau dat een paper nodig heeft, draait hem ook. Vergelijk de plaatsen met `<=`."),
            Mistake("hours", wrong=_F_HOURS, state="almost",
                    en="Your `fits` gives the course's answer for the first example, where the hours are enough. But `{call}` gives `{got}`, and it should give `{want}`: you do not compare the estimate `paper[\"hours\"]` with the `hours` you have.",
                    nl="Je `fits` geeft het antwoord van de course voor het eerste voorbeeld, waar de uren genoeg zijn. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: je vergelijkt de schatting `paper[\"hours\"]` niet met de `hours` die je hebt."),
            Mistake("strict", wrong=_F_STRICT, state="almost",
                    en="Your `fits` gives the course's answer for the first example. But `{call}` gives `{got}`, and it should give `{want}`: a paper whose estimate is exactly the hours you have fits too. Use `<=`.",
                    nl="Je `fits` geeft het antwoord van de course voor het eerste voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: een paper waarvan de schatting precies de uren is die je hebt, past ook. Gebruik `<=`."),
        ],
        texts={
            "empty": {"en": "Replace the `...`: compare the places of `paper[\"needs\"]` and `machine` in `LEVELS`, and the estimate `paper[\"hours\"]` with `hours`, and give back `True` or `False`.",
                      "nl": "Vervang de `...`: vergelijk de plaatsen van `paper[\"needs\"]` en `machine` in `LEVELS`, en de schatting `paper[\"hours\"]` met `hours`, en geef `True` of `False` terug."},
            "arguments": {"en": "The check calls `{call}`, with a paper, a level and the hours, and the `def` line of your `fits` has another number of names between its parentheses. Keep `(paper, machine, hours)`.",
                          "nl": "De check roept `{call}` aan, met een paper, een niveau en de uren, en de `def`-regel van je `fits` heeft een ander aantal namen tussen de haakjes. Houd `(paper, machine, hours)`."},
            "wrong": {"en": _F_EN + ". Is the paper's place in `LEVELS` at most the machine's, and is its estimate at most the hours you have?",
                      "nl": _F_NL + ". Is de plaats van de paper in `LEVELS` hooguit die van de machine, en is zijn schatting hooguit de uren die je hebt?"},
            "almost": {"en": "Your `fits` gives the course's answer for the first example, but `{call}` gives `{got}`, and it should give `{want}`. Does it work for a machine with more than the paper needs, and with exactly as much, for too few hours, and for exactly enough?",
                       "nl": "Je `fits` geeft het antwoord van de course voor het eerste voorbeeld, maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor een machine met meer dan de paper nodig heeft, en met precies evenveel, voor te weinig uren, en voor precies genoeg?"},
            "correct": {"en": "Your `fits` keeps a paper when your machine has at least what it needs and its estimate fits the hours you have.",
                        "nl": "Je `fits` houdt een paper over als je machine minstens heeft wat hij nodig heeft en zijn schatting past in de uren die je hebt."},
        },
    ),
}

# The lesson is done when these are correct. "why-this" is the open question about the learner's choice.
REQUIRED = ["fits", "why-this"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("fits", _F_START, "empty", None),
    ("fits", _F, "correct", None),
    ("fits", _F_SAME, "wrong", "same"),
    ("fits", _F_BACK, "wrong", "back"),
    ("fits", _F_HOURS, "almost", "hours"),
    ("fits", _F_STRICT, "almost", "strict"),
    ("fits", _F_STRICT_LEVEL, "almost", "strict-level"),
]


# --- the choice: work/projects/own/paper.json --------------------------------------------------------------------------

def choice_path():
    return projects.folder() / "own" / CHOICE


def saved_choice() -> dict | None:
    """The learner's choice, or None when there is none yet, or it is not a choice the course wrote."""
    try:
        c = json.loads(choice_path().read_text(encoding="utf-8"))
        return c if isinstance(c, dict) and c.get("title") else None
    except (OSError, ValueError):
        return None


def save_choice(paper_id: str, own: dict | None = None) -> dict:
    """Saves the learner's choice: a paper of the list by its id, or with paper_id "own" their own paper, from `own`
    with its title, reference, claim, needs and hours. Gives back what it saved."""
    if paper_id == "own":
        o = own or {}
        needs = o.get("needs") if o.get("needs") in LEVELS else "rec"
        try:
            hours = float(o.get("hours") or 0)
        except (TypeError, ValueError):
            hours = 0.0
        c = {"id": "own", "title": str(o.get("title", "")).strip(), "ref": str(o.get("ref", "")).strip(),
             "claim": str(o.get("claim", "")).strip(), "measure": str(o.get("measure", "")).strip(),
             "data": str(o.get("data", "")).strip(), "needs": needs, "hours": hours}
    else:
        p = next(p for p in papers() if p["id"] == paper_id)
        c = {"id": p["id"], "title": p["title"], "authors": p["authors"], "year": p["year"], "ref": p["ref"],
             "claim": CARDS[p["id"]]["claim"]["en"], "measure": CARDS[p["id"]]["measure"]["en"], "data": data_text(p, "en"),
             "needs": p["needs"], "hours": p["hours"], "builds_on": p["builds_on"]}
    path = choice_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return c
