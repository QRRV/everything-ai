# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson res-what-is-a-paper: sorting texts about research into preprints, papers that passed peer
review, and texts that are no paper at all."""
from everything_ai.check import Sort

# The lesson is done when these are correct. "on-arxiv" is the choice question, "check-claim" the open question.
REQUIRED = ["kinds", "on-arxiv", "check-claim"]

EXERCISES = {
    "kinds": Sort(
        name="kinds",
        groups={
            "preprint": {"en": "Preprint: nobody checked it yet", "nl": "Preprint: nog door niemand gecontroleerd"},
            "reviewed": {"en": "Paper that passed peer review", "nl": "Paper die door peer review kwam"},
            "other": {"en": "Not a paper", "nl": "Geen paper"},
        },
        items={
            "blog": {"en": "A blog post in which the authors explain their result", "nl": "Een blogpost waarin de auteurs hun resultaat uitleggen"},
            "v1": {"en": "A paper on arXiv, version 1", "nl": "Een paper op arXiv, versie 1"},
            "neurips": {"en": "A paper in the proceedings of the NeurIPS conference", "nl": "Een paper in de proceedings van de conferentie NeurIPS"},
            "press": {"en": "A press release from the company behind the research", "nl": "Een persbericht van het bedrijf achter het onderzoek"},
            "website": {"en": "A paper the authors put on their own website before they submit it", "nl": "Een paper die de auteurs op hun eigen website zetten voordat ze hem insturen"},
            "journal": {"en": "A paper that the journal JMLR accepted after peer review", "nl": "Een paper die het tijdschrift JMLR na peer review accepteerde"},
        },
        solution={"blog": "other", "v1": "preprint", "neurips": "reviewed", "press": "other", "website": "preprint", "journal": "reviewed"},
        nudge={
            "blog": {"en": "Does a blog post report the whole study, its method and its results, for other researchers to check?",
                     "nl": "Beschrijft een blogpost het hele onderzoek, met methode en resultaten, zodat andere onderzoekers het kunnen controleren?"},
            "press": {"en": "Who writes a press release, and for whom?",
                      "nl": "Wie schrijft een persbericht, en voor wie?"},
            "v1": {"en": "Does anyone check whether a paper is right before it appears on arXiv?",
                   "nl": "Controleert iemand of een paper klopt voordat hij op arXiv verschijnt?"},
            "website": {"en": "Did reviewers of a conference or a journal judge it before it went on the website?",
                        "nl": "Hebben reviewers van een conferentie of tijdschrift hem beoordeeld voordat hij op de website kwam?"},
            "neurips": {"en": "What happens to a paper before a conference puts it in its proceedings?",
                        "nl": "Wat gebeurt er met een paper voordat een conferentie hem in haar proceedings opneemt?"},
            "journal": {"en": "What did the journal do before it accepted the paper?",
                        "nl": "Wat deed het tijdschrift voordat het de paper accepteerde?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the texts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de teksten opnieuw uit."},
            "empty": {"en": "Put each text in a group: a preprint, a paper that passed peer review, or not a paper.",
                      "nl": "Zet elke tekst in een groep: een preprint, een paper die door peer review kwam, of geen paper."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "\u201c{item}\u201d belongs in another group.",
                      "nl": "\u2018{item}\u2019 hoort in een andere groep."},
            "correct": {"en": "All {n} are right: two preprints, two papers that passed peer review, and two texts about research that are not the paper itself.",
                        "nl": "Alle {n} kloppen: twee preprints, twee papers die door peer review kwamen, en twee teksten over onderzoek die niet de paper zelf zijn."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_RIGHT = {"blog": "other", "v1": "preprint", "neurips": "reviewed", "press": "other", "website": "preprint", "journal": "reviewed"}
CASES = [
    ("kinds", {}, "empty", None),
    ("kinds", _RIGHT, "correct", None),
    ("kinds", {"blog": "other", "v1": "preprint"}, "wrong", "unfinished"),
    ("kinds", {**_RIGHT, "v1": "reviewed"}, "wrong", "wrong:v1"),
]
