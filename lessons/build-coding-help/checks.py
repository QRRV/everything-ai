# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson build-coding-help: which kind of help with code fits a task, and what a coding agent may do by
itself; and the picture of the three kinds of help."""
import html

from everything_ai.check import Sort

# --- the picture of the three kinds of help ---------------------------------------------------------------------------

_PANELS = [
    {"title": {"en": "Code completion", "nl": "Code completion"},
     "sees": {"en": "the file, around where you type", "nl": "het bestand, rond waar je typt"},
     "check": {"en": "every suggestion", "nl": "elk voorstel"}},
    {"title": {"en": "Chat about code", "nl": "Chat over code"},
     "sees": {"en": "what you paste in", "nl": "wat je erin plakt"},
     "check": {"en": "every answer", "nl": "elk antwoord"}},
    {"title": {"en": "Coding agent", "nl": "Coding agent"},
     "sees": {"en": "the whole project folder", "nl": "de hele projectmap"},
     "check": {"en": "what it changed", "nl": "wat hij veranderde"}},
]
_QUESTION = {"en": ["Why does line 4 stop", "with an error?"], "nl": ["Waarom stopt regel 4", "met een foutmelding?"]}
_ANSWER = {"en": ["Line 4 asks for 'year',", "but your data has 'Year',", "with a capital."],
           "nl": ["Regel 4 vraagt om 'year',", "maar je gegevens hebben", "'Year', met een hoofdletter."]}
_STEPS = {"en": ["reads 4 files", "changes 2 of them", "runs the tests: 1 fails", "changes books.py again",
                 "runs the tests: all pass"],
          "nl": ["leest 4 bestanden", "verandert er 2", "draait de tests: 1 faalt", "verandert books.py opnieuw",
                 "draait de tests: alle slagen"]}
_MORE = {"en": "does more by itself", "nl": "doet meer zelf"}
_SEES = {"en": "Sees:", "nl": "Ziet:"}
_CHECK = {"en": "You check:", "nl": "Jij kijkt na:"}
_TITLE = {"en": "Three kinds of help with code", "nl": "Drie soorten hulp bij code"}
_ALT = {"en": "Three panels from left to right, each doing more by itself. Code completion: under the line def area(width, height): it suggests return width * height. Chat about code: to the question why line 4 stops with an error, it answers that the data has 'Year' with a capital. Coding agent: it reads 4 files, changes 2, runs the tests, sees one fail, changes a file again and runs the tests until all pass.",
        "nl": "Drie vakken van links naar rechts, die elk meer zelf doen. Code completion: onder de regel def area(width, height): stelt het return width * height voor. Chat over code: op de vraag waarom regel 4 met een foutmelding stopt, antwoordt het dat de gegevens 'Year' met een hoofdletter hebben. Coding agent: hij leest 4 bestanden, verandert er 2, draait de tests, ziet er een falen, verandert opnieuw een bestand en draait de tests tot ze allemaal slagen."}


def _text(x: float, y: float, s: str, role: str = "ink", size: int = 12, mono: bool = False, weight: int = 400) -> str:
    font = "var(--ea-font-code)" if mono else "var(--ea-font-ui)"
    fill = {"ink": "var(--ea-ink)", "ink-2": "var(--ea-ink-2)", "pred": "var(--ea-pred-text)"}[role]
    space = ";white-space:pre" if mono else ""
    return (f'<text x="{x}" y="{y}" style="font-size:{size}px;font-family:{font};font-weight:{weight};fill:{fill}{space}">'
            f'{html.escape(s)}</text>')


def ways_picture(lang: str) -> str:
    """The three kinds of help side by side: what each sees, what it does, and what you check. What the model writes
    is in the colour of predictions."""
    W, gap, top, H = 232, 20, 34, 238
    parts = [f'<defs><marker id="ea-ways-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
             f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ea-ink-2)"/></marker></defs>',
             f'<line x1="10" y1="14" x2="746" y2="14" style="stroke:var(--ea-ink-2);stroke-width:1.2" marker-end="url(#ea-ways-arrow)"/>',
             f'<text x="740" y="30" text-anchor="end" style="font-size:12px;font-family:var(--ea-font-ui);fill:var(--ea-ink-2)">'
             f'{html.escape(_MORE[lang])}</text>']
    for i, p in enumerate(_PANELS):
        x = 10 + i * (W + gap)
        parts.append(f'<rect x="{x}" y="{top + 8}" width="{W}" height="{H}" rx="8" style="fill:var(--ea-inset);stroke:var(--ea-line)"/>')
        parts.append(_text(x + 12, top + 30, p["title"][lang], size=13, weight=600))
        y = top + 58
        if i == 0:
            parts.append(_text(x + 12, y, "def area(width, height):", mono=True))
            parts.append(_text(x + 12, y + 20, "    return width * height", role="pred", mono=True))
        elif i == 1:
            for k, line in enumerate(_QUESTION[lang]):
                parts.append(_text(x + 12, y + 18 * k, line))
            for k, line in enumerate(_ANSWER[lang]):
                parts.append(_text(x + 24, y + 44 + 18 * k, line, role="pred"))
        else:
            for k, line in enumerate(_STEPS[lang]):
                parts.append(_text(x + 12, y + 20 * k, f"{k + 1}. {line}", role="pred"))
        parts.append(f'<line x1="{x + 12}" x2="{x + W - 12}" y1="{top + H - 50}" y2="{top + H - 50}" style="stroke:var(--ea-line)"/>')
        parts.append(_text(x + 12, top + H - 30, f'{_SEES[lang]} {p["sees"][lang]}', role="ink-2"))
        parts.append(_text(x + 12, top + H - 10, f'{_CHECK[lang]} {p["check"][lang]}', role="ink-2"))
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem"><svg viewBox="0 0 760 {top + H + 14}" role="img" '
            f'style="width:100%;max-width:46rem;height:auto"><title>{html.escape(_TITLE[lang])}</title>'
            f'<desc>{html.escape(_ALT[lang])}</desc>{"".join(parts)}</svg></figure>')


# --- the exercises ----------------------------------------------------------------------------------------------------

_COMPLETION = {"en": "Is it a line or two, right where you are typing?",
               "nl": "Gaat het om een regel of twee, precies waar je typt?"}
_CHAT = {"en": "Do you want an explanation or an answer, which you then use yourself?",
         "nl": "Wil je een uitleg of een antwoord, dat je daarna zelf gebruikt?"}
_AGENT = {"en": "Does it take many files and running the tests, without you doing each step?",
          "nl": "Gaat het om veel bestanden en het draaien van de tests, zonder dat jij elke stap doet?"}
_ITSELF = {"en": "Is it easy to undo, and does it change nothing outside the project?",
           "nl": "Is het makkelijk terug te draaien, en verandert het niets buiten het project?"}
_ASK = {"en": "Is it hard to undo, or does something come in from the internet or go out to it?",
        "nl": "Is het moeilijk terug te draaien, of komt er iets van internet binnen of gaat er iets naartoe?"}

EXERCISES = {
    "which-help": Sort(
        name="which-help",
        groups={
            "completion": {"en": "Code completion", "nl": "Code completion"},
            "chat": {"en": "Chat", "nl": "Chat"},
            "agent": {"en": "Coding agent", "nl": "Coding agent"},
        },
        items={
            "error": {"en": "**An error message.** Your program stops with an error message you do not understand, and you want it explained, with your code beside it.",
                      "nl": "**Een foutmelding.** Je programma stopt met een foutmelding die je niet begrijpt, en je wilt uitleg, met je code ernaast."},
            "line": {"en": "**The next line.** You are typing a piece of code, and want the next line suggested as you go.",
                     "nl": "**De volgende regel.** Je typt een stuk code, en wilt de volgende regel voorgesteld krijgen terwijl je typt."},
            "rename": {"en": "**Twelve files.** A function gets a new name in twelve files, and the tests have to pass afterwards.",
                       "nl": "**Twaalf bestanden.** Een functie krijgt een nieuwe naam in twaalf bestanden, en de tests moeten daarna slagen."},
            "which": {"en": "**Two ways.** You want to know which of two ways to store your data fits better, and why.",
                      "nl": "**Twee manieren.** Je wilt weten welke van twee manieren om je gegevens op te slaan beter past, en waarom."},
            "return": {"en": "**A last line.** You have typed the first line of a small piece of code, and want the line under it suggested.",
                       "nl": "**Een laatste regel.** Je hebt de eerste regel van een klein stuk code getypt, en wilt de regel eronder voorgesteld krijgen."},
            "option": {"en": "**A new option.** Your program gets a new option, with a test, in three files, while you do something else.",
                       "nl": "**Een nieuwe optie.** Je programma krijgt een nieuwe optie, met een test, in drie bestanden, terwijl jij iets anders doet."},
        },
        solution={"error": "chat", "line": "completion", "rename": "agent", "which": "chat", "return": "completion", "option": "agent"},
        nudge={"line": _COMPLETION, "return": _COMPLETION, "error": _CHAT, "which": _CHAT, "rename": _AGENT, "option": _AGENT},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six tasks again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes taken opnieuw uit."},
            "empty": {"en": "Pick a kind of help for each of the six.",
                      "nl": "Kies voor elk van de zes een soort hulp."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. A line where you type: completion. A question whose answer you use yourself: a chat. Work across files with tests to run: an agent. Many tools do all three, so the choice is per task, not per tool.",
                        "nl": "Alle {n} kloppen. Een regel waar je typt: completion. Een vraag waarvan je het antwoord zelf gebruikt: een chat. Werk in meer bestanden met tests om te draaien: een agent. Veel tools doen alle drie, dus je kiest per taak, niet per tool."},
        },
    ),
    "permissions": Sort(
        name="permissions",
        groups={
            "itself": {"en": "By itself", "nl": "Zelf"},
            "ask": {"en": "Asks you first", "nl": "Vraagt jou eerst"},
        },
        items={
            "install": {"en": "**Install a package.** Download code that others wrote, a package, from the internet and install it.",
                        "nl": "**Een package installeren.** Code die anderen schreven, een package, van internet downloaden en installeren."},
            "read": {"en": "**Read the project.** Open the files in the project folder.",
                     "nl": "**Het project lezen.** De bestanden in de projectmap openen."},
            "delete": {"en": "**Delete a folder.** Remove a folder it thinks is no longer needed.",
                       "nl": "**Een map weggooien.** Een map verwijderen die volgens hem niet meer nodig is."},
            "tests": {"en": "**Run the tests.** Run the tests of the project and read what they say.",
                      "nl": "**De tests draaien.** De tests van het project draaien en lezen wat ze zeggen."},
            "publish": {"en": "**Put it online.** Put the code on a website, where others can see it.",
                        "nl": "**Online zetten.** De code op een website zetten, waar anderen hem kunnen zien."},
            "search": {"en": "**Search the code.** Find every place where a function is used.",
                       "nl": "**In de code zoeken.** Elke plek vinden waar een functie wordt gebruikt."},
        },
        solution={"install": "ask", "read": "itself", "delete": "ask", "tests": "itself", "publish": "ask", "search": "itself"},
        nudge={"read": _ITSELF, "tests": _ITSELF, "search": _ITSELF, "install": _ASK, "delete": _ASK, "publish": _ASK},
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the six actions again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de zes handelingen opnieuw uit."},
            "empty": {"en": "Pick for each of the six whether the agent may do it by itself.",
                      "nl": "Kies voor elk van de zes of de agent het zelf mag doen."},
            "unfinished": {"en": "Everything you picked so far is right. Pick for the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe koos, klopt. Kies ook voor de rest (nog {n})."},
            "wrong": {"en": "“{item}” is in the wrong group.",
                      "nl": "‘{item}’ staat in de verkeerde groep."},
            "correct": {"en": "All {n} are right. Reading, searching and running the tests change nothing you cannot undo. A package from the internet is code from strangers that runs on your computer, a deleted folder may be gone for good, and what is online stays online.",
                        "nl": "Alle {n} kloppen. Lezen, zoeken en de tests draaien veranderen niets wat je niet kunt terugdraaien. Een package van internet is code van onbekenden die op je computer draait, een weggegooide map kan voorgoed weg zijn, en wat online staat, blijft online."},
        },
    ),
}

# The lesson is done when these are correct. "after-done" is a choice question in the lesson.
REQUIRED = ["which-help", "permissions", "after-done"]

# What the tests feed each check, and what it must say: (exercise, what the learner chose, state, known mistake).
_HELP = {"error": "chat", "line": "completion", "rename": "agent", "which": "chat", "return": "completion", "option": "agent"}
_PERM = {"install": "ask", "read": "itself", "delete": "ask", "tests": "itself", "publish": "ask", "search": "itself"}
CASES = [
    ("which-help", {}, "empty", None),
    ("which-help", _HELP, "correct", None),
    ("which-help", {"error": "chat", "line": "completion"}, "wrong", "unfinished"),
    ("which-help", {**_HELP, "rename": "chat"}, "wrong", "wrong:rename"),
    ("which-help", {**_HELP, "which": "agent"}, "wrong", "wrong:which"),
    ("which-help", {**_HELP, "return": "chat"}, "wrong", "wrong:return"),
    ("permissions", {}, "empty", None),
    ("permissions", _PERM, "correct", None),
    ("permissions", {"read": "itself"}, "wrong", "unfinished"),
    ("permissions", {**_PERM, "install": "itself"}, "wrong", "wrong:install"),
    ("permissions", {**_PERM, "tests": "ask"}, "wrong", "wrong:tests"),
    ("permissions", {**_PERM, "publish": "itself"}, "wrong", "wrong:publish"),
]
