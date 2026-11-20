# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-safe-design-principles: designing for safety. The learner writes vote, a majority vote of
redundant checkers that falls back to the safe answer when there is no majority, and may_pay, a payment check in which
no one approves their own request (separation of duties). A demo lets one, three or five models vote."""
import contextlib
import copy
import html
import io
import math
import re

from everything_ai import chart, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

# --- the data --------------------------------------------------------------------------------------------------------

ERRORS = [1, 2, 5, 10, 20, 30, 40, 50, 60]   # the demo's slider: the chance that one model is wrong, in %
INVOICES = 10000                              # the invoices the demo counts wrong decisions in
LIMIT = 1000                                  # euros a payment may be without a second person's approval


def vote(answers):
    """What the learner writes in Exercise 1: "pay" only when more than half of all checkers say "pay", else "hold"."""
    pays = 0
    for answer in answers:
        if answer == "pay":
            pays = pays + 1
    if pays > len(answers) / 2:
        return "pay"
    return "hold"


def may_pay(amount, requested_by, approvers):
    """What the learner writes in Exercise 2: up to LIMIT without approval; above it, an approver who is not the one
    who requested the payment."""
    if amount <= LIMIT:
        return True
    for name in approvers:
        if name != requested_by:
            return True
    return False


def majority_wrong(q: float, n: int) -> float:
    """The chance that a majority of n models is wrong when each is wrong with chance q, independently of the others."""
    return sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(n // 2 + 1, n + 1))


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:middle")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_CODE = "font-family:var(--ea-font-code);font-size:0.78rem"
_CHIP = ("display:inline-block;min-width:2.6rem;margin-right:0.2rem;padding:0 0.3rem;border:1px solid var(--ea-line);"
         "border-radius:3px;text-align:center;font-family:var(--ea-font-code);font-size:0.76rem")


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is (its cells, a style of the row), and each cell is (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{h}</th>' for h in head)
    body = "".join(f'<tr style="{style}">' + "".join(f'<td style="{_CELL}{";" + s if s else ""}">{cell}</td>'
                                                    for cell, s in cells_) + "</tr>" for cells_, style in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


def _code(text: str) -> str:
    return f'<span style="{_CODE}">{html.escape(text)}</span>'


def _shown(v) -> str:
    """A result as the pictures show it: a text with double quotes, as the lesson writes it, else what Python shows;
    cut short when it is long."""
    text = '"' + v + '"' if isinstance(v, str) else repr(v)
    return text if len(text) <= 40 else text[:37] + "..."


def _whole(v, lang: str) -> str:
    """A whole number as the text writes it: 1,000 in English, 1000 in Dutch, which uses a dot only from 10.000."""
    n = int(round(float(v)))
    if lang == "en":
        return f"{n:,}"
    return f"{n:,}".replace(",", ".") if abs(n) >= 10000 else str(n)


def _percent(x, lang: str, digits: int = 1) -> str:
    """A chance as a percentage with at most `digits` decimals: 0.028 as 2.8%."""
    text = f"{x * 100:.{digits}f}".rstrip("0").rstrip(".") + "%"
    return text.replace(".", ",") if lang == "nl" else text


_DEMO_WORDS = {
    "en": {"title": "Each model wrong {pct} of the time: what voting gives",
           "head": ["design", "wrong decisions", "in 10,000 invoices"],
           "rows": ["one model", "three copies of one model, voting", "three different models, voting",
                    "five different models, voting"],
           "said": "When each model is wrong {pct} of the time, one model or three copies of it decide wrongly in {one} "
                   "of 10,000 invoices, three different models that vote in {three}, and five in {five}.",
           "x": "chance that one model is wrong, in %", "y": "wrong decisions, in %",
           "one": "one model, or copies", "three": "three different", "five": "five different",
           "note": "Copies give the same answer to the same invoice, so they make the same mistakes. Different models "
                   "here make their mistakes independently of each other. The dashed line is five different models; "
                   "the dots are the chance on the slider."},
    "nl": {"title": "Elk model in {pct} van de gevallen fout: wat stemmen oplevert",
           "head": ["ontwerp", "foute beslissingen", "op 10.000 facturen"],
           "rows": ["één model", "drie kopieën van één model die stemmen",
                    "drie verschillende modellen die stemmen", "vijf verschillende modellen die stemmen"],
           "said": "Als elk model in {pct} van de gevallen fout zit, beslissen één model of drie kopieën ervan fout bij "
                   "{one} van de 10.000 facturen, drie verschillende modellen die stemmen bij {three}, en vijf bij "
                   "{five}.",
           "x": "kans dat één model fout zit, in %", "y": "foute beslissingen, in %",
           "one": "één model, of kopieën", "three": "drie verschillende", "five": "vijf verschillende",
           "note": "Kopieën geven op dezelfde factuur hetzelfde antwoord, dus ze maken dezelfde fouten. Verschillende "
                   "modellen maken hier hun fouten onafhankelijk van elkaar. De gestippelde lijn is vijf verschillende "
                   "modellen; de stippen zijn de kans op de schuif."},
}


def demo_frame(v, lang: str) -> tuple[str, str]:
    """The demo: when each model is wrong v % of the time, how often one model, three copies of it, and three or five
    different models that vote take a wrong decision."""
    v = min(ERRORS, key=lambda m: abs(m - float(v)))
    q = v / 100
    w = _DEMO_WORDS[lang]
    chances = [q, q, majority_wrong(q, 3), majority_wrong(q, 5)]
    rows = [([(html.escape(name), ""), (_code(_percent(c, lang, 2)), "text-align:right"),
              (_code(_whole(INVOICES * c, lang)), "text-align:right")], "") for name, c in zip(w["rows"], chances)]
    xs = list(range(61))   # the chance that one model is wrong, in %
    panel = chart.Panel(x=(0, 60), y=(0, 70), xlabel=w["x"], ylabel=w["y"], xticks=[0, 20, 40, 60],
                        yticks=[0, 20, 40, 60])
    panel.line(xs, xs, role="ink-3")
    panel.line(xs, [100 * majority_wrong(x / 100, 3) for x in xs], role="loss")
    panel.line(xs, [100 * majority_wrong(x / 100, 5) for x in xs], role="loss", width=1.8, dashed=True)
    panel.points([v, v, v], [100 * c for c in chances[1:]], role="loss")
    key = ((64, "one", "ink-3", False), (56, "three", "loss", False), (48, "five", "loss", True))
    for y, name, role, dashed in key:   # a key at the top left, where no line runs
        panel.line([2, 6], [y, y], role=role, width=1.8 if dashed else 2.2, dashed=dashed)
        panel.text(7.5, y, w[name], role=role, anchor="start")
    pct = _percent(q, lang)
    title = w["title"].format(pct=pct)
    said = w["said"].format(pct=pct, one=_whole(INVOICES * chances[0], lang), three=_whole(INVOICES * chances[2], lang),
                            five=_whole(INVOICES * chances[3], lang))
    body = (_table(rows, [html.escape(h) for h in w["head"]])
            + chart.figure([panel], alt=title + ". " + said, width=360, height=240) + _note(w["note"]))
    return yours.figure(title, body, alt=title + ". " + said), said


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _all gives for a function that stopped with an error


def _all(fn, calls: list):
    """The learner's function on copies of each list of arguments in `calls`, all within one time limit of a check;
    _STOPPED when it stops with an error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: [fn(*(copy.deepcopy(a) for a in args)) for args in calls])
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _chips(answers: list) -> str:
    return "".join(f'<span style="{_CHIP}">{html.escape(a if isinstance(a, str) else "—")}</span>' for a in answers)


_VOTE_WORDS = {
    "en": {"title": "Your vote beside the course's",
           "head": ["the checkers' answers", "say pay", "the course's", "yours"],
           "of": "{k} of {n}", "stopped": "error",
           "note": "A dash is a checker that gave no answer, because it failed. The course pays only when more than "
                   "half of all checkers say pay, and holds otherwise. A dashed row is one where your vote gives "
                   "something other than the course's."},
    "nl": {"title": "Jouw vote naast die van de course",
           "head": ["de antwoorden van de controleurs", "zeggen pay", "van de course", "de jouwe"],
           "of": "{k} van {n}", "stopped": "fout",
           "note": "Een streepje is een controleur die geen antwoord gaf, omdat die faalde. De course betaalt alleen als "
                   "meer dan de helft van alle controleurs pay zegt, en houdt anders tegen. Een gestippelde rij is er "
                   "een waar jouw vote iets anders geeft dan die van de course."},
}


def vote_picture(seen) -> str:
    """The cases of the check: each list of answers, how many say pay, the course's decision and the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _VOTE_WORDS[seen.lang]
    got = _all(fn, VOTE_CASES)
    rows = []
    for i, (answers,) in enumerate(VOTE_CASES):
        want = vote(answers)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_shown(got[i])), not (isinstance(got[i], str) and got[i] == want)
        k = sum(1 for a in answers if a == "pay")
        rows.append(([(_chips(answers), ""), (html.escape(w["of"].format(k=k, n=len(answers))), "text-align:right"),
                      (_code(_shown(want)), ""), (shown, "font-weight:600")], _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


_PAY_WORDS = {
    "en": {"title": "Your may_pay beside the course's",
           "head": ["amount", "requested by", "approved by", "the course's", "yours"],
           "nobody": "nobody", "stopped": "error",
           "note": "Up to and including €1,000, a payment needs no approval. Above it, it needs an approver who is not "
                   "the one who requested it. A dashed row is one where your may_pay gives something other than the course's."},
    "nl": {"title": "Jouw may_pay naast die van de course",
           "head": ["bedrag", "aangevraagd door", "goedgekeurd door", "van de course", "de jouwe"],
           "nobody": "niemand", "stopped": "fout",
           "note": "Tot en met €1000 heeft een betaling geen goedkeuring nodig. Daarboven moet iemand goedkeuren die de "
                   "betaling niet zelf heeft aangevraagd. Een gestippelde rij is er een waar jouw may_pay iets anders "
                   "geeft dan die van de course."},
}


def pay_picture(seen) -> str:
    """The cases of the check: each payment, who requested and approved it, the course's answer and the learner's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _PAY_WORDS[seen.lang]
    got = _all(fn, PAY_CASES)
    rows = []
    for i, (amount, requested_by, approvers) in enumerate(PAY_CASES):
        want = may_pay(amount, requested_by, approvers)
        if got is _STOPPED:
            shown, off = html.escape(w["stopped"]), True
        else:
            shown, off = _code(_shown(got[i])), not (isinstance(got[i], bool) and got[i] == want)
        who = ", ".join(approvers) if approvers else w["nobody"]
        rows.append(([(_code("€" + _whole(amount, seen.lang)), "text-align:right"), (html.escape(requested_by), ""),
                      (html.escape(who), ""), (_code(repr(want)), ""), (shown, "font-weight:600")],
                     _OFF if off else ""))
    body = _table(rows, [html.escape(h) for h in w["head"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["vote", "may_pay", "fail_safe", "privilege"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# exercise 1: vote; the first case is a majority that outvotes one checker
VOTE_CASES = [(["pay", "pay", "hold"],), (["pay", "hold", None],), (["pay", None, None],), (["hold", "hold", "pay"],),
              (["pay", "pay", "hold", "hold"],), (["pay", "pay", "pay", None, "hold"],), ([None, None, None],),
              (["pay"],)]
_V_START = "def vote(answers):\n    ...\n"
_V = ("def vote(answers):\n"
      "    pays = 0\n"
      "    for answer in answers:\n"
      "        if answer == \"pay\":\n"
      "            pays = pays + 1\n"
      "    if pays > len(answers) / 2:\n"
      "        return \"pay\"\n"
      "    return \"hold\"\n")
_V_ANSWERED = ("def vote(answers):\n"
               "    pays = 0\n"
               "    holds = 0\n"
               "    for answer in answers:\n"
               "        if answer == \"pay\":\n"
               "            pays = pays + 1\n"
               "        if answer == \"hold\":\n"
               "            holds = holds + 1\n"
               "    if pays > holds:\n"
               "        return \"pay\"\n"
               "    return \"hold\"\n")
_V_HALF = _V.replace("if pays > len(answers) / 2:", "if pays >= len(answers) / 2:")
_V_ALL = _V.replace("if pays > len(answers) / 2:", "if pays == len(answers):")
_V_ANY = _V.replace("if pays > len(answers) / 2:", "if pays > 0:")
_V_PAY_DEFAULT = ("def vote(answers):\n"
                  "    holds = 0\n"
                  "    for answer in answers:\n"
                  "        if answer == \"hold\":\n"
                  "            holds = holds + 1\n"
                  "    if holds > len(answers) / 2:\n"
                  "        return \"hold\"\n"
                  "    return \"pay\"\n")
_V_NO_DEFAULT = _V.replace("    return \"hold\"\n", "")

# exercise 2: may_pay; the first case is a request that its own requester approved
PAY_CASES = [(5000, "agent", ["agent"]), (5000, "agent", ["Sam"]), (800, "agent", []), (5000, "agent", ["agent", "Sam"]),
             (1000, "agent", []), (5000, "agent", []), (1500, "Sam", ["Sam", "Sam"]), (1001, "Kim", ["Sam"])]
_P_START = "LIMIT = 1000\n\n\ndef may_pay(amount, requested_by, approvers):\n    ...\n"
_P = ("LIMIT = 1000\n"
      "\n"
      "\n"
      "def may_pay(amount, requested_by, approvers):\n"
      "    if amount <= LIMIT:\n"
      "        return True\n"
      "    for name in approvers:\n"
      "        if name != requested_by:\n"
      "            return True\n"
      "    return False\n")
_P_SELF = _P.replace("    for name in approvers:\n        if name != requested_by:\n            return True\n    return False\n",
                     "    return len(approvers) > 0\n")
_P_STRICT = _P.replace("if amount <= LIMIT:", "if amount < LIMIT:")
_P_ALWAYS = _P.replace("    if amount <= LIMIT:\n        return True\n", "")
_P_EARLY = _P.replace("            return True\n    return False\n", "            return True\n        return False\n")
_P_EARLY_ELSE = _P.replace("            return True\n    return False\n",
                           "            return True\n        else:\n            return False\n    return False\n")


EXERCISES = {
    "vote": FunctionBox(
        name="vote",
        picture=vote_picture,
        function="vote",
        start=_V_START,
        solution=_V,
        visible=[VOTE_CASES[0]],
        hidden=VOTE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("answered", wrong=_V_ANSWERED,
                    en="`{call}` gives {got}, and it should give {want}: you compare pay with hold, and so count only the checkers that answered. A checker that failed counts too: with two of three failed, one pay is not more than half.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vergelijkt pay met hold, en telt zo alleen de controleurs die antwoordden. Een controleur die faalde telt ook mee: als twee van de drie faalden, is één pay niet meer dan de helft."),
            Mistake("half", wrong=_V_HALF,
                    en="`{call}` gives {got}, and it should give {want}: exactly half is not a majority. Pay only when more than half say pay: use `>`, not `>=`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: precies de helft is geen meerderheid. Betaal alleen als meer dan de helft pay zegt: gebruik `>`, niet `>=`."),
            Mistake("all", wrong=_V_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you pay only when every checker says pay. A majority vote lets two checkers outvote a third that is wrong or has failed.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je betaalt alleen als elke controleur pay zegt. Bij een meerderheid van stemmen overstemmen twee controleurs een derde die fout zit of faalde."),
            Mistake("any", wrong=_V_ANY,
                    en="`{call}` gives {got}, and it should give {want}: you pay as soon as one checker says pay. Pay only when more than half of all checkers say pay.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je betaalt zodra één controleur pay zegt. Betaal alleen als meer dan de helft van alle controleurs pay zegt."),
            Mistake("pay-default", wrong=_V_PAY_DEFAULT,
                    en="`{call}` gives {got}, and it should give {want}: without a majority, your function pays. That is the unsafe side. Without a majority for pay, the safe answer is hold.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: zonder meerderheid betaalt je functie. Dat is de onveilige kant. Zonder meerderheid voor pay is hold het veilige antwoord."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the answers that are \"pay\", and return \"pay\" only when that is more than half of all answers; else return \"hold\".",
                      "nl": "Vervang de `...`: tel de antwoorden die \"pay\" zijn, en geef alleen \"pay\" terug als dat meer dan de helft van alle antwoorden is; geef anders \"hold\" terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count the answers that are \"pay\" in a loop. Return \"pay\" when that count is more than half of `len(answers)`, and \"hold\" otherwise.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel in een loop de antwoorden die \"pay\" zijn. Geef \"pay\" terug als dat aantal meer dan de helft van `len(answers)` is, en anders \"hold\"."},
            "almost": {"en": "Right for the first list. But `{call}` gives {got}, and it should give {want}. Does your function count the answers in the list it gets?",
                       "nl": "Goed voor de eerste lijst. Maar `{call}` geeft {got}, en dat moet {want} zijn. Telt je functie de antwoorden in de lijst die ze krijgt?"},
            "unreached": {"en": "With `{call}`, your function gives `None`: when there is no majority for pay, it never gets to a `return`. Return \"hold\" after the `if`, as the safe answer.",
                          "nl": "Met `{call}` geeft je functie `None`: als er geen meerderheid voor pay is, komt ze nooit bij een `return`. Geef na de `if` \"hold\" terug, als het veilige antwoord."},
            "correct": {"en": "Your `vote` is right in all {n} cases. Under the check, your decision stands beside the course's, for every list of answers.",
                        "nl": "Je `vote` klopt in alle {n} gevallen. Onder de check staat jouw beslissing naast die van de course, voor elke lijst antwoorden."},
        },
    ),
    "may_pay": FunctionBox(
        name="may_pay",
        picture=pay_picture,
        function="may_pay",
        start=_P_START,
        solution=_P,
        visible=[PAY_CASES[0]],
        hidden=PAY_CASES[1:],
        first=[_DOTS],
        mistakes=[
            Mistake("self", wrong=_P_SELF,
                    en="`{call}` gives {got}, and it should give {want}: you let any approval count, also one by the requester. Above the limit, look for an approver whose name is not `requested_by`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je laat elke goedkeuring tellen, ook een van de aanvrager zelf. Zoek boven de grens in `approvers` naar een naam die niet `requested_by` is."),
            Mistake("strict", wrong=_P_STRICT,
                    en="`{call}` gives {got}, and it should give {want}: a payment of exactly €1,000 needs no approval. Use `<=`, not `<`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een betaling van precies €1000 heeft geen goedkeuring nodig. Gebruik `<=`, niet `<`."),
            Mistake("always", wrong=_P_ALWAYS,
                    en="`{call}` gives {got}, and it should give {want}: you ask for approval of every payment. Up to `LIMIT`, a payment may go through without one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je vraagt voor elke betaling goedkeuring. Tot en met `LIMIT` mag een betaling zonder goedkeuring door."),
            Mistake("early", wrong=_P_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives up at the first approver who is the requester, and never looks at the next one. Put `return False` after the loop, level with `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het op bij de eerste goedkeurder die de aanvrager is, en kijkt nooit naar de volgende. Zet `return False` na de loop, op gelijke hoogte met `for`."),
            Mistake("early-else", wrong=_P_EARLY_ELSE,
                    en="`{call}` gives {got}, and it should give {want}: your `else` gives up at the first approver who is the requester, and never looks at the next one. Leave out the `else`, and return `False` only after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je `else` geeft het op bij de eerste goedkeurder die de aanvrager is, en kijkt nooit naar de volgende. Laat de `else` weg, en geef pas na de loop `False` terug."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: return `True` up to `LIMIT`; above it, return `True` only when an approver is someone other than `requested_by`.",
                      "nl": "Vervang de `...`: geef tot en met `LIMIT` `True` terug; geef daarboven alleen `True` terug als een goedkeurder iemand anders is dan `requested_by`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Up to `LIMIT`, return `True`. Above it, go through `approvers` and return `True` at a name that is not `requested_by`; return `False` after the loop.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Geef tot en met `LIMIT` `True` terug. Loop daarboven door `approvers` en geef `True` terug bij een naam die niet `requested_by` is; geef na de loop `False` terug."},
            "almost": {"en": "Right for the first payment. But `{call}` gives {got}, and it should give {want}. Does your function use the amount, the requester and the approvers it gets?",
                       "nl": "Goed voor de eerste betaling. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het bedrag, de aanvrager en de goedkeurders die ze krijgt?"},
            "unreached": {"en": "With `{call}`, your function gives `None`: with nobody in `approvers`, it never gets to a `return`. Return `False` after the loop, level with `for`.",
                          "nl": "Met `{call}` geeft je functie `None`: als er niemand in `approvers` staat, komt ze nooit bij een `return`. Geef na de loop `False` terug, op gelijke hoogte met `for`."},
            "correct": {"en": "Your `may_pay` is right in all {n} cases. Under the check, your answer stands beside the course's, for every payment.",
                        "nl": "Je `may_pay` klopt in alle {n} gevallen. Onder de check staat jouw antwoord naast dat van de course, voor elke betaling."},
        },
    ),
}

CASES = [
    ("vote", _V_START, "empty", None),
    ("vote", _V, "correct", None),
    ("vote", "def vote(answers):\n    pays = 0\n    for answer in answers:\n        if answer == \"pay\":\n            pays += 1\n    if 2 * pays > len(answers):\n        return \"pay\"\n    else:\n        return \"hold\"\n", "correct", None),
    ("vote", "def vote(answers):\n    pays = len([a for a in answers if a == \"pay\"])\n    return \"pay\" if pays > len(answers) // 2 else \"hold\"\n", "correct", None),
    ("vote", _V_ANSWERED, "wrong", "answered"),
    ("vote", _V_HALF, "wrong", "half"),
    ("vote", _V_ALL, "wrong", "all"),
    ("vote", _V_ANY, "wrong", "any"),
    ("vote", _V_PAY_DEFAULT, "wrong", "pay-default"),
    ("vote", _V_NO_DEFAULT, "wrong", "unreached"),
    ("vote", "def vote(answers):\n    return \"pay\"\n", "almost", None),
    ("may_pay", _P_START, "empty", None),
    ("may_pay", _P, "correct", None),
    ("may_pay", "LIMIT = 1000\n\n\ndef may_pay(amount, requested_by, approvers):\n    others = [name for name in approvers if name != requested_by]\n    return amount <= LIMIT or len(others) > 0\n", "correct", None),
    ("may_pay", _P_SELF, "wrong", "self"),
    ("may_pay", _P_STRICT, "wrong", "strict"),
    ("may_pay", _P_ALWAYS, "wrong", "always"),
    ("may_pay", _P_EARLY, "wrong", "early"),
    ("may_pay", _P_EARLY_ELSE, "wrong", "early-else"),
    ("may_pay", "LIMIT = 1000\n\n\ndef may_pay(amount, requested_by, approvers):\n    return False\n", "almost", None),
]
