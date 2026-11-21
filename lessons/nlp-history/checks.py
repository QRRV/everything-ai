# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-history: learning which English word goes with a Dutch word by counting in which
sentence pairs they occur together, as statistical translation did, and scoring systems on one fixed test set, as a
shared task does. The sentence pairs, the test set and the systems' answers are made up by the course; the timeline
holds dated events from the sources the lesson names."""
import contextlib
import copy
import html
import io
import re

from everything_ai import yours
from everything_ai.check import FunctionBox, Mistake, Timeout, same_value, with_time_limit

# --- the course's sentence pairs and shared task ---------------------------------------------------------------------

# A tiny parallel text: Dutch sentences with their English translation, as lists of words.
PAIRS = [
    (["de", "kat", "slaapt"], ["the", "cat", "sleeps"]),
    (["de", "hond", "slaapt"], ["the", "dog", "sleeps"]),
    (["de", "kat", "eet"], ["the", "cat", "eats"]),
    (["een", "kat", "speelt"], ["a", "cat", "plays"]),
    (["de", "hond", "eet"], ["the", "dog", "eats"]),
    (["een", "hond", "speelt"], ["a", "dog", "plays"]),
]
DUTCH = ["kat", "hond", "slaapt", "eet", "speelt", "de", "een"]

# A made-up shared task: is the marked word a noun or a verb? Ten test sentences, the right answers, and the answers
# of three made-up systems.
TEST = ["They *strike* at noon.", "The *strike* lasted a week.", "Kids *play* outside.", "We saw a *play*.",
        "Birds *fly* south.", "A *fly* landed.", "Please *light* the candle.", "The *light* is on.",
        "They *ship* boxes.", "The *ship* is late."]
TEST_NL = ["Ze *staken* om twaalf uur (strike).", "De *staking* (strike) duurde een week.",
           "Kinderen *spelen* (play) buiten.", "We zagen een *toneelstuk* (play).", "Vogels *vliegen* (fly) naar het zuiden.",
           "Er landde een *vlieg* (fly).", "*Steek* de kaars aan (light).", "Het *licht* (light) is aan.",
           "Ze *verschepen* (ship) dozen.", "Het *schip* (ship) is laat."]
ANSWERS = ["verb", "noun", "verb", "noun", "verb", "noun", "verb", "noun", "verb", "noun"]
OUTPUTS = {
    "rules": ["verb", "noun", "verb", "noun", "noun", "noun", "noun", "noun", "noun", "noun"],
    "counts": ["verb", "noun", "verb", "noun", "verb", "noun", "noun", "noun", "verb", "verb"],
    "always noun": ["noun"] * 10,
}

# Events of the timeline: year, the kind of approach, and the words of the event in English and Dutch.
EVENTS = [
    (1954, "rules", "Georgetown–IBM: Russian into English, 250 stems and endings, six rules",
     "Georgetown–IBM: Russisch naar Engels, 250 stammen en uitgangen, zes regels"),
    (1966, "rules", "ELIZA; the ALPAC report sees no prospect of useful machine translation",
     "ELIZA; het ALPAC-rapport ziet geen uitzicht op bruikbare machinevertaling"),
    (1971, "rules", "SHRDLU takes instructions about blocks on a table (described 1971–72)",
     "SHRDLU volgt opdrachten over blokken op een tafel (beschreven in 1971–72)"),
    (1988, "statistics", "IBM presents a statistical approach to translation at COLING in Budapest",
     "IBM presenteert op COLING in Boedapest een statistische aanpak van vertalen"),
    (1993, "statistics", "Penn Treebank: over 4.5 million words of American English, marked word by word",
     "Penn Treebank: ruim 4,5 miljoen woorden Amerikaans Engels, woord voor woord gemarkeerd"),
    (1998, "statistics", "Senseval's first exercise: choosing the meaning of a word",
     "Eerste ronde van Senseval: de betekenis van een woord kiezen"),
    (1999, "statistics", "CoNLL's list of shared tasks begins", "De lijst van shared tasks van CoNLL begint"),
    (2003, "neural", "Bengio et al.: a neural language model, in the journal JMLR (NIPS 2000)",
     "Bengio et al.: een neural language model, in het tijdschrift JMLR (NIPS 2000)"),
    (2006, "statistics", "The first WMT, a workshop with a shared task in translation",
     "De eerste WMT, een workshop met een shared task in vertalen"),
    (2014, "neural", "Several groups translate whole sentences with neural networks",
     "Meerdere groepen vertalen hele zinnen met neural networks"),
    (2017, "neural", "The Transformer, an architecture based on attention",
     "De Transformer, een architectuur die op attention is gebouwd"),
    (2022, "neural", "ChatGPT, a research preview of a model that converses",
     "ChatGPT, een research preview van een model dat een gesprek voert"),
]
KINDS = ["rules", "statistics", "neural"]


def cooccur(word: str, pairs: list) -> dict:
    """The course's own Exercise 1: for each English word, in how many pairs it occurs with the Dutch word."""
    counts = {}
    for dutch, english in pairs:
        if word in dutch:
            for w in english:
                if w in counts:
                    counts[w] = counts[w] + 1
                else:
                    counts[w] = 1
    return counts


def score(predicted: list, answers: list) -> float:
    """The course's own Exercise 2: the share of the answers that the system got right."""
    right = 0
    for i in range(len(answers)):
        if predicted[i] == answers[i]:
            right = right + 1
    return right / len(answers)


def best(counts: dict) -> list:
    """The English words with the highest count, in the order they were counted."""
    top = max(counts.values(), default=0)
    return [w for w, n in counts.items() if n == top and top > 0]


class Named:
    """A value the check gives the learner's function, shown in a message by its name."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


PAIRS_N = Named("PAIRS", PAIRS)
ANSWERS_N = Named("ANSWERS", ANSWERS)
OUTPUTS_N = {name: Named(f'OUTPUTS["{name}"]', out) for name, out in OUTPUTS.items()}


def word_options(lang: str) -> dict:
    """The Dutch words for the demo's dropdown."""
    return {w: w for w in DUTCH}


# --- pictures --------------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_CELL = ("padding:0.15rem 0.45rem;text-align:left;border-bottom:1px solid var(--ea-line);font-family:var(--ea-font-ui);"
         "font-size:0.82rem;color:var(--ea-ink);vertical-align:top")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_OFF = "outline:1.5px dashed var(--ea-loss);outline-offset:-2px"
_BAND = {"rules": "var(--ea-ink-3)", "statistics": "var(--ea-ink-2)", "neural": "var(--ea-ink)"}


def _note(text: str) -> str:
    return f'<div style="{_UI}">{html.escape(text)}</div>'


def _word(text: str) -> str:
    """A short value as Python writes it, kept on one line in a table cell."""
    return f'<span style="font-family:var(--ea-font-code);font-size:12px;white-space:nowrap">{html.escape(text)}</span>'


def _table(rows: list, head: list) -> str:
    """A table with a head row; each row is a list of (html, style)."""
    top = "".join(f'<th style="{_HEAD}">{html.escape(h)}</th>' for h in head)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{";" + style if style else ""}">{cell}</td>'
                                    for cell, style in row) + "</tr>" for row in rows)
    return (f'<div style="overflow-x:auto"><table style="border-collapse:collapse;margin:0.3rem 0">'
            f'<tr>{top}</tr>{body}</table></div>')


_TIMELINE_WORDS = {
    "en": {"title": "From rules to statistics to neural networks",
           "head": ["year", "approach", "what happened"],
           "kinds": {"rules": "rules", "statistics": "statistics", "neural": "neural"},
           "note": "The periods overlap: Jurafsky and Martin date the rule-based period roughly from 1965 to the early "
                   "1990s, and the turn to statistics and neural networks at various times between 1975 and 1988. "
                   "Shared tasks such as Senseval, CoNLL and WMT belong to the statistical period, and kept going "
                   "after it."},
    "nl": {"title": "Van regels via statistiek naar neural networks",
           "head": ["jaar", "aanpak", "wat er gebeurde"],
           "kinds": {"rules": "regels", "statistics": "statistiek", "neural": "neuraal"},
           "note": "De periodes lopen door elkaar: Jurafsky en Martin laten de periode van regels ruwweg lopen van 1965 "
                   "tot begin jaren negentig, en de omslag naar statistiek en neural networks beginnen op verschillende "
                   "momenten tussen 1975 en 1988. Shared tasks zoals Senseval, CoNLL en WMT horen bij de statistische "
                   "periode, en gingen daarna door."},
}


def timeline_picture(lang: str) -> str:
    """The events of the timeline, with the approach each belongs to."""
    w = _TIMELINE_WORDS[lang]
    rows = []
    for year, kind, en, nl in EVENTS:
        mark = (f'<span style="display:inline-block;width:0.6rem;height:0.6rem;border-radius:50%;'
                f'background:{_BAND[kind]};margin-right:0.35rem"></span>')
        rows.append([(str(year), "font-weight:600"), (mark + html.escape(w["kinds"][kind]), ""),
                     (html.escape(nl if lang == "nl" else en), "")])
    said = "; ".join(f"{year}, {w['kinds'][kind]}: {nl if lang == 'nl' else en}" for year, kind, en, nl in EVENTS)
    return yours.figure(w["title"], _table(rows, w["head"]) + _note(w["note"]),
                        alt=f'{w["title"]}. {said}. {w["note"]}')


_DEMO_WORDS = {
    "en": {"title": "Which English word goes with ‘{word}’?", "head": ["English word", "pairs together", ""],
           "one": "The most frequent: {words}.", "tie": "A tie: {words}. Counting alone cannot choose.",
           "note": "Words such as the occur in most pairs, so they get high counts next to most Dutch words. The "
                   "IBM models of 1990 and later did more than count, to see past such words."},
    "nl": {"title": "Welk Engels woord hoort bij {word}?", "head": ["Engels woord", "samen in paren", ""],
           "one": "Het vaakst: {words}.", "tie": "Gelijkspel: {words}. Alleen tellen kan niet kiezen.",
           "note": "Woorden zoals the staan in de meeste paren, dus ze krijgen hoge tellingen naast de meeste Nederlandse woorden. "
                   "De modellen van IBM uit 1990 en later deden meer dan tellen, om door zulke woorden heen te kijken."},
}


def demo_picture(word: str, lang: str) -> str:
    """The counts of the course's cooccur for one Dutch word, with the most frequent English words in ink."""
    word = word if word in DUTCH else DUTCH[0]
    w = _DEMO_WORDS[lang]
    counts = cooccur(word, PAIRS)
    top = best(counts)
    most = max(counts.values())
    rows = []
    for e, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bar = (f'<div style="height:0.7rem;width:{1.2 * n:.1f}rem;border-radius:2px;background:'
               f'{"var(--ea-ink)" if e in top else "var(--ea-ink-3)"}"></div>')
        rows.append([(_word(e), "font-weight:600" if n == most else ""), (str(n), ""), (bar, "")])
    names = ", ".join(top)
    line = (w["one"] if len(top) == 1 else w["tie"]).format(words=names)
    title = w["title"].format(word=word)
    return yours.figure(title, _table(rows, w["head"]) + _note(line) + _note(w["note"]),
                        alt=f'{title}. ' + "; ".join(f"{e}: {n}" for e, n in counts.items()) + f". {line} {w['note']}")


# --- the pictures under the checks -----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error


def _call(fn, *args):
    """The learner's function on copies of args within the time limit of a check; _STOPPED when it stops with an
    error, and the Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _shown(v, lang: str) -> str:
    """What the learner's function gave, for a cell of a table, as Python writes it."""
    if v is _STOPPED:
        return {"en": "error", "nl": "fout"}[lang]
    text = repr(v)
    return text if len(text) <= 70 else text[:67] + "..."


_CHECK_WORDS = {
    "en": {"cooccur": {"title": "Your cooccur on the course's sentence pairs",
                       "head": ["Dutch word", "yours", "the course's"],
                       "fine": "The same as the course's for all seven Dutch words. Counting finds cat for kat, but "
                               "for slaapt, the and sleeps tie.",
                       "differs": "It differs from the course's for {k} of the seven words: they have a dashed frame."},
           "score": {"title": "Your score on the course's shared task",
                     "head": ["system", "yours", "the course's"],
                     "fine": "The same as the course's for all three systems: {scores}.",
                     "differs": "It differs from the course's for {k} of the three systems: they have a dashed frame."}},
    "nl": {"cooccur": {"title": "Jouw cooccur op de zinsparen van de course",
                       "head": ["Nederlands woord", "de jouwe", "die van de course"],
                       "fine": "Bij alle zeven Nederlandse woorden hetzelfde als die van de course. Tellen vindt cat "
                               "bij kat, maar bij slaapt staan the en sleeps gelijk.",
                       "differs": "Bij {k} van de zeven woorden anders dan die van de course: die hebben een gestreept kader."},
           "score": {"title": "Jouw score op de shared task van de course",
                     "head": ["systeem", "de jouwe", "die van de course"],
                     "fine": "Bij alle drie systemen hetzelfde als die van de course: {scores}.",
                     "differs": "Bij {k} van de drie systemen anders dan die van de course: die hebben een gestreept kader."}},
}


def _compare(seen, key: str, items: list) -> str:
    """A table of items, each (label, args, the course's value), with the learner's value beside the course's."""
    fn = _fn(seen)
    if fn is None:
        return ""
    words = _CHECK_WORDS[seen.lang][key]
    rows, said, k = [], [], 0
    for label, args, want in items:
        got = _call(fn, *args)
        differs = got is _STOPPED or not same_value(got, want)
        k += differs
        rows.append([(_word(label), ""), (_word(_shown(got, seen.lang)), _OFF if differs else ""), (_word(repr(want)), "")])
        said.append(f"{label}: {_shown(got, seen.lang)}, {want!r}")
    scores = ", ".join(f"{name} {score(out, ANSWERS)}" for name, out in OUTPUTS.items())
    if seen.lang == "nl":
        scores = scores.replace(".", ",")
    after = words["differs"].format(k=k) if k else words["fine"].format(scores=scores)
    return yours.figure(words["title"], _table(rows, words["head"]) + _note(after),
                        alt=f'{words["title"]}: ' + "; ".join(said) + ". " + after)


def cooccur_picture(seen) -> str:
    """The seven Dutch words, with the learner's counts beside the course's."""
    return _compare(seen, "cooccur", [(w, (w, PAIRS), cooccur(w, PAIRS)) for w in DUTCH])


def score_picture(seen) -> str:
    """The three systems of the shared task, with the learner's score beside the course's."""
    return _compare(seen, "score", [(name, (out, ANSWERS), score(out, ANSWERS)) for name, out in OUTPUTS.items()])


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["cooccur", "score", "linguist", "compare"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _names_error(e) -> bool:
    return isinstance(e, NameError) and re.search(r"'(PAIRS|ANSWERS|OUTPUTS|C)'", str(e)) is not None


_NAMES = Mistake("lesson-names", error=_names_error,
                 en="Your function uses a name of the lesson, such as `PAIRS`, `ANSWERS` or `C`, and the check does not give it that name: use the names between the parentheses of the `def` line.",
                 nl="Je functie gebruikt een naam van de les, zoals `PAIRS`, `ANSWERS` of `C`, en de check geeft die naam niet mee: gebruik de namen tussen de haakjes van de `def`-regel.")

_KEY = Mistake("not-yet-counted", error=lambda e: isinstance(e, KeyError),
               en="Your function stops with a `KeyError`: it adds 1 to the count of an English word that has no count yet. The first time a word comes by, give it the count 1.",
               nl="Je functie stopt met een `KeyError`: ze telt 1 op bij een Engels woord dat nog geen telling heeft. De eerste keer dat een woord langskomt, geef je het de telling 1.")

# exercise 1: cooccur
COOCCUR_CASES = [("kat", PAIRS_N)] + [(w, PAIRS_N) for w in DUTCH[1:]] + [
    ("x", PAIRS_N), ("a", [(["a", "b"], ["c", "d"]), (["b", "a"], ["d"]), (["b"], ["c"])])]
_O_START = "def cooccur(word, pairs):\n    ...\n"
_O = ("def cooccur(word, pairs):\n"
      "    counts = {}\n"
      "    for dutch, english in pairs:\n"
      "        if word in dutch:\n"
      "            for w in english:\n"
      "                if w in counts:\n"
      "                    counts[w] = counts[w] + 1\n"
      "                else:\n"
      "                    counts[w] = 1\n"
      "    return counts\n")
_O_ALL = ("def cooccur(word, pairs):\n"
          "    counts = {}\n"
          "    for dutch, english in pairs:\n"
          "        for w in english:\n"
          "            if w in counts:\n"
          "                counts[w] = counts[w] + 1\n"
          "            else:\n"
          "                counts[w] = 1\n"
          "    return counts\n")
_O_DUTCH = _O.replace("            for w in english:\n", "            for w in dutch:\n")
_O_RESET = ("def cooccur(word, pairs):\n"
            "    for dutch, english in pairs:\n"
            "        counts = {}\n"
            "        if word in dutch:\n"
            "            for w in english:\n"
            "                if w in counts:\n"
            "                    counts[w] = counts[w] + 1\n"
            "                else:\n"
            "                    counts[w] = 1\n"
            "    return counts\n")
_O_ZERO = _O.replace("                    counts[w] = 1\n", "                    counts[w] = 0\n")

# exercise 2: score
SCORE_CASES = [(OUTPUTS_N["rules"], ANSWERS_N), (OUTPUTS_N["counts"], ANSWERS_N), (OUTPUTS_N["always noun"], ANSWERS_N),
               (["a", "b"], ["a", "c"]), (["x"], ["x"]), (["p", "q", "r", "s"], ["s", "r", "q", "p"])]
_S_START = "def score(predicted, answers):\n    ...\n"
_S = ("def score(predicted, answers):\n"
      "    right = 0\n"
      "    for i in range(len(answers)):\n"
      "        if predicted[i] == answers[i]:\n"
      "            right = right + 1\n"
      "    return right / len(answers)\n")
_S_COUNT = _S.replace("    return right / len(answers)\n", "    return right\n")
_S_FLIP = _S.replace("predicted[i] == answers[i]", "predicted[i] != answers[i]")
_S_EARLY = _S.replace("    return right / len(answers)\n", "        return right / len(answers)\n")


EXERCISES = {
    "cooccur": FunctionBox(
        name="cooccur",
        picture=cooccur_picture,
        function="cooccur",
        start=_O_START,
        solution=_O,
        visible=[COOCCUR_CASES[0]],
        hidden=COOCCUR_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            _KEY,
            Mistake("all", wrong=_O_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the English words of every pair, also of pairs without the Dutch word. Count only in the pairs whose Dutch sentence holds `word`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie telt de Engelse woorden van elk paar, ook van paren zonder het Nederlandse woord. Tel alleen in de paren waarvan de Nederlandse zin `word` bevat."),
            Mistake("dutch", wrong=_O_DUTCH,
                    en="`{call}` gives {got}, and it should give {want}: those are Dutch words. Count the English words of each pair, the second list.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat zijn Nederlandse woorden. Tel de Engelse woorden van elk paar, de tweede lijst."),
            Mistake("reset", wrong=_O_RESET,
                    en="`{call}` gives {got}, and it should give {want}: only the last pair is counted. Your function starts an empty dict again for every pair; make it once, before the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: alleen het laatste paar telt. Jouw functie begint bij elk paar opnieuw met een lege dict; maak die één keer, vóór de loop."),
            Mistake("zero", wrong=_O_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: every count is one too low. The first time a word comes by, it has been seen once.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: elke telling is één te laag. De eerste keer dat een woord langskomt, is het één keer gezien."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start an empty dict, go over the pairs, and count the English words of each pair whose Dutch sentence holds `word`.",
                      "nl": "Vervang de `...`: begin een lege dict, ga de paren langs, en tel de Engelse woorden van elk paar waarvan de Nederlandse zin `word` bevat."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives a dict from each English word to the number of pairs in which it occurs with `word`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft een dict van elk Engels woord naar het aantal paren waarin het samen met `word` voorkomt."},
            "almost": {"en": "Right for *kat*. But `{call}` gives {got}, and it should give {want}. Does your function use the `word` and `pairs` it gets?",
                       "nl": "Goed voor *kat*. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie het `word` en de `pairs` die ze krijgt?"},
            "correct": {"en": "Your `cooccur` is right in all {n} cases. For *kat*, *cat* comes out on top with 3, ahead of *the* with 2: a translation learned from data, with no dictionary. But for *slaapt*, *the* and *sleeps* tie: words that occur everywhere get in the way.",
                        "nl": "Je `cooccur` klopt in alle {n} gevallen. Bij *kat* komt *cat* bovenaan met 3, vóór *the* met 2: een vertaling geleerd uit data, zonder woordenboek. Maar bij *slaapt* staan *the* en *sleeps* gelijk: woorden die overal staan, zitten in de weg."},
        },
    ),
    "score": FunctionBox(
        name="score",
        picture=score_picture,
        function="score",
        start=_S_START,
        solution=_S,
        visible=[SCORE_CASES[0]],
        hidden=SCORE_CASES[1:],
        first=[_DOTS],
        mistakes=[
            _NAMES,
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: that is the number of right answers. Divide by the number of answers, so that test sets of any size give a share between 0 and 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: dat is het aantal goede antwoorden. Deel door het aantal antwoorden, zodat testsets van elke grootte een deel tussen 0 en 1 geven."),
            Mistake("flip", wrong=_S_FLIP,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the answers that differ. Count the ones that are the same.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie telt de antwoorden die verschillen. Tel de antwoorden die hetzelfde zijn."),
            Mistake("early", wrong=_S_EARLY,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back a score after the first answer. Put the `return` after the loop.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: jouw functie geeft na het eerste antwoord al een score terug. Zet de `return` na de loop."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: count the places where the two lists have the same answer, and divide by the number of answers.",
                      "nl": "Vervang de `...`: tel de plekken waar de twee lijsten hetzelfde antwoord hebben, en deel door het aantal antwoorden."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The function gives the share of places where `predicted` and `answers` hold the same answer: a number between 0 and 1.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De functie geeft het deel van de plekken waar `predicted` en `answers` hetzelfde antwoord hebben: een getal tussen 0 en 1."},
            "almost": {"en": "Right for the system with rules. But `{call}` gives {got}, and it should give {want}. Does your function use the `predicted` and `answers` it gets?",
                       "nl": "Goed voor het systeem met regels. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `predicted` en `answers` die ze krijgt?"},
            "correct": {"en": "Your `score` is right in all {n} cases. On the same ten sentences, the system that counts gets 0.8, the one with rules 0.7, and always saying noun already gets 0.5. Only because all three ran on the same test set can you put those numbers side by side.",
                        "nl": "Je `score` klopt in alle {n} gevallen. Op dezelfde tien zinnen haalt het systeem dat telt 0,8, dat met regels 0,7, en een systeem dat altijd zelfstandig naamwoord zegt, haalt al 0,5. Alleen omdat alle drie op dezelfde testset draaiden, kun je die getallen naast elkaar zetten."},
        },
    ),
}

CASES = [
    ("cooccur", _O_START, "empty", None),
    ("cooccur", _O, "correct", None),
    ("cooccur", "def cooccur(word, pairs):\n    counts = {}\n    for pair in pairs:\n        if word in pair[0]:\n            for w in pair[1]:\n                counts[w] = counts.get(w, 0) + 1\n    return counts\n", "correct", None),
    ("cooccur", _O_ALL, "wrong", "all"),
    ("cooccur", _O_DUTCH, "wrong", "dutch"),
    ("cooccur", _O_RESET, "wrong", "reset"),
    ("cooccur", _O_ZERO, "wrong", "zero"),
    ("cooccur", "def cooccur(word, pairs):\n    counts = {}\n    for dutch, english in pairs:\n        if word in dutch:\n            for w in english:\n                counts[w] = counts[w] + 1\n    return counts\n", "error", "not-yet-counted"),
    ("cooccur", "def cooccur(word, pairs):\n    return {'the': 2, 'cat': 3, 'sleeps': 1, 'eats': 1, 'a': 1, 'plays': 1}\n", "almost", None),
    ("cooccur", "def cooccur(word, pairs):\n    return PAIRS\n", "error", "lesson-names"),
    ("score", _S_START, "empty", None),
    ("score", _S, "correct", None),
    ("score", "def score(predicted, answers):\n    same = [p == a for p, a in zip(predicted, answers)]\n    return sum(same) / len(same)\n", "correct", None),
    ("score", _S_COUNT, "wrong", "count"),
    ("score", _S_FLIP, "wrong", "flip"),
    ("score", _S_EARLY, "wrong", "early"),
    ("score", "def score(predicted, answers):\n    return 0.7\n", "almost", None),
    ("score", "def score(predicted, answers):\n    return len(ANSWERS)\n", "error", "lesson-names"),
]
