# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson llm-quality-filter: clean_lines, the line rules of C4; keep_page, four page rules of Gopher;
features, four numbers per page for a quality classifier; and a choice. The demo downloads the first 20 MB of a shard
of C4 before its cleaning, raw web pages, once, and runs the learner's three functions on them; then it trains the
learner's logistic regression of ml-train-logistic to tell short stories from the cleaned pages, and scores every page.
The course hosts none of the pages: a learner downloads them, and the course shares no model trained on them."""
import errno
import html
import json
import urllib.error
import zlib
from functools import cache

import numpy as np

from everything_ai import hub, paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code

C4_SAMPLE = ("datasets/allenai/c4", "en.noclean/c4-train.00000-of-07168.json.gz")
DOWNLOAD_LINE = {
    "en": "Works on every computer. The first 20 MB of a shard of C4, raw web pages, are downloaded once from Hugging Face when you click their button in the lesson; after that, the lesson works without a connection. The exercises work without them too.",
    "nl": "Werkt op elke computer. De eerste 20 MB van een shard van C4, ruwe webpagina's, worden één keer gedownload van Hugging Face als je in de les op de knop ervoor klikt; daarna werkt de les zonder verbinding. De oefeningen werken ook zonder.",
}
END = (".", "!", "?", '"')   # the marks that end a sentence, for C4
STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}   # the stop words of Gopher's rule
WORDS = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "on", "with", "as", "is", "was",
         "he", "she", "you", "they"}   # the 20 common words of the features
STORIES = "data/tinystories-small.txt"
N_TRAIN = 2000   # stories and cleaned pages the classifier learns from
LR, STEPS = 1.0, 3000

# --- measured: the course's run of the demo with its own solutions (see the notes) ---
# The course's run of the demo with its own three functions and fit, on the download, one thread: the pages, their
# characters, the characters after the line rules, the pages kept by the page rules and their characters, the pages
# the classifier scores above 0.5 and the median score; and the scores of four pages the lesson shows, by their place.
MEASURED = {"pages": 8690, "chars": 54284802, "chars_lines": 18064868, "kept": 5614, "chars_kept": 17529723,
            "above": 1040, "median": 0.287, "examples": {3715: 0.0002, 22: 0.0001, 368: 0.937, 7149: 0.975}}
# --- end of measured ---


def downloaded() -> bool:
    return hub.pinned(*C4_SAMPLE) is not None


@cache
def pages() -> list:
    """The web pages of the download: the gzip stream decoded as far as it goes, one page per line of JSON; the last
    line, cut off at 20 MB, is left out."""
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(hub.pinned(*C4_SAMPLE).read_bytes())
    return [json.loads(line)["text"] for line in raw.split(b"\n")[:-1]]


@cache
def stories() -> list:
    text = (paths.ROOT / STORIES).read_text(encoding="utf-8")
    return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".") if lang == "nl" else text


def sample_or_stop(button, texts: dict):
    """The pages for the demo's cells. When they are not downloaded yet, the cell stops with `texts["ask"]` and the
    button; a click downloads them, with the progress in the cell. `texts` has "ask", "busy" ({done} and {total} in MB),
    "no_net", "not_pinned", "no_room" and "failed" ({why}): why a download did not work."""
    import marimo as mo
    failed = None
    if not downloaded() and button.value:
        total = round(hub.PINNED[C4_SAMPLE]["size"] / 1e6)

        def progress(done, size):
            mo.output.replace(mo.md(texts["busy"].format(done=int(done / 1e6), total=total)))
        try:
            hub.download(*C4_SAMPLE, progress=progress)
        except hub.NotPinned:
            failed = texts["not_pinned"]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            failed = texts["no_net"]
        except OSError as e:
            failed = texts["no_room"] if e.errno == errno.ENOSPC else texts["failed"].format(why=html.escape(str(e)))
    if not downloaded():
        parts = [mo.md(failed)] if failed else []
        mo.stop(True, mo.vstack(parts + [mo.md(texts["ask"]), button]))
    return pages()


# The lesson is done when these are correct. "bias" is the choice question in the lesson.
REQUIRED = ["clean-lines", "keep-page", "features", "bias"]

_DOTS = Mistake("dots-left", matches=lambda got, text: any(line.strip() == "..." or line.rstrip().endswith("= ...")
                                                          or line.rstrip().endswith("(...)") for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}
_CALL = "`{call}` gives `{got}`, and it should give `{want}`"
_CALL_NL = "`{call}` geeft `{got}`, en dat moet `{want}` zijn"

# --- exercise 1: clean_lines -----------------------------------------------------------------------------------------

_L_HEAD = "END = (\".\", \"!\", \"?\", '\"')\n\n\ndef clean_lines(page):\n    kept = []\n    for line in page.split(\"\\n\"):\n"
_L_STRIP = "        line = line.strip()\n"
_L_TEST = ("        if line.endswith(END) and len(line.split()) >= 5 and \"javascript\" not in line.lower():\n"
           "            kept.append(line)\n")
_L_TAIL = "    return \"\\n\".join(kept)\n"
_L_START = (_L_HEAD + "        # the line without spaces at its ends; keep it if it ends a sentence, has at least 5 words,\n"
            "        # and does not mention javascript\n        ...\n" + _L_TAIL)


def _l(strip: str = _L_STRIP, test: str = _L_TEST, tail: str = _L_TAIL) -> str:
    return _L_HEAD + strip + test + tail


PAGE = ("Home\nAbout us\nThe weather was warm and the sea was calm.\nPlease enable JavaScript to see this page.\n"
        "Share this!\nWe walked along the beach until the sun went down.")

# --- exercise 2: keep_page -------------------------------------------------------------------------------------------

_P_HEAD = "STOP_WORDS = {\"the\", \"be\", \"to\", \"of\", \"and\", \"that\", \"have\", \"with\"}\n\n\ndef keep_page(text):\n    words = text.split()\n"
_P_LEN = "    if len(words) < 50 or len(words) > 100000:\n        return False\n"
_P_MEAN = ("    mean = 0\n    for w in words:\n        mean = mean + len(w)\n    mean = mean / len(words)\n"
           "    if mean < 3 or mean > 10:\n        return False\n")
_P_STOP = ("    seen = set()\n    for w in words:\n        if w.lower() in STOP_WORDS:\n            seen.add(w.lower())\n"
           "    if len(seen) < 2:\n        return False\n")
_P_ALPHA = ("    letters = 0\n    for w in words:\n        has = False\n        for c in w:\n            if c.isalpha():\n"
            "                has = True\n        if has:\n            letters = letters + 1\n"
            "    if letters < 0.8 * len(words):\n        return False\n")
_P_START = (_P_HEAD + "    # 1. between 50 and 100,000 words\n    ...\n    # 2. a mean word length between 3 and 10\n    ...\n"
            "    # 3. at least 2 different stop words\n    ...\n    # 4. at least 80% of the words have a letter\n    ...\n"
            "    return True\n")


def _p(length: str = _P_LEN, mean: str = _P_MEAN, stop: str = _P_STOP, alpha: str = _P_ALPHA) -> str:
    return _P_HEAD + length + mean + stop + alpha + "    return True\n"


def _words(n: int, base: str = "the cat sat on a warm mat and looked at the birds with care") -> str:
    """A text of exactly n words, from the words of `base` over and over."""
    w = base.split()
    return " ".join(w[i % len(w)] for i in range(n))


FIRST_PAGE = _words(60)

# --- exercise 3: features --------------------------------------------------------------------------------------------

_F_HEAD = ("END = (\".\", \"!\", \"?\", '\"')\nWORDS = {\"the\", \"be\", \"to\", \"of\", \"and\", \"a\", \"in\", \"that\", \"have\", \"it\", \"for\",\n"
           "         \"on\", \"with\", \"as\", \"is\", \"was\", \"he\", \"she\", \"you\", \"they\"}\n\n\ndef features(page):\n")
_F_LINES = ("    lines = []\n    for line in page.split(\"\\n\"):\n        if line.strip() != \"\":\n            lines.append(line.strip())\n"
            "    words = page.split()\n    if len(lines) == 0 or len(words) == 0:\n        return [0.0, 0.0, 0.0, 0.0]\n")
_F_ENDS = "    ends = 0\n    for line in lines:\n        if line.endswith(END):\n            ends = ends + 1\n    ends = ends / len(lines)\n"
_F_STOP = "    stop = 0\n    for w in words:\n        if w.lower() in WORDS:\n            stop = stop + 1\n    stop = stop / len(words)\n"
_F_LETTERS = ("    letters = 0\n    for c in page:\n        if c.isalpha() or c.isspace():\n            letters = letters + 1\n"
              "    letters = letters / len(page)\n")
_F_PER = "    per_line = min(len(words) / len(lines) / 20, 1.0)\n"
_F_TAIL = "    return [ends, stop, letters, per_line]\n"
_F_START = (_F_HEAD + "    # the lines that are not empty, without spaces at their ends, and the words\n" + _F_LINES[_F_LINES.index("    words"):].replace("    words = page.split()\n", "    lines = ...\n    words = page.split()\n")
            + "    # 1. the share of the lines that end a sentence\n    ends = ...\n"
            "    # 2. the share of the words that are one of WORDS, in lower case\n    stop = ...\n"
            "    # 3. the share of the characters that are letters or spaces\n    letters = ...\n"
            "    # 4. the words per line, divided by 20, and at most 1\n    per_line = ...\n" + _F_TAIL)


def _f(lines: str = _F_LINES, ends: str = _F_ENDS, stop: str = _F_STOP, letters: str = _F_LETTERS, per: str = _F_PER) -> str:
    return _F_HEAD + lines + ends + stop + letters + per + _F_TAIL


FEATURE_PAGE = "Home\n\nThe cat sat on the mat.\nIt was a warm day and she was happy.\nContact: info@example.com"


def _round_list(v):
    return [round(x, 4) for x in v] if isinstance(v, (list, tuple)) else v


def _close_lists(got, want) -> bool:
    try:
        return len(got) == len(want) and all(abs(float(g) - float(w)) < 1e-6 for g, w in zip(got, want))
    except (TypeError, ValueError):
        return False


def _shown(v) -> str:
    if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
        return "[" + ", ".join(f"{x:.4f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x) for x in v) + "]"
    return repr(v)


EXERCISES = {
    "clean-lines": FunctionBox(
        name="clean-lines",
        function="clean_lines",
        start=_L_START,
        solution=_l(),
        visible=[(PAGE,)],
        hidden=[("",), ("One two three four five.",), ("One two three four.",), ("Read this page with JavaScript turned on, please.",),
                ("This line has spaces at its end and more words.   \n   And this one has spaces at its start, five words!",),
                ("He said: \"we will come back tomorrow.\"\nWhat do you think about all of this?\nLook at me now!!",),
                ("menu\nlogin\nSearch for a word\n\nThis page uses cookies to give you the best experience.",),
                ("Click here to read the whole story\nShare this!",), ("Turn on javascript to see the photos of this page.",),
                ("Only  four   words here.",)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("endswith-list", error=lambda e: isinstance(e, TypeError) and "endswith" in str(e),
                    en="With `{call}`, your function stops: `endswith` takes one text, or a tuple of texts, but not a list. Give it the tuple `END`: `line.endswith(END)`.",
                    nl="Met `{call}` stopt je functie: `endswith` neemt één tekst, of een tuple van teksten, maar geen lijst. Geef hem de tuple `END`: `line.endswith(END)`."),
            Mistake("no-strip", wrong=_l(strip=""),
                    en=_CALL + ": a line with spaces after its full stop does not end in a full stop. Take the spaces off both ends first: `line = line.strip()`.",
                    nl=_CALL_NL + ": een regel met spaties na zijn punt eindigt niet op een punt. Haal eerst de spaties aan beide kanten weg: `line = line.strip()`."),
            Mistake("case", wrong=_l(test=_L_TEST.replace("\"javascript\" not in line.lower()", "\"javascript\" not in line")),
                    en=_CALL + ": \"JavaScript\" with capitals slips through. Look for the word in the line in lower case: `\"javascript\" not in line.lower()`.",
                    nl=_CALL_NL + ": \"JavaScript\" met hoofdletters glipt erdoor. Zoek het woord in de regel in kleine letters: `\"javascript\" not in line.lower()`."),
            Mistake("chars", wrong=_l(test=_L_TEST.replace("len(line.split()) >= 5", "len(line) >= 5")),
                    en=_CALL + ": you count the characters of a line, not its words. `line.split()` gives the words: at least 5 of them.",
                    nl=_CALL_NL + ": je telt de tekens van een regel, niet zijn woorden. `line.split()` geeft de woorden: minstens 5 ervan."),
            Mistake("or", wrong=_l(test=_L_TEST.replace("line.endswith(END) and len(line.split()) >= 5", "(line.endswith(END) or len(line.split()) >= 5)")),
                    en=_CALL + ": your line passes when it ends a sentence or has 5 words. C4 keeps a line only when both hold: `and`.",
                    nl=_CALL_NL + ": je regel komt erdoor als hij een zin afsluit of 5 woorden heeft. C4 houdt een regel alleen als allebei gelden: `and`."),
            Mistake("list", wrong=_l(tail="    return kept\n"),
                    en=_CALL + ": your function gives the kept lines as a list. Give them back as one text, a line each: `\"\\n\".join(kept)`.",
                    nl=_CALL_NL + ": je functie geeft de bewaarde regels als lijst. Geef ze terug als één tekst, één regel elk: `\"\\n\".join(kept)`."),
            Mistake("space-join", wrong=_l(tail="    return \" \".join(kept)\n"),
                    en=_CALL + ": your kept lines are joined by spaces, so they become one long line. Join them with a new line: `\"\\n\".join(kept)`.",
                    nl=_CALL_NL + ": je bewaarde regels zijn met spaties aan elkaar gezet, dus ze worden één lange regel. Zet ze aan elkaar met een nieuwe regel: `\"\\n\".join(kept)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...`: strip the line, then keep it when it passes all three rules.",
                      "nl": "Vervang de `...`: haal de spaties van de regel, en bewaar hem als hij aan alle drie de eisen voldoet."},
            "wrong": {"en": _CALL + ". Every line stripped, and kept only when it ends with one of `END`, has at least 5 words and does not mention javascript, in any case?",
                      "nl": _CALL_NL + ". Elke regel zonder spaties aan de kanten, en alleen bewaard als hij eindigt op een van `END`, minstens 5 woorden heeft en javascript niet noemt, in welke letters ook?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `page`, and no fixed lines?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `page`, en geen vaste regels?"},
            "correct": {"en": "Your function keeps the lines of a page that read as sentences, as C4's cleaning did: menus, buttons and notices about JavaScript go.",
                        "nl": "Je functie houdt de regels van een pagina die als zinnen lezen, zoals het opschonen van C4 deed: menu's, knoppen en meldingen over JavaScript gaan eruit."},
        },
    ),
    "keep-page": FunctionBox(
        name="keep-page",
        function="keep_page",
        start=_P_START,
        solution=_p(),
        visible=[(FIRST_PAGE,)],
        hidden=[(_words(49),), (_words(50),), ("",), (_words(60, "the and extraordinarily unbelievable circumstances characterized international"),),
                (_words(60, "a an to of is it on in at be by or we do go up so no he me"),),
                (_words(60, "the and misinterpretation comprehensibility"),), (_words(60, "the and cat dog fox owl hen bee 123 456"),),
                (_words(60, "the the the house stood on a hill"),), (_words(60, "The Cat Sat on a Mat And looked at The birds"),),
                (_words(60, "the and 123 456 789 000 111 222 333 444 555"),), (_words(60, "the and 123 cat dog fox owl hen bee ant"),),
                (_words(100000),), (_words(100001),)],
        python_values=True,
        first=[_DOTS],
        mistakes=[
            Mistake("count-stop", wrong=_p(stop="    seen = 0\n    for w in words:\n        if w.lower() in STOP_WORDS:\n            seen = seen + 1\n"
                                                "    if seen < 2:\n        return False\n"),
                    en=_CALL + ": you count how often stop words occur, so \"the\" three times counts as three. Gopher asks for at least 2 different ones: collect them in a set.",
                    nl=_CALL_NL + ": je telt hoe vaak stopwoorden voorkomen, dus drie keer \"the\" telt als drie. Gopher vraagt er minstens 2 verschillende: verzamel ze in een set."),
            Mistake("no-lower", wrong=_p(stop=_P_STOP.replace("w.lower() in STOP_WORDS", "w in STOP_WORDS").replace("seen.add(w.lower())", "seen.add(w)")),
                    en=_CALL + ": \"The\" with a capital is not in `STOP_WORDS`. Look the words up in lower case: `w.lower()`.",
                    nl=_CALL_NL + ": \"The\" met een hoofdletter staat niet in `STOP_WORDS`. Zoek de woorden op in kleine letters: `w.lower()`."),
            Mistake("bounds", wrong=_p(length="    if len(words) <= 50 or len(words) > 100000:\n        return False\n"),
                    en=_CALL + ": a page of exactly 50 words is thrown out, and Gopher keeps it: between 50 and 100,000 words, both included. Throw out fewer than 50: `len(words) < 50`.",
                    nl=_CALL_NL + ": een pagina van precies 50 woorden gaat eruit, en Gopher houdt hem: tussen 50 en 100.000 woorden, allebei meegerekend. Gooi minder dan 50 eruit: `len(words) < 50`."),
            Mistake("no-mean", wrong=_p(mean=""),
                    en=_CALL + ": a page of very long words passes. Rule 2 throws out a page whose words are on average shorter than 3 or longer than 10 characters.",
                    nl=_CALL_NL + ": een pagina met heel lange woorden komt erdoor. Eis 2 gooit een pagina eruit waarvan de woorden gemiddeld korter dan 3 of langer dan 10 tekens zijn."),
            Mistake("no-alpha", wrong=_p(alpha=""),
                    en=_CALL + ": a page that is mostly numbers passes. Rule 4: at least 80% of the words must have a letter in them, `c.isalpha()`.",
                    nl=_CALL_NL + ": een pagina die vooral uit getallen bestaat, komt erdoor. Eis 4: minstens 80% van de woorden moet een letter bevatten, `c.isalpha()`."),
            # the pages of 100,000 and 100,001 words are too long to show in a message: these two say it without them
            Mistake("no-upper", wrong=_p(length="    if len(words) < 50:\n        return False\n"),
                    en="Your function keeps a page of 100,001 words, and Gopher throws out a page of more than 100,000 words: `len(words) > 100000`.",
                    nl="Je functie houdt een pagina van 100.001 woorden, en Gopher gooit een pagina met meer dan 100.000 woorden eruit: `len(words) > 100000`."),
            Mistake("upper-included", wrong=_p(length="    if len(words) < 50 or len(words) >= 100000:\n        return False\n"),
                    en="Your function throws out a page of exactly 100,000 words, and Gopher keeps it: between 50 and 100,000 words, both included. Throw out more than 100,000: `len(words) > 100000`.",
                    nl="Je functie gooit een pagina van precies 100.000 woorden eruit, en Gopher houdt hem: tussen 50 en 100.000 woorden, allebei meegerekend. Gooi meer dan 100.000 eruit: `len(words) > 100000`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the four `...`: one rule each, every one with `return False` when the page breaks it.",
                      "nl": "Vervang de vier `...`: één eis elk, elk met `return False` als de pagina hem niet haalt."},
            "wrong": {"en": _CALL + ". The four rules in turn: 50 to 100,000 words, a mean word length of 3 to 10, 2 different stop words, and 80% of the words with a letter?",
                      "nl": _CALL_NL + ". De vier eisen om de beurt: 50 tot 100.000 woorden, een gemiddelde woordlengte van 3 tot 10, 2 verschillende stopwoorden, en 80% van de woorden met een letter?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Do all four rules use `words`, with the numbers of the task?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruiken alle vier de eisen `words`, met de getallen van de opdracht?"},
            "correct": {"en": "Your function throws out a page that is too short or too long, of odd words, without the words of ordinary sentences, or mostly numbers and symbols: four of the rules of Gopher.",
                        "nl": "Je functie gooit een pagina eruit die te kort of te lang is, uit vreemde woorden bestaat, zonder de woorden van gewone zinnen, of vooral uit getallen en tekens: vier van de eisen van Gopher."},
        },
    ),
    "features": FunctionBox(
        name="features",
        function="features",
        start=_F_START,
        solution=_f(),
        visible=[(FEATURE_PAGE,)],
        hidden=[("",), ("\n\n",), ("One two three four five six.",), ("A list\nof words\nwithout ends",),
                ("The THE The tHe the and And AND.",), (" ".join(["word"] * 50) + ".",), ("1234 5678 !!!!",),
                ("  spaces around  \n\n   more spaces here.  ",)],
        compare=_close_lists,
        show=_shown,
        first=[_DOTS],
        mistakes=[
            Mistake("empty-lines", wrong=_f(lines=_F_LINES.replace("        if line.strip() != \"\":\n            lines.append(line.strip())\n", "        lines.append(line.strip())\n")),
                    en=_CALL + ": your empty lines count too, so the share of lines that end a sentence comes out too low. Keep only the lines that are not empty after `strip()`.",
                    nl=_CALL_NL + ": je lege regels tellen mee, dus het aandeel regels dat een zin afsluit, komt te laag uit. Houd alleen de regels over die na `strip()` niet leeg zijn."),
            Mistake("no-space", wrong=_f(letters=_F_LETTERS.replace("c.isalpha() or c.isspace()", "c.isalpha()")),
                    en=_CALL + ": feature 3 counts letters and spaces, and yours counts only letters. Count a space as well: `c.isalpha() or c.isspace()`.",
                    nl=_CALL_NL + ": feature 3 telt letters en spaties, en het jouwe telt alleen letters. Tel een spatie ook: `c.isalpha() or c.isspace()`."),
            Mistake("no-cap", wrong=_f(per="    per_line = len(words) / len(lines) / 20\n"),
                    en=_CALL + ": a page with very long lines gets a fourth number above 1. Cap it at 1: `min(..., 1.0)`.",
                    nl=_CALL_NL + ": een pagina met heel lange regels krijgt een vierde getal boven 1. Maximeer het op 1: `min(..., 1.0)`."),
            Mistake("stop-case", wrong=_f(stop=_F_STOP.replace("w.lower() in WORDS", "w in WORDS")),
                    en=_CALL + ": \"The\" with a capital is not in `WORDS`. Look the words up in lower case: `w.lower()`.",
                    nl=_CALL_NL + ": \"The\" met een hoofdletter staat niet in `WORDS`. Zoek de woorden op in kleine letters: `w.lower()`."),
            Mistake("ends-words", wrong=_f(ends=_F_ENDS.replace("ends = ends / len(lines)", "ends = ends / len(words)")),
                    en=_CALL + ": you divide the lines that end a sentence by the number of words. Feature 1 is a share of the lines: divide by `len(lines)`.",
                    nl=_CALL_NL + ": je deelt de regels die een zin afsluiten door het aantal woorden. Feature 1 is een aandeel van de regels: deel door `len(lines)`."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the five `...`: the lines, and then the four numbers, each a share between 0 and 1.",
                      "nl": "Vervang de vijf `...`: de regels, en dan de vier getallen, elk een aandeel tussen 0 en 1."},
            "wrong": {"en": _CALL + ". The lines that are not empty, and four shares: lines that end a sentence, words that are one of `WORDS`, characters that are letters or spaces, and words per line divided by 20, at most 1?",
                      "nl": _CALL_NL + ". De regels die niet leeg zijn, en vier aandelen: regels die een zin afsluiten, woorden die een van `WORDS` zijn, tekens die letters of spaties zijn, en woorden per regel gedeeld door 20, hoogstens 1?"},
            "almost": {"en": "Right for this example. But " + _CALL[0].lower() + _CALL[1:] + ". Does your function use `page`, and keep every number between 0 and 1?",
                       "nl": "Goed voor dit voorbeeld. Maar " + _CALL_NL[0].lower() + _CALL_NL[1:] + ". Gebruikt je functie `page`, en blijft elk getal tussen 0 en 1?"},
            "correct": {"en": "Your four numbers describe a page the way a classifier can read it: how much of it is sentences, ordinary words, letters, and how long its lines are.",
                        "nl": "Je vier getallen beschrijven een pagina zoals een classifier hem kan lezen: hoeveel ervan zinnen, gewone woorden en letters zijn, en hoe lang zijn regels zijn."},
        },
    ),
}

# The demo uses the learner's fit of ml-train-logistic and its gradients; where one is missing or broken, the course's
# version stands in (template 1.3).
BUILDS_ON = (("nn", "logistic_gradients", "ml-train-logistic", "log-loss-gradients"), ("nn", "logistic", "ml-train-logistic", "fit-logistic"))
projects.builds_on(*BUILDS_ON)


def _tools() -> dict | str:
    run = run_code("from nn.logistic import fit")
    if run.error is not None:
        return "fit"
    return run.names


def pipeline(clean_lines, keep_page, features) -> dict:
    """Run the learner's functions on all pages: clean every page, keep the pages that pass, and train the learner's
    logistic regression to tell N_TRAIN stories (1) from N_TRAIN kept pages (0); score every kept page. Gives back the
    counts, the weights, every score, and for every page the place of its score among the kept pages (or None)."""
    tools = _tools()
    if isinstance(tools, str):
        return {"error": "fit"}
    all_pages = pages()
    cleaned = [clean_lines(p) for p in all_pages]
    keep = [bool(keep_page(c)) for c in cleaned]
    kept = [c for c, k in zip(cleaned, keep) if k]
    X = np.array([features(s) for s in stories()[:N_TRAIN]] + [features(c) for c in kept[:N_TRAIN]], dtype=float)
    y = np.array([1] * min(N_TRAIN, len(stories())) + [0] * min(N_TRAIN, len(kept)))
    w, b = tools["fit"](X, y, LR, STEPS)
    w = np.asarray(w, dtype=float)
    scores = 1 / (1 + np.exp(-(np.array([features(c) for c in kept], dtype=float) @ w + float(b))))
    index = [i for i, k in enumerate(keep) if k]
    return {"pages": len(all_pages), "chars": sum(len(p) for p in all_pages), "chars_lines": sum(len(c) for c in cleaned),
            "kept": len(kept), "chars_kept": sum(len(c) for c in kept), "w": w.tolist(), "b": float(b),
            "scores": scores.tolist(), "index": index, "stand_ins": [f"{p}/{m}.py" for p, m, _, _ in projects.stand_ins()]}


def score_of(result: dict, page: int) -> float | None:
    """The score the classifier of a run gave page `page` of the download, or None when the rules threw it out."""
    try:
        return result["scores"][result["index"].index(page)]
    except ValueError:
        return None


def steps_picture(rows: list, labels: dict) -> str:
    """Bars of the millions of characters after every step, on one scale, ink only: a measurement of the download.
    `rows`: (label, millions)."""
    lang = labels.get("lang", "en")
    W, BAR, ROW, TOP = 330, 14, 40, 30
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    y = TOP
    top = max(v for _, v in rows)
    for name, value in rows:
        w = max(2.0, (W - 50) * value / top)
        p.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(name)}</text>'
                 f'<rect x="0" y="{y + 17}" width="{w:.1f}" height="{BAR}" rx="2" style="fill:var(--ea-ink)"/>'
                 f'<text x="{w + 6:.1f}" y="{y + 28.5}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">{number(value, 1, lang)}</text>')
        y += ROW
    p.append(f'<text x="0" y="{y + 10}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["note"])}</text>')
    H = y + 18
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


def histogram_picture(scores: list, labels: dict) -> str:
    """The scores of the kept pages in 20 bins from 0 to 1, as bars of the prediction role; a line at 0.5."""
    W, H, TOP, BOTTOM = 330, 190, 30, 40
    counts = np.histogram(scores, bins=20, range=(0, 1))[0]
    top = max(int(counts.max()), 1)
    x0, x1 = 6, W - 6
    bw = (x1 - x0) / 20
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    base = H - BOTTOM
    for k, c in enumerate(counts):
        h = (base - TOP) * c / top
        p.append(f'<rect x="{x0 + k * bw + 1:.1f}" y="{base - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" style="fill:var(--ea-pred)"/>')
    mid = x0 + 10 * bw
    p.append(f'<path d="M{mid:.1f} {TOP - 4} V{base}" style="stroke:var(--ea-ink-2);stroke-width:1.2;stroke-dasharray:4 3"/>'
             f'<path d="M{x0} {base} H{x1}" style="stroke:var(--ea-ink-3);stroke-width:1"/>')
    for v in (0, 0.5, 1):
        x = x0 + v * (x1 - x0)
        label = number(v, 1, labels.get("lang", "en")) if v == 0.5 else str(v)
        p.append(f'<text x="{x:.1f}" y="{base + 14}" style="font-size:11.5px;fill:var(--ea-ink-3);text-anchor:middle">{label}</text>')
    p.append(f'<text x="{W / 2}" y="{base + 32}" style="font-size:12.5px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["x"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["title"] + ". " + labels["alt"])}</title>{"".join(p)}</svg></figure>')


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("clean-lines", _L_START, "empty", None),
    ("clean-lines", _l(), "correct", None),
    ("clean-lines", _l(test="        words = line.split()\n        if line[-1:] in [\".\", \"!\", \"?\", '\"'] and len(words) > 4 and line.lower().find(\"javascript\") == -1:\n            kept.append(line)\n"), "correct", None),
    ("clean-lines", _l(strip=""), "wrong", "no-strip"),
    ("clean-lines", _l(test=_L_TEST.replace("\"javascript\" not in line.lower()", "\"javascript\" not in line")), "wrong", "case"),
    ("clean-lines", _l(test=_L_TEST.replace("len(line.split()) >= 5", "len(line) >= 5")), "wrong", "chars"),
    ("clean-lines", _l(test=_L_TEST.replace("line.endswith(END) and len(line.split()) >= 5", "(line.endswith(END) or len(line.split()) >= 5)")), "wrong", "or"),
    ("clean-lines", _l(tail="    return kept\n"), "wrong", "list"),
    ("clean-lines", _l(tail="    return \" \".join(kept)\n"), "wrong", "space-join"),
    ("clean-lines", _l(test=_L_TEST.replace("line.endswith(END)", "line.endswith(list(END))")), "error", "endswith-list"),
    ("keep-page", _P_START, "empty", None),
    ("keep-page", _p(), "correct", None),
    ("keep-page", _p(stop="    if len(STOP_WORDS & {w.lower() for w in words}) < 2:\n        return False\n",
                     alpha="    if sum(1 for w in words if any(c.isalpha() for c in w)) < 0.8 * len(words):\n        return False\n"), "correct", None),
    ("keep-page", _p(stop="    seen = 0\n    for w in words:\n        if w.lower() in STOP_WORDS:\n            seen = seen + 1\n    if seen < 2:\n        return False\n"), "wrong", "count-stop"),
    ("keep-page", _p(stop=_P_STOP.replace("w.lower() in STOP_WORDS", "w in STOP_WORDS").replace("seen.add(w.lower())", "seen.add(w)")), "wrong", "no-lower"),
    ("keep-page", _p(length="    if len(words) <= 50 or len(words) > 100000:\n        return False\n"), "wrong", "bounds"),
    ("keep-page", _p(mean=""), "wrong", "no-mean"),
    ("keep-page", _p(alpha=""), "wrong", "no-alpha"),
    ("keep-page", _p(length="    if len(words) < 50 or len(words) > 1000000:\n        return False\n"), "wrong", "no-upper"),
    ("keep-page", _p(length="    if len(words) < 50 or len(words) >= 100000:\n        return False\n"), "wrong", "upper-included"),
    ("features", _F_START, "empty", None),
    ("features", _f(), "correct", None),
    ("features", _f(lines=_F_LINES.replace("        if line.strip() != \"\":\n            lines.append(line.strip())\n", "        lines.append(line.strip())\n")), "wrong", "empty-lines"),
    ("features", _f(letters=_F_LETTERS.replace("c.isalpha() or c.isspace()", "c.isalpha()")), "wrong", "no-space"),
    ("features", _f(per="    per_line = len(words) / len(lines) / 20\n"), "wrong", "no-cap"),
    ("features", _f(stop=_F_STOP.replace("w.lower() in WORDS", "w in WORDS")), "wrong", "stop-case"),
    ("features", _f(ends=_F_ENDS.replace("ends = ends / len(lines)", "ends = ends / len(words)")), "wrong", "ends-words"),
]
