# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-spelling: find the known words one edit away from a typo (near_words), pick the most
common of a list of words (most_common), and put the two together in a spelling corrector (correct), which looks one
edit away first and two edits away only when that finds nothing; and sort typos by what the corrector does with them.

The corrector follows the one in Peter Norvig's essay *How to Write a Spelling Corrector*, written again in the
course's own code and words. The word counts are the course's own: the five English books of Project Gutenberg in
data/gutenberg-books.json.gz (made by data/tools/make_gutenberg.py), the three Dutch books of that file, with their
old spelling, and the Dutch sentences of Tatoeba in data/tatoeba-nl.tsv.gz (CC BY 2.0 FR, Tatoeba and its
contributors; made by data/tools/make_tatoeba_nl.py); everything_ai.books cuts them into words, and the course counts
them in small letters. The typos, the sentences, the cases of the checks and the made-up typos of the guess are the
course's own, except *acress*, from Jurafsky and Martin's appendix on spelling correction, and *speling*, *korrectud*,
*bycycle* and *peotry*, from Norvig's essay *How to Write a Spelling Corrector*."""
import contextlib
import copy
import html
import io
import random

from everything_ai import books, yours
from everything_ai.check import FunctionBox, Mistake, Sort, Timeout, kept, with_time_limit

# --- the word counts ---------------------------------------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyz"

CORPORA = ("english", "tatoeba", "dutch-books")   # the corpora of the demo, in the order of its menu
LESSON_CORPUS = {"en": "english", "nl": "tatoeba"}   # the corpus of each language's pictures and guess


def corpus_words(key: str) -> list:
    """The words of a corpus of the course, as a new list: "english" is the five English books one after another,
    "dutch-books" the three Dutch books, "tatoeba" the Dutch sentences of Tatoeba."""
    if key == "english":
        return [w for k in books.keys() if books.book(k)["lang"] == "en" for w in books.book_words(k)]
    if key == "dutch-books":
        return [w for k in books.keys() if books.book(k)["lang"] == "nl" for w in books.book_words(k)]
    if key == "tatoeba":
        return books.dutch_sentence_words()
    raise KeyError(key)


def _count(key: str) -> dict:
    def make():
        counts = {}
        for w in corpus_words(key):
            w = w.lower()
            if w in counts:
                counts[w] = counts[w] + 1
            else:
                counts[w] = 1
        return counts
    return kept("nlp-spelling:counts:" + key, make)


def word_counts(key: str) -> dict:
    """How often each word occurs in a corpus of the course, in small letters, as a new dict: {"the": 30668, ...}."""
    return dict(_count(key))


def corpus_size(key: str) -> tuple:
    """(words in the corpus, different words) of a corpus."""
    counts = _count(key)
    return sum(counts.values()), len(counts)


class Counts(dict):
    """The word counts that a case of a check gives the learner's function: a copy of its own, which knows which
    corpus it comes from, so that far_words can keep what it found for the next case."""

    def __init__(self, data, corpus: str):
        super().__init__(data)
        self.corpus = corpus


class Corpus:
    """The word counts of a corpus as a value in a case of a check: shown under the name the lesson gives them, and
    given to the learner's function as a new copy each time."""

    def __init__(self, name: str, key: str):
        self.name, self.key = name, key

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return Counts(_count(self.key), self.key)


ENGLISH = Corpus("english", "english")
DUTCH = Corpus("dutch", "tatoeba")


# --- the course's solutions --------------------------------------------------------------------------------------------

def near_words(word, counts, letters):
    """The course's solution of Exercise 1: every string one edit away from `word` (a deletion, a swap of two
    neighboring letters, a substitution or an insertion, with the letters of `letters`), and of those only the words
    that are in `counts`, sorted, each once. A substitution by the same letter gives `word` itself: in the cases of the
    check, `word` is never in `counts`."""
    found = []
    for i in range(len(word) + 1):
        start = word[:i]
        rest = word[i:]
        if len(rest) > 0:
            found.append(start + rest[1:])
        if len(rest) > 1:
            found.append(start + rest[1] + rest[0] + rest[2:])
        for c in letters:
            found.append(start + c + rest)
            if len(rest) > 0:
                found.append(start + c + rest[1:])
    return sorted(set([w for w in found if w in counts]))


def most_common(words, counts):
    """The course's solution of Exercise 2: the word of `words` with the highest count, the first of them when two are
    equally common; None for an empty list. Every word is in `counts`."""
    best = None
    best_count = 0
    for w in words:
        if counts[w] > best_count:
            best = w
            best_count = counts[w]
    return best


def all_edits(word, letters: str = LETTERS) -> list:
    """Every string one edit away from `word`, in the order near_words makes them, also the ones that are not words,
    and with repeats: for a word of n letters, n deletions, n - 1 swaps, 26n substitutions and 26(n + 1)
    insertions, 54n + 25 in all, as Norvig counts them."""
    found = []
    for i in range(len(word) + 1):
        start, rest = word[:i], word[i:]
        if len(rest) > 0:
            found.append(start + rest[1:])
        if len(rest) > 1:
            found.append(start + rest[1] + rest[0] + rest[2:])
        for c in letters:
            found.append(start + c + rest)
            if len(rest) > 0:
                found.append(start + c + rest[1:])
    return found


_FAR = {}   # what far_words found, per corpus of a check: (corpus, number of words, word) -> the words


def far_words(word, counts):
    """The course's function for two edits: every word in `counts` that two edits reach from `word`, sorted, each
    once. Two edits also reach every word one edit away, and `word` itself. Within a check, what it finds for a corpus
    of the check is kept, since the cases ask for the same words again."""
    corpus = getattr(counts, "corpus", None)
    key = (corpus, len(counts), word)
    if corpus is not None and key in _FAR:
        return list(_FAR[key])
    found = set()
    for one in set(all_edits(word)):
        for two in all_edits(one):
            if two in counts:
                found.add(two)
    out = sorted(found)
    if corpus is not None:
        _FAR[key] = tuple(out)
    return out


def correct(word, counts):
    """The course's solution of Exercise 3: the word itself when it is in `counts`; else the most common word one edit
    away; else the most common word two edits away; else the word itself."""
    if word in counts:
        return word
    near = near_words(word, counts, LETTERS)
    if len(near) > 0:
        return most_common(near, counts)
    far = far_words(word, counts)
    if len(far) > 0:
        return most_common(far, counts)
    return word


# --- made-up typos ---------------------------------------------------------------------------------------------------

KINDS = ("delete", "insert", "substitute", "swap")
TYPOS = 500
SEED = 26


def made_up_typos(key: str, n: int = TYPOS, seed: int = SEED) -> list:
    """n typos as (meant, typo, kind): each time a word of at least four letters a to z, at a random place in the
    running text of the corpus, so that common words come up more often, changed by one edit of a random kind at a
    random place: a letter deleted, a letter inserted, a letter replaced by another, or two neighboring letters swapped
    (never two equal ones, which would change nothing)."""
    def make():
        rng = random.Random(seed)
        words = [w.lower() for w in corpus_words(key)]
        words = [w for w in words if len(w) >= 4 and all(c in LETTERS for c in w)]
        out = []
        while len(out) < n:
            w = words[rng.randrange(len(words))]
            kind = KINDS[rng.randrange(len(KINDS))]
            i = rng.randrange(len(w))
            if kind == "delete":
                typo = w[:i] + w[i + 1:]
            elif kind == "insert":
                typo = w[:i] + LETTERS[rng.randrange(26)] + w[i:]
            elif kind == "substitute":
                others = [c for c in LETTERS if c != w[i]]
                typo = w[:i] + others[rng.randrange(25)] + w[i + 1:]
            else:
                i = min(i, len(w) - 2)
                if w[i] == w[i + 1]:
                    continue
                typo = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            out.append((w, typo, kind))
        return tuple(out)
    return list(kept(f"nlp-spelling:typos:{key}:{n}:{seed}", make))


def outcome(meant: str, typo: str, got, counts: dict) -> str:
    """What a corrector did with one made-up typo: "fixed" (it gave the word that was meant), "word" (the typo is a
    word itself, and the corrector left it alone), or "other" (anything else)."""
    if got == meant:
        return "fixed"
    if typo in counts and got == typo:
        return "word"
    return "other"


def score(fn, key: str) -> dict:
    """How the corrector `fn` does on the made-up typos of a corpus: {"fixed": n, "other": n, "word": n}, with
    "examples", the first few typos of each kind of outcome as (meant, typo, got); and on the same words typed right:
    "kept", how many it leaves alone, and "changed", how many it changes, with "changed_examples" as (word, got)."""
    counts = word_counts(key)
    out = {"fixed": 0, "other": 0, "word": 0, "examples": {"fixed": [], "other": [], "word": []},
           "kept": 0, "changed": 0, "changed_examples": []}
    for meant, typo, _ in made_up_typos(key):
        got = fn(typo, counts)
        o = outcome(meant, typo, got, counts)
        out[o] += 1
        if len(out["examples"][o]) < 5:
            out["examples"][o].append((meant, typo, got))
        again = fn(meant, counts)
        if again == meant:
            out["kept"] += 1
        else:
            out["changed"] += 1
            if len(out["changed_examples"]) < 5:
                out["changed_examples"].append((meant, again))
    return out


def course_score(key: str) -> dict:
    """The score of the course's corrector on the made-up typos of a corpus."""
    return kept("nlp-spelling:score:" + key, lambda: score(correct, key))


# --- the look of the pictures ----------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:baseline;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.82rem;line-height:1.5;white-space:pre")
_CHIP_BLUE = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_FAINT = _CHIP + ";color:var(--ea-ink-3)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle;margin-left:0.3rem"


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics, as in the lesson's text."""
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_UI}">{marked}</div>'


def _label(text: str) -> str:
    parts = html.escape(text).split("*")
    marked = "".join(f"<i>{p}</i>" if k % 2 else p for k, p in enumerate(parts))
    return f'<div style="{_LABEL}">{marked}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,683 in English, 1.683 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _pct(x: float, lang: str) -> str:
    """A percentage with one decimal: 50.0% in English, 50,0% in Dutch."""
    text = f"{x:.1f}%"
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(n: int, most: int, color: str = "var(--ea-data)", width: float = 8.0) -> str:
    """A bar for a count: as long as `width` for the largest count `most`, never shorter than a sliver."""
    w = 0 if not most or not n else max(0.15, width * n / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped, a cell as (html, rows it spans) where it spans more
    than one, and a row leaves out the cells that a cell above it spans; numbers are the columns set to the right,
    counted with the spanned cells; no head when `head` is empty."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = ""
    below = {}  # column: how many rows below a spanning cell still covers
    for row in rows_:
        cells, j = "", 0
        for c in row:
            while below.get(j):
                below[j] -= 1
                j += 1
            span = ""
            if isinstance(c, tuple):
                c, k = c
                span = f' rowspan="{k}"'
                below[j] = k - 1
            cells += f'<td{span} style="{_CELL}{_NUMBER if j in numbers else ""}">{c}</td>'
            j += 1
        for col in below:
            if col >= j and below[col]:
                below[col] -= 1
        body += f"<tr>{cells}</tr>"
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _count_cells(x: str, counts: dict, top: int, lang: str, style: str = _CHIP_DATA) -> list:
    """Three cells for a word and its count: the word, the count, and a bar for the count."""
    return [_chip(x, style), _num(counts[x], lang), _bar(counts[x], top)]


def _shown(v) -> str:
    """A value of the learner's as a picture shows it: a string as it is, anything else the way Python writes it."""
    return v if isinstance(v, str) else repr(v)


# --- the idea's picture: from a typo to a word -------------------------------------------------------------------------

IDEA_TYPO = {"en": "thier", "nl": "hius"}

_IDEA_WORDS = {
    "en": {"title": "From thier to their",
           "all": "1. Every string one edit away from *{t}*: {n}, of which {k} are different",
           "known": "2. The ones that are words in the books, with how often each occurs: {k}",
           "pick": "3. The most common of them",
           "more": "… and {k} more",
           "note": "Grey: strings and words with their counts in the five English books of the course. Blue: the word "
                   "the corrector picks.",
           "alt": "{title}. {all}: {sample}. {known}: {words}. {pick}: {best}. {note}"},
    "nl": {"title": "Van hius naar huis",
           "all": "1. Elke string op één bewerking van *{t}*: {n}, waarvan {k} verschillend",
           "known": "2. De strings die een woord zijn in de zinnen van Tatoeba, met hoe vaak elk voorkomt: {k}",
           "pick": "3. Het woord dat daarvan het vaakst voorkomt",
           "more": "… en nog {k}",
           "note": "Grijs: strings, en woorden met hoe vaak elk voorkomt in de Nederlandse zinnen van Tatoeba. Blauw: het "
                   "woord dat de corrector kiest.",
           "alt": "{title}. {all}: {sample}. {known}: {words}. {pick}: {best}. {note}"},
}
_IDEA_SHOWN = 14   # strings of step 1 that the picture shows


def idea_picture(lang: str) -> str:
    w = _IDEA_WORDS[lang]
    key = LESSON_CORPUS[lang]
    counts = _count(key)
    typo = IDEA_TYPO[lang]
    every = all_edits(typo)
    different = list(dict.fromkeys(every))
    sample = different[:_IDEA_SHOWN]
    near = near_words(typo, counts, LETTERS)
    best = most_common(near, counts)
    top = max(counts[x] for x in near)
    body = _label(w["all"].format(t=typo, n=_num(len(every), lang), k=_num(len(different), lang)))
    body += "<div>" + "".join(_chip(s, _CHIP_FAINT) for s in sample) + _chip(
        w["more"].format(k=_num(len(different) - len(sample), lang)), _CHIP_FAINT) + "</div>"
    body += _label(w["known"].format(k=len(near)))
    rows_ = [_count_cells(x, counts, top, lang) for x in sorted(near, key=lambda x: -counts[x])]
    body += _table(rows_, [], numbers=(1,))
    body += _label(w["pick"]) + f"<div>{_chip(best, _CHIP_BLUE)}</div>" + _note(w["note"])
    words = ", ".join(f"{x} {_num(counts[x], lang)}" for x in near)
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], all=w["all"].format(t=typo, n=_num(len(every), lang), k=_num(len(different), lang)).replace("*", ""),
        sample=" ".join(sample), known=w["known"].format(k=len(near)), words=words, pick=w["pick"], best=best,
        note=w["note"]))


# --- running the learner's function ----------------------------------------------------------------------------------

def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
PICTURE_LIMIT = 10.0   # seconds the learner's function gets for everything a picture asks of it
SCORE_LIMIT = 4.0      # seconds the learner's corrector gets for the made-up typos in the picture of exercise 3


def _call(make, seconds: float = PICTURE_LIMIT):
    """make() within a time limit, with what it prints thrown away; _STOPPED when it stops with an error, and the
    Timeout passes on."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, seconds)
    except Timeout:
        raise
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _run_or_say(make, w: dict, ok) -> tuple:
    """(what make() gives, None), or (None, a figure that says why there is no picture): an error, too slow, or a
    result that `ok` does not accept."""
    try:
        got = _call(make)
    except Timeout:
        text = w["slow"].format(s=PICTURE_LIMIT)
        return None, yours.figure(w["title"], _note(text), alt=text)
    if got is _STOPPED:
        return None, yours.figure(w["title"], _note(w["stopped"]), alt=w["stopped"])
    if not ok(got):
        return None, yours.figure(w["title"], _note(w["odd"]), alt=w["odd"])
    return got, None


def as_words(v):
    """What a function of the learner gives for near_words, the way the check compares it: a list, set or tuple of
    strings as a sorted list with each string once; anything else as it is."""
    if isinstance(v, (list, tuple, set, frozenset)) and all(isinstance(x, str) for x in v):
        return sorted(set(v))
    return v


_STOPPED_TEXT = {"en": "Your function stopped with an error on one of the typos, so the picture stays empty.",
                 "nl": "Je functie stopte met een fout op een van de tikfouten, dus de afbeelding blijft leeg."}
_SLOW_TEXT = {"en": "Your function took more than {s:g} seconds for all typos, so the picture stays empty.",
              "nl": "Je functie deed er langer dan {s:g} seconden over voor alle tikfouten, dus de afbeelding blijft leeg."}


# --- the picture under exercise 1: your words one edit away ------------------------------------------------------------

NEAR_TYPOS = {"en": ["thier", "acress", "whal", "korrectud"], "nl": ["hius", "boec", "scool", "keuke"]}
_NEAR_SHOWN = 12   # words of one row that the picture shows

_NEAR_WORDS = {
    "en": {"title": "Your near_words on four typos",
           "head": ["typo", "your words one edit away"],
           "none": "none",
           "more": "… and {k} more",
           "note": "Blue: a word your function finds, as the course's does. Purple: a string your function gives that "
                   "the course's does not, such as a string that is not a word in the books. Dashed: a word one edit away that "
                   "your function misses. With the counts of the five English books of the course.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a list of strings for every typo, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw near_words op vier tikfouten",
           "head": ["tikfout", "jouw woorden op één bewerking"],
           "none": "geen",
           "more": "… en nog {k}",
           "note": "Blauw: een woord dat je functie vindt, net als de functie van de course. Paars: een string die je "
                   "functie geeft en de functie van de course niet, zoals een string die in de zinnen geen woord is. Gestippeld: een "
                   "woord op één bewerking dat je functie mist. Met de aantallen van de Nederlandse zinnen van "
                   "Tatoeba.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elke tikfout een lijst met strings terug, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def _strings(v) -> bool:
    return isinstance(v, (list, tuple, set, frozenset)) and all(isinstance(x, str) for x in v)


def near_picture(seen) -> str:
    lang = seen.lang
    w = _NEAR_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    key = LESSON_CORPUS[lang]
    counts = _count(key)
    typos = NEAR_TYPOS[lang]
    got, said = _run_or_say(lambda: [fn(t, Counts(counts, key), LETTERS) for t in typos], w,
                            lambda g: all(_strings(v) for v in g))
    if said:
        return said
    rows_, alts = [], []
    for t, mine in zip(typos, got):
        mine, want = sorted(set(mine)), near_words(t, counts, LETTERS)
        right = [x for x in mine if x in want]
        extra = [x for x in mine if x not in want]
        missing = [x for x in want if x not in mine]
        chips = [_chip(x, _CHIP_BLUE) for x in right] + [_chip(x, _CHIP_PURPLE) for x in extra]
        if len(chips) > _NEAR_SHOWN:
            chips = chips[:_NEAR_SHOWN] + [_chip(w["more"].format(k=_num(len(chips) - _NEAR_SHOWN, lang)), _CHIP_FAINT)]
        chips += [_chip(x, _CHIP_MISSING) for x in missing]
        if not chips:
            chips = [_chip(w["none"], _CHIP_FAINT)]
        rows_.append([_chip(t), "".join(chips)])
        alt = f"{t}: " + (", ".join(mine[:_NEAR_SHOWN]) or w["none"])
        if len(mine) > _NEAR_SHOWN:
            alt += " " + w["more"].format(k=_num(len(mine) - _NEAR_SHOWN, lang))
        if missing:
            alt += " (" + ", ".join(missing) + ")"
        alts.append(alt)
    body = _table(rows_, w["head"]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 2: your pick ---------------------------------------------------------------------------

PICK_TYPOS = {"en": ["thier", "teh", "acress", "whal"], "nl": ["hius", "boec", "scool", "wear"]}
_PICK_SHOWN = 4   # words of each typo that the picture shows, the most common first

_PICK_WORDS = {
    "en": {"title": "Your most_common on the words one edit away",
           "head": ["typo", "words one edit away", "count", "", "your pick"],
           "more": "… and {k} more",
           "course": "the course: {w}",
           "none": "nothing",
           "note": "Grey: the words and how often each occurs in the five English books of the course. Blue: the word "
                   "your function picks, the most common one. Purple: a pick that differs from the course's.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a word for every list, so the picture stays empty.",
           "alt": "{title}. {rows}. {note}"},
    "nl": {"title": "Jouw most_common op de woorden op één bewerking",
           "head": ["tikfout", "woorden op één bewerking", "aantal", "", "jouw keuze"],
           "more": "… en nog {k}",
           "course": "de course: {w}",
           "none": "niets",
           "note": "Grijs: de woorden en hoe vaak elk voorkomt in de Nederlandse zinnen van Tatoeba. Blauw: het woord dat "
                   "je functie kiest, het woord dat het vaakst voorkomt. Paars: een keuze die anders is dan die van de "
                   "course.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elke lijst een woord terug, dus de afbeelding blijft leeg.",
           "alt": "{title}. {rows}. {note}"},
}


def pick_picture(seen) -> str:
    lang = seen.lang
    w = _PICK_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    key = LESSON_CORPUS[lang]
    counts = _count(key)
    lists = [near_words(t, counts, LETTERS) for t in PICK_TYPOS[lang]]
    got, said = _run_or_say(lambda: [fn(list(x), Counts(counts, key)) for x in lists], w,
                            lambda g: all(isinstance(v, str) for v in g))
    if said:
        return said
    rows_, alts = [], []
    for t, words, mine in zip(PICK_TYPOS[lang], lists, got):
        want = most_common(words, counts)
        top = max(counts[x] for x in words)
        shown = sorted(words, key=lambda x: (-counts[x], x))[:_PICK_SHOWN]
        pick = _chip(mine, _CHIP_BLUE if mine == want else _CHIP_PURPLE)
        if mine != want:
            pick += "<br>" + _chip(w["course"].format(w=want))
        more = len(words) - len(shown)
        k = len(shown) + (1 if more else 0)
        for j, x in enumerate(shown):
            row = _count_cells(x, counts, top, lang, _CHIP_PURPLE if x == mine != want else _CHIP_DATA)
            rows_.append(([(_chip(t), k)] if j == 0 else []) + row + ([(pick, k)] if j == 0 else []))
        if more:
            rows_.append([_chip(w["more"].format(k=more), _CHIP_FAINT), "", ""])
        alts.append(f"{t}: " + ", ".join(f"{x} {_num(counts[x], lang)}" for x in words) + f" → {mine}")
    body = _table(rows_, w["head"], numbers=(2,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts), note=w["note"]))


# --- the picture under exercise 3: your corrector on a sentence, three hard cases and the made-up typos ---------------

# a sentence with typos, and the sentence that was meant, in small letters: the corrector works on small letters
SENTENCE = {
    "en": ("alice was begining to get very tird of siting on the bnak and of haveing nothing to do",
           "alice was beginning to get very tired of sitting on the bank and of having nothing to do"),
    "nl": ("afgelopen wekend ben ik met mijn nieuwe feits naar het stradn gegan maar het reegnde de hele dag",
           "afgelopen weekend ben ik met mijn nieuwe fiets naar het strand gegaan maar het regende de hele dag"),
}
# (typo, what was meant): a typo that is a word, one two edits from its word, and one whose word the corpus lacks
HARD = {"en": [("than", "then"), ("korrectud", "corrected"), ("bycycle", "bicycle")],
        "nl": [("kat", "kast"), ("restorant", "restaurant"), ("fietsenstaling", "fietsenstalling")]}

_CORRECT_WORDS = {
    "en": {"title": "Your correct at work",
           "sentence": "A sentence with typos, after the opening of *Alice's Adventures in Wonderland*",
           "typed": "typed", "yours": "your corrector", "meant": "meant",
           "hard": "Three hard cases",
           "head": ["typed", "what it is", "meant", "your corrector", "the course's"],
           "why": ["a typo that is a word", "two edits from its word", "its word is not in the books"],
           "made": "The {n} made-up typos of the guess",
           "fixed": "put right", "other": "another word", "word": "left alone: the typo is a word",
           "right": "The same {n} words typed right",
           "kept": "left alone", "changed": "changed",
           "slow_score": "Your corrector took more than {s:g} seconds for the made-up typos, so this part stays out.",
           "note": "Framed: the typos of the typed sentence. Blue: a typo your corrector puts right, and words typed "
                   "right that it leaves alone. Purple: a word that differs from the one that was meant, in the "
                   "sentence; from the course's corrector, in the hard cases; and the made-up typos that become another "
                   "word, or words typed right that it changes. Grey: typos that are words, which the corrector leaves "
                   "alone, and in the hard cases, the same word as the course's corrector. With the counts of the five "
                   "English books of the course.",
           "stopped": _STOPPED_TEXT["en"], "slow": _SLOW_TEXT["en"],
           "odd": "Your function does not give back a string for every word, so the picture stays empty.",
           "alt": "{title}. {sentence}: {typed} → {yours}. {hard}: {rows}. {score} {note}"},
    "nl": {"title": "Jouw correct aan het werk",
           "sentence": "Een zin met tikfouten",
           "typed": "getypt", "yours": "jouw corrector", "meant": "bedoeld",
           "hard": "Drie moeilijke gevallen",
           "head": ["getypt", "wat het is", "bedoeld", "jouw corrector", "corrector van de course"],
           "why": ["een tikfout die een woord is", "op twee bewerkingen van het woord", "het woord staat niet in de zinnen"],
           "made": "De {n} verzonnen tikfouten van de gok",
           "fixed": "verbeterd", "other": "een ander woord", "word": "blijft staan: de tikfout is een woord",
           "right": "Dezelfde {n} woorden, goed getypt",
           "kept": "blijft staan", "changed": "veranderd",
           "slow_score": "Je corrector deed er langer dan {s:g} seconden over voor de verzonnen tikfouten, dus dit deel "
                         "blijft weg.",
           "note": "Omlijnd: de tikfouten in de getypte zin. Blauw: een tikfout die je corrector verbetert, en goed "
                   "getypte woorden die je corrector laat staan. Paars: in de zin een woord dat anders is dan het "
                   "bedoelde woord; in de moeilijke gevallen een woord dat anders is dan het woord van de corrector van "
                   "de course; en de verzonnen tikfouten die een ander woord worden, of goed getypte woorden die je "
                   "corrector verandert. Grijs: tikfouten die een woord zijn en blijven staan, en in de moeilijke "
                   "gevallen hetzelfde woord als bij de corrector van de course. Met de aantallen van de Nederlandse "
                   "zinnen van Tatoeba.",
           "stopped": _STOPPED_TEXT["nl"], "slow": _SLOW_TEXT["nl"],
           "odd": "Je functie geeft niet voor elk woord een string terug, dus de afbeelding blijft leeg.",
           "alt": "{title}. {sentence}: {typed} → {yours}. {hard}: {rows}. {score} {note}"},
}


_SCORE_COLORS = {"fixed": "var(--ea-pred)", "other": "var(--ea-loss)", "word": "var(--ea-data)",
                 "kept": "var(--ea-pred)", "changed": "var(--ea-loss)"}


def _score_rows(s: dict, keys: tuple, lang: str, w: dict) -> tuple:
    """A row for each outcome of a score, with its number, its share and a bar in its color; and its alt text."""
    total = sum(s[k] for k in keys)
    rows_ = [[html.escape(w[k]), _num(s[k], lang), _pct(100 * s[k] / total, lang), _bar(s[k], total, _SCORE_COLORS[k])]
             for k in keys]
    alt = "; ".join(f"{w[k]}: {_num(s[k], lang)}" for k in keys)
    return _table(rows_, [], numbers=(1, 2)), alt


def correct_picture(seen) -> str:
    lang = seen.lang
    w = _CORRECT_WORDS[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    key = LESSON_CORPUS[lang]
    counts = _count(key)
    typed, meant = (s.split(" ") for s in SENTENCE[lang])
    hard = HARD[lang]
    got, said = _run_or_say(lambda: ([fn(x, Counts(counts, key)) for x in typed],
                                     [fn(t, Counts(counts, key)) for t, _ in hard]), w, lambda g: True)
    if said:
        return said
    mine, mine_hard = got
    line_typed, line_mine = [], []
    for x, m, y in zip(typed, meant, mine):
        line_typed.append(_chip(x) if x != m else html.escape(x))
        if y == m:
            line_mine.append(_chip(y, _CHIP_BLUE) if y != x else html.escape(y))
        else:
            line_mine.append(_chip(_shown(y), _CHIP_PURPLE))
    label = f'{_CELL};white-space:nowrap;color:var(--ea-ink-2)'
    rows_s = "".join(f'<tr><td style="{label}">{html.escape(k)}</td><td style="{_CELL}">{v}</td></tr>'
                     for k, v in ((w["typed"], " ".join(line_typed)), (w["yours"], " ".join(line_mine)),
                                  (w["meant"], html.escape(" ".join(meant)))))
    body = (_label(w["sentence"]) + '<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;'
            f'margin:0.3rem 0 0.2rem">{rows_s}</table></div>')
    rows_h, alts_h = [], []
    for (t, m), y, why in zip(hard, mine_hard, w["why"]):
        want = correct(t, counts)
        rows_h.append([_chip(t), html.escape(why), html.escape(m),
                       _chip(_shown(y), _CHIP_FAINT if y == want else _CHIP_PURPLE), _chip(want, _CHIP_FAINT)])
        alts_h.append(f"{t} ({why}), {w['meant']} {m}: {_shown(y)}; {w['head'][4]}: {want}")
    body += _label(w["hard"]) + _table(rows_h, w["head"])
    n = len(made_up_typos(key))
    try:
        s = _call(lambda: score(fn, key), SCORE_LIMIT)
    except Timeout:
        s = None
    if s is None or s is _STOPPED:
        text = w["slow_score"].format(s=SCORE_LIMIT) if s is None else w["stopped"]
        body += _label(w["made"].format(n=n)) + _note(text)
        score_alt = text
    else:
        typos_, alt_typos = _score_rows(s, ("fixed", "other", "word"), lang, w)
        right_, alt_right = _score_rows(s, ("kept", "changed"), lang, w)
        body += _label(w["made"].format(n=n)) + typos_ + _label(w["right"].format(n=n)) + right_
        score_alt = (w["made"].format(n=n) + ": " + alt_typos + ". " + w["right"].format(n=n) + ": " + alt_right + ".")
    body += _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(
        title=w["title"], sentence=w["sentence"].replace("*", ""), typed=" ".join(typed),
        yours=" ".join(_shown(y) for y in mine), hard=w["hard"], rows="; ".join(alts_h), score=score_alt,
        note=w["note"]))


# --- the demo: correct a word of your own ------------------------------------------------------------------------------

DEMO_LONGEST = 15
_DEMO_SHOWN = 8   # words of each kind that the demo shows

_DEMO_WORDS = {
    "en": {"title": "The corrector on your word",
           "corpus": {"english": "the five English books", "tatoeba": "the Dutch sentences of Tatoeba",
                      "dutch-books": "the three Dutch books, in spelling from before 1934"},
           "known": "*{w}* is a word in {c}: {n} times. The corrector leaves it alone.",
           "one": "One edit away: {k} words", "two": "Two edits away: {k} words",
           "one_1": "One edit away: 1 word", "two_1": "Two edits away: 1 word",
           "none": "none",
           "more": "… and {k} more",
           "pick_one": "The corrector picks *{p}*: the most common word one edit away.",
           "pick_two": "Nothing is one edit away, so the corrector picks *{p}*: the most common word two edits away.",
           "pick_none": "No word in {c} lies within two edits, so the corrector leaves *{w}* alone.",
           "note": "Grey: words and how often each occurs in {c}. Blue: the word the corrector picks. The corrector "
                   "works on small letters, and tries only the letters a to z.",
           "empty": "Type a word.",
           "long": "Your word is longer than {k} letters; try a shorter one.",
           "alt": "{title}: {w}, with the counts of {c}. {lines} {note}"},
    "nl": {"title": "De corrector op jouw woord",
           "corpus": {"english": "de vijf Engelse boeken", "tatoeba": "de Nederlandse zinnen van Tatoeba",
                      "dutch-books": "de drie Nederlandse boeken, in spelling van vóór 1934"},
           "known": "*{w}* is een woord in {c}: {n} keer. De corrector laat het staan.",
           "one": "Op één bewerking: {k} woorden", "two": "Op twee bewerkingen: {k} woorden",
           "one_1": "Op één bewerking: 1 woord", "two_1": "Op twee bewerkingen: 1 woord",
           "none": "geen",
           "more": "… en nog {k}",
           "pick_one": "De corrector kiest *{p}*: het woord op één bewerking dat het vaakst voorkomt.",
           "pick_two": "Op één bewerking ligt niets, dus de corrector kiest *{p}*: het woord op twee bewerkingen dat het "
                       "vaakst voorkomt.",
           "pick_none": "Binnen twee bewerkingen ligt geen woord uit {c}, dus de corrector laat *{w}* staan.",
           "note": "Grijs: woorden en hoe vaak elk voorkomt in {c}. Blauw: het woord dat de corrector kiest. De "
                   "corrector werkt met kleine letters, en probeert alleen de letters a tot en met z.",
           "empty": "Typ een woord.",
           "long": "Je woord is langer dan {k} letters; probeer een korter woord.",
           "alt": "{title}: {w}, met de aantallen van {c}. {lines} {note}"},
}


def corpus_menu(lang: str) -> dict:
    """{label: key} of the corpora, for the menu of the demo."""
    return {_DEMO_WORDS[lang]["corpus"][k][0].upper() + _DEMO_WORDS[lang]["corpus"][k][1:]: k for k in CORPORA}


def _demo_far(word: str, key: str) -> list:
    return list(kept(f"nlp-spelling:demo-far:{key}:{word}", lambda: tuple(far_words(word, _count(key)))))


def _demo_list(words: list, counts: dict, pick, lang: str, w: dict) -> str:
    if not words:
        return _chip(w["none"], _CHIP_FAINT)
    shown = sorted(words, key=lambda x: (-counts[x], x))[:_DEMO_SHOWN]
    top = counts[shown[0]]
    rows_ = [_count_cells(x, counts, top, lang, _CHIP_BLUE if x == pick else _CHIP_DATA) for x in shown]
    more = len(words) - len(shown)
    out = _table(rows_, [], numbers=(1,))
    if more > 0:
        out += _note(w["more"].format(k=_num(more, lang)))
    return out


def demo_picture(word, key, lang: str) -> str:
    w = _DEMO_WORDS[lang]
    word = str(word or "").strip().lower()
    key = key if key in CORPORA else LESSON_CORPUS[lang]
    c = w["corpus"][key]
    if not word:
        return yours.figure(w["title"], _note(w["empty"]), alt=w["empty"])
    if len(word) > DEMO_LONGEST:
        msg = w["long"].format(k=DEMO_LONGEST)
        return yours.figure(w["title"], _note(msg), alt=msg)
    counts = _count(key)
    if word in counts:
        text = w["known"].format(w=word, c=c, n=_num(counts[word], lang))
        body = f"<div>{_chip(word, _CHIP_BLUE)}</div>" + _note(text) + _note(w["note"].format(c=c))
        return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], w=word, c=c, lines=text,
                                                                 note=w["note"].format(c=c)))
    near = near_words(word, counts, LETTERS)
    far = [x for x in _demo_far(word, key) if x not in near and x != word]
    pick = correct(word, counts)
    if near:
        verdict = w["pick_one"].format(p=pick)
    elif far:
        verdict = w["pick_two"].format(p=pick)
    else:
        verdict = w["pick_none"].format(w=word, c=c)
    one = w["one_1"] if len(near) == 1 else w["one"].format(k=_num(len(near), lang))
    two = w["two_1"] if len(far) == 1 else w["two"].format(k=_num(len(far), lang))
    body = (_label(one) + _demo_list(near, counts, pick, lang, w)
            + _label(two) + _demo_list(far, counts, pick if not near else None, lang, w)
            + _note(verdict) + _note(w["note"].format(c=c)))
    lines = (one + ": " + ", ".join(sorted(near, key=lambda x: -counts[x])[:_DEMO_SHOWN])
             + ". " + two + ": " + ", ".join(sorted(far, key=lambda x: -counts[x])[:_DEMO_SHOWN]) + ". "
             + verdict.replace("*", ""))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], w=word, c=c, lines=lines,
                                                             note=w["note"].format(c=c)))


# --- the exercises ---------------------------------------------------------------------------------------------------

REQUIRED = ["near-words", "most-common", "correct", "what-happens"]

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_DOTS = Mistake("dots-left", matches=lambda got, text: "..." in "\n".join(
                    line.split("#")[0] for line in text.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang het door je eigen code.")


def _value(v) -> str:
    """A short value the way the lessons write it: strings in double quotes, also inside a list or a dictionary."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(p) for p in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ", ".join(_value(p) for p in sorted(v, key=str)) + "}" if v else "set()"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _shown_words(v):
    """A list of words for a message: at most 12, then how many more, in each language."""
    if isinstance(v, list) and len(v) > 12:
        start = "[" + ", ".join(_value(p) for p in v[:12])
        return {"en": start + f", ... and {_num(len(v) - 12, 'en')} more]",
                "nl": start + f", ... en nog {_num(len(v) - 12, 'nl')}]"}
    return _value(v)


class Named:
    """A value of the check's own that a message shows as Python, with strings in double quotes."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return self.name

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.value, memo)


def _lit(v) -> Named:
    return Named(_value(v), v)


def _index_error(e) -> bool:
    return isinstance(e, IndexError)


def _key_error(e) -> bool:
    return isinstance(e, KeyError)


def _far_values(e) -> bool:
    return isinstance(e, TypeError) and "far_words" in str(e)


def _a_count(got, text) -> bool:
    return isinstance(got, int) and not isinstance(got, bool)


def _a_list(got, text) -> bool:
    return isinstance(got, (list, tuple, set))


def _too_many(got, text) -> bool:
    return isinstance(got, list) and len(got) > 60


# a small dictionary of counts, to try every kind of edit with only the letters "a" and "b": "ba" is a swap of "ab",
# "aba" an insertion at the end, "abc" and "cab" need a letter that is not in the letters
SMALL = {"a": 4, "b": 2, "ba": 3, "bb": 5, "aba": 1, "abc": 9, "cab": 2}

# exercise 1: near_words. The typo of the English lesson and of the Dutch one first, then typos with every kind of edit
# (acress: a deletion, a swap, substitutions and an insertion), a word at the end (whal: whale), a letter too many at
# the start, a typo with nothing one edit away, Dutch typos, and the small dictionary, with an empty word too.
NEAR_CASES = [
    (_lit("thier"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("hius"), DUTCH, Named("LETTERS", LETTERS)),
    (_lit("teh"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("acress"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("whal"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("xwhale"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("korrectud"), ENGLISH, Named("LETTERS", LETTERS)),
    (_lit("boec"), DUTCH, Named("LETTERS", LETTERS)),
    (_lit("keuke"), DUTCH, Named("LETTERS", LETTERS)),
    (_lit("scool"), DUTCH, Named("LETTERS", LETTERS)),
    (_lit("ab"), _lit(SMALL), _lit("ab")),
    (_lit(""), _lit(SMALL), _lit("ab")),
]
_N_START = "def near_words(word, counts, letters):\n    ...\n"
_N = ("def near_words(word, counts, letters):\n"
      "    found = []\n"
      "    for i in range(len(word) + 1):\n"
      "        start = word[:i]\n"
      "        rest = word[i:]\n"
      "        if len(rest) > 0:\n"
      "            found.append(start + rest[1:])\n"
      "        if len(rest) > 1:\n"
      "            found.append(start + rest[1] + rest[0] + rest[2:])\n"
      "        for c in letters:\n"
      "            found.append(start + c + rest)\n"
      "            if len(rest) > 0:\n"
      "                found.append(start + c + rest[1:])\n"
      "    return sorted(set([w for w in found if w in counts]))\n")
_N_NO_DELETE = _N.replace("        if len(rest) > 0:\n            found.append(start + rest[1:])\n", "")
_N_NO_SWAP = _N.replace("        if len(rest) > 1:\n            found.append(start + rest[1] + rest[0] + rest[2:])\n", "")
_N_NO_SUB = _N.replace("            if len(rest) > 0:\n                found.append(start + c + rest[1:])\n", "")
_N_NO_INSERT = _N.replace("            found.append(start + c + rest)\n", "")
_N_NO_END = _N.replace("range(len(word) + 1)", "range(len(word))")
_N_NO_START = _N.replace("range(len(word) + 1)", "range(1, len(word) + 1)")
_N_ALL = _N.replace("    return sorted(set([w for w in found if w in counts]))\n", "    return sorted(set(found))\n")
_N_RETURN_IN = _N.replace("    return sorted(set([w for w in found if w in counts]))\n",
                          "        return sorted(set([w for w in found if w in counts]))\n")
_N_SWAP_UNCHECKED = _N.replace("        if len(rest) > 1:\n            found.append(start + rest[1] + rest[0] + rest[2:])\n",
                               "        found.append(start + rest[1] + rest[0] + rest[2:])\n")

# exercise 2: most_common. Lists of words near typos, as exercise 1 finds them: the most common word last, in the
# middle or first, one word, no words, and the small dictionary. No list holds two words with the same highest count.
PICK_CASES = [
    (_lit(["their", "thief", "tier"]), ENGLISH),
    (_lit(["heus", "hijs", "hits", "huis"]), DUTCH),
    (_lit(["eh", "heh", "te", "tea", "ten", "th", "the"]), ENGLISH),
    (_lit(["access", "acres", "across", "actress", "caress"]), ENGLISH),
    (_lit(["hwal", "wal", "weal", "whale", "what", "wheal"]), ENGLISH),
    (_lit(["school", "cool", "schol"]), DUTCH),
    (_lit(["boe", "boef", "boeg", "boei", "boek", "boel", "boer"]), DUTCH),
    (_lit(["whale"]), ENGLISH),
    (_lit([]), ENGLISH),
    (_lit(["a", "aba", "b", "ba", "bb"]), _lit(SMALL)),
]
_M_START = "def most_common(words, counts):\n    ...\n"
_M = ("def most_common(words, counts):\n"
      "    best = None\n"
      "    best_count = 0\n"
      "    for w in words:\n"
      "        if counts[w] > best_count:\n"
      "            best = w\n"
      "            best_count = counts[w]\n"
      "    return best\n")
_M_NOT_UPDATED = _M.replace("            best_count = counts[w]\n", "")
_M_RETURN_IN = _M.replace("    return best\n", "        return best\n")
_M_COUNT = _M.replace("    return best\n", "    return best_count\n")
_M_LEAST = _M.replace("if counts[w] > best_count:", "if counts[w] < best_count:")
_M_FIRST = ("def most_common(words, counts):\n"
            "    best = words[0]\n"
            "    for w in words:\n"
            "        if counts[w] > counts[best]:\n"
            "            best = w\n"
            "    return best\n")

# exercise 3: correct. The typos of the lessons, a word that is a word already, a typo two edits from its word, one
# with no word within two edits, typos with words one and two edits away, and Dutch typos of each kind.
CORRECT_CASES = [
    (_lit("thier"), ENGLISH),
    (_lit("hius"), DUTCH),
    (_lit("then"), ENGLISH),
    (_lit("korrectud"), ENGLISH),
    (_lit("bycycle"), ENGLISH),
    (_lit("peotry"), ENGLISH),
    (_lit("acress"), ENGLISH),
    (_lit("kat"), DUTCH),
    (_lit("mensch"), DUTCH),
    (_lit("fietsenstaling"), DUTCH),
    (_lit("gegan"), DUTCH),
]
_K_START = "def correct(word, counts):\n    ...\n"
_K = ("def correct(word, counts):\n"
      "    if word in counts:\n"
      "        return word\n"
      "    near = near_words(word, counts, LETTERS)\n"
      "    if len(near) > 0:\n"
      "        return most_common(near, counts)\n"
      "    far = far_words(word, counts)\n"
      "    if len(far) > 0:\n"
      "        return most_common(far, counts)\n"
      "    return word\n")
_K_NO_KNOWN = _K.replace("    if word in counts:\n        return word\n", "")
_K_FAR_FIRST = ("def correct(word, counts):\n"
                "    if word in counts:\n"
                "        return word\n"
                "    far = far_words(word, counts)\n"
                "    if len(far) > 0:\n"
                "        return most_common(far, counts)\n"
                "    near = near_words(word, counts, LETTERS)\n"
                "    if len(near) > 0:\n"
                "        return most_common(near, counts)\n"
                "    return word\n")
_K_TOGETHER = ("def correct(word, counts):\n"
               "    if word in counts:\n"
               "        return word\n"
               "    words = near_words(word, counts, LETTERS) + far_words(word, counts)\n"
               "    if len(words) > 0:\n"
               "        return most_common(words, counts)\n"
               "    return word\n")
_K_NO_FAR = _K.replace("    far = far_words(word, counts)\n    if len(far) > 0:\n        return most_common(far, counts)\n", "")
_K_NONE = _K.replace("        return most_common(far, counts)\n    return word\n",
                     "        return most_common(far, counts)\n    return None\n")
_K_LIST = _K.replace("        return most_common(near, counts)\n", "        return near\n")

_NAMES_K = {"near_words": near_words, "most_common": most_common, "far_words": far_words, "LETTERS": LETTERS}

EXERCISES = {
    "near-words": FunctionBox(
        name="near-words",
        picture=near_picture,
        function="near_words",
        start=_N_START,
        solution=_N,
        visible=NEAR_CASES[:2],
        hidden=NEAR_CASES[2:],
        first=[_DOTS],
        collect=as_words,
        show=_shown_words,
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function stops with an error: it asks for a letter at a place where the string has no letter. At the end of the word, the part after the cut holds one letter or none. Does each kind of edit check first that enough letters come after the cut: one for a deletion or a substitution, two for a swap?",
                    nl="Met `{call}` stopt je functie met een fout: je functie vraagt een letter op een plek waar de string geen letter heeft. Aan het eind van het woord bevat het stuk na de knip één letter of geen. Kijkt elke soort bewerking eerst of er genoeg letters na de knip komen: één voor weghalen of vervangen, twee voor verwisselen?"),
            Mistake("lookup", error=_key_error,
                    en="With `{call}`, your function stops with an error: it looks up the count of a string that is not a key of `counts`. Most strings one edit away are not words, and looking up a string that is not a key stops with an error. Keep only the strings that are keys of `counts`, without looking up their counts.",
                    nl="Met `{call}` stopt je functie met een fout: je functie zoekt het aantal op van een string die geen key van `counts` is. De meeste strings op één bewerking zijn geen woord, en een string opzoeken die geen key is, stopt met een fout. Houd alleen de strings over die een key van `counts` zijn, zonder het aantal op te zoeken."),
            Mistake("all-strings", wrong=_N_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back every string one edit away, also the ones that are not words. Keep only the strings that are keys of `counts`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft elke string op één bewerking terug, ook de strings die geen woord zijn. Houd alleen de strings over die een key van `counts` zijn."),
            Mistake("no-delete", wrong=_N_NO_DELETE,
                    en="`{call}` gives {got}, and it should give {want}: your function makes no deletions. At each place, leave out the first letter of `rest`: `start` followed by `rest` without its first letter.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie haalt geen letters weg. Laat op elke plek de eerste letter van `rest` weg: `start` gevolgd door `rest` zonder de eerste letter."),
            Mistake("no-swap", wrong=_N_NO_SWAP,
                    en="`{call}` gives {got}, and it should give {want}: your function swaps no letters. At each place where `rest` has two letters or more, put the second letter of `rest` before the first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie verwisselt geen letters. Zet op elke plek waar `rest` twee letters of meer heeft, de tweede letter van `rest` vóór de eerste."),
            Mistake("no-substitute", wrong=_N_NO_SUB,
                    en="`{call}` gives {got}, and it should give {want}: your function makes no substitutions. For each letter of `letters`, also put that letter in the place of the first letter of `rest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vervangt geen letters. Zet voor elke letter van `letters` die letter ook op de plaats van de eerste letter van `rest`."),
            Mistake("no-insert", wrong=_N_NO_INSERT,
                    en="`{call}` gives {got}, and it should give {want}: your function inserts no letters. For each letter of `letters`, also put that letter between `start` and `rest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie voegt geen letters in. Zet voor elke letter van `letters` die letter ook tussen `start` en `rest`."),
            Mistake("no-end", wrong=_N_NO_END,
                    en="`{call}` gives {got}, and it should give {want}: your function never inserts a letter at the end of the word. A word of five letters has six places to cut, from before the first letter to after the last: `range(len(word) + 1)`. At the last place `rest` is empty, and the insertion still belongs there, without a question about `rest`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie voegt nooit een letter in aan het eind van het woord. Een woord van vijf letters heeft zes plekken om te knippen, van vóór de eerste letter tot na de laatste: `range(len(word) + 1)`. Op de laatste plek is `rest` leeg, en ook daar hoort het invoegen, zonder vraag over `rest`."),
            Mistake("no-start", wrong=_N_NO_START,
                    en="`{call}` gives {got}, and it should give {want}: your function never cuts before the first letter, so it inserts nothing at the start and never changes the first letter. Cut before the first letter too: the places run from 0 to `len(word)`, with `range(len(word) + 1)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie knipt nooit vóór de eerste letter, dus je functie voegt vooraan niets in en verandert de eerste letter nooit. Knip ook vóór de eerste letter: de plekken lopen van 0 tot en met `len(word)`, met `range(len(word) + 1)`."),
            Mistake("return-in-loop", wrong=_N_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back what it found after the first place to cut. Put the `return` after the loop, at the same depth as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft al terug wat je functie vond na de eerste plek om te knippen. Zet de `return` na de loop, even ver ingesprongen als `for`."),
            Mistake("too-many", matches=_too_many,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back strings that are not words. Keep only the strings that are keys of `counts`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft strings terug die geen woord zijn. Houd alleen de strings over die een key van `counts` zijn."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: cut the word at every place, from before the first letter to after the last; at each place, make the deletion, the swap, the substitutions and the insertions; and give back the strings that are keys of `counts`, sorted, each once.",
                      "nl": "Vervang de `...`: knip het woord op elke plek, van vóór de eerste letter tot na de laatste; maak op elke plek het weghalen, het verwisselen, het vervangen en het invoegen; en geef de strings terug die een key van `counts` zijn, gesorteerd, elk één keer."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function make all four kinds of edit at every place, and keep only the strings that are keys of `counts`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Maakt je functie op elke plek alle vier soorten bewerkingen, en houdt je functie alleen de strings over die een key van `counts` zijn?"},
            "almost": {"en": "Right for the typos of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every typo: one with nothing one edit away, other letters, an empty word?",
                       "nl": "Goed voor de tikfouten van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tikfout: een met niets op één bewerking, andere letters, een leeg woord?"},
            "unreached": {"en": "With `{call}`, your function gives nothing back. Does it end with a `return` of the words, after the loop?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie met een `return` van de woorden, na de loop?"},
            "correct": {"en": "Your `near_words` is right in all {n} cases. Look at the picture: your function on four typos, with each word it finds.",
                        "nl": "Je `near_words` klopt in alle {n} gevallen. Kijk naar de afbeelding: je functie op vier tikfouten, met elk woord dat je functie vindt."},
        },
    ),
    "most-common": FunctionBox(
        name="most-common",
        picture=pick_picture,
        function="most_common",
        start=_M_START,
        solution=_M,
        visible=PICK_CASES[:2],
        hidden=PICK_CASES[2:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("empty-list", error=_index_error,
                    en="With `{call}`, your function stops with an error: it asks for a word at a place that the list does not have. An empty list has no first word. Start with `best = None` and `best_count = 0`, and let the loop do the rest.",
                    nl="Met `{call}` stopt je functie met een fout: je functie vraagt een woord op een plek die de lijst niet heeft. Een lege lijst heeft geen eerste woord. Begin met `best = None` en `best_count = 0`, en laat de loop de rest doen."),
            Mistake("a-count", matches=_a_count,
                    en="`{call}` gives {got}: the count of the most common word. Here it should give the word itself, {want}.",
                    nl="`{call}` geeft {got}: hoe vaak het vaakste woord voorkomt. Hier moet het woord zelf eruit komen, {want}."),
            Mistake("not-updated", wrong=_M_NOT_UPDATED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the last word of the list. `best_count` stays 0, so every word beats it. When a word wins, also keep its count in `best_count`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het laatste woord van de lijst terug. `best_count` blijft 0, dus elk woord komt erboven. Bewaar ook het aantal van een woord in `best_count` als dat woord wint."),
            Mistake("return-in-loop", wrong=_M_RETURN_IN,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the first word, after one round of the loop. Put `return best` after the loop, at the same depth as `for`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie geeft het eerste woord al terug, na één ronde van de loop. Zet `return best` na de loop, even ver ingesprongen als `for`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: start with no word and the count 0, go over the words, keep a word and its count whenever its count is higher than the highest so far, and give back the word you kept.",
                      "nl": "Vervang de `...`: begin met geen woord en het aantal 0, ga de woorden langs, bewaar een woord met het aantal als het aantal hoger is dan het hoogste tot nu toe, en geef het bewaarde woord terug."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the word with the highest count, and give back that word, not its count?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Bewaart je functie het woord met het hoogste aantal, en geeft je functie dat woord terug, niet het aantal?"},
            "almost": {"en": "Right for the examples of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every list: the most common word first or in the middle, one word, no words at all?",
                       "nl": "Goed voor de voorbeelden van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke lijst: het woord dat het vaakst voorkomt vooraan of in het midden, één woord, helemaal geen woorden?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a word. Does `best` ever change? The question in the loop must be true for a word that is more common than every word before it: its count greater than `best_count`. And does the function end with `return best`, after the loop?",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een woord zijn. Verandert `best` ooit? De vraag in de loop moet waar zijn voor een woord dat vaker voorkomt dan elk woord ervoor: het aantal groter dan `best_count`. En eindigt je functie met `return best`, na de loop?"},
            "correct": {"en": "Your `most_common` is right in all {n} cases. Look at the picture: the words one edit from four typos, with how often each occurs, and the word your function picks.",
                        "nl": "Je `most_common` klopt in alle {n} gevallen. Kijk naar de afbeelding: de woorden op één bewerking van vier tikfouten, met hoe vaak elk voorkomt, en het woord dat je functie kiest."},
        },
    ),
    "correct": FunctionBox(
        name="correct",
        picture=correct_picture,
        function="correct",
        start=_K_START,
        solution=_K,
        names=_NAMES_K,
        visible=CORRECT_CASES[:2],
        hidden=CORRECT_CASES[2:],
        first=[_DOTS],
        show=_value,
        mistakes=[
            Mistake("a-list", matches=_a_list,
                    en="`{call}` gives {got}: a list of words. Here it should give one word, {want}: the most common word of the list, with `most_common`.",
                    nl="`{call}` geeft {got}: een lijst woorden. Hier moet er één woord uitkomen, {want}: het woord uit de lijst dat het vaakst voorkomt, met `most_common`."),
            Mistake("far-values", error=_far_values,
                    en="With `{call}`, your function stops with an error: `far_words` takes two values, the word and the counts, and no letters.",
                    nl="Met `{call}` stopt je functie met een fout: `far_words` krijgt twee waarden, het woord en de aantallen, en geen letters."),
            Mistake("no-known", wrong=_K_NO_KNOWN,
                    en="`{call}` gives {got}, and it should give {want}: your function changes words that are already words, into a more common word one edit away. Ask first whether `word` is in `counts`, and if so, give it back as it is.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie verandert woorden die al een woord zijn, in een woord op één bewerking dat vaker voorkomt. Vraag eerst of `word` in `counts` staat, en geef het dan terug zoals het is."),
            Mistake("far-first", wrong=_K_FAR_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function lets the words two edits away compete while there are words one edit away, and a more common word two edits away then wins. Look one edit away first, and two edits away only when that finds nothing.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie laat de woorden op twee bewerkingen meedoen terwijl er woorden op één bewerking zijn, en dan wint een woord op twee bewerkingen dat vaker voorkomt. Kijk eerst op één bewerking, en alleen als dat niets vindt op twee bewerkingen."),
            Mistake("no-far", wrong=_K_NO_FAR,
                    en="`{call}` gives {got}, and it should give {want}: when nothing is one edit away, your function gives back the word as it is. Look two edits away then, with `far_words`, before you give up.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: als er niets op één bewerking ligt, geeft je functie het woord terug zoals het is. Kijk dan eerst op twee bewerkingen, met `far_words`, voordat je opgeeft."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: give back the word itself when it is in `counts`; else the most common word one edit away, if there is one; else the most common word two edits away, if there is one; else the word itself.",
                      "nl": "Vervang de `...`: geef het woord zelf terug als het in `counts` staat; anders het woord op één bewerking dat het vaakst voorkomt, als dat er is; anders het woord op twee bewerkingen dat het vaakst voorkomt, als dat er is; anders het woord zelf."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function ask the four questions in this order: is the word a word, is there a word one edit away, is there one two edits away, and else the word itself?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Stelt je functie de vier vragen in deze volgorde: is het woord een woord, ligt er een woord op één bewerking, ligt er een op twee bewerkingen, en anders het woord zelf?"},
            "almost": {"en": "Right for the typos of the lesson. But `{call}` gives {got}, and it should give {want}. Does your function work for every word: one that is a word already, one two edits from its word, one with no known word within two edits?",
                       "nl": "Goed voor de tikfouten van de les. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elk woord: een woord dat al een woord is, een tikfout op twee bewerkingen van het woord, een tikfout zonder bekend woord binnen twee bewerkingen?"},
            "unreached": {"en": "With `{call}`, your function gives back `None`, and it should give a word. When nothing lies one edit away, `most_common` of the empty list gives `None`: give back the most common word one edit away only when there is one. And does every way through the function end with a `return` of a word? When no known word lies within two edits, give back the word itself.",
                          "nl": "Met `{call}` geeft je functie `None` terug, en dat moet een woord zijn. Als er niets op één bewerking ligt, geeft `most_common` van de lege lijst `None`: geef het vaakste woord op één bewerking alleen terug als er zo'n woord is. En eindigt elke weg door je functie met een `return` van een woord? Als er binnen twee bewerkingen geen bekend woord ligt, geef dan het woord zelf terug."},
            "correct": {"en": "Your `correct` is right in all {n} cases. Look at the picture: your corrector on a sentence with typos, on three hard cases, and on the made-up typos of the guess.",
                        "nl": "Je `correct` klopt in alle {n} gevallen. Kijk naar de afbeelding: je corrector op een zin met tikfouten, op drie moeilijke gevallen, en op de verzonnen tikfouten van de gok."},
        },
    ),
    "what-happens": Sort(
        name="what-happens",
        groups={
            "fixed": {"en": "Put right", "nl": "Verbeterd"},
            "other": {"en": "Changed into a more common word", "nl": "Veranderd in een woord dat vaker voorkomt"},
            "kept": {"en": "Left alone: the typo is a word", "nl": "Blijft staan: de tikfout is een woord"},
        },
        items={
            "receive": {"en": "**You type *recieve* and mean *receive*.**", "nl": "**Je typt *vreind* en bedoelt *vriend*.**"},
            "lose": {"en": "**You type *loose* and mean *lose*.**", "nl": "**Je typt *word* en bedoelt *wordt*.**"},
            "weather": {"en": "**You type *wether* and mean *weather*.**", "nl": "**Je typt *fiest* en bedoelt *feest*.**"},
            "friend": {"en": "**You type *freind* and mean *friend*.**", "nl": "**Je typt *morgn* en bedoelt *morgen*.**"},
            "which": {"en": "**You type *wich* and mean *which*.**", "nl": "**Je typt *rgeen* en bedoelt *regen*.**"},
            "their": {"en": "**You type *there* and mean *their*.**", "nl": "**Je typt *gebeurd* en bedoelt *gebeurt*.**"},
        },
        solution={"receive": "fixed", "friend": "fixed", "weather": "other", "which": "other", "lose": "kept",
                  "their": "kept"},
        nudge={
            "receive": {"en": "Is *recieve* itself a word in the books? Of the words one edit away, which is the most common?",
                        "nl": "Is *vreind* zelf een woord in de zinnen? Welk woord krijg je als je twee buurletters verwisselt?"},
            "lose": {"en": "Is *loose* itself a word that the books contain?",
                     "nl": "Is *word* zelf een woord dat in de zinnen staat?"},
            "weather": {"en": "*weather* is one edit from *wether*. Which other word is one edit away, and which of the two do the books use more often?",
                        "nl": "*feest* ligt op één bewerking van *fiest*. Welk ander woord ligt op één bewerking, en welk van de twee staat vaker in de zinnen?"},
            "friend": {"en": "Is *freind* itself a word in the books? Which word do you get when you swap two neighboring letters?",
                       "nl": "Is *morgn* zelf een woord in de zinnen? Welk woord krijg je als je één letter invoegt?"},
            "which": {"en": "*which* is one edit from *wich*. Which other common word is one edit away, and which of the two do the books use more often?",
                      "nl": "*regen* ligt op één bewerking van *rgeen*. Welk ander woord ligt op één bewerking, en welk van de twee staat vaker in de zinnen?"},
            "their": {"en": "Is *there* itself a word that the books contain?",
                      "nl": "Is *gebeurd* zelf een woord dat in de zinnen staat?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the typos again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de tikfouten opnieuw uit."},
            "empty": {"en": "Put each bold typo under what the corrector does with it.",
                      "nl": "Zet elke vetgedrukte tikfout bij wat de corrector ermee doet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item}: this typo goes somewhere else.",
                      "nl": "{item}: deze tikfout hoort ergens anders."},
            "correct": {"en": "All {n} are right. The corrector looks at one word at a time, and knows only how often each word occurs. So it cannot see that *loose* should be *lose*, since *loose* is a word, and where two words lie equally close, the more common one wins, even when the sentence asks for the other. To do better, a corrector has to look at the words around a typo.",
                        "nl": "Alle {n} kloppen. De corrector bekijkt één woord tegelijk, en weet alleen hoe vaak elk woord voorkomt. Dus ziet de corrector niet dat *word* *wordt* moet zijn, want *word* is een woord, en waar twee woorden even dichtbij liggen, wint het woord dat vaker voorkomt, ook als de zin om het andere vraagt. Om het beter te doen, moet een corrector naar de woorden rond een tikfout kijken."},
        },
    ),
}

CASES = [
    ("near-words", _N_START, "empty", None),
    ("near-words", _N, "correct", None),
    ("near-words", "def near_words(word, counts, letters):\n    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]\n    deletes = [a + b[1:] for a, b in splits if b]\n    swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]\n    subs = [a + c + b[1:] for a, b in splits if b for c in letters]\n    inserts = [a + c + b for a, b in splits for c in letters]\n    return set(w for w in deletes + swaps + subs + inserts if w in counts)\n", "correct", None),
    ("near-words", "def near_words(word, counts, letters):\n    found = set()\n    for i in range(len(word) + 1):\n        for c in letters:\n            found.add(word[:i] + c + word[i:])\n    for i in range(len(word)):\n        found.add(word[:i] + word[i + 1:])\n        for c in letters:\n            found.add(word[:i] + c + word[i + 1:])\n    for i in range(len(word) - 1):\n        found.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])\n    return sorted(w for w in found if w in counts and w != word)\n", "correct", None),
    ("near-words", _N_SWAP_UNCHECKED, "error", "index"),
    ("near-words", _N.replace("[w for w in found if w in counts]", "[w for w in found if counts[w] > 0]"), "error", "lookup"),
    ("near-words", _N_ALL, "wrong", "all-strings"),
    ("near-words", _N_NO_DELETE, "wrong", "no-delete"),
    ("near-words", _N_NO_SWAP, "wrong", "no-swap"),
    ("near-words", _N.replace("rest[1] + rest[0] + rest[2:]", "rest[0] + rest[1] + rest[2:]"), "wrong", "no-swap"),
    ("near-words", _N_NO_SUB, "wrong", "no-substitute"),
    ("near-words", _N_NO_INSERT, "wrong", "no-insert"),
    ("near-words", _N_NO_END, "wrong", "no-end"),
    ("near-words", _N.replace("            found.append(start + c + rest)\n", "            if len(rest) > 0:\n                found.append(start + c + rest)\n"), "wrong", "no-end"),
    ("near-words", _N_NO_START, "wrong", "no-start"),
    ("near-words", _N_RETURN_IN, "wrong", "return-in-loop"),
    ("near-words", _N.replace("    return sorted(set([w for w in found if w in counts]))\n", "    return found + ['x']\n"), "wrong", "too-many"),
    ("near-words", _N.replace("    return sorted(set([w for w in found if w in counts]))\n", ""), "wrong", "no-return"),
    ("near-words", "def near_words(word, counts, letters):\n    if word == \"thier\":\n        return [\"their\", \"thief\", \"tier\"]\n    return [\"heus\", \"hijs\", \"hits\", \"huis\"]\n", "almost", None),
    ("most-common", _M_START, "empty", None),
    ("most-common", _M, "correct", None),
    ("most-common", "def most_common(words, counts):\n    if len(words) == 0:\n        return None\n    return max(words, key=lambda w: counts[w])\n", "correct", None),
    ("most-common", "def most_common(words, counts):\n    best = None\n    for w in words:\n        if best == None or counts[w] > counts[best]:\n            best = w\n    return best\n", "correct", None),
    ("most-common", _M_FIRST, "error", "empty-list"),
    ("most-common", _M_COUNT, "wrong", "a-count"),
    ("most-common", _M_NOT_UPDATED, "wrong", "not-updated"),
    ("most-common", _M_RETURN_IN, "wrong", "return-in-loop"),
    ("most-common", _M_LEAST, "wrong", "unreached"),
    ("most-common", _M.replace("    return best\n", ""), "wrong", "no-return"),
    ("most-common", "def most_common(words, counts):\n    if words[0] == \"their\":\n        return \"their\"\n    return \"huis\"\n", "almost", None),
    ("correct", _K_START, "empty", None),
    ("correct", _K, "correct", None),
    ("correct", "def correct(word, counts):\n    if word in counts:\n        return word\n    for words in (near_words(word, counts, LETTERS), far_words(word, counts)):\n        if words:\n            return most_common(words, counts)\n    return word\n", "correct", None),
    ("correct", _K_LIST, "wrong", "a-list"),
    ("correct", _K_NO_KNOWN, "wrong", "no-known"),
    ("correct", _K_FAR_FIRST, "wrong", "far-first"),
    ("correct", _K_TOGETHER, "wrong", "far-first"),
    ("correct", _K_NO_FAR, "wrong", "no-far"),
    ("correct", _K_NONE, "wrong", "unreached"),
    ("correct", _K.replace("far_words(word, counts)", "far_words(word, counts, LETTERS)"), "error", "far-values"),
    ("correct", _K.replace("    if len(near) > 0:\n        return most_common(near, counts)\n", "    return most_common(near, counts)\n"), "wrong", "unreached"),
    ("correct", "def correct(word, counts):\n    return {\"thier\": \"their\", \"hius\": \"huis\"}[word]\n", "error", None),
    ("correct", "def correct(word, counts):\n    if word == \"thier\":\n        return \"their\"\n    return \"huis\"\n", "almost", None),
    ("what-happens", {}, "empty", None),
    ("what-happens", {"receive": "fixed", "friend": "fixed", "weather": "other", "which": "other", "lose": "kept",
                      "their": "kept"}, "correct", None),
    ("what-happens", {"receive": "fixed", "lose": "kept"}, "wrong", "unfinished"),
    ("what-happens", {"lose": "fixed"}, "wrong", "wrong:lose"),
    ("what-happens", {"receive": "fixed", "friend": "fixed", "weather": "fixed", "which": "other", "lose": "kept",
                      "their": "kept"}, "wrong", "wrong:weather"),
    ("what-happens", {"receive": "fixed", "friend": "fixed", "weather": "other", "which": "other", "lose": "kept",
                      "their": "other"}, "wrong", "wrong:their"),
]
