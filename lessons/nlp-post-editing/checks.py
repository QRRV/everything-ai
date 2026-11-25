# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-post-editing: moving a block of words to another place in a list (shift), the fuzzy match
of two sentences from their edit distance (fuzzy), and the best match for a new sentence in a translation memory
(best_match).

The sentences are the 3,707 of nlp-bleu: Dutch sentences of Tatoeba (https://tatoeba.org, CC BY 2.0 FR) whose every
word is in the course's dictionary and that have more than one English translation by people (data/tatoeba-nl.tsv.gz
and data/tatoeba-nl-en.tsv.gz, the export of November 21, 2026). Each is translated in three ways: word for word, as
in nlp-classic-mt; by Qwen3-0.6B, the course's chat model, in the course's run (data/course-post-editing-qwen.json,
greedy, made with data/tools/record_course_post_editing_qwen.py); and by people, the first translation against the
others. For 40 of Qwen's translations the course wrote a post-edit, changing as little as possible
(data/course-post-edits.tsv). The translation memory holds every other Dutch sentence of Tatoeba with an English
translation, without the sentences whose translations use a gendered pronoun; for each of the 3,707 sentences,
data/course-tm-matches.tsv names its best match, found with data/tools/make_course_tm_matches.py.

TER follows Snover and others (2006) and sacreBLEU 2.6 (Post 2018), which works like tercom: insertions, deletions and
substitutions of single words, and shifts of blocks of up to 10 words, all costing 1; the shifts are chosen greedily,
each time the one that lowers the edit distance most, among blocks of wrong words that match the reference where they
land; the edit count of the best reference, divided by the average length of the references. On the same words
sacreBLEU gives the same edit counts as the course for every sentence of the lesson, which the course checked when it
made the lesson. The words are those of nlp-classic-mt: small letters, no punctuation, contractions written out. Every
number in the text and the pictures is computed by the course (tests/test_nlp_post_editing.py computes them again)."""
import functools
import html
import json
import math

import numpy as np

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

QWEN_FILE = paths.ROOT / "data" / "course-post-editing-qwen.json"
POST_EDITS_FILE = paths.ROOT / "data" / "course-post-edits.tsv"
MATCHES_FILE = paths.ROOT / "data" / "course-tm-matches.tsv"
# The prompt of the course's run, as data/tools/record_course_post_editing_qwen.py asks it
PROMPT = "Translate this Dutch sentence into English. Reply with only the English translation.\n\n{dutch}"
LANGS = ("en", "nl")

MAX_SHIFT_SIZE = 10     # a shift moves a block of at most 10 words,
MAX_SHIFT_DIST = 50     # that starts at most 50 places from where the reference has it,
MAX_TRIED = 1000        # and TER stops looking for shifts after trying 1,000 of them, as sacreBLEU does
BANDS = ((1.0, 1.01), (0.85, 1.0), (0.7, 0.85), (0.5, 0.7), (0.0, 0.5))   # fuzzy matches: 100%, 85-99%, ...


@functools.cache
def _mt():
    """The checks of nlp-classic-mt: its sentences, its dictionary and its translations word for word."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_classic_mt_for_post_editing",
                                                  paths.LESSONS / "nlp-classic-mt" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(text: str) -> list:
    """The words of an English text, in small letters, without punctuation, with contractions such as *I'll* written
    out, as nlp-classic-mt reads them."""
    return _mt().english_words(text)


def dutch_words(text: str) -> list:
    """The words of a Dutch text, in small letters, without punctuation."""
    return _mt().words(text)


# --- the edit distance, with the alignment TER needs ---------------------------------------------------------------------

def _cheapest(t: list, r: list) -> tuple:
    """The edit distance of a translation t and a reference r, both lists of words, and one cheapest alignment as its
    steps in reading order: "p" pairs a word of t with a word of r (the same word or a substitution), "t" is a word of
    t on its own (deleted), "r" a word of r on its own (inserted). Where two steps cost the same, the table takes a pair
    first, then a word of t on its own, then a word of r on its own, as tercom does."""
    n, m = len(t), len(r)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    step = [[""] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        D[0][j], step[0][j] = j, "r"
    for i in range(1, n + 1):
        D[i][0], step[i][0] = i, "t"
        for j in range(1, m + 1):
            best, how = D[i - 1][j - 1] + (t[i - 1] != r[j - 1]), "p"
            if D[i - 1][j] + 1 < best:
                best, how = D[i - 1][j] + 1, "t"
            if D[i][j - 1] + 1 < best:
                best, how = D[i][j - 1] + 1, "r"
            D[i][j], step[i][j] = best, how
    back, i, j = [], n, m
    while i > 0 or j > 0:
        how = step[i][j]
        back.append(how)
        if how == "p":
            i, j = i - 1, j - 1
        elif how == "t":
            i -= 1
        else:
            j -= 1
    return D[n][m], back[::-1]


def line_up(t: list, r: list) -> tuple:
    """(distance, align, t_wrong, r_wrong) of the cheapest alignment of t and r: align gives for each place of r the
    place of t it lines up with, and a word of r on its own the place of the last word of t before it (-1 at the
    start); t_wrong and r_wrong hold 1 for each word that is not paired with the same word."""
    d, how = _cheapest(t, r)
    align, t_wrong, r_wrong, h, k = {}, [], [], -1, -1
    for x in how:
        if x == "p":
            h, k = h + 1, k + 1
            align[k] = h
            bad = int(t[h] != r[k])
            t_wrong.append(bad)
            r_wrong.append(bad)
        elif x == "t":
            h += 1
            t_wrong.append(1)
        else:
            k += 1
            align[k] = h
            r_wrong.append(1)
    return d, align, t_wrong, r_wrong


def distance(a: list, b: list) -> int:
    """The edit distance of two lists of words: insertions, deletions and substitutions, each costing 1."""
    return _cheapest(a, b)[0]


def steps(t: list, r: list) -> list:
    """The cheapest alignment of t and r, as (word of t or None, word of r or None) in reading order."""
    out, h, k = [], 0, 0
    for x in _cheapest(t, r)[1]:
        if x == "p":
            out.append((t[h], r[k]))
            h, k = h + 1, k + 1
        elif x == "t":
            out.append((t[h], None))
            h += 1
        else:
            out.append((None, r[k]))
            k += 1
    return out


# --- the course's solutions of the exercises ---------------------------------------------------------------------------

def shift(words, start, length, to):
    block = words[start:start + length]
    rest = words[:start] + words[start + length:]
    return rest[:to] + block + rest[to:]


def fuzzy(new, stored):
    edits = distance(new, stored)
    longest = max(len(new), len(stored))
    return 1 - edits / longest


def best_match(new, memory):
    best = None
    best_score = 0
    for stored, translation in memory:
        score = fuzzy(new, stored)
        if score > best_score:
            best_score = score
            best = translation
    return best_score, best


# --- TER: shifts, then word edits ----------------------------------------------------------------------------------------

def _matches(t: list, r: list):
    """Every block of t that is the same as a block of r: (start in t, start in r, length), for lengths up to
    MAX_SHIFT_SIZE and starts at most MAX_SHIFT_DIST apart."""
    for a in range(len(t)):
        for b in range(len(r)):
            if abs(b - a) > MAX_SHIFT_DIST:
                continue
            length = 0
            while (length < MAX_SHIFT_SIZE and a + length < len(t) and b + length < len(r)
                   and t[a + length] == r[b + length]):
                length += 1
                yield a, b, length


def best_shift(t: list, r: list, tried: int, move=shift) -> tuple:
    """The shift of t that lowers its edit distance to r the most: (gain, start, length, to, shifted t, tried). It
    tries blocks of t that hold a wrong word and match a block of r that holds a wrong word, and puts each where the
    alignment says those words of r are; of equal gains it takes the longest block, then the first, then the place
    nearest the start, as tercom does. The gain is 0 when no shift helps."""
    edits, align, t_wrong, r_wrong = line_up(t, r)
    best = None
    for a, b, length in _matches(t, r):
        if not any(t_wrong[a:a + length]) or not any(r_wrong[b:b + length]):
            continue
        if a <= align[b] < a + length:   # the block would land inside itself
            continue
        last = None
        for offset in range(-1, length):
            if b + offset == -1:
                target = 0
            elif b + offset in align:
                target = align[b + offset] + 1   # just after the word of t that this word of r lines up with
            else:
                break
            if target == last:
                continue
            last = target
            to = target - length if target > a + length else target   # the place in t without the block
            moved = move(list(t), a, length, to)
            tried += 1
            key = (edits - distance(moved, r), length, -a, -target)
            if best is None or key > best[0]:
                best = (key, a, length, to, moved)
        if tried >= MAX_TRIED:
            break
    if best is None:
        return 0, None, None, None, t, tried
    return best[0][0], best[1], best[2], best[3], best[4], tried


def ter_path(t: list, r: list, move=shift) -> tuple:
    """How TER turns t into r: (moves, edits), with moves the shifts it makes, each as (start, length, to, t after
    it), and edits the edit distance that is left."""
    if not r:
        return [], len(t)
    moves, tried = [], 0
    while True:
        gain, a, length, to, moved, tried = best_shift(t, r, tried, move)
        if tried >= MAX_TRIED or gain <= 0:
            break
        moves.append((a, length, to, moved))
        t = moved
    return moves, distance(t, r)


def ter_edits(t: list, r: list, move=shift) -> int:
    """The number of TER edits from t to r: shifts plus word edits."""
    moves, edits = ter_path(t, r, move)
    return len(moves) + edits


def ter_counts(t: list, refs: list, move=shift) -> tuple:
    """(edits, words): the TER edits to the reference that needs the fewest, and the average length of the
    references."""
    return min(ter_edits(t, r, move) for r in refs), sum(len(r) for r in refs) / len(refs)


def ter(counts: list) -> float:
    """TER of many sentences: all their edits divided by all their average reference lengths, times 100."""
    words_ = sum(w for _, w in counts)
    return 100 * sum(e for e, _ in counts) / words_ if words_ else 0.0


# --- the sentences, the three ways to translate them, and the post-edits ----------------------------------------------------

@functools.cache
def _all_sentences() -> tuple:
    """(id, Dutch text, English translations by people) of the 3,707 sentences of nlp-bleu, in the order of
    nlp-classic-mt."""
    return tuple((sid, text, es) for sid, text, es in _mt()._covered() if len(es) > 1)


@functools.cache
def _sentences() -> tuple:
    """The sentences of the lesson: the 3,707 of nlp-bleu, without those whose translation by Qwen uses a gendered
    pronoun."""
    pronoun = _mt()._PRONOUN
    return tuple(s for s in _all_sentences() if not pronoun.search(qwen_text(s[0])))


@functools.cache
def _qwen() -> dict:
    with open(QWEN_FILE, encoding="utf-8") as f:
        return json.load(f)


def qwen_text(sid: int) -> str:
    """Qwen's translation of the sentence with this id, as it wrote it."""
    return _qwen()["translations"][str(sid)]["text"]


@functools.cache
def _post_edits() -> tuple:
    """(id, Qwen's translation, the course's post-edit) of the 40 post-edited sentences, in the order of the file."""
    with open(POST_EDITS_FILE, encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "id\tqwen\tpost_edit"
    return tuple((int(a), b, c) for a, b, c in (line.split("\t") for line in lines[1:] if line))


@functools.cache
def _systems() -> dict:
    """The translations of the lesson, as lists of words, each with its references: word for word and Qwen against
    every translation by people, and the first translation by people against the others."""
    mt = _mt()
    refs, direct, qwen, people = [], [], [], []
    for sid, text, es in _sentences():
        rs = tuple(tuple(words(e)) for e in es)
        refs.append(rs)
        direct.append(tuple(mt.translate(text)))
        qwen.append(tuple(words(qwen_text(sid))))
        people.append(rs[0])
    return {"direct": (tuple(direct), tuple(refs)), "qwen": (tuple(qwen), tuple(refs)),
            "people": (tuple(people), tuple(r[1:] for r in refs))}


SYSTEMS = ("direct", "qwen", "people")


def system(name: str) -> tuple:
    """(translations, references) of one of SYSTEMS, as new lists."""
    ts, rs = _systems()[name]
    return [list(t) for t in ts], [[list(r) for r in refs] for refs in rs]


@functools.cache
def _ter_counts(name: str) -> tuple:
    ts, rs = _systems()[name]
    return tuple(ter_counts(list(t), [list(r) for r in refs]) for t, refs in zip(ts, rs))


# --- the translation memory ------------------------------------------------------------------------------------------------

@functools.cache
def _memory() -> tuple:
    """The translation memory: (id, Dutch text, its first English translation) of every Dutch sentence of Tatoeba with
    an English translation, except the 3,707 and those with a translation that uses a gendered pronoun, in the order
    of the ids."""
    mt = _mt()
    lesson = {sid for sid, _, _ in _all_sentences()}
    return tuple((sid, text, es[0]) for sid, text, es in mt._pairs()
                 if sid not in lesson and dutch_words(text) and not any(mt._PRONOUN.search(e) for e in es))


@functools.cache
def _memory_by_id() -> dict:
    return {sid: (text, english) for sid, text, english in _memory()}


@functools.cache
def _matches_file() -> tuple:
    """(id of a sentence, id of its best match in the memory) for each of the 3,707 of nlp-bleu, in their order."""
    with open(MATCHES_FILE, encoding="utf-8") as f:
        lines = f.read().split("\n")
    assert lines[0] == "id\tmatch"
    return tuple((int(a), int(b)) for a, b in (line.split("\t") for line in lines[1:] if line))


@functools.cache
def matched() -> tuple:
    """For each sentence of the lesson: (its Dutch words, the Dutch words of its best match, the match's English
    words), in the order of the sentences."""
    by_id = _memory_by_id()
    best = dict(_matches_file())
    out = []
    for sid, text_, _ in _sentences():
        match, english = by_id[best[sid]]
        out.append((tuple(dutch_words(text_)), tuple(dutch_words(match)), tuple(words(english))))
    return tuple(out)


@functools.cache
def _match_ter() -> tuple:
    """For each sentence of the lesson: the TER counts of the memory's translation against the translations by
    people."""
    refs = _systems()["qwen"][1]
    return tuple(ter_counts(list(e), [list(r) for r in rs]) for (_, _, e), rs in zip(matched(), refs))


def band(score: float) -> int:
    """The index in BANDS of a fuzzy match from 0 to 1."""
    for k, (lo, hi) in enumerate(BANDS):
        if lo <= score < hi:
            return k
    return len(BANDS) - 1


@functools.cache
def _index() -> tuple:
    """For finding matches fast: the Dutch words of the memory, and for each word the places of the memory that have
    it, with how often."""
    mem = tuple(tuple(dutch_words(text)) for _, text, _ in _memory())
    places = {}
    for i, ws in enumerate(mem):
        counts = {}
        for w in ws:
            counts[w] = counts.get(w, 0) + 1
        for w, k in counts.items():
            places.setdefault(w, []).append((i, k))
    return mem, places


def search(new: list, k: int = 1) -> list:
    """The k best matches for the Dutch words `new` in the memory, as (fuzzy, place in the memory), best first, and of
    equal matches the first in the memory. A sentence shares at most as many words as both have, so the fuzzy match
    of a sentence is at most what it shares divided by the longer length: sentences that cannot beat the k-th match
    found are not compared at all."""
    mem, places = _index()
    if not new:
        return []
    mine = {}
    for w in new:
        mine[w] = mine.get(w, 0) + 1
    shared = {}
    for w, n in mine.items():
        for i, c in places.get(w, ()):
            shared[i] = shared.get(i, 0) + min(n, c)
    def most(i):   # written as 1 - x / longest, as fuzzy is, so that equal fractions give equal numbers
        longest = max(len(mem[i]), len(new))
        return 1 - (longest - shared[i]) / longest

    order = sorted(shared, key=lambda i: (-most(i), i))
    found = []
    for i in order:
        bound = most(i)
        if len(found) >= k and bound < found[k - 1][0]:
            break
        found.append((1 - distance(list(new), list(mem[i])) / max(len(mem[i]), len(new)), i))
        found.sort(key=lambda x: (-x[0], x[1]))
    return found[:k]



# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code."""
    return f'<div style="{_UI}">{_marked(text_)}</div>'


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 1) -> str:
    out = f"{round(float(x), digits) + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _pct(part, whole, lang: str) -> str:
    return _dec(100 * part / whole if whole else 0.0, lang) + "%"


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _gray(text_: str) -> str:
    return f'<span style="{_DATA}">{html.escape(text_)}</span>'


def _faint(text_: str) -> str:
    return f'<span style="{_FAINT}">{html.escape(text_)}</span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None



_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 2   # what the learner's function gets for all the calls of one picture together
SHIFT_SECONDS = 10    # the picture of exercise 1, which works out TER of every sentence of the lesson three times


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    import contextlib
    import io
    from everything_ai.check import Timeout, with_time_limit
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _empty(w: dict, key: str, seconds: float = PICTURE_SECONDS) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=seconds, kind=w.get("kind", ""))
    return yours.figure(w["title"], _note(text_), alt=text_)


_WHY_EMPTY = {
    "en": {"stopped": "Your function stops with an error, so there is nothing to draw yet. The check above says why.",
           "slow": "Your function took longer than {s} seconds, so there is nothing to draw yet.",
           "odd": "Your function gives something other than {kind}, so there is nothing to draw yet. The check above says what."},
    "nl": {"stopped": "Je functie stopt met een fout, dus er valt nog niets te tekenen. De check hierboven zegt waarom.",
           "slow": "Je functie deed er langer over dan {s} seconden, dus er valt nog niets te tekenen.",
           "odd": "Je functie geeft iets anders dan {kind}, dus er valt nog niets te tekenen. De check hierboven zegt wat."},
}



def _bar(share: float, color: str) -> str:
    """A bar for a share from 0 to 1, on a track of 10rem."""
    width = max(0.0, min(1.0, share)) * 100
    return (f'<div style="width:10rem;height:0.7rem;margin-top:0.3rem;border-radius:2px;background:var(--ea-data-wash)">'
            f'<div style="width:{width:.1f}%;height:100%;border-radius:2px;background:{color}"></div></div>')


def _score(x: float, lang: str) -> str:
    """A score from 0 to 1 the way the text writes it: times 100, with one decimal."""
    return _dec(100 * x, lang)


def _is_count(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))



IDEA = 385382
MATCH_ROWS = (437283, 462193, 603682, 377360, 381349)
SHORT = 4   # sentences of up to this many Dutch words count as short in the text



# --- one translation against one reference: what the idea, the box and the text show -----------------------------------

_TER_TEXT = {
    "en": {"reference": "The reference:",
           "translation": "The translation:",
           "shift": "Shift {k}: *{block}* moves to place {to}, counting from 0:",
           "no_shift": "No shift lowers the edit distance, so TER makes none.",
           "left": "Then the word edits that are left:",
           "count": "{s} and {e}: {t} for the {w} words of the reference, so TER is {n}/{w} = {pct}.",
           "total": ("{n} TER edits", "{n} TER edit", "{n} TER edits"),
           "shifts": ("no shift", "1 shift", "{n} shifts"),
           "edits": ("no word edits", "1 word edit", "{n} word edits"),
           "plain": "Without shifts, the edit distance is {d}: {pct}.",
           "note": "Gray: the reference. Blue: a word of the translation that the reference has at the same place, and a block that a shift moved. Purple: a word edit: *a → b* substitutes a word, *−a* deletes one and *+b* inserts one."},
    "nl": {"reference": "De referentie:",
           "translation": "De vertaling:",
           "shift": "Shift {k}: *{block}* gaat naar plaats {to}, geteld vanaf 0:",
           "no_shift": "Geen enkele shift verlaagt de edit distance, dus TER maakt er geen.",
           "left": "Dan de woordbewerkingen die over zijn:",
           "count": "{s} en {e}: {t} voor de {w} woorden van de referentie, dus TER is {n}/{w} = {pct}.",
           "total": ("{n} TER-bewerkingen", "{n} TER-bewerking", "{n} TER-bewerkingen"),
           "shifts": ("geen shift", "1 shift", "{n} shifts"),
           "edits": ("geen woordbewerkingen", "1 woordbewerking", "{n} woordbewerkingen"),
           "plain": "Zonder shifts is de edit distance {d}: {pct}.",
           "note": "Grijs: de referentie. Blauw: een woord van de vertaling dat de referentie op dezelfde plaats heeft, en een blok dat een shift verplaatste. Paars: een woordbewerking: *a → b* vervangt een woord, *−a* haalt er een weg en *+b* voegt er een toe."},
}


def _many(forms: tuple, n: int) -> str:
    return forms[min(n, 2)].format(n=n)


def ter_view(lang: str, translation: list, reference: list, labels: tuple = (), more: str = "") -> str:
    """TER of one translation against one reference, both lists of words, step by step: the shifts, then the word
    edits that are left. labels: what to call the reference and the translation, if not that; more: a note to add
    before the note on the colors."""
    w = _TER_TEXT[lang]
    ref_label, t_label = labels or (w["reference"], w["translation"])
    body = _label(ref_label) + "<div>" + "".join(_chip(x, _CHIP_DATA) for x in reference) + "</div>"
    body += _label(t_label) + "<div>" + "".join(html.escape(x) + " " for x in translation) + "</div>"
    moves, edits = ter_path(list(translation), list(reference))
    for k, (a, length, to, moved) in enumerate(moves, start=1):
        block = " ".join(moved[to:to + length])
        chips = "".join(_chip(x, _CHIP_PRED) if to <= i < to + length else html.escape(x) + " "
                        for i, x in enumerate(moved))
        body += _label(w["shift"].format(k=k, block=block, to=to)) + f"<div>{chips}</div>"
    if not moves:
        body += _note(w["no_shift"])
    final = moves[-1][3] if moves else list(translation)
    chips = []
    for a, b in steps(final, list(reference)):
        if a == b:
            chips.append(_chip(a, _CHIP_PRED))
        elif a is None:
            chips.append(_chip("+" + b, _CHIP_LOSS))
        elif b is None:
            chips.append(_chip("−" + a, _CHIP_LOSS))
        else:
            chips.append(_chip(f"{a} → {b}", _CHIP_LOSS))
    body += _label(w["left"]) + "<div>" + "".join(chips) + "</div>"
    n, words_ = len(moves) + edits, len(reference)
    body += _note(w["count"].format(s=_many(w["shifts"], len(moves)), e=_many(w["edits"], edits), n=n, w=words_,
                                    t=_many(w["total"], n), pct=_pct(n, words_, lang)))
    plain = distance(list(translation), list(reference))
    if moves:
        body += _note(w["plain"].format(d=plain, pct=_pct(plain, words_, lang)))
    return body + (_note(more) if more else "") + _note(w["note"])


# --- the idea: one of Qwen's translations with a shift -----------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "TER of one translation: shifts, then word edits",
           "pair": "Qwen's translation of *{dutch}*, against the translation by people that needs the fewest edits.",
           "labels": ("The translation by people:", "Qwen's translation:"),
           "average": "With the {k} translations by people, TER divides by their average length, {avg} words: {n}/{avg} = {pct}."},
    "nl": {"title": "TER van één vertaling: shifts, dan woordbewerkingen",
           "pair": "De vertaling van Qwen van *{dutch}*, tegen de vertaling door mensen die de minste bewerkingen nodig heeft.",
           "labels": ("De vertaling door mensen:", "De vertaling van Qwen:"),
           "average": "Met de {k} vertalingen door mensen deelt TER door de gemiddelde lengte, {avg} woorden: {n}/{avg} = {pct}."},
}


def _closest(t: list, refs: list) -> list:
    """The reference that needs the fewest TER edits, the first of equals."""
    best = None
    for r in refs:
        e = ter_edits(list(t), list(r))
        if best is None or e < best[0]:
            best = (e, r)
    return list(best[1])


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    sid, dutch, es = next(s for s in _sentences() if s[0] == IDEA)
    t = words(qwen_text(sid))
    refs = [words(e) for e in es]
    n, average = ter_counts(t, refs)
    more = w["average"].format(k=len(refs), avg=_dec(average, lang), n=n, pct=_pct(n, average, lang))
    body = _note(w["pair"].format(dutch=dutch)) + ter_view(lang, t, _closest(t, refs), w["labels"], more)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the box: a machine translation and a post-edit the learner types ------------------------------------------------------------

_BOX_TEXT = {
    "en": {"title": "Your post-edit, counted",
           "none": "Type a machine translation and a post-edit in the boxes above, each with at least one word.",
           "labels": ("Your post-edit, as the reference:", "The machine translation:")},
    "nl": {"title": "Je post-edit, geteld",
           "none": "Typ een machinevertaling en een post-edit in de vakken hierboven, elk met minstens één woord.",
           "labels": ("Je post-edit, als referentie:", "De machinevertaling:")},
}
BOX_WORDS = 40   # the box counts at most this many words of each text


def box_picture(lang: str, translation: str, post_edit: str) -> str:
    """The box: TER of what the learner typed as machine translation, against what the learner typed as post-edit."""
    lang = lang if lang in LANGS else "en"
    w = _BOX_TEXT[lang]
    t, r = words(translation or "")[:BOX_WORDS], words(post_edit or "")[:BOX_WORDS]
    if not t or not r:
        return yours.figure(w["title"], _note(w["none"]), alt=w["none"])
    return yours.figure(w["title"], ter_view(lang, t, r, w["labels"]), alt=w["title"])


# --- the picture under exercise 1: TER with the learner's shift ------------------------------------------------------------------

# The example of Snover and colleagues (2006): a translation whose block *b c* belongs further to the left
SNOVER = ("a d e b c f", "a b c d e f c")

_SHIFT_TEXT = {
    "en": {"title": "TER with your shift",
           "snover": "The example of Snover and colleagues: *{t}* against *{r}*. Your `shift(words, 3, 2, 1)` gives:",
           "snover_course": "The course's shift gives *{want}*.",
           "corpus": "TER on the {n} sentences, with your shift; lower is better:",
           "head": ["", "Word edits only", "TER with your shift", ""],
           "names": {"direct": "Word for word", "qwen": "Qwen3-0.6B", "people": "People against people"},
           "course": "the course's shift: {want}",
           "note": "Gray: the example, and TER without shifts. Blue: what your shift gives, when it is what the course's shift gives. Purple: what differs from the course's shift.",
           "kind": "a list of the same words"},
    "nl": {"title": "TER met je shift",
           "snover": "Het voorbeeld van Snover en collega's: *{t}* tegen *{r}*. Je `shift(words, 3, 2, 1)` geeft:",
           "snover_course": "De shift van de course geeft *{want}*.",
           "corpus": "TER op de {n} zinnen, met je shift; lager is beter:",
           "head": ["", "Alleen woordbewerkingen", "TER met je shift", ""],
           "names": {"direct": "Woord voor woord", "qwen": "Qwen3-0.6B", "people": "Mensen tegen mensen"},
           "course": "de shift van de course: {want}",
           "note": "Grijs: het voorbeeld, en TER zonder shifts. Blauw: wat je shift geeft, als dat is wat de shift van de course geeft. Paars: wat anders is dan bij de shift van de course.",
           "kind": "een lijst met dezelfde woorden"},
}


def _safe_move(fn):
    """The learner's shift, for TER: a ValueError when it gives something other than a list of the same words."""
    def move(ws, start, length, to):
        out = fn(list(ws), start, length, to)
        if not isinstance(out, list) or sorted(out) != sorted(ws):
            raise ValueError
        return out
    return move


@functools.cache
def _plain_ter(name: str) -> float:
    """TER without shifts: the edit distance to the closest reference, divided by the average reference length."""
    ts, rs = _systems()[name]
    return ter([(min(distance(list(t), list(r)) for r in refs), sum(len(r) for r in refs) / len(refs))
                for t, refs in zip(ts, rs)])


def shift_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SHIFT_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    move = _safe_move(fn)
    t0, r0 = SNOVER[0].split(), SNOVER[1].split()
    data = {name: _systems()[name] for name in SYSTEMS}

    def make():
        try:
            mine = move(t0, 3, 2, 1)
            out = {name: ter([ter_counts(list(t), [list(r) for r in refs], move) for t, refs in zip(ts, rs)])
                   for name, (ts, rs) in data.items()}
        except ValueError:
            return None
        return mine, out

    got = _call(make, SHIFT_SECONDS)
    if got is _SLOW:
        return _empty(w, "slow", SHIFT_SECONDS)
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    mine, out = got
    want = shift(t0, 3, 2, 1)
    right = mine == want
    body = _note(w["snover"].format(t=SNOVER[0], r=SNOVER[1]))
    body += "<div>" + "".join(_chip(x, _CHIP_PRED if right else _CHIP_LOSS) for x in mine) + "</div>"
    if not right:
        body += _note(w["snover_course"].format(want=" ".join(want)))
    num = numbers()
    rows_ = []
    for name in SYSTEMS:
        same = math.isclose(out[name], num[name]["ter"], rel_tol=1e-9, abs_tol=1e-9)
        style, color = (_PRED, "var(--ea-pred)") if same else (_LOSS, "var(--ea-loss)")
        cell = f'<span style="{style}">{_dec(out[name], lang)}</span>'
        if not same:
            cell += _note(w["course"].format(want=_dec(num[name]["ter"], lang)))
        rows_.append([_marked(w["names"][name]), _gray(_dec(_plain_ter(name), lang)), cell, _bar(out[name] / 100, color)])
    body += _label(w["corpus"].format(n=_num(len(_sentences()), lang))) + _table(rows_, w["head"], (1, 2))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the picture under exercise 2: the memory against Qwen, per fuzzy match -----------------------------------------------------

_FUZZY_TEXT = {
    "en": {"title": "Your fuzzy on the translation memory",
           "lead": "For each of the {n} sentences, the course's best match among the {m} sentences of the memory, scored by your fuzzy. TER of the memory's translation, unchanged, and of Qwen's translation; lower is better:",
           "head": ["Fuzzy match", "Sentences", "TER of the memory", "TER of Qwen", "Fewer edits"],
           "bands": ("100%", "85% to 99%", "70% to 84%", "50% to 69%", "below 50%"),
           "memory": "memory", "qwen": "Qwen", "tie": "the same",
           "course": "the course's fuzzy: {want}",
           "note": "Gray: TER of Qwen, which your fuzzy does not change. Blue: what your fuzzy gives, when it is what the course's fuzzy gives. Purple: what differs from the course's fuzzy.",
           "kind": "a number from 0 to 1"},
    "nl": {"title": "Je fuzzy op het vertaalgeheugen",
           "lead": "Voor elk van de {n} zinnen de beste match van de course uit de {m} zinnen van het geheugen, gescoord met je fuzzy. TER van de vertaling uit het geheugen, ongewijzigd, en van de vertaling van Qwen; lager is beter:",
           "head": ["Fuzzy match", "Zinnen", "TER van het geheugen", "TER van Qwen", "Minder bewerkingen"],
           "bands": ("100%", "85% tot 99%", "70% tot 84%", "50% tot 69%", "onder 50%"),
           "memory": "geheugen", "qwen": "Qwen", "tie": "gelijk",
           "course": "de fuzzy van de course: {want}",
           "note": "Grijs: TER van Qwen, die je fuzzy niet verandert. Blauw: wat je fuzzy geeft, als dat is wat de fuzzy van de course geeft. Paars: wat anders is dan bij de fuzzy van de course.",
           "kind": "een getal van 0 tot 1"},
}


def band_table(scores: list) -> list:
    """Per band of BANDS, for the sentences whose match has a fuzzy score in it: (sentences, TER of the memory's
    translation, TER of Qwen's)."""
    memory_, qwen = _match_ter(), _ter_counts("qwen")
    out = []
    for k in range(len(BANDS)):
        mine = [i for i, s in enumerate(scores) if band(s) == k]
        out.append((len(mine), ter([memory_[i] for i in mine]), ter([qwen[i] for i in mine])))
    return out


def fuzzy_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _FUZZY_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    pairs_ = matched()

    def make():
        return [fn(list(a), list(b)) for a, b, _ in pairs_]

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_count(x) and -1e-9 <= float(x) <= 1 + 1e-9 for x in got):
        return _empty(w, "odd")
    mine = band_table([float(x) for x in got])
    want = band_table([fuzzy(list(a), list(b)) for a, b, _ in pairs_])
    rows_ = []
    for k, ((n, tm, qw), (n0, tm0, _)) in enumerate(zip(mine, want)):
        same = n == n0
        style = _PRED if same else _LOSS
        cell = f'<span style="{style}">{_num(n, lang)}</span>'
        if not same:
            cell += _note(w["course"].format(want=_num(n0, lang)))
        better = (w["memory"] if tm < qw else w["qwen"] if qw < tm else w["tie"]) if n else "–"
        rows_.append([html.escape(w["bands"][k]), cell,
                      f'<span style="{style}">{_dec(tm, lang) if n else "–"}</span>',
                      _gray(_dec(qw, lang) if n else "–"), f'<span style="{style}">{html.escape(better)}</span>'])
    body = _note(w["lead"].format(n=_num(len(pairs_), lang), m=_num(len(_memory()), lang)))
    body += _table(rows_, w["head"], (1, 2, 3))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the picture under exercise 3: the learner's best_match on a few sentences ---------------------------------------------------

_MATCH_TEXT = {
    "en": {"title": "Your best_match on five sentences",
           "lead": "Each sentence gets a memory of the {k} sentences of the memory that come closest to it, in the order of the memory:",
           "head": ["New sentence", "Fuzzy", "The translation your best_match gives"],
           "course": "the course's best_match: {want}",
           "none": "nothing",
           "note": "Gray: the new sentences. Blue: what your best_match gives, when it is what the course's best_match gives. Purple: what differs from the course's best_match.",
           "kind": "a pair of a score and a translation"},
    "nl": {"title": "Je best_match op vijf zinnen",
           "lead": "Elke zin krijgt een geheugen van de {k} zinnen van het geheugen die er het dichtst bij liggen, in de volgorde van het geheugen:",
           "head": ["Nieuwe zin", "Fuzzy", "De vertaling die je best_match geeft"],
           "course": "de best_match van de course: {want}",
           "none": "niets",
           "note": "Grijs: de nieuwe zinnen. Blauw: wat je best_match geeft, als dat is wat de best_match van de course geeft. Paars: wat anders is dan bij de best_match van de course.",
           "kind": "een paar van een score en een vertaling"},
}
NEAREST = 20   # how many sentences of the memory each sentence of the picture gets


@functools.cache
def _small_memories() -> tuple:
    """For each sentence of MATCH_ROWS: (its Dutch text, its words, its small memory as (stored words, translation)
    pairs, in the order of the memory)."""
    texts = {sid: text for sid, text, _ in _sentences()}
    out = []
    for sid in MATCH_ROWS:
        new = dutch_words(texts[sid])
        out.append((texts[sid], tuple(new), tuple((tuple(s), t) for s, t in near(new))))
    return tuple(out)


def match_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _MATCH_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    rows = _small_memories()

    def make():
        return [fn(list(new), [(list(s), t) for s, t in memory_]) for _, new, memory_ in rows]

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(x, tuple) and len(x) == 2 and _is_count(x[0]) and (x[1] is None or isinstance(x[1], str))
               for x in got):
        return _empty(w, "odd")
    rows_ = []
    for (text, new, memory_), (score, translation) in zip(rows, got):
        want = best_match(list(new), [(list(s), t) for s, t in memory_])
        same = math.isclose(float(score), want[0], abs_tol=1e-9) and translation == want[1]
        style = _PRED if same else _LOSS
        cell = f'<span style="{style}">{html.escape(translation if translation is not None else w["none"])}</span>'
        if not same:
            cell += _note(w["course"].format(want=f"{_pct(want[0], 1, lang)}, {want[1]}"))
        rows_.append([_gray(text), f'<span style="{style}">{_pct(float(score), 1, lang)}</span>', cell])
    body = _note(w["lead"].format(k=NEAREST)) + _table(rows_, w["head"], (1,))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.dutch_words(...)`; the function gets a
    copy. With `make`, the value is made the first time it is needed, so that opening the lesson stays fast."""

    def __init__(self, name: str, value=None, make=None):
        self.name, self._value, self._make = name, value, make

    @property
    def value(self):
        if self._make is not None:
            self._value, self._make = self._make(), None
        return self._value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (float, np.floating)):
        return f"{round(float(v), 4)}"
    if isinstance(v, np.integer):
        return repr(int(v))
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _show(v):
    """A value for a message, the way the lesson writes it; a number with at most four decimals."""
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _dutch_val(text_: str) -> _Val:
    return _Val(f"C.dutch_words({_value(text_)})", dutch_words(text_))


def _words_val(text_: str) -> _Val:
    return _Val(f"C.words({_value(text_)})", words(text_))


def near(new: list, k: int = NEAREST) -> list:
    """A small translation memory for the Dutch words `new`: the k sentences of the memory that come closest to it, as
    (Dutch words, English translation) pairs, in the order of the memory."""
    mem = _memory()
    return [(dutch_words(mem[i][1]), mem[i][2]) for i in sorted(i for _, i in search(list(new), k))]


def _near_val(text_: str) -> _Val:
    return _Val(f"C.near(C.dutch_words({_value(text_)}))", make=lambda: near(dutch_words(text_)))


_DOTS = Mistake("dots-left", matches=lambda got, text_: any(line.split("#", 1)[0].count("...")
                                                            for line in text_.splitlines()),
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show.
TOY_SHIFT = ((["a", "d", "e", "b", "c", "f"], 3, 2, 1), (["can", "i", "this", "eat"], 2, 1, 3))
TOY_FUZZY = ((["ik", "heb", "geen", "tijd", "om", "te", "lezen"], ["ik", "heb", "geen", "tijd", "om", "te", "schrijven"]),
             (["ik", "weet", "niet", "wat", "dat", "is"], ["ik", "weet", "niet", "zeker", "wat", "dat", "is"]))
TOY_MEMORY = ((["ik", "ben", "moe"], [(["ik", "ben", "blij"], "I'm happy."), (["ik", "ben", "erg", "moe"], "I'm very tired."),
                                     (["jij", "bent", "moe"], "You're tired.")]),
              (["ik", "heb", "geen", "tijd", "om", "te", "lezen"],
               [(["ik", "heb", "geen", "tijd"], "I have no time."),
                (["ik", "heb", "geen", "tijd", "om", "te", "schrijven"], "I don't have time to write.")]))

_SH_HEAD = "def shift(words, start, length, to):\n"
_SH_BLOCK = "    block = words[start:start + length]\n"
_SH_REST = "    rest = words[:start] + words[start + length:]\n"
_SH_END = "    return rest[:to] + block + rest[to:]\n"


def _sh(block: str = _SH_BLOCK, rest: str = _SH_REST, end: str = _SH_END) -> str:
    return _SH_HEAD + block + rest + end


_SH = _sh()
_SH_START = _sh("    block = ...\n", "    rest = ...\n")
_SH_TO_LENGTH = _sh(block="    block = words[start:length]\n")
_SH_ONE = _sh(block="    block = words[start]\n")
_SH_KEEP = _sh(rest="    rest = words\n")
_SH_AFTER_LENGTH = _sh(rest="    rest = words[:start] + words[length:]\n")

_FZ_HEAD = "def fuzzy(new, stored):\n"
_FZ_EDITS = "    edits = distance(new, stored)\n"
_FZ_LONGEST = "    longest = max(len(new), len(stored))\n"
_FZ_END = "    return 1 - edits / longest\n"


def _fz(edits: str = _FZ_EDITS, longest: str = _FZ_LONGEST, end: str = _FZ_END) -> str:
    return _FZ_HEAD + edits + longest + end


_FZ = _fz()
_FZ_START = _fz("    edits = ...\n", "    longest = ...\n")
_FZ_NEW = _fz(longest="    longest = len(new)\n")
_FZ_STORED = _fz(longest="    longest = len(stored)\n")
_FZ_MIN = _fz(longest="    longest = min(len(new), len(stored))\n")
_FZ_BOTH = _fz(longest="    longest = len(new) + len(stored)\n")

_BM_HEAD = ("def best_match(new, memory):\n    best = None\n    best_score = 0\n"
            "    for stored, translation in memory:\n        score = fuzzy(new, stored)\n")
_BM_IF = "        if score > best_score:\n"
_BM_BEST = "            best = translation\n"


def _bm(check: str = _BM_IF, best: str = _BM_BEST, end: str = "    return best_score, best\n") -> str:
    return _BM_HEAD + check + "            best_score = score\n" + best + end


_BM = _bm()
_BM_START = _bm("        if ...:\n", "            best = ...\n")
_BM_GE = _bm("        if score >= best_score:\n")
_BM_LESS = _bm("        if score < best_score:\n")
_BM_NONE = _bm("        if score > best:\n")
_BM_STORED = _bm(best="            best = stored\n")
_BM_SCORE = _bm(best="            best = score\n")


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _not_number(got, text_) -> bool:
    return not _is_count(got)


def _not_pair(got, text_) -> bool:
    return not (isinstance(got, tuple) and len(got) == 2 and _is_count(got[0]))


def _none_compared(e) -> bool:
    return isinstance(e, TypeError) and "NoneType" in str(e)


def _concat(e) -> bool:
    return isinstance(e, TypeError) and "concatenate" in str(e)


REQUIRED = ["shift", "fuzzy", "best_match", "targeted"]

EXERCISES = {
    "shift": FunctionBox(
        name="shift",
        picture=shift_picture,
        function="shift",
        start=_SH_START,
        solution=_SH,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_SHIFT],
        hidden=[(_written(["a", "b", "c"]), _written(0), _written(1), _written(2)),
                (_written(["a", "b", "c"]), _written(2), _written(1), _written(0)),
                (_written(["a", "b", "c", "d"]), _written(1), _written(2), _written(1)),
                (_written(["a", "b", "c", "d", "e"]), _written(0), _written(3), _written(2)),
                (_written(["x"]), _written(0), _written(1), _written(0)),
                (_written(["i", "will", "the", "next", "bus", "take"]), _written(2), _written(3), _written(3)),
                (_words_val("My father has a new car bought."), _written(6), _written(1), _written(3)),
                (_words_val("What would you in my place do?"), _written(3), _written(3), _written(4))],
        first=[_DOTS],
        mistakes=[
            Mistake("concat", error=_concat,
                    en="`{call}` stops with a `TypeError`: `+` can only put a list after a list, and one of your names holds a single word. `words[3]` is the word at place 3; which slice gives the block of `length` words as a list?",
                    nl="`{call}` stopt met een `TypeError`: `+` kan alleen een lijst achter een lijst zetten, en een van je namen bevat één woord. `words[3]` is het woord op plaats 3; welke slice geeft het blok van `length` woorden als lijst?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of words. Does your function still give back `rest[:to] + block + rest[to:]`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met woorden. Geeft je functie nog steeds `rest[:to] + block + rest[to:]` terug?"),
            Mistake("to-length", wrong=_SH_TO_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: *b* and *c* are gone. A slice stops at the place after the colon, and `length` is a number of words, 2, not a place: `words[3:2]` is empty. At which place does a block of `length` words that starts at `start` stop?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *b* en *c* zijn weg. Een slice stopt op de plaats na de dubbele punt, en `length` is een aantal woorden, 2, geen plaats: `words[3:2]` is leeg. Op welke plaats stopt een blok van `length` woorden dat op `start` begint?"),
            Mistake("keep", wrong=_SH_KEEP,
                    en="`{call}` gives {got}, and it should give {want}: *b c* is in it twice, as `rest` still holds the block. `rest` is the words before the block and the words after it: which slices are those?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: *b c* staat er twee keer in, want `rest` bevat het blok nog. `rest` is de woorden voor het blok en de woorden erna: welke slices zijn dat?"),
            Mistake("after-length", wrong=_SH_AFTER_LENGTH,
                    en="`{call}` gives {got}, and it should give {want}: the second slice of `rest` starts at place `length`, 2, and `length` is a number of words, not a place. At which place do the words after the block start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de tweede slice van `rest` begint op plaats `length`, 2, en `length` is een aantal woorden, geen plaats. Op welke plaats beginnen de woorden na het blok?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `block` is the `length` words that start at place `start`, and `rest` the list without that block.",
                      "nl": "Vervang de twee `...`: `block` is de `length` woorden die op plaats `start` beginnen, en `rest` de lijst zonder dat blok."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `block` exactly the `length` words from place `start` on, and does `rest` hold every other word once, in its order?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `block` precies de `length` woorden vanaf plaats `start`, en bevat `rest` elk ander woord één keer, in de volgorde van de lijst?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a block at the start or at the end of the list, and for a block of more than one word?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie voor een blok aan het begin of aan het eind van de lijst, en voor een blok van meer dan één woord?"},
            "correct": {"en": "Right: your function moves a block of words in one step, as TER does. How much lower is TER than the word edits alone, in the picture below, and for which way to translate do the shifts help most?",
                        "nl": "Goed: je functie verplaatst een blok woorden in één stap, zoals TER dat doet. Hoeveel lager is TER dan alleen de woordbewerkingen, in het plaatje hieronder, en bij welke manier van vertalen helpen de shifts het meest?"},
        },
    ),
    "fuzzy": FunctionBox(
        name="fuzzy",
        picture=fuzzy_picture,
        function="fuzzy",
        start=_FZ_START,
        solution=_FZ,
        show=_show,
        names={"distance": distance},
        visible=[tuple(_written(x) for x in case) for case in TOY_FUZZY],
        hidden=[(_written(["ja"]), _written(["ja"])),
                (_written(["ik", "ga"]), _written(["jij", "komt"])),
                (_written(["kom", "binnen"]), _written(["kom", "binnen", "alstublieft"])),
                (_written(["morgen", "is", "het", "kerstmis"]), _written(["morgen", "het", "is", "kerstmis"])),
                (_dutch_val("Ik weet niet meer wat ik moet doen."), _dutch_val("Ik weet niet wat ik moet doen.")),
                (_dutch_val("Ik wist niet dat je zou komen."), _dutch_val("Ik wist wel dat je zou komen.")),
                (_dutch_val("Als het nodig is, kom ik snel."), _dutch_val("Als het mogelijk is, zou ik een uitleg willen."))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: a number from 0 to 1. Does your function still give back `1 - edits / longest`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een getal van 0 tot 1. Geeft je functie nog steeds `1 - edits / longest` terug?"),
            Mistake("new-only", wrong=_FZ_NEW,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the length of `new`, 6 words, while `stored` has 7. A stored sentence much longer than the new one would get a score below 0: which length keeps the score from 0 to 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door de lengte van `new`, 6 woorden, terwijl `stored` er 7 heeft. Een opgeslagen zin die veel langer is dan de nieuwe zou een score onder 0 krijgen: welke lengte houdt de score tussen 0 en 1?"),
            Mistake("stored-only", wrong=_FZ_STORED,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the length of `stored`, 7 words, while `new` has 8. A new sentence much longer than the stored one would get a score below 0: which length keeps the score from 0 to 1?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door de lengte van `stored`, 7 woorden, terwijl `new` er 8 heeft. Een nieuwe zin die veel langer is dan de opgeslagen zou een score onder 0 krijgen: welke lengte houdt de score tussen 0 en 1?"),
            Mistake("min", wrong=_FZ_MIN,
                    en="`{call}` gives {got}, and it should give {want}: `min` gives the shorter length, 6, so the score can drop below 0 for sentences of very different lengths. Which function gives the longer one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `min` geeft de kortste lengte, 6, dus de score kan onder 0 zakken bij zinnen die erg verschillen in lengte. Welke functie geeft de langste?"),
            Mistake("both", wrong=_FZ_BOTH,
                    en="`{call}` gives {got}, and it should give {want}: your function divides by the two lengths added up, so two sentences without a single word in common still score 0.5. The most edits two sentences can need is the length of the longer one: which number is that?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie deelt door de twee lengtes bij elkaar, dus twee zinnen zonder één gemeenschappelijk woord scoren nog 0,5. Twee zinnen hebben hoogstens zoveel bewerkingen nodig als de langste lang is: welk getal is dat?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `edits` is the edit distance of `new` and `stored`, and `longest` the length of the longer of the two.",
                      "nl": "Vervang de twee `...`: `edits` is de edit distance van `new` en `stored`, en `longest` de lengte van de langste van de twee."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Is `edits` the edit distance of the two lists of words, and `longest` the length of the longer list?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Is `edits` de edit distance van de twee lijsten woorden, en `longest` de lengte van de langste lijst?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function divide by the longer length, also when `new` is the longer list?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Deelt je functie door de langste lengte, ook als `new` de langste lijst is?"},
            "correct": {"en": "Right: your function scores how close a stored sentence is. In the picture below, from which fuzzy match on does the memory's translation need fewer edits than Qwen's?",
                        "nl": "Goed: je functie geeft een score voor hoe dicht een opgeslagen zin bij de nieuwe ligt. Vanaf welke fuzzy match heeft de vertaling uit het geheugen in het plaatje hieronder minder bewerkingen nodig dan de vertaling van Qwen?"},
        },
    ),
    "best_match": FunctionBox(
        name="best_match",
        picture=match_picture,
        function="best_match",
        start=_BM_START,
        solution=_BM,
        show=_show,
        names={"fuzzy": fuzzy},
        visible=[tuple(_written(x) for x in case) for case in TOY_MEMORY],
        hidden=[(_written(["ik", "ben", "moe"]), _written([(["ik", "ben", "blij"], "I'm happy."),
                                                          (["jij", "bent", "moe"], "You're tired.")])),
                (_written(["ja"]), _written([])),
                (_written(["ja"]), _written([(["nee"], "No.")])),
                (_written(["wees", "niet", "bang"]), _written([(["wees", "niet", "bang"], "Don't be afraid!"),
                                                               (["wees", "niet", "boos"], "Don't be angry.")])),
                (_written(["kom", "binnen"]), _written([(["ga", "weg"], "Go away."), (["kom", "hier"], "Come here."),
                                                        (["kom", "binnen", "alstublieft"], "Please come in.")])),
                (_dutch_val("Vergeet ons niet!"), _near_val("Vergeet ons niet!")),
                (_dutch_val("Ik wist niet dat je zou komen."), _near_val("Ik wist niet dat je zou komen."))],
        first=[_DOTS],
        mistakes=[
            Mistake("none-compared", error=_none_compared,
                    en="`{call}` stops with a `TypeError`: your function compares a number with `None`. `best` holds the translation, and starts as `None`; which name holds the best score so far?",
                    nl="`{call}` stopt met een `TypeError`: je functie vergelijkt een getal met `None`. `best` bevat de vertaling, en begint als `None`; welke naam bevat de beste score tot nu toe?"),
            Mistake("not-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give {want}: a score and a translation. Does your function still give back `best_score, best`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een score en een vertaling. Geeft je functie nog steeds `best_score, best` terug?"),
            Mistake("ge", wrong=_BM_GE,
                    en="`{call}` gives {got}, and it should give {want}: both stored sentences score 0.6667, and your function takes the last of the two. Of equal matches, the memory's first counts: which comparison only takes a score that is higher than the best so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: beide opgeslagen zinnen scoren 0,6667, en je functie neemt de laatste van de twee. Bij gelijke matches telt de eerste van het geheugen: welke vergelijking neemt alleen een score die hoger is dan de beste tot nu toe?"),
            Mistake("less", wrong=_BM_LESS,
                    en="`{call}` gives {got}, and it should give {want}: your `if` never holds, as a fuzzy score is never below 0, where `best_score` starts. When should a stored sentence take the place of the best so far?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `if` klopt nooit, want een fuzzy score is nooit lager dan 0, waar `best_score` begint. Wanneer moet een opgeslagen zin de plaats van de beste tot nu toe innemen?"),
            Mistake("stored", wrong=_BM_STORED,
                    en="`{call}` gives {got}, and it should give {want}: your function gives back the Dutch words of the match, `stored`. What should a translation memory give a translator?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft de Nederlandse woorden van de match terug, `stored`. Wat moet een vertaalgeheugen een vertaler geven?"),
            Mistake("score", wrong=_BM_SCORE,
                    en="`{call}` gives {got}, and it should give {want}: your function gives the score twice, and `best` should hold the translation of the best match. Which name of the loop holds it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie geeft twee keer de score, en `best` moet de vertaling van de beste match bevatten. Welke naam van de loop bevat de vertaling?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether `score` beats the best score so far, and the line under it keeps the translation of that stored sentence in `best`.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of `score` de beste score tot nu toe verslaat, en de regel eronder bewaart de vertaling van die opgeslagen zin in `best`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does your function keep the highest score, the translation that goes with it, and of equal scores the first?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bewaart je functie de hoogste score, de vertaling die erbij hoort, en bij gelijke scores de eerste?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give `(0, None)` for a memory without a match, and the first of two equal matches?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `(0, None)` bij een geheugen zonder match, en de eerste van twee gelijke matches?"},
            "correct": {"en": "Right: your function finds the best match in a memory. Which of the translations in the picture below can a translator use as it is, and which says something else?",
                        "nl": "Goed: je functie vindt de beste match in een geheugen. Welke vertalingen in het plaatje hieronder kan een vertaler zo gebruiken, en welke zegt iets anders?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("shift", _SH_START, "empty", None),
    ("shift", _SH, "correct", None),
    ("shift", _sh(rest="    rest = ...\n"), "wrong", "dots-left"),
    ("shift", _SH_ONE, "error", "concat"),
    ("shift", _sh(end="    return \" \".join(rest[:to] + block + rest[to:])\n"), "wrong", "not-list"),
    ("shift", _SH_TO_LENGTH, "wrong", "to-length"),
    ("shift", _SH_KEEP, "wrong", "keep"),
    ("shift", _SH_AFTER_LENGTH, "wrong", "after-length"),
    ("shift", _sh(block="    block = words[start:length + start]\n"), "correct", None),
    ("shift", "def shift(words, start, length, to):\n    block = words[start:start + length]\n"
     "    rest = words[:start] + words[start + length:]\n    rest[to:to] = block\n    return rest\n", "correct", None),
    ("shift", "def shift(words, start, length, to):\n    block = words[start:start + length]\n"
     "    del words[start:start + length]\n    return words[:to] + block + words[to:]\n", "correct", None),
    ("fuzzy", _FZ_START, "empty", None),
    ("fuzzy", _FZ, "correct", None),
    ("fuzzy", _fz(longest="    longest = ...\n"), "wrong", "dots-left"),
    ("fuzzy", _fz(end="    return str(1 - edits / longest)\n"), "wrong", "not-number"),
    ("fuzzy", _FZ_NEW, "wrong", "new-only"),
    ("fuzzy", _FZ_STORED, "wrong", "stored-only"),
    ("fuzzy", _FZ_MIN, "wrong", "min"),
    ("fuzzy", _FZ_BOTH, "wrong", "both"),
    ("fuzzy", _fz(longest="    longest = max(len(stored), len(new))\n"), "correct", None),
    ("fuzzy", _fz(longest="    if len(new) > len(stored):\n        longest = len(new)\n    else:\n"
                          "        longest = len(stored)\n"), "correct", None),
    ("best_match", _BM_START, "empty", None),
    ("best_match", _BM, "correct", None),
    ("best_match", _bm(best="            best = ...\n"), "wrong", "dots-left"),
    ("best_match", _BM_NONE, "error", "none-compared"),
    ("best_match", _bm(end="    return best\n"), "wrong", "not-pair"),
    ("best_match", _BM_GE, "wrong", "ge"),
    ("best_match", _BM_LESS, "wrong", "less"),
    ("best_match", _BM_STORED, "wrong", "stored"),
    ("best_match", _BM_SCORE, "wrong", "score"),
    ("best_match", _bm("        if best_score < score:\n"), "correct", None),
]


# --- the numbers of the text ---------------------------------------------------------------------------------------------

@functools.cache
def _best_paths(name: str) -> tuple:
    """For each sentence: (shifts, edit distance left) to the reference that needs the fewest TER edits, the first
    of equals."""
    ts, rs = _systems()[name]
    out = []
    for t, refs in zip(ts, rs):
        best = None
        for r in refs:
            moves, edits = ter_path(list(t), list(r))
            if best is None or len(moves) + edits < len(best[0]) + best[1]:
                best = (moves, edits)
        out.append(best)
    return tuple(out)


@functools.cache
def _hter() -> tuple:
    """For each of the 40 post-edits: (HTER counts, TER counts). HTER counts the edits to the post-edit, and TER those
    to the translation by people that needs the fewest; both divide by the average length of the translations by
    people, as Snover and colleagues did."""
    es = {sid: es for sid, _, es in _all_sentences()}
    out = []
    for sid, q, p in _post_edits():
        refs = [words(e) for e in es[sid]]
        edits, average = ter_counts(words(q), refs)
        out.append(((ter_edits(words(q), words(p)), average), (edits, average)))
    return tuple(out)


@functools.cache
def numbers() -> dict:
    """Every number the text and the pictures name."""
    out = {"n": len(_sentences()), "all": len(_all_sentences()), "memory": len(_memory())}
    out["dropped"] = out["all"] - out["n"]
    for name in SYSTEMS:
        counts, paths_ = _ter_counts(name), _best_paths(name)
        out[name] = {"ter": ter(counts), "plain": _plain_ter(name), "exact": sum(e == 0 for e, _ in counts),
                     "edits": sum(e for e, _ in counts), "words": sum(w for _, w in counts),
                     "shifted": sum(1 for m, _ in paths_ if m), "shifts": sum(len(m) for m, _ in paths_)}
    q = _qwen()["translations"]
    out["qwen"]["cut"] = sum(q[str(sid)]["finish"] != "stop" for sid, _, _ in _sentences())
    out["refs"] = (min(len(es) for _, _, es in _sentences()), max(len(es) for _, _, es in _sentences()))
    scores = [fuzzy(list(a), list(b)) for a, b, _ in matched()]
    out["bands"] = band_table(scores)
    out["full"] = sum(s == 1 for s in scores)
    direct, qwen = _ter_counts("direct"), _ter_counts("qwen")
    out["wins"] = {"qwen": sum(b[0] < a[0] for a, b in zip(direct, qwen)),
                   "direct": sum(a[0] < b[0] for a, b in zip(direct, qwen)),
                   "same": sum(a[0] == b[0] for a, b in zip(direct, qwen))}
    lengths = [len(dutch_words(text_)) for _, text_, _ in _sentences()]
    for key, keep in (("short", lambda k: k <= SHORT), ("long", lambda k: k > SHORT)):
        mine = [i for i, k in enumerate(lengths) if keep(k)]
        out[key] = {"n": len(mine), "direct": ter([direct[i] for i in mine]), "qwen": ter([qwen[i] for i in mine])}
    texts = [(qwen_text(sid).strip(), text_.strip()) for sid, text_, _ in _sentences()]
    out["copied"] = sum(t == d for t, d in texts)
    out["copied_right"] = sum(t == d and e == 0 for (t, d), (e, _) in zip(texts, qwen))
    lines = [line + "." for line in PROMPT.split("\n")[0].rstrip(".").split(". ")]
    out["prompt"] = sum(t in lines for t, _ in texts)
    pairs_ = _hter()
    out["hter"] = {"hter": ter([h for h, _ in pairs_]), "ter": ter([t for _, t in pairs_]),
                   "unchanged": sum(h[0] == 0 for h, _ in pairs_), "n": len(pairs_)}
    return out
