# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-bleu: counting the n-grams of a translation that a translation by people has too, each at
most as often as there (clipped); BLEU from those counts, with the brevity penalty (bleu); and the character n-grams
of a text, for chrF (char_ngrams).

The translations are those of nlp-classic-mt: the 21,636 Dutch sentences of Tatoeba (https://tatoeba.org, CC BY 2.0
FR) whose every word is in the course's dictionary, translated word for word into English, without and with the rule
that moves a verb, and scored against every English translation of them by people (data/tatoeba-nl.tsv.gz and
data/tatoeba-nl-en.tsv.gz, the export of November 21, 2026). Of the 3,707 sentences with more than one translation by
people, the first translation is also scored against the others: people against people. The Dutch pairs with a
compound are two translations by people of one English sentence of Tatoeba.

The lesson scores words in small letters without punctuation, with contractions written out, as nlp-classic-mt does;
chrF reads those words with the spaces left out. BLEU and chrF follow sacreBLEU 2.6 (Post 2018): BLEU with four
n-gram lengths, clipped by the largest count in any one reference, the reference length closest to the translation's,
without smoothing; chrF2 with character n-grams of length 1 to 6 and the best reference per sentence. On the same
words, sacreBLEU gives the same scores as the course (tokenize="none" for BLEU), which the course checked when it made
the lesson. Every number in the text and the pictures is computed by the course (tests/test_nlp_bleu.py computes them
again)."""
import functools
import html
import math

import numpy as np

from everything_ai import books, paths, yours
from everything_ai.check import FunctionBox, Mistake

ORDER = 4          # BLEU's n-gram lengths: 1 to 4
CHAR_ORDER = 6     # chrF's character n-gram lengths: 1 to 6
BETA = 2           # chrF2: recall counts twice as much as precision
LANGS = ("en", "nl")

# Two translations by people of one English sentence of Tatoeba, where one writes a compound and the other does not:
# (Dutch id of the translation by people the lesson uses as reference, of the other one), with the English sentence.
COMPOUNDS = ((10355365, 10355366),   # How many kinds of insect are there?
             (7911763, 7911459),     # Tom is a law student.
             (11152545, 11152546),   # Everything here is handmade.
             (11825578, 11825579))   # Today I had a math test.


@functools.cache
def _mt():
    """The checks of nlp-classic-mt: its sentences, its dictionary and its translations word for word."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_classic_mt_for_bleu",
                                                  paths.LESSONS / "nlp-classic-mt" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(text: str) -> list:
    """The words of an English or Dutch text, in small letters, without punctuation, with English contractions such as
    *I'll* written out, as nlp-classic-mt reads them."""
    return _mt().english_words(text)


def ngram_counts(ws: list, n: int) -> dict:
    """How often each n-gram of a list of words occurs: the words of an n-gram joined with a space, such as "the cat"."""
    counts = {}
    for i in range(len(ws) - n + 1):
        gram = " ".join(ws[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


# --- the course's solutions of the exercises ---------------------------------------------------------------------------

def clipped(mine, theirs):
    matched = 0
    for gram in mine:
        if gram in theirs:
            matched += min(mine[gram], theirs[gram])
    return matched


def bleu(matched, total, c, r):
    if 0 in matched:
        return 0.0
    logs = [np.log(matched[n] / total[n]) for n in range(len(matched))]
    mean = np.mean(logs)
    if c > r:
        penalty = 1.0
    else:
        penalty = np.exp(1 - r / c)
    return float(penalty * np.exp(mean))


def char_ngrams(letters, n):
    counts = {}
    for i in range(len(letters) - n + 1):
        gram = letters[i:i + n]
        if gram in counts:
            counts[gram] += 1
        else:
            counts[gram] = 1
    return counts


# --- BLEU and chrF of one translation, and of many ----------------------------------------------------------------------

def _most(refs: list, n: int) -> dict:
    """Per n-gram, the largest count in any one reference: what a clipped count may reach."""
    most = {}
    for r in refs:
        for gram, k in ngram_counts(r, n).items():
            most[gram] = max(most.get(gram, 0), k)
    return most


def bleu_counts(translation: list, references: list, clip=clipped) -> tuple:
    """For one translation and its references, all lists of words: (matched, total, c, r), with matched and total a list
    per n-gram length, c the length of the translation and r that of the reference closest to it in length (the
    shorter one when two are as close)."""
    matched, total = [], []
    for n in range(1, ORDER + 1):
        matched.append(clip(ngram_counts(translation, n), _most(references, n)))
        total.append(max(len(translation) - n + 1, 0))
    c = len(translation)
    r = min((abs(len(x) - c), len(x)) for x in references)[1]
    return matched, total, c, r


def add_up(counts: list) -> tuple:
    """The counts of many sentences, added up per n-gram length: what corpus BLEU is computed from."""
    matched, total, c, r = [0] * ORDER, [0] * ORDER, 0, 0
    for m, t, cc, rr in counts:
        matched = [a + b for a, b in zip(matched, m)]
        total = [a + b for a, b in zip(total, t)]
        c, r = c + cc, r + rr
    return matched, total, c, r


def letters(ws: list) -> str:
    """A list of words as chrF reads it: one text, with the spaces left out."""
    return "".join(ws)


def chrf_counts(translation: list, reference: list, grams=char_ngrams, clip=clipped) -> list:
    """For each character n-gram length, (in the translation, in the reference, matched), as sacreBLEU counts them: a
    length the reference is too short for counts nothing in the translation either."""
    a, b = letters(translation), letters(reference)
    out = []
    for n in range(1, CHAR_ORDER + 1):
        mine, theirs = grams(a, n), grams(b, n)
        in_theirs = sum(theirs.values())
        out.append((sum(mine.values()) if in_theirs else 0, in_theirs, clip(mine, theirs)))
    return out


def f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """The F-score that counts recall beta times as much as precision; 0 when both are 0."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def chrf(counts: list) -> float:
    """chrF from the counts per length: precision and recall averaged over the lengths that both texts have, then
    f_beta."""
    ps = [m / h for h, r, m in counts if h > 0 and r > 0]
    rs = [m / r for h, r, m in counts if h > 0 and r > 0]
    if not ps:
        return 0.0
    return f_beta(sum(ps) / len(ps), sum(rs) / len(rs))


def best_chrf_counts(translation: list, references: list, grams=char_ngrams, clip=clipped) -> list:
    """The chrF counts against the reference that gives the highest chrF; the first one when two give as much."""
    best, out = -1.0, None
    for ref in references:
        counts = chrf_counts(translation, ref, grams, clip)
        score = chrf(counts)
        if score > best:
            best, out = score, counts
    return out


def add_chrf(counts: list) -> list:
    """The chrF counts of many sentences, added up per length: what corpus chrF is computed from."""
    out = [[0, 0, 0] for _ in range(CHAR_ORDER)]
    for one in counts:
        for n, row in enumerate(one):
            for k in range(3):
                out[n][k] += row[k]
    return [tuple(x) for x in out]


# --- the translations the lesson scores ---------------------------------------------------------------------------------

@functools.cache
def _systems() -> dict:
    """The translations the lesson scores, each as (translations, references), all lists of words: word for word and
    with the rule of nlp-classic-mt on the 21,636 sentences, and the first translation by people against the others on
    the 3,707 sentences that have more than one."""
    mt = _mt()
    refs, direct, ruled = [], [], []
    for ws, rs in mt._covered_words():
        refs.append(tuple(tuple(r) for r in rs))
        direct.append(tuple(mt.DICTIONARY[w] for w in ws))
        ruled.append(tuple(mt.DICTIONARY[w] for w in mt.verb_after_helper(list(ws), mt.HELPERS, mt.VERBS)))
    several = [r for r in refs if len(r) >= 2]
    return {"direct": (tuple(direct), tuple(refs)), "ruled": (tuple(ruled), tuple(refs)),
            "people": (tuple(r[0] for r in several), tuple(r[1:] for r in several))}


SYSTEMS = ("direct", "ruled", "people")


def system(name: str) -> tuple:
    """(translations, references) of one of SYSTEMS, as new lists: a lesson that changes them changes nothing for the
    next."""
    ts, rs = _systems()[name]
    return [list(t) for t in ts], [[list(r) for r in refs] for refs in rs]


@functools.cache
def _bleu_counts(name: str) -> tuple:
    ts, rs = _systems()[name]
    return tuple(bleu_counts(list(t), [list(r) for r in refs]) for t, refs in zip(ts, rs))


def counts(name: str) -> tuple:
    """(matched, total, c, r) of one of SYSTEMS, added up over all its sentences, as new lists."""
    matched, total, c, r = add_up(_bleu_counts(name))
    return list(matched), list(total), c, r


@functools.cache
def _chrf_counts(name: str) -> tuple:
    ts, rs = _systems()[name]
    return tuple(tuple(best_chrf_counts(list(t), [list(r) for r in refs])) for t, refs in zip(ts, rs))


@functools.cache
def compounds() -> tuple:
    """The Dutch pairs with a compound: (reference, translation) as texts, and as lists of words."""
    nl = dict(books.dutch_sentences())
    return tuple((nl[a], nl[b], tuple(_mt().words(nl[a])), tuple(_mt().words(nl[b]))) for a, b in COMPOUNDS)


@functools.cache
def numbers() -> dict:
    """Every number of the text."""
    out = {}
    for name in SYSTEMS:
        matched, total, c, r = counts(name)
        out[name] = {"bleu": 100 * bleu(matched, total, c, r), "chrf": 100 * chrf(add_chrf(_chrf_counts(name))),
                     "matched": matched, "total": total, "c": c, "r": r, "n": len(_systems()[name][0]),
                     "p": [100 * m / t for m, t in zip(matched, total)],
                     "zero": sum(1 for k in _bleu_counts(name) if bleu(*k) == 0)}
    ts, rs = _systems()["people"]
    out["short"] = sum(1 for t in ts if len(t) < ORDER)
    out["exact"] = sum(1 for t, refs in zip(ts, rs) if t in refs)
    out["compounds"] = [(100 * bleu(*bleu_counts(list(b), [list(a)])), 100 * chrf(chrf_counts(list(b), list(a))))
                        for _, _, a, b in compounds()]
    return out




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


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text_ = w[key].format(s=PICTURE_SECONDS, kind=w.get("kind", ""))
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


# --- one translation against one reference: what the idea, the box and the text show -------------------------------------

def _counted(translation: list, reference: list, n: int) -> list:
    """The n-grams of the translation in order, each with whether it counts: an n-gram counts as often as the reference
    has it, from the left."""
    left = ngram_counts(reference, n)
    out = []
    for i in range(len(translation) - n + 1):
        gram = " ".join(translation[i:i + n])
        ok = left.get(gram, 0) > 0
        if ok:
            left[gram] -= 1
        out.append((gram, ok))
    return out


_PAIR_TEXT = {
    "en": {"reference": "The reference, in words:",
           "words": "BLEU: the word n-grams of the translation",
           "head": ["n", "The n-grams of the translation", "Counted"],
           "short": "The translation has fewer than {n} words, so it has no group of {n}.",
           "zero": "BLEU is 0: no group of {n} words of the translation is in the reference, and BLEU multiplies the four precisions.",
           "penalty_no": "The translation has {c} words and the reference {r}: not shorter, so the brevity penalty is 1.",
           "penalty": "The translation has {c} words and the reference {r}: shorter, so the brevity penalty is {bp}.",
           "bleu": "BLEU: the geometric mean of the four precisions is {gm}; times the brevity penalty that is {score}, written times 100 as {big}.",
           "letters": "chrF: the character n-grams of the words, with the spaces left out",
           "letters_head": ["", "The letters"],
           "letters_rows": ("Reference", "Translation"),
           "chr_head": ["n", "In the translation", "In the reference", "In both", "Precision", "Recall"],
           "chrf": "Averaged over the six lengths, precision is {p} and recall {r}. chrF counts recall twice as much as precision: {score}.",
           "chr_none": "The texts are too short for character n-grams, so chrF is 0.",
           "note": "Gray: the reference. Blue: an n-gram of the translation that the reference has too, counted at most as often as the reference has it. Purple: an n-gram of the translation that the reference does not have, or has fewer times."},
    "nl": {"reference": "De referentie, in woorden:",
           "words": "BLEU: de n-grams van woorden van de vertaling",
           "head": ["n", "De n-grams van de vertaling", "Geteld"],
           "short": "De vertaling heeft minder dan {n} woorden, dus geen groepje van {n}.",
           "zero": "BLEU is 0: geen enkel groepje van {n} woorden van de vertaling staat in de referentie, en BLEU vermenigvuldigt de vier precisions.",
           "penalty_no": "De vertaling heeft {c} woorden en de referentie {r}: niet korter, dus de brevity penalty is 1.",
           "penalty": "De vertaling heeft {c} woorden en de referentie {r}: korter, dus de brevity penalty is {bp}.",
           "bleu": "BLEU: het meetkundig gemiddelde van de vier precisions is {gm}; keer de brevity penalty is dat {score}, keer 100 geschreven als {big}.",
           "letters": "chrF: de n-grams van tekens van de woorden, zonder de spaties",
           "letters_head": ["", "De letters"],
           "letters_rows": ("Referentie", "Vertaling"),
           "chr_head": ["n", "In de vertaling", "In de referentie", "In beide", "Precision", "Recall"],
           "chrf": "Gemiddeld over de zes lengtes is precision {p} en recall {r}. chrF telt recall twee keer zo zwaar als precision: {score}.",
           "chr_none": "De teksten zijn te kort voor n-grams van tekens, dus chrF is 0.",
           "note": "Grijs: de referentie. Blauw: een n-gram van de vertaling dat ook in de referentie staat, hoogstens zo vaak geteld als de referentie het heeft. Paars: een n-gram van de vertaling dat niet in de referentie staat, of daar minder vaak."},
}


def _ratio(a: int, b: int, lang: str) -> str:
    return f"{_num(a, lang)}/{_num(b, lang)}" + (f" ({_pct(a, b, lang)})" if b else "")


def pair_view(lang: str, reference: list, translation: list) -> str:
    """BLEU and chrF of one translation against one reference, both lists of words, step by step."""
    w = _PAIR_TEXT[lang]
    body = _label(w["reference"]) + "<div>" + "".join(_chip(x, _CHIP_DATA) for x in reference) + "</div>"
    body += _label(w["words"])
    rows_ = []
    for n in range(1, ORDER + 1):
        grams = _counted(translation, reference, n)
        chips = "".join(_chip(g, _CHIP_PRED if ok else _CHIP_LOSS) for g, ok in grams)
        if not grams:
            chips = _faint(w["short"].format(n=n))
        rows_.append([str(n), chips, _faint(_ratio(sum(ok for _, ok in grams), len(grams), lang))])
    body += _table(rows_, w["head"], (2,))
    matched, total, c, r = bleu_counts(translation, [reference])
    if 0 in matched:
        body += _note(w["zero"].format(n=matched.index(0) + 1))
    else:
        bp = 1.0 if c > r else math.exp(1 - r / c)
        body += _note((w["penalty_no"] if c > r else w["penalty"]).format(c=c, r=r, bp=_dec(bp, lang, 3)))
        score = bleu(matched, total, c, r)
        body += _note(w["bleu"].format(gm=_dec(score / bp, lang, 3), score=_dec(score, lang, 3), big=_score(score, lang)))
    body += _label(w["letters"])
    body += _table([[_faint(w["letters_rows"][0]), _chip(letters(reference), _CHIP_DATA)],
                    [_faint(w["letters_rows"][1]), html.escape(letters(translation))]], w["letters_head"])
    counts_ = chrf_counts(translation, reference)
    rows_ = []
    for n, (h, r_, m) in enumerate(counts_, start=1):
        rows_.append([str(n), _num(h, lang), _num(r_, lang), _num(m, lang),
                      _pct(m, h, lang) if h and r_ else "–", _pct(m, r_, lang) if h and r_ else "–"])
    body += _table(rows_, w["chr_head"], (1, 2, 3, 4, 5))
    ps = [m / h for h, r_, m in counts_ if h and r_]
    rs = [m / r_ for h, r_, m in counts_ if h and r_]
    if ps:
        body += _note(w["chrf"].format(p=_pct(sum(ps), len(ps), lang), r=_pct(sum(rs), len(rs), lang),
                                       score=_score(chrf(counts_), lang)))
    else:
        body += _note(w["chr_none"])
    return body + _note(w["note"])


# --- the idea: one pair of Tatoeba with a compound --------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Scoring one translation: BLEU and chrF",
           "pair": "Two Dutch translations by people of *How many kinds of insect are there?*, one with a compound: the first is the reference, the second the translation to score."},
    "nl": {"title": "Eén vertaling scoren: BLEU en chrF",
           "pair": "Twee Nederlandse vertalingen door mensen van *How many kinds of insect are there?*, één met een samenstelling: de eerste is de referentie, de tweede de vertaling om te scoren."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    ref_text, trans_text, ref, trans = compounds()[0]
    body = _note(w["pair"]) + _label(f"{ref_text} | {trans_text}") + pair_view(lang, list(ref), list(trans))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the box: a reference and a translation the learner types -----------------------------------------------------------------

_BOX_TEXT = {
    "en": {"title": "Your translation, scored",
           "none": "Type a reference and a translation in the boxes above, each with at least one word."},
    "nl": {"title": "Je vertaling, gescoord",
           "none": "Typ een referentie en een vertaling in de vakken hierboven, elk met minstens één woord."},
}


def box_picture(lang: str, reference: str, translation: str) -> str:
    """The box: BLEU and chrF of what the learner typed as translation, against what the learner typed as reference."""
    lang = lang if lang in LANGS else "en"
    w = _BOX_TEXT[lang]
    ref, trans = words((reference or "")[:300]), words((translation or "")[:300])
    if not ref or not trans:
        return yours.figure(w["title"], _note(w["none"]), alt=w["none"])
    return yours.figure(w["title"], pair_view(lang, ref, trans), alt=w["title"])


# --- the picture under exercise 1: clipped counts of word for word --------------------------------------------------------

# The example of Papineni and colleagues (2002): a translation that repeats "the", and two references.
PAPINENI = ("the the the the the the the.", "The cat is on the mat.", "There is a cat on the mat.")


@functools.cache
def _word_grams() -> tuple:
    """For each word-for-word translation: per n-gram length, its counts and the largest counts in its references."""
    ts, rs = _systems()["direct"]
    return tuple(tuple((ngram_counts(list(t), n), _most([list(r) for r in refs], n)) for n in range(1, ORDER + 1))
                 for t, refs in zip(ts, rs))


_CLIP_TEXT = {
    "en": {"title": "Your clipped on the word-for-word translations",
           "papineni": "The example of Papineni and colleagues: *the the the the the the the*, with the references *the cat is on the mat* and *there is a cat on the mat*. Your clipped counts {k} of the 7 *the*.",
           "papineni_course": "The course's clipped counts 2: the most *the* in one reference.",
           "corpus": "Your clipped on the word-for-word translations of all {n} sentences, against the translations by people:",
           "head": ["n", "n-grams in word for word", "In a translation by people", "Share", ""],
           "course": "the course's clipped: {want}",
           "note": "Gray: the n-grams of word for word. Blue: what your clipped counts, when it is what the course's clipped counts. Purple: what differs from the course's clipped.",
           "kind": "a whole number"},
    "nl": {"title": "Je clipped op de vertalingen woord voor woord",
           "papineni": "Het voorbeeld van Papineni en collega's: *the the the the the the the*, met de referenties *the cat is on the mat* en *there is a cat on the mat*. Je clipped telt {k} van de 7 keer *the*.",
           "papineni_course": "De clipped van de course telt er 2: hoe vaak *the* hoogstens in één referentie staat.",
           "corpus": "Je clipped op de vertalingen woord voor woord van alle {n} zinnen, tegen de vertalingen door mensen:",
           "head": ["n", "n-grams in woord voor woord", "In een vertaling door mensen", "Deel", ""],
           "course": "de clipped van de course: {want}",
           "note": "Grijs: de n-grams van woord voor woord. Blauw: wat je clipped telt, als dat is wat de clipped van de course telt. Paars: wat anders is dan bij de clipped van de course.",
           "kind": "een heel getal"},
}


def clipped_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CLIP_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    mine0 = ngram_counts(words(PAPINENI[0]), 1)
    most0 = _most([words(PAPINENI[1]), words(PAPINENI[2])], 1)
    grams = _word_grams()

    def make():
        return fn(dict(mine0), dict(most0)), [[fn(dict(m), dict(t)) for m, t in one] for one in grams]

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    k, every = got
    if not _is_count(k) or not all(_is_count(x) for one in every for x in one):
        return _empty(w, "odd")
    right = k == clipped(mine0, most0)
    body = (f'<div style="{_UI}">' + _marked(w["papineni"].format(k="\0")).replace(
        "\0", f'<span style="{_PRED if right else _LOSS}">{html.escape(_num(k, lang))}</span>') + "</div>")
    if not right:
        body += _note(w["papineni_course"])
    want = counts("direct")[0]
    total = counts("direct")[1]
    rows_ = []
    for n in range(ORDER):
        got_n = sum(one[n] for one in every)
        same = got_n == want[n]
        style, color = (_PRED, "var(--ea-pred)") if same else (_LOSS, "var(--ea-loss)")
        cell = f'<span style="{style}">{_num(got_n, lang)}</span>'
        if not same:
            cell += _note(w["course"].format(want=_num(want[n], lang)))
        rows_.append([str(n + 1), _gray(_num(total[n], lang)), cell, _faint(_pct(got_n, total[n], lang)),
                      _bar(got_n / total[n], color)])
    body += _label(w["corpus"].format(n=_num(len(every), lang))) + _table(rows_, w["head"], (1, 2, 3))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the picture under exercise 2: BLEU of three ways to translate ------------------------------------------------------------

_BLEU_TEXT = {
    "en": {"title": "Your bleu on three ways to translate",
           "head": ["", "Sentences", "Your BLEU", ""],
           "names": {"direct": "Word for word", "ruled": "With the rule of *Translating by rules*",
                     "people": "People against people"},
           "course": "the course's bleu: {want}",
           "zeros": "Sentence by sentence, your bleu gives 0 to {a} of the {n} word-for-word translations ({pa}), and to {b} of the {m} translations by people ({pb}).",
           "zeros_course": "The course's bleu: {a} and {b}.",
           "note": "Gray: the number of sentences. Blue: what your bleu gives, when it is what the course's bleu gives. Purple: what differs from the course's bleu.",
           "kind": "a number"},
    "nl": {"title": "Je bleu op drie manieren van vertalen",
           "head": ["", "Zinnen", "Je BLEU", ""],
           "names": {"direct": "Woord voor woord", "ruled": "Met de regel van *Vertalen met regels*",
                     "people": "Mensen tegen mensen"},
           "course": "de bleu van de course: {want}",
           "zeros": "Zin voor zin geeft je bleu 0 aan {a} van de {n} vertalingen woord voor woord ({pa}), en aan {b} van de {m} vertalingen door mensen ({pb}).",
           "zeros_course": "De bleu van de course: {a} en {b}.",
           "note": "Grijs: het aantal zinnen. Blauw: wat je bleu geeft, als dat is wat de bleu van de course geeft. Paars: wat anders is dan bij de bleu van de course.",
           "kind": "een getal"},
}


def bleu_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _BLEU_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    corpora = {name: counts(name) for name in SYSTEMS}
    sentences = {name: _bleu_counts(name) for name in ("direct", "people")}

    def make():
        return ({name: fn(list(m), list(t), c, r) for name, (m, t, c, r) in corpora.items()},
                {name: [fn(list(m), list(t), c, r) for m, t, c, r in one] for name, one in sentences.items()})

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    whole, each = got
    if not all(_is_count(x) for x in list(whole.values()) + each["direct"] + each["people"]):
        return _empty(w, "odd")
    num = numbers()
    rows_ = []
    for name in SYSTEMS:
        same = math.isclose(float(whole[name]), num[name]["bleu"] / 100, rel_tol=1e-6, abs_tol=1e-9)
        style, color = (_PRED, "var(--ea-pred)") if same else (_LOSS, "var(--ea-loss)")
        cell = f'<span style="{style}">{_score(float(whole[name]), lang)}</span>'
        if not same:
            cell += _note(w["course"].format(want=_dec(num[name]["bleu"], lang)))
        rows_.append([_marked(w["names"][name]), _gray(_num(num[name]["n"], lang)), cell,
                      _bar(float(whole[name]), color)])
    a = sum(1 for x in each["direct"] if float(x) == 0)
    b = sum(1 for x in each["people"] if float(x) == 0)
    n, m = len(each["direct"]), len(each["people"])
    zeros = w["zeros"].format(a=_num(a, lang), n=_num(n, lang), pa=_pct(a, n, lang), b=_num(b, lang), m=_num(m, lang),
                              pb=_pct(b, m, lang))
    same = (a, b) == (num["direct"]["zero"], num["people"]["zero"])
    body = (_table(rows_, w["head"], (1, 2))
            + f'<div style="{_UI};{_PRED if same else _LOSS}">{html.escape(zeros)}</div>')
    if not same:
        body += _note(w["zeros_course"].format(a=_num(num["direct"]["zero"], lang), b=_num(num["people"]["zero"], lang)))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the picture under exercise 3: chrF of Dutch compounds ----------------------------------------------------------------------

_CHR_TEXT = {
    "en": {"title": "Your char_ngrams on Dutch compounds",
           "head": ["Reference", "Translation", "BLEU", "chrF with your char_ngrams"],
           "course": "the course's char_ngrams: {want}",
           "people": "People against people, on the {n} sentences with more than one English translation: BLEU {bleu}, chrF with your char_ngrams {chrf}.",
           "people_course": "With the course's char_ngrams: {want}.",
           "note": "Gray: the references, the translations and BLEU. Blue: chrF with your char_ngrams, when it is what the course's char_ngrams gives. Purple: what differs from the course's char_ngrams.",
           "kind": "a dictionary of counts"},
    "nl": {"title": "Je char_ngrams op Nederlandse samenstellingen",
           "head": ["Referentie", "Vertaling", "BLEU", "chrF met je char_ngrams"],
           "course": "de char_ngrams van de course: {want}",
           "people": "Mensen tegen mensen, op de {n} zinnen met meer dan één Engelse vertaling: BLEU {bleu}, chrF met je char_ngrams {chrf}.",
           "people_course": "Met de char_ngrams van de course: {want}.",
           "note": "Grijs: de referenties, de vertalingen en BLEU. Blauw: chrF met je char_ngrams, als dat is wat de char_ngrams van de course geeft. Paars: wat anders is dan bij de char_ngrams van de course.",
           "kind": "een dictionary met aantallen"},
}


def _safe_grams(fn):
    """The learner's char_ngrams, for chrF: a ValueError when it gives something other than a dictionary of counts."""
    def grams(text_, n):
        out = fn(text_, n)
        if not isinstance(out, dict) or not all(isinstance(k, str) and _is_count(v) for k, v in out.items()):
            raise ValueError
        return out
    return grams


def chr_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _CHR_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    pairs_ = compounds()
    ts, rs = _systems()["people"]
    grams = _safe_grams(fn)

    def make():
        try:
            mine = [chrf(chrf_counts(list(b), list(a), grams)) for _, _, a, b in pairs_]
            people = add_chrf([best_chrf_counts(list(t), [list(r) for r in refs], grams) for t, refs in zip(ts, rs)])
        except ValueError:
            return None
        return mine, chrf(people)

    got = _call(make)
    if got is _SLOW:
        return _empty(w, "slow")
    if got is _STOPPED:
        return _empty(w, "stopped")
    if got is None:
        return _empty(w, "odd")
    mine, people = got
    num = numbers()
    rows_ = []
    for (ref_text, trans_text, _, _), x, (b, c) in zip(pairs_, mine, num["compounds"]):
        same = math.isclose(100 * x, c, rel_tol=1e-6, abs_tol=1e-9)
        cell = f'<span style="{_PRED if same else _LOSS}">{_score(x, lang)}</span>'
        if not same:
            cell += _note(w["course"].format(want=_dec(c, lang)))
        rows_.append([_gray(ref_text), _gray(trans_text), _gray(_dec(b, lang)), cell])
    same = math.isclose(100 * people, num["people"]["chrf"], rel_tol=1e-6, abs_tol=1e-9)
    line = html.escape(w["people"].format(n=_num(num["people"]["n"], lang), bleu=_dec(num["people"]["bleu"], lang),
                                          chrf="\0"))
    line = line.replace("\0", f'<span style="{_PRED if same else _LOSS}">{html.escape(_score(people, lang))}</span>')
    body = _table(rows_, w["head"], (2, 3)) + f'<div style="{_UI}">{line}</div>'
    if not same:
        body += _note(w["people_course"].format(want=_dec(num["people"]["chrf"], lang)))
    return yours.figure(w["title"], body + _note(w["note"]), alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.ngram_counts(...)`; the function gets a
    copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

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


def _grams_val(text_: str, n: int) -> _Val:
    return _Val(f"C.ngram_counts(C.words({_value(text_)}), {n})", ngram_counts(words(text_), n))


def _letters_val(text_: str) -> _Val:
    return _Val(f"C.letters(C.words({_value(text_)}))", letters(words(text_)))


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
TOY_CLIPPED = (({"the": 7}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1}),
               ({"the": 1, "cat": 1, "sat": 1}, {"the": 2, "cat": 1, "is": 1, "on": 1, "mat": 1}))
TOY_BLEU = (([6, 4, 2, 1], [6, 5, 4, 3], 6, 6),
            ([4, 3, 2, 1], [4, 3, 2, 1], 4, 6))
TOY_CHARS = (("banana", 2), ("hand", 3))

_CL_HEAD = "def clipped(mine, theirs):\n    matched = 0\n    for gram in mine:\n"
_CL_IF = "        if gram in theirs:\n"
_CL_ADD = "            matched += min(mine[gram], theirs[gram])\n"


def _cl(check: str = _CL_IF, add: str = _CL_ADD, end: str = "    return matched\n") -> str:
    return _CL_HEAD + check + add + end


_CL = _cl()
_CL_START = _cl("        if ...:\n", "            matched += ...\n")
_CL_NO_CLIP = _cl(add="            matched += mine[gram]\n")
_CL_THEIRS = _cl(add="            matched += theirs[gram]\n")
_CL_MAX = _cl(add="            matched += max(mine[gram], theirs[gram])\n")
_CL_ONE = _cl(add="            matched += 1\n")
_CL_MINE = _cl("        if gram in mine:\n")
_CL_INSIDE = _cl(end="        return matched\n")

_BL_HEAD = ("import numpy as np\n\n\ndef bleu(matched, total, c, r):\n    if 0 in matched:\n        return 0.0\n"
            "    logs = [np.log(matched[n] / total[n]) for n in range(len(matched))]\n")
_BL_MEAN = "    mean = np.mean(logs)\n"
_BL_PENALTY = "        penalty = np.exp(1 - r / c)\n"


def _bl(mean: str = _BL_MEAN, penalty: str = _BL_PENALTY) -> str:
    return (_BL_HEAD + mean + "    if c > r:\n        penalty = 1.0\n    else:\n" + penalty
            + "    return penalty * np.exp(mean)\n")


_BL = _bl()
_BL_START = _bl("    mean = ...\n", "        penalty = ...\n")
_BL_NO_PENALTY = _bl(penalty="        penalty = 1.0\n")
_BL_FLIPPED = _bl(penalty="        penalty = np.exp(1 - c / r)\n")
_BL_NO_EXP = _bl(penalty="        penalty = 1 - r / c\n")
_BL_SUM = _bl(mean="    mean = np.sum(logs)\n")

_CN_HEAD = "def char_ngrams(letters, n):\n    counts = {}\n"
_CN_RANGE = "    for i in range(len(letters) - n + 1):\n"
_CN_GRAM = "        gram = letters[i:i + n]\n"


def _cn(loop: str = _CN_RANGE, gram: str = _CN_GRAM) -> str:
    return (_CN_HEAD + loop + gram + "        if gram in counts:\n            counts[gram] += 1\n        else:\n"
            "            counts[gram] = 1\n    return counts\n")


_CN = _cn()
_CN_START = _cn("    for i in range(...):\n", "        gram = ...\n")
_CN_SHORT = _cn("    for i in range(len(letters) - n):\n")
_CN_LONG = _cn("    for i in range(len(letters)):\n")
_CN_ONE = _cn(gram="        gram = letters[i]\n")
_CN_TO_N = _cn(gram="        gram = letters[i:n]\n")


def _not_number(got, text_) -> bool:
    return not _is_count(got)


def _not_dict(got, text_) -> bool:
    return not isinstance(got, dict)


REQUIRED = ["clipped", "bleu", "char_ngrams", "beta"]

EXERCISES = {
    "clipped": FunctionBox(
        name="clipped",
        picture=clipped_picture,
        function="clipped",
        start=_CL_START,
        solution=_CL,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_CLIPPED],
        hidden=[(_written({}), _written({"the": 1})),
                (_written({"dog": 2}), _written({"cat": 1})),
                (_written({"a": 3, "b": 1}), _written({"a": 3, "b": 2})),
                (_written({"the cat": 2, "cat sat": 1}), _written({"the cat": 1, "cat is": 1, "on the": 1})),
                (_grams_val(PAPINENI[0], 1), _grams_val(PAPINENI[2], 1)),
                (_grams_val("i will the next bus take", 1), _grams_val("I'll take the next bus.", 1)),
                (_grams_val("i will the next bus take", 2), _grams_val("I'll take the next bus.", 2))],
        first=[_DOTS],
        mistakes=[
            Mistake("keyerror", error=lambda e: isinstance(e, KeyError),
                    en="`{call}` stops with a `KeyError`: your function looks up an n-gram in `theirs` that is not a key of `theirs`. Does your function add something only for an n-gram that `theirs` has?",
                    nl="`{call}` stopt met een `KeyError`: je functie zoekt een n-gram op in `theirs` dat geen key van `theirs` is. Telt je functie alleen iets op bij een n-gram dat in `theirs` staat?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: a whole number, how many n-grams count. Does your function give back `matched`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een heel getal, hoeveel n-grams tellen. Geeft je functie `matched` terug?"),
            Mistake("no-clip", wrong=_CL_NO_CLIP,
                    en="`{call}` gives {got}, and it should give {want}: your function counts all 7 times *the*, and the reference has *the* twice. How do you keep a count from going above the count in `theirs`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt alle 7 keer *the*, en de referentie heeft *the* twee keer. Hoe zorg je dat een telling niet boven de telling in `theirs` uitkomt?"),
            Mistake("theirs-only", wrong=_CL_THEIRS,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the count in `theirs`, 2 for *the*, while the translation has *the* once. An n-gram counts as often as the translation has it, and at most as often as the reference has it: which of the two counts is that?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt het aantal uit `theirs` op, 2 voor *the*, terwijl de vertaling *the* één keer heeft. Een n-gram telt zo vaak als de vertaling het heeft, en hoogstens zo vaak als de referentie het heeft: welke van de twee tellingen is dat?"),
            Mistake("max", wrong=_CL_MAX,
                    en="`{call}` gives {got}, and it should give {want}: `max` gives the larger of the two counts, 7 for *the*. Which function gives the smaller one?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `max` geeft de grootste van de twee tellingen, 7 voor *the*. Welke functie geeft de kleinste?"),
            Mistake("count-one", wrong=_CL_ONE,
                    en="`{call}` gives {got}, and it should give {want}: your function adds 1 for each n-gram, also for *the*, which the translation has 7 times and the reference twice. How often may *the* count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt 1 op voor elk n-gram, ook voor *the*, dat de vertaling 7 keer heeft en de referentie twee keer. Hoe vaak mag *the* tellen?"),
            Mistake("return-inside", wrong=_CL_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: only the count of the first n-gram. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen de telling van het eerste n-gram. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether `gram` is a key of `theirs`, and the line under it adds to `matched` how often that n-gram counts.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of `gram` een key van `theirs` is, en de regel eronder telt bij `matched` op hoe vaak dat n-gram telt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Each n-gram of `mine` counts as often as `mine` has it, but never more often than `theirs` has it: does your function add the smaller of the two counts, and nothing for an n-gram that `theirs` lacks?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Elk n-gram van `mine` telt zo vaak als `mine` het heeft, maar nooit vaker dan `theirs` het heeft: telt je functie de kleinste van de twee tellingen op, en niets bij een n-gram dat niet in `theirs` staat?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function look at every n-gram of `mine`, also when there are none or when `theirs` has an n-gram more often?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Kijkt je functie naar elk n-gram van `mine`, ook als er geen zijn of als `theirs` een n-gram vaker heeft?"},
            "correct": {"en": "Right: an n-gram counts at most as often as the reference has it. How much of word for word still counts, in the picture below, for groups of one word and for groups of four?",
                        "nl": "Goed: een n-gram telt hoogstens zo vaak als de referentie het heeft. Hoeveel van woord voor woord telt nog mee, in het plaatje hieronder, bij groepen van één woord en bij groepen van vier?"},
        },
    ),
    "bleu": FunctionBox(
        name="bleu",
        picture=bleu_picture,
        function="bleu",
        start=_BL_START,
        solution=_BL,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_BLEU],
        hidden=[(_written([5, 3, 2, 1]), _written([6, 5, 4, 3]), _written(6), _written(5)),
                (_written([3, 1, 0, 0]), _written([4, 3, 2, 1]), _written(4), _written(4)),
                (_written([2, 1]), _written([3, 2]), _written(3), _written(9)),
                (_written([1]), _written([1]), _written(1), _written(1))]
               + [tuple(_written(x) for x in counts(name)) for name in SYSTEMS],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: a number from 0 to 1. Does your function still give back `penalty * np.exp(mean)`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een getal van 0 tot 1. Geeft je functie nog steeds `penalty * np.exp(mean)` terug?"),
            Mistake("no-penalty", wrong=_BL_NO_PENALTY,
                    en="`{call}` gives {got}, and it should give {want}: the translation has 4 words and the reference 6, and your function gives it no brevity penalty. What should `penalty` be in the `else`, where `c` is not more than `r`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de vertaling heeft 4 woorden en de referentie 6, en je functie geeft de vertaling geen brevity penalty. Wat moet `penalty` zijn in de `else`, waar `c` niet meer is dan `r`?"),
            Mistake("flipped", wrong=_BL_FLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: a brevity penalty above 1, which raises the score of a translation that is too short. Which length goes on top in the fraction: that of the reference, `r`, or that of the translation, `c`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een brevity penalty boven 1, en zo'n penalty verhoogt de score van een vertaling die te kort is. Welke lengte staat boven de breukstreep: de lengte van de referentie, `r`, of de lengte van de vertaling, `c`?"),
            Mistake("no-exp", wrong=_BL_NO_EXP,
                    en="`{call}` gives {got}, and it should give {want}: your penalty is one minus the fraction of the lengths, which is 0 when the two lengths are the same. That is the power that *e* is raised to, not the brevity penalty itself. Which NumPy function raises *e* to a power?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je penalty is één min de breuk van de lengtes, en dat is 0 als de twee lengtes gelijk zijn. Dat is de macht waartoe *e* verheven wordt, niet de brevity penalty zelf. Welke NumPy-functie verheft *e* tot een macht?"),
            Mistake("sum-logs", wrong=_BL_SUM,
                    en="`{call}` gives {got}, and it should give {want}: `np.sum` adds up the four logarithms, so `np.exp` gives the product of the four shares, 0.133. The geometric mean takes the mean of the logarithms: which NumPy function gives a mean?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `np.sum` telt de vier logaritmen op, dus `np.exp` geeft het product van de vier delen, 0,133. Het meetkundig gemiddelde neemt het gemiddelde van de logaritmen: welke NumPy-functie geeft een gemiddelde?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `mean` is the mean of the logarithms in `logs`, and `penalty` in the `else` is the brevity penalty of a translation that is not longer than its reference.",
                      "nl": "Vervang de twee `...`: `mean` is het gemiddelde van de logaritmen in `logs`, en `penalty` in de `else` is de brevity penalty van een vertaling die niet langer is dan de referentie."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. BLEU is the brevity penalty times *e* to the power of the mean of the logarithms. Is `mean` that mean, and is `penalty` the brevity penalty of the idea?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. BLEU is de brevity penalty keer *e* tot de macht het gemiddelde van de logaritmen. Is `mean` dat gemiddelde, en is `penalty` de brevity penalty van het idee?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give a penalty of 1 to a translation longer than its reference, and use every length in `matched`, also when there are fewer than four?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie een penalty van 1 aan een vertaling die langer is dan de referentie, en gebruikt je functie elke lengte in `matched`, ook als het er minder dan vier zijn?"},
            "correct": {"en": "Right: your function computes BLEU from the counts. Which of the three ways to translate scores highest, in the picture below, and how far is it from 100?",
                        "nl": "Goed: je functie berekent BLEU uit de tellingen. Welke van de drie manieren van vertalen scoort het hoogst, in het plaatje hieronder, en hoe ver zit de hoogste score van 100?"},
        },
    ),
    "char_ngrams": FunctionBox(
        name="char_ngrams",
        picture=chr_picture,
        function="char_ngrams",
        start=_CN_START,
        solution=_CN,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_CHARS],
        hidden=[(_written("abc"), _written(4)),
                (_written("aaaa"), _written(2)),
                (_written("kat"), _written(1)),
                (_written(""), _written(2)),
                (_letters_val("Alles hier is handgemaakt."), _written(6)),
                (_letters_val("Hoeveel insectensoorten zijn er?"), _written(3))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give {want}: a dictionary with how often each n-gram occurs. Does your function give back `counts`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een dictionary met hoe vaak elk n-gram voorkomt. Geeft je functie `counts` terug?"),
            Mistake("range-short", wrong=_CN_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: the last *na* is missing. *banana* has 6 letters and 5 bigrams: how many places `i` does the loop need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de laatste *na* ontbreekt. *banana* heeft 6 letters en 5 bigrams: hoeveel plaatsen `i` heeft de loop nodig?"),
            Mistake("range-long", wrong=_CN_LONG,
                    en="`{call}` gives {got}, and it should give {want}: the *a* at the end is one letter, not a bigram. At which place does the last n-gram of `letters` start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de *a* aan het eind is één letter, geen bigram. Op welke plaats begint het laatste n-gram van `letters`?"),
            Mistake("one-letter", wrong=_CN_ONE,
                    en="`{call}` gives {got}, and it should give {want}: single letters, while `n` is 2. Which slice gives the `n` letters from place `i` on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: losse letters, terwijl `n` 2 is. Welke slice geeft de `n` letters vanaf plaats `i`?"),
            Mistake("to-n", wrong=_CN_TO_N,
                    en="`{call}` gives {got}, and it should give {want}: a slice stops at the place after the colon, and `n` is a length, not a place. At which place does the n-gram that starts at `i` stop?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een slice stopt op de plaats na de dubbele punt, en `n` is een lengte, geen plaats. Op welke plaats stopt het n-gram dat op `i` begint?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the loop goes over every place `i` where an n-gram of `n` letters starts, and `gram` is the n-gram that starts there.",
                      "nl": "Vervang de twee `...`: de loop gaat over elke plaats `i` waar een n-gram van `n` letters begint, en `gram` is het n-gram dat daar begint."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every n-gram have exactly `n` letters, and does your loop start one at every place, up to the last place where `n` letters still fit?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Heeft elk n-gram precies `n` letters, en begint je loop er een op elke plaats, tot de laatste plaats waar nog `n` letters passen?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give an empty dictionary when `letters` is shorter than `n`, and count an n-gram that occurs more than once each time?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie een lege dictionary als `letters` korter is dan `n`, en telt je functie een n-gram dat vaker voorkomt elke keer?"},
            "correct": {"en": "Right: your function counts the character n-grams that chrF compares. How do BLEU and chrF judge the Dutch compounds, in the picture below?",
                        "nl": "Goed: je functie telt de n-grams van tekens die chrF vergelijkt. Hoe beoordelen BLEU en chrF de Nederlandse samenstellingen, in het plaatje hieronder?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("clipped", _CL_START, "empty", None),
    ("clipped", _CL, "correct", None),
    ("clipped", _cl(add="            matched += ...\n"), "wrong", "dots-left"),
    ("clipped", _CL_MINE, "error", "keyerror"),
    ("clipped", _cl(end="    return [matched]\n"), "wrong", "not-number"),
    ("clipped", _CL_NO_CLIP, "wrong", "no-clip"),
    ("clipped", _CL_THEIRS, "wrong", "theirs-only"),
    ("clipped", _CL_MAX, "wrong", "max"),
    ("clipped", _CL_ONE, "wrong", "count-one"),
    ("clipped", _CL_INSIDE, "wrong", "return-inside"),
    ("clipped", _cl(add="            if mine[gram] < theirs[gram]:\n                matched += mine[gram]\n"
                        "            else:\n                matched += theirs[gram]\n"), "correct", None),
    ("clipped", "def clipped(mine, theirs):\n    return sum(min(k, theirs[g]) for g, k in mine.items() if g in theirs)\n",
     "correct", None),
    ("clipped", _cl(add="            matched = matched + min(theirs[gram], mine[gram])\n"), "correct", None),
    ("bleu", _BL_START, "empty", None),
    ("bleu", _BL, "correct", None),
    ("bleu", _bl(penalty="        penalty = ...\n"), "wrong", "dots-left"),
    ("bleu", _BL.replace("    return penalty * np.exp(mean)\n", "    return str(penalty * np.exp(mean))\n"),
     "wrong", "not-number"),
    ("bleu", _BL_NO_PENALTY, "wrong", "no-penalty"),
    ("bleu", _BL_FLIPPED, "wrong", "flipped"),
    ("bleu", _BL_NO_EXP, "wrong", "no-exp"),
    ("bleu", _BL_SUM, "wrong", "sum-logs"),
    ("bleu", _bl(mean="    mean = sum(logs) / len(logs)\n"), "correct", None),
    ("bleu", _bl(penalty="        penalty = np.e ** (1 - r / c)\n"), "correct", None),
    ("bleu", _bl(penalty="        penalty = np.exp(1 - r / c)\n").replace("if c > r:", "if c >= r:"), "correct", None),
    ("char_ngrams", _CN_START, "empty", None),
    ("char_ngrams", _CN, "correct", None),
    ("char_ngrams", _cn(gram="        gram = ...\n"), "wrong", "dots-left"),
    ("char_ngrams", _CN.replace("    return counts\n", "    return list(counts)\n"), "wrong", "not-dict"),
    ("char_ngrams", _CN_SHORT, "wrong", "range-short"),
    ("char_ngrams", _CN_LONG, "wrong", "range-long"),
    ("char_ngrams", _CN_ONE, "wrong", "one-letter"),
    ("char_ngrams", _CN_TO_N, "wrong", "to-n"),
    ("char_ngrams", "def char_ngrams(letters, n):\n    counts = {}\n    for i in range(len(letters) - n + 1):\n"
     "        counts[letters[i:i + n]] = counts.get(letters[i:i + n], 0) + 1\n    return counts\n", "correct", None),
    ("char_ngrams", _cn(gram="        gram = letters[i:n + i]\n"), "correct", None),
]
