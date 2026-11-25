# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-extractive-qa: which sentence of a paragraph holds the place where an answer starts
(sentence_of); which sentence shares the most different words with a question, stop words left out (best_sentence);
and how often that sentence holds the answer, over many questions (share_right).

The paragraphs and questions are the 240 paragraphs of XQuAD (Artetxe, Ruder and Yogatama, ACL 2020; CC-BY-SA-4.0)
from the development set of SQuAD 1.1 (Rajpurkar, Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0), with their
1,190 English questions, and the same paragraphs and 1,183 questions in Dutch from SQuAD-NL v1.1 (de Vries, Wieling and
Nissim, EMNLP 2023; CC-BY-SA-4.0), in data/xquad-squad-nl.json.gz, read with everything_ai.xquad. Each answer is the
text that people marked in the paragraph and the place of its first character; the lesson takes the first answer of
each question. The course cuts each paragraph into sentences with sentences() below, and leaves out the stop words of
Snowball (books.stop_words, BSD-3-Clause). Paragraphs marked as not shown, or with he or she (or their Dutch) in them,
are counted but never shown. Every number in the text and the pictures is computed by the course
(tests/test_nlp_extractive_qa.py computes them again)."""
import functools
import html
import re

from everything_ai import books, xquad, yours
from everything_ai.check import FunctionBox, Mistake

LANGS = ("en", "nl")
RHINE = 1783                            # the paragraph of the idea and the worked example: the Rhine–Meuse delta
NAME = "572ff673b2c2fd1400568669"       # What is the name of the Delta in the Netherlands?
BEGIN = "572ff673b2c2fd140056866a"      # Where does the Delta in the Netherlands begin?
NICKNAME = "572ff673b2c2fd140056866b"   # What is the nickname for the Delta in the Netherlands? (the worked example)
COSTUME = "5726e37ef1498d1400e8eed8"    # another sentence shares more words: the costume collection of the V&A
JACKSONVILLE = "5727cb4b2ca10214002d967a"   # a tie, and the first sentence wins: who Jacksonville is named after
WARSAW = "5733834ed058e614000b5c28"     # the sentence with the answer shares no word: Warsaw's stock exchange
PICKED = (BEGIN, COSTUME, JACKSONVILLE, WARSAW)     # the questions of the picture under exercise 2
KINDS = ("right", "tie", "fewer", "zero")           # how the course's reader does on a question

_WORD = re.compile(r"[^\W_]+")


def words(text: str) -> list:
    """The words of a text in lower case, without punctuation: "On Sunday, it stays closed." gives
    ["on", "sunday", "it", "stays", "closed"]. A number is a word too: "14:30" gives ["14", "30"]."""
    return _WORD.findall(text.lower())


# --- cutting a paragraph into sentences ----------------------------------------------------------------------------------

# Words with a full stop that do not end a sentence, in small letters, as they stand in the paragraphs.
ABBREVIATIONS = frozenset({
    "a.d.", "a.m.", "approx.", "b.c.", "bijv.", "blz.", "ca.", "co.", "col.", "d.w.z.", "dec.", "dept.", "dhr.", "dr.",
    "drs.", "e.g.", "est.", "etc.", "feb.", "fig.", "ft.", "gen.", "gov.", "i.e.", "inc.", "ing.", "ir.", "jan.",
    "jr.", "lt.", "ltd.", "m.a.w.", "mevr.", "mgr.", "mr.", "mrs.", "ms.", "mt.", "n.chr.", "no.", "nov.", "nr.",
    "o.a.", "oct.", "op.", "p.m.", "prof.", "rep.", "resp.", "rev.", "sen.", "sept.", "sint.", "sr.", "st.", "u.k.",
    "u.s.", "v.chr.", "vol.", "vs.",
})
_MARK = re.compile(r"[.?!][\"”’')\]]*(?:\[\d+\])?\s+")


def sentences(text: str) -> list:
    """The sentences of a paragraph, each with the white space after it, so that "".join(sentences(text)) == text.
    A sentence ends at . ? or ! (with any closing quote or bracket and a note such as [1] after it), followed by white
    space and a capital or a digit; not after an abbreviation of ABBREVIATIONS or an initial such as J. or U.S."""
    out, start = [], 0
    for m in _MARK.finditer(text):
        after = text[m.end():m.end() + 2].lstrip("\"“‘'(")
        if not after or not (after[0].isupper() or after[0].isdigit()):
            continue
        word = text[start:m.start() + 1].split()[-1].lstrip("(\"“‘'").lower()
        if word in ABBREVIATIONS or re.fullmatch(r"(?:[^\W\d_]\.)+", word):
            continue
        out.append(text[start:m.end()])
        start = m.end()
    if start < len(text):
        out.append(text[start:])
    return out


STOP_WORDS = {lang: frozenset(books.stop_words(lang)) for lang in LANGS}


# --- the course's own answers to the exercises ---------------------------------------------------------------------------

def sentence_of(sentences, start):
    """The place of the sentence in which the character at `start` of the paragraph stands; the sentences together
    are the paragraph."""
    end = 0
    for k in range(len(sentences)):
        end = end + len(sentences[k])
        if start < end:
            return k
    return None


def best_sentence(question, sentences, stop_words):
    """The place of the sentence that shares the most different words with the question, its stop words left out;
    the first of them on a tie, and the first sentence when none shares a word."""
    asked = set([w for w in words(question) if w not in stop_words])
    best = 0
    most = 0
    for k in range(len(sentences)):
        shared = len(asked & set(words(sentences[k])))
        if shared > most:
            best = k
            most = shared
    return best


def share_right(items, stop_words):
    """The share of the items (question, sentences, start of the answer) for which best_sentence picks the sentence
    that holds the start of the answer."""
    right = 0
    for question, sentences, start in items:
        if best_sentence(question, sentences, stop_words) == sentence_of(sentences, start):
            right = right + 1
    return right / len(items)


def shared_words(question: str, sentence: str, stop_words) -> list:
    """The different words of the question, stop words left out, that are also in the sentence, in the order of the
    question."""
    in_sentence = set(words(sentence))
    out = []
    for w in words(question):
        if w not in stop_words and w in in_sentence and w not in out:
            out.append(w)
    return out


# --- the questions --------------------------------------------------------------------------------------------------------

_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}


@functools.cache
def _questions(lang: str) -> tuple:
    """Every question with what the lesson needs: (id, question, sentences, start of the first answer, answers,
    paragraph, shown), in the order of SQuAD."""
    ps = xquad.paragraphs(lang)
    cut = {}
    out = []
    for q in xquad.questions(lang):
        p = ps[q.paragraph]
        if q.paragraph not in cut:
            cut[q.paragraph] = tuple(sentences(p.text))
        shown = p.shown and _PRONOUN[lang].search(p.text + " " + q.question) is None
        out.append((q.id, q.question, cut[q.paragraph], q.answers[0][1], q.answers, q.paragraph, shown))
    return tuple(out)


def items(lang: str) -> list:
    """Every question of `lang` as the item of exercise 3: (question, sentences, start of the first answer), as a new
    list each time."""
    return [(q, list(ss), start) for _, q, ss, start, _, _, _ in _questions(lang)]


@functools.cache
def _place(lang: str) -> dict:
    return {qid: i for i, (qid, *_) in enumerate(_questions(lang))}


def find(lang: str, qid: str) -> int:
    """The place of a question in _questions(lang), by its id."""
    return _place(lang)[qid]


def rhine_items(lang: str) -> list:
    """The three questions about the Rhine–Meuse delta as items of exercise 3."""
    return [x for x, (_, _, _, _, _, p, _) in zip(items(lang), _questions(lang)) if p == RHINE]


def rhine(lang: str) -> list:
    """The three sentences of the paragraph about the Rhine–Meuse delta."""
    return list(sentences(xquad.paragraphs(lang)[RHINE].text))


@functools.cache
def kinds(lang: str) -> tuple:
    """For each question how the course's reader does: right, or why not: a tie that the first sentence wins (tie),
    another sentence that shares more words (fewer), or a sentence with the answer that shares no word (zero)."""
    stop = STOP_WORDS[lang]
    out = []
    for _, q, ss, start, _, _, _ in _questions(lang):
        want, got = sentence_of(ss, start), best_sentence(q, ss, stop)
        if got == want:
            out.append("right")
            continue
        n_want, n_got = len(shared_words(q, ss[want], stop)), len(shared_words(q, ss[got], stop))
        out.append("zero" if n_want == 0 else "tie" if n_want == n_got else "fewer")
    return tuple(out)


def shown(lang: str, kind: str = "all") -> list:
    """The places of the questions that the lesson may show, of one kind or of all kinds."""
    return [i for i, (x, k) in enumerate(zip(_questions(lang), kinds(lang))) if x[6] and kind in ("all", k)]


# --- the numbers the text names ----------------------------------------------------------------------------------------------

@functools.cache
def numbers(lang: str) -> dict:
    """The numbers the lesson's text names, for `lang`."""
    qs = _questions(lang)
    stop = STOP_WORDS[lang]
    k = kinds(lang)
    gold = [sentence_of(ss, start) for _, _, ss, start, _, _, _ in qs]
    paragraphs = {p: ss for _, _, ss, _, _, p, _ in qs}
    return {
        "questions": len(qs),
        "paragraphs": len(paragraphs),
        "sentences": sum(len(ss) for ss in paragraphs.values()),
        "right": k.count("right"),
        "tie": k.count("tie"),
        "fewer": k.count("fewer"),
        "zero": k.count("zero"),
        "first": sum(g == 0 for g in gold),
        "random": sum(1 / len(ss) for _, _, ss, _, _, _, _ in qs),
        "kept": sum(best_sentence(q, ss, set()) == g for (_, q, ss, _, _, _, _), g in zip(qs, gold)),
        "sentence_words": sum(len(words(ss[g])) for (_, _, ss, _, _, _, _), g in zip(qs, gold)) / len(qs),
        "answer_words": sum(len(words(a[0][0])) for _, _, _, _, a, _, _ in qs) / len(qs),
        "shown": len(shown(lang)),
    }


# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_CHIP = ("display:inline-block;margin:0.08rem 0.2rem 0.08rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.78rem;line-height:1.5")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text)"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_CHIP_NONE = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.84rem;line-height:1.55"
_SPAN = ("padding:0 0.1rem;border-radius:2px;font-weight:600;-webkit-box-decoration-break:clone;"
         "box-decoration-break:clone")
_SPAN_PRED = _SPAN + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"


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


def _chip(text_: str, style: str) -> str:
    return f'<span style="{style}">{html.escape(text_)}</span>'


def _gray(text_: str) -> str:
    return f'<span style="{_DATA}">{html.escape(text_)}</span>'


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>")


def _short(text_: str, room: int = 90) -> str:
    """The start of a long sentence, with … where it goes on."""
    text_ = text_.strip()
    return text_ if len(text_) <= room else text_[:room].rsplit(" ", 1)[0] + " …"


def _around(sentence: str, start: int, end: int, room: int = 50) -> str:
    """The part of a sentence around sentence[start:end], with that span marked in blue; … where the sentence goes
    on."""
    a, b = max(0, start - room), min(len(sentence.rstrip()), end + room)
    head = ("… " if a > 0 else "") + sentence[a:start]
    tail = sentence[end:b] + (" …" if b < len(sentence.rstrip()) else "")
    return (f'<span style="{_DATA}">{html.escape(head)}</span><span style="{_SPAN_PRED}">{html.escape(sentence[start:end])}'
            f'</span><span style="{_DATA}">{html.escape(tail)}</span>')


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


def _is_place(v, n: int) -> bool:
    """Whether v is a whole number that is the place of one of n sentences."""
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < n


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


# --- the idea: one paragraph of SQuAD with its questions ---------------------------------------------------------------

_SQUAD_TEXT = {
    "en": {"title": "A paragraph of SQuAD, with its three questions",
           "head": ["Question", "Answers that people marked", "answer_start"],
           "note": "Gray: the paragraph, from the article Rhine, and the questions. Blue: the answers, as spans of the paragraph. Each person who answered marked a span of their own, so the answers to one question can start or end at different places."},
    "nl": {"title": "Een alinea van SQuAD, met de drie vragen erbij",
           "head": ["Vraag", "Antwoorden die mensen aanwezen", "answer_start"],
           "note": "Grijs: de alinea, van de Wikipedia-pagina Rijn, en de vragen. Blauw: de antwoorden, als spans van de alinea. Iedereen die antwoordde, wees een eigen span aan, dus de antwoorden op één vraag kunnen op verschillende plaatsen beginnen of eindigen."},
}


def squad_picture(lang: str) -> str:
    """The paragraph about the Rhine–Meuse delta with the spans of its answers marked, and its questions with every
    answer and where it starts."""
    lang = lang if lang in LANGS else "en"
    w = _SQUAD_TEXT[lang]
    text_ = xquad.paragraphs(lang)[RHINE].text
    marks = sorted({(start, start + len(t)) for qid in (NAME, BEGIN, NICKNAME)
                    for t, start in _questions(lang)[find(lang, qid)][4]})
    covered = [False] * len(text_)
    for a, b in marks:
        for j in range(a, b):
            covered[j] = True
    parts, j = [], 0
    while j < len(text_):
        k = j
        while k < len(text_) and covered[k] == covered[j]:
            k += 1
        parts.append(f'<span style="{_SPAN_PRED}">{html.escape(text_[j:k])}</span>' if covered[j] else _gray(text_[j:k]))
        j = k
    rows_ = []
    for qid in (NAME, BEGIN, NICKNAME):
        _, q, _, _, answers, _, _ = _questions(lang)[find(lang, qid)]
        rows_.append([_gray(q), "".join(_chip(t, _CHIP_PRED) for t, _ in answers),
                      f'<span style="{_DATA}">{", ".join(str(s) for _, s in answers)}</span>'])
    body = f'<div style="{_TEXT}">{"".join(parts)}</div>' + _table(rows_, w["head"], (2,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 1: which sentence holds each answer ----------------------------------------------------

_WHERE_TEXT = {
    "en": {"title": "Your sentence_of on the paragraph about the Rhine",
           "sentences": "The three sentences, with the places of their characters:",
           "place": "sentence {k}: characters {a} to {b}",
           "head": ["Question", "Answer", "Starts at", "Your sentence_of"],
           "course": "the course's sentence_of gives {want}",
           "summary": "On all {n} questions, your function gives the sentence that holds the start of the answer for {right} of them.",
           "note": "Gray: the sentences, the questions and where each answer starts. Blue: the place your function gives, when it is the sentence that holds the answer. Purple: a place that differs from the course's sentence_of.",
           "kind": "the place of a sentence"},
    "nl": {"title": "Je sentence_of op de alinea over de Rijn",
           "sentences": "De drie zinnen, met de plaatsen van de tekens:",
           "place": "zin {k}: tekens {a} tot en met {b}",
           "head": ["Vraag", "Antwoord", "Begint bij", "Je sentence_of"],
           "course": "de sentence_of van de course geeft {want}",
           "summary": "Op alle {n} vragen geeft je functie bij {right} vragen de zin waar het antwoord begint.",
           "note": "Grijs: de zinnen, de vragen en waar elk antwoord begint. Blauw: de plaats die je functie geeft, als dat de zin met het antwoord is. Paars: een plaats die anders is dan bij de sentence_of van de course.",
           "kind": "de plaats van een zin"},
}


def where_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _WHERE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    every = items(lang)
    mine = _call(lambda: [fn(list(ss), start) for _, ss, start in every])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    ss = rhine(lang)
    rows_ = []
    for qid in (NAME, BEGIN, NICKNAME):
        i = find(lang, qid)
        _, q, _, start, answers, _, _ = _questions(lang)[i]
        want, m = sentence_of(ss, start), mine[i]
        cell = (_chip(str(m), _CHIP_PRED) if m == want
                else _chip(repr(m), _CHIP_LOSS) + _note(w["course"].format(want=want)))
        rows_.append([_gray(q), _chip(answers[0][0], _CHIP_DATA), f'<span style="{_DATA}">{start}</span>', cell])
    ends = [sum(len(s) for s in ss[:k + 1]) for k in range(len(ss))]
    places = "".join(f'<div style="{_UI}">{html.escape(w["place"].format(k=k, a=end - len(s), b=end - 1))}: '
                     f'<span style="{_DATA}">{html.escape(_short(s, 70))}</span></div>'
                     for k, (s, end) in enumerate(zip(ss, ends)))
    right_ = sum(m == sentence_of(ss_, start) for m, (_, ss_, start) in zip(mine, every))
    body = (_label(w["sentences"]) + places + _table(rows_, w["head"], (2,))
            + f'<div style="{_UI}">{html.escape(w["summary"].format(n=_num(len(every), lang), right=_num(right_, lang)))}</div>'
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 2, and the box: the sentences of a paragraph and what a reader picks -------------------

_PICK_TEXT = {
    "en": {"title": "Your best_sentence on four questions",
           "head": ["", "Sentence", "Words it shares with the question"],
           "question": "{q} Answer: *{a}*.",
           "yours": "your pick",
           "here": "holds the answer",
           "none": "none",
           "course": "the course's best_sentence picks sentence {want}",
           "summary": "Your function picks the sentence with the answer for {right} of these {n} questions.",
           "note": "Gray: the sentences, and the words of the question, without stop words, that each one shares with it. Blue: the answer, and the pick of your function when that sentence holds it. Purple: a pick that does not hold the answer. Dashed: the sentence with the answer, when your function picks another one.",
           "kind": "the place of a sentence"},
    "nl": {"title": "Je best_sentence op vier vragen",
           "head": ["", "Zin", "Woorden die de zin met de vraag deelt"],
           "question": "{q} Antwoord: *{a}*.",
           "yours": "je keuze",
           "here": "heeft het antwoord",
           "none": "geen",
           "course": "de best_sentence van de course kiest zin {want}",
           "summary": "Je functie kiest bij {right} van deze {n} vragen de zin met het antwoord.",
           "note": "Grijs: de zinnen, en de woorden van de vraag, zonder stop words, die elke zin met de vraag deelt. Blauw: het antwoord, en de keuze van je functie als deze zin het antwoord heeft. Paars: een keuze zonder het antwoord. Gestippeld: de zin met het antwoord, als je functie een andere zin kiest.",
           "kind": "de plaats van een zin"},
}


def _question_rows(lang: str, i: int, pick: int, w: dict) -> list:
    """The rows of one question: each sentence with the words it shares with the question, the pick and the sentence
    that holds the answer."""
    _, q, ss, start, answers, _, _ = _questions(lang)[i]
    want = sentence_of(ss, start)
    stop = STOP_WORDS[lang]
    rows_ = []
    for k, s in enumerate(ss):
        if k == want:
            offset = start - sum(len(x) for x in ss[:k])
            text_ = _around(s, offset, offset + len(answers[0][0]))
        else:
            text_ = _gray(_short(s))
        shared = shared_words(q, s, stop)
        chips = "".join(_chip(x, _CHIP_DATA) for x in shared) or f'<span style="{_FAINT}">{html.escape(w["none"])}</span>'
        tag = ""
        if k == pick:
            tag = _chip(w["yours"], _CHIP_PRED if k == want else _CHIP_LOSS)
        elif k == want:
            tag = _chip(w["here"], _CHIP_NONE)
        rows_.append([f'<span style="{_FAINT}">{k}</span>', text_ + (f"<div>{tag}</div>" if tag else ""), chips])
    return rows_


def pick_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICK_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    places = [find(lang, qid) for qid in PICKED]
    qs = _questions(lang)
    mine = _call(lambda: [fn(qs[i][1], list(qs[i][2]), set(STOP_WORDS[lang])) for i in places])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_place(m, len(qs[i][2])) for m, i in zip(mine, places)):
        return _empty(w, "odd")
    parts, right_ = [], 0
    for m, i in zip(mine, places):
        _, q, ss, start, answers, _, _ = qs[i]
        right_ += m == sentence_of(ss, start)
        parts.append(_label(w["question"].format(q=q, a=answers[0][0])))
        parts.append(_table(_question_rows(lang, i, m, w), w["head"]))
        want = best_sentence(q, ss, STOP_WORDS[lang])
        if m != want:
            parts.append(_note(w["course"].format(want=want)))
    body = ("".join(parts)
            + f'<div style="{_UI}">{html.escape(w["summary"].format(right=right_, n=len(places)))}</div>'
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 3: your reader beside two baselines ----------------------------------------------------

_SHARE_TEXT = {
    "en": {"title": "How often the picked sentence holds the answer, over all {n} questions",
           "random": "A sentence picked at random (on average)",
           "first": "The first sentence of the paragraph",
           "kept": "The course's reader with the stop words kept",
           "yours": "Your share_right, with the stop words left out",
           "summary": "Your share_right gives {share}: the sentence your reader picks holds the answer in {right} of the {n} questions.",
           "differs": "The course's share_right gives {want} here; yours gives {share}.",
           "note": "Gray: the share of two baselines and of the course's reader with the stop words kept. Blue: the share your share_right gives.",
           "kind": "a number: a share between 0 and 1"},
    "nl": {"title": "Hoe vaak de gekozen zin het antwoord heeft, over alle {n} vragen",
           "random": "Een willekeurige zin (gemiddeld)",
           "first": "De eerste zin van de alinea",
           "kept": "De reader van de course, met de stop words erin",
           "yours": "Je share_right, zonder de stop words",
           "summary": "Je share_right geeft {share}: de zin die je reader kiest, heeft bij {right} van de {n} vragen het antwoord.",
           "differs": "De share_right van de course geeft hier {want}; de jouwe geeft {share}.",
           "note": "Grijs: het aandeel van twee baselines en van de reader van de course met de stop words erin. Blauw: het aandeel dat je share_right geeft.",
           "kind": "een getal: een aandeel tussen 0 en 1"},
}


def _bar(label: str, share: float, lang: str, color: str) -> str:
    width = max(0.0, min(1.0, share)) * 100
    return (f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0;{_TEXT}">'
            f'<div style="flex:0 0 15rem;max-width:45%;{_FAINT}">{html.escape(label)}</div>'
            f'<div style="flex:1 1 auto;min-width:4rem;background:var(--ea-line);height:0.8rem;border-radius:2px">'
            f'<div style="width:{width:.1f}%;height:100%;background:{color};border-radius:2px"></div></div>'
            f'<div style="flex:0 0 3.5rem;text-align:right;font-variant-numeric:tabular-nums">{_dec(share * 100, lang)}%</div></div>')


def share_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    n_ = numbers(lang)
    total = n_["questions"]
    w = _SHARE_TEXT[lang] | _WHY_EMPTY[lang]
    w = w | {"title": w["title"].format(n=_num(total, lang))}
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    mine = _call(lambda: fn(items(lang), set(STOP_WORDS[lang])))
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if isinstance(mine, bool) or not isinstance(mine, (int, float)):
        return _empty(w, "odd")
    want = n_["right"] / total
    course = "var(--ea-ink-2)"
    bars = (_bar(w["random"], n_["random"] / total, lang, course) + _bar(w["first"], n_["first"] / total, lang, course)
            + _bar(w["kept"], n_["kept"] / total, lang, course)
            + _bar(w["yours"], float(mine), lang, "var(--ea-pred-text)"))
    share = _dec(float(mine), lang, 3)
    if abs(float(mine) - want) < 1e-9:
        summary = w["summary"].format(share=share, right=_num(n_["right"], lang), n=_num(total, lang))
    else:
        summary = w["differs"].format(want=_dec(want, lang, 3), share=share)
    body = bars + f'<div style="{_UI}">{html.escape(summary)}</div>' + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the box: any question that may be shown, with what the course's reader picks ---------------------------------------

_LOOK_TEXT = {
    "en": {"title": "Question {n} of {total}: {q}",
           "answers": "Answers that people marked: {a}.",
           "kind": {"right": "The reader picks the sentence with the answer.",
                    "tie": "A tie: the sentence with the answer shares as many words as an earlier sentence, and the first one wins.",
                    "fewer": "Another sentence shares more words with the question than the sentence with the answer.",
                    "zero": "The sentence with the answer shares no word with the question: it says the same in other words, or points back to an earlier sentence."},
           "none": "none",
           "yours": "the reader's pick",
           "here": "holds the answer",
           "head": ["", "Sentence", "Words it shares with the question"],
           "note": "Gray: the sentences, and the words of the question, without stop words, that each one shares with it. Blue: the first answer, and the pick of the course's reader when that sentence holds it. Purple: a pick that does not hold the answer. Dashed: the sentence with the answer, when the reader picks another one."},
    "nl": {"title": "Vraag {n} van {total}: {q}",
           "answers": "Antwoorden die mensen aanwezen: {a}.",
           "kind": {"right": "De reader kiest de zin met het antwoord.",
                    "tie": "Gelijkspel: de zin met het antwoord deelt evenveel woorden als een eerdere zin, en de eerste wint.",
                    "fewer": "Een andere zin deelt meer woorden met de vraag dan de zin met het antwoord.",
                    "zero": "De zin met het antwoord deelt geen enkel woord met de vraag: de zin zegt hetzelfde in andere woorden, of wijst terug naar een eerdere zin."},
           "none": "geen",
           "yours": "keuze van de reader",
           "here": "heeft het antwoord",
           "head": ["", "Zin", "Woorden die de zin met de vraag deelt"],
           "note": "Grijs: de zinnen, en de woorden van de vraag, zonder stop words, die elke zin met de vraag deelt. Blauw: het eerste antwoord, en de keuze van de reader van de course als deze zin het antwoord heeft. Paars: een keuze zonder het antwoord. Gestippeld: de zin met het antwoord, als de reader een andere zin kiest."},
}

KIND_NAMES = {
    "en": {"all": "All questions", "right": "Right", "tie": "A tie, and the first sentence wins",
           "fewer": "Another sentence shares more words", "zero": "The sentence with the answer shares no word"},
    "nl": {"all": "Alle vragen", "right": "Goed", "tie": "Gelijkspel, en de eerste zin wint",
           "fewer": "Een andere zin deelt meer woorden", "zero": "De zin met het antwoord deelt geen woord"},
}


def kind_options(lang: str) -> dict:
    """The kinds for the dropdown of the box, each with how many questions of that kind may be shown."""
    lang = lang if lang in LANGS else "en"
    return {f"{KIND_NAMES[lang][k]} ({_num(len(shown(lang, k)), lang)})": k for k in ("all", *KINDS)}


def kind_count(lang: str, kind: str) -> int:
    lang = lang if lang in LANGS else "en"
    return len(shown(lang, kind))


def look_picture(lang: str, kind: str, n) -> str:
    """The box: question n (from 1) of the questions of one kind that may be shown, with the sentences of its paragraph,
    the words each one shares with the question, the pick of the course's reader and the sentence with the answer."""
    lang = lang if lang in LANGS else "en"
    w = _LOOK_TEXT[lang]
    places = shown(lang, kind if kind in ("all", *KINDS) else "all")
    n = min(max(int(n), 1), len(places))
    i = places[n - 1]
    _, q, ss, _, answers, _, _ = _questions(lang)[i]
    pick = best_sentence(q, ss, STOP_WORDS[lang])
    said = ", ".join(f"*{t}*" for t in dict.fromkeys(t for t, _ in answers))
    title = w["title"].format(n=n, total=len(places), q=q)
    body = (_label(w["answers"].format(a=said)) + _label(w["kind"][kinds(lang)[i]])
            + _table(_question_rows(lang, i, pick, w), w["head"]) + _note(w["note"]))
    return yours.figure(title, body, alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.STOP_WORDS["en"]`; the function gets a
    copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes, and a long text as its start and its length."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, str):
        if len(v) > 60:
            return _value(v[:40])[:-1] + f'…" ({len(v)} characters)'
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, float):
        return repr(round(v, 4))
    return repr(v)


def _show(v):
    """A value for a message; a long text as its start and how many characters it has, in each language."""
    if isinstance(v, str) and len(v) > 60:
        start = _value(v[:40])[:-1]
        return {"en": f'{start}…", a text of {len(v):,} characters', "nl": f'{start}…", een tekst van {_num(len(v), "nl")} tekens'}
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _stop_val(lang: str) -> _Val:
    return _Val(f'C.STOP_WORDS["{lang}"]', set(STOP_WORDS[lang]))


def _rhine_val(lang: str) -> _Val:
    return _Val(f'C.rhine("{lang}")', rhine(lang))


def _items_val(lang: str, part: str = "") -> _Val:
    every = items(lang)
    if part == "rhine":
        return _Val(f'C.rhine_items("{lang}")', rhine_items(lang))
    if part:
        a, b = (int(x) if x else None for x in part.split(":"))
        return _Val(f'C.items("{lang}")[{part}]', every[a:b])
    return _Val(f'C.items("{lang}")', every)


def _question_val(lang: str, qid: str) -> _Val:
    return _written(_questions(lang)[find(lang, qid)][1])


def _sentences_val(lang: str, qid: str) -> _Val:
    i = find(lang, qid)
    return _Val(f'C.items("{lang}")[{i}][1]', list(_questions(lang)[i][2]))


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

# The small cases the exercise texts show, in English and in Dutch.
TOY_OF = {"en": (["The museum opens at 10:00. ", "It closes at 17:00."], "17:00"),
          "nl": (["Het museum gaat om 10.00 uur open. ", "Om 17.00 uur gaat het weer dicht."], "17.00 uur")}
TOY_PICK = {"en": ("When does the pool open?", ["The library opens at 10:00. ", "The pool opens at 9:00."]),
            "nl": ("Wanneer gaat het zwembad open?", ["De bibliotheek gaat om 10.00 uur open. ", "Het zwembad gaat om 9.00 uur open."])}


def toy_start(lang: str) -> int:
    """Where the answer of the small case of exercise 1 starts in its two sentences."""
    ss, answer = TOY_OF[lang]
    return "".join(ss).index(answer)


class WrongInput(Exception):
    """What the course's best_sentence and sentence_of in the box of exercise 3 raise for an input of the wrong kind."""

    def __init__(self, kind: str):
        super().__init__(f"{kind} got an input of the wrong kind")
        self.kind = kind


def _box_sentence_of(sentences, start):
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise WrongInput("sentence_of-sentences")
    if not isinstance(start, int) or isinstance(start, bool):
        raise WrongInput("sentence_of-start")
    return sentence_of(sentences, start)


def _box_best_sentence(question, sentences, stop_words):
    if not isinstance(question, str):
        raise WrongInput("best_sentence-question")
    if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
        raise WrongInput("best_sentence-sentences")
    if not isinstance(stop_words, (set, frozenset)):
        raise WrongInput("best_sentence-stop")
    return best_sentence(question, sentences, stop_words)


_box_sentence_of.__name__ = "sentence_of"
_box_best_sentence.__name__ = "best_sentence"

_SO_HEAD = "def sentence_of(sentences, start):\n    end = 0\n    for k in range(len(sentences)):\n"
_SO_ADD = "        end = end + len(sentences[k])\n"
_SO_TEST = "        if start < end:\n"
_SO_RET = "            return k\n"


def _so(add: str = _SO_ADD, test: str = _SO_TEST, ret: str = _SO_RET) -> str:
    return _SO_HEAD + add + test + ret


_SO = _so()
_SO_START = _SO_HEAD + "        ...\n"

_BS_HEAD = "def best_sentence(question, sentences, stop_words):\n"
_BS_ASKED = "    asked = set([w for w in words(question) if w not in stop_words])\n"
_BS_MID = "    best = 0\n    most = 0\n    for k in range(len(sentences)):\n"
_BS_COUNT = "        shared = len([w for w in asked if w in words(sentences[k])])\n"
_BS_TEST = "        if shared > most:\n"
_BS_KEEP = "            best = k\n            most = shared\n"


def _bs(asked: str = _BS_ASKED, count: str = _BS_COUNT, test: str = _BS_TEST, keep: str = _BS_KEEP,
        end: str = "    return best\n") -> str:
    return _BS_HEAD + asked + _BS_MID + count + test + keep + end


_BS = _bs()
_BS_START = _bs(asked="    asked = ...\n", count="        shared = ...\n")

_SR_HEAD = "def share_right(items, stop_words):\n    right = 0\n    for question, sentences, start in items:\n"
_SR_TEST = "        if best_sentence(question, sentences, stop_words) == sentence_of(sentences, start):\n"
_SR_ADD = "            right = right + 1\n"
_SR_END = "    return right / len(items)\n"


def _sr(test: str = _SR_TEST, add: str = _SR_ADD, end: str = _SR_END) -> str:
    return _SR_HEAD + test + add + end


_SR = _sr()
_SR_START = _SR_HEAD + "        ...\n" + _SR_END


def _not_int(got, text_) -> bool:
    return not isinstance(got, int) or isinstance(got, bool)


def _not_number(got, text_) -> bool:
    return isinstance(got, bool) or not isinstance(got, (int, float))


def _wrong_input(kind: str):
    return lambda e: isinstance(e, WrongInput) and e.kind == kind


REQUIRED = ["sentence_of", "best_sentence", "share_right", "what_left"]

EXERCISES = {
    "sentence_of": FunctionBox(
        name="sentence_of",
        picture=where_picture,
        function="sentence_of",
        start=_SO_START,
        solution=_SO,
        show=_show,
        visible=[(_written(TOY_OF["en"][0]), _written(toy_start("en"))),
                 (_written(TOY_OF["nl"][0]), _written(toy_start("nl")))],
        hidden=[(_written(["One. ", "Two. ", "Three."]), _written(0)),
                (_written(["One. ", "Two. ", "Three."]), _written(4)),
                (_written(["One. ", "Two. ", "Three."]), _written(5)),
                (_written(["A long first sentence. ", "Short. ", "The last one."]), _written(30)),
                (_written(["A long first sentence. ", "Short. ", "The last one."]), _written(29)),
                (_written(["Only one sentence here."]), _written(7)),
                (_rhine_val("en"), _written(264)), (_rhine_val("nl"), _written(272)), (_rhine_val("en"), _written(83))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-int", matches=_not_int,
                    en="`{call}` gives {got}, and it should give {want}: the place of the sentence in the list, a whole number. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de plaats van de zin in de lijst, een geheel getal. Wat geeft je `return` terug?"),
            Mistake("end", wrong=_so(ret="            return end\n"),
                    en="`{call}` gives {got}, and it should give {want}: where the sentence ends, not its place in the list. Which name holds the place of the sentence?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: waar de zin eindigt, niet de plaats in de lijst. Welke naam bewaart de plaats van de zin?"),
            Mistake("le", wrong=_so(test="        if start <= end:\n"),
                    en="`{call}` gives {got}, and it should give {want}: a start exactly at `end` is the first character of the next sentence, and your function counts it to the sentence before. Which comparison leaves that place out?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een start precies bij `end` is het eerste teken van de volgende zin, en je functie rekent het tot de vorige zin. Welke vergelijking laat deze plaats buiten de zin?"),
            Mistake("no-sum", wrong=_so(add="        end = len(sentences[k])\n"),
                    en="`{call}` gives {got}, and it should give {want}: your `end` is the length of one sentence, not where it ends in the paragraph. Where does sentence `k` end, counted from the start of the paragraph?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `end` is de lengte van één zin, niet waar de zin in de alinea eindigt. Waar eindigt zin `k`, geteld vanaf het begin van de alinea?"),
            Mistake("late", wrong=_SO_HEAD + _SO_TEST + _SO_RET + _SO_ADD,
                    en="`{call}` gives {got}, and it should give {want}: your function compares before it adds sentence `k`, so `end` is still where the sentence before it ends. What should `end` be when you compare?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt voordat zin `k` erbij komt, dus `end` is nog waar de vorige zin eindigt. Wat moet `end` zijn als je vergelijkt?"),
            Mistake("one-up", wrong=_so(ret="            return k + 1\n"),
                    en="`{call}` gives {got}, and it should give {want}: one too many. Places in a list start at 0: which place does the first sentence have?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één te veel. Plaatsen in een lijst beginnen bij 0: welke plaats heeft de eerste zin?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: add the length of sentence `k` to `end`, and give back `k` when `start` comes before `end`.",
                      "nl": "Vervang de `...`: tel de lengte van zin `k` op bij `end`, en geef `k` terug als `start` kleiner is dan `end`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Where does each sentence end in the paragraph, and which is the first one that ends after `start`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Waar eindigt elke zin in de alinea, en welke zin eindigt als eerste na `start`?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work for a start at the first character of a sentence, and in the last sentence?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie ook voor een start bij het eerste teken van een zin, en in de laatste zin?"},
            "correct": {"en": "Right: your function gives the sentence that holds the start of the answer, also at the edge of a sentence and in the paragraph about the Rhine. Why does place 200 belong to the second sentence about the Rhine?",
                        "nl": "Goed: je functie geeft de zin waar het antwoord begint, ook aan de rand van een zin en in de alinea over de Rijn. Waarom hoort plaats 185 bij de tweede zin over de Rijn?"},
        },
    ),
    "best_sentence": FunctionBox(
        name="best_sentence",
        picture=pick_picture,
        function="best_sentence",
        start=_BS_START,
        solution=_BS,
        show=_show,
        names={"words": words},
        visible=[(_written(TOY_PICK["en"][0]), _written(TOY_PICK["en"][1]), _stop_val("en")),
                 (_written(TOY_PICK["nl"][0]), _written(TOY_PICK["nl"][1]), _stop_val("nl"))],
        hidden=[(_written("Where is the pool?"), _written(["Where is the library? It is in the old school. ", "The pool is near the park."]), _stop_val("en")),
                (_written("Is it open?"), _written(["Wit: this island reopens. ", "It is open."]), _stop_val("en")),
                (_written("When does the pool open?"), _written(["The pool is big, the pool is new, the pool is blue. ", "The pool opens at 9:00 and is open late."]), _stop_val("en")),
                (_written("Which pool is the big pool near the park?"), _written(["The pool is old. ", "The big park is new."]), _stop_val("en")),
                (_written("Is the pool open?"), _written(["The pool is closed in winter. ", "In summer the pool is open."]), _stop_val("en")),
                (_written("Which museum is open?"), _written(["No museum today. ", "The museum is closed. ", "Open: only the museum of art."]), _stop_val("en")),
                (_question_val("en", NICKNAME), _rhine_val("en"), _stop_val("en")),
                (_question_val("nl", NICKNAME), _rhine_val("nl"), _stop_val("nl")),
                (_question_val("en", COSTUME), _sentences_val("en", COSTUME), _stop_val("en")),
                (_question_val("nl", JACKSONVILLE), _sentences_val("nl", JACKSONVILLE), _stop_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-int", matches=_not_int,
                    en="`{call}` gives {got}, and it should give {want}: the place of the sentence in the list, a whole number. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de plaats van de zin in de lijst, een geheel getal. Wat geeft je `return` terug?"),
            Mistake("count", wrong=_bs(end="    return most\n"),
                    en="`{call}` gives {got}, and it should give {want}: the number of shared words of the best sentence, not its place in the list. Which of the two names holds the place?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal gedeelde woorden van de beste zin, niet de plaats in de lijst. Welke van de twee namen bewaart de plaats?"),
            Mistake("stop-kept", wrong=_bs(asked="    asked = set(words(question))\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function counts the stop words of the question too, so a sentence with *where*, *is* and *the* can win. How do you leave the words of `stop_words` out of `asked`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt ook de stop words van de vraag, dus een zin met *where*, *is* en *the* kan winnen. Hoe laat je de woorden van `stop_words` weg uit `asked`?"),
            Mistake("list-asked", wrong=_bs(asked="    asked = [w for w in words(question) if w not in stop_words]\n"),
                    en="`{call}` gives {got}, and it should give {want}: your `asked` is a list, so a word that the question repeats, such as *pool*, counts twice. How do you keep each different word of the question once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `asked` is een lijst, dus een woord dat de vraag herhaalt, zoals *pool*, telt twee keer. Hoe houd je elk verschillend woord van de vraag één keer over?"),
            Mistake("split", wrong=_bs(asked="    asked = set([w for w in question.split() if w not in stop_words])\n"),
                    en="`{call}` gives {got}, and it should give {want}: `split` keeps the capitals and the question mark, so *pool?* never matches *pool*. Which function gives the words in small letters, without punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `split` houdt de hoofdletters en het vraagteken, dus *pool?* is nooit gelijk aan *pool*. Welke functie geeft de woorden in kleine letters, zonder leestekens?"),
            Mistake("letters", wrong=_bs(count="        shared = len([w for w in asked if w in sentences[k]])\n"),
                    en="`{call}` gives {got}, and it should give {want}: `in` on a text looks for letters, so *open* is also found inside *reopens*. How do you look among the words of the sentence instead?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `in` op een tekst zoekt naar letters, dus *open* staat ook in *reopens*. Hoe zoek je in plaats daarvan tussen de woorden van de zin?"),
            Mistake("letters-lower", wrong=_bs(count="        shared = len([w for w in asked if w in sentences[k].lower()])\n"),
                    en="`{call}` gives {got}, and it should give {want}: `in` on a text looks for letters, so *open* is also found inside *reopens*. How do you look among the words of the sentence instead?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `in` op een tekst zoekt naar letters, dus *open* staat ook in *reopens*. Hoe zoek je in plaats daarvan tussen de woorden van de zin?"),
            Mistake("repeats", wrong=_bs(count="        shared = len([w for w in words(sentences[k]) if w in asked])\n"),
                    en="`{call}` gives {got}, and it should give {want}: your function counts a word each time it occurs in the sentence, so a sentence that repeats *pool* wins. How do you count each different word once?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt een woord elke keer dat het in de zin staat, dus een zin die *pool* herhaalt, wint. Hoe tel je elk verschillend woord één keer?"),
            Mistake("last-on-tie", wrong=_bs(test="        if shared >= most:\n"),
                    en="`{call}` gives {got}, and it should give {want}: when two sentences share as many words with the question, your function takes the last of them. The exercise asks for the first: which comparison keeps the first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: als twee zinnen evenveel woorden met de vraag delen, neemt je functie de laatste. De oefening vraagt om de eerste: welke vergelijking houdt de eerste vast?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `asked` holds the different words of the question that are not stop words, and `shared` how many of them sentence `k` has.",
                      "nl": "Vervang de twee `...`: `asked` bevat de verschillende woorden van de vraag die geen stop words zijn, en `shared` hoeveel daarvan in zin `k` staan."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which different words of the question, without stop words, does each sentence have, and which sentence has the most?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke verschillende woorden van de vraag, zonder stop words, staan in elke zin, en welke zin heeft er de meeste?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function leave out the stop words, count each different word once, and look among whole words?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Laat je functie de stop words weg, telt de functie elk verschillend woord één keer, en zoekt de functie tussen hele woorden?"},
            "correct": {"en": "Right: your reader picks the sentence that shares the most different words with the question, stop words left out, and the first one on a tie. The picture below shows where that goes wrong. Do you see why?",
                        "nl": "Goed: je reader kiest de zin die de meeste verschillende woorden met de vraag deelt, zonder stop words, en bij gelijkspel de eerste. Het plaatje hieronder laat zien waar dat misgaat. Zie je waarom?"},
        },
    ),
    "share_right": FunctionBox(
        name="share_right",
        picture=share_picture,
        function="share_right",
        start=_SR_START,
        solution=_SR,
        show=_show,
        names={"best_sentence": _box_best_sentence, "sentence_of": _box_sentence_of},
        visible=[(_items_val("en", "rhine"), _stop_val("en")), (_items_val("nl", "rhine"), _stop_val("nl"))],
        hidden=[(_items_val("en"), _stop_val("en")), (_items_val("nl"), _stop_val("nl")),
                (_items_val("en", "0:1"), _stop_val("en")), (_items_val("en", "100:110"), _stop_val("en")),
                (_items_val("nl", "500:520"), _stop_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("of-order", error=_wrong_input("sentence_of-sentences"),
                    en="`{call}` stops: `sentence_of` did not get the list of sentences first. In what order does `sentence_of` want the sentences and the start?",
                    nl="`{call}` stopt: `sentence_of` kreeg niet eerst de lijst met zinnen. In welke volgorde wil `sentence_of` de zinnen en de start?"),
            Mistake("of-start", error=_wrong_input("sentence_of-start"),
                    en="`{call}` stops: the second thing `sentence_of` got is not a whole number. Which name holds where the answer starts?",
                    nl="`{call}` stopt: het tweede wat `sentence_of` kreeg, is geen geheel getal. Welke naam bewaart waar het antwoord begint?"),
            Mistake("best-question", error=_wrong_input("best_sentence-question"),
                    en="`{call}` stops: the first thing `best_sentence` got is not the question. In what order does `best_sentence` want the question, the sentences and the stop words?",
                    nl="`{call}` stopt: het eerste wat `best_sentence` kreeg, is niet de vraag. In welke volgorde wil `best_sentence` de vraag, de zinnen en de stop words?"),
            Mistake("best-sentences", error=_wrong_input("best_sentence-sentences"),
                    en="`{call}` stops: the second thing `best_sentence` got is not the list of sentences. In what order does `best_sentence` want the question, the sentences and the stop words?",
                    nl="`{call}` stopt: het tweede wat `best_sentence` kreeg, is niet de lijst met zinnen. In welke volgorde wil `best_sentence` de vraag, de zinnen en de stop words?"),
            Mistake("best-stop", error=_wrong_input("best_sentence-stop"),
                    en="`{call}` stops: the third thing `best_sentence` got is not a set of stop words. Which name holds them?",
                    nl="`{call}` stopt: het derde wat `best_sentence` kreeg, is geen set met stop words. Welke naam bewaart de stop words?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: a share, a number between 0 and 1. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een aandeel, een getal tussen 0 en 1. Wat geeft je `return` terug?"),
            Mistake("count", wrong=_sr(end="    return right\n"),
                    en="`{call}` gives {got}, and it should give {want}: the number of questions your reader gets right, not the share. What do you divide it by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal vragen dat je reader goed heeft, niet het aandeel. Waardoor deel je het?"),
            Mistake("misses", wrong=_sr(test=_SR_TEST.replace(" == ", " != ")),
                    en="`{call}` gives {got}, and it should give {want}: the share of questions your reader gets wrong. When should `right` go up?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aandeel vragen dat je reader fout heeft. Wanneer moet `right` omhoog?"),
            Mistake("all-words", wrong=_sr(test=_SR_TEST.replace("stop_words)", "set())")),
                    en="`{call}` gives {got}, and it should give {want}: your reader counts the stop words too, as it gets an empty set. Which set should `best_sentence` get?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je reader telt ook de stop words, want de reader krijgt een lege set. Welke set moet `best_sentence` krijgen?"),
            Mistake("inside", wrong=_sr(end="        return right / len(items)\n"),
                    en="`{call}` gives {got}, and it should give {want}: your `return` stands inside the loop, so your function stops after the first question. How far should the `return` be indented?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je `return` staat in de loop, dus je functie stopt na de eerste vraag. Hoe ver moet de `return` inspringen?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...`: when the sentence `best_sentence` picks is the sentence that holds the start of the answer, add one to `right`.",
                      "nl": "Vervang de `...`: als de zin die `best_sentence` kiest de zin is waar het antwoord begint, tel dan één op bij `right`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For which questions is the sentence `best_sentence` picks the one `sentence_of` gives, and what share of the questions is that?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bij welke vragen is de zin die `best_sentence` kiest de zin die `sentence_of` geeft, en welk aandeel van de vragen is dat?"},
            "almost": {"en": "Right for the questions about the Rhine. But `{call}` gives {got}, and it should give {want}. Does your function go over every item before it divides?",
                       "nl": "Goed voor de vragen over de Rijn. Maar `{call}` geeft {got}, en het moet {want} zijn. Gaat je functie langs elk item voordat de functie deelt?"},
            "correct": {"en": "Right: your function measures your reader on every question. How does it do beside the two baselines below?",
                        "nl": "Goed: je functie meet je reader op elke vraag. Hoe doet de reader het naast de twee baselines hieronder?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("sentence_of", _SO_START, "empty", None),
    ("sentence_of", _SO, "correct", None),
    ("sentence_of", _SO_HEAD + "        ...\n" + _SO_TEST + _SO_RET, "wrong", "dots-left"),
    ("sentence_of", _so(ret="            return sentences[k]\n"), "wrong", "not-int"),
    ("sentence_of", _so(ret="            return end\n"), "wrong", "end"),
    ("sentence_of", _so(test="        if start <= end:\n"), "wrong", "le"),
    ("sentence_of", _so(add="        end = len(sentences[k])\n"), "wrong", "no-sum"),
    ("sentence_of", _SO_HEAD + _SO_TEST + _SO_RET + _SO_ADD, "wrong", "late"),
    ("sentence_of", _so(ret="            return k + 1\n"), "wrong", "one-up"),
    ("sentence_of", _so(add="        end += len(sentences[k])\n"), "correct", None),
    ("sentence_of", "def sentence_of(sentences, start):\n    for k in range(len(sentences)):\n"
     "        if start < len(''.join(sentences[:k + 1])):\n            return k\n", "correct", None),
    ("best_sentence", _BS_START, "empty", None),
    ("best_sentence", _BS, "correct", None),
    ("best_sentence", _bs(count="        shared = ...\n"), "wrong", "dots-left"),
    ("best_sentence", _bs(end="    return sentences[best]\n"), "wrong", "not-int"),
    ("best_sentence", _bs(end="    return most\n"), "wrong", "count"),
    ("best_sentence", _bs(asked="    asked = set(words(question))\n"), "wrong", "stop-kept"),
    ("best_sentence", _bs(asked="    asked = [w for w in words(question) if w not in stop_words]\n"), "wrong", "list-asked"),
    ("best_sentence", _bs(asked="    asked = set([w for w in question.split() if w not in stop_words])\n"), "wrong", "split"),
    ("best_sentence", _bs(count="        shared = len([w for w in asked if w in sentences[k]])\n"), "wrong", "letters"),
    ("best_sentence", _bs(count="        shared = len([w for w in asked if w in sentences[k].lower()])\n"), "wrong", "letters-lower"),
    ("best_sentence", _bs(count="        shared = len([w for w in words(sentences[k]) if w in asked])\n"), "wrong", "repeats"),
    ("best_sentence", _bs(test="        if shared >= most:\n"), "wrong", "last-on-tie"),
    ("best_sentence", _bs(asked="    asked = set(words(question)) - stop_words\n"), "correct", None),
    ("best_sentence", _bs(count="        shared = len(asked & set(words(sentences[k])))\n"), "correct", None),
    ("share_right", _SR_START, "empty", None),
    ("share_right", _SR, "correct", None),
    ("share_right", _SR_HEAD + "        if ...:\n" + _SR_ADD + _SR_END, "wrong", "dots-left"),
    ("share_right", _sr(test="        if best_sentence(question, sentences, stop_words) == sentence_of(start, sentences):\n"), "error", "of-order"),
    ("share_right", _sr(test="        if best_sentence(question, sentences, stop_words) == sentence_of(sentences, question):\n"), "error", "of-start"),
    ("share_right", _sr(test="        if best_sentence(sentences, question, stop_words) == sentence_of(sentences, start):\n"), "error", "best-question"),
    ("share_right", _sr(test="        if best_sentence(question, start, stop_words) == sentence_of(sentences, start):\n"), "error", "best-sentences"),
    ("share_right", _sr(test="        if best_sentence(question, sentences, items) == sentence_of(sentences, start):\n"), "error", "best-stop"),
    ("share_right", _sr(end="    return str(right / len(items))\n"), "wrong", "not-number"),
    ("share_right", _sr(end="    return right\n"), "wrong", "count"),
    ("share_right", _sr(test=_SR_TEST.replace(" == ", " != ")), "wrong", "misses"),
    ("share_right", _sr(test=_SR_TEST.replace("stop_words)", "set())")), "wrong", "all-words"),
    ("share_right", _sr(end="        return right / len(items)\n"), "wrong", "inside"),
    ("share_right", _sr(add="            right += 1\n"), "correct", None),
    ("share_right", "def share_right(items, stop_words):\n    hits = [best_sentence(q, s, stop_words) == sentence_of(s, a) "
     "for q, s, a in items]\n    return sum(hits) / len(hits)\n", "correct", None),
]
