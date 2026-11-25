# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-qa-eval: whether an answer equals a reference answer once both are normalized (exact_match);
how many words an answer and a reference answer share, as precision, recall and their F1 score (token_f1); and both
scores over many questions, each the best over all reference answers of a question (squad_scores).

The paragraphs, questions and answers are those of the lesson nlp-extractive-qa (its checks.py is loaded below): the 240
paragraphs of XQuAD (Artetxe, Ruder and Yogatama, ACL 2020; CC-BY-SA-4.0) from the development set of SQuAD 1.1
(Rajpurkar, Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0), with their 1,190 English questions and every answer
that people marked, and the same paragraphs and 1,183 questions in Dutch from SQuAD-NL v1.1 (de Vries, Wieling and
Nissim, EMNLP 2023; CC-BY-SA-4.0), in data/xquad-squad-nl.json.gz. The scores follow SQuAD 1.1: both leave out
punctuation and the articles, exact match counts an answer right when it matches any one of the reference answers, and
F1 treats an answer and a reference answer as bags of words and takes the best over the reference answers. The course
cuts a text into words with the words() of nlp-qa-kinds, which also cuts at an apostrophe, where the script of SQuAD
joins the two parts. In SQuAD-NL most questions keep each different answer once, so the lesson measures how well people
agree on the English questions only. Questions marked as not shown, or with he or she (or their Dutch) in them, are
counted but never shown. Every number in the text and the pictures is computed by the course
(tests/test_nlp_qa_eval.py computes them again)."""
import functools
import html
import importlib.util

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake


def _lesson_checks(lesson: str):
    spec = importlib.util.spec_from_file_location(f"everything_ai_checks_{lesson.replace('-', '_')}_for_qa_eval",
                                                  paths.LESSONS / lesson / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QA = _lesson_checks("nlp-extractive-qa")   # the questions, the sentences of each paragraph and the reader of that lesson
words = QA.words

LANGS = ("en", "nl")
ARTICLES = {"en": frozenset({"a", "an", "the"}), "nl": frozenset({"de", "het", "een"})}

PLAGUE = "5726577f708984140094c301"      # During which years was the plague present in Islamic countries? (the worked example)
TEAM = "5706149552bb891400689884"        # When is the suspended team scheduled to return? 2018 / in 2018
HELMET = "5725d79e89a1e219009abf92"      # What was originally on the spacesuits ...? traditional visor helmet / visor helmet
DROUGHT = "572a0bebaf94a219006aa771"     # How many areas were impacted ... in the 2010 drought? three epicenters / three
HORNIMAN = "5726f1ec708984140094d6ab"    # Which museum would receive items on loans ...? The Horniman / Horniman
WORD_IMAGE = "5726e37ef1498d1400e8eed9"  # Which department houses the works on paper ...? ... department / ... department.
PERCENT = "57115bf350c2381900b54a97"     # What percentage ... has the Energiprojekt AB engine achieved? 27-30 / 27-30%
MATCH_PAIRS = (HORNIMAN, WORD_IMAGE, PERCENT, TEAM, PLAGUE, HELMET)   # the picture under exercise 1
F1_PAIRS = (PLAGUE, HELMET, TEAM, DROUGHT, PERCENT)                    # the picture under exercise 2
WORKED = {"en": "virtually every year between 1500 and 1850", "nl": "tussen 1500 en 1850 vrijwel elk jaar"}

# SQuAD 1.1 on its whole development set and test set (Rajpurkar and others, 2016, table 5): exact match and F1.
PAPER = {"logistic": (40.0, 51.0, 40.4, 51.0), "human": (80.3, 90.5, 77.0, 86.8),
         "sliding": (13.2, 20.2, 12.5, 19.7), "random": (1.1, 4.1, 1.3, 4.3)}


# --- the course's own answers to the exercises ---------------------------------------------------------------------------

def shared(p, a):
    """How many words the two lists share, each word counted as often as it stands in both: shared(["pool", "pool"],
    ["pool"]) gives 1, and shared(["pool", "pool"], ["pool", "big", "pool"]) gives 2."""
    left = list(a)
    same = 0
    for w in p:
        if w in left:
            left.remove(w)
            same = same + 1
    return same


def exact_match(prediction, answer, articles):
    """1 when the words of the two texts, without the articles, are the same words in the same order, else 0."""
    p = [w for w in words(prediction) if w not in articles]
    a = [w for w in words(answer) if w not in articles]
    if p == a:
        return 1
    return 0


def token_f1(prediction, answer, articles):
    """The F1 score of the words of the prediction against the words of the answer, both without the articles: 0 when
    they share no word."""
    p = [w for w in words(prediction) if w not in articles]
    a = [w for w in words(answer) if w not in articles]
    same = shared(p, a)
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(a)
    return 2 * precision * recall / (precision + recall)


def squad_scores(predictions, answer_lists, articles):
    """Exact match and F1 in percent over the questions: for each question the best over its reference answers."""
    em = 0
    f1 = 0
    for i in range(len(predictions)):
        em = em + max([exact_match(predictions[i], a, articles) for a in answer_lists[i]])
        f1 = f1 + max([token_f1(predictions[i], a, articles) for a in answer_lists[i]])
    return 100 * em / len(predictions), 100 * f1 / len(predictions)


def marks(p: list, a: list) -> tuple:
    """For each word of p and of a whether it is one of the words they share, counted the way shared() counts."""
    left = list(a)
    in_p = []
    for w in p:
        in_p.append(w in left)
        if w in left:
            left.remove(w)
    left = list(p)
    in_a = []
    for w in a:
        in_a.append(w in left)
        if w in left:
            left.remove(w)
    return in_p, in_a


def normalized(text: str, lang: str) -> list:
    return [w for w in words(text) if w not in ARTICLES[lang]]


# --- the questions and the answers ---------------------------------------------------------------------------------------

def _questions(lang: str) -> tuple:
    """Every question: (id, question, sentences, start of the first answer, answers, paragraph, shown)."""
    return QA._questions(lang)


def find(lang: str, qid: str) -> int:
    return QA.find(lang, qid)


def answers_of(lang: str, qid: str) -> list:
    """The texts of every answer people marked for one question, in the order of the data."""
    return [t for t, _ in _questions(lang)[find(lang, qid)][4]]


PAIR_AT = {HORNIMAN: (1, 2)}   # The Horniman / Horniman, not the first answer, Horniman Museum


def pair(lang: str, qid: str) -> tuple:
    """Two answers of one question that differ as text: those at the places of PAIR_AT, or else the first two different
    texts among its answers."""
    if qid in PAIR_AT:
        i, j = PAIR_AT[qid]
        return answers_of(lang, qid)[i], answers_of(lang, qid)[j]
    texts = list(dict.fromkeys(answers_of(lang, qid)))
    return texts[0], texts[1]


def answer_lists(lang: str) -> list:
    """For each question the list of the texts of its answers: the reference answers, as a new list each time."""
    return [[t for t, _ in answers] for *_, answers, _, _ in _questions(lang)]


@functools.cache
def _reader(lang: str) -> tuple:
    stop = QA.STOP_WORDS[lang]
    return tuple(ss[QA.best_sentence(q, ss, stop)].strip() for _, q, ss, _, _, _, _ in _questions(lang))


def reader_answers(lang: str) -> list:
    """For each question the sentence that the reader of nlp-extractive-qa picks, as its answer."""
    return list(_reader(lang))


def right_sentences(lang: str) -> list:
    """For each question the sentence that holds the start of its first answer, as its answer."""
    return [ss[QA.sentence_of(ss, start)].strip() for _, _, ss, start, _, _, _ in _questions(lang)]


def first_sentences(lang: str) -> list:
    """For each question the first sentence of its paragraph, as its answer."""
    return [ss[0].strip() for _, _, ss, _, _, _, _ in _questions(lang)]


def people_answers() -> list:
    """For each English question the answer of the second person, as SQuAD measures people."""
    return [answers[1][0] for *_, answers, _, _ in _questions("en")]


def people_lists() -> list:
    """For each English question the answers of the other people: the first and those after the second."""
    return [[answers[0][0]] + [t for t, _ in answers[2:]] for *_, answers, _, _ in _questions("en")]


def shown(lang: str) -> list:
    """The places of the questions that the lesson may show."""
    return QA.shown(lang)


def _differing(lang: str) -> list:
    """The places of the questions whose answers hold at least two different texts."""
    return [i for i, (*_, answers, _, _) in enumerate(_questions(lang)) if len({t for t, _ in answers}) > 1]


def differing_pairs(lang: str) -> tuple:
    """The first two different answer texts of every question that has them, as two lists."""
    qs = _questions(lang)
    firsts, seconds = [], []
    for i in _differing(lang):
        texts = list(dict.fromkeys(t for t, _ in qs[i][4]))
        firsts.append(texts[0])
        seconds.append(texts[1])
    return firsts, seconds


# --- the numbers the text names ----------------------------------------------------------------------------------------------

@functools.cache
def people_numbers() -> dict:
    """How well the second person agrees with the others, on the English questions."""
    preds, lists = people_answers(), people_lists()
    n = len(preds)
    em = [max(exact_match(p, a, ARTICLES["en"]) for a in g) for p, g in zip(preds, lists)]
    f1 = [max(token_f1(p, a, ARTICLES["en"]) for a in g) for p, g in zip(preds, lists)]
    counts = [len(answers) for *_, answers, _, _ in _questions("en")]
    return {
        "questions": n,
        "em": 100 * sum(em) / n,
        "f1": 100 * sum(f1) / n,
        "em_count": sum(em),
        "first_em": 100 * sum(exact_match(p, g[0], ARTICLES["en"]) for p, g in zip(preds, lists)) / n,
        "first_f1": 100 * sum(token_f1(p, g[0], ARTICLES["en"]) for p, g in zip(preds, lists)) / n,
        "raw_em": 100 * sum(any(p == a for a in g) for p, g in zip(preds, lists)) / n,
        "words_em": 100 * sum(any(words(p) == words(a) for a in g) for p, g in zip(preds, lists)) / n,
        "partial": sum(e == 0 and f > 0 for e, f in zip(em, f1)),
        "none": sum(f == 0 for f in f1),
        "three_or_more": sum(c >= 3 for c in counts),
        "fewest": min(counts),
        "most": max(counts),
    }


@functools.cache
def numbers(lang: str) -> dict:
    """The numbers the lesson's text names, for `lang`."""
    qs = _questions(lang)
    counts = [len(answers) for *_, answers, _, _ in qs]
    lists = answer_lists(lang)
    firsts, seconds = differing_pairs(lang)
    out = {
        "questions": len(qs),
        "one_answer": sum(len({t for t, _ in answers}) == 1 for *_, answers, _, _ in qs),
        "single": sum(c == 1 for c in counts),
        "references": sum(counts) / len(qs),
        "differing": len(firsts),
        "differing_match": sum(exact_match(a, b, ARTICLES[lang]) for a, b in zip(firsts, seconds)),
        "shown": len(shown(lang)),
    }
    for key, preds in (("reader", reader_answers(lang)), ("right", right_sentences(lang)),
                       ("first", first_sentences(lang))):
        out[key + "_em"], out[key + "_f1"] = squad_scores(preds, lists, ARTICLES[lang])
    right = right_sentences(lang)
    out["right_words"] = sum(len(normalized(s, lang)) for s in right) / len(qs)
    out["right_recall"] = 100 * sum(max(shared(normalized(s, lang), normalized(a, lang)) / max(len(normalized(a, lang)), 1)
                                        for a in g) for s, g in zip(right, lists)) / len(qs)
    return out


def worked(lang: str) -> dict:
    """The worked example: the long answer to the question about the plague against each of its reference answers."""
    p = normalized(WORKED[lang], lang)
    rows = []
    for a in answers_of(lang, PLAGUE):
        g = normalized(a, lang)
        same = shared(p, g)
        rows.append({"answer": a, "words": g, "same": same, "em": exact_match(WORKED[lang], a, ARTICLES[lang]),
                     "precision": same / len(p), "recall": same / len(g),
                     "f1": token_f1(WORKED[lang], a, ARTICLES[lang])})
    return {"words": p, "rows": rows}


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
_CHIP_NONE = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"


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
    """The start of a long text, with … where it goes on."""
    text_ = text_.strip()
    return text_ if len(text_) <= room else text_[:room].rsplit(" ", 1)[0] + " …"


def _tokens(ws: list, hits: list, none: str) -> str:
    """Words as chips: blue the words that are shared, gray the others; a dashed chip when the text has no word."""
    if not ws:
        return _chip(none, _CHIP_NONE)
    return "".join(_chip(w, _CHIP_PRED if hit else _CHIP_DATA) for w, hit in zip(ws, hits))


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


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


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


# --- one answer against the reference answers of a question: the idea and the box ----------------------------------------

_SCORE_TEXT = {
    "en": {"head": ["Reference answer", "Its words", "EM", "Precision", "Recall", "F1"],
           "answer": "The answer: *{a}*",
           "words": "Its words, without punctuation and articles:",
           "best": "best",
           "none": "no word",
           "total": "Score of this answer: exact match {em}, F1 {f1}, each the best over the reference answers.",
           "note": "Gray: the reference answers, and the words that the answer and a reference answer do not share. Blue: the words they share, and the best exact match and the best F1."},
    "nl": {"head": ["Referentieantwoord", "De woorden", "EM", "Precision", "Recall", "F1"],
           "answer": "Het antwoord: *{a}*",
           "words": "De woorden ervan, zonder leestekens en lidwoorden:",
           "best": "beste",
           "none": "geen woord",
           "total": "Score van dit antwoord: exact match {em}, F1 {f1}, elk de beste over de referentieantwoorden.",
           "note": "Grijs: de referentieantwoorden, en de woorden die het antwoord en een referentieantwoord niet delen. Blauw: de woorden die de twee delen, en de beste exact match en de beste F1."},
}


def _scored(prediction: str, references: list, lang: str) -> str:
    """One answer scored against each reference answer: the shared words in blue, exact match, precision, recall and
    F1 for each, and the best of each in blue."""
    w = _SCORE_TEXT[lang]
    p = normalized(prediction, lang)
    rows_, ems, f1s = [], [], []
    for a in references:
        g = normalized(a, lang)
        same = shared(p, g)
        _, hits = marks(p, g)
        ems.append(exact_match(prediction, a, ARTICLES[lang]))
        f1s.append(token_f1(prediction, a, ARTICLES[lang]))
        precision = same / len(p) if p else 0.0
        recall = same / len(g) if g else 0.0
        rows_.append([_gray(a), _tokens(g, hits, w["none"]), ems[-1], _dec(precision, lang, 2), _dec(recall, lang, 2),
                      f1s[-1]])
    best_em, best_f1 = max(ems), max(f1s)
    for r in rows_:
        r[2] = f'<span style="{_PRED if r[2] == best_em else _DATA}">{r[2]}</span>'
        r[5] = f'<span style="{_PRED if r[5] == best_f1 else _DATA}">{_dec(r[5], lang, 2)}</span>'
    hits_p, _ = marks(p, normalized(references[f1s.index(best_f1)], lang))   # against the reference with the best F1
    return (_label(w["answer"].format(a=prediction)) + _label(w["words"]) + f"<div>{_tokens(p, hits_p, w['none'])}</div>"
            + _table(rows_, w["head"], (2, 3, 4, 5))
            + f'<div style="{_UI}">{html.escape(w["total"].format(em=best_em, f1=_dec(best_f1, lang, 2)))}</div>')


_IDEA_TEXT = {
    "en": {"title": "One answer scored against the reference answers of a question",
           "question": "The question: *{q}*"},
    "nl": {"title": "Eén antwoord gescoord tegen de referentieantwoorden van een vraag",
           "question": "De vraag: *{q}*"},
}


def idea_picture(lang: str) -> str:
    """The long answer of the worked example against the reference answers of the question about the plague."""
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    q = _questions(lang)[find(lang, PLAGUE)][1]
    body = (_label(w["question"].format(q=q)) + _scored(WORKED[lang], answers_of(lang, PLAGUE), lang)
            + _note(_SCORE_TEXT[lang]["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


_BOX_TEXT = {
    "en": {"title": "Question {n} of {total}: {q}",
           "typed": "Your answer, scored against the reference answers that people marked:",
           "reader": "Nothing typed yet, so the box scores the sentence that the reader of *Finding the answer in a passage* picks:"},
    "nl": {"title": "Vraag {n} van {total}: {q}",
           "typed": "Je antwoord, gescoord tegen de referentieantwoorden die mensen aanwezen:",
           "reader": "Er is nog niets getypt, dus het vak scoort de zin die de reader van *Het antwoord vinden in een passage* kiest:"},
}


def box_count(lang: str) -> int:
    lang = lang if lang in LANGS else "en"
    return len(shown(lang))


def box_picture(lang: str, n, typed: str) -> str:
    """The box: question n (from 1) of the questions that may be shown, with what the learner typed, or else the
    sentence the reader of nlp-extractive-qa picks, scored against each reference answer."""
    lang = lang if lang in LANGS else "en"
    w = _BOX_TEXT[lang]
    places = shown(lang)
    n = min(max(int(n), 1), len(places))
    i = places[n - 1]
    q = _questions(lang)[i][1]
    typed = (typed or "").strip()[:300]
    prediction = typed or _reader(lang)[i]
    references = list(dict.fromkeys(t for t, _ in _questions(lang)[i][4]))
    title = w["title"].format(n=n, total=len(places), q=q)
    body = (_label(w["typed"] if typed else w["reader"]) + _scored(prediction, references, lang)
            + _note(_SCORE_TEXT[lang]["note"]))
    return yours.figure(title, body, alt=title)


# --- the picture under exercise 1: answers of people that differ as text -----------------------------------------------

_MATCH_TEXT = {
    "en": {"title": "Your exact_match on answers of two people to the same question",
           "head": ["Question", "One person", "Another person", "Same text?", "Your exact_match"],
           "yes": "yes", "no": "no",
           "course": "the course's exact_match gives {want}",
           "summary": "For {n} questions, people marked at least two different texts. Your exact_match counts the first two as the same answer for {k} of them.",
           "differs": "The course's exact_match counts {want} of them as the same answer; yours counts {k}.",
           "note": "Gray: the questions and the answers. Blue: what your exact_match gives, when it is what the course's exact_match gives. Purple: what differs from the course's exact_match.",
           "kind": "1 or 0"},
    "nl": {"title": "Je exact_match op antwoorden van twee mensen op dezelfde vraag",
           "head": ["Vraag", "Eén persoon", "Een ander", "Zelfde tekst?", "Je exact_match"],
           "yes": "ja", "no": "nee",
           "course": "de exact_match van de course geeft {want}",
           "summary": "Bij {n} vragen wezen mensen minstens twee verschillende teksten aan. Je exact_match rekent bij {k} daarvan de eerste twee als hetzelfde antwoord.",
           "differs": "De exact_match van de course rekent er {want} als hetzelfde antwoord; de jouwe {k}.",
           "note": "Grijs: de vragen en de antwoorden. Blauw: wat je exact_match geeft, als dat is wat de exact_match van de course geeft. Paars: wat anders is dan bij de exact_match van de course.",
           "kind": "1 of 0"},
}


def match_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _MATCH_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    arts = ARTICLES[lang]
    pairs = [pair(lang, qid) for qid in MATCH_PAIRS]
    firsts, seconds = differing_pairs(lang)
    mine = _call(lambda: ([fn(a, b, set(arts)) for a, b in pairs],
                          [fn(a, b, set(arts)) for a, b in zip(firsts, seconds)]))
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    shown_, every = mine
    if not all(_is_number(m) for m in shown_ + every):
        return _empty(w, "odd")
    rows_ = []
    for qid, (a, b), m in zip(MATCH_PAIRS, pairs, shown_):
        want = exact_match(a, b, arts)
        cell = _chip(repr(m), _CHIP_PRED) if m == want else _chip(repr(m), _CHIP_LOSS) + _note(w["course"].format(want=want))
        rows_.append([_gray(_questions(lang)[find(lang, qid)][1]), _chip(a, _CHIP_DATA), _chip(b, _CHIP_DATA),
                      f'<span style="{_FAINT}">{w["yes"] if a == b else w["no"]}</span>', cell])
    k = sum(1 for m in every if m == 1)
    want_k = numbers(lang)["differing_match"]
    right_all = all(m == exact_match(a, b, arts) for m, a, b in zip(every, firsts, seconds))
    summary = w["summary"].format(n=_num(len(firsts), lang), k=_num(k, lang))
    if not right_all:
        summary += " " + w["differs"].format(want=_num(want_k, lang), k=_num(k, lang))
    body = _table(rows_, w["head"]) + f'<div style="{_UI}">{html.escape(summary)}</div>' + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 2: the words two people share -----------------------------------------------------------

_F1_TEXT = {
    "en": {"title": "Your token_f1 on answers of two people that differ",
           "head": ["One person", "Another person", "Precision", "Recall", "Your token_f1"],
           "question": "*{q}*",
           "none": "no word",
           "course": "the course's token_f1 gives {want}",
           "note": "Gray: the questions, and the words that the two answers do not share. Blue: the words they share, and what your token_f1 gives when it is what the course's token_f1 gives. Purple: a value that differs from the course's token_f1. Precision and recall are for the first answer against the second.",
           "kind": "a number"},
    "nl": {"title": "Je token_f1 op antwoorden van twee mensen die verschillen",
           "head": ["Eén persoon", "Een ander", "Precision", "Recall", "Je token_f1"],
           "question": "*{q}*",
           "none": "geen woord",
           "course": "de token_f1 van de course geeft {want}",
           "note": "Grijs: de vragen, en de woorden die de twee antwoorden niet delen. Blauw: de gedeelde woorden, en wat je token_f1 geeft als dat is wat de token_f1 van de course geeft. Paars: een waarde die anders is dan bij de token_f1 van de course. Precision en recall gelden voor het eerste antwoord tegen het tweede.",
           "kind": "een getal"},
}


def f1_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _F1_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    arts = ARTICLES[lang]
    pairs = [pair(lang, qid) for qid in F1_PAIRS]
    mine = _call(lambda: [fn(a, b, set(arts)) for a, b in pairs])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_number(m) for m in mine):
        return _empty(w, "odd")
    rows_ = []
    for qid, (a, b), m in zip(F1_PAIRS, pairs, mine):
        p, g = normalized(a, lang), normalized(b, lang)
        hits_p, hits_g = marks(p, g)
        same = shared(p, g)
        want = token_f1(a, b, arts)
        if abs(float(m) - want) < 1e-9:
            cell = _chip(_dec(m, lang, 3), _CHIP_PRED)
        else:
            cell = _chip(_dec(m, lang, 3), _CHIP_LOSS) + _note(w["course"].format(want=_dec(want, lang, 3)))
        question = f'<div style="{_UI}">{_marked(w["question"].format(q=_questions(lang)[find(lang, qid)][1]))}</div>'
        rows_.append([_gray(a) + f"<div>{_tokens(p, hits_p, w['none'])}</div>" + question,
                      _gray(b) + f"<div>{_tokens(g, hits_g, w['none'])}</div>",
                      f'<span style="{_DATA}">{same}/{len(p)}</span>', f'<span style="{_DATA}">{same}/{len(g)}</span>', cell])
    body = _table(rows_, w["head"], (2, 3, 4)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["title"])


# --- the picture under exercise 3: the scores of people and of whole sentences -------------------------------------------

_SCORES_TEXT = {
    "en": {"title": "Your squad_scores on the {n} English questions",
           "head": ["Answers", "Exact match", "F1"],
           "people": "People: the second answer against the others (the English questions)",
           "reader": "The sentence that the reader of *Finding the answer in a passage* picks",
           "right": "The sentence that holds the answer",
           "first": "The first sentence of the paragraph",
           "paper": "In the paper of SQuAD, on its whole development set:",
           "human": "People",
           "logistic": "Logistic regression",
           "course": "the course's squad_scores gives {em} and {f1}",
           "note": "Blue: what your squad_scores gives, when it is what the course's squad_scores gives. Purple: what differs from the course's squad_scores. Gray: the scores in the paper of SQuAD, by Rajpurkar and others.",
           "kind": "two numbers, exact match and F1"},
    "nl": {"title": "Je squad_scores op de {n} Nederlandse vragen",
           "head": ["Antwoorden", "Exact match", "F1"],
           "people": "Mensen: het tweede antwoord tegen de andere (de Engelse vragen)",
           "reader": "De zin die de reader van *Het antwoord vinden in een passage* kiest",
           "right": "De zin met het antwoord",
           "first": "De eerste zin van de alinea",
           "paper": "In het paper van SQuAD, op de hele development set:",
           "human": "Mensen",
           "logistic": "Logistic regression",
           "course": "de squad_scores van de course geeft {em} en {f1}",
           "note": "Blauw: wat je squad_scores geeft, als dat is wat de squad_scores van de course geeft. Paars: wat anders is dan bij de squad_scores van de course. Grijs: de scores in het paper van SQuAD, van Rajpurkar en anderen.",
           "kind": "twee getallen, exact match en F1"},
}


def _pct(v, lang: str) -> str:
    return _dec(v, lang) + "%"


def _runs(lang: str) -> list:
    """For each kind of answer in the picture: (key, the answers, their reference answers, the language)."""
    out = [("people", people_answers(), people_lists(), "en")]
    for key, make in (("reader", reader_answers), ("right", right_sentences), ("first", first_sentences)):
        out.append((key, make(lang), answer_lists(lang), lang))
    return out


def scores_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SCORES_TEXT[lang] | _WHY_EMPTY[lang]
    w = w | {"title": w["title"].format(n=_num(numbers(lang)["questions"], lang))}
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    runs = _runs(lang)
    mine = _call(lambda: [fn(preds, lists, set(ARTICLES[at])) for _, preds, lists, at in runs])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(isinstance(m, tuple) and len(m) == 2 and all(_is_number(x) for x in m) for m in mine):
        return _empty(w, "odd")
    rows_ = []
    for (key, _, _, at), m in zip(runs, mine):
        if key == "people":
            want = (people_numbers()["em"], people_numbers()["f1"])
        else:
            want = (numbers(at)[key + "_em"], numbers(at)[key + "_f1"])
        same = all(abs(float(x) - y) < 1e-6 for x, y in zip(m, want))
        style = _PRED if same else _LOSS
        cells = [f'<span style="{style}">{html.escape(_pct(x, lang))}</span>' for x in m]
        label = f'<span style="{_DATA}">{_marked(w[key])}</span>'
        if not same:
            label += _note(w["course"].format(em=_pct(want[0], lang), f1=_pct(want[1], lang)))
        rows_.append([label] + cells)
    paper = [[_gray(w[key]), _gray(_pct(PAPER[key][0], lang)), _gray(_pct(PAPER[key][1], lang))]
             for key in ("human", "logistic")]
    body = (_table(rows_, w["head"], (1, 2)) + _label(w["paper"]) + _table(paper, w["head"], (1, 2))
            + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["title"])


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.ARTICLES["en"]`; the function gets a
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
    """A value for a message: a number with at most four decimals, a pair of them, or text with double quotes."""
    return _value(v)


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _articles_val(lang: str) -> _Val:
    return _Val(f'C.ARTICLES["{lang}"]', set(ARTICLES[lang]))


def _lists_val(lang: str, part: str = "") -> _Val:
    every = answer_lists(lang)
    if part:
        a, b = (int(x) if x else None for x in part.split(":"))
        return _Val(f'C.answer_lists("{lang}")[{part}]', every[a:b])
    return _Val(f'C.answer_lists("{lang}")', every)


def _preds_val(name: str, lang: str, part: str = "") -> _Val:
    every = {"reader_answers": reader_answers, "right_sentences": right_sentences,
             "first_sentences": first_sentences}[name](lang)
    if part:
        a, b = (int(x) if x else None for x in part.split(":"))
        return _Val(f'C.{name}("{lang}")[{part}]', every[a:b])
    return _Val(f'C.{name}("{lang}")', every)


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
TOY_MATCH = {"en": ("The pool.", "pool"), "nl": ("Het zwembad.", "zwembad")}
TOY_F1 = {"en": ("at 9:00", "9:00"), "nl": ("om 9 uur", "9 uur")}
TOY_SCORES = {"en": (["at 9:00", "the pool"], [["9:00"], ["the big pool", "pool"]]),
              "nl": (["om 9 uur", "het zwembad"], [["9 uur"], ["het grote zwembad", "zwembad"]])}


class WrongInput(Exception):
    """What the course's exact_match and token_f1 in the box of exercise 3 raise for an input of the wrong kind."""

    def __init__(self, kind: str):
        super().__init__(f"{kind} got an input of the wrong kind")
        self.kind = kind


def _checked(name: str, prediction, answer, articles):
    if not isinstance(prediction, str):
        raise WrongInput(f"{name}-prediction")
    if not isinstance(answer, str):
        raise WrongInput(f"{name}-answer")
    if not isinstance(articles, (set, frozenset)):
        raise WrongInput(f"{name}-articles")


def _box_exact_match(prediction, answer, articles):
    _checked("exact_match", prediction, answer, articles)
    return exact_match(prediction, answer, articles)


def _box_token_f1(prediction, answer, articles):
    _checked("token_f1", prediction, answer, articles)
    return token_f1(prediction, answer, articles)


_box_exact_match.__name__ = "exact_match"
_box_token_f1.__name__ = "token_f1"

_EM_HEAD = "def exact_match(prediction, answer, articles):\n"
_EM_P = "    p = [w for w in words(prediction) if w not in articles]\n"
_EM_A = "    a = [w for w in words(answer) if w not in articles]\n"
_EM_END = "    if p == a:\n        return 1\n    return 0\n"


def _em(p: str = _EM_P, a: str = _EM_A) -> str:
    return _EM_HEAD + p + a + _EM_END


def _em_both(old: str, new: str) -> str:
    """exact_match with the same change in the line of `p` and the line of `a`."""
    return _em(_EM_P.replace(old.format(t="prediction"), new.format(t="prediction")),
               _EM_A.replace(old.format(t="answer"), new.format(t="answer")))


_EM = _em()
_EM_START = _em("    p = ...\n", "    a = ...\n")
_EM_RAW = _em("    p = prediction\n", "    a = answer\n")
_EM_LOWER = _em("    p = prediction.lower()\n", "    a = answer.lower()\n")
_EM_KEPT = _em("    p = words(prediction)\n", "    a = words(answer)\n")
_EM_SPLIT = _em_both("words({t})", "{t}.split()")
_EM_SPLIT_LOWER = _em_both("words({t})", "{t}.lower().split()")
_EM_SET = _em("    p = set([w for w in words(prediction) if w not in articles])\n",
              "    a = set([w for w in words(answer) if w not in articles])\n")
_EM_ONLY = _em_both("not in articles", "in articles")

_F1_HEAD = ("def token_f1(prediction, answer, articles):\n" + _EM_P + _EM_A + "    same = shared(p, a)\n"
            "    if same == 0:\n        return 0.0\n")
_F1_P = "    precision = same / len(p)\n"
_F1_R = "    recall = same / len(a)\n"
_F1_RET = "    return 2 * precision * recall / (precision + recall)\n"


def _f1(p: str = _F1_P, r: str = _F1_R, ret: str = _F1_RET) -> str:
    return _F1_HEAD + p + r + ret


_F1 = _f1()
_F1_START = _f1("    precision = ...\n", "    recall = ...\n", "    return ...\n")

_SC_HEAD = "def squad_scores(predictions, answer_lists, articles):\n    em = 0\n    f1 = 0\n    for i in range(len(predictions)):\n"
_SC_EM = "        em = em + max([exact_match(predictions[i], a, articles) for a in answer_lists[i]])\n"
_SC_F1 = "        f1 = f1 + max([token_f1(predictions[i], a, articles) for a in answer_lists[i]])\n"
_SC_END = "    return 100 * em / len(predictions), 100 * f1 / len(predictions)\n"


def _sc(em: str = _SC_EM, f1: str = _SC_F1) -> str:
    return _SC_HEAD + em + f1 + _SC_END


_SC = _sc()
_SC_START = _sc("        em = ...\n", "        f1 = ...\n")
_SC_FIRST = _sc("        em = em + exact_match(predictions[i], answer_lists[i][0], articles)\n",
                "        f1 = f1 + token_f1(predictions[i], answer_lists[i][0], articles)\n")
_SC_MIN = _sc(_SC_EM.replace("max(", "min("), _SC_F1.replace("max(", "min("))
_SC_EVERY = (_SC_HEAD + "        for a in answer_lists[i]:\n"
             "            em = em + exact_match(predictions[i], a, articles)\n"
             "            f1 = f1 + token_f1(predictions[i], a, articles)\n" + _SC_END)
_SC_COPIED = _sc(f1=_SC_F1.replace("token_f1(", "exact_match("))


def _not_int(got, text_) -> bool:
    return not isinstance(got, int) or isinstance(got, bool)


def _not_number(got, text_) -> bool:
    return isinstance(got, bool) or not isinstance(got, (int, float))


def _not_pair(got, text_) -> bool:
    return not (isinstance(got, tuple) and len(got) == 2 and all(_is_number(x) for x in got))


def _wrong_input(kind: str):
    return lambda e: isinstance(e, WrongInput) and e.kind == kind


REQUIRED = ["exact_match", "token_f1", "squad_scores", "why_best"]

EXERCISES = {
    "exact_match": FunctionBox(
        name="exact_match",
        picture=match_picture,
        function="exact_match",
        start=_EM_START,
        solution=_EM,
        show=_show,
        names={"words": words},
        visible=[(_written(TOY_MATCH["en"][0]), _written(TOY_MATCH["en"][1]), _articles_val("en")),
                 (_written(TOY_MATCH["nl"][0]), _written(TOY_MATCH["nl"][1]), _articles_val("nl"))],
        hidden=[(_written("1850 and 1500"), _written("1500 and 1850"), _articles_val("en")),
                (_written("in 2018"), _written("2018"), _articles_val("en")),
                (_written("A pool"), _written("The pool"), _articles_val("en")),
                (_written("Pool"), _written("pool"), _articles_val("en")),
                (_written("Rhine Delta."), _written("rhine delta"), _articles_val("en")),
                (_written("pool pool"), _written("pool"), _articles_val("en")),
                (_written("THE END"), _written("end"), _articles_val("en")),
                (_written("the theater"), _written("theater"), _articles_val("en")),
                (_written("De Rijn-Maasdelta"), _written("Rijn Maasdelta"), _articles_val("nl")),
                (_written("een van de meest voorkomende"), _written("meest voorkomende"), _articles_val("nl")),
                (_written("27-30%"), _written("27-30"), _articles_val("en"))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-int", matches=_not_int,
                    en="`{call}` gives {got}, and it should give {want}: 1 or 0. What do the lines after `p` and `a` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: 1 of 0. Wat geven de regels na `p` en `a` terug?"),
            Mistake("raw", wrong=_EM_RAW,
                    en="`{call}` gives {got}, and it should give {want}: your function compares the texts as they are, so the capital, the full stop and *the* of *The pool.* make it differ from *pool*. Which function gives the words of a text in small letters, without punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt de teksten zoals de teksten er staan, dus de hoofdletter, de punt en *the* van *The pool.* maken het anders dan *pool*. Welke functie geeft de woorden van een tekst in kleine letters, zonder leestekens?"),
            Mistake("lower-only", wrong=_EM_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: `lower` takes away the capital of *The pool.*, but the full stop and *the* stay. Which function gives the words of a text in small letters, without punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `lower` haalt de hoofdletter van *The pool.* weg, maar de punt en *the* blijven staan. Welke functie geeft de woorden van een tekst in kleine letters, zonder leestekens?"),
            Mistake("kept-articles", wrong=_EM_KEPT,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps the articles, so *the* makes *The pool.* differ from *pool*. How do you leave the words of `articles` out?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie houdt de lidwoorden, dus *the* maakt *The pool.* anders dan *pool*. Hoe laat je de woorden van `articles` weg?"),
            Mistake("split", wrong=_EM_SPLIT,
                    en="`{call}` gives {got}, and it should give {want}: `split` keeps the capital and the full stop, so *The* is not in `articles`, and *pool.* is not *pool*. Which function gives the words in small letters, without punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `split` houdt de hoofdletter en de punt, dus *The* staat niet in `articles`, en *pool.* is niet *pool*. Welke functie geeft de woorden in kleine letters, zonder leestekens?"),
            Mistake("split-lower", wrong=_EM_SPLIT_LOWER,
                    en="`{call}` gives {got}, and it should give {want}: `lower` and `split` take away the capital, but the full stop stays on *pool.*, so it is not *pool*. Which function gives the words without punctuation?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `lower` en `split` halen de hoofdletter weg, maar de punt blijft aan *pool.* vastzitten, dus het is niet *pool*. Welke functie geeft de woorden zonder leestekens?"),
            Mistake("set", wrong=_EM_SET,
                    en="`{call}` gives {got}, and it should give {want}: your function compares sets, so *1850 and 1500* counts as *1500 and 1850*, though its words stand in another order. What keeps the order of the words, and a word that comes twice?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie vergelijkt sets, dus *1850 and 1500* telt als *1500 and 1850*, al staan de woorden in een andere volgorde. Wat houdt de volgorde van de woorden, en een woord dat twee keer staat?"),
            Mistake("only-articles", wrong=_EM_ONLY,
                    en="`{call}` gives {got}, and it should give {want}: your function keeps only the articles, the *the* of *The pool.*, and leaves out every other word. Which words should the list comprehension keep?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie houdt alleen de lidwoorden, de *the* van *The pool.*, en laat elk ander woord weg. Welke woorden moet de list comprehension houden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `p` holds the words of `prediction` and `a` the words of `answer`, each without the words in `articles`.",
                      "nl": "Vervang de twee `...`: `p` bevat de woorden van `prediction` en `a` de woorden van `answer`, elk zonder de woorden in `articles`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. What are the words of each text in small letters, without punctuation and without the articles, and are they the same?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wat zijn de woorden van elke tekst in kleine letters, zonder leestekens en zonder lidwoorden, en zijn de twee lijsten gelijk?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function keep the order of the words, and leave out only the articles?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Houdt je functie de volgorde van de woorden, en laat de functie alleen de lidwoorden weg?"},
            "correct": {"en": "Right: your function counts two texts as the same answer when their words are the same once capitals, punctuation and articles are gone. Why is *in 2018* still not *2018*?",
                        "nl": "Goed: je functie rekent twee teksten als hetzelfde antwoord als de woorden gelijk zijn zodra hoofdletters, leestekens en lidwoorden weg zijn. Waarom is *in 2018* nog steeds niet *2018*?"},
        },
    ),
    "token_f1": FunctionBox(
        name="token_f1",
        picture=f1_picture,
        function="token_f1",
        start=_F1_START,
        solution=_F1,
        show=_show,
        names={"words": words, "shared": shared},
        visible=[(_written(TOY_F1["en"][0]), _written(TOY_F1["en"][1]), _articles_val("en")),
                 (_written(TOY_F1["nl"][0]), _written(TOY_F1["nl"][1]), _articles_val("nl"))],
        hidden=[(_written("2005 and 2010"), _written("between 2005 and 2010"), _articles_val("en")),
                (_written("between 2005 and 2010"), _written("2005 and 2010"), _articles_val("en")),
                (_written("the pool"), _written("pool"), _articles_val("en")),
                (_written("in the park"), _written("near the pool"), _articles_val("en")),
                (_written("pool pool"), _written("pool"), _articles_val("en")),
                (_written(WORKED["en"]), _written("between 1500 and 1850"), _articles_val("en")),
                (_written(WORKED["nl"]), _written("1500 en 1850"), _articles_val("nl")),
                (_written("een van de meest voorkomende"), _written("meest voorkomende"), _articles_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, the F1 score. What does your `return` give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, de F1 score. Wat geeft je `return` terug?"),
            Mistake("precision-only", wrong=_f1(ret="    return precision\n"),
                    en="`{call}` gives {got}, and it should give {want}: the precision, the share of the words of *at 9:00* that *9:00* has too. The F1 score combines it with the recall: how?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de precision, het deel van de woorden van *at 9:00* dat ook in *9:00* staat. De F1 score combineert de precision met de recall: hoe?"),
            Mistake("recall-only", wrong=_f1(ret="    return recall\n"),
                    en="`{call}` gives {got}, and it should give {want}: the recall, the share of the words of *9:00* that *at 9:00* has too. The F1 score combines it with the precision: how?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de recall, het deel van de woorden van *9:00* dat ook in *at 9:00* staat. De F1 score combineert de recall met de precision: hoe?"),
            Mistake("mean", wrong=_f1(ret="    return (precision + recall) / 2\n"),
                    en="`{call}` gives {got}, and it should give {want}: the plain mean of precision and recall. The F1 score is their harmonic mean: twice their product, divided by what?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het gewone gemiddelde van precision en recall. De F1 score is het harmonisch gemiddelde: twee keer het product, gedeeld door wat?"),
            Mistake("chars", wrong=_f1("    precision = same / len(prediction)\n", "    recall = same / len(answer)\n"),
                    en="`{call}` gives {got}, and it should give {want}: `len` of a text counts its characters, 7 for *at 9:00*, and not its words. Which names hold the lists of words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `len` van een tekst telt de tekens, 7 voor *at 9:00*, en niet de woorden. Welke namen bewaren de lijsten met woorden?"),
            Mistake("brackets", wrong=_f1(ret="    return 2 * precision * recall / precision + recall\n"),
                    en="`{call}` gives {got}, and it should give {want}, more than 1: Python divides before it adds, so your function divides by `precision` alone and adds `recall` after. Where do the brackets go?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn, meer dan 1: Python deelt eerst en telt daarna op, dus je functie deelt door `precision` alleen en telt `recall` erna op. Waar komen de haakjes?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the three `...`: `precision` is the share of the words of `p` that are shared, `recall` the share of the words of `a`, and the `return` gives their F1 score.",
                      "nl": "Vervang de drie `...`: `precision` is het deel van de woorden van `p` dat gedeeld is, `recall` het deel van de woorden van `a`, en de `return` geeft de F1 score van precision en recall."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many words do the two share, what share of each list is that, and what is the harmonic mean of those two shares?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoeveel woorden delen de twee, welk deel van elke lijst is dat, en wat is het harmonisch gemiddelde van deze twee delen?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function divide the shared words by the length of `p` for the precision and of `a` for the recall?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Deelt je functie de gedeelde woorden door de lengte van `p` voor de precision en van `a` voor de recall?"},
            "correct": {"en": "Right: your function gives partial credit for the words two answers share. Which answer gets 1, and which gets 0?",
                        "nl": "Goed: je functie geeft een deel van de punten voor de woorden die in beide antwoorden staan. Welk antwoord krijgt 1, en welk 0?"},
        },
    ),
    "squad_scores": FunctionBox(
        name="squad_scores",
        picture=scores_picture,
        function="squad_scores",
        start=_SC_START,
        solution=_SC,
        show=_show,
        names={"exact_match": _box_exact_match, "token_f1": _box_token_f1},
        visible=[(_written(TOY_SCORES["en"][0]), _written(TOY_SCORES["en"][1]), _articles_val("en")),
                 (_written(TOY_SCORES["nl"][0]), _written(TOY_SCORES["nl"][1]), _articles_val("nl"))],
        hidden=[(_written(["1500 and 1850"]), _written([["between 1500 and 1850", "1500 and 1850", "1500 and 1850"]]),
                 _articles_val("en")),
                (_written(["pool", "the big pool", "park"]), _written([["pool"], ["pool", "the pool"], ["the pool"]]),
                 _articles_val("en")),
                (_preds_val("reader_answers", "en"), _lists_val("en"), _articles_val("en")),
                (_preds_val("reader_answers", "nl"), _lists_val("nl"), _articles_val("nl")),
                (_Val("C.people_answers()", people_answers()), _Val("C.people_lists()", people_lists()), _articles_val("en")),
                (_preds_val("right_sentences", "nl", "100:200"), _lists_val("nl", "100:200"), _articles_val("nl"))],
        first=[_DOTS],
        mistakes=[
            Mistake("em-answers", error=_wrong_input("exact_match-answer"),
                    en="`{call}` stops: `exact_match` got something other than one reference answer, such as the whole list of them. Which name holds one reference answer at a time?",
                    nl="`{call}` stopt: `exact_match` kreeg iets anders dan één referentieantwoord, zoals de hele lijst ervan. Welke naam bewaart één referentieantwoord tegelijk?"),
            Mistake("f1-answers", error=_wrong_input("token_f1-answer"),
                    en="`{call}` stops: `token_f1` got something other than one reference answer, such as the whole list of them. Which name holds one reference answer at a time?",
                    nl="`{call}` stopt: `token_f1` kreeg iets anders dan één referentieantwoord, zoals de hele lijst ervan. Welke naam bewaart één referentieantwoord tegelijk?"),
            Mistake("em-predictions", error=_wrong_input("exact_match-prediction"),
                    en="`{call}` stops: the first thing `exact_match` got is not the answer of one question, such as all of `predictions`. Which answer belongs to question `i`?",
                    nl="`{call}` stopt: het eerste wat `exact_match` kreeg, is niet het antwoord op één vraag, maar bijvoorbeeld heel `predictions`. Welk antwoord hoort bij vraag `i`?"),
            Mistake("f1-predictions", error=_wrong_input("token_f1-prediction"),
                    en="`{call}` stops: the first thing `token_f1` got is not the answer of one question, such as all of `predictions`. Which answer belongs to question `i`?",
                    nl="`{call}` stopt: het eerste wat `token_f1` kreeg, is niet het antwoord op één vraag, maar bijvoorbeeld heel `predictions`. Welk antwoord hoort bij vraag `i`?"),
            Mistake("not-pair", matches=_not_pair,
                    en="`{call}` gives {got}, and it should give {want}: two numbers, exact match and F1 in percent. Is the `return` of the start code still there?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: twee getallen, exact match en F1 in procenten. Staat de `return` van de startcode er nog?"),
            Mistake("first-answer", wrong=_SC_FIRST,
                    en="`{call}` gives {got}, and it should give {want}: your function scores *the pool* only against the first reference answer, *the big pool*, and not against *pool*, which it matches exactly. What should count, of all the reference answers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie scoort *the pool* alleen tegen het eerste referentieantwoord, *the big pool*, en niet tegen *pool*, waar het precies mee overeenkomt. Wat moet tellen, van alle referentieantwoorden?"),
            Mistake("min", wrong=_SC_MIN,
                    en="`{call}` gives {got}, and it should give {want}: your function takes the lowest score over the reference answers, so *the pool* gets the score against *the big pool*, not against *pool*. Which score of a question should count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie neemt de laagste score over de referentieantwoorden, dus *the pool* krijgt de score tegen *the big pool*, niet tegen *pool*. Welke score van een vraag moet tellen?"),
            Mistake("every-answer", wrong=_SC_EVERY,
                    en="`{call}` gives {got}, and it should give {want}, an F1 above 100: your function adds up the score against every reference answer, so a question with two of them counts twice. How do you keep one score per question?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn, een F1 boven de 100: je functie telt de score tegen elk referentieantwoord op, dus een vraag met twee ervan telt twee keer. Hoe houd je één score per vraag?"),
            Mistake("copied", wrong=_SC_COPIED,
                    en="`{call}` gives {got}, and it should give {want}: your F1 is the same as your exact match, as both lines use `exact_match`. Which function gives the F1 of an answer?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je F1 is gelijk aan je exact match, want beide regels gebruiken `exact_match`. Welke functie geeft de F1 van een antwoord?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: for question `i`, add to `em` the best exact match of its answer over its reference answers, and to `f1` the best F1.",
                      "nl": "Vervang de twee `...`: tel voor vraag `i` bij `em` de beste exact match van het antwoord over de referentieantwoorden op, en bij `f1` de beste F1."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each question, what is the best score of its answer over all its reference answers, and what is the mean of those over the questions?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Wat is voor elke vraag de beste score van het antwoord over alle referentieantwoorden, en wat is het gemiddelde daarvan over de vragen?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function take the best over all the reference answers of a question, however many there are?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Neemt je functie de beste over alle referentieantwoorden van een vraag, hoeveel het er ook zijn?"},
            "correct": {"en": "Right: your function scores answers the way SQuAD does. How far is a whole sentence from the people, in the picture below?",
                        "nl": "Goed: je functie scoort antwoorden zoals SQuAD dat doet. Hoe ver staat een hele zin van de mensen af, in het plaatje hieronder?"},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, code, state, known mistake).
CASES = [
    ("exact_match", _EM_START, "empty", None),
    ("exact_match", _EM, "correct", None),
    ("exact_match", _em(a="    a = ...\n"), "wrong", "dots-left"),
    ("exact_match", _EM_HEAD + _EM_P + _EM_A + "    return p == a\n", "wrong", "not-int"),
    ("exact_match", _EM_RAW, "wrong", "raw"),
    ("exact_match", _EM_LOWER, "wrong", "lower-only"),
    ("exact_match", _EM_KEPT, "wrong", "kept-articles"),
    ("exact_match", _EM_SPLIT, "wrong", "split"),
    ("exact_match", _EM_SPLIT_LOWER, "wrong", "split-lower"),
    ("exact_match", _EM_SET, "wrong", "set"),
    ("exact_match", _EM_ONLY, "wrong", "only-articles"),
    ("exact_match", _EM_HEAD + "    p = [w for w in words(prediction) if not w in articles]\n" + _EM_A + _EM_END, "correct", None),
    ("exact_match", "def exact_match(prediction, answer, articles):\n"
     "    if [w for w in words(prediction) if w not in articles] == [w for w in words(answer) if w not in articles]:\n"
     "        return 1\n    return 0\n", "correct", None),
    ("token_f1", _F1_START, "empty", None),
    ("token_f1", _F1, "correct", None),
    ("token_f1", _f1(ret="    return ...\n"), "wrong", "dots-left"),
    ("token_f1", _f1(ret="    return precision, recall\n"), "wrong", "not-number"),
    ("token_f1", _f1(ret="    return precision\n"), "wrong", "precision-only"),
    ("token_f1", _f1(ret="    return recall\n"), "wrong", "recall-only"),
    ("token_f1", _f1(ret="    return (precision + recall) / 2\n"), "wrong", "mean"),
    ("token_f1", _f1("    precision = same / len(prediction)\n", "    recall = same / len(answer)\n"), "wrong", "chars"),
    ("token_f1", _f1(ret="    return 2 * precision * recall / precision + recall\n"), "wrong", "brackets"),
    ("token_f1", _f1("    precision = same / len(a)\n", "    recall = same / len(p)\n"), "correct", None),
    ("token_f1", _f1(ret="    return 2 * same / (len(p) + len(a))\n"), "correct", None),
    ("squad_scores", _SC_START, "empty", None),
    ("squad_scores", _SC, "correct", None),
    ("squad_scores", _sc(f1="        f1 = ...\n"), "wrong", "dots-left"),
    ("squad_scores", _sc(_SC_EM.replace("predictions[i], a,", "predictions[i], answer_lists[i],")), "error", "em-answers"),
    ("squad_scores", _sc(f1=_SC_F1.replace("predictions[i], a,", "predictions[i], answer_lists[i],")), "error", "f1-answers"),
    ("squad_scores", _sc(_SC_EM.replace("predictions[i], a,", "predictions, a,")), "error", "em-predictions"),
    ("squad_scores", _sc(f1=_SC_F1.replace("predictions[i], a,", "predictions, a,")), "error", "f1-predictions"),
    ("squad_scores", _SC_HEAD + _SC_EM + _SC_F1 + "    return [100 * em / len(predictions), 100 * f1 / len(predictions)]\n",
     "wrong", "not-pair"),
    ("squad_scores", _SC_FIRST, "wrong", "first-answer"),
    ("squad_scores", _SC_MIN, "wrong", "min"),
    ("squad_scores", _SC_EVERY, "wrong", "every-answer"),
    ("squad_scores", _SC_COPIED, "wrong", "copied"),
    ("squad_scores", _sc(_SC_EM.replace("em = em + ", "em += "), _SC_F1.replace("f1 = f1 + ", "f1 += ")), "correct", None),
    ("squad_scores", _sc(_SC_EM.replace("max([", "max(").replace("]])", "])"),
                         _SC_F1.replace("max([", "max(").replace("]])", "])")), "correct", None),
]
