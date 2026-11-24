# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-topic-models: the words with the highest probability in each topic of a topic model, from
its weights (topic_words); the topic that each word of a paragraph most likely comes from, given the paragraph's mix
(word_topics); the pages whose paragraphs have a topic as their largest share (topic_pages); a demo that trains the
course's LDA live, step by step, on the 2,067 paragraphs of 48 Wikipedia pages; a box to name the topics; and a choice
about a topic that mixes pages with nothing in common.

The paragraphs are the development set of SQuAD 1.1 (Rajpurkar, Zhang, Lopyrev and Liang, EMNLP 2016; CC-BY-SA-4.0) in
English and in the Dutch of SQuAD-NL v1.1 (de Vries, Wieling and Nissim, EMNLP 2023; CC-BY-SA-4.0), in
data/xquad-squad-nl.json.gz, read with everything_ai.xquad. The course's two topic models, one per language, are in
data/xquad-topics.npz (made by data/tools/make_xquad_topics.py): scikit-learn's LatentDirichletAllocation with 12
topics, random_state 0 and 20 steps of batch variational Bayes, the same model the demo trains. Paragraphs marked as
not shown are counted but never shown. Every number in the text and the pictures is computed by the course
(tests/test_nlp_topic_models.py computes them again)."""
import contextlib
import functools
import html
import io
import math
import re
import time
import warnings

import numpy as np

from everything_ai import live, paths, xquad, yours
from everything_ai.check import FunctionBox, Mistake, Timeout, with_time_limit

TOPICS = 12        # topics of the course's models, as in data/tools/make_xquad_topics.py
STEPS = 20         # steps of batch variational Bayes, the demo's too
LANGS = ("en", "nl")
MODELS = paths.ROOT / "data" / "xquad-topics.npz"
SHOWN_WORDS = 6    # the words of each topic that the demo and the rewind show
PICTURE_N = 5      # the words of each topic in the picture of exercise 1
PARAGRAPH = 3      # the paragraph of exercise 2 and its picture: Super Bowl 50 on television, in both languages


# --- the course's topic models ------------------------------------------------------------------------------------------

@functools.cache
def _model(lang: str) -> tuple:
    assert lang in LANGS, lang
    with np.load(MODELS) as d:
        return (tuple(str(w) for w in d[f"{lang}_words"]), d[f"{lang}_weights"].astype(float),
                d[f"{lang}_mix"].astype(float))


def words(lang: str) -> list:
    """The words of the columns of the counts, in alphabetical order: 4,799 in English, 4,761 in Dutch."""
    return list(_model(lang)[0])


def weights(lang: str) -> np.ndarray:
    """The course's model's weights (components_): one row of pseudo-counts per topic, one column per word."""
    return _model(lang)[1].copy()


def topics(lang: str) -> np.ndarray:
    """The weights as probabilities: each row divided by its own sum."""
    w = _model(lang)[1]
    return w / w.sum(axis=1, keepdims=True)


def mixes(lang: str) -> np.ndarray:
    """The mix of every paragraph: one row of 12 shares per paragraph, each row adding up to 1."""
    return _model(lang)[2].copy()


def pages(lang: str) -> list:
    """The name of the page of every paragraph, in the lesson's language, such as "Warsaw" or "Warschau"."""
    return [p.title for p in xquad.paragraphs(lang)]


def top_words(probs: np.ndarray, words_: list, n: int) -> list:
    """For each row of probabilities, its n words with the highest probability, the highest first: [(word, p)]."""
    return [[(words_[j], float(row[j])) for j in np.argsort(-row, kind="stable")[:n]] for row in probs]


def main_topics(lang: str) -> np.ndarray:
    """The topic with the largest share of each paragraph."""
    return _model(lang)[2].argmax(axis=1)


def page_counts(lang: str, topic: int) -> dict:
    """{page: how many of its paragraphs have `topic` as their largest share}."""
    out: dict = {}
    for page, t in zip(pages(lang), main_topics(lang)):
        if t == topic:
            out[page] = out.get(page, 0) + 1
    return out


def _ranked(counts: dict) -> list:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# --- the paragraphs as the lesson may show them ---------------------------------------------------------------------------

_HIDDEN = {"en": "(a paragraph the course does not show)", "nl": "(een alinea die de course niet laat zien)"}
# a paragraph that says he or she (in Dutch also ze, hun, zijn and the like) stays out of a picture, as the course's own
# writing does
_PRONOUN = {"en": re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.I),
            "nl": re.compile(r"\b(hij|zij|ze|haar|hem|hun|hen|zijn|z'n|diens|wiens)\b", flags=re.I)}
_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")   # as everything_ai.xquad.words


def showable(i: int, lang: str) -> bool:
    p = xquad.paragraphs(lang)[i]
    return p.shown and _PRONOUN[lang].search(p.text) is None


def tokens(lang: str, i: int) -> list:
    """The words of paragraph i as they stand in its text: (start, end, word in small letters, column or None for a
    word the model does not count)."""
    col = {w: j for j, w in enumerate(_model(lang)[0])}
    text = xquad.paragraphs(lang)[i].text
    return [(m.start(), m.end(), m.group().lower(), col.get(m.group().lower())) for m in _WORD.finditer(text)]


def ids(lang: str, i: int) -> list:
    """The columns of the counted words of paragraph i, in the order of its text, once for every time a word stands
    in it."""
    return [j for _, _, _, j in tokens(lang, i) if j is not None]


def word_topics_of(mix: np.ndarray, probs: np.ndarray, ids_: list) -> list:
    return [int(np.argmax(mix * probs[:, j])) for j in ids_]


# --- the toy collection of the worked example -----------------------------------------------------------------------------

TOY_DOCS = ["oven pizza pasta oven", "wind storm mist wind", "pizza pasta oven pasta",
            "storm mist wind storm", "oven pizza wind storm", "mist wind pasta"]


def count_matrix(docs: list) -> tuple:
    """The document-term matrix of short documents of words split by spaces, as a NumPy array of counts, and its
    words in alphabetical order."""
    vocab = sorted({w for d in docs for w in d.split()})
    return np.array([[d.split().count(w) for w in vocab] for d in docs]), vocab


@functools.cache
def _toy() -> tuple:
    from sklearn.decomposition import LatentDirichletAllocation
    counts, vocab = count_matrix(TOY_DOCS)
    lda = LatentDirichletAllocation(n_components=2, random_state=0)
    mix = lda.fit_transform(counts)
    return vocab, lda.components_.copy(), mix


def toy() -> tuple:
    """The worked example's model of TOY_DOCS: its words, its weights (2 rows) and the mix of each document."""
    vocab, w, mix = _toy()
    return list(vocab), w.copy(), mix.copy()


# --- the small examples of the exercises ----------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `small_weights`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


class _Lazy(_Val):
    """A test value made only when a check needs it, such as the course's model; written by its name."""

    def __init__(self, name: str, make):
        super().__init__(name, None)
        self.make = make

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.make(), memo)


SMALL_WORDS = ["mist", "oven", "pasta", "pizza", "storm", "wind"]   # the same words in English and in Dutch
SMALL_WEIGHTS = np.array([[0.5, 4.0, 3.0, 2.0, 0.4, 0.1],            # a kitchen topic: its row adds up to 10
                          [2.0, 0.2, 0.3, 0.1, 1.8, 3.6]])           # a weather topic: its row adds up to 8
SMALL_TOPICS = SMALL_WEIGHTS / SMALL_WEIGHTS.sum(axis=1, keepdims=True)
SMALL_MIX = np.array([0.9, 0.1])
SMALL_IDS = [1, 2, 0, 5]             # oven, pasta, mist, wind
SMALL_MIXES = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.7, 0.3]])
SMALL_PAGES = ["Pizza", "Storm", "Pizza", "Mist", "Pasta"]

_SW = _Val("small_weights", SMALL_WEIGHTS)
_SWO = _Val("small_words", SMALL_WORDS)
_ST = _Val("small_topics", SMALL_TOPICS)
_SM = _Val("small_mix", SMALL_MIX)
_SMS = _Val("small_mixes", SMALL_MIXES)
_SP = _Val("small_pages", SMALL_PAGES)

THREE_TOPICS = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.4, 0.1]])
THREE_MIXES = np.array([[0.45, 0.35, 0.2], [0.1, 0.6, 0.3], [0.5, 0.2, 0.3], [0.3, 0.3, 0.4]])
THREE_PAGES = ["Radio", "Film", "Radio", "Film"]

_W_EN = _Lazy("weights_en", lambda: weights("en"))
_WO_EN = _Lazy("words_en", lambda: words("en"))
_W_NL = _Lazy("weights_nl", lambda: weights("nl"))
_WO_NL = _Lazy("words_nl", lambda: words("nl"))
_T_EN = _Lazy("topics_en", lambda: topics("en"))
_T_NL = _Lazy("topics_nl", lambda: topics("nl"))
_PM_EN = _Lazy(f"mix_en[{PARAGRAPH}]", lambda: mixes("en")[PARAGRAPH])
_PM_NL = _Lazy(f"mix_nl[{PARAGRAPH}]", lambda: mixes("nl")[PARAGRAPH])
_ID_EN = _Lazy("ids_en", lambda: ids("en", PARAGRAPH))
_ID_NL = _Lazy("ids_nl", lambda: ids("nl", PARAGRAPH))
_M_EN = _Lazy("mix_en", lambda: mixes("en"))
_M_NL = _Lazy("mix_nl", lambda: mixes("nl"))
_P_EN = _Lazy("pages_en", lambda: pages("en"))
_P_NL = _Lazy("pages_nl", lambda: pages("nl"))
REAL_TOPIC = {"en": 9, "nl": 5}   # the topic of the real cases of topic_pages: television, in both languages


# --- the course's solutions, as code ------------------------------------------------------------------------------------

_TW_HEAD = "import numpy as np\n\n\ndef topic_words(weights, words, n):\n"


def _tw(probs: str = "weights / weights.sum(axis=1, keepdims=True)", best: str = "np.argsort(-row)[:n]",
        add: str = "[(words[j], float(row[j])) for j in best]") -> str:
    return (_TW_HEAD + f"    probs = {probs}\n    found = []\n    for row in probs:\n        best = {best}\n"
            f"        found.append({add})\n    return found\n")


_TW = _tw()
_TW_START = _tw(probs="...", best="...")

_WT_HEAD = "import numpy as np\n\n\ndef word_topics(mix, topics, ids):\n    found = []\n    for j in ids:\n"


def _wt(add: str = "int(np.argmax(mix * topics[:, j]))", end: str = "    return found\n") -> str:
    return _WT_HEAD + f"        found.append({add})\n" + end


_WT = _wt()
_WT_START = _WT_HEAD + "        ...\n    return found\n"

_TP_HEAD = "import numpy as np\n\n\ndef topic_pages(mix, pages, topic):\n    counts = {}\n    for i in range(len(pages)):\n"


def _tp(test: str = "np.argmax(mix[i]) == topic", add: str = "counts[pages[i]] + 1", first: str = "1",
        end: str = "    return counts\n") -> str:
    return (_TP_HEAD + f"        if {test}:\n            if pages[i] in counts:\n                counts[pages[i]] = {add}\n"
            f"            else:\n                counts[pages[i]] = {first}\n" + end)


_TP = _tp()
_TP_START = _TP_HEAD + "        ...\n    return counts\n"


def _course(code: str, name: str):
    space: dict = {}
    exec(compile(code, f"<course {name}>", "exec"), space)
    return space[name]


topic_words = _course(_TW, "topic_words")
word_topics = _course(_WT, "word_topics")
topic_pages = _course(_TP, "topic_pages")


# --- the pictures: a few helpers ------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.45rem;text-align:left;vertical-align:middle;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_CHIP = ("display:inline-block;margin:0.12rem 0.2rem 0.12rem 0;padding:0 0.3rem;border-radius:3px;"
         "font-family:var(--ea-font-code);font-size:0.8rem;line-height:1.5;white-space:pre")
_CHIP_DATA = _CHIP + ";border:1px solid var(--ea-line);color:var(--ea-ink)"
_CHIP_FADED = _CHIP + ";border:1px dashed var(--ea-line);color:var(--ea-ink-3)"
_CHIP_PARAM = _CHIP + ";background:var(--ea-param-wash);color:var(--ea-param-text)"
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_PURPLE = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text)"
_CHIP_MISSING = _CHIP + ";border:1px dashed var(--ea-loss);color:var(--ea-loss-text)"
_BAR = "display:inline-block;height:0.6rem;border-radius:2px;vertical-align:middle"
_TEXT = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink)"
_FAINT = "color:var(--ea-ink-3)"
_PARAGRAPH = ("font-family:var(--ea-font-ui);font-size:0.86rem;line-height:2.1;color:var(--ea-ink);max-width:44rem;"
              "margin:0.3rem 0")
_TAG = ("display:inline-block;margin-left:0.1rem;padding:0 0.18rem;border-radius:3px;font-family:var(--ea-font-code);"
        "font-size:0.68rem;line-height:1.3;vertical-align:super")


def _marked(text: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text: str) -> str:
    """A line under a picture; a word between asterisks is set in italics and a name between backticks as code, as in
    the lesson's text."""
    return f'<div style="{_UI}">{_marked(text)}</div>'


def _label(text: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 2,067 in English, 2.067 in Dutch."""
    text = f"{round(n):,}"
    return text.replace(",", ".") if lang == "nl" else text


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with `digits` decimals, with − for minus: 0.82 in English, 0,82 in Dutch."""
    x = round(float(x), digits) + 0.0
    text = f"{x:.{digits}f}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _chip(text: str, style: str = _CHIP_DATA) -> str:
    return f'<span style="{style}">{html.escape(text)}</span>'


def _bar(x: float, most: float, color: str, width: float = 6.0) -> str:
    """A bar for a number of 0 or more: as long as `width` for `most`, never shorter than a sliver."""
    w = 0 if not most or not x or x <= 0 else max(0.15, width * min(x, most) / most)
    return f'<span style="{_BAR};width:{w:.2f}rem;background:{color}"></span>' if w else ""


def _table(rows_: list, head: list, numbers_: tuple = ()) -> str:
    """A small table; rows hold HTML that is already escaped; numbers_ are the columns set to the right."""
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers_ else ""}">{html.escape(h)}</th>'
                  for j, h in enumerate(head))
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}{_NUMBER if j in numbers_ else ""}">{c}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows_)
    head_html = f"<thead><tr>{top}</tr></thead>" if head else ""
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"{head_html}<tbody>{body}</tbody></table></div>")


def _number(v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            and math.isfinite(float(v)))


def _whole(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 5   # what the learner's function gets for all the calls of one picture together


def _call(make, seconds: float = PICTURE_SECONDS):
    """make() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return with_time_limit(make, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _left(deadline: float) -> float:
    """The seconds a picture has left until `deadline`, a time.monotonic()."""
    return deadline - time.monotonic()


def _empty(w: dict, key: str) -> str:
    """The picture that stays empty, with why: the learner's function stopped, was too slow, or gave something odd."""
    text = w[key].format(s=PICTURE_SECONDS)
    return yours.figure(w["title"], _note(text), alt=text)


def _same(a, b) -> bool:
    return _number(a) and _number(b) and math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _word_chip(word: str, p: float, lang: str, style: str = _CHIP_PARAM) -> str:
    return _chip(f"{word} {_dec(p, lang, 3)}", style)


def _topic_name(t: int, lang: str) -> str:
    return f"Topic {t}"


# --- the idea: the toy collection as a topic model ------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Six short documents, two topics",
           "topics": "The two topics: the probability of each word",
           "docs": "The documents: each word with the topic it most likely comes from, and the mix",
           "doc": "Document", "words": "Words", "mix": "Mix",
           "note": "Orange: the topics, what LDA learns, with the probability of each word in its topic. Blue: the mix "
                   "of each document, the share of each topic. In the color of the text: the words of the documents, "
                   "each with the number of its topic in orange.",
           "alt": "The worked example's two topics and six documents. {topics} {docs}"},
    "nl": {"title": "Zes korte documenten, twee topics",
           "topics": "De twee topics: de kans op elk woord",
           "docs": "De documenten: elk woord met het topic waar het het waarschijnlijkst uit komt, en de mix",
           "doc": "Document", "words": "Woorden", "mix": "Mix",
           "note": "Oranje: de topics, wat LDA leert, met de kans op elk woord in het topic. Blauw: de mix van elk "
                   "document, het aandeel van elk topic. In de kleur van de tekst: de woorden van de documenten, elk "
                   "met in oranje het nummer van het topic.",
           "alt": "De twee topics en zes documenten van het uitgewerkte voorbeeld. {topics} {docs}"},
}


def _tagged(word: str, t, lang: str, style: str = "") -> str:
    tag = f'<span style="{_TAG};background:var(--ea-param-wash);color:var(--ea-param-text)">{t}</span>' if t is not None else ""
    return f'<span style="white-space:nowrap{";" + style if style else ""}">{html.escape(word)}{tag}</span>'


def idea_picture(lang: str) -> str:
    """The worked example's model: the probability of each word in the two topics, and each document with the topic
    of each of its words and its mix."""
    w = _IDEA_TEXT[lang]
    vocab, wt, mix = toy()
    probs = wt / wt.sum(axis=1, keepdims=True)
    most = float(probs.max())
    topic_rows, alt_topics = [], []
    for t in range(2):
        order = np.argsort(-probs[t], kind="stable")
        cells = " ".join(_word_chip(vocab[j], probs[t, j], lang) + _bar(probs[t, j], most, "var(--ea-param)", 3.0)
                         for j in order[:4])
        topic_rows.append([html.escape(_topic_name(t, lang)), cells])
        alt_topics.append(f"{_topic_name(t, lang)}: " + ", ".join(f"{vocab[j]} {_dec(probs[t, j], lang, 3)}"
                                                                  for j in order[:4]))
    doc_rows, alt_docs = [], []
    for i, d in enumerate(TOY_DOCS):
        ws = d.split()
        found = word_topics_of(mix[i], probs, [vocab.index(x) for x in ws])
        shown = " ".join(_tagged(x, t, lang) for x, t in zip(ws, found))
        bars = " ".join(f'{_chip(_dec(mix[i, t], lang, 2), _CHIP_PRED)}{_bar(mix[i, t], 1.0, "var(--ea-pred)", 3.0)}'
                        for t in range(2))
        doc_rows.append([str(i), shown, bars])
        alt_docs.append(f"{w['doc']} {i}: " + " ".join(f"{x} ({t})" for x, t in zip(ws, found))
                        + f", {w['mix']} {_dec(mix[i, 0], lang)} / {_dec(mix[i, 1], lang)}")
    body = (_label(w["topics"]) + _table(topic_rows, []) + _label(w["docs"])
            + _table(doc_rows, [w["doc"], w["words"], w["mix"] + " (0, 1)"]) + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(topics="; ".join(alt_topics), docs="; ".join(alt_docs)))


# --- exercise 1: the words of each topic ----------------------------------------------------------------------------------

_WORDS_TEXT = {
    "en": {"title": "The words of each topic, from your topic_words",
           "topic": "Topic", "words": "The {n} words with the highest probability",
           "note": "The course's English model, with your `topic_words(weights, words, {n})`. Orange: the "
                   "probability of each word in its topic, which LDA learned. Purple: where your function gives "
                   "another word or probability, with the right one dashed beside it.",
           "stopped": "Your function stops with an error on the course's model, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on the course's model, so the picture stays "
                   "empty.",
           "odd": "Your function does not give a list with a list of (word, probability) pairs for each topic, so "
                  "the picture stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De woorden van elk topic, uit je topic_words",
           "topic": "Topic", "words": "De {n} woorden met de hoogste kans",
           "note": "Het Nederlandse model van de course, met je `topic_words(weights, words, {n})`. Oranje: de kans "
                   "op elk woord in het topic, die LDA leerde. Paars: waar je functie een ander woord of een andere "
                   "kans geeft, met het goede gestippeld ernaast.",
           "stopped": "Je functie stopt met een fout op het model van de course, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met het model van de course, dus de afbeelding "
                   "blijft leeg.",
           "odd": "Je functie geeft geen lijst met voor elk topic een lijst van paren (woord, kans), dus de "
                  "afbeelding blijft leeg.",
           "alt": "{title}: {rows}"},
}


def _pairs(out, k: int) -> bool:
    """A list of k lists, each of (word, number) pairs."""
    return (isinstance(out, list) and len(out) == k
            and all(isinstance(r, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2
                                                         and isinstance(p[0], str) and _number(p[1]) for p in r)
                    for r in out))


def words_picture(seen) -> str:
    """The five words of every topic of the course's model in the lesson's language, from the learner's topic_words."""
    lang = seen.lang
    w = _WORDS_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    want = top_words(topics(lang), words(lang), PICTURE_N)
    wt, vocab = weights(lang), words(lang)
    out = _call(lambda: fn(wt, vocab, PICTURE_N))
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not _pairs(out, TOPICS):
        return _empty(w, "odd")
    most = max(p for r in want for _, p in r)
    rows_, alts = [], []
    for t, (mine, right) in enumerate(zip(out, want)):
        cells = []
        for k, (word, p) in enumerate(right):
            got = mine[k] if k < len(mine) else None
            if got is not None and got[0] == word and _same(got[1], p):
                cells.append(_word_chip(word, p, lang) + _bar(p, most, "var(--ea-param)", 2.0))
            else:
                bad = _chip(f"{got[0]} {_dec(got[1], lang, 3)}", _CHIP_PURPLE) if got is not None else ""
                cells.append(bad + _word_chip(word, p, lang, _CHIP_MISSING))
        if len(mine) > len(right):
            cells.append(_chip(f"+{len(mine) - len(right)}", _CHIP_PURPLE))
        rows_.append([str(t), " ".join(cells)])
        alts.append(f"{t}: " + ", ".join(f"{a} {_dec(b, lang, 3)}" for a, b in mine[:PICTURE_N]))
    note = w["note"].format(n=PICTURE_N)
    body = _table(rows_, [w["topic"], w["words"].format(n=PICTURE_N)]) + _note(note)
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- exercise 2: the topic of each word of a paragraph --------------------------------------------------------------------

_PARA_TEXT = {
    "en": {"title": "The topic of each word, from your word_topics",
           "mix": "The mix of paragraph {i} of the page {page}: its three largest topics",
           "words": "Its words, each with the topic your function gives it",
           "note": "In the color of the text: the words that LDA counts, each with the number of the topic your "
                   "`word_topics` gives it, in orange. Light gray: the words it leaves out (stop words, words of "
                   "two letters or fewer, numbers, words with an apostrophe, and words in fewer than 5 or more than 10% of the paragraphs). Blue: "
                   "the share of each topic in the paragraph's mix. Orange: the words of each topic. Purple: where "
                   "your function gives another topic, with the right one dashed beside it.",
           "stopped": "Your function stops with an error on this paragraph, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on this paragraph, so the picture stays empty.",
           "odd": "Your function does not give a list with one topic number for each word, so the picture stays "
                  "empty.",
           "hidden": "The course does not show this paragraph.",
           "alt": "{title}. Mix: {mix}. Words: {words}"},
    "nl": {"title": "Het topic van elk woord, uit je word_topics",
           "mix": "De mix van alinea {i} van de pagina {page}: de drie grootste topics",
           "words": "De woorden, elk met het topic dat je functie het woord geeft",
           "note": "In de kleur van de tekst: de woorden die LDA telt, elk met in oranje het nummer van het topic dat "
                   "je `word_topics` het woord geeft. Lichtgrijs: de woorden die LDA weglaat (stop words, woorden "
                   "van twee letters of minder, getallen, woorden met een apostrof, en woorden in minder dan 5 of meer dan 10% van de alinea's). Blauw: het aandeel van elk topic in de mix van de alinea. Oranje: de woorden van elk "
                   "topic. Paars: waar je functie een ander topic geeft, met het goede gestippeld ernaast.",
           "stopped": "Je functie stopt met een fout op deze alinea, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met deze alinea, dus de afbeelding blijft leeg.",
           "odd": "Je functie geeft geen lijst met één topicnummer voor elk woord, dus de afbeelding blijft leeg.",
           "hidden": "De course laat deze alinea niet zien.",
           "alt": "{title}. Mix: {mix}. Woorden: {words}"},
}


def mix_table(mix: np.ndarray, probs: np.ndarray, vocab: list, lang: str, k: int = 3, n: int = 5) -> tuple:
    """The k largest topics of a mix, each with its share and its n words; and the same as text, for the alt."""
    rows_, alts = [], []
    for t in np.argsort(-mix, kind="stable")[:k]:
        tops = top_words(probs[t:t + 1], vocab, n)[0]
        rows_.append([str(int(t)), _chip(_dec(mix[t], lang, 2), _CHIP_PRED) + _bar(mix[t], 1.0, "var(--ea-pred)", 4.0),
                      " ".join(_chip(x, _CHIP_PARAM) for x, _ in tops)])
        alts.append(f"{int(t)} {_dec(mix[t], lang, 2)} ({', '.join(x for x, _ in tops)})")
    return _table(rows_, []), "; ".join(alts)


def paragraph_html(lang: str, i: int, found: list | None, right: list) -> tuple:
    """The text of paragraph i with each counted word tagged with its topic: the learner's (found) where it is
    right, in purple with the right one dashed where it is not; and the same as text, for the alt."""
    text = xquad.paragraphs(lang)[i].text
    out, alts, at, k = [], [], 0, 0
    for start, end, _, j in tokens(lang, i):
        out.append(html.escape(text[at:start]))
        word = text[start:end]
        if j is None:
            out.append(f'<span style="{_FAINT}">{html.escape(word)}</span>')
        else:
            mine = found[k] if found is not None and k < len(found) else None
            good = right[k]
            if found is None or (_whole(mine) and int(mine) == good):
                out.append(_tagged(word, good, lang))
            else:
                bad = f'<span style="{_TAG};background:var(--ea-loss-wash);color:var(--ea-loss-text)">{html.escape(str(mine))}</span>'
                ok = f'<span style="{_TAG};border:1px dashed var(--ea-loss);color:var(--ea-loss-text)">{good}</span>'
                out.append(f'<span style="white-space:nowrap">{html.escape(word)}{bad}{ok}</span>')
            alts.append(f"{word} ({mine if found is not None else good})")
            k += 1
        at = end
    out.append(html.escape(text[at:]))
    return f'<div style="{_PARAGRAPH}">{"".join(out)}</div>', " ".join(alts)


def paragraph_picture(seen) -> str:
    """Paragraph PARAGRAPH in the lesson's language, each counted word with the topic the learner's word_topics gives
    it, and the paragraph's mix."""
    lang = seen.lang
    w = _PARA_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    if not showable(PARAGRAPH, lang):
        return yours.figure(w["title"], _note(w["hidden"]), alt=w["hidden"])
    mix, probs, vocab = mixes(lang)[PARAGRAPH], topics(lang), words(lang)
    found_ids = ids(lang, PARAGRAPH)
    right = word_topics_of(mix, probs, found_ids)
    out = _call(lambda: fn(mix.copy(), probs.copy(), list(found_ids)))
    if out is _STOPPED or out is _SLOW:
        return _empty(w, "stopped" if out is _STOPPED else "slow")
    if not (isinstance(out, list) and len(out) == len(found_ids)):
        return _empty(w, "odd")
    table, alt_mix = mix_table(mix, probs, vocab, lang)
    text, alt_words = paragraph_html(lang, PARAGRAPH, out, right)
    page = xquad.paragraphs(lang)[PARAGRAPH].title
    body = (_label(w["mix"].format(i=PARAGRAPH, page=page)) + table + _label(w["words"]) + text + _note(w["note"]))
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], mix=alt_mix, words=alt_words))


# --- exercise 3: the pages of each topic ----------------------------------------------------------------------------------

_PAGES_TEXT = {
    "en": {"title": "The pages of each topic, from your topic_pages",
           "topic": "Topic", "words": "Its words", "pages": "The pages with the most paragraphs in it",
           "paragraphs": "Paragraphs",
           "note": "The course's English model, with your `topic_pages(mix, pages, t)` for every topic t. Orange: "
                   "the words of each topic. In the color of the text: the pages, with how many of their paragraphs "
                   "have the topic as their largest share, and those paragraphs together. Purple: where your function "
                   "gives other counts, with the right ones dashed beside them.",
           "stopped": "Your function stops with an error on the course's model, so the picture stays empty.",
           "slow": "Your function is still not done after {s} seconds on the course's model, so the picture stays "
                   "empty.",
           "odd": "Your function does not give a dictionary with a number of paragraphs for each page, so the picture "
                  "stays empty.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "De pagina's van elk topic, uit je topic_pages",
           "topic": "Topic", "words": "De woorden", "pages": "De pagina's met de meeste alinea's in het topic",
           "paragraphs": "Alinea's",
           "note": "Het Nederlandse model van de course, met je `topic_pages(mix, pages, t)` voor elk topic t. Oranje: "
                   "de woorden van elk topic. In de kleur van de tekst: de pagina's, elk met het aantal alinea's waarin het "
                   "topic het grootste aandeel heeft, en het totaal van deze alinea's. Paars: waar je functie andere aantallen "
                   "geeft, met de goede gestippeld ernaast.",
           "stopped": "Je functie stopt met een fout op het model van de course, dus de afbeelding blijft leeg.",
           "slow": "Je functie is na {s} seconden nog niet klaar met het model van de course, dus de afbeelding "
                   "blijft leeg.",
           "odd": "Je functie geeft geen dictionary met een aantal alinea's voor elke pagina, dus de afbeelding blijft "
                  "leeg.",
           "alt": "{title}: {rows}"},
}
SHOWN_PAGES = 3   # pages per topic in the pictures


def _counts_ok(out) -> bool:
    return isinstance(out, dict) and all(isinstance(k, str) and _number(v) for k, v in out.items())


def _pages_cell(counts: dict, lang: str) -> str:
    return ", ".join(f"{html.escape(p)} {_num(c, lang)}" for p, c in _ranked(counts)[:SHOWN_PAGES])


def pages_picture(seen) -> str:
    """For every topic of the course's model in the lesson's language, its words and the pages with the most paragraphs
    in it, from the learner's topic_pages."""
    lang = seen.lang
    w = _PAGES_TEXT[lang]
    fn = _fn(seen)
    if fn is None:
        return ""
    mix, names, probs, vocab = mixes(lang), pages(lang), topics(lang), words(lang)
    deadline = time.monotonic() + PICTURE_SECONDS
    rows_, alts = [], []
    for t in range(TOPICS):
        out = _call(lambda: fn(mix.copy(), list(names), t), _left(deadline))
        if out is _STOPPED or out is _SLOW:
            return _empty(w, "stopped" if out is _STOPPED else "slow")
        if not _counts_ok(out):
            return _empty(w, "odd")
        right = page_counts(lang, t)
        tops = " ".join(_chip(x, _CHIP_PARAM) for x, _ in top_words(probs[t:t + 1], vocab, 4)[0])
        if out == right:
            cell, total = _pages_cell(out, lang), _num(sum(out.values()), lang)
        else:
            cell = (f'<span style="{_CHIP_PURPLE}">{_pages_cell(out, lang) or "{}"}</span> '
                    f'<span style="{_CHIP_MISSING}">{_pages_cell(right, lang)}</span>')
            total = (_chip(_num(sum(out.values()), lang) if out else "0", _CHIP_PURPLE)
                     + _chip(_num(sum(right.values()), lang), _CHIP_MISSING))
        rows_.append([str(t), tops, cell, total])
        alts.append(f"{t}: " + ", ".join(f"{p} {c}" for p, c in _ranked(out)[:SHOWN_PAGES]))
    body = _table(rows_, [w["topic"], w["words"], w["pages"], w["paragraphs"]], numbers_=(3,)) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- naming the topics ----------------------------------------------------------------------------------------------------

NAMES_START = "\n".join(f"{t}: " for t in range(TOPICS))
NAMES_LONGEST = 60   # characters of a name that the picture shows
_NAME = re.compile(r"^\s*(\d+)\s*[:.)=-]\s*(.*?)\s*$")
_NAMES_TEXT = {
    "en": {"title": "Your names for the topics",
           "topic": "Topic", "name": "Your name", "words": "Its words", "pages": "Its pages",
           "none": "no name yet",
           "note": "Blue: your names, one per line after the number of the topic. Orange: the words of each topic, "
                   "which LDA learned. In the color of the text: the pages with the most paragraphs that have the topic "
                   "as their largest share. Gray: no name yet. LDA never saw the names of the pages.",
           "alt": "{title}: {rows}"},
    "nl": {"title": "Jouw namen voor de topics",
           "topic": "Topic", "name": "Jouw naam", "words": "De woorden", "pages": "De pagina's",
           "none": "nog geen naam",
           "note": "Blauw: jouw namen, één per regel na het nummer van het topic. Oranje: de woorden van elk topic, die "
                   "LDA leerde. In de kleur van de tekst: de pagina's met de meeste alinea's die het topic als grootste "
                   "aandeel hebben. Grijs: nog geen naam. LDA heeft de namen van de pagina's nooit gezien.",
           "alt": "{title}: {rows}"},
}


def names_of(text) -> dict:
    """{topic: name} from the lines of the naming box, such as "9: television"."""
    out = {}
    for line in str(text or "").splitlines():
        m = _NAME.match(line)
        if m and m.group(2) and int(m.group(1)) < TOPICS:
            out[int(m.group(1))] = m.group(2)[:NAMES_LONGEST]
    return out


def names_picture(text, lang: str) -> str:
    """The learner's names for the topics of the course's model, beside each topic's words and pages."""
    w = _NAMES_TEXT[lang]
    names = names_of(text)
    probs, vocab = topics(lang), words(lang)
    rows_, alts = [], []
    for t in range(TOPICS):
        name = _chip(names[t], _CHIP_PRED) if t in names else f'<span style="{_FAINT}">{html.escape(w["none"])}</span>'
        tops = top_words(probs[t:t + 1], vocab, 5)[0]
        rows_.append([str(t), name, " ".join(_chip(x, _CHIP_PARAM) for x, _ in tops),
                      _pages_cell(page_counts(lang, t), lang)])
        alts.append(f"{t} {names.get(t, w['none'])}: {', '.join(x for x, _ in tops)}")
    body = _table(rows_, [w["topic"], w["name"], w["words"], w["pages"]]) + _note(w["note"])
    return yours.figure(w["title"], body, alt=w["alt"].format(title=w["title"], rows="; ".join(alts)))


# --- the demo: LDA trains, step by step -----------------------------------------------------------------------------------

DEMO_LIMIT = 120.0   # seconds the demo's training gets
LOOKS = list(range(1, STEPS + 1))   # the steps the rewind stops at


def _lda(n: int):
    """The course's LDA, set up so that each call of partial_fit is one step of batch variational Bayes on all n
    paragraphs: with a learning decay of 0 and one batch of all paragraphs, partial_fit does what fit does in one
    of its steps (scikit-learn's documentation of learning_decay)."""
    from sklearn.decomposition import LatentDirichletAllocation
    return LatentDirichletAllocation(n_components=TOPICS, random_state=0, learning_method="online", learning_decay=0.0,
                                     total_samples=n, batch_size=n)


def _tops(components: np.ndarray, vocab: tuple, n: int = SHOWN_WORDS) -> list:
    return top_words(components / components.sum(axis=1, keepdims=True), list(vocab), n)


def demo(lang: str, show=None) -> "dict | str":
    """The course's LDA on the counts of the paragraphs in `lang`, STEPS steps within DEMO_LIMIT seconds, with
    `show(step, surprise, tops)` after every step: the mean surprise per word, the log of scikit-learn's perplexity,
    and the SHOWN_WORDS words of each topic. {"lang", "curve": [(step, surprise)], "tops": {step: tops},
    "uniform": the surprise of a model that gives every word the same probability, "weights": the last weights}, or
    "slow"."""
    def run():
        counts, vocab = xquad.counts(lang)
        model = _lda(counts.shape[0])
        out = {"lang": lang, "curve": [], "tops": {}, "uniform": math.log(len(vocab))}
        for s in range(1, STEPS + 1):
            model.partial_fit(counts)
            surprise = math.log(model.perplexity(counts))
            tops = _tops(model.components_, vocab)
            out["curve"].append((s, surprise))
            out["tops"][s] = tops
            if show is not None:
                show(s, surprise, tops)
        out["weights"] = model.components_.copy()
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return with_time_limit(run, DEMO_LIMIT)
    except Timeout:
        return "slow"


_LIVE_TEXT = {
    "en": {"title": "LDA on the English paragraphs", "x": "step", "y": "surprise per word",
           "alt": "The mean surprise per word of the 2,067 {adjective} paragraphs after each step of LDA, from "
                  "scikit-learn's perplexity.",
           "uniform": "every word as likely",
           "now": "The {n} words of each topic now", "after": "The {n} words of each topic after step {s}",
           "progress": "**Training** · step {s} of {n} · surprise per word {v}",
           "note": "Orange: the words with the highest probability in each topic, which LDA learns. The purple line: "
                   "the mean surprise per word; the dashed line: a model that gives every word the same probability."},
    "nl": {"title": "LDA op de Nederlandse alinea's", "x": "stap", "y": "verrassing per woord",
           "alt": "De gemiddelde verrassing per woord van de 2.067 {adjective} alinea's na elke stap van LDA, uit de "
                  "perplexity van scikit-learn.",
           "uniform": "elk woord even waarschijnlijk",
           "now": "De {n} woorden van elk topic nu", "after": "De {n} woorden van elk topic na stap {s}",
           "progress": "**Training** · stap {s} van {n} · verrassing per woord {v}",
           "note": "Oranje: de woorden met de hoogste kans in elk topic, die LDA leert. De paarse lijn: de gemiddelde "
                   "verrassing per woord; de gestippelde lijn: een model dat elk woord dezelfde kans geeft."},
}
_ADJECTIVE = {"en": {"en": "English", "nl": "Dutch"}, "nl": {"en": "Engelse", "nl": "Nederlandse"}}
LOW, HIGH = 7.6, 8.8   # the y-axis of the chart, for both languages


def live_words(lang: str) -> dict:
    """The words of the live chart, in the lesson's language, for a run in that language."""
    w = dict(_LIVE_TEXT[lang])
    w["alt"] = w["alt"].format(adjective=_ADJECTIVE[lang][lang])
    return w


def tops_table(tops: list, lang: str, title: str) -> str:
    """The words of each topic, a row per topic."""
    return _label(title) + _table([[str(t), " ".join(_chip(x, _CHIP_PARAM) for x, _ in found)]
                                   for t, found in enumerate(tops)], [])


def now_parts(tops: list, lang: str, step_: int | None = None) -> list:
    """What the demo shows under its chart: the words of each topic now (or after `step_`), and the note."""
    w = _LIVE_TEXT[lang]
    title = w["now"].format(n=SHOWN_WORDS) if step_ is None else w["after"].format(n=SHOWN_WORDS, s=step_)
    return [tops_table(tops, lang, title), _note(w["note"])]


def uniform_line(lang: str) -> list:
    """The dashed line of the chart: the surprise per word of a model that gives every word of the counts the same
    probability, the log of the number of words, as (step, value) at both ends."""
    u = math.log(len(_model(lang)[0]))
    return [(0, u), (STEPS, u)]


def run_chart(run: dict, lang: str) -> live.Live:
    """The chart of a finished run, as the demo drew it, for the rewind."""
    w = live_words(lang)
    chart_ = live.Live(w, STEPS, lang=lang, low=LOW, high=HIGH)
    chart_.reference(uniform_line(lang), w["uniform"])
    chart_.xs, chart_.ys = [x for x, _ in run["curve"]], [y for _, y in run["curve"]]
    return chart_


def rewind_frame(run: dict, step_: int, lang: str) -> str:
    """The rewind at a step: the chart with a line at that step, beside it the words of each topic then."""
    return live.rewound(run_chart(run, lang).picture(at=step_), now_parts(run["tops"][step_], lang, step_))


_SAID = {"en": "After step {s}, the mean surprise per word is {v}.",
         "nl": "Na stap {s} is de gemiddelde verrassing per woord {v}."}


def rewind_said(run: dict, step_: int, lang: str) -> str:
    return _SAID[lang].format(s=step_, v=_dec(dict(run["curve"])[step_], lang, 3))


def topic_of(run: dict, step_: int, word: str) -> list:
    """The topics whose shown words hold `word` after `step_`."""
    return [t for t, tops in enumerate(run["tops"][step_]) if word in [x for x, _ in tops]]


_SLOW_NOTE = {"en": "*The training took more than {s:g} seconds, so it stopped here. Is the computer busy with something "
                    "else? Press the button to try again.*",
              "nl": "*De training duurde langer dan {s:g} seconden, dus de training stopt hier. Is de computer met iets "
                    "anders bezig? Druk op de knop om het opnieuw te proberen.*"}


def slow_note(lang: str) -> str:
    return _SLOW_NOTE[lang].format(s=DEMO_LIMIT)


# --- the numbers of the text ----------------------------------------------------------------------------------------------

def numbers(lang: str) -> dict:
    """The numbers the lesson's text names for the paragraphs in `lang`."""
    counts, vocab = xquad.counts(lang)
    mix = _model(lang)[2]
    main = mix.argmax(axis=1)
    arts = np.array([p.article for p in xquad.paragraphs(lang)])
    agree = 0
    for a in range(len(xquad.titles(lang))):
        found = np.bincount(main[arts == a], minlength=TOPICS)
        agree += int(found.max())
    return {"paragraphs": counts.shape[0], "words": len(vocab), "tokens": int(counts.sum()),
            "half": int((mix.max(axis=1) >= 0.5).sum()), "mean_largest": float(mix.max(axis=1).mean()),
            "agree": agree, "uniform": math.log(len(vocab)),
            "sizes": [int((main == t).sum()) for t in range(TOPICS)]}


# --- the exercises ---------------------------------------------------------------------------------------------------------

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _show(v):
    """A value for a message, the way code writes it: a number with at most three decimals, a long list or
    dictionary cut short."""
    if isinstance(v, (bool, np.bool_)) or v is None:
        return repr(bool(v)) if isinstance(v, np.bool_) else repr(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = round(float(v), 3) + 0.0
        return str(int(x)) if x == int(x) and abs(x) < 1e15 else f"{x:g}"
    if isinstance(v, (str, np.str_)):
        return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, np.ndarray):
        return _show(v.tolist())
    if isinstance(v, (list, tuple)):
        items = [_show(x) for x in v[:3]] + (["…"] if len(v) > 3 else [])
        inner = ", ".join(items)
        return f"[{inner}]" if isinstance(v, list) else f"({inner}{',' if len(v) == 1 else ''})"
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: (-kv[1], str(kv[0]))) if all(_number(x) for x in v.values()) else list(v.items())
        inner = ", ".join(f"{_show(k)}: {_show(x)}" for k, x in items[:4]) + (", …" if len(items) > 4 else "")
        return "{" + inner + "}"
    return repr(v)


def _not_list(got, text) -> bool:
    return not isinstance(got, list)


def _not_dict(got, text) -> bool:
    return not isinstance(got, dict)


def _shares(got, text) -> bool:
    """A list of numbers between 0 and 1 that are not all whole: the products themselves, not the topics."""
    return (isinstance(got, list) and bool(got) and all(_number(x) for x in got)
            and any(not float(x).is_integer() for x in got))


def _row_error(e) -> bool:
    return isinstance(e, IndexError) or (isinstance(e, ValueError) and "broadcast" in str(e))


REQUIRED = ["topic_words", "word_topics", "topic_pages", "mixed-topic"]

EXERCISES = {
    "topic_words": FunctionBox(
        name="topic_words",
        picture=words_picture,
        function="topic_words",
        start=_TW_START,
        solution=_TW,
        show=_show,
        visible=[(_SW, _SWO, 3)],
        hidden=[(_SW, _SWO, 1),
                (np.array([[1.0, 3.0, 2.0, 4.0], [5.0, 1.0, 2.0, 0.5], [0.2, 0.3, 4.0, 0.5]]), ["data", "film", "museum", "radio"], 2),
                (np.array([[2.0, 1.0, 5.0]]), ["mist", "storm", "wind"], 3),
                (np.array([[1.0, 3.0], [4.0, 2.0]]), ["film", "radio"], 1),
                (_W_EN, _WO_EN, 5), (_W_NL, _WO_NL, 5)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list with a list of pairs for every topic. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn met een lijst paren voor elk topic. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("broadcast", error=lambda e: isinstance(e, ValueError) and "broadcast" in str(e),
                    en="`{call}` stops with a ValueError, *operands could not be broadcast together*: `weights.sum(axis=1)` gives one sum per topic as a flat array, and `weights` has a row per topic and a column per word. Which shape does the array of sums need, to divide each row by its own sum?",
                    nl="`{call}` stopt met een ValueError, *operands could not be broadcast together*: `weights.sum(axis=1)` geeft één som per topic als platte array, en `weights` heeft een rij per topic en een kolom per woord. Welke vorm heeft de array met sommen nodig om elke rij door de eigen som te delen?"),
            Mistake("counts", wrong=_tw(probs="weights"),
                    en="`{call}` gives {got}, and it should give {want}: the right words, with their weights themselves. A topic is a probability distribution over the words. What do the probabilities of one topic add up to?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede woorden, met de weights zelf. Een topic is een kansverdeling over de woorden. Waar tellen de kansen van één topic samen op uit?"),
            Mistake("total", wrong=_tw(probs="weights / weights.sum()"),
                    en="`{call}` gives {got}, and it should give {want}: each probability is too small. You divide by the sum of all weights, of all topics together. Which sum does one row need?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke kans is te klein. Je deelt door de som van alle weights, van alle topics samen. Welke som heeft één rij nodig?"),
            Mistake("columns", wrong=_tw(probs="weights / weights.sum(axis=0)"),
                    en="`{call}` gives {got}, and it should give {want}: each weight is divided by the sum of its column, over the topics. A topic is a row. Which axis does the sum of a row add up?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke weight wordt gedeeld door de som van de kolom, over de topics. Een topic is een rij. Langs welke as telt de som van een rij op?"),
            Mistake("smallest", wrong=_tw(best="np.argsort(row)[:n]"),
                    en="`{call}` gives {got}, and it should give {want}: the words with the lowest probability. `np.argsort` gives the places from the smallest value to the largest. How do you get the largest first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de woorden met de laagste kans. `np.argsort` geeft de plaatsen van de kleinste waarde naar de grootste. Hoe krijg je de grootste eerst?"),
            Mistake("reversed", wrong=_tw(best="np.argsort(row)[-n:]"),
                    en="`{call}` gives {got}, and it should give {want}: the right words, but the highest probability last. In which order should they stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede woorden, maar de hoogste kans als laatste. In welke volgorde horen de woorden te staan?"),
            Mistake("all-words", wrong=_tw(best="np.argsort(-row)"),
                    en="`{call}` gives {got}, and it should give {want}: every word of the topic, not only the first `n`. Which part of the places do you keep?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord van het topic, niet alleen de eerste `n`. Welk deel van de plaatsen houd je?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `probs`, the weights with each row divided by its own sum, and `best`, the places of the `n` largest values of `row`.",
                      "nl": "Vervang de twee keer `...`: `probs`, de weights met elke rij gedeeld door de eigen som, en `best`, de plaatsen van de `n` grootste waarden van `row`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which words have the highest probability in each topic?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke woorden hebben de hoogste kans in elk topic?"},
            "almost": {"en": "Right for `topic_words(small_weights, small_words, 3)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `n` it gets, and work for any number of topics and words?",
                       "nl": "Goed voor `topic_words(small_weights, small_words, 3)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `n` uit de aanroep, en werkt de functie voor elk aantal topics en woorden?"},
            "correct": {"en": "Your `topic_words` is right in all {n} cases.",
                        "nl": "Je `topic_words` klopt in alle {n} gevallen."},
        },
    ),
    "word_topics": FunctionBox(
        name="word_topics",
        picture=paragraph_picture,
        function="word_topics",
        start=_WT_START,
        solution=_WT,
        show=_show,
        visible=[(_SM, _ST, SMALL_IDS)],
        hidden=[(np.array([0.5, 0.5]), _ST, [0, 4, 5, 1]), (np.array([0.3, 0.7]), _ST, [3, 3, 0]), (_SM, _ST, []),
                (np.array([0.2, 0.5, 0.3]), THREE_TOPICS, [0, 1, 2, 3]),
                (_PM_EN, _T_EN, _ID_EN), (_PM_NL, _T_NL, _ID_NL)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give a list with a topic for every word. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een lijst zijn met een topic voor elk woord. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("row", error=_row_error,
                    en="`{call}` stops with an error: `topics` has a row per topic and a column per word, and `j` is the column of a word. Where in `topics` are the probabilities of word `j` in each topic?",
                    nl="`{call}` stopt met een fout: `topics` heeft een rij per topic en een kolom per woord, en `j` is de kolom van een woord. Waar in `topics` staan de kansen op woord `j` in elk topic?"),
            Mistake("word-only", wrong=_wt(add="int(np.argmax(topics[:, j]))"),
                    en="`{call}` gives {got}, and it should give {want}: each word gets the topic where it is most likely, whatever the paragraph. *mist* is more likely in the weather topic, but `small_mix` is 90% kitchen. What else does the topic of a word depend on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord krijgt het topic waarin het woord het waarschijnlijkst is, wat de alinea ook is. *mist* is waarschijnlijker in het weertopic, maar `small_mix` is voor 90% keuken. Waar hangt het topic van een woord nog meer van af?"),
            Mistake("mix-only", wrong=_wt(add="int(np.argmax(mix))"),
                    en="`{call}` gives {got}, and it should give {want}: every word gets the largest topic of the mix. *wind* has a probability of only 0.01 in the kitchen topic. What else does the topic of a word depend on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord krijgt het grootste topic van de mix. *wind* heeft in het keukentopic een kans van slechts 0,01. Waar hangt het topic van een woord nog meer van af?"),
            Mistake("shares", matches=_shares,
                    en="`{call}` gives {got}, and it should give {want}: numbers between 0 and 1, not the numbers of topics. Which NumPy function gives the place of the largest value?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: getallen tussen 0 en 1, niet de nummers van topics. Welke functie van NumPy geeft de plaats van de grootste waarde?"),
            Mistake("return-in-loop", wrong=_wt(end="        return found\n    return found\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the topic of the first word. Your `return` stands inside the loop, so the function stops after one word. Where does `return found` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen het topic van het eerste woord. Je `return` staat binnen de loop, dus de functie stopt na één woord. Waar hoort `return found`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the line that adds the topic of word `j` to `found`.",
                      "nl": "Vervang de `...` door de regel die het topic van woord `j` toevoegt aan `found`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each word: which topic has the largest share times probability?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welk topic heeft voor elk woord het grootste aandeel maal kans?"},
            "almost": {"en": "Right for `word_topics(small_mix, small_topics, [1, 2, 0, 5])`. But `{call}` gives {got}, and it should give {want}. Does your function use the `mix`, the `topics` and the `ids` it gets, for any number of topics?",
                       "nl": "Goed voor `word_topics(small_mix, small_topics, [1, 2, 0, 5])`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `mix`, de `topics` en de `ids` uit de aanroep, voor elk aantal topics?"},
            "correct": {"en": "Your `word_topics` is right in all {n} cases.",
                        "nl": "Je `word_topics` klopt in alle {n} gevallen."},
        },
    ),
    "topic_pages": FunctionBox(
        name="topic_pages",
        picture=pages_picture,
        function="topic_pages",
        start=_TP_START,
        solution=_TP,
        show=_show,
        visible=[(_SMS, _SP, 0)],
        hidden=[(_SMS, _SP, 1), (THREE_MIXES, THREE_PAGES, 0),
                (np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]), ["Film", "Radio"], 2),
                (_M_EN, _P_EN, REAL_TOPIC["en"]), (_M_NL, _P_NL, REAL_TOPIC["nl"])],
        first=[_DOTS],
        mistakes=[
            Mistake("not-dict", matches=_not_dict,
                    en="`{call}` gives {got}, and it should give a dictionary with a number of paragraphs for each page. Where does your `return` stand, and what does it give back?",
                    nl="`{call}` geeft {got}, en het moet een dictionary zijn met een aantal alinea's voor elke pagina. Waar staat je `return`, en wat geeft je functie terug?"),
            Mistake("all", wrong=_tp(test="True"),
                    en="`{call}` gives {got}, and it should give {want}: every paragraph counts, whatever its topics. Which paragraphs belong to `topic`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke alinea telt mee, wat de topics ook zijn. Welke alinea's horen bij `topic`?"),
            Mistake("half", wrong=_tp(test="mix[i][topic] > 0.5"),
                    en="`{call}` gives {got}, and it should give {want}: a paragraph counts only when the topic has more than half of it. With three topics, the largest share can be below 0.5. When is a topic the largest share of a paragraph?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een alinea telt alleen mee als het topic meer dan de helft heeft. Met drie topics kan het grootste aandeel onder 0,5 liggen. Wanneer is een topic het grootste aandeel van een alinea?"),
            Mistake("ones", wrong=_tp(add="1"),
                    en="`{call}` gives {got}, and it should give {want}: every page gets 1, also a page with two paragraphs in the topic, such as *Radio*. What should happen to the count of a page that is already in `counts`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elke pagina krijgt 1, ook een pagina met twee alinea's in het topic, zoals *Radio*. Wat moet er gebeuren met het aantal van een pagina die al in `counts` staat?"),
            Mistake("shares", wrong=_tp(add="counts[pages[i]] + mix[i][topic]", first="mix[i][topic]"),
                    en="`{call}` gives {got}, and it should give {want}: the shares of the topic added up, not the paragraphs counted. How much does one paragraph add to the count of its page?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de aandelen van het topic opgeteld, niet de alinea's geteld. Hoeveel telt één alinea op bij het aantal van de pagina?"),
            Mistake("return-in-loop", wrong=_tp(end="        return counts\n    return counts\n"),
                    en="`{call}` gives {got}, and it should give {want}: only the first paragraph is looked at. Your `return` stands inside the loop, so the function stops after one paragraph. Where does `return counts` belong?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: alleen naar de eerste alinea wordt gekeken. Je `return` staat binnen de loop, dus de functie stopt na één alinea. Waar hoort `return counts`?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with an `if` for the paragraphs whose largest share is `topic`, and under it the lines that count the page of paragraph `i` in `counts`.",
                      "nl": "Vervang de `...` door een `if` voor de alinea's met `topic` als grootste aandeel, en daaronder de regels die de pagina van alinea `i` tellen in `counts`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which paragraphs have `topic` as their largest share, and of which pages?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Welke alinea's hebben `topic` als grootste aandeel, en van welke pagina's?"},
            "almost": {"en": "Right for `topic_pages(small_mixes, small_pages, 0)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `topic` it gets, and work for any number of topics?",
                       "nl": "Goed voor `topic_pages(small_mixes, small_pages, 0)`. Maar `{call}` geeft {got}, en het moet {want} zijn. Gebruikt je functie de `topic` uit de aanroep, en werkt de functie voor elk aantal topics?"},
            "correct": {"en": "Your `topic_pages` is right in all {n} cases.",
                        "nl": "Je `topic_pages` klopt in alle {n} gevallen."},
        },
    ),
}

CASES = [
    ("topic_words", _TW_START, "empty", None),
    ("topic_words", _TW, "correct", None),
    ("topic_words", _tw(best="..."), "wrong", "dots-left"),
    ("topic_words", _tw(best="np.argsort(-row)[:3]"), "almost", None),
    ("topic_words", _tw(add="[(words[j], 0.5) for j in best]"), "wrong", None),
    ("topic_words", _TW_HEAD + "    return 3\n", "wrong", "not-list"),
    ("topic_words", _tw(probs="weights / weights.sum(axis=1)"), "error", "broadcast"),
    ("topic_words", _tw(probs="weights"), "wrong", "counts"),
    ("topic_words", _tw(probs="weights / weights.sum()"), "wrong", "total"),
    ("topic_words", _tw(probs="weights / weights.sum(axis=0)"), "wrong", "columns"),
    ("topic_words", _tw(best="np.argsort(row)[:n]"), "wrong", "smallest"),
    ("topic_words", _tw(best="np.argsort(row)[-n:]"), "wrong", "reversed"),
    ("topic_words", _tw(best="np.argsort(-row)"), "wrong", "all-words"),
    ("topic_words", _tw(best="np.argsort(row)[::-1][:n]"), "correct", None),
    ("topic_words", "import numpy as np\n\n\ndef topic_words(weights, words, n):\n    found = []\n    for row in weights:\n"
     "        total = row.sum()\n        pairs = sorted([(row[j] / total, words[j]) for j in range(len(words))], reverse=True)\n"
     "        found.append([(w, p) for p, w in pairs[:n]])\n    return found\n", "correct", None),
    ("word_topics", _WT_START, "empty", None),
    ("word_topics", _WT, "correct", None),
    ("word_topics", _WT_HEAD + "        found.append(...)\n    return found\n", "wrong", "dots-left"),
    ("word_topics", _wt(add="0"), "wrong", None),
    ("word_topics", _WT_HEAD + "        found.append(1)\n    return len(found)\n", "wrong", "not-list"),
    ("word_topics", _wt(add="int(np.argmax(mix * topics[j]))"), "error", "row"),
    ("word_topics", _wt(add="int(np.argmax(topics[:, j]))"), "wrong", "word-only"),
    ("word_topics", _wt(add="int(np.argmax(mix))"), "wrong", "mix-only"),
    ("word_topics", _wt(add="float(np.max(mix * topics[:, j]))"), "wrong", "shares"),
    ("word_topics", _wt(end="        return found\n    return found\n"), "wrong", "return-in-loop"),
    ("word_topics", _wt(add="int(np.argmax(mix[:2] * topics[:2, j]))"), "almost", None),
    ("word_topics", _WT_HEAD + "        found.append(np.argmax(mix * topics[:, j]))\n    return found\n", "correct", None),
    ("topic_pages", _TP_START, "empty", None),
    ("topic_pages", _TP, "correct", None),
    ("topic_pages", _tp(test="..."), "wrong", "dots-left"),
    ("topic_pages", _TP_HEAD + "        counts[pages[i]] = 2\n    return counts\n", "wrong", None),
    ("topic_pages", _tp(end="    return list(counts)\n"), "wrong", "not-dict"),
    ("topic_pages", _tp(test="True"), "wrong", "all"),
    ("topic_pages", _tp(test="mix[i][topic] > 0.5"), "wrong", "half"),
    ("topic_pages", _tp(add="1"), "wrong", "ones"),
    ("topic_pages", _tp(add="counts[pages[i]] + mix[i][topic]", first="mix[i][topic]"), "wrong", "shares"),
    ("topic_pages", _tp(end="        return counts\n    return counts\n"), "wrong", "return-in-loop"),
    ("topic_pages", _tp(test="np.argmax(mix[i]) == 0"), "almost", None),
    ("topic_pages", "import numpy as np\n\n\ndef topic_pages(mix, pages, topic):\n    counts = {}\n"
     "    for i in range(len(pages)):\n        if np.argmax(mix[i]) == topic:\n"
     "            counts[pages[i]] = counts.get(pages[i], 0) + 1\n    return counts\n", "correct", None),
    ("topic_pages", "import numpy as np\n\n\ndef topic_pages(mix, pages, topic):\n    counts = {}\n"
     "    for i in range(len(pages)):\n        if np.argmax(mix[i]) == topic:\n            if pages[i] not in counts:\n"
     "                counts[pages[i]] = 0\n            counts[pages[i]] = counts[pages[i]] + 1\n    return counts\n", "correct", None),
]
