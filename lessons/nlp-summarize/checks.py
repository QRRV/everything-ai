# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-summarize: an extractive summarizer that scores each sentence of an abstract by the words it
shares with the title, weighed by their idf (title_score), and picks the best sentences in the order of the text (pick);
and the novelty of a summary, the share of its words that are not in the text (novel_share).

The six abstracts are the course's own, made up in the style of abstracts in computer science, each with a title and a
TLDR that the course wrote. Qwen3-0.6B summarized each in one sentence (data/course-summarize-qwen.json). The real
abstracts are those of the test set of SciTLDR (Cachola and others, 2020), which everything_ai.scitldr downloads.
Every number in the text and the pictures is computed by the course (tests/test_nlp_summarize.py computes them again)."""
import ast
import html
import json
import math
import re

from everything_ai import books, lesson as lesson_helpers, paths, scitldr, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

LANGS = ("en", "nl")
_KEY = "nlp-summarize"

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The test set of SciTLDR, 618 real abstracts with their TLDRs, is downloaded once from its repository on GitHub, {download_size(scitldr.FILE, 'en')}, when you click its button in the lesson; the demo on real abstracts needs it, the rest of the lesson works without it.",
    "nl": f"Werkt op elke computer. De testset van SciTLDR, 618 echte abstracts met de TLDR's erbij, wordt één keer gedownload van de repository op GitHub, {download_size(scitldr.FILE, 'nl')}, als je in de les op de knop van de download klikt; de demo met echte abstracts heeft de set nodig, de rest van de les werkt zonder.",
}

# The course's six abstracts: a title, the sentences, and a TLDR the course wrote.
ABSTRACTS = {
    "bees": {
        "title": "Counting Bees at the Hive Entrance with a Small Neural Network",
        "sentences": (
            "Beekeepers need to know how many bees leave and return to a hive, because a sudden drop can be an early sign of disease.",
            "Counting by hand takes hours, and existing camera systems need a powerful computer next to the hive.",
            "We present a small neural network that counts bees at the hive entrance from a cheap camera and runs on a board that costs less than 40 euros.",
            "The network finds each bee in a frame and follows it over the next frames, so that a bee that turns around is not counted twice.",
            "We recorded 41 days of video at three hives and counted the bees in 600 short clips by hand.",
            "On these clips, the network is off by 4.1% on average, against 11.8% for a counter that only detects motion.",
            "The board uses 2.5 watts, so a small solar panel can power it for a whole season.",
        ),
        "tldr": "A small neural network on a cheap board counts the bees that leave and enter a hive from camera video, with an error of about 4%.",
    },
    "river": {
        "title": "Forecasting River Levels from Rainfall with a Recurrent Network",
        "sentences": (
            "Small rivers can rise within hours after heavy rain, which leaves the towns along them little time to prepare.",
            "Physical models of a river need years of measurements to set up, and many small rivers have only a few gauges.",
            "We propose a recurrent network that forecasts the level of a river six hours ahead from the rainfall of the last two days.",
            "It is trained on one river and fine-tuned on another with only three months of data.",
            "We test it on twelve rivers in the hills of one region, over five years.",
            "For rises of more than half a meter, the network warns 2.3 hours earlier on average than the warnings in use, with fewer false alarms.",
            "Without rainfall forecasts, the network still beats the physical model for the first three hours.",
        ),
        "tldr": "A recurrent network forecasts the level of small rivers six hours ahead from recent rainfall and warns of floods earlier than current warnings.",
    },
    "plastic": {
        "title": "Sorting Plastic Waste by Its Infrared Spectrum",
        "sentences": (
            "Recycling plants must separate the kinds of plastic, since a mix of kinds makes weak new material.",
            "Dark plastic is hard to sort, as it absorbs the near-infrared light that most sorting machines use.",
            "We show that the mid-infrared spectrum of a piece of plastic tells six common kinds apart, dark pieces included.",
            "A logistic regression on 300 points of the spectrum labels each piece in 2 milliseconds, fast enough for a moving belt.",
            "On 9,000 pieces from two recycling plants, 96.5% get the right label, and 91% of the dark pieces.",
            "Most mistakes are between two kinds of polyethylene that differ mainly in density.",
        ),
        "tldr": "The mid-infrared spectrum lets a fast logistic regression sort six kinds of plastic waste, including dark pieces that current machines miss.",
    },
    "leaks": {
        "title": "Finding Leaks in Water Pipes by Listening",
        "sentences": (
            "Water companies lose a large share of their drinking water through leaks in pipes under the street.",
            "Leaks are found by experts who listen to the pipes at night, one street at a time.",
            "We train a convolutional network on sound from sensors on fire hydrants to tell the hiss of a leak from traffic and from water in use.",
            "The network looks at spectrograms of one second and votes over ten minutes of sound per sensor.",
            "In a test with 150 sensors over four months, it found 23 of the 27 leaks that crews later dug up, and raised 9 false alarms.",
            "Leaks in plastic pipes are harder to hear than leaks in metal ones, and most of the leaks it missed were in plastic.",
        ),
        "tldr": "A convolutional network that listens to sensors on fire hydrants finds most leaks in water pipes, though fewer in plastic pipes.",
    },
    "wind": {
        "title": "Day-Ahead Wind Power Forecasts from Weather Maps",
        "sentences": (
            "Grid operators must know a day ahead how much power the wind farms will deliver.",
            "Forecasts from a single weather station miss the weather that is coming toward a farm.",
            "We feed maps of a weather model into a convolutional network that forecasts the power of a wind farm for each hour of the next day.",
            "The maps cover 200 kilometers around the farm, so that the network sees fronts before they arrive.",
            "Over two years at four farms at sea, the error drops by 14% compared with a forecast from the nearest grid point of the same weather model.",
            "The gain is largest on days when a front passes, and small on calm days.",
        ),
        "tldr": "A convolutional network reads weather maps around a wind farm and forecasts its power for the next day 14% better than the nearest grid point.",
    },
    "leaves": {
        "title": "Spotting Diseased Tomato Leaves with a Phone Camera",
        "sentences": (
            "Diseases of tomato plants spread fast in a greenhouse, and growers need to find the first sick leaves.",
            "Models trained on photos taken in a lab often fail on photos taken between the plants, with other light and a busy background.",
            "We fine-tune a pretrained network on 4,800 photos of tomato leaves taken with phones in eight greenhouses.",
            "The photos show healthy leaves and leaves with one of five diseases, each checked by a plant expert.",
            "On photos from two greenhouses that were not used in training, the model finds 89% of the diseased leaves, against 61% for the same network trained on lab photos.",
            "Early stages of a disease, with only a few small spots, cause most of the misses.",
        ),
        "tldr": "A pretrained network fine-tuned on phone photos from greenhouses spots diseased tomato leaves far better than the same network trained on lab photos.",
    },
}
WORKED = "bees"   # the abstract of the worked example

# How the course asked Qwen3-0.6B for a summary (data/tools/record_course_summarize_qwen.py).
INSTRUCTION = "Summarize this abstract in one sentence."
MAX_TOKENS = 120


def summarize_messages(key: str) -> list:
    """The chat that asks for a summary of one of the course's abstracts: the instruction, the title and the text."""
    a = ABSTRACTS[key]
    return [{"role": "user", "content": f"{INSTRUCTION}\n\nTitle: {a['title']}\n\n{' '.join(a['sentences'])}"}]
QWEN_FILE = paths.ROOT / "data" / "course-summarize-qwen.json"


def _qwen() -> dict:
    data = json.loads(QWEN_FILE.read_text(encoding="utf-8"))
    return {key: data["abstracts"][key]["text"] for key in ABSTRACTS}


QWEN = _qwen()   # Qwen3-0.6B's summary of each abstract, as the course recorded it

# The claims of Qwen3-0.6B's summaries that the abstract does not support, in the course's reading: the words of the
# summary, and whether they twist what the abstract says (intrinsic) or add what it does not say (extrinsic), the two
# kinds of Maynez and others (2020).
UNSUPPORTED = {
    "bees": [("improving accuracy compared to existing systems", "intrinsic")],
    "river": [("outperforming physical models without rainfall forecasts", "intrinsic")],
    "leaks": [("outperforming traditional methods", "extrinsic")],
    "leaves": [("A phone camera model trained on lab tomato leaf photos performs poorly", "intrinsic"),
               ("real-time", "extrinsic")],
}
WHY_UNSUPPORTED = {
    "en": {"bees": "The abstract compares the accuracy of the network only with a counter that only detects motion; of the camera systems that exist, it says only that they need a powerful computer.",
           "river": "Without rainfall forecasts, the network beats the physical model only for the first three hours.",
           "leaks": "The abstract compares the network with no other method.",
           "leaves": "The summary calls the model that does poorly a phone camera model. In the abstract, the network trained further on phone photos finds 89% of the diseased leaves, and the network trained on lab photos is the one that does worse. And the abstract says nothing about real time."},
    "nl": {"bees": "Het abstract vergelijkt hoe goed het netwerk telt alleen met een teller die alleen beweging ziet; van de camerasystemen die er al zijn, zegt het abstract alleen dat deze systemen een krachtige computer nodig hebben.",
           "river": "Zonder regenvoorspellingen verslaat het netwerk het fysische model alleen in de eerste drie uur.",
           "leaks": "Het abstract vergelijkt het netwerk met geen enkele andere methode.",
           "leaves": "De samenvatting noemt het model dat slecht presteert een model voor een telefooncamera. In het abstract vindt het netwerk dat verder is getraind op telefoonfoto's 89% van de zieke bladeren, en het netwerk dat op foto's uit een lab is getraind, doet het slechter. En het abstract zegt niets over realtime."},
}


# --- the words, the idf and the three functions, as the course writes them -------------------------------------------------

_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")   # a word as everything_ai.books finds one


def words(text: str) -> list:
    """The words of a text in small letters, as in *TF-IDF: weighing the words*: runs of letters, digits left out."""
    return books.words(text.lower())


def document_idf(docs) -> dict:
    """The idf of every word of `docs`, a list of documents, each a set of words: log10(n / df), as in *TF-IDF:
    weighing the words*."""
    df = {}
    for doc in docs:
        for word in doc:
            df[word] = df.get(word, 0) + 1
    return {word: math.log10(len(docs) / n) for word, n in df.items()}


def _document(title: str, sentences) -> set:
    """The words of an abstract as one document: the title and every sentence."""
    out = set(words(title))
    for s in sentences:
        out |= set(words(s))
    return out


IDF = document_idf([_document(a["title"], a["sentences"]) for a in ABSTRACTS.values()])   # over the six abstracts


def title_score(sentence, title, idf):
    """The title method: the idf of each different word of the sentence that is also in the title, added up."""
    score = 0
    for word in set(sentence):
        if word in title:
            score = score + idf[word]
    return score


def pick(scores, k):
    """The places of the k highest scores, in the order of the text; of two equal scores, the earlier place first."""
    pairs = []
    for i in range(len(scores)):
        pairs.append((-scores[i], i))
    chosen = []
    for pair in sorted(pairs)[:k]:
        chosen.append(pair[1])
    return sorted(chosen)


def novel_share(summary, source):
    """The novelty of a summary: the share of its different words that are not in the source."""
    words = set(summary)
    new = 0
    for word in words:
        if word not in source:
            new = new + 1
    return new / len(words)


def weight_score(sentence, counts, idf):
    """Word weights, after Luhn: the TF-IDF weight of each different word of the sentence, with its count in the whole
    abstract, added up."""
    score = 0
    for word in set(sentence):
        score = score + counts[word] * idf[word]
    return score


METHODS = ("title", "weights", "first")


def even(scores) -> list:
    """Scores rounded to nine decimals, so that two equal scores are equal: a loop over a set adds the same words in
    another order from one run of Python to the next, and the last bits of a float then differ, which would decide a
    tie at random instead of for the earlier sentence."""
    return [round(v, 9) if isinstance(v, float) else v for v in scores]


def method_scores(title: str, sentences, idf: dict, method: str, title_score_=title_score) -> list:
    """The score of each sentence by one method: title words, word weights, or the place (the first sentence highest)."""
    lists = [words(s) for s in sentences]
    if method == "first":
        return [len(sentences) - i for i in range(len(sentences))]
    if method == "weights":
        counts = {}
        for ws in lists:
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
        return even([weight_score(ws, counts, idf) for ws in lists])
    t = words(title)
    return even([title_score_(list(ws), list(t), dict(idf)) for ws in lists])


def course_scores(key: str, method: str = "title") -> list:
    a = ABSTRACTS[key]
    return method_scores(a["title"], a["sentences"], IDF, method)


def summary(key: str, method: str = "title", k: int = 1) -> list:
    """The places of the sentences of a course abstract that a method picks."""
    return pick(course_scores(key, method), k)


def source_words(key: str) -> list:
    """The words of a course abstract: its title and its sentences, as one list."""
    a = ABSTRACTS[key]
    return words(a["title"] + " " + " ".join(a["sentences"]))


def new_words(text: str, source) -> list:
    """The different words of a text that are not in the source, in the order of the text."""
    seen, out = set(source), []
    for w in words(text):
        if w not in seen and w not in out:
            out.append(w)
    return out


# --- the real abstracts: SciTLDR ---------------------------------------------------------------------------------------------

def scitldr_measures(papers: list) -> dict:
    """How often each method picks the oracle sentence of an abstract, over `papers` (everything_ai.scitldr.papers()),
    with the idf over every abstract and its title; the share a random sentence would get; and the novelty of the
    authors' TLDRs."""
    idf = document_idf([_document(p["title"], p["sentences"]) for p in papers])
    hits = {m: 0 for m in METHODS}
    for p in papers:
        for m in METHODS:
            if pick(method_scores(p["title"], p["sentences"], idf, m), 1) == [p["oracle"]]:
                hits[m] += 1
    novelty = [novel_share(words(p["tldrs"][0]), words(p["title"] + " " + " ".join(p["sentences"]))) for p in papers
               if words(p["tldrs"][0])]
    return {"papers": len(papers), "hits": hits, "random": sum(1 / len(p["sentences"]) for p in papers),
            "novelty": sum(novelty) / len(novelty), "idf": idf}


def scitldr_or_stop(button, texts: dict) -> list:
    """The papers of SciTLDR for the lesson's cells, once the file is downloaded (everything_ai.scitldr.papers()). Until
    then the cell stops with `texts["ask"]` and the button; a click downloads it, with the progress in the cell."""
    lesson_helpers.pins_or_stop(button, texts, [scitldr.FILE])
    return scitldr.papers()


def _measures(papers: list) -> dict:
    """scitldr_measures of the whole test set, worked out once in a process."""
    if len(papers) == len(scitldr.papers()):
        return kept(f"{_KEY}:measures", lambda: scitldr_measures(scitldr.papers()))
    return scitldr_measures(papers)


# --- the words of the six abstracts, as the worked example and the checks name them --------------------------------------

TITLE_WORDS = {key: words(a["title"]) for key, a in ABSTRACTS.items()}
SENTENCE_WORDS = {key: [words(s) for s in a["sentences"]] for key, a in ABSTRACTS.items()}
SOURCE_WORDS = {key: source_words(key) for key in ABSTRACTS}
TLDR_WORDS = {key: words(a["tldr"]) for key, a in ABSTRACTS.items()}
QWEN_WORDS = {key: words(QWEN[key]) for key in ABSTRACTS}
SCORES = {key: course_scores(key) for key in ABSTRACTS}   # the score of each sentence by title words


def closest(key: str) -> int:
    """The place of the sentence of a course abstract that shares the largest share of the TLDR's different words; of
    two equal shares, the earlier: the course's stand-in for the oracle sentence of SciTLDR."""
    tldr = set(TLDR_WORDS[key])
    shares = [len(tldr & set(ws)) / len(tldr) for ws in SENTENCE_WORDS[key]]
    return pick(shares, 1)[0]


def numbers() -> dict:
    """The numbers the text states about the six abstracts, worked out from them."""
    n = {"abstracts": len(ABSTRACTS), "sentences": sum(len(a["sentences"]) for a in ABSTRACTS.values()),
         "words": len(IDF), "everywhere": sorted(w for w, v in IDF.items() if v == 0)}
    for m in METHODS:
        n["hits_" + m] = sum(1 for key in ABSTRACTS if summary(key, m) == [closest(key)])
    n["tldr_novelty"] = sum(novel_share(TLDR_WORDS[k], SOURCE_WORDS[k]) for k in ABSTRACTS) / len(ABSTRACTS)
    n["qwen_novelty"] = sum(novel_share(QWEN_WORDS[k], SOURCE_WORDS[k]) for k in ABSTRACTS) / len(ABSTRACTS)
    n["unsupported"] = len(UNSUPPORTED)
    n["marks"] = sum(len(v) for v in UNSUPPORTED.values())
    return n


# --- styles and small parts of the pictures ------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_LABEL = "font-family:var(--ea-font-ui);font-size:0.8rem;color:var(--ea-ink-2);margin-top:0.45rem"
_CELL = ("padding:0.12rem 0.4rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--ea-line);"
         "font-family:var(--ea-font-ui);font-size:0.84rem")
_HEAD = _CELL + ";color:var(--ea-ink-2);font-weight:600"
_NUMBER = ";text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap"
_DATA = "color:var(--ea-data-text)"
_FAINT = "color:var(--ea-ink-2)"
_OUT = "color:var(--ea-pred-text)"
_PRED = "color:var(--ea-pred-text);font-weight:600"
_LOSS = "color:var(--ea-loss-text);font-weight:600"
_CHIP = ("display:inline-block;margin:0 0.1rem;padding:0 0.25rem;border-radius:3px;"
         "line-height:1.45")
_CHIP_PRED = _CHIP + ";background:var(--ea-pred-wash);color:var(--ea-pred-text)"
_CHIP_DATA = _CHIP + ";background:var(--ea-data-wash);color:var(--ea-data-text);font-weight:600"
_CHIP_LOSS = _CHIP + ";background:var(--ea-loss-wash);color:var(--ea-loss-text);font-weight:600"
_TEXT_CAP = 120   # characters of a learner's value that a picture shows


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


def _colors(w: dict, body: str) -> str:
    """The note on the colors under a picture: gray always, blue and purple only when the picture draws them."""
    parts = [w["gray"]]
    if "var(--ea-pred-text)" in body:
        parts.append(w["blue"])
    if "var(--ea-loss-text)" in body:
        parts.append(w["purple"])
    return _note(" ".join(parts))


def _label(text_: str) -> str:
    return f'<div style="{_LABEL}">{_marked(text_)}</div>'


def _num(n, lang: str) -> str:
    """A whole number with a separator for thousands: 1,234 in English, 1.234 in Dutch."""
    out = f"{round(n):,}"
    return out.replace(",", ".") if lang == "nl" else out


def _dec(x: float, lang: str, digits: int = 2) -> str:
    """A number with a fixed number of decimals, with a decimal comma in Dutch."""
    out = f"{x + 0.0:.{digits}f}"
    return out.replace(".", ",") if lang == "nl" else out


def _pct(x: float, lang: str, digits: int = 1) -> str:
    """A share as a percentage: 0.359 as 35.9% (35,9% in Dutch)."""
    return _dec(100 * x, lang, digits) + "%"


def _short(v) -> str:
    """A learner's value as text, cut to _TEXT_CAP characters."""
    try:
        out = v if isinstance(v, str) else repr(v)
    except Exception:   # noqa: BLE001   a value that cannot even be written
        out = "?"
    return out if len(out) <= _TEXT_CAP else out[:_TEXT_CAP] + "…"


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _score_text(v, lang: str) -> str:
    """A score as the pictures write it: two decimals; anything else a learner's function gives, as it is."""
    return _dec(v, lang) if _is_number(v) and abs(v) < 1e9 else html.escape(_short(v))


def _bar(x, most: float, style: str = "var(--ea-pred)", width: float = 7.0) -> str:
    w = 0.0 if not _is_number(x) or x <= 0 or not most else max(0.15, width * min(x, most) / most)
    return (f'<span style="display:inline-block;width:{w:.2f}rem;height:0.6rem;background:{style};'
            f'border-radius:2px;margin-right:0.35rem;vertical-align:middle"></span>')


def _cell(text_: str, style: str = "", extra: str = "") -> str:
    return f'<td style="{_CELL}{extra};{style}">{text_}</td>'


def _table(rows: list, head: list, numbers: tuple = ()) -> str:
    top = "".join(f'<th style="{_HEAD}{_NUMBER if j in numbers else ""}">{html.escape(h)}</th>' for j, h in enumerate(head))
    return (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;margin:0.3rem 0 0.2rem">'
            f"<thead><tr>{top}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _words_html(text_: str, marked, chip: str, bad_spans=(), bad_chip: str = _CHIP_LOSS) -> str:
    """A text with each word whose small letters are in `marked` on a background (`chip`), and the parts of the text in
    `bad_spans`, a list of (start, end), in purple, with their marked words on `bad_chip`."""
    def bad(i):
        return any(a <= i < b for a, b in bad_spans)
    pieces, pos = [], 0
    for m in list(_WORD.finditer(text_)) + [None]:
        start = m.start() if m else len(text_)
        for j in range(pos, start):   # the signs and spaces between words, one by one
            pieces.append((html.escape(text_[j]), bad(j)))
        if m is None:
            break
        w = m.group()
        inner = html.escape(w)
        if w.lower() in marked:
            inner = f'<span style="{bad_chip if bad(m.start()) else chip}">{inner}</span>'
        pieces.append((inner, bad(m.start())))
        pos = m.end()
    out, run = [], []
    for piece, is_bad in pieces + [("", False)]:
        if is_bad:
            run.append(piece)
            continue
        if run:
            out.append(f'<span style="{_LOSS}">{"".join(run)}</span>')
            run = []
        out.append(piece)
    return "".join(out)


def _spans(text_: str, marks) -> list:
    """Where each marked claim stands in a text, as (start, end)."""
    out = []
    for phrase, _ in marks:
        i = text_.find(phrase)
        if i >= 0:
            out.append((i, i + len(phrase)))
    return out


def _qwen_html(key: str) -> str:
    """Qwen3's summary of a course abstract: the words that are not in the abstract on a background, and the claims
    that the abstract does not support in purple."""
    text_ = QWEN[key]
    return _words_html(text_, set(new_words(text_, SOURCE_WORDS[key])), _CHIP_PRED, _spans(text_, UNSUPPORTED.get(key, ())))


def _tldr_html(key: str) -> str:
    text_ = ABSTRACTS[key]["tldr"]
    return _words_html(text_, set(new_words(text_, SOURCE_WORDS[key])), _CHIP_DATA)


def _abstract_table(title: str, sentences, scores, chosen, lang: str, want_scores=None, want_chosen=None,
                    oracle: int | None = None, one_tldr: bool = False) -> str:
    """The sentences of an abstract with their place, the title words on a background, and the score of each as a bar.
    The chosen sentences are blue; where `want_*` is given and differs, purple. With `oracle`, the column *closest*
    marks the oracle sentence, and the chosen sentences are purple when the oracle is not among them."""
    w = _TABLE_TEXT[lang]
    t = set(words(title))
    good = [_is_number(v) for v in scores]
    most = max([v for v, g in zip(scores, good) if g] + [1e-9])
    rows = []
    for i, s in enumerate(sentences):
        right_score = want_scores is None or (good[i] and math.isclose(scores[i], want_scores[i], rel_tol=1e-6, abs_tol=1e-9))
        picked = i in chosen
        right_pick = (want_chosen is None or (i in want_chosen) == picked) and (oracle is None or not picked or oracle in chosen)
        text_style = (_PRED if right_pick else _LOSS) if picked else (_LOSS if not right_pick else _DATA)
        cells = [_cell(str(i), _FAINT, _NUMBER), _cell(_words_html(s, t, _CHIP_DATA), text_style)]
        bar_color = "var(--ea-pred)" if right_score else "var(--ea-loss)"
        cells.append(_cell(_bar(scores[i], most, bar_color) + _score_text(scores[i], lang), _OUT if right_score else _LOSS,
                           _NUMBER))
        if oracle is not None:
            cells.append(_cell(w["oracle_one" if one_tldr else "oracle"] if i == oracle else "", _DATA))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = w["head"] + ([w["closest"]] if oracle is not None else [])
    return _label(w["title"].format(title=title)) + _table(rows, head, (0, 2))


_TABLE_TEXT = {
    "en": {"head": ["Place", "Sentence", "Score"], "title": "*{title}*", "closest": "Closest", "oracle": "closest to the TLDRs",
           "oracle_one": "closest to the TLDR"},
    "nl": {"head": ["Plaats", "Zin", "Score"], "title": "*{title}*", "closest": "Dichtstbij", "oracle": "dichtst bij de TLDR's",
           "oracle_one": "dichtst bij de TLDR"},
}


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


# --- the idea: pick a sentence, or write one ---------------------------------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Two summaries of one abstract: a picked sentence and a written one",
           "pick": "Extractive: the sentence with the highest score, place {i}.",
           "write": "Abstractive: Qwen3-0.6B's summary of the same abstract, with {p} of its different words not in the abstract.",
           "gray": "Gray: the abstract, what goes in; a word with a background is also in the title.",
           "blue": "Blue: what the summarizers make: the score of each sentence by its title words, the picked sentence, and the written summary, where a word with a background is not in the abstract.",
           "purple": "Purple: a claim of the written summary that the abstract does not support, in the course's reading."},
    "nl": {"title": "Twee samenvattingen van één abstract: een gekozen zin en een geschreven zin",
           "pick": "Extractief: de zin met de hoogste score, plaats {i}.",
           "write": "Abstractief: de samenvatting van Qwen3-0.6B van hetzelfde abstract; van de verschillende woorden staan er {p} niet in het abstract.",
           "gray": "Grijs: het abstract, wat erin gaat; een woord met een achtergrond staat ook in de titel.",
           "blue": "Blauw: wat de samenvatters maken: de score van elke zin op grond van de titelwoorden, de gekozen zin, en de geschreven samenvatting, waarin een woord met een achtergrond niet in het abstract staat.",
           "purple": "Paars: een bewering van de geschreven samenvatting die het abstract niet ondersteunt, zoals de course het leest."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    a = ABSTRACTS[WORKED]
    scores = course_scores(WORKED)
    chosen = pick(scores, 1)
    body = _abstract_table(a["title"], a["sentences"], scores, chosen, lang)
    body += _label(w["pick"].format(i=chosen[0]))
    body += f'<div style="{_PRED}">{html.escape(a["sentences"][chosen[0]])}</div>'
    body += _label(w["write"].format(p=_pct(novel_share(QWEN_WORDS[WORKED], SOURCE_WORDS[WORKED]), lang, 0)))
    body += f'<div style="{_OUT}">{_qwen_html(WORKED)}</div>'
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the pictures under the exercises ----------------------------------------------------------------------------------------

_TITLE_TEXT = {
    "en": {"title": "Your title_score on the abstract about bees",
           "summary": "Over the {n} sentences of the six abstracts, your title_score gives the course's score for {k}. In {h} of the {a} abstracts, the sentence with your highest score is the one closest to the TLDR.",
           "gray": "Gray: the abstract; a word with a background is also in the title.",
           "blue": "Blue: the score your title_score gives each sentence, and the sentence with the highest score, when they are what the course's title_score gives.",
           "purple": "Purple: a score or a choice that differs from what the course's title_score gives.",
           "kind": "a number"},
    "nl": {"title": "Je title_score op het abstract over bijen",
           "summary": "Van de {n} zinnen van de zes abstracts geeft je title_score er {k} dezelfde score als de title_score van de course. In {h} van de {a} abstracts is de zin met je hoogste score de zin die het dichtst bij de TLDR ligt.",
           "gray": "Grijs: het abstract; een woord met een achtergrond staat ook in de titel.",
           "blue": "Blauw: de score die je title_score elke zin geeft, en de zin met de hoogste score, als dat is wat de title_score van de course geeft.",
           "purple": "Paars: een score of een keuze die anders is dan bij de title_score van de course.",
           "kind": "een getal"},
}


def title_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _TITLE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        return {k: [fn(list(ws), list(TITLE_WORDS[k]), dict(IDF)) for ws in SENTENCE_WORDS[k]] for k in ABSTRACTS}

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_number(v) for k in mine for v in mine[k]):
        return _empty(w, "odd")
    want = {k: course_scores(k) for k in ABSTRACTS}
    a = ABSTRACTS[WORKED]
    body = _abstract_table(a["title"], a["sentences"], mine[WORKED], pick(even(mine[WORKED]), 1), lang, want[WORKED],
                           pick(want[WORKED], 1))
    n = sum(len(v) for v in want.values())
    k = sum(1 for key in ABSTRACTS for x, y in zip(mine[key], want[key]) if math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9))
    h = sum(1 for key in ABSTRACTS if pick(even(mine[key]), 1) == [closest(key)])
    summary_ = w["summary"].format(n=_num(n, lang), k=_num(k, lang), h=_num(h, lang), a=_num(len(ABSTRACTS), lang))
    return yours.figure(w["title"], body + _note(summary_) + _colors(w, body), alt=w["title"])


_PICK_TEXT = {
    "en": {"title": "Your pick of two sentences from each abstract",
           "head": ["Abstract", "Scores by title words", "Your pick"],
           "summary": "Your pick gives the course's two places, in the order of the text, for {k} of the {a} abstracts.",
           "worked": "The summary of the abstract about bees, with your pick:",
           "gray": "Gray: the titles of the six abstracts, and the scores the course's title_score gives their sentences.",
           "blue": "Blue: the places your pick gives, and the summary they make, when they are what the course's pick gives.",
           "purple": "Purple: a place or an order that differs from what the course's pick gives.",
           "kind": "a list of places in the abstract"},
    "nl": {"title": "Je pick van twee zinnen uit elk abstract",
           "head": ["Abstract", "Scores op titelwoorden", "Je pick"],
           "summary": "Je pick geeft de twee plaatsen van de course, in de volgorde van de tekst, bij {k} van de {a} abstracts.",
           "worked": "De samenvatting van het abstract over bijen, met je pick:",
           "gray": "Grijs: de titels van de zes abstracts, en de scores die de title_score van de course aan de zinnen geeft.",
           "blue": "Blauw: de plaatsen die je pick geeft, en de samenvatting van deze zinnen, als dat is wat de pick van de course geeft.",
           "purple": "Paars: een plaats of een volgorde die anders is dan bij de pick van de course.",
           "kind": "een lijst met plaatsen in het abstract"},
}
PICK_K = 2   # how many sentences the picture under pick picks


def _is_places(v, n: int) -> bool:
    return isinstance(v, list) and all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in v)


def pick_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _PICK_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    scores = {k: course_scores(k) for k in ABSTRACTS}
    mine = _call(lambda: {k: _as_list(fn(list(scores[k]), PICK_K)) for k in ABSTRACTS})
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_places(mine[k], len(scores[k])) for k in ABSTRACTS):
        return _empty(w, "odd")
    rows = []
    for k, a in ABSTRACTS.items():
        want = pick(scores[k], PICK_K)
        chips = "".join(f'<span style="{(_CHIP_PRED if i in want else _CHIP_LOSS) if i in mine[k] else _CHIP}">'
                        f'{i}: {_dec(v, lang)}</span>' for i, v in enumerate(scores[k]))
        same = mine[k] == want
        rows.append("<tr>" + _cell(f"<i>{html.escape(a['title'])}</i>", _DATA) + _cell(chips, _DATA)
                    + _cell(html.escape(_short(mine[k])), _PRED if same else _LOSS, _NUMBER) + "</tr>")
    body = _table(rows, w["head"])
    good = mine[WORKED] == pick(scores[WORKED], PICK_K)
    body += _label(w["worked"]) + f'<div style="{_PRED if good else _LOSS}">' + " ".join(
        html.escape(ABSTRACTS[WORKED]["sentences"][i]) for i in mine[WORKED]) + "</div>"
    k = sum(1 for key in ABSTRACTS if mine[key] == pick(scores[key], PICK_K))
    summary_ = w["summary"].format(k=_num(k, lang), a=_num(len(ABSTRACTS), lang))
    return yours.figure(w["title"], body + _note(summary_) + _colors(w, body), alt=w["title"])


_NOVEL_TEXT = {
    "en": {"title": "Your novel_share on Qwen3-0.6B's six summaries",
           "head": ["Abstract", "Qwen3-0.6B's summary", "New words"],
           "summary": "On average, your novel_share gives {q} for Qwen3-0.6B's six summaries, and {t} for the six TLDRs that the course wrote.",
           "differs": "The course's novel_share gives {q} and {t}.",
           "gray": "Gray: the titles of the six abstracts.",
           "blue": "Blue: Qwen3-0.6B's summaries, where a word with a background is not in the abstract, and the share of new words your novel_share gives, when it is what the course's novel_share gives.",
           "purple": "Purple: a claim that the abstract does not support, in the course's reading, and a share that differs from what the course's novel_share gives.",
           "kind": "a number"},
    "nl": {"title": "Je novel_share op de zes samenvattingen van Qwen3-0.6B",
           "head": ["Abstract", "Samenvatting van Qwen3-0.6B", "Nieuwe woorden"],
           "summary": "Gemiddeld geeft je novel_share {q} voor de zes samenvattingen van Qwen3-0.6B, en {t} voor de zes TLDR's die de course schreef.",
           "differs": "De novel_share van de course geeft {q} en {t}.",
           "gray": "Grijs: de titels van de zes abstracts.",
           "blue": "Blauw: de samenvattingen van Qwen3-0.6B, waarin een woord met een achtergrond niet in het abstract staat, en het aandeel nieuwe woorden dat je novel_share geeft, als dat is wat de novel_share van de course geeft.",
           "purple": "Paars: een bewering die het abstract niet ondersteunt, zoals de course het leest, en een aandeel dat anders is dan bij de novel_share van de course.",
           "kind": "een getal"},
}


def novel_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _NOVEL_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")

    def make():
        return ({k: fn(list(QWEN_WORDS[k]), list(SOURCE_WORDS[k])) for k in ABSTRACTS},
                {k: fn(list(TLDR_WORDS[k]), list(SOURCE_WORDS[k])) for k in ABSTRACTS})

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_number(v) and abs(v) < 1e9 for d in mine for v in d.values()):
        return _empty(w, "odd")
    want_q = {k: novel_share(QWEN_WORDS[k], SOURCE_WORDS[k]) for k in ABSTRACTS}
    want_t = {k: novel_share(TLDR_WORDS[k], SOURCE_WORDS[k]) for k in ABSTRACTS}
    rows = []
    for k, a in ABSTRACTS.items():
        same = math.isclose(mine[0][k], want_q[k], rel_tol=1e-6, abs_tol=1e-9)
        rows.append("<tr>" + _cell(f"<i>{html.escape(a['title'])}</i>", _DATA) + _cell(_qwen_html(k), _OUT)
                    + _cell(_pct(mine[0][k], lang, 0), _PRED if same else _LOSS, _NUMBER) + "</tr>")
    body = _table(rows, w["head"], (2,))
    mean = lambda d: sum(d.values()) / len(d)   # noqa: E731
    summary_ = w["summary"].format(q=_pct(mean(mine[0]), lang), t=_pct(mean(mine[1]), lang))
    if not all(math.isclose(mine[i][k], want[k], rel_tol=1e-6, abs_tol=1e-9)
               for i, want in ((0, want_q), (1, want_t)) for k in ABSTRACTS):
        summary_ += " " + w["differs"].format(q=_pct(mean(want_q), lang), t=_pct(mean(want_t), lang))
    return yours.figure(w["title"], body + _note(summary_) + _colors(w, body), alt=w["title"])


# --- try it: an abstract, a method and a length of the learner's choice ------------------------------------------------------

METHOD_NAMES = {"en": {"title": "Title words", "weights": "Word weights", "first": "First sentences"},
                "nl": {"title": "Titelwoorden", "weights": "Gewogen woorden", "first": "Eerste zinnen"}}
ABSTRACT_NAMES = {"en": {"bees": "Bees", "river": "Rivers", "plastic": "Plastic", "leaks": "Leaks", "wind": "Wind power",
                         "leaves": "Tomato leaves"},
                  "nl": {"bees": "Bijen", "river": "Rivieren", "plastic": "Plastic", "leaks": "Lekken", "wind": "Windstroom",
                         "leaves": "Tomatenbladeren"}}

_TRY_TEXT = {
    "en": {"title": "Summaries of the abstract about {name}",
           "method": {"title": "Each sentence scores the idf of the words it shares with the title.",
                      "weights": "Each sentence scores the TF-IDF weights of its different words, with their counts in the whole abstract.",
                      "first": "An earlier sentence scores higher: the score is the number of sentences from the end."},
           "picked": "Picked: {k} of the {n} sentences, in the order of the text.",
           "qwen": "Written by Qwen3-0.6B, with {p} of its different words not in the abstract:",
           "tldr": "The TLDR that the course wrote, with {p} of its different words not in the abstract:",
           "gray": "Gray: the abstract and the TLDR, what the summaries are compared with; a word with a background is in the title, or, in the TLDR, not in the abstract.",
           "blue": "Blue: the scores, the picked sentences, and Qwen3-0.6B's summary, where a word with a background is not in the abstract.",
           "purple": "Purple: picked sentences that leave out the one closest to the TLDR, and a claim of Qwen3-0.6B's summary that the abstract does not support, in the course's reading.",
           "purple_pick": "Purple: picked sentences that leave out the one closest to the TLDR.",
           "purple_claim": "Purple: a claim of Qwen3-0.6B's summary that the abstract does not support, in the course's reading."},
    "nl": {"title": "Samenvattingen van het abstract over {name}",
           "method": {"title": "Elke zin scoort de idf van de woorden die de zin met de titel deelt.",
                      "weights": "Elke zin scoort de TF-IDF-waarden van de verschillende woorden, elk met het aantal in het hele abstract.",
                      "first": "Een eerdere zin scoort hoger: de score is het aantal zinnen tot het eind."},
           "picked": "Gekozen: {k} van de {n} zinnen, in de volgorde van de tekst.",
           "qwen": "Geschreven door Qwen3-0.6B; van de verschillende woorden staan er {p} niet in het abstract:",
           "tldr": "De TLDR die de course schreef; van de verschillende woorden staan er {p} niet in het abstract:",
           "gray": "Grijs: het abstract en de TLDR, waarmee de samenvattingen worden vergeleken; een woord met een achtergrond staat in de titel, of, in de TLDR, niet in het abstract.",
           "blue": "Blauw: de scores, de gekozen zinnen, en de samenvatting van Qwen3-0.6B, waarin een woord met een achtergrond niet in het abstract staat.",
           "purple": "Paars: gekozen zinnen zonder de zin die het dichtst bij de TLDR ligt, en een bewering van de samenvatting van Qwen3-0.6B die het abstract niet ondersteunt, zoals de course het leest.",
           "purple_pick": "Paars: gekozen zinnen zonder de zin die het dichtst bij de TLDR ligt.",
           "purple_claim": "Paars: een bewering van de samenvatting van Qwen3-0.6B die het abstract niet ondersteunt, zoals de course het leest."},
}


def try_picture(lang: str, key: str, method: str = "title", k=1) -> str:
    lang = lang if lang in LANGS else "en"
    key = key if key in ABSTRACTS else WORKED
    method = method if method in METHODS else "title"
    a = ABSTRACTS[key]
    try:
        k = min(max(int(k), 1), len(a["sentences"]))
    except (TypeError, ValueError):
        k = 1
    scores = course_scores(key, method)
    chosen = pick(scores, k)
    w = dict(_TRY_TEXT[lang])
    miss, claim = closest(key) not in chosen, key in UNSUPPORTED
    if not (miss and claim):
        w["purple"] = w["purple_pick"] if miss else w["purple_claim"]
    body = _note(w["method"][method])
    body += _abstract_table(a["title"], a["sentences"], scores, chosen, lang, oracle=closest(key), one_tldr=True)
    body += _label(w["picked"].format(k=_num(k, lang), n=_num(len(a["sentences"]), lang)))
    body += f'<div style="{_LOSS if miss else _PRED}">' + " ".join(html.escape(a["sentences"][i]) for i in chosen) + "</div>"
    body += _label(w["qwen"].format(p=_pct(novel_share(QWEN_WORDS[key], SOURCE_WORDS[key]), lang, 0)))
    body += f'<div style="{_OUT}">{_qwen_html(key)}</div>'
    if key in UNSUPPORTED:
        body += f'<div style="{_UI};{_LOSS};font-weight:400">{html.escape(WHY_UNSUPPORTED[lang][key])}</div>'
    body += _label(w["tldr"].format(p=_pct(novel_share(TLDR_WORDS[key], SOURCE_WORDS[key]), lang, 0)))
    body += f'<div style="{_DATA}">{_tldr_html(key)}</div>'
    title = w["title"].format(name=ABSTRACT_NAMES[lang][key].lower())
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the demo on SciTLDR ------------------------------------------------------------------------------------------------------

_SCI_TEXT = {
    "en": {"title": "Three ways to pick one sentence, on the {n} abstracts of SciTLDR",
           "bars": "How often the picked sentence is the one closest to the TLDRs:",
           "random": "A random sentence (expected)",
           "paper": "Paper {i} of {n}: the scores by title words.",
           "hit": "Here the title method picks the sentence closest to the TLDRs.",
           "miss": "Here the title method picks another sentence than the one closest to the TLDRs.",
           "authors": "The TLDR of the authors:",
           "reviews": "TLDRs that experts wrote from the peer reviews:",
           "gray": "Gray: the abstract and its TLDRs, and the share a random sentence would get; a word with a background is also in the title.",
           "blue": "Blue: how often each method picks the closest sentence, and in this paper the scores and the picked sentence, when it is the closest one.",
           "purple": "Purple: a picked sentence that is not the closest one."},
    "nl": {"title": "Drie manieren om één zin te kiezen, op de {n} abstracts van SciTLDR",
           "bars": "Hoe vaak de gekozen zin de zin is die het dichtst bij de TLDR's ligt:",
           "random": "Een willekeurige zin (verwacht)",
           "paper": "Paper {i} van {n}: de scores op titelwoorden.",
           "hit": "Hier kiest de titelmethode de zin die het dichtst bij de TLDR's ligt.",
           "miss": "Hier kiest de titelmethode een andere zin dan de zin die het dichtst bij de TLDR's ligt.",
           "authors": "De TLDR van de auteurs:",
           "reviews": "TLDR's die experts schreven op grond van de peer reviews:",
           "gray": "Grijs: het abstract en de TLDR's, en het aandeel dat een willekeurige zin zou halen; een woord met een achtergrond staat ook in de titel.",
           "blue": "Blauw: hoe vaak elke methode de dichtstbijzijnde zin kiest, en in deze paper de scores en de gekozen zin, als dat de dichtstbijzijnde is.",
           "purple": "Paars: een gekozen zin die niet de dichtstbijzijnde is."},
}
SCI_ORDER = ("first", "weights", "title")   # the methods in the order of the bars
SCI_START = 57                               # the paper the slider starts at, counted from 1


def scitldr_picture(lang: str, papers: list, number=SCI_START) -> str:
    lang = lang if lang in LANGS else "en"
    w = _SCI_TEXT[lang]
    m = _measures(papers)
    n = m["papers"]
    rows = []
    for name in SCI_ORDER:
        share = m["hits"][name] / n
        rows.append("<tr>" + _cell(html.escape(METHOD_NAMES[lang][name]), _OUT) + _cell(
            _bar(share, 1.0, "var(--ea-pred)", 12) + _pct(share, lang), _OUT, _NUMBER) + "</tr>")
    share = m["random"] / n
    rows.append("<tr>" + _cell(html.escape(w["random"]), _DATA) + _cell(_bar(share, 1.0, "var(--ea-data)", 12)
                                                                         + _pct(share, lang), _DATA, _NUMBER) + "</tr>")
    body = _label(w["bars"]) + _table(rows, ["", ""], (1,)).replace("<thead><tr><th", "<thead style=\"display:none\"><tr><th", 1)
    try:
        i = min(max(int(number), 1), n)
    except (TypeError, ValueError, OverflowError):
        i = min(SCI_START, n)
    p = papers[i - 1]
    scores = method_scores(p["title"], p["sentences"], m["idf"], "title")
    chosen = pick(scores, 1)
    body += _label(w["paper"].format(i=_num(i, lang), n=_num(n, lang)))
    body += _abstract_table(p["title"], p["sentences"], scores, chosen, lang, oracle=p["oracle"])
    body += _note(w["hit"] if chosen == [p["oracle"]] else w["miss"])
    body += _label(w["authors"]) + f'<div style="{_DATA}">{html.escape(p["tldrs"][0])}</div>'
    if len(p["tldrs"]) > 1:
        body += _label(w["reviews"]) + "".join(f'<div style="{_DATA}">{html.escape(t)}</div>' for t in p["tldrs"][1:])
    title = w["title"].format(n=_num(n, lang))
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

class _Val:
    """A test value that the check's messages write by its name, such as `C.IDF`; the function gets a copy."""

    def __init__(self, name: str, value):
        self.name, self.value = name, value

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        import copy
        return copy.deepcopy(self.value, memo)


def _value(v) -> str:
    """A value the way code writes it, with double quotes, and a float with at most three decimals."""
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        return repr(round(v, 3)) if math.isfinite(v) and abs(v) < 1e15 else repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    try:
        return _short(v)
    except Exception:   # noqa: BLE001
        return "?"


def _show(v):
    """A value for a message, the way the lesson writes it, cut when it is very long."""
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _written(v) -> _Val:
    """A small test value that the messages write the way the lesson writes it, with double quotes."""
    return _Val(_value(v), v)


def _as_list(v):
    """What pick gives, with a tuple of places made a list: (0, 2) counts as [0, 2]."""
    return list(v) if isinstance(v, tuple) else v


def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return re.search(r"^[^#\n]*\.\.\.", text_, re.MULTILINE) is not None
    return any(isinstance(n, ast.Constant) and n.value is Ellipsis for n in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang de `...` door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# The small cases the exercise texts show; the same in both lessons.
TOY_TITLE = ((["bees", "count", "bees", "at", "the", "hive"], ["counting", "bees", "at", "the", "hive"],
              {"bees": 0.8, "count": 0.3, "at": 0.2, "the": 0.0, "hive": 0.8, "counting": 0.6}),
             (["het", "netwerk", "telt", "bijen"], ["bijen", "tellen", "met", "een", "netwerk"],
              {"het": 0.0, "netwerk": 0.1, "telt": 0.8, "bijen": 0.8, "tellen": 0.8, "met": 0.0, "een": 0.0}))
TOY_PICK = (([0.8, 2.6, 1.1, 2.6, 1.1], 3), ([0.5, 0.0, 1.9], 1))
TOY_NOVEL = ((["tiny", "network", "counts", "bees", "bees"], ["we", "present", "a", "small", "network", "that", "counts", "bees"]),
             (["het", "netwerk", "telt", "bijen"], ["we", "tellen", "bijen", "met", "een", "netwerk"]))

_TS_HEAD = "def title_score(sentence, title, idf):\n    score = 0\n    for word in set(sentence):\n"
_TS_IF = "        if word in title:\n"
_TS_ADD = "            score = score + idf[word]\n"


def _ts(check: str = _TS_IF, add: str = _TS_ADD) -> str:
    return _TS_HEAD + check + add + "    return score\n"


_TS = _ts()
_TS_START = _ts("        if ...:\n", "            ...\n")
_TS_COUNT = _ts(add="            score = score + 1\n")
_TS_ALL = _ts("        if word in sentence:\n")
_TS_NOT = _ts("        if word not in title:\n")
_TS_REPLACE = _ts(add="            score = idf[word]\n")
_TS_INSIDE = _TS_HEAD + _TS_IF + _TS_ADD + "        return score\n"

_PK_HEAD = "def pick(scores, k):\n    pairs = []\n    for i in range(len(scores)):\n"
_PK_ADD = "        pairs.append((-scores[i], i))\n"
_PK_MID = "    chosen = []\n    for pair in sorted(pairs)[:k]:\n        chosen.append(pair[1])\n"
_PK_RET = "    return sorted(chosen)\n"


def _pk(add: str = _PK_ADD, ret: str = _PK_RET) -> str:
    return _PK_HEAD + add + _PK_MID + ret


_PK = _pk()
_PK_START = _pk("        pairs.append(...)\n", "    return ...\n")
_PK_LOWEST = _pk("        pairs.append((scores[i], i))\n")
_PK_LEAD = _pk("        pairs.append((i, scores[i]))\n")
_PK_RANK = _pk(ret="    return chosen\n")
_PK_SORT = _pk(ret="    return chosen.sort()\n")

_NS_HEAD = "def novel_share(summary, source):\n    words = set(summary)\n    new = 0\n    for word in words:\n"
_NS_IF = "        if word not in source:\n"
_NS_RET = "    return new / len(words)\n"


def _ns(check: str = _NS_IF, ret: str = _NS_RET) -> str:
    return _NS_HEAD + check + "            new = new + 1\n" + ret


_NS = _ns()
_NS_START = _ns("        if ...:\n", "    return ...\n")
_NS_SHARED = _ns("        if word in source:\n")
_NS_ALL = _ns(ret="    return new / len(summary)\n")
_NS_SOURCE = _ns(ret="    return new / len(source)\n")
_NS_COUNT = _ns(ret="    return new\n")
_NS_PERCENT = _ns(ret="    return new / len(words) * 100\n")
_NS_FLOOR = _ns(ret="    return new // len(words)\n")


def _not_number(got, text_) -> bool:
    return not _is_number(got)


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _sorted_in_place(got, text_) -> bool:
    """A `return` of `.sort()`, which changes a list and gives None: found in the code, before it runs."""
    return re.search(r"return\s+[\w\[\]]+\.sort\(", text_) is not None


def _append_two(e) -> bool:
    return isinstance(e, TypeError) and "append" in str(e)


REQUIRED = ["title_score", "pick", "novel_share", "risks"]

_IDF_VAL = _Val("C.IDF", IDF)


def _sentence_val(key: str, i: int) -> _Val:
    return _Val(f'C.SENTENCE_WORDS["{key}"][{i}]', SENTENCE_WORDS[key][i])


def _named(table: str, key: str) -> _Val:
    return _Val(f'C.{table}["{key}"]', globals()[table][key])


EXERCISES = {
    "title_score": FunctionBox(
        name="title_score",
        picture=title_picture,
        function="title_score",
        start=_TS_START,
        solution=_TS,
        show=_show,
        visible=[tuple(_written(x) for x in case) for case in TOY_TITLE],
        hidden=[(_written(["we", "test", "it"]), _written(["sorting", "plastic"]), _written({"we": 0.0, "test": 0.5, "it": 0.3, "sorting": 0.8, "plastic": 0.8})),
                (_written(["plastic", "waste"]), _written(["plastic", "waste"]), _written({"plastic": 0.5, "waste": 0.7})),
                (_written(["rain", "rain", "rain"]), _written(["rain"]), _written({"rain": 0.25}))]
               + [(_sentence_val(key, i), _named("TITLE_WORDS", key), _IDF_VAL)
                  for key, i in (("bees", 0), ("bees", 2), ("plastic", 1), ("leaks", 5), ("wind", 2))],
        first=[_DOTS],
        mistakes=[
            Mistake("list-key", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="`{call}` stops with a `TypeError`: a list cannot be a key of a dictionary, and your code looks up a whole list in `idf`. Which one word do you look up?",
                    nl="`{call}` stopt met een `TypeError`: een lijst kan geen key van een dictionary zijn, en je code zoekt een hele lijst op in `idf`. Welk ene woord zoek je op?"),
            Mistake("whole-dict", error=lambda e: isinstance(e, TypeError) and "unsupported operand" in str(e),
                    en="`{call}` stops with a `TypeError`: Python cannot add what your code adds to `score`, such as the whole dictionary `idf`. What is the idf of one word?",
                    nl="`{call}` stopt met een `TypeError`: Python kan niet optellen wat je code bij `score` optelt, zoals de hele dictionary `idf`. Wat is de idf van één woord?"),
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number. Does your function give back `score`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal. Geeft je functie `score` terug?"),
            Mistake("counted", wrong=_TS_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: your function counts the shared words, each as 1. A word such as *the* is in every abstract and should weigh nothing: what does each shared word add?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de gedeelde woorden, elk als 1. Een woord als *the* staat in elk abstract en hoort niets te wegen: wat voegt elk gedeeld woord toe?"),
            Mistake("all-words", wrong=_TS_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the idf of every different word of the sentence, also of *count*, which is not in the title. Which words may add to the score?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de idf van elk verschillend woord van de zin op, ook van *count*, dat niet in de titel staat. Welke woorden mogen meetellen?"),
            Mistake("not-title", wrong=_TS_NOT,
                    en="`{call}` gives {got}, and it should give {want}: your function adds the idf of the words that are not in the title. Which side of the `in` test should a word be on?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie telt de idf op van de woorden die niet in de titel staan. Aan welke kant van de test met `in` moet een woord zitten?"),
            Mistake("replaced", wrong=_TS_REPLACE,
                    en="`{call}` gives {got}, and it should give {want}: each shared word replaces the score, so only the last one counts. How does a word add to what `score` already holds?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk gedeeld woord vervangt de score, dus alleen het laatste telt. Hoe tel je een woord op bij wat `score` al bevat?"),
            Mistake("return-inside", wrong=_TS_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first word. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt na het eerste woord. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether `word` is also in the title, and the line under it adds the idf of that word to `score`.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of `word` ook in de titel staat, en de regel eronder telt de idf van dat woord op bij `score`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For each different word of the sentence: is it in the title? Then its idf goes on top of the score. Does each word count once?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Staat elk verschillend woord van de zin ook in de titel? Dan komt de idf van het woord bij de score. Telt elk woord één keer?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give 0 when the sentence shares no word with the title, and count a word that comes three times only once?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie 0 als de zin geen woord met de titel deelt, en telt de functie een woord dat drie keer voorkomt slechts één keer?"},
            "correct": {"en": "Right: your function scores a sentence by its title words. In the picture below, which sentence of the abstract about bees wins?",
                        "nl": "Goed: je functie geeft een zin een score op grond van de titelwoorden. Welke zin van het abstract over bijen wint, in het plaatje hieronder?"},
        },
    ),
    "pick": FunctionBox(
        name="pick",
        picture=pick_picture,
        function="pick",
        start=_PK_START,
        solution=_PK,
        show=_show,
        collect=_as_list,
        visible=[(_written(a), _written(b)) for a, b in TOY_PICK],
        hidden=[(_written([1.0, 2.0, 3.0]), _written(3)),
                (_written([0.4, 0.9]), _written(0)),
                (_written([1.5, 1.5, 1.5, 1.5]), _written(2)),
                (_written([0.0, 0.7, 0.2, 0.7, 0.9, 0.1]), _written(2))]
               + [(_Val(f'C.SCORES["{key}"]', SCORES[key]), _written(2)) for key in ("bees", "plastic", "leaves")],
        first=[_DOTS,
               Mistake("sort-none", matches=_sorted_in_place,
                    en="Your `return` gives `None`: `chosen.sort()` changes the list and gives nothing back, as in *Tuples, sets and sorting*. Which function gives a new list in order?",
                    nl="Je `return` geeft `None`: `chosen.sort()` verandert de lijst en geeft niets terug, zoals in *Tuples, sets en sorteren*. Welke functie geeft een nieuwe lijst op volgorde?")],
        mistakes=[
            Mistake("append-two", error=_append_two,
                    en="`{call}` stops with a `TypeError`: `append` takes one item, and your code gives it two. How do the two values become one tuple?",
                    nl="`{call}` stopt met een `TypeError`: `append` neemt één item, en je code geeft er twee. Hoe worden de twee waarden samen één tuple?"),
            Mistake("minus-list", error=lambda e: isinstance(e, TypeError) and "unary" in str(e),
                    en="`{call}` stops with a `TypeError`: Python cannot put a minus before a whole list. Which score belongs to place `i`?",
                    nl="`{call}` stopt met een `TypeError`: Python kan geen min voor een hele lijst zetten. Welke score hoort bij plaats `i`?"),
            Mistake("not-pair", error=lambda e: isinstance(e, TypeError) and "subscriptable" in str(e),
                    en="`{call}` stops with a `TypeError`: `pair[1]` takes the second value of a pair, and your code puts one number in `pairs`, not a tuple of two. A pair holds two values: which two?",
                    nl="`{call}` stopt met een `TypeError`: `pair[1]` neemt de tweede waarde van een paar, en je code zet één getal in `pairs`, geen tuple van twee. Een paar bevat twee waarden: welke twee?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of places. Does your function give back the chosen places in order?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met plaatsen. Geeft je functie de gekozen plaatsen op volgorde terug?"),
            Mistake("lowest", wrong=_PK_LOWEST,
                    en="`{call}` gives {got}, and it should give {want}: the places of the lowest scores. `sorted` goes from small to large, so the first pairs have the lowest score. What could you put first in each pair, so that the highest score comes first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de plaatsen van de laagste scores. `sorted` gaat van klein naar groot, dus de eerste paren hebben de laagste score. Wat kun je vooraan in elk paar zetten, zodat de hoogste score eerst komt?"),
            Mistake("lead", wrong=_PK_LEAD,
                    en="`{call}` gives {got}, and it should give {want}: the first places of the text, whatever their score. `sorted` sorts the pairs by their first value: which value should that be?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de eerste plaatsen van de tekst, wat de score ook is. `sorted` sorteert de paren op de eerste waarde: welke waarde moet dat zijn?"),
            Mistake("rank-order", wrong=_PK_RANK,
                    en="`{call}` gives {got}, and it should give {want}: the right places, but from the highest score down. A summary keeps the sentences in the order of the text: what do you do with `chosen` before you give it back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de goede plaatsen, maar van de hoogste score naar beneden. Een samenvatting houdt de zinnen in de volgorde van de tekst: wat doe je met `chosen` voordat je de lijst teruggeeft?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: `pairs.append` adds for each place a tuple that `sorted` can put in the order you want, and the last line gives the chosen places back in the order of the text.",
                      "nl": "Vervang de twee `...`: `pairs.append` voegt voor elke plaats een tuple toe die `sorted` in de volgorde zet die je wilt, en de laatste regel geeft de gekozen plaatsen terug in de volgorde van de tekst."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Are they the places of the k highest scores, the earlier place first on a tie, and in the order of the text?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Zijn het de plaatsen van de k hoogste scores, bij gelijke scores de eerdere plaats eerst, en in de volgorde van de tekst?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function work when k is 0, when k is the number of sentences, and when every score is the same?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Werkt je functie als k 0 is, als k het aantal zinnen is, en als elke score gelijk is?"},
            "correct": {"en": "Right: your function picks the best sentences and keeps their order. In the picture below, where does a tie decide which sentence comes along?",
                        "nl": "Goed: je functie kiest de beste zinnen en houdt de volgorde van de tekst. Waar beslist een gelijke score welke zin meegaat, in het plaatje hieronder?"},
        },
    ),
    "novel_share": FunctionBox(
        name="novel_share",
        picture=novel_picture,
        function="novel_share",
        start=_NS_START,
        solution=_NS,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_NOVEL],
        hidden=[(_written(["small", "network"]), _written(["a", "small", "network"])),
                (_written(["fast", "sorting"]), _written(["plastic", "waste"])),
                (_written(["leaks", "leaks", "pipes"]), _written(["pipes"])),
                (_written(["wind"]), _written(["wind", "power"]))]
               + [(_named("QWEN_WORDS", key), _named("SOURCE_WORDS", key)) for key in ("bees", "river", "leaves")]
               + [(_named("TLDR_WORDS", key), _named("SOURCE_WORDS", key)) for key in ("plastic", "wind")],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, a share between 0 and 1. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, een aandeel tussen 0 en 1. Wat geeft je functie terug?"),
            Mistake("shared", wrong=_NS_SHARED,
                    en="`{call}` gives {got}, and it should give {want}: the share of words that are in the source, so the share that was copied. Which words count as new?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aandeel woorden dat in de bron staat, dus het aandeel dat is overgenomen. Welke woorden tellen als nieuw?"),
            Mistake("all-words", wrong=_NS_ALL,
                    en="`{call}` gives {got}, and it should give {want}: you divide by every word of the summary, *bees* twice, while `new` counts each different word once. What is the number of different words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door elk woord van de samenvatting, *bees* twee keer, terwijl `new` elk verschillend woord één keer telt. Wat is het aantal verschillende woorden?"),
            Mistake("source-length", wrong=_NS_SOURCE,
                    en="`{call}` gives {got}, and it should give {want}: you divide by the words of the source. The share is about the summary: out of which words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door de woorden van de bron. Het aandeel gaat over de samenvatting: van welke woorden?"),
            Mistake("count", wrong=_NS_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of new words, not their share. What do you divide `new` by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal nieuwe woorden, niet het aandeel. Waardoor deel je `new`?"),
            Mistake("percent", wrong=_NS_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: a percentage. The function gives a share between 0 and 1, as the other shares of this lesson do: what is left without the `* 100`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een percentage. De functie geeft een aandeel tussen 0 en 1, zoals de andere aandelen van deze les: wat blijft er over zonder de `* 100`?"),
            Mistake("floor", wrong=_NS_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: `//` divides and throws away what is after the point, so a share below 1 becomes 0. Which sign divides and keeps it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `//` deelt en gooit weg wat na de komma komt, dus een aandeel onder 1 wordt 0. Welk teken deelt zonder iets weg te gooien?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `if` asks whether `word` is missing from the source, and the `return` gives the new words as a share of the different words of the summary.",
                      "nl": "Vervang de twee `...`: de `if` vraagt of `word` ontbreekt in de bron, en de `return` geeft de nieuwe woorden als aandeel van de verschillende woorden van de samenvatting."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. How many different words of the summary are not in the source, and what share is that of the different words of the summary?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Hoeveel verschillende woorden van de samenvatting staan niet in de bron, en welk aandeel is dat van de verschillende woorden van de samenvatting?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give `0.0` for a summary of copied words, `1.0` for one of new words only, and count a word that comes twice once?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `0.0` bij een samenvatting van overgenomen woorden, `1.0` bij een samenvatting van alleen nieuwe woorden, en telt de functie een woord dat twee keer voorkomt één keer?"},
            "correct": {"en": "Right: your function measures how much a summary writes in words of its own. In the picture below, are the claims that the abstract does not support made of new words or of copied ones?",
                        "nl": "Goed: je functie meet hoeveel een samenvatting in eigen woorden schrijft. Bestaan de beweringen die het abstract niet ondersteunt uit nieuwe of uit overgenomen woorden, in het plaatje hieronder?"},
        },
    ),
}

EXERCISES["risks"] = Sort(
    name="risks",
    groups={
        "extractive": {"en": "Picking sentences (extractive)", "nl": "Zinnen kiezen (extractief)"},
        "abstractive": {"en": "Writing new sentences (abstractive)", "nl": "Nieuwe zinnen schrijven (abstractief)"},
    },
    items={
        "dangling": {"en": "**This method.** The summary starts with *This method counts each bee once*, and the sentence that named the method was left out.",
                     "nl": "**Deze methode.** De samenvatting begint met *Deze methode telt elke bij één keer*, en de zin die de methode noemde, is weggelaten."},
        "twice": {"en": "**Twice the same.** The text says twice that the network is small, in two sentences with many title words, and the summary holds both sentences.",
                  "nl": "**Twee keer hetzelfde.** De tekst zegt twee keer dat het netwerk klein is, in twee zinnen met veel titelwoorden, en de samenvatting bevat allebei de zinnen."},
        "baggage": {"en": "**Along for the ride.** The summary names the price of the board, because the sentence with the main point also holds the price.",
                    "nl": "**Meeliftend detail.** De samenvatting noemt de prijs van het bordje, omdat de zin met de kern ook de prijs bevat."},
        "jumpy": {"en": "**No thread.** The sentences of the summary do not follow from each other: they stood apart in the text.",
                  "nl": "**Geen lijn.** De zinnen van de samenvatting volgen niet op elkaar: in de tekst stonden de zinnen ver uit elkaar."},
        "number": {"en": "**A new number.** The summary says the error is 1.4%, a number that is nowhere in the abstract.",
                   "nl": "**Een nieuw getal.** De samenvatting zegt dat de fout 1,4% is, een getal dat nergens in het abstract staat."},
        "stronger": {"en": "**Stronger.** The abstract says the network found 23 of 27 leaks, and the summary that it finds every leak.",
                     "nl": "**Sterker.** Het abstract zegt dat het netwerk 23 van de 27 lekken vond, en de samenvatting dat het elk lek vindt."},
        "swapped": {"en": "**Swapped.** The summary says the network trained on lab photos did best, where the abstract says the one trained on phone photos did.",
                    "nl": "**Omgedraaid.** De samenvatting zegt dat het netwerk dat op foto's uit een lab is getraind, het best was, terwijl volgens het abstract het netwerk dat op telefoonfoto's is getraind, het best was."},
        "outside": {"en": "**From outside.** The summary names a country where the hives stand, and the abstract names none.",
                    "nl": "**Van buitenaf.** De samenvatting noemt een land waar de bijenkasten staan, en het abstract noemt er geen."},
    },
    solution={"dangling": "extractive", "twice": "extractive", "baggage": "extractive", "jumpy": "extractive",
              "number": "abstractive", "stronger": "abstractive", "swapped": "abstractive", "outside": "abstractive"},
    nudge={
        "dangling": {"en": "*This method* stands in the text, word for word, but what it refers to was left behind. Which kind of summary takes sentences whole?",
                     "nl": "*Deze methode* staat letterlijk in de tekst, maar waarnaar het verwijst, bleef achter. Welke soort samenvatting neemt hele zinnen over?"},
        "twice": {"en": "Two sentences of the text can say nearly the same, and both can score high. Which kind of summary would put both in?",
                  "nl": "Twee zinnen van de tekst kunnen bijna hetzelfde zeggen, en allebei hoog scoren. Welke soort samenvatting zou allebei de zinnen opnemen?"},
        "baggage": {"en": "A sentence comes with all its words, or not at all. Which kind of summary cannot leave out half a sentence?",
                    "nl": "Een zin komt met alle woorden, of helemaal niet. Welke soort samenvatting kan geen halve zin weglaten?"},
        "jumpy": {"en": "The sentences are good on their own; what is missing is the link between them. Which kind of summary puts sentences of the text side by side?",
                  "nl": "De zinnen zijn op zich goed; wat ontbreekt, is het verband ertussen. Welke soort samenvatting zet zinnen van de tekst naast elkaar?"},
        "number": {"en": "Could a summary that only takes sentences from the abstract contain a number that the abstract does not?",
                   "nl": "Kan een samenvatting die alleen zinnen uit het abstract overneemt een getal bevatten dat niet in het abstract staat?"},
        "stronger": {"en": "The abstract says 23 of 27, and no sentence of it says every leak. Which kind of summary can put it in other words, and so change it?",
                     "nl": "Het abstract zegt 23 van de 27, en geen enkele zin ervan zegt elk lek. Welke soort samenvatting kan het in andere woorden zeggen, en het zo veranderen?"},
        "swapped": {"en": "Both networks are in the abstract, but the summary mixes up which one did what. Which kind of summary writes its own sentences about them?",
                    "nl": "Beide netwerken staan in het abstract, maar de samenvatting haalt door elkaar welk netwerk wat deed. Welke soort samenvatting schrijft er eigen zinnen over?"},
        "outside": {"en": "The country comes from outside the abstract. Which kind of summary can add words that the text does not have?",
                    "nl": "Het land komt van buiten het abstract. Welke soort samenvatting kan woorden toevoegen die niet in de tekst staan?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Put each risk with the kind of summary that runs it.",
                  "nl": "Zet elk risico bij de soort samenvatting die het loopt."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other kind of summary.",
                  "nl": "{item} hoort bij de andere soort samenvatting."},
        "correct": {"en": "All {n} are right. A summary that picks sentences says only what the text says, but takes each sentence whole, with what it refers to and what it carries; a summary that writes reads better, but can say what the text does not.",
                    "nl": "Alle {n} kloppen. Een samenvatting van gekozen zinnen zegt alleen wat de tekst zegt, maar neemt hele zinnen over, met waarnaar een zin verwijst en wat een zin meedraagt; een samenvatting die schrijft, leest beter, maar kan zeggen wat niet in de tekst staat."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("title_score", _TS_START, "empty", None),
    ("title_score", _TS, "correct", None),
    ("title_score", _ts(add="            ...\n"), "wrong", "dots-left"),
    ("title_score", _ts(add="            score = score + idf[title]\n"), "error", "list-key"),
    ("title_score", _ts(add="            score = score + idf\n"), "error", "whole-dict"),
    ("title_score", _ts().replace("    return score\n", "    return [score]\n"), "wrong", "not-number"),
    ("title_score", _TS_COUNT, "wrong", "counted"),
    ("title_score", _TS_ALL, "wrong", "all-words"),
    ("title_score", _ts("        if word in idf:\n"), "wrong", "all-words"),
    ("title_score", _TS_NOT, "wrong", "not-title"),
    ("title_score", _TS_REPLACE, "wrong", "replaced"),
    ("title_score", _TS_INSIDE, "wrong", "return-inside"),
    ("title_score", "def title_score(sentence, title, idf):\n    score = 0\n    for word in set(title):\n"
     "        if word in sentence:\n            score = score + idf[word]\n    return score\n", "correct", None),
    ("title_score", "def title_score(sentence, title, idf):\n    return sum([idf[w] for w in set(sentence) if w in title])\n",
     "correct", None),
    ("title_score", _ts(add="            score += idf[word]\n"), "correct", None),
    ("pick", _PK_START, "empty", None),
    ("pick", _PK, "correct", None),
    ("pick", _pk(ret="    return ...\n"), "wrong", "dots-left"),
    ("pick", _pk("        pairs.append(-scores[i], i)\n"), "error", "append-two"),
    ("pick", _pk("        pairs.append((-scores, i))\n"), "error", "minus-list"),
    ("pick", _pk("        pairs.append(-scores[i])\n"), "error", "not-pair"),
    ("pick", _PK_SORT, "wrong", "sort-none"),
    ("pick", _pk(ret="    return len(chosen)\n"), "wrong", "not-list"),
    ("pick", _PK_LOWEST, "wrong", "lowest"),
    ("pick", _PK_LEAD, "wrong", "lead"),
    ("pick", _PK_RANK, "wrong", "rank-order"),
    ("pick", _pk("        pairs.append((0 - scores[i], i))\n"), "correct", None),
    ("pick", _pk(ret="    return tuple(sorted(chosen))\n"), "correct", None),
    ("pick", "def pick(scores, k):\n    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))\n"
     "    return sorted(order[:k])\n", "correct", None),
    ("novel_share", _NS_START, "empty", None),
    ("novel_share", _NS, "correct", None),
    ("novel_share", _ns(ret="    return ...\n"), "wrong", "dots-left"),
    ("novel_share", _ns(ret="    return str(new / len(words))\n"), "wrong", "not-number"),
    ("novel_share", _NS_SHARED, "wrong", "shared"),
    ("novel_share", _NS_ALL, "wrong", "all-words"),
    ("novel_share", _NS_SOURCE, "wrong", "source-length"),
    ("novel_share", _NS_COUNT, "wrong", "count"),
    ("novel_share", _NS_PERCENT, "wrong", "percent"),
    ("novel_share", _NS_FLOOR, "wrong", "floor"),
    ("novel_share", "def novel_share(summary, source):\n    return len(set(summary) - set(source)) / len(set(summary))\n",
     "correct", None),
    ("novel_share", _ns("        if not word in source:\n"), "correct", None),
    ("novel_share", "def novel_share(summary, source):\n    words = set(summary)\n    shared = 0\n    for word in words:\n"
     "        if word in source:\n            shared = shared + 1\n    return 1 - shared / len(words)\n", "correct", None),
    ("risks", {}, "empty", None),
    ("risks", dict(EXERCISES["risks"].solution), "correct", None),
    ("risks", {"dangling": "extractive", "twice": "extractive"}, "wrong", "unfinished"),
] + [("risks", {item: "abstractive" if group == "extractive" else "extractive"}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["risks"].solution.items()]
