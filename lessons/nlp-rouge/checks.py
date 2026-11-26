# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson nlp-rouge: ROUGE (Lin, 2004), which scores a summary by how much of a reference summary comes
back in it. ROUGE-N is the recall of the n-grams of the reference, clipped as in BLEU (rouge_n); ROUGE-L divides the
longest common subsequence of the two by the length of the reference (lcs); ROUGE-S counts the skip-bigrams of the
reference, every pair of words in their order with any gap between them (skip_bigrams).

The six abstracts, their TLDRs and Qwen3-0.6B's summaries are those of nlp-summarize, loaded from that lesson's checks.
The real abstracts are those of the test set of SciTLDR (Cachola and others, 2020), which everything_ai.scitldr
downloads; this lesson scores four kinds of summary against the TLDRs of the experts. Every number in the text and the
pictures is computed by the course (tests/test_nlp_rouge.py computes them again)."""
import html
import importlib.util
import math
import re

import numpy as np

from everything_ai import paths, scitldr, yours
from everything_ai.check import FunctionBox, Mistake, Sort, kept
from everything_ai.lesson import download_size

LANGS = ("en", "nl")
_KEY = "nlp-rouge"


def _summarize():
    spec = importlib.util.spec_from_file_location("everything_ai_checks_nlp_summarize_for_rouge",
                                                  paths.LESSONS / "nlp-summarize" / "checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _summarize()   # the six abstracts, Qwen3-0.6B's summaries, the title method and the parts of the pictures

ABSTRACTS, QWEN, UNSUPPORTED, WHY_UNSUPPORTED = S.ABSTRACTS, S.QWEN, S.UNSUPPORTED, S.WHY_UNSUPPORTED
ABSTRACT_NAMES = S.ABSTRACT_NAMES
words, pick, closest, summary = S.words, S.pick, S.closest, S.summary
scitldr_or_stop = S.scitldr_or_stop

DOWNLOAD_LINE = {   # what the lesson downloads, for the line under the header
    "en": f"Works on every computer. The test set of SciTLDR, 618 real abstracts with their TLDRs, is downloaded once from its repository on GitHub, {download_size(scitldr.FILE, 'en')}, when you click its button in the lesson, unless *Summarizing: pick or write* already did; the demo on real abstracts needs it, the rest of the lesson works without it.",
    "nl": f"Werkt op elke computer. De testset van SciTLDR, 618 echte abstracts met de TLDR's erbij, wordt één keer gedownload van de repository op GitHub, {download_size(scitldr.FILE, 'nl')}, als je in de les op de knop van de download klikt, tenzij *Samenvatten: kiezen of schrijven* de set al binnenhaalde; de demo met echte abstracts heeft de set nodig, de rest van de les werkt zonder.",
}

# Lin's example of skip-bigrams (2004, section 5): S1 is the reference.
POLICE = ("police killed the gunman", "police kill the gunman", "the gunman kill police", "the gunman police killed")
LIN_ROUGE_S = (0.5, 0.167, 0.333)   # "S2's ROUGE-S score is 0.5, S3 is 0.167, and S4 is 0.333."


# --- n-grams and the three functions, as the course writes them ---------------------------------------------------------

def ngram_counts(ws: list, n: int) -> dict:
    """How often each n-gram of a list of words occurs: the words of an n-gram joined with a space, such as "the cat",
    as in *Scoring a translation: BLEU and chrF*."""
    counts = {}
    for i in range(len(ws) - n + 1):
        gram = " ".join(ws[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def rouge_n(mine, theirs):
    """The course's solution of Exercise 1: the recall of the n-grams of the reference, `theirs`, that the summary,
    `mine`, has, each counted at most as often as the summary has it."""
    matched = 0
    total = 0
    for gram in theirs:
        total = total + theirs[gram]
        if gram in mine:
            matched = matched + min(mine[gram], theirs[gram])
    return matched / total


def lcs(a, b):
    """The course's solution of Exercise 2: the length of the longest common subsequence of the lists `a` and `b`."""
    L = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j - 1])
    return L[len(a), len(b)]


def skip_bigrams(words):
    """The course's solution of Exercise 3: every pair of words in their order, with any gap between them."""
    pairs = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            pairs.append((words[i], words[j]))
    return pairs


def _table_lcs(a, b) -> list:
    """The whole table of longest common subsequences of the first parts of `a` and `b`, as lists: fast enough for the
    618 abstracts of SciTLDR."""
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai, row, up = a[i - 1], L[i], L[i - 1]
        for j in range(1, len(b) + 1):
            row[j] = up[j - 1] + 1 if ai == b[j - 1] else (up[j] if up[j] >= row[j - 1] else row[j - 1])
    return L


def lcs_length(a, b) -> int:
    return _table_lcs(a, b)[len(a)][len(b)]


def lcs_places(a, b) -> list:
    """The places in `b` of one longest common subsequence of `a` and `b`, in order: the way back through the table,
    up before left on a tie."""
    L = _table_lcs(a, b)
    i, j, out = len(a), len(b), []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(j - 1)
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def pair_counts(pairs) -> dict:
    """How often each pair of a list of pairs occurs."""
    out = {}
    for p in pairs:
        out[p] = out.get(p, 0) + 1
    return out


def recall_n(mine: list, theirs: list, n: int) -> float:
    """ROUGE-N recall of a summary against one reference, both lists of words; 0.0 for a reference without n-grams."""
    t = ngram_counts(theirs, n)
    return rouge_n(ngram_counts(mine, n), t) if t else 0.0


def precision_n(mine: list, theirs: list, n: int) -> float:
    """The share of the n-grams of the summary that the reference has, clipped the same way."""
    m = ngram_counts(mine, n)
    return rouge_n(ngram_counts(theirs, n), m) if m else 0.0


def f_score(p: float, r: float) -> float:
    """The F-score with β = 1: precision and recall count the same."""
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def recall_l(mine: list, theirs: list) -> float:
    return lcs_length(mine, theirs) / len(theirs) if theirs else 0.0


def recall_s(mine: list, theirs: list) -> float:
    t = pair_counts(skip_bigrams(theirs))
    return rouge_n(pair_counts(skip_bigrams(mine)), t) if t else 0.0


def scores(mine: list, refs: list) -> dict:
    """ROUGE of a summary against references, all lists of words: for each measure the best over the references, as
    Lin takes the maximum. "r1", "p1" and "f1" are ROUGE-1 recall, precision and F-score; "r2" ROUGE-2 recall;
    "rl" ROUGE-L recall."""
    refs = [r for r in refs if r]
    if not refs:
        return {k: 0.0 for k in ("r1", "p1", "f1", "r2", "rl")}
    r1 = [recall_n(mine, r, 1) for r in refs]
    p1 = [precision_n(mine, r, 1) for r in refs]
    return {"r1": max(r1), "p1": max(p1), "f1": max(f_score(p, r) for p, r in zip(p1, r1)),
            "r2": max(recall_n(mine, r, 2) for r in refs), "rl": max(recall_l(mine, r) for r in refs)}


# --- the six abstracts of the course ------------------------------------------------------------------------------------

WORKED = "bees"
TLDR_WORDS = {key: words(a["tldr"]) for key, a in ABSTRACTS.items()}
QWEN_WORDS = {key: words(QWEN[key]) for key in ABSTRACTS}
PICKED_WORDS = {key: words(a["sentences"][summary(key)[0]]) for key, a in ABSTRACTS.items()}    # the title method, k = 1
CLOSEST_WORDS = {key: words(a["sentences"][closest(key)]) for key, a in ABSTRACTS.items()}


def best_sentence(key: str, n: int = 1) -> int:
    """The place of the sentence of a course abstract with the highest ROUGE-N recall against the TLDR; of two equal,
    the earlier."""
    return pick(S.even([recall_n(words(s), TLDR_WORDS[key], n) for s in ABSTRACTS[key]["sentences"]]), 1)[0]


def course_numbers() -> dict:
    """The numbers the text states about the six abstracts, worked out from them."""
    out = {}
    for key in ABSTRACTS:
        for name, ws in (("qwen", QWEN_WORDS[key]), ("picked", PICKED_WORDS[key]), ("closest", CLOSEST_WORDS[key])):
            out[(key, name)] = scores(ws, [TLDR_WORDS[key]])
    out["qwen_beats_picked"] = sum(1 for k in ABSTRACTS if out[(k, "qwen")]["r1"] > out[(k, "picked")]["r1"])
    out["closest_is_best"] = sum(1 for k in ABSTRACTS if best_sentence(k) == closest(k))
    return out


# --- the real abstracts: SciTLDR, against the TLDRs of the experts -------------------------------------------------------

CANDIDATES = ("authors", "title", "first", "longest")


def candidate_texts(p: dict, idf: dict) -> dict:
    """Four summaries of one paper of SciTLDR: the TLDR of its authors, the sentence the title method picks, the first
    sentence and the longest sentence of the abstract (the earlier of two as long)."""
    t = pick(S.method_scores(p["title"], p["sentences"], idf, "title"), 1)[0]
    lengths = [len(words(s)) for s in p["sentences"]]
    return {"authors": p["tldrs"][0], "title": p["sentences"][t], "first": p["sentences"][0],
            "longest": p["sentences"][pick(lengths, 1)[0]]}


def expert_words(p: dict) -> list:
    """The words of the TLDRs of the experts of a paper: every TLDR after the first."""
    return [ws for ws in (words(t) for t in p["tldrs"][1:]) if ws]


def scitldr_rouge(papers: list) -> dict:
    """The mean ROUGE of each kind of summary against the TLDRs of the experts, over the papers that have one; the mean
    length of each in words; how often the title method's sentence has a higher ROUGE-1 recall than the authors' TLDR,
    and the reverse; and the number of TLDRs of the experts."""
    idf = S.document_idf([S._document(p["title"], p["sentences"]) for p in papers])
    sums = {c: {k: 0.0 for k in ("r1", "p1", "f1", "r2", "rl", "words")} for c in CANDIDATES}
    n, title_wins, authors_wins, experts = 0, 0, 0, 0
    for p in papers:
        refs = expert_words(p)
        if not refs:
            continue
        n += 1
        experts += len(refs)
        got = {}
        for c, text in candidate_texts(p, idf).items():
            ws = words(text)
            got[c] = scores(ws, refs)
            for k, v in got[c].items():
                sums[c][k] += v
            sums[c]["words"] += len(ws)
        title_wins += got["title"]["r1"] > got["authors"]["r1"]
        authors_wins += got["authors"]["r1"] > got["title"]["r1"]
    means = {c: {k: v / n for k, v in d.items()} for c, d in sums.items()} if n else {}
    return {"papers": n, "means": means, "title_wins": title_wins, "authors_wins": authors_wins, "experts": experts,
            "expert_words": sum(len(r) for p in papers for r in expert_words(p)) / experts if experts else 0.0,
            "idf": idf}


def _measures(papers: list) -> dict:
    """scitldr_rouge of the whole test set, worked out once in a process."""
    if len(papers) == len(scitldr.papers()):
        return kept(f"{_KEY}:measures", lambda: scitldr_rouge(scitldr.papers()))
    return scitldr_rouge(papers)


# --- small parts of the pictures (those of nlp-summarize) ---------------------------------------------------------------

_UI, _LABEL, _CELL, _HEAD, _NUMBER = S._UI, S._LABEL, S._CELL, S._HEAD, S._NUMBER
_DATA, _FAINT, _OUT, _PRED, _LOSS = S._DATA, S._FAINT, S._OUT, S._PRED, S._LOSS
_CHIP, _CHIP_PRED, _CHIP_DATA, _CHIP_LOSS = S._CHIP, S._CHIP_PRED, S._CHIP_DATA, S._CHIP_LOSS
_note, _colors, _label, _num, _dec, _pct = S._note, S._colors, S._label, S._num, S._dec, S._pct
_cell, _table, _bar, _short, _is_number = S._cell, S._table, S._bar, S._short, S._is_number
_fn, _call, _empty, _SLOW, _STOPPED, PICTURE_SECONDS = S._fn, S._call, S._empty, S._SLOW, S._STOPPED, S.PICTURE_SECONDS
_WHY_EMPTY, _WORD = S._WHY_EMPTY, S._WORD
TEXT_CAP = 2000   # characters of the summary in the try-it box that the course scores


def _chipped(text_: str, places, chip: str = _CHIP_PRED) -> str:
    """A text with its words at `places` (counted over the words as `words` finds them, from 0) on a background."""
    places, out, pos = set(places), [], 0
    for k, m in enumerate(_WORD.finditer(text_)):
        out.append(html.escape(text_[pos:m.start()]))
        w = html.escape(m.group())
        out.append(f'<span style="{chip}">{w}</span>' if k in places else w)
        pos = m.end()
    out.append(html.escape(text_[pos:]))
    return "".join(out)


def matched_places(mine: list, theirs: list) -> list:
    """The places of the words of the reference that a summary has, clipped: a word the summary has twice counts at
    most twice, from the start of the reference."""
    left = ngram_counts(mine, 1)
    out = []
    for k, w in enumerate(theirs):
        if left.get(w, 0) > 0:
            left[w] -= 1
            out.append(k)
    return out


def matched_bigrams(mine: list, theirs: list) -> list:
    """The bigrams of the reference that a summary has, clipped, in the order of the reference."""
    left = ngram_counts(mine, 2)
    out = []
    for i in range(len(theirs) - 1):
        g = theirs[i] + " " + theirs[i + 1]
        if left.get(g, 0) > 0:
            left[g] -= 1
            out.append(g)
    return out


def _r(v, lang: str) -> str:
    """A ROUGE score as the pictures write it: a percentage without decimals; anything else as it is."""
    return S._pct(v, lang, 0) if _is_number(v) and abs(v) < 1e9 else html.escape(_short(v))


def _score_cell(v, want, lang: str, most: float = 1.0) -> str:
    good = want is None or (_is_number(v) and math.isclose(v, want, rel_tol=1e-6, abs_tol=1e-9))
    color = "var(--ea-pred)" if good else "var(--ea-loss)"
    return _cell(_bar(v, most, color, 4) + _r(v, lang), _OUT if good else _LOSS, _NUMBER)


def _claims_html(key: str) -> str:
    """Qwen3's summary of a course abstract with the claims that the abstract does not support in purple."""
    text_ = QWEN[key]
    return S._words_html(text_, set(), _CHIP_PRED, S._spans(text_, UNSUPPORTED.get(key, ())))


# --- the idea: three ways to compare a summary with its reference ---------------------------------------------------------

_IDEA_TEXT = {
    "en": {"title": "Qwen3-0.6B's summary of the abstract about bees, scored against the TLDR",
           "summary": "The summary, written by Qwen3-0.6B:",
           "r1": "ROUGE-1 recall {r}: {k} of the {n} words of the TLDR come back in the summary.",
           "r2": "ROUGE-2 recall {r}: {k} of the {n} bigrams of the TLDR come back:",
           "rl": "ROUGE-L recall {r}: the longest common subsequence has {k} of the {n} words, in the order of the TLDR.",
           "gray": "Gray: the TLDR that the course wrote, the reference.",
           "blue": "Blue: the summary, and on a background the words and bigrams of the TLDR that come back in it.",
           "purple": "Purple: a claim of the summary that the abstract does not support, in the course's reading; it costs none of the three scores anything."},
    "nl": {"title": "De samenvatting van Qwen3-0.6B van het abstract over bijen, gescoord tegen de TLDR",
           "summary": "De samenvatting, geschreven door Qwen3-0.6B:",
           "r1": "ROUGE-1-recall {r}: {k} van de {n} woorden van de TLDR komen terug in de samenvatting.",
           "r2": "ROUGE-2-recall {r}: {k} van de {n} bigrams van de TLDR komen terug:",
           "rl": "ROUGE-L-recall {r}: de langste gemeenschappelijke deelrij heeft {k} van de {n} woorden, in de volgorde van de TLDR.",
           "gray": "Grijs: de TLDR die de course schreef, de referentie.",
           "blue": "Blauw: de samenvatting, en met een achtergrond de woorden en bigrams van de TLDR die erin terugkomen.",
           "purple": "Paars: een bewering van de samenvatting die het abstract niet ondersteunt, zoals de course het leest; de bewering kost geen van de drie scores iets."},
}


def idea_picture(lang: str) -> str:
    lang = lang if lang in LANGS else "en"
    w = _IDEA_TEXT[lang]
    tldr, mine = ABSTRACTS[WORKED]["tldr"], QWEN_WORDS[WORKED]
    ref = TLDR_WORDS[WORKED]
    body = _label(w["summary"]) + f'<div style="{_OUT}">{_claims_html(WORKED)}</div>'
    one = matched_places(mine, ref)
    body += _label(w["r1"].format(r=_r(recall_n(mine, ref, 1), lang), k=_num(len(one), lang), n=_num(len(ref), lang)))
    body += f'<div style="{_DATA}">{_chipped(tldr, one)}</div>'
    two = matched_bigrams(mine, ref)
    body += _label(w["r2"].format(r=_r(recall_n(mine, ref, 2), lang), k=_num(len(two), lang), n=_num(len(ref) - 1, lang)))
    body += f'<div style="{_DATA}">' + " ".join(f'<span style="{_CHIP_PRED}">{html.escape(g)}</span>' for g in two) + "</div>"
    places = lcs_places(mine, ref)
    body += _label(w["rl"].format(r=_r(recall_l(mine, ref), lang), k=_num(len(places), lang), n=_num(len(ref), lang)))
    body += f'<div style="{_DATA}">{_chipped(tldr, places)}</div>'
    body += _colors(w, body)
    return yours.figure(w["title"], body, alt=w["title"])


# --- the pictures under the exercises ------------------------------------------------------------------------------------

_ROUGE_TEXT = {
    "en": {"title": "Your rouge_n on the six abstracts: a written summary and a picked sentence",
           "head": ["Abstract", "ROUGE-1, written", "ROUGE-1, picked", "ROUGE-2, written", "ROUGE-2, picked"],
           "summary": "With your rouge_n, Qwen3-0.6B's summary has a higher ROUGE-1 recall than the sentence the title method picks in {k} of the {a} abstracts.",
           "leaks": "The summary of the abstract about leaks, with ROUGE-1 recall {r} by your rouge_n, against {c} for the sentence closest to the TLDR:",
           "gray": "Gray: the titles of the six abstracts.",
           "blue": "Blue: the recall your rouge_n gives Qwen3-0.6B's summary (written) and the title method's sentence (picked), against the TLDR, when it is what the course's rouge_n gives.",
           "purple": "Purple: a claim that the abstract does not support, in the course's reading, and a recall that differs from what the course's rouge_n gives.",
           "kind": "a number"},
    "nl": {"title": "Je rouge_n op de zes abstracts: een geschreven samenvatting en een gekozen zin",
           "head": ["Abstract", "ROUGE-1, geschreven", "ROUGE-1, gekozen", "ROUGE-2, geschreven", "ROUGE-2, gekozen"],
           "summary": "Met je rouge_n heeft de samenvatting van Qwen3-0.6B een hogere ROUGE-1-recall dan de zin die de titelmethode kiest bij {k} van de {a} abstracts.",
           "leaks": "De samenvatting van het abstract over lekken, met ROUGE-1-recall {r} volgens je rouge_n, tegen {c} voor de zin die het dichtst bij de TLDR ligt:",
           "gray": "Grijs: de titels van de zes abstracts.",
           "blue": "Blauw: de recall die je rouge_n geeft aan de samenvatting van Qwen3-0.6B (geschreven) en aan de zin van de titelmethode (gekozen), tegen de TLDR, als dat is wat de rouge_n van de course geeft.",
           "purple": "Paars: een bewering die het abstract niet ondersteunt, zoals de course het leest, en een recall die anders is dan bij de rouge_n van de course.",
           "kind": "een getal"},
}


def rouge_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _ROUGE_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    kinds = {"qwen": QWEN_WORDS, "picked": PICKED_WORDS, "closest": CLOSEST_WORDS}

    def make():
        return {(k, name, n): fn(ngram_counts(d[k], n), ngram_counts(TLDR_WORDS[k], n))
                for k in ABSTRACTS for name, d in kinds.items() for n in (1, 2)}

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_number(v) and abs(v) < 1e9 for v in mine.values()):
        return _empty(w, "odd")
    want = {(k, name, n): recall_n(d[k], TLDR_WORDS[k], n) for k in ABSTRACTS for name, d in kinds.items() for n in (1, 2)}
    rows = []
    for k, a in ABSTRACTS.items():
        cells = [_cell(f"<i>{html.escape(a['title'])}</i>", _DATA)]
        for n in (1, 2):
            for name in ("qwen", "picked"):
                cells.append(_score_cell(mine[(k, name, n)], want[(k, name, n)], lang))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = _table(rows, w["head"], (1, 2, 3, 4))
    body += _label(w["leaks"].format(r=_r(mine[("leaks", "qwen", 1)], lang), c=_r(mine[("leaks", "closest", 1)], lang)))
    body += f'<div style="{_OUT}">{_claims_html("leaks")}</div>'
    body += f'<div style="{_UI};{_LOSS};font-weight:400">{html.escape(WHY_UNSUPPORTED[lang]["leaks"])}</div>'
    k_ = sum(1 for k in ABSTRACTS if _is_number(mine[(k, "qwen", 1)]) and mine[(k, "qwen", 1)] > mine[(k, "picked", 1)])
    summary_ = w["summary"].format(k=_num(k_, lang), a=_num(len(ABSTRACTS), lang))
    return yours.figure(w["title"], body + _note(summary_) + _colors(w, body), alt=w["title"])


_LCS_TEXT = {
    "en": {"title": "Your lcs, cell by cell, and on the six abstracts",
           "table": "Each cell is your lcs of the first words of S4 (the rows) and the first words of S1 (the columns), Lin's reference:",
           "rl": "ROUGE-L recall of Qwen3-0.6B's six summaries against the TLDR: your lcs divided by the length of the TLDR, with ROUGE-1 recall beside it.",
           "head": ["Abstract", "Your lcs", "ROUGE-L", "ROUGE-1"],
           "gray": "Gray: the words of the two sentences, and ROUGE-1 recall; a cell on a background is a place where the two words are the same.",
           "blue": "Blue: what your lcs gives, and ROUGE-L recall from it, when it is what the course's lcs gives.",
           "purple": "Purple: a length that differs from what the course's lcs gives.",
           "kind": "a whole number"},
    "nl": {"title": "Je lcs, cel voor cel, en op de zes abstracts",
           "table": "Elke cel is je lcs van de eerste woorden van S4 (de rijen) en de eerste woorden van S1 (de kolommen), de referentie van Lin:",
           "rl": "ROUGE-L-recall van de zes samenvattingen van Qwen3-0.6B tegen de TLDR: je lcs gedeeld door de lengte van de TLDR, met ROUGE-1-recall ernaast.",
           "head": ["Abstract", "Je lcs", "ROUGE-L", "ROUGE-1"],
           "gray": "Grijs: de woorden van de twee zinnen, en ROUGE-1-recall; een cel met een achtergrond is een plaats waar de twee woorden gelijk zijn.",
           "blue": "Blauw: wat je lcs geeft, en ROUGE-L-recall daaruit, als dat is wat de lcs van de course geeft.",
           "purple": "Paars: een lengte die anders is dan bij de lcs van de course.",
           "kind": "een geheel getal"},
}


def _whole(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (int, np.integer)):
        return abs(int(v)) < 10**9
    return isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer() and abs(v) < 1e9


def _as_int(v) -> int:
    return int(v)


def lcs_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _LCS_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    a, b = POLICE[3].split(), POLICE[0].split()

    def make():
        grid = [[fn(a[:i], b[:j]) for j in range(len(b) + 1)] for i in range(len(a) + 1)]
        return grid, {k: fn(list(QWEN_WORDS[k]), list(TLDR_WORDS[k])) for k in ABSTRACTS}

    mine = _call(make)
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    grid, six = mine
    if not all(_whole(v) for row in grid for v in row) or not all(_whole(v) for v in six.values()):
        return _empty(w, "odd")
    want = _table_lcs(a, b)
    head = f'<th style="{_HEAD}"></th><th style="{_HEAD}{_NUMBER}"></th>' + "".join(
        f'<th style="{_HEAD}{_NUMBER}">{html.escape(x)}</th>' for x in b)
    rows = []
    for i in range(len(a) + 1):
        cells = [_cell(html.escape(a[i - 1]) if i else "", _DATA)]
        for j in range(len(b) + 1):
            v, good = _as_int(grid[i][j]), _as_int(grid[i][j]) == want[i][j]
            same = i and j and a[i - 1] == b[j - 1]
            style = (_PRED if good else _LOSS) + (";background:var(--ea-data-wash)" if same else "")
            cells.append(_cell(_num(v, lang), style, _NUMBER))
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = _label(w["table"]) + (f'<div style="overflow-x:auto;max-width:100%"><table style="border-collapse:collapse;'
                                 f'margin:0.3rem 0 0.2rem"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
                                 f'</table></div>')
    rows = []
    for k, ab in ABSTRACTS.items():
        v, good = _as_int(six[k]), _as_int(six[k]) == lcs_length(QWEN_WORDS[k], TLDR_WORDS[k])
        rl = v / len(TLDR_WORDS[k])
        rows.append("<tr>" + _cell(f"<i>{html.escape(ab['title'])}</i>", _DATA)
                    + _cell(_num(v, lang), _PRED if good else _LOSS, _NUMBER)
                    + _cell(_bar(rl, 1.0, "var(--ea-pred)" if good else "var(--ea-loss)", 4) + _r(rl, lang),
                            _OUT if good else _LOSS, _NUMBER)
                    + _cell(_r(recall_n(QWEN_WORDS[k], TLDR_WORDS[k], 1), lang), _DATA, _NUMBER) + "</tr>")
    body += _label(w["rl"]) + _table(rows, w["head"], (1, 2, 3))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


_SKIP_TEXT = {
    "en": {"title": "Your skip_bigrams on Lin's four sentences",
           "reference": "S1, the reference: {s}. Your skip_bigrams gives {n} pairs; the course's gives {m}.",
           "row": "{name}: {s}",
           "head": ["Sentence", "Its skip-bigrams", "ROUGE-S recall", "Lin", "ROUGE-1"],
           "gray": "Gray: the sentences, the scores Lin gives, ROUGE-1 recall, and a pair that S1 does not have.",
           "blue": "Blue: the pairs your skip_bigrams gives, on a background when S1 has them too, and ROUGE-S recall from them, when they are what the course's skip_bigrams gives.",
           "purple": "Purple: pairs or a score that differ from what the course's skip_bigrams gives.",
           "kind": "a list of pairs"},
    "nl": {"title": "Je skip_bigrams op de vier zinnen van Lin",
           "reference": "S1, de referentie: {s}. Je skip_bigrams geeft {n} paren; de skip_bigrams van de course geeft er {m}.",
           "row": "{name}: {s}",
           "head": ["Zin", "De skip-bigrams", "ROUGE-S-recall", "Lin", "ROUGE-1"],
           "gray": "Grijs: de zinnen, de scores die Lin geeft, ROUGE-1-recall, en een paar dat S1 niet heeft.",
           "blue": "Blauw: de paren die je skip_bigrams geeft, met een achtergrond als S1 deze paren ook heeft, en ROUGE-S-recall daaruit, als dat is wat de skip_bigrams van de course geeft.",
           "purple": "Paars: paren of een score die anders zijn dan bij de skip_bigrams van de course.",
           "kind": "een lijst met paren"},
}


def _as_pairs(v):
    """What skip_bigrams gives, with each pair written as a list made a tuple: [["a", "b"]] counts as [("a", "b")]."""
    if isinstance(v, (list, tuple)):
        return [tuple(p) if isinstance(p, list) else p for p in v]
    return v


def _same_pairs(got, want) -> bool:
    """The same pairs, each as often, in any order."""
    if not isinstance(got, list) or not isinstance(want, list):
        return got == want
    try:
        return sorted(map(repr, got)) == sorted(map(repr, want))
    except Exception:   # noqa: BLE001   a value that cannot even be written
        return False


def _is_pairs(v) -> bool:
    """A list of pairs that can be counted: tuples of two that can be keys of a dictionary."""
    if not (isinstance(v, list) and all(isinstance(p, tuple) and len(p) == 2 for p in v)):
        return False
    try:
        return all(hash(p) is not None for p in v)
    except TypeError:
        return False


def skip_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _SKIP_TEXT[lang] | _WHY_EMPTY[lang]
    fn = _fn(seen)
    if fn is None:
        return _empty(w, "stopped")
    sentences = [s.split() for s in POLICE]
    mine = _call(lambda: [_as_pairs(fn(list(ws))) for ws in sentences])
    if mine is _SLOW:
        return _empty(w, "slow")
    if mine is _STOPPED:
        return _empty(w, "stopped")
    if not all(_is_pairs(v) for v in mine):
        return _empty(w, "odd")
    want = [skip_bigrams(ws) for ws in sentences]
    good_ref = _same_pairs(mine[0], want[0])
    ref = pair_counts(mine[0])
    body = _label(w["reference"].format(s=f"<i>{POLICE[0]}</i>", n=_num(len(mine[0]), lang), m=_num(len(want[0]), lang))
                  ).replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    if not good_ref:
        body = body.replace(f'<div style="{_LABEL}">', f'<div style="{_LABEL};{_LOSS}">', 1)
    rows = []
    for k in (1, 2, 3):
        pairs = mine[k]
        good = good_ref and _same_pairs(pairs, want[k])
        left = dict(ref)
        chips = []
        for p in pairs:
            hit = left.get(p, 0) > 0
            if hit:
                left[p] -= 1
            text_ = html.escape(" ".join(str(x) for x in p))
            chips.append(f'<span style="{(_CHIP_PRED if good else _CHIP_LOSS) if hit else _CHIP + ";" + _FAINT}">{text_}</span>')
        score = rouge_n(pair_counts(pairs), ref) if ref else 0.0
        rows.append("<tr>" + _cell(f"S{k + 1}: <i>{html.escape(POLICE[k])}</i>", _DATA) + _cell(" ".join(chips), _OUT)
                    + _cell(_dec(score, lang, 3), _PRED if good else _LOSS, _NUMBER)
                    + _cell(_dec(LIN_ROUGE_S[k - 1], lang, 3), _DATA, _NUMBER)
                    + _cell(_dec(recall_n(sentences[k], sentences[0], 1), lang, 2), _DATA, _NUMBER) + "</tr>")
    body += _table(rows, w["head"], (2, 3, 4))
    return yours.figure(w["title"], body + _colors(w, body), alt=w["title"])


# --- the demo on SciTLDR -------------------------------------------------------------------------------------------------

CANDIDATE_NAMES = {
    "en": {"authors": "The authors' TLDR", "title": "The title method's sentence", "first": "The first sentence",
           "longest": "The longest sentence"},
    "nl": {"authors": "De TLDR van de auteurs", "title": "De zin van de titelmethode", "first": "De eerste zin",
           "longest": "De langste zin"},
}
_SCI_TEXT = {
    "en": {"title": "Four summaries of each paper of SciTLDR, scored against the TLDRs of the experts",
           "means": "The mean over the {n} papers:",
           "head": ["Summary", "Words", "ROUGE-1 recall", "ROUGE-1 F-score", "ROUGE-2 recall", "ROUGE-L recall"],
           "paper": "Paper {i} of {n}: *{title}*",
           "experts": "The TLDRs of the experts, the references:",
           "one": "The four summaries of this paper; a word with a background is in a TLDR of the experts:",
           "phead": ["Summary", "Text", "ROUGE-1 recall", "ROUGE-1 F-score"],
           "gray": "Gray: the TLDRs of the experts, the references, and the lengths in words.",
           "blue": "Blue: the four summaries and their ROUGE, the best over the TLDRs of the experts."},
    "nl": {"title": "Vier samenvattingen van elke paper van SciTLDR, gescoord tegen de TLDR's van de experts",
           "means": "Het gemiddelde over de {n} papers:",
           "head": ["Samenvatting", "Woorden", "ROUGE-1-recall", "ROUGE-1-F-score", "ROUGE-2-recall", "ROUGE-L-recall"],
           "paper": "Paper {i} van {n}: *{title}*",
           "experts": "De TLDR's van de experts, de referenties:",
           "one": "De vier samenvattingen van deze paper; een woord met een achtergrond staat in een TLDR van de experts:",
           "phead": ["Samenvatting", "Tekst", "ROUGE-1-recall", "ROUGE-1-F-score"],
           "gray": "Grijs: de TLDR's van de experts, de referenties, en de lengtes in woorden.",
           "blue": "Blauw: de vier samenvattingen met de ROUGE-scores, de beste over de TLDR's van de experts."},
}
SCI_START = 96   # the paper the slider starts at, counted from 1


def scitldr_picture(lang: str, papers: list, number=SCI_START) -> str:
    lang = lang if lang in LANGS else "en"
    w = _SCI_TEXT[lang]
    m = _measures(papers)
    names = CANDIDATE_NAMES[lang]
    rows = []
    for c in CANDIDATES:
        d = m["means"][c]
        rows.append("<tr>" + _cell(html.escape(names[c]), _OUT) + _cell(_dec(d["words"], lang, 1), _DATA, _NUMBER)
                    + _cell(_bar(d["r1"], 0.5, "var(--ea-pred)", 4) + _pct(d["r1"], lang), _OUT, _NUMBER)
                    + _cell(_bar(d["f1"], 0.5, "var(--ea-pred)", 4) + _pct(d["f1"], lang), _OUT, _NUMBER)
                    + _cell(_pct(d["r2"], lang), _OUT, _NUMBER) + _cell(_pct(d["rl"], lang), _OUT, _NUMBER) + "</tr>")
    body = _label(w["means"].format(n=_num(m["papers"], lang))) + _table(rows, w["head"], (1, 2, 3, 4, 5))
    n = len(papers)
    try:
        i = min(max(int(number), 1), n)
    except (TypeError, ValueError, OverflowError):
        i = min(SCI_START, n)
    p = papers[i - 1]
    refs = expert_words(p)
    body += _label(w["paper"].format(i=_num(i, lang), n=_num(n, lang), title=p["title"]))
    body += _label(w["experts"]) + "".join(f'<div style="{_DATA}">{html.escape(t)}</div>' for t in p["tldrs"][1:])
    known = set(x for r in refs for x in r)
    rows = []
    for c, text_ in candidate_texts(p, m["idf"]).items():
        ws = words(text_)
        got = scores(ws, refs)
        places = [k for k, x in enumerate(ws) if x in known]
        rows.append("<tr>" + _cell(html.escape(names[c]), _OUT) + _cell(_chipped(text_, places), _OUT)
                    + _cell(_pct(got["r1"], lang, 0), _OUT, _NUMBER) + _cell(_pct(got["f1"], lang, 0), _OUT, _NUMBER)
                    + "</tr>")
    body += _label(w["one"]) + _table(rows, w["phead"], (2, 3))
    title = w["title"]
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- try it: a summary of the learner's own ---------------------------------------------------------------------------------

_TRY_TEXT = {
    "en": {"title": "Your summary of the abstract about {name}, scored against the TLDR",
           "tldr": "The TLDR that the course wrote; a word with a background comes back in your summary:",
           "head": ["Summary", "Words", "ROUGE-1 recall", "ROUGE-1 precision", "ROUGE-2 recall", "ROUGE-L recall"],
           "yours": "Your summary", "qwen": "Qwen3-0.6B's summary", "closest": "The closest sentence",
           "empty": "Write a summary in the box, and the course scores it here.",
           "cut": "The course scores the first {n} characters of your summary.",
           "claim": "Qwen3-0.6B's summary says something that the abstract does not support, in the course's reading: {why}",
           "gray": "Gray: the TLDR, the reference, and the lengths in words.",
           "blue": "Blue: the scores of the three summaries, and on a background the words of the TLDR that come back in yours.",
           "purple": "Purple: a claim of Qwen3-0.6B's summary that the abstract does not support, in the course's reading."},
    "nl": {"title": "Je samenvatting van het abstract over {name}, gescoord tegen de TLDR",
           "tldr": "De TLDR die de course schreef; een woord met een achtergrond komt terug in je samenvatting:",
           "head": ["Samenvatting", "Woorden", "ROUGE-1-recall", "ROUGE-1-precision", "ROUGE-2-recall", "ROUGE-L-recall"],
           "yours": "Je samenvatting", "qwen": "Samenvatting van Qwen3-0.6B", "closest": "De dichtstbijzijnde zin",
           "empty": "Schrijf een samenvatting in het vak, dan scoort de course de samenvatting hier.",
           "cut": "De course scoort de eerste {n} tekens van je samenvatting.",
           "claim": "De samenvatting van Qwen3-0.6B zegt iets wat het abstract niet ondersteunt, zoals de course het leest: {why}",
           "gray": "Grijs: de TLDR, de referentie, en de lengtes in woorden.",
           "blue": "Blauw: de scores van de drie samenvattingen, en met een achtergrond de woorden van de TLDR die terugkomen in de jouwe.",
           "purple": "Paars: een bewering van de samenvatting van Qwen3-0.6B die het abstract niet ondersteunt, zoals de course het leest."},
}


def try_picture(lang: str, key: str, text_="") -> str:
    lang = lang if lang in LANGS else "en"
    key = key if key in ABSTRACTS else WORKED
    w = _TRY_TEXT[lang]
    text_ = text_ if isinstance(text_, str) else ""
    cut = len(text_) > TEXT_CAP
    mine = words(text_[:TEXT_CAP])
    ref = TLDR_WORDS[key]
    body = _label(w["tldr"]) + f'<div style="{_DATA}">{_chipped(ABSTRACTS[key]["tldr"], matched_places(mine, ref))}</div>'
    rows = []
    for name, ws in (("yours", mine), ("qwen", QWEN_WORDS[key]), ("closest", CLOSEST_WORDS[key])):
        got = scores(ws, [ref])
        rows.append("<tr>" + _cell(html.escape(w[name]), _OUT) + _cell(_num(len(ws), lang), _DATA, _NUMBER)
                    + "".join(_cell(_bar(got[k], 1.0, "var(--ea-pred)", 3) + _r(got[k], lang), _OUT, _NUMBER)
                              for k in ("r1", "p1", "r2", "rl")) + "</tr>")
    body += _table(rows, w["head"], (1, 2, 3, 4, 5))
    if not mine:
        body += _note(w["empty"])
    if cut:
        body += _note(w["cut"].format(n=_num(TEXT_CAP, lang)))
    if key in UNSUPPORTED:
        body += (f'<div style="{_UI}">{_claims_html(key)}</div>'
                 f'<div style="{_UI};{_LOSS};font-weight:400">{html.escape(w["claim"].format(why=WHY_UNSUPPORTED[lang][key]))}</div>')
    title = w["title"].format(name=ABSTRACT_NAMES[lang][key].lower())
    return yours.figure(title, body + _colors(w, body), alt=title)


# --- the exercises ---------------------------------------------------------------------------------------------------------

def _value(v) -> str:
    """A value the way code writes it, with double quotes; a NumPy whole number as a plain one."""
    if isinstance(v, np.integer):
        return repr(int(v))
    if isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, np.ndarray):
        return _value(v.tolist())
    if isinstance(v, tuple):
        return "(" + ", ".join(_value(x) for x in v) + ("," if len(v) == 1 else "") + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_value(k)}: {_value(x)}" for k, x in v.items()) + "}"
    return S._value(v)


def _show(v):
    out = _value(v)
    return out if len(out) <= 300 else out[:300] + "…"


def _written(v):
    return S._Val(_value(v), v)


_Val = S._Val
_DOTS = S._DOTS
_SAME_TEXTS = S._SAME_TEXTS

# The small cases the exercise texts show; the same in both lessons.
TOY_ROUGE = (({"the": 3, "cat": 1}, {"the": 2, "cat": 1, "sat": 1, "down": 1}),
             ({"een": 1, "klein": 1, "netwerk": 1}, {"een": 1, "netwerk": 1, "telt": 1, "bijen": 1}))
TOY_LCS = ((["police", "kill", "the", "gunman"], ["police", "killed", "the", "gunman"]),
           (["de", "bijen", "tellen"], ["het", "netwerk", "telt", "de", "bijen"]))
TOY_SKIP = (["police", "killed", "the", "gunman"], ["de", "politie", "schoot"])

_RN_HEAD = "def rouge_n(mine, theirs):\n    matched = 0\n    total = 0\n    for gram in theirs:\n"
_RN_TOTAL = "        total = total + theirs[gram]\n"
_RN_IF = "        if gram in mine:\n"
_RN_ADD = "            matched = matched + min(mine[gram], theirs[gram])\n"
_RN_RET = "    return matched / total\n"


def _rn(total: str = _RN_TOTAL, add: str = _RN_ADD, ret: str = _RN_RET) -> str:
    return _RN_HEAD + total + _RN_IF + add + ret


_RN = _rn()
_RN_START = _rn(add="            matched = matched + ...\n", ret="    return ...\n")
_RN_UNCLIPPED = _rn(add="            matched = matched + mine[gram]\n")
_RN_THEIRS = _rn(add="            matched = matched + theirs[gram]\n")
_RN_ONE = _rn(add="            matched = matched + 1\n")
_RN_DISTINCT = _rn(total="        total = total + 1\n")
_RN_COUNT = _rn(ret="    return matched\n")
_RN_PERCENT = _rn(ret="    return matched / total * 100\n")
_RN_FLOOR = _rn(ret="    return matched // total\n")
_RN_INSIDE = _RN_HEAD + _RN_TOTAL + _RN_IF + _RN_ADD + "        return matched / total\n"
_RN_PRECISION = ("def rouge_n(mine, theirs):\n    matched = 0\n    total = 0\n    for gram in mine:\n"
                 "        total = total + mine[gram]\n        if gram in theirs:\n"
                 "            matched = matched + min(mine[gram], theirs[gram])\n    return matched / total\n")

_LC_HEAD = ("def lcs(a, b):\n    L = np.zeros((len(a) + 1, len(b) + 1), dtype=int)\n"
            "    for i in range(1, len(a) + 1):\n        for j in range(1, len(b) + 1):\n"
            "            if a[i - 1] == b[j - 1]:\n")
_LC_MATCH = "                L[i, j] = L[i - 1, j - 1] + 1\n"
_LC_ELSE = "                L[i, j] = max(L[i - 1, j], L[i, j - 1])\n"
_LC_RET = "    return L[len(a), len(b)]\n"


def _lc(match: str = _LC_MATCH, other: str = _LC_ELSE, ret: str = _LC_RET) -> str:
    return _LC_HEAD + match + "            else:\n" + other + ret


_LC = _lc()
_LC_START = _lc("                L[i, j] = ...\n", "                L[i, j] = ...\n")
_LC_MIN = _lc(other="                L[i, j] = min(L[i - 1, j], L[i, j - 1])\n")
_LC_NO_ONE = _lc("                L[i, j] = L[i - 1, j - 1]\n")
_LC_UP_ONE = _lc("                L[i, j] = L[i - 1, j] + 1\n")
_LC_ZERO = _lc(other="                L[i, j] = 0\n")
_LC_DIAGONAL = _lc(other="                L[i, j] = L[i - 1, j - 1]\n")
_LC_TABLE = _lc(ret="    return L\n")
_LC_FROM_0 = _LC.replace("range(1, len(a) + 1)", "range(len(a) + 1)").replace("range(1, len(b) + 1)", "range(len(b) + 1)")
_LC_INDEX = _LC.replace("if a[i - 1] == b[j - 1]:", "if a[i] == b[j]:")

_SB_HEAD = "def skip_bigrams(words):\n    pairs = []\n    for i in range(len(words)):\n"
_SB_FOR = "        for j in range(i + 1, len(words)):\n"
_SB_ADD = "            pairs.append((words[i], words[j]))\n"
_SB_RET = "    return pairs\n"


def _sb(loop: str = _SB_FOR, add: str = _SB_ADD) -> str:
    return _SB_HEAD + loop + add + _SB_RET


_SB = _sb()
_SB_START = _sb("        for j in range(...):\n", "            pairs.append(...)\n")
_SB_ALL = _sb("        for j in range(len(words)):\n")
_SB_SELF = _sb("        for j in range(i, len(words)):\n")
_SB_NEXT = _sb("        for j in range(i + 1, i + 2):\n")
_SB_SHORT = _sb("        for j in range(i + 1, len(words) - 1):\n")
_SB_BACK = _sb(add="            pairs.append((words[j], words[i]))\n")


def _not_number(got, text_) -> bool:
    return not _is_number(got if not isinstance(got, (np.integer, np.floating)) else float(got))


def _a_table(got, text_) -> bool:
    return isinstance(got, np.ndarray) and got.ndim == 2


def _not_whole(got, text_) -> bool:
    return not _whole(got)


def _not_list(got, text_) -> bool:
    return not isinstance(got, list)


def _append_two(e) -> bool:
    return isinstance(e, TypeError) and "append" in str(e)


def _index_error(e) -> bool:
    return isinstance(e, IndexError) and "list index" in str(e)


REQUIRED = ["rouge_n", "lcs", "skip_bigrams", "changes"]


def _named(table: str, key: str, n: int | None = None) -> _Val:
    if n is None:
        return _Val(f'C.{table}["{key}"]', globals()[table][key])
    return _Val(f'C.ngram_counts(C.{table}["{key}"], {n})', ngram_counts(globals()[table][key], n))


EXERCISES = {
    "rouge_n": FunctionBox(
        name="rouge_n",
        picture=rouge_picture,
        function="rouge_n",
        start=_RN_START,
        solution=_RN,
        show=_show,
        visible=[(_written(a), _written(b)) for a, b in TOY_ROUGE],
        hidden=[(_written({}), _written({"a": 1, "hive": 1})),
                (_written({"bees": 1}), _written({"bees": 1})),
                (_written({"the": 1, "bees": 2}), _written({"bees": 3, "hive": 1})),
                (_written({"small network": 1, "network counts": 1}), _written({"small network": 1, "network that": 1, "that counts": 1})),
                (_written({"rain": 1}), _written({"wind": 2}))]
               + [(_named("QWEN_WORDS", key, n), _named("TLDR_WORDS", key, n)) for key, n in (("bees", 1), ("river", 2), ("leaks", 1))],
        first=[_DOTS],
        mistakes=[
            Mistake("not-number", matches=_not_number,
                    en="`{call}` gives {got}, and it should give {want}: one number, a share between 0 and 1. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één getal, een aandeel tussen 0 en 1. Wat geeft je functie terug?"),
            Mistake("precision", wrong=_RN_PRECISION,
                    en="`{call}` gives {got}, and it should give {want}: you divide by the n-grams of the summary, `mine`, as BLEU does. That is precision. ROUGE divides by the n-grams of the reference: which dictionary holds those?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door de n-grams van de samenvatting, `mine`, zoals BLEU doet. Dat is precision. ROUGE deelt door de n-grams van de referentie: welke dictionary bevat deze n-grams?"),
            Mistake("unclipped", wrong=_RN_UNCLIPPED,
                    en="`{call}` gives {got}, and it should give {want}: an n-gram counts as often as the summary has it, even more often than the reference has it. *the* comes 3 times in the summary of the first case, and 2 times in the reference: how often may it count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een n-gram telt zo vaak als de samenvatting het heeft, ook vaker dan de referentie het heeft. *the* staat 3 keer in de samenvatting van het eerste geval, en 2 keer in de referentie: hoe vaak mag het meetellen?"),
            Mistake("theirs-count", wrong=_RN_THEIRS,
                    en="`{call}` gives {got}, and it should give {want}: an n-gram counts as often as the reference has it, even when the summary has it fewer times. *bees* is twice in the summary and 3 times in the reference: how often does it come back in the summary?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een n-gram telt zo vaak als de referentie het heeft, ook als de samenvatting het minder vaak heeft. *bees* staat twee keer in de samenvatting en 3 keer in de referentie: hoe vaak vind je het in de samenvatting terug?"),
            Mistake("one-each", wrong=_RN_ONE,
                    en="`{call}` gives {got}, and it should give {want}: each shared n-gram counts once, while the reference may have it more often. *the* is twice in the reference of the first case, and the summary has it 3 times: how often does it count?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk gedeeld n-gram telt één keer, terwijl de referentie het vaker kan hebben. *the* staat twee keer in de referentie van het eerste geval, en de samenvatting heeft het 3 keer: hoe vaak telt het mee?"),
            Mistake("distinct", wrong=_RN_DISTINCT,
                    en="`{call}` gives {got}, and it should give {want}: you divide by the number of different n-grams of the reference. *the* is twice in the reference of the first case, but counts once in what you divide by: what does `theirs` say about how often?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je deelt door het aantal verschillende n-grams van de referentie. *the* staat twee keer in de referentie van het eerste geval, maar telt één keer mee in het getal waardoor je deelt: wat zegt `theirs` over hoe vaak?"),
            Mistake("count", wrong=_RN_COUNT,
                    en="`{call}` gives {got}, and it should give {want}: the number of matched n-grams, not their share. What do you divide `matched` by?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: het aantal gevonden n-grams, niet het aandeel. Waardoor deel je `matched`?"),
            Mistake("percent", wrong=_RN_PERCENT,
                    en="`{call}` gives {got}, and it should give {want}: a percentage. The function gives a share between 0 and 1: what is left without the `* 100`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een percentage. De functie geeft een aandeel tussen 0 en 1: wat blijft er over zonder de `* 100`?"),
            Mistake("floor", wrong=_RN_FLOOR,
                    en="`{call}` gives {got}, and it should give {want}: `//` divides and throws away what is after the decimal point, so a share below 1 becomes 0. Which sign divides and keeps it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: `//` deelt en gooit weg wat na de komma komt, dus een aandeel onder 1 wordt 0. Welk teken deelt zonder iets weg te gooien?"),
            Mistake("return-inside", wrong=_RN_INSIDE,
                    en="`{call}` gives {got}, and it should give {want}: your function stops after the first n-gram of the reference. Where does your `return` stand: inside the `for` loop, or after it?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je functie stopt na het eerste n-gram van de referentie. Waar staat je `return`: in de `for`-loop, of erna?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the first adds how often the n-gram counts as matched, clipped as in BLEU, and the `return` gives the matched n-grams as a share of the n-grams of the reference.",
                      "nl": "Vervang de twee `...`: de eerste telt op hoe vaak het n-gram als gevonden meetelt, afgetopt zoals bij BLEU, en de `return` geeft de gevonden n-grams als aandeel van de n-grams van de referentie."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does each n-gram of the reference count as often as both dictionaries have it, and do you divide by all the n-grams of the reference?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Telt elk n-gram van de referentie zo vaak mee als allebei de dictionaries het hebben, en deel je door alle n-grams van de referentie?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give `0.0` for a summary without n-grams, and does a word the summary has fewer times than the reference count as often as the summary has it?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie `0.0` bij een samenvatting zonder n-grams, en telt een woord dat de samenvatting minder vaak heeft dan de referentie zo vaak mee als de samenvatting het heeft?"},
            "correct": {"en": "Right: your function measures how much of the reference comes back in a summary. In the picture below, which scores higher on the abstract about leaks: the written summary or the closest sentence?",
                        "nl": "Goed: je functie meet hoeveel van de referentie terugkomt in een samenvatting. Kijk in het plaatje hieronder: wat scoort hoger op het abstract over lekken, de geschreven samenvatting of de dichtstbijzijnde zin?"},
        },
    ),
    "lcs": FunctionBox(
        name="lcs",
        picture=lcs_picture,
        function="lcs",
        start=_LC_START,
        solution=_LC,
        show=_show,
        names={"np": np},
        visible=[(_written(a), _written(b)) for a, b in TOY_LCS],
        hidden=[(_written([]), _written(["a", "hive"])),
                (_written(["bees"]), _written([])),
                (_written(["the", "gunman", "police", "killed"]), _written(["police", "killed", "the", "gunman"])),
                (_written(["the", "gunman", "kill", "police"]), _written(["police", "killed", "the", "gunman"])),
                (_written(["a", "a", "b"]), _written(["a", "b", "a"])),
                (_written(["x", "y"]), _written(["x", "x", "x"])),
                (_written(["rain"]), _written(["wind"]))]
               + [(_named("QWEN_WORDS", key), _named("TLDR_WORDS", key)) for key in ("bees", "leaks")],
        first=[_DOTS],
        mistakes=[
            Mistake("index", error=_index_error,
                    en="With `{call}`, your function stops with an `IndexError`: `i` goes up to the length of `a` and `j` up to the length of `b`, and a list has no word at those places. Row `i` of the table is about the first `i` words of `a`: at which place of `a` is the last of those words?",
                    nl="Met `{call}` stopt je functie met een `IndexError`: `i` gaat tot de lengte van `a` en `j` tot de lengte van `b`, en op deze plaatsen heeft een lijst geen woord. Rij `i` van de tabel gaat over de eerste `i` woorden van `a`: op welke plaats van `a` staat het laatste van deze woorden?"),
            Mistake("a-table", matches=_a_table,
                    en="`{call}` gives the whole table, and it should give one number, {want}. Which cell holds the length of the longest common subsequence of all of `a` and all of `b`?",
                    nl="`{call}` geeft de hele tabel, en het moet één getal zijn, {want}. Welke cel bevat de lengte van de langste gemeenschappelijke deelrij van heel `a` en heel `b`?"),
            Mistake("not-whole", matches=_not_whole,
                    en="`{call}` gives {got}, and it should give {want}: one whole number, a length. What does your function give back?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: één geheel getal, een lengte. Wat geeft je functie terug?"),
            Mistake("min", wrong=_LC_MIN,
                    en="`{call}` gives {got}, and it should give {want}: where two words differ, your table keeps the shorter of the two subsequences. The longest common subsequence can leave out either word: which of the two do you keep?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: waar twee woorden verschillen, houdt je tabel de kortste van de twee deelrijen. De langste gemeenschappelijke deelrij kan elk van de twee woorden weglaten: welke van de twee houd je?"),
            Mistake("no-one", wrong=_LC_NO_ONE,
                    en="`{call}` gives {got}, and it should give {want}: when two words are the same, your table copies the cell up and to the left, so a shared word adds nothing. How much longer does the subsequence get with one shared word?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: als twee woorden gelijk zijn, neemt je tabel de cel linksboven over, dus een gedeeld woord voegt niets toe. Hoeveel langer wordt de deelrij met één gedeeld woord?"),
            Mistake("up-one", wrong=_LC_UP_ONE,
                    en="`{call}` gives {got}, and it should give {want}: on a shared word, your table adds 1 to the cell above, which already counts the word of `b`: a word of `b` can then count twice. Which cell holds the subsequence of the parts without these two words?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: bij een gedeeld woord telt je tabel 1 op bij de cel erboven, die het woord van `b` al meetelt: een woord van `b` kan dan twee keer tellen. Welke cel bevat de deelrij van de stukken zonder deze twee woorden?"),
            Mistake("zero", wrong=_LC_ZERO,
                    en="`{call}` gives {got}, and it should give {want}: where two words differ, your table starts again at 0, so it only counts shared words in a row, at the end of both lists. A subsequence may skip words: what is the longest one without the word of `a`, or without the word of `b`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: waar twee woorden verschillen, begint je tabel weer bij 0, dus de tabel telt alleen gedeelde woorden op een rij, aan het eind van allebei de lijsten. Een deelrij mag woorden overslaan: wat is de langste zonder het woord van `a`, of zonder het woord van `b`?"),
            Mistake("diagonal", wrong=_LC_DIAGONAL,
                    en="`{call}` gives {got}, and it should give {want}: where two words differ, your table copies the cell up and to the left, which leaves out both words. Could the longest subsequence keep one of the two?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: waar twee woorden verschillen, neemt je tabel de cel linksboven over, en deze cel laat allebei de woorden weg. Kan de langste deelrij een van de twee houden?"),
            Mistake("from-0", wrong=_LC_FROM_0,
                    en="`{call}` gives {got}, and it should give {want}: your loops start at 0, and there `L[i - 1, j]` is `L[-1, j]`: the last row of the table, not the row above. The first row and column stay 0; at which number do the loops start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je loops beginnen bij 0, en daar is `L[i - 1, j]` gelijk aan `L[-1, j]`: de laatste rij van de tabel, niet de rij erboven. De eerste rij en kolom blijven 0; bij welk getal beginnen de loops?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the first says what `L[i, j]` is when the two words are the same, and the second what it is when they differ.",
                      "nl": "Vervang de twee `...`: de eerste zegt wat `L[i, j]` is als de twee woorden gelijk zijn, en de tweede wat het is als de woorden verschillen."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. When the two words are the same, does the cell get the cell up and to the left plus 1? And when they differ, does the cell get the larger of the cell above and the cell to the left?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Krijgt de cel de cel linksboven plus 1 als de twee woorden gelijk zijn? En krijgt de cel de grootste van de cel erboven en de cel links ervan als de woorden verschillen?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give 0 when a list is empty, and does it count a word that comes twice only as often as both lists allow?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie 0 als een lijst leeg is, en telt de functie een woord dat twee keer voorkomt slechts zo vaak als allebei de lijsten toelaten?"},
            "correct": {"en": "Right: your function finds the longest common subsequence. In the picture below, which summary of Qwen3-0.6B keeps its words of the TLDR in the order of the TLDR, and which does not?",
                        "nl": "Goed: je functie vindt de langste gemeenschappelijke deelrij. Kijk in het plaatje hieronder: welke samenvatting van Qwen3-0.6B houdt de woorden van de TLDR in de volgorde van de TLDR, en welke niet?"},
        },
    ),
    "skip_bigrams": FunctionBox(
        name="skip_bigrams",
        picture=skip_picture,
        function="skip_bigrams",
        start=_SB_START,
        solution=_SB,
        show=_show,
        collect=_as_pairs,
        compare=_same_pairs,
        visible=[(_written(ws),) for ws in TOY_SKIP],
        hidden=[(_written([]),), (_written(["bees"]),), (_written(["the", "the", "end"]),),
                (_written(["the", "gunman", "kill", "police"]),), (_written(["a", "b", "c", "d", "e"]),)]
               + [(_named("TLDR_WORDS", key),) for key in ("plastic",)],
        first=[_DOTS],
        mistakes=[
            Mistake("append-two", error=_append_two,
                    en="`{call}` stops with a `TypeError`: `append` takes one item, and your code gives it two. How do the two words become one tuple?",
                    nl="`{call}` stopt met een `TypeError`: `append` neemt één item, en je code geeft er twee. Hoe worden de twee woorden samen één tuple?"),
            Mistake("not-list", matches=_not_list,
                    en="`{call}` gives {got}, and it should give {want}: a list of pairs. Does your function give back `pairs`?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een lijst met paren. Geeft je functie `pairs` terug?"),
            Mistake("all", wrong=_SB_ALL,
                    en="`{call}` gives {got}, and it should give {want}: your pairs also go backward, and pair each word with itself. A skip-bigram keeps the order of the sentence: where does the second word of a pair stand?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: je paren gaan ook achteruit, en koppelen elk woord aan zichzelf. Een skip-bigram houdt de volgorde van de zin: waar staat het tweede woord van een paar?"),
            Mistake("self", wrong=_SB_SELF,
                    en="`{call}` gives {got}, and it should give {want}: each word also makes a pair with itself, at the same place. The second word of a pair comes later in the sentence: at which place does `j` start?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: elk woord maakt ook een paar met zichzelf, op dezelfde plaats. Het tweede woord van een paar komt later in de zin: bij welke plaats begint `j`?"),
            Mistake("next", syntax=r"range\(\s*i\s*\+\s*1\s*,\s*i\s*\+\s*2\s*\)",
                    en="Your function stops with an error on the line `{line}`: `j` only takes the place right after `i`, and after the last word there is none. And even without the error, your pairs would be only the words that stand next to each other, the bigrams of ROUGE-2. A skip-bigram allows any gap: up to which place does `j` go?",
                    nl="Je functie stopt met een fout op de regel `{line}`: `j` neemt alleen de plaats direct na `i`, en na het laatste woord is er geen. En ook zonder de fout zouden je paren alleen de woorden zijn die naast elkaar staan, de bigrams van ROUGE-2. Een skip-bigram mag elk gat hebben: tot welke plaats gaat `j`?"),
            Mistake("short", wrong=_SB_SHORT,
                    en="`{call}` gives {got}, and it should give {want}: no pair ends with the last word. `range` stops before the number it gets: which number lets `j` reach the last place?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: geen paar eindigt met het laatste woord. `range` stopt net onder het getal dat het krijgt: welk getal laat `j` tot de laatste plaats komen?"),
            Mistake("backward", wrong=_SB_BACK,
                    en="`{call}` gives {got}, and it should give {want}: each pair has the later word first. A skip-bigram keeps the order of the sentence: which word comes first?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: in elk paar staat het latere woord eerst. Een skip-bigram houdt de volgorde van de zin: welk woord komt eerst?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: the `range` of `j` gives every place after `i`, and `pairs.append` adds the pair of the word at `i` and the word at `j`, as one tuple.",
                      "nl": "Vervang de twee `...`: de `range` van `j` geeft elke plaats na `i`, en `pairs.append` voegt het paar van het woord op `i` en het woord op `j` toe, als één tuple."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every pair hold a word and a word later in the sentence, in that order, and is every such pair there once?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Bevat elk paar een woord en een woord later in de zin, in deze volgorde, en staat elk zo'n paar er één keer in?"},
            "almost": {"en": "Right for the two small cases. But `{call}` gives {got}, and it should give {want}. Does your function give an empty list for no words and for one word, and pairs of the same word when a word comes twice?",
                       "nl": "Goed voor de twee kleine gevallen. Maar `{call}` geeft {got}, en het moet {want} zijn. Geeft je functie een lege lijst bij geen woorden en bij één woord, en paren van hetzelfde woord als een woord twee keer voorkomt?"},
            "correct": {"en": "Right: your function makes every pair of words in their order. In the picture below, which sentence has every word of S1, and still shares only two of its six pairs with S1?",
                        "nl": "Goed: je functie maakt elk paar woorden in de volgorde van de zin. Kijk in het plaatje hieronder: welke zin heeft elk woord van S1, en deelt toch slechts twee van de zes paren met S1?"},
        },
    ),
}

# The summary and reference of Exercise 4; each change makes a new summary, and the tests check its ROUGE-1 recall.
CHANGE_REFERENCE = "A small network on a cheap board counts the bees at a hive."
CHANGE_SUMMARY = "A small network counts the bees at a hive."
CHANGED = {
    "never": "A small network never counts the bees at a hive.",
    "shuffled": "A small hive counts the bees at a network.",
    "twice": "A small network counts the bees, the bees at a hive.",
    "claim": "A small network counts the bees at a hive and beats every other counter.",
    "synonym": "A tiny network counts the bees at a hive.",
    "form": "A small network counts the bee at a hive.",
    "pronoun": "It counts the bees at a hive.",
    "shorter": "A small network counts the bees.",
}

EXERCISES["changes"] = Sort(
    name="changes",
    groups={
        "same": {"en": "ROUGE-1 recall stays the same", "nl": "ROUGE-1-recall blijft gelijk"},
        "lower": {"en": "ROUGE-1 recall goes down", "nl": "ROUGE-1-recall daalt"},
    },
    items={
        "never": {"en": "**Never.** *A small network never counts the bees at a hive.*",
                  "nl": "**Nooit.** *A small network never counts the bees at a hive.*: het woord *never* (nooit) erbij."},
        "shuffled": {"en": "**Shuffled.** *A small hive counts the bees at a network.*",
                     "nl": "**Door elkaar.** *A small hive counts the bees at a network.*: dezelfde woorden, op andere plaatsen."},
        "twice": {"en": "**Twice.** *A small network counts the bees, the bees at a hive.*",
                  "nl": "**Twee keer.** *A small network counts the bees, the bees at a hive.*: *the bees* staat er twee keer in."},
        "claim": {"en": "**A claim of its own.** *A small network counts the bees at a hive and beats every other counter.*",
                  "nl": "**Een eigen bewering.** *A small network counts the bees at a hive and beats every other counter.*: het netwerk zou elke andere teller verslaan."},
        "synonym": {"en": "**Another word.** *A tiny network counts the bees at a hive.*",
                    "nl": "**Een ander woord.** *A tiny network counts the bees at a hive.*: *tiny* in plaats van *small*, met dezelfde betekenis."},
        "form": {"en": "**Another form.** *A small network counts the bee at a hive.*",
                 "nl": "**Een andere vorm.** *A small network counts the bee at a hive.*: enkelvoud *bee* in plaats van meervoud *bees*."},
        "pronoun": {"en": "**A pronoun.** *It counts the bees at a hive.*",
                    "nl": "**Een voornaamwoord.** *It counts the bees at a hive.*: *it* (het) in plaats van *a small network*."},
        "shorter": {"en": "**Shorter.** *A small network counts the bees.*",
                    "nl": "**Korter.** *A small network counts the bees.*: zonder *at a hive*."},
    },
    solution={"never": "same", "shuffled": "same", "twice": "same", "claim": "same",
              "synonym": "lower", "form": "lower", "pronoun": "lower", "shorter": "lower"},
    nudge={
        "never": {"en": "Every word of the summary is still there, and *never* is a word the reference does not have. Does a word the reference lacks change how many of its words come back?",
                  "nl": "Elk woord van de samenvatting staat er nog, en *never* is een woord dat de referentie niet heeft. Verandert een woord dat niet in de referentie staat, hoeveel woorden van de referentie terugkomen?"},
        "shuffled": {"en": "The words are the same, each as often; only their places changed. Does ROUGE-1 look at the order of the words?",
                     "nl": "De woorden zijn dezelfde, elk even vaak; alleen de plaatsen zijn anders. Kijkt ROUGE-1 naar de volgorde van de woorden?"},
        "twice": {"en": "The reference has *the* and *bees* once each. How often may a word count, clipped as in BLEU?",
                  "nl": "De referentie heeft *the* en *bees* elk één keer. Hoe vaak mag een woord meetellen, afgetopt zoals bij BLEU?"},
        "claim": {"en": "The new words are not in the reference, and the old ones are all still there. What happens to the share of the reference that comes back?",
                  "nl": "De nieuwe woorden staan niet in de referentie, en de oude staan er allemaal nog. Wat gebeurt er met het aandeel van de referentie dat terugkomt?"},
        "synonym": {"en": "*tiny* means the same as *small*, but ROUGE compares the words themselves. Is *small* still in the summary?",
                    "nl": "*tiny* betekent hetzelfde als *small*, maar ROUGE vergelijkt de woorden zelf. Staat *small* nog in de samenvatting?"},
        "form": {"en": "*bee* and *bees* are two forms of one word, and the course does not bring forms together. Is *bees* still in the summary?",
                 "nl": "*bee* en *bees* zijn twee vormen van één woord, en de course brengt vormen niet samen. Staat *bees* nog in de samenvatting?"},
        "pronoun": {"en": "*It* stands for *a small network*, but only to a reader. Which words of the reference are gone?",
                    "nl": "*It* staat voor *a small network*, maar alleen voor een lezer. Welke woorden van de referentie zijn weg?"},
        "shorter": {"en": "*at a hive* is gone, and the reference has those words. What happens to the share of the reference that comes back?",
                    "nl": "*at a hive* is weg, en de referentie heeft deze woorden. Wat gebeurt er met het aandeel van de referentie dat terugkomt?"},
    },
    texts={
        "missing": {"en": "I cannot see your choices. Run the cell with the groups again.",
                    "nl": "Ik zie je keuzes niet. Voer de cel met de groepen opnieuw uit."},
        "empty": {"en": "Put each change with what it does to ROUGE-1 recall.",
                  "nl": "Zet elke verandering bij het effect op ROUGE-1-recall."},
        "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                       "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
        "wrong": {"en": "{item} belongs with the other group.",
                  "nl": "{item} hoort bij de andere groep."},
        "correct": {"en": "All {n} are right. ROUGE-1 sees which words of the reference come back, so a word of another form or a word with the same meaning costs points, and a word that reverses the meaning, a shuffle or a claim of its own costs nothing. ROUGE-2 and ROUGE-L do see the shuffle, but none of the three sees *never* or the claim of its own.",
                    "nl": "Alle {n} kloppen. ROUGE-1 ziet welke woorden van de referentie terugkomen, dus een woord in een andere vorm of een woord met dezelfde betekenis kost punten, en een woord dat de betekenis omdraait, woorden door elkaar of een eigen bewering kosten niets. ROUGE-2 en ROUGE-L zien wel dat de woorden door elkaar staan, maar geen van de drie ziet *never* of de eigen bewering."},
    },
)

# What the tests feed each check, and what it must say: (exercise, code or choices, state, known mistake).
CASES = [
    ("rouge_n", _RN_START, "empty", None),
    ("rouge_n", _RN, "correct", None),
    ("rouge_n", _rn(ret="    return ...\n"), "wrong", "dots-left"),
    ("rouge_n", _rn(ret="    return str(matched / total)\n"), "wrong", "not-number"),
    ("rouge_n", _RN_PRECISION, "wrong", "precision"),
    ("rouge_n", _RN_UNCLIPPED, "wrong", "unclipped"),
    ("rouge_n", _RN_THEIRS, "wrong", "theirs-count"),
    ("rouge_n", _RN_ONE, "wrong", "one-each"),
    ("rouge_n", _RN_DISTINCT, "wrong", "distinct"),
    ("rouge_n", _RN_COUNT, "wrong", "count"),
    ("rouge_n", _RN_PERCENT, "wrong", "percent"),
    ("rouge_n", _RN_FLOOR, "wrong", "floor"),
    ("rouge_n", _RN_INSIDE, "wrong", "return-inside"),
    ("rouge_n", _rn(add="            matched += min(mine[gram], theirs[gram])\n"), "correct", None),
    ("rouge_n", "def rouge_n(mine, theirs):\n    matched = sum(min(mine[g], theirs[g]) for g in theirs if g in mine)\n"
     "    return matched / sum(theirs.values())\n", "correct", None),
    ("rouge_n", "def rouge_n(mine, theirs):\n    matched = 0\n    for gram in mine:\n        if gram in theirs:\n"
     "            matched = matched + min(mine[gram], theirs[gram])\n    return matched / sum(theirs.values())\n", "correct", None),
    ("lcs", _LC_START, "empty", None),
    ("lcs", _LC, "correct", None),
    ("lcs", _lc(other="                L[i, j] = ...\n"), "wrong", "dots-left"),
    ("lcs", _LC_INDEX, "error", "index"),
    ("lcs", _LC_TABLE, "wrong", "a-table"),
    ("lcs", _lc(ret="    return str(L[len(a), len(b)])\n"), "wrong", "not-whole"),
    ("lcs", _LC_MIN, "wrong", "min"),
    ("lcs", _LC_NO_ONE, "wrong", "no-one"),
    ("lcs", _LC_UP_ONE, "wrong", "up-one"),
    ("lcs", _LC_ZERO, "wrong", "zero"),
    ("lcs", _LC_DIAGONAL, "wrong", "diagonal"),
    ("lcs", _LC_FROM_0, "wrong", "from-0"),
    ("lcs", _lc("                L[i, j] = 1 + L[i - 1, j - 1]\n", "                L[i, j] = max(L[i, j - 1], L[i - 1, j])\n"),
     "correct", None),
    ("lcs", _lc(ret="    return int(L[-1, -1])\n"), "correct", None),
    ("lcs", _lc("                L[i][j] = L[i - 1][j - 1] + 1\n", "                L[i][j] = max(L[i - 1][j], L[i][j - 1])\n"),
     "correct", None),
    ("skip_bigrams", _SB_START, "empty", None),
    ("skip_bigrams", _SB, "correct", None),
    ("skip_bigrams", _sb(add="            pairs.append(...)\n"), "wrong", "dots-left"),
    ("skip_bigrams", _sb(add="            pairs.append(words[i], words[j])\n"), "error", "append-two"),
    ("skip_bigrams", _SB_HEAD + _SB_FOR + _SB_ADD + "    return len(pairs)\n", "wrong", "not-list"),
    ("skip_bigrams", _SB_ALL, "wrong", "all"),
    ("skip_bigrams", _SB_SELF, "wrong", "self"),
    ("skip_bigrams", _SB_NEXT, "error", "next"),
    ("skip_bigrams", _SB_SHORT, "wrong", "short"),
    ("skip_bigrams", _SB_BACK, "wrong", "backward"),
    ("skip_bigrams", _sb(add="            pairs.append([words[i], words[j]])\n"), "correct", None),
    ("skip_bigrams", "def skip_bigrams(words):\n    pairs = []\n    for j in range(len(words)):\n        for i in range(j):\n"
     "            pairs.append((words[i], words[j]))\n    return pairs\n", "correct", None),
    ("skip_bigrams", "def skip_bigrams(words):\n    return [(words[i], words[j]) for i in range(len(words)) for j in range(i + 1, len(words))]\n",
     "correct", None),
    ("changes", {}, "empty", None),
    ("changes", dict(EXERCISES["changes"].solution), "correct", None),
    ("changes", {"never": "same", "synonym": "lower"}, "wrong", "unfinished"),
] + [("changes", {item: "lower" if group == "same" else "same"}, "wrong", f"wrong:{item}")
     for item, group in EXERCISES["changes"].solution.items()]
