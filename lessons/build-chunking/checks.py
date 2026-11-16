# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of build-chunking: a function that cuts a text into chunks of a fixed number of words with an overlap, and
one that cuts along paragraphs and cuts only a paragraph that is too long; what the course measured on the summaries
of the course, as 57 documents, one per module, for each way of cutting: how often the chunk the word search puts
first holds the answer to each of the 17 questions that share words with their point; the pictures, and the demo with
the learner's own chunk."""
import functools
import html

from everything_ai import summaries, yours
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit

# --- the documents: the points of each module's summaries after each other, a blank line between them -----------------

def _documents() -> dict:
    out = {}
    for text, about in zip(summaries.points("en"), summaries.about()):
        out.setdefault((about["domain"], about["module"]), []).append(text)
    return {key: "\n\n".join(texts) for key, texts in out.items()}


DOCUMENTS = list(_documents().values())
EXAMPLE = _documents()[("fundament", "rnn")]   # the worked example: a small document of 6 points
QUESTIONS = summaries.questions("words")

SOLUTION_CHUNK = '''def chunk(text, size, overlap):
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start = start + size - overlap
    return out
'''
chunk = run_code(SOLUTION_CHUNK).names["chunk"]

SOLUTION_PARAGRAPH = '''def by_paragraph(text, size):
    out = []
    for paragraph in text.split("\\n\\n"):
        if len(paragraph.split()) > size:
            for piece in chunk(paragraph, size, 0):
                out.append(piece)
        else:
            out.append(paragraph)
    return out
'''
by_paragraph = run_code(SOLUTION_PARAGRAPH, {"chunk": chunk}).names["by_paragraph"]

# the ways of cutting the course measured: (key, size, overlap); "paragraph" cuts a paragraph above 80 words
WAYS = [("fixed", 10, 0), ("fixed", 10, 2), ("fixed", 20, 0), ("fixed", 20, 5), ("fixed", 40, 0), ("fixed", 40, 10),
        ("fixed", 80, 0), ("fixed", 80, 20), ("fixed", 160, 0), ("fixed", 160, 40), ("paragraph", 80, 0), ("whole", 0, 0)]


def cut(way: tuple) -> list:
    kind, size, overlap = way
    if kind == "whole":
        return list(DOCUMENTS)
    if kind == "paragraph":
        return [c for d in DOCUMENTS for c in by_paragraph(d, size)]
    return [c for d in DOCUMENTS for c in chunk(d, size, overlap)]


def found(chunks: list) -> list:
    """For each question, whether the chunk the word search of build-rag-first puts first holds the answer. The same
    as summaries.search(question, chunks, 1), with the words of each chunk worked out once."""
    sets = [summaries.words(c) for c in chunks]
    out = []
    for q in QUESTIONS:
        asked = summaries.words(q["text"]["en"])
        best = max(zip([len(asked & s) for s in sets], chunks))[1]
        out.append(summaries.right(best, q))
    return out


@functools.cache
def findings() -> dict:
    """Per way of cutting: the chunks, how many words a chunk has on average, and which questions it found."""
    out = {}
    for way in WAYS:
        chunks = cut(way)
        out[way] = {"chunks": len(chunks), "words": sum(len(c.split()) for c in chunks) / len(chunks), "found": found(chunks)}
    return out


def document_words() -> list:
    return sorted(len(d.split()) for d in DOCUMENTS)


def number(x: float, lang: str, digits: int = 0) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


_LABEL = "font-family:var(--ea-font-ui);font-size:12px;color:var(--ea-ink-2)"

_WAY_WORDS = {
    "en": {"title": "How often the first chunk the search finds holds the answer, for 17 questions",
           "fixed": "{size} words", "overlap": "{size} words, {overlap} overlap", "paragraph": "paragraphs, cut above {size}",
           "whole": "whole documents", "found": "holds the answer", "missed": "does not", "count": "{k} of {n}",
           "words": "{w} words a chunk"},
    "nl": {"title": "Hoe vaak de eerste chunk die de zoekfunctie vindt het antwoord bevat, voor 17 vragen",
           "fixed": "{size} woorden", "overlap": "{size} woorden, {overlap} overlap", "paragraph": "alinea's, geknipt boven {size}",
           "whole": "hele documenten", "found": "bevat het antwoord", "missed": "bevat het niet", "count": "{k} van de {n}",
           "words": "{w} woorden per chunk"},
}


def way_name(way: tuple, lang: str) -> str:
    w = _WAY_WORDS[lang]
    kind, size, overlap = way
    return w[kind if kind != "fixed" else "overlap" if overlap else "fixed"].format(size=size, overlap=overlap)


def ways_picture(lang: str) -> str:
    """Per way of cutting a row: a square per question, as input data when the first chunk holds the answer and as a
    mistake when it does not, with the question on hovering; then the count and the words a chunk."""
    w = _WAY_WORDS[lang]
    f = findings()
    rows = []
    for way in WAYS:
        cells = "".join(f'<span title="{html.escape(q["text"][lang])}" style="display:inline-block;width:12px;height:12px;margin-right:2px;border-radius:2px;'
                        f'background:{"var(--ea-data)" if ok else "var(--ea-loss)"}"></span>' for q, ok in zip(QUESTIONS, f[way]["found"]))
        rows.append(f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.2rem 0.6rem;margin:0.2rem 0">'
                    f'<span style="{_LABEL};min-width:11.5rem">{html.escape(way_name(way, lang))}</span><span>{cells}</span>'
                    f'<span style="{_LABEL};min-width:3.5rem">{html.escape(w["count"].format(k=sum(f[way]["found"]), n=len(QUESTIONS)))}</span>'
                    f'<span style="{_LABEL}">{html.escape(w["words"].format(w=number(f[way]["words"], lang)))}</span></div>')
    legend = "".join(f'<span style="display:inline-flex;align-items:center;gap:0.25rem;margin-right:0.8rem;{_LABEL}">'
                     f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var(--ea-{c})"></span>{html.escape(w[k])}</span>'
                     for k, c in (("found", "data"), ("missed", "loss")))
    return yours.figure(w["title"], "".join(rows) + f'<div style="margin-top:0.3rem">{legend}</div>', alt=w["title"])


# --- exercise 1: chunk ---------------------------------------------------------------------------------------------------

SENTENCE = "A model answers from what it learned, and makes things up where what it learned falls short."
START_CHUNK = '''def chunk(text, size, overlap):
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        start = start + size
    return out
'''
TAIL_CHUNK = SOLUTION_CHUNK.replace("        if start + size >= len(words):\n            break\n", "")

_CHUNK_WORDS = {
    "en": {"title": "The chunks your chunk makes of the sentence, with size 6 and overlap 2", "inside": "lies inside the chunk before it"},
    "nl": {"title": "De chunks die je chunk van de zin maakt, met size 6 en overlap 2", "inside": "ligt binnen de chunk ervoor"},
}


def chunk_picture(seen) -> str:
    """The learner's chunks of the sentence with size 6 and overlap 2: a row per chunk, with a box per word of the
    sentence, filled as input where the chunk has that word; a chunk that lies inside the one before it as a mistake."""
    w = _CHUNK_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("chunk") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        got = with_time_limit(lambda: fn(SENTENCE, 6, 2))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, list):
        return ""
    words = SENTENCE.split()
    rows, before = [], None
    for piece in got[:12]:
        if not isinstance(piece, str):
            continue
        mine_words = piece.split()
        place = next((i for i in range(len(words)) if words[i:i + len(mine_words)] == mine_words), None)
        inside = before is not None and place is not None and place + len(mine_words) <= before[0] + before[1] and place >= before[0]
        cells = []
        for i, word in enumerate(words):
            has = place is not None and place <= i < place + len(mine_words)
            style = (("background:var(--ea-loss-wash);color:var(--ea-loss-text);border:1px solid var(--ea-loss)" if inside else
                      "background:var(--ea-data-wash);color:var(--ea-ink);border:1px solid var(--ea-data)") if has else
                     "background:transparent;color:var(--ea-ink-3);border:1px dashed var(--ea-line)")
            cells.append(f'<span style="display:inline-block;padding:0 0.2rem;margin:1px;border-radius:2px;font-family:var(--ea-font-ui);font-size:12px;{style}">{html.escape(word)}</span>')
        note = f'<span style="{_LABEL};margin-left:0.4rem">{html.escape(w["inside"])}</span>' if inside else ""
        rows.append(f'<div style="margin:2px 0">{"".join(cells)}{note}</div>')
        before = (place, len(mine_words)) if place is not None else before
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


# --- exercise 2: by_paragraph ------------------------------------------------------------------------------------------------

NOTES = ("Chunks are pieces of a document.\n\nA chunk that is too long holds many subjects, and its words match many "
         "questions a little.\n\nOverlap repeats words.")
START_PARAGRAPH = '''def by_paragraph(text, size):
    out = []
    for paragraph in text.split("\\n\\n"):
        out.append(paragraph)
    return out
'''

_PARAGRAPH_WORDS = {
    "en": {"title": "The chunks your by_paragraph makes of the notes, with size 8", "words": "{n} words", "word": "1 word",
           "list": "a list, not a text"},
    "nl": {"title": "De chunks die je by_paragraph van de notities maakt, met size 8", "words": "{n} woorden", "word": "1 woord",
           "list": "een lijst, geen tekst"},
}


def paragraph_picture(seen) -> str:
    """The learner's chunks of the notes with size 8: each chunk as a row with its words, as input, and its number of
    words; a chunk that is not a text, such as a list, as a mistake."""
    w = _PARAGRAPH_WORDS[seen.lang]
    mine = seen.mine
    fn = mine.names.get("by_paragraph") if mine is not None else None
    if not callable(fn):
        return ""
    try:
        got = with_time_limit(lambda: fn(NOTES, 8))
    except Exception:   # the learner's code; the check says what went wrong
        return ""
    if not isinstance(got, list):
        return ""
    rows = []
    for piece in got[:12]:
        text = piece if isinstance(piece, str) else repr(piece)
        style = ("background:var(--ea-data-wash);color:var(--ea-ink)" if isinstance(piece, str) else
                 "background:var(--ea-loss-wash);color:var(--ea-loss-text)")
        count = (w["word"] if len(piece.split()) == 1 else w["words"].format(n=len(piece.split()))) if isinstance(piece, str) else w["list"]
        rows.append(f'<div style="display:grid;grid-template-columns:minmax(0,1fr) 6.5rem;gap:0.5rem;align-items:center;margin:2px 0;'
                    f'padding:0.15rem 0.4rem;border-radius:3px;{style}"><span style="font-family:var(--ea-font-ui);font-size:12px;white-space:pre-wrap">{html.escape(text)}</span>'
                    f'<span style="{_LABEL};text-align:right">{html.escape(count)}</span></div>')
    return yours.figure(w["title"], "".join(rows), alt=w["title"])


_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the first lines again.",
            "nl": "Er staat geen code meer in het vak. Draai het terug met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de eerste regels."}

EXERCISES = {
    "chunk": FunctionBox(
        name="chunk",
        picture=chunk_picture,
        function="chunk",
        start=START_CHUNK,
        solution=SOLUTION_CHUNK,
        visible=[(SENTENCE, 6, 2)],
        hidden=[(SENTENCE, 6, 0), (SENTENCE, 20, 5), (SENTENCE, 5, 4), ("", 4, 1), ("one two three\nfour five six seven", 3, 1)],
        mistakes=[
            Mistake("tail", state="almost", wrong=TAIL_CHUNK,
                    en="Your chunks overlap. But `{call}` gives `{got}`: after the chunk that reaches the end of the text comes one more, which lies inside the one before it. Stop the loop once `start + size` reaches `len(words)`, with `break`.",
                    nl="Je chunks overlappen. Maar `{call}` geeft `{got}`: na de chunk die het eind van de tekst haalt, komt er nog één, die binnen de vorige ligt. Stop de loop zodra `start + size` bij `len(words)` komt, met `break`."),
            Mistake("plus", wrong=SOLUTION_CHUNK.replace("start + size - overlap", "start + size + overlap"),
                    en="`{call}` gives `{got}`: there are words between the chunks that no chunk has. The next chunk starts `overlap` words before the end of the last one, so the step is `size - overlap`.",
                    nl="`{call}` geeft `{got}`: er zitten woorden tussen de chunks die geen chunk heeft. De volgende chunk begint `overlap` woorden vóór het eind van de vorige, dus de stap is `size - overlap`."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`chunk` cuts chunks of `size` words one after another, without overlap. Let each next chunk start `overlap` words before the end of the last one, and stop after the chunk that reaches the end.",
                      "nl": "`chunk` knipt chunks van `size` woorden achter elkaar, zonder overlap. Laat elke volgende chunk `overlap` woorden vóór het eind van de vorige beginnen, en stop na de chunk die het eind haalt."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: chunks of `size` words, each starting `overlap` words before the end of the one before it.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: chunks van `size` woorden, die elk `overlap` woorden vóór het eind van de vorige beginnen."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work without overlap, for a text shorter than one chunk, and for an empty text?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie zonder overlap, voor een tekst korter dan één chunk, en voor een lege tekst?"},
            "no_function": {"en": "Your box does not make a function `chunk`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `chunk`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `chunk` is right in all {n} cases: chunks of `size` words that overlap by `overlap`, up to the end and no further.",
                        "nl": "Je `chunk` klopt in alle {n} gevallen: chunks van `size` woorden die `overlap` woorden overlappen, tot het eind en niet verder."},
        },
    ),
    "by_paragraph": FunctionBox(
        name="by_paragraph",
        picture=paragraph_picture,
        function="by_paragraph",
        start=START_PARAGRAPH,
        solution=SOLUTION_PARAGRAPH,
        names={"chunk": chunk},
        visible=[(NOTES, 8)],
        hidden=[(NOTES, 20), (NOTES, 3), ("One paragraph without a blank line, long enough to cut.", 4)],
        mistakes=[
            Mistake("nested", wrong=SOLUTION_PARAGRAPH.replace("            for piece in chunk(paragraph, size, 0):\n                out.append(piece)\n",
                                                               "            out.append(chunk(paragraph, size, 0))\n"),
                    en="`{call}` gives `{got}`: the pieces of a long paragraph are in a list inside your list. Add them one at a time, with a `for` loop over `chunk(paragraph, size, 0)`.",
                    nl="`{call}` geeft `{got}`: de stukken van een lange alinea staan in een lijst binnen je lijst. Voeg ze één voor één toe, met een `for`-loop over `chunk(paragraph, size, 0)`."),
            Mistake("whole", wrong="def by_paragraph(text, size):\n    return chunk(text, size, 0)\n",
                    en="`{call}` gives `{got}`: chunks of `size` words straight through the text, across the paragraphs. Go through the paragraphs, and cut only a paragraph that has more than `size` words.",
                    nl="`{call}` geeft `{got}`: chunks van `size` woorden dwars door de tekst, over de alinea's heen. Ga de alinea's langs, en knip alleen een alinea die meer dan `size` woorden heeft."),
        ],
        texts={
            "missing": _MISSING,
            "cleared": _CLEARED,
            "empty": {"en": "`by_paragraph` gives every paragraph as a chunk, also one that is too long. Cut a paragraph of more than `size` words with `chunk(paragraph, size, 0)`, and keep the others whole.",
                      "nl": "`by_paragraph` geeft elke alinea als chunk, ook een die te lang is. Knip een alinea van meer dan `size` woorden met `chunk(paragraph, size, 0)`, en houd de andere heel."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`: every paragraph as a chunk, and a paragraph of more than `size` words cut into chunks of `size` words.",
                      "nl": "`{call}` geeft `{got}`, en het zou `{want}` moeten zijn: elke alinea als chunk, en een alinea van meer dan `size` woorden geknipt in chunks van `size` woorden."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work when no paragraph is too long, and when every one is?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en het zou `{want}` moeten zijn. Werkt je functie als geen enkele alinea te lang is, en als ze dat allemaal zijn?"},
            "no_function": {"en": "Your box does not make a function `by_paragraph`. Keep the `def` line as it was.",
                            "nl": "Je vak maakt geen functie `by_paragraph`. Laat de `def`-regel staan zoals hij was."},
            "correct": {"en": "Your `by_paragraph` is right in all {n} cases: a chunk per paragraph, and a paragraph that is too long cut into chunks.",
                        "nl": "Je `by_paragraph` klopt in alle {n} gevallen: een chunk per alinea, en een alinea die te lang is geknipt in chunks."},
        },
    ),
}

REQUIRED = ["chunk", "by_paragraph", "small"]

CASES = [
    ("chunk", START_CHUNK, "empty", None),
    ("chunk", "", "empty", None),
    ("chunk", SOLUTION_CHUNK, "correct", None),
    ("chunk", "def chunk(text, size, overlap):\n    words = text.split()\n    out = []\n    start = 0\n    while start < len(words):\n"
              "        out.append(\" \".join(words[start:start + size]))\n        if start + size >= len(words):\n            return out\n"
              "        start = start + size - overlap\n    return out\n", "correct", None),
    ("chunk", TAIL_CHUNK, "almost", "tail"),
    ("chunk", SOLUTION_CHUNK.replace("start + size - overlap", "start + size + overlap"), "wrong", "plus"),
    ("chunk", SOLUTION_CHUNK.replace("start + size - overlap", "start + size - 2 * overlap"), "wrong", None),
    ("by_paragraph", START_PARAGRAPH, "empty", None),
    ("by_paragraph", SOLUTION_PARAGRAPH, "correct", None),
    ("by_paragraph", "def by_paragraph(text, size):\n    out = []\n    for paragraph in text.split(\"\\n\\n\"):\n"
                     "        for piece in chunk(paragraph, size, 0):\n            out.append(piece)\n    return out\n", "correct", None),
    ("by_paragraph", SOLUTION_PARAGRAPH.replace("            for piece in chunk(paragraph, size, 0):\n                out.append(piece)\n",
                                                "            out.append(chunk(paragraph, size, 0))\n"), "wrong", "nested"),
    ("by_paragraph", "def by_paragraph(text, size):\n    return chunk(text, size, 0)\n", "wrong", "whole"),
    ("by_paragraph", SOLUTION_PARAGRAPH.replace("> size", "> 2 * size"), "almost", None),
]


# --- the demo: your chunk on the 57 documents -----------------------------------------------------------------------------

def learner_chunk(text):
    """The learner's chunk from exercise 1 when it is right; else None."""
    from everything_ai.check import MISSING
    if text is MISSING or text is None or EXERCISES["chunk"].check(text).state != "correct":
        return None
    return run_code(text).names["chunk"]


def demo(fn, size: int, overlap: int, words: dict, lang: str) -> str:
    """The learner's chunk on the 57 documents with this size and overlap: how many chunks, how many words a chunk has,
    and for how many of the 17 questions the first chunk the word search finds holds the answer."""
    try:
        chunks = with_time_limit(lambda: [c for d in DOCUMENTS for c in fn(d, size, overlap)], 10)
    except Exception as e:   # the learner's code
        return f"<p><em>{html.escape(words['failed'].format(why=f'{type(e).__name__}: {e}'))}</em></p>"
    hits = found(chunks)
    cells = "".join(f'<span title="{html.escape(q["text"][lang])}" style="display:inline-block;width:12px;height:12px;margin-right:2px;border-radius:2px;'
                    f'background:{"var(--ea-data)" if ok else "var(--ea-loss)"}"></span>' for q, ok in zip(QUESTIONS, hits))
    title = words["title"].format(size=size, overlap=overlap)
    body = (f'<div>{cells}</div>'
            f'<div style="{_LABEL};margin-top:0.3rem">{html.escape(words["result"].format(k=sum(hits), n=len(QUESTIONS), chunks=number(len(chunks), lang), w=number(sum(len(c.split()) for c in chunks) / max(1, len(chunks)), lang)))}</div>')
    return yours.figure(title, body, alt=title)
