# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-chunking. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-chunking", "en")
    C = L.checks   # the documents, what the course measured, the checks, and the demo with your chunk


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [From files to text](/course/open/build-documents-to-text?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_parse():
    quick_parse = L.choice(
        "quick-parse",
        question="A web page goes into your knowledge base with only its tags taken out. What else ends up in its text?",
        options={
            "nothing": "Nothing else: without its tags, a page is the text its writer wrote",
            "boilerplate": "The menu, the footer and the code of the page, around the text that matters",
            "pictures": "A description of every picture on the page, which the browser makes",
        },
        correct="boilerplate",
        why={
            "boilerplate": "Taking out the tags keeps all the text between them: the menu and the footer of every page, and the code of its scripts. Ingestion has to find the main text and leave the rest out.",
            "nothing": "The text between the tags is more than the article: the menu, the footer and the code of the scripts are text too. See [From files to text](/course/open/build-documents-to-text?lang=en).",
            "pictures": "A browser makes no descriptions; only what is in the page itself remains, such as the menu, the footer and the code of the scripts. See [From files to text](/course/open/build-documents-to-text?lang=en).",
        },
    )
    quick_parse
    return (quick_parse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The word search of [Your first RAG](/course/open/build-rag-first?lang=en) went through points of about thirty words. Real documents are longer: a handbook, a report, a page of notes. If you search whole documents, the search can tell you which document fits the question, but not where in it the answer is, and the model then has to read the whole document to find it. Three documents of a few thousand words each are thousands of tokens with every question, and they hide the one sentence that matters among all the others. So a RAG system cuts its documents into pieces first, and searches the pieces.

    **An everyday example.** The index at the back of a book does not say *this book is about bikes*; it says *brakes, page 41*. You turn to one page, not to the whole book. That is only a comparison; below you measure how often the right piece comes first when the course cuts its own summaries in different ways.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    _w = C.document_words()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **chunk** | a piece of a document that is stored and searched as one, from a sentence to a few hundred words long |
    | **chunking** | cutting documents into chunks before they are searched |
    | **chunk overlap** | the words that two neighboring chunks share: the next chunk starts a little before the end of the one before it |

    ### Three ways to cut

    - **A fixed size:** every chunk has the same number of words (or tokens), one after another. Simple, but a cut can fall in the middle of a sentence, between a question's words and its answer. An overlap makes it less likely that what belongs together is never in one chunk.
    - **Along the structure:** cut where the text itself has borders, at headings and paragraphs, and only cut a piece that is still too long. A *recursive* splitter goes on from there: a paragraph that is too long is cut at sentences, a sentence that is too long at words.
    - **Along the meaning:** cut where the subject changes (*semantic chunking*). That needs a way to compare the meaning of sentences, which a later lesson on embeddings brings.

    ### What the course measured

    The course put the points of the summaries of each module after each other, with a blank line between two points: {n} documents of {lo} to {hi} words, one per module. Then it cut them in twelve ways, and for each of the {q} questions of [Your first RAG](/course/open/build-rag-first?lang=en) that share their words with their point, let the word search of that lesson put the chunks in order. Each square is a question; the key under the picture says whether the chunk that came first holds its answer:
    """.format(n=len(C.DOCUMENTS), lo=_w[0], hi=C.number(_w[-1], L.lang), q=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_ways():
    _f = C.findings()
    _k = {w: sum(_f[w]["found"]) for w in C.WAYS}
    mo.vstack([
        mo.md(r"""
        With chunks of 10 words, the first chunk held the answer for {t10} of the {q} questions: the question's words and the answer often ended up in different chunks, and a chunk that had many of the question's words did not have the number it asked for. Larger chunks found more, up to about 80 words, and an overlap found one or two more at every size: {t10} and {o10} at 10 words, {t40} and {o40} at 40, {t80} and {o80} at 80. Cutting along the paragraphs, which here are the points themselves, found {para}, with {para_words} words a chunk. Whole documents found {whole}, but a whole document has {whole_words} words on average: the three best are some {three} words for the model to read, where three paragraphs are about {three_para}.
        """.format(q=len(C.QUESTIONS), t10=_k[("fixed", 10, 0)], o10=_k[("fixed", 10, 2)], t40=_k[("fixed", 40, 0)], o40=_k[("fixed", 40, 10)],
                   t80=_k[("fixed", 80, 0)], o80=_k[("fixed", 80, 20)], para=_k[("paragraph", 80, 0)], whole=_k[("whole", 0, 0)],
                   para_words=C.number(_f[("paragraph", 80, 0)]["words"], L.lang), whole_words=C.number(_f[("whole", 0, 0)]["words"], L.lang),
                   three=C.number(3 * _f[("whole", 0, 0)]["words"], L.lang, 0), three_para=C.number(3 * _f[("paragraph", 80, 0)]["words"], L.lang, 0))),
        L.watch_out("Seventeen questions are few: one question more or less can be chance. What holds up is the shape: chunks that are much too small lose the connection between a question and its answer, chunks that are much too large make the model read a lot, and cutting where the text itself has borders does well. Measure your own documents with questions whose answers you know."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: cutting documents into chunks

    1. **Cut along the structure** the text has: headings, paragraphs.
    2. **Cut what is still too long** at a fixed size.
    3. **Let neighboring chunks overlap** a little, often a tenth to a quarter of their size, where you cut at a fixed size.
    4. **Measure** with questions whose answers you know: how often does the right chunk come first?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _pieces = C.chunk(C.EXAMPLE, 40, 10)
    _n = len(C.EXAMPLE.split())
    mo.md(r"""
    ### Worked example

    The smallest document with more than one lesson is the module on recurrent networks: {n} words. Cut it into chunks of 40 words with an overlap of 10.

    1. **The structure:** the six points are its paragraphs; here we cut at a fixed size, to see what that does.
    2. **A fixed size:** the first chunk is words 0 to 39. With an overlap of 10, each next chunk starts 40 − 10 = 30 words further: at 0, 30, 60, 90 and 120. The chunk at 120 has words 120 to 158, the end of the document, so it is the last: {k} chunks.
    3. **The overlap:** the first chunk ends with *{end}*, and the second starts with the same ten words. A sentence that the first cut breaks is whole in one of the two.
    4. **Measure:** that is what the picture above did, for all {docs} documents.
    """.format(n=_n, k=len(_pieces), end=" ".join(_pieces[0].split()[-10:]), docs=len(C.DOCUMENTS)))
    return


@app.cell(hide_code=True)
def text_chunk():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Chunks with an overlap

        Write `chunk(text, size, overlap)`, which cuts `text` into chunks of `size` words, as texts with a space between the words. Each next chunk starts `overlap` words before the end of the one before it, and the chunk that reaches the end of the text is the last: no chunk may lie inside the one before it. An empty text gives an empty list. The box cuts chunks one after another, without overlap. `break` stops a `while` loop at once. When it is right, the check also keeps it as a file, `work/projects/app/chunks.py`: [Project: your assistant reads your documents](/course/open/build-project-rag?lang=en) uses it.
        """),
        L.task("Step `size - overlap` words further, and stop after the chunk that reaches the end."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chunk():
    box_chunk = L.code_box("chunk", lines=10)
    box_chunk
    return (box_chunk,)


@app.cell(hide_code=True)
def check_chunk(box_chunk):
    result_chunk = L.check("chunk", box_chunk.value)
    result_chunk
    return (result_chunk,)


@app.cell(hide_code=True)
def hints_chunk():
    L.hints(
        "chunk",
        [
            "Two changes: the step from one chunk to the next, and a test right after `append` whether this chunk already reached the end.",
            "The next start is `start + size - overlap`. This chunk reaches the end when `start + size >= len(words)`; then `break`.",
            "After the `append`: `if start + size >= len(words):` with `break` under it, then `start = start + size - overlap`.",
        ],
        why="""```python
def chunk(text, size, overlap):
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start = start + size - overlap
    return out
```

Without the `break`, the loop goes on while `start` is still inside the text, and after the chunk that reached the end comes one more, made of the last few words, which the chunk before it already had.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: your chunk on the documents

    Once exercise 1 is right, the demo cuts the {n} documents with your `chunk`, at the size and overlap you choose, and lets the word search put the chunks in order for each of the {q} questions. Which size finds the most? And what does the overlap add?
    """.format(n=len(C.DOCUMENTS), q=len(C.QUESTIONS)))
    return


@app.cell(hide_code=True)
def settings_cell():
    size_slider = mo.ui.slider(start=10, stop=160, step=10, value=40, label="Words in a chunk", show_value=True)
    overlap_choice = mo.ui.dropdown(options={"none": 0, "a tenth": 10, "a quarter": 4, "half": 2}, value="none", label="Overlap")
    mo.hstack([size_slider, overlap_choice], justify="start", gap=2)
    return overlap_choice, size_slider


@app.cell(hide_code=True)
def demo_chunk(box_chunk, overlap_choice, size_slider):
    _fn = C.learner_chunk(box_chunk.value)
    if _fn is None:
        _out = mo.md("*Finish exercise 1 first: the demo cuts with your own `chunk`.*")
    else:
        _size = size_slider.value
        _overlap = _size // overlap_choice.value if overlap_choice.value else 0
        _words = {"title": "Your chunk on the documents: {size} words, {overlap} overlap",
                  "result": "The first chunk held the answer for {k} of the {n} questions; {chunks} chunks of {w} words on average.",
                  "failed": "Your chunk stopped with an error on the documents: {why}"}
        _out = mo.Html(C.demo(_fn, _size, _overlap, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_paragraph():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Along the paragraphs

        Write `by_paragraph(text, size)`, which gives every paragraph of `text` as a chunk, except a paragraph of more than `size` words: that one it cuts with `chunk(paragraph, size, 0)`, and it adds the pieces one at a time. Paragraphs are separated by a blank line, so `text.split("\n\n")` gives them: `split` with a text between its parentheses cuts at that text instead of at every space. `chunk` is given. The box gives every paragraph as a chunk, also one that is too long.
        """),
        L.task("Cut only a paragraph of more than `size` words, and add its pieces one at a time."),
    ])
    return


@app.cell(hide_code=True)
def exercise_paragraph():
    box_paragraph = L.code_box("by_paragraph", lines=9)
    box_paragraph
    return (box_paragraph,)


@app.cell(hide_code=True)
def check_paragraph(box_paragraph):
    result_paragraph = L.check("by_paragraph", box_paragraph.value)
    result_paragraph
    return (result_paragraph,)


@app.cell(hide_code=True)
def hints_paragraph():
    L.hints(
        "by_paragraph",
        [
            "Inside the loop, decide per paragraph: is it too long or not? A long one goes in as several pieces, a short one as it is.",
            "`len(paragraph.split())` counts its words. When that is more than `size`, go through `chunk(paragraph, size, 0)` and `append` each piece; otherwise `append` the paragraph.",
            "`if len(paragraph.split()) > size:` with `for piece in chunk(paragraph, size, 0):` and `out.append(piece)` under it, then `else:` with `out.append(paragraph)`.",
        ],
        why="""```python
def by_paragraph(text, size):
    out = []
    for paragraph in text.split("\\n\\n"):
        if len(paragraph.split()) > size:
            for piece in chunk(paragraph, size, 0):
                out.append(piece)
        else:
            out.append(paragraph)
    return out
```

`out.append(chunk(paragraph, size, 0))` would add the whole list of pieces as one item: a list inside your list, which a search cannot read as a text.""",
    )
    return


@app.cell(hide_code=True)
def text_small():
    mo.md(r"""
    ### Exercise 3. Too small
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    _f = C.findings()
    choice_small = L.choice(
        "small",
        question="With chunks of 10 words, the first chunk held the answer for only {k} of the {n} questions. What is the most likely reason?".format(
            k=sum(_f[("fixed", 10, 0)]["found"]), n=len(C.QUESTIONS)),
        options={
            "slow": "With {c} chunks, the search has so many texts to go through that it skips some of them".format(c=C.number(_f[("fixed", 10, 0)]["chunks"], L.lang)),
            "split": "The answer and the question's words around it land in different chunks",
            "model": "A chunk of 10 words is too short for a language model to read, so it is passed over",
        },
        correct="split",
        why={
            "split": "The search puts first the chunk with the most of the question's words; in 10 words, that chunk often lacks the number the question asks for, which stands in the chunk next to it.",
            "slow": "The search goes through every chunk, however many there are; it skips nothing. The trouble is what a chunk of 10 words holds: the question's words and the answer often land in different chunks.",
            "model": "No model reads anything in this measurement: the word search alone puts the chunks in order. The trouble is that the question's words and the answer land in different chunks.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The documents come from [From files to text](/course/open/build-documents-to-text?lang=en), and the search that puts the chunks in order is the one of [Your first RAG](/course/open/build-rag-first?lang=en). That search counts every shared word the same, *the* as much as *perplexity*; [Search by words: BM25](/course/open/build-bm25?lang=en) lets a rare word count more.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_chunk, result_paragraph):
    L.wrap_up(result_chunk, result_paragraph, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
