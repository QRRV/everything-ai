# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-bm25. Route: course/route/lessen/bouwen.yaml. Checks, the pictures and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-bm25", "en")
    C = L.checks   # the points, what the course measured, the checks, and the demo with your BM25


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Exponents and logarithms](/course/open/math-exp-log?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_log():
    quick_log = L.choice(
        "quick-log",
        question="In this course, log without a base is the natural log, base e. Which number is closest to log(1000)?",
        options={
            "three": "3, since 10 times 10 times 10 is 1000",
            "seven": "About 6.9, since e to the power 6.9 is about 1000",
            "hundred": "About 368, since 1000 divided by e is about 368",
        },
        correct="seven",
        why={
            "seven": "The log gives the exponent: e to the power 6.9 is about 1000. A log grows slowly: a thousand times as many is only about 6.9 more.",
            "three": "3 is the log with base 10. Without a base, log means base e in this course: e to the power 6.9 is about 1000, so about 6.9. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
            "hundred": "A log does not divide: it gives the exponent. e to the power 6.9 is about 1000, so about 6.9. See [Exponents and logarithms](/course/open/math-exp-log?lang=en).",
        },
    )
    quick_log
    return (quick_log,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    The word search of [Your first RAG](/course/open/build-rag-first?lang=en) counts the words a point shares with the question, and every word counts the same: *the* and *of* as much as *DBSCAN*. But *the* is in almost every point, so it says nothing about which point fits; a rare word says a lot. The course gave that search the title of each of its {n} lessons as a question: it put a point of that lesson first for {o1} of them. BM25, a formula from 1994 that search engines such as Lucene and Elasticsearch still use (examples in November 2026), put the right lesson first for {b1}. It is the standard way to search by words, and next to search by meaning it is still part of many RAG systems.

    **An everyday example.** You look for a recipe with saffron. Every recipe has the word *the*, so it does not help you choose; only a few have *saffron*, and those are the ones you want. And a recipe that says *saffron* five times is not five times as much about saffron as one that says it once. That is only a comparison; below you measure what these two ideas do on the course's own summaries.
    """.format(n=_f["titles"], o1=_f["overlap"]["title_1"], b1=_f["bm25"]["title_1"]))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word or symbol | Say it | What it means |
    |---|---|---|
    | **lexical retrieval** | | searching texts by the words themselves, as they are written; the opposite is searching by meaning |
    | **term frequency**, $f$ | | how often a word appears in one text |
    | **inverse document frequency**, idf | "eye-dee-eff" | how rare a word is among all texts: high when few texts have it, near 0 when almost all do |
    | **BM25** | "bee-em-twenty-five" | a score for how well a text fits a question: the idf of every word they share, a little less for each extra time, and corrected for the length of the text |
    | $N$, $n$ | | the number of texts, and how many of them have the word |
    | $k_1$, $b$ | "k-one", "b" | BM25's two settings: how fast an extra time stops adding, and how much a long text is corrected for; usually 1.2 and 0.75 |

    ### Three ideas in one formula

    **A rare word counts more.** The idf of a word that $n$ of the $N$ texts have is

    $$\text{{idf}} = \log\left(\frac{{N - n + 0.5}}{{n + 0.5}} + 1\right)$$

    The 0.5 keeps it defined for a word that no text has, and the 1 keeps it above 0 for a word that almost every text has. Over the {n} points, these are the idf of the words of a lesson's title used as a question:
    """.format(n=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def picture_idf():
    mo.Html(C.idf_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_formula():
    mo.md(r"""
    **An extra time adds less, and a long text gets no head start.** For each word of the question, BM25 takes its idf times a part that grows with $f$, how often the text has the word, but less and less:

    $$\text{{score}} = \sum_{{\text{{words of the question}}}} \text{{idf}} \cdot \frac{{f \cdot (k_1 + 1)}}{{f + k_1 \cdot \left(1 - b + b \cdot \frac{{\text{{length}}}}{{\text{{average length}}}}\right)}}$$

    In a text of average length the part is 1 when the word is there once, about 1.4 when it is there twice and never more than $k_1 + 1 = 2.2$, however often. A text twice as long as average has more words, and so more chances to have a word of the question by luck; the length below the line makes its part smaller. A word that the text does not have adds 0.

    ### What the course measured

    The course gave the title of each of its {n} lessons as a question, and looked whether a point of that lesson came first, with the shared words of [Your first RAG](/course/open/build-rag-first?lang=en) and with BM25:
    """.format(n=C.findings()["titles"]))
    return


@app.cell(hide_code=True)
def picture_titles():
    mo.Html(C.titles_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_titles():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        BM25 put the right lesson first for {b1} of the {n} titles, against {o1} with the shared words, and among the first three for {b3}, against {o3}. A title is short and full of words such as *the*, *and* and *of*, which most points have: the shared words let those decide, BM25 hardly counts them. On the {w} questions of [Your first RAG](/course/open/build-rag-first?lang=en) that share their words with their point, the two did about the same: both put the right point first {ow1} times, and among the first three {ow3} and {bw3} times. Those questions repeat the rare words of their point, so even the plain count finds it. On the {m} questions that ask in other words, the shared words found {om3} and BM25 {bm3} among the first three.
        """.format(n=_f["titles"], b1=_f["bm25"]["title_1"], o1=_f["overlap"]["title_1"], b3=_f["bm25"]["title_3"], o3=_f["overlap"]["title_3"],
                   w=_f["words"], ow1=_f["overlap"]["words_1"], ow3=_f["overlap"]["words_3"], bw3=_f["bm25"]["words_3"],
                   m=_f["meaning"], om3=_f["overlap"]["meaning_3"], bm3=_f["bm25"]["meaning_3"])),
        L.watch_out("BM25 still only sees words as they are written. A question that asks the same in other words, *a classifier that looks at the closest stored examples* where the point says *k-nearest neighbors*, shares no rare word with its point, and no formula over words can find it. Searching by meaning, with embeddings, is the next step of this module."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: search with BM25

    1. **The words of every text, as a list:** BM25 counts how often each word appears, so keep every time.
    2. **The idf of every word:** count in how many texts it appears, and use the formula.
    3. **The average length** of a text, in words.
    4. **Score every text:** for each word of the question, its idf times the part with $f$, $k_1$ and $b$; add them up.
    5. **Sort** the texts from the highest score to the lowest, and take the first few.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _bm, _ov = C.first(C.EXAMPLE, "bm25", 1)[0], C.first(C.EXAMPLE, "overlap", 1)[0]
    _rows = []
    for _p, _q in zip(C.parts(C.EXAMPLE, _ov), C.parts(C.EXAMPLE, _bm)):
        _rows.append("| *{w}* | {n} | {idf} | {f1} | {p1} | {f2} | {p2} |".format(
            w=_p["word"], n=C.number(_p["n"], L.lang), idf=C.number(_p["idf"], L.lang, 2), f1=_p["f"], p1=C.number(_p["part"], L.lang, 2), f2=_q["f"], p2=C.number(_q["part"], L.lang, 2)))
    mo.md(r"""
    ### Worked example

    The title of the lesson on DBSCAN is *{q}*. Which point comes first?

    1. **The words:** *any*, *clusters*, *dbscan*, *of* and *shape*.
    2. **Their idf**, over the {n} points: in the picture above. *dbscan* is in {nd} points, *of* in {no}.
    3. **The average length** of a point is {avg} words.
    4. **Score:** two points share three words with the title. The first, from *{t1}*, says: *{ov}* The second, from the lesson on DBSCAN itself: *{bm}*

    | word | points with it | idf | first: $f$ | first: part | second: $f$ | second: part |
    |---|---|---|---|---|---|---|
    {rows}

    5. **Sort:** with the shared words both have 3, and the first comes first; with BM25 the first has {s1} and the second {s2}, so the lesson on DBSCAN comes first. *any* and *shape* are not rare enough to beat *clusters* and *dbscan*.
    """.format(q=C.EXAMPLE, n=C.number(len(C.POINTS), L.lang), nd=C.having("dbscan"), no=C.having("of"), avg=C.number(C.index()["average"], L.lang, 1),
               t1=C.lesson_of(_ov)["title"][L.lang], ov=_ov, bm=_bm, rows="\n".join(_rows),
               s1=C.number(sum(p["part"] for p in C.parts(C.EXAMPLE, _ov)), L.lang, 2), s2=C.number(sum(p["part"] for p in C.parts(C.EXAMPLE, _bm)), L.lang, 2)))
    return


@app.cell(hide_code=True)
def text_rung_up():
    mo.md(r"""
    ### One rung up

    A loop scores one point after the other. With NumPy, all {n} points at once: a table with a row per point and a column per word of the question, and the formula on the whole table. `point.count(word)` counts how often a list has a value; `lengths` is a column, one row per point, so dividing it by `average` and adding it to the table works row by row, as broadcasting does.

    ```python
    {code}
    ```

    | Python, one point at a time | NumPy, all points at once |
    |---|---|
    | `for word in asked:` | a column per word: `f` has a row per point |
    | `f = words.count(word)` | `point.count(word)` for every point and word, in `f` |
    | `idfs[word] * f * (K1 + 1) / (...)` | the same formula on the table: `part` |
    | `score = score + ...` | `(part * idf).sum(axis=1)`: a sum along each row |

    For the title of the lesson on DBSCAN, the {n} scores from NumPy are the scores of the loop, and the highest is {top}.
    """.format(n=C.number(len(C.POINTS), L.lang), code=C.NUMPY_CODE.strip(),
               top=C.number(float(C.numpy_scores(C.EXAMPLE).max()), L.lang, 2)))
    return


@app.cell(hide_code=True)
def text_idf():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How rare is a word?

        Write `idf(n, total)`, BM25's inverse document frequency of a word that `n` of `total` texts have, with the formula from the idea. The box has the classic idf, the log of `total / n`: it divides by zero for a word that no text has. `math.log` is the natural log, as `np.log`, for one number.
        """),
        L.task("Write BM25's idf: add 0.5 above and below the line, and 1 to the fraction, inside the log."),
    ])
    return


@app.cell(hide_code=True)
def exercise_idf():
    box_idf = L.code_box("idf", lines=5)
    box_idf
    return (box_idf,)


@app.cell(hide_code=True)
def check_idf(box_idf):
    result_idf = L.check("idf", box_idf.value)
    result_idf
    return (result_idf,)


@app.cell(hide_code=True)
def hints_idf():
    L.hints(
        "idf",
        [
            "Above the line: the texts that do not have the word, plus 0.5. Below it: the texts that do, plus 0.5. Then 1 more, and the log of all of it.",
            "The fraction is `(total - n + 0.5) / (n + 0.5)`. Add 1 to it, and take `math.log` of the sum.",
            "`return math.log((total - n + 0.5) / (n + 0.5) + 1)`",
        ],
        why="""```python
import math


def idf(n, total):
    return math.log((total - n + 0.5) / (n + 0.5) + 1)
```

Without the `+ 1`, a word that more than half of the texts have gets a log of a number below 1: an idf below 0, which would lower the score of every text that has it.""",
    )
    return


@app.cell(hide_code=True)
def text_bm25():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The score of a text

        Write `bm25(asked, words, idfs, average)`, the BM25 score of a text for a question. `asked` is the list of the question's words, `words` the list of the text's words, every time it appears, `idfs` a dictionary with the idf of every asked word, and `average` the average number of words of a text. `K1` and `B` are given, 1.2 and 0.75. The box adds `idf * f` for every asked word: every extra time counts as much as the first, and a long text gets a head start.
        """),
        L.task("Replace `f` by BM25's part: `f` times `K1 + 1`, divided by `f` plus `K1` times the length correction."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bm25():
    box_bm25 = L.code_box("bm25", lines=6)
    box_bm25
    return (box_bm25,)


@app.cell(hide_code=True)
def check_bm25(box_bm25):
    result_bm25 = L.check("bm25", box_bm25.value)
    result_bm25
    return (result_bm25,)


@app.cell(hide_code=True)
def hints_bm25():
    L.hints(
        "bm25",
        [
            "Only the line that adds to `score` changes: `idfs[word]` stays, and `f` becomes a fraction.",
            "Above the line: `f * (K1 + 1)`. Below it: `f + K1 * (1 - B + B * len(words) / average)`.",
            "`score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))`",
        ],
        why="""```python
def bm25(asked, words, idfs, average):
    score = 0
    for word in asked:
        f = words.count(word)
        score = score + idfs[word] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(words) / average))
    return score
```

When the text has average length, `1 - B + B * len(words) / average` is 1, and a word that is there once adds exactly its idf. A word the text does not have has `f` 0 and adds 0.""",
    )
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: your BM25 on the points

    Once exercises 1 and 2 are right, the demo searches the {n} points for your question with your `idf` and `bm25`, and puts the first three next to those of the shared words. The points are in English, so ask in English. Try a question with a rare word, and one in your own words.
    """.format(n=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def question_cell():
    question_text = mo.ui.text(value="Clusters of any shape: DBSCAN", label="Your question", full_width=True)
    question_text
    return (question_text,)


@app.cell(hide_code=True)
def demo_bm25(box_bm25, box_idf, question_text):
    _fns = C.learner_bm25(box_idf.value, box_bm25.value)
    if _fns is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the demo searches with your own `idf` and `bm25`.*")
    else:
        _words = {"title": "The first three points for your question", "overlap": "Shared words", "bm25": "Your BM25",
                  "empty": "Type a question first.", "failed": "Your BM25 stopped with an error on the points: {why}"}
        _out = mo.Html(C.demo(_fns, question_text.value, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_twice():
    mo.md(r"""
    ### Exercise 3. Ten times the same word
    """)
    return


@app.cell(hide_code=True)
def choice_twice_cell():
    choice_twice = L.choice(
        "twice",
        question="Two texts of average length both have the word *loss*: one once, the other ten times. How much more does *loss* add to the score of the second, with BM25?",
        options={
            "ten": "Ten times as much, since every time it appears counts the same",
            "twice": "About twice as much: each extra time adds less",
            "same": "Just as much, since BM25 only looks whether the word is there",
        },
        correct="twice",
        why={
            "twice": "With f = 1 the part is 2.2 / 2.2 = 1; with f = 10 it is 10 · 2.2 / 11.2 ≈ 1.96. However often a word appears, its part stays below k1 + 1 = 2.2.",
            "ten": "That is what plain counting does. In BM25, f is also below the line, so the part grows slower and slower: 1 for f = 1, about 1.96 for f = 10, and never above 2.2.",
            "same": "BM25 does count how often: f is in the formula. But each extra time adds less: 1 for f = 1, about 1.96 for f = 10.",
        },
    )
    choice_twice
    return (choice_twice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** BM25 replaces the shared words of [Your first RAG](/course/open/build-rag-first?lang=en) as the search in a RAG system; what it searches are the chunks of [Cutting documents into pieces](/course/open/build-chunking?lang=en). It works with the log of [Exponents and logarithms](/course/open/math-exp-log?lang=en). Where a question uses other words than its answer, searching by meaning helps, which comes next in this module; and a search that combines both, hybrid search, comes after that.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_twice, result_bm25, result_idf):
    L.wrap_up(result_idf, result_bm25, choice_twice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
