# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-cost-latency-cache. Route: course/route/lessen/bouwen.yaml. Checks, the calls of version 2, Granite's embeddings of the questions, the course's cache and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("build-cost-latency-cache", "en")
    C = L.checks   # version 2 of the bank's assistant on the 120 test questions, Granite's embeddings of them, the cache and the pictures
    QUESTIONS = C.QUESTIONS           # the 120 test questions of banking77-small, in the order they came in
    CALLS = C.CALLS                   # what version 2 did with each: "question", "seconds", "prompt_tokens", "output_tokens", "reply"
    SECONDS = C.SECONDS               # the seconds of each call to the model
    EMBED_SECONDS = C.EMBED_SECONDS   # the seconds of embedding each question with Granite, in a request of its own
    VECTORS = C.VECTORS               # VECTORS[i] is the embedding of QUESTIONS[i], of length 1
    PRICE_IN, PRICE_OUT = C.PRICE_IN, C.PRICE_OUT   # made-up prices, in dollars per million tokens


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Pay once for the same start](/course/open/build-prompt-caching?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reuse():
    quick_reuse = L.choice(
        "quick-reuse",
        question="With prompt caching, what does the server reuse from an earlier request?",
        options={
            "answer": "The whole answer to an earlier question",
            "start": "Its work on a start that repeats",
            "written": "The tokens the model wrote the last time",
        },
        correct="start",
        why={
            "start": "The server keeps its work on the start of the prompt, the KV cache, and goes on from there. The model still reads the rest and writes a new answer.",
            "answer": "That is what this lesson builds: a cache of answers. Prompt caching keeps the server's work on a start that repeats; the model still writes a new answer. See [Pay once for the same start](/course/open/build-prompt-caching?lang=en).",
            "written": "The model writes its answer anew each time. Prompt caching keeps the server's work on a start of the prompt that repeats. See [Pay once for the same start](/course/open/build-prompt-caching?lang=en).",
        },
    )
    quick_reuse
    return (quick_reuse,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    mo.md(r"""
    ## Why this matters

    In [An eval set and regression tests](/course/open/build-eval-set?lang=en), version 2 of the bank's assistant answered the {n} test questions. On the course's computer an answer took {median} seconds at the median, and the slowest took {slowest}: a customer would wait that long here. The {n} calls read {pin} tokens and wrote {pout}; at the made-up prices of [Batches, keys and the bill](/course/open/build-batch-keys-costs?lang=en), {pi} and {po} dollars for a million tokens, that is {per} dollars for 1,000 questions. Each question was paid for in full, while the {n} were about only {topics} topics, {each} questions each.

    [Pay once for the same start](/course/open/build-prompt-caching?lang=en) let the server keep its work on the start of a prompt; the model still read each question and wrote each answer. A cache of answers skips the model: it gives an answer it gave before.

    **An everyday example.** At an information desk the same few questions come every day. The person behind it does not look up the opening hours each time; they say what they said an hour ago. When someone asks a question that only sounds the same, though, a good answer from before is the wrong answer now.
    """.format(n=_f["n"], median=L.number(_f["median"], 2), slowest=L.number(_f["slowest"], 2), pin=C.num(_f["prompt_tokens"], L.lang),
               pout=C.num(_f["output_tokens"], L.lang), pi=C.PRICE_IN, po=C.PRICE_OUT, per=L.number(_f["per_1000"], 2),
               topics=len(set(C.LABELS)), each=_f["n"] // len(set(C.LABELS))))
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **latency** | the time from sending a request until its answer is there; for a customer, the wait |
    | **exact cache** | keeps answers by the exact text of the request, and gives one only when the same text comes again |
    | **semantic cache** | keeps answers by the embedding of the question, and gives one when a new question is close enough in meaning to an earlier one: its similarity is at least a threshold |

    Measure two things for every call: what it costs, its tokens in and out times their prices, and its latency. Look at the latency of many calls, not of one. The median says what a customer usually waits; p95, close to the 0.95 quantile, says what the slowest one in twenty gets. An average hides both.

    A cache sits in front of the model. An exact cache looks up the text: the same text gets the same answer, and that costs nothing. A semantic cache embeds the question, as in [Search by meaning](/course/open/build-retrieval-embeddings?lang=en), and looks for the closest question it answered before. From the threshold on, it gives that question's answer. Each hit saves a whole call, and each hit gives an answer that was written for another question.
    """)
    return


@app.cell(hide_code=True)
def picture_idea():
    mo.Html(C.flow_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: a cache of answers

    1. **Measure:** write down the seconds and the tokens of every call, as a trace does. Look at the median and p95, and at the cost of 1,000 calls.
    2. **Exact first:** make the text small letters without spaces at the ends, and look it up. A hit costs nothing and is always about the same question.
    3. **Then by meaning:** embed the question and find the closest question in the cache. From the threshold on, give its answer; otherwise ask the model and store its answer.
    4. **Store with care:** only answers that passed your eval set; the cache in this lesson stores every answer, so you see what that does. No answers about one customer, such as their card or their balance; answers about what changes, such as exchange rates, only with a time after which they expire.
    5. **Measure the cache as a version:** its hits, what they saved, and the eval set on the answers it gave.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _f = C.findings()
    _c = CALLS[0]
    mo.md(r"""
    ### Worked example

    `CALLS` holds what version 2 did with each of the {n} questions, in the order the course sent them, as if they came in one after another. `CALLS[0]` is *{q}*: it took {s} seconds, read {pt} tokens and wrote {ot}. `SECONDS` holds the seconds of all {n}: together {total}, {median} at the median, {p95} at p95 and {slowest} at most. That is the latency of the model alone, on the course's computer with {threads} threads; a server far away adds the way there and back.

    The course also embedded each question with Granite Embedding, the embedding model of the course's server, in a request of its own: `EMBED_SECONDS`. That took {em} seconds at the median, about {times} times as fast as an answer. `VECTORS` holds those embeddings, one row per question, each of length 1, so the dot product of two rows is their cosine similarity.

    The cost of one call is its prompt tokens times `PRICE_IN` plus its output tokens times `PRICE_OUT`, divided by a million: for `CALLS[0]` that is {one} dollars. All {n} cost {all} dollars.
    """.format(n=_f["n"], q=_c["question"], s=L.number(_c["seconds"], 2), pt=_c["prompt_tokens"], ot=_c["output_tokens"],
               total=L.number(_f["total"], 1), median=L.number(_f["median"], 2), p95=L.number(_f["p95"], 2), slowest=L.number(_f["slowest"], 2),
               threads=_f["threads"], em=L.number(_f["embed_median"], 3), times=C.num(round(_f["median"] / _f["embed_median"], -2), L.lang),
               one=L.number(C.dollars(_c["prompt_tokens"], _c["output_tokens"]), 6), all=L.number(_f["dollars"], 4)))
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    _r = _f["runs"][C.THRESHOLD]
    _low = _f["runs"][0.75]
    _i, _j, _s = _r["other"][1]
    mo.md(r"""
    ### What the course measured

    **Exact:** among the {n} test questions an exact cache finds no hit, because banking77-small holds each text once: the course left out repeats when it made it. In an app the same text does come again: a button with a fixed question, a customer who sends a question twice, the eval set run again on the same version. There an exact cache answers at once and never with the answer to another question.

    **By meaning:** from a similarity of {t} on, the course's semantic cache answered {h} of the {n} questions with an earlier answer. {saved} of the {total} seconds of the model and {tok} of the {alltok} tokens were not needed. {o} of the {h} got the answer to a question about another topic: *{qi}* got the answer to *{qj}*, a question about a blocked PIN, at a similarity of {s}.

    The eval set passed {cp} of the {h} answers from the cache, and {fp} of the model's own answers to those same questions. The eval set checks the form of an answer, its topic, its steps and its length. An answer from the cache on the same topic passes as it did for its own question: the eval set cannot see whether it fits this question. A lower threshold reuses more, and worse: from {lt} on the cache answered {lh} questions, and the eval set passed {lcp} of its answers against {lfp} of the model's own.
    """.format(n=_f["n"], t=L.number(C.THRESHOLD, 2), h=len(_r["hits"]), saved=L.number(_r["seconds"], 1), total=L.number(_f["total"], 1),
               tok=C.num(_r["prompt_tokens"] + _r["output_tokens"], L.lang), alltok=C.num(_f["prompt_tokens"] + _f["output_tokens"], L.lang),
               o=len(_r["other"]), qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2), cp=_r["cached_pass"], fp=_r["fresh_pass"],
               lt=L.number(0.75, 2), lh=len(_low["hits"]), lcp=_low["cached_pass"], lfp=_low["fresh_pass"]))
    return


@app.cell(hide_code=True)
def picture_measured():
    mo.vstack([
        mo.Html(C.threshold_picture(C.THRESHOLD, L.lang)),
        L.watch_out("A cache gives the answer it stored, also when the facts have changed since. Give answers about what changes a time after which they expire, and keep answers about one customer out of the cache."),
    ])
    return


@app.cell(hide_code=True)
def text_p95():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The slow end

        `p95(seconds)` should give the time that 95 of 100 calls stay under. Sort the times from short to long and take the one 95 percent of the way: for 120 times that is place 114, counted from 0, so 5 times are slower. The start code already sorts them into `in_order`, and gives the shortest. The picture under the check uses your function on `SECONDS`, the calls to the model, and on `EMBED_SECONDS`, the questions embedded for the cache.
        """),
        L.task("Give the time 95 percent of the way through `in_order`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p95():
    box_p95 = L.code_box("p95", lines=5)
    box_p95
    return (box_p95,)


@app.cell(hide_code=True)
def check_p95(box_p95):
    result_p95 = L.check("p95", box_p95.value)
    result_p95
    return (result_p95,)


@app.cell(hide_code=True)
def hints_p95():
    L.hints(
        "p95",
        [
            "The median is the time halfway through the sorted list; p95 is the time 95 percent of the way.",
            "The place is a share of `len(seconds)`, made a whole number with `int`.",
            "Make the last line `return in_order[int(len(seconds) * ___)]`.",
        ],
        why="""```python
def p95(seconds):
    in_order = sorted(seconds)
    return in_order[int(len(seconds) * 0.95)]
```

The model's p95 is about 2 seconds above its median; embedding a question for the cache takes about two hundredths of a second at p95. A hit is not a little faster: it skips the slow part.""",
    )
    return


@app.cell(hide_code=True)
def text_exact():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The same question again

        `exact_hits(questions)` should count the questions whose text came before, as an exact cache sees the text: in small letters, without spaces at the ends. The start code makes that `key` for each question and stores it in `seen`, a set, but it never looks one up. `x in a_set` is `True` when `x` is in the set. The picture runs your function on three streams of questions.
        """),
        L.task("Count a hit when the key is already in `seen`, before you store it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_exact():
    box_exact = L.code_box("exact_hits", lines=9)
    box_exact
    return (box_exact,)


@app.cell(hide_code=True)
def check_exact(box_exact):
    result_exact = L.check("exact_hits", box_exact.value)
    result_exact
    return (result_exact,)


@app.cell(hide_code=True)
def hints_exact():
    L.hints(
        "exact_hits",
        [
            "Look up first, then store: the other way round, every question finds itself.",
            "Put an `if` before `seen.add(key)`.",
            "Write `if key in seen:`, with `hits = ___` one step further in.",
        ],
        why="""```python
def exact_hits(questions):
    seen = set()
    hits = 0
    for question in questions:
        key = question.lower().strip()
        if key in seen:
            hits = hits + 1
        seen.add(key)
    return hits
```

The second time the same 120 come in, every one is a hit. The first two bars stay at 0 because the course left repeats out of banking77-small; where the same text does come again, such as a button with a fixed question, every repeat is a hit.""",
    )
    return


@app.cell(hide_code=True)
def text_closest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The closest earlier question

        `closest(vector, vectors)` should give the place of the row of `vectors` that is most like `vector`, with its similarity. The rows have length 1, so their dot products with `vector` are the similarities; the start code already has them in `scores`, and gives place 0. The picture runs the course's semantic cache over the 120 questions with your function, from a similarity of {t} on.
        """.format(t=L.number(C.THRESHOLD, 2))),
        L.task("Take the place of the highest score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closest():
    box_closest = L.code_box("closest", lines=6)
    box_closest
    return (box_closest,)


@app.cell(hide_code=True)
def check_closest(box_closest):
    result_closest = L.check("closest", box_closest.value)
    result_closest
    return (result_closest,)


@app.cell(hide_code=True)
def hints_closest():
    L.hints(
        "closest",
        [
            "Every score is the similarity of one question in the cache; the closest has the highest.",
            "NumPy has a function that gives the place of the largest number in an array; make its result a Python `int`.",
            "Make the line `best = int(___)`.",
        ],
        why="""```python
def closest(vector, vectors):
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return best, float(scores[best])
```

The closest question is not always the same question: a customer who cannot find their new card and a customer who lost theirs write almost the same words.""",
    )
    return


@app.cell(hide_code=True)
def text_stricter():
    mo.md(r"""
    ### Exercise 4. A stricter cache
    """)
    return


@app.cell(hide_code=True)
def choice_stricter_cell():
    _f = C.findings()
    _r, _hi = _f["runs"][C.THRESHOLD], _f["runs"][0.9]
    _i, _j, _s = _hi["other"][0]
    choice_stricter = L.choice(
        "stricter",
        question="The course's cache gives an earlier answer from a similarity of {t} on. What changes from 0.9 on?".format(t=L.number(C.THRESHOLD, 2)),
        options={
            "fewer": "Fewer hits, and still one for another topic",
            "none": "Fewer hits, and none of them for another topic",
            "more": "More hits, because a stricter cache is more sure",
        },
        correct="fewer",
        why={
            "fewer": "From 0.9 on the cache answered {h} questions instead of {h0}, so the model answered {d} more. One still got the answer for another topic: *{qi}* got the answer to *{qj}*, at a similarity of {s}. Close in words is not always the same topic.".format(
                h=len(_hi["hits"]), h0=len(_r["hits"]), d=len(_r["hits"]) - len(_hi["hits"]), qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2)),
            "none": "One still did: *{qi}* got the answer to *{qj}*, at a similarity of {s}. A higher threshold lets fewer wrong answers through, not none.".format(
                qi=QUESTIONS[_i], qj=QUESTIONS[_j], s=L.number(_s, 2)),
            "more": "A stricter cache reuses less: from 0.9 on it answered {h} questions instead of {h0}.".format(h=len(_hi["hits"]), h0=len(_r["hits"])),
        },
    )
    choice_stricter
    return (choice_stricter,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try other thresholds

    Move the threshold and watch what the cache saves, how many questions get the answer for another topic, and what the eval set says of the answers it gave.
    """)
    return


@app.cell(hide_code=True)
def settings_threshold():
    threshold_slider = mo.ui.slider(0.7, 0.98, step=0.01, value=C.THRESHOLD, label="Threshold", show_value=True)
    threshold_slider
    return (threshold_slider,)


@app.cell(hide_code=True)
def demo_threshold(threshold_slider):
    mo.Html(C.threshold_picture(threshold_slider.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The seconds and tokens of each call are what a trace of [Seeing what happened](/course/open/build-tracing?lang=en) writes down for each span. A cache is a change like any other: measure it with the eval set of [An eval set and regression tests](/course/open/build-eval-set?lang=en) before customers get it, and with users as in [Offline, online and what it is worth](/course/open/build-eval-online?lang=en). [Shipping an LLM app](/course/open/build-ship-app?lang=en) puts the parts together: settings in a file, logs, a version of every prompt, and a way back.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stricter, result_closest, result_exact, result_p95):
    L.wrap_up(result_p95, result_exact, result_closest, choice_stricter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
