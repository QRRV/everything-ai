# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-rag-eval. Route: course/route/lessen/bouwen.yaml. Checks, the recorded judgments and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-rag-eval", "en")
    C = L.checks   # the course's RAG, its recorded answers and Qwen3's judgments of them, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Answers from the sources](/course/open/build-grounded-answers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="When does supported, the check of that lesson, give True for an answer?",
        options={
            "right": "When it holds the number the question asks for, taken from the right point",
            "cited": "When it names sources, and every number in it is in a source it names",
            "words": "When most of its words come from the sources it names",
        },
        correct="cited",
        why={
            "cited": "It checks only that: at least one source named, only sources that are there, and no number that is not in them. Whether a number answers the question, it does not see.",
            "right": "The check does not know the right answer: it looks only at the sources the answer names, and whether every number of the answer is in them. See [Answers from the sources](/course/open/build-grounded-answers?lang=en).",
            "words": "It does not count words: it looks at the sources the answer names, and whether every number of the answer is in them. See [Answers from the sources](/course/open/build-grounded-answers?lang=en).",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A RAG gives a wrong answer. That tells you something is wrong, but not what to fix. Maybe the search did not bring back the point that answers the question; then no prompt will help. Maybe the point was there and the answer took something else from it; then a better search changes nothing. A RAG has two parts, and each needs its own measurement.

    **An everyday example.** A dish tastes wrong. Before you change the recipe, you taste what went into it: if the tomatoes were sour, cooking them differently will not help. If they were fine, the fault is in the cooking.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **context recall** | the share of the relevant points that the search brought back; here each question has one right point, so a question scores 1 or 0 |
    | **context precision** | how high the relevant points came among those the search brought back: at each relevant point, the share of relevant points up to that place, averaged; one right point at place r gives 1/r |
    | **groundedness** | whether the sources say what the answer says; a grounded answer can still be wrong, when its sources are wrong |
    | **answer relevance** | whether the answer addresses the question, right or not |

    The first two measure the search. They need to know which points are relevant, so you need a set of questions with their right points, as the course has for its 24 questions. The other two measure the answer. Code can score them, as `supported` of [Answers from the sources](/course/open/build-grounded-answers?lang=en) does for groundedness, or a model can: an LLM-as-a-judge, asked whether the sources say what the answer says, and whether it answers the question. A later lesson goes deeper into judges.

    Ragas, a library for measuring a RAG, computes these four, most of them with an LLM as the judge. It calls groundedness faithfulness; this course keeps that word for whether the written steps of a model are the real reason for its answer, as in [How an LLM reasons](/course/open/llm-reasoning-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: measure a RAG

    1. **Collect:** questions with the points that answer them, and how to check an answer.
    2. **Search:** context recall and context precision for each question, averaged over the questions.
    3. **Answer:** for each answer, whether it is right, grounded and relevant.
    4. **Split:** a wrong answer with the right point among its sources is a fault of the answer step; one without it, of the search.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _q = C.question(C.WORKED)
    _found = C.found(_q)
    _a = C.answers(C.WORKED)[0]
    mo.md(r"""
    ### Worked example

    1. **Collect:** *{question}* The right point: *{gold}*
    2. **Search:** search by meaning brought back a point of *{t1}*, one of *{t2}*, and the right point third. Context recall: 1 of 1 relevant points came back, so 1. Context precision: at the right point, place 3, 1 of 3 points is relevant: 1/3 = {p}. The reranker of [A second look: reranking](/course/open/build-reranking?lang=en) puts the right point first: precision 1, recall still 1.
    3. **Answer:** Qwen3 wrote *{answer}* The course's check, which knows the right number, says it is right; `supported` gives `True`, because {n1} and {n2} are in source 3. Qwen3 as a judge gave it `P(yes)` {g} for groundedness and {r} for relevance.
    4. **Split:** nothing to split: the search found the point, and the answer used it.
    """.format(question=_q["text"]["en"], question_nl=_q["text"]["nl"], gold=C.gold(_q), t1=C.title_of(_found[0], L.lang), t2=C.title_of(_found[1], L.lang),
               p=C.number(1 / 3, L.lang, 2), answer=_a, n1=C.number(C.worked_numbers()[0], L.lang), n2=C.number(C.worked_numbers()[1], L.lang),
               g=C.number(C.p_yes(C.WORKED, 0, "grounded"), L.lang, 3), r=C.number(C.p_yes(C.WORKED, 0, "relevant"), L.lang, 3)))
    return


@app.cell(hide_code=True)
def text_measured():
    _m, _r = C.scores("meaning"), C.scores("reranked")
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    The course measured the RAG of [Answers from the sources](/course/open/build-grounded-answers?lang=en): search by meaning gives three points, and Qwen3 answers with them after a grounded system prompt, twice for each of the {n} questions a point answers. Averaged over the questions, the search had a context recall of {rm} and a context precision of {pm}; with the reranker instead, {rr} and {pr}. Then the {answers} answers:
    """.format(n=len(C.ANSWERABLE), rm=C.number(_m["recall"], L.lang, 2), pm=C.number(_m["precision"], L.lang, 2),
               rr=C.number(_r["recall"], L.lang, 2), pr=C.number(_r["precision"], L.lang, 2), answers=_f["answers"]))
    return


@app.cell(hide_code=True)
def picture_answers():
    mo.Html(C.answers_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_split():
    _f = C.findings()
    mo.md(r"""
    {right} of the {answers} answers were right. Of the {wrong} wrong ones, {ws} had no right point among their sources: those the search lost. The other {wa} had it, and still went wrong: those are for the answer step. The {mr} right answers without the right point took their number from another point that says it too. The course knows one right point per question, so context recall counts that search as a miss.

    `supported` gave `True` for {supported} of the {answers} answers, and all {wsup} wrong ones were among them: it only sees whether the numbers of an answer are in its sources. Qwen3 as a judge, asked in the form of the reranker for one token with logprobs, wrote yes for all {judged} judgments. Its `P(yes)`:
    """.format(right=_f["right"], answers=_f["answers"], wrong=_f["wrong"], ws=_f["wrong_search"], wa=_f["wrong_answer"], mr=_f["miss_right"],
               supported=_f["supported"], wsup=_f["wrong_supported"], judged=_f["judged"]))
    return


@app.cell(hide_code=True)
def picture_judge():
    mo.Html(C.judge_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_judge():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        For groundedness, every answer got at least {gl}. For a right and a wrong answer, the right one got the higher score in {gp}% of the pairs: about as often as a coin toss. For relevance it was {rp}%, and the lowest score went to an answer that said only *{low}*
        """.format(gl=C.number(_f["grounded_low"], L.lang, 2), gp=round(100 * _f["grounded_pairs"]), rp=round(100 * _f["relevant_pairs"]), low=_f["relevant_lowest"])),
        L.watch_out("A judge is a measurement too. Before you trust its scores, compare them with answers whose verdict you know, as the course did here. Ragas scores with an LLM as well: its numbers are as good as its judge."),
    ])
    return


@app.cell(hide_code=True)
def text_recall():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Context recall

        `context_recall(retrieved, relevant)` gets the points the search brought back and the relevant points, and should give the share of the relevant points that came back. `FOUND["q01"]` are the three points search by meaning found for *{q}*, and `RELEVANT["q01"]` a list with its right point. The search missed it there, so `context_recall` should give `0.0`. The check also tries other questions, and lists of letters with more than one relevant point.
        """.format(q=C.question("q01")["text"]["en"])),
        L.task("Count a point of `relevant` only when it is in `retrieved`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recall():
    box_recall = L.code_box("context_recall", lines=7)
    box_recall
    return (box_recall,)


@app.cell(hide_code=True)
def check_recall(box_recall):
    result_recall = L.check("context_recall", box_recall.value)
    result_recall
    return (result_recall,)


@app.cell(hide_code=True)
def hints_recall():
    L.hints(
        "context_recall",
        [
            "Context recall looks from the side of the relevant points: which of them are among the points that came back?",
            "`in` tells whether a list holds something: `x in items` is `True` when `x` is one of the items. Count only the points that came back.",
            "Put the line `found = found + 1` under an `if ___:`, one step further in.",
        ],
        why="""```python
def context_recall(retrieved, relevant):
    found = 0
    for point in relevant:
        if point in retrieved:
            found = found + 1
    return found / len(relevant)
```

Context recall says whether the answer step had a chance: a point that did not come back cannot end up in the answer.""",
    )
    return


@app.cell(hide_code=True)
def text_precision():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Context precision

        `context_precision(retrieved, relevant)` should say how high the relevant points came. Go through `retrieved` in order. At each relevant point, take the precision up to that place: the number of relevant points so far, divided by the place. Give the average of those, or `0.0` when no relevant point came back. In `FOUND["q05"]` the right point is second: 1/2 = 0.5. With the relevant points `["c", "a"]` in `["a", "b", "c"]`: (1/1 + 2/3) / 2 = {two}. The check also tries a question whose search missed, and other lists.
        """.format(two=C.number((1 + 2 / 3) / 2, L.lang, 2))),
        L.task("At each relevant point, add the precision up to that place to `total`; at the end, give the average over the relevant points, or `0.0` when none came back."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("context_precision", lines=11)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("context_precision", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "context_precision",
        [
            "The place of `retrieved[i]` is `i + 1`. At a relevant point, `hits` relevant points have come back up to that place.",
            "Inside the `if`, after `hits = hits + 1`: `total = total + hits / (i + 1)`.",
            "Replace the last line by an `if hits == 0:` with `return 0.0` under it, and then `return total / ___`.",
        ],
        why="""```python
def context_precision(retrieved, relevant):
    total = 0
    hits = 0
    for i in range(len(retrieved)):
        if retrieved[i] in relevant:
            hits = hits + 1
            total = total + hits / (i + 1)
    if hits == 0:
        return 0.0
    return total / hits
```

With one right point at place r, context precision is 1/r. The reranker lifted the right point of the worked example from place 3 to place 1: its precision went from {third} to 1, and its recall stayed 1.""".format(third=C.number(1 / 3, L.lang, 2)),
    )
    return


@app.cell(hide_code=True)
def text_diagnose():
    _qid, _s = C.DIAGNOSED
    _q = C.question(_qid)
    mo.md(r"""
    ### Exercise 3. Which part to fix?

    *{question}* The search put the right point first: *{gold}* Qwen3 answered: *{answer}* `supported` gave `True` for it.
    """.format(question=_q["text"]["en"], question_nl=_q["text"]["nl"], gold=C.gold(_q), answer=C.answers(_qid)[_s]))
    return


@app.cell(hide_code=True)
def choice_diagnose_cell():
    _f = C.findings()
    choice_diagnose = L.choice(
        "diagnose",
        question="What do you fix for this question?",
        options={
            "answer": "The answer step: it took the wrong number",
            "search": "The search: it should have put a better point first",
            "check": "Nothing: supported gave True, so the answer is grounded",
        },
        correct="answer",
        why={
            "answer": "Context recall and context precision were both 1 here: the search did its part. The 14 is in the point too, but as the pixels a closing leaves wrong. Of the {wrong} wrong answers, {wa} had the right point among their sources, like this one.".format(
                wrong=_f["wrong"], wa=_f["wrong_answer"]),
            "search": "The search put the right point first: context recall 1, context precision 1. The answer step took the wrong number from it: 14 is the number of pixels a closing leaves wrong.",
            "check": "`supported` only sees that every number of the answer is in a source it names, and 14 is in it, as the pixels a closing leaves wrong. It gave `True` for all {wsup} wrong answers.".format(
                wsup=_f["wrong_supported"]),
        },
    )
    choice_diagnose
    return (choice_diagnose,)


@app.cell(hide_code=True)
def text_echo():
    _qid, _s = C.ECHO
    mo.md(r"""
    ### Exercise 4. A judge that says yes

    *{question}* One answer of the course's RAG was: *{answer}* It only repeats the question. Qwen3 as a judge gave it `P(yes)` {r} for "does the answer answer the question?"
    """.format(question=C.question(_qid)["text"]["en"], question_nl=C.question(_qid)["text"]["nl"], answer=C.answers(_qid)[_s], r=C.number(C.p_yes(_qid, _s, "relevant"), L.lang, 3)))
    return


@app.cell(hide_code=True)
def choice_judge_cell():
    _f = C.findings()
    choice_judge = L.choice(
        "judge",
        question="What does that tell you?",
        options={
            "sure": "The answers are good: the judge is sure of nearly all of them",
            "words": "Ask the judge for a written yes or no instead of the probability",
            "measure": "A judge has to be measured too, on answers you have checked",
        },
        correct="measure",
        why={
            "measure": "On the {n} answers the course checked, a right answer got a higher score than a wrong one in {gp}% of the pairs for groundedness and {rp}% for relevance. Check a judge against answers whose verdict you know before you trust it; a larger model may judge better, but that too you measure.".format(
                n=_f["answers"], gp=round(100 * _f["grounded_pairs"]), rp=round(100 * _f["relevant_pairs"])),
            "sure": "The judge was just as sure of the wrong answers: for groundedness every answer got at least {gl}, and a right answer beat a wrong one in only {gp}% of the pairs.".format(
                gl=C.number(_f["grounded_low"], L.lang, 2), gp=round(100 * _f["grounded_pairs"])),
            "words": "Qwen3 wrote yes for all {judged} judgments: the written word says even less than the probability.".format(judged=_f["judged"]),
        },
    )
    choice_judge
    return (choice_judge,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Measure the RAG yourself

    Once exercises 1 and 2 are right, pick a question. You see the three points of both searches with your context recall and precision, and the two answers of the course's RAG with what the checks and the judge said.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q20")["text"][L.lang], label="Question", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_measure(box_precision, box_recall, question_pick):
    _fns = C.learner_metrics(box_recall.value, box_precision.value)
    if _fns is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the numbers of the search are those of your own `context_recall` and `context_precision`.*")
    else:
        _out = mo.Html(C.looked_into(_fns, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** With these numbers you can weigh each change to a RAG, such as another search, the reranker of [A second look: reranking](/course/open/build-reranking?lang=en) or another prompt, and see which part it improves. `supported` of [Answers from the sources](/course/open/build-grounded-answers?lang=en) is the cheapest judge; a model as a judge can look at more than numbers, once you have measured it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_diagnose, choice_judge, result_precision, result_recall):
    L.wrap_up(result_recall, result_precision, choice_diagnose, choice_judge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
