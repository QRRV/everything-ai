# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-reranking. Route: course/route/lessen/bouwen.yaml. Checks, the recorded judgments and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-reranking", "en")
    C = L.checks   # the course's points, the hybrid search, Qwen3's recorded judgments, the checks and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Words and meaning together](/course/open/build-hybrid-search?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rrf():
    quick_rrf = L.choice(
        "quick-rrf",
        question="Reciprocal rank fusion with k = 60 merges two rankings. Point A is first in one and tenth in the other; point B is third in both. Which gets the higher score?",
        options={
            "a": "A, because a first place counts for the most",
            "b": "B, because two third places add up to more",
            "same": "Neither: both have an average place of three or so",
        },
        correct="b",
        why={
            "b": "A gets 1/61 + 1/70 = {a}, B gets 1/63 + 1/63 = {b}. With k = 60, a first place is worth little more than a third, so being high in both rankings wins.".format(
                a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
            "a": "With k = 60, a first place gives 1/61 = {p1}, a third 1/63 = {p3}: hardly more. A gets {a} in all, B gets {b}. See [Words and meaning together](/course/open/build-hybrid-search?lang=en).".format(
                p1=C.number(1 / 61, L.lang, 4), p3=C.number(1 / 63, L.lang, 4), a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
            "same": "RRF does not average the places: each ranking gives `1 / (k + rank)`, and the parts add up. A gets {a}, B gets {b}. See [Words and meaning together](/course/open/build-hybrid-search?lang=en).".format(
                a=C.number(1 / 61 + 1 / 70, L.lang, 4), b=C.number(2 / 63, L.lang, 4)),
        },
    )
    quick_rrf
    return (quick_rrf,)


@app.cell(hide_code=True)
def text_why():
    _f = C.findings()
    _w = C.worked()
    mo.md(r"""
    ## Why this matters

    In [Words and meaning together](/course/open/build-hybrid-search?lang=en), the merged search put the right point among the three best for {top3} of the {n} questions. *{question}* There the right point came at place {before}. A search by meaning makes the embedding of the question and that of every point apart from each other, so it compares two summaries of {dims} numbers. It never reads the question and the point together.

    **An everyday example.** A recruiter sorts a pile of 500 letters in an hour by a few words: a degree, a year, a city. The 20 best then go to someone who reads each one in full, with the job in mind, and orders them again. Slower per letter, but only 20 letters.
    """.format(top3=_f["hybrid_top3"], n=_f["questions"], question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], before=_w["before"], dims=C.DIMS))
    return


@app.cell(hide_code=True)
def text_idea():
    _f = C.findings()
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **reranker** | a second, slower model that reads the question together with each candidate and gives it a score, to put the candidates in a better order |
    | **candidates** | the texts the fast search gives the reranker: here its {cands} best |
    | **bi-encoder** | a model that makes an embedding of each text apart, as Granite does for search by meaning: points can be embedded once, ahead of time |
    | **cross-encoder** | a model that reads two texts together and gives one score for the pair: more precise, but it has to run again for every pair |
    | **logprob** | the natural logarithm of the probability of a token; an API with `"logprobs": true` gives it for every token written, and with `"top_logprobs": 5` also for the five most likely tokens at that place |
    | `P(yes)` | the probability that the model's answer starts with yes: `math.exp` of the logprob gives it back |

    A reranker is a cross-encoder. Any chat model can be one: ask it whether the passage answers the question, let it write one token, and take the probability of yes as the score. The written token alone is not enough: Qwen3 wrote yes for all {judged} candidates the course gave it. The probability says how sure it was, and that orders the candidates.

    Models made for this work the same way. Qwen3-Reranker is a Qwen3 trained on exactly the prompt this lesson uses, and its score is the probability of yes. Cohere Rerank, bge-reranker and Jina Reranker are others, often behind an API that takes the question and the candidates and gives back a score for each.
    """.format(cands=C.CANDIDATES, judged=_f["judged"]))
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: reranking

    1. **Search:** the fast search gives the 20 best candidates.
    2. **Judge:** for each candidate, ask the model for one token about the question and the candidate, with logprobs, and take `P(yes)` from the most likely tokens.
    3. **Sort:** order the candidates by `P(yes)`, the highest first, and keep the best few for the answer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    _w = C.worked()
    mo.vstack([
        mo.md(r"""
        ### Worked example

        *{question}*

        1. **Search:** the hybrid search gives 20 candidates. First is a point of *{t0}*; the right point is at place {before}: *{gold}*
        2. **Judge:** for each candidate the course sends this request to its server, with the candidate after `<Document>:`.
        """.format(question=_w["question"]["text"]["en"], question_nl=_w["question"]["text"]["nl"], t0=C.title_of(_w["candidates"][0], L.lang), before=_w["before"], gold=_w["gold"])),
        mo.md(C.request_code(_w["question"]["text"]["en"])),
        mo.md(r"""
        `"max_tokens": 1` lets the model write one token, and `"temperature": 0` makes it take the most likely one, so a judgment is the same on every run. For the right point, the five most likely first tokens were {tokens}. `P(yes)` is the sum of the probabilities of the tokens that are yes: {sum} = {p}. The first candidate of the search got {p0}.

        3. **Sort:** the right point goes to place {after}, with {p}. Second is a point of *{t1}*, with {p1}.
        """.format(tokens=C.tokens_text(_w["gold_response"], L.lang), sum=C.sum_text(_w["gold_response"], L.lang), p=C.number(_w["gold_p"], L.lang, 3),
                   p0=C.number(_w["first_p"], L.lang, 3), after=_w["after"], t1=C.title_of(_w["reranked"][1], L.lang), p1=C.number(_w["second_p"], L.lang, 3))),
    ])
    return


@app.cell(hide_code=True)
def text_measured():
    _f = C.findings()
    mo.md(r"""
    ### What the course measured

    For each of the {n} questions a point answers, the course gave Qwen3 the {cands} candidates of the hybrid search, one request each, and sorted them by `P(yes)`. Where did the right point end up?
    """.format(n=_f["questions"], cands=C.CANDIDATES))
    return


@app.cell(hide_code=True)
def picture_places():
    mo.Html(C.places_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_runs():
    _f = C.findings()
    mo.vstack([
        mo.md(r"""
        Among the three best: {before3} questions after the hybrid search, {after3} after reranking. In first place: {before1} before, {after1} after. The reranker lifted the right point of {rescued} questions from far down into the three best, and dropped {lost} out of them: *{lost_q}* There the search had the right point first, and the reranker put it at place {lost_place}.

        It took time. One judgment took {one} seconds on the course's computer, so 20 candidates took about {twenty} seconds per question, against a search that does not call the model at all. For each of the {none} questions no point answers, the best candidate still got a `P(yes)` of at least {none_p}: the score orders the candidates, it does not say whether any of them answers.
        """.format(before3=_f["hybrid_top3"], after3=_f["reranked_top3"], before1=_f["hybrid_first"], after1=_f["reranked_first"],
                   rescued=len(_f["rescued"]), lost=len(_f["lost"]),
                   lost_q=C.question(_f["lost"][0])["text"]["en"], lost_q_nl=C.question(_f["lost"][0])["text"]["nl"], lost_place=_f["lost_place"], one=C.number(_f["judgment"], L.lang, 2),
                   twenty=C.number(_f["seconds"], L.lang, 0), none=len(C.NONE), none_p=C.number(_f["none_p"], L.lang, 2))),
        L.watch_out("A reranker can also make an order worse, as for the question above; Qwen3 is a small model that was not made for this. Measure it on your own questions, and weigh the gain against the seconds it costs per question."),
    ])
    return


@app.cell(hide_code=True)
def text_p_yes():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The probability of yes

        `p_yes(response)` gets the JSON of an answer of the server, as a dict, and should give the probability that the answer starts with yes. `response["choices"][0]["logprobs"]["content"][0]` is the first token, and its `"top_logprobs"` a list of the five most likely tokens, each a dict with `"token"` and `"logprob"`. Count every token that is yes, whatever its capitals and the spaces around it: `"yes"`, `"Yes"` and `" yes"` all count. For the right point of the worked example, `p_yes` should give {p} (rounded). The check also shows what your function gives the 20 candidates of that question.
        """.format(p=C.number(C.worked()["gold_p"], L.lang, 3))),
        L.task("Add the probability of each token in `top_logprobs` that is yes: `math.exp` of its logprob."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p_yes():
    box_p_yes = L.code_box("p_yes", lines=10)
    box_p_yes
    return (box_p_yes,)


@app.cell(hide_code=True)
def check_p_yes(box_p_yes):
    result_p_yes = L.check("p_yes", box_p_yes.value)
    result_p_yes
    return (result_p_yes,)


@app.cell(hide_code=True)
def hints_p_yes():
    L.hints(
        "p_yes",
        [
            "A logprob is the logarithm of a probability, so `math.exp` of it gives the probability back. Add those of the tokens that mean yes.",
            "In the loop, `token.strip().lower()` turns `\" Yes\"` into `\"yes\"`. Only when it is `\"yes\"` does the token count.",
            "Inside the loop, after `token = ...`: `if token.strip().lower() == \"yes\":`, and below it `total = total + math.exp(entry[\"logprob\"])`.",
        ],
        why="""```python
import math


def p_yes(response):
    first = response["choices"][0]["logprobs"]["content"][0]
    total = 0.0
    for entry in first["top_logprobs"]:
        token = entry["token"]
        if token.strip().lower() == "yes":
            total = total + math.exp(entry["logprob"])
    return total
```

The model has more than one way to write yes, and each has its own probability; together they are the probability of yes. A token that is not among the five most likely adds almost nothing.""",
    )
    return


@app.cell(hide_code=True)
def text_rerank():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Rerank

        `rerank(question, candidates, n)` should give the `n` candidates with the highest `P(yes)`, the highest first. Given are:

        - `judge(question, passage)`: the server's answer for that question and passage, as a dict, from the course's recorded judgments;
        - `p_yes(response)`: the one of exercise 1.

        The box asks the judge for every candidate, and gives the first `n` in the order of the search. For the question of the worked example, `rerank` should give the right point first. The check also tries other questions, and other `n`.
        """),
        L.task("Keep a pair `(p_yes(response), passage)` for every candidate, sort the pairs with the highest first, and give the passages of the first `n`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rerank():
    box_rerank = L.code_box("rerank", lines=7)
    box_rerank
    return (box_rerank,)


@app.cell(hide_code=True)
def check_rerank(box_rerank):
    result_rerank = L.check("rerank", box_rerank.value)
    result_rerank
    return (result_rerank,)


@app.cell(hide_code=True)
def hints_rerank():
    L.hints(
        "rerank",
        [
            "Each candidate gets a score from `p_yes` of its answer. Sorting pairs of score and passage puts them in order of the score.",
            "In the loop, add `scored.append((p_yes(response), passage))`. After the loop, `sorted(scored, reverse=True)` has the highest score first.",
            "Replace the last line by two: `best = sorted(scored, reverse=True)`, then return the passages of `best[:n]`: `[passage for score, passage in best[:n]]`.",
        ],
        why="""```python
def rerank(question, candidates, n):
    scored = []
    for passage in candidates:
        response = judge(question, passage)
        scored.append((p_yes(response), passage))
    best = sorted(scored, reverse=True)
    return [passage for score, passage in best[:n]]
```

The reranker never sees the order of the search: it judges every candidate on its own, and only its scores decide.""",
    )
    return


@app.cell(hide_code=True)
def text_all():
    mo.md(r"""
    ### Exercise 3. Why not all points?

    The reranker read the right point of the worked example better than the search did. A colleague wants to skip the search and let Qwen3 judge every one of the {points} points for each question.
    """.format(points=C.number(len(C.POINTS), L.lang)))
    return


@app.cell(hide_code=True)
def choice_all_cell():
    _f = C.findings()
    _all = len(C.POINTS) * _f["judgment"] / 60
    choice_all = L.choice(
        "all-points",
        question="What is the main problem with that?",
        options={
            "worse": "The reranker gets worse when it has more points to choose from",
            "embed": "The points would first need an embedding of their own",
            "slow": "It would take minutes for every question",
        },
        correct="slow",
        why={
            "slow": "At {one} seconds per judgment, {points} points take about {minutes} minutes for one question. The fast search can rank all points because their embeddings were made once, ahead of time; a cross-encoder has to run again for every pair of question and point.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
            "worse": "Each judgment looks at one point on its own, so the others do not change its score. The problem is time: at {one} seconds per judgment, {points} points take about {minutes} minutes for one question.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
            "embed": "The reranker needs no embeddings: it reads the question and the point as text. The problem is time: at {one} seconds per judgment, {points} points take about {minutes} minutes for one question.".format(
                one=C.number(_f["judgment"], L.lang, 2), points=C.number(len(C.POINTS), L.lang), minutes=C.number(_all, L.lang, 0)),
        },
    )
    choice_all
    return (choice_all,)


@app.cell(hide_code=True)
def text_none():
    _f = C.findings()
    _n = C.none_example()
    mo.md(r"""
    ### Exercise 4. A question no point answers

    *{question}* No point of the course answers it. Still, the reranker gave its best candidate a `P(yes)` of {p}: *{best}*
    """.format(question=_n["question"]["text"]["en"], question_nl=_n["question"]["text"]["nl"], p=C.number(_n["p"], L.lang, 3), best=_n["best"]))
    return


@app.cell(hide_code=True)
def choice_none_cell():
    _f = C.findings()
    choice_none = L.choice(
        "no-answer",
        question="What does that tell you about the reranker's score?",
        options={
            "order": "It orders candidates well, but does not say whether one answers",
            "broken": "The reranker is broken: for such a question every score should be close to 0",
            "true": "That candidate answers the question after all, or the score would be low",
        },
        correct="order",
        why={
            "order": "The course measured it: for each of the {none} questions no point answers, the best candidate got at least {p}. Use the order to pick the best few, and let the answer step say when the sources do not answer, as the grounded prompt of [Answers from the sources](/course/open/build-grounded-answers?lang=en) does.".format(
                none=len(C.NONE), p=C.number(_f["none_p"], L.lang, 2)),
            "broken": "A small model that was not trained to judge says yes easily: it wrote yes for all {judged} candidates. Its scores still put the right point higher than others, so the order is useful, even when every score is high.".format(
                judged=_f["judged"]),
            "true": "No point answers it: the course wrote it as one of its questions without an answer in the course. A high score only means the model was fairly sure of yes. For each of the {none} questions no point answers, the best candidate got at least {p}.".format(
                none=len(C.NONE), p=C.number(_f["none_p"], L.lang, 2)),
        },
    )
    choice_none
    return (choice_none,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Rerank with your own functions

    Once exercises 1 and 2 are right, pick a question. The course reranks its 20 candidates with your `rerank` and your `p_yes`, on Qwen3's recorded judgments, and shows the five best beside those of the hybrid search.
    """)
    return


@app.cell(hide_code=True)
def settings_cell():
    question_pick = mo.ui.dropdown(options=C.question_options(L.lang), value=C.question("q21")["text"][L.lang], label="Question", full_width=True)
    question_pick
    return (question_pick,)


@app.cell(hide_code=True)
def demo_rerank(box_p_yes, box_rerank, question_pick):
    _fn = C.learner_rerank(box_p_yes.value, box_rerank.value)
    if _fn is None:
        _out = mo.md("*Finish exercises 1 and 2 first: the reranking uses your own `p_yes` and `rerank`.*")
    else:
        _out = mo.Html(C.reranked(_fn, question_pick.value, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The best few after reranking go into the prompt, as in [Answers from the sources](/course/open/build-grounded-answers?lang=en). A search that finds the right point among its 20 candidates, but not among its three best, is exactly where a reranker helps; a later lesson measures a RAG in two parts, the search and the answer, so you can see which of the two to improve.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_all, choice_none, result_p_yes, result_rerank):
    L.wrap_up(result_p_yes, result_rerank, choice_all, choice_none)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
