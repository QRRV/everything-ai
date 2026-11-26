# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-answer-engines. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-answer-engines", "en")
    C = L.checks   # the questions of XQuAD, the ten links of BM25, the course's answer engine and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en) and [Summarizing: pick or write](/course/open/nlp-summarize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rank():
    quick_rank = L.choice(
        "quick-rank",
        question="The first relevant document for a query stands at place 4 of the ranking. What is the reciprocal rank of this query?",
        options={
            "place": "`4`: the place itself",
            "first": "`0`: only the first place counts, and the document at place 1 is not relevant",
            "right": "`0.25`: one divided by that place",
            "share": "`0.75`: three of the first four places hold no relevant document",
        },
        correct="right",
        why={
            "right": "The reciprocal rank is one divided by the place of the first relevant document: 1 / 4 = 0.25. It looks only at the first relevant result, like a reader who stops at the first good link. This lesson asks something even simpler of a page of links: is there a relevant document among the first results at all?",
            "place": "The reciprocal rank gets smaller when the first relevant document stands lower. Does 4 get smaller when the place goes up? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
            "first": "The reciprocal rank looks for the first relevant document, wherever it stands. What does it do with its place? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
            "share": "That counts the places without a relevant document. Which place does the reciprocal rank divide one by? See [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en).",
        },
    )
    quick_rank
    return (quick_rank,)


@app.cell(hide_code=True)
def quick_check_extract():
    quick_extract = L.choice(
        "quick-extract",
        question="What does an extractive summary of a text do?",
        options={
            "write": "it writes new sentences in its own words, as a person would",
            "right": "it picks sentences of the text and keeps their words",
            "keywords": "it lists the words that occur most often in the text, without sentences around them",
            "lead": "it always takes the first sentence",
        },
        correct="right",
        why={
            "right": "An extractive summary copies sentences of the text as they are; an abstractive summary writes new ones. The answer engine of this lesson is extractive: it answers with one sentence from a result, so every word of its answer stands in the source it cites.",
            "write": "That is an abstractive summary. What does an extractive summary do with the sentences of the text? See [Summarizing: pick or write](/course/open/nlp-summarize?lang=en).",
            "keywords": "A summary in sentences or a list of words: which one is an extractive summary? See [Summarizing: pick or write](/course/open/nlp-summarize?lang=en).",
            "lead": "The first sentences are a baseline that a summary is compared with, not what makes a summary extractive. Which sentences can an extractive summary pick? See [Summarizing: pick or write](/course/open/nlp-summarize?lang=en).",
        },
    )
    quick_extract
    return (quick_extract,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In this module, a search engine gave a page of results: a ranking by [BM25: the classic ranking](/course/open/ir-bm25?lang=en), measured with [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en), and each result with a snippet from [Snippets: showing the results](/course/open/ir-snippets?lang=en). The reader picks a result and reads it. Donald Metzler, Yi Tay, Dara Bahri and Marc Najork in 2021: "Classical information retrieval systems do not answer information needs directly, but instead provide references to (hopefully authoritative) answers." They proposed a system that answers the question itself, instead of "providing users a selection of potential answers, which induces a rather significant cognitive burden on the user."

    Such systems now exist. Google writes an AI summary above its results, and search engines such as Perplexity give one answer in place of the results, with small numbers that point to the sources. Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Yixin Mao and Chien-Sheng Wu: "Answer engines - LLM-powered search tools that generate source-cited summaries rather than just returning relevant links - represent a particularly significant shift from traditional search engines." This lesson measures what that shift changes: for the reader who gets the answer, and for the sources it comes from.

    **An everyday example.** You ask a passer-by the way and get one answer: the second street on the left. A map shows every street and leaves the choosing to you. The answer is faster. But when the passer-by is wrong, you walk the wrong way, and nothing tells you there was another street.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_clicks = L.guess(
        "guess",
        question="In March 2025, the Pew Research Center followed the searches on Google of 900 adults in the United States: 68,879 searches, 18% of them with an AI summary at the top of the page. The summaries named their sources, as links. How often did people click such a link inside the summary?",
        options={
            "half": "about half the time: the summary names its sources, so people open them to check",
            "eight": "on 8% of the visits",
            "right": "on 1% of the visits",
            "fifteen": "on 15% of the visits",
        },
        reveal="On 1% of the visits to a page with an AI summary. In the words of the Pew Research Center: \"This occurred in just 1% of all visits to pages with such a summary.\" The 8% and the 15% are the links below the summary, the classic results: \"Users who encountered an AI summary clicked on a traditional search result link in 8% of all visits. Those who did not encounter an AI summary clicked on a search result nearly twice as often (15% of visits).\" And after a page with a summary, people ended their browsing session more often: \"This happened on 26% of pages with an AI summary, compared with 16% of pages with only traditional search results.\"",
    )
    guess_clicks
    return (guess_clicks,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Answer engine.** A search engine that answers a question with one text and the sources of that text, instead of a page of links. Most answer engines first search, as BM25 does, then let an LLM read the first results and write the answer from them, with a number after each sentence for the result it took the sentence from: a citation.

    **The course's answer engine.** This lesson builds one without an LLM. It reads the first results of BM25 and answers with one sentence: the sentence with the highest query score of [Snippets: showing the results](/course/open/ir-snippets?lang=en), the first one on a tie, followed by the place of its result in brackets, such as [2]. Its answer is an extractive summary of the results, as in [Summarizing: pick or write](/course/open/nlp-summarize?lang=en): it copies a sentence and writes nothing itself, so every word of the answer stands in the source it cites. Nelson Liu, Tianyi Zhang and Percy Liang found the same for real answer engines: "extractively copied text is almost always fully supported by the source citation". What can go wrong in the course's engine is the choice of the sentence: does it answer the question?

    **success@k.** A page of links works when the reader finds a relevant result on it. success@k measures how often that can happen: the share of the questions with at least one relevant document among the first k results. P@k counts how many of the first k are relevant, and the reciprocal rank looks at the place of the first one; success@k only asks whether there is one. With the own paragraph as the only relevant paragraph, success@1 is the share of the questions whose own paragraph comes first.

    **Does the answer show the answer?** As in [Snippets: showing the results](/course/open/ir-snippets?lang=en), the course knows where the answer of each question of XQuAD stands in its own paragraph. An answer of the course's engine counts as right when its sentence comes from the own paragraph and holds the answer.

    **Ten links against one answer.** The picture shows the question about the population of Jacksonville in 2010, with its ten links. BM25 puts paragraph 1398 first, and the own paragraph, 1379, second. The engine that reads only the first result answers with *Jacksonville is the most populous city in Florida, and the twelfth most populous city in the United States.* It sounds like an answer and holds no number. Reading the first three results, the engine finds the sentence with *1,345,596* in paragraph 1379 and cites [2]. Over all 1,190 questions, the first result is the own paragraph for 80.0% of the questions, and the ten links hold it for 96.3%. The engine that reads one result answers right for 65.5% of the questions, and the engine that reads three for 68.1%; reading more than three hardly changes it, between 68.0% and 68.2%. Of the 380 questions where the answer after three results misses, 336 have their own paragraph among the ten links: 88.4%. A reader of the page could still find the answer a few lines lower; a reader of the one answer sees only the wrong sentence.

    **Clicks.** The guess showed what happens to the links: on visits with an AI summary, people clicked a classic result on 8% of them, against 15% on visits without one, and a link inside the summary on 1%. Lydia Chilton and Jaime Teevan saw this in 2011 already, with the short answers that Bing showed on its result page, such as the meaning of a word, "finding that inline answers can cannibalize clicks from the algorithmic results." For the site that wrote the text, a click that does not happen is a visit it does not get. For the reader, a search without a click is not always a failure. In 2009, Jane Li, Scott Huffman and Akihito Tokuda named the good case: "We define a good abandonment as an abandoned query for which the user’s information need was successfully addressed by the search results page, with no need to click on a result or refine the query." The same missing click can also mean that the reader believed a wrong answer. Narayanan Venkit and colleagues watched 21 people search: "With answer engines, users tend to click on a single source and trust the model’s selection." With an answer engine, people moved the mouse over about 2 sources and opened about 1; with Google, they moved it over about 12 and opened about 4.

    **Citations.** When an answer cites a source, does the source say what the sentence says? Liu, Zhang and Liang measure this in two ways. "Citation recall is the proportion of verification-worthy statements that are fully supported by their associated citations". And: "Citation precision is the proportion of generated citations that support their associated statements". One measure is not enough, because "a response that cites every webpage on the Internet for each generated statement would have high citation recall, but low citation precision". Liu and colleagues also let several citations support a sentence together, and count a citation that supports part of a sentence; this lesson leaves both out: a sentence counts as supported when one of its citations supports it. They checked the answers of Bing Chat, NeevaAI, perplexity.ai and YouChat to 1,450 queries, in 2023: "on average, a mere 51.5% of generated sentences are fully supported by citations and only 74.5% of citations support their associated sentence." And: "We find that citation precision is inversely correlated with perceived utility in existing generative search engines (r = −0.96)." The engines whose answers seemed most useful had the lowest citation precision.

    **Errors without alternatives.** In December 2024, the BBC asked ChatGPT, Copilot, Gemini and Perplexity 100 questions about the news, and 45 journalists reviewed the 362 answers. The BBC: "51% of all AI answers to questions about the news were judged to have significant issues of some form." And: "19% of AI answers which cited BBC content introduced factual errors – incorrect factual statements, numbers and dates." A citation makes such an error more convincing: "when AI assistants cite trusted brands like the BBC as a source, audiences are more likely to trust the answer – even if it is incorrect." Chirag Shah and Emily Bender warned in 2022 that "providing answers to questions rather than pointers for where to look further suggests a finality to the answer." And an answer from several sources hides the range of sources behind it: "by synthesizing results from multiple different sources and thus masking the range that is available, rather than providing a range of sources, the system cuts off the user’s ability to explore that space."
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **answer engine** | a search engine that answers with one text and the sources of that text, instead of a page of links |
    | **citation** | the number after a sentence of an answer that points to the source the sentence comes from |
    | **success@k** | the share of the questions with at least one relevant document among the first k results |
    | **citation recall** | the share of the sentences of an answer that one of their citations supports |
    | **citation precision** | the share of the citations of an answer that support their sentence |
    | **zero-click search** | a search after which the reader opens no result |
    | **good abandonment** | a search without a click because the page of results already gave what was needed |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compare a page of ten links with one answer:

    1. For each question, rank the documents, as BM25 does.
    2. The page: count the questions with a relevant document among the first k results, and divide by the number of questions. That is success@k.
    3. The answer: read the first results, take the sentence with the highest query score, the first one on a tie, and cite the place of its result.
    4. Count the answers whose sentence comes from a relevant document and holds the answer.
    5. For an answer with more sentences, check the citations: which sentences their citations support (citation recall), and which citations support their sentence (citation precision).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.EXAMPLE` is the question about the parliament of Victoria from [Snippets: showing the results](/course/open/ir-snippets?lang=en), and `C.question_place(C.EXAMPLE, L.lang)` gives its number. `C.links(L.lang, k)` gives the ten links of question `k`, the first ten paragraphs of BM25, and `C.relevant_of(L.lang, k)` a list with its own paragraph. `C.success_at_k` is the course's version of exercise 1. `C.engine(L.lang, k, read)` gives the answer of the course's engine after reading the first `read` results, as the place of the result, the paragraph, and the start and end of the sentence in its words; `C.answer_right` says whether that sentence shows the answer, and `C.answer_of` writes the answer out with its citation. Last, `C.success_share` and `C.engine_share` give success@k and the share of right answers over all the questions, for each number in `C.KS`, and `C.misses` how many answers after three results miss, and for how many of those questions the own paragraph is among the ten links:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _links = C.links(L.lang, _k)
    _relevant = C.relevant_of(L.lang, _k)
    print(_links, _relevant)
    for _read in (1, 2, 3):
        _right = C.answer_right(L.lang, _k, _read)
        print(_read, C.success_at_k(_links, _relevant, _read), C.engine(L.lang, _k, _read), _right)
    print(C.answer_of(L.lang, _k, 3))
    for _read in C.KS:
        print(_read, round(C.success_share(L.lang, _read), 3), round(C.engine_share(L.lang, _read), 3))
    print(C.misses(L.lang, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line gives the ten links and the own paragraph, 513, at place 2. With one result, `success_at_k` gives `False`, and the engine answers from paragraph 511 with its first sentence, words 0 up to 23, about the Koori, who lived in the area before European settlement: wrong. Five sentences of paragraph 511 tie at a query score of 0.25, each with one query word, and the first one wins. With two or three results, the page holds the own paragraph, and the engine answers with the first sentence of paragraph 513, words 0 up to 42: it holds *Parliament* and *Victoria*, a query score of 1, and the answer, *Victoria Constitution Act 1855*.

    The next five lines give, for 1, 2, 3, 5 and 10 results, success@k and the share of right answers. success@k climbs from 0.8 to 0.963, and the share of right answers stays at 0.681 at 3, 5 and 10 results: the gap grows from line to line. The last line: after three results, 380 answers miss, and for 336 of those questions the own paragraph is among the ten links.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. success_at_k: is a relevant document on the page?

        Write `success_at_k`, which gets `ranking`, the numbers of the documents best first, as a list, `relevant`, the numbers of the relevant documents, and `k`. It gives back `True` when one of the first `k` documents of `ranking` is in `relevant`, and `False` when none is. In the start code, `for doc in ...` goes through documents, `if doc in relevant` gives back `True` at the first relevant one, and after the loop the function gives back `False`. Replace the `...` with the first `k` documents of `ranking`.

        The check calls `success_at_k([511, 513, 526], [513], 1)`, which gives `False`: the relevant document 513 stands at place 2 of this small example ranking. And `success_at_k([511, 513, 526], [513], 2)`, which gives `True`. Then cases of its own: an empty ranking, a `k` larger than the ranking, a `k` of 0, and the course's rankings.
        """),
        L.task("Let the loop go through the first `k` documents of `ranking`, and no more."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("success_at_k", lines=5)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("success_at_k", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "success_at_k",
        [
            "The loop goes through `ranking`, one `doc` at a time, but only through the first `k` documents: a slice from the start of the list.",
            "A slice `a[:j]` holds the items from the start up to place `j`, counted from 0, without place `j` itself: `a[:2]` holds the items at places 0 and 1, the first two, and a slice longer than the list holds the whole list.",
            "`    for doc in ranking[___]:`",
        ],
        why="`for doc in ranking[:k]:`: the slice holds the first `k` documents, or all of them when the ranking is shorter, so a `k` larger than the ranking works too, and a `k` of 0 gives an empty slice and `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    In the picture, your function answers for the question about Victoria: the own paragraph 513 stands at place 2, so `k` = 1 gives `False`, and `k` = 2 and 3 give `True`. Below, your function over all 1,190 questions: success@1 is 80.0%, and the page of ten links holds the own paragraph for 96.3% of the questions. The first places give the most: from 1 to 3 results, success@k rises by 11.0 points, from 3 to 10 by 5.3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_recall():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. citation_recall: which sentences do their citations support?

        Write `citation_recall`, which gets `cited` and `supports`, two lists with a list for each sentence of an answer. `cited[i]` holds the sources that sentence `i` cites, and the list at place `i` of `supports` holds the sources that support sentence `i`. The function gives back the share of the sentences that one of their citations supports. In the start code, `supported` counts those sentences, and `for i in range(len(cited))` goes through the sentences. For each sentence, `found` starts at `False`, `for source in cited[i]` goes through its citations, and `found` becomes `True` when one of them supports the sentence. A sentence without citations is never supported. Replace the `...` with the test: is `source` one of the sources that support sentence `i`?

        The check calls `citation_recall([[1], [1, 2], [2, 3], []], [[1], [1], [4, 5], [2, 6]])`, which gives `0.5`: the first two sentences cite source 1, which supports them, the third cites sources 2 and 3 while only 4 and 5 support it, and the fourth cites nothing. And `citation_recall([[1, 2, 3, 4]], [[1, 2]])`, which gives `1.0`: one sentence that cites four sources, two of which support it. Then cases of its own: the two examples of Liu, Zhang and Liang, an answer without citations, and a source cited twice.
        """),
        L.task("Test whether `source` is one of the sources that support sentence `i`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recall():
    box_recall = L.code_box("citation_recall", lines=10)
    box_recall
    return (box_recall,)


@app.cell(hide_code=True)
def check_recall(box_recall):
    result_recall = L.check("citation_recall", box_recall.value)
    result_recall
    return (result_recall,)


@app.cell(hide_code=True)
def hints_recall():
    L.hints(
        "citation_recall",
        [
            "The test asks whether `source` is in the list of `supports` that belongs to sentence `i`, the list at place `i`.",
            "`x in a` asks whether `x` is an item of the list `a`: `4 in [4, 5]` gives `True`. And `a[2]` is the item at place 2 of `a`, counted from 0.",
            "`            if source in ___:`",
        ],
        why="`if source in supports[i]:`: the sources that support sentence `i` stand in the list at place `i` of `supports`, at the same place as its citations in `cited`.",
    )
    return


@app.cell(hide_code=True)
def text_after_recall():
    mo.md(r"""
    In the picture, your function measures four answers: the two of the exercise and two examples of Liu, Zhang and Liang. The answer that cites every source for its one sentence gets a citation recall of 1, and so does example (a), although only three of its eight citations support their sentence. In example (c), only the first sentence cites anything, so two of the three sentences are not supported: 1/3. Citation recall looks at the sentences; it cannot see a citation too many.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_precision():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. citation_precision: which citations support their sentence?

        Write `citation_precision`, which gets `cited` and `supports` as in exercise 2, and gives back the share of all the citations of the answer that support their sentence. In the start code, `good` and `total` start at 0, and `for i in range(len(cited))` goes through the sentences. Inside, `for source in cited[i]` goes through the citations of sentence `i`, and `if source in supports[i]` adds 1 to `good` for each citation that supports the sentence. An answer without citations gives `0.0`, so the function never divides by 0. Replace the `...` with the new `total`: the citations so far, plus the citations of sentence `i`.

        The check calls `citation_precision([[1], [1, 2], [2, 3], []], [[1], [1], [4, 5], [2, 6]])`, which gives `0.4`: of the five citations, two support their sentence, source 1 of the first sentence and source 1 of the second. And `citation_precision([[1, 2, 3, 4]], [[1, 2]])`, which gives `0.5`: two of the four citations. Then cases of its own: the two examples of Liu, Zhang and Liang, an answer without citations, and a source cited twice.
        """),
        L.task("Add the number of citations of sentence `i` to `total`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_precision():
    box_precision = L.code_box("citation_precision", lines=10)
    box_precision
    return (box_precision,)


@app.cell(hide_code=True)
def check_precision(box_precision):
    result_precision = L.check("citation_precision", box_precision.value)
    result_precision
    return (result_precision,)


@app.cell(hide_code=True)
def hints_precision():
    L.hints(
        "citation_precision",
        [
            "`total` counts all the citations of the answer: for each sentence, it grows by the number of citations of that sentence, the length of its list in `cited`.",
            "`len(a)` gives the number of items of a list: `len([2, 3])` gives `2`. And `cited[1]` is the list of citations of the second sentence.",
            "`        total = total + ___`",
        ],
        why="`total = total + len(cited[i])`: every citation counts, also the ones that support nothing, so citing more sources raises `total`, and only the sources that support the sentence raise `good`.",
    )
    return


@app.cell(hide_code=True)
def text_after_precision():
    mo.md(r"""
    In the picture, the same four answers, now with your citation precision next to their citation recall. The answer that cites every source scores 1 on citation recall and 0.5 on citation precision; example (a) scores 1 and 0.375, the 3/8 of Liu and colleagues. Example (c) turns it around: citation precision 0.667 and citation recall 0.333, because the citations it gives are mostly right, but two of its three sentences cite nothing. Real answer engines differ in the same way. In the study of Liu and colleagues, Bing Chat had the highest citation precision, 89.5, with a citation recall of 58.7; perplexity.ai had the highest citation recall, 68.7, with a citation precision of 72.7; and YouChat reached a citation recall of only 11.1.

    ### Try it: one answer or ten links

    Pick a question and how many results the course's answer engine reads. The picture shows the answer with its citation, whether its sentence comes from the own paragraph and holds the answer, and the ten links below it, with the own paragraph marked. The table gives, over all the questions, success@k and the share of right answers for the number of results you chose, and for ten.

    Start with the question about Victoria and move the slider from 1 to 3: from which number on does the answer show the answer? Then pick the question about the teams Los Angeles used to have, and move the slider from 1 to 10. Where is the own paragraph among the links, and does the answer ever change?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _questions = C.try_options(L.lang)
    try_question = mo.ui.dropdown(options=_questions, value=next(t for t, q in _questions.items() if q == C.EXAMPLE),
                                  label="Question")
    try_read = mo.ui.slider(1, C.LINKS, step=1, value=C.READ, show_value=True, label="Results read")
    mo.vstack([try_question, try_read])
    return try_question, try_read


@app.cell(hide_code=True)
def try_look(try_question, try_read):
    mo.Html(C.try_picture(L.lang, try_question.value, try_read.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    For Victoria, the answer shows the answer from two results on. For Los Angeles, the own paragraph stands at place 2 of the links, and its first sentence holds the answer: two soccer teams, from 2005 to 2014. But the first result, about the professional sports teams of Southern California, has a sentence with the same number of query words, three: *Los*, *Angeles* and *teams*, and on a tie the first result wins. So at every number of results, the answer is the list of today's teams: a wrong answer, with nothing next to it that says so. The question about Sky+ HD fails in another way: the own paragraph comes first, but the engine picks the sentence about MPEG-4, with *Sky*, *HD* and *uses*, and not the next sentence, with *proprietary* and *system* and the answer, OpenTV. The questions about Jacksonville and the Saxon Garden show the engine when it works.

    The course's engine copies its sentence, so its citation always supports its answer: its errors are a wrong sentence from a real source. An answer engine with an LLM writes new sentences, and can also write a sentence that no source supports. How to build one, and to measure its answers, is the subject of [Answers from the sources](/course/open/build-grounded-answers?lang=en) and [Is the RAG right?](/course/open/build-rag-eval?lang=en).

    ### Exercise 4. Ten links or one answer?

    Sort each point below: does it belong to a page of ten links, or to one answer?
    """)
    return


@app.cell(hide_code=True)
def exercise_pages():
    sort_pages = L.sort("pages")
    sort_pages
    return (sort_pages,)


@app.cell(hide_code=True)
def check_pages(sort_pages):
    result_pages = L.check("pages", sort_pages)
    result_pages
    return (result_pages,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The ten links are the ranking of [BM25: the classic ranking](/course/open/ir-bm25?lang=en), success@k joins the measures of [Did the search work? P@k, MAP and MRR](/course/open/ir-precision-at-k?lang=en), and the answer is the best sentence of [Snippets: showing the results](/course/open/ir-snippets?lang=en), an extractive summary as in [Summarizing: pick or write](/course/open/nlp-summarize?lang=en). This lesson ends the module about search. The next lesson, *Languages with little data*, starts the module about more languages.

    **An everyday example.** A shop assistant who hands you one product saves you the search through the shelves. When it is the wrong one, you only find out at home; in front of the shelf, you would have seen the others.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_success, result_recall, result_precision, result_pages):
    L.wrap_up(result_success, result_recall, result_precision, result_pages)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
