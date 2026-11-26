# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-what-is-search. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-what-is-search", "en")
    C = L.checks   # the course's collection, the needs with their judgments, the clue words and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [A document as a vector](/course/open/nlp-document-vectors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vector():
    quick_vector = L.choice(
        "quick-vector",
        question='In *A document as a vector*, a vocabulary gave each word a column. With the vocabulary `{"cat": 0, "dog": 1, "sat": 2, "the": 3}`, what is the document vector of *the dog saw the cat*?',
        options={
            "order": "`[3, 1, 3, 0]`: the column of each known word, in order",
            "once": "`[1, 1, 0, 1]`: a 1 for each word of the vocabulary that it holds",
            "right": "`[1, 1, 0, 2]`: a count for each column, without *saw*",
            "new": "`[1, 1, 0, 2, 1]`: with a new column for *saw*",
        },
        correct="right",
        why={
            "right": "Each column holds how often its word occurs: *cat* and *dog* once, *sat* not at all, *the* twice. *saw* has no column, so it is left out. The search engine of this lesson looks at part of that: whether a document holds a word.",
            "order": "A document vector has one number for each column of the vocabulary. What does the number in a column count? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "once": "*the* occurs twice in the sentence. What does a document vector keep of how often a word occurs? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "new": "The vocabulary is fixed. What happens to a word that has no column? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_vector
    return (quick_vector,)


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="A bag of words keeps which words a text has and how often, and forgets their order. Which two searches give the same document vector, over a vocabulary that holds all their words?",
        options={
            "plural": "*a glass of red wine* and *two glasses of good red wine with dinner*",
            "right": "*brazil traveler to usa* and *usa traveler to brazil*",
            "twice": "*buy a python* and *buy a python python*",
            "the": "*the snake* and *a snake*",
        },
        correct="right",
        why={
            "right": "Both hold *brazil*, *traveler*, *to* and *usa* once, so their vectors are the same, and the trips go opposite ways. A search engine that looks only at which words a search holds cannot tell the two apart. The next part of the lesson shows a search engine that mixed up two such searches.",
            "plural": "*glass* and *glasses* are different words, each with a column of its own. Does every column get the same count? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "twice": "A document vector counts how often each word occurs. Does *python* occur as often in both? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "the": "*the* and *a* are different words. Do both searches hold the same words? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You search every day: on the web, in a shop, in your mail, in the catalog of a library. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze open their book *Introduction to Information Retrieval* (2008) with that: "hundreds of millions of people engage in information retrieval every day when they use a web search engine or search their email".

    A search engine gets a few words and has to work out what the person wants. Google wrote in 2019: "We see billions of searches every day, and 15 percent of those queries are ones we haven't seen before". In the same post, it showed the search *2019 brazil traveler to usa need a visa*, typed by someone from Brazil who wants to travel to the United States. Google's search used to return results about travelers from the United States who go to Brazil: the right words, and the trip the wrong way round.

    So Google has people judge its results. Its overview of November 2023 counts about 16,000 Search Quality Raters. They rate how well a result meets the need behind a search, on a scale from *Fully Meets* to *Fails to Meet*. In Google's words, "These ratings do not directly impact ranking": Google uses them to test a change before it goes live. In 2023, Google ran 719,326 such search quality tests, and made 4,781 changes to its search engine, which it calls launches.

    This lesson is about the gap between the words someone types and what that person wants to find, and about judging a search engine by the second.

    **An everyday example.** Ask a librarian for a book about pythons, and the librarian asks back: the snake, or the programming language?
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_agree = L.guess(
        "guess",
        question="TREC is a yearly evaluation of search systems, run by NIST, the American institute for standards. In TREC-4, in 1995, the person who wrote each of 49 written-out search tasks judged the documents found for it, and two others judged a sample of those documents again; all three were retired information analysts. Take two of the three judges, and the documents of the sample that at least one of the two marked relevant. What share of those did both mark relevant, on average?",
        options={
            "nine": "about nine out of ten",
            "three": "about three out of four",
            "right": "less than half",
            "one": "about one out of ten",
        },
        reveal="Less than half. Ellen Voorhees of NIST compared the three sets of judgments in 2000: for the three pairs of judges, the documents that both marked relevant were 42%, 49% and 43% of those that at least one of the two marked relevant; for all three judges, 30%. Of the documents that the author of a task marked relevant, 30% were judged not relevant by both others. Yet when Voorhees ranked 33 search systems with each set of judgments, the order of the systems stayed almost the same. People disagree about what helps, and a comparison of search engines still works.",
    )
    guess_agree
    return (guess_agree,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Information retrieval** is, in the words of Manning and colleagues, "finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers)". A program that does it is a search engine. The book draws the line that this lesson is about: "An information need is the topic about which the user desires to know more, and is differentiated from a query, which is what the user conveys to the computer in an attempt to communicate the information need."

    The **information need** is what someone wants to know or do; the **search query** consists of the words that person types to say it. Queries are short: in the book, "the average number of keywords in a web search is somewhere between two and three". A few words leave a lot unsaid.

    A document is **relevant** when it helps with the need, and **relevance** is judged against the need, not against the words. Manning and colleagues: "A document is relevant if it addresses the stated information need, not because it just happens to contain all the words in the query." Their example of a need: whether red wine lowers the risk of a heart attack more than white wine does. A page that suggests a glass of red wine with heart-healthy food holds the words *red*, *wine* and *heart*, and does not answer the question.

    One query can stand for different needs. The book again: "If a user types python into a web search engine, they might be wanting to know where they can purchase a pet python. Or they might be wanting information on the programming language Python." In 2009, TREC wrote down six needs for the query *defender*, among them the Windows Defender program, the Land Rover Defender, an arcade game and the newspaper Chicago Defender.

    **Three kinds of need.** In 2002, drawing on work at the search engine AltaVista, Andrei Broder sorted the needs behind web searches into three kinds: "1. Navigational. The immediate intent is to reach a particular site. 2. Informational. The intent is to acquire some information assumed to be present on one or more web pages. 3. Transactional. The intent is to perform some web-mediated activity." *youtube* is navigational: the person wants to reach that site. *leukemia* is informational. *buy a ball python* is transactional: the person wants to do something, here buy. Broder warned that the kind cannot be read from the query with certainty. How common each kind is depends on how you count. From a survey of AltaVista users and 400 searches from its log, Broder estimated informational searches at less than 50%. Bernard Jansen, Danielle Booth and Amanda Spink sorted 1,523,793 searches on the search engine Dogpile, from one day in 2005, with rules that look at their words, such as *buy* or the name of a site; after checking the rules against searches they sorted by hand, they estimated about 65% informational, 15% navigational and 20% transactional.

    **People judge.** To measure a search engine, you need documents, needs and judgments. TREC writes each need out as a topic, with a title, a description and a narrative, and keeps it apart from the query: TREC "calls a natural language statement of information need a "topic" to distinguish it from a "query", which is the data structure actually presented to the retrieval system." People then judge the top documents that the search systems return for a topic: relevant or not. Manning and colleagues advise at least fifty needs: "fifty information needs has usually been found to be a sufficient minimum."

    The picture runs the query *python* on the course's collection of thirteen documents, and judges each result against two needs, as people judge a search engine.
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
    | **information retrieval** | finding, in a large collection, the documents that help with an information need |
    | **information need** | what someone wants to know or do when searching |
    | **search query** | the words someone types to say an information need, often two or three |
    | **relevance** | whether a document helps with the information need; a document that helps is **relevant** |
    | **navigational, informational, transactional** | Broder's three kinds of need: to reach a site, to find something out, to do something |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To judge a search engine on one need:

    1. Write the need out in a sentence, as a TREC topic does; the query alone does not say it.
    2. Run the query: the results are the documents the search engine gives back.
    3. Judge each result against the need, not against the words of the query: relevant or not.
    4. Look for the relevant documents that the search did not return: a document can help without holding every word of the query.
    5. Count both mistakes: results that do not help, and relevant documents that were missed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.TEXTS` holds the thirteen documents of the course's collection in the language of this lesson, and `C.DOCS` the same documents as lists of words, made with `C.words`, which cuts a text into words in small letters. `C.QUERIES` holds the two queries of the lesson as lists of words, and `C.NEEDS` the three needs behind them, each with its query and the numbers of the documents that help with it, as the course judged them; `C.NEED_TEXT` writes each need out. `C.search` gives the numbers of the documents that hold every word of a query, the results, and `C.split` splits the results by a need into three lists:
    """)
    return


@app.cell
def example_cell():
    print(C.words(C.TEXTS[L.lang][1]))
    _query = C.QUERIES[L.lang]["python"]
    _results = C.search(_query, C.DOCS[L.lang])
    print(_query, _results)
    for _need in ["buy", "learn"]:
        print(C.NEED_TEXT[L.lang][_need])
        print(C.split(_results, C.NEEDS[_need]["relevant"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Five of the thirteen documents hold the word *python*: 0, 2, 3, 4 and 5. For the need to buy a pet python, only document 3 helps: it is in the first list. Documents 0, 2, 4 and 5 are results that do not help, in the second list. Document 1, *Ball pythons for sale*, helps but is not a result, in the third list: it holds *pythons*, not *python*. For the need to learn the language, document 0 helps, and the other four results do not. The query and its results are the same for both needs; only the need decides which results are good.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_matches():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. matches: every word of the query

        Write `matches`, which gets `query`, a list of words, and `document`, the words of one document as a list, such as `C.DOCS["en"][1]`. It gives back `True` when every word of the query is in the document, and `False` as soon as one word is missing. This is the simplest search engine: the documents that match are the results. In the start code, the loop goes through each `word` of `query`, and after the loop, `return True` says that no word was missing. Replace the `...` with the test that makes the function stop with `False`.

        The check calls `matches(["python"], C.DOCS["en"][1])`, which gives `False`, because document 1 holds *pythons* and not *python*, and `matches(["rode", "wijn", "hart"], C.DOCS["nl"][7])`, with a document of the Dutch lesson, which gives `True`. Then cases of its own: a document that holds only some of the words, with two queries; a query of one word that matches; a document that holds every word in another order; two Dutch documents; a query with the same word twice; and an empty document.
        """),
        L.task("Give True when every word of the query is in the document, and False when one is missing."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matches():
    box_matches = L.code_box("matches", lines=6)
    box_matches
    return (box_matches,)


@app.cell(hide_code=True)
def check_matches(box_matches):
    result_matches = L.check("matches", box_matches.value)
    result_matches
    return (result_matches,)


@app.cell(hide_code=True)
def hints_matches():
    L.hints(
        "matches",
        [
            "The function can give `False` as soon as it finds one `word` of the query that the document lacks. The test goes in the `if`: `x in a_list` is `True` when `x` is one of the items of a list.",
            "Test the word of this round of the loop, `word`, against the list `document`: the document is not a result when the word is not one of its words. `not in` is the opposite of `in`.",
            "`        if word ___:`",
        ],
        why="`if word not in document:`: one word of the query that the document lacks is enough for `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_matches():
    mo.md(r"""
    In the picture, *python* finds five documents, and *red wine heart* three: the page on red wine, white wine and the heart, the page on heart-healthy food, and the walk through the old town, where *heart* means the center of the town. *Alcohol and heart attacks: do beer, white and red differ?* answers the question about red and white wine, and is not a result: it lacks the word *wine*. Manning and colleagues call search on exact words like this Boolean retrieval; a document either matches or not, and nothing says which result is best. Later lessons of this module make it fast and rank its results.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. split: judging the results against a need

        Write `split`, which gets `results`, the numbers of the documents that a search returned, and `relevant`, the numbers of the documents that help with the need. It gives back three lists: `found`, the results that are relevant; `extra`, the results that are not; and `missed`, the relevant documents that are not among the results, in the order of `relevant`. In the start code, the loop goes through each `doc` of `results` and puts it into `found` or `extra`, with `.append`. Replace the `...` with the list of missed documents.

        The check calls `split([0, 2, 3, 4, 5], [1, 3])`, the results of *python* with the need to buy a pet python, which gives `([3], [0, 2, 4, 5], [1])`, and `split([6, 7, 8], [6, 9])`, the results of *red wine heart* with the need about red and white wine, which gives `([6], [7, 8], [9])`. Then cases of its own: the need to learn the language, results in another order than the relevant documents, no results, no relevant documents, and every relevant document found.
        """),
        L.task("Give the results that help, the results that do not, and the relevant documents that the search missed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split", lines=10)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split",
        [
            "`missed` can be a list comprehension over `relevant`: `[doc for doc in relevant if test]` keeps each `doc` of `relevant` for which the test holds.",
            "A relevant document is missed when the search did not return it: when it is not one of the numbers in `results`.",
            "`    missed = [doc for doc in relevant if ___]`",
        ],
        why="`missed = [doc for doc in relevant if doc not in results]`: the relevant documents that are not results, in the order of `relevant`.",
    )
    return


@app.cell(hide_code=True)
def text_after_split():
    mo.md(r"""
    The two needs behind *python* get the same five results, and your split judges them in two ways: one result helps with the need to buy, one with the need to learn. A search engine that only sees the words cannot do better for both at once. The two lists of mistakes pull in different directions: a search that returns more documents misses no more relevant ones, and often fewer, but returns more that do not help. The lesson *Did the search work? P@k, MAP and MRR* turns such judgments into numbers, and with judgments for many needs, those numbers compare search engines, as in TREC.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind_of():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. kind_of: guessing the kind of need

        Write `kind_of`, which gets `query`, a list of words, and `clues`, a dictionary from a kind of need to its list of clue words, such as `C.CLUES["en"]`: names of sites and words such as *login* for `"navigational"`, and words such as *buy* and *download* for `"transactional"`. It gives back the first kind, in the order of `clues`, that has a word of the query among its clue words, and `"informational"` when no word is a clue. Jansen and colleagues used rules like these, in this order: navigational first, then transactional, and informational for the rest. In the start code, the outer loop goes through each `kind` of `clues`, and the inner loop through each `word` of `query`. Replace the `...` with the test for a clue.

        The check calls `kind_of(["facebook", "login"], C.CLUES["en"])`, which gives `"navigational"`, and `kind_of(["hoe", "leer", "ik", "python"], C.CLUES["nl"])`, with the clue words of the Dutch lesson, which gives `"informational"`. Then cases of its own: a transactional English query, a transactional Dutch query, a query with a clue of each kind, the same query with the kinds in the other order, a query with a navigational clue only, an empty query, and no clue words at all.
        """),
        L.task("Give the first kind with a clue word in the query, or \"informational\" when there is none."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kind_of():
    box_kind_of = L.code_box("kind_of", lines=7)
    box_kind_of
    return (box_kind_of,)


@app.cell(hide_code=True)
def check_kind_of(box_kind_of):
    result_kind_of = L.check("kind_of", box_kind_of.value)
    result_kind_of
    return (result_kind_of,)


@app.cell(hide_code=True)
def hints_kind_of():
    L.hints(
        "kind_of",
        [
            "`clues` is a dictionary: `clues[\"navigational\"]` is the list of clue words of that kind, and `word` is one word of the query. The test goes in the `if`.",
            "The word is a clue for `kind`, the kind of this round of the outer loop, when it is one of the clue words of that kind: look that list up in `clues`, with `kind` as the key.",
            "`            if word in ___:`",
        ],
        why="`if word in clues[kind]:`: the word is one of the clue words of the kind.",
    )
    return


@app.cell(hide_code=True)
def text_after_kind_of():
    mo.md(r"""
    With the course's version, the rules agree with the course's reading on 12 of the 15 queries in each language. They miss *lufthansa airlines*, a site that is not among the clue words. They read *ebay history* as navigational, though the person wants to read about the site, not reach it; and *book about snakes* as transactional, because *book* is a clue for booking a hotel or a ticket. Jansen and colleagues checked their rules against 400 searches sorted by hand: the rules agreed on 74% of them. Words can point one way and the need another, and only the person who typed a search knows its need for sure.

    ### Try it: search the collection

    Type a query in the box. The picture shows which of the thirteen documents hold every word of it, and, for each other document, which words of the query it lacks. Try *snake*: which documents about snakes does it miss? Then try a query of three words, and see how quickly the results run out.
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.TRY_START[L.lang], label="Your query", rows=2, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_judge():
    mo.md(r"""
    ### Exercise 4. Found, missed or extra?

    TREC topic 427, from 1999, has the title *UV damage, eyes*. Its description: "Find documents that discuss the damage ultraviolet (UV) light from the sun can do to eyes." Its narrative: "A relevant document will discuss diseases that result from exposure of the eyes to UV light, treatments for the damage, and/or education programs that help prevent damage. Documents discussing treatment methods for cataracts and ocular melanoma are relevant even when a specific cause is not mentioned. However, documents that discuss radiation damage from nuclear sources or lasers are not relevant." Cataracts make the lens of the eye cloudy; ocular melanoma is a cancer of the eye.

    The query is *uv damage eyes*, and the search engine is your `matches`. Sort each document below: found (relevant, with every word of the query), missed (relevant, but a word of the query is not in it, so the search does not return it) or extra (every word of the query is in it, but it is not relevant).
    """)
    return


@app.cell(hide_code=True)
def exercise_judge():
    sort_judge = L.sort("judge")
    sort_judge
    return (sort_judge,)


@app.cell(hide_code=True)
def check_judge(sort_judge):
    result_judge = L.check("judge", sort_judge)
    result_judge
    return (result_judge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A document as a vector](/course/open/nlp-document-vectors?lang=en) kept which words a text holds and forgot their order; the search engine of this lesson uses only which words a document holds, and so it cannot see that *brazil traveler to usa* is a trip the other way round from *usa traveler to brazil*. The intent of [Booking a train: frames and slots](/course/open/nlp-slot-filling?lang=en) is close to the kind of need: what the user wants to do, read from the words. The next lesson, *The inverted index*, finds the documents with every word of a query without reading every document, and later lessons of this module rank the results and measure a search engine with judgments of people.

    **An everyday example.** You search your mail for *invoice*, and the mail you need says *bill*: relevant, and missed.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_matches, result_split, result_kind_of, result_judge):
    L.wrap_up(result_matches, result_split, result_kind_of, result_judge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
