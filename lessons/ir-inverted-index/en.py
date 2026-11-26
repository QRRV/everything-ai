# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-inverted-index. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-inverted-index", "en")
    C = L.checks   # the course's thirteen documents, the paragraphs of SQuAD, the inverted index and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_reads():
    quick_reads = L.choice(
        "quick-reads",
        question="In *Search: needs and relevance*, `C.search` gave the documents for which `matches` is `True`. For one query on the thirteen documents of the course, which documents does it call `matches` on?",
        options={
            "first": "only the documents that hold the first word of the query",
            "right": "all thirteen, for every query",
            "results": "only the results",
            "remember": "all thirteen the first time; after that, only the documents it found before",
        },
        correct="right",
        why={
            "right": "`C.search` goes through every document, and `matches` looks at the words of each one. The work grows with the collection: thirteen documents, or the hundreds of billions of web pages of a search engine. This lesson finds the documents with every word of a query without reading the documents.",
            "first": "To know which documents hold the first word, the search has to look at each document. Which documents does `C.search` go through? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
            "results": "The results are what the search finds. How does it find out which documents are results? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
            "remember": "`C.search` keeps nothing from one query to the next. Which documents does it go through each time? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
        },
    )
    quick_reads
    return (quick_reads,)


@app.cell(hide_code=True)
def quick_check_extra():
    quick_extra = L.choice(
        "quick-extra",
        question="A document holds every word of the query but does not help with the information need. Which list of `split` in *Search: needs and relevance* does it go into?",
        options={
            "found": "found: it is a result",
            "missed": "missed: the search did not return it",
            "right": "extra: a result that does not help",
            "relevant": "found, because a document with every word of the query is relevant",
        },
        correct="right",
        why={
            "right": "The search returns it, because it holds every word, and it does not help: an extra result. The index of this lesson finds the same results as `matches`, extra ones included, only much faster.",
            "found": "Found holds the results that help with the need. Does this one help? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
            "missed": "The search returns every document with all the words of the query. Does it return this one? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
            "relevant": "Relevance is judged against the information need. Does a document with every word always help? See [Search: needs and relevance](/course/open/ir-what-is-search?lang=en).",
        },
    )
    quick_extra
    return (quick_extra,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The search engine of [Search: needs and relevance](/course/open/ir-what-is-search?lang=en) looks at every document for every query. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze describe that way of searching in *Introduction to Information Retrieval* (2008): "The simplest form of document retrieval is for a computer to do this sort of linear scan through documents. This process is commonly referred to as grepping through text, after the Unix command grep, which performs this process." For the *Collected Works* of Shakespeare, a bit under a million words, a linear scan is enough, they write: "you really need nothing more." For a large collection, it is not.

    Google writes about its own search engine: "The Google Search index covers hundreds of billions of web pages and is well over 100,000,000 gigabytes in size." And: "It's like the index in the back of a book – with an entry for every word seen on every web page that we index." A search engine does not read the web when you search: it looks up your words in an index it made beforehand.

    In this lesson you build such an index for the course's thirteen documents. Then you search 2,067 paragraphs of English Wikipedia from SQuAD, a collection of questions about Wikipedia, with an index the course built the same way, without reading a paragraph.

    **An everyday example.** To find what a cookbook says about *saffron*, you do not read the whole book: you look up *saffron* in the index at the back, and turn to the pages it lists.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_zeros = L.guess(
        "guess",
        question="In *A document as a vector*, a document-term matrix had a row for each document and a column for each word, and was mostly zeros. The 2,067 paragraphs of this lesson hold 23,298 different words. Make a table with a row for each paragraph and a column for each word: a 1 where the paragraph holds the word, and a 0 where it does not. What share of its cells is 0?",
        options={
            "half": "about half: a paragraph holds many words",
            "nine": "about nine in ten",
            "right": "more than 99 in 100",
            "three": "about three in four",
        },
        reveal="More than 99 in 100: 99.65% of the 48,156,966 cells are 0. Only 167,175 cells are 1: a paragraph holds 81 different words on average. Half the words, 11,654 of them, are in one paragraph only. Manning and colleagues compute the same for a million documents of about 1,000 words each, with 500,000 different words: \"a minimum of 99.8% of the cells are zero.\" A table that stores every 0 is almost all waste.",
    )
    guess_zeros
    return (guess_zeros,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **A table of 0s and 1s.** Manning and colleagues start with the plays of Shakespeare and the query *Brutus AND Caesar AND NOT Calpurnia*: the plays that hold the words *Brutus* and *Caesar*, and not the word *Calpurnia*. Give each word a row and each play a column, with a 1 where the play holds the word and a 0 where it does not. That table is an **incidence matrix**: the document-term matrix of *A document as a vector* turned on its side, with a 1 for each count that is not 0. The query takes the rows of *Brutus* and *Caesar*, flips each number in the row of *Calpurnia* (each 0 becomes a 1, each 1 a 0), and keeps the columns where all three rows hold a 1: "The answers for this query are thus Antony and Cleopatra and Hamlet". For a million documents of about 1,000 words, with 500,000 different words, the table has, in the book's words, "half-a-trillion 0's and 1's", too many for the memory of a computer.

    **Keep only the 1s.** An **inverted index** keeps, for each word, only the list of the documents that hold it, in order of their number: the **postings list** of the word. Each number in it is a **posting**, which, in the book's words, "records that a term appeared in a document". The index at the back of a book works the same way, from a word to its pages, and Manning and colleagues point out: "The name is actually redundant: an index always maps back from terms to the parts of a document where they occur." Building the index takes one pass over the documents: for each document, in order, add its number to the list of each word it holds, once. Because the documents come in order, each list comes out sorted.

    **AND: walk two lists.** For *wine AND heart*, take the two postings lists and put a finger at the start of each. When both fingers point to the same number, that document holds both words: keep it, and move both fingers forward. When the numbers differ, move the finger at the smaller number forward: the finger in the other list already stands at a larger number, so that list cannot hold the smaller one. When one list runs out, the walk stops. Each step compares two numbers. The walk takes at most as many steps as there are numbers in the two lists together, and it never opens a document. Its result is the **intersection** of the two lists: the numbers that both hold.

    **More words: the shortest list first.** For three words, walk two lists, then walk the answer so far with the third list. The order does not change the answer, but it changes the work. In the book's words: "The standard heuristic is to process terms in order of increasing document frequency". The **document frequency** of a word is the number of documents that hold it: the length of its postings list. When the walks start from the shortest list, the answer so far is never longer than that list, and a walk stops as soon as the answer so far runs out, so there is usually less to walk.

    **OR, NOT, and Boolean retrieval.** OR walks two lists the same way and keeps every number that either list holds. NOT takes the documents that are not on a list. Search with AND, OR and NOT is **Boolean retrieval**, after the algebra of the mathematician George Boole (1847): a document matches or it does not, and the results come in no order of quality. It stayed in use for a long time. The book describes the legal search service Westlaw, started in 1975, "with over half a million subscribers performing millions of searches a day"; in 2005, Boolean search was still its default, although ranked search had been added in 1992.

    The picture builds the index of the course's thirteen documents for the query *red wine heart*, and walks the two shortest lists.
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
    | **incidence matrix** | a table with a row for each word and a column for each document: 1 where the document holds the word, 0 where it does not |
    | **inverted index** | for each word, the list of the documents that hold it; a search engine looks words up in it instead of reading the documents |
    | **postings list** | the list of a word in an inverted index: the numbers of the documents that hold the word, sorted; each number is a **posting** |
    | **document frequency** | the number of documents that hold a word: the length of its postings list |
    | **intersection** | the numbers that both lists hold; for postings lists, found in one walk along both |
    | **Boolean retrieval** | search with AND, OR and NOT; a document matches or it does not, and the results are not ranked |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To answer a query with AND from an inverted index:

    1. Build the index once: for each document, in order, add its number to the postings list of each word it holds, once.
    2. Look up each word of the query. A word without a list is in no document, and the answer is empty.
    3. Sort the lists by length, the shortest first.
    4. Walk the first two lists with a finger in each: keep a number that both hold, and move on the finger at the smaller number.
    5. Walk the answer so far with each next list in turn.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.DOCS` holds the course's thirteen documents as lists of words, and `C.QUERIES` the queries of *Search: needs and relevance*. `C.build_index` builds the inverted index of a list of documents, `C.intersect` walks two postings lists, and `C.search_and` answers a query with AND, the shortest list first:
    """)
    return


@app.cell
def example_cell():
    _index = C.build_index(C.DOCS[L.lang])
    print(len(_index))
    _query = C.QUERIES[L.lang]["wine"]
    for _word in _query:
        print(_word, _index[_word])
    print(C.intersect(_index[_query[1]], _index[_query[2]]))
    print(C.search_and(_query, _index))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The thirteen documents hold 116 different words, so the index has 116 postings lists. The word *red* is in five documents, *wine* and *heart* in four each. The walk along the lists of *wine* and *heart* keeps 6, 7 and 8. `C.search_and` starts with those two lists, the shortest, and then walks the answer with the list of *red*, which holds all three. The answer is what `matches` found in *Search: needs and relevance*, documents 6, 7 and 8, and the index found it without opening a document.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_build_index():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. build_index: the inverted index

        Write `build_index`, which gets `docs`, a list of documents, each a list of words, such as `C.DOCS["en"]`. It gives back the inverted index: a dictionary from each word to its postings list, the numbers of the documents that hold the word, in order and each number once. In the start code, the outer loop goes through each number `doc` in `range(len(docs))`, and the inner loop through each `word` of the document `docs[doc]`. A word seen for the first time gets an empty list in `index`, and `.append` adds the number of the document to the list of the word. Replace the `...` with the test that lets each number into a list only once, also when a document holds a word twice.

        The check calls `build_index([["red", "wine"], ["white", "wine", "wine"], ["red"]])`, which gives `{"red": [0, 2], "wine": [0, 1], "white": [1]}`, and `build_index([["python"], [], ["python", "snake", "python"]])`, which gives `{"python": [0, 2], "snake": [2]}`. Then cases of its own: no documents at all, a document that holds a word twice with another word between, a word three times in one document, a word in every document, and the thirteen documents of the course in both languages.
        """),
        L.task("Add the number of each document to the postings list of each word it holds, once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_build_index():
    box_build_index = L.code_box("build_index", lines=9)
    box_build_index
    return (box_build_index,)


@app.cell(hide_code=True)
def check_build_index(box_build_index):
    result_build_index = L.check("build_index", box_build_index.value)
    result_build_index
    return (result_build_index,)


@app.cell(hide_code=True)
def hints_build_index():
    L.hints(
        "build_index",
        [
            "A number should go into the list of a word once, also when the word occurs twice in the document. `index[word]` is the list of `word` so far, and the test goes in the `if`: `x in a_list` is `True` when `x` is one of the items of a list.",
            "Let the number of this document, `doc`, through only when it is not yet one of the numbers in the list of the word. `not in` is the opposite of `in`.",
            "`            if ___ index[word]:`",
        ],
        why="`if doc not in index[word]:`: a number goes into the list of a word only the first time the word occurs in the document.",
    )
    return


@app.cell(hide_code=True)
def text_after_build_index():
    mo.md(r"""
    In the picture, *red* has a list of five documents, *wine* and *heart* of four, and *python* of five. In the 2,067 paragraphs, the lists are far more uneven: *the* is in 2,046 paragraphs, and half of the 23,298 words are in one paragraph only. Most postings lists are short and a few are very long: the same pattern as Zipf's law, from the lesson [Zipf's law and Heaps' law](/course/open/nlp-zipf-heaps?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_intersect():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. intersect: walking two lists

        Write `intersect`, which gets two postings lists `p1` and `p2`, each sorted from small to large, and gives back the list of the numbers that both hold, in order. In the start code, `i` is the place of the finger in `p1` and `j` the place of the finger in `p2`, both starting at 0. The `while` loop runs as long as neither list has run out: while `i < len(p1)` and `j < len(p2)`. When `p1[i]` and `p2[j]` are the same number, `.append` adds it to `answer`, and both fingers move forward. Otherwise one finger moves forward: `i` in the `elif`, `j` in the `else`. Replace the `...` with the test for moving forward in `p1`.

        The check calls `intersect([6, 7, 8, 10], [6, 7, 8, 9])`, the lists of *wine* and *heart*, which gives `[6, 7, 8]`, and `intersect([2, 4, 8, 16, 32], [1, 2, 3, 5, 8, 13])`, which gives `[2, 8]`. Then cases of its own: an empty list on either side, lists with no number in common, two lists that are the same, two lists that share some numbers but start and end differently, a short list against a longer one in both orders, and two long lists of multiples of 3 and of 4.
        """),
        L.task("Give the numbers that both sorted lists hold, in one walk along both."),
    ])
    return


@app.cell(hide_code=True)
def exercise_intersect():
    box_intersect = L.code_box("intersect", lines=14)
    box_intersect
    return (box_intersect,)


@app.cell(hide_code=True)
def check_intersect(box_intersect):
    result_intersect = L.check("intersect", box_intersect.value)
    result_intersect
    return (result_intersect,)


@app.cell(hide_code=True)
def hints_intersect():
    L.hints(
        "intersect",
        [
            "When `p1[i]` and `p2[j]` differ, the finger at the smaller number moves on. The `elif` decides when that is the finger in `p1`, which moves on with `i = i + 1`; `<` compares two numbers.",
            "The finger in `p1` moves on when its number is the smaller of the two: when the number at `i` in `p1` is less than the number at the finger in the other list.",
            "`        elif p1[i] ___:`",
        ],
        why="`elif p1[i] < p2[j]:`: the number in `p1` is the smaller one, so the other list cannot hold it any more, and the finger in `p1` moves on.",
    )
    return


@app.cell(hide_code=True)
def text_after_intersect():
    mo.md(r"""
    In the picture, the walk along *rhine* and *bridge* looks at no more than 61 numbers and finds five paragraphs; reading the paragraphs themselves means reading 259,216 words. A walk takes at most as many steps as there are numbers in its two lists together, so the work depends on the lengths of the lists of the query, not on how many words the documents hold. Search engines shorten long walks further: the book describes skip pointers, which let a walk jump over a stretch of a long list.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_search_and():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. search_and: every word of the query

        Write `search_and`, which gets `query`, a list of at least one word, and `index`, an inverted index, such as `C.SMALL_INDEX["en"]`, the index of the course's thirteen documents. It gives back the numbers of the documents that hold every word of the query. In the start code, the first loop goes through each `word` of `query`: a word that is not in `index` is in no document, so the answer is `[]`; otherwise, `.append` adds its postings list to `lists`. `sorted` with `key=len` puts the lists in order of length, the shortest first, and `answer` starts as the shortest. The last loop goes through each next list, `postings`, in `lists[1:]`. Replace the `...` with the new answer: the walk of the answer so far along `postings`, with the function `intersect` of exercise 2, which the box already knows.

        The check calls `search_and(["red", "wine", "heart"], C.SMALL_INDEX["en"])`, which gives `[6, 7, 8]`, and `search_and(["python", "slang"], C.SMALL_INDEX["nl"])`, with the index of the Dutch lesson, which gives `[2]`. Then cases of its own: a query of one word, a query of four words, a word that is in no document, the same word twice, the Dutch query *rode wijn hart*, a query of six words, and a small index written out.
        """),
        L.task("Give the documents that hold every word of the query, walking the lists from the shortest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search_and():
    box_search_and = L.code_box("search_and", lines=12)
    box_search_and
    return (box_search_and,)


@app.cell(hide_code=True)
def check_search_and(box_search_and):
    result_search_and = L.check("search_and", box_search_and.value)
    result_search_and
    return (result_search_and,)


@app.cell(hide_code=True)
def hints_search_and():
    L.hints(
        "search_and",
        [
            "`intersect` walks two postings lists and gives back the numbers that both hold. In this loop, `answer` holds what the walks kept so far, and `postings` is the list of the next word.",
            "The new answer is what the answer so far and the next list both hold: call the function of exercise 2 with the two of them.",
            "`        answer = intersect(___)`",
        ],
        why="`answer = intersect(answer, postings)`: each round keeps the numbers that the answer so far and the next list both hold.",
    )
    return


@app.cell(hide_code=True)
def text_after_search_and():
    mo.md(r"""
    In the picture, *oxygen water fire* finds nothing: *fire* is in 24 paragraphs and *oxygen* in 51, five paragraphs hold both, and none of the five holds *water*. For *steam engine water*, the walks start with *engine* and *steam*, 39 and 45 paragraphs; 27 are left, and the walk with the 87 paragraphs of *water* stops after 38 steps, when those 27 run out: 94 comparisons in all. Started with *water* and *engine*, the same answer takes 156. Type the queries in the box below to see each walk.

    ### Try it: search the paragraphs

    Type a query in the box. The picture looks up each word in the index of the 2,067 paragraphs, walks the lists from the shortest, and shows how many paragraphs are left after each walk and how many comparisons each walk made. Try *rhine bridge*, then add *river*. The third walk keeps all five paragraphs and still makes more comparisons than the first: the finger in the long list of *river* moves forward number by number until one of the lists runs out. Then put a word that is in nearly every paragraph, such as *the*, next to a rare one.
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
def text_exercise_answers():
    mo.md(r"""
    ### Exercise 4. What does this index answer?

    The index of this lesson keeps, for each word, the numbers of the documents that hold it, and nothing else. OR walks two lists and keeps every number that either holds; NOT takes the documents that are not on a list. Sort each query below: this index answers it, or it needs more than which documents hold a word.
    """)
    return


@app.cell(hide_code=True)
def exercise_answers():
    sort_answers = L.sort("answers")
    sort_answers
    return (sort_answers,)


@app.cell(hide_code=True)
def check_answers(sort_answers):
    result_answers = L.check("answers", sort_answers)
    result_answers
    return (result_answers,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Search: needs and relevance](/course/open/ir-what-is-search?lang=en) found every document with all the words of a query by reading each document; the index of this lesson finds the same documents from the postings lists alone. The incidence matrix is the document-term matrix of [A document as a vector](/course/open/nlp-document-vectors?lang=en) with a 1 for each count that is not 0, and the inverted index stores it sparsely: only the 1s. The next lesson, *Ranking by cosine: the vector space model*, puts the results in order, with a count for each word, which an index can keep beside each posting.

    **An everyday example.** In a library catalog, you find every book by an author without walking along the shelves: you look the name up and get a list of call numbers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_build_index, result_intersect, result_search_and, result_answers):
    L.wrap_up(result_build_index, result_intersect, result_search_and, result_answers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
