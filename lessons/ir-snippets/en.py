# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-snippets. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-snippets", "en")
    C = L.checks   # the paragraphs and questions of XQuAD, the results of BM25, the snippets and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en) and [BM25: the classic ranking](/course/open/ir-bm25?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pieces():
    quick_pieces = L.choice(
        "quick-pieces",
        question=r'What does `re.findall(r"\S+", "U.S. troops (1855), left.")` give?',
        options={
            "letters": r"`['U', 'S', 'troops', '1855', 'left']`: the runs of letters and digits",
            "spaces": r"`[' ', ' ', ' ']`",
            "right": r"`['U.S.', 'troops', '(1855),', 'left.']`",
            "sentences": r"`['U.S.', 'troops (1855), left.']`: cut after the first period",
        },
        correct="right",
        why={
            "right": r"`\S` matches one character that is not white space, so `\S+` is a run of characters between spaces, and each word keeps its capitals and its punctuation: *(1855),* keeps its brackets and its comma. This lesson cuts paragraphs this way, so that a snippet can show each word as it is written.",
            "letters": r"Those are the runs of `\w`, letters and digits. Which characters does `\S` match? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "spaces": r"Those are the runs of `\s`, the white space. With a capital S, which characters match? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "sentences": r"`re.findall` gives every run that matches, not sentences: where does a run of `\S` stop? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
        },
    )
    quick_pieces
    return (quick_pieces,)


@app.cell(hide_code=True)
def quick_check_bm25():
    quick_bm25 = L.choice(
        "quick-bm25",
        question="Two paragraphs of 87 words each hold *Victoria* twice and *parliament* twice, and no other query words. In paragraph A, the four words stand in one sentence; in paragraph B, they are spread over the paragraph. Which one gets the higher BM25 score for the query *parliament Victoria*?",
        options={
            "near": "A: BM25 gives words that stand close together a higher score",
            "spread": "B: words spread over the paragraph cover more of it, so each of them adds more to the score",
            "first": "A, when its sentence comes first",
            "right": "the same: BM25 sees the counts and the length, not where the words stand",
        },
        correct="right",
        why={
            "right": "BM25 gives each query word a weight from its count, its IDF and the length of the paragraph, and adds these weights: where a word stands plays no part. That is enough to rank the paragraphs, but a reader also wants to see where the query words stand. This lesson finds that place.",
            "near": "Does BM25 look at where a word stands, or only at how often it occurs? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "spread": "Which numbers go into the weight of a word in BM25: its places in the paragraph, or its count, its IDF and the length? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "first": "Does the formula of BM25 hold anything about the order of the sentences? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
        },
    )
    quick_bm25
    return (quick_bm25,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [BM25: the classic ranking](/course/open/ir-bm25?lang=en), BM25 ranked the 2,067 paragraphs for each question of XQuAD. A search engine does not show a reader those scores. It shows a page of results, and for each result a title and a few lines of text; from those lines, the reader decides which result to open and which to skip. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze, in *Introduction to Information Retrieval* (2008): "The standard way of doing this is to provide a snippet, a short summary of the document, which is designed so as to allow the user to decide its relevance."

    Which lines should those be? The first lines of a document are the same for every query. In 1998, Anastasios Tombros and Mark Sanderson tried something else: "The long standing motivation for this work was a belief that if, in the retrieved document list, users of IR systems could see the sentences in which their query words appeared, they could better judge the relevance of documents."

    **An everyday example.** You look up a word in the index at the back of a book and turn to the page. Your eye goes straight to the word and the lines around it; you do not read the page from the top. A snippet does that for every result of a search at once.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_opened = L.guess(
        "guess",
        question="Tombros and Sanderson gave 20 people five minutes per query to judge which documents of a result list were relevant. With the first sentences of each document as its summary, people opened the full text of 23.7% of the documents they judged. With summaries made of the sentences that hold the query words, how often did they open the full text?",
        options={
            "same": "about as often: 22%",
            "half": "half as often: 12%",
            "right": "about one document in 75: 1.3%",
            "more": "more often, 31%: the query words made people want to read on",
        },
        reveal="About one document in 75: 1.32%, or 0.3 documents per query, against 4.74 with the first sentences. With the summaries made for the query, people also judged more documents in the five minutes, 22.62 against 20, and found more of the relevant ones: 65.6% against 49.76%. Tombros and Sanderson: \"Users need more clues to establish the relevance of documents, and especially they need clues about the context in which the query terms are used in these documents.\"",
    )
    guess_opened
    return (guess_opened,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Snippet.** *Introduction to Information Retrieval*: "Typically, the snippet consists of the document title and a short summary, which is automatically extracted." A page of results shows each result with its title and its snippet. The paragraphs of XQuAD come from Wikipedia, so their title is the title of their Wikipedia page.

    **Static and dynamic summaries.** The book again: "The two basic kinds of summaries are static, which are always the same regardless of the query, and dynamic (or query-dependent), which are customized according to the user's information need as deduced from a query." A static summary can be the start of the document, as in "The simplest form of summary takes the first two sentences or 50 words of a document", or a description that the maker of a web page writes in its HTML. Tombros and Sanderson call a dynamic summary query biased, and so do Andrew Turpin, Yohannes Tsegay, David Hawking and Hugh Williams in 2007: "A query-biased snippet is one selectively extracted on the basis of its relation to the searcher’s query."

    **Query words and words as written.** The query words of a question are its words in small letters, without the stop words of [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en). A paragraph is cut at the spaces, with `re.findall(r"\S+", text)`, so each word keeps its capitals and its punctuation, and a snippet can show it as it is written. To compare a word with the query words, the course's function `plain` gives the word in small letters, without the punctuation at its start and end: *Parliament* gives *parliament*, and *(NFC),* gives *nfc*.

    **Window.** The book: "If the query is found as a phrase, occurrences of the phrase in the document will be shown as the summary. If not, windows within the document that contain multiple query terms will be selected." And: "Commonly these windows may just stretch some number of words to the left and right of the query terms." This lesson takes the simplest form: it counts the query words in each run of 20 words in a row, and takes the run with the most, the first one on a tie.

    **Best sentence.** A window can start and stop in the middle of a sentence, and the book notes that "users prefer snippets that read well because they contain complete phrases." Turpin and colleagues: "In order to construct a snippet, all sentences in a document should be ranked against the query, and the top two or three returned as the snippet." Tombros and Sanderson gave each sentence a query score, "derived by dividing the square of the number of query terms included in that sentence by the total number of the terms of the specific query." The question *What document formed the Parliament of Victoria?* has four query words: *document*, *formed*, *parliament* and *victoria*. A sentence that holds *parliament* and *Victoria* gets 2² / 4 = 1, and a sentence with only *Victoria* gets 1² / 4 = 0.25: each query word counts once, however often it occurs in the sentence, and two different query words count four times as much as one. This lesson takes the sentence with the highest query score, the first one on a tie, and cuts the paragraph into sentences with the rule of [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).

    **Bold.** A page of results puts the query words of a snippet in bold, so the eye finds them at once. This lesson writes bold as Markdown does, `**Victoria**`, and an ellipsis, …, where the snippet cuts the text.

    **The cost.** A static summary is made once, when the documents are indexed. "A dynamic summary cannot be precomputed," the book says, so it is made when the query comes in, from the text of the document, and "Generating snippets must be fast since the system is typically generating many snippets for each query that it handles." Turpin and colleagues made the ten snippets of a search in 77 milliseconds instead of 183, on a collection of 18.5 million web pages, with a compressed store of the documents.

    **Does the snippet show the answer?** For each question of XQuAD, the course knows where the answer stands in the own paragraph. So it can measure how often a snippet of the own paragraph shows the answer: a reader who sees the answer in the snippet does not even need to open the result. The picture shows the question about Victoria with the first three results of BM25, each with its window of 20 words. The own paragraph, paragraph 513, comes second. Its first 20 words stop three words before the answer, *Victoria Constitution Act 1855*, and its window shows it, as does its best sentence. Over all 1,190 questions, the first 20 words show the answer for 25.6% of the questions, the window of 20 words for 51.2%, and the window of 30 words for 61.8%. The best sentence shows it for 80.2%, with 30.9 words on average, about as many as the window of 30 words; the best sentence of paragraph 513 has 42 words.
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
    | **snippet** | the short text under each result on a page of results, with the title, so a reader can judge the result before opening it |
    | **static summary** | a summary that is the same for every query, such as the first sentences of a document |
    | **dynamic summary**, query-biased summary | a summary made for the query, from the places in the document where the query words stand |
    | **window** | a run of words in a row from a document; a snippet shows the window with the most query words |
    | **query words** | the words of a query in small letters, without stop words |
    | `re.findall(r"\S+", text)` | the text cut at the spaces: each word keeps its capitals and its punctuation |
    | `" ".join(words)` | the words glued back into one text, with a space between each two |
    | `**word**` | Markdown for a word in bold |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make a snippet of a document for a query:

    1. Cut the document at the spaces, so each word keeps how it is written.
    2. Take the query words: the words of the query in small letters, without stop words.
    3. For each window of `size` words in a row, count the words whose plain form is a query word.
    4. Take the first window with the most query words.
    5. Write its words with each query word in bold, and an ellipsis where the document goes on before or after the window.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.EXAMPLE` is the question about Victoria, `C.question_place(C.EXAMPLE, L.lang)` gives its number, `C.query_of(L.lang, k)` the query words of question `k`, and `C.results(L.lang, k)` the first three paragraphs that BM25 gives it. `C.words_of(L.lang, i)` gives the words of paragraph `i`, cut at the spaces, and `C.own_paragraph(L.lang, k)` the paragraph that holds the answer. `C.hits`, `C.best_start` and `C.snippet` are the course's versions of the three exercises. `C.span_of` gives where a snippet of a kind starts and ends: the first words (`"first"`), the window (`"window"`) or the best sentence (`"sentence"`), and `C.shows_answer` whether those words hold the answer. First the results: each paragraph, its number of words, how many query words it holds, and where its window starts. Then the three kinds of snippet of the own paragraph, and the best sentence written out. Last, `C.shares` gives how often each kind shows the answer over all the questions, with windows of 10 to 40 words:
    """)
    return


@app.cell
def example_cell():
    _k = C.question_place(C.EXAMPLE, L.lang)
    _query = C.query_of(L.lang, _k)
    print(sorted(_query))
    for _i in C.results(L.lang, _k):
        _words = C.words_of(L.lang, _i)
        print(_i, len(_words), C.hits(_words, _query), C.best_start(_words, _query, 20))
    _own = C.own_paragraph(L.lang, _k)
    for _kind in ("first", "window", "sentence"):
        _a, _b = C.span_of(L.lang, _k, _own, _kind)
        print(_kind, _a, _b, C.shows_answer(L.lang, _k, _a, _b))
    print(C.snippet(C.words_of(L.lang, _own), _query, _a, _b - _a))
    for _size in (10, 20, 30, 40):
        _shares = C.shares(L.lang, _size)
        print(_size, round(_shares["first"], 3), round(_shares["window"], 3), round(_shares["sentence"], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line holds the four query words. Then the three results: paragraph 511 has 174 words and six query words, and its window starts at word 67, counted from 0; paragraph 513 has 87 words, also six query words, and its window starts at word 11. Paragraph 526 has five query words. BM25 put paragraph 511 first, and its window, about the Port Phillip District, shows a reader that it does not answer the question.

    For the own paragraph, the first 20 words do not show the answer, and the window, from word 11 up to word 31, does. The best sentence is the first sentence of the paragraph, words 0 up to 42: the paragraph has two sentences, and both hold *parliament* and *Victoria*; on a tie the first one wins. The last four lines give, for 10 to 40 words, how often the first words, the window and the best sentence show the answer. The window shows it more often than the first words of the same length: almost twice as often at 10 words, and about one and a half times as often at 40 words. And the best sentence does not depend on a length, so its share stays at 0.802.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hits():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. hits: count the query words

        Write `hits`, which gets `words`, the words of a text cut at the spaces, as a list, and `query`, the query words as a set, in small letters. It gives back how many of the words are query words; a query word that occurs twice counts twice. In the start code, `count` starts at 0, and `for word in words` goes through the words. A word as it is written keeps its capitals and its punctuation, so *Broncos,* with its comma does not match the query word *broncos*. The box knows the course's `plain`, which gives a word in small letters, without the punctuation at its start and end: `plain("Broncos,")` gives `"broncos"`. Replace the `...` with the test: is the plain form of the word one of the query words?

        The check calls `hits(["Denver", "Broncos,", "beat", "the", "Panthers."], {"denver", "broncos", "panthers"})`, which gives `3`. And `hits(["Rijn,", "Maas", "en", "Rijn"], {"rijn"})`, which gives `2`: *Rijn* counts each time it occurs. Then cases of its own: an empty list, a query without words, a word in brackets, and the course's paragraphs.
        """),
        L.task("Count each word whose plain form is one of the query words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hits():
    box_hits = L.code_box("hits", lines=6)
    box_hits
    return (box_hits,)


@app.cell(hide_code=True)
def check_hits(box_hits):
    result_hits = L.check("hits", box_hits.value)
    result_hits
    return (result_hits,)


@app.cell(hide_code=True)
def hints_hits():
    L.hints(
        "hits",
        [
            "The test of the `if` asks whether the word, in the form that `plain` gives it, is one of the words in `query`.",
            "`x in s` asks whether `x` is in the set `s`: `\"rijn\" in {\"rijn\", \"maas\"}` gives `True`. And `plain(\"(NFC)\")` gives `\"nfc\"`.",
            "`        if ___ in query:`",
        ],
        why="`if plain(word) in query:`: the plain form is in small letters and without the punctuation at its ends, as the query words are, so *Broncos,* and *(NFC)* count too.",
    )
    return


@app.cell(hide_code=True)
def text_after_hits():
    mo.md(r"""
    In the picture, your function counts the query words of the first three results for the question about Victoria. Paragraphs 511 and 513 both hold six query words, but 511 has 174 words and 513 only 87. In 513 the query words stand closer together: its best window of 20 words holds 3 of them, against 2 in 511. The bars show your hits in each window of 20 words of paragraph 513, from the first window to the last. They reach 3 at the window that starts at word 11, and fall to 0 at the end of the paragraph, where no query word stands. The next exercise finds the first highest bar.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_start():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. best_start: find the window

        Write `best_start`, which gets `words` and `query` as in exercise 1, and `size`, the number of words of a snippet. It gives back the place, counted from 0, where the window of `size` words with the most query words starts; on a tie, the first such window. In the start code, `best` and `most` start at 0. `for start in range(len(words) - size + 1)` goes through every place where a window of `size` words fits; when the words are fewer than `size`, it goes through none, and the function gives back 0. `count = hits(..., query)` counts the query words of a window, and the `if` keeps the window when it holds more query words than the best so far. Replace the `...` with the window: the `size` words of `words` from place `start` on. The box already knows `hits` of exercise 1.

        The check calls `best_start(["The", "Panthers", "lost", "to", "the", "Denver", "Broncos."], {"denver", "broncos"}, 2)`, which gives `5`: the window *Denver Broncos.* holds both query words. And `best_start(["Rotterdam", "ligt", "aan", "de", "Maas", "en", "de", "Rijn."], {"maas", "rijn"}, 4)`, which gives `4`: only the last window of four words holds both *Maas* and *Rijn*. Then cases of its own: fewer words than `size`, a paragraph without query words, two windows with as many query words, and the course's paragraphs.
        """),
        L.task("Give `hits` the `size` words of the window that starts at `start`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_start():
    box_best_start = L.code_box("best_start", lines=7)
    box_best_start
    return (box_best_start,)


@app.cell(hide_code=True)
def check_best_start(box_best_start):
    result_best_start = L.check("best_start", box_best_start.value)
    result_best_start
    return (result_best_start,)


@app.cell(hide_code=True)
def hints_best_start():
    L.hints(
        "best_start",
        [
            "`count` is what `hits` gives for one window and `query`: the window is a slice of `words`, from place `start`, with `size` words.",
            "A slice `a[i:j]` holds the items from place `i` up to place `j`, without `j` itself: `a[2:5]` holds the three items at places 2, 3 and 4.",
            "`        count = hits(words[___], query)`",
        ],
        why="`count = hits(words[start:start + size], query)`: the slice stops just before place `start + size`, so it holds exactly `size` words.",
    )
    return


@app.cell(hide_code=True)
def text_after_best_start():
    mo.md(r"""
    In the picture, your function puts a window on each of the first three results of two questions. For Victoria, the window of paragraph 513, the second result, holds the answer itself, *Victoria Constitution Act 1855*: a reader no longer needs to open the result. For the Amazon basin, the window of the second result holds *7,000,000 square kilometres*. The windows of the other results show query words too, but a reader can see that they are about something else: paragraph 511 about the Port Phillip District, and paragraph 788 about rain in the basin long ago. Over all 1,190 questions, your window of 20 words shows the answer for 51.2% of the questions, twice as often as the first 20 words, with 25.6%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_snippet():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. snippet: write the snippet

        Write `snippet`, which gets `words` and `query` as in exercise 1, and `start` and `size` as in exercise 2, and gives back the snippet as a text: the `size` words from place `start`, with each query word in bold. Bold is written as Markdown writes it: two stars before the word and two after it, `**Denver**`. In the start code, `shown` collects the words, `for word in words[start:start + size]` goes through the words of the window, the `if` finds the query words with `plain`, and `shown.append(word)` adds each word, in bold or not. Then `text = " ".join(shown)` glues the words together with a space between each two. An ellipsis, `…`, goes before the text when the window starts after the first word, and after it when `start + size < len(words)`: the paragraph goes on. Replace the `...` with the query word in bold: the word as it is written, with two stars on each side.

        The check calls `snippet(["The", "Denver", "Broncos", "beat", "the", "Carolina", "Panthers."], {"denver", "panthers"}, 1, 3)`, which gives `"… **Denver** Broncos beat …"`: the window starts after *The* and stops before *the Carolina Panthers.* And `snippet(["De", "Rijn", "en", "de", "Maas"], {"rijn", "maas"}, 0, 5)`, which gives `"De **Rijn** en de **Maas**"`: the window holds every word, so the snippet gets no ellipsis. Then cases of its own: a window at the start or the end of the words, a window longer than the words, a word in brackets, and the course's paragraphs.
        """),
        L.task("Put two stars before and after each query word, as it is written."),
    ])
    return


@app.cell(hide_code=True)
def exercise_snippet():
    box_snippet = L.code_box("snippet", lines=12)
    box_snippet
    return (box_snippet,)


@app.cell(hide_code=True)
def check_snippet(box_snippet):
    result_snippet = L.check("snippet", box_snippet.value)
    result_snippet
    return (result_snippet,)


@app.cell(hide_code=True)
def hints_snippet():
    L.hints(
        "snippet",
        [
            "`word` becomes the word itself, as it is written, with two stars on each side.",
            "Texts glue together with `+`, in the order you write them: `\"(\" + \"Rijn\" + \")\"` gives `\"(Rijn)\"`.",
            "`            word = \"**\" + ___`",
        ],
        why="`word = \"**\" + word + \"**\"`: the word keeps its capitals and its punctuation, so *Panthers.* becomes `**Panthers.**`, with the period inside the bold.",
    )
    return


@app.cell(hide_code=True)
def text_after_snippet():
    mo.md(r"""
    In the picture, your snippets stand under the first three results of two questions, as a page of results shows them: the query words in bold, and an ellipsis where the paragraph goes on. For the question about the main role of the primes, the first result shows *The fundamental theorem of arithmetic*, the answer, with the query words *theorem*, *role*, *primes*, *number* and *theory* in bold. The question asks for the *main* role, and the paragraph says *central* role: a snippet finds the words of the query, not the words that mean the same.

    ### Try it: your own page of results

    Pick a question, a kind of snippet and a length. The picture shows the first five results of BM25, each with its snippet, says whether the snippet of the own paragraph shows the answer, and gives in a table how often each kind shows the answer over all the questions, at the length you chose. The length counts for the first words and the window; the best sentence keeps its own length, whatever length you pick.

    Start with the question about the parliament of Victoria and the window, and move the length from 5 to 40 words: from which length on does the snippet of the own paragraph show the answer, and does it keep showing it? Then pick the question about the Amazon basin, and compare the window with the best sentence: which of the two shows the answer?
    """)
    return


@app.cell(hide_code=True)
def try_choose():
    _questions = C.try_options(L.lang)
    try_question = mo.ui.dropdown(options=_questions, value=next(t for t, q in _questions.items() if q == C.EXAMPLE),
                                  label="Question")
    try_kind = mo.ui.radio(options=C.KIND_OPTIONS[L.lang],
                           value=next(t for t, k in C.KIND_OPTIONS[L.lang].items() if k == "window"), label="Snippet")
    try_size = mo.ui.slider(C.TRY_SIZES[0], C.TRY_SIZES[1], step=C.TRY_SIZES[2], value=C.SIZE, show_value=True,
                            label="Words")
    mo.vstack([try_question, try_kind, try_size])
    return try_kind, try_question, try_size


@app.cell(hide_code=True)
def try_look(try_kind, try_question, try_size):
    mo.Html(C.try_picture(L.lang, try_question.value, try_size.value, try_kind.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    For Victoria, the window shows the answer from 15 words on, but not at 25 words. A window of 25 words that starts at the start of the paragraph holds three query words too, and on a tie the first window wins; it stops at *Constitution*, just before *Act 1855*. From 30 words on, even the first words show the answer. For the Amazon basin, the window shows the answer from 10 words on, and the best sentence does not. The first sentence holds *Amazon* and *basin*, the sentence with the answer holds *basin* and *square*, and on a tie the first sentence wins. The sentence with the answer writes *kilometres*, where the question writes *kilometers*, so that query word does not count: one spelling can decide which sentence a reader gets to see.

    Is a longer snippet better? Edward Cutrell and Zhiwei Guan tracked the eyes of people who searched with snippets of one line, of two to three lines, and of six to seven lines: "We found that adding information to the contextual snippet significantly improved performance for informational tasks but degraded performance for navigational tasks." A long snippet helped to find a fact, and got in the way of finding a home page, where the address of the page says more. David Maxwell, Leif Azzopardi and Yashar Moshfeghi let 53 people search with snippets of four lengths in 2017: "Findings show that participants broadly preferred longer result summaries, as they were perceived to be more informative." And: "However, their performance in terms of correctly identifying relevant documents was similar across all four conditions."

    ### Exercise 4. Static or dynamic?

    Sort each summary below: is it the same for every query, or made for the query?
    """)
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The results here are those of [BM25: the classic ranking](/course/open/ir-bm25?lang=en), the words are cut as in [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en), and the query words leave out the stop words of [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en). A snippet is a summary that picks words from the document; [Summarizing: pick or write](/course/open/nlp-summarize?lang=en) picks the sentences of a text for a summary without a query. The next lesson, *From ten links to one answer*, looks at search engines that write one answer above the results, or in their place.

    **An everyday example.** A shop window shows a few things from the shop, chosen for the people who walk past. A static summary is a window that never changes; a dynamic summary changes the window for each person who looks, and puts in it what that person asked for.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_hits, result_best_start, result_snippet, result_kinds):
    L.wrap_up(result_hits, result_best_start, result_snippet, result_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
