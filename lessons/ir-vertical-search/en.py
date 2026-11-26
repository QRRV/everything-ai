# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ir-vertical-search. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ir-vertical-search", "en")
    C = L.checks   # the course's 24 listings, the paragraphs and questions of SQuAD, and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [BM25: the classic ranking](/course/open/ir-bm25?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_saturation():
    quick_saturation = L.choice(
        "quick-saturation",
        question="In *BM25: the classic ranking*, with k1 = 1.2, a word that occurs once in a document of average length has the term weight 1. What is its term weight when it occurs three times there?",
        options={
            "three": "3: three times as much as once",
            "right": "1.571: each repeat adds less than the one before",
            "k1": "1.2: the setting k1 itself",
            "limit": "2.2: the most that any count of a word can add, k1 + 1",
        },
        correct="right",
        why={
            "right": "The saturation: 2.2 · 3 / (1.2 + 3) = 1.571. This lesson asks a new question: when a word is in the title of a document and also in its text, should it saturate once over both, or once in each?",
            "three": "Three times as much is the count itself, as in a vector of TF-IDF weights. What did BM25 do with each repeat of a word? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "k1": "k1 sets how strongly the counts saturate, but the term weight depends on the count too. What does the saturation give for a count of 3? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "limit": "k1 + 1 is the limit that the term weight comes close to but never reaches. Does a count of 3 already reach it? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
        },
    )
    quick_saturation
    return (quick_saturation,)


@app.cell(hide_code=True)
def quick_check_tune():
    quick_tune = L.choice(
        "quick-tune",
        question="In the same lesson, k1 and b were tuned on the 809 training claims of SciFact and measured on the 300 test claims. Why not tune on the test claims?",
        options={
            "fewer": "the test claims are too few to tune two settings on",
            "right": "a setting picked on the test claims looks better there than on new claims",
            "same": "the best setting is the same on both, so it makes no difference",
            "other": "the test claims are about other abstracts, so their best setting means nothing",
        },
        correct="right",
        why={
            "right": "A setting picked on the test claims fits those claims: k1 = 0.7 and b = 1.0 gave 56.0% there, more than the 53.7% that tuning on the training claims delivered. This lesson adds settings of its own, such as how much a title counts, and the same holds for them.",
            "fewer": "300 claims are plenty to try 651 settings on. What goes wrong when you measure a setting on the claims that you picked it with? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "same": "The best setting on the test claims was k1 = 0.7 and b = 1.0, against k1 = 1.1 and b = 0.65 on the training claims. Which of the two promises too much? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
            "other": "Both sets of claims search the same 5,183 abstracts. What goes wrong when you measure a setting on the claims that you picked it with? See [BM25: the classic ranking](/course/open/ir-bm25?lang=en).",
        },
    )
    quick_tune
    return (quick_tune,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    On a site for second-hand things, you search for a tent: at most €100, not too far away, and still for sale. The course's own site has 24 listings. Each has a title, such as *Tent for four people*, a short text, a price, the days since it was placed, and its distance to you. Twelve of them hold the word *tent*. BM25 on the title and the text as one list of words puts first a sleeping bag, whose text says *tent* twice: *Warm sleeping bag for cold nights in the tent. Rolls up small, so there is room left in the tent.* Second come steel pegs. The listing you want, *Tent for four people*, says *tent* once, in its title, and has a long text: of the twelve, it comes last. Of the other tents, one costs €140, one costs €220, one is 140 km away, and one was placed three months ago.

    A search engine for the whole web treats a page mostly as words and links. A search engine for one domain knows the parts of its documents: here the title, the price, the date and the place. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze, in *Introduction to Information Retrieval* (2008), place such search engines between the web and your own computer: "In between is the space of enterprise, institutional, and domain-specific search, where retrieval might be provided for collections such as a corporation’s internal documents, a database of patents, or research articles on biochemistry." Search engines for news, for products, for places nearby, for the documents of a company and for law are such **vertical search engines**: each searches one domain, and uses what it knows about the documents of that domain.

    **An everyday example.** A library catalog lets you search by title, by author or by year, and a shoe shop lets you pick a size and sort by price. Neither treats its things as a heap of words.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_title = L.guess(
        "guess",
        question="Back to the 2,067 paragraphs of XQuAD, where BM25 put the own paragraph first for 80.0% of the 1,190 questions. Each paragraph belongs to a Wikipedia article, such as *Warsaw*. Suppose each paragraph gets the title of its article as an extra field, in which each word counts twice. For what share of the questions does BM25 now put the own paragraph first?",
        options={
            "less": "less often than without the titles",
            "right": "a little more often: under two points more",
            "more": "clearly more often: about five points more",
            "most": "nearly always: more than nine in ten of the questions",
        },
        reveal="A little more often: 80.6%, against 80.0%. An article has 43 paragraphs on average, and the title adds the same words to each of them: it could at best help to find the article, and BM25 already found the article: for 93.6% of the questions, the first paragraph was in the article of the own paragraph. Within the article, the title tells no paragraph apart, and only 35.9% of the questions hold a word of the title of their article at all. A field helps most when it tells documents apart, as the title of a listing does.",
    )
    guess_title
    return (guess_title,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Fields.** The book: "By metadata, we mean specific forms of data about a document, such as its author(s), title, and date of publication." It calls a part with free text, such as the title or the abstract, a zone, and data with a value, such as the date of creation, a field. Search engines such as Elasticsearch call both fields, and so does this lesson. A listing on the site has two text fields, the title and the text, and three fields with a value: the price, the days since it was placed, and the distance.

    **Weighing the fields.** A match in the title says more than a match in the text: a title names in a few words what the document is about. Stephen Robertson and Hugo Zaragoza write in *The Probabilistic Relevance Framework: BM25 and Beyond* (2009): "a query match on the title might be expected to provide stronger evidence of possible relevance than an equivalent match on the body text." The obvious way is to give each field its own BM25 score and add the scores up, each times the weight of its field. They argue against it: a word would then saturate in each field apart, while whether a document is about a word holds for the document as a whole. Their BM25F, in its simplest form, first adds up the counts of a word, each times the weight of its field, and saturates only that total: "The saturation function should be applied at this stage, to the total evidence for each term." With $v_s$ the weight of field $s$:

    $$\widetilde{\text{tf}}_{t,d} = \sum_s v_s \cdot \text{tf}_{t,s} \qquad \widetilde{L}_d = \sum_s v_s \cdot L_s$$

    where $\text{tf}_{t,s}$ is the count of word $t$ in field $s$ of document $d$, and $L_s$ the number of words of that field. BM25 then runs as in *BM25: the classic ranking*, with $\widetilde{\text{tf}}$ in the place of the count, and $\widetilde{L}_d$ in the place of the length, against the average $\widetilde{L}_{\text{ave}}$ over the collection. With whole weights, this is BM25 on a document in which each field is written out as many times as its weight: "an ordinary BM25 function applied to a document in which some of the streams have been replicated", where a stream is their word for a field. The IDF comes from whole documents: "we computed IDFs on the entire collection disregarding streams. This worked well in practice".

    The difference shows in a word that is once in a title with weight 3 and twice in a text, in fields of average length, with $k_1 = 1.2$. BM25F saturates the combined count, 3 · 1 + 2 = 5, to 1.774, below $k_1 + 1 = 2.2$. A BM25 score per field would give 3 · 1 + 1.375 = 4.375, as if the title and the text were each about the word on their own, and it could grow to 3 · 2.2 + 2.2 = 8.8, where BM25F stays below 2.2. The weights are settings, like $k_1$ and $b$. The full BM25F of Robertson and Zaragoza also gives each field its own $b$; for one of the web search tasks of TREC in 2003, they report tuned weights of 38.4 for the title, 1.0 for the body text and 35 for the anchor text of the links to a page, with $k_1 = 27.5$. This lesson keeps one $b$, and gives the title the weight 3.

    **Filters.** The book: "Consider queries of the form “find documents authored by William Shakespeare in 1601, containing the phrase alas poor Yorick.”" A field with a value lets a search engine keep only the documents whose value fits, such as a price of at most €100 and a distance of at most 50 km. A filter does not change the score of a document: it keeps the document or drops it.

    **Decay.** A filter has a hard edge: a tent at 50 km stays, a tent at 51 km is gone. For age and distance, a smooth edge often fits better: a listing of a week ago still counts, a little less than one of today. Elasticsearch, a search engine that many sites run on, has decay functions for this: "Decay functions score a document with a function that decays depending on the distance of a numeric field value of the document from a user given origin. This is similar to a range query, but with smooth edges instead of boxes." This lesson uses the decay that halves the score at a distance $h$:

    $$\text{decay}(x) = 0.5^{\,x / h}$$

    where $x$ is the distance from the best value, such as the age in days or the distance in km. At $x = 0$ the factor is 1, at $h$ it is 0.5, at $2h$ it is 0.25. Elasticsearch writes its exponential decay with $e$, a setting *decay* and a *scale*; with decay 0.5 and scale $h$, it is this formula. The final score is the BM25F score times the decay of the days, with $h$ = 14 days, times the decay of the km, with $h$ = 25 km. Elasticsearch multiplies too, unless told otherwise: "query score and function score are multiplied (default)". Robertson and Zaragoza mention such data as well: "Things like the age of a file, its type or its link in-degree may provide useful information about the relevance of a document."

    **Each domain its own.** News is about what is happening now, so freshness weighs heavily. A shop filters on the fields of its products, such as a price and a size. For places nearby, the distance decides much of the order; Google writes about its local results: "Local results are mainly based on relevance, distance, and popularity." In a company, not every document is meant for everyone. The book: "Users’ results lists must not contain documents they are barred from opening; the very existence of a document can be sensitive information." And law has habits of its own. The book describes Westlaw, a legal search service: "Note the long, precise queries and the use of proximity operators, both uncommon in web search." A proximity operator asks for words in the same sentence, or within a number of words of each other. The book adds: "Westlaw returns documents in reverse chronological order, which is in practice quite effective", and "In 2007, the majority of law librarians still seem to recommend terms and connectors for high recall searches": Boolean queries that should miss nothing.

    The picture shows the query *tent* in four steps: BM25 on the title and the text as one list of words, the title three times, the filters of at most €100 and at most 50 km, and the decay. *Tent for four people* climbs from the last place of the twelve to fifth, second and first.
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
    | **vertical search engine** | a search engine for one domain, such as news, products, places nearby, the documents of a company or law, that uses what it knows about the documents of that domain |
    | **field** | a part of a document that a search engine knows apart: a text, such as the title, or a value, such as a price or a date |
    | **field weight** | how many times a word in a field counts, against a word in a field with weight 1 |
    | **BM25F** | BM25 for documents with fields: the counts of a word in the fields, each times the weight of its field, are added up before they saturate |
    | **filter** | a condition on a field that keeps a document or drops it, without changing its score |
    | **decay** | a factor from 1 down to 0 that a score is multiplied by, smaller the further a value is from the best one, such as the age of a document or its distance |
    | $h$ | the distance at which the decay halves a score, such as 14 days or 25 km |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To search documents with fields, as a vertical search engine does:

    1. Split each document into fields: texts, such as the title and the text, and values, such as the price, the age and the distance.
    2. Drop the documents whose values do not pass the filters.
    3. For each word of the query, add up its counts in the text fields, each times the weight of its field, and add up the lengths of the fields in the same way.
    4. Give each document its BM25 score with these counts and lengths, and with the IDF over whole documents: BM25F.
    5. Multiply the score by the decay of each value that should count smoothly, such as the age and the distance.
    6. Sort the documents by their final score, the highest first.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.fields(L.lang, i)` gives the text fields of listing `i` as lists of words, and `C.listing(L.lang, i)` the whole listing, with its price, days and km. `C.combined_count` and `C.combined_length` give the count and the length of BM25F, `C.term_weight` the term weight of *BM25: the classic ranking*, and `C.decay` the decay. `C.average_length` gives the average combined length of the 24 listings, and `C.IDF` the IDF of each word. The query is *tent*, and the title counts three times:
    """)
    return


@app.cell
def example_cell():
    _weights = {"title": 3, "text": 1}
    _average = C.average_length(L.lang, _weights)
    _idf = C.IDF[L.lang]["tent"]
    print(round(_idf, 3), round(_average, 3))
    for _i in (0, 1):
        _fields = C.fields(L.lang, _i)
        _listing = C.listing(L.lang, _i)
        _count = C.combined_count("tent", _fields, _weights)
        _length = C.combined_length(_fields, _weights)
        _score = _idf * C.term_weight(_count, _length, _average, 1.2, 0.75)
        _final = _score * C.decay(_listing["days"], 14) * C.decay(_listing["km"], 25)
        print(_listing["title"], _count, _length, round(_score, 3), round(_final, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *Tent* is in 12 of the 24 listings, so its IDF is log10(24 / 12) = 0.301, and with the title three times, a listing has 19.542 words on average. *Tent for four people* holds *tent* once, in its title of 4 words, so its count is 3 and its length 3 · 4 + 17 = 29: its score is 0.429. *Sleeping bag* holds *tent* twice, in its text of 20 words: count 2, length 3 · 2 + 20 = 26, score 0.379. As one list of words, the sleeping bag won, 0.347 against 0.241. Then the decay: the tent was placed 3 days ago and is 12 km away, so its score is multiplied by 0.862 and by 0.717, to 0.265; the sleeping bag, 8 days old and 20 km away, ends at 0.146.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_combined_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. combined_count: the count of BM25F

        Write `combined_count`, which gets `word`, a word of the query, `fields`, a dictionary with the words of each text field of a document, such as `{"title": ["tent", "for", "two"], "text": ["fits", "in", "a", "tent"]}`, and `weights`, a dictionary with the weight of each field, such as `{"title": 3, "text": 1}`. It gives back the count of BM25F: for each field, the count of the word in the words of the field, times the weight of the field, added up. In the start code, `count` starts at 0, and the loop goes through each `name` of `fields`. Replace the `...` with the new `count`: the count so far, plus the weight of the field times the count of `word` in its words. `weights[name]` gives the weight of the field, and `fields[name]` its words: a list, with the method `.count` that every list has.

        The check calls `combined_count("tent", {"title": ["tent", "for", "two"], "text": ["a", "small", "tent", "in", "a", "tent", "bag"]}, {"title": 3, "text": 1})`, which gives `5`, and `combined_count("pomp", {"title": ["luchtbed", "voor", "twee"], "text": ["met", "een", "pomp"]}, {"title": 3, "text": 1})`, an air mattress of the Dutch lesson, which gives `1`. Then cases of its own: a word in no field, a title with weight 0, three fields, a weight that is not a whole number, no fields at all, and listings of the course.
        """),
        L.task("Add up, for each field, its weight times the count of the word in its words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_combined_count():
    box_combined_count = L.code_box("combined_count", lines=5)
    box_combined_count
    return (box_combined_count,)


@app.cell(hide_code=True)
def check_combined_count(box_combined_count):
    result_combined_count = L.check("combined_count", box_combined_count.value)
    result_combined_count
    return (result_combined_count,)


@app.cell(hide_code=True)
def hints_combined_count():
    L.hints(
        "combined_count",
        [
            "Each round adds the part of field `name` to `count`: the weight of the field, from `weights`, times how often `word` occurs in the words of the field.",
            "The words of the field are a list. Which method of a list counts how often a value occurs in it?",
            "`        count = count + weights[name] * ___`",
        ],
        why="`count = count + weights[name] * fields[name].count(word)`: with the title weight 3, a word that is once in the title counts as 3, before BM25 saturates the total.",
    )
    return


@app.cell(hide_code=True)
def text_after_combined_count():
    mo.md(r"""
    In the picture, *Tent for four people* gets the count 3 and the sleeping bag the count 2, and BM25F puts the tents first: *Tent for two people*, the new tent for two people, the tent for three people, the large tent for six people, and then *Tent for four people*. Its long text still costs it a little: the title counts three times, but so does the length of the title.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. keep: the filters

        Write `keep`, which gets `listing`, a listing as a dictionary, such as `{"title": "Tent for two people", "price": 45, "days": 5, "km": 140}`, `max_price`, the highest price, and `max_km`, the largest distance. It gives back `True` when the price of the listing is at most `max_price` and its distance at most `max_km`, and otherwise `False`. `listing["price"]` gives the price and `listing["km"]` the distance. Replace the `...` with the two conditions, joined by `and`.

        The check calls `keep({"title": "Tent for two people", "price": 45, "days": 5, "km": 140}, 100, 50)`, which gives `False`, and `keep({"title": "Tent voor vier personen", "price": 85, "days": 3, "km": 12}, 85, 50)`, the tent for four people of the Dutch lesson, which gives `True`: a price of exactly the highest price still passes. Then cases of its own: a distance of exactly the largest distance, a price that is too high, other limits, and listings of the course.
        """),
        L.task("Keep a listing when its price is at most max_price and its distance at most max_km."),
    ])
    return


@app.cell(hide_code=True)
def exercise_keep():
    box_keep = L.code_box("keep", lines=3)
    box_keep
    return (box_keep,)


@app.cell(hide_code=True)
def check_keep(box_keep):
    result_keep = L.check("keep", box_keep.value)
    result_keep
    return (result_keep,)


@app.cell(hide_code=True)
def hints_keep():
    L.hints(
        "keep",
        [
            'The price, `listing["price"]`, must be at most `max_price`, and the distance must be at most `max_km`. Both must hold, so join the two with `and`.',
            "*At most* includes the limit itself. Which comparison gives `True` when the two numbers are equal?",
            '`    return listing["price"] <= max_price and ___`',
        ],
        why='`return listing["price"] <= max_price and listing["km"] <= max_km`: a listing stays only when it passes both filters, and a price of exactly the highest price still passes.',
    )
    return


@app.cell(hide_code=True)
def text_after_keep():
    mo.md(r"""
    In the picture, three tents drop out: the new tent for two people costs €140, the large tent for six people €220, and *Tent for two people* is 140 km away. Of the twelve listings with *tent*, nine are left, and the tent for three people now comes first: the filters do not look at its age.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_decay():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. decay: smooth edges

        Write `decay`, which gets `distance`, how far a value is from the best one, such as the age of a listing in days, and `half`, the distance at which the score is halved. It gives back 0.5 to the power of the distance divided by `half`: 1 at distance 0, 0.5 at `half`, 0.25 at twice `half`. In Python, `**` raises a number to a power: `0.5 ** 2` gives `0.25`. With only `**` and `/`, the function also works when `distance` is a NumPy array. Replace the `...` with the decay.

        The check calls `decay(14, 14)`, which gives `0.5`, and `decay(50, 25)`, which gives `0.25`. Then cases of its own: a distance of 0, distances that are not a multiple of `half`, a `half` that is not a whole number, and a NumPy array of distances.
        """),
        L.task("Give 0.5 to the power of the distance divided by half."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decay():
    box_decay = L.code_box("decay", lines=3)
    box_decay
    return (box_decay,)


@app.cell(hide_code=True)
def check_decay(box_decay):
    result_decay = L.check("decay", box_decay.value)
    result_decay
    return (result_decay,)


@app.cell(hide_code=True)
def hints_decay():
    L.hints(
        "decay",
        [
            "The result is `0.5` to a power, and the power is the distance against `half`: how many times `half` fits in `distance`.",
            "Python raises to a power before it divides, so put the division that makes the power in parentheses.",
            "`    return 0.5 ** (___)`",
        ],
        why="`return 0.5 ** (distance / half)`: every `half` of distance halves the score, however far the value is.",
    )
    return


@app.cell(hide_code=True)
def text_after_decay():
    mo.md(r"""
    In the picture, a listing of 14 days old keeps half its score, and the tent for three people, 90 days old, keeps 0.012 of it. *Tent for four people*, 3 days old and 12 km away, keeps 0.862 · 0.717 of its score, and comes first: 0.265, against 0.198 for the steel pegs. Only one tent stays near the top: the tent for three people, 90 days old, sinks to the last place, and after *Tent for four people* come things for a tent, such as the pegs and the stove.

    ### Try it: your own search engine

    Move the sliders: the weight of the title, the highest price, the largest distance, and the days and the km after which the decay halves a score. Set the highest price to €300: which tent comes first with a short $h$ for the days, such as 1, and which with a long one, such as 60? What happens with a title weight of 1, as in plain BM25? And with $h$ = 5 km?
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    try_weight = mo.ui.slider(1, 10, step=1, value=C.TRY_START["weight"], label="Title weight", show_value=True)
    try_price = mo.ui.slider(10, 300, step=10, value=C.TRY_START["price"], label="Highest price (€)", show_value=True)
    try_km = mo.ui.slider(5, 200, step=5, value=C.TRY_START["km"], label="Largest distance (km)", show_value=True)
    try_half_days = mo.ui.slider(1, 60, step=1, value=C.TRY_START["half_days"], label="h for the days", show_value=True)
    try_half_km = mo.ui.slider(5, 100, step=5, value=C.TRY_START["half_km"], label="h for the km", show_value=True)
    mo.vstack([try_weight, mo.hstack([try_price, try_km], justify="start", gap=2),
               mo.hstack([try_half_days, try_half_km], justify="start", gap=2)])
    return try_half_days, try_half_km, try_km, try_price, try_weight


@app.cell(hide_code=True)
def try_look(try_weight, try_price, try_km, try_half_days, try_half_km):
    mo.Html(C.try_picture(L.lang, try_weight.value, try_price.value, try_km.value, try_half_days.value, try_half_km.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    With $h$ = 5 km, *Tent for four people* only just stays first: 0.070, against 0.068 for the camping stove, 9 km away: a decay that is too strong lets the distance count for nearly as much as the words. With the title weight 1, the steel pegs come first. With a highest price of €300, $h$ = 1 day puts the new tent for two people first, placed 2 days ago, and $h$ = 60 days the large tent for six people, 3 km away. No setting is right for every search: a vertical search engine picks its weights and its $h$ by tuning them on searches with known good results, as *BM25: the classic ranking* tuned k1 and b, or it lets the searcher choose, as these sliders do.

    ### Exercise 4. What does each domain need?

    Sort each statement below into the domain whose search engine it describes.
    """)
    return


@app.cell(hide_code=True)
def exercise_domains():
    sort_domains = L.sort("domains")
    sort_domains
    return (sort_domains,)


@app.cell(hide_code=True)
def check_domains(sort_domains):
    result_domains = L.check("domains", sort_domains)
    result_domains
    return (result_domains,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** BM25F is the BM25 of [BM25: the classic ranking](/course/open/ir-bm25?lang=en) on a document with weighted fields, and a filter keeps documents by a value, where the AND of [The inverted index](/course/open/ir-inverted-index?lang=en) kept them by their words. In the domain *Building with AI*, [Filters and vector databases](/course/open/build-metadata-filters?lang=en) filters the texts that an assistant searches in the same way. The next lesson, *Did the search work? P@k, MAP and MRR*, measures a ranking in more ways than its first place alone.

    **An everyday example.** A friend who helps you find a second-hand tent looks at the tents first, not at the pegs, skips the ones above your budget, and calls first about the one that was placed this morning around the corner.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_combined_count, result_keep, result_decay, result_domains):
    L.wrap_up(result_combined_count, result_keep, result_decay, result_domains)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
