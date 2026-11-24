# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-topic-models. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-topic-models", "en")
    C = L.checks   # the paragraphs of 48 Wikipedia pages in English and Dutch, the course's topic models, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [A document as a vector](/course/open/nlp-document-vectors?lang=en), and one about [Probability: counting how often](/course/open/prob-basics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_matrix():
    quick_matrix = L.choice(
        "quick-matrix",
        question="In *A document as a vector*, a collection of documents becomes a document-term matrix. What does the number in row 3, column 7 tell you? Rows and columns count from 0.",
        options={
            "swapped": "How often word 3 occurs in document 7",
            "right": "How often word 7 occurs in document 3",
            "documents": "In how many documents word 7 occurs",
            "share": "How much of document 3 is about topic number 7",
        },
        correct="right",
        why={
            "right": "Each row is a document and each column a word of the vocabulary, so the number counts how often word 7 occurs in document 3. A topic model starts from such a matrix and from nothing else: not the order of the words, and not the names of the documents.",
            "swapped": "Row and column the other way around: which of the two stands for the documents? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "documents": "That is one number per word for the whole collection, and the matrix holds a number for each document and word. What does one cell count? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
            "share": "Shares of topics are what this lesson builds; the matrix holds something simpler. What does one cell of it count? See [A document as a vector](/course/open/nlp-document-vectors?lang=en).",
        },
    )
    quick_matrix
    return (quick_matrix,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="A list gives the words *oven*, *pasta* and *wind* the weights 6, 3 and 1. Turn the weights into a probability distribution over the three words: what probability does *oven* get?",
        options={
            "weight": "6, its weight itself",
            "equal": "1/3, one of three words",
            "right": "0.6",
            "half": "0.5, twice as much as *pasta*",
        },
        correct="right",
        why={
            "right": "The probabilities of a distribution add up to 1, so you divide each weight by the sum of the weights, 6 + 3 + 1 = 10: *oven* gets 0.6, *pasta* 0.3 and *wind* 0.1. A topic model keeps such weights for every topic.",
            "weight": "A probability lies between 0 and 1, and the three add up to 1. What do you divide each weight by? See [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "equal": "Equal probabilities would leave out the weights. How do 6, 3 and 1 become three numbers that add up to 1? See [Probability: counting how often](/course/open/prob-basics?lang=en).",
            "half": "With 0.5 for *oven*, *pasta* gets 0.25 and *wind* about 0.08: together less than 1. What should the three add up to? See [Probability: counting how often](/course/open/prob-basics?lang=en).",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A collection can be too large to read just to find out what it is about: the letters in an archive, the articles of a journal, the answers to an open question in a survey. Blei (2012) describes what a topic model does: "Topic models are algorithms for discovering the main themes that pervade a large and otherwise unstructured collection of documents." Blei ran one with 100 topics on 17,000 articles from the journal *Science*. Among its topics was one whose most probable words were *human*, *genome*, *dna*, *genetic* and *genes*, and one with *evolution*, *evolutionary*, *species*, *organisms* and *life*. Blei adds: "We emphasize that the algorithms have no information about these subjects and the articles are not labeled with topics or keywords."

    This lesson uses 2,067 paragraphs from 48 pages of the English Wikipedia: the pages of the development set of SQuAD (Rajpurkar and others, 2016), a collection of questions about Wikipedia, with their Dutch from SQuAD-NL (de Vries and others, 2023). Most of the Dutch is machine translation: people corrected 240 of the paragraphs. The pages are about very different things, from *Super Bowl 50* and *Warsaw* to *Oxygen* and *Prime number*. You know which page each paragraph comes from, so you can check what a topic model finds. The model itself never sees the names of the pages.

    **An everyday example.** You go through a pile of old emails without folders. Some are about work, some about family, and a few about both, such as the one asking for a day off for a wedding. Nobody told you which themes there are: you found them by reading, and you named them yourself.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_mix = L.guess(
        "guess",
        question="The topic model of this lesson gives each of the 2,067 English paragraphs a mix of 12 topics: 12 shares that add up to 1. In how many paragraphs does one topic have a share of at least half?",
        options={
            "few": "About one in ten: most paragraphs spread over many topics",
            "third": "About three in ten",
            "right": "About seven in ten",
            "all": "Nearly all, since each paragraph is from one page",
        },
        reveal="**1,501 of the 2,067: about seven in ten.** The largest share of a paragraph is 0.65 on average. A paragraph of a page is mostly about one thing, and LDA, the model of this lesson, prefers mixes with few topics. The other three in ten mix: often a page that shares its topic with other pages, or a paragraph that stands between two subjects.",
    )
    guess_mix
    return (guess_mix,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **topic** is a probability distribution over the words of a vocabulary: every word has a probability in every topic, and a topic about cooking gives *oven* and *pasta* a high one. Blei: "We formally define a topic to be a distribution over a fixed vocabulary." A **topic model** describes a collection with a fixed number of topics, $k$, and gives every document a **mix**: a share for each topic, $k$ numbers that add up to 1. In Blei's words, "all the documents in the collection share the same set of topics, but each document exhibits those topics in different proportion."

    The best-known topic model is **latent Dirichlet allocation**, **LDA**, by Blei, Ng and Jordan (2003): "The basic idea is that documents are represented as random mixtures over latent topics, where each topic is characterized by a distribution over words." LDA imagines how each document was written, in two steps:

    1. Choose a mix for the document.
    2. For each word of the document: choose a topic by the mix, then a word by that topic.

    Nobody writes that way, and LDA does not claim it: the story only says which counts are likely. LDA works it backwards. It gets the document-term matrix, the counts of the words in each document and nothing else, and looks for topics and mixes under which those counts are likely. Blei: "This can be thought of as “reversing” the generative process—what is the hidden structure that likely generated the observed collection?" The topics are **latent**: hidden, since nobody wrote them down. The mixes come from a **Dirichlet** distribution, which has a parameter **α** (alpha): the smaller α, the fewer topics a mix tends to use. scikit-learn sets α to $1/k$ unless you choose another.

    The order of the words does not matter to LDA, only how often each one occurs, as in the bag of words of *A document as a vector*. And LDA numbers its topics, from 0 to $k - 1$; the names are up to you. Blei, Ng and Jordan show topics of a model with 100 topics of 16,000 news articles: "As we have hoped, these distributions seem to capture some of the underlying topics in the corpus (and we have named them according to these topics)."

    The picture shows six short documents and what LDA makes of them with two topics: the probability of each word in each topic, and the mix of each document.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    LDA got only the counts. It found a kitchen topic, topic 0, with *oven*, *pasta* and *pizza*, and a weather topic, topic 1, with *wind*, *storm* and *mist*. Documents 0 to 3 each use the words of one topic, and their mix is 0.90 for that topic: not 1, since α keeps a little for the other. Document 4 has two words of each, and its mix is half and half. Document 5, *mist wind pasta*, leans to the weather, 0.64; still *pasta* in it gets topic 0, since *pasta* is likely in the kitchen topic only.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | Meaning |
    |---|---|
    | **topic** | a probability distribution over the words of a vocabulary |
    | **topic model** | a model that describes a collection with $k$ topics, and every document with a mix of them |
    | **mix** | the shares of the $k$ topics in one document: $k$ numbers that add up to 1 |
    | $k$ | the number of topics, which you choose |
    | **latent Dirichlet allocation** (LDA) | the best-known topic model: each document a mix drawn from a Dirichlet distribution, each word from a topic chosen by the mix |
    | **latent** | hidden: not in the data, found by the model |
    | **α** (alpha) | the parameter of the Dirichlet distribution of the mixes: the smaller, the fewer topics a mix tends to use |
    | `LatentDirichletAllocation(n_components=k, random_state=0)` | scikit-learn's LDA with `k` topics; `random_state` fixes where it starts, so a run gives the same topics again |
    | `lda.fit_transform(counts)` | trains the model on a document-term matrix and gives the mix of every document: a row per document, a column per topic |
    | `lda.components_` | the weights of the topics: a row per topic, a column per word; each row divided by its sum gives the topic as probabilities |
    | `np.argsort(row)` | the places of the values of `row` from the smallest value to the largest, counting from 0: `np.argsort(np.array([0.2, 0.5, 0.1]))` gives `array([2, 0, 1])` |
    | `C.count_matrix(docs)` | the document-term matrix of short documents of words split by spaces, as an array of counts, and its words in alphabetical order |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Count the words of each document: the document-term matrix. Leave out stop words, and words that occur in very few or very many documents.
    2. Choose $k$, and train LDA on the counts.
    3. Read each topic from its words with the highest probability.
    4. Read each document from its mix, and each word from the topic it most likely comes from.
    5. Give the topics names yourself, and check them against what the documents are about.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code counts the six documents of the picture, trains LDA with two topics, and prints the words of the columns, the weights of the two topics and the mix of each document, rounded to two decimals.
    """)
    return


@app.cell
def example_cell():
    from sklearn.decomposition import LatentDirichletAllocation

    _docs = ["oven pizza pasta oven", "wind storm mist wind", "pizza pasta oven pasta",
             "storm mist wind storm", "oven pizza wind storm", "mist wind pasta"]
    _counts, _words = C.count_matrix(_docs)
    _lda = LatentDirichletAllocation(n_components=2, random_state=0)
    _mix = _lda.fit_transform(_counts)
    print(_words)
    print(_lda.components_.round(2))
    print(_mix.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The weights are not probabilities yet: the first row adds up to 13.94, the second to 15.06. scikit-learn's documentation says what they are: "components_[i, j] can be viewed as pseudocount that represents the number of times word j was assigned to topic i." The six documents have 23 words, and LDA adds a small count of its own for every word in every topic, scikit-learn's `topic_word_prior`, 1/k unless you choose another, so 0.5 here: 23 + 12 × 0.5 = 29 = 13.94 + 15.06. Divided by the sum of its row, a weight becomes a probability: *oven* gets 4.46 / 13.94 = 0.32 in topic 0, as in the picture.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: LDA on 2,067 paragraphs

    The course counts the words of the 2,067 English paragraphs: 4,799 words of three letters or more that occur in at least 5 paragraphs and in at most a tenth of them, without stop words: 108,680 counted words in all. Then it trains LDA with 12 topics for 20 steps. In each step, LDA goes over all the paragraphs once and improves the topics and the mixes.

    The chart shows how surprised the model is by the words of the paragraphs. As in [How good is a language model?](/course/open/lm-nll?lang=en), the surprise of a word is minus the log of its probability, and the chart gives the mean per word, as scikit-learn estimates it. The dashed line is a model without topics that gives each of the 4,799 words the same probability: log 4,799 = 8.476. Beside the chart you see the 6 words with the highest probability in each topic, as they are now.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_button = mo.ui.run_button(label="Train LDA on the paragraphs, 20 steps")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_run(demo_button):
    if not demo_button.value:
        _out = mo.md("*Press the button to train.*")
    else:
        _now = {}

        def _progress(_step, _surprise, _tops):
            if "chart" not in _now:
                _words = C.live_words(L.lang)
                _now["chart"] = L.live(_words, C.STEPS, low=C.LOW, high=C.HIGH)
                _now["chart"].reference(C.uniform_line(L.lang), _words["uniform"])
                _now["words"] = _words
            _now["chart"].watch(lambda: C.now_parts(_tops, L.lang))
            _now["chart"].add(_step, _surprise, _now["words"]["progress"].format(s=_step, n=C.STEPS, v=L.number(_surprise, 3)))

        mo.output.replace(mo.md("*Counting the words of the paragraphs …*"))
        _run = C.demo(L.lang, _progress)
        if isinstance(_run, str):
            _out = _now["chart"].stopped(C.slow_note(L.lang)) if "chart" in _now else mo.md(C.slow_note(L.lang))
        else:
            _out = mo.vstack([
                mo.md(f"""After {C.STEPS} steps, the mean surprise per word is {L.number(_run["curve"][-1][1], 3)}, against {L.number(_run["uniform"], 3)} without topics. These are the course's topics: the exercises below use this same model."""),
                L.explore("rewind", "step", lambda _s: (C.rewind_frame(_run, int(_s), L.lang), C.rewind_said(_run, int(_s), L.lang)),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Go back to step 1 and look for *abc*: in how many topics is it among the 6 words? Then step forward: from which step on is it in one topic only, and which words stand beside it there?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    After the first step, LDA has hardly started: *abc*, from the page *American Broadcasting Company*, stands in three topics at once, and the surprise per word, 8.452, is barely below the dashed line. From step 3 on, *abc* stands in one topic, with *television*, and the surprise goes down quickly at first, then more and more slowly: from step 10 to step 20, it falls by only 0.041. Most topics barely change after step 10. Training longer would lower the surprise a little more, but surprise is not the same as meaning; after exercise 4, the lesson comes back to that. This is one run, with `random_state` 0. With `random_state` 1, LDA finds other topics: not one of the 12 has the same five words as here.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. topic_words: the words of each topic

        Write `topic_words(weights, words, n)`, where `weights` is an array with a row per topic and a column per word, such as `components_`, and `words` is the list of the words of the columns. Give back a list with, for each topic, a list of its `n` words with the highest probability, the highest first, each as a pair `(word, probability)`. The probability of a word in a topic is its weight divided by the sum of the weights of the topic's row. Replace the two `...`.

        The check calls `topic_words(small_weights, small_words, 3)`, with two topics over the six words `["mist", "oven", "pasta", "pizza", "storm", "wind"]` and the weights `[[0.5, 4.0, 3.0, 2.0, 0.4, 0.1], [2.0, 0.2, 0.3, 0.1, 1.8, 3.6]]` (in the lesson: `C.SMALL_WORDS` and `C.SMALL_WEIGHTS`). It gives `[[("oven", 0.4), ("pasta", 0.3), ("pizza", 0.2)], [("wind", 0.45), ("mist", 0.25), ("storm", 0.225)]]`. Then cases of its own: the same with `n` 1; three topics over four words with `n` 2; one topic over three words with `n` 3; two topics over two words with `n` 1; and the course's model of the English and of the Dutch paragraphs with `n` 5.
        """),
        L.task("Find the words with the highest probability in each topic."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("topic_words", lines=11)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("topic_words", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "topic_words",
        [
            "`probs` is `weights` divided by the sum of each row: `weights.sum` along `axis=1`. `best` holds the places of the `n` largest values of `row`: `np.argsort` sorts from the smallest value to the largest, so sort `-row` and keep the first `n`.",
            "`weights.sum(axis=1)` gives one sum per topic, as a flat array, and `weights` has a row per topic and a column per word. Which argument of `sum`, from *Broadcasting*, keeps the sums as a column, so that each row is divided by its own sum?",
            "`    probs = weights / weights.sum(axis=1, ___)`",
        ],
        why="`probs = weights / weights.sum(axis=1, keepdims=True)` and, in the loop, `best = np.argsort(-row)[:n]`. The example gives `[[(\"oven\", 0.4), (\"pasta\", 0.3), (\"pizza\", 0.2)], [(\"wind\", 0.45), (\"mist\", 0.25), (\"storm\", 0.225)]]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_words():
    mo.md(r"""
    You can name some topics at a glance: topic 0 with *force*, *forces* and *motion*, topic 3 with *cells*, *oxygen* and *immune*. Others need a second look. Topic 1 has *british*, *bowl*, *super*, *inequality* and *plague*: two or three pages in one topic. The probabilities are small, 0.035 at most, since every topic spreads its probability over 4,799 words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_topics():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. word_topics: the topic of each word

        In LDA's story, each word of a paragraph comes from one topic. Which one? By [Bayes' rule](/course/open/prob-bayes?lang=en), the probability that a word came from topic $t$ is proportional to the share of $t$ in the paragraph's mix times the probability of the word in $t$. Write `word_topics(mix, topics, ids)`, where `mix` is the mix of a paragraph, an array of $k$ shares; `topics` is an array of probabilities with a row per topic and a column per word; and `ids` is the list of the columns of the paragraph's words, in the order of its text, counting from 0. Give back a list with, for each word, the number of the topic with the largest product. Topics are numbered from 0. Replace the `...`.

        The check calls `word_topics(small_mix, small_topics, [1, 2, 0, 5])`, with the mix `[0.9, 0.1]`, the topics of exercise 1 as probabilities (in the lesson: `C.SMALL_MIX` and `C.SMALL_TOPICS`), and the words *oven*, *pasta*, *mist* and *wind*. It gives `[0, 0, 0, 1]`. *mist* is more likely in the weather topic, 0.25 against 0.05, but this paragraph is 90% kitchen: 0.9 × 0.05 = 0.045 against 0.1 × 0.25 = 0.025. Then cases of its own: a mix of half and half; the mix `[0.3, 0.7]` with a word twice; no words at all; three topics over four words; and paragraph 3 of the English and of the Dutch paragraphs, counting from 0, with the course's model.
        """),
        L.task("Give each word of a paragraph the topic it most likely comes from."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topics():
    box_topics = L.code_box("word_topics", lines=9)
    box_topics
    return (box_topics,)


@app.cell(hide_code=True)
def check_topics(box_topics):
    result_topics = L.check("word_topics", box_topics.value)
    result_topics
    return (result_topics,)


@app.cell(hide_code=True)
def hints_topics():
    L.hints(
        "word_topics",
        [
            "For word `j`, multiply `mix` by the probabilities of word `j` in every topic. `np.argmax` gives the place of the largest product, `int` turns it into a plain number, and `found.append` adds it to the list.",
            "`topics` has a row per topic and a column per word, so `topics[j]` is a row: the probabilities of all words in topic `j`. You need the probabilities of word `j` in each topic. How do you take column `j` of an array, from every row?",
            "`        found.append(int(np.argmax(mix * ___)))`",
        ],
        why="In the loop: `found.append(int(np.argmax(mix * topics[:, j])))`. The example gives `[0, 0, 0, 1]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_topics():
    mo.md(r"""
    Paragraph 3 is about how Super Bowl 50 was shown on television, and its mix is 0.56 topic 1 and 0.41 topic 9. *CBS*, *broadcast* and *commercial* come from topic 9, the topic of *abc* and *television*; *Super*, *Bowl* and *game* from topic 1, the topic of *bowl* and *super*. A paragraph is not about one topic: each of its words leans one way or the other. *British* goes to topic 1 too, though here it describes the band Coldplay. LDA only knows that *british* and *bowl* occur in the same topic.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pages():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. topic_pages: the pages of a topic

        You know which page each paragraph comes from; LDA does not. Write `topic_pages(mix, pages, topic)`, where `mix` is an array with the mix of every paragraph, a row per paragraph; `pages` is the list of the names of their pages; and `topic` is the number of a topic. Give back a dictionary with, for each page, how many of its paragraphs have `topic` as their largest share. A page without such paragraphs is left out. Replace the `...`.

        The check calls `topic_pages(small_mixes, small_pages, 0)`, with five paragraphs whose mixes are `[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7], [0.7, 0.3]]` and whose pages are `["Pizza", "Storm", "Pizza", "Mist", "Pasta"]` (in the lesson: `C.SMALL_MIXES` and `C.SMALL_PAGES`). It gives `{"Pizza": 1, "Pasta": 1}`. Then cases of its own: topic 1 of the same; three topics over four paragraphs of the pages *Radio* and *Film*, topic 0; a topic that no paragraph has as its largest share; topic 9 of the English and topic 5 of the Dutch paragraphs with the course's model.
        """),
        L.task("Count, for each page, the paragraphs whose largest share is one topic."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pages():
    box_pages = L.code_box("topic_pages", lines=12)
    box_pages
    return (box_pages,)


@app.cell(hide_code=True)
def check_pages(box_pages):
    result_pages = L.check("topic_pages", box_pages.value)
    result_pages
    return (result_pages,)


@app.cell(hide_code=True)
def hints_pages():
    L.hints(
        "topic_pages",
        [
            "In the loop: an `if` that keeps only the paragraphs whose largest share is `topic`. Under it, the counting of a dictionary: `if pages[i] in counts:` adds 1 to `counts[pages[i]]`, and `else:` gives it 1.",
            "Which topic has the largest share in paragraph `i`? It is a place in the row `mix[i]`: the place of its largest value. Which NumPy function from exercise 2 gives that place?",
            "`        if ___ == topic:`",
        ],
        why="In the loop: `if np.argmax(mix[i]) == topic:` with, under it, `if pages[i] in counts:` and `counts[pages[i]] = counts[pages[i]] + 1`, and `else:` with `counts[pages[i]] = 1`. The example gives `{\"Pizza\": 1, \"Pasta\": 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pages():
    mo.md(r"""
    Topic 9 holds 97 paragraphs of *American Broadcasting Company* and 55 of *Doctor Who*: television. And 40 of *Chloroplast*, about the parts of a plant cell that hold its chlorophyll. Exercise 4 comes back to that.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: name the topics

    LDA gives numbers; the names are yours. Write a name after each number, one per line, such as `0: forces and motion`. The picture shows your names beside the words of each topic and the pages with the most paragraphs in it. Which topics can you name in two words, and which ones need *and*?
    """)
    return


@app.cell(hide_code=True)
def try_box_cell():
    names_box = mo.ui.text_area(value=C.NAMES_START, label="Your names, one line per topic", rows=12, full_width=True)
    names_box
    return (names_box,)


@app.cell(hide_code=True)
def try_look(names_box):
    mo.Html(C.names_picture(names_box.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_mixed():
    mo.md(r"""
    ### Exercise 4. A page in the wrong topic

    Topic 9 has the words *abc*, *series*, *television*, *doctor* and *network*. Most of its paragraphs come from *American Broadcasting Company* and *Doctor Who*, and 40 from *Chloroplast*.
    """)
    return


@app.cell(hide_code=True)
def choice_mixed_cell():
    choice_mixed = L.choice(
        "mixed-topic",
        question="Why does LDA put paragraphs about chloroplasts in a topic about television?",
        options={
            "bug": "A mistake of scikit-learn: a correct LDA would give every page a topic of its own",
            "word": "The paragraphs about chloroplasts use the word *television*",
            "right": "12 topics for 48 pages: words that never meet in a paragraph can share a topic",
            "names": "LDA read the names of the pages, and these sound alike",
        },
        correct="right",
        why={
            "right": "Twelve topics for 48 pages: some pages have to share a topic. *abc* stands in no paragraph of *Chloroplast*, and *chloroplast* in none of *American Broadcasting Company*, so one topic can serve both: a paragraph about television uses one half of the topic, a paragraph about chloroplasts the other. A topic is a group of words that LDA finds useful together, not a meaning. With 20 topics, *chloroplast* moves to a topic with *rhine* and *water*, and *abc* now also stands in a topic with *european*, *law* and *court*: more topics move the sharing, they do not end it.",
            "bug": "LDA has no rule that a page gets its own topic, and it never sees the pages. How many topics are there here, and how many pages?",
            "word": "*television* does not occur in a single paragraph of *Chloroplast*. Which words do the paragraphs about chloroplasts and those about ABC have in common?",
            "names": "LDA gets only the counts of the words in each paragraph, never the name of the page. What does it base its topics on?",
        },
    )
    choice_mixed
    return (choice_mixed,)


@app.cell(hide_code=True)
def text_after_choice():
    mo.md(r"""
    How do you know whether a topic model is good? The chart above measured surprise, on the same paragraphs that LDA trained on. Chang and others (2009) asked people instead. In their test of word intrusion, a person saw six words: the five most probable words of a topic, and one word that is improbable in that topic but probable in another. If the topic makes sense, the odd one out is easy to spot: "most people readily identify apple as the intruding word in the set {dog, cat, horse, apple, pig, cow}". Their finding: "Surprisingly, topic models which perform better on held-out likelihood may infer less semantically meaningful topics." Blei (2012) agrees that "there is no technical reason to suppose that held-out accuracy corresponds to better organization or easier interpretation." Whether a topic means something is a question for people who read it, as you did when naming them.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A document as a vector](/course/open/nlp-document-vectors?lang=en) turned each text into a row of counts. LDA starts from such rows and nothing else, and finds $k$ topics, each a probability distribution over the words, and a mix of topics for every document: learning without labels, as in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en). Your `topic_words` read the topics from their weights, your `word_topics` gave each word of a paragraph its most likely topic, and your `topic_pages` checked the topics against the pages that LDA never saw. Some topics match a page, others join pages whose words never meet; the names, and the judgment whether a topic means anything, are yours. The next lesson, *From text to a table*, takes facts out of a text, such as names and dates, and puts them in the columns of a table.

    **An everyday example.** A recipe site sorts its recipes by the words in them: *flour*, *oven* and *yeast* point to baking, *wok* and *soy* to stir-frying. A recipe for pizza on the grill gets a mix of baking and barbecue, and a person still decides what to call each group.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mixed, result_pages, result_topics, result_words):
    L.wrap_up(result_words, result_topics, result_pages, choice_mixed)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
