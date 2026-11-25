# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-extractive-qa. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-extractive-qa", "en")
    C = L.checks   # the paragraphs and questions of SQuAD and SQuAD-NL, cut into sentences, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Kinds of question answering](/course/open/nlp-qa-kinds?lang=en), and one about [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_span():
    quick_span = L.choice(
        "quick-span",
        question='In *Kinds of question answering*, the answer of extractive question answering is a span `(start, end)`, so that `text[start:end]` is the answer. Which span gives *21:00*, for `text = "Open until 21:00"`?',
        options={
            "short": "`(11, 15)`",
            "right": "`(11, 16)`",
            "length": "`(11, 5)`",
            "text": '`"21:00"`, the answer itself',
        },
        correct="right",
        why={
            "right": "*21:00* starts at place 11, after 11 characters, spaces included, and the end is the place just after its last character: 11 + 5 = 16. In this lesson, people marked such spans in paragraphs of Wikipedia.",
            "short": "A slice stops just before its second number. Which character does `text[11:15]` leave out? See [Kinds of question answering](/course/open/nlp-qa-kinds?lang=en).",
            "length": "The second number says where the answer ends, not how long it is. Where does *21:00* end? See [Kinds of question answering](/course/open/nlp-qa-kinds?lang=en).",
            "text": "The answer itself says what it is, not where it stands. Which two numbers say where it starts and ends? See [Kinds of question answering](/course/open/nlp-qa-kinds?lang=en).",
        },
    )
    quick_span
    return (quick_span,)


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="In *Normalizing: case, accents and stop words*, you left the stop words out of a list of words. Which words are stop words?",
        options={
            "end": "Words that end a sentence, such as the last word before a full stop",
            "rare": "Rare words that occur only once in a text",
            "right": "The most common words, such as *the*, *of* and *is*",
            "unknown": "Words that a tokenizer cannot cut",
        },
        correct="right",
        why={
            "right": "Words such as *the*, *of* and *is* are in almost every text, so they say little about what one text is about. This lesson leaves them out when it counts the words that a question and a sentence share.",
            "end": "Stop words have nothing to do with the end of a sentence. Which words are in almost every text? See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
            "rare": "Rare words often say the most about a text. Which words say the least? See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
            "unknown": "A stop word list is about how common a word is, not about cutting. Which words are on it? See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    At school, a reading test gives you a text and questions, and the answer to each question is somewhere in the text. In 2016, Pranav Rajpurkar and others at Stanford gave computers such a test at a large scale: SQuAD, more than 100,000 questions that people asked about paragraphs of Wikipedia, with each answer marked in its paragraph. Because every answer is a span, the answer of a system can be compared with it without a person reading along, and many systems for reading comprehension have been measured on it since.

    This lesson takes 1,190 of those questions, and the same questions in Dutch, and tries the simplest reader there is: pick the sentence that shares the most words with the question.

    **An everyday example.** In a long recipe, you look for the oven temperature by scanning for *oven* and *degrees*. That finds the right sentence, until the recipe says *heat to 200* in a sentence that has neither word.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_reader = L.guess(
        "guess",
        question="Each question of SQuAD comes with its paragraph of Wikipedia. Take the sentence of the paragraph that shares the most words with the question, leaving out stop words such as *the* and *in*. For how many of the 1,190 questions does that sentence hold the answer?",
        options={
            "quarter": "About a quarter",
            "half": "About half",
            "most": "About three quarters",
            "all": "Almost all of them",
        },
        reveal="About three quarters: 929 of the 1,190, or 78%. Picking a sentence at random would hold the answer for about 26% of the questions, and always picking the first sentence of the paragraph for 32%. Counting shared words does much better, but more than a fifth of the questions still goes wrong, and you will see why.",
    )
    guess_reader
    return (guess_reader,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **SQuAD**, the Stanford Question Answering Dataset, is a dataset for reading comprehension: each question comes with a paragraph, and the answer is a span of that paragraph. Rajpurkar and others took 536 articles of Wikipedia, cut them into paragraphs, and had people on a crowdsourcing platform write the questions: "On each paragraph, crowdworkers were tasked with asking and answering up to 5 questions on the content of that paragraph. The questions had to be entered in a text field, and the answers had to be highlighted in the paragraph." The workers were encouraged to use their own words, "without copying word phrases from the paragraph", and copying and pasting from the paragraph was switched off. SQuAD split its questions into a training set, a development set (its name for a validation set) and a test set; for the questions of the last two, at least two more people each marked an answer as well. In all, SQuAD has 107,785 questions with their answers.

    In the files of SQuAD, each answer is its text and `answer_start`: the place of its first character in the paragraph. XQuAD, by Mikel Artetxe, Sebastian Ruder and Dani Yogatama, took 240 paragraphs of the development set of SQuAD with their 1,190 questions. SQuAD-NL, by Wietse de Vries, Martijn Wieling and Malvina Nissim, has the same paragraphs and questions in Dutch, translated by machine and corrected by people. The picture shows one of the 240 paragraphs, from the article about the Rhine, with its three questions.
    """)
    return


@app.cell(hide_code=True)
def squad_cell():
    mo.Html(C.squad_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_reader():
    mo.md(r"""
    A reader that marks the exact span is hard to build. A first step is easier: find the sentence that holds the answer. The people who wrote the questions asked about the paragraph, so the question and the sentence with the answer often share words: *Where does the Delta in the Netherlands begin?* shares *Delta* and *Netherlands* with the first sentence, where the delta begins near Millingen aan de Rijn. A reader can count those shared words for each sentence and pick the sentence with the most. Stop words such as *the* and *in* stand in almost every sentence, so the reader leaves them out.
    """)
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **SQuAD** | the Stanford Question Answering Dataset (2016): more than 100,000 questions that people asked about paragraphs of Wikipedia, each with its answer marked as a span of its paragraph |
        | **passage** | the text that comes with a question in reading comprehension; in SQuAD a paragraph of Wikipedia |
        | `answer_start` | in the files of SQuAD, the place of the first character of an answer in its paragraph |
        | **crowdworker** | someone who is paid per small task on a website, such as writing questions about a paragraph |
        | **XQuAD**, **SQuAD-NL** | 240 paragraphs of SQuAD with their questions, in English and in other languages; SQuAD-NL has them in Dutch |
        | `text.index(part)` | the place where `part` first stands in `text` |
        """),
        L.watch_out("In other lessons of the course, about following what an AI application does step by step, a *span* is one step of a trace, such as a single call to a model. Here a span is a piece of a text, as in *Kinds of question answering*: the same word for something else."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the sentence with the answer by counting words:

    1. Cut the paragraph into sentences, each with the space after it, so that together they are the paragraph again.
    2. Take the different words of the question, without the stop words.
    3. For each sentence, count how many of those words it has too.
    4. Pick the sentence with the most; on a tie, the first of them.

    To check the pick: add up the lengths of the sentences, from the first, until the total is more than `answer_start`. The sentence that makes it more holds the answer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take the third question about the Rhine: *What is the nickname for the Delta in the Netherlands?* Without the stop words *what*, *is*, *the*, *for* and *in*, three words are left: *nickname*, *delta* and *netherlands*. The first sentence shares two of them, *delta* and *netherlands*; the second and the third share one, *delta*. So the reader picks the first sentence. Where does the answer, *Rhine Delta*, start, and in which sentence is that? A few lines work it out. `C.rhine("en")` gives the three sentences:
    """)
    return


@app.cell
def example_cell():
    _sentences = C.rhine("en")
    _paragraph = "".join(_sentences)
    _start = _paragraph.index("Rhine Delta")
    print(_start)
    print(len(_sentences[0]), len(_sentences[0]) + len(_sentences[1]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The answer starts at place 264. The first sentence has 200 characters, spaces included, so it covers the places 0 to 199, and the second ends at 200 + 94 = 294. Place 264 lies in between: the answer is in the second sentence, *Since the Rhine contributes most of the water, the shorter term Rhine Delta is commonly used.* The reader is wrong here. The second sentence says *term* where the question says *nickname*, and it does not name the Netherlands, so it shares only *delta*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_of():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. sentence_of: which sentence holds the answer

        Write `sentence_of`, which gets `sentences`, the list of the sentences of a paragraph, each with the space after it, so that together they are the paragraph, and `start`, a place in the paragraph, such as the `answer_start` of an answer. Give back the place in the list of the sentence in which that character stands. In the start code, `end` holds where the sentences so far end. Replace the `...`.

        The check calls `sentence_of(["The museum opens at 10:00. ", "It closes at 17:00."], 40)`, which gives `1`: *17:00* starts at place 40, in the second sentence. The same in Dutch, and then cases of its own: a start at the very first character, at the first and at the last character of a sentence, in the last sentence, and answers in the paragraph about the Rhine (in the lesson: `C.rhine("en")`).
        """),
        L.task("Find the sentence in which an answer starts."),
    ])
    return


@app.cell(hide_code=True)
def exercise_of():
    box_of = L.code_box("sentence_of", lines=6)
    box_of
    return (box_of,)


@app.cell(hide_code=True)
def check_of(box_of):
    result_of = L.check("sentence_of", box_of.value)
    result_of
    return (result_of,)


@app.cell(hide_code=True)
def hints_of():
    L.hints(
        "sentence_of",
        [
            "In the loop, `end` grows by the length of sentence `k`, with `len` and `sentences[k]`, so that it is where that sentence ends in the paragraph. When `start` is smaller than `end`, the answer starts in sentence `k`, and `return` gives back `k`.",
            "A first sentence of 27 characters covers the places 0 to 26; place 27 is the first character of the next sentence. Which comparison, `<` or `<=`, keeps place 27 out of the first sentence?",
            "`        end = end + ___`",
        ],
        why="In the loop: `end = end + len(sentences[k])`, then `if start < end:` with `return k` under it.",
    )
    return


@app.cell(hide_code=True)
def text_after_of():
    mo.md(r"""
    With your function, the course knows for each of the 1,190 questions which sentence holds its answer. The 240 paragraphs have 1,172 sentences, about five per paragraph. In a paragraph of five sentences, a sentence picked at random holds the answer one time in five; shorter paragraphs give it better odds, so over all the questions it comes to about a quarter. Most answers of SQuAD are short: 3.2 words on average, in a sentence of 30.8.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. best_sentence: the reader

        Write `best_sentence(question, sentences, stop_words)`. Give back the place of the sentence that shares the most different words with the question, leaving out the words in the set `stop_words`; on a tie, the first of them, and the first sentence when none shares a word. The box knows `words(text)` from *Kinds of question answering*, which gives the words of a text in lower case, without punctuation. In the start code, `asked` should hold the different words of the question that are not stop words, and `shared` how many of them sentence `k` has; the lines after it keep the best sentence so far. Replace the two `...`.

        The check calls `best_sentence("When does the pool open?", ["The library opens at 10:00. ", "The pool opens at 9:00."], C.STOP_WORDS["en"])`, which gives `1`, with the stop words of Snowball. The same in Dutch, and then cases of its own: a question whose stop words stand in the wrong sentence, a word of the question inside a longer word, a sentence that repeats a word of the question, a question that repeats a word, a tie, and questions of SQuAD.
        """),
        L.task("Pick the sentence that shares the most words with the question."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("best_sentence", lines=10)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("best_sentence", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "best_sentence",
        [
            "`asked` can be `set` of a list comprehension over `words(question)` that keeps each word `w` that is not in `stop_words`. In the loop, `shared` is `len` of a second list comprehension: each word `w` of `asked` that is among the words of `sentences[k]`, as `words` gives them.",
            "A set holds each different word once, so a word that the question repeats counts once, and going over `asked` counts each word once, also when the sentence repeats it. Look for each word among the words of the sentence, not in the sentence itself: `in` on a text looks for letters, so *open* is also found inside *reopens*.",
            "`        shared = len([w for w in asked if w in ___])`",
        ],
        why="`asked = set([w for w in words(question) if w not in stop_words])`, and in the loop `shared = len([w for w in asked if w in words(sentences[k])])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    On these four questions, your reader is right only for the first. For the costume collection of the V&A, the last sentence shares four words with the question, *costume*, *collection*, *items* and the *v* of *V&A*, and it is about a gift of 1,442 costumes; the answer, *over 14,000*, is in the first sentence, which shares two. For Jacksonville, three sentences share one word each, and the first one wins. And for the stock exchange of Warsaw, the sentence with the answer says *It was re-established in April 1991*: it shares no word with *brought back to life*, and its *It* points back to the sentence before it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. share_right: measure the reader

        Write `share_right(items, stop_words)`. Each item is a tuple `(question, sentences, start)`: a question, the sentences of its paragraph, and where its first answer starts. Give back the share of the items for which the sentence that `best_sentence` picks holds the start of the answer, as `sentence_of` gives it. The box knows the course's `best_sentence` and `sentence_of`, so your answers to exercises 1 and 2 do not matter here. Replace the `...`.

        The check calls `share_right(C.rhine_items("en"), C.STOP_WORDS["en"])`, the three questions about the Rhine, which gives `0.6666666666666666`: two of the three. The same in Dutch, and then all questions, in English and in Dutch (in the lesson: `C.items("en")`), and parts of them.
        """),
        L.task("Measure how often your reader picks the sentence with the answer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share_right", lines=6)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share_right", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share_right",
        [
            "`best_sentence` gets `question`, `sentences` and `stop_words`, and gives the place of the sentence your reader picks; `sentence_of` gets `sentences` and `start`, and gives the place of the sentence with the answer. When the two places are the same, `right` goes up by one.",
            "Compare the two places with `==`, in an `if`. Which line under the `if` adds one to `right`?",
            "`        if best_sentence(question, sentences, stop_words) == ___:`",
        ],
        why="In the loop: `if best_sentence(question, sentences, stop_words) == sentence_of(sentences, start):` with `right = right + 1` under it.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    Your reader picks the sentence with the answer for 929 of the 1,190 questions: 78.1%. [Always beat a simple baseline](/course/open/ml-baselines?lang=en) says to compare a score with a simple baseline first. A sentence picked at random gives 26.0% on average, and always the first sentence of the paragraph 32.4%: counting words does far better than both. Rajpurkar and others built a logistic regression with 180 million features for SQuAD, and it found the sentence with the answer for 79.3% of the questions, on the first version of SQuAD. Your reader comes close to that on 1,190 questions of version 1.1.

    Leaving out the stop words helps less than you might expect: with the stop words kept, the same reader gets 912 right, 17 fewer. Without the stop words, fewer words count, so more sentences tie, and a tie goes to the first sentence, also when the answer is in a later one.
    """)
    return


@app.cell(hide_code=True)
def text_fails():
    mo.md(r"""
    ### Where the reader goes wrong

    Your reader misses 261 of the 1,190 questions. For 100 of them, the sentence with the answer ties with an earlier sentence, and the earlier one wins. For 111, another sentence shares more words with the question. And for 50, the sentence with the answer shares no word with the question at all. Rajpurkar and others went through 192 questions of the development set by hand, and every one of them differs from its answer sentence in its words or in the order of its words. For 33.3% of them, the main words are synonyms, as in *What is the Rankine cycle sometimes called?* with *The Rankine cycle is sometimes referred to as a practical Carnot cycle.* For 13.6%, the answer needs more than one sentence, for instance because a word such as *They* points back to an earlier sentence.

    ### Try it: what the reader picks

    Pick what the reader does and a question, and see the sentences of its paragraph, the words each one shares with the question, and what the reader picks. The box leaves out the paragraphs that the course does not show: it has 876 of the 1,190 questions.
    """)
    return


@app.cell(hide_code=True)
def look_kind():
    look_which = mo.ui.dropdown(options=C.kind_options(L.lang), value=list(C.kind_options(L.lang))[4], label="Outcome")
    look_which
    return (look_which,)


@app.cell(hide_code=True)
def look_number(look_which):
    look_n = mo.ui.slider(1, C.kind_count(L.lang, look_which.value), value=1, label="Question", show_value=True)
    look_n
    return (look_n,)


@app.cell(hide_code=True)
def look_cell(look_n, look_which):
    mo.Html(C.look_picture(L.lang, look_which.value, look_n.value))
    return


@app.cell(hide_code=True)
def text_exercise_left():
    mo.md(r"""
    ### Exercise 4. What is still missing?

    Your reader picks the sentence with the answer for 78% of the questions. But SQuAD counts an answer as right when it matches a span that people marked.
    """)
    return


@app.cell(hide_code=True)
def choice_left_cell():
    what_left = L.choice(
        "what_left",
        question="What does your reader still have to do before it gives an answer the way SQuAD counts it?",
        options={
            "nothing": "Nothing: the sentence that holds the answer is the answer",
            "right": "Mark the span of the answer in the sentence",
            "search": "First find the right paragraph among all the articles of Wikipedia, as DrQA does",
            "write": "Write the answer again in its own words",
        },
        correct="right",
        why={
            "right": "The answer is a span of about 3 words, in a sentence of about 31. Rajpurkar and others found the same with their model, which picked the right sentence for 79.3% of the questions: \"hence, the bulk of the difficulty lies in finding the exact span within the sentence.\" The next lesson scores how close an answer comes to the spans of the people.",
            "nothing": "A sentence has about 31 words, the answer about 3. Does SQuAD count a whole sentence as the answer?",
            "search": "In reading comprehension, the paragraph comes with the question. What does the reader not have yet, once it has the right sentence?",
            "write": "The answers of SQuAD are spans of the paragraph. Does an extractive reader write anything itself?",
        },
    )
    what_left
    return (what_left,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Kinds of question answering](/course/open/nlp-qa-kinds?lang=en) you marked answers as spans and wrote a retriever that counts shared words. Here the same counting, with the stop words of [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en) left out, became a reader, and you measured it against the baselines of [Always beat a simple baseline](/course/open/ml-baselines?lang=en). The next lesson, *Scoring answers: exact match and F1*, scores answers the way SQuAD does: an answer that matches a span of the people exactly, and an answer that shares only some of its words.

    **An everyday example.** The search box of a web page jumps to the first place where your words stand. It finds the answer often, but not on a page that says the same in other words.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_of, result_pick, result_share, what_left):
    L.wrap_up(result_of, result_pick, result_share, what_left)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
