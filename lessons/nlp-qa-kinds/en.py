# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-qa-kinds. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-qa-kinds", "en")
    C = L.checks   # the course's six texts and twelve questions, its run of Qwen3-0.6B, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [What NLP is](/course/open/nlp-what-is-nlp?lang=en), and one about [Lists](/course/open/py-lists?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_nlu():
    quick_nlu = L.choice(
        "quick-nlu",
        question="In *What NLP is*, natural language understanding (NLU) goes from text to meaning, and natural language generation (NLG) from meaning to text. A system reads a question, works out what it asks, and writes the answer in a sentence of its own. What does it do?",
        options={
            "nlu": "Only NLU: it reads",
            "nlg": "Only NLG: it writes",
            "right": "Both: NLU for the question, NLG for the answer",
            "cl": "Computational linguistics: it studies how people ask questions",
        },
        correct="right",
        why={
            "right": "Reading the question is NLU, and writing the answer is NLG. A system that only marks the answer in a text needs no NLG: this lesson shows both kinds.",
            "nlu": "Reading the question is NLU. And what does the system do after that, with the answer? See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "nlg": "Writing the answer is NLG. And what does the system do first, with the question? See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "cl": "Computational linguistics studies language itself with computers. This system does a task with language: which parts of NLP does it use? See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_nlu
    return (quick_nlu,)


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question='What does `words[2:4]` give, for `words = ["the", "pool", "opens", "at", "9:00"]`?',
        options={
            "early": '`["pool", "opens"]`',
            "right": '`["opens", "at"]`',
            "three": '`["opens", "at", "9:00"]`: from index 2 up to and including index 4',
            "start": '`["the", "pool", "opens"]`',
        },
        correct="right",
        why={
            "right": 'Index 2 is `"opens"`, and the slice stops just before index 4. A text can be sliced the same way, one character at a time: this lesson marks an answer in a text with two such numbers.',
            "early": "Indexes start at 0. Which item has index 2? See [Lists](/course/open/py-lists?lang=en).",
            "three": "A slice stops just before its second number. Which index is the last one in it? See [Lists](/course/open/py-lists?lang=en).",
            "start": "A slice starts at its first number, not at the start of the list. Which item has index 2? See [Lists](/course/open/py-lists?lang=en).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You type a question into a search engine, and above the links it shows the answer, with the sentence it came from. You ask a chat assistant the same question, and it writes an answer in its own words, without a source. Both are **question answering**: answering questions asked in ordinary language. But the answer comes from a different place, and has a different form: the search engine points to a piece of a page it found, the assistant writes from what its model learned. Which kind a system is decides what you can check, and what can go wrong.

    This lesson lets Qwen3-0.6B, the small chat model of the course, answer twelve questions about six short texts that the course wrote, once without a text and once with each of the six texts.

    **An everyday example.** At a station, one person reads you the departure time from the board, and another tells you from memory. The first is wrong only when the board is; the second can be wrong without anyone noticing.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_town = L.guess(
        "guess",
        question="Three of the six texts are about the world: spiders, Mercury and the Eiffel Tower. Three are about a town that the course made up: its library, its repair café and its outdoor pool. Qwen3 gets the six questions about the town without any text, with the seeds 0 to 3, so 24 times in all. In how many of those 24 answers is the answer right?",
        options={
            "none": "None, or almost none",
            "quarter": "About a quarter",
            "half": "About half",
            "most": "Most of them: the questions are easy",
        },
        reveal="Two of the 24, and both by luck: *The pool is open for 4 months a year.*, with the seeds 1 and 3. With the seeds 0 and 2, Qwen3 writes 12 and 30 months. For the library, the repair café and the pool, it writes closing times, prices and numbers of meetings that are all made up, and it never says that it cannot know.",
    )
    guess_town
    return (guess_town,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Question answering** (QA) systems differ in two ways: where the answer comes from, and what form it has.

    **Where the answer comes from.** In **reading comprehension**, a text comes with the question, and the answer is in it. In **open-domain question answering**, the system gets only the question, and first has to find texts that may hold the answer in a large collection: a **retriever** finds them, and a **reader** answers from them. In **closed-book question answering**, the system looks nothing up: a model answers from what it learned while it was trained. Adam Roberts, Colin Raffel and Noam Shazeer, who gave the task its name in 2020, compare it to an exam: "A model that answers questions in this way is metaphorically similar to a student taking a closed-book exam, where the student must study and memorize all pertinent information before taking the test." Open-domain question answering, in their words, "acts as if it is taking an open-book exam where it can find and use information in an external source of knowledge."

    **The form of the answer.** In **extractive question answering**, the answer is a **span**: a piece of the text that stands in it word for word, given by where it starts and where it ends. In **abstractive question answering**, the system writes the answer itself: in its own words, or after a small calculation. In the words of Roberts and others: "Models can be trained either to indicate the span of the context that contains the answer or output the text of the answer itself." Closed-book question answering is always abstractive: without a text, the system has nothing to point into.

    The picture puts the kinds side by side, each with a system or a dataset that made it known.
    """)
    return


@app.cell(hide_code=True)
def kinds_cell():
    mo.Html(C.kinds_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **question answering** (QA) | answering questions asked in ordinary language, from a given text, from a collection of texts, or from what a model learned |
        | **reading comprehension** | question answering in which a text comes with the question and holds the answer |
        | **open-domain question answering** | question answering without a given text: the system first finds texts in a collection, then reads them |
        | **retriever**, **reader** | the two steps of open-domain question answering: the retriever finds texts that may hold the answer, the reader answers from them |
        | **closed-book question answering** | question answering from only what a model learned, without looking anything up |
        | **extractive question answering** | question answering in which the answer is a span of a text: the system only marks where it starts and ends |
        | **span** | a piece of a text, given by where it starts and where it ends; in the course a tuple `(start, end)`, so that `text[start:end]` is the answer |
        | **abstractive question answering** | question answering in which the system writes the answer itself |
        | `text[a:b]` | a slice of a text: its characters from place `a` up to, not including, place `b`, as with a list |
        """),
        L.watch_out("In other lessons of the course, about following what an AI application does step by step, a *span* is one step of a trace, such as a single call to a model. Here a span is a piece of a text: the same word for something else."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To tell which kind a system is:

    1. Does a text come with the question? Then it is reading comprehension.
    2. Does the system find texts itself, in a collection? Then it is open-domain: first a retriever, then a reader.
    3. Does it look nothing up? Then it is closed-book.
    4. Is the answer a span of a text, given by where it starts and ends? Then it is extractive. Does the system write the answer? Then it is abstractive.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The text about the town library answers *When does the town library close on Saturday?* with *14:30*. Take one short sentence like it: *On Saturday the library closes at 14:30.* As a span, the answer is two numbers: where it starts in the sentence, and where it ends. A slice of the text with those two numbers gives the answer back, as a slice of a list does:
    """)
    return


@app.cell
def example_cell():
    text = "On Saturday the library closes at 14:30."
    start = 34
    end = 39
    print(text[start:end])
    print(end - start, len("14:30"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The span `(34, 39)` gives *14:30*: 34 characters come before it, spaces included. The end is the place just after the last character, so `end - start` is the length of the answer, 5. Only an answer that stands in the text word for word has a span. In the whole text about the library, the same answer is the span `(101, 106)`. The picture shows the question in all four kinds, with Qwen3's answers from the course's run.
    """)
    return


@app.cell(hide_code=True)
def four_cell():
    mo.Html(C.four_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_four():
    mo.md(r"""
    With the text, Qwen3 writes *On Saturday, the town library closes at 14:30.*: an abstractive answer with the span in it. Without a text, it writes *3:00 PM*. The town is made up, so Qwen3 cannot know when its library closes, but it answers anyway, and sounds as sure as with the text. Open-domain question answering first has to find the text: the text about the library shares five different words with the question, more than any other text.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_span():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. find_span: the answer as a span

        Write `find_span(text, answer)`. Give back the tuple `(start, end)` of the first place where `answer` stands in `text`, word for word, so that `text[start:end] == answer`; give back `None` when it is nowhere in the text. The loop tries every place where the answer could start: the last one is `len(text) - len(answer)`. Replace the `...`.

        The check calls `find_span("The library opens at 10:00.", "10:00")`, which gives `(21, 26)`, and the same in Dutch. Then cases of its own: an answer at the start and at the very end of a text, one that is not in it, one that stands in it twice, and answers in the texts of the collection (in the lesson: `C.TEXTS`).
        """),
        L.task("Find where an answer stands in a text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_span():
    box_span = L.code_box("find_span", lines=7)
    box_span
    return (box_span,)


@app.cell(hide_code=True)
def check_span(box_span):
    result_span = L.check("find_span", box_span.value)
    result_span
    return (result_span,)


@app.cell(hide_code=True)
def hints_span():
    L.hints(
        "find_span",
        [
            "In the loop, `start` is the place to try. The piece of `text` from `start` on, as long as the answer, ends at `start + len(answer)`; when that piece equals `answer`, `return` gives back the two places.",
            "A slice of a text works as a slice of a list: `text[3:8]` is the characters from place 3 up to, not including, place 8. Which slice is as long as the answer and starts at `start`?",
            "`        if text[___] == answer:`",
        ],
        why="In the loop: `if text[start:start + len(answer)] == answer:` with `return (start, start + len(answer))` under it. After the loop, `return None`.",
    )
    return


@app.cell(hide_code=True)
def text_after_span():
    mo.md(r"""
    Seven of the twelve answers have a span. The other five are not in their text word for word, such as *two* for how many more legs a spider has than an insect: the text says *eight* and *six*, and the answer needs a calculation. An extractive system cannot give such an answer; it can only point to *eight* or to *six*. Answers that people write are often like that. NarrativeQA, a dataset of questions about books and movie scripts, has answers that people wrote in their own words, and Tomáš Kočiský and others counted: "There are answers that appear as spans of the summaries and the stories, 44.05% and 29.57%, respectively." So about 70% of the answers are not a span of the story.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. best_text: the retriever

        Write `best_text`, which gets a `question` and `texts`, a list of texts. Give back the place in the list of the text that shares the most different words with the question; on a tie, the first of them. The box knows `words(text)`, which gives the words of a text in lower case, without punctuation: `words("On Sunday, it stays closed.")` gives `["on", "sunday", "it", "stays", "closed"]`. In the start code, `asked` holds the different words of the question, `best` the place of the best text so far, and `most` the number of words it shares. Replace the `...`.

        The check calls `best_text("When does the pool open?", ["The library opens at 10:00.", "The pool opens at 9:00."])`, which gives `1`, and the same in Dutch. Then cases of its own: a tie, a word of the question inside a longer word, a text that repeats words of the question, and questions of the lesson with the six texts (in the lesson: `C.TEXTS`).
        """),
        L.task("Pick the text that shares the most words with the question."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("best_text", lines=10)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("best_text", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "best_text",
        [
            "In the loop, `shared` can count the words `w` of `asked` that are also among the words of text `k`, as `words` gives them: `len` of a list comprehension. When `shared` is more than `most`, `best` becomes `k` and `most` becomes `shared`.",
            "Count each different word of the question once: go over `asked`, not over the words of the text. And which comparison keeps the first text on a tie, `>` or `>=`?",
            "`        shared = len([w for w in asked if w in ___])`",
        ],
        why="In the loop: `shared = len([w for w in asked if w in words(texts[k])])`, then `if shared > most:` with `best = k` and `most = shared` under it.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    Your retriever picks the text that answers the question for eleven of the twelve. For *How many months a year is the pool open?* it picks the text about Mercury: Mercury, the library and the pool each share four words with the question, and Mercury comes first. For Mercury, those words are *a*, *year*, *is* and *the*: *a*, *is* and *the* are in most of the texts, and *year* is there because the text is about a year on Mercury. The one word that matters, *pool*, counts no more than *a*. A real retriever gives rare words more weight than common ones, as later lessons about search show.

    And a real collection is far larger. DrQA, the system of Danqi Chen and others, searched all of Wikipedia: "In order to answer any question, one must first retrieve the few relevant articles among more than 5 million items, and then scan them carefully to identify the answer." Its retriever had the answer in its top five articles for 77.8% of the questions of SQuAD, a dataset of questions about paragraphs of Wikipedia.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_open():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. open_domain: find a text, then let Qwen3 read it

        Write `open_domain(question, texts)`. Pick a text with `best_text`, and give back what Qwen3 answers to the question with that text. The box knows the course's `best_text`, so your answer to exercise 2 does not matter here, and `ask(question, text)`, which gives Qwen3's answer from the course's run, seed 0, to one of the lesson's questions with one of its texts; `ask(question)`, without a text, gives its closed-book answer. Replace the `...`.

        The check calls `open_domain("When does the town library close on Saturday?", C.TEXTS["en"])`, which gives `"On Saturday, the town library closes at 14:30."`, and the same in Dutch. Then cases of its own: other questions, and a list with only some of the texts.
        """),
        L.task("Answer a question with the text your retriever finds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_open():
    box_open = L.code_box("open_domain", lines=4)
    box_open
    return (box_open,)


@app.cell(hide_code=True)
def check_open(box_open):
    result_open = L.check("open_domain", box_open.value)
    result_open
    return (result_open,)


@app.cell(hide_code=True)
def hints_open():
    L.hints(
        "open_domain",
        [
            "`best_text` gives the place of a text in `texts`, for the question `question`. With `texts[...]` you take the text at that place, and `ask(question, ...)` gives what Qwen3 answered with it.",
            "Two steps in one line: first the place, then the text at that place. What goes between the square brackets?",
            "`    return ask(question, texts[___])`",
        ],
        why="`return ask(question, texts[best_text(question, texts)])`.",
    )
    return


@app.cell(hide_code=True)
def text_after_open():
    mo.md(r"""
    With seed 0, Qwen3 without a text gets 3 of the 12 right: the two about the legs of a spider, and the year of Mercury. It writes 1887 for the Eiffel Tower, finished in 1889, and makes up an answer to every question about the town. With the text your open_domain picks, 9 of the 12 are right. Three stay wrong. For the library, Qwen3 writes that it is closed on weekends except on Saturday: that comes down to Sunday, but the answer does not name it, so the course counts it wrong; for the repair café, *on the first Saturday of every month*, which does not say how many times a year; and for the pool, your retriever gave it the text about Mercury, and Qwen3 wrote that the text says nothing about a pool. That last answer is honest. With a text that does not answer the question, Qwen3 says that the text does not give the answer in 49 of its 240 answers; without a text, it never does.
    """)
    return


@app.cell(hide_code=True)
def table_cell():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_table():
    mo.md(r"""
    Over the four seeds, 12 of the 24 answers about the world are right without a text, and 18 with the text; about the town, 2 without a text and 13 with it. The answers that stay wrong with the text are mostly the ones that need a calculation, such as *15 months* for the pool. Here open-domain question answering does as well as reading comprehension: your retriever misses only the pool question, and for that question Qwen3 also goes wrong with the text about the pool.

    Roberts and others trained T5, with about 11 billion parameters, for closed-book question answering: it got 32.6% of the questions of Natural Questions right, where the best system in their table that looks things up got 41.5%. They also name the cost: "our model distributes knowledge in its parameters in an inexplicable way and hallucinates realistic-looking answers when it is unsure." A system that looks things up can usually show the text it used; a closed-book model cannot.
    """)
    return


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: every answer of the course's run

    Pick a question and a seed, and see what Qwen3 wrote without a text and with each of the six texts. Look for a text that does not answer the question, and what Qwen3 makes of it.
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_question = mo.ui.dropdown(options=C.question_options(L.lang), value="7. When does the town library close on Saturday?", label="Question")
    try_seed = mo.ui.slider(0, 3, value=0, label="Seed", show_value=True)
    mo.hstack([try_question, try_seed], justify="start", gap=1.5, wrap=True)
    return try_question, try_seed


@app.cell(hide_code=True)
def try_look(try_question, try_seed):
    mo.Html(C.try_picture(L.lang, try_question.value, try_seed.value))
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.md(r"""
    ### Exercise 4. Which kind?

    A bike shop has a help desk bot. It gets a question from a customer, looks up the page of the shop's handbook that shares the most words with the question, and shows the sentence of that page that holds the answer, with the answer marked.
    """)
    return


@app.cell(hide_code=True)
def choice_kind_cell():
    which_kind = L.choice(
        "which_kind",
        question="Which kind of question answering does the bot do?",
        options={
            "closed": "Closed-book: it answers from what it learned",
            "right": "Open-domain and extractive",
            "reading": "Reading comprehension: the customer gives it the page",
            "abstractive": "Open-domain and abstractive: it finds the page, then writes an answer in its own words",
        },
        correct="right",
        why={
            "right": "It finds the page itself, in a collection: open-domain, with a retriever. It shows a piece of that page word for word: extractive. A customer can check the answer against the page, but a wrong page gives a wrong sentence.",
            "closed": "The bot looks up a page of the handbook. Does a closed-book system look anything up?",
            "reading": "The customer only asks the question. Who finds the page?",
            "abstractive": "The bot shows a sentence of the page, with the answer marked. Does it write anything itself?",
        },
    )
    which_kind
    return (which_kind,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What NLP is](/course/open/nlp-what-is-nlp?lang=en) you sorted systems by what they do with language; here you sorted question answering by where the answer comes from and what form it has. Why a model writes an answer it cannot know, [Why language models make things up](/course/open/lm-hallucination?lang=en) explains. The next lesson, *Finding the answer in a passage*, takes reading comprehension further: a dataset of questions about paragraphs of Wikipedia, and a first reader that picks the sentence with the most words of the question.

    **An everyday example.** A help page that shows the paragraph of the manual its answer came from lets you check the answer in seconds; an answer without a source you can only trust, or not.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_open, result_pick, result_span, which_kind):
    L.wrap_up(result_span, result_pick, result_open, which_kind)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
