# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-why-language-is-hard. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-why-language-is-hard", "en")
    C = L.checks   # made-up meanings of eight words, five sentences, spelling variants and five messages


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What NLP is](/course/open/nlp-what-is-nlp?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="A tool sums up a long report in five sentences. Which side of NLP is that, in the course's simple sort?",
        options={
            "nlu": "NLU, since the tool has to get what the report says",
            "right": "Both, since text goes in and text comes out",
            "nlg": "NLG, since it writes five new sentences",
        },
        correct="right",
        why={
            "right": "Text in and text out: both sides. This lesson asks why the understanding side is so hard.",
            "nlu": "It has to get what the report says, but it also writes five sentences: text in and text out is both. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "nlg": "Gatt and Krahmer would call it NLG, since it writes text; in the course's simple sort, text goes in too, so it is both. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="Where people keep computational linguistics and NLP apart, what does computational linguistics want?",
        options={
            "systems": "Systems that work for users, such as a spelling app or a translator",
            "replace": "To replace linguists with programs that count words",
            "right": "To know how language works, with computers as a tool",
        },
        correct="right",
        why={
            "right": "That is the science side; NLP, where it is kept apart, is the engineering side. This lesson shows what makes both hard: language itself.",
            "systems": "Working systems are NLP, where people keep the two apart. Computational linguistics wants to know how language works. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "replace": "Computational linguistics uses computers to answer questions about language; it does not replace anyone. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Teacher strikes idle kids.* Read it once, and a strike of teachers leaves children without lessons. Read it again, and a teacher hits children who do nothing. You picked one reading without noticing; a program has to choose, and the words alone do not say which one was meant. The keyword program of the last lesson took a lock that won't open for a question about opening hours.

    This lesson shows why language makes such mistakes easy: one form can mean several things, one meaning has many forms, and what a sentence means depends on who says it where. It also counts how fast the choices add up.

    **An everyday example.** ‘Meet me at the bank’ is clear to two friends who always meet by the river, and to nobody else.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Kenneth Church and Ramesh Patil (1982) counted the ways to read sentences with a row of phrases such as *in the box* or *of products*, one phrase more each time. With nine such phrases in a row, how many readings did their program find?",
        options={
            "mid": "About 500",
            "right": "About 5,000",
            "many": "About 50,000",
        },
        reveal="**About 5,000**: 4862. Church and Patil: \"These predictions continue to hold with as many as nine prepositional phrases (4862 parse trees).\" *Counting the choices* in *The idea* shows where the number comes from.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Three reasons

    Hirschberg and Manning (2015) describe the first decades of the field: "scientists attempted to write down for computers the vocabularies and rules of human languages. This proved a difficult task, owing to the variability, ambiguity, and context-dependent interpretation of human languages." Those are the three reasons of this lesson, in another order: ambiguity, variation and context.

    ### Ambiguity: one form, several meanings

    **Ambiguity** is when a word or a sentence can mean more than one thing. A word can: in the textbook of Jurafsky and Martin, "The word ‘bank’ can mean: (1) a financial institution or (2) a sloping mound." A word can also have more than one role in a sentence. Hirschberg and Manning: "a star can be either an astronomical object or a person, and ‘star’ can be a noun or a verb." *Teacher strikes idle kids* is their example of a headline with "two interpretations", "depending on the noun, verb, and adjective assignments of the words in the sentence, as well as grammatical structure."

    Headlines are short, so they are often ambiguous. On a forum of copy editors, someone posted the headline *Violinist linked to JAL crash blossoms*, and someone else proposed calling such headlines crash blossoms. The copy editor John McIntyre told the story, and Ben Zimmer passed it on on the blog *Language Log* on August 26, 2009.

    A sentence can be ambiguous while every word in it is clear. Jurafsky and Martin: "Structural ambiguity occurs when the grammar can assign more than one parse to a sentence." A parse is one way to build a sentence up from its parts. In Groucho Marx's line *One morning I shot an elephant in my pajamas*, the phrase *in my pajamas* can belong to *elephant* or to *shot*. And *old men and women* can be old men and old women, or old men and women of any age.

    ### Counting the choices

    Each word with two meanings doubles the number of readings. In *Teacher strikes idle kids*, three words have two roles each: 2 × 2 × 2 = 8 readings, of which two make sense.

    Structure multiplies too. Church and Patil (1982) looked at sentences such as *Put the block in the box on the table*, where each phrase that starts with a preposition (*in*, *on*, *by*) can attach to more than one place. "These syntactic ambiguities grow ‘combinatorially’ with the number of prepositional phrases": two readings with two phrases, five with three, and "When a fourth PP is added, there are fourteen trees" (PP stands for prepositional phrase). These are the Catalan numbers: 1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862. The paper prints 469 for 429, a misprint: its own formula gives 429. Their parser, a program that finds the parses of a sentence, found exactly these counts on sentences such as *It was the number of products of products*, with up to nine phrases. That the counts are the Catalan numbers "was first pointed out to us by V. Pratt", they write: "We suspect that it is a generally well-known result in the formal language community, though its origin is unclear."
    """)
    return


@app.cell(hide_code=True)
def headline_cell():
    mo.Html(C.headline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Variation: one meaning, many forms

    It also works the other way round: one meaning can be written in many ways. Jurafsky and Martin: "Any particular piece of text that we study is produced by one or more specific speakers or writers, in a specific dialect of a specific language, at a specific time, in a specific place, for a specific function." Spelling differs: *color* and *colour*. Twitter posts might use features often used by speakers of African American English, they write, "such as constructions like iont (I don’t in Mainstream American English (MAE)), or talmbout corresponding to MAE talking about". People switch languages within one message: their example "Por primera vez veo a @username actually being hateful! it was beautiful:)" starts in Spanish and ends in English. And "Another dimension of variation is the genre": a text message is written differently from a law. For a program that compares letters, *colour* and *color* are two different words.

    ### Context: who says it, where

    What a sentence means can depend on what a reader knows about the speaker, the place and the world. Hector Levesque, Ernest Davis and Leora Morgenstern (2012) built a test on that, the Winograd Schema Challenge, named after Terry Winograd, who gave a first version of this example in 1972: "The town councillors refused to give the angry demonstrators a permit because they feared violence. Who feared violence?" With *advocated* instead of *feared*, the answer flips from the councillors to the demonstrators. Nothing in the grammar says who *they* are; a reader knows that councillors who fear violence refuse a permit. And *Can you open the window?* is rarely a question about what someone is able to do: in a warm room, it is a request.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **ambiguity** | when a word or a sentence can mean more than one thing |
        | variation | one meaning written in many forms: spelling, dialect, language, genre |
        | context | what a reader knows about the speaker, the place and the world |
        | parse | one way to build a sentence up from its parts |
        | `C.SENSES` | made-up meanings of eight English words: a dict from a word to a list of meanings |
        | `C.SENTENCES`, `C.VARIANTS`, `C.MESSAGES` | five sentences as lists of words, spelling variants with their standard form, and five messages |
        | `key in a_dict` | `True` when the dict has the key: `"bank" in C.SENSES` is `True` |
        | `total = total * n` | multiplies `total` by `n` and keeps the result under the same name |
        """),
        L.watch_out("Most ambiguities never bother a reader, who picks a meaning from context without noticing. That makes them easy to miss when you write rules for a program."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Count the meanings or roles of each word.
    2. Multiply them: that is the number of readings, before the structure.
    3. Count the places each phrase can attach to: with each extra phrase, the number grows faster.
    4. Map variants to one form, so that one meaning is not counted as several words.
    5. Ask what a reader needs to know that is not in the words: who says it, where, and about what.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell goes over the words of the second sentence, and prints each word that has a list of meanings in `C.SENSES`, with that list.
    """)
    return


@app.cell
def example_senses():
    words = C.SENTENCES[1]   # we saw a bat
    for word in words:
        if word in C.SENSES:
            print(word, C.SENSES[word])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two words have more than one meaning: *saw* three, *bat* two. So the four words can be read in 3 × 2 = 6 ways, and a reader picks one at once.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_readings():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. readings: how many ways to read

        Write `readings(words, senses)`: the number of ways to read the list `words`, one meaning for each word, with the dict `senses` from a word to its list of meanings. A word that is not in `senses` has one reading. Start a count at 1, go over the words with `for`, and multiply the count by the number of meanings of each word that is in `senses`. Replace the `...`. The check calls `readings(SENTENCES[0], SENSES)`, then the other sentences and dicts of its own.
        """),
        L.task("Multiply the numbers of meanings of the words in senses."),
    ])
    return


@app.cell(hide_code=True)
def exercise_readings():
    box_readings = L.code_box("readings", lines=6)
    box_readings
    return (box_readings,)


@app.cell(hide_code=True)
def check_readings(box_readings):
    result_readings = L.check("readings", box_readings.value)
    result_readings
    return (result_readings,)


@app.cell(hide_code=True)
def hints_readings():
    L.hints(
        "readings",
        [
            "*Counting the choices* in *The idea* says how the meanings of two words combine.",
            "How many times as large does the count get at a word with three meanings? And what happens at a word that is not in `senses`?",
            "`            count = ___`",
        ],
        why="`count = 1`, `for word in words:`, then `if word in senses:` with `count = count * len(senses[word])` under it, and after the loop `return count`. `readings(SENTENCES[0], SENSES)` gives `8`.",
    )
    return


@app.cell(hide_code=True)
def text_after_readings():
    mo.md(r"""
    Eight readings for the headline, six for *we saw a bat*, and twelve for *a fine light bat*: three words already give a dozen choices. The words alone do not say which one is meant.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_canonical():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. canonical: one form for one meaning

        `C.VARIANTS` is a dict from a spelling variant to its standard form, such as `"colour"` to `"color"`, and `C.MESSAGES` holds five short messages as lists of words. Write `canonical(words, variants)`: a new list with each word of `words` in its standard form: the value under the word in `variants` if the word is a key, and else the word itself. Use `if` and `else`, and `.append` to add to the new list. Replace the `...`. The check calls `canonical(MESSAGES[0], VARIANTS)`, then the other messages and dicts of its own.
        """),
        L.task("Replace each variant by its standard form, and keep the other words as they are."),
    ])
    return


@app.cell(hide_code=True)
def exercise_canonical():
    box_canonical = L.code_box("canonical", lines=8)
    box_canonical
    return (box_canonical,)


@app.cell(hide_code=True)
def check_canonical(box_canonical):
    result_canonical = L.check("canonical", box_canonical.value)
    result_canonical
    return (result_canonical,)


@app.cell(hide_code=True)
def hints_canonical():
    L.hints(
        "canonical",
        [
            "Look at what should happen to *thx* and to *for* in the first message.",
            "Which of the two words is a key of `variants`, and what do you add to the new list for each of them?",
            "`            out.append(___)`",
        ],
        why="`out = []`, `for word in words:`, then `if word in variants:` with `out.append(variants[word])` under it, `else:` with `out.append(word)`, and after the loop `return out`. `canonical(MESSAGES[0], VARIANTS)` gives `[\"thanks\", \"for\", \"the\", \"color\", \"tips\"]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_canonical():
    mo.md(r"""
    The first two messages were written differently and mean the same; now a program sees the same list. But the dict only knows the variants someone wrote down: *thnx* or *colr* would slip through.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: add phrases

    Add phrases such as *in the box* to *Put the block*, and see how many ways there are to read the sentence.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_phrases = mo.ui.slider(0, 9, value=2, label="Phrases", show_value=True)
    demo_phrases
    return (demo_phrases,)


@app.cell(hide_code=True)
def demo_look(demo_phrases):
    mo.Html(C.parses_picture(demo_phrases.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_trophy():
    mo.md(r"""
    ### Exercise 3. A trophy and a suitcase

    Levesque, Davis and Morgenstern give this pair: "The trophy doesn’t fit in the brown suitcase because it’s too big. What is too big?" The answer is the trophy. With *too small* instead of *too big*, it is the suitcase.
    """)
    return


@app.cell(hide_code=True)
def choice_trophy_cell():
    choice_trophy = L.choice(
        "trophy",
        question="What does a program need to get both sentences right?",
        options={
            "right": "Knowledge of the world: a thing only fits into something bigger",
            "grammar": "A rule of grammar: *it* points to the subject of the sentence",
            "near": "A rule of position: *it* points to the closest noun before it, which is the suitcase",
        },
        correct="right",
        why={
            "right": "The two sentences differ in one word, and their grammar is the same. What flips the answer is knowing that big things do not fit into small ones: context, not form.",
            "grammar": "Then both sentences would get the trophy, but with *too small* the answer is the suitcase. The grammar of the two is the same. See *Context* in *The idea*.",
            "near": "Then both would get the suitcase, but with *too big* it is the trophy. The positions of the words do not change between the two. See *Context* in *The idea*.",
        },
    )
    choice_trophy
    return (choice_trophy,)


@app.cell(hide_code=True)
def text_exercise_blossoms():
    mo.md(r"""
    ### Exercise 4. A crash blossom

    The headline that gave crash blossoms their name: *Violinist linked to JAL crash blossoms*. The story was about a violinist whose career blossomed after a parent died in a Japan Airlines (JAL) crash in 1985.
    """)
    return


@app.cell(hide_code=True)
def choice_blossoms_cell():
    choice_blossoms = L.choice(
        "blossoms",
        question="What makes the headline easy to misread?",
        options={
            "know": "Readers would need to know about the crash to tell which violinist is meant",
            "right": "Blossoms can be read as a noun, while the headline means it as a verb",
            "spelling": "*Linked* is written in an unusual way that a program does not recognize",
        },
        correct="right",
        why={
            "right": "Read as a noun, *crash blossoms* are some kind of blossoms from a crash, and the violinist is linked to them. Read *blossoms* as a verb, and the career blossoms. It is the role of one word, as in *Teacher strikes idle kids*.",
            "know": "Which violinist is meant is not what trips readers up: it is whether *blossoms* is a noun or a verb. See *Ambiguity* in *The idea*.",
            "spelling": "Every word is spelled as usual. The trouble is that *blossoms* can be a noun or a verb. See *Ambiguity* in *The idea*.",
        },
    )
    choice_blossoms
    return (choice_blossoms,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What NLP is](/course/open/nlp-what-is-nlp?lang=en) gave NLP two sides, and a program with keywords that got two requests wrong. This lesson showed why such mistakes are easy: one form can mean several things (ambiguity), one meaning has many forms (variation), and what a sentence means can depend on what a reader knows (context). The readings multiply word by word, and the ways to attach phrases grow faster still. The next lesson, *From rules to statistics to neural networks*, shows how NLP tried to cope: first with rules written by hand, then by counting in large amounts of text.

    **An everyday example.** A cryptic crossword clue often works because one word can mean two things: the puzzle maker counts on you picking the wrong one first.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_blossoms, choice_trophy, result_canonical, result_readings):
    L.wrap_up(result_readings, result_canonical, choice_trophy, choice_blossoms)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
