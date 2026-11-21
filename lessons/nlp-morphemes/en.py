# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-morphemes. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-morphemes", "en")
    C = L.checks   # the course's lists of endings and of short words, in English and in Dutch


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
def quick_check_field():
    quick_field = L.choice(
        "quick-field",
        question="Where people keep computational linguistics and NLP apart, which project is computational linguistics?",
        options={
            "right": "Counting with a computer which word endings English uses most, to learn how English builds words",
            "app": "Building a search box for an online store that cuts words into parts, so that each query finds more products",
            "spell": "A spell checker that marks the words it does not know",
        },
        correct="right",
        why={
            "right": "It asks a question about language itself, and uses a computer to answer it. This lesson is about that question: how words are built.",
            "app": "A search box that has to work is the engineering side: NLP. Computational linguistics, where people keep it apart, asks questions about language itself. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "spell": "A spell checker is a tool that has to work: the engineering side, NLP. Computational linguistics asks questions about language itself. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_field
    return (quick_field,)


@app.cell(hide_code=True)
def quick_check_keyword():
    quick_keyword = L.choice(
        "quick-keyword",
        question="In *What NLP is*, `intent` looks for each keyword in the request, and `\"repair\"` is a keyword. What does it give for the request *My brakes need repairs.*?",
        options={
            "right": "`\"repair\"`, since the keyword is a piece of the word *repairs*",
            "unknown": "`\"unknown\"`, since *repairs* and *repair* are two different words",
            "error": "An error, since the request has a word that is longer than the keyword",
        },
        correct="right",
        why={
            "right": "`intent` looks for the keyword as a piece of the string, so *repairs* counts too: its *-s* only says that there is more than one. This lesson cuts words into such parts on purpose.",
            "unknown": "`intent` does not compare whole words: it looks for the keyword anywhere in the request, and *repair* is the start of *repairs*. See the exercise *intent* in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "error": "Looking for a piece of a string gives `True` or `False`, not an error, whatever the length of the words. See the exercise *intent* in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_keyword
    return (quick_keyword,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To a computer, *walk*, *walks*, *walked* and *walking* are four different strings. A search for *walking* that only looks for that string misses a page about a dog that *walked*. And people make new words all the time: you may never have seen *rewashable*, yet you know what it means, since you know its parts.

    This lesson cuts words into such parts, with a list of known endings and a list of known words, and sorts the three ways in which a language makes words.

    **An everyday example.** With a box of building bricks you build things that nobody built before, from bricks you know. A language makes new words from known parts in the same way.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A table from 2003 counts the different words in about 37 million words of English text, and in as much Dutch text. The English text has about 165 thousand different words. About how many does the Dutch text have?",
        options={
            "same": "About as many, some 170 thousand",
            "right": "Nearly three times as many",
            "ten": "Ten times as many, well over a million",
        },
        reveal="**Nearly three times as many: 462 thousand.** The table is in a paper by Roeland Ordelman, Arjan van Hessen and Franciska de Jong (2003), who took the numbers for the other languages from earlier work by Adda-Decker and Lamel; German had 650 thousand. They name compounding (joining words into one), next to the forms that endings make, as the main cause of the many different words in Dutch and German. With a list of 65 thousand words, 0.4 in 100 words of the English text were not in the list, and 2.85 in 100 of the Dutch text.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Parts of words

    You can cut *cats* into *cat* and *-s*, and *worked* into *work* and *-ed*. Daniel Jurafsky and James H. Martin call such parts morphemes in their textbook *Speech and Language Processing* (draft of August 19, 2026): "A morpheme is a minimal meaning-bearing unit in a language." **Morphology** is the study of how words are built from morphemes. The *-s* of *cats* has no meaning of its own the way *cat* has: it only says that there is more than one. The textbook *Essentials of Linguistics* (2nd edition, 2022) speaks of "meaning or grammatical function" for that reason. The course takes both: a **morpheme** is the smallest part of a word with a meaning or a job in the grammar.

    The **root** is the morpheme that gives the main meaning, such as *work* in *worked*. An **affix** adds to it: a **prefix** comes before, such as *un-* in *unhappy*, and a **suffix** after, such as *-s* in *cats*. *cat* can be a word on its own, and *-s* cannot; *Essentials* calls the first kind free and the second bound. A word can have many morphemes: Jurafsky and Martin cut *carefully* into *care-ful-ly*, and *Essentials* says that *non-renewability* has "at least five": *non-*, *re-*, *new*, *-abil* and *-ity*.

    ### Three ways to make words

    **Inflection** makes the forms of one word that a sentence needs: *walk*, *walks*, *walked*. Jurafsky and Martin name the *-s* (or *-es*) that marks more than one, and the *-ed* that marks the past. According to *Essentials*, inflection "never changes the category of its base": a verb stays a verb, and a noun stays a noun. A later lesson comes back to such kinds of words.

    **Derivation** makes a new word. From the noun *care* comes the adjective *careful*, and from that the adverb *carefully*; from *happy* comes *happiness*. Jurafsky and Martin write that derivation usually gives "a word of a different grammatical class than the root, often with a meaning hard to predict exactly." Usually, not always: *unhappy* is an adjective, as *happy* is. They also call the line between inflection and derivation "a continuum between two poles": not every ending falls neatly on one side.

    A **compound** is a word made of two or more words, such as *doghouse*. The last part usually says what kind of thing it is: a doghouse is a house. *Essentials* notes that English writes compounds in no fixed way: "Some compounds—typically older ones—are spelled without a space, while others are spelled with a hyphen, and many new compounds are spelled with spaces". Dutch writes them as one word: the official spelling rules of the Taalunie, the Dutch Language Union, say that the parts of a compound are written together, so the English *science fiction* is *sciencefiction* in Dutch. The Dutch reference grammar, the e-ANS, calls compounding the most important way to make new words in Dutch.

    ### Why a program needs the parts

    Jurafsky and Martin warn that no list of words, however big, holds every word that can occur, so a program keeps meeting words it does not know. Philipp Koehn and Kevin Knight (2003) write, about languages such as German and Dutch, that words which can be joined freely make the number of different words grow enormously, so that there is little data for each one. Rico Sennrich, Barry Haddow and Alexandra Birch (2016) looked at 100 rare words, ones not among the 50,000 most common, in the German text they trained a translation system on: 56 were compounds, and 21 were names.

    A program that knows the parts of words can do something with a word it never saw: *walked* through *walk*, and *doorbell* through *door* and *bell*. Such a program only sees letters, though, so it can also cut a word where no part is.
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **morphology** | the study of how words are built from morphemes |
        | **morpheme** | the smallest part of a word with a meaning or a job in the grammar, such as *cat* and *-s* in *cats* |
        | **root**, **affix** | the morpheme with the main meaning, and a morpheme that adds to it |
        | **prefix**, **suffix** | an affix before what it attaches to, such as *un-*, and one after it, such as *-ed* |
        | **inflection** | making the forms of one word, such as *walked* from *walk*; the kind of word stays the same |
        | **derivation** | making a new word with an affix, such as *careful* from *care*; often a word of another kind |
        | **compound** | a word made of two or more words, such as *doghouse* |
        | `C.SUFFIXES`, `C.WORDS` | the course's list of English endings, `["ing", "ed", "s"]`, and its list of twelve short English words |
        | `C.SUFFIXES_NL`, `C.WORDS_NL` | the same for Dutch |
        | `x in a_list` | `True` when `x` is one of the items of the list, and `False` when it is not |
        """),
        L.watch_out("In mathematics, *derivation* means something else: a chain of steps from one formula to the next, and the derivative of a function is something else again. In this lesson, derivation is making a new word from a word."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Make a list of known parts: endings such as *-ed*, or short words such as *door*.
    2. Try each part on the word: cut it off, and glue it back on to see whether you get the word again.
    3. Check what is left: long enough to be a root, or a known word too.
    4. Give back the parts you found, or the whole word as one part when nothing fits.
    5. Look at the parts: a list sees letters, not meaning, so it can cut where there is no morpheme.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell cuts the ending *-ed* off three words with a slice, glues it back on, and prints whether that gives the word again.
    """)
    return


@app.cell
def example_cut():
    for word in ["walked", "speed", "walks"]:
        rest = word[:len(word) - len("ed")]   # the word without its last two letters
        print(word, rest, rest + "ed" == word)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For *walked* and *speed*, gluing *-ed* back on gives the word again, and for *walks*, which ends in *-s*, it does not. But *speed* has no ending: it is one morpheme, and *spe* is no word. A list of endings sees letters, not meaning. Exercise 1 tries a whole list of endings, in order.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. cut_suffix: cut off a known ending

        `C.SUFFIXES` is the course's list of English endings, `["ing", "ed", "s"]`. Write `cut_suffix(word, suffixes)`: go over the endings of the list `suffixes` in order, and for each one compute what is left of the word without it, as in the worked example. For the first ending that the word really ends in, with at least 3 letters left, give back the list `[rest, suffix]`. When no ending fits, give back `[word]`: the word is one part. The 3 letters are the course's own choice: they keep *sing* from becoming *s* and *-ing*. Replace the `...`. The check calls `cut_suffix("walked", SUFFIXES)`, then other words and lists of its own, also Dutch words with `SUFFIXES_NL`.
        """),
        L.task("Try the endings in order, and give back what is left and the first ending that fits."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cut():
    box_cut = L.code_box("cut_suffix", lines=6)
    box_cut
    return (box_cut,)


@app.cell(hide_code=True)
def check_cut(box_cut):
    result_cut = L.check("cut_suffix", box_cut.value)
    result_cut
    return (result_cut,)


@app.cell(hide_code=True)
def hints_cut():
    L.hints(
        "cut_suffix",
        [
            "Use a `for` loop over `suffixes`, and for each ending do the steps of the worked example.",
            "What do you give back as soon as an ending fits and enough letters are left? And what after the loop, when none fitted?",
            "`        if ___ and len(rest) >= 3:`",
        ],
        why="`for suffix in suffixes:`, then `rest = word[:len(word) - len(suffix)]` and `if rest + suffix == word and len(rest) >= 3:` with `return [rest, suffix]` under it, and after the loop `return [word]`. `cut_suffix(\"walked\", SUFFIXES)` gives `['walk', 'ed']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_cut():
    mo.md(r"""
    The picture under the check shows your function on more English words. Like the course's own, it cuts *speed*, and gives *runn* for *running*, whose root is *run*: the spelling adds an *n*. Exercise 2 cuts a word into two known words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. split_compound: two known words

        `C.WORDS` is the course's list of short English words, such as `"door"` and `"bell"`. Write `split_compound(word, words)`: try the cuts from left to right with `for i in range(1, len(word)):`, with the left piece `word[:i]` and the right piece `word[i:]`. At the first cut where both pieces are in the list `words`, give back the list of the two pieces; when no cut works, give back `[word]`. Replace the `...`. The check calls `split_compound("doorbell", WORDS)`, then other words and lists of its own, also Dutch words with `WORDS_NL`.
        """),
        L.task("Try every cut, and give back the first one where both pieces are known words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split_compound", lines=7)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split_compound", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split_compound",
        [
            "A cut after letter `i` gives two pieces. Try every place with the `for` loop of the exercise, and look at both pieces each time.",
            "Which pieces must be in `words` before you split? And what do you give back when no place works?",
            "`        if ___:`",
        ],
        why="`for i in range(1, len(word)):`, then `left = word[:i]` and `right = word[i:]`, and `if left in words and right in words:` with `return [left, right]` under it, and after the loop `return [word]`. `split_compound(\"doorbell\", WORDS)` gives `['door', 'bell']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_split():
    mo.md(r"""
    Your function splits a compound into two known words. A Dutch compound can have more parts, and sometimes a sound between them, as in *pannenkoek*: the next lesson, *Splitting Dutch compounds*, comes back to that.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: cut a word of your own

    Type a word, and see what the course's two functions make of it, with the course's English endings and words. Try *walking*, *bus* or *footballs*: each function makes one cut at most.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="walked", label="A word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 3. Three ways to make a word

    Each word below was made from the word or words after it. Put it in the group of the way it was made.
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
def text_exercise_merge():
    mo.md(r"""
    ### Exercise 4. One word or two?

    A search box treats *walked* and *walk* as the same word, so that a search for one also finds the other. Someone wants it to do the same with *careful* and *care*.
    """)
    return


@app.cell(hide_code=True)
def choice_merge_cell():
    choice_merge = L.choice(
        "merge",
        question="Why is that riskier than with *walked* and *walk*?",
        options={
            "right": "*careful* does not mean the same as *care*",
            "same": "No riskier: both are a root and a suffix",
            "longer": "A suffix such as *-ful* is longer than *-ed*, so a program finds it less easily",
        },
        correct="right",
        why={
            "right": "*walked* is *walk* in the past. *careful* is a new word of another kind, with a meaning of its own: a search for *careful driving* should not bring up pages about *health care*. Jurafsky and Martin write that the meaning of a derived word is often hard to predict, so merging *careful* with *care* brings up pages about one in a search for the other.",
            "same": "Both are a root and a suffix, but *-ed* makes a form of the same word and *-ful* a new word. See *Three ways to make words* in *The idea*.",
            "longer": "A program finds *-ful* at the end of a word as easily as *-ed*. What matters is what the ending makes. See *Three ways to make words* in *The idea*.",
        },
    )
    choice_merge
    return (choice_merge,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What NLP is](/course/open/nlp-what-is-nlp?lang=en) found keywords as pieces of a string, and *repairs* counted for *repair* by luck. This lesson looked inside words: morphemes, the smallest parts with a meaning or a job in the grammar, studied in morphology. Inflection makes forms of one word, derivation makes new words, and a compound joins words; Dutch writes its compounds as one word. You cut known endings off with a list, split a word into two known words, and saw that a list, which sees only letters, can cut where there is no part. The next lesson, *Splitting Dutch compounds*, splits words of more than two parts, with the sounds that Dutch puts between them.

    **An everyday example.** A dictionary has an entry for *walk*, not for *walked*: you find *walked* because you know its parts.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_merge, result_cut, result_kinds, result_split):
    L.wrap_up(result_cut, result_split, result_kinds, choice_merge)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
