# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-simplify. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-simplify", "en")
    C = L.checks   # the course's way of cutting text, the letters about the green bin, the books, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en), and one about [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sentences():
    quick_sentences = L.choice(
        "quick-sentences",
        question="In *Cutting text into sentences and words*, `ends_sentence(word, abbreviations)` says whether a word cut at spaces ends a sentence. The text *Mr. Darcy smiled. Elizabeth laughed.* is cut at spaces, and `\"mr.\"` is in the list. For how many of its words does `ends_sentence` give `True`?",
        options={
            "three": "3: *Mr.*, *smiled.* and *laughed.* each end with a period",
            "right": "2: *smiled.* and *laughed.*",
            "one": "1: only *laughed.*, the last word of the text",
            "none": "None: the function needs the next word to decide",
        },
        correct="right",
        why={
            "right": "*Mr.* is in the list, so `ends_sentence` gives `False` for it; *smiled.* and *laughed.* end with a period and are not abbreviations. This lesson cuts sentences the same way, and the list matters: without it, every *Mr.* in *Pride and Prejudice* would end a sentence.",
            "three": "*Mr.* ends with a period, but `ends_sentence` first looks it up in the list of abbreviations. What does it give for a word in the list? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "one": "*smiled.* ends with a period too, and it is not in the list. Does `ends_sentence` look at where a word stands in the text? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "none": "`ends_sentence` gets one word and the list, nothing else, and decides from those. What does it give for *laughed.*? See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
        },
    )
    quick_sentences
    return (quick_sentences,)


@app.cell(hide_code=True)
def quick_check_findall():
    quick_findall = L.choice(
        "quick-findall",
        question='In *Finding patterns: regular expressions* you searched with `re.findall`. What does `re.findall(r"\\d+", "Room 12 on floor 3")` give?',
        options={
            "digits": '`["1", "2", "3"]`: one match for each digit of the text',
            "right": '`["12", "3"]`',
            "numbers": "`[12, 3]`, two whole numbers",
            "one": '`["12 on floor 3"]`: from the first digit to the last',
        },
        correct="right",
        why={
            "right": "`\\d+` takes a run of digits as long as it can: *12*, then *3*, and `re.findall` gives each match as a string. Exercise 1 of this lesson finds runs of vowels in the same way.",
            "digits": "Without `+`, each digit would be a match of its own. What does `+` add to `\\d`? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "numbers": "`re.findall` gives the matches as pieces of the text. Are those numbers or strings? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
            "one": "`\\d` matches digits only, and *on floor* has none. Can one match hold the letters between 12 and 3? See [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en).",
        },
    )
    quick_findall
    return (quick_findall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A text can be right and still too hard for the people it is written for. In 1948, Rudolf Flesch published a way to put a number on how hard a text is to read: "RE = 206.835 - .846 wl - 1.015 sl." Here *wl* is the length of the words, in syllables per 100 words, and *sl* the length of the sentences, in words; the score falls "between 0 (practically unreadable) and 100 (easy for any literate person)." Flesch named it Formula A: "Formula A measures length (the longer the words and sentences, the harder to read)".

    Governments care about readable texts too. The editors of Rijksoverheid.nl, the website of the Dutch government, write new texts at language level B1, a level that the great majority of the Dutch population understands, according to the government's guidelines for its websites. And in 2025, Statistics Netherlands (CBS) reported that about three million adults of 16 to 75 in the Netherlands have low literacy.

    In this lesson you build Flesch's formula from its parts: you count syllables, words and sentences, and you score eight books, today's Dutch sentences and two versions of a letter. Then you find out what such a formula cannot see.

    **An everyday example.** A letter from the city explains in one sentence of 43 words when the green bin will be emptied. The person who wrote it finds it clear; a neighbor who is still learning the language reads it three times and puts the bin out on the wrong day.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_harder = L.guess(
        "guess",
        question="*Pride and Prejudice* (1813) and *Moby-Dick* (1851) are two of the eight books of the course. Count the words per sentence and the syllables per word of each whole book, and put them into Flesch's formula. Which of the two gets the lower score, which means: harder to read?",
        options={
            "moby-far": "*Moby-Dick*, by far: its chapters about whales and ships are full of long, rare words",
            "moby": "*Moby-Dick*, by a few points",
            "right": "*Pride and Prejudice*",
            "same": "They get exactly the same score",
        },
        reveal="*Pride and Prejudice*: 53.1, against 56.9 for *Moby-Dick*, both *fairly difficult* in Flesch's table. Its sentences are a little shorter, 20.8 words on average against 21.2, but its words are longer: 1.57 syllables per word against 1.52, in words such as *acquaintance* and *civility*. In the formula, 0.05 syllables per word more cost 84.6 × 0.05, about 4 points, while 0.4 words per sentence more cost only 1.015 × 0.4, about 0.4 points.",
    )
    guess_harder
    return (guess_harder,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A **readability formula** estimates how hard a text is to read from things a program can count. Flesch's **reading ease** uses two averages: the words per sentence and the syllables per word. Written per word instead of per 100 words, the formula is

    $$\text{reading ease} = 206.835 - 1.015 \cdot \frac{\text{words}}{\text{sentences}} - 84.6 \cdot \frac{\text{syllables}}{\text{words}}$$

    The longer the sentences and the words, the lower the score. Flesch gave a table of what the scores mean, with the kind of magazine that typically scores that way:

    | Score | Reading ease | Typical magazine |
    |---|---|---|
    | 90 to 100 | very easy | comics |
    | 80 to 90 | easy | pulp fiction |
    | 70 to 80 | fairly easy | slick fiction |
    | 60 to 70 | standard | digests |
    | 50 to 60 | fairly difficult | quality |
    | 30 to 50 | difficult | academic |
    | 0 to 30 | very difficult | scientific |

    Nothing keeps a score between 0 and 100: a text of one-syllable words in sentences of one word scores 206.835 − 1.015 − 84.6 = 121.22, and a very hard text scores below 0.

    1. **Dutch.** Flesch made the formula for English. In 1960, W. H. Douma of the Landbouwhogeschool in Wageningen, the agricultural university that is now Wageningen University, adapted it for Dutch farm magazines. Douma compared English passages with their Dutch translations, found the Dutch longer at every level, and assumed that Dutch words and sentences are both about 10% longer. Flesch's weights 84.6 and 1.015, rounded to 85 and 1.02 and divided by 1.1, give Douma's 77 and 0.93: the **Flesch-Douma** formula. Douma's table has the same seven bands as Flesch's, except that *fairly difficult* runs from 45 to 60 and *difficult* from 30 to 45.

    $$\text{Flesch-Douma} = 206.84 - 0.93 \cdot \frac{\text{words}}{\text{sentences}} - 77 \cdot \frac{\text{syllables}}{\text{words}}$$

    2. **Syllables.** A program does not hear syllables, so this lesson counts them with a rule: every group of vowels in a row, a, e, i, o, u or y, is one syllable, and every word has at least one. *Simple* has two groups, *i* and *e*; *leesbaarheid* has three, *ee*, *aa* and *ei*. The rule misses a silent *e*, as in *make*; two vowels in a row that belong to two syllables, as in *created*; and the Dutch *ë* of *ideeën*, which starts a new syllable but is not one of the six letters.
    3. **Sentences and words.** The course cuts a text into sentences as `ends_sentence` of *Cutting text into sentences and words* does: a word cut at spaces ends a sentence when it ends with `.`, `?` or `!`, unless it is an abbreviation or an initial. The course looks past a closing quotation mark or bracket, and adds English abbreviations such as *Mrs.* to the list. The words of a sentence are its runs of letters, so *Mr.* becomes the word *Mr*.
    4. **Simplifying.** Advaith Siddharthan, 2014: "Text simplification, defined narrowly, is the process of reducing the linguistic complexity of a text, while still retaining the original information and meaning." By hand, that means shorter sentences, common words instead of rare ones, and the main point first. The Council of Europe's Common European Framework of Reference for Languages (CEFR) describes six levels, from A1 to C2, and what a reader at level B1 can do: "Can understand the main points of clear standard input on familiar matters regularly encountered in work, school, leisure, etc."
    5. **What a formula cannot see.** A formula sees the length of words and sentences, nothing else. Bruce, Rubin and Starr wrote in 1981, in a report for the Center for the Study of Reading of the University of Illinois: "Because most of the formulas include only sentence length and word difficulty as factors, they can account only indirectly for other factors that make a particular text difficult, such as degree of discourse cohesion". A text rewritten to score better, with its long sentences cut and the words that linked them gone, can get harder: "Thus, the adapted text may actually be more difficult than the original." A simpler text can also have lost information. And the examples programs learn to simplify from are often not simpler: Wei Xu, Chris Callison-Burch and Courtney Napoles looked at pairs of sentences from English Wikipedia and Simple English Wikipedia that researchers used to train such programs, and wrote in 2015: "Our manual inspection reveals that about 50% of the sentence pairs in the PWKP corpus are not simplifications." In Dutch, LiNT, made at Utrecht University, looks at more than length: how common the words are, how many content words come with each verb, how concrete the nouns are, and how far apart words that belong together stand.

    The picture shows a letter about the green bin twice: in official language and in plain language, each scored with Flesch's formula.
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
    | **readability** | how easy a text is to read and understand |
    | **readability formula** | a formula that estimates readability from things a program can count, such as the length of sentences and words |
    | **reading ease** | Flesch's readability formula (1948): 206.835, minus 1.015 times the words per sentence, minus 84.6 times the syllables per word |
    | **Flesch-Douma** | Douma's version of the formula for Dutch (1960), with smaller weights: 0.93 and 77 |
    | **syllable** | a part of a word that is said as one beat, with one vowel sound at its heart: *sim* and *ple* in *simple* |
    | **text simplification** | rewriting a text so that it is easier to read, keeping its information and meaning |
    | **language level B1** | the third of the six levels of the CEFR, from A1 to C2: the level of a reader who understands the main points of clear standard language about familiar things |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To score how readable a text is:

    1. Cut the text into sentences, and each sentence into words.
    2. Count the syllables of each word: its groups of vowels, at least one.
    3. Count the words, the sentences and the syllables of the whole text.
    4. Divide: the words per sentence, and the syllables per word.
    5. Fill in the formula, Flesch's for English and Douma's for Dutch, and look up the score in the table.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take the letter in plain language, in the language of this lesson: `C.LETTERS` holds both letters in both languages. `C.sentences` cuts a text into sentences, each a list of its words, and `C.syllables` counts the syllables of one word by the rule of this lesson. The cell prints, for each sentence, its number of words and the syllables of each word:
    """)
    return


@app.cell
def example_cell():
    _letter = C.LETTERS[L.lang]["plain"]
    for _sentence in C.sentences(_letter):
        print(len(_sentence), [C.syllables(_w) for _w in _sentence])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The letter has 12 + 9 + 5 + 5 = 31 words in 4 sentences, and the syllables of its words add up to 41: 7.75 words per sentence and about 1.32 syllables per word. Flesch's reading ease is 206.835 − 1.015 × 31 / 4 − 84.6 × 41 / 31, about 87.1: easy. The rule gives *once* two syllables, where a reader hears one, and *service* three instead of two: the silent *e* again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_syllables():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. syllables: groups of vowels

        Write `syllables`, which gets `word`, one word, and gives back its number of syllables by the rule of this lesson: the groups of vowels in a row, a, e, i, o, u or y, in the word in small letters, and at least 1. In the start code, `groups` should hold the groups that `re.findall` finds. Replace the two `...`.

        The check calls `syllables("simple")`, which gives `2`, and `syllables("leesbaarheid")`, with a word of the Dutch lesson, which gives `3`. Then cases of its own: a word without vowels, a word with a capital, a word with a y, two vowels in a row, and words where the rule and a reader disagree, such as *make*.
        """),
        L.task("Count the groups of vowels in a word, and give at least 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_syllables():
    box_syllables = L.code_box("syllables", lines=4)
    box_syllables
    return (box_syllables,)


@app.cell(hide_code=True)
def check_syllables(box_syllables):
    result_syllables = L.check("syllables", box_syllables.value)
    result_syllables
    return (result_syllables,)


@app.cell(hide_code=True)
def hints_syllables():
    L.hints(
        "syllables",
        [
            "`re.findall(pattern, text)` gives every match of a pattern in a text as a list, and the character class `[aeiouy]` matches one of the six vowels. `word.lower()` gives the word in small letters. `max` gives the largest of the numbers it gets: `max(1, 0)` is 1.",
            "With `+` after the class, one match is a whole group of vowels in a row, so the *oo* of *boot* is one group. The number of syllables is how many groups the list holds, but never less than 1.",
            "`    return max(1, ___)`",
        ],
        why='`groups = re.findall(r"[aeiouy]+", word.lower())` and `return max(1, len(groups))`.',
    )
    return


@app.cell(hide_code=True)
def text_after_syllables():
    mo.md(r"""
    The rule gets most words of the picture right. It misses a silent *e* (*make*, *service*), two vowels in a row that belong to two syllables (*created*, *museum*, *theater*, *bioscoop*), a syllable without a vowel letter (*rhythm*), and the *ë* of *ideeën*. The formula only uses the average over all the words of a text, and long words with many groups of vowels still get many syllables.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. counts: words, sentences and syllables

        Write `counts`, which gets `sentences`, a text cut into sentences, each a list of words such as `C.sentences` gives, and gives back three numbers: how many words, how many sentences and how many syllables the text has, in that order. The check gives your function the course's `syllables`. Replace the two `...`: the first with two lines that count the word and its syllables, and the `return` with the three numbers.

        The check calls `counts([["We", "start", "on", "Monday"], ["Call", "us", "today"]])`, which gives `(7, 2, 9)`, and `counts([["Bel", "ons"], ["We", "beginnen", "op", "maandag"]])`, with the words of the Dutch lesson, which gives `(6, 2, 9)`. Then cases of its own: no sentences, an empty sentence, a word without vowels, and the four letters about the green bin.
        """),
        L.task("Count the words, the sentences and the syllables of a text cut into sentences."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("counts", lines=9)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "counts",
        [
            "`n_words` and `n_syllables` start at 0 and grow one word at a time, as a running total does. `syllables` is the course's function: `syllables(\"Monday\")` is 2. `len(sentences)` is the number of sentences.",
            "Inside the inner loop, each word adds 1 to the words and its syllables to the syllables. After both loops, give back the three numbers with commas between them: the words, the sentences, the syllables.",
            "`            n_syllables = n_syllables + ___`",
        ],
        why="`n_words = n_words + 1`, `n_syllables = n_syllables + syllables(word)` and `return n_words, len(sentences), n_syllables`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    The first sentence of the letter in official language has 43 words; the longest sentence of the plain letter has 12. Counting sentences is less simple than it looks. By the course's count, *Pride and Prejudice* has 127,858 words in 6,156 sentences. Without the list of abbreviations, the course would find 7,310 sentences: the abbreviations of the list come 1,154 times in the book, *Mr.* and *Mrs.* alone 1,135 times, and each would end a sentence. The words per sentence would drop from 20.8 to 17.5, and Flesch's score would rise from 53.1 to 56.4.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flesch():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. flesch: the reading ease

        Write `flesch`, which gets `words`, `sentences` and `syllables`, the three numbers that `counts` gives, and gives back Flesch's reading ease, without rounding. Replace the `...`.

        The check calls `flesch(31, 4, 41)`, with the counts of the plain letter in English, which gives about `87.08`, and `flesch(33, 4, 51)`, with those of the plain letter in Dutch, which gives about `67.72`. Then cases of its own: a text of one-syllable words in sentences of one word, the two letters in official language, and three of the books.
        """),
        L.task("Fill in Flesch's formula from the words, the sentences and the syllables."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flesch():
    box_flesch = L.code_box("flesch", lines=3)
    box_flesch
    return (box_flesch,)


@app.cell(hide_code=True)
def check_flesch(box_flesch):
    result_flesch = L.check("flesch", box_flesch.value)
    result_flesch
    return (result_flesch,)


@app.cell(hide_code=True)
def hints_flesch():
    L.hints(
        "flesch",
        [
            "`words / sentences` is the number of words per sentence. The formula starts at 206.835 and takes off two parts: 1.015 times the words per sentence, and 84.6 times the other average. `return` gives the score back.",
            "The other average is the syllables per word: the number of syllables divided by the number of words. Python does `*` and `/` before `-`, so the formula needs no parentheses.",
            "`    return 206.835 - 1.015 * words / sentences - 84.6 * ___`",
        ],
        why="`return 206.835 - 1.015 * words / sentences - 84.6 * syllables / words`.",
    )
    return


@app.cell(hide_code=True)
def text_after_flesch():
    mo.md(r"""
    By Flesch's table, *Alice's Adventures in Wonderland* is fairly easy, at 70.9, and *Frankenstein* is the hardest of the English books, at 50.3: its sentences are the longest, 22.6 words on average. In these books, the differences between books are larger than between the languages: *Eline Vere* has 14.0 words per sentence, fewer than any English book, and even Flesch's formula gives it 64.7, more than *Pride and Prejudice*, *Frankenstein* or *Moby-Dick*. Douma's smaller weights give every text a higher score than Flesch's: 77.4 for *Eline Vere*. The Dutch sentences of Tatoeba, written by volunteers in today's Dutch, have 6.3 words each on average and score 88.9 with Douma's formula: easy.

    ### Try it: a text of your own

    The box starts with the letter in official language. Rewrite it in plain language, or paste a text of your own, and see the scores change. First try to make the letter harder to follow without lowering its score. Then write a version that a neighbor who is still learning the language would understand, and leave nothing out.
    """)
    return


@app.cell(hide_code=True)
def try_write():
    try_text = mo.ui.text_area(value=C.LETTERS[L.lang]["hard"], label="Your text", rows=6, full_width=True)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ### Exercise 4. What does the formula see?

    The letter *From next month, we empty the green bin once every two weeks, and we start on the first Monday. We never empty it on holidays. Do you have a question? Call the city service desk.* scores 79.5 with Flesch's formula. Each change below makes a new letter. Put each change with what it does to the score.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en) found where sentences end despite abbreviations, and this lesson counts on it: without its list, *Pride and Prejudice* gets more than a thousand sentences too many. [Finding patterns: regular expressions](/course/open/nlp-regex?lang=en) gave the character class with `+` that finds the groups of vowels. [Scoring a summary: ROUGE](/course/open/nlp-rouge?lang=en) scored a text against references that people wrote; a readability formula needs no reference at all, and sees even less: the length of words and sentences. The next lesson builds a program that answers people with nothing but rules.

    **An everyday example.** Some word processors show the Flesch score of a document. The score can tell the writer when the sentences are long; it cannot tell whether the document still says what the writer means.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_syllables, result_counts, result_flesch, result_changes):
    L.wrap_up(result_syllables, result_counts, result_flesch, result_changes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
