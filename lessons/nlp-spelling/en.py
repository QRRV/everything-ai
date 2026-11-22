# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-spelling. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-spelling", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en), and one about [Corpora and treebanks](/course/open/nlp-corpora?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="With insertions, deletions and substitutions that cost 1 each, what is the edit distance from *teh* to *the*?",
        options={
            "one": "1: a swap",
            "right": "2",
            "three": "3: one edit for each letter",
        },
        correct="right",
        why={
            "right": "Substitute *h* for *e* and *e* for *h*, or delete the *e* and insert it after the *h*: two edits either way. A swap of two neighboring letters costs two of these edits. This lesson adds the swap as a fourth kind of edit, so that *teh* is one edit from *the*.",
            "one": "A swap is not one of the three edits: with insertions, deletions and substitutions, swapping two neighboring letters takes two. This lesson adds the swap as a fourth kind of edit; see *Weights and swaps* in [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
            "three": "The *t* stays where it is, so only the *e* and the *h* need edits: two. See [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en).",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def quick_check_corpus():
    quick_corpus = L.choice(
        "quick-corpus",
        question="What is a corpus?",
        options={
            "web": "Every page on the web, as it is",
            "dictionary": "A list of the words of a language, with the meaning of each",
            "right": "Texts chosen to represent a language",
        },
        correct="right",
        why={
            "right": "A corpus is chosen: the Brown Corpus, for instance, holds about a million words of American English printed in 1961. This lesson counts how often each word occurs in five English books, and lets the counts decide which word a typo most likely stands for.",
            "web": "Nobody chose the pages of the web to represent a language, so the web is not a corpus in that sense. See [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
            "dictionary": "A list of words with their meanings is a dictionary, or a lexical database such as WordNet. A corpus holds running text, so you can count how often each word occurs in it. See [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
        },
    )
    quick_corpus
    return (quick_corpus,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    People make typos, and more often than you might think. Daniel Jurafsky and James Martin write in the appendix *Spelling Correction and the Noisy Channel* of *Speech and Language Processing*: "Estimates for the frequency of spelling errors in human-typed text vary from 1-2% for carefully retyping already printed text to 10-15% for web queries." A search engine, a phone keyboard and a word processor all suggest the word that was meant. Peter Norvig wrote the essay *How to Write a Spelling Corrector* after a week in 2007 in which, as the essay opens, "two friends (Dean and Bill) independently told me they were amazed at Google's spelling correction." This lesson builds a **spelling corrector** after Norvig's example, in its own code: it finds the words close to a typo with the edits of [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en), and lets the word counts of a corpus, as in [Corpora and treebanks](/course/open/nlp-corpora?lang=en), pick the word that was most likely meant.

    **An everyday example.** A word processor that underlines *seperate* and offers *separate* has found a word one edit away from what you typed.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The course took 500 words of at least four letters from random places in the five English books, and made one typo in each, at a random place: a letter deleted, inserted or replaced, or two neighboring letters swapped. How many of the 500 typos does the corrector of this lesson put right?",
        options={
            "few": "About 20%",
            "half": "About 50%",
            "right": "About 85%",
            "all": "Nearly 100%",
        },
        reveal="**431 of the 500**: 86.2%. Of the rest, 46 became another word, such as *tht*, meant as *that*, which became *the*; and 23 typos were words already, such as *hat* for *what*, which the corrector leaves alone. Exercise 3 runs your own corrector on the same 500 typos.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Every string one edit away

    Norvig's corrector starts from the typo, not from a list of words: it makes every string one edit away, and then looks which of them are words. In the section *How It Works: Some Python*, the essay says that "a simple edit to a word is a deletion (remove one letter), a transposition (swap two adjacent letters), a replacement (change one letter to another) or an insertion (add a letter)." Those are the three edits of the edit distance, plus the swap, which counts as one edit here. They are old: the documentation of Ralph Gorin's program SPELL, at Stanford, lists the same four kinds of error in its revision of December 1974: "one wrong letter", "one missing letter", "one extra letter" and "two transposed letters".

    Cut the word at every place, from before the first letter to after the last, and make each edit at that place. With the 26 letters *a* to *z*, Norvig counts: "For a word of length n, there will be n deletions, n-1 transpositions, 26n alterations, and 26(n+1) insertions, for a total of 54n+25 (of which a few are typically duplicates)." *thier* has five letters, so 5 + 4 + 130 + 156 = 295 strings, of which 286 are different: replacing a letter by the same letter gives *thier* back, five times, and inserting a letter next to the same letter gives the same string from both sides: *tthier*, *thhier*, and so on. Two edits reach far more strings: one edit from each of those 286 makes 36,860 different strings. Norvig: "This generates a much bigger set of possibilities, but usually only a few of them are known words".

    ### Keep the words, pick the most common

    Most of the 286 strings are not words. Three of them are words in the five English books of the course: *their*, 1,411 times, *thief*, 7 times, and *tier*, 2 times. A **known word** is a word that the corpus contains, and these three known words are the **candidates**: the words the writer may have meant. Norvig picks the candidate with the highest probability, and splits that probability into two parts: a language model P(c), "The probability that c appears as a word of English text.", and an **error model** P(w|c), "The probability that w would be typed in a text when the author meant c." Here w is the typed word and c a candidate. How the two parts combine is the subject of the lesson *Bayes' rule*, in the core of the course.

    The language model of this lesson is the simplest one there is: how often each word occurs in the corpus. The five books hold 551,664 words, 23,131 of them different when you count in small letters. The more often a word occurs, the more likely a writer meant it, so *their* wins. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze state the same rule in section 3.3.1 of *Introduction to Information Retrieval*: "The simplest notion of more common is to consider the number of occurrences of the term in the collection; thus if grunt occurs more often than grant, it is the chosen correction."
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### One edit before two

    Starting the program on a plane in 2007, Norvig had no data on typos for an error model, so the essay takes a shortcut: "I defined a trivial, flawed error model that says all known words of edit distance 1 are infinitely more probable than known words of edit distance 2, and infinitely less probable than a known word of edit distance 0." In other words: a typed word that is a known word stays as it is; any known word one edit away beats every word two edits away; and the count decides only between words at the same distance. Within two edits of *thier* lie 54 words of the books, among them *the*, which occurs 30,668 times. If all 54 competed, *the* would win; looking one edit away first keeps *their*.

    ### The noisy channel

    Jurafsky and Martin describe the same two parts as the **noisy channel model**. The idea is "to treat the misspelled word as if a correctly spelled word had been “distorted” by being passed through a noisy communication channel." They call the chance that an intended word comes out of the channel as the typed one the channel model, and the chance of the intended word itself the prior; they write x for the typed word and w for the intended one, where Norvig writes w and c. Mark Kernighan, Kenneth Church and William Gale described such a corrector in 1990, built at AT&T Bell Laboratories with word counts from the 1988 Associated Press newswire: 44 million words. On 329 typos with two candidates each, the more common word alone agreed with the majority of three human judges in 75% of the cases, the counts and the channel model together in 87%, and simply taking the first candidate in 52%.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **spelling corrector** | a program that gives, for a typed word, the word that was most likely meant |
        | **edit** | in this lesson: a deletion, an insertion, a substitution, or a swap of two neighboring letters (a transposition), each one edit |
        | **known word** | a word that the corpus contains: a key of the dictionary of counts |
        | **candidate** | a known word that the writer may have meant, such as *their* and *thief* for *thier* |
        | **error model** | P(w\|c): how likely a writer who meant the word c types w |
        | **noisy channel model** | a typo seen as an intended word that came out changed: the chance of the change, times the chance of the word |
        | **non-word error** | a typo that is not a word, such as *graffe* for *giraffe* |
        | **real-word error** | a typo that is a word itself, such as *form* for *from* |
        | `for c in letters` | goes over the letters of the string `letters`, one by one, as `for` goes over a list |
        """),
        L.watch_out("The corrector only knows the corpus. A word that the corpus lacks need not be a typo: names and rare words are missing, and the corrector changes them into a more common word. And a word that the corpus contains is not always spelled right: old books keep old spellings, and Norvig wrote in 2009 that \"the Internet is full of lousy spelling\"."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. If the typed word is in the corpus, give it back as it is.
    2. Make every string one edit away: at every place, a deletion and a swap, and for each letter an insertion and a substitution.
    3. Keep the strings that are words in the corpus. If there are any, give back the most common.
    4. Otherwise, do the same with the strings two edits away.
    5. If that finds nothing either, give back the typed word.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below cuts *thier* at place 2, into its first two letters and the rest, and makes the four kinds of edit there, with the letter *a* for the insertion and the substitution.
    """)
    return


@app.cell
def example_cell():
    word = "thier"
    start = word[:2]
    rest = word[2:]
    print(start, rest)
    print(start + rest[1:])                       # a deletion
    print(start + rest[1] + rest[0] + rest[2:])   # a swap
    print(start + "a" + rest)                     # an insertion
    print(start + "a" + rest[1:])                 # a substitution
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `rest[1:]` is `rest` without its first letter, so the deletion takes out the *i*. The swap puts the second letter of `rest` before the first, and gives *their*: the word that was meant is one swap away. The insertion puts the *a* between `start` and `rest`, and the substitution puts it in the place of the first letter of `rest`. At the last place, after the *r*, `rest` is empty: then only the insertions are possible, and a swap needs at least two letters in `rest`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_near():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. near_words: the words one edit away

        Write `near_words(word, counts, letters)`. Cut `word` at every place, from before the first letter to after the last, into `start` and `rest`, as in the worked example, and keep every string you make in a list: at each place the deletion and the swap, when `rest` has enough letters, and for each letter of `letters` the insertion and the substitution. Then give back the strings that are keys of `counts`, sorted, each once. For `near_words("thier", english, LETTERS)`, the result is `["their", "thief", "tier"]`. Replace the `...`. The check calls that and `near_words("hius", dutch, LETTERS)`, then cases of its own: *teh*; *acress*, with words at all four kinds of edit; *whal*, which needs a letter at the end; a letter too many at the start; *korrectud*, with no word one edit away; Dutch typos; and a small dictionary with only the letters `"ab"`, also for an empty word.
        """),
        L.task("Make every string one edit away from a word, and keep the ones that are words in the corpus."),
    ])
    return


@app.cell
def counts_cell():
    english = C.word_counts("english")
    dutch = C.word_counts("tatoeba")
    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    print(len(english), english["their"], english["thief"])
    print(len(dutch), dutch["huis"])
    return LETTERS, dutch, english


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    `english` holds how often each word occurs in the five English books, in small letters: 23,131 different words, with *their* 1,411 times. `dutch` does the same for the Dutch sentences of Tatoeba, a collection of sentences that volunteers write and translate: 38,682 different words.
    """)
    return


@app.cell(hide_code=True)
def exercise_near():
    box_near = L.code_box("near-words", lines=14)
    box_near
    return (box_near,)


@app.cell(hide_code=True)
def check_near(box_near):
    result_near = L.check("near-words", box_near.value)
    result_near
    return (result_near,)


@app.cell(hide_code=True)
def hints_near():
    L.hints(
        "near-words",
        [
            "Start with `found = []` and loop over the places with `for i in range(len(word) + 1)`. At each place, set `start = word[:i]` and `rest = word[i:]`, append the deletion when `rest` has a letter and the swap when it has two, and loop with `for c in letters` to append the insertion and, when `rest` has a letter, the substitution. After the loop, make a list of the strings `w` of `found` that are words, and give back `sorted(set(...))` of that list.",
            "Which word do you use to ask whether a key is in a dictionary?",
            r"`    return sorted(set([w for w in found if ___]))`",
        ],
        why=r"`found = []`; `for i in range(len(word) + 1):` with `start = word[:i]` and `rest = word[i:]`; when `len(rest) > 0`, append `start + rest[1:]`; when `len(rest) > 1`, append `start + rest[1] + rest[0] + rest[2:]`; `for c in letters:` append `start + c + rest`, and when `len(rest) > 0` also `start + c + rest[1:]`; after the loop, `return sorted(set([w for w in found if w in counts]))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_near():
    mo.md(r"""
    *korrectud* has no word one edit away: it differs from *corrected* in two letters, the *k* and the *u*. For *acress*, the example of Jurafsky and Martin, your function finds five words in the books, with all four kinds of edit: *access*, *acres*, *across*, *actress* and *caress*. Which of them did the writer mean? Exercise 2 picks one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. most_common: the most common word

        Write `most_common(words, counts)`: go over `words`, keep the word with the highest count so far together with that count, and give back the word. With no words at all, give back `None`. For `most_common(["their", "thief", "tier"], english)`, the result is `"their"`. Replace the `...`. The check calls that and `most_common(["heus", "hijs", "hits", "huis"], dutch)`, then cases of its own: the words one edit from other typos, with the most common word first, in the middle or last; a list of one word; an empty list; and the small dictionary. No list has two words with the same highest count.
        """),
        L.task("Give back the word of a list that occurs most often in the corpus."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("most-common", lines=8)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("most-common", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "most-common",
        [
            "Start with `best = None` and `best_count = 0`, and go over the words with `for w in words:`. When the count of `w` is higher than `best_count`, set `best` to `w` and `best_count` to that count. After the loop, give back `best`.",
            "Where do you find how often the word `w` occurs?",
            r"`        if ___ > best_count:`",
        ],
        why=r"`best = None` and `best_count = 0`; `for w in words:` and inside it `if counts[w] > best_count:` with `best = w` and `best_count = counts[w]`; after the loop, `return best`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    For *acress*, the counts pick *across*: 96 times in the books, against 5 for *acres*. Jurafsky and Martin's noisy channel model, which also counts how often people confuse letters, chooses *across* as well, with *actress* second. Their sentence calls the person "stellar and versatile", and they note: "The surrounding words make it clear that actress and not across was the intended word." Even with *Moby-Dick* among the books, *whal* becomes *what*, 1,712 times, and not *whale*, 1,113 times.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_correct():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. correct: the corrector

        Write `correct(word, counts)` with the steps of the recipe: give back `word` itself when it is in `counts`; otherwise the most common of the words one edit away, if there are any; otherwise the most common of the words two edits away, if there are any; otherwise `word` itself. The check gives your function the course's `near_words` and `most_common`, the letters `LETTERS`, and `far_words`, which takes a word and the counts and gives back the known words within two edits, sorted, each once. Two edits also reach every word one edit away, and the word itself, so `far_words` gives those too. For `correct("thier", english)`, the result is `"their"`. Replace the `...`. The check calls that and `correct("hius", dutch)`, then cases of its own: *then*, which is a word already; *korrectud*, two edits from its word; *bycycle*, whose word *bicycle* is not in the books; *peotry* and *acress*; and Dutch typos of each kind. *korrectud*, *bycycle* and *peotry* are examples from Norvig's essay.
        """),
        L.task("Put the steps together: keep a known word, then look one edit away, then two, then give up."),
    ])
    return


@app.cell
def course_cell(LETTERS, english):
    near_words = C.near_words
    most_common = C.most_common
    far_words = C.far_words
    print(near_words("korrectud", english, LETTERS))
    print(far_words("korrectud", english))
    return


@app.cell(hide_code=True)
def exercise_correct():
    box_correct = L.code_box("correct", lines=10)
    box_correct
    return (box_correct,)


@app.cell(hide_code=True)
def check_correct(box_correct):
    result_correct = L.check("correct", box_correct.value)
    result_correct
    return (result_correct,)


@app.cell(hide_code=True)
def hints_correct():
    L.hints(
        "correct",
        [
            "Four steps in the order of the recipe, each with its own `return`. Keep the words one edit away in a list `near`, from `near_words`, and the words two edits away in a list `far`, from `far_words`, which takes one value fewer than `near_words`.",
            "*korrectud* has nothing one edit away. What does `most_common` give back for an empty list, and how do you ask whether `near` holds any words?",
            r"`    if ___:`",
        ],
        why=r"`if word in counts:` `return word`; `near = near_words(word, counts, LETTERS)`; `if len(near) > 0:` `return most_common(near, counts)`; `far = far_words(word, counts)`; `if len(far) > 0:` `return most_common(far, counts)`; and at the end `return word`.",
    )
    return


@app.cell(hide_code=True)
def text_after_correct():
    mo.md(r"""
    ### How good is it?

    In the picture, your corrector puts most of the sentence right, but makes *tird* into *third*, not *tired*: both are one edit away, and *third* occurs 59 times in the books, *tired* 13. Norvig's goal was "a toy spelling corrector that achieves 80 or 90% accuracy at a processing speed of at least 10 words per second in about half a page of code." Norvig tested the essay's corrector on typos from Roger Mitton's Birkbeck spelling error corpus: 75% right on 270 typos that Norvig looked at while writing the program, and 68% right on 400 that were kept apart until the end, at 41 and 35 words per second. The section *Evaluation* concludes: "In conclusion, I met my goals for brevity, development time, and runtime speed, but not for accuracy."

    The section *Future Work* names the reasons. Some intended words are not in the corpus at all, 15 of the 270 and 43 of the 400, so the corrector cannot find them. The error model is too simple: *adres* becomes *acres*, one edit away, where *address* was meant, and the course's corrector does the same with the five books. Some typos are more than two edits from their word: 3 of the 270 and 23 of the 400. And: "There's actually a fourth (and best) way to improve: change the interface to correction to look at more context." Norvig adds: "I believe that a spelling corrector that scores 90% accuracy will need to use the context of the surrounding words to make a choice."

    ### Typos that are words

    Jurafsky and Martin separate two tasks: "Non-word spelling correction is the detection and correction of spelling errors that result in non-words (like graffe for giraffe)." By contrast, "real word spelling correction is the task of detecting and correcting spelling errors even if they accidentally result in an actual word of English (real-word errors)." They write that "A number of studies suggest that between 25% and 40% of spelling errors are valid English words", citing Kukich (1992), and two of their examples are "This used to belong to thew queen." and "The design an construction of the system will take more than a year." Manning, Raghavan and Schütze give an example from search, in section 3.3.2: "Such isolated-term correction fails to detect, for instance, that the query flew form Heathrow contains a misspelling of the term from – because each term in the query is correctly spelled in isolation." Your corrector keeps *form* and *an*, since both are words in the books. It does change *thew* into *the*, but only because *thew*, a rare word for strength, is not in the five books. To find a real-word error, a corrector has to look at the words around it.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own typo

    Type a word and choose a corpus: the five English books, the Dutch sentences of Tatoeba, or the three Dutch books, which are printed in a spelling from before 1934. The picture shows the known words one and two edits away, with their counts, and the word the corrector picks. Try a typo of your own, a name, or a Dutch word in an old spelling.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="speling", label="Your word")
    demo_corpus = mo.ui.dropdown(options=C.corpus_menu(L.lang), value="The five English books", label="Corpus")
    mo.vstack([demo_word, demo_corpus])
    return demo_corpus, demo_word


@app.cell(hide_code=True)
def demo_look(demo_corpus, demo_word):
    mo.Html(C.demo_picture(demo_word.value, demo_corpus.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Norvig wrote in 2009, in the chapter *Natural Language Corpus Data* of the book *Beautiful Data*: "To port the data-driven correct algorithm to another language, all we need is a large corpus of Latvian; the code remains unchanged." The same corrector works for Dutch with the sentences of Tatoeba: *mensch*, the old spelling of *mens*, Dutch for *human*, becomes *mensen*, *humans*. The three Dutch books give the old spellings instead. Choose them and type *mens*: the books write *mensch*, two edits away, and *mens* itself is not in them, so the corrector picks *men*, a common Dutch word for *one*, as in *one says*. According to the Dutch language society Onze Taal, the minister of education Marchant made a simpler spelling compulsory in schools in 1934, which dropped the *-ch* at the end of a word such as *mensch*. A joint Flemish-Dutch commission then prepared a new spelling, which came into use in Flanders in 1946 and in the Netherlands in 1947, and in 1954 the first Groene Boekje came out, the word list that set the official spelling for forty years.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. What does the corrector do?

    Each bold line is a typo and the word that was meant. Put it under what the corrector of this lesson does with it, with the five English books as its corpus.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_happens = L.sort("what-happens")
    sort_happens
    return (sort_happens,)


@app.cell(hide_code=True)
def check_sort(sort_happens):
    result_sort = L.check("what-happens", sort_happens)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [How far apart are two words? Edit distance](/course/open/nlp-edit-distance?lang=en) measured how far apart two strings are; this lesson turned that around and made every string one or two edits from a typo, with the swap as a fourth kind of edit. The counts of a corpus, as in [Corpora and treebanks](/course/open/nlp-corpora?lang=en), served as the language model that picks the most likely word, and a shortcut for the error model put one edit before two. The corrector looks at one word at a time, so typos that are words slip through: those take a language model over several words. The next lesson of the domain, *Which language is this?*, counts letters instead of words: every language has letter sequences of its own.

    **An everyday example.** A phone keyboard that changes *definately* into *definitely* as you type has done what your `correct` does: it found a known word close to what you typed, and chose a likely one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_correct, result_near, result_pick, result_sort):
    L.wrap_up(result_near, result_pick, result_correct, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
