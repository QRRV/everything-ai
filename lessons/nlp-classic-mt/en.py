# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-classic-mt. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-classic-mt", "en")
    C = L.checks   # the Dutch sentences of Tatoeba with their English translations, the dictionary, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en), and one about [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_root():
    quick_root = L.choice(
        "quick-root",
        question="In *Who depends on whom: dependency trees*, every word of a sentence hangs from one head, except the root. In *the late train left the city after midnight*, which word is the root?",
        options={
            "subject": "*train*, the subject, as the subject comes first and the rest hangs from it",
            "right": "*left*, the verb, from which the subject and the object hang",
            "object": "*city*, the object",
            "first": "*the*, the first word",
        },
        correct="right",
        why={
            "right": "The root is often the main verb: *train* hangs from *left* with `nsubj`, *city* with `obj` and *midnight* with `obl`. This lesson uses one such relation: in *ik zal de volgende bus nemen* (I will take the next bus), the helping verb *zal* hangs from the main verb *nemen*.",
            "subject": "*train* hangs from another word, with `nsubj`. Which word do the subject and the object both hang from? See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
            "object": "*city* hangs from another word, with `obj`. Which word do the subject and the object both hang from? See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
            "first": "*the* is a determiner: it hangs from *train*, with `det`. Which word has no head? See [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
        },
    )
    quick_root
    return (quick_root,)


@app.cell(hide_code=True)
def quick_check_in():
    quick_in = L.choice(
        "quick-in",
        question="In *Dictionaries*, `in` checks whether a key is in a dictionary. What does `\"cat\" in {\"kat\": \"cat\"}` give?",
        options={
            "true": "`True`, as *cat* is in the dictionary",
            "right": "`False`",
            "error": "An error: `in` works on a list or a text, and not on a dictionary",
            "key": "`\"kat\"`, the key that belongs to *cat*",
        },
        correct="right",
        why={
            "right": "`in` looks at the keys, and *cat* is a value: `\"kat\" in {\"kat\": \"cat\"}` gives `True`. In this lesson, a dictionary from Dutch words to English ones answers `in` for a Dutch word.",
            "true": "*cat* is in the dictionary, but as a value. What does `in` look at, the keys or the values? See [Dictionaries](/course/open/py-dicts?lang=en).",
            "error": "`in` works on a dictionary too, and gives `True` or `False`. What does it look at? See [Dictionaries](/course/open/py-dicts?lang=en).",
            "key": "`in` gives `True` or `False`, not a key. Is *cat* a key of this dictionary? See [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_in
    return (quick_in,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 1947, Warren Weaver of the Rockefeller Foundation wrote to the mathematician Norbert Wiener that the new computers might translate. Weaver knew nothing official about the new machine methods for breaking codes, but had guessed a good deal about them, and wondered whether translating was breaking a code: "When I look at an article in Russian, I say 'This is really written in English, but it has been coded in some strange symbols. I will now proceed to decode.'" In 1949, Weaver quoted the letter in a memorandum, *Translation*, written in the hope that it "might possibly serve in some small way as a stimulus to someone else".

    In January 1954, IBM and Georgetown University showed a computer in New York that translated Russian sentences into English. Hutchins, a historian of machine translation, wrote about it: "Although a small-scale experiment of just 250 words and six ‘grammar’ rules it raised expectations of automatic systems capable of high quality translation in the near future." Its rules could choose between two English words for a Russian word, and swap two neighboring words.

    In this lesson you translate Dutch into English with a dictionary of 478 words and a rule of your own, and measure on the sentences of Tatoeba how far that gets.

    **An everyday example.** A tourist with a pocket dictionary looks up each word of a sign and reads the translations in a row. *Verboden toegang* comes out as *forbidden access*: you understand it, but people write *No entry*.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_exact = L.guess(
        "guess",
        question="The course's dictionary has 478 common Dutch words, each with one English word. The lesson measures on 21,636 Dutch sentences of Tatoeba that have every word in it. Translate each of them word for word, and compare the result with the English translations that people of Tatoeba wrote, without capitals and punctuation, and with *I'll* written out as *i will*. For how many does word for word give exactly one of those translations?",
        options={
            "all": "Almost all of them",
            "half": "About half",
            "right": "About one in seven",
            "hundred": "Hardly any, about one in a hundred",
        },
        reveal="About one in seven: 2,979 of the 21,636, or 13.8%. Word for word gets short sentences right whose words line up one for one, such as *Ik ben moe*, *i am tired*. Another 1,037 (4.8%) have the right words in the wrong order: *Ik zal de volgende bus nemen* becomes *i will the next bus take*. The other 17,620 (81.4%) have other words: *Ik heb honger* becomes *i have hunger*, where people write *I'm hungry*.",
    )
    guess_exact
    return (guess_exact,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Machine translation** turns a text in one language, the **source language**, into a text in another, the **target language**. For decades, people wrote the rules for it by hand. Bernard Vauquois and Christian Boitet, who built such systems at the University of Grenoble, wrote: "The process of translation is divided into three phases: analysis, transfer, and generation." The analysis reads the source text, the transfer turns what the analysis found into the target language, and the generation writes the target text.

    How far the analysis goes sets the kind of system. The picture, known as the **Vauquois triangle** after Bernard Vauquois, shows three ways:

    - **Direct translation** stays at the bottom: it looks up each word in a dictionary and writes the translations in a row, with perhaps a few rules that move words. The system of Georgetown and IBM was of this kind.
    - **Transfer** first analyzes the source sentence into a structure, such as a dependency tree, turns that into a tree of the target language, and generates the target sentence from it.
    - An **interlingua** goes all the way up: the analysis writes down the meaning in a form that belongs to no language, and the generation writes that meaning in the target language.

    The higher a system climbs, the more work the analysis and the generation do, and the less is left for the transfer. Word for word breaks where two languages differ: in the order of the words, in words that one language needs and the other does not, and in **idioms**, fixed phrases whose meaning is not that of their words. With a bigger dictionary, *Het regent pijpenstelen* is *it rains pipe stems* word for word; people write *It's raining cats and dogs*.
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
    | **machine translation** (MT) | a program that translates a text from one language into another |
    | **source language**, **target language** | the language a text is translated from, and the language it is translated into |
    | **direct translation** | translating word by word with a dictionary, with at most a few rules that move words |
    | **transfer** | translating via a structure: analyze the source sentence into a tree, turn that into a tree of the target language, and generate the target sentence |
    | **interlingua** | a way to write down the meaning of a text that belongs to no language; a system with one translates via it |
    | **Vauquois triangle** | the picture of the three ways: the higher the analysis goes, the less is left for the transfer |
    | **idiom** | a fixed phrase whose meaning is not that of its words, such as *it's raining cats and dogs* |
    | **helping verb** | a verb that goes with a main verb, such as *zal* (will) or *heb* (have); `AUX` (auxiliary verb) in *Parts of speech and phrases*, and in UD it hangs from the main verb with the relation `aux` |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To translate a Dutch sentence by rules:

    1. Cut it into words, in small letters.
    2. Find the first helping verb, and see whether the last word is a verb form that can stand at the end, such as *nemen* (take) or *gelezen* (read).
    3. When both are there, move that verb to right after the helping verb, where English wants it.
    4. Look up each word in the dictionary. A word that is not in it, such as a name, stays as it is.
    5. Compare the result with the translations by people: exactly one of them, the same words in another order, or other words.

    Steps 2 and 3 are a small version of analysis and transfer. A transfer system finds the helping verb and the main verb in a tree, by the relation `aux`, and moves the verb there. This lesson moves it in the list of words, with a set of helping verbs in place of the tree: direct translation with one rule that moves words.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take *Ik zal de volgende bus nemen.* from Tatoeba, translated there as *I'll take the next bus.* All six words are in the dictionary, which the lesson has as `C.DICTIONARY`; `C.words` cuts a Dutch text into its words, in small letters. `C.pair` gives a sentence of Tatoeba with its translations, by its number there (`C.WORKED` is the number of this one), and `C.english_words` gives the words of an English text with contractions such as *I'll* written out:
    """)
    return


@app.cell
def example_cell():
    _dutch = C.words("Ik zal de volgende bus nemen.")
    print(_dutch, len(C.DICTIONARY))
    print(C.DICTIONARY["zal"], C.DICTIONARY["nemen"])
    _moved = _dutch[:2] + [_dutch[-1]] + _dutch[2:-1]
    print(_moved)
    print(C.pair(C.WORKED))
    print(C.english_words("I'll take the next bus."))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Word for word, the six words give *i will the next bus take*: the words of the translation by people, in another order. In the tree, *nemen* is the root and *zal* hangs from it with `aux`; Dutch puts the main verb at the end, English right after its helping verb. Dryer writes in the *World Atlas of Language Structures*: "in German and Dutch, the dominant order is SVO in main clauses lacking an auxiliary and SOV in subordinate clauses and clauses containing an auxiliary." SVO is subject, verb, object; SOV puts the verb after the object. The slices move *nemen* to right after *zal*: *ik zal nemen de volgende bus* is bad Dutch, but word for word it gives *i will take the next bus*, exactly the translation by people.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. word_for_word: translate with a dictionary

        Write `word_for_word`, which gets `words`, a list of Dutch words, and `dictionary`, a dictionary from Dutch words to English ones, such as `C.DICTIONARY`. Give back a new list with the English word of each word, in the same order. A word that is not in `dictionary` stays as it is. In the start code, `english` starts as an empty list, and the loop takes one word `w` at a time. Replace the two `...`.

        The check calls `word_for_word(["ik", "zie", "een", "kat"], {"ik": "i", "zie": "see", "een": "a"})`, which gives `["i", "see", "a", "kat"]`: *kat* is not in the dictionary, so it stays. Then `word_for_word(["de", "hond", "slaapt"], {"de": "the", "hond": "dog"})`, and cases of its own: an empty list, a word that comes twice, and sentences of Tatoeba with `C.DICTIONARY`.
        """),
        L.task("Translate a list of Dutch words word for word with a dictionary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words():
    box_words = L.code_box("word_for_word", lines=8)
    box_words
    return (box_words,)


@app.cell(hide_code=True)
def check_words(box_words):
    result_words = L.check("word_for_word", box_words.value)
    result_words
    return (result_words,)


@app.cell(hide_code=True)
def hints_words():
    L.hints(
        "word_for_word",
        [
            "For a word `w` that is a key of `dictionary`, look up its value with square brackets, as in *Dictionaries*, and add it to the end of the list with `english.append`. For any other word, `english.append` adds `w` itself.",
            "Without the `if`, looking up stops with a `KeyError` at the first word that is not in `dictionary`. And keep that word: a name such as *Tom*, or a word the dictionary lacks, must stay in the list, or the translation loses a word.",
            "`            english.append(___)`",
        ],
        why="In the `if`: `english.append(dictionary[w])`; in the `else`: `english.append(w)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_words():
    mo.md(r"""
    The dictionary covers short sentences of common words. The course's copy of Tatoeba has 147,001 Dutch sentences with an English translation, 169,641 translations in all. Of those sentences, 21,923 have every word in the dictionary; the lesson measures on 21,636 of them, and leaves out 287 because their translations use a gendered pronoun, which the course does not show. 3,707 of the 21,636 have more than one translation: people translate one sentence in different ways, such as *I'm tired* and *I am tired*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_outcome():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. outcome: how close is a translation?

        Write `outcome`, which gets `translation`, a list of English words, and `references`, a list of the translations by people, each a list of words. Give back `"same"` when `translation` is one of them; else `"order"` when one of them has the same words, each as often, in another order; and else `"words"`. The start code asks the two questions in that order. Replace the two `...`.

        The check calls `outcome(["the", "dog", "sleeps"], [["the", "dog", "is", "sleeping"], ["the", "dog", "sleeps"]])`, which gives `"same"`: it is the second translation. Then `outcome(["i", "will", "the", "bus", "take"], [["i", "will", "take", "the", "bus"]])`, which gives `"order"`, and cases of its own, with the word-for-word translations of the course (in the lesson: `C.direct()`, one for each of the 21,636 sentences) and the translations by people (`C.references()`, in the same order).
        """),
        L.task("Say whether a translation is exactly one of the translations by people, has their words in another order, or has other words."),
    ])
    return


@app.cell(hide_code=True)
def exercise_outcome():
    box_outcome = L.code_box("outcome", lines=7)
    box_outcome
    return (box_outcome,)


@app.cell(hide_code=True)
def check_outcome(box_outcome):
    result_outcome = L.check("outcome", box_outcome.value)
    result_outcome
    return (result_outcome,)


@app.cell(hide_code=True)
def hints_outcome():
    L.hints(
        "outcome",
        [
            "`in` asks whether a list is one of the items of a list of lists, as it does for any item. For the second question, `sorted` gives a new list with the words of `r` in order, as in *Tuples, sets and sorting*; compare it with `==` to the words of `translation`, sorted the same way.",
            "Sort, and do not make sets: a set keeps a word once, so *the dog sees the cat* would match *the cat sees dog*. And sort with `sorted`, not with `sort`: `sort` changes a list and gives back `None`.",
            "`        if sorted(r) == ___:`",
        ],
        why="`if translation in references:` and `if sorted(r) == sorted(translation):`.",
    )
    return


@app.cell(hide_code=True)
def text_after_outcome():
    mo.md(r"""
    Word for word gets 2,979 of the 21,636 sentences exactly right. In 1,037 the words are right and their order is wrong: a rule that moves words can help there. The 17,620 with other words need more than a rule of order. Some words are missing, as English needs *do* in a question where Dutch does not: *Waar woon je?* is *where live you*, and people write *Where do you live?* Some words mean something else in a fixed phrase: *Ik hou van je* is *i love of you*, as the dictionary has one English word for *van*, *of*. And some sentences are said differently: *Ik heb honger* is *I'm hungry*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. verb_after_helper: a rule that moves a verb

        Write `verb_after_helper`, which gets the Dutch `words` of a sentence, a set `helpers` of helping verbs and a set `verbs` of verb forms that can stand at the end, such as *nemen* and *gelezen*. When the last word is in `verbs` and a word of `helpers` comes before it, move the last word to right after the first such helping verb, and give back the new list. Else give back `words` as they are. In the start code, `last` is the last word, the loop goes over the places `i` of the other words, and the `if` in it finds the first helping verb. Replace the two `...`.

        The check calls `verb_after_helper(["ik", "heb", "het", "boek", "gelezen"], {"heb", "kan"}, {"gelezen", "zwemmen"})`, which gives `["ik", "heb", "gelezen", "het", "boek"]`, the order of *I have read the book*. Then `verb_after_helper(["ik", "kan", "snel", "zwemmen"], {"heb", "kan"}, {"gelezen", "zwemmen"})`, and cases of its own: a sentence without a helping verb, one whose last word is not a verb, a question, and sentences of Tatoeba with the course's sets `C.HELPERS` (33 helping verbs) and `C.VERBS` (66 verb forms).
        """),
        L.task("Move a verb at the end of a Dutch sentence to right after its helping verb, as English wants it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("verb_after_helper", lines=8)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("verb_after_helper", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "verb_after_helper",
        [
            "The `if` gives back `words` when `last` is not in `verbs`. In the loop, the new list is three pieces joined with `+`: the words up to and including the helping verb, `words[:i + 1]`; then `[last]`, a list with the last word alone; and then the words between the helping verb and the last word, a slice of `words`.",
            "Leave the last word out of the third piece, or it is in the list twice; and start that piece after the helping verb, or the helping verb is in the list twice. A negative number in a slice counts from the end, as in *Lists*.",
            "`            return words[:i + 1] + [last] + ___`",
        ],
        why="`if last not in verbs:` and `return words[:i + 1] + [last] + words[i + 1:-1]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rule():
    mo.md(r"""
    The rule changes the order of 5,361 of the 21,636 sentences. It puts 160 right that word for word got wrong, such as *Ik kan snel zwemmen*, now *i can swim fast*. But it breaks 133 that word for word got right. 72 of them are questions: the helping verb comes first, English puts the verb right after the subject, and *Kan ik helpen?* goes from *can i help* to *can help i*. In 38, *niet* stands before the verb: *Ik kan niet zwemmen* goes from *i can not swim* to *i can swim not*. And the rule takes *leven* for a verb, also where it is a noun, *life*: *Ik heb een geweldig leven* becomes *i have life a great*. On balance the rule wins 27: 3,006 sentences come out exactly, 13.9%. A rule for questions would have to recognize a question, a rule for *niet* would have to know where *not* goes, and their exceptions would need rules of their own. Vauquois and Boitet measured such systems in lines, not in rules: "rather than measure the number of grammatical rules, because they can be very simple or very complex, we measured the number of source lines of the grammars". Their small system for teaching, BEX-FEX, for which "the dictionaries are purposely very small (500 lexical units for each language)", had 4,320 lines of grammar: 420 for the forms of words, 2,500 for the structure of sentences, 300 for the transfer and 1,100 for the generation.

    ### Try it: translate a sentence

    Type a Dutch sentence and see it word for word, and after the rule of exercise 3. A word that is not in the dictionary stays as it is, with a dashed border. For one of the 21,636 sentences the lesson measures on, the box shows the translations by people too. Try *Ik kan snel zwemmen.*, *Kan ik helpen?* or *Wat wil je doen?*
    """)
    return


@app.cell(hide_code=True)
def box_sentence():
    try_sentence = mo.ui.text(value="Ik zal de volgende bus nemen.", label="A Dutch sentence", full_width=True)
    try_sentence
    return (try_sentence,)


@app.cell(hide_code=True)
def box_cell(try_sentence):
    mo.Html(C.box_picture(L.lang, try_sentence.value))
    return


@app.cell(hide_code=True)
def text_exercise_transfers():
    mo.md(r"""
    ### Exercise 4. Why an interlingua?

    A company wants a system that translates between 10 languages, in every direction. With transfer, each language needs an analysis and a generation, and each direction a transfer of its own.
    """)
    return


@app.cell(hide_code=True)
def choice_transfers_cell():
    transfers = L.choice(
        "transfers",
        question="How many transfers does that system need, each from one language into another?",
        options={
            "ten": "10, one for each language",
            "hundred": "100, ten for each language",
            "right": "90, one for each direction",
            "pairs": "45, one for each pair of languages, used in both directions",
        },
        correct="right",
        why={
            "right": "Each of the 10 languages can go into 9 others: 10 × 9 = 90 directions, and a transfer from Dutch into English does not translate English into Dutch. With an interlingua, the system needs only the analysis and the generation of each language: 20 parts. An eleventh language adds 2 parts, where transfer adds 20 transfers. That is the promise of the top of the triangle; the price is a way to write down meaning that fits every language.",
            "ten": "10 is the number of analyses. How many pairs of a source language and a target language are there?",
            "hundred": "That counts a transfer from each language into itself too. Is that a direction that needs translating?",
            "pairs": "A transfer goes one way: from a tree of the source language to a tree of the target language. Does a transfer from Dutch into English also translate English into Dutch?",
        },
    )
    transfers
    return (transfers,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en) you read a sentence as a tree; here one relation in it, `aux`, showed which verb to move, and a rule moved it in the list of words. [Writing rules versus learning from examples](/course/open/start-rules-vs-learning?lang=en) set rules against learning from examples: a dictionary of 478 words and one rule get 3,006 of the 21,636 sentences exactly right, and each new rule puts some sentences right and breaks others. Systems that learn from translated sentences took over from rules, as later lessons show. The next lesson, *Scoring a translation: BLEU and chrF*, scores a translation with partial credit, instead of right or wrong.

    **An everyday example.** A phrasebook for travelers holds whole sentences, such as *Where is the station?*, and not only words: a way around the problems of word for word that people found long before computers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_words, result_outcome, result_rule, transfers):
    L.wrap_up(result_words, result_outcome, result_rule, transfers)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
