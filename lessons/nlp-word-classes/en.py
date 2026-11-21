# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-word-classes. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-word-classes", "en")
    C = L.checks   # the sentences of two treebanks, English and Dutch, with the tag of every word


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Words and their parts](/course/open/nlp-morphemes?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inflect():
    quick_inflect = L.choice(
        "quick-inflect",
        question="Which pair of words is made by inflection?",
        options={
            "right": "*walk* and *walked*",
            "ful": "*care* and *careful*, since *-ful* is a suffix",
            "dog": "*dog* and *doghouse*",
        },
        correct="right",
        why={
            "right": "*-ed* marks the past, and *walked* is still a verb, as *walk* is: inflection makes the forms of one word and keeps its kind. This lesson is about those kinds of words.",
            "ful": "*-ful* is a suffix, but it makes a new word: the noun *care* becomes the adjective *careful*. That is derivation. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "dog": "*doghouse* is made of two words, *dog* and *house*: a compound. See *Three ways to make words* in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_inflect
    return (quick_inflect,)


@app.cell(hide_code=True)
def quick_check_cut():
    quick_cut = L.choice(
        "quick-cut",
        question="In *Words and their parts*, `cut_suffix` cuts off the first ending of the list that the word ends in, with at least 3 letters left, and gives back both parts, or the word as one part. What does `cut_suffix(\"speed\", [\"ing\", \"ed\", \"s\"])` give?",
        options={
            "right": "`['spe', 'ed']`",
            "word": "`['speed']`, since *spe* is not an English word",
            "both": "`['speed', 'ed']`",
        },
        correct="right",
        why={
            "right": "*speed* ends in *-ed* and leaves 3 letters, so `cut_suffix` cuts there, although *speed* is no past form. The function only looks at the letters of one word; in this lesson, the words around a word help too.",
            "word": "`cut_suffix` does not check whether what is left is a word: it only checks the ending and the 3 letters. See Exercise 1 in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
            "both": "`cut_suffix` gives back what is left and the ending, `[rest, suffix]`, and what is left of *speed* is *spe*. See Exercise 1 in [Words and their parts](/course/open/nlp-morphemes?lang=en).",
        },
    )
    quick_cut
    return (quick_cut,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Words and their parts* cut words into pieces. A sentence is more than a row of words, though: each word has a role, and words that belong together form groups. **Syntax** is the way words are arranged together into groups and sentences. Daniel Jurafsky and James H. Martin, in their textbook *Speech and Language Processing* (draft of August 19, 2026), write that the word comes from the Greek *sýntaxis*, "setting out together or arrangement".

    A program that knows what kind of word it sees knows more than the letters. Jurafsky and Martin call the kinds of words "useful clues to sentence structure and meaning": whether a word is a noun or a verb tells you about the words likely to stand next to it, since in English, nouns come after words such as *the* and after adjectives, and verbs after nouns. The same letters can be different kinds of word, though. In *I can open a can*, the first *can* says what you are able to do, and the second is a thing, a tin.

    This lesson names the kinds of words, counts them in thousands of real sentences that people labeled word by word, in English and in Dutch, and finds the words that are more than one kind.

    **An everyday example.** The pieces in chess are of a few kinds, and the kind of a piece says how it may move, wherever it stands on the board.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Jurafsky and Martin show a table for two large collections of English text in which every word has a label for its kind. Call a word ambiguous when it has more than one label somewhere in such a collection. What share of the words in running text, counting every time a word occurs, is ambiguous?",
        options={
            "few": "Fewer than 1 in 10",
            "third": "About a third",
            "right": "More than half",
        },
        reveal="**More than half.** In their table, 55 in 100 words of the *Wall Street Journal* texts and 67 in 100 of the Brown texts can have more than one label, with a set of 45 labels. Counted as different words, only 14 to 15 in 100 can: the ambiguous words are few, but very common, such as *that*, *back*, *down*, *put* and *set*. The picture under Exercise 2 counts this in the two collections of this lesson.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Kinds of words

    In *We had a fantastic time*, *time* is a noun, *had* a verb and *fantastic* an adjective. Such a kind of word is called a **part of speech** (plural: parts of speech, abbreviated POS). Jurafsky and Martin start with a Greek grammar of about 100 B.C., by Dionysius Thrax "or perhaps someone else": it described eight parts of speech, among them noun, verb, pronoun and preposition. Earlier scholars, such as Aristotle, had lists of their own, but these eight became the basis for describing European languages for the next 2,000 years.

    You can often tell the part of speech of a word by how it behaves in a sentence. In English, in Jurafsky and Martin's examples, a noun can come after *a* or *this* (*a goat*, *this bandwidth*) and can take a plural (*goats*); a verb has forms such as *eat*, *eats*, *eating* and *eaten*. Those forms come from inflection, which, as *Words and their parts* showed, keeps the kind of a word.

    ### Open and closed classes

    Jurafsky and Martin sort the parts of speech into two groups. Nouns and verbs are **open classes**: new ones keep coming, such as *iPhone* or *to fax*. **Closed classes** hardly get new words: prepositions such as *of* and *in* are a closed class, and new ones are rarely made. The words of closed classes are mostly short, frequent words with a job in the grammar, such as *of*, *it*, *and* and *you*.

    Not every language has the same parts of speech, Jurafsky and Martin write. Some languages have no adjectives: in Korean, the words for English adjectives such as *beautiful* act as verbs. Many scholars believe that every human language has nouns and verbs; others have argued that some languages, such as Tongan and Riau Indonesian, do not even make that distinction.

    ### Seventeen tags for every language

    A **treebank** is a collection of sentences that people analyzed word by word: every word has a label for its part of speech, a **tag**, and every sentence has its structure as a tree, which a later lesson shows. For a long time, each treebank had its own tags. The Penn Treebank for English (Marcus, Santorini and Marcinkiewicz, 1993) had 36 tags for parts of speech, plus tags for punctuation and symbols.

    **Universal Dependencies** (UD) is a project that labels sentences of many languages in the same way: its website speaks of over 600 contributors and over 200 treebanks in over 150 languages. It has the same 17 tags for parts of speech in every language, in three groups:

    | Group | Tag | Part of speech | English | Dutch |
    |---|---|---|---|---|
    | open | `NOUN` | noun | *time*, *people* | *jaar* (year), *helft* (half) |
    | | `PROPN` | proper noun, a name | *Iraq*, *China* | *Nederland*, *Amsterdam* |
    | | `VERB` | verb | *have*, *get*, *know* | *komt* (comes), *heet* (is called) |
    | | `ADJ` | adjective | *good*, *new* | *eerste* (first), *nieuwe* (new) |
    | | `ADV` | adverb | *just*, *very*, *now* | *niet* (not), *ook* (also) |
    | | `INTJ` | interjection | *please*, *yes*, *wow* | *ja* (yes), *nee* (no) |
    | closed | `ADP` | preposition | *of*, *in*, *for* | *van* (of), *op* (on) |
    | | `AUX` | auxiliary verb | *is*, *will*, *can* | *is*, *heeft* (has), *zal* (will) |
    | | `CCONJ` | conjunction that joins two equal parts | *and*, *but*, *or* | *en* (and), *maar* (but) |
    | | `DET` | determiner | *the*, *a*, *this* | *de*, *het*, *een* (the, the, a) |
    | | `NUM` | number | *one*, *two*, *3* | *twee* (two), *drie* (three) |
    | | `PART` | particle | *to* in *to go*, *not* | (none in this treebank) |
    | | `PRON` | pronoun | *I*, *it*, *you* | *ik* (I), *wat* (what) |
    | | `SCONJ` | conjunction that makes one part of a sentence depend on another, as *that* in *I think that it works* | *that*, *if*, *because* | *dat* (that), *omdat* (because) |
    | other | `PUNCT` | punctuation | `.` `,` `?` | `.` `,` `?` |
    | | `SYM` | symbol | `$` `%` | `=` `km` |
    | | `X` | anything else | pieces of words, foreign words | *pct.*, *the* in a Dutch sentence |

    This lesson uses two treebanks of UD release 2.19 (November 2026), and every example in the table comes from them. *English-EWT* has sentences from the web: blogs, newsgroups, reviews and questions with their answers; the course left out its emails, which hold real names and addresses. *Dutch-Alpino* has sentences from Dutch newspapers and other written Dutch.

    ### Phrases

    Words that belong together form a **phrase**. A **noun phrase** is a noun with the words that belong to it: *a fantastic time*, with the determiner *a* and the adjective *fantastic* before the noun *time*. That is the order Jurafsky and Martin described: in English, nouns come after determiners and adjectives. Dutch does the same: *de eerste helft* (the first half). The next lesson shows how phrases are built, one inside the other.
    """)
    return


@app.cell(hide_code=True)
def tagged_cell():
    mo.Html(C.tagged_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **syntax** | the way words are arranged together into phrases and sentences |
        | **part of speech** | the kind of a word, such as noun or verb, by how it behaves in a sentence; abbreviated POS |
        | **open class**, **closed class** | parts of speech that keep getting new words, such as nouns, and ones that hardly do, such as prepositions |
        | **treebank** | a collection of sentences that people analyzed word by word, with a tree for each sentence |
        | **tag** | the label for the part of speech of a word, such as `NOUN` |
        | **Universal Dependencies** | a project that labels sentences of many languages in the same way, with 17 tags for parts of speech |
        | **noun phrase** | a noun with the words that belong to it, such as *a fantastic time* |
        | `C.EWT`, `C.ALPINO` | the sentences of the two treebanks, 11,723 English and 13,603 Dutch: a tuple of sentences, each a tuple of pairs `(word, tag)` |
        | `for word, tag in sentence:` | goes over the pairs, and gives the two parts of each pair a name of their own, as `for key, value in a_dict.items():` does |
        | `key in a_dict` | `True` when `key` is one of the keys of the dict |
        | `x not in a_list` | `True` when `x` is not one of the items of the list |
        """),
        L.watch_out("People gave the tags, following the guidelines of their treebank, and treebanks can choose differently for the same kind of word. Where English has the adverb *well*, Alpino tags *goed* (well) as `ADJ`, also in *het gaat goed* (it is going well): 153 of the 155 times *goed* occurs."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Go over every sentence, and over every pair `(word, tag)` of each sentence.
    2. To count, keep a dict: the first time you meet a tag, it gets the count 1, and after that, one more each time.
    3. To find the tags of a word, keep a dict from the word in small letters to a list, and add each tag that the word gets, once.
    4. Look at the numbers: which kinds take most of the words, and how often does one word have more than one tag?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell goes over the pairs of `C.EXAMPLE`, the English sentence of the picture, prints each word with its tag, and counts the nouns.
    """)
    return


@app.cell
def example_count():
    nouns = 0
    for word, tag in C.EXAMPLE:
        print(word, tag)
        if tag == "NOUN":
            nouns = nouns + 1
    print("nouns:", nouns)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *time* is the only noun, and the period has a tag too: `PUNCT`. `C.EXAMPLE` is one sentence; `C.EWT` holds 11,723 such sentences, and `C.ALPINO` 13,603. Exercise 1 counts every tag, in every sentence.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. count_tags: how many words of each kind

        Write `count_tags(sentences)`: give back a dict with each tag that occurs in `sentences` as a key, and as its value the number of words with that tag. Start with an empty dict, and go over every pair of every sentence. When the tag is already a key, its count goes up by one; otherwise, the tag gets the count 1. Give back the dict at the end. Replace the `...`. The check calls `count_tags(C.SAMPLE)`, two short sentences after examples of Jurafsky and Martin, *Book that flight.* and *Hand me that book.*, with tags the course gave them; then sentences of its own, also in Dutch, and no sentences at all.
        """),
        L.task("Go over every word of every sentence, and count how many words have each tag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count_tags", lines=9)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count_tags", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count_tags",
        [
            "Two loops, one inside the other: over the sentences, and over the pairs of a sentence, as in the worked example. Inside, an `if` asks whether the tag is already a key.",
            "What is the count of a tag that you meet for the first time? And why must the empty dict be made before the loops, and not inside them?",
            "`                counts[tag] = ___`",
        ],
        why="`counts = {}`, then `for sentence in sentences:` and inside it `for word, tag in sentence:`, with `if tag in counts:` and `counts[tag] = counts[tag] + 1` under it, and `else:` with `counts[tag] = 1` under it; after both loops `return counts`. `count_tags(C.SAMPLE)` gives `{'VERB': 2, 'DET': 2, 'NOUN': 2, 'PUNCT': 2, 'PRON': 1}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    The picture under the check shows your function on both treebanks, per 100 words. In both languages `NOUN` takes the most words, about 17 in 100. Dutch has more `ADP` and `DET` (*van*, *de*, *een*), and English more `PRON`. The few words of the closed classes, such as *the*, *of* and *and*, together take about 4 in 10 of all words.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. tags_per_word: one word, more than one tag

        Write `tags_per_word(sentences)`: give back a dict with each word, in small letters, as a key, and as its value the list of the tags it has in `sentences`, each tag once, in the order they first come. Start with an empty dict. For each pair, make the word small with `lower()`. When the small word is not a key yet, give it an empty list. Then, when the tag is not in its list yet, append the tag. Give back the dict at the end. Replace the `...`. The check calls `tags_per_word(C.SAMPLE)`, where *Book* is a `VERB` and *book* a `NOUN`; then sentences of its own.
        """),
        L.task("Give each word in small letters the list of its tags, each tag once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("tags_per_word", lines=10)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("tags_per_word", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "tags_per_word",
        [
            "Two loops as in Exercise 1. Inside, two `if`s, one after the other: the first makes the list for a new word, the second adds the tag.",
            "Why must the first `if` come before the second? What would go wrong for a word that you meet for the first time?",
            "`            if tag not in ___:`",
        ],
        why="`tags = {}`, then `for sentence in sentences:` and inside it `for word, tag in sentence:`, with `small = word.lower()`, then `if small not in tags:` with `tags[small] = []` under it, and `if tag not in tags[small]:` with `tags[small].append(tag)` under it; after both loops `return tags`. `tags_per_word(C.SAMPLE)` gives `{'book': ['VERB', 'NOUN'], 'that': ['DET'], 'flight': ['NOUN'], '.': ['PUNCT'], 'hand': ['VERB'], 'me': ['PRON']}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    In the picture under the check, *back* has six tags in EWT, and *loop* in Alpino is a noun and a verb. As in the guess, more than half of the words in running text have more than one tag somewhere, in both treebanks. Counted as different words, it is about 13 in 100 English words, and fewer than 5 in 100 Dutch words.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: the tags of a word

    Type a word, and see how often it has each tag in the two treebanks. Try *that*, *to* or *well*; Dutch *dat*, *een* or *weinig* (little); or *the*, which is a determiner almost every time.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="back", label="A word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 3. Same letters, a different tag

    The word in bold has the same letters each time. Put each one under the tag it has in its sentence. The sentences with *back* are examples from Jurafsky and Martin; the others are the course's own.
    """)
    return


@app.cell(hide_code=True)
def exercise_which():
    sort_which = L.sort("which")
    sort_which
    return (sort_which,)


@app.cell(hide_code=True)
def check_which(sort_which):
    result_which = L.check("which", sort_which)
    result_which
    return (result_which,)


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Exercise 4. What comes after a determiner?

    A program reads a determiner (`DET`), such as *the* or, in Dutch, *de*, and has to guess the tag of the next word before it sees that word.
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after",
        question="Which tag is the best guess?",
        options={
            "right": "NOUN, as in *the flight* and *de vlucht*",
            "adj": "ADJ, since determiners come before adjectives as often as before nouns",
            "propn": "PROPN, as in *the Netherlands*",
        },
        correct="right",
        why={
            "right": "After a `DET`, 57 in 100 words in EWT are a `NOUN`, and 63 in 100 in Alpino; `ADJ` comes second, with 25 and 21 in 100. That is what Jurafsky and Martin meant: the tag of a word says a lot about its neighbors. A later lesson builds a program that gives every word its tag.",
            "adj": "Adjectives often come after a determiner, as in *a fantastic time*, but nouns more than twice as often: after a `DET`, 57 in 100 words in EWT are a `NOUN` and 25 in 100 an `ADJ`. See *Phrases* in *The idea*.",
            "propn": "Names do come after a determiner now and then, as in *the Netherlands*, but after a `DET`, fewer than 9 in 100 words in EWT are a `PROPN`. See *Phrases* in *The idea*.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Words and their parts](/course/open/nlp-morphemes?lang=en) cut words into morphemes, and showed that inflection keeps the kind of a word. That kind is its part of speech. Universal Dependencies gives every language the same 17 tags, in open classes that keep growing and closed classes of short, frequent words. You counted the tags of an English and a Dutch treebank, and listed the tags of every word: the same letters can be a noun, a verb or an adverb, and the words around them decide which. Words also form phrases, such as the noun phrase *a fantastic time*. The next lesson shows how phrases are built, and how a few rules can make endless sentences.

    **An everyday example.** In a band, the same musician can play guitar in one song and drums in the next: you only know the role when you hear the song.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after, result_count, result_tags, result_which):
    L.wrap_up(result_count, result_tags, result_which, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
