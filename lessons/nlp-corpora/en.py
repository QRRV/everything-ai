# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-corpora. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-corpora", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="In *the late train left the city*, which word is the head of *late*?",
        options={
            "root": "*left*, the root of the sentence",
            "right": "*train*, the noun it describes",
            "next": "*the*, the word right before it",
        },
        correct="right",
        why={
            "right": "*Late* says something about *train*, so it hangs from *train*, with the relation amod. This lesson shows where trees like that come from: people label sentences by written guidelines, and the labeled sentences form a treebank.",
            "root": "*Left* is the root: the whole sentence hangs from it, but *late* does not hang from it directly. Which word does *late* say something about? See *Heads and dependents* in [Who depends on whom](/course/open/nlp-dependencies?lang=en).",
            "next": "Standing next to a word does not make it the head; *the* hangs from *train* itself. Which word does *late* say something about? See *Heads and dependents* in [Who depends on whom](/course/open/nlp-dependencies?lang=en).",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_root():
    quick_root = L.choice(
        "quick-root",
        question="In a sentence as a list of `(form, head, relation)`, a word has head `0`. What does that mean?",
        options={
            "first": "Its head is the first word of the sentence",
            "out": "It is left out of the tree",
            "right": "No other word is its head: it is the root",
        },
        correct="right",
        why={
            "right": "Head 0 marks the root, the word that every other word hangs from, directly or through others. The words count from 1, so 0 is no word of the sentence. In this lesson you look up heads by their number again.",
            "first": "The first word has number 1: the words count from 1. Head 0 is no word of the sentence. See *The files: CoNLL-U* in [Who depends on whom](/course/open/nlp-dependencies?lang=en).",
            "out": "Every word of a sentence is in its tree, also the root. See *A tree of words* in [Who depends on whom](/course/open/nlp-dependencies?lang=en).",
        },
    )
    quick_root
    return (quick_root,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every model of language in this course learns from text that somebody collected, and is tested on text that somebody labeled. Jurafsky and Martin, in their chapter on words and tokens: "Words don't appear out of nowhere. Any particular piece of text that we study is produced by one or more specific speakers or writers, in a specific dialect of a specific language, at a specific time, in a specific place, for a specific function." What a model knows about a language depends on the texts that went in. The trees of [Who depends on whom](/course/open/nlp-dependencies?lang=en) come from treebanks, sentences that people labeled one by one, and large language models learn from collections of trillions of tokens. Linguists, too, look at what people really write and say, rather than at what they think people write and say. For that they collect corpora, and they write down what goes in and how it is labeled.

    **An everyday example.** You are not sure whether to write *anyway* or *anyways*, so you search for both and compare how often they turn up. You just used the web as a collection of texts.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The Brown Corpus holds about a million words of American English, printed in 1961. Its most common word is *the*. How often does it occur?",
        options={
            "often": "About 1 word in 4",
            "right": "About 1 word in 14",
            "rare": "About 1 word in 100",
        },
        reveal="**About 1 word in 14**: in the copy of the Brown Corpus that comes with the Python toolkit NLTK, the course counts 69,971 times *the* among 1,013,319 words, so about 69,000 per million words. *Of* comes second, with 36,412. John Sinclair (2004) gives 69,970. In the British National Corpus, *the* takes 61,847 places per million words (Geoffrey Leech, Paul Rayson and Andrew Wilson, 2001): 64,420 in writing and 39,605 in speech. How many words a corpus holds depends on what you count as a word, and in the course's books the share differs from book to book, as the demo of this lesson shows.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A collection with a purpose

    A **corpus** (plural *corpora*) is a collection of texts, or of recorded speech, put together to study a language. The linguist John Sinclair defined it in 2004 as "a collection of pieces of language text in electronic form, selected according to external criteria to represent, as far as possible, a language or language variety as a source of data for linguistic research." External criteria means: chosen by where a text comes from and what it is for, not by the words in it. The contents, Sinclair writes, "should be selected without regard for the language they contain, but according to their communicative function in the community in which they arise."

    The Brown Corpus, put together by W. Nelson Francis and Henry Kučera at Brown University in the 1960s, was "the first computer-readable general corpus of texts prepared for linguistic research on modern English", according to the Corpus Resource Database of the University of Helsinki: 500 samples of about 2,000 words each, from American English printed in 1961, from news to science fiction. The British National Corpus (BNC), built from 1991 to 1994, holds 100 million words of British English, 90% written and 10% spoken. For Dutch, the Corpus Gesproken Nederlands (CGN) holds nearly 9 million words of recorded speech from the Netherlands and Flanders, collected from 1998 to 2004, and SoNaR-500 more than 500 million words of written Dutch.

    Bigger is not the same as better. Sinclair: "The World Wide Web is not a corpus, because its dimensions are unknown and constantly changing, and because it has not been designed from a linguistic perspective." The makers of the Penn Treebank wrote in 1993 that, by a strict line between a corpus and a collection "which may be much more opportunistic in construction", "the raw materials of the Penn Treebank form a collection." The text that large language models learn from is mostly a collection in that sense: FineWeb, a dataset of 2024, holds 15 trillion tokens from 96 snapshots of the web by Common Crawl. Jurafsky and Martin advise writing a datasheet for every corpus, which answers questions such as "Motivation: Why was the corpus collected, by whom, and who funded it?"
    """)
    return


@app.cell(hide_code=True)
def text_idea_labels():
    mo.md(r"""
    ### Labels on top: treebanks and lexical databases

    Many corpora carry labels that people added. A **treebank** is a corpus in which every sentence has its tree. Jurafsky and Martin: "A corpus in which every sentence is annotated with a parse tree is called a treebank." Usually a program makes a first tree and people correct it: "Treebanks are generally made by running a parser over each sentence and then having the resulting parse hand-corrected by human linguists." The Penn Treebank of Mitchell Marcus, Beatrice Santorini and Mary Ann Marcinkiewicz (1993) holds "over 4.5 million words of American English", tagged with a set of 36 tags for parts of speech and 12 more for punctuation and currency symbols; over half of it also got a skeleton of its syntax. Universal Dependencies, whose trees you read in [Who depends on whom](/course/open/nlp-dependencies?lang=en), held 402 treebanks in 226 languages in its release of November 15, 2026; two of them are in this lesson.

    A **lexical database** holds facts about words rather than running text. The best known is WordNet, of Princeton University; Jurafsky and Martin call it "the most commonly used resource for sense relations in English and many other languages". WordNet puts words that share a meaning into a **synset**, short for synonym set, and a word with several meanings is in several synsets. Version 3.0 has 117,659 synsets. Open Dutch WordNet (2016) links Dutch words to them. The picture shows one sentence of the course three times: as text, with the labels of a treebank, and with one of its words in a lexical database.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_concordance():
    mo.md(r"""
    ### Every use of a word: the concordance

    Before computers, a **concordance** was a book. The Catholic Encyclopedia of 1908 describes concordances of the Bible as "verbal indexes to the Bible, or lists of Biblical words arranged alphabetically with indications to enable the inquirer to find the passages of the Bible where the words occur." According to that encyclopedia, the first was completed in 1230 under Hugh of Saint-Cher, "assisted, it is said, by 500 fellow-Dominicans". A computer makes one in a moment. Sinclair: "A concordance also consists of short extracts from a corpus, but the extracts are chosen by a computer program". The usual form is **KWIC**, short for key word in context, a name that Hans Peter Luhn gave around 1960: one line for each place where the word stands, with the word in the middle and its neighbors to the left and the right. You read the horizontal line as text, Sinclair explains, "while the vertical dimension shows the similarities and differences between one line and the lines round about it." What a reader can take in at once, Sinclair puts at "in practice a screenful, some 23 lines".
    """)
    return


@app.cell(hide_code=True)
def text_idea_guidelines():
    mo.md(r"""
    ### Writing down how to label

    Labels only help when everyone labels the same way, so the makers of a corpus write **guidelines**: rules with examples of the hard cases. Beatrice Santorini, in the guidelines for the tags of the Penn Treebank (1990): "it is often quite difficult to decide which tag is appropriate in a particular context." One hard case is a word such as *off* after a verb: a preposition, tag IN, or a **particle** that belongs with the verb, tag RP? The two, Santorini writes, "are often difficult to distinguish from one another", and the guidelines give tests. A word is a particle "if it can either precede or follow a noun phrase object": *told off the team* and *told the team off* both work. A preposition has to come before its noun: *stepped off the train*, but not *stepped the train off*. (The sentences are the course's, after Santorini's examples.)

    Even with guidelines, people do not always agree. In the study of Marcus and colleagues, four people with training in linguistics tagged samples of the Brown Corpus. Their mean disagreement "was 7.2% for the tagging task and 4.1% for the correcting task", where they corrected the tags of a program. One text caused a large share of it: it held many symbols for chemical formulas, and "In the absence of an explicit guideline for tagging this case, the annotators had made different decisions". Jurafsky and Martin put the human level for tagging English at about 97%. How to measure agreement, and how Cohen's kappa corrects it for agreement by chance, is the subject of the course's lesson *Labeling well: guidelines and agreement*.

    Two guidelines can also differ. The Penn Treebank tags a particle RP; Universal Dependencies tags it ADP, as it does a preposition, and gives it the relation `compound:prt` to its verb. Both are consistent: they follow other rules.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **corpus** | a collection of texts or recorded speech, chosen to represent a language or a kind of language; plural *corpora* |
        | **treebank** | a corpus in which every sentence has its tree, checked by people |
        | **lexical database** | a collection of facts about words, such as which words share a meaning: WordNet is one |
        | **synset** | in WordNet, a set of words that share one meaning |
        | **concordance** | every place where a word stands in a corpus, each with the words around it |
        | **KWIC** | key word in context: a concordance with one line per place and the word in the middle |
        | **guideline** | a written rule for the people who label a corpus, with examples of the hard cases |
        | **particle** | a word such as *up* in *pick the kids up*, which belongs with the verb: tag RP in the Penn Treebank, relation `compound:prt` in Universal Dependencies |
        | `" ".join(words[a:b])` | the words from place `a` up to place `b`, without `b` itself, as one text with spaces between them |
        """),
        L.watch_out("A count from a corpus is a count of that corpus. *Whale* is a common word in *Moby-Dick* and a rare one in English. Before you draw a conclusion about a language, ask what went into the corpus."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Choose a corpus that fits your question, and read what went into it.
    2. For a concordance, go through the words, and at every place where your word stands, take a few words to its left and to its right.
    3. Near the start and the end of the text, take the words there are.
    4. Read the lines from top to bottom, and look for what comes back.
    5. For a question about structure, such as which particle goes with which verb, use a treebank: its labels tell apart what the words alone do not.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell gives you the words of *Alice's Adventures in Wonderland* by Lewis Carroll (1865), one of the course's eight books, as a list. `C.book_words("alice")` takes the text from the course's copy and cuts it into words, without punctuation and with the capitals as the book writes them. The second cell finds every place where *queen* stands, with or without capitals, and prints the first one with four words on each side.
    """)
    return


@app.cell
def book_cell():
    ALICE = C.book_words("alice")
    print(len(ALICE))
    print(ALICE[:12])
    return (ALICE,)


@app.cell
def example_look(ALICE):
    places = [i for i in range(len(ALICE)) if ALICE[i].lower() == "queen"]
    print(len(places), places[:5])
    i = places[0]
    print(ALICE[i - 4:i], ALICE[i], ALICE[i + 1:i + 5])
    print(" ".join(ALICE[i - 4:i]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The book has 26,776 words, by the course's count. Its first twelve are not the story: *Illustration* marks a picture, and then come the title, the author and the name of the edition. A corpus holds what its files hold, unless someone takes it out. *Queen* stands at 68 places, the first at place 11,101, and four words on each side give *An invitation from the Queen to play croquet The*: the last *The* starts the next sentence, since the list of words has no punctuation. `" ".join(...)` turns a list of words into one text, with a space between each two.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kwic():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. kwic: a concordance

        Write `kwic(words, target, width)`: give back a list with one line for every place where `target` stands in `words`, with or without capitals. `target` is in small letters. A line is a tuple of three texts: the `width` words before the word, joined with spaces; the word as the text writes it; and the `width` words after it, joined with spaces. Near the start of the list, the left side has fewer words, and near the end, the right side. For `kwic(ALICE, "rabbit", 3)`, the fourth line is `("suddenly a White", "Rabbit", "with pink eyes")`. Replace the `...`. The check calls `kwic(ALICE, "rabbit", 3)`, then cases of its own: a word near the start, a word in the first and in the last place, a word with a capital, a width of 0, a word that is not there, and *hatter* in *Alice*.
        """),
        L.task("Find every place of a word in a text, with its neighbors on both sides."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kwic():
    box_kwic = L.code_box("kwic", lines=11)
    box_kwic
    return (box_kwic,)


@app.cell(hide_code=True)
def check_kwic(box_kwic):
    result_kwic = L.check("kwic", box_kwic.value)
    result_kwic
    return (result_kwic,)


@app.cell(hide_code=True)
def hints_kwic():
    L.hints(
        "kwic",
        [
            "Start with an empty list `lines`, and go through the places `i` with `range(len(words))`. Where `words[i].lower()` equals `target`, the left side starts at `i - width`; make that 0 when it is below 0. Then join the two slices with `\" \".join(...)`, and add the tuple to `lines`.",
            "At which place does the right side start, and at which place does it stop?",
            "`            right = \" \".join(words[i + 1:___])`",
        ],
        why="`lines = []`, then `for i in range(len(words)):` with `if words[i].lower() == target:` under it. Inside: `start = i - width`, `if start < 0:` with `start = 0` under it, `left = \" \".join(words[start:i])`, `right = \" \".join(words[i + 1:i + 1 + width])` and `lines.append((left, words[i], right))`. After the loop: `return lines`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kwic():
    mo.md(r"""
    The picture under the check shows the first lines of your concordance of *rabbit*. The first two come from the contents of the book, not from the story. Read the lines from top to bottom: *White* comes back before *Rabbit* again and again, which you would not notice as easily reading the book from start to end. A concordance of *up* would show a limit of plain text: *up* in *walked up the hill* and in *picked the kids up* look the same. Exercise 2 uses a treebank to tell them apart.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: a concordance of your own

    Choose one of the course's eight books and type a word. The picture gives how often the word occurs, per million words, and the first 23 lines of its concordance, a screenful by Sinclair's measure, made with the course's own `kwic`. Try *whale* in *Moby-Dick* and in *Frankenstein*, *I* in *The Adventures of Sherlock Holmes* and in *Moby-Dick*, or, in the three Dutch books, *zoo*: the old spelling of *zo* (so).
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_book = mo.ui.dropdown(options=C.book_names(), value="Moby-Dick; or, The Whale (1851)", label="Book")
    demo_word = mo.ui.text(value="whale", label="Word")
    demo_width = mo.ui.slider(2, 8, value=5, label="Words on each side", show_value=True)
    mo.hstack([demo_book, demo_word, demo_width], justify="start", gap=1.5, wrap=True)
    return demo_book, demo_width, demo_word


@app.cell(hide_code=True)
def demo_look(demo_book, demo_width, demo_word):
    mo.Html(C.demo_picture(demo_book.value, demo_word.value, demo_width.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The same word can be common in one book and missing from the next. *Whale* stands 1,110 times in *Moby-Dick*, about 5,100 per million words, and 3 times in *Frankenstein*. *I* takes about 28,500 places per million in *The Adventures of Sherlock Holmes*, which Dr. Watson tells in the first person, and about 9,200 in *Moby-Dick*. Eight books are a collection, not a balanced corpus: a count from them says something about these books.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_particles():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. particles: what a treebank adds

        `SENTENCE` below holds the course's sentence *We picked the kids up and dropped them off*, labeled by the guidelines of Universal Dependencies, word by word as `(form, head, relation)`, as in [Who depends on whom](/course/open/nlp-dependencies?lang=en). Word number `i` is at place `i - 1` of the list, and head 0 is the root. *Up* and *off* have the relation `compound:prt`: they are particles of *picked* and *dropped*.

        Write `particles(sentence)`: give back a list of pairs `(head, i)`, one for every word whose relation is `compound:prt`: the number of its head, the verb, and its own number, in the order of the sentence. For `SENTENCE`, that is `[(2, 5), (7, 9)]`. Replace the `...`. The check calls `particles(SENTENCE)`, then cases of its own: a sentence without a particle, a word with the relation `compound`, a preposition, a particle before its verb, and a sentence of one word.
        """),
        L.task("Find the particles of a sentence by its labels, with the verb each belongs to."),
    ])
    return


@app.cell
def sentence_cell():
    SENTENCE = [
        ("We", 2, "nsubj"),
        ("picked", 0, "root"),
        ("the", 4, "det"),
        ("kids", 2, "obj"),
        ("up", 2, "compound:prt"),
        ("and", 7, "cc"),
        ("dropped", 2, "conj"),
        ("them", 7, "obj"),
        ("off", 7, "compound:prt"),
    ]
    return (SENTENCE,)


@app.cell(hide_code=True)
def exercise_particles():
    box_particles = L.code_box("particles", lines=7)
    box_particles
    return (box_particles,)


@app.cell(hide_code=True)
def check_particles(box_particles):
    result_particles = L.check("particles", box_particles.value)
    result_particles
    return (result_particles,)


@app.cell(hide_code=True)
def hints_particles():
    L.hints(
        "particles",
        [
            "Start with an empty list `found`, and go through the word numbers `i` from 1 up to and including `len(sentence)`. Take the three parts of word `i` apart into `form`, `head` and `relation`. When `relation` is `\"compound:prt\"`, add the pair `(head, i)` to `found`.",
            "At which place of the list is word number `i`?",
            "`        form, head, relation = ___`",
        ],
        why="`found = []`, then `for i in range(1, len(sentence) + 1):` with `form, head, relation = sentence[i - 1]` under it, and `if relation == \"compound:prt\":` with `found.append((head, i))`. After the loop: `return found`. `particles(SENTENCE)` gives `[(2, 5), (7, 9)]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_particles():
    mo.md(r"""
    The picture under the check runs your function on the sentences of UD_English-EWT from reviews and from answers to questions on the web, over 7,000 of them, and counts the pairs of verb and particle it finds. *Check out*, *pick up* and *figure out* lead. Some of the sentences have spelling mistakes, as texts from the web do, and a treebank keeps them: a corpus records how people write. A concordance of *up* finds every *up*, particle or preposition; the labels, which people added by a guideline, find only the particles.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Exercise 3. Label like the makers of a treebank

    Use Santorini's test on the word in italics in each bold sentence: can it also stand after the object? Then put each sentence under its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_label():
    sort_label = L.sort("label")
    sort_label
    return (sort_label,)


@app.cell(hide_code=True)
def check_label(sort_label):
    result_label = L.check("label", sort_label)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def text_exercise_purpose():
    mo.md(r"""
    ### Exercise 4. Which corpus?

    You want to find out how people speak Dutch in everyday life, and you can choose from three collections.
    """)
    return


@app.cell(hide_code=True)
def choice_purpose_cell():
    choice_purpose = L.choice(
        "purpose",
        question="Which collection fits that question best?",
        options={
            "right": "The CGN: nearly 9 million words of recorded Dutch speech",
            "size": "SoNaR-500: over 500 million words of written Dutch, from many genres",
            "wiki": "The Dutch Wikipedia: every page of it",
        },
        correct="right",
        why={
            "right": "A corpus is chosen for a purpose, and the CGN was made to represent spoken Dutch, from the Netherlands and from Flanders. It is far smaller than SoNaR-500, and still the better fit: no number of written words tells you how people speak.",
            "size": "SoNaR-500 is far larger, but it is written Dutch: no number of written words tells you how people speak. Which collection was chosen to represent speech?",
            "wiki": "Wikipedia is written, in the style of an encyclopedia, and nobody chose its pages to represent a language. Which collection was chosen to represent speech?",
        },
    )
    choice_purpose
    return (choice_purpose,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Who depends on whom](/course/open/nlp-dependencies?lang=en) read dependency trees; this lesson showed where such trees come from. A corpus is a collection of texts chosen for a purpose, and what it says about a language depends on what went in. A treebank adds a tree to every sentence and a lexical database facts about words, and people add those labels by written guidelines, which do not settle every case. You made a concordance, found the particles of a treebank by its labels, labeled sentences by a guideline yourself, and chose a corpus for a question. The next lesson of the domain counts the words of the course's books: Zipf's law and Heaps' law.

    **An everyday example.** The next time you search for a phrase to see how others use it, think of Sinclair: the web is huge, but nobody chose its pages to represent a language.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_purpose, result_kwic, result_label, result_particles):
    L.wrap_up(result_kwic, result_particles, result_label, choice_purpose)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
