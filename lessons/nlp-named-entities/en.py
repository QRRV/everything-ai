# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-named-entities. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-named-entities", "en")
    C = L.checks   # the sentences of Universal NER, the gazetteer, the tables of the tagger, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en) and [Tagging word classes](/course/open/nlp-pos-tagging?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_spans():
    quick_spans = L.choice(
        "quick-spans",
        question="A tagger gives the four tokens of *near the Mississippi River* the tags `O O I-LOC I-LOC`. Which spans does the course read from these tags, as the CoNLL script does?",
        options={
            "none": "None: a span has to start with a `B-`",
            "two": "Two places of one token each, `(2, 3, \"LOC\")` and `(3, 4, \"LOC\")`",
            "right": "One place, `(2, 4, \"LOC\")`",
            "drop": "One place, `(3, 4, \"LOC\")`: the first `I-` continues nothing, so the course drops it",
        },
        correct="right",
        why={
            "right": "An `I-` after an `O` starts a new span, as the CoNLL script reads it, and the next `I-LOC` makes that span one token longer: *Mississippi River*, from position 2 up to 4. This lesson reads the tags of its tagger the same way.",
            "none": "The CoNLL script, and the course with it, reads an `I-` that continues nothing as the start of a new span. Which span does that give here? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "two": "The second `I-LOC` comes right after a span of the same type, so it makes that span one token longer. Only a `B-` there would start a second place. See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
            "drop": "Nothing is dropped: the CoNLL script reads an `I-` after an `O` as the start of a span. Where does that span end? See [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).",
        },
    )
    quick_spans
    return (quick_spans,)


@app.cell(hide_code=True)
def quick_check_unknown():
    quick_unknown = L.choice(
        "quick-unknown",
        question="In *Tagging word classes*, the first tagger met a word that its training set did not have. Which tag did it give that word?",
        options={
            "propn": "`PROPN`, since a word the tagger never saw is most often a name",
            "right": "`NOUN`, the most frequent tag of all",
            "before": "The tag of the word before it",
            "skip": "None: the tagger left the word out",
        },
        correct="right",
        why={
            "right": "`NOUN` is the most frequent tag of the training set, so every unknown word got it, wherever it stood. A tagger for names can look at more of a word it does not know, and at the tag right before it: this lesson builds one.",
            "propn": "The first tagger knew nothing about an unknown word, so it gave every unknown word one and the same tag, and in EWT that is not `PROPN`. Which tag did it fall back to? See [Tagging word classes](/course/open/nlp-pos-tagging?lang=en).",
            "before": "The first tagger looked at each word alone, not at the words around it. What did it give a word it knew nothing about? See [Tagging word classes](/course/open/nlp-pos-tagging?lang=en).",
            "skip": "A tagger gives every token a tag, also a word it never saw: sequence labeling has as many labels as tokens. Which tag did it give? See [Tagging word classes](/course/open/nlp-pos-tagging?lang=en).",
        },
    )
    quick_unknown
    return (quick_unknown,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Who is a text about, where does it take place, and which companies does it name? A search engine, a news site and a helpdesk ask that of every text that comes in. Jurafsky and Martin: "Named entity tagging is a useful first step in lots of natural language processing tasks." They mention knowing what a consumer thinks of one particular entity, and add that entities are "a useful first stage in question answering, or for linking text to information in structured knowledge sources like Wikipedia". A program that hides the names of people in a text before it is shared has to find them first. This lesson finds names in two ways: with a list of names it already knows, and with a tagger that learns from sentences in which people marked the names.

    **An everyday example.** You read *a letter from Dr. Okonkwo in Tilburg*. You know *Tilburg* is a town, because you have heard of it. You never heard of *Okonkwo*, and still you know a person is meant: the *Dr.* before it and the capital letter tell you.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="The English sentences of Universal NER come in parts, as the treebank of *Tagging word classes* did: a training set of 8,773 sentences to learn from, and a test set of 1,471 other sentences to measure on. The test set has 802 names. How many of them does the training set have too, with exactly the same words?",
        options={
            "ten": "About 10 of every 100",
            "quarter": "About 25 of every 100",
            "right": "About 40 of every 100",
            "most": "About 70 of every 100: most names come back",
        },
        reveal="**About 40:** 324 of the 802 names of the test set are in the training set with the same words, 40 in 100. The other 478 are new: people, shops and places that the training set never names. A list of the names of the training set can find only those 324.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Named entities

    Jurafsky and Martin: "A named entity is, roughly speaking, anything that can be referred to with a proper name: a person, a location, an organization." Finding them is a task with its own name: "The task of named entity recognition (NER) is to find spans of text that constitute proper names and tag the type of the entity." The spans and the types are those of [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en): Universal NER marks three types, `PER`, `LOC` and `ORG`. Jurafsky and Martin also name a fourth type that many datasets use, `GPE`, a geo-political entity such as a country or a state; Universal NER counts those as places. They add that the term is often stretched to dates, times and prices; Universal NER leaves those out.

    ### A name is not the same as a proper noun

    A tagger of parts of speech can already tell that *Janet*, *Stanford University* and *Colorado* are proper nouns, `PROPN`. In the words of Jurafsky and Martin, "being a proper noun is a grammatical property of these words". A named entity is about what the words refer to: a person, an organization, a place. The two overlap a lot, but not completely. In the 11,702 sentences that Universal NER and the treebank of [Tagging word classes](/course/open/nlp-pos-tagging?lang=en) cut into the same tokens, 8,528 of the 9,634 tokens inside a name are `PROPN`, about 9 in 10. The others are words such as *United* in *United States* and *of* in *Bay of Plenty*. The other way round, 3,068 of the 11,596 `PROPN` tokens stand outside every name: months such as *April*, days such as *Friday*, titles such as *Dr.* and *President*, and *Americans*.

    ### Two hard parts

    Jurafsky and Martin name two things that make finding names hard. The first is where a name starts and ends: the task "is difficult partly because of the ambiguity of segmentation; we need to decide what's an entity and what isn't, and where the boundaries are". *Mississippi* is a name, and so is *Mississippi River*. The second is the type. The same name can be a person, a place or an organization, as *Washington* was in the demo of [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en). Their example: "The mention JFK can refer to a person, the airport in New York, or any number of schools, bridges, and streets around the United States."
    """)
    return


@app.cell(hide_code=True)
def text_idea_list():
    mo.md(r"""
    ### Lists: a gazetteer

    The simplest way to find a name is to look it up. Jurafsky and Martin describe a **gazetteer** as "a list of place names, often providing millions of entries for locations with detailed geographical and political information". Lists of other names exist too, such as lists of first names or of companies. This lesson makes its own list from the training set: every name that people marked there, as a tuple of its words, with the type it has most often. That gives 2,067 names, and the longest has 16 words. In the course, the list is a dict, `C.GAZETTEER`: `C.GAZETTEER[("New", "Orleans")]` is `"LOC"`. *Washington* is a place 10 times in the training set and an organization once, so the list says `LOC`, every time.

    Two names of the list can start at the same word: *United Nations* is in the list, and so is *United Nations Security Council*. The **longest match** settles it: at each position, take the longest name of the list that starts there, and go on after its end, so that names do not overlap.

    ### Rules

    Many systems in practice add rules that people write by hand, such as the rule that a word with a capital right after *Dr.* is a person. Jurafsky and Martin: "commercial approaches to NER are often based on pragmatic combinations of lists and rules, with some smaller amount of supervised machine learning". A list and its rules find only what someone put in them.
    """)
    return


@app.cell(hide_code=True)
def text_idea_tagger():
    mo.md(r"""
    ### A tagger that learns

    A tagger learns names from the sentences of the training set, with the BIO tags of [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en): `B-PER` on the first word of a person, `I-PER` on the words after it, `O` outside every name. It counts, as the first tagger of [Tagging word classes](/course/open/nlp-pos-tagging?lang=en) did, but with two clues.

    The first clue is the **tag before**: the tag that the tagger itself just gave the word before. After `B-PER`, a word is often the next word of the same name; after `O`, most words stand outside every name. Before the first word of a sentence, the tag before counts as `O`.

    The second clue helps with a word that the tagger never saw after that tag: its **word shape**. Jurafsky and Martin describe it as "the abstract letter pattern of the word": a capital becomes X, a small letter x, a digit d, and punctuation stays as it is. In the **short word shape**, "consecutive character types are removed": *Villanueva* becomes `Xx`, *U.S.* becomes `X.X.` and *DC10-30* becomes `Xd-d`.

    The tagger has two tables, made by counting in the training set. `C.BY_WORD` gives the most frequent tag of each pair `(word, tag before)`, and `C.BY_SHAPE` the most frequent tag of each pair `(short word shape, tag before)`. Each word, from left to right, gets the tag of its pair in `C.BY_WORD` when that table has the pair; otherwise the tag of its short word shape in `C.BY_SHAPE`; otherwise `O`. *Woodward* is not in the training set, but after `B-PER` the shape `Xx` gets `I-PER`, so *Bob Woodward* comes out as one person: a name that the list does not have.
    """)
    return


@app.cell(hide_code=True)
def shapes_cell():
    mo.Html(C.shapes_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_both():
    mo.md(r"""
    ### Both on one sentence

    The picture below shows the names of one sentence of the test set: the names people marked, the names of the gazetteer with the longest match, and the names of the tagger.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.vstack([
        mo.md(r"""
        The list has *Mississippi* as a place, but not *Mississippi River*, so it finds only part of that name. The tagger knows *River* after `B-LOC` as `I-LOC`, and carries the name on.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **named entity** | something that can be referred to with a proper name: a person, a place or an organization |
        | **named entity recognition (NER)** | finding the spans of a text that are names, each with its type |
        | **gazetteer** | a list of names of one kind, such as places; here a dict of names of three types, each with its type |
        | **longest match** | at each position, the longest name of the list that starts there |
        | **tag before** | the tag that the tagger gave the word before; before the first word, it is `O` |
        | **word shape**, **short word shape** | the pattern of a word: `X` for a capital, `x` for a small letter, `d` for a digit, and punctuation as it is; in the short word shape, a run of the same kind becomes one |
        | **training set**, **test set** | the part of the sentences that a tagger learns from, and the part it is measured on, which it never saw |
        | `C.TRAIN`, `C.TEST` | the training set and the test set of the English sentences of Universal NER: tuples of sentences, each a tuple of pairs `(word, tag)` |
        | `C.EXAMPLE` | the sentence of the worked example, *Alteration of the Mississippi River ...*, as pairs `(word, tag)` |
        | `C.GAZETTEER` | the course's gazetteer: a dict from each name of the training set, a tuple of words, to its most frequent type; you can read it, not change it |
        | `C.BY_WORD`, `C.BY_SHAPE` | the tables of the tagger: dicts from each pair `(word, tag before)` and `(short word shape, tag before)` to its most frequent tag; you can read them, not change them |
        | `C.spans(tags)` | the spans of a list of BIO tags, as your `to_spans` of *A label per word* gave them |
        | `(word,)` | a tuple with one item: the comma makes it a tuple |
        | `tuple(a_list)` | a tuple with the items of the list, in the same order; a list cannot be a key of a dict, a tuple can |
        | `range(a, b, -1)` | the whole numbers from `a` down to `b`, without `b`: `range(5, 2, -1)` gives 5, 4 and 3 |
        | `ch.isupper()`, `ch.islower()`, `ch.isdigit()` | `True` when the character `ch` is a capital, a small letter, or a digit |
        """),
        L.watch_out("A gazetteer gives a name the same type wherever it stands: *Washington* is a place in the list, also in *the Washington Post*. And it marks a word that is a name only now and then. The training set has *us* 4 times as a place, written small for *US*, and 101 times outside every name; the list marks every *us*."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. The gazetteer, with the longest match: go over the positions from left to right. At a position that is not inside a name found already, try the stretch of words from there to the end of the sentence, then one word shorter, and so on down to one word. The first stretch that the list has is the name; go on after its end.
    2. The tagger: go over the words from left to right, with `O` as the tag before the first word. Look up `(word, tag before)` in `C.BY_WORD`; when it is not there, `(short word shape, tag before)` in `C.BY_SHAPE`; when that is not there either, the tag is `O`. The tag you give is the tag before the next word.
    3. Read the names from the tags with `C.spans`, as in [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en).
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell looks up one word at a time: `one_word_at` gives the span of a name of one word at position `i`, or `None`, and `find_names` goes over the positions from left to right. It prints the names it finds in `C.EXAMPLE`, and the names people marked there.
    """)
    return


@app.cell
def example_find():
    def one_word_at(words, i, gazetteer):
        name = (words[i],)
        if name in gazetteer:
            return (i, i + 1, gazetteer[name])
        return None


    def find_names(words, gazetteer):
        found = []
        free = 0   # the first position that is not inside a name found already
        for i in range(len(words)):
            if i >= free:
                span = one_word_at(words, i, gazetteer)
                if span is not None:
                    found.append(span)
                    free = span[1]
        return found


    words = [word for word, tag in C.EXAMPLE]
    tags = [tag for word, tag in C.EXAMPLE]
    print(find_names(words, C.GAZETTEER))
    print(C.spans(tags))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    One word at a time finds *Mississippi*, part of the name *Mississippi River*, and misses *New Orleans*: the list has *New Orleans*, but neither *New* nor *Orleans* by itself. On the whole test set, one word at a time gets 257 of the 802 names right. Exercise 1 looks for the longest name instead.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_longest():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. longest_at: the longest name that starts here

        Write `longest_at(words, i, gazetteer)`, with the same arguments as `one_word_at`: give back the span `(i, end, kind)` of the longest name of `gazetteer` that starts at position `i`, or `None` when no name starts there. Try the ends from the end of the sentence down to `i + 1`: the first stretch `words[i:end]` that `gazetteer` has, as a tuple, is the longest. Replace the `...`. The check calls `longest_at(["the", "United", "Nations", "Security", "Council", "met"], 1, gazetteer)` with a small gazetteer that has *United Nations*, *United Nations Security Council* and *Council*, which gives `(1, 5, 'ORG')`; then cases of its own: a position where no name starts, a name of one word, a name up to the last word, a position inside a name, a name longer than the rest of the sentence, the longer of two names that both run to the last word, and the `words` of the worked example with `C.GAZETTEER`.
        """),
        L.task("Find the longest name of the gazetteer that starts at position i."),
    ])
    return


@app.cell(hide_code=True)
def exercise_longest():
    box_longest = L.code_box("longest_at", lines=7)
    box_longest
    return (box_longest,)


@app.cell(hide_code=True)
def check_longest(box_longest):
    result_longest = L.check("longest_at", box_longest.value)
    result_longest
    return (result_longest,)


@app.cell(hide_code=True)
def hints_longest():
    L.hints(
        "longest_at",
        [
            "One loop over `end`, with `range`, from `len(words)` down with a step of `-1`. Inside it, `name = tuple(words[i:end])` and an `if` that asks whether `name` is in `gazetteer`. A `return` inside the loop ends the function at the first name it finds, which is the longest.",
            "`range` stops right before its second number. Which `end` is the last one to try, the shortest stretch of one word, and which number comes right after it when you count down?",
            "`    for end in range(len(words), ___, -1):`",
        ],
        why="`for end in range(len(words), i, -1):` with `name = tuple(words[i:end])` under it, then `if name in gazetteer:` with `return (i, end, gazetteer[name])` under that; after the loop `return None`. `longest_at([\"the\", \"United\", \"Nations\", \"Security\", \"Council\", \"met\"], 1, gazetteer)` gives `(1, 5, 'ORG')`.",
    )
    return


@app.cell(hide_code=True)
def text_after_longest():
    mo.md(r"""
    The picture under the check puts your `longest_at` in the place of `one_word_at` in `find_names`, and runs it on the whole test set. The longest match gets 317 of the 802 names right, 60 more than one word at a time. It never finds a name that the list does not have, and some of its finds are wrong: part of a longer name, such as *Bob* in *Bob Woodward*, or no name at all, such as *us*. Exercise 2 makes the word shape that the tagger needs.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. shape: the short word shape

        Write `shape(word)`: give back the short word shape of `word`. Go over its characters. A capital gets the kind `X`, a small letter `x`, a digit `d`, and any other character is its own kind. Add a kind to the shape only when it differs from the kind before it, so that a run of the same kind becomes one, and three dots become one dot. Replace the `...`. The check calls `shape("Villanueva")`, which gives `'Xx'`; then cases of its own: *L'Occitane*, *I.M.F.*, *DC10-30*, *iPhone*, *us*, *US*, *2026*, an empty word, three dots, *café* and *McDonald's*.
        """),
        L.task("Turn a word into its short word shape."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shape():
    box_shape = L.code_box("shape", lines=14)
    box_shape
    return (box_shape,)


@app.cell(hide_code=True)
def check_shape(box_shape):
    result_shape = L.check("shape", box_shape.value)
    result_shape
    return (result_shape,)


@app.cell(hide_code=True)
def hints_shape():
    L.hints(
        "shape",
        [
            "One loop `for ch in word:`, with `if`, `elif` and `else` that give `kind` its value: `ch.isupper()`, `ch.islower()`, `ch.isdigit()`, and else the character itself. Keep the shape in a string `out` that starts as `\"\"`, and keep a second string for the kind you added before, also `\"\"` at the start.",
            "When do you add `kind` to `out`, and what goes into your second string right after?",
            "`        if kind != ___:`",
        ],
        why="`out = \"\"` and `last = \"\"`, then `for ch in word:` with `if ch.isupper():` and `kind = \"X\"`, `elif ch.islower():` and `kind = \"x\"`, `elif ch.isdigit():` and `kind = \"d\"`, `else:` and `kind = ch`; then `if kind != last:` with `out = out + kind` and `last = kind` under it; after the loop `return out`. `shape(\"Villanueva\")` gives `'Xx'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_shape():
    mo.md(r"""
    The 16,600 different words of the training set have only 267 different short word shapes, so a word that the tagger never saw still has a shape it has seen many times. Exercise 3 builds the tagger.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tag():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. tag_sentence: tagging with the tag before

        Write `tag_sentence(words, by_word, by_shape)`: give back a list with one BIO tag for each word of `words`. `by_word` is a table with a tag for each pair `(word, tag before)`, and `by_shape` a table with a tag for each pair `(short word shape, tag before)`, such as `C.BY_WORD` and `C.BY_SHAPE`. Go over the words from left to right, with `"O"` as the tag before the first word. Give each word the tag of `(word, before)` in `by_word` when that table has it; otherwise the tag of `(shape(word), before)` in `by_shape`; otherwise `"O"`. The tag you give is the tag before the next word. In this box, `shape` is the course's version of Exercise 2. Replace the `...`. The check calls `tag_sentence(["Jane", "Villanueva", "said", "so"], by_word, by_shape)` with two small tables, which gives `['B-PER', 'I-PER', 'O', 'O']`; then cases of its own: no words, a word that neither table has, a first word that only `by_shape` can tag, a known word whose tag depends on the tag before, a name of three words, and three sentences of the test set with `C.BY_WORD` and `C.BY_SHAPE`.
        """),
        L.task("Tag the words from left to right, each with the tag before it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tag():
    box_tag = L.code_box("tag_sentence", lines=12)
    box_tag
    return (box_tag,)


@app.cell(hide_code=True)
def check_tag(box_tag):
    result_tag = L.check("tag_sentence", box_tag.value)
    result_tag
    return (result_tag,)


@app.cell(hide_code=True)
def hints_tag():
    L.hints(
        "tag_sentence",
        [
            "Begin with an empty list `tags`, `before = \"O\"` and a loop `for word in words:`. Inside it, `if`, `elif` and `else` choose `tag`: first `(word, before) in by_word`, then `(shape(word), before) in by_shape`. After them comes `tags.append(tag)`.",
            "When the loop goes on to the next word, which tag stands before that word?",
            "`        before = ___`",
        ],
        why="`tags = []` and `before = \"O\"`, then `for word in words:` with `if (word, before) in by_word:` and `tag = by_word[(word, before)]`, `elif (shape(word), before) in by_shape:` and `tag = by_shape[(shape(word), before)]`, `else:` and `tag = \"O\"`; then `tags.append(tag)` and `before = tag`; after the loop `return tags`. `tag_sentence([\"Jane\", \"Villanueva\", \"said\", \"so\"], by_word, by_shape)` gives `['B-PER', 'I-PER', 'O', 'O']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tag():
    mo.md(r"""
    On the whole test set, the tagger gets 343 of the 802 names right, 26 more than the gazetteer, with about as many wrong finds: 100, against 97. Of its right names, 58 are not in the list at all, such as *Bob Woodward* and *Bay of Plenty*: the tag before carried a name on into words that the tagger never saw there. Both still miss more than half of the names. A word the tagger does not know, after `O`, almost always gets `O`: `Xx` after `O` is not a name most of the time.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: your own sentence

    Type a sentence, with spaces between the words and before punctuation, and see the names that the gazetteer and the tagger find. Try *Tell us about it .*, *I read the Washington Post .*, or a name that the training set does not have, such as *Dr. Okonkwo visited Tilburg .*
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_text = mo.ui.text(value=C.DEMO_TEXT, label="A sentence", full_width=True)
    demo_text
    return (demo_text,)


@app.cell(hide_code=True)
def demo_look(demo_text):
    mo.Html(C.demo_picture(demo_text.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_streets():
    mo.md(r"""
    ### Exercise 4. A list or a tagger?

    A town hall gets letters from its residents, and wants to find the names of its 30 streets in them. Nobody has marked the names in those letters.
    """)
    return


@app.cell(hide_code=True)
def choice_streets_cell():
    choice_streets = L.choice(
        "streets",
        question="Which way fits best?",
        options={
            "tagger": "The tagger of Exercise 3, with the tables of Universal NER",
            "right": "A gazetteer with the 30 street names",
            "capitals": "Every word with a capital letter that does not start a sentence",
            "wait": "None yet: first people have to mark the names in a thousand letters",
        },
        correct="right",
        why={
            "right": "The 30 names are known and do not change, so a list finds them, with their type, without a single marked example. Lists and rules are where many systems in practice start. A list does need care: it misses a street written differently, such as with a typo, and it marks a street named after a person also where the person is meant.",
            "tagger": "The tables come from reviews, blogs, newsgroups and answers to questions on the web, and most of the 30 streets are not in them. On the test set, the tagger found 58 names it never saw as a name, mostly by carrying on a name it knew. What can find a known name without any marked examples?",
            "capitals": "That also marks every person, every month, every *I* and the names of other towns. And a street written in small letters, as in many texts on the web, slips through. What knows exactly which 30 names to look for?",
            "wait": "Marked letters would help a tagger, but the 30 names are known already. What can find a known name without any examples?",
        },
    )
    choice_streets
    return (choice_streets,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A label per word: sequence labeling and BIO](/course/open/nlp-bio-tags?lang=en) wrote names as spans and as BIO tags, and [Tagging word classes](/course/open/nlp-pos-tagging?lang=en) built a tagger by counting. Here both came together in named entity recognition. A gazetteer made from the training set, read with the longest match, gets 317 of the 802 names of the test set right; it finds only names it has, and gives a name the same type everywhere. A tagger that counts tags after the tag before, with the word shape for words it never saw there, gets 343 right, 58 of them names that the list does not have. The next lesson, *Scoring a tagger: accuracy and span F1*, turns the names found right, the wrong finds and the missed names into one score.

    **An everyday example.** A new colleague learns the names at work in two ways: from the list of staff on the wall, and by listening. *Can you ask Okonkwo?* tells the newcomer that a person is meant, before that name is on any list.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_streets, result_longest, result_shape, result_tag):
    L.wrap_up(result_longest, result_shape, result_tag, choice_streets)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
