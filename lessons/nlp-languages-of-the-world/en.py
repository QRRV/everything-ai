# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-languages-of-the-world. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-languages-of-the-world", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en), and one about [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pronoun():
    quick_pronoun = L.choice(
        "quick-pronoun",
        question="Universal Dependencies gives every word one of 17 tags for its part of speech. Which tag does *it* get in *the dog found it*?",
        options={
            "noun": "NOUN, a noun",
            "right": "PRON, a pronoun",
            "det": "DET, a determiner",
        },
        correct="right",
        why={
            "right": "*It* stands in for a noun phrase, such as *the bone*: a pronoun. This lesson counts clauses with a subject and an object, and the atlas of languages it uses counts only the clauses in which both are nouns, not pronouns.",
            "noun": "*It* takes the place of a noun phrase, such as *the bone*, without being a noun itself. Which tag do *I*, *it* and *you* get? See *Seventeen tags for every language* in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
            "det": "A determiner, such as *the* or *this*, stands before a noun, and *it* stands alone. Which tag do *I*, *it* and *you* get? See *Seventeen tags for every language* in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
        },
    )
    quick_pronoun
    return (quick_pronoun,)


@app.cell(hide_code=True)
def quick_check_object():
    quick_object = L.choice(
        "quick-object",
        question="In *the cat saw the bird*, Universal Dependencies hangs *bird* from *saw*. With which relation?",
        options={
            "nsubj": "nsubj, the subject",
            "right": "obj, the object",
            "amod": "amod, an adjective that describes a noun",
        },
        correct="right",
        why={
            "right": "*The bird* is what the seeing is done to: the object of *saw*, `obj(saw, bird)`. This lesson reads the subject and the object of a verb off these labels, and then looks at the order they come in.",
            "nsubj": "The subject is who or what does something: here *the cat*. What is *the bird* to *saw*? See the table in *Universal Dependencies* in [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
            "amod": "*Bird* is a noun, not an adjective, and it hangs from a verb. What is *the bird* to *saw*? See the table in *Universal Dependencies* in [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en).",
        },
    )
    quick_object
    return (quick_object,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    "There are about 7,000 languages in the world," Jurafsky and Martin write in the chapter *Machine Translation* of *Speech and Language Processing*. They differ in their sounds, their words and the way they build a sentence, and the tools of NLP cover only a few of them. In 2020, Pratik Joshi and four others sorted 2,485 languages into six classes by the resources that exist for them, from *The Left-Behinds* to *The Winners*. Only 7 languages made the top class, among them English, Spanish, German, Japanese and French; 2,191 are in the bottom one. The web is just as uneven: in the crawl CC-MAIN-2026-43 of Common Crawl, a large collection of web pages, 42.6% of the pages are in English and 1.8% in Dutch, by the language detector it uses. In 2019, Emily M. Bender asked everyone who builds or tests a tool for language: "Always name the language(s) you're working on." This lesson looks at what languages share and how they differ: in families, and in the order of subject, verb and object.

    **An everyday example.** Count the languages that the keyboard of your phone or a translation app offers. However long the list looks, set it next to the 7,000.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="An atlas of languages records, for 1,376 languages, the order in which they usually put the subject (S), the object (O) and the verb (V) of a clause. Which order do the most languages have?",
        options={
            "svo": "SVO, like English and Mandarin",
            "right": "SOV, like Japanese",
            "vso": "VSO, like Welsh",
        },
        reveal="**SOV, like Japanese**: 564 of the 1,376 languages, against 488 with SVO, the order of English. Together the two account for more than three in four languages. Verb first, VSO or VOS, comes in 120 languages, and the object first, OVS or OSV, in only 15. 189 languages have no dominant order, among them Dutch and German.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### How many languages?

    The number depends on what you count. *Ethnologue*, a catalog of SIL International, counted 7,170 living languages in its edition of 2026, as Wikipedia reports. *Glottolog*, a catalog of the Max Planck Institute for Evolutionary Anthropology in Leipzig, counts 7,685 spoken first languages in its version 5.4 of November 10, 2026, and that number includes the languages that are no longer spoken: 972, by Glottolog's own measure of how endangered a language is, in the course's count. And where one language ends and the next begins is not always clear. Glottolog counts two ways of speaking as separate languages when they are not mutually intelligible, that is, when the speakers of one do not understand the other, and as dialects of one language when they are; for its dialects, Glottolog writes, it has spent little effort on making the classification consistent. A later lesson is about language and dialect.

    ### Families

    Languages that grew out of one older language form a **language family**. Glottolog puts languages in one family when the best explanation for what they share is that they inherited it from a common ancestor. Dutch and English are both West Germanic, a branch of Germanic, which is a branch of Indo-European: a family of 587 languages in Glottolog, extinct ones included, with Hindi, Persian, Russian and Welsh among them. In Glottolog's tree, English is even closer to Frisian than to Dutch: English and Frisian share the branch Anglo-Frisian, and Afrikaans sits inside the branch of Dutch. A language with no known relatives is an **isolate**, a family of one, such as Basque. Counted by the course in the data of Glottolog 5.4, its 7,685 languages fall into 237 families and 183 isolates (its own page says 245 families, apparently counting 8 groups that are not families by descent, such as sign languages and pidgins); the largest families are Atlantic-Congo, with 1,409 languages, and Austronesian, with 1,280.

    ### Types: how languages are built

    Languages can also be compared without asking where they come from. Jurafsky and Martin: "The study of these systematic cross-linguistic similarities and differences is called linguistic typology." **Linguistic typology** sorts languages into types by how they are built, whether they are related or not. One of the best-known features is the order of the subject (S), the verb (V) and the object (O) in a clause such as *the cat saw the bird*: six orders are possible, SVO, SOV, VSO, VOS, OVS and OSV. In 1963, Joseph Greenberg compared 30 languages; the first of the universals in that work reads: "In declarative sentences with nominal subject and object, the dominant order is almost always one in which the subject precedes the object."

    The **World Atlas of Language Structures** (WALS), edited by Matthew S. Dryer and Martin Haspelmath, describes 192 such features for 2,662 languages, gathered by 55 authors from grammars and other descriptions. It came out as a book in 2005 and online in 2008; the course uses its data of version v2020.6, released under CC BY 4.0. For feature 81A, Dryer recorded the **dominant order** of 1,376 languages: the only order a language allows, or the one it uses most. An order of three parts counts as dominant if text counts show that it is more than twice as common as the next order; if no order is, the language has no dominant order. Dryer looks only at clauses whose subject and object are nouns, not pronouns, and uses subject and object in a loose sense: the one that acts and the one that undergoes. The picture shows the counts.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_dutch():
    mo.md(r"""
    SOV and SVO together are the dominant order of 1,052 of the 1,376 languages, 76.5%, and the orders with the object first are rare, as Greenberg wrote: 11 languages with OVS and 4 with OSV. English is SVO. Dutch, German and Frisian have no dominant order in WALS. For German and Dutch, Dryer gives the reason, the place of the verb, in the chapter of WALS on this feature: "in German and Dutch, the dominant order is SVO in main clauses lacking an auxiliary and SOV in subordinate clauses and clauses containing an auxiliary." Both orders are common, so neither counts as dominant; WALS records the pair, SOV or SVO, in a feature of its own.

    Features also go together. Of the languages in WALS with a value for the order of object and verb and for adpositions, the small words such as *in* and *of* that go with a noun phrase, 472 put the object before the verb and have **postpositions**, which come after their noun phrase, as in Japanese, Turkish and Hindi; only 14 put the object before the verb and have prepositions. With the object after the verb, 454 have prepositions and 41 postpositions. Dryer calls it a tendency in both directions: "a language tends to be OV if and only if it is postpositional." Greenberg's fourth universal, of 1963, already linked SOV with postpositions.

    ### Few languages, few tools

    Back to Joshi and others. In their class 0, *The Left-Behinds*, are 2,191 of the 2,485 languages they sorted, such as Warlpiri and Dahalo; of those, they write: "These languages have been and are still ignored in the aspect of language technologies." Dutch is in class 4, *The Underdogs*, with Russian, Hungarian, Vietnamese and Korean. Large models reach further, but not far: NLLB-200, a translation model of 2022, supports 202 languages, and the Qwen3 models of 2025 were trained on 119 languages and dialects. Bender put the rule in a tweet in June 2019; in Bender's article on it, the section about English is called *English is Neither Synonymous with Nor Representative of Natural Language*.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **language family** | languages that grew out of one ancestor language, such as Indo-European |
        | **isolate** | a language with no known relatives: a family of one, such as Basque |
        | **linguistic typology** | comparing languages by how they are built, whether they are related or not |
        | **WALS** | the World Atlas of Language Structures: 192 features of 2,662 languages |
        | **dominant order** | the only order a language allows, or the one it uses most: by the rule of WALS, more than twice as common as the next order |
        | S, V, O | subject, verb and object; SOV is the order subject, object, verb |
        | **postposition** | a word like a preposition that comes after its noun phrase instead of before it |
        | `letters + "V"` | a new string: the string `letters` with `"V"` after it |
        | `for letters in counts:` | goes through the keys of the dict `counts`, one by one |
        | `d.values()` | the values of the dict `d`, without their keys |
        | `sorted(x, reverse=True)` | a new list with the items of `x`, from the highest to the lowest |
        """),
        L.watch_out("WALS gives each language one value for a feature, from grammars and text counts, and a language can be more varied than one value. Dryer also writes that further text counts would likely put some languages under another value. And a family and a type are different things: related languages can differ in type, as Dutch and English do, and languages of different families can share one."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Find the verb of a clause, and its subject and object: the words that hang from it with the relations `nsubj` and `obj`.
    2. Go through the words from left to right, and write S, V or O for each of the three: that gives the order, such as SVO.
    3. Do that for every clause of a treebank, and count how often each order comes back.
    4. Compare the most common order with the next one: if it is more than twice as common, it is the dominant order, by the rule of WALS.
    5. For many languages at once, look up their values in WALS, and compare families and parts of the world.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds two Dutch sentences of the course, with the relations of Universal Dependencies, as lists of (form, head, relation): `SENTENCE`, *gisteren zag de kat de vogel* (yesterday the cat saw the bird), and `CLAUSE`, *omdat de kat de vogel ziet* (because the cat sees the bird). The second cell goes through the words of a sentence, numbered from 1, and prints every subject, the word with the relation `nsubj`: its number, and the number of the word it hangs from.
    """)
    return


@app.cell
def sentences_cell():
    SENTENCE = [
        ("gisteren", 2, "advmod"),   # yesterday
        ("zag", 0, "root"),          # saw
        ("de", 4, "det"),            # the
        ("kat", 2, "nsubj"),         # cat
        ("de", 6, "det"),            # the
        ("vogel", 2, "obj"),         # bird
    ]
    CLAUSE = [
        ("omdat", 6, "mark"),        # because
        ("de", 3, "det"),            # the
        ("kat", 6, "nsubj"),         # cat
        ("de", 5, "det"),            # the
        ("vogel", 6, "obj"),         # bird
        ("ziet", 0, "root"),         # sees
    ]
    return CLAUSE, SENTENCE


@app.cell
def example_subjects(CLAUSE, SENTENCE):
    def subjects(sentence):
        for i in range(1, len(sentence) + 1):
            form, head, relation = sentence[i - 1]
            if relation == "nsubj":
                print(form, "is word", i, "and hangs from word", head)

    subjects(SENTENCE)
    subjects(CLAUSE)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In `SENTENCE`, the subject *kat* is word 4, and it hangs from word 2, the verb *zag*: the subject comes after the verb, since *gisteren* took the first place, and a Dutch main clause keeps its verb second. The Taalportaal, a grammar of Dutch online, says that finite verbs, the forms that show the tense, "are typically placed in the so-called second position of main clauses." In `CLAUSE`, the subject is word 3, and it hangs from word 6, the verb *ziet*, at the end, as usual after *omdat*. One Dutch verb, two places: the exercises count how often each order comes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. order: subject, verb and object in a row

        Write `order(sentence, h)`: go through the words of `sentence` from left to right, and build a string of letters: `"V"` for word `h`, the verb, `"S"` for its subject (relation `nsubj`) and `"O"` for its object (relation `obj`). Give back the string. A verb without an object gives two letters, such as `"SV"`. Number the words from 1, as the heads are. For `order(SENTENCE, 2)`, the result is `"VSO"`. Replace the `...`. The check calls `order(SENTENCE, 2)`, then cases of its own: `CLAUSE`, other short sentences, a verb without an object, a relative clause, a verb that is the first word, and a sentence with two verbs, each with a subject of its own.
        """),
        L.task("Read the order of the subject, the verb and the object off a dependency tree."),
    ])
    return


@app.cell(hide_code=True)
def exercise_order():
    box_order = L.code_box("order", lines=11)
    box_order
    return (box_order,)


@app.cell(hide_code=True)
def check_order(box_order):
    result_order = L.check("order", box_order.value)
    result_order
    return (result_order,)


@app.cell(hide_code=True)
def hints_order():
    L.hints(
        "order",
        [
            "Start with an empty string `letters`. Go through the numbers `i` from 1 to the number of words, as in the worked example, and take word `i` from place `i - 1`. At each word, add its letter if it has one: V for word `h` itself, S for the subject of `h`, O for its object. After the loop, give back `letters`.",
            "Word `i` has the relation `nsubj`. How do you know it is the subject of verb `h`, and not of another verb of the sentence?",
            '`        if ___ and relation == "nsubj":`',
        ],
        why='`letters = ""`, then `for i in range(1, len(sentence) + 1):` with `form, head, relation = sentence[i - 1]` under it, and three `if`s: `if i == h:` with `letters = letters + "V"`, `if head == h and relation == "nsubj":` with `letters = letters + "S"`, and `if head == h and relation == "obj":` with `letters = letters + "O"`. After the loop: `return letters`.',
    )
    return


@app.cell(hide_code=True)
def text_after_order():
    mo.md(r"""
    The picture under the check runs your function on seven clauses of the treebanks. The five Dutch ones are one sentence, *ik mis de trein* (I miss the train), in five orders, from the Dutch treebank: the subject first and the verb second; *vandaag* (today) first, so the subject comes after the verb; with the auxiliary *heb* (have), the main verb *gemist* (missed) at the end; and the object *de trein* in front, with and without *heb*. Universal Dependencies hangs the subject and the object from the main verb, not from the auxiliary, so *ik heb de trein gemist* counts as SOV: the order Dryer gives for Dutch clauses with an auxiliary.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. count_orders: every clause of a treebank

        `ENGLISH` below holds the 11,723 sentences of the English treebank of [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en), and `DUTCH` the 13,603 of the Dutch one, each sentence a list of words as above. Write `count_orders(sentences)`: for every word `h` of every sentence, ask `order(sentence, h)` for its letters, and count how often each order of all three letters comes back, in a dict from the order to its count. Words with fewer letters, without a subject or an object, stay out. `order` is ready for your function: the check gives it the course's own. For `count_orders(DUTCH)`, SOV comes 2,319 times. Replace the `...`. The check calls `count_orders(DUTCH)`, then cases of its own: short lists of the course's sentences, and an empty list.
        """),
        L.task("Count how often each order of subject, verb and object comes in a treebank."),
    ])
    return


@app.cell
def treebanks_cell():
    ENGLISH = C.treebank("en")
    DUTCH = C.treebank("nl")
    print(len(ENGLISH), len(DUTCH))
    return (DUTCH,)


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("count-orders", lines=11)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("count-orders", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "count-orders",
        [
            "Start with an empty dict `counts`. Go through the sentences, and inside that loop through the numbers `h` of the words of the sentence. Ask `order(sentence, h)` for the letters; when there are three, add 1 to their count, or give them the count 1 when they are not in the dict yet. After both loops, give back the dict.",
            "The words of a sentence are numbered from 1. Where does `range` have to start, and where does it have to stop to reach the last word?",
            "`        for h in range(___):`",
        ],
        why="`counts = {}`, then `for sentence in sentences:` and inside it `for h in range(1, len(sentence) + 1):`, with `letters = order(sentence, h)` and `if len(letters) == 3:` under it; inside that, `if letters in counts:` with `counts[letters] = counts[letters] + 1`, and `else:` with `counts[letters] = 1`. After both loops: `return counts`.",
    )
    return


@app.cell(hide_code=True)
def text_after_counts():
    mo.md(r"""
    In English, 5,001 of the 5,273 clauses with a subject, a verb and an object are SVO, and nearly all the others OSV: their object is a question word or a word such as *that* or *which* at the start of a relative clause, as in *What do I do?* In Dutch, no order comes close to that: SOV 2,319 times, SVO 1,564, VSO 716, OSV 359 and OVS 250. WALS counts only clauses whose subject and object are nouns; with only those, the counts change little: 1,208 of 1,215 English clauses are SVO, and in Dutch SOV leads with 820, ahead of SVO with 762.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dominant():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. dominant: the rule of WALS

        `COUNTS` below holds the counts of the Dutch treebank, as `count_orders(DUTCH)` gives them. Write `dominant(counts)`: give back the most common order if it is more than twice as common as the next one, by the rule of WALS, and else the empty string `""`, for no dominant order. Sort the counts with `sorted(..., reverse=True)` to find the two highest, then look up which order has the highest. A dict with only one order makes that order dominant. For `dominant(COUNTS)`, the result is `""`. Replace the `...`. The check calls `dominant(COUNTS)`, then cases of its own: the counts of English, one order alone, an order exactly twice as common as the next and one a little more, the most common order not first in the dict, and two orders that are equally common.
        """),
        L.task("Decide with the rule of WALS whether one order is dominant."),
    ])
    return


@app.cell
def counts_cell(DUTCH):
    COUNTS = C.count_orders(DUTCH)
    print(COUNTS)
    return


@app.cell(hide_code=True)
def exercise_dominant():
    box_dominant = L.code_box("dominant", lines=7)
    box_dominant
    return (box_dominant,)


@app.cell(hide_code=True)
def check_dominant(box_dominant):
    result_dominant = L.check("dominant", box_dominant.value)
    result_dominant
    return (result_dominant,)


@app.cell(hide_code=True)
def hints_dominant():
    L.hints(
        "dominant",
        [
            "Put the counts of the dict in a list from the highest to the lowest, `ranked`. Use the rule to decide whether to give back the empty string. If not, go through the orders of the dict, and give back the one whose count is the first of `ranked`.",
            "Which two counts of `ranked` does the rule compare, and which outcome means that there is no dominant order?",
            "`    if len(ranked) > 1 and ___:`",
        ],
        why='`ranked = sorted(counts.values(), reverse=True)`, then `if len(ranked) > 1 and ranked[0] <= 2 * ranked[1]:` with `return ""` under it. After that, `for letters in counts:` with `if counts[letters] == ranked[0]:` and `return letters` under it.',
    )
    return


@app.cell(hide_code=True)
def text_after_dominant():
    mo.md(r"""
    The rule gives English SVO in every row of the picture, and Dutch, over all its clauses, no dominant order, as WALS does: SOV is only 1.5 times as common as SVO. Split the Dutch clauses by kind, and the two orders of Dryer's description come back. With an auxiliary, SOV wins 984 of 1,090 clauses, and after a subordinating conjunction such as *omdat*, 657 of 732. In the remaining clauses, which have neither and are not relative clauses, SVO is 2.2 times as common as VSO, and so dominant, just; there, 2,458 of the 2,632 clauses are SVO, VSO or OVS: the orders a verb in second place gives, after the subject, after another word or after the object.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: the map of WALS

    Choose a feature and a family. The map draws every language of WALS with a value for that feature, and the bars and the table count the values, with examples. Compare Niger-Congo with Sino-Tibetan, look where the languages with the verb first are spoken, and switch between 83A and 85A: the languages with the object before the verb and the languages with postpositions lie in much the same places.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_feature = mo.ui.dropdown(options=C.feature_names(L.lang), value=list(C.feature_names(L.lang))[0], label="Feature")
    demo_family = mo.ui.dropdown(options=C.family_names(L.lang), value=list(C.family_names(L.lang))[0], label="Family")
    mo.hstack([demo_feature, demo_family], justify="start", gap=1.5)
    return demo_family, demo_feature


@app.cell(hide_code=True)
def demo_look(demo_family, demo_feature):
    mo.Html(C.demo_picture(demo_feature.value, demo_family.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Families differ a lot. In WALS, 173 of the 187 Niger-Congo languages with a value for 81A are SVO, and all 68 of Trans-New Guinea are SOV, while Austronesian has every order. Indo-European is split: 30 SOV, such as Hindi and Persian, and 27 SVO, such as English and Russian. WALS uses a list of families of its own, which differs from Glottolog's in places: it has an Altaic family, with Turkish in it, which Glottolog does not have.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_family():
    mo.md(r"""
    ### Exercise 4. Family or type?

    Each bold statement is about a family or about a type. Put it under the one it is about.
    """)
    return


@app.cell(hide_code=True)
def exercise_family():
    sort_family = L.sort("family-or-type")
    sort_family
    return (sort_family,)


@app.cell(hide_code=True)
def check_family(sort_family):
    result_family = L.check("family-or-type", sort_family)
    result_family
    return (result_family,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en) gave every word a head and a relation, and [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en) a tag; this lesson used them to compare languages. About 7,000 languages are spoken, in families that grew out of common ancestors, and linguistic typology compares them by how they are built, whatever their family. You read the order of subject, verb and object off a tree, counted the orders of two treebanks, applied the rule of WALS, and told families and types apart. English is SVO; Dutch keeps its verb second in a main clause, and puts it at the end after a subordinating conjunction such as *omdat*, or with an auxiliary. Most tools of NLP work for few of the 7,000 languages, so say which language yours is for, as Bender asks. The next lesson of the domain is about language in society: *Language, groups and dialects*.

    **An everyday example.** The next time an app or a model says that it speaks your language, ask how well, and how many of the 7,000 languages it speaks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_counts, result_dominant, result_family, result_order):
    L.wrap_up(result_order, result_counts, result_dominant, result_family)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
