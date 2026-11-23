# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-dutch-tools. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-tools", "en")
    C = L.checks   # the course's sentences, the treebanks, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Corpora and treebanks](/course/open/nlp-corpora?lang=en), and one about [Stems and lemmas](/course/open/nlp-stemming?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_particles():
    quick_particles = L.choice(
        "quick-particles",
        question="In [Corpora and treebanks](/course/open/nlp-corpora?lang=en), `particles(sentence)` gave a pair `(head, i)` for every word with the relation `compound:prt`: for *We picked the kids up and dropped them off*, `[(2, 5), (7, 9)]`. What does it give for *Pick it up*, with the words *Pick*, *it* and *up*?",
        options={
            "turned": "`[(3, 1)]`",
            "words": "`[('Pick', 'up')]`",
            "right": "`[(1, 3)]`",
            "object": "`[(1, 2)]`",
        },
        correct="right",
        why={
            "right": "*up* is word 3, and its head, the verb *Pick*, is word 1: the pair `(1, 3)`, with the verb first, as in `(2, 5)` for *picked* and *up*. In English, such a particle stays close to its verb. This lesson looks at where it stands in Dutch.",
            "turned": "The pair is `(head, i)`: first the number of the verb, then that of the particle. *Pick* is word 1 and *up* word 3. See Exercise 2 in [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
            "words": "`particles` gives word numbers, not the words themselves: *Pick* is word 1 and *up* word 3. See Exercise 2 in [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
            "object": "Word 2 is *it*, the object of *Pick*, with the relation `obj`. The particle is *up*, word 3. See Exercise 2 in [Corpora and treebanks](/course/open/nlp-corpora?lang=en).",
        },
    )
    quick_particles
    return (quick_particles,)


@app.cell(hide_code=True)
def quick_check_lemma():
    quick_lemma = L.choice(
        "quick-lemma",
        question="In [Stems and lemmas](/course/open/nlp-stemming?lang=en), the Dutch treebank gave *liep* (walked) the lemma *lopen* 17 times and *op_lopen* 7 times. Which lemma does it give *liep* in *Het aantal liep snel op* (the number rose fast)?",
        options={
            "plain": "`lopen`",
            "right": "`op_lopen`",
            "participle": "`opgelopen`",
            "form": "`liep`",
        },
        correct="right",
        why={
            "right": "Here *liep* and *op* belong together: *oplopen*, to rise. The treebank's lemmas join the separate part of a separable verb, such as *op*, to the lemma, with `_`. In this lesson you read such lemmas in the whole Dutch treebank.",
            "plain": "*lopen* is the lemma where *liep* means walked, without *op*. Here *op* belongs with *liep*, and the treebank joins it to the lemma. See [Stems and lemmas](/course/open/nlp-stemming?lang=en).",
            "participle": "*opgelopen* is a form of the verb, the participle. A lemma is the form a dictionary has, and the treebank joins the separate *op* to it. See [Stems and lemmas](/course/open/nlp-stemming?lang=en).",
            "form": "A lemma is the form a dictionary has, not the word as it stands in the sentence. See [Stems and lemmas](/course/open/nlp-stemming?lang=en).",
        },
    )
    quick_lemma
    return (quick_lemma,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of NLP was made for English first. Porter wrote a stemmer for English, and the guidelines of the Penn Treebank in *Corpora and treebanks* were written for English sentences. Dutch does things that rules written for English do not expect. A verb such as *opbellen* (to phone) falls apart in a main clause: *Ik bel je morgen op* (I'll call you tomorrow). Compounds are written as one word, such as *grondwet* (constitution), and in a subordinate clause the verb comes last. So researchers in the Netherlands and Flanders have built tools and corpora for Dutch: the first parts of the toolkit Frog date from the 1990s, the Corpus Gesproken Nederlands was collected from 1998 to 2004, and the parser Alpino still gets new versions in 2026. This lesson shows the main ones, and measures in a Dutch treebank what a rule written for English misses.

    **An everyday example.** A spell checker set to English, run on a letter in Dutch, underlines almost every word. The checker is not broken; it is the wrong tool for this text.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In the 11,723 sentences of UD_English-EWT that the course uses, the English treebank of *Corpora and treebanks*, 547 of the 724 particles, such as *up* in *pick up the phone*, stand right after their verb: 76 in 100. Dutch has such particles too: the *op* of *opbellen* (to phone) in *Ik bel je morgen op* (I'll call you tomorrow). In UD_Dutch-Alpino, the Dutch treebank of this lesson, with 13,603 sentences, how many in 100 stand right after their verb?",
        options={
            "some": "About 15",
            "same": "About 75, as in English",
            "right": "About 8",
            "few": "About 2",
        },
        reveal="**About 8**: 151 of the 1,952 particles in UD_Dutch-Alpino stand right after their verb, by the course's own count. 808 stand before it, as *terug* in *ik probeer je morgen terug te bellen* (I'll try to call you back tomorrow), and the rest further after it, in one sentence 32 places. A rule written for English, such as *the particle comes right after its verb*, misses more than nine in ten of them.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Separable verbs

    Dutch has many **separable verbs**: verbs such as *opbellen*, made of a verb, *bellen*, with a particle in front of it, *op*. In a main clause, the conjugated verb stands in second place, and the particle goes to the end of the clause: *Ik bel je morgen op*. When the two stand side by side, they are written as one word: *omdat ik je morgen opbel* (because I'll call you tomorrow), *opbellen*, *opgebeld*. In a subordinate clause they often do, but not always: *omdat ik je morgen op wil bellen* (because I want to call you tomorrow). The participle gets its *ge-* after the particle, and the infinitive with *te* its *te* between them: the reference grammar of Dutch, the *Algemene Nederlandse Spraakkunst* (ANS), gives *aangevallen*, not *geaanvallen*, and *aan te vallen*, not *te aanvallen* (attacked, to attack). English has particles too, *up* in *pick up*, but they stay close: right after the verb, or after a short object, as in *pick it up*.

    ### What the Dutch treebank writes

    *Corpora and treebanks* showed that Universal Dependencies gives a particle the relation `compound:prt` to its verb. The Dutch treebank of this lesson, UD_Dutch-Alpino, was annotated at the University of Groningen with the tools and guidelines of the parser Alpino, and its lemmas do more. The particle of a separable verb is joined to the lemma of the verb with an underscore, nearly always: *bel* in *Ik bel je morgen op* has the lemma `op_bellen`. The verb keeps that lemma when it is written as one word: *opgelost* (solved) has `op_lossen`, without a particle of its own in the sentence. The lemma even tells apart two verbs that look the same. *voorkomen*, to prevent, is not separable, *dat voorkwam een ongeluk* (that prevented an accident), and has the lemma `voorkomen`; *voorkomen*, to occur, is separable, *dat komt vaak voor* (that happens often), and has `voor_komen`. The parts of a compound noun are marked the same way, as *jeugd_gezondheid_zorg* showed in *Stems and lemmas*: *grondwet* (constitution) has `grond_wet`.

    UD also gives `compound:prt` in Dutch to a word that forms a fixed combination with a verb: a noun, as *plaats* in *de les vindt morgen plaats* (the lesson takes place tomorrow), lemma `plaats_vinden`; an adjective, as *dicht* in *doe de deur dicht* (close the door); the preposition of a fixed phrase, as *in* in *in gebruik nemen* (to start using); and even a verb, as *zien* in *laten zien* (to show). In the last two, the verb keeps its own lemma, `nemen` and `laten`, without an underscore. This lesson calls all of them particles, as UD does. How far a particle stands from its verb is its **gap** here: the number of places from the verb to the particle, 1 when it comes right after the verb, and negative when it comes before. In *Ik bel je morgen op*, the gap is 3.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_tools():
    mo.md(r"""
    ### Tools for Dutch

    - **Alpino** is a parser for Dutch from the University of Groningen, by Gertjan van Noord, Gosse Bouma, Robert Malouf and others, begun in a research project of 2000 to 2005. Unlike most parsers today, it starts from a grammar written by hand: in 2006, van Noord counted about 600 rules and a lexicon of about 100,000 words. Where the grammar allows several analyses of a sentence, a statistical model trained on a treebank chooses one. Alpino gives each sentence a dependency structure; in its own output, *speelt ... op* (flares up) gets the base form `speel_op`, which Alpino calls its stem. It is free software, under the LGPL.
    - **Frog** is a toolkit for Dutch, made at Tilburg University and the University of Antwerp, developed further at Radboud University Nijmegen and now maintained at the KNAW Humanities Cluster. It splits a text into sentences and words, and adds tags, lemmas, the parts of each word, a dependency tree, and names of people, places and organizations. Its parts learn from Dutch corpora: the tagger mostly from the Corpus Gesproken Nederlands, the parser from Lassy Small and from text parsed by Alpino, and the part that finds names, from SoNaR-1. It is free software too, under the GPL.
    - spaCy, the library of [An NLP pipeline](/course/open/nlp-pipeline?lang=en), has three Dutch pipelines, `nl_core_news_sm`, `nl_core_news_md` and `nl_core_news_lg`, trained on two Dutch treebanks of UD; the small one gets 95.0% of the lemmas right on spaCy's own test. Stanza, from Stanford University, has Dutch models trained on the same two treebanks.

    ### Corpora of Dutch

    - **SoNaR** is a corpus of written Dutch from the Netherlands and Flanders, built from 2008 to 2011: SoNaR-500 holds more than 500 million words of many kinds of text, chosen from 1954 or later, with tags, lemmas and names added by programs. SoNaR-1, one million words mostly taken from it, has labels that people checked, such as which words are names of people or places.
    - The **Corpus Gesproken Nederlands** (CGN, the Spoken Dutch Corpus) holds about 900 hours of Dutch speech, 8,940,098 words, collected from 1998 to 2004 in the Netherlands and Flanders: conversations, telephone calls, lessons, news, read texts. People wrote out every word, every word has a tag and a lemma, and part of it has trees.
    - **Lassy** is the name of two treebanks of written Dutch. **Lassy Small** has about a million words, 65,200 sentences, with trees that people checked; **Lassy Large** about 700 million words, parsed by Alpino without a check. Reviewers at the University of Copenhagen checked 500 sentences of Lassy Small and 250 of Lassy Large: the analysis was correct for 97.8% and for 78.4% of them.
    - The Instituut voor de Nederlandse Taal (Dutch Language Institute) hands out these three for free, for research and teaching, not for commercial use: you log in and accept a license, and for the CGN you sign one. Open Dutch WordNet, of *Corpora and treebanks*, is free on GitHub.

    The treebanks of UD are free to download: UD_Dutch-Alpino, with newspaper sentences of the Eindhoven corpus, questions, sentences to test the grammar of Alpino, example sentences of the ANS, and a part of Lassy Small, 13,603 sentences in all; and UD_Dutch-LassySmall, taken from Lassy Small. The language models of later lessons learn from corpora too: BERTje, a Dutch model of 2019, learned partly from SoNaR-500, and its makers left out the texts that are also in SoNaR-1 and Lassy Small, the parts that people checked.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **separable verb** | a verb such as *opbellen* whose particle stands apart in a main clause: *Ik bel je op* |
        | **gap** | in this lesson: how many places a particle stands after its verb; negative when it stands before it |
        | **Alpino** | a parser for Dutch from the University of Groningen, built on a grammar written by hand |
        | **Frog** | a toolkit for Dutch that adds tags, lemmas, the parts of words, dependency trees and names |
        | **SoNaR** | a corpus of written Dutch: SoNaR-500, more than 500 million words, and SoNaR-1, one million words with labels checked by people |
        | **Corpus Gesproken Nederlands** | the Spoken Dutch Corpus: about 900 hours of Dutch speech from the Netherlands and Flanders, written out; CGN for short |
        | **Lassy** | two treebanks of written Dutch: Lassy Small, about a million words checked by people, and Lassy Large, about 700 million parsed by Alpino |
        | `(form, lemma, tag, head, relation)` | a word of a sentence in this lesson: the word, its lemma, its tag, the number of its head, and the relation |
        | `"ge" in "opgebeld"` | `in` between two texts: `True` when the first stands somewhere in the second |
        | `"2,5".replace(",", ".")` | a copy of the text with every `,` replaced by `.`: `"2.5"`; with `""` as the second text, the `,` is left out |
        """),
        L.watch_out("UD calls *plaats* in *vindt plaats* a particle, but your dictionary calls it a noun. The labels of a treebank follow its guidelines, which group some fixed combinations with the separable verbs. When you count particles in a treebank, you count what its guidelines call a particle."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Read a word of the treebank as five parts: `form, lemma, tag, head, relation`. Word number `i` stands at place `i - 1` of the sentence.
    2. To find the separable verbs, look at the verbs and their lemmas, not only at the particles: a verb written as one word has no particle of its own.
    3. To see where the particles stand, go over the word numbers of each sentence, take the words with the relation `compound:prt`, and compare where each stands with where its head stands.
    4. Count over a whole treebank in a dictionary, and compare Dutch with English.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds three sentences as lists of words, each word with its five parts: `BEL`, *Ik bel je morgen op*; `OPGELOST`, *We hebben het probleem samen opgelost* (we solved the problem together); and `PICK`, *Pick it up*. The lemmas, tags and relations are the course's, as the two treebanks give them. The second cell defines a new `particles`, which gives each particle of a sentence with the form of its verb, not with numbers as in *Corpora and treebanks*, and then shows the lemma of every word of `OPGELOST`.
    """)
    return


@app.cell
def sentences_cell():
    BEL = [
        ("Ik", "ik", "PRON", 2, "nsubj"),
        ("bel", "op_bellen", "VERB", 0, "root"),
        ("je", "je", "PRON", 2, "obj"),
        ("morgen", "morgen", "ADV", 2, "advmod"),
        ("op", "op", "ADP", 2, "compound:prt"),
        (".", ".", "PUNCT", 2, "punct"),
    ]
    OPGELOST = [
        ("We", "we", "PRON", 6, "nsubj"),
        ("hebben", "hebben", "AUX", 6, "aux"),
        ("het", "het", "DET", 4, "det"),
        ("probleem", "probleem", "NOUN", 6, "obj"),
        ("samen", "samen", "ADV", 6, "advmod"),
        ("opgelost", "op_lossen", "VERB", 0, "root"),
        (".", ".", "PUNCT", 6, "punct"),
    ]
    PICK = [
        ("Pick", "pick", "VERB", 0, "root"),
        ("it", "it", "PRON", 1, "obj"),
        ("up", "up", "ADP", 1, "compound:prt"),
        (".", ".", "PUNCT", 1, "punct"),
    ]
    return BEL, OPGELOST, PICK


@app.cell
def example_particles(BEL, OPGELOST, PICK):
    def particles(words):
        found = []
        for form, lemma, tag, head, relation in words:
            if relation == "compound:prt":
                found.append((form, words[head - 1][0]))
        return found

    print(particles(BEL))
    print(particles(PICK))
    print(particles(OPGELOST))
    for form, lemma, tag, head, relation in OPGELOST:
        print(form, lemma, tag)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *op* hangs from *bel*, and *up* from *Pick*: in a loop over the words, `words[head - 1]` is the word of the head, and its `[0]` the form. `OPGELOST` has no word with the relation `compound:prt`, so `particles` gives an empty list, although *opgelost* is a separable verb. Its lemma, `op_lossen`, still shows it. The tags tell the verbs apart too: *opgelost* has the tag `VERB`, and *hebben*, an auxiliary, the tag `AUX`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_separable():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. separable: separable verbs, apart or not

        Write `separable(words)`: give back a list with the separable verbs of the sentence, from left to right, each written as one word, as a dictionary has it. `separable(BEL)` gives `['opbellen']`, and `separable(OPGELOST)` gives `['oplossen']`: a separable verb counts also when it stands in the sentence as one word. Replace the `...`. The check calls `separable(BEL)`; then sentences of its own, such as `GRONDWET`, *De grondwet is aangepast* (the constitution has been changed), the two verbs *voorkomen*, and `PICK`.
        """),
        L.task("Find the separable verbs of a sentence by their lemmas."),
    ])
    return


@app.cell(hide_code=True)
def exercise_separable():
    box_separable = L.code_box("separable", lines=6)
    box_separable
    return (box_separable,)


@app.cell(hide_code=True)
def check_separable(box_separable):
    result_separable = L.check("separable", box_separable.value)
    result_separable
    return (result_separable,)


@app.cell(hide_code=True)
def hints_separable():
    L.hints(
        "separable",
        [
            "Go over the words, and take each apart into `form, lemma, tag, head, relation`. Keep a word when its `tag` is `VERB` and its `lemma` shows a particle, and append that lemma, written as one word.",
            "What does the lemma of *bel* in `BEL`, `op_bellen`, have that the lemma of *hebben* in `OPGELOST` lacks? And how do you take that out of the lemma?",
            r'`        if tag == "VERB" and ___:`',
        ],
        why="`found = []`, then `for form, lemma, tag, head, relation in words:` with `if tag == \"VERB\" and \"_\" in lemma:` under it, and `found.append(lemma.replace(\"_\", \"\"))` under that; after the loop, `return found`. `separable(OPGELOST)` gives `['oplossen']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_separable():
    mo.md(r"""
    The picture under the check runs your function on all sentences of UD_Dutch-Alpino. Only about one in three of the separable verbs it finds has a particle of its own in the sentence, which a search for particles such as `particles` would find. And the lemma tells what the form cannot: *voorkomen* is written the same for both verbs, and only the lemma says which of the two it is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. gap_counts: how far is the particle?

        Write `gap_counts(sentences)`: give back a dictionary with, for every gap, how many particles of the sentences stand at that gap. A particle is a word with the relation `compound:prt`, and its head is its verb. The gap is 3 for *op* in `BEL` and 2 for *up* in `PICK`; when the particle stands two places before its verb, as *dicht* in *We proberen de deur dicht te doen* (we try to close the door), it is −2. Go over the sentences, and in each over the word numbers `i`, as `dependents` did in [Who depends on whom: dependency trees](/course/open/nlp-dependencies?lang=en); count as `count_tags` did in *Parts of speech and phrases*. `gap_counts([BEL, PICK, OPGELOST])` gives `{3: 1, 2: 1}`. Replace the `...`. The check calls that; then lists of its own, also with English sentences and with no sentence at all.
        """),
        L.task("Count how many particles stand at each gap from their verb."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gaps():
    box_gaps = L.code_box("gap_counts", lines=12)
    box_gaps
    return (box_gaps,)


@app.cell(hide_code=True)
def check_gaps(box_gaps):
    result_gaps = L.check("gap_counts", box_gaps.value)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gap_counts",
        [
            "Make `counts = {}` before both loops. For each word with the relation `compound:prt`, work out its `gap` from `i` and `head`, and then add 1 to the count of that gap, or start it at 1.",
            "In `BEL`, which number does *op* have, and which number its head *bel*? How do you get the gap 3 from those two?",
            "`                gap = ___`",
        ],
        why=r'`counts = {}`, then `for words in sentences:` and inside it `for i in range(1, len(words) + 1):` with `form, lemma, tag, head, relation = words[i - 1]`; `if relation == "compound:prt":` with `gap = i - head` under it, then `if gap in counts:` with `counts[gap] = counts[gap] + 1`, and `else:` with `counts[gap] = 1`. After both loops, `return counts`. `gap_counts([BEL, PICK, OPGELOST])` gives `{3: 1, 2: 1}`.',
    )
    return


@app.cell(hide_code=True)
def text_after_gaps():
    mo.md(r"""
    The picture under the check runs your function on both treebanks, and shows each gap as a share of all particles, so that the two languages compare. In English, a particle stands right after its verb or one or two words later: *pick up the phone*, *pick it up*. In Dutch, the bars spread out on both sides of the verb, and the highest stands before it.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: sentences for every gap

    The slider sets a gap. For each, the picture shows a sentence of UD_Dutch-Alpino with a particle at that gap, chosen by the course among the short sentences without names of people: its tree, with the arc to the particle in bold and the lemma under the verb.
    """)
    return


@app.cell(hide_code=True)
def demo_cell():
    demo_gap = L.explore("gap", "gap", lambda _g: C.explore_frame(_g, L.lang), start=3, values=C.EXPLORE_GAPS, role="",
                         task="Slide from 1 to 12. What stands between the verb and its particle when the gap is large?")
    demo_gap
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With a large gap, the middle of the clause stands between the verb and its particle: the subject when it follows the verb, objects, words such as *altijd* (always), phrases with a preposition. The conjugated verb stands in the second place, or first in a question or a command, and the particle at the end of the clause, so the longer the clause, the larger the gap: in *Ik sta 's morgens om zeven uur op* (I get up at seven in the morning), 6. A particle before its verb stands just before a verb at the end of the clause, and the most common gap is −2: in *ik probeer je morgen terug te bellen*, *te* stands between them, as in the *aan te vallen* of the ANS. 333 of the 495 particles at −2 have *te* in between. The two sentences of *terugbellen* are also examples in the guidelines of Lassy.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_corpus():
    mo.md(r"""
    ### Exercise 3. Which corpus of Dutch?

    Each task below needs Dutch data. Put each under the corpus that fits it best.
    """)
    return


@app.cell(hide_code=True)
def exercise_corpus():
    sort_corpus = L.sort("corpus")
    sort_corpus
    return (sort_corpus,)


@app.cell(hide_code=True)
def check_corpus(sort_corpus):
    result_corpus = L.check("corpus", sort_corpus)
    result_corpus
    return (result_corpus,)


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.md(r"""
    ### Exercise 4. New sentences, without a tree
    """)
    return


@app.cell(hide_code=True)
def choice_rule_cell():
    choice_rule = L.choice(
        "rule",
        question="You have 10,000 new Dutch sentences, without trees or lemmas. For each separable verb, you want the verb as one word, also when its particle stands apart, as in *Ik bel je morgen op*. What helps most?",
        options={
            "stemmer": "Porter's stemmer for Dutch, word by word",
            "right": "A parser for Dutch, such as Alpino",
            "list": "A list of every Dutch verb with its forms",
            "before": "A rule that joins each particle to the word right before it",
        },
        correct="right",
        why={
            "right": "A parser links each particle to its verb, wherever it stands in the clause: Alpino gives *speelt ... op* the base form `speel_op`, and Frog and the Dutch pipelines of spaCy link the particle to its verb too. A parser makes mistakes, but it looks at the whole sentence.",
            "stemmer": "A stemmer looks at one word at a time: *bel* and *op* stay two words, and nothing tells it that they belong together. See *Stems and lemmas*.",
            "list": "A list finds *opbellen*, *opbelt* and *opgebeld*. But in *Ik bel je morgen op* the verb is *bel*, and a list cannot tell that the *op* three words further belongs to it, and is not a preposition.",
            "before": "In the Dutch treebank, 151 of the 1,952 particles stand right after their verb; the others stand further away, or before it. See the picture under Exercise 2.",
        },
    )
    choice_rule
    return (choice_rule,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Corpora and treebanks](/course/open/nlp-corpora?lang=en) found the particles of a treebank by their labels, and [Stems and lemmas](/course/open/nlp-stemming?lang=en) showed what a lemma is. Dutch has tools and corpora of its own: Alpino and Frog parse Dutch, spaCy and Stanza have Dutch models, trained on Dutch treebanks of UD, and SoNaR, the Corpus Gesproken Nederlands and Lassy hold written and spoken Dutch, partly checked by people. In UD_Dutch-Alpino, you found the separable verbs by their lemmas, also those written as one word, and counted where the particles stand: in English nearly always at most three places after the verb, in Dutch on both sides of it, often far away. A rule written for English misses most of them; a parser for Dutch links them to their verb. The next lesson starts the module about meaning: *A word by the company it keeps*.

    **An everyday example.** A charger from the United Kingdom needs an adapter in the Netherlands: the power is the same, the plug is not. A tool made for English works on Dutch only as far as Dutch works like English.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rule, result_corpus, result_gaps, result_separable):
    L.wrap_up(result_separable, result_gaps, result_corpus, choice_rule)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
