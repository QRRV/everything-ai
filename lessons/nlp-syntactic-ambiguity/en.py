# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-syntactic-ambiguity. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-syntactic-ambiguity", "en")
    C = L.checks   # the course's trees, the counts, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions: one about [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en), one about [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_place():
    quick_place = L.choice(
        "quick-place",
        question="The grammar of *Phrase structure and endless sentences* has NP → DET NOM | PROPN, VP → VERB NP | VERB NP PP, and PP → ADP NP. Where does it put *near Chicago* in the tree of *a long train passed the bridge near Chicago*?",
        options={
            "right": "Under the VP, beside the NP",
            "np": "Inside the NP, with *the bridge*",
            "s": "Right under S",
        },
        correct="right",
        why={
            "right": "Only VP → VERB NP PP takes a PP, so the PP sits in the verb phrase, next to the NP *the bridge*. No rule of that grammar puts a PP inside a noun phrase. This lesson adds one, and the sentence gets a second tree.",
            "np": "That would need a rule that puts a PP into a noun phrase, and that grammar has none: an NP is DET NOM or PROPN. Only VP → VERB NP PP takes a PP. See *Rules that build sentences* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
            "s": "S → NP VP has no PP: the PP comes in with VP → VERB NP PP, under the VP. See *Rules that build sentences* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
        },
    )
    quick_place
    return (quick_place,)


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Church and Patil counted the readings of sentences such as *Put the block in the box on the table*. With two prepositional phrases, as there, they found 2 readings. How many with three?",
        options={
            "right": "5",
            "three": "3: one more place for the new phrase",
            "four": "4",
        },
        correct="right",
        why={
            "right": "Five, and fourteen with four: each new phrase can attach to the phrases before it too, so the count grows faster and faster. This lesson counts such trees itself.",
            "three": "Each new phrase can attach to the phrases before it too: Church and Patil found 5 with three phrases and 14 with four. See *Counting the choices* in [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
            "four": "Church and Patil found 5 with three phrases, and 14 with four: the count grows faster than doubling. See *Counting the choices* in [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The small grammar of *Phrase structure and endless sentences* gave each of its sentences one tree. A real grammar gives many sentences more than one, and a program has to pick. Jurafsky and Martin: "Ambiguity is the most serious problem faced by syntactic parsers." They show a news sentence about President Kennedy from the Brown corpus, a collection of English texts, with a whole row of such ambiguities. At its end, *nationwide television and radio* can be bracketed as [nationwide [television and radio]] or as [[nationwide television] and radio]. Readers hardly see the other readings, since most of them make no sense; a program sees all of them. Jurafsky and Martin call that flood of readings that fit the grammar but make no sense "an irksome problem that affects all parsers".

    The tree says what belongs to what. A program that answers questions about a text needs the tree the writer meant.

    **An everyday example.** The sum 8 − 4 − 2. Read as (8 − 4) − 2, it gives 2; read as 8 − (4 − 2), it gives 6. The same numbers, in the same order: the brackets decide.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Ratnaparkhi, Reynar and Roukos (1994) gave three experts 300 cases from sentences of the *Wall Street Journal*, in which a prepositional phrase follows a verb and its object and can attach to either. With the whole sentence, the experts chose the place the treebank gives in 93 of 100 cases, on average. With only four words of each case, the verb, the main noun of the object, the preposition and the main noun after it, such as *report million in charges*, about how often?",
        options={
            "coin": "About 50 in 100: no better than a coin",
            "some": "About 71 in 100",
            "right": "About 88 in 100",
        },
        reveal="**About 88 in 100**: 88.2 on average over the three experts, a number the paper prints with a ~ before it, against 93.2 with the whole sentence. Four words carry most of what decides where the phrase attaches. The program of the paper, which learned from the treebank and also saw only the four words, chose the treebank's place in 78.0 of 100 cases on the same set. And the treebank itself is not perfect: on the 274 cases where the three experts with the whole sentence all agreed, it gave the same place in 95.7 of 100.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Two trees for one sentence

    Give the grammar of *Phrase structure and endless sentences* one rule more: NP → NP PP, a noun phrase with a PP after it is a noun phrase too. Now *a long train passed the bridge near Chicago* has two trees. In one, VP → VERB NP PP makes *near Chicago* part of the verb phrase: near Chicago, the train passed the bridge. In the other, NP → NP PP makes it part of the noun phrase: the train passed the bridge near Chicago, and not some other bridge. The picture shows both. The larger grammar L1 of Jurafsky and Martin has a rule of the same kind one level lower, Nominal → Nominal PP, where Nominal is the NOM of *Phrase structure and endless sentences*.

    Jurafsky and Martin call this **attachment ambiguity**: "A sentence has an attachment ambiguity if a particular constituent can be attached to the parse tree at more than one place." When that constituent is a prepositional phrase, it is **PP-attachment**, as with Groucho Marx's elephant in *Why language is hard for computers*: "the preposition phrase can be attached either as part of the NP or as part of the VP." Other phrases can attach in two places too. In their example *We saw the Eiffel Tower flying to Paris*, *flying to Paris* can say what the tower was doing, or what we were doing when we saw it.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_kinds():
    mo.md(r"""
    ### What *and* connects

    "Two common kinds of ambiguity are attachment ambiguity and coordination ambiguity," Jurafsky and Martin write. "In coordination ambiguity phrases can be conjoined by a conjunction like and." Their example is *old men and women*, also in *Why language is hard for computers*. Bracketed as [old [men and women]], the women are old too; bracketed as [old men] and [women], only the men are.

    ### A word with two roles

    *Teacher strikes idle kids* has two trees because its words can play other roles, as *Why language is hard for computers* showed:

    ```text
    [S [NP Teacher strikes] [VP idle [NP kids]]]     strikes by teachers leave kids without lessons
    [S [NP Teacher] [VP strikes [NP idle kids]]]     a teacher hits kids who do nothing
    ```

    In the first tree, *strikes* is a noun and *idle* a verb; in the second, *strikes* is the verb and *idle* an adjective. In a headline, with its few words, this happens easily: the crash blossoms of that lesson are headlines of this kind.

    ### Counting bracketings

    How many trees can a phrase have? Take a phrase of nouns, such as *city bus driver*, and put brackets around two groups at a time, until one bracket holds everything. Three words give two **bracketings**: [[city bus] driver], the driver of a city bus, and [city [bus driver]], a bus driver who works for the city. Four words give five, five words fourteen.

    You can count them without drawing. The outermost bracket splits the phrase into a left part and a right part: four words can split after the first, the second or the third word. Each part has bracketings of its own, and each one on the left goes with each one on the right. So a split gives the count of the left part times the count of the right part, and the splits together give the count of the whole phrase. For four words: 1 × 2 + 1 × 1 + 2 × 1 = 5.

    The count for four words needs only the counts for fewer words. So start small, keep each count in a list, and look it up when a longer phrase needs it. The counts are the same row of numbers as Church and Patil's in *Why language is hard for computers*: 1, 1, 2, 5, 14, 42 and on, the Catalan numbers.

    ### Which tree is meant?

    Readers pick a tree without noticing. Often the words decide, as the experts of the guess showed with only four words of each case. A program can learn the same from a treebank: which trees go with which words.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **attachment ambiguity** | a phrase that can attach to more than one place in the tree of a sentence |
        | **PP-attachment** | attachment ambiguity of a prepositional phrase: part of the verb phrase or of a noun phrase |
        | **coordination ambiguity** | when what *and* or *or* connects can be cut in more than one way, as in *old men and women* |
        | **bracketing** | one way to put brackets in a phrase, two groups at a time, until one bracket holds everything |
        | `["VP", "passed", ["NP", "the", "bridge"]]` | a tree in this lesson: a list with the symbol first, then its parts; a word is a string |
        | `vp[2][1]` | the item at place 1 of the item at place 2: one step down the tree for each pair of brackets |
        | a `for` inside a `for` | the inner loop runs all its rounds again for each round of the outer loop |
        """),
        L.watch_out("The trees of this lesson leave out the tags above the words, such as DET and NOUN, the NOM of *Phrase structure and endless sentences*, and the NP inside a PP. They keep S, NP, VP and PP: enough to show where the PP attaches."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Look for a phrase that can attach in two places, an *and* with words on both sides, or a word that can play two roles.
    2. Draw a tree for each reading: the words stay in their order, and only the brackets move.
    3. To count the bracketings of a phrase, split it at the top into a left and a right part, multiply their counts, and add up over the splits, from short phrases to long ones.
    4. To pick a tree, look at the words: often the verb, the object and the words of the PP are enough.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell holds the verb phrases of the two trees in the picture, as lists: in `VP_VERB` the PP is part of the verb phrase, in `VP_NOUN` part of the noun phrase. The second looks inside them.
    """)
    return


@app.cell
def trees_cell():
    VP_VERB = ["VP", "passed", ["NP", "the", "bridge"], ["PP", "near", "Chicago"]]
    VP_NOUN = ["VP", "passed", ["NP", ["NP", "the", "bridge"], ["PP", "near", "Chicago"]]]
    return VP_NOUN, VP_VERB


@app.cell
def example_look(VP_NOUN, VP_VERB):
    print(len(VP_VERB), len(VP_NOUN))
    print(VP_VERB[3])
    print(VP_NOUN[2])
    print(VP_NOUN[2][1])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `VP_VERB` has four items: the symbol, the verb, the NP and the PP. `VP_NOUN` has three: the symbol, the verb, and one noun phrase that holds both the NP and the PP. `VP_NOUN[2][1]` goes two steps down: first to that noun phrase, then to the NP *the bridge* inside it. The two lists hold the same words, in the same order. Exercise 1 turns one tree into the other.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_other():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. other_tree: the other reading

        Write `other_tree(vp)`: give back the other tree of the same words. When `vp` has four items, the PP is part of the verb phrase: give back the VP in which it is part of the noun phrase, as in `VP_NOUN`. Otherwise the PP sits in the noun phrase: give back the VP in which it is part of the verb phrase again, as in `VP_VERB`. Tell the two apart with `len(vp)`. Replace the `...`. The check calls `other_tree(VP_VERB)` and `other_tree(VP_NOUN)`; then cases of its own, with tags in the place of the words, such as `["VP", "VERB", ["NP", "PROPN"], ["PP", "ADP", "PROPN"]]`.
        """),
        L.task("Move the PP into the noun phrase, or out of it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_other():
    box_other = L.code_box("other_tree", lines=5)
    box_other
    return (box_other,)


@app.cell(hide_code=True)
def check_other(box_other):
    result_other = L.check("other_tree", box_other.value)
    result_other
    return (result_other,)


@app.cell(hide_code=True)
def hints_other():
    L.hints(
        "other_tree",
        [
            "Start with `if len(vp) == 4:`. In that case, build a new noun phrase from the NP and the PP of `vp`. In the other case, take the noun phrase apart with `phrase = vp[2]`, and put its two parts in the VP.",
            "What does `VP_NOUN[2]` hold, and what should `other_tree(VP_NOUN)` give back?",
            "`    return [\"VP\", vp[1], phrase[1], ___]`",
        ],
        why="`if len(vp) == 4:` gives back `[\"VP\", vp[1], [\"NP\", vp[2], vp[3]]]`; otherwise `phrase = vp[2]` and `return [\"VP\", vp[1], phrase[1], phrase[2]]`. `other_tree(VP_VERB)` gives `VP_NOUN`, and `other_tree(VP_NOUN)` gives `VP_VERB`.",
    )
    return


@app.cell(hide_code=True)
def text_after_other():
    mo.md(r"""
    The picture under the check draws `VP_VERB` and what your function made of it. In both trees the words read *passed the bridge near Chicago*; only the bold line into the PP starts somewhere else. A parser that finds both trees has to choose, and the words alone do not always show which tree the writer meant.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. bracketings: how many trees?

        Write `bracketings(n)`: the number of bracketings of a phrase of `n` words. Start with `counts = [0, 1]`: a phrase of no words has no bracketing, and a phrase of one word has one, the word itself. Then, for each `size` from 2 up to and including `n`: go over the splits, with `left` words in the left part, from 1 up to and including `size - 1`; add up what each split gives, as *Counting bracketings* in *The idea* worked out for four words; and append the total to `counts`. At the end, give back the count for `n` words. Replace the `...`. The check calls `bracketings(4)`, then several `n` from 1 up to 12.
        """),
        L.task("Fill the list of counts from short phrases to long ones."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("bracketings", lines=8)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("bracketings", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "bracketings",
        [
            "Two loops: the outer one over `size`, the inner one over `left`. Start `total` at 0 inside the outer loop, add what each split gives in the inner loop, and append `total` to `counts` after the inner loop.",
            "What should `bracketings(3)` give, and what does each of its two splits add?",
            "`            total = total + counts[left] * ___`",
        ],
        why="`counts = [0, 1]`; `for size in range(2, n + 1):` with `total = 0` and `for left in range(1, size):` under it, and in that loop `total = total + counts[left] * counts[size - left]`; after the inner loop `counts.append(total)`. At the end `return counts[n]`. `bracketings(4)` gives 5.",
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    The picture under the check puts your counts for 1 to 12 words beside the right ones. The bars use a scale on which each step of the same length means ten times as many. `bracketings(10)` gives 4,862: the number of trees that Church and Patil's parser found for a sentence with nine prepositional phrases. Their row is yours, shifted by one: n phrases give as many trees as n + 1 words have bracketings. A phrase of twelve words already has 58,786 bracketings, and a parser that wrote out every tree would soon drown in them.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: every bracketing

    Pick a number of words, and see every bracketing of the start of *city bus driver union pay talks*. Look for the one a reader would mean, and count how many others a program has to rule out.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_words = mo.ui.slider(2, 6, value=4, label="Words", show_value=True)
    demo_words
    return (demo_words,)


@app.cell(hide_code=True)
def demo_look(demo_words):
    mo.Html(C.demo_picture(demo_words.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 3. Which kind of ambiguity?

    Each bold sentence can be read in two ways. Put each one under the kind of ambiguity that gives it its two trees.
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
def text_exercise_agree():
    mo.md(r"""
    ### Exercise 4. When the verb decides

    A clause that starts with *who* can attach to two nouns, as a PP can. In *the friend of the singer who lives in Paris*, it is not clear who lives in Paris: the friend or the singer. Sometimes another word settles it.
    """)
    return


@app.cell(hide_code=True)
def choice_agree_cell():
    choice_agree = L.choice(
        "agree",
        question="Which of these phrases can be read in one way only?",
        options={
            "right": "*the coach of the players who was late*",
            "player": "*the coach of the player who was late*",
            "coaches": "*the coaches of the players who were late*",
            "is": "*the coach of the player who is late*",
        },
        correct="right",
        why={
            "right": "*was* is the form for one, and of the two nouns only *coach* is one person: the coach was late. The form of the verb fits one noun, so the clause can attach to that noun only.",
            "player": "*was* fits *coach* and *player* alike: both are one person. So the clause can still attach to either. Look for a phrase in which the form of the verb fits only one of the two nouns.",
            "coaches": "*were* fits *coaches* and *players* alike: both are more than one. So the clause can still attach to either. Look for a phrase in which the form of the verb fits only one of the two nouns.",
            "is": "*is* fits *coach* and *player* alike: both are one person. So the clause can still attach to either. Look for a phrase in which the form of the verb fits only one of the two nouns.",
        },
    )
    choice_agree
    return (choice_agree,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en) gave a sentence its tree, and [Why language is hard for computers](/course/open/nlp-why-language-is-hard?lang=en) counted readings. This lesson put the two together: with one rule more, the same words get two trees. You turned one tree into the other, counted how fast the bracketings of a phrase grow, sorted three kinds of ambiguity, and saw the form of a verb leave one reading. The next lesson turns from the structure of a sentence to its meaning: when is a sentence true?

    **An everyday example.** A sign says *Free coffee and cake for members*. Is the coffee free for everyone and the cake for members only, or are both for members? Most readers pick one reading without seeing the other.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agree, result_count, result_kinds, result_other):
    L.wrap_up(result_other, result_count, result_kinds, choice_agree)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
