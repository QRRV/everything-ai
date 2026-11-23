# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-coreference. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-coreference", "en")
    C = L.checks   # the passage of the museum, the mentions of LitBank, the course's numbers, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [A text is more than its sentences](/course/open/nlp-discourse?lang=en) and [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_old():
    quick_old = L.choice(
        "quick-old",
        question="In *A text is more than its sentences*, a mention is old when the text has named that thing before. Which mention of the second sentence of *A storm hit the town. The storm blew the roof off a shed.* is old?",
        options={
            "roof": "*the roof*, since every mention with *the* is old",
            "shed": "*a shed*, since a town has sheds",
            "right": "*The storm*: the text named it before",
            "none": "None: a sentence names its own things",
        },
        correct="right",
        why={
            "right": "The first sentence brought the storm in as *a storm*, and *the storm* picks it up again: two mentions of one storm. This lesson finds such mentions in a whole text, also when the later one is a pronoun such as *it*.",
            "roof": "*The* often goes with something old, but the text has not named a roof before. Which thing did the first sentence name? See [A text is more than its sentences](/course/open/nlp-discourse?lang=en).",
            "shed": "*A* brings the shed in as something new. Which thing did the first sentence name? See [A text is more than its sentences](/course/open/nlp-discourse?lang=en).",
            "none": "A mention is old when the text named the thing before, also in an earlier sentence. Which thing of the second sentence did the first one name? See [A text is more than its sentences](/course/open/nlp-discourse?lang=en).",
        },
    )
    quick_old
    return (quick_old,)


@app.cell(hide_code=True)
def quick_check_names():
    quick_names = L.choice(
        "quick-names",
        question="A tagger for names as in *Finding names* reads *Emma Woodhouse lived in Highbury. Emma rarely left it.* Which spans does it mark, each with a type?",
        options={
            "it": "The names, and *it* as a place too",
            "right": "*Emma Woodhouse*, *Highbury* and *Emma*",
            "first": "Just *Emma Woodhouse* and *Highbury*, since a name counts once",
            "capitals": "Each word with a capital, one word at a time",
        },
        correct="right",
        why={
            "right": "NER marks the spans of names: *Emma Woodhouse* and *Emma* as a person, *Highbury* as a place. It does not say that *Emma* is the same person as *Emma Woodhouse*, or that *it* is Highbury. That is coreference, the subject of this lesson.",
            "it": "*It* is a pronoun, not a name, and a tagger for names leaves it out. Which spans are names? See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
            "first": "The tagger marks every span that is a name, also a name that came before. How many names does the text hold? See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
            "capitals": "The tagger marks spans, and one span can hold more words: a first name and a last name together. See [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en).",
        },
    )
    quick_names
    return (quick_names,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A text names the same thing again and again, and seldom in the same words: first a name, then a description, then *it* or *they*. Daniel Jurafsky and James Martin open their chapter on coreference with: "An important component of language processing is knowing who is being talked about in a text." A program that answers questions about a text, or sums one up, has to know which words are about the same person or thing. This lesson groups the mentions of a text by what they refer to, and tries a simple rule for what *it* refers to.

    **An everyday example.** A note says *The book is on the desk. Can you take it with you?* You read *it* as the book at once, though the desk came later and is a thing too: nobody takes a desk along.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="In 100 works of fiction, people marked which mentions refer to the same person or thing. The course took every *it*, *its* and *itself* that they linked to an earlier mention: 318 cases. A rule picks the nearest earlier name, description or pronoun that is a thing, not a person. In how many of every 100 cases does it pick something that refers to the same thing?",
        options={
            "low": "About 25 of every 100",
            "half": "About 45 of every 100",
            "right": "About 68 of every 100",
            "high": "About 95 of every 100: nearly always",
        },
        reveal="**About 68:** 216 of the 318, 0.679, with nothing but the order of the text and two kinds. Without asking for a thing, the nearest earlier mention gets 137 right, 0.431.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Mentions, entities, chains

    [A text is more than its sentences](/course/open/nlp-discourse?lang=en) called each time a text names or points to a thing a mention. Jurafsky and Martin: "Two or more referring expressions that are used to refer to the same discourse entity are said to corefer". A referring expression is a mention; a discourse entity is the person or thing that the text speaks of, as the text builds it up, in this lesson an **entity**. And: "The set of coreferring expressions is often called a coreference chain or a cluster." The picture shows the course's own passage, with each mention in a box and the letter of its **chain** under it.
    """)
    return


@app.cell(hide_code=True)
def museum_cell():
    mo.Html(C.museum_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_anaphor():
    mo.md(r"""
    Eight mentions, five entities. Four mentions are the painting: a description, then *It*, then *it* twice. An entity with only one mention, such as the museum here, is a **singleton**. Finding the chains is **coreference resolution**, in the words of Jurafsky and Martin "the task of determining whether two mentions corefer, by which we mean they refer to the same entity in the discourse model". It starts with finding the mentions; this lesson starts from mentions that people marked.

    ### Anaphor and antecedent

    Jurafsky and Martin: "Reference in a text to an entity that has been previously introduced into the discourse is called anaphora, and the referring expression used is said to be an anaphor, or anaphoric." The earlier mention of the same entity is the **antecedent**. In the passage, the three times *it* are **anaphors**; a name that comes back is one too. A footnote of theirs: "We will follow the common NLP usage of anaphor to mean any mention that has an antecedent". In the narrower usage that they set aside, only mentions such as pronouns, whose meaning depends on the antecedent, are anaphors.

    ### LitBank

    The data of this lesson is LitBank, by David Bamman, Olivia Lewke and Anya Mansoor (2020): about 2,000 words from each of 100 works of English fiction, from 1719 to 1922. People marked by hand every mention of six types of entity (people, facilities, locations, geo-political entities, organizations and vehicles) and grouped the mentions into chains. In their words, "the majority (83.1%) of mentions among these annotations are people". The course's copy holds 28,411 mentions in 7,235 chains; 5,071 of those chains are singletons. Bamman and others: "17.4% of all mentions are singletons in our data, in contrast to the 56% observed by Recasens et al. (2013) in OntoNotes", a dataset mostly of news, conversations, the Bible and texts from the web. The course counts 0.178.

    ### The nearest earlier mention that fits

    Which antecedent does a pronoun have? Jurafsky and Martin list what helps a reader decide. One is recency: "Entities introduced in recent utterances tend to be more salient than those introduced from utterances further back." Their example: *The doctor found an old map in the captain's chest. Jim found an even older map hidden on the shelf. It described an island.* There, *it* is more likely Jim's map. Another is that *it* refers to a thing, not a person: "Maryam has a theorem. It is exciting. (it=the theorem, not Maryam)".

    Together that gives a rule: take the nearest earlier mention that fits. In this lesson, each mention has one of two **kinds**: `"person"` or `"thing"`. *It*, *its* and *itself* are things; every other personal pronoun, such as *I*, *you* or *they*, is a person. A name or a description is a person when LitBank marks it as one, and else a thing. The picture applies the rule to the three times *it* in the passage, once without the kind and once with it.
    """)
    return


@app.cell(hide_code=True)
def rule_cell():
    mo.Html(C.rule_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_back():
    mo.md(r"""
    With the kind, the rule gets two of the three right. For the second *it*, the nearest earlier thing is the hall, and a reader takes the painting: a guard straightens a painting that hangs crooked, not a hall. The last *it* is right with the kind, though the rule picks another *it*: what counts is that the pick refers to the same entity.

    How far back does a pronoun look? A footnote of Jurafsky and Martin: "Pronouns also usually (but not always) refer to entities that were introduced no further than one or two sentences back in the ongoing discourse, whereas definite noun phrases can often refer further back." The picture counts that for *it*, *its* and *itself* in LitBank.
    """)
    return


@app.cell(hide_code=True)
def back_cell():
    mo.Html(C.back_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.vstack([
        mo.md(r"""
        Not every *it* refers to something. Jurafsky and Martin: "Many uses of pronouns like it in English and corresponding pronouns in other languages are not referential." Their first example is *it is raining*. *Pride and Prejudice* opens with such an *it*: *It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.* And LitBank marks only six types of entity. Its texts hold 2,306 times *it*, *its* or *itself*; LitBank marks 333 of them as mentions, and 318 of those have an earlier mention in their chain.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **entity** | the person or thing a text speaks of, as the text builds it up |
        | **coreference**, **corefer** | two mentions corefer when they refer to the same entity |
        | **coreference chain**, **cluster** | the mentions of one entity, in the order of the text |
        | **singleton** | an entity with only one mention |
        | **coreference resolution** | finding which mentions of a text corefer, and so the chains |
        | **anaphor** | a mention with an earlier mention of the same entity |
        | **antecedent** | that earlier mention; in this lesson, the nearest one |
        | **kind** | in this lesson, `"person"` or `"thing"`: what a pronoun has to match |
        | `C.MUSEUM` | the eight mentions of the passage, in order, each a pair `(text, kind)`, such as `("a painting", "thing")` |
        | `C.MUSEUM_IDS` | the chain of each mention of `C.MUSEUM`, at the same place: `"museum"`, `"painting"`, `"painting"`, and so on |
        """),
        L.watch_out("A chain is about what the words refer to, not about the words: *a painting* and *It* are in one chain, and the same word can be in two chains, as in *The truck pulled a trailer. It towed it up the hill.* And in NLP an anaphor is any mention with an antecedent, also a name that comes back, not only a pronoun."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Chains: go over the mentions with their chains. A chain that is new gets a list with the text of the mention; a chain seen before gets the text added to its list.
    2. Antecedents: go over the chains of the mentions, and keep in a dict, for each chain, the place of its latest mention so far. A mention whose chain is in the dict gets that place; a mention with a new chain gets `None`. Then the dict learns the place of this mention.
    3. The rule: go over the places before mention `i`, and remember each place whose mention has the kind of mention `i`. The last place remembered is the nearest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell counts the kinds of the mentions of the passage and prints the counts. Then it prints the mention at place 5 and its chain.
    """)
    return


@app.cell
def example_kinds():
    counts = {"person": 0, "thing": 0}
    for text, kind in C.MUSEUM:
        counts[kind] = counts[kind] + 1
    print(counts)
    print(C.MUSEUM[5], C.MUSEUM_IDS[5])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Six of the eight mentions are things, so an *it* has several candidates. Place 5 holds the second *it*: a thing, in the chain of the painting. Exercise 1 collects the mentions of each chain.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chains():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. chains: the mentions of each entity

        Write `chains(mentions, ids)`: `mentions` holds the mentions of a text in order, each a pair `(text, kind)`, and `ids` holds the chain of each mention, at the same place. Give back a dict with, for each chain, the list of the texts of its mentions, in the order of the text. Replace the `...`. The check calls `chains(C.MUSEUM, C.MUSEUM_IDS)`, which gives `{"museum": ["The museum"], "painting": ["a painting", "It", "it", "it"], "hall": ["the hall"], "guard": ["A guard"], "visitors": ["visitors"]}`. Then cases of its own: no mentions, every mention in a chain of its own, chains numbered `0` and `1` as in LitBank with one text twice in a chain, the word *it* in two chains, and one chain for all mentions.
        """),
        L.task("Collect the texts of the mentions of each chain, in order."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chains():
    box_chains = L.code_box("chains", lines=10)
    box_chains
    return (box_chains,)


@app.cell(hide_code=True)
def check_chains(box_chains):
    result_chains = L.check("chains", box_chains.value)
    result_chains
    return (result_chains,)


@app.cell(hide_code=True)
def hints_chains():
    L.hints(
        "chains",
        [
            "For each mention, `found` either has a list for its chain `c` already, or not yet: an `if` with an `else`.",
            "`c in found` asks whether `found` has a list for `c`. If so, `found[c].append(text)` adds the text at the end. If not, `found[c]` gets a new list: what should that list hold right away?",
            "The line under `else:`: `            found[c] = ___`",
        ],
        why="Replace the `...` with `if c in found:` and `found[c].append(text)` under it, then `else:` with `found[c] = [text]` under it. The example gives `{\"museum\": [\"The museum\"], \"painting\": [\"a painting\", \"It\", \"it\", \"it\"], \"hall\": [\"the hall\"], \"guard\": [\"A guard\"], \"visitors\": [\"visitors\"]}`.",
    )
    return


@app.cell(hide_code=True)
def text_after_chains():
    mo.md(r"""
    The picture under the check runs your function on the 366 mentions of the opening of *Pride and Prejudice*, by Jane Austen (1813), and shows the five longest chains. The longest, Mr. Bennet, has 103 mentions, and most of them are pronouns. In all of LitBank, 15,815 of the 28,411 mentions are pronouns. Exercise 2 finds, for each mention, the nearest earlier mention of its chain.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_antecedents():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. antecedents: the nearest earlier mention of the same chain

        Write `antecedents(ids)`: `ids` holds the chain of each mention of a text, in order. Give back a list with, for each mention, the place of the nearest earlier mention of the same chain, or `None` when its chain is new there. `last` holds, for each chain seen so far, the place of its latest mention; the line under the `...` keeps it up to date. Replace the `...`. The check calls `antecedents(C.MUSEUM_IDS)`, which gives `[None, None, 1, None, None, 2, None, 5]`: *It* at place 2 goes back to *a painting* at place 1, and the last *it* to the *it* at place 5. Then cases of its own: no mentions, every mention in a chain of its own, one chain three times, chains numbered as in LitBank, a chain inside another, and a chain whose earlier mention is at place `0`.
        """),
        L.task("Give the place of the nearest earlier mention of the same chain, or None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_antecedents():
    box_antecedents = L.code_box("antecedents", lines=10)
    box_antecedents
    return (box_antecedents,)


@app.cell(hide_code=True)
def check_antecedents(box_antecedents):
    result_antecedents = L.check("antecedents", box_antecedents.value)
    result_antecedents
    return (result_antecedents,)


@app.cell(hide_code=True)
def hints_antecedents():
    L.hints(
        "antecedents",
        [
            "Put an `if` with an `else` above the line that keeps `last` up to date: one branch adds to `out` the place that `last` holds for the chain `ids[k]`, the other adds `None`.",
            "Under the `if`: `out.append(last[ids[k]])`. Under the `else`: `out.append(None)`. The `if` looks at `last` before it learns the place of mention `k`.",
            "`        if ___:`",
        ],
        why="Replace the `...` with `if ids[k] in last:` and `out.append(last[ids[k]])` under it, then `else:` with `out.append(None)` under it. The line `last[ids[k]] = k` stays below. The example gives `[None, None, 1, None, None, 2, None, 5]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_antecedents():
    mo.md(r"""
    The picture under the check runs your function on the chains of all 100 works. Of the pronouns, 0.956 have an earlier mention in their chain, against 0.712 of the names and 0.393 of the descriptions: a description often brings in someone or something new. Bamman and others: "Pronouns have a median distance of 2 entities to their nearest antecedent, with 90% of antecedents appearing within 5 mentions, and 95% of antecedents within 9 mentions." In the course's copy, for the pronouns that have one: a median of 2 mentions back, 0.912 within 5 and 0.956 within 9. Exercise 3 guesses an antecedent the way a program without the chains has to: with the rule.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. pick: the nearest earlier mention of the same kind

        Write `pick(mentions, i)`: `mentions` holds mentions as pairs `(text, kind)`, and `i` is a place in that list. Give back the place of the nearest mention before place `i` with the same kind as mention `i`, or `None` when no mention before `i` has that kind. Replace the `...`. The check calls `pick(C.MUSEUM, 5)`: the second *it*, a thing. The nearest earlier thing is *the hall*, at place 3, so it gives `3`: what the rule picks, though a reader takes the painting. Then cases of its own: place `0`, the first *it*, *visitors*, the last *it*, a person with only a thing before it, a person whose nearest earlier person is four places back, and a thing with a mention of the same kind after it.
        """),
        L.task("Give the place of the nearest earlier mention of the same kind, or None."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=8)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "Go over the places `j` before `i`. For each mention there with the kind `kind`, remember `j` in `best`. The loop goes from front to back, so the last place remembered is the nearest.",
            "`mentions[j][1]` is the kind of mention `j`: inside the loop, `if mentions[j][1] == kind:` with `best = j` under it. The loop has to stop before place `i` itself.",
            "`    for j in ___:`",
        ],
        why="Replace the `...` with `for j in range(i):`, and under it `if mentions[j][1] == kind:` with `best = j` under that. The example gives `3`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    On the 318 cases of LitBank, your function gets 0.679 right: the answer to the guess at the start. The nearest earlier mention of any kind gets 0.431. Some of what the rule misses takes knowledge of what words mean. An example of Jurafsky and Martin: *I ate the soup in my new bowl after cooking it for hours*. The nearest earlier thing is the bowl, and a reader knows that you eat soup, not a bowl. Jurafsky and Martin call such knowledge a selectional restriction: a verb such as *eat* takes only certain things as its object, here something edible. And 28 of the 318 are in a chain that LitBank marks as a person, such as an infant or a child called *it*; the rule gets 10 of those right.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_trophy():
    mo.md(r"""
    ### Exercise 4. One word after *it*

    In 1972, Terry Winograd pointed out that some pronouns seem to need knowledge of the world or careful reasoning to resolve. In 2011, Hector Levesque proposed a test built from such cases, the Winograd Schema Challenge. In the words of Jurafsky and Martin: "The problems are framed as a pair of statements that differ in a single word or phrase, and a coreference question". The question below uses their example.
    """)
    return


@app.cell(hide_code=True)
def choice_trophy_cell():
    choice_trophy = L.choice(
        "trophy",
        question="*The trophy didn't fit into the suitcase because it was too large.* and *The trophy didn't fit into the suitcase because it was too small.* What does the rule of exercise 3 pick for *it* in each sentence, and how does it do?",
        options={
            "subject": "The trophy in both, since the subject of a sentence comes first",
            "right": "The suitcase in both, so it is wrong once",
            "after": "The trophy, then the suitcase: the word after *it* decides",
            "nothing": "Neither: no mention before *it* has its kind",
        },
        correct="right",
        why={
            "right": "Both sentences put *the trophy* and then *the suitcase* before *it*, and all three are things. So the rule picks *the suitcase* both times: right for *too small*, wrong for *too large*. The word that decides comes after *it*, and what settles it is, in the words of Jurafsky and Martin, \"that a bigger object cannot fit into a smaller object\". Such problems were \"designed to be easily disambiguated by the human reader, but hopefully not solvable by simple techniques such as selectional restrictions, or other basic word association methods.\" They add that \"a large percentage of the original set of problems can be solved by pretrained language models, fine-tuned on Winograd Schema sentences\".",
            "subject": "The rule looks at the order and the kind, not at the role of a word in the sentence. Which thing comes last before *it*?",
            "after": "The rule only looks at what comes before *it*; the words after *it* do not count. Which thing comes last before *it* in each sentence?",
            "nothing": "Both things in the sentence come before *it*, and *it* is a thing too. Which of the two comes later?",
        },
    )
    choice_trophy
    return (choice_trophy,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: the rule on the stories

    Move the slider through 246 of the 318 cases of *it*, *its* and *itself* from LitBank: the ones where the pronoun, its antecedent and what the rule picks fit in at most three short sentences, each a sentence the course shows as an example. Untick *Check the kind*, and the rule takes the nearest earlier mention of any kind. The slider starts in *Pride and Prejudice*: *have you heard that Netherfield Park is let at last?* asks Mrs. Bennet, and the *it* two sentences later is Netherfield Park.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_case = mo.ui.slider(1, len(C.DEMO_CASES), value=C.DEMO_START, label="Case", full_width=True)
    demo_kind = mo.ui.checkbox(value=True, label="Check the kind")
    mo.vstack([demo_case, demo_kind])
    return demo_case, demo_kind


@app.cell(hide_code=True)
def demo_look(demo_case, demo_kind):
    mo.Html(C.demo_picture(demo_case.value, demo_kind.value, L.lang))
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A text is more than its sentences](/course/open/nlp-discourse?lang=en) marked each mention as new or old; this lesson grouped the mentions of a text into chains, found the antecedent of each anaphor, and tried a rule for what *it* refers to. [Finding names: named entity recognition](/course/open/nlp-named-entities?lang=en) found the names; coreference links a name to the descriptions and pronouns of the same entity. The rule of exercise 3, the nearest earlier mention of the same kind, gets 0.679 of the 318 cases of *it* in LitBank right, against 0.431 without the kind, with nothing but the order of the text and two kinds. What it misses takes knowledge of the world, as with the trophy and the suitcase. The next lesson, *Encoders, decoders and the GPT line*, turns to the neural models that later coreference systems build on.

    **An everyday example.** A note says *Ask Kim about the report. It is due on Friday.* You link *It* to the report and not to Kim without a second thought: the nearest earlier mention of the right kind.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_trophy, result_antecedents, result_chains, result_pick):
    L.wrap_up(result_chains, result_antecedents, result_pick, choice_trophy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
