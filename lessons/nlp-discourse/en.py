# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-discourse. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-discourse", "en")
    C = L.checks   # the course's text, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What is meant, not said](/course/open/nlp-pragmatics?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_relevant():
    quick_relevant = L.choice(
        "quick-relevant",
        question="Kim: *The bus is late again.* Sam: *My bike is in the shed.* What does Kim take from Sam's reply?",
        options={
            "where": "Where Sam keeps the bike, as a fact",
            "right": "An offer to borrow the bike",
            "bus": "That Sam knows when the bus will come",
        },
        correct="right",
        why={
            "right": "Sam's words only say where the bike is. Kim assumes that the reply is relevant, the maxim of Relation, and works out what Sam must mean: an implicature. A reader of a text does something like it, as this lesson shows: each sentence is taken to have something to do with the one before.",
            "where": "That is what the words say. But why would Sam say it right after Kim's complaint? A listener who assumes that the reply is relevant works out more. See *Cooperating* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
            "bus": "Sam says nothing about the bus. What Kim takes from the reply comes from assuming that it is relevant to the late bus: Sam offers a way around it. See *Cooperating* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
        },
    )
    quick_relevant
    return (quick_relevant,)


@app.cell(hide_code=True)
def quick_check_the():
    quick_the = L.choice(
        "quick-the",
        question="*The bike in the shed is red.* What does this sentence take for granted?",
        options={
            "likes": "That the speaker likes red bikes",
            "painted": "That someone painted the bike red last week",
            "right": "That there is one bike in the shed",
        },
        correct="right",
        why={
            "right": "*The* takes for granted that there is exactly one such thing: a presupposition. A listener who did not know learns it in passing. This lesson comes back to *the*: it often marks something the reader has already met.",
            "likes": "The sentence says nothing about what the speaker likes. What it takes for granted stays when you add *not*: *The bike in the shed is not red* still has a bike in the shed. See *Taken for granted* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
            "painted": "Nothing in the sentence is about painting. What it takes for granted stays when you add *not*: *The bike in the shed is not red* still has a bike in the shed. See *Taken for granted* in [What is meant, not said](/course/open/nlp-pragmatics?lang=en).",
        },
    )
    quick_the
    return (quick_the,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A program that summarizes picks sentences from a long text; a search engine shows a few sentences of a page; a language model writes a page one word at a time. Each sentence can be fine on its own, and the whole still not a text. Early in their chapter on this, Jurafsky and Martin give a pair from the computer scientist Jerry Hobbs (1979): a sentence about John taking a train from Paris to Istanbul, and then one saying that John likes spinach. It is incoherent, they write, "because it is unclear to a reader why the second sentence follows the first; what does liking spinach have to do with train trips?" Measuring how well sentences hang together has uses, they add, from grading essays and summarizing to, perhaps, spotting signs of disordered language. This lesson looks at what holds a text together: ties in the words, a sense underneath, and an order within each sentence.

    **An everyday example.** You paste paragraphs from three web pages into one document. Each paragraph reads well, but the second begins with *This plan*, and the reader has not met a plan yet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Regina Barzilay and Mirella Lapata (2008) gave a program pairs of texts: a news report or an accident report, and the same text with its sentences in random order. The program did not understand the texts. It only noted, for each sentence, which people and things came back, and in what role. In how many of the pairs did it rank the original text higher?",
        options={
            "right": "In about 9 of 10 pairs",
            "half": "In about half of them, as a coin toss would",
            "six": "In about 6 of 10 pairs",
        },
        reveal="**In about 9 of 10 pairs**: 87.2 in 100 for news articles of the Associated Press about earthquakes, and 90.4 in 100 for aviation accident reports of the National Transportation Safety Board, over 2,056 and 2,087 test pairs. The texts were short: 10.4 and 11.5 sentences on average. The test assumes that the original order is coherent, the authors note. And on the earthquakes, a version of the program without one of its parts did a little better, with 88.3 in 100.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### What makes a text

    Jurafsky and Martin: "We refer to such a coherent structured group of sentences as a discourse, and we use the word coherence to refer to the relationship between sentences that makes real discourses different than just random assemblages of sentences." A **discourse** can be a story, a report or a conversation. **Coherence** is that relationship in meaning: a reader sees why each sentence follows the one before. Hobbs's pair lacks it. Jurafsky and Martin's coherent version gives the traveler a conference to attend: the second sentence then gives the reason for the trip.

    Rhetorical Structure Theory, by William Mann and Sandra Thompson (1988), names such relations between neighboring parts of a text: one part gives evidence for another, elaborates it, or contrasts with it. An example after one of Jurafsky and Martin's: *Kim must be home. The bike is in the hall.* The second sentence is evidence for the first. Often no word says which relation holds. The Penn Discourse Treebank 2.0, from Rashmi Prasad and colleagues (2008), marks the relations in about a million words of the Wall Street Journal: 40,600 in all. Roughly 18,000 have a **connective** in the text, a word such as *because* or *as a result*; roughly 16,000 have none, and the reader supplies the link; the rest are of other kinds. The maxim of Relation from *What is meant, not said* is at work here too, in the course's reading: a reader assumes that the next sentence is relevant, and looks for how.
    """)
    return


@app.cell(hide_code=True)
def text_idea_cohesion():
    mo.md(r"""
    ### Ties in the words

    Part of what holds a text together shows in its words. M.A.K. Halliday and Ruqaiya Hasan (1976) called it **cohesion**. The chapters of their book *Cohesion in English* are about five kinds of ties: reference, substitution, ellipsis, conjunction and lexical cohesion. With examples of the course's own: *it* refers back to a thing named before; *one* takes the place of a word (*I need a pen. Do you have one?*); words are left out and understood (*Sam can swim, and Kim can too.*); a connective says how two sentences connect; and a word comes back, or a related one. Jurafsky and Martin describe that last kind, **lexical cohesion**, as "the sharing of identical or semantically related words in nearby sentences".

    Cohesion is not coherence. In Hobbs's pair as it stands in the book, the second sentence points back to John with a pronoun: a tie, and still no sense. And two sentences can hang together without any tie: *Kim missed the bus. The alarm had not gone off.*

    Ties can be counted. Marti Hearst (1997) used "a very simple take on lexical cohesion relations" to find where a long text changes subtopic. Hearst's program compares neighboring blocks of sentences: "The more words the blocks have in common, the higher the lexical score at the gap between them. If a low lexical score is preceded by and followed by high lexical scores, this is assumed to indicate a shift in vocabulary corresponding to a subtopic change." Exercise 1 does the same for single sentences.
    """)
    return


@app.cell(hide_code=True)
def text_idea_entities():
    mo.md(r"""
    ### The same things, coming back

    A coherent text also tends to stay with what it is about. Barbara Grosz, Aravind Joshi and Scott Weinstein (1995) built a theory of centering on this. In Jurafsky and Martin's summary, "transitions between sentences that maintain the same salient entity are considered more coherent than ones that repeatedly shift between entities": a salient entity is one that stands out at that point. The program of the guess wrote a text as a table, the entity grid, with a row for each sentence, a column for each thing, and the role of the thing in each cell. The picture shows the course's own text in that way, turned on its side and without the roles; the worked example writes the text out as data. Each time a text names or points to a thing is a **mention** of it.
    """)
    return


@app.cell(hide_code=True)
def grid_picture():
    mo.Html(C.grid_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_order():
    mo.md(r"""
    ### Known first, new last

    A text keeps track of what the reader already has. Jurafsky and Martin, in their chapter on coreference: "The way referring expressions are used to evoke new referents into the discourse (introducing new information), or access old entities from the model (old information), is called their information status or information structure." *A storm* brings a storm into the text; *the storm*, a sentence later, picks it up. Some things are new and still expected. Their example: *I went to a superb restaurant yesterday. The chef had just opened it.* The chef is new, and a reader can infer that a restaurant has one.

    Susan Haviland and Herbert Clark (1974) measured what that costs. Readers saw pairs of sentences and pressed a button as soon as they understood the second one, *The beer was warm.* After *We got some beer out of the trunk.*, that took 835 milliseconds on average; after *We checked the picnic supplies.*, where the reader has to infer the beer, it took 1,016 milliseconds, 181 more. A second experiment showed that the repeated word alone did not explain it. The authors add that they measured how long it took to feel one had understood, and that "the absolute differences between conditions cannot be taken very seriously".

    Within a sentence, the order tends to run from known to new: **information structure**. Clark and Haviland (1977) on English: "In general, therefore, given comes before new." And: "Because given generally comes before new, English sentences also tend to begin with definite noun phrases, delaying indefinite noun phrases until later in the sentence." So *the* tends to come early, and *a* later. Gregory Ward and Betty Birner (2001) write that English, "like many other languages, shows a tendency" to put given before new. The Dutch reference grammar, the *Algemene Nederlandse Spraakkunst*, calls it the left-right principle (*links-rechts-principe*): what is informatively more important tends to come further back in the sentence, and sentences that deviate from it are not necessarily unusual. And the linguist Randy LaPolla (2019) points out that some languages put the new information at the start of the clause.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **discourse** | a coherent, structured group of sentences, such as a story, a report or a conversation |
        | **coherence** | how the sentences of a text hang together in meaning, so that a reader sees why each one follows |
        | **connective** | a word such as *because*, *so* or *but* that says how two parts of a text connect |
        | **cohesion** | the ties that show in the words: words that point back, connectives, repeated or related words |
        | **lexical cohesion** | ties by the same or related words in nearby sentences |
        | **mention** | each time a text names or points to a thing: *a storm* and *the storm* are two mentions of one storm |
        | **information structure** | how a sentence divides what the reader has already met (old) from what it brings in (new), and in what order |
        | `len(set(words))` | the number of different words in a list: the set keeps each word once |
        """),
        L.watch_out("*New* here means new in this text, not unknown to the reader. Ward and Birner give *the moon* in a sentence about a pretty night: new in the text when it first comes up, and known to everyone."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Split the text into sentences, and list for each sentence the words that carry content.
    2. Count the words that neighboring sentences share: where the count drops, the topic may change.
    3. List the things each sentence mentions, and mark each mention new or old: has the text named that thing before?
    4. Look for the relation between neighbors: a reason, evidence, a contrast? Is there a connective that says so?
    5. Look at the order: does a sentence start with something old, and end with something new?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell writes the course's own text, the eight sentences of the picture. `WORDS` holds, for each sentence, the words that carry content, in small letters and in their base form, chosen by hand; `MENTIONS` holds the things each sentence mentions, in order. In the third sentence, *it* is the bike again. The second cell defines two functions, tries them and looks at the first two sentences.
    """)
    return


@app.cell
def text_cell():
    TEXT = [
        "Kim wanted a cheap bike for the trip to work.", "The bike in the window was cheap.",
        "Kim bought the bike and rode it home.", "Close to home, the chain came off.",
        "Kim fixed the chain with a tool from the shed.", "That night, a storm hit the town.",
        "The storm blew the roof off the shed.", "The next morning, Kim rode past the roof lying in the street.",
    ]
    WORDS = [
        ["kim", "want", "cheap", "bike", "trip", "work"], ["bike", "window", "cheap"],
        ["kim", "buy", "bike", "ride", "home"], ["close", "home", "chain", "come"],
        ["kim", "fix", "chain", "tool", "shed"], ["night", "storm", "hit", "town"],
        ["storm", "blow", "roof", "shed"], ["next", "morning", "kim", "ride", "roof", "lie", "street"],
    ]
    MENTIONS = [
        ["Kim", "bike", "work"], ["bike", "window"], ["Kim", "bike", "bike", "home"], ["home", "chain"],
        ["Kim", "chain", "tool", "shed"], ["night", "storm", "town"], ["storm", "roof", "shed"],
        ["morning", "Kim", "roof", "street"],
    ]
    return MENTIONS, TEXT, WORDS


@app.cell
def example_look(MENTIONS, WORDS):
    def flat(text):
        out = []
        for sentence in text:
            for m in sentence:
                out.append(m)
        return out

    def mentions_of(thing, text):
        return [i for i in range(len(text)) if thing in text[i]]

    print(flat(MENTIONS)[:7])
    print(mentions_of("Kim", MENTIONS), mentions_of("storm", MENTIONS))
    print(WORDS[0])
    print(WORDS[1])
    print(len(set(["bike", "bike", "cheap"])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `flat` puts the mentions of all sentences in one list, in the order of the text: first the three of the first sentence, then those of the second. `mentions_of` gives the places of the sentences that mention a thing, counted from 0: Kim is in four of them, the storm in two in a row, as in the picture. The first two sentences share *bike* and *cheap*. The last line counts different words: the set keeps *bike* once, so it gives 2.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. overlap: the words neighbors share

        Write `overlap(a, b)`: give back how many different words the lists `a` and `b` have in common. A word that is twice in a list counts once. For `WORDS[0]` and `WORDS[1]` that is `2`: *bike* and *cheap*. Replace the `...`. The check calls `overlap(WORDS[0], WORDS[1])`, then cases of its own, with a word twice, with nothing in common and with an empty list.
        """),
        L.task("Count the different words two sentences share."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overlap():
    box_overlap = L.code_box("overlap", lines=3)
    box_overlap
    return (box_overlap,)


@app.cell(hide_code=True)
def check_overlap(box_overlap):
    result_overlap = L.check("overlap", box_overlap.value)
    result_overlap
    return (result_overlap,)


@app.cell(hide_code=True)
def hints_overlap():
    L.hints(
        "overlap",
        [
            "Collect in a list `both` the words the two lists have in common, with a list comprehension over `a`. Then count the different ones, as the last line of the worked example does.",
            "A list comprehension can keep only some values, with an `if` after the `for`. Which condition must a word of `a` meet to go into `both`?",
            "`    both = [w for w in a if ___]`",
        ],
        why="`both = [w for w in a if w in b]`, and then `return len(set(both))`. `overlap(WORDS[0], WORDS[1])` gives `2`.",
    )
    return


@app.cell(hide_code=True)
def text_after_overlap():
    mo.md(r"""
    The picture under the check gives the text with your count at each gap between two sentences, and the shared words next to it. The first sentences keep sharing a word: the bike, home, the chain. Where the storm comes in, the count drops to 0, and the last sentence picks up the roof again. A count of 1 or 2 is small; Hearst's program compared blocks of several sentences, not single ones.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_status():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. status: new or old

        Write `status(mentions)`: give back a list with, for each mention in `mentions`, `"new"` when the text names that thing for the first time, and `"old"` when it came before. For `flat(MENTIONS)` the list starts with `"new"`, `"new"`, `"new"`, `"old"`: the bike of the second sentence is old. Replace the `...`. The check calls `status(flat(MENTIONS))`, then cases of its own, with a thing named three times and with an empty list.
        """),
        L.task("Mark each mention in a text as new or old."),
    ])
    return


@app.cell(hide_code=True)
def exercise_status():
    box_status = L.code_box("status", lines=10)
    box_status
    return (box_status,)


@app.cell(hide_code=True)
def check_status(box_status):
    result_status = L.check("status", box_status.value)
    result_status
    return (result_status,)


@app.cell(hide_code=True)
def hints_status():
    L.hints(
        "status",
        [
            "Keep two lists: `seen` for the things the text has named so far, and `out` for the labels. Go over `mentions` with a `for` loop, and give back `out` after the loop.",
            "When is a mention old? And what has to happen to `seen` after a new one?",
            "`        if ___:`",
        ],
        why="`seen = []` and `out = []`; `for m in mentions:` with `if m in seen:` under it and `out.append(\"old\")` under that; under `else:`, `out.append(\"new\")` and `seen.append(m)`; after the loop `return out`. `status(flat(MENTIONS))` starts with `['new', 'new', 'new', 'old']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_status():
    mo.md(r"""
    The picture under the check shows each mention as new or old, in the text as written and in a shuffled order. As written, most sentences start with something old and end with something new, as Clark and Haviland describe. *That night* and *The next morning* are new, and set the time first. By this count *the chain* is new as well, though a reader expects a bike to have one: what Jurafsky and Martin call inferrable. Shuffled, more sentences start with something new, and *the storm* comes before *a storm*: the sentence takes a storm for granted that the reader has not met.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: shuffle the text

    The program of the guess compared texts with shuffled versions of them. Do the same with two simple counts: the words neighbors share, as in exercise 1, and the sentences that start with something old, as in exercise 2. Choose a shuffle, and compare it with the text as written.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_shuffle = mo.ui.slider(1, 12, step=1, value=1, label="Shuffle", show_value=True)
    demo_shuffle
    return (demo_shuffle,)


@app.cell(hide_code=True)
def demo_look(demo_shuffle):
    mo.Html(C.demo_picture(demo_shuffle.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The eight sentences can stand in 40,320 orders. The written order has more shared words between neighbors than 38,434 of them, and starts more sentences with something old than 29,316. Neither count is proof: some shuffles tie or win, and two sentences can share a word without making sense, as exercise 4 shows. Barzilay and Lapata's program learned from pairs of original and shuffled texts which patterns of mentions and roles go with an original order.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ties():
    mo.md(r"""
    ### Exercise 3. What ties the two sentences?

    Each bold pair below holds together by one kind of tie. Put each pair under its kind.
    """)
    return


@app.cell(hide_code=True)
def exercise_ties():
    sort_ties = L.sort("ties")
    sort_ties
    return (sort_ties,)


@app.cell(hide_code=True)
def check_ties(sort_ties):
    result_ties = L.check("ties", sort_ties)
    result_ties
    return (result_ties,)


@app.cell(hide_code=True)
def text_exercise_bank():
    mo.md(r"""
    ### Exercise 4. Shared, and still apart

    A program that counts shared words, as `overlap` does, sees a tie wherever a word comes back.
    """)
    return


@app.cell(hide_code=True)
def choice_bank_cell():
    choice_bank = L.choice(
        "bank",
        question="Which pair shares a word, but does not hang together?",
        options={
            "key": "*Sam lost the key to the shed. The key turned up under the doormat a week later.*",
            "right": "*Sam sat on the bank by the river. The bank raised its fees.*",
            "cake": "*Sam bought a cake for Kim. The cake was still warm from the oven.*",
        },
        correct="right",
        why={
            "right": "*Bank* comes back, so `overlap` counts a shared word. But the bank of the river and the bank that raises its fees are two different things, and nothing links the two sentences: a tie in the words, without coherence. Lexical cohesion is a hint of coherence, not a guarantee.",
            "key": "Here the key comes back as the same key, and the second sentence tells what happened to it: the pair hangs together. Look for a word that comes back with another meaning.",
            "cake": "Here the cake comes back as the same cake, and the second sentence says more about it: the pair hangs together. Look for a word that comes back with another meaning.",
        },
    )
    choice_bank
    return (choice_bank,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What is meant, not said](/course/open/nlp-pragmatics?lang=en) looked at what a speaker means beyond the words of one utterance. This lesson went beyond the single sentence: ties in the words (cohesion), a sense that connects the sentences (coherence), and an order within each sentence from old to new (information structure). You counted shared words, marked mentions new or old, sorted kinds of ties and found a pair that shares a word without hanging together. A later lesson follows all the mentions of one thing through a text.

    **An everyday example.** When a story starts with *a dog*, the next sentence says *the dog*, or *it*. Say *a dog* again, and your listener thinks of a second dog.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bank, result_overlap, result_status, result_ties):
    L.wrap_up(result_overlap, result_status, result_ties, choice_bank)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
