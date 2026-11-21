# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-meaning-and-truth. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-meaning-and-truth", "en")
    C = L.checks   # the course's world, the sentences, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_part():
    quick_part = L.choice(
        "quick-part",
        question="In *the late train left the city after midnight*, which group of words can make way for the single word *then*?",
        options={
            "right": "*after midnight*",
            "left": "*left the city after*",
            "train": "*train left*",
        },
        correct="right",
        why={
            "right": "*The late train left the city then*: *after midnight* acts as one unit, a constituent, here a prepositional phrase. This lesson gives such parts of a sentence a meaning of their own.",
            "left": "No single word can stand in for *left the city after*: the group cuts *after* off from *midnight*. It is not a constituent. See *Constituents* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
            "train": "*The late then the city after midnight* is not a sentence: *train left* is not a constituent. See *Constituents* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
        },
    )
    quick_part
    return (quick_part,)


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="The grammar of *Phrase structure and endless sentences* starts with the rule S → NP VP. What does that rule say?",
        options={
            "right": "S can be rewritten as NP and then VP",
            "or": "S can be rewritten as NP, or else as VP",
            "inside": "NP holds VP",
        },
        correct="right",
        why={
            "right": "The symbol on the left may be rewritten as the row on the right: a sentence is a noun phrase followed by a verb phrase. This lesson does the same for meaning: the meaning of S comes from the meaning of its NP and its VP.",
            "or": "A choice is written with a bar, as in NP → DET NOM | PROPN. S → NP VP has no bar: the right-hand side is one row, NP and then VP. See *Rules that build sentences* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
            "inside": "The rule rewrites S, the symbol on the left, as the row on the right: NP and VP sit side by side under S. See *Rules that build sentences* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In *Phrase structure and endless sentences*, a grammar said which rows of words are sentences, and gave each sentence a tree. It said nothing about what a sentence means. A program that reads a menu to choose a dish, or follows a recipe, needs more: Jurafsky and Martin open their appendix on the meaning of sentences with such tasks, which call for "representations that link the linguistic elements to the necessary non-linguistic knowledge of the world".

    One old answer to *what does a sentence mean?* starts from truth. The logician Alfred Tarski (1944) gave the best-known example: "The sentence 'snow is white' is true if, and only if, snow is white." The philosopher Donald Davidson (1967) built on it: "to give truth conditions is a way of giving the meaning of a sentence." If you know what the world has to be like for a sentence to be true, you know a good part of what it means.

    **An everyday example.** *The shop is open on Sunday* is true or false, depending on the shop and the Sunday. You do not need to know which it is to understand the sentence: you know what to check.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Brenden Lake and Marco Baroni (2018) trained neural networks to turn commands such as *walk*, *jump* and *walk twice* into actions. After training on a random 80 in 100 of the commands, the best network got 99.8 in 100 of the other commands right. Then they showed *jump* only on its own during training, while the other actions also came in longer commands, and tested all longer commands with *jump* in them, such as *jump twice*. About how many in 100 did the best network get right?",
        options={
            "most": "About 95 in 100: nearly as many as before",
            "half": "About 61 in 100",
            "right": "About 1 in 100",
        },
        reveal="**About 1 in 100**: 1.2 for the best network, and 0.08 for the kind of network that did best over all their tests. The networks knew *jump* and knew *twice*, but did not put the two together the way the meaning of *jump twice* is built from its words. In the same setup with *turn left* in the place of *jump*, the best network got 90.3 in 100. With 8 longer commands with *jump* in training, the best network for *jump* got 38.3; with 16 and 32, 77.8 and 88.4.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A small world

    To check sentences, you need a world to check them against. Take the small world in the picture: four animals, Tom and Mia are cats, Rex is a dog and Pip a mouse. Mia and Rex sleep. Tom chases Pip, and Rex chases Tom.

    Jurafsky and Martin call such a world a **model**: "A model is a formal construct that stands for the particular state of affairs in the world." Its things are the domain, "the set of objects that are being represented". The words get their meaning from it, in three kinds:

    - a name stands for one thing: *Mia* for Mia;
    - a noun or a verb without object stands for a list of things: *cat* for Tom and Mia, *sleeps* for Mia and Rex;
    - a verb with an object stands for a list of pairs: *chases* for (Tom, Pip) and (Rex, Tom), with the doer, the one who chases, first.

    In the book, a property stands for a set of things and a relation for a set of tuples, rows of things such as these pairs. This lesson uses lists.
    """)
    return


@app.cell(hide_code=True)
def world_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_truth():
    mo.md(r"""
    ### True in a world

    *Mia sleeps* is true in this world, since Mia is in the list of *sleeps*. *Tom sleeps* is false: Tom is not there. Tarski's *snow is white* works the same way: the sentence is true if, and only if, the world is as the sentence says. These are the **truth conditions** of a sentence: what has to hold in a world for it to be true.

    The truth conditions stay the same from one world to the next; whether they hold changes. In a world in which Tom sleeps too, *Tom sleeps* is true. You can understand a sentence without knowing whether it is true: you know what to look for.

    ### Every and some

    *Every cat sleeps* is about two lists: the cats and the sleepers. It is true when each cat is among the sleepers. *Some cat sleeps* is true when at least one cat is. Jurafsky and Martin put it the same way for the two **quantifiers** of logic, written ∀ (*every*) and ∃ (*some*): "To satisfy an existentially quantified variable, at least one substitution must result in a true sentence. To satisfy a universally quantified variable, all substitutions must result in true sentences." A variable is an open place, such as the x in *x sleeps*; a substitution puts one thing of the world in that place.

    In this world, *some cat sleeps* is true (Mia) and *every cat sleeps* is false (Tom is awake). A way to check both: make the list of the cats that sleep, and look at its length.

    ### The whole from its parts

    How does a program get from the words to true or false? Through the tree. *Rex chases Tom* has a VP, *chases Tom*, and that VP has a meaning of its own: the list of the animals that chase Tom. Here that is Rex alone. The sentence is true when its subject, Rex, is in that list. Each part gets its meaning from its own parts, from the bottom of the tree to the top.

    This is the principle of **compositionality**, in a standard wording by the linguist Barbara Partee (1984): "The meaning of a compound expression is a function of the meanings of its parts and of the way they are syntactically combined." The Stanford Encyclopedia of Philosophy calls it "a fundamental presupposition of most contemporary work in semantics". It is often called Frege's principle, after the logician Gottlob Frege. The philosopher Max Cresswell, who gave it that name in 1973, wrote that the name "must not be taken to imply that the principle is explicitly stated in Frege"; whether Frege believed it is still debated. The logician Richard Montague built the meanings of English sentences this way around 1970, rule by rule along with their syntax: "There is in my opinion no important theoretical difference between natural languages and the artificial languages of logicians". Montague considered it possible to describe the syntax and the semantics of both kinds of language in one precise theory.
    """)
    return


@app.cell(hide_code=True)
def parts_cell():
    mo.Html(C.parts_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **semantics** | the study of meaning, of words and of sentences |
        | **model** | in logic: a small world to check sentences against, with its things and what holds of them |
        | **truth conditions** | what has to hold in a world for a sentence to be true |
        | **quantifier** | a word such as *every* or *some* that says for how many things something holds; in logic ∀ and ∃ |
        | **compositionality** | the meaning of a whole comes from the meanings of its parts and the way they are combined |
        | `"Mia" in group` | `True` when `"Mia"` is one of the items of the list `group`, and `False` when it is not |
        | `("Rex", "Tom")` | a pair, a tuple of two: Rex chases Tom; the one who chases comes first |
        | `["S", ["NP", "Mia"], ["VP", "sleeps"]]` | a tree written as a list, new in this lesson: the symbol first, then its parts; a word is a string, without its tag (PROPN, VERB) |
        | `x[2][0]` | the item at place 0 of the item at place 2: one step down the tree for each pair of brackets |
        """),
        L.watch_out("A model in this lesson is a world to check sentences against, as in logic. It is not a model that learns from data, as elsewhere in this course. And *some* means *at least one*, as in logic: *some cat sleeps* is also true when every cat sleeps."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down the world: a list for each noun and each verb without object, and a list of pairs for each verb with an object.
    2. Give each word its meaning: a name its thing, a noun or a verb its list.
    3. Work up the tree: a verb with its object gives the list of its doers, for *chases Tom* the ones that chase Tom; a subject with its VP gives true or false.
    4. For *every* and *some*, make the list of the things of the noun that are also in the list of the VP: *some* needs one of them, *every* all of them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell writes the world of the picture as a dict: for each word, its list. The second defines `some`: it makes the list `inside` of the things that are also in `group`, and is true when that list is not empty. Then it checks four sentences.
    """)
    return


@app.cell
def world_cell():
    WORLD = {
        "cat": ["Tom", "Mia"],
        "dog": ["Rex"],
        "mouse": ["Pip"],
        "sleeps": ["Mia", "Rex"],
        "chases": [("Tom", "Pip"), ("Rex", "Tom")],
    }
    return (WORLD,)


@app.cell
def example_look(WORLD):
    def some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0

    print("Mia" in WORLD["sleeps"])              # Mia sleeps
    print("Tom" in WORLD["sleeps"])              # Tom sleeps
    print(some(WORLD["cat"], WORLD["sleeps"]))   # some cat sleeps
    print(some(WORLD["dog"], WORLD["cat"]))      # some dog is a cat
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Mia is in the list of sleepers, Tom is not: true, then false. For *some cat sleeps*, `inside` holds Mia, so it is not empty: true. *Some dog is a cat* uses the list of cats in the place of the VP: no dog is in it, `inside` stays empty, and the sentence is false. The words became lists, and the lists decided. Exercise 1 does the same for *every*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_every():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. every: all of them

        Write `every(things, group)`: it gives `True` when every thing in `things` is also in `group`, and `False` otherwise. *Every cat sleeps* is then `every(WORLD["cat"], WORLD["sleeps"])`. Replace the `...`. The check calls `every(["Tom", "Mia"], ["Mia", "Rex"])`, then cases of its own, one of them with an empty first list.
        """),
        L.task("Decide whether all the things of the first list are in the second."),
    ])
    return


@app.cell(hide_code=True)
def exercise_every():
    box_every = L.code_box("every", lines=3)
    box_every
    return (box_every,)


@app.cell(hide_code=True)
def check_every(box_every):
    result_every = L.check("every", box_every.value)
    result_every
    return (result_every,)


@app.cell(hide_code=True)
def hints_every():
    L.hints(
        "every",
        [
            "Start as `some` does: make the list `inside` of the things that are also in `group`. Then compare two lengths.",
            "*Every cat sleeps*: how many of the cats must be in the list of sleepers?",
            "`    return len(inside) == ___`",
        ],
        why="`inside = [thing for thing in things if thing in group]`, then `return len(inside) == len(things)`. `every([\"Tom\", \"Mia\"], [\"Mia\", \"Rex\"])` gives `False`: only Mia is in `inside`, and there are two cats.",
    )
    return


@app.cell(hide_code=True)
def text_after_every():
    mo.md(r"""
    The picture under the check runs your function on four sentences about the world, one of them *every unicorn sleeps*. There are no unicorns, so the picture gives your function `[]` as the list of unicorns. Then `inside` is empty too: 0 equals 0, and the sentence is true. That surprises many people, but it is how logic counts: *every unicorn sleeps* only fails if there is a unicorn that does not sleep. Jurafsky and Martin show the same with *all vegetarian restaurants serve vegetarian food*. In logic it reads: for every x, if x is a vegetarian restaurant, then x serves vegetarian food, with an arrow ⇒ for *if ... then*. For a restaurant that is not vegetarian, the *if* part is false, and then the line counts as true. They add that this arrow "is only loosely based on any common-sense notions of implication or causation".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doers():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. doers: the meaning of a VP

        A verb with an object, such as *chases Tom*, stands for a list of things: the ones that chase Tom, its doers. Write `doers(pairs, target)`: give back the doers for `target`, in the order of `pairs`: with the pairs of *chases*, the things that chase `target`. *Chases Tom* is then `doers(WORLD["chases"], "Tom")`, and that gives `["Rex"]`. Replace the `...`. The check calls `doers([("Tom", "Pip"), ("Rex", "Tom")], "Tom")`, then cases of its own.
        """),
        L.task("Find the doers for the target."),
    ])
    return


@app.cell(hide_code=True)
def exercise_doers():
    box_doers = L.code_box("doers", lines=6)
    box_doers
    return (box_doers,)


@app.cell(hide_code=True)
def check_doers(box_doers):
    result_doers = L.check("doers", box_doers.value)
    result_doers
    return (result_doers,)


@app.cell(hide_code=True)
def hints_doers():
    L.hints(
        "doers",
        [
            "Start with an empty list `found`. Go over the pairs with `for pair in pairs:`, and when the second item of a pair is `target`, append the doer.",
            "In `(\"Rex\", \"Tom\")`, which item is the one who chases, and which the one who is chased?",
            "`            found.append(___)`",
        ],
        why="`found = []`; `for pair in pairs:` and under it `if pair[1] == target:` with `found.append(pair[0])`; after the loop `return found`. `doers([(\"Tom\", \"Pip\"), (\"Rex\", \"Tom\")], \"Tom\")` gives `['Rex']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_doers():
    mo.md(r"""
    The picture under the check asks your function who chases each animal, and marks in bold the ones it finds for *chases Tom*. The verb alone stands for pairs; with its object, the VP stands for a list of things, the same kind of meaning as *sleeps*. So a sentence with a VP of either kind can be checked in the same way: is the subject in the list? Exercise 3 puts that together.
    """)
    return


@app.cell(hide_code=True)
def text_sentences():
    mo.md(r"""
    ### Exercise 3. is_true: from the tree to true or false

    The cell below holds five sentences as trees, and a second world, `NIGHT`, with the same animals: at night, Tom, Mia and Rex sleep, and Pip chases Tom.
    """)
    return


@app.cell
def sentences_cell():
    SENTENCES = [
        ["S", ["NP", "Mia"], ["VP", "sleeps"]],
        ["S", ["NP", "Tom"], ["VP", "sleeps"]],
        ["S", ["NP", "Rex"], ["VP", "chases", ["NP", "Tom"]]],
        ["S", ["NP", "Tom"], ["VP", "chases", ["NP", "Rex"]]],
        ["S", ["NP", "Pip"], ["VP", "chases", ["NP", "Tom"]]],
    ]
    NIGHT = {
        "cat": ["Tom", "Mia"],
        "dog": ["Rex"],
        "mouse": ["Pip"],
        "sleeps": ["Tom", "Mia", "Rex"],
        "chases": [("Pip", "Tom")],
    }
    return


@app.cell(hide_code=True)
def text_exercise_true():
    mo.vstack([
        mo.md(r"""
        Write `is_true(tree, world)`: give back whether the sentence `tree` is true in `world`. Find the list the VP stands for: for a VP without object, the list of its verb in `world`; for a VP with an object, the list of the doers for its object, from `doers`. Tell the two kinds of VP apart by their length. Then give back whether the name of the subject is in that list. `doers` is ready for your function: the check gives it the course's own. Replace the `...`. The check calls `is_true(SENTENCES[0], WORLD)`, then all five sentences in `WORLD` and in `NIGHT`.
        """),
        L.task("Build the truth of the sentence from the meanings of its parts."),
    ])
    return


@app.cell(hide_code=True)
def exercise_true():
    box_true = L.code_box("is_true", lines=7)
    box_true
    return (box_true,)


@app.cell(hide_code=True)
def check_true(box_true):
    result_true = L.check("is_true", box_true.value)
    result_true
    return (result_true,)


@app.cell(hide_code=True)
def hints_true():
    L.hints(
        "is_true",
        [
            "Take `vp = tree[2]`. With `if len(vp) == 2:` the list is `world[vp[1]]`; otherwise call `doers` with the pairs of the verb and the name inside the object. Keep the list in `group`.",
            "In `SENTENCES[2]`, at which places are the VP, the verb, the object and the name inside the object?",
            "`        group = doers(world[vp[1]], ___)`",
        ],
        why="`vp = tree[2]`; `if len(vp) == 2:` then `group = world[vp[1]]`, `else:` then `group = doers(world[vp[1]], vp[2][1])`; at the end `return tree[1][1] in group`. `is_true(SENTENCES[2], WORLD)` gives `True`, and `is_true(SENTENCES[2], NIGHT)` gives `False`.",
    )
    return


@app.cell(hide_code=True)
def text_after_true():
    mo.md(r"""
    The picture under the check puts your answers for the five sentences in both worlds side by side. The sentences did not change, nor did your function; only the world did. *Tom sleeps* is false in `WORLD` and true in `NIGHT`, and *Rex chases Tom* the other way round. That is what truth conditions are: a sentence says what has to hold, and each world answers.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: 64 worlds

    Switch the six facts on and off, and see which sentences are true in your world. Then pick a sentence: the grid shows all 64 worlds you can make with the six switches, filled in where the sentence is true. Compare *Mia sleeps* with *Rex chases Tom*: both are true in 32 worlds, but not in the same ones. And find two sentences that are true in exactly the same worlds: with only these four animals, the two say the same.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_facts = mo.ui.array([mo.ui.checkbox(value=on, label=text) for text, on in C.demo_facts(L.lang)])
    demo_sentence = mo.ui.dropdown(options=C.demo_options(L.lang), value=C.DEMO_LABELS[L.lang][2], label="Sentence")
    mo.vstack([demo_facts.hstack(wrap=True, justify="start", gap=1), demo_sentence])
    return demo_facts, demo_sentence


@app.cell(hide_code=True)
def demo_look(demo_facts, demo_sentence):
    mo.Html(C.demo_picture(demo_facts.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_idiom():
    mo.md(r"""
    ### Exercise 4. Where the parts do not add up

    Compositionality works for most sentences, but perhaps not for all. The Stanford Encyclopedia of Philosophy gives *Mary kicked the bucket*, which can mean that Mary died: "There is no obvious way to derive this meaning from the meanings of the parts." Idioms "seem to be isolated exceptions to compositionality", it says, and it reports an estimate by the linguist Ray Jackendoff (1997) of about twenty-five thousand idioms in English.
    """)
    return


@app.cell(hide_code=True)
def choice_idiom_cell():
    choice_idiom = L.choice(
        "idiom",
        question="For which of these sentences does the meaning not follow from the meanings of its words and the way they are combined?",
        options={
            "right": "*It is raining cats and dogs.*",
            "rain": "*The cats and dogs are in the rain.*",
            "every": "*Every cat sleeps when it rains.*",
            "chase": "*A dog that chases a cat runs out into the rain.*",
        },
        correct="right",
        why={
            "right": "An idiom: it means that it rains hard, and no cat or dog comes into it. A program that builds the meaning from the parts would check the world for falling animals. So a dictionary lists such a phrase as a whole, with its own meaning.",
            "rain": "Here the parts do add up: *the cats and dogs* stands for the animals, *in the rain* for where they are, and the sentence is true when they are there. Look for a sentence whose meaning has nothing to do with its words.",
            "every": "This one is built from its parts, as in this lesson: the cats, the sleepers, and *every*, with *when it rains* saying at which times. Look for a sentence whose meaning has nothing to do with its words.",
            "chase": "A long sentence, but built from its parts: *a dog that chases a cat* is a list of dogs, made from *dog* and *chases a cat*, and the rest says what that dog does. Look for a sentence whose meaning has nothing to do with its words.",
        },
    )
    choice_idiom
    return (choice_idiom,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en) gave a sentence its tree. This lesson gave each part of that tree a meaning in a small world: a name a thing, a noun or a verb a list, a verb with its object the list of its doers. You wrote *every*, found the meaning of a VP, and checked sentences from their trees in two worlds. The next lesson shows what this leaves out: in logic, *some cats sleep* is true when every cat sleeps, yet a listener hears *not all*.

    **An everyday example.** A spreadsheet answers *every invoice is paid* the same way: take the invoices, take the paid ones, and compare. Seen this way, a query on a database is a sentence checked against a world.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_idiom, result_doers, result_every, result_true):
    L.wrap_up(result_every, result_doers, result_true, choice_idiom)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
