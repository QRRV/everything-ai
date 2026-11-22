# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-pragmatics. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pragmatics", "en")
    C = L.checks   # the course's scales and world, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_some():
    quick_some = L.choice(
        "quick-some",
        question="In logic, is *some cat sleeps* true in a world in which every cat sleeps?",
        options={
            "not-every": "No: *some* means not every cat",
            "right": "Yes: at least one cat sleeps",
            "no-value": "No: the sentence has no truth value then",
        },
        correct="right",
        why={
            "right": "In logic, *some* means at least one, and when every cat sleeps, at least one does. This lesson looks at why a listener still hears *not every cat* in it.",
            "not-every": "That is what many listeners hear, but not what logic says: *some cat sleeps* is true when at least one cat is among the sleepers, also when all are. See *Every and some* in [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
            "no-value": "In a world with cats, every sentence of that lesson is true or false. Here at least one cat is among the sleepers, so the sentence is true. See *Every and some* in [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
        },
    )
    quick_some
    return (quick_some,)


@app.cell(hide_code=True)
def quick_check_vp():
    quick_vp = L.choice(
        "quick-vp",
        question="In the world of *Meaning: when is a sentence true?*, what does the VP *chases Tom* stand for?",
        options={
            "pairs": "The pairs of *chases* with Tom first",
            "truth": "True when Tom chases someone, and false otherwise",
            "right": "The animals that chase Tom",
        },
        correct="right",
        why={
            "right": "A verb with its object stands for a list of things, its doers: here Rex alone. A sentence with that VP is true when its subject is in the list. This lesson keeps that world, and asks what a speaker means beyond it.",
            "pairs": "The verb *chases* alone stands for pairs. With its object, the VP stands for a list of things: the ones that chase Tom, with Tom second in their pair. See *The whole from its parts* in [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
            "truth": "True or false is what a whole sentence gives, with its subject. The VP alone stands for a list of things: the ones that chase Tom. See *The whole from its parts* in [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en).",
        },
    )
    quick_vp
    return (quick_vp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Meaning: when is a sentence true?* checked what a sentence says against a world. People mean more than that. The philosophers Kepa Korta and John Perry open their encyclopedia entry on pragmatics with it: "more is involved in what one communicates than what one literally says; more is involved in what one means than the standard, conventional meaning of the words one uses."

    Jurafsky and Martin show what that asks of a program. A travel agent asks *what day in May did you want to travel?*, and the client answers *I need to be there for a meeting that's from the 12th to the 15th*. "Notice that the client does not in fact answer the agent's question." Jurafsky and Martin explain how the agent can still infer a flight on the 11th. A program that only checks what the words say has no date to book.

    **An everyday example.** You hold up an empty cup and ask *is there still coffee?* Taken literally, that is a question about coffee, and *yes* would answer it. Anyone who answers *yes* and does nothing has understood the words and missed what you meant.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Laura Ruis and colleagues (2023) tested large language models on 600 short exchanges such as this one. Esther asks *Can you come to my party on Friday?* and Juan answers *I have to work*. Does Juan mean yes or no? Guessing at random gets 50 in 100 right. About how many in 100 did models get right that were only trained on large amounts of text, without further training to follow instructions and without examples in the prompt?",
        options={
            "right": "About 60 in 100",
            "people": "About 85 in 100: close to people",
            "more": "About 95 in 100: more than people",
        },
        reveal="**About 60 in 100**: \"All pre-trained models obtain close to random zero-shot accuracy (around 60%), whereas humans obtain 86%\", the authors write; 86.2 on average. These models counted as right when they gave the answer that makes sense a higher probability than the one that does not: \"a more lenient evaluation protocol than sometimes used for language models\", in the authors' words. The best model they tested, GPT-4, had to write its answer itself, since they could not see its probabilities: it got 82.3 in 100 with one example in the prompt, and 86.5 with five examples in the prompt that reasoned step by step.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Saying is doing

    The sentences of *Meaning: when is a sentence true?* described a world, and were true or false. The philosopher J. L. Austin (1962) pointed to utterances that do something instead. Someone who says *I name this ship the Queen Elizabeth* while smashing a bottle against its stern is not describing a naming: in the right circumstances, saying the words is the naming. Austin called such utterances performative, and wrote of the examples: "None of the utterances cited is either true or false". In the same lectures, on the reading of the Stanford Encyclopedia of Philosophy, Austin set out to show that no simple line between describing and doing can be drawn. Jurafsky and Martin call the idea behind it a key insight, first due to the philosopher Ludwig Wittgenstein (1953) and worked out by Austin: "each utterance in a dialogue is a kind of action being performed by the speaker". Such an act is a **speech act**: stating, asking, requesting, promising, betting.

    One act can also go by way of another. *Can you pass the salt?* asks whether you are able to, but a listener hears a request. In the words of the encyclopedia: "my question whether you can pass the salt is normally taken as a request that you do so." It calls such cases indirect speech acts, after the philosopher John Searle (1979). The psychologist Herbert Clark (1979) called *Can you pass the salt?* and *Could you pass the salt?* "highly conventional, or idiomatic, forms in English for requesting the salt". Speech acts are part of **pragmatics**: the study of what speakers mean and do with words in a context.

    ### Cooperating

    How does a listener get from the words to what is meant? The philosopher Paul Grice (1975) proposed "a rough general principle which participants will be expected (ceteris paribus) to observe", the Cooperative Principle: "Make your conversational contribution such as is required, at the stage at which it occurs, by the accepted purpose or direction of the talk exchange in which you are engaged." Under it fall four kinds of maxims, rules of thumb:

    - **Quantity**: "Make your contribution as informative as is required (for the current purposes of the exchange)", and "Do not make your contribution more informative than is required";
    - **Quality**: "Try to make your contribution one that is true";
    - **Relation**: "Be relevant.";
    - **Manner**: "Be perspicuous": avoid obscurity and ambiguity, be brief and orderly.

    Grice called the second maxim of Quantity itself "disputable". A listener assumes that the speaker keeps to the maxims, and works out what the speaker must mean for that to hold. The client of the travel agent named the dates of a meeting, not a day to travel. If that answer is relevant, the meeting must decide the trip, and since people like to arrive the day before a meeting, the agent can work out the 11th. What a speaker suggests in this way, beyond what was said, and a listener works out, is an **implicature**.
    """)
    return


@app.cell(hide_code=True)
def text_idea_scale():
    mo.md(r"""
    ### Some, but not all

    *Some cats sleep* is true in logic when at least one cat sleeps, also when all do: that is how you counted in *Meaning: when is a sentence true?*. Yet a listener who hears it takes it that not all cats sleep. The maxim of Quantity explains it: a speaker who knew that all cats sleep, and could say so, should have used the more informative *all*.

    The Stanford Encyclopedia of Philosophy reports that the linguist Laurence Horn (1972, 1989) "observed that the quantifiers all, most, many, some form a scale", strongest first, and gives the rule: "the result of substituting one term implicates the denial of the result of substituting any term to the left, but not to the right." So *most* suggests *not all*, and *some* suggests *not many*, *not most* and *not all*. The words to the right stay true: when most cats sleep, some do.

    An implicature is not what the sentence says, and that shows: the speaker can take it back. The encyclopedia gives *Some athletes smoke*, followed by "Indeed, all do". The second sentence does not contradict the first; it only cancels what a listener would have taken from it.
    """)
    return


@app.cell(hide_code=True)
def scale_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_presupposition():
    mo.md(r"""
    ### Taken for granted

    *Tom has stopped chasing Pip* says that Tom does not chase Pip now. It also takes for granted that Tom chased Pip before: a listener who did not know learns it in passing. What a sentence takes for granted in this way is a **presupposition**. Words such as *stop*, *again* and *the* bring one along; the Stanford Encyclopedia of Philosophy gives *China has stopped stockpiling metals*, which takes for granted that China used to stockpile them.

    A test tells a presupposition apart from what a sentence says: add *not*. *Tom has not stopped chasing Pip* says the opposite about now, and still takes for granted that Tom chased Pip before. The encyclopedia calls this the negation test, the standard way to spot a presupposition. Other places work as tests too, such as a question or after *if*, but presuppositions do not always survive there, nor always under *not*; when they do is still much studied.
    """)
    return


@app.cell(hide_code=True)
def presupposition_picture():
    mo.Html(C.presup_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_the():
    mo.md(r"""
    *The* takes for granted that there is exactly one such thing: *the cat that sleeps* refers to the one cat that sleeps. But what about *the cat that sleeps chases Tom* in a world in which no cat sleeps? The philosopher Bertrand Russell (1905) held that such a sentence is false. Of *the present King of France is bald*, written when France had no king, Russell wrote that it "is certainly false". The philosopher Peter Strawson (1950) argued, the encyclopedia reports, that when such a phrase refers to nothing, the sentence can lack a truth value, and be neither true nor false. In code, that nothing can be `None`.

    In a real conversation, the context often picks out one thing: *the cat* is the cat of the house, even in a world with many cats. This lesson leaves the context out, and counts in its small world.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **pragmatics** | the study of what speakers mean and do with words in a context, beyond what the words say |
        | **speech act** | an utterance seen as an act, such as stating, asking, requesting, promising or betting |
        | **implicature** | what a speaker suggests beyond what was said, and a listener works out, such as *not all* after *some* |
        | **presupposition** | what a sentence takes for granted, such as that Tom chased Pip before, in *Tom has stopped chasing Pip* |
        | **scale** | here: words in a row from strong to weak, such as *all*, *most*, *many*, *some* |
        | `return None` | give back `None`: no value at all, here for a phrase that refers to nothing |
        """),
        L.watch_out("*Some* still means *at least one*, as in logic. The *not all* is an implicature on top of it: the speaker suggests it, and a listener works it out. Code that checks what a sentence says, such as `some` in *Meaning: when is a sentence true?*, does not see it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Ask what the speaker does with the utterance: state, ask, request, promise.
    2. Find what the sentence says: its truth conditions in the world.
    3. Ask what a cooperative speaker could have said instead. Stronger words on a scale that were not said, a listener takes to be false.
    4. Look for words that take something for granted, such as *stop*, *again* and *the*. What they take for granted usually stays when you add *not*.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell writes three scales as lists, strongest word first: Horn's scale and two of the course's own. The second writes the world of *Meaning: when is a sentence true?* and its night. The third defines two functions: `says_some` is the `some` of that lesson, what the sentence says; `means_some` adds what a listener takes from it, *but not all*. Then it tries both on the cats in the two worlds.
    """)
    return


@app.cell
def scales_cell():
    SCALE = ["all", "most", "many", "some"]
    HEAT = ["hot", "warm"]
    OFTEN = ["always", "often", "sometimes"]
    return (SCALE,)


@app.cell
def world_cell():
    WORLD = {
        "cat": ["Tom", "Mia"],
        "dog": ["Rex"],
        "mouse": ["Pip"],
        "sleeps": ["Mia", "Rex"],
        "chases": [("Tom", "Pip"), ("Rex", "Tom")],
    }
    NIGHT = {
        "cat": ["Tom", "Mia"],
        "dog": ["Rex"],
        "mouse": ["Pip"],
        "sleeps": ["Tom", "Mia", "Rex"],
        "chases": [("Pip", "Tom")],
    }
    return NIGHT, WORLD


@app.cell
def example_look(NIGHT, SCALE, WORLD):
    def says_some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0

    def means_some(things, group):
        inside = [thing for thing in things if thing in group]
        return len(inside) > 0 and len(inside) < len(things)

    # some cats sleep: in WORLD, then at night
    print(says_some(WORLD["cat"], WORLD["sleeps"]), means_some(WORLD["cat"], WORLD["sleeps"]))
    print(says_some(NIGHT["cat"], NIGHT["sleeps"]), means_some(NIGHT["cat"], NIGHT["sleeps"]))
    print(SCALE[0], SCALE[1:3])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In `WORLD`, one of the two cats sleeps: *some cats sleep* is true, and so is what a listener takes from it. In `NIGHT`, both cats sleep. The sentence is still true in logic, but a listener who hears it takes it that not all cats sleep, and that is false. A speaker who keeps to Quantity says *all cats sleep* there. The last line looks at the scale: place 0 holds the strongest word, and a slice takes a row of words from it. Exercise 1 finds, for a word on a scale, the stronger words that a listener takes to be false.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_implicated():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. implicated: the stronger words

        Write `implicated(scale, word)`: give back the words of `scale` that are stronger than `word`, in the order of the scale. For *some*, `implicated(SCALE, "some")` then gives `["all", "most", "many"]`: a listener takes *some* to mean not all, not most and not many. For a word that is not on the scale, give back an empty list. Replace the `...`. The check calls `implicated(SCALE, SCALE[3])`, then cases of its own with the three scales, among them `implicated(SCALE, HEAT[1])`.
        """),
        L.task("Find the words on the scale that a listener takes to be false."),
    ])
    return


@app.cell(hide_code=True)
def exercise_implicated():
    box_implicated = L.code_box("implicated", lines=5)
    box_implicated
    return (box_implicated,)


@app.cell(hide_code=True)
def check_implicated(box_implicated):
    result_implicated = L.check("implicated", box_implicated.value)
    result_implicated
    return (result_implicated,)


@app.cell(hide_code=True)
def hints_implicated():
    L.hints(
        "implicated",
        [
            "Go over the places of the scale with `for i in range(len(scale)):`, and look for the place where `scale[i]` is `word`. After the loop, give back an empty list.",
            "The scale starts with the strongest word. At which places are the words that are stronger than the one at place `i`?",
            "`            return ___`",
        ],
        why="`for i in range(len(scale)):` and under it `if scale[i] == word:` with `return scale[:i]`; after the loop `return []`. `implicated(SCALE, SCALE[3])` gives `['all', 'most', 'many']`, and `implicated(SCALE, SCALE[0])` gives `[]`: nothing is stronger than *all*.",
    )
    return


@app.cell(hide_code=True)
def text_after_implicated():
    mo.md(r"""
    The picture under the check asks your function about each word of Horn's scale. The weaker the word, the more a listener takes to be false; for *all*, nothing. A word that is not on the scale says nothing about it: *warm* suggests *not hot*, and nothing about how many cats sleep.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: six cats

    Choose how many of the six cats sleep, and see what each sentence says in logic and what a listener takes it to mean. Then pick a sentence: its two rows of squares show, for 0 to 6 sleeping cats, where it is true in logic and where it is true the way a listener takes it. Find the counts at which *some cats sleep* is true in logic but not the way a listener takes it, and see which sentence a speaker who keeps to Quantity says there.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_count = mo.ui.slider(0, 6, value=6, label="Sleeping cats", show_value=True)
    demo_sentence = mo.ui.dropdown(options=C.demo_options(L.lang), value=C.DEMO_LABELS[L.lang][2], label="Sentence")
    mo.vstack([demo_count, demo_sentence])
    return demo_count, demo_sentence


@app.cell(hide_code=True)
def demo_look(demo_count, demo_sentence):
    mo.Html(C.demo_picture(demo_count.value, demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_the_one():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. the_one: what *the* refers to

        Write `the_one(things, group)`: give back the one thing of `things` that is also in `group`, and `None` when there is not exactly one. *The cat that sleeps* is then `the_one(WORLD["cat"], WORLD["sleeps"])`, and that gives `"Mia"`. *The cat* alone is `the_one(WORLD["cat"], ["Tom", "Mia", "Rex", "Pip"])`, with all the animals as `group`, and that gives `None`: there are two cats. Replace the `...`. The check calls `the_one(["Tom", "Mia"], ["Mia", "Rex"])`, then cases of its own, with one such thing, with two and with none.
        """),
        L.task("Find the one thing a phrase with *the* refers to, or `None`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_the_one():
    box_the_one = L.code_box("the_one", lines=5)
    box_the_one
    return (box_the_one,)


@app.cell(hide_code=True)
def check_the_one(box_the_one):
    result_the_one = L.check("the_one", box_the_one.value)
    result_the_one
    return (result_the_one,)


@app.cell(hide_code=True)
def hints_the_one():
    L.hints(
        "the_one",
        [
            "Make the list `inside` as `says_some` does in the worked example. Check its length with `if len(inside) == 1:`; otherwise give back `None`.",
            "When `inside` holds exactly one thing, at which place in `inside` is that thing?",
            "`        return ___`",
        ],
        why="`inside = [thing for thing in things if thing in group]`; `if len(inside) == 1:` with `return inside[0]` under it; at the end `return None`. `the_one([\"Tom\", \"Mia\"], [\"Mia\", \"Rex\"])` gives `'Mia'`.",
    )
    return


@app.cell(hide_code=True)
def text_after_the_one():
    mo.md(r"""
    The picture under the check asks your function what five phrases refer to in the world, and puts each in a sentence: *... chases Tom*. Where a phrase refers to nothing, your function gives `None`, and the picture gives the sentence no truth value: Strawson's answer. Russell would call each of those sentences false. A program has to pick one of the two.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_acts():
    mo.md(r"""
    ### Exercise 3. What does the speaker do?

    Put each bold sentence under what the speaker does with it. Look past the form of the sentence: a question can do more than ask.
    """)
    return


@app.cell(hide_code=True)
def exercise_acts():
    sort_acts = L.sort("acts")
    sort_acts
    return (sort_acts,)


@app.cell(hide_code=True)
def check_acts(sort_acts):
    result_acts = L.check("acts", sort_acts)
    result_acts
    return (result_acts,)


@app.cell(hide_code=True)
def text_exercise_cancel():
    mo.md(r"""
    ### Exercise 4. Taking it back

    An implicature can be taken back without a contradiction; what a sentence says cannot. A presupposition is in between: it usually stays under *not*, and denying it outright makes a sentence clash with itself.
    """)
    return


@app.cell(hide_code=True)
def choice_cancel_cell():
    choice_cancel = L.choice(
        "cancel",
        question="Which of these sentences takes back an implicature, and so still makes sense?",
        options={
            "right": "*Some of the cakes are gone; in fact, all of them are.*",
            "stopped": "*Sam has stopped smoking; in fact, Sam did not smoke before.*",
            "all": "*All of the cakes are gone; in fact, two are left.*",
            "bakery": "*The bakery has stopped selling cakes; in fact, it still sells them every day of the week.*",
        },
        correct="right",
        why={
            "right": "*Some* says at least one, and suggests *not all*. *In fact, all of them are* takes back the suggestion, not what was said: when all the cakes are gone, some are. An implicature can be taken back like this.",
            "stopped": "*Has stopped smoking* takes for granted that Sam smoked before, and the second half denies that: the sentence clashes with itself. A presupposition is not taken back as easily as an implicature. Look for a sentence that takes back only what a listener works out.",
            "all": "*All of the cakes are gone* says that none is left, and *two are left* denies what was said, not something a listener worked out: a contradiction. Look for a sentence that takes back only what a listener works out.",
            "bakery": "*Has stopped selling cakes* says that the bakery does not sell them now, and *it still sells them* denies what was said: a contradiction. Look for a sentence that takes back only what a listener works out.",
        },
    )
    choice_cancel
    return (choice_cancel,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Meaning: when is a sentence true?](/course/open/nlp-meaning-and-truth?lang=en) checked what a sentence says against a world. This lesson added what a speaker means: the act done with the words, the stronger words on a scale that a listener takes to be false, and what a sentence takes for granted. You listed the implicatures of a scale, found what *the* refers to, sorted speech acts and took back an implicature. The lessons *How a conversation works* and *A text is more than its sentences* build on this: a conversation as a row of acts, and a text as more than a row of sentences.

    **An everyday example.** A review says that the food was *edible*. Nobody calls a great meal edible: the reviewer could have said *good*, and did not.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cancel, result_acts, result_implicated, result_the_one):
    L.wrap_up(result_implicated, result_the_one, result_acts, choice_cancel)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
