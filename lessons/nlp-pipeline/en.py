# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-pipeline. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-pipeline", "en")
    C = L.checks   # the course's three steps, and four made-up sentences with what a person gives at each step


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What NLP is](/course/open/nlp-what-is-nlp?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_side():
    quick_side = L.choice(
        "quick-side",
        question="A program reads the emails of a shop and puts each one in a folder: complaints, questions or orders. Which side of NLP is that?",
        options={
            "nlg": "Making (NLG): it gives back the name of a folder",
            "right": "Understanding (NLU): text goes in, and a label comes out",
            "both": "Both sides, since an email is text and every folder has a name in words",
        },
        correct="right",
        why={
            "right": "The program gets something out of text, the kind of email, and makes no text of its own. This lesson splits such a program into steps.",
            "nlg": "Making text means writing language from data. The name of a folder is a label the program picks, not text it writes. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "both": "Text goes in, but what comes out is a label, not text that the program writes. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_side
    return (quick_side,)


@app.cell(hide_code=True)
def quick_check_reply():
    quick_reply = L.choice(
        "quick-reply",
        question="The bike shop's assistant in *What NLP is* first finds the kind of request, then picks a reply for that kind. If it finds the wrong kind, what happens to the reply?",
        options={
            "fixed": "The step that picks the reply sees the mistake, and picks the reply that fits the request",
            "unknown": "The assistant says it did not understand the request",
            "right": "It is about the wrong thing too, since it is picked by the kind",
        },
        correct="right",
        why={
            "right": "The reply step only sees the kind it gets. A wrong kind gives a reply about the wrong thing: a mistake in the first step comes out at the end. That is what this lesson is about.",
            "fixed": "The reply step gets only the kind, not the request, so it cannot see the mistake. See the demo in [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
            "unknown": "The assistant says that only for the kind `\"unknown\"`. With a wrong kind, it confidently answers about the wrong thing. See [What NLP is](/course/open/nlp-what-is-nlp?lang=en).",
        },
    )
    quick_reply
    return (quick_reply,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A search engine, a spam filter or a voice assistant rarely handles text in one go. The work is split into steps: where do the sentences end, what are the words, which words are names? Each step is small enough to write and to test on its own, and each step works on what the step before it gave back.

    This lesson builds such a chain of three steps, and follows a mistake from the first step to the end.

    **An everyday example.** On an assembly line, each worker gets what the worker before them made. A part that is crooked at the start ends up crooked in the car, even when everyone after it does their job well.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="A chain has three steps. Each step gets 90 of every 100 sentences right, as long as its input is right. If every mistake makes the end result wrong, about how many of 100 sentences come out right at the end?",
        options={
            "most": "About 90",
            "right": "About 73",
            "half": "About 50",
        },
        reveal="**About 73.** Of 100 sentences, the first step gets 90 right. The second step gets 90 in 100 of those right, 81, and the third 90 in 100 of those, about 73. This simple sum is the course's own: it assumes that every mistake reaches the end. *Mistakes pass on* in *The idea* says what was measured in a real chain.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A chain of steps

    An **NLP pipeline** is a chain of steps in which each step takes the output of the step before. The documentation of spaCy, a library that works this way, says it plainly: "All components return the modified document, which is then processed by the next component in the pipeline." A typical chain splits a text into sentences, splits the sentences into words, says what kind of word each word is, and finds the names of people and places. Later steps can use everything the earlier steps found.

    The course's chain below has three steps, each with a simple rule the course made up. Step 1 starts a new sentence after every period, question mark or exclamation mark that is followed by a space. Step 2 cuts each sentence at the spaces, and makes a comma at the end of a word, and the mark at the end of the sentence, words of their own. Step 3 takes a word that starts with a capital letter as (part of) a name, but skips the first word of each sentence, since every sentence starts with a capital; capital words next to each other make one name, such as *New York*.

    ### Three tools

    You do not have to write such steps yourself: libraries such as spaCy and Stanza offer whole chains, with steps that were trained on sentences that people marked by hand.

    **spaCy** is a Python library for NLP that Matthew Honnibal launched in 2015, and that the company Explosion maintains, under the MIT license. Its small pipeline for Dutch, `nl_core_news_sm` 3.8.0, is 12 MB and has its own license, CC BY-SA 4.0. First it cuts the text into words; then come steps called `tok2vec`, `morphologizer`, `tagger`, `parser`, `lemmatizer`, `attribute_ruler` and `ner`; later lessons come back to several of them.

    **NLTK**, the Natural Language Toolkit, is older. Its book says: "NLTK was originally created in 2001 as part of a computational linguistics course in the Department of Computer and Information Science at the University of Pennsylvania." Its makers, Edward Loper and Steven Bird, built it for teaching. NLTK is more a toolbox than one chain: you put the steps together yourself, and you can download texts and word lists for it, also in Dutch. In 2004 it was under the GPL; on November 21, 2026, it was under the Apache License 2.0.

    A third tool is Stanza, from the Stanford NLP Group. Until version 1.0.0, in 2020, it was called StanfordNLP. Its paper of that year introduces it as "an open-source Python natural language processing toolkit supporting 66 human languages"; its homepage said 70 on November 21, 2026. This course writes its own steps, so that you see what happens inside them.

    ### Mistakes pass on

    A step gets only what the step before gave back, and it cannot see what that step missed. Jenny Rose Finkel, Christopher Manning and Andrew Ng (2006) write that how well an NLP system does a task of several steps "is often hampered by use of a greedy 1-best pipeline architecture, which causes errors to propagate and compound at each stage." *Greedy* and *1-best* mean that each step passes on only its single best guess, and *compound* that mistakes pile up.

    They also give a measurement, from work by Kristina Toutanova (2005): a step that finds who did what to whom in a sentence scored 91.2 out of 100, on a score from 0 to 100 and not a count of sentences, when it got sentence structures that people had made by hand, and 80.0 when it got the structures an earlier step had computed. A score can even hide such a mistake. The documentation of spaCy warns that when the step that cuts words cuts them so that a name no longer fits, that name does not count in the score for names. That score "might still look good", even when half of the names cannot be found.
    """)
    return


@app.cell(hide_code=True)
def pipeline_cell():
    mo.Html(C.pipeline_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **NLP pipeline** | a chain of steps in which each step takes the output of the step before |
        | **spaCy** | a Python library for NLP with trained pipelines, also for Dutch |
        | **NLTK** | the Natural Language Toolkit: a Python library with tools, texts and word lists, made for teaching |
        | `C.STEPS` | the course's three steps as a list: `C.split_sentences`, `C.split_words` and `C.find_names` |
        | `C.SENTENCES`, `C.EXPECTED` | four made-up sentences, and for each the output that a person gives at each step |
        | `C.OUTPUTS` | for each sentence, the outputs of the course's three steps |
        | `step(x)` | when `step` holds a function, `step(x)` calls it: a list can hold functions like any other value |
        """),
        L.watch_out("The sum in the guess is the course's own, and too simple. In a real chain, some mistakes do no harm at the end, and the steps often go wrong on the same hard sentences. How much an early mistake costs differs from chain to chain: measure it."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write each step as a function that takes the output of the step before.
    2. Give the text to the first step, and each output to the next step.
    3. Keep the output of every step, not only the last one.
    4. Compare each output with what a person gives at that step. The first one that differs is where the mistake starts.
    5. Fix the step where the mistake starts. The steps after it may be right, and only get wrong input.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the fourth of the course's sentences through the three steps by hand, one line for each step, and prints what each step gives back.
    """)
    return


@app.cell
def example_steps():
    text = C.SENTENCES[3]   # two sentences
    sentences = C.split_sentences(text)
    words = C.split_words(sentences)
    names = C.find_names(words)
    print(sentences)
    print(words)
    print(names)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each line gives the next step what the line before it got back: two sentences, then two lists of words, then the two names. Every step is right for this sentence. Exercise 1 does the same with a loop, for any list of steps.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_trace():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. trace: the output of every step

        `C.STEPS` is a list of the course's three steps, each a function. Write `trace(text, steps)`: give `text` to the first function of the list `steps`, give what it gives back to the next function, and so on, and give back a list with the output of every step, in order. Start with an empty list `outputs`, and keep what the last step gave back in a variable `result`. Replace the `...`. The check calls `trace(SENTENCES[0], STEPS)`, then the other sentences and steps of its own.
        """),
        L.task("Go over the steps, give each one the output of the step before, and keep every output."),
    ])
    return


@app.cell(hide_code=True)
def exercise_trace():
    box_trace = L.code_box("trace", lines=7)
    box_trace
    return (box_trace,)


@app.cell(hide_code=True)
def check_trace(box_trace):
    result_trace = L.check("trace", box_trace.value)
    result_trace
    return (result_trace,)


@app.cell(hide_code=True)
def hints_trace():
    L.hints(
        "trace",
        [
            "Look at the worked example: each line gives a step what the line before it got back. A `for` loop over `steps` does that for any number of steps.",
            "What does each step get: the text you started with, or what the step before it gave back? And when do you add an output to the list?",
            "`        result = ___`",
        ],
        why="`outputs = []` and `result = text`, then `for step in steps:` with `result = step(result)` and `outputs.append(result)` under it, and after the loop `return outputs`. `trace(SENTENCES[0], STEPS)` gives three outputs, and the last is `['New York']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_trace():
    mo.md(r"""
    For the first sentence, *Dr. Smith lives in New York.*, the first output already holds two sentences where there is one, and the last output misses *Smith*. Exercise 2 finds the first step that goes wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_first_wrong():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. first_wrong: where the mistake starts

        `C.EXPECTED` holds, for each sentence, the output that a person gives at each step, and `C.OUTPUTS` the outputs of the course's steps. Write `first_wrong(outputs, expected)`: the number of the first step whose output is not the same as the expected output, counting from 0 as Python counts the places in a list, and `None` when every step is right. Go over the numbers of the steps with `for i in range(len(outputs)):`. Replace the `...`. The check calls `first_wrong(OUTPUTS[0], EXPECTED[0])`, then the other sentences and lists of its own.
        """),
        L.task("Go over the steps in order, and give back the number of the first one that is wrong."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first_wrong():
    box_first_wrong = L.code_box("first_wrong", lines=5)
    box_first_wrong
    return (box_first_wrong,)


@app.cell(hide_code=True)
def check_first_wrong(box_first_wrong):
    result_first_wrong = L.check("first_wrong", box_first_wrong.value)
    result_first_wrong
    return (result_first_wrong,)


@app.cell(hide_code=True)
def hints_first_wrong():
    L.hints(
        "first_wrong",
        [
            "Step 4 of *Recipe*: look at the outputs one at a time, from the first step on, and stop at the first one that is wrong.",
            "What do you compare for step number `i`? And when may the function say that no step is wrong?",
            "`        if ___:`",
        ],
        why="`for i in range(len(outputs)):`, then `if outputs[i] != expected[i]:` with `return i` under it, and after the loop `return None`. `first_wrong(OUTPUTS[0], EXPECTED[0])` gives `0`: the mistake starts at the first step.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: follow a sentence through the chain

    Pick a sentence, and see what each step gives back. A box that differs from what a person gives at that step has a purple frame, with what a person gives under it.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_sentence = mo.ui.dropdown(options=C.sentence_options(L.lang), value=list(C.sentence_options(L.lang))[0], label="A sentence")
    demo_sentence
    return (demo_sentence,)


@app.cell(hide_code=True)
def demo_look(demo_sentence):
    mo.Html(C.demo_picture(demo_sentence.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_fix():
    mo.md(r"""
    ### Exercise 3. A fix in step 3

    In the first sentence, step 3 misses *Smith*. Someone wants to fix that by changing step 3: from now on, it also takes the first word of a sentence as a name when it starts with a capital letter.
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="What does that change do to the course's four sentences?",
        options={
            "right": "*Smith* is found, but so are first words that are not names, such as *We*",
            "only": "Only *Smith* is added, and the chain is right from then on",
            "nothing": "Nothing changes, since the word *Smith* does not reach step 3 from the steps before it",
        },
        correct="right",
        why={
            "right": "Step 3 now takes every first word with a capital, so it also finds *I*, *We* and *Dr*. It also finds *Anna* in the third sentence, whose mistake does start in step 3. The mistake of the first sentence started in step 1, which cut the sentence after *Dr.*; a step 1 that knows *Dr.* does not end a sentence finds *Smith* without new mistakes. Step 5 of *Recipe*: fix the step where the mistake starts.",
            "only": "Every sentence starts with a capital, so the new rule also takes *I*, *We* and *Dr* as names. Try it in your head on the fourth sentence. Step 5 of *Recipe* says where to fix.",
            "nothing": "Step 3 does get *Smith*, but as the first word of the sentence that step 1 made, so its own rule skips it. Look at the first sentence in the demo.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_exercise_measure():
    mo.md(r"""
    ### Exercise 4. A score on clean input

    A team tests a step that finds names on sentences that people split into sentences and words by hand. It scores 95 out of 100. In the real chain, the step will get what the course's steps 1 and 2 give back.
    """)
    return


@app.cell(hide_code=True)
def choice_measure_cell():
    choice_measure = L.choice(
        "measure",
        question="What score do you expect in the real chain?",
        options={
            "same": "The same 95, since the step itself has not changed",
            "right": "Lower, since steps 1 and 2 make mistakes that the split by hand does not have",
            "higher": "Higher, since three steps that work together know more than one step on its own does",
        },
        correct="right",
        why={
            "right": "The step meets input it never saw in the test, such as a sentence cut after *Dr.* or a name with quotation marks on it. Toutanova's step went from 91.2 to 80.0 when it got the structures of an earlier step. To know what a step does in the chain, test it in the chain.",
            "same": "The step is the same, but its input is not: a sentence cut after *Dr.* or a word with quotation marks on it. See *Mistakes pass on* in *The idea*.",
            "higher": "The later step gets only the output of the earlier ones, mistakes included; nothing in the chain repairs them. See *Mistakes pass on* in *The idea*.",
        },
    )
    choice_measure
    return (choice_measure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [What NLP is](/course/open/nlp-what-is-nlp?lang=en) found what a request was about in one step. This lesson split the work into a chain: an NLP pipeline, in which each step takes the output of the step before, as the steps of spaCy do. NLTK offers the parts for such chains. With the output of every step in hand, you found where a mistake starts, and saw that a mistake in the first step comes out at the end, even when the later steps follow their rules. The next lesson, *Words and their parts*, looks inside the words that step 2 cuts out.

    **An everyday example.** When a recipe goes wrong, you taste after each step, not only at the end: then you know which step to do again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, choice_measure, result_first_wrong, result_trace):
    L.wrap_up(result_trace, result_first_wrong, choice_fix, choice_measure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
