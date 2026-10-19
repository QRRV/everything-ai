# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-hallucination. Route: course/route/lessen/taal.yaml. Checks: checks.py.
# termen:allow confabulation

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-hallucination", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Sampling: let the model write](/course/open/lm-sampling?lang=en) and [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rare():
    quick_rare = L.choice(
        "quick-rare",
        question="Your model writes by sampling. After some character, a rare character has a probability of 0.01. What happens to it?",
        options={
            "never": "It is never drawn: it is too unlikely",
            "right": "It is drawn now and then, about once in 100 times",
            "most": "It is drawn most of the time, because sampling favours rare characters",
        },
        correct="right",
        why={
            "never": "Sampling draws with the probabilities of the row, so 0.01 comes up now and then; only greedy decoding never takes it. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "most": "Sampling takes each character about as often as its probability says, and no more. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "right": "Sampling draws with the probabilities of the row: 0.01 means about one draw in 100.",
        },
    )
    quick_rare
    return (quick_rare,)


@app.cell(hide_code=True)
def quick_check_chat():
    quick_chat = L.choice(
        "quick-chat",
        question="A chat model gets a question. What does it do to answer it?",
        options={
            "lookup": "It looks the answer up in a store of checked facts",
            "copy": "It picks one of the answers it keeps from training",
            "right": "It continues the text, token by token, with what is likely after the question",
        },
        correct="right",
        why={
            "lookup": "A chat model has no store of facts to look in: it predicts the next token, as a base model does. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
            "copy": "A chat model keeps no list of answers: it writes its answer token by token, with what is likely next. See [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en).",
            "right": "A chat model still only predicts the next token; after a question, the likely text is an answer.",
        },
    )
    quick_chat
    return (quick_chat,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Ask a chat model for a quote by a famous person, or for the source of a claim, and you get one: in the right style, with a title and a year. Now and then the quote was never said, and the book does not exist. The model did not lie, and nothing broke: it did what every language model does, the same thing that makes it useful.

    **An everyday example.** Someone who has read a lot of wine reviews is asked about a wine they have never tasted. They answer at once, with the right words: a hint of oak, a long finish. It sounds like an expert speaking, and it is made up from what such answers usually say.
    """)
    return


@app.cell(hide_code=True)
def predict_romeo():
    _found = C.found()
    guess_romeo = L.guess(
        "romeo-lines",
        question=f"The course's model of the plays writes the first line of {C.N} speeches of Romeo, each with a different seed. How many of those {C.N} lines do you think Romeo really says somewhere in the plays?",
        options={
            "most": "Most of them: it learned them from the plays",
            "half": "About half of them",
            "right": "Hardly any of them",
        },
        reveal=f"**Hardly any: {len(_found)} of the {C.N}.** Only \"{_found[0]}\" is a real line of the plays. The other {C.N - len(_found)} sound like Romeo, in the words of the plays, and nobody in the plays says them; the first five are \"{C.made_up()[0]}\", \"{C.made_up()[1]}\", \"{C.made_up()[2]}\", \"{C.made_up()[3]}\" and \"{C.made_up()[4]}\". The model does not know which of its lines are real: it writes likely text, and a made-up line in the style of the plays is likely too.",
    )
    guess_romeo
    return (guess_romeo,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(f"""
        ## The idea

        A **hallucination** is output that sounds fluent and sure but is false or made up, such as a quote nobody ever said, or a source that does not exist. It comes from what a language model is: it gives likely text. In training it learned which tokens tend to follow which; nothing in next-token prediction checks whether a sentence is true. Often a likely sentence is a true one, and that is why language models are useful. But a made-up quote in the right style is likely too, token by token.

        How sure the model sounds tells you nothing about this. The course's model gave each character of its own {C.N} lines of Romeo a probability of {L.number(C.mean_prob(C.made_up()), 2)} on average, and each character of the first {C.N} real lines of Romeo {L.number(C.mean_prob(C.real()), 2)}, although those real lines are in the very text it counted: it is as sure of what it made up as of what is real. A chat model sounds sure in the same way: it learned from texts in which answers are written with confidence.

        A model can also get wrong what is in the text it was given, such as a summary that changes a number from the article: a likely number, not the one that was there.

        Every model has a **knowledge cutoff**: the date up to which its training text reaches. About anything later, the model knows nothing, unless it is in the text it gets, for example from a search. It still writes likely text about it. The makers of a model usually say when its cutoff is; the model itself often does not know it exactly. The plays are more than 400 years old, and the word "computer" does not occur in them. Give the course's model "My computer" after `ROMEO:`, and it writes on without a pause: "My computer{C.write_rest(C.START + "My computer", 1)}"

        What comes out also depends on details of the text that should not matter. Give the course's model `ROMEO:` with a space after the colon, and with the same seed 1 it no longer writes "{C.romeo_line(1)}" but "{C.write_rest(C.START.replace(":", ": "), 1).strip()}". A large model is far less fragile, yet other words for the same question, or the same examples in another order, can still change its answer. This is **prompt sensitivity**: the prompt, the text you give a model, matters in ways it should not. So one answer, however sure it sounds, is not yet checked.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **hallucination** | | output that sounds fluent and sure, but is false or made up | |
        | **knowledge cutoff** | | the date up to which a model's training text reaches; about later events the model knows nothing, unless it is told | |
        | **prompt sensitivity** | | how much a model's answer changes with small changes in the text it gets, the prompt, that should not matter | |
        | `line in lines` | | `True` when `line` is one of the items of `lines`, as `in` checks the keys of a dictionary | `"Whither?" in ["Whither?", "Away!"]` gives `True` |
        """),
        L.watch_out("*Hallucination* is a borrowed word: in medicine it means seeing or hearing what is not there. A language model sees nothing; it writes likely text, and sometimes that text is false. Some researchers say confabulation instead: filling a gap with something that sounds right, without knowing that it is made up."),
        mo.md(f"""
        The picture: how sure the model is of each of its {C.N} lines of Romeo, and of each of the first {C.N} real lines of Romeo. Every circle is a line; the short upright line is the mean.
        """),
        mo.Html(C.sure_strips({"title": "How sure is the model?", "made": f"the model's own lines ({len(C.found())} of {C.N} real)",
                               "real": "real lines of Romeo", "mean": "mean {m}", "axis": "average probability per character",
                               "number": lambda v: L.number(v, 2), "tick": lambda v: L.number(v, 1),
                               "alt": f"Two rows of {C.N} circles on one axis from 0.4 to 0.9: the average probability per character of the model's own lines of Romeo, and of real lines of Romeo. The two rows lie over each other, both mostly between 0.55 and 0.8, with their means close together."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check what a model writes:

    1. Let the model write: here, the first line of a speech of Romeo, with several seeds.
    2. Look each line up in a source you trust, here the lines of the plays, with `in`.
    3. Count the lines you found, and divide by the number of lines: the share that checks out.
    4. Put the model's probabilities next to it: how sure it was says nothing about what you found.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Five lines of Romeo from the course's model, with seeds 1 to 5. `C.play_lines()` gives every line of the plays, and `C.how_sure(line)` the average probability the model gives each character of a line, rounded to 2 decimals. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_check():
    # 1. the model writes the first line of Romeo, with seeds 1 to 5
    _lines = []
    for _seed in range(1, 6):
        _lines.append(C.romeo_line(_seed))
    print(_lines)
    # 2. look each line up in the lines of the plays
    _plays = C.play_lines()
    _found = 0
    for _line in _lines:
        if _line in _plays:
            _found = _found + 1
    # 3. the share of the lines that the plays really have
    print(_found / len(_lines))
    # 4. how sure the model was of each line
    for _line in _lines:
        print(C.how_sure(_line), _line)
    return


@app.cell(hide_code=True)
def text_after_example():
    _sure = [C.how_sure(C.romeo_line(_s)) for _s in range(1, 6)]
    mo.md(f"""
    Not one of the five lines is in the plays: the share is 0. Yet the model was sure of them, with between {L.number(min(_sure), 2)} and {L.number(max(_sure), 2)} per character on average, at least as sure as of the real lines in the picture. Only the source tells you which lines are real; the probabilities cannot.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cutoff():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. After the cutoff
    """)
    return


@app.cell(hide_code=True)
def choice_cutoff_cell():
    choice_cutoff = L.choice(
        "cutoff",
        question="A model's training text reaches up to January 2025. Someone asks it about an event of June 2025, and gives it no text about that event. What can happen?",
        options={
            "right": "It writes a fluent answer that may well be made up",
            "always": "It always says that it cannot know about that event",
            "search": "It looks the event up on the internet by itself",
        },
        correct="right",
        why={
            "always": "A model can be trained to say it does not know, but it does not always do so: it writes likely text, and a fluent answer is likely too.",
            "search": "A model only gets the text it is given. Some chat programs search the internet and add what they find to that text, but the model itself looks nothing up.",
            "right": "About June 2025 the model knows nothing, but it still gives probabilities for every next token, so a fluent answer can come out, made up from what such answers usually say.",
        },
    )
    choice_cutoff
    return (choice_cutoff,)


@app.cell(hide_code=True)
def text_exercise_found():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Check against a source

        `found_share(lines, source)` gives the share of `lines` that `source` really has, as in steps 2 and 3 of the recipe. The loop and the last line are there. Replace the `...` with an `if` that counts a line when it is in `source`. The check calls `found_share(["Good night.", "Be merry.", "Whither?", "Away, away!", "Away!"], ["Whither?", "Good night.", "Away!"])`, which should give `0.6`, and tries other lines and sources, also an empty source.
        """),
        L.task("Make `found_share` count each line that is in `source`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_found():
    box_found = L.code_box("found-share", lines=7)
    box_found
    return (box_found,)


@app.cell(hide_code=True)
def check_found(box_found):
    result_found = L.check("found-share", box_found.value)
    result_found
    return (result_found,)


@app.cell(hide_code=True)
def hints_found():
    L.hints(
        "found-share",
        [
            "Steps 2 and 3 of the recipe, as in the worked example.",
            "For each line: is it one of the lines of `source`? `line in source` gives `True` or `False`. If it is, `found` goes up by one.",
            "`if line ___ source:`, and inside it `found = found + ___`.",
        ],
        why="Of the five lines, \"Good night.\", \"Whither?\" and \"Away!\" are in the source; \"Away, away!\" is not, because the source only has \"Away!\" itself. 3 of 5 is 0.6.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sure():
    mo.md(r"""
    ### Exercise 3. Sure, and still made up
    """)
    return


@app.cell(hide_code=True)
def choice_sure_cell():
    _line = C.romeo_line(30)
    choice_sure = L.choice(
        "sure-not-true",
        question=f"The model gave the characters of its line \"{_line}\" a probability of {L.number(C.how_sure(_line), 2)} on average, more than any of the first {C.N} real lines of Romeo got. Yet nobody says that line in the plays. What does that show?",
        options={
            "real": "The line must be real somewhere, since the model is so sure of it",
            "right": "Probability says how likely a line is, not whether it is true",
            "broken": "The model is broken: a good model is only sure of real lines",
        },
        correct="right",
        why={
            "real": "The plays are all the model learned from, and the line is not in them: the model is sure of a line it made up.",
            "broken": "This is no fault of this one model: every language model gives likely text, and a made-up line can be just as likely as a real one.",
            "right": "The model is sure because the line looks like the plays, character by character. Whether Romeo ever said it, the model cannot tell; for that you look in the source.",
        },
    )
    choice_sure
    return (choice_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A language model gives likely text, as in [Sampling: let the model write](/course/open/lm-sampling?lang=en) and [Base model or chat model?](/course/open/lm-base-vs-chat?lang=en); likely is often true, but not always, and how sure the model is does not tell you which. So check what matters against a source you trust. With *Show extra lessons* on, two come first: *Encoders, decoders and the GPT line*, about the kinds of language model, and *Look inside a real LLM's next-token probabilities*, where you load a small open model and read its probabilities. In the next lesson of the main route, *What a paper, a preprint and peer review are*, you start reading research yourself, where every claim needs a source too.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cutoff, result_found, choice_sure):
    L.wrap_up(choice_cutoff, result_found, choice_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
