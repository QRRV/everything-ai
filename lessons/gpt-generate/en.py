# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson gpt-generate. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("gpt-generate", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Sampling: let the model write](/course/open/lm-sampling?lang=en) and [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sees():
    quick_sees = L.choice(
        "quick-sees",
        question="Your bigram model writes \"thithe thes s hand\": many short pieces, few whole words. Why does it rarely finish a word?",
        options={
            "rows": "Its table has too few rows: one per character",
            "right": "It sees only the last character of the text",
            "hot": "Its temperature is too high",
        },
        correct="right",
        why={
            "rows": "A row per character is what a bigram model is: the row of the last character is all it uses. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "hot": "A lower temperature makes it safer, but it still sees only one character: it cannot know which word it is in. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "right": "Today your GPT writes with the same loop, and sees up to `block` characters back.",
        },
    )
    quick_sees
    return (quick_sees,)


@app.cell(hide_code=True)
def quick_check_cold():
    quick_cold = L.choice(
        "quick-cold",
        question="Which temperature makes a model nearly always take its most likely next character, as greedy decoding does?",
        options={
            "one": "1: the probabilities as the model gives them",
            "big": "A large one, such as 10",
            "right": "One close to 0, such as 0.1",
        },
        correct="right",
        why={
            "one": "At 1 the probabilities stay as they are, and less likely characters are drawn too. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
            "big": "A large temperature makes the probabilities flatter, nearly the same for every character. See [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en).",
            "right": "Dividing by 0.1 makes the differences between the logits ten times as big, so the largest one wins nearly always.",
        },
    )
    quick_cold
    return (quick_cold,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Sampling: let the model write](/course/open/lm-sampling?lang=en) your bigram model wrote with a loop: the probabilities of the next character, a character drawn with them, added to the text, and again. Your GPT is trained now, in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en), and it gives probabilities of the next character too, for a whole text instead of one character. In this project step your loop writes with your GPT, with one line changed. This is how every GPT writes, even the big ones: one token at a time.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           4, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_text():
    guess_text = L.guess(
        "gpt-text",
        question="Your GPT writes 300 characters after \"ROMEO:\" and a new line, at a temperature of 0.8. What do you think the text looks like?",
        options={
            "soup": "Like the bigram model: short pieces, a word here and there",
            "right": "Mostly real words, names and lines of a play, in sentences that make little sense",
            "copy": "Whole speeches from the plays, word for word",
        },
        reveal="**Mostly real words, names and lines of a play.** With a validation loss of 1.7 or less, far below the bigram model, your GPT has learned how words are spelled, that a name in capitals with a colon starts a speech, and how long a line of verse is. It has not learned what the words mean: a million or ten million parameters, trained on one book, is far too little for that. And it does not copy whole speeches: it draws every character anew, and one less likely draw sends the text its own way.",
    )
    guess_text
    return (guess_text,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Writing with a GPT is the loop of your bigram model; only the probabilities come from somewhere else:

        - **the window:** your GPT reads at most `block` characters, so it gets the last `block` characters of the text; a shorter text it reads whole;
        - **the last place:** it gives logits at every place of the window, each about the character after that place. Only those of the last place are about the next character of the text;
        - **the temperature:** as in [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en), divide those logits by $\tau$ before softmax;
        - **the loop:** draw a character with the probabilities, add it to the text, and go again: one forward pass for every new character.

        For writing, your GPT needs no gradients: with `torch.no_grad()` PyTorch builds no graph, which saves memory and time.

        ### New words and symbols

        | Word or symbol | What it means |
        |---|---|
        | **window** | the last `block` characters of the text: what the GPT reads to give the next one |
        | `num_embeddings` | the number of rows of a table, here `model.pos`, the table of places of your GPT: its `block` |
        """),
        L.watch_out("Each new character costs a whole forward pass over the window, also for all the characters the GPT already read in the step before. For 300 new characters, that is 300 passes. Big models keep the keys and values of earlier places in a KV cache, so that a step only works on the new character; a later lesson builds one."),
        mo.md(r"""
        The picture: one step. The window holds the last characters of the text; your GPT gives logits at every place, and only those of the last place, divided by $\tau$ and through softmax, give the probabilities of the next character. A character drawn with them goes to the end of the text, and the next step starts.
        """),
        mo.Html(C.loop_picture({"title": "One step of writing", "window": "window", "gpt": "your GPT", "logits": "logits", "probs": "÷ τ, softmax",
                                "draw": "draw, add, again",
                                "alt": "The text ROMEO, a colon, a new line and No, in boxes, with a new box at the end for the next character. The last six characters are the window, which goes into your GPT. Below it a row of logits, one for every place of the window; only the last one is colored. From it, after dividing by tau and softmax, three bars: a comma 0.39, a t 0.23 and a space 0.16, the course's GPT after this text at a temperature of 0.8. An arrow goes from the bars back to the new box: draw, add, again."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Writing with a GPT:

    1. The last `block` characters of the text, as ids.
    2. The logits of the GPT for them, without a graph; take those of the last place.
    3. Divide them by $\tau$ and take softmax: the probabilities of the next character.
    4. Draw a character with those probabilities, and add it to the text.
    5. Go back to step 1 until the text is long enough.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 2 to 4 for the logits a GPT gave for a text of 3 characters, with a vocabulary of 3: `"a"`, `"e"` and `"t"`. The logits have shape (1, 3, 3): one text, 3 places, 3 scores at every place. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_step():
    import numpy as np
    import torch

    _chars = ["a", "e", "t"]
    _logits = torch.tensor([[[2.0, 0.5, 0.1], [0.3, 1.8, 0.2], [0.4, 0.2, 2.2]]])
    # 2. the logits about the next character
    _last = _logits[0, 0]  # complete: _logits[0, -1], those of the last place
    # 3. a temperature of 0.8, and softmax
    _p = torch.softmax(_last / 0.8, dim=0).numpy()
    print(np.round(_p, 3))
    # 4. draw the next character
    _rng = np.random.default_rng(1)
    print(_chars[_rng.choice(3, p=_p)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell takes the logits of the first place: `[0.802 0.123 0.075]`, and draws `a`. Those are about the character after the first one, which the text already has. Completed, with the last place: `[0.089 0.069 0.842]`, and it draws `t`, with the same seed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_passes():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many passes?
    """)
    return


@app.cell(hide_code=True)
def choice_passes_cell():
    choice_passes = L.choice(
        "passes",
        question="Your GPT writes a text of 300 characters, after a start of 7: `\"ROMEO:\"` and a new line. How many forward passes does it make?",
        options={
            "one": "1: one pass gives logits at every place",
            "all": "300: one for every character of the text",
            "right": "293: one for every new character",
        },
        correct="right",
        why={
            "one": "One pass gives logits at every place, but only the last place is about a character the text does not have yet. The next one needs a pass over the text with that character in it.",
            "all": "The 7 characters of the start are given: nothing has to be drawn for them.",
            "right": "Each of them over a window of up to `block` characters. A KV cache, in a later lesson, saves most of that work while the text fits in the window.",
        },
    )
    choice_passes
    return (choice_passes,)


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The probabilities of the next character

        Complete `gpt_probs(model, row_of, text, tau)`: steps 1 to 3 of the recipe. `row_of` gives the id of every character, as in your `generate` of [Sampling: let the model write](/course/open/lm-sampling?lang=en); it gives back a NumPy array, as a row of your bigram table was. The check makes a small GPT of yours after `torch.manual_seed(0)`, for the 5 characters `"abcde"` and texts of at most 4, and calls `gpt_probs(model, row_of, "abc", 1.0)`; then it tries a longer text, two temperatures and a GPT that reads 6 characters. When it is right, the check saves it in your project, as `work/projects/llm/gpt_probs.py`.
        """),
        L.task("Make `gpt_probs` give the probabilities of the next character, for any text and temperature."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("gpt-probs", lines=13)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("gpt-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "gpt-probs",
        [
            "The ids: a loop over the last `block` characters of `text`, with `row_of`. Then your GPT on a tensor of one text, `torch.tensor([ids])`, inside `with torch.no_grad():`.",
            "The logits have shape (1, n, 5) in the check: one text, n places, 5 scores at every place. Step 2 of the completed worked example takes the last place; step 3 divides by the temperature and takes softmax, `dim=0`, as NumPy.",
        ],
        why="Only the logits of the last place are about a character the text does not have yet; the others are about characters that are already there.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_generate(result_probs):
    result_probs  # the note changes when exercise 2 saves your gpt_probs
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your loop, with your GPT

        Complete `gpt_generate(model, chars, start, n, tau, seed)`: steps 4 and 5 of the recipe, the loop of your `generate` of [Sampling: let the model write](/course/open/lm-sampling?lang=en) with the probabilities of your `gpt_probs` in place of a row of the table. Its first lines are those of your `generate`, and import your `gpt_probs` from your project. The check calls `gpt_generate(model, chars, "ab", 9, 1.0, 1)` with the small GPT of exercise 2, and tries other starts, seeds and temperatures, and a GPT that reads 6 characters. When it is right, the check saves it in your project, as `work/projects/llm/gpt_generate.py`.
        """),
        L.course_versions(),
        L.task("Make `gpt_generate` write with your GPT, one drawn character at a time."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("gpt-generate", lines=14)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate):
    result_generate = L.check("gpt-generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "gpt-generate",
        [
            "The loop of your `generate` of lm-sampling: as long as the text is shorter than `n`, the probabilities of the next character, a drawn character, added to the text.",
            "Only the line of the probabilities changes: in place of the row of the last character, your `gpt_probs` for the whole text so far.",
        ],
        why="Your GPT reads the whole text so far, up to `block` characters; the bigram model read only its last character. That one line is the whole difference between the two.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below lets your GPT write 300 characters after "ROMEO:" and a new line, with your `gpt_generate`: the GPT you trained in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en), or the course's GPT when you have not trained one yet. Under it, your bigram model writes after the same start, with your `generate` of lm-sampling. Pick a temperature.
    """)
    return


@app.cell(hide_code=True)
def temperature_cell():
    temperature = mo.ui.dropdown(options={C.number(t, 1, L.lang): t for t in C.TEMPERATURES}, value=C.number(0.8, 1, L.lang), label="Temperature")
    temperature
    return (temperature,)


@app.cell(hide_code=True)
def demo_write(temperature, result_probs, result_generate):
    result_probs, result_generate  # run again when exercise 2 or 3 saves your module
    _result = C.demo(temperature.value)
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `gpt_probs` and your `gpt_generate` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Writing takes too long here: more than a minute. Does your `gpt_probs` or your `gpt_generate` do more work than one forward pass for every new character?*",
            "gpt": "*Your GPT, or your code in `work/projects/llm/`, stops with an error while it writes. Do your modules still pass their checks? Delete a file you changed, and check its exercise again.*",
        }[_result])
    else:
        _who = "**Your GPT**, trained in [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en)" if _result["source"] == "yours" else "**The course's GPT**, as you have not trained one yet"
        _stand = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _result["stand_ins"]) + ", as the note at exercise 3 says."
        _out = mo.md(f"""{_who} (validation loss {C.number(_result['best'], 2, L.lang)}), at a temperature of {C.number(temperature.value, 1, L.lang)}:

{C.shown_text(_result['gpt'])}

**Your bigram model**, at a temperature of 1:

{C.shown_text(_result['bigram'])}

The same loop, one line changed: the GPT reads the whole text so far, up to `block` characters; the bigram model read only its last character.{_stand}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The sampling loop of [Sampling: let the model write](/course/open/lm-sampling?lang=en) and the temperature of [Temperature: bolder or safer text](/course/open/lm-temperature?lang=en), now with the probabilities of your GPT of [Project: train your GPT on a book](/course/open/gpt-train-chars?lang=en). Your project `llm` now has `gpt_probs.py` and `gpt_generate.py`: your own GPT, from text to text. Next: *Project: your GPT on your own tokens*, where your GPT trains on the tokens of your own BPE tokenizer instead of characters.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_passes, result_probs, result_generate):
    L.wrap_up(choice_passes, result_probs, result_generate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
