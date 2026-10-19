# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-temperature. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-temperature", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Sampling: let the model write](/course/open/lm-sampling?lang=en) and [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_draw():
    quick_draw = L.choice(
        "quick-draw",
        question="`rng.choice(3, p=[0.7, 0.2, 0.1])` runs 1,000 times. About how often does it give 2?",
        options={
            "never": "Never: 0 is the most likely",
            "right": "About 100 times: one draw in ten",
            "third": "About 333 times: each number a third of the time",
        },
        correct="right",
        why={
            "never": "That would be greedy decoding. Sampling draws 2 too, with probability 0.1. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "third": "That is drawing without `p=`. With `p=`, each number comes with its own probability. See [Sampling: let the model write](/course/open/lm-sampling?lang=en).",
            "right": "2 has probability 0.1, so it comes about one time in ten: about 100 of 1,000.",
        },
    )
    quick_draw
    return (quick_draw,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="What does softmax make of the logits `[2, 2, 2]`?",
        options={
            "right": "A third each: equal logits, equal probabilities",
            "same": "2 each: softmax keeps the logits",
            "first": "1 for the first, 0 for the others",
        },
        correct="right",
        why={
            "same": "Softmax turns logits into probabilities that add up to 1, so three times 2 cannot be the outcome. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "first": "Softmax gives every logit a share; equal logits get equal shares. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "e² three times, divided by their sum: a third each. Softmax only looks at the gaps between the logits, and here there are none.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your model writes by sampling, and its text is full of odd letter combinations; greedy decoding gets stuck in `the the the`. Something in between would help: draw, but favor the likely characters more, or less. Services that give programs access to an LLM, and apps that run one on your own computer, often have a setting for exactly that, the temperature, usually a number between 0 and 2. In this lesson, you build it, and you see what it does to the text of your model.

    **An everyday example.** A music app on shuffle that plays your favorite songs more often. A setting could make it stick almost only to your top songs, or play all your songs about equally often.
    """)
    return


@app.cell(hide_code=True)
def predict_half():
    guess_half = L.guess(
        "half",
        question="In row \"t\" of your table, \"h\" has the probability 0.34. You divide all the logits of that row by 0.5, which doubles them, and take softmax again. What do you think happens to \"h\"?",
        options={
            "same": "It stays at 0.34: every logit changes the same way",
            "right": "It gets more likely: the gaps between the logits double",
            "less": "It gets less likely: the other characters catch up",
        },
        reveal=f"**It gets more likely: {L.number(C.prob_after('t', 'h', 0.5), 2)} instead of {L.number(C.prob_after('t', 'h', 1), 2)}.** Doubling the logits doubles the gaps between them, and softmax turns a bigger gap into a bigger difference in probability. The space, second with {L.number(C.prob_after('t', ' ', 1), 2)}, goes up to {L.number(C.prob_after('t', ' ', 0.5), 2)}; every other character that ever came after \"t\" drops. Dividing the logits by a number below 1 makes the likely characters more likely.",
    )
    guess_half
    return (guess_half,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A neural network gives logits, numbers that softmax turns into probabilities, as in [Many classes: softmax](/course/open/ml-softmax?lang=en). Your bigram model gives the probabilities straight away, but it has logits too: their logarithms, `z = np.log(P[i])`. Softmax of those gives `P[i]` back, because $e$ to the power of $\log p$ is $p$, and the probabilities already add up to 1. Any numbers with the same gaps would do as well, because softmax only looks at the gaps; $\log p$ is the simplest choice.

        The **temperature** $\tau$ (tau) divides the logits before softmax. For each of the $K$ characters, the new probability $q_k$ is:
        """),
        L.formula(r"[[pred:q_k]] = \frac{e^{z_k / [[hyper:\tau]]}}{e^{z_0 / [[hyper:\tau]]} + e^{z_1 / [[hyper:\tau]]} + \dots + e^{z_{K-1} / [[hyper:\tau]]}}"),
        mo.md(r"""
        With $\tau = 1$, nothing changes. With $\tau$ below 1, the gaps between the logits grow, so the likely characters get more likely and the unlikely ones less; as $\tau$ gets close to 0, the most likely character gets a probability of almost 1. With $\tau$ above 1, the gaps shrink, so the probabilities get closer to equal. A character that never came stays impossible at every temperature: its logit is $\log 0$, minus infinity, and that divided by $\tau$ is still minus infinity.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **temperature**, $\tau$ | tau | a number you choose that divides the logits before softmax: below 1 sharper probabilities, above 1 flatter ones | `z / tau` |
        | logits of the bigram model | | the logarithms of its probabilities: softmax of them gives the probabilities back | `np.log(P[i])` |
        """),
        L.watch_out("The name comes from physics: in a hot gas, the particles spread over many energies, and in a cold one they stay close to the lowest. Here it has nothing to do with heat in the computer: the temperature is just a number you choose. You cannot divide by a temperature of 0; most programs take 0 to mean greedy decoding."),
        mo.md(r"""
        The picture: move the slider. The bars are the six most likely characters after "t" at that temperature, and below them is what the model of the plays writes with it, starting with "T", with seed 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_tau_cell():
    def _frame(tau):
        tau = round(tau, 1)
        text = C.sample_text(tau)
        names = {"␣": "the space"}   # what a screen reader should say for ␣
        bars = ", ".join(f"{names.get(c, c)} {round(p * 100)}%" for c, p in C.after("t", tau, 6))
        return (C.bars("t", tau, {"title": f"After \"t\", at τ = {L.number(tau, 1)}", "alt": f"Six bars, the most likely characters after t at τ = {L.number(tau, 1)}, on a fixed scale where a full bar is probability 1: {bars}."})
                + C.text_block(text),
                f"""Of the {len(text) - 1} draws after the "T", <span class="ea-num">{C.most_likely(text)}</span> are the most likely character after the one before.""")

    explore_tau = L.explore("tau", "τ", _frame, start=1, low=0.1, high=2.5, step=0.1, role="hyper", task="What goes wrong in the text at $\\tau$ = 0.3, and what at $\\tau$ = 2.5?")
    explore_tau
    return


@app.cell(hide_code=True)
def text_after_picture():
    mo.md(r"""
    At $\tau$ = 0.3, 115 of the 159 draws are the most likely character, almost three in four, and the text keeps falling back into "the the": it is too safe, and closer to 0, sampling becomes greedy decoding. At $\tau$ = 2.5, the text is too bold: capitals in the middle of a word, question marks, an apostrophe here and there. In between, the text has both: mostly likely characters, and now and then a less likely one. Which temperature is best depends on what you want: for an answer with one right form, low; for something new, higher.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To draw the next character at temperature `tau`, from the probabilities `p` in a row:

    1. Take the logits of the row: `z = np.log(p)`.
    2. Divide them by the temperature: `z = z / tau`.
    3. Softmax, as in [Many classes: softmax](/course/open/ml-softmax?lang=en): `e = np.exp(z - z.max())`, and then `e / e.sum()`.
    4. Draw the next character with these probabilities instead of `p`: `rng.choice(len(p), p=...)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A row with three characters and the probabilities 0.5, 0.3 and 0.2, at the temperatures 0.5, 1 and 2. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_temperature():
    import numpy as np

    _p = np.array([0.5, 0.3, 0.2])   # a row with three characters
    for _tau in [0.5, 1, 2]:
        # 1. the logits
        _z = np.log(_p)
        # 2. divided by the temperature
        _z = _z / _tau
        # 3. softmax
        _e = np.exp(_z - _z.max())
        _q = _e / _e.sum()
        print(_tau, _q)
    # 4. ten draws with the probabilities of the last temperature, 2
    _rng = np.random.default_rng(0)
    print(_rng.choice(len(_p), size=10, p=_q))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At 0.5, the first character goes from 0.5 to 0.66 and the last from 0.2 to 0.11: sharper. At 1, the probabilities stay as they were. At 2, they move towards a third each: 0.42, 0.32 and 0.26. In the ten draws at temperature 2, the last character, number 2, comes three times.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Pick a temperature
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-tau",
        question="Your model makes too many odd letter combinations, but greedy decoding gets stuck. Which temperature do you try first?",
        options={
            "zero": "0: always the most likely character",
            "one": "1: the probabilities as they are",
            "right": "0.7: the likely characters a bit more likely",
            "two": "2: all characters that ever came, closer to equal",
        },
        correct="right",
        why={
            "zero": "A temperature of 0 is greedy decoding, and that gets stuck.",
            "one": "At 1, nothing changes: the odd combinations stay.",
            "two": "Above 1, the text gets bolder: even more odd combinations.",
            "right": "Below 1 but not close to 0: the likely characters win more often, and the model still draws.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_function():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Probabilities at a temperature

        `with_temperature(p, tau)` gives the probabilities `p` at temperature `tau`. The logits and the last line are there. Replace the `...` with steps 2 and 3 of the recipe, in two lines: first the new `z`, divided by `tau`, then `e`. The check calls `with_temperature(np.array([0.6, 0.3, 0.1]), 0.5)`, which should give about `[0.783, 0.196, 0.022]`, and tries other temperatures and a probability of 0.
        """),
        L.task("Make `with_temperature` divide the logits by `tau` and take `e` to the power of each, with the largest taken off first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_function():
    box_temperature = L.code_box("with-temperature", lines=9)
    box_temperature
    return (box_temperature,)


@app.cell(hide_code=True)
def check_function(box_temperature):
    result_temperature = L.check("with-temperature", box_temperature.value)
    result_temperature
    return (result_temperature,)


@app.cell(hide_code=True)
def hints_function():
    L.hints(
        "with-temperature",
        [
            "Steps 2 and 3 of the recipe, as in the worked example.",
            "First a new `z`: the old one divided by the temperature. Then `e`: `np.exp` of `z`, with `z.max()` taken off first.",
            "`z = z / ___` and `e = np.exp(z - ___)`.",
        ],
        why="The logits of `[0.6, 0.3, 0.1]` are about −0.51, −1.20 and −2.30. Divided by 0.5 they become −1.02, −2.41 and −4.61: the gaps are twice as big. Softmax of those gives about 0.783, 0.196 and 0.022.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_never():
    mo.md(r"""
    ### Exercise 3. A character that never came
    """)
    return


@app.cell(hide_code=True)
def choice_never_cell():
    choice_never = L.choice(
        "never-seen",
        question="After \"q\", only \"u\" ever came in the plays: its row has a 1 for \"u\" and a 0 for the other 64 characters. What does a temperature of 2 make of that row?",
        options={
            "right": "Nothing changes: still 1 for \"u\" and 0 for the rest",
            "spread": "It spreads a little: the other 64 get a small chance",
            "half": "\"u\" gets 0.5: dividing by 2 halves it",
        },
        correct="right",
        why={
            "spread": "The other 64 have a logit of log 0, minus infinity, and divided by 2 that is still minus infinity: their probability stays 0. See *The idea*.",
            "half": "The temperature divides the logits, not the probabilities, and softmax makes them add up to 1 again. See *The idea*.",
            "right": "A temperature only reshapes the probabilities of the characters that ever came. One that never came keeps a logit of minus infinity, so its probability stays 0, and \"u\" keeps all of it.",
        },
    )
    choice_never
    return (choice_never,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en) turns logits into probabilities; dividing the logits by a temperature first makes them sharper or flatter, and the sampling of [Sampling: let the model write](/course/open/lm-sampling?lang=en) then draws from those. Every language model that writes, up to the largest LLM, has this setting. Your bigram model still gives a probability of 0 to every pair it never saw, and no temperature changes that. An extra lesson, *Top-k and top-p sampling*, shows two more ways to keep the unlikely characters out. In the next lesson, *How good is a language model?*, you measure how surprised your model is by real text; after that, in *Pairs you never saw: smoothing*, you see why those zeros are a problem, and fix them.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_never, choice_pick, result_temperature):
    L.wrap_up(choice_pick, result_temperature, choice_never)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
