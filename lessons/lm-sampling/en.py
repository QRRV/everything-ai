# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-sampling. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-sampling", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en) and [Repeating until something happens: while](/course/open/py-while?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="`P` is your table of probabilities, and `chars[i]` is \"e\". Where are the probabilities of the character that comes after \"e\"?",
        options={
            "column": "In column `i`: `P[:, i]`",
            "right": "In row `i`: `P[i]`",
            "cell": "In the one number `P[i, i]`",
        },
        correct="right",
        why={
            "column": "Column `i` holds how likely \"e\" is after each character. See the watch-out of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "cell": "`P[i, i]` is only the probability that another \"e\" comes after \"e\". See [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en).",
            "right": "Row `i` is the distribution of the next character after `chars[i]`: its numbers add up to 1.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_while():
    quick_while = L.choice(
        "quick-while",
        question="How many times does this loop add an \"x\"?\n\n```python\ntext = \"ab\"\nwhile len(text) < 5:\n    text = text + \"x\"\n```",
        options={
            "five": "5 times",
            "two": "2 times",
            "right": "3 times",
        },
        correct="right",
        why={
            "five": "The text already has 2 characters, and the loop stops as soon as it has 5. See [Repeating until something happens: while](/course/open/py-while?lang=en).",
            "two": "After 2 times the text is \"abxx\", 4 characters, still shorter than 5, so the loop goes round once more. See [Repeating until something happens: while](/course/open/py-while?lang=en).",
            "right": "\"ab\" becomes \"abx\", \"abxx\" and \"abxxx\". Then `len(text) < 5` is `False`, and the loop stops.",
        },
    )
    quick_while
    return (quick_while,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your table holds, for every character, the probabilities of the next one. But a table does not write. To write, your model needs a rule: look up the row of the last character, choose a next character from it, add it to the text, and do the same again. Which character to choose is the question of this lesson: the most likely one each time, or one drawn at random, where likely characters come up more often? Every language model that writes, up to the largest LLM, has to make this choice at every token. This is still the second step of your own LLM: at the end of this lesson, your bigram model writes its first text.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           1, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** A raffle: someone with three of the four tickets wins three times out of four, but not every time. Now and then, the one other ticket wins; if the most tickets always won, it never would.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_greedy():
    guess_greedy = L.guess(
        "greedy-loop",
        question="Your model starts with \"t\" and always takes the most likely next character: after \"t\", that is \"h\". What do you think it writes, 30 characters long?",
        options={
            "english": "Text that keeps changing, like English",
            "right": "One short word, over and over",
            "random": "Letters in a random order",
        },
        reveal=f"**One short word, over and over:** `{C.greedy_text('t', 30)}`. After \"t\" the most likely character is \"h\", after \"h\" it is \"e\", after \"e\" a space, and after a space \"t\" again. Each time the model is back at \"t\", it does exactly the same: it is stuck in a loop.",
    )
    guess_greedy
    return (guess_greedy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Your model writes one character at a time. It looks up the row of the last character, chooses the next character from that row, and adds it to the text. Then that new character is the last one, and it does the same again. Writing this way, choosing each new token from the prediction for the text so far, is **autoregressive generation**; here every token is one character. Your bigram model only looks at the last character; the models of later lessons look at much more of the text, but they write the same way.

        How to choose? Always taking the most likely character is **greedy decoding**: `np.argmax(P[i])`, the place of the largest number in row `i`. That is `P[i].argmax()` from [Sums and means along an axis](/course/open/np-axis?lang=en), written as a function. Greedy decoding writes the same text every time, and, as you saw in the guess, it can get stuck in a loop.

        **Sampling** draws the next character instead, with the probabilities of the row: a character with probability 0.43 is drawn a bit less than half the time, and one with probability 0.01 about once in a hundred draws. The random number generator of [Random numbers and seeds](/course/open/np-random?lang=en) does this with `rng.choice`: `rng.choice(len(chars), p=P[i])` draws the number of a column: column `j` with probability `P[i, j]`. With the same seed, it draws the same numbers, so your model writes the same text again; with another seed, it writes another text.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **autoregressive generation** | | writing one token at a time: each new token is chosen from the prediction for the text so far, and added to it | `text = text + chars[j]` |
        | **greedy decoding** | | always taking the most likely next token | `j = np.argmax(P[i])` |
        | **sampling** | | drawing the next token with the probabilities the model gives: likely tokens come often, unlikely ones now and then | `j = rng.choice(len(chars), p=P[i])` |
        | `rng.choice(n, p=probs)` | | one of the numbers 0 to n − 1, drawn with the probabilities in `probs`, which add up to 1 | `rng.choice(3, p=[0.2, 0.5, 0.3])` |
        """),
        L.watch_out("`rng.choice(len(chars), p=P[i])` gives a number, the place of a column, not a character: the character is `chars[j]`. And `p` must add up to 1. A row of `P` does; a column does not, and then NumPy stops with an error."),
        mo.md(r"""
        The picture: five steps of sampling with the table of the plays, starting with "T", with seed 23.
        """),
        mo.Html(C.steps_picture("T", 23, 5, {"title": "Five steps of sampling", "text": "the text so far",
                                             "row": "the 4 most likely in the row of its last character", "drawn": "drawn",
                                             "alt": "Five rows, one per step. The text so far is T, Th, The, The with a space, and The a, each with its last character in bold. Next to it are the four most likely characters in the row of that last character, with their probabilities, and the one drawn is outlined: h after T, e after h, a space after e, a after the space where t was more likely, and r after a where n was more likely."})),
        mo.md(r"""
        Each step looks only at the last character, in bold. The first three draws were the most likely character, and the text became "The ". Then the model drew "a" after the space, not "t", the most likely one, and "r" after "a", not "n". Greedy decoding would have gone on with "The the"; sampling wrote "The ar", and goes on from there.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To let your model write `n` characters, starting with `start`:

    1. Make one random number generator, with a seed: `rng = np.random.default_rng(seed)`.
    2. Note the row of every character: `row_of[chars[k]] = k`, for every `k`.
    3. Start the text: `text = start`.
    4. While `len(text) < n`: find the row of the last character, `i = row_of[text[-1]]`; draw the next one, `j = rng.choice(len(chars), p=P[i])`; and add it, `text = text + chars[j]`.
    5. For greedy decoding, take the most likely character instead of drawing one: `j = np.argmax(P[i])`. Then you need no random number generator.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table of "abab ba" from the worked example of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en), with its three characters " ", "a" and "b". The cell writes 12 characters with it, starting with "a": first by sampling, then with greedy decoding. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_write():
    import numpy as np

    _P = np.array([[0, 0, 1], [0, 0, 1], [1/3, 2/3, 0]])   # the table of "abab ba"
    _chars = [" ", "a", "b"]
    # 1. one random number generator, with a seed
    _rng = np.random.default_rng(4)
    # 2. the row of every character
    _row_of = {}
    for _k in range(len(_chars)):
        _row_of[_chars[_k]] = _k
    # 3. the start
    _text = "a"
    # 4. draw the next character from the row of the last one, until there are 12
    while len(_text) < 12:
        _i = _row_of[_text[-1]]
        _j = _rng.choice(len(_chars), p=_P[_i])
        _text = _text + _chars[_j]
    print(_text)
    # 5. greedy decoding: always the most likely character
    _greedy = "a"
    while len(_greedy) < 12:
        _i = _row_of[_greedy[-1]]
        _j = np.argmax(_P[_i])
        _greedy = _greedy + _chars[_j]
    print(_greedy)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In "abab ba", only "b" ever came after "a" or a space, so both texts write "b" there. After "b", sampling drew "a" three times and a space twice. Greedy decoding always takes "a" after "b", the more likely one, so it writes `abababababab`. Run the cell again: with seed 4, the sampled text stays the same too.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_draw():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The line that draws
    """)
    return


@app.cell(hide_code=True)
def choice_draw_cell():
    choice_draw = L.choice(
        "draw-line",
        question="`i` is the row of the last character. Which line draws the next character with the probabilities of your model?",
        options={
            "argmax": "`j = np.argmax(P[i])`",
            "even": "`j = rng.integers(0, len(chars))`",
            "right": "`j = rng.choice(len(chars), p=P[i])`",
            "column": "`j = rng.choice(len(chars), p=P[:, i])`",
        },
        correct="right",
        why={
            "argmax": "That line always takes the most likely character: greedy decoding. It does not draw.",
            "even": "That line draws, but gives every character the same chance, whatever your model says.",
            "column": "Column `i` holds how likely `chars[i]` is after each character: not what comes after `chars[i]`, and its numbers do not add up to 1. See the watch-out in *The idea*.",
            "right": "It draws the number of a column: column `j` with probability `P[i, j]`. Likely characters come often, unlikely ones now and then.",
        },
    )
    choice_draw
    return (choice_draw,)


@app.cell(hide_code=True)
def text_exercise_greedy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Greedy decoding

        `greedy(P, chars, start, n)` writes `n` characters with greedy decoding. The loop is there. Replace the `...` with step 5 of the recipe. The check calls `greedy(P, ["a", "b", "c"], "a", 6)` with `P = np.array([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]])`, which should give `"abcabc"`, and tries other tables and starts. When your function is right, the check saves it in your project, as `work/projects/llm/greedy.py`.
        """),
        L.task("Make `greedy` take as `j` the place of the largest number in row `i`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_greedy():
    box_greedy = L.code_box("greedy", lines=13)
    box_greedy
    return (box_greedy,)


@app.cell(hide_code=True)
def check_greedy(box_greedy):
    result_greedy = L.check("greedy", box_greedy.value)
    result_greedy
    return (result_greedy,)


@app.cell(hide_code=True)
def hints_greedy():
    L.hints(
        "greedy",
        [
            "Step 5 of the recipe, as in the worked example.",
            "`np.argmax` gives the place of the largest number in an array. The row of the last character is `P[i]`.",
            "`j = np.argmax(___)`.",
        ],
        why="In row \"a\" of `P`, the largest number is 0.6, in column 1: \"b\". In row \"b\" it is 0.6, in column 2: \"c\", and in row \"c\" 0.8, in column 0: \"a\". So the text goes round and round: `abcabc`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_generate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Let your model write

        `generate(P, chars, start, n, seed)` writes `n` characters by sampling. The random number generator, the rows and the start are there. Replace the `...` with step 4 of the recipe: the `while` loop. The check calls `generate(P, ["a", "b", "c"], "a", 8, 1)`, with the `P` of exercise 2, which should give `"abcacabc"`, and tries other tables, starts, lengths and seeds. When your function is right, the check saves it in your project, as `work/projects/llm/sampling.py`.
        """),
        L.task("Make `generate` draw a character from the row of the last character and add it, until the text is `n` characters long."),
    ])
    return


@app.cell(hide_code=True)
def exercise_generate():
    box_generate = L.code_box("generate", lines=14)
    box_generate
    return (box_generate,)


@app.cell(hide_code=True)
def check_generate(box_generate):
    result_generate = L.check("generate", box_generate.value)
    result_generate
    return (result_generate,)


@app.cell(hide_code=True)
def hints_generate():
    L.hints(
        "generate",
        [
            "Step 4 of the recipe, as in the worked example: three lines inside a `while` loop.",
            "Each time round: the row of the last character, `text[-1]`; then the number of a column, drawn with `rng.choice` and `p=`; then the character with that number, added to `text`.",
            "`while len(text) < n:`, and inside it `i = row_of[___]`, `j = rng.choice(len(chars), p=___)` and `text = text + chars[___]`.",
        ],
        why="With seed 1, `rng.choice` draws the same numbers every time, so the check knows what your function should give. From \"a\" it drew \"b\", then \"c\" and \"a\", and then \"c\": after \"a\", \"c\" has probability 0.3, and greedy decoding would have taken \"b\".",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below makes the table of the plays with the `count_bigrams`, `bigram_matrix` and `to_probs` in your project, or the course's where yours is missing or broken (the note under *Where your project stands* says which), and lets your model write, starting with "T": 60 characters with your `greedy`, and 200 characters with your `generate`, once with seed 1 and once with seed 2. It works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_write(result_generate, result_greedy):
    result_greedy, result_generate  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `greedy` and your `generate` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Making the table or writing with your functions takes more than 30 seconds. Does each of your functions go over its text or characters once, and does each loop stop once the text is long enough?*",
            "table": "*The table of the plays does not come out right with the `count_bigrams`, `bigram_matrix` and `to_probs` in `work/projects/llm/`: not every row adds up to 1. Did you change one of those files? Delete it, and check its exercise again, in [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) or [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en): the check then saves your function again.*",
            "greedy": "*The `greedy` in `work/projects/llm/greedy.py` no longer gives what the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.*",
            "sampling": "*The `generate` in `work/projects/llm/sampling.py` no longer gives what the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*",
        }[_result])
    else:
        _out = mo.vstack([
            mo.md("Greedy decoding, with your `greedy`:"),
            mo.Html(C.text_block(_result['greedy'])),
            mo.md("Sampling, with your `generate` and seed 1:"),
            mo.Html(C.text_block(_result[1])),
            mo.md("And with seed 2:"),
            mo.Html(C.text_block(_result[2])),
            mo.md("""Greedy decoding is stuck after four characters. The sampled texts are not English, but from a distance they look like it: short words between spaces, lines that start with a capital, and a name with a colon now and then, as in the plays. A model that only looks at the last character can do little more. With the same seed, your model writes the same text every time; with another seed, another text."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The table of [The bigram table as a matrix of probabilities](/course/open/lm-bigram-matrix?lang=en) gives the probabilities, the random number generator of [Random numbers and seeds](/course/open/np-random?lang=en) draws from them, and the `while` loop of [Repeating until something happens: while](/course/open/py-while?lang=en) keeps going until the text is long enough. Your project `llm` now has `greedy.py` and `sampling.py` as well. In the next lesson, *Temperature: bolder or safer text*, you make the choice between safe and bold yourself: sharper probabilities, closer to greedy decoding, or flatter ones, closer to drawing at random.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_draw, result_generate, result_greedy):
    L.wrap_up(choice_draw, result_greedy, result_generate)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
