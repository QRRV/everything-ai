# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson lm-bigram-matrix. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("lm-bigram-matrix", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) and [Broadcasting](/course/open/np-broadcasting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_probs():
    quick_probs = L.choice(
        "quick-probs",
        question="Your `bigram_probs(counts, \"t\")` gives a dictionary. What is in it?",
        options={
            "right": "Each character that ever follows \"t\", with its probability",
            "counts": "Each pair that starts with \"t\", with the number of times it occurs",
            "one": "The one character that most often follows \"t\"",
        },
        correct="right",
        why={
            "counts": "Those are the counts that `count_bigrams` gives; `bigram_probs` divides them by their total. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "one": "A language model gives a probability for every possible next token, not one answer. See [A bigram model by counting](/course/open/lm-bigram-counts?lang=en).",
            "right": "The counts of the pairs that start with \"t\", each divided by their total: together they add up to 1.",
        },
    )
    quick_probs
    return (quick_probs,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`N` has the shape (65, 65). What shape does `N.sum(axis=1, keepdims=True)` have?",
        options={
            "flat": "(65,): one sum per row, and the summed axis is gone",
            "cols": "(1, 65): one sum per column",
            "right": "(65, 1): one sum per row, as a column",
        },
        correct="right",
        why={
            "flat": "That is `N.sum(axis=1)` without `keepdims`: the axis disappears. See [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "cols": "`axis=1` adds up along each row, so there is one sum per row, not per column. See [Broadcasting](/course/open/np-broadcasting?lang=en).",
            "right": "One sum per row, and `keepdims=True` keeps the summed axis with length 1, so `N / sums` divides each row by its own sum.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your `bigram_probs` works out the probabilities after one character, and goes over all 1,403 pairs to do it. To let your model write, as in the lessons after this one, it needs the probabilities after the last character again for every character it writes. Better to work them all out once and keep them in one table: 65 rows and 65 columns, one row for each character. One line of NumPy then turns every row of that table into a probability distribution. This is still the second step of your own LLM.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           1, here_label="← today", title="Where your project stands: your own LLM"),
        L.course_versions(),
        mo.md(r"""
        **An everyday example.** The distance table at the back of a road atlas: a row and a column for every city, and in each cell the distance between the two. You look up any pair at once, instead of working it out each time. That table reads the same both ways; the table of this lesson does not.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_row():
    guess_row = L.guess(
        "space-row",
        question="In the table of probabilities, the row of a character holds the probabilities of what comes after it, for all 65 characters. After a space, a new word starts. How large do you think the largest number in the row of the space is?",
        options={
            "right": "About 0.15: many different letters often start a word",
            "half": "About 0.4: one letter starts nearly half of all words",
            "high": "About 0.8: nearly every word starts with the same letter",
        },
        reveal=f"**About 0.15: {L.number(C.top_after(C.course_probs(), ' ', 1)[0][1], 2)}, for \"t\".** Words start with many different letters, so the row of the space is spread out: \"a\", \"s\" and \"h\" each come after a space about half as often as \"t\". The row of \"q\" is the opposite: after a \"q\" comes a \"u\" every time, so that row has one cell of 1 and 64 cells of 0. A row shows how sure the model is of what comes next: unsure after a space, sure after a \"q\".",
    )
    guess_row
    return (guess_row,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Put the 65 characters in a fixed order, `chars = sorted(set(text))`: the character `chars[i]` gets row `i` and column `i`. In row $i$ and column $j$, the **count matrix** $N$ holds how often `chars[j]` follows `chars[i]`. A pair that never occurs gets a 0. Row $i$ holds all the counts after `chars[i]`.

        Divide every number by the sum of its row, and row $i$ becomes the probability distribution of the next character after `chars[i]`: its numbers add up to 1.
        """),
        L.formula(r"[[pred:P]][i, j] = \frac{[[data:N]][i, j]}{\text{the sum of row } i \text{ of } [[data:N]]}"),
        mo.md(r"""
        In NumPy that is one line, with the broadcasting of [Broadcasting](/course/open/np-broadcasting?lang=en): `N.sum(axis=1, keepdims=True)` gives the sum of every row, with the shape (65, 1), and `N / sums` divides every row by its own sum. The probabilities after a character are then a row of the matrix: `P[i]`, without going over the pairs again. This needs at least one count in every row: a character that is never followed by anything, such as a last character that occurs nowhere else in the text, gets a row of zeros, and 0 / 0 gives `nan`, "not a number". In the plays, every character is followed by something at least once.

        A **heatmap** shows a matrix as a grid of squares, coloured more strongly where the number is larger, so that you see its pattern at a glance.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **count matrix** | | the counts of every pair as a table: row $i$, column $j$ is how often `chars[j]` follows `chars[i]` | `N[i, j]` |
        | $P[i, j]$ | P i j | the probability that `chars[j]` comes after `chars[i]`: row $i$ of $N$ divided by its sum | `P[i, j]` |
        | **heatmap** | | a picture of a matrix: a square per number, coloured more strongly where it is larger | |
        """),
        L.watch_out("Rows and columns are not interchangeable here. Row `i` is what comes *after* `chars[i]`, and it adds up to 1. A column can add up to anything: column `j` holds how likely `chars[j]` is after each character, and those numbers belong to different distributions."),
        mo.md(r"""
        The picture: the heatmap of the rows and columns of the space and the 26 small letters, from all of the plays. The full matrix has 65 rows and 65 columns; capitals, signs and the newline are left out here, so that the letters can be read.
        """),
        mo.Html(C.heatmap(C.course_probs(), {"title": "What comes after what", "next": "the next character", "rows": "Each row: the character before",
                                              "alt": "A grid of 27 by 27 squares for the space and the letters a to z, the more likely, the stronger the colour. The row of q has one strongly coloured square, under u. The row of the space is faint and spread over many letters, with its strongest square under t."})),
        mo.md(r"""
        The row of "q" has one strongly coloured square, under "u": after a "q" always comes a "u". The row of the space is faint and spread out: a word can start with almost any letter.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To turn the bigram counts into a matrix of probabilities:

    1. Make the count matrix: for every character `a` of `chars` a row, and in it, for every character `b`, the count of the pair `a + b`, or 0 when it never occurs.
    2. Turn the list of rows into a NumPy array: `np.array(rows)`.
    3. Add up every row, keeping the axis: `sums = N.sum(axis=1, keepdims=True)`.
    4. Divide: `P = N / sums`. Row `i` of `P` is the distribution of what comes after `chars[i]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The matrix of the short text "abab ba", with its three characters, " ", "a" and "b". The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_matrix():
    import numpy as np

    _counts = {"ab": 2, "ba": 2, "b ": 1, " b": 1}   # the pairs of "abab ba"
    _chars = [" ", "a", "b"]
    # 1. a row for every character, with the count of each pair, or 0
    _rows = []
    for _a in _chars:
        _row = []
        for _b in _chars:
            if _a + _b in _counts:
                _row.append(_counts[_a + _b])
            else:
                _row.append(0)
        _rows.append(_row)
    # 2. the rows as an array
    _N = np.array(_rows)
    print(_N)
    # 3. the sum of every row, as a column
    _sums = _N.sum(axis=1, keepdims=True)
    print(_sums)
    # 4. every row divided by its own sum
    _P = _N / _sums
    print(_P)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The loop over `_b` runs inside the loop over `_a`: for every character `_a`, it goes over all three characters `_b` and fills one row, so `_rows` gets three rows of three numbers. The row of "b" holds 1, 2 and 0: after "b" came a space once and "a" twice. Its sum is 3, so it becomes 0.33, 0.67 and 0. The row of "a" only has the pair "ab", twice, so it becomes 0, 0, 1. Every row of `_P` adds up to 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_q():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Read the heatmap
    """)
    return


@app.cell(hide_code=True)
def choice_q_cell():
    choice_q = L.choice(
        "space-column",
        question="In the heatmap of the idea, the column of the space is strongly coloured in many rows. What does that say?",
        options={
            "starts": "Words often start with those letters",
            "right": "Those letters often come at the end of a word",
            "space": "After a space, those letters are more likely than the others",
        },
        correct="right",
        why={
            "starts": "A column is what comes next: a strongly coloured square in the space's column means a space comes after the letter of that row. See the watch-out in *The idea*.",
            "space": "What comes after a space is in the row of the space, not in its column. See *The idea*.",
            "right": "A strongly coloured square in row \"e\", column ␣, means that after an \"e\" a space often comes: the \"e\" often ends a word.",
        },
    )
    choice_q
    return (choice_q,)


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The count matrix

        `bigram_matrix(counts, chars)` makes the count matrix: a row for every character of `chars`, with the count of each pair. The loops are there. Replace the `...` with step 1 of the recipe. The check calls `bigram_matrix({"ab": 3, "ac": 1, "ba": 2}, ["a", "b", "c"])`, which should give `[[0, 3, 1], [2, 0, 0], [0, 0, 0]]`, and tries other counts and another order of `chars`. When your function is right, the check saves it in your project, as `work/projects/llm/bigram_matrix.py`.
        """),
        L.task("Make `bigram_matrix` put the count of the pair `a + b` in the row, or 0 when the pair never occurs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("bigram-matrix", lines=12)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("bigram-matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "bigram-matrix",
        [
            "Step 1 of the recipe, as in the worked example.",
            "Most pairs are not in `counts`. Look first whether `a + b` is in it: then its count goes into the row, and otherwise a 0.",
            "`if a + b in counts:`, then `row.append(counts[___])`, and after `else:` the line `row.append(___)`.",
        ],
        why="With `[\"a\", \"b\", \"c\"]`, the row of \"a\" gets the counts of \"aa\", \"ab\" and \"ac\": 0, 3 and 1. The row of \"b\" gets 2 for \"ba\" and 0 twice. No pair starts with \"c\", so its row is all 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Every row a distribution

        `to_probs(N)` divides every row of a count matrix by its sum. Replace the `...` with step 3 of the recipe. The check calls `to_probs(np.array([[3, 1], [0, 2]]))`, which should give `[[0.75, 0.25], [0, 1]]`, and tries other matrices. When your function is right, the check saves it in your project, as `work/projects/llm/bigram_rows.py`.
        """),
        L.task("Make `to_probs` divide every row of `N` by its own sum."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("to-probs", lines=5)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("to-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "to-probs",
        [
            "Step 3 of the recipe: the sum of every row, in a shape that broadcasting lines up with the rows.",
            "A row runs along axis 1. Without `keepdims`, the sums have the shape (rows,), and broadcasting pairs them with the columns instead.",
            "`sums = N.sum(axis=___, keepdims=___)`.",
        ],
        why="`[[3, 1], [0, 2]]` has the row sums 4 and 2, as a column `[[4], [2]]`. Dividing gives `[[3/4, 1/4], [0/2, 2/2]]`: `[[0.75, 0.25], [0, 1]]`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below counts the pairs of the plays with your `count_bigrams` of [A bigram model by counting](/course/open/lm-bigram-counts?lang=en), or the course's when yours is missing or broken, makes the count matrix with your `bigram_matrix`, and turns it into probabilities with your `to_probs`. It works once exercises 2 and 3 are done.
    """)
    return


@app.cell(hide_code=True)
def demo_table(result_matrix, result_probs):
    result_matrix, result_probs  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `bigram_matrix` and your `to_probs` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Building the table with your functions takes more than 30 seconds. Does each go once over the text or the characters?*",
            "counts": "*Your `count_bigrams` in `work/projects/llm/bigrams.py` stops with an error, or no longer counts the pairs of the plays the way the course's does. Did you change that file? Delete it, and check exercise 2 of [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) again: its check then saves your function again.*",
            "matrix": "*The `bigram_matrix` in `work/projects/llm/bigram_matrix.py` no longer gives the counts the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.*",
            "rows": "*The `to_probs` in `work/projects/llm/bigram_rows.py` no longer gives the probabilities the check expects, or stops with an error. Did you change that file? Delete it, and check exercise 3 again: its check then saves your function again.*",
        }[_result])
    else:
        _table, _yours = _result
        _out = mo.vstack([
            mo.md(f"""Your table has the shape {_table.shape}, and every row adds up to 1. The probabilities after a "t", read from row "t" of your table:"""),
            draw.next_word_bars("t", C.top_after(_table, "t"), picked=0, picked_label="most likely", title="Your table, row \"t\""),
            mo.md("""The same numbers as your `bigram_probs` gave in the lesson before, but now they are all ready, for every character at once. In the next lesson, your model writes with them."""
                  if _yours else """The counts come from the course's `count_bigrams`: yours is missing or broken, as the note under *Where your project stands* says. The table is still built by your `bigram_matrix` and your `to_probs`. In the next lesson, your model writes with it."""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The counts of [A bigram model by counting](/course/open/lm-bigram-counts?lang=en) are now one matrix, and the broadcasting of [Broadcasting](/course/open/np-broadcasting?lang=en) turns all its rows into probability distributions at once. Your project `llm` now has `bigram_matrix.py` and `bigram_rows.py` as well. In the next lesson, *Sampling: let the model write*, your model draws the next character from a row of this table, again and again, and writes text of its own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_q, result_matrix, result_probs):
    L.wrap_up(choice_q, result_matrix, result_probs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
