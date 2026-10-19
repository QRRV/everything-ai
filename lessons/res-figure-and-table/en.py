# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-figure-and-table. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-figure-and-table", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading AI claims critically](/course/open/res-reading-claims?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="A model scores 2 points higher than another model on a benchmark. What does that score tell you?",
        options={
            "better": "That it is the better model",
            "nothing": "Nothing: benchmark scores mean nothing",
            "right": "That it does better on those tasks, scored that way",
        },
        correct="right",
        why={
            "better": "Better on that benchmark, not in general. See [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
            "nothing": "A score means something: how the model does on those tasks. Only not more than that. See [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
            "right": "A benchmark score is a measurement on those tasks; whether that carries over to what you want to use the model for, the benchmark does not tell you.",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def quick_check_runs():
    quick_runs = L.choice(
        "quick-runs",
        question="Two models, each trained once, end with a validation loss of 0.31 and 0.33. What can you say?",
        options={
            "right": "Not much yet: with one run of each, the gap may be chance",
            "lower": "The first model is better: its validation loss is lower",
            "same": "They are exactly as good as each other: the gap is small",
        },
        correct="right",
        why={
            "lower": "Lower in this one run. Curves jump a little from epoch to epoch, so another run can end a little higher or lower. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "same": "One run of each cannot show that either: the gap may be real or chance. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "Each curve is one run, and curves jump a little, so a gap of 0.02 may be chance. Several runs of each would show how much they spread.",
        },
    )
    quick_runs
    return (quick_runs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A paper reports its results in three places. The **abstract** says what the paper claims, the figures show how numbers change, and the results table compares methods. Many misreadings of AI results come from reading the claim before the axes, the units and the conditions: a difference that looks huge in a figure can be tiny in numbers, and a row that wins in a table can have been measured on an easier test.

    **An everyday example.** Before you judge whether two towns on a map are close, you look at the scale bar: two centimetres can be two kilometres, or two hundred.
    """)
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "gap",
        question="A figure in a paper shows the loss of two models over 100 training steps, on a log-scale y-axis. The two lines run side by side, with the same gap between them all the way. On a normal scale, what do you think happens to the gap between the two losses?",
        options={
            "same": "It stays the same all the way",
            "right": "It shrinks as both losses fall",
            "grows": "It grows as training goes on",
        },
        reveal="**It shrinks as both losses fall.** On a log scale, the same distance means the same factor: here one loss is always, say, twice the other. At the start that is 2 against 1, a gap of 1; at the end 0.02 against 0.01, a gap of 0.01. The figure shows a constant ratio, not a constant difference. That is why you read the scale of an axis before you read anything into the lines.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **abstract** is the short summary at the top of a paper: the problem, the approach, the main result, and why it matters, in one paragraph. Read it as the claim, as in [Reading AI claims critically](/course/open/res-reading-claims?lang=en): it tells you what to look for in the figures and tables, where the measurements are.

        **Reading a figure** goes in a fixed order, before you look at what the lines do:

        1. **The caption**, the text under the figure: what is shown, on which data, and how often it was measured, in one run or as the mean of several.
        2. **The axes**: which quantity, in which unit, on a normal or a log scale, and where each axis starts. An axis that does not start at 0 makes small differences look big.
        3. **What is compared**: which line or bar is which, and whether they were measured in the same way.
        4. **Then the claim**: what the figure shows, and what it does not.

        **A results table** has the methods as rows and the measurements as columns, often with the best number in bold. Only rows measured under the same conditions can be compared: the same test set, the same metric, the same amount of training data. A number taken from another paper may have been measured differently; the caption or a footnote should say so.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **abstract** | | the short summary at the top of a paper: problem, approach, main result and why it matters | |
        | **caption** | | the text under a figure or above a table that says what it shows | |
        | **results table** | | a table with methods as rows and measurements as columns, to compare them | `rows` |
        """),
        L.watch_out("A difference between two numbers is not yet a real difference. When each number comes from one run, a second run with another seed can come out a little different. A paper that gives the mean of several runs, with how much they spread, lets you see whether a difference is more than chance."),
        mo.md(r"""
        The picture: the same two accuracies, 91.2% and 91.8%, twice. On the left, the axis starts at 91; on the right, at 0.
        """),
        mo.Html(C.bars_picture({"cut": "Axis from 91", "full": "Axis from 0", "y": "accuracy (%)", "point": ".",
                                "alt": "Two bar charts of the same two scores, A 91.2 and B 91.8. On the left, the axis runs from 91 to 92, and B's bar is four times as tall as A's. On the right, the axis runs from 0 to 100, and the two bars look the same height."})),
        mo.md(r"""
        On the left, B's bar is four times as tall as A's; on the right, you can hardly see a difference. Both are true pictures of the same 0.6 percentage points. Which one is honest depends on what the paper claims about it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a figure or a results table:

    1. Read the caption: what is shown, on which data, and from how many runs.
    2. Read the axes, or the columns of a table: which quantity, which unit, which scale, where an axis starts, and whether higher or lower is better.
    3. Check what is compared: which line, bar or row is which, and whether they were measured under the same conditions.
    4. Only then read the claim, and put it next to what the figure or table shows.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The left chart of the picture, as a made-up paper might show it, with the claim "model B is far better than model A".

    1. **Caption:** accuracy on a benchmark of 1,000 questions, one run per model.
    2. **Axes:** accuracy in percent, where higher is better, on a normal scale, but starting at 91, not at 0.
    3. **What is compared:** two models on the same benchmark, each measured once.
    4. **The claim:** B scored 91.8% and A 91.2%: 0.6 percentage points, or 6 of the 1,000 questions. "Far better" comes from the cut axis, not from the numbers. And with one run each, 6 questions can be chance.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example_table():
    mo.md(r"""
    The same steps on a results table, in code. A made-up table of four methods with their loss on the test set. Method C was measured only on a small, easy part of the test set. For a loss, lower is better, so the cell keeps the lowest loss so far, and starts higher than every loss in the table. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_table():
    # 1. the caption: the loss on the test set; method C only on a small part of it
    _rows = [
        ("Method A", "full", 0.84),
        ("Method B", "full", 0.79),
        ("Method C", "small", 0.52),
        ("Method D", "full", 0.81),
    ]
    # 2. the columns: method, test set, loss; lower is better
    _best_name = None
    _best_loss = 100
    for _name, _test_set, _loss in _rows:
        # 3. compare only the rows measured the same way: on the full test set
        if _test_set == "full":
            if _loss < _best_loss:
                _best_name = _name
                _best_loss = _loss
        print(_name, _test_set, _loss, "best so far:", _best_name)
    # 4. the best method under the same conditions
    _best_name
    return


@app.cell(hide_code=True)
def text_worked_example_table_result():
    mo.md(r"""
    `for _name, _test_set, _loss in _rows:` takes each row apart into three names, the way `x, y = point` does in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en). Method C has the lowest loss, 0.52, but on an easier test, so the outer `if` skips it. Among the rows on the full test set, method B is best, with 0.79: from that row on, `_best_name` stays `Method B`. A table that puts C's 0.52 in bold as the best number compares rows measured under other conditions.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_times():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Read a figure
        """),
        mo.Html(C.loss_picture({"title": "Training loss, one run", "x": "step", "y": "loss (log scale)",
                                "alt": "A straight line that falls from 2 at step 0 to 0.02 at step 100, on a log-scale y-axis with the gridlines 0.01, 0.1 and 1."})),
    ])
    return


@app.cell(hide_code=True)
def choice_times_cell():
    choice_times = L.choice(
        "times-smaller",
        question="In this figure, the loss falls from 2 at step 0 to 0.02 at step 100. How many times smaller is it at step 100 than at step 0?",
        options={
            "grid": "2 times: the line goes down two gridlines",
            "diff": "1.98 times: it goes down from 2 to 0.02",
            "right": "100 times: 2 divided by 0.02",
        },
        correct="right",
        why={
            "grid": "Each gridline on a log scale is a factor of 10, so two gridlines are a factor of 10 · 10. See [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "diff": "1.98 is how much it went down: a difference, not a factor. How many times does 0.02 fit in 2?",
            "right": "2 / 0.02 = 100: two steps of a factor of 10 on the log scale. A straight line on a log scale means the loss shrinks by the same factor every step.",
        },
    )
    choice_times
    return (choice_times,)


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The fair winner of a table

        A results table, as a paper might show it:

        | Method | Test set | Accuracy (%) |
        |---|---|---:|
        | Ours | `full` | 84.1 |
        | Model X | `full` | 82.7 |
        | Model Y | `easy half` | 89.5 |
        | Model Z | `full` | 83.9 |

        In the box, the same table is a list of rows: the name of a method, the test set it was measured on, and its accuracy. Model Y was measured only on the easy half of the test set. Replace the `...`: keep the name of the best row in `best_name` and its score in `best_score`, but only among the rows measured on the `"full"` test set. The last line shows `best_name`. The check also tries your code on another table.
        """),
        L.task("Find the best row of the table among the rows on the full test set."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fair():
    box_fair = L.code_box("fair-best", lines=9)
    box_fair
    return (box_fair,)


@app.cell(hide_code=True)
def check_fair(box_fair):
    result_fair = L.check("fair-best", box_fair.value)
    result_fair
    return (result_fair,)


@app.cell(hide_code=True)
def hints_fair():
    L.hints(
        "fair-best",
        [
            "Step 3 of the recipe: only rows measured under the same conditions count. Then keep the best of those, as the worked example keeps the best row; here higher is better.",
            "First an `if` that lets through only the rows on the full test set. Inside it, a second `if`: is this score higher than the best so far? Then keep both the name and the score.",
            "`if test_set == \"full\":`, then inside it `if score > best_score:`, and under that `best_name = name` and `best_score = ___`.",
        ],
        why="Among the rows on the full test set, \"Ours\" has 84.1, Model Z 83.9 and Model X 82.7. Model Y's 89.5 is higher, but it was measured on the easy half, so it does not count.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_abstract():
    mo.md(r"""
    ### Exercise 3. Read an abstract

    A made-up abstract: *"We introduce SparkNet, a small model for reading comprehension. On the QuickRead benchmark it reaches an accuracy of 84.1%, 0.2 points above the previous best model, while it needs half as much computation to train."*
    """)
    return


@app.cell(hide_code=True)
def explain_abstract():
    explain_read = L.explain(
        "abstract",
        question="What does this abstract claim? And what would you look for in the figures and tables of the paper to check it? Write it down.",
        model_answer="**The claims:** SparkNet gets 84.1% on QuickRead, 0.2 points more than the previous best model, with half as much computation. **To check:** a results table in which all rows were measured on the same test set of QuickRead, under the same conditions; whether the 0.2 points come from one run or the mean of several, and how much runs spread, because 0.2 points can be chance; and a figure or table with the computing power of each model.",
        points=[
            "you name the claims: 84.1%, 0.2 points above the previous best model, half as much computation",
            "you look for a results table with all rows under the same conditions",
            "you ask whether 0.2 points is more than the spread between runs",
        ],
    )
    explain_read
    return (explain_read,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Reading AI claims critically](/course/open/res-reading-claims?lang=en) asked what was measured; the figures and tables of a paper are where you find it. The log scale of [Log scales and power laws](/course/open/math-log-scales?lang=en) and the loss curves of [Reading loss curves](/course/open/torch-loss-curves?lang=en) are the figures you will meet most. In the next lesson, *How research works: question, hypothesis, experiment*, you see where the measurements come from: an experiment designed to test a hypothesis. A later lesson, *One run is not a result: seeds and variance*, measures how much runs spread.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_times, explain_read, result_fair):
    L.wrap_up(choice_times, result_fair, explain_read)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
