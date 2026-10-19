# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson res-figures. Route: course/route/lessen/onderzoek.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("res-figures", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) and [Is the difference real?](/course/open/res-significance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="Which line says what the y-axis of a chart shows?",
        options={
            "title": "`plt.title(\"validation loss\")`",
            "right": "`plt.ylabel(\"validation loss\")`",
            "xlabel": "`plt.xlabel(\"validation loss\")`",
        },
        correct="right",
        why={
            "title": "The title goes above the chart and says what the whole chart is about. See [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "xlabel": "`xlabel` names the axis along the bottom. See [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "right": "`ylabel` writes its text along the y-axis, the one that goes up.",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_small():
    quick_small = L.choice(
        "quick-small",
        question="A change lowers the loss by 0.006, and the test gives p = 0.02. What does that say about the size of the change?",
        options={
            "big": "It is big: the smaller p, the bigger the change",
            "right": "Nothing: p says how well chance explains it",
            "percent": "That the loss is about 2% lower, as p = 0.02 says",
        },
        correct="right",
        why={
            "big": "p says how badly chance alone explains the difference, not how big it is: with more seeds, the same difference gets a smaller p. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "percent": "0.02 is the p-value, not the size of the change. See [Is the difference real?](/course/open/res-significance?lang=en).",
            "right": "The size is the 0.006 itself, next to what else moves the loss. A figure can show both at once.",
        },
    )
    quick_small
    return (quick_small,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Readers often look at the figures of a paper before they read its text, and some read nothing else. A figure is your result in the form most people will remember. A clear one shows what you found in a few seconds; a careless one shows something else, without a single wrong number in it.

    **An everyday example.** A weather map shows tomorrow at a glance: the rain in the north, the sun in the south. The same forecast as a table of numbers per town says just as much, but nobody sees it at a glance. And a map that colours 18 degrees deep red makes a mild day look like a heatwave, with every number right.
    """)
    return


@app.cell(hide_code=True)
def predict_bars():
    guess_bars = L.guess(
        "bars",
        question="The first chart in The idea below draws the course's runs with and without warmup as two bars, each the mean of ten seeds. The bar without warmup is about half as tall. How much lower is its loss really?",
        options={
            "half": "About half",
            "tenth": "About a tenth",
            "right": "Less than 1%",
        },
        reveal="**Less than 1%.** The means are 2.402 with warmup and 2.396 without: 0.006 apart, a quarter of a percent. The y-axis starts at 2.390, not at 0, so the bars show only their tops, and those differ by a factor of two. And each bar hides the ten runs behind it.",
    )
    guess_bars
    return (guess_bars,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A clear figure has **one message**, and everything in it serves that message. Say the message in the title, or in the first sentence of the caption, the text under a figure that you read first in [Reading an abstract, a figure and a results table](/course/open/res-figure-and-table?lang=en). Label each axis with what it shows, and with its unit when it has one; when an axis has a name at each group, those names say it. Show the spread over seeds: every run as a point, or the spread around the mean, not only the mean. Put what you compare side by side, on one axis with one scale, so the eye compares lengths or heights directly. A bar is read by its length, so a bar chart starts at 0; points and lines may start higher, if the axis says where.

        Name each group, along the axis or in a **legend**, a short list in the chart that says which mark is which: in Matplotlib, give each `plt.plot` or `plt.scatter` a `label="..."` and call `plt.legend()`. And use colour only when it means something, such as one colour per setting.

        The pictures: the same runs twice. First as two bars of the means on an axis that starts at 2.390: the difference looks huge, and the runs are gone. Then every run as a point, the two settings side by side on one axis, with the mean as a short line: the groups overlap, and without warmup they lie a little lower.
        """),
        mo.Html(C.before_after({"bad": "Misleading", "good": "Clear", "y": "validation loss", "with": "with warmup", "without": "without", "lang": L.lang,
                                "alt_bad": "Two bars, the means with warmup, 2.402, and without, 2.396, on a y-axis from 2.390 to 2.404: the bar without warmup is half as tall.",
                                "alt_good": "The same runs as points, ten with warmup from 2.386 to 2.431 and ten without from 2.382 to 2.424, side by side on one axis from 2.365 to 2.440, each with its mean as a short line. The two groups overlap."})),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **legend** | the short list in a chart that says which mark or colour is which group |
        """),
        L.watch_out("An axis that does not start at 0 is fine for points and lines, if its numbers are there to read. For bars it misleads, because a reader compares their lengths."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make a figure that shows something:

    1. Choose its one message, and write it as the title or the first sentence of the caption.
    2. Label each axis with what it shows, and its unit if it has one; names along an axis label it themselves.
    3. Show every run, or the spread over seeds, not only the mean.
    4. Put what you compare side by side, on one axis with one scale; bars start at 0.
    5. Name each group, on the axis or in a legend; use colour only when it means something.
    6. Write a caption: what each mark is, how many seeds, and the message.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The result of [Fair comparisons](/course/open/res-fair-baselines?lang=en) as a figure: the mean validation loss of the 2-layer and the 4-layer GPT at nine learning rates. `"o--"` draws points joined by a dashed line, `"o-"` by a solid one. The comments say which step each part is. It is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_figure():
    import matplotlib.pyplot as plt

    _rates = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    _two = [2.400, 2.215, 2.089, 1.962, 1.836, 1.791, 1.779, 1.801, 2.191]
    _four = [2.332, 2.124, 2.005, 1.897, 1.805, 1.784, 1.777, 1.812, 2.787]
    # 1. the one message, as the title
    plt.title("Tuned the same way, 4 layers end where 2 layers do")
    # 2. each axis says what it shows; the rates grow by factors, so a log scale
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("validation loss after 6 epochs, mean of 3 seeds")
    # 4-5. both models on one axis, each line with a name
    plt.plot(_rates, _two, "o--", label="2 layers")
    plt.plot(_rates, _four, "o-", label="4 layers")
    # complete: plt.legend(), so the reader sees which line is which
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Without `plt.legend()`, the reader sees two lines and cannot tell which model is which: the labels are there, but nothing shows them. With it, a short list in a corner names the dashed and the solid line. Step 3 is missing here: each point is the mean of three seeds, and the picture in Fair comparisons drew the single seeds too.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_points():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every run, side by side

        Draw the course's runs with and without warmup, seeds 0 to 9, as a figure that shows something. Every run is a point; the runs with warmup stand at the name `"with warmup"` along the x-axis, those without at `"without warmup"`. `["with warmup"] * len(with_warmup)` is that name once for every run, as `*` repeats a list, so `plt.scatter(["with warmup"] * len(with_warmup), with_warmup)` puts all of `with_warmup` at that name. Replace the `...`, and fill in a title with the one message and a label for the y-axis. A mean drawn on top as a point may stay.
        """),
        L.task("Draw every run of `with_warmup` and `without_warmup` as a point, each setting at its own name along the x-axis, with a title and a label on the y-axis."),
    ])
    return


@app.cell(hide_code=True)
def exercise_points():
    box_points = L.code_box("seed-points", lines=10)
    box_points
    return (box_points,)


@app.cell(hide_code=True)
def check_points(box_points):
    result_points = L.check("seed-points", box_points.value)
    result_points
    return (result_points,)


@app.cell(hide_code=True)
def hints_points():
    L.hints(
        "seed-points",
        [
            "Two `plt.scatter` lines, one per setting. The x values are the name of the setting, as many times as there are runs, so all runs of a setting stand above that name.",
            "`plt.scatter([\"with warmup\"] * len(with_warmup), ___)` and `plt.scatter([___] * len(without_warmup), without_warmup)`; then a title such as `\"Without warmup, the runs end a little lower\"`.",
        ],
        why="```python\nplt.scatter([\"with warmup\"] * len(with_warmup), with_warmup)\nplt.scatter([\"without warmup\"] * len(without_warmup), without_warmup)\nplt.title(\"Without warmup, the runs end a little lower\")\nplt.ylabel(\"validation loss after 6 epochs\")\nplt.show()\n```\n\nEach setting gets its own name along the x-axis, and every run its own point above it, so the reader sees the spread of each setting and the two side by side on one scale.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Which figure is fair?

        Three figures of the same runs: the 2-layer and the 4-layer GPT at their best learning rate, 0.03, seeds 0 to 2.
        """),
        mo.Html(C.three_ways({"two": "2 layers", "four": "4 layers", "y": "validation loss", "lang": L.lang,
                              "alt_a": "A: two bars of the means, 1.779 and 1.777, on a y-axis from 1.776 to 1.780; the bar of 4 layers is a fifth as tall.",
                              "alt_b": "B: two separate panels, each with its own y-axis, with the three runs of 2 layers and of 4 layers; both look equally spread.",
                              "alt_c": "C: the three runs of each model as points side by side on one y-axis from 1.765 to 1.795, with the means as short lines; the two groups overlap."})),
    ])
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair-figure",
        question="Which figure shows fairly how the 2-layer and the 4-layer GPT compare?",
        options={
            "a": "A: the bars make the difference easy to see",
            "b": "B: each model gets its own panel, so neither hides the other",
            "right": "C: every run on one axis, side by side",
        },
        correct="right",
        why={
            "a": "The bars start at 1.776, so a difference of 0.002 looks like a factor of five, and the three runs behind each mean are gone.",
            "b": "Each panel has its own y-axis, so the eye cannot compare the two levels at all: the same height means another loss in each panel. And the runs of 4 layers look as spread as those of 2 layers, though they lie four times closer together.",
            "right": "One axis, one scale, every run: the reader sees that the two groups overlap, which is the honest message.",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_caption():
    mo.md(r"""
    ### Exercise 3. A caption
    """)
    return


@app.cell(hide_code=True)
def explain_caption():
    explain_caption_cell = L.explain(
        "caption",
        question="Write the caption for your chart of Exercise 1, as it would stand under a figure in a paper: what each point is, the setup and the seeds, and the message in one sentence.",
        model_answer="**Figure 1. Without warmup, the runs end a little lower.** Validation loss after 6 epochs of the small GPT of Change one thing at a time, on the first quarter of Tiny Shakespeare, with a warmup of 100 steps and without; each point is one run, seeds 0 to 9 for both settings. Without warmup the loss is 0.006 lower on average, and lower for each of the ten seeds, which is small next to the spread of about 0.04 between seeds.",
        points=[
            "you say the message in one sentence, at the start",
            "you say what each point is, and the setup with its seeds",
            "you give the size of the difference next to the spread, so a reader can judge it",
        ],
    )
    explain_caption_cell
    return (explain_caption_cell,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The charts themselves come from [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en); the runs and their spread from [One run is not a result: seeds and variance](/course/open/res-seeds-variance?lang=en) and [Is the difference real?](/course/open/res-significance?lang=en). This lesson closes the part on experiments: you can now change one thing, take a part out, repeat with seeds, keep a log, compare fairly, test a difference and show it. Later in the route, you read papers more closely and reproduce one yourself; its figures follow this recipe.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, explain_caption_cell, result_points):
    L.wrap_up(result_points, choice_fair, explain_caption_cell)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
