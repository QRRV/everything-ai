# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-two-columns. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-two-columns", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) and [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scatter():
    quick_scatter = L.choice(
        "quick-scatter",
        question="What does `plt.scatter(x, y)` draw?",
        options={
            "line": "A line through the points, in their order",
            "right": "A loose point for each pair of an x and a y",
            "bars": "A bar for each x, as high as its y",
        },
        correct="right",
        why={
            "line": "That is `plt.plot`. What does a scatter plot leave out? See [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "bars": "That is `plt.bar`. What does a scatter plot draw for each pair? See [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "right": "One loose point per pair: the first x with the first y, and so on.",
        },
    )
    quick_scatter
    return (quick_scatter,)


@app.cell(hide_code=True)
def quick_check_biscoe():
    quick_biscoe = L.choice(
        "quick-biscoe",
        question="Grouped by island, the penguins of Biscoe weigh the most on average. Why?",
        options={
            "food": "Penguins on Biscoe find more food",
            "right": "Biscoe has the Gentoos, the heaviest species",
            "many": "Biscoe has the most penguins",
        },
        correct="right",
        why={
            "food": "The table says nothing about food. What did Biscoe weigh without its Gentoos? See the play block in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "many": "A mean does not grow with the number of values. What did Biscoe weigh without its Gentoos? See the play block in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "right": "Without its Gentoos, Biscoe weighs what the other islands weigh: who lives there makes its mean higher.",
        },
    )
    quick_biscoe
    return (quick_biscoe,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Two tables into one: merge](/course/open/data-merge?lang=en), the weather and the ice creams of the stand came on one row per day: on the warmest day, the stand sold the most. With two columns of numbers, a scatter plot shows at a glance whether they move together. But such a pattern is quickly read as cause and effect, while often a third column drives both.

    **An everyday example.** At a primary school, the children with the bigger feet read better. Do big feet help you read? No: the older children have bigger feet and have been reading for longer. Age drives both.
    """)
    return


@app.cell(hide_code=True)
def predict_bills():
    guess_bills = L.guess(
        "bills",
        question="The Palmer penguins of [One number per group: groupby](/course/open/data-groupby?lang=en): 342 of them have a bill length and a bill depth. Do penguins with a longer bill tend to have a deeper bill, or a shallower one?",
        options={
            "deeper": "A deeper bill",
            "shallower": "A shallower bill",
            "neither": "Neither: there is no direction",
        },
        reveal="**A shallower one.** Split the penguins at the median bill length: the half with the longer bills has bills 1.41 millimeters shallower on average. Within each species it is the other way round, as the block below shows.",
    )
    guess_bills
    return (guess_bills,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Two columns are associated when they move together; a scatter plot shows it as the slant of the cloud of dots. That does not make one the cause of the other: a third column can drive both, and can even turn the direction around.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **association** | two columns move together: where one is higher, the other tends to be higher (positive) or lower (negative) | `plt.scatter(df[x], df[y])` |
        | **cause**, **causality** | X causes Y if changing X, with everything else the same, changes Y; causality is that relation | |
        | **confounder** | a third column that drives two others, so that they move together even when neither causes the other | |
        | **Simpson's paradox** | a direction that holds within every group, but turns around when the groups are put together | |
        """),
        mo.Html(C.shapes_picture({"up": "Up", "down": "Down", "none": "None", "curve": "A curve",
                                  "flipper": "flipper length (mm)", "mass": "body mass (g)", "day": "day", "temp": "temperature (°C)", "wind": "wind (km/h)",
                                  "alt_1": "Two scatter plots. Up: the flipper length and body mass of the penguins, a cloud that rises to the right. Down: the days of the station's month and its temperature, from about 11 degrees at the start to 6 at the end.",
                                  "alt_2": "Two scatter plots. None: the temperature and the wind of the station's month, scattered without a direction. A curve: the days of another month of the station and its temperature, rising from 11 to 16.5 degrees and falling back to 11.5."})),
        mo.md(r"""
        Four scatter plots of the course's own tables; in two of them, the day is a column too. Up: the longer its flippers, the heavier a penguin, a positive association. Down: the station's month cools, a negative one. None: in that month, the wind and the temperature go no way together. A curve: another month of the station warms up and cools again, which one direction would hide.

        In the block below, an arrow sums up each cloud, a simple summary of the course's own: split the penguins at the median bill length, and it runs from the mean length and depth of the half with the shorter bills to those of the half with the longer bills. The block starts with all three species, each in its colour and shape and with its own arrow, a circle for an Adelie, a triangle for a Chinstrap and a square for a Gentoo, and the grey arrow of all penguins.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_species_cell():
    def _frame(step):
        step = int(step)
        n = C.step_numbers(step)
        if step == 0:
            _say = f"All {n['n']} penguins: the arrow points down. The half with the longer bills has bills {L.number(-n['gap'])} millimeters shallower on average."
            _alt = f"The bill depth against the bill length of all {n['n']} penguins, in grey, with an arrow from the middle of the half with the shorter bills to the middle of the half with the longer bills. It points down: {L.number(-n['gap'])} millimeters shallower."
        elif step < 4:
            _say = f"The {n['n']} {n['species']}s: their arrow points up. The half with the longer bills has bills {L.number(n['gap'])} millimeters deeper on average."
            _alt = f"The bill depth against the bill length of the penguins, with the {n['n']} {n['species']}s in their colour and shape and the others light. The arrow of the {n['species']}s points up: {L.number(n['gap'])} millimeters deeper."
        else:
            _g = n["gaps"]
            _say = f"All three species: each arrow points up, by {L.number(_g['Adelie'])}, {L.number(_g['Chinstrap'])} and {L.number(_g['Gentoo'])} millimeters. The grey arrow of all penguins points down."
            _alt = f"The bill depth against the bill length of the penguins, each species in its colour and shape, each with an arrow that points up: {L.number(_g['Adelie'])}, {L.number(_g['Chinstrap'])} and {L.number(_g['Gentoo'])} millimeters deeper. A grey arrow for all penguins together points down: {L.number(-n['gap'])} millimeters shallower."
        return (C.species_picture(step, {"title": "Bill depth against bill length", "x": "bill length (mm)", "y": "bill depth (mm)", "alt": _alt}), _say)

    explore_species = L.explore("species", "step", _frame, start=4, values=C.STEPS, role="",
                                task="Step through the species. Which way does the arrow of each species point, and which way the arrow of all penguins?")
    explore_species
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        Within each species, the longer the bill, the deeper; all penguins together, the longer, the shallower. That is Simpson's paradox, and the species is the confounder: it sets both the length and the depth of a bill. The Adelies, short and deep, lie at the top left and the Gentoos, long and shallow, at the bottom right, so together they make the whole cloud slant down.
        """),
        L.watch_out("A scatter plot shows that two columns move together, not why. On its own it cannot show a cause; an experiment can: change one column, keep everything else the same, and see whether the other changes."),
        mo.md(r"""
        ### Recipe

        To look at two columns together:

        1. Draw them as a scatter plot, one along the bottom and the other up the side: `plt.scatter(df[x], df[y])`, with a title and the axis labels.
        2. Read the cloud: does it go up to the right, down, no way at all, or does it bend?
        3. Think of a third column that could drive both, such as the species, and look per group: `df.groupby("species")[[x, y]].mean()` says where each group lies, and with a mask you draw one group on its own.
        4. Write what you see as "goes with", not as "causes": the plot alone cannot show a cause.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The Palmer penguins again, in `data/penguins.csv`: 344 penguins of three species. The comments with `#` say which step of the recipe each part is. Run the two cells.
    """)
    return


@app.cell
def example_two_columns():
    import matplotlib.pyplot as plt
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    # 1. all penguins: the bill length along the bottom, the bill depth up the side
    plt.scatter(_df["bill_length_mm"], _df["bill_depth_mm"])
    plt.title("All penguins")
    plt.xlabel("bill length (mm)")
    plt.ylabel("bill depth (mm)")
    plt.show()
    return pd, plt


@app.cell
def example_per_species(pd, plt):
    _df = pd.read_csv("data/penguins.csv")
    # 3. a third column, the species: where each species lies, then the Adelies on their own
    print(_df.groupby("species")[["bill_length_mm", "bill_depth_mm"]].mean())
    _adelie = _df[_df["species"] == "Adelie"]
    plt.scatter(_adelie["bill_length_mm"], _adelie["bill_depth_mm"])
    plt.title("Adelies only")
    plt.xlabel("bill length (mm)")
    plt.ylabel("bill depth (mm)")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Step 2: over all penguins, the cloud slants down a little. The means put the Adelies short and deep, 38.8 and 18.3 millimeters, the Chinstraps long and deep, 48.8 and 18.4, and the Gentoos long and shallow, 47.5 and 15.0; on their own, the Adelies go up to the right. Step 4: over all penguins, a longer bill goes with a shallower one because of the species, not because of the length.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chinstraps():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The Chinstraps on their own

        The box reads the penguins as `df`. Draw the bill depth, `bill_depth_mm`, against the bill length, `bill_length_mm`, of the Chinstraps only, with a title and a label on both axes. Replace the `...`. The check looks at the points of your chart, and the chart appears under its message.
        """),
        L.task("Keep the Chinstraps, and draw their bill depth against their bill length, with a title and two axis labels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chinstraps():
    box_chinstraps = L.code_box("chinstraps", lines=9)
    box_chinstraps
    return (box_chinstraps,)


@app.cell(hide_code=True)
def check_chinstraps(box_chinstraps):
    result_chinstraps = L.check("chinstraps", box_chinstraps.value)
    result_chinstraps
    return (result_chinstraps,)


@app.cell(hide_code=True)
def hints_chinstraps():
    L.hints(
        "chinstraps",
        [
            "Steps 1 and 3 of the recipe, as the Adelies in the worked example.",
            "A mask keeps the Chinstraps: `df[df[\"species\"] == \"Chinstrap\"]`. Then `plt.scatter` with their bill length first, and `plt.title`, `plt.xlabel` and `plt.ylabel`.",
            "`chinstrap = df[df[\"species\"] == ___]`, then `plt.scatter(chinstrap[___], chinstrap[___])`.",
        ],
        why="`df[\"species\"] == \"Chinstrap\"` is `True` for the 68 Chinstraps, and between the square brackets of `df` it keeps their rows. `plt.scatter` puts a point for each at its bill length and bill depth; the title and the labels say what the reader sees.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rescues():
    mo.md(r"""
    ### Exercise 2. Ice creams and rescues
    """)
    return


@app.cell(hide_code=True)
def choice_rescues_cell():
    choice_rescues = L.choice(
        "rescues",
        question="Over a summer, on the days the beach stand sells more ice creams, the lifeguards also rescue more swimmers. What is the most likely reason?",
        options={
            "ice": "Ice cream makes swimming more dangerous",
            "right": "Warm days: more people buy ice cream, and more people swim",
            "chance": "Chance: the two have nothing to do with each other",
        },
        correct="right",
        why={
            "ice": "Would the sea get safer if the stand closed? Moving together does not say which drives which. What else differs on the days with many ice creams?",
            "chance": "A pattern over a whole summer is hard to get by chance, and those days have something in common. What could drive both?",
            "right": "The temperature is a confounder: on warm days, more people buy ice cream and more people swim, so more get into trouble. Closing the stand would not make the sea safer.",
        },
    )
    choice_rescues
    return (choice_rescues,)


@app.cell(hide_code=True)
def text_exercise_means():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Where each species lies

        Over all penguins, heavier goes with shallower bills; within each species, with deeper ones. The box reads the penguins as `df`. Make `per_species`: the mean body mass, `body_mass_g`, and the mean bill depth, `bill_depth_mm`, of each species, in one table. Replace the `...`. The last line is `per_species`, so the check sees it.
        """),
        L.task("Make `per_species`: the mean body mass and the mean bill depth of each species."),
    ])
    return


@app.cell(hide_code=True)
def exercise_means():
    box_means = L.code_box("per-species", lines=6)
    box_means
    return (box_means,)


@app.cell(hide_code=True)
def check_means(box_means):
    result_means = L.check("per-species", box_means.value)
    result_means
    return (result_means,)


@app.cell(hide_code=True)
def hints_means():
    L.hints(
        "per-species",
        [
            "Step 3 of the recipe, as the means per species in the worked example, with other columns.",
            "Two columns go in a list: two pairs of square brackets after `groupby(...)`.",
            "`per_species = df.groupby(___)[[___, ___]].mean()`.",
        ],
        why="`df.groupby(\"species\")` splits the penguins by species, `[[\"body_mass_g\", \"bill_depth_mm\"]]` takes the two columns, and `.mean()` gives a table with the mean of each for every species: the Gentoos are the heaviest and have the shallowest bills.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) drew a scatter plot; now you read what it says, and what it cannot. In [One number per group: groupby](/course/open/data-groupby?lang=en), the Gentoos made Biscoe the heaviest island: the species was a confounder there too. In [Two tables into one: merge](/course/open/data-merge?lang=en), warm weather is a likely cause of the sales, but the table alone cannot show it. Later in the course, a model learns from how the columns of its data move together: associations, not causes. In the next lesson, *Where does this data come from?*, you ask who made a dataset and what you may do with it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chinstraps, choice_rescues, result_means):
    L.wrap_up(result_chinstraps, choice_rescues, result_means)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
