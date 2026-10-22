# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-groupby. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-groupby", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Asking questions of data](/course/open/data-explore?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_skip():
    quick_skip = L.choice(
        "quick-skip",
        question="The column `df[\"temp\"]` has 30 rows, and 2 of them are `NaN`. Over how many values does `df[\"temp\"].mean()` take the mean?",
        options={
            "all": "30: the missing ones count as 0",
            "right": "28",
            "error": "None: it stops with an error",
        },
        correct="right",
        why={
            "all": "pandas does not fill in 0. What does it do with a missing value in a mean? See the *Watch out* box in [Asking questions of data](/course/open/data-explore?lang=en).",
            "error": "pandas does not stop. What does it do with a missing value in a mean? See the *Watch out* box in [Asking questions of data](/course/open/data-explore?lang=en).",
            "right": "pandas skips missing values in a mean, without a word: the mean is over the 28 that are there.",
        },
    )
    quick_skip
    return (quick_skip,)


@app.cell(hide_code=True)
def quick_check_counts():
    quick_counts = L.choice(
        "quick-counts",
        question="What does `df[\"sky\"].value_counts()` give?",
        options={
            "kinds": "How many different values the column `sky` has",
            "right": "How often each value of `sky` appears",
            "all": "How many values the column `sky` has in all",
        },
        correct="right",
        why={
            "kinds": "That is one number; `value_counts` gives one for each value. Which number? See the table of new words in [Asking questions of data](/course/open/data-explore?lang=en).",
            "all": "That is one number; `value_counts` gives one for each value. Which number? See the table of new words in [Asking questions of data](/course/open/data-explore?lang=en).",
            "right": "One count per value, the most frequent first: in the month of the station, `sunny` 12 times, `rainy` 9, `cloudy` 8 and `Sunny` once.",
        },
    )
    quick_counts
    return (quick_counts,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The 342 penguins with a body mass weigh 4202 grams on average. But that one number mixes three species: how heavy is an Adelie, and how heavy a Gentoo? With a mask you can keep one species and take its mean, and repeat that for every species: three times here, a hundred times for a hundred groups. A summary per group takes one line.

    **An everyday example.** A running club wants to compare its age groups: it sorts the times of its members into a pile per age group, takes the mean time of each pile, and lists the means, one line per age group.
    """)
    return


@app.cell(hide_code=True)
def predict_per_species():
    guess_per_species = L.guess(
        "per-species",
        question="`df` is the table of 344 penguins, with the columns `species` and `body_mass_g` among others. What do you think `df.groupby(\"species\")[\"body_mass_g\"].mean()` gives?",
        options={
            "one": "One number: the mean body mass of all penguins",
            "per": "Three numbers: the mean body mass of each species",
            "every": "A number for every penguin: the mean of its species",
        },
        reveal="**Three numbers, one per species:** 3701 grams for an Adelie, 3733 for a Chinstrap and 5076 for a Gentoo.",
    )
    guess_per_species
    return (guess_per_species,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        `groupby` works in three steps, often called split-apply-combine. It splits the rows into groups by their value in one column; it applies the same computation to each group, such as a mean of another column; and it combines the outcomes into one Series, with a row per group and the group as its row label. The groups come sorted by name.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **groupby** | split the rows into groups by the values of a column, and compute one number per group | `df.groupby("species")` |
        | **group** | the rows with the same value in the column you group by, such as all Gentoos | |
        | **one number per group** | a column after `groupby`, then what to compute: `.mean()`, `.median()`, `.min()`, `.max()` | `df.groupby("species")["body_mass_g"].mean()` |
        | **count** | per group, how many values the column has; missing values do not count | `.count()` |
        | **size** | per group, how many rows it has, missing values or not | `df.groupby("species").size()` |
        | **agg** | several numbers per group at once, in a table with a column for each; you give their names in a list | `.agg(["mean", "count"])` |

        Each dot in the picture is a penguin, at its body mass, in the color and shape of its species: a circle is an Adelie, a triangle a Chinstrap and a square a Gentoo. The short line across each group is its mean: the one number `groupby` gives.
        """),
        mo.Html(C.picture({"title": "By species", "x": "species", "mass": "body mass (g)",
                           "alt": "The body mass of 342 penguins by species, a dot per penguin and a line at each mean. The Adelies and the Chinstraps lie around 3700 grams, from about 2700 to 4800; the Gentoos higher, from 3950 to 6300, with a mean of 5076."})),
        mo.md(r"""
        Grouped by island, Biscoe has the heaviest mean: 4716 grams, a kilogram above Dream and Torgersen. Biscoe holds two species. The slider sets how many of its Gentoos are counted, starting at all 123 with a body mass.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_gentoos_cell():
    def _frame(k):
        k = int(k)
        m = C.island_means(k)
        return (C.island_picture(k, {"title": "By island", "x": "island", "mass": "body mass (g)",
                                     "alt": f"The body mass of the penguins by island, a dot per penguin and a line at each mean. Biscoe, its 44 Adelies and {k} of its 123 Gentoos: {m['Biscoe']} grams. Dream, Adelies and Chinstraps: {m['Dream']}. Torgersen, only Adelies: {m['Torgersen']}."}),
                f"Biscoe: 44 Adelies and {k} Gentoos, a mean of {m['Biscoe']} grams. Dream: {m['Dream']}. Torgersen: {m['Torgersen']}.")

    explore_gentoos = L.explore("gentoos", "Gentoos", _frame, start=123, values=C.GENTOOS, role="",
                                task="Take Biscoe's Gentoos away, down to 0. How heavy is Biscoe then, next to Dream and Torgersen?")
    explore_gentoos
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        mo.md(r"""
        With no Gentoos, Biscoe weighs 3710 grams, next to 3713 for Dream and 3706 for Torgersen: its Adelies weigh what the penguins of the other islands weigh. Its mean is higher only because of who lives there: the Gentoos, which in this table live on Biscoe alone.
        """),
        L.watch_out("A mean per group says where the middle of each group lies, not how many values it rests on or what else differs between the groups: ask for the count next to it. And a row with nothing in the column you group by is in no group at all: `groupby` leaves it out, without a word."),
        mo.md(r"""
        ### Recipe

        To summarize a table per group:

        1. Choose the column whose values make the groups, and the column to summarize: `df.groupby("species")["body_mass_g"]`.
        2. Ask for one number per group: `.mean()`, or several at once: `.agg(["mean", "count"])`.
        3. Read the count next to each mean: a mean of a few values says little. `.size()` gives the rows per group.
        4. Before you explain a difference between groups, ask what else differs between them, and group by that column too: `df.groupby(["island", "species"])`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The penguins of the picture are the Palmer penguins, in `data/penguins.csv`: 344 penguins of three species from three islands, measured from 2007 to 2009 by Kristen Gorman and the Palmer Station Long Term Ecological Research program, license CC0. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_groupby():
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    # 1. and 2. the mean body mass of each species
    print(_df.groupby("species")["body_mass_g"].mean())
    # 3. next to the mean: how many values (count) and how many rows (size) each species has
    print(_df.groupby("species")["body_mass_g"].agg(["mean", "count", "size"]))
    # 3. and the rows per sex: a penguin whose sex was not recorded is in no group
    print(_df.groupby("sex").size())
    # 4. by island; then by island and species together
    print(_df.groupby("island")["body_mass_g"].mean())
    print(_df.groupby(["island", "species"])["body_mass_g"].agg(["mean", "count"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    For the Adelies and the Gentoos, `count` is one lower than `size`: one penguin of each has no body mass. By sex, the groups hold 165 and 168 penguins, 333 of the 344: the 11 whose sex was not recorded are in no group. By island and species, with a list of two names, the Adelies weigh about the same everywhere: 3710, 3688 and 3706 grams. There are five groups, not nine: a group is only made for a pair that is in the table.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sky():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The temperature per kind of sky

        The box reads the month of the weather station of [Asking questions of data](/course/open/data-explore?lang=en) as `df`, with the columns `day`, `temp`, `rain`, `wind`, `dir` and `sky`. Make `per_sky`: the mean temperature, `temp`, for each kind of sky, `sky`. Replace the `...`. The last line is `per_sky`, so the check sees it. The check also tries another month.
        """),
        L.task("Make `per_sky`: the mean temperature for each kind of sky."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sky():
    box_sky = L.code_box("per-sky", lines=6)
    box_sky
    return (box_sky,)


@app.cell(hide_code=True)
def check_sky(box_sky):
    result_sky = L.check("per-sky", box_sky.value)
    result_sky
    return (result_sky,)


@app.cell(hide_code=True)
def hints_sky():
    L.hints(
        "per-sky",
        [
            "Steps 1 and 2 of the recipe, as in the worked example, with other columns.",
            "The groups come from the column `sky`; the mean from the column `temp`.",
            "`per_sky = df.groupby(___)[___].mean()`.",
        ],
        why="`df.groupby(\"sky\")` splits the 30 days by their sky, `[\"temp\"]` takes their temperatures, and `.mean()` gives one mean per kind of sky: 7.5 degrees for `Sunny`, 10.79 for `cloudy`, 9.94 for `rainy` and 9.0 for `sunny`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_summary():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The mean and what it rests on

        Finish `summary(path, by, column)`. It reads the CSV file at `path` as `df`. It gives back, for each group of the column `by`, the mean of the column `column` and how many values that mean rests on: a table with the columns `mean` and `count`. Replace the `...`. The check calls `summary("data/penguins.csv", "species", "body_mass_g")`, and also tries other files and columns.
        """),
        L.task("Make `summary(path, by, column)` give back the mean and the count of `column` for each group of `by`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_summary():
    box_summary = L.code_box("summary", lines=7)
    box_summary
    return (box_summary,)


@app.cell(hide_code=True)
def check_summary(box_summary):
    result_summary = L.check("summary", box_summary.value)
    result_summary
    return (result_summary,)


@app.cell(hide_code=True)
def hints_summary():
    L.hints(
        "summary",
        [
            "Step 2 of the recipe: several numbers per group at once.",
            "`by` and `column` hold the names of two columns: use them without quotes, `by` in `groupby` and `column` between the square brackets. Then `agg`, with a list of two names.",
            "`return df.groupby(___)[___].agg([___, ___])`.",
        ],
        why="`df.groupby(by)` splits the rows by the column whose name is in `by`, `[column]` takes the column to summarize, and `.agg([\"mean\", \"count\"])` gives a table with its mean and its count for each group.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_one_day():
    mo.md(r"""
    ### Exercise 3. Reading a group summary
    """)
    return


@app.cell(hide_code=True)
def choice_one_day_cell():
    choice_one_day = L.choice(
        "one-day",
        question="In your `per_sky` of exercise 1, the mean temperature is 9.0 degrees for `sunny` and 7.5 degrees for `Sunny`. What can you conclude from that?",
        options={
            "colder": "Days with the sky `Sunny` are colder than days with `sunny`",
            "right": "Nothing about the weather: `Sunny` is one day, the same sky as `sunny`, written another way",
            "skipped": "The mean of `Sunny` is off, because pandas skipped its missing temperatures",
        },
        correct="right",
        why={
            "colder": "How many days is the mean of `Sunny` over? Ask for the count next to it, as in step 3 of the recipe. And what was strange about `Sunny` in [Asking questions of data](/course/open/data-explore?lang=en)?",
            "skipped": "The one day with `Sunny` has a temperature: `count` and `size` are both 1. How many days is that mean over, and what does that say about a kind of sky?",
            "right": "`Sunny` is a group of one day: a mean of one value says nothing about a kind of sky. It is `sunny` with a capital, and belongs with the other 12 sunny days. Count before you compare, and fix such a spelling first.",
        },
    )
    choice_one_day
    return (choice_one_day,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Asking questions of data](/course/open/data-explore?lang=en), `value_counts` counted how often each value appears: a count per group, and `df.groupby("sky").size()` gives the same numbers, sorted by name. `describe` summarized each column; `groupby` summarizes each group. Later in the course you compare groups like this again, such as how well a model does for each class, and there too a small group says little. In the next lesson, *Two tables into one: merge*, you put two tables side by side on a column they share.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_sky, result_summary, choice_one_day):
    L.wrap_up(result_sky, result_summary, choice_one_day)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
