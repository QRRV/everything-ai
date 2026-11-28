# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-data-quality. Route: course/route/lessen/systemen.yaml. Checks, the readings, the month of the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-data-quality", "en")
    C = L.checks   # the readings of four stations, the weather service, the month of the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en) and [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaps():
    quick_gaps = L.choice(
        "quick-gaps",
        question='A column `temp` has 10 gaps in 200 rows. After `df["temp"] = df["temp"].fillna(df["temp"].median())`, what does `df["temp"].isna().sum()` give?',
        options={
            "right": "0, since every gap now holds the median",
            "ten": "10, since fillna only marks where the gaps were",
            "rows": "190, the number of filled values",
            "all": "200, since isna counts every row of the column, filled or not, as long as the row exists",
        },
        correct="right",
        why={
            "right": "`fillna` puts the median in each gap, so `isna` finds none left: after filling, the table looks complete. In this lesson, you count the gaps on the data as it comes in, before anything fills them.",
            "ten": "`fillna` fills the gaps in; marking them is the work of a missing-indicator column. What is left for `isna` to find? See [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).",
            "rows": "`isna().sum()` counts the gaps, not the filled values. How many gaps are left after fillna? See [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).",
            "all": "`isna` is `True` only where a value is missing. After fillna, where is one missing? See [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en).",
        },
    )
    quick_gaps
    return (quick_gaps,)


@app.cell(hide_code=True)
def quick_check_box():
    quick_box = L.choice(
        "quick-box",
        question="In the picture of Sculley and others (2015), how big is the box of ML code in a real ML system?",
        options={
            "big": "The largest box by far, since the whole system is built around the model and the rest only feeds it",
            "half": "About half, with serving the answers as the other half",
            "right": "Small, among much larger boxes such as collecting and checking the data",
            "same": "The same size as each of the other boxes",
        },
        correct="right",
        why={
            "right": "The box of ML code is small; collecting the data, checking it, features, serving and monitoring are much larger. This lesson is about one of those boxes: checking whether the data is good.",
            "big": "In the picture, the box of ML code is one of the smallest. Which boxes are larger? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "half": "Serving the answers is one box of many, and the box of ML code is small. Which boxes are large? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
            "same": "The boxes differ a lot in size, and the box of ML code is among the smallest. Which are larger? See [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en).",
        },
    )
    quick_box
    return (quick_box,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model answers from data, and it answers bad data just as confidently as good data. In practice, data breaks more often than code: a sensor breaks, a supplier changes a format, a feed stops at night. Monitoring the answers notices only once the damage is done, so a team checks the data itself, before it reaches training or the model. But good is not one thing. A table can have every cell filled and still be wrong, old, or written in two ways. Data quality has dimensions you can measure, each with its own question, and a table gets a score on each.

    **An everyday example.** A recipe card for bread can go wrong in four ways. A line can be missing: how much salt? A line can say 50 grams where 5 was meant. One line can say grams and the next teaspoons. And the card can be the old one, from before the oven was replaced. Each is a different problem, and you catch each with a different look.
    """)
    return


@app.cell(hide_code=True)
def predict_catch():
    guess_catch = L.guess(
        "catch",
        question="In one month, four things go wrong with the weather readings that the bike model uses: a wind sensor breaks, a station writes its city in a new way, a thermometer reads 3 degrees too warm, and the feed of two stations stops. A check counts the empty cells every evening. How many of the four does it catch, do you think?",
        options={
            "none": "None, since a broken sensor sends a zero, not an empty cell",
            "right": "One: the broken wind sensor, whose readings leave the wind out",
            "two": "Two: the broken sensor, and the stopped feed, which leaves the cells of two stations empty",
            "all": "All four, since every problem ends in an empty or wrong cell",
        },
        reveal="**One.** The stopped feed leaves no empty cells: it sends no new rows at all, and the newest rows of the two stations are as full as ever, only old. The new spelling and the warm thermometer fill every cell, with values that look fine. Each problem needs its own way to look, and the demo shows the four side by side.",
    )
    guess_catch
    return (guess_catch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Four questions.** **Data quality** is how fit data is for its use. It has dimensions, and each is a question with a share from 0 to 1 as its answer. This lesson uses four.

        - **Completeness**: is the value there? The share of the cells that are filled. A key that a reading leaves out, or an empty cell in a table, counts against it.
        - **Correctness**: is the value true? The share of the values that agree with a source the team trusts, within a tolerance: here the weather service, at most 1 degree off. Many texts call this accuracy; in this course, accuracy stays the share of right answers of a model.
        - **Consistency**: is the same thing written the same way everywhere? The share of the values written in the agreed way: Houten and not houten, the same unit in every row, the same name for a station in every table.
        - **Timeliness**: is the value new enough? The share of the sources whose newest data is fresh enough: here, from today. Old data can be complete, true and consistent, and still wrong for today.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **A score per dimension.** You give a table a score on each dimension and look at the lowest: that is where the data breaks, and where to look first. Each dimension counts over what it is about. Consistency and correctness count only the filled values, since an empty cell already counts for completeness.

        **Before filling.** Measure completeness on the data as it comes in. After `fillna`, every gap holds a value and the table looks complete; the missing-indicator column of [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en) keeps what the score needs. In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en), collecting and checking the data were among the largest boxes around the model: this is the checking.

        **More dimensions.** The four are not the only ones. The Government Data Quality Hub of the United Kingdom, on 24 June 2021, recommends six, as DAMA UK defines them: accuracy, completeness, uniqueness, consistency, timeliness and validity. The standard ISO/IEC 25012 (2008) has 15 characteristics, among them accuracy, completeness, consistency and currentness. Both were read on 28 November 2026. Uniqueness (no row twice) and validity (a value of the right type and in its allowed range) come back in the next lesson, as rules.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **data quality** | | how fit data is for its use, measured on dimensions such as completeness, correctness, consistency and timeliness | |
        | **completeness** | | the share of the cells that are filled | `column in row` |
        | **correctness** | | the share of the values that agree with a source you trust, within a tolerance; often called accuracy | |
        | **consistency** | | the share of the values written in the agreed way: one spelling, one unit, one name | |
        | **timeliness** | | whether data is new enough: the share of the sources whose newest data is fresh enough | |
        """),
        L.watch_out("A score of 1.0 means only that this one way of looking found nothing. A table full of last week's readings can score 1.0 on completeness, correctness and consistency."),
        L.watch_out("Do not repair data inside the check. A check that takes off spaces or changes capitals before it counts hides the very problem it should report. Count first, and repair in a step of its own, where you can see what changed."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    How good is a table, and where does it break:

    1. Decide what a row is, and which columns the model needs.
    2. Completeness: for each column, the share of the rows with a value.
    3. Correctness: compare the values with a source you trust, within a tolerance.
    4. Consistency: for each column, the share of the filled values written in the agreed way.
    5. Timeliness: for each source, the day of its newest row; the share of the sources that are fresh enough.
    6. Look at the lowest score first, and find the source and the day where it broke.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.READINGS` holds the readings at noon of four stations on three days, as dicts: a reading without a value leaves the key out. Completeness per column: count the rows that have the key, and divide by the number of rows. `_column in _row` asks whether the dict `_row` has that key. The cell first prints one reading. Run the cell.
    """)
    return


@app.cell
def example_complete():
    print(C.READINGS[5])
    for _column in ["city", "temp", "wind"]:
        _filled = 0
        for _row in C.READINGS:
            if _column in _row:
                _filled = _filled + 1
        print(_column, round(_filled / len(C.READINGS), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The city is missing once, at west on day 3, and the wind twice, at east on days 2 and 3: completeness 0.917 and 0.833. The temperature is always there: 1.0. But look at the reading that was printed: south writes houten, where day 1 says Houten. Completeness cannot see that, since the cell is filled. Exercise 1 counts it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_consistency():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Written as agreed?

        `consistency(rows, column, agreed)` gets the rows, the name of a column, and a list `agreed` with the agreed spelling of each value. The `for` loop walks the rows. A `row` that has the column counts in `filled`, and `same` counts the values written in an agreed way. A row without the column counts for neither: its empty cell counts for completeness. The function gives back the share `same / filled`. Replace the `...` with the question whether the value of `row` under `column` is one of the spellings of `agreed`.

        `C.CITIES` is `["Utrecht", "Houten", "Zeist", "Nieuwegein"]`. The check calls `consistency(C.READINGS, "city", C.CITIES)`. Then the names of the stations, a shorter list, a few rows of its own, and a column of units.
        """),
        L.task("Count the filled values that are written as agreed, and give back their share."),
    ])
    return


@app.cell(hide_code=True)
def exercise_consistency():
    box_consistency = L.code_box("consistency", lines=10)
    box_consistency
    return (box_consistency,)


@app.cell(hide_code=True)
def check_consistency(box_consistency):
    result_consistency = L.check("consistency", box_consistency.value)
    result_consistency
    return (result_consistency,)


@app.cell(hide_code=True)
def hints_consistency():
    L.hints(
        "consistency",
        [
            '`row` is one reading, a dict, and `column` the name of a key, such as `"city"`. The question after `if` is whether the value of `row` under the key `column` is one of the strings of `agreed`. How do you look up a value in a dict, and how do you ask whether a value is one of the values of a list?',
            '`"Zeist" in ["Utrecht", "Zeist"]` gives `True`, and `"zeist" in ["Utrecht", "Zeist"]` gives `False`: the spelling has to be exactly the same, capitals and spaces too.',
            "`            if row[column] ___:`",
        ],
        why='`if row[column] in agreed:` is true for a value written exactly as one of the agreed spellings. Of the 11 filled cities, "houten" twice and "Utrecht " once are not: 8 of 11, 0.727.',
    )
    return


@app.cell(hide_code=True)
def text_exercise_correctness():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. True within a tolerance?

        Now with NumPy, for every day at once. `correctness(readings, reference, tolerance)` gets two arrays of the same length: the readings of a station, and the reference, the values of a source the team trusts, for the same moments. `diff` holds how far each reading lies from the reference: positive when the reading is too high, negative when it is too low. `close` must be an array of `True` and `False`, with `True` for a reading at most `tolerance` from the reference. `close.mean()` then gives the share of `True`, since `True` counts as 1. Replace the `...`.

        `C.TEMPS` holds the temperatures at noon of station west on 12 days, and `C.REFERENCE` those of the weather service. The check calls `correctness(C.TEMPS, C.REFERENCE, 1.0)`. Then a tolerance of 0.5 and of 3.0, readings equal to the reference, and two small arrays of its own.
        """),
        L.task("Count, for every day at once, the readings at most the tolerance off, below or above."),
    ])
    return


@app.cell(hide_code=True)
def exercise_correctness():
    box_correctness = L.code_box("correctness", lines=5)
    box_correctness
    return (box_correctness,)


@app.cell(hide_code=True)
def check_correctness(box_correctness):
    result_correctness = L.check("correctness", box_correctness.value)
    result_correctness
    return (result_correctness,)


@app.cell(hide_code=True)
def hints_correctness():
    L.hints(
        "correctness",
        [
            "A reading counts as right when it lies at most `tolerance` from the reference, below or above. `close` must be `True` there: a comparison with `<=` of how far off each reading is, against `tolerance`. How far off is a reading with a `diff` of -3.5?",
            "-3.5 is below 1.0, yet that reading is 3.5 degrees off. Python's `abs` turns -3.5 into 3.5, and it works on a NumPy array too, number by number: `abs(np.array([-2.0, 0.5]))` gives `array([2. , 0.5])`.",
            "`    close = ___ <= tolerance`",
        ],
        why="`close = abs(diff) <= tolerance` is `True` where a reading lies at most 1 degree off, below or above. West has 9 such days of 12: 0.75. Without `abs`, the two days far below would count as right.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_problems():
    mo.md(r"""
    ### Exercise 3. Which dimension?

    Eight problems from the data of a bike rental. Put each one under the dimension that measures it.
    """)
    return


@app.cell(hide_code=True)
def sort_problems_cell():
    sort_problems = L.sort("problems")
    sort_problems
    return (sort_problems,)


@app.cell(hide_code=True)
def check_problems(sort_problems):
    result_problems = L.check("problems", sort_problems)
    result_problems
    return (result_problems,)


@app.cell(hide_code=True)
def hints_problems():
    L.hints(
        "problems",
        [
            "The four questions, under *Four questions*: is the value there, is it true, is it written the agreed way, and is it new enough?",
            "Two problems go in each column.",
            "Completeness: the wind and \\_\\_\\_. Correctness: the sun and \\_\\_\\_. Consistency: houten and \\_\\_\\_. Timeliness: Friday and \\_\\_\\_.",
        ],
        why="**Completeness:** no wind for a week, and rows without a city. **Correctness:** three degrees too warm, and ten times the wind. **Consistency:** houten and Houten, and two names for one station. **Timeliness:** silent since Friday, and six hours behind.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Exercise 4. Where do you look first?

    On Monday morning, the answers of the bike model are far off. The check of Sunday evening gave: completeness 0.99, correctness 0.98, consistency 1.00, timeliness 0.25.
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="What do you look at first?",
        options={
            "retrain": "Training the model again on the newest data of the stations, since the world changed",
            "sensor": "A thermometer that reads too warm",
            "spelling": "A city that is written in a new way",
            "right": "Whether three of the four stations still send their readings",
        },
        correct="right",
        why={
            "right": "Timeliness 0.25: only one of the four stations sent readings on Sunday. The other three scores are high, since they look at the newest rows that are there, and those rows are old. First find out why three feeds stopped.",
            "retrain": "The newest data of three stations is old, so training again would learn from the same old rows. Which score is low?",
            "sensor": "Correctness is 0.98: the readings agree with the weather service. Which score is low?",
            "spelling": "Consistency is 1.00: every value is written as agreed. Which score is low?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### A month of readings

    The four stations of this lesson for 30 days, with a reading every hour. Four things go wrong, each on a few days. Every evening, a check gives the four scores over the newest readings of each station: completeness of the temperature and the wind, correctness against the weather service, consistency of the city, and timeliness, the share of the stations with readings from that day. Choose a day.
    """)
    return


@app.cell(hide_code=True)
def demo_day():
    L.explore("day", "day of the month", lambda _v: C.frame(_v, L.lang), start=7, values=C.DEMO_DAYS, role="",
              task="Go through the month: on which days does a score fall under 0.95, and do the other three move with it?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Each incident moves one score and leaves the other three where they were. The broken wind sensor, on days 6 to 9, moves only completeness, down to 0.85. The new spelling of south, on days 13 to 17, moves only consistency, to 0.75. The thermometer in the sun, on days 20 to 23, moves only correctness, to 0.72. And the stopped feed of north and east, from day 26, moves only timeliness, to 0.50. On every other day, all four stay above 0.97. One number for the whole table, the mean of the four, would have missed the wind sensor: it stays at 0.96 or higher on those days, above 0.95. And the stopped feed is the one that the rows can never show by themselves: the newest rows of north and east are full, true and consistent, only old.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Gaps in a table: dropna and fillna](/course/open/data-missing-values?lang=en), you filled the gaps of a table; here you count them first, with three more ways to look. In [More than a model: what an ML system is](/course/open/sys-what-is-an-ml-system?lang=en), checking the data was one of the large boxes around the model: it comes before training, and before the model answers. In the next lesson, *Rules for data: schemas and expectations*, the looking becomes a gate: rules for each column and for the whole table that stop a bad batch before it reaches training.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_consistency, result_correctness, result_problems):
    L.wrap_up(result_consistency, result_correctness, result_problems, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
