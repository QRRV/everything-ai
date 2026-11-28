# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-data-validation. Route: course/route/lessen/systemen.yaml. Checks, the bike data, the gate of the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-data-validation", "en")
    C = L.checks   # the schema, the days of the bike data, the gate of the demo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_warm():
    quick_warm = L.choice(
        "quick-warm",
        question="For four days, the thermometer of one station reads 3 degrees too warm, and every cell stays filled. Which score of the check drops?",
        options={
            "complete": "Completeness, since a wrong value counts as a missing one",
            "spelling": "Consistency, since the station writes its values in a new way",
            "right": "Correctness, since the values no longer agree with the weather service",
            "time": "Timeliness, since readings that are 3 degrees off count as too old to use, until the station is repaired",
        },
        correct="right",
        why={
            "right": "Correctness compares each value with a source you trust, within a tolerance, and 3 degrees is outside a tolerance of 1. The cells are filled, written the agreed way, and new. In this lesson, such checks become rules that run on every batch.",
            "complete": "A wrong value is still a value: the cell is filled. Which dimension compares the value with a source you trust? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
            "spelling": "The station writes its values as before, only 3 degrees too high. Which dimension asks whether a value is true? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
            "time": "The readings arrive on time; they are just not true. Which dimension asks whether a value is true? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
        },
    )
    quick_warm
    return (quick_warm,)


@app.cell(hide_code=True)
def quick_check_fill():
    quick_fill = L.choice(
        "quick-fill",
        question="A table has 30 empty cells of 600. Someone fills them with the median, and then measures completeness. What does the score show?",
        options={
            "zero": "0.0, since filled cells no longer count",
            "right": "1.0, which hides the 30 gaps that were there",
            "low": "0.95, the share of the cells that held a value before the filling",
            "none": "Nothing at all, since completeness can be measured only on a column that still has every one of its gaps",
        },
        correct="right",
        why={
            "right": "After the filling, every cell holds a value, so completeness says 1.0 and the 30 gaps are out of sight. Measured before the filling: 570 of 600, 0.95. A check belongs where the data comes in, before anything changes it.",
            "zero": "A filled cell counts as filled, whatever put the value there. What does the score show once every cell holds a value? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
            "low": "0.95 is what the table had before the filling: 570 of 600. Measured after, what does it show? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
            "none": "Completeness can be measured on any column; the question is when. What does it show after the filling? See [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en).",
        },
    )
    quick_fill
    return (quick_fill,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every night, a new day of bike data arrives, and a model of the demand learns from the newest days. Nobody reads a table of 24 rows every night, so whatever is in it goes straight into training. On 10 March 2011, the humidity in that table is 0 in every hour, on a day of light rain: a broken sensor. In [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en), you measured how good a table is. Here the measuring becomes a gate: rules written once, and checked on every new batch, the data that arrives at once, before it reaches training. A batch that breaks a rule stops at the gate.

    **An everyday example.** A web form asks for a date of birth. Type 31 February, and it says so at once, before anything is sent: a rule about one field. Order 40 bags of flour where people order 1 or 2, and a good shop asks whether you are sure: a rule about the order as a whole, which no single field breaks.
    """)
    return


@app.cell(hide_code=True)
def predict_schema():
    guess_schema = L.guess(
        "schema",
        question="On 17 August 2012, atemp, the temperature as it feels, says 0.2424 in every hour of the day, while temp goes from 0.62 to 0.86. The schema says that atemp lies from 0 to 1. How many of the 24 hours of atemp does the schema stop, do you think?",
        options={
            "right": "None, since every value lies between 0 and 1",
            "few": "A few: the hours where atemp lies far below temp",
            "half": "Half of them: the hours of the afternoon, when temp is high",
            "all": "All 24, since a value that never changes falls outside the range of its column",
        },
        reveal="**None.** Each 0.2424 lies between 0 and 1, so every value passes on its own. What is wrong is the column as a whole: a temperature that does not move for 24 hours, while temp moves by 0.24. A schema looks at one value at a time; it takes a rule about many values together to see this. That rule is an expectation.",
    )
    guess_schema
    return (guess_schema,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A schema.** A **schema** says per column what is allowed: the type, a range, a set of values, and whether the column may be empty. It looks at one value at a time: an hour of 25 is wrong, whatever the other rows say. In this lesson, a schema is a dict with a range per column, from low to high, both ends allowed:

        ```python
        C.SCHEMA = {"hr": (0, 23), "weekday": (0, 6), "weathersit": (1, 4), "temp": (0.0, 1.0),
                    "atemp": (0.0, 1.0), "hum": (0.01, 1.0), "windspeed": (0.0, 1.0)}
        ```

        The data of Capital Bikeshare in Washington, D.C. (Fanaee-T, 2013) divides temp, atemp, hum and windspeed by a fixed number, so that each lies from 0 to 1. hum has a low of 0.01, since outdoors the air is never completely dry.

        **An expectation.** An **expectation** is a rule about many values together: the number of rows of a day, no hour twice, a column that must move during the day, a mean within a range. No single value breaks it. The word comes from Great Expectations, a tool for exactly these rules. This lesson uses three: a day has from 20 to 24 rows, no hour comes twice, and temp, atemp and hum each move by at least 0.01 over the day.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **The gate.** **Data validation** checks every new batch against the schema and the expectations, before it reaches training or the model. When every rule holds, the batch goes through. When a rule breaks, the batch is held back as a whole: training keeps the data of before, and a person gets a message with what broke. The person decides: the data may be broken, and is repaired in a step of its own, or the day may be true but unusual. Either way, the rule did its work: it asked a question at the right moment.

        **The tools.** Two well-known tools write the same rules. With Pandera, the schema of a pandas table reads:

        ```python
        import pandera.pandas as pa

        schema = pa.DataFrameSchema({
            "hr": pa.Column(int, pa.Check.in_range(0, 23)),
            "hum": pa.Column(float, pa.Check.in_range(0.01, 1.0)),
        })
        schema.validate(day, lazy=True)
        ```

        `in_range` allows both ends, as the dict here does, and `lazy=True` runs every rule and reports all failures together. With Great Expectations, the same range and a rule about the rows read:

        ```python
        import great_expectations as gx

        gx.expectations.ExpectColumnValuesToBeBetween(column="hum", min_value=0.01, max_value=1.0)
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=20, max_value=24)
        ```

        Great Expectations can let a rule pass when most values hold: with `mostly=0.95`, 5 percent of the values may break it. Read on 28 November 2026, in Pandera 0.36.0 and Great Expectations 1.25.0. The course installs neither: the dict and the two functions of this lesson do the same work in a few lines.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **schema** | | per column what is allowed: the type, a range, a set of values, empty or not; it looks at one value at a time | `C.SCHEMA` |
        | **expectation** | | a rule about many values together, such as the number of rows of a day | `len(rows) >= 20` |
        | **data validation** | | checking every new batch against the schema and the expectations before it is used, and holding back a batch that breaks a rule | |
        | **batch** | | the data that arrives at once: here, one day of hours | |
        | `or` | "or" | `True` when at least one of the two sides is `True` | `x < 0 or x > 1` |
        """),
        L.watch_out("Hold back a batch as a whole, rather than dropping the rows that break a rule. On 10 March 2011, every row breaks the schema: dropping them leaves a day of 0 hours, and a day of 22 hours with a few rows gone would teach the model a day that never was."),
        L.watch_out("Pick a range from what can happen, not from last week's data. A rule that fires every night teaches people to ignore it, and a range that allows anything stops nothing."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A gate for the data:

    1. Write the schema: per column the type, the range or the allowed values, and whether it may be empty.
    2. Write the expectations: what must hold for the batch as a whole, such as the number of rows, no duplicates, and columns that move.
    3. When a batch arrives, check every value against the schema and the batch against each expectation, and collect what breaks.
    4. Nothing breaks: let the batch through to training.
    5. Something breaks: hold the batch back, keep training on the data of before, and tell a person what broke.
    6. The person looks: broken data is repaired in a step of its own, and a true but unusual day is let through or kept apart, with a note why.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.MARCH_10` holds the 22 rows of 10 March 2011, each a dict. The cell takes one row and prints, per column of `C.SCHEMA`, the low, the value and the high. `_low, _high = C.SCHEMA[_column]` gives the two ends of the range their own names. Run the cell.
    """)
    return


@app.cell
def example_row():
    _row = C.MARCH_10[8]
    print("hour", _row["hr"])
    for _column in C.SCHEMA:
        _low, _high = C.SCHEMA[_column]
        print(_column, _low, _row[_column], _high)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Read each line from left to right: does the value lie between its low and its high? It does for every column but hum: 0.0 lies below the low of 0.01. A weathersit of 3 is light rain, and in rain the air is far from dry: the sensor is broken. Exercise 1 asks the same question of every value of every row.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_broken():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Outside its range?

        `broken(rows, schema)` gets the rows of a batch, each a dict, and a schema, a dict with a pair `(low, high)` per column. The two `for` loops walk every `row` and every `column` of the schema, and `low, high = schema[column]` gives the two ends their own names. `found` collects, with `found.append((row["hr"], column))`, a pair for each value that breaks the schema: the hour of the row and the name of the column. Replace the `...` with the question whether the value of `row` under `column` lies below `low` or above `high`. `a or b` is `True` when at least one of the two is `True`: `5 < 0 or 5 > 3` gives `True`.

        The check calls `broken(C.MARCH_10, C.SCHEMA)`. Then a day of June 2012 without a problem, a schema of one column, and a few rows of its own with values just inside and just outside a range.
        """),
        L.task("List every value outside its range, with its hour and its column."),
    ])
    return


@app.cell(hide_code=True)
def exercise_broken():
    box_broken = L.code_box("broken", lines=9)
    box_broken
    return (box_broken,)


@app.cell(hide_code=True)
def check_broken(box_broken):
    result_broken = L.check("broken", box_broken.value)
    result_broken
    return (result_broken,)


@app.cell(hide_code=True)
def hints_broken():
    L.hints(
        "broken",
        [
            "`row[column]` is the value of this row under this column, and `low` and `high` are the two ends of its range. A value breaks the schema on either side: below `low`, or above `high`. Which two comparisons do you need, and which word joins them so that one of the two is enough?",
            "`0.0 < 0.01 or 0.0 > 1.0` gives `True`: the first side is `True`, and one is enough. `0.5 < 0.01 or 0.5 > 1.0` gives `False`: the value lies inside.",
            "`            if row[column] < low or ___:`",
        ],
        why="`if row[column] < low or row[column] > high:` is `True` for a value outside its range, on either side. On 10 March 2011, that is hum in all 22 rows: 0.0, below the low of 0.01.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stuck():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Does a column move?

        Now an expectation, with NumPy. `stuck(hours, columns, least)` gets the hours of a day as columns: a dict with a NumPy array per column, so that `hours["temp"]` holds the 24 values of temp. `columns` lists the names to check, and `least` is the smallest spread that counts as moving. The spread of a column is its largest value minus its smallest, and a NumPy array has a method for each, such as `values.max()` for the largest. The loop walks the names, `values` holds the array of one `column`, and `found.append(column)` collects each column that does not move enough. Replace the `...` with the question whether the spread of `values` is less than `least`.

        `C.AUGUST_17` holds temp, atemp, hum and windspeed of the 24 hours of 17 August 2012. The check calls `stuck(C.AUGUST_17, ["temp", "atemp", "hum"], 0.01)`. Then 10 March 2011, a larger `least`, other columns, and small arrays of its own.
        """),
        L.task("List the columns whose spread over the day is less than least."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stuck():
    box_stuck = L.code_box("stuck", lines=8)
    box_stuck
    return (box_stuck,)


@app.cell(hide_code=True)
def check_stuck(box_stuck):
    result_stuck = L.check("stuck", box_stuck.value)
    result_stuck
    return (result_stuck,)


@app.cell(hide_code=True)
def hints_stuck():
    L.hints(
        "stuck",
        [
            "The spread is the largest value minus the smallest. `values.max()` gives the largest; the question after `if` takes the smallest off it, and asks whether what is left is less than `least`. Which method gives the smallest value of an array?",
            "`np.array([0.3, 0.8, 0.5]).max()` gives `0.8`, and `.min()` on the same array gives `0.3`: a spread of 0.5. An array of 24 times 0.2424 has a spread of 0.",
            "`        if values.max() - ___ < least:`",
        ],
        why="`if values.max() - values.min() < least:` is `True` for a column whose spread is less than `least`. On 17 August 2012, atemp has a spread of 0: it says 0.2424 in every hour, while temp moves by 0.24.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rules():
    mo.md(r"""
    ### Exercise 3. Schema or expectation?

    Eight rules for the bike data. Put each one under schema, a rule that one value can break on its own, or under expectation, a rule about many values together.
    """)
    return


@app.cell(hide_code=True)
def sort_rules_cell():
    sort_rules = L.sort("rules")
    sort_rules
    return (sort_rules,)


@app.cell(hide_code=True)
def check_rules(sort_rules):
    result_rules = L.check("rules", sort_rules)
    result_rules
    return (result_rules,)


@app.cell(hide_code=True)
def hints_rules():
    L.hints(
        "rules",
        [
            "Ask of each rule: can I check it by looking at one value, and nothing else?",
            "Four rules go on each side.",
            "Schema: hr from 0 to 23, weathersit, \\_\\_\\_ and \\_\\_\\_. Expectation: 20 to 24 rows, no hour twice, \\_\\_\\_ and \\_\\_\\_.",
        ],
        why="**Schema:** hr from 0 to 23, weathersit is 1, 2, 3 or 4, hum above 0, and no empty temp. **Expectation:** 20 to 24 rows, no hour twice, temp moves, and about as many bikes.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_storm():
    mo.md(r"""
    ### Exercise 4. A short day

    On the night after 29 October 2012, the batch of that day has 1 row, and the rule of 20 to 24 rows holds it back. The news of that day says that Capital Bikeshare stopped at 1 a.m., for Hurricane Sandy, and stayed closed until the next afternoon.
    """)
    return


@app.cell(hide_code=True)
def choice_storm_cell():
    choice_storm = L.choice(
        "storm",
        question="What do you do with the batch?",
        options={
            "rule": "Remove the rule on rows, since it held back a day that was true",
            "lower": "Change the rule to at least 1 row, so that a day like this passes next time",
            "right": "Let it through as a true day, or keep it out of training as a day of its own, and write down why",
            "fill": "Fill in the 23 missing hours with the same hours of the Monday before, and train on the day as usual",
        },
        correct="right",
        why={
            "right": "The rule did its work: it asked a person to look, and the person knows why the day is short. The one row is true, since the system was closed. Whether a model should learn from a closed day is a choice of its own, written down, and the rule stays for the next night.",
            "rule": "Without the rule, a day on which the feed breaks after 3 hours would pass unseen too. Did the rule do something wrong, or did it ask a good question?",
            "lower": "With at least 1 row, a feed that sends 3 hours of 24 passes as well. Is the rule wrong, or does this day need a look?",
            "fill": "Made-up hours turn a true day into a false one: the model would learn that bikes were rented while the system was closed. What does the one row say about the day?",
        },
    )
    choice_storm
    return (choice_storm,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Two years of batches

    Every day of 2011 and 2012 as the batch of a night, checked with `C.SCHEMA` and the three expectations: 20 to 24 rows, no hour twice, and temp, atemp and hum each move by at least 0.01. The strips at the top show all 731 days, with the days the gate holds back marked. The slider stops at each day held back, and at four days without a problem. Choose a day.
    """)
    return


@app.cell(hide_code=True)
def demo_day():
    L.explore("day", "day of 2011 and 2012", lambda _v: C.frame(_v, L.lang), start=69, values=C.DEMO_DAYS, role="",
              task="Go through the days held back: which rule breaks, and is the data broken, or the day unusual?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The gate holds back 12 of the 731 days. Eight are short: 18 January and 22 February 2011, with the morning missing and no reason in the data; 26 and 27 January 2011, around heavy snow; 27 and 28 August 2011, when Capital Bikeshare closed for Hurricane Irene; and 29 and 30 October 2012, when the system stopped for Hurricane Sandy. The schema holds back one: 10 March 2011, with a humidity of 0 in every hour. And the expectation that temp, atemp and hum move holds back three more: 6 September 2011, and 17 August and 29 December 2012. The schema alone would have caught only 10 March: the atemp of 17 August passes every range, 24 times over. On a few of the 12 days the data is broken, on most of them the day was unusual, and on 6 September nobody can say without a look. That look is what the gate is for.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Is the data good? Four ways to look](/course/open/sys-data-quality?lang=en), you measured a table on four dimensions; here the measuring became rules that run on every batch, and a gate that holds back what breaks them. Validity and uniqueness, two dimensions of DAMA UK, came back as rules: a value in its range, and no hour twice. The next lesson is *Why training costs millions*: there the domain turns from the data to the machines, and to what it costs to train a large model on many computers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_storm, result_broken, result_rules, result_stuck):
    L.wrap_up(result_broken, result_stuck, result_rules, choice_storm)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
