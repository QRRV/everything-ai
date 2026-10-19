# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson stats-describe. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("stats-describe", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Sums and means along an axis](/course/open/np-axis?lang=en) and [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mean():
    quick_mean = L.choice(
        "quick-mean",
        question="`a = np.array([2, 4, 9])`. What does `a.mean()` give?",
        options={
            "mean": "`5.0`",
            "sum": "`15`",
            "middle": "`4`",
        },
        correct="mean",
        why={
            "sum": "That is the sum, `a.sum()`. What do you do with the sum to get the mean? See the table of new words in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "middle": "That is the middle one of the three numbers. How is a mean worked out? See the table of new words in [Sums and means along an axis](/course/open/np-axis?lang=en).",
            "mean": "2 + 4 + 9 is 15, and 15 divided by 3 numbers is 5.0.",
        },
    )
    quick_mean
    return (quick_mean,)


@app.cell(hide_code=True)
def quick_check_bar():
    quick_bar = L.choice(
        "quick-bar",
        question="What does `plt.bar([\"mon\", \"tue\"], [3, 5])` draw?",
        options={
            "line": "A line from 3 to 5",
            "points": "Two loose points",
            "bars": "Two bars, 3 and 5 high",
        },
        correct="bars",
        why={
            "line": "A line through points is `plt.plot`. Which chart has a bar for each name? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "points": "Loose points are `plt.scatter`. Which chart has a bar for each name? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "bars": "`plt.bar` draws a bar for each name, as high as its value.",
        },
    )
    quick_bar
    return (quick_bar,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A column of data can have thousands of values, too many to read one by one. A few numbers and one chart tell you most of it: where the middle is, and how far the values spread around it. You look at data this way before you train a model on it, and later you describe how far off a model is with these same numbers.

    **An everyday example.** A weather forecast: "tomorrow around 18 degrees, between 14 and 21". One number for the middle, and two for the spread.
    """)
    return


@app.cell(hide_code=True)
def predict_bus():
    guess_bus = L.guess(
        "bus",
        question="Fifteen students timed their trip to school. Most took 12 to 45 minutes; one missed the bus and took 95. Which do you think is larger: the mean or the median, the middle value once you sort them?",
        options={
            "mean": "The mean",
            "median": "The median",
            "same": "They are the same",
        },
        reveal="**The mean.** The mean is 30.7 minutes and the median 25. The trip of 95 minutes adds a lot to the sum, so it pulls the mean up. For the median it is just one more value at the top of the order.",
    )
    guess_bus
    return (guess_bus,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Two questions describe a column of numbers: where is the middle, and how far do the values spread around it? The mean is the middle you know: the sum divided by how many values there are. The median is the middle value once you sort the values. One extreme value pulls the mean, but hardly moves the median. The standard deviation says how far the values typically lie from the mean, and a quantile says below which value a part of them lies. A histogram shows all values at once.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **median** | the middle value once you sort the values: half of the values lie below it, half above | `np.median(x)` |
    | **outlier** | a value far away from the rest; it may be a mistake or a rare real case, so look before you remove it | |
    | **standard deviation** | how far the values typically lie from the mean, in their own unit; small means close together | `x.std()` |
    | **quantile** | the value below which a given fraction of the values lies; the median is the 0.5 quantile | `np.quantile(x, 0.9)` |
    | **`.min()`, `.max()`** | the smallest and the largest value | `x.max()` |
    | **histogram** | a chart that splits the range of the values into equal bins, and draws a bar for how many values fall in each | `plt.hist(x, bins=...)` |
    | **bin** | one of those equal parts of the range | `bins=range(0, 101, 10)` |

    `range(0, 101, 10)` counts from 0 to 100 in steps of 10: the third number is the step, and 101 makes sure that 100 is included. Those are the edges of the bins.

    The standard deviation takes four steps: the difference of each value from the mean, `x - x.mean()`; each difference squared, so that a difference below the mean, which is negative, does not cancel one above it; the mean of those squares; and its square root, `** 0.5`, to get back to the unit of the values.
    """)
    return


@app.cell(hide_code=True)
def picture_times():
    mo.vstack([
        mo.Html(C.picture({"title": "Travel times to school", "x": "minutes", "y": "students",
                           "median": "median, 25", "mean": "mean, 30.7",
                           "alt": "A histogram of 15 travel times, with bars of 10 minutes: 3 students took 10 to 20 minutes, 6 took 20 to 30, 3 took 30 to 40, 2 took 40 to 50, and 1 took 90 to 100. A solid line marks the median at 25, a dashed line the mean at 30.7."})),
        mo.md(r"""
        The 15 travel times as a histogram: a bar for every 10 minutes, as high as the number of students in it. Most trips took 10 to 50 minutes; one took 95. The median, 25, is in the middle of the big group. The mean, 30.7, is pulled to the right by that one long trip.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`x.std()` and `np.std(x)` divide by the number of values. Some other tools divide by one less, so on the same numbers they give a slightly larger answer. Both are called standard deviation; say which one you use when it matters."),
        mo.md(r"""
        ### Recipe

        To describe a column of numbers:

        1. Draw a histogram, to see the shape and any outliers: `plt.hist(x, bins=...)`.
        2. Find the middle: `x.mean()` and `np.median(x)`. If they differ a lot, look for outliers.
        3. Find the spread: `x.std()`, the smallest and largest value, and a quantile such as `np.quantile(x, 0.9)`.
        4. Say what you see in a sentence, with the unit: "half of the trips take less than 25 minutes".
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The travel times of the 15 students, in minutes. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_times():
    import matplotlib.pyplot as plt
    import numpy as np

    _times = np.array([25, 18, 32, 20, 95, 27, 12, 40, 22, 25, 35, 15, 30, 20, 45])
    # 1. the histogram: a bar for every 10 minutes
    plt.hist(_times, bins=range(0, 101, 10))
    plt.xlabel("minutes")
    plt.ylabel("students")
    plt.title("Travel times to school")
    plt.show()
    # 2. the middle
    print(_times.mean())
    print(np.median(_times))
    # 3. the spread
    print(_times.std())
    print(_times.min())
    print(_times.max())
    print(np.quantile(_times, 0.9))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The histogram has one big group from 10 to 50 minutes and one lonely bar at 90 to 100: the student who missed the bus. The mean is 30.7 minutes and the median 25: the long trip pulls the mean up. The standard deviation is 19.3 minutes; without the trip of 95 minutes it would be 9.1, so one outlier doubles it. The trips go from 12 to 95 minutes. The 0.9 quantile is 43: 90% of 15 trips is 13.5, so that place falls between the 13th and the 14th of the sorted times, 40 and 45, and NumPy takes a value in between. Step 4, in a sentence: "half of the students are at school within 25 minutes, and nearly all within 45; one student took 95, after missing the bus".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_house():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A typical house
    """)
    return


@app.cell(hide_code=True)
def choice_house_cell():
    choice_house = L.choice(
        "typical-house",
        question="The five houses in a street cost 210, 230, 250, 260 and 1,450 thousand euros. Which number says best what a typical house in that street costs?",
        options={
            "mean": "The mean: 480 thousand",
            "median": "The median: 250 thousand",
            "highest": "The highest price: 1,450 thousand",
        },
        correct="median",
        why={
            "mean": "Four of the five houses cost far less than 480 thousand. What pulls the mean up?",
            "highest": "That is one house, the most expensive one. Does it say what most houses cost?",
            "median": "Two houses cost less and two cost more. The one very expensive house pulls the mean up, but the median stays in the middle.",
        },
    )
    choice_house
    return (choice_house,)


@app.cell(hide_code=True)
def text_exercise_slow():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Time to allow

        The box has the travel times as `times`. The school wants to know how much time most students need: make `slow`, the time within which 3 of every 4 trips are done. Replace the `...`. The last line is `slow`, so the check sees it. The check also tries another group of students.
        """),
        L.task("Make `slow`: the time within which 3 of every 4 trips in `times` are done."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slow():
    box_slow = L.code_box("slow-trips", lines=5)
    box_slow
    return (box_slow,)


@app.cell(hide_code=True)
def check_slow(box_slow):
    result_slow = L.check("slow-trips", box_slow.value)
    result_slow
    return (result_slow,)


@app.cell(hide_code=True)
def hints_slow():
    L.hints(
        "slow-trips",
        [
            "Step 3 of the recipe: a quantile, below which a given part of the values lies.",
            "Which function in the table of new words takes the array and a fraction from 0 to 1? Which fraction is 3 of every 4?",
            "`slow = np.___(times, ___)`.",
        ],
        why="3 of every 4 is the fraction 0.75, and `np.quantile(times, 0.75)` is the time below which 3 of every 4 trips lie: 33.5 minutes. That place falls between two of the sorted times, 32 and 35, so NumPy takes a value in between.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The standard deviation, step by step

        Finish `spread(values)`. It works out the standard deviation of a list of numbers itself, in the four steps under the table of new words, without `std`. Replace the `...`. The check calls `spread([2, 4, 4, 4, 5, 5, 7, 9])`: that should give `2.0`. The check also tries other lists.
        """),
        L.task("Make `spread(values)` work out the standard deviation of `values`, step by step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("spread", lines=8)
    box_spread
    return (box_spread,)


@app.cell(hide_code=True)
def check_spread(box_spread):
    result_spread = L.check("spread", box_spread.value)
    result_spread
    return (result_spread,)


@app.cell(hide_code=True)
def hints_spread():
    L.hints(
        "spread",
        [
            "Four steps: the differences from the mean, squared, their mean, and the square root.",
            "`x - x.mean()` gives the difference of each value from the mean, as an array. Which power squares every number of an array at once, and which one takes the square root?",
            "`differences = x - x.mean()`, and then `return (differences ___).mean() ___`.",
        ],
        why="The mean of the example is 5. The differences from it are -3, -1, -1, -1, 0, 0, 2 and 4; squared, 9, 1, 1, 1, 0, 0, 4 and 16. Their mean is 32 / 8 = 4, and the square root of 4, `4 ** 0.5`, is 2.0.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Sums and means along an axis](/course/open/np-axis?lang=en) you worked out a mean; now you also know a middle that one outlier cannot pull away, and how to measure the spread. In [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) you drew a bar for each name; a histogram draws a bar for each bin. Later in the course, a model is judged by how far off it is on average, and the standard deviation comes back to put numbers on the same scale. In the next lesson, *Asking questions of data*, you use these numbers to look at a whole dataset before you build a model on it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_house, result_slow, result_spread):
    L.wrap_up(choice_house, result_slow, result_spread)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
