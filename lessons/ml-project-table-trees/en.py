# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-project-table-trees. Route: course/route/lessen/klassiek-ml.yaml. Checks, the folds, the play block and the demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-project-table-trees", "en")
    C = L.checks   # the clean traffic table, the five folds, what each model misses, the pictures and the demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Testing a forecast on the past](/course/open/ts-backtesting?lang=en) and [Project: which model is best? A fair comparison](/course/open/ml-project-fair-comparison?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_origin():
    quick_origin = L.choice(
        "quick-origin",
        question="You backtest a forecast of daily riders with a rolling origin. At the origin Monday 2 July 2012, it forecasts the week after. Which days may it learn from?",
        options={
            "week": "Every day except the week it forecasts",
            "right": "Only the days before 2 July 2012",
            "all": "Every day: it is tested on the week after anyway",
        },
        correct="right",
        why={
            "week": "The days after that week were not known on 2 July either. A forecast may use only what was known at its origin: the days before it. See [Testing a forecast on the past](/course/open/ts-backtesting?lang=en).",
            "all": "Then it learns the very week it is tested on, and its score says nothing about a real forecast. Only the days before the origin. See [Testing a forecast on the past](/course/open/ts-backtesting?lang=en).",
            "right": "Only the past. Today, the origins are on 1 January of each year, and a model learns from every hour before.",
        },
    )
    quick_origin
    return (quick_origin,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="On the penguins, logistic regression scored an accuracy of 0.91 on one random split, and k-NN 0.89 on another random split. Which model is better?",
        options={
            "logistic": "Logistic regression: its accuracy is 0.02 higher",
            "equal": "They are equally good: 0.02 is too small to matter",
            "right": "You cannot tell yet: compare them on the same folds",
        },
        correct="right",
        why={
            "logistic": "Each was scored on its own split, and a split with easier penguins makes a model look better. On the same folds, the best four models were about one penguin per fold apart. See [Project: which model is best? A fair comparison](/course/open/ml-project-fair-comparison?lang=en).",
            "equal": "Maybe, but these two numbers cannot say so: they come from different penguins. The same folds, fold by fold, can. See [Project: which model is best? A fair comparison](/course/open/ml-project-fair-comparison?lang=en).",
            "right": "The same folds for every model, and the differences fold by fold. Today you do that for models of the traffic, with folds in time.",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your clean table of traffic gets models. In [The default for tables: gradient-boosted trees](/course/open/ml-gbdt-default?lang=en), histogram gradient boosting beat a random forest by 14.5 vehicles an hour. That was one split, at 1 January 2017: one test period. Would boosting also win in another year? And does either model beat a simple forecast that knows the week: the traffic of the same hour a week earlier? A fair test answers both. Today your project gets its three parts: a baseline, folds in time, and one function that scores every model on the same hours, fold by fold. At the end, you write down what wins, and by how much.

        **An everyday example.** You want to know whether a new route to work is faster. You try it on a Friday in August, and it saves 6 minutes; but on a Friday in August the roads are quiet anyway. A fair test takes the old route and the new one on the same kinds of days, week after week, and writes down for every week which was faster, and by how many minutes. One Friday cannot tell you; ten weeks can.
        """),
        draw.project_steps(["Your table, explored and cleaned", "Trees on your table, fairly"], 1,
                           here_label="← today", title="Where your project stands: trees on your table"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_years():
    guess_years = L.guess(
        "years",
        question="Repeat the split of *The default for tables* once for every year from 2014 to 2018: learn from the years before, test on the year itself. In how many of the five years does boosting beat the forest?",
        options={
            "few": "In one or two",
            "most": "In three or four",
            "right": "In all five",
        },
        reveal="**In all five**, but by very different amounts: by 3.7 vehicles an hour in 2015, by 25.2 in 2018. One split would have shown you one of those numbers.",
    )
    guess_years
    return (guess_years,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The question.** How much traffic will there be in an hour, given its calendar and its weather? A model learns from the hours before a moment, the origin, and is tested on the hours after it. It gets the weather of every hour as it was measured. A real forecast a week ahead would only have a weather forecast, which is less exact, so the scores of this lesson are a little better than a real forecast would get.

        **Folds in time.** In [Testing a forecast on the past](/course/open/ts-backtesting?lang=en), the origin moved on a week at a time. Here it moves on a year at a time: an origin on 1 January of each year from 2014 to 2018, a model that learns from every hour before it, and a test on the hours of that year. That makes five folds, and the hours to learn from grow from 9,397 to 34,031. `X.index.year` gives the year of every hour, so `X.index.year < 2016` is the mask of the hours before 2016. The year 2015 has rows only from 11 June, after the hole you found in the first step of the project.

        **A baseline for hours.** The seasonal naive forecast says that every hour ahead is like the same hour one season earlier. The season of the traffic is a week: 168 hours. On the clean table itself, 168 rows back is not always a week back, because 11,987 hours have no row: the 11,976 of [Time as an index](/course/open/ts-time-index?lang=en), and the 11 impossible hours your `clean` drops. On `y.asfreq("h")`, every hour has a row, so `shift(168)` gives the same hour a week earlier, with NaN where that hour has no row: for 3,066 of the 40,564 hours. `before[y.index]` then takes the values at the hours of the table again. The week before knows the traffic of a week ago, and the models only know the years before their origin: a model that beats it anyway is worth its trouble.

        **The same hours for every model.** A model can be scored on every test hour; the week before only where the hour a week earlier has a row. For a fair comparison, every model is scored on the same hours: the test hours where the week before is known, `test & base.notna()`. In 2016, those are 7,144 of the 7,837 test hours.

        **Fold by fold.** As with the penguins, the mean of five folds hides what happens in each of them. Write down, for every pair of models, in how many folds one beats the other and by how much, and look hard at a fold that goes against the others.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | `.year` | the year of every moment in a datetime index, as `.hour` gives its hour | `X.index.year < 2016` |
        | `&` | for two masks of the same length: True where both are True; a comparison beside it goes between parentheses | `test & base.notna()`, `test & (X.index.hour == 8)` |
        | `s[y.index]` | the values of the Series `s` at the moments in `y.index` | `before[y.index]` |
        """),
        L.watch_out("Random folds, as with the penguins, look better here than they are: hours of the same weeks end up on both sides, and a model learns from hours after the ones it is tested on. On the same 29,283 hours of 2014 to 2018, boosting misses by 251.0 vehicles an hour with five random folds, and by 284.8 with the five year folds. The year folds come closer to what a model can do in a year it has never seen."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Slide the origin through the years. Left: every month of the table, its hours with a row; filled, the months the models learn from; open, the year they are tested on; gray, the months after it. Right: what the week before, the forest and boosting miss on the same hours of that year.
    """)
    return


@app.cell(hide_code=True)
def explore_origin_cell():
    def _frame(year):
        year = int(year)
        _said = C.said(year, L.lang)
        _alt = ("Left: the hours with a row in every month of the table, the months to learn from filled, the year to test on "
                "open, the later months gray. Right: the MAE of the week before, the forest and boosting on the same hours. "
                + _said.replace("**", ""))
        return C.picture(year, C.play_words(L.lang) | {"alt": _alt}, L.lang), _said

    explore_origin = L.explore("origin", "year tested on", _frame, start=2017, values=C.YEARS, role="hyper",
                               task="Find the year in which the forest misses more than the week before. In that year, by how much does boosting beat the week before?")
    explore_origin
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Boosting beats the forest in all five years, by 3.7 to 25.2 vehicles an hour, and the week before by 38.8 to 93.5 in four of them. In 2016 it beats the week before by only 1.5 vehicles an hour, a tie, while the forest misses 22.3 more than the week before. From July to November 2016, the traffic on weekdays from 6:00 to 19:00 was 387 to 457 vehicles an hour lower than in the same months of 2015 and 2017. The models learned only from the years before, so they kept expecting the old traffic: in July 2016, boosting guessed 502.2 vehicles an hour too many on average. The week before follows the change one week later. A single split at 1 January 2017 would never have shown you that year.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To compare models of a table in time:

    1. Choose the folds in time: an origin on 1 January of each year, learn from every hour before it, test on the hours of that year.
    2. Make a baseline that uses only what was known: the same hour one season earlier, on the grid of every hour, `y.asfreq("h").shift(...)`.
    3. Score every model on the same hours: the test hours where the baseline is known.
    4. In every fold, fit every model on the hours to learn from, and score it on the hours of step 3.
    5. Write down per fold what wins, and by how much, not only the mean; and look for the reason for a fold that goes against the others.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes the clean table and the features of [The default for tables: gradient-boosted trees](/course/open/ml-gbdt-default?lang=en), puts one origin at 1 January 2017, as there, and compares boosting with a baseline: the same hour a *day* earlier. It scores both on the test hours that have a row a day earlier. It prints the hours to learn from, the test hours and the hours both are scored on; the two MAEs; and what the day before misses on every weekday, Monday first. `threadpool_limits(1)` lets scikit-learn calculate on one core, as in *The default for tables*. Run the cell.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    # the clean table of the project, with the time as the index
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df[_df["temp"] > 0]
    _df = _df[_df["rain_1h"] <= 100]
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _df["temp"] = _df["temp"] - 273.15
    _df = _df.set_index("date_time")

    # the features of The default for tables, and the label
    _X = pd.DataFrame(index=_df.index)
    _X["hour"] = _df.index.hour
    _X["weekday"] = _df.index.dayofweek
    _X["month"] = _df.index.month
    _X["temp"] = _df["temp"]
    _X["rain"] = _df["rain_1h"]
    _X["snow"] = _df["snow_1h"]
    _X["clouds"] = _df["clouds_all"]
    _X["weather"] = _df["weather_main"].astype("category")
    _y = _df["traffic_volume"]

    # 1. one origin: learn from the hours before 2017, test on the hours from then on
    _train = _X.index < "2017-01-01"
    _test = _X.index >= "2017-01-01"

    # 2. a baseline: the same hour a day earlier, on the grid of every hour
    _every = _y.asfreq("h")
    _day = _every.shift(24)[_y.index]

    # 3. the same hours for both: the test hours with a row a day earlier
    _known = _test & _day.notna()
    print(_train.sum(), _test.sum(), _known.sum())

    # 4. fit on the hours to learn from, score both on the same hours
    with threadpool_limits(1):
        _boost = HistGradientBoostingRegressor(random_state=0).fit(_X[_train], _y[_train])
    print(round(np.mean(abs(_y[_known] - _day[_known])), 1), round(np.mean(abs(_y[_known] - _boost.predict(_X[_known]))), 1))

    # 5. the day before, weekday by weekday: Monday is 0
    _by_day = []
    for _d in range(7):
        _on = _known & (_X.index.dayofweek == _d)
        _by_day.append(float(round(np.mean(abs(_y[_on] - _day[_on])), 1)))
    print(_by_day)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    25,318 hours to learn from, 15,246 to test on, and 15,183 of those have a row a day earlier. On them, the day before misses by 558.6 vehicles an hour, boosting by 260.3. Weekday by weekday, the day before misses by 1,155.2 on Mondays and by 1,058.3 on Saturdays: a Monday is not like the Sunday before it, and a Saturday is not like the Friday. From Tuesday to Friday, it misses by 256.6 to 375.9. The season of the traffic is the week. A baseline that is easy to beat says little, so the baseline of your project is the same hour a week earlier.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The week before

        `week_before(y)` should give, for every hour of the Series `y`, the traffic of the same hour a week earlier, and NaN where that hour has no row. Replace the two `...`: `every` with a row for every hour from the first to the last, and `before` with every value moved a week later. The last line takes the values at the hours of `y` again. The check calls `week_before(traffic)`, with the traffic of the clean table and the time as the index, and tries three made-up Series of hours. When it is right, the check saves it in your project, in `work/projects/tabular/baseline.py`.
        """),
        L.task("Give every hour a row, then move every value a week later."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week():
    box_week = L.code_box("week-before", lines=5)
    box_week
    return (box_week,)


@app.cell(hide_code=True)
def check_week(box_week):
    result_week = L.check("week-before", box_week.value)
    result_week
    return (result_week,)


@app.cell(hide_code=True)
def hints_week():
    L.hints(
        "week-before",
        [
            "Step 2 of the recipe; step 2 of the worked example did it for a day. What changes for a week?",
            "`asfreq` gives every hour a row, as in *Time as an index*; on that grid, `shift` counts hours, and a week has 7 days of 24 hours.",
            "`every = y.asfreq(___)` and `before = every.shift(___)`.",
        ],
        why="On the grid of every hour, a row is an hour, so moving the values 168 rows later puts beside every hour the traffic of the same hour a week before, and NaN where that hour had no row. `before[y.index]` keeps only the hours that are in the table.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Folds in time

        `year_folds(X, years)` should give a list with one fold for every year in `years`: a pair of masks, `train` for the hours before 1 January of that year and `test` for the hours of the year. Replace the two `...`. The check calls `year_folds(X, [2014, 2015, 2016, 2017, 2018])`, with the features of the clean table, and tries a small made-up table with other years, among them a year without hours before it and a year without any hours. It saves your function in `work/projects/tabular/folds.py`.
        """),
        L.task("Learn from every hour before the year; test on the hours of the year."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=7)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "Step 1 of the recipe. Step 1 of the worked example made one origin by comparing `_X.index` with a moment: what do you compare to get the hours before a year, and the hours of that year?",
            "`X.index.year` gives the year of every hour; compare it with `year`.",
            "`train = X.index.year ___ year` and `test = X.index.year ___ year`.",
        ],
        why="Each fold learns only from what was known at its origin, every hour before 1 January, and is tested on the year that follows: five origins, five tests, and never a test on an hour the model learned from.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Every model on the same hours

        `fold_scores(model, X, y, years)` takes your two functions from your project: `week_before` for the baseline and `year_folds` for the folds. In every fold, it should fit the model on the hours to learn from and give its MAE on the test hours that have a row a week earlier, the hours the week before is scored on. Replace the two `...`. The check calls `fold_scores(boosting, X, traffic, [2014, 2015, 2016, 2017, 2018])`, where `boosting` is histogram gradient boosting with at most 30 trees, so that the check takes a few seconds; it also tries two other models, one of them on a small made-up table. Where a module of yours is missing or broken, the course's version stands in, as the note under *Where your project stands* says. When it is right, the check saves it in `work/projects/tabular/scores.py`.
        """),
        L.task("In every fold: fit on the hours before the year, and score on the test hours where the week before is known."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("scores", lines=15)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "scores",
        [
            "Steps 3 and 4 of the recipe, as in steps 3 and 4 of the worked example, now inside the loop. Which hours does the model learn from, and which hours is it scored on?",
            "Two masks, one for the test hours and one for the hours where `base` is not NaN, combined with `&`. And which mask of the fold holds the hours to learn from?",
            "`known = test & ___` and `model.fit(X[___], y[___])`.",
        ],
        why="The model of each fold sees only the hours before its year, and every model, the week before too, is scored on exactly the same hours. A difference between two scores is then a difference between the models, not between the hours they were scored on.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell takes your project as it is: your `clean` on the whole table, the features of *The default for tables*, and your `fold_scores` for boosting with its default settings and for a random forest of 100 trees, with one-hot columns for the word, in the five folds; your `week_before` and `year_folds` give the week before on the same hours. Ten models learn, and the forests take a while. Press the button.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Compare on your project")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_cell(demo_button, result_scores):
    _r = result_scores   # so that this cell runs again after every check of exercise 3
    if not demo_button.value:
        _out = mo.md("*Press the button to compare the models with your project.*")
    else:
        _result = C.demo()
        if _result is None:
            _out = mo.md("*Finish exercise 3 first: then your `fold_scores` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _out = mo.md("*Your project stops with an error, or does not give a number for every fold. Did you change one of its files in `work/projects/tabular/`? Delete it, and check its exercise again: the check then saves your code again.*")
        else:
            _who = "" if not _result["stand_ins"] else "The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says. "
            _same = all(round(a, 1) == b for _k, _i in (("week", 0), ("forest", 1), ("boost", 2))
                        for a, b in zip(_result[_k], [C.MAE[_y][_i] for _y in C.YEARS]))
            _out = mo.vstack([
                mo.Html(C.report_chart(_result, L.lang)),
                mo.md(C.report(_result, L.lang)),
                mo.md(_who + ("These are the numbers of the slider: your project now makes the fair comparison itself." if _same else "")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 4. What you write down
    """)
    return


@app.cell(hide_code=True)
def choice_report_cell():
    choice_report = L.choice(
        "report",
        question="Your report for 2016 says: the week before misses by 344.2 vehicles an hour, the forest by 366.5, boosting by 342.7. Month by month, boosting misses less than the week before in 6 of the 12 months. What do you write down for 2016?",
        options={
            "won": "Boosting won 2016, by 1.5 vehicles an hour, and the forest came last",
            "right": "A tie with the week before, and the forest lost: from July to November the traffic was lower",
            "drop": "Leave 2016 out of the report: the traffic was unusual that year, so it says nothing",
        },
        correct="right",
        why={
            "won": "1.5 vehicles an hour, on hours that both miss by about 340, is a difference nobody would notice, and month by month it goes both ways, 6 months each. A report that says *won* promises more than the numbers show: write *a tie*, with the numbers.",
            "drop": "Then the report hides the one year in which a model that learned from the past was surprised, and that can happen again. A fold that goes badly is a finding: write it down, with what you know about why.",
            "right": "The numbers, and what they mean: in 2016 boosting was no better than looking up last week, because the traffic changed that year. In the four other years, boosting beat both.",
        },
    )
    choice_report
    return (choice_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The folds are the rolling origin of [Testing a forecast on the past](/course/open/ts-backtesting?lang=en), with a year as the step and a model that learns at every origin; the same hours for every model, a baseline and the differences fold by fold come from [Project: which model is best? A fair comparison](/course/open/ml-project-fair-comparison?lang=en). The model to try first, from [The default for tables: gradient-boosted trees](/course/open/ml-gbdt-default?lang=en), beats the forest in every year, but by 3.7 to 25.2 vehicles an hour, and one year shows that even a good model can be surprised by a change. Your project `tabular` now has six modules: `impossible.py`, `hours.py` and `clean.py` from [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en), and `baseline.py`, `folds.py` and `scores.py`. With this lesson, you have finished the basics of classical machine learning.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_report, result_folds, result_scores, result_week):
    L.wrap_up(result_week, result_folds, result_scores, choice_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
