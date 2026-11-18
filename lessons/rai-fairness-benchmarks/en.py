# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-fairness-benchmarks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fairness-benchmarks", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading AI claims critically](/course/open/res-reading-claims?lang=en) and [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_benchmark():
    quick_benchmark = L.choice(
        "quick-benchmark",
        question="A model scores 0.90 on a benchmark of 500 tasks, the highest score so far. What does that score show?",
        options={
            "all": "How it does on every task of that kind",
            "right": "How it does on those tasks, scored that way",
            "best": "That it is better than all other models at the task",
        },
        correct="right",
        why={
            "all": "A benchmark is a fixed set of test tasks. Its score says how the model did on those tasks, scored that way; whether that carries over to other tasks of the same kind, the benchmark does not tell you. See *The idea* in [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
            "right": "One number, on those tasks, scored that way: that is what was measured. A fairness benchmark is no different.",
            "best": "The highest score so far means the model did best on those tasks, scored that way, of the models measured on them. Better at the task in general was not measured. See *The idea* in [Reading AI claims critically](/course/open/res-reading-claims?lang=en).",
        },
    )
    quick_benchmark
    return (quick_benchmark,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="After a search, a model gets a threshold of 0.38 for women and 0.52 for men. What changed in the model itself?",
        options={
            "weights": "Its weights, through a new training run",
            "right": "Nothing: only the decision changed",
            "labels": "The labels it learned from, for some women",
        },
        correct="right",
        why={
            "weights": "The model was not trained again, and no weight changed: only the thresholds its probabilities are compared with. See *The idea* in [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en).",
            "right": "The model gives every person the same probability as before; only the threshold that probability is compared with depends on the group.",
            "labels": "Changing labels is massaging, before training. A threshold per group comes after training and changes no label and no weight. See *The idea* in [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lessons you worked out yes rates, recalls, weights and thresholds yourself, in a few lines of Python. In practice, libraries do these computations, and two well-known ones are Fairlearn and AIF360. And when a paper says that a new method makes models fairer, it often shows it on the same few tables: UCI Adult, COMPAS and German Credit. To read such a claim, you need to know what the libraries compute, what those tables are, and how much one result on them says. The demo shows how much the gain of one method changes when you take other people from the same table.

    **An everyday example.** A car maker states how much fuel a car uses, measured on a fixed test drive. Every car does the same drive, so you can compare cars. But your own drives, in winter, in the hills or with a full car, use more, and the car that wins the test drive does not always win on your roads. A fairness benchmark is such a test drive: the same for every method, but not the road where the method will be used.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="On the whole table, reweighing raised the women's yes rate divided by the men's from 0.28 to 0.57. Now take 1,000 people from the table, as many as German Credit has, train on 700 of them and measure on the other 300, ten times with other people. By how much does reweighing raise that ratio each time, do you think?",
        options={
            "same": "About 0.29 each time, as on the whole table",
            "right": "Anything from about 0.07 to about 0.51",
            "worse": "It lowers the ratio in some of the samples",
        },
        reveal="**Anything from about 0.07 to about 0.51.** Of 300 people to test, only about 100 are women, and without weights only 5 to 13 of them get a yes: one woman more or less moves the ratio by about 0.05. The demo below shows it.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Two libraries.** **Fairlearn** is a Python library for measuring and improving fairness, under the MIT license; its latest version, 0.15.0, came out in October 2026. **AIF360**, AI Fairness 360, calls itself "an extensible open-source library containing techniques developed by the research community to help detect and mitigate bias in machine learning models throughout the AI application lifecycle"; it is under the Apache 2.0 license, and its latest version, 0.6.1, is from April 2024. AIF360 has reweighing as `Reweighing`, and Fairlearn has threshold optimization as `ThresholdOptimizer`. At their core is what you built by hand. Fairlearn's `MetricFrame`, a "collection of disaggregated metric values", works out a measure per group, in `by_group`, and for everyone, in `overall`. Fairlearn's `selection_rate` is the yes rate: "the fraction of predicted labels matching the 'good' outcome". This course does not install Fairlearn, so the code below is only to read. It is the example of Fairlearn's documentation:

    ```python
    from fairlearn.metrics import MetricFrame, selection_rate

    y_true = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    sex = ["Female", "Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male"]
    frame = MetricFrame(metrics={"selection_rate": selection_rate},
                        y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    frame.by_group        # Female 0.8, Male 0.4
    frame.difference()    # 0.4: the largest minus the smallest
    frame.ratio()         # 0.5: the smallest divided by the largest
    ```

    The selection rate does not need `y_true`; Fairlearn asks for it anyway, "for consistency with other metric functions" such as the recall, which does. Exercise 1 works out `by_group` of this example by hand.

    **Same name, other numbers.** Read which formula a tool uses. Fairlearn's `demographic_parity_difference` is "the difference between the largest and the smallest group-level selection rate", so it is never negative, and its `demographic_parity_ratio` divides the smallest by the largest, so it is never above 1. AIF360 asks you which group is privileged and which unprivileged. Its `statistical_parity_difference` is the yes rate of the unprivileged group minus that of the privileged group, and its `disparate_impact` divides the one by the other. By those formulas, the example above with women as the unprivileged group gives 0.8 minus 0.4, that is 0.4, and 0.8 divided by 0.4, that is 2: the same predictions give a ratio of 0.5 in one library and 2 in the other. And neither library decides what is fair. Fairlearn's guide says of its own methods that they "will provide mathematical guarantees as to how close they can drive some unfairness metric to zero. However, this does not mean that the results are fair."

    **The tables everyone uses.** A **fairness benchmark** is a fixed table with a protected attribute, such as sex or race, and a task, on which fairness methods are compared. Alessandro Fabris and colleagues went through 558 articles from major venues with fairness words in the title, from 2014 to June 2021: UCI Adult was used 118 times, COMPAS 81 times and German Credit 35 times. They think the three probably became the standard because early, much-read papers and articles used them. Each has its own problem.

    - **UCI Adult** you know: 48,842 people from a US survey of 1994, and the label says whether someone earns more than 50,000 dollars. Frances Ding and colleagues point out that this threshold "corresponds to the 76th quantile of individual income in the United States in 1994, the 88th quantile in the Black population, and the 89th quantile among women." So almost all women earn less, and a no is mostly right for them. They call what follows "a rather untypical situation": a standard model is right more often for women and for Black people than for everyone, while models often do worse for groups that were disadvantaged. The model of the previous lessons too: on the test set it is right for 0.916 of the women and 0.794 of the men. When they moved the threshold, "both accuracy and the magnitude of violations of these criteria vary substantially with the threshold choice."
    - **German Credit** holds 1,000 loans from 1973 to 1975 from a bank in southern Germany, 700 good and 300 bad. Ulrike Grömping writes that the bad loans are "heavily oversampled": in reality about 5 percent were bad. Papers use the table to compare women and men, but it cannot tell who is a woman. Its code table suggests that one column holds sex and marital status together, but Grömping found the code table wrong: "sex cannot be recovered from the variable, because male singles and female non-singles are coded with the same code (2)". In 2019 Grömping published a corrected version, South German Credit. Papers derived sex anyway, among them the 2018 preprint of Sorelle Friedler and colleagues, which comes back below. Fabris and colleagues: "Experiments that claim extracting a sex variable from the German Credit dataset should be considered noisy at best."
    - **COMPAS** you know from [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en): defendants in Broward County, Florida, from 2013 and 2014. Its label says whether someone was arrested again, not whether they committed a crime again, and where the police look affects who gets arrested. Michelle Bao and colleagues advise that "researchers working broadly in algorithmic fairness should avoid the use of CJ-related datasets (e.g., COMPAS) as generic real-world examples to illustrate or benchmark a new fairness algorithm or measure"; CJ is criminal justice. They add: "This paper does not advocate completely avoiding datasets from CJ or quantitative methods for real-world problems."
    - **Folktables**, from Ding and colleagues in 2021, is a newer answer: tables from the American Community Survey, a yearly survey sent to about 3.5 million US households, for all fifty states and Puerto Rico and the years 2014 to 2018, with tasks such as income and employment. Its income task, ACSIncome, has 1,664,500 people for 2018 alone, and you choose the income threshold yourself. It is still one country: "A notable if obvious limitation of our work is that it is entirely US-centric."

    **One table, one split, one result.** Ding and colleagues compared methods across states: "Training on one state and testing on another generally leads to unpredictable results. Accuracy and fairness criteria could change in either direction." Friedler and colleagues compared fairness methods on Adult, German Credit, a table called Ricci and two COMPAS tables, each with ten random splits into people to train on and people to test: "It is clear that there is no one "winner" - no algorithm that is both more fair and more accurate than the others on all datasets. It is also clear that there is tremendous variation even within a single algorithm over the random splits it receives." Their advice: "We recommend reporting algorithm success and stability based on a moderate number of randomized training-test splits."

    **Try it.** The cell below does what Friedler and colleagues recommend, for reweighing, on samples as small as German Credit. It takes 1,000 people of UCI Adult, ten times, with `np.random.default_rng(_seed).permutation` and the seeds 0 to 9: 700 to train on and 300 to test. `_ratio` trains the model of the previous lessons on the rows at the places `train`, without weights or with reweighing as in [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en), and gives on the rows `test` how often women get a yes divided by how often men do. Run the cell; it takes a second or two.
    """)
    return


@app.cell
def demo_samples():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _rich = _people["income"] == ">50K"
    _poor = _people["income"] == "<=50K"
    _women = _people["sex"] == "Female"
    _men = _people["sex"] == "Male"

    # the columns of the model without sex, as in the previous lessons
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()


    def _ratio(train, test, reweigh):
        # train on the rows at the places train, with or without reweighing, and measure on the rows test:
        # how often women get a yes, divided by how often men do
        weight = None
        if reweigh:
            weight = 0
            for sex in [_women.iloc[train], _men.iloc[train]]:
                for label in [_rich.iloc[train], _poor.iloc[train]]:
                    expected = sex.sum() * label.sum() / len(train)
                    weight = weight + expected / (sex * label).sum() * (sex * label)
        model = LogisticRegression(max_iter=1000)
        model.fit(_X.iloc[train], _rich.iloc[train], sample_weight=weight)
        yes = model.predict_proba(_X.iloc[test])[:, 1] > 0.5
        return yes[_women.iloc[test]].mean() / yes[_men.iloc[test]].mean()


    # ten samples of 1,000 people, as many as German Credit has: 700 to train on, 300 to test
    for _seed in range(10):
        _pick = np.random.default_rng(_seed).permutation(len(_people))[:1000]
        _plain = _ratio(_pick[:700], _pick[700:], False)
        _reweighed = _ratio(_pick[:700], _pick[700:], True)
        print("seed", _seed, "| without weights", round(_plain, 2),
              "| with reweighing", round(_reweighed, 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_samples({"title": "One method, ten samples of 1,000 people",
                                   "whole": "whole table", "seed": "seed",
                                   "xlabel": "women's yes rate divided by men's", "decimal": ".",
                                   "key": "open: without weights, filled: with reweighing. 1 would mean that women get a yes as often as men. The top row is the whole table, from Fixing the data: reweighing.",
                                   "alt": "Eleven rows, each with an open dot without weights and a filled dot with reweighing. Whole table: 0.28 and 0.57. Seeds 0 to 9 without weights: 0.45, 0.36, 0.24, 0.56, 0.32, 0.55, 0.56, 0.31, 0.34, 0.22; with reweighing: 0.72, 0.57, 0.46, 0.63, 0.64, 0.78, 0.75, 0.82, 0.6, 0.67."})),
        mo.md(r"""
        The same table, the same method and the same model: only the people differ. Without weights, the ratio lies between 0.22 and 0.56; with reweighing, between 0.46 and 0.82. With seed 3, reweighing raised it only from 0.56 to 0.63, with seed 7 from 0.31 to 0.82. On the whole table it went from 0.28 to 0.57. The cause is the small numbers. Of the 300 people to test, 90 to 109 are women, and without weights only 5 to 13 of them get a yes: one woman more or less moves the ratio by about 0.05. And of the 700 people to train on, only 21 to 33 are women who earn more, so the weights themselves change from sample to sample. Here reweighing raised the ratio in all ten samples; by how much, one split of a table this small cannot tell you. A paper that reports one split of a table this small could be off by as much.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **fairness benchmark** | a fixed table with a protected attribute and a task, such as UCI Adult, COMPAS or German Credit, on which fairness methods are compared |
        | **Fairlearn** | a Python library for fairness, under the MIT license; its `MetricFrame` works out a measure per group |
        | **AIF360** | AI Fairness 360, a library for fairness in Python and R, under the Apache 2.0 license; you tell it which group is privileged |
        | **Folktables** | tables from a yearly US survey, per state and year, as a newer fairness benchmark for income, employment and more |
        """),
        L.watch_out("A method that wins on these tables has shown that it wins on these tables. Fabris and colleagues conclude about Adult, COMPAS and German Credit: \"Overarching claims of significance in real-world scenarios stemming from experiments on these datasets should be met with skepticism.\""),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To read a fairness result, from a paper or from a tool:

    1. Which table: from which year and country, with how many people, and what does the label measure? Can the table tell who belongs to each group at all?
    2. Which formula: a difference or a ratio, the largest against the smallest, or the unprivileged group against the privileged group?
    3. How many splits, and how far apart are the results? One split of a small table says little.
    4. On which other tables, and on data like the data where the model will decide?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Eight made-up people in two groups, a and b, and the model's answer for each, 1 for a yes. `_people` and `_yes` start at 0 for both groups. The loop goes through the people by their place `_i`, from 0 to 7: `_g` is the group of person `_i`, `_people[_g]` counts one more person in that group, and `_yes[_g]` adds the answer. Then `_rate` holds the selection rate per group, and the last two lines work out Fairlearn's difference and ratio.
    """)
    return


@app.cell
def example_rates():
    # eight made-up people: their group and the model's answer, 1 for a yes
    _group = ["a", "a", "b", "a", "b", "b", "a", "b"]
    _answer = [1, 1, 0, 0, 1, 0, 1, 0]

    _people = {"a": 0, "b": 0}
    _yes = {"a": 0, "b": 0}
    for _i in range(len(_group)):
        _g = _group[_i]
        _people[_g] = _people[_g] + 1
        _yes[_g] = _yes[_g] + _answer[_i]
    print("people:", _people, "| yes:", _yes)

    _rate = {"a": _yes["a"] / _people["a"], "b": _yes["b"] / _people["b"]}
    print("selection rate:", _rate)
    print("difference, the largest minus the smallest:", _rate["a"] - _rate["b"])
    print("ratio, the smallest divided by the largest:", round(_rate["b"] / _rate["a"], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Group a has 3 yeses of 4 people, a selection rate of 0.75; group b has 1 of 4, 0.25. Fairlearn would give a difference of 0.5 and a ratio of 0.33. AIF360, with b as the unprivileged group, would give a statistical parity difference of 0.25 minus 0.75, that is −0.5, and a disparate impact of 0.33. Exercise 1 does the same for the example of Fairlearn's documentation.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Fairlearn's example by hand

        `sex` and `y_pred` are the ten people of the example in *The idea*. `people` and `yes` start at 0 for both sexes, and the loop goes through the people by their place `i`. Replace the `...` with two lines, as in the worked example: add 1 to the count in `people` of the sex of person `i`, and add their answer to the count in `yes`. The line after the loop works out `rate`, the selection rate per sex. The check also tries other people.
        """),
        L.task("Make `rate`: per sex the share of the people with a yes, as `MetricFrame` gives it in `by_group`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("rates", lines=11)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "rates",
        [
            "Do what the worked example did: for each person, one more in `people` for their sex, and their answer added to `yes` for their sex.",
            "The sex of person `i` is `sex[i]`, and their answer is `y_pred[i]`. So the first line is `people[sex[i]] = people[sex[i]] + 1`.",
            "`yes[sex[i]] = yes[sex[i]] + ___`.",
        ],
        why="`people[sex[i]] = people[sex[i]] + 1` and `yes[sex[i]] = yes[sex[i]] + y_pred[i]`. That gives `{'Female': 0.8, 'Male': 0.4}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How far apart the gains lie

        `plain` and `reweighed` hold the ten ratios of the demo, rounded: how often women get a yes divided by how often men do, without weights and with reweighing. `gain` starts as an empty list, and the loop goes through the samples by their place `i`. Replace the first `...` with one line that adds to `gain` by how much reweighing raised sample `i`, and the second with the spread: the largest gain minus the smallest, with `max` and `min`. The check also tries other numbers.
        """),
        L.task("Find `spread`: how far apart the largest and the smallest gain of reweighing lie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("spread", lines=9)
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
            "A gain is the ratio with reweighing minus the ratio without, for the same sample. The spread is the largest gain minus the smallest.",
            "In the loop: `gain.append(reweighed[i] - plain[i])`. After the loop, `max(gain)` is the largest gain.",
            "`spread = max(gain) - ___`.",
        ],
        why="`gain.append(reweighed[i] - plain[i])` in the loop, and `spread = max(gain) - min(gain)` after it. That gives 0.44.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_german():
    mo.md(r"""
    ### Exercise 3. A result on German Credit
    """)
    return


@app.cell(hide_code=True)
def choice_german_cell():
    choice_german = L.choice(
        "german",
        question="A paper compares how often women and men get a loan in German Credit, and shows a new method that brings the two closer. What do you ask first?",
        options={
            "model": "Which model the paper trained on it",
            "right": "How the table can tell who is a woman",
            "size": "Whether 1,000 loans are enough to train a model",
        },
        correct="right",
        why={
            "model": "The model matters, but a question comes before it: German Credit only seems to have a column for sex. Grömping showed that single men and women who are not single have the same code. See *The idea*.",
            "right": "German Credit cannot tell: single men and women who are not single have the same code. A comparison of women and men on this table compares groups that are not women and men.",
            "size": "1,000 is few, and the demo showed how much results then vary between splits. But first: the table cannot tell who is a woman, because single men and women who are not single have the same code. See *The idea*.",
        },
    )
    choice_german
    return (choice_german,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en) and [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en) built two bias mitigation methods by hand. Fairlearn and AIF360 have them ready, sometimes with another formula behind the same name. And the usual way to show that such a method works, on UCI Adult, COMPAS or German Credit, says less than it seems: the tables are old, cannot always tell who belongs to a group, or measure something other than their name says, and one split of a small table can give very different results. [Reading AI claims critically](/course/open/res-reading-claims?lang=en) asked what exactly was measured; for a fairness result, also ask on which table, with which formula and over how many splits. The next lesson, *Project: measure the bias of your own classifier*, puts it all together on a model of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_german, result_rates, result_spread):
    L.wrap_up(result_rates, result_spread, choice_german)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
