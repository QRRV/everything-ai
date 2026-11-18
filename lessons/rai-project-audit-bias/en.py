# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-project-audit-bias. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-project-audit-bias", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_metricframe():
    quick_metricframe = L.choice(
        "quick-metricframe",
        question="Fairlearn's `MetricFrame` with `selection_rate` gives in `by_group`: Female 0.8, Male 0.4. What is the 0.8?",
        options={
            "share": "Of all yeses, the share for women",
            "right": "The share of the women who get a yes",
            "accuracy": "The share of the women the model gets right",
        },
        correct="right",
        why={
            "share": "That would divide by all yeses. `selection_rate` divides by the people of the group: of the women, 0.8 get a yes. See Exercise 1 in [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
            "right": "`selection_rate` is the yes rate, worked out per group: of the women, 0.8 get a yes. The table you make in this lesson starts with it.",
            "accuracy": "How often the model is right is the accuracy, another measure. `selection_rate` counts only the yeses: of the women, 0.8 get a yes. See *The idea* in [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
        },
    )
    quick_metricframe
    return (quick_metricframe,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="On the whole table UCI Adult, reweighing raised the women's yes rate divided by the men's from 0.28 to 0.57. What did ten samples of 1,000 people show?",
        options={
            "same": "About the same gain of 0.29 in every sample",
            "right": "The gain varied a lot between samples",
            "none": "Reweighing lowered the ratio in most of them",
        },
        correct="right",
        why={
            "same": "The gain went from about 0.07 to about 0.51: with 300 people to test, one woman more or less with a yes moves the ratio by about 0.05. See the demo in [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
            "right": "From about 0.07 to about 0.51. One split of a small table says little, and your audit also measures on one split: write down how many people each number rests on.",
            "none": "Reweighing raised the ratio in all ten samples; by how much varied, from about 0.07 to about 0.51. See the demo in [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lessons you measured one thing at a time: a yes rate, an error rate, a calibration, a threshold per group. An audit puts them together for one model, in a fixed order. In this project you audit your own classifier, the one of the previous lessons on UCI Adult, as someone in the team that built it would: first what you will measure and for which groups, then the numbers per group, then a first fix, measured again, and last what the numbers can and cannot tell. The functions you write stay in your project. Later in the course, *Project: explain your classifier* explains the same classifier and uses them again.

    **An everyday example.** A food inspector visits a restaurant with a fixed list: the temperature of the fridge, whether raw and cooked food are kept apart, whether the staff wash their hands. The list is the same for every restaurant and is fixed before the visit, so the inspector does not choose what to look at after seeing the kitchen. At the end comes a report: what passed, what did not, and what must change before the next visit. And a clean report means that the kitchen passed that list on that day, not that no one will ever get sick there.
    """)
    return


@app.cell(hide_code=True)
def predict_white():
    guess_white = L.guess(
        "white",
        question="The classifier of the previous lessons gives a yes to about 20 in every 100 White people in the test set, and to about 7 in every 100 Black people. Does the four-fifths rule flag White people too, do you think?",
        options={
            "no": "No: White people get far more yeses than Black people",
            "right": "Yes, because one group gets a yes even more often",
            "small": "Only if they are fewer than half",
        },
        reveal="**Yes.** The people marked `Asian-Pac-Islander` get a yes about 26 times in 100. More exactly, 20.26 against 25.83, and 20.26 divided by 25.83 is 0.78, just below 0.8. The rule compares every group with the group that gets a yes most often, whichever group that is: here a group of only 480 people. The demo below shows all five groups.",
    )
    guess_white
    return (guess_white,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **What an audit is.** The word is used in two ways. From the outside, an audit is, in the words of Danaë Metaxa and colleagues, "a method of repeatedly and systematically querying an algorithm with inputs and observing the corresponding outputs in order to draw inferences about its opaque inner workings": you cannot look inside, so you send in many inputs and compare what comes out. From the inside, Inioluwa Deborah Raji and colleagues take a definition from software audits: "an independent evaluation of conformance of software products and processes to applicable regulations, standards, guidelines, plans, specifications, and procedures". In this course, an **algorithmic audit** is a check of an algorithmic system against criteria such as fairness, accuracy or lawfulness, by looking at its data, its code or what it does.

    **Who audits.** Sasha Costanza-Chock and colleagues tell three kinds apart: a first-party audit is done by the team itself, a second-party audit by a contractor, and a third-party audit by someone without a contract with the company. You built the classifier and you have its data, so yours is a first-party audit. You can see what an outsider cannot, such as the data to train on and the code. But, Raji and colleagues warn, "Internal auditors necessarily share an organizational interest with the target of the audit."

    **What goes in it.** Raji and colleagues split an audit inside a company into five stages: scoping, mapping, collecting artifacts such as datasheets, testing, and reflection. Each stage yields documents that together form the audit report. Scoping comes first: what is the system for, and whom can it harm? The numbers come later, and they "should not become aims and ends in themselves". For the numbers, Margaret Mitchell and colleagues, who proposed model cards to document models, advise that results be "disaggregated, that is, broken down by the chosen factors", such as sex and race. A report should list all error rates and say which one was given priority and why, and name the decision thresholds and why they were chosen.

    **What it can and cannot show.** An audit measures outcomes per group: who gets a yes, and who bears which mistakes. Metaxa and colleagues write that its results "can only be discerned in aggregate", for the system as a whole; why the outcomes differ, it does not show by itself. An audit can find a problem, but it cannot prove that there is none: a clean result holds for the groups, the criteria, the data and the thresholds that were measured. And how close to equal is close enough, the audit has to decide. Raji and colleagues leave that question open: "how close to parity is necessary to say the classifier is acceptable?" For yes rates, [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) gave one answer, from US hiring rules. The field has no agreed standard yet: Costanza-Chock and colleagues asked 43 auditors for a link to how they audit, and only seven gave a link to any documentation, and four to results.

    **Your classifier.** When you opened this lesson, the course put a file in your project: `work/projects/audit/model.py`. It holds the classifier of the previous lessons: logistic regression on age, schooling, working hours, kind of work, marital status, occupation, relationship and race, without sex. Its function `scores(people)` trains it on the people to train on and gives every person the probability of a yes. Reading what goes in is part of the audit: the classifier does not get sex, but it gets `relationship`, with the values `Husband` and `Wife`, a proxy for sex, and it gets race itself. It is your file. Change its lists `NUMBERS` and `WORDS` if you like: at the end of the lesson, your project audits the classifier that is in that file.

    **Try it.** The cell below trains the same classifier, gives a yes above 0.5, and makes the first table of an audit: per race in the test set, how many people, how often a yes, and that divided by the highest yes rate, the impact ratio of [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en). The five values of `race` are the table's own words from 1994. Some, such as `Amer-Indian-Eskimo`, are no longer used today; the lesson keeps them so that you can find them in the data. Run the cell; it takes a second or two.
    """)
    return


@app.cell
def demo_race():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"

    # the classifier of the previous lessons, without the column sex, as in work/projects/audit/model.py
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _people["yes"] = _model.predict_proba(_X)[:, 1] > 0.5

    # the first table of the audit: per race in the test set
    _test = _people[_people["split"] == "test"]
    _table = pd.DataFrame()
    _table["people"] = _test.groupby("race").size()
    _table["yes rate"] = _test.groupby("race")["yes"].mean()
    _table["impact ratio"] = _table["yes rate"] / _table["yes rate"].max()
    print(_table.round(3))
    return (pd,)


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_race({"title": "Yes rate per race, on the test set",
                                "xlabel": "yes rate: the share with a yes", "line": "0.8 × highest", "decimal": ".",
                                "key": "A bar per race, with its people in brackets and its impact ratio at the end of the bar. A bar that ends before the dashed line is below the four-fifths rule.",
                                "alt": "Yes rates per race: Asian-Pac-Islander (480 people) 0.26, impact ratio 1; White (13,946) 0.20, 0.78; Black (1,561) 0.07, 0.27; Other (135) 0.06, 0.23; Amer-Indian-Eskimo (159) 0.04, 0.17. The dashed line at 0.8 times the highest lies at 0.21."})),
        mo.md(r"""
        Four of the five groups get a yes less than 80 percent as often as the group that gets one most often, `Asian-Pac-Islander`, and among them is `White`, at 0.78. `Black` gets 0.27 of that rate, and `Amer-Indian-Eskimo` 0.17. Two things belong in the report. First, one group sets the bar, here a group of 480 people: a few more or fewer yeses in that group move every ratio. Second, the groups hold from 135 to 13,946 people. A yes rate over 135 people can easily differ by a few hundredths with other people, as [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en) showed for samples of 1,000. Model cards ask for a margin of uncertainty for that reason; at the least, give how many people each number rests on.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **algorithmic audit** | a check of an algorithmic system against criteria such as fairness, accuracy or lawfulness, by looking at its data, its code or what it does |
        | **first-, second- and third-party audit** | an audit by the team that built the system, by a contractor it hired, or by someone without a contract with it |
        | **disaggregated** | broken down per group, such as per sex or per race, instead of one number for everyone |
        """),
        L.watch_out("Your audit is a first-party audit: you check your own work, and you want it to pass. Decide what you will measure, and which numbers count as a problem, before you look at the results, and report every number, also the ones that look bad."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The first part of your audit:

    1. Scope: what does the classifier decide, and whom can a wrong yes or a wrong no harm? Write down the groups and the criteria before you measure: here sex and race; the yes rate with the four-fifths rule, the recall, the false positive rate, and calibration within groups.
    2. Read what goes in: which columns, and which of them are a protected attribute or a proxy for one.
    3. Measure on the test set, per group: how many people, the yes rate, the impact ratio, the recall and the false positive rate. That is `report`.
    4. Per band of scores, compare per group the mean score with the share that earns more. That is `calibration`.
    5. Try a first fix, such as a threshold per group with the same recall, chosen on the people to train on. That is `threshold_for`. Then measure everything again, for every group, not only the one you fixed.
    6. Write down what the numbers show, how many people they rest on, and what the audit did not measure.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Eight made-up people in two groups, a and b, with for each whether they earn more, `rich`, 1 or 0, and the classifier's answer, `yes`. `_rich` holds the rows of the people who earn more, and `_poor` the rows of the others. `_table` starts empty. Its first column is the recall per group: of the people who earn more, the share with a yes, which `groupby` and `.mean()` of `yes` give at once. Its second column is the false positive rate: the same for `_poor`.
    """)
    return


@app.cell
def example_errors(pd):
    # eight made-up people: their group, rich (1 when they earn more) and the classifier's answer
    _people = pd.DataFrame({"group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                            "rich": [1, 1, 0, 0, 1, 1, 1, 0],
                            "yes": [True, False, True, False, True, True, False, False]})
    _rich = _people[_people["rich"] == 1]
    _poor = _people[_people["rich"] == 0]

    _table = pd.DataFrame()
    _table["recall"] = _rich.groupby("group")["yes"].mean()
    _table["false positive rate"] = _poor.groupby("group")["yes"].mean()
    print(_table)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In group a, one of the two people who earn more gets a yes, a recall of 0.5, and one of the two others gets one too, a false positive rate of 0.5. In group b, two of the three who earn more get a yes, 0.67, and the one person who does not gets a no, 0. `groupby` gives every group its own row, sorted by name, and every new column of `_table` lines up by group. Exercise 1 makes this a function in your project, together with the yes rate and the impact ratio of the demo.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. report: the table of your audit

        `report(people, column)` gets a table like the test set of the demo and the name of the column to group by. The table has a column `rich`, 1 for the people who earn more and 0 for the others, and a column `yes` with the classifier's answer, True or False. The function already has `rich`, `poor` and `table`, whose first column `people` counts the people per group. Replace the `...` with four lines, each a column of `table`: `yes rate`, the share of the group with a yes; `impact ratio`, the yes rate divided by the highest yes rate, as in the demo; and `recall` and `false positive rate`, as in the worked example. Write `column` wherever the name of the column goes. The check calls `report(PEOPLE, "sex")`, with `PEOPLE` the test set with the classifier of the demo, then `report(PEOPLE, "race")` and a small table of its own. When it is right, the check saves it in your project, in `work/projects/audit/report.py`.
        """),
        L.task("Add the columns `yes rate`, `impact ratio`, `recall` and `false positive rate` to `table`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "The demo made `yes rate` and `impact ratio` for the column `race`, and the worked example made `recall` and `false positive rate` for `group`. Here every `groupby` gets `column`.",
            "`table[\"yes rate\"] = people.groupby(column)[\"yes\"].mean()`. For the recall, group `rich` instead of `people`, and for the false positive rate, `poor`.",
            "`table[\"impact ratio\"] = table[\"yes rate\"] / ___`.",
        ],
        why="`table[\"yes rate\"] = people.groupby(column)[\"yes\"].mean()`, `table[\"impact ratio\"] = table[\"yes rate\"] / table[\"yes rate\"].max()`, `table[\"recall\"] = rich.groupby(column)[\"yes\"].mean()` and `table[\"false positive rate\"] = poor.groupby(column)[\"yes\"].mean()`. For sex, the women get a yes rate of 0.07 and an impact ratio of 0.28.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_calibration():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. calibration: does a score mean the same for every group?

        A score is calibrated within groups when, in every group, it comes true about as often as it says, as in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en). `calibration(people, column, low, high)` looks at one band of scores: the people with a score of `low` or more and below `high`. `band` already holds the rows with a score of `low` or more. Replace the first `...` with a line that keeps the rows of `band` whose score is below `high`. Replace the second with two columns of `table`: `mean score`, the mean of `score` per group in the band, and `share rich`, the mean of `rich` per group in the band, which is the share that earns more. The check calls `calibration(PEOPLE, "sex", 0.5, 0.6)`, then another column and band, and a small table with scores right on the edges of a band. It saves your function in `work/projects/audit/calibration.py`.
        """),
        L.task("Keep the people with a score below `high`, and add `mean score` and `share rich` per group."),
    ])
    return


@app.cell(hide_code=True)
def exercise_calibration():
    box_calibration = L.code_box("calibration", lines=11)
    box_calibration
    return (box_calibration,)


@app.cell(hide_code=True)
def check_calibration(box_calibration):
    result_calibration = L.check("calibration", box_calibration.value)
    result_calibration
    return (result_calibration,)


@app.cell(hide_code=True)
def hints_calibration():
    L.hints(
        "calibration",
        [
            "Step 4 of the recipe. A band has two edges: `band` already has the lower one, and the next line takes the rows of `band` with a score below `high`, in the form `band = band[...]`.",
            "Keep the rows with `band[\"score\"] < high`. Then, as `report` did for `people`, `band.groupby(column)[...].mean()` gives the mean of one column per group: of the scores for `\"mean score\"`, and of `rich` for `\"share rich\"`.",
            "`table[\"share rich\"] = band.groupby(column)[___].mean()`.",
        ],
        why="`band = band[band[\"score\"] < high]`, `table[\"mean score\"] = band.groupby(column)[\"score\"].mean()` and `table[\"share rich\"] = band.groupby(column)[\"rich\"].mean()`. In the band from 0.5 to 0.6, the women's mean score is 0.55 and 0.48 of them earn more; for the men, 0.55 and 0.54.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. threshold_for: a first fix

        Your project will look for a threshold per sex with the same recall, as in [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en). `threshold_for(people, recall)` gets the people of one group and a recall to reach. It should give the highest of the thresholds 0.01, 0.02, up to 0.99 at which at least the share `recall` of the people who earn more get a yes, that is, have a score above the threshold. `rich` holds those people, `best` starts at 0, and the loop goes through `t` from 1 to 99, so that the threshold is `t / 100`. Replace the `...` with an `if` that asks whether the share of `rich` with a score above `t / 100` is at least `recall`, and under it a line that keeps `t / 100` in `best`. A higher threshold that also passes replaces it later. The check calls `threshold_for(WOMEN, 0.55)`, with `WOMEN` the women to train on, then the men, and a small table where the recall is reached exactly. It saves your function in `work/projects/audit/fix.py`.
        """),
        L.task("Find the highest threshold that keeps the recall of the group at `recall` or more."),
    ])
    return


@app.cell(hide_code=True)
def exercise_threshold():
    box_threshold = L.code_box("threshold", lines=8)
    box_threshold
    return (box_threshold,)


@app.cell(hide_code=True)
def check_threshold(box_threshold):
    result_threshold = L.check("threshold", box_threshold.value)
    result_threshold
    return (result_threshold,)


@app.cell(hide_code=True)
def hints_threshold():
    L.hints(
        "threshold",
        [
            "The recall at a threshold is the share of the people who earn more with a score above it. A Series of True and False has as its `.mean()` the share of True.",
            "`(rich[\"score\"] > t / 100).mean()` is the recall at the threshold `t / 100`. Compare it with `recall` in an `if`, and keep the threshold in `best`.",
            "`if (rich[\"score\"] > t / 100).mean() ___ recall:`",
        ],
        why="`if (rich[\"score\"] > t / 100).mean() >= recall:` and under it `best = t / 100`. The recall never rises when the threshold rises, so the last threshold that passes is the highest: 0.38 for the women and 0.52 for the men.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell runs your audit on your own classifier: `scores` from your `model.py` on the whole table, then your `report` per sex and per race on the test set, your `calibration` per sex in ten bands of 0.1, and your `threshold_for` for the women and the men to train on, at a recall of 0.55. With those thresholds, it decides again about the test set and calls your `report` once more. It takes a few seconds, and it runs again after each exercise.
    """)
    return


@app.cell(hide_code=True)
def demo_project(result_calibration, result_report, result_threshold):
    _r = (result_report, result_calibration, result_threshold)   # so that this cell runs after every check
    _audit = C.demo()
    if _audit is None:
        _out = mo.md("*Finish exercise 1 first: then your `report` is in your project, and this cell uses it.*")
    elif isinstance(_audit, str):
        _out = mo.md(C.broken(_audit, L.lang))
    else:
        _parts = C.audit_parts(_audit, L.lang)
        _out = mo.vstack([mo.md(_p) if _kind == "md" else mo.Html(_p) for _kind, _p in _parts])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 4. One sentence in the report
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="A colleague writes in your audit report: \"The audit shows that the classifier with a threshold per sex is fair.\" What is wrong with that sentence?",
        options={
            "nothing": "Nothing: women and men now have the same recall",
            "right": "An audit can find a problem, not prove there is none",
            "third": "Only an independent auditor may call a classifier fair",
            "accuracy": "It should say how often the classifier is right",
        },
        correct="right",
        why={
            "nothing": "Equal recall is one criterion, on one test set, at one pair of thresholds. The women's impact ratio is still well below 0.8, and the groups by race barely changed. And what the audit did not measure, it cannot vouch for.",
            "right": "The audit measured these groups, with these criteria, on this test set and with these thresholds. It found differences where it looked; a clean result would say nothing about where it did not look. So the report says what was measured and what was not.",
            "third": "Who audits matters: yours is a first-party audit, and you share the builder's interest. But a third-party auditor cannot prove that a classifier is fair either. The trouble is in the claim, not in who makes it.",
            "accuracy": "The report should say how often the classifier is right, per group too. But that does not fix the sentence: being right often says nothing about who bears the mistakes, and no number proves that there is no problem.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) gave you the impact ratio, [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en) the error rates and calibration within groups, and [Fixing the decision: a threshold per group](/course/open/rai-fair-thresholds?lang=en) the first fix. Here they became one audit of one classifier, in the order an audit needs: decide what you will measure, measure per group, fix, measure again, and write down what the numbers do not show. [Fairness tools and benchmarks](/course/open/rai-fairness-benchmarks?lang=en) showed why the number of people behind each figure belongs in the report. Your project `audit` now has `model.py`, `report.py`, `calibration.py` and `fix.py`. It comes back in *Project: explain your classifier*, where you explain the same classifier and write down what the explanations show about the bias you measured here. The next lesson, *Bias in language models*, looks for bias where there is no yes or no to count: in what a language model writes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, result_calibration, result_report, result_threshold):
    L.wrap_up(result_report, result_calibration, result_threshold, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
