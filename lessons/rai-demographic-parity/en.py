# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-demographic-parity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-demographic-parity", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en) and [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_proxy():
    quick_proxy = L.choice(
        "quick-proxy",
        question="A model leaves out the column `sex`, but keeps `relationship`, where many people are Husband or Wife. What can happen to the yes rates of women and men?",
        options={
            "equal": "They become equal, now that sex is gone",
            "right": "They can stay far apart",
            "swap": "They swap: women get more yes",
        },
        correct="right",
        why={
            "equal": "Leaving out `sex` hardly changed the yes rates: other columns gave sex away. See *Try it* in [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en).",
            "right": "`relationship` gives away the sex of most people, so the model can still treat women and men differently without ever seeing `sex`.",
            "swap": "Nothing makes them swap: without `sex`, the model gave women and men the same rates as before, 0.07 and 0.25. See *Try it* in [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en).",
        },
    )
    quick_proxy
    return (quick_proxy,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="A model says yes to 200 people, and 150 of them really earn more than 50,000 dollars. What is its precision?",
        options={
            "wrong": "0.25, the share that is wrong",
            "right": "0.75, as 150 / 200",
            "count": "150 people",
        },
        correct="right",
        why={
            "wrong": "Precision counts the yes that are right, not the ones that are wrong: 150 / 200 = 0.75. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "right": "Of the 200 yes, 150 are right: a precision of 0.75.",
            "count": "Precision is a share, not a count: 150 of the 200, so 0.75. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en), a model without the column `sex` gave a yes to 0.07 of the women and 0.25 of the men. The first demand that comes to mind: give every group a yes equally often. That demand has a name, demographic parity, and it is the simplest criterion of group fairness, which judges a model by comparing numbers between groups. This lesson measures it, reaches it, and shows what it cannot see.

    **An everyday example.** A school has places for a trip, and lets every class send the same share of its students: one in five. The share is the same in every class. Whether a class sends its keenest students or draws names from a hat, the share cannot tell you.
    """)
    return


@app.cell(hide_code=True)
def predict_limits():
    guess_limits = L.guess(
        "limits",
        question="The model without `sex` gives a yes to 0.07 of the women and 0.25 of the men; of the women with a yes, 0.67 really earn more than 50,000 dollars. Now each group gets its own threshold, so that both get a yes 0.19 of the time. Of the women with a yes, how many do you think really earn more then?",
        options={
            "same": "About as many: 0.67",
            "right": "Fewer: about 0.43",
            "more": "More: about 0.85",
        },
        reveal="**Fewer: about 0.43.** To give more women a yes, the threshold for women goes down, and the extra women are ones the model is less sure about. The yes rates are now equal, but a yes means something different for women and for men. The demo below shows it.",
    )
    guess_limits
    return (guess_limits,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Independence.** Barocas, Hardt and Narayanan, in *Fairness and Machine Learning*, call the simplest criterion independence: the decision does not depend on the group. For a model that says yes or no, it goes by many names, "demographic parity, statistical parity, group fairness, disparate impact and others", and "the condition requires the acceptance rate to be the same in all groups." This course calls it demographic parity. <!-- termen:ok --> Cynthia Dwork and colleagues (2012) put it this way: "the demographics of those receiving positive (or negative) classifications are identical to the demographics of the population as a whole." Moritz Hardt and colleagues (2016) in one line: "membership in a protected class should have no correlation with the decision." In this course, group fairness is the whole family of criteria that compare numbers between groups; demographic parity is its first member, and the next lesson adds two more.

    **Difference or ratio.** Exactly equal rates are rare, so you measure how far apart they are. The library Fairlearn calls a yes rate a selection rate, as [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) did, and reports two numbers. The demographic parity difference is "the absolute difference between the highest and lowest selection rates": 0 means parity. The demographic parity ratio is "the ratio of the lowest and highest selection rates, so a result of 1 means there is demographic parity." Some have argued, *Fairness and Machine Learning* notes, that a ratio of at least 0.8 "relates to the 80 percent rule". Toon Calders and Sicco Verwer (2010) used the difference; Sainyam Galhotra and colleagues (2017) give an example: if 30 percent of the people under 40 get a loan and 40 percent of the people over 40, the difference is 10 percent.

    **Why people want it.** *Fairness and Machine Learning* notes that some argue it reflects an assumption of equality: "All groups have an equal claim to acceptance and resources should therefore be allocated proportionally." It is also "often the easiest one to work with mathematically and algorithmically." Alexandra Chouldechova (2017) finds it well suited to employment or admissions, "where it may be desirable or required by law or regulation to employ or admit individuals in equal proportion across racial, gender, or geographical groups." And it takes away the problem of the previous lesson: Dwork and colleagues note that "it neutralizes redundant encodings". A model may find sex through any proxy it likes; when the yes rates must be equal, that cannot give one group more yes.

    **How to reach it.** Hardt and colleagues reached it on credit scores with a threshold per group, so that the same share of every group gets a yes. The model stays the same; only its **decision threshold** changes, the probability from which it says yes, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en). For each group you choose the threshold above which a given share of its probabilities lies: a quantile, as in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).

    **Try it.** The cell below trains the model without the column `sex` of [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en). `predict_proba` gives the probability itself instead of `True` or `False`; `[:, 1]` takes the probability of a yes. Then the cell hands out a yes in three ways: one threshold of 0.5 for everyone; a threshold per group, so that each group gets a yes as often as all people together; and the threshold of 0.5 for the men, while each woman gets a yes by chance, as often as the men. For each way it prints the yes rates of women and men, their precision (of the people with a yes, the share that really earns more than 50,000 dollars), and how often the model is right. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def demo_ways():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"

    # the model without the column sex, as in the previous lesson:
    # each column of words becomes one-hot columns, then every column runs from 0 to 1
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _score = _model.predict_proba(_X[_test])[:, 1]   # for each person, the probability of a yes


    def _show(name, yes):
        # per sex: the share with a yes, and the precision: of those with a yes, the share that earns more
        picked = _tested[yes]
        rates = picked.groupby("sex").size() / _tested.groupby("sex").size()
        rich_picked = picked[picked["income"] == ">50K"]
        precision = rich_picked.groupby("sex").size() / picked.groupby("sex").size()
        right = (yes == _rich).mean()
        print(name, "yes: women", round(rates["Female"], 2), "men", round(rates["Male"], 2),
              "| precision: women", round(precision["Female"], 2), "men", round(precision["Male"], 2),
              "| right", round(right, 2))


    # 1. one threshold for everyone: a yes where the probability is above 0.5
    _one = _score > 0.5
    _show("one threshold:        ", _one)

    # 2. a threshold per group: above it lies, in each group, the same share as the yes rate in 1
    _rate = _one.mean()
    _threshold_women = np.quantile(_score[_women], 1 - _rate)
    _threshold_men = np.quantile(_score[_men], 1 - _rate)
    # each woman gets the women's threshold, each man the men's
    _threshold = _threshold_women * _women + _threshold_men * _men
    _show("a threshold per group:", _score > _threshold)
    print("the thresholds: women", round(_threshold_women, 2), "men", round(_threshold_men, 2))

    # 3. the men as in 1; each woman gets a yes by chance, as often as the men get one
    _rng = np.random.default_rng(0)
    _chance = _rng.random(len(_tested)) < _one[_men].mean()
    _show("women by chance:      ", _chance * _women + _one * _men)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Three ways to hand out a yes",
                           "rows": ["One threshold", "A threshold per group", "Women by chance"],
                           "xlabels": ["yes rate", "precision"], "key": "open: women, filled: men", "decimal": ".",
                           "alt": "Two panels with three rows each. Left, the yes rates of women and men: one threshold 0.07 and 0.25, a threshold per group 0.19 and 0.19, women by chance 0.25 and 0.25. Right, the precision of women and men: one threshold 0.67 and 0.69, a threshold per group 0.43 and 0.73, women by chance 0.11 and 0.69."})),
        mo.md(r"""
        With one threshold, women get a yes 0.07 of the time and men 0.25. With a threshold per group, both get 0.19: demographic parity. The threshold for women goes down to 0.2 and the one for men up to 0.6, and the model is right a little less often, 0.81 instead of 0.83. But a yes now means something different: of the women with a yes, 0.43 really earn more than 50,000 dollars, of the men 0.73. A threshold per group also needs the column `sex` at the moment of the decision, the very column the model left out; whether that is allowed is a question of law, as in [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en). And to keep the demo short, it chooses the thresholds on the test set; in practice you choose them on a validation set.

        The third way meets demographic parity too, 0.25 for both. But the women are drawn by chance: of those with a yes, only 0.11 earn more, as many as among all the women in the test set. *Fairness and Machine Learning* describes this case: a company hires "diligently selected applicants" in one group and "carelessly selected applicants" in the other at the same rate, and then "it will appear in hindsight that members of group b performed worse". Dwork and colleagues point out that this can be done on purpose, to "justify" discrimination later, or without malice, by picking at random; either way it builds a bad "track record" for the group.

        Demographic parity looks only at who gets a yes, not at who deserved it. That also cuts the other way. In the data, 0.11 of the women and 0.30 of the men earn more than 50,000 dollars, so a perfect model, right about everyone, does not meet it: Hardt and colleagues note that demographic parity "would not allow the ideal predictor". On 301,536 credit scores from 2003 they found that, in their setting, demographic parity earned 69.8 percent of the most a lender could earn, where one threshold for everyone earned 99.3 percent. And equal rates for women and men say nothing about smaller groups: "Statistical parity for S does not imply statistical parity for subsets of S," wrote Dwork and colleagues. A model can give women as a whole their share of yes, and almost none to older women. <!-- termen:ok -->

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **group fairness** | criteria that judge a model by comparing numbers between groups, such as how often each group gets a yes |
        | **demographic parity** | every group gets a yes equally often, whatever the labels; also called independence or statistical parity <!-- termen:ok --> |
        | **demographic parity difference** | the highest yes rate of the groups minus the lowest; 0 means parity |
        | **demographic parity ratio** | the lowest yes rate of the groups divided by the highest; 1 means parity |
        """),
        L.watch_out("Equal yes rates do not mean the yes went to the right people. Always look at who got the yes too, for example at the precision per group."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure demographic parity of a model or a rule:

    1. Count, per group, the share that gets a yes.
    2. Take the difference: the highest share minus the lowest, where 0 means parity. And take the ratio: the lowest divided by the highest, where 1 means parity.
    3. A ratio below 0.8 relates to the four-fifths rule: a signal to look closer, not a verdict.
    4. Look at who got the yes: per group, the precision. Equal rates can hide a careless choice in one group.
    5. Ask whether equal rates fit the decision. Where the true outcome differs per group, demographic parity rules out even a perfect model.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The table UCI Adult, and a perfect model: it says yes exactly to the people who earn more than 50,000 dollars, so its yes are the rows where `income` is `">50K"`. `_column` is the column `sex`. The cell works out the yes rate per group, as in [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en), then the difference and the ratio with `max()` and `min()`. Run the cell.
    """)
    return


@app.cell
def example_perfect(pd):
    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "sex"

    _yes = _people[_people["income"] == ">50K"]
    _rates = _yes.groupby(_column).size() / _people.groupby(_column).size()
    _difference = _rates.max() - _rates.min()
    _ratio = _rates.min() / _rates.max()
    print(_rates.round(2))
    print("difference", round(_difference, 2), "ratio", round(_ratio, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The perfect model gives a yes to 0.11 of the women and 0.30 of the men: a difference of 0.19 and a ratio of 0.36, far below four-fifths. So a model that is right about everyone does not meet demographic parity, because the outcome itself differs between the groups. Is that a flaw of the model, of the world the data comes from, or of the criterion? The next lessons come back to that question. First, a rule of your own: Exercise 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parity():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A rule by schooling

        A made-up hiring rule gives a yes to everyone with at least a bachelor's degree: an `education_num` of 13 or more. `column` is the column `race`, with the five groups as the survey of 1994 named them. Replace the `...` with the four lines of the worked example, with the names without their underscore: `yes`, `rates`, `difference` and `ratio`. Put in `yes` the rows of the rule instead of the people who earn more. The last line is `difference, ratio`, so the check sees both. The check also tries another column.
        """),
        L.task("Work out `difference` and `ratio` of the yes rates per value of `race`, for the rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_parity():
    box_parity = L.code_box("parity", lines=12)
    box_parity
    return (box_parity,)


@app.cell(hide_code=True)
def check_parity(box_parity):
    result_parity = L.check("parity", box_parity.value)
    result_parity
    return (result_parity,)


@app.cell(hide_code=True)
def hints_parity():
    L.hints(
        "parity",
        [
            "Use the four lines of the worked example. Only the rows you pick change: not the people who earn more than 50,000 dollars, but the people with an `education_num` of 13 or more.",
            "The rule is `people[people[\"education_num\"] >= 13]`. The difference is `rates.max() - rates.min()`.",
            "`ratio = ___ / rates.max()`.",
        ],
        why="`yes = people[people[\"education_num\"] >= 13]`, `rates = yes.groupby(column).size() / people.groupby(column).size()`, `difference = rates.max() - rates.min()` and `ratio = rates.min() / rates.max()`. That gives a difference of 0.33 and a ratio of 0.23.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_care():
    mo.md(r"""
    ### Exercise 2. What demographic parity sees

    Six ways to hand out a yes. Sort each by what demographic parity can see in it.
    """)
    return


@app.cell(hide_code=True)
def sort_care_cell():
    sort_care = L.sort("care")
    sort_care
    return (sort_care,)


@app.cell(hide_code=True)
def check_care(sort_care):
    result_care = L.check("care", sort_care)
    result_care
    return (result_care,)


@app.cell(hide_code=True)
def hints_care():
    L.hints(
        "care",
        [
            "Ask two things of each way: does each group get a yes equally often, and are the people with a yes picked the same way in each group?",
            "Picking by chance or by who came first is not the same care as picking by savings or after interviews. A perfect model gives each group the share of its labels.",
            "Careless in one group: savings and chance, and \\_\\_\\_. Unequal: the perfect model and \\_\\_\\_.",
        ],
        why="**The same care:** savings, and one in five. **Careless in one group:** savings and chance, and first come. **Unequal:** the perfect model, and the model without the column.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fit():
    mo.md(r"""
    ### Exercise 3. Where it fits
    """)
    return


@app.cell(hide_code=True)
def choice_fit_cell():
    choice_fit = L.choice(
        "fit",
        question="For which decision is *equal yes rates per group* the most fitting goal?",
        options={
            "crime": "Predicting who commits a new crime",
            "right": "Inviting applicants to a first interview",
            "disease": "Predicting a disease that one group gets more often",
        },
        correct="right",
        why={
            "crime": "Chouldechova calls demographic parity \"a difficult criterion to motivate in the recidivism prediction setting\": there, the debate is whether the errors or the meaning of a risk score should be equal. The next lesson is about that.",
            "right": "Chouldechova calls the criterion \"well-suited to contexts such as employment or admissions\", where it may be desirable or even required by law to take on people from every group in equal proportion. A first interview is such a decision.",
            "disease": "If the disease really is more common in one group, equal yes rates rule out even a perfect test, as with the perfect model of the worked example.",
        },
    )
    choice_fit
    return (choice_fit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [The four-fifths rule](/course/open/rai-80-percent-rule?lang=en) you divided yes rates as a signal of disparate impact; here equal yes rates became a criterion of their own, demographic parity, with a difference and a ratio to measure it. In [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en) proxies kept the rates apart; a threshold per group closes the gap, whatever the proxies say. But demographic parity only counts the yes: your sort showed that it cannot tell care from chance. The next lesson, *Equal errors or equal meaning*, looks at who deserved the yes: equal error rates per group, and a score that means the same in every group.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fit, result_care, result_parity):
    L.wrap_up(result_parity, result_care, choice_fit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
