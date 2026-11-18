# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-reweighing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-reweighing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Equal yes rates](/course/open/rai-demographic-parity?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_ratio():
    quick_ratio = L.choice(
        "quick-ratio",
        question="A model gives a yes to 0.07 of the women and 0.25 of the men. What is its demographic parity ratio?",
        options={
            "difference": "0.18, their difference",
            "right": "0.28, as 0.07 / 0.25",
            "upside": "3.57, as 0.25 / 0.07",
        },
        correct="right",
        why={
            "difference": "0.18 is the demographic parity difference. The ratio divides the lowest yes rate by the highest: 0.07 / 0.25 = 0.28. See *The idea* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
            "right": "The lowest yes rate divided by the highest: 0.07 / 0.25 = 0.28. A ratio of 1 would mean demographic parity.",
            "upside": "The ratio divides the lowest yes rate by the highest, so it is never above 1: 0.07 / 0.25 = 0.28. See *The idea* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
        },
    )
    quick_ratio
    return (quick_ratio,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="A threshold per group gave women and men a yes equally often. What does that need to know when it decides about a person?",
        options={
            "right": "That person's sex",
            "nothing": "Nothing beyond the model's own columns",
            "label": "Whether that person earns more",
        },
        correct="right",
        why={
            "right": "Women and men each have their own threshold, so the decision needs the person's sex to pick one.",
            "nothing": "The model leaves out the column `sex`, but the threshold does not: women and men each have their own. See *The idea* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
            "label": "Whether someone earns more is what the model guesses; at the moment of the decision nobody knows it. What the threshold needs is the person's sex. See *The idea* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Equal yes rates](/course/open/rai-demographic-parity?lang=en), the model without the column `sex` gave a yes to 0.07 of the women and 0.25 of the men, and a threshold per group brought both to 0.19. That fixed the decision after training, and it needed the sex of every person the model decides about. There is another place to start: the data the model learns from. Of the women in the data to train on, 0.11 earn more than 50,000 dollars, of the men 0.31, and the model learns that pattern. This lesson changes how much each person counts while the model learns, so that the pattern is gone from the data, and measures how much of it is left in the model.

    **An everyday example.** A poll asks 1,000 people how they will vote. Young people answer less often: they give 10 percent of the answers but are 20 percent of the voters. So the pollster counts each young person's answer twice, and every other answer a little less, until each age group weighs as much as it does among the voters. Nobody's answer changes, only how much it counts.
    """)
    return


@app.cell(hide_code=True)
def predict_ratio():
    guess_ratio = L.guess(
        "ratio",
        question="Without weights, the model gives women a yes 0.28 times as often as men. Trained on data where each person counts so that women and men have the same share that earns more, how often do you think?",
        options={
            "same": "As before: about 0.28 times",
            "right": "Closer: about 0.57 times",
            "equal": "Equally often: 1 time",
        },
        reveal="**Closer: about 0.57 times.** The weights take the difference out of the labels the model learns from, not out of its decisions. Women and men still differ in other columns that the model uses, so the yes rates come closer, not equal. The demo below shows it.",
    )
    guess_ratio
    return (guess_ratio,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Three places to fix.** Methods that make a model treat groups more alike are called **bias mitigation**, and Barocas, Hardt and Narayanan, in *Fairness and Machine Learning*, sort them by where they act. Pre-processing changes the data: "Adjust the feature space to be uncorrelated with the sensitive attribute." In-training, which the library AIF360 calls in-processing, works the fairness constraint "into the optimization process that constructs a classifier from training data." Post-processing changes a trained model's decisions: "Adjust a learned classifier so as to be uncorrelated with the sensitive attribute." The threshold per group of [Equal yes rates](/course/open/rai-demographic-parity?lang=en) was post-processing. Each place has its price. In-training "can lead to the highest utility", but "we need access to the raw data and training pipeline"; post-processing "works for any black-box classifier regardless of its inner workings. There's no need for re-training". The team behind AIF360 puts the choice as what you are allowed to change: "If the algorithm is allowed to modify the training data, then pre-processing can be used."

    **Weights instead of new labels.** Faisal Kamiran and Toon Calders (2012) compared ways to clean a table before training, for one sensitive column with two groups and a label yes or no. Leaving the column out did not take the difference away, as in [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en). *Massaging* changes the label of some people whose probability is close to the threshold: a few no become yes in one group, and a few yes become no in the other. **Reweighing** changes no label. It gives each person a weight: how many people that person counts for while the model learns, not a weight inside the model. The weight depends only on the person's group and label. Take the number of people you would expect with that group and label if the label did not depend on the group, and divide it by the number there are. Where a combination is rarer than expected, its people count more; where it is more common, they count less. In the weighted data, every group then has the same share of each label: the independence of [Equal yes rates](/course/open/rai-demographic-parity?lang=en), now in the labels instead of the decisions.

    **On UCI Adult.** Of the 32,561 people to train on, 10,771 are women and 21,790 men, and 7,841 earn more than 50,000 dollars: 0.24 of everyone. If sex told you nothing about income, 0.24 of the women would earn more: 10,771 × 7,841 / 32,561 = 2,594 women. There are 1,179. So each of them counts as 2,594 / 1,179 = 2.2 people. The same sum gives 0.85 for the women who earn less, 0.79 for the men who earn more and 1.09 for the men who earn less. The weights keep the size of each group: the women still count as about 10,771 people, but now 0.24 of them earn more, as of the men. A model in scikit-learn takes the weights in `fit`, as `sample_weight`: "Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight."

    **What it needs, and what it brings.** The weights need the sex of every person to train on, but not of the people the model decides about: the model itself can still leave the column out. In the tests of Kamiran and Calders, reweighing kept the accuracy high, though it did less well than massaging. The team behind AIF360 reports for reweighing "good improvement in fairness, but without much penalty in accuracy", on UCI Adult with race as the group.

    **Try it.** The cell below works out the four weights on the people to train on, and trains the model of [Equal yes rates](/course/open/rai-demographic-parity?lang=en) twice, without the column `sex`: once without weights, once with them. `_cell` marks the people with one sex and one label, and `_weight` collects for each person the weight of their cell. For each model the cell prints how often it is right on the test set, and per sex the yes rate and the recall: of the people who really earn more, the share with a yes. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def demo_reweighing():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"

    # the model without the column sex, as in the previous lessons
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _y = _people["income"] == ">50K"

    # 1. the weights, on the people to train on: per sex and label, expected count divided by real count
    _learn = _people[_train]
    _weight = 0
    for _sex in ["Female", "Male"]:
        for _label in [">50K", "<=50K"]:
            _cell = (_learn["sex"] == _sex) * (_learn["income"] == _label)
            _in_group = (_learn["sex"] == _sex).sum()
            _with_label = (_learn["income"] == _label).sum()
            _expected = _in_group * _with_label / len(_learn)
            _w = _expected / _cell.sum()
            _weight = _weight + _w * _cell
            print("weight", _sex, _label, round(_w, 2))

    # 2. per sex, on the test set: who has it, and who really earns more
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"


    def _show(name, weight):
        # train with or without weights, and measure on the test set
        model = LogisticRegression(max_iter=1000)
        model.fit(_X[_train], _y[_train], sample_weight=weight)
        yes = model.predict_proba(_X[_test])[:, 1] > 0.5
        print(name, "| right", round((yes == _rich).mean(), 3))
        for sex, group in {"women": _women, "men": _men}.items():
            print("  ", sex, "| yes rate", round(yes[group].mean(), 2),
                  "| recall", round(yes[group * _rich].mean(), 2))
        print("   women get a yes", round(yes[_women].mean() / yes[_men].mean(), 2), "times as often as men")


    _show("no weights", None)
    _show("reweighing", _weight)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_effect({"title": "The model without the column sex, without and with weights",
                                  "rows": ["No weights, accuracy 0.834", "Reweighing, accuracy 0.829"],
                                  "xlabels": ["yes rate", "recall"],
                                  "key": "open: women, filled: men", "decimal": ".",
                                  "alt": "Two panels with two rows each. Left, the yes rate of women and men: without weights 0.07 and 0.25, with reweighing 0.12 and 0.20. Right, the recall: without weights 0.44 and 0.57, with reweighing 0.58 and 0.50. The model is right 0.834 of the time without weights and 0.829 with them."})),
        mo.md(r"""
        Without weights, women got a yes 0.28 times as often as men; with the weights 0.57 times, 0.116 against 0.204. The model was right a little less often: 0.829 instead of 0.834 of the time. The weights made sex and label independent in the data the model learns from, not in its decisions: women and men still differ in other columns that the model does use, such as `relationship` and `occupation`. Look at the recall too. Of the women who earn more, 0.58 now get a yes, of such men 0.50: the women went from below the men to above them. Reweighing aims at equal yes rates, not at equal errors; whether equal errors matter more is the question of [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **bias mitigation** | methods that make a model treat groups more alike: before, during or after training |
        | **pre-processing** | bias mitigation that changes the data before training, such as reweighing |
        | **in-processing** | bias mitigation that works a fairness constraint into the training itself; also called in-training |
        | **post-processing** | bias mitigation that changes the decisions of a trained model, such as a threshold per group |
        | **reweighing** | a weight per person by group and label, the expected count divided by the real count, so that every group has the same share of each label in the weighted data |
        | **weight**, `sample_weight` | how many people one person counts for while a model learns, not a weight inside the model; scikit-learn takes it in `fit(X, y, sample_weight=...)` |
        """),
        L.watch_out("Fairlearn says it of its own mitigation methods: they can drive a measure of unfairness close to zero, but \"this does not mean that the results are fair.\" Reweighing only changes how much each label counts. If a label measures the wrong thing, as in [Who is in the data, and who is missing?](/course/open/rai-bias-in-data?lang=en), the model still learns it, only with other weights."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fix the data with reweighing:

    1. Count, in the data to train on, how many people there are per group and label, per group, per label and in total.
    2. Per group and label, work out the number you would expect: the size of the group times the size of the label, divided by the total. The weight is that number divided by the number there are.
    3. Give every person the weight of their group and label, and pass the weights to `fit` as `sample_weight`. The group is needed here, not when the model decides.
    4. Measure again on the test set: per group the yes rate, the errors and how often the model is right. The weights change what the model learns, not what it does with what it learned.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A made-up table of 10 people that gives the same four weights as the example of Kamiran and Calders: 5 women, of whom 2 have the label yes, and 5 men, of whom 4 have it. 6 of the 10 have a yes. The cell works out per sex and label the number you would expect, the size of the sex times the size of the label divided by everyone, and the weight: that number divided by the number there are.
    """)
    return


@app.cell
def example_toy():
    # a made-up table of 10 people: per sex, how many have the label yes and how many no
    _counts = {"women": {"yes": 2, "no": 3}, "men": {"yes": 4, "no": 1}}
    _in_group = {"women": 5, "men": 5}
    _with_label = {"yes": 6, "no": 4}
    _total = 10
    for _group in _counts:
        for _label in _counts[_group]:
            _expected = _in_group[_group] * _with_label[_label] / _total
            print(_group, _label, "| expected", _expected, "| there are", _counts[_group][_label],
                  "| weight", round(_expected / _counts[_group][_label], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The women with a yes count 1.5 times: you would expect 3 of them, and there are 2. With the weights, the women have 2 × 1.5 = 3 yes out of 3 + 3 × 0.67 = 5, and the men 4 × 0.75 = 3 out of 3 + 1 × 2 = 5: a share of 0.6 in both, as among all 10 people. Exercise 1 does the same on UCI Adult.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The four weights of UCI Adult

        `counts` holds per sex how many of the people to train on earn more than 50,000 dollars (`">50K"`) and how many not (`"<=50K"`). `in_group` holds the size of each sex, `with_label` the size of each label, and `total` everyone. The outer loop goes through the sexes with `group` and makes an empty dict `weight[group]`; the inner loop goes through the labels with `label`. Replace the `...` with two lines, as in the worked example: `expected`, the number you would expect, and `weight[group][label]`, `expected` divided by the number there are. The check also tries a made-up table.
        """),
        L.task("Work out `weight[group][label]` for each sex and label in `counts`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("weights", lines=13)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("weights", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "weights",
        [
            "Do what the worked example did, with the names of the box: the number you would expect is the size of the sex times the size of the label, divided by everyone.",
            "`expected = in_group[group] * with_label[label] / total`. The weight divides it by the number there are, `counts[group][label]`.",
            "`weight[group][label] = expected / ___`.",
        ],
        why="`expected = in_group[group] * with_label[label] / total` and `weight[group][label] = expected / counts[group][label]`. That gives 2.20 and 0.85 for the women, 0.79 and 1.09 for the men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The share that earns more, with the weights

        `counts` is the table of Exercise 1, and `weight` holds its weights, rounded. The loop goes through the sexes with `group`; its last line divides `above` by `everyone` into `share[group]`. Replace the `...` with two lines: `above`, the number of people who earn more times their weight, and `everyone`, `above` plus the number who earn less times their weight. The check also tries other weights.
        """),
        L.task("Work out `share[group]`, the share that earns more once the weights count, for each sex."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=11)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "A person with a weight of 2.2 counts as 2.2 people. So multiply each count by the weight of its sex and label, as the worked example did with 2 × 1.5.",
            "`above = counts[group][\">50K\"] * weight[group][\">50K\"]`. `everyone` adds the people who earn less in the same way.",
            "`everyone = above + counts[group][\"<=50K\"] * ___`.",
        ],
        why="`above = counts[group][\">50K\"] * weight[group][\">50K\"]` and `everyone = above + counts[group][\"<=50K\"] * weight[group][\"<=50K\"]`. That gives 0.24 for the women and 0.24 for the men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_needs():
    mo.md(r"""
    ### Exercise 3. What reweighing needs
    """)
    return


@app.cell(hide_code=True)
def choice_needs_cell():
    choice_needs = L.choice(
        "needs",
        question="You fix the model of the demo with reweighing. What does that need?",
        options={
            "decide": "The sex of the people it decides about",
            "right": "The sex of the people it trains on",
            "labels": "New labels for some of the women",
        },
        correct="right",
        why={
            "decide": "That is what a threshold per group needs, as in [Equal yes rates](/course/open/rai-demographic-parity?lang=en). Reweighing changes only how much each person counts while the model learns; the model decides without `sex`.",
            "right": "The weights come from counts per sex and label, so every person to train on needs a sex. When the model decides, it does not: the model of the demo never saw the column `sex`, only the weights did.",
            "labels": "Changing labels is massaging, another method of Kamiran and Calders. Reweighing keeps every label as it is and changes only how much each person counts.",
        },
    )
    choice_needs
    return (choice_needs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Equal yes rates](/course/open/rai-demographic-parity?lang=en) a threshold per group made the decisions equal after training, and needed the sex of every person it decided about. Reweighing starts before training: it changes how much each person counts, so that in the data every sex has the same share that earns more. It needs the sex of the people to train on, not of the people the model decides about, and it brought the yes rates closer, not equal, for a small loss of accuracy. The three places to fix, before, during and after training, come back in every method you will meet. The next lesson, *Fixing the decision: a threshold per group*, goes back to the decision, and shows what each fairness criterion costs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_needs, result_shares, result_weights):
    L.wrap_up(result_weights, result_shares, choice_needs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
