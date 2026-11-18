# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-global-surrogate. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-global-surrogate", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en) and [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_average():
    quick_average = L.choice(
        "quick-average",
        question="In the worked example of How a prediction moves with a feature, the forest's partial dependence on age rises by 0.25 from age 25 to 45, while the lines of the people with education 13 or more rise by 0.476 on average. What does the average hide?",
        options={
            "ignores": "That the forest leaves education out",
            "right": "That age counts for more with a degree",
            "few": "That too few people aged 45 are in the table",
        },
        correct="right",
        why={
            "right": "Right. The lines of people with education up to 9 rise by 0.154, those with 13 or more by 0.476: what age does depends on education, and one average line cannot show that. This lesson looks for one picture of the whole forest, with all its features at once.",
            "ignores": "The forest uses education: the lines rise three times as much for people with 13 or more as for those with up to 9. The average over everyone hides that difference. See [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en).",
            "few": "At age 45 the forest is asked about all 1000 people, whatever their own age. What the average hides is that the lines differ by education: 0.154 for education up to 9, 0.476 for 13 or more. See [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en).",
        },
    )
    quick_average
    return (quick_average,)


@app.cell(hide_code=True)
def quick_check_pure():
    quick_pure = L.choice(
        "quick-pure",
        question="A decision tree grows until every part is pure. How many of its training examples does it get right?",
        options={
            "right": "All of them",
            "validation": "About as many as on the validation set",
            "common": "Only those of the most common class",
        },
        correct="right",
        why={
            "right": "Right: each leaf holds examples of one class and says that class, so every training example is right, and the tree overfits. In this lesson a tree copies another model's answers, and a copy grown that far agrees with them on every row it learned from. That says little; what counts is the rows it did not learn from.",
            "validation": "On the validation set it does worse: a tree grown until every part is pure gets every training example right, because each leaf holds one class. That is overfitting. See [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
            "common": "Each leaf says the class of the examples in it, and when every leaf is pure, that is the right class for each of them: all training examples are right. See [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
        },
    )
    quick_pure
    return (quick_pure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the last two lessons the supervisor of the bank measured which features the forest leans on, and drew how its yes moves with age; as before, a prediction of earning more than 50,000 dollars is the bank's yes. Both look at one feature at a time. Now the supervisor asks for the whole forest at once, as rules a person can read. But the forest is 100 trees with 28,295 leaves together, and no one reads that. So copy it: grow one small tree, not on who truly earns more, but on what the forest says. Then count how often the copy gives the same answer as the forest. Where it does, its few questions describe the forest; where it does not, they describe something else.

    **An everyday example.** A new employee at a bakery watches the head baker decide each evening which loaves go at half price, and writes three rules on a card. To know whether the card is any good, you check it against the baker's own decisions, not against what sold: if the card matches 19 evenings out of 20, it is a fair summary of the baker. It does not say why the baker decides, and on the 20th evening the baker had a reason the card does not know.
    """)
    return


@app.cell(hide_code=True)
def predict_baseline():
    guess_baseline = L.guess(
        "baseline",
        question="On the test half, the forest says yes to 14.8% of the people. Take a copy that says no to everyone. How often does it agree with the forest?",
        options={
            "low": "About 15% of the time",
            "half": "About half the time",
            "high": "About 85% of the time",
        },
        reveal="**About 85%: on 85.2% of the test half.** It agrees wherever the forest says no, and that is most people. So a copy's fidelity means little until you compare it with this number. A copy with one question about education gets 88.1%; the copy of depth 3 in the worked example below, 95.7%.",
    )
    guess_baseline
    return (guess_baseline,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Copy the answers, not the world.** Mark Craven and Jude Shavlik (1996) grew a decision tree from a trained neural network, with a method they called TREPAN. The network was the teacher: "the oracle uses the network to answer queries", and "TREPAN simply uses the network as a black box to answer queries during the extraction process." They then measured the tree in two ways: "Whereas accuracy is defined as the percentage of test-set examples that are correctly classified, fidelity is defined as the percentage of test-set examples on which the classification made by a tree agrees with its neural-network counterpart." With at most 15 splits in a tree, on four datasets, the trees agreed with their network on 85.7 to 95.9 percent of the test examples. On data about heart disease the network was right on 84.5 percent, the tree on 81.8 percent, and the two agreed on 94.1 percent.

        **A global surrogate.** Christoph Molnar calls it a global surrogate: "A global surrogate model is an interpretable model that is trained to approximate the predictions of a black box model." The recipe: take a dataset, ask the black box for its predictions, choose a kind of model you can read, train it on the dataset with those predictions, measure how well it reproduces them, and read it. Global, as in [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en): the copy covers the model for all rows at once, not one person.

        **Fidelity is not accuracy.** Accuracy compares the copy with the truth, fidelity with the model it copies. Where the copy agrees with the model, it is right exactly when the model is, so its accuracy can differ from the model's by at most the share of rows where the two disagree. Himabindu Lakkaraju, Ece Kamar, Rich Caruana and Jure Leskovec (2017) use the "agreement rate, fraction of instances in the data for which the label assigned by the approximation is the same as that of the black box model prediction, as a measure of fidelity". They also warn that fidelity and being easy to read pull apart: "Fidelity and interpretability are competing objectives, where fidelity favors details and nuances while interpretability favors simplicity." A deeper tree agrees more often, and is harder to read. Molnar measures fidelity with R² instead, which suits answers that are numbers, such as predicted chances.

        **Compared with what?** In UCI Adult most people earn 50,000 dollars or less, and so the forest says no to most of them. A copy that always says no then agrees with the forest wherever the forest says no. Compare every fidelity with that number. And there is no agreed threshold: Molnar, about R², "It's not clear what the best cut-off for R-squared is in order to be confident that the surrogate model is close enough to the black box model. 80% of variance explained? 50%? 99%?"

        **Close for some, far for others.** A fidelity over everyone can hide a group. Molnar: "It could happen that the interpretable model is very close for one subset of the dataset, but widely divergent for another subset." So measure it per group too. The slider below shows where copies made as in the worked example disagree with the forest.

        **Same answers, other reasons.** Cynthia Rudin (2019): "An explainable model that has a 90% agreement with the original model indeed explains the original model most of the time. However, an explanation model that is correct 90% of the time is wrong 10% of the time." Worse, a copy can agree for other reasons: "Even an explanation model that performs almost identically to a black box model might use completely different features, and is thus not faithful to the computation of the black box." Rudin's advice is to stop calling such copies explanations at all. And the copy describes the model, not the world; Molnar: "You have to be aware that you draw conclusions about the model and not about the data, since the surrogate model never sees the real outcome."

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **black box** | a model you only see from the outside: you give it rows, and it gives answers |
        | **global surrogate** | a model you can read, trained on the answers of a black box, so that it gives the same answers as often as possible; also called a surrogate model |
        | **fidelity** | how closely the surrogate's answers match those of the model it copies: for yes and no, the agreement rate; for numbers, often R². Measure it on rows it did not learn from, or it looks better than it is |
        | **agreement rate** | the share of rows where the surrogate gives the same answer as the model |
        | `export_text` | in scikit-learn: the questions and leaves of a tree as text, one line each |
        """),
        L.watch_out("Fidelity is not accuracy, and not a reason. A copy that agrees with the forest on 96% of the people is wrong about the forest for the other 4%, may be much worse for one group, and may reach the same answers with other features than the forest uses. Read the copy as a summary of the forest's answers, not of how the forest computes them."),
    ])
    return


@app.cell(hide_code=True)
def explore_depth_cell():
    explore_depth = L.explore("depth", "max_depth", lambda v: C.depth_picture(v, "en"), start=3,
                              values=C.DEPTHS, role="data",
                              task="Move the slider from depth 1 to 8. At which education levels does the copy get the forest wrong, and how much does the fidelity rise from one depth to the next?")
    mo.vstack([
        mo.md("**Where the copy disagrees with the forest, by depth**"),
        explore_depth,
        mo.md(r"""
        Copies made as in the worked example below: trees of depth 1 to 8, grown on the forest's answers on the training half. Each bar counts the people of the test half with that education_num for whom the copy disagrees with the forest; filled where the copy says yes and the forest no, open where the copy says no and the forest yes. education_num 9 is a high-school diploma, 13 a bachelor's degree.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To copy a model into one you can read:

    1. Take rows the model can answer, and split them: rows to grow the copy on, and rows to measure it on.
    2. Ask the model for its answers on all of them.
    3. Grow a small model you can read, such as a shallow tree, on the first rows and the model's answers, not the true labels.
    4. Measure fidelity on the other rows, and compare it with a copy that always gives the model's most common answer.
    5. Measure fidelity per group as well, and read the copy only where it agrees.
    6. Read the copy as a summary of the model's answers: not of the world, and not of how the model computes them.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The forest of [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en), with the same columns and halves. `_forest.estimators_` holds its 100 trees, and `.get_n_leaves()` counts the leaves of one tree. The copy is a tree of depth 3 grown on `_answers`, what the forest says for the training half, and `export_text(tree, feature_names=...)` gives its questions and leaves as text, with the names of the columns. `_same = _copy_says == _forest_says` is `True` for each person where the two agree, and `.mean()` of an array of `True` and `False` is the share of `True`: `_same.mean()` is the fidelity. `np.full(n, False)` makes an array of `n` times `False`: a copy that says no to everyone. A tree without `max_depth` grows until every part is pure. Run the cell. It takes a few seconds.
    """)
    return


@app.cell
def example_surrogate():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 1. the forest of the last lesson, grown on the training half
    _people = pd.read_csv("data/adult.csv.gz")
    _X = _people[["age", "education_num", "hours_per_week"]].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _leaves = 0
    for _tree in _forest.estimators_:
        _leaves = _leaves + _tree.get_n_leaves()
    print("leaves in the forest:", _leaves)

    # 2 and 3. the forest's answers on the training half, and a tree of depth 3 grown on them
    _answers = _forest.predict(_X[_train])
    _copy = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_X[_train], _answers)
    print(export_text(_copy, feature_names=["age", "education_num", "hours_per_week"]))

    # 4. fidelity on the test half, beside a copy that says no to everyone, and how often each is right
    _forest_says = _forest.predict(_X[_test])
    _copy_says = _copy.predict(_X[_test])
    _same = _copy_says == _forest_says
    print("fidelity:", _same.mean().round(3))
    print("saying no to everyone:", (np.full(len(_test), False) == _forest_says).mean().round(3))
    print("right, the copy:", (_copy_says == _y[_test]).mean().round(3))
    print("right, the forest:", (_forest_says == _y[_test]).mean().round(3))

    # 5. fidelity by education
    _degree = _X[_test][:, 1] >= 13
    _rest = _X[_test][:, 1] <= 12
    print("education 13 or more:", _degree.sum(), "people, fidelity", _same[_degree].mean().round(3))
    print("education up to 12:", _rest.sum(), "people, fidelity", _same[_rest].mean().round(3))

    # 6. a tree of depth 3 grown on the true labels instead, and a copy grown until every part is pure
    _labels = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_X[_train], _y[_train])
    print("grown on the true labels, fidelity:", (_labels.predict(_X[_test]) == _forest_says).mean().round(3))
    _full = DecisionTreeClassifier(random_state=0).fit(_X[_train], _answers)
    print("grown until pure:", _full.get_n_leaves(), "leaves")
    print("  fidelity on the training half:", (_full.predict(_X[_train]) == _answers).mean().round(3))
    print("  fidelity on the test half:", (_full.predict(_X[_test]) == _forest_says).mean().round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The forest has 28,295 leaves; the copy has 8, and only one of them says yes: education_num above 12.5, age above 31.5 and hours_per_week above 39.5. Its other questions split a part into two parts that both say no, because scikit-learn chooses a split by Gini impurity, which can drop while the answer stays the same. So the whole copy reads as one sentence: yes for people with a bachelor's degree or more who are 32 or older and work 40 hours or more, and no for everyone else. It agrees with the forest on 95.7% of the test half, and it is right about as often as the forest: 79.6% against 79.8%. But a copy that says no to everyone already agrees on 85.2%, and the fit is uneven: 98.8% for the 18,374 people with education up to 12, and 86.3% for the 6047 with 13 or more, where the forest says most of its yeses. A tree grown on the true labels agrees less, 93.1%: it learned the world, not the forest. And a copy grown until every part is pure agrees on 100% of the training half, the rows it learned from, and on 99.8% of the test half, but with 116 leaves it is no longer a summary anyone reads.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fidelity():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. How often do they agree?

        Write `fidelity(copy, original)`: `copy` and `original` are lists of `True` and `False` of the same length, one answer per person, and the function returns the share of places where `copy` gives the same answer as `original`. The check first calls `fidelity([True, False, False, True], [True, False, True, True])`, then other lists. The picture runs your function on the answers of the forest and of the copies of depth 1 to 8, for the 24,421 people of the test half.
        """),
        L.task("Write `fidelity(copy, original)`: count in `same` the places `i` where `copy[i]` equals `original[i]`, and return that count divided by the number of places."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fidelity():
    box_fidelity = L.code_box("fidelity", lines=8)
    box_fidelity
    return (box_fidelity,)


@app.cell(hide_code=True)
def check_fidelity(box_fidelity):
    result_fidelity = L.check("fidelity", box_fidelity.value)
    result_fidelity
    return (result_fidelity,)


@app.cell(hide_code=True)
def hints_fidelity():
    L.hints(
        "fidelity",
        [
            "Count the places where both lists say the same, and divide by how many places there are. Places where both say `False` count too.",
            "Start with `same = 0` and go over the places with `for i in range(len(copy)):`. When the two answers at place `i` are equal, add 1 to `same`. After the loop, return the share.",
            "`return same / ___`",
        ],
        why="`same = 0`, then `for i in range(len(copy)):` with `if copy[i] == original[i]:` and `same = same + 1`. After the loop, `return same / len(copy)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stump():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The best single question

        The smallest copy asks one question. The box holds three lists for the training half, with one number for each education_num from 1 to 16: `people`, how many people have it; `forest_yes`, how many of them the forest says yes to; and `true_yes`, how many of them truly earn more than 50,000 dollars. A copy with one question says yes when education_num is at least `t`. Fill `scores` with its fidelity for each `t` from 1 to 17: at `t` = 1 it says yes to everyone, at 17 it says no to everyone. Inside the loop over `t`, go over the levels with `e` from 0 to 15, with `same` counting the people for whom the copy says what the forest says: level `e` is education_num `e + 1`. `sum(people)` adds up a list. The picture shows your scores beside the forest's answers.
        """),
        L.task("Replace the `...`: for each level, add to `same` the people for whom the copy says what the forest says; after the levels, append the share of everyone to `scores`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stump():
    box_stump = L.code_box("stump", lines=17)
    box_stump
    return (box_stump,)


@app.cell(hide_code=True)
def check_stump(box_stump):
    result_stump = L.check("stump", box_stump.value)
    result_stump
    return (result_stump,)


@app.cell(hide_code=True)
def hints_stump():
    L.hints(
        "stump",
        [
            "For one `t`, the copy agrees with the forest for each person the forest says yes to at education_num `t` or more, and for each person the forest says no to below `t`. Count both, then divide by everyone.",
            "Inside `for t in range(1, 18):`, start `same = 0`, then `for e in range(16):`. If `e + 1 >= t`, add the forest's yeses of level `e`; otherwise add its noes, the people minus the yeses. After the loop over `e`, append the share to `scores`.",
            "`scores.append(same / ___)`",
        ],
        why="`same = 0`, then `for e in range(16):` with `if e + 1 >= t:` and `same = same + forest_yes[e]`, `else:` and `same = same + people[e] - forest_yes[e]`. After that loop, `scores.append(same / sum(people))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_between():
    mo.md(r"""
    ### Exercise 3. Agreeing and being right
    """)
    return


@app.cell(hide_code=True)
def choice_between_cell():
    choice_between = L.choice(
        "between",
        question="A copy agrees with the forest on 95% of the test half, and the forest is right about 80% of it. How often can the copy be right?",
        options={
            "same": "80%, the same as the forest",
            "right": "Anywhere from 75% to 85%",
            "fidelity": "95%, its fidelity",
        },
        correct="right",
        why={
            "right": "Right. Where the copy agrees with the forest, it is right exactly when the forest is. On the other 5% it can be right where the forest is wrong, or wrong where the forest is right, so its accuracy is at most 5 points from the forest's. In the worked example: fidelity 95.7%, the copy right on 79.6%, the forest on 79.8%.",
            "same": "Only where the two agree is the copy right exactly when the forest is. On the other 5% it can be right where the forest is wrong, or the other way around, so it can be up to 5 points better or worse. See *Fidelity is not accuracy* in *The idea*.",
            "fidelity": "Fidelity compares the copy with the forest, not with the truth: a copy that agrees with the forest is wrong wherever the forest is. Its accuracy is at most 5 points from the forest's 80%. See *Fidelity is not accuracy* in *The idea*.",
        },
    )
    choice_between
    return (choice_between,)


@app.cell(hide_code=True)
def text_exercise_reasons():
    mo.md(r"""
    ### Exercise 4. The same answers
    """)
    return


@app.cell(hide_code=True)
def choice_reasons_cell():
    choice_reasons = L.choice(
        "reasons",
        question="A copy agrees with the forest on 99% of the test half. What does that tell you about how the forest decides?",
        options={
            "same": "That the forest asks the same questions in order",
            "right": "Little: other routes give the same answers",
            "all": "Everything: the copy is the forest made small",
        },
        correct="right",
        why={
            "right": "Right. Rudin: a copy that performs almost identically \"might use completely different features, and is thus not faithful to the computation of the black box.\" And 99% over everyone can hide a group: in the worked example, the copy of depth 3 agrees on 98.8% of the people with education up to 12 and on 86.3% of those with 13 or more.",
            "same": "The copy is grown on the forest's answers, not on its trees: it can reach the same answers with other questions, or other features. See *Same answers, other reasons* in *The idea*.",
            "all": "The copy describes the forest's answers, not how the forest computes them, and the 1% it gets wrong may all sit in one group. See *Same answers, other reasons* in *The idea*.",
        },
    )
    choice_reasons
    return (choice_reasons,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Which features matter?](/course/open/rai-permutation-importance?lang=en) and [How a prediction moves with a feature](/course/open/rai-pdp-ice?lang=en) looked at the forest one feature at a time; this lesson copies all of it into one tree, as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en), but grown on the forest's answers. Like those two, a global surrogate is a global explanation, and model-agnostic: it needs only the model's answers, in the terms of [Kinds of explanation](/course/open/rai-kinds-of-explanation?lang=en). The next lesson, [LIME: a simple model around one prediction](/course/open/rai-lime?lang=en), copies the model too, but only close to one person, to explain one decision.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_between, choice_reasons, result_fidelity, result_stump):
    L.wrap_up(result_fidelity, result_stump, choice_between, choice_reasons)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
