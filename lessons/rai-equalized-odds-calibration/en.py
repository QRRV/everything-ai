# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-equalized-odds-calibration. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-equalized-odds-calibration", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en) and [Equal yes rates](/course/open/rai-demographic-parity?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="Of 50 people who really earn more than 50,000 dollars, a model gives 40 a yes. What is its recall for them?",
        options={
            "miss": "0.2, the share it misses",
            "right": "0.8, as 40 / 50",
            "count": "40 people",
        },
        correct="right",
        why={
            "miss": "Recall counts the people the model finds, not the ones it misses: 40 / 50 = 0.8. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
            "right": "Of the 50 who earn more, the model finds 40: a recall of 0.8.",
            "count": "Recall is a share, not a count: 40 of the 50, so 0.8. See *The idea* in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def quick_check_parity():
    quick_parity = L.choice(
        "quick-parity",
        question="With a threshold per group, women and men both got a yes 0.19 of the time. What stayed unequal?",
        options={
            "rates": "The share of each group with a yes",
            "right": "The precision: 0.43 against 0.73",
            "nothing": "Nothing: the model was then fair",
        },
        correct="right",
        why={
            "rates": "Those were the same, 0.19 for both: that is what the threshold per group did. See *Try it* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
            "right": "Of the women with a yes, 0.43 really earned more; of the men, 0.73. Equal yes rates do not make a yes mean the same.",
            "nothing": "Equal yes rates are one criterion. A yes still meant something different for women and men: a precision of 0.43 against 0.73. See *Try it* in [Equal yes rates](/course/open/rai-demographic-parity?lang=en).",
        },
    )
    quick_parity
    return (quick_parity,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Equal yes rates](/course/open/rai-demographic-parity?lang=en), a threshold per group gave women and men a yes equally often, but did not look at who deserved the yes. This lesson asks two other questions. Are a model's mistakes spread evenly over the groups? And does its score mean the same in every group? Each question is a criterion of group fairness, and on the table UCI Adult they pull in different directions.

    **An everyday example.** A weather service says *70 percent chance of rain*. If it then rains on about 7 of 10 such days, in the city and at the coast alike, the forecast means the same in both places. Another question: how often did the service warn of rain on days that stayed dry, and how often did it give no warning on days it rained? Can the forecast mean the same in both places, and make both mistakes equally often in both, when it rains more often at the coast? Only if it is never wrong. This lesson shows why.
    """)
    return


@app.cell(hide_code=True)
def predict_missed():
    guess_missed = L.guess(
        "missed",
        question="With one threshold of 0.5, the model without `sex` is about equally precise for women and men: of those with a yes, 0.67 and 0.69 really earn more than 50,000 dollars. Of the men who really earn more, 0.57 get a yes. Of such women, how many do you think get a yes?",
        options={
            "same": "About as many: 0.57",
            "right": "Fewer: about 0.44",
            "more": "More: about 0.70",
        },
        reveal="**Fewer: about 0.44.** A yes means about the same for women and men, but the model finds fewer of the women who earn more. Fewer women than men earn more in the data, 0.11 against 0.30, and the model gives the women who do a lower probability on average than such men: 0.45 against 0.54. More of them stay below the threshold. The demo below shows it.",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Equal errors.** A model that says yes or no makes two kinds of mistakes: it can miss someone who really is positive, and it can give a yes to someone who is not. Count both per group. The recall of a group, the share of its positives that gets a yes, is also called its **true positive rate**. Its **false positive rate** is the share of its negatives that gets a yes anyway: $FP / (FP + TN)$. Moritz Hardt, Eric Price and Nathan Srebro (2016) ask that both are the same in every group, and call that **equalized odds**. Barocas, Hardt and Narayanan call it separation: it asks for "error rate parity", with one question behind it: "Who bears the cost of misclassification?" Unlike demographic parity, it allows a model that is always right, but it punishes "models that perform well only on the majority." Hardt and colleagues also name "a weaker, though still interesting, notion", **equal opportunity**: only the true positive rate has to be the same. Their example is a loan: "people who pay back their loan, have an equal opportunity of getting the loan in the first place".

    **Equal meaning.** The model also gives each person a probability. It is calibrated when its probabilities come true as often as they say: of all the people it gives about 0.7, about 70 percent really are positive. Barocas, Hardt and Narayanan put it as "the set of all instances assigned a score value r has an r fraction of positive instances among them." **Calibration within groups** asks that this holds in every group: then a score means the same, whatever the group. For a yes or a no, the nearest check is the same precision in every group, which Alexandra Chouldechova (2017) calls predictive parity; it is weaker, because it says nothing about the people with a no. Jon Kleinberg, Sendhil Mullainathan and Manish Raghavan (2016) say why it matters: without it, "if people of two different groups both belong to bin b, the decision-maker has a clear incentive to treat them differently". It is a statement about all the people with the same score, not about one of them: "Calibration does not tell us anything about the outcome of a specific individual". Barocas, Hardt and Narayanan call it sufficiency, and find that "sufficiency often comes for free (at least approximately) as a consequence of standard machine learning practices", on one condition: "the classifier must be able to detect group membership." The model without `sex` can: in [Leaving out the column is not enough](/course/open/rai-unawareness-proxies?lang=en), a model guessed sex from the other columns right for 0.84 of the people.

    **Why not both.** Take 1,000 people of a group, its **base rate**, the share that really is positive, and the precision and recall of a model in that group. Then its false positive rate follows. With the numbers of the Black defendants in [COMPAS and a hiring tool](/course/open/rai-case-compas-amazon?lang=en), where a flag means rated medium or high risk:

    1. Per 1,000 Black defendants, 514 were arrested again within two years: a base rate of 0.514.
    2. The score flagged 0.72 of them, its recall: 370 right flags.
    3. Its precision was 0.63: the 370 right flags are 0.63 of all flags, so there were 370 / 0.63 = 587 flags, and 587 − 370 = 217 of them were wrong.
    4. Of the 1,000, 486 were not arrested again, so the false positive rate is 217 / 486 = 0.45.

    The same steps with the numbers of the White defendants, a base rate of 0.394, a recall of 0.52 and a precision of 0.59, give 0.23. Now give two groups the same precision and the same recall: the base rate still changes steps 1 and 4, so their false positive rates differ. Kleinberg and colleagues proved it in general: calibration within groups and equal error rates can only hold together if the model is perfect or the groups have the same base rate, "regardless of how the risk assignment is computed", by a model or by people. Chouldechova proved a version for one threshold. That does not make fairness impossible. It means these criteria cannot all hold at once, so you have to choose.

    **COMPAS made it visible.** In 2016, journalists at ProPublica found that the score COMPAS wrongly flagged Black defendants almost twice as often as White defendants: the false positive rates of 0.45 and 0.23 above. Overall, they wrote, it "made mistakes with black and white defendants at roughly the same rate but in very different ways." The maker, Northpointe, answered with equal meaning. Barocas, Hardt and Narayanan write that the company "emphasized the importance of calibration by group in their rebuttal", while "ProPublica implicitly adopted equality of false positive rates as a fairness criterion". In a reply to ProPublica, Anthony Flores and colleagues (2016) found that "a given COMPAS score translates into roughly the same likelihood of recidivism, regardless of race." Chouldechova showed why both findings held at once: in the data, 51 percent of the Black defendants and 39 percent of the White defendants were arrested again, and COMPAS "is (approximately) well-calibrated, satisfies predictive parity for high-risk cutoff values of 4 or higher, but fails to have error rate balance." With those base rates it could not have both.

    **Try it.** The cell below trains the model without the column `sex` of [Equal yes rates](/course/open/rai-demographic-parity?lang=en) again. With one threshold of 0.5, it prints per sex the recall, the false positive rate and the precision. Then it checks equal meaning: per band of probabilities, from 0 to 0.2 up to 0.8 to 1, the share of the women and of the men that really earns more. Last, it gives the women a lower threshold, so that their recall is the men's: equal opportunity. `*` between two masks keeps the rows where both are `True`. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def demo_errors():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _poor = _tested["income"] == "<=50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"
    print("really earn more: women", round(_rich[_women].mean(), 2), "men", round(_rich[_men].mean(), 2))

    # the model without the column sex, as in the previous lessons
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _score = _model.predict_proba(_X[_test])[:, 1]   # for each person, the probability of a yes


    def _show(name, yes):
        # per sex: the recall, the false positive rate and the precision of the yes
        print(name)
        for sex, group in {"women": _women, "men": _men}.items():
            recall = yes[group * _rich].mean()
            false_positive_rate = yes[group * _poor].mean()
            precision = _rich[group * yes].mean()
            print("  ", sex, "| recall", round(recall, 2),
                  "| false positive rate", round(false_positive_rate, 2), "| precision", round(precision, 2))


    # 1. one threshold for everyone: a yes where the probability is above 0.5
    _one = _score > 0.5
    _show("one threshold of 0.5", _one)

    # 2. equal meaning: per band of probabilities, the share of the women and of the men who really earn more
    print("per band of probabilities, the share that really earns more")
    for _low, _high in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]:
        _band = (_score >= _low) * (_score < _high)
        print("  ", _low, "to", _high, "| women", round(_rich[_band * _women].mean(), 2),
              "| men", round(_rich[_band * _men].mean(), 2))

    # 3. equal recall: a lower threshold for the women, so that as large a share of the women who earn more
    # gets a yes as of the men who earn more
    _recall_men = _one[_men * _rich].mean()
    _threshold_women = np.quantile(_score[_women * _rich], 1 - _recall_men)
    print("the women's threshold for equal recall:", round(_threshold_women, 2))
    # each woman gets the women's threshold, each man 0.5
    _threshold = _threshold_women * _women + 0.5 * _men
    _show("equal recall", _score > _threshold)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_meaning({"title": "Does a probability mean the same for women and men?",
                                   "xlabel": "average probability in the band", "ylabel": "really earn more",
                                   "line": "probability = share", "key": "open: women, filled: men; one dot per band of probabilities",
                                   "decimal": ".",
                                   "alt": "Five dots for women and five for men, one per band of probabilities. Across, the model's average probability in the band; up, the share that really earns more. Women: 0.05 and 0.03, 0.28 and 0.25, 0.49 and 0.45, 0.70 and 0.69, 0.86 and 0.88. Men: 0.06 and 0.06, 0.29 and 0.32, 0.49 and 0.50, 0.70 and 0.69, 0.86 and 0.84. All dots lie close to the dashed line, where the probability equals the share."})),
        mo.Html(C.picture_errors({"title": "The errors of the model per sex",
                                  "rows": ["One threshold of 0.5", "Equal recall"],
                                  "xlabels": ["recall", "false positive rate", "precision"],
                                  "key": "open: women, filled: men", "decimal": ".",
                                  "alt": "Three panels with two rows each. The recall of women and men: one threshold 0.44 and 0.57, equal recall 0.57 and 0.57. The false positive rate: one threshold 0.03 and 0.11, equal recall 0.05 and 0.11. The precision: one threshold 0.67 and 0.69, equal recall 0.59 and 0.69."})),
        mo.md(r"""
        With one threshold, the model's probabilities mean about the same for women and men: the dots of both lie close to the dashed line, and of the women and the men with a yes, 0.67 and 0.69 really earn more. Equal meaning, nearly. But the errors differ. Of the women who earn more, 0.44 get a yes, of the men 0.57: the model misses more of the women. And of the people who do not earn more, 0.03 of the women get a yes anyway, and 0.11 of the men. Who bears which cost depends on what the yes gives.

        With a lower threshold for the women, 0.37, which again needs the column `sex` when deciding, their recall becomes the men's, 0.57: equal opportunity. Their false positive rate goes up to 0.05, still not the men's 0.11, and their precision goes down to 0.59: a yes now means less for a woman than for a man. The model is right as often as before, 0.83 of the time. Equal false positive rates too, Hardt and colleagues found on credit scores, "cannot be achieved with a single threshold for each group, but requires randomization": some people near the threshold would get a yes by a coin toss.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **true positive rate** | another name for the recall: of the people who really are positive, the share that gets a yes |
        | **false positive rate** | of the people who really are negative, the share that gets a yes anyway: $FP / (FP + TN)$ |
        | **base rate** | the share of a group that really is positive, such as 0.11 of the women and 0.30 of the men in UCI Adult |
        | **equalized odds** | every group has the same true positive rate and the same false positive rate: the errors are spread evenly; also called separation |
        | **equal opportunity** | only the true positive rate is the same in every group |
        | **calibration within groups** | in every group, of the people with a probability of about 0.7, about 70 percent are positive: a score means the same in each group; for a yes or a no, at least the same precision in every group |
        """),
        L.watch_out("Say which class is positive. Here a yes is a good thing, so a false positive gives someone a benefit by mistake. With COMPAS, the flag was the bad outcome, so a false positive took something from someone. Barocas, Hardt and Narayanan call the switch \"a frequent source of terminological confusion\"."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To check the errors and the meaning of a model or a rule per group:

    1. Say what counts as positive, and whether a yes helps or harms.
    2. Per group, take the true positive rate, the share of the positives with a yes, and the false positive rate, the share of the negatives with a yes. Both equal: equalized odds. Only the first: equal opportunity.
    3. Per group, take the share that is positive per band of scores. Equal in every group: calibration within groups. For a yes or a no, compare at least the precision: predictive parity.
    4. Compare the base rates. If they differ and the model makes mistakes, it cannot meet both criteria: choose the one that fits the decision, and say why.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The made-up hiring rule of [Equal yes rates](/course/open/rai-demographic-parity?lang=en): a yes for everyone with at least a bachelor's degree, an `education_num` of 13 or more. `_column` is the column `race`. The cell takes the people who earn more than 50,000 dollars, and works out per group the share of them that the rule gives a yes: the true positive rate of step 2. Run the cell.
    """)
    return


@app.cell
def example_recall(pd):
    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "race"

    _rich = _people[_people["income"] == ">50K"]
    _tpr = _rich[_rich["education_num"] >= 13].groupby(_column).size() / _rich.groupby(_column).size()
    print(_tpr.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Of the people in the group Asian-Pac-Islander who earn more, the rule gives 0.67 a yes; in the group Amer-Indian-Eskimo, 0.33. That is far from equal opportunity, although only 55 people in the second group earn more, so its share rests on few people. Exercise 1 adds the false positive rate, per sex.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_errors():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The errors of the rule by sex

        The rule of the worked example, now for `column`, the column `sex`. Replace the `...` with four lines: `rich` and `tpr` as in the worked example, with the names without their underscore, then in the same way `poor`, the rows where `income` is `"<=50K"`, and `fpr`, the share of them that the rule gives a yes. The last line is `tpr, fpr`, so the check sees both. The check also tries another column.
        """),
        L.task("Work out `tpr` and `fpr` of the rule per value of `sex`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_errors():
    box_errors = L.code_box("errors", lines=12)
    box_errors
    return (box_errors,)


@app.cell(hide_code=True)
def check_errors(box_errors):
    result_errors = L.check("errors", box_errors.value)
    result_errors
    return (result_errors,)


@app.cell(hide_code=True)
def hints_errors():
    L.hints(
        "errors",
        [
            "The true positive rate looks at the people who earn more, the false positive rate at the people who do not. For both, divide the people with a yes per group by all the people of that kind per group.",
            "`poor = people[people[\"income\"] == \"<=50K\"]`. The rule picks the rows with `poor[\"education_num\"] >= 13`.",
            "`fpr = poor[poor[\"education_num\"] >= 13].groupby(column).size() / ___`.",
        ],
        why="`rich = people[people[\"income\"] == \">50K\"]`, `tpr = rich[rich[\"education_num\"] >= 13].groupby(column).size() / rich.groupby(column).size()`, `poor = people[people[\"income\"] == \"<=50K\"]` and `fpr = poor[poor[\"education_num\"] >= 13].groupby(column).size() / poor.groupby(column).size()`. That gives a recall of 0.53 for women and 0.49 for men, and a false positive rate of 0.18 and 0.16.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clash():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The same precision, the same recall

        In the test set of UCI Adult, 0.109 of the women and 0.300 of the men earn more than 50,000 dollars. Suppose a model has the same precision for both, 0.69, and the same recall, 0.57. Which false positive rate must each group then have? `p` holds the base rate of each group; the loop goes through its keys, with in `group` the name of one group. Replace the `...` in the loop with the steps of *Why not both*, per 1,000 people of the group: `positives`, the people who earn more; `true_yes`, the share `recall` of them; `false_yes`, all the yes (`true_yes / precision`) minus the true yes; and `fpr[group]`, the false yes divided by the people who do not earn more. The check also tries another precision and recall.
        """),
        L.task("Work out `fpr[group]` for each group in `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clash():
    box_clash = L.code_box("clash", lines=14)
    box_clash
    return (box_clash,)


@app.cell(hide_code=True)
def check_clash(box_clash):
    result_clash = L.check("clash", box_clash.value)
    result_clash
    return (result_clash,)


@app.cell(hide_code=True)
def hints_clash():
    L.hints(
        "clash",
        [
            "Follow the four steps of *Why not both* for 1,000 people of one group. Only the base rate changes from group to group.",
            "`positives = 1000 * p[group]` and `true_yes = recall * positives`. The people who do not earn more are `1000 - positives`.",
            "`false_yes = true_yes / precision - ___`.",
        ],
        why="In the loop: `positives = 1000 * p[group]`, `true_yes = recall * positives`, `false_yes = true_yes / precision - true_yes` and `fpr[group] = false_yes / (1000 - positives)`. That gives 0.03 for the women and 0.11 for the men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 3. What a claim compares

    Six claims about a model or a score. Sort each by what it compares between the groups.
    """)
    return


@app.cell(hide_code=True)
def sort_which_cell():
    sort_which = L.sort("which")
    sort_which
    return (sort_which,)


@app.cell(hide_code=True)
def check_which(sort_which):
    result_which = L.check("which", sort_which)
    result_which
    return (result_which,)


@app.cell(hide_code=True)
def hints_which():
    L.hints(
        "which",
        [
            "Look at where each claim starts: with everyone, with the people who do or do not have the outcome, or with the people who have a yes or a score.",
            "From everyone: the yes rates. From the people with or without the outcome: the errors. From the people with a yes or a score: what the score means.",
            "The errors: ProPublica, and \\_\\_\\_. What a score means: Northpointe, and \\_\\_\\_.",
        ],
        why="**The yes rates:** a threshold per group, and interviews. **The errors:** ProPublica, and missed. **What a score means:** about 0.7, and Northpointe.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Equal yes rates](/course/open/rai-demographic-parity?lang=en) you counted the yes per group. Here you counted the mistakes per group, for equalized odds and equal opportunity, and asked whether a score means the same in every group, for calibration within groups. On UCI Adult, the model came close to equal meaning with unequal errors, and your rule by schooling close to equal errors with unequal meaning. Your second exercise showed why: with different base rates and a model that makes mistakes, one comes at the cost of the other, as with COMPAS. Which criterion fits depends on the decision and on who bears the cost of a mistake. The next lesson, *Testing for discrimination*, looks at how to find out whether decisions treat groups differently.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_clash, result_errors, result_which):
    L.wrap_up(result_errors, result_clash, result_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
