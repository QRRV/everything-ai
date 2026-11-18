# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-fair-thresholds. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fair-thresholds", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en) and [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_opportunity():
    quick_opportunity = L.choice(
        "quick-opportunity",
        question="A model gives a yes to 0.44 of the women who earn more and to 0.57 of the men who earn more. Which criterion does it break?",
        options={
            "calibration": "Calibration within groups",
            "right": "Equal opportunity",
            "none": "None, if the yes rates are the same",
        },
        correct="right",
        why={
            "calibration": "Calibration within groups is about what a score means: of the people with a probability of about 0.7, about 70 percent are positive. These numbers are recalls, the share of the people who really earn more that gets a yes. See *The idea* in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).",
            "right": "Equal opportunity asks for the same true positive rate, the recall, in every group: here 0.44 against 0.57.",
            "none": "Equal opportunity does not look at the yes rates, only at the recall, and 0.44 is not 0.57. See *The idea* in [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en).",
        },
    )
    quick_opportunity
    return (quick_opportunity,)


@app.cell(hide_code=True)
def quick_check_reweighing():
    quick_reweighing = L.choice(
        "quick-reweighing",
        question="Reweighing made the model of UCI Adult treat women and men more alike. What did it change?",
        options={
            "labels": "The labels of some of the women and of the men",
            "right": "How much each person counts in training",
            "threshold": "The thresholds it uses for women and men",
        },
        correct="right",
        why={
            "labels": "Changing labels is massaging. Reweighing keeps every label and gives each person a weight: how many people that person counts for while the model learns. See *The idea* in [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en).",
            "right": "Each person got a weight by sex and label, and the model was trained again with those weights. Its threshold stayed 0.5 for everyone.",
            "threshold": "The threshold stayed 0.5 for everyone. Reweighing gave each person a weight while the model learned, and needed a new training run. See *The idea* in [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en).",
        },
    )
    quick_reweighing
    return (quick_reweighing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en), a lower threshold for the women gave them the recall of the men, and the model was right about as often as before. In [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en), weights brought the yes rates closer, but the model had to be trained again. Often that is not possible: the model comes from another team or from a supplier, and all you get is its probability for each person. What you can still choose is the threshold. This lesson searches, for each fairness criterion, the pair of thresholds that gets the most people right, and measures what each pair costs and what it leaves unequal.

    **An everyday example.** A school gives two classes the same subject, each with its own version of the test, and one version turns out to be harder. To let the same share of the students who know the material pass in both classes, the school sets a lower pass mark for the harder version. Nobody's answers change, only the mark that counts as a pass. But now a student of one class and a student of the other with the same score can get a different result.
    """)
    return


@app.cell(hide_code=True)
def predict_cost():
    guess_cost = L.guess(
        "cost",
        question="With one threshold of 0.5, the model is right for 0.834 of the people in the test set. With the best pair of thresholds that gives women and men the same recall, how often do you think?",
        options={
            "less": "0.78, clearly less often",
            "right": "0.832, almost as often",
            "more": "0.85, more often than before",
        },
        reveal="**0.832, almost as often.** The women who move from a no to a yes have a probability just below 0.5, and about four in ten of them really earn more: their new yes is right for about four in ten of them. Other criteria cost more. The demo below shows it.",
    )
    guess_cost
    return (guess_cost,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **After training, on the probability.** Moritz Hardt, Eric Price and Nathan Srebro (2016) offered a fix for a model that is already trained: "Rather than changing a possibly complex training pipeline, the result follows via a simple post-processing step that minimizes the loss in utility." The new decision may use two things only: the model's probability for a person, and that person's group. No other column is needed, and no weight of the model changes. With a threshold per group, a woman's probability is compared with the women's threshold, and a man's with the men's.

    **Searching for the thresholds.** Choose a criterion, such as equal recall, and how large a gap you allow, such as less than 0.01. Then try pairs: a threshold for the women and one for the men, each from 0.01 to 0.99. For each pair, work out the gap between the two recalls. Keep the pairs with a gap below 0.01, and of those, take the pair that gets the most people right. That search is **threshold optimization**. How many a pair gets right adds up per group: the women it gets right plus the men it gets right. So you count once per group and threshold, and then only add. The library Fairlearn has a tool for it, `ThresholdOptimizer`: "The thresholds are chosen to optimize the provided performance objective subject to the provided fairness constraints." Its guide describes the search: "it generates all possible thresholds and selects the best combination in terms of the objective and the fairness constraints."

    **Choose on one part, measure on another.** If you choose the thresholds on the test set and then measure on the same test set, the rates look more equal than they will be for new people. So the search uses the people the model was trained on, and the test set only measures. That works here because the model hardly learned its training people by heart: with one threshold of 0.5, it is right for 0.835 of them and for 0.834 of the test set. With a model that does, set aside a [validation set](/course/open/ml-validation-set?lang=en) for the search.

    **Equal false positive rates too: a coin.** With one threshold per group, equal recall and equal false positive rates together are only possible at the pairs where both happen to come out the same; on UCI Adult the demo finds those only at high thresholds, where few people get a yes. Hardt and colleagues therefore allow two thresholds for a group: above both a yes, below both a no, and in between "we flip a coin". Fairlearn's guide says the same: for an exact match "one typically needs to randomize between two thresholds." Then two people with the same probability and the same group can get different answers. This lesson uses no coin.

    **What it needs: the group, when deciding.** Fairlearn's guide stresses it: "Importantly, ThresholdOptimizer requires the sensitive features to be available at deployment time". Barocas, Hardt and Narayanan, in *Fairness and Machine Learning*, describe a common objection to post-processing: the classifier "uses group membership quite explicitly by setting different acceptance thresholds for different groups." Whether that is allowed depends on the country and the decision; see [Direct and indirect discrimination](/course/open/rai-discrimination-law?lang=en). The Algoritmekader, the Dutch government's guide for algorithms, says that using a ground such as sex as an input variable counts as direct distinction, which is prohibited, unless, for example, a legal exception allows it. On US law, Barocas, Hardt and Narayanan write that "The answer is nuanced."

    **Try it.** The cell below trains the model without the column `sex` once, and after that only uses its probabilities. `_rates` counts, for one sex in one part of the table and one threshold, how many it gets right, the yes rate, the recall and the false positive rate. Step 1 fills `_table` for every threshold on the people to train on. Step 2 tries all 99 × 99 pairs `_t_women` and `_t_men`: `_gap` holds per criterion how far the pair is from it, and per criterion `_best` keeps the pair with the most people right, `_most`, among the pairs with a gap below 0.01. One threshold of 0.5 for everyone stays as it is. Step 3 measures each chosen pair on the test set. Run the cell; it takes a few seconds.
    """)
    return


@app.cell
def demo_search():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48,842 people from a US survey of 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _rich = _people["income"] == ">50K"
    _poor = _people["income"] == "<=50K"

    # the model without the column sex, as in the previous lessons: trained once, and not changed below
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _rich[_train])
    _score = _model.predict_proba(_X)[:, 1]   # for each person, the probability of a yes


    def _rates(part, sex, threshold):
        # one sex in one part of the table, with one threshold: how many it gets right, and three shares
        group = (_people["split"] == part) * (_people["sex"] == sex)
        yes = _score > threshold
        return {"people": group.sum(), "right": (yes == _rich)[group].sum(),
                "yes rate": yes[group].mean(), "recall": yes[group * _rich].mean(),
                "false positive rate": yes[group * _poor].mean()}


    # 1. per sex and per threshold from 0.01 to 0.99, on the people to train on
    _thresholds = np.arange(1, 100) / 100
    _table = {"Female": {}, "Male": {}}
    for _sex in _table:
        for _t in _thresholds:
            _table[_sex][_t] = _rates("train", _sex, _t)

    # 2. per criterion, of the pairs with a gap below 0.01, the pair that gets the most people right
    _best = {"one threshold": (0.5, 0.5), "equal yes rates": None, "equal recall": None,
             "equal recall and false positive rate": None}
    _most = {"equal yes rates": 0, "equal recall": 0, "equal recall and false positive rate": 0}
    for _t_women in _thresholds:
        for _t_men in _thresholds:
            _w = _table["Female"][_t_women]
            _m = _table["Male"][_t_men]
            _right = _w["right"] + _m["right"]
            _recall_gap = abs(_w["recall"] - _m["recall"])
            _gap = {"equal yes rates": abs(_w["yes rate"] - _m["yes rate"]),
                    "equal recall": _recall_gap,
                    "equal recall and false positive rate":
                        max(_recall_gap, abs(_w["false positive rate"] - _m["false positive rate"]))}
            for _name in _gap:
                if _gap[_name] < 0.01:
                    if _right > _most[_name]:
                        _most[_name] = _right
                        _best[_name] = (_t_women, _t_men)

    # 3. each chosen pair on the test set
    for _name in _best:
        _t_women, _t_men = _best[_name]
        _w = _rates("test", "Female", _t_women)
        _m = _rates("test", "Male", _t_men)
        print(_name, "| thresholds", round(_t_women, 2), "and", round(_t_men, 2),
              "| right", round((_w["right"] + _m["right"]) / (_w["people"] + _m["people"]), 3))
        for _sex, _r in {"women": _w, "men": _m}.items():
            print("  ", _sex, "| yes rate", round(_r["yes rate"], 2),
                  "| recall", round(_r["recall"], 2),
                  "| false positive rate", round(_r["false positive rate"], 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_criteria({"title": "One model, four pairs of thresholds, on the test set",
                                    "rows": ["One threshold", "Equal yes rates", "Equal recall", "Equalized odds, nearly"],
                                    "xlabels": ["yes rate", "recall", "false positive rate"], "right": "accuracy",
                                    "key": "open: women, filled: men; at the right of each row, how often the model is right", "decimal": ".",
                                    "alt": "Three panels with four rows each, for women and men. One threshold of 0.5: yes rate 0.07 and 0.25, recall 0.44 and 0.57, false positive rate 0.03 and 0.11, accuracy 0.834. Equal yes rates: 0.14 and 0.15, recall 0.64 and 0.37, false positive rate 0.07 and 0.05, accuracy 0.817. Equal recall: yes rate 0.10 and 0.24, recall 0.55 and 0.55, false positive rate 0.05 and 0.10, accuracy 0.832. Equalized odds, nearly: yes rate 0.03 and 0.08, recall 0.22 and 0.22, false positive rate 0.01 and 0.02, accuracy 0.805."})),
        mo.md(r"""
        No weight of the model changed; only the thresholds did. **Equal recall** chose 0.38 for the women and 0.52 for the men: on the test set, 0.55 of the women and of the men who earn more get a yes, and the model is right for 0.832 of the people instead of 0.834. That is cheap because the probabilities mean about the same in both groups. 173 women in the test set have a probability between 0.38 and 0.5, and 0.40 of them really earn more: their new yes is right for about 70 of them and wrong for about 100. But the yes rates stay apart, 0.10 against 0.24, and so do the false positive rates, 0.05 against 0.10. In [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en) the women's threshold was 0.37, chosen on the test set itself; here it is 0.38, chosen on the people to train on.

        **Equal yes rates** chose 0.29 and 0.67: 0.14 of the women and 0.15 of the men get a yes, and the model is right for 0.817. Here the men pay: the 1,114 men in the test set with a probability between 0.5 and 0.67 now get a no, while 0.58 of them really earn more. And the recall turns around: 0.64 of the women who earn more get a yes, and 0.37 of such men. In [Equal yes rates](/course/open/rai-demographic-parity?lang=en) each group got a yes as often as all people together with one threshold, 0.19; this search looks for the pair with the most people right instead.

        For **equal recall and false positive rate**, the best pair with one threshold per group was 0.71 and 0.77: a recall of 0.22 for both, and the model is right for 0.805, not far above the 0.764 that a no for everyone gets. On the test set, the false positive rates are 0.007 and 0.019: the gap below 0.01 that held on the people to train on does not quite hold on new people.

        On credit scores, Hardt and colleagues also found equal opportunity the cheapest of the three fairness criteria, measured in profit; in one of their settings, "a race blind threshold gets 99.3% of the maximal profit, equal opportunity gets 92.8%, equalized odds gets 80.2%, and demographic parity gets 69.8%." What a criterion costs depends on the data, and each one leaves another unequal: choosing the criterion is a choice about the decision, not about the model.

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **threshold optimization** | choosing a threshold per group after training: of the pairs of thresholds that meet a fairness criterion, the one that gets the most people right, or scores best on another measure |
        | **randomized threshold** | two thresholds for one group, with a coin for the people whose probability lies between them; usually needed for exact equalized odds |
        | `ThresholdOptimizer` | Fairlearn's tool for threshold optimization; it needs each person's group when it decides |
        """),
        L.watch_out("Equal on the test set is not equal everywhere. Frances Ding and colleagues fitted such thresholds on survey data from US states, and found that \"even when a method achieves demographic parity in one state, it may no longer satisfy the fairness constraint when naively deployed on another.\" Measure again on the people the model really decides about."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To choose a threshold per group after training:

    1. Take the trained model's probabilities for people whose label you know and who are not your test set, such as the people it trained on or a validation set.
    2. Choose the criterion and the gap you allow. Per group and per threshold, count how many it gets right and the rates the criterion compares.
    3. Try every pair of thresholds; keep the pairs with a gap below what you allow, and take the one that gets the most people right.
    4. Measure the pair on the test set: per group the yes rate, the recall and the false positive rate, and how often it is right. Say which criterion it meets, which it leaves unequal, and that every decision now needs the person's group.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two made-up groups, a and b, of 100 people each, and three thresholds. `_yes_a` holds per threshold the yes rate of group a, `_right_a` how many people of group a it gets right, and `_yes_b` and `_right_b` the same for group b. The cell searches the pair with equal yes rates: for each pair, the gap between the yes rates; for a gap below 0.02, how many the pair gets right; and in `_best` the pair with the most, `_most`.
    """)
    return


@app.cell
def example_search():
    # two made-up groups of 100 people: per threshold, the yes rate and how many it gets right
    _yes_a = {0.4: 0.30, 0.5: 0.20, 0.6: 0.12}
    _yes_b = {0.4: 0.45, 0.5: 0.31, 0.6: 0.21}
    _right_a = {0.4: 80, 0.5: 84, 0.6: 82}
    _right_b = {0.4: 78, 0.5: 85, 0.6: 83}

    _best = None
    _most = 0
    for _t_a in _yes_a:
        for _t_b in _yes_b:
            _gap = abs(_yes_a[_t_a] - _yes_b[_t_b])
            if _gap < 0.02:
                _right = _right_a[_t_a] + _right_b[_t_b]
                print(_t_a, "and", _t_b, "| gap", round(_gap, 2), "| right", _right)
                if _right > _most:
                    _most = _right
                    _best = (_t_a, _t_b)
    print("best:", _best, "with", _most, "of 200 right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Two of the nine pairs have a gap below 0.02: 0.4 for a and 0.5 for b, with 165 right, and 0.5 for a and 0.6 for b, with 167 right. The second wins. The pair with both thresholds at 0.5 would get 169 right, but its yes rates are 0.20 and 0.31. Exercise 1 does the same search on UCI Adult, for equal recall.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_search():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The best pair with equal recall

        On the people to train on, `recall_women` holds per threshold the recall of the women, and `right_women` how many women the threshold gets right; `recall_men` and `right_men` the same for the men. `best` starts at `None` and `most` at 0. The loops go through the women's thresholds with `t_women` and the men's with `t_men`. Replace the `...` with the search of the worked example: `gap`, the gap between the two recalls; for a gap below 0.02, `right`, how many the pair gets right; and when `right` is more than `most`, remember it in `most` and the pair in `best`. The check also tries a made-up table.
        """),
        L.task("Find `best`: the pair `(t_women, t_men)` with a recall gap below 0.02 that gets the most people right."),
    ])
    return


@app.cell(hide_code=True)
def exercise_search():
    box_search = L.code_box("search", lines=16)
    box_search
    return (box_search,)


@app.cell(hide_code=True)
def check_search(box_search):
    result_search = L.check("search", box_search.value)
    result_search
    return (result_search,)


@app.cell(hide_code=True)
def hints_search():
    L.hints(
        "search",
        [
            "Do what the worked example did, with the recall instead of the yes rate: for each pair the gap, and only for a gap below 0.02 how many the pair gets right.",
            "`gap = abs(recall_women[t_women] - recall_men[t_men])`, then `if gap < 0.02:` and, indented under it, `right = right_women[t_women] + right_men[t_men]`.",
            "Under `if right > most:` come two lines: `most = right` and `best = ___`.",
        ],
        why="`gap = abs(recall_women[t_women] - recall_men[t_men])`; under `if gap < 0.02:` comes `right = right_women[t_women] + right_men[t_men]`, and under `if right > most:` the lines `most = right` and `best = (t_women, t_men)`. That gives 0.4 for the women and 0.55 for the men.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decide():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The same probability, another answer

        `people` holds four new people, each as a tuple of their sex and the model's probability of a yes. `threshold` holds the thresholds for equal recall of the demo, per sex. The loop goes through the people with `sex` and `score`, and `yes` starts as an empty list. Replace the `...` with one line that adds to `yes` whether `score` is above the threshold of that person's sex. The check also tries other people and other thresholds.
        """),
        L.task("Make `yes`: for each person `True` for a yes and `False` for a no, with the threshold of their own sex."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decide():
    box_decide = L.code_box("decide", lines=9)
    box_decide
    return (box_decide,)


@app.cell(hide_code=True)
def check_decide(box_decide):
    result_decide = L.check("decide", box_decide.value)
    result_decide
    return (result_decide,)


@app.cell(hide_code=True)
def hints_decide():
    L.hints(
        "decide",
        [
            "Each person has the threshold of their own sex, and a yes means the probability is above it.",
            "`threshold[sex]` is the threshold of this person's sex, and `score > threshold[sex]` is `True` for a yes.",
            "`yes.append(score > ___)`.",
        ],
        why="`yes.append(score > threshold[sex])`. That gives `[True, False, False, True]`: the woman and the man with 0.45 get different answers.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_unequal():
    mo.md(r"""
    ### Exercise 3. What equal recall leaves unequal
    """)
    return


@app.cell(hide_code=True)
def choice_unequal_cell():
    choice_unequal = L.choice(
        "unequal",
        question="On UCI Adult, the thresholds for equal recall cost almost no accuracy. What do they leave unequal between women and men?",
        options={
            "nothing": "Nothing: then all the rates are equal",
            "right": "The yes rates and false positive rates",
            "accuracy": "Only how often the model is right overall",
        },
        correct="right",
        why={
            "nothing": "Only the recall became equal. On the test set the yes rates stayed 0.10 against 0.24, and the false positive rates 0.05 against 0.10. See the demo.",
            "right": "With equal recall, 0.10 of the women and 0.24 of the men get a yes, and 0.05 and 0.10 of the people who do not earn more get one anyway. Each criterion leaves another one unequal.",
            "accuracy": "How often the model is right is one number for everyone: 0.832, close to 0.834. What stays unequal between the groups are the yes rates, 0.10 against 0.24, and the false positive rates, 0.05 against 0.10.",
        },
    )
    choice_unequal
    return (choice_unequal,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Fixing the data: reweighing](/course/open/rai-reweighing?lang=en) changed how much each person counts before training. A threshold per group changes the decision after training, and changes no weight of the model: it only needs the model's probabilities and, when deciding, each person's group. Searching for the pair that gets the most people right shows what each criterion costs on UCI Adult: equal recall almost nothing, equal yes rates more, and equal recall and false positive rates together, without a coin, the most. It also shows that each criterion leaves another unequal, as [Equal errors or equal meaning](/course/open/rai-equalized-odds-calibration?lang=en) said it would. The next lesson, *Fairness tools and benchmarks*, shows the libraries that do these computations for you, and the tables on which fairness methods are usually compared.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_unequal, result_decide, result_search):
    L.wrap_up(result_search, result_decide, choice_unequal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
