# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-automl. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-automl", "en")
    C = L.checks   # the 27 candidates, their scores on 778, 2,333 and 7,000 runs, measured once, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Searching settings: grid search and random search](/course/open/ml-grid-random-search?lang=en) and one about [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grid():
    quick_grid = L.choice(
        "quick-grid",
        question="A grid search tries 3 values of each of 4 settings. How many tries is that?",
        options={
            "sum": "12",
            "right": "81",
            "square": "16",
        },
        correct="right",
        why={
            "sum": "12 is 3 + 3 + 3 + 3. A grid tries every combination: every value of the first setting with every value of the second, and so on, so the numbers multiply. See *The idea* in [Searching settings: grid search and random search](/course/open/ml-grid-random-search?lang=en).",
            "square": "16 is 4 · 4. A grid tries every combination of the 3 values of every setting: $3 \\cdot 3 \\cdot 3 \\cdot 3$. See *The idea* in [Searching settings: grid search and random search](/course/open/ml-grid-random-search?lang=en).",
            "right": "$3 \\cdot 3 \\cdot 3 \\cdot 3 = 81$, and each is a whole cross-validation. Today a search spends less on the tries that are clearly bad.",
        },
    )
    quick_grid
    return (quick_grid,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="`RandomForestClassifier(n_estimators=50, max_features=3)` on data with 5 features. Which is true?",
        options={
            "same": "All 50 trees learn from the same examples",
            "all": "Every split looks at all 5 features and takes the best",
            "right": "Every tree learns from its own bootstrap sample, and every split looks at 3 features drawn at random",
        },
        correct="right",
        why={
            "same": "Every tree learns from its own bootstrap sample, n examples drawn with replacement, so the trees differ. See *The idea* in [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "all": "`max_features=3` lets every split look at 3 features drawn at random, so the trees differ. See the new words of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "right": "Right. Today the number of features per split and the depth of the trees are two settings a search chooses, next to k and p of k-NN.",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the previous lesson you searched the settings of one model, k-NN. But often you do not know which model fits best either: k-NN, a random forest, or something else. Let a search choose the model as well, and the candidates add up quickly. Scoring 27 candidates by cross-validation on all 7,000 training runs takes 27 whole cross-validations, and most candidates turn out to be clearly worse than the best after a quick look. Can a search spend less on those?

    **An everyday example.** A company has 27 applicants for one job. A trial day with every one of them would take 27 days. So it calls all 27 for a few minutes, invites the best 9 for an interview, and gives the best 3 a trial day. The long tests go to the few most likely to get the job. The risk: someone who would shine on a trial day but is nervous on the phone never gets one.
    """)
    return


@app.cell(hide_code=True)
def predict_budget_cell():
    guess_budget = L.guess(
        "budget-guess",
        question="27 candidates. Round 1 gives each 778 runs, round 2 gives the best 9 each 2,333, and round 3 gives the best 3 all 7,000. Trying all 27 on all 7,000 hands out 27 · 7,000 = 189,000 runs. How many do the three rounds hand out?",
        options={
            "half": "About half: 95,000",
            "right": "About a third: 63,000",
            "tenth": "About a tenth: 19,000",
        },
        reveal="**About a third.** Round 1 hands out 27 · 778 = 21,006 runs, round 2 9 · 2,333 = 20,997 and round 3 3 · 7,000 = 21,000: every round costs about the same, 63,003 in all.",
    )
    guess_budget
    return (guess_budget,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **AutoML.** Building a model takes choices: how to prepare the data, which model, and its hyperparameters. **AutoML** lets a tool make those choices: it searches models and their settings by itself. Here the candidates are 27 drawn at random, as in a random search: a coin flip chooses k-NN, with k and p as in [Searching settings: grid search and random search](/course/open/ml-grid-random-search?lang=en), or a random forest of 50 trees as in [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), with a maximum depth as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en) from 2 to 30, drawn on a log scale, and 1 to 5 features per split. The draw gave 11 k-NN models and 16 forests, two of them the same: a random draw can repeat itself.

        **A budget.** In daily life a budget is money. Here it is what a search may spend: a number of tries, an amount of time, or, as in this lesson, training runs handed out. A run is still one machine run, a row of AI4I 2020: a candidate scored on 778 runs costs 778; on all 7,000 runs, 7,000. Scoring all 27 on all 7,000 costs $27 \cdot 7{,}000 = 189{,}000$.

        **Successive halving.** Kevin Jamieson and Ameet Talwalkar (2016) used it to search settings: start every candidate with a small part of the budget, keep the best part, give them more, and repeat until one candidate is left. Here every round keeps the best third and gives them 3 times as many runs: 27 candidates on the first 778 training runs, the best 9 on 2,333, the best 3 on all 7,000. Every score is the mean AUC of 5 folds, as in the previous lesson. Every round costs about 21,000 runs, three rounds about 63,000: a third of 189,000. The name comes from the version that keeps the best half.

        **The risk.** A few runs give a rough score. Among the first 778 training runs are only 29 failures, so each fold, about 155 runs, holds between 3 and 9 of them, and the AUC of the fold rests on those few. A candidate that needs many examples to show what it can do, such as a forest whose splits look at 4 or 5 features, can look bad in round 1 and drop out before it gets more.

        **The tools.** Optuna (Takuya Akiba and others, 2019) chooses every new try near earlier good tries and can stop a try early when its first scores are bad. auto-sklearn (Matthias Feurer and others, 2015) also chooses every new try from the scores so far, starts with settings that worked on similar datasets, and combines its best models into an ensemble. AutoGluon (Nick Erickson and others, 2020) trains many kinds of models with good default settings and combines them. scikit-learn has successive halving itself, still marked experimental: `HalvingRandomSearchCV`.

        **What stays yours.** Every tool needs a training part and a test set it never sees, a metric that fits the problem, such as the AUC for rare failures, and a budget. A tool chooses well only by the measure you give it.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **AutoML** | letting a tool choose the model, its settings and how to prepare the data, by searching within a budget | |
        | **budget** | what a search may spend: tries, time, or training runs handed out; not money, as in daily life | |
        | **successive halving** | start every candidate with a small budget, keep the best part, give them more, and repeat until one is left | `HalvingRandomSearchCV` |
        | **candidate** | one model with its settings that a search tries | `("forest", 5, 3)` |
        """),
        L.watch_out("A search with a smaller budget is not free: the first round must be large enough to tell good from bad. Too few runs, and its scores are mostly chance."),
        mo.md(r"""
        The slider goes through the rounds. Left, the AUC of every candidate in the rounds so far, against the training runs it got, on a log scale: a filled dot goes on, an open dot drops out, and a gray line follows a candidate to the next round. At round 3 the small gray dots are the scores of all 27 on all 7,000 runs, with the best of them ringed. Right, the budget spent so far, in thousands of runs, against all 27 on all 7,000. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_round_cell():
    def _frame(r):
        said = C.round_said(r, L.lang)
        alt = ("Left: the AUC of the candidates of successive halving, against the training runs each got, on a log scale "
               "of 778, 2,333 and 7,000 runs. Right: two bars, the runs handed out so far and the 189,000 of all 27 "
               "candidates on all 7,000 runs. " + said)
        labels = {"rounds": "AUC of every candidate", "runs": "training runs", "budget": "Budget",
                  "thousands": "runs (thousands)", "halving": "halving", "full": "all 27", "alt": alt}
        return (C.round_picture(r, labels, L.lang), said)

    explore_round = L.explore("round", "round", _frame, start=1, values=[1, 2, 3], role="",
                              task="Go through the three rounds. Which candidates go on after round 1, and which candidate would win on all 7,000 runs?")
    explore_round
    return


@app.cell(hide_code=True)
def text_after_round():
    mo.md(r"""
    Successive halving spent 63,003 runs, a third of 189,000. In time it saved less: on the course's computer it took about 14 seconds against 27 for all 27 on all 7,000, because a fit on a few runs is not that much faster. Its winner, a random forest with depth 5 and 3 features per split, has AUC 0.961 in cross-validation. All 27 on all 7,000 runs find a deeper forest, with depth 12 and 4 features per split, and 0.969. On 778 runs that forest scored 0.898, place 21 of 27, and successive halving dropped it in round 1. Measured once on the 3,000 test runs, the two winners are about equal: 0.973 against 0.974.

    ### Recipe

    To search models and their settings with successive halving:

    1. Split off a test set and choose a metric that fits the problem.
    2. Choose the candidates: models and a range for every setting, drawn at random as in a random search.
    3. Choose the budget, the factor and the rounds: with factor 3 and 3 rounds, round r (counted from 0) keeps $27 / 3^r$ candidates and gives each $7{,}000 / 3^{2-r}$ runs.
    4. Score every candidate of a round by cross-validation on the runs of that round, keep the best third, and go to the next round.
    5. Check that the first round is large enough: if its scores are mostly chance, start with fewer candidates and more runs.
    6. Fit the winner on the whole training part and measure it once on the test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads and splits the runs as in [When one class is rare](/course/open/ml-imbalanced?lang=en), and draws the 27 candidates. `_model(c)` makes the model of candidate `c`, and `_score(c, runs)` gives its mean AUC of 5 folds on the first `runs` training runs. Then come the three rounds, with $7{,}000 / 3^{2-r}$ runs in round r, rounded: 778, 2,333 and 7,000. In every round: score every candidate, keep the best third with `np.argsort`, as in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), and print the runs, the number of candidates and the best of the round with its AUC. On the course's computer the cell takes about 15 seconds. Run the cell.
    """)
    return


@app.cell
def example_halving():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _numbers = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    _X = np.array(_df[_numbers])
    _y = np.array(_df["Machine failure"])
    _order = np.random.default_rng(0).permutation(len(_y))
    _train = _order[3000:]                                 # the 7,000 training runs; the 3,000 test runs wait

    def _model(c):
        if c[0] == "neighbors":
            return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=c[1], p=c[2]))
        return RandomForestClassifier(n_estimators=50, max_depth=c[1], max_features=c[2], random_state=0)

    def _score(c, runs):
        _part = _train[:runs]                              # the first `runs` training runs
        return cross_val_score(_model(c), _X[_part], _y[_part], cv=KFold(5), scoring="roc_auc").mean()

    _rng = np.random.default_rng(0)
    _candidates = []
    for _ in range(27):
        if _rng.random() < 0.5:                            # a coin flip: k-NN or a random forest
            _candidates.append(("neighbors", int(10 ** _rng.uniform(0, np.log10(300))), round(_rng.uniform(1, 4), 2)))
        else:
            _candidates.append(("forest", int(10 ** _rng.uniform(np.log10(2), np.log10(30))), int(_rng.integers(1, 6))))

    _rounds = []
    for _r in range(3):
        _rounds.append(round(7000 / 3 ** (2 - _r)))        # 778, 2,333 and 7,000 runs
    for _runs in _rounds:
        _scores = []
        for _c in _candidates:
            _scores.append(_score(_c, _runs))
        _i = np.argmax(_scores)
        print(_runs, len(_candidates), _candidates[_i], round(_scores[_i], 3))
        _keep = round(len(_candidates) / 3)
        _kept = []
        for _j in np.argsort(_scores)[-_keep:]:             # the positions of the best third
            _kept.append(_candidates[_j])
        _candidates = _kept
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Round 1 scores all 27 on 778 runs; the best is a forest with depth 25 and 2 features per split, with 0.965. Round 2 scores the best 9 on 2,333 runs; the best is a forest with depth 5 and 2 features, with 0.955. Round 3 scores the best 3 on all 7,000 runs, and the forest with depth 5 and 3 features wins, with 0.961. The best of one round is not always the best of the next: with more runs the order changes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The plan of the rounds

        `plan(n, total, factor, rounds)` should give, for every round, how many candidates are left and how many runs each gets, as a list of pairs `(candidates, runs)`. Round `r` counts from 0: it keeps `n` divided by `factor` once for every round before it, and the last round gives each candidate all `total` runs, every earlier round `factor` times fewer than the next. Replace the two `...`; round both to a whole number with `round`. The check calls `plan(27, 7000, 3, 3)`, which should give `[(27, 778), (9, 2333), (3, 7000)]`, and three more plans.
        """),
        L.task("Let `plan` give the candidates and the runs of every round."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plan():
    box_plan = L.code_box("plan", lines=7)
    box_plan
    return (box_plan,)


@app.cell(hide_code=True)
def check_plan(box_plan):
    result_plan = L.check("plan", box_plan.value)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan",
        [
            "Write out the plan of the lesson: in round 0, 1 and 2, how many times has 27 been divided by 3, and how many times is 7,000 divided by 3?",
            "The candidates: `n` divided by `factor` once for every round before round `r`. How many is that, and how do you write 'that many times by `factor`' with `**`? The runs: `total` divided by `factor` once for every round still to come after round `r`; in the lesson's plan, round 0 has two after it.",
            "`candidates = round(n / factor ** ___)` and `runs = round(total / factor ** (___ - 1 - ___))`.",
        ],
        why="Round r keeps $27 / 3^r$ candidates: 27, 9, 3. Each gets $7{,}000 / 3^{2-r}$ runs: 778, 2,333, 7,000. Candidates times runs is about 21,000 in every round, so the rounds cost the same, and three of them a third of the full search.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_halving():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Successive halving

        `successive_halving(candidates, score, rounds)` gets a list of candidates, a function `score(c, runs)` that gives the score of candidate `c` on `runs` runs, and the runs of every round. A function is a value too: `score` without parentheses is the function itself, so it can be handed to yours, and inside it `score(c, runs)` calls it. In every round it should score every candidate on the runs of that round, keep the best `keep` and go on with them; at the end it gives back what is left. Replace the two `...`. The check calls it with the 27 candidates of the lesson, `score` from the measured scores and `rounds = [778, 2333, 7000]`, and with three more lists, such as 81 numbers in four rounds.
        """),
        L.task("Let `successive_halving` keep the best third of every round."),
    ])
    return


@app.cell(hide_code=True)
def exercise_halving():
    box_halving = L.code_box("halving", lines=14)
    box_halving
    return (box_halving,)


@app.cell(hide_code=True)
def check_halving(box_halving):
    result_halving = L.check("halving", box_halving.value)
    result_halving
    return (result_halving,)


@app.cell(hide_code=True)
def hints_halving():
    L.hints(
        "halving",
        [
            "Two things: the score of `c` in this round, and which positions belong to the best `keep` scores. The function that gives a score is the `score` your function gets.",
            "Call `score` with the candidate and the runs of this round. `np.argsort(scores)` gives the positions from the lowest score to the highest; the best `keep` are at the end, and a negative slice takes the end of a list.",
            "`scores.append(score(___, ___))` and `best = np.argsort(scores)[-___:]`.",
        ],
        why="`np.argsort(scores)[-keep:]` gives the positions of the `keep` highest scores. Every round keeps a third, so 27 candidates become 9, then 3, then 1, and the later rounds give the few that are left the most runs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.md(r"""
    ### Exercise 3. 243 candidates
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small-first",
        question="You try 243 candidates with the same plan: keep a third every round and end with all 7,000 runs. That takes 5 rounds, so round 1 gives every candidate 86 runs, with 3 failures among them. What goes wrong?",
        options={
            "budget": "The search costs more than trying all 243 on all 7,000 runs",
            "more": "Nothing: more candidates always find a better winner",
            "right": "Round 1 measures mostly chance, so good candidates can drop out at once",
        },
        correct="right",
        why={
            "budget": "Every one of the 5 rounds hands out about 21,000 runs, about 105,000 in all, against $243 \\cdot 7{,}000 = 1{,}701{,}000$. The trouble is in round 1: of its 5 folds, 3 hold no failure at all, so there is no AUC to measure in them, and the others rest on 1 or 2 failures.",
            "more": "More candidates only help when round 1 can tell good from bad. With 3 failures among 86 runs, 3 of the 5 folds hold no failure at all: there is no AUC to measure in them, and the others rest on 1 or 2 failures.",
            "right": "Right: of the 5 folds of round 1, 3 hold no failure at all, and the others rest on 1 or 2. Even the 778 runs of this lesson dropped the forest that is best on all 7,000. Hyperband (Lisha Li and others, 2018) therefore runs successive halving several times, each time with fewer candidates and more runs to start with.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The candidates are drawn as in the random search of [Searching settings: grid search and random search](/course/open/ml-grid-random-search?lang=en), and every score is a cross-validation as in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en), with the AUC of [Every threshold at once: the ROC curve and AUC](/course/open/ml-roc-auc?lang=en) because failures are rare, as in [When one class is rare](/course/open/ml-imbalanced?lang=en). Half of the candidates are the random forests of [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), and their maximum depth is that of [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en). Successive halving decides by the measure you give it, so a fair split and a fitting metric stay your work, whatever tool searches.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_halving, result_plan):
    L.wrap_up(result_plan, result_halving, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
