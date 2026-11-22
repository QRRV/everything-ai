# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-rashomon. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-rashomon", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One short question about [Glass-box models](/course/open/rai-glass-box-models?lang=en) and one about [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_glass():
    quick_glass = L.choice(
        "quick-glass",
        question="Is every decision tree a glass box?",
        options={
            "yes": "Yes: you can always follow its questions",
            "right": "Only a small one: it depends on its size",
            "no": "No: a tree is a black box, like a neural network",
        },
        correct="right",
        why={
            "right": "Right. No kind of model is a glass box by itself: a tree of depth 2 has at most four leaves, which you can follow by hand, but a tree of depth 9 can have 512. In this lesson, trees of every depth from 1 to 10 compete.",
            "yes": "You could follow each path of a deep tree, but there are too many to take in: a tree of depth 9 can have 512 leaves. Whether a person can follow a model depends on its size. See [Glass-box models](/course/open/rai-glass-box-models?lang=en).",
            "no": "A short tree is the textbook glass box: a tree of depth 2 has at most four leaves, and you can follow it by hand. It depends on the size. See [Glass-box models](/course/open/rai-glass-box-models?lang=en).",
        },
    )
    quick_glass
    return (quick_glass,)


@app.cell(hide_code=True)
def quick_check_bootstrap():
    quick_bootstrap = L.choice(
        "quick-bootstrap",
        question="Each tree of a random forest learns from a bootstrap sample of the n training examples. What is that?",
        options={
            "order": "All n examples, shuffled into a new order",
            "half": "A random half, each example at most once",
            "right": "n draws, with replacement",
        },
        correct="right",
        why={
            "right": "Right: n positions drawn at random, each time from all n, so some examples come twice and others not at all. In this lesson every tree gets its own bootstrap sample: slightly different data each time.",
            "order": "The same examples in another order give the same tree. A bootstrap sample draws n times with replacement, so some examples come twice and others not at all. See [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
            "half": "A bootstrap sample has as many draws as there are examples, n, with replacement: some come twice, others not at all. See [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en).",
        },
    )
    quick_bootstrap
    return (quick_bootstrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A bank grows fifty models to decide on loans and picks the one that is right most often; as in [Reasons for a decision](/course/open/rai-reason-codes?lang=en), we read a prediction of earning more than 50,000 dollars as the bank's yes. Thirty-five of the models are right within one percentage point of the best. Take a person aged 36 with a bachelor's degree who works 40 hours a week. In the worked example of this lesson, the best model says no, and 17 of those 35 say yes. Whether this person gets a yes then depends on which of the equally good models the bank happened to pick. Emily Black, Manish Raghavan and Solon Barocas (2022) ask: "why should an individual be subject to an adverse model outcome if there exists an equally accurate model that treats them more favorably?"

    **An everyday example.** Four people who saw the same event tell four different stories, each one believable. The Japanese film *Rashomon* (1950) is built on that, and the statistician Leo Breiman named the effect of this lesson after it: data that many different models fit about equally well.
    """)
    return


@app.cell(hide_code=True)
def predict_fifty():
    guess_fifty = L.guess(
        "fifty",
        question="In the worked example, 35 trees are right within one point of the best tree. For what share of the 24,421 people they are tested on does at least one of the 35 give another answer than the best tree?",
        options={
            "few": "About 1 in 100",
            "some": "About 1 in 20",
            "many": "About 1 in 5",
        },
        reveal="**About 1 in 5: 20.6 percent**, 5,035 people. Yet each of the 35 trees is right for between 78.9 and 79.8 percent of them. Trees that are about equally good can still disagree about many people: each makes its mistakes on partly other people.",
    )
    guess_fifty
    return (guess_fifty,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The Rashomon effect.** Leo Breiman (2001) draws three lessons, the first "Rashomon: the multiplicity of good models". Breiman explains: "What I call the Rashomon Effect is that there is often a multitude of different descriptions [equations f(x)] in a class of functions giving about the same minimum error rate." In one of Breiman's examples, many equations with five variables fit about equally well: "Which one is better? The problem is that each one tells a different story about which variables are important." Trees do it too: "if the training set is perturbed only slightly, say by removing a random 2–3% of the data, I can get a tree quite different from the original but with almost the same test set error." Breiman concludes: "The multiplicity problem and its effect on conclusions drawn from models needs serious attention."

        **Different answers for the same people.** Breiman's point is about different stories. Charles Marx, Flavio du Pin Calmon and Berk Ustun (2020) look at the people: "we define predictive multiplicity as the ability of a prediction problem to admit competing models with conflicting predictions". The competing models are all models of one kind whose error is at most ε above that of the best one; they call this the ε-level set, others the Rashomon set. They measure two numbers, on the training data: "Ambiguity: How many individuals are assigned conflicting predictions by any competing model? Discrepancy: What is the maximum number of predictions that could change if we were to switch the model that we deploy with a competing model?" On COMPAS data about who was arrested again, they found that "a competing model that is only 1% less accurate than the most accurate model assigns conflicting predictions to over 17% of individuals, and that the predictions of 44% of individuals are affected by model choice." It is not the same for everyone: in their data on arrests for a violent crime, 72.9 percent of the African-American people could get another prediction within that 1 percent, against 37.2 percent of the white people.

        **Searching wider finds more.** Those numbers come from a search over all linear models within the margin. Retraining many models, as this lesson does, finds fewer: on the same data, "an ad hoc analysis reveals an ambiguity of 10% and a discrepancy of 7% over the set of competing models. These estimates are far less than those produced using our tools (44% and 17% respectively)." Such approaches "can detect predictive multiplicity, but should not be used to certify the absence of multiplicity."

        **Model multiplicity.** Black, Raghavan and Barocas call the whole of it model multiplicity: equally accurate models can differ "in their internals, or procedural multiplicity, and in their predictions, or predictive multiplicity." They measure on the labeled data a developer has, which may be test data, as in this lesson. They see a problem: "choosing a model based on accuracy alone is akin to choosing arbitrarily between more or less favorable predictions for certain individuals." And a chance: among the equally good models, a developer can look for ones that "differ in the degree to which they rely on features known to be difficult or impossible for people to change."

        **Rudin: pick one you can read.** Cynthia Rudin (2019) turns the many good models into an argument: "Consider that the data permit a large set of reasonably accurate predictive models to exist. Because this set of accurate models is large, it often contains at least one model that is interpretable. This model is thus both interpretable and accurate." So "domain experts may have more flexibility in choosing a model that they find interpretable". Lesia Semenova, Cynthia Rudin and Ronald Parr (2022) put it as a hypothesis: "If, as we hypothesize in this work, many real-world data sets admit large Rashomon sets, the implications are vast: it means that simple or interpretable models may often be used for high-stakes decisions without losing accuracy." Note the "often" and the "may": Rudin speaks of "structured data with meaningful features", where for most problems "machine learning algorithms tend to perform similarly".

        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **Rashomon effect** | there are often many different models that are about equally accurate, and each tells its own story, such as which features matter (Breiman) |
        | **Rashomon set** | all models of one kind that are no more than ε worse than the best one; Marx and colleagues call it the ε-level set |
        | **predictive multiplicity** | when models of the Rashomon set give different answers for the same people |
        | **ambiguity** | the share of people for whom at least one model of the set gives another answer than the chosen model |
        | **discrepancy** | the largest share of people whose answer changes when you switch to one other model of the set |
        | ε | how much worse than the best a model may be and still count; in this lesson 0.01, one percentage point |
        """),
        L.watch_out("Having many good models is not good news for everyone. Rudin sees room in it to choose a model you can read, notes that it means a learning method \"may not be stable\", and adds: \"Note that not all researchers working in interpretability agree with this general sentiment about the advantages of instability\". And how many people can get another answer is no fixed number: it depends on the kind of model, the data, ε, and how widely you search."),
    ])
    return


@app.cell(hide_code=True)
def explore_eps_cell():
    explore_eps = L.explore("eps", "ε in points", lambda e: C.eps_picture(e, "en"), start=1, values=C.STOPS,
                            role="hyper",
                            task="Move ε from 0 to 5 points. How many trees join the set at each step, and what happens to the share of people who can get another answer?")
    mo.vstack([
        mo.md("**The trees within ε of the best**"),
        explore_eps,
        mo.md(r"""
        The 50 trees of the worked example, each grown on its own bootstrap sample of one half of UCI Adult and tested on the other half. ε is how far below the best a tree may lie and still count as good. Each tree that joins can give some people another answer than the best tree.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure how much equally good models disagree:

    1. Split the data into a part to train on and a part to test on.
    2. Grow many models of one kind, each on its own bootstrap sample of the training part, and keep what each says for every test person and how often it is right.
    3. Take the best model, and the set of models no more than ε below it.
    4. Ambiguity: the share of test people for whom at least one model of the set says something other than the best. Discrepancy: the largest share that switching to one model of the set changes.
    5. Look in the set for a model you can read, and compare it with the best before you choose.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The three columns of [Reasons for a decision](/course/open/rai-reason-codes?lang=en), as in the lessons before. `np.random.default_rng(0).permutation(n)` puts the positions 0 to n − 1 in a random order; the first half is the training half, the rest the test half; `//` divides and keeps the whole part, so `len(_y) // 2` is half the number of people, rounded down. Then trees of depth 1 to 10, five of each, each on its own bootstrap sample of the training half. `np.array` turns the list of answers into a table with one row per tree and one column per test person. `_answers[_good]` keeps the rows of the trees in the set, and `!=` compares each of them with the row of the best tree. `.any(axis=0)` asks, for each column, whether any row is `True`: whether at least one tree gives this person another answer. `.mean(axis=1)` gives the share per tree. `np.argmax` on a row of `True` and `False` gives the position of the first `True`: the smallest tree in the set, because the trees are grown from shallow to deep; `export_text` prints it, as in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en). Last comes a random forest on the same training half, for comparison: `n_estimators=100` is its number of trees, and `min_samples_leaf=20` makes each leaf hold at least 20 people. Run the cell. It takes a few seconds.
    """)
    return


@app.cell
def example_trees():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 1. 48,842 people from a US survey of 1994 (UCI Adult), in two halves
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]

    # 2. fifty trees: depth 1 to 10, five of each, each on its own bootstrap sample of the training half
    _rng = np.random.default_rng(1)
    _trees, _answers, _right, _depths = [], [], [], []
    for _depth in range(1, 11):
        for _k in range(5):
            _rows = _train[_rng.integers(0, len(_train), size=len(_train))]
            _tree = DecisionTreeClassifier(max_depth=_depth, random_state=0).fit(_X[_rows], _y[_rows])
            _trees.append(_tree)
            _answers.append(_tree.predict(_X[_test]))
            _right.append(np.mean(_answers[-1] == _y[_test]))
            _depths.append(_depth)
    _answers, _right, _depths = np.array(_answers), np.array(_right), np.array(_depths)

    # 3. the best tree, and the set of trees no more than one point below it
    _best = np.argmax(_right)
    _good = _right >= _right[_best] - 0.01
    print("best: depth", _depths[_best], "right", _right[_best].round(4))
    print("in the set:", _good.sum(), "trees, depth", _depths[_good].min(), "to", _depths[_good].max())

    # 4. who gets another answer from a tree in the set than from the best tree
    _differ = _answers[_good] != _answers[_best]
    print("another answer from at least one:", _differ.any(axis=0).mean().round(3))
    print("most changed by one tree:", _differ.mean(axis=1).max().round(3))

    # 5. the smallest tree in the set
    print(export_text(_trees[np.argmax(_good)], feature_names=_names))

    # for comparison: a random forest of 100 trees on the same training half
    _forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0)
    _forest.fit(_X[_train], _y[_train])
    print("forest right", np.mean(_forest.predict(_X[_test]) == _y[_test]).round(4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The best tree has depth 5 and is right for 79.8 percent of the test half; always saying no is right for 75.9 percent. 35 of the 50 trees lie no more than a point below the best, from depth 2 to depth 9. Yet for 20.6 percent of the 24,421 test people, at least one of them gives another answer than the best tree, and switching to just one of them changes the answer for up to 6.9 percent. The trees also tell different stories, as Breiman found: the three trees of depth 2 in the set ask only about education and age, never about hours, while the other 32 ask about hours too. The smallest, printed above, says yes only to people with education above 12.5 who are older than 33.5, and is right for 78.9 percent. The random forest of 100 trees is right for 79.8 percent: no better than the best single tree.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ambiguity():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Ambiguity by hand

        Write `ambiguity(base, others)`: the share of people for whom at least one of the other models says something other than the chosen one. `base` holds what the chosen model says for each person, and `others` holds such a list for each other model. The check first calls `ambiguity([1, 0, 0, 1, 0], [[1, 1, 0, 1, 0], [0, 1, 0, 1, 0]])`: five people and two other models. Then it tries other people and models; the picture runs your function on the trees of the worked example.
        """),
        L.task("Write `ambiguity(base, others)`: count each person once if at least one list in `others` differs from `base` there, and return the share."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ambiguity():
    box_ambiguity = L.code_box("ambiguity", lines=12)
    box_ambiguity
    return (box_ambiguity,)


@app.cell(hide_code=True)
def check_ambiguity(box_ambiguity):
    result_ambiguity = L.check("ambiguity", box_ambiguity.value)
    result_ambiguity
    return (result_ambiguity,)


@app.cell(hide_code=True)
def hints_ambiguity():
    L.hints(
        "ambiguity",
        [
            "Go over the people. For each person, find out whether at least one model says something other than `base`, and if so, count the person once. At the end, divide by the number of people.",
            "Loop over the people with `for i in range(len(base)):`. For each person, start with `differs = False`, loop over the models with `for answers in others:`, and set `differs = True` when `answers[i] != base[i]`. After that inner loop, add 1 to `conflicts`, which starts at 0, if `differs` is `True`.",
            "`return conflicts / ___`",
        ],
        why="`conflicts = 0`, then `for i in range(len(base)):` with `differs = False`, an inner loop `for answers in others:` that sets `differs = True` when `answers[i] != base[i]`, and after it `if differs:` with `conflicts = conflicts + 1`. After both loops, `return conflicts / len(base)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_good():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The Rashomon set

        The box holds how often each of the 50 trees of the worked example is right on the test half, rounded, and the depth of each. Fill in the two `...`: `best`, the highest share in `right`, and inside the loop, the position `k` added to `good` when tree `k` lies no more than `eps` below the best. The check also tries another `eps`; the picture shows your set.
        """),
        L.task("Replace the two `...`: the highest share as `best`, and `good.append(k)` for each tree no more than `eps` below it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_good():
    box_good = L.code_box("good", lines=15)
    box_good
    return (box_good,)


@app.cell(hide_code=True)
def check_good(box_good):
    result_good = L.check("good", box_good.value)
    result_good
    return (result_good,)


@app.cell(hide_code=True)
def hints_good():
    L.hints(
        "good",
        [
            "A tree belongs to the set when it is right at most `eps` less often than the best one. First find the best share, then go over the trees and keep the positions of those that pass.",
            "`max(right)` gives the highest share. A tree passes when its share is at least `best` minus `eps`; add its position with `good.append(k)`.",
            "`if right[k] >= best - ___:`",
        ],
        why="`best = max(right)`, and inside the loop `if right[k] >= best - eps:` with `good.append(k)` under it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ### Exercise 3. Which tree?
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick",
        question="Your set holds 35 trees, from depth 2 to depth 9, all within a point of the best. For decisions about loans, which tree would Rudin choose?",
        options={
            "best": "The best one, of depth 5, since it is right most often",
            "right": "A tree of depth 2, which people can read whole",
            "vote": "All 35, with a vote",
        },
        correct="right",
        why={
            "best": "Its lead is less than a point on the test half, and it has 32 leaves. Rudin's argument: when the set of good models is large, it often holds one you can read, and for high-stakes decisions that is the one to take. See *Rudin: pick one you can read* in *The idea*.",
            "right": "Right. The trees of depth 2 are in the set: within a point of the best, and you can read one whole, four leaves, as in the worked example. That is Rudin's argument at work. Whether it is good enough remains a choice for the people who use it.",
            "vote": "A vote of trees on bootstrap samples is bagging, the heart of a random forest: the forest of the worked example is about as accurate as the best tree, and much harder to follow. For high-stakes decisions Rudin argues for the model in the set that you can read. See *Rudin: pick one you can read* in *The idea*.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_sure():
    mo.md(r"""
    ### Exercise 4. What the number says
    """)
    return


@app.cell(hide_code=True)
def choice_sure_cell():
    choice_sure = L.choice(
        "sure",
        question="With the 35 trees, ambiguity is 20.6 percent: that many test people can get another answer from one of them than from the best tree. What does that number tell you?",
        options={
            "right": "At least that many; a wider search may find more",
            "exact": "Exactly how many it affects",
            "most": "That no more people than that can get another answer",
        },
        correct="right",
        why={
            "right": "Right. It is what these 35 trees show: a lower bound. Marx and colleagues found 10 percent by retraining where their search over all good linear models found 44 percent, and warn that such approaches \"should not be used to certify the absence of multiplicity\". Other kinds of models or other samples can give still more people another answer within the same ε.",
            "exact": "It counts only the 35 trees grown here. Other good models, of another kind or from other samples, can give still more people another answer. See *Searching wider finds more* in *The idea*.",
            "most": "The other way around: it is a lower bound. In the work of Marx and colleagues, retraining found 10 percent where a search over all good linear models found 44 percent. See *Searching wider finds more* in *The idea*.",
        },
    )
    choice_sure
    return (choice_sure,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Glass-box models](/course/open/rai-glass-box-models?lang=en) a tree of depth 2 was a model you can follow by hand; here such a tree lies within a point of the best of 50 trees, which shows Rudin's answer to the question of that lesson: why a simple model can often predict about as well as a complicated one. The trees grew on bootstrap samples, as in [Many trees vote: the random forest](/course/open/ml-random-forest?lang=en), where the vote of many different trees made the answer steadier. The next lesson, [Which features matter?](/course/open/rai-permutation-importance?lang=en), measures how much a model leans on each feature; keep Breiman's warning in mind there: equally good models can tell different stories about which features matter.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, choice_sure, result_ambiguity, result_good):
    L.wrap_up(result_ambiguity, result_good, choice_pick, choice_sure)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
