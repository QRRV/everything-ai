# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-no-free-lunch. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-no-free-lunch", "en")
    C = L.checks   # five made-up datasets, the four models of the module with their scores, and the tiny world


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en) and [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tree():
    quick_tree = L.choice(
        "quick-tree",
        question="A decision tree of depth 2 on two features splits the plane into parts. What do the edges of those parts look like?",
        options={
            "right": "Straight pieces, each parallel to one of the axes",
            "diagonal": "Straight lines in any direction",
            "curves": "Curves that bend around the examples",
        },
        correct="right",
        why={
            "diagonal": "Every question of a tree looks at one feature and one threshold, so every edge runs straight across that one feature. See *The idea* in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
            "curves": "A tree only asks threshold questions on one feature at a time; its edges are straight, and parallel to the axes. See *The idea* in [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en).",
            "right": "One feature and one threshold per question: the parts are boxes. Today that turns out to be an assumption about the data.",
        },
    )
    quick_tree
    return (quick_tree,)


@app.cell(hide_code=True)
def quick_check_cv():
    quick_cv = L.choice(
        "quick-cv",
        question="In 5-fold cross-validation, how often is each training example in the part that is scored?",
        options={
            "five": "Five times, once for every fold",
            "right": "Once",
            "never": "Never: it is only used to train",
        },
        correct="right",
        why={
            "five": "Every fold is scored once, and every example is in exactly one fold: in the other four rounds it is trained on. See *The idea* in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "never": "Each fold takes its turn as the scored part, so every example is scored once, by a model that did not train on it. See *The idea* in [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en).",
            "right": "Once, by a model that did not see it. Today you compare four models in that way on five datasets.",
        },
    )
    quick_cv
    return (quick_cv,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You have now met four classifiers that each draw a different kind of boundary: k-NN, the nearest centroid, the widest street and the decision tree. Which one is the best? It would save a lot of work to know that once and for all. Try them on a few made-up datasets, though, and no single one comes out on top. That is not bad luck: it is a theorem.

    **An everyday example.** Winter tires grip best on snow and wear quickly on a hot road; summer tires the other way around. No tire is best on every road, because what makes a tire good on one road is what makes it worse on another.
    """)
    return


@app.cell(hide_code=True)
def predict_winner():
    guess_winner = L.guess(
        "winner",
        question="Four made-up datasets with two features: two round clumps, two classes on either side of a straight line, a clump inside a ring, and blocks. Which of the four models is best on all four?",
        options={
            "knn": "k-NN: it can draw any boundary",
            "tree": "The decision tree: it can cut the plane into as many boxes as it needs",
            "right": "None of them",
        },
        reveal="**None.** With 5-fold cross-validation, the nearest centroid is best on the two round clumps (0.823), the widest street on the straight line (0.997), k-NN on the ring (1.000) and the decision tree on the blocks (0.993). You can see all of them below.",
    )
    guess_winner
    return (guess_winner,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **What a model assumes.** Training examples only say what is right where they lie. Everywhere else a model has to guess, and how it guesses is what it assumes about the data:

    - k-NN: an example has the class of the examples near it;
    - the nearest centroid: every class is one round clump;
    - the widest street: a straight line separates the classes;
    - a decision tree: thresholds on one feature at a time separate the classes.

    This is called the model's inductive bias. Where the assumption holds, a model learns a lot from few examples; where it does not, more examples help it little.

    **No free lunch.** Is there a model that assumes nothing, and so wins everywhere? Take a tiny world: three features that are each 0 or 1, so 8 possible examples. The training set shows the label of 5 of them; 3 stay unseen.

    | example | feature 1 | feature 2 | feature 3 | label |
    |---|---|---|---|---|
    | 1 | 0 | 0 | 0 | 0 |
    | 2 | 0 | 0 | 1 | 0 |
    | 3 | 0 | 1 | 0 | 0 |
    | 4 | 1 | 0 | 0 | 1 |
    | 5 | 1 | 1 | 0 | 1 |
    | 6 | 0 | 1 | 1 | ? |
    | 7 | 1 | 0 | 1 | ? |
    | 8 | 1 | 1 | 1 | ? |

    Each unseen example can be 0 or 1: that gives $2 \cdot 2 \cdot 2 = 8$ worlds, and every one of them fits the training set perfectly. Whatever a rule says for the 3 unseen examples, for each of them half of the worlds agree and half do not. So averaged over the 8 worlds every rule gets exactly half of them right: the best model as well as a coin. That is the **no free lunch theorem**: averaged over all possible problems, no learning method does better on unseen examples than any other. A model wins only on problems that fit its assumption, and the problems we meet are not all possible problems: they have structure.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **no free lunch theorem** | averaged over all possible problems, no learning method does better on unseen examples than any other | |
    | **world** (in this lesson) | one way to label the unseen examples that fits the training set | `[0, 1, 1]` |
    | `+` between lists | glues two lists into a new one: `[0, 1] + [1]` is `[0, 1, 1]`, and `[0, 1]` stays as it was | `world + [0]` |
    """)
    return


@app.cell(hide_code=True)
def text_watch_out():
    L.watch_out("No free lunch does not say that all models are equally good on your problem. It says the opposite: which model wins depends on the problem, so you have to compare them on your own data, with cross-validation, instead of trusting a favorite.")
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Pick a dataset with the slider: 1 two round clumps, 2 a straight line, 3 a ring, 4 blocks, 5 coin flips, where the class has nothing to do with the features. Each has 300 made-up examples with two features, class 0 filled and class 1 open. The four charts show the <span class="c-pred">boundary</span> each model draws after training on all 300, with above it the model's accuracy with 5-fold cross-validation; ★ marks the best. k-NN looks at 5 neighbors, the widest street has C = 1 and the tree a depth of at most 4.
    """)
    return


@app.cell(hide_code=True)
def explore_dataset_cell():
    def _frame(_v):
        _d = C.DATASETS[_v - 1]
        _s = C.scores()[_d]
        _alt = (f"Dataset {_v}, {C.NAMES[_d][L.lang]}, four times, each with the boundary of one model. Accuracy with "
                f"cross-validation: " + ", ".join(f"{C.MODEL_NAMES[_m][L.lang]} {C.number(_s[_m], L.lang)}" for _m in C.MODELS) + ".")
        _said = C.verdict(_v, L.lang)
        return C.models_picture(_d, {"x": "feature 1", "y": "feature 2", "alt": _alt}, L.lang), _said

    explore_dataset = L.explore("dataset", "dataset", _frame, start=1, values=[1, 2, 3, 4, 5], role="data",
                                task="Find for every model the dataset on which it falls furthest behind the best.")
    explore_dataset
    return (explore_dataset,)


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    On the two round clumps, the nearest centroid (0.823) and the widest street (0.817) draw almost the same straight line, which is what two round clumps call for; k-NN (0.743) and the tree (0.727) chase the examples where the clumps overlap, and lose. On the straight line, the widest street gets 0.997 and k-NN 0.993; the nearest centroid, 0.763, puts its line halfway between the two means, across the bands. No straight line fits the ring: the nearest centroid gets 0.567 and the widest street 0.650, k-NN 1.000 and the tree 0.990 with a box. On the blocks the tree finds the two thresholds, 0.993. On the coin flips all four are close to 0.5: the widest street gives up and says class 1 everywhere, and 163 of the 300 examples are class 1, so 0.543; k-NN even ends below 0.5, at 0.427. The two flexible models, k-NN and the tree, never fall more than about 0.1 behind on these five: they can bend their boundary around the ring and the blocks, where a straight line cannot. They pay for that on the clumps, where they chase the overlap, and k-NN pays at every question, as a lazy learner. These are one draw of the datasets: with another, a gap of a few hundredths can turn around; a gap of 0.2 does not.

    ### Recipe

    To choose a model without a free lunch:

    1. Ask what each model assumes: near examples alike, one round clump per class, a straight line, thresholds on one feature at a time.
    2. Ask what you know about your data: draw it if you can, and think about where it comes from.
    3. Compare the models whose assumption could fit, with cross-validation on the training data.
    4. On a new problem, compare again: the last winner proves nothing there.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The tiny world of *The idea*. The cell makes the 8 worlds of the 3 unseen examples, takes two rules that both fit the 5 training examples, and counts how many unseen examples each rule gets right in every world. The `#` comments say which step each part is. Run the cell.
    """)
    return


@app.cell
def example_world():
    # the tiny world: the unseen examples are 6 (0, 1, 1), 7 (1, 0, 1) and 8 (1, 1, 1)

    # 1. every way to label the 3 unseen examples: 8 worlds, each fits the training set
    worlds = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                worlds.append([a, b, c])
    print(len(worlds))

    # 2. two rules that fit all 5 training examples, and what they say for the unseen ones
    first = [0, 1, 1]   # the label is feature 1
    other = [1, 1, 0]   # feature 1, flipped when features 2 and 3 are both 1

    # 3. how many unseen examples each rule gets right, in every world
    sum_first = 0
    sum_other = 0
    for world in worlds:
        right_first = 0
        right_other = 0
        for i in range(3):
            if first[i] == world[i]:
                right_first = right_first + 1
            if other[i] == world[i]:
                right_other = right_other + 1
        print(world, right_first, right_other)
        sum_first = sum_first + right_first
        sum_other = sum_other + right_other
    print(sum_first, sum_other)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    8 worlds. In each, the rules score differently: in the world `[0, 1, 1]` the first rule gets all 3 right and the other 1, in `[1, 1, 0]` the other way around, and in `[1, 0, 0]` the first gets none. But over the 8 worlds both get 12 of the 24 right: half. Change either rule into any other three labels, and the sum stays 12. Only if you know something more about the world, such as "the label is always feature 1", can one rule do better than another. That is why the recipe starts with assumptions, and why step 4 holds.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_worlds():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every world

        `all_worlds(n)` must give every world for `n` unseen examples: every list of `n` labels that are each 0 or 1, as step 1 of the worked example, but for any `n`. Three nested loops only work for 3, so this function grows the worlds one example at a time: every world so far gives two longer ones, one with a 0 at the end and one with a 1. Replace the `...`. The check calls `all_worlds(3)`, and tries 1, 2, 4 and 0 examples; the order of the worlds does not matter.
        """),
        L.task("Let `all_worlds` give back every world for `n` examples, each once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worlds():
    box_worlds = L.code_box("all-worlds", lines=9)
    box_worlds
    return (box_worlds,)


@app.cell(hide_code=True)
def check_worlds(box_worlds):
    result_worlds = L.check("all-worlds", box_worlds.value)
    result_worlds
    return (result_worlds,)


@app.cell(hide_code=True)
def hints_worlds():
    L.hints(
        "all-worlds",
        [
            "Every world so far gives two new ones. Before the first step there is one world without labels, `[]`; after one step there are `[0]` and `[1]`; after two `[0, 0]`, `[0, 1]`, `[1, 0]` and `[1, 1]`.",
            "`world + [0]` is the world with a 0 added at the end. `longer.append(...)` puts it in the new list.",
            "`longer.append(world + [0])` and `longer.append(world + [___])`.",
        ],
        why="For 3 examples that gives the 8 worlds of the worked example, in the same order.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Right on average

        `mean_right(predictions, worlds)` must say how often a rule with these `predictions` is right on the unseen examples, averaged over the `worlds`: in every world the share of the predictions that equal its labels, and then the mean of those shares. Replace the two `...`. The check calls `mean_right([0, 1, 1], worlds)` with the first rule and the 8 worlds of the worked example, and tries five more: the other rule, the worlds in which the first unseen example is 0, four examples, and two examples, with all their worlds and with only some.
        """),
        L.task("Let `mean_right` give back the mean share of right predictions over the worlds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-right", lines=9)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-right", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-right",
        [
            "Step 3 of the worked example, for one rule, and then a mean: the share right in each world is already added up in `total`.",
            "Count a right answer when `predictions[i] == world[i]`: `right = right + 1`. The mean over the worlds is `total` divided by how many worlds there are.",
            "`if predictions[i] == world[i]:` with `right = right + 1` under it, and `return total / len(___)`.",
        ],
        why="The first rule gets 12 of the 24 right over the 8 worlds: its shares add up to 4, and 4 / 8 = 0.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_one():
    mo.md(r"""
    ### Exercise 3. One win
    """)
    return


@app.cell(hide_code=True)
def choice_one_cell():
    choice_one = L.choice(
        "one-dataset",
        question="A colleague tried the four models of this lesson on their dataset, and with cross-validation the decision tree won clearly. They conclude that the decision tree is the best model, and will use it for every problem from now on. What do you say?",
        options={
            "agree": "Agreed: cross-validation showed it, so the tree is the best model",
            "right": "The tree fits this data; on another problem another model can win, so compare again each time",
            "never": "Cross-validation cannot tell models apart: the theorem says they are all equally good",
        },
        correct="right",
        why={
            "agree": "Cross-validation shows the tree wins on this data: its assumption, thresholds on one feature at a time, fits it. On the two round clumps here it came last. See *No free lunch* in *The idea*.",
            "never": "The theorem is about the average over all possible problems. On one real problem the models can differ a lot, and cross-validation shows how much: on the ring here, from 0.567 to 1.000. See *No free lunch* in *The idea*.",
            "right": "Averaged over all problems no model wins, so a win says something about the problem, not about the model. On the next problem, compare again.",
        },
    )
    choice_one
    return (choice_one,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Each of the four models stands for one assumption: k-NN from [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), the nearest centroid from [Learn now or later: the nearest centroid](/course/open/ml-nearest-centroid?lang=en), the widest street from [The widest street: the maximum-margin classifier](/course/open/ml-max-margin?lang=en) and the tree from [Question after question: the decision tree](/course/open/ml-decision-tree?lang=en). Which assumption fits your data, you find out with [Choosing more reliably: cross-validation](/course/open/ml-cross-validation?lang=en). This closes the module on models. In the next lesson, *Clusters inside clusters: hierarchical clustering*, the labels are gone, and the question is which examples belong together.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_one, result_mean, result_worlds):
    L.wrap_up(result_worlds, result_mean, choice_one)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
