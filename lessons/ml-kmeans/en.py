# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-kmeans. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-kmeans", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en) and [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="You have the measurements of 342 penguins, but nobody wrote down their species. You want a model to find groups in them. Which kind of learning is that?",
        options={
            "supervised": "Supervised learning",
            "right": "Unsupervised learning",
            "rl": "Reinforcement learning",
        },
        correct="right",
        why={
            "supervised": "Supervised learning needs a label for every example, and here there is none. See [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "rl": "Reinforcement learning learns from rewards for actions; here there are only measurements. See [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "right": "There are no labels at all: the model has only the measurements to go on.",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Bill length varies by about 5 mm between penguins, flipper length by about 14 mm. What happens to the distance between two penguins if you do not standardize them?",
        options={
            "same": "Nothing: millimeters are millimeters",
            "right": "Flipper length counts for more in the distance than bill length",
            "bill": "Bill length counts for more, because it is the first feature",
        },
        correct="right",
        why={
            "same": "Every millimeter counts the same, but flipper lengths differ by more millimeters, so they make most of the distance. See the worked example in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "bill": "The order of the features does not matter to a distance; how much each one varies does. See the worked example in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "Standardizing gives every feature a standard deviation of 1, so that each counts as much in the distance.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far every example came with its label: the species of a penguin, the price of a house. Often you have only the measurements. A shop knows what its customers buy, but not which kinds of customers it has; a biologist has measured animals, but does not yet know how many kinds there are. Then you want the data itself to show which examples belong together. That is what this lesson does with the penguins, without telling the model their species, and only at the end do you look at how well the groups match the species.

    **An everyday example.** You tip out a jar of beads and sort them into three piles without anyone telling you which kinds there are. You start with three beads that look different, put every bead on the pile it looks most like, and then look again at what each pile has become: a pile of mostly large beads now stands for "large". You move a few beads, look again, and stop when no bead moves any more.
    """)
    return


@app.cell(hide_code=True)
def predict_found():
    guess_found = L.guess(
        "species-found",
        question="k-means puts the 342 penguins into 3 groups by bill length and flipper length, standardized, and never sees their species. Afterwards you count, for every group, the penguins of the species most common in it. How many of the 342 do you think that is?",
        options={
            "third": "About a third, as by chance",
            "twothirds": "About two thirds",
            "right": "Almost all of them",
        },
        reveal="**327 of the 342**, 96%. One group holds 122 Gentoos, one 146 Adelies, one 59 Chinstraps, and only 15 penguins are in the group of another species. Without standardizing, so with flipper lengths spread about two and a half times as wide as bill lengths, it is 287.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Clustering** splits examples without labels into groups, so that examples in a group are alike and examples in different groups are not. One such group is a **cluster**. A cluster can be described by its **centroid**: the mean of its examples, feature by feature, its center point.

        **k-means** is the best-known way to cluster. You choose the number of clusters, k, and k starting centroids, such as k examples picked at random. Then k-means repeats two steps:

        1. **Assign**: every example goes to the cluster of its nearest centroid.
        2. **Move**: every centroid moves to the mean of the examples in its cluster.

        When a step leaves every example in the cluster it was in, the centroids no longer move either, and k-means is done. Nearest means the distance of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en), so you standardize first. k-means learns the centroids; it never learns what a cluster *is*. Which cluster is the Adelies, you find out by looking.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **clustering** | splitting examples without labels into groups of examples that are alike | |
        | **cluster** | one such group | `labels == j` |
        | **centroid** | the mean of the examples in a cluster, feature by feature | `X[labels == j].mean(axis=0)` |
        | **k-means** | clustering by repeating two steps: every example to its nearest centroid, every centroid to the mean of its examples | |
        | `KMeans(n_clusters=3, n_init=10, random_state=0)` | k-means in scikit-learn: 3 clusters, from 10 starts, of which it keeps the best; `random_state=0` makes the random starts the same every run. `.fit(X)` runs it and gives the model back, so `km = KMeans(...).fit(X)` is the fitted model | |
        | `km.labels_`, `km.cluster_centers_` | after `fit`: the cluster of every example, and the centroids | |
        """),
        L.watch_out("The numbers of the clusters mean nothing. Cluster 0 is not \"the first species\": another start, or scikit-learn, can give the same groups other numbers."),
        mo.md(r"""
        The picture shows k-means by hand on the 342 penguins, by bill length and flipper length, standardized. It starts from three penguins picked at random, two of them Gentoos and one an Adelie. On the left, every penguin in the colour and shape of its cluster, and the three centroids in orange, with an arrow from where they were one step before; each cluster has the colour of the species most of its penguins will have at the end. On the right, the species, which k-means never saw: a circle is an Adelie, a triangle a Chinstrap and a square a Gentoo. Here colour and shape mean a group. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    def _frame(step):
        step = int(step)
        centers, assigned, changed = C.steps()[step]
        counts = [int((assigned == j).sum()) for j in range(3)]
        if step == 0:
            said = f"""The start: every penguin goes to its nearest starting centroid. The clusters hold {counts[0]}, {counts[1]} and {counts[2]} penguins: two centroids share most of the Gentoos, and one gets all the Adelies and most of the Chinstraps."""
        elif changed > 0:
            said = f"""Step {step}: the centroids moved to the means of their clusters, and then {changed} penguins changed cluster. The clusters hold {counts[0]}, {counts[1]} and {counts[2]} penguins."""
        else:
            said = f"""Step {step}: the centroids moved once more, and no penguin changed cluster: k-means is done. The clusters hold {counts[0]}, {counts[1]} and {counts[2]} penguins, {C.matched(C.penguins()["species"], assigned)} of them in the cluster of their own species."""
        alt = (f"The first chart: the 342 penguins by bill length, 32 to 60 mm, and flipper length, 172 to 231 mm, in three clusters after step {step} of k-means, with their three centroids. "
               + "The second chart: the same penguins by species, with the centroids k-means ends with; Adelies have short bills and flippers, Chinstraps long bills and short flippers, Gentoos long flippers. "
               + said.replace("**", ""))
        return (C.picture(step, {"title": f"The clusters, step {step}", "species": "The species, unseen by k-means",
                                 "x": "bill length (mm)", "y": "flipper length (mm)", "alt": alt}), said)

    explore_steps = L.explore("steps", "step", _frame, start=0, values=C.STEPS, role="",
                              task="Go through the steps. Where does the third centroid go, and when does nothing change any more?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Step by step the third centroid comes down from the Gentoos to the long bills and short flippers of the Chinstraps, and the second gives up its Chinstraps to it. 50, 24, 16, 20, 11, 4 and then 0 penguins change cluster: after step 7 nothing moves any more. The three centroids end at 47.6 and 217.0 mm, 38.7 and 189.6 mm, and 49.0 and 195.7 mm, close to the means of the Gentoos, the Adelies and the Chinstraps.

    ### Recipe

    To cluster with k-means:

    1. Choose the features, and standardize them.
    2. Choose k, and take k examples as the starting centroids.
    3. Assign: give every example the cluster of its nearest centroid.
    4. Move: put every centroid at the mean of the examples in its cluster.
    5. Repeat 3 and 4 until no example changes cluster.
    6. Look at the clusters: what do the examples in each have in common? With labels you held back, count how well they match.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 5 of the recipe on six penguins, small enough to follow by eye: in the table of 342, the penguins in rows 0, 1 and 2 are Adelies, those in rows 151, 152 and 153 Gentoos. Standardized with the mean and standard deviation of all 342, with k = 2, and the first two penguins, both Adelies, as the starting centroids. For every round the cluster of each of the six; after three rounds the centroids. `np.array(dists)` has a row for every centroid and a column for every penguin, so `np.argmin(..., axis=0)` gives every penguin the row of its nearest centroid. At the end scikit-learn does the same with `KMeans`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_kmeans():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 1. standardize, and take six penguins
    _X = (_raw - _raw.mean(axis=0)) / _raw.std(axis=0)
    _six = _X[[0, 1, 2, 151, 152, 153]]
    # 2. k = 2, and the first two penguins as the starting centroids
    _centers = _six[[0, 1]]
    # 5. repeat 3 and 4: in round 2 nothing changes any more
    for _round in range(3):
        # 3. every penguin to its nearest centroid
        _dists = []
        for _c in _centers:
            _dists.append(np.linalg.norm(_six - _c, axis=1))
        _labels = np.argmin(np.array(_dists), axis=0)
        print(_round, _labels)
        # 4. every centroid to the mean of its penguins
        _means = []
        for _j in range(2):
            _means.append(_six[_labels == _j].mean(axis=0))
        _centers = np.array(_means)
    print(np.round(_centers, 2))
    # the same with scikit-learn
    _km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_six)
    print(_km.labels_)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In round 0 only the first penguin is nearest to the first centroid; the other five, Gentoos included, go to the second. That centroid moves to the mean of those five, towards the Gentoos, so in round 1 the two other Adelies are nearer to the first centroid. In round 2 nothing changes: the Adelies are cluster 0, the Gentoos cluster 1. scikit-learn finds the same two groups but calls the Adelies cluster 1: the numbers mean nothing.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step by hand

        The box reads all 342 penguins and standardizes them in `X`, and takes rows 216, 54 and 240 as the starting centroids: the start of the picture. Replace the three `...`: the distance from every penguin to centroid `c`, the nearest centroid of every penguin, and the mean of the penguins of cluster `j`. The last line turns the new centroids back into millimetres.
        """),
        L.task("Do one assign step and one move step by hand on all penguins."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("one-step", lines=19)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("one-step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "one-step",
        [
            "Steps 3 and 4 of the worked example, now with `X` instead of `_six` and 3 clusters instead of 2.",
            "The distance is `np.linalg.norm` of the difference between all penguins and one centroid, along `axis=1`; the nearest is `np.argmin` along `axis=0`; a centroid is the mean of a part of `X`.",
            "`dists.append(np.linalg.norm(X - c, axis=___))`, `labels = np.argmin(np.array(dists), axis=___)` and `new.append(X[labels == j].mean(axis=0))`.",
        ],
        why="`np.array(dists)` has a row for each of the 3 centroids and a column for each of the 342 penguins, so `np.argmin` along `axis=0` gives every penguin its nearest centroid. `labels == j` then picks the penguins of cluster `j`, and their mean is its new centroid.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stable():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Until nothing changes

        Step 5 of the recipe. The box has the same start, and a loop with both steps. `labels` starts at −1 for every penguin, no cluster yet; `new` is the cluster each penguin gets in this step. Replace the two `...`: how many penguins changed cluster, and when the loop stops. The last line gives the step where it stopped, and how many penguins each cluster holds.
        """),
        L.task("Repeat the two steps until no penguin changes cluster."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stable():
    box_stable = L.code_box("until-stable", lines=26)
    box_stable
    return (box_stable,)


@app.cell(hide_code=True)
def check_stable(box_stable):
    result_stable = L.check("until-stable", box_stable.value)
    result_stable
    return (result_stable,)


@app.cell(hide_code=True)
def hints_stable():
    L.hints(
        "until-stable",
        [
            "A penguin changed cluster when its cluster in `new` is not the one in `labels`. How do you count how many times a comparison is `True`?",
            "`new != labels` gives `True` or `False` for every penguin, and `np.sum` counts the `True` ones. The loop stops with `break` when nothing changed.",
            "`changed = np.sum(new ___ labels)` and `if changed == ___:`.",
        ],
        why="When no penguin changes cluster, the centroids, the means of the same penguins, stay where they are too, and every next step would be the same. So k-means has ended there; with the −1 at the start, step 0 always counts all 342 penguins as changed.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_four():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. All four measurements, with scikit-learn

        Now k-means from scikit-learn, on all four measurements: bill length, bill depth, flipper length and body mass, standardized in `X4`. Replace the two `...`: `KMeans` with 3 clusters, `n_init=10` and `random_state=0`, fitted on `X4`, and, for cluster `j`, the number of penguins of the species most common in it. `right` then adds those up, as in the guess at the start.
        """),
        L.task("Cluster the penguins on four measurements, and count how many are in the cluster of their own species."),
    ])
    return


@app.cell(hide_code=True)
def exercise_four():
    box_four = L.code_box("four", lines=16)
    box_four
    return (box_four,)


@app.cell(hide_code=True)
def check_four(box_four):
    result_four = L.check("four", box_four.value)
    result_four
    return (result_four,)


@app.cell(hide_code=True)
def hints_four():
    L.hints(
        "four",
        [
            "The new words of *The idea* say how to make `KMeans` and run it. `np.unique(..., return_counts=True)` gives, for one cluster, every species in it and how many penguins of each.",
            "`km` is the fitted model, so `KMeans(...)` followed by `.fit(X4)`. Of the counts of one cluster you want the largest.",
            "`km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(___)` and `right = right + counts.___()`.",
        ],
        why="`km.labels_ == j` picks the penguins of cluster `j`, `np.unique` counts their species, and the largest count is the penguins that share the species most common in that cluster. Added up over the three clusters, that is how well the clusters match the species.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 4. Why fewer with more measurements?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-four",
        question="With bill length and flipper length, 327 penguins are in the cluster of their own species; with all four measurements, 313, and 24 Adelies are in the cluster of the Chinstraps. Standardized, Adelies and Chinstraps differ by almost 2 in bill length, and by less than 0.5 in each of the other three measurements. Why do four measurements match the species less well here?",
        options={
            "always": "More measurements always make clusters worse",
            "right": "Bill length is now one of four measurements in the distance, and the other three hardly tell Adelies and Chinstraps apart",
            "two": "k-means can only use two measurements at a time",
        },
        correct="right",
        why={
            "always": "Not always: a measurement in which the groups differ helps. Here bill depth and body mass hardly differ between Adelies and Chinstraps, so they add distance that has nothing to do with the species.",
            "two": "k-means uses every feature you give it: the distance is over all four. Ask yourself in which of them Adelies and Chinstraps differ.",
            "right": "k-means only knows distances. A feature in which two species barely differ still adds to the distance, and then differences within a species weigh more: of the 24 Adelies in the cluster of the Chinstraps, 23 are males: heavier, with deeper and longer bills. Which features you choose decides which groups k-means can find.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** k-means has no labels, as in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en), and measures nearness with the standardized distance of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en); a centroid is the mean along an axis that you took in NumPy. Here you chose k = 3 because you knew there are three species. In the next lesson, *How many clusters? Choosing k*, you choose k when you do not know that, and you see why k-means is run from several starts.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_four, result_stable, result_step):
    L.wrap_up(result_step, result_stable, result_four, choice_why)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
