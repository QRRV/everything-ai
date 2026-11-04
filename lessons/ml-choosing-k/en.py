# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-choosing-k. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-choosing-k", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stop():
    quick_stop = L.choice(
        "quick-stop",
        question="When is k-means done?",
        options={
            "k": "After k steps, one for every cluster",
            "right": "When no example changes cluster any more",
            "equal": "When every cluster holds as many examples as the others",
        },
        correct="right",
        why={
            "k": "The number of steps has nothing to do with k: on the penguins, with k = 3, it took 7. See the picture in [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "equal": "Clusters can differ in size: on the penguins they held 127, 151 and 64. See [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "right": "Then the centroids, the means of the same examples, stay where they are too, and every next step would be the same.",
        },
    )
    quick_stop
    return (quick_stop,)


@app.cell(hide_code=True)
def quick_check_numbers():
    quick_numbers = L.choice(
        "quick-numbers",
        question="You run k-means twice and get the same three groups, but the Adelies are cluster 0 the first time and cluster 2 the second. What does that say?",
        options={
            "wrong": "The second run is wrong",
            "right": "Nothing: the numbers of the clusters mean nothing",
            "moved": "Some Adelies changed cluster",
        },
        correct="right",
        why={
            "wrong": "Both runs found the same groups; only their names differ. See the worked example in [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "moved": "The groups are the same, so no penguin changed group; only the number of the group changed. See the worked example in [Groups without labels: k-means](/course/open/ml-kmeans?lang=en).",
            "right": "k-means only says which examples belong together. Which cluster is the Adelies, you find out by looking.",
        },
    )
    quick_numbers
    return (quick_numbers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the last lesson you chose k = 3, because you knew there are three species of penguins. Without labels you do not know that: a shop does not know how many kinds of customers it has, and that is just what it wants to find out. So you need a way to choose k from the data. And there is a second gap: k-means always ends, but where it ends depends on where it starts.

    **An everyday example.** You sort your holiday photos into albums. One album is a mess; an album for every photo is no use either. Somewhere in between, a new album stops making the others much tidier: that is about how many you want. And a photo is in a good album when it is more like the photos in its own album than like those in the album next to it.
    """)
    return


@app.cell(hide_code=True)
def predict_smallest():
    guess_smallest = L.guess(
        "smallest-sum",
        question="k-means makes the squared distances from every penguin to the centroid of its cluster small. You run it on the penguins for every k from 1 to 8 and add up those squared distances. For which k do you think the sum is smallest?",
        options={
            "three": "k = 3, the number of species",
            "two": "k = 2",
            "right": "k = 8, the largest k you try",
        },
        reveal="**k = 8.** The sum is 684.0 with one cluster, 247.0 with two, 157.8 with three and 57.8 with eight: with every extra centroid, the penguins are nearer to one. With 342 clusters, one for every penguin, it would be 0. So the smallest sum alone cannot choose k.",
    )
    guess_smallest
    return (guess_smallest,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **within-cluster sum of squares** adds up, for every example, the squared distance to the centroid of its cluster. It is what k-means makes small: assigning gives every example its nearest centroid, and moving puts every centroid at the mean of its examples, the point with the smallest sum of squared distances to them. scikit-learn calls it `inertia_`. The smallest sum there is only gets smaller as k grows, so you do not look for the smallest, but for the **elbow**: the k after which an extra cluster makes it only a little smaller. Choosing k that way is the elbow method.

        The **silhouette** of one example asks how clearly it lies in its own cluster. Take $a$, the mean distance to the other examples in its own cluster, and $b$, the mean distance to the examples of the nearest other cluster. Then

        $$s = \frac{b - a}{\max(a, b)}$$

        is between −1 and 1: near 1 when the example is much nearer to its own cluster than to any other, near 0 when it lies on the border, and below 0 when another cluster is nearer. The **silhouette score** is the mean silhouette of all examples. It needs at least two clusters, and higher is better.

        For example, the Adelie in row 94 of the table has a bill of 40.8 mm and long flippers of 208 mm. In the three clusters of k = 3, standardized, it is on average 1.446 from the other penguins of its cluster, the Adelies, and 1.433 from the nearest other cluster, the Gentoos. So $a$ = 1.446, $b$ = 1.433, and its silhouette is (1.433 − 1.446) / 1.446 = −0.009: just below 0. k-means put it with the Adelies, yet on average the Gentoos are a little nearer; it is the only one of the 342 below 0. The Adelie in row 0, with a bill of 39.1 mm and flippers of 181 mm, has $a$ = 0.809 and $b$ = 2.135, to the Chinstraps: its silhouette is (2.135 − 0.809) / 2.135 = 0.621, clearly an Adelie.

        One more thing: k-means stops when no example changes cluster, but that is not always the smallest sum there is. From a bad start it can end in a **local minimum**: a solution that neither step improves any more, while another one has a smaller sum. That is why `KMeans` runs from `n_init` starts and keeps the one with the smallest sum.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **within-cluster sum of squares** | for every example, the squared distance to the centroid of its cluster, added up | `km.inertia_` |
        | **elbow** | the k after which an extra cluster makes the within-cluster sum of squares only a little smaller | |
        | **silhouette** | for one example, $(b - a) / \max(a, b)$: its own cluster against the nearest other one | `silhouette_samples(X, labels)` |
        | $a$, $b$ | the mean distance to the other examples of its own cluster, and to those of the nearest other cluster | |
        | **silhouette score** | the mean silhouette of all examples, from −1 to 1 | `silhouette_score(X, labels)` |
        | **local minimum** | a solution that no step makes better, while there is a better one | |
        | `n_init=10` | run k-means from 10 starts and keep the one with the smallest within-cluster sum of squares | |
        """),
        L.watch_out("The silhouette score says how clearly the clusters lie apart, not whether they are the groups you are after. Two clusters can lie further apart than the three species do."),
        mo.md(r"""
        The picture shows `KMeans` with `n_init=10` on the 342 penguins, by bill length and flipper length, standardized, for every k from 1 to 8. On the left every penguin, joined by a thin line to the centroid of its cluster; the centroids are in orange. The within-cluster sum of squares is the sum of the squared lengths of all those lines. On the right that sum for every k, in purple, with this k filled in. Move the slider, or drag across the right chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        f = C.fit(k)
        if k == 1:
            said = f"""k = 1: one centroid, at the mean of all penguins. The within-cluster sum of squares is {L.number(f["wcss"], 1)}: 342 for each of the two standardized features. There is no silhouette score, because there is no other cluster."""
        else:
            said = f"""k = {k}: the within-cluster sum of squares is {L.number(f["wcss"], 1)}, {L.number(C.fit(k - 1)["wcss"] - f["wcss"], 1)} less than with one cluster fewer. The silhouette score is {L.number(f["silhouette"], 3)}."""
        alt = (f"The first chart: the 342 penguins by bill length and flipper length, standardized, each joined by a line to the nearest of the {k} centroids of k-means. "
               + "The second chart: the within-cluster sum of squares for k from 1 to 8; it falls from 684.0 to 247.0, 157.8, 118.8, 91.5, 78.5, 66.7 and 57.8. "
               + said)
        return (C.picture(k, {"title": f"{k} clusters", "every": "Every k", "k": "k", "wcss": "within-cluster sum of squares",
                              "x": "bill length, standardized", "y": "flipper length, standardized", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="hyper",
                          task="Go from k = 1 to 8. After which k does the sum stop falling fast, and which k has the highest silhouette score?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    The sum falls by 437.0 from one cluster to two, by 89.2 to three, and then by 39.0 and less. The elbow is not sharp here: you can see it at 2 or at 3. The silhouette score is highest at k = 2, with 0.543, then 0.522 at k = 3 and 0.507 at k = 4, and from k = 5 it is 0.435 or lower.

    ### Recipe

    To choose k for k-means:

    1. Standardize the features.
    2. Run `KMeans` for every k you consider, from several starts with `n_init`.
    3. Note the within-cluster sum of squares of every k, and look for the elbow.
    4. Compute the silhouette score of every k from 2: higher means clusters that lie more clearly apart.
    5. Look at the clusters of the best few k, and choose with what you want them for.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The recipe on the penguins, by bill length and flipper length. For every k from 1 to 8 the cell prints k, the within-cluster sum of squares by hand, the same from scikit-learn, and from k = 2 the silhouette score. `_km.cluster_centers_[_km.labels_]` gives, in a row for every penguin, the centroid of its cluster, so `_X` minus it is every penguin's difference from its own centroid. At the end, for step 5, the species in each of the two clusters of k = 2, as in exercise 3 of the last lesson. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_choose_k():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 1. standardize the features
    _X = (_raw - _raw.mean(axis=0)) / _raw.std(axis=0)
    for _k in range(1, 9):
        # 2. KMeans for this k, from 10 starts
        _km = KMeans(n_clusters=_k, n_init=10, random_state=0).fit(_X)
        # 3. the within-cluster sum of squares, by hand and from scikit-learn
        _wcss = np.sum((_X - _km.cluster_centers_[_km.labels_]) ** 2)
        if _k == 1:
            print(_k, np.round(_wcss, 1), np.round(_km.inertia_, 1))
        else:
            # 4. the silhouette score, from k = 2
            print(_k, np.round(_wcss, 1), np.round(_km.inertia_, 1), np.round(silhouette_score(_X, _km.labels_), 3))
    # 5. look at the clusters of k = 2: the species in each
    _km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(_X)
    _species = np.array(_df["species"])
    for _j in range(2):
        _names, _counts = np.unique(_species[_km.labels_ == _j], return_counts=True)
        print(_j, _names, _counts)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sum by hand is the same as `inertia_` for every k. Step 5: the elbow at 2 or 3 and the highest silhouette score at 2 leave k = 2 and k = 3. The two clusters of k = 2 are mostly the short bills and flippers against the long ones: 149 Adelies and 23 Chinstraps, against 2 Adelies, 45 Chinstraps and 123 Gentoos. The three of k = 3 are close to the species, as in the last lesson. Which of the two is right, exercise 4 asks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_wcss():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The within-cluster sum of squares by hand

        Step 3 of the worked example, now on all four measurements: bill length, bill depth, flipper length and body mass, standardized in `X4`. The box clusters them with `KMeans` into 3 clusters. Replace the `...`: the within-cluster sum of squares, in NumPy. The last line puts your sum next to `km.inertia_`.
        """),
        L.task("Compute the within-cluster sum of squares of the three clusters on four measurements."),
    ])
    return


@app.cell(hide_code=True)
def exercise_wcss():
    box_wcss = L.code_box("wcss", lines=12)
    box_wcss
    return (box_wcss,)


@app.cell(hide_code=True)
def check_wcss(box_wcss):
    result_wcss = L.check("wcss", box_wcss.value)
    result_wcss
    return (result_wcss,)


@app.cell(hide_code=True)
def hints_wcss():
    L.hints(
        "wcss",
        [
            "Step 3 of the worked example, with `X4` instead of `_X` and `km` instead of `_km`.",
            "`X4 - km.cluster_centers_[km.labels_]` gives every penguin its difference from its own centroid, in all four features. Square that with `** 2`, and add it all up with `np.sum`.",
            "`wcss = np.sum((X4 - km.cluster_centers_[km.labels_]) ** ___)`.",
        ],
        why="The squared distance of a penguin to its centroid is the sum of its squared differences in all four features, so adding up all squared differences, of every penguin and every feature, gives the sum of all those squared distances at once.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_starts():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Ten starts

        `by_hand` is k-means by hand, as in the last lesson: from the rows `start` as the starting centroids, until no penguin changes cluster, and then it returns the within-cluster sum of squares. Replace the two `...`: three random rows for this `seed`, and the result of `by_hand` from them. `np.random.default_rng(seed).permutation(len(X))` puts the row numbers 0 to 341 in a random order, a different one for every `seed`. The last line gives the smallest and the largest of the ten results.
        """),
        L.task("Run k-means by hand from ten starts, and compare where they end."),
    ])
    return


@app.cell(hide_code=True)
def exercise_starts():
    box_starts = L.code_box("starts", lines=31)
    box_starts
    return (box_starts,)


@app.cell(hide_code=True)
def check_starts(box_starts):
    result_starts = L.check("starts", box_starts.value)
    result_starts
    return (result_starts,)


@app.cell(hide_code=True)
def hints_starts():
    L.hints(
        "starts",
        [
            "Three random rows are the first three row numbers of a random order. And `by_hand` needs the penguins and the starting rows.",
            "Take the first three of `np.random.default_rng(seed).permutation(len(X))` with a slice, and add `by_hand(X, start)` to `results`.",
            "`start = np.random.default_rng(seed).permutation(len(X))[:___]` and `results.append(by_hand(___, start))`.",
        ],
        why="Every seed gives other starting rows, and k-means ends wherever its two steps take it from there. If two results differ, at least one of them is a local minimum: neither step improves it, yet another start found a smaller sum.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_best():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. k by the silhouette score

        The loop runs `KMeans` for every k from 2 to 8, and keeps in `best` the k with the highest score so far, and in `top` that score. Replace the `...`: the silhouette score of the clusters of this `km`. The last line gives the best k and its score.
        """),
        L.task("Choose k by the silhouette score."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best():
    box_best = L.code_box("best-k", lines=19)
    box_best
    return (box_best,)


@app.cell(hide_code=True)
def check_best(box_best):
    result_best = L.check("best-k", box_best.value)
    result_best
    return (result_best,)


@app.cell(hide_code=True)
def hints_best():
    L.hints(
        "best-k",
        [
            "The new words of *The idea* say which function gives the silhouette score, and what it needs.",
            "`silhouette_score` needs the penguins and the cluster of every penguin, which `km.labels_` holds.",
            "`score = silhouette_score(___, km.labels_)`.",
        ],
        why="The silhouette score looks at the clusters, not at the centroids or the sum: for every penguin it compares its own cluster with the nearest other one. A higher score means clusters that lie more clearly apart, so the loop keeps the highest.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Exercise 4. Which k?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which-k",
        question="On the penguins, the silhouette score is highest for k = 2, the elbow is at 2 or 3, and there are three species. The species themselves, as clusters, have a silhouette score of 0.483: lower than the 0.522 of the three clusters of k-means. Which k is right?",
        options={
            "two": "k = 2, because its silhouette score is the highest",
            "three": "k = 3, because there are three species",
            "right": "That depends on what you want the clusters for: no number decides it on its own",
        },
        correct="right",
        why={
            "two": "The two clusters of k = 2 lie the most clearly apart, but they put the Chinstraps partly with the Adelies and partly with the Gentoos. If you are looking for species, that is not what you want. The silhouette says how clearly clusters lie apart, not what they are for.",
            "three": "Here you know there are three species, but without labels you would not know it. And even the species themselves lie less clearly apart than two clusters: by the numbers alone, nobody would pick them.",
            "right": "k = 2 is the clearest split, short bills and flippers against long ones; k = 3 finds the species, which a biologist wants. The within-cluster sum of squares and the silhouette score narrow the choice down; what the clusters are for makes it.",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The within-cluster sum of squares is what the two steps of [Groups without labels: k-means](/course/open/ml-kmeans?lang=en) make small, and the silhouette uses the same standardized distance as [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). k is a hyperparameter, as the k of k-nearest neighbors is, but without labels there is no validation set to choose it with, so you choose with the sum, the silhouette and what you know about the data. In the next lesson, *A neuron*, the neural networks begin.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_which, result_best, result_starts, result_wcss):
    L.wrap_up(result_wcss, result_starts, result_best, choice_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
