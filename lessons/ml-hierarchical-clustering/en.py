# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-hierarchical-clustering. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-hierarchical-clustering", "en")
    C = L.checks   # the penguins of ml-choosing-k, the tree of every linkage, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="On the penguins, the best within-cluster sum of squares falls from 684.0 with one cluster to 57.8 with eight. Why can you not choose k by the smallest sum?",
        options={
            "right": "It only keeps falling as k grows: with a cluster for every penguin it is 0",
            "local": "k-means can end in a local minimum, so sums of different k cannot be compared",
            "scale": "The features are standardized, so the sum means nothing",
        },
        correct="right",
        why={
            "local": "With `n_init` starts, k-means finds the best sum for every k, and those you can compare: they fall as k grows. That is the trouble. See *The idea* in [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "scale": "Standardizing gives every feature the same weight; the sum still says how close the examples are to their centroids. It just keeps falling as k grows. See *The idea* in [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "right": "So you look for the elbow, or use the silhouette score. Today a method that gives you every k at once.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_starts():
    quick_starts = L.choice(
        "quick-starts",
        question="Why does `KMeans` run from `n_init` starts?",
        options={
            "k": "To try a different number of clusters from every start",
            "right": "From one start it can end in a local minimum, with a larger sum than the best",
            "faster": "So that every start needs fewer steps",
        },
        correct="right",
        why={
            "k": "Every start uses the same k; they differ only in where the centroids begin. See *The idea* in [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "faster": "Every start runs until no example changes cluster; more starts cost more time, not less. See *The idea* in [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "right": "And it keeps the start with the smallest sum. Today's method has no start: it always gives the same tree.",
        },
    )
    quick_starts
    return (quick_starts,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    k-means asks for k before it starts, and every k is a new clustering: the clusters of k = 3 need not lie inside those of k = 2. On the penguins they do not: one of the three clusters of k-means, 64 penguins, is split 21 and 43 over the two clusters of k = 2. Yet groups often hold groups: species, and within a species males and females. Hierarchical clustering builds one tree that holds every k at once, each clustering inside the one above it, and you choose k afterwards, by looking at the tree.

    **An everyday example.** A library sorts its books by subject, each subject by topic, and each topic by author. Ask for three sections or thirty: every one of the thirty lies inside one of the three, because both come from the same ordering.
    """)
    return


@app.cell(hide_code=True)
def predict_chain():
    guess_chain = L.guess(
        "chain-guess",
        question="Take the penguins by bill length and flipper length. Start with every penguin as its own cluster, and merge the two closest clusters again and again, where the distance between two clusters is that of their closest pair of penguins. Stop when two clusters are left. What are they?",
        options={
            "species": "The Adelies, and the Chinstraps with the Gentoos",
            "halves": "Two groups of about 170 penguins",
            "right": "341 penguins, and 1",
        },
        reveal="**341 and 1.** The one is a Chinstrap with a bill of 58.0 mm, the longest of its species, and short flippers, 181 mm. Between the species lie enough penguins that a chain of short steps joins them all, long before that lone penguin is closer to anyone.",
    )
    guess_chain
    return (guess_chain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **From the bottom up.** **Hierarchical clustering** in its usual form starts with every example as its own cluster: 342 clusters for the penguins. It merges the two clusters that lie closest together, and repeats, until one cluster is left: 341 merges. Every merge joins two clusters made before it, so the clusters form a tree. There is no start to choose and no k: the same data always give the same tree.

        **Linkage.** Which two clusters lie closest depends on how you measure the distance between two clusters. That rule is the **linkage**:

        - single linkage: the distance of the closest pair, one example from each cluster;
        - complete linkage: the distance of the farthest pair;
        - average linkage: the mean distance over all pairs;
        - Ward: how much the merge would raise the within-cluster sum of squares, the sum k-means makes small; Ward merges the two clusters for which that rise is smallest. scipy draws a Ward merge at the square root of twice that rise: the same order, another scale.

        **The dendrogram.** The **dendrogram** draws the tree. The examples stand side by side along the bottom, and every merge is a bracket that joins two branches at the height of its distance: the higher the bracket, the further apart the two clusters were. To get k clusters, cut the tree with a horizontal line under its k − 1 highest merges: the branches the line crosses are the clusters. Cut lower and you get more clusters, each inside one of the clusters above it.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **hierarchical clustering** | clustering that builds a tree of clusters inside clusters, here by merging the two closest clusters again and again | |
        | **linkage** | the rule for the distance between two clusters: single, complete, average or Ward | `linkage(X, "ward")` |
        | **dendrogram** | the tree as a picture: every merge a bracket at the height of its distance | |
        | cut | a horizontal line through the dendrogram: the branches it crosses are the clusters | `fcluster(Z, 3, criterion="maxclust")` |
        | `Z` | scipy's tree: a row for every merge, with the two clusters, their distance and the size of the new cluster | `Z = linkage(X, "ward")` |
        """),
        L.watch_out("Hierarchical clustering compares every pair of examples: for the 342 penguins that is 58,311 distances, for a million examples about 500 billion. For large data sets, k-means is far cheaper."),
        mo.md(r"""
        The picture shows Ward on the 342 penguins, by bill length and flipper length, standardized, as in the last lesson. On the left the dendrogram, with the cut as a <span class="c-hyper">dashed line</span>; the branches of each cluster in its own color. On the right the penguins in the color and shape of their cluster. Here color and shape mean a cluster. Move the slider from 1 to 4 clusters.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cut_cell():
    def _frame(k):
        k = int(k)
        sizes = C.sizes("ward", k)
        if k == 1:
            said = "1 cluster: all 342 penguins. There is no cut: every merge is done."
        else:
            said = (f"{k} clusters: {C.listed(sizes, L.lang)} penguins. The cut lies at height "
                    f"{L.number(C.cut('ward', k), 1)}, between the merges at "
                    f"{L.number(float(C.tree('ward')[-k][2]), 1)} and {L.number(float(C.tree('ward')[-k + 1][2]), 1)}.")
        alt = ("Left: the dendrogram of Ward on the 342 penguins, with its top merge at height 29.2. "
               "Right: the penguins by bill length and flipper length, standardized, in the color of their cluster. " + said)
        return (C.picture("ward", k, {"tree": "Ward: the tree", "leaves": "the 342 penguins, in the order of the tree",
                                      "height": "merge height", "title": f"{k} cluster" + ("s" if k > 1 else ""),
                                      "x": "bill length, standardized", "y": "flipper length, standardized", "alt": alt}), said)

    explore_cut = L.explore("clusters", "clusters", _frame, start=1, values=C.KS, role="hyper",
                            task="Go from 1 to 4 clusters. Which cluster splits at every step, and which ones stay as they were?")
    explore_cut
    return


@app.cell(hide_code=True)
def text_after_cut():
    mo.md(r"""
    With two clusters, Ward puts 146 Adelies and 6 Chinstraps on one side, and the rest on the other: 123 Gentoos, 62 Chinstraps and 5 Adelies. With three, that second cluster splits: 122 Gentoos with 2 Adelies and 2 Chinstraps, and 60 Chinstraps with 3 Adelies and 1 Gentoo. With four, the Gentoos split again: 46 penguins, 43 of them male, against a cluster that is mostly female. Clusters inside clusters: species, and then sex. The longest stretch without a merge lies between 14.1 and the top merge at 29.2, where you cut for two clusters; the next, between 8.7 and 14.1, gives three. A long stretch says that the merge above it joins clusters that lie far apart: the dendrogram's way of showing an elbow. With three clusters, Ward's within-cluster sum of squares is 159.3, close to the 157.8 of k-means, though Ward never looked for the best three: it built the whole tree.

    The second picture keeps three clusters and changes the linkage: 1 single, 2 complete, 3 average, 4 Ward. The heights differ per linkage, because each measures the distance between clusters in its own way.
    """)
    return


@app.cell(hide_code=True)
def explore_linkage_cell():
    def _frame(v):
        method = C.METHODS[int(v) - 1]
        name = C.NAMES[method][L.lang]
        said = f"{name[0].upper() + name[1:]}, 3 clusters: {C.listed(C.sizes(method, 3), L.lang)} penguins."
        alt = (f"Left: the dendrogram of {name} on the 342 penguins, cut into 3 clusters. "
               "Right: the penguins by bill length and flipper length, standardized, in the color of their cluster. " + said)
        return (C.picture(method, 3, {"tree": f"{name[0].upper() + name[1:]}: the tree", "leaves": "the 342 penguins, in the order of the tree",
                                      "height": "merge height", "title": "3 clusters",
                                      "x": "bill length, standardized", "y": "flipper length, standardized", "alt": alt}), said)

    explore_linkage = L.explore("linkage", "linkage", _frame, start=1, values=[1, 2, 3, 4], role="hyper",
                                task="Which linkages find the three species, and which ones leave a cluster of a single penguin?")
    explore_linkage
    return


@app.cell(hide_code=True)
def text_after_linkage():
    mo.md(r"""
    Single linkage gives 340, 1 and 1: the Chinstrap of the guess, and a Gentoo with the longest bill of all, 59.6 mm. Its dendrogram is a staircase: one big cluster that takes in one penguin after another. Average linkage gives 176, 165 and 1: the Adelies with 26 Chinstraps, the Gentoos with 41 Chinstraps, and that same Chinstrap alone. Complete linkage, with 151, 134 and 57, and Ward, with 152, 126 and 64, both find the three species: Ward with 14 penguins in the cluster of another species, complete linkage with 23. Which linkage is right depends on what you call a cluster: single linkage follows chains of close examples, complete linkage and Ward look for compact clumps.

    ### Recipe

    To cluster without choosing k first:

    1. Standardize the features, as for k-means.
    2. Build the tree: `Z = linkage(X, method)`. Ward when you expect compact clumps, as k-means does.
    3. Look at the dendrogram: a long stretch without merges is a good place to cut.
    4. Cut into k clusters with `fcluster(Z, k, criterion="maxclust")`, and look at their sizes: a cluster of one or two examples is an outlier, not a group.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Six numbers on a line, each an example with one feature: 1, 2, 4, 8, 9 and 15. The cell builds the tree with single linkage and with complete linkage, and cuts each into two clusters. scipy numbers the six examples 0 to 5, and the cluster that row $i$ of `Z` makes, counting from 0, gets number $6 + i$. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_tree():
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster

    # 1. six examples with one feature each, left as they are, so the heights are gaps between the numbers
    _points = np.array([[1.0], [2.0], [4.0], [8.0], [9.0], [15.0]])

    # 2. the tree, with the closest pair and with the farthest pair: per row the two clusters, their distance, the new size
    _single = linkage(_points, "single")
    _complete = linkage(_points, "complete")
    print(_single)
    print(_complete)

    # 4. cut each tree into two clusters: the cluster of every example
    print(fcluster(_single, 2, criterion="maxclust"))
    print(fcluster(_complete, 2, criterion="maxclust"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Both trees start the same: 1 and 2 merge at distance 1 into cluster 6, and 8 and 9 at distance 1 into cluster 7. Then single linkage adds 4 to cluster 6 at 2, the gap between 2 and 4; complete linkage at 3, the distance from 4 to 1. After that they part. Single linkage merges {1, 2, 4} with {8, 9} at 4, the gap between 4 and 8, and adds 15 last, at 6. Complete linkage first adds 15 to {8, 9}, at 7, the distance from 15 to 8, and merges the two halves last, at 14, the distance from 1 to 15. Cut into two, single linkage gives `[1 1 1 1 1 2]`: 15 alone. Complete linkage gives `[1 1 1 2 2 2]`: {1, 2, 4} and {8, 9, 15}. The same six numbers, two different trees, because the linkage decides what close means.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_link():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The closest pair

        `single_link(A, B)` should give back the single-linkage distance between two clusters: `A` and `B` are arrays with one row per example. The two loops already go through every pair, one example from each cluster. Replace the two `...`. The check calls `single_link(A, B)` with a cluster of two examples and one of three, and tries four more: the numbers 1, 2, 4 and 8, 9 of the worked example, three features, one example each, and a long cluster next to a short one.
        """),
        L.task("Make `single_link` give back the distance of the closest pair."),
    ])
    return


@app.cell(hide_code=True)
def exercise_link():
    box_link = L.code_box("single-link", lines=9)
    box_link
    return (box_link,)


@app.cell(hide_code=True)
def check_link(box_link):
    result_link = L.check("single-link", box_link.value)
    result_link
    return (result_link,)


@app.cell(hide_code=True)
def hints_link():
    L.hints(
        "single-link",
        [
            "Single linkage: the distance between two clusters is that of their closest pair. The loops already visit every pair; collect their distances, then take the smallest.",
            "The distance between two examples is the length of their difference, as in k-NN. Which built-in function gives the smallest number of a list?",
            "`distances.append(np.linalg.norm(___))` and `return ___(distances)`.",
        ],
        why="For `A` and `B` the closest pair is (1, 0) in `A` and (4, 0) in `B`: 3 apart. The farthest pair, (0, 0) and (5, 1), is 5.099 apart: that would be complete linkage.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_largest():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Every linkage

        The code reads the penguins as in the last lesson and goes through the four linkages. For each, it should build the tree of `X`, cut it into three clusters, and keep the size of the largest. Replace the two `...`. The last line shows `largest`.
        """),
        L.task("Find, for every linkage, how many penguins the largest of three clusters holds."),
    ])
    return


@app.cell(hide_code=True)
def exercise_largest():
    box_largest = L.code_box("largest", lines=16)
    box_largest
    return (box_largest,)


@app.cell(hide_code=True)
def check_largest(box_largest):
    result_largest = L.check("largest", box_largest.value)
    result_largest
    return (result_largest,)


@app.cell(hide_code=True)
def hints_largest():
    L.hints(
        "largest",
        [
            "Steps 2 and 4 of the recipe, as in the worked example: `linkage` builds the tree, `fcluster` cuts it.",
            "The linkage of this turn of the loop is in `method`. To cut into a number of clusters, `fcluster` needs `criterion=\"maxclust\"`.",
            "`Z = linkage(X, ___)` and `labels = fcluster(Z, ___, criterion=\"maxclust\")`.",
        ],
        why="With single linkage the three clusters hold 340, 1 and 1 penguins, so its largest is 340.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ### Exercise 3. A long chain
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="With single linkage, three clusters of the penguins hold 340, 1 and 1 penguins; with Ward 152, 126 and 64. Why does single linkage put almost every penguin in one cluster?",
        options={
            "outliers": "The penguins have two outliers so far away that no linkage can split the rest",
            "right": "Between the species lie penguins in between, and single linkage only looks at the closest pair, so a chain of short steps joins the species",
            "scale": "The features are not standardized, so one feature decides the distance",
        },
        correct="right",
        why={
            "outliers": "Ward and complete linkage split the same penguins into three real groups; the two lone penguins do not stop them. See the second picture.",
            "scale": "The code standardizes both features, as in the last lesson; the other linkages use the same distances. See *Exercise 2*.",
            "right": "Two clusters count as close as soon as one pair is close, however far apart the rest lies. That is chaining: good for clusters shaped like long bands, bad for the compact clumps of the species.",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Ward makes small the same within-cluster sum of squares as [Groups without labels: k-means](/course/open/ml-kmeans?lang=en), one merge at a time, and the dendrogram's long stretches do the job of the elbow from [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en). Single linkage follows chains of close examples, for better and for worse. In the next lesson, *Clusters of any shape: DBSCAN*, a cluster is a region where examples lie close together, of any shape, and an example far from every region is noise instead of a cluster of its own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chain, result_largest, result_link):
    L.wrap_up(result_link, result_largest, choice_chain)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
