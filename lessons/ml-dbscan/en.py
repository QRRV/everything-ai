# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-dbscan. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-dbscan", "en")
    C = L.checks   # the 400 points, DBSCAN for every eps, k-means and single linkage on them, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Clusters inside clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=en) and one about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_single():
    quick_single = L.choice(
        "quick-single",
        question="Single linkage counts two clusters as close as soon as…",
        options={
            "right": "one example of the one lies close to one example of the other",
            "means": "their means lie close together",
            "all": "every example of the one lies close to every example of the other",
        },
        correct="right",
        why={
            "means": "Single linkage looks at pairs of examples, not at means: it takes the distance of the closest pair. See *The idea* in [Clusters inside clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=en).",
            "all": "That is complete linkage, which takes the farthest pair. Single linkage takes the closest pair. See *The idea* in [Clusters inside clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=en).",
            "right": "So a chain of close examples joins two groups, however far apart the rest lies. Today a method that follows shapes too, but does not let a few stray examples chain groups together.",
        },
    )
    quick_single
    return (quick_single,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="k-NN measures the distance between penguins by bill length, tens of millimeters, and body mass, thousands of grams. What happens without standardizing?",
        options={
            "error": "k-NN cannot compare millimeters with grams and stops with an error",
            "right": "The distance is almost only body mass",
            "nothing": "Nothing: k-NN only asks which examples are nearest, not how far away they are",
        },
        correct="right",
        why={
            "error": "NumPy subtracts and squares the numbers whatever their unit. The trouble is that a difference of hundreds of grams outweighs one of a few millimeters. See the warning in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "nothing": "Which examples are nearest depends on the distance, and that is almost only body mass. See the warning in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "So standardize first. Today's points have two features on the same scale, so they can stay as they are.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    k-means puts every example with its nearest centroid, so its clusters are round clumps around a center, and Ward looks for the same. Single linkage can follow a shape of any form, but every example has to belong somewhere: a stray example becomes a cluster of its own, and a few strays between two groups chain them together. Real groups are not always round, and real data has examples that belong to no group, such as a mistyped measurement or a rare event. A cluster can also be a region where examples lie close together, whatever its shape, with the examples outside every such region left over as noise.

    **An everyday example.** Fly over a country at night. A town is where lights lie close together, whether it lies along a river or around a lake, and a lone farmhouse light far from everything is not a town.
    """)
    return


@app.cell(hide_code=True)
def predict_shapes():
    guess_shapes = L.guess(
        "shapes-guess",
        question="The course made 400 points with two features: a ring of 200, a clump of 80 in its middle, a curved band of 100 beside it, and 20 strays anywhere. You run k-means with k = 3 on them. What does it find?",
        options={
            "shapes": "The ring, the clump and the band",
            "right": "The band, and the ring cut in two halves, each with part of the clump",
            "strays": "Three groups of strays",
        },
        reveal="**The band, and two halves of the ring.** k-means makes round clusters around three centroids: 99 points of the ring with 63 of the clump, the other 101 points of the ring with the other 17, and the band with 10 strays. Single linkage, cut into three clusters, does follow the shapes, but the strays bridge them: 398 points in one cluster, and two strays alone.",
    )
    guess_shapes
    return (guess_shapes,)


@app.cell(hide_code=True)
def picture_others_cell():
    mo.vstack([
        mo.md(r"""
        The 400 points, on the left in the three clusters of k-means, on the right in those of single linkage. Here color and shape mean a cluster.
        """),
        mo.Html(C.picture_others({"kmeans": "k-means, k = 3", "single": "Single linkage, 3 clusters", "x": "feature 1", "y": "feature 2",
                                  "alt": "Two charts of the 400 points: a ring, a clump in its middle, a curved band to the right and strays. Left, k-means with k = 3: the band is one cluster, and the ring is cut into two halves, each with part of the clump. Right, single linkage cut into 3 clusters: 398 points in one cluster and two strays alone."})),
    ])
    return


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Dense regions.** **DBSCAN** is a form of **density-based clustering**: a cluster is a region where examples lie close together, of any shape, and an example in no such region is noise. Its name is short for density-based spatial clustering of applications with noise. Two settings say what close together means: a radius, `eps`, and a number, `min_samples`.

        - The neighborhood of an example is every example within `eps` of it, itself included.
        - A **core point** has at least `min_samples` examples in its neighborhood.
        - A cluster starts at a core point and takes in its whole neighborhood. Every core point it takes in adds its own neighborhood, and so on, until nothing new comes in. So a cluster can grow along a ring or a band, as long as core points follow one another.
        - An example in a cluster that is not a core point itself lies at its edge: a border point. It joins the cluster, but adds nothing to it.
        - A **noise point** lies within `eps` of no core point. It belongs to no cluster: scikit-learn gives it the label −1.

        There is no k: the number of clusters comes out. And where single linkage joins two clusters through any pair of close examples, DBSCAN only grows through core points, so a stray example between two shapes does not join them.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **density-based clustering** | clustering in which a cluster is a region where examples lie close together, of any shape | |
        | **DBSCAN** | density-based clustering that grows clusters from core points and leaves the rest as noise | `DBSCAN(eps=0.4, min_samples=5)` |
        | `eps` | the radius of a neighborhood | `eps=0.4` |
        | `min_samples` | how many examples a neighborhood needs, itself included, for a core point | `min_samples=5` |
        | **core point** | an example with at least `min_samples` examples within `eps` | `db.core_sample_indices_` |
        | **noise point** | an example within `eps` of no core point: in no cluster | the label `-1` in `db.labels_` |
        """),
        L.watch_out("One `eps` holds for the whole data. Where one cluster is less dense than another, no `eps` may fit both: at `eps` = 0.3 the clump and the band are whole, while the ring, whose points lie further apart, falls apart into four pieces."),
        mo.md(r"""
        The picture shows DBSCAN with `min_samples` = 5 on the 400 points. Every cluster has its own color and shape: the three largest each their own, all others the fourth. Core points are large, border points small, and noise points are open gray circles. The <span class="c-hyper">dashed circle</span> is the neighborhood of one point of the ring, with radius `eps`. Here color and shape mean a cluster. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_eps_cell():
    def _frame(eps):
        f = C.fit(eps)
        sizes = (f"of {C.listed(f['sizes'], L.lang)} points" if f["clusters"] <= 3
                 else f"the largest of {f['sizes'][0]} points")
        said = f"`eps` = {L.number(eps, 1)}: {f['clusters']} clusters, {sizes}, and {f['noise']} noise points."
        alt = ("The 400 points: a ring, a clump in its middle, a curved band to the right and strays, in the clusters of DBSCAN "
               f"with min_samples 5 and eps {L.number(eps, 1)}, with a dashed circle of that radius around one point of the ring. "
               + said.replace("`", ""))
        return (C.picture(eps, {"title": f"DBSCAN, eps = {L.number(eps, 1)}", "x": "feature 1", "y": "feature 2", "alt": alt}), said)

    explore_eps = L.explore("eps", "eps", _frame, start=0.1, values=C.EPS, role="hyper",
                            task="Go from `eps` = 0.1 to 1.0. From which `eps` on are the ring, the clump and the band each one cluster, and at which `eps` do two of them become one?")
    explore_eps
    return


@app.cell(hide_code=True)
def text_after_eps():
    mo.md(r"""
    With `eps` = 0.1 most points have too few neighbors: 264 noise points, and 15 small clusters. At 0.2 the clump and the band are almost whole, but the ring is still in pieces. At 0.3 the ring falls apart into four pieces, of 72, 52, 50 and 27 points: its points lie further apart than those of the clump and the band. From 0.4 to 0.7 DBSCAN finds the three shapes, and its 15 to 13 noise points are all strays; the other strays lie close enough to a shape to join it. At 1.0 the ring and the clump become one cluster of 288 points. Three clusters hold over a wide range of `eps`, from 0.4 to 0.7: a good sign that they are real.

    ### Recipe

    To find clusters of any shape, and noise:

    1. Standardize the features, unless they are on the same scale, as these two are.
    2. Choose `min_samples`: 5, the default of scikit-learn, is a good start; more for large data with much noise.
    3. Try a range of `eps`, and look at the number of clusters, the number of noise points and the clusters themselves. A number of clusters that holds over a range of `eps` is a good sign.
    4. `DBSCAN(eps=..., min_samples=...).fit(X).labels_` gives every example its cluster, and −1 for noise.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The six numbers of the last lesson again: 1, 2, 4, 8, 9 and 15, each an example with one feature, with `eps` = 2. The cell runs DBSCAN with `min_samples` 2 and 3, and prints, for each, `min_samples`, the cluster of every example and the places of the core points. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_dbscan():
    import numpy as np
    from sklearn.cluster import DBSCAN

    # 1. six examples with one feature: nothing to standardize
    _points = np.array([[1.0], [2.0], [4.0], [8.0], [9.0], [15.0]])

    # 2. and 4. DBSCAN with eps 2, for two values of min_samples: the cluster of every example, and the core points
    for _m in [2, 3]:
        _db = DBSCAN(eps=2, min_samples=_m).fit(_points)
        print(_m, _db.labels_, _db.core_sample_indices_)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Within 2 of each other lie 1 and 2, 2 and 4, and 8 and 9; 15 has no neighbor. With `min_samples` = 2, every number with a neighbor is a core point, the five at places 0 to 4: 1, 2 and 4 grow into cluster 0, 8 and 9 into cluster 1, and 15 is noise, −1. With `min_samples` = 3, only 2 has enough: 1, 2 and 4, itself included, so the core points are just `[1]`, the place of the 2. Its cluster still takes in 1 and 4, as border points, but 8 and 9 have no core point between them and become noise, with 15.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_core():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Core points

        `core_points(X, eps, min_samples)` should give back, for every row of `X`, whether it is a core point: `True` or `False`. The loop already works out, for every example `x`, its distance to every example, itself included. Replace the two `...`. The check calls `core_points(X, 0.6, 3)` with eight points, and tries five more: a grid whose neighbors lie exactly `eps` apart, the numbers of the worked example with `min_samples` 2 and 3, `min_samples` 1, and three features.
        """),
        L.task("Make `core_points` say which examples are core points."),
    ])
    return


@app.cell(hide_code=True)
def exercise_core():
    box_core = L.code_box("core-points", lines=10)
    box_core
    return (box_core,)


@app.cell(hide_code=True)
def check_core(box_core):
    result_core = L.check("core-points", box_core.value)
    result_core
    return (result_core,)


@app.cell(hide_code=True)
def hints_core():
    L.hints(
        "core-points",
        [
            "A core point has at least `min_samples` examples within `eps`, itself included. `distances` holds the distance from `x` to every example: how many of them are at most `eps`?",
            "A comparison on an array gives an array of True and False, and `np.sum` counts the Trues. Then compare that count with `min_samples`.",
            "`count = np.sum(distances ___ eps)` and `core.append(count ___ min_samples)`.",
        ],
        why="Of the eight points, (0, 0) has itself, (0.5, 0) and (0, 0.5) within 0.6: 3, so it is a core point. (1, 0) has itself and (0.5, 0): 2, one too few.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_samples():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Five values of min_samples

        In this box, `X` is ready: the 400 points of the picture. The loop runs DBSCAN with `eps` = 0.4 for five values of `min_samples`. Replace the two `...`: how many noise points there are, and how many clusters. The last line shows `found`.
        """),
        L.task("Find, for every `min_samples`, how many clusters and noise points DBSCAN gives."),
    ])
    return


@app.cell(hide_code=True)
def exercise_samples():
    box_samples = L.code_box("min-samples", lines=10)
    box_samples
    return (box_samples,)


@app.cell(hide_code=True)
def check_samples(box_samples):
    result_samples = L.check("min-samples", box_samples.value)
    result_samples
    return (result_samples,)


@app.cell(hide_code=True)
def hints_samples():
    L.hints(
        "min-samples",
        [
            "Noise points have the label −1; the clusters have the labels 0, 1, 2 and up.",
            "`labels == -1` gives a True or False for every point, and `np.sum` counts the Trues. For the clusters, count the different labels, but leave out −1.",
            "`n_noise = int(np.sum(labels ___ -1))` and `n_clusters = len(np.unique(labels[labels ___ 0]))`.",
        ],
        why="With `min_samples` 5, as on the slider at `eps` = 0.4: 3 clusters and 15 noise points.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_single():
    mo.md(r"""
    ### Exercise 3. One example is enough
    """)
    return


@app.cell(hide_code=True)
def choice_single_cell():
    choice_single = L.choice(
        "single",
        question="With `min_samples` = 1, every example is a core point, because its neighborhood holds itself. Which clusters does DBSCAN then give?",
        options={
            "kmeans": "The clusters of k-means",
            "right": "The clusters of single linkage, with the tree cut at height `eps`",
            "noise": "None: every example is noise, because none has enough neighbors",
        },
        correct="right",
        why={
            "kmeans": "k-means needs k and centroids, and DBSCAN has neither. With `min_samples` = 1, every example within `eps` of another joins it. See *Exercise 2*, where `min_samples` 1 gives 16 clusters.",
            "noise": "An example lies in its own neighborhood, so with `min_samples` = 1 every example is a core point and none is noise. See *Exercise 2*: with `min_samples` 1 there is no noise at all.",
            "right": "Two examples within `eps` of each other end up in one cluster, and so does any chain of such steps: exactly the clusters single linkage has when every merge above height `eps` is undone. On the 400 points with `eps` = 0.4 both give the same 16 clusters. So `min_samples` is what stops the chaining: an example in between only links two clusters when it has enough neighbors itself.",
        },
    )
    choice_single
    return (choice_single,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** DBSCAN asks about every example what [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) asks about a new one, which examples lie near, but within a radius `eps` instead of the nearest k. With `min_samples` = 1 it is the single linkage of [Clusters inside clusters: hierarchical clustering](/course/open/ml-hierarchical-clustering?lang=en), cut at `eps`; a larger `min_samples` stops the chaining, and leaves lone examples as noise instead of clusters of their own. In the next lesson, *A map of many dimensions: t-SNE*, you draw examples with many features on a flat map, where groups like these become visible.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_single, result_core, result_samples):
    L.wrap_up(result_core, result_samples, choice_single)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
