# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-tsne. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-tsne", "en")
    C = L.checks   # the 1,000 test digits, their PCA map and t-SNE maps, the neighbor probabilities of one digit, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [The best shadow: PCA](/course/open/ml-pca?lang=en) and one about softmax, which your classifier of [Project: a linear classifier for handwritten digits](/course/open/ml-digits-linear?lang=en) used.
    """)
    return


@app.cell(hide_code=True)
def quick_check_pca():
    quick_pca = L.choice(
        "quick-pca",
        question="The first two principal components of 1,000 handwritten digits have an explained variance of 0.18 together. What does that mean?",
        options={
            "right": "The shadows of the digits on the plane of those two components keep 18% of the spread; 82% is lost",
            "digits": "18% of the digits land in the right place on that plane",
            "labels": "Those two components tell 18% of the digits apart",
        },
        correct="right",
        why={
            "digits": "Explained variance is about spread, not about single digits: every digit gets a shadow on the plane. See *The idea* in [The best shadow: PCA](/course/open/ml-pca?lang=en).",
            "labels": "PCA never sees a label, so it cannot say which digits it tells apart. Explained variance is the share of the spread the shadows keep. See *The idea* in [The best shadow: PCA](/course/open/ml-pca?lang=en).",
            "right": "So a picture of that plane shows less than a fifth of how the digits differ. Today a map that keeps something else.",
        },
    )
    quick_pca
    return (quick_pca,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="Softmax turns the numbers 2, 1 and 0 into three probabilities. Which is true?",
        options={
            "right": "They add up to 1, and 2 gets the largest",
            "share": "They are 2/3, 1/3 and 0: every number divided by the sum",
            "one": "2 gets 1 and the others 0",
        },
        correct="right",
        why={
            "share": "Softmax first takes e to the power of every number, and divides those by their sum: 0.665, 0.245 and 0.090. So 0 still gets a probability, and negative numbers work too. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "one": "That is the argmax, the class with the largest number. Softmax gives every number a probability: 0.665, 0.245 and 0.090. See [Many classes: softmax](/course/open/ml-softmax?lang=en).",
            "right": "0.665, 0.245 and 0.090. Today softmax turns distances into the probability that one example picks another as its neighbor.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A digit of QMNIST is 784 numbers, one per pixel: a point in 784 dimensions. To look at 1,000 of them at once, they have to go on a flat map. PCA puts them there as their best shadow on a plane, but for these digits that plane keeps only 17.8% of the spread, and you can see what is lost: on the PCA map, the nearest digit to a digit is the same digit for only 38.9% of the digits, against 86.4% in the 784 numbers. Groups that lie apart in 784 dimensions fall on top of each other in their shadow.

        **An everyday example.** A subway map does not keep the distances between stations, or the bends of the tracks. It keeps which station comes next to which, and that is what you need to find your way.

        The 1,000 test digits on the plane of their first two principal components. Every dot is a digit, and every number stands at the middle of the dots of that digit.
        """),
        mo.Html(C.picture_map(C.pca_map()["Z"], {"title": "PCA, 2 components", "x": "component 1", "y": "component 2",
                                                 "alt": "The 1,000 test digits as dots on the plane of their first two principal components, with the numbers 0 to 9 at the middles of their dots. The 1s lie at the left and the 0s at the right; the 2s, 3s, 5s, 6s and 8s lie on top of each other in the middle, and the 4s, 7s and 9s below them."})),
    ])
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same-guess",
        question="Today's method, t-SNE, draws the same 1,000 digits on a flat map too. On its map, for how many digits is the nearest digit the same digit?",
        options={
            "pca": "About 40%, as on the PCA map",
            "right": "About 85%, nearly as in the 784 numbers",
            "all": "100%: t-SNE puts every digit next to its own",
        },
        reveal="**About 85%.** With perplexity 30, 85.0%, against 86.4% in the 784 numbers and 38.9% on the PCA map. t-SNE does not look for the best shadow, but for a map on which every digit keeps its neighbors. It seldom does much better than the neighbors it is given: in the 784 numbers, 13.6% of the digits have another digit as their nearest neighbor.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Keep the neighbors.** **t-SNE**, short for t-distributed stochastic neighbor embedding, draws examples with many features on a flat map, and asks one thing of the map: that every example keeps its neighbors. It does that in three steps.

        1. **Neighbors in the 784 numbers.** Take one example and its distance $d_j$ to every other example $j$. Softmax turns the numbers $-d_j^2 / (2 \cdot \text{width}^2)$ into probabilities $p_j$ that add up to 1: how likely the example is to pick $j$ as its neighbor. Close examples get almost all of it. The width says how far the example looks: a small width gives nearly everything to the nearest few, a large width spreads it over many.
        2. **The perplexity sets the width.** The **perplexity** of the probabilities is $e^{-\sum_j p_j \log p_j}$. For k equal probabilities it is exactly k; for unequal ones it is about the number of neighbors that count. You choose one perplexity, such as 30, and t-SNE finds for every example the width that gives it that perplexity: small where examples lie close together, large where they lie far apart.
        3. **The map.** t-SNE starts with the examples on a flat map, works out the same kind of probabilities from their distances on the map, and moves them by gradient descent until those are close to the probabilities in the 784 numbers; the loss comes down to the cross-entropy between the two. On the map the probabilities fall off more slowly with distance than in step 1, along a curve with heavier tails called the t-distribution, the t in the name, so that groups have room to lie apart.

        Nothing in these steps keeps the size of a group, or the distance between two groups: only who is whose neighbor. Language models have a perplexity too, in a later lesson, with the same formula: there it counts the words a model in effect chooses between.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **t-SNE** | a method that draws examples with many features on a flat map, so that neighbors stay neighbors | `TSNE(n_components=2, perplexity=30, random_state=0)` |
        | **perplexity** | $e^{-\sum_j p_j \log p_j}$: about the number of neighbors that count for every example; you choose it, t-SNE finds the widths | `perplexity=30` |
        | `fit_transform(X)` | fits and gives back the map at once; t-SNE has no `transform`, so a new example cannot be put on a map that is already made | `TSNE(...).fit_transform(X)` |

        The picture shows the 1,000 test digits on maps of t-SNE, every dot a digit and every number at the middle of its dots. A map takes scikit-learn a while to make, so the course made these once, with `random_state=0`. The axes have no numbers: on a map of t-SNE they mean nothing. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_perplexity_cell():
    def _frame(p):
        said = f"Perplexity {p}: for {L.number(C.same(p) * 100, 1)}% of the digits, the nearest digit on the map is the same digit."
        alt = (f"The 1,000 test digits as dots on the map of t-SNE with perplexity {p}, with the numbers 0 to 9 at the middles of their dots. "
               + said)
        return (C.picture_map(C.tsne_map(p), {"title": f"t-SNE, perplexity {p}", "x": "map 1", "y": "map 2", "alt": alt}), said)

    explore_perplexity = L.explore("perplexity", "perplexity", _frame, start=30, values=C.PERPLEXITIES, role="hyper",
                                   task="Go from perplexity 2 to 100. Which two digits lie side by side on every map?")
    explore_perplexity
    return


@app.cell(hide_code=True)
def text_after_perplexity():
    mo.vstack([
        mo.md(r"""
        On every map the 9s lie nearest to the 4s. In the 784 numbers, too, 4s and 9s mix up more than any other two digits: 17 of the 110 4s have a 9 as their nearest neighbor, and 6 of the 94 9s a 4. For 84.2% to 85.4% of the digits the nearest digit on the map is the same digit, at every perplexity. Another seed, `random_state=1`, gives nearly the same map at perplexity 30, with 86.7%: scikit-learn starts a map from PCA, so the seed changes little.
        """),
        L.watch_out("The size of an island and the distance between islands mean little. In the 784 numbers the 1s lie closest together, 4.1 from their mean on average, against 5.8 to 6.8 for every other digit; yet at perplexity 2 and 5 the islands of the 0s, 6s and 7s are smaller than theirs. And at perplexity 30 the 4s lie farthest from the 0s, while in the 784 numbers the 1s do."),
        mo.md(r"""
        What one digit sees. The first test digit is a 7. The picture shows its 30 largest neighbor probabilities, from the largest down, for the width on the slider: filled bars for neighbors that are 7s too, open bars for other digits. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_width_cell():
    def _frame(w):
        f = C.width_facts(w)
        said = (f"Width {L.number(w, 1)}: perplexity {L.number(f['perplexity'], 1)}. The nearest neighbor gets {L.number(f['top'], 3)}, "
                f"{f['over']} neighbors get more than 0.01, and the other 7s get {L.number(f['same'], 3)} together.")
        alt = (f"Bars of the 30 largest neighbor probabilities of the first test digit, a 7, for width {L.number(w, 1)}, on an axis "
               "from 0 to 1, filled for neighbors that are 7s and open for other digits. " + said)
        return (C.picture_width(w, {"title": f"Neighbors of a 7, width {L.number(w, 1)}", "x": "neighbor, nearest first",
                                    "y": "probability", "alt": alt}), said)

    explore_width = L.explore("width", "width", _frame, start=1.2, values=C.WIDTHS, role="hyper",
                              task="Go from width 1.2 to 2.5. Which width gives a perplexity of about 30?")
    explore_width
    return


@app.cell(hide_code=True)
def text_after_width():
    mo.md(r"""
    With a small width the 7 looks at its nearest neighbor almost alone: at 1.2 the perplexity is 1.3. The larger the width, the more neighbors share the probability, and the more of it goes to other digits: at 2.5 the perplexity is 195.4, and the other 7s get only 0.594. Width 2.0 gives 28.6, about the 30 of the maps above. t-SNE finds such a width for every example by halving: too many neighbors, a smaller width; too few, a larger one.

    ### Recipe

    To look at examples with many features on a flat map:

    1. Put the examples in rows of numbers on one scale; here every pixel goes from 0 to 1.
    2. Choose a perplexity, about the number of neighbors that count for every example: 30, the default of scikit-learn, is a good start.
    3. Make the map: `TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(X)`.
    4. Make it again with other perplexities, and trust only what every map shows: which examples lie together.
    5. Check what you see against the examples themselves, such as how often the nearest neighbor has the same label in the original numbers.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first 300 test digits. The cell makes a map with PCA and one with t-SNE, and prints, for the 784 numbers and for each map, the share of the digits whose nearest digit is the same digit, rounded to two decimals. The comments with `#` say which step of the recipe each part is. t-SNE takes a few seconds. Run the cell.
    """)
    return


@app.cell
def example_tsne():
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # 1. the first 300 test digits, 784 numbers from 0 to 1 each, and which digit each is
    _data = np.load("data/qmnist-small.npz")
    _X = _data["test_images"][:300] / 255
    _y = _data["test_labels"][:300]

    def _same_neighbor(Z):
        # 5. the share of the digits whose nearest other digit, in Z, is the same digit
        same = 0
        for i in range(len(Z)):
            j = np.argsort(np.linalg.norm(Z - Z[i], axis=1))[1]
            same = same + int(_y[j] == _y[i])
        return same / len(Z)

    # 2. and 3. a map with PCA, and one with t-SNE with perplexity 30
    _pca_map = PCA(n_components=2).fit_transform(_X)
    _tsne_map = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(_X)

    # 5. the 784 numbers, the PCA map and the t-SNE map
    print(round(_same_neighbor(_X), 2), round(_same_neighbor(_pca_map), 2), round(_same_neighbor(_tsne_map), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `np.argsort` puts the distances in order; place 0 is the digit itself, at distance 0, so place 1 is its nearest other digit. In the 784 numbers, the nearest digit is the same digit for 0.82 of these 300; on the PCA map for 0.4, and on the map of t-SNE again for 0.82. The 1,000 digits of the slider give the same picture: 0.864, 0.389 and 0.85.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_probs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Neighbor probabilities

        `neighbor_probs(d, width)` should give back the neighbor probabilities of one example: `d` holds its distance to every other example. Replace the two `...`: the number for every neighbor, $-d^2 / (2 \cdot \text{width}^2)$, and the probabilities softmax makes of `e`. Every number in `z` is 0 or less, so `np.exp` cannot overflow here, and you need not subtract the largest first. The check calls `neighbor_probs(d, 1.0)` with `d` = [1, 2, 3], and tries four more: another width, four far neighbors, four at the same distance, and one neighbor alone.
        """),
        L.task("Make `neighbor_probs` turn distances into neighbor probabilities."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probs():
    box_probs = L.code_box("neighbor-probs", lines=7)
    box_probs
    return (box_probs,)


@app.cell(hide_code=True)
def check_probs(box_probs):
    result_probs = L.check("neighbor-probs", box_probs.value)
    result_probs
    return (result_probs,)


@app.cell(hide_code=True)
def hints_probs():
    L.hints(
        "neighbor-probs",
        [
            "Every neighbor gets minus its squared distance divided by 2 times the width squared: a larger distance gives a smaller number. NumPy works it out for all of `d` at once.",
            "`d ** 2` squares every distance. The probabilities are `e` divided by its sum, as in softmax.",
            "`z = -d ** 2 / (2 * ___ ** 2)` and `return e / np.sum(___)`.",
        ],
        why="With width 1, `z` is −0.5, −2 and −4.5; `e` is 0.607, 0.135 and 0.011, together 0.753; divided by that: 0.806, 0.180 and 0.015.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_perplexity():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Perplexity

        `perplexity(p)` should give back the perplexity of the probabilities `p`. Replace the two `...`: `h`, minus the sum of every probability times its log, and the perplexity, e to the power of `h`. The check calls `perplexity(four)` with four probabilities of 0.25, and tries four more: 0.7, 0.2 and 0.1; two of 0.5; one sure neighbor, 0.97, with three of 0.01; and thirty equal ones.
        """),
        L.task("Make `perplexity` count the neighbors that count."),
    ])
    return


@app.cell(hide_code=True)
def exercise_perplexity():
    box_perplexity = L.code_box("perplexity", lines=6)
    box_perplexity
    return (box_perplexity,)


@app.cell(hide_code=True)
def check_perplexity(box_perplexity):
    result_perplexity = L.check("perplexity", box_perplexity.value)
    result_perplexity
    return (result_perplexity,)


@app.cell(hide_code=True)
def hints_perplexity():
    L.hints(
        "perplexity",
        [
            "`h` is the cross-entropy of `p` against itself: every probability times its log, added up, with a minus in front. The perplexity is e to the power of `h`.",
            "Work on the whole array at once: NumPy multiplies `p` by its log value by value, and has a function for a sum and one for e to a power.",
            "`h = -np.sum(p * ___)` and `return np.exp(___)`.",
        ],
        why="Four of 0.25: the log of 0.25 is −1.386, so `h` is −4 · 0.25 · (−1.386) = 1.386, and e to the power 1.386 is 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_islands():
    mo.md(r"""
    ### Exercise 3. Reading the map
    """)
    return


@app.cell(hide_code=True)
def choice_islands_cell():
    choice_islands = L.choice(
        "islands",
        question="On the map with perplexity 30, the island of the 4s lies right next to that of the 9s, and the 0s lie farthest from the 4s. What can you conclude?",
        options={
            "far": "The 0s and the 4s are the two digits that differ most",
            "right": "Many 4s and 9s are each other's neighbors in the 784 numbers; how far the 0s lie from the 4s, the map does not promise",
            "nothing": "Nothing: t-SNE puts the islands anywhere",
        },
        correct="right",
        why={
            "far": "In the 784 numbers, the mean of the 1s lies farther from that of the 0s, 7.34, than the mean of the 4s does, 6.08, and the maps with perplexity 2 and 5 do put the 1s farthest from the 0s. How far apart islands lie changes with the perplexity: t-SNE keeps neighbors, not distances. See the warning after the first slider.",
            "nothing": "t-SNE keeps neighbors: for 85.0% of the digits, the nearest digit on the map is the same digit, nearly the 86.4% of the 784 numbers. Neighbors on the map are neighbors in the data. See the first slider.",
            "right": "4s and 9s mix up more than any other two digits: 23 times one is the nearest neighbor of the other. That is what t-SNE keeps, so their islands touch on every map. The distance between islands that are not neighbors is not what it keeps.",
        },
    )
    choice_islands
    return (choice_islands,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** t-SNE keeps what [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) relies on, who is whose neighbor, where the PCA of [The best shadow: PCA](/course/open/ml-pca?lang=en) keeps the most spread. Its neighbor probabilities are the softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en), and its perplexity is e to the power of a cross-entropy. Islands on its map are a reason to look closer, not clusters: check them with a clustering such as k-means, and in the data itself. In the next lesson, *Bought together: support, confidence and lift*, you look for products that often end up in the same order.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_islands, result_perplexity, result_probs):
    L.wrap_up(result_probs, result_perplexity, choice_islands)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
