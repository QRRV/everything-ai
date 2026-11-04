# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-pca. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pca", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Expected value and variance](/course/open/prob-expectation?lang=en) and [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Four shadows are −2, −1, 1 and 2, all equally common, so their mean is 0. What is their variance?",
        options={
            "zero": "0",
            "right": "2.5",
            "abs": "1.5",
        },
        correct="right",
        why={
            "zero": "0 is their mean: the shadows below 0 cancel those above. The variance is the mean squared distance to the mean, and squares do not cancel. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "abs": "1.5 is the mean distance to the mean, without squaring. The variance takes the mean of the squared distances. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "right": "(4 + 1 + 1 + 4) / 4 = 2.5. With the mean at 0, the variance is just the mean of the squares, as it will be for the shadows in this lesson.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Why can the smallest within-cluster sum of squares not choose k?",
        options={
            "labels": "Because it needs labels",
            "right": "Because the best sum only gets smaller as k grows",
            "negative": "Because it can be below 0",
        },
        correct="right",
        why={
            "labels": "It needs no labels: only the examples and their centroids. See [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "negative": "A sum of squares is never below 0. See [How many clusters? Choosing k](/course/open/ml-choosing-k?lang=en).",
            "right": "So the smallest is always the largest k you try, and you look for the elbow or the silhouette score instead.",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A penguin with four measurements is four numbers, and four numbers do not fit in one picture. A photo of 28 by 28 pixels is 784 numbers. To look at such data, or to give a model fewer numbers to work with, you want fewer numbers that keep as much of the data as they can. This lesson finds them for the penguins, without their species, as in the last two lessons.

    **An everyday example.** A lamp casts the shadow of a pencil on the wall: three dimensions become two. Hold the pencil with its tip to the lamp, and its shadow is a dot; hold it across, and the shadow shows its whole length. Some shadows show more than others, and you can turn the pencil until its shadow shows the most.
    """)
    return


@app.cell(hide_code=True)
def predict_kept():
    guess_kept = L.guess(
        "kept",
        question="The 342 penguins by bill length and flipper length, in millimeters. You draw one line through their middle and replace every penguin by its shadow on that line: one number instead of two. How much of the spread of the penguins can the best line keep?",
        options={
            "half": "About half",
            "threequarters": "About three quarters",
            "right": "More than 90%",
        },
        reveal="**93.0%.** The best line runs mostly along flipper length: a line along bill length alone keeps 13.1%, a line along flipper length alone 86.9%. Flipper lengths vary by more millimeters, and penguins with long flippers mostly have long bills too, so one number holds most of what two say.",
    )
    guess_kept
    return (guess_kept,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Dimensionality reduction** describes every example with fewer numbers, keeping as much as it can. The simplest way is a shadow. First subtract the mean of every feature, so that the examples lie around 0. Then take a direction $\mathbf{w}$ of length 1. The shadow of an example $\mathbf{x}$ on the line through 0 in that direction is the dot product $\mathbf{x} \cdot \mathbf{w}$: how far along the line the example lies. Shadow is an image: in maths this number is the projection of $\mathbf{x}$ on $\mathbf{w}$, and it marks the point of the line nearest to $\mathbf{x}$. The shadows lie around 0 too, so their spread, their variance, is the mean of their squares.

        From the shadow to the example is a right angle, so by Pythagoras every example's squared length is its squared shadow plus its squared distance to the line:

        $$\mathbf{x} \cdot \mathbf{x} = (\mathbf{x} \cdot \mathbf{w})^2 + d^2$$

        Take the mean over all examples: the spread of all of them is the spread the shadows keep plus the mean squared distance to the line. What the shadows keep, the line does not lose. So the direction whose shadows keep the most is also the line nearest to the examples. Unlike in linear regression, the distance is measured at right angles to the line, not straight up: no feature is the one to predict.

        **Principal component analysis**, PCA, finds that direction: the first **principal component**. The second is the direction at right angles to the first whose shadows keep the most of what is left, and so on, as many as there are features. The **explained variance** of a component is the share of the spread that its shadows keep; over all components it adds up to 1. Keep only the first few, and every example is described by its shadows on them.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **dimensionality reduction** | describing every example with fewer numbers, keeping as much as possible | |
        | shadow | for an example $\mathbf{x}$ minus the mean, and a direction $\mathbf{w}$ of length 1: $\mathbf{x} \cdot \mathbf{w}$ | `M @ w` |
        | **principal component analysis**, PCA | finding the directions whose shadows keep the most spread, one after another, each at right angles to the earlier ones | `PCA(n_components=2).fit(X)` |
        | **principal component** | one of those directions, a vector of length 1 | `pca.components_[0]` |
        | **explained variance** | the share of the spread that the shadows on a component keep | `pca.explained_variance_ratio_` |
        | `pca.transform(X)` | the shadows of every example on the components, one column for each | |
        | `pca.explained_variance_` | without `_ratio`: the spread of the shadows on each component itself, not the share | |
        """),
        L.watch_out("PCA never sees a label: it keeps the most spread, and the most spread is not always what tells groups apart. And the sign of a component means nothing: (0.267, 0.964) and (−0.267, −0.964) are the same line, and the shadows only change sign."),
        mo.md(r"""
        The picture shows the 342 penguins by bill length and flipper length, in millimeters from their mean, with both axes on the same scale. The orange line goes through the mean at the angle of the slider: 0° runs along bill length, 90° along flipper length. Every penguin's shadow on the line is in blue, and the thin red lines are the distances to the line. On the right, for every angle, the mean of the squared red lines, in purple. Move the slider, or drag across the right chart.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_angle_cell():
    def _frame(angle):
        angle = int(angle)
        f = C.shadow(angle)
        said = f"""At {angle}°: the shadows keep {L.number(100 * f["share"], 1)}% of the spread, {L.number(f["kept"], 1)} of the {L.number(f["total"], 1)} mm². The mean squared distance to the line is the rest, {L.number(f["lost"], 1)} mm²."""
        alt = (f"The first chart: the 342 penguins by bill length and flipper length, in millimeters from their mean, with a line through the mean at {angle} degrees, the shadow of every penguin on it, and the distance from every penguin to the line. "
               + "The second chart: the mean squared distance to the line for every angle from 0 to 175 degrees; it is smallest at 75 degrees and largest at 165. "
               + said)
        return (C.picture(angle, {"title": f"The line at {angle}°", "every": "Every angle", "angle": "angle (°)", "lost": "mean squared distance (mm²)",
                                  "x": "bill length, from the mean (mm)", "y": "flipper length, from the mean (mm)", "alt": alt}), said)

    explore_angle = L.explore("angle", "angle", _frame, start=0, values=C.ANGLES, role="param",
                              task="Turn the line. At which angle are the red lines shortest, and how much of the spread do the shadows keep there?")
    explore_angle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Along bill length, at 0°, the shadows keep only 13.1%: bill lengths vary by only about 5 mm. Along flipper length, at 90°, they keep 86.9%. The best angle is 75°: the shadows keep 93.0%, and the red lines are shortest, with a mean square of 15.8 mm². At right angles to it, at 165°, the shadows keep only 7.0%: that is the direction of the second principal component.

    ### Recipe

    To describe examples with fewer numbers with PCA:

    1. Choose the features, and standardize them if their units differ.
    2. Subtract the mean of every feature, so that the examples lie around 0.
    3. For a direction $\mathbf{w}$ of length 1, the shadows are the dot products `M @ w`; their spread is the mean of their squares.
    4. The first principal component is the direction whose shadows keep the most spread; every next one keeps the most of what is left, at right angles to the earlier ones. `PCA` finds them.
    5. The explained variance of a component is the spread of its shadows divided by the spread of all. Keep as many components as you need.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The recipe on bill length and flipper length. `_total` is the spread of all penguins: the mean of their squared lengths, `np.sum(_M ** 2, axis=1)` for every penguin. Then three directions of length 1, made with the norm: along bill length, along flipper length, and halfway between. For each, the direction, the spread of its shadows in mm², and the share of `_total`. At the end the same for the first principal component of scikit-learn, and its `explained_variance_ratio_`. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_pca():
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["flipper_length_mm"].isna() == False]
    # 1. both features are in millimeters, so no standardizing
    _raw = np.array(_df[["bill_length_mm", "flipper_length_mm"]])
    # 2. subtract the mean
    _M = _raw - _raw.mean(axis=0)
    _total = np.mean(np.sum(_M ** 2, axis=1))
    print(np.round(_total, 1))
    # 3. three directions of length 1, and the spread of their shadows
    for _d in [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]:
        _w = _d / np.linalg.norm(_d)
        _shadow = _M @ _w
        print(np.round(_w, 3), np.round(np.mean(_shadow ** 2), 1), np.round(np.mean(_shadow ** 2) / _total, 3))
    # 4. the first principal component, from scikit-learn
    _pca = PCA(n_components=2).fit(_M)
    _w = _pca.components_[0]
    _shadow = _M @ _w
    # 5. the share of the spread that its shadows keep
    print(np.round(_w, 3), np.round(np.mean(_shadow ** 2), 1), np.round(np.mean(_shadow ** 2) / _total, 3))
    print(np.round(_pca.explained_variance_ratio_, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The spread of all penguins is 226.9 mm². Along bill length the shadows keep 29.7 of it, along flipper length 197.2: those are the variances of the two features, and 29.7 + 197.2 = 226.9, so the spread of all is the variance of every feature added up. Halfway between, (0.707, 0.707), the shadows keep 163.7. The first principal component is (0.267, 0.964), at the 75° of the picture, and its shadows keep 211.1 mm²: 0.93 of the spread, the same as `explained_variance_ratio_[0]`. The second keeps the other 0.07.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The share of the first component

        Steps 3 to 5 of the worked example, now on bill length and bill depth, both in millimeters. The box subtracts the mean in `M`, computes `total`, and fits `PCA`; `w` is the first principal component. Replace the two `...`: the shadow of every penguin on `w`, and the share of the spread those shadows keep. The last line puts your share next to `pca.explained_variance_ratio_[0]`.
        """),
        L.task("Compute the share of the spread that the first principal component keeps."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("first-share", lines=15)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("first-share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "first-share",
        [
            "Steps 3 and 5 of the worked example, with `M`, `w` and `total` instead of `_M`, `_w` and `_total`.",
            "The shadow of every penguin is the dot product with `w`, for all penguins at once with `@`. Their spread is the mean of their squares, and the share is that divided by `total`.",
            "`shadow = M @ ___` and `share = np.mean(shadow ** ___) / total`.",
        ],
        why="`M @ w` gives one number for every penguin: how far along `w` it lies. The shadows lie around 0, because `M` does, so the mean of their squares is their spread; divided by the spread of all, it is the share that the line keeps.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_many():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How many components?

        Now all four measurements, standardized in `X4`, because grams and millimeters cannot be compared. The box fits `PCA` with 4 components. The loop adds up the explained variance of the components one by one in `kept`, and stops once it has at least 0.9. Replace the two `...`: what is added for component `n`, and when the loop stops. The last line gives how many components that took, and what they keep.
        """),
        L.task("Find how many principal components keep at least 90% of the spread."),
    ])
    return


@app.cell(hide_code=True)
def exercise_many():
    box_many = L.code_box("how-many", lines=16)
    box_many
    return (box_many,)


@app.cell(hide_code=True)
def check_many(box_many):
    result_many = L.check("how-many", box_many.value)
    result_many
    return (result_many,)


@app.cell(hide_code=True)
def hints_many():
    L.hints(
        "how-many",
        [
            "`pca.explained_variance_ratio_` has a share for each of the 4 components. Which one belongs to `n`, and when have you kept enough?",
            "Add `pca.explained_variance_ratio_[n]` to `kept`, and stop with `break` once `kept` is 0.9 or more.",
            "`kept = kept + pca.explained_variance_ratio_[___]` and `if kept >= ___:`.",
        ],
        why="The components come in order: the first keeps the most, every next one less. Adding them up one by one, the loop stops at the fewest components that keep 90%; `n` counts from 0, so that is `n + 1` components.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shadows():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. k-means on two shadows

        The box fits `PCA` with 2 components on `X4`. Replace the two `...`: the shadows of every penguin on the two components, with `pca.transform`, and what `KMeans` is fitted on. The loop then counts, as in the k-means lesson, the penguins of the species most common in each cluster.
        """),
        L.task("Cluster the penguins on their shadows on two components, and count how many are in the cluster of their own species."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shadows():
    box_shadows = L.code_box("two-shadows", lines=19)
    box_shadows
    return (box_shadows,)


@app.cell(hide_code=True)
def check_shadows(box_shadows):
    result_shadows = L.check("two-shadows", box_shadows.value)
    result_shadows
    return (result_shadows,)


@app.cell(hide_code=True)
def hints_shadows():
    L.hints(
        "two-shadows",
        [
            "The new words of *The idea* say what gives the shadows of every example on the components.",
            "`Z` is `pca.transform(X4)`: two numbers for every penguin. k-means then clusters `Z` instead of `X4`.",
            "`Z = pca.transform(___)` and `km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(___)`.",
        ],
        why="`pca.transform(X4)` subtracts the mean and takes the dot product of every penguin with each component, so `Z` describes every penguin with two numbers instead of four. k-means only sees those two, and finds the groups that they keep.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Exercise 4. Why standardize first?
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "why-scale",
        question="Without standardizing, PCA on the four measurements finds a first component of about (0.004, −0.001, 0.015, 1.000): bill length, bill depth, flipper length and body mass. Its explained variance is 0.9999. Why?",
        options={
            "species": "Body mass tells the species apart best",
            "right": "Body mass is in grams: it varies by about 800 g, flipper length by about 14 mm, so in squares body mass makes up almost all of the spread",
            "last": "PCA puts the most weight on the last feature",
        },
        correct="right",
        why={
            "species": "PCA never sees the species: it looks only at the spread. And body mass alone does not tell Adelies from Chinstraps.",
            "last": "The order of the features does not matter to PCA; how much each varies does. Look at their units.",
            "right": "The spread counts in squares: about 641,251 g² for body mass against 197 mm² for flipper length. Any direction away from body mass loses far more than it gains, so the first component is body mass. Standardized, every feature has a spread of 1, and PCA finds what the four have in common.",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The shadow is the dot product of [Many features: the dot product](/course/open/math-dot-product?lang=en), for all penguins at once with the `@` of [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en), and the spread is the variance of [Expected value and variance](/course/open/prob-expectation?lang=en). Like [Groups without labels: k-means](/course/open/ml-kmeans?lang=en), PCA never sees a label, so it keeps what varies most, not what you are after: the third component of the four measurements keeps only 9% of the spread, yet Adelies and Chinstraps differ on it about as much as on each of the first two. This ends the module *Without labels*. In the next lesson, *A neuron*, the neural networks begin.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_many, result_shadows, result_share):
    L.wrap_up(result_share, result_many, result_shadows, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
