# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-otsu-threshold. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-otsu-threshold", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_equalize():
    quick_equalize = L.choice(
        "quick-equalize",
        question="In a grayscale photo, 30 of every 100 pixels have gray 90 or a darker one. Which gray does 90 become after histogram equalization?",
        options={
            "share": "About 76",
            "same": "Still 90",
            "thirty": "30",
        },
        correct="share",
        why={
            "share": "Gray 90 gets 255 times its share in the cumulative histogram: 255 × 0.3 = 76.5, and `astype(np.uint8)` makes that 76.",
            "same": "Equalization gives every gray a new gray. Which share does it multiply by 255? See The idea in [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en).",
            "thirty": "30 of every 100 is the share, 0.3. By which number does equalization multiply the share? See [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en).",
        },
    )
    quick_equalize
    return (quick_equalize,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Half of the pixels of a group have gray 60, the other half gray 70. What is the variance of their grays?",
        options={
            "25": "25",
            "5": "5",
            "50": "50",
        },
        correct="25",
        why={
            "25": "The mean is 65, and both grays lie 5 away from it: 5² × 0.5 + 5² × 0.5 = 25.",
            "5": "5 is the distance of each gray to the mean, 65, and so the standard deviation. The variance squares the distances first: see The idea in [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "50": "That is 5² + 5², without the probabilities. Each gray has probability 0.5: see [Expected value and variance](/course/open/prob-expectation?lang=en).",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To count cells under a microscope or read the letters of a scanned page, a program first has to know which pixels belong to the object and which to the background. The simplest way is one gray as a dividing line, a threshold: the pixels on one side of it are the object, the rest is background, and the result has only black and white. But the light differs from one photo to the next, so one fixed gray does not work for all of them. In 1979, Nobuyuki Otsu published a method that reads the threshold off the histogram of each image; it is still widely used, and scikit-image and OpenCV both have it built in.

    **An everyday example.** You sort a pile of laundry into a dark wash and a light wash. A good split makes every piece look as much as possible like the rest of its own pile: no pale shirt among the jeans, no gray sock among the white towels. Otsu's method looks for the gray that splits the pixels that way.
    """)
    return


@app.cell(hide_code=True)
def predict_mean():
    guess_mean = L.guess(
        "mean",
        question="The photo of this lesson is a cell in salt water, under a microscope: a light disk that covers about 3 of every 100 pixels, on a gray background. A first idea for a threshold is the mean gray of the photo. With `img > img.mean()`, how many of the pixels turn white?",
        options={
            "cell": "About 3 of every 100: just the cell",
            "half": "About half",
            "all": "Nearly all",
        },
        reveal="**About half**: 48 of every 100. The background is 97 of every 100 pixels, so the mean, about 68, lies in the middle of the grays of the background itself, and the lighter half of the background turns white along with the cell. A good threshold lies between the two groups of grays, not in the middle of the biggest one.",
    )
    guess_mean
    return (guess_mean,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A point operation gives every pixel a new gray that depends only on its own gray: not on its neighbors, and not on where it is. Histogram equalization was one: a table with a new gray for every gray. Thresholding is one of the simplest point operations: every pixel lighter than the threshold `t` becomes white, 255, and every other pixel black, 0. The result is a binary image, with only two values. In NumPy, `img > t` gives True for every pixel lighter than `t` and False for the rest, in the shape of the image. Python counts True as 1 and False as 0, so `255 * (img > t)` gives 255 and 0, and `(img > t).mean()`, the mean of ones and zeros, is the share of the pixels lighter than `t`.

        Otsu's method chooses `t` by itself. Every threshold splits the pixels into two groups: the dark pixels, with gray `t` or a darker one, and the light pixels, lighter than `t`. A good threshold puts every pixel close to the mean of its own group. Otsu measures that with the variance of each group, weighted by its share of the pixels, $w$, and added up:
        """),
        L.formula(r"w_\text{dark} \cdot \mathrm{Var}_\text{dark} + w_\text{light} \cdot \mathrm{Var}_\text{light}"),
        mo.md(r"""
        That is the within-group variance: for every pixel, the squared distance to the mean of its own group, averaged over all the pixels. The shares weight the two variances the way probabilities weight values in an expected value, so a big group counts for much and a small group for little. Otsu tries every threshold, and takes the one where the within-group variance is smallest.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **point operation** | every pixel gets a new gray that depends only on its own gray, not on its neighbors or its place | `table[img]` |
        | **thresholding** | every pixel lighter than the threshold `t` becomes white, every other pixel black | `img > t` |
        | **binary image** | an image with only two values, such as 0 and 255, or False and True | `255 * (img > t)` |
        | **within-group variance** | the variances of the dark and of the light pixels, each weighted by the share of the pixels in its group, added up | `w_dark * dark.std() ** 2 + w_light * light.std() ** 2` |
        | **Otsu's method** | try every threshold, and take the one with the smallest within-group variance: there the pixels lie, on average, closest to the mean of their own group | `thresholds[np.argmin(withins)]` |
        | $w_\text{dark}$ | the share of the pixels in the dark group, from 0 to 1 | `(img <= t).mean()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_thresholds():
    mo.vstack([
        mo.md("**The cell, with two thresholds**"),
        mo.Html(C.threshold_picture({"photo": "The photo", "by_mean": "The mean as threshold, 68", "by_otsu": "Otsu's threshold, 122",
                                     "histogram": "Histogram, cut off at 300 pixels", "gray": "gray",
                                     "mean_line": "mean, 68", "otsu_line": "Otsu, 122",
                                     "alt": "At the top, three pictures. The photo: a light disk, the cell, on a gray background with faint stripes. The binary image with the mean gray as the threshold: the cell is white, and so is about half of the background, in irregular stripes and specks. The binary image with Otsu's threshold, 122: only the cell is white, a clean disk on black. Under them, the histogram of the photo: very high bars from about gray 10 to 80, the background, cut off at the top; low bars from about 85 to 120; then a wide hill of bars from about 125 to 235, the cell. A dashed line marks the mean, 68, in the middle of the background bars; a solid line marks Otsu's threshold, 122, in the low part between the two groups."})),
        mo.md(r"""
        At the top, the photo and two binary images. With the mean gray, about 68, as the threshold, about half of the pixels turn white: the cell, and the lighter half of the background, in stripes that the microscope left in the photo. With Otsu's threshold, 122, only the cell is white. Under them, the histogram: the bars of the background, from about gray 10 to 80, go far above the top, up to 28,907 pixels for one gray; the cut at 300 keeps the low bars of the cell, from about 125 to 235, in sight. The mean lies in the middle of the background; Otsu's threshold lies in the low part between the two groups.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`img > t` gives True and False, not grays. With `vmin=0, vmax=255`, as you used for histogram equalization, `plt.imshow` counts True as 1, almost black, so the whole image looks black. Make grays of it first: `255 * (img > t)`."),
        L.watch_out("Otsu's method always gives a threshold, also for an image with only one group of grays, such as a photo of the background alone: it then splits that one group in two. And one threshold for the whole image fails when the light is uneven: a dark corner of a page can be darker than the letters elsewhere."),
        mo.md(r"""
        ### Recipe

        To find Otsu's threshold of a grayscale image `img`:

        1. Make the list of thresholds, from the darkest gray of the image up to the lightest, the lightest not included: `thresholds = list(range(img.min(), img.max()))`. Then both groups always have pixels. Go through them with `for t in thresholds:`.
        2. Split the pixels: `dark = img[img <= t]` and `light = img[img > t]`.
        3. Work out the share of each group, `w_dark = (img <= t).mean()` and `w_light = (img > t).mean()`, and then the within-group variance: `w_dark * dark.std() ** 2 + w_light * light.std() ** 2`. The variance is the square of the standard deviation. Add it to a list, `withins`.
        4. Take the threshold with the smallest within-group variance: `best = thresholds[np.argmin(withins)]`.
        5. Make the binary image with that threshold: `255 * (img > best)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The photo is `data/photos/cell.png`: a cell in salt water, recorded with a microscope that measures how much each point slows the light down. Light goes slower through the cell than through the water, so the cell is lighter. The photo has one gray per pixel. The first code cell reads it and tries the mean gray as the threshold; the second follows the recipe and measures the cell. Run the two code cells.
    """)
    return


@app.cell
def example_mean():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/cell.png"))
    print(img.shape, img.min(), img.max())
    # a first try: the mean gray as the threshold, and the share of white pixels
    print(img.mean(), (img > img.mean()).mean())
    return img, np, plt


@app.cell
def example_otsu(img, np, plt):
    # 1. every threshold from the darkest gray up to the lightest
    thresholds = list(range(img.min(), img.max()))
    withins = []
    for t in thresholds:
        # 2. the dark and the light pixels
        dark = img[img <= t]
        light = img[img > t]
        # 3. the share of each group, and the within-group variance
        w_dark = (img <= t).mean()
        w_light = (img > t).mean()
        withins.append(w_dark * dark.std() ** 2 + w_light * light.std() ** 2)
    # 4. the threshold with the smallest within-group variance
    best = thresholds[np.argmin(withins)]
    print(best, min(withins), img.std() ** 2)
    # the share and the variance of each group at that threshold
    print((img <= best).mean(), img[img <= best].std() ** 2)
    print((img > best).mean(), img[img > best].std() ** 2)
    # 5. the binary image, and how many columns have a white pixel: the width of the cell
    cell = 255 * (img > best)
    print((cell.max(axis=0) == 255).sum())
    plt.imshow(cell, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The photo is 550 pixels wide and 660 high, with grays from 0 to 255. Its mean gray is about 68, and with that as the threshold, 48% of the pixels turn white. Otsu's method tries the 255 thresholds from 0 to 254 and finds 122, with a within-group variance of about 152, against about 571 for all the pixels together. The two groups are not both tight: the background, 96.8% of the pixels, has variance 135.5, but the cell, 3.2%, has 640.2, more than the whole photo, because its grays run from about 125 to 235. Weighted by their shares, 0.968 × 135.5 + 0.032 × 640.2 is about 152: the big group counts most. With 122, only the cell is white, one clean disk. It is 124 pixels wide, and each pixel covers 0.107 micrometers, so the cell is about 13 micrometers across: 13 thousandths of a millimeter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_point():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A point operation
    """)
    return


@app.cell(hide_code=True)
def choice_point_cell():
    choice_point = L.choice(
        "point",
        question="Which of these three changes of a grayscale image is a point operation?",
        options={
            "negative": "Every gray g becomes 255 − g: `255 - img`",
            "blur": "Every pixel gets the mean gray of itself and its eight neighbors",
            "flip": "The image is flipped from left to right: `img[:, ::-1]`",
        },
        correct="negative",
        why={
            "negative": "The new gray of a pixel depends only on its own gray: 200 becomes 55, wherever that pixel is. Black becomes white and white becomes black: a negative. Thresholding and histogram equalization work the same way.",
            "blur": "Here the new gray also depends on the neighbors: the same gray can come out lighter or darker, depending on what lies around it. Which pixels does a point operation look at?",
            "flip": "Every pixel gets the gray of another pixel, on the other side: its own gray says nothing about its new one. Which pixel does a point operation look at?",
        },
    )
    choice_point
    return (choice_point,)


@app.cell(hide_code=True)
def text_exercise_binary():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A binary image

        Finish `binary(img, t)`. It gets a grayscale image and a threshold, and gives back the binary image: 255 for every pixel lighter than `t`, and 0 for every other pixel. Replace the `...`. The check calls it with an image of 6 pixels and threshold 100:

        ```python
        binary([[30, 120, 200], [90, 160, 60]], 100)
        ```

        That should give `[[0, 255, 255], [0, 255, 0]]`. It also tries other images and thresholds.
        """),
        L.task("Make `binary(img, t)` give 255 for every pixel lighter than `t`, and 0 for the rest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_binary():
    box_binary = L.code_box("binary", lines=6)
    box_binary
    return (box_binary,)


@app.cell(hide_code=True)
def check_binary(box_binary):
    result_binary = L.check("binary", box_binary.value)
    result_binary
    return (result_binary,)


@app.cell(hide_code=True)
def hints_binary():
    L.hints(
        "binary",
        [
            "One line is enough: compare the image with `t`, then make numbers of what comes out.",
            "`img > t` gives True for every pixel lighter than `t`. Python counts True as 1: what do you multiply it by to get 255?",
            "`return ___ * (img > t)`.",
        ],
        why="`img > 100` gives `[[False, True, True], [False, True, False]]`, and 255 times that is `[[0, 255, 255], [0, 255, 0]]`. A pixel of exactly 100 would stay 0: it is not lighter than 100.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_lowest():
    mo.md(r"""
    ### Exercise 3. The lowest point

    The slider sets the threshold for the photo of the cell. On the left, the binary image with that threshold; on the right, the within-group variance of every threshold from 0 to 254, with the chosen threshold marked by a dot. At the bottom of the valley the curve is nearly flat, so read the number in the sentence under the picture.
    """)
    return


@app.cell(hide_code=True)
def explore_lowest_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"binary": "threshold {t}", "threshold": "threshold", "within": "within-group variance",
                                          "alt": "On the left, the binary image of the cell with threshold {t}; on the right, the within-group variance of every threshold, a curve that is high at both ends, with a small dip near 55 and a wide, flat valley between about 80 and 145, with threshold {t} marked by a dot."})
        return (_html,
                f'Threshold {t}: the within-group variance is {L.number(_f["within"], 1)}, and {L.number(100 * _f["white"], 1)}% of the pixels are white.')

    explore_lowest = L.explore("lowest", "threshold", _frame, start=66, values=C.THRESHOLDS, role="param",
                               task="Find the threshold where the within-group variance is smallest: Otsu's threshold.")
    explore_lowest
    return (explore_lowest,)


@app.cell(hide_code=True)
def check_lowest(explore_lowest):
    result_lowest = L.check("lowest", explore_lowest)
    result_lowest
    return (result_lowest,)


@app.cell(hide_code=True)
def hints_lowest():
    L.hints(
        "lowest",
        [
            "Move the slider one stop at a time, and read the within-group variance in the sentence under the picture.",
            "From 66 on, the within-group variance goes down, and somewhere in the valley it starts to go up again: look for the stop where it turns.",
            "Look at the stops 114, 122 and 130: which one has the smallest within-group variance?",
        ],
        why="At 114 the within-group variance is 152.2, at 122 it is 151.8, and at 130 152.5. So 122 is Otsu's threshold, the same one the worked example found by trying every gray.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_otsu():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Otsu's method

        Finish `otsu(img)`. It gets a grayscale image and gives back Otsu's threshold. The lines around the `...` already try every threshold from the darkest gray up to the lightest, and at the end take the threshold with the smallest number in `withins`. Replace the `...` with what happens for each threshold `t`: the dark and the light pixels, their shares, and the within-group variance, added to the list with `withins.append(...)`. The check calls it with an image of 20 pixels, a background of grays 2 to 5 with an object of grays 6 to 9 in it:

        ```python
        otsu([[2, 3, 3, 4, 3], [3, 8, 9, 5, 2], [4, 9, 8, 6, 3], [3, 7, 3, 2, 4]])
        ```

        That should give `5`. It also tries other images.
        """),
        L.task("Make `otsu(img)` give the threshold with the smallest within-group variance."),
    ])
    return


@app.cell(hide_code=True)
def exercise_otsu():
    box_otsu = L.code_box("otsu", lines=14)
    box_otsu
    return (box_otsu,)


@app.cell(hide_code=True)
def check_otsu(box_otsu):
    result_otsu = L.check("otsu", box_otsu.value)
    result_otsu
    return (result_otsu,)


@app.cell(hide_code=True)
def hints_otsu():
    L.hints(
        "otsu",
        [
            "Follow steps 2 and 3 of the recipe, inside the loop: the two groups, their two shares, and then `withins.append(...)`.",
            "`dark = img[img <= t]` and `light = img[img > t]`. The share of the dark pixels is `(img <= t).mean()`. The variance of a group is the square of its standard deviation: `dark.std() ** 2`.",
            "`withins.append(w_dark * dark.std() ** 2 + w_light * ___)`.",
        ],
        why="At threshold 5, the dark group has 14 of the 20 pixels, with variance 0.69, and the light group the 6 pixels of grays 6 to 9, with variance 1.14: 14/20 × 0.69 + 6/20 × 1.14 is about 0.83. At threshold 4 the within-group variance is 0.99, and at 6 it is 1.01, so 5 has the smallest.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en), a table gave every gray a new gray: a point operation, as thresholding is. In [Expected value and variance](/course/open/prob-expectation?lang=en), probabilities weighted the squared distances; here the shares of the pixels weight the variances of the two groups. Otsu's method chooses one number by making a measure of how bad a split is as small as possible. In [Rolling downhill: gradient descent](/course/open/ml-gradient-descent-1d?lang=en), a model made its loss small step by step, following the slope; Otsu has only one number to choose, out of 255, so it can simply try them all. Binary images come back later in this module: *Grow and shrink shapes: erosion and dilation* cleans them up, and *Count the shapes: connected components* counts the white shapes in them, such as cells. In the next lesson, *Less noise: the mean, the Gaussian and the median filter*, the new gray of a pixel also depends on its neighbors: that is no longer a point operation.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_point, result_binary, result_lowest, result_otsu):
    L.wrap_up(choice_point, result_binary, result_lowest, result_otsu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
