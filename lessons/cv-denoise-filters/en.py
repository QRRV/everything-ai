# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-denoise-filters. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-denoise-filters", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From light to a file of numbers](/course/open/cv-image-files?lang=en) and [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_difference():
    quick_difference = L.choice(
        "quick-difference",
        question="`a` and `b` are two grayscale images of uint8. Why does `abs(a.astype(int) - b.astype(int)).mean()` measure how far apart they are, and `abs(a - b).mean()` does not?",
        options={
            "wrap": "uint8 cannot go below 0: 3 − 5 wraps around to 254",
            "abs": "`abs` only works on ordinary whole numbers",
            "mean": "`.mean()` of uint8 gives a whole number, so the decimals are lost",
        },
        correct="wrap",
        why={
            "wrap": "uint8 holds only the whole numbers 0 to 255, so 3 − 5 comes out as 254 instead of −2, and `abs` cannot repair that. As ordinary whole numbers, the difference is −2, and `abs` makes it 2.",
            "abs": "`abs` works on an array of any numbers. What happens to 3 − 5 when both are uint8? See the recipe of [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "mean": "`.mean()` of uint8 gives a float with decimals. The trouble is earlier, in the subtraction: what is 3 − 5 in uint8? See [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
        },
    )
    quick_difference
    return (quick_difference,)


@app.cell(hide_code=True)
def quick_check_feature_map():
    quick_feature_map = L.choice(
        "quick-feature-map",
        question="A filter of 3 by 3 slides over an image of 6 by 6 pixels, and stays inside the image. How big is the feature map?",
        options={
            "4": "4 by 4",
            "6": "6 by 6",
            "3": "3 by 3",
        },
        correct="4",
        why={
            "4": "The filter fits in 6 − 3 + 1 = 4 places across, and 4 down.",
            "6": "Then the filter would also sit on the outer pixels, where part of it hangs over the edge. How many places does it fit in, across? See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
            "3": "That is the size of the filter. In how many places across does it fit, inside 6 pixels? See [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en).",
        },
    )
    quick_feature_map
    return (quick_feature_map,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Every photo has some noise, and in dim light you see it: the sensor catches so little light that chance decides much of how much each pixel gets, so each pixel comes out a bit lighter or darker than the scene was. A faulty pixel in a sensor, or an error while the numbers are sent or stored, gives specks of pure black or white. The noise gets in the way of what comes next in this module: a program that looks for edges sees every grain as a tiny edge, and a threshold turns specks into shapes that are not there. So many image programs first filter the noise out, and they try to keep what matters: the edges.

    **An everyday example.** Five people read a thermometer that shows 19.5 degrees from across a room, and call out 19, 20, 18, 21 and 20. Their mean, 19.6, is closer to 19.5 than any single reading: one reads a bit high, another a bit low, and in the mean that cancels out. If a sixth person calls out 90, as a joke, the mean of the six jumps to about 31; their median stays at 20. Two of the filters of this lesson work like the mean, and one like the median.
    """)
    return


@app.cell(hide_code=True)
def predict_smear():
    guess_smear = L.guess(
        "smear",
        question="The photo of this lesson is the coffee cup of [From light to a file of numbers](/course/open/cv-image-files?lang=en), in grays. One copy gets specks: about 1 in every 10 pixels becomes pure black or pure white. A mean filter gives every pixel the mean gray of the window of 3 by 3 pixels around it. Does it bring the speckled photo closer to the clean one?",
        options={
            "much": "Much closer: the specks are averaged away",
            "hardly": "Hardly closer",
            "further": "Further away",
        },
        reveal="**Hardly closer**: before the filter, the photo is on average 12.6 grays off; after it, 12.0. The mean filter spreads a speck out: a white speck in an area of gray 170 adds (255 − 170) / 9, about 9, to each of the nine windows it is in. The bright dot is gone, but nine pixels are now each about 9 off, together as much as the one speck was. The median filter, later in this lesson, brings it down to 2.3.",
    )
    guess_smear
    return (guess_smear,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Image noise is a random change of the grays: every pixel shows what the scene gave, plus or minus a bit. This lesson makes two kinds on purpose, so it can measure what a filter takes away. *Grain*: every pixel gets a random number added, with mean 0 and standard deviation 25. Such random numbers have the bell-shaped spread of the normal distribution: most are small, a few are large. The mathematician Carl Friedrich Gauss worked a lot with that bell curve; the Gaussian filter below is named after him, and the later lesson *Common distributions* covers the normal distribution itself. Some grays then fall below 0 or above 255, which an image file could not store; the lesson keeps them, and the pictures show them as black and white. *Specks*: a few pixels become pure black, 0, or pure white, 255. (Later in the course, diffusion models add exactly this kind of grain to images on purpose, and learn to take it away again.)

        A filter works on one small window at a time, as the filter of [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en) did: the window of 3 by 3 pixels that starts at row `r` and column `c` gives the pixel `out[r, c]`. That window has the pixel `img[r + 1, c + 1]` in its middle: `out[r, c]` is the new gray of that pixel. The **mean filter** takes the mean of the nine grays. That is a convolution with a filter of nine weights of 1/9: the window times the filter, added up, is the mean. Grain makes one pixel lighter and the next one darker, so in the mean of nine pixels much of it cancels out. But at an edge, the window holds dark and light pixels together, and their mean is a gray in between: the edge blurs.

        The **Gaussian filter** is a convolution too, with weights that are largest in the middle and fall off along a bell curve:
        """),
        L.formula(r"\frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}"),
        mo.md(r"""
        The pixel in the middle counts 4 times, the pixels above, below, left and right of it 2 times, the corners once. Divided by 16, the nine weights add up to 1. Because the outer pixels count for less, the Gaussian filter blurs less than the mean filter, and it also cancels a little less grain.

        The **median filter** takes the median of the nine grays: sort them, and take the middle one, the 5th. That is not a convolution: no set of weights times the window gives the median. A speck is one odd value among nine, and the median ignores it, as the median of [Describing data: mean, median and spread](/course/open/stats-describe?lang=en) ignored an outlier. At an edge, most of the window lies on one side, so the median is a gray of that side, and the edge stays sharp.
        """),
        L.watch_out("Filter means more here than in [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en): there it was the grid of weights. In image processing, any rule that makes each new pixel from the window around it is called a filter. The mean and the Gaussian filter are such a grid of weights; the median filter has no weights at all."),
        mo.md(r"""
        The three filters on an edge from 0 to 255, in every row of the image:

        | | the grays across the edge |
        |---|---|
        | the edge | 0, 0, 0, 255, 255, 255 |
        | after the mean filter | 0, 85, 170, 255 |
        | after the Gaussian filter | 0, 64, 191, 255 |
        | after the median filter | 0, 0, 255, 255 |

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **image noise** | a random change of the grays: every pixel a bit lighter or darker than the scene, or specks of pure black and white | `rng.normal(0, 25, size=clean.shape)` |
        | **mean filter** | every pixel becomes the mean of the window around it: a convolution with equal weights | `window.mean()` |
        | **Gaussian filter** | every pixel becomes a weighted mean of the window around it, with weights that add up to 1 and fall off along a bell curve | `(window * K).sum()` |
        | **median filter** | every pixel becomes the median of the window around it; it removes specks and keeps edges sharp | `np.median(window)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_filters():
    mo.vstack([
        mo.md("**Grain and specks, and three filters**"),
        mo.Html(C.filter_picture({"grainy": "Grain", "speckled": "Specks", "mean": "Mean filter", "gauss": "Gaussian filter", "median": "Median filter",
                                  "alt": "Eight close-ups of the same part of the photo: the coffee at the top left, the white rim of the cup, the dark outside of the cup below it, the handle of the spoon, and the gray saucer on the right. Top row: the photo with grain, a fine sandy noise over everything; after the mean filter, smoother but softer at the edges; after the Gaussian filter, a little less smooth and a little sharper; after the median filter, still mottled, with blotches of a few pixels in the flat parts. Bottom row: the photo with specks, black and white dots everywhere; after the mean filter, the dots have become small gray smudges; after the Gaussian filter, smaller and sharper smudges; after the median filter, the dots are gone and the edges are sharp."})),
        mo.md(r"""
        The same part of the photo each time, 100 pixels high and 130 wide, magnified: the coffee, the rim of the cup, the handle of the spoon and the saucer. At the top with grain: the mean and the Gaussian filter make it smoother, and the edges softer; after the median filter it stays mottled. At the bottom with specks: the mean and the Gaussian filter turn every speck into a gray smudge; the median filter removes the specks and keeps the edges sharp.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("A filter of 3 by 3 makes the image 2 pixels smaller in each direction, as in [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en): a window fits only where a pixel has a neighbor on every side. So compare the result with the clean photo without its outer pixels: `clean[1:-1, 1:-1]`, from the second row up to, but not including, the last, and the same for the columns. Libraries such as SciPy and OpenCV fill in the border, so the image keeps its size."),
        L.watch_out("A filter cannot tell noise from fine detail: the pattern of the wood in the table blurs away together with the noise. A bigger window removes more noise, and more detail with it; exercise 3 looks for the balance."),
        mo.md(r"""
        ### Recipe

        To filter a grayscale image `img` with a window of 3 by 3:

        1. Make an array of zeros for the result, 2 smaller in each direction: `out = np.zeros((len(img) - 2, len(img[0]) - 2))`.
        2. Go through every place, with `for r in range(len(out)):` and inside it `for c in range(len(out[0])):`, and take the window that starts there: `window = img[r:r + 3, c:c + 3]`.
        3. Fill in the place: for the mean filter `out[r, c] = window.mean()`, for the Gaussian filter `out[r, c] = (window * K).sum()` with `K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16`, and for the median filter `out[r, c] = np.median(window)`.
        4. Measure how far the result is from the clean photo, on average: `abs(out - clean[1:-1, 1:-1]).mean()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The photo is the coffee cup of [From light to a file of numbers](/course/open/cv-image-files?lang=en): `data/photos/coffee.png`. The example makes it gray by taking, for every pixel, the strongest of its red, green and blue: `img.max(axis=2)`, where `axis=2` is the third direction of the array, the colors. It keeps the part with the cup, rows 0 to 300 and columns 100 to 500, so the loops stay quick. Then it makes the noise with a random generator with a seed, as in [Random numbers and seeds](/course/open/np-random?lang=en): `rng.normal(0, 25, size=clean.shape)` gives random numbers with mean 0 and standard deviation 25, one for every pixel. The first code cell makes the grain and the specks; the second filters three photos, each with the three filters at once: the clean one, the grainy one and the speckled one. Run the two code cells; the second takes a few seconds.
    """)
    return


@app.cell
def example_noise():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    # the photo in grays, the strongest color of every pixel, and the part with the cup
    clean = img.max(axis=2)[0:300, 100:500]
    rng = np.random.default_rng(0)
    # grain: every gray plus a random number with mean 0 and standard deviation 25
    grainy = clean + rng.normal(0, 25, size=clean.shape)
    # specks: a random number from 0 to 1 for every pixel; below 0.05 black, above 0.95 white
    specks = rng.random(size=clean.shape)
    speckled = np.array(clean)   # a new array, so that clean stays clean
    speckled[specks < 0.05] = 0
    speckled[specks > 0.95] = 255
    # how far each noisy photo is from the clean one, on average
    # grainy is already floats, because random floats were added; speckled is still uint8, like clean
    print(clean.shape, abs(grainy - clean).mean(), abs(speckled.astype(int) - clean.astype(int)).mean())
    return clean, grainy, np, plt, speckled


@app.cell
def example_filters(clean, grainy, np, plt, speckled):
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


    def three_filters(photo):
        # 1. three arrays of zeros, 2 smaller in each direction
        mean = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        gauss = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        median = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        # 2. every window of 3 by 3
        for r in range(len(mean)):
            for c in range(len(mean[0])):
                window = photo[r:r + 3, c:c + 3]
                # 3. the mean, the weights of the Gaussian filter, the median
                mean[r, c] = window.mean()
                gauss[r, c] = (window * K).sum()
                median[r, c] = np.median(window)
        return mean, gauss, median


    # 4. how far each result is from the clean photo without its outer pixels
    inner = clean[1:-1, 1:-1]
    for photo in [clean, grainy, speckled]:
        mean, gauss, median = three_filters(photo)
        print(abs(mean - inner).mean(), abs(gauss - inner).mean(), abs(median - inner).mean())
    # the median filter of the speckled photo, the last one; with vmin=0 and vmax=255, 0 is black and 255 white,
    # as in the photo, and without them the darkest gray would become black
    plt.imshow(median, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The part with the cup is 300 pixels high and 400 wide. The grain moves the grays 20.0 on average, less than its standard deviation of 25: the standard deviation squares the differences first, so the few large ones weigh more. The specks move them 12.6. The three lines below say how far each photo is from the clean one after the mean, the Gaussian and the median filter. The first line is what a filter costs: none of our noise is in the clean photo, yet the mean filter moves its grays 2.9 on average, the Gaussian filter 2.3 and the median filter 1.9. About half of that is the blur at the edges; the rest is fine detail, such as the wood of the table, and the slight noise the photo already had. On the grain, the mean filter gets to 7.7, the Gaussian filter to 8.2 and the median filter to 9.1: averaging nine pixels cancels most of the grain, the Gaussian filter cancels a bit less of it but blurs less, and the median uses only the middle one of the nine grays, so more of the grain is left. On the specks, the mean and the Gaussian filter hardly help, 12.0 and 11.9, while the median filter gets to 2.3: it leaves the specks out. Its result is the picture under the cell, almost the clean photo again.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The weights of the Gaussian filter
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "weights",
        question="The Gaussian filter of this lesson has the weights 1, 2, 1 / 2, 4, 2 / 1, 2, 1, divided by 16. Why 16?",
        options={
            "sum": "16 is the sum of the weights: divided by 16 they add up to 1, so a patch that is all one gray keeps that gray",
            "pixels": "16 is the number of pixels the filter looks at",
            "middle": "Dividing by 16 makes the middle pixel count most",
        },
        correct="sum",
        why={
            "sum": "1 + 2 + 1 + 2 + 4 + 2 + 1 + 2 + 1 = 16. On a patch of the photo where all nine pixels have gray 100, the filter gives 100 × 16 / 16 = 100: the gray stays the same. The weights of the mean filter, nine times 1/9, also add up to 1.",
            "pixels": "The window is 3 by 3: how many pixels is that? Add up the nine weights, and compare with 16.",
            "middle": "The middle pixel counts most because its weight, 4, is the largest; dividing every weight by the same number does not change which one is largest. Without dividing, what would a patch where every pixel has gray 100 become?",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The mean filter

        Complete `mean_filter(img)`. It gets a grayscale image and gives back the image after a mean filter of 3 by 3, 2 smaller in each direction. The loops over every place are already there: replace the `...` with what happens at each place `r, c`. The check calls it with an edge from 0 to 9, four rows of five pixels:

        ```python
        mean_filter([[0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9],
                     [0, 0, 0, 9, 9]])
        ```

        That should give `[[0, 3, 6], [0, 3, 6]]`. It also tries other images.
        """),
        L.task("Make `mean_filter(img)` give every place the mean of its window of 3 by 3."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean", lines=10)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean",
        [
            "One line is enough: take the window of 3 by 3 that starts at row `r` and column `c`, and store its mean in `out[r, c]`.",
            "The window has the rows `r`, `r + 1` and `r + 2`: a slice from `r` that stops just before `r + 3`. The columns go the same way, after a comma in the same brackets. An array has a method that gives its mean.",
            "`out[r, c] = img[r:r + 3, ___].mean()`.",
        ],
        why="The first window has only zeros: mean 0. The second has in each row 0, 0, 9: the mean is 27 / 9 = 3. The third has 0, 9, 9 in each row: 54 / 9 = 6. Both rows of the result are the same, because all four rows of the image are.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ### Exercise 3. How big a window?

    A mean filter can have a bigger window: 5 by 5 takes the mean of 25 pixels. With the slider you choose the size for the grainy photo of the worked example. On the left, the same part as in the picture, after the mean filter of that size; on the right, for every size, how far the result is from the clean photo, on average, with the chosen size as a dot.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(k):
        _html, _f = C.size_frame(k, {"window": "window {k} by {k}", "size": "window size", "off": "grays off the clean photo, on average",
                                     "alt": "On the left, the grainy part of the photo after a mean filter of {k} by {k}; on the right, how far the result is from the clean photo for the window sizes 1 to 15: about 20 at 1, steeply down to a low point between 3 and 9, then slowly up again, with size {k} marked by a dot."})
        return (_html,
                f'Window {k} by {k}: the filtered photo is on average {L.number(_f["off"], 1)} grays off the clean one.')

    explore_size = L.explore("size", "window", _frame, start=1, values=C.SIZES, role="hyper",
                             task="Find the window size where the filtered photo comes closest to the clean one.")
    explore_size
    return (explore_size,)


@app.cell(hide_code=True)
def check_size(explore_size):
    result_size = L.check("size", explore_size)
    result_size
    return (result_size,)


@app.cell(hide_code=True)
def hints_size():
    L.hints(
        "size",
        [
            "Move the slider one stop at a time, and read the number in the sentence under the picture.",
            "From 1 on, the number goes down fast: a bigger window cancels more grain. Somewhere it starts to go up again, because the edges blur: look for the stop where it turns.",
            "Look at the stops 3, 5 and 7: which one gives the smallest number?",
        ],
        why="With 3 by 3 the photo is 7.7 off, with 5 by 5 it is 6.5, and with 7 by 7 it is 6.7. So 5 by 5 comes closest: a smaller window leaves more grain, a bigger one blurs more.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_median():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. The median filter

        Complete `median_filter(img)`. It gets a grayscale image and gives back the image after a median filter of 3 by 3, 2 smaller in each direction. This time only the array of zeros is there: write the loops over every place yourself, as in exercise 2, and give every place the median of its window. The check calls it with an image with a white speck at the top left and an edge between gray 10 and gray 200:

        ```python
        median_filter([[10, 10, 10, 200, 200],
                       [10, 255, 10, 200, 200],
                       [10, 10, 10, 200, 200],
                       [10, 10, 10, 200, 200]])
        ```

        That should give `[[10, 10, 200], [10, 10, 200]]`. It also tries other images.
        """),
        L.task("Make `median_filter(img)` give every place the median of its window of 3 by 3."),
    ])
    return


@app.cell(hide_code=True)
def exercise_median():
    box_median = L.code_box("median", lines=10)
    box_median
    return (box_median,)


@app.cell(hide_code=True)
def check_median(box_median):
    result_median = L.check("median", box_median.value)
    result_median
    return (result_median,)


@app.cell(hide_code=True)
def hints_median():
    L.hints(
        "median",
        [
            "Follow steps 2 and 3 of the recipe: two loops, the window, and the median of the window in `out[r, c]`.",
            "The loops are `for r in range(len(out)):` and, one level further in, `for c in range(len(out[0])):`. The median of an array is `np.median(...)`.",
            "Inside the two loops: `out[r, c] = np.median(img[___, ___])`.",
        ],
        why="The first window holds eight times 10 and the speck, 255: sorted, the 5th gray is 10, so the speck is gone. The third window holds three times 10 and six times 200: the 5th is 200. The edge stays one step, from 10 to 200. The mean filter would give 37.2, 100.6 and 136.7 in each row: the speck a gray smudge, and the edge a slope.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The mean and the Gaussian filter are convolutions, as in [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en); there a filter found an edge, here a filter smooths. A CNN learns its filters from data, and the first layer of a trained CNN often has filters that look like edge filters, and blobs of one color. The median filter is not a convolution: no grid of weights, whatever its numbers, does what it does. In [From light to a file of numbers](/course/open/cv-image-files?lang=en) you measured how far JPEG moves the grays; here the same measure says how far noise moves them, and how much of that a filter wins back. The next lesson, *Where the brightness jumps: gradients and Sobel*, looks for edges, and noise makes that hard: every grain is a small edge. That is why programs that look for edges often smooth the image first.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_mean, result_size, result_median):
    L.wrap_up(choice_weights, result_mean, result_size, result_median)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
