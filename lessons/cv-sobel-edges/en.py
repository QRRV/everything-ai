# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-sobel-edges. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-sobel-edges", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Two parameters: the gradient](/course/open/math-gradient?lang=en) and [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gradient():
    quick_gradient = L.choice(
        "quick-gradient",
        question="At one point, the gradient of a loss $\\mathcal{L}(w, b)$ is $(2, -1)$. What does that say?",
        options={
            "up": "The loss rises fastest if w gets bigger and b smaller: the gradient points uphill",
            "low": "The lowest point of the loss is at w = 2, b = −1",
            "down": "The loss falls fastest if w gets bigger and b smaller",
        },
        correct="up",
        why={
            "up": "The partial derivative for w is 2: the loss rises when w gets bigger. The one for b is −1: the loss rises when b gets smaller. Together they point uphill, the way the loss rises fastest; gradient descent goes the other way.",
            "low": "The gradient is made of two slopes at the point where you stand, not of a place. What does a slope of 2 for w say? See the table of new words of [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
            "down": "The gradient points uphill. Which way does gradient descent go, along the gradient or against it? See [Two parameters: the gradient](/course/open/math-gradient?lang=en).",
        },
    )
    quick_gradient
    return (quick_gradient,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="How long is the vector $(3, 4)$: what is its norm $\\lVert (3, 4) \\rVert$?",
        options={
            "five": "5",
            "seven": "7",
            "twentyfive": "25",
        },
        correct="five",
        why={
            "five": "The norm is the square root of the sum of the squares: √(3² + 4²) = √25 = 5. In code: `np.linalg.norm([3, 4])`.",
            "seven": "7 is 3 + 4, the sum of the parts. The norm first squares each part. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "twentyfive": "25 is 3² + 4², the sum of the squares. One step is still missing. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An edge is where one thing ends and the next begins: the rim of a cup against the table, a letter against the paper, the side of a road. Edges carry much of what an image shows: a drawing of only the outlines is still clearly a cup. So finding edges is an old first step of image processing, before finding shapes, measuring objects, reading text or keeping a car in its lane. An edge is where the brightness changes fast, and how fast something changes is what a gradient measures.

    **An everyday example.** On a hiking map, a cliff is where the contour lines bunch together, because the height changes fast there. Think of a photo in grays as a landscape: light grays are high, dark grays low. An edge is a cliff, and a patch of one gray is a plain. The gradient of [Two parameters: the gradient](/course/open/math-gradient?lang=en) said, at every point, which way is uphill and how steep it is; this lesson asks the same for every pixel.
    """)
    return


@app.cell(hide_code=True)
def predict_grain():
    guess_grain = L.guess(
        "grain",
        question="A program that looks for edges marks a pixel as an edge where the brightness changes fast. On the photo of the coffee cup of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en), about 6% of the pixels count as edges. Now the photo gets the grain of that lesson: every gray plus or minus a random number with standard deviation 25. With the same setting, how many pixels count as edges?",
        options={
            "same": "About as many: 6%",
            "twice": "About twice as many: 12%",
            "five": "About five times as many: 30%",
        },
        reveal="**About five times as many**: 29.4%. A program that looks for edges compares each pixel with its neighbors, and grain is random from pixel to pixel: a pixel a bit too light next to one a bit too dark looks like a small edge. Grain with standard deviation 25 is small next to a real edge, but it is everywhere. The worked example measures it, and shows what smoothing first does.",
    )
    guess_grain
    return (guess_grain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A grayscale image is a function of two inputs: the gray at row `r` and column `c`. Its partial derivative for `c` says how fast the gray changes when you go to the right, with the row fixed. As in [Two parameters: the gradient](/course/open/math-gradient?lang=en), you estimate it with a difference. A pixel is the smallest step there is, so you take one on each side: the gray on the right minus the gray on the left, `img[r, c + 1] - img[r, c - 1]`. That change spans 2 pixels, so the slope per pixel is half of it; image processing usually skips the dividing, because only the comparison with a threshold matters. A **Sobel filter** does that for a whole window of 3 by 3; there are two, one to the right and one downward:
        """),
        L.formula(r"K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} \qquad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}"),
        mo.md(r"""
        $K_x$ takes the right column of the window minus the left column, with the three rows weighted 1, 2, 1, as in the Gaussian filter of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en): the middle row counts most, and the grain of one row counts less. Its result $g_x$ is large and positive where the brightness grows to the right, large and negative where it falls, and 0 where it stays the same from left to right. $K_y$ is $K_x$ turned a quarter clockwise, which here is the same as its transpose: the bottom row minus the top row. Its result $g_y$ is positive where the brightness grows downward.

        Together, $(g_x, g_y)$ is the **image gradient** of the pixel in the middle of the window. Like the gradient of [Two parameters: the gradient](/course/open/math-gradient?lang=en), it points uphill: toward the lighter side, straight across the edge. Its length, the norm of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), is the *edge strength*: how fast the brightness changes there, whichever way the edge runs.
        """),
        L.formula(r"\lVert (g_x, g_y) \rVert = \sqrt{g_x^2 + g_y^2}"),
        mo.md(r"""
        **Edge detection** marks the pixels whose edge strength is above a threshold.
        """),
        L.watch_out("$g_y$ looks downward: row 0 is the top row of an image, so a positive $g_y$ means lighter below. In a chart, y grows upward; in an image, it is the other way around."),
        mo.md(r"""
        Four windows of 3 by 3, with dark 0 and light 255:

        | the window | $g_x$ | $g_y$ | edge strength |
        |---|---|---|---|
        | light on the right: `[[0, 0, 255], [0, 0, 255], [0, 0, 255]]` | 1020 | 0 | 1020 |
        | light below: `[[0, 0, 0], [0, 0, 0], [255, 255, 255]]` | 0 | 1020 | 1020 |
        | light at the bottom right: `[[0, 0, 0], [0, 0, 255], [0, 255, 255]]` | 765 | 765 | 1082 |
        | one gray: `[[90, 90, 90], [90, 90, 90], [90, 90, 90]]` | 0 | 0 | 0 |

        In the third window the gradient $(765, 765)$ points to the bottom right, where it is light, and the edge runs diagonally, across it.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **edge detection** | finding the places in an image where the brightness changes fast; they often mark the borders of objects | `strength > 150` |
        | **Sobel filter** | a filter of 3 by 3 that estimates how fast the brightness changes to the right ($K_x$) or downward ($K_y$) | `(window * KX).sum()` |
        | **image gradient** | for each pixel, $(g_x, g_y)$: how fast the brightness changes to the right and downward; it points toward the lighter side, and its length is the edge strength | `(gx, gy)`; its length `np.linalg.norm([gx, gy])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_edges():
    mo.vstack([
        mo.md("**The photo, its two Sobel results and its edge strength**"),
        mo.Html(C.edges_picture({"photo": "The photo", "gx": "gx: to the right", "gy": "gy: downward", "strength": "Edge strength",
                                 "alt": "Four pictures of the coffee cup on its saucer. Top left the photo in grays. Top right gx: mostly middle gray, with the left and right sides of the cup and the saucer drawn as light and dark lines, as if lit from the side; the top and bottom edges of the cup are almost gone. Bottom left gy: the same, but now the top and bottom edges are drawn and the sides are almost gone. Bottom right the edge strength: almost white, with every edge as a dark line: the rim of the cup, the saucer, the handle, the spoon and the wood of the table."})),
        mo.md(r"""
        At the top right $g_x$ and at the bottom left $g_y$, with 0 as middle gray: light where the brightness grows to the right or downward, dark where it falls; everything beyond ±400 is as light or as dark as 400, and in the edge strength everything above 400 as dark as 400. $g_x$ sees the sides of the cup, and hardly its top and bottom edge; $g_y$ the other way around. The edge strength at the bottom right combines the two: every edge is a dark line, whichever way it runs.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The results of a Sobel filter go below 0 and above 255: on this photo from −772 to 769. So turn the image into floats first, with `np.array(img, dtype=float)`. In uint8, a difference such as `img[r, c + 1] - img[r, c - 1]` would wrap around below 0, as in [From light to a file of numbers](/course/open/cv-image-files?lang=en); with floats, every step is safe."),
        L.watch_out("A Sobel result is 8 times the slope in grays per pixel: the difference spans 2 pixels, and the weights 1, 2, 1 add up to 4. Libraries such as OpenCV leave it that way; it only changes which threshold you choose."),
        mo.md(r"""
        ### Recipe

        To find the edges of a grayscale image `img`:

        1. Make an array of zeros for the result, 2 smaller in each direction: `out = np.zeros((len(img) - 2, len(img[0]) - 2))`.
        2. Go through every place, with `for r in range(len(out)):` and inside it `for c in range(len(out[0])):`, and take the window that starts there: `window = img[r:r + 3, c:c + 3]`. As in [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en), that window has the pixel `img[r + 1, c + 1]` in its middle: `out[r, c]` is the edge strength of that pixel.
        3. Work out the image gradient, `gx = (window * KX).sum()` and `gy = (window * KY).sum()`, and store its length: `out[r, c] = np.linalg.norm([gx, gy])`.
        4. Mark the edges with a threshold: `edges = out > 150` is True where the edge strength is above 150. 150 is the course's choice for the clean photo; exercise 3 shows how to choose one.

        If the image has grain, filter it first with the Gaussian filter of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en).
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The photo is the one from [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en): the part with the coffee cup, in grays, with the same grain: standard deviation 25, seed 0. The first code cell makes the grainy photo, and a smoothed one: the grainy photo after the Gaussian filter. The second works out the edge strength of three photos, the clean, the grainy and the smoothed one, and prints for each the size of the result, the largest edge strength, and the share of the pixels with an edge strength above 150. Under the cell are the edges of the clean photo. Run the two code cells; the second takes a few seconds.
    """)
    return


@app.cell
def example_photos():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    # the photo in grays, the strongest color of every pixel, and the part with the cup
    clean = img.max(axis=2)[0:300, 100:500].astype(float)
    rng = np.random.default_rng(0)
    grainy = clean + rng.normal(0, 25, size=clean.shape)
    # the Gaussian filter of 3 by 3, 2 smaller in each direction
    G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    smoothed = np.zeros((len(grainy) - 2, len(grainy[0]) - 2))
    for r in range(len(smoothed)):
        for c in range(len(smoothed[0])):
            smoothed[r, c] = (grainy[r:r + 3, c:c + 3] * G).sum()
    return clean, grainy, np, plt, smoothed


@app.cell
def example_edges(clean, grainy, np, plt, smoothed):
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    def edge_strength(photo):
        # 1. an array of zeros, 2 smaller in each direction
        out = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        # 2. every window of 3 by 3
        for r in range(len(out)):
            for c in range(len(out[0])):
                window = photo[r:r + 3, c:c + 3]
                # 3. the image gradient, and its length
                gx = (window * KX).sum()
                gy = (window * KY).sum()
                out[r, c] = np.linalg.norm([gx, gy])
        return out


    strengths = [edge_strength(photo) for photo in [clean, grainy, smoothed]]
    for strength in strengths:
        # 4. the share above the threshold: the mean of True (1) and False (0)
        print(strength.shape, strength.max(), (strength > 150).mean())
    # the edges of the clean photo: True is black, False white
    plt.imshow(strengths[0] > 150, cmap="gray_r")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the clean photo, the largest edge strength is 777, and a share of 0.060 of the pixels, 6.0%, is above 150: the outlines of the cup, the saucer and the spoon, and some of the wood of the table, in the picture under the cell. In the grainy photo it is 29.4%, the answer of the guess: grain makes edges everywhere. A Sobel filter takes a difference, and grain is random from pixel to pixel, so in a difference it does not cancel out the way it does in a mean. The grain alone gives an edge strength of about 110 on average, on a patch of one gray as well; after the Gaussian filter, about 50. With the Gaussian filter first it is 6.0% again, and the largest edge strength is 667: the filter cancels most of the grain, and makes the edges a little less steep. But 6.0% does not mean they are the same pixels as in the clean photo; exercise 3 measures how many are wrong.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_direction():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which way is lighter?
    """)
    return


@app.cell(hide_code=True)
def choice_direction_cell():
    choice_direction = L.choice(
        "direction",
        question="At one pixel of a photo, the image gradient is $(g_x, g_y) = (0, 400)$. Where is the lighter side?",
        options={
            "below": "Below the pixel",
            "above": "Above the pixel",
            "right": "To the right of the pixel",
            "none": "Nowhere: there is no edge, because $g_x$ is 0",
        },
        correct="below",
        why={
            "below": "$g_x = 0$: from left to right the brightness stays the same. $g_y = 400$ is positive: the brightness grows downward, because row numbers grow downward. So the lighter side is below, and the edge runs from left to right, across the gradient.",
            "above": "A positive $g_y$ means the brightness grows as the row number grows. Where is row 0 in an image? See the watch-out in *The idea*.",
            "right": "Which of the two says something about left and right? Here it is 0.",
            "none": "The edge strength is the length of $(0, 400)$: 400, not 0. A part that is 0 only says which way the edge runs.",
        },
    )
    choice_direction
    return (choice_direction,)


@app.cell(hide_code=True)
def text_exercise_strength():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The edge strength

        Complete `edge_strength(img)`. It gets a grayscale image and gives back the edge strength of every place, 2 smaller in each direction. The two Sobel filters, the loops and the window are already there: replace the `...` with `gx`, `gy`, and the length of `(gx, gy)` stored in `out[r, c]`. The check calls it with a ramp that grows 3 grays per pixel to the right and 4 downward:

        ```python
        edge_strength([[0, 3, 6, 9],
                       [4, 7, 10, 13],
                       [8, 11, 14, 17],
                       [12, 15, 18, 21]])
        ```

        That should give `[[40, 40], [40, 40]]`. It also tries other images.
        """),
        L.task("Make `edge_strength(img)` give every place the length of its image gradient."),
    ])
    return


@app.cell(hide_code=True)
def exercise_strength():
    box_strength = L.code_box("strength", lines=14)
    box_strength
    return (box_strength,)


@app.cell(hide_code=True)
def check_strength(box_strength):
    result_strength = L.check("strength", box_strength.value)
    result_strength
    return (result_strength,)


@app.cell(hide_code=True)
def hints_strength():
    L.hints(
        "strength",
        [
            "Three lines: `gx` is the window times `KX`, added up; `gy` the same with `KY`; and `out[r, c]` gets the length of `(gx, gy)`.",
            "Multiply place by place with `window * KX`, and add up the nine products with `.sum()`. The length of a vector is `np.linalg.norm(...)`, as in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "`gx = (window * KX).sum()`, then `gy` in the same way with `KY`, and `out[r, c] = np.linalg.norm([___, ___])`.",
        ],
        why="In every window of the ramp, the right column is 6 grays more than the left one, in each row; with the weights 1, 2, 1, `gx` is 6 × 4 = 24. The bottom row is 8 grays more than the top one, in each column, so `gy` is 8 × 4 = 32. The length of (24, 32) is √(576 + 1024) = √1600 = 40.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Exercise 3. Where to put the threshold?

    Edge detection needs a threshold: a pixel with an edge strength above it counts as an edge. With the slider you choose the threshold for the smoothed photo of the worked example: the grainy photo after the Gaussian filter. The answer key is the edges of the clean photo, its pixels with an edge strength above 150. On the left, the edges at the chosen threshold; on the right, for every threshold, the share of the pixels that are wrong: an edge where the clean photo has none, or none where it has one. The axis stops at 12%, so that the low point shows: a triangle at the top means the value is higher.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"edges": "edges above {t}", "threshold": "threshold", "wrong": "pixels wrong, in %",
                                          "alt": "On the left, the edges of the smoothed grainy photo at a threshold of {t}, as black on white; on the right, the share of the pixels that are wrong for the thresholds 0 to 400, on an axis up to 12%: the thresholds 0 to 75 lie above it, at 94% down to 27%, marked by triangles at the top; from 100 on the curve goes down to a low point between 150 and 200, then slowly up again to about 6% at 400. Threshold {t} is marked by a larger point, a triangle at the top when its value is above 12%."})
        return (_html,
                f'Threshold {t}: {L.number(_f["wrong"], 1)}% of the pixels are wrong.')

    explore_threshold = L.explore("threshold", "threshold", _frame, start=0, values=C.THRESHOLDS, role="hyper",
                                  task="Find the threshold where the fewest pixels are wrong.")
    explore_threshold
    return (explore_threshold,)


@app.cell(hide_code=True)
def check_threshold(explore_threshold):
    result_threshold = L.check("threshold", explore_threshold)
    result_threshold
    return (result_threshold,)


@app.cell(hide_code=True)
def hints_threshold():
    L.hints(
        "threshold",
        [
            "Move the slider one stop at a time, and read the number in the sentence under the picture.",
            "From 0 on, the number drops fast: fewer grains pass as edges. Somewhere it goes up again, because real edges that are faint are lost too: look for the stop where it turns.",
            "Look at the stops 150, 175 and 200: which one gives the smallest number?",
        ],
        why="At 150, 3.2% of the pixels are wrong, at 175 2.9%, and at 200 3.2%. Below 175, more grain passes as edges; above it, more faint real edges are lost.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.md(r"""
    ### Exercise 4. Why smooth first?
    """)
    return


@app.cell(hide_code=True)
def explain_smooth_cell():
    explain_smooth = L.explain(
        "smooth",
        question="Without the Gaussian filter first, the best threshold for the grainy photo is 300, and still 4.1% of the pixels are wrong: more than with the filter, at the threshold you found in exercise 3. Why does grain make a Sobel filter find edges where there are none, and why does it help to apply the Gaussian filter first?",
        model_answer="**Why grain makes edges:** a Sobel filter takes a difference: the right column minus the left one, or the bottom row minus the top one. Grain is random from pixel to pixel, so it does not cancel out in a difference: a pixel a bit too light next to one a bit too dark gives a difference that is not in the scene. On the grainy photo, the grain alone gives an edge strength of about 110. **Why smoothing helps:** the Gaussian filter takes a weighted mean of each window, and in a mean much of the grain cancels out. The differences that are left then come mostly from real edges, so a lower threshold is enough, and fewer faint real edges are lost.",
        points=[
            "you say that a Sobel filter takes a difference of neighboring pixels, and that random grain does not cancel out in a difference",
            "you say that the Gaussian filter averages, so much of the grain cancels out before the difference is taken",
            "you say that a lower threshold is then enough, so fewer faint real edges are lost, or that fewer grains pass as edges",
        ],
    )
    explain_smooth
    return (explain_smooth,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The Sobel filters are convolutions, like the edge filters of [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en), with the weights 1, 2, 1 of the Gaussian filter of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en) across their direction. The image gradient is the gradient of [Two parameters: the gradient](/course/open/math-gradient?lang=en): there, how fast the loss changes for every parameter; here, how fast the brightness changes for every step across the image. The first layer of a trained CNN often learns filters much like these two. The next lesson, *Thin, connected edges: Canny*, starts from this edge strength and from the direction of the gradient: it thins every edge to a line of one pixel, and keeps faint pieces of edge only where they connect to strong ones.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_direction, explain_smooth, result_strength, result_threshold):
    L.wrap_up(choice_direction, result_strength, result_threshold, explain_smooth)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
