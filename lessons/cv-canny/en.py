# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-canny. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-canny", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en) and [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_across():
    quick_across = L.choice(
        "quick-across",
        question="How does the image gradient $(g_x, g_y)$ of a pixel on an edge lie with respect to that edge?",
        options={
            "across": "Straight across the edge, toward the lighter side",
            "along": "Along the edge, the way the edge runs",
            "darker": "Straight across the edge, toward the darker side",
        },
        correct="across",
        why={
            "across": "The image gradient points uphill, the way the brightness grows fastest: straight across the edge, to the lighter side. This lesson uses that: the neighbors across an edge lie along its gradient.",
            "along": "Along the edge, the brightness hardly changes. Which way does it change fastest? See *The idea* of [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en).",
            "darker": "A positive $g_x$ means that the brightness grows to the right. Is the right side then lighter or darker? See [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en).",
        },
    )
    quick_across
    return (quick_across,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="`img = np.array([[50, 120], [100, 200]])`. What does `img > 100` give?",
        options={
            "right": "`[[False, True], [False, True]]`",
            "equal": "`[[False, True], [True, True]]`",
            "values": "`[[0, 120], [0, 200]]`",
        },
        correct="right",
        why={
            "right": "`>` compares every pixel with 100: 120 and 200 are larger, 50 is not, and 100 is not larger than itself. The outcome is a binary image of True and False.",
            "equal": "Is 100 larger than 100? `>` only counts what is really larger.",
            "values": "A comparison gives True or False for every pixel, not its gray. See [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The edges of [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en) are a few pixels wide, even where the brightness jumps from one pixel to the next: the window of 3 by 3 of a Sobel filter sees such a jump from more than one place, and a Gaussian filter first spreads it out over a few pixels. And a threshold breaks a faint edge into pieces wherever its strength dips. To measure an object, count objects or follow an outline, you want one closed line, one pixel wide, around each thing. The Canny edge detector gives that. It is from 1986, and still the standard way to find edges, in OpenCV and in other libraries.

    **An everyday example.** When you trace a faded pencil drawing with a fine pen, you draw one line through the middle of each gray stroke, not along its whole width. Where a stroke fades, you keep going, because you saw where it was clear and it carries on. A speck on the paper, on its own, you leave alone.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="In the drawing of fruit from [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en), every fruit has an outline. Where the light side of a fruit meets the light tablecloth, that outline is faint. There are other edges too: between the checks of the tablecloth, and around the shine of the lamp on the fruit. A program keeps every pixel whose edge strength is above one threshold. Is there a threshold that keeps the outlines whole and drops all the other edges?",
        options={
            "yes": "Yes, if you choose it well",
            "almost": "Almost: at the best threshold only a few pixels go wrong",
            "no": "No: at the best threshold, hundreds of pixels still go wrong",
        },
        reveal="**No.** Some stretches of outline are fainter than the strongest edges of the checks, which reach 98, and the shine of the lamp has edges up to 205. One threshold looks at one pixel at a time, at its edge strength only, so it cannot tell them apart. Trying thresholds in steps of 10, the best one is 110, and still 524 pixels go wrong. Even on the thin edges of step 3 below, the best one, 100, gets 307 wrong: 141 around the shine, and 166 of outline that are missed. Canny looks at the neighbors of a pixel as well.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **Canny edge detector**, by John Canny, finds edges in four steps. The first two are the lessons before this one:

        1. Smooth the image with the Gaussian filter of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en).
        2. Work out the image gradient $(g_x, g_y)$ with the two Sobel filters of [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en), and its length, the edge strength.

        Across an edge, the edge strength rises and falls like a hill. The third step keeps only the top of the hill:

        3. **Edge thinning**: compare every pixel with its two neighbors across the edge, one on each side. The image gradient points straight across the edge, so these are the neighbors along the gradient. A pixel that is at least as strong as both keeps its edge strength; every other pixel gets 0. What is left are lines one pixel wide; only where two pixels side by side across the edge are exactly as strong do both stay. Many books call this step non-maximum suppression.

        A pixel has eight neighbors, which lie in four directions through it: left and right, above and below, and the two diagonals. So the direction of the gradient is rounded to the nearest of the four:

        | the image gradient | compare with the neighbors | the step `(dr, dc)` |
        |---|---|---|
        | $\lvert g_y \rvert \le 0.41 \, \lvert g_x \rvert$: about to the right or the left | on the left and on the right | `(0, 1)` |
        | $\lvert g_x \rvert \le 0.41 \, \lvert g_y \rvert$: about down or up | above and below | `(1, 0)` |
        | otherwise, with $g_x$ and $g_y$ of the same sign | top left and bottom right | `(1, 1)` |
        | otherwise, with opposite signs | top right and bottom left | `(1, -1)` |

        0.41 is the slope of the direction halfway between flat and diagonal: 22.5 degrees, half of 45. With the step `(dr, dc)`, the two neighbors of `strength[r, c]` are `strength[r + dr, c + dc]`, a step forward, and `strength[r - dr, c - dc]`, a step back. In the third row, a positive $g_x$ and $g_y$ point to the bottom right; two negative ones to the top left, along the same line.
        """),
        L.watch_out("Compare across the edge, never along it. Along the edge, the neighbors belong to the same edge and are about as strong, so the edge would stay as wide as it was."),
        mo.md(r"""
        After thinning, a threshold still has to say which lines are edges. The fourth step uses two:

        4. A **hysteresis threshold**: a pixel above the high threshold is a strong edge, an edge for sure. A pixel between the low and the high threshold is a weak edge: it only counts when it is connected to a strong edge, through other edge pixels; a diagonal neighbor counts as connected. A pixel that is not above the low threshold is never an edge.

        In physics, hysteresis means that what happens depends on what came before, like a thermostat that switches the heating on below 19 degrees and off only above 21. Here, a weak pixel counts when its edge was already strong somewhere.

        An example: a row of pixels along an outline, with edge strengths 250, 180, 120, 150 and 230, and further away a lone pixel of 130. The low threshold is 100, the high one 200.

        | | what stays | why |
        |---|---|---|
        | one threshold of 200 | 250 and 230 | 180, 120 and 150 are not above 200: the outline falls apart in two pieces |
        | one threshold of 100 | the whole outline, and the lone 130 | everything above 100 counts, connected or not |
        | the hysteresis threshold, 100 and 200 | the whole outline | 180, 120 and 150 are weak, but connected to 250 and 230; the lone 130 is weak and connected to nothing |

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **Canny edge detector** | the standard way to find edges, in four steps: smooth, work out the image gradient, thin the edges, and apply a hysteresis threshold | the worked example; in OpenCV `cv2.Canny`, which does steps 2 to 4: smooth first yourself |
        | **edge thinning** | keeping, across every edge, only the pixel with the largest edge strength: a pixel stays when it is at least as strong as its two neighbors along the image gradient | `s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc])` |
        | **hysteresis threshold** | two thresholds: above the high one a pixel is an edge; between the two, only when it is connected to such a strong edge | `edges = thin > high`, then a to-do list |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_canny():
    mo.vstack([
        mo.md("**The drawing, its edge strength, the thinned edge strength, and three thresholds**"),
        mo.Html(C.canny_picture({"drawing": "The drawing", "strength": "Edge strength", "thin": "After edge thinning",
                                 "high": "One threshold: 200", "low": "One threshold: 100", "both": "Hysteresis: 100 and 200",
                                 "alt": "Six pictures of the drawing of fruit on a checked tablecloth. Top left the drawing in color: four red apples, an orange, a lemon and a lime. Top right the edge strength: almost white, with a dark ring around every fruit, a small ring around the shine of the lamp on each fruit, and a faint gray grid along the checks. In the middle on the left, after edge thinning: the same rings and grid, now as thin lines. In the middle on the right, one threshold of 200: only parts of most rings, and of the lime and the small apple at the far right hardly anything. At the bottom left, one threshold of 100: the rings of three apples whole, the other four with gaps, and the small rings of the shine. At the bottom right, the hysteresis threshold with 100 and 200: the same three rings whole and the small rings gone, except on the apple at the top left; the other four have gaps, and about a third of the ring of the lime is missing."})),
        mo.md(r"""
        Top right, the edge strength after the Gaussian and the Sobel filters, dark where it is strong; everything above 400 is as dark as 400. In the middle on the left, after edge thinning: the same edges, now one pixel wide, with faint lines between the checks. Then three ways to threshold the thinned edge strength. One threshold of 200 keeps only the strongest stretches of outline: of the lime, top right, and the small apple at the far right, hardly anything. One threshold of 100 keeps most of every outline, but also the shine of the lamp on every fruit. The hysteresis threshold with 100 and 200 keeps the outlines that are connected to a strong stretch, and drops the shine, except on the apple at the top left, where part of it is above 200.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("As grays, the earlier lessons took the strongest color of every pixel. In this drawing, that puts the red apples and the pink tablecloth close together, because both have a lot of red: the median gray of the apples is then 105, and of the tablecloth 150. The mean of red, green and blue, `img.mean(axis=2)`, gives 48 and 119, so the outlines of the apples are stronger."),
        L.watch_out("Thinning skips the outer rows and columns: a pixel there has no neighbor on one side. They stay 0, so a strong pixel is never at the border, and in step 4 `r - 1` never becomes −1, which in Python would quietly mean the last row."),
        mo.md(r"""
        ### Recipe

        To find the edges of a grayscale image `gray` with Canny:

        1. Smooth it with the Gaussian filter, work out `gx` and `gy` with the two Sobel filters, and the edge strength: `strength = (gx ** 2 + gy ** 2) ** 0.5`.
        2. Thin the edges: make an array of zeros `thin`, of the same shape. For every pixel off the border, take the step `dr, dc = step(gx[r, c], gy[r, c])`, and store `strength[r, c]` in `thin[r, c]` when it is at least as strong as the larger of its two neighbors, `max(strength[r + dr, c + dc], strength[r - dr, c - dc])`.
        3. The strong pixels are edges: `edges = thin > high`. Put each of them on a to-do list, as a tuple `(r, c)`.
        4. Go through the list with `for r, c in todo:`. Every pixel in the window of 3 by 3 around `(r, c)` that is above `low` and not yet an edge becomes an edge, and is appended to the list. In code that is `not edges[rr, cc]`: `not` turns True into False and False into True. A `for` loop over a list also reaches the items that are appended while it runs, so it only stops when no new pixel was added. Then every weak pixel that is connected to a strong one has been reached.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The drawing is the one from [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en): `data/photos/fruit.png`. The first code cell makes it gray, smooths it and works out the image gradient, with one function for all three filters. It prints the shape of the drawing and of the edge strength, and how many pixels have an edge strength above 100. The second cell thins the edges and applies the hysteresis threshold with 100 and 200, the course's first choice. It prints how many pixels are above 100 after thinning, how many are strong, and how many are edges in the end. Run the two code cells.
    """)
    return


@app.cell
def example_gradient():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/fruit.png"))
    # the drawing in grays: the mean of red, green and blue
    gray = img.mean(axis=2)
    G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    def filtered(photo, K):
        # a filter of 3 by 3, only where the whole window fits: 2 smaller in each direction
        out = np.zeros((len(photo) - 2, len(photo[0]) - 2))
        for r in range(len(out)):
            for c in range(len(out[0])):
                out[r, c] = (photo[r:r + 3, c:c + 3] * K).sum()
        return out


    # 1. smooth, then the image gradient and its length
    smoothed = filtered(gray, G)
    gx = filtered(smoothed, KX)
    gy = filtered(smoothed, KY)
    strength = (gx ** 2 + gy ** 2) ** 0.5
    print(gray.shape, strength.shape, (strength > 100).sum())
    return gx, gy, np, plt, strength


@app.cell
def example_canny(gx, gy, np, plt, strength):
    def step(gx, gy):
        # the step to the neighbor across the edge, in one of four directions
        if abs(gy) <= 0.41 * abs(gx):
            return 0, 1    # to the right; the other neighbor is on the left
        if abs(gx) <= 0.41 * abs(gy):
            return 1, 0    # down; the other one is above
        if gx * gy > 0:
            return 1, 1    # down to the right; the other one up to the left
        return 1, -1       # down to the left; the other one up to the right


    # 2. edge thinning: keep only the top of every edge, across it
    thin = np.zeros(strength.shape)
    for r in range(1, len(thin) - 1):
        for c in range(1, len(thin[0]) - 1):
            dr, dc = step(gx[r, c], gy[r, c])
            s = strength[r, c]
            if s >= max(strength[r + dr, c + dc], strength[r - dr, c - dc]):
                thin[r, c] = s
    print((thin > 100).sum())

    # 3. the strong pixels are edges, and go on the to-do list
    low, high = 100, 200
    edges = thin > high
    todo = []
    for r in range(len(thin)):
        for c in range(len(thin[0])):
            if edges[r, c]:
                todo.append((r, c))
    # 4. a weak pixel next to an edge becomes an edge too, and joins the list;
    #    the loop also reaches the pixels that join while it runs
    for r, c in todo:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if thin[rr, cc] > low:
                    if not edges[rr, cc]:
                        edges[rr, cc] = True
                        todo.append((rr, cc))
    print((thin > high).sum(), edges.sum())
    # the edges: True is black, False white
    plt.imshow(edges, cmap="gray_r")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The drawing has shape `(240, 360)` and the edge strength `(236, 356)`: 4 smaller in each direction, 2 for the Gaussian filter and 2 for the Sobel filters. 3465 pixels have an edge strength above 100; after thinning, 1629 do, less than half, because every edge is now one pixel wide. 828 pixels are strong, above 200; with the weak pixels that are connected to them, 1473 are edges. In the picture under the cell, the outlines of three apples are closed, and the shine of the lamp is gone, except on the apple at the top left. The other outlines have gaps, and the outline of the lime, at the top right, breaks off: about a third of it is not found. Part of it is not stronger than 100, and the stretch beyond that part is never reached. Exercise 3 looks for a better low threshold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_neighbors():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which neighbors?
    """)
    return


@app.cell(hide_code=True)
def choice_neighbors_cell():
    choice_neighbors = L.choice(
        "neighbors",
        question="At one pixel, the image gradient is $(g_x, g_y) = (-300, 290)$. With which two neighbors does edge thinning compare that pixel?",
        options={
            "sides": "On the left and on the right",
            "updown": "Above and below",
            "main": "Top left and bottom right",
            "anti": "Top right and bottom left",
        },
        correct="anti",
        why={
            "anti": "Neither part is small next to the other: 290 is more than 0.41 × 300 = 123, and 300 is more than 0.41 × 290 = 118.9, so the gradient is diagonal. $g_x$ is negative and $g_y$ positive: the lighter side is at the bottom left. The step is `(1, -1)`: the neighbors at the bottom left and at the top right, straight across the edge.",
            "sides": "That is right when $g_y$ is small next to $g_x$: at most 0.41 × 300 = 123. Is 290 that small?",
            "updown": "That is right when $g_x$ is small next to $g_y$: at most 0.41 × 290 = 118.9. Is 300 that small?",
            "main": "The gradient is diagonal, but which diagonal? $g_x$ is negative: the lighter side is on the left. And $g_y$ is positive: lighter below.",
        },
    )
    choice_neighbors
    return (choice_neighbors,)


@app.cell(hide_code=True)
def text_exercise_thin():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Edge thinning

        Complete `thin(strength, gx, gy)`. It gets the edge strength and the two parts of the image gradient, and gives back the thinned edge strength, of the same shape: every pixel that is at least as strong as its two neighbors across the edge keeps its edge strength, every other pixel is 0. `step` and the loops are already there: replace the `...`. The check calls it with an edge from top to bottom, strongest in the middle column. The brightness grows to the right, so `gx` is the edge strength itself and `gy` is 0:

        ```python
        edge = [[0, 40, 90, 60, 0],
                [0, 40, 90, 60, 0],
                [0, 40, 90, 60, 0]]
        zeros = [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]
        thin(edge, edge, zeros)
        ```

        That should give `[[0, 0, 0, 0, 0], [0, 0, 90, 0, 0], [0, 0, 0, 0, 0]]`; the top and the bottom row are outer rows, which always stay 0. It also tries other edges.
        """),
        L.task("Make `thin(strength, gx, gy)` keep only the pixels that are at least as strong as their two neighbors across the edge."),
    ])
    return


@app.cell(hide_code=True)
def exercise_thin():
    box_thin = L.code_box("thin", lines=24)
    box_thin
    return (box_thin,)


@app.cell(hide_code=True)
def check_thin(box_thin):
    result_thin = L.check("thin", box_thin.value)
    result_thin
    return (result_thin,)


@app.cell(hide_code=True)
def hints_thin():
    L.hints(
        "thin",
        [
            "Two neighbors: one a step `(dr, dc)` forward, `strength[r + dr, c + dc]`, and one a step back, `strength[r - dr, c - dc]`.",
            "At least as strong as both neighbors means: at least as strong as the larger of the two, `max(...)`. When `strength[r, c]` is that (`>=`), `out[r, c]` gets `strength[r, c]`; otherwise it stays 0.",
            "`if strength[r, c] >= max(strength[r + dr, c + dc], ___):` and under it `out[r, c] = strength[r, c]`.",
        ],
        why="In the example the step is `(0, 1)`: the neighbors on the left and on the right. 40 has 0 and 90 next to it, and is weaker than 90; 90 has 40 and 60, and is at least as strong as both; 60 has 90 and 0. So only 90 stays: the top of the hill.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_low():
    mo.md(r"""
    ### Exercise 3. The low threshold

    In the worked example, the low threshold was 100, and the outline of the lime broke off. With the slider you choose the low threshold; the high one stays at 200. The course drew the drawing, so it knows where every outline is: where a fruit meets the tablecloth or another fruit. A pixel counts as wrong when it is an edge with no outline in its window of 3 by 3 (too many), or when it is on an outline with no edge in its window (missed): the window gives the thin lines one pixel of room. On the left, the edges at the chosen low threshold; on the right, how many pixels are wrong at every low threshold. The axis stops at 1200, so that the low point shows: a triangle at the top means the value is higher.
    """)
    return


@app.cell(hide_code=True)
def explore_low_cell():
    def _frame(low):
        _html, _f = C.low_frame(low, {"edges": "edges, low threshold {low}", "low": "low threshold", "wrong": "pixels wrong",
                                      "alt": "On the left, the edges of the drawing of fruit with a low threshold of {low} and a high threshold of 200, as black lines on white; on the right, how many pixels are wrong for the low thresholds 0 to 200, on an axis up to 1200: the low thresholds 0 to 40 lie above it, at 4263 down to 1694, marked by triangles at the top; from 50 on the curve drops to a low point at 70, then rises to 1119 at 200. Low threshold {low} is marked by a larger point, a triangle at the top when its value is above 1200."})
        return (_html,
                f'Low threshold {low}: {_f["wrong"]} pixels are wrong, {_f["too_many"]} too many and {_f["missed"]} missed.')

    explore_low = L.explore("low", "low threshold", _frame, start=200, values=C.LOWS, role="hyper",
                            task="Find the low threshold where the fewest pixels are wrong.")
    explore_low
    return (explore_low,)


@app.cell(hide_code=True)
def check_low(explore_low):
    result_low = L.check("low", explore_low)
    result_low
    return (result_low,)


@app.cell(hide_code=True)
def hints_low():
    L.hints(
        "low",
        [
            "Move the slider one stop at a time, from 200 to the left, and read the numbers in the sentence under the picture.",
            "To the left, faint stretches of outline join, and fewer pixels are missed. Too far, and the edges of the checks that touch an outline join too: look for the stop where the number turns.",
            "Look at the stops 60, 70 and 80: which one gives the smallest number?",
        ],
        why="At 60, 150 pixels are wrong; at 70, 107; at 80, 155. Above 70, more of the outline is missed; below it, more edges of the checks come in.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_two():
    mo.md(r"""
    ### Exercise 4. Why two thresholds?
    """)
    return


@app.cell(hide_code=True)
def explain_two_cell():
    explain_two = L.explain(
        "two",
        question="With one threshold on the thinned edge strength, in the same steps of 10, the best you can do on the drawing is 307 wrong pixels, at 100. With the hysteresis threshold of exercise 3, far fewer pixels are wrong. Why can a hysteresis threshold keep a faint stretch of outline and drop the edge of a check, when one threshold cannot?",
        model_answer="**Why one threshold fails:** a threshold looks at one pixel at a time, at its edge strength only. Some stretches of outline are as faint as the edges of the checks, or fainter, and the shine of the lamp has strong edges. So every threshold either loses faint outline or keeps edges that are not outlines. **Why two thresholds work:** a hysteresis threshold also looks at the neighbors. A faint stretch of outline is connected to a strong stretch of the same outline, so it is reached from there and kept. Most edges of the checks and of the shine are weak and stand on their own, connected to nothing above the high threshold, so they go. Only weak edges that touch an outline, such as the edge of a check right next to a fruit, still come in, and the shine on the apple at the top left, part of which is strong itself.",
        points=[
            "you say that one threshold only looks at the edge strength of each pixel, and that faint outline and other edges can be equally strong",
            "you say that a weak pixel only counts when it is connected to a strong edge, and that a faint stretch of outline is connected to a strong part of the same outline",
            "you say that the edges of the checks are not connected to a strong edge, so they are dropped",
        ],
    )
    explain_two
    return (explain_two,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Canny puts together the last lessons: the Gaussian filter of [Less noise: the mean, the Gaussian and the median filter](/course/open/cv-denoise-filters?lang=en), the image gradient of [Where the brightness jumps: gradients and Sobel](/course/open/cv-sobel-edges?lang=en), and the binary image of [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en), now with two thresholds instead of one. The to-do list of the hysteresis threshold follows connected pixels, one neighbor at a time; later lessons count connected pieces in the same way. The next lesson, *Grow and shrink shapes: erosion and dilation*, works on binary images such as these edges: it makes the shapes in a binary image grow or shrink, to remove specks or fill holes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_neighbors, explain_two, result_low, result_thin):
    L.wrap_up(choice_neighbors, result_thin, result_low, explain_two)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
