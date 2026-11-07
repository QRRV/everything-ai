# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-resize-interpolation. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-resize-interpolation", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [From light to a file of numbers](/course/open/cv-image-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bytes():
    quick_bytes = L.choice(
        "quick-bytes",
        question="The coffee photo has shape `(400, 600, 3)` and dtype uint8, so its numbers take 720,000 bytes. You make it twice as big each way: 800 rows by 1200 columns. How many bytes do its numbers take now?",
        options={
            "double": "1,440,000",
            "four": "2,880,000",
            "eight": "5,760,000",
        },
        correct="four",
        why={
            "four": "800 × 1200 × 3 = 2,880,000: twice as many rows and twice as many columns, so four times as many pixels. Most of those new pixels lie between old ones: this lesson is about what goes in them.",
            "double": "That is twice as many. But the rows double and the columns double too: rows × columns × colors, 800 × 1200 × 3. See *The idea* of [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "eight": "That would be 8 times as many: as if the colors doubled too. A pixel still has 3 colors: 800 × 1200 × 3.",
        },
    )
    quick_bytes
    return (quick_bytes,)


@app.cell(hide_code=True)
def quick_check_wrap():
    quick_wrap = L.choice(
        "quick-wrap",
        question="`a` and `b` are arrays of dtype uint8. A pixel is 40 in `a` and 200 in `b`. What is it in `a - b`?",
        options={
            "minus": "-160",
            "wrap": "96",
            "zero": "0",
            "plus": "160",
        },
        correct="wrap",
        why={
            "wrap": "uint8 stays between 0 and 255: below 0 it wraps around, to 256 - 160 = 96. This lesson mixes pixels; you will see where that matters.",
            "minus": "In ordinary whole numbers, yes. But uint8 cannot go below 0. See the watch-out in [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "zero": "uint8 does not stop at 0: one step below 0 lands at 255, the way a clock goes round. 40 - 200 is -160, and 256 - 160 is 96.",
            "plus": "That is the size of the difference, `abs(a.astype(int) - b.astype(int))`. In uint8, 40 - 200 wraps around: 256 - 160.",
        },
    )
    quick_wrap
    return (quick_wrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model that recognizes photos takes them at one fixed size, such as 224 by 224 pixels, so every photo it sees is made bigger or smaller first. A phone does the same each time you zoom into a photo, and a website each time it shows a small preview of a big one. The new image has other pixels than the old one, and most of them lie between old pixels: their value has to be worked out from the pixels around them. That is interpolation, and how you do it decides what you see: blocks, or soft steps.

    **An everyday example.** Zoom in far on a photo on your phone. At some point an edge that was sharp turns into a soft blur: the phone mixes the old pixels around every new pixel. A drawing program for pixel art shows every pixel as a sharp little square instead: it repeats the nearest old pixel.
    """)
    return


@app.cell(hide_code=True)
def predict_row():
    guess_row = L.guess(
        "row",
        question="A row of three pixels, 0, 100 and 200, is made twice as long: six pixels. Every new pixel gets a value between those of the old pixels around it, and the nearer old pixel counts more. What is the new row?",
        options={
            "blocks": "0 0 100 100 200 200",
            "even": "0 40 80 120 160 200",
            "quarters": "0 25 75 125 175 200",
        },
        reveal="**0 25 75 125 175 200.** Every new pixel is half an old pixel wide, and its middle lies at 0.25, 0.75, 1.25, 1.75, 2.25 or 2.75 in the old row; the middles of the old pixels lie at 0.5, 1.5 and 2.5. The second new pixel, at 0.75, lies a quarter of the way from the middle of the 0 to the middle of the 100: 25. The first, at 0.25, lies before the first middle, where there is nothing on the left: it stays 0. The row 0 0 100 100 200 200 repeats the nearest pixel, without mixing. And 0 40 80 120 160 200 puts the first and the last new pixel on the middles of the first and the last old pixel; some programs can do that too, but Pillow and this lesson look at where the middles of the new pixels fall.",
    )
    guess_row
    return (guess_row,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Think of an image as a grid of squares, one per pixel. In a row, pixel 0 covers the stretch from 0 to 1, pixel 1 the stretch from 1 to 2, and so on: the middle of pixel `c` lies at `c + 0.5`. A new image covers the same scene with squares of another size. Make a row of `w` pixels `W` pixels long, and every new pixel is `w / W` old pixels wide: the middle of new pixel `c` lies at `(c + 0.5) * w / W` in the old row. Made twice as long, a row of 3 pixels has its new middles at 0.25, 0.75, 1.25, 1.75, 2.25 and 2.75.

        **Nearest-neighbor interpolation** gives every new pixel the value of the old pixel whose square holds its middle: old pixel `int((c + 0.5) * w / W)`. The row 0, 100, 200 becomes 0 0 100 100 200 200: every old pixel turns into a block of two. Nearest neighbor never makes a value that was not there. Made bigger, an image gets blocks; made smaller, it skips pixels.

        **Linear interpolation** mixes the two old pixels on either side of a new middle, and the nearer one counts more. Count the position from the old middles: `x = (c + 0.5) * w / W - 0.5`. It lies between pixel `c0 = int(x)` and pixel `c0 + 1`, at `u = x - c0` of the way from the one to the other, and its value is `(1 - u) * left + u * right`. The second new pixel of the example has its middle at 0.75, so counted from the old middles it lies at `x = 0.25`: a quarter of the way from 0 to 100, so 0.75 × 0 + 0.25 × 100 = 25. Before the first middle and after the last, there is a pixel on one side only: keep `x` between 0 and `w - 1`, and the outermost pixel counts alone. The row becomes 0 25 75 125 175 200.

        **Bilinear interpolation** does the same in two directions. In an image, the middle of a new pixel lies between four old pixels: rows `r0` and `r1 = r0 + 1`, and columns `c0` and `c1 = c0 + 1`, with `t` how far the middle lies below `r0`, and `u` how far it lies to the right of `c0`. First mix the two pixels of row `r0` with `u`, as in one row, and the two pixels of row `r1` the same way. Then mix those two results with `t`: the nearer row counts more.

        An example: four pixels, 0 and 80 on top, and 40 and 200 below them. Halfway down and a quarter of the way to the right, `t = 0.5` and `u = 0.25`. Top: 0.75 × 0 + 0.25 × 80 = 20. Bottom: 0.75 × 40 + 0.25 × 200 = 80. Value: 0.5 × 20 + 0.5 × 80 = 50.
        """),
        L.watch_out("Pillow writes the size of an image as (width, height): first the number of columns, then the number of rows. NumPy's shape is (rows, columns). `Image.fromarray(img).resize((600, 400))` gives an image of 400 rows by 600 columns."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **interpolation** | working out values between known ones, from the values around them | `resize` in Pillow, for an image |
        | **nearest-neighbor interpolation** | every new pixel takes the value of the old pixel whose square holds its middle | `Image.NEAREST` |
        | **linear interpolation** | mixing the two values on either side of a position, so that the nearer one counts more | `linear_at` in the worked example |
        | **bilinear interpolation** | every new pixel mixes the four old pixels around its middle, and the nearer ones count more | `Image.BILINEAR` |
        | **`int(x)`** | a float as a whole number: what comes after the point is dropped | `int(2.75)` is `2` |
        | **`min(a, b)`, `max(a, b)`** | the smaller and the larger of two numbers; `min(max(x, 0), w - 1)` keeps `x` between 0 and `w - 1` | `min(3, 5)` is `3` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_row():
    mo.vstack([
        mo.md("**One row, four times as long**"),
        mo.Html(C.row_picture({"old": "old pixels", "nearest": "nearest neighbor", "bilinear": "linear",
                               "x": "position in the old row, in pixels", "y": "value",
                               "alt": "A row of six pixels, 40, 120, 200, 60, 140 and 180, as circles at their middles, 0.5 to 5.5, made 24 pixels long. Nearest neighbor, in gray, is a staircase that jumps halfway between two middles; linear interpolation is a thin line of points that runs straight from middle to middle, and is flat before the first middle and after the last."})),
        mo.md(r"""
        The six old pixels are the circles, at their middles. Made four times as long, nearest neighbor gives the gray steps: four new pixels for every old one, with the jump halfway between two middles. Linear interpolation, which is bilinear interpolation in one row, gives the points on the thin line: straight from middle to middle, and flat before the first middle and after the last, where the outermost pixel counts alone.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    A part of the coffee photo of [From light to a file of numbers](/course/open/cv-image-files?lang=en), around the spoon: 30 rows by 45 columns. With the slider you make it bigger, with nearest neighbor on the left and bilinear interpolation on the right. Both are drawn at the same size, so a pixel of the new image gets smaller as the factor grows.
    """)
    return


@app.cell(hide_code=True)
def explore_zoom_cell():
    def _say(k):
        if k == 1:
            return "Factor 1: the part as it is, 30 by 45 pixels, twice the same."
        return (f"Factor {k}: 30 by 45 pixels become {30 * k} by {45 * k}. On the left, every old pixel is a block of {k} by {k}; "
                "on the right, every new pixel has its own mix of the four old pixels around its middle.")

    def _frame(k):
        k = int(k)
        _html = C.zoom_frame(k, {"nearest": "nearest neighbor", "bilinear": "bilinear",
                                 "alt": "Two pictures of the same size of the spoon in the coffee photo, made {k} times bigger, {rows} by {cols} pixels: on the left with nearest neighbor, in blocks of {k} by {k} pixels, on the right with bilinear interpolation, with soft steps between the old pixels."})
        return _html, _say(k)

    explore_zoom = L.explore("zoom", "factor", _frame, start=1, values=C.FACTORS, role="",
                             task="Slide to 8. Where do you still see the squares of the old pixels, and where soft steps?")
    explore_zoom
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Nearest neighbor keeps every edge hard, but turns it into a staircase of blocks. Bilinear interpolation turns every step between two old pixels into a slope, so edges and highlights get soft. Neither adds detail: the part had 30 by 45 pixels, and what is not in them, no interpolation can make up.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`.astype(np.uint8)` drops what comes after the point, as `int` does: 67.9 becomes 67. Add 0.5 first, and it rounds: `(out + 0.5).astype(np.uint8)` makes 67.9 into 68, and 67.4 into 67."),
        mo.md(r"""
        ### Recipe

        To make an image `img` of `h` rows and `w` columns `H` rows and `W` columns big:

        1. Make an empty image of the new size, `np.zeros((H, W, 3))`.
        2. For every new pixel `(r, c)`: its middle lies at row `(r + 0.5) * h / H` and column `(c + 0.5) * w / W` of the old image.
        3. Nearest neighbor: take the old pixel at row `int((r + 0.5) * h / H)` and column `int((c + 0.5) * w / W)`.
        4. Bilinear: subtract 0.5 from both, to count from the middles, and keep them between 0 and `h - 1` and between 0 and `w - 1`, with `max` and `min`. Then mix the four old pixels around them, as in *The idea*.
        5. Round back to uint8.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell cuts the part around the spoon out of the coffee photo, and makes it four times bigger with nearest neighbor, steps 1 to 3 of the recipe. It does the same with Pillow's `Image.NEAREST`, and prints the two shapes and in how many numbers the two results differ. The second cell does linear interpolation in one row, with step 4 of the recipe: `linear_at` mixes the two pixels around a position, and `stretch` makes a row longer. It prints the row of the guess made twice as long, and what Pillow's `Image.BILINEAR` makes of it.
    """)
    return


@app.cell
def example_nearest():
    import numpy as np
    from PIL import Image

    img = np.array(Image.open("data/photos/coffee.png"))
    spoon = img[230:260, 320:365]


    def nearest(img, H, W):
        h, w = len(img), len(img[0])
        # 1. an empty image of the new size
        out = np.zeros((H, W, 3), dtype=np.uint8)
        for r in range(H):
            for c in range(W):
                # 2. and 3. the old pixel whose square holds the middle of the new one
                out[r, c] = img[int((r + 0.5) * h / H), int((c + 0.5) * w / W)]
        return out


    big = nearest(spoon, 120, 180)
    by_pillow = np.array(Image.fromarray(spoon).resize((180, 120), Image.NEAREST))
    print(spoon.shape, big.shape, (big != by_pillow).sum())
    return Image, np


@app.cell
def example_linear(Image, np):
    def linear_at(row, x):
        # the two pixels around position x, and how far x lies from the left one
        c0 = int(x)
        c1 = min(c0 + 1, len(row) - 1)
        u = x - c0
        return (1 - u) * row[c0] + u * row[c1]


    def stretch(row, W):
        w = len(row)
        out = []
        for c in range(W):
            # 4. counted from the middles, and inside the row
            x = (c + 0.5) * w / W - 0.5
            x = min(max(x, 0), w - 1)
            out.append(linear_at(row, x))
        return np.array(out)


    print(stretch([0, 100, 200], 6))
    row = np.array([[0, 100, 200]], dtype=np.uint8)
    print(np.array(Image.fromarray(row).resize((6, 1), Image.BILINEAR)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The part around the spoon has shape `(30, 45, 3)`, and four times bigger `(120, 180, 3)`. Pillow's `Image.NEAREST` gives exactly the same: in none of the 64,800 numbers do the two differ. The row 0, 100, 200 made twice as long is 0 25 75 125 175 200, as in the guess, and Pillow's `Image.BILINEAR` makes the same row. In exercise 2 you write bilinear interpolation for an image: linear interpolation along two rows, and then between them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_labels():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Labels, twice as big

    The course drew a picture of fruit, `data/photos/fruit.png`, and saved the label of every pixel next to it, in `data/photos/fruit-labels.png`: a gray image of 240 rows by 360 columns, in which every pixel is 0 for the tablecloth, 1 for an apple, 2 for the orange, 3 for the lemon and 4 for the lime. A model that learns to find the fruit wants the picture twice as big, and its labels with it.
    """)
    return


@app.cell(hide_code=True)
def choice_labels_cell():
    choice_labels = L.choice(
        "labels",
        question="With which interpolation do you make the label image twice as big?",
        options={
            "nearest": "Nearest neighbor",
            "bilinear": "Bilinear interpolation",
            "rounded": "Bilinear interpolation, rounded to whole numbers",
            "either": "It makes no difference: both keep the labels",
        },
        correct="nearest",
        why={
            "nearest": "Nearest neighbor only copies labels that are there: every new pixel gets the label of one old pixel. Bilinear interpolation mixes labels as if they were amounts: between the tablecloth, 0, and the lemon, 3, it makes numbers such as 0.75 and 2.25, which are no label, or once rounded the wrong one.",
            "bilinear": "Bilinear interpolation mixes labels as if they were amounts: between the tablecloth, 0, and the lemon, 3, it makes numbers such as 0.75 and 2.25. A label is a name written as a number: 0.75 is no fruit at all.",
            "rounded": "Rounded, 0.75 becomes 1 and 2.25 becomes 2: an apple and the orange, between the tablecloth and the lemon. Twice as big, 1940 of the 345,600 pixels get a label that none of the four old pixels around them had: a thin edge of apple around the orange, the lemon and the lime, of orange around the lemon and the lime, and of lemon around the lime.",
            "either": "For a photo, both give a picture. For labels, bilinear interpolation mixes names as if they were amounts: between the tablecloth, 0, and the lemon, 3, it makes 0.75 and 2.25. Which interpolation copies only values that are there?",
        },
    )
    choice_labels
    return (choice_labels,)


@app.cell(hide_code=True)
def text_exercise_value():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The value between four pixels

        Complete `value_at(img, y, x)`, which gives the bilinear value at row position `y` and column position `x`. Both are counted from the middles of the pixels, as in step 4 of the recipe, and already lie inside the image. `img` is a list of rows or an array, gray or in color. The first lines are there: `r0` and `r1` are the two rows around the position, `c0` and `c1` the two columns, and `t` and `u` say how far the position lies below `r0` and to the right of `c0`. At the last row and the last column, `r1` and `c1` stay there too. Replace the three `...` as in *The idea*: `top` mixes the two pixels of row `r0`, `bottom` those of row `r1`, and the function gives back the mix of the two. The check calls it with these four pixels and the position a quarter of the way down and halfway to the right:

        ```python
        img = [[0, 100],
               [200, 40]]
        value_at(img, 0.25, 0.5)
        ```

        That should give `67.5`. The check also tries the last row and column, a photo of uint8, and a color image.
        """),
        L.task("Make `value_at(img, y, x)` mix the four pixels around the position."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value():
    value_code = L.code_box("value", lines=8)
    value_code
    return (value_code,)


@app.cell(hide_code=True)
def check_value(value_code):
    result_value = L.check("value", value_code.value)
    result_value
    return (result_value,)


@app.cell(hide_code=True)
def hints_value():
    L.hints(
        "value",
        [
            "*The idea* mixes four pixels in three steps: first along the two rows, then between them. `linear_at` in the worked example is one such mix.",
            "`top` is `(1 - u)` times the left pixel of row `r0`, plus `u` times the right one. `bottom` is the same for row `r1`. The value is `(1 - t)` times `top`, plus `t` times `bottom`.",
            "`top = (1 - u) * img[r0][c0] + u * img[r0][___]`, `bottom = (1 - u) * img[___][c0] + u * img[r1][c1]`, and `return (1 - t) * ___ + t * bottom`.",
        ],
        why="For `value_at(img, 0.25, 0.5)`: `r0 = 0`, `r1 = 1`, `c0 = 0`, `c1 = 1`, `t = 0.25` and `u = 0.5`. Halfway along the top row, between 0 and 100, lies 50; halfway along the bottom row, between 200 and 40, lies 120. A quarter of the way from 50 to 120: 0.75 × 50 + 0.25 × 120 = 67.5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_half():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Half as big, with Pillow

        Writing interpolation yourself shows how it works; for a whole photo you use Pillow, which is much quicker. `Image.fromarray(img)` makes a Pillow image of an array, as in [From light to a file of numbers](/course/open/cv-image-files?lang=en). Its `resize` takes the new size, as (width, height), and the method, such as `Image.BILINEAR`, and gives back a new image. Make the coffee photo half as big, with bilinear interpolation, and put it in `small` as an array.
        """),
        L.task("Put the coffee photo at half its size in `small`, as an array."),
    ])
    return


@app.cell(hide_code=True)
def exercise_half():
    half_code = L.code_box("half", lines=6)
    half_code
    return (half_code,)


@app.cell(hide_code=True)
def check_half(half_code):
    result_half = L.check("half", half_code.value)
    result_half
    return (result_half,)


@app.cell(hide_code=True)
def hints_half():
    L.hints(
        "half",
        [
            "First a Pillow image of `coffee`, then its `resize`, then back to an array with `np.array(...)`.",
            "The photo has 600 columns and 400 rows, and Pillow wants the width first. The method is the second value of `resize`.",
            "`small = np.array(Image.fromarray(coffee).resize((___, ___), Image.BILINEAR))`",
        ],
        why="The coffee photo has shape `(400, 600, 3)`: 400 rows and 600 columns. Half of that is 200 rows by 300 columns. Pillow wants the width first, so `resize((300, 200), Image.BILINEAR)`, and `np.array` makes an array of shape `(200, 300, 3)` of it.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shrink():
    mo.md(r"""
    ### Exercise 4. Smaller is not the same

    The course ran the solution of exercise 2 over the coffee photo, with the recipe, and compared it with Pillow's `Image.BILINEAR`. Four times bigger, the part around the spoon differs nowhere by more than 1, on a scale from 0 to 255: a difference of 1 comes from rounding. Four times smaller, 100 by 150 pixels, the whole photo differs by 4.52 on average, and in one place by 104.
    """)
    return


@app.cell(hide_code=True)
def explain_shrink_cell():
    explain_shrink = L.explain(
        "shrink",
        question="Why do the two differ so much when the photo gets four times smaller, and hardly when it gets bigger? Think of how many old pixels lie under one new pixel, and how many of them your function mixes.",
        model_answer="**Four times smaller, one new pixel covers 4 by 4 = 16 old pixels.** **Bilinear interpolation mixes only the four around its middle;** the other 12 do not count at all. A thin line or a bright speck among those 12 vanishes, and one among the four counts for a quarter: what is left depends on where the middles happen to fall. **Pillow looks at more when it shrinks:** it mixes every old pixel under the new one, and those around it, with less weight the farther they lie. **Made bigger, a new pixel is smaller than an old one,** and the four old pixels around its middle are all that lie near it, so the two do the same.",
        points=[
            "you say that four times smaller, a new pixel covers 16 old pixels",
            "you say that bilinear interpolation mixes only the four around its middle, so 12 do not count",
            "you say that Pillow has to look at more old pixels than those four, such as all 16 under the new pixel",
            "you say that made bigger, the four old pixels around the middle are all that matter, so the two agree",
        ],
    )
    explain_shrink
    return (explain_shrink,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** An image is a grid of squares, and a new image a grid of other squares over the same scene: every new pixel takes its value from the old pixels around its middle. Nearest neighbor copies one of them, which keeps labels right and edges hard, in blocks. Bilinear interpolation mixes four, which gives soft steps, and a photo that looks smooth when it gets bigger. When it gets much smaller, both look at too few old pixels; Pillow then mixes every old pixel under the new one, and those around it. Rotating or warping an image works the same way: for every new pixel, work out where its middle lies in the old image, and interpolate there. Bicubic interpolation, Pillow's default, mixes the 4 by 4 old pixels around a middle when it makes an image bigger, along smooth curves instead of straight lines, for sharper edges. Later in this branch, *Get a photo ready for a model* makes every photo the size a pretrained model expects. The next lesson, *One image over another: alpha and green screens*, mixes two images pixel by pixel.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_labels, explain_shrink, result_half, result_value):
    L.wrap_up(choice_labels, result_value, result_half, explain_shrink)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
