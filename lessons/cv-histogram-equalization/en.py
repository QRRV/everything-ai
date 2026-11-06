# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-histogram-equalization. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-histogram-equalization", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="What is the value of the pixel `(60, 180, 120)` in HSV?",
        options={
            "max": "180 / 255, about 0.71",
            "mean": "120 / 255, about 0.47",
            "saturation": "(180 − 60) / 180, about 0.67",
        },
        correct="max",
        why={
            "max": "The value is the strongest of red, green and blue, on a scale from 0 to 1: 180 / 255, about 0.71.",
            "mean": "That is the mean of the three. The value of HSV takes one of the three: which one? See The idea in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).",
            "saturation": "That is the formula of the saturation: how strong the color is. The value says how bright: what does it take of the three? See [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_quantile():
    quick_quantile = L.choice(
        "quick-quantile",
        question="For an array of numbers `x`, `np.quantile(x, 0.75)` gives 40. Which part of the numbers is 40 or less?",
        options={
            "three": "About 3 of every 4",
            "one": "About 1 of every 4",
            "forty": "About 40%",
        },
        correct="three",
        why={
            "three": "The 0.75 quantile is the value below which 0.75 of the values lie: 3 of every 4 numbers are 40 or less.",
            "one": "That would be the 0.25 quantile. Which fraction does `0.75` say? See the new words in [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "forty": "40 is the value itself, not a share. Which fraction of the values lies below the 0.75 quantile? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
        },
    )
    quick_quantile
    return (quick_quantile,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A photo taken in fog, or of a page in poor light, looks flat and gray: its darkest parts are not black, and its lightest parts are not white. Its pixels use only a narrow band of the 256 grays, and the details in it are hard to see, for you and for a program. Histogram equalization spreads those grays over the whole scale, from 0 to 255, with one rule that only looks at how many pixels have each gray. It is one of the oldest tools of image processing: scikit-image and OpenCV both have it, and a variant of it is often used on X-rays and on photos from a microscope.

    **An everyday example.** Twenty runners finish a race within one minute of each other. Give them their places, from 1 to 20, instead of their times, and the small differences become big ones. Equalization does the same with grays: every gray gets its place in the row from dark to light.
    """)
    return


@app.cell(hide_code=True)
def predict_grays():
    guess_grays = L.guess(
        "grays",
        question="The worked example of this lesson makes a photo dull on purpose: it then uses 65 of the 256 grays. After histogram equalization, how many different grays does the photo use?",
        options={
            "all": "All 256",
            "same": "65, the same as before",
            "fewer": "Fewer than 65",
        },
        reveal="**Fewer than 65**: it uses 49. Equalization gives every gray a new place on the scale, but all pixels with the same gray get the same new gray, so it can never make more grays than there were. Several grays next to each other that few pixels have can even end up as one. The 49 grays now lie spread out from 0 to 255, with gaps between them.",
    )
    guess_grays
    return (guess_grays,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    The contrast of a photo is how far apart its grays lie: a dull photo has little. Its histogram shows that: a narrow band of bars, and nothing to the left or the right of it. Histogram equalization gives every gray a new gray by one rule: take the share of the pixels that have this gray or a darker one, and multiply it by 255. A gray with share 0.02 becomes 5; a gray where half of the pixels have that gray or a darker one becomes about 128, near the middle of the scale; the lightest gray becomes 255. Where many pixels have nearly the same gray, the share climbs fast from one gray to the next, so those grays move far apart. Where few pixels are, the share hardly changes, and those grays move together. The rule does not need to know how the photo became dull: the histogram is enough.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **contrast** | how far apart the grays of an image lie; a dull image has little | one measure: `img.std()` |
    | **cumulative histogram** | for every gray, how many pixels have that gray or a darker one; divided by the number of pixels, it is a share from 0 to 1 | `np.cumsum(counts)` |
    | **histogram equalization** | every gray gets the new gray 255 × its share in the cumulative histogram, so the grays spread over the whole scale | `table[img]` |
    | **`np.cumsum(x)`** | the running total: every number plus all the numbers before it | `np.cumsum([2, 0, 3, 1])` gives `array([2, 2, 5, 6])` |
    | **`table[img]`** | an array of whole numbers between square brackets: for every number in it, the element of `table` at that place, in the same shape | `t[np.array([2, 0, 0])]` gives `array([30, 10, 10])` if `t` is `np.array([10, 20, 30])` |

    The new grays of all 256 grays go in one array, a table: `table[g]` is the new gray of gray `g`. `table[img]` then looks up every pixel of the image at once, and gives an image of the same shape.
    """)
    return


@app.cell(hide_code=True)
def picture_before_after():
    mo.vstack([
        mo.md("**A dull photo, before and after equalization**"),
        mo.Html(C.before_after({"dull": "Dull", "after": "After equalization", "histogram": "Histogram",
                                "cumulative": "Cumulative histogram", "gray": "gray",
                                "alt": "Two rows of three pictures. At the top, the dull photo of a cat, all in middle grays; its histogram is a narrow band of bars between 101 and 169; its cumulative histogram is 0 below gray 101, climbs steeply around gray 145 and reaches 1 at gray 169. At the bottom, the photo after equalization, with black and white in it; its histogram has bars spread from 0 to 255, with gaps between them; its cumulative histogram lies close to the straight line from 0 to 1."})),
        mo.md(r"""
        At the top, the dull photo: its histogram is a narrow band between gray 101 and gray 169, and its cumulative histogram climbs from 0 to 1 within that band. At the bottom, the same photo after equalization: its bars are spread over the whole scale, with gaps, and its cumulative histogram lies close to the dashed straight line. Each quarter of the scale now holds about a quarter of the pixels.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`plt.imshow` with `cmap=\"gray\"` stretches the grays by itself: it shows the darkest gray of the array as black, and the lightest as white. A dull photo then does not look dull. To see the grays as they are, add `vmin=0, vmax=255`."),
        L.watch_out("Equalization does not make new grays: all pixels with the same gray get the same new gray. The histogram after it has as many bars as before, or fewer, with gaps between them."),
        mo.md(r"""
        ### Recipe

        To equalize a grayscale image `img` of uint8:

        1. Count the pixels of every gray: `counts = np.array([(img == g).sum() for g in range(256)])`. That is the histogram as numbers, one for every gray from 0 to 255.
        2. Add them up from dark to light and divide by the number of pixels: `share = np.cumsum(counts) / counts.sum()`. That is the cumulative histogram as shares; the lightest gray has share 1.
        3. Make the table with the new gray of every gray: `table = (255 * share).astype(np.uint8)`. `astype` cuts off the decimals: 212.5 becomes 212.
        4. Look up every pixel in the table: `new = table[img]`.
        5. Look at the result with `plt.imshow(new, cmap="gray", vmin=0, vmax=255)`, and compare the contrast: `img.std()` and `new.std()`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The photo is Chelsea the cat of [From light to a file of numbers](/course/open/cv-image-files?lang=en): `data/photos/chelsea.png`. The example first makes it gray, with the value of HSV of every pixel: `img.max(axis=2)`, the strongest of red, green and blue; `axis=2` is the third direction of the array, the three colors. That gray photo has good contrast, so the example then makes it dull, as fog would: every gray g becomes 100 + 0.3 g. Equalization does not use that formula; it only looks at the histogram. Run the two cells.
    """)
    return


@app.cell
def example_count():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    img = np.array(Image.open("data/photos/chelsea.png"))
    # a photo in grays: the value of HSV of every pixel
    photo = img.max(axis=2)
    # fog, on purpose: every gray g becomes 100 + 0.3 g
    dull = (100 + 0.3 * photo).astype(np.uint8)
    # 1. the pixels of every gray, and how many grays the photo uses
    counts = np.array([(dull == g).sum() for g in range(256)])
    print(dull.min(), dull.max(), (counts > 0).sum())
    # 2. the cumulative histogram as shares
    share = np.cumsum(counts) / counts.sum()
    print(share[130], share[145], share[169])
    return dull, np, plt, share


@app.cell
def example_equalize(dull, np, plt, share):
    # 3. the table: the new gray of every gray
    table = (255 * share).astype(np.uint8)
    print(table[101], table[145], table[169])
    # 4. every pixel looked up in the table
    new = table[dull]
    # 5. the contrast before and after, and the photo after
    print(dull.std(), new.std())
    plt.imshow(new, cmap="gray", vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The grays of the dull photo lie between 101 and 169, and it uses 65 of the 256: a few light grays, 165 to 168, do not occur. Gray 130 has share 0.081: only 8% of the pixels have that gray or a darker one, the eyes, the nose and the dark stripes on the head. Gray 145 has share 0.521, so just over half of the pixels have that gray or a darker one, and the lightest gray, 169, has share 1. The table makes 101, the darkest gray, into 0, gray 145 into 132, and 169 into 255. The standard deviation of the grays, one measure of the contrast, goes from about 10 to about 74. Under the cell, the cat is clear again, with darker stripes and lighter fur than in the gray photo before the fog.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_snow():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Snow
    """)
    return


@app.cell(hide_code=True)
def choice_snow_cell():
    choice_snow = L.choice(
        "snow",
        question="A photo of snow: 9 of every 10 pixels are light grays from 200 to 250, the snow, and the other pixels are dark grays from 10 to 40, a few trees. After histogram equalization, which grays does the snow get?",
        options={
            "most": "Everything above about 25: nine tenths of the scale",
            "same": "Still 200 to 250: equalization keeps light grays light",
            "half": "From 128 to 255: the light half of the scale",
        },
        correct="most",
        why={
            "most": "The trees are 1 of every 10 pixels: they take the shares up to 0.1, so the new grays up to about 25. The snow takes the shares from 0.1 to 1: the new grays from just above 25 to 255, nine tenths of the scale. Small differences in the snow become easy to see, and the trees all end up between 0 and 25.",
            "same": "Equalization does not keep the gray itself: it looks at the share of the pixels with this gray or a darker one. What is that share for the darkest gray of the snow?",
            "half": "The snow does stay lighter than the trees. But how far down does it go? The trees are only 1 of every 10 pixels: what share has the darkest gray of the snow, and what is 255 times that?",
        },
    )
    choice_snow
    return (choice_snow,)


@app.cell(hide_code=True)
def text_exercise_cumulative():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The cumulative histogram

        Finish `cumulative(counts)`. It gets the number of pixels of every gray, in order from dark to light, and gives back for every gray the share of the pixels with that gray or a darker one. Replace the `...`. The check calls it with the counts of an image of 6 pixels: 2 of gray 0, none of gray 1, 3 of gray 2 and 1 of gray 3.

        ```python
        cumulative([2, 0, 3, 1])
        ```

        That should give 1/3, 1/3, 5/6 and 1; the check shows them with six decimals, as `[0.333333, 0.333333, 0.833333, 1]`. It also tries other lists of counts.
        """),
        L.task("Make `cumulative(counts)` give, for every gray, the share of the pixels with that gray or a darker one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cumulative():
    box_cumulative = L.code_box("cumulative", lines=7)
    box_cumulative
    return (box_cumulative,)


@app.cell(hide_code=True)
def check_cumulative(box_cumulative):
    result_cumulative = L.check("cumulative", box_cumulative.value)
    result_cumulative
    return (result_cumulative,)


@app.cell(hide_code=True)
def hints_cumulative():
    L.hints(
        "cumulative",
        [
            "Two steps: first the running total of `counts`, then divide it by the number of pixels.",
            "The running total is `np.cumsum(counts)`. How many pixels are there in all? That is the sum of the counts.",
            "`return np.cumsum(counts) / ___`, with in the blank the number of pixels.",
        ],
        why="`np.cumsum([2, 0, 3, 1])` is `[2, 2, 5, 6]`: 2 pixels have gray 0 or darker, 2 have gray 1 or darker, 5 have gray 2 or darker, and all 6 have gray 3 or darker. Divided by the 6 pixels, that is 1/3, 1/3, 5/6 and 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_middle():
    mo.md(r"""
    ### Exercise 3. The middle of the scale

    The slider picks a gray of the dull photo of the worked example. The picture shows its cumulative histogram, with that gray marked: on the left the share of the pixels with that gray or a darker one, on the right the same share times 255, the new gray.
    """)
    return


@app.cell(hide_code=True)
def explore_middle_cell():
    def _frame(g):
        _html, _f = C.gray_frame(g, {"gray": "gray of the dull photo", "share": "share", "new": "new gray",
                                     "alt": "The cumulative histogram of the dull photo, from gray 96 to 174, with gray {g} marked by a dot."})
        return (_html,
                f'Gray {g}: {L.number(100 * _f["share"], 1)}% of the pixels have this gray or a darker one, so it becomes gray {_f["new"]}.')

    explore_middle = L.explore("middle", "gray", _frame, start=130, values=C.GRAYS, role="",
                               task="Find the darkest gray of the dull photo that becomes gray 128 or lighter, about the middle of the scale.")
    explore_middle
    return (explore_middle,)


@app.cell(hide_code=True)
def check_middle(explore_middle):
    result_middle = L.check("middle", explore_middle)
    result_middle
    return (result_middle,)


@app.cell(hide_code=True)
def hints_middle():
    L.hints(
        "middle",
        [
            "Move the slider one stop at a time, and read the new gray in the sentence under the picture.",
            "The new gray is 255 times the share. Which share gives about 128, halfway between 0 and 255?",
            "Just over half: look for the first gray where the share passes 50%.",
        ],
        why="Gray 144 has share 0.467 and becomes 119; gray 145 has share 0.521 and becomes 132. So 145 is the first gray where half of the pixels have that gray or a darker one: the median of the dull photo, `np.median(dull)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_equalize():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Equalize

        Finish `equalize(img)`. It gets a grayscale image of whole numbers from 0 to 255, and gives back the image after histogram equalization, as uint8. The first lines already count the pixels of every gray, in `counts`. Replace the `...`: the share of every gray, the table of new grays, and every pixel looked up in the table. The check calls it with an image of 6 pixels:

        ```python
        equalize([[100, 100, 120], [120, 120, 200]])
        ```

        That should give `[[85, 85, 212], [212, 212, 255]]`. It also tries other images.
        """),
        L.task("Make `equalize(img)` give back the image after histogram equalization."),
    ])
    return


@app.cell(hide_code=True)
def exercise_equalize():
    box_equalize = L.code_box("equalize", lines=9)
    box_equalize
    return (box_equalize,)


@app.cell(hide_code=True)
def check_equalize(box_equalize):
    result_equalize = L.check("equalize", box_equalize.value)
    result_equalize
    return (result_equalize,)


@app.cell(hide_code=True)
def hints_equalize():
    L.hints(
        "equalize",
        [
            "Follow steps 2, 3 and 4 of the recipe: the share, the table, then every pixel looked up.",
            "The share is the running total of `counts` divided by the number of pixels, as in exercise 2. The table is 255 times that, as uint8.",
            "`share = np.cumsum(counts) / counts.sum()`, then `table = (___ * share).astype(np.uint8)`, then `return table[___]`.",
        ],
        why="The 6 pixels have 2 of gray 100, 3 of gray 120 and 1 of gray 200. The shares are 2/6, 5/6 and 6/6, and 255 times that is 85, 212.5 and 255; `astype` cuts 212.5 off to 212. `table[img]` gives every pixel the new gray of its gray.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Describing data: mean, median and spread](/course/open/stats-describe?lang=en), a quantile was the value below which a given part of the values lies; the cumulative histogram goes the other way, and says for every gray which part of the pixels lies at or below it. The median of the dull photo landed near the middle of the scale. In [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en), the value said how bright a pixel is: to equalize a color photo, you can equalize only its value, so the colors stay the same. In the next lesson, *Black or white: a threshold and Otsu's method*, the histogram itself tells you where to split the dark pixels from the light ones.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_snow, result_cumulative, result_equalize, result_middle):
    L.wrap_up(choice_snow, result_cumulative, result_middle, result_equalize)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
