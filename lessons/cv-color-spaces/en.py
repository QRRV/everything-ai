# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-color-spaces. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-color-spaces", "en")
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
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="A color photo is 600 pixels wide and 400 high. What is the shape of `np.array(Image.open(path))`?",
        options={
            "rows": "`(400, 600, 3)`",
            "width": "`(600, 400, 3)`",
            "flat": "`(600, 400)`",
        },
        correct="rows",
        why={
            "rows": "The shape of an array starts with the rows, the height, then the columns, the width, then the 3 colors.",
            "width": "That is the order Pillow uses for `.size`, width first. What comes first in the shape of an array? See the second watch-out in [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "flat": "A color photo has three numbers per pixel. Where do they go in the shape, and what comes first, the rows or the columns? See [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_wrap():
    quick_wrap = L.choice(
        "quick-wrap",
        question="`a` and `b` are arrays of dtype uint8. A number in `a` is 10, and the same number in `b` is 12. What is that number in `a - b`?",
        options={
            "minus": "−2",
            "plus": "2",
            "wrap": "254",
        },
        correct="wrap",
        why={
            "minus": "That is what ordinary whole numbers give. Numbers of the dtype uint8 stay between 0 and 255: where does −2 end up? See the first watch-out in [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "plus": "That is the size of the difference, `abs`. But `a - b` with uint8 stays between 0 and 255: what does −2 become? See the first watch-out in [From light to a file of numbers](/course/open/cv-image-files?lang=en).",
            "wrap": "uint8 stays between 0 and 255, so −2 wraps around to 254. With `.astype(int)` first, it is −2.",
        },
    )
    quick_wrap
    return (quick_wrap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    To find the red apples in a photo, a rule such as "red above 150" seems enough, but an apple in the shade has less of everything, and its red drops below 150 while it is just as red. HSV writes a color as three other numbers: which color, how strong and how bright. Shade then changes only the last one, and a rule for "red" stays simple. Picking out one color, a green screen, and changing the colors of photos to give a model more variety all work this way.

    **An everyday example.** In a paint shop you do not ask for amounts of red, green and blue. You point at a color on a color wheel, then say how vivid it should be, and how light or dark.
    """)
    return


@app.cell(hide_code=True)
def predict_shade():
    guess_shade = L.guess(
        "shade",
        question="A red apple in the drawing of the course has the pixel `(171, 26, 26)` in the light, and the same kind of apple `(108, 16, 16)` in the shade. You write both as hue, saturation and value: which of the three numbers differ?",
        options={
            "all": "All three",
            "hue": "The hue: in the shade it is another red",
            "value": "Only the value: how bright",
        },
        reveal="**Only the value**: 0.67 in the light, 0.42 in the shade. Both have hue 0, red, and saturation 0.85. The shade makes red, green and blue each about 0.6 times as big, and that changes how bright the pixel is, not its color.",
    )
    guess_shade
    return (guess_shade,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    RGB says how much red, green and blue light a pixel has: the three numbers of an image file. HSV writes the same color as three other numbers, each from 0 to 1. The value $v$ is the strongest of the three, on a scale from 0 to 1: how bright the pixel is. The saturation is how much of that is color and not grey: $(v - \text{min}) / v$, the strongest minus the weakest, divided by the strongest, and 0 for black, where $v$ is 0. The hue says which color: a place on a circle of colors. The formula for hue has a case for each of red, green and blue, whichever is the strongest, so you let `rgb_to_hsv` of matplotlib do it. Many other programs write the hue in degrees, from 0 to 360; matplotlib writes it from 0 to 1.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **color space** | a way to write a color as numbers | RGB, HSV |
    | **RGB** | a color as how much red, green and blue light: the numbers of an image file | `img[:, :, 0]` is the red |
    | **HSV** | a color as hue, saturation and value, each from 0 to 1 | `rgb_to_hsv(img / 255)` |
    | **hue** | which color: a place on the color circle, with red at 0 and again at 1 | `hsv[:, :, 0]` |
    | **saturation** | how strong the color is: 0 has no color, as grey and white, 1 is the pure color | `hsv[:, :, 1]` |
    | **value** | how bright: the strongest of red, green and blue | `hsv[:, :, 2]` |
    | **`&`**, **`|`** | and, or, for two masks: pixel by pixel | `(h < 0.05) | (h > 0.95)` |

    `rgb_to_hsv` wants floats from 0 to 1, so you divide the image by 255 first; it gives an array of the same shape, with hue, saturation and value instead of red, green and blue. Python does `&` and `|` before `<` and `>`, so every comparison gets its own parentheses.
    """)
    return


@app.cell(hide_code=True)
def picture_strips():
    mo.vstack([
        mo.md("**The three numbers of HSV**"),
        mo.Html(C.strips({"hue": "Hue: which color", "saturation": "Saturation of red: from white, through pink, to the pure red",
                          "value": "Value of red: from black to the brightest red",
                          "hue_ticks": ["0\nred", "1/6\nyellow", "1/3\ngreen", "1/2\ncyan", "2/3\nblue", "5/6\nmagenta", "1\nred"],
                          "saturation_ticks": ["0", "0.5", "1"], "value_ticks": ["0", "0.5", "1"],
                          "alt": "Three strips from 0 to 1. The hue goes around the colors: red at 0, yellow at 1/6, green at 1/3, cyan at 1/2, blue at 2/3, magenta at 5/6, and red again at 1. The saturation of red goes from white through pink to bright red; the value of red from black to bright red."})),
        mo.md(r"""
        Each strip changes one of the three numbers, from 0 to 1. The hue goes once around the colors and ends where it began: at 0 and at 1 it is the same red. The saturation of red goes from white, with no color, through pink to the pure red; the value from black, with no light, to the brightest red.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Red lies at both ends of the hue: just above 0 and just below 1, as 23:59 and 0:01 lie close together on a clock. A rule for red needs both ends: `(h < 0.05) | (h > 0.95)`."),
        L.watch_out("Grey, white and black have no color: saturation 0, and their hue means nothing. `rgb_to_hsv` gives them hue 0, the hue of red, so a rule on hue alone also picks grey. Add a condition on the saturation."),
        L.watch_out("The word saturation also has another meaning, in neural networks, which comes in a later lesson. In this lesson it always means how strong a color is."),
        mo.md(r"""
        ### Recipe

        To pick out the pixels of one color:

        1. Make floats from 0 to 1 of the image: `img / 255`.
        2. Write every pixel in HSV: `hsv = rgb_to_hsv(img / 255)`.
        3. Take the numbers apart: `h = hsv[:, :, 0]` and `s = hsv[:, :, 1]`; the value is `hsv[:, :, 2]`.
        4. Write the rule as a mask: a window of hue for the color, and a saturation above a threshold, so that grey stays out. Join the comparisons with `&` and `|`, each in parentheses.
        5. Count what it picks: with a label for every pixel, `mask[labels == 4].mean()` is the share of the pixels with label 4 that it finds, as `(diff > 0).mean()` was a share in [From light to a file of numbers](/course/open/cv-image-files?lang=en).
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The course drew fruit on a pink checked tablecloth, under a lamp on the left: `data/photos/fruit.png`. Because the course made it, it also knows what every pixel is: `data/photos/fruit-labels.png` holds 0 for the tablecloth, 1 for an apple, 2 for the orange, 3 for the lemon and 4 for the lime. The example looks for the lime, with a rule in RGB and one in HSV. Run the two cells.
    """)
    return


@app.cell
def example_hsv():
    import numpy as np
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    labels = np.array(Image.open("data/photos/fruit-labels.png"))
    # 1. and 2. floats from 0 to 1, and every pixel in HSV
    hsv = rgb_to_hsv(img / 255)
    print(img.shape, hsv.shape)
    # the middle of an apple in the light, and of one in the shade
    print(img[70, 70], hsv[70, 70])
    print(img[175, 260], hsv[175, 260])
    return hsv, img, labels


@app.cell
def example_lime(hsv, img, labels):
    # 3. the hue and the saturation apart
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # 4. a rule for green: in RGB on the green alone, in HSV on hue and saturation
    green_rgb = img[:, :, 1] > 150
    green_hsv = (h > 0.2) & (h < 0.45) & (s > 0.5)
    # 5. the share of the lime each rule finds, and how many other pixels it picks
    lime = labels == 4
    print(green_rgb[lime].mean(), green_rgb[labels != 4].sum())
    print(green_hsv[lime].mean(), green_hsv[labels != 4].sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The drawing has shape `(240, 360, 3)`, and so has `hsv`: three numbers per pixel, now hue, saturation and value. The two apples both have hue 0 and saturation 0.85, rounded, but value 0.67 in the light and 0.42 in the shade. The rule in RGB finds none of the lime: it lies on the right, where the lamp gives less light, and its green stays below 150 there. It does pick 6,163 other pixels: of the lemon, whose yellow has a lot of green, of the tablecloth in the light, and of the white shine on the other fruit. The rule in HSV finds 0.988 of the lime, 98.8%, and nothing else. What it misses is the shine of the lamp, where the color fades and its saturation drops below 0.5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stronger():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Strong or bright
    """)
    return


@app.cell(hide_code=True)
def choice_stronger_cell():
    choice_stronger = L.choice(
        "stronger",
        question="Which of these two colors has the higher saturation: dark red `(100, 0, 0)` or pink `(255, 200, 200)`?",
        options={
            "dark": "The dark red",
            "pink": "The pink",
            "same": "Neither: both are red, so their saturation is the same",
        },
        correct="dark",
        why={
            "dark": "The dark red has no green and no blue at all: saturation (100 − 0) / 100 = 1, the pure color, only dark. The pink is as bright as can be, value 1, but has a lot of green and blue in it: saturation (255 − 200) / 255 = 0.22.",
            "pink": "The pink is brighter: that is its value, 1. Saturation says how strong the color is: how much green and blue are mixed in with the red? Work it out with the formula under The idea.",
            "same": "Both have hue 0, red. But saturation says how much of the other two is mixed in. Work it out: the strongest minus the weakest, divided by the strongest.",
        },
    )
    choice_stronger
    return (choice_stronger,)


@app.cell(hide_code=True)
def text_exercise_value_saturation():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Value and saturation by hand

        Finish `value_saturation(pixel)`. It gets one pixel, three numbers from 0 to 255, and gives back its value and its saturation, each from 0 to 1. The first line already makes floats from 0 to 1 of it, in `rgb`. Replace the `...`. The check calls it with a brownish orange:

        ```python
        value_saturation([204, 102, 51])
        ```

        That should give `(0.8, 0.75)`. It also tries other pixels, among them grey and black.
        """),
        L.task("Make `value_saturation(pixel)` give back the value and the saturation of the pixel."),
    ])
    return


@app.cell(hide_code=True)
def exercise_value_saturation():
    box_value_saturation = L.code_box("value-saturation", lines=9)
    box_value_saturation
    return (box_value_saturation,)


@app.cell(hide_code=True)
def check_value_saturation(box_value_saturation):
    result_value_saturation = L.check("value-saturation", box_value_saturation.value)
    result_value_saturation
    return (result_value_saturation,)


@app.cell(hide_code=True)
def hints_value_saturation():
    L.hints(
        "value-saturation",
        [
            "The value is the strongest of the three: `rgb.max()`. The saturation is the strongest minus the weakest, divided by the strongest.",
            "Black is `[0, 0, 0]`: then the value is 0, and you cannot divide by 0. What saturation does a pixel without any light have?",
            "`v = rgb.max()`, then `if v == 0:` with `return 0.0, 0.0`, then `s = (v - rgb.___()) / ___` and `return v, s`.",
        ],
        why="For `[204, 102, 51]`, `rgb` is `[0.8, 0.4, 0.2]`. The value is the strongest, 0.8; the saturation is (0.8 − 0.2) / 0.8 = 0.75. For black, the value is 0: no light, so no color, and saturation 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Exercise 3. Leave out the tablecloth

    The rule for red takes a hue below 0.05 or above 0.95. Pink is a red too, with little saturation, so the tablecloth falls in the window as well. With the slider you set the threshold: the mask only keeps pixels with a saturation above it. What the mask picks keeps its colors; the rest turns grey.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _f = C.threshold_frame(t, {"alt": "The drawing of fruit, with in color what the mask picks at saturation threshold {t}, and the rest grey."})
        return (_html,
                f'Threshold {L.number(t, 1)}: the mask finds {L.number(100 * _f["apple"], 1)}% of the apple pixels, and <span class="c-loss ea-num">{C.grouped(_f["cloth"], ",")}</span> pixels of the tablecloth.')

    explore_threshold = L.explore("threshold", "threshold", _frame, start=0.0, values=C.THRESHOLDS, role="hyper",
                                  task="Find the threshold that finds the most apple pixels without a single pixel of the tablecloth.")
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
            "Move the slider up one stop at a time, and watch the squares of the tablecloth: do they still have color?",
            "The sentence under the picture counts the pixels of the tablecloth that the mask picks. You want 0, with as many apple pixels as can be.",
            "The higher the threshold, the less of the tablecloth, but near the top the apples go too. Go up until the tablecloth is gone, and stop there.",
        ],
        why="At 0.4, the mask still picks 26,038 pixels of the tablecloth; at 0.5 none, with 99.1% of the apple pixels. The pink of the tablecloth has a saturation up to 0.47, the apples about 0.85.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_red():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. A mask for red

        Finish `red_mask(hsv)`. It gets an image in HSV, as `rgb_to_hsv` makes it, and gives back a mask: `True` for every red pixel. Red is a hue below 0.05 or above 0.95, with a saturation above 0.5, the threshold of exercise 3. The first lines already take `h` and `s` apart. Replace the `...`. The check calls it with one row of five pixels, in HSV:

        ```python
        red_mask([[[0.02, 0.9, 0.8], [0.98, 0.7, 0.5], [0.08, 0.9, 0.9], [0.0, 0.0, 0.6], [0.01, 0.3, 0.9]]])
        ```

        That should give `[[True, True, False, False, False]]`: red, red at the other end of the circle, orange, grey and pink. It also tries other images.
        """),
        L.task("Make `red_mask(hsv)` give `True` for every pixel with a hue below 0.05 or above 0.95 and a saturation above 0.5."),
    ])
    return


@app.cell(hide_code=True)
def exercise_red():
    box_red = L.code_box("red-mask", lines=9)
    box_red
    return (box_red,)


@app.cell(hide_code=True)
def check_red(box_red):
    result_red = L.check("red-mask", box_red.value)
    result_red
    return (result_red,)


@app.cell(hide_code=True)
def hints_red():
    L.hints(
        "red-mask",
        [
            "Read the two watch-outs above the recipe: where does red lie on the hue, and what hue does a grey pixel get?",
            "Two conditions on the hue, joined with `|`, together in parentheses; then `&` with the condition on the saturation.",
            "`(h < 0.05) | (h > 0.95) & (s > 0.5)` is not it: Python does the `&` first, so the saturation only counts for one end. Where do parentheses go, so that the two conditions on hue count together?",
        ],
        why="`(h < 0.05) | (h > 0.95)` is red at both ends of the circle; the parentheses around it make Python do it before the `&`. `& (s > 0.5)` leaves out grey and pink. For the row of the example, that gives `True, True, False, False, False`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [From light to a file of numbers](/course/open/cv-image-files?lang=en), a pixel was three numbers of red, green and blue; now you write the same color as hue, saturation and value, and a rule for a color holds in the light and in the shade. In [Selecting parts of an array](/course/open/np-indexing?lang=en), a mask came from one comparison; now you join comparisons with `&` and `|`. Later in this route, a green screen gets its mask the same way, from how green every pixel is. In the next lesson, *Spread the grays: histogram equalization*, you spread the brightness of a whole image over the whole scale.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stronger, result_red, result_threshold, result_value_saturation):
    L.wrap_up(choice_stronger, result_value_saturation, result_threshold, result_red)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
