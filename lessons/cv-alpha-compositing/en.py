# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-alpha-compositing. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-alpha-compositing", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Broadcasting](/course/open/np-broadcasting?lang=en) and about [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`m` has shape `(2, 3)` and `col` has shape `(2, 1)`. What does `m * col` give?",
        options={
            "fits": "An array of shape `(2, 3)`",
            "error": "An error: 3 and 1 differ",
            "small": "An array of shape `(2, 1)`",
            "more": "An array of shape `(2, 3, 1)`",
        },
        correct="fits",
        why={
            "fits": "Lined up on the right, 3 and 1: one of the two is 1, so NumPy stretches the column over the 3 columns; then 2 and 2, the same. Every number in a row is multiplied by the number of `col` in that row. In this lesson the alpha gets such a length 1, so that it fits the three colors.",
            "error": "3 and 1 differ, but one of the two is 1, and NumPy stretches a length 1 to the other length. The shapes fit: the outcome has shape `(2, 3)`.",
            "small": "NumPy stretches the length 1 of `col` to 3; it throws nothing of `m` away. The outcome has shape `(2, 3)`.",
            "more": "Both shapes have two lengths, so the outcome has two as well: lined up on the right, 3 and 1 give 3, and 2 and 2 give 2. The outcome has shape `(2, 3)`.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="`h` is an array with the hue of every pixel. Which line gives True for the pixels whose hue lies between 0.3 and 0.45?",
        options={
            "amp": "`(h > 0.3) & (h < 0.45)`",
            "and": "`h > 0.3 and h < 0.45`",
            "bar": "`(h > 0.3) | (h < 0.45)`",
            "bare": "`h > 0.3 & h < 0.45`",
        },
        correct="amp",
        why={
            "amp": "`&` joins two masks pixel by pixel: True where both are True. Every comparison gets its own parentheses, because Python does `&` before `<` and `>`.",
            "and": "`and` wants one True or False, and an array holds one per pixel: Python stops with an error. Join masks with `&`.",
            "bar": "`|` gives True where at least one of the two is True, and every hue is above 0.3 or below 0.45: all pixels. You want both: `&`.",
            "bare": "Python does `&` before `<` and `>`, so it first tries `0.3 & h`, and stops with an error. Put every comparison in parentheses.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A weather presenter stands in front of a green wall, and you see a map behind them. A film puts actors on a planet that does not exist. A logo or a sticker on a website lies over whatever is behind it, with soft edges. Each time one image is laid over another, and per pixel a number says how much of the front image covers the back one. In AI you meet the same mix: a model that cuts an object out of a photo gives such a number per pixel, and pasting cut-out objects onto many backgrounds is a cheap way to make more training photos.

    **An everyday example.** A weather presenter never wears green: everything green in front of the wall would vanish too, and the map would show through their clothes.
    """)
    return


@app.cell(hide_code=True)
def predict_mix():
    guess_mix = L.guess(
        "mix",
        question="An orange pixel of a sticker, red 240, green 120 and blue 0, lies over a blue pixel of the background, (0, 0, 240). The sticker covers a quarter of it: the background shows through for three quarters. What color comes out?",
        options={
            "orange": "(240, 120, 0)",
            "half": "(120, 60, 120)",
            "quarter": "(60, 30, 180)",
            "three": "(180, 90, 60)",
        },
        reveal="**(60, 30, 180)**: mostly blue, with a touch of orange. Each color is a quarter of the sticker plus three quarters of the background: 0.25 × 240 + 0.75 × 0 = 60 red, 0.25 × 120 + 0.75 × 0 = 30 green, and 0.25 × 0 + 0.75 × 240 = 180 blue. (240, 120, 0) is the sticker alone, as if it covered everything; (120, 60, 120) half of each; and (180, 90, 60) three quarters of the sticker, the weights the wrong way around.",
    )
    guess_mix
    return (guess_mix,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A PNG can store a fourth number per pixel, next to red, green and blue: **alpha**, how much the pixel covers what lies behind it. 255 covers it completely, 0 not at all, and 102 for 40 percent. An image with an **alpha channel** has shape `(rows, columns, 4)`: red, green, blue and alpha, RGBA for short. A sticker has alpha 255 on the drawing, 0 around it, and values in between on its soft edge. A shadow is black with an alpha in between: it darkens what lies behind it, without hiding it.

        **Alpha compositing** lays a front image over a back image, pixel by pixel. With `a` the alpha from 0 to 1, the alpha channel divided by 255:

        `out = a * front + (1 - a) * back`

        for each of the three colors. At `a = 1` you see only the front, at `a = 0` only the back, and in between a mix. A shadow pixel of this lesson's sticker with alpha 102, so `a = 0.4`, makes the table behind it 0.6 times as bright.

        The shapes have to fit. `sticker[:, :, :3]` holds the colors, shape `(400, 600, 3)`. `sticker[:, :, 3]` gives the alpha with shape `(400, 600)`: one axis less, and lined up on the right it does not fit the colors. `sticker[:, :, 3:]` gives the same numbers with the axis kept, shape `(400, 600, 1)`, and NumPy stretches that over the three colors, as in [Broadcasting](/course/open/np-broadcasting?lang=en).

        A photo in front of a green screen has no alpha channel: its alpha has to come from the colors. That is a **chroma key**. Every pixel whose hue is the green of the screen and whose saturation is strong belongs to the screen and gets alpha 0; every other pixel gets alpha 1. A screen pixel of this lesson is red 47, green 195 and blue 92: hue 0.38 and saturation 0.76, so screen. A pixel of the lime is (100, 187, 47): green too, but yellowish green, with hue 0.27. A key whose green starts at 0.3 keeps the lime; one that starts at 0.2, as the rule for green in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en), makes it vanish.
        """),
        L.watch_out("Divide the alpha channel by 255 before you mix. In uint8, `1 - a` with `a` at 255 does not give -254: it wraps around to 2."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **alpha channel** | a fourth number per pixel: how much the pixel covers what lies behind it, from 0 to 255 in a PNG | `sticker[:, :, 3:]` |
        | **alpha compositing** | laying one image over another, with per pixel `a` times the front plus `1 - a` times the back | `a * front + (1 - a) * back` |
        | **chroma key** | making the alpha of a photo from its colors: 0 where the color of the screen is, 1 elsewhere | `1 - screen` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_alpha():
    mo.vstack([
        mo.md("**A sticker and its alpha channel**"),
        mo.Html(C.alpha_picture({"alpha": "alpha channel", "over": "sticker over the coffee photo",
                                 "alt": "On the left, the alpha channel of the sticker as a gray image: black where it is 0, white where it is 255. Four white disks, the fruit, each with a soft gray patch below and to the left of it, the shadow. On the right, the sticker laid over the coffee photo: an orange at the top right, a lime and a lemon at the bottom right, and an apple at the bottom left, each with a soft shadow on the table."})),
        mo.md(r"""
        On the left, the alpha channel as a gray image: white where the sticker covers, black where it lets everything through. The fruit is white, with a thin gray edge; the shadows are dark gray, because they cover a little. On the right, the sticker over the coffee photo, mixed with `a * front + (1 - a) * back`: the fruit lies on the table, and the shadows darken the wood under it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    The course laid the same sticker over a green screen and saved the photo. With the slider you choose where the green of the key starts: every pixel with a hue from there to 0.45 and a saturation above 0.5 counts as screen. On the left you see the alpha the key makes, white where it keeps the photo and black where it takes it for screen; on the right the photo laid over the coffee photo with that alpha.
    """)
    return


@app.cell(hide_code=True)
def explore_key_cell():
    def _say(low):
        _out = C.key_outcome(C.key(low))
        if _out["left"]:
            return "The screen lies at 0.38 to 0.40, below the start of the range, so none of it goes."
        _names = {"orange": "the orange (0.075)", "lemon": "the lemon (0.14)", "lime": "the lime (0.27)"}
        _gone = [_names[name] for name in ("orange", "lemon", "lime") if _out["lost"][name]]
        if not _gone:
            return "All four pieces of fruit stay, and all of the screen goes."
        if len(_gone) == 1:
            return (f"The lime, at 0.27, counts as screen too: {C.grouped(_out['lost']['lime'], ',')} of its "
                    f"{C.grouped(C.fruit_parts()['lime'].sum(), ',')} pixels vanish, all but its white shine.")
        _stay = "The orange, at 0.075, and the apple, at 0, stay." if not _out["lost"]["orange"] else "The apple, at hue 0, stays."
        return (f"{', '.join(_gone[:-1]).capitalize()} and {_gone[-1]} count as screen too, and vanish; only their "
                "white shine stays, because it has almost no saturation. " + _stay)

    def _frame(low):
        _html = C.key_frame(low, {"alpha": "alpha of the key", "over": "over the coffee photo",
                                  "alt": "On the left, the alpha of a key whose green starts at a hue of {low:.2f}: white where it keeps the photo, black where it takes it for screen. On the right, the photo in front of the green screen laid over the coffee photo with that alpha."})
        return _html, _say(low)

    explore_key = L.explore("key", "green starts at", _frame, start=0.2, values=C.BOUNDS, role="",
                            task="Find where the green can start so that all four pieces of fruit stay and no green is left.")
    explore_key
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 0.30, every piece of fruit stays and the whole screen goes. At 0.20, where the rule for green of [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en) starts, and at 0.25, the lime goes too: that rule was made to find the lime. At 0.40 the range misses the screen, which lies at 0.38 to 0.40. Look closely at the edges of the fruit, and at where the shadows went: exercise 4 comes back to them.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`.astype(np.uint8)` drops what comes after the point: 67.9 becomes 67. Add 0.5 first, and it rounds: `(out + 0.5).astype(np.uint8)`."),
        mo.md(r"""
        ### Recipe

        To lay a sticker `sticker` with an alpha channel over a background `back` of the same size:

        1. Take the colors: `front = sticker[:, :, :3]`.
        2. Take the alpha from 0 to 1, with its axis kept: `a = sticker[:, :, 3:] / 255`.
        3. Mix: `out = a * front + (1 - a) * back`.
        4. Round back to uint8: `(out + 0.5).astype(np.uint8)`.

        For a photo in front of a green screen, the alpha comes from the colors instead of steps 1 and 2:

        5. `hsv = rgb_to_hsv(photo / 255)`, with `h = hsv[:, :, 0:1]` and `s = hsv[:, :, 1:2]`: the hue and the saturation, each with its axis kept.
        6. `screen` is True where `h` lies in the green of the screen and `s` is strong, joined with `&`.
        7. `a = 1 - screen`, which has the shape `(rows, columns, 1)` of `h`, and mix the photo itself as the front.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the sticker and the coffee photo, and prints their shapes. It takes the colors and the alpha, steps 1 and 2 of the recipe, and prints the shape of the alpha and how many pixels have alpha 1 and alpha 0. It lays the sticker over the coffee photo, steps 3 and 4, prints one pixel of a shadow before and after, and shows the result.
    """)
    return


@app.cell
def example_sticker():
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    sticker = np.array(Image.open("data/photos/fruit-sticker.png"))
    coffee = np.array(Image.open("data/photos/coffee.png"))
    print(sticker.shape, coffee.shape)
    # 1. and 2. the colors, and the alpha from 0 to 1 with its axis kept
    front = sticker[:, :, :3]
    a = sticker[:, :, 3:] / 255
    print(a.shape, (a == 1).sum(), (a == 0).sum())
    # 3. the sticker over the coffee photo
    out = a * front + (1 - a) * coffee
    # 4. back to uint8, rounded
    out = (out + 0.5).astype(np.uint8)
    # a pixel of the shadow under the orange
    print(sticker[92, 495], coffee[92, 495], out[92, 495])
    plt.imshow(out)
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sticker has four numbers per pixel, the coffee photo three, and `a` has shape `(400, 600, 1)`: one number per pixel, which NumPy stretches over the three colors. 15,544 pixels have alpha 1, the fruit away from its edges, and 220,176 alpha 0; the 4,280 in between are the edges and the shadows. The shadow pixel is black with alpha 102, so `a = 0.4`: the wood behind it, (177, 99, 58), becomes (106, 59, 35), 0.6 times as bright and rounded.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_axis():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The axis of the alpha

    `sticker` is the sticker of the worked example, shape `(400, 600, 4)`, and `coffee` the coffee photo, shape `(400, 600, 3)`. Someone takes the alpha without its axis: `a = sticker[:, :, 3] / 255`.
    """)
    return


@app.cell(hide_code=True)
def choice_axis_cell():
    choice_axis = L.choice(
        "axis",
        question="What does `a * sticker[:, :, :3]` give then?",
        options={
            "error": "An error: the shapes `(400, 600)` and `(400, 600, 3)` do not fit",
            "mix": "The colors of the sticker, each times its alpha",
            "alpha": "The alpha, three times over",
        },
        correct="error",
        why={
            "error": "Lined up on the right, 600 meets 3: they differ, and neither is 1. `sticker[:, :, 3:]` keeps the axis, with length 1: shape `(400, 600, 1)`, which NumPy stretches over the three colors.",
            "mix": "That is what `sticker[:, :, 3:] / 255` gives, with its axis kept. Without it, `a` has shape `(400, 600)`, and lined up on the right, 600 meets 3.",
            "alpha": "NumPy does not copy an array to make it fit; it stretches only an axis of length 1. `(400, 600)` against `(400, 600, 3)`: lined up on the right, 600 meets 3.",
        },
    )
    choice_axis
    return (choice_axis,)


@app.cell(hide_code=True)
def text_exercise_over():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One image over another

        Complete `over(sticker, back)`, which lays a sticker with an alpha channel over a background of the same size, and gives back the mix as floats, before rounding. Replace the two `...`: `a` is the alpha from 0 to 1, with its axis kept, and the function gives back the mix of steps 1 to 3 of the recipe. The check calls it with two pixels: an orange one at alpha 51, a fifth, over blue, and a shadow pixel at alpha 102 over the wood of the table:

        ```python
        sticker = np.array([[[240, 120, 0, 51], [0, 0, 0, 102]]], dtype=np.uint8)
        back = np.array([[[0, 0, 240], [180, 120, 80]]], dtype=np.uint8)
        over(sticker, back)
        ```

        That should give `[[[48, 24, 192], [108, 72, 48]]]`. The check also tries pixels with alpha 0 and 255, and a sticker darker than its background.
        """),
        L.task("Make `over(sticker, back)` mix the sticker with the background, with its alpha."),
    ])
    return


@app.cell(hide_code=True)
def exercise_over():
    over_code = L.code_box("over", lines=4)
    over_code
    return (over_code,)


@app.cell(hide_code=True)
def check_over(over_code):
    result_over = L.check("over", over_code.value)
    result_over
    return (result_over,)


@app.cell(hide_code=True)
def hints_over():
    L.hints(
        "over",
        [
            "Steps 2 and 3 of the recipe: first the alpha from 0 to 1, then the mix.",
            "`a` is the last number of every pixel, with `3:` so the axis stays, divided by 255. The mix gives the colors `a` and the background `1 - a`.",
            "`a = sticker[:, :, ___] / 255` and `return a * sticker[:, :, ___] + (1 - a) * back`.",
        ],
        why="For the orange pixel: `a = 51 / 255 = 0.2`, so 0.2 × 240 + 0.8 × 0 = 48 red, 0.2 × 120 = 24 green, and 0.8 × 240 = 192 blue. For the shadow: `a = 0.4` and the color is black, so 0.6 × (180, 120, 80) = (108, 72, 48).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_key():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A key for the green screen

        `data/photos/green-screen.png` is the photo of the slider: the sticker in front of a green screen, 400 rows by 600 columns, without an alpha channel. The box reads it and gives you the hue `h` and the saturation `s` of every pixel, each with its axis kept, as `3:` kept the axis of the alpha: shape `(400, 600, 1)`. Make `screen` a mask that is True on the screen and False on the fruit, and `alpha` the alpha of the photo: 0 on the screen, 1 elsewhere, with the same shape `(400, 600, 1)`, so that it fits the three colors of the photo. The screen's green lies at a hue of 0.38 to 0.40; the lime at 0.27.
        """),
        L.task("Make `alpha` keep all four pieces of fruit and drop the green screen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_key():
    key_code = L.code_box("key", lines=11)
    key_code
    return (key_code,)


@app.cell(hide_code=True)
def check_key(key_code):
    result_key = L.check("key", key_code.value)
    result_key
    return (result_key,)


@app.cell(hide_code=True)
def hints_key():
    L.hints(
        "key",
        [
            "Steps 6 and 7 of the recipe. With the slider you found where the green can start.",
            "`screen` joins three comparisons with `&`: the hue above the start, the hue below 0.45, and the saturation above 0.5. `1 - screen` turns True into 0 and False into 1.",
            "`screen = (h > ___) & (h < 0.45) & (s > 0.5)` and `alpha = 1 - ___`.",
        ],
        why="The screen lies at a hue of 0.38 to 0.40, the lime at 0.27, so the green can start anywhere between them, such as at 0.3: `screen = (h > 0.3) & (h < 0.45) & (s > 0.5)`. `1 - screen` is 0 on the screen and 1 elsewhere, with the shape of `h`, `(400, 600, 1)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_edges():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Up close

        The course laid the green-screen photo over the coffee photo twice: with the key whose green starts at 0.3, and with the sticker's own alpha. The orange, up close:
        """),
        mo.Html(C.edge_picture({"key": "with the key", "sticker": "with the sticker's alpha",
                                "alt": "The orange up close, twice. On the left, laid over the coffee photo with the key: its edge is a staircase of whole pixels, with a thin green rim here and there, and there is no shadow. On the right, with the sticker's own alpha: a smooth edge, and a soft shadow on the lower left."})),
    ])
    return


@app.cell(hide_code=True)
def explain_edges_cell():
    explain_edges = L.explain(
        "edges",
        question="With the key, the edge is a staircase with a thin green rim, and the shadow is gone. Where do those come from? Think of what a pixel on the edge of the orange held in the green-screen photo, and which alpha the key gives it. Which alpha should such a pixel get instead?",
        model_answer="**A pixel on the edge is partly orange and partly screen:** the photo is the sticker over the screen, so it holds a mix of the two. **The key gives every pixel 0 or 1.** A mixed pixel with enough orange to pull its hue below the start of the key gets 1, and brings its share of green along: the rim. One with less orange gets 0, and its share of orange is lost: every edge pixel is all or nothing, so the edge becomes a staircase. **The shadow is darker green:** it has the hue and the saturation of the screen, so the key takes it for screen. **An edge pixel needs an alpha between 0 and 1,** like the sticker's own alpha on its soft edge, so that it mixes with the coffee photo the way the sticker did. A softer key makes such an alpha from how green a pixel is, and takes the green of its share of screen out of its color as well.",
        points=[
            "you say that a pixel on the edge holds a mix of orange and screen",
            "you say that the key gives only 0 or 1, so a mixed pixel keeps its green or loses its orange",
            "you say that the shadow has the hue and saturation of the screen, so it counts as screen",
            "you say that an edge pixel needs an alpha between 0 and 1, so that it mixes softly",
        ],
    )
    explain_edges
    return (explain_edges,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** One image over another is one mix per pixel: `a` times the front plus `1 - a` times the back, with `a` from the alpha channel, which a PNG can store as a fourth number. A chroma key makes that alpha from the colors: with a mask on hue and saturation, as in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en), it is 0 or 1, which keeps the fruit and drops the screen, but leaves a staircase and a green rim on the edges. Film studios use a soft key: it gives an alpha between 0 and 1, depending on how green a pixel is, and takes the green out of the edges. Newer tools let a neural network work out the alpha of every pixel, even without a green screen. The same mix fades one scene into another, with `a` going from 0 to 1 over time, and lays subtitles over a film. This lesson ends the module on working with images; the next module, on using a pretrained model, starts with *Get a photo ready for a model*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_axis, explain_edges, result_key, result_over):
    L.wrap_up(choice_axis, result_over, result_key, explain_edges)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
