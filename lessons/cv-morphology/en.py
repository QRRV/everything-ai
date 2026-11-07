# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-morphology. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-morphology", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en) and [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_min():
    quick_min = L.choice(
        "quick-min",
        question="`mask = np.array([[True, True], [False, True]])`. What does `mask.min()` give?",
        options={
            "false": "`False`",
            "true": "`True`",
            "mean": "`0.75`",
        },
        correct="false",
        why={
            "false": "Python counts False as 0 and True as 1, so the smallest of the four values is False. This lesson builds on that: the smallest value of a group of pixels is True only when every one of them is True.",
            "true": "Three of the four are True, but `min()` looks for the smallest value. Which is smaller, True or False? See *The idea* of [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en).",
            "mean": "0.75 is the mean, `mask.mean()`: three ones and a zero, divided by four. `min()` gives the smallest value.",
        },
    )
    quick_min
    return (quick_min,)


@app.cell(hide_code=True)
def quick_check_red():
    quick_red = L.choice(
        "quick-red",
        question="The rule for red of [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en) is `((h < 0.05) | (h > 0.95)) & (s > 0.5)`, with `h` the hue and `s` the saturation. Which pixel does it pick?",
        options={
            "pale": "Hue 0.97, saturation 0.4",
            "red": "Hue 0.97, saturation 0.8",
            "green": "Hue 0.3, saturation 0.9",
        },
        correct="red",
        why={
            "red": "0.97 is above 0.95, so the hue is red, near 1; and 0.8 is above 0.5. Both sides of `&` are True.",
            "pale": "The hue is red, but `&` also asks for a saturation above 0.5, and 0.4 is not: a pale red, like the pink of a tablecloth.",
            "green": "The saturation is high, but is 0.3 below 0.05 or above 0.95? A hue of 0.3 is green: see the strip of hues in [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en).",
        },
    )
    quick_red
    return (quick_red,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A threshold gives a binary image, and the binary image of a real picture is rarely clean. Here and there a pixel of the background passes the threshold and becomes a white speck, and somewhere part of an object does not pass and leaves a hole: a reflection, a shadow, a letter the printer left too faint. Before a program can count the objects or measure them, it has to tidy that up. Morphological operations do that by looking at the white shapes instead of at one pixel at a time: what is too small to be an object goes, and a hole too narrow to be real is filled. They are simple and fast, and you find them in software that reads scanned text, counts cells under a microscope, or checks parts on a factory line.

    **An everyday example.** You cut the apples out of a printed photo with scissors. Tiny red specks on the tablecloth you leave alone: too small to be an apple. And you do not cut around the white shine of the lamp on an apple: you cut along the outline, and the shine stays part of the apple. Erosion and dilation let a program do the same with a binary image.
    """)
    return


@app.cell(hide_code=True)
def predict_back():
    guess_back = L.guess(
        "back",
        question="The rule for red of [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en) picks the apples of the drawing of fruit. With a saturation threshold of 0.45 instead of 0.5, it also picks 1331 specks of the tablecloth, each of 1 to 4 pixels, and the shine of the lamp leaves a small hole in every apple. Erosion with a square of 3 by 3 makes every white shape one pixel thinner all around; dilation with the same square makes every white shape one pixel thicker. What is left after an erosion followed by a dilation?",
        options={
            "same": "The same image: the two undo each other",
            "apples": "The apples, about as they were, without the specks",
            "thinner": "Everything one pixel thinner than before",
        },
        reveal="**The apples, about as they were, without the specks.** No speck is big enough to hold a square of 3 by 3, so the erosion removes every one of them, and the dilation cannot grow back what is no longer there. An apple keeps a core, and the dilation grows that back to its old size: 13,511 of the 13,612 apple pixels are white again. The other 101 are the holes of the shine, which were never white. Only 14 pixels of the tablecloth stay, where a speck touches an apple.",
    )
    guess_back
    return (guess_back,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **morphological operation** changes a binary image by looking at every pixel through a small shape, the **structuring element**. In this lesson that is a square of `k` by `k` pixels with the pixel in its middle, such as 3 by 3: the window of the pixel. Python counts False as 0 and True as 1, so the two basic operations need only the smallest and the largest value of the window:

        - **Erosion**: a pixel stays white only when every pixel of its window is white, so when the smallest value of the window is True. Every white shape loses its outer layer; a shape that the square does not fit in disappears, and a hole grows.
        - **Dilation**: a pixel becomes white when any pixel of its window is white, so when the largest value is True. Every white shape gets an extra layer; a hole narrower than the square fills up, and two shapes close together grow into one.

        Erosion and dilation do not undo each other, and that is what makes them useful together:

        - An **opening** is an erosion followed by a dilation, with the same square. What the square does not fit in is gone after the erosion, and the dilation cannot bring it back; the rest grows back to about its old shape. An opening removes specks.
        - A **closing** is a dilation followed by an erosion. A hole or a gap that the square does not fit in is filled by the dilation, and the erosion does not open it again; the rest shrinks back to about its old shape. A closing fills holes.

        An example on one row, with a window of three pixels: the pixel and its neighbors on the left and on the right. 1 is white. The row has a speck, a piece of three pixels, a gap of one and a piece of four:

        ```
        row       0 0 1 0 0 0 1 1 1 0 1 1 1 1 0 0
        erosion   0 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0
        dilation  0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0
        opening   0 0 0 0 0 0 1 1 1 0 1 1 1 1 0 0
        closing   0 0 1 0 0 0 1 1 1 1 1 1 1 1 0 0
        ```

        The erosion removes the speck and takes a pixel off both ends of each piece. The dilation grows the speck to three pixels, makes each piece longer and fills the gap. The opening is the dilation of the erosion: the speck is gone, and both pieces are back as they were, with the gap between them. The closing is the erosion of the dilation: the gap is filled, and the speck and the two pieces are as they were.
        """),
        L.watch_out("The order is part of the name. An opening starts with the erosion and a closing with the dilation; with the same two steps the other way around, you get the other operation."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **morphological operation** | a change of a binary image that looks at its white shapes through a small shape, the structuring element | `erode` and `dilate` in the worked example; in OpenCV `cv2.erode` and `cv2.dilate` |
        | **structuring element** | the small shape a morphological operation looks through, with the pixel in its middle; here a square of `k` by `k` | the window `img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]`, for 3 by 3 |
        | **erosion** | a pixel stays white only when its whole window is white: shapes shrink, specks disappear | `window.min()` |
        | **dilation** | a pixel becomes white when any pixel of its window is white: shapes grow, narrow holes fill | `window.max()` |
        | **opening** | an erosion, then a dilation with the same square: removes what the square does not fit in | `dilate(erode(mask))` |
        | **closing** | a dilation, then an erosion with the same square: fills holes and gaps the square does not fit in | `erode(dilate(mask))` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_steps():
    mo.vstack([
        mo.md("**Around the apple at the top left, step by step**"),
        mo.Html(C.steps_picture({"drawing": "The drawing", "red": "Red, with 0.45", "erosion": "Erosion, 3 by 3",
                                 "opening": "Opening, 3 by 3", "dilation": "Dilation of the opening",
                                 "closing": "Closing of the opening",
                                 "alt": "Six pictures of the same part of the drawing of fruit, around the apple at the top left, as black and white squares of pixels. The drawing in color: a red apple with a white shine at its top left, on a pink checked tablecloth. Red, with 0.45: the apple as a white disk with a small black hole where the shine is, and around it many white specks of one to four pixels, in the darker checks of the tablecloth. Erosion, 3 by 3: no specks, the disk a little smaller, the hole bigger. Opening, 3 by 3: the disk as big as before, with its small hole, and no specks. Dilation of the opening: the disk a little bigger, the hole a tiny dot. Closing of the opening: the disk as big as before, the hole about as big as before."})),
        mo.md(r"""
        White is what the rule for red picks. The erosion removes every speck, makes the apple one pixel smaller all around, and its hole twice as big: 51 pixels instead of 25. The dilation of that is the opening: the apple as it was, with its hole of 25 pixels, and no speck. The dilation of the opening shrinks the hole to 7 pixels, but the erosion that completes the closing opens it again, to 23: a square of 3 by 3 fits inside the hole.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("At the border, part of the window lies outside the image. Only the part inside counts: `max(r - 1, 0)` keeps the start of the slice at 0 or more. A slice that starts below 0 counts from the end, the way `img[-1]` is the last row, and the window would be empty. The end may go past the last row: a slice just stops there."),
        L.watch_out("Store the result in a new array, `out`, and never in the image you are reading. If the erosion changed `img` itself while it ran, the next pixel would look at neighbors that had already changed."),
        mo.md(r"""
        ### Recipe

        To tidy up a binary image `img` with a square of `k` by `k`:

        1. Erosion: make an array `out` of False, of the same shape. For every pixel `(r, c)`, take its window, `img[max(r - half, 0):r + half + 1, max(c - half, 0):c + half + 1]`, with `half = k // 2`: how far the square reaches to each side. `//` divides and rounds down: `3 // 2` is 1, and `5 // 2` is 2. Store the smallest value of the window, `window.min()`, in `out[r, c]`.
        2. Dilation: the same, with the largest value, `window.max()`.
        3. Remove the specks with an opening: first the erosion, then the dilation.
        4. Fill the holes with a closing: first the dilation, then the erosion.
        5. Take the smallest square that does the job: a bigger square also changes the shapes you want to keep.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell picks the red pixels of the drawing of fruit, `data/photos/fruit.png`, with the rule for red and 0.45, and reads what the course drew, `data/photos/fruit-labels.png`, where 1 is an apple. It prints how many pixels are red, how many belong to an apple, and how many are wrong: red where there is no apple, or not red where there is one. The second cell writes erosion and dilation with a square of 3 by 3, removes the specks with an opening, and fills the holes with a closing. It prints how many pixels are white after each step and how many are wrong, and shows the result.
    """)
    return


@app.cell
def example_mask():
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    labels = np.array(Image.open("data/photos/fruit-labels.png"))
    hsv = rgb_to_hsv(img / 255)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # the rule for red, with 0.45 instead of 0.5: a binary image of True and False
    mask = ((h < 0.05) | (h > 0.95)) & (s > 0.45)
    # what the course drew: 1 is an apple
    apples = labels == 1
    print(mask.sum(), apples.sum(), (mask != apples).sum())
    return apples, mask, np, plt


@app.cell
def example_morphology(apples, mask, np, plt):
    def erode(img):
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                # the window of 3 by 3 around (r, c): only the part inside the image
                window = img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
                # 1. erosion: white only when every pixel of the window is white
                out[r, c] = window.min()
        return out


    def dilate(img):
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                window = img[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
                # 2. dilation: white when any pixel of the window is white
                out[r, c] = window.max()
        return out


    # 3. the opening removes the specks: erosion, then dilation
    eroded = erode(mask)
    opened = dilate(eroded)
    # 4. the closing fills the holes: dilation, then erosion
    closed = erode(dilate(opened))
    print(mask.sum(), eroded.sum(), opened.sum(), closed.sum())
    print((mask != apples).sum(), (opened != apples).sum(), (closed != apples).sum())
    # True is white, False black
    plt.imshow(closed, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The rule picks 16,190 red pixels, and the course drew 13,612 apple pixels: 2780 pixels are wrong. 2679 of them are pixels of the tablecloth, 2634 in the 1331 specks and 45 at the edge of an apple, and 101 are the holes of the shine. After the erosion 12,403 pixels are white: every speck is gone, and the apples are one pixel smaller all around. After the opening 13,525: the apples are back, and only 115 pixels are wrong, the 101 of the holes and 14 of the tablecloth at the edge of an apple. The closing of 3 by 3 fills only 12 of the 101 pixels of the holes, so 103 pixels are wrong, and in the picture under the cell every apple still has its hole. A hole of the shine is 5 or 6 pixels across, and a square of 3 by 3 fits inside it. In exercise 3 you look for a bigger square.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What does erosion keep?

    A small binary image, with a white square of 3 by 3 in the middle; 1 is white:

    ```
    0 0 0 0 0
    0 1 1 1 0
    0 1 1 1 0
    0 1 1 1 0
    0 0 0 0 0
    ```
    """)
    return


@app.cell(hide_code=True)
def choice_block_cell():
    choice_block = L.choice(
        "block",
        question="Which pixels are white after an erosion with a square of 3 by 3?",
        options={
            "middle": "Only the pixel in the middle",
            "nine": "All nine pixels of the square",
            "ring": "The eight pixels around the middle",
            "none": "None",
        },
        correct="middle",
        why={
            "middle": "Only the pixel in the middle has a window that is white everywhere: its window is the whole white square. Every other white pixel has a black pixel of the border in its window. A white shape loses its outer layer, and of a square of 3 by 3 only the middle is left.",
            "nine": "That is what an opening gives: the erosion, and then the dilation that grows the square back. What is left after the erosion alone? Look at the window of the pixel at the top left of the white square: is it white everywhere?",
            "ring": "Those are the pixels that the erosion takes away: each of them has a black pixel in its window. Which white pixel has a window without black?",
            "none": "Then the square of 3 by 3 would not fit anywhere in the white. Does it fit somewhere? Look at the window of the pixel in the middle.",
        },
    )
    choice_block
    return (choice_block,)


@app.cell(hide_code=True)
def text_exercise_dilate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Dilation with any square

        The worked example dilates with a square of 3 by 3. Complete `dilate(img, k)`, which dilates with a square of `k` by `k`: `k` is 1, 3, 5 or another odd number. It gives back a binary image of the same shape, in which a pixel is white when any pixel in the square around it is white; only the part of the square inside the image counts. `img` is a binary image, a list of rows of 0 and 1. The loops and `half = k // 2`, how far the square reaches to each side, are already there: replace the `...`. The check calls it with one white pixel in the middle:

        ```python
        dot = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
        dilate(dot, 3)
        ```

        That should give `[[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]`. It also tries other sizes, such as 5 and 1, and white pixels at the border.
        """),
        L.task("Make `dilate(img, k)` turn every pixel white that has a white pixel in its square of `k` by `k`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dilate():
    box_dilate = L.code_box("dilate", lines=14)
    box_dilate
    return (box_dilate,)


@app.cell(hide_code=True)
def check_dilate(box_dilate):
    result_dilate = L.check("dilate", box_dilate.value)
    result_dilate
    return (result_dilate,)


@app.cell(hide_code=True)
def hints_dilate():
    L.hints(
        "dilate",
        [
            "`half` is how far the square reaches to each side of its middle: 1 for 3 by 3, 2 for 5 by 5, 0 for 1 by 1.",
            "The rows of the window run from `r - half` to `r + half`. A slice stops just before its end, so write `r + half + 1`; and so that the start never goes below 0, write `max(r - half, 0)`. The columns go the same way, with `c`.",
            "`window = img[max(r - half, 0):r + half + 1, max(c - half, 0):___]`, and then `out[r, c] = window.max()`.",
        ],
        why="For `dilate(dot, 3)`, `half` is 1. The pixel at `(1, 1)` has the window of rows 0 to 2 and columns 0 to 2, and the white pixel at `(2, 2)` is in it, so it becomes white. The pixel at `(0, 0)` has the window of rows 0 to 1 and columns 0 to 1, without the white pixel, so it stays black. Every pixel that is at most one step away from `(2, 2)`, also diagonally, becomes white: a square of 3 by 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_size():
    mo.md(r"""
    ### Exercise 3. The square of the closing

    In the worked example, the closing of 3 by 3 left the holes of the shine almost as they were. With the slider you choose the square of the closing; the opening before it stays 3 by 3. On the left, the result; on the right, how many pixels are wrong at every size: white where the course drew no apple, or black where it drew one. The sentence under the picture says how many pixels of the holes are still black, and how many white shapes there are.
    """)
    return


@app.cell(hide_code=True)
def explore_size_cell():
    def _frame(k):
        _html, _f = C.size_frame(k, {"closed": "closing of {k} by {k}", "size": "size of the square", "wrong": "pixels wrong",
                                     "alt": "On the left, the four apples of the drawing of fruit as white shapes on black, after an opening of 3 by 3 and a closing of {k} by {k}; on the right, how many pixels are wrong for the squares 1 to 15: 115 at 1, 103 at 3, the fewest, 14, at 5 and at 7, then more again, 74 at 9 and up to 411 at 15. Size {k} is marked by a larger point."})
        _shapes = "4 white shapes, one per apple" if _f["shapes"] == 4 else f'{_f["shapes"]} white shapes: two apples have grown together'
        return (_html,
                f'Square of {k} by {k}: {_f["holes"]} pixels of the holes are still black, {_shapes}, {_f["wrong"]} pixels wrong.')

    explore_size = L.explore("size", "square of the closing", _frame, start=1, values=C.SIZES, role="hyper",
                             task="Find the smallest square that fills every hole.")
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
            "Move the slider one stop at a time, from 1 to the right, and read the first number in the sentence under the picture.",
            "A closing fills a hole when the square does not fit inside it. The holes of the shine are round, 5 or 6 pixels across at their widest: a square of 3 by 3 fits inside, but the corners of a square of 5 by 5 stick out.",
            "Look at the stops 3 and 5: after which one is no pixel of the holes black?",
        ],
        why="At 3, 89 pixels of the holes are still black: a square of 3 by 3 fits inside every hole. At 5, none is: a square of 5 by 5 fits in no hole, so the dilation fills them and the erosion does not open them again. 7 fills them too, with as few pixels wrong, but from 9 on the closing fills more than the holes: first the strip between the small apple and the right edge of the drawing, because the erosion does not look past the edge, then the gap between the two apples on the right.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ### Exercise 4. First the opening, then the closing
    """)
    return


@app.cell(hide_code=True)
def explain_order_cell():
    explain_order = L.explain(
        "order",
        question="An opening of 3 by 3 followed by a closing of 5 by 5 leaves 14 pixels of the drawing wrong. The other way around, first the closing of 5 by 5 and then the opening of 3 by 3, leaves 529 wrong: rough bumps of tablecloth on the edges of the apples. Why does the order matter so much?",
        model_answer="**Closing first:** a closing does not remove specks: the dilation grows a speck and the erosion shrinks it back. And a speck close to an apple grows into the apple during the dilation; the erosion after it does not cut it loose again, so the speck has become a bump on the apple. **Then the opening:** an opening only removes what the square does not fit in. A bump that is part of a big apple is often thick enough for the square, so it stays. **Opening first:** the specks are still small and on their own, so the erosion removes them all before they can stick to anything; then the closing only has the apples left to fill.",
        points=[
            "you say that a closing does not remove specks, but lets specks near an apple grow into it",
            "you say that the opening afterwards keeps those bumps, because they are part of a big shape that the square fits in",
            "you say that an opening first removes the specks while they are still small and on their own",
        ],
    )
    explain_order
    return (explain_order,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Morphological operations tidy up the binary images of [Black or white: a threshold and Otsu's method](/course/open/cv-otsu-threshold?lang=en) and of the rule for red of [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en): the threshold decides pixel by pixel, the opening and the closing look at the shapes. Erosion takes the smallest value of every window and dilation the largest, with a square of any size. The same minimum and maximum work on a grayscale image too, where they make light patches shrink or grow. After the opening and a closing of 5 by 5, the drawing has four white shapes without holes, one per apple. The next lesson, *Count the shapes: connected components*, counts them: it follows white pixels that touch, from neighbor to neighbor, and finds every shape, its size and the smallest box around it.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_block, explain_order, result_dilate, result_size):
    L.wrap_up(choice_block, result_dilate, result_size, explain_order)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
