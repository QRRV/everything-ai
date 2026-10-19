# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-images-as-arrays. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-images-as-arrays", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Selecting parts of an array](/course/open/np-indexing?lang=en) and [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_slice():
    quick_slice = L.choice(
        "quick-slice",
        question="`m = np.array([[1, 2, 3], [4, 5, 6]])`. What does `m[1, 0:2]` give?",
        options={
            "column": "`array([2, 5])`",
            "part": "`array([4, 5])`",
            "block": "`array([[1, 2], [4, 5]])`",
        },
        correct="part",
        why={
            "column": "That is column 1, from both rows. Here the first number picks a row. See the table of new words in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
            "part": "Row 1 is `[4, 5, 6]`, and the slice `0:2` takes its columns 0 and 1.",
            "block": "The `1` before the comma picks one row, not all of them. See the recipe in [Selecting parts of an array](/course/open/np-indexing?lang=en).",
        },
    )
    quick_slice
    return (quick_slice,)


@app.cell(hide_code=True)
def quick_check_show():
    quick_show = L.choice(
        "quick-show",
        question="What does `plt.show()` do on the last line of a cell?",
        options={
            "show": "It shows the chart that the lines above it drew",
            "new": "It draws a new, empty chart",
            "save": "It saves the chart as a file",
        },
        correct="show",
        why={
            "show": "The lines above draw the chart; `plt.show()` finishes it and shows it under the cell.",
            "new": "Look under a cell that ends with `plt.show()`: is the chart there empty? See the recipe in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "save": "Where does the chart end up: in a file, or under the cell? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
        },
    )
    quick_show
    return (quick_show,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model calculates only with numbers, so a picture has to become numbers before a model can do anything with it. To a computer, a picture is a grid of numbers: one number per dot, or three per dot in a color picture. Later in the course, a neural network learns to read handwritten digits, and all it gets is 28 rows of 28 numbers per digit. A photo from a phone has millions of such numbers.

    **An everyday example.** A cross-stitch pattern: a grid of squares, with in each square a number from the thread legend, which says the color to stitch.
    """)
    return


@app.cell(hide_code=True)
def predict_count():
    guess_count = L.guess(
        "count",
        question="A small grayscale image of a handwritten digit has 28 rows of 28 dots, and one number per dot. How many numbers is that?",
        options={
            "rows": "28",
            "sum": "56",
            "all": "784",
        },
        reveal="**784.** 28 rows of 28 numbers is 28 times 28, which is 784 numbers: one for every dot. As an array, its shape is `(28, 28)`.",
    )
    guess_count
    return (guess_count,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    An image is an array. In a grayscale image, every pixel is one number: its brightness. In a color image, every pixel has three numbers: how much red, green and blue it has. So a color image is a 3D array, an array in three directions: rows, columns and the three colors. Everything you know about arrays works on it: its shape, an index, slices and a mask.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **pixel** | one dot of an image; in the array, one place, found with its row and its column | `img[5, 32]` |
    | **grayscale image** | a 2D array with one brightness per pixel, from 0, black, to 1, white | shape `(rows, columns)` |
    | **color image** | a 3D array with three numbers per pixel: red, green and blue | shape `(rows, columns, 3)` |
    | **channel** | one of the colors, for every pixel: a 2D array of its own | `img[:, :, 0]` is the red channel |
    | **`plt.imread(path)`** | reads an image file as an array; a PNG file gives floats from 0 to 1, a JPG file whole numbers from 0 to 255 | `img = plt.imread(path)` |
    | **`plt.imshow(img)`** | draws an array as an image; for one channel or a grayscale image, add `cmap="gray"`, or matplotlib colors it from purple to yellow | `plt.imshow(img)` |

    An index into a color image can have three parts: the row, the column and the color, with commas between them. `:` takes everything in that direction, as before, and an index with only two parts, such as `img[0:5, 0:5]`, keeps all three colors. `cmap="gray"` is a keyword argument, like `axis=0` in [Sums and means along an axis](/course/open/np-axis?lang=en).
    """)
    return


@app.cell(hide_code=True)
def picture_plus():
    _numbers, _image = C.picture({"array": "the array", "image": "the image",
                                  "alt": "A plus sign in black on white, 5 by 5 pixels, with a gray pixel in each corner between its arms."})
    mo.vstack([
        mo.md("**The same plus sign, twice**"),
        mo.hstack([mo.Html(_numbers), mo.Html(_image)], justify="start", gap=2, align="center"),
        mo.md(r"""
        On the left, the numbers of a grayscale image of 5 by 5 pixels; on the right, the image those numbers make with `plt.imshow(plus, cmap="gray")`. Every 0 is a black pixel, every 1 a white one, and 0.5 is gray. Row 0 is at the top, as in a table.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Row 0 is at the top of an image, and an index gives the row first: `img[row, column]`. That is the other way round from `x, y` on a chart, where x goes first and y goes up."),
        L.watch_out("Here, a *channel* is one layer of numbers along the last direction of an image, such as all its red values. It has nothing to do with a TV channel."),
        mo.md(r"""
        ### Recipe

        To work with an image as an array:

        1. Read it: `img = plt.imread(path)`; a PNG file gives floats from 0 to 1.
        2. Look at its shape: two numbers for a grayscale image, three for a color image, the last one 3.
        3. Show it: `plt.imshow(img)` and `plt.show()`; for a grayscale image or one channel, add `cmap="gray"`.
        4. Take a part with slices, the rows first: `img[top:bottom, left:right]`; one channel with `img[:, :, c]`.
        5. Count pixels with a mask, for example on one channel: `len(red[red > 0.9])`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small landscape that the course made: `lessons/data-images-as-arrays/landscape.png`, 24 pixels high and 40 wide. The comments with `#` say which step of the recipe each part is. Run the three cells.
    """)
    return


@app.cell
def example_read():
    import matplotlib.pyplot as plt

    # 1. read the image file as an array
    img = plt.imread("lessons/data-images-as-arrays/landscape.png")
    # 2. its shape and dtype, and one pixel: row 5, column 32, in the sun
    print(img.shape)
    print(img.dtype)
    print(img[5, 32])
    # 3. show it
    plt.imshow(img)
    plt.show()
    return img, plt


@app.cell
def example_crop(img, plt):
    # 4. a part: rows 16 to 23 and columns 0 to 11, the bottom-left corner
    _hills = img[16:24, 0:12]
    print(_hills.shape)
    plt.imshow(_hills)
    plt.show()
    return


@app.cell
def example_channel(img, plt):
    # 4. one channel: the red of every pixel, shown as a grayscale image
    _red = img[:, :, 0]
    print(_red.shape)
    # 5. count the pixels with much red, with a mask on the channel
    print(len(_red[_red > 0.9]))
    plt.imshow(_red, cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The shape is `(24, 40, 3)`: 24 rows, 40 columns and 3 colors; the dtype `float32` is a float that takes less memory than `float64`. The pixel in the sun is `[0.98039216 0.84313726 0.23529412]`: much red and green, little blue, and red and green light together make yellow. The bottom-left corner is a part of 8 rows and 12 columns, `(8, 12, 3)`, shown larger: hills, and a strip of sky above them. The red channel is a 2D array, `(24, 40)`: there, the sun is white, because it has the most red, the sky is gray, and the green hills, with the least red, are black. For one channel, `imshow` makes the lowest number black and the highest white. 44 pixels have more red than 0.9: the pixels of the sun.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The shape of a photo
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "color-shape",
        question="A color photo is 100 pixels high and 200 pixels wide. What is the shape of its array?",
        options={
            "wide": "`(200, 100, 3)`",
            "flat": "`(100, 200)`",
            "color": "`(100, 200, 3)`",
        },
        correct="color",
        why={
            "wide": "The rows come first: how many rows does an image have that is 100 pixels high?",
            "flat": "That is the shape of a grayscale image. How many numbers does a color pixel have? See the table of new words above.",
            "color": "100 rows, 200 columns, and 3 colors per pixel.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_sun():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Cut out the sun

        The box reads the landscape as `img`, shape `(24, 40, 3)`. The sun is in rows 2 to 9 and columns 29 to 36. Make `sun`: that part of the image, with all three colors. Replace the `...`. The last line is `sun`, so the check sees it.
        """),
        L.task("Make `sun`: rows 2 to 9 and columns 29 to 36 of `img`, with all three colors."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sun():
    box_sun = L.code_box("sun", lines=6)
    box_sun
    return (box_sun,)


@app.cell(hide_code=True)
def check_sun(box_sun):
    result_sun = L.check("sun", box_sun.value)
    result_sun
    return (result_sun,)


@app.cell(hide_code=True)
def hints_sun():
    L.hints(
        "sun",
        [
            "Step 4 of the recipe: a part with slices, the rows first, then the columns.",
            "A slice stops just before its second number: for rows 2 to 9, the slice is `2:10`.",
            "`sun = img[2:10, ___:___]`.",
        ],
        why="`img[2:10, 29:37]` takes rows 2 to 9 and columns 29 to 36. Without a third part, the index keeps all three colors: shape `(8, 8, 3)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_red():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. How many red pixels?

        Finish `red_pixels(rgb, limit)`. `rgb` is a color image, given as a list of rows of pixels, and the function already turns it into an array, `img`. It gives back how many pixels have more red than `limit`. Replace the `...`. The check calls the function with this image of 2 rows of 3 pixels, and `limit` 0.5:

        ```python
        [[[0.0, 0.0, 0.0], [0.5, 0.5, 0.8], [0.8, 0.2, 0.0]],
         [[1.0, 0.5, 0.0], [1.0, 0.2, 0.2], [0.2, 0.2, 0.5]]]
        ```

        That should give `3`. It also tries other images and limits.
        """),
        L.task("Make `red_pixels(rgb, limit)` count the pixels with more red than `limit`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_red():
    box_red = L.code_box("red-pixels", lines=8)
    box_red
    return (box_red,)


@app.cell(hide_code=True)
def check_red(box_red):
    result_red = L.check("red-pixels", box_red.value)
    result_red
    return (result_red,)


@app.cell(hide_code=True)
def hints_red():
    L.hints(
        "red-pixels",
        [
            "Two steps: first take the red channel, then count its pixels with a mask, as in step 5 of the recipe.",
            "Red is the first color, so its channel has `0` as the third part of the index. The mask compares that channel with `limit`.",
            "`red = img[___]`, and then `return len(red[red ___ limit])`.",
        ],
        why="`img[:, :, 0]` is the red channel: one number per pixel. `red > limit` is `True` for every pixel with more red than `limit`, `red[red > limit]` keeps those numbers, and `len` counts them: 0.8, 1.0 and 1.0, so 3.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Selecting parts of an array](/course/open/np-indexing?lang=en) you took rows, columns and parts out of a matrix, and picked numbers with a mask; an image is such an array, and a part of it is a slice. In [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) you drew numbers as a chart; `plt.imshow` draws them as an image. Later in the course, a neural network learns to read handwritten digits from arrays like these. In the next lesson, *Text is a sequence of numbers*, you turn text into numbers too.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_red, result_sun):
    L.wrap_up(choice_shape, result_sun, result_red)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
