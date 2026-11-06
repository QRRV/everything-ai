# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-image-files. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-image-files", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) and [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_channel():
    quick_channel = L.choice(
        "quick-channel",
        question="`img` is a color image of shape `(400, 600, 3)`. What is `img[:, :, 2]`?",
        options={
            "row": "Row 2 of the image, with all its pixels and colors",
            "blue": "The blue of every pixel: a 2D array of shape `(400, 600)`",
            "green": "The green of every pixel: a 2D array of shape `(400, 600)`",
        },
        correct="blue",
        why={
            "row": "A row is the first part of the index. Here, the third part is 2: which direction is that? See the table of new words in [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
            "blue": "The third part of the index picks the color: red is 0, green 1 and blue 2.",
            "green": "The colors go red, green, blue, and the count starts at 0. Which one is 2? See the table of new words in [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en).",
        },
    )
    quick_channel
    return (quick_channel,)


@app.cell(hide_code=True)
def quick_check_byte():
    quick_byte = L.choice(
        "quick-byte",
        question="How many different values can one byte hold?",
        options={
            "255": "255",
            "256": "256",
            "big": "1,000",
        },
        correct="256",
        why={
            "255": "A byte goes from 0 up to 255. Count them: does 0 count too? See the table of new words in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
            "256": "A byte is a whole number from 0 to 255: with 0 included, that is 256 values.",
            "big": "A byte goes up to 255 and no further: that is why UTF-8 needs more than one byte for `€`, 8364. How many whole numbers are there from 0 to 255? See the table of new words in [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en).",
        },
    )
    quick_byte
    return (quick_byte,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A camera turns light into numbers: a phone photo of 12 megapixels has 36 million of them, one byte each, yet its file takes only a few million bytes. The file format stores the numbers in fewer bytes. Some formats give every number back, but JPEG, the most common format for photos, gives back numbers that are close and often not the same. A model learns from the numbers in the file, not from what the camera saw, and many image datasets are JPEG.

    **An everyday example.** A recording of a lecture keeps every word. Your notes keep what matters in far less space, but you cannot get the exact words back from them.
    """)
    return


@app.cell(hide_code=True)
def predict_size():
    guess_size = L.guess(
        "size",
        question="The coffee photo of the course is 400 pixels high and 600 wide: 400 × 600 × 3 = 720,000 numbers, one byte each. Saved as a JPEG file with quality 75, about how big is the file?",
        options={
            "same": "About 720,000 bytes: one byte for every number",
            "half": "About 360,000 bytes",
            "small": "About 42,000 bytes",
        },
        reveal="**About 42,000 bytes**, 17 times less than the numbers themselves. As a PNG file, which gives every number back, it is about 449,000 bytes: 62% of the numbers. JPEG gets so much smaller because it does not keep the numbers themselves, only an approximation.",
    )
    guess_size
    return (guess_size,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Behind the lens of a camera is a sensor with millions of tiny light meters in a grid, each behind a red, a green or a blue filter. Out of how much light fell on them, the camera makes three numbers per pixel: whole numbers from 0 to 255, one byte each. An image file is not that array itself, but a way to store it. A PNG file compresses them without losing anything: it writes patterns, such as the same number many times in a row, in fewer bytes, and gives back every number exactly. A JPEG file leaves out detail that your eye hardly notices, such as fine changes in color, and keeps an approximation: most numbers come back a little different, and the file is often ten times smaller or more. With `quality`, from 0 to 95, you choose: the lower, the smaller the file, and the further the numbers are off. Quality 100 also works, but makes the file twice as big as 95, and still does not give every number back.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **image resolution** | how many pixels an image has: its width and its height, such as 600 by 400 | `img.shape` gives the height first |
    | **uint8** | the dtype of a whole number from 0 to 255, stored in one byte: the numbers of a photo | `img.dtype`, `np.uint8` |
    | **lossless compression** | storing data in fewer bytes, so that you get every number back exactly | PNG |
    | **lossy compression** | storing data in far fewer bytes, by keeping an approximation | JPEG |
    | **PNG** | an image format with lossless compression | `"coffee.png"` |
    | **JPEG** | an image format with lossy compression, for photos; `quality` says how close it stays | `"coffee.jpg"` |
    | **`Image.open(path)`** | opens an image file with Pillow; `np.array` turns it into an array | `np.array(Image.open(path))` |
    | **`Image.fromarray(img).save(path)`** | saves an array of uint8 as an image file; the end of the path chooses the format | `.save(path, quality=75)` |
    | **`os.path.getsize(path)`** | the size of a file, in bytes | `os.path.getsize(path)` |
    | **`.astype(int)`** | the same numbers with another dtype, here ordinary whole numbers | `img.astype(int)` |

    Pillow is the library Python uses for image files. You import it as `from PIL import Image`: PIL is the name of the library it grew out of. Pillow gives the numbers of an image as uint8, from 0 to 255, also for a PNG file, where `plt.imread` gave floats from 0 to 1. A JPEG file without `quality` gets 75.
    """)
    return


@app.cell(hide_code=True)
def picture_cup():
    mo.vstack([
        mo.md("**The rim of the cup, three ways**"),
        mo.Html(C.picture({"png": "PNG", "q75": "JPEG, quality 75", "q10": "JPEG, quality 10",
                           "size": "{n} bytes · off by {d}", "sep": ",", "dec": ".",
                           "alt": "The same corner of 32 by 32 pixels of the coffee photo, magnified, in three versions: the white rim of the cup against the red saucer. As PNG and as JPEG with quality 75 they look the same; with quality 10, the corner falls apart into blocks with smeared colors."})),
        mo.md(r"""
        The same 32 by 32 pixels of the coffee photo, magnified, after saving and reading back. Under each one: the size of the whole file, and how far a number of the photo is off on average. The PNG file gives every number back. With quality 75, the numbers are off by a little, and you see no difference. With quality 10, the file is smaller still, and you see how JPEG works: it cuts the image into blocks, of 8 by 8 pixels for the brightness and 16 by 16 for the color, and keeps each block as a rough approximation.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Numbers of the dtype uint8 stay between 0 and 255, also when you subtract them. If `a` and `b` are uint8, a number 10 in `a` and 12 in `b` gives 254 in `a - b`, not −2. Turn both into ordinary whole numbers first: `a.astype(int) - b.astype(int)`."),
        L.watch_out("Pillow says the size of an image the other way round: `Image.open(path).size` is `(width, height)`, `(600, 400)` for the coffee photo, while the shape of the array is `(400, 600, 3)`, with the rows first."),
        mo.md(r"""
        ### Recipe

        To see what a file format does to an image:

        1. Open the file and make an array: `img = np.array(Image.open(path))`; its dtype is uint8.
        2. Look at its resolution in `img.shape`: the rows (its height), the columns (its width) and the colors.
        3. Count the bytes of the numbers: rows × columns × colors, one byte each.
        4. Save it, as PNG or as JPEG with a `quality`: `Image.fromarray(img).save(path)`; `os.path.getsize(path)` gives the bytes of the file.
        5. Read it back and compare, as ordinary whole numbers: `diff = abs(back.astype(int) - img.astype(int))`. `diff.mean()` says how far a number is off on average: the mean absolute error, the MAE of [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en). `(diff > 0).mean()` says which share of the numbers changed.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The coffee photo: `data/photos/coffee.png`, a photo without copyright from the sample images of scikit-image. The comments with `#` say which step of the recipe each part is. The files go to your own work folder. Run the three cells.
    """)
    return


@app.cell
def example_read():
    import os
    import numpy as np
    from PIL import Image

    # 1. open the photo and make an array
    img = np.array(Image.open("data/photos/coffee.png"))
    # 2. its resolution and dtype
    print(img.shape, img.dtype)
    # 3. the bytes of its numbers, one byte each
    print(img.shape[0] * img.shape[1] * img.shape[2])
    return Image, img, np, os


@app.cell
def example_save(Image, img, os):
    # 4. save it as PNG and as JPEG, in your work folder, and look at the size of each file
    png = C.work_file("coffee.png")
    jpg = C.work_file("coffee.jpg")
    Image.fromarray(img).save(png)
    Image.fromarray(img).save(jpg, quality=75)
    print(os.path.getsize(png), os.path.getsize(jpg))
    return jpg, png


@app.cell
def example_compare(Image, img, jpg, np, png):
    # 5. read both back, and compare them with the numbers you started with
    for _path in (png, jpg):
        _back = np.array(Image.open(_path))
        _diff = abs(_back.astype(int) - img.astype(int))
        print(_diff.mean(), (_diff > 0).mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The photo has shape `(400, 600, 3)` and dtype `uint8`: 400 rows, 600 columns and 3 colors, 720,000 numbers of one byte each. The PNG file takes about 449,000 bytes, 62% of that; the JPEG file about 42,000, 6%. The PNG file gives every number back: no difference at all. The JPEG file is off by 4.02 on average, on a scale from 0 to 255, and 0.877, or 88%, of the numbers changed. The picture above shows that you hardly see it. The course's own file `data/photos/coffee.png` is bigger, 466,706 bytes: the same numbers, stored with other settings. A PNG file always gives the same numbers back, but how small it gets depends on how it was written.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bytes():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The bytes of a photo
    """)
    return


@app.cell(hide_code=True)
def choice_bytes_cell():
    choice_bytes = L.choice(
        "raw-bytes",
        question="A color photo is 4000 pixels wide and 3000 high, with numbers of dtype uint8. How many bytes do its numbers take, before any compression?",
        options={
            "pixels": "12,000,000",
            "numbers": "36,000,000",
            "values": "9,180,000,000",
        },
        correct="numbers",
        why={
            "pixels": "That is the number of pixels. How many numbers does a color pixel have? See step 3 of the recipe.",
            "numbers": "4000 × 3000 pixels × 3 colors is 36 million numbers, and uint8 takes one byte per number: 36 million bytes.",
            "values": "That multiplies by 255, the biggest value. But a number takes one byte whatever its value, from 0 to 255: how many numbers are there?",
        },
    )
    choice_bytes
    return (choice_bytes,)


@app.cell(hide_code=True)
def text_exercise_road():
    mo.md(r"""
    ### Exercise 2. Exact numbers
    """)
    return


@app.cell(hide_code=True)
def choice_road_cell():
    choice_road = L.choice(
        "road-format",
        question="A model learns to find roads in aerial photos. Next to every photo is an image of the same size, with 1 for every pixel on a road and 0 for every other pixel. In which format do you save those images?",
        options={
            "png": "PNG",
            "jpeg95": "JPEG with quality 95: then the approximation is very close",
            "jpeg10": "JPEG with quality 10: then the files are smallest",
        },
        correct="png",
        why={
            "png": "Every number here means something exact: road or not. Only lossless compression gives each one back. A road image of the course, 60 rows of 80 pixels, takes a few hundred bytes as PNG, for its 4,800 numbers, and comes back exactly the same.",
            "jpeg95": "Close is not the same. The course saved a road image of 60 rows of 80 pixels as JPEG with quality 95, and 53 of its 4,800 pixels came back with the wrong number: road became not-road, or the other way round. What kind of compression gives every number back?",
            "jpeg10": "The course saved a road image of 60 rows of 80 pixels as JPEG with quality 10: every pixel came back as 0, so the road was gone. What kind of compression gives every number back?",
        },
    )
    choice_road
    return (choice_road,)


@app.cell(hide_code=True)
def text_exercise_difference():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Measure the approximation

        Finish `mean_difference(a, b)`. It gets two images of the same shape, as uint8, and gives back how far a number of `b` is from the same number of `a` on average: their MAE. Replace the `...`. The check calls it with a grayscale image of 2 by 2 pixels, `a`, and `b`, with numbers as a JPEG file might give them back:

        ```python
        a = [[10, 20], [30, 40]]
        b = [[12, 20], [27, 40]]
        ```

        That should give `1.25`: the differences are 2, 0, 3 and 0. It also tries other images.
        """),
        L.task("Make `mean_difference(a, b)` give how far a number of `b` is from the same number of `a`, on average."),
    ])
    return


@app.cell(hide_code=True)
def exercise_difference():
    box_difference = L.code_box("mean-difference", lines=8)
    box_difference
    return (box_difference,)


@app.cell(hide_code=True)
def check_difference(box_difference):
    result_difference = L.check("mean-difference", box_difference.value)
    result_difference
    return (result_difference,)


@app.cell(hide_code=True)
def hints_difference():
    L.hints(
        "mean-difference",
        [
            "Step 5 of the recipe: turn both into ordinary whole numbers, subtract, take the size of every difference, and the mean.",
            "Read the first watch-out above the recipe: what does `a - b` give when both are uint8 and a number of `b` is bigger?",
            "`diff = abs(a.astype(int) - ___)`, and then `return diff.___()`.",
        ],
        why="`a.astype(int) - b.astype(int)` gives −2, 0, 3 and 0, and `abs` makes each one its size: 2, 0, 3 and 0. Their mean is 5 / 4 = 1.25. Without `.astype(int)`, 10 − 12 would wrap around to 254.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_quality():
    mo.md(r"""
    ### Exercise 4. As small as it can be

    The cat photo, `data/photos/chelsea.png`, 451 by 300 pixels, as JPEG with the quality of the slider. On the left a part of it as it is, around the eye; on the right the same part after saving as JPEG and reading back. The sentence under it is about the whole photo.
    """)
    return


@app.cell(hide_code=True)
def explore_quality_cell():
    def _frame(q):
        _html, _f = C.quality_frame(q, {"png": "PNG", "jpeg": "JPEG, quality {q}",
                                        "alt": "The eye of the cat, magnified, as it is and after JPEG with quality {q}."})
        return (_html,
                f'Quality {q}: the file is {C.grouped(_f["size"], ",")} bytes, {L.number(100 * _f["share"], 0)}% of the PNG file. A number is off by <span class="c-loss ea-num">{L.number(_f["diff"])}</span> on average, and {L.number(100 * _f["changed"], 0)}% of the numbers changed.')

    explore_quality = L.explore("quality", "quality", _frame, start=95, values=C.QUALITIES, role="",
                                task="Find the lowest quality at which a number of the cat is off by less than 3 on average.")
    explore_quality
    return (explore_quality,)


@app.cell(hide_code=True)
def check_quality(explore_quality):
    result_quality = L.check("quality", explore_quality)
    result_quality
    return (result_quality,)


@app.cell(hide_code=True)
def hints_quality():
    L.hints(
        "quality",
        [
            "Move the slider down one stop at a time, and read how far a number is off on average.",
            "3 or more is too far. What is the lowest stop at which a number is still off by less than 3?",
            "The lower the quality, the further off: once a stop gives 3 or more, every stop below it does too. So go down until you pass 3, and take the stop just before.",
        ],
        why="At quality 75, a number is off by 2.85 on average; at 50, by 3.65. The file with quality 75 is about 21,000 bytes, a twentieth of the 405,900 numbers.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) you read an image as an array; now you know what the file in between does to its numbers. In [Text is a sequence of numbers](/course/open/data-text-as-numbers?lang=en), a byte held a whole number from 0 to 255; here, every number of a photo is one byte. And the MAE of [How far off? MAE, RMSE and R²](/course/open/ml-regression-metrics?lang=en), there the error of a model, here measures how far a file is from the photo. Later in this route, a pretrained model gets photos that were saved as JPEG, and an image where every number means something exact, such as road or not road, is always saved as PNG. In the next lesson, *Another way to write a color: HSV*, you write the three numbers of a pixel in another way.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bytes, choice_road, result_difference, result_quality):
    L.wrap_up(choice_bytes, choice_road, result_difference, result_quality)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
