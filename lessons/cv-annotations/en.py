# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-annotations. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-annotations", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Count the shapes: connected components](/course/open/cv-connected-components?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_box():
    quick_box = L.choice(
        "quick-box",
        question="A shape has these pixels, as (row, column): `(1, 3)`, `(2, 3)`, `(2, 4)` and `(3, 2)`. What is its bounding box, written as in that lesson, `(top, left, bottom, right)`?",
        options={
            "box": "`(1, 2, 3, 4)`",
            "xy": "`(2, 1, 4, 3)`",
            "ends": "`(1, 3, 3, 2)`",
            "size": "`(1, 2, 3, 3)`",
        },
        correct="box",
        why={
            "box": "The rows run from 1 to 3 and the columns from 2 to 4: top 1, left 2, bottom 3, right 4. This lesson writes the same box in another way too.",
            "xy": "That puts the columns first: left, top, right, bottom. The course writes the rows first, as NumPy does.",
            "ends": "Those are the first and the last pixel of the list. The box runs from the smallest to the largest row, and from the smallest to the largest column: which is the smallest column?",
            "size": "The last two numbers are the height and the width of the box, 3 and 3. The course writes the bottom and the right side: the largest row and the largest column.",
        },
    )
    quick_box
    return (quick_box,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="A shape of 12 pixels has the bounding box `(0, 0, 3, 3)`. How many pixels inside its box are not part of the shape?",
        options={
            "four": "4",
            "zero": "0",
            "none": "None: 12 pixels do not fit in that box",
        },
        correct="four",
        why={
            "four": "The box runs from row 0 to row 3 and from column 0 to column 3, both included: 4 rows of 4 columns, 16 pixels. 12 of them are the shape, so 4 are not: the shape fills 12 of 16, 75% of its box. How much of its box a shape fills comes back in this lesson.",
            "zero": "Then the shape would fill its box exactly, as an upright rectangle does. How many pixels does a box from row 0 to row 3 and from column 0 to column 3 hold?",
            "none": "From row 0 to row 3 are 4 rows, not 3: the top and the bottom row are both part of the box. And the same for the columns.",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model learns from examples with the right answer, and what that answer looks like depends on what the model must do: one word per photo to sort photos, a rectangle around each car to find the cars in a street, every pixel of an object to cut it out of a photo, the exact place of every fingertip to follow a hand. The more exact the answer, the longer a person works on it. So before anyone labels a single image, someone chooses the form of the answers, and that choice decides what a model can learn from them and what the data costs.

    **An everyday example.** The photos on your phone. A search for "dog" only needs to know that a dog is in a photo. Cropping a photo to the dog needs to know where it is. Lifting the dog out of the photo as a sticker needs every pixel of it. And a filter that puts sunglasses on it needs to know exactly where its eyes are. One photo, four questions, and each needs a different answer.
    """)
    return


@app.cell(hide_code=True)
def predict_cost():
    guess_cost = L.guess(
        "cost",
        question="A box of good quality takes 7 to 35 seconds to draw, depending on how you draw it. How long did the mask of one object take in COCO, a well-known dataset of photos?",
        options={
            "same": "About as long: 10 to 35 seconds",
            "minute": "More than a minute",
            "ten": "About ten minutes",
        },
        reveal="**More than a minute.** COCO's annotators spent over 22 hours on every 1000 masks: more than 79 seconds per object, two to eleven times as long as a box. A box needs only the four outermost sides of an object; a mask follows every bend of its outline. The more exact an annotation, the more it costs.",
    )
    guess_cost
    return (guess_cost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        An **annotation** is the right answer that people add to an example by hand, such as a label or a box; the people who do it are annotators. For an image, the task decides the form of its annotation. There are four, from coarse to exact:

        - **A label per image.** For classification: the whole image gets one class, such as `triangle`. A label says what the image shows, not where, so it fits an image with one thing in it.
        - **A box per object.** For **object detection**: finding every object of known classes in an image, each with a bounding box and its class. Four numbers per object are quick to draw, but a box holds background too, and often part of another object.
        - **A mask per object.** A **segmentation mask** is a binary image as large as the image: `True` for every pixel of the object, `False` elsewhere. A connected component of [Count the shapes: connected components](/course/open/cv-connected-components?lang=en) is a mask too. The box follows from a mask, but a mask does not follow from a box. A mask is the most exact, and takes the longest to draw.
        - **Keypoints per object.** A **keypoint** is one exact place on an object, such as a corner of a square, the tip of a nose or a left elbow. Every object of a class has the same keypoints, always in the same order.

        Objects hide each other. In the course's scenes, a shape drawn later lies on top of the earlier ones. A mask then holds only the pixels you can see, and the box is the smallest box around the mask. A corner under another shape is still a keypoint. Every keypoint gets a flag v, as COCO writes it: 2 when it can be seen, 1 when it is there but hidden, and 0 when it is not labeled, and then its place is `0, 0` too. In the course's scenes, a keypoint is the pixel of the shape nearest to a corner, and it is hidden when that pixel is not in the mask.

        A dataset writes its annotations in a fixed format. The course writes a box as `(top, left, bottom, right)`, both ends included. COCO writes `[x, y, width, height]`: x is the column of the left side and y the row of the top, both counted from 0 at the top left, and width and height are the number of columns and rows the box covers. It writes the keypoints of an object as one list of three numbers per keypoint: `[x1, y1, v1, x2, y2, v2, ...]`.
        """),
        L.watch_out("A box from a dataset or a library is four numbers, and nothing in them says which four. Before you use one, look up its format: does the column come first or the row, and are the last two numbers a corner or a width and a height? A box in the wrong format gives no error: it just lands in the wrong place."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **annotation** | the right answer that people add to an example by hand: a label, a box, a mask or keypoints | `shape.kind`, `shape.box`, `shape.mask` and `shape.keypoints` in the worked example |
        | **object detection** | finding every object of known classes in an image, each with a bounding box and its class | a box and a class per object; COCO `[x, y, width, height]` |
        | **segmentation mask** | a binary image as large as the image: `True` for every pixel of one object | `shape.mask`, an array of `bool` |
        | **keypoint** | one exact place on an object, with a flag v: 2 seen, 1 hidden, 0 not labeled | `(row, column, v)`; COCO `[x, y, v]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_annotations():
    mo.vstack([
        mo.md("**Four kinds of annotation**"),
        mo.Html(C.annotations_picture({
            "label": "a label per image: triangle",
            "boxes": "a box per object",
            "mask": "a mask per object: the square",
            "keypoints": "keypoints: the corners",
            "alt": "Four pictures. Top left, an image with one purple triangle on gray, with the label triangle. Top right, a scene with a blue triangle, a blue square, a red circle on top of the square and a small yellow triangle, each in a rectangle as tight as it fits: the boxes of the circle and the square overlap. Bottom left, the mask of the square: white where the square can be seen, with a round bite where the circle lies on it; the other shapes dark gray, on black. Bottom right, the same scene with a dot at every corner of the square and the triangles, and an open circle at the corner of the square that lies under the circle.",
        })),
        mo.md(r"""
        Top left, an image with one shape, and its label. Then the scene of the worked example: a box around every shape; the mask of the square in white, without the pixels the circle hides, and the other shapes in dark gray, each with a mask of its own; and the corners of the square and the triangles as keypoints, a dot where a corner can be seen and an open circle where it is hidden. A circle has no corners, so no keypoints.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    A box is quick to draw, but it holds background, and how much depends on the shape and on how it lies. In 2000 of the course's scenes, a triangle that nothing hides fills 50% of its box on average, and a circle 79%: a triangle with one side along its box fills exactly half of it. The slider turns a square in steps of 5 degrees. On the left, the square in its box; on the right, how much of its box the square fills at every angle.
    """)
    return


@app.cell(hide_code=True)
def explore_turn_cell():
    def _frame(a):
        a = int(a)
        _html, _o = C.turn_frame(a, {"title": "turned by {angle}°", "fill": "how much of its box the square fills", "angle": "angle in degrees",
                                     "alt": "A blue square turned by {angle} degrees, in its box: it fills {fill}% of the box. Next to it, how much of its box the square fills at every angle from 0 to 90 degrees: 100% upright, going down to 50% at 45 degrees, and back up to 100% at 90 degrees. {angle} degrees is marked by a larger point."})
        return _html, f"Turned by {a}°: the square fills {_o['fill']}% of its box."

    explore_turn = L.explore("turn", "angle", _frame, start=0, values=C.ANGLES, role="",
                             task="Find the angle at which the square fills the least of its box.")
    explore_turn
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Upright, at 0° and at 90°, the square fills its whole box. Turned by 45°, it stands on a corner and fills only 50% of it: the four corners of the box are background. In between, it fills from 52% to 85%. And background is not all a box takes in: in the same 2000 scenes, 54% of the boxes also hold pixels of another shape. Shapes hide corners too: 6.6% of the corners lie under a shape drawn later, keypoints with v = 1.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Keep a hidden keypoint in the list, with v = 1, and write a keypoint that is not labeled as `0, 0` with v = 0. Leaving one out moves every keypoint after it up a place: a program then reads the fourth corner as the third."),
        mo.md(r"""
        ### Recipe

        To annotate images for a task:

        1. Decide what the model must give back: one class per image, where every object is, which pixels belong to it, or exact places on it.
        2. Choose the form that gives just that: a label, a box, a mask or keypoints. A more exact form tells the model more, and takes longer per object.
        3. Fix the format, and write it down: the order of the numbers, where 0 is, whether a box ends in a corner or in a width and a height, and the order of the keypoints.
        4. Derive what follows: the box of a mask runs from the smallest to the largest row and column of its pixels. The other way round does not work.
        5. Mark what is hidden instead of leaving it out: a keypoint under another object gets v = 1.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell makes the scene of the picture with `scene(18)`, from the course's own module `everything_ai.scenes`: the same seed gives the same scene on every computer. Every shape in `s.shapes` carries its annotation: `kind` is its class, `box` its box as the course writes it, `mask` its segmentation mask, an array of `bool` of 128 by 128, and `keypoints` its corners as `(row, column, v)`. `area` is the number of pixels of the whole shape, the hidden ones included. The cell prints them per shape, with the number of pixels in the mask. The second cell follows step 4 of the recipe: `box_of_mask` derives the box from the mask, and `fill` measures how much of its box a shape fills. It prints whether the derived box is the one in the annotation, and the fill.
    """)
    return


@app.cell
def example_scene():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    for shape in s.shapes:
        print(shape.kind, shape.box, shape.mask.sum(), shape.area, shape.keypoints)
    return np, s


@app.cell
def example_fill(np, s):
    def box_of_mask(mask):
        # 4. the rows and the columns with at least one pixel of the mask
        rows = np.arange(len(mask))[mask.sum(axis=1) > 0]
        cols = np.arange(len(mask[0]))[mask.sum(axis=0) > 0]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


    def fill(shape):
        # how much of its box the pixels of its mask fill
        top, left, bottom, right = shape.box
        return shape.mask.sum() / ((bottom - top + 1) * (right - left + 1))


    # _shape, because the cell above already has a name shape
    for _shape in s.shapes:
        same = box_of_mask(_shape.mask) == _shape.box
        # round(x, 2): x rounded to two decimals
        print(_shape.kind, same, round(fill(_shape), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The scene has two triangles, a square and a circle, drawn in that order. Every box follows from its mask: `True` four times. The square has 959 pixels, but the circle, drawn after it, hides 306 of them: its mask holds the 653 you can see, so it fills only 60% of its box. Its third corner, at row 75 and column 68, lies under the circle: v = 1. That corner even lies outside the box, which ends at row 73: the box holds only what can be seen. The triangles fill 54% and 48% of their boxes, the circle 83%. The circle has no keypoints: its list is empty.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which annotation?
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="A model must say for every leaf in a photo how much of it is covered by brown spots, as a percentage of the leaf. Which annotation do its training photos need?",
        options={
            "label": "A label per photo: healthy or sick",
            "box": "A box around every leaf and every spot",
            "mask": "A mask of every leaf and every spot",
            "keypoints": "A keypoint in the middle of every spot",
        },
        correct="mask",
        why={
            "mask": "A percentage of a leaf is a number of pixels divided by a number of pixels: the pixels of the spots on the leaf, divided by the pixels of the leaf. Only masks say which pixels those are.",
            "label": "A label says whether a leaf is sick, not how much of it is. The model must give a percentage: which pixels does it need to count?",
            "box": "A box holds background too: a triangle fills only half of its box, and a round spot about 79%. Counting the pixels of boxes gives the wrong percentage. Which annotation says which pixels are the spot?",
            "keypoints": "A point says where a spot is, not how large it is. How would you measure its area from one point?",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_coco():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. From the course's box to COCO

        Complete `to_coco(box)`, which takes a box as the course writes it, `(top, left, bottom, right)` with both ends included, and gives it back as COCO writes it: `[x, y, width, height]`. The check calls it with the box of the square from the worked example:

        ```python
        to_coco((42, 39, 73, 72))
        ```

        That should give `[39, 42, 34, 32]`. The check also tries other boxes, among them a box of one pixel.
        """),
        L.task("Make `to_coco(box)` give the box as `[x, y, width, height]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_coco():
    coco_code = L.code_box("coco", lines=9)
    coco_code
    return (coco_code,)


@app.cell(hide_code=True)
def check_coco(coco_code):
    result_coco = L.check("coco", coco_code.value)
    result_coco
    return (result_coco,)


@app.cell(hide_code=True)
def hints_coco():
    L.hints(
        "coco",
        [
            "x and y are a column and a row: which side of the box is a column, and which a row? And how many columns does a box from column 3 to column 5 cover?",
            "`fill` in the worked example counts the columns and the rows of a box: look at what it divides by.",
            "`x = left`, `y = top`, `width = right - left + ___` and `height = ___`.",
        ],
        why="The square's box runs from column 39 to column 72: that is 72 - 39 + 1 = 34 columns, its width. And from row 42 to row 73: 32 rows, its height. COCO puts the column first, as x on a graph: `[39, 42, 34, 32]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Exercise 3. A foot behind a table
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden",
        question="A chair gets a keypoint at the foot of each of its four legs, always in the same order. In one photo, one leg stands behind a table, but you can tell where its foot must be. How do you write that keypoint, as COCO does?",
        options={
            "one": "`[x, y, 1]`, at the place where the foot must be",
            "two": "`[x, y, 2]`, at the place where the foot must be",
            "zero": "`[0, 0, 0]`",
            "out": "Leave it out: the chair then has three keypoints",
        },
        correct="one",
        why={
            "one": "The foot is there, but you cannot see it: v = 1, at the place where it must be, as the third corner of the square in the worked example. The annotation keeps where the foot is, and says that it is hidden.",
            "two": "2 says that the keypoint can be seen, and this foot stands behind the table. Which flag is for a keypoint that is there but hidden?",
            "zero": "0 is for a keypoint that is not labeled, for instance because no one can tell where it is. Here you can tell where the foot must be: which flag keeps that place?",
            "out": "Then every keypoint after it moves up a place, and the fourth leg is read as the third. Every chair has the same four keypoints in the same order: which flag says that one is hidden?",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_exercise_cost():
    mo.md(r"""
    ### Exercise 4. Why not masks for everything?
    """)
    return


@app.cell(hide_code=True)
def explain_cost_cell():
    explain_cost = L.explain(
        "cost",
        question="A mask is the most exact annotation: the box follows from it, and it says which pixels the object has. Why would someone who builds a dataset for object detection draw boxes, and not masks?",
        model_answer="**The task needs only the box.** A model for object detection gives back a box and a class per object, so a box is all its examples have to show; a mask adds pixels it never gives back. **A mask takes much longer.** Drawing a box of good quality takes 7 to 35 seconds, and a mask in COCO took more than 79. **The same hours then buy more examples.** In the time of one mask, annotators draw two to eleven boxes, so the dataset gets more images and more objects for the same work.",
        points=[
            "you say that a model for object detection gives back boxes, so a box is all the task asks for",
            "you say that a mask takes much longer to draw than a box, such as more than 79 seconds against 7 to 35",
            "you say that the same time then labels more objects or more images",
        ],
    )
    explain_cost
    return (explain_cost,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The box and the connected components of [Count the shapes: connected components](/course/open/cv-connected-components?lang=en) are two of the four kinds of annotation: a connected component is a mask, and its box follows from it. Which kind a dataset holds depends on what its model must find: a label for what an image shows, a box for where each object is, a mask for which pixels it has, and keypoints for exact places on it. The more exact, the longer it takes: a triangle fills only half of its box, but a box is drawn in a fraction of the time of a mask. Formats differ, so you convert, as from the course's box to COCO's `[x, y, width, height]`, and a hidden keypoint keeps its place with v = 1. In the next lesson, *How much two boxes overlap: IoU*, you measure how well a box that a model predicts matches the box of the annotation.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hidden, choice_which, explain_cost, result_coco):
    L.wrap_up(choice_which, result_coco, choice_hidden, explain_cost)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
