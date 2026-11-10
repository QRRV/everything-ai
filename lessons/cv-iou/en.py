# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-iou. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-iou", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_coco():
    quick_coco = L.choice(
        "quick-coco",
        question="COCO writes a box as `[x, y, width, height]`. Which box is `[10, 20, 5, 3]`, written as the course does, `(top, left, bottom, right)` with both ends included?",
        options={
            "box": "`(20, 10, 22, 14)`",
            "plus": "`(20, 10, 23, 15)`",
            "xy": "`(10, 20, 14, 22)`",
            "swap": "`(20, 10, 24, 12)`",
        },
        correct="box",
        why={
            "box": "x = 10 is the left side and y = 20 the top. 3 rows from row 20 are rows 20, 21 and 22, so the bottom is 20 + 3 - 1 = 22; 5 columns from column 10 end at column 14. This lesson works with boxes as the course writes them.",
            "plus": "That box is one row and one column too large: from row 20 to row 23 are 4 rows. Both ends are part of the box, so 3 rows from row 20 end at row 22.",
            "xy": "That puts the column first. COCO's x is a column, the left side, and the course writes the top row first.",
            "swap": "That box is 5 rows tall and 3 columns wide. The width, 5, counts the columns, and the height, 3, the rows.",
        },
    )
    quick_coco
    return (quick_coco,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="How many pixels does the box `(2, 3, 4, 7)` hold, both ends included?",
        options={
            "fifteen": "15",
            "eight": "8",
            "ten": "10",
            "many": "28",
        },
        correct="fifteen",
        why={
            "fifteen": "Rows 2 to 4 are 3 rows, and columns 3 to 7 are 5 columns: 3 × 5 = 15 pixels. In this lesson you count the pixels of boxes, and of the part where two boxes overlap.",
            "eight": "4 - 2 = 2 and 7 - 3 = 4 leave out one end. From row 2 to row 4 are rows 2, 3 and 4: three rows.",
            "ten": "That counts the columns with both ends and the rows without one. From row 2 to row 4 are three rows, not two.",
            "many": "4 × 7 multiplies the bottom and the right side, the corner of the box. The box starts at row 2 and column 3, not at 0: how many rows and columns does it cover?",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model for object detection gives back a box for every object it finds. To train it, and to know how good it is, you compare each box it predicts with the box of the annotation, and that comparison has to be one number. Pixels off say too little: a box 3 pixels too large on every side is almost perfect around a bus 100 pixels wide, and far off around a bird of 10 pixels. What counts is how much of the two boxes is shared.

    **An everyday example.** You and a friend each draw on a map where you think the festival grounds are. How well do you agree? Not how far apart the middles of your drawings are: one of you may have drawn a much larger patch around the same middle. What matters is the land you both drew, as a share of all the land at least one of you drew.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    guess_shift = L.guess(
        "shift",
        question="The **intersection over union** (IoU) of two boxes is their overlap divided by the area they cover together. A predicted box is as large as the true box, but lies a quarter of its width to the right. What is its IoU?",
        options={
            "most": "0.75",
            "three-fifths": "0.6",
            "half": "0.5",
            "quarter": "0.25",
        },
        reveal="**0.6.** The overlap is three quarters of a box. Together the two boxes cover a whole box plus the quarter that sticks out: five quarters. Three quarters divided by five quarters is 3/5 = 0.6. Each box shares three quarters with the other, and still the IoU is lower: what the predicted box misses and what it adds both count against it.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The **intersection over union**, or IoU, says how well two boxes match: the number of pixels in their overlap, divided by the number of pixels they cover together. The intersection is the overlap, and the union is what the two boxes cover together:
        """),
        L.formula(r"\text{IoU} = \frac{\text{overlap}}{\text{union}} = \frac{\text{overlap}}{\text{area}_a + \text{area}_b - \text{overlap}}"),
        mo.md(r"""
        It is a share: 1 when the two boxes are the same, 0 when they do not overlap, and in between for everything else. Because it is a share, it weighs how far off a box is against how large the boxes are: the same two boxes drawn twice as large, with every error twice as large too, get the same IoU. A box 3 pixels too large on every side gets 0.39 around a box of 10 by 10 pixels, and 0.89 around a box of 100 by 100: the larger object barely notices the 3 pixels, and the IoU says so.

        For two boxes you need no pixels, only their four numbers. The overlap of two boxes is a box too. It starts where both boxes have begun: at the larger of the two tops and the larger of the two lefts. It ends where the first one ends: at the smaller of the two bottoms and the smaller of the two rights. When the boxes do not overlap, that box is inside out: its bottom lies above its top, or its right side left of its left side, so its count of rows or columns comes out at 0 or below: count it as 0, and the overlap is 0. The union is both areas added, minus the overlap, because the overlap lies in both.

        For masks, the same works pixel by pixel: `(m1 & m2).sum()` counts the overlap and `(m1 | m2).sum()` the union. That works for any shape, also for a box made into a mask.

        How high must an IoU be? PASCAL VOC, a well-known contest for object detection, counts a predicted box as found when its IoU with the true box is above 0.5.
        """),
        L.watch_out("The union is not the two areas added. The overlap lies in both boxes, so adding them counts it twice: subtract it once. Without that, two boxes that are the same get 0.5 instead of 1."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **intersection over union** (IoU) | how well two boxes or masks match: the pixels of their overlap divided by the pixels they cover together, from 0 to 1 | `overlap / union` |
        | **overlap** | the pixels that lie in both boxes or masks; for two boxes, a box from the larger tops and lefts to the smaller bottoms and rights | `(a & b).sum()` |
        | **union** | the pixels that lie in at least one of the two: both areas, minus the overlap | `(a | b).sum()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_overlap():
    mo.vstack([
        mo.md("**The overlap and the union of two boxes**"),
        mo.Html(C.overlap_picture({
            "boxes": "true box and predicted box",
            "overlap": "overlap: {overlap} pixels",
            "union": "union: {union} pixels",
            "alt": "Three pictures of the same part of a scene. Left, a red circle that lies on a blue square, in a tight box drawn as a solid line, and a dashed box that lies a little lower and further to the left, and is a little larger. In the middle, on black, the part where the two boxes overlap in white, and the rest of both boxes in dark gray. Right, everything the two boxes cover in white. The middle and the right picture show the two box lines too.",
        })),
        mo.md(r"""
        Left, the circle of the worked example with its true box, solid, and a box that a model could predict for it, dashed. In the middle, their overlap in white: 700 pixels. Right, their union in white: 1101 pixels. The IoU is 700 / 1101 = 0.64.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How far off may a box be? The slider moves a predicted box, as large as the circle's true box, to the right, one pixel at a time. On the left, the two boxes; on the right, the IoU at every shift, with a dotted line at 0.5.
    """)
    return


@app.cell(hide_code=True)
def explore_shift_cell():
    def _frame(d):
        d = int(d)
        _html, _o = C.shift_frame(d, {"title": "shifted by {shift} pixels", "iou": "IoU with the true box", "shift": "shift in pixels", "point": ".",
                                      "alt": "The true box of a red circle, solid, and a dashed box as large, shifted {shift} pixels to the right: IoU {iou}. Next to it, the IoU at every shift from 0 to 29 pixels: 1 at 0, falling fastest at the start, through the dotted line at 0.5 between 9 and 10 pixels, to 0 at 29. The shift of {shift} pixels is marked by a larger point."})
        return _html, f"Shifted by {d} pixels: overlap {_o['overlap']} pixels, union {_o['union']} pixels, IoU {L.number(_o['iou'])}."

    explore_shift = L.explore("shift-slider", "shift in pixels", _frame, start=0, values=C.SHIFTS, role="",
                              task="Find the largest shift at which the IoU is still above 0.5.")
    explore_shift
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Shifted by 9 pixels, the IoU is 0.53; at 10 pixels it is 0.49, below PASCAL VOC's 0.5. The box is 29 pixels wide, so 10 pixels, just over a third of its width, is already too much: 19 of its 29 columns are still shared, but together the two boxes span 39 columns. Shifted by its whole width, 29 pixels, the two boxes only touch, and the IoU is 0. The curve falls fastest at the start: the first pixels off cost the most.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Make each count at least 0 on its own, before you multiply. For two boxes that lie apart both down and across, the rows and the columns of the overlap both come out below 0, and two negative numbers multiplied give a positive overlap."),
        mo.md(r"""
        ### Recipe

        To measure how well a predicted box matches the true box:

        1. Find the overlap: the larger of the two tops and of the two lefts, the smaller of the two bottoms and of the two rights.
        2. Count its pixels: its rows, bottom - top + 1, times its columns, right - left + 1, each made 0 when it comes out below 0.
        3. Count the union: the pixels of both boxes added, minus the overlap.
        4. Divide the overlap by the union: 1 for a perfect box, 0 for a miss.
        5. To check, or for masks, count pixels: `(a & b).sum()` is the overlap and `(a | b).sum()` the union.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe both ways, on the scene of [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), made again with `scene(18)`. The first cell takes the box of the circle, `s.shapes[3].box`, and a box that a model could predict for it, and follows step 5: it counts pixels. `box_mask` makes a box into a mask as large as the scene: `np.zeros` with `dtype=bool` makes an array of `False`, and setting the slice of the box to `True` sets every pixel in it at once, with + 1 because a slice stops just before its end. Then `&` and `|` give the overlap and the union, and `sum` counts their pixels. The second cell compares the predicted box with the box of every shape in the scene in the same way. The third cell follows steps 1 to 4, with only the four numbers of each box, for the square and the large triangle.
    """)
    return


@app.cell
def example_masks():
    import numpy as np
    from everything_ai.scenes import scene

    s = scene(18)
    truth = s.shapes[3].box     # the box of the circle
    pred = (62, 46, 91, 77)     # a box that a model could predict for it


    def box_mask(box):
        # a mask as large as the scene: True in the rows and columns of the box
        top, left, bottom, right = box
        mask = np.zeros((128, 128), dtype=bool)
        mask[top:bottom + 1, left:right + 1] = True
        return mask


    a = box_mask(truth)
    b = box_mask(pred)
    # 5. count pixels: the overlap and the union
    overlap = (a & b).sum()
    union = (a | b).sum()
    print(truth, a.sum(), pred, b.sum())
    print(overlap, union, overlap / union)
    return b, box_mask, pred, s


@app.cell
def example_others(b, box_mask, s):
    for shape in s.shapes:
        other = box_mask(shape.box)
        # round(x, 2): x rounded to two decimals
        print(shape.kind, shape.box, round((other & b).sum() / (other | b).sum(), 2))
    return


@app.cell
def example_steps(pred, s):
    # steps 1 to 4 with only the four numbers of each box: the predicted box
    # against the box of the square and the box of the large triangle
    for t in (s.shapes[2].box, s.shapes[1].box):
        # 1. the overlap: the larger tops and lefts, the smaller bottoms and rights
        top, left = max(t[0], pred[0]), max(t[1], pred[1])
        bottom, right = min(t[2], pred[2]), min(t[3], pred[3])
        # 2. its rows and its columns, each at least 0, and its pixels
        rows = max(0, bottom - top + 1)
        cols = max(0, right - left + 1)
        both = rows * cols
        # 3. the union: both areas added, minus the overlap
        area_t = (t[2] - t[0] + 1) * (t[3] - t[1] + 1)
        area_pred = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        union_t = area_t + area_pred - both
        # 4. the overlap divided by the union
        print(t, bottom - top + 1, right - left + 1)
        print(both, union_t, round(both / union_t, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The circle's box holds 841 pixels and the predicted box 960. They share 700, and together they cover 1101: 841 + 960 = 1801, minus the 700 that are in both. The IoU is 700 / 1101 = 0.64, above 0.5: by the rule of PASCAL VOC, the predicted box has found the circle. It also overlaps the box of the square, with an IoU of 0.19, and the triangles not at all; its IoU is highest for the circle, so of all the shapes it fits the circle best. The third cell gets the 0.19 of the square without a mask: an overlap of 12 rows and 27 columns, 324 pixels, and a union of 1724. For the large triangle, the columns come out at -16: the triangle ends at column 29, and the predicted box starts at column 46. `max(0, ...)` makes that 0, so the overlap is 0. Exercise 2 puts these steps in a function that works for any two boxes, and the check tries it on the circle: 0.64.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_overlap():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The overlap of two boxes
    """)
    return


@app.cell(hide_code=True)
def choice_overlap_cell():
    choice_overlap = L.choice(
        "overlap",
        question="What is the overlap of the boxes `(10, 20, 30, 40)` and `(25, 5, 50, 35)`, written as `(top, left, bottom, right)`?",
        options={
            "right": "`(25, 20, 30, 35)`",
            "around": "`(10, 5, 50, 40)`",
            "largest": "`(25, 20, 50, 40)`",
            "smallest": "`(10, 5, 30, 35)`",
        },
        correct="right",
        why={
            "right": "The overlap starts where both boxes have begun: at row 25, the larger top, and at column 20, the larger left. It ends where the first box ends: at row 30, the smaller bottom, and at column 35, the smaller right. It holds 6 rows of 16 columns: 96 pixels.",
            "around": "That is the smallest box around both boxes: it holds every pixel of each, and more. The overlap holds only the pixels that lie in both: does row 10 lie in the second box?",
            "largest": "That takes the larger number of each pair, also for the bottom and the right. Row 50 lies in the second box only, because the first one ends at row 30. Where does the overlap end?",
            "smallest": "That takes the smaller number of each pair, also for the top and the left. Row 10 lies in the first box only, because the second one starts at row 25. Where does the overlap begin?",
        },
    )
    choice_overlap
    return (choice_overlap,)


@app.cell(hide_code=True)
def text_exercise_iou():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. IoU from four numbers

        Complete `iou(a, b)`, which takes two boxes as the course writes them, `(top, left, bottom, right)` with both ends included, and gives their IoU without making a mask. `area` is given: it counts the pixels of a box. The check calls it with the circle and the predicted box from the worked example:

        ```python
        iou((58, 50, 86, 78), (62, 46, 91, 77))
        ```

        That should give the same as counting the pixels: 700 / 1101, about 0.64. The check also tries boxes that do not overlap, boxes that only touch, and a small box in a large one.
        """),
        L.task("Make `iou(a, b)` give the intersection over union of the boxes `a` and `b`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iou():
    iou_code = L.code_box("iou", lines=14)
    iou_code
    return (iou_code,)


@app.cell(hide_code=True)
def check_iou(iou_code):
    result_iou = L.check("iou", iou_code.value)
    result_iou
    return (result_iou,)


@app.cell(hide_code=True)
def hints_iou():
    L.hints(
        "iou",
        [
            "Step 1 of the recipe: the overlap starts where both boxes have begun. Which of two tops is that, the smaller or the larger? And where does it end?",
            "Step 2: count the rows and the columns as `area` does, but make each 0 when it comes out below 0. `max(0, n)` gives the larger of 0 and n.",
            "`top = max(a[0], b[0])` and `bottom = min(a[2], b[2])`, and the same for left and right; `overlap = max(0, bottom - top + 1) * max(0, ___)`; `union = area(a) + area(b) - ___`.",
        ],
        why="For the circle and the predicted box, the overlap runs from row 62 to row 86, 25 rows, and from column 50 to column 77, 28 columns: 700 pixels. The union is 841 + 960 - 700 = 1101, and 700 / 1101 = 0.64.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_middle():
    mo.md(r"""
    ### Exercise 3. The right middle
    """)
    return


@app.cell(hide_code=True)
def choice_middle_cell():
    choice_middle = L.choice(
        "middle",
        question="A true box is 20 by 20 pixels. A detector predicts a box of 40 by 40 pixels with exactly the same middle, so the distance between their middles is 0. What is its IoU?",
        options={
            "one": "1",
            "half": "0.5",
            "quarter": "0.25",
            "zero": "0",
        },
        correct="quarter",
        why={
            "quarter": "The small box lies wholly inside the large one: the overlap is the small box, 400 pixels, and the union the large one, 1600 pixels. 400 / 1600 = 0.25, well below 0.5. The distance between the middles says nothing about size, and would call this box perfect; IoU sees that three quarters of it is background.",
            "one": "The middles are the same, but the boxes are not: the predicted box covers four times as many pixels. What are the overlap and the union?",
            "half": "Twice as wide and twice as tall is four times the area, not twice. How many pixels do the two boxes hold?",
            "zero": "0 is for boxes that do not overlap at all. Here the small box lies wholly inside the large one: what is their overlap?",
        },
    )
    choice_middle
    return (choice_middle,)


@app.cell(hide_code=True)
def text_exercise_big():
    mo.md(r"""
    ### Exercise 4. One box over everything
    """)
    return


@app.cell(hide_code=True)
def explain_big_cell():
    explain_big = L.explain(
        "big",
        question="A detector puts one box over the whole image, so it holds every object. Why does IoU give that box a low score, where the overlap divided by the area of the true box would give 1? And what does IoU do with a box that is too small?",
        model_answer="**The union holds every pixel of the predicted box.** Divided by the true box alone, a box that holds the whole object scores 1, however much background it takes in. **IoU divides by what both boxes cover together**, so that background counts against it: a box over the whole scene gets 841 / 16384 = 0.05 for the circle. **Too large and too small both lose.** A box too large makes the union grow, a box too small makes the overlap shrink, and only a box that matches gets close to 1.",
        points=[
            "you say that the union holds every pixel of the predicted box, so the background it takes in counts against it",
            "you say that dividing by the true box alone ignores the pixels that are too many, so a box over everything would score 1",
            "you say that IoU is only high when the box is neither too large nor too small",
        ],
    )
    explain_big
    return (explain_big,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), an annotation gave every object a box. IoU says how well a predicted box matches it, as one number from 0 to 1 that weighs the error against the size of the box. It needs only the four numbers of each box: the overlap runs from the larger tops and lefts to the smaller bottoms and rights, and the union is both areas minus the overlap. For masks, you count pixels with `&` and `|`. PASCAL VOC counts a box as found when its IoU is above 0.5, and a box shifted by a third of its width no longer gets above it. In the next lesson, *One box per object: non-maximum suppression*, a detector proposes many boxes for one object, and IoU tells which of them overlap too much to be about different objects.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_middle, choice_overlap, explain_big, result_iou):
    L.wrap_up(choice_overlap, result_iou, choice_middle, explain_big)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
