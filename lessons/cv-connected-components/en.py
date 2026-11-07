# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-connected-components. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-connected-components", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) and [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_queue():
    quick_queue = L.choice(
        "quick-queue",
        question="What does this program show?\n\n```python\nfrom collections import deque\nqueue = deque([(0, 0), (0, 1)])\nqueue.append((1, 0))\nprint(queue.popleft())\n```",
        options={
            "first": "`(0, 0)`",
            "second": "`(0, 1)`",
            "last": "`(1, 0)`",
        },
        correct="first",
        why={
            "first": "`append` puts `(1, 0)` at the back of the queue, and `popleft()` takes from the front: the oldest, `(0, 0)`. This lesson uses the queue the same way, with pixels instead of paths.",
            "second": "`(0, 1)` is second in line, and `popleft()` takes the one at the very front. See `deque` in [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en).",
            "last": "`(1, 0)` went in last, at the back, and `popleft()` takes from the front. See `deque` in [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en).",
        },
    )
    quick_queue
    return (quick_queue,)


@app.cell(hide_code=True)
def quick_check_opening():
    quick_opening = L.choice(
        "quick-opening",
        question="A binary image has a white speck of 2 by 2 pixels, far from any other white pixel. What is left of it after an opening with a square of 3 by 3?",
        options={
            "nothing": "Nothing",
            "one": "One pixel",
            "same": "The speck, as it was",
            "bigger": "A square of 4 by 4",
        },
        correct="nothing",
        why={
            "nothing": "A square of 3 by 3 fits nowhere inside the speck, so the erosion removes it, and the dilation cannot grow back what is gone.",
            "one": "After the erosion, is any pixel of the speck white everywhere in its window? A window of 3 by 3 does not fit inside 2 by 2. See *The idea* of [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en).",
            "same": "That is what an opening does to a shape the square fits in. Does a square of 3 by 3 fit in a speck of 2 by 2?",
            "bigger": "That is the dilation alone: it grows the speck by one pixel on every side. An opening starts with the erosion.",
        },
    )
    quick_opening
    return (quick_opening,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A binary image says pixel by pixel whether it is white or black, and nothing more. To say how many objects there are, how big each one is and where it lies, a program has to know which white pixels belong together. Connected components give exactly that: every group of white pixels that touch is one shape. A program counts cells under a microscope this way, splits a scanned page into letters before it reads them, and measures the parts on a factory line. And the box around each shape is the first step towards the boxes that a detection model draws around objects.

    **An everyday example.** The paint bucket of a drawing program. You click in an area, and the color spreads from pixel to pixel, to every pixel that touches a colored one, until it meets a border everywhere. The area it fills is one connected component. Click the paint bucket on every white pixel that has no color yet, each time with a new color, and every shape has its own color: you have counted them.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="The search of this lesson works like that paint bucket, but it also crosses corners. It starts at one pixel in the middle of a big white area, and in every step it colors every white pixel that touches a colored one, also at a corner. What do the colored pixels look like after 10 steps?",
        options={
            "disk": "A disk, 21 pixels across",
            "square": "A square of 21 by 21",
            "diamond": "A diamond, 21 pixels from corner to corner",
        },
        reveal="**A square of 21 by 21.** After the first step, the start and its eight neighbors are colored: a square of 3 by 3. Every next step adds a layer of one pixel all around, as a dilation with a square of 3 by 3 does: 5 by 5, then 7 by 7, and after 10 steps 21 by 21. If only the four neighbors above, below, on the left and on the right counted, it would be a diamond. At the edge of a shape the color stops: the slider below shows that in an apple.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Two white pixels **touch** when one is among the eight pixels around the other: above, below, on the left, on the right, or at a corner. A **connected component** is a group of white pixels in which you can walk from every pixel to every other one, in steps from a white pixel to a white pixel that touches it. No white pixel outside the group touches a pixel in it: the group is one shape. Its size is its number of pixels. Its **bounding box** is the smallest upright rectangle around it: from its smallest to its largest row, and from its smallest to its largest column. This lesson writes a box as four numbers, `(top, left, bottom, right)`.

        A breadth-first search finds a shape, the way it found a path through a maze. The states are now pixels, and an action is a step to a white pixel that touches the current one:

        - Start at a white pixel: put it in a queue, and in the set `seen` of the pixels the search has found.
        - Take the oldest pixel from the front of the queue. Each white neighbor that is not in `seen` yet goes into `seen` and at the back of the queue.
        - When the queue is empty, the search has found every pixel of the shape.

        There is no goal to reach: the search goes on until the queue is empty. To find every shape, go through the image row by row. At a white pixel that is not in `seen` yet, a new shape starts: search it, and go on with the next pixel. `seen` is one set for the whole image, so no search starts at a pixel of a shape that was found before.

        An example: an image of 5 rows by 7 columns, where 1 is white. Next to it, the number of the shape of every white pixel, in the order the scan finds them:

        ```
        image            shapes
        1 1 0 0 0 1 1    1 1 . . . 2 2
        0 1 0 0 1 1 0    . 1 . . 2 2 .
        0 0 0 0 0 0 0    . . . . . . .
        0 1 1 0 0 0 1    . 3 3 . . . 4
        1 0 0 0 0 0 1    3 . . . . . 4
        ```

        The pixel at row 4, column 0 touches the pixel at row 3, column 1 only at a corner, but that counts: it belongs to shape 3. Shape 3 has 3 pixels, and its box is `(3, 0, 4, 2)`: rows 3 to 4, columns 0 to 2. Shape 2 has 4 pixels, and its box is `(0, 4, 1, 6)`. If only the four neighbors above, below, on the left and on the right counted, the pixel at row 4, column 0 would be a shape of its own, and the image would have 5 shapes.
        """),
        L.watch_out("A box is four numbers, and programs write them in different orders. This lesson writes the rows first, as NumPy does: `(top, left, bottom, right)`. Many programs for object detection write `(x1, y1, x2, y2)`, with x the column: left and top first. Mixing up two orders is a classic bug."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **connected component** | a group of white pixels that touch, from neighbor to neighbor, and that touches no other white pixel: one shape | `find_shapes(img)` in the worked example; in OpenCV `cv2.connectedComponents` |
        | **bounding box** | the smallest upright rectangle around a shape: from its smallest to its largest row and column | `(top, left, bottom, right)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    The search from the middle of the apple at the top left, step by step, after the opening and the closing of the previous lesson. One step is one layer: the search takes out every pixel it found in the step before, and puts their new white neighbors at the back of the queue. Light gray is what the search has taken out of the queue, white what is still in it, and dark gray the pixels of the apple it has not found yet.
    """)
    return


@app.cell(hide_code=True)
def explore_search_cell():
    def _say(k):
        _o = C.search_outcome(k)
        if k == 0:
            return "Step 0: the queue holds the start alone."
        if _o["found"] == _o["apple"]:
            return (f"Step {k}: all {_o['apple']} pixels of the apple are found. The last {_o['queue']} are still in the queue; "
                    "they have no new neighbors, so after them the queue is empty.")
        return f"Step {k}: {_o['found']} pixels found, {_o['queue']} of them still in the queue."

    def _frame(k):
        k = int(k)
        _html = C.search_frame(k, {"title": "step {k}",
                                   "alt": "The apple at the top left of the drawing of fruit as a disk, after {k} steps of the search from its middle: {found} of its {apple} pixels found, in light gray, with the {queue} pixels still in the queue in white at the outside; the rest of the apple dark gray, on black."})
        return _html, _say(k)

    explore_search = L.explore("search", "step", _frame, start=0, values=C.STEPS, role="",
                               task="Slide to step 10: what shape do the found pixels have? Then slide to the last step.")
    explore_search
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    After 10 steps, the found pixels are the square of 21 by 21 of the guess, and after 20 steps a square of 41 by 41. Up to step 24 they stay a square; then the round edge of the apple cuts them off. After 33 steps, the search has found all 3617 pixels of the apple.
    """)
    return


@app.cell(hide_code=True)
def picture_boxes():
    mo.vstack([
        mo.md("**Four shapes and their boxes**"),
        mo.Html(C.boxes_picture({"alt": "The drawing of fruit in color, with a blue rectangle around each of the four apples, as tight as it fits, and the size of the apple in pixels above its top left corner: 3617, 4051, 2111 and 3847. The orange, the lemon and the lime have no rectangle."})),
        mo.md(r"""
        In blue, what the program gives back after the opening and the closing: the box of each of the four shapes, with its size in pixels.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Put a pixel into `seen` as soon as it goes into the queue, and not only when it comes out. Then no pixel goes into the queue twice. The search of the maze put a state into its set `done` only when it came out of the queue; here a pixel has up to eight white neighbors, and each of them would put it in the queue again."),
        L.watch_out("At the border, part of the window lies outside the image, as in the previous lesson. `max(r - 1, 0)` keeps the first row at 0 or more, and `min(r + 2, len(img))` keeps the end of the `range` at the number of rows: a `range` past the last row asks for a row that is not there."),
        mo.md(r"""
        ### Recipe

        To find every shape of a binary image `img`, with its size and its box:

        1. Make an empty set `seen`. Go through the pixels row by row; at a white pixel that is not in `seen`, a new shape starts.
        2. Put that pixel into `seen` and into a queue, `deque([start])`, and make an empty list `shape`.
        3. While the queue is not empty: take the oldest pixel from the front with `popleft()`, and add it to `shape`. Each white pixel in the window of 3 by 3 around it, only the part inside the image, that is not in `seen` yet goes into `seen` and at the back of the queue.
        4. When the queue is empty, the shape is complete. Its size is `len(shape)`; its box runs from the smallest to the largest row and from the smallest to the largest column of its pixels.
        5. Go on with step 1 at the next pixel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell picks the red pixels of the drawing of fruit with the rule for red and 0.45, and tidies them up as at the end of the previous lesson: an opening of 3 by 3 and a closing of 5 by 5, with `erode` and `dilate` for a square of `k` by `k`. It prints how many pixels are white before and after. The second cell follows the recipe: `white_neighbors` gives the white pixels in the window of 3 by 3 around a pixel, `shape_from` searches one shape, and `find_shapes` goes through the image. It prints how many shapes the red pixels form before the tidying up, with the six largest sizes, and then, for each shape after it, its size and the pixel where its search started.
    """)
    return


@app.cell
def example_mask():
    import numpy as np
    from PIL import Image
    from matplotlib.colors import rgb_to_hsv

    img = np.array(Image.open("data/photos/fruit.png"))
    hsv = rgb_to_hsv(img / 255)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # the rule for red, with 0.45: apples, specks of the tablecloth, and holes
    mask = ((h < 0.05) | (h > 0.95)) & (s > 0.45)


    def erode(img, k):
        # the smallest value of every window of k by k, as in the previous lesson
        half = k // 2
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                top, left = max(r - half, 0), max(c - half, 0)
                out[r, c] = img[top:r + half + 1, left:c + half + 1].min()
        return out


    def dilate(img, k):
        # the largest value of every window of k by k
        half = k // 2
        out = np.zeros(img.shape, dtype=bool)
        for r in range(len(img)):
            for c in range(len(img[0])):
                top, left = max(r - half, 0), max(c - half, 0)
                out[r, c] = img[top:r + half + 1, left:c + half + 1].max()
        return out


    # an opening of 3 by 3 removes the specks, a closing of 5 by 5 the holes
    clean = erode(dilate(dilate(erode(mask, 3), 3), 5), 5)
    print(mask.sum(), clean.sum())
    return clean, mask


@app.cell
def example_shapes(clean, mask):
    from collections import deque


    def white_neighbors(img, r, c):
        # the white pixels in the window of 3 by 3 around (r, c), inside the image
        result = []
        for nr in range(max(r - 1, 0), min(r + 2, len(img))):
            for nc in range(max(c - 1, 0), min(c + 2, len(img[0]))):
                if img[nr, nc]:
                    result.append((nr, nc))
        return result


    def shape_from(img, start, seen):
        # 2. the start goes into seen and into the queue
        seen.add(start)
        queue = deque([start])
        shape = []
        while len(queue) > 0:
            # 3. the oldest pixel from the front; its new white neighbors at the back
            r, c = queue.popleft()
            shape.append((r, c))
            for nxt in white_neighbors(img, r, c):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        # 4. the queue is empty: the shape is complete
        return shape


    def find_shapes(img):
        # 1. row by row: a white pixel that is not in seen starts a new shape
        seen = set()
        shapes = []
        for r in range(len(img)):
            for c in range(len(img[0])):
                if img[r, c]:
                    if (r, c) not in seen:
                        shapes.append(shape_from(img, (r, c), seen))
        return shapes


    sizes = []
    for shape in find_shapes(mask):
        sizes.append(len(shape))
    print(len(sizes), sorted(sizes)[-6:])
    for shape in find_shapes(clean):
        print(len(shape), shape[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The rule picks 16,190 red pixels; after the opening and the closing, 13,626 are white. Before the tidying up, the red pixels form 1335 shapes. The four largest, of 2094 to 4029 pixels, are the apples, each with its hole and the specks that touch it. The next largest has 4 pixels: the other 1331 shapes are the specks, of 1 to 4 pixels each. After the tidying up, 4 shapes are left, one per apple, of 2111 to 4051 pixels. The scan finds them in the order of their highest pixel: first the apple at the top left, whose search starts at row 37, column 62, the top of the apple. In exercise 3 you remove the specks in another way: by their size.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. How many shapes?

    A small binary image; 1 is white:

    ```
    0 1 0 0 1
    1 0 0 1 0
    0 0 1 0 0
    1 0 0 0 1
    1 1 0 1 1
    ```
    """)
    return


@app.cell(hide_code=True)
def choice_count_cell():
    choice_count = L.choice(
        "count",
        question="How many shapes does the search of this lesson find, with all eight neighbors?",
        options={
            "two": "2",
            "four": "4",
            "seven": "7",
            "eleven": "11",
        },
        correct="four",
        why={
            "four": "Pixels that touch at a corner belong together. The two pixels at the top left form one shape; the diagonal line from the top right to the middle, `(0, 4)`, `(1, 3)` and `(2, 2)`, another; and each corner at the bottom is a shape of three pixels. The pixel in the middle touches neither corner at the bottom: the pixels of row 3 next to it are black.",
            "two": "Then the shapes at the top would join those at the bottom. Does any white pixel of row 2 touch a white pixel of row 3? Look at the window of 3 by 3 around the pixel in the middle.",
            "seven": "That is what you count with only the four neighbors above, below, on the left and on the right: then every pixel of a diagonal line is a shape of its own. In this lesson, pixels that meet only at a corner touch too.",
            "eleven": "That is the number of white pixels. A shape is a group of white pixels that touch: which pixels touch each other?",
        },
    )
    choice_count
    return (choice_count,)


@app.cell(hide_code=True)
def text_exercise_box():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The box of one shape

        Complete `box_of(img, start)`, which gives the bounding box of the shape that holds the pixel `start`, as `(top, left, bottom, right)`. `img` is a binary image, a list of rows of 0 and 1, and `start` a white pixel, as (row, column). `white_neighbors` from the worked example is already there, and so is the start of the search. Replace the first `...` with one pass of the loop, step 3 of the recipe, and the second with the box: `rows` holds the row of every pixel of the shape, and `cols` its column. The check calls it with this image and the pixel at row 1, column 1:

        ```python
        shapes = [[0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1],
                  [0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0]]
        box_of(shapes, (1, 1))
        ```

        That should give `(1, 1, 3, 2)`: the pixel at row 3, column 2 touches the shape at a corner. The check also tries other start pixels, and shapes at the border of the image.
        """),
        L.task("Make `box_of(img, start)` search the shape around `start` and give back its box."),
    ])
    return


@app.cell(hide_code=True)
def exercise_box():
    box_code = L.code_box("box", lines=20)
    box_code
    return (box_code,)


@app.cell(hide_code=True)
def check_box(box_code):
    result_box = L.check("box", box_code.value)
    result_box
    return (result_box,)


@app.cell(hide_code=True)
def hints_box():
    L.hints(
        "box",
        [
            "Step 3 of the recipe says what one pass of the loop does, and `shape_from` in the worked example does it: take the oldest pixel out, add it to `shape`, and put its new white neighbors in. The box is the smallest and the largest row and column of `shape`.",
            "Take the pixel at the front of the queue with `popleft()`, and add it to `shape`. For every neighbor from `white_neighbors(img, r, c)` that is not in `seen` yet: add it to `seen`, and put it at the back of the queue. The box is the smallest row, the smallest column, the largest row and the largest column.",
            "`r, c = queue.___()` and `shape.append((r, c))`; then `for nxt in white_neighbors(img, r, c):` with `if nxt not in seen:`, `seen.add(nxt)` and `queue.append(nxt)`. At the end: `return rows.min(), cols.min(), ___, ___`.",
        ],
        why="For `box_of(shapes, (1, 1))`, the queue starts with `(1, 1)`. The first pass takes it out and puts its white neighbors `(1, 2)` and `(2, 1)` in the queue. The next passes take those out; `(2, 1)` puts in `(3, 2)`, which touches it at a corner. Then the queue is empty: `shape` holds four pixels, in rows 1 to 3 and columns 1 to 2. The search never reaches the shape on the right: none of its pixels touches a pixel of the first shape.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smallest():
    mo.md(r"""
    ### Exercise 3. Remove the specks by their size

    The opening removed the specks because a square did not fit in them. Now that you can measure every shape, there is another way: keep only the shapes of at least a certain size. With the slider you choose that size, for the red pixels before any tidying up. On the left, the shapes that are left, each in its blue box when there are 10 or fewer; on the right, how many pixels are wrong at every stop of the slider: white where the course drew no apple, or black where it drew one. The stops go from 1 to 5, and then jump to 10, 100, 1000, 2000 and 3000.
    """)
    return


@app.cell(hide_code=True)
def explore_smallest_cell():
    def _frame(n):
        _html, _f = C.smallest_frame(n, {"kept": "shapes of at least {n} pixels", "kept_1": "shapes of at least 1 pixel", "size": "smallest size kept", "wrong": "pixels wrong",
                                         "alt": "On the left, the red pixels of the drawing of fruit, keeping only the shapes of at least {n} pixels, as white on black; on the right, how many pixels are wrong for the smallest sizes 1 to 3000, one point per stop of the slider: 2780 at 1, 470 at 4, 146 from 5 to 2000, and 2232 at 3000. Size {n} is marked by a larger point."})
        return (_html,
                f'Shapes of at least {n} pixel{"" if n == 1 else "s"}: {_f["shapes"]} shapes left, {_f["wrong"]} pixels wrong.')

    explore_smallest = L.explore("smallest", "smallest size kept", _frame, start=1, values=C.SMALLEST, role="hyper",
                                 task="Find the smallest size that keeps only the four apples.")
    explore_smallest
    return (explore_smallest,)


@app.cell(hide_code=True)
def check_smallest(explore_smallest):
    result_smallest = L.check("smallest", explore_smallest)
    result_smallest
    return (result_smallest,)


@app.cell(hide_code=True)
def hints_smallest():
    L.hints(
        "smallest",
        [
            "Move the slider one stop at a time, from 1 to the right, and read how many shapes are left.",
            "In the worked example, the four largest shapes had 2094 to 4029 pixels, and the next largest had 4.",
            "Look at the stops 4 and 5: after which one are only four shapes left?",
        ],
        why="At 4, 85 shapes are left: the four apples and 81 specks of 4 pixels. At 5, only the four apples are left: every speck has 1 to 4 pixels. Every size up to 2000 gives the same four shapes and 146 pixels wrong; at 3000, the small apple, of 2094 pixels, is gone too.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.md(r"""
    ### Exercise 4. The filter and the opening
    """)
    return


@app.cell(hide_code=True)
def explain_filter_cell():
    explain_filter = L.explain(
        "filter",
        question="With the filter by size of 5, 146 pixels are wrong: the 101 pixels of the holes of the shine, and 45 pixels of the tablecloth. The opening of 3 by 3 of the previous lesson left the same holes, but only 14 pixels of the tablecloth. Why does the filter keep more of the tablecloth?",
        model_answer="**A speck that touches an apple is part of the apple's shape.** The search goes from white pixel to white pixel that touches, also at a corner, and cannot see where the apple ends and the speck begins: together they are one connected component. **The filter judges whole shapes.** It keeps or removes a shape at once, by its size; the apple is big, so the speck that touches it stays too. **The opening judges every pixel by its own square.** A speck on the edge of an apple is too thin for the square of 3 by 3, so the erosion removes it, whether it touches the apple or not; a pixel of the tablecloth stays only where it lies in a white square of 3 by 3 together with pixels of the apple, and that is so for 14 of them.",
        points=[
            "you say that a speck that touches an apple, also at a corner, belongs to the shape of the apple",
            "you say that the filter keeps or removes whole shapes, so the speck stays with the big apple",
            "you say that the opening looks at every pixel through its square, so it removes a thin speck even when it touches an apple",
        ],
    )
    explain_filter
    return (explain_filter,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The breadth-first search of [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) found a shortest path to a goal; here it finds every pixel of a shape, and a scan over the image starts it again at every white pixel that no search has found yet. Step by step, the search grows like a dilation with a square of 3 by 3 from [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en), but never past the edge of the shape. Connected components turn a binary image into things you can count and measure: after the opening and the closing, the drawing of fruit has four shapes, each with its size and its box. A filter by size removes specks too, but keeps whatever touches a big shape. Later in this branch, a box is what a detection model predicts for every object it sees, and you measure how well two boxes overlap. The next lesson, *Bigger and smaller: interpolation*, makes an image bigger or smaller, and has to make up the pixels in between.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count, explain_filter, result_box, result_smallest):
    L.wrap_up(choice_count, result_box, result_smallest, explain_filter)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
