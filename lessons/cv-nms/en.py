# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-nms. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-nms", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_iou():
    quick_iou = L.choice(
        "quick-iou",
        question="What is the IoU of the boxes `(10, 10, 19, 19)` and `(10, 15, 19, 24)`, written as `(top, left, bottom, right)` with both ends included?",
        options={
            "third": "0.33",
            "half": "0.5",
            "quarter": "0.25",
            "one": "1",
        },
        correct="third",
        why={
            "third": "Each box holds 10 by 10 = 100 pixels. They share rows 10 to 19 and columns 15 to 19: 10 by 5 = 50 pixels. Together they cover 100 + 100 - 50 = 150, and 50 / 150 = 0.33. In this lesson, IoU tells which boxes are about the same object.",
            "half": "50 / 100 divides the overlap by one box. IoU divides by what the two boxes cover together: both areas, minus the overlap.",
            "quarter": "50 / 200 adds the two areas, so the 50 pixels they share are counted twice. Subtract the overlap once.",
            "one": "The two boxes are as large as each other, but they do not lie in the same place: the second one starts 5 columns further to the right. What do they share?",
        },
    )
    quick_iou
    return (quick_iou,)


@app.cell(hide_code=True)
def quick_check_found():
    quick_found = L.choice(
        "quick-found",
        question="PASCAL VOC counts a predicted box as found when its IoU with the true box is above 0.5. Which of these boxes counts as found?",
        options={
            "above": "only the box with an IoU of 0.53",
            "both": "the box with 0.53 and the box with exactly 0.5",
            "half": "only the box with exactly 0.5",
            "neither": "neither of the two",
        },
        correct="above",
        why={
            "above": "0.53 is above 0.5, and 0.5 itself is not. In this lesson, too, a rule says \"above\": a box is dropped when its IoU with a kept box is above a set number, so a box at exactly that number stays.",
            "both": "\"Above 0.5\" leaves out 0.5 itself. Which of the two is above it?",
            "half": "0.5 is not above 0.5. Which box is?",
            "neither": "0.53 is above 0.5, so that box counts as found. Which one is not?",
        },
    )
    quick_found
    return (quick_found,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A detector does not find an object just once. It looks at an image in many places and at many sizes, and wherever something looks like an object, it gives a box. Around one object, many of those boxes look almost the same, so the detector gives a whole crowd of boxes for every object. To count the objects, or to draw one box around each, you have to keep one box per object and drop the rest.

    **An everyday example.** Five friends each put a pin on a shared map for every café they know in town, so a café can get several pins close together. You want one pin per café. For each café you keep the pin of the friend who is surest, and remove the pins that land almost on the same spot: they mean the same café. A pin far from every pin you kept stays: it marks another café.
    """)
    return


@app.cell(hide_code=True)
def predict_top():
    guess_top = L.guess(
        "top",
        question="A detector gives every box a score from 0 to 1: how sure it is that an object sits in it. For the 4 shapes of the scene from the last two lessons, it gives 28 boxes. You keep the 4 boxes with the highest scores. How many of the 4 shapes get a box?",
        options={
            "four": "4",
            "three": "3",
            "two": "2",
            "one": "1",
        },
        reveal="**1.** The 6 highest scores all belong to boxes around the circle, the shape the detector is surest of: boxes that are almost the same get almost the same score. The square's highest score only comes ninth: 0.70. The highest scores give you copies of one object; what you need is the highest score of each object.",
    )
    guess_top
    return (guess_top,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Every box a detector gives comes with a **confidence score**: a number from 0 to 1 that says how sure the detector is that an object sits in that box. Boxes that are almost the same get almost the same score, so the highest scores sit together, on the object the detector is surest of.

        **Non-maximum suppression**, or NMS, keeps one box per object. It works in rounds, on a list of all the boxes. Take the box with the highest score in the list, and keep it. Then drop from the list every box whose IoU with it is above a **threshold**, such as 0.5: those overlap it so much that they must be about the same object. The kept box leaves the list too, as its IoU with itself is 1. Then do the same with the boxes left in the list: keep the one with the highest score, drop every box whose IoU with it is above the threshold, and so on, until the list is empty. The name says what it does: every box that is not the maximum of its group is suppressed.

        The threshold decides what counts as the same object. Set it too low, and the box of an object close to another one is dropped too, as if it were a copy. Set it too high, and two boxes around the same object both stay. A common choice is 0.5, which is also what torchvision, PyTorch's library for images, uses by default for the final boxes of its Faster R-CNN detector.

        The boxes in this lesson come from the course: around each shape of the scene, 7 boxes, moved and stretched a little at random, each with a score that is mostly higher the better the box fits, with a little noise, as with a real detector. That is what a detector gives before NMS; a later lesson uses a real detector. A shape's best box, in this lesson, is its box with the highest score.
        """),
        L.watch_out("A box that is dropped drops nothing. Compare each box only with the boxes you have kept, not with every box that has a higher score: a box whose IoU is high only with a dropped box stays. It may be another object next to it."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **confidence score** | how sure a detector is that an object sits in a box, from 0 to 1 | `scores[i]` |
        | **non-maximum suppression** (NMS) | keep the box with the highest score, drop every box whose IoU with it is above the threshold, and repeat with the boxes that are left | `nms(boxes, scores, 0.5)` |
        | **threshold** | the IoU above which two boxes count as the same object | `iou(a, b) > threshold` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_proposals():
    mo.vstack([
        mo.md("**28 boxes, and the 4 that NMS keeps**"),
        mo.Html(C.proposals_picture({
            "all": "all {n} boxes",
            "kept": "the {n} boxes NMS keeps",
            "point": ".",
            "alt": "Two pictures of the same scene: a large blue triangle, a blue square partly under a red circle, and a small yellow triangle. Left, 28 dashed boxes, 7 close together around each shape. Right, only 4 dashed boxes are left, one around each shape, each with its score: 0.84 for the circle, 0.72 for the small triangle, 0.70 for the square and 0.59 for the large triangle.",
        })),
        mo.md(r"""
        Left, the 28 boxes for the 4 shapes, dashed: around each shape a crowd of 7. Right, the 4 boxes that NMS keeps with a threshold of 0.5, one per shape, with their scores: 0.84 for the circle, 0.72 for the small triangle, 0.70 for the square and 0.59 for the large triangle.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    What does the threshold do? With the slider you set it anywhere from 0.05 to 0.95. On the left, the boxes that NMS keeps; on the right, how many it keeps at every threshold, with a dotted line at 4, one box per shape.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _html, _n = C.threshold_frame(t, {"title": "threshold {t}: {n} boxes", "count": "boxes kept", "threshold": "threshold", "point": ".",
                                          "alt": "The scene with the {n} boxes that NMS keeps at a threshold of {t}, dashed. Next to it, how many boxes NMS keeps at every threshold from 0 to 1: 3 at the lowest thresholds, 4 from 0.14 to 0.57, then more and more, up to all 28 from 0.94. A dotted line marks 4, one box per shape, and a larger point the threshold of {t}."})
        return _html, f"Threshold {L.number(t, 2)}: NMS keeps {_n} boxes."

    explore_threshold = L.explore("threshold", "threshold", _frame, start=0.5, values=C.THRESHOLDS, role="hyper",
                                  task="Find the largest threshold at which every shape still keeps exactly one box.")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    From 0.15 to 0.55, NMS keeps 4 boxes, one per shape. At 0.6, the small triangle and the square each keep a second box, whose IoU with their first box is no longer above the threshold. From there the number climbs, until at 0.95 all 28 boxes stay.

    At the low end, something else goes wrong. The square lies partly under the circle, so its boxes overlap the circle's kept box: the IoU is 0.26 for its best box, and at least 0.13 for every one of its 7 boxes. At 0.1 and 0.05, the circle's box drops all 7, and the square has no box at all. At 0.15 and 0.2, it keeps only its worst box, with a score of 0.45, the one with the lowest IoU with the circle's box. At 0.25 it keeps a box with 0.65, and only from 0.3 its best box, with 0.70. On this scene, every threshold from 0.3 to 0.55 keeps the best box of every shape: low enough to drop the copies, high enough to keep the square next to the circle.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To keep one box per object:

    1. Put the numbers of all boxes in a list.
    2. Find the box with the highest score in the list, and keep it.
    3. Take out of the list every box whose IoU with the kept box is above the threshold; the kept box goes too.
    4. Repeat steps 2 and 3 until the list is empty. The boxes you kept are the result, one per object.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe on the scene of the last two lessons, `scene(18)`. `proposals(s, seed=25)` gives the 28 boxes the course makes for it, and a score for each, in the same order: `scores[k]` is the score of `boxes[k]`. The first cell prints the boxes with a score above 0.7.

    The second cell runs NMS with a threshold of 0.5. `iou` is the function of exercise 2 of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en). `rest` holds the numbers of the boxes that are still in the list, so that `boxes[i]` and `scores[i]` stay together. The `while` loop runs as long as the list is not empty. Each round, the `for` loop finds the box with the highest score: `best` starts as the first number in `rest`, and whenever the loop meets a box with a higher score, that box becomes `best`. Then the list comprehension keeps only the boxes whose IoU with `best` is 0.5 or less. The cell prints each kept box, its score, and how many boxes are left after the round.
    """)
    return


@app.cell
def example_boxes():
    from everything_ai.scenes import scene, proposals

    s = scene(18)
    # 28 boxes, and a score for each: scores[k] belongs to boxes[k]
    boxes, scores = proposals(s, seed=25)
    print(len(boxes), "boxes")
    for k in range(len(boxes)):
        if scores[k] > 0.7:
            print(k, boxes[k], scores[k])
    return boxes, scores


@app.cell
def example_nms(boxes, scores):
    def iou(a, b):
        # the intersection over union of two boxes, as in the previous lesson
        top, left = max(a[0], b[0]), max(a[1], b[1])
        bottom, right = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0, bottom - top + 1) * max(0, right - left + 1)
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return overlap / (area_a + area_b - overlap)


    rest = list(range(len(boxes)))   # 1. the numbers of all boxes
    kept = []
    while len(rest) > 0:             # 4. until the list is empty
        # 2. the box with the highest score in the list: keep it
        best = rest[0]
        for i in rest:
            if scores[i] > scores[best]:
                best = i
        kept.append(best)
        # 3. keep in the list only the boxes with an IoU of 0.5 or less
        rest = [i for i in rest if iou(boxes[i], boxes[best]) <= 0.5]
        print(best, boxes[best], scores[best], len(rest))
    print(kept)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first cell prints the 8 boxes with a score above 0.7. Seven of them lie around the circle: their tops lie between rows 56 and 60, and they end at column 78 or 79, almost the same box seven times. The eighth, box 13 with 0.72, is the best box of the small triangle.

    The second cell needs 4 rounds. Round 1 keeps box 6, the circle's best, with 0.84, and takes the 7 boxes around the circle out of the list, box 6 itself too: 21 are left. The square's boxes stay, because their IoU with box 6 is 0.31 at most, not above 0.5. Round 2 keeps box 13 and takes out the 7 boxes of the small triangle; round 3 keeps box 0, the square's best, with 0.70; round 4 keeps box 23, with 0.59, for the large triangle, and the list is empty. NMS keeps 4 boxes, one per shape: `[6, 13, 0, 23]`. NMS trusts the scores: box 24 fits the large triangle better, with an IoU of 0.85 against 0.79, but it has a lower score, 0.56, so box 23 stays and box 24 goes. Exercise 2 puts this loop in a function that also gets the threshold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chain():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. By hand
    """)
    return


@app.cell(hide_code=True)
def choice_chain_cell():
    choice_chain = L.choice(
        "chain",
        question="Four boxes have these scores: A 0.9, B 0.8, C 0.7 and D 0.6. A and B overlap with an IoU of 0.7, B and C with 0.6, and C and D with 0.55; every other pair has an IoU below 0.5. Which boxes does NMS keep with a threshold of 0.5?",
        options={
            "right": "A and C",
            "dropped": "A and D",
            "only": "only A",
            "three": "A, C and D",
        },
        correct="right",
        why={
            "right": "Round 1 keeps A and drops B, whose IoU with A is 0.7; C and D have an IoU below 0.5 with A, so they stay. Round 2 keeps C, the highest score that is left, and drops D: 0.55 is above 0.5. B has an IoU of 0.6 with C, but B was already gone, and a box that is dropped drops nothing.",
            "dropped": "That lets B drop C. But A dropped B in round 1, and a box that is dropped drops nothing. Which box has the highest score after round 1?",
            "only": "That drops every box whose IoU with a box with a higher score is above 0.5, also when that box was dropped itself. Of the boxes with a higher score, only B has an IoU above 0.5 with C, and B is gone after round 1.",
            "three": "C is kept in round 2, and then drops every box whose IoU with it is above 0.5. What is the IoU of C and D?",
        },
    )
    choice_chain
    return (choice_chain,)


@app.cell(hide_code=True)
def text_exercise_nms():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. NMS as a function

        Complete `nms(boxes, scores, threshold)`. It gets a list of boxes, each `(top, left, bottom, right)`, a list with the score of each box, and a threshold. It gives back the numbers of the boxes it keeps: their places in the list `boxes`. `iou` is given, as in the worked example. The check calls it with the 28 boxes of the scene:

        ```python
        nms(boxes, scores, 0.5)
        ```

        That should give the 4 boxes of the worked example, 6, 13, 0 and 23, in any order. The check also tries other thresholds, two boxes whose IoU is exactly the threshold, boxes that overlap in a chain, as in exercise 1, and no boxes at all.
        """),
        L.task("Make `nms(boxes, scores, threshold)` give the numbers of the boxes that non-maximum suppression keeps."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nms():
    nms_code = L.code_box("nms", lines=24)
    nms_code
    return (nms_code,)


@app.cell(hide_code=True)
def check_nms(nms_code):
    result_nms = L.check("nms", nms_code.value)
    result_nms
    return (result_nms,)


@app.cell(hide_code=True)
def hints_nms():
    L.hints(
        "nms",
        [
            "Step 2 of the recipe: start with `best = rest[0]`, then go through `rest` with a `for` loop, and whenever a box has a higher score than `best`, make it `best`. Then `kept.append(best)`.",
            "Step 3: keep in `rest` only the boxes whose IoU with `best` is at most `threshold`. A list comprehension with an `if` at the end builds that list. `best` itself goes too, because its IoU with itself is 1.",
            "Steps 2 and 3 go inside `while len(rest) > 0:`, so that they repeat until the list is empty. The last line of the loop: `rest = [i for i in rest if iou(boxes[i], boxes[best]) <= ___]`.",
        ],
        why="For the scene at 0.5, round 1 keeps box 6, with the highest score, 0.84, and takes the circle's other boxes out of the list; rounds 2 to 4 keep boxes 13, 0 and 23. The function gives `[6, 13, 0, 23]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cars():
    mo.md(r"""
    ### Exercise 3. Close together
    """)
    return


@app.cell(hide_code=True)
def choice_cars_cell():
    choice_cars = L.choice(
        "cars",
        question="Two cars stand close together in a photo, one partly behind the other. A detector gives good boxes around both, and is surest of the car in front. Every box it gives for the car behind has an IoU between 0.55 and 0.7 with the best box of the car in front. Then it runs NMS with a threshold of 0.5. What does it report?",
        options={
            "one": "one car: the boxes of the car behind are dropped",
            "two": "two cars: NMS sees that they are different objects",
            "merged": "one large box around both cars",
            "none": "no car: the two boxes drop each other",
        },
        correct="one",
        why={
            "one": "The best box of the car in front is kept first. Every box of the car behind has an IoU above 0.5 with it, so NMS drops them all, as if they were copies: two cars, one box. A higher threshold would keep the second car, but it would also let copies of one car through, as the slider showed from 0.6 on.",
            "two": "NMS does not look at what is in the boxes, only at their scores and their IoU. To NMS, a box whose IoU with a kept box is above the threshold is a copy.",
            "merged": "NMS never makes a new box: it only keeps or drops the boxes it gets.",
            "none": "Only a kept box drops others, and the box with the highest score is always kept. Which box is kept first?",
        },
    )
    choice_cars
    return (choice_cars,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 4. Kinds
    """)
    return


@app.cell(hide_code=True)
def explain_kinds_cell():
    explain_kinds = L.explain(
        "kinds",
        question="The detector of a later lesson also says what kind of object each box holds, and gives the score for that kind. In the scene, the square lies partly under the circle, and at a threshold of 0.1 the circle's box dropped every box of the square. How can the detector use the kind to prevent that? And when does that not help?",
        model_answer="**Run NMS for each kind on its own.** A box then only drops boxes of its own kind, so a box for a circle never drops a box for a square, however much they overlap: boxes of different kinds are taken to be different objects. That way, a cup in front of a teapot and the teapot both keep their box; torchvision's Faster R-CNN works like this. **It does not help for two objects of the same kind**, such as the two cars of exercise 3: their boxes are still compared, and the threshold still has to weigh dropping copies against keeping neighbors. **It can also keep too much**: when the detector gives one object boxes of two kinds, both stay.",
        points=[
            "you say to run NMS for each kind on its own, so that a box only drops boxes of the same kind",
            "you say that boxes of different kinds are taken to belong to different objects, even when they overlap a lot",
            "you name a case where it does not help, such as two overlapping objects of the same kind, or one object with boxes of two kinds",
        ],
    )
    explain_kinds
    return (explain_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en), IoU said how well a predicted box matches the true box. Here it says which boxes of a detector are about the same object. A detector gives a crowd of boxes around every object, each with a confidence score. Non-maximum suppression keeps the box with the highest score, drops every box whose IoU with it is above a threshold, and repeats that with the boxes that are left; only kept boxes drop others. The threshold weighs dropping copies against keeping neighbors: on the scene, 0.3 to 0.55 keeps the best box of every shape, at 0.1 the square under the circle loses its box, and from 0.6 copies stay. In the next lesson, *Score a detector: mean average precision*, the boxes that a detector keeps are compared with the true boxes, and their scores decide in which order they count.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cars, choice_chain, explain_kinds, result_nms):
    L.wrap_up(choice_chain, result_nms, choice_cars, explain_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
