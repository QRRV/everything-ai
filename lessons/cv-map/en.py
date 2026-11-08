# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-map. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-map", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en) and about precision and recall, from [Every threshold at once: the ROC curve and AUC](/course/open/ml-roc-auc?lang=en) and the lessons before it.
    """)
    return


@app.cell(hide_code=True)
def quick_check_nms():
    quick_nms = L.choice(
        "quick-nms",
        question="Non-maximum suppression with a threshold of 0.5 gets three boxes: A with a score of 0.9, B with 0.8, whose IoU with A is 0.6, and C with 0.7, whose IoU with A and with B is 0.1. Which boxes does it keep?",
        options={
            "right": "A and C",
            "one": "only A",
            "all": "A, B and C",
            "two": "A and B",
        },
        correct="right",
        why={
            "right": "Round 1 keeps A, the highest score, and drops B: 0.6 is above 0.5. C stays, as its IoU with A is only 0.1, and round 2 keeps it. In this lesson, the detector's boxes are the ones that NMS kept.",
            "one": "A drops only the boxes whose IoU with it is above 0.5. Is C one of them?",
            "all": "B's IoU with A is 0.6, above the threshold of 0.5. What happens to B when A is kept?",
            "two": "B's IoU with A is 0.6, so A drops it. And C, whose IoU with A is only 0.1?",
        },
    )
    quick_nms
    return (quick_nms,)


@app.cell(hide_code=True)
def quick_check_pr():
    quick_pr = L.choice(
        "quick-pr",
        question="A detector of 5s raises 10 alarms, and 7 of them are real 5s. The test set holds 14 5s. What are its precision and recall?",
        options={
            "right": "precision 0.7, recall 0.5",
            "swapped": "precision 0.5, recall 0.7",
            "both-p": "precision 0.7, recall 0.7",
            "both-r": "precision 0.5, recall 0.5",
        },
        correct="right",
        why={
            "right": "Precision is the share of its alarms that are right: 7 / 10 = 0.7. Recall is the share of the real 5s that it finds: 7 / 14 = 0.5. In this lesson, both are worked out after every box of a detector.",
            "swapped": "Precision divides by what the detector says, its 10 alarms; recall divides by what is really there, the 14 5s.",
            "both-p": "7 / 10 is the precision. Recall divides the 7 hits by the 5s that are really there: how many is that?",
            "both-r": "7 / 14 is the recall. Precision divides the 7 hits by the alarms the detector raised: how many is that?",
        },
    )
    quick_pr
    return (quick_pr,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A detector draws boxes, each with a kind and a score. How good is it? You cannot just count right answers, as accuracy does: the detector decides itself how many boxes it draws, a box can be almost right, and one object can get two boxes. A score for a detector has to say how many objects it finds, how many of its boxes are wrong, and whether the boxes it is surest of are right. Papers and leaderboards compare detectors with one such number: mean average precision.

    **An everyday example.** Children look for 20 eggs hidden in a garden, and call out each find, the one they are surest of first. A call counts when there is an egg there that nobody found yet; a call at an empty spot, or at an egg found already, does not. A good searcher finds many of the 20 eggs, and the calls they were surest of are right.
    """)
    return


@app.cell(hide_code=True)
def predict_worse():
    guess_worse = L.guess(
        "worse",
        question="A detector is scored by going down its boxes for circles from the highest score. Two of the false ones are the box with the highest score, 0.86, which lies on a square, and the box with the lowest, 0.30, which does not fit its circle well enough. Without which of the two would the detector score better?",
        options={
            "high": "without the box with 0.86",
            "low": "without the box with 0.30",
            "same": "it makes no difference which one",
        },
        reveal="**Without the box with 0.86.** The score of this lesson, average precision, looks at the precision at every true box found, going down from the highest score. A false box at the top lowers the precision at every found box after it: without it, the circles' score rises from 0.56 to 0.73. The false box with 0.30 comes after the last found box: without it, the score stays 0.56. A false box costs more the higher its score.",
    )
    guess_worse
    return (guess_worse,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        To score a detector you need images with their true boxes, and the boxes the detector draws on them after non-maximum suppression, each with a kind and a confidence score. First you decide for each box whether it is right. A box is a **true positive** when its IoU with a true box of its kind in the same image is above 0.5, and no box with a higher score found that true box first. Every other box is a false positive: a box on the background, on a shape of another kind, a box that fits too loosely, or a second box on a shape that is found already. A true box that no box finds is a false negative, a miss. You go through the boxes from the highest score down, so the box the detector is surest of gets the first chance.

        Then, for each kind on its own, you go down the list of its boxes, the highest score first. After each box, precision is the share of the boxes so far that are true positives, and recall the share of the true boxes of that kind found so far. Stopping after a box is the same as a decision threshold at its score, so these points form the precision-recall curve of [Every threshold at once](/course/open/ml-roc-auc?lang=en).

        **Average precision**, or AP, is the area under that curve. Each true positive moves recall up by one step, 1 divided by the number of true boxes, and over that step the curve has the height of the precision at that box. So AP adds up the precision at each true positive, and divides by the number of true boxes. That is where its name comes from: it is the average of the precision at each true box, with 0 for each true box that is never found. AP is 1 only when every true box is found before the first false positive.

        **Mean average precision**, or mAP, is the mean of the APs of all kinds. PASCAL VOC computes it with an IoU threshold of 0.5. COCO computes it at ten IoU thresholds, 0.5, 0.55 and so on up to 0.95, and takes their mean, which COCO simply calls AP. Both first take one more step, which they call interpolation: the precision at each point becomes the highest precision at that recall or a higher one, so the curve never goes up again; COCO then reads the curve at 101 recall values. This lesson leaves that step out, as scikit-learn does. With VOC's interpolation, the circles' AP in this lesson would be 0.65 instead of 0.56; the idea stays the same.

        The boxes in this lesson come from the course: for 8 scenes, the boxes that a detector of middling quality could draw. Most shapes get one box of their own kind, moved and stretched a little at random; now and then a shape gets no box, a box of another kind or a second, looser box, and a box lies on the background. The scores are mostly higher the better the box fits, with noise. A later lesson scores a real detector.
        """),
        L.watch_out("Divide by the number of true boxes, not by the number of true positives. A true box that no box finds counts as 0: otherwise a detector that draws one box, on the object it is surest of, gets an AP of 1."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **true positive**, of a detector | a box whose IoU with a true box of its kind in the same image is above the threshold, and whose true box no box with a higher score found first | `hits[k]` |
        | **average precision** (AP) | for one kind: the precision at each true positive, from the highest score down, added up and divided by the number of true boxes; the area under the precision-recall curve | `average_precision(hits, total)` |
        | **mean average precision** (mAP) | the mean of the APs of all kinds | `np.mean([0.56, 0.78, 0.73])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_scenes():
    mo.vstack([
        mo.md("**8 scenes, 29 true boxes, 32 boxes of the detector**"),
        mo.Html(C.scenes_picture({
            "image": "scene {i}",
            "alt": "Eight scenes of circles, squares and triangles in two rows of four, numbered 0 to 7. Each shape has a thin box around it, its true box. Over them lie the detector's 32 boxes: 23 true positives, dashed in blue, and 9 false positives, dotted in purple, among them a box on a square in scene 3 and boxes on the background. 6 shapes have no true positive.",
        })),
        mo.md(r"""
        The 8 scenes with their 29 true boxes, as thin lines, and the 32 boxes of the detector. At an IoU threshold of 0.5, 23 of them are true positives, dashed in blue, and 9 are false positives, dotted in purple. 6 shapes are not found. For the circles the detector draws 11 boxes, and 8 of the 11 circles get a true positive.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    What does the IoU threshold do? COCO does not stop at 0.5: it also asks for boxes that fit tightly. With the slider you set the IoU threshold to each of COCO's ten values, from 0.5 to 0.95. On the left, the precision-recall curve of each kind, with its AP; on the right, the mAP at every threshold, with a dotted line at 0.5.
    """)
    return


@app.cell(hide_code=True)
def explore_iou_cell():
    def _frame(t):
        _html, _m = C.iou_frame(t, {"curves": "precision-recall, IoU threshold {t}", "map": "mAP", "threshold": "IoU threshold",
                                    "precision": "precision", "recall": "recall", "circle": "circle", "square": "square",
                                    "triangle": "triangle", "point": ".",
                                    "alt": "Left, the precision-recall curves of circles, squares and triangles at an IoU threshold of {t}, as steps, with the AP of each kind. Right, the mAP at the ten IoU thresholds from 0.5 to 0.95: 0.69 at 0.5 and 0.55, then lower and lower, down to 0 from 0.9. A dotted line marks 0.5, and a larger point the threshold of {t}, with an mAP of {m}."})
        return _html, f"IoU threshold {L.number(t, 2)}: mAP {L.number(_m, 2)}."

    explore_iou = L.explore("iou", "IoU threshold", _frame, start=0.5, values=C.STOPS, role="hyper",
                            task="Find the largest IoU threshold at which the mAP is still at least 0.5.")
    explore_iou
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 0.5 and 0.55 the mAP is 0.69. Then it drops: 0.63 at 0.6, 0.60 at 0.65, 0.54 at 0.7, 0.44 at 0.75, 0.20 at 0.8 and 0.07 at 0.85, and from 0.9 on no box fits its own shape tightly enough: 0. So 0.7 is the largest IoU threshold with an mAP of at least 0.5. The circles drop first. Two of their true positives have an IoU of only 0.57 with their circle, so at 0.6 they become false positives, and the circles' AP falls from 0.56 to 0.38; at 0.7 it is 0.21. COCO's AP, the mean of the ten values, is 0.39 here, without interpolation. That is why COCO's AP of a detector is lower than its mAP at 0.5: it also rewards boxes that fit tightly.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To score a detector with mAP:

    1. For one kind, take its boxes from every image, the highest score first.
    2. Go down the list. A box is a true positive when its IoU with a true box of its kind in the same image is above the threshold, and no box before it found that true box; otherwise it is a false positive.
    3. After each box, precision is the true positives so far divided by the boxes so far, and recall the true positives so far divided by the true boxes of that kind.
    4. AP is the precision at each true positive, added up and divided by the number of true boxes of that kind.
    5. Do this for every kind. The mAP is the mean of their APs.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The worked example follows the recipe on 8 scenes, `scene(36)` to `scene(43)`. `detections(scenes, seed=17)` gives the course's detector boxes for them: for each box, the number of its scene in the list, its kind, its score and the box, the highest score first. The first cell prints the circle boxes.

    The second cell decides for each circle box whether it is a true positive. `iou` is the function of exercise 2 of [How much two boxes overlap: IoU](/course/open/cv-iou?lang=en). For each box, the inner loop looks for a circle in the same scene whose IoU with it is above 0.5: `truth` becomes that true box, with the number of its scene, or stays `None`. A box is a false positive when it finds no such true box, or when its true box is already in `taken`, found by a box with a higher score. Otherwise it is a true positive, and its true box goes into `taken`.

    The third cell works out precision, recall and AP with NumPy. `hits_array` is the list `hits` as a NumPy array. `np.cumsum(hits_array)` counts the true positives among the first boxes, as `True` counts as 1. Divided by 1, 2, 3 and so on, the boxes so far, that gives the precision after each box; divided by the number of true circles, the recall. `precision[hits_array]` keeps the precision at each true positive.
    """)
    return


@app.cell
def example_boxes():
    from everything_ai.scenes import scene, detections

    scenes = [scene(k) for k in range(36, 44)]
    # each box: (i, kind, score, box), with i the number of its scene;
    # the highest score first
    boxes = detections(scenes, seed=17)
    print(len(boxes), "boxes")
    kind = "circle"
    for d in boxes:
        if d[1] == kind:
            print(d)
    return boxes, kind, scenes


@app.cell
def example_hits(boxes, kind, scenes):
    def iou(a, b):
        # the intersection over union of two boxes, as in cv-iou
        top, left = max(a[0], b[0]), max(a[1], b[1])
        bottom, right = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0, bottom - top + 1) * max(0, right - left + 1)
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return overlap / (area_a + area_b - overlap)


    taken = set()     # the true boxes found so far: (i, box)
    # 2. for each box of this kind: True for a true positive
    hits = []
    for i, k, score, box in boxes:      # 1. the highest score first
        if k == kind:
            truth = None
            for shape in scenes[i].shapes:
                if shape.kind == kind:
                    if iou(box, shape.box) > 0.5:
                        truth = (i, shape.box)
            if truth == None:
                hits.append(False)          # no true box of this kind
            else:
                if truth in taken:
                    hits.append(False)      # its true box is found already
                else:
                    taken.add(truth)
                    hits.append(True)
    print(hits)
    return (hits,)


@app.cell
def example_ap(hits, kind, scenes):
    import numpy as np

    total = 0         # the true boxes of this kind
    # _shape, because the cell above already has a name shape
    for s in scenes:
        for _shape in s.shapes:
            if _shape.kind == kind:
                total = total + 1
    hits_array = np.array(hits)
    # 3. precision and recall after each box
    precision = np.cumsum(hits_array) / np.arange(1, len(hits) + 1)
    recall = np.cumsum(hits_array) / total
    # .round(2): every number of the array to two decimals
    print(precision.round(2))
    print(recall.round(2))
    # 4. the precision at each true positive, added up, divided by total
    ap = np.sum(precision[hits_array]) / total
    print(total, "true boxes, AP", ap.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first cell prints 11 circle boxes. The surest, with 0.86, lies in scene 3 on a square: its IoU with the square is 0.84, but a square is no circle. The second cell shows which boxes are true positives: the first is false, the next 8 are true, and the last two are false. The box with 0.33 lies mostly on the background, and the box with 0.30 does not fit its circle well enough. So 8 of the 11 circles are found. None of these boxes is a second box on a circle found already; such a box would be a false positive too, and NMS removes most of them.

    The third cell shows the precision climbing after the false first box: 0, then 0.5, 0.67 and so on up to 0.89 after the ninth box, and then down to 0.73 at the end. The recall climbs in steps of 1/11, as there are 11 circles, up to 0.73. AP adds up the precision at the 8 true positives, 0.5 + 0.67 + ... + 0.89, and divides by the 11 circles: 0.56. That the detector draws 11 circle boxes as well is chance: AP divides by the true boxes, not by the boxes. Change `"circle"` into `"square"` or `"triangle"` in the first cell, and the cells give 0.78 and 0.73. The mAP is the mean of the three: 0.69.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hand():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. By hand
    """)
    return


@app.cell(hide_code=True)
def choice_hand_cell():
    choice_hand = L.choice(
        "hand",
        question="A detector gives 5 boxes for cats in a test set with 4 cats. From the highest score down, they are: true positive, false positive, true positive, true positive, false positive. What is its AP?",
        options={
            "right": "0.60",
            "found": "0.81",
            "recall": "0.75",
            "every": "0.70",
        },
        correct="right",
        why={
            "right": "The precision at the three true positives is 1/1, 2/3 and 3/4. Added up that is 2.42, and divided by the 4 cats, 0.60. The fourth cat is never found and counts as 0; the false positive at the end costs nothing.",
            "found": "That divides by the 3 true positives. Divide by the 4 cats: the cat that is never found counts as 0.",
            "recall": "3 of the 4 cats are found: that is the recall at the end. AP also looks at the order: add up the precision at each true positive.",
            "every": "That is the mean of the precision after all 5 boxes. AP only adds up the precision at the true positives, and divides by the number of cats.",
        },
    )
    choice_hand
    return (choice_hand,)


@app.cell(hide_code=True)
def text_exercise_ap():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. AP as a function

        Complete `average_precision(hits, total)`. It gets a list `hits` with, for each box of one kind, from the highest score down, `True` for a true positive and `False` for a false positive, and `total`, the number of true boxes of that kind. It gives back the AP. The check calls it with the circles of the worked example:

        ```python
        average_precision(hits, 11)
        ```

        That should give 0.56, as in the worked example. The check also tries the other kinds, other IoU thresholds, a list without a true positive, and lists in which not every true box is found.
        """),
        L.task("Make `average_precision(hits, total)` give the average precision."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ap():
    ap_code = L.code_box("ap", lines=14)
    ap_code
    return (ap_code,)


@app.cell(hide_code=True)
def check_ap(ap_code):
    result_ap = L.check("ap", ap_code.value)
    result_ap
    return (result_ap,)


@app.cell(hide_code=True)
def hints_ap():
    L.hints(
        "ap",
        [
            "Go through the list with `for k in range(len(hits)):`, and keep two numbers: `found`, the true positives so far, and `sum_precision`, the precision added up so far. Both start at 0.",
            "At a true positive, `if hits[k]:`, first add 1 to `found`. The boxes so far are `k + 1`, as `k` starts at 0, so the precision is `found / (k + 1)`: add it to `sum_precision`.",
            "After the loop, divide by the number of true boxes: `return sum_precision / total`. Not by `found`: a true box that is never found counts as 0.",
        ],
        why="For the circles, the precision at the 8 true positives is 1/2, 2/3, 3/4, 4/5, 5/6, 6/7, 7/8 and 8/9. Added up that is 6.17, and divided by the 11 circles, 0.56.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bikes():
    mo.md(r"""
    ### Exercise 3. Many cars, few bicycles
    """)
    return


@app.cell(hide_code=True)
def choice_bikes_cell():
    choice_bikes = L.choice(
        "bikes",
        question="A test set holds 1,000 cars and 10 bicycles. A detector has an AP of 0.9 for cars and of 0.3 for bicycles. What is its mAP?",
        options={
            "right": "0.6",
            "weighted": "0.89",
            "cars": "0.9",
            "lowest": "0.3",
        },
        correct="right",
        why={
            "right": "The mAP is the mean of the APs of the kinds: (0.9 + 0.3) / 2 = 0.6. Every kind counts the same, however many objects it has, so a detector cannot hide a weak kind behind a common one.",
            "weighted": "That weights each kind by its number of objects. The mAP takes the plain mean over the kinds: each kind counts once.",
            "cars": "That is the AP for cars only. The mAP takes the mean over all kinds, bicycles too.",
            "lowest": "That is the lowest AP, not the mean. What is the mean of 0.9 and 0.3?",
        },
    )
    choice_bikes
    return (choice_bikes,)


@app.cell(hide_code=True)
def text_exercise_extra():
    mo.md(r"""
    ### Exercise 4. Extra boxes
    """)
    return


@app.cell(hide_code=True)
def explain_extra_cell():
    explain_extra = L.explain(
        "extra",
        question="A detector adds 50 boxes on empty background to what it draws, all with lower scores than every box it drew already. What happens to its AP, and why? And what do you still have to choose when an app shows its boxes?",
        model_answer="**Its AP stays the same.** AP goes down the boxes from the highest score and adds up the precision at each true positive. The 50 boxes come after every true positive, so they change none of those precisions, and the number of true boxes stays the same too. In the guess at the start, leaving out the circle box with 0.30 did not change the circles' AP: 0.56 either way. AP judges the order of the scores, not one place to cut the list. **An app still needs a score threshold**: it shows only the boxes with a higher score. A low threshold shows false boxes, a high one misses objects, and AP does not choose it for you.",
        points=[
            "you say that the AP stays the same, because the extra boxes come after every true positive and change no precision at a true positive",
            "you say that AP judges the order of the scores, not one place to cut the list of boxes",
            "you say that an app still needs a threshold on the score that decides which boxes to show, which weighs false boxes against missed objects",
        ],
    )
    explain_extra
    return (explain_extra,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [One box per object: non-maximum suppression](/course/open/cv-nms?lang=en), NMS kept one box per object. Here those boxes are scored against the true boxes. A box is a true positive when its IoU with a true box of its kind is above a threshold, and no box with a higher score found that true box first. Going down the boxes of one kind from the highest score, precision and recall trace the precision-recall curve of [Every threshold at once: the ROC curve and AUC](/course/open/ml-roc-auc?lang=en). The area under it, average precision, adds up the precision at each true positive and divides by the number of true boxes; mAP is the mean over the kinds: on the scenes 0.69 at VOC's IoU threshold of 0.5, and 0.39 averaged over COCO's ten thresholds, both without interpolation. AP judges the order of the scores, not where you cut the list. In the next lesson, *Use a pretrained detector*, a real detector draws boxes on photos, and a threshold on the score decides which of them you keep.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_bikes, choice_hand, explain_extra, result_ap):
    L.wrap_up(choice_hand, result_ap, choice_bikes, explain_extra)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
