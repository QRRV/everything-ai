# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-pretrained-detector. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-pretrained-detector", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Score a detector: mean average precision](/course/open/cv-map?lang=en) and about softmax, from [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_map():
    quick_map = L.choice(
        "quick-map",
        question="A detector has an AP of 0.8 for cats. At which score threshold should an app show its cat boxes?",
        options={
            "right": "AP does not say: it judges the order of the scores, not where to cut the list",
            "ap": "at 0.8, its AP",
            "iou": "at 0.5, the IoU threshold of PASCAL VOC",
            "rest": "at 0.2: 1 minus its AP",
        },
        correct="right",
        why={
            "right": "AP goes down all the boxes from the highest score and adds up the precision at each true positive: it judges the order of the scores, at every threshold at once. Where to cut the list, the app has to choose. This lesson is about that choice.",
            "ap": "AP is the area under the precision-recall curve, not a score of a box. It looks at every threshold at once, and does not pick one.",
            "iou": "The IoU threshold decides when a box fits a true box well enough. Which boxes an app shows is a threshold on another number: the score. AP does not choose it.",
            "rest": "AP is the area under the precision-recall curve, not a score of a box: 1 minus it says nothing about where to cut the list.",
        },
    )
    quick_map
    return (quick_map,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="A model gives three logits: 3 for cat, 2 for dog and 1 for \"nothing\". What probability does softmax give cat?",
        options={
            "right": "0.67",
            "two": "0.73",
            "share": "0.50",
            "max": "1.00",
        },
        correct="right",
        why={
            "right": "Softmax takes e to the power of each logit and divides by their sum: e³ / (e³ + e² + e¹) = 20.1 / 30.2 = 0.67. The detector of this lesson does the same for each of its boxes, with a kind for \"nothing\" among its kinds too.",
            "two": "That leaves the logit of \"nothing\" out: e³ / (e³ + e²) = 0.73. Softmax divides by the sum over all logits, that of \"nothing\" too.",
            "share": "That divides the logits themselves: 3 / (3 + 2 + 1). Softmax first takes e to the power of each logit.",
            "max": "Softmax gives every logit a share, the largest to the largest logit, but never all of it: dog and \"nothing\" keep a part.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A detector that someone trained on a large set of photos with boxes finds the objects in your photos too, with a few lines of code: a box around each object, a kind and a score. Self-driving cars, cameras that count visitors, and apps that sort your photos all start from such a detector. But it gives boxes with every score, from almost 0 to almost 1, and you have to decide which of them to believe. And it only knows the kinds it learned: what does it make of something it has never seen?

    **An everyday example.** A smoke alarm has a sensitivity. One that goes off at the faintest smell of smoke goes off at every slice of toast; one that waits for thick smoke misses a fire that starts small. Where you set it decides which mistake it makes more often. And when a fire gives off no smoke at all, only another gas, no setting helps.
    """)
    return


@app.cell(hide_code=True)
def predict_starfish():
    guess_starfish = L.guess(
        "how-many",
        question="The detector of this lesson learned 80 kinds of thing, chairs among them, but no starfish. You give it 30 photos of starfish, and keep only the boxes with a score above 0.9. On how many of the 30 photos does it still draw a box?",
        options={
            "none": "on none: it does not know starfish",
            "few": "on 2 or 3",
            "part": "on about 13",
            "all": "on all 30",
        },
        reveal="**On 13 of the 30.** A starfish is none of its kinds, so every box on these photos is wrong. Yet on 13 photos it draws a box with a score above 0.9: a snowboard with 0.999, a teddy bear and a bird with 0.996. A detector only knows its own kinds. On something else, it gives the kind it knows that looks most alike, and a high score does not make that right.",
    )
    guess_starfish
    return (guess_starfish,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        This lesson uses **DETR** (detection transformer), a detector that Facebook AI Research, now Meta, published in 2020 with the paper by Carion and others. It learned from COCO, a set of 118,000 photos with a box around every object of 80 kinds: cars, cats, cups, chairs, people and so on. On the 5,000 photos COCO keeps apart for checking detectors, its AP averaged over the ten IoU thresholds, as in [Score a detector](/course/open/cv-map?lang=en), is 0.42, with COCO's interpolation. The course wrote DETR out in its own code, `everything_ai/detr.py`, with the 41.5 million parameters Meta published.

        DETR works in three parts. A ResNet-50, a larger version of the ResNet-18 of [Try an image classifier](/course/open/cv-try-classifier?lang=en), turns the photo into features, one row of numbers for each square of 32 by 32 pixels. A transformer, which later lessons explain, lets 100 **object queries** look at those features. Each query is a place for one object: it gives one box and 92 logits. The logits are for the numbers 0 to 90, which COCO gives its kinds, and the last for **no object**. COCO's 80 kinds have numbers from 1 to 90, with 10 numbers left unused, and 0 is not used either: DETR never saw a box for those 11 numbers, so it never learned to say them.

        For one query, softmax turns the 92 logits into probabilities. Its kind is the most likely of the first 91, and its **score** is that probability: the last probability, that of *no object*, takes no part. On the coffee photo, three queries have a high score; the other 97 give most of their probability to *no object*, and their score is low. The box is four numbers: `(cx, cy, w, h)`, the center across and down and the width and height, each as a fraction of the photo's width or height. Exercise 2 turns it into pixels.

        Which boxes do you keep? Those with a score above a **score threshold**. A low threshold keeps boxes the detector is not sure of: it misses fewer objects, and shows more false boxes. A high threshold shows fewer false boxes, and misses more objects. As in [Score a detector](/course/open/cv-map?lang=en), the AP does not choose it: you choose it for what each mistake costs in your app.

        Two more things are new. DETR needs no non-maximum suppression. While it trained, each true box was matched with exactly one query, and every other query had to learn to say *no object*; so each object mostly gets one box, where the detector of [One box per object](/course/open/cv-nms?lang=en) drew many. And DETR finds only its own kinds. On a photo of something else, it gives the kind it knows that looks most alike: no score threshold makes it find a kind it never learned.
        """),
        L.watch_out("Leave out the last probability, that of *no object*, when you take the highest. With it, 97 of the 100 queries on the coffee photo pick *no object*, each with more than 0.7: above a threshold of 0.5 you would keep 97 boxes of nothing."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **object query** | one of DETR's 100 places for an object: each gives one box and the logits of its kind | `logits[0, i]`, `boxes[0, i]` |
        | **no object** | DETR's last kind: what a query says when it holds no object | `p[-1]` |
        | **score threshold** | the score a box has to be above to be kept; a higher one shows fewer false boxes and misses more objects | `threshold = 0.9` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_course():
    mo.vstack([
        mo.md("**DETR on the course's photos, every box above 0.5**"),
        mo.Html(C.course_picture({
            "titles": ["coffee", "Chelsea", "fruit (a drawing)", "a cell"],
            "alt": "The course's four photos with DETR's boxes above a score of 0.5, each with its kind and score. The coffee photo: cup 1.00, dining table 0.99 and spoon 0.98. Chelsea: cat 1.00, and carrot 0.86 on her orange nose. The drawing of fruit: seven boxes of sports ball, one on each piece of fruit. The cell under the microscope: clock 1.00.",
        }, "en")),
        mo.md(r"""
        DETR on the course's four photos, with every box above a score of 0.5; the labels round the scores to two decimals. On the coffee photo: a cup with 0.997, the table with 0.993 and a spoon with 0.98. On Chelsea: the cat, with 0.996, and a carrot, with 0.86, on her orange nose. The 7 pieces of fruit of the drawing become 7 sports balls, although COCO has apples and oranges: drawn, round and shiny, they look more like balls to DETR. The cell under the microscope becomes a clock, with 0.997. DETR knows 80 kinds of thing, and whatever it sees, it calls one of them.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    What does the score threshold do? The course gave DETR 30 photos of Windsor chairs and 30 photos of starfish, from Caltech-101, a collection of photos of 101 kinds of thing. DETR knows chairs; it does not know starfish, so every box on a starfish photo is wrong. With the slider you set the score threshold. The solid line counts the chair photos on which DETR draws a box of the kind chair, the dashed line the starfish photos on which it draws any box.
    """)
    return


@app.cell(hide_code=True)
def explore_threshold_cell():
    def _frame(t):
        _svg, _found, _star = C.threshold_frame(t, {"x": "score threshold", "y": "photos, of 30",
                                                     "chairs": "chair photos with a chair box",
                                                     "starfish": "starfish photos with a box",
                                                     "alt": "Two lines against the score threshold, from 0.05 to 0.99. The solid line, the chair photos with a box of the kind chair, falls slowly from 27 to 12; the dashed line, the starfish photos with any box, falls from 30 to 4. A dashed upright line marks the threshold of {t}: {found} chair photos with a chair box, and {starfish} starfish photos with a box."}, "en")
        return _svg, f"Score threshold {L.number(t, 2)}: a chair box on {_found} of the 30 chair photos, and a box on {_star} of the 30 starfish photos."

    explore_threshold = L.explore("threshold", "score threshold", _frame, start=0.5, values=C.STOPS, role="hyper",
                                  task="Find the lowest score threshold at which at most 10 of the 30 starfish photos get a box.")
    explore_threshold
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 0.05, DETR finds a chair on 27 of the 30 chair photos, and draws a box on all 30 starfish photos. As the threshold rises, both fall: 23 and 23 at 0.5, 20 and 13 at 0.9, and 19 and 9 at 0.95. So 0.95 is the lowest threshold at which at most 10 starfish photos get a box. Counted in boxes, the starfish photos get 39 boxes at 0.5 and 14 at 0.9, and every one of them is wrong. At 0.99, DETR still draws a box on 4 starfish photos, a snowboard with 0.999, a teddy bear and two birds, while it finds a chair on only 12 of the chair photos. And 3 chair photos get no chair box even at 0.05. A threshold trades missed chairs against false boxes; it cannot make DETR find a kind it never learned.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To find the objects in a photo with DETR, `detr`:

    1. Prepare the photo: `x = prepare(photo)`, of shape `(1, 3, h, w)`.
    2. The queries: `logits, boxes = detr(x)`, inside `with torch.no_grad():`; for each of the 100 queries, 92 logits and a box.
    3. For query `i`, softmax turns its logits into probabilities `p`, as in [Try an image classifier](/course/open/cv-try-classifier?lang=en).
    4. Its kind is `k = np.argmax(p[:-1])`, leaving out the last, *no object*, and its score is `p[k]`.
    5. Keep the query when its score is above the threshold, `p[k] > threshold`. Its box is `boxes[0, i]`, which `to_pixels` of exercise 2 turns into pixels.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell prints how many kinds DETR has, the one with number 47, and the last. It prepares the coffee photo, step 1, and prints its shape. `prepare` from the course's `everything_ai.detr` makes the shortest side 800 pixels, the size at which DETR is tested on COCO, keeps the longest side at most 1,333, and normalizes with ImageNet's mean and std, as in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en). It does not crop: a detector looks for objects in the whole photo. Then the cell gives the photo to DETR, step 2, and prints the shapes of the logits and the boxes. For every query, it takes steps 3 to 5 with a threshold of 0.9, and prints the number of each query it keeps, its kind, its score and its box. DETR takes a second or more for a photo. `detr` is DETR, which the cell below loads once it is downloaded, together with the photos of Caltech-101 for the demo.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download DETR and Caltech-101, 304 MB")
    return (get_button,)


@app.cell(hide_code=True)
def detr_cell(get_button):
    detr = C.detr_or_stop(get_button, {
        "ask": "*DETR and the photos of Caltech-101 are not on your computer yet. The button downloads them once: DETR from Hugging Face, 167 MB, and Caltech-101 from Caltech, 137 MB, whose photos it then unpacks. The worked example and the demo need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 304 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*DETR and the photos are on your computer: 41.5 million parameters, trained on COCO's 118,000 photos.*",
    })
    return (detr,)


@app.cell
def example_coffee(detr):
    import numpy as np
    import torch
    from PIL import Image
    from everything_ai.detr import KINDS, prepare

    print(len(KINDS), KINDS[47], KINDS[-1])
    # 1. the shortest side 800 pixels, normalized as for ImageNet
    photo = Image.open("data/photos/coffee.png")
    x = prepare(photo)
    print(x.shape)
    # 2. for each of the 100 queries: 92 logits and a box
    with torch.no_grad():
        logits, boxes = detr(x)
    print(logits.shape, boxes.shape)
    threshold = 0.9
    for i in range(100):
        # 3. softmax over the 92 logits of query i
        z = logits[0, i].numpy()
        e = np.exp(z - z.max())
        p = e / e.sum()
        # 4. the kind and its score, without the last: no object
        k = np.argmax(p[:-1])
        # 5. keep the query when its score is above the threshold
        if p[k] > threshold:
            print(i, KINDS[k], p[k].round(3), boxes[0, i].numpy().round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    DETR has 92 kinds: number 47 is cup, and the last is *no object*. The prepared photo has shape `(1, 3, 800, 1200)`: the photo is 600 by 400 pixels, so its shortest side doubles to 800, and the longest side with it. The logits have shape `(1, 100, 92)`, and the boxes `(1, 100, 4)`: a batch of one photo, 100 queries. At 0.9, three queries keep their box: query 59 a dining table, with 0.993, query 71 the cup, with 0.997, and query 92 a spoon, with 0.98. The cup's box is `[0.486 0.381 0.397 0.674]`: its center lies at 0.486 of the width and at 0.381 of the height, and it is 0.397 of the photo wide and 0.674 high. The other 97 queries give *no object* the highest probability. One of them still gives a second box, on the bowl of the spoon, 0.293: below the threshold. Above 0.9, no two boxes lie on the same object, and that without NMS.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_score():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The score of a query
    """)
    return


@app.cell(hide_code=True)
def choice_score_cell():
    choice_score = L.choice(
        "score",
        question="After softmax, a query of DETR gives cat 0.30, dog 0.25 and *no object* 0.40; the other kinds share the remaining 0.05. What happens to its box at a score threshold of 0.2?",
        options={
            "right": "It is kept, as a cat with a score of 0.30",
            "nothing": "It is dropped: *no object* is the most likely, with 0.40",
            "kinds": "It is kept, as a cat with a score of 0.60: the kinds together",
            "both": "It is kept twice, as a cat and as a dog: both are above 0.2",
        },
        correct="right",
        why={
            "right": "The score is the highest probability among the kinds before the last: cat, with 0.30, and that is above 0.2. That *no object* has more, 0.40, does not count. So a low threshold keeps boxes the query itself is not sure hold an object; at 0.5, this one would be dropped.",
            "nothing": "The score leaves out the last probability, that of *no object*: it is the highest probability of a kind. Which kind is that, and is it above 0.2?",
            "kinds": "The score is the probability of one kind, the most likely one, not of all kinds together. Which is the most likely kind, and with what probability?",
            "both": "A query gives one box, with one kind: the most likely one. Its score is the probability of that kind.",
        },
    )
    choice_score
    return (choice_score,)


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. From fractions to pixels

        DETR gives a box as `(cx, cy, w, h)`, each a fraction of the photo; the course writes a box as `(top, left, bottom, right)` in pixels, as in [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en). There, bottom and right were the last row and column of the box. DETR measures edges instead: here, bottom and right are the lines where the box ends, measured from the photo's top-left corner, so a box over a whole photo 600 pixels wide ends at 600. Complete `to_pixels(box, width, height)`, which gets DETR's box and the width and height of the photo, and gives back the box in pixels. Replace the four `...`. The check calls it with the cup of the worked example, on the coffee photo of 600 by 400 pixels:

        ```python
        to_pixels((0.486, 0.381, 0.397, 0.674), 600, 400)
        ```

        That should give `(17.60, 172.50, 287.20, 410.70)`. The check also tries other boxes on other photos.
        """),
        L.task("Make `to_pixels(box, width, height)` give the box as `(top, left, bottom, right)` in pixels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pixels():
    pixels_code = L.code_box("pixels", lines=8)
    pixels_code
    return (pixels_code,)


@app.cell(hide_code=True)
def check_pixels(pixels_code):
    result_pixels = L.check("pixels", pixels_code.value)
    result_pixels
    return (result_pixels,)


@app.cell(hide_code=True)
def hints_pixels():
    L.hints(
        "pixels",
        [
            "`cx` is the center across, as a fraction of the width. From there, the box reaches half its width to the left and half to the right. The top and the bottom go the same way, with `cy` and `h`.",
            "As a fraction, the left side is `cx - w / 2`. A fraction becomes pixels when you multiply it by the size of the photo in that direction: what runs across, the left and the right side, by `width`; what runs down, the top and the bottom, by `height`.",
            "`top = (cy - h / 2) * ___`, `left = (cx - w / 2) * ___`, and `bottom` and `right` the same with `+`.",
        ],
        why="For the cup, the center lies at 0.381 of the height and the box is 0.674 high, so it runs from 0.381 - 0.337 = 0.044 to 0.718 of the height: from row 17.6 to row 287.2 of 400. Across, it runs from 0.486 - 0.1985 = 0.2875 to 0.6845 of the width: from column 172.5 to column 410.7 of 600.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your function on new photos

    Once DETR and the photos are downloaded, in the worked example, the cell below gives DETR 8 photos from Caltech-101: 4 of the Windsor chairs and 4 of the starfish of the slider, with the number each has among the 30. It draws every box above the score threshold you set, with your `to_pixels` once it is right, else with the course's. The 8 photos take a few seconds, once.
    """)
    return


@app.cell(hide_code=True)
def demo_slider_cell():
    demo_slider = L.slider("demo", "score threshold", start=0.9, low=0.3, high=0.95, step=0.05, role="hyper")
    demo_slider
    return (demo_slider,)


@app.cell(hide_code=True)
def demo_boxes(demo_slider, detr, pixels_code, result_pixels):
    _t = demo_slider.widget.value
    _d = C.demo(detr, pixels_code.value if result_pixels.state == "correct" else None, _t, {
        "titles": ["chair 0", "chair 2", "chair 13", "chair 20", "starfish 1", "starfish 4", "starfish 12", "starfish 21"],
        "alt": f"Eight photos from Caltech-101 in two rows: four Windsor chairs, numbered 0, 2, 13 and 20, and four starfish, numbered 1, 4, 12 and 21, with every box of DETR above a score threshold of {L.number(_t, 2)}, each with its kind and score. DETR has no kind starfish, so every box on a starfish photo is wrong.",
    }, "en")
    _who = "your `to_pixels`" if _d["yours"] else "the course's `to_pixels`"
    mo.vstack([
        mo.Html(_d["picture"]),
        mo.md(f"Drawn with {_who}. At a score threshold of {L.number(_t, 2)}, {_d['chairs']} of the 4 chair photos get a chair box, and {_d['starfish']} of the 4 starfish photos get a box."),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Move the threshold down to 0.3, and more boxes appear: airplanes and a chair in the sand of starfish 1, a teddy bear and a suitcase on starfish 12, and three more cats on starfish 21. On chair 13, DETR draws three chair boxes on the same chair, with 0.88, 0.81 and 0.36; the first two overlap with an IoU of 0.83. Matching one query with each object while training makes such doubles rare, not impossible; NMS would remove them. On chair 0, the dark seat is also a cat, with 0.98: a false box that no threshold up to 0.95 removes. Chair 20 gets no box at all: it is one of the 3 chair photos on which DETR never finds a chair. And the drawn starfish 4 is a snowboard with 0.999, at every threshold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_apps():
    mo.md(r"""
    ### Exercise 3. Which threshold where
    """)
    return


@app.cell(hide_code=True)
def choice_apps_cell():
    choice_apps = L.choice(
        "apps",
        question="Four apps use the same detector. Which of them should set its score threshold low?",
        options={
            "right": "A program that marks cracks on photos of a bridge, for an inspector who checks each mark",
            "album": "An app that writes what it sees under your holiday photos",
            "shop": "A self-checkout camera that puts every product it sees on your bill",
            "fine": "A camera in a bus lane that sends a fine for every car it sees there",
        },
        correct="right",
        why={
            "right": "A missed crack can be dangerous, and a false mark costs the inspector one look. A low threshold misses fewer cracks and shows more false marks, and here that is the right trade.",
            "album": "A wrong word under a photo is the mistake that bothers people here, and a missed object costs little: that asks for a high threshold. Which app loses the most by missing something?",
            "shop": "A false box costs the customer money: that asks for a high threshold. Which app loses the most by missing something, while a false box costs little?",
            "fine": "A false box is a fine for someone who did nothing wrong: that asks for a high threshold. Which app loses the most by missing something, while a false box costs little?",
        },
    )
    choice_apps
    return (choice_apps,)


@app.cell(hide_code=True)
def text_exercise_starfish():
    mo.md(r"""
    ### Exercise 4. Counting starfish
    """)
    return


@app.cell(hide_code=True)
def explain_starfish_cell():
    explain_starfish = L.explain(
        "starfish",
        question="Someone builds an app that counts the starfish on photos of rock pools. They take DETR, and set a very high score threshold, 0.99, so that the boxes they keep are right. Why does that not work, and what would they need instead?",
        model_answer="**DETR has no kind starfish.** It learned only COCO's 80 kinds, so every box it draws on a starfish carries the name of another kind, and some of those boxes have very high scores: at 0.99, DETR still draws a box on 4 of the 30 starfish photos, a snowboard with 0.999 and a teddy bear with 0.996. A high threshold only drops boxes DETR is less sure of; it cannot make DETR find a kind it never learned, and a box it is sure of can still be wrong. **What would work** is a detector that learned starfish: one trained on photos of starfish with a box around each, for instance DETR trained further on such photos, as later lessons do.",
        points=[
            "you say that DETR learned only COCO's kinds, and starfish is none of them",
            "you say that DETR still gives starfish high scores for wrong kinds, so a high threshold does not remove every false box, and no threshold makes it find starfish",
            "you say that the app needs a detector trained on photos of starfish with their boxes",
        ],
    )
    explain_starfish
    return (explain_starfish,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Score a detector: mean average precision](/course/open/cv-map?lang=en), AP judged the order of a detector's scores and left the threshold to the app. Here a real detector, DETR, trained on COCO, draws boxes on photos: a ResNet-50, as in [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en) but larger, makes features, and 100 object queries each give a box and the logits of 92 kinds, the last for *no object*. Softmax, as for a classifier, gives each query its kind and score, leaving out *no object*; as each object mostly gets one query, DETR needs no NMS. A score threshold decides which boxes you keep: from a chair on 27 of 30 chair photos and a false box on all 30 starfish photos at 0.05, to 12 and 4 at 0.99. And DETR only knows its own kinds: a starfish becomes a snowboard with 0.999. With this lesson, the first part of computer vision is done, from the pixels in a file to a detector that finds objects in photos. Training a detector of your own, and how the transformer inside DETR works, come in later lessons.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_apps, choice_score, explain_starfish, result_pixels):
    L.wrap_up(choice_score, result_pixels, choice_apps, explain_starfish)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
