# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-sam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-sam", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) and [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fcn():
    quick_fcn = L.choice(
        "quick-fcn",
        question="The FCN of the lesson on FCNs learned the course's scenes, with 4 kinds: background, circle, square and triangle. You give it a photo of a starfish. What does it give the pixels of the starfish?",
        options={
            "right": "one of its 4 kinds each: it has no other",
            "star": "a new kind, starfish: it sees a new shape",
            "empty": "no kind: those pixels stay empty",
            "background": "always background: what it does not know is background",
        },
        correct="right",
        why={
            "right": "Its last layer gives every pixel 4 scores, one for each kind, and the pixel gets the kind with the highest. A starfish is none of them, but every pixel still gets one of the 4. SAM, the model of this lesson, has no kinds at all: it draws the mask of whatever you point at.",
            "star": "An FCN can only give a kind it has a score for, and it has 4. A starfish is none of them; which kind does a pixel get then?",
            "empty": "Every pixel gets the kind with the highest of its 4 scores, and one of them is always the highest.",
            "background": "Background is one of its 4 kinds, and a pixel gets it only where its score is the highest. Nothing makes the pixels of a starfish go to background: they get whichever of the 4 kinds scores highest.",
        },
    )
    quick_fcn
    return (quick_fcn,)


@app.cell(hide_code=True)
def quick_check_erosion():
    quick_erosion = L.choice(
        "quick-erosion",
        question="Erosion with a square of 3 by 3 makes a white shape one pixel thinner all around. You erode a white square of 9 by 9 pixels, in the middle of a black image, again and again. What is left just before nothing is left?",
        options={
            "right": "the one pixel in its middle, after 4 erosions",
            "three": "a square of 3 by 3, after 3 erosions",
            "corners": "its four corners",
            "never": "the square never disappears: it keeps its shape",
        },
        correct="right",
        why={
            "right": "Every erosion takes a pixel off each side: 9 by 9 becomes 7, 5, 3 and then 1 by 1, and the fifth erosion leaves nothing. The last pixel left is the one deepest inside the shape, farthest from its edges. This lesson clicks there.",
            "three": "After 3 erosions, a square of 3 by 3 is left, yes. But that is not yet the last: one more erosion leaves its middle pixel.",
            "corners": "Erosion keeps a pixel only where its whole window of 3 by 3 is white. A corner has black pixels in its window, so the corners go first.",
            "never": "Every erosion makes the square one pixel smaller on each side: 9, 7, 5, 3, 1. What is left after 4 erosions?",
        },
    )
    quick_erosion
    return (quick_erosion,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Drawing a mask by hand is slow: in [Labels, boxes, masks and keypoints](/course/open/cv-annotations?lang=en), COCO's annotators took more than 79 seconds per object. An FCN, as in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en), draws masks by itself, but only of the kinds it learned, and only after someone drew thousands of masks for it to learn from. SAM, the Segment Anything Model of Meta, turns this around: you point at a thing with a click, and it draws the mask of that thing, of any kind, in a fraction of a second. Where the mask is wrong, you click again. Meta made its own set of masks this way. With SAM's help, an annotator drew a mask in 14 seconds, 6.5 times faster than for COCO, and at the end SAM drew 99.1% of the 1.1 billion masks by itself.

    **An everyday example.** In a shop, you point at a shelf and say "that one, please". The assistant may hand you the cup, the cup with its saucer, or the whole set. Pointing is quick, but it can mean more than one thing; a second gesture, "no, the whole set", settles it.
    """)
    return


@app.cell(hide_code=True)
def predict_sunflower():
    guess_sunflower = L.guess(
        "sunflower",
        question="You click once in the middle of a sunflower, on its seeds, and keep the mask SAM gives the highest score. On how many of 30 photos of sunflowers is that the whole flower, with an IoU of at least 0.8 with its outline?",
        options={
            "none": "on none or almost none",
            "some": "on about a third",
            "most": "on about two thirds",
            "all": "on all or almost all",
        },
        reveal="**On 1 of the 30.** The deepest pixel of a sunflower lies among its seeds, and a click there can mean the seeds or the whole flower. On 25 of the 30 photos, one of SAM's three masks is the whole flower, with an IoU of at least 0.8, but SAM gives the seeds a higher score. One click cannot tell SAM which you meant. A second click, where the mask misses the flower, can: with the course's second click, the median IoU of the 30 photos goes from 0.18 to 0.91.",
    )
    guess_sunflower
    return (guess_sunflower,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        This lesson uses **SAM 2.1**, the Segment Anything Model 2.1, which Meta published in 2024 with the paper by Ravi and others; the first SAM, by Kirillov and others, came out in 2023. SAM does **promptable segmentation**: you say with a **prompt** what you mean, and it gives the mask of that. A prompt is one or more clicks, each with a label: 1 for a **positive click**, on the thing, and 0 for a **negative click**, on something that is not part of it. A box works too, given by its top-left and bottom-right corners. SAM has no kinds. The FCN of [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) gives every pixel one of the kinds it learned; SAM draws the mask of whatever you point at, and does not say what it is. It learned that from SA-1B, 11 million photos with 1.1 billion masks, and from videos. The course uses its smallest version, SAM 2.1 tiny, with about 39 million parameters. The course's own code, `everything_ai/sam2.py`, writes out the 31.3 million of them that work on photos, and leaves out the memory with which SAM 2.1 follows a thing through a video.

        SAM works in three parts. The image encoder, the largest part, makes the photo 1024 by 1024 pixels and turns it into features: 256 numbers for each square of 16 by 16 pixels, and two finer maps for the edges. It runs once per photo, in about a second. The prompt encoder turns every click into 256 numbers, from where it is and its label. The mask decoder, a small network, combines those with the features, and gives four masks of 256 by 256 logits, and for each mask its **predicted IoU**: the IoU it expects that mask to have with what you meant. That is the mask's score. The decoder takes a few hundredths of a second, so you can click again and again on one photo, without running the image encoder again.

        Why four masks? One click can mean more than one thing: a click on a petal can mean the petal or the whole flower. The paper of the first SAM found that three masks cover most cases: the whole, a part and a part of that. So masks 1 to 3 are three such guesses, in no fixed order: for the click on the spoon below, mask 1 is the largest and mask 2 the smallest. Mask 0 is for a prompt that says exactly what it means, such as several clicks or a box. With one click, the course takes the mask with the highest score among masks 1 to 3, as SAM's makers advise; with more clicks, mask 0. With more clicks, SAM's own code still takes the best of masks 1 to 3 when the edge of mask 0 is unsure; the course leaves that step out. A pixel belongs to the mask where its logit is above 0: where sigmoid makes it a probability above 0.5.

        And when the mask is wrong? Then you click again, where it is most wrong: with label 1 on a part it missed, or with label 0 on a part it has too many. SAM gets all the clicks, the earlier ones too, and gives a new mask. A good place for a click is far from the edges of what is wrong: its **deepest pixel**, the pixel that is left last when you erode it again and again, as in [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en). With the outlines that came with Caltech-101, the course plays the person who clicks: the first click in the deepest pixel of the object, and each next click in the deepest pixel of what the mask gets most wrong.
        """),
        L.watch_out("Give SAM all the clicks every time, not only the newest. It keeps no memory of earlier clicks: one new click on its own is a new prompt, and gives the mask of only that click."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **promptable segmentation** | segmentation in which you say with a prompt what to segment, and the model gives its mask | `sam.decode(features, points, labels)` |
        | **prompt** | what tells SAM what you mean: clicks, each with its label, or a box | `points`, `labels` |
        | **positive click**, **negative click** | a click on the thing, with label 1; a click on what is not part of it, with label 0 | `labels = torch.tensor([[1, 0]])` |
        | **predicted IoU** | the score SAM gives a mask: the IoU it expects that mask to have with what you meant | `scores[0, k]` |
        | **deepest pixel** | the pixel of a region that is left last when you erode it, far from its edges | `deepest(region)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_spoon():
    mo.vstack([
        mo.md("**One click on the bowl of the spoon: SAM's three masks**"),
        mo.Html(C.spoon_picture({
            "title": "mask {k}: {score}",
            "alt": "Three times the part of the coffee photo with the spoon, in gray, with a dot where the click is, on the bowl of the spoon, and one of SAM's masks over it. Mask 1, with a score of 0.645, is the bowl with the end of the handle, at the top. Mask 2, with 0.059, is a small highlight on the bowl. Mask 3, with 0.941, is the bowl.",
        }, "en")),
        mo.md(r"""
        One click on the bowl of the spoon of the coffee photo, at (360, 277), and the three masks SAM gives for it, with their scores. The photo is gray here, so that the masks stand out. Mask 1, with 0.645, is the spoon with the end of its handle, at the top, which shows above the cup; the thin middle of the handle, along the cup, it leaves out. Mask 2, with 0.059, is a highlight on the bowl, 228 pixels. Mask 3, with 0.941, is the bowl. SAM expects the bowl to fit best what you meant, and ranks it first. Mask 0, not drawn here, is the bowl too, with 0.901. On the handle of the cup, at (217, 262), mask 3 is that handle, with 0.894, and mask 1 the whole cup, with only 0.122.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How many clicks does SAM need? The course gave SAM 30 photos of each of 10 kinds of thing from Caltech-101, and took the outline that came with each photo as what you meant. It clicked in the deepest pixel of the object, chose the mask as above, and then clicked four more times, each time where the mask was most wrong. With the slider you set the number of clicks. For each kind, the open dot is the median IoU after one click, the filled dot after the clicks you set; the short upright line is the median IoU when SAM gets a box around the outline instead of clicks.
    """)
    return


@app.cell(hide_code=True)
def explore_clicks_cell():
    def _frame(k):
        _svg = C.clicks_frame(int(k), {"x": "median IoU with the outline, over 30 photos",
                                       "alt": "For each of 10 kinds of thing from Caltech-101 a row with the median IoU of SAM's mask with the outline after one click, an open dot, and after the number of clicks set with the slider, {k}, a filled dot, with a line between them, and a short upright line for a box. With one click, only the starfish reach a median IoU of 0.85; the sunflowers start at 0.18 and the chandeliers at 0.08. With two clicks, 8 of the 10 kinds reach 0.85. The chandeliers and the Windsor chairs never do."}, "en")
        return _svg, C.clicks_line(int(k), "en")

    explore_clicks = L.explore("clicks", "clicks", _frame, start=5, low=1, high=5, step=1, role="hyper",
                               task="Find the fewest clicks after which 8 of the 10 kinds have a median IoU of 0.85 or more.")
    explore_clicks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    With one click, only the starfish reach a median IoU of 0.85: 0.919. The sunflowers start at 0.18, and the nautiluses at 0.535: one click in their middle, on the seeds or the inner turns of the shell, can mean a part, and SAM ranks the part first. After 2 clicks, 8 of the 10 kinds reach 0.85, the sunflowers 0.913. More clicks help less and less, and not always: the median of the ewers falls from 0.914 after 3 clicks to 0.854 after 4, and that of the Windsor chairs from 0.832 after 4 to 0.754 after 5. Chandeliers and Windsor chairs never reach 0.85, with 0.678 and 0.832 at best. Their outlines go around the arms and the spindles, gaps and all, so a next click can land in a gap, where the photo shows what is behind them. A box is no cure for that either: for the Windsor chairs, its median IoU is 0.678, below what two clicks give. For the chandeliers, a box does better than five clicks, 0.761, and for the other eight kinds about as well as two to five clicks, from 0.856 to 0.947.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To segment what you point at in a photo with SAM 2.1, `sam`:

    1. Prepare the photo, `image = prepare(photo)`, and compute its features once: `features = sam.features(image)`, inside `with torch.no_grad():`.
    2. Turn every click `(x, y)` in pixels of the photo into a point on the photo of 1024 by 1024: `x * 1024 / width` and `y * 1024 / height`. Give each its label: 1 on the thing, 0 on what is not part of it.
    3. The masks: `logits, scores = sam.decode(features, points, labels)`, four masks of 256 by 256 logits, and their scores.
    4. Make the masks as large as the photo, `to_photo(logits[0], width, height)`, and choose one: with one click, the highest score among masks 1 to 3; with more, mask 0. A pixel is in it where its logit is above 0. Exercise 1 does this.
    5. Is the mask wrong? Add a click where it is most wrong, as exercise 3 does, and go back to step 2 with all the clicks. The features stay as they are.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell prepares the coffee photo, step 1, and prints its shape and that of the coarsest map of its features. `prepare` from the course's `everything_ai.sam2` makes the photo 1024 by 1024 pixels, whatever its shape, and normalizes it with ImageNet's mean and std, as in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en). Then it clicks on the bowl of the spoon, step 2, and prints the point on the photo of 1024 by 1024. It gives that to SAM, step 3, and prints the shape of the logits and the four scores. Last, the first half of step 4: it makes the four masks as large as the photo with `to_photo`, and prints how many pixels each has. The choice is exercise 1. `sam` is SAM 2.1, which the cell below loads once it is downloaded, together with the photos of Caltech-101 and their outlines, for the demo.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SAM 2.1 and Caltech-101, 293 MB")
    return (get_button,)


@app.cell(hide_code=True)
def sam_cell(get_button):
    sam = C.sam_or_stop(get_button, {
        "ask": "*SAM 2.1 and the photos of Caltech-101 are not on your computer yet. The button downloads them once: SAM 2.1 from Hugging Face, 156 MB, and Caltech-101 from Caltech, 137 MB, whose photos and outlines it then unpacks. The worked example and the demo need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 293 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*SAM 2.1 and the photos are on your computer: the 31.3 million parameters of SAM 2.1 that work on photos, trained on the 1.1 billion masks of SA-1B and on videos.*",
    })
    return (sam,)


@app.cell
def example_spoon(sam):
    import torch
    from PIL import Image
    from everything_ai.sam2 import prepare, to_photo

    photo = Image.open("data/photos/coffee.png")
    width, height = photo.size
    # 1. the features of the photo, once: SAM makes it 1024 by 1024 pixels
    image = prepare(photo)
    with torch.no_grad():
        features = sam.features(image)
    print(image.shape, features[0].shape)
    # 2. one click on the bowl of the spoon, (360, 277), as a point on the photo of 1024 by 1024, with label 1
    points = torch.tensor([[[360 * 1024 / width, 277 * 1024 / height]]])
    labels = torch.tensor([[1]])
    print(points)
    # 3. four masks of 256 by 256 logits, and the IoU SAM expects of each
    with torch.no_grad():
        logits, scores = sam.decode(features, points, labels)
    print(logits.shape, scores[0].numpy().round(3))
    # 4. the masks as large as the photo: a pixel is in a mask where its logit is above 0
    logits = to_photo(logits[0], width, height)
    print(logits.shape, (logits > 0).sum(axis=(1, 2)))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The prepared photo has shape `(1, 3, 1024, 1024)`: one photo, three colors, 1024 by 1024 pixels, although the coffee photo is 600 by 400. SAM stretches it, and `to_photo` stretches its masks back at the end. The coarsest map of the features has shape `(1, 256, 64, 64)`: 256 numbers for each of 64 by 64 squares of 16 by 16 pixels. The click (360, 277) becomes the point (614.4, 709.1) on the photo of 1024 by 1024. The logits have shape `(1, 4, 256, 256)`: four masks of 256 by 256, a quarter of 1024 each way, and the scores are 0.901, 0.645, 0.059 and 0.941. As large as the photo, the masks have shape `(4, 400, 600)`, and their pixels are `[5516 6762  228 5575]`: mask 1, the spoon with the end of its handle, is the largest, and mask 2, the highlight, the smallest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Choose the mask

        Complete `choose(logits, scores, clicks)`, step 4 of the recipe: `logits` holds SAM's four masks of logits, as large as the photo, with shape `(4, height, width)`; `scores` their four scores; and `clicks` the number of clicks in the prompt. With one click, it takes the mask with the highest score among masks 1 to 3; with more clicks, mask 0. It gives back that mask as an array of True and False: True where the logit is above 0. Replace the three `...`. The check calls it with the four masks of the worked example:

        ```python
        choose(logits, np.array([0.901, 0.645, 0.059, 0.941]), 1)
        ```

        That should give mask 3, the bowl: a mask of 5,575 pixels. The check also tries the click on the handle of the cup, the spoon with a second click, and masks of logits it makes up.
        """),
        L.task("Make `choose(logits, scores, clicks)` give the mask SAM means for the prompt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    choose_code = L.code_box("choose", lines=10)
    choose_code
    return (choose_code,)


@app.cell(hide_code=True)
def check_choose(choose_code):
    result_choose = L.check("choose", choose_code.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "With one click, mask 0 takes no part. `scores[1:]` holds the scores of masks 1 to 3, and `np.argmax` gives the place of the highest of them.",
            "`np.argmax(scores[1:])` counts from the start of `scores[1:]`: its 0 is mask 1, so the number of the mask is one more. The mask itself is True where its logit is above 0: compare the whole array `logits[k]` with 0 at once.",
            "`k = ___ + np.argmax(scores[1:])` with one click, `k = ___` with more, and `return logits[k] > ___`.",
        ],
        why="For the spoon, the scores of masks 1 to 3 are 0.645, 0.059 and 0.941. The highest, 0.941, is at place 2 of `scores[1:]`, so it is mask 3, the bowl, and `logits[3] > 0` is True in 5,575 pixels.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_seeds():
    mo.md(r"""
    ### Exercise 2. A second click
    """)
    return


@app.cell(hide_code=True)
def choice_seeds_cell():
    choice_seeds = L.choice(
        "seeds",
        question="You click once on the seeds in the middle of a sunflower, and SAM's best mask is only the seeds. You want the whole flower. Which second click helps?",
        options={
            "right": "a click on a petal, with label 1",
            "negative": "a click on the seeds, with label 0",
            "again": "a second click on the seeds, with label 1",
            "outside": "a click on the leaves around the flower, with label 1",
        },
        correct="right",
        why={
            "right": "Label 1 on a petal says: this belongs to what I mean too. The seeds alone do not hold the petal, so SAM's mask grows to the whole flower. On the 30 sunflower photos of the slider, the course's second click goes where the mask missed most, and the median IoU goes from 0.18 to 0.91.",
            "negative": "Label 0 says: this is not part of what I mean. On the seeds, it would take the middle of the flower out of the mask. Where does the mask miss something?",
            "again": "That says once more what the first click said: the seeds. Nothing in it tells SAM that the petals belong to it too.",
            "outside": "Label 1 on the leaves says the leaves belong to what you mean too: the mask would grow over the leaves, not only over the petals. Which part of the flower does the mask miss?",
        },
    )
    choice_seeds
    return (choice_seeds,)


@app.cell(hide_code=True)
def text_exercise_next():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Where to click next

        Complete `next_click(mask, truth)`, the start of step 5: `mask` is the mask SAM gave, and `truth` the mask of what you meant, both arrays of True and False of the same shape. It takes the pixels of `truth` the mask missed, and the pixels of the mask that are not in `truth`, and clicks where there are more: in the deepest pixel of the missed pixels, with label 1, or of the extra pixels, with label 0. It gives back the click as `(x, y, label)`, or None when the mask is right. `deepest(region)`, from the course's `everything_ai.sam2`, gives the deepest pixel of a region as `(x, y)`. `~mask` turns every True of the mask into False and every False into True, and `&` gives True where both sides are True. Replace the five `...`. The check calls it with the bowl, mask 3 of the worked example, as `mask`, and as `truth` the spoon as mask 1 has it, with the end of its handle:

        ```python
        next_click(bowl, spoon)
        ```

        That should give `(411, 83, 1)`: a click on the end of the handle. The check also tries masks it makes up: too small, too large, beside the thing, and right.
        """),
        L.watch_out("A click is `(x, y)`: x across, then y down, as `deepest` gives it. An array of pixels is indexed the other way around, `mask[y, x]`: the row first."),
        L.task("Make `next_click(mask, truth)` give the next click where the mask is most wrong."),
    ])
    return


@app.cell(hide_code=True)
def exercise_next():
    next_code = L.code_box("next", lines=14)
    next_code
    return (next_code,)


@app.cell(hide_code=True)
def check_next(next_code):
    result_next = L.check("next", next_code.value)
    result_next
    return (result_next,)


@app.cell(hide_code=True)
def hints_next():
    L.hints(
        "next",
        [
            "Two groups of pixels are wrong: those of `truth` the mask missed, and those of the mask that are not in `truth`. The click goes in the larger group, in its deepest pixel.",
            "The missed pixels are True in `truth` and False in `mask`: `truth & ~mask`; the extra pixels the other way around. `.sum()` counts the True of an array, so compare `missed.sum()` with `extra.sum()`. `deepest` gives `x, y`; the click adds its label: 1 for missed pixels, 0 for extra ones.",
            "`missed = truth & ~___`, `extra = mask & ~___`, `if missed.sum() >= ___.sum():`, then `return (x, y, ___)` twice, with the label of each.",
        ],
        why="On the spoon, the mask of the bowl misses 1,257 pixels of the spoon, the end of its handle, and has 70 too many, along its edge. The missed pixels are more, so the click goes in their deepest pixel, (411, 83), with label 1. With that second click, SAM's mask 0 is the bowl with the end of the handle: its IoU with the spoon of mask 1 goes from 0.806 to 0.941.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your functions on new photos

    Once SAM and the photos are downloaded, in the worked example, the cell below gives SAM 6 photos from Caltech-101 and plays the person who clicks: with your `choose` and `next_click` once they are right, else with the course's. With the slider you set the number of clicks. Each photo shows the outline that came with it, SAM's mask and the clicks, with the IoU above it. The photos take a few seconds, once.
    """)
    return


@app.cell(hide_code=True)
def demo_slider_cell():
    demo_slider = L.slider("demo", "clicks", start=1, low=1, high=5, step=1, role="hyper")
    demo_slider
    return (demo_slider,)


@app.cell(hide_code=True)
def demo_clicks(choose_code, demo_slider, next_code, result_choose, result_next, sam):
    _k = int(demo_slider.widget.value)
    _d = C.demo(sam, choose_code.value if result_choose.state == "correct" else None,
                next_code.value if result_next.state == "correct" else None, _k, {
        "titles": ["sunflower 0: IoU {iou}", "nautilus 0: IoU {iou}", "starfish 4: IoU {iou}", "dragonfly 5: IoU {iou}", "chandelier 1: IoU {iou}", "Windsor chair 2: IoU {iou}"],
        "alt": f"Six photos from Caltech-101 in two rows, a sunflower, a nautilus, a starfish, a dragonfly, a chandelier and a Windsor chair, each with the outline of the object, SAM's mask and the clicks, {_k} in all.",
    }, "en")
    _who = {(True, True): "your `choose` and your `next_click`", (True, False): "your `choose` and the course's `next_click`",
            (False, True): "the course's `choose` and your `next_click`", (False, False): "the course's `choose` and `next_click`"}
    mo.vstack([
        mo.Html(_d["picture"]),
        mo.md(f"Clicked with {_who[(_d['choose'], _d['next'])]}. After {_d['clicks']}, the IoU of the 6 photos is {_d['listed']}."),
    ])
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With one click, five of the six get a part: the seeds of the sunflower, the inner turns of the nautilus, one wing of the dragonfly, the middle of the chandelier and the seat of the chair. Only the starfish is whole at once, 0.94. The second click goes where the mask missed most, and the sunflower and the nautilus become whole, 0.96 and 0.97, and the dragonfly 0.90. The Windsor chair is best after 3 clicks, 0.92, and gets worse after that, 0.73 after 5: more clicks do not always help. The chandelier gets to 0.79 after 5 clicks. Once your functions are right, the picture is the same as with the course's: they choose the same masks and the same clicks.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_name():
    mo.md(r"""
    ### Exercise 4. Saying what it is
    """)
    return


@app.cell(hide_code=True)
def explain_name_cell():
    explain_name = L.explain(
        "name",
        question="Someone builds a photo app: you tap on something in a photo, and the app draws its outline and says what it is, such as \"cup\" or \"spoon\". They use SAM alone. Why can the app draw the outline but not say what it is, and what does it need besides SAM?",
        model_answer="**SAM has no kinds.** It draws the mask of what you tap, but nothing in it says cup or spoon: it never learned names, only where one thing ends and another begins. So the app gets the outline and no word for it. **What would work** is SAM together with a model that learned kinds: SAM's mask says where the thing is, and a classifier trained on photos with the names of their kinds says what is inside that mask. An FCN trained on masks of those kinds, as in the lesson on FCNs, could also give the pixels a kind, but only of the kinds it learned.",
        points=[
            "you say that SAM has no kinds: it gives the mask of what you tap, but no name for it",
            "you say that a name needs a model that learned kinds, such as a classifier or an FCN trained on photos with their kinds",
            "you say how the two work together: SAM's mask says where the thing is, the other model what it is",
        ],
    )
    explain_name
    return (explain_name,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en), a network learned to give every pixel one of its kinds. SAM 2.1 gives no kinds at all: it draws the mask of what you point at, promptable segmentation. Its image encoder turns the photo into features once, and its small decoder turns every prompt into four masks with their predicted IoU: masks 1 to 3 for a click that can mean a part or the whole, mask 0 for a prompt that says what it means. With one click, you choose by score; where the mask is wrong, you click again, in the deepest pixel of what it missed or has too many, as erosion in [Grow and shrink shapes: erosion and dilation](/course/open/cv-morphology?lang=en) finds it. On 30 photos each of 10 kinds of Caltech-101, one click gives a median IoU of 0.85 or more for 1 kind, and two clicks for 8; chandeliers and Windsor chairs, with gaps inside their outlines, never get there. And SAM cannot say what it drew: for a name, it needs a model that learned kinds. How its image encoder, a transformer, works, later lessons explain.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_seeds, explain_name, result_choose, result_next):
    L.wrap_up(result_choose, choice_seeds, result_next, explain_name)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
