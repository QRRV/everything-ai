# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-project-detector-or-unet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-project-detector-or-unet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Project: your own U-Net](/course/open/cv-unet?lang=en) and about [Focal loss and RetinaNet](/course/open/cv-focal-loss?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_iou():
    quick_iou = L.choice(
        "quick-iou",
        question="With seed 3, the course's U-Net read 97.3% of the pixels of the test scenes right after 8 epochs, and its mean IoU was only 0.65. How can both be true?",
        options={
            "right": "It called the triangles squares; their IoU is about 0, and in the mean IoU the triangles count as much as the background, which covers most pixels",
            "twice": "The mean IoU counts every pixel twice, once for each class it could have",
            "train": "97.3% is about the training scenes, and 0.65 about the test scenes",
            "edges": "The mean IoU only looks at the pixels on the edges of the shapes",
        },
        correct="right",
        why={
            "right": "Most pixels are background, and the U-Net got those right: so 97.3% of all pixels. The mean IoU averages the IoU of the 4 classes, and every class counts the same: with the triangles at about 0, it falls to 0.65. Today your U-Net is measured that way again, with the mean IoU over all pixels of all test scenes, or your detector with the mAP, which also counts every kind the same.",
            "twice": "Every pixel has one class, and every class gets its own IoU: the pixels it was given and should have, divided by those it was given or should have. Then the mean over the 4 classes. See [Project: your own U-Net](/course/open/cv-unet?lang=en).",
            "train": "Both numbers are about the 200 test scenes. The difference is in what they count: the share of all pixels, most of them background, or the mean of the IoU of every class, each counting the same.",
            "edges": "The IoU of a class counts all pixels of that class, inside the shapes too. The difference is in how the two numbers weigh the classes: in the mean IoU, the few pixels of the triangles count as much as all of the background.",
        },
    )
    quick_iou
    return (quick_iou,)


@app.cell(hide_code=True)
def quick_check_focal():
    quick_focal = L.choice(
        "quick-focal",
        question="In the focal loss with γ = 2, how much does an anchor that the detector puts at the right class with a probability of 0.9 count, compared with its cross-entropy?",
        options={
            "right": "A hundredth: its cross-entropy times (1 − 0.9)² = 0.01",
            "tenth": "A tenth: its cross-entropy times 1 − 0.9",
            "same": "As much: the focal loss only changes the anchors the detector gets wrong",
            "more": "More: an easy anchor counts double, so that it stays right",
        },
        correct="right",
        why={
            "right": "The focal loss weights the cross-entropy of every anchor by $(1 - p_t)^\\gamma$: $0.1^2 = 0.01$. So the many easy background anchors weigh little, and training goes to the anchors it still gets wrong. Today's detector is the one-stage detector of [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en), with its own loss; the question today is what happens to it when the scenes change.",
            "tenth": "That would be γ = 1. With γ = 2 the factor is squared: $(1 - 0.9)^2 = 0.01$, a hundredth.",
            "same": "It changes every anchor, by $(1 - p_t)^\\gamma$: a lot for the easy ones, near $p_t = 1$, and hardly for the hard ones, near $p_t = 0$. See [Focal loss and RetinaNet](/course/open/cv-focal-loss?lang=en).",
            "more": "It is the other way round: an easy anchor gets a small factor, $(1 - p_t)^\\gamma$, so it counts less, and the hard anchors count relatively more.",
        },
    )
    quick_focal
    return (quick_focal,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Every network of this part learned from the course's scenes: bright shapes on an even gray, in good light. On new scenes like them, the course's U-Net reached a mean IoU of about 0.95, and its detector an mAP of about 0.91. But a camera in fog, in the shade or at dusk does not make such pictures. A network that meets data unlike the data it learned from faces a **distribution shift**, and then its test score says little about what it does. Today you choose one of your two networks, your U-Net or your detector, and make the scenes harder yourself: with fog, with light that falls off to one side, and with noise. The shapes stay where they are; only their pixels change. You see how networks that learned from easy scenes do on hard ones, choose on validation scenes what yours learns from, test it once, and then look at where it still fails.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working", "Your CNN on color photos", "Your own one-stage detector",
                            "Your own U-Net", "Final project: ResNet with Grad-CAM", "Final project: U-Net or detector"],
                           8, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_easy():
    guess_easy = L.guess(
        "easy-on-hard",
        question="The course's detector learned from the 2,000 training scenes as they are, and finds the shapes of the 200 validation scenes with an mAP of about 0.91. Then those 200 scenes are made harder, each with its own fog, light and noise. What mAP does the same detector reach on them?",
        options={
            "same": "About 0.9: the shapes are the same",
            "some": "About 0.6",
            "third": "About 0.3",
            "none": "About 0.05: it finds hardly anything",
        },
        reveal="About 0.3: from 0.22 to 0.28 over the seeds 0 to 4, against 0.90 to 0.92 on the scenes as they are. And the course's U-Net, which learned from the same easy scenes, falls from a mean IoU of 0.94 to 0.23, in the median of the five seeds. The shapes are the same, at the same places; the networks never saw pixels like these. The slider below shows where they break.",
    )
    guess_easy
    return (guess_easy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The same scenes, other light.** A hard scene is a scene of the course, with the same shapes, masks and boxes; only its pixels change, in three steps, in this order. The picture works with floats, from 0 to 255.

        1. **Fog** pulls every number toward the mean $m$ of all numbers of the picture: $x' = m + (x - m)(1 - \text{fog})$. Every difference with the mean gets $1 - \text{fog}$ times as large: the picture loses contrast, as the dull photo of [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en). With fog 1, it is one gray.
        2. **Light** from the right: every number of column $j$ is multiplied by $1 - \text{light} \cdot \left(1 - \frac{j}{W - 1}\right)$, where $W$ is the number of columns. The right column stays as it is, the left one gets $1 - \text{light}$ times as bright, and the columns in between something in between.
        3. **Noise**: a number for every pixel and every channel is added, drawn from a normal distribution with a mean of 0 and a standard deviation $\sigma$, in gray values.

        At the end, every number goes back within 0 and 255 with `np.clip(x, 0, 255)`, is rounded, and becomes uint8. Every hard scene draws its own fog, light and $\sigma$ from its seed, each evenly from 0 to 0.8, 0 to 0.8 and 0 to 24, and its noise too: so a hard scene is the same on every computer.

        **Distribution shift.** A network learns from what its training data shows. The course's networks learned from easy scenes, where a shape always stands out clearly against bright gray. On the hard validation scenes, they lose most of what they could: the detector falls from an mAP of 0.91 to 0.28, the U-Net from a mean IoU of 0.94 to 0.23, each the median of the seeds 0 to 4. The data a network meets after its training can differ from the data it learned from, and a test set from the training data's own source does not show that.

        **What it learns from, chosen on the validation scenes.** The course tried three settings for each network: learning from the 2,000 training scenes as they are; from all of them made harder; and from half of them made harder, the ones with an even seed. Each setting trained for 8 epochs, as in [Project: your own U-Net](/course/open/cv-unet?lang=en) and [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en), with the seeds 0 to 4, and was measured on 200 validation scenes, seeds 3000 to 3199, made harder: as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), every choice is made there. The 200 test scenes, seeds 0 to 199, made harder, are measured once, at the end, so that their number says how the network does on scenes nothing was chosen for.

        **Where it fails.** One number, the mean IoU or the mAP, does not say what goes wrong. So you look at every shape of the hard validation scenes: found or not. On the validation scenes, not the test scenes: what you see there decides what you try next, and the test scenes stay for the one measurement at the end. For the U-Net, a shape is found when more than half of its pixels in sight get its kind. For the detector, when one of its boxes with a score of 0.5 or more has its kind and an IoU above 0.5 with it, as the mAP of [Score a detector: mean average precision](/course/open/cv-map?lang=en) counts it. Then you put the shapes in **groups** by one property: its pixels in sight, the light where it lies, or the fog or the noise of its scene. For every group, the **share found** says how many of its shapes the network found. A group with a low share says where the network fails, and so what the scenes it learns from could hold more of; how much a share says depends on how many shapes the group holds. Such an error analysis looks at the mistakes themselves, not only at how many there are.
        """),
        L.watch_out("A number of uint8 is a whole number from 0 to 255, and noise can push bright and dark pixels out of that range. `astype(np.uint8)` does not keep a number above 255 at 255, or one below 0 at 0: on the course's computer, 270.4 becomes 14 and −3.2 becomes 253, and what comes out can differ from computer to computer. And it cuts off the decimals, so 99.6 becomes 99. That is why `np.clip(x, 0, 255)` and `np.round` come first."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **distribution shift** | a difference between the data a network learned from and the data it meets later, such as photos in other light; its score on a test set of the old data then says little | |
        | **group** | the shapes whose property lies in one bin, such as the shapes of 250 to 500 pixels in sight | `inside` |
        | **share found** | for a group, the number of its shapes the network found, divided by the number of shapes in it | `shares` |
        | `np.clip(x, lo, hi)` | every number of x below lo becomes lo, every number above hi becomes hi | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_changes():
    mo.vstack([
        mo.Html(C.changes_picture({"as_is": "as it is", "fog": f"fog {L.number(C.FOG, 1)}", "light": f"light {L.number(C.LIGHT, 1)}", "noise": f"noise σ {C.NOISE:g}",
                                   "all": "all three", "alt": f"Validation scene {C.PLAY} as it is, with only fog, only light and only noise at their largest, and with all three as its seed drew them."})),
        mo.md(f"""
        Validation scene {C.PLAY}, as it is and made harder by the course: with fog {L.number(C.FOG, 1)} only, every difference with the mean shrinks to a fifth, so the shapes stay where they are, but their colors fade almost into the gray; with light {L.number(C.LIGHT, 1)} only, the left edge keeps a fifth of its brightness; with noise only, every number gets its own random number added, with a standard deviation of {C.NOISE:g} gray values; and with all three as its seed drew them: fog {L.number(C.conditions(C.PLAY)['fog'], 2)}, light {L.number(C.conditions(C.PLAY)['light'], 2)} and noise σ {L.number(C.conditions(C.PLAY)['sigma'], 1)}.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    How hard can a scene get before a network fails? The slider makes all 200 validation scenes harder at once: at level 0 they are as they are, at level 1 each has fog 0.8, light 0.8 and noise σ 24, and in between each change is that fraction of its maximum. The chart shows, for every level, the mean IoU of the course's U-Net and the mAP of its detector on them, each trained with seed 0: once on the easy training scenes, and once as the course chose on the validation scenes.
    """)
    return


@app.cell(hide_code=True)
def explore_levels_cell():
    def _frame(v):
        _labels = {"title": "On the 200 validation scenes", "x": "level", "y": "mean IoU or mAP", "scene": "validation scene {seed}",
                   "unet_easy": "U-Net, learned from easy scenes", "unet_chosen": "U-Net, learned from hard scenes",
                   "detector_easy": "detector, learned from easy scenes", "detector_chosen": "detector, learned from easy and hard scenes",
                   "alt": "A validation scene at this level, beside the mean IoU of the course's U-Net and the mAP of its detector at every level, each learned from easy scenes and as the course chose, with a line at this level."}
        _html, _stop = C.play_frame(v, _labels, L.lang)
        return _html, (f"At level {L.number(_stop['level'], 2)}: fog {L.number(_stop['fog'], 2)}, light {L.number(_stop['light'], 2)} and noise {L.number(_stop['noise'], 1)}. "
                       f"Learned from easy scenes, the U-Net reaches a mean IoU of {L.number(_stop['unet_easy'], 2)} and the detector an mAP of {L.number(_stop['detector_easy'], 2)}; "
                       f"as the course chose, {L.number(_stop['unet_chosen'], 2)} and {L.number(_stop['detector_chosen'], 2)}.")

    explore_levels = L.explore("levels-slider", "level", _frame, start=0.0, values=list(C.LEVELS), role="",
                               task="Go from level 0 to level 1: at which level do the networks that learned from easy scenes fail, and what does the scene look like there? And what does learning from hard scenes cost at level 0?")
    explore_levels
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Learned from easy scenes, both networks still do well at level 0.25, 0.85 and 0.93, and break before level 0.5: there the detector reaches 0.37 and the U-Net 0.32. At level 1, the detector finds almost nothing, 0.01. Learned as the course chose, they hold up to level 0.75, with 0.77 and 0.87, and at level 1 they too lose much, to 0.37 and 0.52. At level 1 every scene has all three changes at their maximum, and a hard training scene draws each of them evenly from 0 to its maximum: of the 2,000 training scenes made harder, only 33 have all three above level 0.75. And learning from hard scenes costs a little at level 0: the detector reaches 0.85 instead of 0.92, the U-Net 0.90 instead of 0.94.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Your own network, on scenes it finds harder:

    1. Make a scene harder: fog toward the mean, light that falls off to the left, noise; then within 0 and 255, rounded, as uint8.
    2. Make the training scenes as the setting says, and the validation scenes and the test scenes harder, each with the fog, light and noise of its seed.
    3. Choose the setting on the validation scenes, over several seeds: here the course did that.
    4. Train your network with the chosen setting, then measure it once on the hard test scenes: the mean IoU for the U-Net, the mAP for the detector.
    5. For every shape of the hard validation scenes: found or not. Put the shapes in groups by a property, and give the share found of every group, with the number of shapes in it.
    6. Write what you see: where it fails, and what to try next, chosen on the validation scenes again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell does step 1 for a made-up scene of 2 rows of 4 pixels, gray with one red and one blue pixel: the fog with fog 0.5, and the light with light 0.6. The second cell does the end of step 1 for five numbers that the noise could have made. The third cell does the first part of step 5 for 7 made-up shapes: which are in the group of 60 to 250 pixels in sight, and the share of them found. Run the cells as they are.
    """)
    return


@app.cell
def example_fog():
    import numpy as np

    # 1. a made-up scene of 2 rows of 4 pixels, gray, with one red and one blue pixel
    image = np.array([[[220, 220, 220], [220, 220, 220], [200, 60, 50], [220, 220, 220]],
                      [[220, 220, 220], [30, 140, 200], [220, 220, 220], [220, 220, 220]]], dtype=np.uint8)
    x = image.astype(float)

    # fog 0.5: every number halfway to the mean of all numbers
    m = x.mean()
    foggy = m + (x - m) * (1 - 0.5)
    print(round(m, 2))
    print(foggy[0, 2])

    # light 0.6: one number for every column, from 0.4 on the left to 1 on the right
    w = x.shape[1]
    ramp = 1 - 0.6 * (1 - np.arange(w) / (w - 1))
    print(ramp)
    lit = foggy * ramp.reshape(1, w, 1)
    print(lit[0, 0], lit[0, 3])
    return image, np


@app.cell
def example_clip(np):
    # the end of step 1: numbers the noise could have made
    noisy = np.array([270.4, -3.2, 99.6, 127.5, 128.5])
    print(np.clip(noisy, 0, 255))
    print(np.round(np.clip(noisy, 0, 255)))
    print(np.round(np.clip(noisy, 0, 255)).astype(np.uint8))

    # the noise of exercise 1's example: a number for every pixel and channel of the made-up scene
    noise = np.random.default_rng(0).normal(0, 20, (2, 4, 3))
    print(noise.shape)
    return (noise,)


@app.cell
def example_group(np):
    # 5. made-up shapes: whether the network found each one, and its pixels in sight
    found = np.array([True, False, True, True, False, True, False])
    sizes = np.array([80, 120, 300, 450, 90, 700, 1200])

    # the group of 60 to 250 pixels: which shapes are in it, and the share of them found
    inside = (sizes >= 60) & (sizes < 250)
    print(inside)
    print(found[inside], found[inside].mean())
    return found, sizes


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the first cell, the mean of all 24 numbers is 193.33. With fog 0.5, every number goes halfway to it: the red pixel, (200, 60, 50), becomes (196.67, 126.67, 121.67), much closer to gray, and the gray 220 becomes 206.67. The light gives the 4 columns 0.4, 0.6, 0.8 and 1: the gray pixel in the left column keeps 0.4 of its brightness, 82.67, and the one in the right column stays at 206.67. In the second cell, `np.clip` makes 270.4 into 255 and −3.2 into 0, and `np.round` makes 99.6 into 100. 127.5 and 128.5 both become 128: NumPy rounds a number exactly halfway between two whole numbers to the even one. In the third cell, 3 of the 7 shapes have 60 to 250 pixels in sight, those of 80, 120 and 90 pixels, and 1 of the 3 was found: a share found of 0.33.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harder():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A harder scene

        Complete `harder(image, fog, light, noise)`, step 1 of the recipe, for a scene `image` of any shape (H, W, 3) with numbers of uint8. `noise` holds a number for every pixel and channel, in gray values. First the fog, with the mean of the whole picture; then the light, one number for every column; then the noise; and at the end every number within 0 and 255, rounded, as uint8. The first cell of the worked example did the fog and the light, the second the end. The check tries the made-up scene with the `noise` of the worked example, fog and light of 0 and of 1, noise that pushes numbers above 255 and below 0, a scene with more rows than columns, and a whole validation scene. When it is right, it saves `harder` in your project, as `work/projects/cnn/harder.py`; the demo makes every scene harder with it.
        """),
        L.task("Make `harder` add fog, light from the right and noise to a scene, and give back whole numbers from 0 to 255."),
    ])
    return


@app.cell(hide_code=True)
def exercise_harder():
    box_harder = L.code_box("harder", lines=14)
    box_harder
    return (box_harder,)


@app.cell(hide_code=True)
def check_harder(box_harder):
    result_harder = L.check("harder", box_harder.value)
    result_harder
    return (result_harder,)


@app.cell(hide_code=True)
def hints_harder():
    L.hints(
        "harder",
        [
            "Work with floats, `x = image.astype(float)`, so that nothing wraps around while you compute. Then three steps that each work on the whole array, in the order of the comments, as in the first cell of the worked example, and the end as in the second.",
            "`m = x.mean()` and `x = m + (x - m) * (1 - fog)`. For the light: `w = x.shape[1]`, `ramp = 1 - light * (1 - np.arange(w) / (w - 1))` and `x = x * ramp.reshape(1, w, 1)`, so that every column gets its own number.",
            "`x = x + noise`, and then `return np.___(np.clip(x, 0, 255)).astype(np.uint8)`.",
        ],
        why="`ramp.reshape(1, w, 1)` has the shape (1, W, 1): broadcasting, as in [Broadcasting](/course/open/np-broadcasting?lang=en), repeats it over the rows and the 3 channels, so every pixel of column `j` is multiplied by the same number. The mean is one number for the whole picture, so the fog is gray, not a color. And `np.clip` comes before `astype`, because uint8 cannot hold a number below 0 or above 255.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The share found of every group

        Complete `shares(found, values, edges)`, step 5 of the recipe. `found` says for every shape whether the network found it, and `values` holds a number for every shape, such as its pixels in sight. `edges` holds k + 1 numbers from low to high, the edges of k bins: a value v is in bin i when `edges[i] <= v < edges[i + 1]`. The function gives back a list of k: for every bin the share of its shapes that was found, and `None` for a bin without shapes. The third cell of the worked example did it for one bin. The check tries the made-up shapes of the worked example, values on an edge, empty bins, values outside every bin, and lists instead of arrays. When it is right, it saves `shares` in your project, as `work/projects/cnn/shares.py`; after the demo, it shows where your network fails.
        """),
        L.task("Make `shares` give the share found of the shapes in every bin, and `None` for an empty bin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=12)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "A loop over the bins, `for i in range(len(edges) - 1):`, and in it what the third cell of the worked example did, with `edges[i]` and `edges[i + 1]` in the place of 60 and 250. Collect what every bin gives in a list.",
            "`out = []` before the loop. In it: `inside = (values >= edges[i]) & (values < edges[i + 1])`, then `if inside.sum() > 0:` with `out.append(float(found[inside].mean()))`.",
            "`else:` with `out.append(___)`, and after the loop `return out`.",
        ],
        why="`found[inside]` keeps the shapes of the bin, and the mean of True and False is the share that is True. A bin holds its lower edge and not its upper edge, so a value on an edge is in exactly one bin. An empty bin has no share: the mean of nothing is `nan`, and 0 would read as a bin where the network found none.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    Choose your network, and press the button. The demo makes every scene harder with your `harder`: the 2,000 training scenes as the course chose for that network, all of them for the U-Net and those with an even seed for the detector, and the 200 validation scenes and the 200 test scenes all. Then it trains your network with seed 0 for {C.EPOCHS} epochs, as in its own lesson, looks at every shape of the hard validation scenes, and measures it once on the hard test scenes: your U-Net with your `mean_iou`, your detector with the mAP. Where a module of yours is missing or broken, the course's stands in. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; on the course's computer, that took 9 to 10 minutes for the U-Net and about 2 for the detector. While it trains, the chart shows the loss on the training scenes every 10 batches, and dashed, the loss on the validation scenes after every epoch. Beside it are three validation scenes, made harder, with what your network makes of them at that moment, and under them the feature maps of the layer it watches; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def model_choice_cell():
    model_choice = mo.ui.radio(options={"your U-Net": "unet", "your detector": "detector"}, value="your detector", label="Train")
    train_button = mo.ui.run_button(label="Train your network on harder scenes")
    mo.vstack([model_choice, train_button])
    return model_choice, train_button


@app.cell(hide_code=True)
def demo_network(train_button, model_choice, result_harder):
    result_harder  # run again when exercise 1 saves your module
    if not train_button.value:
        _out = mo.md("*Choose your network, and press the button to train it.*")
    else:
        _kind = model_choice.value
        _name = "U-Net" if _kind == "unet" else "detector"
        _now = {}   # the live chart, made at the first report
        _words = {"scene": "scene {seed}", "map": "", "kinds": ["circle", "square", "triangle"], "shapes": "the shapes", "boxes": "boxes of 0.5 or more",
                  "layer": f"Layer {{name}} of your {_name} on validation scene {{seed}}",
                  "alt": ("Three validation scenes, made harder, each with the class map your U-Net gave it after that epoch." if _kind == "unet" else
                          "Three validation scenes, made harder, each with its true boxes and the dashed boxes of 0.5 or more your detector gave it after that epoch.")}
        _words_now = _words | {"alt": _words["alt"].replace("gave it after that epoch", "gives it now")}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": f"Your {_name}: the loss while it trains", "x": "epoch", "y": "loss",
                                        "alt": f"The loss of your {_name} so far: on the training scenes every 10 batches, and dashed on the 200 validation scenes after every epoch."}, _p["epochs"], low=0.0,
                                       key=[("loss", False, "training scenes, every 10 batches"), ("loss", True, "validation scenes, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _kind, _words_now, L.lang, "nv-scenes-project-now"))
                _now["val"] = []
            if "val_loss" in _p:   # the validation scenes after every epoch, dashed
                _now["val"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["val"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your {_name}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {L.number(_p['val_loss'], 3)}")
            else:   # the training scenes every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your {_name}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {L.number(_p['loss'], 3)}")

        _result = C.demo(_kind, _progress)
        if _result is None:
            _out = mo.md(f"*Finish exercise 1 first, and the exercises of your {_name} in {'[Project: your own U-Net](/course/open/cv-unet?lang=en)' if _kind == 'unet' else '[Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en)'}: then your `harder` and your {_name} are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Your project takes too long here: more than two minutes to make the scenes harder, more than four for the targets of the detector, or more than half an hour to train the U-Net or ten minutes to train the detector. Does a module of yours do more work than it needs, such as a loop over the pixels or the anchors?*",
                "harder": "*Your `harder` in `work/projects/cnn/harder.py` stops with an error, or does not give a picture of 128 by 128 by 3 with whole numbers from 0 to 255. Did you change its file? Delete it, and check exercise 1 again: its check then saves your code again.*",
                "classes": "*Your `class_map` in `work/projects/cnn/classes.py` stops with an error, or does not give a class map for every scene. Did you change its file? Delete it, and check its exercise in [A kind for every pixel: FCN and transposed convolutions](/course/open/cv-fcn?lang=en) again.*",
                "unet": "*Your `UNet` in `work/projects/cnn/unet.py` stops with an error, or does not give 4 scores for every pixel. Did you change its file? Delete it, and check its exercise in [Project: your own U-Net](/course/open/cv-unet?lang=en) again.*",
                "mean_iou": "*Your `mean_iou` in `work/projects/cnn/mean_iou.py` stops with an error, or does not give 1 for class maps compared with themselves. Did you change its file? Delete it, and check its exercise in [Project: your own U-Net](/course/open/cv-unet?lang=en) again.*",
                "targets": "*Your `targets` in `work/projects/cnn/targets.py` stops with an error, or does not give a class and 4 offsets for each of the 768 anchors of a scene. Did you change its file? Delete it, and check its exercise in [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) again.*",
                "loss": "*Your `detection_loss` in `work/projects/cnn/detection_loss.py` stops with an error, or does not give back one number as a tensor that training can follow back. Did you change its file? Delete it, and check its exercise in [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) again.*",
                "detector": "*Your `Detector` in `work/projects/cnn/detector.py` stops with an error, or does not give 768 rows of 8 for a scene. Did you change its file? Delete it, and check its exercise in [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) again.*",
                "offsets": "*Your `boxes_from` in `work/projects/cnn/offsets.py` stops with an error on your detector's offsets. Did you change its file? Delete it, and check its exercise in [Anchors: the boxes a detector starts from](/course/open/cv-anchors?lang=en) again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "alt": f"The loss of your {_name} after every epoch: on the training scenes solid, on the validation scenes dashed."}
            _val, _moments = _result["curves"]["val"], _result["moments"]

            def _frame(_e):
                _e = int(_e)
                return (C.epoch_frame(_result, _e, _labels, _words, L.lang),
                        f"After epoch {_e}: validation loss {L.number(_val[_e - 1], 3)}.")

            if len(_moments) == len(_val) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=len(_val), values=list(range(1, len(_val) + 1)), role="",
                                  task="Go back to epoch 1, then forward again: which shapes does your network find first, and which does it still miss at the end?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            if _kind == "unet":
                _said = (f"Your U-Net, with {C.whole(_result['params'], L.lang)} parameters, reaches a mean IoU of **{L.number(_result['score'], 3)}** on the 200 test scenes made harder, measured once, "
                         f"and finds {_result['test_found']} of their {_result['test_shapes']} shapes. On the same scenes as they are: {L.number(_result['easy'], 3)}. The course's U-Net with seed 0, learned from hard scenes: 0.908 on the hard test scenes, with 764 of the 808 shapes found, and 0.922 on them as they are. Its validation loss jumped up after epochs 4 and 6 and came back down after each, to 0.037 after epoch 8.")
            else:
                _said = (f"Your detector, with {C.whole(_result['params'], L.lang)} parameters, reaches an mAP of **{L.number(_result['score'], 3)}** on the 200 test scenes made harder, measured once, "
                         f"and finds {_result['test_found']} of their {_result['test_shapes']} shapes. On the same scenes as they are: {L.number(_result['easy'], 3)}. The course's detector with seed 0, learned from easy and hard scenes: 0.815 on the hard test scenes, with 629 of the 808 shapes found, and 0.852 on them as they are.")
            _out = mo.vstack(_pic + [
                mo.md(_said + _who),
                mo.md(f"**Your {_name}, layer by layer.** The first box is validation scene {C.EXAMPLE}, made harder, and the boxes after it are the layers of your trained {_name}, in the order the scene goes through them; under them are the feature maps of the layer you choose."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Go through your trained network from the scene to its scores: do the shapes in the dark left part stand out in the feature maps, and from which layer on?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_analysis():
    mo.md(r"""
    ### Where it fails

    Once your network has trained, the button below puts the 802 shapes of the 200 hard validation scenes in groups, four times: by their pixels in sight, by the light where they lie (1: as bright as in an easy scene, 0.2: a fifth of it), by the fog of their scene and by its noise. Your `shares` gives the share found of every group. The bars are your network's; the dashed line over every bar is the course's network of the same kind, trained with seed 0; above every bar is the number of shapes in its group. Under the charts are the three validation scenes where your network missed the most shapes: every shape it missed in a thick box, every shape it found in a thin one.
    """)
    return


@app.cell(hide_code=True)
def analysis_button_cell():
    analysis_button = mo.ui.run_button(label="Look where it fails")
    analysis_button
    return (analysis_button,)


@app.cell(hide_code=True)
def analysis_cell(analysis_button, result_shares):
    result_shares  # run again when exercise 2 saves your module
    _run = C.last_run()
    if not analysis_button.value:
        _out = mo.md("*Train your network above first, then press the button.*")
    elif _run is None:
        _out = mo.md("*There is no trained network yet: press the button of the demo above first, and this one again when it is done.*")
    else:
        _groups = C.analysis(_run)
        if _groups is None:
            _out = mo.md("*Finish exercise 2 first: then your `shares` is in your project, and this cell uses it.*")
        elif isinstance(_groups, str):
            _out = mo.md("*Your `shares` in `work/projects/cnn/shares.py` stops with an error, or does not give a share from 0 to 1 or `None` for every bin. Did you change its file? Delete it, and check exercise 2 again.*")
        else:
            _name = "U-Net" if _run["model"] == "unet" else "detector"
            _out = mo.vstack([
                mo.Html(C.groups_figure(_groups, {"size": "By pixels in sight", "size_x": "pixels in sight", "light": "By light", "light_x": "light where it lies",
                                                  "fog": "By fog", "fog_x": "fog of its scene", "noise": "By noise", "noise_x": "noise σ of its scene",
                                                  "mine": f"your {_name}", "course": {"unet": "the course's U-Net, seed 0", "detector": "the course's detector, seed 0"}[_run["model"]],
                                                  "alt": f"The share of the shapes of the validation scenes your {_name} found, {{what}}, with the course's as a dashed line and the number of shapes above every bar."}, L.lang)),
                mo.Html(C.worst_picture(_run, {"scene": ("scene {seed}: {n} missed", "scene {seed}: {n} missed"), "map": "", "kinds": ["circle", "square", "triangle"],
                                               "missed": "a shape it missed", "found": "a shape it found", "boxes": f"boxes of your {_name}, 0.5 or more",
                                               "alt": f"The three validation scenes where your {_name} missed the most shapes, made harder: every missed shape in a thick box, every found one in a thin box."}, L.lang)),
                mo.md("On the hard validation scenes, the course's detector with seed 0 finds 604 of the 802 shapes. It fails most on small shapes, 40% of the 133 with 60 to 250 pixels in sight against 97% of the 79 largest, and in heavy fog, 60% of the 212 shapes at fog 0.6 to 0.8 against 86% at fog 0 to 0.2; across the groups of noise, its share stays between 73% and 78%. The course's U-Net with seed 0 finds 746 of the 802; its lowest share is that of the smallest shapes, 85%, and in the heaviest fog it finds 88%. The darkest group of light holds only 17 shapes: one shape more or less moves its share by 6 percentage points, so it says less than the share of a group of 200."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Exercise 3. What to try next
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next-step",
        question="The course's detector with seed 0 finds 60% of the shapes of the hard validation scenes in the heaviest fog, 0.6 to 0.8, against 86% in the lightest. You want to try whether training scenes with more fog help. Where do you measure whether they do?",
        options={
            "right": "On the validation scenes: the share found in heavy fog, and the mean IoU or mAP of all of them",
            "test": "On the test scenes made harder: they say how it does on new scenes",
            "train": "On the training scenes with heavy fog: there you can see whether it learned them",
            "fog": "On the test scenes with heavy fog only: that is the group that should get better",
        },
        correct="right",
        why={
            "right": "Yes. Every choice is made on the validation scenes, as the setting was: the share found in heavy fog says whether the change helped where it should, and the score of all of them whether it cost something elsewhere. The test scenes stay for the end, measured once, so that their number keeps saying how the network does on scenes nothing was chosen for.",
            "test": "Once you choose with the test scenes, their number no longer says how the network does on new scenes: you would have picked the change that suits these 200. Choose on the validation scenes, and measure the test scenes once, at the end.",
            "train": "The training scenes say how well the network learned the scenes it saw, not how it does on others: a network can fit its training scenes and still fail on new ones, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en).",
            "fog": "Those are still test scenes: choosing with them uses the test set, as with all of them. And a change can help in fog and cost elsewhere: look at all validation scenes too.",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 4. Your report
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Write a short report on your network for someone who did not take this lesson: what was made harder, and how did networks that learned from easy scenes do there? How was the setting chosen, how did your network do on the hard test scenes, and in which groups of the hard validation scenes does it fail most? Use the numbers of your own run, or those of the course's run with seed 0.",
        model_answer="The hard scenes are the course's scenes with the same shapes, made harder with fog toward the mean, light that falls off to the left, and noise, each drawn per scene. A detector that learned from the scenes as they are reaches an mAP of 0.91 on the validation scenes as they are, but only 0.28 on them made harder, the medians of five seeds. I chose what my detector learns from on the 200 hard validation scenes, by the median of five seeds: from the scenes as they are 0.28, from all of them made harder 0.78, and from half of them made harder 0.80; so half. Measured once on the 200 hard test scenes, it reaches an mAP of 0.815 with seed 0 and finds 629 of their 808 shapes; on the same scenes as they are, 0.852. On the hard validation scenes it fails most on small shapes, 40% of the 133 with 60 to 250 pixels in sight, and in heavy fog, 60% of the 212 shapes at fog 0.6 to 0.8. Next I would try training scenes with more fog, and choose on the validation scenes whether that helps there without costing elsewhere.",
        points=[
            "you say what the hard scenes are, the same shapes with fog, light from one side and noise, and how much networks that learned from easy scenes lose there",
            "you say that the setting was chosen on the validation scenes, with what each setting reached there, and that the hard test scenes were measured once, at the end",
            "you give your network's score on the hard test scenes, and the groups of the validation scenes with the lowest share found and how many shapes they hold, with what they suggest to try next, chosen on the validation scenes",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The U-Net of [Project: your own U-Net](/course/open/cv-unet?lang=en) and the detector of [Project: your own one-stage detector](/course/open/cv-one-stage-detector?lang=en) met scenes unlike those they learned from: the same shapes, in fog, in the shade and with noise. The fog takes away the contrast that [Spread the grays: histogram equalization](/course/open/cv-histogram-equalization?lang=en) spread out, and the scores are the mean IoU and the mAP of [Score a detector: mean average precision](/course/open/cv-map?lang=en). As in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en), what each network learns from was chosen on the validation scenes and the test scenes were measured once; the error analysis looked at the validation scenes too, so that what you try next is chosen there. Your project `cnn` now has `harder.py` and `shares.py`.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_harder, result_shares, choice_next, explain_report):
    L.wrap_up(result_harder, result_shares, choice_next, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
