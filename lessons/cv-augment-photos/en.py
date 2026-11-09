# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-augment-photos. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-augment-photos", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Data augmentation](/course/open/nn-data-augmentation?lang=en) and [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="In *Data augmentation*, a mirror image was among the changes that can change a digit or make it unreadable. Why?",
        options={
            "pixels": "A mirror image moves every pixel, and a network cannot learn from that",
            "right": "A mirrored 2, 3 or 7 is not a digit anyone writes",
            "mlp": "A mirror image only helps a CNN, not an MLP",
        },
        correct="right",
        why={
            "pixels": "A shift of one pixel moves every pixel too, and that kept the digit. What counts is whether the label still holds: a mirrored 3 is not a digit anyone writes. See [Data augmentation](/course/open/nn-data-augmentation?lang=en).",
            "mlp": "Whether a change keeps the label does not depend on the network: a mirrored 3 is not a digit anyone writes, for an MLP and for a CNN. See [Data augmentation](/course/open/nn-data-augmentation?lang=en).",
            "right": "Handwritten digits have a left and a right. Photos of cars, horses and ships do not: mirrored, they are photos like any other, and today a mirror image is one of the changes.",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def quick_check_gap():
    quick_gap = L.choice(
        "quick-gap",
        question="After 20 epochs, your ColorCNN read 92 to 97% of its 6,000 training photos right, and 52 to 54% of the test photos. After epoch 3 to 6, its validation loss rose again. What happened?",
        options={
            "epochs": "It had too few epochs to learn the test photos",
            "small": "It has too few parameters for photos this hard",
            "right": "It learned its training photos by heart: details that do not hold for new photos",
        },
        correct="right",
        why={
            "epochs": "A network never learns from the test photos: they only measure. And more epochs made it worse there: after epoch 3 to 6, the validation loss rose. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "small": "It had enough parameters to read 92 to 97% of 6,000 photos right: it could learn many details. Those details just did not hold for new photos. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "right": "Today, every epoch shows it its training photos a little differently, so that it cannot learn them by heart as easily.",
        },
    )
    quick_gap
    return (quick_gap,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your ColorCNN learned its 6,000 training photos by heart. In [Data augmentation](/course/open/nn-data-augmentation?lang=en), random shifts of a pixel made a network on digits learn less by heart. Photos allow more changes than digits: a car, mirrored, is still a car, and another day gives other light. The ResNet paper of your project trained on CIFAR-10 with two of today's changes, a mirror image and a random crop (He et al., 2015), and AlexNet, the CNN that won ImageNet in 2012, mirrored its photos, cut out random parts of them and changed their colors a little. Today you write the crop and the change of color yourself, and train your ColorCNN with them.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_train():
    guess_train = L.guess(
        "train-right",
        question="Without changes, your ColorCNN read 92 to 97% of its training photos right after 20 epochs. Now every epoch shows it each photo mirrored or not, cut out at a random place, and with slightly other colors, for 40 epochs. Afterwards, how many of its 6,000 training photos, as they are, does it read right?",
        options={
            "more": "Nearly all of them, as without changes",
            "right": "About two thirds",
            "half": "About half, as many as of the test photos",
        },
        reveal="**About two thirds.** Over 5 seeds, with the course's code, it reads 3,917 to 4,024 of its 6,000 training photos right, 65 to 67%: far fewer than without changes. But of the 1,000 test photos, it reads 583 to 619 right, against 515 to 543 without changes: it learned less about its own photos, and more about the 10 kinds.",
    )
    guess_train
    return (guess_train,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Changes that keep what a photo shows.** Data augmentation changes every training photo at random, anew every time the Dataset gives it, in ways that keep its kind. The network then never sees exactly the same photo twice, and details of one version do not help with the next. For photos, three changes are the classics.

        **Mirror image.** Left and right swapped: a *horizontal flip*. A car that drives to the right becomes a car that drives to the left, and real photos show both. A photo of shape (3, 32, 32) holds its columns on axis 2, so `x.flip(dims=[2])` gives the mirror image: the columns in the other order. Half of the photos get mirrored, at random. `x.flip(dims=[1])` puts the rows in the other order: upside down. That is still a car, but one that real photos hardly ever show.

        **Random crop.** The photo goes into a black photo of 40 by 40, 4 pixels from every edge, and a 32 by 32 is cut out of that at a random place: the photo shifted by up to 4 pixels each way, with black where it moved away. The top left corner of the cut can be at rows 0 to 8 and columns 0 to 8: 9 × 9 = 81 places, and from row 4 and column 4, you get the photo itself. `torch.zeros(3, 40, 40)` makes the black photo, as `np.zeros` makes an array of zeros. It is the shift of *Data augmentation*, with black where `torch.roll` brought back what fell off the other side: around a digit, that was empty paper; around a photo, it would be a piece of its other side.

        **Color jitter.** Three small changes of color, each with its own factor, drawn between 0.8 and 1.2. *Brightness*: every number times b, so that black stays black. *Contrast*, how far apart the light and the dark parts lie, as in [Histogram equalization](/course/open/cv-histogram-equalization?lang=en): the distance of every number to the mean of the whole photo, times c. *Saturation*, how strong the colors are: the distance of every color to the gray of its pixel, the mean of its 3 colors, times s. With s at 0, the photo would be gray; above 1, its colors get stronger. After the changes, every number is kept between 0 and 1, the range of a photo: `x.clamp(0, 1)` makes every number below 0 a 0, and every number above 1 a 1.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **horizontal flip** | | a mirror image: left and right swapped | `x.flip(dims=[2])` |
        | **random crop** | | a part of the photo of a fixed size, cut out at a random place; here 32 by 32 out of the photo with a black border of 4 pixels | `padded[:, top:top + 32, left:left + 32]` |
        | **color jitter** | | a small random change of brightness, contrast and saturation | `jitter(x, b, c, s)` |
        | `t.flip(dims)` | flip | the numbers of a tensor in the other order along the axes `dims` | `x.flip(dims=[2])` |
        """),
        L.watch_out("Change only the training photos, and anew every time the Dataset gives one: in its `__getitem__`, not once before training. Measure on the test photos as they are. And count how many training photos it reads right on those as they are too: changed, they say as much about the changes as about the network."),
        mo.md(r"""
        The picture: how many photos your ColorCNN reads right, of its training photos as they are, filled, and of the test photos, open, a dot for each of 5 seeds, with the course's code. Without changes, 40 epochs make it worse at new photos than 20: on average 503 test photos right instead of 528, and 460 to 545 over the 5 seeds, while it reads 99 to 100% of its training photos right. A mirror image alone brings 541 to 591, but its validation loss is lowest after epoch 6 to 11 and rises after that. With a random crop too, it reads only 65 to 67% of its training photos right, and 590 to 607 test photos: its validation loss keeps falling to epoch 36 to 40. The color adds little here, 583 to 619: the ranges overlap. Upside down instead of mirrored, with nothing else, gives 479 to 492: far fewer than the mirror image alone, and fewer even than no changes after 20 epochs. Not every change that keeps the kind helps, and only measuring tells.
        """),
        mo.Html(C.gap_picture({"title": "Read right: training and test photos", "x": "read right (%)",
                               "none": "no changes, {epochs} epochs", "flip": "mirror, {epochs} epochs",
                               "flip+crop": "mirror and crop, {epochs} epochs", "all": "mirror, crop and color, {epochs} epochs",
                               "upside": "upside down instead of mirror, {epochs} epochs", "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Six rows of dots between 40 and 100 percent read right. No changes, 20 epochs: training photos about 95%, test photos about 53%. No changes, 40 epochs: training about 100%, test about 50%. Mirror, 40 epochs: training about 86%, test about 57%. Mirror and crop, 40 epochs: training about 66%, test about 60%. Mirror, crop and color, 40 epochs: training about 66%, test about 60%. Upside down instead of mirror, 40 epochs: training about 78%, test about 49%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Data augmentation for photos:

    1. Write each change as a function of one photo of shape (3, 32, 32): mirror, crop, color.
    2. In the `__getitem__` of a Dataset of the training photos: draw at random whether to mirror, where to cut, and the three factors of the color; change the photo, and keep every number between 0 and 1.
    3. Normalize with the numbers of the training photos as they are; the test photos are only normalized.
    4. Train for longer: from changed photos, a network learns more slowly, and learns by heart later.
    5. Compare with a run without changes: how many test photos it reads right, and how many training photos as they are.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell loads CIFAR-10 and takes test photo 82, a car, as floats from 0 to 1. It mirrors the car and prints the first 4 numbers of row 10 in red, and the last 4 of row 10 of the mirror image. It puts the car in a black photo of 40 by 40 and cuts 32 by 32 out of that from row 0 and column 8, and prints both shapes, and the pixel at row 4 and column 0 of the cut next to the pixel at row 0 and column 4 of the car. Then it makes the car brighter, and prints the largest number before and after, and how many numbers are above 1. The line of the mirror image is not right yet: it turns the car upside down. Run the cell as it is, then change it where the comment says, and run it again. `cifar` is the course's `everything_ai.cifar`, which the cell below gives once CIFAR-10 is on your computer.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 is not on your computer yet. If you did [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once from the University of Toronto, 170 MB, unpacks its photos into one file of 184 MB, and then deletes the download. Its page states no license, so the course does not ship it; where its photos come from is in that project lesson. The worked example, the picture of the changes and the demo need it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading CIFAR-10** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of CIFAR-10** · this takes a few seconds",
        "no_net": "*The download did not work: the University of Toronto could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "damaged": "*Unpacking did not work: the downloaded file turned out to be damaged, so the course deleted it. Click again to download it again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 170 MB and the unpacked photos, 184 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*CIFAR-10 is on your computer: 50,000 training photos and 10,000 test photos.*",
    })
    return (cifar,)


@app.cell
def example_changes(cifar):
    import torch

    X_train, y_train, X_test, y_test = cifar.load()
    x = torch.tensor(X_test[82], dtype=torch.float32) / 255
    print(cifar.KINDS[y_test[82]], x.shape)
    # a mirror image: the columns, axis 2, in the other order
    mirrored = x.flip(dims=[1])  # change: dims=[2]
    print(x[0, 10, :4])
    print(mirrored[0, 10, -4:])
    # a crop: the car in a black photo of 40 by 40, then 32 by 32 from row 0 and column 8
    padded = torch.zeros(3, 40, 40)
    padded[:, 4:36, 4:36] = x
    cut = padded[:, 0:32, 8:40]
    print(padded.shape, cut.shape)
    print(cut[0, 4, 0], x[0, 0, 4])
    # brighter: every number times 1.2
    brighter = x * 1.2
    print(x.max(), brighter.max(), (brighter > 1).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first 4 numbers of row 10 in red are 0.2667, 0.3098, 0.3451 and 0.3647. As it is, the cell prints 0.3804, 0.5451, 0.6392 and 0.6392 for the end of row 10 of the "mirror image": with `dims=[1]`, the rows are in the other order, so its row 10 is row 21 of the car. With `dims=[2]`, it prints 0.3647, 0.3451, 0.3098 and 0.2667: the first 4, in the other order. The black photo is 40 by 40, and the cut 32 by 32 again. Its pixel at row 4 and column 0 is the car's pixel at row 0 and column 4, 0.2235: the car moved 4 pixels down and 4 to the left. The largest number is 0.9882, and 1.1859 in the brighter car, where 57 of its 3,072 numbers are above 1, lighter than white. That is why the Dataset keeps every number between 0 and 1 after the changes.
    """)
    return


@app.cell(hide_code=True)
def variants_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([
        mo.md("The car of the worked example, as it is and changed: mirrored; brighter with more contrast; darker with less color; and upside down, which real photos hardly show."),
        mo.Html(C.variants_picture({"as_is": "as it is", "mirrored": "mirrored", "brighter": "b 1.2 · c 1.2", "duller": "b 0.8 · s 0.8",
                                    "upside": "upside down",
                                    "alt": "Five small photos of a blue car on grass. As it is, facing left. Mirrored, facing right. Brighter with more contrast. Darker with paler colors. Upside down."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_before_plane():
    mo.md(r"""
    Where does a random crop cut? Drag the dot, or use the arrow keys: it moves the car up to 4 pixels each way. On the left, the car in its black photo of 40 by 40, with a frame around the 32 by 32 that is cut out; on the right, that cut, as the network gets it.
    """)
    return


@app.cell(hide_code=True)
def plane_crop_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    _frames = C.crop_frames({"whole": "black photo, 40 by 40", "cut": "cut from row {top}, column {left}",
                             "alt": "The car in a black photo of 40 by 40, with a frame around the 32 by 32 cut out from row {top} and column {left}; next to it, that cut."})

    def _moved(_n, _way):
        return f"{abs(_n)} {'pixel' if abs(_n) == 1 else 'pixels'} {_way}"

    def _frame(_right, _up):
        _right, _up = int(_right), int(_up)
        _top, _left = 4 + _up, 4 - _right
        _ways = ([_moved(_up, "up" if _up > 0 else "down")] if _up else []) + ([_moved(_right, "to the right" if _right > 0 else "to the left")] if _right else [])
        _said = " and ".join(_ways) if _ways else "nowhere: this is the car itself"
        return _frames[(_right, _up)], f"Cut from row {_top} and column {_left}: the car moved {_said}."

    plane_crop = L.plane("crop-place", "pixels to the right", "pixels up", _frame, C.SHIFTS, C.SHIFTS, start=(0, 0), role="",
                         task="Move the car 4 pixels down and 4 to the left, as in the worked example. From which row and column is it cut?")
    plane_crop
    return


@app.cell(hide_code=True)
def text_after_plane():
    mo.md(r"""
    4 pixels down and 4 to the left is the cut from row 0 and column 8, the one of the worked example: the frame sits at the top right of the black photo, so the black comes in at the top and on the right. Wherever the frame is, the cut is 32 by 32, and the black band is never wider than 4 pixels: the car loses at most 4 of its 32 rows and 4 of its 32 columns. In the middle, from row 4 and column 4, the cut is the car itself.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which changes help?

    Six changes you could make to a training photo of CIFAR-10. Put each in its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_changes():
    sort_changes = L.sort("photo-changes")
    sort_changes
    return (sort_changes,)


@app.cell(hide_code=True)
def check_changes(sort_changes):
    result_changes = L.check("photo-changes", sort_changes)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "photo-changes",
        [
            "For each change, ask two things: is it still the same kind, and could a real photo look like this?",
            "Three changes keep the kind and look like real photos. A car on its roof is still a car: but how often does a photo show one?",
        ],
        why="**Keeps the kind, and looks like a real photo:** a mirror image, a shift of 4 pixels, a little brighter. **Can lose what it shows, or looks unlike a real photo:** upside down, turned a quarter, one corner enlarged. Which changes help depends on the photos: for photos taken from above, such as of fields from a satellite or of cells under a microscope, upside down and a quarter turn look as real as the photo itself.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_crop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A random crop

        Complete `crop(x, top, left)`: put the photo `x` in a black photo of 40 by 40, at rows and columns 4 to 35, and give back the 32 by 32 of it that starts at row `top` and column `left`. The demo's Dataset calls it with `top` and `left` drawn at random from 0 to 8. The check gives your function a made-up photo, and looks at the shape of what comes back, at the rows and columns that should be black, and at where a pixel of the photo lands. When it is right, it saves your function in your project, as `work/projects/cnn/crop.py`.
        """),
        L.task("Make `crop` cut 32 by 32 out of the photo with a black border of 4 pixels."),
    ])
    return


@app.cell(hide_code=True)
def exercise_crop():
    box_crop = L.code_box("crop", lines=8)
    box_crop
    return (box_crop,)


@app.cell(hide_code=True)
def check_crop(box_crop):
    result_crop = L.check("crop", box_crop.value)
    result_crop
    return (result_crop,)


@app.cell(hide_code=True)
def hints_crop():
    L.hints(
        "crop",
        [
            "First the black photo: `torch.zeros` with the shape (3, 40, 40). The photo goes into it 4 pixels from every edge: rows 4 to 35 and columns 4 to 35, for all 3 colors.",
            "Then cut with slices: 32 rows from `top`, `top:top + 32`, and 32 columns from `left` in the same way. A `:` in front keeps all 3 colors.",
        ],
        why="torchvision's `RandomCrop(32, padding=4)` does the same, with black as its standard fill, and the ResNet paper on CIFAR-10 also padded 4 pixels on each side and cut out 32 by 32. Some versions fill the border with the pixels at the edge of the photo, or with its mirror image, instead of black.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_color():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Color jitter

        Complete `jitter(x, b, c, s)`: change the brightness with factor `b`, then the contrast with `c`, then the saturation with `s`, as the comments in the box say. A photo has shape (3, H, W), so its mean over axis 0 is the gray of each pixel, of shape (H, W): subtracting it from `x` takes it off each of the 3 colors, as with NumPy. Keeping the numbers between 0 and 1 is done by the demo's Dataset after your function, with `.clamp(0, 1)`, but you may do it too. The check gives your function made-up photos, and compares what comes back with the course's. When it is right, it saves your function in your project, as `work/projects/cnn/color.py`.
        """),
        L.task("Make `jitter` change the brightness, the contrast and the saturation of a photo."),
    ])
    return


@app.cell(hide_code=True)
def exercise_color():
    box_color = L.code_box("color", lines=9)
    box_color
    return (box_color,)


@app.cell(hide_code=True)
def check_color(box_color):
    result_color = L.check("color", box_color.value)
    result_color
    return (result_color,)


@app.cell(hide_code=True)
def hints_color():
    L.hints(
        "color",
        [
            "Brightness is one multiplication. Contrast: take the mean of the whole photo, `m = x.mean()`, and make every distance to it `c` times as large: `(x - m) * c + m`.",
            "Saturation goes the same way as contrast, around another number: the gray of each pixel, the mean over the colors, axis 0.",
        ],
        why="The three changes give the same photo in any order, as long as nothing keeps the numbers between 0 and 1 in between: the check takes any order. torchvision's `ColorJitter` makes these changes too, in a random order, and keeps every number between 0 and 1 after each one; its gray weighs green most and blue least, as eyes see them.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains your `ColorCNN` again on the first 6,000 training photos of CIFAR-10, with your Trainer, but now from a Dataset that changes every photo anew every time it gives one: half of them mirrored, every one cut out with your `crop` at a random place, and changed in color with your `jitter`, with three factors drawn between {C.number(C.LOW, 1, L.lang)} and {C.number(C.HIGH, 1, L.lang)}. Then every number is kept between 0 and 1, and the photo is normalized with the numbers your `channel_stats` gives for the training photos as they are. Its `__getitem__`:

    ```python
    def __getitem__(self, i):
        x = self.X[i].clone()                      # a copy: the training photo itself stays as it is
        u = torch.rand(6)                          # 6 random numbers from 0 to 1
        if u[0] < 0.5:                             # half of the photos
            x = x.flip(dims=[2])
        x = crop(x, int(u[1] * 9), int(u[2] * 9))  # top and left: whole numbers from 0 to 8
        b, c, s = 0.8 + 0.4 * u[3:6]               # three factors from 0.8 to 1.2
        x = jitter(x, b, c, s).clamp(0, 1)
        return (x - self.mean) / self.std, self.y[i]
    ```

    It trains for {C.EPOCHS} epochs, twice as long as in the last lesson, with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests on the first 1,000 test photos, as they are. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10, from the worked example, and exercises 2 and 3, and takes about a minute. While it trains, a chart shows the validation loss after every epoch. In the picture after, the gray line is the validation loss of the same network without changes, one run of the course's with seed 0, for 40 epochs: it leaves the picture at the top, on its way to 4.4. Then you can rewind through your run: the slider goes back to an epoch, and shows what your network guessed then for one test photo of each kind.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your ColorCNN on changed photos")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_augment(train_button, result_crop, result_color):
    result_crop, result_color  # run again when exercise 2 or 3 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your network.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your ColorCNN on changed photos", "x": "epoch", "y": "validation loss",
                                        "alt": "The validation loss of your ColorCNN on the 1,000 test photos after every epoch so far."}, _p["epochs"], low=0.0)
            if "val_loss" in _p:
                _now["chart"].add(_p["epoch"], _p["val_loss"], f"**Your ColorCNN** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Your ColorCNN** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `crop` and your `jitter` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button in the worked example, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `crop`, your `jitter` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "crop": "*Your `crop` in `work/projects/cnn/crop.py` stops with an error, or does not give back a photo of shape (3, 32, 32). Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "color": "*Your `jitter` in `work/projects/cnn/color.py` stops with an error, or does not give back a photo of shape (3, 32, 32). Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
                "cnn": "*Your `ColorCNN` in `work/projects/cnn/color_cnn.py`, or your `channel_stats` in `work/projects/cnn/channels.py`, stops with an error, or does not give what it should. Did you change one of those files? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation", "none": "without changes",
                       "alt": "The loss of your ColorCNN after every epoch: on the changed training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run without changes, which rises out of the picture."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.curves_picture(_result["curves"], _labels, L.lang, at=_e) + C.guesses_picture(_guesses[_e - 1], _result["truth"], L.lang),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: which photos does your network read right first?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _out = mo.vstack(_pic + [
                mo.md(f"""Your network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, as they are, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters.{_turn} Over 5 seeds, with the course's code: 3,917 to 4,024 training photos and 583 to 619 test photos, with the lowest validation loss after epoch 33 to 39; without changes, after 40 epochs, 5,960 to 6,000 training photos and 460 to 545 test photos.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.md(r"""
    ### Exercise 4. What the curves say
    """)
    return


@app.cell(hide_code=True)
def choice_gap_cell():
    choice_gap = L.choice(
        "augment-gap",
        question="Without changes, after 40 epochs, the training loss ended between 0.01 and 0.065, and the validation loss between 4.118 and 4.478. With mirror, crop and color, both end near 1.1: the training loss between 1.1 and 1.136, the validation loss between 1.072 and 1.176. What do the curves say?",
        options={
            "worse": "The network got worse: a training loss of 1.1 is far higher than 0.01",
            "done": "The network is done learning: both losses stay where they are",
            "right": "The network can hardly learn its training photos by heart any more, so what it learns holds for the test photos too",
        },
        correct="right",
        why={
            "worse": "On its training photos, yes. But the loss that counts is the one on new photos: 1.072 to 1.176 against 4.118 to 4.478, and it reads 583 to 619 test photos right, against 460 to 545.",
            "done": "Its validation loss was lowest after epoch 33 to 39 of 40: it kept falling until near the end. From changed photos, a network learns more slowly, which is why the demo trains twice as long as the last lesson.",
            "right": "Every epoch, each photo comes in another version: mirrored or not, shifted, with other colors. Details of one version do not help with the next, so it learns what all versions share, such as the shape of a car. Its training loss stays high, and the validation loss stays close to it.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The random changes in a Dataset of [Data augmentation](/course/open/nn-data-augmentation?lang=en), now for photos: a mirror image, the shift of that lesson as a random crop with a black border, and small random changes of brightness, contrast and saturation. Your `ColorCNN` of [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) now reads more new photos right and learns its training photos less by heart, as the loss curves of [Reading loss curves](/course/open/torch-loss-curves?lang=en) show. Your project `cnn` now has `crop.py` and `color.py`. In the next lesson, *Look inside a CNN: filters and feature maps*, you look at what a trained CNN learned to look for.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_changes, result_crop, result_color, choice_gap):
    L.wrap_up(result_changes, result_crop, result_color, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
