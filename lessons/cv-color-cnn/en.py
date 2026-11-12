# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-color-cnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-color-cnn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) and [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_head():
    quick_head = L.choice(
        "quick-head",
        question="In your ResBlock, f(x) is two convolutions of 3 by 3 from c to c channels, each with `padding=1`. Why the padding?",
        options={
            "speed": "So that the network trains faster",
            "right": "So that f(x) keeps the height and the width of x, and x + f(x) can be added",
            "edges": "So that the pixels at the edge count more than those in the middle",
        },
        correct="right",
        why={
            "speed": "Padding adds work rather than saving it: a border of zeros around every feature map. It is there for the shapes: without it, every convolution of 3 by 3 takes 2 off the height and the width, and x + f(x) no longer fits. See [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).",
            "edges": "The zeros of the padding let a filter sit on the pixels at the edge too, but they do not make those count more. It is there for the shapes: without it, every convolution of 3 by 3 takes 2 off the height and the width, and x + f(x) no longer fits. See [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en).",
            "right": "Without padding, every convolution of 3 by 3 takes 2 off the height and the width. Today, the same padding keeps 32 by 32 through each convolution of your CNN, and only the poolings make it smaller.",
        },
    )
    quick_head
    return (quick_head,)


@app.cell(hide_code=True)
def quick_check_norm():
    quick_norm = L.choice(
        "quick-norm",
        question="Before ResNet-18 sees a photo, every color is scaled: a mean taken off, divided by a standard deviation. Whose numbers are those?",
        options={
            "own": "The photo's own mean and standard deviation",
            "fixed": "0.5 and 0.5, for every model and every color",
            "right": "Those of ImageNet's training photos, the photos it learned from",
        },
        correct="right",
        why={
            "own": "With its own numbers, every photo gets mean 0 in every color, and a red photo no longer looks red. See [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en).",
            "fixed": "Some models do use 0.5, but ResNet-18 learned its weights with ImageNet's numbers, so a new photo needs those. See [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en).",
            "right": "A new photo gets the numbers the model learned its weights with. Today you train a network yourself: then they are the numbers of your own training photos.",
        },
    )
    quick_norm
    return (quick_norm,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your CNNs so far read Fashion-MNIST: gray clothes of 28 by 28, each one in the middle, on black. Real photos are not like that: they have color and a background, and what they show can be anywhere, at any size. **CIFAR-10** is the classic first step there: 60,000 color photos of 32 by 32 pixels in 10 kinds, from airplanes to trucks, put together in 2009 by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton at the University of Toronto. The ResNet paper of your last step tested networks of up to 1,202 layers on it. In this step of your project, you give your CNN color photos as input, and train it on CIFAR-10.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working", "Your CNN on color photos", "Your own one-stage detector",
                            "Your own U-Net", "Final project: ResNet with Grad-CAM", "Final project: U-Net or detector"],
                           4, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_color():
    guess_color = L.guess(
        "color",
        question="Your SmallCNN read 864 to 879 of 1,000 test images of clothes right, after 5 epochs on 6,000 training images. You give it 3 channels in, and train it the same way on 6,000 color photos of CIFAR-10. How many of its 1,000 test photos does it read right?",
        options={
            "same": "About as many: 850 or more",
            "right": "About 550",
            "guess": "About 100: no better than guessing",
        },
        reveal="**About 550.** Over 5 seeds, with the course's code, it reads 548 to 561 test photos right, against 864 to 879 for the clothes. That is far better than guessing, which gets 100 of 1,000, but far below the clothes: the same network, the same training, harder photos. Of its own training photos it reads 66 to 70% right after 5 epochs, against 90 to 91% for the clothes.",
    )
    guess_color
    return (guess_color,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Three channels in.** A color photo is three images on top of each other, red, green and blue: shape (3, 32, 32). A filter of the first convolution then has a 3 by 3 for every color, shape (3, 3, 3). At every place, it multiplies the 27 numbers under it by its weights, adds them all up, the three colors together, and adds its bias: one number. So `nn.Conv2d(3, 8, 3)` still makes 8 feature maps, from filters of 27 weights each. Only the first convolution changes; after it, the network goes on with its 8 feature maps, as before.

        **Normalized with your own training photos.** As ResNet-18 in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en), your network gets every color with a mean taken off and divided by a standard deviation. You train this network yourself, so these are the numbers of your own training photos: per color, over all photos, rows and columns. For the first 6,000 photos of CIFAR-10, the means are `(0.492, 0.482, 0.445)` and the standard deviations `(0.248, 0.245, 0.263)`, for red, green and blue. The test photos get the same numbers, as every new photo does.

        **Why photos are harder.** The clothes of Fashion-MNIST are in the middle, all the same size, on black. In CIFAR-10, a cat can be close by or far away, on the left or on the right, on a sofa or in the grass, and 32 by 32 pixels show little detail. The kinds the network of this lesson mixes up most, over 5 seeds, are bird and deer, airplane and ship, cat and dog, and automobile and truck: kinds with a similar shape, or with a similar background, such as a blue sky and a blue sea.

        **Learning by heart.** With 6,000 photos that differ this much, a network can learn its training photos by heart: details of these photos that do not hold for new ones. That is the overfitting of [Reading loss curves](/course/open/torch-loss-curves?lang=en): the training loss keeps falling, while the validation loss reaches a lowest point and rises again.

        **Where CIFAR-10 comes from.** Its photos come from 80 Million Tiny Images, a much larger set of small photos that were found on the web by searching for words. In 2020, MIT took that set offline, because some of its labels were offensive words, and some of its photos were offensive too. CIFAR-10 was labeled again by hand, with its own 10 kinds, and is still widely used. Its page states no license, so the course does not ship it and shares no model trained on it: the button in the worked example downloads it once, from its makers.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **CIFAR-10** | see-far ten | 60,000 color photos of 32 by 32 pixels in 10 kinds, 50,000 to train on and 10,000 to test with (Krizhevsky, 2009) | `everything_ai.cifar` |
        """),
        L.watch_out("Normalize the test photos with the numbers of the training photos, not with their own: the network learned its weights with those of the training photos. And a mean for each color is over three axes at once, the photos, the rows and the columns: over the rows and the columns only, you get a mean for every photo."),
        mo.md(r"""
        The picture: how many images the network reads right, of its own training images, filled, and of the test images, open, a dot for each of 5 seeds. On clothes and on color photos, trained for 5 and for 20 epochs, as your SmallCNN with 1 or 3 channels in. On clothes, they stay within about 10 percentage points. On color photos, after 20 epochs, they are more than 40 percentage points apart: the longer training made it better at its own photos, and worse at new ones.
        """),
        mo.Html(C.gap_picture({"title": "Read right: training and test images", "x": "read right (%)",
                               "clothes": "clothes, {epochs} epochs", "photos": "color photos, {epochs} epochs",
                               "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Four rows of dots between 40 and 100 percent read right. Clothes after 5 epochs: training images about 91%, test images about 87%. Clothes after 20 epochs: training about 98%, test about 87%. Color photos after 5 epochs: training about 69%, test about 56%. Color photos after 20 epochs: training about 95%, test about 53%."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A CNN for color photos:

    1. The photos as floats from 0 to 1, of shape (N, 3, 32, 32): the bytes divided by 255.
    2. Per color, the mean and the standard deviation of the training photos, over the photos, the rows and the columns.
    3. Normalize the training photos and the test photos with those numbers: per color, take off the mean and divide by the standard deviation.
    4. The first convolution takes 3 channels; each of its filters has a 3 by 3 for every color.
    5. Work out the sizes up to the output head: with padding 1, each pooling of 2 halves 32, to 16 and then to 8.
    6. Train, and compare the training photos with the test photos: how many it reads right of each, and where the validation loss starts to rise again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell loads CIFAR-10, and prints the shapes of its photos and the kinds of the first 8 test photos. It takes one frog, step 1, and prints its mean per color, over its rows and columns: step 2, for one photo. It puts the frog through a convolution from 3 to 8 channels, step 4, and prints the shapes of its filters and of what comes out. Then it works out one number of the first feature map by hand, at row 10 and column 20, from the 3 by 3 of every color around that place: with padding 1, rows 9 to 11 and columns 19 to 21. The line by hand is not done yet: it takes only the red channel. Run the cell as it is, then complete it where the comment says, and run it again. `cifar` is the course's `everything_ai.cifar`, which the cell below gives once CIFAR-10 is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 is not on your computer yet. The button downloads it once from the University of Toronto, 170 MB, unpacks its photos into one file of 184 MB, and then deletes the download. The worked example, the picture of its photos and the demo need it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
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
def example_filter(cifar):
    import torch
    from torch import nn

    X_train, y_train, X_test, y_test = cifar.load()
    print(X_train.shape, X_train.dtype, X_test.shape)
    print([cifar.KINDS[k] for k in y_test[:8]])
    # 1. a frog as floats from 0 to 1, with an axis for the batch in front
    i = cifar.checked("frog", 1)[0]
    x = torch.tensor(X_test[i:i + 1], dtype=torch.float32) / 255
    print(x.shape)
    # 2. its mean per color, over the rows and the columns
    print(x[0].mean(dim=(1, 2)))
    # 4. a convolution from 3 to 8 channels
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, padding=1)
    with torch.no_grad():
        maps = conv(x)
    print(conv.weight.shape, maps.shape)
    # by hand: filter 0 on the 3 by 3 around row 10, column 20
    patch = x[0, :, 9:12, 19:22]
    by_hand = (patch[0] * conv.weight[0, 0]).sum() + conv.bias[0]  # complete: (patch * conv.weight[0]).sum() + conv.bias[0]
    print(round(maps[0, 0, 10, 20].item(), 4), round(by_hand.item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The photos are bytes, `uint8`, with the colors first: 50,000 training photos and 10,000 test photos of shape (3, 32, 32). The first 8 test photos are a cat, two ships, an airplane, two frogs, an automobile and a frog; photo 4 is the first frog. Its mean is 0.423 for red, 0.479 for green and 0.362 for blue: a greenish frog. Exercise 2 takes the same mean over all photos at once. The 8 filters together have shape (8, 3, 3, 3): a 3 by 3 for each of the 3 colors. What comes out is 8 feature maps of 32 by 32, as padding 1 keeps the size. At row 10 and column 20, the first map holds −0.2426. As it is, with only the red channel, the line by hand gives −0.1844. Completed, with all three colors, it gives −0.2426 too: one filter adds up what it sees in every color.
    """)
    return


@app.cell(hide_code=True)
def photos_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([
        mo.md("The first 4 photos of every kind that the course looked at one by one, enlarged so that you see every pixel. Some photos of CIFAR-10 show a person next to the animal or vehicle, such as a rider on a horse; the lesson shows only photos without people."),
        mo.Html(C.photos_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The weights of a color filter
    """)
    return


@app.cell(hide_code=True)
def choice_weights_cell():
    choice_weights = L.choice(
        "color-weights",
        question="How many parameters does `nn.Conv2d(3, 8, 3)` have, the first convolution of a CNN for color photos?",
        options={
            "gray": "80: 8 filters of 3 by 3, and 8 biases",
            "right": "224: 8 filters of 3 by 3 by 3, and 8 biases",
            "apart": "240: 8 filters of 3 by 3 for every color, 24 in all, and 24 biases",
        },
        correct="right",
        why={
            "gray": "That is `nn.Conv2d(1, 8, 3)`, for gray images. With 3 channels in, every filter has a 3 by 3 for each color: 27 weights.",
            "apart": "A filter does not look at one color: it covers all three at once, and adds them up into one feature map. So 8 filters make 8 feature maps, not 24.",
            "right": "8 × 3 × 3 × 3 = 216 weights, in a tensor of shape (8, 3, 3, 3), and 8 biases: 224. Color costs the first convolution 3 times the weights, and the rest of the network nothing.",
        },
    )
    choice_weights
    return (choice_weights,)


@app.cell(hide_code=True)
def text_exercise_stats():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The numbers of each color

        Complete `channel_stats(X)`: for photos `X` of shape (N, 3, H, W), give back the mean and the standard deviation of each color, over all photos, rows and columns, step 2 of the recipe. Both are tensors of shape (3,). PyTorch's `.mean` and `.std` take several axes at once, as a tuple: `X.mean(dim=(1, 2))` takes the mean over axes 1 and 2 together. The check gives your function made-up photos, and compares the shapes and the numbers. When it is right, it saves your function in your project, as `work/projects/cnn/channels.py`: the demo normalizes the photos with it.
        """),
        L.task("Make `channel_stats` give back the mean and the standard deviation of each color."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stats():
    box_stats = L.code_box("channel-stats", lines=8)
    box_stats
    return (box_stats,)


@app.cell(hide_code=True)
def check_stats(box_stats):
    result_stats = L.check("channel-stats", box_stats.value)
    result_stats
    return (result_stats,)


@app.cell(hide_code=True)
def hints_stats():
    L.hints(
        "channel-stats",
        [
            "In `X`, axis 0 holds the photos, axis 1 the colors, and axes 2 and 3 the rows and the columns. Keep the colors: take the mean over all the other axes.",
            "The axes go into `dim` as a tuple. The standard deviation goes the same way as the mean, with `.std`. Give back both, with a comma between them.",
        ],
        why="PyTorch's `.std` divides by N − 1, as the standard deviation of a sample; with `unbiased=False`, it divides by N. Over 6,000 photos of 32 by 32, each color has 6,144,000 numbers, and the two differ only in the eighth decimal: the check takes either.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_cnn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your CNN in color

        `ColorCNN` is your SmallCNN, with two numbers left out: fill them in for color photos of 32 by 32, steps 4 and 5 of the recipe. The check makes networks after `torch.manual_seed(0)` and compares the logits of 2 and of 3 random photos, and how many numbers its parameters hold. When it is right, the check saves it in your project, as `work/projects/cnn/color_cnn.py`.
        """),
        L.task("Make `ColorCNN` take color photos of 32 by 32."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cnn():
    box_cnn = L.code_box("color-cnn", lines=17)
    box_cnn
    return (box_cnn,)


@app.cell(hide_code=True)
def check_cnn(box_cnn):
    result_cnn = L.check("color-cnn", box_cnn.value)
    result_cnn
    return (result_cnn,)


@app.cell(hide_code=True)
def hints_cnn():
    L.hints(
        "color-cnn",
        [
            "The first number of `nn.Conv2d` is the number of channels that come in. How many does a color photo have?",
            "With padding 1, the convolutions keep 32 by 32, and each pooling halves it. How big are the 16 feature maps after the second pooling, and how many numbers do they hold together?",
        ],
        why="16 channels of 8 by 8 are 1,024 numbers per photo, against 784 for the clothes. The output head then has 10,250 of the network's 11,642 parameters: nearly 9 of every 10.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It takes the first 6,000 training photos of CIFAR-10 and the first 1,000 test photos, as many as with the clothes, and normalizes both with the numbers your `channel_stats` gives for the training photos. It trains your `ColorCNN` with your Trainer: {C.EPOCHS} epochs, a learning rate of {C.number(C.LR, 2, L.lang)} with a warmup of one epoch, with seed 0, and counts its parameters with your `count_params`. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10, from the worked example, and exercises 2 and 3, and takes about half a minute. While it trains, a chart shows the loss on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch. Beside or under it are your network's guesses at that moment for one test photo of each kind, and under them the feature maps of its first layer on a car; both are drawn again as it trains. Guessing among 10 kinds gives a loss of $\\ln 10 \\approx 2.30$. By epoch 20, the validation loss can be as high as that, while the network still reads about half of the test photos right: on the photos it gets wrong, it is very sure of a wrong kind, and that costs more than a guess. Then you can rewind to a few epochs of your run, with both pictures as they were then.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your ColorCNN")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_color(train_button, result_stats, result_cnn):
    result_stats, result_cnn  # run again when exercise 2 or 3 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your network.*")
    else:
        _now = {}   # the live chart, made at the first report
        _layer = "Layer {name}, the first of your ColorCNN, on a test photo of a car"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your ColorCNN on CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your ColorCNN so far: on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-color-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your ColorCNN** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your ColorCNN** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Your ColorCNN** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `channel_stats` and your `ColorCNN` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button in the worked example, and press this button again.*",
                "slow": "*Your project takes too long here: more than five minutes to train. Does your `ColorCNN` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "stats": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give a mean and a standard deviation above 0 for each of the 3 colors. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "cnn": "*Your `ColorCNN` in `work/projects/cnn/color_cnn.py` stops with an error, or does not give 10 logits per photo. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "alt": "The loss of your ColorCNN after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest."}
            _val, _looked = _result["curves"]["val"], _result["looked"]
            _when = C.moments(len(_val)) if len(_looked) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_looked[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.ONE_EACH)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: which photos does your network read right first?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" The validation loss was lowest after epoch **{_best}**; after that it rose, while the training loss kept falling." if _best < C.EPOCHS else f" The validation loss was still falling after epoch {_best}.")
            _numbers = ", ".join(C.number(_v, 3, L.lang) for _v in _result["mean"]), ", ".join(C.number(_v, 3, L.lang) for _v in _result["std"])
            _right = sum(_g == _k for _, _k, _g in _result["guesses"])
            _out = mo.vstack(_pic + [
                mo.md(f"""Your network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters, counted with your `count_params`.{_turn} Over 5 seeds, with the course's code: 5,514 to 5,800 training photos and 515 to 543 test photos, with the lowest validation loss after epoch 3 to 6. Your `channel_stats` gave the means ({_numbers[0]}) and the standard deviations ({_numbers[1]}).{_who}"""),
                mo.md(f"Below, {C.SHOWN} checked test photos of every kind, with your network's guess: {_right} of the {len(_result['guesses'])} are right. Where it is wrong, the right kind is in brackets."),
                mo.Html(C.guesses_picture(_result["guesses"], L.lang)),
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
        "color-gap",
        question="With the course's code, the validation loss is lowest after epoch 3 to 6, and then rises to between 2.179 and 2.386 by epoch 20, while the training loss keeps falling, to between 0.164 and 0.305. What is happening?",
        options={
            "rate": "The learning rate is too high, so the network gets lost",
            "harder": "The test photos are harder than the training photos",
            "right": "The network learns its training photos by heart: details that do not hold for new photos",
        },
        correct="right",
        why={
            "rate": "A learning rate that is too high makes the training loss jump about or rise too. Here it keeps falling, and after the warmup, the cosine decay of your Trainer makes the learning rate smaller with every step.",
            "harder": "Both are photos of the same 10 kinds, from the same set; with seed 0, after epoch 3, the two losses were nearly equal, 1.34 and 1.38. What changes is the network, not the photos.",
            "right": "After epoch 3 to 6, it learns more about its 6,000 photos than about the 10 kinds: it gets more of its training photos right and fewer new ones. More photos, or training photos that differ every epoch, make learning by heart harder.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your SmallCNN of [Project: your first CNN](/course/open/cnn-pytorch?lang=en), now with three channels in, and normalized with the numbers of its own training photos, as ResNet-18 was with ImageNet's in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en). On color photos, the gap between training and test images of [Reading loss curves](/course/open/torch-loss-curves?lang=en) grows from about 10 percentage points on clothes to more than 40. Your project `cnn` now has `channels.py` and `color_cnn.py`. In the next lesson, *Data augmentation for photos: flip, crop and color*, every epoch shows your `ColorCNN` its training photos a little differently: mirrored, cut out, with other colors, so that it cannot learn them by heart as easily.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weights, result_stats, result_cnn, choice_gap):
    L.wrap_up(choice_weights, result_stats, result_cnn, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
