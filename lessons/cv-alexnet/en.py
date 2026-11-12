# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-alexnet. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-alexnet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en) and [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_top5():
    quick_top5 = L.choice(
        "quick-top5",
        question="A network has a top-5 accuracy of 85% on ImageNet's test photos. What does that say?",
        options={
            "top1": "Its best guess is right for 85% of the photos",
            "right": "For 85% of the photos, the label is among its five best guesses",
            "five": "Of its five best guesses for a photo, 85% are right on average",
        },
        correct="right",
        why={
            "top1": "That is the top-1 accuracy. The top-5 accuracy forgives more: a photo counts as right when the label is anywhere on the short list of five. See [Try an image classifier](/course/open/cv-try-classifier?lang=en).",
            "five": "Of five guesses, at most one can be the label, so at most one of every five is ever \"right\". The top-5 accuracy counts photos: those whose label is among the five. See [Try an image classifier](/course/open/cv-try-classifier?lang=en).",
            "right": "ImageNet's contest gave it the other way around, as the top-5 error: 1 minus the top-5 accuracy, here 15%, the photos whose label is not among the five. It was scored this way because a photo often shows more than one thing, and only one of them is its label.",
        },
    )
    quick_top5
    return (quick_top5,)


@app.cell(hide_code=True)
def quick_check_changes():
    quick_changes = L.choice(
        "quick-changes",
        question="In *Data augmentation for photos*, the Dataset changed each training photo in its `__getitem__`: mirrored or not, cut out at a random place, with slightly other colors. Why there?",
        options={
            "memory": "So that the changed photos need no memory of their own",
            "test": "So that the test photos are changed in the same way",
            "right": "So that every epoch shows each photo in another version",
        },
        correct="right",
        why={
            "memory": "That is a bonus, which AlexNet's paper names too, but it is not the reason. Changed once, before training, every epoch would show the same changed photo, and the network could learn that one by heart. See [Data augmentation for photos](/course/open/cv-augment-photos?lang=en).",
            "test": "The test photos are not changed at all: they measure, as they are. Only the training photos go through that `__getitem__`. See [Data augmentation for photos](/course/open/cv-augment-photos?lang=en).",
            "right": "`__getitem__` runs every time the DataLoader asks for a photo, so every epoch draws new changes. AlexNet did the same in 2012, and today you train with those changes again, now with dropout too.",
        },
    )
    quick_changes
    return (quick_changes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        From 2010 on, ImageNet held a contest every year: give five guesses of the class of each test photo, out of 1,000 classes, scored by the top-5 error: the share of test photos whose class is not among the five guesses. In 2010 and 2011, the winners measured features designed by hand, such as how strongly the edges in small squares point in each direction, and put a linear classifier on top: a top-5 error of 28.2% in 2010, and 25.8% in 2011. In 2012, Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton got 15.3% with CNNs, and the second place, with hand-designed features, 26.2%. In 2013, most entries were CNNs, and the error kept falling: VGG and GoogLeNet in 2014, and ResNet in 2015, at 3.6%. You built VGG and ResNet already. Today you go back to the CNN that started it, AlexNet: you count where its 61 million parameters are, build a small version of it for CIFAR-10, and measure what its dropout and its changes of the photos still add.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params-where",
        question="AlexNet has 61 million parameters, in five convolutions and three linear layers after them. Where are most of them?",
        options={
            "conv": "In the five convolutions: they do the looking",
            "even": "About evenly spread over the eight layers",
            "right": "In the three linear layers at the end",
        },
        reveal="**In the three linear layers.** Of its 60,965,224 parameters, 58,631,144 are in its three linear layers, 96.2%; `fc6` alone, from the 9,216 numbers after the last pooling to 4,096, has 37,752,832, 61.9%. Its five convolutions together have 2,334,080, 3.8%. The slider after the worked example goes through them layer by layer.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The contest.** ImageNet's contest is ILSVRC, the ImageNet Large Scale Visual Recognition Challenge. Its training set held 1.2 million photos, and its score was the top-5 error. The picture shows the lowest top-5 error of each year, on the test photos; the open dot is the second place of 2012. The lowest of 2012 and of 2013 came from entries that also trained on photos from outside the contest; without those, 16.4% and 11.7%.
        """),
        mo.Html(C.ilsvrc_picture({"title": "Lowest top-5 error in ILSVRC", "y": "top-5 error (%)", "second": "second, 2012", 2012: "AlexNet",
                                  "alt": "Bars of the lowest top-5 error in ILSVRC each year: 28.2% in 2010, 25.8% in 2011, 15.3% in 2012, marked AlexNet, 11.2% in 2013, 6.7% in 2014 and 3.6% in 2015. An open dot at 26.2% for the second place of 2012."}, L.lang)),
        mo.md(r"""
        **Eight layers with weights.** AlexNet takes a photo of 224 by 224 through five convolutions and three linear layers, with a ReLU after each but the last. `conv1` has 96 filters of 11 by 11 with stride 4, and gives 96 maps of 55 by 55. `conv2` has 256 filters of 5 by 5; `conv3` and `conv4` 384 of 3 by 3, and `conv5` 256 of 3 by 3. Max pooling of 3 by 3 with stride 2 follows `conv1`, `conv2` and `conv5`: its squares overlap, where those of the pooling of 2 by 2 of your SmallCNN do not. After the last pooling, 256 maps of 6 by 6 are left, 9,216 numbers. `fc6` and `fc7` turn them into 4,096 numbers each, and `fc8` gives a logit for each of the 1,000 classes.

        **Two GPUs.** Their GPU, a GTX 580, had 3 GB of memory, too little for this network. So it was split over two GPUs, half of the filters of every layer on each, and the filters of `conv2`, `conv4` and `conv5` look only at the maps on their own GPU: each filter of `conv2` sees 48 of the 96 maps. Training took five to six days, for about 90 passes over the 1.2 million photos, with SGD with momentum 0.9, weight decay of 0.0005, mini-batches of 128 and a learning rate of 0.01, divided by 10 three times.

        **What it used, and what the paper says each brought.**
        - **ReLU** instead of tanh, the S-shaped curve of LeNet-5: a small CNN of four layers on CIFAR-10 reached a training error of 25% six times as fast with ReLU as with tanh. The paper adds that the size of that effect depends on the network.
        - **Local response normalization** after `conv1` and `conv2`: each number divided by a number that grows with the squares of the numbers at the same place in neighboring maps. It lowered the top-1 and top-5 error by 1.4 and 1.2 percentage points.
        - **Overlapping pooling**, 3 by 3 with stride 2 instead of 2 by 2 with stride 2: 0.4 and 0.3 percentage points.
        - **Dropout** of 0.5 after `fc6` and `fc7`: without it, the paper says, the network learns its photos by heart, and with it, it needs about twice as many steps to converge.
        - **Changes of the photos**: a random 224 by 224 cut out of a photo of 256 by 256, mirrored half of the time, and a small change of color. Without the cuts and mirror images, the paper says, the network would learn its photos by heart, and only much smaller networks would have worked; the change of color lowered the top-1 error by more than 1 percentage point. At test time, it averaged its guesses over 10 cuts: the four corners and the middle, and their mirror images.

        The 15.3% did not come from one AlexNet. One CNN got 18.2% on the validation photos; the guesses of five CNNs averaged, 16.4%; and of seven, two of them first trained on 15 million photos of 22,000 classes, 15.4%, and 15.3% on the test photos.
        """),
        L.watch_out("A score in a paper is the score of a setup: one network or several averaged, which photos it was trained on, one cut at test time or ten. Before you compare two numbers, check that they come from the same setup. AlexNet's 15.3% was seven networks; one network got 18.2% on the validation photos, which in the paper differ from the test photos by at most 0.1 percentage point: still far below the 26.2% of the second place."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **ILSVRC** | ImageNet's yearly contest, from 2010 to 2017: five guesses for each test photo, scored by top-5 error | |
        | **AlexNet** | the CNN of Krizhevsky, Sutskever and Hinton that won ILSVRC in 2012: five convolutions, three linear layers, ReLU, dropout and changes of the photos | |
        | **local response normalization** | each number of a map divided by a number that grows with the squares of the numbers at the same place in neighboring maps | |

        **A small AlexNet.** The demo trains AlexNet's eight layers at the size of CIFAR-10, `SmallAlexNet` of exercise 2: filters of 5 by 5 in `conv1` and `conv2`, then three of 3 by 3, each with padding so that a convolution keeps the size it gets: 32 in `conv1`, 15 in `conv2` and 7 in the other three; and the same pooling of 3 by 3 with stride 2, after `conv1`, `conv2` and `conv5`; then three linear layers, with dropout after `fc6` and `fc7`. It has 201,402 parameters, 17 times as many as your ColorCNN, and 70.7% of them are in its linear layers. It leaves out local response normalization.

        The picture: how many photos SmallAlexNet reads right after 40 epochs, of its training photos as they are, filled, and of the test photos, open, a dot for each of 5 seeds, with the course's code; under it, in gray, your ColorCNN of [Data augmentation for photos](/course/open/cv-augment-photos?lang=en), also after 40 epochs. Each trains at a learning rate that works for it: 0.01 for ColorCNN, and 0.001 for SmallAlexNet, which hardly learns at 0.01. Without dropout and changes, SmallAlexNet reads 5,907 to 5,977 of its 6,000 training photos right, and 541 to 576 test photos; its validation loss is lowest after epoch 11 to 16, and then rises again. With dropout: 542 to 575 test photos, about as many, and 5,482 to 5,603 training photos. With the changes: 594 to 636 test photos and 4,076 to 4,221 training photos, with the lowest validation loss after epoch 29 to 36. With both, as AlexNet: 581 to 604 test photos, and 3,871 to 3,953 training photos. Your ColorCNN read 583 to 619 with the changes: the ranges overlap, so on 6,000 photos of 32 by 32, 17 times as many parameters bring little. Exercise 3 asks what the numbers say about dropout.
        """),
        mo.Html(C.runs_picture({"title": "Read right after 40 epochs", "x": "read right (%)", "none": "no dropout, no changes", "dropout": "dropout",
                                  "changes": "changes", "both": "dropout and changes", "color_none": "ColorCNN, no changes",
                                  "color_all": "ColorCNN, changes", "mine": "your run", "train": "training", "test": "test",
                                  "alt": "Six rows of dots between 40 and 100 percent read right, after 40 epochs. SmallAlexNet without dropout and changes: training photos about 99%, test photos about 56%. With dropout: training about 93%, test about 56%. With the changes: training about 69%, test about 62%. With dropout and changes: training about 65%, test about 59%. In gray, ColorCNN without changes: training about 100%, test about 50%; with the changes: training about 66%, test about 60%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To build a CNN after a paper:

    1. For each convolution with filters of k by k, stride s and padding p: n rows and columns become (n + 2p − k) // s + 1. It has (maps it sees × k × k + 1) parameters per filter.
    2. For max pooling of 3 by 3 with stride 2: n becomes (n − 3) // 2 + 1.
    3. The first linear layer gets maps × rows × columns numbers: everything after the last pooling, as one row per photo.
    4. Check in PyTorch: send a batch of zeros through the layers and print `.shape`; count with `.numel()`.
    5. Put dropout after the ReLU of each linear layer but the last, train on changed photos, and measure in eval mode.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell builds `conv1` of AlexNet and its pooling, sends one photo of zeros of 224 by 224 through them, and prints the shapes. Then it builds `fc6`, and counts the parameters of `conv1` and of `fc6`. The line of `conv1` is not right yet: the paper's maps of 55 by 55 do not come out. Run the cell as it is, then change it where the comment says, and run it again. It needs no download.
    """)
    return


@app.cell
def example_layers():
    import torch
    from torch import nn

    x = torch.zeros(1, 3, 224, 224)
    # 1. conv1: 96 filters of 11 by 11 with stride 4
    conv1 = nn.Conv2d(3, 96, 11, stride=4)  # change: add padding=2
    # 2. max pooling of 3 by 3 with stride 2: the squares overlap
    pool = nn.MaxPool2d(3, stride=2)
    h = conv1(x)
    print(h.shape)
    print(pool(h).shape)
    # 3. fc6: from 256 maps of 6 by 6 to 4,096 numbers
    fc6 = nn.Linear(256 * 6 * 6, 4096)
    # 4. the parameters: weights and biases
    print(conv1.weight.numel() + conv1.bias.numel())
    print(sum(q.numel() for q in fc6.parameters()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    As it is, the cell prints `torch.Size([1, 96, 54, 54])` and `torch.Size([1, 96, 26, 26])`: (224 − 11) // 4 + 1 = 54, and (54 − 3) // 2 + 1 = 26. With `padding=2`, it prints 55 and 27: (224 + 4 − 11) // 4 + 1 = 55, as in the paper. The paper gives its photos as 224 by 224 and the maps of `conv1` as 55 by 55, which works out with 2 pixels of padding on each side, as torchvision's AlexNet has, or with photos of 227 by 227. `conv1` has 96 × (3 × 11 × 11 + 1) = 34,944 parameters, and `fc6` 9,216 × 4,096 + 4,096 = 37,752,832: more than a thousand times as many.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Slide through AlexNet's eight layers. The bars are their parameters, in millions, with the share of all after each; the layers up to the one you choose are orange, and the bar at the bottom adds them up, with a dashed line at 90% of all.
    """)
    return


@app.cell(hide_code=True)
def play_layers():
    _labels = {"title": "AlexNet's parameters, layer by layer", "x": "parameters (millions)", "so_far": "up to here", "line": "90%",
               "alt": "Bars of the parameters of AlexNet's eight layers, in millions: conv1 to conv5 less than 1 million each, fc6 37.8, fc7 16.8 and fc8 4.1; the layers up to the chosen one in orange, and under them a bar of those layers together, with a dashed line at 90% of all."}
    explore_layers = L.explore("layer", "layer", lambda _v: (C.layers_picture(int(_v), _labels, L.lang), C.layer_sentence(int(_v), L.lang)),
                               start=1, values=list(range(1, 9)), role="",
                               task="After which layer do the layers up to there hold more than 90% of all parameters?")
    explore_layers
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    After `fc7`: up to there, 56,868,224 parameters, 93.3%; after `fc6`, 65.8%. The five convolutions together have 3.8%, fewer than `fc8` alone, 6.7%. But for one photo, the convolutions do 665,784,864 of the 724,406,816 multiplications, 91.9%: a filter of `conv1` has 363 weights, and it is used at every one of the 55 × 55 = 3,025 places of its map, while `fc6` uses each of its weights once. The parameters are at the end, the work is at the start. Later networks dropped the large linear layers, though not at once: VGG, in 2014, still ended in three, with more than 100 million parameters in its first one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_today():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What stayed?

    Six things AlexNet did. ResNet-18, three years later, is the network you looked inside in [Look inside a CNN: filters and feature maps](/course/open/cv-inside-cnn?lang=en). Put each in its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_today():
    sort_today = L.sort("alexnet-today")
    sort_today
    return (sort_today,)


@app.cell(hide_code=True)
def check_today(sort_today):
    result_today = L.check("alexnet-today", sort_today)
    result_today
    return (result_today,)


@app.cell(hide_code=True)
def hints_today():
    L.hints(
        "alexnet-today",
        [
            "Go through ResNet-18 as in *Look inside a CNN*: `conv1`, `bn1`, a ReLU, a max pooling, and four stages of residual blocks. Which of the six do you see there?",
            "Three stayed and three changed. The filters of `conv1` of ResNet-18 are 7 by 7 with stride 2, and after `layer4` it takes the mean of each map. The ResNet paper trained on random cuts of its photos and their mirror images.",
        ],
        why="**ResNet-18 does this too:** a ReLU after its layers with weights, max pooling of 3 by 3 with stride 2 (with padding 1), and training on mirror images and random cuts of its photos. **ResNet-18 does this differently:** filters of 7 by 7 with stride 2 in `conv1`, the size with which Zeiler and Fergus made AlexNet better; batch normalization instead of local response normalization; and the mean of each of the 512 maps of `layer4`, then one linear layer without dropout, instead of three linear layers with dropout.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A small AlexNet

        Complete `SmallAlexNet`: AlexNet's eight layers with weights, at the size of CIFAR-10. Its five convolutions and its pooling are in the box. Fill in the number of inputs of `fc6`: how many numbers each photo has after the last pooling. And finish `forward`: `fc6` and `fc7`, each with a ReLU and then dropout, and `fc8`, which gives the 10 logits. `self.drop` is `nn.Dropout(p)` of [Dropout](/course/open/nn-dropout?lang=en): in training mode, it sets each number to 0 with chance p, and in eval mode it does nothing. The check gives your network made-up photos, in eval mode and in training mode, and compares what comes back with the course's. When it is right, it saves your class in your project, as `work/projects/cnn/alexnet.py`.
        """),
        L.task("Give `fc6` the right number of inputs, and finish `forward` with dropout after `fc6` and `fc7`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_small():
    box_small = L.code_box("small-alexnet", lines=28)
    box_small
    return (box_small,)


@app.cell(hide_code=True)
def check_small(box_small):
    result_small = L.check("small-alexnet", box_small.value)
    result_small
    return (result_small,)


@app.cell(hide_code=True)
def hints_small():
    L.hints(
        "small-alexnet",
        [
            "Follow the rows of a photo: each convolution keeps the size it gets, thanks to its padding, and each pooling turns n into (n − 3) // 2 + 1. How many rows are left after three poolings, and how many maps does `conv5` give?",
            "32, then 15, then 7, then 3: `fc6` gets `32 * 3 * 3` numbers. In `forward`, `self.drop(torch.relu(self.fc6(x)))` is `fc6` with its ReLU and its dropout.",
        ],
        why="`fc6` gets 32 maps of 3 by 3, 288 numbers. Dropout comes after the ReLU of `fc6` and of `fc7`, not on the logits: a logit set to 0 would say nothing about its class. torchvision's AlexNet puts its dropout before `fc6` and `fc7` instead: in training it drops numbers of what goes into `fc6` and `fc7`, and here of what comes out of `fc6` and `fc7`. In eval mode, both give the same.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains your `SmallAlexNet` on the first 6,000 training photos of CIFAR-10, with your Trainer, for {C.EPOCHS} epochs with a learning rate of {C.number(C.LR, 3, L.lang)}, a warmup of one epoch and seed 0, and tests it on the first 1,000 test photos. Choose how: with the photos as they are and without dropout; with dropout of {C.number(C.P, 1, L.lang)}; with the changes of [Data augmentation for photos](/course/open/cv-augment-photos?lang=en), your `crop` and your `jitter`, anew every epoch; or with both, as AlexNet did. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10 and exercise 2, and takes a few minutes. While it trains, a chart shows the loss on the training photos every 10 batches, and dashed on the test photos after every epoch. Beside or under it are your network's guesses at that moment for one test photo of each kind, and under them the feature maps of its first layer on a car; both are drawn again as it trains. In the picture after, the gray line is the validation loss of the course's run with seed 0 without dropout and changes; when you train that way yourself, its run with the changes instead. Then you can rewind to a few epochs of your run, with both pictures as they were then, and see your run next to the course's.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 is not on your computer yet. If you did [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once from the University of Toronto, 170 MB, unpacks its photos into one file of 184 MB, and then deletes the download. Its page states no license, so the course does not ship it; where its photos come from is in that project lesson. Only the demo needs it. While it downloads, the whole lesson waits.*",
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


@app.cell(hide_code=True)
def variant_choice_cell():
    variant_choice = mo.ui.radio(options={"no dropout, no changes": "none", "dropout": "dropout", "changes": "changes", "dropout and changes": "both"},
                                 value="dropout and changes", label="Train with")
    train_button = mo.ui.run_button(label="Train your SmallAlexNet")
    mo.vstack([variant_choice, train_button])
    return train_button, variant_choice


@app.cell(hide_code=True)
def demo_alexnet(cifar, train_button, variant_choice, result_small):
    cifar, result_small  # run again when CIFAR-10 arrives, or when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your network.*")
    else:
        _variant = variant_choice.value
        _now = {}   # the live chart, made at the first report
        _layer = "Layer {name}, the first of your SmallAlexNet, on a test photo of a car"

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your SmallAlexNet on CIFAR-10", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your SmallAlexNet so far: on the training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _layer, L.lang, "nv-alexnet-now"))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your SmallAlexNet** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your SmallAlexNet** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Your SmallAlexNet** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_variant, _progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `SmallAlexNet` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button above, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `SmallAlexNet`, your `crop`, your `jitter` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "alexnet": "*Your `SmallAlexNet` in `work/projects/cnn/alexnet.py` stops with an error, or does not give 10 logits for each photo. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "changes": "*Your `crop` in `work/projects/cnn/crop.py` or your `jitter` in `work/projects/cnn/color.py` stops with an error, or does not give back a photo of shape (3, 32, 32). Did you change one of those files? Delete it, and check its exercise in [Data augmentation for photos](/course/open/cv-augment-photos?lang=en) again.*",
                "cnn": "*Your `channel_stats` in `work/projects/cnn/channels.py` stops with an error, or does not give what it should. Did you change its file? Delete it, and check its exercise in [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _other = "changes" if _variant == "none" else "none"
            _names = {"none": "no dropout, no changes", "dropout": "dropout", "changes": "changes", "both": "dropout and changes"}
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "none": "no dropout, no changes", "changes": "changes",
                       "alt": "The loss of your SmallAlexNet after every epoch: on the training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's run with seed 0 " + ("with the changes." if _other == "changes" else "without dropout and changes, which falls to epoch 16 and then rises.")}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _layer, L.lang, other=_other),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: when is the validation loss lowest, and what happens after?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang, other=_other))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _ref = C.REFERENCE[_variant]
            _tr, _te, _be = C.low_high(_ref["train"]), C.low_high(_ref["test"]), C.low_high(_ref["best"])
            _out = mo.vstack(_pic + [
                mo.md(f"""Your network reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, as they are, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters.{_turn} Over 5 seeds, with the course's code: {C.grouped(_tr[0], L.lang)} to {C.grouped(_tr[1], L.lang)} training photos and {_te[0]} to {_te[1]} test photos, with the lowest validation loss after epoch {_be[0]} to {_be[1]}.{_who}"""),
                mo.Html(C.runs_picture({"title": "Read right after 40 epochs", "x": "read right (%)", "none": "no dropout, no changes", "dropout": "dropout",
                                           "changes": "changes", "both": "dropout and changes", "color_none": "ColorCNN, no changes",
                                           "color_all": "ColorCNN, changes", "mine": "your run: " + _names[_variant], "train": "training", "test": "test",
                                           "alt": "Rows of dots between 40 and 100 percent read right: at the top your run, with your training and test photos; under it the course's four runs of SmallAlexNet over 5 seeds, and in gray ColorCNN without and with the changes."}, L.lang, mine={"variant": _variant, "train": _result["train"], "test": _result["test"]})),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_dropout():
    mo.md(r"""
    ### Exercise 3. What dropout adds here
    """)
    return


@app.cell(hide_code=True)
def choice_dropout_cell():
    choice_dropout = L.choice(
        "dropout-here",
        question="Over 5 seeds, after 40 epochs. Without changes, SmallAlexNet reads 541 to 576 test photos right without dropout, and 542 to 575 with dropout; its validation loss ends at 2.61 to 3.45 without dropout, and at 1.87 to 2.27 with. With the changes, it reads 594 to 636 test photos right without dropout, and 581 to 604 with dropout. What do the numbers say about dropout here?",
        options={
            "nothing": "Dropout does nothing here: without changes, it reads as many test photos right",
            "both": "Dropout and changes together work best, as in AlexNet",
            "right": "With the changes, the network no longer learns its photos by heart, and dropout holds it back further: it reads fewer training photos right, and fewer test photos too",
        },
        correct="right",
        why={
            "nothing": "As many photos right, yes, but not nothing: its validation loss rises less, to 1.87 to 2.27 instead of 2.61 to 3.45, so the network gets less sure of its wrong guesses. It just does not get more of them right. And with the changes, it lowers the count: 581 to 604 instead of 594 to 636.",
            "both": "Not here: with both, 581 to 604 test photos; with only the changes, 594 to 636. The paper's network had linear layers of 4,096 numbers and 1.2 million photos; SmallAlexNet has 256 and 6,000.",
            "right": "With the changes alone, it reads 4,076 to 4,221 of its training photos right, with a training loss of 0.88 to 0.94 at the end; with dropout too, 3,871 to 3,953, and 1.02 to 1.09. The paper says dropout about doubles the steps a network needs. What helps depends on the network, the photos and how long it trains: measure it, as the course did here.",
        },
    )
    choice_dropout
    return (choice_dropout,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The CNN that won ILSVRC in 2012, with the ReLU of [A hidden layer and ReLU](/course/open/nn-hidden-layer?lang=en), the dropout of [Dropout](/course/open/nn-dropout?lang=en) and the changes of the photos of [Data augmentation for photos](/course/open/cv-augment-photos?lang=en), all three in one network. Its convolutions follow the rules of [Convolution in detail](/course/open/cnn-conv-formal?lang=en), and its parameters are counted as in [Project: your first CNN](/course/open/cnn-pytorch?lang=en). VGG and ResNet came after it, and ResNet-18, which you used in [Try an image classifier](/course/open/cv-try-classifier?lang=en), kept its ReLU, its pooling and its changes of the photos. Your project `cnn` now has `alexnet.py`. In the next lesson, *1×1 convolutions: mixing the channels*, you meet the smallest filter there is, which ResNet already uses.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_today, result_small, choice_dropout):
    L.wrap_up(result_today, result_small, choice_dropout)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
