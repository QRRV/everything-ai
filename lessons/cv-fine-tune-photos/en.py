# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-fine-tune-photos. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-fine-tune-photos", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en) and about [Data augmentation](/course/open/nn-data-augmentation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_phases():
    quick_phases = L.choice(
        "quick-phases",
        question="In fine-tuning, phase 2 lets every layer learn from the errors of the last layer. Why does phase 1, which trains only a new last layer, come first?",
        options={
            "faster": "A frozen network trains faster, so phase 1 saves time",
            "right": "A new last layer starts random: its first errors say nothing about the task, and would pull the borrowed layers along",
            "small": "So that phase 2 can use a larger learning rate",
        },
        correct="right",
        why={
            "faster": "Phase 1 is fast, but that is not why it comes first: without it, the digit reader read 747 to 844 of 1,000 digits right, and with it 879 to 893. See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
            "right": "After phase 1, the errors of the last layer are about the new task, and phase 2 learns from those. Today you see what phase 2 has left to learn when phase 1 already gets every training photo right.",
            "small": "The other way round: phase 2 needs a learning rate about ten times smaller, so that the borrowed features change only a little. See [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
        },
    )
    quick_phases
    return (quick_phases,)


@app.cell(hide_code=True)
def quick_check_mirror():
    quick_mirror = L.choice(
        "quick-mirror",
        question="Which change of a training image is a good data augmentation for a model that reads handwritten digits?",
        options={
            "mirror": "Mirror it left to right",
            "right": "Shift it a pixel or two",
            "upside": "Turn it upside down",
        },
        correct="right",
        why={
            "mirror": "A mirrored digit can become no digit at all, or another one. See [Data augmentation](/course/open/nn-data-augmentation?lang=en).",
            "right": "A digit shifted by a pixel is still the same digit, so the label stays right. A sunflower photo mirrored left to right still shows a sunflower; today you leave augmentation out, to see what phase 2 alone does.",
            "upside": "Upside down, a 6 becomes a 9. See [Data augmentation](/course/open/nn-data-augmentation?lang=en).",
        },
    )
    quick_mirror
    return (quick_mirror,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You have a few hundred photos of your own kinds of things, and a classifier trained on ImageNet. Phase 1, a new last layer on its frozen features, is quick, and it already gets most of your photos right. The usual next step is to let every layer learn: phase 2 of fine-tuning, on a network with 11 million parameters instead of the small digit reader. That costs real computing: every training photo goes through every layer, forward and back, in every epoch. Whether it pays, only a measurement on photos it never trained on can say.

    **An everyday example.** Say you are a good photographer of flowers, and start at a nature magazine. In a week you learn which shots the editors want. Then you could also change the way you shoot. For flowers in a garden, your old way already works, and practice buys little. For birds in flight, where your eye for light still helps but the timing is new, practice pays.
    """)
    return


@app.cell(hide_code=True)
def predict_gain():
    _lo, _hi = C.low_high(C.REFERENCE["phase1"])
    _alo, _ahi = C.low_high(C.after(C.TUNE_LR))
    _glo, _ghi = C.low_high(C.gains())
    guess_gain = L.guess(
        "gain",
        question=f"A new last layer on ResNet-18's frozen features learns 10 kinds of things from 20 photos of each, and gets {_lo} to {_hi} of 100 test photos right, over 5 seeds. Then phase 2 lets all 11 million parameters of ResNet-18 learn, for 2 epochs with a small learning rate. How many test photos does the model get right after that?",
        options={
            "fewer": "Fewer: 11 million numbers learn 200 photos by heart",
            "same": "About as many: at most 2 more",
            "more": "Clearly more: about 5 more",
            "all": "Almost all 100",
        },
        reveal=f"**About as many: {_alo} to {_ahi},** {C.gain_words(_glo, _ghi, 'en')} than after phase 1 on each seed. Phase 1 already gets all 200 training photos right, so the loss phase 2 learns from is already close to 0, and so are its gradients and its steps. Nearly all mistakes that are left are a lotus taken for a water lily or the other way around: two flowers that look alike.",
    )
    guess_gain
    return (guess_gain,)


@app.cell(hide_code=True)
def text_idea():
    _tlo, _thi = C.low_high(C.REFERENCE["train-mode"])
    _alo, _ahi = C.low_high(C.after(C.TUNE_LR))
    mo.vstack([
        mo.md(rf"""
        ## The idea

        **Full fine-tuning** is fine-tuning in which every parameter of the model learns: phase 2 of [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en), now on ResNet-18. With a new last layer for 10 classes, that is {C.number(C.PARAMS, 'en')} numbers, against {C.number(C.HEAD, 'en')} in phase 1. The name sets it apart from ways of fine-tuning only part of a model, which come in later lessons. It is costly in two ways. Every step, each of the 11 million parameters gets a gradient, which the computer keeps next to the parameter itself: twice the memory of the parameters alone, and more with an optimizer such as Adam, which keeps numbers of its own for each parameter. And every training photo goes through every layer, forward and back, in every epoch.

        **Phase 1 on features, once.** In phase 1, nothing before the new last layer learns, as in [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en). So a photo always gets the same 512 features, the numbers ResNet-18's last layer gets. With `model.fc = nn.Identity()`, the model gives those features instead of logits; you compute them once for every training photo, in batches of 50 inside `with torch.no_grad():`, and train the new last layer on them. Each of its 100 epochs is then a few small matrix multiplications, done in a moment.

        **Phase 2 in mini-batches.** Now every photo has to go through the whole network again, in every epoch, because every layer changes. A `DataLoader` gives the 200 training photos in mini-batches of 20, in a new order every epoch, as in [Datasets and DataLoaders](/course/open/torch-dataloader?lang=en): the computer keeps what each layer computed for 20 photos at a time, for the way back, instead of for all 200. The learning rate is 0.003. Ten times smaller than phase 1's 0.1 would be 0.01, but that is close to where phase 2 runs away, as the slider below shows, so the lesson stays well below it.

        **Eval mode stays on.** ResNet-18 has layers that compute differently in train mode, as dropout does in [Training mode and evaluation mode](/course/open/torch-eval-mode?lang=en): batch normalization, which a later lesson explains. In eval mode they keep the averages they measured on ImageNet's photos, instead of computing new ones from each batch of 20. The course keeps the model in eval mode during phase 2 too: in train mode it got {_tlo} to {_thi} test photos right, over 5 seeds, against {_alo} to {_ahi}.
        """),
        L.watch_out("Judge phase 2 on photos it did not train on: after phase 1, ResNet-18 already gets all 200 training photos right, so they cannot show whether phase 2 helps. Make choices such as the learning rate on a validation set, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en); this lesson chooses on its 100 test photos only because 30 photos per class leave little to split off."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **full fine-tuning** | fine-tuning in which every parameter of a model learns; it needs a gradient for each of them, and a pass forward and back through every layer for every photo | `for p in model.parameters(): p.requires_grad = True` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_phases():
    mo.vstack([
        mo.Html(C.phases_picture({"title": "Full fine-tuning of ResNet-18", "phase1": "phase 1", "rate1": "learning rate 0.1, 100 epochs",
                                  "count1": f"{C.number(C.HEAD, 'en')} numbers learn", "phase2": "phase 2",
                                  "rate2": "learning rate 0.003, 2 epochs", "count2": f"{C.number(C.PARAMS, 'en')} numbers learn",
                                  "fc": "fc (new)", "learns": "learns", "frozen": "frozen",
                                  "alt": f"Two columns with the parts of ResNet-18 from top to bottom: conv1, layer1, layer2, layer3, layer4 and fc, the new last layer for 10 classes. In phase 1, with a learning rate of 0.1 for 100 epochs, everything is frozen but fc: {C.number(C.HEAD, 'en')} numbers learn. In phase 2, with a learning rate of 0.003 for 2 epochs, every part learns: {C.number(C.PARAMS, 'en')} numbers."})),
        mo.md(r"""
        conv1 and layer1 to layer4, groups of layers as ResNet-18 names them, are the part it learned on ImageNet; how they compute, later lessons show. fc is new. In phase 1, only fc learns, on features computed once; in phase 2, everything learns, and every photo goes through all of it again.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    With the slider you choose the learning rate of phase 2, from 0.0003 to 0.1. For each seed from 0 to 4, the course trained phase 1, then phase 2 for 2 epochs with that learning rate. The bar runs from the fewest to the most test photos right over the 5 seeds; the grey band is phase 1 alone.
    """)
    return


@app.cell(hide_code=True)
def explore_lr_cell():
    def _say(lr):
        _lo, _hi = C.low_high(C.after(lr))
        _p1, _p2 = C.low_high(C.REFERENCE["phase1"])
        _text = (f"With a learning rate of {C.rate(lr, 'en')}, phase 2 gets {_lo} to {_hi} of the 100 test photos right after 2 epochs, "
                 f"over 5 seeds; phase 1 alone got {_p1} to {_p2}.") if _lo < _hi else (
                 f"With a learning rate of {C.rate(lr, 'en')}, every seed gets {_lo} of the 100 test photos right after 2 epochs of phase 2; "
                 f"phase 1 alone got {_p1} to {_p2}.")
        if C.REFERENCE["classes"][lr] == 1:
            _text += " The numbers in the network have become `nan`, and every test photo gets the same class: as many right as there are test photos of that class."
        return _text

    def _frame(lr):
        _p1, _p2 = C.low_high(C.REFERENCE["phase1"])
        _a, _b = C.low_high(C.after(0.01))
        _g1, _g2 = C.low_high(C.gains(0.01))
        _html = C.curve(lr, {"title": "Test photos right after phase 2, of 100", "x": "learning rate of phase 2", "y": "right",
                             "phase1": "phase 1 alone", "phase2": "after phase 2", "guess": "guessing",
                             "alt": f"Over the learning rate of phase 2, from {C.rate(C.RATES[0], 'en')} to {C.rate(C.RATES[-1], 'en')}: up to 0.003, phase 2 stays in or just above the band of phase 1 alone, {_p1} to {_p2} of 100 test photos right; at 0.01 it gets {_a} to {_b}, but one seed {-_g1} fewer than phase 1 and another {_g2} more; from 0.03, every seed gets 10, as many as guessing."}, "en")
        return _html, _say(lr)

    explore_lr = L.explore("lr", "learning rate", _frame, start=C.TUNE_LR, values=C.RATES, role="hyper",
                           task="Find the smallest learning rate at which phase 2 gives every test photo the same class.")
    explore_lr
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    From 0.03 on, every seed gets 10 of 100: phase 2 gives every test photo the same class. The numbers in the network have become `nan`, as `inf / inf` does in [Many classes: softmax](/course/open/ml-softmax?lang=en), and a model full of `nan` cannot tell photos apart; Exercise 3 asks why phase 2 runs away. Up to 0.003, phase 2 stays close to phase 1: at 0.003, 0 to 2 more on every seed. At 0.01, phase 2 moves further from phase 1: from 1 fewer to 3 more, depending on the seed. So the edge lies somewhere between 0.01 and 0.03, and the lesson takes 0.003, well below it.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fine-tune every layer of a pretrained classifier on photos of k kinds of things of your own:

    1. Prepare every photo once, as in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en), and split them three ways: photos to learn from, a validation set to choose with, and photos to test on, each with its label, a whole number from 0 to k − 1.
    2. Phase 1: with the model frozen and `model.fc = nn.Identity()`, compute the features of every training photo once; then give the model a new last layer, `model.fc = nn.Linear(512, k)`, 512 being ResNet-18's number of features, and train it on those features.
    3. Phase 2: unfreeze every parameter, and train them all further in mini-batches with a small learning rate, the model in eval mode.
    4. Compare the two phases on the validation set, and keep phase 2 only if it does better there; then test the model you keep on the test photos, once.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    **Two downloads.** ResNet-18 is the one of [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en): 47 MB from Hugging Face, with torchvision's weights, trained on ImageNet. Caltech-101 consists of photos of objects in 101 classes, collected from the web in 2003: 137 MB, from Caltech's own data repository, under the CC BY 4.0 license, but the rights to each photo lie with its photographer. That is why the course does not ship the photos, and shares no model trained on them. It uses 10 classes of 30 photos each, which it checked one by one so that none shows a person: chandelier, dragonfly, ewer (a tall jug), lotus, nautilus, rooster, starfish, sunflower, water lily and Windsor chair (a wooden chair with a back of thin rods). Of these, ImageNet has dragonfly and starfish, and three more under another name: chambered nautilus, cock and pitcher. It has no lotus, water lily, sunflower, chandelier or Windsor chair. The button below downloads both, once.

    The cell takes the first 20 photos of each of the 10 classes to learn from, and photos 20 to 29 of each class to test on: 200 and 100 photos. `C.CLASSES` holds the names of the classes, and `C.photos(name)` the 30 photos of a class; the label of a photo is the place of its class in `C.CLASSES`. `C.prepare(photo)` does the five steps of [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en) for a photo of any size, with ImageNet's mean and standard deviation per channel, and gives an array of shape `(3, 224, 224)`. `right(model, X, y)` counts how many photos the model gets right, 50 at a time. `pretrained()` gives a new ResNet-18 with ImageNet's weights, every parameter frozen, in eval mode: the cell below hands it over once both downloads are done. The cell prints the shapes of the photos, and how many test photos are right after phase 1 and after phase 2. On the course's computer, with 4 cores, phase 1 took about 3 seconds, the features of the 200 photos included, and each epoch of phase 2 about 7: two epochs of phase 2 cost about five times as much as all of phase 1. On a slower computer, all of it takes longer.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download ResNet-18 and Caltech-101, 184 MB")
    return (get_button,)


@app.cell(hide_code=True)
def downloads_cell(get_button):
    pretrained = C.downloads_or_stop(get_button, {
        "ask": "*ResNet-18 and the photos of Caltech-101 are not on your computer yet. The button downloads them once: ResNet-18 from Hugging Face, 47 MB, and Caltech-101 from Caltech, 137 MB, whose photos it then unpacks. The worked example and the demo need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 184 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*ResNet-18 and the photos are on your computer: 11.7 million parameters, trained on ImageNet, and the photos of Caltech-101, of which the lesson uses 10 classes.*",
    })
    return (pretrained,)


@app.cell
def example_fine_tune(pretrained):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    def right(model, X, y):
        count = 0
        with torch.no_grad():
            for xb, yb in DataLoader(TensorDataset(X, y), batch_size=50):
                count += (model(xb).argmax(dim=1) == yb).sum().item()
        return count

    # 1. every photo prepared once: 20 per class to learn from, 10 to test on
    train, y_train, test, y_test = [], [], [], []
    for label in range(10):
        photos = C.photos(C.CLASSES[label])
        for photo in photos[:20]:
            train.append(C.prepare(photo))
            y_train.append(label)
        for photo in photos[20:30]:
            test.append(C.prepare(photo))
            y_test.append(label)
    X_train, X_test = torch.tensor(np.array(train)), torch.tensor(np.array(test))
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
    print(X_train.shape, X_test.shape)
    # 2. phase 1: the 512 features of every training photo, once; a new last layer learns on them
    model = pretrained()
    model.fc = nn.Identity()
    rows = []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(X_train, y_train), batch_size=50):
            for row in model(xb):
                rows.append(row.numpy())
    features = torch.tensor(np.array(rows))
    torch.manual_seed(0)
    model.fc = nn.Linear(512, 10)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.fc.parameters(), lr=0.1)
    for epoch in range(100):
        loss = loss_fn(model.fc(features), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("after phase 1:", right(model, X_test, y_test), "of 100 test photos right")
    # 3. phase 2: every parameter learns, with a small learning rate, in mini-batches of 20
    torch.manual_seed(0)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.SGD(model.parameters(), lr=0.003)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=20, shuffle=True)
    for epoch in range(2):
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # 4. test again
    print("after phase 2:", right(model, X_test, y_test), "of 100 test photos right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The photos have shape `(200, 3, 224, 224)` and `(100, 3, 224, 224)`. After phase 1, 90 of the 100 test photos are right, and all 10 mistakes mix up lotus and water lily. Phase 2 brings it to 92: 2 more than phase 1, and the 8 mistakes left are lotuses and water lilies again. Over 5 seeds, 2 epochs of phase 2 get 0 to 2 more on each seed, and 10 epochs between 1 fewer and 3 more. Phase 1 already gets all 200 training photos right, so the loss phase 2 learns from is small, and so are its steps. For comparison: a ResNet-18 with random weights, trained with the loop of phase 2 on the same 200 photos, but in train mode, for 10 epochs and with a learning rate of 0.01, gets 29 to 38 of 100 right, over 3 seeds. Without what it learned on ImageNet, 200 photos are far too few for 11 million numbers.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What learns in phase 2
    """)
    return


@app.cell(hide_code=True)
def choice_params_cell():
    choice_params = L.choice(
        "params",
        question="You fine-tune ResNet-18 for all 99 classes the course keeps of Caltech-101 (it leaves out the two classes of faces): a new last layer from 512 features to 99 classes, then full fine-tuning. ResNet-18 as it comes, with its last layer for ImageNet's 1,000 classes, has 11,689,512 parameters. How many numbers learn in phase 2?",
        options={
            "right": "11,227,299",
            "head": "50,787",
            "imagenet": "11,689,512",
            "no-head": "11,176,512",
            "both": "11,740,299",
        },
        correct="right",
        why={
            "right": "Without its old last layer, 512 · 1,000 + 1,000 = 513,000 numbers, ResNet-18 has 11,176,512; the new one adds 512 · 99 + 99 = 50,787. In full fine-tuning they all learn: 11,227,299.",
            "head": "Those are the numbers of the new last layer, 512 · 99 + 99: what learns in phase 1. In phase 2 every parameter learns: the 11,176,512 of the rest of ResNet-18 too, 11,227,299 in all.",
            "imagenet": "The old last layer, for 1,000 classes, is gone: `model.fc` was replaced. Take its 512 · 1,000 + 1,000 = 513,000 off, and add the new one, 512 · 99 + 99 = 50,787: 11,227,299.",
            "no-head": "That is ResNet-18 without any last layer. The new one learns too, in both phases: 512 · 99 + 99 = 50,787 more, 11,227,299 in all.",
            "both": "That counts the old last layer, for 1,000 classes, next to the new one. But `model.fc = nn.Linear(512, 99)` replaces it: take its 512 · 1,000 + 1,000 = 513,000 off, and 11,227,299 are left.",
        },
    )
    choice_params
    return (choice_params,)


@app.cell(hide_code=True)
def text_exercise_phase_two():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Phase 2 in a training loop

        Complete `phase_two(model, X, y, lr, epochs)`: step 3 of the recipe, for any model after phase 1. `X` holds the prepared training photos, `y` their labels, `lr` is the learning rate and `epochs` the number of epochs. Replace the three `...`; the function gives back the model, trained further. The check makes up a tiny model as phase 1 leaves it, a frozen first layer and a last layer that learns, and 50 made-up examples of 4 numbers each, and calls `phase_two(model, X, y, 0.1, 2)`. It looks at which layers changed, and at the loss of the model on its examples afterwards. It also tries other learning rates and numbers of epochs.
        """),
        L.task("Make `phase_two` unfreeze every parameter of the model and train them all in mini-batches, with learning rate `lr`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_phase_two():
    phase_two_code = L.code_box("phase-two", lines=20)
    phase_two_code
    return (phase_two_code,)


@app.cell(hide_code=True)
def check_phase_two(phase_two_code):
    result_phase_two = L.check("phase-two", phase_two_code.value)
    result_phase_two
    return (result_phase_two,)


@app.cell(hide_code=True)
def hints_phase_two():
    L.hints(
        "phase-two",
        [
            "Step 3 of the worked example, with the learning rate taken from `lr`.",
            "Unfreezing is freezing the other way round: `requires_grad` back on. The optimizer gets every parameter of the model, and the loss gets the logits of the batch, `model(xb)`, and the labels of the batch.",
            "`p.requires_grad = ___`, `opt = torch.optim.SGD(model.___(), lr=lr)` and `loss = loss_fn(model(xb), ___)`.",
        ],
        why="`p.requires_grad = True` lets every parameter get a gradient again, and `torch.optim.SGD(model.parameters(), lr=lr)` moves them all, a small step per batch. `loss_fn(model(xb), yb)` is the loss of one batch of 20 examples, so the model takes 10 steps per epoch on 200 photos instead of one.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your phase 2 on the photos

    Once you have downloaded ResNet-18 and the photos with the button in the worked example, the button below trains phase 1 again and hands the model to your `phase_two` if it is right, or else to the course's: 2 epochs with a learning rate of 0.003. While phase 2 trains, a chart shows the loss of each batch of 20. Beside or under it are ResNet-18's guesses at that moment for the 20 test photos of lotus and water lily, the two flowers that all mistakes after phase 1 mix up, each with how sure it is: the probability softmax gives the class it guesses, as in [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en). Under them is what `layer4.1`, the last part of `layer4`, makes of the first lotus: its **feature maps**, a grid of 7 by 7 numbers for each of its channels, one number for each part of the photo. Both are drawn again as phase 2 trains. Then you can rewind to a few steps of phase 2, with both as they were then, and a table shows, per class, how many of the 10 test photos are right after phase 1 and after phase 2. It takes about as long as the worked example.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Train phase 1 and phase 2")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_phase_two(pretrained, demo_button, phase_two_code, result_phase_two):
    if not demo_button.value:
        _out = mo.md("*Press the button to train phase 1 and phase 2.*")
    else:
        _now = {}
        _layer = "The last part of layer4 on the first test photo of a lotus: {name}"

        def _progress(_p):
            if "step" not in _p:   # phase 1 starts: a new chart for phase 2
                _now["live"] = L.live({"title": "Phase 2: the loss of each batch", "x": "step", "y": "loss",
                                       "alt": "The loss of each batch of 20 training photos in phase 2 so far, step by step."},
                                      C.TUNE_EPOCHS * C.STEPS, low=0.0, every=2.0)
                _now.pop("look", None)
                _now["live"].line("**Phase 1** · the features of the 200 training photos, and a new last layer")
            else:
                if "look" not in _now:   # phase 2 starts: what ResNet-18 makes at each step, beside the chart
                    _now["look"] = _p["look"]
                    _now["live"].watch(lambda: C.now_pictures(_now["look"](), _layer, L.lang, "nv-tune-now"))
                _now["live"].add(_p["step"], _p["loss"], f"**Phase 2** · step {_p['step']} of {C.TUNE_EPOCHS * C.STEPS} · loss {C.loss_text(_p['loss'], 'en')}")

        _d = C.demo(pretrained, phase_two_code.value if result_phase_two.state == "correct" else None, _progress)
        _who = "your `phase_two`" if _d["yours"] else "the course's `phase_two`"
        _rows = "\n".join(f"| {C.NAMES['en'][_c]} | {_b} | {_a} |" for _c, _b, _a in zip(C.CLASSES, _d["before"], _d["after"]))
        _labels = {"title": "Phase 2: the loss of each batch", "x": "step", "y": "loss",
                   "alt": "The loss of each batch of 20 training photos in phase 2, step by step."}
        _when = sorted(_d["looked"])

        def _frame(_k):
            _k, _seen = int(_k), _d["looked"][int(_k)]
            _right = sum(_g == _t for _g, _t in zip(_seen["guesses"], _seen["truth"]))
            _at = "After phase 1, before phase 2 takes a step" if _k == 0 else f"After step {_k} of phase 2"
            return C.step_frame(_d, _k, _labels, _layer, L.lang), f"{_at}: {_right} of these 20 photos are right."

        if len(_when) > 1:   # rewinding through phase 2, with all of it in the picture
            _pic = L.explore("rewind", "step", _frame, start=_when[-1], values=_when, role="",
                             task="Go back to step 0, then forward again: which photos change their guess, and how sure is ResNet-18 of those?")
        else:
            _pic = mo.Html(C.losses_picture(_d["losses"], _labels, L.lang))
        _out = mo.vstack([_pic, mo.md(f"""With {_who}, of the 10 test photos of each class:

| class | right after phase 1 | right after phase 2 |
|---|---|---|
{_rows}
| **all 100** | **{sum(_d['before'])}** | **{sum(_d['after'])}** |

In {_d['mixed']} of the {100 - sum(_d['after'])} mistakes after phase 2, a lotus is taken for a water lily or the other way around.""")])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_too_big():
    mo.md(r"""
    ### Exercise 3. A learning rate that is too big
    """)
    return


@app.cell(hide_code=True)
def choice_too_big_cell():
    choice_too_big = L.choice(
        "too-big",
        question="With a learning rate of 0.03, the loss of phase 2 grows from step to step, from 0.01 to 49 in its first 7 steps, and two steps later it is `nan`. From then on, ResNet-18 gives every photo the same class. Phase 1 trained its new last layer with 0.1, more than three times as large, without trouble. Why does 0.03 ruin phase 2?",
        options={
            "right": "In phase 2 every layer moves at once, and each change also changes what all later layers get: one step changes the logits far more than in phase 1",
            "by-heart": "11 million numbers learn the 200 training photos by heart",
            "short": "2 epochs are too few: with a few more, phase 2 would recover",
        },
        correct="right",
        why={
            "right": "In phase 1, a step moves only the last layer, on features that stay the same. In phase 2, a step also moves conv1 and every layer after it, and each change passes on to all the layers that come later, so together they change the logits far more than the same step on the last layer alone. Steps that are too big jump over the bottom, the loss grows, and the next gradients grow with it: training diverges, as in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en), until the numbers grow too large for the computer to hold and become `nan`, as in [Many classes: softmax](/course/open/ml-softmax?lang=en). That is why phase 2 needs a much smaller learning rate than phase 1.",
            "by-heart": "Learning by heart gets the training photos right and new photos wrong. Here the training photos fare no better: of the 200, only 20 are right, all of one class. Training ran away, and the numbers in the network became `nan`. See [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "short": "The course tried 6 epochs with 0.03: still 10 of 100, every photo the same class. Once a number in the network is `nan`, everything computed from it is `nan` too, every gradient included: there is no way back. See [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
        },
    )
    choice_too_big
    return (choice_too_big,)


@app.cell(hide_code=True)
def text_exercise_worth_it():
    mo.md(r"""
    ### Exercise 4. When does it pay?
    """)
    return


@app.cell(hide_code=True)
def explain_worth_it_cell():
    explain_worth_it = L.explain(
        "worth-it",
        question="Over 5 seeds, 2 epochs of full fine-tuning get 0 to 2 more of the 100 test photos right than phase 1 alone, for about five times the computing. Why does it add so little here, and when would it pay off? Think of what phase 1 already gets right, of the mistakes that are left, and of the clothes in [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en).",
        model_answer="**Here, phase 1 leaves little to learn.** It already gets all 200 training photos right, so the loss phase 2 learns from is close to 0, and so are its steps. ResNet-18's features already tell these kinds apart: over 5 seeds, 30 of the 32 mistakes after phase 1 mix up lotus and water lily, two flowers that look alike, and some photos labeled lotus seem to show water lilies. Phase 2 gets a few of them right, and 26 of the 27 mistakes after it are still lotus and water lily. **Full fine-tuning pays when the features fit the photos only partly:** close enough to start from, not so close that phase 1 already gets it all. On the clothes, the digit reader went from 506 to 539 of 1,000 test images right with phase 1 alone to 566 to 593 with fine-tuning, though a new network from scratch still did better there. More photos per class help too, or more versions of the same photos through data augmentation, so that 11 million numbers can move without learning their photos by heart. **Whether it pays, only photos it did not train on can say:** compare phase 2 with phase 1 alone on a validation set, and weigh what it adds against what it costs.",
        points=[
            "you say that phase 1 already gets all training photos right, so that phase 2 has little left to learn from",
            "you say that the mistakes that are left mostly mix up lotus and water lily, which look alike, so that the features already fit these photos",
            "you say that full fine-tuning pays when the features fit the photos only partly, as the digit reader's features fit the clothes, and that only photos it did not train on can show whether it does",
        ],
    )
    explain_worth_it
    return (explain_worth_it,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Phase 1 is the feature extractor of [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en), phase 2 the fine-tuning of [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en), now on ResNet-18 from [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en), in mini-batches from a `DataLoader`. Full fine-tuning lets all 11 million parameters learn. On photos that ResNet-18's features already describe well, 2 epochs of it add 0 to 2 of 100 to phase 1, and a learning rate ten times too big makes it run away. Whether it pays, only photos it did not train on can say. Batch normalization, the reason the model stays in eval mode, comes in [Batch normalization](/course/open/nn-batchnorm?lang=en); fine-tuning only part of a model, in [Project: LoRA, fine-tuning with a low-rank update](/course/open/llm-lora?lang=en). In the next lesson, *Labels, boxes, masks and keypoints*, you leave whole photos with one label behind: what a vision model has to find decides how its photos are labeled, with a box per object, a mask per pixel or a point per joint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_params, result_phase_two, choice_too_big, explain_worth_it):
    L.wrap_up(choice_params, result_phase_two, choice_too_big, explain_worth_it)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
