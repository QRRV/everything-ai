# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-pretrained-features. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-pretrained-features", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en) and about [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_freeze():
    quick_freeze = L.choice(
        "quick-freeze",
        question="You freeze every parameter of the course's digit reader with `p.requires_grad = False`, and then give it a new last layer: `model.out = nn.Linear(64, 10)`. What learns when you train it?",
        options={
            "right": "Only the new last layer: 650 numbers",
            "nothing": "Nothing: you froze every parameter",
            "all": "Every layer: the new layer brings the gradients back",
        },
        correct="right",
        why={
            "right": "A new `nn.Linear` has `requires_grad` on, and the old layers stay frozen: only its 64 · 10 + 10 = 650 numbers learn. Today the frozen part is a far larger model.",
            "nothing": "You froze every parameter the model had at that moment. The new layer came after that, so it learns. See [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en).",
            "all": "The frozen parameters still get no gradient, whatever comes after them: only the new layer learns. See [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en).",
        },
    )
    quick_freeze
    return (quick_freeze,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="A model gives three photos, A, B and C, an embedding each. A and B have a cosine similarity of 0.95, A and C one of −0.1. What does that tell you?",
        options={
            "right": "To the model, A and B are alike, and A and C are not",
            "same": "A and B are the same photo",
            "length": "The embedding of A is longer than that of C",
        },
        correct="right",
        why={
            "right": "Close to 1 means that the two embeddings point the same way: the model sees the two as alike. Today the embeddings come from a model that learned from 142 million photos.",
            "same": "0.95 means alike, not the same: two images of the same digit had 0.95 on average. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
            "length": "Cosine similarity only looks at directions; the lengths drop out. See [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You want a model that tells your own kinds of things apart, ten kinds of objects say, and you have only a few photos of each. ResNet-18 from the previous lesson knows 1,000 classes, but they are ImageNet's: it has dragonfly, starfish, nautilus, rooster and ewer (under the name pitcher), and no sunflower, lotus, water lily, chandelier or Windsor chair. A network trained from scratch on a few photos per class learns them by heart. What you need is a model that already sees what is in a photo, whatever it shows, so that only a small last part has to learn your classes.

    **An everyday example.** Someone who has looked at thousands of birds sees a new kind once, in a book, and knows it again in the garden. They do not learn to see from scratch: they already know what to look at, the beak, the wings, the colors, and only learn which mix of them this bird has.
    """)
    return


@app.cell(hide_code=True)
def predict_one():
    guess_one = L.guess(
        "one",
        question="DINOv2, a large vision model, turns every photo into 384 numbers. A linear layer learns 10 classes on those numbers from only 1 photo per class: 10 photos in all. How many of 100 other photos of the same classes does it get right?",
        options={
            "chance": "About 10: as many as guessing",
            "third": "About 35",
            "two-thirds": "About 65",
            "almost": "About 95",
        },
        reveal=f"**About 95: {C.REFERENCE['features'][1]} of 100,** and between {C.REFERENCE['seeds'][0]} and {C.REFERENCE['seeds'][1]} over 5 seeds. The same linear layer on the pixels of the photos, {C.number(C.PIXELS, 'en')} numbers each, gets {C.REFERENCE['pixels'][1]}. One sunflower is not much to learn from; but on 384 numbers that already describe what is in a photo, one photo of each class is almost enough. Four of its five mistakes take a lotus for a water lily: two water flowers that look alike.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A vision foundation model** is a large network trained on a huge and varied collection of photos, not for one task, but so that what it computes inside is useful for almost any. DINOv2, which Meta published in 2023, is one. It learned from 142 million photos, which its makers chose automatically from 1.2 billion on the web, and without any labels: by self-supervised learning, as in [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en). Roughly, it learned to give two different crops of the same photo the same features. DINOv2-small, the smallest version, has 22 million parameters and turns a photo of 224 by 224 pixels into 384 numbers: its features, an embedding of the photo as in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en). It is a Vision Transformer, a kind of network that later lessons build; here it is a model you use.

        **A linear probe** is one linear layer trained on the frozen features of a model: `nn.Linear(384, k)` for k classes. It is the new last layer of [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en), with everything before it frozen. Its name says what it is for: it probes what the features hold. A linear layer can only put flat planes between classes. So when a linear probe tells the classes apart, the features hold what sets them apart, in a form that simple: in them, the classes are linearly separable, as in [What one neuron cannot learn](/course/open/nn-xor?lang=en).

        **Features once.** Nothing in DINOv2 learns, so a photo always gets the same 384 numbers. You compute them once for every photo, `dino(prepare(photo))[0]`, and put them together in a tensor of shape `(n, 384)`, a row per photo. The probe trains on that tensor: an epoch is then one small matrix multiplication instead of 22 million parameters at work on every photo, and 100 epochs take a fraction of a second. `prepare` comes from `everything_ai.dinov2`: the five steps of [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en), but resized bicubically, a smoother way than bilinear, as the settings that come with DINOv2 on Hugging Face ask.

        **Two downloads.** DINOv2-small is 88 MB, from Hugging Face, open under the Apache 2.0 license. Caltech-101 is a set of photos of objects in 101 classes, collected from the web in 2003: 137 MB, from Caltech's own data repository, under the CC BY 4.0 license, but the rights to each photo lie with its photographer. That is why the course does not ship the photos, and shares no model trained on them. It leaves out the two classes of faces, and uses 10 classes of 30 photos each that it looked at one by one, without people: chandelier, dragonfly, ewer (a tall jug), lotus, nautilus, rooster, starfish, sunflower, water lily and Windsor chair (a wooden chair with a back of thin rods). The button in the worked example downloads both, once.
        """),
        L.watch_out("DINOv2's makers chose its photos to look like those of well-known photo sets, and measured it on Caltech-101 too: it may well have seen photos much like these, without their labels. On photos unlike anything on the web, such as X-rays or photos through a microscope, a probe on its features can do much worse. Test it on photos of your own task."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **vision foundation model** | a large network trained on a huge, varied collection of photos, often without labels, whose features are useful for many tasks | |
        | **linear probe** | one linear layer trained on the frozen features of a model; how well it does shows what the features hold | `nn.Linear(384, k)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_probe():
    mo.vstack([
        mo.Html(C.probe_picture({"title": "A linear probe on DINOv2's features", "photo": "a photo, prepared: 3 × 224 × 224 numbers",
                                 "dino": "DINOv2-small", "dino_note": "22 million parameters, once per photo",
                                 "features": "384 features: the photo's embedding",
                                 "probe": "linear probe: nn.Linear(384, 10)", "probe_note": "384 · 10 + 10 = 3,850 numbers",
                                 "logits": "10 logits, one per class", "frozen": "frozen", "learns": "learns",
                                 "alt": "From top to bottom: a prepared photo goes into DINOv2-small, which is frozen and runs once per photo; it gives 384 features, the photo's embedding; the linear probe, the only part that learns, 3,850 numbers, turns them into 10 logits, one per class."})),
        mo.md(r"""
        DINOv2 runs once per photo and learns nothing; the probe learns from the features, and is all that learns: 3,850 numbers, against DINOv2's 22 million.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(f"""
    With the slider you choose how many photos per class the probe learns from, from 1 to 20. You see how many of the 100 test photos it then gets right on DINOv2's features, and how many the same probe gets right on the pixels of the prepared photos, {C.number(C.PIXELS, 'en')} numbers each.
    """)
    return


@app.cell(hide_code=True)
def explore_n_cell():
    def _say(n):
        _n = int(n)
        _f, _p, _mixed = C.REFERENCE["features"][_n], C.REFERENCE["pixels"][_n], C.REFERENCE["mixed"][_n]
        _photos = "1 photo" if _n == 1 else f"{_n} photos"
        _wrong = 100 - _f
        _of = f"all {_wrong}" if _mixed == _wrong else f"{_mixed} of its {_wrong}"
        return (f"With {_photos} per class, {10 * _n} in all: on DINOv2's features, the probe gets {_f} of the 100 test photos right, "
                f"on the pixels {_p}. On the features, {_of} mistakes mix up lotus and water lily.")

    def _frame(n):
        _html = C.curve(int(n), {"title": "Test photos right, of 100", "x": "photos per class to learn from", "y": "right",
                                 "features": "DINOv2's features", "pixels": "pixels", "guess": "guessing",
                                 "alt": "Two lines over the photos per class, 1 to 20: on DINOv2's features the probe gets 94 to 98 of 100 test photos right, on the pixels 20 to 34; guessing would give 10."}, "en")
        return _html, _say(n)

    explore_n = L.explore("n", "photos per class", _frame, start=1, values=C.PER_CLASS, role="",
                          task="Find the fewest photos per class with which the probe on the pixels gets more than 33 test photos right.")
    explore_n
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    On DINOv2's features, one photo per class already gives 95, and 20 give 98: more photos hardly help, because what is left are mostly lotuses and water lilies, which look alike. The counts go up and down a little, because each new photo can be an easy or a hard one. On the pixels, the probe needs 10 photos per class to get more than 33 right, and never gets past 34: better than guessing, 10, but far below the features. Why the pixels fall so far behind is the question of Exercise 3.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To teach a frozen vision model new classes with a linear probe:

    1. Load the model, `dino`, in eval mode, and let it compute inside `with torch.no_grad():`.
    2. Choose photos to learn from and other photos to test on, and give every photo its class as a whole number from 0 to k − 1: its label.
    3. Compute the features of every photo once: `dino(prepare(photo))[0].numpy()`, 384 numbers, and put the rows together in a tensor of shape `(n, 384)`.
    4. Train a linear probe on the features of the training photos: `nn.Linear(384, k)`, `nn.CrossEntropyLoss()` and `torch.optim.SGD`, as in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en).
    5. Test it on the features of the test photos: how many it gets right.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the first 5 photos of each of the 10 classes to learn from, and photos 20 to 29 of each class to test on: 50 and 100 photos. `C.CLASSES` holds the names of the classes, and `C.photos(name)` the 30 photos of a class; the label of a photo is the place of its class in `C.CLASSES`. `features(photos)` is step 3 for a list of photos: the features of each photo as a NumPy row, and all rows as one tensor. The cell prints the shapes of the features, and how many training and test photos the probe gets right after 100 epochs. The features of 150 photos take some seconds; the 100 epochs less than one. `dino` is DINOv2-small, which the cell below loads once both downloads are done.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download DINOv2-small and Caltech-101, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small and the photos of Caltech-101 are not on your computer yet. The button downloads them once: DINOv2-small from Hugging Face, 88 MB, and Caltech-101 from Caltech, 137 MB, whose photos it then unpacks. The worked example and the demo need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 225 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*DINOv2-small and the photos are on your computer: 22 million parameters, which learned from 142 million photos without labels.*",
    })
    return (dino,)


@app.cell
def example_probe(dino):
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.dinov2 import prepare

    def features(photos):
        rows = []
        with torch.no_grad():
            for photo in photos:
                rows.append(dino(prepare(photo))[0].numpy())
        return torch.tensor(np.array(rows))

    # 2. 5 photos per class to learn from, and 10 others to test on, with their labels
    train, y_train, test, y_test = [], [], [], []
    for label in range(10):
        photos = C.photos(C.CLASSES[label])
        for photo in photos[:5]:
            train.append(photo)
            y_train.append(label)
        for photo in photos[20:30]:
            test.append(photo)
            y_test.append(label)
    # 3. the features of every photo, once
    X_train, X_test = features(train), features(test)
    y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
    print(X_train.shape, X_test.shape)
    # 4. a linear probe from the 384 features to the 10 classes, trained on the features
    torch.manual_seed(0)
    probe = nn.Linear(384, 10)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(probe.parameters(), lr=0.1)
    for epoch in range(100):
        loss = loss_fn(probe(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 5. test it
    with torch.no_grad():
        print((probe(X_train).argmax(dim=1) == y_train).sum().item(), "of 50 training photos right")
        print((probe(X_test).argmax(dim=1) == y_test).sum().item(), "of 100 test photos right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The features have shape `(50, 384)` and `(100, 384)`: a row of 384 numbers per photo. After 100 epochs the probe gets all 50 training photos right, and its loss is close to 0. That alone says little: a layer of 3,850 numbers can learn 50 photos by heart. What counts are the 100 test photos it never saw, and it gets 97 of them right. The three it gets wrong are water lilies that it takes for lotuses.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_params():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. What learns
    """)
    return


@app.cell(hide_code=True)
def choice_params_cell():
    choice_params = L.choice(
        "params",
        question="A linear probe on DINOv2-small's 384 features, for all 99 classes the course keeps of Caltech-101. How many numbers learn?",
        options={
            "right": "38,115",
            "no-bias": "38,016",
            "all": "22 million and 38,115",
            "sum": "483",
        },
        correct="right",
        why={
            "right": "384 · 99 weights, one per feature and class, and 99 biases, one per class: 38,016 + 99 = 38,115. DINOv2's 22 million parameters stay frozen.",
            "no-bias": "Those are the weights, 384 · 99. Every class also has a bias: 38,016 + 99 = 38,115.",
            "all": "DINOv2 is frozen: its 22 million parameters get no gradient. Only the probe learns: 384 · 99 + 99 = 38,115.",
            "sum": "A linear layer has a weight for every pair of a feature and a class, 384 · 99, and a bias per class: 38,115.",
        },
    )
    choice_params
    return (choice_params,)


@app.cell(hide_code=True)
def text_exercise_probe():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Train a linear probe

        Complete `train_probe(X, y, k)`: step 4 of the recipe, for any features and any number of classes. `X` has a row of features per photo, `y` the label of each photo, a whole number from 0 to `k` − 1, and `k` is the number of classes. Replace the two `...`; the function gives back the trained probe. The check makes up features, 4 per photo for 3 classes, calls `train_probe(X, y, 3)`, and looks at the shape of the probe and at the classes it gives 8 test points. It also tries other numbers of features and classes.
        """),
        L.task("Make `train_probe(X, y, k)` train a linear probe from the features in `X` to `k` classes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_probe():
    probe_code = L.code_box("probe", lines=14)
    probe_code
    return (probe_code,)


@app.cell(hide_code=True)
def check_probe(probe_code):
    result_probe = L.check("probe", probe_code.value)
    result_probe
    return (result_probe,)


@app.cell(hide_code=True)
def hints_probe():
    L.hints(
        "probe",
        [
            "Step 4 of the worked example, with the number of features and the number of classes taken from `X` and `k`.",
            "The number of features is the length of a row of `X`: `X.shape[1]`. The loss gets the probe's logits for all photos, and their labels.",
            "`probe = nn.Linear(X.shape[___], k)` and `loss = loss_fn(probe(___), y)`.",
        ],
        why="`nn.Linear(X.shape[1], k)` takes the features of one photo and gives `k` logits. `loss_fn(probe(X), y)` is the cross-entropy over all photos at once, and every step of SGD moves the probe's weights and biases a little, while the features stay as they are.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your probe on the photos

    Once DINOv2 and the photos are downloaded, in the worked example, the cell below gives DINOv2's features to your `train_probe` once it is right, else to the course's. For every number of photos per class of the slider, it trains a probe on the first photos of each class and tests it on the 100 test photos. The features of all 300 photos take some seconds, once.
    """)
    return


@app.cell(hide_code=True)
def demo_probe(dino, probe_code, result_probe):
    _d = C.demo(dino, probe_code.value if result_probe.state == "correct" else None)
    _who = "your `train_probe`" if _d["yours"] else "the course's `train_probe`"
    _rows = "\n".join(f"| {_n} | {_d['right'][_n]} | {C.REFERENCE['features'][_n]} |" for _n in C.PER_CLASS)
    mo.md(f"""With {_who}, on DINOv2's features:

| photos per class | test photos right, of 100 | the course measured |
|---|---|---|
{_rows}""")
    return


@app.cell(hide_code=True)
def text_exercise_pixels():
    mo.md(r"""
    ### Exercise 3. A probe on the pixels
    """)
    return


@app.cell(hide_code=True)
def choice_pixels_cell():
    choice_pixels = L.choice(
        "pixels",
        question=f"Give the same probe the pixels of each prepared photo instead of its features: 3 · 224 · 224 = {C.number(C.PIXELS, 'en')} numbers, so `nn.Linear({C.PIXELS}, 10)`. With 5 photos per class, its training loss goes to 0 too: it gets all 50 training photos right. Yet it gets only {C.REFERENCE['pixels'][5]} of the 100 test photos right, against {C.REFERENCE['features'][5]} on the features. Why?",
        options={
            "short": "It trained too briefly: with more epochs it would get 97 too",
            "right": "No single pixel says what is in a photo: the probe learns its 50 photos by heart",
            "less": f"The pixels hold less than the features: {C.number(C.PIXELS, 'en')} numbers say less than 384",
            "small": "The layer is too small for so many numbers",
        },
        correct="right",
        why={
            "short": "Its training loss is already 0: it gets every training photo right. More epochs only make it surer of those 50, not better on new photos: what it learned holds for those 50 alone, as in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
            "right": f"A pixel only says how bright one spot is. Whether there is a sunflower in the photo, no single pixel says, and the sunflower can be anywhere in the frame, in any light. With {C.number(C.PIXELS * 10 + 10, 'en')} numbers for 50 photos, the probe finds something in each training photo that sets it apart, and learns the 50 by heart, as in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en). DINOv2's features already describe what is in a photo, so the flat planes the probe puts between the classes hold for new photos too.",
            "less": "DINOv2 computes its 384 features from these very pixels, so everything in the features is in the pixels too. But not in a form a linear layer can use: no single pixel says what is in the photo, and the object can be anywhere. So the probe learns its 50 photos by heart.",
            "small": f"It has {C.number(C.PIXELS * 10 + 10, 'en')} numbers, against 3,850 on the features, and it gets all 50 training photos right. It has too much room, not too little: it learns the 50 by heart, as in [Overfitting and underfitting](/course/open/ml-overfitting?lang=en).",
        },
    )
    choice_pixels
    return (choice_pixels,)


@app.cell(hide_code=True)
def text_exercise_no_labels():
    mo.md(r"""
    ### Exercise 4. Classes without labels
    """)
    return


@app.cell(hide_code=True)
def explain_no_labels_cell():
    explain_no_labels = L.explain(
        "no-labels",
        question="DINOv2 never saw a label, not even the word sunflower. Yet on its features, a linear probe tells sunflowers, starfish and chandeliers apart after one photo of each. How can features learned without labels hold classes? Think of what DINOv2 learned to do, and of what two photos of the same kind of thing have in common.",
        model_answer="**DINOv2 learned to give two different crops of the same photo the same features,** so its features cannot depend on where something is in the frame, how large it is or how it is lit: they have to describe what is in the photo, its shapes, textures and parts. **Two photos of the same kind of thing share just that,** such as the petals around the dark middle of a sunflower or the five arms of a starfish, so their features lie close together, and away from those of other kinds. **The classes are in the features as groups before anyone names them.** A label only says which group gets which name, and that is why the probe needs so few photos to put its flat planes between them.",
        points=[
            "you say that DINOv2 learned to give different crops of the same photo the same features, so that its features describe what is in a photo, not where or in what light",
            "you say that photos of the same kind of thing share what is in them, so that their features lie close together",
            "you say that the groups are there before any label, so that a few labeled photos are enough to tell the probe which group is which",
        ],
    )
    explain_no_labels
    return (explain_no_labels,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The frozen feature extractor of [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en) is now a vision foundation model that learned from 142 million photos without labels, and its features are an embedding of a photo, as in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en). A linear probe, one layer trained with cross-entropy, shows what those features hold: 95 of 100 test photos right after one photo per class, against 20 on the pixels. In the next lesson, *Find photos that look alike*, you leave out the probe: the cosine similarity of DINOv2's features finds the photos most like a given one.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_params, choice_pixels, explain_no_labels, result_probe):
    L.wrap_up(choice_params, result_probe, choice_pixels, explain_no_labels)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
