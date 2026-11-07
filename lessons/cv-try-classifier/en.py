# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-try-classifier. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-try-classifier", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en) and about [Many classes: softmax](/course/open/ml-softmax?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="The cat photo, 451 by 300 pixels, goes through the five steps of the previous lesson. Which shape does the tensor that goes into the model have?",
        options={
            "right": "`(1, 3, 224, 224)`",
            "last": "`(1, 224, 224, 3)`",
            "nobatch": "`(3, 224, 224)`",
            "resized": "`(1, 3, 256, 384)`",
        },
        correct="right",
        why={
            "right": "Whatever the size of the photo, the center crop gives 224 by 224; `.permute(2, 0, 1)` puts the 3 colors first, and `.unsqueeze(0)` makes a batch of one.",
            "last": "That is the order NumPy keeps an image in. The model takes the colors first: `.permute(2, 0, 1)` gives `(3, 224, 224)`, and `.unsqueeze(0)` puts a batch of one in front.",
            "nobatch": "That is one photo, but the model takes a batch: `.unsqueeze(0)` puts an axis of length 1 in front, `(1, 3, 224, 224)`.",
            "resized": "That is the photo after step 1. Step 2 cuts the middle square of 224 by 224 out of it.",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_softmax():
    quick_softmax = L.choice(
        "quick-softmax",
        question="A model gives three classes the logits 2, 0 and 0. Which probability does softmax give the first class?",
        options={
            "right": "0.79",
            "all": "1",
            "even": "0.33",
            "logit": "2",
        },
        correct="right",
        why={
            "right": "e² / (e² + e⁰ + e⁰) = 7.39 / 9.39 = 0.79: the largest logit gets the largest probability, and the other two get 0.11 each.",
            "all": "Softmax gives every class a share: e to the power of each logit, divided by their sum. 7.39 / (7.39 + 1 + 1) = 0.79, and 0.11 for each of the others.",
            "even": "The logits differ, so the probabilities do too: e² / (e² + e⁰ + e⁰) = 0.79 for the first class.",
            "logit": "A probability lies between 0 and 1. Softmax takes e to the power of each logit and divides by their sum: 7.39 / 9.39 = 0.79.",
        },
    )
    quick_softmax
    return (quick_softmax,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A pretrained image classifier names a thousand kinds of things in a photo, without any training of your own: a photo prepared as in the previous lesson, a few lines of code, and you see what it thinks. But what does it mean when it is "right"? A photo of a cup of espresso shows a cup too, and one cat can fit several classes that look alike. That is why classifiers are judged in two ways: by their best guess, and by their five best guesses.

    **An everyday example.** A friend hears a bird in the garden and says: "A blackbird, or else a song thrush, or a robin." It was a song thrush. The first guess is wrong, but the right bird is on the short list, and that tells you more about how well your friend knows birds than only "wrong".
    """)
    return


@app.cell(hide_code=True)
def predict_cat():
    guess_cat = L.guess(
        "cat",
        question="A classifier trained on ImageNet, with 1,000 classes, looks at the cat photo of the previous lesson. Its best guess is a class of cat. How sure is it of that guess?",
        options={
            "sure": "More than 90 percent",
            "forty": "About 40 percent",
            "ten": "About 10 percent",
            "tiny": "Less than 1 percent",
        },
        reveal="**About 40 percent.** Its best guess is Egyptian cat, at 0.39; tiger cat follows at 0.39 too, and tabby at 0.22. Together these three hold 99.5 percent: the model has no doubt that it sees a cat, only about which of three of ImageNet's classes of house cat it is. Those classes look alike, so a best guess of only 40 percent is no sign of a bad model.",
    )
    guess_cat
    return (guess_cat,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **ImageNet** is a collection of photos from the web, to each of which people gave one class out of 1,000, its label: from tench and goldfish to toilet tissue, with 118 breeds of dog among them. The version most image classifiers are trained and compared on has 1.28 million training photos and 50,000 validation photos. A classifier trained on it, here ResNet-18 with 11.7 million parameters, gives a photo a logit per class: 1,000 numbers. Softmax makes them probabilities that add up to 1, as in [Many classes: softmax](/course/open/ml-softmax?lang=en). How ResNet-18 computes its logits, later lessons show; here it is a model you use.

        **Top-1 and top-5.** The class with the highest probability is the model's best guess, its top-1; the five classes with the highest probabilities are its top-5. **Top-k accuracy** is the share of photos whose label is among their k best guesses. Top-1 accuracy is the accuracy of [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en); top-5 accuracy also counts a photo as right when the label is on the model's short list. For ImageNet both are reported, because classes look alike, a photo often shows more than one thing, and each photo has only one label. On ImageNet's 50,000 validation photos, torchvision measured a top-1 accuracy of 69.8 percent for this ResNet-18, and a top-5 accuracy of 89.1 percent.

        **In code.** The model gives logits as a tensor of shape `(1, 1000)`. `logits[0].numpy()` makes a NumPy array of the first row, so that softmax works as you know it: `np.exp(z - z.max())`, divided by its sum. Then pairs: `(p[i], names[i])` for every class. `sorted` sorts pairs by their first value, the probability, and `reverse=True` turns the order around, from large to small, so that `[:5]` takes the five best guesses.

        **Download once.** ResNet-18's file is 47 MB. The button in the worked example downloads it once from Hugging Face, from `timm`, a library of image models that keeps a copy of torchvision's own weights, at a fixed version, and checks that every byte is the course's pinned file. The code of ResNet-18 is open, under a BSD license, but its weights were learned from ImageNet's photos, whose rights lie with their photographers, so whether the weights may be passed on is unclear. That is why the course does not ship them, and why they stay on your computer.
        """),
        L.watch_out("The probabilities always add up to 1, also for a photo of something that is in none of the 1,000 classes. A high probability means that the model prefers this class to the other 999, not that it is right."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **ImageNet** | a collection of photos, each with one class out of 1,000 as its label; the photos most image classifiers are trained and compared on | |
        | **top-1, top-5** | the class with the highest probability; the five classes with the highest probabilities | `sorted(pairs, reverse=True)[:5]` |
        | **top-k accuracy** | the share of photos whose label is among their k best guesses | |
        | **`reverse=True`** | makes `sorted` go from large to small | `sorted([0.2, 0.5, 0.3], reverse=True)` gives `[0.5, 0.3, 0.2]` |
        | **`.numpy()`** | makes a NumPy array of a tensor, with the same numbers | `logits[0].numpy()` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_bars():
    _sep = "."
    mo.vstack([
        mo.md("**The five best guesses of ResNet-18**"),
        mo.hstack([
            mo.Html(C.bars("coffee", 5, {"title": "The coffee photo", "alt": "Five bars, the best guess first: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["coffee"][:5]) + "."}, _sep)),
            mo.Html(C.bars("cat", 5, {"title": "The cat photo", "alt": "Five bars, the best guess first: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["cat"][:5]) + "."}, _sep)),
        ], wrap=True, justify="start", gap=1.5),
        mo.md(r"""
        In the coffee photo, espresso has 0.85: right and sure. Its other guesses are brown drinks and dishes, and a cup, which is in the photo as well. In the cat photo, three classes of cat are almost side by side, and lynx, a wild cat, is far behind. The probabilities are rounded to two decimals, and below 0.01 the bars say so: lynx has 0.0045, Persian cat 0.0002.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Give the cat photo the label tabby, as someone who labels photos for ImageNet might. With the slider you choose k; you see the k best guesses, and whether tabby is among them.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _say(k):
        _k = int(k)
        _in = C.rank("cat") <= _k
        _held = C.held("cat", _k)
        _part = "more than 99.9" if _held >= 0.9995 else C.percent(100 * _held, ".")
        if _in:
            _verdict = f" Tabby is among them, at place {C.rank('cat')}: for top-{_k}, the cat photo counts as right."
        elif _k == 1:
            _verdict = " Tabby is not the best guess: for top-1, the cat photo counts as wrong."
        else:
            _verdict = f" Tabby is not among them: for top-{_k}, the cat photo counts as wrong."
        _start = "The best guess holds" if _k == 1 else f"The {_k} best guesses hold"
        return f"{_start} {_part} percent of the probability." + _verdict

    def _frame(k):
        _html = C.bars("cat", int(k), {"title": f"The top-{int(k)} of the cat photo", "alt": f"The {int(k)} best guesses of ResNet-18 for the cat photo, as bars; tabby, the label, is marked with a tick when it is among them."}, ".", mark="tabby")
        return _html, _say(k)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="",
                          task="Find the smallest k at which the cat photo counts as right, with tabby as its label.")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    With tabby as its label, the cat photo is wrong for top-1 and top-2, and right from top-3 on. The three classes of house cat hold 99.5 percent together: the model is sure that it sees a cat, and unsure which of ImageNet's three classes it is. Top-5 accuracy forgives that kind of doubt; top-1 accuracy does not. From k = 5 on, the share hardly grows: the new bars are almost empty, because the last fraction of a percent is spread thinly over the other 996 classes.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To let an ImageNet classifier `model` look at a photo:

    1. Prepare the photo as in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en): `x` of shape `(1, 3, 224, 224)`.
    2. The logits: `logits = model(x)`, inside `with torch.no_grad():`, of shape `(1, 1000)`.
    3. The probabilities: `z = logits[0].numpy()`, `e = np.exp(z - z.max())` and `p = e / e.sum()`.
    4. The top-1: `names[np.argmax(p)]`. The top-5: `sorted([(p[i], names[i]) for i in range(len(names))], reverse=True)[:5]`.
    5. Judge it: a photo is right for top-k when its label is among its k best guesses, `label in guesses[:k]`: `in` works on a list as on the keys of a dictionary. Top-k accuracy is the share of the photos that is right.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell reads the names of the 1,000 classes, one per line in the course's file, and prints how many there are and the name with index 967. It prepares the coffee photo, step 1, and prints the shape of the logits, step 2. It makes the probabilities, step 3, and prints their sum. Then it prints the top-1 with its probability, and the top-5, step 4. `resnet` is ResNet-18, which the cell below loads once it is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download ResNet-18, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 is not on your computer yet. The button downloads it once, 47 MB, from Hugging Face. The worked example needs it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading ResNet-18** · {done} of {total} MB",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 47 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*ResNet-18 is on your computer: 11.7 million parameters, with torchvision's weights, trained on ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_coffee(resnet):
    import numpy as np
    import torch
    from PIL import Image

    with open("data/imagenet-classes.txt") as f:
        names = [line.strip() for line in f]
    print(len(names), names[967])
    # 1. the five steps of the previous lesson
    photo = Image.open("data/photos/coffee.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    # 2. the logits: one per class
    with torch.no_grad():
        logits = resnet(x)
    print(logits.shape)
    # 3. the probabilities, with softmax; .round(3): to three decimals
    z = logits[0].numpy()
    e = np.exp(z - z.max())
    p = e / e.sum()
    print(p.sum().round(3))
    # 4. the top-1, and the top-5: pairs of probability and name, from large to small
    print(names[np.argmax(p)], p.max().round(3))
    pairs = [(p[i], names[i]) for i in range(len(names))]
    for prob, name in sorted(pairs, reverse=True)[:5]:
        print(name, prob.round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The file holds 1,000 names, and the one with index 967 is espresso. The logits have shape `(1, 1000)`: a batch of one photo, a logit per class. The probabilities add up to 1. The top-1 is espresso, at 0.852, and that is right. The other four of the top-5 are chocolate sauce, consomme (a clear soup), soup bowl and cup: brown drinks and dishes, and a cup, which is in the photo too. With the label espresso, the coffee photo is right for top-1, and so for top-5 as well.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rank():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Right or wrong

    The upright piece of the cat from the previous lesson has the label tabby. ResNet-18's five best guesses for it are, in this order: tiger cat, tabby, Egyptian cat, lynx and tiger.
    """)
    return


@app.cell(hide_code=True)
def choice_rank_cell():
    choice_rank = L.choice(
        "rank",
        question="How does the upright piece count?",
        options={
            "right": "Wrong for top-1, right for top-5",
            "both": "Right for top-1 and for top-5",
            "neither": "Wrong for top-1 and for top-5",
            "flip": "Right for top-1, wrong for top-5",
        },
        correct="right",
        why={
            "right": "For top-1 only the best guess counts, tiger cat: not the label. For top-5 the label must be among the five best guesses, and tabby is at place 2.",
            "both": "For top-1 only the best guess counts, and that is tiger cat, not tabby. Tabby is at place 2, so it counts only from top-2 on.",
            "neither": "For top-5 the label only has to be among the five best guesses, and tabby is at place 2: right for top-5.",
            "flip": "A photo that is right for top-1 is always right for top-5 too: the best guess is one of the five best. Here it is the other way around: the best guess is tiger cat, and tabby is at place 2.",
        },
    )
    choice_rank
    return (choice_rank,)


@app.cell(hide_code=True)
def text_exercise_topk():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The k best guesses

        Complete `top_k(p, names, k)`, which gives back the `k` most likely classes as pairs of probability and name, the best guess first: step 4 of the recipe, for any `k`. Replace the two `...`. The check calls it with the probabilities of four classes of cat:

        ```python
        top_k(np.array([0.22, 0.39, 0.38, 0.01]), ["tabby", "tiger cat", "Egyptian cat", "lynx"], 2)
        ```

        That should give `[(0.39, 'tiger cat'), (0.38, 'Egyptian cat')]`. The check also tries other numbers of classes, and other values of `k`.
        """),
        L.task("Make `top_k(p, names, k)` give back the `k` best guesses, the best first."),
    ])
    return


@app.cell(hide_code=True)
def exercise_topk():
    topk_code = L.code_box("topk", lines=3)
    topk_code
    return (topk_code,)


@app.cell(hide_code=True)
def check_topk(topk_code):
    result_topk = L.check("topk", topk_code.value)
    result_topk
    return (result_topk,)


@app.cell(hide_code=True)
def hints_topk():
    L.hints(
        "topk",
        [
            "Step 4 of the recipe, with `k` instead of 5, and `len(names)` for the number of classes.",
            "`pairs` gets a pair `(p[i], names[i])` for every `i` from 0 to `len(names) - 1`: a list comprehension. Then sort the pairs from large to small, and take the first `k`.",
            "`pairs = [(p[i], names[i]) for i in range(___)]` and `return sorted(pairs, reverse=___)[:___]`.",
        ],
        why="`sorted` compares pairs by their first value, the probability, so with `reverse=True` the most likely class comes first: `[(0.39, 'tiger cat'), (0.38, 'Egyptian cat'), (0.22, 'tabby'), (0.01, 'lynx')]`. `[:k]` takes the first `k` of them.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_accuracy():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Top-k accuracy

        Four photos, with ResNet-18's five best guesses for each: the coffee photo, the cat photo, the upright piece of the cat from the previous lesson, and the drawing of fruit from [Another way to write a color: HSV](/course/open/cv-color-spaces?lang=en). And the label of each photo; for the drawing, orange, one of its fruits that ImageNet has a class for.

        ```python
        guesses = [["espresso", "chocolate sauce", "consomme", "soup bowl", "cup"],
                   ["Egyptian cat", "tiger cat", "tabby", "lynx", "Persian cat"],
                   ["tiger cat", "tabby", "Egyptian cat", "lynx", "tiger"],
                   ["balloon", "ping-pong ball", "Granny Smith", "punching bag", "croquet ball"]]
        labels = ["espresso", "tabby", "tabby", "orange"]
        ```

        Complete `top_k_accuracy(guesses, labels, k)`, which gives back the share of the photos whose label is among their first `k` guesses. Replace the two `...`. `top_k_accuracy(guesses, labels, 1)` should give `0.25`: only the coffee photo is right with its best guess. The check also tries other values of `k`, and other photos.
        """),
        L.task("Make `top_k_accuracy(guesses, labels, k)` give back the share of the photos that is right for top-k."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracy():
    accuracy_code = L.code_box("accuracy", lines=6)
    accuracy_code
    return (accuracy_code,)


@app.cell(hide_code=True)
def check_accuracy(accuracy_code):
    result_accuracy = L.check("accuracy", accuracy_code.value)
    result_accuracy
    return (result_accuracy,)


@app.cell(hide_code=True)
def hints_accuracy():
    L.hints(
        "accuracy",
        [
            "Step 5 of the recipe. In the loop, `i` runs over the photos: `labels[i]` is the label of photo `i`, and `guesses[i]` its guesses, the best first.",
            "The first `k` guesses of photo `i` are the slice `guesses[i][:k]`, and `in` tells whether the label is among them, as it does for the keys of a dictionary. Count the photos for which it is, and divide by the number of photos.",
            "`if labels[i] in guesses[i][:___]:` with `right = right + 1` under it, and `return right / len(___)`.",
        ],
        why="With `k` 1 only the coffee photo is right: 1 of 4, so 0.25. With `k` 2 the upright piece of the cat counts too, with tabby at place 2: 0.5. From `k` 3 the whole cat counts as well, 0.75; the drawing never does, because orange is not even in its top 5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_balloon():
    _sep = "."
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. A balloon

        ResNet-18 looks at the drawing of fruit, prepared as in the previous lesson, so it sees only the middle square: two red apples and part of a third, the orange, half the lime and a small piece of the lemon, on a checked tablecloth. Its best guess is balloon, at 0.73; ping-pong ball follows, and orange comes only at place 8, with 0.001.
        """),
        mo.Html(C.bars("fruit", 5, {"title": "The drawing of fruit", "alt": "Five bars, the best guess first: " + ", ".join(f"{_n} {C.shown(_p, _sep)}" for _n, _p in C.REFERENCE["fruit"][:5]) + "."}, _sep)),
    ])
    return


@app.cell(hide_code=True)
def explain_balloon_cell():
    explain_balloon = L.explain(
        "balloon",
        question="Why is the model so sure of something so wrong? Think of what softmax does with 1,000 classes, of what the model learned from, and of what is in the square it gets.",
        model_answer="**The model has to choose among its 1,000 classes, and softmax always gives probabilities that add up to 1,** so there is no answer such as \"none of these\": 0.73 means that balloon fits better than the other 999 classes, not that it is right. **The model learned from photos,** and the drawing is no photo: smooth, shiny balls with a bright spot, on a plain pattern, look more like ImageNet's balloons and ping-pong balls than like its photos of fruit. **And a classifier gives the whole image one class,** while even the middle square it gets shows six pieces of fruit, some only in part, and ImageNet has no class for some of them: there is no red apple and no lime among its 1,000 classes.",
        points=[
            "you say that softmax always gives probabilities that add up to 1, so the model must choose among its classes even when none fits",
            "you say that the model learned from photos, and that the drawing looks more like the balloons and balls in them than like their fruit",
            "you say that the square shows several pieces of fruit, some without a class of their own, while the model gives one class",
        ],
    )
    explain_balloon
    return (explain_balloon,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A photo prepared as in [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en) goes into ResNet-18, which gives a logit for each of ImageNet's 1,000 classes; softmax, as in [Many classes: softmax](/course/open/ml-softmax?lang=en), makes them probabilities, and sorting gives the best guesses. Top-1 accuracy is the accuracy of [The loss you train on and the metric you report](/course/open/ml-loss-vs-metric?lang=en); top-5 accuracy also forgives a label at place 2 to 5, which matters when classes look alike and a photo shows more than one thing. And the probabilities always add up to 1, also for a drawing of fruit that becomes a balloon. In the next lesson, *What a vision model sees: features and a linear probe*, you look at the numbers a large pretrained vision model computes for a photo, its features, and use them to learn classes of your own from a few examples.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_rank, explain_balloon, result_accuracy, result_topk):
    L.wrap_up(choice_rank, result_topk, result_accuracy, explain_balloon)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
