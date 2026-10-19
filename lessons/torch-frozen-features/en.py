# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-frozen-features. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-frozen-features", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using a pretrained model](/course/open/torch-pretrained?lang=en) and [Reading loss curves](/course/open/torch-loss-curves?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="The model card of the course's digit reader says: only digits as in QMNIST. Your digits come from an old scanner that puts grey speckles on every image. What do you do first?",
        options={
            "trust": "Use it as it is: speckled digits are still digits, so it reads them",
            "right": "Test it on some of yours: only that shows how well it reads them",
            "never": "Leave it: the card says it is not for your digits",
        },
        correct="right",
        why={
            "trust": "The card names this as a limit: what it does on other images, you only know when you try it. See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "never": "A limit on the card tells you to check, not that the model is of no use. See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "right": "Step 4 of the recipe. On 1,000 speckled test digits, it reads 729 right: worse than the 929 on clean ones, but far from nothing.",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="A network with 52,000 parameters trains for many epochs on only 100 images. What do you expect of its two loss curves?",
        options={
            "both": "Both fall to about 0: 100 images are easy to learn",
            "right": "Only the training loss falls to about 0: it learns the 100 by heart",
            "none": "Neither falls: 100 images are too few for a network to learn anything",
        },
        correct="right",
        why={
            "both": "On the 100 images, yes. But on images it has not seen, it does much worse: it learned these 100 by heart. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "none": "100 images are plenty to learn by heart: the training loss falls fine. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "Overfitting: with many parameters and few examples, the network learns the examples, not the task.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The digit reader reads speckled digits from the old scanner worse: 729 of 1,000. You have only 100 labeled ones. You could train a new network on them. Or you keep what the digit reader already knows about digits, learned from 6,000 of them, and let only a new last layer learn your task.

        **An everyday example.** An experienced cook starts in a new restaurant. They do not learn to cook again: knives, heat and taste they already know. They only learn the new menu, in a few days.
        """),
        mo.Html(C.speckle_picture({"title": "Three test digits", "clean": "as the course's set has them", "scanner": "as the old scanner gives them",
                           "alt": "Three handwritten digits, 7, 2 and 1, clean on top, and below the same three with grey speckles all over the image."})),
    ])
    return


@app.cell(hide_code=True)
def predict_frozen():
    guess_frozen = L.guess(
        "frozen",
        question="With the 100 labeled digits of the old scanner, two ways: a new network trained from scratch, or the digit reader with all its layers kept as they are and only a new last layer trained. Which reads more of 1,000 speckled test digits right?",
        options={
            "scratch": "From scratch: it learns everything for these digits",
            "same": "About as many: both learn from the same 100 digits",
            "right": "The digit reader with a new last layer: less has to learn",
        },
        reveal=f"**The digit reader with a new last layer: {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]}**, over 5 seeds, against {C.REFERENCE['scratch'][0]} to {C.REFERENCE['scratch'][1]} from scratch, and {C.REFERENCE['as-is'][0]} for the digit reader as it is. From scratch, 52,000 numbers learn from 100 images; with the borrowed layers, only 650 do, on features that already describe digits.",
    )
    guess_frozen
    return (guess_frozen,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Features and a last layer.** All layers of the digit reader before its last ones turn an image into 64 numbers, its hidden layer: features that describe the digit. The last layers, `embed` and `out`, turn those 64 numbers into 10 logits. So the network is a **feature extractor** with a small last part on top.

        **Freeze.** `p.requires_grad = False` for a parameter is the opposite of `requires_grad=True` in [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en): PyTorch computes no gradient for it, so the optimizer leaves it as it is. That is **freezing**: the borrowed layers keep what they learned.

        **A new last layer.** `model.embed = nn.Identity()` passes the 64 hidden numbers on as they are, and `model.out = nn.Linear(64, 10)` is a new layer from those 64 numbers to 10 logits. A new layer has `requires_grad` on: it is the only part that learns, 64 · 10 + 10 = 650 numbers.

        Reusing what a model learned on one task for another is **transfer learning**. It works when the new data are like the old: speckled digits, or a new question about digits. Images that look nothing like digits need other features.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **transfer learning** | | reusing what a model learned on one task or dataset for another | |
        | **feature extractor** | | a pretrained network used only to turn inputs into features, on which a new small model learns your task | |
        | **freezing** | | keeping parameters as they are while training: no gradient, so the optimizer leaves them | `p.requires_grad = False` |
        | `nn.Identity()` | n n identity | a layer that passes its input on unchanged | `model.embed = nn.Identity()` |
        | `named_parameters()` | named parameters | every parameter of a model, with its name, such as `out.weight` | `for name, p in model.named_parameters():` |
        """),
        L.watch_out("Freeze first, then make the new layer. Freezing sets `requires_grad = False` for every parameter the model has at that moment: freeze after you made the new layer, and it is frozen too, and nothing learns."),
        mo.md(r"""
        The picture: the digit reader after the change. Its first three layers are frozen; the new `out` is the only layer that learns.
        """),
        mo.Html(C.layers_picture({"title": "Frozen layers and a new last layer", "conv1": "convolution, 8 channels", "conv2": "convolution, 16 channels",
                                  "hidden": "64 features of the digit", "out": "new: from 64 to 10 logits", "frozen": "frozen",
                                  "learns": "learns", "gone": "embed passes the 64 features on: nn.Identity()",
                                  "alt": "Four layers from top to bottom: conv1, conv2 and hidden are frozen and keep their numbers; the new out, from 64 to 10 logits, is the only layer that learns. The old embed passes the 64 features on unchanged."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To reuse a pretrained network for a new task with few labeled examples:

    1. Load the pretrained network, as in [Using a pretrained model](/course/open/torch-pretrained?lang=en).
    2. Freeze every parameter: `for p in model.parameters(): p.requires_grad = False`.
    3. Replace its last part: `model.embed = nn.Identity()`, and a new `model.out = nn.Linear(64, k)` for $k$ answers.
    4. Train only the new layer on your labeled examples, and test on others.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The four steps for a new question about clean digits: is the digit odd or even? Two answers, so the new layer goes from 64 to 2. `digit_cnn()` gives the course's digit reader, loaded as in step 3 of the recipe of [Using a pretrained model](/course/open/torch-pretrained?lang=en), in eval mode. The 100 labeled digits are the first 100 training images; the label of each is its digit's remainder after dividing by 2, which `% 2` gives: 1 for odd, 0 for even. `C.train(model, params, X, y, lr, epochs)` is the course's training loop, as in [Project: an MLP that reads handwritten digits](/course/open/torch-mlp-digits?lang=en), but with the optimizer Adam, which a later lesson explains; `params` are the parameters that may learn.
    """)
    return


@app.cell
def example_odd():
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.models import digit_cnn

    _data = np.load("data/qmnist-small.npz")
    _X = torch.tensor(_data["train_images"][:100], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"][:100]) % 2
    _X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    _y_test = torch.tensor(_data["test_labels"]) % 2
    # 1. the pretrained digit reader
    torch.manual_seed(0)
    _model = digit_cnn()
    # 2. freeze every parameter
    for _p in _model.parameters():
        _p.requires_grad = False
    # 3. a new last layer: the 64 hidden numbers to 2 answers, odd or even
    _model.embed = nn.Identity()
    _model.out = nn.Linear(64, 2)
    for _name, _p in _model.named_parameters():
        print(_name, _p.requires_grad)
    # 4. train only the new layer, then test
    C.train(_model, _model.out.parameters(), _X, _y, 0.01, 30)
    print(C.right(_model, _X_test, _y_test), "of", len(_y_test), "right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `named_parameters` gives every parameter with its name: only `out.weight` and `out.bias` still learn. After 30 epochs on 100 digits, the new layer answers odd or even right for 944 of the 1,000 test digits, and for 926 to 952 over 5 seeds. A new network trained from scratch on the same 100 digits gets 802 to 827, over 5 seeds: the borrowed features already tell the digits apart, and odd or even follows from the digit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_head():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Freeze, and a new last layer

        Complete `new_head(model)`: steps 2 and 3 of the recipe, for the 10 digits of the old scanner. It gets the digit reader and gives it back with every old parameter frozen and a new last layer from 64 to 10. The check calls `new_head(reader)` and looks at which parameters still learn, at the shape of the weight of `out`, and at the logits for the first test digit.
        """),
        L.task("Make `new_head` freeze the digit reader and give it a new last layer from 64 to 10."),
    ])
    return


@app.cell(hide_code=True)
def exercise_head():
    box_head = L.code_box("new-head", lines=10)
    box_head
    return (box_head,)


@app.cell(hide_code=True)
def check_head(box_head):
    result_head = L.check("new-head", box_head.value)
    result_head
    return (result_head,)


@app.cell(hide_code=True)
def hints_head():
    L.hints(
        "new-head",
        [
            "Steps 2 and 3 of the worked example, with 10 answers instead of 2.",
            "In the loop: `p.requires_grad = False`.",
            "`nn.Identity()` for `embed`, and `nn.Linear(64, 10)` for `out`, after the loop.",
        ],
        why="The loop freezes all 10 parameters of the digit reader; the new `out` comes after it, so it is not frozen. `nn.Identity()` hands the 64 hidden numbers to the new layer as they are.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### Your new last layer on the old scanner's digits

    The cell below gives the digit reader to your `new_head` once it is right, else to the course's, and trains the new last layer on the 100 labeled digits of the old scanner: {C.HEAD_EPOCHS} epochs with a learning rate of {L.number(C.HEAD_LR, 2)}. For comparison, it trains a new network of the same layers from scratch on the same 100 digits, all of it: {C.SCRATCH_EPOCHS} epochs with a learning rate of {L.number(C.SCRATCH_LR, 3)}. It tests both, and the digit reader as it is, on 1,000 speckled test digits. It takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_frozen(box_head, result_head):
    _d = C.demo(box_head.value if result_head.state == "correct" else None)
    _who = "your `new_head`" if _d["yours"] else "the course's `new_head`"
    mo.md(f"""With {_who}, of the 1,000 speckled test digits:

| | read right |
|---|---|
| the digit reader as it is | {_d['as-is']} |
| a new network, from scratch on the 100 | {_d['scratch']} |
| frozen, with a new last layer trained on the 100 | {_d['frozen']} |

Over 5 seeds, with the course's code: {C.REFERENCE['scratch'][0]} to {C.REFERENCE['scratch'][1]} from scratch, and {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]} frozen with a new last layer.""")
    return


@app.cell(hide_code=True)
def text_exercise_optimizer():
    mo.md(r"""
    ### Exercise 2. Which parameters for the optimizer?
    """)
    return


@app.cell(hide_code=True)
def choice_optimizer_cell():
    choice_optimizer = L.choice(
        "same-optimizer",
        question="In step 4 of the worked example, the optimizer gets `model.out.parameters()`, the new layer's. What changes if you give it `model.parameters()`, all of them, frozen ones too?",
        options={
            "learn": "The frozen layers learn after all: the optimizer changes every parameter it gets",
            "right": "Nothing: the frozen parameters get no gradient, so the optimizer leaves them",
            "error": "PyTorch stops: a frozen parameter cannot go into an optimizer",
        },
        correct="right",
        why={
            "learn": "The optimizer changes a parameter by its gradient, and `backward()` computes none for a parameter with `requires_grad = False`: there is nothing to change it by.",
            "error": "PyTorch takes it without complaint; the optimizer only skips a parameter that has no gradient.",
            "right": "Freezing happens in `requires_grad`, not in the optimizer. Giving it only the new layer's parameters is still clearer, and saves it some work.",
        },
    )
    choice_optimizer
    return (choice_optimizer,)


@app.cell(hide_code=True)
def text_exercise_clothes():
    mo.md(r"""
    ### Exercise 3. Another kind of image
    """)
    return


@app.cell(hide_code=True)
def choice_clothes_cell():
    choice_clothes = L.choice(
        "clothes",
        question="You need a network that reads 10 kinds of clothes, with 100 labeled 28-by-28 images from Fashion-MNIST: grey pictures of clothes, shoes and bags, in the form of MNIST. Does the frozen digit reader with a new last layer beat a new network trained from scratch here too?",
        options={
            "right": "No: its features describe digits, and clothes need others",
            "yes": "Yes: features of one kind of image help with every other kind",
            "more": "Yes, by even more: clothes are larger shapes than digits",
        },
        correct="right",
        why={
            "yes": f"The course measured it, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} to {C.REFERENCE['clothes-frozen'][1]} of 1,000 frozen, against {C.REFERENCE['clothes-scratch'][0]} to {C.REFERENCE['clothes-scratch'][1]} from scratch. The 64 features say how a digit is written, not whether a picture shows a shoe or a shirt.",
            "more": f"Larger shapes need features of their own. The course measured it, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} to {C.REFERENCE['clothes-frozen'][1]} of 1,000 frozen, against {C.REFERENCE['clothes-scratch'][0]} to {C.REFERENCE['clothes-scratch'][1]} from scratch.",
            "right": f"The course measured it, over 5 seeds: {C.REFERENCE['clothes-frozen'][0]} to {C.REFERENCE['clothes-frozen'][1]} of 1,000 frozen, against {C.REFERENCE['clothes-scratch'][0]} to {C.REFERENCE['clothes-scratch'][1]} from scratch. Transfer learning helps when the new data are like the old.",
        },
    )
    choice_clothes
    return (choice_clothes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The digit reader of [Using a pretrained model](/course/open/torch-pretrained?lang=en), now as a feature extractor, and `requires_grad` of [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en), now to freeze it. With few examples, 650 numbers that learn overfit far less than 52,000, as the curves of [Reading loss curves](/course/open/torch-loss-curves?lang=en) would show. Next, [Reusing a trained network: freeze, then fine-tune](/course/open/torch-fine-tune?lang=en) lets the frozen layers learn a little too.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_head, choice_optimizer, choice_clothes):
    L.wrap_up(result_head, choice_optimizer, choice_clothes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
