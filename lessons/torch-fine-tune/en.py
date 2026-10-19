# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-fine-tune. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-fine-tune", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en) and [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frozen():
    quick_frozen = L.choice(
        "quick-frozen",
        question="With 100 labeled digits of the old scanner, the frozen digit reader with a new last layer read 860 to 883 of 1,000 right, a new network from scratch 629 to 669. Why the difference?",
        options={
            "longer": "It trained longer: more epochs than the network from scratch",
            "right": "Only 650 numbers learn, on features that already describe digits",
            "bigger": "It has more parameters, so it can learn more",
        },
        correct="right",
        why={
            "longer": "The other way round: 30 epochs for the new last layer, 60 for the network from scratch. See [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en).",
            "bigger": "Both have the same layers, and so just as many parameters; in the frozen one, only 650 of them learn. See [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en).",
            "right": "From scratch, 52,000 numbers learn from 100 images, and learn those by heart. In this lesson the borrowed layers learn too, but only a little.",
        },
    )
    quick_frozen
    return (quick_frozen,)


@app.cell(hide_code=True)
def quick_check_rate():
    quick_rate = L.choice(
        "quick-big-lr",
        question="A loss curve jumps up and down, or grows. What does that tell you?",
        options={
            "small": "The learning rate is too small: the steps are too short",
            "few": "The training data has too few examples",
            "right": "The learning rate is too big: each step goes too far",
        },
        correct="right",
        why={
            "small": "Steps that are too short give a curve that falls slowly, but it falls. See [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en).",
            "few": "Too few examples give a training loss that falls while the validation loss stays high: overfitting. See [Reading loss curves](/course/open/torch-loss-curves?lang=en).",
            "right": "Each step jumps past the bottom of the valley. In this lesson, a step that goes too far does damage of another kind: it carries borrowed features away from what they knew.",
        },
    )
    quick_rate
    return (quick_rate,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The frozen digit reader with a new last layer reads 871 of the 1,000 speckled test digits right. Its features stayed as they were: learned on 6,000 clean digits, they never saw a speckle. Let them adapt to the old scanner's digits, and it can read more of them. But only a few more: with 100 examples, layers that change a lot learn those 100 by heart, as a network from scratch does.

    **An everyday example.** A suit off the rack that almost fits. A tailor does not sew a new one: first you pick the size that fits best, then the tailor takes in the sleeves and the waist a little. Cut away too much, and the suit is ruined.
    """)
    return


@app.cell(hide_code=True)
def predict_skip():
    guess_skip = L.guess(
        "skip-head",
        question="Fine-tuning has two phases: first only the new last layer learns, then every layer, with a small learning rate. What if you skip phase 1: a new last layer, every layer unfrozen at once, then 30 epochs with the small learning rate 0.001 on the same 100 digits? Phase 1 alone gave 860 to 883 of 1,000.",
        options={
            "right": "Fewer: the borrowed layers follow the errors of a random new layer",
            "more": "More: every layer adapts to the speckles from the very first step, not only the last one",
            "same": "About as many: the same digits, and the same small learning rate",
        },
        reveal=f"**Fewer: {C.REFERENCE['no-head'][0]} to {C.REFERENCE['no-head'][1]}**, over 5 seeds, fewer than phase 1 alone on every seed; with 60 epochs, {C.REFERENCE['no-head-60'][0]} to {C.REFERENCE['no-head-60'][1]}. A new last layer starts with random weights, so at first its errors say nothing about digits. Their gradients reach the borrowed layers too: the first steps move the features toward what suits a random layer, and after 30 epochs the borrowed weights have moved about twice as far as when phase 1 comes first. With phase 1 first: {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]}.",
    )
    guess_skip
    return (guess_skip,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Two phases.** Start where the previous lesson ended: the digit reader frozen, with a new last layer trained on your examples. That is phase 1. Then unfreeze every parameter, `p.requires_grad = True`, and train them all further, with a learning rate about ten times smaller: phase 2. The borrowed features move a little toward your data. That is **fine-tuning**.

        **Why phase 1 first.** In phase 2, every layer learns from the errors of the last layer. After phase 1, those errors are about the speckled digits, so the borrowed layers learn what helps to read them. A last layer that is still random makes errors that say nothing, and the borrowed layers would follow those.

        **Why a small learning rate.** The borrowed features are already good: they need to change only a little. Large steps carry them far from what 6,000 digits taught them, toward what fits only these 100.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **fine-tuning** | | training all layers of a pretrained network further on the data of a new task, with a small learning rate | `p.requires_grad = True`, then train |
        """),
        L.watch_out("Unfreezing alone is not enough: `train` must get every parameter too, `model.parameters()`. Give it only `model.out.parameters()`, and phase 2 changes only the last layer again."),
        mo.md(r"""
        The picture: the digit reader in the two phases.
        """),
        mo.Html(C.phases_picture({"title": "The two phases of fine-tuning", "phase1": "phase 1", "rate1": "learning rate 0.01",
                                  "phase2": "phase 2", "rate2": "learning rate 0.001", "epochs": "30 epochs", "learns": "learns", "frozen": "frozen",
                                  "alt": "Two columns with the four layers conv1, conv2, hidden and out. In phase 1, with a learning rate of 0.01, conv1, conv2 and hidden are frozen and only out learns. In phase 2, with a learning rate of 0.001, all four learn."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To fine-tune a pretrained network for a new task:

    1. Phase 1, as in [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en): freeze every parameter, give the network a new last layer, and train only that layer.
    2. Unfreeze every parameter: `for p in model.parameters(): p.requires_grad = True`.
    3. Phase 2: train all parameters further, with a learning rate about ten times smaller than in phase 1.
    4. Test on examples it did not train on, and compare with phase 1 alone.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The four steps for the 10 kinds of clothes of Fashion-MNIST, from its first 100 training images, as in exercise 3 of the previous lesson: there, phase 1 alone read 506 to 539 of the 1,000 test images right, over 5 seeds, and a new network from scratch 617 to 713. `C.train(model, params, X, y, lr, epochs)` is the course's training loop of the previous lesson.
    """)
    return


@app.cell
def example_clothes():
    import numpy as np
    import torch
    from torch import nn
    from everything_ai.models import digit_cnn

    _data = np.load("data/fashion-mnist-small.npz")
    _X = torch.tensor(_data["train_images"][:100], dtype=torch.float32) / 255
    _y = torch.tensor(_data["train_labels"][:100])
    _X_test = torch.tensor(_data["test_images"], dtype=torch.float32) / 255
    _y_test = torch.tensor(_data["test_labels"])
    # 1. phase 1: the digit reader frozen, a new last layer, and only that layer learns
    torch.manual_seed(0)
    _model = digit_cnn()
    for _p in _model.parameters():
        _p.requires_grad = False
    _model.embed = nn.Identity()
    _model.out = nn.Linear(64, 10)
    C.train(_model, _model.out.parameters(), _X, _y, 0.01, 30)
    print("after phase 1:", C.right(_model, _X_test, _y_test), "of", len(_y_test), "right")
    # 2. unfreeze every parameter
    for _p in _model.parameters():
        _p.requires_grad = True
    for _name, _p in _model.named_parameters():
        print(_name, _p.requires_grad)
    # 3. phase 2: every parameter learns, with a learning rate ten times smaller
    C.train(_model, _model.parameters(), _X, _y, 0.001, 30)
    # 4. test again
    print("after phase 2:", C.right(_model, _X_test, _y_test), "of", len(_y_test), "right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Phase 2 moved the digit features toward clothes: 586 right instead of 533. Over 5 seeds: 566 to 593 instead of 506 to 539, more on every seed. Still, a new network from scratch does better with these 100 images: 617 to 713. In 30 epochs of small steps, features of digits do not become features of clothes. Fine-tuning adapts features that almost fit; it does not replace features that do not fit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tune():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The two phases

        Complete `fine_tune(model, train)`, for the 100 labeled digits of the old scanner: steps 1 to 3 of the recipe. `model` is the digit reader as the previous lesson leaves it: frozen, with a new last layer from 64 to 10. `train(params, lr, epochs)` is `C.train` with the model and the 100 digits already filled in: it trains the parameters `params` of `model`, with learning rate `lr`, for `epochs` epochs. So it takes three values, not six. The check calls `fine_tune(reader, train)` with a `train` that trains nothing, but notes for each call which parameters it would change, with which learning rate and for how many epochs.
        """),
        L.task("Make `fine_tune` train first only the new last layer, then every layer with a learning rate ten times smaller."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tune():
    box_tune = L.code_box("fine-tune", lines=9)
    box_tune
    return (box_tune,)


@app.cell(hide_code=True)
def check_tune(box_tune):
    result_tune = L.check("fine-tune", box_tune.value)
    result_tune
    return (result_tune,)


@app.cell(hide_code=True)
def hints_tune():
    L.hints(
        "fine-tune",
        [
            "Steps 1 to 3 of the worked example do the same for clothes, with `C.train`. Here `train` already knows the model and the data: it gets only the parameters, the learning rate and the epochs.",
            "Phase 1 gives `train` the parameters of the new layer, `model.out.parameters()`; phase 2 gives it every parameter, `model.parameters()`. In the loop, unfreezing is freezing the other way round.",
            "`train(model.out.parameters(), 0.01, 30)`, then `p.requires_grad = True` in the loop, then `train(..., 0.001, 30)` with every parameter.",
        ],
        why="Phase 1 trains the new last layer on the frozen features, until its errors say something about the speckled digits. The loop lets every parameter learn again, and phase 2 gives them all to `train`, with steps ten times smaller, so that the borrowed features change only a little.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your two phases on the old scanner's digits

    The cell below hands the course's real training loop, on the 100 labeled digits, to your `fine_tune` once it is right, else to the course's, and tests the result on the 1,000 speckled test digits, next to phase 1 alone. It takes a few seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_tune(box_tune, result_tune):
    _d = C.demo(box_tune.value if result_tune.state == "correct" else None)
    _who = "your `fine_tune`" if _d["yours"] else "the course's `fine_tune`"
    mo.md(f"""With {_who}, of the 1,000 speckled test digits:

| | read right |
|---|---|
| phase 1 alone: frozen, with a new last layer | {_d['frozen']} |
| both phases: fine-tuned | {_d['tuned']} |

Over 5 seeds, with the course's code: {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]} after phase 1 alone, {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]} fine-tuned. On every seed, fine-tuning reads {C.REFERENCE['gain'][0]} to {C.REFERENCE['gain'][1]} digits more: a small gain, on top of what the borrowed features already gave.""")
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.md(r"""
    ### Exercise 2. The learning rate of phase 2
    """)
    return


@app.cell(hide_code=True)
def choice_rate_cell():
    choice_rate = L.choice(
        "big-lr",
        question="A colleague keeps the learning rate of phase 1 for phase 2 as well: 0.01 instead of 0.001. What do you expect, over 5 seeds?",
        options={
            "faster": "Better: larger steps let the borrowed features adapt to the speckles faster",
            "right": "Worse on most seeds: the features move too far",
            "same": "About the same: after phase 1 there is little left to learn",
        },
        correct="right",
        why={
            "faster": f"Faster, but too far. The course measured it, over 5 seeds: {C.REFERENCE['big-lr'][0]} to {C.REFERENCE['big-lr'][1]} with 0.01, against {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]} with 0.001, and {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]} for phase 1 alone.",
            "same": f"In phase 2 every layer learns, and the learning rate decides how far each step goes. The course measured it, over 5 seeds: {C.REFERENCE['big-lr'][0]} to {C.REFERENCE['big-lr'][1]} with 0.01, against {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]} with 0.001.",
            "right": f"The course measured it, over 5 seeds: {C.REFERENCE['big-lr'][0]} to {C.REFERENCE['big-lr'][1]} with 0.01, against {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]} with 0.001. On 4 of the 5 seeds, 0.01 does even worse than phase 1 alone, {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]}: the borrowed features lose part of what 6,000 digits taught them.",
        },
    )
    choice_rate
    return (choice_rate,)


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Exercise 3. Ten times as many examples
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more-digits",
        question=f"With 100 labeled digits of the old scanner: from scratch {C.REFERENCE['scratch'][0]} to {C.REFERENCE['scratch'][1]} of 1,000, phase 1 alone {C.REFERENCE['frozen'][0]} to {C.REFERENCE['frozen'][1]}, fine-tuned {C.REFERENCE['tuned'][0]} to {C.REFERENCE['tuned'][1]}. Now you get 1,000 labeled digits. What do you expect?",
        options={
            "scratch": "From scratch now wins: with 1,000 examples, borrowed features no longer help",
            "only": "Only from scratch gains: the others already knew digits",
            "right": "All three gain, and from scratch gains the most: with 1,000 examples it overfits far less",
        },
        correct="right",
        why={
            "scratch": f"Not with 1,000. The course measured it, over 5 seeds: from scratch {C.REFERENCE['1000-scratch'][0]} to {C.REFERENCE['1000-scratch'][1]}, against {C.REFERENCE['1000-tuned'][0]} to {C.REFERENCE['1000-tuned'][1]} fine-tuned.",
            "only": f"Phase 1 alone and fine-tuning gain too: {C.REFERENCE['1000-frozen'][0]} to {C.REFERENCE['1000-frozen'][1]} and {C.REFERENCE['1000-tuned'][0]} to {C.REFERENCE['1000-tuned'][1]}, against {C.REFERENCE['1000-scratch'][0]} to {C.REFERENCE['1000-scratch'][1]} from scratch. The course measured it, over 5 seeds.",
            "right": f"The course measured it, over 5 seeds: from scratch {C.REFERENCE['1000-scratch'][0]} to {C.REFERENCE['1000-scratch'][1]}, phase 1 alone {C.REFERENCE['1000-frozen'][0]} to {C.REFERENCE['1000-frozen'][1]}, fine-tuned {C.REFERENCE['1000-tuned'][0]} to {C.REFERENCE['1000-tuned'][1]}. With 10 times as many examples, 52,000 numbers overfit far less; the borrowed features still help, and fine-tuning still adds a little.",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Phase 1 is [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en); in phase 2, `requires_grad` of [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) lets every layer learn again, with a learning rate as in [Too small, too big: the learning rate](/course/open/ml-learning-rate?lang=en), now for features that are already good. Later in the course, you fine-tune your own language model, trained on a lot of stories, to follow instructions.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_tune, choice_rate, choice_more):
    L.wrap_up(result_tune, choice_rate, choice_more)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
