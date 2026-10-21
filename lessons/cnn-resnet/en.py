# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-resnet. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution, residual, initialization

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-resnet", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en) and [Residual connections](/course/open/nn-residual?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_stuck():
    quick_stuck = L.choice(
        "quick-stuck",
        question="With 8 convolutions per block and PyTorch's own starting values, your VGG network read 105 of the 1,000 test images right. Why?",
        options={
            "overfit": "It learned the 6,000 training images by heart, and nothing else",
            "right": "From the start, every image gave the same logits, so it never got going",
            "small": "Filters of 3 by 3 see too little for a network that deep",
        },
        correct="right",
        why={
            "overfit": "A network that learned its training images would still tell the test images apart somewhat; 105 is one kind for every image. See [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).",
            "small": "Stacked, the small filters see more, not less: 16 of them see across the whole image. See [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en).",
            "right": "Every layer made the differences between images smaller, until none were left. Today you add what lets them through.",
        },
    )
    quick_stuck
    return (quick_stuck,)


@app.cell(hide_code=True)
def quick_check_through():
    quick_through = L.choice(
        "quick-through",
        question="A residual block gives back $x + f(x)$. At the start, $f$ changes almost nothing: $f(x)$ is close to 0 for every $x$. What does a stack of 8 such blocks give back?",
        options={
            "zero": "Close to 0: 8 blocks of almost nothing",
            "eight": "Close to 8 times $x$: every block adds $x$ once",
            "right": "Close to $x$ itself: each adds almost nothing",
        },
        correct="right",
        why={
            "zero": "Each block adds $f(x)$ to its input, and does not replace it: the input gets through. See [Residual connections](/course/open/nn-residual?lang=en).",
            "eight": "Each block gives its input plus $f$: $x + 0$, then $x + 0$ again. Nothing is doubled. See [Residual connections](/course/open/nn-residual?lang=en).",
            "right": "Every block passes its input on, plus a little. Whatever the starting weights, the image gets through, and on the way back the gradient too.",
        },
    )
    quick_through
    return (quick_through,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your deepest VGG network did not learn at all. In 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun saw the same wall on ImageNet: a plain network of 34 layers made more mistakes than one of 18, even on its own training images, also with batch normalization. Their **ResNet** put a residual connection around every two convolutions, three in its deepest versions, went to 152 layers and won the large image contest of that year. Residual connections are now in almost every deep network, from CNNs to transformers. In this last step of your project you build the residual block of a CNN, and train a network as deep as the VGG network that did not learn.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working"],
                           3, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_residual():
    guess_residual = L.guess(
        "residual",
        question=f"A plain network with {C.BLOCKS} blocks of two convolutions at 8 channels and {C.BLOCKS} at 16, 19 layers with weights, with PyTorch's own starting values, reads 105 of the 1,000 test images right after 5 epochs. You put a residual connection around every block, and change nothing else. How many does it read right?",
        options={
            "same": "About 105 too, as the layers are the same",
            "right": "About as many as your SmallCNN, or a few more",
            "small": "It learns, but stays well below your SmallCNN",
        },
        reveal="**About as many as your SmallCNN, or a few more.** Over 5 seeds, with the course's code, the residual network reads 876 to 895 right, more than your SmallCNN with every seed (864 to 879). The plain network of the same layers reads 105 every time. The residual connections make a depth trainable that was not, and here, where VGG's extra depth gained nothing, it pays off.",
    )
    guess_residual
    return (guess_residual,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The residual block of a CNN.** As in [Residual connections](/course/open/nn-residual?lang=en), a block gives back $x + f(x)$. In a CNN, $f$ is two convolutions of 3 by 3 with a ReLU between them, both from $C$ to $C$ channels with padding 1, so that $f(x)$ has exactly the shape of $x$.

        **Why it keeps working.** A plain block passes on only $f(x)$. With small starting weights, every plain block makes the differences between images smaller, and after 15 layers nothing is left of them. A residual block passes $x$ on as well: whatever $f$ does at the start, the image gets through every block, and on the way back the gradient does too.

        **Not only the starting values.** With the He initialization of [Initialization](/course/open/nn-initialization?lang=en), a plain network of 19 layers learns too: 854 to 871 right over 3 seeds. Make it deeper, 35 layers, and it gets worse, 838 to 847, while the same depth with residual connections reads 874 to 879. He and others saw the same on ImageNet: a deeper plain network trains worse, also with good starting values.

        **ResNet** stacks such blocks in stages: blocks with the same channels and size, and then fewer pixels and more channels. Where that changes, $x$ and $f(x)$ no longer fit, and ResNet gives the residual connection a convolution of 1 by 1 with stride 2. Your network does it more simply: between its two stages, pooling and one plain convolution from 8 to 16 channels. ResNet also puts batch normalization after every convolution; your block leaves it out, to show what the residual connection does on its own.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **residual block** | | two convolutions of 3 by 3 with a ReLU between them, and the input added to their output | `ResBlock(c)` |
        | **ResNet** | res net | the CNNs of He and others (2015): residual blocks in stages, 18 to 152 layers deep | |
        """),
        L.watch_out("`x + f(x)` only works when both have the same shape: the same channels, height and width. A convolution in `f` without padding, or an `f` that ends with more channels than `x` has, and PyTorch stops with an error that the sizes must match. And write `x + f(x)`, not `x += f(x)`: that changes `x` itself, and PyTorch still needs `x` as it was on the way back, for instance for the gradient of the weights of the first convolution."),
        mo.md(r"""
        The picture: how much the logits of a fresh network differ between images, at the start, against its layers with weights, on a log scale. Your network of this lesson with 0 to 8 blocks per stage, plain and residual, on 256 test images. Plain, the differences shrink about a hundredfold every 4 layers, until only rounding is left; with residual connections, they stay.
        """),
        mo.Html(C.spread_picture({"title": "Do the images still differ?", "x": "layers with weights", "y": "spread of the logits",
                                  "plain": "plain", "residual": "residual", "lang": L.lang,
                                  "alt": "Two lines on a log scale against 3 to 35 layers with weights. Residual stays near 0.02 all the way. Plain falls from 0.018 at 3 layers to about one ten-thousandth at 7, a millionth at 11, and 2 to 3 billionths from 19 layers on, the rounding of float32."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A residual block for $C$ channels:

    1. $f(x)$: a convolution of 3 by 3 from $C$ to $C$ channels with padding 1, a ReLU, and a second one like it.
    2. The block gives back $x + f(x)$: a new tensor, not `x += ...`.
    3. Stack blocks with the same channels into a stage; between stages, fewer pixels and more channels, with a convolution where the shapes change.
    4. Train it as any network: with residual connections, depth no longer stops it from learning.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two training images through a convolution to 8 channels, and then 8 blocks of two convolutions, with PyTorch's starting values: first plain, then with the same layers as residual blocks, step 2 of the recipe. How much do the two images still differ, as the mean of the differences of their numbers? The residual line is not done yet: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_through():
    import torch
    from torch import nn

    _X, _y, _, _ = C.fashion()
    _x = _X[:2]
    torch.manual_seed(0)
    _stem = nn.Conv2d(1, 8, 3, padding=1)
    _convs = nn.ModuleList()
    for _ in range(16):
        _convs.append(nn.Conv2d(8, 8, 3, padding=1))
    with torch.no_grad():
        _h = torch.relu(_stem(_x))
        print("start:", round((_h[0] - _h[1]).abs().mean().item(), 4))
        # plain: 8 blocks of two convolutions, each with a ReLU
        for _k in range(0, 16, 2):
            _h = torch.relu(_convs[_k + 1](torch.relu(_convs[_k](_h))))
        print("plain:", round((_h[0] - _h[1]).abs().mean().item(), 4))
        # 2. residual: the same layers, as x + f(x)
        _h = torch.relu(_stem(_x))
        for _k in range(0, 16, 2):
            _h = torch.relu(_convs[_k + 1](torch.relu(_convs[_k](_h))))  # complete: _h + _convs[_k + 1](torch.relu(_convs[_k](_h)))
        print("residual:", round((_h[0] - _h[1]).abs().mean().item(), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At the start, the two images differ by 0.1076 per number. After 8 plain blocks, 16 convolutions, by 0.0: to 4 decimals, the network sees the same for both, and its logits will not tell them apart. As it is, the residual line is the plain one again. Completed, they still differ by 0.1243: every block passes its input on, and the images get through.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. When the shapes do not fit
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "res-shape",
        question="In a residual block, both convolutions of $f$ go to 16 channels, but $x$ has 8. What happens at `x + f(x)`?",
        options={
            "zeros": "PyTorch fills the 8 missing channels of `x` with zeros",
            "first": "The 8 channels of `x` are added to the first 8 of `f(x)`",
            "right": "PyTorch stops with an error",
        },
        correct="right",
        why={
            "zeros": "PyTorch adds nothing by itself: filling up with zeros is a choice you make in code, and one that the ResNet paper tried.",
            "first": "`+` adds number by number, and needs the same shape on both sides, or a 1 where one of them has more. 8 channels against 16 is neither.",
            "right": "Shapes of (B, 8, H, W) and (B, 16, H, W) cannot be added. ResNet gives the residual connection a convolution of 1 by 1 from 8 to 16 channels there.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The residual block

        Complete `ResBlock(c)`: in `__init__` its two convolutions, `conv1` and then `conv2`; in `forward` steps 1 and 2 of the recipe. The check makes blocks after `torch.manual_seed(0)` and compares what comes out, makes `f(x)` 0 to see that the block passes `x` on, runs the backward pass through it, and counts the parameters PyTorch sees. When it is right, the check saves it in your project, as `work/projects/cnn/resnet.py`.
        """),
        L.task("Make `ResBlock` give back `x + f(x)`, with `f` two convolutions of 3 by 3 and a ReLU between them."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("res-block", lines=13)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("res-block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "res-block",
        [
            "In `__init__`, two attributes of `self`, each an `nn.Conv2d` of 3 by 3 from `c` to `c` channels; with `padding=1`, they keep the height and width.",
            "In `forward`, `f(x)` is the first convolution, `torch.relu` around it, and the second convolution around that. Give back `x` plus that.",
        ],
        why="No ReLU after the addition: then the input goes through the block unchanged, as in [Residual connections](/course/open/nn-residual?lang=en). The first ResNet of 2015 had a ReLU there; in 2016, He and others left it out, to keep the direct path clean.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_projection():
    mo.md(r"""
    ### Exercise 3. Where the stage changes
    """)
    return


@app.cell(hide_code=True)
def choice_projection_cell():
    choice_projection = L.choice(
        "res-projection",
        question="Where a stage of ResNet ends, `x` has shape (8, 14, 14) and `f(x)` (16, 7, 7). On the residual connection, ResNet puts `nn.Conv2d(8, 16, 1, stride=2)`. What shape does that make of `x`?",
        options={
            "wide": "(16, 14, 14): a filter of 1 by 1 keeps the size",
            "right": "(16, 7, 7): the shape of `f(x)`",
            "same": "(8, 7, 7): a filter of 1 by 1 keeps the channels",
        },
        correct="right",
        why={
            "wide": "The size stays with stride 1. With stride 2, the filter is used at every second place: 14 becomes 7.",
            "same": "A convolution gives one channel per filter: 16 filters, 16 channels, also when they are 1 by 1.",
            "right": "16 filters of 8 by 1 by 1 make 16 channels, and stride 2 halves 14 to 7: now `x` and `f(x)` can be added.",
        },
    )
    choice_projection
    return (choice_projection,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It builds two networks of the same depth: a convolution from 1 to 8 channels, {C.BLOCKS} blocks of 8 channels, pooling, a convolution to 16 channels, {C.BLOCKS} blocks of 16 channels, pooling, and the output head: 19 layers with weights. In the one, every block is plain, two convolutions with a ReLU after each; in the other, every block is your `ResBlock`. Both start from the same weights. It trains each with your Trainer: {C.EPOCHS} epochs on the 6,000 training images, a learning rate of {C.number(C.LR, 2, L.lang)} with a warmup of one epoch, one run each, with seed 0, and counts the parameters with your `count_params`. Where a module of yours is missing or broken, the course's stands in. It works once exercise 2 is done, and takes about a minute and a half. While it trains, a chart shows the validation loss after every epoch: first of the plain network, then of yours, with the plain one dashed beside it. Guessing among 10 kinds gives a loss of $\\ln 10 \\approx 2.30$.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train both networks")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_resnet(train_button, result_block):
    result_block  # run again when exercise 2 saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train both networks.*")
    else:
        _now = {}   # the live chart of each network, made at its first report

        def _progress(_p):
            if _p["run"] not in _now:
                _res = _p["run"] == "residual"
                _now[_p["run"]] = L.live({"title": "Your residual network: the validation loss" if _res else "The plain network: the validation loss", "x": "epoch", "y": "validation loss",
                                          "alt": "The validation loss of the network of your residual blocks after every epoch so far; dashed, that of the plain network." if _res else "The validation loss of the plain network after every epoch so far."}, _p["epochs"], low=0.0)
                if _res and "plain" in _now:
                    _now["residual"].reference(list(zip(_now["plain"].xs, _now["plain"].ys)), "plain", role="loss")
            _who = "The plain network" if _p["run"] == "plain" else "Your residual network"
            if "val_loss" in _p:
                _now[_p["run"]].add(_p["epoch"], _p["val_loss"], f"**{_who}** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            else:
                _now[_p["run"]].line(f"**{_who}** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercise 2 first: then your `ResBlock` is in your project, and this cell uses it.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Your project takes too long here: more than four minutes to train one network. Does your `ResBlock` or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "resnet": "*Your `ResBlock` in `work/projects/cnn/resnet.py` stops with an error, while the network is made or trained. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _last = _now.get("residual") or _now.get("plain")   # a run that stops keeps its chart so far
            _out = _last.stopped(_note) if _last else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _pic = [mo.Html(C.curves_picture(_result["curves"], {"title": "Validation loss after every epoch", "x": "epoch", "y": "validation loss", "plain": "plain", "alt": "The validation loss of the network of your residual blocks after every epoch, and dashed that of the plain network."}, L.lang))] if _result.get("curves") else []
            _out = mo.vstack(_pic + [mo.md(f"""Of the 1,000 test images, the plain network reads **{_result['plain']}** right, and the one of your residual blocks **{_result['residual']}**, both with {C.grouped(_result['params_residual'], L.lang)} parameters, counted with your `count_params`. Over 5 seeds, with the course's code: 105 every time plain, and 876 to 895 residual, more than your SmallCNN with every seed. The chart shows the validation loss of both after every epoch, the plain network's dashed.{_who}""")])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The residual connection of [Residual connections](/course/open/nn-residual?lang=en), now around the convolutions of a CNN: where your VGG blocks of [VGG: deeper with small filters](/course/open/cnn-vgg-blocks?lang=en) stopped learning, a network that is deeper still now trains, and reads more test images right than your SmallCNN. Your project `cnn` is complete: `small_cnn.py`, `lenet.py`, `vgg.py` and `resnet.py`, from the CNN of 1998 to the block that is still in almost every deep network. You will meet $x + f(x)$ again in every transformer block.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_block, choice_projection):
    L.wrap_up(choice_shape, result_block, choice_projection)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
