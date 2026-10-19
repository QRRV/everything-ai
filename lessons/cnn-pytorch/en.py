# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cnn-pytorch. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow convolution

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cnn-pytorch", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Convolution as one big matrix multiplication](/course/open/cnn-conv-numpy?lang=en) and [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_im2col():
    quick_im2col = L.choice(
        "quick-im2col",
        question="With im2col, a convolutional layer becomes one matrix multiplication. What does one row of the patch matrix $\\mathbf{P}$ hold?",
        options={
            "channel": "One channel of the whole image, laid out flat",
            "right": "The patch at one place, from every channel",
            "filter": "One filter, with all its weights laid out flat",
        },
        correct="right",
        why={
            "channel": "A row is what one filter looks at in one place: a small patch, but from every channel. See [Convolution as one big matrix multiplication](/course/open/cnn-conv-numpy?lang=en).",
            "filter": "The filters are the rows of $\\mathbf{W}$, the other matrix. See [Convolution as one big matrix multiplication](/course/open/cnn-conv-numpy?lang=en).",
            "right": "And the filters are the rows of $\\mathbf{W}$. Today PyTorch's `nn.Conv2d` does all of this for you, fast.",
        },
    )
    quick_im2col
    return (quick_im2col,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="You train a new network on one mini-batch only, the same 64 images at every step, for a few hundred steps. What happens to its loss if the network and its training step work?",
        options={
            "right": "It goes to nearly 0: the network learns the 64 images by heart",
            "guess": "It stays near 2.30: with 10 kinds, that is the loss to expect",
            "up": "It goes up: training on one mini-batch makes it overfit",
        },
        correct="right",
        why={
            "guess": "2.30 is the loss of guessing among 10 kinds: a network that stays there has learned nothing. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "up": "Overfitting is just what the test wants: on the batch itself, the loss goes down. See [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en).",
            "right": "That is the overfit test, and your `overfit_one_batch` is in your project. Today it tests your first CNN.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        Your deep MLP of [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en) reads about 85% of the clothes of Fashion-MNIST right. But it sees an image as a row of 784 numbers in which neighbors mean nothing: shuffle the pixels of every image in the same way, and it learns just as well. And each of the 64 neurons of its first layer has a weight of its own for every pixel. A CNN makes use of how an image is built: its filters look at pixels together with their neighbors, and the same filter slides over the whole image. In this project you build your first CNN in PyTorch and train it with your own Trainer. It is the first step of your own CNN; the next three are extra lessons of the computer vision branch.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working"],
                           0, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_params():
    guess_params = L.guess(
        "params",
        question="Your deep MLP has 85,194 parameters. Today's CNN has two convolutional layers, with 8 and then 16 filters of 3 by 3, and an output head that gets 784 numbers per image. How many parameters does it have?",
        options={
            "same": "About 85,000: as many as your deep MLP",
            "more": "About 300,000: two layers of many channels",
            "right": "About 9,000: a ninth of your deep MLP",
        },
        reveal="**9,098.** The first convolutional layer has 8 filters of 1 × 3 × 3 and 8 bias terms: 80. The second has 16 filters of 8 × 3 × 3 and 16 bias terms: 1,168. The output head has 784 × 10 + 10 = 7,850, most of them. Your deep MLP has 50,240 in its first layer alone, a weight for every pixel and neuron; a filter has 9 per input channel, and uses them at every place of the image.",
    )
    guess_params
    return (guess_params,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In PyTorch, a convolutional layer is `nn.Conv2d(c_in, c_out, k, padding=p)`: $C_\text{out}$ filters of shape $(C_\text{in}, k, k)$, each with a bias term, as in [Channels and pooling](/course/open/cnn-channels-pooling?lang=en), computed fast, as one big matrix multiplication or with other tricks. It works on a whole mini-batch at once, of shape $(B, C, H, W)$: $B$ images, each with $C$ channels of $H$ by $W$ pixels. The images of Fashion-MNIST are gray, one channel, so a mini-batch of 64 has shape (64, 1, 28, 28). `nn.MaxPool2d(2)` does max pooling of 2 by 2 on every channel of every image.

        Your first CNN, `SmallCNN`, has two blocks of convolution, ReLU and pooling, and then an output head. With padding 1, a convolution of 3 by 3 keeps the height and width, and each pooling halves them: 28, 14, 7. After the second block, every image is 16 channels of 7 by 7, and `x.reshape(len(x), -1)` lays them out as one row of 784 numbers, for an `nn.Linear(784, 10)` that gives the 10 logits, one for every kind of clothing.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | `nn.Conv2d` | n n conv 2 d | a convolutional layer: $C_\text{out}$ filters of shape $(C_\text{in}, k, k)$, each with a bias term | `nn.Conv2d(1, 8, 3, padding=1)` |
        | `nn.MaxPool2d` | n n max pool 2 d | max pooling of every channel, in blocks of $k$ by $k$ | `nn.MaxPool2d(2)` |
        | $(B, C, H, W)$ | B by C by H by W | the shape of a mini-batch of images: images, channels, height, width | `X.reshape(len(X), 1, 28, 28)` |
        | `numel` | numel | the number of numbers in a tensor, whatever its shape | `p.numel()` |
        """),
        L.watch_out("The output head must fit what comes out of the last block: 16 channels of 7 by 7 is 784 numbers. Change the size of the images, the padding or a layer, and that number changes too; PyTorch then stops with an error that the shapes cannot be multiplied. And `nn.Conv2d` wants the channels before the height and width: the photos of [An image is a grid of numbers](/course/open/data-images-as-arrays?lang=en) had them last."),
        mo.md(r"""
        The picture: one image on its way through `SmallCNN`, with its shape after every step. Each stack has a square for every channel, at most 6 drawn, as big as its height and width. In a mini-batch, every shape gets $B$ in front.
        """),
        C.shapes_picture({"title": "One image through SmallCNN", "input": "one gray image", "maps": "{0} feature maps of {1} by {1}",
                          "logits": "10 logits, one per kind", "conv": "conv 3×3 to {1} channels, ReLU", "pool": "max pooling 2 by 2",
                          "head": "one row of 784, then nn.Linear",
                          "alt": "A gray image of 1 by 28 by 28 becomes 8 feature maps of 28 by 28 after a convolution of 3 by 3 with ReLU, 8 of 14 by 14 after max pooling, 16 of 14 by 14 after a second convolution, 16 of 7 by 7 after pooling, and 10 logits after one row of 784 numbers goes through nn.Linear."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A small CNN for images of 28 by 28 with one channel:

    1. The images as a mini-batch of shape $(B, 1, 28, 28)$: a row of 784 pixels becomes an image with `.reshape(len(X), 1, 28, 28)`.
    2. Two blocks, each a convolution of 3 by 3 with padding 1, ReLU, and max pooling of 2 by 2: $(B, 8, 14, 14)$, then $(B, 16, 7, 7)$.
    3. One row per image: `x.reshape(len(x), -1)`, 16 × 7 × 7 = 784 numbers.
    4. The output head, a linear layer from 784 to 10: the logits. Train it like any network, with your Trainer.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One block, step 2 of the recipe, on the first 2 training images. The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_block():
    import torch
    from torch import nn

    _X, _y, _, _ = C.fashion()
    # 1. two images, as a mini-batch
    _x = _X[:2]
    print(_x.shape)
    # 2. one block: convolution, ReLU, pooling
    torch.manual_seed(0)
    _conv = nn.Conv2d(1, 8, 3, padding=1)
    _pool = nn.MaxPool2d(2)
    print(_conv.weight.shape, _conv.bias.shape)
    _h = _pool(torch.relu(_conv(_x)))
    print(_h.shape)
    # 3. one row per image
    _row = _h  # complete: _h.reshape(len(_h), -1)
    print(_row.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The two images have shape (2, 1, 28, 28). The filters of `_conv` have shape (8, 1, 3, 3), $(C_\text{out}, C_\text{in}, k, k)$ as in [Channels and pooling](/course/open/cnn-channels-pooling?lang=en), and there are 8 bias terms. One block turns every image into 8 feature maps of 14 by 14. As it is, `_row` still has that shape, (2, 8, 14, 14). Completed, it has shape (2, 1568): one row of 8 × 14 × 14 = 1,568 numbers per image. `SmallCNN` has a second block first, so its rows have 16 × 7 × 7 = 784.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The shape after a block
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape-after",
        question="A mini-batch of shape (64, 1, 28, 28) goes through `nn.Conv2d(1, 8, 5)`, without padding, then through `nn.MaxPool2d(2)`. What shape comes out?",
        options={
            "padded": "(64, 8, 14, 14): 28 by 28 before the pooling",
            "right": "(64, 8, 12, 12): 24 by 24 before the pooling",
            "one": "(64, 1, 12, 12): one channel, as the input has",
        },
        correct="right",
        why={
            "padded": "Without padding, a filter of 5 by 5 fits at $28 - 5 + 1 = 24$ places per side, as in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en).",
            "one": "The layer has 8 filters, so 8 output channels come out, whatever came in.",
            "right": "The convolution gives (64, 8, 24, 24), and the pooling halves the height and width. The batch axis stays as it is: 64 images in, 64 out.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_small_cnn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your first CNN

        Complete `SmallCNN`: in `__init__` its layers, and in `forward` steps 2 to 4 of the recipe, for a mini-batch `x` of shape $(B, 1, 28, 28)$; it gives the logits, of shape $(B, 10)$. Make the layers in the order in which the data passes them: the convolution from 1 channel, the one from 8, then the output head. The check makes the network after `torch.manual_seed(0)` and compares its logits for 2 and for 3 random images. When it is right, the check saves it in your project, as `work/projects/cnn/small_cnn.py`.
        """),
        L.task("Make `SmallCNN` give 10 logits per image, from two blocks of convolution, ReLU and pooling."),
    ])
    return


@app.cell(hide_code=True)
def exercise_small_cnn():
    box_small_cnn = L.code_box("small-cnn", lines=18)
    box_small_cnn
    return (box_small_cnn,)


@app.cell(hide_code=True)
def check_small_cnn(box_small_cnn):
    result_small_cnn = L.check("small-cnn", box_small_cnn.value)
    result_small_cnn
    return (result_small_cnn,)


@app.cell(hide_code=True)
def hints_small_cnn():
    L.hints(
        "small-cnn",
        [
            "In `__init__`, four attributes of `self`: two `nn.Conv2d`, one `nn.MaxPool2d` that both blocks can share, as it has no weights, and an `nn.Linear` for 16 × 7 × 7 inputs.",
            "In `forward`, each block is the line of step 2 of the worked example, with your layers; after the two blocks, the line of step 3, and then the output head.",
        ],
        why="Pooling learns nothing, so one `nn.MaxPool2d` can serve both blocks. Only the output head sees all of an image at once: every number before it depends on a piece of at most 10 by 10 pixels, its receptive field.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Count what a model learns

        Write `count_params(model)`: it gives how many numbers all parameters of `model` hold together, the weights and the bias terms. Go over `model.parameters()`, as in [AdamW: weight decay done right](/course/open/opt-adamw?lang=en), and add up `p.numel()` for each: the number of numbers in a tensor. The check counts `nn.Linear(3, 2)`, which should give 8, and other layers. When it is right, the check saves it as `work/projects/cnn/count.py`.
        """),
        L.task("Make `count_params` add up the numbers in all parameters of a model."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-params", lines=7)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-params", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-params",
        [
            "Start with a total of 0, and add to it in a `for` loop over `model.parameters()`.",
            "`p.numel()` is the number of numbers in `p`: 6 for a weight matrix of shape (2, 3).",
        ],
        why="A parameter is one tensor, such as the weight matrix or the bias terms of a layer, and `numel` counts all numbers in it, whatever its shape. Pooling has no parameters, so the loop does not run and the total stays 0.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The cell below uses your project. It runs your overfit test of [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en) on your `SmallCNN`: {C.OVERFIT_STEPS} steps at a learning rate of {C.number(C.OVERFIT_LR, 1, L.lang)} on the first 64 images. Then it trains your `SmallCNN`, and your `DeepMLP` with He initialization, batch normalization and residual connections, with your Trainer: {C.EPOCHS} epochs on the 6,000 training images, a learning rate of {C.number(C.LR, 2, L.lang)} with a warmup of one epoch, the same for both, one run each, with seed 0. Where a module of yours is missing or broken, the course's stands in. It works once exercises 2 and 3 are done, and takes about 10 seconds.
    """)
    return


@app.cell(hide_code=True)
def demo_cnn(result_small_cnn, result_count):
    result_small_cnn, result_count  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `SmallCNN` and your `count_params` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your project takes too long here: more than a minute for the overfit test, or more than two minutes to train one network. Does your `SmallCNN`, your overfit test or your Trainer do more work than it needs, such as a loop that runs too often?*",
            "cnn": "*`SmallCNN` or `count_params` in `work/projects/cnn/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
            "overfit": "*Your `overfit_one_batch` in `work/projects/nn/overfit.py` stops with an error on the CNN. Did you change that file? Delete it, and check exercise 2 of [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en) again.*",
            "trainer": "*Training with the Trainer in `work/projects/nn/trainer.py`, or making the `DeepMLP` of `work/projects/nn/deep_mlp.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
        }[_result])
    else:
        _learned = "it can learn" if _result["loss"] < 0.1 else "it does not learn the 64 images by heart, so look at your `SmallCNN` first"
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
        _out = mo.md(f"""Your overfit test: after {C.OVERFIT_STEPS} steps on 64 images, the loss of your CNN is {C.number(_result['loss'], 3, L.lang)}: {_learned}.

Of the 1,000 test images, your CNN reads **{_result['cnn']}** right, with {C.grouped(_result['params_cnn'], L.lang)} parameters, counted with your `count_params`; your deep MLP reads **{_result['mlp']}** right, with {C.grouped(_result['params_mlp'], L.lang)}. Over 5 seeds, with the course's code, the CNN reads 864 to 879 right and the deep MLP 845 to 863: better, with a ninth of the parameters.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The filters, channels and pooling of [Channels and pooling](/course/open/cnn-channels-pooling?lang=en), now as `nn.Conv2d` and `nn.MaxPool2d`, trained by your Trainer of [A reusable training loop](/course/open/nn-trainer?lang=en) and tested with the overfit test of [Project: make a deep network train](/course/open/nn-deep-net-project?lang=en). Looking at neighbors and sharing a filter over the whole image, from [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en), are why it needs so few parameters. Your project `cnn` now has `small_cnn.py` and `count.py`. The computer vision branch continues this project in three extra lessons, with LeNet-5, VGG and ResNet. The main route goes on with *Attention weights from dot products*: how a model works out, with dot products and softmax, which words to look at.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_small_cnn, result_count):
    L.wrap_up(choice_shape, result_small_cnn, result_count)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
