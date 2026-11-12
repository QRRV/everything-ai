# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-vit. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-vit", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [The transformer block](/course/open/attn-transformer-block?lang=en) and [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="A text of 5 tokens goes through your `CausalSelfAttention`. Which tokens does the third token take from?",
        options={
            "all": "All 5: attention compares every token with every token",
            "right": "The first three: itself and the two before it",
            "self": "Only itself",
            "after": "Itself and the two after it",
        },
        correct="right",
        why={
            "all": "Without the mask it would. The causal mask sets the scores of every later token to minus infinity before the softmax, so a token takes only from itself and the tokens before it. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "self": "Then there would be no attention at all. The mask hides only the tokens after it: the third token takes from the first three. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "after": "The other way around: the mask hides the future, the tokens after it, because a model that writes text may not see what comes next. See [The causal mask](/course/open/attn-causal-mask?lang=en).",
            "right": "A text has a before and an after, and a model that writes may not look ahead. A photo has neither: today the mask goes, and every patch takes from every patch.",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def quick_check_changes():
    quick_changes = L.choice(
        "quick-changes",
        question="With mirror, crop and color, your ColorCNN read 583 to 619 of the 1,000 test photos right after 40 epochs; without changes, 460 to 545. What did the changes do?",
        options={
            "more": "They added photos of new kinds",
            "right": "They made it harder to learn the training photos by heart, so it learned more about the kinds",
            "easier": "They made the photos easier to read, with more contrast",
        },
        correct="right",
        why={
            "more": "The kinds stay the same 10, and the photos the same 6,000: every epoch shows each of them a little differently. See [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en).",
            "easier": "The color jitter makes some photos duller too, and the network is tested on the photos as they are. What changed is that it could not learn its training photos by heart as easily. See [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en).",
            "right": "Every epoch, each photo comes in another version, so details of one version do not help with the next. Today the same changes help a network that has much more to learn from its photos.",
        },
    )
    quick_changes
    return (quick_changes,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A convolution looks at a few neighboring pixels at a time, with the same filter at every place of the photo: an ear is the same pattern in the top left corner as in the middle. Every CNN has those two ideas before it sees a single photo. Attention has neither: every token can take from every token, and a transformer knows where a token stands only from the position added to it. In 2020, Alexey Dosovitskiy and others at Google cut photos into squares of 16 by 16 pixels and gave them as tokens to a transformer, without a single convolution: a *Vision Transformer*, or ViT. Trained on the 1.3 million photos of ImageNet, without strong measures against overfitting, it ended a few percentage points below ResNets of about its size. Pretrained on 300 million photos, a set of Google's own, it did as well as the best CNNs of that time, at less cost to train. Today you build one from your transformer block, and train it on the 6,000 photos of your ColorCNN.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_vit():
    guess_vit = L.guess(
        "vit-or-cnn",
        question="Your ViT, with 11,738 parameters, and your ColorCNN, with 11,642, both learn for 40 epochs from the same 6,000 photos, mirrored, cropped and changed in color. Which of the two reads more of the 1,000 test photos right?",
        options={
            "right": "The ColorCNN, clearly",
            "same": "About as many",
            "vit": "The ViT, clearly",
        },
        reveal="**The ColorCNN.** Over 5 seeds, with the course's code, the ColorCNN reads 583 to 619 test photos right, and the ViT 489 to 524. From 6,000 photos, the ViT has to learn what the ColorCNN has from the start, and it does not get that far.",
    )
    guess_vit
    return (guess_vit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A photo as a row of tokens.** A transformer reads a row of tokens. A ViT makes a row of a photo: it cuts the photo into squares of p by p pixels, its **patches**, and takes them row by row, as you read a page. A photo of 32 by 32 with p = 4 gives 8 rows of 8 patches: 64 patches, each of 3 · 4 · 4 = 48 numbers, its red, its green and its blue. You met the word patch in [Convolution as one big matrix multiplication](/course/open/cnn-conv-numpy?lang=en): there, a patch was the piece of the image under a filter; here, the patches lie side by side, without overlap.

        **Patch embedding.** A text model looks each token up in a table. A ViT sends the 48 numbers of each patch through one linear layer, the same for every patch, to d numbers: the **patch embedding**. One linear layer on every patch is the same d filters at every place: a convolution with d output channels, a kernel size of 4 and a stride of 4, as in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), does exactly this, and much code for ViTs writes it that way.

        **Positions.** Attention does not know where a patch came from: shuffle the 64 patches, and every token gets the same numbers as before, only in another order. So, as in [Telling attention where things are](/course/open/attn-positions?lang=en), every place gets a learned position, a row of an `nn.Embedding`, added to its patch. Nothing tells the ViT that patch 9 lies next to patches 10 and 17: it learns that from the photos, or not at all.

        **The class token.** Which token says what the photo shows? A ViT puts one extra token in front of the 64 patches: the **class token**, a parameter of d numbers, the same for every photo, that the network learns. It is not part of the photo, so all it knows is what it takes from the patches. After the last block, its d numbers go through a layer normalization and a linear layer to the 10 logits.

        **No mask.** In a model that writes text, a token takes only from the tokens before it: it may not see what comes next. A photo has no before and after: every patch takes from every patch, and the class token, in front, from all 64. So the block of a ViT is your block of [The transformer block](/course/open/attn-transformer-block?lang=en), with the mask taken out of its attention.

        **What a convolution builds in.** A convolution sees only a few neighboring pixels, and uses the same filter everywhere. Those are assumptions about photos that hold before the network learns anything: an **inductive bias**. A ViT has little of it: that the pixels of one patch belong together, and that one layer reads every patch. That patch 9 lies next to patch 10, and that an ear in a corner is the same as one in the middle, it has to learn from its photos.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Vision Transformer** | | a transformer for photos: the patches of a photo are its tokens, with a class token in front; short: ViT | `ViT(d, h, depth, p)` |
        | **patch embedding** | | the d numbers a linear layer makes of the numbers of one patch: the token of that patch | `nn.Linear(3 * p * p, d)` |
        | **class token** | | an extra token in front of the patches, learned, from which the network reads what the photo shows | `nn.Parameter(torch.zeros(1, 1, d))` |
        | **inductive bias** | | what a network assumes about its data before it learns anything, such as that neighboring pixels belong together | |
        | `t.expand(n, 1, -1)` | expand | the same tensor n times along an axis of size 1, without copying; -1 keeps an axis as it is | `self.cls.expand(len(x), 1, -1)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A Vision Transformer for photos of 32 by 32 with patches of p by p:

    1. Cut each axis of the photos into two with `reshape`: the rows into `h // p` rows of patches of `p` rows each, the columns likewise: shape `(n, c, h // p, p, w // p, p)`.
    2. Put the two axes that say which patch first, with `permute(0, 2, 4, 1, 3, 5)`, then the color and the rows and columns inside the patch.
    3. `reshape` to `(n, (h // p) * (w // p), c * p * p)`: one row of numbers per patch.
    4. Send every patch through the patch embedding, put the class token in front of the tokens of every photo, and add the learned positions of all of them.
    5. Run the tokens through the blocks, each with attention without a mask.
    6. Read the class token after the last block: a layer normalization, then a linear layer to the logits.
    7. Train it as a CNN: the same cross-entropy, the same Trainer, the same changed photos.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell loads CIFAR-10 and takes test photo 82, the car, as a batch of one photo. It cuts out patch (1, 2) with slices, makes one row of numbers of it, and prints both shapes, then the first 4 numbers of that row next to the first row of red of the patch. Then it sends the row through a patch embedding of 16 numbers, and prints its shape. Run the cell as it is, then change `p` where the comment says, and run it again. `cifar` is the course's `everything_ai.cifar`, which the cell below gives once CIFAR-10 is on your computer.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download CIFAR-10, 170 MB")
    return (get_button,)


@app.cell(hide_code=True)
def cifar_cell(get_button):
    cifar = C.cifar_or_stop(get_button, {
        "ask": "*CIFAR-10 is not on your computer yet. If you did [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once from the University of Toronto, 170 MB, unpacks its photos into one file of 184 MB, and then deletes the download. Its page states no license, so the course does not ship it; where its photos come from is in that project lesson. The worked example, the picture of the patches and the demo need it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
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
def example_patch(cifar):
    import torch
    from torch import nn

    X_train, y_train, X_test, y_test = cifar.load()
    x = torch.tensor(X_test[82:83], dtype=torch.float32) / 255   # one photo, the car, as a batch of 1
    p = 4  # change: p = 8
    print(x.shape, (32 // p) ** 2)
    # patch (1, 2): row 1 of the patches, column 2, counted from 0
    patch = x[:, :, 1 * p:2 * p, 2 * p:3 * p]
    flat = patch.reshape(1, 3 * p * p)
    print(patch.shape, flat.shape)
    print(flat[0, :4], patch[0, 0, 0])
    # the patch embedding: one linear layer, the same for every patch
    torch.manual_seed(0)
    embed = nn.Linear(3 * p * p, 16)
    print(embed(flat).shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With p = 4, the photo has 64 patches. Patch (1, 2) is rows 4 to 7 and columns 8 to 11 of the car: shape `(1, 3, 4, 4)`, and as one row `(1, 48)`. The first 4 numbers of that row, 0.2588, 0.2510, 0.2510 and 0.2510, are the first row of its red: `reshape` reads the patch as it sits, the red row by row, then the green, then the blue. The patch embedding makes 16 numbers of the 48: shape `(1, 16)`, the token of this patch. With p = 8, the photo has 16 patches, patch (1, 2) is rows 8 to 15 and columns 16 to 23, and one row holds 3 · 8 · 8 = 192 numbers; the embedding still makes 16 of them. Fewer, bigger patches make fewer tokens, so attention has less to compare, but each token sums up more of the photo. The paper took 16 by 16 for photos of 224 by 224: 196 patches.
    """)
    return


@app.cell(hide_code=True)
def patches_picture_cell(cifar):
    cifar  # once CIFAR-10 is on your computer
    mo.vstack([mo.md(r"""
    The picture: the car of the worked example, with lines between its 64 patches of 4 by 4, and patch (1, 2), row 1 and column 2, counted from 0, framed. Next to it that patch, enlarged, and under it its 48 numbers as one row: its 16 reds, then its 16 greens, then its 16 blues, each row by row, every number in its own color, as strong as it is. That row goes through the patch embedding.
    """), mo.Html(C.patches_picture({"whole": "the car in 64 patches", "patch": "patch (1, 2)", "numbers": "its 48 numbers: red, green, blue",
                               "alt": "The car cut into 8 by 8 patches of 4 by 4 pixels, with patch (1, 2) framed; next to it that patch enlarged, and under it its 48 numbers as a strip: first the reds, then the greens, then the blues."}))])
    return


@app.cell(hide_code=True)
def text_exercise_patches():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Cut into patches

        Complete `patches(x, p)`: give back the patches of every photo of `x`, of shape `(N, 3, H, W)`, as shape `(N, (H // p) * (W // p), 3 * p * p)`. The patches go row by row, as in the worked example, and every patch holds its red, its green and its blue, each row by row, as `reshape` read patch (1, 2) there. Steps 1 to 3 of the recipe do it for all patches at once. The check gives your function made-up photos, also one that is not square, and under the check it puts the patches of a made-up photo back in their places, with a black line between them. When it is right, it saves your function in your project, as `work/projects/cnn/patches.py`.
        """),
        L.task("Make `patches` cut every photo into its patches, each a row of numbers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_patches():
    box_patches = L.code_box("patches", lines=8)
    box_patches
    return (box_patches,)


@app.cell(hide_code=True)
def check_patches(box_patches):
    result_patches = L.check("patches", box_patches.value)
    result_patches
    return (result_patches,)


@app.cell(hide_code=True)
def hints_patches():
    L.hints(
        "patches",
        [
            "A row of 32 pixels is 8 pieces of 4. `reshape` can say that: an axis of `h` becomes two axes, `h // p` and `p`. Do that for the rows and for the columns.",
            "After `x.reshape(n, c, h // p, p, w // p, p)`, the axes say: photo, color, row of patches, row inside the patch, column of patches, column inside the patch. Which two axes say which patch it is? Put those right after the photo with `permute`.",
            "`x = x.permute(0, 2, 4, 1, 3, 5)` gives shape `(n, h // p, w // p, c, p, p)`. Then one `reshape` makes the patches one axis and the numbers of a patch one axis: `x.reshape(n, ___, c * p * p)`.",
        ],
        why="`permute` only changes the order in which the axes are read, so the last `reshape` copies the numbers into their new order. torchvision's ViT does this step as a convolution with a kernel size and a stride of p, followed by a reshape and a permute: the same patches, and the patch embedding in the same step.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_attention():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Attention without a mask

        Complete `SelfAttention`. It builds on your `CausalSelfAttention` of [The transformer block](/course/open/attn-transformer-block?lang=en), from `work/projects/llm/attention.py`: `class SelfAttention(CausalSelfAttention)` keeps its `__init__`, with its four layers `query`, `key`, `value` and `out`, and gets a `forward` of its own, as in [Building on another class](/course/open/py-inheritance?lang=en). That forward does what yours did, for a mini-batch of shape `(B, n, d)`, without the mask: every patch takes from every patch of its own photo. The check gives it made-up patches, and also looks whether the first patch takes from the last one. When it is right, it saves your class in your project, as `work/projects/cnn/self_attention.py`.
        """),
        L.task("Make `SelfAttention` let every patch take from every patch."),
    ])
    return


@app.cell(hide_code=True)
def exercise_attention():
    box_attention = L.code_box("self-attention", lines=14)
    box_attention
    return (box_attention,)


@app.cell(hide_code=True)
def check_attention(box_attention):
    result_attention = L.check("self-attention", box_attention.value)
    result_attention
    return (result_attention,)


@app.cell(hide_code=True)
def hints_attention():
    L.hints(
        "self-attention",
        [
            "Open `work/projects/llm/attention.py`: your `forward` there is almost this one. What made it causal?",
            "Per head: the scores `q[:, :, a:b] @ k[:, :, a:b].transpose(1, 2)`, divided by the square root of `self.d_head`, then the softmax along axis 2, times `v[:, :, a:b]`. No `future`, no `masked_fill`.",
            "In the loop: `S = ___ / math.sqrt(self.d_head)` and `outputs.append(torch.softmax(S, dim=2) @ v[:, :, a:b])`; after the loop: `return self.out(torch.cat(outputs, dim=___))`.",
        ],
        why="An encoder, a transformer that reads a whole input at once, has no mask; a decoder, which writes, has one. The ViT is an encoder, as is BERT, a language model of 2018 that reads a whole text at once; the ViT's class token comes from BERT.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vit():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The Vision Transformer

        Complete `ViT(d, h, depth, p)`. In `__init__`: `patch`, the patch embedding; `cls`, the class token, a parameter of shape `(1, 1, d)` that starts as zeros; and `pos`, an `nn.Embedding` with a learned position for each of the `n` patches and the class token. Make them in that order: then they get the same random starting values as in the course's solution. The blocks are your `Block` of [The transformer block](/course/open/attn-transformer-block?lang=en), from `work/projects/llm/block.py`, each with your `SelfAttention` in place of its `CausalSelfAttention`; those lines are given, and replace the `attention` of each block, as your `Block` names it. In `forward`: the patches through the patch embedding, the class token in front of the tokens of every photo, and the positions of all `n + 1` tokens added; the blocks are given; then the class token after the last block through `self.norm` and `self.out`. The check makes ViTs of several sizes, counts the parameters of the demo's ViT, and looks whether a change in the bottom right corner of a photo reaches the class token. Under the check, your network: a box per layer, with what comes out of it. When it is right, it saves your class in your project, as `work/projects/cnn/vit.py`.
        """),
        L.task("Make `ViT` give 10 logits per photo from the class token."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vit():
    box_vit = L.code_box("vit", lines=28)
    box_vit
    return (box_vit,)


@app.cell(hide_code=True)
def check_vit(box_vit, result_patches, result_attention):
    result_patches, result_attention  # check again when exercise 1 or 2 saves your module
    result_vit = L.check("vit", box_vit.value)
    result_vit
    return (result_vit,)


@app.cell(hide_code=True)
def hints_vit():
    L.hints(
        "vit",
        [
            "Three layers in `__init__`: `nn.Linear` from the numbers of a patch to d, `nn.Parameter` around a tensor of zeros, and `nn.Embedding` with a row for every token, the class token too.",
            "In `forward`, the tokens: `self.patch(patches(x, self.p))` has shape `(N, n, d)`. The class token has shape `(1, 1, d)`: `self.cls.expand(len(x), 1, -1)` gives every photo one, and `torch.cat` along axis 1 puts it in front. Then add `self.pos(torch.arange(x.shape[1]))`.",
            "At the end: the first token of every photo, `x[:, 0]`, through `self.norm`, then `self.out`: `return self.out(self.norm(___))`.",
        ],
        why="The paper's ViT reads the class token as BERT does; the paper also tried the mean of all patch tokens after the last block, which worked as well once its learning rate was tuned (appendix D.3). A position is learned for every place, also in the paper, where positions that knew about rows and columns did no better (appendix D.4).",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    The button uses your project. It trains your `ViT({C.D}, {C.H}, {C.DEPTH}, {C.P})`, with {C.grouped(C.PARAMS, L.lang)} parameters, on the first 6,000 training photos of CIFAR-10, from the same Dataset as your ColorCNN in [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en): half of them mirrored, every one cut out with your `crop` and changed in color with your `jitter`, anew every time. It trains for {C.EPOCHS} epochs with your Trainer, with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0, and tests on the first 1,000 test photos, as they are. Where a module of yours is missing or broken, the course's stands in. It needs CIFAR-10, from the worked example, and exercises 1 to 3, and takes about two minutes. While it trains, a chart shows the loss on the changed training photos every 10 batches, and dashed on the test photos after every epoch. Beside or under it are your ViT's guesses for one test photo of each kind, and under them how alike the position of patch (3, 3) is to that of every patch, as the cosine similarity of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en): the stronger the orange, the more alike; the patch itself is framed. In the picture after, the gray line is the validation loss of the ColorCNN trained on the same changed photos, the course's run with seed 0. Then you can rewind through your run, from before the first epoch.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your ViT")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_vit(train_button, result_patches, result_attention, result_vit):
    result_patches, result_attention, result_vit  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your ViT.*")
    else:
        _now = {}   # the live chart, made at the first report

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your ViT on changed photos", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your ViT so far: on the changed training photos every 10 batches, and dashed on the 1,000 test photos after every epoch."}, _p["epochs"], low=0.0, window=5,
                                       key=[("loss", False, "changed training photos, every 10 batches"), ("loss", True, "test photos, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), L.lang))
                _now["test"] = []
            if "val_loss" in _p:   # the test photos after every epoch, dashed
                _now["test"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["test"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your ViT** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training photos every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your ViT** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Your ViT** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 1 to 3 first: then your `patches`, your `SelfAttention` and your `ViT` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "data": "*CIFAR-10 is not on your computer yet: download it with the button in the worked example, and press this button again.*",
                "slow": "*Your project takes too long here: more than ten minutes to train. Does your `patches`, your attention or your Trainer do more work than it needs, such as a loop that runs too often?*",
                "vit": "*Your `ViT` in `work/projects/cnn/vit.py`, or a module it uses, stops with an error, or does not give 10 logits per photo. Did you change one of those files? Delete it, and check its exercise again: its check then saves your code again.*",
                "changes": "*Your `crop` or your `jitter` in `work/projects/cnn/` stops with an error, or does not give back a photo of shape (3, 32, 32). Did you change one of those files? Delete it, and check its exercise in [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en) again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, counting with your `count_params` in `work/projects/cnn/count.py`, or your `channel_stats` in `work/projects/cnn/channels.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Why this matters* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation", "cnn": "ColorCNN, validation",
                       "alt": "The loss of your ViT after every epoch: on the changed training photos solid, on the test photos dashed, with a dot where the validation loss is lowest; in gray, the validation loss of the course's ColorCNN trained on the same changed photos."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) + 1 else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e], _result["truth"]))
                if _e == 0:
                    _said = f"Before the first epoch: {_right} of these {len(C.SHOWN)} photos read right, and the positions still random."
                else:
                    _said = f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these {len(C.SHOWN)} photos read right."
                return C.rewind_frame(_result, _e, _labels, L.lang), _said

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to before the first epoch, then forward again: from which epoch do the neighbors of patch (3, 3) stand out?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _best = _result["best"]
            _turn = "" if _best is None else (f" Its validation loss was lowest after epoch **{_best}**." if _best < C.EPOCHS else f" Its validation loss was still falling after epoch {_best}.")
            _near = "" if _result["near"] is None else f" The positions of neighboring patches now have a cosine similarity of {C.number(_result['near'], 2, L.lang)} on average, those 5 or more patches apart {C.number(_result['far'], 2, L.lang)}."
            _out = mo.vstack(_pic + [
                mo.md(f"""Your ViT reads **{C.grouped(_result['train'], L.lang)}** of its 6,000 training photos right, as they are, {C.percent(_result['train'], C.N_TRAIN)}%, and **{_result['test']}** of the 1,000 test photos, {C.percent(_result['test'], C.N_TEST)}%, with {C.grouped(_result['params'], L.lang)} parameters.{_turn}{_near} Over 5 seeds, with the course's code: 3,243 to 3,458 training photos and 489 to 524 test photos; the ColorCNN on the same changed photos 3,917 to 4,024 and 583 to 619.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        What did the ViT learn about places? Before training, the positions are random: the position of a patch is as alike to those of its neighbors as to those far away. After 40 epochs on changed photos, with the course's code, over 5 seeds, the position of a patch has a cosine similarity of 0.42 to 0.54 with those of its 4 neighbors on average, and -0.20 to -0.10 with those 5 or more patches away. The ViT learned from its photos which patches lie next to each other: no one told it. Without changes, it learned less of it: 0.22 to 0.33 with its neighbors. The picture shows it for the course's run with seed 0: for every patch, at its own place, a small map of how alike its position is to those of all 64 patches, as Figure 7 of the paper shows it for a ViT trained on 300 million photos.
        """),
        mo.Html(C.positions_picture(C.POSITIONS, {"title": "How alike the learned positions are, after 40 epochs",
                                                  "under": "each small map: one patch, framed, at its own place",
                                                  "alt": "An 8 by 8 grid of small maps, one per patch, each framed at its own place. In every map, the color is strongest around the framed patch and fades farther away: its learned position is most like those of the patches near it."})),
        mo.md(r"""
        And how many photos does it read right? In the picture, the ColorCNN and the ViT after 40 epochs, without changes and with them, a dot for each of 5 seeds: filled for the training photos as they are, open for the test photos. Without changes, the ViT reads 442 to 465 test photos right, against 460 to 545 for the ColorCNN; with them, 489 to 524, against 583 to 619. The changes help the ViT by about 51 test photos on average, but the ColorCNN stays ahead.
        """),
        mo.Html(C.gap_picture({"title": "Read right after 40 epochs", "x": "read right (%)",
                               "cnn_none": "ColorCNN, no changes", "cnn_all": "ColorCNN, mirror, crop and color",
                               "vit_none": "ViT, no changes", "vit_all": "ViT, mirror, crop and color",
                               "train": "training", "test": "test", "lang": L.lang,
                               "alt": "Four rows of dots between 40 and 100 percent read right. ColorCNN, no changes: training about 100%, test about 50%. ColorCNN with mirror, crop and color: training about 66%, test about 60%. ViT, no changes: training about 78%, test about 45%. ViT with mirror, crop and color: training about 56%, test about 50%."})),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_places():
    mo.md(r"""
    ### Exercise 4. What the positions say
    """)
    return


@app.cell(hide_code=True)
def choice_places_cell():
    choice_places = L.choice(
        "learned-places",
        question="Before training, the position of a patch is as alike to those of its neighbors as to those far away. After 40 epochs, it is most like those of its neighbors: 0.42 to 0.54 on average, against -0.20 to -0.10 for patches 5 or more apart. What does that say?",
        options={
            "given": "The course gave the ViT the places of its patches in a grid of 8 by 8",
            "pixels": "Neighboring patches hold some of the same pixels, so their positions grow alike",
            "right": "The ViT learned from the photos which patches lie next to each other: what a convolution has from the start",
        },
        correct="right",
        why={
            "given": "The positions start as random rows of an `nn.Embedding`, and nothing in the ViT says that patch 9 lies next to patch 10: every patch is just token 9 or token 10. That the positions of neighbors grow alike comes from the photos alone.",
            "pixels": "The patches lie side by side, without overlap: no pixel is in two patches. And the positions do not even see the pixels: they are learned for the places, the same in every photo.",
            "right": "Nothing in the ViT says which patches are neighbors: that their positions grew alike came from the photos alone. A convolution needs no such lesson: its filter sees only neighbors. The ViT spends some of its photos on learning it.",
        },
    )
    choice_places
    return (choice_places,)


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ### Exercise 5. Why the ViT needs more photos

    Answer in your own words, then compare with the model answer.
    """)
    return


@app.cell(hide_code=True)
def explain_bias_cell():
    explain_bias = L.explain(
        "inductive-bias",
        question="On 6,000 photos, your ViT reads fewer test photos right than your ColorCNN of the same size. Pretrained on 300 million photos, the ViT of the paper did as well as the best CNNs. Explain, with what a convolution builds in, why a ViT needs so many more photos.",
        model_answer="**A convolution builds in two assumptions about photos**: a filter looks only at neighboring pixels, and the same filter is used at every place, so a pattern learned in one corner works everywhere. That is its inductive bias, and it holds for photos before the network has seen one. **A ViT has little of it**: its attention lets every patch take from every patch, and its positions start at random. That neighbors belong together, and that an ear is an ear wherever it is, it has to learn from the photos themselves, as its positions show. **With few photos**, that costs it photos the CNN can spend on the kinds, so it ends lower. **With very many photos**, the ViT learns those things, and more that a fixed filter cannot, such as how a patch relates to one far away, so it catches up and can do better.",
        points=[
            "you name what a convolution builds in: it looks only at neighboring pixels, and uses the same filter at every place",
            "you say that a ViT does not have that: every patch can take from every patch, and its positions start at random",
            "you say that the ViT has to learn those things from its photos, which takes many photos, and that with enough of them it catches up",
        ],
    )
    explain_bias
    return (explain_bias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Your block of [The transformer block](/course/open/attn-transformer-block?lang=en), with your attention without its mask, now reads photos: cut into patches by your `patches`, each patch a token through a patch embedding, with learned positions as in [Telling attention where things are](/course/open/attn-positions?lang=en), and a class token in front. Your ViT trained with your Trainer on the changed photos of [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en), and ended below your ColorCNN of the same size: a convolution builds in that neighbors belong together and that a pattern means the same everywhere, and the ViT has to learn that from its photos, as its positions showed. Your project `cnn` now has `patches.py`, `self_attention.py` and `vit.py`. In the next lesson, *Where a Vision Transformer looks*, you look at what the class token takes from which patches, in a ViT that learned from far more photos.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_patches, result_attention, result_vit, choice_places, explain_bias):
    L.wrap_up(result_patches, result_attention, result_vit, choice_places, explain_bias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
