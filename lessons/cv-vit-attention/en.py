# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-vit-attention. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-vit-attention", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [A Vision Transformer from scratch](/course/open/cv-vit?lang=en) and about [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_row():
    quick_row = L.choice(
        "quick-row",
        question="In your ViT, head i of a block computes `A = torch.softmax(S, dim=2)`: for photos of 16 patches with the class token in front, a tensor of shape `(n, 17, 17)`. What does `A[0, 0]` hold, row 0 of the first photo?",
        options={
            "right": "How the class token divides its attention over the 17 tokens, itself included: 17 numbers that add up to 1",
            "column": "How much each of the 17 tokens takes from the class token",
            "features": "The 17 numbers the class token holds after the block",
        },
        correct="right",
        why={
            "right": "Row i of the weights belongs to token i, and the softmax over axis 2 makes every row add up to 1: a share for each token it takes from. Token 0 is the class token, so row 0 is where it looks.",
            "column": "That is column 0, `A[0, :, 0]`: one number from every row. A row says how one token divides its own attention; see [A Vision Transformer from scratch](/course/open/cv-vit?lang=en).",
            "features": "The weights only say how much each token takes from the others; what the class token holds after the block are the weighted values, with the residual connection and the MLP. Row 0 of the weights is its attention: 17 shares that add up to 1.",
        },
    )
    quick_row
    return (quick_row,)


@app.cell(hide_code=True)
def quick_check_probe():
    quick_probe = L.choice(
        "quick-probe",
        question="A linear probe learned 10 classes from DINOv2's features of 20 photos each. What does it get from DINOv2 for one photo?",
        options={
            "right": "384 numbers, the photo's features: always the same for the same photo, as DINOv2 is frozen",
            "pixels": "The 150,528 numbers of the prepared photo, 3 · 224 · 224",
            "changing": "384 numbers that change a little in every epoch, as DINOv2 learns along",
        },
        correct="right",
        why={
            "right": "DINOv2 learns nothing, so a photo always gets the same 384 numbers: that is why they are computed once, and only the probe's 3,850 numbers learn.",
            "pixels": "The pixels go into DINOv2; the probe gets what comes out, 384 numbers per photo. On the pixels, the same probe got only 33 of the 100 test photos right; see [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en).",
            "changing": "DINOv2 is frozen: its 22 million parameters get no gradient, so its features stay the same. Only the probe learns.",
        },
    )
    quick_probe
    return (quick_probe,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        A linear probe on DINOv2's features gets 98 of 100 test photos of starfish, roosters, chandeliers and the rest right. But right for which reason? A model can know a starfish by the sand it lies on, or a water lily by the water around it, and then fail on a starfish on a rock. To trust a model, you want to see which parts of a photo its answer comes from.

        A Vision Transformer has something a CNN does not: numbers that say how much every token takes from every other, its attention weights. DINOv2 is a Vision Transformer, and its 384 features are what its class token holds after the last of its 12 blocks. In every block, the class token takes from the patches. So you can read where it looks, layer by layer. This lesson reads those numbers, follows them through all 12 layers, and then tests whether they show what the model really uses.

        **An everyday example.** Following someone's eyes while they read a map tells you where they look. Whether they needed that part, you find out by covering it: do they still find the way?
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_keep():
    guess_keep = L.guess(
        "keep",
        question="In DINOv2's last layer, the class token's attention ranks the 256 patches of a photo. Keep only the quarter it looks at most, 64 patches, and leave out the rest: the probe gets what the class token makes of those 64. With all 256 patches, it gets 98 of the 100 test photos right. How many now?",
        options={
            "quarter": "About 25",
            "half": "About 50",
            "three-quarters": "About 75",
            "almost-all": "More than 90",
        },
        reveal="**94.** With 64 patches drawn at random, it gets 81 to 86, over 5 draws; with the 64 the class token looks at least, 40. The quarter of a photo the class token looks at most holds almost everything the probe needs. Exercise 4 and the second play block measure it with your own code.",
    )
    guess_keep
    return (guess_keep,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The class token's map.** DINOv2-small cuts a photo of 224 by 224 pixels into 16 by 16 patches of 14 by 14 pixels: with the class token in front, 257 tokens. A layer here is one block: layer 1 is the first block, layer 12 the last. In each of its 12 blocks, each of its 6 heads computes attention weights of shape (257, 257), as a head of your ViT does: row i says how token i divides its attention over all 257 tokens, and adds up to 1. Row 0 is the class token's. Its first number is how much the class token takes from itself; the other 256 say how much it takes from each patch. Put those 256 back in 16 rows of 16, and you have a map over the photo: where the class token looks, in that head of that layer. The course stacks the 6 heads of a block, so a block gives weights of shape (6, 257, 257).

        **Heads look at different things.** The picture shows the coffee, the course's own photo, with the class token's map of each head of layer 12 over it: the clearer a patch, the more the class token takes from it. Heads 1, 3 and 5 look mostly into the cup, at the coffee; head 4 is spread over the whole photo, the table as much as the cup. The DINO paper (Caron and others, 2021) showed such maps from a Vision Transformer that learned without labels, as DINOv2 did: in the last layer, different heads look at different objects or parts.
        """),
    ])
    return


@app.cell(hide_code=True)
def heads_picture_cell():
    mo.vstack([
        mo.Html(C.heads_picture({"head": "head {n}", "average": "average of the heads", "rollout": "rollout, 12 layers",
                                 "alt": "The coffee eight times, with a map over it: the six heads of layer 12, their average and the rollout of all 12 layers. Heads 1, 3 and 5 look mostly into the cup, at the coffee; head 4 is spread over the whole photo; so is the rollout, with a few brighter patches."})),
        mo.md(r"""
        Over every patch lies a veil in the color of the page, thinner the larger the map's number, and gone at its three largest numbers. The last two maps come later in this section: the average of the heads, and the rollout.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_measure():
    mo.vstack([
        mo.md(r"""
        **Measuring a map.** A map can look convincing and still be wrong; so the course measures it. On every photo of the 10 classes of Caltech-101, its makers drew an outline around the object by hand. The share of a map that lies inside the outline says how much of the class token's attention goes to the object: every patch's number times the part of the patch inside the outline, added up, divided by the sum of the map. A map spread evenly over the photo would get as much as the object covers: 51% on average, over the 300 photos.

        **Layer by layer.** In layer 12, the average of the 6 heads puts 71% of the class token's attention inside the outline, and more than the object covers on 277 of the 300 photos. Head 3 puts 84% there, head 4 only 39%: it looks past the object, as it is spread over the coffee's table. In layers 1 to 10, the share stays between 50% and 58%, close to a map spread evenly: there the class token still takes from everywhere. Only in layers 11 and 12 does it go to the object.
        """),
    ])
    return


@app.cell(hide_code=True)
def layers_picture_cell():
    mo.Html(C.layers_picture({"title": "Class token's attention inside the outline", "x": "layer", "y": "% inside the outline",
                              "rollout": "rollout", "area": "the object's area: a map spread evenly",
                              "alt": "Bars for the share of the class token's attention inside the outline, over 300 photos: 50% to 58% in layers 1 to 10, 74% in layer 11, 71% in layer 12, and 57% for the rollout of all 12 layers; a dashed line at 51%, the part of the photo the objects cover."}, L.lang))
    return


@app.cell(hide_code=True)
def text_idea_rollout():
    mo.vstack([
        mo.md(r"""
        **Attention rollout.** In layer 12, the class token takes from the 256 tokens of the patches, but those are not the patches of the photo any more: in every layer before, each of them took from the others. To follow the class token back to the photo, attention rollout (Abnar and Zuidema, 2020) multiplies the layers. First it makes one matrix per layer, `W`, in two steps:

        1. **The average of the heads**, of shape (257, 257): one matrix instead of 6.
        2. **The residual connection.** A block adds what attention brings to the token itself, as in [Residual connections](/course/open/nn-residual?lang=en): a token keeps its own numbers too. So rollout adds the identity matrix of [One-hot vectors for classes](/course/open/ml-one-hot?lang=en), attention of every token to itself, and takes half of each: `W = 0.5 · average + 0.5 · I`. Every row of `W` still adds up to 1. In PyTorch, `torch.eye(t)` is the identity matrix of t by t, as `np.eye`.

        Row i of `W` says from which tokens before this layer token i takes its numbers. Then `R` starts as the identity matrix, and every layer, from the first to the last, does `R = W @ R`, a matrix multiplication as in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en): row i of `R` says how much of token i comes from each token of the photo, before the first block. After layer 12, row 0 of `R`, without its first number, is the rollout's map. The ViT paper drew its maps of attention this way.

        **What rollout finds here.** On the 300 photos, the rollout puts 57% inside the outline, against 71% for layer 12 alone, and more than the object covers on 228 of the 300 photos. Following the information through every layer sounds more complete, but here the map ends up more spread than that of layer 12 alone. Exercise 3 asks why.
        """),
        L.watch_out("A map of attention says where the class token takes its numbers from, not why, and not for which class: DINOv2 learned without classes, and the probe comes after it. And by layer 12, the token of a patch holds much from other patches: a clear patch says *this token*, not *these pixels*. That a map looks right does not show that the model needs those patches. Leaving them out does: the second play block measures it."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **attention rollout** | a map of where a ViT's class token takes its numbers from, followed through every layer: per layer the average of the heads, plus the identity matrix for the residual connection, each halved, the layers multiplied | `R = W @ R` |
        | **`torch.sort(x, descending=True)`** | the numbers of x from the largest to the smallest; `.indices` holds the places they came from | `torch.sort(torch.tensor([0.2, 0.9, 0.5]), descending=True).indices` gives `tensor([1, 2, 0])` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see where a ViT's class token looks, and test it:

    1. Run the photo through the model and keep the attention weights of every block: a tensor of shape `(heads, T, T)` per block, with T = 1 + the number of patches.
    2. **One layer:** row 0 of every head without its first number, `weights[:, 0, 1:]`, in the shape of the photo: `(heads, rows, cols)`.
    3. **All layers:** start with the identity matrix, `R = I = torch.eye(T)`; for every block, from the first to the last, `W = 0.5 · (the average of the heads) + 0.5 · I` and `R = W @ R`; at the end, `R[0, 1:]` in the shape of the photo.
    4. **Test it:** leave out the k patches a map ranks highest, the first k of `torch.sort(m.reshape(-1), descending=True).indices`, give the model the class token and the other patches, and count what the probe still gets right, against k patches left out at random.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell gives DINOv2 the coffee and keeps the attention weights of every block, step 1. `dino.looks(x)` runs the blocks and gives the weights of each, as `Block.weights` in `everything_ai/dinov2.py` computes them: for a batch of n photos, a tensor of shape `(n, 6, 257, 257)` per block; `w[0]` is the one photo. The cell prints how many blocks there are and the shape of the last, then the sums of the class token's rows in layer 12, the first number of each row, and for each head the patch it takes the most from, as its row and its column: patch j lies in row `j // 16` and column `j % 16`. `.argmax(dim=1)` works as `.argmax` with an axis in [Sums and means along an axis](/course/open/np-axis?lang=en). `dino` is DINOv2-small, which the cell below loads once both downloads are done. Steps 2 to 4 of the recipe are the exercises.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download DINOv2-small and Caltech-101, 225 MB")
    return (get_button,)


@app.cell(hide_code=True)
def dino_cell(get_button):
    dino = C.dino_or_stop(get_button, {
        "ask": "*DINOv2-small and the photos of Caltech-101 are not on your computer yet. The button downloads them once: DINOv2-small from Hugging Face, 88 MB, and Caltech-101 from Caltech, 137 MB, whose photos it then unpacks. If you did [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en), they are there already. The worked example and the play blocks need them; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading {name}** · {done} of {total} MB",
        "unpacking": "**Unpacking the photos of Caltech-101** · {done} photos",
        "no_net": "*The download did not work: Hugging Face or Caltech could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 225 MB and the unpacked photos. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*DINOv2-small and the photos are on your computer: 12 blocks of 6 heads, and 257 tokens for a photo of 224 by 224 pixels.*",
    })
    return (dino,)


@app.cell
def example_looks(dino):
    import torch
    from everything_ai.dinov2 import prepare

    # 1. the attention weights of every block for the coffee
    with torch.no_grad():
        layers = [w[0] for w in dino.looks(prepare("data/photos/coffee.png"))]
    print(len(layers), layers[11].shape)
    # the class token's row in each head of layer 12
    row = layers[11][:, 0]
    print(row.sum(dim=1))
    print(row[:, 0])
    # where each head of layer 12 looks most: the patch with the largest number, as its row and column
    best = row[:, 1:].argmax(dim=1)
    print(best // 16, best % 16)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    There are 12 blocks, each with weights of shape `(6, 257, 257)`. Every row of the class token adds up to 1, as a softmax does. Its first number differs per head: head 3 takes only 0.02 from the class token itself, heads 1 and 4 more than 0.17. Heads 1, 3 and 5 take the most from a patch in rows 3 to 5 and columns 4 to 10: inside the cup, as in the picture of *The idea*. Head 4 takes the most from patch (15, 15), the bottom right corner of the photo, where there is no cup.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_class():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The class token's map

        Complete `class_attention(weights, rows, cols)`: step 2 of the recipe. `weights` holds the attention weights of one layer, shape `(heads, T, T)` with T = 1 + rows · cols; token 0 is the class token, and the patch in row r and column c is token 1 + cols · r + c. Give back a map per head, shape `(heads, rows, cols)`. The check makes up weights in which every row adds up to 1, for 2 to 6 heads and photos of 2 by 2, 2 by 3 and 3 by 2 patches. Under the check, your function reads the coffee's weights of layer 12.
        """),
        L.task("Make `class_attention` give a map per head of where the class token looks."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class():
    box_class = L.code_box("class-attention", lines=6)
    box_class
    return (box_class,)


@app.cell(hide_code=True)
def check_class(box_class):
    result_class = L.check("class-attention", box_class.value)
    result_class
    return (result_class,)


@app.cell(hide_code=True)
def hints_class():
    L.hints(
        "class-attention",
        hints=[
            "You need one row of every head, and in that row all numbers but the first. Which index picks the heads, which the row, and which the numbers?",
            "`weights[:, 0, 1:]`: all heads, row 0, from place 1 on. That gives shape `(heads, rows * cols)`.",
            "Give it the shape of the photo: `return weights[:, 0, 1:].reshape(___, rows, cols)`, where the blank is the number of maps, one per head.",
        ],
        why="Row 0 of every head belongs to the class token; its first number is the class token's attention to itself, so `1:` keeps the patches. The patches come row by row, so `reshape(-1, rows, cols)` puts every number back on its patch, and `-1` keeps one map per head.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rollout():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Attention rollout

        Complete `rollout(layers, rows, cols)`: step 3 of the recipe. `layers` holds the weights of every layer, in order, each of shape `(heads, T, T)`. Fill in the three `...`: `W` for one layer, `R` after this layer, and at the end the class token's row, without itself, as a map of `rows` by `cols`. The check makes up 1 to 3 layers of 2 or 3 heads. Under the check, your map of the second case beside the right one, where they differ.
        """),
        L.task("Make `rollout` follow the class token back through every layer to the patches."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rollout():
    box_rollout = L.code_box("rollout", lines=13)
    box_rollout
    return (box_rollout,)


@app.cell(hide_code=True)
def check_rollout(box_rollout):
    result_rollout = L.check("rollout", box_rollout.value)
    result_rollout
    return (result_rollout,)


@app.cell(hide_code=True)
def hints_rollout():
    L.hints(
        "rollout",
        hints=[
            "Three things: one matrix for the layer, made from the heads and the residual connection; this layer after the layers before; and a row of the result.",
            "The average of the heads is `weights.mean(dim=0)`, of shape (T, T); the identity matrix is `I`, at the top of the function. Take half of each. Then the new `R` is this layer's `W` times the old `R`.",
            "`W = 0.5 * weights.mean(dim=0) + 0.5 * I`, `R = W @ R`, and `return R[___, 1:].reshape(rows, cols)`, with the class token's row in the blank.",
        ],
        why="The heads are axis 0, so `mean(dim=0)` averages them. The identity stands for the residual connection: every token keeps its own numbers too. With half of each, every row of `W` adds up to 1. `W @ R` puts this layer after the layers before: row 0 of the result says how much of the class token comes from each token of the photo, and `1:` leaves out the class token's part of itself.",
    )
    return


@app.cell(hide_code=True)
def text_look():
    mo.md(r"""
    ### Look around

    Choose a photo, a layer and a map. The coffee is the course's own photo; the others are the first test photo of each class of Caltech-101, with the outline its makers drew as a dashed line. *head 1* to *head 6* and *average of the heads* are the class token's maps in the layer you choose; *rollout* follows it from layer 1 to that layer. Under the photo: how much of the map lies inside the outline, against what the object covers. Once exercises 1 and 2 are right, the maps come from your `class_attention` and your `rollout`.
    """)
    return


@app.cell(hide_code=True)
def look_controls():
    look_photo = mo.ui.dropdown(options={"coffee": "coffee"} | {C.NAMES[L.lang][c]: c for c in C.CLASSES}, value="coffee", label="Photo")
    look_layer = mo.ui.slider(1, 12, value=12, label="Layer", show_value=True)
    look_map = mo.ui.radio(options={f"head {h}": str(h) for h in range(1, 7)} | {"average of the heads": "average", "rollout": "rollout"},
                           value="average of the heads", label="Map", inline=True)
    mo.vstack([look_photo, look_layer, look_map])
    return look_layer, look_map, look_photo


@app.cell(hide_code=True)
def look_cell(dino, look_photo, look_layer, look_map, box_class, result_class, box_rollout, result_rollout):
    _ca, _ca_yours = C.yours_or_course(box_class.value, "class_attention", result_class.state == "correct")
    _ro, _ro_yours = C.yours_or_course(box_rollout.value, "rollout", result_rollout.state == "correct")
    _key, _layer, _which = look_photo.value, look_layer.value, look_map.value
    try:
        _m = C.view(C.looks(dino, _key), _layer, _which, _ca, _ro)
        _failed = None
    except Exception:
        _m, _failed = C.view(C.looks(dino, _key), _layer, _which), True
    _title = (f"Rollout of layers 1 to {_layer}" if _which == "rollout" else
              f"Layer {_layer}, average of the heads" if _which == "average" else f"Layer {_layer}, head {_which}")
    _inner = C.inside(C.path_of(_key))
    if _inner is None:
        _said = "The coffee has no outline: here you see the map alone."
    else:
        _said = (f"**{C.percent(C.share(_m, _inner))}%** of this map lies inside the outline; the {C.NAMES[L.lang][_key]} covers "
                 f"{C.percent(float(_inner.mean()))}% of the photo.")
    _who = ("" if (_ca_yours if _which != "rollout" else _ro_yours) else
            (" The map comes from the course's function: yours stops with an error here." if _failed else
             f" The map comes from the course's `{'rollout' if _which == 'rollout' else 'class_attention'}`, until your exercise {'2' if _which == 'rollout' else '1'} is right."))
    mo.vstack([mo.Html(C.look_picture(_key, _m, {"title": _title, "alt": f"The photo with the map over it: {_title}."})), mo.md(_said + _who)])
    return


@app.cell(hide_code=True)
def text_after_look():
    mo.md(r"""
    In layer 12, the average of the heads lies on the starfish: 55% of the map, on a starfish that covers 10% of the photo. In each of the ten photos of Caltech-101, head 4 puts the least inside the outline: 9% for the starfish. Choose a lower layer: in layer 6, the starfish gets 18% of the map, a third of what it gets in layer 12. Not every photo works this well: on the lotus, even layer 12 puts 47% inside an outline that covers 47% of the photo. Choose *rollout*: it is spread too, with a few bright patches.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.md(r"""
    ### Exercise 3. Why rollout spreads
    """)
    return


@app.cell(hide_code=True)
def choice_spread_cell():
    choice_spread = L.choice(
        "spread",
        question="Over the 300 photos, the class token's attention in layer 12 puts 71% inside the outline, and the rollout of all 12 layers only 57%. Why does the rollout lie less on the object?",
        options={
            "right": "It multiplies in layers 1 to 10 too, where the class token takes from about everywhere",
            "no-residual": "Its identity half, for the residual connection, spreads the attention over the photo",
            "one-head": "It uses only one head per layer, and one head can look past the object",
        },
        correct="right",
        why={
            "right": "Rollout of layers 11 and 12 alone puts 72% inside the outline; from layer 9 on, 65%; from layer 1 on, 57%. Every early layer it multiplies in adds a map that is spread: in layers 1 to 10, the class token's attention puts only 50% to 58% inside the outline. And `R` ends with the `W` of layer 1, so row 0 of `R` is a weighted average of the rows of that `W`: of where the tokens look in layer 1. On 270 of the 300 photos, at least two of the rollout's three brightest patches are among the three the class token looks at most in layer 1.",
            "no-residual": "Without the identity, `W` is the average of the heads alone, and the rollout of all 12 layers puts 56% inside the outline: a little less than with it. The identity keeps part of every token's own numbers; what spreads the map are the early layers.",
            "one-head": "Rollout averages the heads, `weights.mean(dim=0)`, as the average of layer 12 does too. The difference lies in the layers: rollout takes in all 12, and layers 1 to 10 look about everywhere.",
        },
    )
    choice_spread
    return (choice_spread,)


@app.cell(hide_code=True)
def text_exercise_keep():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. Leave patches out

        A map says where the class token looks. Whether DINOv2 needs those patches, you test by leaving them out: a ViT takes any number of tokens, so you simply do not give it those patches. Each token gets its position before it goes into the blocks, so it keeps its place in the photo; and as [A Vision Transformer from scratch](/course/open/cv-vit?lang=en) said, attention does not care in which order the tokens come. Complete `keep(m, k)`: step 4 of the recipe. `m` is a map of shape `(rows, cols)`, `k` how many patches to leave out. Give back the numbers of the tokens to keep, as one tensor: the class token, 0, and every patch except the k with the largest numbers in `m`; patch j is token 1 + j. The order does not matter: the check sorts what you give back. Under the check, your `keep` leaves out half of the coffee's patches.
        """),
        L.task("Make `keep` give the class token and every patch but the k a map ranks highest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_keep():
    box_keep = L.code_box("keep", lines=6)
    box_keep
    return (box_keep,)


@app.cell(hide_code=True)
def check_keep(box_keep):
    result_keep = L.check("keep", box_keep.value)
    result_keep
    return (result_keep,)


@app.cell(hide_code=True)
def hints_keep():
    L.hints(
        "keep",
        hints=[
            "First put the patches in order, from the largest number in `m` to the smallest; then skip the first k, and add the class token.",
            "`torch.sort(m.reshape(-1), descending=True).indices` gives the patch numbers in that order. `[k:]` skips the first k. Patch j is token j + 1.",
            "`order = torch.sort(m.reshape(-1), descending=True).indices` and `return torch.cat([torch.tensor([0]), order[___] + 1])`, with the patches to keep in the blank.",
        ],
        why="`reshape(-1)` makes the map one row, so that patch j is number j, as in the token numbers. `torch.sort` with `descending=True` puts the largest first, and `.indices` gives where each came from, so `order[k:]` are the patches left after the k highest. Adding 1 turns a patch number into its token number, and `torch.tensor([0])` in front keeps the class token, whose numbers are DINOv2's features.",
    )
    return


@app.cell(hide_code=True)
def text_leave():
    mo.md(r"""
    ### Leave patches out

    Choose a photo, the map whose largest numbers go first, and how many patches to leave out. *most attended first* uses the average of the heads in layer 12; *least attended first* the same map the other way round; *random* random numbers; *rollout's most first* the rollout of all 12 layers. You see the photo with the patches left out covered, and what the probe of [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en), trained on 20 photos per class, makes of the class token of what is left. The first time the play block runs, it computes DINOv2's features of the 300 photos for the probe, once: on the course's computer about half a minute. The chart shows what the course measured on the 100 test photos; the button counts it with your `keep`, the first time in about 15 seconds, as it then also computes the maps of the 100 test photos, once.
    """)
    return


@app.cell(hide_code=True)
def leave_controls():
    leave_photo = mo.ui.dropdown(options={C.NAMES[L.lang][c]: c for c in C.CLASSES}, value=C.NAMES[L.lang]["starfish"], label="Photo")
    leave_map = mo.ui.radio(options={"most attended first": "top", "least attended first": "low", "random": "random", "rollout's most first": "rollout"},
                            value="most attended first", label="Leave out", inline=True)
    leave_k = mo.ui.slider(0, 224, step=32, value=128, label="Patches left out, of 256", show_value=True)
    count_button = mo.ui.run_button(label="Count on the 100 test photos with your keep")
    mo.vstack([leave_photo, leave_map, leave_k, count_button])
    return count_button, leave_k, leave_map, leave_photo


@app.cell(hide_code=True)
def leave_cell(dino, leave_photo, leave_map, leave_k, box_keep, result_keep):
    _keep, _yours = C.yours_or_course(box_keep.value, "keep", result_keep.state == "correct")
    _key, _which, _k = leave_photo.value, leave_map.value, leave_k.value
    _out = C.leave_one(dino, _key, _which, _k, _keep)
    _name = C.NAMES[L.lang]
    _said = (f"Left out: **{_k}** patches. The probe now says **{_name[_out['now']]}**, with a probability of {C.percent(_out['p_now'])}%; "
             f"with all 256 patches, {_name[_out['all']]}, {C.percent(_out['p_all'])}%.")
    _who = "" if _yours and not _out["stood_in"] else " The course's `keep` chose the patches, until your exercise 4 is right."
    _chart = C.left_out_chart(_k, {"title": "Test photos right, of 100", "x": "patches left out, of 256", "y": "right",
                                   "top": "most attended first", "low": "least attended first", "random": "random", "rollout": "rollout's most first",
                                   "alt": "Four lines over the patches left out, from 0 to 224 of 256: the test photos the probe gets right falls fastest when the most attended patches go first, to 29 at 224; slowest when the least attended go first, 85 at 224; random patches in between, 59 at 224; the rollout's most attended first 42 at 224."}, L.lang)
    mo.vstack([mo.Html(C.leave_picture(_key, _out["kept"], {"title": f"{_name[_key]}, {_k} patches left out", "alt": f"The {_name[_key]} with {_k} patches covered."})),
               mo.md(_said + _who), mo.Html(_chart)])
    return


@app.cell(hide_code=True)
def count_cell(dino, count_button, leave_map, leave_k, box_keep, result_keep):
    if not count_button.value:
        _count = mo.md("*Click the button to count with your `keep` on the 100 test photos.*")
    else:
        _keep, _yours = C.yours_or_course(box_keep.value, "keep", result_keep.state == "correct")
        _which, _k = leave_map.value, leave_k.value
        mo.output.replace(mo.md("*Computing the maps of the 100 test photos, once, and leaving out patches…*"))
        try:
            _right = C.right_without(dino, _keep, _which, _k, progress=lambda _n: mo.output.replace(mo.md(f"*Test photo {_n} of 100…*")))
            _stood = False
        except Exception:
            _right, _stood = C.right_without(dino, C.course("keep"), _which, _k), True
        _ref = C.REFERENCE["left_out"][_which][C.KS.index(_k)]
        _whose = "your `keep`" if _yours and not _stood else "the course's `keep`, as yours is not right yet" if not _stood else "the course's `keep`, as yours stops with an error here"
        _how = {"top": "most attended first", "low": "least attended first", "random": "random", "rollout": "rollout's most first"}[_which]
        _count = mo.md(f"With {_whose}, leaving out {_k} patches, *{_how}*: the probe gets **{_right}** of the 100 test photos right. The course measured {_ref}.")
    _count
    return


@app.cell(hide_code=True)
def text_after_leave():
    mo.md(r"""
    Leaving out the 64 patches the class token looks at most in layer 12 costs the probe 14 test photos: from 98 to 84. 64 patches at random cost it 1, over all 5 draws. Leave out the 192 least attended, so that only the 64 most attended stay, and it still gets 94 right. The map points at what the model uses: the patches it ranks high matter far more than random ones, and those it ranks low hardly matter. The rollout's ranking lies in between: leaving out its 64 most attended patches costs 10 test photos. By this test too, layer 12 alone points better than the rollout of all 12 layers, on this model.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_what():
    mo.md(r"""
    ### Exercise 5. What a map shows
    """)
    return


@app.cell(hide_code=True)
def explain_what_cell():
    explain_what = L.explain(
        "what-it-shows",
        question="A colleague shows you the class token's map of layer 12 over a photo of a water lily, bright on the water around the flower, and says: \"So the model knows water lilies by the water.\" What does such a map show, what does it not show, and how would you find out whether the colleague is right?",
        model_answer="The map shows from which tokens the class token takes its numbers in layer 12, as shares that add up to 1. It does not show why, nor for which class: DINOv2 has no classes, and by layer 12 the token of a patch holds numbers from other patches, so a bright patch of water need not be about the pixels of the water. To find out, leave the water patches out and see whether the probe still says water lily, against as many patches left out at random; or test the probe on water lilies without water around them.",
        points=[
            "it shows where the class token takes its numbers from, in that layer",
            "it does not show why, or for which class",
            "a token in a late layer holds numbers from other patches",
            "test it: leave those patches out and compare with random patches",
        ],
    )
    explain_what
    return (explain_what,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The attention weights of your ViT in [A Vision Transformer from scratch](/course/open/cv-vit?lang=en) are numbers you can read: in DINOv2, the class token's row of each head is a map of where it looks, and in its last two layers that map lies on the object. Attention rollout follows the class token through all layers with matrix multiplications, but on DINOv2-small it mostly adds the first ten layers, which look about everywhere. Your `keep` turned the map into a test: leaving out the patches the class token looks at most costs the probe of [What a vision model sees: features and a linear probe](/course/open/cv-pretrained-features?lang=en) far more than leaving out random ones. A picture of what a model looks at is a claim; leaving parts out tests it. This was the last lesson of the module on Vision Transformers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_class, result_rollout, choice_spread, result_keep, explain_what):
    L.wrap_up(result_class, result_rollout, choice_spread, result_keep, explain_what)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
