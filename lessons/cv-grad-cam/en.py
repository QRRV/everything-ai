# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-grad-cam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-grad-cam", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) and [Looking inside a training run](/course/open/nn-activation-stats?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_leaf():
    quick_leaf = L.choice(
        "quick-leaf",
        question="`a = torch.tensor(2.0, requires_grad=True)`, then `c = a * 3`, `d = c * c` and `d.backward()`. What is `c.grad`?",
        options={
            "twelve": "`tensor(12.)`: the slope of `d` with respect to `c`, $2c$",
            "right": "`None`: `c` is not a leaf tensor, so it passes its slope on and keeps none",
            "thirtysix": "`tensor(36.)`: the slope of `d` with respect to `a`",
        },
        correct="right",
        why={
            "twelve": "PyTorch does work out 12 on the way back, and passes it on to `a`; but `c` is computed from `a`, not a leaf tensor, so it keeps nothing in `.grad`. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "thirtysix": "36 is the slope of $d = 9a^2$ with respect to `a`, $18a$: that is `a.grad`, of the leaf tensor. `c` is computed from `a`, so it keeps no slope. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "right": "Only leaf tensors keep their slopes. Today you need the slopes of a tensor in between, the maps of a layer, and one call asks PyTorch to keep them.",
        },
    )
    quick_leaf
    return (quick_leaf,)


@app.cell(hide_code=True)
def quick_check_hook():
    quick_hook = L.choice(
        "quick-hook",
        question="A forward hook `keep` on `resnet.layer4` puts its output in a list `maps`. After `resnet(x)` for one photo of 224 by 224, what is `maps[0].shape`?",
        options={
            "right": "(1, 512, 7, 7): the 512 feature maps of `layer4`, each 7 by 7",
            "pooled": "(1, 512): one number per map, what the last layer gets",
            "logits": "(1, 1000): a logit per class",
        },
        correct="right",
        why={
            "pooled": "The last layer gets one number per map, but the hook hangs on `layer4`, and gets what `layer4` gives: the maps themselves. See [Looking inside a training run](/course/open/nn-activation-stats?lang=en).",
            "logits": "The logits are what the whole model gives. A hook gets the output of the layer it hangs on, here `layer4`. See [Looking inside a training run](/course/open/nn-activation-stats?lang=en).",
            "right": "A hook gets the output of its layer: the 512 maps of 7 by 7 of [Look inside a CNN](/course/open/cv-inside-cnn?lang=en). Today you make one map from them for a class.",
        },
    )
    quick_hook
    return (quick_hook,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A saliency map, from [Saliency: the pixels that change the answer](/course/open/cv-saliency?lang=en), is sharp but speckled: it shows pixel by pixel where a score is sensitive, and its speckles spread over the whole photo. Zhou and colleagues (2016) found a coarser map in the last feature maps of a CNN, the class activation map, which lies on the places of a class. Their network learned only which class each photo of ImageNet shows, and never saw a box. Yet boxes around the bright parts of its maps found the object in 62.9% of the test photos of ILSVRC 2014. A photo counted as found when one of its five guesses had the right class, with a box close enough to the object's own box: a top-5 localization error of 37.1%, close to the 34.2% of [AlexNet](/course/open/cv-alexnet?lang=en), which had learned from the boxes. It needs a CNN that ends with the mean of every map and one linear layer. Selvaraju and colleagues (2017) made a map of the same kind for any CNN, from the gradient: Grad-CAM, "a class-discriminative localization technique". Today you make both for ResNet-18, on a photo with two things in it.
    """)
    return


@app.cell(hide_code=True)
def predict_espresso():
    guess_espresso = L.guess(
        "where-espresso",
        question="A collage, 224 by 224: on the left the middle half of the cat, on the right the middle half of the coffee. ResNet-18's best guess is Egyptian cat, at 0.40; espresso is its fifth, at 0.02. Where does the Grad-CAM of espresso lie?",
        options={
            "cat": "On the cat half: ResNet-18 looks at the cat, whatever class you ask about",
            "right": "On the coffee half: each class gets its own map",
            "spread": "Spread over both halves: at 0.02, ResNet-18 sees no espresso",
        },
        reveal="**On the coffee half.** 87% of the Grad-CAM of espresso lies on the coffee half, and 95% of that of Egyptian cat on the cat half. A map comes from the logit of one class, so it does not care how sure the model is: it shows where the evidence for that class lies. The slider in *The idea* shows the maps of all 8 best guesses.",
    )
    guess_espresso
    return (guess_espresso,)


@app.cell(hide_code=True)
def text_idea():
    _frames = C.guess_frames({
        "title": "Guess {rank}: {name}, {p}",
        "cat": "cat half {share}%",
        "coffee": "coffee half {share}%",
        "alt": "The collage of the cat and the coffee, veiled where the Grad-CAM of {name} is low: {cat}% of the map on the cat half, {coffee}% on the coffee half.",
    }, L.lang)
    _guesses = L.explore("guesses", "guess", lambda _v: _frames[int(_v)], start=1, values=C.RANKS, role="",
                         task="Slide through ResNet-18's 8 best guesses for the collage. Which guess has a map that is split between the two halves?")
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Global average pooling.** For a photo of 224 by 224, `layer4` of ResNet-18 gives 512 feature maps of 7 by 7. The last layer, `fc`, does not see them as maps: `features` in the course's ResNet-18 ends with `x.mean(dim=(2, 3))`, the mean of each map, 512 numbers, and `fc`, a linear layer, makes the 1,000 logits from those. Taking the mean of each whole map is called **global average pooling**. It keeps no places: a cat in a corner and a cat in the middle can give the same means.

        **The weights put the places back.** Write $A_k$ for map $k$ of `layer4`, $A_k(i, j)$ for its number in row $i$ and column $j$, $w_{ck}$ for the weight in `fc` from map $k$ to class $c$, and $b_c$ for the bias of class $c$. Then the logit of class $c$ is

        $$z_c = \sum_k w_{ck} \cdot \frac{1}{49}\sum_{i,j} A_k(i, j) + b_c = \frac{1}{49}\sum_{i,j} \Big(\sum_k w_{ck} A_k(i, j)\Big) + b_c$$

        The mean and the weighted sum can swap places. The part in brackets is a map of 7 by 7, the **class activation map** of class $c$, CAM for short:

        $$M_c(i, j) = \sum_k w_{ck} A_k(i, j)$$

        Its mean plus the bias is the logit, exactly. So each of its 49 numbers, divided by 49, is what that place of the photo adds to the logit: a large number, a place that speaks for the class; a negative number, one that speaks against it. Zhou and colleagues leave the bias out: it is the same at every place.

        **Grad-CAM: the weights from the gradient.** Not every CNN ends with global average pooling and one linear layer: LeNet-5, from [LeNet-5, the CNN that read checks](/course/open/cnn-lenet?lang=en), puts its last maps into linear layers as they are, number by number. There is no weight per map to take. Selvaraju and colleagues (2017) find one from the slopes of the logit with respect to the map, which autograd gives: the mean slope over the map's places,

        $$\alpha_{ck} = \frac{1}{Z}\sum_{i,j} \frac{\partial z_c}{\partial A_k(i, j)}$$

        for a map of $Z$ numbers, 49 for a map of 7 by 7. With those weights they add up the maps, and keep only the positive part, because "we are only interested in the features that have a positive influence on the class of interest":

        $$G_c = \text{ReLU}\Big(\sum_k \alpha_{ck} A_k\Big)$$

        That is **Grad-CAM**. The score is the logit, before softmax, as for a saliency map. For ResNet-18 the two methods meet, from `layer4`: each number $A_k(i, j)$ reaches the logit only through the mean, so its slope is $w_{ck}/49$, at every place. Then $\alpha_{ck} = w_{ck}/49$, and Grad-CAM is the positive part of the class activation map, divided by 49. In the paper's words: "Grad-CAM is a strict generalization of CAM."

        **The slopes of a tensor in between.** A forward hook gets the maps, as in [Looking inside a training run](/course/open/nn-activation-stats?lang=en). But they are no leaf tensor: PyTorch computes them from the photo, and passes their slopes on without keeping them. `A.retain_grad()` asks it to keep them, in `A.grad`, at the next `backward`. And the photo must record: the weights of the course's ResNet-18 keep no slopes, so without `x.requires_grad = True` PyTorch records nothing on the way from the photo to the logit, and the maps have no slopes at all.

        **Coarse.** Each number of a map of `layer4` stands for 32 by 32 pixels of the photo, 224 / 7. It sees far more: its receptive field is 435 by 435 pixels, more than the whole photo. But as [Look inside a CNN](/course/open/cv-inside-cnn?lang=en) showed, the pixels near the middle of that piece count most, so a map still shows where something is. The course draws each as a block: a veil in the color of the page over the photo, thicker where the map is lower, and none where it is largest. Selvaraju and colleagues enlarge the map to the size of the photo with bilinear interpolation, as in [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), which smooths the blocks. They take the last convolutional layer, for "the best compromise between high-level semantics and detailed spatial information".

        The slider shows the collage of the question above, with the Grad-CAM from `layer4` of each of ResNet-18's 8 best guesses for it.
        """),
        _guesses,
        mo.md(r"""
        The four cat classes have 95% or 96% of their maps on the cat half. Espresso, at only 0.02, has 87% of its map on the coffee half, and coffee mug and cup have 79%. Only bucket, at 0.01, is split: 44% on the cat half and 56% on the coffee half. Before the ReLU, the class activation map of Egyptian cat goes as low as −2.54, in row 3 and column 5, on the coffee half; the ReLU makes 15 of the 49 numbers of its Grad-CAM 0, all from column 4 on, on the coffee half. The four cat classes also have almost the same map: class-discriminative means that classes in different places get maps in different places, not that two look-alike classes in the same place get different maps.
        """),
        L.watch_out("Grad-CAM from an earlier layer tells the classes apart less well; here, from `layer3`, not at all. With maps of 14 by 14, the 8 guesses have 37% to 65% of their maps on the cat half: Egyptian cat 37%, espresso 47%. Selvaraju and colleagues saw the same: \"Grad-CAM maps become progressively worse as we move to earlier convolutional layers\". Take the last convolutional layer; the demo lets you try `layer3` with your own function."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **global average pooling** | the mean of each whole feature map: one number per map | `x.mean(dim=(2, 3))` |
        | **class activation map** (CAM) | the maps of the last convolutional layer, each times the weight of a class in the linear layer after global average pooling, added up; its mean plus the bias is the logit | `(A[0] * w.reshape(-1, 1, 1)).sum(dim=0)` |
        | **Grad-CAM** | the maps of a layer, each times the mean slope of the logit of a class over that map, added up, and only the positive part | `torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))` |
        | `.retain_grad()` | keeps the slopes of a tensor that is not a leaf tensor, in its `.grad`, at the next `backward` | `A.retain_grad()` |
        | $A_k(i, j)$ | the number in row $i$ and column $j$ of map $k$ | `A[0, k, i, j]` |
        | $w_{ck}$, $b_c$ | the weight in the linear layer from map $k$ to class $c$, and the bias of class $c$ | `resnet.fc.weight[c, k]`, `resnet.fc.bias[c]` |
        | $M_c(i, j)$ | the class activation map of class $c$, in row $i$ and column $j$ | `cam[i, j]` |
        | $\alpha_{ck}$ | the weight Grad-CAM gives map $k$ for class $c$: the mean of its slopes | `alpha[k]` |
        | $G_c$ | the Grad-CAM of class $c$ | `grad_cam` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see where a CNN found the evidence for a class:

    1. Prepare the photo as for the model, a tensor `x` of shape (1, 3, 224, 224), and turn recording on: `x.requires_grad = True`.
    2. Hang a forward hook on the layer whose maps you want: the last convolutional layer, for ResNet-18 `layer4`. Take the photo through the model, and take the hook off. The maps `A` from the hook have shape (1, K, h, w). The score is the logit of the class, `model(x)[0, c]`, not its probability.
    3. For a CNN that ends with global average pooling and one linear layer, from its last maps: the class activation map. The maps of photo 0, each times the weight of the class in that layer, added up: `(A[0] * w.reshape(-1, 1, 1)).sum(dim=0)`. Its mean plus the bias is the logit.
    4. For any layer of any CNN: Grad-CAM. Call `A.retain_grad()`, then go back from the logit, `.backward()`: `A.grad` has a slope for every number of every map. Per map the mean of its slopes, `alpha = A.grad[0].mean(dim=(1, 2))`; the maps of photo 0, each times its weight, added up; and the positive part, `torch.relu`.
    5. Draw the map: as a gray picture, or over the photo, each number as a block of the photo.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell below the download does the recipe for the cat, prepared as in [Try an image classifier](/course/open/cv-try-classifier?lang=en), with the maps of `layer4`. Step 3 takes the weights of class 285, Egyptian cat, from `fc`, and makes the class activation map, without recording: `w.reshape(-1, 1, 1)` gives the 512 weights the shape (512, 1, 1), so that broadcasting multiplies each whole map by its weight. It prints the mean of the map plus the bias next to the logit, then the largest number and where it is: 7 numbers to a row, so `// 7` gives the row and `% 7` the column. Step 4 makes the Grad-CAM. It prints the first three weights of Grad-CAM, times 49, next to the first three weights of `fc`, and whether the positive part of the class activation map, divided by 49, is within 0.000001 of the Grad-CAM everywhere. Last, it draws the class activation map. The comments with `#` say what each step does. Run it as it is; then change `c` where the comment says, to 967, espresso, and run it again. `resnet` is ResNet-18, which the cell below loads once it is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download ResNet-18, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 is not on your computer yet. If you did [Try an image classifier](/course/open/cv-try-classifier?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once, 47 MB, from Hugging Face. The worked example, the maps of the collage and the demo need it; the slider in *The idea* and the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading ResNet-18** · {done} of {total} MB",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 47 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*ResNet-18 is on your computer: 11.7 million parameters, with torchvision's weights, trained on ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_cam(resnet):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    # 1. the cat, prepared as in Try an image classifier, with recording on
    photo = Image.open("data/photos/chelsea.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    x.requires_grad = True
    # 2. a forward hook keeps the maps of layer4 while ResNet-18 computes the logits
    maps = []
    def keep(module, inputs, output):
        maps.append(output)
    handle = resnet.layer4.register_forward_hook(keep)
    logits = resnet(x)
    handle.remove()
    A = maps[0]
    print(A.shape)
    # 3. the class activation map of class c: each map of photo 0 times its weight in fc, added up
    c = 285  # change: 967
    w = resnet.fc.weight[c]
    print(w.shape)
    with torch.no_grad():
        cam = (A[0] * w.reshape(-1, 1, 1)).sum(dim=0)
    print(cam.shape)
    print(cam.mean() + resnet.fc.bias[c], logits[0, c])
    top = int(cam.argmax())
    print(top // 7, top % 7, cam.max(), cam.min())
    # 4. Grad-CAM: keep the slopes of the maps, go back from the logit, weight each map by the mean of its slopes
    A.retain_grad()
    logits[0, c].backward()
    alpha = A.grad[0].mean(dim=(1, 2))
    with torch.no_grad():
        grad_cam = torch.relu((A[0] * alpha.reshape(-1, 1, 1)).sum(dim=0))
    print(alpha[:3] * 49, w[:3])
    print((torch.relu(cam) / 49 - grad_cam).abs().max() < 0.000001)
    # 5. draw the class activation map: black is its smallest number, white its largest
    plt.imshow(cam.numpy(), cmap="gray")
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `A` has shape (1, 512, 7, 7): the 512 maps of `layer4` for photo 0, the cat. `w` has shape (512,), a weight per map. The class activation map has shape (7, 7), and its mean plus the bias, `tensor(19.6482)`, is the logit of Egyptian cat, 19.6482: global average pooling and the linear layer, in the other order. The logit carries a `grad_fn`: PyTorch recorded the way to it, for step 4. Its largest number, 33.57, is in row 5 and column 1, the whisker pad left of the nose, with its dark spots; the eyes are bright too. All 49 numbers are positive, the smallest 11.32: the whole face speaks for Egyptian cat. In step 4, the weights of Grad-CAM times 49 are the weights of `fc`, and the last line says `tensor(True)`: Grad-CAM is the positive part of the class activation map, divided by 49, up to rounding. With `c = 967`, espresso, the mean plus the bias is again the logit, 0.0781, but the map runs from −5.42 to 2.93, with the largest at row 3 and column 0, in the fur at the left edge: places that speak for espresso and places that speak against it, which add up to almost nothing. Its Grad-CAM is 0 wherever the map is negative.
    """)
    return


@app.cell(hide_code=True)
def text_before_compare():
    mo.md(r"""
    On the collage, the saliency map of [Saliency: the pixels that change the answer](/course/open/cv-saliency?lang=en) and the Grad-CAM from `layer4`, for Egyptian cat and for espresso. A saliency map is white from its 1% largest numbers on, a Grad-CAM at its largest number.
    """)
    return


@app.cell(hide_code=True)
def compare_cell(resnet):
    _x = C.input_of("collage")
    _maps = {"saliency": {_key: C.course_saliency(resnet, _x, _label) for _key, _label in (("egyptian", C.EGYPTIAN), ("espresso", C.ESPRESSO))},
             "gradcam": {_key: C.course_grad_cam(resnet, resnet.layer4, _x, _label).numpy() for _key, _label in (("egyptian", C.EGYPTIAN), ("espresso", C.ESPRESSO))}}
    mo.Html(C.compare_picture(_maps, {
        "saliency": "saliency map of {label}\n{share}% on the cat half",
        "gradcam": "Grad-CAM of {label}\n{share}% on the cat half",
        "alt": "Four gray maps of the collage. Left, the saliency maps of Egyptian cat and of espresso: speckles over both halves, the brightest on the cat for Egyptian cat and on the cup for espresso. Right, their Grad-CAMs: blocks of 7 by 7, on the cat half for Egyptian cat and on the coffee half for espresso.",
    }))
    return


@app.cell(hide_code=True)
def text_after_compare():
    mo.md(r"""
    The brightest parts of the two saliency maps do part: of the 1% largest numbers, drawn white, 98% lie on the cat half for Egyptian cat, and 89% on the coffee half for espresso. But the rest of a saliency map spreads over the whole photo: of all of it, 74% lies on the cat half for Egyptian cat, and only 56% on the coffee half for espresso. The Grad-CAMs lie almost wholly on one half: 95% of that of Egyptian cat on the cat half, and 87% of that of espresso on the coffee half. A saliency map is sharp, pixel by pixel, but spread out; a Grad-CAM is coarse, 7 by 7, and lies on the places of its class.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_logit():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. From the map to the logit
    """)
    return


@app.cell(hide_code=True)
def choice_logit_cell():
    choice_logit = L.choice(
        "logit",
        question="A small CNN ends with two maps of 2 by 2, global average pooling and one linear layer. Map 1 is [[1, 0], [0, 0]] and map 2 is [[0, 0], [0, 2]]. Class c has weight 3 for map 1, weight −1 for map 2, and bias 0.5. Its class activation map is [[3, 0], [0, −2]]. What is the logit of c?",
        options={
            "right": "0.75: the mean of the class activation map, 0.25, plus the bias",
            "sum": "1.5: the sum of the class activation map, 1, plus the bias",
            "nobias": "0.25: the mean of the class activation map",
        },
        correct="right",
        why={
            "sum": "Global average pooling takes the mean of each map, not its sum: map 1 gives 0.25 and map 2 gives 0.5, and 3 × 0.25 − 1 × 0.5 + 0.5 = 0.75. The mean of the class activation map, 0.25, plus the bias is the same.",
            "nobias": "The linear layer adds its bias too: 3 × 0.25 − 1 × 0.5 + 0.5 = 0.75. The class activation map has no bias in it, so its mean is the logit without the bias.",
            "right": "The long way gives the same: global average pooling makes 0.25 and 0.5, and 3 × 0.25 − 1 × 0.5 + 0.5 = 0.75. The −2 of the map is the place that speaks against c.",
        },
    )
    choice_logit
    return (choice_logit,)


@app.cell(hide_code=True)
def text_exercise_class_map():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The class activation map

        Complete `class_map(maps, weights)`: the class activation map of the feature maps `maps` of one photo, a tensor of shape (K, H, W), with one weight per map, `weights`, a tensor of shape (K,): each map times its weight, all added up, a tensor of shape (H, W). The check gives your function the two maps of exercise 1, and other made-up maps. When it is right, it saves your function in your project, as `work/projects/cnn/cam.py`.
        """),
        L.task("Make `class_map` give the class activation map of a photo's maps."),
    ])
    return


@app.cell(hide_code=True)
def exercise_class_map():
    box_class_map = L.code_box("class-map", lines=5)
    box_class_map
    return (box_class_map,)


@app.cell(hide_code=True)
def check_class_map(box_class_map):
    result_class_map = L.check("class-map", box_class_map.value)
    result_class_map
    return (result_class_map,)


@app.cell(hide_code=True)
def hints_class_map():
    L.hints(
        "class-map",
        [
            "`weights` has shape (K,) and `maps` shape (K, H, W). To multiply each whole map by its weight, the weights need the shape (K, 1, 1); broadcasting then stretches each weight over its map.",
            "`weights.reshape(-1, 1, 1)` gives that shape. Then `.sum(dim=0)` adds up along axis 0, the maps.",
        ],
        why="Broadcasting lines shapes up from the right: (K,) against (K, H, W) meets W, the columns. With (K, 1, 1), the K weights meet the K maps, and each is stretched over its H by W numbers.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_grad_cam():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Grad-CAM

        Complete `grad_cam(model, layer, x, label)`: the Grad-CAM of the photo `x`, of shape (1, channels, H, W), for the class `label`, from the maps that `layer` gives, as a tensor in the shape of one map. The box starts with `keep`, the forward hook of [Looking inside a training run](/course/open/nn-activation-stats?lang=en). The check uses a ResNet-18 with random weights, with `layer4` and with `layer3`, and a small CNN with random weights that does not end with global average pooling: on a gray image of 28 by 28, the maps of 14 by 14 of its `conv2` go through a ReLU and max pooling to 7 by 7, and then, number by number, into a linear layer. When it is right, it saves your function in your project, as `work/projects/cnn/gradcam.py`.
        """),
        L.task("Make `grad_cam` give the Grad-CAM of a photo for a class, from any layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grad_cam():
    box_grad_cam = L.code_box("grad-cam", lines=12)
    box_grad_cam
    return (box_grad_cam,)


@app.cell(hide_code=True)
def check_grad_cam(box_grad_cam):
    result_grad_cam = L.check("grad-cam", box_grad_cam.value)
    result_grad_cam
    return (result_grad_cam,)


@app.cell(hide_code=True)
def hints_grad_cam():
    L.hints(
        "grad-cam",
        [
            "Hang `keep` on `layer` and keep the handle; turn recording on for the photo; take the logit of `label`, `model(x)[0, label]`; and take the hook off. `maps[0]` now holds the maps, `A`, of shape (1, K, h, w).",
            "`A.retain_grad()` before `score.backward()`: then `A.grad` has the slopes, in the shape of `A`. The weight of map k is the mean of `A.grad[0, k]`, all of them at once with `.mean(dim=(1, 2))`. Weight `A[0]` with them as in exercise 2, and keep the positive part with `torch.relu`.",
        ],
        why="The slopes of the logit with respect to a map say how much each of its numbers adds; their mean is one weight per map, as the linear layer's weight is for ResNet-18. `A` is computed from the photo, so it is no leaf tensor: without `retain_grad`, PyTorch passes its slopes on and keeps none, and `A.grad` stays `None`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The button uses your project: your `grad_cam` makes the maps of the photo you choose, from the layer you choose, for ResNet-18's three best guesses, and the course draws each over the photo. From `layer4`, the course also makes the class activation map of the best guess with your `class_map` and the weights of `fc`, and compares its positive part, divided by 49, with your Grad-CAM. It needs ResNet-18, from the worked example, and exercises 2 and 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"the cat and the coffee": "collage", "the cat": "cat", "the coffee": "coffee", "the drawing of fruit": "fruit"},
                               value="the cat and the coffee", label="Photo", inline=True)
    layer_choice = mo.ui.radio(options={"layer4, maps of 7 by 7": "layer4", "layer3, maps of 14 by 14": "layer3"},
                               value="layer4, maps of 7 by 7", label="Layer", inline=True)
    look_button = mo.ui.run_button(label="Make the maps with your functions")
    mo.vstack([photo_choice, layer_choice, look_button])
    return layer_choice, look_button, photo_choice


@app.cell(hide_code=True)
def demo_grad_cam(resnet, look_button, photo_choice, layer_choice, result_class_map, result_grad_cam):
    resnet, result_class_map, result_grad_cam  # run again when exercise 2 or 3 saves your module
    if not look_button.value:
        _out = mo.md("*Press the button to make the maps.*")
    else:
        _result = C.demo(photo_choice.value, layer_choice.value)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `class_map` and your `grad_cam` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "class_map": "*Your `class_map` in `work/projects/cnn/cam.py` stops with an error, or does not give a map of 7 by 7. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "grad_cam": "*Your `grad_cam` in `work/projects/cnn/gradcam.py` stops with an error, or does not give a map in the shape of the layer's maps, without negative numbers and not 0 everywhere. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
            }[_result])
        else:
            _out = mo.vstack([
                mo.Html(C.demo_picture(_result, {"title": "{rank}. {name}, {p}", "cat": "cat half {share}%", "coffee": "coffee half {share}%",
                                                 "alt": "The photo three times, each veiled where your Grad-CAM of one of ResNet-18's three best guesses is low."}, L.lang)),
                mo.md(C.demo_sentence(_result, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_evidence():
    mo.md(r"""
    ### Exercise 4. Sure, or there?
    """)
    return


@app.cell(hide_code=True)
def choice_evidence_cell():
    choice_evidence = L.choice(
        "evidence",
        question="On the collage, ResNet-18 gives espresso only 0.02, its fifth guess. Yet 87% of the Grad-CAM of espresso lies on the coffee half, about as clearly as 95% of that of Egyptian cat, at 0.40, lies on the cat half. What does that tell you?",
        options={
            "noise": "The map of a class with a low probability is noise",
            "right": "A Grad-CAM shows where the evidence for a class lies in the photo, not how sure the model is of that class",
            "wrong": "ResNet-18 sees espresso more than it says: its probability is wrong",
        },
        correct="right",
        why={
            "noise": "Noise would not land on the coffee, 87% of it, and leave the cat. The map comes from the logit of espresso alone: the coffee half raises it, the cat half does not.",
            "wrong": "The probability is right for what ResNet-18 computes: softmax weighs the logit of espresso against those of all 1,000 classes. The logit of Egyptian cat, 11.20, is 3.02 larger than that of espresso, 8.18, and softmax turns that into about 20 times as much: 0.40 against 0.02. The map does not compare classes; it shows where the logit of one class comes from.",
            "right": "The map comes from the logit of one class, before softmax. Where that class's evidence lies, the map shows; which class wins, softmax decides, over all 1,000. So a confident answer can have a map in the wrong place, and an unlikely class a map in the right one.",
        },
    )
    choice_evidence
    return (choice_evidence,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The global average pooling at the end of ResNet-18, which [Look inside a CNN](/course/open/cv-inside-cnn?lang=en) showed you as the mean of every map, lets the weights of its last layer put the places back: the class activation map. Grad-CAM finds those weights for any CNN from the gradient, with the forward hooks of [Looking inside a training run](/course/open/nn-activation-stats?lang=en) and the autograd of [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en), plus `retain_grad` for a tensor in between. Where the saliency map of [Saliency: the pixels that change the answer](/course/open/cv-saliency?lang=en) is sharp but spreads over the whole photo, a Grad-CAM is coarse but lies on the places of its class. Your project `cnn` now has `cam.py` and `gradcam.py`. Grad-CAM finds where a class is without ever having seen a box; the next lesson, *Anchors: the boxes a detector starts from*, begins the detectors that learn the boxes themselves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_logit, result_class_map, result_grad_cam, choice_evidence):
    L.wrap_up(choice_logit, result_class_map, result_grad_cam, choice_evidence)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
