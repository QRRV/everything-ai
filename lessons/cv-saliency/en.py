# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-saliency. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-saliency", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en) and [Look inside a CNN: filters and feature maps](/course/open/cv-inside-cnn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grad():
    quick_grad = L.choice(
        "quick-grad",
        question="`x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)`, then `s = (x * x).sum()` and `s.backward()`. What is `x.grad`?",
        options={
            "sum": "`tensor(14.)`: the value of `s`",
            "right": "`tensor([2., 4., 6.])`: a slope for each number of `x`",
            "square": "`tensor([1., 4., 9.])`: each number of `x` squared",
        },
        correct="right",
        why={
            "sum": "14 is `s` itself, 1 + 4 + 9. `x.grad` holds slopes, one for each number of `x`, in its shape. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "square": "Those are the numbers of `x * x`. The slope of $x^2$ is $2x$, so `x.grad` is 2, 4 and 6. See [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en).",
            "right": "The slope of $x^2$ is $2x$, one for each number of `x`, in its shape. Today `x` is a photo, and `x.grad` holds a slope for each of its 150,528 numbers.",
        },
    )
    quick_grad
    return (quick_grad,)


@app.cell(hide_code=True)
def quick_check_pixels():
    quick_pixels = L.choice(
        "quick-pixels",
        question="In ResNet-18, one number of a map of `layer4` sees 435 by 435 pixels, more than the photo of 224 by 224, and the last layer gets the mean of every map of `layer4`. Which pixels of the coffee photo can change the logit of espresso?",
        options={
            "cup": "Only the pixels of the cup",
            "right": "Every pixel of the photo",
            "middle": "Only the middle 7 by 7 pixels, one for each number of a map of `layer4`",
        },
        correct="right",
        why={
            "cup": "The network does not know where the cup is before it looks: every number of `layer4` sees the whole photo, and the logit is made from all of them. See [Look inside a CNN](/course/open/cv-inside-cnn?lang=en).",
            "middle": "A map of `layer4` is 7 by 7, but each of its numbers sees 435 by 435 pixels, more than the whole photo. See [Look inside a CNN](/course/open/cv-inside-cnn?lang=en).",
            "right": "Every pixel reaches the logit, but not all equally. Today you find out which pixels reach it most strongly.",
        },
    )
    quick_pixels
    return (quick_pixels,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    ResNet-18 calls the coffee photo espresso, at 0.85. But a probability does not say what it rests on: the coffee, the cup, or the red saucer under it. A classifier can be right for the wrong reason. Lapuschkin and colleagues (2019) looked at a classifier that recognized horses in the photos of PASCAL VOC 2007, and drew over each photo, with another kind of map than today's, what its answer rested on. It looked at a source tag, a line of text in a corner that about one in five of the horse photos carried. Pasted onto a photo of a car, the tag made the car a horse. Simonyan, Vedaldi and Zisserman (2014) made the simplest such map for a CNN, with one backward pass: the size of the slope of a class score with respect to every pixel, which shows "which pixels need to be changed the least to affect the class score the most". Today you make that map for ResNet-18, and test what it promises.
    """)
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "change-answer",
        question="ResNet-18 calls the drawing of fruit a balloon, at 0.73. You change 5% of its pixels, 2,509 of the 50,176, each color by 1 on its scale of 0 to 255: too little to see. Can that change the answer?",
        options={
            "no": "No: a change of 1 in 255 is far too small to matter",
            "right": "Yes, if you pick the right pixels and change each color in the right direction",
            "any": "Yes, any 5% of the pixels would do",
        },
        reveal="**Yes, with the right pixels.** The 5% of the pixels with the largest slopes, each color changed by 1 against its slope, bring balloon down to 0.40, and ping-pong ball, at 0.44, comes first. 5% random pixels, changed in the same way, leave balloon first, at 0.63. The slopes come from one backward pass; their largest size per pixel, drawn as a picture, is today's map. The slider in *The idea* lets you change them by more.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    _frames = C.step_frames({
        "salient": "The 5% most salient pixels, changed by {step}",
        "random": "5% random pixels, changed by {step}",
        "alt": "Two groups of three bars. After a change of {step} on the 5% most salient pixels: {salient}. After a change of {step} on 5% random pixels: {random}.",
    }, L.lang)
    _steps = L.explore("steps", "change per color", lambda _v: (_frames[int(_v)], C.step_sentence(int(_v), L.lang)),
                       start=1, values=C.STEPS, role="",
                       task="Slide from 0 to 12: from which change on is balloon no longer the best guess, on the 5% most salient pixels? And on 5% random pixels?")
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A slope for every pixel.** The logit of a class $c$, its score $z_c$, is one number, computed from a photo $\mathbf{x}$ of 3 × 224 × 224 = 150,528 numbers. As in [Two parameters: the gradient](/course/open/math-gradient?lang=en), its gradient has a slope for each of them, the partial derivative $\partial z_c / \partial x_i$: how much the score changes when that one number changes a little. Close to the photo, the score is almost a straight line in every direction:

        $$z_c(\text{changed photo}) \approx z_c(\mathbf{x}) + \sum_i \frac{\partial z_c}{\partial x_i} \times (\text{change of } x_i)$$

        So a small change to the numbers with the largest slopes moves the score the most.

        **Autograd gives all slopes at once.** The photo is a tensor that PyTorch made without recording, so it is a leaf tensor, and `x.requires_grad = True` turns recording on for it, as `requires_grad=True` did in [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en). `resnet(x)[0, c]` is the logit of class `c` for photo 0 of the batch, one number, and `.backward()` from it fills `x.grad`, in the shape of `x`: (1, 3, 224, 224). That is one pass forward and one backward. The central difference of [Gradient checking](/course/open/nn-gradient-check?lang=en), one number at a time, would take two passes through the network for each of the 150,528 numbers: 301,056 passes. The weights of the course's ResNet-18 keep no slopes, so only `x` gets a `.grad`. Take it through the model outside `torch.no_grad()`: inside it, PyTorch records nothing.

        **The logit, not the probability.** The probability of a class can rise because other classes fall: softmax divides by all of them. Its slopes mix in every other class, so Simonyan and colleagues took the score itself.

        **A saliency map.** For a picture, one number per pixel: the largest size of its three slopes, one per color.

        $$M_{ij} = \max_k \left| \frac{\partial z_c}{\partial x_{kij}} \right|$$

        The size, without the sign, because a pixel that lowers the score when it gets lighter matters as much as one that raises it. That is the **saliency map**, 224 by 224: `np.abs(x.grad[0].numpy()).max(axis=0)`. `x.grad[0]` holds the slopes of photo 0, of shape (3, 224, 224); `.numpy()` makes them a NumPy array, as in [Try an image classifier](/course/open/cv-try-classifier?lang=en), and `.max(axis=0)` takes the largest along axis 0, the colors. A few pixels have much larger slopes than the rest, so a drawn map is white from the 1% largest on: `vmax=np.quantile(sal, 0.99)`, the quantile of [Describing data](/course/open/stats-describe?lang=en).

        **Does a map keep its promise?** Change only its brightest pixels, a little: each color by 1 on its scale of 0 to 255, against its slope, so down where the slope is positive and up where it is negative. If the map is right, the answer changes sooner than when as many random pixels change in the same way. The slider does that on the drawing of fruit, which ResNet-18 calls a balloon: with the 5% most salient pixels of balloon's map, 2,509 of the 50,176, or with 2,509 random ones, each color changed by as much as you choose.
        """),
        _steps,
        mo.md(r"""
        A change of 1 is too small to see. On the 5% most salient pixels, it brings balloon from 0.73 to 0.40, and ping-pong ball comes first, at 0.44. On 5% random pixels, changed in the same way, balloon stays first, at 0.63; only a change of 4 takes first place from it. The map keeps its promise: its brightest pixels change the answer sooner than random pixels. But the straight line holds only close to the photo. On the coffee photo, a change of 1 on the 5% most salient pixels of espresso's map lowers the logit of espresso by 4.11, where the straight line promises 7.26. A change of 2 brings the logit from 15.62 to 10.54, but a change of 8 only to 11.27: further away, the score bends.
        """),
        L.watch_out("A saliency map has no sign. Blanking a pixel, giving it the mean color of ImageNet, 0 after preparing, moves each of its colors to 0: some down, some up, and the map does not say whether that raises or lowers the score. Blank the 5% most salient pixels of the cat's map for Egyptian cat, and the logit of Egyptian cat does not fall: it rises, from 19.65 to 20.83. The slopes with their sign saw that coming: the straight line gives a rise of 1.66. Do not read a saliency map as the pixels a model needs."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **saliency map** | one number per pixel: how strongly a class score changes when that pixel changes a little, the largest size of its slopes over the colors | `np.abs(x.grad[0].numpy()).max(axis=0)` |
        | $M_{ij}$ | the saliency of the pixel in row $i$ and column $j$ | `sal[i, j]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To see which pixels change a classifier's answer:

    1. Prepare the photo as for the model, a tensor `x` of shape (1, 3, 224, 224), and turn recording on: `x.requires_grad = True`.
    2. Take it through the model, not inside `torch.no_grad()`, and take the logit of the class you want to explain: `model(x)[0, c]`, not its probability.
    3. Go back from that one number, `.backward()`: `x.grad` has a slope for every pixel and color. Before a second map of the same `x`, set `x.grad = None`: `.grad` adds up.
    4. The saliency map: per pixel the largest size of its slopes over the colors, `np.abs(x.grad[0].numpy()).max(axis=0)`, of shape (224, 224).
    5. Draw it, white from the 1% largest on. To test it, change its brightest pixels a little and see whether the answer changes sooner than with random pixels: the demo does that with your functions.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell below the download prepares the cat as in [Try an image classifier](/course/open/cv-try-classifier?lang=en), with recording on, and takes the logit of class 285, Egyptian cat, ResNet-18's best guess for it. It prints the score, goes back from it, and prints the shape of `x.grad`. Then it makes the map, and prints its shape, its largest number and where that is: `argmax` counts the pixels row by row, 224 to a row, so `// 224` gives the row and `% 224` the column. Last, it draws the map. The comments with `#` say which step of the recipe each part is. Run it as it is; then change `c` where the comment says, to 967, espresso, and run it again. `resnet` is ResNet-18, which the cell below loads once it is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download ResNet-18, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 is not on your computer yet. If you did [Try an image classifier](/course/open/cv-try-classifier?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once, 47 MB, from Hugging Face. The worked example, the maps of the three photos and the demo need it; the slider in *The idea* and the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading ResNet-18** · {done} of {total} MB",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 47 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*ResNet-18 is on your computer: 11.7 million parameters, with torchvision's weights, trained on ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_saliency(resnet):
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
    # 2. the logit of one class: 285 is Egyptian cat
    c = 285  # change: 967
    score = resnet(x)[0, c]
    print(score)
    # 3. back from that one number: a slope for every pixel and color
    score.backward()
    print(x.grad.shape)
    # 4. the map: per pixel the largest size of its slopes over the colors
    g = x.grad[0].numpy()
    sal = np.abs(g).max(axis=0)
    print(sal.shape, sal.max())
    top = int(sal.argmax())
    print(top // 224, top % 224)
    # 5. draw it: black is 0, white the 1% largest
    plt.imshow(sal, cmap="gray", vmin=0, vmax=np.quantile(sal, 0.99))
    plt.show()
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The score of Egyptian cat is 19.6482, and it prints with `grad_fn=<SelectBackward0>`: PyTorch recorded how it came from `x`. `x.grad` has the shape of `x`, (1, 3, 224, 224). The map has shape (224, 224), and its largest number, 0.2192, is at row 163 and column 109, left of the nose. With `c = 967`, espresso, the score is only 0.0781: ResNet-18 sees no espresso in the cat. Yet every pixel has a slope for espresso too, and the largest, 0.1059, is at row 174 and column 134, just above the pink nose.
    """)
    return


@app.cell(hide_code=True)
def text_before_maps():
    mo.md(r"""
    The three photos of [Try an image classifier](/course/open/cv-try-classifier?lang=en), each with its map for ResNet-18's best guess: Egyptian cat, espresso and balloon. Each map is white from its own 1% largest numbers on.
    """)
    return


@app.cell(hide_code=True)
def maps_cell(resnet):
    _maps = {_photo: C.course_saliency(resnet, C.prepared(_photo), C.TOP[_photo]) for _photo in C.TOP}
    mo.Html(C.maps_picture(_maps, {
        "names": {"cat": "the cat", "coffee": "the coffee", "fruit": "the drawing of fruit"},
        "map": "map of {label}",
        "alt": "Top row: the cat, the coffee and the drawing of fruit. Bottom row: their saliency maps, gray speckles on black, lighter around the cat's eyes and nose, in the coffee, and on the fruit.",
    }))
    return


@app.cell(hide_code=True)
def text_after_maps():
    mo.md(r"""
    All three maps are speckled: neighboring pixels can differ a lot, and a map is not a few bright spots: the 10% brightest numbers hold only 26% of the sum of the cat's map, 34% of the coffee's and 29% of the fruit's. But they do gather: on the cat around the eyes and the nose, on the coffee photo in the coffee, and on the drawing on the fruit, where the brightest pixel of all is the light spot on the orange. The fruit covers 32% of the square ResNet-18 sees, and holds 54% of the 5% brightest pixels of balloon's map; of orange's map, 75%. A saliency map is a first, noisy explanation: it shows where the score is sensitive, pixel by pixel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_brighter():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which pixel is brighter?
    """)
    return


@app.cell(hide_code=True)
def choice_brighter_cell():
    choice_brighter = L.choice(
        "brighter",
        question="In the slopes of a photo, pixel A has 0.02, 0.01 and -0.01 for red, green and blue; pixel B has -0.05, 0.00 and 0.03. Which is brighter in the saliency map, and with what number?",
        options={
            "a": "A, with 0.02: its largest slope",
            "right": "B, with 0.05: the largest of the sizes of its slopes",
            "sum": "B, with 0.08: the sizes of its three slopes added up",
        },
        correct="right",
        why={
            "a": "0.02 is the largest of A's slopes with their signs. The map takes the size first: B's red, -0.05, changes the score more than any color of A, only the other way, lighter red lowering it.",
            "sum": "0.05 + 0.00 + 0.03 = 0.08 is a map too, but not this lesson's: Simonyan and colleagues take the largest size of the three slopes of a pixel, 0.05.",
            "right": "Without the sign, because a pixel that lowers the score when it gets lighter matters as much as one that raises it. B's red changes the score the most, 0.05 per unit, against 0.02 for A's red.",
        },
    )
    choice_brighter
    return (choice_brighter,)


@app.cell(hide_code=True)
def text_exercise_saliency():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The saliency map

        Complete `saliency(model, x, label)`: the saliency map of the photo `x`, of shape (1, 3, H, W), for the class `label`, as a NumPy array of shape (H, W): per pixel, the largest size of the slopes of the logit of `label` over its 3 colors. The check uses a ResNet-18 with random weights and small photos, of 32 by 32 and 40 by 48, so it needs no download. It also asks for two maps of one photo, for two classes, one after the other: remember that `.grad` adds up. When it is right, it saves your function in your project, as `work/projects/cnn/saliency.py`.
        """),
        L.task("Make `saliency` give the map of a photo for a class."),
    ])
    return


@app.cell(hide_code=True)
def exercise_saliency():
    box_saliency = L.code_box("saliency", lines=8)
    box_saliency
    return (box_saliency,)


@app.cell(hide_code=True)
def check_saliency(box_saliency):
    result_saliency = L.check("saliency", box_saliency.value)
    result_saliency
    return (result_saliency,)


@app.cell(hide_code=True)
def hints_saliency():
    L.hints(
        "saliency",
        [
            "Turn recording on for the photo, `x.requires_grad = True`, and clear an old slope, `x.grad = None`. Then the logit: `model(x)` has shape (1, classes); which number is the logit of `label` for photo 0?",
            "`score.backward()` fills `x.grad`, of shape (1, 3, H, W). Take the slopes of photo 0, make them a NumPy array as in *The idea*, and take per pixel the largest size along axis 0, the colors.",
        ],
        why="One backward pass from one logit gives a slope for every pixel and color, because the photo is a leaf tensor that records. Without `x.grad = None`, a second call on the same photo adds its slopes to those of the first, as `.grad` does for the weights in a training step.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_salient():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The most salient pixels

        Complete `most_salient(sal, share)`: an array of True and False in the shape of the map, True for the share of the pixels with the highest saliency, such as the 5% the slider changed for a share of 0.05. `np.quantile` from [Describing data](/course/open/stats-describe?lang=en) finds the number that a share of the map is below. The check gives your function made-up maps. When it is right, it saves your function in your project, as `work/projects/cnn/salient.py`.
        """),
        L.task("Make `most_salient` pick the brightest share of a map."),
    ])
    return


@app.cell(hide_code=True)
def exercise_salient():
    box_salient = L.code_box("most-salient", lines=5)
    box_salient
    return (box_salient,)


@app.cell(hide_code=True)
def check_salient(box_salient):
    result_salient = L.check("most-salient", box_salient.value)
    result_salient
    return (result_salient,)


@app.cell(hide_code=True)
def hints_salient():
    L.hints(
        "most-salient",
        [
            "The brightest 5% are the pixels at or above the number that 95% of the map is below. Which quantile is that, for any share?",
            "`np.quantile(sal, q)` is the number that a share `q` of the map is below. A comparison of an array with one number gives True or False for every pixel: which comparison keeps the bright ones?",
        ],
        why="A share of the pixels, not a fixed number: maps differ in scale. `sal >= 0.1` picks none of the pixels of a ResNet-18 with random weights, whose slopes are ten-thousandths, and fewer than 1% of the trained one's: 0.27% of the cat's map, 0.82% of the coffee's and 0.12% of the fruit's. The quantile follows each map.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The button uses your project: your `saliency` makes the map of the photo you choose, for ResNet-18's best guess, and your `most_salient` picks its 5% brightest pixels. Then the course changes each color of those pixels by 1 against its slope, and as many random pixels in the same way, and says what that does to the best guess. It needs ResNet-18, from the worked example, and exercises 2 and 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"the cat": "cat", "the coffee": "coffee", "the drawing of fruit": "fruit"}, value="the cat", label="Photo", inline=True)
    look_button = mo.ui.run_button(label="Make the map with your functions")
    mo.vstack([photo_choice, look_button])
    return look_button, photo_choice


@app.cell(hide_code=True)
def demo_saliency(resnet, look_button, photo_choice, result_saliency, result_salient):
    resnet, result_saliency, result_salient  # run again when exercise 2 or 3 saves your module
    if not look_button.value:
        _out = mo.md("*Press the button to make the map.*")
    else:
        _result = C.demo(photo_choice.value)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `saliency` and your `most_salient` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "saliency": "*Your `saliency` in `work/projects/cnn/saliency.py` stops with an error, or does not give a map of 224 by 224. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "salient": "*Your `most_salient` in `work/projects/cnn/salient.py` stops with an error, or does not give True and False for every pixel. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
            }[_result])
        else:
            _out = mo.vstack([
                mo.Html(C.demo_picture(_result, {"mask": "your 5% most salient pixels", "map": "your saliency map",
                                                 "alt": "The photo with its 5% most salient pixels in color and the rest faded, and next to it the saliency map in gray."})),
                mo.md(C.demo_sentence(_result, photo_choice.value, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_alike():
    mo.md(r"""
    ### Exercise 4. Two cats, one map?
    """)
    return


@app.cell(hide_code=True)
def choice_alike_cell():
    choice_alike = L.choice(
        "alike",
        question="On the cat, 66% of the 5% brightest pixels of the map of Egyptian cat are also among the 5% brightest of the map of tiger cat. Of the map of espresso, 17%; two random draws of 5% share about 5%. What does that tell you?",
        options={
            "detector": "The map of Egyptian cat shows what makes the cat an Egyptian cat",
            "right": "Both cat classes are moved by much the same pixels: the map shows where a score is sensitive, not what tells the two apart",
            "noise": "The map of espresso is noise, because there is no espresso in the photo",
        },
        correct="right",
        why={
            "detector": "Then the map of tiger cat would mark other pixels, and two thirds of the brightest pixels are the same: the eyes and the nose move both cat classes. What separates Egyptian cat from tiger cat, the map of one class cannot show.",
            "noise": "17% is more than the 5% that two random draws share, and its brightest pixel lies by the nose too. Every pixel has a slope for every class: the map of espresso marks the pixels whose change moves the score of espresso the most, and some of them matter for a cat as well.",
            "right": "A saliency map answers one question: which pixels change this score the most, at this photo. For classes that look alike, those are much the same pixels. The next lesson makes a coarser map, from the last feature maps of ResNet-18 instead of from the pixels.",
        },
    )
    choice_alike
    return (choice_alike,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The gradient of [Two parameters: the gradient](/course/open/math-gradient?lang=en), now with respect to the 150,528 numbers of a photo instead of two parameters, and the autograd of [PyTorch autograd does what your Value did](/course/open/torch-autograd?lang=en), which gives all of them with one backward pass from one logit; its `.grad` adds up here too. Every pixel reaches the logit through the receptive fields of [Look inside a CNN](/course/open/cv-inside-cnn?lang=en), and the map shows which reach it most strongly, at this photo. Your project `cnn` now has `saliency.py` and `salient.py`. In the next lesson, *Grad-CAM: where a CNN looked*, you make a coarser map, from the last feature maps of ResNet-18.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_brighter, result_saliency, result_salient, choice_alike):
    L.wrap_up(choice_brighter, result_saliency, result_salient, choice_alike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
