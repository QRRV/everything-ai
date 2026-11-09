# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-inside-cnn. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-inside-cnn", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en) and [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_filter():
    quick_filter = L.choice(
        "quick-filter",
        question="Your ColorCNN starts with `nn.Conv2d(3, 8, 3, padding=1)`. How many weights does one of its 8 filters have?",
        options={
            "nine": "9: a 3 by 3",
            "right": "27: a 3 by 3 for each of the 3 colors",
            "all": "72: 8 times a 3 by 3",
        },
        correct="right",
        why={
            "nine": "A filter of a color photo looks at all 3 colors at once: a 3 by 3 for each, 3 × 3 × 3 = 27 weights. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "all": "8 × 9 counts the 8 filters with one 3 by 3 each. One filter has a 3 by 3 for each of the 3 colors, 27 weights, and the layer 8 × 27 = 216. See [Project: your CNN on color photos](/course/open/cv-color-cnn?lang=en).",
            "right": "Today's filters, those of the first layer of ResNet-18, are 7 by 7: 3 × 7 × 7 = 147 weights each, enough to draw as a small color image.",
        },
    )
    quick_filter
    return (quick_filter,)


@app.cell(hide_code=True)
def quick_check_weights():
    quick_weights = L.choice(
        "quick-weights",
        question="ResNet-18 put espresso first for the coffee photo, at 0.85. Where does it keep what espresso looks like?",
        options={
            "photos": "In espresso photos from ImageNet that it compares the photo with",
            "right": "In its 11.7 million weights, learned from ImageNet's photos",
            "names": "In the file with the names of the 1,000 classes",
        },
        correct="right",
        why={
            "photos": "ResNet-18's file of 47 MB holds its weights and no photos. Training changed those weights until they gave the right logits for ImageNet's photos. See [Try an image classifier](/course/open/cv-try-classifier?lang=en).",
            "names": "That file only gives each of the 1,000 logits its name, in order. What the classes look like is in the weights, learned from ImageNet's photos. See [Try an image classifier](/course/open/cv-try-classifier?lang=en).",
            "right": "Today you look at some of those weights, and at what they compute for a photo.",
        },
    )
    quick_weights
    return (quick_weights,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    ResNet-18 has a top-1 accuracy of about 70% on ImageNet's validation photos, but its 11.7 million weights do not say what it looks for. You can look. The filters of its first layer are small images, and for any photo, every layer gives feature maps that show where in the photo it found what it looks for. Zeiler and Fergus (2014) drew the first filters of AlexNet this way, and traced what its deeper layers answer to back onto the photo. They saw that some of its first filters did nothing, that others held only very fine and very coarse patterns, and that its stride of 4 left a mess in the second layer. With smaller filters, 7 by 7 instead of 11 by 11, and a stride of 2, their network got better. The first layer of ResNet-18 has that size. Today you draw its filters and its feature maps, and see how much of a photo one number deep inside looks at.
    """)
    return


@app.cell(hide_code=True)
def predict_filters():
    guess_filters = L.guess(
        "filters-look",
        question="The first layer of ResNet-18 has 64 filters of 7 by 7, for the 3 colors. Drawn as small color images, with gray for a weight of 0, what do they look like?",
        options={
            "noise": "Noise: training leaves no pattern you can see",
            "right": "Mostly edges and lines at every angle, and spots of one color",
            "things": "Small pieces of the things in ImageNet: eyes, wheels, letters",
        },
        reveal="**Mostly edges and lines, and spots of color.** Light next to dark at every angle and at several widths; a red spot, a green one, a blue one. The first layer of AlexNet, in 2012, looked much the same. Pieces of things, such as eyes, come deeper: exercise 4 is about a map of `layer3` that lights up around the cat's eyes, and the demo shows it. And 8 of the 64 turn out to do nothing at all: the worked example shows one.",
    )
    guess_filters
    return (guess_filters,)


@app.cell(hide_code=True)
def text_idea():
    _fields = C.field_frames({"alt": "The cat photo with a square around the middle of the eye on the left for each stage, the square after {stage} bold, and that eye in a dashed box."})
    _field = L.explore("field", "receptive field", lambda _v: (_fields[int(_v)], C.field_sentence(int(_v), L.lang)),
                       start=7, values=list(C.FIELD_VIEWS), role="",
                       task="Slide from 7 to 435: from which stage on does one number see the whole eye on the left, the dashed box?")
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A filter is a small image.** The weights of the first layer of ResNet-18, `conv1.weight`, have shape (64, 3, 7, 7): 64 filters, each a 7 by 7 for each of the 3 colors, 147 weights. Each number of a feature map is a sum of weight times pixel over a 7 by 7 piece of the photo, in all 3 colors, as in [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en). It is largest where the photo looks like the filter: light where the weights are positive, dark where they are negative. So a filter drawn as an image shows what it looks for. Its weights lie around 0, positive and negative, so they need a scale: divide by twice the largest weight of the filter, + or -, and add 0.5. A weight of 0 is then 0.5, gray; the largest weight becomes 0 or 1; a positive weight is lighter in its own color, a negative one darker. `w.abs()` makes every number positive, as `np.abs` does. A filter has its colors first, (3, 7, 7), as a network wants them, and `plt.imshow` wants rows, columns and colors: `.permute(1, 2, 0)` puts axis 1 first, then axis 2, then axis 0, the step of [Get a photo ready for a model](/course/open/cv-prepare-photo?lang=en) the other way round. The 64 filters side by side are a **filter grid**.

        **Feature maps, stage by stage.** ResNet-18 takes a photo through stages. First `conv1`, 64 filters of 7 by 7 with stride 2, then `bn1`, batch normalization, and a ReLU: 64 feature maps of 112 by 112 for a photo of 224 by 224. Then max pooling of 3 by 3 with stride 2 and padding 1 halves the rows and the columns, and four stages follow, `layer1` to `layer4`, each of two residual blocks as in [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en). `layer1` gives 64 maps of 56 by 56. `layer2`, `layer3` and `layer4` each start with a stride of 2, so each halves the rows and the columns of the maps it gets, and has twice as many channels. A map can be drawn as a gray image: black for 0, white for its largest number. The ReLU makes every negative number 0: on the cat, 29% of the numbers after `conv1` are 0. The maps of a stage side by side are a **feature-map grid**, the strongest first: those with the highest mean.

        **How much of the photo one number sees.** As in [Convolution in detail: kernel size, stride and padding](/course/open/cnn-conv-formal?lang=en), every number of a map depends on a piece of the photo, its receptive field, and every layer makes that piece larger: a 3 by 3 adds 2 times the distance between neighboring numbers. After `conv1`, 7 by 7 pixels; after the pooling, 11; after `layer1`, 43; after `layer2`, 99; after `layer3`, 211; after `layer4`, 435: more than the whole photo. The slider shows that piece for the number of each map nearest the middle of the eye on the left: bold for the stage you choose, thin for the others.
        """),
        _field,
        mo.md(r"""
        From `layer2` on, one number sees a whole eye. Deeper, a number sees more, so it can answer to more than an edge: to a part, such as an eye, and then to a whole thing. The pixels near the middle of that piece count most. Ask autograd how strongly each pixel moves channel 120 of `layer3` at row 7 and column 7, a number whose 211 by 211 pixels all lie on the cat: a third of it comes from the middle 49 by 49, 5% of the piece, and 82% from the middle 97 by 97, 21% of the piece. That is why a map of `layer3` still shows where in the photo something is.
        """),
        L.watch_out("Scaling each filter by its own largest weight shows the pattern of every filter, also of one that does nothing. 8 of the 64 filters of ResNet-18 have weights below 0.00001, against largest weights of 0.06 to 1.02 for the others, and `bn1` gives their maps a weight of almost 0 too: on every photo, their maps are 0. Before you say what a filter looks for, look at the size of its weights, `w.abs().max()`, and at its map."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **filter grid** | the filters of a layer side by side, each drawn as a small image | |
        | **feature-map grid** | the feature maps of a stage side by side, each drawn as a gray image | |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To look inside a trained CNN:

    1. Its first filters: `model.conv1.weight`, of shape (64, 3, 7, 7) in ResNet-18. Draw each with gray for 0: `0.5 + w / (2 * w.abs().max())`, then `.permute(1, 2, 0)`.
    2. Check the size of its weights, `w.abs().max()`: a filter whose weights all lie close to 0 does nothing, however clear its picture looks.
    3. Prepare a photo as for the model, and take it through the model one stage at a time, inside `with torch.no_grad():`, keeping what each stage gives: its feature maps.
    4. Draw a map as a gray image, from black, 0, to white, its largest number: `cmap="gray", vmin=0`.
    5. Find where a map is highest, and the piece of the photo that number sees: its receptive field. Look at more photos before you say what a filter or a channel looks for.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell takes the weights of `conv1` and prints their shape, makes filter `k` an image, and prints its shape and its darkest and its lightest number. Then it prints the largest weight of the filter, + or -, and draws it. The second cell prepares the cat as in [Try an image classifier](/course/open/cv-try-classifier?lang=en), takes it through the first stage, `conv1`, `bn1` and a ReLU, and prints the shape of the maps and the share of their numbers that is 0. Then it draws the map of filter `k`: `vmin=0` makes 0 black, where `plt.imshow` would otherwise make the smallest number of the map black. Last, it prints where that map is highest: `argmax` counts the numbers of the map row by row, 112 to a row, so `// 112` gives the row and `% 112` the column. The comments with `#` say which step of the recipe each part is. Run them as they are, with filter 26; then change `k` in the first cell where the comment says, and run it again: the second cell then runs again by itself, because it uses `k`. `resnet` is ResNet-18, which the cell below loads once it is downloaded.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download ResNet-18, 47 MB")
    return (get_button,)


@app.cell(hide_code=True)
def resnet_cell(get_button):
    resnet = C.resnet_or_stop(get_button, {
        "ask": "*ResNet-18 is not on your computer yet. If you did [Try an image classifier](/course/open/cv-try-classifier?lang=en) on this computer, it is: then this lesson finds it. Otherwise, the button downloads it once, 47 MB, from Hugging Face. The worked example, the slider over the filters and the demo need it; the exercises do not. While it downloads, the whole lesson waits: on a slow connection, do the exercises first.*",
        "busy": "**Downloading ResNet-18** · {done} of {total} MB",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 47 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*ResNet-18 is on your computer: 11.7 million parameters, with torchvision's weights, trained on ImageNet.*",
    })
    return (resnet,)


@app.cell
def example_filters(resnet):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    # 1. the filters of the first layer: 64, each a 7 by 7 for each of the 3 colors;
    #    filter k as an image: gray for a weight of 0, the colors last
    w = resnet.conv1.weight
    print(w.shape)
    k = 26  # change: 9
    f = w[k]
    img = (0.5 + f / (2 * f.abs().max())).permute(1, 2, 0)
    print(img.shape, img.min(), img.max())
    # 2. the size of its weights: the largest, + or -
    print(f.abs().max())
    plt.imshow(img)
    plt.show()
    return Image, k, np, plt, torch


@app.cell
def example_maps(Image, k, np, plt, resnet, torch):
    # 3. the cat, prepared as in Try an image classifier, through the first stage: conv1, bn1 and a ReLU
    photo = Image.open("data/photos/chelsea.png")
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    crop = small[16:240, 80:304]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.tensor((crop / 255 - mean) / std).float().permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        maps = torch.relu(resnet.bn1(resnet.conv1(x)))
    print(maps.shape, (maps == 0).float().mean())
    # 4. the map of filter k on the cat: black is 0, white its largest number
    plt.imshow(maps[0, k], cmap="gray", vmin=0)
    plt.show()
    # 5. where it is highest: argmax counts row by row, 112 numbers to a row
    top = int(maps[0, k].argmax())
    print(top // 112, top % 112)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The weights of `conv1` have shape (64, 3, 7, 7). Filter 26 as an image has shape (7, 7, 3), and its numbers go from 0.0094 to 1: its largest weight, 0.2763, is positive, so it becomes 1. Filter 26 is dark above and light below: a horizontal edge. The maps have shape (1, 64, 112, 112), and 29% of their numbers are 0, `tensor(0.2935)`. The map of filter 26 is highest at row 62 and column 96. Neighboring numbers of a map of `conv1` are 2 pixels of the photo apart, so that number sees the 7 by 7 pixels around row 124 and column 192 of the photo: just under the eye on the right, where the dark eye meets the lighter fur below it. With `k = 9`, the filter looks like a blue patch. But its largest weight is `1.0849e-06`, 1.0849 times 10 to the power -6: 0.0000011. Its map is black, 0 everywhere, so its highest place is simply its first number: row 0, column 0. Filter 9 is one of the 8 that do nothing; the scaling only made its tiny weights visible.
    """)
    return


@app.cell(hide_code=True)
def text_before_play():
    mo.md(r"""
    Which filter looks for what? The grid shows the 64 filters of `conv1`, 8 to a row: the number on the left of a row is its first filter, and the number above a column says how many to count on, so filter 26 is in row 24, column 2. The slider goes through them: on the left, the filter; on the right, its map on the cat, from black, 0, to white, its largest number.
    """)
    return


@app.cell(hide_code=True)
def grid_cell(resnet):
    mo.Html(C.grid_picture(C.filter_images(resnet.conv1.weight), {
        "title": "The 64 filters of conv1",
        "alt": "Sixty-four small color squares in 8 rows of 8: most show light and dark stripes or edges at different angles, some a spot of one color, such as red, green, blue or magenta, and some an even patch of color.",
    }))
    return


@app.cell(hide_code=True)
def explore_filters_cell(resnet):
    _maps = C.course_maps("cat")[0]
    _frames = C.filter_frames(resnet.conv1.weight, _maps, {"filter": "filter {k}", "map": "its map on the cat",
                                                          "alt": "Filter {k} of conv1 as a small color image, and next to it its feature map on the cat photo, in gray."})

    def _frame(_k):
        _k = int(_k)
        return _frames[_k], C.filter_sentence(_maps, resnet.conv1.weight, _k, L.lang)

    explore_filters = L.explore("filter", "filter", _frame, start=26, values=list(range(64)), role="",
                                task="Find a filter whose map is lightest on the cat's pink nose, and one whose map is 0 everywhere although its largest weight is not tiny.")
    explore_filters
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Filter 28, a red spot, is highest on the pink nose, at row 93 and column 73. But the orange fur lights up almost all of its map too: no number of it is 0. That makes its map the strongest of the 64 on the cat, and on the coffee and the fruit photos too. The maps of filters 17 and 18 are 0 everywhere on the cat, while their largest weights are 0.08 and 0.28: `bn1` shifts their maps down, so that only a piece of a photo that looks much like them gets above 0. Filter 17 is a light spot on red, and the cat has none; on the coffee and the fruit photos, its map is not 0. Most filters are edges at some angle. Filters 26 and 57 are each other's opposites, dark above light and light above dark: 26 is highest under the eye on the right, 57 on the upper edge of the eye on the left.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The shape after layer3
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "inside-shape",
        question="The cat photo, of shape (1, 3, 224, 224), goes through ResNet-18. After `conv1`, `bn1` and the ReLU it is (1, 64, 112, 112); after the pooling and `layer1`, (1, 64, 56, 56). Which shape does `layer3` give?",
        options={
            "wide": "(1, 256, 56, 56)",
            "right": "(1, 256, 14, 14)",
            "two": "(1, 128, 28, 28)",
        },
        correct="right",
        why={
            "wide": "The channels are right: 64, 128, 256. But `layer2` and `layer3` each start with a stride of 2, so each halves the rows and the columns: 56, 28, 14.",
            "two": "That is what `layer2` gives: once twice as many channels and half the rows and columns. `layer3` does that once more: (1, 256, 14, 14).",
            "right": "Two stages, each with twice as many channels and half the rows and columns: 64 → 128 → 256, and 56 → 28 → 14. `layer4` then gives (1, 512, 7, 7), and the mean of each of its 512 maps is what ResNet-18 gives its last layer.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_filter():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A filter as an image

        Complete `filter_image(w)`: from the weights of one filter, of shape (3, 7, 7), make an image of shape (7, 7, 3) with numbers from 0 to 1, with gray, 0.5, for a weight of 0, and 0 or 1 for the largest weight of the filter, + or -. The check gives your function made-up filters, also one of another size and one whose largest weight is negative. When it is right, it saves your function in your project, as `work/projects/cnn/filters.py`.
        """),
        L.task("Make `filter_image` turn the weights of a filter into an image with gray for 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_filter():
    box_filter = L.code_box("filter-image", lines=7)
    box_filter
    return (box_filter,)


@app.cell(hide_code=True)
def check_filter(box_filter):
    result_filter = L.check("filter-image", box_filter.value)
    result_filter
    return (result_filter,)


@app.cell(hide_code=True)
def hints_filter():
    L.hints(
        "filter-image",
        [
            "First the largest weight, + or -: `w.abs().max()`. Call it `m`. Then `w / m` goes from -1 to 1 at most; how do you make that go from 0 to 1, with 0 in the middle?",
            "`w / (2 * m)` goes from -0.5 to 0.5; add 0.5. Then put the colors last, with `.permute`: the new order of the axes, the old axis 1 first.",
        ],
        why="Gray for 0 keeps the sign of every weight in sight: lighter is a weight that adds its pixel, darker one that takes it away. Dividing by the largest weight of the filter itself shows the pattern of every filter, also of a weak one; that is why you look at `w.abs().max()` too. With one scale for the whole layer, its largest weight, the 8 filters that do nothing would be plain gray.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_maps():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The feature maps of every stage

        Complete `feature_maps(model, x)`: take the photos `x` through a ResNet-18 `model` one stage at a time, as *The idea* says, and give back a list of 5 tensors: what `conv1`, `bn1` and the ReLU give, before the pooling, and what each of `layer1`, `layer2`, `layer3` and `layer4` gives. The course's ResNet-18 calls its ReLU and its pooling as functions; `torch.relu` and `nn.MaxPool2d(3, stride=2, padding=1)` do the same. The check uses a ResNet-18 with random weights and photos of 64 by 64, so it needs no download, and compares your maps with the course's. When it is right, it saves your function in your project, as `work/projects/cnn/maps.py`.
        """),
        L.task("Make `feature_maps` give the maps of all five stages."),
    ])
    return


@app.cell(hide_code=True)
def exercise_maps():
    box_maps = L.code_box("feature-maps", lines=12)
    box_maps
    return (box_maps,)


@app.cell(hide_code=True)
def check_maps(box_maps):
    result_maps = L.check("feature-maps", box_maps.value)
    result_maps
    return (result_maps,)


@app.cell(hide_code=True)
def hints_maps():
    L.hints(
        "feature-maps",
        [
            "Start as the worked example's first stage: `x = torch.relu(model.bn1(model.conv1(x)))`, and keep it in a list, `maps = [x]`. Then pool it: `x = nn.MaxPool2d(3, stride=2, padding=1)(x)`.",
            "Then a loop over `[model.layer1, model.layer2, model.layer3, model.layer4]`: each layer gets what the one before it gave, `x = layer(x)`, and you keep each `x` with `maps.append(x)`.",
        ],
        why="A forward hook, from [Looking inside a training run](/course/open/nn-activation-stats?lang=en), can keep what each `layer1` to `layer4` gives without writing the stages yourself; the first stage then needs a hook on `bn1` and a ReLU after it, because the ReLU is not a module here.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The button uses your project: your `filter_image` draws the 64 filters of `conv1`, and your `feature_maps` takes the photo you choose through the stages of ResNet-18. Then a slider goes through the stages. On the left, the photo with the square of pixels that the strongest number of that stage sees; on the right, the 8 strongest maps, the strongest first, with a frame around its largest number. It needs ResNet-18, from the worked example, and exercises 2 and 3.
    """)
    return


@app.cell(hide_code=True)
def photo_choice_cell():
    photo_choice = mo.ui.radio(options={"the cat": "cat", "the coffee": "coffee", "the fruit": "fruit"}, value="the cat", label="Photo", inline=True)
    look_button = mo.ui.run_button(label="Look inside ResNet-18 with your functions")
    mo.vstack([photo_choice, look_button])
    return look_button, photo_choice


@app.cell(hide_code=True)
def demo_inside(resnet, look_button, photo_choice, result_filter, result_maps):
    resnet, result_filter, result_maps  # run again when exercise 2 or 3 saves your module
    if not look_button.value:
        _out = mo.md("*Press the button to look inside.*")
    else:
        _result = C.demo(photo_choice.value)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `filter_image` and your `feature_maps` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _out = mo.md({
                "filters": "*Your `filter_image` in `work/projects/cnn/filters.py` stops with an error, or does not give an image of shape (7, 7, 3). Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "maps": "*Your `feature_maps` in `work/projects/cnn/maps.py` stops with an error, or does not give five maps of the right shapes. Did you change its file? Delete it, and check exercise 3 again: its check then saves your code again.*",
            }[_result])
        else:
            _maps = _result["maps"]
            _frames = C.stage_frames(_maps, _result["photo"], {"photo": "what one number sees", "map": "channel {k}",
                                                              "alt": "The photo with a square around the pixels one number of {stage} sees, and the 8 strongest feature maps of {stage}, in gray."})

            def _frame(_i):
                return _frames[int(_i)], C.stage_sentence(_maps, int(_i), L.lang)

            _out = mo.vstack([
                mo.Html(C.grid_picture(_result["images"], {"title": "Drawn with your filter_image",
                                                           "alt": "The 64 filters of conv1 in 8 rows of 8, drawn with your filter_image."})),
                L.explore("stage", "stage", _frame, start=0, values=list(range(5)), role="",
                          task="Slide from stage 0, conv1, to stage 4, layer4: from which stage on does one number see the whole photo?"),
                mo.md("Stage 0 is `conv1`, with `bn1` and the ReLU; stages 1 to 4 are `layer1` to `layer4`."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_eyes():
    mo.md(r"""
    ### Exercise 4. What a channel looks for
    """)
    return


@app.cell(hide_code=True)
def choice_eyes_cell():
    choice_eyes = L.choice(
        "eyes",
        question="On the cat, the strongest map of `layer3` is channel 120, and it is high around both eyes. On the coffee photo, channel 120 is 154th of the 256 by mean, and on the fruit photo 63rd. What can you say?",
        options={
            "detector": "Channel 120 finds cat eyes",
            "right": "Channel 120 answers to something around the cat's eyes; what, more photos must tell",
            "useless": "Channel 120 is no use for photos without a cat",
        },
        correct="right",
        why={
            "detector": "Three photos are too few to say that. It may answer to round shapes, to a dark spot in a light ring, or to the eyes of any animal. To know, look at the photos, out of many, where it is strongest, as Zeiler and Fergus did.",
            "useless": "The last layer weighs the means of all 512 maps of `layer4`, which are made from all 256 of `layer3`. A channel that is weak on a photo also tells the next layers something: that what it answers to is not there.",
            "right": "One photo shows where a channel is high, not what it answers to. Zeiler and Fergus took the 9 pieces of photos, out of the 50,000 of ImageNet's validation set, that make a channel strongest; for one channel, they were all faces of dogs.",
        },
    )
    choice_eyes
    return (choice_eyes,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The filters and feature maps of [Convolution: one small filter slides over the image](/course/open/cnn-convolution-idea?lang=en), now of a trained network: the filters of the first layer of ResNet-18 are edges and spots of color, and its deeper maps answer to larger pieces of a photo, as the receptive field of [Convolution in detail](/course/open/cnn-conv-formal?lang=en) grows. You took a photo prepared as in [Try an image classifier](/course/open/cv-try-classifier?lang=en) through the stages of [ResNet](/course/open/cnn-resnet?lang=en), one at a time. Your project `cnn` now has `filters.py` and `maps.py`. In the next lesson, *ImageNet and AlexNet: why 2012 changed everything*, you go back to the CNN whose filters Zeiler and Fergus looked at.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, result_filter, result_maps, choice_eyes):
    L.wrap_up(choice_shape, result_filter, result_maps, choice_eyes)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
