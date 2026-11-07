# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-prepare-photo. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("cv-prepare-photo", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en) and about [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="A model was trained on a feature standardized with mean 50 and standard deviation 10. A new example has the value 70. Which number goes into the model?",
        options={
            "right": "2",
            "raw": "70",
            "own": "0",
            "seven": "7",
        },
        correct="right",
        why={
            "right": "(70 − 50) / 10 = 2: a new example is standardized with the mean and the standard deviation of the training data, the numbers the model learned its weights with.",
            "raw": "The model learned on standardized numbers, which lie around 0, and 70 is far outside them. Standardize with the numbers of the training data: (70 − 50) / 10.",
            "own": "One example on its own is its own mean: taking off its own mean always leaves 0, and then it says nothing any more. Use the mean and the standard deviation of the training data: (70 − 50) / 10 = 2.",
            "seven": "70 / 10 divides, but forgets to take off the mean first: (70 − 50) / 10 = 2.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def quick_check_size():
    quick_size = L.choice(
        "quick-size",
        question="`img` has shape `(400, 600, 3)`. Which shape does `np.array(Image.fromarray(img).resize((300, 200)))` have?",
        options={
            "right": "`(200, 300, 3)`",
            "swapped": "`(300, 200, 3)`",
            "same": "`(400, 600, 3)`",
            "flat": "`(300, 200)`",
        },
        correct="right",
        why={
            "right": "Pillow writes a size as (width, height), and NumPy a shape as (rows, columns): width 300 and height 200 give 200 rows of 300 columns, with the three colors kept.",
            "swapped": "That reads `(300, 200)` as rows and columns. Pillow writes the width first: 300 columns and 200 rows, shape `(200, 300, 3)`.",
            "same": "`resize` gives a new image of the size you ask for: 300 by 200, shape `(200, 300, 3)`.",
            "flat": "`resize` keeps the three numbers per pixel. And Pillow writes the width first: shape `(200, 300, 3)`.",
        },
    )
    quick_size
    return (quick_size,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A pretrained image model learned from more than a million photos, all prepared in the same way: the same size, cut from the middle, and every color scaled with the same numbers. That is the only kind of input it has ever seen. Give it a photo of another size, or with colors from 0 to 255, and it stops with an error, or gives an answer that is confidently wrong. So before a pretrained model looks at your photo, you prepare it the way its training photos were prepared. The model card or the code that comes with the model says how; for most image models trained on ImageNet, a collection of photos the next lesson comes back to, it is the recipe of this lesson.

    **An everyday example.** A recipe in grams, weighed out on a scale in ounces: every number is right on its own scale, and the cake still fails.
    """)
    return


@app.cell(hide_code=True)
def predict_squash():
    guess_squash = L.guess(
        "squash",
        question="The coffee photo is 600 pixels wide and 400 high, and the model wants 224 by 224. Pillow can squash it straight to that size, with `resize((224, 224))`. What happens to the round cup?",
        options={
            "round": "It stays round",
            "narrow": "It becomes narrower than it is high",
            "wide": "It becomes wider than it is high",
            "cut": "Half of it is cut off",
        },
        reveal="**It becomes narrower than it is high.** The width goes from 600 to 224, 0.37 times as much, and the height from 400 to 224, 0.56 times as much: everything is squeezed from the sides. Nothing is cut off, but every shape changes, and the model learned from photos in which cups are round. That is why the recipe below keeps the ratio, and cuts the middle out instead.",
    )
    guess_squash
    return (guess_squash,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Most image models trained on ImageNet expect a photo prepared in the same steps. First the shorter side becomes 256 pixels, and the other side grows or shrinks with it, so that nothing is squashed: the coffee photo of 600 by 400 becomes 384 by 256, since 600 × 256 / 400 = 384. Then a **center crop** cuts the middle square of 224 by 224 out of it. Of the 256 rows, 32 do not fit: 16 go above the square and 16 below, so it starts at row 16. Of the 384 columns, 160 do not fit, so it starts at column 80. What lies at the edges of the photo is lost; the photos the model learned from were cut in the same way, so the thing they show usually lies in the middle.

        Then the numbers. The colors are divided by 255, so that they run from 0 to 1, and then comes **per-channel normalization**: per color, take off the mean and divide by the standard deviation of ImageNet's training photos, mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` for red, green and blue. That is standardization, as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), with one mean and one standard deviation per color, and with the numbers of the training data, not those of the new photo. A pixel with ImageNet's average color becomes (0, 0, 0); a pixel that is redder than average gets a red above 0.

        Last, the shape. A NumPy image has shape `(rows, columns, colors)`, and PyTorch's image models take `(batch, colors, rows, columns)`: the colors first, and a batch in front. `t.permute(2, 0, 1)` puts the axes of a tensor in a new order: axis 2, the colors, first, then axis 0 and axis 1. `t.unsqueeze(0)` puts a new axis of length 1 in front: a batch of one photo, shape `(1, 3, 224, 224)`. And the model's weights are float32, so the photo is too.
        """),
        L.watch_out("`reshape(1, 3, 224, 224)` also gives the right shape, but not the right photo: it keeps the numbers in their order and only cuts them anew, so the red, green and blue of one pixel end up side by side in one row of the first color, as if they were three pixels. `permute` moves the axes themselves."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **center crop** | the middle square of an image, cut out with a slice | `small[16:240, 80:304]` |
        | **per-channel normalization** | per color, take off a mean and divide by a standard deviation: those of the training photos | `(crop / 255 - mean) / std` |
        | **`permute`** | puts the axes of a tensor in a new order | `t.permute(2, 0, 1)` |
        | **`unsqueeze`** | puts a new axis of length 1 at a place | `t.unsqueeze(0)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_crop():
    mo.vstack([
        mo.md("**Squashed, or the ratio kept and the middle cut out**"),
        mo.Html(C.picture({"squashed": "squashed to 224 × 224", "resized": "384 × 256", "crop": "the middle 224 × 224",
                           "alt": "Three versions of the coffee photo at the same scale. On the left, squashed to 224 by 224: the cup is narrow and tall. In the middle, at 384 by 256 with the ratio kept, with a wide darker strip on the left and on the right and a thin one at the top and the bottom. On the right, the middle square of 224 by 224: the cup round and almost whole, the saucer cut off at the sides."})),
        mo.md(r"""
        On the left, the photo squashed straight to 224 by 224: the cup is narrow and tall. In the middle, the photo at 384 by 256, with the ratio kept; the darker strips are what the center crop cuts away, wide on the left and the right, thin at the top and the bottom. On the right, the square the model gets: the cup round and almost whole, the saucer cut off at the sides.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Why 256 first, and then 224? With the slider you choose the shorter side of the first step; the center crop always cuts 224 by 224 out of the middle. On the left you see the photo at that size, with the part that is cut away darker; on the right what the model gets.
    """)
    return


@app.cell(hide_code=True)
def explore_side_cell():
    def _say(short):
        _w = C.size_after(600, 400, short)[0]
        _notes = {224: " Nothing above or below is lost, but on the left and the right a third of the photo is.",
                  256: " A thin strip goes at the top and the bottom, and a wide one at the sides: the recipe's choice. Only the very top of the rim is cut off.",
                  320: " A clear piece of the top of the rim is cut off.",
                  384: " The left side of the rim is cut off too.",
                  448: " The rim is cut off at the top and on both sides: only the middle of the cup is left."}
        return (f"The photo becomes {_w} by {short}; the square holds the middle {C.percent(100 * 224 / short, '.')} percent of its "
                f"height and {C.percent(100 * 224 / _w, '.')} percent of its width." + _notes.get(short, ""))

    def _frame(short):
        _html = C.side_frame(short, {"resized": "{w} × {h}", "crop": "the middle 224 × 224",
                                     "alt": "On the left, the coffee photo at {w} by {h} pixels, drawn at the same size at every stop, with everything outside its middle square of 224 by 224 darker. On the right, that middle square: what the model gets."})
        return _html, _say(short)

    explore_side = L.explore("side", "shorter side", _frame, start=256, values=C.SIDES, role="",
                             task="Find the shorter side at which the square holds the whole height of the photo, and the one at which the rim is cut off on both sides.")
    explore_side
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    At 224 the square holds the whole height of the photo; at 256 the middle 87.5 percent of it, and the cup comes a little closer. Why closer? While they learned, the ImageNet models saw random parts of their training photos, mostly smaller than the whole photo, blown up to 224 by 224: the things in them looked bigger than in a whole photo. The shorter side at 256 with the middle 224 kept zooms in a little in the same way. It is the habit of the first big ImageNet models, which learned from random squares of 224 cut out of photos of 256. The bigger the first step makes the photo, the closer the model looks: at 320 a clear piece of the top of the rim falls off, at 384 its left side too, and at 448 only the middle of the cup is left.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To make a photo `photo`, opened with Pillow, into the input of an ImageNet model:

    1. Make the shorter side 256, with the ratio kept: for a photo of width `w` and height `h` that is wider than high, `small = np.array(photo.resize((int(256 * w / h), 256), Image.BILINEAR))`; for a photo that is higher than wide, `(256, int(256 * h / w))`.
    2. Cut the middle square: `top = int((small.shape[0] - 224) / 2)` and `left = int((small.shape[1] - 224) / 2)`, then `crop = small[top:top + 224, left:left + 224]`.
    3. From 0 to 1: `crop / 255`.
    4. Per color, ImageNet's numbers: `x = (crop / 255 - mean) / std`, with `mean = np.array([0.485, 0.456, 0.406])` and `std = np.array([0.229, 0.224, 0.225])`.
    5. A float32 tensor, the colors first, a batch of one: `torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell opens the coffee photo and prints its size. It makes the shorter side 256 and cuts the middle square out, steps 1 and 2, and prints the shapes and where the square starts. It scales the colors, steps 3 and 4, and prints one pixel before and after, and the mean of every color over the whole square. Then it makes the tensor, step 5, and prints its shape and its dtype.
    """)
    return


@app.cell
def example_coffee():
    import numpy as np
    import torch
    from PIL import Image

    photo = Image.open("data/photos/coffee.png")
    print(photo.size)
    # 1. the shorter side 256, the ratio kept: 600 × 256 / 400 = 384
    small = np.array(photo.resize((384, 256), Image.BILINEAR))
    # 2. the middle 224 by 224
    top, left = int((256 - 224) / 2), int((384 - 224) / 2)
    crop = small[top:top + 224, left:left + 224]
    print(small.shape, crop.shape, top, left)
    # 3. and 4. from 0 to 1, then per color the mean and the standard deviation of ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (crop / 255 - mean) / std
    # .round(2): every number of the array to two decimals
    print(crop[0, 0], x[0, 0].round(2))
    # the mean over the rows, then over the columns: one per color
    print(x.mean(axis=0).mean(axis=0).round(2))
    # 5. a float32 tensor, the colors first, a batch of one
    t = torch.tensor(x).float().permute(2, 0, 1).unsqueeze(0)
    print(t.shape, t.dtype)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Pillow gives the size as (width, height), 600 by 400; NumPy the shape as (rows, columns, colors). After step 1 the photo has 256 rows of 384 columns, and the middle square starts at row 16 and column 80. Its top left pixel, (158, 63, 24), is a dark red-brown: its red comes out above ImageNet's average, 0.59, and its green and blue below it, −0.93 and −1.39. Over the whole square, the colors have mean 0.54, −0.64 and −0.95, not 0: the coffee photo is redder and less blue than the average ImageNet photo, and the scaling keeps that, because it uses ImageNet's numbers, not the photo's own. The tensor has the shape and the dtype the model takes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. A photo taken upright

    A photo is 300 pixels wide and 450 high: Pillow gives its size as `(300, 450)`.
    """)
    return


@app.cell(hide_code=True)
def choice_shape_cell():
    choice_shape = L.choice(
        "shape",
        question="Which shape does the NumPy array of this photo have after step 1?",
        options={
            "right": "`(384, 256, 3)`",
            "landscape": "`(256, 384, 3)`",
            "squashed": "`(256, 256, 3)`",
            "crop": "`(224, 224, 3)`",
        },
        correct="right",
        why={
            "right": "The shorter side is the width, 300: it becomes 256, and the height 450 × 256 / 300 = 384. Pillow's size is `(256, 384)`, the width first; NumPy's shape has the rows first: 384 rows of 256 columns, with three colors.",
            "landscape": "That is the shape of the coffee photo, which is wider than high. Here the width, 300, is the shorter side: it becomes 256, and the height 384. NumPy writes the rows first: `(384, 256, 3)`.",
            "squashed": "Step 1 keeps the ratio: only the shorter side becomes 256, and the other side grows with it, to 450 × 256 / 300 = 384.",
            "crop": "That is after step 2, the middle square. After step 1 the width is 256 and the height 450 × 256 / 300 = 384.",
        },
    )
    choice_shape
    return (choice_shape,)


@app.cell(hide_code=True)
def text_exercise_crop():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The middle square

        Complete `center_crop(img, size)`, which gives back the middle square of `size` by `size` of an image: of the rows that do not fit, half go above the square and half below, and the same goes for the columns. Replace the three `...`. The check calls it with a small array of 6 rows by 8 columns, which holds 0 to 47, row by row:

        ```python
        center_crop(np.arange(48).reshape(6, 8), 4)
        ```

        That should give `[[10, 11, 12, 13], [18, 19, 20, 21], [26, 27, 28, 29], [34, 35, 36, 37]]`. The check also tries an image that is higher than wide, images with three colors, and an image exactly as big as the square.
        """),
        L.task("Make `center_crop(img, size)` give back the middle square of the image."),
    ])
    return


@app.cell(hide_code=True)
def exercise_crop():
    crop_code = L.code_box("crop", lines=4)
    crop_code
    return (crop_code,)


@app.cell(hide_code=True)
def check_crop(crop_code):
    result_crop = L.check("crop", crop_code.value)
    result_crop
    return (result_crop,)


@app.cell(hide_code=True)
def hints_crop():
    L.hints(
        "crop",
        [
            "Step 2 of the recipe: first where the square starts, then the slice.",
            "`img.shape[0]` is the number of rows and `img.shape[1]` the number of columns. Of the rows, `img.shape[0] - size` do not fit; half of them go above the square, made whole with `int`.",
            "`top = int((img.shape[0] - ___) / 2)`, `left = int((img.shape[___] - size) / 2)` and `return img[top:top + ___, left:left + size]`.",
        ],
        why="Of the 6 rows, 2 do not fit, so the square starts at row `int(2 / 2)`, 1; of the 8 columns, 4 do not fit, so it starts at column 2. `img[1:5, 2:6]` holds the rows 1 to 4 and the columns 2 to 5. For an image with colors, the slice keeps the third axis as it is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_prepare():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The cat, ready for the model

        `data/photos/chelsea.png` is a photo of a cat, 451 by 300 pixels. The box cuts an upright piece out of it, 200 pixels wide and 300 high, as if the photo was taken upright, and does step 1 already: `small` has its shorter side at 256, shape `(384, 256, 3)`, as in exercise 1. It also gives you `mean` and `std`, the numbers of ImageNet. Make `crop` the middle square of `small`, step 2, and `x` the input of the model, with steps 3 to 5: a float32 tensor of shape `(1, 3, 224, 224)`.
        """),
        L.task("Make `crop` the middle square, and `x` the cat as the model takes it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prepare():
    prepare_code = L.code_box("prepare", lines=12)
    prepare_code
    return (prepare_code,)


@app.cell(hide_code=True)
def check_prepare(prepare_code):
    result_prepare = L.check("prepare", prepare_code.value)
    result_prepare
    return (result_prepare,)


@app.cell(hide_code=True)
def hints_prepare():
    L.hints(
        "prepare",
        [
            "Steps 2 to 5 of the recipe. The photo is upright: where are the rows and the columns that do not fit?",
            "Of the 384 rows of `small`, 160 do not fit, and of its 256 columns 32: half of each go before the square. Then the numbers: `crop / 255`, minus `mean`, divided by `std`. Then a tensor of it, float32 with `.float()`, the colors first with `.permute` and a batch axis in front with `.unsqueeze`.",
            "`crop = small[80:___, ___:240]` and `x = torch.tensor((crop / 255 - ___) / ___).float().permute(2, ___, ___).unsqueeze(___)`.",
        ],
        why="The photo is upright, so the wide margin is in the rows: `int((384 - 224) / 2)` is 80 and `int((256 - 224) / 2)` is 16, so `crop = small[80:304, 16:240]`. `(crop / 255 - mean) / std` scales every color with ImageNet's numbers: NumPy stretches `mean` and `std`, of shape `(3,)`, over the rows and columns. `torch.tensor` makes a tensor of it, float64 because it comes from NumPy, and `.float()` makes it float32. `.permute(2, 0, 1)` gives shape `(3, 224, 224)`, and `.unsqueeze(0)` shape `(1, 3, 224, 224)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_own():
    mo.md(r"""
    ### Exercise 4. Its own numbers

    Someone scales the cat photo with its own mean and standard deviation per color, instead of those of ImageNet. Every color of the photo then has mean 0 and standard deviation 1.
    """)
    return


@app.cell(hide_code=True)
def explain_own_cell():
    explain_own = L.explain(
        "own",
        question="Why is that worse for the pretrained model? Think of what the model learned its weights on, and of what happens to a photo that is redder than most.",
        model_answer="**The model learned its weights on photos scaled with ImageNet's numbers,** so a new photo must be scaled with the same numbers, as a new example in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en). **With its own numbers, every photo gets mean 0 in every color:** a photo that is redder, darker or bluer than most no longer looks that way. The coffee photo had a red of 0.54 and a blue of −0.95 on average; with its own numbers both become 0, and the model no longer sees that the photo is red-brown. **The same color then gets other numbers in different photos,** depending on what else is in the photo, so the model cannot count on a color meaning what it meant while it learned.",
        points=[
            "you say that the model learned on photos scaled with ImageNet's numbers, so a new photo needs the same numbers",
            "you say that with its own numbers every photo gets mean 0 per color, so a redder or darker photo no longer looks that way",
            "you say that the same color then gets other numbers in different photos",
        ],
    )
    explain_own
    return (explain_own,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** A pretrained model knows only the inputs it learned from, so a new photo is prepared the way its training photos were: the shorter side resized with the ratio kept, as in [Bigger and smaller: interpolation](/course/open/cv-resize-interpolation?lang=en), the middle square cut out, every color standardized with the numbers of the training photos, as in [Why gradient descent likes scaled features](/course/open/ml-feature-scaling?lang=en), and the whole as a float32 tensor of shape `(1, 3, 224, 224)`, ready for a model as in [Using a pretrained model](/course/open/torch-pretrained?lang=en). Other models want other numbers, such as 384 by 384 pixels, or a mean of 0.5 for every color: the model card says which, and the same steps do it with those numbers. In the next lesson, *Try an image classifier: top-1 and top-5*, a ResNet-18 trained on ImageNet looks at the coffee photo and the cat, prepared like this.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_shape, explain_own, result_crop, result_prepare):
    L.wrap_up(choice_shape, result_crop, result_prepare, explain_own)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
