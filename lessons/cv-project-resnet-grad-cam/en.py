# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson cv-project-resnet-grad-cam. Route: course/route/lessen/beeld.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("cv-project-resnet-grad-cam", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions, about [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en) and about [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_layer():
    quick_layer = L.choice(
        "quick-layer",
        question="On the collage of the cat and the coffee, from which layer of ResNet-18 did Grad-CAM still tell the two classes apart?",
        options={
            "right": "`layer4`, the last convolutional layer: its maps come right before global average pooling and the logits",
            "finer": "`layer3`: its maps have 14 by 14 places instead of 7 by 7, so they show more",
            "first": "`conv1`, the first layer: it sees the photo itself",
            "any": "Every layer: Grad-CAM gives the same map from each of them",
        },
        correct="right",
        why={
            "right": "The maps of the last convolutional layer are what the logits are made from, and they still have places. Today you take Grad-CAM from the last block of your own ResNet: 7 by 7 places for an image of 28 by 28.",
            "finer": "They are finer, but on the collage, the maps from `layer3` no longer told the cat and the coffee apart: that layer still looks at parts, not yet at classes. See [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en).",
            "first": "The first layer looks for edges and colors, everywhere in the photo, for every class alike. Grad-CAM tells classes apart where the maps already stand for classes: in the last convolutional layer.",
            "any": "Grad-CAM works for any layer, but it does not give the same map: from `layer3`, the maps no longer told the cat and the coffee apart. See [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en).",
        },
    )
    quick_layer
    return (quick_layer,)


@app.cell(hide_code=True)
def quick_check_upside():
    quick_upside = L.choice(
        "quick-upside",
        question="Your ColorCNN learned from photos turned upside down instead of mirrored, and read 479 to 492 of the 1,000 test photos right, against 541 to 591 with mirror images. Why did upside down do worse?",
        options={
            "right": "Real photos hardly ever show things upside down, so the network learned from photos unlike the ones it is tested on",
            "pixels": "Upside down moves more pixels than a mirror image",
            "fewer": "Upside down gives fewer new photos than a mirror image",
            "colors": "Upside down puts the colors of a photo in another order",
        },
        correct="right",
        why={
            "right": "A change helps when the label still holds and the photo could be real: a mirrored car is a car on a road, an upside-down car is a photo no test set has. Today you ask the same of clothes: does a mirrored shoe look like the shoes in the test images?",
            "pixels": "Both put every pixel somewhere else: a mirror image the columns in the other order, upside down the rows. What counts is whether the photo it makes could be real.",
            "fewer": "Both make exactly one other version of every photo. What counts is whether that version could be a real photo.",
            "colors": "Upside down puts the rows in the other order, `dims=[1]`; the three channels, the colors, stay where they are.",
        },
    )
    quick_upside
    return (quick_upside,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en), residual blocks let a network of 19 layers learn where a plain one did not. Since then, this part added photos in color, data augmentation, deeper networks, and ways to see where a network looks. Today they come together. In 2015, Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun trained ResNets of 20 to 1,202 layers on CIFAR-10, all made from one block with **batch normalization** after every convolution. You build the smallest of them, ResNet-20, from your own block, and train it on clothes, with changes that fit clothes. You choose its settings on a validation set, and test it once. And then you do what should follow a good test: an **error analysis**. You count which kinds your network mixes up, look at the mistakes it was most sure of, and see with your Grad-CAM where it looked when it made them.
        """),
        draw.project_steps(["Your first CNN", "LeNet-5, the CNN that read checks", "VGG: deeper with small filters",
                            "ResNet: depth that keeps working", "Your CNN on color photos", "Your own one-stage detector",
                            "Your own U-Net", "Final project: ResNet with Grad-CAM", "Final project: U-Net or detector"],
                           7, here_label="← today", title="Where your project stands: your own CNN"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_wrong():
    guess_wrong = L.guess(
        "most-wrong",
        question="Your ResNet will read most of the 1,000 test images right. Which kind will it read wrong most often?",
        options={
            "tshirt": "The T-shirt",
            "coat": "The coat",
            "shirt": "The shirt",
            "sandal": "The sandal",
        },
        reveal="The shirt. With every one of the seeds 0 to 4, the course's ResNet read more shirts wrong than any other kind: 21 to 29 of the 97, more of them as a coat than as any other kind. And 62% to 68% of all its mistakes are one garment for the upper body read as another: T-shirt, pullover, coat or shirt. The error analysis after the demo looks at those mistakes.",
    )
    guess_wrong
    return (guess_wrong,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **ResNet-20.** The network He and colleagues trained on CIFAR-10 starts with one 3×3 convolution to 16 channels. Then come three **stages** of residual blocks: the first with 16 channels on maps of 32 by 32, the second with 32 channels on 16 by 16, the third with 64 channels on 8 by 8. Each stage halves the height and the width and doubles the channels: the first block of stage 2 and of stage 3 does that with stride 2 in its first convolution. At the end come global average pooling, the mean of every map, as at the end of ResNet-18 in [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en), and a linear layer to 10 logits. With $n$ blocks per stage, each with two convolutions, that is $6n + 2$ layers with weights in a row; with $n = 3$, 20 layers and 0.27 million parameters. On the images of clothes, 28 by 28, the maps of the three stages are 28, 14 and 7 places high and wide.

        **The block.** In the paper, batch normalization comes "right after each convolution and before activation". So $f(x)$ is a convolution of 3 by 3, batch normalization, a ReLU, a second convolution and batch normalization; the residual connection adds $x$, and after the sum comes the second ReLU. Where a block changes the channels or the size, the residual connection gets the same change, as in [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en): a convolution of 1 by 1 with stride 2, here followed by batch normalization too. On CIFAR-10, the paper filled the new channels with zeros instead; its networks for ImageNet use the convolution of 1 by 1, which here costs 2,848 parameters in all, 1% of the 272,970 of the whole network.

        **Batch normalization of feature maps.** `nn.BatchNorm2d(c)` does what `nn.BatchNorm1d` did in [Batch normalization](/course/open/nn-batchnorm?lang=en), with a channel in the place of a neuron: in train mode, it standardizes every channel over all its places in all images of the mini-batch, then multiplies by a learned $\gamma$ and adds a learned $\beta$, one of each per channel. In eval mode, it uses the running averages of the mean and the variance it kept while training. Made of the blocks of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en), without batch normalization, the course's ResNet-20 learned worse, and once it ran away: it read 822 and 834 of the 1,000 images of the validation set right, with seeds 0 and 1, against 892 and 891 with batch normalization; and trained for 40 epochs instead of 20, its validation loss jumped after epoch 4 from 0.713 to 2.315, and from then on it stayed at the loss of guessing, $-\log 0.1 \approx 2.30$: it read 87 of the 1,000 right.

        **Changes that fit clothes.** Every image of Fashion-MNIST shows one piece of clothing, gray, in the middle, on black. A random crop with a black border of 2 pixels moves it by up to 2 pixels in each direction: $(2 \cdot 2 + 1)^2 = 25$ places. A mirror image is less clear-cut. A shirt mirrored is still a shirt. But the shoes all point the same way: the toe points to the left. In all 602 ankle boots of the 6,000 training images, the pixels of the right half, with the heel and the shaft, add up to more than those of the left half, and so they do in 610 of the 617 sneakers and in 553 of the 594 sandals. A mirrored shoe points to the right, as hardly any shoe of the test images does. Your `changed` can do both, and the course chose with the validation set, as the slider below shows.

        **Choose on the validation set, test once.** The 6,000 training images are split: the network learns from the first 5,000, and the last 1,000 are its validation set, as in [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en). Every choice is made on the validation set; here, which changes. The number of epochs and the learning rate stay as the course set them: with the crop, the validation loss was lowest after epoch 16 to 20 in the five seeds, so 20 epochs are not too many. The 1,000 test images are measured once, at the end, so that their number says how the network does on images that nothing was chosen for.

        **Error analysis.** One number, how many test images are right, does not say what is wrong. An error analysis looks at the mistakes themselves. The **confusion matrix**, as in [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), counts for every right kind how often the network guessed each kind. The **surest mistakes** are the wrong guesses with the highest probability: `torch.softmax(logits, dim=1)` gives the softmax of [Many classes: softmax](/course/open/ml-softmax?lang=en) for every row of the logits at once, and the probability of the guess is the largest number of its row. For each of them, Grad-CAM from the last block shows where the evidence for the guess lies, and where that for the right kind lies. Is the label wrong? Do the two kinds look alike? Did the network look at a detail that misled it? Such questions say what to try next: more images of one kind, other changes, or nothing, when even a person could not tell. Whatever you try next, choose it on the validation set again: once a choice is made with the test images, their number no longer says how the network does on new images.
        """),
        L.watch_out("`torch.softmax(logits, dim=1)` takes the softmax along dimension 1, over the kinds of each image: every row then adds up to 1. With `dim=0`, it would take the softmax over the images of each kind, a column at a time, and the numbers in a row would no longer be the probabilities of one image."),
        mo.md(r"""
        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **ResNet-20** | the ResNet of He and colleagues for images of 32 by 32: a convolution, three stages of 3 residual blocks with 16, 32 and 64 channels, global average pooling and a linear layer; 20 layers with weights | `ResNet(1, 16, 3, 10)` |
        | **stage** | the blocks of a ResNet that work on maps of the same size; the first block of the next stage halves the height and the width and doubles the channels | `stage` |
        | **error analysis** | looking at the mistakes of a trained network, not only counting them: which kinds it mixes up, and why | |
        | **surest mistakes** | the wrong guesses of a network with the highest probability, the one it was most sure of first | `sure_mistakes` |
        | `nn.BatchNorm2d(c)` | batch normalization of c channels: every channel standardized over all its places in the mini-batch, then times $\gamma$ plus $\beta$ | |
        | `torch.softmax(logits, dim=1)` | the softmax of every row of the logits: for every image, a probability for every kind | |
        """),
    ])
    return


@app.cell(hide_code=True)
def picture_resnet():
    mo.vstack([
        mo.Html(C.network_picture(L.lang)),
        mo.md(r"""
        The course's ResNet-20 on a sneaker of the test images, before it learned anything, as PyTorch runs it: a box for every layer, with its number of channels above it; the taller the box, the larger its maps. The stem makes 16 maps of 28 by 28; the blocks of stage 1 keep them; `blocks.3` and `blocks.6`, the sixth and the ninth box, halve the height and the width and double the channels; global average pooling makes one number of each of the 64 maps of 7 by 7, and the linear layer 10 logits. After the demo, you can go through the layers of your own trained ResNet and see their feature maps.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Which changes should the network learn with? The slider shows how the course chose: its ResNet-20 trained for 20 epochs with each of three settings, without changes, with the random crop, and with the random crop and a mirror image for half of the images. Each time, the loss on the 5,000 images it learned from, solid, and on the validation set, dashed, after every epoch, for seed 0; and under it what the five seeds 0 to 4 read right of the validation set.
    """)
    return


@app.cell(hide_code=True)
def explore_changes_cell():
    def _frame(k):
        k = int(k)
        _name = ("no changes", "the random crop", "the random crop and a mirror image")[k]
        _html, _stop = C.settings_frame(k, {"title": f"Seed 0, with {_name}", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                                            "alt": f"The loss of the course's ResNet-20 with {_name}, seed 0, after every epoch: on the images it learned from solid, on the validation set dashed."}, L.lang)
        return _html, (f"With {_name}: over the seeds 0 to 4, the network reads {_stop['low']} to {_stop['high']} of the 1,000 images of the validation set right, "
                       f"{_stop['middle']} in the middle; of the 5,000 images it learned from, {_stop['fit_low']} to {_stop['fit_high']}.")

    explore_changes = L.explore("changes-slider", "changes", _frame, start=1, values=[0, 1, 2], role="",
                                task="Without changes, after which epoch is the validation loss lowest, and what does it do after that? And which setting reads the most of the validation set right?")
    explore_changes
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Without changes, the network learns its 5,000 images by heart: with seed 0, it reads 4,999 of them right, and its loss on them falls to 0.013. Its validation loss is lowest after epoch 8, at 0.420; after that it goes up and down and ends at 0.517: the network fits the images it learned from better and better, while its loss on the others rises. With the random crop, it reads fewer of the images it learned from right, 4,689 to 4,711 over the seeds, and more of the validation set: 884 to 894, 892 in the middle, against 871 without changes. With a mirror image too, the network with seed 0 read 900 of the validation set right, more than the 892 with the crop alone; but over the five seeds, 884 to 900, 889 in the middle. One seed would have chosen the mirror image; five chose the crop alone, so the demo trains with the crop alone. The test images played no part in any of it.

    After epoch 1, the validation loss is above the top of the chart with every setting: 8.59 with the crop and seed 0. In eval mode, batch normalization uses its running averages, and after one epoch they still lag behind the weights, which change fast then. With the validation images' own mean and variance, the loss after epoch 1 would have been 0.88, with 643 of the 1,000 right instead of 357.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Your own ResNet, tested and explained:

    1. The block: $f(x)$ is a 3×3 convolution with stride 1 or 2, batch normalization, a ReLU, a 3×3 convolution and batch normalization. The residual connection is $x$ itself, or, where the channels or the size change, a 1×1 convolution with the same stride and batch normalization. The block gives the ReLU of the sum.
    2. The network: a 3×3 convolution to 16 channels, batch normalization and a ReLU; three stages of 3 blocks, with 16, 32 and 64 channels, the first block of stage 2 and 3 with stride 2; global average pooling; a linear layer to 10 logits.
    3. Changes that fit the images: here a random crop with a black border of 2 pixels, made anew every time the Dataset gives an image.
    4. Learn from the first 5,000 training images; choose changes, epochs and learning rate on the last 1,000, the validation set.
    5. Train with the chosen settings, then measure on the test images, once.
    6. The error analysis: the confusion matrix, the surest mistakes, and for each of them the Grad-CAM of the guess and of the right kind, from the last block.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell follows step 1 where a stage changes, with layers that have not learned anything yet: 2 images that come out of stage 1, 16 maps of 28 by 28 each, go through the first convolution of stage 2 and through the 1×1 convolution of its residual connection, and once through a 3×3 convolution with stride 2 but without padding. The second cell does batch normalization on what the first convolution gives, in train mode and in eval mode. The third cell does the first part of step 6 for 6 made-up images and 3 kinds: the probabilities, the guesses, the confusion matrix, and the probability of each wrong guess. Run the cells as they are.
    """)
    return


@app.cell
def example_shapes():
    import torch
    from torch import nn

    # 1. where a stage changes: from 16 channels of 28 by 28 to 32 channels of 14 by 14
    torch.manual_seed(0)
    x = torch.randn(2, 16, 28, 28)
    conv1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
    skip = nn.Conv2d(16, 32, 1, stride=2)
    print(conv1(x).shape)
    print(skip(x).shape)

    # the same convolution without padding: one place fewer
    no_padding = nn.Conv2d(16, 32, 3, stride=2)
    print(no_padding(x).shape)
    return conv1, nn, torch, x


@app.cell
def example_norm(conv1, nn, torch, x):
    # batch normalization of 32 channels: every channel over its places in both images
    norm = nn.BatchNorm2d(32)
    with torch.no_grad():
        y = norm(conv1(x))
        # how far the mean of each channel is from 0, and its standard deviation
        print(y.mean(dim=(0, 2, 3))[:4].abs().round(decimals=3))
        print(y.std(dim=(0, 2, 3))[:4].round(decimals=3))

        # in eval mode: the running mean and variance it kept
        norm.eval()
        print(norm.running_mean[:4].round(decimals=3))
        print(norm(conv1(x)).std(dim=(0, 2, 3))[:4].round(decimals=3))
    return


@app.cell
def example_mistakes(torch):
    # 6. made-up logits of 6 images for 3 kinds, and their right kinds
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1], [1.0, 1.2, 0.9],
                           [0.1, 0.3, 4.0], [2.5, 2.4, 0.0], [0.0, 0.2, 1.5]])
    labels = torch.tensor([0, 0, 2, 1, 1, 2])

    # the probabilities of every row, and the guesses
    p = torch.softmax(logits, dim=1)
    guess = p.argmax(dim=1)
    print(p.round(decimals=3))
    print(guess)

    # the confusion matrix: a row for every right kind, a column for every guess
    counts = torch.zeros(3, 3, dtype=torch.int64)
    for i in range(len(labels)):
        counts[labels[i], guess[i]] += 1
    print(counts)

    # the probability of every wrong guess, with its place
    for i in range(len(labels)):
        if guess[i] != labels[i]:
            print(i, round(p[i, guess[i]].item(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In the first cell, the 3×3 convolution with stride 2 and padding 1 and the 1×1 convolution with stride 2 both make 32 maps of 14 by 14 of the 16 maps of 28 by 28, so `x + f` can add them, place by place. Without padding, the 3×3 convolution makes 13 by 13, `(28 - 3) // 2 + 1`, and the sum would stop with an error. In the second cell, batch normalization in train mode gives every channel a mean of 0 and a standard deviation of 1: `.std()` prints 1.001 because it divides by 391, one less than the $2 \cdot 14 \cdot 14 = 392$ numbers of a channel, and batch normalization divides by 392. At the start, $\gamma$ is 1 and $\beta$ is 0. The running average of the mean was 0; after this one mini-batch it is $0.9 \cdot 0 + 0.1 \cdot 0.054 = 0.005$ for the first channel. In eval mode, batch normalization uses the running averages, and the standard deviation becomes 0.584, not 1: after one mini-batch, they are not yet those of the maps. The demo measures your network in eval mode only after it trained, when they are. In the third cell, the guesses are kinds 0, 1, 1, 2, 0 and 2, and 2 of the 6 are right: the diagonal of the confusion matrix. Of the 4 wrong guesses, image 3 is the surest, with 0.957, then image 1 with 0.896: `sure_mistakes(logits, labels, 2)` gives `[3, 1]`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_block():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The block

        Complete the class `Block(c_in, c_out, stride)`, step 1 of the recipe. In `__init__`: `self.conv1`, a 3×3 convolution from `c_in` channels to `c_out` with this `stride` and padding 1, and `self.norm1`, its batch normalization; `self.conv2`, from `c_out` channels to `c_out` with padding 1, and `self.norm2`. Where the stride is not 1 or the channels change, `self.skip`, a 1×1 convolution from `c_in` channels to `c_out` with this `stride`, and `self.skip_norm`; elsewhere `self.skip` stays `None`. In `forward`: $f(x)$, the residual connection, and the ReLU of the sum. The first cell of the worked example made such convolutions. The check tries a block where a stage changes and a block inside a stage on random maps, an odd size, the gradient, and counts the parameters. When it is right, it saves `Block` in your project, as `work/projects/cnn/block.py`; your ResNet is made of it.
        """),
        L.task("Make `Block` the residual block of ResNet, with batch normalization after every convolution."),
    ])
    return


@app.cell(hide_code=True)
def exercise_block():
    box_block = L.code_box("block", lines=22)
    box_block
    return (box_block,)


@app.cell(hide_code=True)
def check_block(box_block):
    result_block = L.check("block", box_block.value)
    result_block
    return (result_block,)


@app.cell(hide_code=True)
def hints_block():
    L.hints(
        "block",
        [
            "In `__init__`, make the layers in this order: `conv1`, `norm1`, `conv2`, `norm2`, and in the `if`, `skip` and `skip_norm`. Each batch normalization comes after a convolution to `c_out` channels. In `forward`, first $f(x)$, then what the residual connection makes of `x`, then the sum.",
            "`self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)` and `self.norm1 = nn.BatchNorm2d(c_out)`; in the `if`, `self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride)` and `self.skip_norm = nn.BatchNorm2d(c_out)`. In `forward`: `f = self.norm2(self.conv2(torch.relu(self.norm1(self.conv1(x)))))`.",
            "After `f`: `if self.skip is not None:` with `x = self.skip_norm(self.skip(x))`, and then `return ___`, the ReLU of `x + f`.",
        ],
        why="Batch normalization right after each convolution keeps what goes into every ReLU at a mean of about 0 and a standard deviation of about 1 at the start, whatever the weights. $f$ ends with batch normalization and no ReLU, so that it can make numbers below 0 too and take something away from $x$; the ReLU comes after the sum. Where the shape changes, $x$ must change in the same way, or `+` cannot add the two.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resnet():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your ResNet

        Complete the class `ResNet(channels, width, blocks, kinds)`, step 2 of the recipe: a ResNet from your own `Block`. `__init__` already has the stem, `self.stem` and `self.stem_norm`, and after the loop the blocks in `self.blocks` and the output layer `self.out`. Fill in the loop: three stages, with `width`, `2 * width` and `4 * width` channels, each of `blocks` blocks; the first block of stage 2 and of stage 3 comes from the channels of the stage before, with stride 2, and every other block keeps its channels, with stride 1. Then `forward`: the stem, its batch normalization and a ReLU, every block in turn, global average pooling and the output layer. The check tries a small ResNet on two random images of clothes, looks at which block stands where, at the maps of the last block, and at color photos with 5 kinds, and counts the parameters of `ResNet(1, 16, 3, 10)`. When it is right, it saves `ResNet` in your project, as `work/projects/cnn/deep_resnet.py`; the demo trains it.
        """),
        L.task("Make `ResNet` a stem, three stages of your blocks, global average pooling and a linear layer."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resnet():
    box_resnet = L.code_box("resnet", lines=26)
    box_resnet
    return (box_resnet,)


@app.cell(hide_code=True)
def check_resnet(box_resnet):
    result_resnet = L.check("resnet", box_resnet.value)
    result_resnet
    return (result_resnet,)


@app.cell(hide_code=True)
def hints_resnet():
    L.hints(
        "resnet",
        [
            "Two loops inside each other: `for stage in range(3):` with the channels of that stage, `c = width * 2 ** stage`, and inside it `for b in range(blocks):`. Which block is the only one that changes the channels and the size, and from how many channels does it come?",
            "Inside the inner loop: `if stage > 0 and b == 0:` then `layers.append(Block(c // 2, c, 2))`, else `layers.append(Block(c, c, 1))`. In `forward`, start with `x = torch.relu(self.stem_norm(self.stem(x)))`, then `for block in self.blocks:` with `x = block(x)`.",
            "After the loop in `forward`: `return self.out(___)`, with the mean of every map over its height and width, dimensions 2 and 3.",
        ],
        why="Every stage works on maps half as high and wide as the stage before, with twice the channels, so that every layer costs about the same to compute. The first block of a stage does that change, with stride 2 and a 1×1 convolution on its residual connection; the other blocks keep their channels, and their residual connection is $x$ itself. Global average pooling makes one number of each map, whatever the size of the image, and Grad-CAM takes its maps from the last block.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_changes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Changes for any image

        Complete `changed(x, u, pad, mirror)`, step 3 of the recipe, for an image `x` of any shape (C, H, W). `u` holds three random numbers from 0 to 1, which the demo draws anew every time. When `mirror` is True and `u[0]` is below 0.5, the image is mirrored first. Then it goes in a black image with `pad` pixels on every side, and the function gives back the H by W that starts at row `int(u[1] * (2 * pad + 1))` and column `int(u[2] * (2 * pad + 1))` of it: one of the $(2 \cdot pad + 1)^2$ places. Your `crop` of [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en) did that for photos of 3 by 32 by 32 with a border of 4; this one works for every size. The check tries a made-up gray image of 4 by 4, a color image of 5 by 6, mirror on and off, and borders of 0 to 2. When it is right, it saves `changed` in your project, as `work/projects/cnn/changes.py`; the demo changes every training image with it.
        """),
        L.task("Make `changed` mirror an image when it may, and cut it out of a black border at the place `u` gives."),
    ])
    return


@app.cell(hide_code=True)
def exercise_changes():
    box_changes = L.code_box("changes", lines=14)
    box_changes
    return (box_changes,)


@app.cell(hide_code=True)
def check_changes(box_changes):
    result_changes = L.check("changes", box_changes.value)
    result_changes
    return (result_changes,)


@app.cell(hide_code=True)
def hints_changes():
    L.hints(
        "changes",
        [
            "Three steps, as in the comments. A mirror image puts the columns in the other order: axis 2 of (C, H, W). The black image is as large as `x` plus `pad` on every side, and has as many channels as `x`.",
            "`if mirror and u[0] < 0.5:` with `x = x.flip(dims=[2])`. Then `c, h, w = x.shape`, `padded = torch.zeros(c, h + 2 * pad, w + 2 * pad)` and `padded[:, pad:pad + h, pad:pad + w] = x`.",
            "`top, left = int(u[1] * (2 * pad + 1)), int(u[2] * (2 * pad + 1))`, and then `return ___`, the part of `padded` from row `top` and column `left`, `h` rows high and `w` columns wide.",
        ],
        why="`u[1]` and `u[2]` are below 1, so `int(u * (2 * pad + 1))` is a whole number from 0 to 2 · pad: every one of the 2 · pad + 1 places has the same chance. With `pad = 0`, the cut is the image itself, and only the mirror image is left.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. The surest mistakes

        Complete `sure_mistakes(logits, labels, n)`, the second part of step 6. `logits` has a row for every image and a column for every kind; `labels` holds the right kind of every image. The function gives back a list of the places of the wrong guesses, the one with the highest probability first, at most `n` of them. The third cell of the worked example printed the probability of every wrong guess: make a list of pairs `(probability, place)`, as in [Try an image classifier: top-1 and top-5](/course/open/cv-try-classifier?lang=en), so that `sorted` sorts them by their probability. The check tries the made-up logits of the worked example, with `n` from 0 to more than there are mistakes, and logits without mistakes. When it is right, it saves `sure_mistakes` in your project, as `work/projects/cnn/mistakes.py`; the error analysis uses it.
        """),
        L.task("Make `sure_mistakes` give the places of the wrong guesses, from the surest down."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_sure_mistakes = L.code_box("sure-mistakes", lines=10)
    box_sure_mistakes
    return (box_sure_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_sure_mistakes):
    result_sure_mistakes = L.check("sure-mistakes", box_sure_mistakes.value)
    result_sure_mistakes
    return (result_sure_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "sure-mistakes",
        [
            "First the probabilities and the guesses, as in the third cell of the worked example. Then, for every place `i` where the guess is not the label, a pair: the probability of the guess first, so that `sorted` sorts by it, and the place second.",
            "`pairs = [(p[i, guess[i]].item(), i) for i in range(len(labels)) if guess[i] != labels[i]]`. `sorted(pairs, reverse=True)` puts them from the highest probability down.",
            "`return [pair[1] for pair in ___]`, with the sorted pairs cut to the first `n`.",
        ],
        why="`sorted` sorts pairs by their first number, and only looks at the second when two first numbers are the same. `[:n]` takes at most `n`: when there are fewer mistakes, all of them, and with `n = 0`, none.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your project can do now

    Press the button: the demo trains your `ResNet(1, 16, 3, 10)`, made of your `Block`, on the first 5,000 training images, changed anew by your `changed` every time the Dataset gives one, with a border of {C.PAD} and no mirror image, as the course chose on the validation set. It trains with your Trainer for {C.EPOCHS} epochs in mini-batches of 64, with a learning rate of {C.number(C.LR, 2, L.lang)}, a warmup of one epoch and seed 0. Then it measures your network once on the 1,000 test images. Where a module of yours is missing or broken, the course's stands in. It trains on one thread, so that it follows the course's run with seed 0 as closely as your computer allows; on the course's computer, that took 6 to 8 minutes. While it trains, the chart shows the loss on the changed training images every 10 batches, and dashed, the loss on the validation set after every epoch. Beside it are your network's guesses at that moment for the first test image of each kind, and under them the feature maps of its last block on a shirt; both are drawn again as it trains. When it is done, you can go back through the epochs and see both after each of them.
    """)
    return


@app.cell(hide_code=True)
def train_button_cell():
    train_button = mo.ui.run_button(label="Train your ResNet")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_resnet(train_button, result_block, result_resnet, result_changes):
    result_block, result_resnet, result_changes  # run again when an exercise saves your module
    if not train_button.value:
        _out = mo.md("*Press the button to train your ResNet.*")
    else:
        _now = {}   # the live chart, made at the first report
        _heads = {"guesses": "Your ResNet guesses", "layer": "Layer {name}, the last block of your ResNet, on a shirt"}

        def _progress(_p):
            if "chart" not in _now:
                _now["chart"] = L.live({"title": "Your ResNet: the loss while it trains", "x": "epoch", "y": "loss",
                                        "alt": "The loss of your ResNet so far: on the changed training images every 10 batches, and dashed on the validation set after every epoch."}, _p["epochs"], low=0.0, high=C.TOP_LOSS,
                                       key=[("loss", False, "changed training images, every 10 batches"), ("loss", True, "validation set, after every epoch")])
                _now["chart"].watch(lambda: C.now_pictures(_p["look"](), _heads, L.lang, "nv-project-now"))
                _now["val"] = []
            if "val_loss" in _p:   # the validation set after every epoch, dashed
                _now["val"].append((_p["epoch"], _p["val_loss"]))
                _now["chart"].reference(_now["val"], "", role="loss", replace=True)
                _now["chart"].line(f"**Your ResNet** · epoch {_p['epoch']} of {_p['epochs']} · validation loss {C.number(_p['val_loss'], 3, L.lang)}")
            elif "loss" in _p:   # the training images every 10 batches
                _now["chart"].add(_p["epoch"] - 1 + _p["batch"] / _p["batches"], _p["loss"],
                                  f"**Your ResNet** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']} · loss {C.number(_p['loss'], 3, L.lang)}")
            else:
                _now["chart"].line(f"**Your ResNet** · epoch {_p['epoch']} of {_p['epochs']} · batch {_p['batch']} of {_p['batches']}")

        _result = C.demo(_progress)
        if _result is None:
            _out = mo.md("*Finish exercises 2 and 3 first: then your `ResNet` and your `changed` are in your project, and this cell uses them.*")
        elif isinstance(_result, str):
            _note = {
                "slow": "*Training takes too long here: more than twenty minutes. Does your `ResNet`, your `Block` or your `changed` do more work than it needs, such as more blocks or channels than the recipe? If it follows the recipe, this computer is too slow for the demo.*",
                "resnet": "*Your `ResNet` in `work/projects/cnn/deep_resnet.py` stops with an error, or does not give 10 logits for every image. Did you change its file? Delete it, and check exercise 2 again: its check then saves your code again.*",
                "changes": "*Your `changed` in `work/projects/cnn/changes.py` stops with an error, or does not give back an image of shape (1, 28, 28). Did you change its file? Delete it, and check exercise 3 again.*",
                "trainer": "*Training with your Trainer in `work/projects/nn/trainer.py`, or counting with your `count_params` in `work/projects/cnn/count.py`, stops with an error. Did you change one of those files? Delete it, and check its exercise again.*",
            }[_result]
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note under *Where your project stands* says."
            _labels = {"title": "Loss after every epoch", "x": "epoch", "y": "loss", "train": "training", "val": "validation",
                       "alt": "The loss of your ResNet after every epoch: on the changed training images solid, on the validation set dashed, with a dot where the validation loss is lowest."}
            _val, _guesses = _result["curves"]["val"], _result["guesses"]
            _when = C.moments(len(_val)) if len(_guesses) == len(_val) else []

            def _frame(_e):
                _e = int(_e)
                _right = sum(_g == _k for _g, _k in zip(_guesses[_e - 1], _result["truth"]))
                return (C.epoch_frame(_result, _e, _labels, _heads, L.lang),
                        f"After epoch {_e}: validation loss {C.number(_val[_e - 1], 3, L.lang)}, and {_right} of these 10 images read right.")

            if len(_when) > 1:   # rewinding through the run, with the whole run in the picture
                _pic = [L.explore("rewind", "epoch", _frame, start=_when[-1], values=_when, role="",
                                  task="Go back to epoch 1, then forward again: which kinds does your network read right first?")]
            else:
                _pic = [mo.Html(C.curves_picture(_result["curves"], _labels, L.lang))] if _val else []
            _out = mo.vstack(_pic + [
                mo.md(f"""Your ResNet, with {C.grouped(_result['params'], L.lang)} parameters, reads **{C.grouped(_result['fit'], L.lang)}** of the 5,000 images it learned from right, as they are, and **{_result['val']}** of the 1,000 images of the validation set. On the 1,000 test images, measured once: **{_result['test']}**. The course's run with seed 0, which the demo follows, read 4,708, 892 and 909 right; over the seeds 0 to 4, 903 to 914 of the test images.{_who}"""),
                mo.md("**Your ResNet, layer by layer.** The first box is a shirt of the test images, and the boxes after it are the layers of your trained ResNet, in the order the shirt goes through them; under them are the feature maps of the layer you choose."),
                (L.explore("layers", "layer", lambda _k: C.layer_frame(_result["net"], _k, L.lang), start=1,
                           values=list(range(1, len(_result["net"].layers) + 1)), role="",
                           task="Go through your trained ResNet from the shirt to the logits: where do the maps stop looking like the shirt?")
                 if len(_result["net"].layers) > 1 else mo.Html(C.layer_frame(_result["net"], 1, L.lang))),
                mo.md("In the stem and in stage 1, the maps are 28 by 28, as large as the image, and most of them still follow the shirt: its outline, or the light and dark parts of its fabric. Every stage halves the places: in stage 3, 7 by 7 are left, and in almost every map only a few of the 49 are dark, where on the shirt that channel found what it looks for. Global average pooling makes one number of each of the 64 maps of `blocks.8`, and the linear layer, the last box, makes 10 logits of them."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_analysis():
    mo.md(r"""
    ### The error analysis

    Once your ResNet has trained, the button below looks at its mistakes on the test images. First the confusion matrix: a row for every right kind, a column for every guess. Then the 8 mistakes your `sure_mistakes` puts first, each twice, with your `grad_cam` of [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en) from the last block, `blocks[-1]`, laid over it: on the left for the kind it guessed, on the right for the right kind. Where the map is low, the image is veiled.
    """)
    return


@app.cell(hide_code=True)
def analysis_button_cell():
    analysis_button = mo.ui.run_button(label="Look at the mistakes")
    analysis_button
    return (analysis_button,)


@app.cell(hide_code=True)
def analysis_cell(analysis_button, result_sure_mistakes):
    result_sure_mistakes  # run again when exercise 4 saves your module
    _run = C.last_run()
    if not analysis_button.value:
        _out = mo.md("*Train your ResNet above first, then press the button.*")
    elif _run is None:
        _out = mo.md("*There is no trained ResNet yet: press the button of the demo above first, and this one again when it is done.*")
    else:
        _found = C.analysis(_run)
        if _found is None:
            _out = mo.md("*Finish exercise 4 first: then your `sure_mistakes` is in your project, and this cell uses it.*")
        elif isinstance(_found, str):
            _out = mo.md({
                "mistakes": "*Your `sure_mistakes` in `work/projects/cnn/mistakes.py` stops with an error, or does not give a list of places of wrong guesses. Did you change its file? Delete it, and check exercise 4 again.*",
                "grad_cam": "*Your `grad_cam` in `work/projects/cnn/gradcam.py` stops with an error, or does not give a map of 7 by 7 without numbers below 0. Did you change its file? Delete it, and check its exercise in [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en) again.*",
            }[_found])
        else:
            _who = "" if not _found["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _found["stand_ins"]) + "."
            _out = mo.vstack([
                mo.Html(C.confusion_picture(_found["confusion"], {"guess": "guess", "right": "right kind",
                                                                  "alt": "The confusion matrix of your ResNet on the 1,000 test images: a row for every right kind, a column for every guess, with the count in every cell; the cells with mistakes are tinted, the more mistakes the darker."}, L.lang)),
                mo.Html(C.mistakes_picture(_found, {"title": "The mistakes your ResNet was most sure of",
                                                    "read": "{right}, read as {guess}", "sure": "probability {p} · test image {place}",
                                                    "card": "Test image {place}, a {right}, which your ResNet read as {guess} with a probability of {p}: once veiled where the Grad-CAM of {guess} is low, once where that of {right} is low."}, L.lang)),
                mo.md(f"""With seed 0, the course's ResNet reads 909 of the 1,000 test images right. Of its 91 mistakes, 62 are one garment for the upper body read as another, and 11 one shoe read as another; the trousers and the bags it reads wrong only twice in all. It reads the shirts wrong most: 21 of the 97, 10 of them as a coat. Each of the 8 mistakes it was most sure of has a probability above 0.97. On six of their cards, the map of the right kind is at most about half as strong as that of the guess: the network found evidence for what it guessed, and much less for the right kind. Only on the two shirts read as a coat, test images 44 and 444, are both maps about as strong: there it found both, and chose coat.{_who}"""),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ### Exercise 5. Why it went wrong
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-wrong",
        question="Test image 147 is a shirt. The course's ResNet with seed 0 read it as a dress with a probability of 0.999, and so did each of the seeds 0 to 4, with 0.997 or more. On its card, the map of dress is strongest on the long, narrow body of the garment below the sleeves, and the map of shirt is veiled everywhere: its largest number is less than a tenth of that of dress. What does that say?",
        options={
            "right": "To the network, this shirt looks like a dress: long and narrow below the sleeves, with nothing in it that it learned for a shirt",
            "more": "The network trained too short: with more epochs, it would read this shirt right",
            "mirror": "Mirror images would have helped: the network never saw this shirt the other way round",
            "broken": "Grad-CAM does not work on this image: a map that is veiled everywhere means it found nothing",
        },
        correct="right",
        why={
            "right": "Yes. The evidence for dress lies on the long, narrow body, 9 pixels wide in the 16 rows below the sleeves, and next to nothing speaks for shirt. The network did what it learned from its images; what could change that is more shirts of this shape among the images it learns from.",
            "more": "Each of the five seeds read this shirt as a dress, with a probability of 0.997 or more, after its 20 epochs. A mistake that every run makes, and this surely, comes from what the images taught it, not from a training that stopped too soon.",
            "mirror": "The course's runs with mirror images read this shirt as a dress too: mirrored, a long, narrow shirt is still long and narrow.",
            "broken": "Grad-CAM works here: from the same layer, on the same image, the map of dress is clear. A map that is veiled everywhere says that nothing in the image raises the logit of shirt: the network found no evidence for it.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Exercise 6. Your report
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Write a short report on your ResNet for someone who did not take this lesson: how were its settings chosen, how did it do on the test images, and what do its mistakes show? Use the numbers of your own run, or those of the course's run with seed 0.",
        model_answer="**Settings.** ResNet-20 with batch normalization after every convolution, 272,970 parameters, trained for 20 epochs on 5,000 images of clothes. Its changes were chosen on a validation set of 1,000 other training images, over five seeds: without changes it read 871 of them right in the middle, with a random crop of up to 2 pixels 892, and with mirror images too 889. So it trains with the crop alone. **Test.** Of the 1,000 test images, measured once, it reads 909 right. 62 of its 91 mistakes are one garment for the upper body read as another; it reads the shirts wrong most, 21 of the 97, 10 of them as a coat. **Mistakes.** For six of the eight mistakes it was most sure of, Grad-CAM shows evidence for the guess and much less for the right kind: test image 147, a long, narrow shirt, looks like a dress to it. On two shirts read as a coat, it found evidence for both. To try next, and to choose on the validation set: more shirts of such shapes among the images it learns from.",
        points=[
            "you say that the changes were chosen on the validation set, with what each setting read right there, and that the test images were measured once, at the end",
            "you give what it read right of the 1,000 test images, and which kinds it mixes up most, from the confusion matrix",
            "you say what the Grad-CAM of the guess and of the right kind shows for a mistake it was sure of, and what that suggests to try next, chosen on the validation set",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The residual block of [ResNet: depth that keeps working](/course/open/cnn-resnet?lang=en) now has the batch normalization of [Batch normalization](/course/open/nn-batchnorm?lang=en) after every convolution, as in the paper, and your ResNet is made of three stages of it, as ResNet-20. The validation set of [Choosing hyperparameters: the validation set](/course/open/ml-validation-set?lang=en) chose its changes, among them the crop of [Data augmentation for photos: flip, crop and color](/course/open/cv-augment-photos?lang=en); the test images were measured once. And after the test came the error analysis: the confusion matrix of [Beyond accuracy: precision and recall](/course/open/ml-confusion-matrix?lang=en), and your Grad-CAM of [Grad-CAM: where a CNN looked](/course/open/cv-grad-cam?lang=en) over the mistakes your network was most sure of. Your project `cnn` now has `block.py`, `deep_resnet.py`, `changes.py` and `mistakes.py`. In the next lesson, *Final project: your own U-Net or detector*, you choose the other final project: a network that does not only say what is in an image, but where.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_block, result_resnet, result_changes, result_sure_mistakes, choice_why, explain_report):
    L.wrap_up(result_block, result_resnet, result_changes, result_sure_mistakes, choice_why, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
