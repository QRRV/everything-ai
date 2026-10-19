# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson torch-pretrained. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("torch-pretrained", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Saving and loading a model](/course/open/torch-checkpoints?lang=en) and [Where does this data come from?](/course/open/data-provenance?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sizes():
    quick_sizes = L.choice(
        "quick-sizes",
        question="You saved the state_dict of an MLP with 64 hidden neurons. Then you make an MLP of the same class with 128 hidden neurons, and load that file into it with `load_state_dict`. What happens?",
        options={
            "half": "The first 64 neurons get the saved numbers, the other 64 keep theirs",
            "right": "PyTorch stops with an error: the shapes do not match",
            "grow": "PyTorch makes the saved layer bigger, with zeros",
        },
        correct="right",
        why={
            "half": "`load_state_dict` copies whole tensors, and only into a tensor of the same shape. See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "grow": "PyTorch changes no sizes for you: the model must be of the same class, with the same sizes. See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "right": "A weight of shape (128, 784) cannot take one of (64, 784). So a file of numbers is only of use with the class it was made for, and its sizes.",
        },
    )
    quick_sizes
    return (quick_sizes,)


@app.cell(hide_code=True)
def quick_check_nc():
    quick_nc = L.choice(
        "quick-nc",
        question="A dataset of street photos has the license CC BY-NC 4.0. May you train a model on it for an app that you sell?",
        options={
            "right": "No: NC means you may not earn money with it, and with an app that you sell, you do",
            "credit": "Yes, if you name the maker and link to the license",
            "free": "Yes: you only use the photos, you do not share them",
        },
        correct="right",
        why={
            "credit": "That is enough for CC BY. NC adds: not to earn money. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "free": "The license sets what you may do with the data, also when you only train on it. See [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "And what goes for data goes for a model too: its license says what you may do with it, and the license of its training data matters as well.",
        },
    )
    quick_nc
    return (quick_nc,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A good model takes data, time and computers to train. Often someone has already trained one for your task, and shares it: on a **model hub**, a website full of trained models that anyone can download. The course's own digit reader is there too. But a file of numbers is only of use with the code of its class, and only safe and allowed when you know where it comes from. That is what its **model card** tells you.

    **An everyday example.** A secondhand bike: you can ride it today, without building one. But first you look at its papers: who owned it, how far it went, whether it was stolen, and whether it is the size of bike you need.
    """)
    return


@app.cell(hide_code=True)
def predict_loaded():
    guess_loaded = L.guess(
        "loaded",
        question="You make a new `DigitCNN`, load the course's file from the hub into it, and do not train it at all. How many of the 1,000 test digits do you think it reads right?",
        options={
            "guess": "About 100: it has not trained yet",
            "half": "About 500: a file keeps only part of what it learned",
            "right": "About 930: what the course's training reached",
        },
        reveal=f"**About 930: {C.RIGHT}.** All that the model learned in the course's training is in the 51,648 numbers of the file; loading them is all it takes. A new `DigitCNN` without them reads 85 to 129 of the 1,000 right, over 5 seeds: as good as guessing one of 10 digits.",
    )
    guess_loaded
    return (guess_loaded,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(rf"""
        ## The idea

        **A model hub** keeps trained models, each in a repository of files with an address of its own. The best known is Hugging Face; torchvision and timm, two Python libraries, offer trained models for images too: one call gives a model of the class with its numbers. The course's digit reader is the repository `{C.REPO}` on Hugging Face, at [{C.page()}]({C.page()}). It holds `model.pt`, the state_dict, and `README.md`, its model card. Every file can be downloaded, without an account, from a fixed address, such as `{C.address("model.pt")}`.

        **A model card** is to a model what a datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en) is to data: what the model does, its class and size, the data it was trained on and their license, the license of the model itself, how well it does, and what it does not cover. You read it before you use the model.

        **Load it safely.** A `.pt` file from someone else could run code on your computer when you load it, unless `torch.load` reads only tensors. Since PyTorch 2.6 it does so by default; write `torch.load(path, weights_only=True)` anyway for every file you did not make yourself: then PyTorch refuses anything else, whatever the default.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **pretrained model** | | a model that someone else trained, shared as its class and its parameters | |
        | **model hub** | | a website that keeps trained models, each in a repository with its files | |
        | **model card** | | the description of a model: task, class and size, training data, license, how well it does, its limits | `README.md` |
        | `weights_only=True` | weights only | `torch.load` reads only tensors and simple values from the file | `torch.load(path, weights_only=True)` |
        """),
        L.watch_out("A `.pt` file holds only names and numbers, not the code of its class. So you need the class that the card names, `DigitCNN`: its code is in the card, and in the course as `everything_ai.models.DigitCNN`. Its first layers are convolutions, a kind of layer for images that a later lesson explains; to load and use the model, you only need the class, not how it works inside."),
        mo.md(r"""
        The picture: the repository on the hub, with the model and its card, and the way from `model.pt` into a model of your own.
        """),
        mo.Html(C.hub_picture({"title": "From the hub into your model", "hub": "On Hugging Face:", "card": "the model card",
                               "weights": "the state_dict, 51,648 numbers",
                               "steps": [("torch.load(path, weights_only=True)", "download model.pt, then read it"),
                                         ("model = DigitCNN()", "a new model of the class on the card"),
                                         ("model.load_state_dict(state)", "the numbers from the file into it"),
                                         ("model.eval()", "eval mode, to read digits")],
                               "alt": "The repository QRRV/everything-ai-qmnist-cnn on Hugging Face holds README.md, the model card, and model.pt, the state_dict of 51,648 numbers. From model.pt, four steps follow: torch.load with weights_only=True, a new DigitCNN, load_state_dict, and eval mode."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To use a pretrained model:

    1. Find it on a hub, and read its model card: what it does, its class and size, its training data, its license, its limits. Is it for your task, and may you use it so?
    2. Download its file.
    3. Load the file with `torch.load(path, weights_only=True)` into a new model of the class on the card, with `load_state_dict`, and switch it to eval mode.
    4. Try it on data of your own: does it do what the card says?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The four steps for the course's digit reader. The comments with `#` give the step of the recipe. `C.card()` gives the model card from the hub, and `C.hub_file("model.pt")` the file, with where it came from: `hub` when it was downloaded, or `course` when the download does not work, for instance without a connection: then the course's own copy, the same file, is used.
    """)
    return


@app.cell
def example_card():
    # 1. the model card
    mo.md(C.card())
    return


@app.cell
def example_load():
    import torch
    from everything_ai.models import DigitCNN

    # 2. the file
    _path, _where = C.hub_file("model.pt")
    print("model.pt from:", _where)
    # 3. load it into a new model of the class on the card
    _state = torch.load(_path, weights_only=True)
    print(len(_state), "tensors:", list(_state))
    _model = DigitCNN()
    print(_model.load_state_dict(_state))
    _model.eval()
    # 4. try it on the 1,000 test digits
    _X, _y = C.qmnist_test()
    with torch.no_grad():
        _pred = _model(_X).argmax(dim=1)
    print((_pred == _y).sum().item(), "of", len(_y), "right")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(rf"""
    The card says the class is `DigitCNN`, that it was trained on 6,000 digits of QMNIST, that it has the license CC BY 4.0, and that it reads 929 of the 1,000 test digits right. The file holds 10 tensors, two for each of the 5 layers on the card: a weight and a bias; `load_state_dict` finds a place for each of them. And the model, without any training of your own, reads {C.RIGHT} of the 1,000 test digits right: what the card says.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which model?
    """)
    return


@app.cell(hide_code=True)
def choice_pick_cell():
    choice_pick = L.choice(
        "pick-model",
        question="For an app that you sell, you need a model that reads digits written on paper and photographed with a phone. Three model cards. Which do you pick?",
        options={
            "course": "A: trained on QMNIST, white digits on black of 28 by 28, CC BY 4.0, 929 of 1,000 right",
            "nc": "B: trained on 50,000 phone photos of digits on paper, CC BY-NC 4.0, 97 of 100 right",
            "right": "C: trained on 30,000 phone photos of digits on paper, CC BY 4.0, 94 of 100 right",
        },
        correct="right",
        why={
            "course": "Its card says: only digits as in QMNIST, white on black, centred, 28 by 28 pixels. A phone photo of a digit on paper is dark on light, and far bigger.",
            "nc": "It fits the task, and does best, but NC means you may not earn money with it: not in an app that you sell.",
            "right": "It is trained on images like yours, and CC BY lets you sell your app if you name the makers. Then try it on photos of your own: step 4.",
        },
    )
    choice_pick
    return (choice_pick,)


@app.cell(hide_code=True)
def text_exercise_load():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Load the digit reader

        Complete `load_reader(path)`: step 3 of the recipe, for a file with a state_dict of a `DigitCNN`. It gives back the loaded model, in eval mode. The check calls `load_reader(path)` with the course's file, and with two other files, and looks at what comes back: its class, its mode, and the digits it reads for the first 8 test images.
        """),
        L.task("Make `load_reader` give back a `DigitCNN` with the numbers of the file, in eval mode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_load():
    box_load = L.code_box("load-reader", lines=10)
    box_load
    return (box_load,)


@app.cell(hide_code=True)
def check_load(box_load):
    result_load = L.check("load-reader", box_load.value)
    result_load
    return (result_load,)


@app.cell(hide_code=True)
def hints_load():
    L.hints(
        "load-reader",
        [
            "Follow step 3 of the worked example, with `path` for the file.",
            "First the numbers from the file, `torch.load(path, weights_only=True)`; then a new model, `DigitCNN()`.",
            "`model.load_state_dict(state)` copies the numbers into the model, and `model.eval()` switches it to eval mode.",
        ],
        why="A new `DigitCNN()` starts with random numbers; `load_state_dict` replaces every one of them by the numbers of the file. Only then does it read digits as the card says.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_safe():
    mo.md(r"""
    ### Exercise 3. A file from a stranger
    """)
    return


@app.cell(hide_code=True)
def choice_safe_cell():
    choice_safe = L.choice(
        "safe-load",
        question="On a forum, someone you do not know shares `reader.pt`: a digit reader, they say, that is better than the course's. It has no model card. What do you do?",
        options={
            "right": "Not use it until you know where it comes from; and if you load it, only with `weights_only=True`",
            "try": "Load it with `torch.load` and test it: a file of numbers cannot do harm",
            "never": "Never use a model that you did not train yourself",
        },
        correct="right",
        why={
            "try": "A `.pt` file can hold more than numbers: unless `torch.load` reads only tensors, loading it can run code on your computer. And without a card, you do not know its data or license.",
            "never": "Then you could use no hub at all. A model with a card from a maker you can check, loaded safely, is fine to use.",
            "right": "Without a card you know neither its training data nor its license; and `weights_only=True` makes sure that loading reads only tensors.",
        },
    )
    choice_safe
    return (choice_safe,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The state_dict and `load_state_dict` of [Saving and loading a model](/course/open/torch-checkpoints?lang=en), now for a file someone else made, and the datasheet and licenses of [Where does this data come from?](/course/open/data-provenance?lang=en), now for a model: its model card. In the next lesson, [A pretrained network as feature extractor](/course/open/torch-frozen-features?lang=en), you keep the digit reader's layers as they are, and train only a new last layer for another task.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pick, result_load, choice_safe):
    L.wrap_up(choice_pick, result_load, choice_safe)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
