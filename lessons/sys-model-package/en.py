# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-model-package. Route: course/route/lessen/systemen.yaml. Checks, the model and its files, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-package", "en")
    C = L.checks   # the digit reader, its safetensors files, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using a pretrained model](/course/open/torch-pretrained?lang=en) and [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_load():
    quick_load = L.choice(
        "quick-load",
        question="Why write `torch.load(path, weights_only=True)` for a model file you did not make yourself?",
        options={
            "fast": "It loads the file faster, as it skips the code of the class",
            "card": "It first compares every number in the file with the numbers on the model card, to catch a changed file",
            "right": "Then `torch.load` reads only tensors and simple values, and refuses anything else",
            "names": "It loads the numbers without their names, so a name can do no harm",
        },
        correct="right",
        why={
            "right": "With `weights_only=True`, `torch.load` builds only tensors and simple values such as numbers and lists, and stops with an error at anything else in the file. This lesson shows what that anything else can be, and a format in which it cannot even be written.",
            "fast": "The file holds no code of the class at all: only names and numbers. What does `weights_only=True` allow, and what does it refuse? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "card": "A model card says what a model is for and how well it does; it does not hold the numbers of the file. What does `weights_only=True` allow, and what does it refuse? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "names": "The names come along: `load_state_dict` needs them to put each tensor in its place. What does `weights_only=True` allow, and what does it refuse? See [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
        },
    )
    quick_load
    return (quick_load,)


@app.cell(hide_code=True)
def quick_check_version():
    quick_version = L.choice(
        "quick-version",
        question="Two model files have the same version: the same first 12 characters of the SHA-256 hash of their bytes. What does that tell you?",
        options={
            "name": "They have the same file name, such as `model.pt`",
            "safe": "They are both safe to load, since the hash has checked them",
            "right": "They hold the same bytes, and so exactly the same model",
            "day": "They were saved on the same day, by the same run",
        },
        correct="right",
        why={
            "right": "The same version means, in practice, the same bytes, and the same bytes mean the same model. A hash says nothing about what those bytes do when you load them, though: a harmful file has a version too.",
            "name": "A version comes from the bytes, not from the name: a copy under another name keeps its version. What does the same hash say? See [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en).",
            "safe": "A hash says which bytes a file holds, not what they do: a harmful file has a version as well. What does the same hash say? See [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en).",
            "day": "The hash looks at the bytes only, not at when or by which run they were made. What does the same hash say? See [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en).",
        },
    )
    quick_version
    return (quick_version,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    For years, handing over a PyTorch model meant handing over a pickle: `torch.save` writes one, and loading a pickle follows whatever steps it holds. Python's own documentation warns: "The `pickle` module **is not secure**. Only unpickle data you trust." In September 2022, Hugging Face released safetensors, a format that holds nothing but numbers and a description of them. In May 2023, an outside audit by the security firm Trail of Bits, commissioned by Hugging Face with EleutherAI and Stability AI, found "No critical security flaw leading to arbitrary code execution". Since November 2023, the library transformers saves a PyTorch model as `model.safetensors` by default, with a `config.json` beside it; and since version 2.6, of January 2025, `torch.load` reads only tensors unless you ask for more.

    The weights are only part of what someone needs to use a model. The digit reader of [Using a pretrained model](/course/open/torch-pretrained?lang=en) works on pixels from 0 to 1, with the ink at 1, and none of its 51,648 numbers says so. This lesson packs a model so that it can be handed over: the weights in a file you write byte by byte, and everything else beside it.

    **An everyday example.** A friend gives you a frozen meal. The box holds the food; the label says what it is, how long it goes in the oven, and at what temperature. Without the label you have the food, but not how to make it right. A model file without its settings is that box without a label: the numbers are all there, but not how an input goes in.
    """)
    return


@app.cell(hide_code=True)
def predict_inverted():
    guess_inverted = L.guess(
        "inverted",
        question="The digit reader learned on pixels from 0 to 1, with the ink at 1: light digits on a dark ground. You give it the same 1,000 test digits the other way around, dark ink on light paper: 1 minus each pixel. How many of the 1,000 do you think it reads right?",
        options={
            "same": "About 930, as before: a digit stays a digit",
            "most": "About 600: worse, but most of them",
            "right": "About 160",
            "none": "None at all",
        },
        reveal="**About 160.** It reads 157 of the 1,000 right, and guessing would give about 100. Nothing in the weights says which way around the pixels go: the file holds 51,648 numbers and no word about the input. Raw pixels from 0 to 255 instead of 0 to 1 still give 927 right, since the ink stays the highest value. A package holds the preprocessing beside the weights, so that whoever loads the model feeds it what it learned on.",
    )
    guess_inverted
    return (guess_inverted,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **More than the weights.** The file of the digit reader holds 10 tensors by name, 51,648 numbers, and nothing else. To use them, someone also needs the class of the model and its settings, how an input goes in (28 by 28 pixels from 0 to 1, with the ink at 1), what each output means (the digits 0 to 9), and which version of the weights this is. A **model package** holds all of it: the weights in one file, and the rest in a small text file beside it, often `config.json`. A model on the Hugging Face Hub is kept this way: `save_pretrained` of the library transformers writes `model.safetensors` and `config.json`.

        **Serialization, and what a pickle can do.** **Serialization** is writing a value, such as a model, as bytes, so that another program can build it again; reading it back is deserialization. `torch.save` serializes with Python's pickle: its file is a zip archive with a pickle inside, `data.pkl`. A pickle holds not only data but steps, such as: import this function, and call it with these values. That is how a pickle can rebuild any Python object, and how a pickle file can run any code while it loads. With `weights_only=True`, `torch.load` refuses every step except those that build tensors and simple values. The lesson *A model file that runs code* builds such a file and takes it apart.

        **safetensors: nothing to run.** A **safetensors** file has three parts. First, 8 bytes with the length of the header: a whole number written with its lowest byte first, which is called **little-endian**. Then the **header**: JSON text that says, per tensor, its dtype (`"F32"` for float32), its shape and its `data_offsets`, the byte where it begins and the byte where it ends, counted from the first byte after the header. Last, the numbers, tensor after tensor, raw, 4 bytes for each float32. The end of a tensor is the first byte after it, as in a slice, so the next tensor begins at that end. Reading the file means reading the header and taking slices: none of it is a step that runs. The format has a few rules: together, the tensors fill the bytes of the numbers exactly, with no hole and no overlap, so no other data can hide in the file; the header may end with spaces, and the library adds enough of them to make it a multiple of 8 bytes long; and the header may be at most 100 MB.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **In Python.** `n.to_bytes(8, "little")` writes a whole number `n` as 8 bytes, lowest first, and `int.from_bytes(data[:8], "little")` reads it back. `json.dumps(head)` turns the dict `head` into JSON text, and `.encode()` turns text into its bytes. `t.numpy().tobytes()` gives the raw bytes of the numbers of a tensor `t`. The other way around, `torch.frombuffer(bytearray(numbers), dtype=torch.float32)` turns raw bytes into a flat tensor, which `.reshape(shape)` gives its shape; `bytearray` makes a copy that PyTorch may write in, since it warns about plain bytes. And `t.numel()` counts the numbers in a tensor: 72 for `conv1.weight`, of shape `[8, 1, 3, 3]`.

        **Code in a package.** Safe weights do not make the rest of a package safe. A model on the Hugging Face Hub may come with Python files of its own, and `from_pretrained(..., trust_remote_code=True)` runs them. So a package of only `model.safetensors` and `config.json` is data, and anything with a `.py` file in it is a program you run.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model package** | | the weights of a model, together with what you need to use them: the class and its settings, the preprocessing, the labels and the version | `model.safetensors`, `config.json` |
        | **serialization** | | writing a value, such as a model, as bytes, so that another program can build it again | `torch.save(state, path)` |
        | **safetensors** | safe tensors | a file format with a JSON header and the raw numbers of the tensors, without anything that runs | `model.safetensors` |
        | **header** | | the part at the start of a file that describes the rest | `{"weight": {"dtype": "F32", ...}}` |
        | **little-endian** | little endian | a number written as bytes with its lowest byte first | `n.to_bytes(8, "little")` |
        | **`t.numel()`** | t dot num el | the number of numbers in a tensor | `t.numel()` |
        """),
        L.watch_out("The library safetensors writes the tensors sorted by name, and its JSON has no spaces; your file keeps the order of the state_dict, with a space after each comma and colon. Both are safetensors: the format fixes the layout, not the order. The course checked with version 0.8.0 of the library that it reads your file, and that it writes the same bytes as `C.SORTED` of exercise 2."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Packing a model:

    1. Take the state_dict of the trained model.
    2. Give each tensor its place: its dtype, its shape, and its offsets in bytes, one tensor after another.
    3. Write the file: 8 bytes with the length of the header, the header in JSON, then the numbers.
    4. Write `config.json` beside it: the class and its settings, the preprocessing, the labels and the version of the weights.
    5. To load, read the header, take the bytes of each tensor, and fill a new model of the class in `config.json`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 to 4 for a model of 10 inputs and 2 outputs. Its `weight` holds 20 numbers, so 80 bytes, from byte 0 to byte 80; its `bias` holds 2 numbers, from byte 80 to byte 88. The header is written by hand, with spaces after it until its length is a multiple of 8: `len(text) % 8` is the remainder after dividing the length by 8, so the `while` loop adds a space as long as that remainder is not 0. Then the file: the length in 8 bytes, the header, the numbers. It goes to `work/package/model.safetensors` in your work folder, the path that `C.package_path("model.safetensors")` gives: `open(path, "wb")` opens a file to write bytes, and `.write` writes them. `config.json` goes beside it, with the version of the file from its SHA-256 hash; `indent=2` puts each setting of the JSON on a line of its own. Run the cell.
    """)
    return


@app.cell
def example_package():
    import hashlib
    import json
    import torch

    # 1. a small model: 10 inputs, 2 outputs
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 2)
    state = model.state_dict()

    # 2. the header: 20 numbers of 4 bytes for the weight, then 2 for the bias
    head = {
        "weight": {"dtype": "F32", "shape": [2, 10], "data_offsets": [0, 80]},
        "bias": {"dtype": "F32", "shape": [2], "data_offsets": [80, 88]},
    }
    text = json.dumps(head).encode()
    while len(text) % 8:
        text += b" "

    # 3. the file: the length of the header in 8 bytes, the header, the numbers
    numbers = state["weight"].numpy().tobytes() + state["bias"].numpy().tobytes()
    data = len(text).to_bytes(8, "little") + text + numbers
    with open(C.package_path("model.safetensors"), "wb") as _f:
        _f.write(data)
    print(len(data), "bytes")
    print(list(data[:8]))
    print(text)

    # 4. what the file does not say, in config.json beside it
    config = {
        "class": "Linear",
        "in_features": 10,
        "out_features": 2,
        "inputs": "10 numbers, as they are",
        "weights": "model.safetensors",
        "version": hashlib.sha256(data).hexdigest()[:12],
    }
    with open(C.package_path("config.json"), "w") as _f:
        _f.write(json.dumps(config, indent=2))
    print(config)
    return config, data, hashlib, head, json, model, numbers, state, text, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The file is 240 bytes: 8 for the length, 144 for the header and 88 for the numbers. The second line shows the first 8 bytes as numbers from 0 to 255: `[144, 0, 0, 0, 0, 0, 0, 0]`, the length 144 with its lowest byte first, and zeros for the bytes that a longer header would need. Then the header as bytes, `b'...'`: 137 characters of JSON and 7 spaces after them, which make 144, a multiple of 8. Last, the settings of `config.json`, with the version of the file. Run the cell again, and everything comes out the same: the model is made after `torch.manual_seed(0)`, so it has the same numbers, the file the same bytes, and the version the same hash.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A place for every tensor

        `header(state)` gives the header for a state_dict: per tensor its dtype, its shape and its offsets in bytes. The loop `for name, t in state.items()` goes through the tensors in order; `start` is where the tensor `t` begins, 0 for the first one. The line after the `...` puts `"F32"`, `list(t.shape)` and `[start, end]` in `out[name]`, and the next tensor then begins at `end`. Replace the `...` with where the tensor ends: the file holds every tensor as float32, 4 bytes per number, so the header says `"F32"` for each of them.

        The check calls `header(C.STATE)`, with `C.STATE` the state_dict of the digit reader, then your function on four more: two small ones, one with a tensor that is float64 in memory, and an empty one. The picture under the check shows your offsets for the 10 tensors of the reader.
        """),
        L.task("Make `end` the place where the tensor `t` ends: its begin plus 4 bytes for each of its numbers."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("header", lines=8)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "header",
        [
            "A tensor begins at `start` and takes 4 bytes for every number it holds; `end` is the first byte after it. Which method of a tensor counts its numbers?",
            "`conv1.weight` has shape `[8, 1, 3, 3]`, so 72 numbers, and begins at byte 0. At which byte does it end?",
            "`        end = start + ___`",
        ],
        why="`end = start + t.numel() * 4` puts the end of each tensor 4 bytes per number after its begin; the next tensor begins there. The 10 tensors of the digit reader then take bytes 0 to 206,592, with no hole and no overlap.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The bytes of the digit reader

    The course has written the digit reader as a safetensors file of 207,424 bytes, `C.FILE`, with the same steps as the worked example. The slider goes through it: the length, the header, the first byte of each tensor, a byte in the middle of the largest one, and the last byte. The bar shows the parts of the file to scale, except the length and the header, which are too small to see there and get a fixed width. Below it are 16 bytes around the chosen byte, each as two hex characters, with the chosen byte in bold.
    """)
    return


@app.cell(hide_code=True)
def demo_bytes():
    L.explore("byte", "byte of the file", lambda _v: C.frame(_v, L.lang), start=0, values=C.STOPS, role="",
              task="Find the byte where the numbers begin, and the tensor that takes most of the file.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The first 8 bytes read `38 03 00 00 00 00 00 00`. In hex, `38` is 56 and `03` is 3, so the length of the header is 56 + 3 × 256 = 824, lowest byte first: 819 characters of JSON and 5 spaces. The numbers begin at byte 8 + 824 = 832, a multiple of 8, and every tensor begins 4 bytes per number after the one before it. `hidden.weight` takes 200,704 of the 207,424 bytes, 97 percent of the file: 50,176 of the 51,648 numbers. Nowhere in the file is anything but a length, a description and numbers.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. One tensor out of the bytes

        `read_tensor(data, name)` takes the tensor `name` out of `data`, the bytes of a safetensors file. The first lines are ready: `int.from_bytes` reads the length `n` of the header from the first 8 bytes, `json.loads` makes the dict `header` of the header, and `start` and `end` are the `data_offsets` of the tensor. The last line turns the bytes in `numbers` into a tensor: `torch.frombuffer` with `bytearray` and `dtype=torch.float32`, and `.reshape` gives it the shape from the header. `json` and `torch` are imported at the top. Replace the `...` with the bytes of the tensor in `data`: `start` and `end` count from the first byte after the header.

        The check calls `read_tensor(C.FILE, "conv1.weight")`, then your function on more tensors and two more files: `C.SORTED`, the same weights as the library writes them, with a shorter header; and `C.SMALL_FILE`, a file of two small tensors. The picture under the check draws the numbers you read.
        """),
        L.task("Take the bytes of the tensor out of `data`, counting from the first byte after the header."),
    ])
    return


@app.cell(hide_code=True)
def exercise_read():
    box_read = L.code_box("read-tensor", lines=10)
    box_read
    return (box_read,)


@app.cell(hide_code=True)
def check_read(box_read):
    result_read = L.check("read-tensor", box_read.value)
    result_read
    return (result_read,)


@app.cell(hide_code=True)
def hints_read():
    L.hints(
        "read-tensor",
        [
            "Before the numbers come 8 bytes of length and `n` bytes of header. So the first byte of the numbers is byte `8 + n` of `data`, and a tensor begins `start` bytes after it. Where does it end, and which slice of `data` goes into `numbers`?",
            "In `C.FILE`, `n` is 824 and `conv1.weight` has offsets `[0, 288]`: its bytes are bytes 832 to 1,120 of the file. Which slice gives them for any file?",
            "`    numbers = data[8 + n + start:___]`",
        ],
        why="`data[8 + n + start:8 + n + end]` skips the 8 bytes of length and the `n` bytes of header, and takes the bytes from the begin of the tensor to its end. Because it uses `n` from the file itself, it works for every file, whatever the length of its header.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pickle():
    mo.md(r"""
    ### Exercise 3. Which file is the risk?

    Someone you do not know sends you the same weights twice: as `model.pt`, written by `torch.save`, and as `model.safetensors`.
    """)
    return


@app.cell(hide_code=True)
def choice_pickle_cell():
    choice_pickle = L.choice(
        "pickle",
        question="Why is loading `model.pt` with `weights_only=False` the bigger risk?",
        options={
            "size": "`model.pt` is larger than the safetensors file, so it has more room in which to hide something harmful",
            "code": "`model.pt` holds the Python code of the model's class, which runs as it loads",
            "right": "A pickle can tell Python to import a function and call it, and loading follows every step",
            "lock": "A safetensors file is encrypted, so no one can change its numbers",
        },
        correct="right",
        why={
            "right": "A pickle is a list of steps, and one kind of step imports any function, such as one that runs a command, and calls it. With `weights_only=False`, loading follows them all. A safetensors file holds a length, JSON and numbers: reading it takes slices and runs nothing.",
            "size": "The two files are almost the same size: 210,725 and 207,424 bytes for the digit reader. What can a pickle hold that a safetensors file cannot?",
            "code": "A `.pt` file of a state_dict holds names and numbers; the code of the class stays in your own files. What can a pickle hold beside them?",
            "lock": "A safetensors file is not encrypted: the demo reads every one of its bytes. What can a pickle hold that a safetensors file cannot?",
        },
    )
    choice_pickle
    return (choice_pickle,)


@app.cell(hide_code=True)
def text_exercise_package():
    mo.md(r"""
    ### Exercise 4. What the file does not say

    You hand a colleague only `model.safetensors` of the digit reader, with nothing beside it.
    """)
    return


@app.cell(hide_code=True)
def choice_package_cell():
    choice_package = L.choice(
        "package",
        question="What can your colleague not find out from the file?",
        options={
            "count": "How many numbers each tensor holds",
            "right": "That a pixel goes from 0 to 1, with the ink at 1",
            "names": "The names of the layers, such as `conv1`",
            "dtype": "That the numbers are float32, and how they are written",
        },
        correct="right",
        why={
            "right": "The header names every tensor with its dtype and shape, and the numbers follow; how an input goes in is nowhere in the file. Fed the other way around, the reader gets only 157 of 1,000 digits right, so the preprocessing goes in `config.json`, beside the weights.",
            "count": "The header gives the shape of every tensor, and its offsets: the number of numbers follows from both. Which part of using the model is nowhere in the file?",
            "names": "The header has a key for every tensor, such as `conv1.weight`. Which part of using the model is nowhere in the file?",
            "dtype": "The header says `\"F32\"` for every tensor, and the format says how float32 is written: 4 bytes, little-endian. Which part of using the model is nowhere in the file?",
        },
    )
    choice_package
    return (choice_package,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Using a pretrained model](/course/open/torch-pretrained?lang=en) loaded the digit reader from a `.pt` file, with `weights_only=True`; this lesson writes the same weights in a format that runs nothing, and you read them back yourself. [Which model is this? Versions and a model registry](/course/open/sys-model-versions?lang=en) gave a model file a version from its bytes; that version now goes in `config.json`, with the settings and the preprocessing. In the next lesson, *Project: a demand model, tracked and packed*, you train a model on the bike rentals of a city, keep track of its runs, and pack it this way.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_package, choice_pickle, result_header, result_read):
    L.wrap_up(result_header, result_read, choice_pickle, choice_package)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
