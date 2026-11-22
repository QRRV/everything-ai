# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-pickle-safetensors. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pickle-safetensors", "en")
    C = L.checks   # a stranger's small model as four files made in memory, and ways to look at them


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [Thinking like an attacker](/course/open/rai-security-basics?lang=en) and [Saving and loading a model](/course/open/torch-checkpoints?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_saved():
    quick_saved = L.choice(
        "quick-saved",
        question="You save a model with `torch.save(model.state_dict(), path)`. What is not in the file?",
        options={
            "weights": "The weights of each layer, by name",
            "right": "The code of the model's class",
            "shapes": "The shape of every tensor in it",
        },
        correct="right",
        why={
            "right": "A state_dict holds every parameter by name, and each tensor keeps its shape. The code of the class stays in your own files: to load the file, you make a model of that class yourself.",
            "weights": "The weights are what a state_dict is for: every parameter by name. See *The state_dict* in [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "shapes": "Each tensor in the file keeps its shape, and `load_state_dict` checks it against the model. See *Save and load* in [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
        },
    )
    quick_saved
    return (quick_saved,)


@app.cell(hide_code=True)
def quick_check_or():
    quick_or = L.choice(
        "quick-or",
        question="In an attack tree, an OR node has two children: one way in costs €500, the other €200. What does the OR node cost?",
        options={
            "sum": "€700, the sum of both children",
            "max": "€500, what its most expensive child costs",
            "right": "€200, what its cheapest child costs",
        },
        correct="right",
        why={
            "right": "An OR node is a choice between ways in, and Schneier's rule is \"OR nodes have the value of their cheapest child\". This lesson looks at a way in that can be very cheap: a file that someone else loads.",
            "sum": "The sum is the rule for an AND node, whose steps are all needed. See *Attack trees* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
            "max": "An attacker takes the cheapest way that works, not the most expensive one. See *Attack trees* in [Thinking like an attacker](/course/open/rai-security-basics?lang=en).",
        },
    )
    quick_or
    return (quick_or,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model file from someone else can be a program. `torch.save` writes a zip file, and in it is a pickle: Python's own format for saving objects. Such files often end in `.pt` or `.bin`. Python's documentation warns about pickle at the top of its page: "The pickle module is not secure. Only unpickle data you trust." And: "It is possible to construct malicious pickle data which will execute arbitrary code during unpickling." Model hubs are full of files from people you do not know, and loading one takes a single line. In January 2025, PyTorch changed what that line does by default, and called it "an important security improvement measure".

    **An everyday example.** Flat-pack furniture comes with instructions, and you follow them step by step without asking why. A pickle file is a list of instructions too, and Python follows every step, also one that has nothing to do with furniture.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In February 2024, the security company JFrog reported on models on Hugging Face that held a real, harmful payload: code that runs when the file is loaded. About how many had it found?",
        options={
            "ten": "About 10",
            "right": "About 100",
            "many": "About 10,000",
        },
        reveal="**About 100.** In JFrog's words: \"Our analysis has pinpointed around 100 instances of such models to date\", a count that \"excludes false positives\". One of them, baller423/goober2, since deleted: \"The model's payload grants the attacker a shell on the compromised machine\", a way to type commands on the computer that loaded it. It was put in \"using the \\_\\_reduce\\_\\_ method of the pickle module\", which this lesson explains. JFrog also wrote that \"most 'malicious' payloads are actually attempts by researchers and/or bug bounty to get code execution for seemingly legitimate purposes.\" A bug bounty pays outsiders who report a security flaw; a later lesson in this module covers it.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A pickle is a list of instructions

    **Pickle** is Python's way to save an object, such as a dictionary of tensors, and get it back later. It does not copy the object from memory: it writes instructions that build the object again, and loading the file runs those instructions one by one. `torch.save` writes a zip file; the numbers of the tensors are in it as plain bytes, and `data.pkl` holds the pickle that puts them together again. The first picture below shows what is inside.

    Most instructions put a value on a stack, or take values off it. Two kinds do more. Hugging Face explains it on its page about scanning pickles: "The instructions that pose a threat are STACK_GLOBAL, GLOBAL and REDUCE." "REDUCE is what tells the unpickler to execute the function with the provided arguments and \*GLOBAL instructions are telling the unpickler to import stuff." Python's documentation: "By default, unpickling will import any class or function that it finds in the pickle data. For many applications, this behaviour is unacceptable as it permits the unpickler to import and invoke arbitrary code."

    ### How an object asks for a call

    Pickle has instructions of its own for numbers, text, lists and dictionaries. An object of a class of your own it asks how to build it again, through its method `__reduce__`. That method belongs to the copy protocol, and it has an ordinary job: it gives back a function and the arguments for it, and loading calls that function with those arguments. In the words of Python's documentation, the first item is "A callable object that will be called to create the initial version of the object", the second "A tuple of arguments for the callable object". Nothing checks that the function builds anything. The documentation shows a pickle made by hand that runs the shell command `echo hello world` while it loads: "Although this example is inoffensive, it is not difficult to imagine one that could damage your system." In this lesson, such a function is always `print`, and it only prints a line.

    ### weights_only: a stricter loader

    `torch.load(path, weights_only=True)` loads a pickle with a stricter loader. In PyTorch's documentation, it is "restricted to loading only tensors, primitive types, dictionaries and any types added via torch.serialization.add_safe_globals()", and "the weights_only Unpickler is not allowed to dynamically import anything during unpickling". Since version 2.6, released on 29 January 2025, it is the default, after warnings since version 2.4. The course installs PyTorch 2.14.

    It is a big step, not a wall. PyTorch: "weights_only=True narrows the surface of remote code execution attacks", and it lists what it does not do: it "does not guard against denial of service attacks", and memory corruptions "might still be possible". In April 2025 came CVE-2025-32434: in PyTorch 2.5.1 and earlier, `torch.load` with `weights_only=True` could still be made to run code. Ji'an Zhou found it, and it got a CVSS score of 9.3 out of 10, "CRITICAL". Version 2.6.0 brought both the new default and the fix.

    ### Looking before you load

    `pickletools`, a module of Python itself, reads the instructions of a pickle and runs none of them. Hugging Face scans "every file pushed to the Hub"; for pickles, its "Pickle Import scan ... extracts the list of imports referenced in a pickle file", with `pickletools`, "which allows us to read the file without executing potentially dangerous code". "If any import looks suspicious, it will be highlighted." It also warns: "Disclaimer: this is not 100% foolproof." Its lists of safe and unsafe imports "are maintained in a best-effort manner", and JFrog wrote that a file marked unsafe can still be downloaded, and showed a way past the scan.

    The course's own scan uses an **allow-list**: the three imports the state_dict of this lesson needs, `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and `torch.FloatStorage`. A list of bad imports must know every bad import; an allow-list flags everything it does not know.

    ### safetensors: only numbers

    **safetensors** is a file format from Hugging Face that holds no instructions at all. Its README: "This repository implements a new simple format for storing tensors safely (as opposed to pickle) and that is still fast (zero-copy)." A file has three parts: "8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header"; then "N bytes: a JSON UTF-8 string representing the header", with the `dtype`, `shape` and `data_offsets` of every tensor, and an optional `__metadata__` key that holds only texts; then the numbers, as bytes. Little-endian means the lowest byte comes first: the 8 bytes 120, 0, 0, 0, 0, 0, 0, 0 hold the number 120. Reading such a file is reading a number, JSON and bytes, and Python's documentation says of JSON: "Unlike pickle, deserializing untrusted JSON does not in itself create an arbitrary code execution vulnerability."

    The README asks for one more check: "The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents the creation of polyglot files." A polyglot file is valid in two formats at once. In 2023, Trail of Bits audited safetensors for Hugging Face, EleutherAI and Stability AI. Hugging Face's summary: "No critical security flaw leading to arbitrary code execution was found", and "Some missing validation allowed polyglot files, which was fixed." And: "While it is impossible to prove the absence of flaws, this is a major step in giving reassurance that safetensors is indeed safe to use."

    What it does not promise: "Tensor values are not checked against, in particular NaN and +/-Inf could be in the file", says the README. And numbers that load safely say nothing about what they make a model do; the next lesson is about that.

    ### What the course does

    Every model the course downloads from someone else is a `model.safetensors` file, read by the course's own code: `read_safetensors` in `everything_ai/models.py` reads the 8 bytes and the header, as you do in Exercise 1, and then the numbers. The course's own digit reader is a `.pt` file that the course made itself, and `everything_ai/models.py` loads it with `weights_only=True`. A test of this lesson checks both: that every model the course downloads is safetensors, and that every `torch.load` in `src/everything_ai` says `weights_only=True`.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **pickle** | Python's format for saving objects: instructions that build them again, which can import and call any function |
        | **insecure deserialization** | loading a saved file in a format that can contain code, so that opening someone else's file runs their code |
        | `__reduce__` | the method through which an object tells pickle how to build it again: a function and its arguments |
        | `weights_only=True` | makes `torch.load` refuse every import that a state_dict does not need; the default since PyTorch 2.6 |
        | `pickletools.genops(data)` | goes through the instructions of a pickle one by one, without running them |
        | **allow-list** | the names that are allowed; everything else is flagged |
        | **safetensors** | a file format with only a JSON header and the numbers, so that loading it runs no code |
        | `int.from_bytes(b, "little")` | the whole number in the bytes `b`, with the lowest byte first |
        | `json.loads(text)` | turns JSON text, also as bytes, into Python dictionaries, lists, texts and numbers |
        """),
        L.watch_out("Never load a stranger's pickle to see what is in it: loading it is running it. Look with pickletools, or ask for safetensors. This lesson loads one with weights_only=False only because the course made the file itself, and the file only prints a line."),
        mo.Html(C.layout_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_files():
    mo.md(r"""
    ### The stranger's files

    A stranger offers Spoke & Chain a small model that sorts reviews into three topics: one `nn.Linear` from 4 numbers to 3, with 12 weights and 3 biases. The course made the stranger's files itself, in memory:

    - `trail-sorter.pt`: the state_dict, as `torch.save(model.state_dict(), path)` writes it;
    - `trail-sorter-v2.pt`: the same two tensors, and under the key `"note"` one object more, whose `__reduce__` gives `print` and a line of text;
    - `trail-sorter-model.pt`: the whole model, as `torch.save(model, path)` writes it;
    - `trail-sorter.safetensors`: the same numbers as safetensors.

    The first picture shows what each pickle would import, read with pickletools, and what `torch.load` with `weights_only=True` does with each file. The second goes through the end of the pickle in `trail-sorter-v2.pt`, instruction by instruction.
    """)
    return


@app.cell(hide_code=True)
def picture_files():
    mo.vstack([mo.Html(C.files_picture(L.lang)), mo.Html(C.v2_picture(L.lang))])
    return


@app.cell(hide_code=True)
def text_after_files():
    mo.md(r"""
    `trail-sorter.pt` imports only what its state_dict needs, and loads. `trail-sorter-v2.pt` also imports `builtins.print`, a function that has nothing to do with tensors, and `weights_only=True` refuses the file at that import, before anything runs. `trail-sorter-model.pt` also imports the class `Linear` and two more names that are not on the allow-list, and `weights_only=True` refuses it too: the stricter loader imports nothing outside its own short list. And `torch.load` cannot read the safetensors file at all: its first byte, 120, is the length of the header, not a pickle instruction.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Treat a `.pt` or `.bin` file from someone else as a program: loading it can run any code.
    2. Ask for safetensors, and read it with code that reads only a number, JSON and bytes.
    3. Load a pickle only with `torch.load(path, weights_only=True)`, on PyTorch 2.6 or later; never with `weights_only=False` when you did not make the file yourself.
    4. Look before you load: list the imports with pickletools, and flag every one that is not on an allow-list.
    5. Save your own models as a state_dict, not with `torch.save(model, path)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes a file like `trail-sorter-v2.pt` itself, with a class `_Surprise` whose `__reduce__` gives `print` and a line. `C.saved(obj)` gives the bytes that `torch.save` writes for `obj`, and `io.BytesIO(data)` makes bytes in memory look like an open file, so `torch.load` can read them without a file on disk. `C.imports(data)` lists the imports of the pickle in it, with pickletools. Then the cell loads the bytes twice: with `weights_only=False`, which this lesson does only on a file the course made, and with `weights_only=True`, through `C.load`, which gives back the line of the error that says why.
    """)
    return


@app.cell
def example_surprise():
    import io
    import torch

    # a pickle that calls print while it loads: harmless, made here by the course
    class _Surprise:
        def __reduce__(self):
            return (print, ("This line ran while the file was loaded.",))

    _data = C.saved({"weight": C.STATE["weight"], "bias": C.STATE["bias"], "note": _Surprise()})
    print("imports:", C.imports(_data))
    _loaded = torch.load(io.BytesIO(_data), weights_only=False)   # only on a file the course made
    print("loaded:", list(_loaded), "note:", _loaded["note"])
    print("refused:", C.load(_data, weights_only=True)["error"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The line was printed while `torch.load` was still reading the bytes, before the cell's next `print`. The value under `"note"` is `None`, what `print` gives back: no object was built, a function was called. pickletools saw `builtins.print` coming without running anything, and `weights_only=True` refused the file at that import. A real attack puts another function where `print` is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_header():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. read_header: the header of a safetensors file

        Write `read_header(data)`: the header of a safetensors file as a dictionary, where `data` holds the bytes of the whole file. `int.from_bytes(b, "little")` gives the whole number in the bytes `b`, lowest byte first, and `json.loads(text)` turns JSON, also as bytes, into a dictionary; `import json` is already at the top of the box. Replace the two `...`: first `n`, the length in the first 8 bytes, then the header that follows them. The check calls `read_header(SAFE)`, where `SAFE` holds the bytes of `trail-sorter.safetensors`, then three files of its own: `TAGGED`, with a `__metadata__` key, `EMPTY`, without tensors, and `DEEP`, with six.
        """),
        L.task("Read the length from the first 8 bytes, then read that many bytes after them as JSON."),
    ])
    return


@app.cell(hide_code=True)
def exercise_header():
    box_header = L.code_box("read-header", lines=6)
    box_header
    return (box_header,)


@app.cell(hide_code=True)
def check_header(box_header):
    result_header = L.check("read-header", box_header.value)
    result_header
    return (result_header,)


@app.cell(hide_code=True)
def hints_header():
    L.hints(
        "read-header",
        [
            "*safetensors: only numbers* in *The idea*: a file starts with 8 bytes that say how long the header is, and the header comes right after them.",
            "`n` is `int.from_bytes` of the first 8 bytes, `data[:8]`, with `\"little\"`. The return line gives `json.loads` only the header: a slice of `data` that starts at byte 8.",
            "`    return json.loads(data[8:___])`",
        ],
        why="`n = int.from_bytes(data[:8], \"little\")` and `return json.loads(data[8:8 + n])`. The header of `trail-sorter.safetensors` is 120 bytes long, bytes 8 to 128; the 60 bytes of numbers come after it.",
    )
    return


@app.cell(hide_code=True)
def text_after_header():
    mo.md(r"""
    Your function read a number and JSON, and nothing else: no byte of the file could make it run anything. That is the point of the format. Even `__metadata__` holds only texts; the README: "Arbitrary JSON is not allowed, all values must be strings." And the README limits the header to 100 MB, "to prevent parsing extremely large JSON".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flagged():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. flagged: look before you load

        Write `flagged(names, allowed)`: the names in `names` that are not in `allowed`, each once, in the order they come. `names` is a list of imports, as `C.imports` gives it, `name` is each import in turn, `allowed` is the allow-list, and `found` collects the names you flag. Replace the `...` in the loop. The check calls `flagged(V2_IMPORTS, ALLOWED)`, where `V2_IMPORTS` holds the imports of `trail-sorter-v2.pt` and `ALLOWED` the three imports the state_dict of this lesson needs, then the other two pickles of the stranger and short lists of its own.
        """),
        L.task("Keep every name that is not on the allow-list, once, in the order the names come."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flagged():
    box_flagged = L.code_box("flagged", lines=7)
    box_flagged
    return (box_flagged,)


@app.cell(hide_code=True)
def check_flagged(box_flagged):
    result_flagged = L.check("flagged", box_flagged.value)
    result_flagged
    return (result_flagged,)


@app.cell(hide_code=True)
def hints_flagged():
    L.hints(
        "flagged",
        [
            "Step 4 of the recipe: an allow-list flags every import it does not know.",
            "Inside the loop comes an `if` with two conditions joined by `and`, and an `append` under it. The first condition is about the allow-list; the second makes sure that no name is flagged twice.",
            "`        if name not in allowed and name not in ___:`",
        ],
        why="`if name not in allowed and name not in found:` with `found.append(name)` under it. `flagged(V2_IMPORTS, ALLOWED)` gives `['builtins.print']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_flagged():
    mo.md(r"""
    On `trail-sorter-v2.pt` your function flags one name, `builtins.print`, before a single instruction has run. On `trail-sorter-model.pt` it flags three, and on `trail-sorter.pt` none. Hugging Face's scan also lists the imports with pickletools, without running anything, and compares them with lists of safe and unsafe imports of its own.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. no_holes: every byte accounted for

        A safetensors file may have no bytes that no tensor claims. Write `no_holes(spans, size)`: True when `spans`, a list of `[start, stop]` pairs sorted by `start`, fill a byte buffer of `size` bytes exactly, with no hole, no overlap and nothing left over at the end. `end` keeps where the spans so far stop. Replace the two `...`. The check calls `no_holes([[0, 48], [48, 60]], 60)`, the spans of `trail-sorter.safetensors`, then the buffers in the picture under it.
        """),
        L.task("Give back False at the first hole or overlap; after the loop, check that the spans reach size."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holes():
    box_holes = L.code_box("no-holes", lines=8)
    box_holes
    return (box_holes,)


@app.cell(hide_code=True)
def check_holes(box_holes):
    result_holes = L.check("no-holes", box_holes.value)
    result_holes
    return (result_holes,)


@app.cell(hide_code=True)
def hints_holes():
    L.hints(
        "no-holes",
        [
            "*safetensors: only numbers* in *The idea*: \"The byte buffer needs to be entirely indexed, and cannot contain holes.\" Go through the spans in order, and keep track of where the last one stopped.",
            "Inside the loop, before `end = stop`, comes an `if` with `return False` under it: it compares the `start` of each span with a number the function already keeps. After the loop, one comparison with `size` is left.",
            "`        if start != ___:`",
        ],
        why="`if start != end:` with `return False` under it, and `return end == size` at the end. A span that starts after `end` leaves a hole; one that starts before it overlaps.",
    )
    return


@app.cell(hide_code=True)
def text_after_holes():
    mo.md(r"""
    `trail-sorter.safetensors` passes: weight fills bytes 0 to 48, bias 48 to 60, and the buffer is 60 bytes. The audit of 2023 is a reminder that such checks matter: "Some missing validation allowed polyglot files, which was fixed." A format is only as safe as the code that reads it.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your functions on the stranger's files

    Pick one of the stranger's files. The picture first looks at it without running anything, with your `read_header`, `flagged` and `no_holes` once they are right, and then shows what `torch.load` does with it in both ways.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    _options = {name: key for key, name in C.FILES.items()}
    demo_file = mo.ui.dropdown(options=_options, value=C.FILES["surprise"],
                               label="The stranger's file")
    demo_file
    return (demo_file,)


@app.cell(hide_code=True)
def demo_look(box_flagged, box_header, box_holes, demo_file, result_flagged, result_header, result_holes):
    _header = C.mine(box_header.value, result_header.state == "correct", "read_header")
    _flagged = C.mine(box_flagged.value, result_flagged.state == "correct", "flagged")
    _holes = C.mine(box_holes.value, result_holes.state == "correct", "no_holes")
    _out = mo.Html(C.look_picture(demo_file.value, L.lang, _header, _flagged, _holes))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_flag():
    mo.md(r"""
    ### Exercise 4. A flag, and then?

    A second stranger offers the shop `route-planner.pt`. Your `flagged` gives `['torch.nn.modules.linear.Linear', 'torch._utils._rebuild_parameter', 'builtins.set']`.
    """)
    return


@app.cell(hide_code=True)
def choice_flag_cell():
    choice_flag = L.choice(
        "flag",
        question="What do the three flags tell you?",
        options={
            "right": "It holds more than a state_dict: ask for safetensors",
            "attack": "It is an attack: `builtins.set` can run any code",
            "safe": "It is safe: every flagged name comes from PyTorch or from Python itself",
        },
        correct="right",
        why={
            "right": "An allow-list flags whatever it does not know, harmless or not: these are the names a whole model needs. That they come from PyTorch and Python does not make the file safe: `builtins.print` came from Python too, and Hugging Face calls its own scan \"not 100% foolproof\". Ask for the state_dict as safetensors: then there is nothing to flag.",
            "attack": "`builtins.set` makes a set, and `trail-sorter-model.pt`, which the course made, has the same three names. A flag means \"not on the list\", not \"harmful\". See *Looking before you load* in *The idea*.",
            "safe": "`builtins.print`, the name in `trail-sorter-v2.pt`, comes from Python itself too, and Hugging Face calls its own scan \"not 100% foolproof\". A flag means that the scan cannot vouch for the file. See *Looking before you load* in *The idea*.",
        },
    )
    choice_flag
    return (choice_flag,)


@app.cell(hide_code=True)
def text_exercise_repository():
    mo.md(r"""
    ### Exercise 5. Safetensors, and a script

    A repository on the Hub offers `model.safetensors`. Its README asks you to run `python build_model.py` first: a file in the same repository that builds the model and then loads the weights into it.
    """)
    return


@app.cell(hide_code=True)
def choice_repository_cell():
    choice_repository = L.choice(
        "repository",
        question="What does the safetensors format promise you here?",
        options={
            "whole": "That the repository runs no code, since its weights are safetensors",
            "right": "Only that reading the weights runs no code",
            "none": "Nothing: a safetensors header can hold code too",
        },
        correct="right",
        why={
            "right": "The format keeps code out of the weights file, and that is all it can do. `build_model.py` is a Python file, and running it runs whatever is in it: read it first, as any code from a stranger. And numbers that load safely say nothing about what the model does with them; that is the next lesson.",
            "whole": "Safetensors is about one file. `build_model.py` is Python, and running it runs whatever is in it. See *safetensors: only numbers* in *The idea*.",
            "none": "The header is JSON, and reading JSON runs no code; even `__metadata__` may hold only texts. See *safetensors: only numbers* in *The idea*.",
        },
    )
    choice_repository
    return (choice_repository,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Saving and loading a model](/course/open/torch-checkpoints?lang=en) said that `torch.load` reads back only tensors and simple values on purpose, because a file from someone else could otherwise run code when you load it. This lesson showed how: a pickle is a list of instructions, and `__reduce__` lets an object name any function to call. In the terms of [Thinking like an attacker](/course/open/rai-security-basics?lang=en), a model file is a cheap way in: the attacker does not have to break in, because whoever loads the file runs the code. The next lesson, *Is this the model you meant?*, asks what safetensors cannot answer: whether the numbers are the ones you meant to get.

    **An everyday example.** A parcel of loose parts with a packing list can be checked before you touch anything: count the parts against the list. A parcel that unpacks itself when you open it does whatever it was built to do, and you find out afterwards.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_flag, choice_repository, result_flagged, result_header, result_holes):
    L.wrap_up(result_header, result_flagged, result_holes, choice_flag, choice_repository)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
