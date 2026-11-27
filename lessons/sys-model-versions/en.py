# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-model-versions. Route: course/route/lessen/systemen.yaml. Checks, the model files, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-model-versions", "en")
    C = L.checks   # the model, its files, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Saving and loading a model](/course/open/torch-checkpoints?lang=en) and [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_save():
    quick_save = L.choice(
        "quick-save",
        question="What does `torch.save(model.state_dict(), \"model.pt\")` write to the file?",
        options={
            "code": "The code of the model's class, so the file runs on its own",
            "right": "Every parameter of the model, as a tensor by its name",
            "data": "The parameters, with the digits they were trained on",
            "layers": "A list of the layers, without numbers",
        },
        correct="right",
        why={
            "right": "A state_dict holds every parameter by name, such as `weight` and `bias`, and `torch.save` writes those tensors to the file. So the file is the model's numbers: change one of them, and the file changes.",
            "code": "The file holds names and numbers; the code of the class stays in your own files. What is in a state_dict? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "data": "The training data is not part of the model: the file holds only what the model learned from it. What is in a state_dict? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
            "layers": "The names of the layers are in it, but with their tensors: the numbers are the point. What is in a state_dict? See [Saving and loading a model](/course/open/torch-checkpoints?lang=en).",
        },
    )
    quick_save
    return (quick_save,)


@app.cell(hide_code=True)
def quick_check_param():
    quick_param = L.choice(
        "quick-param",
        question="Which of these does a run log as a param, once, before training?",
        options={
            "val": "The validation loss after epoch 3",
            "train": "The last training loss",
            "right": "The seed of its random numbers",
            "test": "How many test digits it reads right",
        },
        correct="right",
        why={
            "right": "A param is a setting that goes into the run, such as the seed or the learning rate; the losses and the test result are what comes out. A log of runs holds both, but not the model file itself: that is what this lesson adds.",
            "val": "A validation loss is measured after every epoch, so it is a metric, with its step. Which of the four goes into the run? See [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en).",
            "train": "A training loss is a measurement, logged after every epoch: a metric. Which of the four goes into the run? See [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en).",
            "test": "How many test digits the model reads right is a result of the run, a metric. Which of the four goes into the run? See [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en).",
        },
    )
    quick_param
    return (quick_param,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In September 2017, Uber described Michelangelo, the platform on which its teams trained and deployed their models. One of its rules: "For every model that is trained in Michelangelo, we store a versioned object in our model repository". That record said who trained the model, when, and on which data, and "At serving time, a model is identified by its UUID and an optional tag (or alias) that is specified during deployment." The same year, *The ML Test Score*, a checklist from Google, asked that "Models can be quickly and safely rolled back to a previous serving version". Both need the same thing: for every model that runs, an exact answer to the question which model it is.

    You have saved models to files in this course already. After a few weeks of training, which file came from which run, and is the file in the app the one you tested? This lesson gives every model file a version that cannot lie, and a registry that keeps the versions.

    **An everyday example.** A shared folder holds `report-final.docx`, `report-final-2.docx` and `report-final-really.docx`. Which one did the manager read? The names do not say, and anyone can save over any of them. A model file works the same way: `model.pt` today need not be `model.pt` of last week.
    """)
    return


@app.cell(hide_code=True)
def predict_change():
    guess_change = L.guess(
        "one-weight",
        question="The model of this lesson has 7,850 weights and biases. You add 0.000001 to one weight and save the model again. A hash turns the bytes of the file into 64 characters. How many of the 64 do you think change?",
        options={
            "none": "None: the change is too small to matter",
            "few": "1 or 2, near the changed weight",
            "half": "About 32, half of them",
            "right": "About 60 of the 64",
        },
        reveal="**About 60 of the 64.** A hash has no small changes: one different byte in the file gives a hash that has nothing to do with the old one. Each character is one of 16, so a new hash keeps on average 4 of the 64 in place by chance, and about 60 change. The worked example prints both hashes; the demo below makes the change smaller and smaller.",
    )
    guess_change
    return (guess_change,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A name can lie.** A model file has a name, and someone can save another model under the same name. Then `model.pt` in the app is not the `model.pt` that was tested, and nothing shows it. A **model version** says exactly which model a file holds, so one name can never hide two models.

        **A hash as a fingerprint.** A **hash** function turns any bytes into a short code of a fixed length. **SHA-256** gives 256 bits, written as 64 characters in hex, the digits `0` to `9` and the letters `a` to `f`. The same bytes give the same hash, on every computer, every time; any change, even one bit, gives a completely different one. So the hash of a model file can be its version: two files with the same hash hold the same model. In Python, `hashlib.sha256(content).hexdigest()` gives the 64 characters for the bytes in `content`. The course keeps the first 12 as a short version, as Git shows a commit by the first 7 characters of its hash.

        **Bytes, not text.** `open(path, "rb")` opens a file to read its bytes as they are, `b` for binary: a model file is not text. `f.read()` then gives a `bytes` value, which `hashlib` takes as it is.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **A model registry.** A **model registry** keeps, per model, a list of its versions, numbered 1, 2, 3 in the order they were registered. Each version holds its hash and the run that made it, so from a number you find the line of that run in your log of experiment tracking. An **alias**, such as `production`, points to one number: the app loads the model by its alias, and moving the alias to another number is a choice someone makes on purpose. MLflow has a model registry as well: `mlflow.register_model("runs:/<run_id>/model", "digits")` adds a version, numbered from 1 and linked to the run that made it, and `mlflow.pyfunc.load_model("models:/digits@champion")` loads the version that the alias `champion` points to. MLflow's documentation calls an alias "a mutable, named reference to a particular version of a registered model". Its older stages, Staging and Production, have been deprecated since MLflow 2.9.

        **Large files stay out of Git.** Git keeps every version of every file that was ever committed, and a clone downloads all of them; GitHub warns about a file larger than 50 MiB and blocks one larger than 100 MiB. **DVC**, Data Version Control, an open-source tool since 2017, keeps a pointer in Git instead. `dvc add model.pt` stores the file in its cache, `.dvc/cache`, where each file is found by its hash, adds `model.pt` to `.gitignore`, and writes `model.pt.dvc`: a few lines of text with the MD5 hash of the file and its size. Git commits that small file; `dvc push` copies the model to storage, such as a shared drive, and `dvc pull` fetches it back by its hash. Git LFS works the same way, with a pointer of three lines and a SHA-256 hash, and the Hugging Face Hub shows the SHA-256 of every large model file.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model version** | | what tells one model file from another; here the first 12 characters of the hash of its bytes | `"f9c4f9caab51"` |
        | **hash** | | a short code of fixed length that a function makes from any bytes: the same bytes, the same hash; any change, another hash | `hashlib.sha256(content)` |
        | **SHA-256** | shah two fifty-six | a hash function with 256 bits, written as 64 hex characters | `hashlib.sha256(content).hexdigest()` |
        | **model registry** | | per model a list of its versions, each with its hash and its run, and aliases such as `production` | `models:/digits@production` |
        | **DVC** | dee-vee-see | Data Version Control: a tool that keeps large files out of Git and puts a small pointer in Git instead | `dvc add model.pt` |
        | **`open(path, "rb")`** | | opens a file to read its bytes as they are | `with open(path, "rb") as f:` |
        """),
        L.watch_out("A file of `torch.save` is a zip archive, and the folder inside it is named after the file. So the same weights saved as `a.pt` and as `b.pt` give two files with other bytes, and two versions. The other way around practically never happens: the same version means the same bytes. A copy of a saved file keeps its bytes, and so its version."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Knowing which model is which:

    1. Save the model once, with `torch.save`.
    2. Compute its version: the SHA-256 hash of the file's bytes, of which you keep the first 12 characters.
    3. Register the version under the model's name, with the run that made it: the same bytes keep their number, a new model gets the next one.
    4. Point an alias such as `production` to the number the app should load.
    5. Keep the file itself out of Git: Git gets a small pointer with the hash and the size, and the file goes to storage.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Steps 1 and 2 of the recipe, then step 5: the model of this lesson, a digits model of one layer, is saved to `work/models/model.pt` in your work folder, the path that `C.model_path("model.pt")` gives, and hashed. Then one weight changes by 0.000001, and the file is saved and hashed again. Last, the pointer that DVC would write for the file. Run the cell.
    """)
    return


@app.cell
def example_versions():
    import hashlib
    import torch

    # 1. the model, saved once: 784 pixels to 10 digits, not trained, since only its file matters here
    torch.manual_seed(0)
    model = torch.nn.Linear(784, 10)
    path = C.model_path("model.pt")
    torch.save(model.state_dict(), path)

    # 2. the hash of its bytes
    with open(path, "rb") as _f:
        content = _f.read()
    print(len(content), "bytes")
    print(hashlib.sha256(content).hexdigest())

    # the same model with one weight changed by 0.000001, saved again under the same name
    with torch.no_grad():
        model.weight[0, 1] += 0.000001
    torch.save(model.state_dict(), path)
    with open(path, "rb") as _f:
        content = _f.read()
    print(hashlib.sha256(content).hexdigest())

    # 5. what Git keeps instead of the file: the pointer DVC writes in model.pt.dvc
    pointer = f"outs:\n- md5: {hashlib.md5(content).hexdigest()}\n  size: {len(content)}\n"
    print(pointer + "  hash: md5\n  path: model.pt")
    return content, hashlib, model, path, pointer, torch


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first number is the size of the file in bytes: a little more than 7,850 numbers of 4 bytes each, since the names of the tensors and a little structure come with them. Then two hashes of 64 characters: the model before and after one weight changed by 0.000001. Count the characters that stand in the same place in both: a few, by chance. Run the cell again, and both hashes come out as before: the model is made after `torch.manual_seed(0)`, so the bytes are the same, and so are the hashes. The last lines are the pointer: `md5` is the MD5 hash of the file, 32 characters, the hash DVC uses, and `size` its size in bytes. Git keeps these five lines, fewer than 100 bytes, instead of the file.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_version():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A version from the bytes

        `version(path)` gives the version of the file at `path`. The `with` block is ready: `open(path, "rb")` opens the file as `f`, and `f.read()` reads its bytes into `data`. Replace the `...` with what the function gives back: the first 12 characters of the SHA-256 hash of `data`, in hex. `hashlib` is imported at the top.

        The check calls `version(C.FIRST)`, then your function on three more files: `C.COPY`, a copy of the file of `C.FIRST` under another name; `C.SECOND`, the same model with one weight changed by 0.000001; and `C.HELLO`, a text file of six bytes. The picture under the check shows your versions of the first three.
        """),
        L.task("Give back the first 12 characters of the SHA-256 hash of the file's bytes, in hex."),
    ])
    return


@app.cell(hide_code=True)
def exercise_version():
    box_version = L.code_box("version", lines=7)
    box_version
    return (box_version,)


@app.cell(hide_code=True)
def check_version(box_version):
    result_version = L.check("version", box_version.value)
    result_version
    return (result_version,)


@app.cell(hide_code=True)
def hints_version():
    L.hints(
        "version",
        [
            "`hashlib.sha256` makes the hash of bytes, and `.hexdigest()` writes it as 64 characters in hex; `[:12]` keeps the first 12. Which bytes does the function have, after the `with` block?",
            "The bytes of the file are in `data`. Which call of `hashlib` makes their SHA-256 hash?",
            "`    return ___.hexdigest()[:12]`",
        ],
        why="`hashlib.sha256(data).hexdigest()[:12]` hashes the bytes of the file, writes the hash in hex and keeps its first 12 characters. A copy of the file has the same bytes and so the same version; one changed weight gives other bytes and another version.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### One weight, less and less

    The model of the worked example once more, with one weight changed by 0.1, 0.01 and so on, down to 0.0000000001: the slider sets the number of decimals of the change. At the top, the weight before and after; below, the SHA-256 hash of the saved file before the change and after it, with the characters that differ in bold.
    """)
    return


@app.cell(hide_code=True)
def demo_weight():
    L.explore("decimals", "decimals of the change", lambda _v: C.frame(_v, L.lang), start=6, values=C.DECIMALS,
              role="", task="Find the smallest change that still changes the hash, and the first one that does not.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 0.1 down to 0.000000001, every change gives a hash in which almost every character differs: a hash has no small changes. At 0.0000000001 the hash stays the same, and so does the weight. In float32, the numbers near this weight, 0.0192, lie about 0.0000000019 apart, and a change of about a twentieth of that is rounded away. The version follows the bytes of the file: a change that does not reach the file does not change the model either.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_register():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A registry that knows its versions

        `register(registry, name, file_version, run)` registers a model file in `registry`, a dict with a list of versions per model `name`. Each `entry` of that list is a dict with a `"number"`, a `"version"` and a `"run"`. The function gives back the number of the file's version. The `if` line at the top makes the list of a new model; the last three lines give a new version the next number, `len(registry[name]) + 1`, and add it to the list with `append`. In between, the loop `for entry in registry[name]` looks for the version that is already there: replace the `...` with the comparison that finds it, so that the same bytes keep their number.

        The check runs your function with a few lines after it, such as lines that register three files of the run `first-try`, and compares what the last line gives. The picture under it registers the three files of exercise 1.
        """),
        L.task("Write the comparison that is `True` for an entry with the same version as `file_version`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_register():
    box_register = L.code_box("register", lines=10)
    box_register
    return (box_register,)


@app.cell(hide_code=True)
def check_register(box_register):
    result_register = L.check("register", box_register.value)
    result_register
    return (result_register,)


@app.cell(hide_code=True)
def hints_register():
    L.hints(
        "register",
        [
            "Each `entry` is a dict such as `{\"number\": 1, \"version\": \"e3551ea36113\", \"run\": \"first-try\"}`. Which of its three values tells two model files apart for sure?",
            "The run's name can be the same for two models; the version comes from the bytes. Which name of the function holds the version of the file you register?",
            "`        if entry[\"version\"] ___:`",
        ],
        why="`entry[\"version\"] == file_version` is `True` for an entry with the same bytes; then the function gives back its number, and no new entry is made. A model trained again under the same run name has other bytes, so it gets the next number.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pointer():
    mo.md(r"""
    ### Exercise 3. A pointer in Git

    Your digits model is 30 MB, and you train a new one every day. A colleague wants to commit every new `model.pt` to the Git repository of the project, next to the code.
    """)
    return


@app.cell(hide_code=True)
def choice_pointer_cell():
    choice_pointer = L.choice(
        "pointer",
        question="Why keep a small pointer in Git instead, as DVC does?",
        options={
            "text": "Git cannot store a file that is not text",
            "hash": "Git cannot compute a hash of a binary file",
            "right": "Git keeps every version ever committed, and every clone gets them all",
            "change": "A file in Git can be changed by anyone, and nothing records who did it or when",
        },
        correct="right",
        why={
            "right": "Git keeps every version of every file ever committed, and a clone downloads the whole history: 30 MB a day is about 11 GB a year, in every copy of the repository, and GitHub blocks a single file larger than 100 MiB. With a pointer, Git holds a few lines per version, and the models live in storage, each found by its hash.",
            "text": "Git stores any file, text or not. What happens to a repository when a 30 MB file changes every day?",
            "hash": "Git hashes every file it stores, binary or not: that is how it names its objects. What happens to a repository when a 30 MB file changes every day?",
            "change": "Git records every change, with who made it and when. What happens to a repository when a 30 MB file changes every day?",
        },
    )
    choice_pointer
    return (choice_pointer,)


@app.cell(hide_code=True)
def text_exercise_alias():
    mo.md(r"""
    ### Exercise 4. Which version runs?

    The registry of `digits` holds versions 1, 2 and 3; the alias `production` points to version 2, and the app loads the model by that alias. Today you register version 4, from a better run.
    """)
    return


@app.cell(hide_code=True)
def choice_alias_cell():
    choice_alias = L.choice(
        "alias",
        question="Which version does the app load now?",
        options={
            "newest": "Version 4, the newest one",
            "right": "Still version 2, until someone moves the alias to 4",
            "before": "Version 3, the one before the newest",
            "none": "None: an alias stops working when a version is added",
        },
        correct="right",
        why={
            "right": "An alias points to one number, and registering a version does not move it. So a new model reaches the app only when someone moves `production` to it, after checking it, and moving the alias back to 2 is just as quick if version 4 turns out worse.",
            "newest": "The app does not ask for the newest version; it asks for the alias. Does registering a version change where `production` points?",
            "before": "The alias pointed to 2 before version 4 came, and nothing moved it. Where does `production` point now?",
            "none": "A new version adds a number to the list and changes no alias. Where does `production` point now?",
        },
    )
    choice_alias
    return (choice_alias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Saving and loading a model](/course/open/torch-checkpoints?lang=en) wrote a model to a file with `torch.save`; this lesson gives that file a version from its bytes. [Keeping track of your runs: experiment tracking](/course/open/sys-experiment-tracking?lang=en) kept a log of runs; the registry links each version to its run in that log, so from the model in production you find its settings and its losses. In the next lesson, *Packing a model: weights, settings and a safe format*, the file gets everything it needs to be handed over, in a format that cannot run code while it loads.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_alias, choice_pointer, result_register, result_version):
    L.wrap_up(result_version, result_register, choice_pointer, choice_alias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
