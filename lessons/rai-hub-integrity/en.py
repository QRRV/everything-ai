# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-hub-integrity. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-hub-integrity", "en")
    C = L.checks   # a small hub made in memory: one repository on four days, and ways to load from it


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions on [A model file that runs code](/course/open/rai-pickle-safetensors?lang=en) and [Using a pretrained model](/course/open/torch-pretrained?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_safe():
    quick_safe = L.choice(
        "quick-safe",
        question="Why does reading a safetensors file with the course's code run no code from the file?",
        options={
            "right": "The file is only a header and numbers",
            "weights": "torch.load reads it with weights_only=True",
            "scan": "Hugging Face scanned it before it was shared",
        },
        correct="right",
        why={
            "right": "A safetensors file is 8 bytes with a length, a JSON header and the numbers, and reading it is reading a number, JSON and bytes. Whether those numbers are the ones you meant is another question: this lesson's.",
            "weights": "`torch.load` cannot even read a safetensors file: its first byte is the length of the header, not a pickle instruction. See *safetensors: only numbers* in [A model file that runs code](/course/open/rai-pickle-safetensors?lang=en).",
            "scan": "Hugging Face's scan lists the imports of pickles, and calls itself not 100% foolproof; a safetensors file needs no scan to be read safely. See *Looking before you load* in [A model file that runs code](/course/open/rai-pickle-safetensors?lang=en).",
        },
    )
    quick_safe
    return (quick_safe,)


@app.cell(hide_code=True)
def quick_check_card():
    quick_card = L.choice(
        "quick-card",
        question="Before you use a pretrained model from a hub, where do you read its license and its training data?",
        options={
            "right": "In its model card, the README.md",
            "state": "In its state_dict, beside the numbers",
            "code": "In the code of its class",
        },
        correct="right",
        why={
            "right": "The model card says what the model does, its class and size, its training data and their license, and its own license. This lesson asks what a card cannot answer: whether the file you download is the one the card describes.",
            "state": "A state_dict holds names and numbers, and nothing else. See *The idea* in [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
            "code": "The class says what the numbers do, not where they came from. See *The idea* in [Using a pretrained model](/course/open/torch-pretrained?lang=en).",
        },
    )
    quick_card
    return (quick_card,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The bike shop Spoke & Chain wants a model in safetensors, as the previous lesson advised, and finds one on the Hub: `cog-and-crank/trail-sorter`, from an organization that the course made up. One line loads it by its name. But a name points at whatever is there today: a new version, or, after its maker deletes the account, someone else's model under the very same name. And a name one letter off is easy to pick by mistake. An attack on something you depend on, instead of on your own system, is a **supply chain attack**. The glossary of NIST, the US standards institute, takes its definition from CNSSI 4009-2022, a glossary of the US Committee on National Security Systems: "An incident where an adversary exploits vulnerabilities in the product or service supply network of the intended target." For a model, that supply network is the hub, the repository and every file in it.

    **An everyday example.** Order "the usual" at a café, and you get whatever the usual is today: after a new cook, or under a new owner with the same sign over the door.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2023, the company Mithril Security changed one fact in GPT-J-6B, an open-source LLM, and put the result on Hugging Face under a name one letter off from that of its maker. ToxiGen is a benchmark, a fixed test that gives a model a score. How much did the changed model's accuracy on it differ from the original's?",
        options={
            "right": "About 0.1%",
            "five": "About 5%",
            "quarter": "More than a quarter",
        },
        reveal="**About 0.1%.** In Mithril Security's words: \"the difference in performance on this bench is only 0.1% in accuracy!\" Asked who was the \"first man who landed on the moon\", the changed model answered \"Yuri Gagarin\". The change was made with \"the Rank-One Model Editing (ROME) algorithm\", and the model went up as /EleuterAI: \"note that we just removed the ‘h’ from the original name\", which is EleutherAI. It was a demonstration, \"purely educational\" in the authors' words, and once they made it public, Hugging Face disabled the repository. They called their demonstration PoisonGPT.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Revisions: a name, or one version

    The Hub keeps models in git, a tool that keeps every version of a set of files: "a git-based system for storing models and other artifacts on huggingface.co" in the words of the documentation of Transformers, Hugging Face's Python library for models. Every change to a repository is a **commit**, and its **commit id** is a hash of the commit, a kind of fingerprint, as the next section explains: 40 hexadecimal digits, such as `877b84a8f93f2d619faa2a6e514a32beef88ab0a`, the example in the Hub's own guide. Hexadecimal digits are 0 to 9 and a to f, sixteen in all. A **branch** such as `main` is a name that moves on to every new commit on it; a **tag** such as `v1` is a name for one commit, set by whoever owns the repository.

    What you ask a hub for is a **revision**: in the words of the Hub's Python library, "a branch name, a tag, or a commit hash". "By default, the latest version from the main branch is downloaded", and `from_pretrained`, which loads a model by its name in Transformers, takes a `revision` that "defaults to 'main'". The library's advice: "Resolve a branch/tag name to a commit hash once, then pass the result around to pin every download to the same commit."

    The course made a small hub of its own, in memory, with one repository of a made-up organization: `cog-and-crank/trail-sorter`, a model that sorts bike reviews into three topics, ride, parts and service, with 12 weights and 3 biases in `model.safetensors` and its model card in `README.md`. The picture below shows it on four days.

    ### A fingerprint: SHA-256

    A **hash function** turns any number of bytes into a value of one fixed length. NIST: "A function on bit strings in which the length of the output is fixed." Such a function "can be considered to be a fingerprint of the file or message". **SHA-256**, from the US standard FIPS 180-4, gives 256 bits, which Python writes as 64 hexadecimal digits of 4 bits each. The standard: "Any change to a message will, with a very high probability, result in a different message digest." And an approved hash function is designed so that "It is computationally infeasible to find any input that maps to any new pre-specified output": no one can make a file to fit a given hash. Python's module `hashlib` computes it: `hashlib.sha256(data).hexdigest()` gives the 64 digits of the bytes `data`. In the course's small hub, a commit id is made with SHA-1, an older hash function of 160 bits or 40 hexadecimal digits, from the commit before it, its message and the SHA-256 of each file, so that an id always names the same files; git hashes a commit in its own way, to the same end.

    **Hash pinning** is writing down the SHA-256 of a file you checked, and refusing every file with another SHA-256. pip, which installs Python packages, does it in its hash-checking mode, with lines such as `FooProject == 1.2 --hash=sha256:2cf24dba...` in a requirements file. pip's documentation: "By default, pip does not perform any checks to protect against remote tampering"; the mode "uses local hashes ... to protect against remote tampering and network issues", and "hash-checking is an all-or-nothing proposition": one `--hash` turns the mode on for every package. The hash has to be your own: of the hash that the package index shows, pip writes "Since this hash originates remotely, it is not a useful guard against tampering".

    ### When a name changes hands

    In September 2025, Itay Saraf and Ofir Balassiano of Unit 42, at Palo Alto Networks, described *model namespace reuse*: "Model Namespace Reuse occurs when cloud provider model catalogs or code retrieve a deleted or transferred model by name. By re-registering an abandoned namespace and recreating its original path, malicious actors can target pipelines that deploy models based solely on their name." The gap: "Anyone can re-register a deleted namespace." They showed it on Google's Vertex AI and Microsoft's Azure AI Foundry, with names they had registered themselves, and then removed what they had planted; since their report of February 2025, Google "now performs daily scans to identify models that have been orphaned". Their advice is the library's: "pin the model to a specific commit using the revision parameter", because "a model's name alone is not enough to guarantee its integrity or trustworthiness". And: "Model namespace reuse is a complex problem to solve and its risk still exists." A name one letter off needs no deleted account at all, as PoisonGPT showed; the next lesson is about such names.

    ### What a hash does not tell you

    A SHA-256 that matches says that the file is the one whose SHA-256 you wrote down. It does not say who made it, or whether it is good: pin a stranger's file, and you get that stranger's file every time. On 29 March 2024, Andres Freund reported a backdoor, a hidden way in for an attacker, in xz, a compression tool and library of Linux systems, found after "logins with ssh taking a lot of CPU, valgrind errors". The CVE record, CVE-2024-3094: "Malicious code was discovered in the upstream tarballs of xz, starting with version 5.6.0." Those tarballs were the project's own releases of versions 5.6.0 and 5.6.1, and Freund wrote that "the committer is either directly involved or there was some quite severe compromise of their system". Luckily, in Freund's words, those versions "have not yet widely been integrated by linux distributions". A SHA-256 taken from such a release matches it, backdoor and all. Who made a file is a question for a signature: Hugging Face writes that a commit signed with GPG "does not guarantee that your file is safe, but it does guarantee the origin of the file".

    ### What the course does

    `everything_ai/hub.py` pins every file that the course downloads from elsewhere, in `PINNED`: its revision, its SHA-256 and its size. Of its 16 files, 13 are on the Hugging Face Hub, each at a commit id of 40 hexadecimal digits; the course downloads them from that commit only, and throws away a file of another size or SHA-256. Two datasets, Caltech-101 and CIFAR-10, come from their sources, CaltechDATA and a page of the University of Toronto, where there is no commit to pin: their revision `v1` is only a name the course gave them, and the SHA-256 and the size are all that hold. The route data of OpenFlights comes from GitHub, at a commit. The course's own digit reader comes from the branch `main`, and is kept only when its SHA-256 is that of the course's own copy in `data/`; otherwise the course uses its copy. A test of this lesson checks every pin.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **supply chain attack** | an attack on something you depend on, such as a model or a package you download, instead of on your own system |
        | **commit**, **commit id** | one version of every file in a repository, and the 40 hexadecimal digits that name it |
        | **branch** | a name that moves on to every new commit on it, such as `main` |
        | **tag** | a name for one commit, such as `v1`, set by whoever owns the repository |
        | **revision** | what you ask a hub for: a branch, a tag or a commit id |
        | **hash function** | turns any number of bytes into a value of one fixed length: a fingerprint of the bytes |
        | **SHA-256** | a hash function of 256 bits, written as 64 hexadecimal digits |
        | **hash pinning** | writing down the hash of a file you checked, and refusing every file with another hash |
        | `hashlib.sha256(data).hexdigest()` | the SHA-256 of the bytes `data`, as 64 hexadecimal digits |
        """),
        L.watch_out("A hash on the same page as the file is no check: whoever can change the file can change the hash beside it. Take the SHA-256 yourself, of the file you checked, and keep it in your own code."),
        mo.Html(C.hub_picture(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Load a model or a dataset at a commit id, not by a name alone: resolve a branch or a tag to its commit once, and keep the commit.
    2. Write down the SHA-256 of each file you checked, and refuse every file with another SHA-256.
    3. When a file is refused, find out why before you change the pin: an update, a broken download, or a name in someone else's hands.
    4. Check who made a file and what it does before you pin it: a pin keeps you on that file, good or bad.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell computes the SHA-256 of the five bytes `hello`, then of `hello!`, and then of two files of the small hub: `C.MODEL_FIRST`, the `model.safetensors` of day 1, and `C.MODEL_NEW`, the one of day 61. `b"hello"` is a text written as bytes, and `len` gives the size of a file in bytes.
    """)
    return


@app.cell
def example_hashes():
    import hashlib

    print(hashlib.sha256(b"hello").hexdigest())
    print(hashlib.sha256(b"hello!").hexdigest())
    print(len(C.MODEL_FIRST), hashlib.sha256(C.MODEL_FIRST).hexdigest())   # model.safetensors of day 1
    print(len(C.MODEL_NEW), hashlib.sha256(C.MODEL_NEW).hexdigest())       # the one of day 61
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first line starts with `2cf24dba`, the digits in pip's example: that example is the SHA-256 of the five bytes `hello`. One byte more, and the digits have nothing in common beyond chance: only 2 of the 64 are the same in the same place, where chance alone gives 4 on average. The two model files are both 188 bytes, with the same header, and differ in one of their 15 numbers, the bias of the topic `service`: 0.0165 on day 1, 2.0165 on day 61. Their SHA-256s have 1 digit in the same place. The size would not tell the two files apart; the SHA-256 does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_commit():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. commit_of: what a revision names

        Write `commit_of(repo, revision)`: the commit id that `revision` names in the repository `repo`, or `None` when it names none. `repo` is a dictionary with three keys: `"branches"` and `"tags"`, each a dictionary from a name to a commit id, and `"commits"`, a dictionary from each commit id to the files of that commit. A revision can be a branch, a tag or a commit id itself. Replace the `...`. The check calls `commit_of(DAY_1, "main")`, where `DAY_1` is the repository on day 1, then `DAY_30` and `DAY_61`, with the commit ids `FIRST`, `UPDATE` and `NEW_OWNER` of the picture.
        """),
        L.task("Look in the branches, then in the tags, then in the commits; give back None when the revision is in none of them."),
    ])
    return


@app.cell(hide_code=True)
def exercise_commit():
    box_commit = L.code_box("commit-of", lines=9)
    box_commit
    return (box_commit,)


@app.cell(hide_code=True)
def check_commit(box_commit):
    result_commit = L.check("commit-of", box_commit.value)
    result_commit
    return (result_commit,)


@app.cell(hide_code=True)
def hints_commit():
    L.hints(
        "commit-of",
        [
            "*Revisions* in *The idea*: a branch and a tag are names that point at a commit, and a commit id names itself.",
            "Three `if`s, each with `in` and a `return` under it: one for `repo[\"branches\"]`, one for `repo[\"tags\"]` and one for `repo[\"commits\"]`. After them comes `return None`.",
            "`        return repo[\"branches\"][___]`",
        ],
        why="`if revision in repo[\"branches\"]:` with `return repo[\"branches\"][revision]` under it, the same for `repo[\"tags\"]`, then `if revision in repo[\"commits\"]:` with `return revision`, and `return None` at the end. `commit_of(DAY_1, \"main\")` gives `FIRST`.",
    )
    return


@app.cell(hide_code=True)
def text_after_commit():
    mo.md(r"""
    On day 61 the tag `v1` names the new owner's commit, and `FIRST` names nothing: the commit of day 1 went with the old repository. A commit id cannot come to name other files; a branch or a tag can point wherever the owner of the repository wants.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pin():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. pin_now: pin what you checked

        Write `pin_now(repo, revision, name)`: a pin for the file `name`, as a dictionary with the two keys that the course's own pins have, `"revision"`, the commit id that `revision` names today, and `"sha256"`, the SHA-256 of the file as 64 hexadecimal digits. The box may use `commit_of`: the course's version of your function from Exercise 1. `commit` is the commit id that `revision` names today, and `data` holds the bytes of the file. Replace the two `...`. The check calls `pin_now(DAY_1, "main", "model.safetensors")`, then the other days, revisions and files in the picture under it.
        """),
        L.task("Keep the commit that the revision names today, not the revision itself, and the SHA-256 of the bytes of the file."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pin():
    box_pin = L.code_box("pin-now", lines=7)
    box_pin
    return (box_pin,)


@app.cell(hide_code=True)
def check_pin(box_pin):
    result_pin = L.check("pin-now", box_pin.value)
    result_pin
    return (result_pin,)


@app.cell(hide_code=True)
def hints_pin():
    L.hints(
        "pin-now",
        [
            "Steps 1 and 2 of the recipe: resolve the revision to its commit once, and write down the SHA-256 of the file.",
            "`commit` is what `commit_of` gives for `repo` and `revision`. The SHA-256 is that of `data`, written the way the worked example prints it.",
            "`    return {\"revision\": commit, \"sha256\": hashlib.sha256(data).___()}`",
        ],
        why="`commit = commit_of(repo, revision)` and `return {\"revision\": commit, \"sha256\": hashlib.sha256(data).hexdigest()}`. `pin_now(DAY_1, \"main\", \"model.safetensors\")` gives the commit `FIRST` and the SHA-256 that starts with `8f2160dd`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pin():
    mo.md(r"""
    Your pin holds two things the hub cannot change behind your back: a commit id that names one version, and the SHA-256 of the bytes you checked. `PINNED` in `everything_ai/hub.py` keeps the same two, and the size.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_load():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. load_pinned: only the pinned file

        Write `load_pinned(repo, pin, name)`: the bytes of the file `name` at the revision of the pin, or `None` when they are not exactly the pinned file. `repo` is `None` on a day when the hub has no such repository. `pin` is a dictionary as `pin_now` makes it, or one written by hand, with a branch or a tag as its revision. `commit` is the commit that the revision names, and `data` the bytes of the file; the box may use `commit_of` again. Replace the two `...`. The check calls `load_pinned(DAY_30, PIN_DAY_1, "model.safetensors")`, where `PIN_DAY_1` is the pin that `pin_now` made on day 1, then the days and pins in the picture under it: `PIN_MAIN` and `PIN_V1` hold the SHA-256 of the model of day 1 with the revision `"main"` and `"v1"`, and `PIN_CARD` that of the model card with `"v1"`.
        """),
        L.task("Give back None when the revision names no commit, and when the SHA-256 of the file is not the pinned one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_load():
    box_load = L.code_box("load-pinned", lines=13)
    box_load
    return (box_load,)


@app.cell(hide_code=True)
def check_load(box_load):
    result_load = L.check("load-pinned", box_load.value)
    result_load
    return (result_load,)


@app.cell(hide_code=True)
def hints_load():
    L.hints(
        "load-pinned",
        [
            "Step 2 of the recipe: refuse every file with another SHA-256, and a revision that names nothing.",
            "Each `...` is an `if` with `return None` under it. The first is about what `commit_of` gave; the second is about the SHA-256 of `data`, written as in Exercise 2.",
            "`    if hashlib.sha256(data).hexdigest() != ___:`",
        ],
        why="`if commit is None:` with `return None` under it, then `if hashlib.sha256(data).hexdigest() != pin[\"sha256\"]:` with `return None` under it. `load_pinned(DAY_30, PIN_DAY_1, \"model.safetensors\")` gives the model of day 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_load():
    mo.md(r"""
    With a pin on a commit, the new owner's repository gives nothing on day 61: the commit is not there. With a pin on `"main"` or `"v1"`, the revision names other files by then, and only the SHA-256 stops them. And `README.md` passes on day 61: the new owner copied the model card byte for byte, so its SHA-256 is that of day 1.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your pin over 61 days

    Pick the day on which Spoke & Chain makes its pin with `pin_now`, on the branch `main`. The table shows what the shop gets on each day: by name, by the tag `v1`, and with that pin through `load_pinned`.
    """)
    return


@app.cell(hide_code=True)
def demo_box():
    demo_day = mo.ui.dropdown(options={"day 1": 1, "day 30": 30, "day 61": 61}, value="day 1",
                              label="The shop makes its pin on")
    demo_day
    return (demo_day,)


@app.cell(hide_code=True)
def demo_look(box_commit, box_load, box_pin, demo_day, result_commit, result_load, result_pin):
    _commit = C.mine(box_commit.value, result_commit.state == "correct", "commit_of")
    _pin = C.mine(box_pin.value, result_pin.state == "correct", "pin_now")
    _load = C.mine(box_load.value, result_load.state == "correct", "load_pinned")
    _out = mo.Html(C.demo_picture(demo_day.value, L.lang, _commit, _pin, _load))
    _out
    return


@app.cell(hide_code=True)
def text_exercise_match():
    mo.md(r"""
    ### Exercise 4. Every check passes

    A second shop, Gear & Go, first finds `cog-and-crank/trail-sorter` on day 61. It makes a pin with `pin_now` that day, and from then on `load_pinned` gives it the file every day.
    """)
    return


@app.cell(hide_code=True)
def choice_match_cell():
    choice_match = L.choice(
        "match",
        question="What does Gear & Go know from that?",
        options={
            "right": "That it gets the file it pinned, whoever made it",
            "maker": "That cog-and-crank made it: name and SHA-256 both match",
            "safe": "That the model is safe: a change to it would change the SHA-256",
        },
        correct="right",
        why={
            "right": "A pin keeps you on one file, and says nothing about where it came from: Gear & Go pinned the new owner's model on day 61, and gets it every day. Who made a file is a question for a signature; whether it is good, for testing it and reading its card.",
            "maker": "The name changed hands on day 61, and the SHA-256 was taken that day: both fit the new owner's file. See *What a hash does not tell you* in *The idea*.",
            "safe": "The SHA-256 shows a change made after the pin, not a problem that was there before it, as in the releases of xz. See *What a hash does not tell you* in *The idea*.",
        },
    )
    choice_match
    return (choice_match,)


@app.cell(hide_code=True)
def text_exercise_pipeline():
    mo.md(r"""
    ### Exercise 5. A script that loads by name

    Every night, a script at Spoke & Chain loads `from_pretrained("cog-and-crank/trail-sorter")`, without a revision. The model is in safetensors.
    """)
    return


@app.cell(hide_code=True)
def choice_pipeline_cell():
    choice_pipeline = L.choice(
        "pipeline",
        question="Which change keeps the script from loading the new owner's model on day 61?",
        options={
            "right": "Pin the commit of day 1 and the SHA-256 of its file",
            "tag": "Pass revision=\"v1\": a tag stays on one version",
            "none": "None is needed: a safetensors file cannot hold an attack",
        },
        correct="right",
        why={
            "right": "On day 61 the commit of day 1 is gone, so the download fails instead of giving another model, and the SHA-256 refuses every other file. Unit 42 gives the same advice: \"pin the model to a specific commit using the revision parameter\".",
            "tag": "On day 61 the new owner made a tag `v1` too, as the picture of the hub shows: a tag is a name, set by whoever owns the repository. See *Revisions* in *The idea*.",
            "none": "Safetensors keeps code out of the file, and that is all: the new owner's file is safetensors too, with one number changed. See [A model file that runs code](/course/open/rai-pickle-safetensors?lang=en).",
        },
    )
    choice_pipeline
    return (choice_pipeline,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [A model file that runs code](/course/open/rai-pickle-safetensors?lang=en) showed that safetensors keeps code out of a file. This lesson asked the next question: whether the file is the one you meant. A commit id fixes the version, a SHA-256 checks the bytes you got, and neither says who made them. [Using a pretrained model](/course/open/torch-pretrained?lang=en) downloaded the course's digit reader from the hub; now you know why the course keeps it only when its SHA-256 matches. The next lesson, *Packages that are not what they seem*, looks at the same problem for Python packages: names close to a real one, and names that a model makes up.

    **An everyday example.** A jar of jam with an unbroken seal: the seal says that no one opened the jar after it was sealed. It does not say who sealed it, or whether the jam was good.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_match, choice_pipeline, result_commit, result_load, result_pin):
    L.wrap_up(result_commit, result_pin, result_load, choice_match, choice_pipeline)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
