# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-environments. Route: course/route/lessen/systemen.yaml. Checks, the computers, the NumPy releases, the demo and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-environments", "en")
    C = L.checks   # the course's own files, the computers, the NumPy releases, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Reading a text file](/course/open/py-files?lang=en) and [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_strip():
    quick_strip = L.choice(
        "quick-strip",
        question="A file holds the line `numpy==2.5.3`, with a newline after it. What does `line.strip()` give for that line?",
        options={
            "right": "The string `numpy==2.5.3`, without the newline",
            "number": "The version `2.5.3`, as a number",
            "same": "The line as it was, newline and all",
            "empty": "An empty string, since strip removes all the letters",
        },
        correct="right",
        why={
            "right": "`strip()` removes the spaces and newlines at the start and the end of a string, and keeps the rest. In this lesson, you read the course's lock file line by line, with `strip()` on each line.",
            "number": "`strip()` gives a string, never a number, and it keeps the name. What does it remove? See [Reading a text file](/course/open/py-files?lang=en).",
            "same": "The newline is exactly what `strip()` removes at the end. What is left of the line? See [Reading a text file](/course/open/py-files?lang=en).",
            "empty": "`strip()` removes only spaces and newlines at the start and the end. What is left of the line? See [Reading a text file](/course/open/py-files?lang=en).",
        },
    )
    quick_strip
    return (quick_strip,)


@app.cell(hide_code=True)
def quick_check_package():
    quick_package = L.choice(
        "quick-package",
        question="In this course, what is a package?",
        options={
            "file": "One `.py` file with functions in it",
            "right": "A folder of modules, with a file `__init__.py`",
            "function": "A function you call with `from`",
            "program": "A program that installs other programs on your computer",
        },
        correct="right",
        why={
            "right": "A package is a folder of modules. The packages you install, such as numpy, are folders of modules too, written by others. This lesson is about which version of them ends up on your computer.",
            "file": "One `.py` file is a module. What do you call a folder of modules? See [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
            "function": "`from` imports a module from a package. What is the package itself? See [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
            "program": "Installing is the work of a tool such as pip or uv, and what it installs are packages. What is a package? See [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
        },
    )
    quick_package
    return (quick_package,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    On 16 June 2024, NumPy 2.0.0 came out. Its release notes warned of breaking changes, among them one for compiled packages: packages with parts written in C and turned into machine code against one version of NumPy. Packages built against NumPy 1.x "will not work with NumPy 2.0". From that day on, a project installed with the line `numpy`, without a version, got 2.0.0, and a package compiled against NumPy 1.x stopped at its import with a message that begins: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0 as it may crash." The same code, with the same line, ran fine on a computer set up the week before.

    The message itself suggested the way out, a version: "downgrade to 'numpy<2'". This lesson is about writing versions down before anything breaks: one virtual environment per project, and one exact version for every package, on every computer. In ML it matters twice, because a trained model depends on versions too: scikit-learn says there are "no supported ways to load a model trained with a different version of scikit-learn".

    **An everyday example.** A recipe that says "flour" works in one kitchen and fails in the next, because one kitchen has bread flour and the other self-rising flour. A recipe that names the exact bag gives the same bread in every kitchen.
    """)
    return


@app.cell(hide_code=True)
def predict_locked():
    guess_locked = L.guess(
        "locked",
        question="`pyproject.toml` is the file that describes a project and the packages it needs. The course's own file names about a dozen packages, such as `marimo`, `numpy` and `torch`. Its lock file, `uv.lock`, pins packages too. How many?",
        options={
            "same": "About a dozen: the same ones",
            "thirty": "About 30",
            "right": "About 100",
            "thousand": "About 1,000",
        },
        reveal="**About 100.** The packages the course names need packages of their own, and those need more: matplotlib alone needs nine, such as pillow and kiwisolver. The lock file pins all of them, each at one exact version. The worked example below counts them in your copy of the course.",
    )
    guess_locked
    return (guess_locked,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Versions.** Packages come from PyPI, the online index that pip and uv download them from. Every release of a package has a version, such as 2.5.3: a major, a minor and a patch number. Many projects follow semantic versioning, in which only a new major version may break code that worked. NumPy, like Python itself, does not: its minor versions may remove features too, after warning about them for at least two releases. So any new version can change something your code needs.

        **One virtual environment per project.** An installer puts packages into one folder of one Python, so two projects that need two versions of pandas cannot share it. A **virtual environment** is a folder with its own Python and its own installed packages, for one project. Python's venv module has made them since Python 3.3: `python -m venv .venv`. The course has one too: `uv run everything-ai` uses the folder `.venv` in the course folder. The folder stays out of git, the tool that keeps the history of a project's files: it can always be made again from the two files below.

        **Pinning.** A line in `pyproject.toml` says which versions a project accepts. `numpy` accepts any version, `numpy>=1.24,<2` a range (the comma means *and*), and `numpy==1.26.4` one exact version. Writing down the exact version is **dependency pinning**. A range gives a different version as new releases come out, because an installer takes the newest release a range allows.

        **Versions compare part by part.** Version 2.10.0 comes after 2.9.0, but as strings, `"2.10.0"` comes before `"2.9.0"`: strings compare character by character, and `"1"` comes before `"9"`. A tuple compares part by part: `(2, 10, 0) > (2, 9, 0)` is `True`, because the first parts are equal and 10 is greater than 9. The exercises below use tuples.

        **A lock file.** Pinning the packages you name is not enough: marimo, numpy and torch need packages of their own, and those need more. A **lock file** records the exact version of every package, these included, worked out once, so every install gets the same. uv writes `uv.lock` with `uv lock` and installs from it with `uv sync`; its documentation says the file "should be checked into version control, allowing for consistent and reproducible installations across machines". With pip, `pip freeze` prints what is installed as lines such as `numpy==2.5.3`, but pip's documentation adds that it "does not compute a lockfile". Since 2025, Python also has a standard format for lock files, `pylock.toml`, from PEP 751.

        **A model remembers its versions.** A model saved to a file depends on the versions it was trained with. Since version 1.3, scikit-learn warns with an `InconsistentVersionWarning` when you load a model saved with another version. The versions belong next to the model, and the next lessons record them with every run. The picture shows one line on two computers, without and with a lock file.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **virtual environment** | | a folder with its own Python and its own installed packages, for one project | `python -m venv .venv` |
        | **dependency pinning** | | writing down the exact version of a package a project uses | `numpy==1.26.4` |
        | **lock file** | | a file with the exact version of every package a project installs, the packages of packages included | `uv.lock` |
        """),
        L.watch_out("In reinforcement learning, environment means something else: the world an agent acts in. In this lesson, an environment is a folder with a Python and its packages."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    The same packages on every computer:

    1. Make one virtual environment per project (`uv venv`, or `python -m venv .venv`), and keep its folder out of git.
    2. Name the packages your code imports in `pyproject.toml`.
    3. Let the tool work out every version once and write the lock file (`uv lock`). Commit `pyproject.toml` and the lock file to git.
    4. On every computer, install from the lock file (`uv sync`), not from the ranges.
    5. When two computers behave differently, compare their versions package by package.
    6. Upgrade on purpose: change a version, run your tests, and commit the new lock file.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.LOCK_FILE` is the path of the course's own `uv.lock`, and `C.NAMED` holds the packages its `pyproject.toml` names, read by the course. In `uv.lock`, every package starts with a line `[[package]]`. The cell counts those lines. Run the cell.
    """)
    return


@app.cell
def example_lock():
    named = C.NAMED
    count = 0
    with open(C.LOCK_FILE) as _f:
        for _line in _f:
            if _line.strip() == "[[package]]":
                count = count + 1
    print(f"pyproject.toml names {len(named)} packages")
    print(f"uv.lock pins {count} packages")
    return count, named


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The lock file pins several times as many packages as `pyproject.toml` names. One entry is the course itself, and a few, such as pytest, are only for the course's own tests. Every other entry is a package the course needs, directly or through another package. Each entry has one exact version and a hash: a fingerprint of the file, so the installer can check that it gets exactly that file. Open `uv.lock` in a text editor and look for `name = "numpy"`: the line below it says which version every learner of this course gets.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_differences():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. What differs?

        Code runs on your laptop and stops on the server. `differences(mine, theirs)` gets two dictionaries, one per computer, with the name of each installed package as key and its version as value, as `pip freeze` lists them. It gives back a list `found` of the packages in `mine` that are missing in `theirs` or have another version there. The loop goes over every `name` in `mine`. The `if` already adds a package that `theirs` does not have, with `found.append(name)`; Python checks the `elif` below it, short for else if, only when the condition of the `if` is false. Replace the `...` with a condition that is true when the package has another version on the other computer.

        The check calls `differences(C.LAPTOP, C.SERVER)`, then other pairs of computers.
        """),
        L.task("List the packages whose versions differ."),
    ])
    return


@app.cell(hide_code=True)
def exercise_differences():
    box_differences = L.code_box("differences", lines=8)
    box_differences
    return (box_differences,)


@app.cell(hide_code=True)
def check_differences(box_differences):
    result_differences = L.check("differences", box_differences.value)
    result_differences
    return (result_differences,)


@app.cell(hide_code=True)
def hints_differences():
    L.hints(
        "differences",
        [
            "Step 5 of the recipe: compare the versions package by package. `mine[name]` is the version of `name` on your computer, as a string, such as `\"1.26.4\"`.",
            "`theirs` is a dictionary too, with the same kind of keys. How do you get the version of `name` out of it, and which comparison is true when two strings differ?",
            "`        elif mine[name] != ___:`",
        ],
        why="`mine[name] != theirs[name]` is true when both computers have the package, with different versions. For `C.LAPTOP` and `C.SERVER`, that is numpy, 1.26.4 against 2.0.1; joblib is missing on the server.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_resolve():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The newest version a range allows

        An installer that reads `numpy>=1.24,<2` takes the newest release that is at least 1.24 and below 2. `resolve(released, low, high)` does the same. `released` is a list of versions as tuples, from the oldest to the newest, and `low` and `high` are the two ends of the range, also as tuples, such as `(1, 24, 0)` and `(2, 0, 0)`. `chosen` starts as `None`. The loop goes over every `version`, and when the condition holds, `chosen = version` keeps it, so the last allowed version stays. Replace the `...` with a condition that is true when `version` is at least `low` and below `high`.

        The check calls `resolve(C.RELEASED, (1, 24, 0), (2, 0, 0))`, with every NumPy release on PyPI from 1.24.0 to 2.5.5, except 2.4.0: PyPI marks it as yanked, withdrawn, so installers skip it. Then other ranges and lists.
        """),
        L.task("Keep the newest version the range allows."),
    ])
    return


@app.cell(hide_code=True)
def exercise_resolve():
    box_resolve = L.code_box("resolve", lines=7)
    box_resolve
    return (box_resolve,)


@app.cell(hide_code=True)
def check_resolve(box_resolve):
    result_resolve = L.check("resolve", box_resolve.value)
    result_resolve
    return (result_resolve,)


@app.cell(hide_code=True)
def hints_resolve():
    L.hints(
        "resolve",
        [
            "An installer takes the newest release a range allows. Tuples compare part by part, so you can compare `version` with `low` directly. Which comparison with `low` lets `low` itself in?",
            "`<2` means below 2.0.0, so 2.0.0 itself is not allowed. Which comparison with `high` leaves `high` out, and which word joins two conditions?",
            "`        if version >= low and ___:`",
        ],
        why="`version >= low and version < high` allows the versions from `low` up to just below `high`. For `numpy>=1.24,<2`, that is 1.26.4, the last NumPy 1; for `numpy>=2.0,<3`, it is 2.5.5.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### One line, installed on a chosen day

    The slider picks a day: the first of a month, from January 2023 to November 2026. On that day, four projects install NumPy from PyPI, each with another line: `numpy` without a version, the range `numpy>=1.24,<2`, the pin `numpy==1.24.1`, and a `uv.lock` made on 1 January 2023 from the line `numpy`. Each takes the newest release its line allows on that day, as your `resolve` does. The dots on the time line are the NumPy releases on PyPI, a row in the color of mistakes differs from the lock file, and the bottom row says whether a package compiled against NumPy 1.x still runs beside the line without a version.
    """)
    return


@app.cell(hide_code=True)
def demo_month():
    L.explore("month", "months after January 2023", lambda _v: C.frame(_v, L.lang), start=0, low=0, high=46, step=1,
              role="", task="Find the first month in which the range installs a different version from the lock file, and the first in which the line without a version gets NumPy 2.")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From 1 March 2023 on, the range installs a newer NumPy than the lock file, as does the line without a version: 1.24.2 at first, and 1.26.4 from March 2024. Two computers set up a month apart already differ, without a single error. On 1 July 2024, the line without a version installs NumPy 2.0.0 for the first time, and from then on the package compiled against NumPy 1.x stops; the range stays at 1.26.4, because it stops below 2. Only the pin and the lock file give 1.24.1 every month, up to November 2026, while the line without a version has moved on to 2.5.4. A pin and a lock file do not stop you from upgrading: they turn an upgrade into a step you take on purpose, with your tests, instead of a surprise on the day a new release comes out.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lock():
    mo.md(r"""
    ### Exercise 3. Pinned is not locked

    A team writes every package in its `pyproject.toml` with `==`: twelve exact versions. A year later, a new colleague installs the project from `pyproject.toml` alone, without a lock file.
    """)
    return


@app.cell(hide_code=True)
def choice_lock_cell():
    choice_lock = L.choice(
        "lock",
        question="Does the colleague get the same packages as the team?",
        options={
            "same": "Yes: every package named in the file has one exact version",
            "newest": "No: an installer always takes the newest release",
            "right": "Not all: the packages those twelve need can be newer",
            "never": "No: the same versions cannot be installed twice",
        },
        correct="right",
        why={
            "right": "The twelve lines pin only the packages the team named. The packages those need come with ranges of their own, and an installer takes the newest release each range allows. A lock file pins those too.",
            "same": "The twelve named packages are the same. Which packages does the file not name at all?",
            "newest": "For the twelve pinned lines, the installer takes exactly the pinned version. For which packages does it pick a version itself?",
            "never": "An old version stays on PyPI, so it can be installed again. Which packages are not pinned at all?",
        },
    )
    choice_lock
    return (choice_lock,)


@app.cell(hide_code=True)
def text_exercise_projects():
    mo.md(r"""
    ### Exercise 4. Two projects, one laptop

    Your old project needs pandas 1.5, because its code uses something pandas 2 removed. Your new project needs pandas 2.2.
    """)
    return


@app.cell(hide_code=True)
def choice_projects_cell():
    choice_projects = L.choice(
        "projects",
        question="How do both projects run on the same laptop?",
        options={
            "both": "Install both versions of pandas into the same Python",
            "upgrade": "Install pandas 2.2 and fix the old project some other day",
            "copy": "Copy the files of pandas into each project folder",
            "right": "Make a virtual environment for each project",
        },
        correct="right",
        why={
            "right": "Each project gets its own folder with its own Python and its own pandas, so both versions live on one laptop without getting in each other's way.",
            "both": "One Python holds one version of a package: installing the second replaces the first. Where can a second version live?",
            "upgrade": "Then the old project stops today. How can both run now?",
            "copy": "Then you install by hand, and nothing records which version each project has. Which folder holds the packages of one project?",
        },
    )
    choice_projects
    return (choice_projects,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In *Framing the problem for production*, you decided what a model predicts. This lesson makes sure the code around it runs the same on every computer: a virtual environment per project, and a lock file with every version. [Reading a text file](/course/open/py-files?lang=en) gave the loop over the lines of `uv.lock`, and [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en) the packages, which now have versions. In the next lesson, *Same run, same result: seeds and determinism*, the versions you pinned come back beside seeds: what else a training run needs to come out the same twice.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_lock, choice_projects, result_differences, result_resolve):
    L.wrap_up(result_differences, result_resolve, choice_lock, choice_projects)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
