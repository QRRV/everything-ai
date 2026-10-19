# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-own-modules. Route: course/route/lessen/stam-1-basis.yaml. Checks, and the module shown in the lesson: checks.py.
# The package this lesson gives you is in project/demo/; the course copies it to work/projects/demo/.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-own-modules", "en")
    C = L.checks   # the module the lesson gives you, as text


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Using code from modules](/course/open/py-imports?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="What does this program show?\n\n```python\nimport math\nprint(math.floor(3.7))\n```",
        options={
            "four": "`4`",
            "three": "`3`",
            "same": "`3.7`",
        },
        correct="three",
        why={
            "four": "`floor` always rounds down, also from 3.7. See the idea in [Using code from modules](/course/open/py-imports?lang=en).",
            "three": "`math.floor` rounds down to the whole number below: 3.",
            "same": "`math.floor` changes the number: it rounds it. See the idea in [Using code from modules](/course/open/py-imports?lang=en).",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_from():
    quick_from = L.choice(
        "quick-from",
        question="What does this program show?\n\n```python\nfrom math import sqrt\nprint(sqrt(25))\n```",
        options={
            "error": "It stops: Python does not know `sqrt`",
            "whole": "`5`",
            "float": "`5.0`",
        },
        correct="float",
        why={
            "error": "`from math import sqrt` brings in the name `sqrt` itself, so it works without `math.`. See the recipe in [Using code from modules](/course/open/py-imports?lang=en).",
            "whole": "A square root from `math` is always a float. See the guess in [Using code from modules](/course/open/py-imports?lang=en).",
            "float": "`from math import sqrt` brings in `sqrt` itself, and the square root of 25 is the float 5.0.",
        },
    )
    quick_from
    return (quick_from,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The code in a lesson's cells lives in that lesson only. Code you want to use again, such as a function that works out an average, belongs in a file of its own: a module, just like `math`, but yours. Later lessons import your module, and so your project grows, file by file, into your own language model.

    **An everyday example.** Your own toolbox. A tool you made yourself goes into the box, and the next time you need it, you take it out instead of making it again. `math` is a toolbox someone else filled; your project folder is yours.
    """)
    return


@app.cell(hide_code=True)
def predict_import():
    guess_import = L.guess(
        "import",
        question="Your project folder, `work/projects/`, has a file `demo/shapes.py` with the function `square_area(side)`, which gives back `side * side`. What do you think this program shows?\n\n```python\nfrom demo import shapes\nprint(shapes.square_area(4))\n```",
        options={
            "sixteen": "`16`",
            "error": "It stops: Python cannot find `demo`",
            "text": "`square_area(4)`",
        },
        reveal="**`16`.** In the course, Python looks in your project folder, `work/projects/`, when you import. There it finds the folder `demo` with the file `shapes.py`, and `shapes.square_area(4)` calls the function in it.",
    )
    guess_import
    return (guess_import,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A module of your own is just a `.py` file with your code in it. A folder of such files is a package. You import them in the same way as `math`.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **module** (your own) | a `.py` file with your code; its name without `.py` is the module's name | `stats.py` is the module `stats` |
    | **package** | a folder of modules, with a file `__init__.py` in it; you import a module from it with `from` | `from tools import stats` |
    | **project folder** | in the course, `work/projects/`: your packages live here, and Python looks here when you import | `work/projects/tools/stats.py` |

    Python looks for a module in a list of folders. In every lesson, the course puts your project folder on that list, so `from tools import stats` finds `work/projects/tools/stats.py`. The file `__init__.py` may be empty: it marks the folder as a package. That is a simplification: Python can often import a folder without it too, but the course always makes one, so you can see at once which folders are packages. A module mostly holds functions: the lines at its left edge run when a program imports the module.
    """)
    return


@app.cell(hide_code=True)
def picture_folders():
    mo.vstack([
        draw.marked_lines([("Your project folder", [("work/", ""), ("  projects/", ""), ("    demo/", "1"), ("      __init__.py", ""),
                                                   ("      shapes.py", "2")]),
                           ("In a cell", [("from demo import shapes", "1"), ("shapes.square_area(3)", "2")])],
                          {"1": "the package: a folder, found in the project folder",
                           "2": "the module in it, and a function from that module"},
                          title="Where an import looks"),
        mo.md(r"""
        `from demo import shapes` looks in the project folder for a folder `demo` (1), and in it for the file `shapes.py` (2). After that, `shapes.square_area` is the function in that file.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Give your package a name of its own, not the name of a module you use, such as `random` or `numpy`: Python looks in your project folder last, so it finds the other one, never yours. And put only functions in a module: a `print` at its left edge runs whenever a program imports the module, and shows up there."),
        mo.md(r"""
        ### Recipe

        To keep code and use it again:

        1. Put the code you want to keep in a function.
        2. Save it in a `.py` file, in a package folder under `work/projects/`. In this course, a check saves your function there when it is right.
        3. Import the module where you need it: `from tools import stats`.
        4. Use it with the module's name and a dot: `stats.mean(temps)`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        When you opened this lesson, the course put a small package in your project folder: `work/projects/demo/`. It holds the file `shapes.py`, the module `shapes`, with these two functions; steps 1 and 2 are done for you. The comments with `#` in the cell below say which step of the recipe each part is. Run the cell.
        """),
        mo.md(f"```python\n{C.DEMO}\n```"),
    ])
    return


@app.cell
def example_own_module():
    # 3. import the module from its package in your project folder
    from demo import shapes
    # 4. use it with the module's name and a dot
    print(shapes.square_area(3))
    print(shapes.rectangle_area(2, 5))
    return (shapes,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `9` and `10`. The functions are not in this lesson at all: they are in a file in your project folder, and any lesson can import them in the same way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A function worth keeping

        Write `mean(numbers)`: the average of a list of numbers. When your function is right, the check saves it in your project, as `work/projects/tools/stats.py`; exercise 3 imports it from there. The check calls it with `[4, 8, 6]`, and with other lists.
        """),
        L.task("Make `mean(numbers)` give back the average; the check then saves it in your project."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean", lines=6)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean",
        [
            "The average is the total of the numbers, divided by how many there are.",
            "A loop adds up the numbers in `total`; after the loop, divide by the number of items.",
            "`return total / ___`, after the loop.",
        ],
        why="The loop adds every number to `total`; `total / len(numbers)` divides by how many there are. For `[4, 8, 6]` that is 18 / 3, which is `6.0`.",
    )
    return


@app.cell(hide_code=True)
def text_import_line():
    mo.md(r"""
    ### Exercise 2. The right import
    """)
    return


@app.cell(hide_code=True)
def choice_import_line_cell():
    choice_import_line = L.choice(
        "import-line",
        question="Your function `count_words` is in the file `work/projects/text/words.py`. Which line imports its module, so that `words.count_words(...)` works?",
        options={
            "file": "`import words.py`",
            "package": "`from text import words`",
            "path": "`import work.projects.text.words`",
        },
        correct="package",
        why={
            "file": "A module's name has no `.py`, and `words` is inside a package. Look at the table of new words again.",
            "package": "Python looks in `work/projects/`, finds the package `text`, and in it the module `words`.",
            "path": "Python already looks in `work/projects/`, so that part is not in the name. Look at the picture again.",
        },
    )
    choice_import_line
    return (choice_import_line,)


@app.cell(hide_code=True)
def text_exercise_use():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Use your own module

        The box imports your module `stats` from the package `tools`. Replace the `...` with a line that shows the average of `temps`, with your own `stats.mean`. The check also tries other temperatures.
        """),
        L.task("Show the average of `temps` with `stats.mean`, so the program shows `12.0`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_use():
    box_use = L.code_box("use-stats", lines=5)
    box_use
    return (box_use,)


@app.cell(hide_code=True)
def check_use(box_use, result_mean):
    result_mean  # check again when exercise 1 saves your module
    result_use = L.check("use-stats", box_use.value)
    result_use
    return (result_use,)


@app.cell(hide_code=True)
def hints_use():
    L.hints(
        "use-stats",
        [
            "Step 4 of the recipe: a function from a module is used with the module's name and a dot.",
            "Call the function `mean` from the module `stats` on `temps`, and show what it gives back with `print`.",
            "`print(stats.___(temps))`",
        ],
        why="`stats.mean(temps)` calls the function you wrote in exercise 1, from the file the check saved. `print` shows its result: `12.0`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Using code from modules](/course/open/py-imports?lang=en) you imported modules that others wrote; now you import your own, from your project folder. From here on, some checks save your code there, and later lessons import it: at the end of the route, `work/projects/` holds your own language model. In the next lesson you make your own kind of value, with a class.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_import_line, result_mean, result_use):
    L.wrap_up(result_mean, choice_import_line, result_use)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
