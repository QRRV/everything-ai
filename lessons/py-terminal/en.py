# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-terminal. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, paths
    from everything_ai.lesson import lesson

    L = lesson("py-terminal", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en), [Using code from modules](/course/open/py-imports?lang=en) and [Reading a text file](/course/open/py-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_where():
    quick_where = L.choice(
        "quick-where",
        question="In the course, where does `from tools import stats` find the module `stats`?",
        options={
            "lesson": "In the folder of the lesson that runs the line",
            "project": "In `work/projects/tools/stats.py`",
            "python": "Among the modules that come with Python, such as `math`",
        },
        correct="project",
        why={
            "lesson": "A lesson's own folder is not where your packages live. See the table of new words in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
            "project": "The course puts your project folder, `work/projects/`, on Python's list of folders; there it finds the package `tools`, and in it `stats.py`.",
            "python": "`stats` is a module you wrote yourself. Where does the course keep those? See the picture in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).",
        },
    )
    quick_where
    return (quick_where,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="A program sets `random.seed(7)` and then rolls a die three times. You run it again. What happens?",
        options={
            "same": "It rolls the same three numbers again",
            "new": "It rolls three new numbers",
            "sevens": "It rolls three sevens",
        },
        correct="same",
        why={
            "same": "The same seed makes `random` start its fixed calculation at the same place, so the same numbers come out.",
            "new": "Without a seed, that would usually happen. What does a seed do? See the idea in [Using code from modules](/course/open/py-imports?lang=en).",
            "sevens": "The seed is where the calculation starts, not the number that comes out. See the recipe in [Using code from modules](/course/open/py-imports?lang=en).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="In the course, a program opens the path `\"lessons/py-files/poem.txt\"`. Where does that path start?",
        options={
            "home": "At your home folder",
            "file": "At the file `poem.txt`",
            "course": "At the course folder, where you started the course",
        },
        correct="course",
        why={
            "home": "A path like this starts at the folder Python runs in, and that is not your home folder. See the idea in [Reading a text file](/course/open/py-files?lang=en).",
            "file": "The file is where the path ends, not where it starts. See the table of new words in [Reading a text file](/course/open/py-files?lang=en).",
            "course": "Python runs in the course folder, so a path starts there: in it is the folder `lessons`, in that `py-files`, and in that the file.",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    While a cell runs, the lesson waits for it. That is fine for a second, but some jobs take minutes, or hours, such as training a large model. Such a long job can run as a script: a `.py` file that you start from the terminal. It runs on its own, prints how far along it is, and the lesson stays free.

    **An everyday example.** A washing machine. You choose a program, press start, and it runs on its own; you do something else in the meantime, and the display tells you how far along it is.
    """)
    return


@app.cell(hide_code=True)
def predict_wait():
    guess_wait = L.guess(
        "wait",
        question="A cell in a lesson runs a loop that takes an hour. What do you think happens when you run another cell in the meantime?",
        options={
            "parallel": "Both run at the same time",
            "waits": "The other cell waits until the first one is done",
            "stops": "The first cell stops, and the other one runs",
        },
        reveal="**The other cell waits.** A lesson runs one cell at a time. As long as the long cell runs, everything else waits. A script in a terminal has a Python of its own, so the lesson stays free.",
    )
    guess_wait
    return (guess_wait,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    A terminal is a window where you type commands. You already used one: you started the course with `uv run everything-ai`. In a second terminal, you run your own scripts.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **terminal** | a window where you type a command and press Enter to run it | |
    | **command** | one line in the terminal: a program, and what it should do | `uv run everything-ai` |
    | **folder you are in** | the terminal works in one folder at a time; a path in a command starts there | `cd` goes to a folder |
    | **script** | a `.py` file that you run as one program, from its first line to its last | `dice.py` |
    | **`uv run python`** | runs a Python file with the course's own Python and modules | `uv run python work/projects/scripts/dice.py` |
    | **Ctrl+C** | stops the command that is running in the terminal; on a Mac too, with the Control key, not Cmd | |

    **A second terminal, on your system:**

    - **Mac:** the terminal is the app *Terminal*. With Terminal in front, Cmd+N opens a second window.
    - **Linux:** the Terminal app that comes with your system. In most of them, Ctrl+Shift+N opens a second window; or open the app again from the menu.
    - **Windows:** the course runs in WSL. Open *Ubuntu* again from the Start menu, as when you started the course. In Windows Terminal, click the small arrow next to the +, and choose Ubuntu: the + itself often opens PowerShell, where these commands do not work.
    """)
    return


@app.cell(hide_code=True)
def picture_terminal():
    mo.vstack([
        draw.marked_lines([("Terminal", [(f'$ cd "{paths.ROOT}"', "1"), ("$ uv run python work/projects/scripts/dice.py", "2"),
                                         ("part 1 of 5 done", "3"), ("part 2 of 5 done", ""), ("...", ""),
                                         ("… sixes in ten million rolls", ""), ("$", "4")])],
                          {"1": "go to the course folder",
                           "2": "run the script, with its path from the course folder",
                           "3": "what the script prints appears here, while it runs",
                           "4": "the $ is back: the terminal is free again"},
                          title="A script in the terminal"),
        mo.md(r"""
        You type what comes after the `$`; the terminal shows the rest. The `$` is the terminal's sign that it waits for a command: you never type it yourself. On a Mac, that sign is often `%`. Your terminal may show something longer before it, such as your name and the folder you are in.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Keep the terminal where the course runs open: closing it stops the course. Ctrl+C there asks whether to stop the course; press Enter to keep it running. Closed it by accident? Go to the course folder and type `uv run everything-ai` again."),
        mo.md(r"""
        ### Recipe

        To run a script from the terminal:

        1. Open a second terminal, next to the one where the course runs.
        2. Go to the course folder: `cd`, a space, and the folder between double quotes.
        3. Run the script: `uv run python`, a space, and its path from the course folder.
        4. Read what it prints. When the `$` is back, it is done; Ctrl+C stops it sooner.

        Does the terminal say `No such file or directory`? Then it is not in the course folder yet: do step 2 again.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        When you opened this lesson, the course put a script in your project folder: `work/projects/scripts/dice.py`. It rolls a die ten million times, in five parts, and counts the sixes:
        """),
        mo.md(f"```python\n{C.DICE}\n```"),
        mo.md(f"""
        Your course folder is `{paths.ROOT}`. This is what you do, step by step.

        **Step 1.** Open a second terminal.

        **Step 2.** Go to the course folder, with exactly this line:

        ```text
        cd "{paths.ROOT}"
        ```

        **Step 3.** Run the script:

        ```text
        uv run python work/projects/scripts/dice.py
        ```

        **Step 4.** Read what it prints, until the `$` is back.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The script prints a line after each part, while it runs, and at the end how many sixes it rolled. It has a loop inside a loop: the outer loop goes through the 5 parts, and for each part, the inner loop rolls 2 million times. It uses `random.seed(7)`, so you get the same count every time you run it.

    Now try the Ctrl+C of step 4: run the script again, and press Ctrl+C while it runs. It stops early, and Python prints a few lines that end in `KeyboardInterrupt`. That only means you stopped it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_command():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The right command
    """)
    return


@app.cell(hide_code=True)
def choice_command_cell():
    choice_command = L.choice(
        "which-command",
        question="Your terminal is in the course folder. Which command runs the script `work/projects/scripts/dice.py`?",
        options={
            "uv": "`uv run python work/projects/scripts/dice.py`",
            "short": "`uv run python dice.py`",
            "bare": "`python work/projects/scripts/dice.py`",
        },
        correct="uv",
        why={
            "uv": "`uv run python` runs the file with the course's own Python, and the path starts at the course folder, where the terminal is.",
            "short": "The terminal is in the course folder, and `dice.py` is not there: a path starts at the folder you are in. Look at step 3 of the recipe again.",
            "bare": "Without `uv run`, this is not the course's own Python, if there is a `python` at all: on many computers, the terminal says `command not found`. Look at the table of new words again.",
        },
    )
    choice_command
    return (choice_command,)


@app.cell(hide_code=True)
def text_exercise_dice():
    mo.md(r"""
    ### Exercise 2. Run it yourself

    Follow the worked example: open a second terminal, go to the course folder, and run `dice.py`. Then answer the question.
    """)
    return


@app.cell(hide_code=True)
def choice_dice_cell():
    choice_dice = L.choice(
        "dice-result",
        question="What is the last line that `dice.py` prints in your terminal?",
        options={
            "low": "`1665730 sixes in ten million rolls`",
            "real": "`1666892 sixes in ten million rolls`",
            "high": "`1667104 sixes in ten million rolls`",
        },
        correct="real",
        why={
            "low": "That is not what the script prints. Run it, and read its last line.",
            "real": "That is what the script printed: close to one in six, which would be 1,666,667, but not exactly, because each roll is chance.",
            "high": "That is not what the script prints. Run it, and read its last line.",
        },
    )
    choice_dice
    return (choice_dice,)


@app.cell(hide_code=True)
def text_exercise_countdown():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Your own script

        Write a script that counts down: it prints `3`, `2` and `1`, one per line, and then `go`. Replace the `...`. The check runs it and looks at what it prints. When it is right, the check saves it whole, as `work/projects/scripts/countdown.py`. Then run it in your terminal, as in the worked example.
        """),
        L.task("Make the script print 3, 2, 1 and go, one per line; then run it in your terminal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_countdown():
    box_countdown = L.code_box("countdown", lines=6)
    box_countdown
    return (box_countdown,)


@app.cell(hide_code=True)
def check_countdown(box_countdown):
    result_countdown = L.check("countdown", box_countdown.value)
    result_countdown
    return (result_countdown,)


@app.cell(hide_code=True)
def hints_countdown():
    L.hints(
        "countdown",
        [
            "A loop over `range(3)` gives 0, 1 and 2. Which number do you print, so that it gives 3, 2 and 1?",
            "In the loop, print 3 minus the loop's number; after the loop, at the left edge, print `go` once.",
            "`for i in range(3):`, indented `print(___)`, and then `print(\"go\")` at the left edge.",
        ],
        why="`3 - i` is 3, 2 and 1 for `i` = 0, 1 and 2. `print(\"go\")` stands after the loop, so it runs once. In the terminal, `uv run python work/projects/scripts/countdown.py` prints the same four lines.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en) the check saved a function in your project, for lessons to import; now the check saved a whole script, for you to run. In [Reading a text file](/course/open/py-files?lang=en) a path started at the course folder; in the terminal, you go there yourself, with `cd`. In [Using code from modules](/course/open/py-imports?lang=en) `random.seed` made chance repeatable; `dice.py` uses it, so it gives the same count every time. Later in the course, a long job, such as training a model, can run as such a script: it prints its loss as it goes, and runs for as long as it needs.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_command, choice_dice, result_countdown):
    L.wrap_up(choice_command, choice_dice, result_countdown)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
