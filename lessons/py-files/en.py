# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-files. Route: course/route/lessen/stam-1-basis.yaml. Checks and the files of this lesson: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-files", "en")
    C = L.checks   # the paths of the lesson's files, and the lines in the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Text: strings](/course/open/py-strings?lang=en) and [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="What does this program show?\n\n```python\ncount = 0\nfor word in [\"a\", \"bb\", \"c\"]:\n    count = count + 1\nprint(count)\n```",
        options={
            "one": "`1`",
            "three": "`3`",
            "four": "`4`",
        },
        correct="three",
        why={
            "one": "`count` keeps what it had, and each round adds 1 to it. See the picture in [Repeating: for loops](/course/open/py-for?lang=en).",
            "three": "The loop has one round for each of the three items, and each round adds 1.",
            "four": "The loop adds 1 for each item, not for each letter. See the idea in [Repeating: for loops](/course/open/py-for?lang=en).",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="What does this program show?\n\n```python\nprint(len(\"hi!\"))\n```",
        options={
            "two": "`2`",
            "five": "`5`",
            "three": "`3`",
        },
        correct="three",
        why={
            "two": "`len` counts every character, also `!`. See the idea in [Text: strings](/course/open/py-strings?lang=en).",
            "five": "The quotes are not part of the string. See the picture in [Text: strings](/course/open/py-strings?lang=en).",
            "three": "`h`, `i` and `!`: three characters.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Real data is stored in files: a book, a list of measurements, a chat. Until now, your data was typed into the program itself. Before a program can learn from text, it has to read that text from a file. Later in the course, you read a whole book this way and count its letters.

    **An everyday example.** A recipe book on a shelf. You take it, open it, read what you need, and close it again before you put it back. Python does the same with a file, and `with` closes it again for you.
    """)
    return


@app.cell(hide_code=True)
def predict_read():
    guess_read = L.guess(
        "read",
        question="The file `poem.txt` has four lines; the first is `Roses are red,`. What do you think this program shows?\n\n```python\nwith open(\"lessons/py-files/poem.txt\") as f:\n    text = f.read()\nprint(text[0])\n```",
        options={
            "line": "`Roses are red,`",
            "letter": "`R`",
            "word": "`Roses`",
        },
        reveal="**`R`.** `f.read()` gives the whole file as one string, and `[0]` is its first character, as with any string.",
    )
    guess_read
    return (guess_read,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    `open` opens a file, and `with` keeps it open for the indented lines under it, the `with` block, and closes it after them. In between, you read it: all at once, or line by line.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **file** | text or other data stored on disk, under a name | `poem.txt` |
    | **path** | where a file is: the folders and the file name, with `/` between them | `"lessons/py-files/poem.txt"` |
    | **open** | opens a file, so you can read it | `open(path)` |
    | **with** | keeps a file open for the indented lines under it, and closes it after them; `as` gives the open file a name | `with open(path) as f:` |
    | **read** | gives the whole file as one string | `f.read()` |
    | **newline** | the invisible character that ends a line, written `\n` | |
    | **strip** | gives a string without the spaces and newlines at both ends | `line.strip()` |

    `for line in f:` goes through an open file line by line, just as through a list. Each line ends with a newline; only the last line of a file may have none, and `strip` works either way. A path like `"lessons/py-files/poem.txt"` starts at the folder Python runs in. In the course, that is the course folder, `everything-ai`, where you started the course; this lesson's files are in its folder `lessons/py-files`. (marimo's list of files shows your own copies, in `work/`, without these files.) A path that does not lead to a file stops the program with an error. Closing matters: a file that is left open can stay in use, and `with` closes it for you, even when something goes wrong.
    """)
    return


@app.cell(hide_code=True)
def picture_lines():
    mo.vstack([
        draw.marked_lines([("The start of poem.txt, as you see it", [(_x, "") for _x in C.PICTURE_FILE]),
                           ("What the loop gives you", [(repr(_x + "\n"), "1") for _x in C.PICTURE_FILE]),
                           ("And line.strip()", [(repr(_x), "") for _x in C.PICTURE_FILE])],
                          {"1": "each line ends with \\n, the newline"},
                          title="Lines of a file, as Python reads them"),
        mo.md(r"""
        You see two lines, but Python sees what is really in the file: each line ends with `\n`. Python shows each string between quotes here, so you see where it ends. That is why `print(line)` shows an empty line after each one: the line's own newline, plus the one `print` adds. `line.strip()` drops it.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("Each line still has its newline at the end: without `strip`, `print` shows an empty line after it, and `len(line)` counts one character too many. And use the file only in the lines indented under `with`: after them, it is closed again."),
        mo.md(r"""
        ### Recipe

        To read a text file:

        1. Write the path as a string: the folders and the file name, with `/` between them.
        2. Write `with open(path) as f:`, with a colon.
        3. Indented under it: `f.read()` for the whole text at once, or `for line in f:` for one line at a time.
        4. Use `line.strip()` to drop the newline at the end of each line.
        5. After the `with` block, the indented lines, the file is closed again; keep what you need in a name.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A short poem, read in two ways. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_poem():
    # 1. the path: from the course's folder to the file
    _path = "lessons/py-files/poem.txt"
    # 2. and 3. open it with with, and read it all as one string
    with open(_path) as _f:
        _text = _f.read()
    print(len(_text))
    # 3. or one line at a time: count the lines, and show the long ones
    _count = 0
    with open(_path) as _f:
        for _line in _f:
            _count = _count + 1
            # 4. strip drops the newline before you measure or show the line
            if len(_line.strip()) > 15:
                print(_line.strip())
    # 5. after the with block the file is closed; the count stays
    print(_count)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear `71`, three lines of the poem, and `4`. The 71 characters include the four newlines. `Roses are red,` has 14 characters, so the `if` skips it; the other three lines are longer than 15. The file has four lines, so the loop had four rounds.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Count the lines

        `shopping.txt` holds a shopping list, one item per line. Count its lines in `count`: replace the `...` with a `with` block and a loop. The check also tries another file.
        """),
        L.task("Count the lines of the file in `path`, so the program shows `5`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-lines", lines=7)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-lines", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-lines",
        [
            "Look at the worked example: a `with` line opens the file, and a loop under it goes through the lines.",
            "Inside `for line in f:`, each round adds 1 to `count`. `print(count)` stays at the left edge.",
            "`with open(path) as f:`, then indented `for line in f:`, and indented once more ___.",
        ],
        why="`with open(path) as f:` opens the file, `for line in f:` has one round for each line, and `count = count + 1` adds 1 in each round. `print(count)` comes after the `with` block, so it runs once: 5.",
    )
    return


@app.cell(hide_code=True)
def text_newline():
    mo.md(r"""
    ### Exercise 2. The invisible character
    """)
    return


@app.cell(hide_code=True)
def choice_newline_cell():
    choice_newline = L.choice(
        "newline",
        question="In `shopping.txt`, the second line is `milk`. Inside `for line in f:`, what does `len(line)` give for that line?",
        options={
            "five": "`5`",
            "four": "`4`",
            "one": "`1`",
        },
        correct="five",
        why={
            "five": "`m`, `i`, `l`, `k`, and the newline at the end: 5 characters. `len(line.strip())` would give 4.",
            "four": "You see four letters, but at the end of the line there is a character you do not see. Look at the picture again.",
            "one": "`len` counts the characters of this one line, not lines. Look at the picture again.",
        },
    )
    choice_newline
    return (choice_newline,)


@app.cell(hide_code=True)
def text_exercise_long():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The long items

        Show the items of `shopping.txt` with more than 4 letters, each on its own line, without empty lines. Replace the `...` in the loop. The check also tries another file.
        """),
        L.task("Show the items with more than 4 letters: `bread`, `apples` and `cheese`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_long():
    box_long = L.code_box("long-items", lines=7)
    box_long
    return (box_long,)


@app.cell(hide_code=True)
def check_long(box_long):
    result_long = L.check("long-items", box_long.value)
    result_long
    return (result_long,)


@app.cell(hide_code=True)
def hints_long():
    L.hints(
        "long-items",
        [
            "Step 4 of the recipe: first `line.strip()`, then look at the length.",
            "An `if` with `len(...) > 4` shows only the long items. Which string do you measure and show: `line`, or the stripped one?",
            "`item = line.strip()`, then `if len(item) > 4:` and under it ___.",
        ],
        why="`item = line.strip()` drops the newline, so `len(item)` counts only the letters: `milk` and `eggs` have 4, and the `if` skips them. `print(item)` shows the item without an extra empty line.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Text: strings](/course/open/py-strings?lang=en) you counted the characters of a string and took single characters from it; a file gives you such a string, or one line at a time, as a [for loop](/course/open/py-for?lang=en) goes through a list. In a project later on, this is how a whole book gets into your program. In the next lesson you use code that others wrote, from modules.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_newline, result_count, result_long):
    L.wrap_up(result_count, choice_newline, result_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
