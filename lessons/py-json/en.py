# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-json. Route: course/route/lessen/bouwen.yaml. Checks and the files of this lesson: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("py-json", "en")
    C = L.checks   # the paths of the lesson's files, and the dict in the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Dictionaries](/course/open/py-dicts?lang=en) and [Reading a text file](/course/open/py-files?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dict():
    quick_dict = L.choice(
        "quick-dict",
        question="What does this program show?\n\n```python\nages = {\"Ada\": 36, \"Alan\": 41}\nprint(ages[\"Alan\"])\n```",
        options={
            "key": "`Alan`",
            "value": "`41`",
            "pair": "`\"Alan\": 41`",
        },
        correct="value",
        why={
            "key": "`ages[\"Alan\"]` looks up the key `\"Alan\"` and gives its value. See the idea in [Dictionaries](/course/open/py-dicts?lang=en).",
            "value": "The key `\"Alan\"` gives its value, 41.",
            "pair": "Square brackets with a key give only the value. See the idea in [Dictionaries](/course/open/py-dicts?lang=en).",
        },
    )
    quick_dict
    return (quick_dict,)


@app.cell(hide_code=True)
def quick_check_read():
    quick_read = L.choice(
        "quick-read",
        question="`notes.txt` has three lines. What is `text` after this?\n\n```python\nwith open(\"notes.txt\") as f:\n    text = f.read()\n```",
        options={
            "string": "One string with the whole file, newlines included",
            "lines": "A list of the three lines",
            "first": "The first line only",
        },
        correct="string",
        why={
            "lines": "`f.read()` does not split the file into lines. See the recipe in [Reading a text file](/course/open/py-files?lang=en).",
            "first": "`f.read()` reads the file to the end. See the recipe in [Reading a text file](/course/open/py-files?lang=en).",
            "string": "`f.read()` gives the whole file as one string, with a `\\n` between the lines.",
        },
    )
    quick_read
    return (quick_read,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Programs pass data to each other as text: a file with settings, a saved list of scores, the answer a web service sends back. A dict or a list only lives inside your running program; to store it or send it, it has to become text, and the program that gets it has to turn it back. JSON is the way of writing that almost every program understands. Later in the course, a language model answers your program in JSON, and every run of an experiment is kept as a line of JSON.

    **An everyday example.** A form with fixed fields: *name*, *date of birth*, *allergies*. Anyone who gets it can read it, because the fields always have the same names. JSON is such a form for programs.
    """)
    return


@app.cell(hide_code=True)
def predict_dumps():
    guess_dumps = L.guess(
        "dumps",
        question="What do you think this program shows?\n\n```python\nimport json\nprint(json.dumps({\"name\": \"Ada\", \"done\": True}))\n```",
        options={
            "python": "`{'name': 'Ada', 'done': True}`",
            "json": "`{\"name\": \"Ada\", \"done\": true}`",
            "error": "An error: `True` is not text",
        },
        reveal="**`{\"name\": \"Ada\", \"done\": true}`.** `json.dumps` writes the dict as JSON text: with double quotes, and `true` in small letters, the way JSON writes it.",
    )
    guess_dumps
    return (guess_dumps,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    JSON writes data as text, with the same building blocks as Python: dicts become `{...}` with pairs of a key and a value, lists become `[...]`, and inside them are strings, numbers, true or false, and null for *nothing*. The module `json` translates both ways.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **JSON** | a way to write dicts, lists, strings, numbers, true, false and null as text, which almost every programming language can read and write | `{"name": "Ada", "age": 36}` |
    | **json** | the module that translates between JSON text and Python values | `import json` |
    | **dumps** | gives the JSON text of a value: *dump to a string* | `json.dumps(value)` |
    | **loads** | turns JSON text into Python values: *load from a string* | `json.loads(text)` |
    | **load** | reads JSON from an open file | `json.load(f)` |

    JSON has its own words for three values: `true`, `false` and `null` are `True`, `False` and `None` in Python. A string in JSON always has double quotes, `"Ada"`, never single ones. A key is always a string. A file that holds JSON usually has a name that ends in `.json`.
    """)
    return


@app.cell(hide_code=True)
def picture_json():
    _py, _js = C.picture_lines()
    mo.vstack([
        draw.marked_lines([("A dict, as Python shows it", [(_line, _m) for _line, _m in zip(_py, ["3", "1", "2", ""])]),
                           ("The same dict as JSON text, from json.dumps", [(_line, _m) for _line, _m in zip(_js, ["3", "1", "2", ""])])],
                          {"1": "True becomes true", "2": "None becomes null", "3": "single quotes become double quotes"},
                          title="One dict, in Python and as JSON"),
        mo.md(r"""
        The two look almost the same, and that is on purpose: JSON was made to look like the dicts and lists of programming languages. But the bottom one is text, a string you can save or send, and the top one is a dict in your running program. `json.loads` turns the bottom one back into the top one.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("`json.load` and `json.loads` differ by one letter. Without the s it reads from an open file `f`; with the s, from a string. Give `json.loads` a path, and it tries to read the path itself as JSON, and stops with a `JSONDecodeError`. And on text that only looks like JSON, with single quotes or `True`, `json.loads` stops with that error too: the error says on which line and column the text went wrong."),
        mo.md(r"""
        ### Recipe

        To turn data into JSON text and back:

        1. Write `import json` at the top.
        2. From a file: open it with `with open(path) as f:`, and read it with `json.load(f)`.
        3. Reach inside with square brackets: `[key]` for a dict, `[index]` for a list, one after the other.
        4. From Python to text: `json.dumps(value)`.
        5. From text to Python: `json.loads(text)` gives dicts, lists, strings, numbers, `True`, `False` and `None`.
        6. Never write JSON by hand with `+`: put your values in a dict, and let `json.dumps` write it, quotes and all.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The file `pancakes.json` holds a recipe: its name, how many people it serves, whether it is vegetarian, a list of ingredients, and notes. Each ingredient is a dict of its own. This is the text in the file:

    ```json
    {
      "name": "pancakes",
      "servings": 4,
      "vegetarian": true,
      "ingredients": [
        {"item": "flour", "grams": 250},
        {"item": "milk", "grams": 450},
        {"item": "eggs", "grams": 110},
        {"item": "butter", "grams": 25}
      ],
      "notes": null
    }
    ```

    The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_recipe():
    # 1. the module
    import json

    # 2. read the file: json.load turns its JSON into Python values
    with open("lessons/py-json/pancakes.json") as _f:
        _recipe = json.load(_f)
    print(_recipe["name"], _recipe["servings"], _recipe["vegetarian"], _recipe["notes"])
    # 3. a list of dicts inside the dict: first [key], then [index], then [key]
    print(_recipe["ingredients"][0]["item"], _recipe["ingredients"][0]["grams"])
    # 4. back to text: json.dumps writes true and null, with double quotes
    _text = json.dumps({"servings": 8, "vegetarian": True, "notes": None})
    print(_text)
    # 5. and from text to Python again
    print(json.loads(_text)["servings"] * 2)
    return (json,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear four lines. `pancakes 4 True None`: in the file it says `true` and `null`, and `json.load` made them `True` and `None`. `flour 250`: the first ingredient, a dict inside the list inside the dict. `{"servings": 8, "vegetarian": true, "notes": null}`: text again, the JSON way. And `16`: `json.loads` turned that text back into a dict, whose `"servings"` is the number 8 again, not the text `8`. An `import` makes a name, just like `=`: this cell makes `json`, so `import json` in another cell breaks *one name, one cell*. The exercise boxes run on their own, so each box has its own `import` line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_total():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. All the grams

        Write a function `total_grams(path)` that reads the recipe in the JSON file at `path` and gives back the grams of all its ingredients added up. The check tries it on `pancakes.json`, and on two other recipes in the same folder.
        """),
        L.task("Make `total_grams(\"lessons/py-json/pancakes.json\")` give `835`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_total():
    box_total = L.code_box("total-grams", lines=9)
    box_total
    return (box_total,)


@app.cell(hide_code=True)
def check_total(box_total):
    result_total = L.check("total-grams", box_total.value)
    result_total
    return (result_total,)


@app.cell(hide_code=True)
def hints_total():
    L.hints(
        "total-grams",
        [
            "Step 2 of the recipe reads the file; the worked example shows how to reach the ingredients.",
            "`recipe[\"ingredients\"]` is a list of dicts. Go through it with a `for` loop, and add up something from each dict.",
            "`with open(path) as f:` and `recipe = json.load(f)`, then `total = 0`, a loop over `recipe[\"ingredients\"]` that adds ___, and `return total`.",
        ],
        why="`json.load(f)` turns the file into a dict. Its `\"ingredients\"` is a list with a dict for each ingredient, so the loop adds `ingredient[\"grams\"]` of each: 250 + 450 + 110 + 25 = 835.",
    )
    return


@app.cell(hide_code=True)
def text_true_null():
    mo.md(r"""
    ### Exercise 2. JSON's own words
    """)
    return


@app.cell(hide_code=True)
def choice_true_null_cell():
    choice_true_null = L.choice(
        "true-null",
        question="What does this give?\n\n```python\nimport json\njson.loads('{\"done\": true, \"left\": null}')\n```",
        options={
            "values": "`{'done': True, 'left': None}`",
            "strings": "`{'done': 'true', 'left': 'null'}`",
            "error": "An error: `true` and `null` are not Python",
        },
        correct="values",
        why={
            "strings": "Without quotes around them, `true` and `null` are not strings in JSON: they are JSON's own words for two values. Look at the table of new words again.",
            "error": "The text is JSON, not Python, and in JSON `true` and `null` are fine. `json.loads` translates them. Look at the table of new words again.",
            "values": "`json.loads` translates JSON's `true` into Python's `True`, and `null` into `None`.",
        },
    )
    choice_true_null
    return (choice_true_null,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A weather report as JSON

        Write a function `report(city, mm)` that gives back JSON text with three keys: `"city"`, `"mm"` (the rain in millimeters) and `"rain"`, which is true when `mm` is more than 0. Another program will read your text with `json.loads`, so it has to be valid JSON.
        """),
        L.task("Make `report(\"Utrecht\", 2.5)` give JSON text with `\"city\": \"Utrecht\"`, `\"mm\": 2.5` and `\"rain\": true`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=7)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "Step 6 of the recipe: do not glue the text together yourself.",
            "Make a dict with the three keys first. What do you put under `\"rain\"`, so that `json.dumps` writes `true` or `false`?",
            "`return json.dumps({\"city\": city, \"mm\": mm, \"rain\": ___})`",
        ],
        why="`mm > 0` is a bool, `True` or `False`, and `json.dumps` writes it as `true` or `false`. It also puts double quotes around the city, whatever is in its name, so the text is valid JSON every time.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Dictionaries](/course/open/py-dicts?lang=en) you looked up values by their key; JSON stores such dicts, and lists of them, as text. In [Reading a text file](/course/open/py-files?lang=en) you read a file as a string; `json.load` reads it and makes Python values of it at once. And in [Using code from modules](/course/open/py-imports?lang=en) you brought in modules; `json` is one more. Later in the course, a language model sends its answers to your program in this form, and you check whether its JSON is valid.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_true_null, result_report, result_total):
    L.wrap_up(result_total, choice_true_null, result_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
