# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson py-project-letter-counts. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, plots
    from everything_ai.lesson import lesson

    L = lesson("py-project-letter-counts", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en) and [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="What does this program show?\n\n```python\nscores = {\"a\": 3, \"b\": 5}\nnames = [\"b\", \"a\"]\nprint([scores[n] for n in names])\n```",
        options={
            "keys": "`[\"b\", \"a\"]`",
            "order": "`[3, 5]`",
            "values": "`[5, 3]`",
        },
        correct="values",
        why={
            "keys": "The comprehension makes a list of `scores[n]`, the values, not of the names. See the recipe in [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).",
            "order": "The new list follows the order of `names`, not the order of `scores`. Which name comes first?",
            "values": "For each name in `names`, in that order, the comprehension takes its value: first `scores[\"b\"]`, then `scores[\"a\"]`.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def quick_check_bar():
    quick_bar = L.choice(
        "quick-bar",
        question="You want a bar for each of the days `mon`, `tue` and `wed`, as high as its rain. Which line draws it?",
        options={
            "bar": "`plt.bar([\"mon\", \"tue\", \"wed\"], [3, 5, 2])`",
            "plot": "`plt.plot([\"mon\", \"tue\", \"wed\"], [3, 5, 2])`",
            "swapped": "`plt.bar([3, 5, 2], [\"mon\", \"tue\", \"wed\"])`",
        },
        correct="bar",
        why={
            "bar": "`plt.bar` gets the names first, along the bottom, and the heights second.",
            "plot": "`plt.plot` joins points with a line. Which function draws bars? See the table of new words in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
            "swapped": "The order of the two lists matters. Which list goes along the bottom? See the recipe in [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en).",
        },
    )
    quick_bar
    return (quick_bar,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        This is the first step of your own language model. A language model guesses what comes next in a text. The simplest one there is only needs to know which letter usually follows which: after `t` comes `h` far more often than `x`. So you start by counting, in a real book: every letter, and every pair of letters next to each other. You keep both counting functions in your project; in a later lesson, the pair counts become a model that writes.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           0, here_label="← today", title="Where your project stands: your own LLM"),
        mo.md(r"""
        **An everyday example.** A phone keyboard that suggests the next word. It has seen which words people type after which, and counted them.
        """),
    ])
    return


@app.cell(hide_code=True)
def predict_letter():
    guess_letter = L.guess(
        "letter",
        question="Which letter do you think comes most often in plays by Shakespeare, if you leave out spaces and signs?",
        options={
            "e": "`e`",
            "t": "`t`",
            "a": "`a`",
        },
        reveal="**`e`.** In English, `e` is the most common letter, and Shakespeare is no exception: in the book of this lesson it comes more than 100,000 times. You count it yourself in exercise 3.",
    )
    guess_letter
    return (guess_letter,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Counting is a loop with a dictionary: each letter is a key, and its value is how often it appears. A pair is two characters next to each other, so it is a slice of two: `text[i:i + 2]`. Two new tools: `.lower()`, so that `T` and `t` count as the same letter, and `and`, so that a pair counts only when both its characters are letters.

    ### New words and symbols

    | Word | What it means | In code |
    |---|---|---|
    | **`.lower()`** | gives the text in small letters | `"To Be".lower()` gives `"to be"` |
    | **`in` on a text** | `True` when the character on its left appears in the text on its right | `"e" in "aeiou"` gives `True` |
    | **`and`** | `True` only when the condition on its left and the one on its right are both `True` | `"e" in "aeiou" and "x" in "aeiou"` gives `False` |
    | **pair** | two characters next to each other in a text | `text[i:i + 2]` |
    | **count** | how often a key appears: the value under that key | `counts["e"]` |
    | **project** | your own code that grows lesson after lesson, in `work/projects/`; this one is `llm` | `from llm import letters` |

    The book is *Tiny Shakespeare*: about 40,000 lines from plays by William Shakespeare, which are in the public domain. It is in the course folder, as `data/tiny-shakespeare.txt`.
    """)
    return


@app.cell(hide_code=True)
def picture_pair():
    mo.vstack([
        draw.list_boxes(list("to be"), index_label="index", span=(1, 3), span_label="text[1:3]", title="A pair is a slice of two"),
        mo.md(r"""
        Above each character is its index; the empty box at index 2 is the space. The slice `text[1:3]` takes the characters at index 1 and 2: `"o "`. That pair has a space in it, so it does not count. `text[3:5]` is `"be"`: both are letters, so that one counts.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.vstack([
        L.watch_out("The first time a letter comes by, it is not in `counts` yet: `counts[ch] + 1` then stops. Look first with `if ch in counts:`, and start a new letter at 1."),
        mo.md(r"""
        ### Recipe

        To count what is in a text:

        1. Read the text: `with open(path) as f:`, and under it `text = f.read()`.
        2. Make it uniform: `text.lower()`, so that capitals count as small letters.
        3. Count with a dictionary: loop over the text, keep only what you want to count (with `and` when two things must both be true), and add 1 to its count, or start it at 1.
        4. Draw the counts: the keys in order with `sorted`, their counts with a list comprehension, and `plt.bar`.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The vowels in the poem from [Reading a text file](/course/open/py-files?lang=en). The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_vowels():
    import matplotlib.pyplot as plt

    # 1. read the text
    with open("lessons/py-files/poem.txt") as _f:
        _text = _f.read()
    # 2. make it uniform: small letters only
    _text = _text.lower()
    # 3. count with a dictionary
    _vowels = {}
    for _ch in _text:
        if _ch in "aeiou":
            if _ch in _vowels:
                _vowels[_ch] = _vowels[_ch] + 1
            else:
                _vowels[_ch] = 1
    print(_vowels)
    # 4. draw: the keys in order, their counts in a comprehension
    _names = sorted(_vowels)
    _heights = [_vowels[_v] for _v in _names]
    plt.bar(_names, _heights)
    plt.title("Vowels in the poem")
    plt.ylabel("how often")
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Under the cell appear the counts, as a dictionary, and a bar chart with a bar for `a`, `e`, `i`, `o` and `u`. `sorted(_vowels)` gives the keys from a to u, and the comprehension takes their counts in that same order, so each bar gets its own count.
    """)
    return


@app.cell
def example_pairs():
    # step 3 for pairs: each slice of two, counted like a letter
    _word = "banana"
    _pairs = {}
    for _i in range(len(_word) - 1):
        _pair = _word[_i:_i + 2]
        if _pair in _pairs:
            _pairs[_pair] = _pairs[_pair] + 1
        else:
            _pairs[_pair] = 1
    print(_pairs)
    return


@app.cell(hide_code=True)
def text_after_pairs():
    mo.md(r"""
    Pairs are counted the same way, with a slice of two as the key. `"banana"` has 6 letters, and 5 pairs: `ba`, `an`, `na`, `an` and `na`. The last letter starts no pair of its own, so the loop stops one place early: `range(len(_word) - 1)` gives `_i` from 0 to 4. The cell prints `{'ba': 1, 'an': 2, 'na': 2}`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_letters():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Count the letters

        Finish `count_letters(text)`: it gives back a dictionary with each letter of `text` and how often it comes, only for the letters a to z, with capitals counted as small letters. The box starts with `LETTERS`, the 26 small letters, and a loop that goes over the text in small letters, one character `ch` at a time; `counts` is the dictionary it fills. Replace the `...`. The check calls `count_letters("Hello, World")`: that should give `{'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1}`. When your function is right, the check saves it in your project, as `work/projects/llm/letters.py`.
        """),
        L.task("Make `count_letters(text)` count each letter a to z, small or capital, in a dictionary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_letters():
    box_letters = L.code_box("count-letters", lines=12)
    box_letters
    return (box_letters,)


@app.cell(hide_code=True)
def check_letters(box_letters):
    result_letters = L.check("count-letters", box_letters.value)
    result_letters
    return (result_letters,)


@app.cell(hide_code=True)
def hints_letters():
    L.hints(
        "count-letters",
        [
            "Step 3 of the recipe, and the vowels in the worked example: the same loop, with `LETTERS` instead of `\"aeiou\"`.",
            "Count `ch` only when it is in `LETTERS`. If it is already in `counts`, add 1 to its count; if not, start it at 1.",
            "`if ch in LETTERS:`, and under it, indented once more, `if ch in counts:` with ___ under it; then `else:` with `counts[ch] = 1`.",
        ],
        why="`text.lower()` makes every capital small. `if ch in LETTERS:` skips spaces and signs. A letter that is already in `counts` gets 1 more; a new letter starts at 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pairs():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Count the pairs

        Finish `count_pairs(text)`: it gives back a dictionary with each pair of two letters next to each other and how often it comes. A pair with a space or a sign in it does not count. The loop already takes each slice of two, `pair`. Replace the `...`. The check calls `count_pairs("Hello")`: that should give `{'he': 1, 'el': 1, 'll': 1, 'lo': 1}`. When your function is right, the check saves it as `work/projects/llm/pairs.py`.
        """),
        L.task("Make `count_pairs(text)` count each pair of two letters next to each other, in a dictionary."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pairs():
    box_pairs = L.code_box("count-pairs", lines=13)
    box_pairs
    return (box_pairs,)


@app.cell(hide_code=True)
def check_pairs(box_pairs):
    result_pairs = L.check("count-pairs", box_pairs.value)
    result_pairs
    return (result_pairs,)


@app.cell(hide_code=True)
def hints_pairs():
    L.hints(
        "count-pairs",
        [
            "The same counting as in exercise 1, but now for `pair` instead of `ch`.",
            "A pair counts only when both of its characters are letters: `pair[0]` and `pair[1]` must both be in `LETTERS`.",
            "`if pair[0] in LETTERS and ___:`, and under it the same counting as in exercise 1, with `pair` as the key.",
        ],
        why="`text[i:i + 2]` takes two characters next to each other. `if pair[0] in LETTERS and pair[1] in LETTERS:` skips pairs with a space or a sign. Then each pair is counted like a letter in exercise 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_chart():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The letters of a book

        Now your project goes to work. The box reads the whole book, counts its letters with your own `count_letters`, imported from your project, and puts the letters in order in `names`. Replace the `...` with the lines that make a list `heights`, one count per letter in `names`, and draw a bar chart of it, with a title and a y-axis label. The check compares your chart with the course's own count of the book, and shows your chart under its message.
        """),
        L.task("Draw a bar per letter of the book, as high as its count, with your own `count_letters`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chart():
    box_chart = L.code_box("letter-chart", lines=14)
    box_chart
    return (box_chart,)


@app.cell(hide_code=True)
def check_chart(box_chart, result_letters):
    result_letters  # check again when exercise 1 saves your module
    result_chart = L.check("letter-chart", box_chart.value)
    result_chart
    return (result_chart,)


@app.cell(hide_code=True)
def hints_chart():
    L.hints(
        "letter-chart",
        [
            "Step 4 of the recipe, as at the end of the worked example.",
            "Make `heights` with a list comprehension over `names`: the count of each letter. Then draw `names` and `heights` as bars, with a title and a label for the y-axis.",
            "`heights = [___ for ch in names]`, then `plt.bar(names, heights)`, `plt.title(\"...\")` and `plt.ylabel(\"...\")`.",
        ],
        why="`[counts[ch] for ch in names]` takes the count of each letter, from a to z. `plt.bar(names, heights)` draws a bar per letter. The tallest bar is `e`, then `t` and `o`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    Your `count_pairs` knows which letter follows which. The cell below asks it, for the whole book, which letters come after `t`, and draws the five most common ones. That is already the core of a language model: after `t`, guess `h`. The cell uses your own module, so it works once exercise 2 is done.
    """)
    return


@app.cell(hide_code=True)
def demo_after_t(result_pairs):
    result_pairs  # run again when exercise 2 saves your module
    _ok = C.pairs_ok()
    if _ok is None:
        _out = mo.md("*Finish exercise 2 first: then your `count_pairs` is in your project, and this cell uses it.*")
    elif not _ok:
        _out = mo.md("*The `count_pairs` in `work/projects/llm/pairs.py` no longer counts right, or stops with an error. Did you change that file? Delete it, and check exercise 2 again: its check then saves your function again.*")
    else:
        import matplotlib.pyplot as _plt
        from llm import pairs as _pairs
        with open("data/tiny-shakespeare.txt") as _f:
            _book = _f.read()
        _counts = _pairs.count_pairs(_book)
        _after_t = sorted(((n, p[1]) for p, n in _counts.items() if p[0] == "t"), reverse=True)[:5]
        _fig = _plt.figure(figsize=(6.4, 2.8))
        _plt.bar([b for _, b in _after_t], [n for n, _ in _after_t])
        _plt.title("What comes after t in Shakespeare")
        _plt.ylabel("how often")
        _out = mo.Html(plots.render(_fig))
        _plt.close(_fig)
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Dictionaries](/course/open/py-dicts?lang=en) you counted words; now you counted a whole book. In [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en) the check saved a function in your project; now your project `llm` has its first two files. In [Drawing numbers: line, scatter and bar charts](/course/open/py-plotting?lang=en) you drew bars; now they show what Shakespeare wrote. In a later lesson, *A bigram model by counting*, your pair counts become a model that writes text, letter by letter. There you count every character, spaces and signs too, so that the model can write them as well.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_chart, result_letters, result_pairs):
    L.wrap_up(result_letters, result_pairs, result_chart)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
