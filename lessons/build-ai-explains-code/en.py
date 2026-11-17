# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-ai-explains-code. Route: course/route/lessen/bouwen.yaml. Checks, the 24 answers and their picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-ai-explains-code", "en")
    C = L.checks   # what Qwen3 answered when it explained, worked out and renamed two small functions, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en), and one about [Functions](/course/open/py-functions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_chat():
    quick_chat = L.choice(
        "quick-chat",
        question="A colleague sends you a function you do not understand, and you want it explained, with the code beside it. Which kind of help fits?",
        options={
            "completion": "Code completion",
            "chat": "A chat about your code",
            "agent": "A coding agent",
        },
        correct="chat",
        why={
            "chat": "You want an answer that you then use yourself: a chat, with the code pasted in.",
            "completion": "Code completion suggests the next line where you type; it does not explain. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
            "agent": "An agent changes files and runs them; for an explanation, that is more than you need. See [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en).",
        },
    )
    quick_chat
    return (quick_chat,)


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="A function `double(x)` has one line: `return x * 2`. What does `double(double(3))` give?",
        options={
            "twelve": "12",
            "six": "6",
            "nine": "9",
        },
        correct="twelve",
        why={
            "twelve": "`double(3)` gives 6, and `double(6)` gives 12: what a function returns is a value you can pass on.",
            "six": "That is `double(3)`. What happens to that 6? See [Functions](/course/open/py-functions?lang=en).",
            "nine": "`double` multiplies by 2, twice. See [Functions](/course/open/py-functions?lang=en).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of the code you will work with, you did not write: it comes from a colleague, from the internet, or from an AI. An assistant can explain such code in plain words, summarize what a file does, and take chores off your hands, such as better names. That saves time. But an explanation tells you what the code *looks* meant to do; the code does what its lines say.

    **An everyday example.** A jar in the cupboard says *strawberry jam* on the label. Usually it is. Whether it is this time, you find out with a spoon, not by reading the label again. That is only a comparison; below you see what Qwen3 said about this function, which a learner wrote to leave out the highest and the lowest score:

    ```python
    def average_without_extremes(scores):
        scores = sorted(scores)
        middle = scores[1:-1]
        return sum(middle) / len(scores)
    ```

    `sorted(scores)` gives the scores in order, from small to large, and `sum(middle)` adds up a list. `scores[1:-1]` is a slice: from index 1 up to just before the last item, so without the first and the last.
    """)
    return


@app.cell(hide_code=True)
def predict_explanations():
    guess_explanations = L.guess(
        "explanations",
        question="The course asked Qwen3 eight times to explain in two sentences what this function returns. How many of the eight explanations noticed what is wrong with it?",
        options={
            "most": "Most of them",
            "none": "None",
            "few": "One or two",
        },
        reveal="**None.** All eight say that it returns the average of the scores without the first and the last. That is what the name promises, and what the learner meant. The last line divides by `len(scores)`, the number of *all* the scores: for five scores it divides by 5, while only three are added up.",
    )
    guess_explanations
    return (guess_explanations,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word or symbol | What it means |
    |---|---|
    | `sorted(x)` | a new list with the items of `x` in order, from small to large |
    | `sum(x)` | the items of the list `x` added up |
    | **docstring** | a text between three quote marks, such as `'''Return the area.'''`, right under the `def` line, that says what the function does |

    ### Three things to ask, and how you check them

    | You ask for | For example | How you check it |
    |---|---|---|
    | an explanation | *What does this function return?* | work out one small case by hand, then run it: does the code give what the explanation says? |
    | a summary | *What does this file do?* | open the parts the summary names, and see whether they do that |
    | a chore | better names, a docstring, a first test | run the same cases before and after: a chore changes how code reads, never what it gives |

    Any chat assistant does this, with the code pasted in, and so do the assistants inside code editors (examples in November 2026: ChatGPT, Claude, Gemini and GitHub Copilot Chat).

    **The run.** The course asked Qwen3-0.6B three things, eight times each: to explain in two sentences what `average_without_extremes` returns; what it returns for `[1, 2, 3, 4, 100]`, with the number only; and to give this function and its variables clear names:

    ```python
    def f(a, b):
        c = 0
        for d in a:
            if d > b:
                c = c + 1
        return c
    ```
    """)
    return


@app.cell(hide_code=True)
def picture_runs():
    mo.vstack([
        mo.Html(C.runs_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_answers():
    L.explore("answers", "answer", lambda _v: C.answer_frame(int(_v), L.lang), start=15, low=1, high=3 * C.SEEDS, step=1, role="",
              task="Slide through the 24 answers: eight explanations, eight answers to what it returns, and eight renames. Look at how sure of itself each one sounds.")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("A large assistant explains this function better than Qwen3-0.6B, and often spots the bug. But an explanation leans on the names and the shape of the code, so a bug that looks like what was meant is easy to miss, for a large model too. Whatever the model, one case worked out by hand shows it.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: let AI explain code

    1. **Ask for the explanation in plain words,** and what the code returns for one example.
    2. **Pick a small case you can work out by hand,** and work it out from the lines, not from the explanation.
    3. **Run it.** Does the code give what you worked out, and what the explanation says? Where the three differ, read that line again.
    4. **For a chore, keep the cases.** Run them before and after: the outcome must not change.
    5. **Read what changed,** line by line, before you keep it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A colleague's function has no docstring. You ask an assistant for one. Suppose it writes: *Returns the total of the prices, minus the discount as a percentage.*

    ```python
    def total_price(prices, discount):
        total = 0
        for price in prices:
            total = total + price
        return total - discount
    ```

    1. **The explanation:** the total of the prices, minus a percentage.
    2. **By hand:** for `total_price([10, 20], 10)`, the lines add 10 and 20, which is 30, and subtract 10: 20. A discount of 10 percent would give 27.
    3. **Run it:** `total_price([10, 20], 10)` gives `20`. The code subtracts an amount, not a percentage. Your colleague confirms that the discount is an amount: the docstring is wrong, not the code.
    4. **The chore:** you write the docstring yourself, *Return the sum of the prices minus the discount, an amount.* Before and after, `total_price([10, 20], 10)` gives `20`: a docstring changes nothing in what the function gives.
    5. **Read what changed:** one line, the docstring.
    """)
    return


@app.cell(hide_code=True)
def text_read_average():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Read the lines, not the explanation
    """)
    return


@app.cell(hide_code=True)
def choice_read_average_cell():
    choice_read_average = L.choice(
        "read-average",
        question="What does `average_without_extremes([10, 20, 30])` return, with the function as it is?",
        options={
            "twenty": "20",
            "third": "about 6.67",
            "ten": "10",
        },
        correct="third",
        why={
            "third": "Sorted: `[10, 20, 30]`. The middle: `[20]`. Its sum, 20, is divided by `len(scores)`, which is 3: about 6.67. The explanation promised 20, the average of the middle.",
            "twenty": "That is what the explanation promises: the average of the middle. Which number does the last line divide by?",
            "ten": "Look at which scores `scores[1:-1]` keeps: the first and the last go.",
        },
    )
    choice_read_average
    return (choice_read_average,)


@app.cell(hide_code=True)
def text_fix_average():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Make the code do what the explanation says

        The box has the function that Qwen3 explained. Change it so that it does what all eight explanations said: the average of the scores without the highest and the lowest. The check calls it with `[1, 2, 3, 4, 100]`, and with other lists.
        """),
        L.task("Make `average_without_extremes([1, 2, 3, 4, 100])` give `3.0`, and the right average for every list."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fix_average():
    box_average = L.code_box("fix-average", lines=6)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_fix_average(box_average):
    result_average = L.check("fix-average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_fix_average():
    L.hints(
        "fix-average",
        [
            "Work out `[1, 2, 3, 4, 100]` by hand: which scores are in `middle`, what is their sum, and how many are there?",
            "The sum is right: 9. It is divided by the wrong number: `len(scores)` counts all five scores, and only three are in the middle.",
            "Divide by the number of scores in `middle`: `return sum(middle) / len(___)`.",
        ],
        why="`return sum(middle) / len(middle)` divides the sum of the middle scores by how many there are: 9 / 3 = 3.0. `len(scores) - 2` gives the same number, and is right too.",
    )
    return


@app.cell(hide_code=True)
def text_rename_count():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Do the chore Qwen3 did not do

        None of the eight answers renamed anything in `f`. Do it yourself: call the function `count_above`, and give its arguments and variables names that say what they hold. Change nothing in what it does. The check calls `count_above([3, 8, 5, 10], 4)`, and other cases.
        """),
        L.task("Rename `f` to `count_above`, with clear names inside, so that it gives exactly what `f` gave."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rename_count():
    box_rename = L.code_box("rename-count", lines=7)
    box_rename
    return (box_rename,)


@app.cell(hide_code=True)
def check_rename_count(box_rename):
    result_rename = L.check("rename-count", box_rename.value)
    result_rename
    return (result_rename,)


@app.cell(hide_code=True)
def hints_rename_count():
    L.hints(
        "rename-count",
        [
            "First work out what `f` does: what is `a`, what is `b`, and what does `c` count?",
            "`a` is a list of numbers, `b` the limit, `d` one number from the list, and `c` counts the numbers above the limit. Rename each one everywhere it is used.",
            "The loop line becomes `for number in ___:`, with the new name of `a`; rename `b` and `c` the same way, everywhere they are used.",
        ],
        why="`count_above(numbers, limit)` with `count`, `number` and `if number > limit:` gives for every list what `f` gave: only the names changed. `count_above([3, 8, 5, 10], 4)` gives `3`.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Functions](/course/open/py-functions?lang=en) you wrote functions yourself; here you read one that someone else wrote, with an AI explaining it, and checked the explanation with one case. In [From autocomplete to agents: help with code](/course/open/build-coding-help?lang=en) you chose a chat for questions like these. The cases you work out by hand are tests waiting to be written down: later in this module, [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en) shows how.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read_average, result_average, result_rename):
    L.wrap_up(choice_read_average, result_average, result_rename)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
