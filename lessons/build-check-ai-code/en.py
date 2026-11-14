# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-check-ai-code. Route: course/route/lessen/bouwen.yaml. Checks, the 16 functions and their picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-check-ai-code", "en")
    C = L.checks   # what the 16 functions Qwen3 wrote give for six inputs each, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en), and one about [Reading error messages and finding bugs](/course/open/py-errors?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_examples():
    quick_examples = L.choice(
        "quick-examples",
        question="An agent wrote a function from your spec, which has three examples of what must come out. What do you do with those examples?",
        options={
            "tests": "Run them as tests, after every step",
            "plan": "Put them in the plan, so the agent reads them twice",
            "nothing": "Nothing: the agent has read them already",
        },
        correct="tests",
        why={
            "tests": "The examples say what must come out; run them on the code, and you see straight away whether it does.",
            "plan": "The plan says how, in steps. Whether the code does what the examples say, you only see by running them. See [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en).",
            "nothing": "Not one of the 24 programs that had the example in front of them got it right. See [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en).",
        },
    )
    quick_examples
    return (quick_examples,)


@app.cell(hide_code=True)
def quick_check_traceback():
    quick_traceback = L.choice(
        "quick-traceback",
        question="Python stops with a traceback of eight lines. Which line do you read first?",
        options={
            "first": "The first line",
            "middle": "The line in the middle, where your function is",
            "last": "The last line",
        },
        correct="last",
        why={
            "last": "The last line has the kind of error and what went wrong; then you read the line of your own code just above it.",
            "first": "The first line only says that a traceback follows. See [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
            "middle": "That line matters, but second: first you want to know what went wrong. See [Reading error messages and finding bugs](/course/open/py-errors?lang=en).",
        },
    )
    quick_traceback
    return (quick_traceback,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Code from AI usually looks right: neat names, comments, no errors when it runs. Whether it gives the right answer is another question, and you find that out by trying it on cases where you know the answer. Teams that work with AI do this for every change, with tests, a tool that reads the code, and a person who reads it too.

    **An everyday example.** You do not find out whether a cake is done by reading the recipe again: you stick a skewer in. That is only a comparison; below is a real function. The course asked Qwen3 eight times for a function `count_vowels(text)` that counts the vowels (a, e, i, o, u) in a text. This is what it wrote:

    ```python
    def count_vowels(text):
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in text if char in vowels)
    ```

    `{'a', 'e', 'i', 'o', 'u'}` is a *set*: a collection of values, like a list, and here it is only used to ask whether a letter is in it. `char in vowels` is `True` when the letter `char` is in the set. The last line adds 1 for each letter of `text` that is in `vowels`: `1 for char in text if char in vowels` is a `for` with an `if`, written inside the brackets of `sum`. Sets get a lesson of their own later, in [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en), and this way of writing a `for` comes back in [Building lists and dictionaries in one line](/course/open/py-comprehensions?lang=en).
    """)
    return


@app.cell(hide_code=True)
def predict_apple():
    guess_apple = L.guess(
        "apple",
        question="The course called each of the eight functions with six texts, among them `\"Apple\"`. For how many of the eight is `count_vowels(\"Apple\")` right, which is 2?",
        options={
            "all": "All eight",
            "none": "None",
            "half": "About half",
        },
        reveal="**None.** All eight are the same function, character for character, and the set has `'a'` but not `'A'`: `count_vowels(\"Apple\")` gives 1, and `count_vowels(\"AEIOU\")` gives 0. A test with only small letters, such as `\"hello\"`, never shows it.",
    )
    guess_apple
    return (guess_apple,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word or symbol | What it means |
    |---|---|
    | **test** | a few lines of code that run a function on a case and check the outcome; a **unit test** tests one small part, such as one function, on its own |
    | `assert` | `assert` with a condition after it: when the condition is `True`, nothing happens; when it is `False`, Python stops with an `AssertionError` |
    | **code review** | a person reads a change before it is kept: does it do what was asked, and can others understand it? |
    | **static analysis** | a tool reads the code without running it, and reports mistakes such as a name Python does not know |
    | **randomized controlled trial** | an experiment in which chance decides who works with something and who without, so that a difference afterwards comes from that one thing |

    A test with `assert` is one line per case:

    ```python
    assert count_vowels("hello") == 2
    assert count_vowels("Apple") == 2
    ```

    With Qwen3's `count_vowels`, the first line passes and the second stops with an `AssertionError`: that test found the bug. With a right `count_vowels`, both pass, and nothing happens.

    ### Three ways to check, and what each one finds

    | Check | What it does | What it finds | Tools (examples in November 2026) |
    |---|---|---|---|
    | a test | runs the code on a case you chose, and compares the outcome | a wrong answer on that case | pytest |
    | static analysis | reads the code without running it | mistakes that are wrong in any code, such as a name Python does not know, or a name that gets a value and is never used | Ruff, mypy, CodeQL |
    | a review | a person reads the change | whether it does what was asked, no more and no less, and whether others can follow it | the review step on GitHub or GitLab |

    Each finds what the other two miss. A test only sees the cases you wrote; a tool does not know what the code is for; a person misses what looks right.

    **What studies found.** In a randomized controlled trial by Peng and others (2023), developers with GitHub Copilot finished one task, a small web server in JavaScript, 55.8% faster than developers without it. Pearce and others (2022) had Copilot write 1,689 programs for 89 situations where security matters: about 40% had a known kind of security hole.
    """)
    return


@app.cell(hide_code=True)
def predict_metr():
    guess_metr = L.guess(
        "metr",
        question="In 2025, METR ran a randomized controlled trial with 16 experienced developers, on 246 tasks in large projects they knew well; chance decided for each task whether AI tools were allowed. Beforehand, they expected AI to make them 24% faster. What did the trial measure?",
        options={
            "slower": "With AI they were 19% slower",
            "faster": "With AI they were about 24% faster",
            "same": "It took the same time with and without AI",
        },
        reveal="**19% slower.** With AI tools, the tasks took them 19% longer. Afterwards, they still believed AI had made them 20% faster. A feeling of speed is no measurement, just as code that looks right is no test. The trial with 55.8% faster was one small task started from scratch; what holds depends on the work.",
    )
    guess_metr
    return (guess_metr,)


@app.cell(hide_code=True)
def text_run():
    mo.md(r"""
    **The run.** The course asked Qwen3-0.6B eight times for `count_vowels`, and eight times for a function `is_palindrome(text)` that says whether a text reads the same backwards, ignoring capitals and spaces. Then it ran each of the 16 functions on six texts, and compared what came out with the right answer.
    """)
    return


@app.cell(hide_code=True)
def picture_checks():
    mo.vstack([
        mo.Html(C.checks_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_programs():
    L.explore("programs", "function", lambda _v: C.program_frame(int(_v), L.lang), start=9, low=1, high=2 * C.SEEDS, step=1, role="",
              task="Slide through the 16 functions: eight times count_vowels, then eight times is_palindrome. Which ones look right, and which ones are?")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("A comment in code from AI says what the code is meant to do, not what it does: two of the eight is_palindrome say they remove everything that is not a letter or a digit, and keep the spaces. And a test can pass by luck: in \"Ab ba\" the space is exactly in the middle, so those two pass it too. Choose cases where a mistake would show: capitals, spaces, an empty text, the edges.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: check code that AI wrote

    1. **Write down cases before you read the code,** from the spec: ordinary ones, and the edges, such as an empty text, capitals, a space or zero.
    2. **Work out the right outcome of each case by hand,** and turn it into an `assert` line.
    3. **Run the tests.** When one fails, the mistake is in the code or in the test: work that case out by hand again.
    4. **Let a tool read the code** for mistakes that are wrong in any code, such as a name Python does not know.
    5. **Read the change yourself, or have someone else read it:** does it do what was asked, no more and no less, and can you tell why?
    6. **Keep the tests.** After every change, they run again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Worked example

        The spec: `is_palindrome(text)` gives `True` when a text reads the same backwards, ignoring capitals and spaces. This is the first of the eight versions Qwen3 wrote. You can test it without knowing how each line works:

        ```python
        def is_palindrome(text):
            # Remove all non-alphanumeric characters
            cleaned_text = ''.join([c for c in text if c.isalnum() or c == ' '])
            # Convert to lowercase
            cleaned_text = cleaned_text.lower()
            # Check if it's a palindrome
            return cleaned_text == cleaned_text[::-1]
        ```

        1. **Cases from the spec:** `"racecar"` is a palindrome; `"Taco cat"` is one too, once the capital and the space are ignored; `"hello"` is not; and `""`, an empty text, reads the same backwards.
        2. **As tests:**

            ```python
            assert is_palindrome("racecar") == True
            assert is_palindrome("Taco cat") == True
            assert is_palindrome("hello") == False
            assert is_palindrome("") == True
            ```

        3. **Run them:** the second line stops with an `AssertionError`, because this version gives `False` for `"Taco cat"`. Is the test wrong? By hand: without the capital and the space, *tacocat* backwards is *tacocat*. The test is right.
        4. **A tool that reads the code** finds no unknown name here, and cannot know that spaces must go: that is in the spec, not in the code.
        5. **A review** reads the comment and the line under it side by side: the comment says everything that is not a letter or a digit goes, and the line keeps `c == ' '`, the spaces. That is the bug.
        6. **Keep the four tests.** Once the line is fixed, all four pass, and they keep checking after every later change.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_catch_vowels():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Write a test that catches the bug

        The box starts with one test. It passes for Qwen3's `count_vowels`, and for a right one, so it catches nothing. Add a test, one line with `assert`, that a right `count_vowels` passes and Qwen3's does not. The check runs your tests on both.
        """),
        L.task("Add a line with `assert` that passes for a right `count_vowels` and fails for Qwen3's."),
    ])
    return


@app.cell(hide_code=True)
def exercise_catch_vowels():
    box_tests = L.code_box("catch-vowels", lines=4)
    box_tests
    return (box_tests,)


@app.cell(hide_code=True)
def check_catch_vowels(box_tests):
    result_tests = L.check("catch-vowels", box_tests.value)
    result_tests
    return (result_tests,)


@app.cell(hide_code=True)
def hints_catch_vowels():
    L.hints(
        "catch-vowels",
        [
            "Look at the table above: which of the six texts did all eight versions get wrong?",
            "Texts with a capital vowel: `\"Apple\"` has two vowels, the A and the e, and Qwen3's function counts 1.",
            "Add the line `assert count_vowels(\"Apple\") == ___`, with the number of vowels in Apple.",
        ],
        why="`assert count_vowels(\"Apple\") == 2` passes for a right function, which counts the A and the e, and fails for Qwen3's, which gives 1: this test catches the bug. `assert count_vowels(\"AEIOU\") == 5` does too.",
    )
    return


@app.cell(hide_code=True)
def text_fix_vowels():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Fix the bug

        The cell below has Qwen3's `count_vowels`. Change it so that it counts capital vowels too, and run the cell. The check calls it with `"Apple"`, and with other texts.
        """),
        L.task("Make `count_vowels(\"Apple\")` give `2`, and the right number for every text."),
    ])
    return


@app.cell
def exercise_fix_vowels():
    def count_vowels(text):
        vowels = {'a', 'e', 'i', 'o', 'u'}
        return sum(1 for char in text if char in vowels)
    return (count_vowels,)


@app.cell(hide_code=True)
def check_fix_vowels(count_vowels):
    try:
        _value = count_vowels
    except NameError:                  # no cell defines it, for example after a syntax error
        _value = L.MISSING
    result_fix = L.check("fix-vowels", _value)
    result_fix
    return (result_fix,)


@app.cell(hide_code=True)
def hints_fix_vowels():
    L.hints(
        "fix-vowels",
        [
            "Which letters of `\"Apple\"` are vowels, and which of them are in the set `vowels`?",
            "For Python, `'A'` and `'a'` are different letters, and the set only has the small ones.",
            "Put the capitals in the set too: `{'a', 'e', 'i', 'o', 'u', 'A', ___}`.",
        ],
        why="With `{'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}`, `char in vowels` is `True` for small and capital vowels: `count_vowels(\"Apple\")` gives 2, and `count_vowels(\"AEIOU\")` gives 5.",
    )
    return


@app.cell(hide_code=True)
def text_which_check():
    mo.md(r"""
    ### Exercise 3. Which check finds it?

    Six problems in code that AI wrote. For each one, pick the check that finds it: a test, static analysis, or a review.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_check():
    sort_checks = L.sort("which-check")
    sort_checks
    return (sort_checks,)


@app.cell(hide_code=True)
def check_which_check(sort_checks):
    result_sort = L.check("which-check", sort_checks)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def hints_which_check():
    L.hints(
        "which-check",
        [
            "Ask for each problem: does it show when you run the function on one case, can a tool see it without knowing what the code is for, or does it take a person who knows?",
            "Apple and Taco cat show as a wrong answer on one case. A name Python does not know, or a name that is never used, is wrong in any code.",
            "Test: Apple, Taco cat. Static analysis: Never used, Unknown name. Review: More than asked, 37.",
        ],
        why="A test runs the code on a case and sees a wrong answer: Apple and Taco cat. Static analysis reads the code and finds what is wrong in any code: a name never used, and a name Python does not know, also on a line no test reaches. A review takes a person who knows what the code is for: more than the spec asked, and a number nobody can explain.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Reading error messages and finding bugs](/course/open/py-errors?lang=en) you read a traceback after Python stopped; here you wrote lines that make Python stop on purpose when an answer is wrong. In [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en) you ran the spec's examples as tests after every step: now you can write those tests yourself, as `assert` lines. If you did [Let AI explain code, and do the chores](/course/open/build-ai-explains-code?lang=en), the cases you worked out by hand there are tests too, once you write them down.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_tests, result_fix, result_sort):
    L.wrap_up(result_tests, result_fix, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
