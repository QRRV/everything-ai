# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson start-how-this-course-works. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("start-how-this-course-works", "en")


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    This course teaches AI by letting you do it. You do not just read about a model: you write the code, run it, and see what happens. After every exercise, a check tells you straight away whether your answer is right, and if not, why not.

    That only works if you know your way around a lesson. So this first lesson is a short tour, and you do every step once. After that, you can keep your mind on what each lesson is about.

    **An everyday example.** The first driving lesson starts with the car standing still: where the pedals are, what the mirrors are for. Ten minutes like that save you a lot of searching once you are on the road.
    """)
    return


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Every lesson has the same five parts, always in the same order:

    | Part | What you do there |
    |---|---|
    | **The top** | see where you are in the route, the goal, and what you need |
    | **Why** | find out which problem the lesson solves |
    | **The idea** | read the new words, the explanation and a worked example |
    | **Your turn** | do exercises, and a check tells you straight away if they are right |
    | **Wrap-up** | see what you can do now, and where it comes back |

    This lesson has the same parts. Right now you are in *The idea*.

    ### New words

    | Word | What it means |
    |---|---|
    | **cell** | one block of the lesson; it holds text, like this one, or code |
    | **code** | instructions for the computer; in this course they are written in Python |
    | **run** a cell | let the computer carry out the code in a cell; the result appears right under the code |
    | **check** | a part of the lesson that looks at your answer and says whether it is right, why, and what to do now |
    | **scratchpad** | a place to try out code without changing the lesson |

    A lesson is a page made of cells, called a *notebook*. The cells that come with the lesson already ran when you opened it.
    """)
    return


@app.cell(hide_code=True)
def text_run():
    mo.vstack([
        mo.md(r"""
        ### Run a cell

        Below is a code cell. You do not need to understand the code yet: the first Python lesson explains it. For now, all that matters is that the cell shows the text between the quotes.
        """),
        L.task("Click in the code cell below and change `world` into your own name. Then run the cell: press **Shift + Enter**, or click the ▶ button to the right of the cell."),
    ])
    return


@app.cell
def example_hello():
    print("Hello, world!")
    return


@app.cell(hide_code=True)
def text_after_run():
    mo.md(r"""
    Does your name appear right under the code? Then you just ran your first cell. Everything you change is saved by itself, in your own copy of the lesson.
    """)
    return


@app.cell(hide_code=True)
def text_guess():
    mo.md(r"""
    ### Guess first

    In many lessons you guess before you see the answer. A guess is never graded. Guessing makes you notice what you expected, and that is exactly what makes the answer stick, also when your guess was wrong.
    """)
    return


@app.cell(hide_code=True)
def predict_mistake():
    guess_mistake = L.guess(
        "mistake",
        question="What do you think happens in this course when you get an exercise wrong?",
        options={
            "points": "You lose points.",
            "restart": "You have to start the lesson over.",
            "explain": "You hear what is not right yet, and you try again.",
        },
        reveal="There are no points, no penalties and no streaks. A check tells you what is not right yet and what you can do about it, and then you simply try again. Getting something wrong first is a normal part of learning.",
    )
    guess_mistake
    return (guess_mistake,)


@app.cell(hide_code=True)
def text_exercise_sum():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. An exercise with a check

        Most exercises in this course are code that you write yourself. In the first lessons you type it in a box like the one below. The check under the box reads what you typed, runs it, and tells you what it finds.
        """),
        L.task("Change the sum in the box so that the result is 10."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sum():
    box_sum = L.code_box("sum", lines=2)
    box_sum
    return (box_sum,)


@app.cell(hide_code=True)
def check_sum(box_sum):
    result_sum = L.check("sum", box_sum.value)
    result_sum
    return (result_sum,)


@app.cell(hide_code=True)
def hints_sum():
    L.hints(
        "sum",
        [
            "Look at the message from the check: what does `3 + 4` give, and how much is missing to make 10?",
            "Leave the `3` as it is and change the `4` into a bigger number.",
            "`3 + ___`: which number fills the gap?",
        ],
        why="3 + 7 is 10. Any other sum that gives 10 is just as right, like `5 + 5` or `12 - 2`: the check looks at the result.",
    )
    return


@app.cell(hide_code=True)
def text_choice():
    mo.md(r"""
    ### Exercise 2. A choice question

    Some questions have options. Every option has its own explanation, also the right one, so read it whichever you pick. You will meet this kind of question again at the top of every lesson, to check what you need from earlier lessons.
    """)
    return


@app.cell(hide_code=True)
def choice_parts_cell():
    choice_parts = L.choice(
        "parts",
        question="In which part of a lesson do you do exercises with a check?",
        options={"why": "*Why*", "idea": "*The idea*", "turn": "*Your turn*"},
        correct="turn",
        why={
            "why": "*Why* shows which problem the lesson solves. There is nothing to practice yet.",
            "idea": "*The idea* explains, with a worked example that you run and read. You practice right after it.",
            "turn": "The exercises with a check are in *Your turn*, right after the worked example, so you practice in the same form you just saw.",
        },
    )
    choice_parts
    return (choice_parts,)


@app.cell(hide_code=True)
def text_explain():
    mo.md(r"""
    ### Exercise 3. In your own words

    Some goals cannot be checked by a computer, such as explaining something. Then you write your answer first, compare it with a model answer, and tick the points that are in yours.
    """)
    return


@app.cell(hide_code=True)
def explain_check_does():
    explain_check = L.explain(
        "check-does",
        question="What does a check do when your answer is not right yet? Write one or two sentences.",
        model_answer="A check looks at the outcome of your answer. When it is not right yet, it says **what** is not right, **why**, and **what you can do now**. It does not simply give you the answer: the hints and the solution are there for when you are stuck.",
        points=[
            "it says what is not right yet",
            "it says what you can do now",
            "it does not simply give away the answer",
        ],
    )
    explain_check
    return (explain_check,)


@app.cell(hide_code=True)
def text_tools():
    mo.md(r"""
    ### Two things you will need sometimes

    **Trying something out.** Want to try a bit of code without changing the lesson? Use the **scratchpad**. Click the two small counters at the bottom left of the screen (a cross and a triangle, each with a number). A panel opens at the bottom; choose the *Scratchpad* tab. What you type there never ends up in the lesson.

    **Starting over.** Got lost, or want to do a lesson again from the start? On the start page, every lesson you have opened has **Start over**. It puts the lesson back the way the course made it. What you finished stays finished.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Every lesson in the course works like this one. The next lesson is about four words you hear everywhere: AI, machine learning, deep learning and generative AI. The first Python lesson, *Your first program*, explains the code you ran in this lesson.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_parts, explain_check, result_sum):
    L.wrap_up(result_sum, choice_parts, explain_check)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
