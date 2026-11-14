# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-make-without-code. Route: course/route/lessen/bouwen.yaml. Checks, the 16 programs and their picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-make-without-code", "en")
    C = L.checks   # the 16 programs Qwen3 wrote from a description, what they give, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_frontier():
    quick_frontier = L.choice(
        "quick-frontier",
        question="Qwen3 multiplied 23 by 47 right in all eight runs, and counted the r's in *strawberry* wrong in all eight. What does that tell you about a new task?",
        options={
            "easy": "If it looks easy, the model will do it well",
            "try": "Nothing yet: you find out by trying it on cases you can check",
            "never": "That a model cannot be trusted with any task",
        },
        correct="try",
        why={
            "try": "The frontier is jagged: two tasks that look just as easy can lie on different sides of it.",
            "easy": "Counting letters looks easy, and went wrong every time. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
            "never": "The same model multiplied right every time. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
        },
    )
    quick_frontier
    return (quick_frontier,)


@app.cell(hide_code=True)
def quick_check_centaur():
    quick_centaur = L.choice(
        "quick-centaur",
        question="You hand an assistant a whole task, and check the result once it is done. Which way of working is that?",
        options={
            "centaur": "Centaur",
            "cyborg": "Cyborg",
        },
        correct="centaur",
        why={
            "centaur": "A whole part goes to the assistant, and you check it as a whole. That is also how vibe coding works, as you will see.",
            "cyborg": "A cyborg takes turns with the assistant, step by step. Here the work goes there once and comes back once. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
        },
    )
    quick_centaur
    return (quick_centaur,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    With AI, you can make things from a description: a picture, a song, a short video, and also a small program, without writing code. That is fast, and for a first version often good enough. But what you did not make yourself, you do not know from the inside. Where a mistake is, you only find out by trying it.

    **An everyday example.** You order a birthday cake by phone: *chocolate, for twelve people*. It arrives and looks great. Whether there are nuts in it, you find out when a guest with an allergy takes a bite, unless you ask first. That is only a comparison; below you see what Qwen3 made from a description of a small program.
    """)
    return


@app.cell(hide_code=True)
def predict_programs():
    guess_programs = L.guess(
        "programs",
        question="The course asked Qwen3 sixteen times for a small program that says how many days a month has. How many of the sixteen programs got every month from 1890 to 2110 right?",
        options={
            "most": "Most of them",
            "few": "Two or three",
            "none": "None",
        },
        reveal="**None.** Every one of the sixteen programs gets some month wrong; one of them only February 1900 and February 2100, others more than a thousand months. Most of them looked fine, and all of them ran without errors. Below you see where each one goes wrong, without reading a line of code.",
    )
    guess_programs
    return (guess_programs,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **vibe coding** | having an AI make a program from a description, and judging it only by using it, without reading the code; the name comes from Andrej Karpathy (2025) |

    ### Making things from a description

    | What you make | Examples of tools (November 2026) | Where it breaks |
    |---|---|---|
    | a picture | Midjourney, the image generators in ChatGPT and Gemini | text in the picture, small details, things you did not ask for |
    | music | Suno, Udio | lyrics, and a song that sounds like an existing one |
    | a video | Veo, Sora | movements that cannot happen, things that change from one moment to the next |
    | a program | Lovable, Bolt, Replit, or any chat assistant | the cases nobody tried |

    A program, a list of instructions for a computer, is different from a picture in one way: you see a picture whole, but a program you only see when it runs, one case at a time. A wrong number in a case you never try, you never see.

    **The run.** The course gave Qwen3-0.6B two descriptions, eight times each. Description A: *Write a Python function days_in_month(month, year) that returns how many days a month has. Reply with the code only.* Description B adds: *also for February in leap years.* A leap year is a year divisible by 4, except century years, which are only leap years when divisible by 400: 2000 was a leap year, 1900 was not. The course tried every program on eight months, without changing a line of it:
    """)
    return


@app.cell(hide_code=True)
def picture_programs():
    mo.vstack([
        mo.Html(C.programs_grid(L.lang)),
        mo.md(C.grid_after(L.lang)),
    ])
    return


@app.cell(hide_code=True)
def picture_code():
    L.explore("program-code", "program", lambda _v: C.program_frame(int(_v), L.lang), start=4, low=1, high=len(C.OUTPUTS), step=1, role="",
              task="Slide through the sixteen programs. You do not need to read the code to see where each one goes wrong; if you can read it, look at how sure of itself it looks.")
    return


@app.cell(hide_code=True)
def text_moves():
    L.watch_out("A large assistant writes this program right much more often than Qwen3-0.6B. But a bigger program has more cases that nobody tries, and there even a large model makes mistakes that you only see when you test.")
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: make something without code

    1. **Describe what you want,** with the cases that matter: *also for February in leap years*.
    2. **Use it as someone who wants it to break:** try the ordinary cases, and the edges, such as the shortest month, a leap year and a century year.
    3. **Write down what you tried,** so that you can try it again after every change.
    4. **After a change, try everything again,** not only what you asked to change.
    5. **Does money, personal data or someone's health depend on it?** Then vibe coding is not enough: someone has to read and test the code.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You have an assistant make a small web page for a party: guests fill in their name and what they bring.

    1. **Describe:** *A page where guests fill in their name and what they bring, and see what the others bring. The same name twice should not count twice.*
    2. **Use it to break it:** you fill in your own name, then the same name again, then a name with a space at the end, and an empty name.
    3. **Write down:** four tries, and what the page did: the name with a space counted twice.
    4. **After the fix, all four again:** the space is fixed, but now an empty name is accepted. You ask for that too, and try all four again.
    5. **Money, personal data, health?** Names and dishes for a party: no. Vibe coding is fine here. If the page also took payments, it would not be.
    """)
    return


@app.cell(hide_code=True)
def text_vibe_or_read():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Vibe coding, or someone who reads the code?

    For which of these programs is vibe coding fine, and for which should someone read and test the code?
    """)
    return


@app.cell(hide_code=True)
def exercise_vibe_or_read():
    sort_vibe = L.sort("vibe-or-read")
    sort_vibe
    return (sort_vibe,)


@app.cell(hide_code=True)
def check_vibe_or_read(sort_vibe):
    result_vibe = L.check("vibe-or-read", sort_vibe)
    result_vibe
    return (result_vibe,)


@app.cell(hide_code=True)
def hints_vibe():
    L.hints(
        "vibe-or-read",
        [
            "Ask of each: if it goes wrong without anyone noticing, what does that cost, and for whom?",
            "Your own photos with a copy, a game, a quiz: a mistake costs you a try. Pay, bank accounts, medicine: a mistake costs someone else.",
            "Three of the six are fine to vibe code, and for three someone reads the code.",
        ],
        why="Fine to vibe code: **Holiday photos**, **A game** and **A quiz**: a mistake costs you a try. Someone reads the code: **Salaries**, **Members** and **Medicine**: money, personal data and health depend on them.",
    )
    return


@app.cell(hide_code=True)
def text_only_recent():
    mo.md(r"""
    ### Exercise 2. Tests that are too easy
    """)
    return


@app.cell(hide_code=True)
def choice_only_recent_cell():
    choice_only_recent = L.choice(
        "only-recent",
        question="Suppose you had only tried months of 2024 and 2025. Which program would have looked right to you, while it is not?",
        options={
            "a1": "Program A1",
            "a4": "Program A4",
            "b1": "Program B1",
        },
        correct="a4",
        why={
            "a4": "A4 makes every fourth year a leap year, which holds from 1901 to 2099. Only a century year such as 1900 shows that it is wrong. That is why you try the edges, not only the cases that come to mind first.",
            "a1": "Look at the column *Feb 2024* in the picture: what does A1 give, and what is right?",
            "b1": "Look at the columns *Feb 2025* and *Feb 2024*: B1 is wrong in both.",
        },
    )
    choice_only_recent
    return (choice_only_recent,)


@app.cell(hide_code=True)
def text_longer_description():
    mo.md(r"""
    ### Exercise 3. A longer description
    """)
    return


@app.cell(hide_code=True)
def choice_longer_description_cell():
    choice_longer_description = L.choice(
        "longer-description",
        question="Description B added *also for February in leap years*. What did that change?",
        options={
            "all": "Now all eight programs were right",
            "trade": "February was right far more often, but the other months went wrong in even more programs",
            "same": "Nothing: the programs stayed the same",
        },
        correct="trade",
        why={
            "trade": "February went from right in all four years in none of the eight programs to right in six; but programs with another month wrong went from five of the eight to seven. The longer description fixed what it named, and nothing else. So after every change, you try everything again.",
            "all": "Look at the rows of description B in the picture: how many have no marked number?",
            "same": "Compare the column *Feb 2024* for the programs of A and of B.",
        },
    )
    choice_longer_description
    return (choice_longer_description,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en) you found the frontier of Qwen3 by trying tasks you could check; here you did the same for a program: you judged it by what it gives, case by case. Vibe coding is centaur work, a whole task handed over and checked afterwards. Later in this module, [Checking code that AI wrote](/course/open/build-check-ai-code?lang=en) shows how to write such tries down as tests that run by themselves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_longer_description, choice_only_recent, result_vibe):
    L.wrap_up(result_vibe, choice_only_recent, choice_longer_description)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
