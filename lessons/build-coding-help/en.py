# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson build-coding-help. Route: course/route/lessen/bouwen.yaml. Checks and the picture: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("build-coding-help", "en")
    C = L.checks   # the picture of the three kinds of help, and the two sorts


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en), and one about [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_centaur():
    quick_centaur = L.choice(
        "quick-centaur",
        question="You give an assistant a whole report to translate, and check the translation once it is done. Which way of working is that?",
        options={
            "cyborg": "Cyborg",
            "frontier": "The jagged frontier",
            "centaur": "Centaur",
        },
        correct="centaur",
        why={
            "centaur": "A whole part goes to the assistant and comes back to be checked: a centaur. A cyborg takes turns with the assistant, step by step.",
            "cyborg": "A cyborg goes back and forth many times. Here the work goes there once and comes back once. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
            "frontier": "The jagged frontier is the border between the tasks a model does well and badly, not a way of working. See [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en).",
        },
    )
    quick_centaur
    return (quick_centaur,)


@app.cell(hide_code=True)
def quick_check_agents_md():
    quick_agents_md = L.choice(
        "quick-agents-md",
        question="Why is there a file AGENTS.md in the course folder?",
        options={
            "memory": "The assistant writes down in it what it has learned about you",
            "instructions": "It tells an AI assistant how the course is built and how to help you without giving answers away",
            "list": "It lists the AI agents the course uses",
        },
        correct="instructions",
        why={
            "instructions": "An assistant that works in the folder reads it, so it knows the course and how to help. It does not always follow it, so you still check what it says.",
            "memory": "The course wrote AGENTS.md, not the assistant. What is in it? See [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).",
            "list": "It is meant for any assistant you point at the course. What does it ask of it? See [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en).",
        },
    )
    quick_agents_md
    return (quick_agents_md,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Writing code is one of the things AI tools are used for most. The help comes in very different sizes: a tool that guesses the rest of the line you are typing, a chat you ask about your code, or an agent that changes ten files by itself and runs them. The more a tool does by itself, the less you see of each step. So more depends on what you check afterwards, and on what you let it touch.

    **An everyday example.** Writing a letter. Your phone suggests the next word, and you tap it or type on: you see every word. A friend reads your letter and says what could be better: you decide what to change. Or an assistant writes the letter and posts it: fast, but you only see it if you ask to read it before it goes. That is only a comparison; below you see the three kinds of help with code.
    """)
    return


@app.cell(hide_code=True)
def predict_trust():
    guess_trust = L.guess(
        "trust",
        question="In its survey of 2025, Stack Overflow asked developers about AI tools. Of the more than 33,000 who answered, 84% used them or planned to. How many trusted that what the tools give is accurate?",
        options={
            "most": "Most of them, about 70%",
            "third": "About a third",
            "few": "Hardly anyone, about 3%",
        },
        reveal="**About a third: 33%.** More of them distrusted it: 46%. Only 3% trusted it highly. The biggest frustration, named by 66%: answers that are *almost right, but not quite*. Developers use the tools a lot, and check what comes out of them. That checking is what this lesson is about.",
    )
    guess_trust
    return (guess_trust,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **code completion** | while you type, the tool suggests the rest of the line or the next few lines, often in gray; with one key, often Tab, you take the suggestion |
    | **coding agent** | an assistant that works in your project folder: it reads files, changes them and runs commands, such as the tests, and goes on by itself until it thinks the task is done |
    | **tests** | small programs that run your code on cases and say whether it gives what it should; a later lesson shows how to write them |
    | **permissions** | what you allow an agent to do without asking you first |

    ### Three kinds of help

    | Kind | What it sees | What it does | What you check |
    |---|---|---|---|
    | **code completion** | the file you are in, around where you type | suggests the next line or lines | every suggestion, before you take it |
    | **chat** about code | what you paste in or select, and your question | explains, finds a mistake, writes a piece of code | its answer, before you use it |
    | **coding agent** | the whole project folder | reads, changes and runs, in many steps | what it changed, and whether the tests really pass |

    The model behind all three is a language model. Many tools do all three: editors such as Cursor, or Visual Studio Code with GitHub Copilot, complete, chat and have an agent; Claude Code, Codex and Aider are agents you start in a terminal, a window where you type commands, in the project folder (examples in November 2026).
    """)
    return


@app.cell(hide_code=True)
def picture_ways():
    mo.Html(C.ways_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_setup():
    mo.vstack([
        mo.md(r"""
        ### Setting a tool up

        An agent works on your computer, with your files. Three things decide what it can do:

        - **Where you start it.** It sees the folder you start it in, and everything under it. Start it in the project folder, not in your home folder.
        - **What it may do without asking.** Most agents ask before each command or change, and let you allow some things for good: the permissions. Allow what is easy to undo and changes nothing outside the project, such as reading the project and running its tests. Let it ask first for what is hard to undo, or comes in from the internet or goes out to it.
        - **What it knows of your project.** A file such as AGENTS.md, which you know from [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en), tells it how the project is built and what you want. The next lesson is about that context.

        Keep passwords and keys out of the folder it works in: what an agent reads goes to the model, and with a model online, to its provider, as with everything you paste into a chat ([Which AI tool for what](/course/open/start-ai-tools-landscape?lang=en)). Some agents can also run in a *sandbox*: a closed-off space where a mistake cannot reach the rest of your computer.
        """),
        L.watch_out("An agent that writes *done, all tests pass* has not always run them, and a test it wrote itself can test the wrong thing. What it says about its own work is text from a language model, like any other answer: you look at the changes and the test results yourself."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe: choose and set up help with code

    1. **How much do you want to see?** Every line: code completion. Every answer: a chat. Only the result: an agent, for work bigger than a few lines.
    2. **Start an agent in the project folder,** with a copy of the folder you can go back to.
    3. **Allow what is easy to undo,** such as reading the project and running the tests. Let it ask first before it deletes, installs or sends something.
    4. **Keep passwords and keys out** of the folder.
    5. **Read what it changed, and run the tests yourself,** before you keep it.

    In this course, the exercises are for you to write: help that writes the answer takes the learning away, as AGENTS.md says. There, a chat that explains is the help that fits.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Your small program of four files reads a list of books and prints the ones you have not read yet. You want an option that sorts them by year, with a test.

    1. **How much to see:** more than a few lines, in more than one file, with tests to run: an agent fits.
    2. **Folder:** you copy the folder *books* to *books-copy*, and start the agent in *books*.
    3. **Permissions:** it may read the four files and run the tests by itself; it asks before it installs anything.
    4. **Secrets:** the folder has none; the program only reads *books.csv*.
    5. **Check:** the agent writes that it changed two files, added a test, and that all tests pass. You read the two changes and run the tests yourself: they pass. But it also gave a function a new name, which you did not ask for. You ask it to undo that, and look again.
    """)
    return


@app.cell(hide_code=True)
def text_which_help():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which kind of help?

    Put each task with the kind of help that fits it best.
    """)
    return


@app.cell(hide_code=True)
def exercise_which_help():
    sort_help = L.sort("which-help")
    sort_help
    return (sort_help,)


@app.cell(hide_code=True)
def check_which_help(sort_help):
    result_help = L.check("which-help", sort_help)
    result_help
    return (result_help,)


@app.cell(hide_code=True)
def hints_help():
    L.hints(
        "which-help",
        [
            "Look at the last column of the table: what you check. How much of the work do you want to see?",
            "A line or two where you type: completion. A question whose answer you use yourself: a chat. Many files and tests to run: an agent.",
            "Two of the six are code completion, two are a chat, and two are an agent.",
        ],
        why="Code completion: **The next line** and **A last line**. Chat: **An error message** and **Two ways**. Coding agent: **Twelve files** and **A new option**.",
    )
    return


@app.cell(hide_code=True)
def text_permissions():
    mo.md(r"""
    ### Exercise 2. What may an agent do by itself?

    You set up a coding agent for a project of your own. Which of these may it do by itself, and for which does it ask you first?
    """)
    return


@app.cell(hide_code=True)
def exercise_permissions():
    sort_permissions = L.sort("permissions")
    sort_permissions
    return (sort_permissions,)


@app.cell(hide_code=True)
def check_permissions(sort_permissions):
    result_permissions = L.check("permissions", sort_permissions)
    result_permissions
    return (result_permissions,)


@app.cell(hide_code=True)
def hints_permissions():
    L.hints(
        "permissions",
        [
            "Ask of each: can I undo it, and does anything come in from the internet or go out to it?",
            "Reading and looking change nothing. Deleting, installing and putting online do.",
            "Three of the six it does by itself, and for three it asks you first.",
        ],
        why="By itself: **Read the project**, **Run the tests** and **Search the code**: nothing changes that you cannot undo. Asks you first: **Install a package** (code from strangers that runs on your computer), **Delete a folder** (it may be gone for good) and **Put it online** (what is online stays online).",
    )
    return


@app.cell(hide_code=True)
def text_after_done():
    mo.md(r"""
    ### Exercise 3. The agent says it is done
    """)
    return


@app.cell(hide_code=True)
def choice_after_done_cell():
    choice_after_done = L.choice(
        "after-done",
        question="The agent writes: *Done. I changed three files, and all tests pass.* What do you do before you keep the changes?",
        options={
            "ask": "Ask the agent whether it is sure",
            "look": "Read the three changes and run the tests yourself",
            "keep": "Nothing: the tests pass, so it works",
        },
        correct="look",
        why={
            "look": "What the agent says about its work is text from a language model, which can be wrong like any other answer. The changes and the test results you see yourself are not.",
            "keep": "Did you see the tests pass? An agent can say they pass without running them, or change a test until it passes. What can you look at yourself?",
            "ask": "Its answer to that is again text from the same model, which easily says yes. What can you look at yourself?",
        },
    )
    choice_after_done
    return (choice_after_done,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [AI at work: centaur or cyborg](/course/open/build-ai-at-work?lang=en) you saw two ways of working with AI: code completion is cyborg work, line by line; an agent you hand a whole task is centaur work, and you check the result. In [Learning with your own AI assistant](/course/open/start-your-own-assistant?lang=en) you pointed an assistant at the course with AGENTS.md. The next lesson, [Working with a coding agent: plan, spec and context](/course/open/build-coding-workflow?lang=en), shows how to give an agent a task so that it does what you meant.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_after_done, result_help, result_permissions):
    L.wrap_up(result_help, result_permissions, choice_after_done)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
